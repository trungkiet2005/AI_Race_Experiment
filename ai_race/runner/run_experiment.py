"""Build and execute config-driven AI Race experiments."""
from __future__ import annotations

import argparse
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_race.dataio.config_loader import load_json, validate_experiment, validate_game
from ai_race.dataio.recorder import RunJournal, write_all_results_csv
from ai_race.engine.agent import RaceAgent
from ai_race.engine.game import AIRaceGame
from ai_race.engine.state import GameConfig
from ai_race.models.factory import get_send_batch, init_offline_backend
from ai_race.paths import CONFIGS_DIR, PROMPTS_DIR, RESULTS_DIR
from ai_race.runner.batch import run_games_batched


def model_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-").lower()
    return slug or "model"


def _load_agents(exp: dict[str, Any], agents_name: Optional[str]) -> tuple[str, dict]:
    name = agents_name or str(exp.get("agents", "companies_default"))
    path = CONFIGS_DIR / "agents" / f"{name}.json"
    return name, load_json(path)


def _agents_for_language(agents_cfg: dict, language: str) -> list[RaceAgent]:
    names = list(agents_cfg.get("names", []))
    if len(names) != 2:
        raise ValueError("Agent configuration must define exactly two names")
    personas_by_language = agents_cfg.get("personas", {}) or {}
    personas = list(personas_by_language.get(language, ["", ""]))
    if len(personas) != 2:
        raise ValueError(f"Agent configuration must define two {language!r} personas")
    return [
        RaceAgent(name=str(name), persona_text=str(persona))
        for name, persona in zip(names, personas)
    ]


def build_games_for_model(
    exp: dict[str, Any],
    model: str,
    agents_cfg: Optional[dict] = None,
    agents_name: Optional[str] = None,
) -> list[AIRaceGame]:
    """Construct all treatment × language × repetition races for one model."""
    validate_experiment(exp)
    if agents_cfg is None:
        resolved_agents_name, agents_cfg = _load_agents(exp, agents_name)
    else:
        resolved_agents_name = agents_name or str(
            agents_cfg.get("name", exp.get("agents", "companies_default"))
        )

    repetitions = int(exp["repetitions"])
    base_seed = int(exp["seed"])
    languages = list(exp.get("languages", ["en"]))
    games: list[AIRaceGame] = []

    for game_name in exp["games"]:
        game_path = CONFIGS_DIR / "game" / f"{game_name}.json"
        game_data = validate_game(load_json(game_path))
        for language in languages:
            config = GameConfig.from_dict(
                game_data,
                language=str(language),
                model=str(model),
                # Hosted routes may forward a seed without confirming it was applied,
                # so the claim is an explicit configuration choice, not a default.
                sampling_seed_applied=bool(
                    exp.get("samplingSeedApplied", exp.get("useOffline", True))
                ),
                run_phase=str(exp.get("runPhase", "pilot")),
            )
            template_name = config.prompt_template.format(language=language)
            template_path = PROMPTS_DIR / f"{template_name}.txt"
            template = template_path.read_text(encoding="utf-8")
            for rep in range(repetitions):
                # Deliberately independent of the treatment name: the same repetition
                # shares horizon and fixed-seat setback draws across risk conditions.
                game_seed = base_seed + rep
                game_id = (
                    f"{config.name}__{model_slug(model)}__{language}__"
                    f"{resolved_agents_name}__rep{rep:04d}"
                )
                games.append(
                    AIRaceGame(
                        config,
                        _agents_for_language(agents_cfg, str(language)),
                        template=template,
                        game_id=game_id,
                        seed=game_seed,
                        rep=rep,
                    )
                )
    return games


def make_mock_send_batch(strategy: str) -> Any:
    """A deterministic diagnostic backend retained for CI/Kaggle smoke checks."""
    normalized = strategy.strip().lower()
    if normalized not in {"safe", "unsafe", "random"}:
        raise ValueError("Mock strategy must be safe, unsafe, or random")

    def send(prompts: list[str], seeds: Optional[list[int]] = None) -> list[str]:
        seeds = list(seeds or range(len(prompts)))
        outputs: list[str] = []
        for seed in seeds:
            if normalized == "random":
                action = random.Random(int(seed)).choice(["SAFE", "UNSAFE"])
            else:
                action = normalized.upper()
            outputs.append(f"ACTION: {action}")
        return outputs

    return send


def _write_manifest(
    path: Path,
    *,
    experiment: dict[str, Any],
    model: str,
    n_races: int,
    n_turns: int,
    status: str,
    error: Optional[str] = None,
) -> None:
    manifest = {
        "schema_version": "ai-race-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "model": model,
        "run_phase": str(experiment.get("runPhase", "pilot")),
        "status": status,
        "n_races": n_races,
        "n_turns": n_turns,
    }
    if error:
        manifest["error"] = str(error)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment(
    exp: dict[str, Any],
    *,
    output_root: Path,
    mock_strategy: Optional[str] = None,
) -> list[Any]:
    """Run all configured models and persist a separate result directory per model."""
    validate_experiment(exp)
    all_results: list[Any] = []
    for model_index, model in enumerate(exp["models"]):
        model_dir = output_root / model_slug(str(model))
        journal = RunJournal(model_dir, reset=True)
        _write_manifest(
            model_dir / "run_manifest.json",
            experiment=exp,
            model=str(model),
            n_races=0,
            n_turns=0,
            status="running",
        )
        try:
            games = build_games_for_model(exp, str(model))
            if mock_strategy:
                send_batch = make_mock_send_batch(mock_strategy)
                max_retries = 0
            elif bool(exp.get("useOffline", True)):
                settings_name = str(exp.get("offlineSettings", "offline_settings"))
                settings = load_json(
                    CONFIGS_DIR / "offline" / f"{settings_name}.json"
                )
                preset_name = str(
                    (exp.get("modelPresets", {}) or {}).get(
                        model,
                        settings["modelPreset"],
                    )
                )
                preset = load_json(CONFIGS_DIR / "offline" / f"{preset_name}.json")
                init_offline_backend(settings, preset, force=model_index > 0)
                send_batch = get_send_batch(
                    str(model),
                    offline=True,
                    batch_size=int(settings.get("batchSize", 0)),
                )
                max_retries = int(
                    exp.get(
                        "maxParseRetries",
                        settings.get("maxParseRetries", 3),
                    )
                )
            else:
                send_batch = get_send_batch(
                    str(model),
                    offline=False,
                    backend=str(exp.get("backend", "api")),
                    proxy_options=dict(exp.get("proxyOptions", {}) or {}),
                )
                max_retries = int(exp.get("maxParseRetries", 3))

            results = run_games_batched(
                games,
                send_batch,
                verbose=bool(exp.get("verbose", True)),
                max_parse_retries=max_retries,
                on_round_complete=journal.record_round,
            )
        except Exception as exc:
            _write_manifest(
                model_dir / "run_manifest.json",
                experiment=exp,
                model=str(model),
                n_races=journal.race_count,
                n_turns=journal.turn_count,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _write_manifest(
            model_dir / "run_manifest.json",
            experiment=exp,
            model=str(model),
            n_races=journal.race_count,
            n_turns=journal.turn_count,
            status="completed",
        )
        all_results.extend(results)
    write_all_results_csv(all_results, output_root / "all_results.csv")
    return all_results


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "experiment",
        nargs="?",
        default=str(CONFIGS_DIR / "experiment" / "baseline.json"),
    )
    parser.add_argument("--output", type=Path)
    parser.add_argument("--mock", choices=["safe", "unsafe", "random"])
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    exp = validate_experiment(load_json(args.experiment))
    output = args.output or RESULTS_DIR / exp["name"]
    results = run_experiment(exp, output_root=Path(output), mock_strategy=args.mock)
    print(f"Saved {len(results)} AI races to {output}")


if __name__ == "__main__":
    main()
