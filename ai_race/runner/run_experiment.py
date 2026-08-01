"""Build and execute config-driven AI Race experiments."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import random
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_race.dataio.config_loader import (
    load_json,
    personas_sha256,
    validate_agents,
    validate_experiment,
    validate_game,
)
from ai_race.dataio.recorder import RunJournal, write_all_results_csv
from ai_race.engine.agent import RaceAgent
from ai_race.engine.game import AIRaceGame
from ai_race.engine.state import GameConfig
from ai_race.models.factory import get_send_batch, init_offline_backend
from ai_race.paths import CONFIGS_DIR, PROMPTS_DIR, REPO_ROOT, RESULTS_DIR
from ai_race.prompts.sensitivity import apply_prompt_variant, get_prompt_variant
from ai_race.runner.batch import run_games_batched

# Manifests written by this runner for a real backend (not --mock) use this schema
# so results/scripts/analyze_ai_race.py can verify a protocol_signature across runs
# (needed to estimate a persona effect via --fit-logit instead of only describing
# it). The previous schema, "ai-race-results-v1", omitted source/decoding/seed
# provenance, so every local run fell back to an "unverified" signature keyed on
# its output path -- persona was then perfectly confounded with the run batch even
# when every other setting was identical. See docs/running-proxy-pilots.md.
MANIFEST_SCHEMA_PROXY_RUN = "ai-race-proxy-run-v1"
MANIFEST_SCHEMA_LEAN = "ai-race-results-v1"


def model_slug(name: str) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "-", str(name)).strip("-").lower()
    return slug or "model"


def _load_agents(exp: dict[str, Any], agents_name: Optional[str]) -> tuple[str, dict]:
    name = agents_name or str(exp.get("agents", "companies_default"))
    path = CONFIGS_DIR / "agents" / f"{name}.json"
    return name, load_json(path)


def _agents_for_language(agents_cfg: dict, language: str) -> list[RaceAgent]:
    validate_agents(agents_cfg)
    names = list(agents_cfg.get("names", []))
    personas_by_language = agents_cfg.get("personas", {}) or {}
    personas = list(personas_by_language.get(language, ["", ""]))
    if len(personas) != 2:
        raise ValueError(f"Agent configuration must define two {language!r} personas")
    probabilities_by_language = agents_cfg.get("personaProbabilities", {}) or {}
    probabilities = list(probabilities_by_language.get(language, [100.0, 100.0]))
    if len(probabilities) != 2:
        raise ValueError(f"Agent configuration must define two {language!r} personaProbabilities")
    condition = str(agents_cfg.get("personaCondition", "none")).strip()
    if condition != "none" and not all(str(text).strip() for text in personas):
        # Otherwise a run labelled with a persona condition would render neutral
        # prompts, and its rows would claim a manipulation that never happened.
        raise ValueError(
            f"personaCondition={condition!r} has no {language!r} persona text; "
            "translate the personas before running that language"
        )
    roles = list(agents_cfg.get("personaRoles", ["", ""]))
    return [
        RaceAgent(
            name=str(name),
            persona_text=str(persona),
            persona_probability=float(probability),
            persona_role=str(role),
        )
        for name, persona, probability, role in zip(names, personas, probabilities, roles)
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

    validate_agents(agents_cfg)
    persona_condition = str(agents_cfg.get("personaCondition", "none")).strip()
    persona_hash = personas_sha256(agents_cfg.get("personas", {}) or {})

    repetitions = int(exp["repetitions"])
    base_seed = int(exp["seed"])
    languages = list(exp.get("languages", ["en"]))
    games: list[AIRaceGame] = []
    prompt_variant_id = str(exp.get("promptVariant", "canonical"))
    prompt_variant = get_prompt_variant(prompt_variant_id)

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
                persona_condition=persona_condition,
                persona_sha256=persona_hash,
                prompt_version=(
                    prompt_variant.version
                    if prompt_variant_id != "canonical"
                    else str(game_data.get("promptVersion", "ai-race-fairgame-v3"))
                ),
            )
            template_name = config.prompt_template.format(language=language)
            template_path = PROMPTS_DIR / f"{template_name}.txt"
            template = apply_prompt_variant(
                template_path.read_text(encoding="utf-8"), prompt_variant_id
            )
            for rep in range(repetitions):
                # Deliberately independent of the treatment name: the same repetition
                # shares horizon and fixed-seat setback draws across risk conditions.
                game_seed = base_seed + rep
                variant_component = (
                    "" if prompt_variant_id == "canonical"
                    else f"__prompt-{prompt_variant_id}"
                )
                game_id = (
                    f"{config.name}__{model_slug(model)}__{language}__"
                    f"{resolved_agents_name}{variant_component}__rep{rep:04d}"
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


def _agents_provenance(exp: dict[str, Any]) -> dict[str, Any]:
    """Describe the seat/persona configuration a run actually used."""

    name = str(exp.get("agents", "companies_default"))
    path = CONFIGS_DIR / "agents" / f"{name}.json"
    agents_cfg = validate_agents(load_json(path))
    return {
        "agents_name": name,
        "agents_config_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "persona_condition": str(agents_cfg.get("personaCondition", "none")).strip(),
        "persona_roles": [str(role) for role in agents_cfg.get("personaRoles", ["", ""])],
        "persona_sha256": personas_sha256(agents_cfg.get("personas", {}) or {}),
    }


def _source_tree_sha256() -> str:
    """Hash every tracked-format source file under ai_race/ and FAIRGAME/src/.

    Mirrors kaggle/experiments/baseline.py's source_tree_sha256() so a run
    executed locally and one executed on Kaggle can be told apart (or matched)
    on the same basis: two runs sharing this hash used byte-identical engine,
    config, and prompt code, not merely "the same git commit at some point".
    """
    digest = hashlib.sha256()
    roots = [REPO_ROOT / "ai_race", REPO_ROOT / "FAIRGAME" / "src"]
    files = sorted(
        path
        for root in roots
        if root.is_dir()
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".txt"}
    )
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _prompt_provenance(exp: dict[str, Any]) -> dict[str, str]:
    """Hash the exact rendered template text this experiment's first game used.

    Every game in one experiment shares a prompt template/version in this
    project (only maxPrivateRisk varies by game), so the first game stands in
    for all of them; a differing template elsewhere would already fail the
    "same games" comparability the analyser expects.
    """
    game_name = str(exp["games"][0])
    game_data = validate_game(load_json(CONFIGS_DIR / "game" / f"{game_name}.json"))
    language = str(list(exp.get("languages", ["en"]))[0])
    prompt_variant_id = str(exp.get("promptVariant", "canonical"))
    prompt_variant = get_prompt_variant(prompt_variant_id)
    version = (
        prompt_variant.version
        if prompt_variant_id != "canonical"
        else str(game_data.get("promptVersion", "ai-race-fairgame-v3"))
    )
    template_name = str(game_data.get("promptTemplate", "ai_race_{language}")).format(
        language=language
    )
    raw_template = (PROMPTS_DIR / f"{template_name}.txt").read_text(encoding="utf-8")
    rendered = apply_prompt_variant(raw_template, prompt_variant_id)
    return {
        "version": version,
        "sha256": hashlib.sha256(rendered.encode("utf-8")).hexdigest(),
    }


def _mechanism_provenance(exp: dict[str, Any]) -> dict[str, Any]:
    """Collect the paper-mechanism constants and the set of risk treatments run.

    Fails loudly if two games in the same experiment disagree on anything other
    than maxPrivateRisk -- that would mean the runs are not the "same mechanism,
    different treatment" design the analyser assumes.
    """
    games = [
        validate_game(load_json(CONFIGS_DIR / "game" / f"{name}.json"))
        for name in exp["games"]
    ]
    first = games[0]
    fixed_fields = (
        "minRounds",
        "stopProbability",
        "racePrize",
        "safeProgress",
        "unsafeProgress",
        "stagePayoffs",
    )
    for other in games[1:]:
        mismatched = [field for field in fixed_fields if other.get(field) != first.get(field)]
        if mismatched:
            raise ValueError(
                f"Games {first['name']!r} and {other['name']!r} disagree on "
                f"{mismatched}; manifest mechanism provenance assumes only "
                "maxPrivateRisk varies across an experiment's games"
            )
    stage_payoffs = dict(first["stagePayoffs"])
    return {
        "minimum_rounds": int(first["minRounds"]),
        "stop_probability": float(first["stopProbability"]),
        "risk_levels": sorted({float(game["maxPrivateRisk"]) for game in games}),
        "race_prize": float(first["racePrize"]),
        "stage_payoff": {
            "safe": {
                "safe": float(stage_payoffs["safeSafe"]),
                "unsafe": float(stage_payoffs["safeUnsafe"]),
            },
            "unsafe": {
                "safe": float(stage_payoffs["unsafeSafe"]),
                "unsafe": float(stage_payoffs["unsafeUnsafe"]),
            },
        },
        "progress": {
            "safe": float(first["safeProgress"]),
            "unsafe": float(first["unsafeProgress"]),
        },
    }


def _proxy_decoding_and_seed_provenance(exp: dict[str, Any]) -> dict[str, dict[str, Any]]:
    """Describe the proxy backend's actual request contract, not an assumed one.

    Mirrors the honesty rules in kaggle/benchmarks/ai_race_baseline.py: a value
    is only ``_effective``/``_applied`` when this process can confirm the
    provider actually used it, not merely that the SDK accepted the parameter.
    """
    from ai_race.models.proxy import (
        DEFAULT_MAX_TOKENS,
        DEFAULT_MAX_TRANSPORT_RETRIES,
        DEFAULT_TEMPERATURE,
    )

    proxy_options = dict(exp.get("proxyOptions", {}) or {})
    send_seed = bool(proxy_options.get("send_seed", False))
    if send_seed:
        seed_contract = {
            "requested": True,
            "forwarded_by_sdk": True,
            "applied": None,
            "applied_known": False,
            "status": "forwarded_to_provider_application_unconfirmed",
            "strip_detection": "not_applicable_proxy_backend",
        }
    else:
        seed_contract = {
            "requested": False,
            "forwarded_by_sdk": False,
            "applied": False,
            "applied_known": True,
            "status": "not_requested_send_seed_disabled_in_config",
            "strip_detection": "not_applicable_seed_not_requested",
        }
    decoding = {
        "temperature_requested": float(proxy_options.get("temperature", DEFAULT_TEMPERATURE)),
        "temperature_forwarded_by_sdk": True,
        "temperature_effective": None,
        "temperature_effective_confirmed": False,
        "temperature_status": "forwarded_to_provider_effective_value_unconfirmed",
        "output_token_limit_parameter": "max_tokens",
        "output_token_limit": int(proxy_options.get("max_tokens", DEFAULT_MAX_TOKENS)),
        "max_parse_retries": int(exp.get("maxParseRetries", 3)),
        "max_transport_retries": int(
            proxy_options.get("max_transport_retries", DEFAULT_MAX_TRANSPORT_RETRIES)
        ),
    }
    return {"decoding": decoding, "sampling_seed_provenance": seed_contract}


def _unverified_decoding_and_seed_provenance(backend: str) -> dict[str, dict[str, Any]]:
    """Fallback for backends whose request contract this runner doesn't model yet.

    Fills every key the analyser requires so the manifest is structurally valid,
    but every value that would claim knowledge of the provider's actual
    behaviour is left explicitly unconfirmed -- this must never be read as
    "equivalent to the proxy path", only as "not yet audited".
    """
    status = f"backend_{backend}_not_yet_documented_for_{MANIFEST_SCHEMA_PROXY_RUN}"
    return {
        "decoding": {
            "temperature_requested": None,
            "temperature_forwarded_by_sdk": None,
            "temperature_effective": None,
            "temperature_effective_confirmed": False,
            "temperature_status": status,
            "output_token_limit_parameter": "unknown",
            "output_token_limit": 0,
            "max_parse_retries": 0,
            "max_transport_retries": 0,
        },
        "sampling_seed_provenance": {
            "requested": None,
            "forwarded_by_sdk": None,
            "applied": None,
            "applied_known": False,
            "status": status,
            "strip_detection": "not_audited",
        },
    }


def _package_versions() -> dict[str, Optional[str]]:
    versions: dict[str, Optional[str]] = {}
    for package in ("numpy", "pandas", "openai", "torch", "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _write_manifest(
    path: Path,
    *,
    experiment: dict[str, Any],
    model: str,
    n_races: int,
    n_turns: int,
    status: str,
    error: Optional[str] = None,
    mock_strategy: Optional[str] = None,
) -> None:
    backend = "offline" if bool(experiment.get("useOffline", True)) else str(
        experiment.get("backend", "api")
    )
    manifest: dict[str, Any] = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "model": model,
        "run_phase": str(experiment.get("runPhase", "pilot")),
        **_agents_provenance(experiment),
        "status": status,
        "n_races": n_races,
        "n_turns": n_turns,
    }
    if mock_strategy:
        manifest["schema_version"] = MANIFEST_SCHEMA_LEAN
    else:
        prompt = _prompt_provenance(experiment)
        decoding_and_seed = (
            _proxy_decoding_and_seed_provenance(experiment)
            if backend == "proxy"
            else _unverified_decoding_and_seed_provenance(backend)
        )
        manifest.update(
            {
                "schema_version": MANIFEST_SCHEMA_PROXY_RUN,
                "source_sha256": _source_tree_sha256(),
                "prompt_version": prompt["version"],
                "prompt_sha256": prompt["sha256"],
                "model_route": model,
                "llm_backend_mro": [backend],
                "mechanism": _mechanism_provenance(experiment),
                "package_versions": _package_versions(),
                **decoding_and_seed,
            }
        )
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
            mock_strategy=mock_strategy,
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
                mock_strategy=mock_strategy,
            )
            raise
        _write_manifest(
            model_dir / "run_manifest.json",
            experiment=exp,
            model=str(model),
            n_races=journal.race_count,
            n_turns=journal.turn_count,
            status="completed",
            mock_strategy=mock_strategy,
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
