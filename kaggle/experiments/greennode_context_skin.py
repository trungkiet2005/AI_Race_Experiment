"""Run paired, mechanism-invariant narrative-context shards on Ollama GPUs.

The workload is self-contained at the orchestration layer: every skin starts
from the same experiment JSON and game JSONs, then replaces only the rendered
template and prompt-version label. Repetition ``r`` therefore has the same game
seed, horizon stream, sampling seed, and fixed-seat setback draws in every skin.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import re
import time
from pathlib import Path
from typing import Any

from ai_race.dataio.config_loader import load_json
from ai_race.dataio.recorder import RunJournal
from ai_race.engine.state import GameConfig
from ai_race.engine.game import AIRaceGame
from ai_race.prompts.context_skins import (
    ACTION_CODE_MAPPINGS,
    ACTION_CODE_PROTOCOL,
    CONTEXT_SKIN_PROTOCOL,
    SKINS,
    action_code_mapping_for_rep,
    context_skin_sha256,
    get_context_skin,
    render_context_skin,
)
from ai_race.runner.batch import run_games_batched
from ai_race.runner.run_experiment import build_games_for_model
from kaggle.experiments.greennode_prompt_sensitivity import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    OllamaBatchBackend,
    agents_provenance,
    atomic_json,
    gpu_name,
    model_provenance,
    request_json,
    sha256_file,
    source_tree_sha256,
    utc_now,
)


LANE_SKINS = {
    "a": [
        "technology_race",
        "logistics_contract",
        "hospital_deployment",
        "robotic_expedition",
    ],
    "b": [
        "abstract_contest",
        "crystal_guild_contract",
        "colony_life_support",
        "fictional_cartography",
    ],
}
# Thirty-two paired race seeds per risk/skin keeps the public pilot exploratory
# while giving materially better coverage than the two-seed execution smoke.
PROFILE_REPETITIONS = {"smoke": 2, "pilot": 32}

_MECHANISM_FIELDS = (
    "engine",
    "n_players",
    "safe_progress",
    "unsafe_progress",
    "payoff_safe_safe",
    "payoff_safe_unsafe",
    "payoff_unsafe_safe",
    "payoff_unsafe_unsafe",
    "min_rounds",
    "stop_probability",
    "max_rounds_safety_cap",
    "race_prize",
    "max_private_risk",
    "history_mode",
    "agents_communicate",
    "reveal_opponent_persona_prior",
)


def mechanism_signature(config: GameConfig) -> dict[str, Any]:
    """Return every field that can affect state transitions or payoffs."""
    return {name: getattr(config, name) for name in _MECHANISM_FIELDS}


def mechanism_sha256(config: GameConfig) -> str:
    payload = json.dumps(
        mechanism_signature(config), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


class OpaqueContextGame(AIRaceGame):
    """AI Race game whose displayed history uses its balanced P/Q mapping."""

    def __init__(self, *args: Any, action_code_mapping: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.action_code_mapping = action_code_mapping

    def build_round_prompts(self) -> list[str]:
        prompts = super().build_round_prompts()
        mapping = self.action_code_mapping
        translated: list[str] = []
        for prompt in prompts:
            # These labels originate only in the engine-rendered history. Replace
            # Unsafe first so its Safe suffix can never be rewritten twice.
            prompt = re.sub(
                r"\bUNSAFE\b", mapping.unsafe_code, prompt, flags=re.IGNORECASE
            )
            prompt = re.sub(
                r"\bSAFE\b", mapping.safe_code, prompt, flags=re.IGNORECASE
            )
            translated.append(prompt)
        return translated


def build_context_games(
    experiment: dict[str, Any], model: str, skin_id: str
) -> list[Any]:
    """Build one skin without mutating the caller's frozen experiment object."""
    skin = get_context_skin(skin_id)
    effective = copy.deepcopy(experiment)
    effective["agents"] = "context_skin_neutral"
    effective.pop("promptVariant", None)
    games = build_games_for_model(effective, model)
    opaque_games: list[OpaqueContextGame] = []
    for game in games:
        mapping = action_code_mapping_for_rep(game.rep)
        # ``build_games_for_model`` intentionally shares one immutable config
        # across repetitions. Here prompt metadata differs by balanced mapping,
        # so detach it before labeling the exact prompt contract.
        config = copy.deepcopy(game.config)
        config.prompt_version = (
            f"{skin.version}:{ACTION_CODE_PROTOCOL}:{mapping.id}"
        )
        game_id = (
            f"{game.game_id}__context-{skin_id}__action-map-{mapping.id}"
        )
        opaque_games.append(
            OpaqueContextGame(
                config,
                game.agents,
                template=render_context_skin(skin_id, mapping.id),
                game_id=game_id,
                seed=game.seed,
                rep=game.rep,
                action_code_mapping=mapping,
            )
        )
    return opaque_games


def build_fully_crossed_context_games(
    experiment: dict[str, Any], model: str, skin_id: str
) -> list[OpaqueContextGame]:
    """Cross both opaque mappings within every risk/context/seed block.

    The original exploratory pilot balanced mappings by repetition parity.  That
    design protected the context contrast on average, but it could not estimate
    a mapping main effect or a context-by-mapping interaction without relying on
    different seeds.  This builder duplicates each base race under both mappings
    while preserving its game seed and sampling-seed streams.  The mapping id is
    added only to prompt metadata and the game id, never to the mechanism.
    """
    skin = get_context_skin(skin_id)
    effective = copy.deepcopy(experiment)
    effective["agents"] = "context_skin_neutral"
    effective.pop("promptVariant", None)
    base_games = build_games_for_model(effective, model)
    opaque_games: list[OpaqueContextGame] = []
    for game in base_games:
        for mapping_id, mapping in ACTION_CODE_MAPPINGS.items():
            config = copy.deepcopy(game.config)
            config.prompt_version = (
                f"{skin.version}:{ACTION_CODE_PROTOCOL}:{mapping_id}"
            )
            game_id = (
                f"{game.game_id}__context-{skin_id}__action-map-{mapping_id}"
            )
            opaque_games.append(
                OpaqueContextGame(
                    config,
                    game.agents,
                    template=render_context_skin(skin_id, mapping_id),
                    game_id=game_id,
                    seed=game.seed,
                    rep=game.rep,
                    action_code_mapping=mapping,
                )
            )
    return opaque_games


_OPAQUE_ACTION_RE = re.compile(
    r"^\s*ACTION\s*:\s*(P|Q)\s*$", flags=re.IGNORECASE
)
_SAFE_CODE_IN_PROMPT_RE = re.compile(
    r"\b(P|Q) advances your .*? and carries no additional risk\.",
    flags=re.IGNORECASE,
)


def mapping_from_prompt(prompt: str):
    """Recover the declared mapping from the prompt, never from row order."""
    match = _SAFE_CODE_IN_PROMPT_RE.search(prompt)
    if match is None:
        raise RuntimeError("Cannot identify the opaque Safe code from prompt")
    safe_code = match.group(1).upper()
    mapping_id = "safe_p" if safe_code == "P" else "safe_q"
    return ACTION_CODE_MAPPINGS[mapping_id]


class OpaqueActionBackend:
    """Translate strict P/Q outputs while preserving the emitted raw text."""

    def __init__(self, backend: OllamaBatchBackend) -> None:
        self.backend = backend

    def __call__(
        self, prompts: list[str], seeds: list[int] | None = None
    ) -> list[dict[str, Any]]:
        raw_responses = self.backend(prompts, seeds=seeds)
        adapted: list[dict[str, Any]] = []
        for prompt, raw in zip(prompts, raw_responses):
            raw_text = str(raw)
            match = _OPAQUE_ACTION_RE.fullmatch(raw_text)
            if match is None:
                # The canonical parser will fail closed and trigger an unchanged-
                # prompt retry. Never rescue an action embedded in prose.
                adapted.append({"text": raw_text, "raw_response": raw_text})
                continue
            mapping = mapping_from_prompt(prompt)
            opaque_code = match.group(1).upper()
            adapted.append(
                {
                    "text": f"ACTION: {mapping.decode(opaque_code)}",
                    "raw_response": raw_text,
                    "opaque_action_code": opaque_code,
                    "action_code_mapping": mapping.id,
                }
            )
        return adapted


def crn_contract(games: list[Any]) -> list[dict[str, Any]]:
    """Machine-readable paired keys, independent of context and lane."""
    rows = [
        {
            "game_name": game.config.name,
            "max_private_risk": game.config.max_private_risk,
            "rep": game.rep,
            "game_seed": game.seed,
            "seat0_round1_sampling_seed": game.sampling_seed(0, 1),
            "seat1_round1_sampling_seed": game.sampling_seed(1, 1),
            "mechanism_sha256": mechanism_sha256(game.config),
        }
        for game in games
    ]
    if len({(row["game_name"], row["rep"]) for row in rows}) != len(rows):
        raise RuntimeError("Duplicate context-skin CRN pairing keys")
    return rows


def run_skin_shard(
    *,
    root: Path,
    output_root: Path,
    experiment: dict[str, Any],
    experiment_path: Path,
    skin_id: str,
    model: str,
    backend: OllamaBatchBackend,
    common: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    skin = get_context_skin(skin_id)
    output_dir = output_root / skin_id
    manifest_path = output_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            print(f"[resume] skip completed skin {skin_id}", flush=True)
            return previous

    games = build_context_games(experiment, model, skin_id)
    contracts = crn_contract(games)
    mechanism_hashes = sorted({row["mechanism_sha256"] for row in contracts})
    if len(mechanism_hashes) != len(experiment["games"]):
        raise RuntimeError("Unexpected number of unique mechanism signatures")

    expected_races = len(games)
    journal = RunJournal(output_dir, reset=True)
    effective_experiment = copy.deepcopy(experiment)
    effective_experiment["contextSkin"] = skin_id
    effective_experiment.pop("contextSkins", None)
    manifest = {
        "schema_version": "ai-race-context-skin-run-v1",
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        **common,
        "experiment_name": str(experiment["name"]),
        "experiment": effective_experiment,
        "experiment_config_sha256": sha256_file(experiment_path),
        "effective_experiment_sha256": hashlib.sha256(
            json.dumps(
                effective_experiment, sort_keys=True, separators=(",", ":")
            ).encode("utf-8")
        ).hexdigest(),
        "context_skin": {
            "id": skin.id,
            "family": skin.family,
            "description": skin.description,
            "protocol": CONTEXT_SKIN_PROTOCOL,
            "bounded_cover_story_set_size": len(SKINS),
            "scope": "bounded_eight_skin_cover_story_set",
            "prompt_version_prefix": skin.version,
            "template_sha256_by_action_mapping": {
                mapping_id: context_skin_sha256(skin_id, mapping_id)
                for mapping_id in ACTION_CODE_MAPPINGS
            },
        },
        "action_code_factor": {
            "protocol": ACTION_CODE_PROTOCOL,
            "display_order": ["P", "Q"],
            "mappings": {
                mapping_id: {
                    "safe_code": mapping.safe_code,
                    "unsafe_code": mapping.unsafe_code,
                }
                for mapping_id, mapping in ACTION_CODE_MAPPINGS.items()
            },
            "assignment": "safe_p for even rep; safe_q for odd rep",
            "balanced_within_skin": True,
            "separate_from_context_estimand": True,
        },
        "crn": {
            "pair_key": ["game_name", "rep", "player_index"],
            "seed_independent_of_context": True,
            "n_race_keys": len(contracts),
            "contract_sha256": hashlib.sha256(
                json.dumps(contracts, sort_keys=True, separators=(",", ":")).encode(
                    "utf-8"
                )
            ).hexdigest(),
        },
        "mechanism_sha256": mechanism_hashes,
        "game_config_sha256": {
            game_name: sha256_file(
                root / "ai_race" / "configs" / "game" / f"{game_name}.json"
            )
            for game_name in experiment["games"]
        },
        "expected_races": expected_races,
        **agents_provenance(root, experiment),
        "n_races": 0,
        "n_turns": 0,
        "error": None,
    }
    atomic_json(manifest_path, manifest)
    try:
        results = run_games_batched(
            games,
            backend,
            verbose=False,
            max_parse_retries=int(experiment.get("maxParseRetries", 3)),
            on_round_complete=journal.record_round,
        )
        if len(results) != expected_races or journal.race_count != expected_races:
            raise RuntimeError(
                f"Incomplete coverage: expected={expected_races}, "
                f"results={len(results)}, journal={journal.race_count}"
            )
        # A result's stochastic streams remain auditable after behavior diverges.
        base_seed = int(experiment["seed"])
        if any(result.game_seed != base_seed + result.rep for result in results):
            raise RuntimeError("A context skin violated the frozen CRN seed mapping")
    except Exception as error:
        manifest.update(
            status="failed",
            completed_utc=utc_now(),
            n_races=journal.race_count,
            n_turns=journal.turn_count,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(manifest_path, manifest)
        raise

    manifest.update(
        status="completed",
        completed_utc=utc_now(),
        n_races=journal.race_count,
        n_turns=journal.turn_count,
    )
    atomic_json(manifest_path, manifest)
    print(
        f"[completed] {skin_id}: {journal.race_count} races, "
        f"{journal.turn_count} decisions",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANE_SKINS), required=True)
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_REPETITIONS), default="smoke"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ai_race/configs/experiment/context_skin_invariance.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--required-gpu", default="H100")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment = load_json(config_path)
    configured_skins = list(experiment.get("contextSkins", []))
    partitioned = LANE_SKINS["a"] + LANE_SKINS["b"]
    if set(configured_skins) != set(SKINS) or set(partitioned) != set(SKINS):
        raise RuntimeError("Config and lane partition must cover every skin exactly once")
    if set(LANE_SKINS["a"]) & set(LANE_SKINS["b"]):
        raise RuntimeError("Context-skin lanes overlap")
    experiment["repetitions"] = PROFILE_REPETITIONS[args.profile]
    experiment["runPhase"] = "pilot"
    experiment["samplingSeedApplied"] = True

    detected_gpu = gpu_name()
    if args.required_gpu.lower() not in detected_gpu.lower():
        raise RuntimeError(
            f"GPU mismatch: required {args.required_gpu!r}, detected {detected_gpu!r}"
        )
    model_info = model_provenance(args.endpoint, args.model)
    ollama_version = request_json(args.endpoint, "/api/version").get("version")
    base_backend = OllamaBatchBackend(
        endpoint=args.endpoint,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workers=args.workers,
    )
    probe = "Reply with exactly ACTION: P or ACTION: Q."
    if base_backend.one(probe, 260726) != base_backend.one(probe, 260726):
        raise RuntimeError("Ollama fixed-seed reproducibility probe failed")
    backend = OpaqueActionBackend(base_backend)

    common = {
        "profile": args.profile,
        "lane": args.lane,
        "hostname": platform.node(),
        "gpu_name": detected_gpu,
        "ollama_version": ollama_version,
        "model": {
            "short_name": args.model,
            "path": f"ollama://localhost/{args.model}@{model_info['digest']}",
            "engine": "ollama",
            "config_sha256": model_info["digest"],
        },
        "ollama_model": model_info,
        "source_sha256": source_tree_sha256(root, (Path(__file__),)),
        "decoding": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "max_parse_retries": int(experiment.get("maxParseRetries", 3)),
            "num_ctx": 4096,
            "workers": args.workers,
            "seed_requested": True,
            "seed_probe_exact_match": True,
        },
        "package_versions": {
            "python": platform.python_version(),
            "ollama": str(ollama_version),
        },
        "base_seed": int(experiment["seed"]),
    }
    lane_manifest_path = output_root / "lane_manifest.json"
    lane_manifest = {
        "schema_version": "ai-race-context-skin-lane-v1",
        "status": "running",
        "started_utc": utc_now(),
        **common,
        "skins": LANE_SKINS[args.lane],
        "repetitions": int(experiment["repetitions"]),
        "runs": [],
    }
    atomic_json(lane_manifest_path, lane_manifest)
    started = time.monotonic()
    try:
        for skin_id in LANE_SKINS[args.lane]:
            lane_manifest["runs"].append(
                run_skin_shard(
                    root=root,
                    output_root=output_root,
                    experiment=experiment,
                    experiment_path=config_path,
                    skin_id=skin_id,
                    model=args.model,
                    backend=backend,
                    common=common,
                    resume=not args.no_resume,
                )
            )
            atomic_json(lane_manifest_path, lane_manifest)
    except Exception as error:
        lane_manifest.update(
            status="failed",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(lane_manifest_path, lane_manifest)
        raise
    lane_manifest.update(
        status="completed",
        completed_utc=utc_now(),
        elapsed_seconds=round(time.monotonic() - started, 3),
        error=None,
    )
    atomic_json(lane_manifest_path, lane_manifest)
    print(f"Lane {args.lane.upper()} completed in {lane_manifest['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
