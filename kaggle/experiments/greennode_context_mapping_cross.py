"""Run the fully crossed context × opaque-action-mapping diagnostic on Ollama.

Every risk/context/repetition block is duplicated under both Safe=P and Safe=Q
while retaining identical game and sampling seeds.  This runner is deliberately
separate from the completed parity-balanced pilot so its frozen artifacts remain
reconstructable and the follow-up has an unambiguous protocol identity.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

from ai_race.dataio.config_loader import load_json, validate_experiment
from ai_race.dataio.recorder import RunJournal
from ai_race.prompts.context_skins import (
    ACTION_CODE_MAPPINGS,
    ACTION_CODE_PROTOCOL,
    CONTEXT_SKIN_PROTOCOL,
    SKINS,
    context_skin_sha256,
    get_context_skin,
)
from ai_race.runner.batch import run_games_batched
from kaggle.experiments.greennode_context_skin import (
    OpaqueActionBackend,
    build_fully_crossed_context_games,
    mechanism_sha256,
)
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


PROFILE_REPETITIONS = {"smoke": 2, "pilot": 32}
PROTOCOL = "ai-race-context-mapping-cross-v1"
CROSS_LANE_SKINS = {"a": tuple(SKINS), "b": tuple(SKINS)}
LANE_REP_PARITY = {"a": 0, "b": 1}


def require_resume_match(previous: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = [key for key, value in expected.items() if previous.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "refusing to resume context-mapping output with mismatched provenance: "
            + ", ".join(mismatches)
        )


def crossed_crn_contract(games: list[Any]) -> list[dict[str, Any]]:
    """Validate two mappings per paired block and return its immutable receipt."""
    rows = []
    for game in games:
        mapping_id = game.action_code_mapping.id
        rows.append(
            {
                "game_name": game.config.name,
                "max_private_risk": game.config.max_private_risk,
                "rep": game.rep,
                "mapping_id": mapping_id,
                "game_seed": game.seed,
                "seat0_round1_sampling_seed": game.sampling_seed(0, 1),
                "seat1_round1_sampling_seed": game.sampling_seed(1, 1),
                "mechanism_sha256": mechanism_sha256(game.config),
            }
        )

    observed_keys = {
        (row["game_name"], row["rep"], row["mapping_id"]) for row in rows
    }
    if len(observed_keys) != len(rows):
        raise RuntimeError("Duplicate fully-crossed CRN cell")

    by_block: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for row in rows:
        by_block.setdefault((row["game_name"], row["rep"]), []).append(row)
    expected_mappings = set(ACTION_CODE_MAPPINGS)
    for block, block_rows in by_block.items():
        if {row["mapping_id"] for row in block_rows} != expected_mappings:
            raise RuntimeError(f"Missing action mapping in paired block {block}")
        invariant_fields = (
            "game_seed",
            "seat0_round1_sampling_seed",
            "seat1_round1_sampling_seed",
            "mechanism_sha256",
        )
        for field in invariant_fields:
            if len({row[field] for row in block_rows}) != 1:
                raise RuntimeError(
                    f"CRN invariant {field!r} differs across mappings in {block}"
                )
    return rows


def run_crossed_skin(
    *,
    root: Path,
    output_root: Path,
    experiment: dict[str, Any],
    experiment_path: Path,
    skin_id: str,
    lane: str,
    model: str,
    backend: OpaqueActionBackend,
    common: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    skin = get_context_skin(skin_id)
    output_dir = output_root / skin_id
    manifest_path = output_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            require_resume_match(
                previous,
                {
                    "protocol": PROTOCOL,
                    "profile": common["profile"],
                    "lane": lane,
                    "model": common["model"],
                    "decoding": common["decoding"],
                    "source_sha256": common["source_sha256"],
                    "experiment_config_sha256": sha256_file(experiment_path),
                },
            )
            print(f"[resume] skip completed skin {skin_id}", flush=True)
            return previous

    games = [
        game
        for game in build_fully_crossed_context_games(experiment, model, skin_id)
        if game.rep % 2 == LANE_REP_PARITY[lane]
    ]
    contracts = crossed_crn_contract(games)
    mechanism_hashes = sorted({row["mechanism_sha256"] for row in contracts})
    if len(mechanism_hashes) != len(experiment["games"]):
        raise RuntimeError("Unexpected number of unique mechanism signatures")

    expected_races = (
        len(experiment["games"])
        * (int(experiment["repetitions"]) // 2)
        * len(ACTION_CODE_MAPPINGS)
    )
    if len(games) != expected_races:
        raise RuntimeError(
            f"Coverage construction failed: expected={expected_races}, got={len(games)}"
        )

    journal = RunJournal(output_dir, reset=True)
    effective_experiment = copy.deepcopy(experiment)
    effective_experiment["contextSkin"] = skin_id
    effective_experiment.pop("contextSkins", None)
    manifest: dict[str, Any] = {
        "schema_version": "ai-race-context-mapping-cross-run-v1",
        "protocol": PROTOCOL,
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        **common,
        "experiment_name": str(experiment["name"]),
        "rep_partition": f"rep % 2 == {LANE_REP_PARITY[lane]}",
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
            "template_sha256_by_action_mapping": {
                mapping_id: context_skin_sha256(skin_id, mapping_id)
                for mapping_id in ACTION_CODE_MAPPINGS
            },
        },
        "action_code_factor": {
            "protocol": ACTION_CODE_PROTOCOL,
            "mappings": {
                mapping_id: {
                    "safe_code": mapping.safe_code,
                    "unsafe_code": mapping.unsafe_code,
                }
                for mapping_id, mapping in ACTION_CODE_MAPPINGS.items()
            },
            "assignment": "both mappings within every risk/context/rep block",
            "fully_crossed_within_seed": True,
        },
        "crn": {
            "paired_block_key": ["game_name", "rep"],
            "cell_key": ["game_name", "rep", "mapping_id"],
            "seed_independent_of_context_and_mapping": True,
            "n_mapping_cells": len(contracts),
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
        base_seed = int(experiment["seed"])
        if any(result.game_seed != base_seed + result.rep for result in results):
            raise RuntimeError("A crossed cell violated the frozen CRN seed mapping")
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
    parser.add_argument("--lane", choices=sorted(CROSS_LANE_SKINS), required=True)
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_REPETITIONS), default="smoke"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(
            "ai_race/configs/experiment/context_mapping_fully_crossed.json"
        ),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--required-model-digest", required=True)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--required-gpu", default="6000")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment = validate_experiment(load_json(config_path))
    if args.model != DEFAULT_MODEL or args.temperature != 0.0 or args.max_tokens != 16:
        raise RuntimeError(
            "frozen protocol requires the default model, temperature=0, and max_tokens=16"
        )
    if experiment.get("actionCodeAssignment") != "fully_crossed_within_seed":
        raise RuntimeError("Config must freeze fully_crossed_within_seed assignment")
    if set(experiment.get("actionCodeMappings", [])) != set(ACTION_CODE_MAPPINGS):
        raise RuntimeError("Config must contain both frozen action-code mappings")
    if set(experiment.get("contextSkins", [])) != set(SKINS):
        raise RuntimeError("Config must cover every frozen context skin")
    experiment["repetitions"] = PROFILE_REPETITIONS[args.profile]
    if experiment["repetitions"] % 2:
        raise RuntimeError("repetitions must be even for balanced lane sharding")
    experiment["runPhase"] = "pilot"
    experiment["samplingSeedApplied"] = True

    detected_gpu = gpu_name()
    if args.required_gpu and args.required_gpu.lower() not in detected_gpu.lower():
        raise RuntimeError(
            f"GPU mismatch: required {args.required_gpu!r}, detected {detected_gpu!r}"
        )
    model_info = model_provenance(args.endpoint, args.model)
    if model_info["digest"] != args.required_model_digest:
        raise RuntimeError("resolved Ollama digest does not match --required-model-digest")
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
    lane_manifest: dict[str, Any] = {
        "schema_version": "ai-race-context-mapping-cross-lane-v1",
        "protocol": PROTOCOL,
        "status": "running",
        "started_utc": utc_now(),
        **common,
        "skins": CROSS_LANE_SKINS[args.lane],
        "lane_shard": f"rep % 2 == {LANE_REP_PARITY[args.lane]}",
        "repetitions": int(experiment["repetitions"]),
        "mappings_per_seed": len(ACTION_CODE_MAPPINGS),
        "runs": [],
    }
    atomic_json(lane_manifest_path, lane_manifest)
    started = time.monotonic()
    try:
        for skin_id in CROSS_LANE_SKINS[args.lane]:
            lane_manifest["runs"].append(
                run_crossed_skin(
                    root=root,
                    output_root=output_root,
                    experiment=experiment,
                    experiment_path=config_path,
                    skin_id=skin_id,
                    lane=args.lane,
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


if __name__ == "__main__":
    main()
