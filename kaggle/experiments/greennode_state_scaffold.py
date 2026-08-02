"""Run the 2x2 state-computation scaffold factorial on Ollama GPUs."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path
from typing import Any

from ai_race.audit.state_scaffold import (
    SCAFFOLD_CONDITIONS,
    STATE_SCAFFOLD_PROTOCOL,
    add_state_scaffold,
)
from ai_race.dataio.config_loader import load_json, validate_experiment
from ai_race.dataio.recorder import RunJournal
from ai_race.runner.batch import run_games_batched
from kaggle.experiments.greennode_context_mapping_cross import crossed_crn_contract
from kaggle.experiments.greennode_context_skin import (
    OpaqueActionBackend,
    build_fully_crossed_context_games,
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


LANE_CONDITIONS = {
    "a": tuple(SCAFFOLD_CONDITIONS),
    "b": tuple(SCAFFOLD_CONDITIONS),
}
LANE_REP_PARITY = {"a": 0, "b": 1}
PROFILE_REPETITIONS = {"smoke": 2, "pilot": 32}
SKIN_ID = "abstract_contest"


def require_resume_match(previous: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = [key for key, value in expected.items() if previous.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "refusing to resume state-scaffold output with mismatched provenance: "
            + ", ".join(mismatches)
        )


def build_condition_games(
    experiment: dict[str, Any], model: str, condition_id: str
) -> list[Any]:
    effective = copy.deepcopy(experiment)
    effective.pop("scaffoldConditions", None)
    base = build_fully_crossed_context_games(effective, model, SKIN_ID)
    return add_state_scaffold(base, condition_id)


def run_condition(
    *,
    root: Path,
    output_root: Path,
    experiment: dict[str, Any],
    experiment_path: Path,
    condition_id: str,
    lane: str,
    model: str,
    backend: OpaqueActionBackend,
    common: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    output_dir = output_root / condition_id
    manifest_path = output_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            require_resume_match(
                previous,
                {
                    "protocol": STATE_SCAFFOLD_PROTOCOL,
                    "profile": common["profile"],
                    "lane": lane,
                    "model": common["model"],
                    "decoding": common["decoding"],
                    "source_sha256": common["source_sha256"],
                    "experiment_config_sha256": sha256_file(experiment_path),
                    "condition": {
                        "id": SCAFFOLD_CONDITIONS[condition_id].id,
                        "transition_card": SCAFFOLD_CONDITIONS[condition_id].transition_card,
                        "terminal_card": SCAFFOLD_CONDITIONS[condition_id].terminal_card,
                        "length_placebo": SCAFFOLD_CONDITIONS[condition_id].length_placebo,
                        "hidden_information_disclosed": False,
                    },
                },
            )
            print(f"[resume] skip completed {condition_id}", flush=True)
            return previous

    condition = SCAFFOLD_CONDITIONS[condition_id]
    games = [
        game
        for game in build_condition_games(experiment, model, condition_id)
        if game.rep % 2 == LANE_REP_PARITY[lane]
    ]
    crn_rows = crossed_crn_contract(games)
    expected_races = (
        len(experiment["games"]) * int(experiment["repetitions"])
    )
    if len(games) != expected_races:
        raise RuntimeError(
            f"coverage mismatch for {condition_id}: {len(games)} != {expected_races}"
        )
    expected_version_prefix = f"{STATE_SCAFFOLD_PROTOCOL}:{condition_id}:"
    if any(
        not game.config.prompt_version.startswith(expected_version_prefix)
        for game in games
    ):
        raise RuntimeError("state-scaffold prompt-version contract failed")

    journal = RunJournal(output_dir, reset=True)
    manifest: dict[str, Any] = {
        "schema_version": "ai-race-state-scaffold-run-v1",
        "protocol": STATE_SCAFFOLD_PROTOCOL,
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        **common,
        "context_skin": SKIN_ID,
        "condition": {
            "id": condition.id,
            "transition_card": condition.transition_card,
            "terminal_card": condition.terminal_card,
            "length_placebo": condition.length_placebo,
            "hidden_information_disclosed": False,
        },
        "rep_partition": f"rep % 2 == {LANE_REP_PARITY[lane]}",
        "action_mapping": "both mappings within every risk/rep block",
        "crn_contract_rows": len(crn_rows),
        "experiment": experiment,
        "experiment_config_sha256": sha256_file(experiment_path),
        "expected_races": expected_races,
        **agents_provenance(root, experiment),
        "n_races": 0,
        "n_turns": 0,
        "error": None,
    }
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        results = run_games_batched(
            games,
            backend,
            verbose=False,
            max_parse_retries=int(experiment.get("maxParseRetries", 3)),
            on_round_complete=journal.record_round,
        )
        if len(results) != expected_races or journal.race_count != expected_races:
            raise RuntimeError("state-scaffold run did not complete every race")
    except Exception as error:
        manifest.update(
            status="failed",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            n_races=journal.race_count,
            n_turns=journal.turn_count,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(manifest_path, manifest)
        raise
    manifest.update(
        status="completed",
        completed_utc=utc_now(),
        elapsed_seconds=round(time.monotonic() - started, 3),
        n_races=journal.race_count,
        n_turns=journal.turn_count,
    )
    atomic_json(manifest_path, manifest)
    print(f"[completed] {condition_id}: {journal.race_count} races", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANE_CONDITIONS), required=True)
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_REPETITIONS), default="smoke"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ai_race/configs/experiment/state_scaffold_factorial.json"),
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
    configured = set(experiment.get("scaffoldConditions", []))
    if configured != set(SCAFFOLD_CONDITIONS):
        raise RuntimeError("config must contain the frozen 2x2 scaffold cells")
    if experiment.get("contextSkins") != [SKIN_ID]:
        raise RuntimeError("state scaffold is frozen to abstract_contest")
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
    output_root.mkdir(parents=True, exist_ok=True)
    lane_manifest = {
        "schema_version": "ai-race-state-scaffold-lane-v1",
        "protocol": STATE_SCAFFOLD_PROTOCOL,
        "status": "running",
        "started_utc": utc_now(),
        **common,
        "conditions": list(LANE_CONDITIONS[args.lane]),
        "lane_shard": f"rep % 2 == {LANE_REP_PARITY[args.lane]}",
        "runs": [],
    }
    lane_path = output_root / "lane_manifest.json"
    atomic_json(lane_path, lane_manifest)
    started = time.monotonic()
    try:
        for condition_id in LANE_CONDITIONS[args.lane]:
            lane_manifest["runs"].append(
                run_condition(
                    root=root,
                    output_root=output_root,
                    experiment=experiment,
                    experiment_path=config_path,
                    condition_id=condition_id,
                    lane=args.lane,
                    model=args.model,
                    backend=backend,
                    common=common,
                    resume=not args.no_resume,
                )
            )
            atomic_json(lane_path, lane_manifest)
    except Exception as error:
        lane_manifest.update(
            status="failed",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(lane_path, lane_manifest)
        raise
    lane_manifest.update(
        status="completed",
        completed_utc=utc_now(),
        elapsed_seconds=round(time.monotonic() - started, 3),
        error=None,
    )
    atomic_json(lane_path, lane_manifest)


if __name__ == "__main__":
    main()
