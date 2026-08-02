"""Run the paired positive payoff-scale invariance diagnostic on Ollama GPUs."""
from __future__ import annotations

import argparse
import copy
import json
import platform
import time
from pathlib import Path
from typing import Any

from ai_race.audit.payoff_scale import (
    PAYOFF_SCALES,
    PAYOFF_SCALE_PROTOCOL,
    payoff_scale_id,
    payoff_scale_signature,
    scale_games,
)
from ai_race.dataio.config_loader import load_json, validate_experiment
from ai_race.dataio.recorder import RunJournal
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


LANE_SCALES = {"a": PAYOFF_SCALES, "b": PAYOFF_SCALES}
LANE_REP_PARITY = {"a": 0, "b": 1}
PROFILE_REPETITIONS = {"smoke": 2, "pilot": 32}


def require_resume_match(previous: dict[str, Any], expected: dict[str, Any]) -> None:
    mismatches = [key for key, value in expected.items() if previous.get(key) != value]
    if mismatches:
        raise RuntimeError(
            "refusing to resume payoff-scale output with mismatched provenance: "
            + ", ".join(mismatches)
        )


def build_scaled_games(
    experiment: dict[str, Any], model: str, scale: float
) -> list[Any]:
    effective = copy.deepcopy(experiment)
    effective.pop("payoffScales", None)
    effective.pop("payoffScaleAssignment", None)
    return scale_games(build_games_for_model(effective, model), scale)


def validate_scaled_games(games: list[Any], scale: float) -> None:
    expected_version = f"{PAYOFF_SCALE_PROTOCOL}:{payoff_scale_id(scale)}"
    if not games:
        raise RuntimeError("payoff-scale builder produced no races")
    if any(game.config.prompt_version != expected_version for game in games):
        raise RuntimeError("payoff-scale prompt-version contract failed")
    keys = {(game.config.name, game.rep) for game in games}
    if len(keys) != len(games):
        raise RuntimeError("duplicate risk/repetition cell in payoff-scale lane")
    for game in games:
        signature = payoff_scale_signature(game)
        if signature["game_seed"] != int(game.seed):
            raise RuntimeError("game seed changed during payoff scaling")


def run_scale(
    *,
    root: Path,
    output_root: Path,
    experiment: dict[str, Any],
    experiment_path: Path,
    scale: float,
    lane: str,
    model: str,
    backend: OllamaBatchBackend,
    common: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    scale_name = payoff_scale_id(scale)
    output_dir = output_root / scale_name
    manifest_path = output_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            require_resume_match(
                previous,
                {
                    "protocol": PAYOFF_SCALE_PROTOCOL,
                    "profile": common["profile"],
                    "lane": lane,
                    "model": common["model"],
                    "decoding": common["decoding"],
                    "source_sha256": common["source_sha256"],
                    "experiment_config_sha256": sha256_file(experiment_path),
                    "payoff_scale": scale,
                },
            )
            print(f"[resume] skip completed {scale_name}", flush=True)
            return previous

    games = [
        game
        for game in build_scaled_games(experiment, model, scale)
        if game.rep % 2 == LANE_REP_PARITY[lane]
    ]
    validate_scaled_games(games, scale)
    expected_races = (
        len(experiment["games"]) * int(experiment["repetitions"]) // 2
    )
    if len(games) != expected_races:
        raise RuntimeError(
            f"coverage mismatch for {scale_name}: {len(games)} != {expected_races}"
        )

    journal = RunJournal(output_dir, reset=True)
    manifest: dict[str, Any] = {
        "schema_version": "ai-race-payoff-scale-run-v1",
        "protocol": PAYOFF_SCALE_PROTOCOL,
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        **common,
        "payoff_scale": scale,
        "payoff_scale_id": scale_name,
        "rep_partition": f"rep % 2 == {LANE_REP_PARITY[lane]}",
        "payoff_fields_scaled": [
            "payoff_safe_safe",
            "payoff_safe_unsafe",
            "payoff_unsafe_safe",
            "payoff_unsafe_unsafe",
            "race_prize",
        ],
        "non_payoff_fields_unchanged": True,
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
            raise RuntimeError("payoff-scale run did not complete every race")
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
    print(f"[completed] {scale_name}: {journal.race_count} races", flush=True)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANE_SCALES), required=True)
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_REPETITIONS), default="smoke"
    )
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ai_race/configs/experiment/payoff_scale_invariance.json"),
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
    configured = tuple(float(value) for value in experiment.get("payoffScales", []))
    if set(configured) != set(PAYOFF_SCALES):
        raise RuntimeError("config must contain the four frozen payoff scales")
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
    backend = OllamaBatchBackend(
        endpoint=args.endpoint,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workers=args.workers,
    )
    probe = "Reply with exactly ACTION: SAFE or ACTION: UNSAFE."
    if backend.one(probe, 260726) != backend.one(probe, 260726):
        raise RuntimeError("Ollama fixed-seed reproducibility probe failed")

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
        "schema_version": "ai-race-payoff-scale-lane-v1",
        "protocol": PAYOFF_SCALE_PROTOCOL,
        "status": "running",
        "started_utc": utc_now(),
        **common,
        "scales": list(LANE_SCALES[args.lane]),
        "lane_shard": f"rep % 2 == {LANE_REP_PARITY[args.lane]}",
        "runs": [],
    }
    lane_path = output_root / "lane_manifest.json"
    atomic_json(lane_path, lane_manifest)
    started = time.monotonic()
    try:
        for scale in LANE_SCALES[args.lane]:
            lane_manifest["runs"].append(
                run_scale(
                    root=root,
                    output_root=output_root,
                    experiment=experiment,
                    experiment_path=config_path,
                    scale=scale,
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
