"""Run the versioned AI Race surface-form sensitivity matrix on two Ollama pods."""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Any

from ai_race.dataio.config_loader import load_json
from ai_race.dataio.recorder import RunJournal
from ai_race.prompts.sensitivity import VARIANTS, apply_prompt_variant
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


LANE_VARIANTS = {
    "a": [
        "canonical",
        "order_actions_reversed",
        "order_payoffs_reversed",
        "order_state_reversed",
        "position_goal_first",
        "position_risk_near_response",
        "lexical_synonyms",
        "paraphrase_instruction",
        "voice_impersonal",
    ],
    "b": [
        "format_markdown",
        "format_xml",
        "format_dense",
        "format_extra_spacing",
        "format_numbered_state",
        "emphasis_uppercase",
        "boundary_compact",
        "emotional_importance",
        "noise_minor_typo",
    ],
}
PROFILE_REPETITIONS = {"smoke": 2, "pilot": 10}


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def run_variant_shard(
    *,
    root: Path,
    output_root: Path,
    variant_id: str,
    repetitions: int,
    model: str,
    backend: OllamaBatchBackend,
    common: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    variant = VARIANTS[variant_id]
    output_dir = output_root / variant_id
    manifest_path = output_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            print(f"[resume] skip completed variant {variant_id}", flush=True)
            return previous

    config_dir = root / "ai_race" / "configs"
    experiment_path = config_dir / "experiment" / "baseline.json"
    experiment = load_json(experiment_path)
    experiment.update(
        name=f"surface_sensitivity__{variant_id}",
        repetitions=repetitions,
        runPhase="pilot",
        samplingSeedApplied=True,
        promptVariant=variant_id,
    )
    games = build_games_for_model(experiment, model)
    expected_races = len(games)
    canonical_template = (root / "ai_race" / "prompts" / "ai_race_en.txt").read_text(
        encoding="utf-8"
    )
    effective_template = apply_prompt_variant(canonical_template, variant_id)
    journal = RunJournal(output_dir, reset=True)
    manifest = {
        "schema_version": "ai-race-surface-sensitivity-run-v1",
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        **common,
        "experiment_name": experiment["name"],
        "experiment": experiment,
        "experiment_config_sha256": sha256_file(experiment_path),
        "effective_experiment_sha256": text_sha256(
            json.dumps(experiment, sort_keys=True, separators=(",", ":"))
        ),
        "prompt_variant": {
            "id": variant.id,
            "family": variant.family,
            "description": variant.description,
            "interpretation": variant.interpretation,
        },
        "prompt_version": (
            "ai-race-fairgame-v3" if variant_id == "canonical" else variant.version
        ),
        "prompt_sha256": text_sha256(effective_template),
        "prompt_characters": len(effective_template),
        "game_config_sha256": {
            game_name: sha256_file(config_dir / "game" / f"{game_name}.json")
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
        f"[completed] {variant_id}: {journal.race_count} races, "
        f"{journal.turn_count} decisions",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=sorted(LANE_VARIANTS), required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_REPETITIONS), default="smoke")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--required-gpu", default="H100")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    variants = LANE_VARIANTS[args.lane]
    repetitions = PROFILE_REPETITIONS[args.profile]
    detected_gpu = gpu_name()
    if args.required_gpu.lower() not in detected_gpu.lower():
        raise RuntimeError(
            f"GPU mismatch: required {args.required_gpu!r}, detected {detected_gpu!r}"
        )
    model_info = model_provenance(args.endpoint, args.model)
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
            "max_parse_retries": 3,
            "num_ctx": 4096,
            "workers": args.workers,
            "seed_requested": True,
            "seed_probe_exact_match": True,
        },
        "package_versions": {
            "python": platform.python_version(),
            "ollama": str(ollama_version),
        },
        "base_seed": 260726,
    }
    lane_manifest_path = output_root / "lane_manifest.json"
    lane_manifest = {
        "schema_version": "ai-race-surface-sensitivity-lane-v1",
        "status": "running",
        "started_utc": utc_now(),
        **common,
        "variants": variants,
        "repetitions": repetitions,
        "runs": [],
    }
    atomic_json(lane_manifest_path, lane_manifest)
    started = time.monotonic()
    try:
        for variant_id in variants:
            lane_manifest["runs"].append(
                run_variant_shard(
                    root=root,
                    output_root=output_root,
                    variant_id=variant_id,
                    repetitions=repetitions,
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
