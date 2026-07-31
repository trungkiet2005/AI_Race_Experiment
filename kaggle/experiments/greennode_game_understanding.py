"""Run game-understanding probes or calculator-aided behavior on GreenNode."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import time
from pathlib import Path
from typing import Any

from ai_race.audit.game_understanding import (
    AUDIT_PROTOCOL,
    add_decision_aid,
    build_probe_bank,
    canonical_rules_context,
    probe_conditions,
    render_probe,
    score_probe_response,
)
from ai_race.dataio.config_loader import load_json
from ai_race.dataio.recorder import RunJournal
from ai_race.engine.round import response_text
from ai_race.runner.batch import run_games_batched
from ai_race.runner.run_experiment import build_games_for_model
from kaggle.experiments.greennode_prompt_sensitivity import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    OllamaBatchBackend,
    atomic_json,
    gpu_name,
    model_provenance,
    request_json,
    sha256_file,
    source_tree_sha256,
    utc_now,
)


PROFILE_REPETITIONS = {
    "smoke": {"probes": 1, "behavior": 2},
    "pilot": {"probes": 5, "behavior": 10},
}
BASE_SEED = 260726


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def append_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()


def probe_bank_payload() -> list[dict[str, Any]]:
    return [
        {
            "id": item.id,
            "domain": item.domain,
            "direct_question": item.direct_question,
            "paraphrase_question": item.paraphrase_question,
            "answer_type": item.answer_type,
            "expected": item.expected,
            "allowed": list(item.allowed),
            "calculator_note": item.calculator_note,
        }
        for item in build_probe_bank()
    ]


def run_probes(
    *,
    output_root: Path,
    repetitions: int,
    backend: OllamaBatchBackend,
    common: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    output_dir = output_root / "probes"
    manifest_path = output_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed":
            print("[resume] probe audit already completed", flush=True)
            return prior

    output_dir.mkdir(parents=True, exist_ok=True)
    row_path = output_dir / "probe_outputs.jsonl"
    if row_path.exists():
        row_path.unlink()
    items = build_probe_bank()
    context = canonical_rules_context()
    requests: list[dict[str, Any]] = []
    for item_index, item in enumerate(items):
        for rep in range(repetitions):
            sampling_seed = BASE_SEED + item_index * 100 + rep
            for condition in probe_conditions(item):
                requests.append(
                    {
                        "item": item,
                        "condition": condition,
                        "rep": rep,
                        "sampling_seed": sampling_seed,
                        "prompt": render_probe(item, condition, rules_context=context),
                    }
                )
    manifest = {
        "schema_version": "ai-race-game-understanding-run-v1",
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        **common,
        "lane": "probes",
        "audit_protocol": AUDIT_PROTOCOL,
        "repetitions": repetitions,
        "probe_items": len(items),
        "expected_requests": len(requests),
        "probe_bank_sha256": text_sha256(
            json.dumps(probe_bank_payload(), sort_keys=True, separators=(",", ":"))
        ),
        "rules_context_sha256": text_sha256(context),
        "n_outputs": 0,
        "strict_valid": 0,
        "semantic_valid": 0,
        "strict_correct": 0,
        "semantic_correct": 0,
        "error": None,
    }
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        for start in range(0, len(requests), 128):
            batch = requests[start : start + 128]
            responses = backend(
                [request["prompt"] for request in batch],
                seeds=[request["sampling_seed"] for request in batch],
            )
            rows: list[dict[str, Any]] = []
            for request, response in zip(batch, responses):
                item = request["item"]
                raw = response_text(response)
                score = score_probe_response(item, raw)
                rows.append(
                    {
                        "protocol": AUDIT_PROTOCOL,
                        "item_id": item.id,
                        "domain": item.domain,
                        "answer_type": item.answer_type,
                        "expected": item.expected,
                        "condition": request["condition"],
                        "rep": request["rep"],
                        "sampling_seed": request["sampling_seed"],
                        "prompt": request["prompt"],
                        "raw_response": raw,
                        "strict_valid": score.strict_valid,
                        "semantic_valid": score.semantic_valid,
                        "strict_correct": score.strict_correct,
                        "semantic_correct": score.semantic_correct,
                        "parsed": score.parsed,
                    }
                )
            append_jsonl(row_path, rows)
            manifest["n_outputs"] += len(rows)
            for field in (
                "strict_valid", "semantic_valid", "strict_correct", "semantic_correct"
            ):
                manifest[field] += sum(int(row[field]) for row in rows)
            atomic_json(manifest_path, manifest)
            print(
                f"[probes] {manifest['n_outputs']}/{manifest['expected_requests']}",
                flush=True,
            )
        if manifest["n_outputs"] != manifest["expected_requests"]:
            raise RuntimeError("Probe output coverage is incomplete")
    except Exception as error:
        manifest.update(
            status="failed",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(manifest_path, manifest)
        raise
    manifest.update(
        status="completed",
        completed_utc=utc_now(),
        elapsed_seconds=round(time.monotonic() - started, 3),
    )
    atomic_json(manifest_path, manifest)
    return manifest


def run_behavior_condition(
    *,
    root: Path,
    output_root: Path,
    condition: str,
    repetitions: int,
    model: str,
    backend: OllamaBatchBackend,
    common: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    output_dir = output_root / "behavior" / condition
    manifest_path = output_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("status") == "completed":
            print(f"[resume] behavior {condition} already completed", flush=True)
            return prior
    experiment_path = root / "ai_race" / "configs" / "experiment" / "baseline.json"
    experiment = load_json(experiment_path)
    experiment.update(
        name=f"game_understanding_behavior__{condition}",
        repetitions=repetitions,
        runPhase="pilot",
        samplingSeedApplied=True,
    )
    games = build_games_for_model(experiment, model)
    if condition == "calculator_decision_card":
        games = add_decision_aid(games)
    elif condition != "canonical":
        raise ValueError(f"Unknown behavior condition: {condition}")
    journal = RunJournal(output_dir, reset=True)
    manifest = {
        "schema_version": "ai-race-game-understanding-run-v1",
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        **common,
        "lane": "behavior",
        "audit_protocol": AUDIT_PROTOCOL,
        "condition": condition,
        "repetitions": repetitions,
        "expected_races": len(games),
        "experiment": experiment,
        "experiment_config_sha256": sha256_file(experiment_path),
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
        if len(results) != len(games) or journal.race_count != len(games):
            raise RuntimeError("Behavior coverage is incomplete")
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
    print(
        f"[behavior] {condition}: {journal.race_count} races, {journal.turn_count} decisions",
        flush=True,
    )
    return manifest


def run_behavior(**kwargs: Any) -> dict[str, Any]:
    runs = [
        run_behavior_condition(condition=condition, **kwargs)
        for condition in ("canonical", "calculator_decision_card")
    ]
    return {"status": "completed", "runs": runs}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=("probes", "behavior"), required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_REPETITIONS), default="smoke")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--required-gpu", default="H100")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    repetitions = PROFILE_REPETITIONS[args.profile][args.lane]
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
    probe = "Reply with exactly ANSWER: 5"
    if backend.one(probe, BASE_SEED) != backend.one(probe, BASE_SEED):
        raise RuntimeError("Ollama fixed-seed reproducibility probe failed")
    common = {
        "profile": args.profile,
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
        "base_seed": BASE_SEED,
        "decoding": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_ctx": 4096,
            "workers": args.workers,
            "seed_requested": True,
            "seed_probe_exact_match": True,
        },
        "package_versions": {
            "python": platform.python_version(),
            "ollama": str(ollama_version),
        },
    }
    output_root.mkdir(parents=True, exist_ok=True)
    if args.lane == "probes":
        result = run_probes(
            output_root=output_root,
            repetitions=repetitions,
            backend=backend,
            common=common,
            resume=not args.no_resume,
        )
    else:
        result = run_behavior(
            root=root,
            output_root=output_root,
            repetitions=repetitions,
            model=args.model,
            backend=backend,
            common=common,
            resume=not args.no_resume,
        )
    atomic_json(
        output_root / f"{args.lane}_summary.json",
        {
            "status": "completed",
            "completed_utc": utc_now(),
            "lane": args.lane,
            "profile": args.profile,
            "result": result,
        },
    )


if __name__ == "__main__":
    main()
