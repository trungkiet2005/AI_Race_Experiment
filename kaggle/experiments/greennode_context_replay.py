"""Run context comprehension gates and matched fixed-state replay on Ollama.

This is intentionally separate from ``greennode_context_skin.py``.  The live
runner estimates total trajectory effects; this runner freezes engine-reachable
states so later action differences cannot feed back into the compared prompts.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable

from ai_race.audit.context_replay import (
    CONTEXT_COMPREHENSION_PROTOCOL,
    CONTEXT_REPLAY_PROTOCOL,
    comprehension_summary,
    generate_reachable_states,
    paired_coverage,
    run_comprehension_matrix,
    run_replay_matrix,
)
from ai_race.dataio.config_loader import load_game_config, load_json
from ai_race.prompts.context_skins import (
    ACTION_CODE_MAPPINGS,
    ACTION_CODE_PROTOCOL,
    CONTEXT_SKIN_PROTOCOL,
    SKINS,
    context_skin_sha256,
)
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
PROFILE_STATES_PER_RISK = {"smoke": 2, "pilot": 32}
ESTIMAND_COLUMNS = [
    "pair_id",
    "state_id",
    "trajectory_id",
    "game_name",
    "max_private_risk",
    "source_seed",
    "round",
    "player_index",
    "skin_id",
    "mapping_id",
    "sampling_seed",
    "action",
    "unsafe",
    "opaque_action_code",
    "parse_failed",
    "retry_count",
    "own_progress",
    "opponent_progress",
    "progress_gap",
    "own_stage_payoff",
    "opponent_stage_payoff",
    "own_unsafe_count",
    "opponent_unsafe_count",
    "prompt_sha256",
]


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    _atomic_text(path, payload)


def atomic_csv(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def _admission_by_cell(rows: list[dict[str, Any]]) -> dict[str, Any]:
    cells: dict[str, Any] = {}
    passed = True
    for skin_id in sorted({str(row["skin_id"]) for row in rows}):
        for mapping_id in ACTION_CODE_MAPPINGS:
            subset = [
                row
                for row in rows
                if row["skin_id"] == skin_id and row["mapping_id"] == mapping_id
            ]
            summary = comprehension_summary(subset)
            cells[f"{skin_id}/{mapping_id}"] = summary
            passed &= bool(summary["passed"])
    return {
        "passed": passed,
        "rule": "every skin/action-mapping cell must pass the frozen thresholds",
        "overall": comprehension_summary(rows),
        "by_cell": cells,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANE_SKINS), required=True)
    parser.add_argument(
        "--profile", choices=sorted(PROFILE_STATES_PER_RISK), default="smoke"
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
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--required-gpu", default="H100")
    parser.add_argument("--stop-on-admission-failure", action="store_true")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.is_file() and not args.no_resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            if previous.get("lane") != args.lane or previous.get("profile") != args.profile:
                raise RuntimeError("Completed replay output belongs to a different lane/profile")
            print(f"[resume] completed context replay: {output_root}", flush=True)
            return

    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment = load_json(config_path)
    configured_skins = list(experiment.get("contextSkins", []))
    partition = LANE_SKINS["a"] + LANE_SKINS["b"]
    if set(configured_skins) != set(SKINS) or set(partition) != set(SKINS):
        raise RuntimeError("Context config and replay lanes must cover every skin exactly once")
    if set(LANE_SKINS["a"]) & set(LANE_SKINS["b"]):
        raise RuntimeError("Replay lanes overlap")

    states_per_risk = PROFILE_STATES_PER_RISK[args.profile]
    base_seed = int(experiment["seed"])
    configs = [
        load_game_config(
            root / "ai_race" / "configs" / "game" / f"{game_name}.json",
            model=args.model,
        )
        for game_name in experiment["games"]
    ]
    detected_gpu = gpu_name()
    if args.required_gpu.lower() not in detected_gpu.lower():
        raise RuntimeError(
            f"GPU mismatch: required {args.required_gpu!r}, detected {detected_gpu!r}"
        )
    model_info = model_provenance(args.endpoint, args.model)
    ollama_version = request_json(args.endpoint, "/api/version").get("version")
    source_hash = source_tree_sha256(root, (Path(__file__),))

    states = generate_reachable_states(
        configs, states_per_config=states_per_risk, base_seed=base_seed
    )
    state_path = output_root / "reachable_states.jsonl"
    atomic_jsonl(state_path, [state.to_dict() for state in states])

    expected_comprehension = len(LANE_SKINS[args.lane]) * len(ACTION_CODE_MAPPINGS) * 16
    expected_replay = len(states) * len(LANE_SKINS[args.lane]) * len(ACTION_CODE_MAPPINGS)
    manifest: dict[str, Any] = {
        "schema_version": "ai-race-context-fixed-state-run-v1",
        "status": "running",
        "evidence_class": "protocol",
        "started_utc": utc_now(),
        "completed_utc": None,
        "lane": args.lane,
        "profile": args.profile,
        "skins": LANE_SKINS[args.lane],
        "protocols": {
            "replay": CONTEXT_REPLAY_PROTOCOL,
            "comprehension": CONTEXT_COMPREHENSION_PROTOCOL,
            "context_skin": CONTEXT_SKIN_PROTOCOL,
            "action_codes": ACTION_CODE_PROTOCOL,
        },
        "source_sha256": source_hash,
        "experiment_config_sha256": sha256_file(config_path),
        "game_config_sha256": {
            game_name: sha256_file(
                root / "ai_race" / "configs" / "game" / f"{game_name}.json"
            )
            for game_name in experiment["games"]
        },
        "template_sha256": {
            f"{skin_id}/{mapping_id}": context_skin_sha256(skin_id, mapping_id)
            for skin_id in LANE_SKINS[args.lane]
            for mapping_id in ACTION_CODE_MAPPINGS
        },
        "model": {
            "name": args.model,
            "digest": model_info.get("digest"),
            "size": model_info.get("size"),
            "details": model_info.get("details", {}),
            "engine": "ollama",
        },
        "hardware": {
            "hostname": platform.node(),
            "gpu_name": detected_gpu,
            "ollama_version": ollama_version,
            "python": platform.python_version(),
        },
        "decoding": {
            "comprehension_temperature": 0.0,
            "replay_temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "max_parse_retries": args.max_parse_retries,
            "seed_requested": True,
        },
        "state_bank": {
            "generator": "real AIRaceGame with deterministic scripted actions",
            "selection": "deterministic round-stratified selection",
            "states_per_risk": states_per_risk,
            "n_states": len(states),
            "base_seed": base_seed,
            "artifact": _artifact(state_path),
        },
        "expected_comprehension_rows": expected_comprehension,
        "expected_replay_rows": expected_replay,
        "admission": None,
        "coverage": None,
        "artifacts": {},
        "error": None,
    }
    atomic_json(manifest_path, manifest)

    try:
        comprehension_backend = OllamaBatchBackend(
            endpoint=args.endpoint,
            model=args.model,
            temperature=0.0,
            max_tokens=args.max_tokens,
            workers=args.workers,
        )
        reproducibility_prompt = "Return exactly one line: ANSWER: YES"
        if comprehension_backend.one(reproducibility_prompt, base_seed) != comprehension_backend.one(
            reproducibility_prompt, base_seed
        ):
            raise RuntimeError("Ollama fixed-seed reproducibility probe failed")
        comprehension_rows = run_comprehension_matrix(
            configs,
            LANE_SKINS[args.lane],
            comprehension_backend,
            seed=base_seed,
            batch_size=args.batch_size,
        )
        if len(comprehension_rows) != expected_comprehension:
            raise RuntimeError(
                f"Comprehension coverage mismatch: {len(comprehension_rows)} != {expected_comprehension}"
            )
        comprehension_path = output_root / "comprehension_raw.jsonl"
        atomic_jsonl(comprehension_path, comprehension_rows)
        admission = _admission_by_cell(comprehension_rows)
        admission_path = output_root / "comprehension_admission.json"
        atomic_json(admission_path, admission)
        manifest["admission"] = admission
        manifest["artifacts"].update(
            comprehension_raw=_artifact(comprehension_path),
            comprehension_admission=_artifact(admission_path),
        )
        atomic_json(manifest_path, manifest)

        if args.stop_on_admission_failure and not admission["passed"]:
            manifest.update(
                status="completed",
                evidence_class="diagnostic_comprehension_failed",
                completed_utc=utc_now(),
            )
            atomic_json(manifest_path, manifest)
            print("[gate] comprehension failed; replay intentionally not launched", flush=True)
            return

        replay_backend = OllamaBatchBackend(
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            workers=args.workers,
        )
        replay_rows = run_replay_matrix(
            states,
            LANE_SKINS[args.lane],
            replay_backend,
            batch_size=args.batch_size,
            max_parse_retries=args.max_parse_retries,
        )
        if len(replay_rows) != expected_replay:
            raise RuntimeError(
                f"Replay coverage mismatch: {len(replay_rows)} != {expected_replay}"
            )
        coverage = paired_coverage(replay_rows, LANE_SKINS[args.lane])
        if not coverage["passed"]:
            raise RuntimeError("Matched replay pair coverage failed")

        replay_path = output_root / "replay_raw.jsonl"
        estimand_path = output_root / "paired_estimand_input.csv"
        coverage_path = output_root / "paired_coverage.json"
        atomic_jsonl(replay_path, replay_rows)
        atomic_csv(estimand_path, replay_rows, ESTIMAND_COLUMNS)
        atomic_json(coverage_path, coverage)
        manifest["coverage"] = coverage
        manifest["artifacts"].update(
            replay_raw=_artifact(replay_path),
            paired_estimand_input=_artifact(estimand_path),
            paired_coverage=_artifact(coverage_path),
        )
        manifest.update(
            status="completed",
            evidence_class=("pilot" if admission["passed"] else "diagnostic_comprehension_failed"),
            completed_utc=utc_now(),
        )
        atomic_json(manifest_path, manifest)
        print(
            f"[completed] lane={args.lane} states={len(states)} "
            f"replay_rows={len(replay_rows)} admission={admission['passed']}",
            flush=True,
        )
    except Exception as error:
        manifest.update(
            status="failed",
            evidence_class="failed",
            completed_utc=utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
