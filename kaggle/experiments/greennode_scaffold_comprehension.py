"""Run the frozen state-scaffold comprehension admission gate on Ollama."""
from __future__ import annotations

import argparse
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable

from ai_race.audit.context_replay import CONTEXT_COMPREHENSION_PROTOCOL
from ai_race.audit.scaffold_comprehension import (
    SCAFFOLD_COMPREHENSION_PROTOCOL,
    build_scaffold_probe_requests,
    request_bank_sha256,
    run_scaffold_comprehension,
    scaffold_admission_summary,
)
from ai_race.audit.state_scaffold import SCAFFOLD_CONDITIONS, STATE_SCAFFOLD_PROTOCOL
from ai_race.dataio.config_loader import load_game_config, load_json, validate_experiment
from ai_race.prompts.context_skins import ACTION_CODE_MAPPINGS
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


PROFILE_REPETITIONS = {"smoke": 1, "pilot": 5}


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    payload = "".join(
        json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(payload, encoding="utf-8", newline="\n")
    os.replace(temporary, path)


def _artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def write_admission_artifacts(
    output_root: Path,
    rows: list[dict[str, Any]],
    summary: dict[str, Any],
    provenance: dict[str, Any],
) -> tuple[Path, Path, dict[str, Any]]:
    """Atomically write raw evidence first, then its self-contained admission."""
    raw_path = output_root / "comprehension_raw.jsonl"
    admission_path = output_root / "admission.json"
    _atomic_jsonl(raw_path, rows)
    admission = {
        "schema_version": "ai-race-state-scaffold-admission-v1",
        "protocol": SCAFFOLD_COMPREHENSION_PROTOCOL,
        "generated_utc": utc_now(),
        "behavior_source_sha256": provenance["behavior_source_sha256"],
        "behavior_experiment_config_sha256": provenance[
            "behavior_experiment_config_sha256"
        ],
        "model_digest": provenance["model"]["digest"],
        "decoding": provenance["behavior_decoding"],
        **summary,
        "provenance": provenance,
        "artifacts": {"comprehension_raw": _artifact(raw_path, output_root)},
    }
    atomic_json(admission_path, admission)
    return raw_path, admission_path, admission


def require_completed_resume_match(
    output_root: Path,
    previous: dict[str, Any],
    expected_provenance: dict[str, Any],
) -> dict[str, Any]:
    """Validate every evidence-bearing resume binding before skipping work."""
    if previous.get("status") != "completed":
        raise RuntimeError("resume validation requires a completed run manifest")
    expected_decoding = {
        **expected_provenance["decoding"],
        "seed_probe_exact_match": True,
    }
    expected_fields = {
        "protocol": SCAFFOLD_COMPREHENSION_PROTOCOL,
        "profile": expected_provenance["profile"],
        "repetitions": expected_provenance["repetitions"],
        "condition_ids": expected_provenance["condition_ids"],
        "mapping_ids": expected_provenance["mapping_ids"],
        "admission_source_sha256": expected_provenance[
            "admission_source_sha256"
        ],
        "behavior_source_sha256": expected_provenance["behavior_source_sha256"],
        "behavior_experiment_config_sha256": expected_provenance[
            "behavior_experiment_config_sha256"
        ],
        "request_bank_sha256": expected_provenance["request_bank_sha256"],
        "model": expected_provenance["model"],
        "decoding": expected_decoding,
        "behavior_decoding": expected_provenance["behavior_decoding"],
        "expected_requests": expected_provenance["expected_requests"],
    }
    mismatches = [
        key for key, value in expected_fields.items() if previous.get(key) != value
    ]
    if mismatches:
        raise RuntimeError(
            "refusing to resume scaffold admission with mismatched provenance: "
            + ", ".join(mismatches)
        )

    raw_path = output_root / "comprehension_raw.jsonl"
    admission_path = output_root / "admission.json"
    if not raw_path.is_file() or not admission_path.is_file():
        raise RuntimeError("completed scaffold admission is missing an evidence artifact")
    actual_raw = _artifact(raw_path, output_root)
    actual_admission = _artifact(admission_path, output_root)
    if previous.get("artifacts") != {
        "comprehension_raw": actual_raw,
        "admission": actual_admission,
    }:
        raise RuntimeError("completed run-manifest artifact hashes do not match disk")

    admission = json.loads(admission_path.read_text(encoding="utf-8"))
    if admission.get("schema_version") != "ai-race-state-scaffold-admission-v1":
        raise RuntimeError("completed admission has an unexpected schema")
    if admission.get("protocol") != SCAFFOLD_COMPREHENSION_PROTOCOL:
        raise RuntimeError("completed admission has an unexpected protocol")
    admission_expected = {
        "behavior_source_sha256": expected_provenance["behavior_source_sha256"],
        "behavior_experiment_config_sha256": expected_provenance[
            "behavior_experiment_config_sha256"
        ],
        "model_digest": expected_provenance["model"]["digest"],
        "decoding": expected_provenance["behavior_decoding"],
    }
    admission_mismatches = [
        key for key, value in admission_expected.items() if admission.get(key) != value
    ]
    provenance_keys = (
        "profile",
        "repetitions",
        "condition_ids",
        "mapping_ids",
        "admission_source_sha256",
        "behavior_source_sha256",
        "behavior_experiment_config_sha256",
        "request_bank_sha256",
        "model",
        "behavior_decoding",
        "expected_requests",
    )
    recorded_provenance = admission.get("provenance")
    if not isinstance(recorded_provenance, dict):
        admission_mismatches.append("provenance")
    else:
        admission_mismatches.extend(
            f"provenance.{key}"
            for key in provenance_keys
            if recorded_provenance.get(key) != expected_provenance.get(key)
        )
        if recorded_provenance.get("decoding") != expected_decoding:
            admission_mismatches.append("provenance.decoding")
    if admission.get("artifacts") != {"comprehension_raw": actual_raw}:
        admission_mismatches.append("artifacts.comprehension_raw")
    if previous.get("admission_passed") is not bool(admission.get("passed")):
        admission_mismatches.append("admission_passed")
    if admission_mismatches:
        raise RuntimeError(
            "refusing to resume scaffold admission with mismatched admission: "
            + ", ".join(admission_mismatches)
        )
    return admission


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
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
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--required-gpu", default="6000")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "run_manifest.json"

    if args.model != DEFAULT_MODEL or args.temperature != 0.0 or args.max_tokens != 16:
        raise RuntimeError(
            "frozen protocol requires the default model, temperature=0, and max_tokens=16"
        )
    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment = validate_experiment(load_json(config_path))
    condition_ids = list(experiment.get("scaffoldConditions", []))
    mapping_ids = list(experiment.get("actionCodeMappings", []))
    if condition_ids != list(SCAFFOLD_CONDITIONS):
        raise RuntimeError("config must contain the frozen scaffold conditions in order")
    if mapping_ids != list(ACTION_CODE_MAPPINGS):
        raise RuntimeError("config must contain both frozen opaque mappings in order")
    if experiment.get("contextSkins") != ["abstract_contest"]:
        raise RuntimeError("scaffold comprehension is frozen to abstract_contest")

    game_paths = {
        name: root / "ai_race" / "configs" / "game" / f"{name}.json"
        for name in experiment["games"]
    }
    configs = [load_game_config(path, model=args.model) for path in game_paths.values()]
    config = min(configs, key=lambda item: abs(item.max_private_risk - 0.6))
    repetitions = PROFILE_REPETITIONS[args.profile]
    requests = build_scaffold_probe_requests(
        config,
        condition_ids=condition_ids,
        mapping_ids=mapping_ids,
        repetitions=repetitions,
        seed=int(experiment["seed"]),
    )
    expected_requests = (
        len(condition_ids) * len(mapping_ids) * repetitions * 16
    )
    if len(requests) != expected_requests:
        raise RuntimeError(
            f"admission request coverage mismatch: {len(requests)} != {expected_requests}"
        )

    detected_gpu = gpu_name()
    if args.required_gpu and args.required_gpu.lower() not in detected_gpu.lower():
        raise RuntimeError(
            f"GPU mismatch: required {args.required_gpu!r}, detected {detected_gpu!r}"
        )
    model_info = model_provenance(args.endpoint, args.model)
    if model_info.get("digest") != args.required_model_digest:
        raise RuntimeError("resolved Ollama digest does not match --required-model-digest")
    ollama_version = request_json(args.endpoint, "/api/version").get("version")
    admission_source_hash = source_tree_sha256(root, (Path(__file__),))
    behavior_source_hash = source_tree_sha256(
        root, (root / "kaggle" / "experiments" / "greennode_state_scaffold.py",)
    )
    prompt_bank_hash = request_bank_sha256(requests)
    provenance = {
        "profile": args.profile,
        "repetitions": repetitions,
        "condition_ids": condition_ids,
        "mapping_ids": mapping_ids,
        "protocols": {
            "admission": SCAFFOLD_COMPREHENSION_PROTOCOL,
            "probe_bank_and_scoring": CONTEXT_COMPREHENSION_PROTOCOL,
            "behavioral_scaffold": STATE_SCAFFOLD_PROTOCOL,
        },
        "admission_source_sha256": admission_source_hash,
        "behavior_source_sha256": behavior_source_hash,
        "behavior_experiment_config_sha256": sha256_file(config_path),
        "game_config_sha256": {
            name: sha256_file(path) for name, path in game_paths.items()
        },
        "request_bank_sha256": prompt_bank_hash,
        "model": model_info,
        "hardware": {
            "hostname": platform.node(),
            "gpu_name": detected_gpu,
            "ollama_version": ollama_version,
            "python": platform.python_version(),
        },
        "decoding": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "seed_requested": True,
        },
        "behavior_decoding": {
            "temperature": 0.0,
            "max_tokens": 16,
            "workers": args.workers,
            "seed_requested": True,
            "seed_probe_exact_match": True,
        },
        "base_seed": int(experiment["seed"]),
        "expected_requests": expected_requests,
    }
    if manifest_path.is_file() and not args.no_resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            admission = require_completed_resume_match(
                output_root, previous, provenance
            )
            print(f"[resume] verified completed scaffold admission: {output_root}", flush=True)
            if not admission.get("passed"):
                raise SystemExit(2)
            return
    manifest: dict[str, Any] = {
        "schema_version": "ai-race-state-scaffold-admission-run-v1",
        "protocol": SCAFFOLD_COMPREHENSION_PROTOCOL,
        "status": "running",
        "evidence_class": "protocol",
        "started_utc": utc_now(),
        "completed_utc": None,
        **provenance,
        "artifacts": {},
        "error": None,
    }
    atomic_json(manifest_path, manifest)

    try:
        backend = OllamaBatchBackend(
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            workers=args.workers,
        )
        reproducibility_prompt = "Return exactly one line: ANSWER: YES"
        probe_a = backend.one(reproducibility_prompt, int(experiment["seed"]))
        probe_b = backend.one(reproducibility_prompt, int(experiment["seed"]))
        if probe_a != probe_b:
            raise RuntimeError("Ollama fixed-seed reproducibility probe failed")
        provenance["decoding"]["seed_probe_exact_match"] = True
        rows = run_scaffold_comprehension(
            requests, backend, batch_size=args.batch_size
        )
        summary = scaffold_admission_summary(
            rows,
            config,
            condition_ids=condition_ids,
            mapping_ids=mapping_ids,
            repetitions=repetitions,
        )
        raw_path, admission_path, admission = write_admission_artifacts(
            output_root, rows, summary, provenance
        )
        manifest.update(
            status="completed",
            evidence_class=("admitted" if admission["passed"] else "diagnostic_comprehension_failed"),
            completed_utc=utc_now(),
            admission_passed=bool(admission["passed"]),
            artifacts={
                "comprehension_raw": _artifact(raw_path, output_root),
                "admission": _artifact(admission_path, output_root),
            },
        )
        manifest["decoding"]["seed_probe_exact_match"] = True
        atomic_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(
            status="failed",
            evidence_class="failed",
            completed_utc=utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(manifest_path, manifest)
        raise

    print(
        f"[completed] requests={len(rows)} admission={admission['passed']} "
        f"raw_sha256={manifest['artifacts']['comprehension_raw']['sha256']}",
        flush=True,
    )
    # A scientifically valid failed gate still writes complete evidence, but a
    # non-zero process status prevents shell pipelines from launching gameplay.
    if not admission["passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
