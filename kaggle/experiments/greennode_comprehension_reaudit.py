"""Re-audit atomic AI-Race comprehension on two GreenNode mailbox workers.

This coordinator deliberately runs one frozen greedy smoke repetition.  It uses
the 41-item atomic bank from :mod:`ai_race.audit.game_understanding`, rather
than the older state scaffold, and retains enough evidence to re-render and
re-score every response.  Calculator conditions measure uptake of a disclosed
tool result; they are never pooled into unaided-understanding accuracy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time
import uuid
from typing import Any, Sequence

from ai_race.audit.game_understanding import (
    AUDIT_PROTOCOL,
    ProbeItem,
    build_probe_bank,
    canonical_rules_context,
    probe_conditions,
    render_probe,
    score_probe_response,
)
from kaggle.experiments.greennode_crossmodel_scaffold_admission import (
    GREENNODE_MODELS,
)
from kaggle.experiments.greennode_heterogeneous_dyad import (
    MODEL_LABELS,
    PROTOCOL as MAILBOX_TRANSPORT_PROTOCOL,
    atomic_json,
    sha256_file,
    utc_now,
)


REAUDIT_PROTOCOL = "ai-race-game-understanding-reaudit-v1"
SCHEMA_VERSION = "ai-race-game-understanding-reaudit-run-v1"
BASE_SEED = 260802
REPETITIONS = 1
EXPECTED_ITEMS = 41
EXPECTED_REQUESTS_PER_MODEL = 137
MODEL_KEYS = ("qwen25_7b", "mistral7_01")
SOURCE_RELATIVE_PATHS = (
    "ai_race/audit/game_understanding.py",
    "ai_race/engine/game.py",
    "ai_race/engine/scoring.py",
    "ai_race/engine/state.py",
    "ai_race/prompts/ai_race_en.txt",
    "kaggle/experiments/greennode_comprehension_reaudit.py",
    "kaggle/experiments/greennode_heterogeneous_dyad.py",
)
CLAIM_BOUNDARY = (
    "Unaided rows measure rule and arithmetic performance under the frozen prompt "
    "bank; they do not establish an internal world model. Calculator rows measure "
    "uptake of a disclosed verified result and are excluded from unaided accuracy."
)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def object_sha256(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value).encode("utf-8")).hexdigest()


def text_sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def atomic_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(_canonical_json(row) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def probe_bank_payload(items: Sequence[ProbeItem] | None = None) -> list[dict[str, Any]]:
    selected = list(items) if items is not None else build_probe_bank()
    return [asdict(item) for item in selected]


def sampling_seed(model_key: str, item_id: str, condition: str) -> int:
    payload = f"{REAUDIT_PROTOCOL}:{BASE_SEED}:{model_key}:{item_id}:{condition}:0"
    digest = hashlib.sha256(payload.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "big") & 0x7FFFFFFF


def build_frozen_requests() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build the complete one-repetition bank and its exact provenance hashes."""
    items = build_probe_bank()
    if len(items) != EXPECTED_ITEMS:
        raise RuntimeError(
            f"atomic probe bank changed: expected {EXPECTED_ITEMS}, got {len(items)}"
        )
    context = canonical_rules_context()
    requests: list[dict[str, Any]] = []
    ordinal = 0
    for item in items:
        for condition in probe_conditions(item):
            prompt = render_probe(item, condition, rules_context=context)
            requests.append(
                {
                    "ordinal": ordinal,
                    "repetition": 0,
                    "item_id": item.id,
                    "domain": item.domain,
                    "answer_type": item.answer_type,
                    "expected": item.expected,
                    "allowed": list(item.allowed),
                    "condition": condition,
                    "measurement_class": (
                        "calculator_tool_uptake"
                        if condition == "calculator"
                        else "unaided_understanding"
                    ),
                    "prompt": prompt,
                    "prompt_sha256": text_sha256(prompt),
                }
            )
            ordinal += 1
    if len(requests) != EXPECTED_REQUESTS_PER_MODEL:
        raise RuntimeError(
            "atomic prompt coverage changed: expected "
            f"{EXPECTED_REQUESTS_PER_MODEL}, got {len(requests)}"
        )
    request_keys = [
        (row["item_id"], row["condition"], row["repetition"])
        for row in requests
    ]
    if len(request_keys) != len(set(request_keys)):
        raise RuntimeError("frozen request bank contains duplicate keys")
    hashes = {
        "probe_bank_sha256": object_sha256(probe_bank_payload(items)),
        "rules_context_sha256": text_sha256(context),
        "rendered_request_bank_sha256": object_sha256(requests),
    }
    return requests, hashes


def source_receipts(repo_root: Path) -> tuple[list[dict[str, Any]], str]:
    receipts: list[dict[str, Any]] = []
    for relative in SOURCE_RELATIVE_PATHS:
        path = repo_root / relative
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen source artifact: {path}")
        receipts.append(
            {
                "path": relative,
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return receipts, object_sha256(receipts)


def load_admission_receipts(root: Path) -> dict[str, dict[str, Any]]:
    """Load exact prior receipts and require an immutable model digest."""
    receipts: dict[str, dict[str, Any]] = {}
    for model_key in MODEL_KEYS:
        spec = GREENNODE_MODELS[model_key]
        path = root / str(spec["short_name"]) / "smoke" / "admission.json"
        if not path.is_file():
            raise FileNotFoundError(f"missing admission receipt: {path}")
        payload = json.loads(path.read_text(encoding="utf-8"))
        model = payload.get("model")
        if not isinstance(model, dict) or not model.get("digest"):
            raise RuntimeError(f"admission receipt lacks model digest: {path}")
        expected_source = f"hf://{spec['repo_id']}@{spec['revision']}"
        if model.get("kaggle_model_source") != expected_source:
            raise RuntimeError(f"admission model source mismatch: {path}")
        if payload.get("status") not in {"complete", "completed"}:
            raise RuntimeError(f"admission receipt is not complete: {path}")
        receipts[model_key] = {
            "path": path.as_posix(),
            "sha256": sha256_file(path),
            "status": payload["status"],
            "passed": bool(payload.get("passed", False)),
            "evidence_class": payload.get("evidence_class"),
            "protocol": payload.get("protocol"),
            "model_source": model["kaggle_model_source"],
            "model_digest": model["digest"],
        }
    return receipts


def validate_worker_receipt(
    receipt: dict[str, Any], *, model_key: str, worker_id: str
) -> None:
    spec = GREENNODE_MODELS[model_key]
    if receipt.get("status") != "ready":
        raise RuntimeError(f"worker {worker_id!r} is not ready")
    if receipt.get("protocol") != MAILBOX_TRANSPORT_PROTOCOL:
        raise RuntimeError(f"worker {worker_id!r} transport protocol mismatch")
    if receipt.get("worker_id") != worker_id:
        raise RuntimeError(f"worker {worker_id!r} receipt worker_id mismatch")
    if receipt.get("model_key") != model_key:
        raise RuntimeError(f"worker {worker_id!r} model_key mismatch")
    model = receipt.get("model")
    if not isinstance(model, dict):
        raise RuntimeError(f"worker {worker_id!r} lacks model provenance")
    if (
        model.get("repo_id") != spec["repo_id"]
        or model.get("revision") != spec["revision"]
    ):
        raise RuntimeError(f"worker {worker_id!r} model revision mismatch")
    decoding = receipt.get("decoding")
    if not isinstance(decoding, dict):
        raise RuntimeError(f"worker {worker_id!r} lacks decoding provenance")
    if (
        float(decoding.get("temperature", -1.0)) != 0.0
        or decoding.get("do_sample") is not False
        or decoding.get("sampling_seed_applied") is not False
    ):
        raise RuntimeError(f"worker {worker_id!r} is not greedy T0")


def load_worker_receipts(
    queue_root: Path, workers: dict[str, str]
) -> dict[str, dict[str, Any]]:
    ready: dict[str, dict[str, Any]] = {}
    for model_key, worker_id in workers.items():
        path = queue_root / "workers" / worker_id / "ready.json"
        if not path.is_file():
            raise FileNotFoundError(f"worker is not ready: {path}")
        receipt = json.loads(path.read_text(encoding="utf-8"))
        validate_worker_receipt(receipt, model_key=model_key, worker_id=worker_id)
        ready[model_key] = {
            **receipt,
            "ready_receipt_path": path.as_posix(),
            "ready_receipt_sha256": sha256_file(path),
        }
    return ready


def wait_for_response(path: Path, timeout_seconds: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout_seconds
    while not path.is_file():
        if time.monotonic() >= deadline:
            raise TimeoutError(f"timed out waiting for {path}")
        time.sleep(0.1)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("error"):
        raise RuntimeError(str(payload["error"]))
    return payload


def validate_response_envelope(payload: dict[str, Any], request_id: str) -> None:
    if payload.get("protocol") != MAILBOX_TRANSPORT_PROTOCOL:
        raise RuntimeError(
            "worker transport protocol mismatch: expected "
            f"{MAILBOX_TRANSPORT_PROTOCOL!r}, got {payload.get('protocol')!r}"
        )
    if payload.get("request_id") != request_id:
        raise RuntimeError(
            "worker request_id mismatch: expected "
            f"{request_id!r}, got {payload.get('request_id')!r}"
        )
    if not isinstance(payload.get("responses"), list):
        raise RuntimeError("worker response envelope lacks a response list")


def _rate(rows: Sequence[dict[str, Any]], field: str) -> float | None:
    if not rows:
        return None
    return sum(bool(row[field]) for row in rows) / len(rows)


def summarize_rows(
    rows: Sequence[dict[str, Any]], requests: Sequence[dict[str, Any]]
) -> dict[str, Any]:
    """Re-render/re-score all rows and report fail-closed coverage/integrity."""
    request_by_key = {
        (str(request["item_id"]), str(request["condition"]), 0): request
        for request in requests
    }
    item_by_id = {item.id: item for item in build_probe_bank()}
    expected_keys = {
        (model_key, item_id, condition, repetition)
        for model_key in MODEL_KEYS
        for item_id, condition, repetition in request_by_key
    }
    observed_keys: list[tuple[str, str, str, int]] = []
    unexpected: list[tuple[str, str, str, int]] = []
    prompt_mismatches = 0
    prompt_hash_mismatches = 0
    rescore_mismatches = 0
    for row in rows:
        key = (
            str(row.get("model_key")),
            str(row.get("item_id")),
            str(row.get("condition")),
            int(row.get("repetition", -1)),
        )
        observed_keys.append(key)
        request = request_by_key.get((key[1], key[2], key[3]))
        if key not in expected_keys or request is None:
            unexpected.append(key)
            continue
        if row.get("prompt") != request["prompt"]:
            prompt_mismatches += 1
        if row.get("prompt_sha256") != text_sha256(str(row.get("prompt", ""))):
            prompt_hash_mismatches += 1
        item = item_by_id[key[1]]
        rescored = asdict(score_probe_response(item, str(row.get("raw_response", ""))))
        if row.get("rescore") != rescored:
            rescore_mismatches += 1

    observed_set = set(observed_keys)
    missing = sorted(expected_keys - observed_set)
    duplicate_count = len(observed_keys) - len(observed_set)
    coverage_passed = not missing and not unexpected and duplicate_count == 0
    integrity_passed = (
        prompt_mismatches == 0
        and prompt_hash_mismatches == 0
        and rescore_mismatches == 0
    )

    cells: dict[str, dict[str, Any]] = {}
    for model_key in MODEL_KEYS:
        model_rows = [row for row in rows if row.get("model_key") == model_key]
        domains = sorted({str(row["domain"]) for row in model_rows})
        conditions = sorted({str(row["condition"]) for row in model_rows})
        for domain in domains:
            for condition in conditions:
                subset = [
                    row
                    for row in model_rows
                    if row["domain"] == domain and row["condition"] == condition
                ]
                if not subset:
                    continue
                cells[f"{model_key}/{domain}/{condition}"] = {
                    "model_key": model_key,
                    "domain": domain,
                    "condition": condition,
                    "measurement_class": (
                        "calculator_tool_uptake"
                        if condition == "calculator"
                        else "unaided_understanding"
                    ),
                    "n": len(subset),
                    "strict_parse_rate": _rate(
                        [row["rescore"] for row in subset], "strict_valid"
                    ),
                    "semantic_parse_rate": _rate(
                        [row["rescore"] for row in subset], "semantic_valid"
                    ),
                    "strict_accuracy": _rate(
                        [row["rescore"] for row in subset], "strict_correct"
                    ),
                    "semantic_accuracy": _rate(
                        [row["rescore"] for row in subset], "semantic_correct"
                    ),
                }

    by_model: dict[str, dict[str, Any]] = {}
    for model_key in MODEL_KEYS:
        model_rows = [row for row in rows if row.get("model_key") == model_key]
        unaided = [row for row in model_rows if row["condition"] != "calculator"]
        calculator = [row for row in model_rows if row["condition"] == "calculator"]
        by_model[model_key] = {
            "n": len(model_rows),
            "expected_n": len(requests),
            "unaided_n": len(unaided),
            "unaided_semantic_accuracy": _rate(
                [row["rescore"] for row in unaided], "semantic_correct"
            ),
            "calculator_tool_uptake_n": len(calculator),
            "calculator_tool_uptake_semantic_accuracy": _rate(
                [row["rescore"] for row in calculator], "semantic_correct"
            ),
            "strict_parse_rate": _rate(
                [row["rescore"] for row in model_rows], "strict_valid"
            ),
        }

    return {
        "schema_version": "ai-race-game-understanding-reaudit-summary-v1",
        "protocol": REAUDIT_PROTOCOL,
        "audit_passed": coverage_passed and integrity_passed,
        "coverage": {
            "passed": coverage_passed,
            "expected_rows": len(expected_keys),
            "observed_rows": len(rows),
            "unique_rows": len(observed_set),
            "missing_count": len(missing),
            "unexpected_count": len(unexpected),
            "duplicate_count": duplicate_count,
            "missing_examples": [list(key) for key in missing[:20]],
            "unexpected_examples": [list(key) for key in unexpected[:20]],
        },
        "integrity": {
            "passed": integrity_passed,
            "prompt_mismatches": prompt_mismatches,
            "prompt_hash_mismatches": prompt_hash_mismatches,
            "rescore_mismatches": rescore_mismatches,
        },
        "by_model": by_model,
        "by_domain_condition": cells,
        "claim_boundary": CLAIM_BOUNDARY,
    }


def _source_commit(repo_root: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _evidence_class(admissions: dict[str, dict[str, Any]]) -> str:
    return (
        "admitted_comprehension_reaudit_smoke"
        if all(receipt["passed"] for receipt in admissions.values())
        else "diagnostic_unadmitted_comprehension_reaudit"
    )


def run(args: argparse.Namespace) -> int:
    repo_root = args.repo_root.resolve()
    if not (repo_root / "ai_race").is_dir():
        raise FileNotFoundError(f"repo root lacks ai_race package: {repo_root}")
    if not str(args.lane_block).strip():
        raise ValueError("lane block must be non-empty")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite existing run: {manifest_path}")

    requests, bank_hashes = build_frozen_requests()
    sources, source_bundle_sha256 = source_receipts(repo_root)
    workers = {
        "qwen25_7b": args.qwen_worker,
        "mistral7_01": args.mistral_worker,
    }
    manifest: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "protocol": REAUDIT_PROTOCOL,
        "source_audit_protocol": AUDIT_PROTOCOL,
        "status": "running",
        "evidence_class": "pending_validation",
        "started_utc": utc_now(),
        "completed_utc": None,
        "lane_block": args.lane_block,
        "source_commit": _source_commit(repo_root),
        "source_artifacts": sources,
        "source_bundle_sha256": source_bundle_sha256,
        **bank_hashes,
        "design": {
            "profile": "smoke",
            "repetitions": REPETITIONS,
            "probe_items": EXPECTED_ITEMS,
            "requests_per_model": EXPECTED_REQUESTS_PER_MODEL,
            "models": list(MODEL_KEYS),
            "temperature": 0.0,
            "do_sample": False,
            "sampling_seed_applied": False,
            "mailbox_transport_protocol": MAILBOX_TRANSPORT_PROTOCOL,
            "calculator_measurement": "tool_uptake_not_unaided_understanding",
        },
        "claim_boundary": CLAIM_BOUNDARY,
        "workers": {},
        "admission_receipts": {},
        "expected_responses": len(requests) * len(workers),
        "n_responses": 0,
        "error": None,
    }
    atomic_json(manifest_path, manifest)
    started = time.monotonic()
    try:
        ready = load_worker_receipts(args.queue_root, workers)
        admissions = load_admission_receipts(args.admission_root)
        evidence_class = _evidence_class(admissions)
        manifest.update(
            workers=ready,
            admission_receipts=admissions,
            evidence_class=evidence_class,
        )
        atomic_json(manifest_path, manifest)

        pending: dict[str, tuple[Path, Path, str]] = {}
        for model_key, worker_id in workers.items():
            request_id = f"comprehension-reaudit-{args.lane_block}-{uuid.uuid4().hex}"
            request_path = (
                args.queue_root / "requests" / worker_id / f"{request_id}.json"
            )
            response_path = (
                args.queue_root / "responses" / worker_id / f"{request_id}.json"
            )
            atomic_json(
                request_path,
                {
                    "protocol": MAILBOX_TRANSPORT_PROTOCOL,
                    "audit_protocol": REAUDIT_PROTOCOL,
                    "request_id": request_id,
                    "lane_block": args.lane_block,
                    "model_key": model_key,
                    "repetition": 0,
                    **bank_hashes,
                    "prompts": [row["prompt"] for row in requests],
                    "seeds": [
                        sampling_seed(model_key, row["item_id"], row["condition"])
                        for row in requests
                    ],
                },
            )
            pending[model_key] = (request_path, response_path, worker_id)

        output_rows: list[dict[str, Any]] = []
        mailbox_rows: list[dict[str, Any]] = []
        for model_key, (request_path, response_path, worker_id) in pending.items():
            payload = wait_for_response(response_path, args.timeout_seconds)
            validate_response_envelope(payload, request_path.stem)
            responses = payload["responses"]
            if len(responses) != len(requests):
                raise RuntimeError(
                    f"{model_key}: expected {len(requests)} responses, "
                    f"received {len(responses)}"
                )
            mailbox_rows.append(
                {
                    "model_key": model_key,
                    "worker_id": worker_id,
                    "request_id": request_path.stem,
                    "request_sha256": sha256_file(request_path),
                    "response_sha256": sha256_file(response_path),
                    "n_responses": len(responses),
                    "response_protocol": payload["protocol"],
                }
            )
            item_by_id = {item.id: item for item in build_probe_bank()}
            admission = admissions[model_key]
            for request, raw_response in zip(requests, responses):
                item = item_by_id[request["item_id"]]
                raw = str(raw_response)
                score = asdict(score_probe_response(item, raw))
                output_rows.append(
                    {
                        **request,
                        "protocol": REAUDIT_PROTOCOL,
                        "source_audit_protocol": AUDIT_PROTOCOL,
                        "lane_block": args.lane_block,
                        "model_key": model_key,
                        "model": MODEL_LABELS[model_key],
                        "model_digest": admission["model_digest"],
                        "worker_id": worker_id,
                        "request_id": request_path.stem,
                        "sampling_seed": sampling_seed(
                            model_key, request["item_id"], request["condition"]
                        ),
                        "sampling_seed_applied": False,
                        "temperature": 0.0,
                        "raw_response": raw,
                        "rescore": score,
                        "prior_admission_passed": admission["passed"],
                        "admission_receipt_sha256": admission["sha256"],
                        "evidence_class": evidence_class,
                    }
                )

        summary = summarize_rows(output_rows, requests)
        if not summary["audit_passed"]:
            raise RuntimeError("fail-closed coverage or re-score integrity check failed")
        raw_path = output / "comprehension_reaudit_raw.jsonl"
        summary_path = output / "domain_condition_summary.json"
        mailbox_path = output / "mailbox_audit.json"
        atomic_jsonl(raw_path, output_rows)
        atomic_json(summary_path, summary)
        atomic_json(mailbox_path, mailbox_rows)
        artifacts = {
            raw_path.name: sha256_file(raw_path),
            summary_path.name: sha256_file(summary_path),
            mailbox_path.name: sha256_file(mailbox_path),
        }
        manifest.update(
            status="completed",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            n_responses=len(output_rows),
            audit_passed=True,
            artifacts=artifacts,
            model_summary=summary["by_model"],
        )
        atomic_json(manifest_path, manifest)
        return 0
    except Exception as error:
        manifest.update(
            status="failed",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(error).__name__}: {error}",
            audit_passed=False,
        )
        atomic_json(manifest_path, manifest)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--queue-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--admission-root", type=Path, required=True)
    parser.add_argument("--qwen-worker", required=True)
    parser.add_argument("--mistral-worker", required=True)
    parser.add_argument("--lane-block", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=600.0)
    return parser.parse_args()


def main() -> int:
    return run(parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
