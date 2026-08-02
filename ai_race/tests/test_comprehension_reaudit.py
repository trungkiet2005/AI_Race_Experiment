"""Acceptance tests for the two-worker atomic comprehension re-audit."""
from __future__ import annotations

from argparse import Namespace
from dataclasses import asdict
import json
from pathlib import Path

import pytest

from ai_race.audit.game_understanding import (
    build_probe_bank,
    probe_conditions,
    score_probe_response,
)
from kaggle.experiments.greennode_crossmodel_scaffold_admission import (
    GREENNODE_MODELS,
)
from kaggle.experiments.greennode_heterogeneous_dyad import (
    PROTOCOL as MAILBOX_TRANSPORT_PROTOCOL,
    atomic_json,
)
from kaggle.experiments import greennode_comprehension_reaudit as reaudit


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]


def _perfect_rows(requests):
    item_by_id = {item.id: item for item in build_probe_bank()}
    rows = []
    for model_key in reaudit.MODEL_KEYS:
        for request in requests:
            item = item_by_id[request["item_id"]]
            raw = f"ANSWER: {item.expected}"
            rows.append(
                {
                    **request,
                    "model_key": model_key,
                    "raw_response": raw,
                    "rescore": asdict(score_probe_response(item, raw)),
                }
            )
    return rows


def _ready_receipt(model_key: str, worker_id: str) -> dict:
    spec = GREENNODE_MODELS[model_key]
    return {
        "status": "ready",
        "protocol": MAILBOX_TRANSPORT_PROTOCOL,
        "worker_id": worker_id,
        "model_key": model_key,
        "model": {
            "repo_id": spec["repo_id"],
            "revision": spec["revision"],
            "family": spec["family"],
            "short_name": spec["short_name"],
        },
        "decoding": {
            "temperature": 0.0,
            "do_sample": False,
            "sampling_seed_applied": False,
            "max_new_tokens": 16,
            "batch_size": 1,
        },
    }


def _write_prerequisites(tmp_path: Path) -> tuple[Path, Path]:
    queue_root = tmp_path / "queue"
    admission_root = tmp_path / "admission"
    worker_by_model = {
        "qwen25_7b": "worker-qwen",
        "mistral7_01": "worker-mistral",
    }
    for model_key, worker_id in worker_by_model.items():
        atomic_json(
            queue_root / "workers" / worker_id / "ready.json",
            _ready_receipt(model_key, worker_id),
        )
        spec = GREENNODE_MODELS[model_key]
        atomic_json(
            admission_root / spec["short_name"] / "smoke" / "admission.json",
            {
                "schema_version": "test-admission-v1",
                "status": "complete",
                "protocol": "prior-scaffold-admission",
                "passed": False,
                "evidence_class": "diagnostic_comprehension_failed",
                "model": {
                    "kaggle_model_source": (
                        f"hf://{spec['repo_id']}@{spec['revision']}"
                    ),
                    "digest": f"digest-{model_key}",
                },
            },
        )
    return queue_root, admission_root


def _args(tmp_path: Path) -> Namespace:
    queue_root, admission_root = _write_prerequisites(tmp_path)
    return Namespace(
        repo_root=REPOSITORY_ROOT,
        queue_root=queue_root,
        output=tmp_path / "output",
        admission_root=admission_root,
        qwen_worker="worker-qwen",
        mistral_worker="worker-mistral",
        lane_block="reaudit-smoke-block-0",
        timeout_seconds=1.0,
    )


def test_frozen_bank_is_complete_atomic_and_hash_stable() -> None:
    requests, hashes = reaudit.build_frozen_requests()
    items = build_probe_bank()
    assert len(items) == reaudit.EXPECTED_ITEMS == 41
    assert len(requests) == reaudit.EXPECTED_REQUESTS_PER_MODEL == 137
    assert len({(row["item_id"], row["condition"]) for row in requests}) == 137
    assert sum(len(probe_conditions(item)) for item in items) == 137
    assert sum(row["condition"] == "calculator" for row in requests) == 41
    assert all(
        row["measurement_class"] == "calculator_tool_uptake"
        for row in requests
        if row["condition"] == "calculator"
    )
    assert all(
        row["measurement_class"] == "unaided_understanding"
        for row in requests
        if row["condition"] != "calculator"
    )
    assert hashes == reaudit.build_frozen_requests()[1]
    assert set(hashes) == {
        "probe_bank_sha256",
        "rules_context_sha256",
        "rendered_request_bank_sha256",
    }


def test_summary_rechecks_complete_coverage_and_separates_tool_uptake() -> None:
    requests, _ = reaudit.build_frozen_requests()
    rows = _perfect_rows(requests)
    summary = reaudit.summarize_rows(rows, requests)
    assert summary["audit_passed"]
    assert summary["coverage"] == {
        "passed": True,
        "expected_rows": 274,
        "observed_rows": 274,
        "unique_rows": 274,
        "missing_count": 0,
        "unexpected_count": 0,
        "duplicate_count": 0,
        "missing_examples": [],
        "unexpected_examples": [],
    }
    for model_key in reaudit.MODEL_KEYS:
        model = summary["by_model"][model_key]
        assert model["n"] == 137
        assert model["unaided_n"] == 96
        assert model["calculator_tool_uptake_n"] == 41
        assert model["unaided_semantic_accuracy"] == 1.0
        assert model["calculator_tool_uptake_semantic_accuracy"] == 1.0
    assert all(
        cell["measurement_class"] == "calculator_tool_uptake"
        for cell in summary["by_domain_condition"].values()
        if cell["condition"] == "calculator"
    )
    assert "not establish an internal world model" in summary["claim_boundary"]


def test_summary_fails_closed_on_missing_duplicate_prompt_or_rescore() -> None:
    requests, _ = reaudit.build_frozen_requests()
    rows = _perfect_rows(requests)
    missing = reaudit.summarize_rows(rows[:-1], requests)
    assert not missing["audit_passed"]
    assert missing["coverage"]["missing_count"] == 1

    duplicate = reaudit.summarize_rows(rows + [dict(rows[0])], requests)
    assert not duplicate["audit_passed"]
    assert duplicate["coverage"]["duplicate_count"] == 1

    changed_prompt = [dict(row) for row in rows]
    changed_prompt[0]["prompt"] += "tampered"
    prompt_summary = reaudit.summarize_rows(changed_prompt, requests)
    assert not prompt_summary["audit_passed"]
    assert prompt_summary["integrity"]["prompt_mismatches"] == 1
    assert prompt_summary["integrity"]["prompt_hash_mismatches"] == 1

    changed_score = [dict(row) for row in rows]
    changed_score[0]["rescore"] = dict(changed_score[0]["rescore"])
    changed_score[0]["rescore"]["semantic_correct"] = False
    score_summary = reaudit.summarize_rows(changed_score, requests)
    assert not score_summary["audit_passed"]
    assert score_summary["integrity"]["rescore_mismatches"] == 1


def test_worker_and_response_envelopes_require_exact_protocol_ids_and_t0() -> None:
    receipt = _ready_receipt("qwen25_7b", "worker-qwen")
    reaudit.validate_worker_receipt(
        receipt, model_key="qwen25_7b", worker_id="worker-qwen"
    )
    wrong_decoding = json.loads(json.dumps(receipt))
    wrong_decoding["decoding"]["temperature"] = 0.7
    with pytest.raises(RuntimeError, match="not greedy T0"):
        reaudit.validate_worker_receipt(
            wrong_decoding, model_key="qwen25_7b", worker_id="worker-qwen"
        )

    envelope = {
        "protocol": MAILBOX_TRANSPORT_PROTOCOL,
        "request_id": "request-1",
        "responses": [],
        "error": None,
    }
    reaudit.validate_response_envelope(envelope, "request-1")
    with pytest.raises(RuntimeError, match="protocol mismatch"):
        reaudit.validate_response_envelope(
            {**envelope, "protocol": reaudit.REAUDIT_PROTOCOL}, "request-1"
        )
    with pytest.raises(RuntimeError, match="request_id mismatch"):
        reaudit.validate_response_envelope(envelope, "request-2")


def test_coordinator_retains_complete_raw_evidence_and_fail_closed_manifest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)
    requests, expected_hashes = reaudit.build_frozen_requests()
    item_by_id = {item.id: item for item in build_probe_bank()}
    answers = [f"ANSWER: {item_by_id[row['item_id']].expected}" for row in requests]

    def fake_wait(path: Path, timeout_seconds: float):
        assert timeout_seconds == 1.0
        payload = {
            "protocol": MAILBOX_TRANSPORT_PROTOCOL,
            "request_id": path.stem,
            "responses": answers,
            "error": None,
        }
        atomic_json(path, payload)
        return payload

    monkeypatch.setattr(reaudit, "wait_for_response", fake_wait)
    assert reaudit.run(args) == 0
    manifest = json.loads(
        (args.output / "run_manifest.json").read_text(encoding="utf-8")
    )
    raw = [
        json.loads(line)
        for line in (args.output / "comprehension_reaudit_raw.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    summary = json.loads(
        (args.output / "domain_condition_summary.json").read_text(encoding="utf-8")
    )
    mailbox = json.loads(
        (args.output / "mailbox_audit.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "completed"
    assert manifest["audit_passed"]
    assert manifest["n_responses"] == manifest["expected_responses"] == 274
    assert manifest["evidence_class"] == (
        "diagnostic_unadmitted_comprehension_reaudit"
    )
    assert manifest["design"]["repetitions"] == 1
    assert manifest["design"]["temperature"] == 0.0
    assert all(manifest[key] == value for key, value in expected_hashes.items())
    assert all(manifest["admission_receipts"][key]["model_digest"] for key in reaudit.MODEL_KEYS)
    assert len(raw) == 274 and summary["audit_passed"] and len(mailbox) == 2
    assert all(row["prompt"] and row["raw_response"] for row in raw)
    assert all(row["rescore"]["semantic_correct"] for row in raw)
    assert all(row["lane_block"] == args.lane_block for row in raw)
    assert all(row["model_digest"].startswith("digest-") for row in raw)


def test_bad_response_envelope_marks_manifest_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = _args(tmp_path)

    def fake_wait(path: Path, timeout_seconds: float):
        payload = {
            "protocol": MAILBOX_TRANSPORT_PROTOCOL,
            "request_id": "wrong-request",
            "responses": [],
            "error": None,
        }
        atomic_json(path, payload)
        return payload

    monkeypatch.setattr(reaudit, "wait_for_response", fake_wait)
    with pytest.raises(RuntimeError, match="request_id mismatch"):
        reaudit.run(args)
    manifest = json.loads(
        (args.output / "run_manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["status"] == "failed"
    assert manifest["audit_passed"] is False
    assert "request_id mismatch" in manifest["error"]
