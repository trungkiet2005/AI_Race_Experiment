from __future__ import annotations

import hashlib
import json
from pathlib import Path
import copy

import pytest

from ai_race.audit.scaffold_comprehension import (
    MIN_DOMAIN_ACCURACY,
    MIN_STRICT_PARSE_RATE,
    build_scaffold_probe_requests,
    request_bank_sha256,
    run_scaffold_comprehension,
    scaffold_admission_summary,
    scaffold_rules_context,
)
from ai_race.dataio.config_loader import load_game_config
from kaggle.experiments.greennode_scaffold_comprehension import (
    require_completed_resume_match,
    write_admission_artifacts,
)


ROOT = Path(__file__).resolve().parents[2]


def medium_config():
    return load_game_config(
        ROOT / "ai_race/configs/game/ai_race_risk_60.json", model="LocalQwen"
    )


def perfect_rows(*, conditions=("none",), mappings=("safe_p",), repetitions=1):
    requests = build_scaffold_probe_requests(
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=repetitions,
        seed=260726,
    )
    answers = [f"ANSWER: {item['probe'].expected}" for item in requests]
    cursor = 0

    def backend(prompts, *, seeds):
        nonlocal cursor
        assert len(prompts) == len(seeds)
        batch = answers[cursor : cursor + len(prompts)]
        cursor += len(prompts)
        return batch

    rows = run_scaffold_comprehension(requests, backend, batch_size=7)
    assert cursor == len(requests)
    return requests, rows


def test_exact_scaffold_cards_are_present_in_their_admission_contexts() -> None:
    config = medium_config()
    expected = {
        "none": (False, False, False),
        "transition": (True, False, False),
        "terminal": (False, True, False),
        "transition_terminal": (True, True, False),
        "length_placebo": (False, False, True),
    }
    for condition, markers in expected.items():
        context = scaffold_rules_context(
            config, mapping_id="safe_p", condition_id=condition
        )
        assert ("[VERIFIED TRANSITION TOOL RESULT]" in context) is markers[0]
        assert ("[VERIFIED TERMINAL TOOL RESULT]" in context) is markers[1]
        assert ("[LENGTH-MATCHED CONTROL]" in context) is markers[2]
        assert "SAFE" not in context.upper()
        assert "UNSAFE" not in context.upper()


def test_probe_matrix_is_deterministic_complete_and_strictly_scored() -> None:
    conditions = ("none", "transition_terminal")
    mappings = ("safe_p", "safe_q")
    requests, rows = perfect_rows(
        conditions=conditions, mappings=mappings, repetitions=2
    )
    assert len(requests) == len(rows) == 2 * 2 * 2 * 16
    assert request_bank_sha256(requests) == request_bank_sha256(requests)
    assert len({row["sampling_seed"] for row in rows}) == len(rows)
    assert all(row["strict_valid"] and row["semantic_correct"] for row in rows)
    assert all(
        row["prompt_sha256"]
        == hashlib.sha256(row["prompt"].encode("utf-8")).hexdigest()
        for row in rows
    )


def test_admission_freezes_asymmetric_domain_and_strict_thresholds() -> None:
    conditions = ("none",)
    mappings = ("safe_p",)
    _, rows = perfect_rows(
        conditions=conditions, mappings=mappings, repetitions=5
    )
    summary = scaffold_admission_summary(
        rows,
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=5,
    )
    assert summary["passed"]
    assert summary["thresholds"] == {
        "strict_parse_rate": MIN_STRICT_PARSE_RATE,
        "domain_semantic_accuracy": MIN_DOMAIN_ACCURACY,
    }

    state_rows = [row for row in rows if row["domain"] == "state_update"]
    for row in state_rows[:2]:
        row["semantic_correct"] = False
    at_threshold = scaffold_admission_summary(
        rows,
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=5,
    )
    cell = at_threshold["by_cell"]["none/safe_p"]
    assert cell["by_domain"]["state_update"]["semantic_accuracy"] == 0.9
    assert at_threshold["passed"]
    state_rows[2]["semantic_correct"] = False
    below_threshold = scaffold_admission_summary(
        rows,
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=5,
    )
    assert not below_threshold["passed"]

    for row in state_rows[:3]:
        row["semantic_correct"] = True
    for row in rows[:4]:
        row["strict_valid"] = False
    strict_at_threshold = scaffold_admission_summary(
        rows,
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=5,
    )
    assert strict_at_threshold["by_cell"]["none/safe_p"]["strict_parse_rate"] == 0.95
    assert strict_at_threshold["passed"]
    rows[4]["strict_valid"] = False
    strict_below = scaffold_admission_summary(
        rows,
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=5,
    )
    assert not strict_below["passed"]


def test_admission_fails_closed_on_missing_and_duplicate_probe_cells() -> None:
    conditions = ("none",)
    mappings = ("safe_p",)
    _, rows = perfect_rows(conditions=conditions, mappings=mappings, repetitions=1)
    missing = scaffold_admission_summary(
        rows[:-1],
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=1,
    )
    assert not missing["passed"]
    assert missing["coverage"]["missing_count"] == 1
    duplicated = scaffold_admission_summary(
        rows + [dict(rows[0])],
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=1,
    )
    assert not duplicated["passed"]
    assert duplicated["coverage"]["duplicate_count"] == 1


def test_short_backend_batch_is_retained_and_rejected_as_incomplete() -> None:
    conditions = ("none",)
    mappings = ("safe_p",)
    requests = build_scaffold_probe_requests(
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=1,
        seed=260726,
    )

    def short_backend(prompts, *, seeds):
        assert len(prompts) == len(seeds)
        return [f"ANSWER: {requests[0]['probe'].expected}"]

    rows = run_scaffold_comprehension(requests, short_backend, batch_size=16)
    assert len(rows) == 1
    summary = scaffold_admission_summary(
        rows,
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=1,
    )
    assert not summary["passed"]
    assert summary["coverage"]["missing_count"] == 15


def test_artifacts_embed_provenance_and_hash_the_raw_jsonl(tmp_path: Path) -> None:
    conditions = ("none",)
    mappings = ("safe_p",)
    requests, rows = perfect_rows(
        conditions=conditions, mappings=mappings, repetitions=1
    )
    summary = scaffold_admission_summary(
        rows,
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=1,
    )
    provenance = {
        "request_bank_sha256": request_bank_sha256(requests),
        "behavior_source_sha256": "source-hash",
        "behavior_experiment_config_sha256": "config-hash",
        "model": {"digest": "abc"},
        "behavior_decoding": {
            "temperature": 0.0,
            "max_tokens": 16,
            "workers": 16,
            "seed_requested": True,
            "seed_probe_exact_match": True,
        },
    }
    raw_path, admission_path, admission = write_admission_artifacts(
        tmp_path, rows, summary, provenance
    )
    assert raw_path.name == "comprehension_raw.jsonl"
    assert admission_path.name == "admission.json"
    recorded = json.loads(admission_path.read_text(encoding="utf-8"))
    assert recorded["provenance"] == provenance
    assert recorded["artifacts"]["comprehension_raw"]["sha256"] == hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()
    assert admission["passed"]


def test_resume_revalidates_provenance_and_both_artifact_hashes(tmp_path: Path) -> None:
    conditions = ("none",)
    mappings = ("safe_p",)
    requests, rows = perfect_rows(
        conditions=conditions, mappings=mappings, repetitions=1
    )
    summary = scaffold_admission_summary(
        rows,
        medium_config(),
        condition_ids=conditions,
        mapping_ids=mappings,
        repetitions=1,
    )
    expected = {
        "profile": "smoke",
        "repetitions": 1,
        "condition_ids": list(conditions),
        "mapping_ids": list(mappings),
        "admission_source_sha256": "admission-source",
        "behavior_source_sha256": "behavior-source",
        "behavior_experiment_config_sha256": "config-hash",
        "request_bank_sha256": request_bank_sha256(requests),
        "model": {"name": "model", "digest": "model-digest"},
        "decoding": {
            "temperature": 0.0,
            "max_tokens": 16,
            "workers": 16,
            "batch_size": 128,
            "seed_requested": True,
        },
        "behavior_decoding": {
            "temperature": 0.0,
            "max_tokens": 16,
            "workers": 16,
            "seed_requested": True,
            "seed_probe_exact_match": True,
        },
        "expected_requests": 16,
    }
    recorded = copy.deepcopy(expected)
    recorded["decoding"]["seed_probe_exact_match"] = True
    raw_path, admission_path, admission = write_admission_artifacts(
        tmp_path, rows, summary, recorded
    )

    def receipt(path: Path) -> dict:
        return {
            "path": path.relative_to(tmp_path).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }

    previous = {
        "status": "completed",
        "protocol": admission["protocol"],
        **recorded,
        "admission_passed": True,
        "artifacts": {
            "comprehension_raw": receipt(raw_path),
            "admission": receipt(admission_path),
        },
    }
    assert require_completed_resume_match(tmp_path, previous, expected)["passed"]

    wrong_model = copy.deepcopy(previous)
    wrong_model["model"]["digest"] = "other-digest"
    with pytest.raises(RuntimeError, match="mismatched provenance"):
        require_completed_resume_match(tmp_path, wrong_model, expected)

    raw_path.write_text(raw_path.read_text(encoding="utf-8") + "{}\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hashes"):
        require_completed_resume_match(tmp_path, previous, expected)
