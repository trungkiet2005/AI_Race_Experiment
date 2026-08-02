"""Acceptance tests for context-clean comprehension admission v3."""
from __future__ import annotations

from dataclasses import asdict

import pytest

from ai_race.audit.game_understanding import build_probe_bank, score_probe_response
from kaggle.experiments import greennode_comprehension_admission_v3 as v3


def _perfect_rows(requests):
    items = {item.id: item for item in build_probe_bank()}
    return [
        {
            **request,
            "model_key": model,
            "raw_response": f"ANSWER: {items[request['item_id']].expected}",
            "rescore": asdict(score_probe_response(items[request["item_id"]], f"ANSWER: {items[request['item_id']].expected}")),
        }
        for model in v3.MODEL_KEYS
        for request in requests
    ]


def test_clean_context_has_invariant_rules_and_no_round1_state_leak() -> None:
    context = v3.invariant_rules_context()
    assert "Accumulated stage payoff means the sum" in context
    assert "draw is strictly less than" in context
    assert "equality does not cause a setback" in context
    assert "complete final payoff zero" in context
    assert all(marker not in context for marker in v3._FORBIDDEN_CLEAN_MARKERS)


def test_frozen_bank_is_complete_crossed_and_anatomically_clean() -> None:
    requests, hashes = v3.build_frozen_requests()
    assert len(requests) == v3.EXPECTED_REQUESTS_PER_MODEL == 274
    assert len({(r["context_condition"], r["item_id"], r["condition"]) for r in requests}) == 274
    assert sum(r["condition"] == "calculator" for r in requests) == 82
    assert {r["context_condition"] for r in requests} == set(v3.CONTEXT_CONDITIONS)
    for request in requests:
        v3.validate_prompt_anatomy(request["prompt"], context_condition=request["context_condition"])
        if request["context_condition"] == "clean_rules_only":
            assert all(marker not in request["prompt"] for marker in v3._FORBIDDEN_CLEAN_MARKERS)
    assert hashes == v3.build_frozen_requests()[1]


def test_winner_setback_probe_is_self_contained() -> None:
    item = next(item for item in build_probe_bank() if item.id == "terminal_winner_setback")
    for condition in ("direct", "paraphrase", "calculator"):
        prompt = v3.render_admission_prompt(item, condition, context_condition="clean_rules_only")
        assert "scenario above" not in prompt.lower()
        assert "Accumulated stage payoffs are you=10" in prompt
        assert "Your setback draw is 0.10" in prompt


def test_summary_uses_only_clean_unaided_item_balanced_rows_for_gate() -> None:
    requests, _ = v3.build_frozen_requests()
    rows = _perfect_rows(requests)
    # Make every calculator and every conflicting-scaffold response wrong.
    for row in rows:
        if row["condition"] == "calculator" or row["context_condition"] == "conflicting_round1_scaffold":
            row["raw_response"] = "ANSWER: impossible"
            item = next(i for i in build_probe_bank() if i.id == row["item_id"])
            row["rescore"] = asdict(score_probe_response(item, row["raw_response"]))
    summary = v3.summarize_rows(rows, requests)
    assert summary["audit_passed"]
    assert summary["all_models_admitted"]
    for model in v3.MODEL_KEYS:
        result = summary["by_model"][model]
        assert result["admission"]["passed"]
        assert result["admission"]["calculator_excluded"]
        assert result["contexts"]["clean_rules_only"]["unaided"]["n_items"] == 41
        assert result["contexts"]["clean_rules_only"]["calculator_tool_uptake"]["accuracy"] == 0.0
        assert result["contexts"]["conflicting_round1_scaffold"]["unaided"]["accuracy"] == 0.0


def test_domain_threshold_is_fail_closed_and_item_balanced() -> None:
    requests, _ = v3.build_frozen_requests()
    rows = _perfect_rows(requests)
    items = {item.id: item for item in build_probe_bank()}
    for row in rows:
        if row["model_key"] == "qwen25_7b" and row["context_condition"] == "clean_rules_only" and row["measurement_class"] == "unaided_understanding" and row["domain"] == "expected_payoff":
            row["raw_response"] = "ANSWER: impossible"
            row["rescore"] = asdict(score_probe_response(items[row["item_id"]], row["raw_response"]))
    summary = v3.summarize_rows(rows, requests)
    assert summary["audit_passed"]
    assert not summary["by_model"]["qwen25_7b"]["admission"]["passed"]
    assert summary["by_model"]["mistral7_01"]["admission"]["passed"]


def test_anatomy_and_integrity_tampering_fail_closed() -> None:
    requests, _ = v3.build_frozen_requests()
    rows = _perfect_rows(requests)
    target = next(row for row in rows if row["context_condition"] == "clean_rules_only")
    target["prompt"] += "\nThis is the state of the race before your current decision:"
    summary = v3.summarize_rows(rows, requests)
    assert not summary["audit_passed"]
    assert summary["integrity"]["prompt_mismatches"] == 1
    assert summary["integrity"]["prompt_hash_mismatches"] == 1
    assert summary["integrity"]["prompt_anatomy_failures"] == 1


def test_unknown_context_and_duplicate_question_are_rejected() -> None:
    item = build_probe_bank()[0]
    with pytest.raises(ValueError, match="unknown context"):
        v3.render_admission_prompt(item, "calculator", context_condition="unknown")
    prompt = v3.render_admission_prompt(item, "calculator", context_condition="clean_rules_only")
    with pytest.raises(ValueError, match="exactly one question"):
        v3.validate_prompt_anatomy(prompt + "QUESTION: second", context_condition="clean_rules_only")
