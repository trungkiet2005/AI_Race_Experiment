from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_race.audit.context_recognition import (
    parse_recognition_response,
    render_recognition_prompt,
    render_rules_only_scenario,
    run_recognition_matrix,
    summarize_recognition,
)
from ai_race.dataio.config_loader import load_game_config
from ai_race.prompts.context_skins import ACTION_CODE_MAPPINGS, SKINS


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def medium_config():
    return load_game_config(
        ROOT / "ai_race" / "configs" / "game" / "ai_race_risk_60.json",
        model="Mock",
    )


def test_rules_only_scenarios_exclude_gameplay_and_comprehension(medium_config):
    hashes = set()
    for skin_id in SKINS:
        for mapping_id in ACTION_CODE_MAPPINGS:
            scenario = render_rules_only_scenario(
                medium_config, skin_id=skin_id, mapping_id=mapping_id
            )
            prompt = render_recognition_prompt(
                medium_config, skin_id=skin_id, mapping_id=mapping_id
            )
            assert "This is the state" not in scenario
            assert "This is the history" not in scenario
            assert "Choose between" not in scenario
            assert "Your goal is to maximize" not in scenario
            assert "ACTION:" not in scenario
            assert "correct answer" not in prompt.casefold()
            assert "calculate a payoff" in prompt
            assert "AI Race" not in prompt
            assert "prisoner's dilemma" not in prompt.casefold()
            hashes.add(prompt)
    assert len(hashes) == len(SKINS) * len(ACTION_CODE_MAPPINGS)


def test_recognition_parser_is_strict_and_candidate_contract_is_closed():
    valid = parse_recognition_response(
        '{"recognition_class":"specific_named_match","candidate":"Named task",'
        '"confidence":"high"}'
    )
    assert valid.strict_valid
    assert valid.candidate == "Named task"
    generic = parse_recognition_response(
        '{"recognition_class":"generic_structural_resemblance","candidate":null,'
        '"confidence":"medium"}'
    )
    assert generic.strict_valid
    assert generic.candidate is None
    assert not parse_recognition_response(
        '{"recognition_class":"generic_structural_resemblance",'
        '"candidate":"named task","confidence":"medium"}'
    ).strict_valid
    assert not parse_recognition_response("```json\n{}\n```").strict_valid
    assert not parse_recognition_response(
        '{"recognition_class":"no_recognition","candidate":"game",'
        '"confidence":"low"}'
    ).strict_valid
    assert not parse_recognition_response(
        '{"recognition_class":"uncertain","candidate":null,"confidence":"low",'
        '"explanation":"extra"}'
    ).strict_valid


class _RetryBackend:
    def __init__(self):
        self.calls: list[tuple[list[str], list[int]]] = []

    def __call__(self, prompts, seeds=None):
        seeds = list(seeds or [])
        self.calls.append((list(prompts), seeds))
        if len(self.calls) == 1:
            return ["not json" for _ in prompts]
        return [
            json.dumps(
                {
                    "recognition_class": "no_recognition",
                    "candidate": None,
                    "confidence": "low",
                },
                separators=(",", ":"),
            )
            for _ in prompts
        ]


def test_matrix_is_complete_deterministic_and_retries_exact_prompt(medium_config):
    backend = _RetryBackend()
    skins = ["technology_race", "abstract_contest"]
    rows = run_recognition_matrix(
        medium_config,
        skins,
        backend,
        repetitions=2,
        seed=91,
        batch_size=99,
        max_parse_retries=1,
    )
    assert len(rows) == len(skins) * len(ACTION_CODE_MAPPINGS) * 2
    assert all(row["strict_valid"] and row["retry_count"] == 1 for row in rows)
    assert all(len(row["attempt_history"]) == 2 for row in rows)
    first_prompts, first_seeds = backend.calls[0]
    retry_prompts, retry_seeds = backend.calls[1]
    assert retry_prompts == first_prompts
    assert retry_seeds != first_seeds
    assert len({(row["skin_id"], row["mapping_id"], row["repetition"]) for row in rows}) == len(rows)

    rerun = run_recognition_matrix(
        medium_config,
        skins,
        lambda prompts, seeds=None: [
            '{"recognition_class":"no_recognition","candidate":null,"confidence":"low"}'
            for _ in prompts
        ],
        repetitions=2,
        seed=91,
    )
    assert [row["sampling_seed"] for row in rerun] == [
        row["sampling_seed"] for row in rows
    ]


def test_summary_keeps_mapping_pair_stability_and_evidence_boundary(medium_config):
    rows = run_recognition_matrix(
        medium_config,
        ["technology_race"],
        lambda prompts, seeds=None: [
            '{"recognition_class":"generic_structural_resemblance",'
            '"candidate":null,"confidence":"medium"}'
            for _ in prompts
        ],
        repetitions=3,
        seed=7,
    )
    summary = summarize_recognition(rows)
    assert summary["overall"]["n"] == 6
    assert summary["overall"]["strict_valid_rate"] == 1.0
    assert summary["mapping_stability"]["n_complete_strict_pairs"] == 3
    assert summary["mapping_stability"]["recognition_class_agreement_rate"] == 1.0
    assert "not evidence" in summary["evidence_boundary"]
