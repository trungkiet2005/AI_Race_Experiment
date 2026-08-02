from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_race.audit.state_scaffold import add_state_scaffold
from kaggle.experiments.greennode_context_skin import (
    build_fully_crossed_context_games,
)
from kaggle.experiments.greennode_state_scaffold import (
    LANE_CONDITIONS,
    LANE_REP_PARITY,
)


ROOT = Path(__file__).resolve().parents[2]


def experiment() -> dict:
    payload = json.loads(
        (ROOT / "ai_race/configs/experiment/state_scaffold_factorial.json").read_text(
            encoding="utf-8"
        )
    )
    payload["repetitions"] = 1
    return payload


def test_every_scaffold_condition_runs_on_both_lane_shards() -> None:
    assert set(LANE_CONDITIONS["a"]) == set(LANE_CONDITIONS["b"])
    assert set(LANE_REP_PARITY.values()) == {0, 1}


@pytest.mark.parametrize(
    ("condition", "transition_expected", "terminal_expected"),
    [
        ("none", False, False),
        ("transition", True, False),
        ("terminal", False, True),
        ("transition_terminal", True, True),
        ("length_placebo", False, False),
    ],
)
def test_scaffold_cards_are_factorial_and_keep_opaque_contract(
    condition: str, transition_expected: bool, terminal_expected: bool
) -> None:
    base = build_fully_crossed_context_games(
        experiment(), "LocalQwen", "abstract_contest"
    )
    games = add_state_scaffold(base, condition)
    assert len(games) == 6  # three risk levels x both mappings
    prompt = games[0].build_round_prompts()[0]
    assert ("[VERIFIED TRANSITION TOOL RESULT]" in prompt) is transition_expected
    assert ("[VERIFIED TERMINAL TOOL RESULT]" in prompt) is terminal_expected
    assert "ACTION: P" in prompt and "ACTION: Q" in prompt
    assert "SAFE" not in prompt.upper()
    assert "UNSAFE" not in prompt.upper()
    assert prompt.rstrip().endswith("ACTION: P or ACTION: Q.") or prompt.rstrip().endswith(
        "ACTION: Q or ACTION: P."
    )


def test_scaffold_rows_are_computed_from_live_state() -> None:
    base = build_fully_crossed_context_games(
        experiment(), "LocalQwen", "abstract_contest"
    )
    game = add_state_scaffold([base[0]], "transition_terminal")[0]
    prompt = game.build_round_prompts()[0]
    assert prompt.count("- You ") == 8
    assert "expected final payoff=" in prompt
    game.apply_round_responses(["ACTION: SAFE", "ACTION: UNSAFE"])
    next_prompt = game.build_round_prompts()[0]
    assert next_prompt != prompt
    assert "private risk becomes" in next_prompt


def test_placebo_matches_joint_tool_character_length() -> None:
    base = build_fully_crossed_context_games(
        experiment(), "LocalQwen", "abstract_contest"
    )
    joint = add_state_scaffold([base[0]], "transition_terminal")[0]
    placebo = add_state_scaffold([base[0]], "length_placebo")[0]
    joint_prompt = joint.build_round_prompts()[0]
    placebo_prompt = placebo.build_round_prompts()[0]
    joint_block = joint_prompt[joint_prompt.index("[VERIFIED TRANSITION TOOL RESULT]") :]
    joint_block = joint_block.split("\n\n[FINAL RESPONSE CONTRACT]", 1)[0]
    placebo_block = placebo_prompt[placebo_prompt.index("[LENGTH-MATCHED CONTROL]") :]
    placebo_block = placebo_block.split("\n\n[FINAL RESPONSE CONTRACT]", 1)[0]
    assert len(placebo_block) == len(joint_block)
