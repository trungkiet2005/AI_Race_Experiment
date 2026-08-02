from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kaggle.interpretability.greennode_context_fast_sae import (
    DISCOVERY_CONTEXT_PAIRS,
    DISCOVERY_SKINS,
    EVALUATION_CONTEXT_PAIRS,
    EVALUATION_SKINS,
    _feature_ranking,
    build_context_pair_examples,
    canonical_action_scores,
    grouped_trajectory_split,
)
from ai_race.prompts.context_skins import SKINS


def _state(game: str, trajectory: str, state: str) -> dict:
    return {
        "game_name": game,
        "trajectory_id": trajectory,
        "state_id": state,
    }


def test_context_pair_split_is_complete_disjoint_and_pairwise():
    assert set(DISCOVERY_SKINS).isdisjoint(EVALUATION_SKINS)
    assert set((*DISCOVERY_SKINS, *EVALUATION_SKINS)) == set(SKINS)
    assert all(len(pair) == 2 for pair in (*DISCOVERY_CONTEXT_PAIRS, *EVALUATION_CONTEXT_PAIRS))


def test_opaque_sequence_scores_decode_canonically_under_both_mappings():
    scores = {"P": -1.0, "Q": -2.0}
    safe_p = canonical_action_scores(scores, "safe_p")
    safe_q = canonical_action_scores(scores, "safe_q")
    assert safe_p["emitted_code"] == safe_q["emitted_code"] == "P"
    assert safe_p["action"] == "safe"
    assert safe_q["action"] == "unsafe"
    assert safe_p["unsafe_log_odds"] == -1.0
    assert safe_q["unsafe_log_odds"] == 1.0
    with pytest.raises(ValueError, match="exactly P and Q"):
        canonical_action_scores({"P": -1.0}, "safe_p")


def test_grouped_split_never_crosses_a_source_trajectory_and_stratifies_games():
    states = []
    for game in ("risk10", "risk60", "risk90"):
        for trajectory in range(5):
            for state_index in range(2):
                states.append(_state(game, f"trajectory-{trajectory}", f"{game}-{trajectory}-{state_index}"))
    split = grouped_trajectory_split(states, eval_fraction=0.4, seed=17)
    assert set(split.values()) == {"discovery", "evaluation"}
    for game in ("risk10", "risk60", "risk90"):
        values = {value for group, value in split.items() if group.startswith(f"{game}|")}
        assert values == {"discovery", "evaluation"}
    assert split == grouped_trajectory_split(states, eval_fraction=0.4, seed=17)


def test_pair_builder_uses_only_requested_state_split_and_exact_matches():
    records = []
    codes = []
    skins = DISCOVERY_CONTEXT_PAIRS[0]
    for state_split, state_id in (("discovery", "state-d"), ("evaluation", "state-e")):
        for skin_index, skin in enumerate(skins):
            for mapping_index, mapping in enumerate(("safe_p", "safe_q")):
                records.append(
                    {
                        "state_id": state_id,
                        "trajectory_group_id": f"g-{state_id}",
                        "state_split": state_split,
                        "skin_id": skin,
                        "mapping_id": mapping,
                        "action": "unsafe" if skin_index else "safe",
                        "unsafe_log_odds": float(skin_index + mapping_index),
                    }
                )
                codes.append(np.asarray([skin_index, mapping_index, 1.0], dtype=np.float32))
    rows, delta = build_context_pair_examples(
        records,
        np.stack(codes),
        [skins],
        required_state_split="discovery",
    )
    assert len(rows) == 2
    assert {row["state_id"] for row in rows} == {"state-d"}
    assert {row["action_flip"] for row in rows} == {1}
    np.testing.assert_array_equal(delta, np.asarray([[1, 0, 0], [1, 0, 0]], dtype=np.float32))


def test_network_volume_guard_is_declared_in_runner_source():
    # Static regression guard: this test is platform-independent on Windows CI.
    source = Path(__file__).parents[2] / "kaggle" / "interpretability" / "greennode_context_fast_sae.py"
    text = source.read_text(encoding="utf-8")
    assert 'startswith("/network-volume")' in text
    assert "no Ollama or historical labels" in text


def test_feature_selection_order_is_frozen_by_discovery_data_only():
    discovery_rows = [
        {"delta_unsafe_log_odds": -1.0, "action_flip": 0},
        {"delta_unsafe_log_odds": 1.0, "action_flip": 1},
        {"delta_unsafe_log_odds": -0.5, "action_flip": 0},
        {"delta_unsafe_log_odds": 0.5, "action_flip": 1},
    ]
    # Feature 0 changes strongly only in flip pairs; feature 1 is less specific.
    discovery_delta = np.asarray(
        [[0.0, -1.0], [4.0, 1.0], [0.0, -0.5], [3.0, 0.5]], dtype=np.float32
    )
    evaluation_rows = [
        {"delta_unsafe_log_odds": -1.0, "action_flip": 0},
        {"delta_unsafe_log_odds": 1.0, "action_flip": 1},
    ]
    first = _feature_ranking(
        discovery_rows,
        discovery_delta,
        evaluation_rows,
        np.asarray([[100.0, 0.0], [-100.0, 0.0]], dtype=np.float32),
        min_prevalence=0.0,
    )
    second = _feature_ranking(
        discovery_rows,
        discovery_delta,
        evaluation_rows,
        np.asarray([[0.0, 100.0], [0.0, -100.0]], dtype=np.float32),
        min_prevalence=0.0,
    )
    assert [row["feature_id"] for row in first] == [row["feature_id"] for row in second]
    assert first[0]["feature_id"] == 0
