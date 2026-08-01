"""Tests for strategy_analysis.classify_nplayer.

Cross-checks CS's predicted trajectory against
ai_race.engine_nplayer.strategies.strategy_trajectory directly, since that is
the ground truth for "what CS actually does" in the N-player engine.
"""
from __future__ import annotations

import pytest

from ai_race.engine_nplayer.strategies import CanonicalStrategy, strategy_trajectory
from strategy_analysis.classify_nplayer import (
    CANONICAL_STRATEGIES_NPLAYER,
    classify_trajectory_nplayer,
    predict_strategy_nplayer,
)


def test_predict_as_au_ignore_co_players():
    others = [(0, 1, 1, 0), (1, 1, 0, 0)]
    assert predict_strategy_nplayer("AS", others) == (0, 0, 0, 0)
    assert predict_strategy_nplayer("AU", others) == (1, 1, 1, 1)


def test_predict_cs_matches_engine_strategy_trajectory():
    # Two co-players with a mixed Safe/Unsafe history; CS should be SAFE only
    # when BOTH were Safe the previous round.
    others_actions = [
        ["safe", "safe", "unsafe", "safe", "safe"],
        ["safe", "unsafe", "unsafe", "safe", "unsafe"],
    ]
    others_ints = [tuple(0 if a == "safe" else 1 for a in seq) for seq in others_actions]

    predicted = predict_strategy_nplayer("CS", others_ints)
    expected = tuple(
        action.value == "unsafe"
        for action in strategy_trajectory(CanonicalStrategy.CS, others_actions)
    )
    expected_ints = tuple(int(v) for v in expected)
    assert predicted == expected_ints


def test_predict_cs_round1_is_always_safe():
    others = [(1, 1, 1), (1, 1, 1)]
    assert predict_strategy_nplayer("CS", others)[0] == 0


def test_classify_perfect_as_trajectory():
    own = (0, 0, 0, 0)
    others = [(1, 1, 0, 1), (0, 1, 1, 1)]
    result = classify_trajectory_nplayer(own, others)
    assert result.best_strategies == ("AS",)
    assert result.unique_best_strategy == "AS"


def test_classify_perfect_au_trajectory():
    own = (1, 1, 1, 1)
    others = [(0, 0, 0, 0), (1, 0, 1, 0)]
    result = classify_trajectory_nplayer(own, others)
    assert result.best_strategies == ("AU",)


def test_classify_perfect_cs_trajectory():
    others = [(0, 1, 0), (0, 0, 1)]
    predicted_cs = predict_strategy_nplayer("CS", others)
    result = classify_trajectory_nplayer(predicted_cs, others)
    assert "CS" in result.best_strategies


def test_classify_retains_ties():
    # All-Safe co-players the whole race: AS, CS all predict all-Safe.
    others = [(0, 0, 0), (0, 0, 0)]
    own = (0, 0, 0)
    result = classify_trajectory_nplayer(own, others)
    assert set(result.best_strategies) >= {"AS", "CS"}
    assert result.unique_best_strategy is None


def test_classify_rejects_length_mismatch():
    with pytest.raises(ValueError):
        classify_trajectory_nplayer((0, 0), [(0, 0, 0)])


def test_classify_rejects_empty_trajectory():
    with pytest.raises(ValueError):
        classify_trajectory_nplayer((), [()])


def test_canonical_strategies_nplayer_has_no_cas():
    assert "CAS" not in CANONICAL_STRATEGIES_NPLAYER
    assert set(CANONICAL_STRATEGIES_NPLAYER) == {"AS", "AU", "CS"}


def test_predict_strategy_rejects_unknown_name():
    with pytest.raises(ValueError):
        predict_strategy_nplayer("CAS", [(0, 1)])


def test_predict_strategy_rejects_mismatched_co_player_lengths():
    with pytest.raises(ValueError):
        predict_strategy_nplayer("CS", [(0, 1), (0, 1, 1)])


def test_behind_unsafe_exploratory_requires_gaps():
    others = [(0, 0), (0, 0)]
    with pytest.raises(ValueError):
        predict_strategy_nplayer("BEHIND_UNSAFE_EXPLORATORY", others)


def test_behind_unsafe_exploratory_matches_negative_gap():
    others = [(0, 0), (0, 0)]
    gaps = [-1.0, 2.0]
    predicted = predict_strategy_nplayer(
        "BEHIND_UNSAFE_EXPLORATORY", others, progress_gaps_before=gaps
    )
    assert predicted == (1, 0)
