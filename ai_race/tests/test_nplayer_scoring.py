"""Formula and outcome tests for the N-player engine.

The load-bearing test here is ``test_n2_matches_two_player_engine``: the
N-player payoff formula (Appendix B of the reference paper, minus p_fo) must
reduce, at n_players=2 with cost=1/benefit=4/speed=1.5, to exactly the
shipped two-player payoff matrix -- that identity is what justifies calling
this a generalisation of the existing mechanism rather than a different one.
"""
from __future__ import annotations

import pytest

from ai_race.engine.state import Action as TwoPlayerAction
from ai_race.engine.state import GameConfig as TwoPlayerGameConfig
from ai_race.engine_nplayer.scoring import (
    effective_private_risk,
    joint_round_outcome,
    race_outcomes,
    terminal_scoring,
)
from ai_race.engine_nplayer.state import Action, NPlayerGameConfig


def _config(n_players: int = 3, **overrides) -> NPlayerGameConfig:
    defaults = dict(
        name="test",
        n_players=n_players,
        safe_progress=1.0,
        speed=1.5,
        cost=1.0,
        benefit=4.0,
    )
    defaults.update(overrides)
    return NPlayerGameConfig(**defaults)


def test_n2_matches_two_player_engine():
    n_config = _config(n_players=2)
    two_player = TwoPlayerGameConfig(name="two")
    for own, opponent in (
        (Action.SAFE, Action.SAFE),
        (Action.SAFE, Action.UNSAFE),
        (Action.UNSAFE, Action.SAFE),
        (Action.UNSAFE, Action.UNSAFE),
    ):
        k_safe = int(own is Action.SAFE) + int(opponent is Action.SAFE)
        got = n_config.stage_payoff_for(own, k_safe)
        want = two_player.stage_payoff(
            TwoPlayerAction(own.value), TwoPlayerAction(opponent.value)
        )
        assert got == pytest.approx(want), (own, opponent)


def test_n2_progress_matches_two_player_engine():
    n_config = _config(n_players=2)
    two_player = TwoPlayerGameConfig(name="two")
    assert n_config.safe_progress == two_player.safe_progress
    assert n_config.unsafe_progress() == pytest.approx(two_player.unsafe_progress)


@pytest.mark.parametrize(
    "k_safe, expected_safe, expected_unsafe",
    [
        # N=3, cost=1, benefit=4, speed=1.5. share_weight = k + 1.5*(3-k).
        (0, None, 4.0 / 3.0),  # weight = 4.5
        (1, -1.0 + 4.0 / 4.0, 1.5 * 4.0 / 4.0),  # weight = 4.0
        (2, -1.0 + 4.0 / 3.5, 1.5 * 4.0 / 3.5),  # weight = 3.5
        (3, -1.0 + 4.0 / 3.0, None),  # everyone Safe: split evenly, no weighting
    ],
)
def test_stage_payoff_hand_computed_for_n3(k_safe, expected_safe, expected_unsafe):
    config = _config(n_players=3)
    if expected_safe is not None:
        assert config.stage_payoff_safe(k_safe) == pytest.approx(expected_safe)
    if expected_unsafe is not None:
        assert config.stage_payoff_unsafe(k_safe) == pytest.approx(expected_unsafe)


def test_stage_payoff_safe_rejects_out_of_range_k():
    config = _config(n_players=3)
    with pytest.raises(ValueError):
        config.stage_payoff_safe(0)
    with pytest.raises(ValueError):
        config.stage_payoff_safe(4)


def test_stage_payoff_unsafe_rejects_out_of_range_k():
    config = _config(n_players=3)
    with pytest.raises(ValueError):
        config.stage_payoff_unsafe(-1)
    with pytest.raises(ValueError):
        config.stage_payoff_unsafe(3)


def test_joint_round_outcome_requires_n_actions():
    config = _config(n_players=3)
    with pytest.raises(ValueError, match="exactly 3 actions"):
        joint_round_outcome(config, [Action.SAFE, Action.SAFE])


def test_joint_round_outcome_payoffs_sum_matches_manual_k():
    config = _config(n_players=4)
    actions = [Action.SAFE, Action.UNSAFE, Action.UNSAFE, Action.SAFE]
    payoffs, increments = joint_round_outcome(config, actions)
    # k_safe = 2 (indices 0 and 3)
    assert payoffs[0] == pytest.approx(config.stage_payoff_safe(2))
    assert payoffs[3] == pytest.approx(config.stage_payoff_safe(2))
    assert payoffs[1] == pytest.approx(config.stage_payoff_unsafe(2))
    assert payoffs[2] == pytest.approx(config.stage_payoff_unsafe(2))
    assert increments == [1.0, 1.5, 1.5, 1.0]


def test_race_outcomes_solo_winner():
    assert race_outcomes([10.0, 5.0, 3.0, 1.0]) == ["winner", "loser", "loser", "loser"]


def test_race_outcomes_partial_tie():
    assert race_outcomes([10.0, 10.0, 5.0, 3.0]) == ["tie", "tie", "loser", "loser"]


def test_race_outcomes_full_tie():
    assert race_outcomes([7.0, 7.0, 7.0]) == ["tie", "tie", "tie"]


def test_race_outcomes_requires_at_least_two_players():
    with pytest.raises(ValueError, match="at least two players"):
        race_outcomes([1.0])


def test_terminal_scoring_prize_split_among_tied_leaders():
    result = terminal_scoring(
        stage_payoffs=[1.0, 1.0, 1.0],
        progress=[10.0, 10.0, 5.0],
        unsafe_counts=[0, 0, 0],
        rounds_played=5,
        max_private_risk=0.5,
        race_prize=90.0,
        setback_draws=[1.0, 1.0, 1.0],
    )
    assert result.outcomes == ["tie", "tie", "loser"]
    assert result.prizes == pytest.approx([45.0, 45.0, 0.0])
    assert result.setback_eligible == [True, True, False]


def test_terminal_scoring_prize_undivided_for_solo_winner():
    result = terminal_scoring(
        stage_payoffs=[1.0, 1.0, 1.0],
        progress=[10.0, 5.0, 3.0],
        unsafe_counts=[3, 0, 0],
        rounds_played=3,
        max_private_risk=0.5,
        race_prize=90.0,
        setback_draws=[1.0, 1.0, 1.0],
    )
    assert result.outcomes == ["winner", "loser", "loser"]
    assert result.prizes == pytest.approx([90.0, 0.0, 0.0])
    assert result.private_risks[0] == pytest.approx(0.5)


def test_effective_private_risk_matches_fraction_times_max():
    assert effective_private_risk(0.6, unsafe_count=3, rounds_played=6) == pytest.approx(0.3)
    assert effective_private_risk(0.6, unsafe_count=0, rounds_played=6) == 0.0
    assert effective_private_risk(0.6, unsafe_count=0, rounds_played=0) == 0.0
