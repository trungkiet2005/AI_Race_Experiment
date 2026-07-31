"""Pure tests for the paper's stage, progress, risk, and terminal rules."""
from __future__ import annotations

import pytest

from ai_race.engine.scoring import (
    effective_private_risk,
    joint_round_outcome,
    terminal_scoring,
    unsafe_fraction,
)
from ai_race.engine.state import Action, GameConfig


@pytest.fixture
def config() -> GameConfig:
    return GameConfig(name="matrix", max_private_risk=0.6)


@pytest.mark.parametrize(
    ("actions", "payoffs", "increments"),
    [
        ((Action.SAFE, Action.SAFE), [1.0, 1.0], [1.0, 1.0]),
        ((Action.SAFE, Action.UNSAFE), [0.6, 2.4], [1.0, 1.5]),
        ((Action.UNSAFE, Action.SAFE), [2.4, 0.6], [1.5, 1.0]),
        ((Action.UNSAFE, Action.UNSAFE), [2.0, 2.0], [1.5, 1.5]),
    ],
)
def test_joint_round_outcome_covers_the_full_stage_matrix(
    config: GameConfig,
    actions: tuple[Action, Action],
    payoffs: list[float],
    increments: list[float],
) -> None:
    assert joint_round_outcome(config, actions) == (payoffs, increments)


def test_joint_round_outcome_requires_exactly_two_actions(config: GameConfig) -> None:
    with pytest.raises(ValueError, match="exactly two"):
        joint_round_outcome(config, [Action.SAFE])


@pytest.mark.parametrize(
    ("maximum", "unsafe_count", "rounds", "expected"),
    [
        (0.6, 0, 5, 0.0),
        (0.6, 2, 4, 0.3),
        (0.6, 4, 4, 0.6),
        (0.6, 4, 0, 0.0),
        (2.0, 4, 4, 1.0),
        (-0.5, 4, 4, 0.0),
    ],
)
def test_effective_private_risk_is_frequency_scaled_and_bounded(
    maximum: float,
    unsafe_count: int,
    rounds: int,
    expected: float,
) -> None:
    assert effective_private_risk(maximum, unsafe_count, rounds) == pytest.approx(expected)


def test_unsafe_fraction_handles_an_empty_history() -> None:
    assert unsafe_fraction(3, 0) == 0.0
    assert unsafe_fraction(3, 4) == pytest.approx(0.75)


def test_winner_keeps_stage_payoff_and_full_prize_without_setback() -> None:
    scored = terminal_scoring(
        stage_payoffs=[10.0, 8.0],
        progress=[6.0, 5.0],
        unsafe_counts=[2, 5],
        rounds_played=5,
        max_private_risk=0.6,
        race_prize=100.0,
        setback_draws=[0.5, 0.0],
    )
    assert scored.outcomes == ["winner", "loser"]
    assert scored.prizes == [100.0, 0.0]
    assert scored.private_risks == pytest.approx([0.24, 0.6])
    assert scored.setback_eligible == [True, False]
    assert scored.setbacks == [False, False]
    assert scored.final_payoffs == [110.0, 8.0]


def test_winner_setback_erases_both_stage_payoff_and_prize() -> None:
    scored = terminal_scoring(
        stage_payoffs=[10.0, 8.0],
        progress=[6.0, 5.0],
        unsafe_counts=[2, 5],
        rounds_played=5,
        max_private_risk=0.6,
        race_prize=100.0,
        setback_draws=[0.1, 0.0],
    )
    assert scored.setbacks == [True, False]
    assert scored.final_payoffs == [0.0, 8.0]


def test_tied_players_split_prize_and_resolve_setbacks_independently() -> None:
    scored = terminal_scoring(
        stage_payoffs=[10.0, 8.0],
        progress=[5.0, 5.0],
        unsafe_counts=[5, 0],
        rounds_played=5,
        max_private_risk=0.6,
        race_prize=100.0,
        setback_draws=[0.5, 0.0],
    )
    assert scored.outcomes == ["tie", "tie"]
    assert scored.prizes == [50.0, 50.0]
    assert scored.setback_eligible == [True, True]
    assert scored.setbacks == [True, False]
    assert scored.final_payoffs == [0.0, 58.0]


def test_loser_is_never_set_back_even_when_draw_is_below_private_risk() -> None:
    scored = terminal_scoring(
        stage_payoffs=[3.0, 4.0],
        progress=[4.0, 5.0],
        unsafe_counts=[5, 0],
        rounds_played=5,
        max_private_risk=0.9,
        race_prize=100.0,
        setback_draws=[0.0, 0.5],
    )
    assert scored.setback_eligible == [False, True]
    assert scored.setbacks[0] is False
    assert scored.final_payoffs[0] == 3.0


@pytest.mark.parametrize(
    "overrides",
    [
        {"safe_progress": float("nan")},
        {"unsafe_progress": float("inf")},
        {"payoff_safe_safe": float("nan")},
        {"payoff_safe_unsafe": float("-inf")},
        {"race_prize": float("nan")},
    ],
)
def test_game_config_rejects_nonfinite_mechanism_values(overrides) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        GameConfig(name="invalid", **overrides)


@pytest.mark.parametrize("progress", [[float("nan"), 1.0], [1.0, float("inf")]])
def test_terminal_outcome_rejects_nonfinite_progress(progress) -> None:
    from ai_race.engine.scoring import race_outcomes

    with pytest.raises(ValueError, match="progress values must be finite"):
        race_outcomes(progress)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("rounds_played", 0, "at least one"),
        ("unsafe_counts", [6, 0], "Unsafe counts"),
        ("setback_draws", [-0.1, 0.2], "Setback draws"),
        ("max_private_risk", float("nan"), "maximum private risk"),
    ],
)
def test_terminal_scoring_rejects_impossible_inputs(field, value, message) -> None:
    inputs = {
        "stage_payoffs": [2.0, 2.0],
        "progress": [5.0, 5.0],
        "unsafe_counts": [1, 1],
        "rounds_played": 5,
        "max_private_risk": 0.6,
        "race_prize": 100.0,
        "setback_draws": [0.2, 0.2],
    }
    inputs[field] = value
    with pytest.raises(ValueError, match=message):
        terminal_scoring(**inputs)
