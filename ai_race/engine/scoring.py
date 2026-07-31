"""Pure scoring functions for the idealised two-player AI Race."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .state import Action, GameConfig


@dataclass(frozen=True)
class TerminalScoring:
    outcomes: list[str]
    prizes: list[float]
    private_risks: list[float]
    setback_eligible: list[bool]
    setback_draws: list[float]
    setbacks: list[bool]
    final_payoffs: list[float]


def unsafe_fraction(unsafe_count: int, rounds_played: int) -> float:
    if rounds_played <= 0:
        return 0.0
    return float(unsafe_count) / float(rounds_played)


def effective_private_risk(
    max_private_risk: float,
    unsafe_count: int,
    rounds_played: int,
) -> float:
    """Risk accumulated by one player before winner eligibility is known."""
    risk = float(max_private_risk) * unsafe_fraction(unsafe_count, rounds_played)
    return min(1.0, max(0.0, risk))


def joint_round_outcome(
    config: GameConfig,
    actions: Sequence[Action | str],
) -> tuple[list[float], list[float]]:
    """Return stage payoffs and progress increments from simultaneous actions."""
    if len(actions) != 2:
        raise ValueError("AI Race round scoring requires exactly two actions")
    normalized = [Action.coerce(action) for action in actions]
    payoffs = [
        config.stage_payoff(normalized[0], normalized[1]),
        config.stage_payoff(normalized[1], normalized[0]),
    ]
    increments = [config.progress_for(action) for action in normalized]
    return payoffs, increments


def race_outcomes(progress: Sequence[float], *, tolerance: float = 1e-9) -> list[str]:
    if len(progress) != 2:
        raise ValueError("AI Race terminal scoring requires exactly two players")
    if not all(math.isfinite(float(value)) for value in progress):
        raise ValueError("AI Race progress values must be finite")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("AI Race tie tolerance must be finite and non-negative")
    if abs(float(progress[0]) - float(progress[1])) <= tolerance:
        return ["tie", "tie"]
    if progress[0] > progress[1]:
        return ["winner", "loser"]
    return ["loser", "winner"]


def terminal_scoring(
    *,
    stage_payoffs: Sequence[float],
    progress: Sequence[float],
    unsafe_counts: Sequence[int],
    rounds_played: int,
    max_private_risk: float,
    race_prize: float,
    setback_draws: Sequence[float],
) -> TerminalScoring:
    """Resolve prize, private setback, and final task payoff.

    A deterministic draw is supplied for both seats even when a seat is not eligible.
    This keeps RNG streams aligned across treatments. The draw only affects a winner
    or tied winner.
    """
    lengths = {
        len(stage_payoffs),
        len(progress),
        len(unsafe_counts),
        len(setback_draws),
    }
    if lengths != {2}:
        raise ValueError("Terminal scoring expects two values for every player field")
    if rounds_played <= 0:
        raise ValueError("Terminal scoring requires at least one completed round")
    if not math.isfinite(float(max_private_risk)) or not 0 <= max_private_risk <= 1:
        raise ValueError("Terminal maximum private risk must be finite and in [0, 1]")
    if not math.isfinite(float(race_prize)) or race_prize < 0:
        raise ValueError("Terminal race prize must be finite and non-negative")
    if not all(math.isfinite(float(value)) for value in (*stage_payoffs, *progress)):
        raise ValueError("Terminal payoff and progress values must be finite")
    if any(
        isinstance(count, bool)
        or not isinstance(count, int)
        or not 0 <= count <= rounds_played
        for count in unsafe_counts
    ):
        raise ValueError("Unsafe counts must be integers in [0, rounds_played]")
    if not all(
        math.isfinite(float(value)) and 0 <= float(value) <= 1
        for value in setback_draws
    ):
        raise ValueError("Setback draws must be finite and in [0, 1]")

    outcomes = race_outcomes(progress)
    tie = outcomes == ["tie", "tie"]
    prizes = [
        float(race_prize) / 2.0 if tie else (float(race_prize) if outcome == "winner" else 0.0)
        for outcome in outcomes
    ]
    risks = [
        effective_private_risk(max_private_risk, count, rounds_played)
        for count in unsafe_counts
    ]
    eligible = [outcome in {"winner", "tie"} for outcome in outcomes]
    draws = [float(value) for value in setback_draws]
    setbacks = [
        bool(is_eligible and draw < risk)
        for is_eligible, draw, risk in zip(eligible, draws, risks)
    ]
    final_payoffs = [
        0.0 if setback else float(stage) + prize
        for stage, prize, setback in zip(stage_payoffs, prizes, setbacks)
    ]
    return TerminalScoring(
        outcomes=outcomes,
        prizes=prizes,
        private_risks=risks,
        setback_eligible=eligible,
        setback_draws=draws,
        setbacks=setbacks,
        final_payoffs=final_payoffs,
    )
