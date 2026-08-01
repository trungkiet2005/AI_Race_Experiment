"""Pure scoring functions for the N-player AI Race.

Generalises ``ai_race.engine.scoring`` from exactly two players to any
``n_players >= 2``, following the group payoff of Appendix B in the reference
paper (``references/papers/sources/jair-12225/...md``, "N-player AI Race Definition") with the
found-out term ``p_fo`` dropped. Race outcome and terminal scoring generalise
by comparing each player's progress against the group maximum instead of
against the one other player.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

from .state import Action, NPlayerGameConfig


@dataclass(frozen=True)
class NPlayerTerminalScoring:
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
    """Risk accumulated by one player before winner eligibility is known.

    Identical formula to ``ai_race.engine.scoring.effective_private_risk`` --
    it never depended on the number of players, only on one player's own
    trajectory, so it is duplicated here rather than imported to keep this
    package independent of ``ai_race.engine``.
    """
    risk = float(max_private_risk) * unsafe_fraction(unsafe_count, rounds_played)
    return min(1.0, max(0.0, risk))


def joint_round_outcome(
    config: NPlayerGameConfig,
    actions: Sequence[Action | str],
) -> tuple[list[float], list[float]]:
    """Return stage payoffs and progress increments from simultaneous actions.

    Payoff depends only on ``k`` -- how many of the ``n_players`` chose SAFE
    this round -- and the asking player's own action, per
    :meth:`NPlayerGameConfig.stage_payoff_for`. This is the group
    generalisation of the two-player 2x2 payoff-matrix lookup.
    """
    if len(actions) != config.n_players:
        raise ValueError(
            f"AI Race round scoring requires exactly {config.n_players} actions, "
            f"got {len(actions)}"
        )
    normalized = [Action.coerce(action) for action in actions]
    k_safe = sum(1 for action in normalized if action is Action.SAFE)
    payoffs = [config.stage_payoff_for(action, k_safe) for action in normalized]
    increments = [config.progress_for(action) for action in normalized]
    return payoffs, increments


def race_outcomes(progress: Sequence[float], *, tolerance: float = 1e-9) -> list[str]:
    """``"winner"`` for the sole leader, ``"tie"`` for every player sharing the
    max, ``"loser"`` otherwise. Reduces to the two-player rule when len == 2.
    """
    if len(progress) < 2:
        raise ValueError("AI Race terminal scoring requires at least two players")
    values = [float(value) for value in progress]
    if not all(math.isfinite(value) for value in values):
        raise ValueError("AI Race progress values must be finite")
    if not math.isfinite(float(tolerance)) or tolerance < 0:
        raise ValueError("AI Race tie tolerance must be finite and non-negative")
    best = max(values)
    at_max = [value >= best - tolerance for value in values]
    solo_leader = sum(at_max) == 1
    return [
        ("winner" if solo_leader else "tie") if leader else "loser"
        for leader in at_max
    ]


def terminal_scoring(
    *,
    stage_payoffs: Sequence[float],
    progress: Sequence[float],
    unsafe_counts: Sequence[int],
    rounds_played: int,
    max_private_risk: float,
    race_prize: float,
    setback_draws: Sequence[float],
) -> NPlayerTerminalScoring:
    """Resolve prize, private setback, and final task payoff for N players.

    A deterministic draw is supplied for every seat even when a seat is not
    eligible, keeping RNG streams aligned across treatments -- the draw only
    affects a winner or a tied leader. The prize is split evenly among
    however many players share the lead (2 for a pairwise tie, all N for an
    all-Safe or all-equally-fast-Unsafe group).
    """
    n_players = len(stage_payoffs)
    lengths = {
        len(stage_payoffs),
        len(progress),
        len(unsafe_counts),
        len(setback_draws),
    }
    if lengths != {n_players} or n_players < 2:
        raise ValueError(
            "Terminal scoring expects the same number of values (>= 2) for "
            "every player field"
        )
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
    winner_count = sum(1 for outcome in outcomes if outcome in {"winner", "tie"})
    prizes = [
        float(race_prize) / winner_count if outcome in {"winner", "tie"} else 0.0
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
    return NPlayerTerminalScoring(
        outcomes=outcomes,
        prizes=prizes,
        private_risks=risks,
        setback_eligible=eligible,
        setback_draws=draws,
        setbacks=setbacks,
        final_payoffs=final_payoffs,
    )
