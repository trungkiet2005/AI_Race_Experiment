"""Stage-game classification and Nash equilibria of the AI Race.

The point of this module is a distinction the source paper makes and that is easy
to lose: the *stage* game is not a Prisoner's Dilemma, but the *repeated* game with
accumulated private risk behaves like a social dilemma above a risk threshold.
Reporting only "Unsafe dominates each round" would state the first half and imply
the wrong second half.

Everything is derived from a :class:`~ai_race.engine.state.GameConfig`. Nothing
here is calibrated to observed play.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Mapping, Sequence

from ..engine.state import Action, GameConfig
from .payoffs import (
    STRATEGY_ORDER,
    expected_horizon,
    expected_payoff_matrix,
    unconditional_expected_payoff,
)


@dataclass(frozen=True)
class StageGame:
    """The 2x2 stage game in the standard Prisoner's-Dilemma letters.

    Safe maps to Cooperate and Unsafe to Defect, because Unsafe is the action that
    raises this player's payoff while lowering the other's.
    """

    temptation: float  # T: Unsafe against Safe
    reward: float  # R: mutual Safe
    punishment: float  # P: mutual Unsafe
    sucker: float  # S: Safe against Unsafe


def stage_game(config: GameConfig) -> StageGame:
    return StageGame(
        temptation=config.stage_payoff(Action.UNSAFE, Action.SAFE),
        reward=config.stage_payoff(Action.SAFE, Action.SAFE),
        punishment=config.stage_payoff(Action.UNSAFE, Action.UNSAFE),
        sucker=config.stage_payoff(Action.SAFE, Action.UNSAFE),
    )


def stage_game_ordering(config: GameConfig) -> str:
    """The strict payoff ordering as a string such as ``"T>P>R>S"``."""

    game = stage_game(config)
    labels = {
        "T": game.temptation,
        "R": game.reward,
        "P": game.punishment,
        "S": game.sucker,
    }
    ranked = sorted(labels, key=lambda letter: labels[letter], reverse=True)
    parts: list[str] = [ranked[0]]
    for previous, letter in zip(ranked, ranked[1:]):
        parts.append("=" if labels[letter] == labels[previous] else ">")
        parts.append(letter)
    return "".join(parts)


# The four orderings with settled names in the 2x2 literature. Anything else is
# reported by its ordering rather than given an invented label.
_NAMED_ORDERINGS: Mapping[str, str] = {
    "T>R>P>S": "prisoners_dilemma",
    "T>P>R>S": "deadlock",
    "T>R>S>P": "chicken",
    "R>T>P>S": "stag_hunt",
    "R>T>S>P": "harmony",
}


def stage_game_class(config: GameConfig) -> str:
    """Name the stage game, or fall back to its ordering when it has no name.

    With the paper's parameters this is ``"deadlock"``: Unsafe strictly dominates
    each round exactly as Defect does, but mutual Unsafe is *better* than mutual
    Safe at stage level (``P > R``), so the Prisoner's-Dilemma condition ``R > P``
    fails. Calling it a Prisoner's Dilemma would misstate what the repetition and
    the accumulated risk are actually doing.
    """

    ordering = stage_game_ordering(config)
    return _NAMED_ORDERINGS.get(ordering, f"unnamed_2x2({ordering})")


def _riskless(config: GameConfig) -> GameConfig:
    return replace(config, max_private_risk=0.0)


def social_dilemma_threshold(config: GameConfig) -> float:
    """The ``p_r^max`` above which mutual Safe beats mutual Unsafe.

    Mutual Safe never triggers the terminal risk lottery, so its expected payoff is
    flat in ``p_r^max`` while mutual Unsafe falls linearly. Above the crossing
    point the repeated game is a social dilemma even though the stage game is not.

    A value at or below 0 means mutual Safe already wins at zero risk; above 1 means
    it never does. Both are reported as-is rather than clamped, so a changed
    mechanism shows up as an out-of-range threshold instead of a plausible number.
    """

    mutual_safe = unconditional_expected_payoff(_riskless(config), "AS", "AS")
    mutual_unsafe_riskless = unconditional_expected_payoff(_riskless(config), "AU", "AU")
    if mutual_unsafe_riskless == 0:
        raise ValueError(
            "mutual Unsafe has zero riskless payoff; the social-dilemma threshold "
            "is undefined for this mechanism"
        )
    return 1.0 - mutual_safe / mutual_unsafe_riskless


def nash_equilibria(
    payoff_matrix: Mapping[tuple[str, str], float],
    strategies: Sequence[str],
    *,
    tolerance: float = 1e-9,
) -> list[tuple[str, str]]:
    """Every pure-strategy Nash profile, by exhaustive best-response search.

    Asymmetric profiles are returned too, not only the diagonal. The paper's two
    structural claims are about them: the four AU/CAS combinations stand or fall
    together, and no profile containing AS survives. Restricting the search to the
    diagonal would make both unverifiable.

    ``tolerance`` treats a deviation gain within Monte Carlo error as no gain; the
    payoffs for conditional strategies are simulated, so exact comparison would turn
    sampling noise into a spurious equilibrium failure.
    """

    names = list(strategies)
    equilibria: list[tuple[str, str]] = []
    for row in names:
        for column in names:
            row_best = all(
                payoff_matrix[(row, column)] + tolerance
                >= payoff_matrix[(deviation, column)]
                for deviation in names
            )
            if not row_best:
                continue
            # The game is symmetric, so the column player's payoff against `row` is
            # the row-player payoff of the transposed pair.
            column_best = all(
                payoff_matrix[(column, row)] + tolerance
                >= payoff_matrix[(deviation, row)]
                for deviation in names
            )
            if column_best:
                equilibria.append((row, column))
    return equilibria


def symmetric_nash_strategies(
    payoff_matrix: Mapping[tuple[str, str], float],
    strategies: Sequence[str],
    *,
    tolerance: float = 1e-9,
) -> list[str]:
    """The strategies that are a best reply to themselves."""

    equilibria = nash_equilibria(payoff_matrix, strategies, tolerance=tolerance)
    return [row for row, column in equilibria if row == column]


def unconditional_nash_regions(config: GameConfig) -> dict[str, float | str]:
    """Closed-form ``p_r^max`` boundaries of the two-strategy AS/AU game.

    Both boundaries are linear in ``p_r^max`` because the only risk-bearing payoffs
    are the two AU ones, so they can be solved rather than searched:

    * ``(AS, AS)`` is an equilibrium once ``Pi(AS,AS) >= Pi(AU,AS)``.
    * ``(AU, AU)`` remains one while ``Pi(AU,AU) >= Pi(AS,AU)``.

    Between the two the game has a coordination structure with two stable rest
    points, which is the regime the paper highlights.
    """

    riskless = _riskless(config)
    mutual_safe = unconditional_expected_payoff(riskless, "AS", "AS")
    unsafe_against_safe = unconditional_expected_payoff(riskless, "AU", "AS")
    safe_against_unsafe = unconditional_expected_payoff(riskless, "AS", "AU")
    mutual_unsafe = unconditional_expected_payoff(riskless, "AU", "AU")

    always_safe_from = (
        1.0 - mutual_safe / unsafe_against_safe
        if unsafe_against_safe > 0
        else float("-inf")
    )
    always_unsafe_until = (
        1.0 - safe_against_unsafe / mutual_unsafe
        if mutual_unsafe > 0
        else float("-inf")
    )
    risk = float(config.max_private_risk)
    if risk < always_safe_from:
        region = "always_unsafe_only"
    elif risk > always_unsafe_until:
        region = "always_safe_only"
    else:
        region = "bistable_coordination"
    return {
        "always_safe_nash_from": float(always_safe_from),
        "always_unsafe_nash_until": float(always_unsafe_until),
        "max_private_risk": risk,
        "region": region,
    }


def equilibrium_summary(
    config: GameConfig,
    *,
    method: str = "exact",
    replications: int = 10_000,
    seed: int | None = None,
    tolerance: float = 1e-9,
) -> dict[str, object]:
    """Everything ``theory_equilibria.csv`` needs for one risk treatment.

    Defaults to the exact payoff matrix. Under Monte Carlo the caller must widen
    ``tolerance`` past the sampling error, or cells that are equal in the game will
    read as strict preferences and equilibria will go missing.
    """

    matrix = expected_payoff_matrix(
        config,
        method=method,
        replications=replications,
        seed=seed,
    )
    equilibria = nash_equilibria(matrix, STRATEGY_ORDER, tolerance=tolerance)
    symmetric = symmetric_nash_strategies(matrix, STRATEGY_ORDER, tolerance=tolerance)
    regions = unconditional_nash_regions(config)
    return {
        "max_private_risk": float(config.max_private_risk),
        "expected_horizon": expected_horizon(config),
        "stage_game_class": stage_game_class(config),
        "stage_game_ordering": stage_game_ordering(config),
        "social_dilemma_threshold": social_dilemma_threshold(config),
        "above_social_dilemma_threshold": bool(
            float(config.max_private_risk) > social_dilemma_threshold(config)
        ),
        "nash_equilibria": "|".join(f"{row},{column}" for row, column in equilibria),
        "symmetric_nash_strategies": "|".join(symmetric),
        "as_is_nash": "AS" in symmetric,
        "unconditional_region": regions["region"],
        "always_safe_nash_from": regions["always_safe_nash_from"],
        "always_unsafe_nash_until": regions["always_unsafe_nash_until"],
        "payoff_method": method,
        "replications": int(replications) if method == "monte_carlo" else 0,
        "seed": seed if method == "monte_carlo" else None,
    }
