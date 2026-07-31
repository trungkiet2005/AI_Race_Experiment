"""Game-theoretic and evolutionary analysis of the AI Race mechanism.

This package is deliberately data-free. Everything here is a property of the
*game* defined by a :class:`~ai_race.engine.state.GameConfig`, so it produces the
same numbers whether the players are humans, LLMs, or nobody at all. Keeping it
separate from the analyser is what stops a theoretical prediction from being read
as a fit to observed model behaviour.

Every entry point takes a ``GameConfig`` rather than loose parameters: if someone
changes ``unsafeProgress`` or a stage payoff in a configuration file, the theory
must move with it instead of silently describing a game nobody played.
"""
from __future__ import annotations

from .equilibria import (
    nash_equilibria,
    social_dilemma_threshold,
    stage_game_class,
    stage_game_ordering,
    symmetric_nash_strategies,
    unconditional_nash_regions,
)
from .evolution import (
    expected_unsafe_frequency,
    fitness_in_population,
    fixation_probability,
    small_mutation_stationary,
)
from .payoffs import (
    STRATEGY_ORDER,
    UNCONDITIONAL_STRATEGIES,
    exact_expected_payoff,
    expected_horizon,
    expected_payoff,
    expected_payoff_matrix,
    horizon_distribution,
    matrix_to_rows,
    payoff_method,
    sample_horizon,
    self_play_unsafe_frequency,
    simulate_matchup,
    unconditional_expected_payoff,
)

__all__ = [
    "STRATEGY_ORDER",
    "UNCONDITIONAL_STRATEGIES",
    "exact_expected_payoff",
    "expected_horizon",
    "expected_payoff",
    "expected_payoff_matrix",
    "expected_unsafe_frequency",
    "fitness_in_population",
    "fixation_probability",
    "horizon_distribution",
    "matrix_to_rows",
    "nash_equilibria",
    "payoff_method",
    "sample_horizon",
    "self_play_unsafe_frequency",
    "simulate_matchup",
    "small_mutation_stationary",
    "social_dilemma_threshold",
    "stage_game_class",
    "stage_game_ordering",
    "symmetric_nash_strategies",
    "unconditional_expected_payoff",
    "unconditional_nash_regions",
]
