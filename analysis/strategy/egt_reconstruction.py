"""Faithful reconstruction of the paper's reduced evolutionary AI-race model.

The source paper specifies the game, four deterministic strategies, and the
pairwise-comparison process, but does not release the model code, payoff-matrix
Monte Carlo seeds, or an EGTtools version.  This module therefore reconstructs
the disclosed model in dependency-light Python.  Expected matchup quantities are
summed over the geometric horizon rather than estimated by Monte Carlo; the
finite-population stationary quantities are estimated with independent seeded
Markov chains that implement the transition rule documented by EGTtools.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Sequence

import numpy as np


STRATEGIES: tuple[str, ...] = ("AS", "AU", "CS", "CAS")
SAFE = 0
UNSAFE = 1


@dataclass(frozen=True)
class ModelParameters:
    """Parameters disclosed in arXiv:2607.26034v1."""

    stage_payoffs: tuple[tuple[float, float], tuple[float, float]] = (
        (1.0, 0.6),
        (2.4, 2.0),
    )
    safe_progress: float = 1.0
    unsafe_progress: float = 1.5
    race_prize: float = 100.0
    min_rounds: int = 5
    stop_probability: float = 0.2
    population_size: int = 100

    def __post_init__(self) -> None:
        if self.min_rounds < 1:
            raise ValueError("min_rounds must be positive")
        if not 0.0 < self.stop_probability <= 1.0:
            raise ValueError("stop_probability must lie in (0, 1]")
        if self.population_size < 2:
            raise ValueError("population_size must be at least two")


@dataclass(frozen=True)
class MatchOutcome:
    """Expected-payoff ingredients for one fixed realised horizon."""

    horizon: int
    actions_a: tuple[int, ...]
    actions_b: tuple[int, ...]
    payoff_a: float
    payoff_b: float
    unsafe_fraction_a: float
    unsafe_fraction_b: float


@dataclass(frozen=True)
class ExpectedGame:
    """Ordered-strategy payoff and behaviour matrices."""

    max_private_risk: float
    payoff_matrix: np.ndarray
    unsafe_fraction_matrix: np.ndarray
    horizon_tail_mass: float
    max_horizon: int


@dataclass(frozen=True)
class ChainSummary:
    """One independent finite-population chain summary."""

    seed: int
    samples: int
    strategy_frequencies: tuple[float, ...]
    unsafe_frequency: float
    moves: int


def _strategy_action(strategy: str, *, round_index: int, opponent_previous: int | None) -> int:
    if strategy == "AS":
        return SAFE
    if strategy == "AU":
        return UNSAFE
    if strategy not in {"CS", "CAS"}:
        raise ValueError(f"unknown strategy {strategy!r}")
    if round_index == 0:
        return SAFE if strategy == "CS" else UNSAFE
    if opponent_previous is None:
        raise ValueError("conditional strategy needs the opponent's previous action")
    return int(opponent_previous)


def strategy_pair_trajectories(
    strategy_a: str,
    strategy_b: str,
    horizon: int,
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Generate simultaneous trajectories for an ordered strategy pair."""

    if horizon < 1:
        raise ValueError("horizon must be positive")
    actions_a: list[int] = []
    actions_b: list[int] = []
    for round_index in range(horizon):
        previous_a = actions_a[-1] if actions_a else None
        previous_b = actions_b[-1] if actions_b else None
        action_a = _strategy_action(
            strategy_a,
            round_index=round_index,
            opponent_previous=previous_b,
        )
        action_b = _strategy_action(
            strategy_b,
            round_index=round_index,
            opponent_previous=previous_a,
        )
        actions_a.append(action_a)
        actions_b.append(action_b)
    return tuple(actions_a), tuple(actions_b)


def fixed_horizon_outcome(
    strategy_a: str,
    strategy_b: str,
    horizon: int,
    max_private_risk: float,
    params: ModelParameters | None = None,
) -> MatchOutcome:
    """Return expected payoffs after integrating out the private setback draw.

    The paper's equations multiply the complete payoff of a winner or tied winner
    (accumulated stage payoff plus prize) by one minus effective private risk.
    A loser keeps accumulated stage payoffs and faces no setback lottery.
    """

    params = params or ModelParameters()
    if not 0.0 <= max_private_risk <= 1.0:
        raise ValueError("max_private_risk must lie in [0, 1]")
    actions_a, actions_b = strategy_pair_trajectories(strategy_a, strategy_b, horizon)
    stage_a = sum(params.stage_payoffs[a][b] for a, b in zip(actions_a, actions_b))
    stage_b = sum(params.stage_payoffs[b][a] for a, b in zip(actions_a, actions_b))
    progress = {SAFE: params.safe_progress, UNSAFE: params.unsafe_progress}
    progress_a = sum(progress[action] for action in actions_a)
    progress_b = sum(progress[action] for action in actions_b)
    unsafe_fraction_a = sum(actions_a) / horizon
    unsafe_fraction_b = sum(actions_b) / horizon

    if math.isclose(progress_a, progress_b, abs_tol=1e-12):
        prize_a = prize_b = params.race_prize / 2.0
        eligible_a = eligible_b = True
    elif progress_a > progress_b:
        prize_a, prize_b = params.race_prize, 0.0
        eligible_a, eligible_b = True, False
    else:
        prize_a, prize_b = 0.0, params.race_prize
        eligible_a, eligible_b = False, True

    payoff_a = stage_a + prize_a
    payoff_b = stage_b + prize_b
    if eligible_a:
        payoff_a *= 1.0 - max_private_risk * unsafe_fraction_a
    if eligible_b:
        payoff_b *= 1.0 - max_private_risk * unsafe_fraction_b
    return MatchOutcome(
        horizon=horizon,
        actions_a=actions_a,
        actions_b=actions_b,
        payoff_a=float(payoff_a),
        payoff_b=float(payoff_b),
        unsafe_fraction_a=float(unsafe_fraction_a),
        unsafe_fraction_b=float(unsafe_fraction_b),
    )


def expected_game(
    max_private_risk: float,
    *,
    params: ModelParameters | None = None,
    tail_tolerance: float = 1e-13,
) -> ExpectedGame:
    """Sum the ordered matchup matrix over the paper's geometric horizon."""

    params = params or ModelParameters()
    if not 0.0 < tail_tolerance < 1.0:
        raise ValueError("tail_tolerance must lie in (0, 1)")
    n = len(STRATEGIES)
    payoff_matrix = np.zeros((n, n), dtype=float)
    unsafe_matrix = np.zeros((n, n), dtype=float)
    continuation = 1.0 - params.stop_probability
    horizon = params.min_rounds
    remaining = 1.0

    while remaining > tail_tolerance:
        probability = params.stop_probability * continuation ** (horizon - params.min_rounds)
        for i, strategy_i in enumerate(STRATEGIES):
            for j, strategy_j in enumerate(STRATEGIES):
                outcome = fixed_horizon_outcome(
                    strategy_i,
                    strategy_j,
                    horizon,
                    max_private_risk,
                    params,
                )
                payoff_matrix[i, j] += probability * outcome.payoff_a
                unsafe_matrix[i, j] += probability * outcome.unsafe_fraction_a
        horizon += 1
        remaining = continuation ** (horizon - params.min_rounds)

    # The discarded tail is tiny but rescaling makes constant functions integrate
    # to exactly one under the numerically truncated distribution.
    included_mass = 1.0 - remaining
    payoff_matrix /= included_mass
    unsafe_matrix /= included_mass
    return ExpectedGame(
        max_private_risk=float(max_private_risk),
        payoff_matrix=payoff_matrix,
        unsafe_fraction_matrix=unsafe_matrix,
        horizon_tail_mass=float(remaining),
        max_horizon=horizon - 1,
    )


def fitnesses(counts: Sequence[int], payoff_matrix: np.ndarray) -> np.ndarray:
    """Expected payoff by strategy under random matching without self-play."""

    state = np.asarray(counts, dtype=float)
    if state.ndim != 1 or payoff_matrix.shape != (state.size, state.size):
        raise ValueError("counts and payoff_matrix dimensions do not agree")
    population_size = int(state.sum())
    if population_size < 2:
        raise ValueError("population must contain at least two individuals")
    total = payoff_matrix @ state
    return (total - np.diag(payoff_matrix)) / (population_size - 1)


def population_unsafe_frequency(counts: Sequence[int], unsafe_matrix: np.ndarray) -> float:
    """Expected decision-weighted Unsafe rate under random ordered matching."""

    state = np.asarray(counts, dtype=float)
    population_size = int(state.sum())
    if population_size < 2:
        raise ValueError("population must contain at least two individuals")
    ordered_pairs = np.outer(state, state)
    ordered_pairs[np.diag_indices_from(ordered_pairs)] -= state
    return float(np.sum(ordered_pairs * unsafe_matrix) / (population_size * (population_size - 1)))


def _draw_strategy(rng: np.random.Generator, counts: Sequence[int], total: int) -> int:
    draw = int(rng.integers(total))
    cumulative = 0
    for index, count in enumerate(counts):
        cumulative += int(count)
        if draw < cumulative:
            return index
    raise AssertionError("categorical draw exceeded population")


def simulate_pairwise_comparison_chain(
    game: ExpectedGame,
    *,
    beta: float,
    mutation: float,
    seed: int,
    burn_in: int = 100_000,
    steps: int = 500_000,
    thin: int = 50,
    population_size: int = 100,
) -> ChainSummary:
    """Estimate the finite-population stationary distribution with one chain.

    An updating individual is sampled first.  With probability ``mutation`` it
    changes uniformly to one of the other strategies.  Otherwise a role model is
    sampled from the remaining population and copied with Fermi probability
    ``logistic(beta * (f_role - f_focal))``.  This matches the transition rule in
    EGTTools' pairwise-comparison process at docs-branch commit df7f5fb.
    """

    if beta < 0.0:
        raise ValueError("beta must be non-negative")
    if not 0.0 < mutation < 1.0:
        raise ValueError("mutation must lie in (0, 1)")
    if burn_in < 0 or steps < 1 or thin < 1:
        raise ValueError("burn_in, steps, and thin are invalid")
    rng = np.random.default_rng(seed)
    n = len(STRATEGIES)
    counts = np.full(n, population_size // n, dtype=np.int64)
    counts[: population_size % n] += 1
    samples = 0
    accumulated = np.zeros(n, dtype=float)
    accumulated_unsafe = 0.0
    moves = 0

    total_iterations = burn_in + steps
    for iteration in range(total_iterations):
        focal = _draw_strategy(rng, counts, population_size)
        target = focal
        if rng.random() < mutation:
            candidate = int(rng.integers(n - 1))
            target = candidate if candidate < focal else candidate + 1
        else:
            role_counts = counts.copy()
            role_counts[focal] -= 1
            role = _draw_strategy(rng, role_counts, population_size - 1)
            if role != focal:
                current_fitness = fitnesses(counts, game.payoff_matrix)
                difference = beta * (current_fitness[role] - current_fitness[focal])
                if difference >= 0:
                    adoption = 1.0 / (1.0 + math.exp(-min(difference, 700.0)))
                else:
                    exp_difference = math.exp(max(difference, -700.0))
                    adoption = exp_difference / (1.0 + exp_difference)
                if rng.random() < adoption:
                    target = role
        if target != focal:
            counts[focal] -= 1
            counts[target] += 1
            moves += 1

        if iteration >= burn_in and (iteration - burn_in) % thin == 0:
            accumulated += counts / population_size
            accumulated_unsafe += population_unsafe_frequency(counts, game.unsafe_fraction_matrix)
            samples += 1

    return ChainSummary(
        seed=int(seed),
        samples=samples,
        strategy_frequencies=tuple(float(value) for value in accumulated / samples),
        unsafe_frequency=float(accumulated_unsafe / samples),
        moves=moves,
    )


def run_independent_chains(
    game: ExpectedGame,
    *,
    beta: float,
    mutation: float,
    seeds: Iterable[int],
    burn_in: int,
    steps: int,
    thin: int,
    population_size: int = 100,
) -> list[ChainSummary]:
    return [
        simulate_pairwise_comparison_chain(
            game,
            beta=beta,
            mutation=mutation,
            seed=seed,
            burn_in=burn_in,
            steps=steps,
            thin=thin,
            population_size=population_size,
        )
        for seed in seeds
    ]

