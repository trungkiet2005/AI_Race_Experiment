"""Finite-population evolutionary dynamics in the small-mutation limit.

As the mutation rate goes to zero a well-mixed population is monomorphic almost
all the time, and the full Markov chain over every population composition collapses
to an embedded chain over the four monomorphic states. Its transition rates are the
fixation probabilities of a single mutant. That reduction turns a chain with
``C(Z+3, 3) = 176_851`` states at ``Z = 100`` into a 4x4 eigenvector problem, so it
needs nothing beyond NumPy.

What the reduction buys and what it costs: it reproduces the treatment-dependent
ordering of dominant strategies — the substantive result — but it cannot represent
a *finite* mutation rate, where stationary mass spreads into the interior of the
simplex. Anything that varies ``mu`` needs the full chain, and this module refuses
to pretend otherwise rather than accepting a ``mu`` argument it would ignore.
"""
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .payoffs import STRATEGY_ORDER

#: Identifies results produced by the limit rather than by a finite-mutation chain.
MUTATION_REGIME = "small_mutation_limit"


def fitness_in_population(
    payoff_matrix: Mapping[tuple[str, str], float],
    resident: str,
    mutant: str,
    k: int,
    Z: int,
) -> tuple[float, float]:
    """Average payoffs with ``k`` mutants and ``Z - k`` residents.

    Pairing excludes self-interaction, hence the ``Z - 1`` denominator and the
    ``k - 1`` count of same-type partners for a mutant.
    """

    if not 0 < k < Z:
        raise ValueError("k must be strictly between 0 and Z")
    if Z < 2:
        raise ValueError("Z must be at least 2")
    mutant_fitness = (
        (k - 1) * payoff_matrix[(mutant, mutant)]
        + (Z - k) * payoff_matrix[(mutant, resident)]
    ) / (Z - 1)
    resident_fitness = (
        k * payoff_matrix[(resident, mutant)]
        + (Z - k - 1) * payoff_matrix[(resident, resident)]
    ) / (Z - 1)
    return float(mutant_fitness), float(resident_fitness)


def fixation_probability(
    payoff_matrix: Mapping[tuple[str, str], float],
    mutant: str,
    resident: str,
    *,
    Z: int,
    beta: float,
) -> float:
    """Probability that one ``mutant`` takes over a ``resident`` population.

    Pairwise-comparison (Fermi) update:
    ``rho = 1 / (1 + sum_{i=1}^{Z-1} prod_{j=1}^{i} exp(-beta (f_mutant(j) - f_resident(j))))``

    The products are accumulated in log space. At ``beta = 2`` with payoffs of order
    100 the exponents reach several hundred, and a direct product would overflow to
    ``inf`` and silently return a fixation probability of exactly zero.
    """

    if mutant == resident:
        return 1.0 / float(Z)

    log_terms = np.empty(Z - 1, dtype=float)
    running = 0.0
    for index, j in enumerate(range(1, Z)):
        mutant_fitness, resident_fitness = fitness_in_population(
            payoff_matrix,
            resident,
            mutant,
            j,
            Z,
        )
        running += -float(beta) * (mutant_fitness - resident_fitness)
        log_terms[index] = running

    # log(1 + sum(exp(log_terms))) without forming the sum itself.
    stacked = np.concatenate(([0.0], log_terms))
    shift = float(np.max(stacked))
    total = shift + float(np.log(np.sum(np.exp(stacked - shift))))
    return float(np.exp(-total))


def small_mutation_stationary(
    payoff_matrix: Mapping[tuple[str, str], float],
    strategies: Sequence[str] | None = None,
    *,
    Z: int,
    beta: float,
) -> dict[str, float]:
    """Stationary distribution of the embedded monomorphic chain.

    ``M[a][b] = rho_{a->b} / (n - 1)`` for ``a != b``: from a resident population of
    ``a``, one of the other ``n - 1`` strategies appears as a mutant with equal
    probability and fixates with probability ``rho``. The diagonal absorbs the rest.
    """

    names = list(strategies) if strategies is not None else list(STRATEGY_ORDER)
    n = len(names)
    if n < 2:
        raise ValueError("the embedded chain needs at least two strategies")

    transitions = np.zeros((n, n), dtype=float)
    for row, resident in enumerate(names):
        for column, mutant in enumerate(names):
            if row == column:
                continue
            transitions[row, column] = fixation_probability(
                payoff_matrix,
                mutant,
                resident,
                Z=Z,
                beta=beta,
            ) / (n - 1)
        transitions[row, row] = 1.0 - transitions[row].sum()

    values, vectors = np.linalg.eig(transitions.T)
    stationary = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    stationary = np.abs(stationary)
    total = stationary.sum()
    if total <= 0:
        raise ValueError("the embedded chain has no usable stationary distribution")
    stationary = stationary / total
    return {name: float(value) for name, value in zip(names, stationary)}


def expected_unsafe_frequency(
    stationary: Mapping[str, float],
    unsafe_frequency_by_strategy: Mapping[str, float],
) -> float:
    """Population Unsafe frequency implied by a stationary distribution.

    Each strategy's own Unsafe frequency has to be supplied rather than assumed:
    for CS and CAS it depends on who they meet, so it is a property of the
    simulated matchups, not of the strategy label.
    """

    missing = set(stationary) - set(unsafe_frequency_by_strategy)
    if missing:
        raise ValueError(
            f"no Unsafe frequency supplied for {sorted(missing)}; the conditional "
            "strategies' rates must be measured, not assumed"
        )
    return float(
        sum(
            weight * float(unsafe_frequency_by_strategy[name])
            for name, weight in stationary.items()
        )
    )
