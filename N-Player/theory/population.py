"""Finite-population evolutionary dynamics for the N-team DSAIR.

Reproduces Appendix B's "Methods: Payoffs Over Group Samplings" (multivariate
hypergeometric group sampling, Eq. 29), its transition-probability /
risk-dominance machinery (Eq. 30-31), and the small-mutation-limit stationary
distribution used throughout the paper's N-team figures (S6-S9, S12).

Structurally this mirrors ``ai_race/theory/evolution.py`` (same Fermi
pairwise-comparison rule, same log-sum-exp accumulation to avoid overflow at
large beta, same eigenvector construction for the stationary distribution).
The one thing that changes going from two players to N is *how group fitness
is averaged*: the two-player module averages directly over the population
(``fitness_in_population`` in ``ai_race/theory/evolution.py``); here a focal
player's opponents are a hypergeometric sample of size ``N-1`` drawn from the
rest of the population, so the average has to be taken over every possible
group composition first (:func:`average_payoff_i`/:func:`average_payoff_j`,
Eq. 29) before the same Fermi/eigenvector machinery applies. At ``n=2`` the
two constructions are shown to coincide exactly in
``tests/test_population.py``.

Every function here takes payoff **as a function of group composition**
(``payoff_of_k: k -> float``) rather than a fixed matrix, since Appendix B's
``Pi_ij(k)`` already depends on how many of the ``N`` group members chose the
focal strategy -- see ``theory.conditions`` for the raw per-round
``pi_SAFE(k)``/``pi_UNSAFE(k)`` and ``theory.welfare`` for the averaged
whole-race ``Pi_AS,AU(k)`` etc. that would typically be passed in here.

**One convention throughout, self-inclusive counts**: every ``payoff_of_k``
passed to :func:`fitness_in_population`/:func:`fixation_probability`/
:func:`small_mutation_stationary` -- for *either* the mutant or the resident
strategy -- takes ``k`` as that strategy's own total count in the group,
self included (``1 <= k <= n``). This is what ``theory.welfare``'s
``average_payoff_{as,cs}_vs_au`` already return natively; a strategy whose
natural description counts something else (``average_payoff_au_vs_{as,cs}``
counts *co-players*, self excluded) needs converting to this convention
before being passed in here -- see ``theory/stationary.py``'s
``build_payoff_lookup`` for that conversion. Keeping to one convention is
what lets the same two functions for a strategy pair be reused unchanged
regardless of which one is currently mutant and which is resident;
:func:`fitness_in_population` does the one remaining internal conversion
(residents are naturally read off by *co-player* count) itself.
"""
from __future__ import annotations

from math import comb
from typing import Callable, Mapping, Sequence

import numpy as np

#: A strategy's payoff as a function of how many of the N group members
#: (including the focal player, for the "own type" side of Eq. 29) chose that
#: same strategy.
PayoffOfK = Callable[[int], float]


def hypergeometric_pmf(k: int, n_trials: int, x: int, z: int) -> float:
    """``H(k, N, x, Z)``: probability of drawing ``k`` type-i and ``N-k``
    type-j individuals, without replacement, from a population of ``x``
    type-i and ``Z-x`` type-j individuals in ``N`` trials (Hauert et al., 2007,
    as cited in Appendix B).
    """

    if not 0 <= k <= n_trials:
        raise ValueError("k must be between 0 and n_trials inclusive")
    if not 0 <= x <= z:
        raise ValueError("x must be between 0 and z inclusive")
    if n_trials > z:
        raise ValueError("n_trials cannot exceed the population size z")
    type_i_ways = comb(x, k) if k <= x else 0
    remaining_trials = n_trials - k
    type_j_ways = comb(z - x, remaining_trials) if remaining_trials <= (z - x) else 0
    total_ways = comb(z, n_trials)
    return (type_i_ways * type_j_ways) / total_ways


def average_payoff_i(
    payoff_i_of_k: PayoffOfK,
    *,
    x: int,
    z: int,
    n: int,
) -> float:
    """``P_ij(x)`` (Eq. 29): average payoff to a type-i strategist when ``x``
    of the ``z`` population members are type-i.

    ``payoff_i_of_k(k)`` is the type-i payoff when ``k`` of the ``n`` group
    members (the focal player included) are type-i, for ``1 <= k <= n``.
    """

    if not 1 <= x <= z:
        raise ValueError("x must be at least 1 (a type-i individual must exist)")
    total = 0.0
    for k in range(0, n):
        weight = hypergeometric_pmf(k, n - 1, x - 1, z - 1)
        if weight == 0.0:
            continue
        total += weight * payoff_i_of_k(k + 1)
    return total


def average_payoff_j(
    payoff_j_of_k: PayoffOfK,
    *,
    x: int,
    z: int,
    n: int,
) -> float:
    """``P_ji(x)`` (Eq. 29): average payoff to a type-j strategist when ``x``
    of the ``z`` population members are type-i.

    ``payoff_j_of_k(k)`` is the type-j payoff when ``k`` of the focal
    player's ``n-1`` co-players are type-i, for ``0 <= k <= n-1``.
    """

    if not 0 <= x <= z - 1:
        raise ValueError("z - x must be at least 1 (a type-j individual must exist)")
    total = 0.0
    for k in range(0, n):
        weight = hypergeometric_pmf(k, n - 1, x, z - 1)
        if weight == 0.0:
            continue
        total += weight * payoff_j_of_k(k)
    return total


def fitness_in_population(
    mutant_payoff_of_k: PayoffOfK,
    resident_payoff_of_k: PayoffOfK,
    *,
    x: int,
    z: int,
    n: int,
) -> tuple[float, float]:
    """Average payoffs with ``x`` mutants and ``z - x`` residents, N-team version
    of ``ai_race.theory.evolution.fitness_in_population``.

    Both ``mutant_payoff_of_k`` and ``resident_payoff_of_k`` use the *same*
    convention: a strategy's payoff as a function of its own total count in
    the group, self included (``1 <= k <= n``) -- e.g.
    ``theory.welfare.average_payoff_as_vs_au`` as-is. This single convention
    is what makes a payoff-of-k function reusable unchanged regardless of
    whether its strategy is currently playing mutant or resident: only the
    *resident*'s reading needs converting, and this function does that
    conversion internally (``own count = n - (count of mutants among the
    other n-1)``) rather than asking the caller to supply two differently
    -indexed functions per strategy. Getting this wrong is subtle and silent
    -- both conventions are valid-looking Python callables ``int -> float``,
    they just disagree on what the integer means, so there is no type error
    to catch it; ``tests/test_population.py`` cross-checks the N=2 reduction
    to guard against regressing this.
    """

    if not 0 < x < z:
        raise ValueError("x must be strictly between 0 and z")
    if z < n:
        raise ValueError("z must be at least n for group sampling to be defined")
    mutant_fitness = average_payoff_i(mutant_payoff_of_k, x=x, z=z, n=n)
    resident_fitness = average_payoff_j(
        lambda k: resident_payoff_of_k(n - k), x=x, z=z, n=n
    )
    return float(mutant_fitness), float(resident_fitness)


def fixation_probability(
    mutant_payoff_of_k: PayoffOfK,
    resident_payoff_of_k: PayoffOfK,
    *,
    z: int,
    n: int,
    beta: float,
) -> float:
    """Probability that one mutant fixates in a resident population of size ``z``.

    Same pairwise-comparison (Fermi) rule and log-sum-exp accumulation as
    ``ai_race.theory.evolution.fixation_probability``; the only change is that
    ``fitness_in_population`` here averages over hypergeometric group draws
    (Eq. 29) instead of a direct pairwise average. At ``beta=0`` this returns
    exactly ``1/z``, matching the paper's stated neutral-drift limit (Methods,
    "when beta = 0, rho = 1/Z").
    """

    if z < n:
        raise ValueError("z must be at least n for group sampling to be defined")

    log_terms = np.empty(z - 1, dtype=float)
    running = 0.0
    for index, j in enumerate(range(1, z)):
        mutant_fitness, resident_fitness = fitness_in_population(
            mutant_payoff_of_k, resident_payoff_of_k, x=j, z=z, n=n
        )
        running += -float(beta) * (mutant_fitness - resident_fitness)
        log_terms[index] = running

    stacked = np.concatenate(([0.0], log_terms))
    shift = float(np.max(stacked))
    total = shift + float(np.log(np.sum(np.exp(stacked - shift))))
    return float(np.exp(-total))


def small_mutation_stationary(
    payoff_of_k: Mapping[tuple[str, str], PayoffOfK],
    strategies: Sequence[str],
    *,
    z: int,
    n: int,
    beta: float,
) -> dict[str, float]:
    """Stationary distribution of the embedded monomorphic chain over ``strategies``.

    ``payoff_of_k[(own, opponent)]`` must give own's payoff as a function of
    how many group members (self included, per :func:`average_payoff_i`'s
    convention) play ``own``'s strategy, for every ordered pair drawn from
    ``strategies``. Same eigenvector construction as
    ``ai_race.theory.evolution.small_mutation_stationary``.
    """

    names = list(strategies)
    m = len(names)
    if m < 2:
        raise ValueError("the embedded chain needs at least two strategies")

    transitions = np.zeros((m, m), dtype=float)
    for row, resident in enumerate(names):
        for column, mutant in enumerate(names):
            if row == column:
                continue
            transitions[row, column] = fixation_probability(
                payoff_of_k[(mutant, resident)],
                payoff_of_k[(resident, mutant)],
                z=z,
                n=n,
                beta=beta,
            ) / (m - 1)
        transitions[row, row] = 1.0 - transitions[row].sum()

    values, vectors = np.linalg.eig(transitions.T)
    stationary = np.real(vectors[:, int(np.argmin(np.abs(values - 1.0)))])
    stationary = np.abs(stationary)
    total = stationary.sum()
    if total <= 0:
        raise ValueError("the embedded chain has no usable stationary distribution")
    stationary = stationary / total
    return {name: float(value) for name, value in zip(names, stationary)}


def risk_dominant(
    payoff_i_of_k: PayoffOfK,
    payoff_j_of_k: PayoffOfK,
    *,
    n: int,
) -> bool:
    """Eq. 31: whether strategy i is risk-dominant against j in the large-``Z``
    limit, i.e. ``sum_{k=1}^{n} Pi_ij(k) >= sum_{k=0}^{n-1} Pi_ji(k)``.
    """

    lhs = sum(payoff_i_of_k(k) for k in range(1, n + 1))
    rhs = sum(payoff_j_of_k(k) for k in range(0, n))
    return lhs >= rhs
