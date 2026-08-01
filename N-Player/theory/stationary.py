"""Assembles the AS/AU/CS stationary distribution for the N-team DSAIR.

Bridges ``theory.welfare``'s per-strategy-pair payoff functions into
``theory.population``'s finite-population machinery -- this is the piece that
actually reproduces "frequency of AU" as plotted in the paper's Figures
S6-S9 and S12, and is what ``figures/reproduce_paper_figures.py`` calls.

The one thing to get right here is the *convention* each payoff function
uses (see ``theory/population.py``'s module docstring): everything passed to
``theory.population.small_mutation_stationary`` must count a strategy's own
total presence in the group, self included (``1 <= k <= n``).

- ``theory.welfare.average_payoff_{as,cs}_vs_au`` are already in that
  convention (they were written from the SAFE-side player's own point of
  view) -- used unchanged.
- ``theory.welfare.average_payoff_au_vs_{as,cs}`` count SAFE *co-players*
  instead (self excluded, ``0 <= k < n``) -- converted here via ``k -> n -
  k``.
- AS and CS never actually meet AU-free of each other in the small-mutation
  limit's embedded chain view of an (AS, CS) pair: with no AU present both
  always play SAFE (CS only ever deviates in response to an UNSAFE co-player),
  so their payoff is the constant :func:`theory.welfare.homogeneous_payoff`,
  independent of ``k``.
"""
from __future__ import annotations

from typing import Sequence

from theory.population import PayoffOfK, small_mutation_stationary
from theory.welfare import (
    average_payoff_as_vs_au,
    average_payoff_au_vs_as,
    average_payoff_au_vs_cs,
    average_payoff_cs_vs_au,
    homogeneous_payoff,
)

#: Default strategy set, matching the paper's N-player appendix (no N-player CAS).
STRATEGIES: tuple[str, ...] = ("AS", "AU", "CS")


def build_payoff_lookup(
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    B: float,
    W: float,
    pr: float = 0.0,
    pfo: float = 0.0,
    gamma: float = 0.0,
) -> dict[tuple[str, str], PayoffOfK]:
    """The 6 ordered-pair payoff-of-k functions ``theory.population`` needs
    for the 3-strategy (AS, AU, CS) game, all converted to its self
    -inclusive-own-count convention.
    """

    def as_vs_au(k: int) -> float:
        return average_payoff_as_vs_au(
            k, n=n, s=s, b=b, c=c, B=B, W=W, pr=pr, pfo=pfo, gamma=gamma
        )

    def au_vs_as(k: int) -> float:
        # k is AU's own count including self (1..n); the welfare function
        # natively counts SAFE co-players instead (0..n-1) -- convert.
        return average_payoff_au_vs_as(
            n - k, n=n, s=s, b=b, B=B, W=W, pr=pr, pfo=pfo
        )

    def cs_vs_au(k: int) -> float:
        return average_payoff_cs_vs_au(
            k, n=n, s=s, b=b, c=c, B=B, W=W, pr=pr, pfo=pfo, gamma=gamma
        )

    def au_vs_cs(k: int) -> float:
        return average_payoff_au_vs_cs(
            n - k, n=n, s=s, b=b, B=B, W=W, pr=pr, pfo=pfo
        )

    def homogeneous(strategy: str) -> PayoffOfK:
        value = homogeneous_payoff(
            strategy, n=n, s=s, b=b, c=c, B=B, W=W, pr=pr, pfo=pfo
        )
        return lambda _k: value

    return {
        ("AS", "AU"): as_vs_au,
        ("AU", "AS"): au_vs_as,
        ("CS", "AU"): cs_vs_au,
        ("AU", "CS"): au_vs_cs,
        ("AS", "CS"): homogeneous("AS"),
        ("CS", "AS"): homogeneous("CS"),
    }


def stationary_distribution(
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    B: float,
    W: float,
    z: int,
    beta: float,
    pr: float = 0.0,
    pfo: float = 0.0,
    gamma: float = 0.0,
    strategies: Sequence[str] = STRATEGIES,
) -> dict[str, float]:
    """The small-mutation stationary distribution over ``strategies`` for one
    set of DSAIR parameters -- what Figures S6-S9/S12 plot (as "frequency").
    """

    payoff_of_k = build_payoff_lookup(
        n=n, s=s, b=b, c=c, B=B, W=W, pr=pr, pfo=pfo, gamma=gamma
    )
    return small_mutation_stationary(payoff_of_k, strategies, z=z, n=n, beta=beta)


def au_frequency(
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    B: float,
    W: float,
    z: int,
    beta: float,
    pr: float = 0.0,
    pfo: float = 0.0,
    gamma: float = 0.0,
    strategies: Sequence[str] = STRATEGIES,
) -> float:
    """Convenience wrapper: just the AU share of ``stationary_distribution``."""

    stationary = stationary_distribution(
        n=n, s=s, b=b, c=c, B=B, W=W, z=z, beta=beta,
        pr=pr, pfo=pfo, gamma=gamma, strategies=strategies,
    )
    return stationary.get("AU", 0.0)
