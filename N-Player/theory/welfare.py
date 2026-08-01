"""Average whole-race payoffs and social welfare for the N-team DSAIR.

Reproduces Appendix B's "Average payoffs for the repeated games" (the
``Pi_AS,AU(k)``, ``Pi_AU,AS(k)``, ``Pi_CS,AU(k)``, ``Pi_AU,CS(k)`` family) and
Appendix C's personal-vs-collective-risk (``gamma``) scaling, generalised
from the two-player Eq. 32 to N players. Builds on ``theory.conditions``'
per-round ``pi_SAFE(k)``/``pi_UNSAFE(k)``; ``theory.population`` consumes the
functions here as its ``payoff_of_k`` inputs.

**Convention, and why it is not cross-checked against
``ai_race/theory/payoffs.py``**: the paper states explicitly that every entry
of its payoff matrix (Eq. 2, and by extension Eq.-unnumbered Appendix B
matrix reproduced here) is a *per-round rate* -- "obtaining on average B/2W
per round", "obtaining pi11 ... as the intermediate benefit per round". The
functions below follow that convention literally (prize terms divided by
``n*W`` or ``W``, stage payoffs left at their raw per-round value).
``ai_race/theory/payoffs.py`` instead computes *total accumulated* payoff
over the whole race (``horizon * stage_payoff + prize``, prize undivided) --
a different, equally valid scale calibrated to match what
``ai_race.engine``'s recorded ``final_payoffs`` actually accumulate. The two
are not the same quantity and are not meant to be numerically equal; this
module's own two-player reduction (below) is checked instead directly
against the paper's literal Eq. 2, which is the correct ground truth for
"what Appendix B's formula says", independent of that other module's choice.

**On the bare "p" in the source text**: the extracted PDF renders the
``(1 - p_r)`` disaster-survival factor in ``Pi_AU,AS(k)`` and ``Pi_AU,CS(k)``
as a lone "p" (a known OCR gap flagged at the top of the source markdown --
the paper's dense formulas use a math font that does not map to Unicode).
Read as ``(1 - p_r)`` here, the only reading consistent with (a) the
analogous two-player cells in Eq. 2 of the main text and (b) the reduction
check in ``tests/test_welfare.py``.
"""
from __future__ import annotations

from typing import Literal, Mapping

from theory.conditions import stage_payoff_safe, stage_payoff_unsafe

Strategy = Literal["AS", "AU", "CS"]


def expected_horizon(*, min_rounds: int, stop_probability: float) -> float:
    """``E[W] = min_rounds + (1 - p) / p``, same formula as
    ``ai_race.theory.payoffs.expected_horizon`` (duplicated rather than
    imported to keep this package independent of ``ai_race.engine``'s
    ``GameConfig``, per ``ai_race.engine_nplayer``'s own convention).
    """

    if not 0 < stop_probability <= 1:
        raise ValueError("stop_probability must be in (0, 1]")
    return float(min_rounds) + (1.0 - stop_probability) / stop_probability


def average_payoff_as_vs_au(
    k: int,
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
) -> float:
    """``Pi_AS,AU(k)``: AS's average payoff when ``k`` of ``n`` group members
    (AS itself included) chose SAFE, ``1 <= k <= n``.

    ``k == n`` is the all-SAFE state (no AU present, so no disaster risk at
    all: untouched by ``pr``/``gamma``, matching the two-player Eq. 2 diagonal
    cell ``Pi_AS,AS``). For ``1 <= k < n`` the AU co-players can trigger a
    disaster that also hits AS's payoff with probability ``pr * gamma``
    (Appendix C; ``gamma=0`` is the main text's personal-risk assumption,
    ``gamma=1`` is fully collective/shared risk).
    """

    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n inclusive")
    if k == n:
        return B / (n * W) + stage_payoff_safe(n, n=n, s=s, b=b, c=c, pfo=pfo)
    base = stage_payoff_safe(k, n=n, s=s, b=b, c=c, pfo=pfo)
    return (1.0 - pr * gamma) * base


def average_payoff_au_vs_as(
    k: int,
    *,
    n: int,
    s: float,
    b: float,
    B: float,
    W: float,
    pr: float = 0.0,
    pfo: float = 0.0,
) -> float:
    """``Pi_AU,AS(k)``: AU's average payoff when ``k`` of the ``n-1``
    co-players (AU excluded) chose SAFE, ``0 <= k < n``.

    Unaffected by ``gamma`` -- AU always bears its own personal disaster risk
    ``pr`` regardless of whether co-players share in the consequences.
    """

    if not 0 <= k < n:
        raise ValueError("k must be between 0 and n - 1 inclusive")
    n_unsafe_including_self = n - k
    prize_term = s * B / (W * n_unsafe_including_self)
    stage_term = stage_payoff_unsafe(k, n=n, s=s, b=b, pfo=pfo)
    return (1.0 - pr) * (prize_term + stage_term)


def average_payoff_cs_vs_au(
    k: int,
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
) -> float:
    """``Pi_CS,AU(k)``: CS's average payoff when ``k`` of ``n`` group members
    (CS itself included) chose SAFE in round 1, ``1 <= k <= n``.

    CS plays SAFE in round 1, then mirrors what it saw; against AU it ends up
    playing UNSAFE (i.e. ``pi(0)_UNSAFE``-rate) for the remaining
    ``W/s - 1`` rounds. ``k == n`` is again the all-SAFE state, identical to
    ``average_payoff_as_vs_au``'s ``k == n`` branch.
    """

    if not 1 <= k <= n:
        raise ValueError("k must be between 1 and n inclusive")
    if k == n:
        return B / (n * W) + stage_payoff_safe(n, n=n, s=s, b=b, c=c, pfo=pfo)
    horizon_ratio = W / s
    base = (s / W) * stage_payoff_safe(k, n=n, s=s, b=b, c=c, pfo=pfo) + (
        horizon_ratio - 1.0
    ) * stage_payoff_unsafe(0, n=n, s=s, b=b, pfo=pfo)
    return (1.0 - pr * gamma) * base


def average_payoff_au_vs_cs(
    k: int,
    *,
    n: int,
    s: float,
    b: float,
    B: float,
    W: float,
    pr: float = 0.0,
    pfo: float = 0.0,
) -> float:
    """``Pi_AU,CS(k)``: AU's average payoff when ``k`` of the ``n-1`` CS
    co-players played SAFE in round 1, ``0 <= k < n``.

    AU wins the race in ``W/s`` rounds; ``k`` counts how many CS co-players
    were still SAFE (round 1) when AU met them, which sets the round-1 share
    of the prize. Unaffected by ``gamma``, same reasoning as
    ``average_payoff_au_vs_as``.
    """

    if not 0 <= k < n:
        raise ValueError("k must be between 0 and n - 1 inclusive")
    n_unsafe_including_self = n - k
    prize_term = s * B / (W * n_unsafe_including_self)
    horizon_ratio = W / s
    stage_term = (s / W) * stage_payoff_unsafe(k, n=n, s=s, b=b, pfo=pfo) + (
        horizon_ratio - 1.0
    ) * stage_payoff_unsafe(0, n=n, s=s, b=b, pfo=pfo)
    return (1.0 - pr) * (prize_term + stage_term)


def homogeneous_payoff(
    strategy: Strategy,
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    B: float,
    W: float,
    pr: float = 0.0,
    pfo: float = 0.0,
) -> float:
    """Payoff to a player using ``strategy`` in a population where every group
    member uses the same strategy -- ``Pi_AS,AS == Pi_CS,CS`` (both equal the
    all-SAFE branch shared with :func:`average_payoff_as_vs_au`'s ``k == n``
    case) and ``Pi_AU,AU`` (every group member UNSAFE, so all ``n`` players
    tie for the prize).

    Used by :func:`social_welfare` for the homogeneous end states that the
    small-mutation-limit population spends almost all its time in.
    """

    if strategy in ("AS", "CS"):
        return B / (n * W) + stage_payoff_safe(n, n=n, s=s, b=b, c=c, pfo=pfo)
    if strategy == "AU":
        prize_term = s * B / (W * n)
        stage_term = stage_payoff_unsafe(0, n=n, s=s, b=b, pfo=pfo)
        return (1.0 - pr) * (prize_term + stage_term)
    raise ValueError(f"unknown strategy {strategy!r}; expected 'AS', 'AU', or 'CS'")


def social_welfare(
    stationary: Mapping[str, float],
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    B: float,
    W: float,
    pr: float = 0.0,
    pfo: float = 0.0,
) -> float:
    """Population-average payoff (Appendix E's "average population payoff" /
    social welfare) implied by a small-mutation stationary distribution.

    In the small-mutation limit the population spends almost all its time in
    one of the homogeneous states, so the population-average payoff is the
    stationary-weighted average of :func:`homogeneous_payoff` over each
    strategy -- same reasoning as
    ``ai_race.theory.evolution.expected_unsafe_frequency``, applied to payoff
    instead of Unsafe frequency.
    """

    missing = set(stationary) - {"AS", "AU", "CS"}
    if missing:
        raise ValueError(f"unknown strategies in stationary distribution: {missing}")
    return sum(
        weight
        * homogeneous_payoff(
            strategy, n=n, s=s, b=b, c=c, B=B, W=W, pr=pr, pfo=pfo  # type: ignore[arg-type]
        )
        for strategy, weight in stationary.items()
        if weight > 0.0
    )
