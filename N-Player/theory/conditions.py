"""Closed-form N-team DSAI conditions.

Reproduces Appendix B of Han, Pereira, Santos & Lenaerts (JAIR 2020),
"N-player AI Race Definition" / "Analytical Conditions and DSAI Zones in
N-team Interactions": Eq. 21-28 and the resulting three-zone classification
(compliance / dilemma / innovation) shown in Figures S7-S9.

``pi_stage_safe``/``pi_stage_unsafe`` implement the *general* per-round payoff
from Appendix B, including the found-out probability ``pfo`` that the paper
defines but that ``ai_race.engine_nplayer`` deliberately drops (see its
README's "Deliberately out of scope" section) -- this module exists to
reproduce the paper's analysis on its own terms, independent of what the LLM
runner simulates. At ``pfo=0`` these two functions are exactly the formulas
implemented in ``ai_race.engine_nplayer.state.NPlayerGameConfig.stage_payoff_safe``/
``stage_payoff_unsafe``; see ``tests/test_conditions.py`` for the cross-check.

All thresholds are returned as raw numbers rather than clamped to ``[0, 1]``:
a threshold outside that range is a legitimate result (it means one regime
dominates for every risk probability) and clamping it would hide that.
"""
from __future__ import annotations

from typing import Literal

Regime = Literal["early", "late"]
Strategy = Literal["AS", "CS"]
Zone = Literal["compliance", "dilemma", "innovation"]


def harmonic_number(n: int) -> float:
    """``H_N = sum_{i=1}^{N} 1/i``, used by the early-DSAI risk-dominance bound (Eq. 26)."""

    if n < 1:
        raise ValueError("n must be a positive integer")
    return sum(1.0 / i for i in range(1, n + 1))


def stage_payoff_safe(
    k_safe: int,
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    pfo: float = 0.0,
) -> float:
    """``pi_SAFE(k)``: payoff to one SAFE player when ``k_safe`` of ``n`` chose SAFE.

    ``k_safe`` includes the player asking (``1 <= k_safe <= n``); ``k_safe == n``
    is the paper's special case (everyone SAFE) where the benefit is split evenly
    rather than weighted by speed, since there is no Unsafe player to weight against.
    """

    if not 1 <= k_safe <= n:
        raise ValueError("k_safe must be between 1 and n inclusive")
    if k_safe == n:
        return -c + b / n
    n_unsafe = n - k_safe
    share_weight = k_safe + s * n_unsafe
    return -c + (1.0 - pfo) * b / share_weight + pfo * b / k_safe


def stage_payoff_unsafe(
    k_safe: int,
    *,
    n: int,
    s: float,
    b: float,
    pfo: float = 0.0,
) -> float:
    """``pi_UNSAFE(k)``: payoff to one UNSAFE player when ``k_safe`` of ``n`` chose SAFE."""

    if not 0 <= k_safe < n:
        raise ValueError("k_safe must be between 0 and n - 1 inclusive")
    n_unsafe = n - k_safe
    share_weight = k_safe + s * n_unsafe
    return (1.0 - pfo) * s * b / share_weight


def welfare_condition_threshold(
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    B: float,
    W: float,
    pfo: float = 0.0,
) -> float:
    """Eq. 21: ``p_r`` above which all-SAFE has greater collective welfare than all-UNSAFE.

    ``Pi_AS,AU(N) > Pi_AU,AS(0)``, rewritten as a threshold on ``p_r``.
    """

    numerator = B + W * (b - n * c)
    denominator = s * B + W * (1.0 - pfo) * b
    return 1.0 - numerator / denominator


def early_dsai_welfare_threshold(*, s: float) -> float:
    """Eq. 22: early-DSAI reduction of Eq. 21 (``B/W >> b``).

    Identical to the two-player condition and independent of ``N`` and ``pfo``.
    """

    return 1.0 - 1.0 / s


def late_dsai_welfare_threshold(
    *,
    n: int,
    b: float,
    c: float,
    pfo: float = 0.0,
) -> float:
    """Eq. 23: late-DSAI reduction of Eq. 21 (``B/W << b``).

    Requires ``b > n*c`` for the collective-welfare case to be reachable at all;
    the threshold is strictly increasing in ``n`` (a larger race needs a higher
    disaster risk before collective safety pays off).
    """

    return 1.0 - (b - n * c) / ((1.0 - pfo) * b)


def early_dsai_risk_dominance_threshold(*, n: int, s: float) -> float:
    """Eq. 26: closed-form risk-dominance threshold of AS/CS against AU, early DSAI.

    Common to both AS and CS (the paper notes both reduce to the same bound
    against AU here). Approaches 1 as ``n`` grows without bound, since
    ``H_N > log(N)``.
    """

    h_n = harmonic_number(n)
    return 1.0 - 1.0 / (n * h_n * s)


def late_dsai_risk_dominance_threshold_as(
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    pfo: float = 0.0,
) -> float:
    """Eq. 27: late-DSAI risk-dominance threshold of AS against AU."""

    total_safe = sum(
        stage_payoff_safe(i, n=n, s=s, b=b, c=c, pfo=pfo) for i in range(1, n + 1)
    )
    total_unsafe = sum(
        stage_payoff_unsafe(i, n=n, s=s, b=b, pfo=pfo) for i in range(0, n)
    )
    return 1.0 - total_safe / total_unsafe


def late_dsai_risk_dominance_threshold_cs(
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    pfo: float = 0.0,
) -> float:
    """Eq. 28: late-DSAI risk-dominance threshold of CS against AU.

    Reduces to ``(N-1)*pi(0)_UNSAFE + pi(N)_SAFE`` in the numerator, which
    simplifies to the one-line form the paper gives, ``(1/N)(1 -
    pi(N)_SAFE/pi(0)_UNSAFE)``.
    """

    pi_n_safe = stage_payoff_safe(n, n=n, s=s, b=b, c=c, pfo=pfo)
    pi_0_unsafe = stage_payoff_unsafe(0, n=n, s=s, b=b, pfo=pfo)
    return (1.0 / n) * (1.0 - pi_n_safe / pi_0_unsafe)


def late_dsai_risk_dominance_threshold(
    strategy: Strategy,
    *,
    n: int,
    s: float,
    b: float,
    c: float,
    pfo: float = 0.0,
) -> float:
    """Dispatch to Eq. 27 (``strategy="AS"``) or Eq. 28 (``strategy="CS"``)."""

    if strategy == "AS":
        return late_dsai_risk_dominance_threshold_as(n=n, s=s, b=b, c=c, pfo=pfo)
    if strategy == "CS":
        return late_dsai_risk_dominance_threshold_cs(n=n, s=s, b=b, c=c, pfo=pfo)
    raise ValueError(f"unknown strategy {strategy!r}; expected 'AS' or 'CS'")


def dsai_zone(
    pr: float,
    *,
    regime: Regime,
    n: int,
    s: float,
    b: float = 4.0,
    c: float = 1.0,
    pfo: float = 0.0,
    strategy: Strategy = "CS",
) -> Zone:
    """Classify a risk probability ``pr`` into one of the three DSAI zones.

    - **compliance** (region I): safety is both the preferred collective outcome
      and the one selected by social dynamics.
    - **dilemma** (region II): safety is preferred collectively, but social
      dynamics selects Unsafe -- the zone where regulation is needed.
    - **innovation** (region III): Unsafe is both preferred and selected.

    ``regime="early"`` uses Eq. 22/26 (strategy-independent); ``regime="late"``
    uses Eq. 23 and Eq. 27/28 (``strategy`` picks AS or CS against AU).
    """

    if regime == "early":
        welfare_threshold = early_dsai_welfare_threshold(s=s)
        dominance_threshold = early_dsai_risk_dominance_threshold(n=n, s=s)
    elif regime == "late":
        welfare_threshold = late_dsai_welfare_threshold(n=n, b=b, c=c, pfo=pfo)
        dominance_threshold = late_dsai_risk_dominance_threshold(
            strategy, n=n, s=s, b=b, c=c, pfo=pfo
        )
    else:
        raise ValueError(f"unknown regime {regime!r}; expected 'early' or 'late'")

    if pr > dominance_threshold:
        return "compliance"
    if pr < welfare_threshold:
        return "innovation"
    return "dilemma"
