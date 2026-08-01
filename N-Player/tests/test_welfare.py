"""Tests for theory.welfare: the Pi_AS,AU(k)/Pi_AU,AS(k)/Pi_CS,AU(k)/Pi_AU,CS(k)
family (Appendix B), gamma-scaling (Appendix C), and social welfare
(Appendix E).

Ground truth is the two-player main-text Eq. 2, worked out by hand for each
function at N=2 (see the module docstring in theory/welfare.py for why this
is checked against Eq. 2 directly rather than against
``ai_race/theory/payoffs.py``, which uses a different, total-payoff scale).
"""
from __future__ import annotations

import pytest

from theory.conditions import stage_payoff_safe, stage_payoff_unsafe
from theory.welfare import (
    average_payoff_as_vs_au,
    average_payoff_au_vs_as,
    average_payoff_au_vs_cs,
    average_payoff_cs_vs_au,
    expected_horizon,
    homogeneous_payoff,
    social_welfare,
)

N, S, B_STAGE, C, B_PRIZE, W = 2, 1.5, 4.0, 1.0, 100.0, 9.0
PR, PFO = 0.2, 0.0

PI_12 = stage_payoff_safe(1, n=N, s=S, b=B_STAGE, c=C, pfo=PFO)  # AS meets AU
PI_21 = stage_payoff_unsafe(1, n=N, s=S, b=B_STAGE, pfo=PFO)  # AU meets AS
PI_22 = stage_payoff_unsafe(0, n=N, s=S, b=B_STAGE, pfo=PFO)  # AU meets AU
PI_11 = stage_payoff_safe(2, n=N, s=S, b=B_STAGE, c=C, pfo=PFO)  # AS meets AS


def test_average_payoff_as_vs_au_all_safe_branch():
    value = average_payoff_as_vs_au(
        5, n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W, pr=PR
    )
    expected = B_PRIZE / (5 * W) + stage_payoff_safe(5, n=5, s=S, b=B_STAGE, c=C)
    assert value == pytest.approx(expected)


def test_average_payoff_as_vs_au_rejects_out_of_range_k():
    with pytest.raises(ValueError):
        average_payoff_as_vs_au(0, n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W)
    with pytest.raises(ValueError):
        average_payoff_as_vs_au(6, n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W)


def test_average_payoff_as_vs_au_reduces_to_two_player_eq2():
    # Eq. 2: Pi_AS,AU = pi12, no prize term and (at gamma=0) no risk scaling.
    value = average_payoff_as_vs_au(
        1, n=N, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W, pr=PR, gamma=0.0
    )
    assert value == pytest.approx(PI_12)


def test_average_payoff_as_vs_au_gamma_scaling_appendix_c():
    # Appendix C: "payoffs of AS ... when playing with AU is scaled by (1 - pr*gamma)".
    value = average_payoff_as_vs_au(
        1, n=N, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W, pr=PR, gamma=1.0
    )
    assert value == pytest.approx((1.0 - PR) * PI_12)


def test_average_payoff_au_vs_as_reduces_to_two_player_eq2():
    # Eq. 2: Pi_AU,AS = (1 - pr) [ sB/W + pi21 ].
    value = average_payoff_au_vs_as(1, n=N, s=S, b=B_STAGE, B=B_PRIZE, W=W, pr=PR)
    expected = (1.0 - PR) * (S * B_PRIZE / W + PI_21)
    assert value == pytest.approx(expected)


def test_average_payoff_au_vs_as_rejects_out_of_range_k():
    with pytest.raises(ValueError):
        average_payoff_au_vs_as(-1, n=5, s=S, b=B_STAGE, B=B_PRIZE, W=W)
    with pytest.raises(ValueError):
        average_payoff_au_vs_as(5, n=5, s=S, b=B_STAGE, B=B_PRIZE, W=W)


def test_average_payoff_cs_vs_au_reduces_to_two_player_eq2():
    # Eq. 2: Pi_CS,AU = s/W pi12 + (W/s - 1) pi22.
    value = average_payoff_cs_vs_au(
        1, n=N, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W, pr=PR, gamma=0.0
    )
    expected = (S / W) * PI_12 + (W / S - 1.0) * PI_22
    assert value == pytest.approx(expected)


def test_average_payoff_cs_vs_au_all_safe_branch_matches_as_vs_au():
    cs_value = average_payoff_cs_vs_au(5, n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W)
    as_value = average_payoff_as_vs_au(5, n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W)
    assert cs_value == pytest.approx(as_value)


def test_average_payoff_au_vs_cs_reduces_to_two_player_eq2():
    # Eq. 2: Pi_AU,CS = (1 - pr) [ sB/W + s/W pi21 + (W/s - 1) pi22 ].
    value = average_payoff_au_vs_cs(1, n=N, s=S, b=B_STAGE, B=B_PRIZE, W=W, pr=PR)
    expected = (1.0 - PR) * (
        S * B_PRIZE / W + (S / W) * PI_21 + (W / S - 1.0) * PI_22
    )
    assert value == pytest.approx(expected)


def test_average_payoff_au_vs_cs_rejects_out_of_range_k():
    with pytest.raises(ValueError):
        average_payoff_au_vs_cs(-1, n=5, s=S, b=B_STAGE, B=B_PRIZE, W=W)
    with pytest.raises(ValueError):
        average_payoff_au_vs_cs(5, n=5, s=S, b=B_STAGE, B=B_PRIZE, W=W)


def test_homogeneous_payoff_as_and_cs_are_equal():
    as_value = homogeneous_payoff("AS", n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W)
    cs_value = homogeneous_payoff("CS", n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W)
    assert as_value == pytest.approx(cs_value)


def test_homogeneous_payoff_au_matches_hand_derivation():
    value = homogeneous_payoff("AU", n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W, pr=PR)
    expected = (1.0 - PR) * (
        S * B_PRIZE / (W * 5) + stage_payoff_unsafe(0, n=5, s=S, b=B_STAGE)
    )
    assert value == pytest.approx(expected)


def test_homogeneous_payoff_rejects_unknown_strategy():
    with pytest.raises(ValueError):
        homogeneous_payoff("CAS", n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W)  # type: ignore[arg-type]


def test_social_welfare_is_stationary_weighted_average():
    stationary = {"AS": 0.3, "AU": 0.5, "CS": 0.2}
    params = dict(n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W, pr=PR)
    expected = sum(
        weight * homogeneous_payoff(strategy, **params)  # type: ignore[arg-type]
        for strategy, weight in stationary.items()
    )
    assert social_welfare(stationary, **params) == pytest.approx(expected)


def test_social_welfare_ignores_zero_weight_strategies():
    params = dict(n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W, pr=PR)
    full = social_welfare({"AS": 1.0, "AU": 0.0, "CS": 0.0}, **params)
    as_only = homogeneous_payoff("AS", **params)
    assert full == pytest.approx(as_only)


def test_social_welfare_rejects_unknown_strategy_key():
    with pytest.raises(ValueError):
        social_welfare(
            {"CAS": 1.0}, n=5, s=S, b=B_STAGE, c=C, B=B_PRIZE, W=W, pr=PR
        )


def test_expected_horizon_matches_two_player_theory_module():
    from ai_race.theory.payoffs import expected_horizon as two_player_expected_horizon
    from ai_race.engine.state import GameConfig

    config = GameConfig(
        name="cross-check",
        payoff_safe_safe=1.0,
        payoff_safe_unsafe=0.6,
        payoff_unsafe_safe=2.4,
        payoff_unsafe_unsafe=2.0,
        min_rounds=5,
        stop_probability=0.2,
    )
    assert expected_horizon(
        min_rounds=5, stop_probability=0.2
    ) == pytest.approx(two_player_expected_horizon(config))


def test_expected_horizon_rejects_invalid_stop_probability():
    with pytest.raises(ValueError):
        expected_horizon(min_rounds=5, stop_probability=0.0)
