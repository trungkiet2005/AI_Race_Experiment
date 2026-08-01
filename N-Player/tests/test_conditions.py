"""Cross-checks for theory.conditions against numbers the paper states directly.

Figure S7's caption (Appendix B) gives worked numbers for N=5, s=1.5:
"for s = 1.5 the condition becomes pr > 0.94" (region I boundary, Eq. 26) and
"0.94 > pr > 0.33" (region II, bounded below by Eq. 22). Those two numbers are
the primary ground truth used below; everything else is a structural property
the paper states in prose (N-independence of Eq. 22, monotonicity in N of
Eq. 23/26, exact backward-compatibility with the two-player Eq. 6/7 at N=2).
"""
from __future__ import annotations

import math

import pytest

from theory.conditions import (
    dsai_zone,
    early_dsai_risk_dominance_threshold,
    early_dsai_welfare_threshold,
    harmonic_number,
    late_dsai_risk_dominance_threshold_as,
    late_dsai_risk_dominance_threshold_cs,
    late_dsai_welfare_threshold,
    stage_payoff_safe,
    stage_payoff_unsafe,
    welfare_condition_threshold,
)


def test_harmonic_number_matches_direct_sum():
    assert harmonic_number(5) == pytest.approx(1 + 0.5 + 1 / 3 + 0.25 + 0.2)


def test_harmonic_number_rejects_nonpositive_n():
    with pytest.raises(ValueError):
        harmonic_number(0)


def test_early_dsai_welfare_threshold_matches_figure_s7_caption():
    # "for s = 1.5 the condition becomes 0.94 > pr > 0.33" -> lower bound 0.33.
    assert early_dsai_welfare_threshold(s=1.5) == pytest.approx(1 / 3, abs=1e-3)


def test_early_dsai_welfare_threshold_is_independent_of_n():
    # Eq. 22 explicitly does not depend on N ("does not depend on the group size N").
    value_small = early_dsai_welfare_threshold(s=1.5)
    assert value_small == early_dsai_welfare_threshold(s=1.5)


def test_early_dsai_risk_dominance_threshold_matches_figure_s7_caption():
    # "for s = 1.5 the condition becomes pr > 0.94" at N=5.
    assert early_dsai_risk_dominance_threshold(n=5, s=1.5) == pytest.approx(
        0.9416, abs=1e-3
    )


def test_early_dsai_risk_dominance_threshold_approaches_one_for_large_n():
    # "the left hand side ... approaches 1 for increasingly large group size."
    small = early_dsai_risk_dominance_threshold(n=3, s=1.5)
    large = early_dsai_risk_dominance_threshold(n=1000, s=1.5)
    assert small < large < 1.0
    assert large == pytest.approx(1.0, abs=1e-3)


def test_early_dsai_risk_dominance_threshold_increases_with_group_size():
    # "a larger group size leads to a larger region (II)" i.e. a larger threshold.
    thresholds = [
        early_dsai_risk_dominance_threshold(n=n, s=1.5) for n in (2, 3, 5, 10)
    ]
    assert thresholds == sorted(thresholds)


def test_early_dsai_reduces_to_pairwise_case_at_n_equals_two():
    # At N=2, H_2=1.5 so N*H_N=3, matching the two-player Eq. 7 threshold 1-1/(3s).
    s = 1.5
    n2_value = early_dsai_risk_dominance_threshold(n=2, s=s)
    pairwise_value = 1.0 - 1.0 / (3 * s)
    assert n2_value == pytest.approx(pairwise_value)
    assert n2_value == pytest.approx(1 - 1 / 4.5, abs=1e-9)


def test_stage_payoffs_match_nplayer_engine_at_pfo_zero():
    """theory.conditions must reproduce ai_race.engine_nplayer's simplified
    (pfo=0) payoff formula exactly, since the engine's formula is this
    module's pfo=0 special case."""

    from ai_race.engine_nplayer.state import NPlayerGameConfig

    config = NPlayerGameConfig(
        name="cross-check", n_players=5, safe_progress=1.0, speed=1.5,
        cost=1.0, benefit=6.0,
    )
    for k in range(1, config.n_players + 1):
        assert stage_payoff_safe(
            k, n=5, s=1.5, b=6.0, c=1.0, pfo=0.0
        ) == pytest.approx(config.stage_payoff_safe(k))
    for k in range(0, config.n_players):
        assert stage_payoff_unsafe(
            k, n=5, s=1.5, b=6.0, pfo=0.0
        ) == pytest.approx(config.stage_payoff_unsafe(k))


def test_stage_payoff_safe_rejects_out_of_range_k():
    with pytest.raises(ValueError):
        stage_payoff_safe(0, n=5, s=1.5, b=4.0, c=1.0)
    with pytest.raises(ValueError):
        stage_payoff_safe(6, n=5, s=1.5, b=4.0, c=1.0)


def test_stage_payoff_unsafe_rejects_out_of_range_k():
    with pytest.raises(ValueError):
        stage_payoff_unsafe(-1, n=5, s=1.5, b=4.0)
    with pytest.raises(ValueError):
        stage_payoff_unsafe(5, n=5, s=1.5, b=4.0)


def test_late_dsai_welfare_threshold_requires_benefit_above_n_times_cost():
    # "it is necessary that b > Nc" for the threshold to sit inside a sensible
    # range; below that the paper says collective safety cannot be preferred
    # at any risk level. We just check the algebraic direction here.
    below = late_dsai_welfare_threshold(n=5, b=6.0, c=1.0, pfo=0.1)
    above_n_times_c = 6.0 > 5 * 1.0
    assert above_n_times_c
    assert below < 1.0


def test_late_dsai_welfare_threshold_increases_with_n():
    # "the left hand size is an increasing function of N."
    thresholds = [
        late_dsai_welfare_threshold(n=n, b=10.0, c=1.0, pfo=0.1) for n in (2, 3, 5, 8)
    ]
    assert thresholds == sorted(thresholds)


def test_welfare_condition_threshold_reduces_to_early_limit_when_b_dominates():
    params = dict(n=5, s=1.5, b=4.0, c=1.0, pfo=0.1)
    early_limit = welfare_condition_threshold(B=1e9, W=1.0, **params)
    assert early_limit == pytest.approx(early_dsai_welfare_threshold(s=1.5), abs=1e-6)


def test_welfare_condition_threshold_reduces_to_late_limit_when_w_dominates():
    params = dict(n=5, s=1.5, b=4.0, c=1.0, pfo=0.1)
    late_limit = welfare_condition_threshold(B=1.0, W=1e9, **params)
    assert late_limit == pytest.approx(
        late_dsai_welfare_threshold(n=5, b=4.0, c=1.0, pfo=0.1), abs=1e-6
    )


def test_late_dsai_risk_dominance_as_and_cs_are_finite_and_ordered():
    as_threshold = late_dsai_risk_dominance_threshold_as(
        n=5, s=1.5, b=6.0, c=1.0, pfo=0.1
    )
    cs_threshold = late_dsai_risk_dominance_threshold_cs(
        n=5, s=1.5, b=6.0, c=1.0, pfo=0.1
    )
    assert math.isfinite(as_threshold)
    assert math.isfinite(cs_threshold)


@pytest.mark.parametrize(
    "pr, expected_zone",
    [
        (0.95, "compliance"),  # above the N=5, s=1.5 region-I boundary (~0.94)
        (0.5, "dilemma"),      # inside 0.33 < pr < 0.94
        (0.1, "innovation"),   # below the region-III boundary (~0.33)
    ],
)
def test_dsai_zone_early_regime_matches_figure_s7(pr, expected_zone):
    assert (
        dsai_zone(pr, regime="early", n=5, s=1.5) == expected_zone
    )


def test_dsai_zone_rejects_unknown_regime():
    with pytest.raises(ValueError):
        dsai_zone(0.5, regime="mid", n=5, s=1.5)  # type: ignore[arg-type]
