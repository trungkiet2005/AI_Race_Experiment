"""Integration tests for theory.stationary: end-to-end AS/AU/CS stationary
distribution built from theory.welfare's payoff functions through
theory.population's group-sampling machinery.

These are the tests that would have caught the mutant/resident convention
bug fixed in theory/population.py's fitness_in_population -- Phase 2's own
unit tests all used symmetric toy functions (constants, or hand-built pairs
that happened to be built consistently), which cannot expose a convention
mismatch. Real strategy payoffs (SAFE-side vs AU) are asymmetric by
construction, so this is the first place the bug could actually show up.
"""
from __future__ import annotations

import pytest

from theory.stationary import au_frequency, build_payoff_lookup, stationary_distribution
from theory.welfare import average_payoff_as_vs_au, average_payoff_au_vs_as

PARAMS = dict(n=5, s=1.5, b=4.0, c=1.0, B=10_000.0, W=100.0, pfo=0.5)


def test_build_payoff_lookup_as_vs_au_matches_welfare_directly():
    lookup = build_payoff_lookup(pr=0.6, **PARAMS)
    for k in range(1, PARAMS["n"] + 1):
        assert lookup[("AS", "AU")](k) == pytest.approx(
            average_payoff_as_vs_au(k, pr=0.6, **PARAMS)
        )


def test_build_payoff_lookup_au_vs_as_applies_n_minus_k_transform():
    lookup = build_payoff_lookup(pr=0.6, **PARAMS)
    n = PARAMS["n"]
    for k in range(1, n + 1):
        assert lookup[("AU", "AS")](k) == pytest.approx(
            average_payoff_au_vs_as(n - k, pr=0.6, **{
                key: value for key, value in PARAMS.items() if key != "c"
            })
        )


def test_build_payoff_lookup_as_and_cs_are_constant_and_equal():
    lookup = build_payoff_lookup(pr=0.6, **PARAMS)
    n = PARAMS["n"]
    as_values = {lookup[("AS", "CS")](k) for k in range(1, n + 1)}
    cs_values = {lookup[("CS", "AS")](k) for k in range(1, n + 1)}
    assert len(as_values) == 1
    assert len(cs_values) == 1
    assert as_values == cs_values


def test_stationary_distribution_sums_to_one():
    stationary = stationary_distribution(z=100, beta=0.1, pr=0.6, **PARAMS)
    assert sum(stationary.values()) == pytest.approx(1.0)
    assert set(stationary) == {"AS", "AU", "CS"}


def test_stationary_distribution_favours_au_deep_in_innovation_zone():
    # n=5, s=1.5: early-DSAI innovation boundary is pr < 1 - 1/1.5 = 0.333
    # (theory.conditions.early_dsai_welfare_threshold). pr=0.05 is deep inside.
    stationary = stationary_distribution(z=100, beta=0.5, pr=0.05, **PARAMS)
    assert stationary["AU"] > 0.9


def test_stationary_distribution_favours_safety_deep_in_compliance_zone():
    # n=5, s=1.5: early-DSAI compliance boundary is pr > ~0.9416
    # (theory.conditions.early_dsai_risk_dominance_threshold). pr=0.99 is
    # deep inside.
    stationary = stationary_distribution(z=100, beta=0.5, pr=0.99, **PARAMS)
    assert stationary["AS"] + stationary["CS"] > 0.9
    assert stationary["AU"] < 0.1


def test_au_frequency_matches_stationary_au_key():
    stationary = stationary_distribution(z=100, beta=0.3, pr=0.6, **PARAMS)
    assert au_frequency(z=100, beta=0.3, pr=0.6, **PARAMS) == pytest.approx(
        stationary["AU"]
    )


def test_stationary_distribution_two_strategy_subset_matches_dominant_zone():
    # Restricting to {AS, AU} only (no CS) should show the same qualitative
    # shift between innovation and compliance zones.
    innovation = stationary_distribution(
        z=100, beta=0.5, pr=0.05, strategies=("AS", "AU"), **PARAMS
    )
    compliance = stationary_distribution(
        z=100, beta=0.5, pr=0.99, strategies=("AS", "AU"), **PARAMS
    )
    assert innovation["AU"] > innovation["AS"]
    assert compliance["AS"] > compliance["AU"]
