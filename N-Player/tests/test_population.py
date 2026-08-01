"""Tests for theory.population: hypergeometric group sampling, fixation
probability, and the small-mutation stationary distribution (Appendix B,
Eq. 29-31).

Ground truth comes from three places: (1) direct combinatorial identities for
the hypergeometric pmf, (2) the paper's own stated neutral-drift limit
("when beta = 0, rho = 1/Z"), and (3) an exact reduction to the existing
two-player machinery in ``ai_race/theory/evolution.py`` at N=2 -- group
sampling of size 2 from a population is, by construction, the same average
the two-player module computes directly.
"""
from __future__ import annotations

import pytest

from theory.population import (
    average_payoff_i,
    average_payoff_j,
    fitness_in_population,
    fixation_probability,
    hypergeometric_pmf,
    risk_dominant,
    small_mutation_stationary,
)


def test_hypergeometric_pmf_known_value():
    # C(3,1)*C(2,1)/C(5,2) = 3*2/10 = 0.6
    assert hypergeometric_pmf(1, 2, 3, 5) == pytest.approx(0.6)


def test_hypergeometric_pmf_sums_to_one_over_k():
    total = sum(hypergeometric_pmf(k, 4, 6, 10) for k in range(0, 5))
    assert total == pytest.approx(1.0)


def test_hypergeometric_pmf_rejects_invalid_inputs():
    with pytest.raises(ValueError):
        hypergeometric_pmf(-1, 4, 6, 10)
    with pytest.raises(ValueError):
        hypergeometric_pmf(5, 4, 6, 10)
    with pytest.raises(ValueError):
        hypergeometric_pmf(0, 4, 11, 10)
    with pytest.raises(ValueError):
        hypergeometric_pmf(0, 11, 6, 10)


def test_average_payoff_constant_function_is_normalisation_invariant():
    # If every group composition pays the same constant, the weighted average
    # must return that constant regardless of x, z, n.
    assert average_payoff_i(lambda k: 7.0, x=4, z=20, n=5) == pytest.approx(7.0)
    assert average_payoff_j(lambda k: 7.0, x=4, z=20, n=5) == pytest.approx(7.0)


def test_average_payoff_i_rejects_x_below_one():
    with pytest.raises(ValueError):
        average_payoff_i(lambda k: 1.0, x=0, z=10, n=3)


def test_average_payoff_j_rejects_x_at_population_size():
    with pytest.raises(ValueError):
        average_payoff_j(lambda k: 1.0, x=10, z=10, n=3)


def test_average_payoff_reduces_to_pairwise_formula_at_n_equals_two():
    """At N=2, Eq. 29 must collapse to the main text's Eq. 3:
    PA(k) = [(k-1) Pi_AA + (Z-k) Pi_AB] / (Z-1)."""

    pi_aa, pi_ab = 3.0, 1.5
    z, x = 20, 7

    def payoff_i_of_k(k: int) -> float:
        # k=2 means both group members (including self) are type-i -> AA.
        # k=1 means only self is type-i -> AB.
        return pi_aa if k == 2 else pi_ab

    expected = ((x - 1) * pi_aa + (z - x) * pi_ab) / (z - 1)
    assert average_payoff_i(payoff_i_of_k, x=x, z=z, n=2) == pytest.approx(expected)


def test_fitness_in_population_matches_two_player_evolution_module():
    """Cross-check against ai_race.theory.evolution.fitness_in_population for
    a concrete 2x2 payoff matrix, translated into payoff-of-k functions."""

    from ai_race.theory.evolution import fitness_in_population as pairwise_fitness

    payoff_matrix = {
        ("A", "A"): 2.0,
        ("A", "B"): 0.5,
        ("B", "A"): 3.0,
        ("B", "B"): 1.0,
    }

    # Both functions use the same self-inclusive-own-count convention (k=1..n):
    # k=2 means both group members (including self) are the focal's own type.
    def a_payoff_of_k(k: int) -> float:
        return payoff_matrix[("A", "A")] if k == 2 else payoff_matrix[("A", "B")]

    def b_payoff_of_k(k: int) -> float:
        return payoff_matrix[("B", "B")] if k == 2 else payoff_matrix[("B", "A")]

    z, x = 15, 6
    expected_a, expected_b = pairwise_fitness(payoff_matrix, "B", "A", x, z)
    got_a, got_b = fitness_in_population(a_payoff_of_k, b_payoff_of_k, x=x, z=z, n=2)
    assert got_a == pytest.approx(expected_a)
    assert got_b == pytest.approx(expected_b)


def test_fixation_probability_neutral_drift_at_beta_zero():
    # Paper: "when beta = 0, rho_{B,A} = 1/Z, representing the transition
    # probability at neutral limit."
    z = 40
    rho = fixation_probability(
        lambda k: 5.0, lambda k: -3.0, z=z, n=4, beta=0.0
    )
    assert rho == pytest.approx(1.0 / z)


def test_fixation_probability_equal_fitness_is_neutral_regardless_of_beta():
    z = 30
    rho = fixation_probability(
        lambda k: 4.0, lambda k: 4.0, z=z, n=3, beta=2.0
    )
    assert rho == pytest.approx(1.0 / z)


def test_fixation_probability_favours_higher_payoff_mutant():
    # A mutant strategy that always outperforms the resident should fixate
    # with probability well above the neutral 1/Z.
    z = 30
    rho = fixation_probability(
        lambda k: 10.0, lambda k: 1.0, z=z, n=3, beta=1.0
    )
    assert rho > 1.0 / z


def test_risk_dominant_ties_favour_the_focal_strategy():
    assert risk_dominant(lambda k: 1.0, lambda k: 1.0, n=5)


def test_risk_dominant_detects_strict_advantage():
    assert risk_dominant(lambda k: 10.0, lambda k: 1.0, n=5)
    assert not risk_dominant(lambda k: 1.0, lambda k: 10.0, n=5)


def test_small_mutation_stationary_uniform_when_all_fitness_equal():
    payoff_of_k = {
        ("A", "A"): lambda k: 1.0,
        ("A", "B"): lambda k: 1.0,
        ("B", "A"): lambda k: 1.0,
        ("B", "B"): lambda k: 1.0,
    }
    stationary = small_mutation_stationary(
        payoff_of_k, ["A", "B"], z=20, n=3, beta=1.0
    )
    assert stationary["A"] == pytest.approx(0.5, abs=1e-6)
    assert stationary["B"] == pytest.approx(0.5, abs=1e-6)
    assert sum(stationary.values()) == pytest.approx(1.0)


def test_small_mutation_stationary_favours_dominant_strategy():
    payoff_of_k = {
        ("A", "A"): lambda k: 5.0,
        ("A", "B"): lambda k: 5.0,
        ("B", "A"): lambda k: 1.0,
        ("B", "B"): lambda k: 1.0,
    }
    stationary = small_mutation_stationary(
        payoff_of_k, ["A", "B"], z=30, n=3, beta=1.0
    )
    assert stationary["A"] > stationary["B"]


def test_small_mutation_stationary_rejects_single_strategy():
    with pytest.raises(ValueError):
        small_mutation_stationary({}, ["A"], z=20, n=3, beta=1.0)
