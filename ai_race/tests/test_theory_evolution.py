"""Acceptance tests for the small-mutation evolutionary limit.

The limit reproduces the source paper's high-risk phase — CS taking over above
roughly ``p_r^max = 0.6``, and Always Safe holding negligible mass everywhere — but
it provably *cannot* separate AU from CAS. Those two are exactly payoff-equivalent
against each other, so their mutual fixation probabilities are both the neutral
``1/Z`` and the limit splits them evenly at every treatment below the CS threshold.
Separating them is a finite-mutation result and needs the full chain.

The tests below assert what the limit actually delivers, including that
degeneracy. Asserting the paper's AU-then-CAS ordering here would mean asserting
something the method cannot produce.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai_race.engine.state import GameConfig
from ai_race.theory.evolution import (
    MUTATION_REGIME,
    expected_unsafe_frequency,
    fitness_in_population,
    fixation_probability,
    small_mutation_stationary,
)
from ai_race.theory.payoffs import (
    STRATEGY_ORDER,
    expected_payoff_matrix,
    self_play_unsafe_frequency,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "ai_race" / "configs" / "game"
Z = 100
BETA = 2.0


def at_risk(risk: float) -> GameConfig:
    config = GameConfig.from_dict(
        json.loads(
            (CONFIG_DIRECTORY / "ai_race_risk_10.json").read_text(encoding="utf-8")
        )
    )
    return replace(config, max_private_risk=risk)


def stationary_at(risk: float, *, beta: float = BETA) -> dict[str, float]:
    return small_mutation_stationary(
        expected_payoff_matrix(at_risk(risk)),
        Z=Z,
        beta=beta,
    )


def test_fitness_excludes_self_interaction():
    matrix = {
        ("A", "A"): 10.0,
        ("A", "B"): 0.0,
        ("B", "A"): 20.0,
        ("B", "B"): 5.0,
    }
    mutant, resident = fitness_in_population(matrix, "B", "A", 3, 5)
    # Mutant A meets 2 other A and 2 B, out of Z - 1 = 4 partners.
    assert mutant == pytest.approx((2 * 10.0 + 2 * 0.0) / 4)
    # Resident B meets 3 A and 1 other B.
    assert resident == pytest.approx((3 * 20.0 + 1 * 5.0) / 4)

    with pytest.raises(ValueError):
        fitness_in_population(matrix, "B", "A", 0, 5)
    with pytest.raises(ValueError):
        fitness_in_population(matrix, "B", "A", 5, 5)


def test_neutral_fixation_is_one_over_z():
    """A strategy that cannot be told apart from the resident drifts neutrally."""

    matrix = {
        (own, opponent): 7.0 for own in ("A", "B") for opponent in ("A", "B")
    }
    assert fixation_probability(matrix, "B", "A", Z=Z, beta=BETA) == pytest.approx(
        1.0 / Z
    )
    assert fixation_probability(matrix, "A", "A", Z=Z, beta=BETA) == pytest.approx(
        1.0 / Z
    )


def test_fixation_is_ordered_by_advantage():
    advantaged = {
        ("A", "A"): 10.0,
        ("A", "B"): 10.0,
        ("B", "A"): 1.0,
        ("B", "B"): 1.0,
    }
    strong = fixation_probability(advantaged, "A", "B", Z=Z, beta=BETA)
    weak = fixation_probability(advantaged, "B", "A", Z=Z, beta=BETA)
    assert strong > 1.0 / Z > weak
    assert 0.0 <= weak < strong <= 1.0


def test_fixation_survives_the_payoff_scale_of_this_game():
    """Payoffs here reach 120, so ``beta * delta_f`` overflows a naive product.

    Accumulating the products directly would give ``inf`` and a fixation
    probability of exactly zero — plausible-looking, and wrong.
    """

    matrix = expected_payoff_matrix(at_risk(0.1))
    probability = fixation_probability(matrix, "AS", "AU", Z=Z, beta=BETA)
    assert 0.0 <= probability < 1.0 / Z
    # Strong selection against a hopeless mutant, but still a finite number.
    assert probability == probability  # not NaN


def test_stationary_distribution_is_a_distribution():
    for risk in (0.1, 0.6, 0.9):
        stationary = stationary_at(risk)
        assert set(stationary) == set(STRATEGY_ORDER)
        assert sum(stationary.values()) == pytest.approx(1.0)
        assert all(0.0 <= value <= 1.0 for value in stationary.values())


@pytest.mark.parametrize("risk", [0.1, 0.6, 0.9])
def test_always_safe_holds_negligible_stationary_mass(risk):
    """Figure S8's conclusion: AS is squeezed out at every treatment."""

    assert stationary_at(risk)["AS"] < 0.05


@pytest.mark.parametrize("risk", [0.1, 0.6, 0.9])
def test_nothing_selects_between_always_unsafe_and_cas(risk):
    """The documented limitation of the small-mutation reduction.

    ``Pi(CAS, AU) = Pi(AU, AU)`` and ``Pi(AU, CAS) = Pi(CAS, CAS)``, so a mutant of
    either invades a population of the other at exactly the neutral rate ``1/Z``,
    for every treatment and every selection strength. Which of the two dominates —
    the paper's transition at ``p_r^max ~ 0.2`` — is a finite-mutation result that
    this method cannot reach.
    """

    matrix = expected_payoff_matrix(at_risk(risk))
    for beta in (0.01, 2.0, 10.0):
        assert fixation_probability(
            matrix, "CAS", "AU", Z=Z, beta=beta
        ) == pytest.approx(1.0 / Z)
        assert fixation_probability(
            matrix, "AU", "CAS", Z=Z, beta=beta
        ) == pytest.approx(1.0 / Z)


@pytest.mark.parametrize("risk", [0.1, 0.6])
def test_the_degenerate_pair_splits_its_mass_evenly(risk):
    """Below the CS transition the two neutral states hold everything, 50/50.

    Above it the split is set by the outflow from CS, which is not symmetric, so
    this only holds where AU and CAS are the whole distribution.
    """

    stationary = stationary_at(risk)
    assert stationary["AU"] == pytest.approx(stationary["CAS"], abs=1e-9)
    assert stationary["AU"] == pytest.approx(0.5, abs=1e-3)


def test_the_high_risk_phase_transition_is_reproduced():
    """Figure S5's upper transition: CS takes over above roughly 0.6."""

    assert stationary_at(0.1)["CS"] < 0.05
    assert stationary_at(0.6)["CS"] < 0.05
    assert stationary_at(0.9)["CS"] > 0.9
    # Monotone in between, and located above the paper's 0.6 boundary.
    assert stationary_at(0.5)["CS"] < stationary_at(0.7)["CS"] < stationary_at(0.9)["CS"]


def test_unsafe_dominant_strategies_hold_the_mass_below_the_transition():
    """The behavioural claim the limit does support, phrased over the pair."""

    for risk in (0.1, 0.6):
        stationary = stationary_at(risk)
        assert stationary["AU"] + stationary["CAS"] > 0.95
    high = stationary_at(0.9)
    assert high["AU"] + high["CAS"] < 0.05


def test_stronger_selection_concentrates_the_distribution():
    """Figure S6's direction: raising beta sharpens the outcome."""

    weak = stationary_at(0.9, beta=0.01)
    strong = stationary_at(0.9, beta=2.0)
    assert strong["CS"] > weak["CS"]
    assert max(strong.values()) > max(weak.values())


def test_expected_unsafe_frequency_falls_with_the_treatment():
    rates = {}
    for risk in (0.1, 0.6, 0.9):
        config = at_risk(risk)
        stationary = stationary_at(risk)
        per_strategy = {
            name: self_play_unsafe_frequency(config, name) for name in STRATEGY_ORDER
        }
        rates[risk] = expected_unsafe_frequency(stationary, per_strategy)
    assert rates[0.1] > 0.95
    assert rates[0.6] > 0.95
    assert rates[0.9] < 0.05
    assert rates[0.1] >= rates[0.6] > rates[0.9]


def test_expected_unsafe_frequency_refuses_to_guess_a_missing_rate():
    """CS and CAS have no intrinsic Unsafe rate, so defaulting one would be a lie."""

    stationary = stationary_at(0.1)
    with pytest.raises(ValueError, match="must be measured"):
        expected_unsafe_frequency(stationary, {"AS": 0.0, "AU": 1.0})


def test_the_regime_is_labelled():
    assert MUTATION_REGIME == "small_mutation_limit"


def test_the_embedded_chain_needs_at_least_two_strategies():
    with pytest.raises(ValueError):
        small_mutation_stationary({("AU", "AU"): 1.0}, ["AU"], Z=Z, beta=BETA)
