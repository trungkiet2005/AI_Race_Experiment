"""Acceptance tests for the closed-form and simulated payoff matrices.

Most of these compare against numbers published in the source paper's SI, which
makes them independent of this repository's code — the strongest kind of check
available here. They are written against values recomputed from ``GameConfig``
rather than against the rounded figures printed in the paper, so a mechanism change
in a configuration file surfaces as a failure instead of a silent mismatch.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai_race.engine.state import GameConfig
from ai_race.theory.payoffs import (
    STRATEGY_ORDER,
    exact_expected_payoff,
    expected_horizon,
    expected_payoff_matrix,
    horizon_distribution,
    matrix_to_rows,
    payoff_method,
    self_play_unsafe_frequency,
    simulate_matchup,
    unconditional_expected_payoff,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "ai_race" / "configs" / "game"


def load_config(name: str = "ai_race_risk_10") -> GameConfig:
    return GameConfig.from_dict(
        json.loads((CONFIG_DIRECTORY / f"{name}.json").read_text(encoding="utf-8"))
    )


def at_risk(risk: float) -> GameConfig:
    return replace(load_config(), max_private_risk=risk)


def test_expected_horizon_matches_the_paper():
    """SI 3.1: E[W] = 5 + (1 - 0.2) / 0.2 = 9."""

    config = load_config()
    assert expected_horizon(config) == pytest.approx(9.0)
    # Derived, not hardcoded: a different stop probability must move it.
    assert expected_horizon(replace(config, stop_probability=0.5)) == pytest.approx(6.0)


def test_stage_payoffs_are_the_papers_pi_vector():
    config = load_config()
    pi = (
        config.payoff_safe_safe,
        config.payoff_safe_unsafe,
        config.payoff_unsafe_safe,
        config.payoff_unsafe_unsafe,
    )
    assert pi == pytest.approx((1.0, 0.6, 2.4, 2.0))


def test_closed_form_payoffs_match_the_paper():
    """SI 3.1's four unconditional expected payoffs, rebuilt from the config."""

    config = load_config()
    horizon = expected_horizon(config)
    prize = config.race_prize
    risk = config.max_private_risk

    assert unconditional_expected_payoff(config, "AS", "AS") == pytest.approx(
        prize / 2 + horizon * config.payoff_safe_safe
    )
    assert unconditional_expected_payoff(config, "AS", "AS") == pytest.approx(59.0)

    # The loser never faces the risk lottery, so no (1 - p) factor here.
    assert unconditional_expected_payoff(config, "AS", "AU") == pytest.approx(
        horizon * config.payoff_safe_unsafe
    )
    assert unconditional_expected_payoff(config, "AS", "AU") == pytest.approx(5.4)

    assert unconditional_expected_payoff(config, "AU", "AS") == pytest.approx(
        (1 - risk) * (prize + horizon * config.payoff_unsafe_safe)
    )
    assert unconditional_expected_payoff(config, "AU", "AS") == pytest.approx(
        (1 - risk) * 121.6
    )

    assert unconditional_expected_payoff(config, "AU", "AU") == pytest.approx(
        (1 - risk) * (prize / 2 + horizon * config.payoff_unsafe_unsafe)
    )
    assert unconditional_expected_payoff(config, "AU", "AU") == pytest.approx(
        (1 - risk) * 68.0
    )


@pytest.mark.parametrize("risk", [0.1, 0.6, 0.9])
def test_closed_form_scales_with_the_treatment(risk):
    config = at_risk(risk)
    assert unconditional_expected_payoff(config, "AU", "AU") == pytest.approx(
        (1 - risk) * 68.0
    )
    # Mutual Safe never triggers the lottery, so it is flat in the treatment.
    assert unconditional_expected_payoff(config, "AS", "AS") == pytest.approx(59.0)


@pytest.mark.parametrize(
    "own,opponent",
    [("CS", "AU"), ("AU", "CS"), ("CAS", "AS"), ("CS", "CAS")],
)
def test_closed_form_refuses_conditional_strategies(own, opponent):
    """Silently returning a number here is the easiest failure to miss."""

    with pytest.raises(ValueError, match="no closed form"):
        unconditional_expected_payoff(load_config(), own, opponent)


def test_horizon_distribution_is_a_proper_pmf():
    config = load_config()
    atoms = horizon_distribution(config)
    assert sum(mass for _, mass in atoms) == pytest.approx(1.0)
    assert all(mass >= 0 for _, mass in atoms)
    assert atoms[0][0] == config.min_rounds
    assert atoms[-1][0] == config.max_rounds_safety_cap
    assert sum(horizon * mass for horizon, mass in atoms) == pytest.approx(
        expected_horizon(config), abs=1e-6
    )


@pytest.mark.parametrize("own", ["AS", "AU"])
@pytest.mark.parametrize("opponent", ["AS", "AU"])
def test_exact_enumeration_agrees_with_the_closed_form(own, opponent):
    """Two independent derivations of the same quantity.

    They differ only by the safety cap, which the closed form ignores and the
    enumeration includes; that tail carries about 1e-9 of probability mass.
    """

    config = load_config()
    assert exact_expected_payoff(config, own, opponent) == pytest.approx(
        unconditional_expected_payoff(config, own, opponent), abs=1e-6
    )


@pytest.mark.parametrize("own", ["AS", "AU"])
@pytest.mark.parametrize("opponent", ["AS", "AU"])
def test_monte_carlo_converges_to_the_closed_form(own, opponent):
    """The cross-check between the two computation routes.

    Far stronger than comparing each route to a constant: it fails if either the
    simulator or the algebra drifts, including if the simulator starts drawing the
    setback instead of taking its expectation.
    """

    config = load_config()
    simulated = simulate_matchup(
        config,
        own,
        opponent,
        replications=200_000,
        seed=260726,
    )
    closed = unconditional_expected_payoff(config, own, opponent)
    assert simulated == pytest.approx(closed, rel=0.01)


def test_conditional_antisocial_safe_is_indistinguishable_from_always_unsafe():
    """SI 3.4: CAS facing an Unsafe-from-round-one opponent just plays Unsafe.

    A structural identity, not an approximation. The four AU/CAS profiles stand or
    fall together downstream because of it, so it is asserted exactly.
    """

    for risk in (0.1, 0.6, 0.9):
        config = at_risk(risk)
        mutual_unsafe = exact_expected_payoff(config, "AU", "AU")
        assert exact_expected_payoff(config, "CAS", "AU") == pytest.approx(
            mutual_unsafe, abs=1e-12
        )
        assert exact_expected_payoff(config, "AU", "CAS") == pytest.approx(
            mutual_unsafe, abs=1e-12
        )
        assert exact_expected_payoff(config, "CAS", "CAS") == pytest.approx(
            mutual_unsafe, abs=1e-12
        )


def test_monte_carlo_reproduces_the_cas_identity_only_within_sampling_error():
    """Why the exact route is the default for the equilibrium search.

    Under sampling the identity holds only approximately, and an exhaustive
    best-response search reads the residual as a strict preference.
    """

    config = load_config()
    simulated = simulate_matchup(config, "CAS", "AU", replications=200_000, seed=1)
    exact = exact_expected_payoff(config, "AU", "AU")
    assert simulated == pytest.approx(exact, rel=0.01)
    assert simulated != pytest.approx(exact, abs=1e-9)


def test_monte_carlo_is_bit_for_bit_reproducible():
    config = load_config()
    first = simulate_matchup(config, "CS", "CAS", replications=2_000, seed=99)
    second = simulate_matchup(config, "CS", "CAS", replications=2_000, seed=99)
    assert first == second
    different = simulate_matchup(config, "CS", "CAS", replications=2_000, seed=100)
    assert different != first


def test_conditional_pairs_are_played_out_against_each_other():
    """CS versus CAS must alternate, not follow a fixed opponent sequence.

    CS opens Safe and CAS opens Unsafe; from round two each copies the other, so
    the two seats swap actions every round forever. Generating one side against a
    frozen opponent trajectory would produce a different, wrong game.
    """

    config = replace(load_config(), min_rounds=4, stop_probability=1.0)
    # Round 1: CS Safe, CAS Unsafe. Round 2: CS copies Unsafe, CAS copies Safe...
    # CS is Unsafe in rounds 2 and 4 of a four-round race.
    assert self_play_unsafe_frequency(config, "CS") == pytest.approx(0.0)
    cs_against_cas = exact_expected_payoff(config, "CS", "CAS")
    cas_against_cs = exact_expected_payoff(config, "CAS", "CS")
    # Both seats play two Unsafe rounds out of four, so the race is a tie and the
    # two orderings are mirror images with equal payoff.
    assert cs_against_cas == pytest.approx(cas_against_cs)
    assert cs_against_cas == pytest.approx(
        (1 - config.max_private_risk * 0.5)
        * (
            config.race_prize / 2
            + 2 * config.payoff_safe_unsafe
            + 2 * config.payoff_unsafe_safe
        )
    )


def test_self_play_unsafe_frequency_is_measured_not_assumed():
    config = load_config()
    rates = {name: self_play_unsafe_frequency(config, name) for name in STRATEGY_ORDER}
    assert rates["AS"] == pytest.approx(0.0)
    assert rates["AU"] == pytest.approx(1.0)
    # A CS population never leaves Safe; a CAS population never leaves Unsafe.
    assert rates["CS"] == pytest.approx(0.0)
    assert rates["CAS"] == pytest.approx(1.0)


def test_payoff_matrix_is_square_and_labelled():
    config = load_config()
    matrix = expected_payoff_matrix(config)
    assert len(matrix) == len(STRATEGY_ORDER) ** 2
    rows = matrix_to_rows(matrix)
    assert {row["method"] for row in rows} == {"exact_enumeration"}

    sampled = expected_payoff_matrix(config, method="monte_carlo", replications=200, seed=3)
    methods = {
        payoff_method(own, opponent, method="monte_carlo")
        for own, opponent in sampled
    }
    assert methods == {"closed_form", "monte_carlo"}
    assert (
        sum(
            1
            for own, opponent in sampled
            if payoff_method(own, opponent, method="monte_carlo") == "closed_form"
        )
        == 4
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("payoff_unsafe_safe", 9.0),
        ("payoff_safe_unsafe", 0.1),
        ("race_prize", 10.0),
        ("min_rounds", 12),
        ("stop_probability", 0.5),
    ],
)
def test_the_matrix_follows_the_configured_mechanism(field, value):
    """A changed mechanism must move the theory, not be ignored by it."""

    config = load_config()
    changed = replace(config, **{field: value})
    assert exact_expected_payoff(config, "CS", "CAS") != pytest.approx(
        exact_expected_payoff(changed, "CS", "CAS")
    )


@pytest.mark.parametrize(
    "field,value",
    [("unsafe_progress", 5.0), ("safe_progress", 0.2), ("safe_progress", 1.4)],
)
def test_progress_magnitudes_cannot_change_any_expected_payoff(field, value):
    """A deliberate non-effect, recorded so it is not mistaken for a bug.

    Both seats play the same number of rounds, so their progress differs only
    through their Unsafe counts, and Unsafe is required to be strictly faster than
    Safe. The leader is therefore whoever went Unsafe more often, whatever the step
    sizes are — and payoffs depend on the *rank*, never on the margin. Rescaling the
    step sizes moves no cell of the matrix.

    This is the same identity the analyser reports as
    ``progress_gap_before = 0.5 * unsafe_count_diff_before`` on observed data.
    """

    config = load_config()
    rescaled = replace(config, **{field: value})
    for own in STRATEGY_ORDER:
        for opponent in STRATEGY_ORDER:
            assert exact_expected_payoff(config, own, opponent) == pytest.approx(
                exact_expected_payoff(rescaled, own, opponent)
            )
