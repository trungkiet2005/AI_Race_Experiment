"""Acceptance tests for the stage-game classification and Nash equilibria.

Every expected value below appears in the source paper's SI section on the
relationship to the repeated Prisoner's Dilemma, so these check this repository
against an external result rather than against itself. The two structural tests at
the end — the AU/CAS block and "AS is never an equilibrium" — catch failure modes
that comparing numbers alone would let through.
"""
from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from ai_race.engine.state import GameConfig
from ai_race.theory.equilibria import (
    equilibrium_summary,
    nash_equilibria,
    social_dilemma_threshold,
    stage_game,
    stage_game_class,
    stage_game_ordering,
    symmetric_nash_strategies,
    unconditional_nash_regions,
)
from ai_race.theory.payoffs import STRATEGY_ORDER, expected_payoff_matrix

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
CONFIG_DIRECTORY = REPOSITORY_ROOT / "ai_race" / "configs" / "game"


def load_config() -> GameConfig:
    return GameConfig.from_dict(
        json.loads(
            (CONFIG_DIRECTORY / "ai_race_risk_10.json").read_text(encoding="utf-8")
        )
    )


def at_risk(risk: float) -> GameConfig:
    return replace(load_config(), max_private_risk=risk)


def matrix_at(risk: float):
    return expected_payoff_matrix(at_risk(risk))


def test_stage_game_is_a_deadlock_not_a_prisoners_dilemma():
    """SI 3.4a: T = 2.4 > P = 2 > R = 1 > S = 0.6, so ``R > P`` fails."""

    config = load_config()
    game = stage_game(config)
    assert (game.temptation, game.punishment, game.reward, game.sucker) == pytest.approx(
        (2.4, 2.0, 1.0, 0.6)
    )
    assert stage_game_ordering(config) == "T>P>R>S"
    assert stage_game_class(config) == "deadlock"


def test_restoring_the_pd_condition_reclassifies_the_stage_game():
    """The classifier reads the config; it does not return a constant."""

    config = load_config()
    prisoners_dilemma = replace(config, payoff_safe_safe=2.2)
    assert stage_game_ordering(prisoners_dilemma) == "T>R>P>S"
    assert stage_game_class(prisoners_dilemma) == "prisoners_dilemma"

    chicken = replace(config, payoff_safe_safe=2.2, payoff_unsafe_unsafe=0.3)
    assert stage_game_class(chicken) == "chicken"

    harmony = replace(config, payoff_safe_safe=3.0, payoff_unsafe_unsafe=0.3)
    assert stage_game_class(harmony) == "harmony"


def test_an_unnamed_ordering_is_reported_rather_than_mislabelled():
    config = replace(load_config(), payoff_safe_unsafe=5.0)
    label = stage_game_class(config)
    assert label.startswith("unnamed_2x2(")
    assert "S>" in label


def test_social_dilemma_threshold_matches_the_paper():
    """SI 3.4b: p* = 1 - 59/68 = 0.132353."""

    config = load_config()
    assert social_dilemma_threshold(config) == pytest.approx(1 - 59 / 68)
    assert social_dilemma_threshold(config) == pytest.approx(0.132353, abs=1e-6)


def test_the_two_high_treatments_are_social_dilemmas_and_the_low_one_is_not():
    """The point of the threshold: repetition turns a Deadlock into a dilemma."""

    threshold = social_dilemma_threshold(load_config())
    assert 0.1 < threshold, "0.1 sits just below the threshold"
    assert 0.6 > threshold and 0.9 > threshold


def test_unconditional_nash_boundaries_match_the_paper():
    """SI 3.4c: (AU,AU) alone below 0.5148, both up to 0.9206, then (AS,AS)."""

    regions = unconditional_nash_regions(load_config())
    assert regions["always_safe_nash_from"] == pytest.approx(1 - 59 / 121.6)
    assert regions["always_safe_nash_from"] == pytest.approx(0.514803, abs=1e-6)
    assert regions["always_unsafe_nash_until"] == pytest.approx(1 - 5.4 / 68)
    assert regions["always_unsafe_nash_until"] == pytest.approx(0.920588, abs=1e-6)


@pytest.mark.parametrize(
    "risk,region",
    [
        (0.1, "always_unsafe_only"),
        (0.5, "always_unsafe_only"),
        (0.6, "bistable_coordination"),
        (0.9, "bistable_coordination"),
        (0.95, "always_safe_only"),
    ],
)
def test_unconditional_regions_are_labelled_by_treatment(risk, region):
    assert unconditional_nash_regions(at_risk(risk))["region"] == region


def test_two_strategy_equilibria_follow_the_closed_form_boundaries():
    """The search and the algebra must agree on the same three regions."""

    boundaries = unconditional_nash_regions(load_config())
    below = boundaries["always_safe_nash_from"] - 0.01
    inside = (boundaries["always_safe_nash_from"] + boundaries["always_unsafe_nash_until"]) / 2
    above = boundaries["always_unsafe_nash_until"] + 0.01

    pair = ("AS", "AU")
    assert symmetric_nash_strategies(
        expected_payoff_matrix(at_risk(below), strategies=pair), pair
    ) == ["AU"]
    assert sorted(
        symmetric_nash_strategies(
            expected_payoff_matrix(at_risk(inside), strategies=pair), pair
        )
    ) == ["AS", "AU"]
    assert symmetric_nash_strategies(
        expected_payoff_matrix(at_risk(above), strategies=pair), pair
    ) == ["AS"]


@pytest.mark.parametrize(
    "risk,expected",
    [(0.1, ["AU", "CAS"]), (0.6, ["AU", "CAS"]), (0.9, ["CS"])],
)
def test_four_strategy_symmetric_equilibria_match_table_pd_nash(risk, expected):
    assert sorted(symmetric_nash_strategies(matrix_at(risk), STRATEGY_ORDER)) == expected


@pytest.mark.parametrize("risk", [0.1, 0.6, 0.9])
def test_always_safe_is_never_an_equilibrium(risk):
    """SI 3.4d, and a structural check rather than a numeric one.

    A CAS mutant plays Unsafe once and Safe forever after, ending half a step ahead
    of an AS population and winning with almost no accumulated risk. No treatment
    changes that, so AS must be absent at every risk level.
    """

    equilibria = nash_equilibria(matrix_at(risk), STRATEGY_ORDER)
    assert "AS" not in symmetric_nash_strategies(matrix_at(risk), STRATEGY_ORDER)
    assert all("AS" not in profile for profile in equilibria)


@pytest.mark.parametrize("risk", [0.1, 0.6, 0.9])
def test_the_four_au_cas_profiles_stand_or_fall_together(risk):
    """SI 3.4: CAS is indistinguishable from AU against an Unsafe opener.

    Numeric comparison alone would miss this. If a sampling-noise tolerance let one
    of the four through and not another, the equilibrium set would look plausible
    and be structurally impossible.
    """

    equilibria = set(nash_equilibria(matrix_at(risk), STRATEGY_ORDER))
    block = {
        ("AU", "AU"),
        ("AU", "CAS"),
        ("CAS", "AU"),
        ("CAS", "CAS"),
    }
    present = equilibria & block
    assert present in (set(), block)


def test_equilibrium_summary_carries_what_the_table_needs():
    summary = equilibrium_summary(load_config())
    assert summary["stage_game_class"] == "deadlock"
    assert summary["as_is_nash"] is False
    assert summary["above_social_dilemma_threshold"] is False
    assert summary["expected_horizon"] == pytest.approx(9.0)
    assert set(summary["symmetric_nash_strategies"].split("|")) == {"AU", "CAS"}
    assert summary["payoff_method"] == "exact"

    high = equilibrium_summary(at_risk(0.9))
    assert high["above_social_dilemma_threshold"] is True
    assert high["symmetric_nash_strategies"] == "CS"


def test_monte_carlo_needs_a_wider_equilibrium_tolerance():
    """Why ``equilibrium_summary`` defaults to the exact matrix.

    With sampled payoffs the four AU/CAS profiles differ by noise, and a search at
    exact-matrix tolerance drops two of them — producing a structurally impossible
    equilibrium set from numbers that individually look fine.
    """

    sampled = expected_payoff_matrix(
        at_risk(0.1),
        method="monte_carlo",
        replications=5_000,
        seed=260726,
    )
    strict = set(nash_equilibria(sampled, STRATEGY_ORDER, tolerance=1e-9))
    generous = set(nash_equilibria(sampled, STRATEGY_ORDER, tolerance=1.0))
    block = {("AU", "AU"), ("AU", "CAS"), ("CAS", "AU"), ("CAS", "CAS")}
    assert strict & block != block
    assert generous & block == block
