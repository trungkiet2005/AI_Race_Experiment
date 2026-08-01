from __future__ import annotations

import numpy as np
import pytest

from strategy_analysis.egt_reconstruction import (
    ModelParameters,
    expected_game,
    fitnesses,
    fixed_horizon_outcome,
    population_unsafe_frequency,
    simulate_pairwise_comparison_chain,
    strategy_pair_trajectories,
)
from results.scripts.reproduce_egt_model import (
    DEFAULT_LLM_ROOT,
    DEFAULT_LLM_SENSITIVITY_ROOT,
    build_comparison_rows,
)


def test_conditional_strategies_copy_simultaneous_previous_action() -> None:
    cs, cas = strategy_pair_trajectories("CS", "CAS", 6)
    assert cs == (0, 1, 0, 1, 0, 1)
    assert cas == (1, 0, 1, 0, 1, 0)

    cs_against_au, au = strategy_pair_trajectories("CS", "AU", 5)
    assert cs_against_au == (0, 1, 1, 1, 1)
    assert au == (1, 1, 1, 1, 1)


def test_fixed_horizon_setback_discount_applies_only_to_winner_or_tie() -> None:
    # AU beats AS.  AU's stage payoff plus prize is discounted; AS keeps stage payoff.
    outcome = fixed_horizon_outcome("AU", "AS", 5, 0.6)
    assert outcome.payoff_a == pytest.approx((5 * 2.4 + 100) * 0.4)
    assert outcome.payoff_b == pytest.approx(5 * 0.6)

    # AU versus AU ties and each player's complete payoff is discounted.
    tied = fixed_horizon_outcome("AU", "AU", 5, 0.6)
    assert tied.payoff_a == pytest.approx((5 * 2.0 + 50) * 0.4)
    assert tied.payoff_b == tied.payoff_a


@pytest.mark.parametrize("risk", [0.1, 0.6, 0.9])
def test_expected_unconditional_payoffs_match_paper_equations(risk: float) -> None:
    game = expected_game(risk)
    # E[W] = 9 under min 5 and stop probability 0.2.
    assert game.payoff_matrix[0, 0] == pytest.approx(59.0, abs=1e-9)
    assert game.payoff_matrix[0, 1] == pytest.approx(5.4, abs=1e-9)
    assert game.payoff_matrix[1, 0] == pytest.approx((1 - risk) * 121.6, abs=1e-9)
    assert game.payoff_matrix[1, 1] == pytest.approx((1 - risk) * 68.0, abs=1e-9)
    assert game.horizon_tail_mass < 1e-13


def test_ordered_payoff_matrix_is_role_consistent() -> None:
    game = expected_game(0.6)
    # Reversing the ordered strategy pair swaps the two players, so computing
    # the focal payoff in the reversed match must recover the original opponent.
    for i in range(4):
        for j in range(4):
            forward = fixed_horizon_outcome(("AS", "AU", "CS", "CAS")[i], ("AS", "AU", "CS", "CAS")[j], 11, 0.6)
            reverse = fixed_horizon_outcome(("AS", "AU", "CS", "CAS")[j], ("AS", "AU", "CS", "CAS")[i], 11, 0.6)
            assert forward.payoff_b == pytest.approx(reverse.payoff_a)
    assert np.all(np.isfinite(game.payoff_matrix))


def test_population_fitness_removes_self_interaction() -> None:
    payoff = np.array([[1.0, 3.0], [2.0, 4.0]])
    result = fitnesses([2, 1], payoff)
    assert result[0] == pytest.approx((1.0 + 3.0) / 2.0)
    assert result[1] == pytest.approx((2.0 + 2.0) / 2.0)


def test_population_unsafe_rate_uses_ordered_matches_without_self() -> None:
    unsafe = np.array([[0.0, 0.25], [0.75, 1.0]])
    # One player of each type means the only ordered matches are 0->1 and 1->0.
    assert population_unsafe_frequency([1, 1], unsafe) == pytest.approx(0.5)


def test_seeded_chain_is_reproducible_and_normalised() -> None:
    game = expected_game(0.6)
    kwargs = dict(
        beta=2.0,
        mutation=0.02,
        seed=42,
        burn_in=300,
        steps=1_000,
        thin=10,
        population_size=20,
    )
    first = simulate_pairwise_comparison_chain(game, **kwargs)
    second = simulate_pairwise_comparison_chain(game, **kwargs)
    assert first == second
    assert sum(first.strategy_frequencies) == pytest.approx(1.0)
    assert 0.0 <= first.unsafe_frequency <= 1.0
    assert first.samples == 100


def test_invalid_model_parameters_fail_closed() -> None:
    with pytest.raises(ValueError):
        ModelParameters(stop_probability=0.0)
    with pytest.raises(ValueError):
        strategy_pair_trajectories("UNKNOWN", "AS", 5)


def test_primary_and_temperature_sensitivity_protocols_are_not_pooled() -> None:
    assert DEFAULT_LLM_ROOT.name == "live_pilot_t0"
    assert DEFAULT_LLM_SENSITIVITY_ROOT.name == "live_pilot_t07"
    chain_rows = []
    for regime in ("main_reference", "supplement_reference", "reported_best_fit"):
        for risk in (0.1, 0.6, 0.9):
            chain_rows.append(
                {
                    "regime": regime,
                    "max_private_risk": risk,
                    "unsafe_frequency_mean": risk,
                }
            )
    primary = [
        {
            "context": "technology_race",
            "max_private_risk": risk,
            "unsafe_rate_decision_weighted": 0.0,
            "mean_minimum_mismatch_rate": 0.0,
            "unique_classification_rate": 0.0,
        }
        for risk in (0.1, 0.6, 0.9)
    ]
    sensitivity = [
        {
            "context": "technology_race",
            "max_private_risk": risk,
            "unsafe_rate_decision_weighted": 0.2,
            "mean_minimum_mismatch_rate": 0.1,
            "unique_classification_rate": 0.3,
        }
        for risk in (0.1, 0.6, 0.9)
    ]
    rows = build_comparison_rows(chain_rows, primary, sensitivity)
    assert rows[0]["llm_primary_t0_unsafe_technology_race"] == 0.0
    assert rows[0]["llm_sensitivity_t07_unsafe_technology_race"] == 0.2
    assert "llm_unsafe_technology_race" not in rows[0]
