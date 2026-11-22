"""Hand-computable checks for the best-response solver."""
from __future__ import annotations

import pytest

from scripts import analyze_best_response as br


def single_round(strategy, risk):
    return br.solve(strategy, risk, min_rounds=1, stop_prob=1.0, cap=1, tie="min")


def test_one_round_against_always_safe_matches_hand_value() -> None:
    # Safe ties and pays 1 + 50 = 51. Unsafe wins and pays (2.4 + 100)(1 - risk).
    low = single_round("AS", 0.1)
    assert low["value"] == pytest.approx(102.4 * 0.9)
    assert low["q_root"][2] == 1.0
    high = single_round("AS", 0.6)
    assert high["value"] == pytest.approx(51.0)
    assert high["q_root"][2] == 0.0


def test_one_round_against_always_unsafe() -> None:
    # Safe loses and keeps 0.6. Unsafe ties: (2.0 + 50)(1 - risk).
    sol = single_round("AU", 0.1)
    assert sol["value"] == pytest.approx(52.0 * 0.9)
    sol = single_round("AU", 0.9)
    assert sol["value"] == pytest.approx(max(0.6, 52.0 * 0.1))
    assert sol["q_root"][2] == 1.0 or sol["q_root"][2] == 0.0


def test_forward_statistics_for_fixed_horizon() -> None:
    fwd = br.forward(single_round("AS", 0.1))
    assert fwd["expected_rounds"] == pytest.approx(1.0)
    assert fwd["pooled_unsafe_rate"] == pytest.approx(1.0)
    assert fwd["residual_mass"] == pytest.approx(0.0)


def test_expected_horizon_is_nine_rounds() -> None:
    fwd = br.forward(br.solve("AS", 0.1))
    assert fwd["expected_rounds"] == pytest.approx(9.0, abs=1e-6)
