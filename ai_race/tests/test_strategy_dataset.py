"""Synthetic AS/AU/CS/CAS dataset generation and nearest-strategy baseline checks."""
from __future__ import annotations

import random

from strategy_analysis.classify import CANONICAL_STRATEGIES, classify_trajectory
from strategy_analysis.evaluate_baseline import evaluate
from strategy_analysis.generate_dataset import generate_records, sample_horizon


def test_sample_horizon_respects_minimum_and_cap() -> None:
    rng = random.Random(0)
    for _ in range(500):
        horizon = sample_horizon(
            rng, min_rounds=5, stop_probability=0.2, max_rounds_safety_cap=20
        )
        assert 5 <= horizon <= 20


def _make_dataset(noise_levels: tuple[float, ...], games_per_pair: int = 5) -> list[dict]:
    return list(
        generate_records(
            strategies=CANONICAL_STRATEGIES,
            games_per_pair=games_per_pair,
            noise_levels=noise_levels,
            min_rounds=5,
            stop_probability=0.2,
            max_rounds_safety_cap=20,
            seed=1234,
        )
    )


def test_generate_records_covers_all_ordered_strategy_pairs() -> None:
    records = _make_dataset((0.0,))
    seen_pairs = {(r["true_strategy"], r["opponent_true_strategy"]) for r in records}
    expected_pairs = {(a, b) for a in CANONICAL_STRATEGIES for b in CANONICAL_STRATEGIES}
    assert seen_pairs == expected_pairs
    for record in records:
        assert len(record["own_actions"]) == record["horizon"]
        assert len(record["opponent_actions"]) == record["horizon"]
        assert record["horizon"] >= 5


def test_noiseless_dataset_is_perfectly_recovered_by_nearest_strategy_baseline() -> None:
    records = _make_dataset((0.0,), games_per_pair=10)
    summary = evaluate(records)
    noise_zero = summary["noise_levels"]["0.0"]
    assert noise_zero["tied_accuracy"] == 1.0
    for per_strategy in noise_zero["per_strategy"].values():
        assert per_strategy["tied_accuracy"] == 1.0


def test_noisy_dataset_degrades_baseline_accuracy_but_not_to_zero() -> None:
    records = _make_dataset((0.0, 0.2), games_per_pair=20)
    summary = evaluate(records)
    clean = summary["noise_levels"]["0.0"]["tied_accuracy"]
    noisy = summary["noise_levels"]["0.2"]["tied_accuracy"]
    assert clean == 1.0
    assert 0.0 < noisy < 1.0


def test_classify_trajectory_matches_generator_for_pure_as_vs_au() -> None:
    records = _make_dataset((0.0,), games_per_pair=1)
    as_vs_au = next(
        r
        for r in records
        if r["true_strategy"] == "AS" and r["opponent_true_strategy"] == "AU"
    )
    result = classify_trajectory(as_vs_au["own_actions"], as_vs_au["opponent_actions"])
    assert result.best_strategies == ("AS",)
