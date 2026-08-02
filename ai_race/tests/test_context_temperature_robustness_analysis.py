"""Focused tests for the temperature-robustness analysis helpers."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "results"
    / "scripts"
    / "analyze_context_temperature_robustness.py"
)
SPEC = importlib.util.spec_from_file_location("temperature_robustness", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_spearman_rank_stability_and_constant_boundary() -> None:
    assert np.isclose(MODULE.spearman_rho([1, 2, 3], [3, 2, 1]), -1.0)
    assert np.isclose(MODULE.spearman_rho([1, 2, 3], [1, 2, 3]), 1.0)
    assert np.isnan(MODULE.spearman_rho([0, 0, 0], [1, 2, 3]))


def test_mapping_assignment_is_balanced_by_repetition_parity() -> None:
    observed = [MODULE.mapping_for_rep(rep) for rep in range(6)]
    assert observed == ["safe_p", "safe_q", "safe_p", "safe_q", "safe_p", "safe_q"]


def test_cluster_bootstrap_resamples_clusters_not_rows() -> None:
    frame = pd.DataFrame(
        {
            "cluster": ["a", "a", "b", "b"],
            "value": [0.0, 0.0, 2.0, 2.0],
        }
    )
    low, high = MODULE.cluster_bootstrap_mean(
        frame, "value", "cluster", repetitions=2_000
    )
    assert low == 0.0
    assert high == 2.0

