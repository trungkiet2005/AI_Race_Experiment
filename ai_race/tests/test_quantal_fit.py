"""The logit-QRE solver must interpolate between a coin flip and the exact best response."""
from __future__ import annotations

import numpy as np
import pytest

from scripts import analyze_best_response as br
from scripts import analyze_quantal_fit as qf

R_MAX = 12


def reachable(P):
    mask = np.zeros(P.shape, dtype=bool)
    for r in range(R_MAX):
        mask[r, :, : r + 1, : r + 1] = True
    return mask


@pytest.mark.parametrize("strategy,risk", [("AS", 0.1), ("AU", 0.9), ("CS", 0.6), ("CAS", 0.9)])
def test_qre_converges_to_best_response_for_large_lambda(strategy, risk) -> None:
    Z = qf.solve_qre(qf.scripted(strategy), risk, 1e4)
    p_qre = qf.sigmoid(Z)
    p_br = qf.br_table(strategy, risk, "mix")
    m = reachable(p_br)
    decided = m & (np.abs(Z) > 30)
    assert decided.sum() > 0
    assert np.all(np.abs(p_qre[decided] - p_br[decided]) < 1e-6)
    tied = m & (p_br == 0.5)
    assert np.all(np.abs(p_qre[tied] - 0.5) < 1e-3)


@pytest.mark.parametrize("strategy,risk", [("AS", 0.1), ("AU", 0.9), ("CS", 0.6)])
def test_qre_is_a_coin_flip_as_lambda_vanishes(strategy, risk) -> None:
    for lam in (0.0, 1e-9):
        p = qf.sigmoid(qf.solve_qre(qf.scripted(strategy), risk, lam))
        assert np.max(np.abs(p - 0.5)) < 1e-6


def test_qre_error_is_monotone_in_lambda() -> None:
    p_br = qf.br_table("CAS", 0.9, "mix")
    m = reachable(p_br)
    errs = [np.mean(np.abs(qf.sigmoid(qf.solve_qre(qf.scripted("CAS"), 0.9, lam))[m] - p_br[m]))
            for lam in (0.001, 0.03, 1.0, 100.0)]
    assert all(a > b for a, b in zip(errs, errs[1:]))


def test_deterministic_rival_belief_matches_the_scripted_rival() -> None:
    a = qf.solve_qre(qf.fixed_belief(1.0), 0.6, 0.1)
    b = qf.solve_qre(qf.scripted("AU"), 0.6, 0.1)
    assert np.array_equal(a, b)


def test_qre_value_function_agrees_with_best_response_module() -> None:
    sol = br.solve("CS", 0.6, lam=0.1)
    Z = qf.solve_qre(qf.scripted("CS"), 0.6, 0.1)
    for r in (0, 3, 9):
        assert np.allclose(qf.sigmoid(Z[r]), sol["policy"][r], atol=1e-12)
