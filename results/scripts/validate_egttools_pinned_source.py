"""Execute pinned EGTtools source and compare its transition matrix numerically.

The requested ``docs`` branch predates binary ``PairwiseComparison`` wheels for
this workstation's Python.  Its pure-Python ``StochDynamics`` class implements
the same pairwise-comparison transition rule.  This script loads that official
file unchanged from the ignored clone and compares a two-strategy finite chain
against an independent repo-native construction.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import sys
import types

import numpy as np
from scipy.linalg import eig


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy_analysis.egt_reconstruction import expected_game


DOCS_COMMIT = "df7f5fb7787658b3fd3ab21343ff50a3e2a5d439"
DEFAULT_SOURCE = (
    REPO_ROOT
    / "tmp"
    / "EGTTools-docs"
    / "src"
    / "egttools"
    / "analytical"
    / "sed_analytical.py"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "results"
    / "open_source"
    / "egt_reproduction"
    / "egttools_pinned_source_validation.json"
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_pinned_stoch_dynamics(source: Path):
    """Load the upstream file without modifying it or requiring its C++ module."""

    package_name = "_pinned_egttools"
    package = types.ModuleType(package_name)
    package.__path__ = []

    # This validation uses two strategies only.  These mutually inverse helpers
    # supply the combinatorial indexing normally exported by EGTtools' C++ core.
    package.calculate_nb_states = lambda population_size, strategies: (
        population_size + 1
        if strategies == 2
        else math.comb(population_size + strategies - 1, strategies - 1)
    )
    package.sample_simplex = lambda index, population_size, strategies: (
        np.array([index, population_size - index], dtype=np.int64)
        if strategies == 2
        else (_ for _ in ()).throw(NotImplementedError("validation is two-strategy only"))
    )
    package.calculate_state = lambda population_size, state: int(state[0])
    sys.modules[package_name] = package

    analytical_name = f"{package_name}.analytical"
    analytical = types.ModuleType(analytical_name)
    analytical.__path__ = []
    sys.modules[analytical_name] = analytical

    module_name = f"{analytical_name}.sed_analytical"
    spec = importlib.util.spec_from_file_location(module_name, source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"could not load {source}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module.StochDynamics


def _fermi(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-min(value, 700.0)))
    exponential = math.exp(max(value, -700.0))
    return exponential / (1.0 + exponential)


def _repo_transition(payoff: np.ndarray, population_size: int, beta: float, mu: float) -> np.ndarray:
    """Independent two-strategy construction in column-stochastic orientation."""

    transition = np.zeros((population_size + 1, population_size + 1), dtype=float)
    for count_0 in range(population_size + 1):
        count_1 = population_size - count_0
        if count_0 == 0:
            transition[1, 0] = mu
            transition[0, 0] = 1.0 - mu
            continue
        if count_1 == 0:
            transition[population_size - 1, population_size] = mu
            transition[population_size, population_size] = 1.0 - mu
            continue
        fitness_0 = ((count_0 - 1) * payoff[0, 0] + count_1 * payoff[0, 1]) / (
            population_size - 1
        )
        fitness_1 = ((count_1 - 1) * payoff[1, 1] + count_0 * payoff[1, 0]) / (
            population_size - 1
        )
        increase_0 = (count_1 / population_size) * (
            mu
            + (1.0 - mu)
            * (count_0 / (population_size - 1))
            * _fermi(beta * (fitness_0 - fitness_1))
        )
        decrease_0 = (count_0 / population_size) * (
            mu
            + (1.0 - mu)
            * (count_1 / (population_size - 1))
            * _fermi(beta * (fitness_1 - fitness_0))
        )
        transition[count_0 + 1, count_0] = increase_0
        transition[count_0 - 1, count_0] = decrease_0
        transition[count_0, count_0] = 1.0 - increase_0 - decrease_0
    return transition


def _stationary(transition: np.ndarray) -> np.ndarray:
    values, vectors = eig(transition)
    index = int(np.argmin(np.abs(values - 1.0)))
    vector = np.abs(vectors[:, index].real)
    return vector / vector.sum()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if not args.source.is_file():
        raise FileNotFoundError(
            f"pinned clone missing: {args.source}; clone Socrats/EGTTools branch docs under tmp first"
        )

    population_size = 10
    beta = 2.0
    mutation = 0.02
    max_private_risk = 0.6
    payoff = expected_game(max_private_risk).payoff_matrix[:2, :2]
    stoch_dynamics = _load_pinned_stoch_dynamics(args.source)
    official_evolver = stoch_dynamics(
        nb_strategies=2,
        payoffs=payoff,
        pop_size=population_size,
        group_size=2,
        mu=mutation,
    )
    official_transition = official_evolver.calculate_full_transition_matrix(beta).toarray()
    official_stationary = official_evolver.calculate_stationary_distribution(beta)
    repo_transition = _repo_transition(payoff, population_size, beta, mutation)
    repo_stationary = _stationary(repo_transition)

    transition_max_abs = float(np.max(np.abs(official_transition - repo_transition)))
    stationary_max_abs = float(np.max(np.abs(official_stationary - repo_stationary)))
    passed = transition_max_abs < 1e-12 and stationary_max_abs < 1e-10
    result = {
        "schema_version": "egttools-pinned-source-validation-v1",
        "passed": passed,
        "official_source": {
            "repository": "https://github.com/Socrats/EGTTools",
            "branch": "docs",
            "commit": DOCS_COMMIT,
            "relative_file": str(args.source.relative_to(REPO_ROOT)),
            "sha256": _sha256(args.source),
            "executed_class": "egttools.analytical.StochDynamics",
            "source_modified": False,
        },
        "binary_pairwise_comparison": {
            "executed": False,
            "blocker": (
                "No egttools==0.1.14.2 wheel exists for Windows CPython 3.13; "
                "the docs-commit package imports its compiled numerical module at package initialisation."
            ),
        },
        "validation_case": {
            "strategies": ["AS", "AU"],
            "max_private_risk": max_private_risk,
            "population_size": population_size,
            "beta": beta,
            "mutation": mutation,
            "payoff_matrix": payoff.tolist(),
        },
        "comparison": {
            "transition_matrix_max_abs_difference": transition_max_abs,
            "stationary_distribution_max_abs_difference": stationary_max_abs,
            "official_column_sums_max_abs_error": float(
                np.max(np.abs(official_transition.sum(axis=0) - 1.0))
            ),
            "repo_column_sums_max_abs_error": float(
                np.max(np.abs(repo_transition.sum(axis=0) - 1.0))
            ),
            "official_stationary_distribution": official_stationary.tolist(),
            "repo_stationary_distribution": repo_stationary.tolist(),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit("pinned EGTtools source validation failed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

