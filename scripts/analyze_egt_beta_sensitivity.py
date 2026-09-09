"""Test whether the theory-versus-LLM risk-response gap survives selection strength.

The manuscript compares the reconstructed finite-population benchmark against the
two admitted frontier routes at a single selection strength.  The reconstruction
script `scripts/reproduce_egt_model.py` hard-codes only three parameter points
(beta=2 with mu=0.02, beta=2 with mu=0.01, and beta=0.01 with mu=0.05), all
inherited from the source paper, so the repository cannot currently say whether
the reported gap is a property of the evolutionary model or an artefact of
beta=2.  A reviewer will ask exactly that.  This script answers it by layering a
sensitivity sweep on the existing reconstruction: it imports
`analysis.strategy.egt_reconstruction.expected_game` for the payoff matrices and
`run_independent_chains` for the Markov chains, and changes nothing about the
payoff arithmetic, the geometric horizon law, or the effective-risk formula.

What is swept and why.  Selection strength beta runs over the logarithmic grid
{0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 2, 3, 10}, which spans from effectively
neutral drift (beta=0.001, where a fitness difference of order 10 ECU shifts the
Fermi adoption probability by under one percent) through the paper's two
published points (0.01 and 2) to strong selection (beta=10, near best-response
imitation).  Each beta is crossed with the three mutation conventions the source
paper and its supplement actually use, read off the existing regime table:
`one_over_Z` fixes mu = 1/Z = 0.01 (the Figure S5 convention), `beta_over_Z`
sets mu = beta/Z, which is the main-text convention and is the only reason
mu = 0.02 accompanies beta = 2 there, and `fixed_0p05` fixes mu = 0.05 (the
reported best-fit convention).  In this module's chain runner `mutation` is the
per-update probability that the sampled focal individual switches uniformly to
one of the other three strategies, so `beta_over_Z` genuinely varies with beta
while the other two do not.  All three risk levels of the manuscript, 0.1, 0.6
and 0.9, are run at every cell, because the reviewer's question is about the
shape of the risk response and not about any single level.

Cells that coincide numerically, such as beta=1 under `one_over_Z` and under
`beta_over_Z`, are simulated once and reported under both conventions.

Evidence boundary.  This inherits the boundary of the lane it extends.  It is a
repo-native reconstruction of the disclosed model, not a bitwise reproduction of
the authors' undisclosed analysis code, payoff Monte Carlo seeds, or EGTtools
revision, and the compiled EGTtools `PairwiseComparison` path could not be
executed on this interpreter, so the parity audit that this lane relies on used
the pinned pure-Python `StochDynamics` class instead.  The stationary quantities
below are chain estimates with a between-chain spread reported alongside them,
never a single chain presented as the stationary distribution.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import platform
import sys
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from analysis.strategy.egt_reconstruction import (
    STRATEGIES,
    ExpectedGame,
    ModelParameters,
    expected_game,
    run_independent_chains,
)

DEFAULT_OUTPUT = REPO_ROOT / "results" / "open_source" / "egt_reproduction"
PUBLISHED_SUMMARY = DEFAULT_OUTPUT / "egt_stationary_summary.csv"
ROUTE_SUMMARY = (
    REPO_ROOT
    / "results"
    / "frontier"
    / "egt_frontier_comparison_v2"
    / "llm_strategy_summary_primary_t0.csv"
)
RECONSTRUCTION_SCRIPT = REPO_ROOT / "scripts" / "reproduce_egt_model.py"
RECONSTRUCTION_MODULE = REPO_ROOT / "analysis" / "strategy" / "egt_reconstruction.py"
PINNED_SOURCE_VALIDATION = DEFAULT_OUTPUT / "egttools_pinned_source_validation.json"
PAPER_SOURCE = REPO_ROOT / "references" / "papers" / "sources" / "arXiv-2607.26034v1" / "paper.tex"

BETA_GRID = (0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 3.0, 10.0)
RISKS = (0.1, 0.6, 0.9)
MUTATION_RULES = ("one_over_Z", "beta_over_Z", "fixed_0p05")
POPULATION_SIZE = ModelParameters().population_size
SEED_BASE = 990_726

ROUTE_LABELS = {
    "gemini": "google/gemini-3-flash-preview",
    "claude": "anthropic/claude-sonnet-5@default",
}

REFERENCE_CELL = {"beta": 2.0, "mutation": 0.02, "regime": "main_reference"}
MIXING_SUSPECT_RANGE = 0.05
REFERENCE_TOLERANCE = 0.02

SHAPE_CRITERIA = {
    "low_risk_at_least": 0.85,
    "medium_risk_window": (0.25, 0.80),
    "high_risk_window": (0.10, 0.65),
    "minimum_decline_per_step": 0.05,
}

EVIDENCE_BOUNDARY = (
    "Repo-native reconstruction of the disclosed reduced evolutionary model, not a "
    "bitwise reproduction: the source paper's analysis code, generated payoff "
    "matrices, payoff Monte Carlo seeds, and EGTtools revision are not public in "
    "arXiv v1. The compiled EGTtools PairwiseComparison class could not be executed "
    "on this interpreter because no matching wheel exists for it, so this lane's "
    "parity audit ran the pinned pure-Python StochDynamics class instead. Stationary "
    "quantities are seeded-chain estimates reported with their between-chain spread, "
    "not exact stationary distributions, and the frontier route rates are descriptive "
    "prompted self-play endpoints rather than draws from the population process."
)

ESTIMAND = (
    "For each (selection strength beta, mutation rate mu, maximum private risk r): the "
    "stationary population share of AS, AU, CS and CAS and the implied "
    "decision-weighted Unsafe fraction in a well-mixed population of Z=100 under "
    "Fermi imitation with uniform mutation, and the distance between the resulting "
    "three-point risk profile and each admitted frontier route's observed profile."
)


def mutation_for(rule: str, beta: float) -> float:
    if rule == "one_over_Z":
        return 1.0 / POPULATION_SIZE
    if rule == "beta_over_Z":
        return beta / POPULATION_SIZE
    if rule == "fixed_0p05":
        return 0.05
    raise ValueError(f"unknown mutation rule {rule!r}")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _cell_seeds(beta: float, mutation: float, risk: float, chains: int) -> list[int]:
    key = f"{beta!r}|{mutation!r}|{risk!r}".encode("utf-8")
    offset = int.from_bytes(hashlib.sha256(key).digest()[:4], "big") % 1_000_000
    return [SEED_BASE + offset + index for index in range(chains)]


def _run_cell(task: dict[str, Any]) -> dict[str, Any]:
    game: ExpectedGame = task["game"]
    summaries = run_independent_chains(
        game,
        beta=task["beta"],
        mutation=task["mutation"],
        seeds=task["seeds"],
        burn_in=task["burn_in"],
        steps=task["steps"],
        thin=task["thin"],
        population_size=POPULATION_SIZE,
    )
    chains = [asdict(summary) for summary in summaries]
    unsafe = np.array([chain["unsafe_frequency"] for chain in chains], dtype=float)
    frequencies = np.array([chain["strategy_frequencies"] for chain in chains], dtype=float)
    record: dict[str, Any] = {
        "beta": task["beta"],
        "mutation": task["mutation"],
        "max_private_risk": task["risk"],
        "chains": len(chains),
        "samples_per_chain": int(min(chain["samples"] for chain in chains)),
        "seeds": task["seeds"],
        "unsafe_frequency_mean": float(unsafe.mean()),
        "unsafe_frequency_min": float(unsafe.min()),
        "unsafe_frequency_max": float(unsafe.max()),
        "unsafe_frequency_between_chain_range": float(unsafe.max() - unsafe.min()),
        "unsafe_frequency_between_chain_sd": float(unsafe.std(ddof=1)),
        "unsafe_frequency_between_chain_sem": float(unsafe.std(ddof=1) / math.sqrt(unsafe.size)),
        "accepted_moves_mean": float(np.mean([chain["moves"] for chain in chains])),
    }
    for index, strategy in enumerate(STRATEGIES):
        column = frequencies[:, index]
        record[f"frequency_{strategy}_mean"] = float(column.mean())
        record[f"frequency_{strategy}_min"] = float(column.min())
        record[f"frequency_{strategy}_max"] = float(column.max())
        record[f"frequency_{strategy}_between_chain_range"] = float(column.max() - column.min())
    record["max_strategy_between_chain_range"] = max(
        record[f"frequency_{strategy}_between_chain_range"] for strategy in STRATEGIES
    )
    return record


def run_cells(
    tasks: list[dict[str, Any]],
    *,
    workers: int,
) -> list[dict[str, Any]]:
    if workers <= 1:
        return [_run_cell(task) for task in tasks]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(_run_cell, tasks))


def build_tasks(
    games: dict[float, ExpectedGame],
    pairs: list[tuple[float, float]],
    *,
    chains: int,
    burn_in: int,
    steps: int,
    thin: int,
) -> list[dict[str, Any]]:
    tasks: list[dict[str, Any]] = []
    for beta, mutation in pairs:
        for risk in RISKS:
            tasks.append(
                {
                    "game": games[risk],
                    "beta": beta,
                    "mutation": mutation,
                    "risk": risk,
                    "seeds": _cell_seeds(beta, mutation, risk, chains),
                    "burn_in": burn_in,
                    "steps": steps,
                    "thin": thin,
                }
            )
    return tasks


def read_published_reference() -> dict[float, dict[str, float]]:
    with PUBLISHED_SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = [row for row in csv.DictReader(handle) if row["regime"] == REFERENCE_CELL["regime"]]
    if not rows:
        raise FileNotFoundError(f"no {REFERENCE_CELL['regime']} rows in {PUBLISHED_SUMMARY}")
    reference: dict[float, dict[str, float]] = {}
    for row in rows:
        reference[float(row["max_private_risk"])] = {
            "beta": float(row["beta"]),
            "mutation": float(row["mutation"]),
            "unsafe_frequency_mean": float(row["unsafe_frequency_mean"]),
            "unsafe_frequency_min": float(row["unsafe_frequency_min"]),
            "unsafe_frequency_max": float(row["unsafe_frequency_max"]),
            **{
                f"frequency_{strategy}_mean": float(row[f"frequency_{strategy}_mean"])
                for strategy in STRATEGIES
            },
        }
    return reference


def read_route_profiles() -> dict[str, dict[float, dict[str, float]]]:
    with ROUTE_SUMMARY.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    profiles: dict[str, dict[float, dict[str, float]]] = {}
    for row in rows:
        context = str(row["context"])
        if context not in ROUTE_LABELS:
            continue
        profiles.setdefault(context, {})[float(row["max_private_risk"])] = {
            "unsafe_rate_decision_weighted": float(row["unsafe_rate_decision_weighted"]),
            "player_trajectories": int(float(row["player_trajectories"])),
            "decisions": int(float(row["decisions"])),
        }
    for context, cells in profiles.items():
        missing = set(RISKS) - set(cells)
        if missing:
            raise ValueError(f"route {context} is missing risk levels {sorted(missing)}")
    return profiles


def verify_reference(
    cells: list[dict[str, Any]],
    reference: dict[float, dict[str, float]],
) -> dict[str, Any]:
    checks: list[dict[str, Any]] = []
    for risk in RISKS:
        published = reference[risk]
        ours = next(
            row
            for row in cells
            if row["max_private_risk"] == risk
            and row["beta"] == REFERENCE_CELL["beta"]
            and row["mutation"] == REFERENCE_CELL["mutation"]
        )
        difference = abs(ours["unsafe_frequency_mean"] - published["unsafe_frequency_mean"])
        strategy_difference = max(
            abs(ours[f"frequency_{strategy}_mean"] - published[f"frequency_{strategy}_mean"])
            for strategy in STRATEGIES
        )
        checks.append(
            {
                "max_private_risk": risk,
                "published_unsafe_frequency_mean": published["unsafe_frequency_mean"],
                "sweep_unsafe_frequency_mean": ours["unsafe_frequency_mean"],
                "absolute_difference": difference,
                "published_between_chain_range": published["unsafe_frequency_max"]
                - published["unsafe_frequency_min"],
                "sweep_between_chain_range": ours["unsafe_frequency_between_chain_range"],
                "max_strategy_share_absolute_difference": strategy_difference,
                "within_tolerance": bool(
                    difference <= REFERENCE_TOLERANCE
                    and strategy_difference <= REFERENCE_TOLERANCE
                ),
            }
        )
    return {
        "cell": dict(REFERENCE_CELL),
        "published_artifact": str(PUBLISHED_SUMMARY.relative_to(REPO_ROOT)),
        "tolerance_absolute": REFERENCE_TOLERANCE,
        "passed": all(check["within_tolerance"] for check in checks),
        "checks": checks,
    }


def shape_flags(profile: dict[float, float]) -> dict[str, Any]:
    low, medium, high = (profile[risk] for risk in RISKS)
    step = SHAPE_CRITERIA["minimum_decline_per_step"]
    medium_low, medium_high = SHAPE_CRITERIA["medium_risk_window"]
    high_low, high_high = SHAPE_CRITERIA["high_risk_window"]
    flags = {
        "shape_low_risk_near_total": bool(low >= SHAPE_CRITERIA["low_risk_at_least"]),
        "shape_monotone_decline": bool(low - medium >= step and medium - high >= step),
        "shape_medium_is_middling": bool(medium_low <= medium <= medium_high),
        "shape_high_is_low_but_nonzero": bool(high_low <= high <= high_high),
    }
    flags["shape_matches_llm_pattern"] = bool(all(flags.values()))
    return flags


def build_sensitivity_rows(
    cells_by_key: dict[tuple[float, float, float], dict[str, Any]],
    pairs_by_rule: dict[str, list[tuple[float, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rule in MUTATION_RULES:
        for beta, mutation in pairs_by_rule[rule]:
            profile = {
                risk: cells_by_key[(beta, mutation, risk)]["unsafe_frequency_mean"]
                for risk in RISKS
            }
            flags = shape_flags(profile)
            profile_suspect = any(
                cells_by_key[(beta, mutation, risk)]["unsafe_frequency_between_chain_range"]
                > MIXING_SUSPECT_RANGE
                for risk in RISKS
            )
            flags["profile_mixing_suspect"] = bool(profile_suspect)
            flags["shape_matches_llm_pattern_well_mixed"] = bool(
                flags["shape_matches_llm_pattern"] and not profile_suspect
            )
            for risk in RISKS:
                cell = cells_by_key[(beta, mutation, risk)]
                row: dict[str, Any] = {
                    "mutation_rule": rule,
                    "beta": beta,
                    "mutation": mutation,
                    "population_size": POPULATION_SIZE,
                    **{key: value for key, value in cell.items() if key != "seeds"},
                    "seeds": "|".join(str(seed) for seed in cell["seeds"]),
                }
                row.update(flags)
                row["mixing_suspect"] = bool(
                    cell["unsafe_frequency_between_chain_range"] > MIXING_SUSPECT_RANGE
                )
                row["is_published_main_reference"] = bool(
                    beta == REFERENCE_CELL["beta"] and mutation == REFERENCE_CELL["mutation"]
                )
                rows.append(row)
    return rows


def build_fit_rows(
    cells_by_key: dict[tuple[float, float, float], dict[str, Any]],
    pairs_by_rule: dict[str, list[tuple[float, float]]],
    routes: dict[str, dict[float, dict[str, float]]],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for context, cells in sorted(routes.items()):
        observed = np.array(
            [cells[risk]["unsafe_rate_decision_weighted"] for risk in RISKS], dtype=float
        )
        for rule in MUTATION_RULES:
            for beta, mutation in pairs_by_rule[rule]:
                predicted = np.array(
                    [
                        cells_by_key[(beta, mutation, risk)]["unsafe_frequency_mean"]
                        for risk in RISKS
                    ],
                    dtype=float
                )
                deviation = predicted - observed
                spread = max(
                    cells_by_key[(beta, mutation, risk)]["unsafe_frequency_between_chain_range"]
                    for risk in RISKS
                )
                profile = {risk: float(value) for risk, value in zip(RISKS, predicted)}
                row: dict[str, Any] = {
                    "route_context": context,
                    "route": ROUTE_LABELS[context],
                    "mutation_rule": rule,
                    "beta": beta,
                    "mutation": mutation,
                    "rmse_percentage_points": float(
                        np.sqrt(np.mean(deviation**2)) * 100.0
                    ),
                    "max_absolute_deviation_percentage_points": float(
                        np.max(np.abs(deviation)) * 100.0
                    ),
                    "max_between_chain_range_percentage_points": float(spread * 100.0),
                }
                for risk, predicted_value, observed_value in zip(RISKS, predicted, observed):
                    tag = f"r{str(risk).replace('.', 'p')}"
                    row[f"theory_unsafe_{tag}"] = float(predicted_value)
                    row[f"route_unsafe_{tag}"] = float(observed_value)
                    row[f"deviation_{tag}_percentage_points"] = float(
                        (predicted_value - observed_value) * 100.0
                    )
                row.update(shape_flags(profile))
                row["profile_mixing_suspect"] = bool(spread > MIXING_SUSPECT_RANGE)
                row["shape_matches_llm_pattern_well_mixed"] = bool(
                    row["shape_matches_llm_pattern"] and not row["profile_mixing_suspect"]
                )
                rows.append(row)
    best_by_route: dict[str, dict[str, Any]] = {}
    best_mixed_by_route: dict[str, dict[str, Any]] = {}
    for row in rows:
        current = best_by_route.get(row["route_context"])
        if current is None or row["rmse_percentage_points"] < current["rmse_percentage_points"]:
            best_by_route[row["route_context"]] = row
        if row["profile_mixing_suspect"]:
            continue
        current_mixed = best_mixed_by_route.get(row["route_context"])
        if (
            current_mixed is None
            or row["rmse_percentage_points"] < current_mixed["rmse_percentage_points"]
        ):
            best_mixed_by_route[row["route_context"]] = row
    for row in rows:
        row["is_best_fit_for_route"] = bool(row is best_by_route[row["route_context"]])
        row["is_best_fit_for_route_well_mixed"] = bool(
            row is best_mixed_by_route.get(row["route_context"])
        )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--burn-in", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--thin", type=int, default=100)
    parser.add_argument("--workers", type=int, default=11)
    args = parser.parse_args()
    if args.chains < 2:
        raise ValueError("at least two independent chains are required to report a spread")
    args.output.mkdir(parents=True, exist_ok=True)

    games = {risk: expected_game(risk) for risk in RISKS}
    pairs_by_rule = {
        rule: [(beta, mutation_for(rule, beta)) for beta in BETA_GRID] for rule in MUTATION_RULES
    }
    unique_pairs = sorted({pair for pairs in pairs_by_rule.values() for pair in pairs})

    reference = read_published_reference()
    reference_pair = (REFERENCE_CELL["beta"], REFERENCE_CELL["mutation"])
    if reference_pair not in unique_pairs:
        raise ValueError("the published main-reference cell is not inside the swept grid")
    reference_cells = run_cells(
        build_tasks(
            games,
            [reference_pair],
            chains=args.chains,
            burn_in=args.burn_in,
            steps=args.steps,
            thin=args.thin,
        ),
        workers=min(args.workers, len(RISKS)),
    )
    verification = verify_reference(reference_cells, reference)
    if not verification["passed"]:
        (args.output / "egt_beta_sensitivity.json").write_text(
            json.dumps(
                {
                    "schema_version": "egt-beta-sensitivity-v1",
                    "status": "aborted_reference_cell_mismatch",
                    "evidence_boundary": EVIDENCE_BOUNDARY,
                    "reference_verification": verification,
                },
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        print(json.dumps({"status": "aborted", "reference_verification": verification}, indent=2))
        return 1

    remaining = [pair for pair in unique_pairs if pair != reference_pair]
    cells = reference_cells + run_cells(
        build_tasks(
            games,
            remaining,
            chains=args.chains,
            burn_in=args.burn_in,
            steps=args.steps,
            thin=args.thin,
        ),
        workers=args.workers,
    )
    cells_by_key = {
        (cell["beta"], cell["mutation"], cell["max_private_risk"]): cell for cell in cells
    }

    sensitivity_rows = build_sensitivity_rows(cells_by_key, pairs_by_rule)
    sensitivity_fields = [
        "mutation_rule",
        "beta",
        "mutation",
        "population_size",
        "max_private_risk",
        "chains",
        "samples_per_chain",
        "seeds",
        *[
            f"frequency_{strategy}_{suffix}"
            for strategy in STRATEGIES
            for suffix in ("mean", "min", "max", "between_chain_range")
        ],
        "unsafe_frequency_mean",
        "unsafe_frequency_min",
        "unsafe_frequency_max",
        "unsafe_frequency_between_chain_range",
        "unsafe_frequency_between_chain_sd",
        "unsafe_frequency_between_chain_sem",
        "max_strategy_between_chain_range",
        "accepted_moves_mean",
        "shape_low_risk_near_total",
        "shape_monotone_decline",
        "shape_medium_is_middling",
        "shape_high_is_low_but_nonzero",
        "shape_matches_llm_pattern",
        "profile_mixing_suspect",
        "shape_matches_llm_pattern_well_mixed",
        "mixing_suspect",
        "is_published_main_reference",
    ]
    _write_csv(args.output / "egt_beta_sensitivity.csv", sensitivity_rows, sensitivity_fields)

    routes = read_route_profiles()
    fit_rows = build_fit_rows(cells_by_key, pairs_by_rule, routes)
    fit_fields = [
        "route_context",
        "route",
        "mutation_rule",
        "beta",
        "mutation",
        "rmse_percentage_points",
        "max_absolute_deviation_percentage_points",
        "max_between_chain_range_percentage_points",
        *[
            f"{prefix}_r{str(risk).replace('.', 'p')}{suffix}"
            for risk in RISKS
            for prefix, suffix in (
                ("theory_unsafe", ""),
                ("route_unsafe", ""),
                ("deviation", "_percentage_points"),
            )
        ],
        "shape_low_risk_near_total",
        "shape_monotone_decline",
        "shape_medium_is_middling",
        "shape_high_is_low_but_nonzero",
        "shape_matches_llm_pattern",
        "profile_mixing_suspect",
        "shape_matches_llm_pattern_well_mixed",
        "is_best_fit_for_route",
        "is_best_fit_for_route_well_mixed",
    ]
    _write_csv(args.output / "egt_beta_fit_to_routes.csv", fit_rows, fit_fields)

    best_selectors = (
        ("best_fit_per_route", "is_best_fit_for_route"),
        ("best_fit_per_route_well_mixed", "is_best_fit_for_route_well_mixed"),
    )
    best_tables = {
        label: {
            row["route_context"]: {
                key: row[key]
                for key in (
                    "route",
                    "mutation_rule",
                    "beta",
                    "mutation",
                    "rmse_percentage_points",
                    "max_absolute_deviation_percentage_points",
                    "theory_unsafe_r0p1",
                    "theory_unsafe_r0p6",
                    "theory_unsafe_r0p9",
                    "route_unsafe_r0p1",
                    "route_unsafe_r0p6",
                    "route_unsafe_r0p9",
                    "shape_matches_llm_pattern",
                    "profile_mixing_suspect",
                )
            }
            for row in fit_rows
            if row[selector]
        }
        for label, selector in best_selectors
    }
    best = best_tables["best_fit_per_route"]
    shape_matching = sorted(
        {
            (row["mutation_rule"], row["beta"], row["mutation"])
            for row in sensitivity_rows
            if row["shape_matches_llm_pattern"]
        }
    )
    shape_matching_well_mixed = sorted(
        {
            (row["mutation_rule"], row["beta"], row["mutation"])
            for row in sensitivity_rows
            if row["shape_matches_llm_pattern_well_mixed"]
        }
    )
    suspect_cells = sorted(
        {
            (row["mutation_rule"], row["beta"], row["mutation"], row["max_private_risk"])
            for row in sensitivity_rows
            if row["mixing_suspect"]
        }
    )
    provenance = {
        "schema_version": "egt-beta-sensitivity-v1",
        "status": "complete",
        "estimand": ESTIMAND,
        "evidence_boundary": EVIDENCE_BOUNDARY,
        "question": (
            "Is the reported gap between the evolutionary benchmark and the admitted "
            "frontier routes robust to selection strength, or an artefact of beta=2?"
        ),
        "grid": {
            "beta": list(BETA_GRID),
            "mutation_rules": {
                rule: {
                    "definition": {
                        "one_over_Z": "mu = 1/Z = 0.01 (Figure S5 convention)",
                        "beta_over_Z": "mu = beta/Z (main-text convention; gives mu=0.02 at beta=2)",
                        "fixed_0p05": "mu = 0.05 (reported best-fit convention)",
                    }[rule],
                    "pairs": [
                        {"beta": beta, "mutation": mutation}
                        for beta, mutation in pairs_by_rule[rule]
                    ],
                }
                for rule in MUTATION_RULES
            },
            "mutation_semantics": (
                "mutation is the per-update probability that the sampled focal "
                "individual switches uniformly to one of the other three strategies, "
                "as implemented in analysis.strategy.egt_reconstruction."
                "simulate_pairwise_comparison_chain"
            ),
            "max_private_risk": list(RISKS),
            "unique_simulated_parameter_pairs": [
                {"beta": beta, "mutation": mutation} for beta, mutation in unique_pairs
            ],
            "coinciding_pairs_simulated_once": True,
        },
        "model": {
            "population_size": POPULATION_SIZE,
            "strategies": list(STRATEGIES),
            "payoff_construction": "analysis.strategy.egt_reconstruction.expected_game (unchanged)",
            "chain_runner": "analysis.strategy.egt_reconstruction.run_independent_chains (unchanged)",
            "maximum_discarded_horizon_mass": max(
                game.horizon_tail_mass for game in games.values()
            ),
        },
        "numerics": {
            "chains_per_cell": args.chains,
            "burn_in": args.burn_in,
            "steps": args.steps,
            "thin": args.thin,
            "samples_per_chain": min(row["samples_per_chain"] for row in sensitivity_rows),
            "seed_base": SEED_BASE,
            "seed_rule": (
                "seed = SEED_BASE + (sha256('beta|mutation|risk')[:4] mod 1e6) + chain index; "
                "deliberately disjoint from the reproduction script's seed scheme so the "
                "reference check is an independent-seed check"
            ),
            "seeds_by_cell": {
                f"beta={beta},mu={mutation},risk={risk}": cells_by_key[(beta, mutation, risk)][
                    "seeds"
                ]
                for beta, mutation in unique_pairs
                for risk in RISKS
            },
            "uncertainty": "between_independent_chain_spread_diagnostic_not_confidence_interval",
        },
        "distance": {
            "primary": "rmse_percentage_points",
            "definition": (
                "root mean squared deviation, in percentage points, between the theory's "
                "three-point Unsafe profile at risks 0.1/0.6/0.9 and the route's observed "
                "decision-weighted Unsafe rates at the same three risks"
            ),
            "why": (
                "the three risk levels are the only commensurable coordinates between the "
                "two designs, RMSE weights them equally, penalises a single badly missed "
                "condition more than a mean absolute error would, and reports in the same "
                "percentage-point units the manuscript's tables use; the Chebyshev "
                "distance is reported beside it so one bad condition stays visible"
            ),
            "secondary": "max_absolute_deviation_percentage_points",
        },
        "shape_criteria": {
            "description": (
                "the qualitative pattern the admitted routes show: near-total Unsafe play "
                "at risk 0.1, a middling rate at 0.6, and a lower but clearly non-zero "
                "rate at 0.9, declining monotonically"
            ),
            **{
                key: (list(value) if isinstance(value, tuple) else value)
                for key, value in SHAPE_CRITERIA.items()
            },
        },
        "reference_verification": verification,
        "routes": {
            context: {
                "route": ROUTE_LABELS[context],
                "profile": {
                    str(risk): cells[risk]["unsafe_rate_decision_weighted"] for risk in RISKS
                },
                "player_trajectories_per_cell": {
                    str(risk): cells[risk]["player_trajectories"] for risk in RISKS
                },
                "decisions_per_cell": {
                    str(risk): cells[risk]["decisions"] for risk in RISKS
                },
            }
            for context, cells in sorted(routes.items())
        },
        "best_fit_per_route": best,
        "best_fit_per_route_well_mixed": best_tables["best_fit_per_route_well_mixed"],
        "cells_matching_llm_shape": [
            {"mutation_rule": rule, "beta": beta, "mutation": mutation}
            for rule, beta, mutation in shape_matching
        ],
        "cells_matching_llm_shape_well_mixed": [
            {"mutation_rule": rule, "beta": beta, "mutation": mutation}
            for rule, beta, mutation in shape_matching_well_mixed
        ],
        "any_cell_matches_llm_shape": bool(shape_matching),
        "any_well_mixed_cell_matches_llm_shape": bool(shape_matching_well_mixed),
        "mixing_diagnostic": {
            "suspect_between_chain_range_threshold": MIXING_SUSPECT_RANGE,
            "reading": (
                "a cell whose independent chains disagree by more than this threshold on "
                "the aggregate Unsafe fraction has not demonstrably mixed within the given "
                "burn-in and steps, so its chain mean must not be read as the stationary "
                "distribution. Very low mutation, which mu=beta/Z produces at small beta, "
                "makes the four monomorphic states near-absorbing and is the main cause."
            ),
            "suspect_cell_count": len(suspect_cells),
            "total_cell_count": len(sensitivity_rows),
            "suspect_cells": [
                {
                    "mutation_rule": rule,
                    "beta": beta,
                    "mutation": mutation,
                    "max_private_risk": risk,
                }
                for rule, beta, mutation, risk in suspect_cells
            ],
        },
        "sources_read": {
            path_label: {
                "path": str(path.relative_to(REPO_ROOT)) if path.is_file() else str(path),
                "sha256": _sha256(path),
            }
            for path_label, path in {
                "reconstruction_module": RECONSTRUCTION_MODULE,
                "reconstruction_script": RECONSTRUCTION_SCRIPT,
                "published_chain_summary": PUBLISHED_SUMMARY,
                "frontier_route_summary": ROUTE_SUMMARY,
                "pinned_source_validation": PINNED_SOURCE_VALIDATION,
                "paper_source": PAPER_SOURCE,
                "this_script": Path(__file__).resolve(),
            }.items()
        },
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
        "output_files": [
            "egt_beta_sensitivity.csv",
            "egt_beta_fit_to_routes.csv",
            "egt_beta_sensitivity.json",
        ],
        "csv_layout": {
            "egt_beta_sensitivity.csv": (
                "one row per (mutation_rule, beta, mutation, max_private_risk) with the four "
                "strategy shares as columns; each share and the aggregate Unsafe fraction "
                "carry a chain mean, a chain minimum, a chain maximum, and a between-chain "
                "range, and the Unsafe fraction additionally carries a between-chain "
                "standard deviation and standard error; mixing_suspect marks a cell whose "
                "independent chains did not demonstrably mix"
            ),
            "egt_beta_fit_to_routes.csv": (
                "one row per (route, mutation_rule, beta, mutation) giving the three-point "
                "profile distance to that route, with is_best_fit_for_route marking the "
                "minimum-RMSE cell per route"
            ),
        },
    }
    (args.output / "egt_beta_sensitivity.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "output": str(args.output),
                "cells": len(unique_pairs) * len(RISKS),
                "reference_verification_passed": verification["passed"],
                "any_cell_matches_llm_shape": bool(shape_matching),
                "any_well_mixed_cell_matches_llm_shape": bool(shape_matching_well_mixed),
                "cells_matching_llm_shape_well_mixed": provenance[
                    "cells_matching_llm_shape_well_mixed"
                ],
                "mixing_suspect_cell_count": len(suspect_cells),
                "best_fit_per_route": best,
                "best_fit_per_route_well_mixed": best_tables[
                    "best_fit_per_route_well_mixed"
                ],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
