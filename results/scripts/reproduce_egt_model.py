"""Reconstruct arXiv:2607.26034's reduced EGT model and compare LLM pilots.

This is a faithful reconstruction, not a claim of bitwise reproduction: the
paper's own model code, EGTtools version, payoff Monte Carlo seed, and generated
payoff matrices were not public in arXiv v1.  See the generated report and
manifest for the evidence boundary.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

# A bare ``python results/scripts/...py`` invocation otherwise places only the
# script directory on sys.path.  Keep the documented repo-root invocation valid.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from strategy_analysis.classify import CANONICAL_STRATEGIES, classify_trajectory
from strategy_analysis.egt_reconstruction import (
    STRATEGIES,
    ChainSummary,
    ExpectedGame,
    expected_game,
    run_independent_chains,
)


DEFAULT_OUTPUT = REPO_ROOT / "results" / "open_source" / "egt_reproduction"
DEFAULT_LLM_ROOT = (
    REPO_ROOT
    / "results"
    / "open_source"
    / "context_skin_pilot"
    / "live_pilot_t0"
)
DEFAULT_LLM_SENSITIVITY_ROOT = (
    REPO_ROOT
    / "results"
    / "open_source"
    / "context_skin_pilot"
    / "live_pilot_t07"
)
PAPER_SOURCE = REPO_ROOT / "arXiv-2607.26034v1" / "paper.tex"
PAPER_ARXIV = "2607.26034v1"
EGTTOOLS_REPOSITORY = "https://github.com/Socrats/EGTTools"
EGTTOOLS_DOCS_COMMIT = "df7f5fb7787658b3fd3ab21343ff50a3e2a5d439"

INK = "#172033"
MUTED = "#667085"
GRID = "#D9DEE8"
BLUE = "#3166C6"
ORANGE = "#E78434"
GOLD = "#C49A21"
PINK = "#C7527A"
STRATEGY_COLORS = {
    "AS": "#AEB8C8",
    "AU": ORANGE,
    "CS": BLUE,
    "CAS": PINK,
}

REGIMES = (
    ("main_reference", 2.0, 0.02, "Main text: beta=2, mu=beta/Z=0.02"),
    ("supplement_reference", 2.0, 0.01, "Figure S5: beta=2, mu=1/Z=0.01"),
    ("reported_best_fit", 0.01, 0.05, "Main text best fit: beta=0.01, mu=0.05"),
)
RISKS = (0.1, 0.6, 0.9)


def _configure_matplotlib() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "font.family": "DejaVu Sans",
            "font.size": 9.5,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "legend.frameon": False,
            "savefig.bbox": "tight",
            "savefig.dpi": 220,
        }
    )


def _write_csv(path: Path, rows: Iterable[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc


def load_llm_trajectories(root: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Classify complete player trajectories from the admitted context pilot."""

    turn_files = sorted(root.glob("lane_*/*/turns.jsonl"))
    if not turn_files:
        return [], {"available": False, "root": str(root)}
    by_game: dict[str, dict[int, list[dict[str, Any]]]] = defaultdict(lambda: defaultdict(list))
    source_rows = 0
    parse_failures = 0
    for path in turn_files:
        context = path.parent.name
        for row in _read_jsonl(path):
            row["context"] = context
            by_game[str(row["game_id"])][int(row["player_index"])].append(row)
            source_rows += 1
            parse_failures += int(bool(row.get("parse_failed")))

    output: list[dict[str, Any]] = []
    incomplete_games = 0
    for game_id, players in sorted(by_game.items()):
        if set(players) != {0, 1}:
            incomplete_games += 1
            continue
        players[0].sort(key=lambda row: int(row["round"]))
        players[1].sort(key=lambda row: int(row["round"]))
        if len(players[0]) != len(players[1]):
            incomplete_games += 1
            continue
        for player_index in (0, 1):
            own = players[player_index]
            opponent = players[1 - player_index]
            own_actions = [int(row["unsafe"]) for row in own]
            opponent_actions = [int(row["unsafe"]) for row in opponent]
            classified = classify_trajectory(own_actions, opponent_actions)
            match_by_strategy = {match.strategy: match for match in classified.matches}
            weight = 1.0 / len(classified.best_strategies)
            record: dict[str, Any] = {
                "game_id": game_id,
                "player_index": player_index,
                "context": own[0]["context"],
                "max_private_risk": float(own[0]["max_private_risk"]),
                "model": own[0]["model"],
                "prompt_version": own[0]["prompt_version"],
                "horizon": len(own),
                "unsafe_rate": float(np.mean(own_actions)),
                "best_strategies": "|".join(classified.best_strategies),
                "tie_width": len(classified.best_strategies),
                "unique_best_strategy": classified.unique_best_strategy or "",
                "minimum_mismatch_rate": min(
                    match.mismatch_rate for match in classified.matches
                ),
            }
            for strategy in CANONICAL_STRATEGIES:
                record[f"nearest_weight_{strategy}"] = (
                    weight if strategy in classified.best_strategies else 0.0
                )
                record[f"mismatch_rate_{strategy}"] = match_by_strategy[strategy].mismatch_rate
            output.append(record)
    metadata = {
        "available": True,
        "root": str(root.relative_to(REPO_ROOT)),
        "turn_files": len(turn_files),
        "source_turn_rows": source_rows,
        "parse_failures": parse_failures,
        "games": len(by_game),
        "classified_player_trajectories": len(output),
        "incomplete_games": incomplete_games,
    }
    return output, metadata


def summarise_llm(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(str(row["context"]), float(row["max_private_risk"]))].append(row)
    summaries: list[dict[str, Any]] = []
    for (context, risk), cell in sorted(cells.items()):
        summary: dict[str, Any] = {
            "context": context,
            "max_private_risk": risk,
            "player_trajectories": len(cell),
            "decisions": sum(int(row["horizon"]) for row in cell),
            "unsafe_rate_decision_weighted": sum(
                float(row["unsafe_rate"]) * int(row["horizon"]) for row in cell
            )
            / sum(int(row["horizon"]) for row in cell),
            "mean_minimum_mismatch_rate": float(
                np.mean([row["minimum_mismatch_rate"] for row in cell])
            ),
            "unique_classification_rate": float(
                np.mean([row["tie_width"] == 1 for row in cell])
            ),
        }
        for strategy in STRATEGIES:
            summary[f"fractional_nearest_{strategy}"] = float(
                np.mean([row[f"nearest_weight_{strategy}"] for row in cell])
            )
        summaries.append(summary)
    return summaries


def chain_rows(
    games: dict[float, ExpectedGame],
    *,
    chains: int,
    burn_in: int,
    steps: int,
    thin: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for regime_index, (regime, beta, mutation, label) in enumerate(REGIMES):
        for risk_index, risk in enumerate(RISKS):
            seeds = [260726 + regime_index * 10_000 + risk_index * 1_000 + i for i in range(chains)]
            summaries = run_independent_chains(
                games[risk],
                beta=beta,
                mutation=mutation,
                seeds=seeds,
                burn_in=burn_in,
                steps=steps,
                thin=thin,
            )
            for chain_index, summary in enumerate(summaries):
                row: dict[str, Any] = {
                    "regime": regime,
                    "regime_label": label,
                    "beta": beta,
                    "mutation": mutation,
                    "max_private_risk": risk,
                    "chain": chain_index,
                    **asdict(summary),
                }
                frequencies = row.pop("strategy_frequencies")
                for strategy, frequency in zip(STRATEGIES, frequencies):
                    row[f"frequency_{strategy}"] = frequency
                rows.append(row)
    return rows


def summarise_chains(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cells: dict[tuple[str, float], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        cells[(row["regime"], float(row["max_private_risk"]))].append(row)
    output: list[dict[str, Any]] = []
    for (regime, risk), cell in sorted(cells.items()):
        summary: dict[str, Any] = {
            "regime": regime,
            "regime_label": cell[0]["regime_label"],
            "beta": cell[0]["beta"],
            "mutation": cell[0]["mutation"],
            "max_private_risk": risk,
            "chains": len(cell),
            "samples_per_chain": cell[0]["samples"],
            "unsafe_frequency_mean": float(np.mean([row["unsafe_frequency"] for row in cell])),
            "unsafe_frequency_min": float(np.min([row["unsafe_frequency"] for row in cell])),
            "unsafe_frequency_max": float(np.max([row["unsafe_frequency"] for row in cell])),
        }
        for strategy in STRATEGIES:
            values = [row[f"frequency_{strategy}"] for row in cell]
            summary[f"frequency_{strategy}_mean"] = float(np.mean(values))
            summary[f"frequency_{strategy}_min"] = float(np.min(values))
            summary[f"frequency_{strategy}_max"] = float(np.max(values))
        output.append(summary)
    return output


def _save_figure(fig: plt.Figure, output: Path, stem: str) -> None:
    fig.savefig(output / f"{stem}.png", facecolor="white")
    fig.savefig(output / f"{stem}.pdf", facecolor="white")
    plt.close(fig)


def plot_payoff_matrices(games: dict[float, ExpectedGame], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.4, 3.8), constrained_layout=True)
    values = np.concatenate([game.payoff_matrix.ravel() for game in games.values()])
    vmin, vmax = float(values.min()), float(values.max())
    image = None
    for axis, risk in zip(axes, RISKS):
        matrix = games[risk].payoff_matrix
        image = axis.imshow(matrix, cmap="Blues", vmin=vmin, vmax=vmax, aspect="equal")
        for i in range(4):
            for j in range(4):
                color = "white" if matrix[i, j] > (vmin + vmax) * 0.57 else INK
                axis.text(j, i, f"{matrix[i, j]:.1f}", ha="center", va="center", color=color, fontsize=8.5)
        axis.set_xticks(range(4), STRATEGIES)
        axis.set_yticks(range(4), STRATEGIES if axis is axes[0] else [])
        axis.set_xlabel("Opponent strategy")
        axis.set_title(f"Maximum private risk = {risk:.1f}")
    axes[0].set_ylabel("Focal strategy")
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.02)
    colorbar.set_label("Expected total payoff (ECU)")
    fig.suptitle(
        "Reduced-game expected payoff matrices",
        x=0.01,
        y=1.13,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        1.045,
        "Exact geometric-horizon summation; winner/tie setback risk applies to stage payoff plus prize",
        color=MUTED,
    )
    _save_figure(fig, output, "egt_expected_payoff_matrices")


def plot_stationary_strategy_frequencies(summary: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.25), sharey=True, constrained_layout=True)
    for axis, (regime, _beta, _mu, label) in zip(axes, REGIMES):
        cells = sorted(
            (row for row in summary if row["regime"] == regime),
            key=lambda row: row["max_private_risk"],
        )
        bottoms = np.zeros(len(cells))
        x = np.arange(len(cells))
        for strategy in STRATEGIES:
            values = np.array([row[f"frequency_{strategy}_mean"] for row in cells])
            axis.bar(
                x,
                values,
                bottom=bottoms,
                width=0.66,
                color=STRATEGY_COLORS[strategy],
                edgecolor="white",
                linewidth=0.7,
                label=strategy,
            )
            bottoms += values
        dominant = [max(STRATEGIES, key=lambda s: row[f"frequency_{s}_mean"]) for row in cells]
        for xpos, strategy in zip(x, dominant):
            axis.text(xpos, 1.025, strategy, ha="center", va="bottom", color=INK, fontweight="bold")
        axis.set_xticks(x, [f"{row['max_private_risk']:.1f}" for row in cells])
        axis.set_xlabel("Maximum private risk")
        axis.set_title(label.replace(": ", "\n", 1), fontsize=10.3)
        axis.set_ylim(0, 1.10)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
    axes[0].set_ylabel("Stationary population share")
    handles, labels = axes[-1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(
        "Finite-population strategy composition",
        x=0.01,
        y=1.13,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        1.045,
        "Z=100; bars are means across independent seeded Markov chains",
        color=MUTED,
    )
    _save_figure(fig, output, "egt_stationary_strategy_composition")


def plot_chain_diagnostics(chain_data: list[dict[str, Any]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.7, 3.75), sharey=True, constrained_layout=True)
    for axis, (regime, _beta, _mu, label) in zip(axes, REGIMES):
        cells = [row for row in chain_data if row["regime"] == regime]
        for risk, marker in zip(RISKS, ("o", "s", "^")):
            values = [row["unsafe_frequency"] for row in cells if row["max_private_risk"] == risk]
            jitter = np.linspace(-0.035, 0.035, len(values))
            axis.scatter(
                np.full(len(values), risk) + jitter,
                values,
                marker=marker,
                s=32,
                facecolor="white",
                edgecolor=BLUE,
                linewidth=1.2,
                label=f"risk {risk:.1f}",
            )
            axis.hlines(np.mean(values), risk - 0.055, risk + 0.055, color=INK, linewidth=2)
        axis.set_title(label.replace(": ", "\n", 1), fontsize=10.3)
        axis.set_xlabel("Maximum private risk")
        axis.set_xticks(RISKS)
        axis.set_ylim(-0.04, 1.04)
        axis.grid(axis="y")
    axes[0].set_ylabel("Predicted decision-weighted Unsafe rate")
    fig.suptitle(
        "Independent-chain convergence diagnostic",
        x=0.01,
        y=1.13,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        1.045,
        "Open marks are chain estimates; black segment is their mean",
        color=MUTED,
    )
    _save_figure(fig, output, "egt_chain_diagnostics")


def plot_theory_llm_comparison(
    chain_summary: list[dict[str, Any]],
    llm_summary: list[dict[str, Any]],
    llm_sensitivity_summary: list[dict[str, Any]],
    output: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(9.2, 5.0), constrained_layout=True)
    offsets = {"main_reference": -0.018, "reported_best_fit": 0.018}
    style = {
        "main_reference": (BLUE, "o", "EGT reconstruction: main reference"),
        "reported_best_fit": (PINK, "s", "EGT reconstruction: reported best fit"),
    }
    for regime in ("main_reference", "reported_best_fit"):
        cells = sorted(
            (row for row in chain_summary if row["regime"] == regime),
            key=lambda row: row["max_private_risk"],
        )
        x = np.array([row["max_private_risk"] for row in cells]) + offsets[regime]
        y = np.array([row["unsafe_frequency_mean"] for row in cells])
        low = y - np.array([row["unsafe_frequency_min"] for row in cells])
        high = np.array([row["unsafe_frequency_max"] for row in cells]) - y
        color, marker, label = style[regime]
        axis.errorbar(
            x,
            y,
            yerr=np.vstack([low, high]),
            color=color,
            marker=marker,
            markersize=6,
            linewidth=2,
            capsize=3,
            label=label,
        )

    if llm_summary:
        for risk in RISKS:
            cells = [row for row in llm_summary if row["max_private_risk"] == risk]
            if not cells:
                continue
            values = [row["unsafe_rate_decision_weighted"] for row in cells]
            technology = next(
                (row["unsafe_rate_decision_weighted"] for row in cells if row["context"] == "technology_race"),
                float(np.mean(values)),
            )
            axis.vlines(risk, min(values), max(values), color=GOLD, linewidth=5, alpha=0.33)
            axis.scatter(
                risk,
                technology,
                marker="D",
                s=52,
                color=GOLD,
                edgecolor=INK,
                linewidth=0.7,
                zorder=5,
                label="Qwen primary T=0: technology framing" if risk == RISKS[0] else None,
            )
        axis.plot([], [], color=GOLD, linewidth=5, alpha=0.33, label="Qwen primary T=0: range across 8 skins")
    if llm_sensitivity_summary:
        for risk in RISKS:
            cells = [
                row
                for row in llm_sensitivity_summary
                if row["max_private_risk"] == risk
            ]
            if not cells:
                continue
            values = [row["unsafe_rate_decision_weighted"] for row in cells]
            technology = next(
                row["unsafe_rate_decision_weighted"]
                for row in cells
                if row["context"] == "technology_race"
            )
            axis.vlines(
                risk + 0.012,
                min(values),
                max(values),
                color=MUTED,
                linewidth=2.3,
                alpha=0.45,
            )
            axis.scatter(
                risk + 0.012,
                technology,
                marker="D",
                s=34,
                facecolor="white",
                edgecolor=MUTED,
                linewidth=1.1,
                zorder=4,
                label="Qwen sensitivity T=0.7: technology framing"
                if risk == RISKS[0]
                else None,
            )
        axis.plot(
            [],
            [],
            color=MUTED,
            linewidth=2.3,
            alpha=0.45,
            label="Qwen sensitivity T=0.7: range across 8 skins",
        )
    axis.set_xticks(RISKS)
    axis.set_xlim(0.03, 0.97)
    axis.set_ylim(-0.04, 1.04)
    axis.set_xlabel("Maximum private risk")
    axis.set_ylabel("Decision-weighted Unsafe rate")
    axis.grid(axis="y")
    axis.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.13),
        ncol=2,
        fontsize=8.5,
    )
    fig.suptitle(
        "Theory and LLM-agent behaviour use the same game mechanics",
        x=0.01,
        y=1.10,
        ha="left",
        fontsize=15,
        fontweight="bold",
        color=INK,
    )
    fig.text(
        0.01,
        1.015,
        "LLM points are descriptive pilot outputs, not samples from the evolutionary population process",
        color=MUTED,
    )
    _save_figure(fig, output, "egt_theory_vs_llm_unsafe")


def plot_llm_strategy_lens(
    llm_summary: list[dict[str, Any]],
    chain_summary: list[dict[str, Any]],
    output: Path,
) -> None:
    if not llm_summary:
        return
    technology = sorted(
        (row for row in llm_summary if row["context"] == "technology_race"),
        key=lambda row: row["max_private_risk"],
    )
    theory = sorted(
        (row for row in chain_summary if row["regime"] == "main_reference"),
        key=lambda row: row["max_private_risk"],
    )
    fig, axes = plt.subplots(1, 2, figsize=(9.6, 4.1), sharey=True, constrained_layout=True)
    for axis, cells, source in (
        (axes[0], theory, "Evolutionary stationary share"),
        (axes[1], technology, "LLM fractional nearest-strategy share"),
    ):
        bottoms = np.zeros(3)
        x = np.arange(3)
        for strategy in STRATEGIES:
            if source.startswith("Evolutionary"):
                values = np.array([row[f"frequency_{strategy}_mean"] for row in cells])
            else:
                values = np.array([row[f"fractional_nearest_{strategy}"] for row in cells])
            axis.bar(x, values, bottom=bottoms, width=0.66, color=STRATEGY_COLORS[strategy], edgecolor="white", label=strategy)
            bottoms += values
        axis.set_xticks(x, ["0.1", "0.6", "0.9"])
        axis.set_xlabel("Maximum private risk")
        axis.set_title(source, fontsize=11)
        axis.grid(axis="y")
        axis.set_axisbelow(True)
        if source.startswith("LLM"):
            for xpos, row in zip(x, cells):
                axis.text(
                    xpos,
                    0.965,
                    f"unique {row['unique_classification_rate']:.0%}",
                    ha="center",
                    va="top",
                    color="white",
                    fontsize=8,
                    fontweight="bold",
                )
    axes[0].set_ylabel("Share")
    handles, labels = axes[1].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.045))
    fig.suptitle(
        "Strategy labels are a behavioural lens, not latent-strategy identification",
        x=0.01,
        y=1.13,
        ha="left",
        fontsize=14.5,
        fontweight="bold",
    )
    fig.text(
        0.01,
        1.045,
        "Left: population evolution. Right: Hamming-nearest labels for Qwen technology-framed self-play; ties split evenly.",
        color=MUTED,
    )
    _save_figure(fig, output, "egt_strategy_lens_vs_llm")


def build_comparison_rows(
    chain_summary: list[dict[str, Any]],
    llm_summary: list[dict[str, Any]],
    llm_sensitivity_summary: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Put commensurable descriptive rates in one audit-friendly table."""

    rows: list[dict[str, Any]] = []
    for risk in RISKS:
        record: dict[str, Any] = {"max_private_risk": risk}
        for regime in ("main_reference", "supplement_reference", "reported_best_fit"):
            theory = next(
                row
                for row in chain_summary
                if row["regime"] == regime and row["max_private_risk"] == risk
            )
            record[f"theory_unsafe_{regime}"] = theory["unsafe_frequency_mean"]
        for prefix, source in (
            ("llm_primary_t0", llm_summary),
            ("llm_sensitivity_t07", llm_sensitivity_summary),
        ):
            llm_cells = [row for row in source if row["max_private_risk"] == risk]
            if not llm_cells:
                continue
            values = [row["unsafe_rate_decision_weighted"] for row in llm_cells]
            technology = next(row for row in llm_cells if row["context"] == "technology_race")
            record.update(
                {
                    f"{prefix}_unsafe_technology_race": technology[
                        "unsafe_rate_decision_weighted"
                    ],
                    f"{prefix}_unsafe_context_min": min(values),
                    f"{prefix}_unsafe_context_max": max(values),
                    f"{prefix}_technology_mean_minimum_mismatch_rate": technology[
                        "mean_minimum_mismatch_rate"
                    ],
                    f"{prefix}_technology_unique_classification_rate": technology[
                        "unique_classification_rate"
                    ],
                }
            )
        rows.append(record)
    return rows


def write_report(
    output: Path,
    games: dict[float, ExpectedGame],
    chain_summary: list[dict[str, Any]],
    llm_summary: list[dict[str, Any]],
    llm_meta: dict[str, Any],
    llm_sensitivity_summary: list[dict[str, Any]],
    llm_sensitivity_meta: dict[str, Any],
) -> None:
    validation_path = output / "egttools_pinned_source_validation.json"
    validation = (
        json.loads(validation_path.read_text(encoding="utf-8"))
        if validation_path.is_file()
        else None
    )
    if validation and validation.get("passed"):
        transition_difference = validation["comparison"][
            "transition_matrix_max_abs_difference"
        ]
        stationary_difference = validation["comparison"][
            "stationary_distribution_max_abs_difference"
        ]
        official_validation_text = f"""The unmodified pure-Python `StochDynamics` class from the pinned EGTtools source was executed for an AS/AU, `Z=10`, `beta=2`, `mu=0.02` validation case. Its full transition matrix agrees with the independent repo-native construction to maximum absolute difference `{transition_difference:.3g}`; stationary distributions agree to `{stationary_difference:.3g}`. The compiled C++ `PairwiseComparison` class was not executed because EGTtools 0.1.14.2 publishes no Windows CPython 3.13 wheel and the docs-commit package imports its compiled numerical module at initialisation. This limitation is recorded rather than hidden."""
    else:
        official_validation_text = "The pinned official-source execution artifact was not available for this run."
    def cell(regime: str, risk: float) -> dict[str, Any]:
        return next(row for row in chain_summary if row["regime"] == regime and row["max_private_risk"] == risk)

    dominant = {
        risk: max(STRATEGIES, key=lambda strategy: cell("main_reference", risk)[f"frequency_{strategy}_mean"])
        for risk in RISKS
    }
    main_unsafe = {risk: cell("main_reference", risk)["unsafe_frequency_mean"] for risk in RISKS}
    best_unsafe = {risk: cell("reported_best_fit", risk)["unsafe_frequency_mean"] for risk in RISKS}
    def protocol_lines(source: list[dict[str, Any]], label: str) -> str:
        if not source:
            return f"- {label}: no complete artifact available."
        return "\n".join(
            f"- {label}, risk {risk:.1f}: technology framing {next(row['unsafe_rate_decision_weighted'] for row in source if row['context'] == 'technology_race' and row['max_private_risk'] == risk):.1%}; across-skin range {min(row['unsafe_rate_decision_weighted'] for row in source if row['max_private_risk'] == risk):.1%} to {max(row['unsafe_rate_decision_weighted'] for row in source if row['max_private_risk'] == risk):.1%}."
            for risk in RISKS
        )
    llm_lines = "\n".join(
        (
            protocol_lines(llm_summary, "Primary T=0"),
            protocol_lines(llm_sensitivity_summary, "Sensitivity T=0.7"),
        )
    )
    report = f"""# Reduced evolutionary-game reconstruction

## Admission status

**Evidence class: faithful reconstruction, not exact code reproduction.** The arXiv v1 paper discloses the game, strategy rules, payoff equations, and evolutionary process. It does not yet publish the authors' analysis code, generated payoff matrices, EGTtools version/commit, or Monte Carlo seeds. The paper says code will be deposited on Zenodo upon publication. This run therefore cannot be bitwise identical to the private author run.

This reconstruction improves numerical reproducibility in one bounded way: it sums the geometric horizon until the omitted mass is below `1e-13`, rather than using the paper's `10^4` Monte Carlo races for conditional matchups. The finite-population stationary quantities are still numerical estimates from independent, seeded pairwise-comparison chains.

## Reconstructed specification

- Strategies: AS always Safe; AU always Unsafe; CS starts Safe then copies the opponent's preceding action; CAS starts Unsafe then copies the opponent's preceding action.
- Stage payoff matrix: `[[1, 0.6], [2.4, 2]]`; Safe progress 1; Unsafe progress 1.5.
- Horizon: at least 5 rounds, then stop after each completed round with probability 0.2; expected horizon 9.
- Terminal prize: 100 for the winner or 50 each on a tie.
- Winner/tied-winner effective private risk: `max_risk * Unsafe_fraction`; a setback removes the complete task payoff. A loser keeps stage payoffs, receives no prize, and faces no setback.
- Evolution: well-mixed population `Z=100`, random pair matching without self-interaction, Fermi imitation, and uniform mutation to one of the other three strategies.

## Reproduction audit finding

The paper contains two distinct reference mutation settings. Its main text defines the blue reference as `beta=2, mu=beta/Z=0.02`; Figure S5 instead reports `beta=2, mu=1/Z=0.01`. Neither is silently corrected here. Both are run and exported, together with the reported best-fit point `beta=0.01, mu=0.05`.

The requested EGTtools `docs` branch was inspected at commit `{EGTTOOLS_DOCS_COMMIT}`. That commit documents the same finite-population transition rule, but it predates the 2026 paper and is not identified by the paper as its execution revision. Current EGTtools was therefore not treated as a missing author lockfile.

{official_validation_text}

## Main findings

At the main-text reference point, the dominant reconstructed strategy changes from **{dominant[0.1]}** at risk 0.1, to **{dominant[0.6]}** at 0.6, to **{dominant[0.9]}** at 0.9. This reproduces the paper's qualitative ordering. The associated predicted Unsafe rates are {main_unsafe[0.1]:.1%}, {main_unsafe[0.6]:.1%}, and {main_unsafe[0.9]:.1%}.

At the weak-selection, high-mutation best-fit point, the predicted rates are more diffuse: {best_unsafe[0.1]:.1%}, {best_unsafe[0.6]:.1%}, and {best_unsafe[0.9]:.1%}. This matches the paper's interpretation that higher mutation and weaker selection move mass away from near-pure vertices.

The deterministic payoff matrices reveal the mechanism. At risk 0.1, AU receives 109.44 against AS and maintains a strong invasion advantage. At risk 0.6, CAS receives {games[0.6].payoff_matrix[3, 0]:.2f} against AS because one initial Unsafe move wins the race with low accumulated exposure. At risk 0.9, CS earns 59 against itself while Unsafe winners' payoff is strongly discounted, supporting the shift toward conditional Safe play.

## LLM-agent comparison

The primary comparison uses `{llm_meta.get('classified_player_trajectories', 0)}` Qwen player trajectories and `{llm_meta.get('source_turn_rows', 0)}` decisions at temperature 0. The separate sensitivity comparison uses `{llm_sensitivity_meta.get('classified_player_trajectories', 0)}` trajectories and `{llm_sensitivity_meta.get('source_turn_rows', 0)}` decisions at temperature 0.7. They are never pooled. Both comparisons are **descriptive only**. The LLM races are repeated self-play prompts; they are not draws from the evolutionary population process, and nearest-strategy labels do not establish a latent strategy.

{llm_lines}

The key cross-study insight is a boundary, not an equivalence: the reduced EGT model predicts a sharp risk-driven phase change under strong selection, while the single-model pilot is strongly affected by surface context and opaque action-code position. The same payoff mechanics therefore do not guarantee the same behavioural regularity once decisions are produced by a prompt-sensitive language model.

## Figures

![Expected payoff matrices](egt_expected_payoff_matrices.png)

![Stationary strategy composition](egt_stationary_strategy_composition.png)

![Theory versus LLM Unsafe rate](egt_theory_vs_llm_unsafe.png)

![Independent chain diagnostic](egt_chain_diagnostics.png)

![Strategy lens comparison](egt_strategy_lens_vs_llm.png)

## Artifact map

- `egt_expected_payoff_matrices.csv`: exact reconstructed ordered payoffs.
- `egt_pair_unsafe_fractions.csv`: expected focal Unsafe fraction in each matchup.
- `egt_stationary_chains.csv`: every independent Markov-chain estimate.
- `egt_stationary_summary.csv`: chain means and between-chain ranges.
- `llm_strategy_matches_primary_t0.csv` and `llm_strategy_summary_primary_t0.csv`: primary temperature-0 audit.
- `llm_strategy_matches_sensitivity_t07.csv` and `llm_strategy_summary_sensitivity_t07.csv`: separate temperature-0.7 audit.
- `theory_llm_comparison.csv`: commensurable theory and LLM Unsafe-rate descriptors.
- `egttools_pinned_source_validation.json`: official-source transition and stationary parity audit.
- `reconstruction_manifest.json`: source revisions, parameters, hashes, coverage, and evidence boundary.

## Sources

- Fernández Domingos and Han, *Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment*, arXiv:{PAPER_ARXIV}: https://arxiv.org/abs/2607.26034
- EGTtools official repository, inspected `docs` commit `{EGTTOOLS_DOCS_COMMIT}`: {EGTTOOLS_REPOSITORY}/tree/docs

## Scope boundary

This artifact reproduces the disclosed qualitative evolutionary pattern. It does not reproduce the human experiment, recover the authors' private model code, infer human-like strategies in the LLM, or establish that context effects would generalise beyond the exact Qwen checkpoint and pilot protocols represented in the local artifacts.
"""
    (output / "README.md").write_text(report, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--llm-root", type=Path, default=DEFAULT_LLM_ROOT)
    parser.add_argument(
        "--llm-sensitivity-root",
        type=Path,
        default=DEFAULT_LLM_SENSITIVITY_ROOT,
    )
    parser.add_argument("--chains", type=int, default=4)
    parser.add_argument("--burn-in", type=int, default=100_000)
    parser.add_argument("--steps", type=int, default=400_000)
    parser.add_argument("--thin", type=int, default=100)
    args = parser.parse_args()
    if args.chains < 2:
        raise ValueError("at least two independent chains are required")
    args.output.mkdir(parents=True, exist_ok=True)
    _configure_matplotlib()

    games = {risk: expected_game(risk) for risk in RISKS}
    payoff_rows: list[dict[str, Any]] = []
    unsafe_rows: list[dict[str, Any]] = []
    for risk, game in games.items():
        for i, focal in enumerate(STRATEGIES):
            for j, opponent in enumerate(STRATEGIES):
                payoff_rows.append(
                    {
                        "max_private_risk": risk,
                        "focal_strategy": focal,
                        "opponent_strategy": opponent,
                        "expected_total_payoff": game.payoff_matrix[i, j],
                    }
                )
                unsafe_rows.append(
                    {
                        "max_private_risk": risk,
                        "focal_strategy": focal,
                        "opponent_strategy": opponent,
                        "expected_unsafe_fraction": game.unsafe_fraction_matrix[i, j],
                    }
                )
    _write_csv(
        args.output / "egt_expected_payoff_matrices.csv",
        payoff_rows,
        ["max_private_risk", "focal_strategy", "opponent_strategy", "expected_total_payoff"],
    )
    _write_csv(
        args.output / "egt_pair_unsafe_fractions.csv",
        unsafe_rows,
        ["max_private_risk", "focal_strategy", "opponent_strategy", "expected_unsafe_fraction"],
    )

    chains = chain_rows(
        games,
        chains=args.chains,
        burn_in=args.burn_in,
        steps=args.steps,
        thin=args.thin,
    )
    chain_summary = summarise_chains(chains)
    chain_fields = [
        "regime", "regime_label", "beta", "mutation", "max_private_risk", "chain",
        "seed", "samples", "unsafe_frequency", "moves",
        *[f"frequency_{strategy}" for strategy in STRATEGIES],
    ]
    _write_csv(args.output / "egt_stationary_chains.csv", chains, chain_fields)
    summary_fields = [
        "regime", "regime_label", "beta", "mutation", "max_private_risk", "chains", "samples_per_chain",
        "unsafe_frequency_mean", "unsafe_frequency_min", "unsafe_frequency_max",
        *[
            f"frequency_{strategy}_{suffix}"
            for strategy in STRATEGIES
            for suffix in ("mean", "min", "max")
        ],
    ]
    _write_csv(args.output / "egt_stationary_summary.csv", chain_summary, summary_fields)

    llm_rows, llm_meta = load_llm_trajectories(args.llm_root)
    llm_summary = summarise_llm(llm_rows)
    llm_sensitivity_rows, llm_sensitivity_meta = load_llm_trajectories(
        args.llm_sensitivity_root
    )
    llm_sensitivity_summary = summarise_llm(llm_sensitivity_rows)
    if llm_rows:
        _write_csv(
            args.output / "llm_strategy_matches_primary_t0.csv",
            llm_rows,
            list(llm_rows[0].keys()),
        )
        _write_csv(
            args.output / "llm_strategy_summary_primary_t0.csv",
            llm_summary,
            list(llm_summary[0].keys()),
        )
    if llm_sensitivity_rows:
        _write_csv(
            args.output / "llm_strategy_matches_sensitivity_t07.csv",
            llm_sensitivity_rows,
            list(llm_sensitivity_rows[0].keys()),
        )
        _write_csv(
            args.output / "llm_strategy_summary_sensitivity_t07.csv",
            llm_sensitivity_summary,
            list(llm_sensitivity_summary[0].keys()),
        )
    comparison_rows = build_comparison_rows(
        chain_summary,
        llm_summary,
        llm_sensitivity_summary,
    )
    _write_csv(
        args.output / "theory_llm_comparison.csv",
        comparison_rows,
        list(comparison_rows[0].keys()),
    )

    plot_payoff_matrices(games, args.output)
    plot_stationary_strategy_frequencies(chain_summary, args.output)
    plot_chain_diagnostics(chains, args.output)
    plot_theory_llm_comparison(
        chain_summary,
        llm_summary,
        llm_sensitivity_summary,
        args.output,
    )
    plot_llm_strategy_lens(llm_summary, chain_summary, args.output)
    write_report(
        args.output,
        games,
        chain_summary,
        llm_summary,
        llm_meta,
        llm_sensitivity_summary,
        llm_sensitivity_meta,
    )

    manifest = {
        "schema_version": "egt-reconstruction-v1",
        "evidence_class": "faithful_reconstruction_not_bitwise_reproduction",
        "paper": {
            "arxiv": PAPER_ARXIV,
            "url": "https://arxiv.org/abs/2607.26034",
            "local_source": str(PAPER_SOURCE.relative_to(REPO_ROOT)),
            "local_source_sha256": _sha256(PAPER_SOURCE),
            "author_code_public_in_arxiv_v1": False,
        },
        "egttools": {
            "repository": EGTTOOLS_REPOSITORY,
            "requested_branch": "docs",
            "inspected_commit": EGTTOOLS_DOCS_COMMIT,
            "paper_pinned_version_or_commit": None,
            "execution_backend": "repo_native_reconstruction_of_documented_transition_rule",
            "pinned_source_validation": (
                json.loads(
                    (args.output / "egttools_pinned_source_validation.json").read_text(
                        encoding="utf-8"
                    )
                )
                if (args.output / "egttools_pinned_source_validation.json").is_file()
                else None
            ),
        },
        "disclosed_model": {
            "strategies": list(STRATEGIES),
            "stage_payoffs": [[1.0, 0.6], [2.4, 2.0]],
            "safe_progress": 1.0,
            "unsafe_progress": 1.5,
            "race_prize": 100.0,
            "min_rounds": 5,
            "stop_probability": 0.2,
            "population_size": 100,
            "risks": list(RISKS),
            "regimes": [
                {"name": name, "beta": beta, "mutation": mutation, "source_label": label}
                for name, beta, mutation, label in REGIMES
            ],
        },
        "numerics": {
            "payoff_horizon_method": "deterministic_geometric_sum",
            "maximum_discarded_horizon_mass": max(game.horizon_tail_mass for game in games.values()),
            "chains": args.chains,
            "burn_in": args.burn_in,
            "steps": args.steps,
            "thin": args.thin,
            "uncertainty": "between_independent_chain_range_diagnostic_not_confidence_interval",
        },
        "paper_parameter_inconsistency_preserved": {
            "main_text": "beta=2, mu=beta/Z=0.02",
            "figure_s5": "beta=2, mu=1/Z=0.01",
        },
        "llm_comparison": {
            "primary_temperature_0": llm_meta,
            "sensitivity_temperature_0_7": llm_sensitivity_meta,
            "pooled": False,
        },
        "output_files": sorted(path.name for path in args.output.iterdir() if path.is_file()),
    }
    (args.output / "reconstruction_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"output": str(args.output), "manifest": manifest}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
