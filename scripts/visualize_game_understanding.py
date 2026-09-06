"""Create publication figures for the game-understanding pilot audit."""
from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np


INK = "#07111F"
MUTED = "#697586"
GRID = "#D9DEE5"
PAPER = "#F8F7F2"
CYAN = "#007E89"
CYAN_LIGHT = "#7CCFD1"
AMBER = "#E39A2D"

DOMAIN_ORDER = [
    "rule_recall",
    "stage_payoff",
    "state_reconstruction",
    "state_transition",
    "terminal_scoring",
    "expected_payoff",
]
DOMAIN_LABELS = {
    "rule_recall": "Rule recall",
    "stage_payoff": "Stage payoff",
    "state_reconstruction": "State reconstruction",
    "state_transition": "State transition",
    "terminal_scoring": "Terminal scoring",
    "expected_payoff": "Expected payoff",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def configure() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 15,
            "axes.labelcolor": MUTED,
            "axes.edgecolor": GRID,
            "xtick.color": MUTED,
            "ytick.color": INK,
            "text.color": INK,
            "figure.facecolor": PAPER,
            "axes.facecolor": PAPER,
            "savefig.facecolor": PAPER,
        }
    )


def accuracy_figure(rows: list[dict[str, Any]], output_dir: Path) -> None:
    """Horizontal paired bars: unaided task accuracy versus disclosed calculator."""
    cells: dict[tuple[str, str], list[bool]] = defaultdict(list)
    for row in rows:
        arm = "calculator" if row["condition"] == "calculator" else "unaided"
        cells[(row["domain"], arm)].append(bool(row["semantic_correct"]))
    y = np.arange(len(DOMAIN_ORDER))
    height = 0.33
    unaided = [sum(cells[(domain, "unaided")]) / len(cells[(domain, "unaided")]) for domain in DOMAIN_ORDER]
    calculator = [sum(cells[(domain, "calculator")]) / len(cells[(domain, "calculator")]) for domain in DOMAIN_ORDER]

    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    ax.barh(y + height / 2, unaided, height, color=CYAN_LIGHT, edgecolor=CYAN, linewidth=0.8, label="Unaided prompt pool")
    ax.barh(y - height / 2, calculator, height, color=AMBER, edgecolor=INK, linewidth=0.6, label="Verified calculator disclosed")
    for values, offset in ((unaided, height / 2), (calculator, -height / 2)):
        for index, value in enumerate(values):
            ax.text(min(value + 0.018, 1.02), index + offset, f"{100 * value:.0f}%", va="center", fontsize=9, fontweight="bold")
    ax.set_yticks(y, [DOMAIN_LABELS[domain] for domain in DOMAIN_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(0, 1.08)
    ax.set_xticks(np.arange(0, 1.01, 0.2), [f"{value:.0%}" for value in np.arange(0, 1.01, 0.2)])
    ax.xaxis.grid(True, color=GRID, linewidth=0.7)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_title("Rule recall is not full-game comprehension", loc="left", pad=20)
    ax.text(0, 1.035, "Semantic accuracy by audit domain · Qwen2.5 7B F16 · 5 fixed-seed repetitions", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.legend(frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.47, -0.17))
    fig.text(0.01, 0.01, "Pilot diagnostic; repeated deterministic outputs measure reliability, not independent sample size. Calculator results are disclosed-answer tool uptake.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"game_understanding_accuracy.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def behavior_figure(rows: list[dict[str, str]], output_dir: Path) -> None:
    """Grouped point-and-interval comparison of canonical and calculator prompts."""
    fig, (ax_rate, ax_payoff) = plt.subplots(1, 2, figsize=(11.5, 5.5), gridspec_kw={"width_ratios": [1.25, 1]})
    risks = [0.1, 0.6, 0.9]
    conditions = [
        ("canonical", "Canonical", CYAN, -0.015),
        ("calculator_decision_card", "Calculator card", AMBER, 0.015),
    ]
    lookup = {(row["condition"], float(row["max_private_risk"])): row for row in rows}
    for condition, label, color, offset in conditions:
        selected = [lookup[(condition, risk)] for risk in risks]
        rates = np.array([float(row["unsafe_rate"]) for row in selected])
        lows = np.array([float(row["unsafe_rate_cluster_ci95_low"]) for row in selected])
        highs = np.array([float(row["unsafe_rate_cluster_ci95_high"]) for row in selected])
        x = np.array(risks) + offset
        ax_rate.errorbar(x, rates, yerr=[rates - lows, highs - rates], fmt="o-", color=color, markeredgecolor=INK, markeredgewidth=0.6, capsize=4, linewidth=2, markersize=7, label=label)
        payoffs = [float(row["mean_final_payoff"]) for row in selected]
        ax_payoff.plot(x, payoffs, "o-", color=color, markeredgecolor=INK, markeredgewidth=0.6, linewidth=2, markersize=7, label=label)
        alignment = "right" if condition == "canonical" else "left"
        label_shift = -0.004 if condition == "canonical" else 0.004
        for xx, value in zip(x, payoffs):
            ax_payoff.text(xx + label_shift, value + 2.2, f"{value:.1f}", ha=alignment, fontsize=8.5, fontweight="bold", color=color)

    for ax in (ax_rate, ax_payoff):
        ax.set_xticks(risks, ["10%", "60%", "90%"])
        ax.set_xlabel("Maximum private risk treatment")
        ax.grid(axis="y", color=GRID, linewidth=0.7)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
    ax_rate.set_ylim(0.35, 0.82)
    ax_rate.set_yticks(np.arange(0.4, 0.81, 0.1), [f"{value:.0%}" for value in np.arange(0.4, 0.81, 0.1)])
    ax_rate.set_ylabel("UNSAFE decisions")
    ax_rate.set_title("Behavioral action rate", loc="left")
    ax_payoff.set_ylim(0, 78)
    ax_payoff.set_ylabel("Mean final payoff (per player)")
    ax_payoff.set_title("Realized task payoff", loc="left")
    handles, labels = ax_rate.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=2, loc="lower center", bbox_to_anchor=(0.5, 0.045))
    fig.suptitle("A correct calculator changes context, not necessarily safety", x=0.06, ha="left", fontsize=16, fontweight="bold")
    fig.text(0.06, 0.91, "Paired pilot: 10 races per risk × condition; identical hidden horizons; cluster-bootstrap intervals by repetition", color=MUTED, fontsize=10)
    fig.text(0.01, 0.012, "Behavioral pilot only. The decision card enumerates current-round arithmetic but does not predict the opponent or terminal round.", color=MUTED, fontsize=8.5)
    fig.tight_layout(rect=(0, 0.15, 1, 0.88))
    for suffix in ("png", "pdf"):
        fig.savefig(output_dir / f"calculator_behavior_ablation.{suffix}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-outputs", type=Path, required=True)
    parser.add_argument("--analysis-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure()
    accuracy_figure(read_jsonl(args.probe_outputs), args.output_dir)
    behavior_figure(read_csv(args.analysis_dir / "behavior_by_risk.csv"), args.output_dir)
    print(f"Wrote publication figures to {args.output_dir}")


if __name__ == "__main__":
    main()
