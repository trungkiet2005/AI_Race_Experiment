"""Mine descriptive insights from the validated EGT/frontier comparison.

The figure keeps evolutionary reconstruction and hosted model routes separate.
It is an analysis artifact, not a new inferential result: the hosted points are
decision-weighted summaries from two routes and the strategy labels are nearest
rule matches, not claims about latent policies.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "frontier" / "egt_frontier_comparison_v2"
COMPARISON = OUTPUT / "theory_llm_comparison.csv"
MODEL_SUMMARY = OUTPUT / "llm_strategy_summary_primary_t0.csv"

INK = "#172033"
MUTED = "#667085"
GRID = "#D9DEE8"
BLUE = "#3166C6"
ORANGE = "#E78434"
PINK = "#C7527A"
GOLD = "#C49A21"
STRATEGY_COLORS = {"AS": "#AEB8C8", "AU": ORANGE, "CS": BLUE, "CAS": PINK}
MODEL_COLORS = {"claude": PINK, "gemini": BLUE}


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "font.family": "serif",
            "font.serif": ["DejaVu Serif", "Times New Roman", "STIXGeneral"],
            "mathtext.fontset": "stix",
            "font.size": 9,
            "xtick.color": INK,
            "ytick.color": INK,
            "grid.color": GRID,
            "grid.linewidth": 0.5,
            "grid.alpha": 0.8,
            "axes.axisbelow": True,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 600,
        }
    )


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def build_derived(comparison: list[dict[str, str]], model_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    comp = {round(f(row, "max_private_risk"), 1): row for row in comparison}
    rows: list[dict[str, object]] = []
    for row in model_rows:
        risk = round(f(row, "max_private_risk"), 1)
        model = row["context"]
        theory = f(comp[risk], "theory_unsafe_main_reference")
        best_fit = f(comp[risk], "theory_unsafe_reported_best_fit")
        frontier = f(row, "unsafe_rate_decision_weighted")
        rows.append(
            {
                "model": model,
                "max_private_risk": risk,
                "frontier_unsafe_rate": frontier,
                "theory_main_reference": theory,
                "theory_reported_best_fit": best_fit,
                "frontier_minus_theory_main_pp": 100 * (frontier - theory),
                "frontier_minus_theory_best_fit_pp": 100 * (frontier - best_fit),
                "mean_minimum_mismatch_rate": f(row, "mean_minimum_mismatch_rate"),
                "unique_classification_rate": f(row, "unique_classification_rate"),
                "fractional_nearest_AS": f(row, "fractional_nearest_AS"),
                "fractional_nearest_AU": f(row, "fractional_nearest_AU"),
                "fractional_nearest_CS": f(row, "fractional_nearest_CS"),
                "fractional_nearest_CAS": f(row, "fractional_nearest_CAS"),
            }
        )
    return rows


def plot(rows: list[dict[str, object]]) -> None:
    risk = np.array([0.1, 0.6, 0.9])
    by_model = {
        model: [row for row in rows if row["model"] == model]
        for model in sorted({str(row["model"]) for row in rows})
    }
    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.4), constrained_layout=True)

    ax = axes[0, 0]
    reference = {round(float(row["max_private_risk"]), 1): row for row in rows if row["model"] == "claude"}
    ax.plot(
        risk,
        [100 * float(reference[x]["theory_main_reference"]) for x in risk],
        color=INK,
        marker="o",
        linewidth=2,
        label="EGT strong-selection reference",
    )
    ax.plot(
        risk,
        [100 * float(reference[x]["theory_reported_best_fit"]) for x in risk],
        color=GOLD,
        marker="D",
        linewidth=2,
        label="EGT reported best fit",
    )
    for model, model_data in by_model.items():
        ordered = sorted(model_data, key=lambda row: float(row["max_private_risk"]))
        ax.plot(
            [float(row["max_private_risk"]) for row in ordered],
            [100 * float(row["frontier_unsafe_rate"]) for row in ordered],
            color=MODEL_COLORS.get(model, BLUE),
            marker="s",
            linewidth=2,
            label=f"Frontier: {model.title()}",
        )
    ax.set(
        title="A. Risk response is model-specific",
        xlabel="Maximum private risk",
        ylabel="Unsafe rate (%)",
        ylim=(-3, 103),
        xticks=risk,
        xticklabels=["10%", "60%", "90%"],
    )
    ax.grid(axis="y")
    ax.legend(fontsize=8, loc="lower left")

    ax = axes[0, 1]
    width = 0.17
    offsets = {"claude": -width / 2, "gemini": width / 2}
    for model, model_data in by_model.items():
        ordered = sorted(model_data, key=lambda row: float(row["max_private_risk"]))
        ax.bar(
            risk + offsets.get(model, 0),
            [float(row["frontier_minus_theory_main_pp"]) for row in ordered],
            width=width,
            color=MODEL_COLORS.get(model, BLUE),
            label=model.title(),
        )
    ax.axhline(0, color=INK, linewidth=1)
    ax.set(
        title="B. Hosted routes depart from strong-selection EGT",
        xlabel="Maximum private risk",
        ylabel="Frontier minus EGT (percentage points)",
        xticks=risk,
        xticklabels=["10%", "60%", "90%"],
    )
    ax.grid(axis="y")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    x = np.arange(len(risk))
    for model, model_data in by_model.items():
        bottom = np.zeros(len(risk))
        for strategy in ("AS", "AU", "CS", "CAS"):
            values = np.array(
                [
                    next(
                        float(row[f"fractional_nearest_{strategy}"])
                        for row in model_data
                        if round(float(row["max_private_risk"]), 1) == float(level)
                    )
                    for level in risk
                ]
            )
            ax.bar(
                x + offsets.get(model, 0),
                values,
                width=width,
                bottom=bottom,
                color=STRATEGY_COLORS[strategy],
                edgecolor=MODEL_COLORS.get(model, BLUE),
                linewidth=0.8,
                label=strategy if model == sorted(by_model)[0] else "_nolegend_",
            )
            bottom += values
        ax.text(
            float(x[-1] + offsets.get(model, 0)),
            1.03,
            model.title(),
            ha="center",
            va="bottom",
            fontsize=8,
            color=MODEL_COLORS.get(model, BLUE),
            fontweight="bold",
        )
    ax.set(
        title="C. Nearest-rule composition by route",
        xlabel="Maximum private risk",
        ylabel="Fraction of classified trajectories",
        ylim=(0, 1.14),
        xticks=x,
        xticklabels=["10%", "60%", "90%"],
    )
    ax.legend(ncol=4, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    ax.grid(axis="y")

    ax = axes[1, 1]
    for model, model_data in by_model.items():
        ax.scatter(
            [100 * float(row["unique_classification_rate"]) for row in model_data],
            [100 * float(row["mean_minimum_mismatch_rate"]) for row in model_data],
            s=65,
            color=MODEL_COLORS.get(model, BLUE),
            label=model.title(),
        )
        for row in model_data:
            ax.annotate(
                f"{int(100 * float(row['max_private_risk']))}%",
                (100 * float(row["unique_classification_rate"]), 100 * float(row["mean_minimum_mismatch_rate"])),
                xytext=(4, 3),
                textcoords="offset points",
                fontsize=7,
            )
    ax.set(
        title="D. Behavioural labels are an imperfect lens",
        xlabel="Unique nearest-rule classification (%)",
        ylabel="Mean minimum mismatch (%)",
        xlim=(0, 105),
        ylim=(0, 35),
    )
    ax.grid(axis="both")
    ax.legend(fontsize=8)

    fig.suptitle("Evolutionary theory and frontier behaviour: a boundary analysis", fontsize=16, fontweight="bold", color=INK)
    fig.text(
        0.5,
        -0.015,
        "EGT is a faithful reconstruction; frontier points are descriptive self-play summaries. Nearest-rule labels do not establish latent strategies.",
        ha="center",
        fontsize=8.5,
        color=MUTED,
    )
    for ax in axes.flat:
        for spine in ax.spines.values():
            spine.set_color(INK)
    fig.savefig(OUTPUT / "egt_frontier_insights.pdf")
    fig.savefig(OUTPUT / "egt_frontier_insights.png", dpi=220)
    plt.close(fig)


def main() -> None:
    configure_style()
    comparison = read_csv(COMPARISON)
    model_rows = read_csv(MODEL_SUMMARY)
    rows = build_derived(comparison, model_rows)
    with (OUTPUT / "egt_frontier_insights.csv").open("w", encoding="utf-8", newline="") as handle:
        fields = list(rows[0])
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    summary = {
        "schema_version": "egt-frontier-insights-v1",
        "evidence_class": "descriptive_frontier_vs_faithful_egt_boundary",
        "source_artifacts": [
            str(COMPARISON.relative_to(ROOT)),
            str(MODEL_SUMMARY.relative_to(ROOT)),
            str((OUTPUT / "egttools_pinned_source_validation.json").relative_to(ROOT)),
        ],
        "routes": sorted({str(row["model"]) for row in rows}),
        "rows": len(rows),
        "interpretation": [
            "The frontier routes do not reproduce the sharp strong-selection EGT phase change.",
            "Route differences are large relative to the three risk checkpoints, so pooled endpoint claims are not justified.",
            "Nearest-rule classification is least unique where mismatch is largest; it is a diagnostic lens, not a latent-policy estimator.",
        ],
    }
    (OUTPUT / "egt_frontier_insights.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    report = """# Frontier/EGT insight mining

This report is generated from the validated EGT reconstruction and the two
admitted frontier routes. It is descriptive. The evolutionary process and the
LLM self-play process are different objects, and nearest-strategy labels are
not treated as latent policy recovery.

## Readout

- The strong-selection EGT reference predicts a sharp fall in Unsafe behaviour
  at high private risk, while both hosted routes retain substantial Unsafe
  behaviour at the same checkpoint.
- The hosted routes disagree materially at every risk checkpoint. This is a
  route heterogeneity result, not evidence for a provider-independent frontier
  policy.
- Strategy matching is strongest as a diagnostic at intermediate risk, but
  minimum mismatch and non-unique matches remain common. Aggregate Unsafe rate
  should therefore remain the headline behavioural measure.

The accompanying four-panel figure follows the archived game-theory visual
language: compact panels, explicit theory/reference layers, restrained colors,
vector PDF export, and a visible evidence-boundary caption.
"""
    (OUTPUT / "egt_frontier_insights.md").write_text(report, encoding="utf-8")
    plot(rows)
    print(json.dumps({"status": "complete", "output": str(OUTPUT), "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
