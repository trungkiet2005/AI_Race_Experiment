"""Draw the main-paper within-population trajectory-diversity panel.

The diversity claim in the manuscript is quantitative, so it needs a figure a
reader can check a number against rather than an embedding they have to trust.
Both statistics come straight from
``results/cross_model_pilot_synthesis/data/trajectory_diversity_rarefaction.csv``:
Hill q=1, the effective number of distinct paired five-round action sequences,
and the mean pairwise Hamming distance between those sequences. The two answer
different questions. Hill q=1 counts how many distinct policies a population
produces and is blind to how far apart they are; Hamming measures how far apart
they are and is blind to how many there are. A population can score low on one
and high on the other, so reporting both is what makes "more diverse"
falsifiable.

Every cell is rarefied to 20 trajectories so the human sample, which is five to
seven times larger, cannot look more diverse merely by being bigger, and every
interval is a cluster bootstrap over races for the checkpoints and over
participants for the humans, because both seats of one race share its sampled
horizon and setback draw.

The figure is drawn at the measured single-column text width. It is separated
from the analysis script on purpose: the analysis owns the numbers, this owns
their presentation, so a presentation change can never move a published value.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from publication_style import (  # noqa: E402
    GRID,
    INK,
    MUTED,
    TEXT_WIDTH_IN,
    configure_publication_style,
    save_publication_figure,
    style_axis,
)

DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data" / "trajectory_diversity_rarefaction.csv"
STEM = ROOT / "figures" / "paper" / "trajectory_diversity_main"

RISKS = ("0.1", "0.6", "0.9")
RISK_MARKERS = {"0.1": "o", "0.6": "s", "0.9": "D"}

# Admission verdicts come from the frozen endpoint-admission campaign, so the
# figure can show which checkpoints were entitled to a behavioural reading at
# all. GPT-5 nano is no longer offered on the audited identity, so it has no
# verdict rather than a failing one.
ADMITTED = {"Gemini 3 Flash": True, "Claude Opus 5": True, "Claude Sonnet 5": True,
            "Gemini 3.1 Flash Lite": False, "Gemini 3.5 Flash Lite": False,
            "GPT-5.4 nano": False, "GPT-5 nano": None}

HUMAN = "Human"
ORDER = [HUMAN, "GPT-5.4 nano", "Gemini 3.5 Flash Lite", "Gemini 3 Flash",
         "GPT-5 nano", "Claude Sonnet 5", "Gemini 3.1 Flash Lite", "Claude Opus 5"]

COL_ADMITTED = "#2F6283"
COL_NOT_ADMITTED = "#C27A35"
COL_HUMAN = INK


def load() -> dict[tuple[str, str], dict[str, float]]:
    rows = {}
    with DATA.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            rows[(row["population"], row["risk_cap"])] = row
    return rows


def colour_for(population: str) -> str:
    if population == HUMAN:
        return COL_HUMAN
    return COL_ADMITTED if ADMITTED.get(population) else COL_NOT_ADMITTED


def draw_panel(ax, rows, value_key, low_key, high_key, xlabel, xlim) -> None:
    for index, population in enumerate(ORDER):
        y = len(ORDER) - 1 - index
        colour = colour_for(population)
        if population == HUMAN:
            # A reference band, so every checkpoint is read against the human
            # interval rather than against a single human point estimate.
            lows = [float(rows[(population, r)][low_key]) for r in RISKS]
            highs = [float(rows[(population, r)][high_key]) for r in RISKS]
            ax.axvspan(min(lows), max(highs), color=GRID, alpha=0.55, zorder=0, lw=0)
        for risk in RISKS:
            row = rows[(population, risk)]
            value = float(row[value_key])
            low = float(row[low_key])
            high = float(row[high_key])
            offset = {"0.1": 0.26, "0.6": 0.0, "0.9": -0.26}[risk]
            ax.plot([low, high], [y + offset] * 2, color=colour, lw=1.0,
                    solid_capstyle="butt", zorder=2, alpha=0.9)
            ax.plot([value], [y + offset], marker=RISK_MARKERS[risk], ms=3.6,
                    markerfacecolor=colour, markeredgecolor="white",
                    markeredgewidth=0.5, zorder=3, linestyle="none")
    ax.set_yticks(range(len(ORDER)))
    labels = []
    for population in reversed(ORDER):
        mark = {True: "✓", False: "✗", None: "–"}[ADMITTED.get(population)]
        labels.append(population if population == HUMAN else f"{population}  {mark}")
    ax.set_yticklabels(labels)
    ax.set_ylim(-0.6, len(ORDER) - 0.4)
    ax.set_xlim(*xlim)
    ax.set_xlabel(xlabel)
    style_axis(ax, grid_axis="x")


def main() -> None:
    configure_publication_style()
    rows = load()
    missing = [key for population in ORDER for key in [(population, r) for r in RISKS] if key not in rows]
    if missing:
        raise SystemExit(f"missing diversity cells: {missing}")

    # bbox_inches="tight" trims to the ink, so the saved width is the axes plus
    # the y labels, not the declared figsize. Declaring the difference keeps the
    # PDF at the column width, which is the only way the 8 pt floor enforced by
    # save_publication_figure is also the size that prints.
    fig, axes = plt.subplots(2, 1, figsize=(TEXT_WIDTH_IN - 0.35, 4.55))
    draw_panel(axes[0], rows, "q1_mean", "q1_ci_low", "q1_ci_high",
               "Effective distinct trajectories (Hill $q{=}1$, of 20)", (0, 21))
    draw_panel(axes[1], rows, "mean_pairwise_hamming", "hamming_ci_low", "hamming_ci_high",
               "Mean pairwise Hamming distance", (-0.02, 0.58))

    axes[0].set_title("a", loc="left", pad=6, fontsize=10.4, color=INK, fontweight="bold")
    axes[0].annotate("how many distinct policies a population produces",
                     xy=(0.0, 1.0), xycoords="axes fraction", xytext=(11, 2),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=9.0, color=MUTED, annotation_clip=False)
    axes[1].set_title("b", loc="left", pad=6, fontsize=10.4, color=INK, fontweight="bold")
    axes[1].annotate("how far apart those policies are",
                     xy=(0.0, 1.0), xycoords="axes fraction", xytext=(11, 2),
                     textcoords="offset points", ha="left", va="bottom",
                     fontsize=9.0, color=MUTED, annotation_clip=False)

    handles = [plt.Line2D([], [], marker=RISK_MARKERS[r], color=MUTED, lw=0, ms=3.6,
                          markerfacecolor=MUTED, markeredgecolor="white",
                          markeredgewidth=0.5, label=f"risk {r}") for r in RISKS]
    axes[1].legend(handles=handles, loc="lower right", frameon=False,
                   fontsize=8.2, handletextpad=0.4, borderaxespad=0.2, ncol=3,
                   columnspacing=1.0)

    fig.subplots_adjust(hspace=0.46)
    paths = save_publication_figure(fig, STEM, formats=("pdf", "png"))
    for path in paths:
        print(f"wrote {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
