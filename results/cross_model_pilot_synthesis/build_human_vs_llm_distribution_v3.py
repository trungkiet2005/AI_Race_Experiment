#!/usr/bin/env python3
"""Redraw human_vs_llm_distribution.png as overlapping smoothed density curves.

Supersedes build_human_vs_llm_distribution_v2.py's 8-row small-multiples grid
per user feedback (too tall to place in the paper once it's smoothed instead
of binned bars) and a first single-axis attempt that overlaid all eight
populations directly (too cluttered -- 8 crossing wiggly lines is a spaghetti
chart regardless of color). This version splits the eight populations into
two families of ~3-4 plus Human in every panel: row 1 is the three original
baseline checkpoints (GPT-5 nano, GPT-5.4 nano, Gemini 3 Flash), row 2 is the
two Gemini "lite" checkpoints plus GPT-5.6 Luna/Terra. Each panel now overlays
at most 5 curves instead of 8, and the whole figure is 2 rows x 3 risk-level
columns -- compact enough for a paper page, still an overlapping+smoothed
design.

Every LLM curve is that checkpoint's neutral/no-persona lane. GPT-5.6
Luna/Terra used to be a persona-pooled exception (they had only the persona
sweep); they now have a real baseline + R0_neutral lane, so that caveat is
gone and all seven checkpoints are like-for-like.

Boundary reflection (mirroring data about 0 and 100 before evaluating the KDE)
avoids the density leaking past the valid 0-100% range, which a naive KDE
would do given how much mass several checkpoints have right at the 0 or 100
boundary.

The y-axis is log-scaled: checkpoint peak densities span nearly three orders
of magnitude (a spread-out human curve around ~18%/bin vs. a near-point-mass
LLM checkpoint around ~930%/bin), and a linear shared axis lets the tallest
spike flatten every other curve to a barely-visible line.

Categorical colors are the dataviz skill's validated reference palette slots
(not the project's ad hoc PALETTE, which had two hues -- cyan and teal -- too
close together per `validate_palette.py`), chosen per group to clear the
all-pairs CVD/normal-vision floors within each panel (checked directly with
the skill's validator, see the assignment comments below). Human is a
near-black neutral, deliberately outside the categorical gate: it's styled as
a reference (thicker line, filled) rather than competing on hue with the
LLM series.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import gaussian_kde

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "cross_model_pilot_synthesis" / "figures"
HUMAN_CSV = ROOT / "public_dataset" / "airace_deidentified_long.csv"
BEDROCK_MANTLE_ROOT = ROOT / "results" / "frontier" / "bedrock_mantle"

PALETTE = {
    "navy": "#0B132B", "slate": "#64748B", "grid": "#DCE3ED",
    # Reference-palette categorical slots (validate_palette.py --pairs all, see docstring).
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "green": "#008300", "violet": "#4a3aa7", "magenta": "#e87ba4",
    "human": "#3a3a38",
}
RISK_LEVELS = [0.1, 0.6, 0.9]
RISK_LABELS = {0.1: "Risk cap = 10%", 0.6: "Risk cap = 60%", 0.9: "Risk cap = 90%"}
# Each facet row's colours pass validate_palette.py's all-pairs CVD +
# normal-vision floors on their own. Rows never share an axes, so the last row
# reuses the first row's hues rather than inventing a 9th categorical slot.
GROUPS: list[tuple[str, list[str]]] = [
    ("Original baseline checkpoints", ["gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview"]),
    ("Lite / newer-generation checkpoints",
     ["google/gemini-3.1-flash-lite-preview", "google/gemini-3.5-flash-lite", "gpt-5.6-luna", "gpt-5.6-terra"]),
    ("Claude (Bedrock)", ["claude-opus-5", "claude-sonnet-5"]),
]

NEUTRAL_INPUTS = {
    "gpt-5-nano": ["results/frontier/openai/baseline/gpt-5-nano", "results/frontier/openai/persona/R0_neutral/gpt-5-nano"],
    "gpt-5.4-nano": ["results/frontier/openai/baseline/gpt-5.4-nano", "results/frontier/openai/persona/R0_neutral/gpt-5.4-nano"],
    "google/gemini-3-flash-preview": [
        "results/frontier/baseline/google-gemini-3-flash-preview",
        "results/frontier/persona/R0_neutral/google-gemini-3-flash-preview",
    ],
    "google/gemini-3.1-flash-lite-preview": ["results/frontier/baseline/google-gemini-3.1-flash-lite-preview"],
    "google/gemini-3.5-flash-lite": ["results/frontier/baseline/google-gemini-3.5-flash-lite"],
    "gpt-5.6-luna": ["results/frontier/bedrock_mantle/luna/baseline/openai.gpt-5.6-luna",
                      "results/frontier/bedrock_mantle/luna/persona/R0_neutral/openai.gpt-5.6-luna"],
    "gpt-5.6-terra": ["results/frontier/bedrock_mantle/terra/baseline/openai.gpt-5.6-terra",
                       "results/frontier/bedrock_mantle/terra/persona/R0_neutral/openai.gpt-5.6-terra"],
    "claude-opus-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-opus-5",
                       "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-opus-5"],
    "claude-sonnet-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-sonnet-5",
                         "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-sonnet-5"],
}
ROW_ORDER = ["human", "gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview",
             "google/gemini-3.1-flash-lite-preview", "google/gemini-3.5-flash-lite",
             "gpt-5.6-luna", "gpt-5.6-terra", "claude-opus-5", "claude-sonnet-5"]
ROW_LABELS = {
    "human": "Human", "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}
ROW_COLORS = {
    "human": PALETTE["human"], "gpt-5-nano": PALETTE["blue"], "gpt-5.4-nano": PALETTE["orange"],
    "google/gemini-3-flash-preview": PALETTE["aqua"], "google/gemini-3.1-flash-lite-preview": PALETTE["yellow"],
    "google/gemini-3.5-flash-lite": PALETTE["green"], "gpt-5.6-luna": PALETTE["violet"],
    "gpt-5.6-terra": PALETTE["magenta"],
    "claude-opus-5": PALETTE["blue"], "claude-sonnet-5": PALETTE["orange"],
}
GRID = np.linspace(0, 100, 401)


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5, "axes.titleweight": "bold",
        "axes.titlesize": 11, "axes.labelsize": 9, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def human_values_by_risk() -> dict[float, list[float]]:
    import pandas as pd
    df = pd.read_csv(HUMAN_CSV)
    per_participant = df.groupby(["participant_id", "max_private_risk"], as_index=False)["decision"].mean()
    out = {}
    for risk, g in per_participant.groupby("max_private_risk"):
        out[float(risk)] = (g["decision"] * 100).tolist()
    return out


def llm_values_by_risk(model: str) -> dict[float, list[float]]:
    out: dict[float, list[float]] = {r: [] for r in RISK_LEVELS}
    for d in NEUTRAL_INPUTS[model]:
        p = ROOT / d / "players.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                risk = float(row["max_private_risk"])
                if risk in out:
                    out[risk].append(float(row["unsafe_frequency"]) * 100)
    return out



def reflected_density(values: list[float], lo: float = 0.0, hi: float = 100.0) -> np.ndarray:
    """Boundary-reflected KDE, evaluated on GRID, scaled to %-per-10-point-bin units."""
    arr = np.asarray(values, dtype=float)
    n = len(arr)
    if n == 0:
        return np.zeros_like(GRID)
    std = arr.std()
    if n < 2 or std < 1e-6:
        # Degenerate/near-point-mass sample: render as a narrow synthetic bump
        # (a true delta function isn't drawable) rather than let gaussian_kde
        # raise on a singular covariance matrix.
        center = arr.mean()
        bw = 1.5
        density = np.exp(-0.5 * ((GRID - center) / bw) ** 2) / (bw * np.sqrt(2 * np.pi))
    else:
        kde = gaussian_kde(arr)
        density = kde(GRID) + kde(2 * lo - GRID) + kde(2 * hi - GRID)
    mass = np.trapezoid(density, GRID)
    if mass > 0:
        density = density / mass
    return density * 10 * 100  # probability mass per 10-point-wide bin, as a %


Y_FLOOR = 0.5  # log-scale floor (%/10-pt bin); density values are clipped up to this so log10 stays finite


def main() -> None:
    setup_plot()
    data: dict[str, dict[float, list[float]]] = {"human": human_values_by_risk()}
    for model in NEUTRAL_INPUTS:
        data[model] = llm_values_by_risk(model)

    # Raw (%-per-10pt-bin) densities: GPT-5 nano peaks around 115, Gemini 3 Flash
    # around 930 -- three orders of magnitude apart from the flatter curves. A
    # log y-axis (not a second axis, not per-curve rescaling) is what lets every
    # curve in a panel sit directly on top of the others and still show shape,
    # instead of one spike flattening the rest.
    densities: dict[str, dict[float, np.ndarray]] = {
        row: {risk: np.clip(reflected_density(data[row][risk]), Y_FLOOR, None) for risk in RISK_LEVELS}
        for row in ROW_ORDER
    }
    ymax = max(d.max() for row in densities.values() for d in row.values()) * 1.3

    # One header strip + one plot row per facet group.
    n_g = len(GROUPS)
    fig = plt.figure(figsize=(13.5, 4.4 * n_g))
    gs = fig.add_gridspec(2 * n_g, 3, height_ratios=[0.46, 1] * n_g, hspace=0.42, wspace=0.12,
                           left=0.075, right=0.985, top=1 - 0.11 / n_g * 1.6, bottom=0.16 / n_g)

    for i, (group_label, members) in enumerate(GROUPS):
        panel_rows = ["human"] + members
        header_ax = fig.add_subplot(gs[2 * i, :])
        header_ax.axis("off")

        plot_row = []
        for j, risk in enumerate(RISK_LEVELS):
            ax = fig.add_subplot(gs[2 * i + 1, j])
            plot_row.append(ax)
            for row in panel_rows:
                y = densities[row][risk]
                n = len(data[row][risk])
                is_human = row == "human"
                if is_human:
                    ax.fill_between(GRID, Y_FLOOR, y, color=ROW_COLORS[row], alpha=0.15, zorder=1)
                ax.plot(GRID, y, color=ROW_COLORS[row], linewidth=2.8 if is_human else 1.8,
                        alpha=1.0 if is_human else 0.95, label=f"{ROW_LABELS[row]} (n={n})",
                        zorder=5 if is_human else 3)
            ax.set_xlim(0, 100)
            ax.set_yscale("log")
            ax.set_ylim(Y_FLOOR, ymax)
            ax.spines[["top", "right"]].set_visible(False)
            ax.grid(axis="y", which="major", color=PALETTE["grid"], linewidth=0.6)
            if i == 0:
                ax.set_title(RISK_LABELS[risk], fontsize=13)
            if i == n_g - 1:
                ax.set_xlabel("Unsafe rate (%)")
            if j == 0:
                ax.set_ylabel("Smoothed density (%/10-pt bin,\nlog scale)", fontsize=8.5)

        handles, labels = plot_row[0].get_legend_handles_labels()
        header_ax.text(0.5, 1.0, group_label, fontsize=11, fontweight="bold",
                        color=PALETTE["navy"], ha="center", va="top", transform=header_ax.transAxes)
        header_ax.legend(handles, labels, loc="lower center", ncol=len(panel_rows), frameon=False,
                          fontsize=8.7, bbox_to_anchor=(0.5, 0.0))

    fig.suptitle(
        "Humans are spread across the full range; every LLM checkpoint is concentrated\n"
        "(smoothed, overlapping density curves, split into three checkpoint families so each panel stays readable)",
        fontsize=13.5, y=0.985,
    )
    fig.savefig(FIGURES / "human_vs_llm_distribution.png", dpi=220, facecolor="white")
    fig.savefig(FIGURES / "human_vs_llm_distribution.pdf", facecolor="white")
    plt.close(fig)
    print("wrote", FIGURES / "human_vs_llm_distribution.png")


if __name__ == "__main__":
    main()
