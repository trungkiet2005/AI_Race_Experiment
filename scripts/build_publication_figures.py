#!/usr/bin/env python3
"""Build the generated portion of the manuscript figure set.

Figures 1 and 2 are author-supplied artwork and are deliberately not
regenerated. Figure 4 is generated from the checked-in first-five-trajectory
table with the same publication style as the other quantitative figures. The
former Figure 8 artwork is retained as an archived diagnostic but is not part
of the current manuscript figure set. All other exported figures read a
checked-in result table or an admitted raw run. New analyses that expand model
coverage must use a new output stem rather than changing a protected manuscript
figure.
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import FancyBboxPatch, Rectangle
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.build_manuscript_clustering_figures import (  # noqa: E402
    BASELINE_INPUTS,
    LABELS,
    flatten_features,
    load_trajectories,
)
from scripts.publication_style import (  # noqa: E402
    BLUE,
    BLUE_LIGHT,
    CATEGORICAL,
    FULL_WIDTH_IN,
    GOLD,
    GOLD_LIGHT,
    GREEN,
    GREEN_LIGHT,
    GRID,
    GREY_LIGHT,
    INK,
    LINE,
    MIN_TEXT_POINTS,
    MODEL_COLOURS,
    MODEL_MARKERS,
    MUTED,
    RED,
    RED_LIGHT,
    TEXT_WIDTH_IN,
    WHITE,
    configure_publication_style,
    panel_label,
    save_publication_figure,
    set_percent_axis,
    style_axis,
)

PAPER = ROOT / "figures" / "paper"
CLUSTER = PAPER / "llm_human_clustering"
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
EGT_TABLE = ROOT / "results" / "frontier" / "egt_frontier_comparison_v2" / "theory_llm_comparison.csv"
POSITION_TABLE = DATA / "nplayer_position_effect_by_persona.csv"
ARCHETYPE_TABLE = DATA / "human_cluster_summary.csv"
PROJECTION_TABLE = DATA / "llm_human_cluster_projection_unified.csv"
FRESH_FRONTIER_INPUT = ROOT / "results/kaggle-benchmarks/frontier_full_20260908/derived/ai_race_analysis/player_metrics.csv"
SCRIPTED_FIGURE_SCRIPT = ROOT / "scripts" / "figures" / "fig_scripted_opponent.py"
SCRIPTED_DERIVED = ROOT / "results" / "derived" / "scripted_opponent_campaign" / "risk_versus_rival.json"

MANUAL_FIGURE_FILES = {
    "figure_1_mechanism": [PAPER / "AIRaceOverview.pdf"],
    "figure_2_persona": [PAPER / "ExpOverview.pdf"],
}

BASELINE_ORDER = list(BASELINE_INPUTS) + ["human"]
# Filled by build_rate_figure so the provenance record can state the interval
# method and the cluster count actually used for each population.
RATE_FIGURE_CLUSTERS: dict[str, dict[str, object]] = {}
CLUSTER_NAMES = {
    0: "Persister",
    1: "Aggressive starter / reciprocator",
    2: "Cautious starter",
    3: "Reciprocal catch-up",
}
CLUSTER_COLOURS = {0: BLUE, 1: GREEN, 2: GOLD, 3: RED}
CLUSTER_FEATURES = [
    ("overall_unsafe_rate", "Overall Unsafe rate"),
    ("reciprocity", "Opponent reciprocity"),
    ("position_sensitivity", "Position sensitivity"),
    ("own_autocorrelation", "Own-action persistence"),
    ("first_round_unsafe", "First-round Unsafe"),
]


def _rounded_box(ax: plt.Axes, x: float, y: float, width: float, height: float,
                 *, face: str, edge: str = LINE, radius: float = 0.025,
                 linewidth: float = 1.0) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y), width, height,
            boxstyle=f"round,pad=0.008,rounding_size={radius}",
            facecolor=face, edgecolor=edge, linewidth=linewidth, zorder=1,
        )
    )


def _agent(ax: plt.Axes, x: float, y: float, *, colour: str, scale: float = 1.0) -> None:
    """Draw a small editable agent glyph, avoiding opaque stock artwork."""

    ax.add_patch(plt.Circle((x, y + 0.035 * scale), 0.028 * scale,
                            facecolor=colour, edgecolor=INK, linewidth=0.8, zorder=3))
    ax.add_patch(FancyBboxPatch((x - 0.045 * scale, y - 0.045 * scale),
                                0.09 * scale, 0.065 * scale,
                                boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=WHITE, edgecolor=INK, linewidth=0.8, zorder=3))
    ax.plot([x - 0.018 * scale, x + 0.018 * scale],
            [y - 0.012 * scale, y - 0.012 * scale], color=colour,
            linewidth=1.2, zorder=4)


def build_mechanism() -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.38),
                             gridspec_kw={"width_ratios": [1.42, 1.0, 1.22], "wspace": 0.28})
    ax = axes[0]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "A", "Repeated race")
    for y, label, colour, step, fill in [
        (0.68, "Safe", GREEN, "1.0 step", GREEN_LIGHT),
        (0.34, "Unsafe", RED, "1.5 steps + private risk", RED_LIGHT),
    ]:
        ax.plot([0.08, 0.84], [y, y], color=GRID, linewidth=2.0, zorder=0)
        ax.annotate("", xy=(0.84, y), xytext=(0.12, y),
                    arrowprops={"arrowstyle": "-|>", "color": colour, "lw": 1.6})
        _agent(ax, 0.14, y + 0.02, colour=colour, scale=0.72)
        ax.text(0.08, y + 0.095, label, color=colour, weight="bold", fontsize=8.5)
        ax.text(0.42, y + 0.095, step, color=MUTED, fontsize=8.0)
    ax.text(0.50, 0.14, "Simultaneous actions", ha="center", fontsize=8.0, color=INK)
    ax.text(0.50, 0.07, "hidden stopping time  |  prize B",
            ha="center", fontsize=8.0, color=INK)

    ax = axes[1]
    ax.set_xlim(-0.2, 2.2); ax.set_ylim(-0.2, 2.2); ax.axis("off")
    panel_label(ax, "B", "Two-player payoff")
    for i in range(2):
        for j in range(2):
            face = GREEN_LIGHT if i == 0 else RED_LIGHT
            ax.add_patch(Rectangle((j, 1 - i), 1, 1, facecolor=face,
                                   edgecolor=INK, linewidth=0.8))
    values = [["1.0", "0.6"], ["2.4", "2.0"]]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, 1.5 - i, values[i][j], ha="center", va="center",
                    fontsize=10.5, weight="bold")
    ax.text(0.5, 2.08, "Opponent", ha="center", fontsize=8.0, color=MUTED)
    ax.text(-0.14, 1.0, "Own", rotation=90, ha="center", va="center", fontsize=8.0, color=MUTED)
    ax.text(0.5, -0.05, "Safe", ha="center", fontsize=8.0, color=GREEN)
    ax.text(1.5, -0.05, "Unsafe", ha="center", fontsize=8.0, color=RED)
    ax.text(-0.05, 1.5, "Safe", ha="right", va="center", fontsize=8.0, color=GREEN)
    ax.text(-0.05, 0.5, "Unsafe", ha="right", va="center", fontsize=8.0, color=RED)

    ax = axes[2]
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "C", "N-player stage payoff")
    ax.text(0.50, 0.79, r"$D = k+s(N-k)$", ha="center", fontsize=10.0, weight="bold")
    _rounded_box(ax, 0.04, 0.25, 0.42, 0.35, face=GREEN_LIGHT, edge=GREEN)
    _rounded_box(ax, 0.54, 0.25, 0.42, 0.35, face=RED_LIGHT, edge=RED)
    ax.text(0.25, 0.49, "Safe", ha="center", color=GREEN, weight="bold", fontsize=8.5)
    ax.text(0.25, 0.36, r"$b/D-c$", ha="center", fontsize=11.0)
    ax.text(0.75, 0.49, "Unsafe", ha="center", color=RED, weight="bold", fontsize=8.5)
    ax.text(0.75, 0.36, r"$s\,b/D$", ha="center", fontsize=11.0)
    ax.text(0.50, 0.10, r"$b=4,\ c=1,\ s=1.5$; terminal prize is $B$", ha="center", fontsize=8.0, color=MUTED)
    return save_publication_figure(fig, PAPER / "AIRaceOverview", formats=("pdf", "png", "svg"))


def build_persona() -> list[Path]:
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.18),
                             gridspec_kw={"width_ratios": [1.0, 1.45, 1.15], "wspace": 0.26})
    ax = axes[0]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "A", "No persona")
    _rounded_box(ax, 0.08, 0.28, 0.84, 0.43, face=GREY_LIGHT, edge=LINE)
    _agent(ax, 0.30, 0.48, colour=BLUE, scale=0.95)
    _agent(ax, 0.70, 0.48, colour=BLUE, scale=0.95)
    ax.text(0.50, 0.32, "same game prompt", ha="center", fontsize=8.0, color=MUTED)
    ax.text(0.50, 0.16, "baseline", ha="center", fontsize=8.0, weight="bold")

    ax = axes[1]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "B", "Risk-aware framing")
    _rounded_box(ax, 0.03, 0.20, 0.94, 0.57, face=GOLD_LIGHT, edge=GOLD)
    for idx in range(6):
        x = 0.10 + idx * 0.145
        colour = [BLUE, BLUE, GREEN, GREEN, RED, RED][idx]
        ax.add_patch(Rectangle((x, 0.42), 0.09, 0.16, facecolor=WHITE,
                               edgecolor=colour, linewidth=1.0))
        ax.text(x + 0.045, 0.50, f"R{idx + 1}", ha="center", va="center",
                fontsize=8.0, weight="bold", color=colour)
    ax.annotate("risk-averse", xy=(0.10, 0.32), xytext=(0.10, 0.32),
                ha="center", fontsize=8.0, color=MUTED)
    ax.annotate("risk-seeking", xy=(0.88, 0.32), xytext=(0.88, 0.32),
                ha="center", fontsize=8.0, color=MUTED)
    ax.text(0.50, 0.88, "six levels adapted from the gamble scale", ha="center", fontsize=8.0)
    ax.text(0.50, 0.10, "prompt condition, not a measured trait", ha="center", fontsize=8.0, color=MUTED)

    ax = axes[2]; ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")
    panel_label(ax, "C", "Social framing")
    ax.plot([0.50, 0.50], [0.19, 0.77], color=GRID, linewidth=1.0)
    for x, colour, title, sub in [
        (0.20, GREEN, "Cooperative", "mutual\nrestraint"),
        (0.80, RED, "Adversarial", "other's\nexpense"),
    ]:
        _agent(ax, x, 0.54, colour=colour, scale=0.72)
        ax.text(x, 0.30, title, ha="center", fontsize=8.0, weight="bold", color=colour)
        ax.text(x, 0.18, sub, ha="center", fontsize=8.0, color=MUTED, linespacing=1.0)
    return save_publication_figure(fig, PAPER / "ExpOverview", formats=("pdf", "png", "svg"))


def build_egt() -> list[Path]:
    table = pd.read_csv(EGT_TABLE)
    required = {
        "max_private_risk", "theory_unsafe_main_reference", "theory_unsafe_reported_best_fit",
        "llm_primary_t0_unsafe_technology_race", "llm_primary_t0_unsafe_context_min",
        "llm_primary_t0_unsafe_context_max",
    }
    if not required.issubset(table.columns):
        raise RuntimeError(f"EGT table is missing columns: {sorted(required - set(table.columns))}")
    risks = table["max_private_risk"].to_numpy(float)
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.82),
                             gridspec_kw={"width_ratios": [1.18, 1.0], "wspace": 0.38})
    fig.subplots_adjust(left=0.12, right=0.99, top=0.86, bottom=0.24)
    ax = axes[0]
    ax.fill_between(risks, table["llm_primary_t0_unsafe_context_min"],
                    table["llm_primary_t0_unsafe_context_max"], color=GREY_LIGHT,
                    edgecolor=LINE, linewidth=0.6, alpha=0.95,
                    label="Prompt-context range", zorder=1)
    ax.plot(risks, table["theory_unsafe_main_reference"], color=BLUE, marker="o",
            linewidth=1.9, label="EGT reference", zorder=3)
    ax.plot(risks, table["theory_unsafe_reported_best_fit"], color=GOLD, marker="s",
            linewidth=1.5, linestyle="--", label="EGT reported fit", zorder=3)
    ax.scatter(risks, table["llm_primary_t0_unsafe_technology_race"], color=RED,
               marker="D", s=38, edgecolor=INK, linewidth=0.6,
               label="Prompted route", zorder=4)
    ax.set_xticks(risks, ["10%", "60%", "90%"])
    ax.set_xlabel("Maximum private setback risk")
    ax.set_ylabel("Unsafe rate")
    set_percent_axis(ax); style_axis(ax)
    panel_label(ax, "A", "Risk response")
    ax.legend(frameon=False, loc="lower left", fontsize=8, ncol=2,
              handlelength=1.6, columnspacing=0.9)

    ax = axes[1]
    delta = table["llm_primary_t0_unsafe_technology_race"] - table["theory_unsafe_main_reference"]
    colours = [RED if value < 0 else GREEN for value in delta]
    ax.axhline(0, color=INK, linewidth=0.8)
    bars = ax.bar(np.arange(len(risks)), delta, color=colours, alpha=0.88,
                  edgecolor=INK, linewidth=0.5, width=0.54, zorder=2)
    for bar, value in zip(bars, delta):
        y = value + (0.035 if value >= 0 else -0.045)
        va = "bottom" if value >= 0 else "top"
        ax.text(bar.get_x() + bar.get_width() / 2, y, f"{value:+.2f}",
                ha="center", va=va, fontsize=8.0, color=INK)
    ax.set_xticks(np.arange(len(risks)), ["10%", "60%", "90%"])
    ax.set_xlabel("Risk cap")
    ax.set_ylabel("Prompted route minus EGT")
    limit = max(0.45, float(np.max(np.abs(delta))) * 1.32)
    ax.set_ylim(-limit, limit)
    style_axis(ax)
    panel_label(ax, "B", "The gap changes sign")
    return save_publication_figure(fig, PAPER / "egt_theory_vs_llm_unsafe", formats=("pdf", "png", "svg"))


BOOTSTRAP_RESAMPLES = 4000
BOOTSTRAP_BASE_SEED = 20260908


def _cluster_labels(population: str, player_ids: np.ndarray) -> np.ndarray:
    """Return the independent experimental unit for each trajectory row.

    For a model population the unit is the race: both seats of a race share the
    sampled horizon and the private-setback draw under the common-random-number
    design, so they are not separate random samples. For the human population
    the unit is the participant, because each participant is one independent
    subject in the source study.
    """

    identifiers = np.asarray([str(value) for value in player_ids])
    if population == "human":
        return identifiers
    return np.asarray([value.split("::")[0] for value in identifiers])


def _cluster_bootstrap_interval(
    values: np.ndarray, clusters: np.ndarray, *, seed: int,
    resamples: int = BOOTSTRAP_RESAMPLES,
) -> tuple[float, float, float, int]:
    """Percentile bootstrap that resamples clusters, not single decisions.

    The point estimate is the unweighted mean over trajectory rows and is
    therefore identical to the previous row-level estimator; only the interval
    changes.
    """

    values = np.asarray(values, dtype=float)
    point = float(values.mean())
    unique, inverse = np.unique(np.asarray(clusters), return_inverse=True)
    n_clusters = int(unique.size)
    if n_clusters < 2:
        return point, float("nan"), float("nan"), n_clusters
    sums = np.zeros(n_clusters, dtype=float)
    counts = np.zeros(n_clusters, dtype=float)
    np.add.at(sums, inverse, values)
    np.add.at(counts, inverse, 1.0)
    rng = np.random.default_rng(seed)
    picks = rng.integers(0, n_clusters, size=(resamples, n_clusters))
    draws = sums[picks].sum(axis=1) / counts[picks].sum(axis=1)
    return (
        point,
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
        n_clusters,
    )


def build_rate_figure(frame: pd.DataFrame) -> list[Path]:
    rows = []
    cluster_report: dict[str, dict[str, object]] = {}
    for idx, population in enumerate(BASELINE_ORDER):
        subset = frame.loc[frame["population"].eq(population)]
        values = subset["unsafe_rate"].to_numpy(float)
        clusters = _cluster_labels(population, subset["player_id"].to_numpy())
        seed = BOOTSTRAP_BASE_SEED + idx
        point, low, high, n_clusters = _cluster_bootstrap_interval(
            values, clusters, seed=seed
        )
        rows.append((population, point, low, high))
        cluster_report[population] = {
            "cluster_unit": "participant_id" if population == "human" else "game_id",
            "n_clusters": n_clusters,
            "n_trajectories": int(values.size),
            "point_estimate": point,
            "ci_low": low,
            "ci_high": high,
            "seed": seed,
        }
    RATE_FIGURE_CLUSTERS.clear()
    RATE_FIGURE_CLUSTERS.update(cluster_report)
    race_clusters = sorted(
        {
            int(info["n_clusters"])
            for population, info in cluster_report.items()
            if population != "human"
        }
    )
    human_clusters = int(cluster_report["human"]["n_clusters"])
    fig, ax = plt.subplots(figsize=(TEXT_WIDTH_IN, 3.56))
    fig.subplots_adjust(left=0.28, right=0.98, top=0.96, bottom=0.31)
    y = np.arange(len(rows))[::-1]
    for yi, (population, mean, low, high) in zip(y, rows):
        colour = MODEL_COLOURS.get(population, INK)
        if np.isfinite(low):
            ax.plot([low, high], [yi, yi], color=colour, linewidth=2.0, zorder=2)
            ax.plot([low, low], [yi - 0.07, yi + 0.07], color=colour, linewidth=1.0)
            ax.plot([high, high], [yi - 0.07, yi + 0.07], color=colour, linewidth=1.0)
        ax.scatter(mean, yi, color=colour, marker=MODEL_MARKERS[population], s=42,
                   edgecolor=INK, linewidth=0.5, zorder=3)
        ax.text(min(1.115, (high if np.isfinite(high) else mean) + 0.025), yi,
                f"{mean:.0%}", va="center", fontsize=8.0, color=INK)
    labels = [LABELS[population] for population, *_ in rows]
    ax.set_yticks(y, labels)
    ax.set_xlim(0, 1.18); ax.set_xlabel("Mean Unsafe rate over rounds 1 to 5")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style_axis(ax, grid_axis="x")
    panel_label(ax, "A", "Aggregate level")
    race_text = (
        f"{race_clusters[0]} races" if len(race_clusters) == 1
        else f"{min(race_clusters)} to {max(race_clusters)} races"
    )
    fig.text(0.50, 0.012,
             "Points are mean Unsafe rates over player-race trajectories;\n"
             "whiskers are 95% race-clustered bootstrap intervals\n"
             f"({race_text} per model, {human_clusters} participants for humans; "
             f"{BOOTSTRAP_RESAMPLES} resamples).",
             ha="center", va="bottom", fontsize=8.0, color=MUTED, linespacing=1.25)
    return save_publication_figure(fig, CLUSTER / "05b_unsafe_rate_by_group", formats=("pdf", "png", "svg"))


def build_archetype_figure() -> list[Path]:
    summary = pd.read_csv(ARCHETYPE_TABLE)
    projection = pd.read_csv(PROJECTION_TABLE)
    if sorted(summary["cluster"].astype(int).tolist()) != [0, 1, 2, 3]:
        raise RuntimeError("Archetype summary must contain exactly four human clusters")
    if int(summary["n"].sum()) != 341:
        raise RuntimeError(f"Archetype summary must cover 341 human participants, found {summary['n'].sum()}")
    ordered_models = ["human"] + list(BASELINE_INPUTS)
    if not set(ordered_models).issubset(set(projection["model"])):
        raise RuntimeError("Projection table does not cover the manuscript model roster")
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.26),
                             gridspec_kw={"width_ratios": [1.0, 1.62], "wspace": 0.40})
    fig.subplots_adjust(left=0.13, right=0.99, top=0.90, bottom=0.34)
    ax = axes[0]
    values = summary.set_index("cluster")[[name for name, _ in CLUSTER_FEATURES]].to_numpy(float)
    means = values.mean(axis=0); scales = values.std(axis=0, ddof=0)
    z = np.divide(values - means, scales, out=np.zeros_like(values), where=scales > 1e-12)
    image = ax.imshow(z, cmap="RdBu_r", norm=TwoSlopeNorm(vmin=-1.8, vcenter=0, vmax=1.8), aspect="auto")
    for i in range(z.shape[0]):
        for j in range(z.shape[1]):
            ax.text(j, i, f"{z[i, j]:+.1f}", ha="center", va="center", fontsize=8.0,
                    color=INK if abs(z[i, j]) < 1.0 else WHITE)
    ax.set_xticks(np.arange(len(CLUSTER_FEATURES)), [label for _, label in CLUSTER_FEATURES], rotation=42, ha="right")
    ax.set_yticks(np.arange(4), [CLUSTER_NAMES[i] for i in range(4)])
    ax.set_xlabel("Feature mean, standardized across human archetypes")
    panel_label(ax, "A", "Human archetype signatures")
    ax.spines[:].set_visible(False)
    cbar = fig.colorbar(image, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Relative signature", fontsize=8.0)
    cbar.ax.tick_params(labelsize=8.0)

    ax = axes[1]
    bottom = np.zeros(len(ordered_models))
    for cluster in range(4):
        values = []
        for model in ordered_models:
            row = projection[(projection["model"] == model) & (projection["cluster"] == cluster)]
            if len(row) != 1:
                raise RuntimeError(f"Missing unique projection row for {model}, cluster {cluster}")
            values.append(float(row.iloc[0]["share"]) * 100)
        ax.bar(np.arange(len(ordered_models)), values, bottom=bottom,
               color=CLUSTER_COLOURS[cluster], edgecolor=WHITE, linewidth=0.6,
               label=CLUSTER_NAMES[cluster], hatch="//" if cluster == 2 else None)
        bottom += values
    ax.axvline(0.5, color=INK, linewidth=0.8, linestyle=(0, (3, 2)), alpha=0.8)
    ax.set_xticks(np.arange(len(ordered_models)), ["Human"] + [LABELS[m] for m in BASELINE_INPUTS], rotation=30, ha="right")
    ax.set_ylabel("Share nearest to human archetype")
    ax.set_ylim(0, 100)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(100, decimals=0))
    panel_label(ax, "B", "Model coverage of human diversity")
    style_axis(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, loc="lower center",
               bbox_to_anchor=(0.56, 0.015), ncol=2, fontsize=8.0)
    return save_publication_figure(fig, CLUSTER / "02_archetype_signatures_and_coverage", formats=("pdf", "png", "svg"))


def build_tsne(frame: pd.DataFrame) -> list[Path]:
    features = StandardScaler().fit_transform(flatten_features(frame))
    config = {"n_components": 2, "perplexity": 30, "random_state": 20260908,
              "init": "pca", "learning_rate": "auto", "max_iter": 1500}
    embedding = TSNE(**config).fit_transform(features)
    coordinates = frame[["population", "player_id", "unsafe_rate"]].copy()
    coordinates["tsne_x"] = embedding[:, 0]
    coordinates["tsne_y"] = embedding[:, 1]
    DATA.mkdir(parents=True, exist_ok=True)
    coordinates.to_csv(DATA / "tsne_coordinates.csv", index=False)
    xpad = max(0.1, np.ptp(embedding[:, 0]) * 0.04)
    ypad = max(0.1, np.ptp(embedding[:, 1]) * 0.04)
    xlim = (embedding[:, 0].min() - xpad, embedding[:, 0].max() + xpad)
    ylim = (embedding[:, 1].min() - ypad, embedding[:, 1].max() + ypad)
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 3.45))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.35, 1, 1, 1, 1], wspace=0.20, hspace=0.42,
                          left=0.025, right=0.995, top=0.95, bottom=0.05)
    placements = {
        "human": gs[:, 0],
        "gpt-5-nano": gs[0, 1], "gpt-5.4-nano": gs[0, 2],
        "google/gemini-3-flash-preview": gs[0, 3],
        "google/gemini-3.1-flash-lite-preview": gs[0, 4],
        "google/gemini-3.5-flash-lite": gs[1, 1],
        "claude-opus-5": gs[1, 2],
        "claude-sonnet-5": gs[1, 3],
    }
    for population, slot in placements.items():
        ax = fig.add_subplot(slot)
        focus = frame["population"].eq(population).to_numpy()
        ax.scatter(embedding[~focus, 0], embedding[~focus, 1], s=7, color=GRID,
                   alpha=0.48, linewidths=0, rasterized=True)
        ax.scatter(embedding[focus, 0], embedding[focus, 1], s=16,
                   color=MODEL_COLOURS[population], marker=MODEL_MARKERS[population],
                   alpha=0.88, edgecolor=WHITE, linewidth=0.25, rasterized=True)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
        ax.text(0.01, 1.02, LABELS[population], transform=ax.transAxes,
                fontsize=8.5 if population == "human" else 8.0, weight="bold",
                va="bottom", ha="left")
    return save_publication_figure(fig, CLUSTER / "01_tsne_hero_human_left", formats=("pdf", "png", "svg"))


def build_distribution() -> list[Path]:
    from results.cross_model_pilot_synthesis import build_human_vs_llm_distribution_v3 as source

    data = {"human": source.human_values_by_risk()}
    for model in source.NEUTRAL_INPUTS:
        data[model] = source.llm_values_by_risk(model)
    densities = {
        model: {risk: np.clip(source.reflected_density(data[model][risk]), source.Y_FLOOR, None)
                for risk in source.RISK_LEVELS}
        for model in source.ROW_ORDER
    }
    ymax = max(value.max() for model in densities.values() for value in model.values()) * 1.25
    fig, axes = plt.subplots(3, 3, figsize=(TEXT_WIDTH_IN, 6.2), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.72, "wspace": 0.16})
    fig.subplots_adjust(left=0.13, right=0.98, top=0.88, bottom=0.16)
    legend_handles = {}
    for row_idx, (group_label, members) in enumerate(source.GROUPS):
        panel_rows = ["human"] + members
        for col_idx, risk in enumerate(source.RISK_LEVELS):
            ax = axes[row_idx, col_idx]
            for model in panel_rows:
                values = densities[model][risk]
                if model == "human":
                    ax.fill_between(source.GRID, source.Y_FLOOR, values, color=INK, alpha=0.10, zorder=1)
                ax.plot(source.GRID, values, color=MODEL_COLOURS.get(model, CATEGORICAL[0]),
                        linewidth=1.8 if model != "human" else 2.3,
                        label=LABELS.get(model, source.ROW_LABELS[model]), zorder=4 if model == "human" else 3)
            ax.set_xlim(0, 100); ax.set_yscale("log"); ax.set_ylim(source.Y_FLOOR, ymax)
            style_axis(ax)
            if row_idx == 0:
                ax.set_title(["10%", "60%", "90%"][col_idx], pad=5)
            if col_idx == 0:
                ax.set_ylabel("Density\n(log scale)")
                ax.text(-0.02, 1.18, group_label, transform=ax.transAxes, fontsize=8.2,
                        weight="bold", ha="left", va="bottom")
            if row_idx == 2:
                ax.set_xlabel("Unsafe rate (%)")
            letter = chr(ord("A") + row_idx * 3 + col_idx)
            ax.text(0.02, 1.03, letter, transform=ax.transAxes, fontsize=8.0, weight="bold")
            for handle, label in zip(*axes[row_idx, 0].get_legend_handles_labels()):
                legend_handles.setdefault(label, handle)
    fig.text(0.01, 0.995, "Distributional comparison", fontsize=10.0, weight="bold", va="top")
    fig.text(0.01, 0.972, "Human trajectories are broad; model checkpoints occupy narrower, model-specific bands.",
             fontsize=8.0, color=MUTED, va="top")
    fig.legend(list(legend_handles.values()), list(legend_handles.keys()), frameon=False,
               loc="lower center", bbox_to_anchor=(0.55, 0.015), ncol=4, fontsize=8.0)
    return save_publication_figure(fig, PAPER / "human_vs_llm_distribution", formats=("pdf", "png", "svg"))


def build_position() -> list[Path]:
    table = pd.read_csv(POSITION_TABLE)
    expected_models = ["gpt-5-nano", "gpt-5.4-nano"]
    if set(table["model"]) != set(expected_models):
        raise RuntimeError("N-player position source does not cover the two manuscript models")
    levels = ["none", "R1", "R2", "R3", "R4", "R5", "R6"]
    labels = ["Baseline", "R1", "R2", "R3", "R4", "R5", "R6"]
    fig, axes = plt.subplots(1, 2, figsize=(TEXT_WIDTH_IN, 2.95), sharey=True,
                             gridspec_kw={"wspace": 0.28})
    fig.subplots_adjust(left=0.24, right=0.98, top=0.90, bottom=0.27)
    for ax, model, title, colour in zip(axes, expected_models, ["GPT-5 nano", "GPT-5.4 nano"], [BLUE, RED]):
        sub = table[table["model"].eq(model)].set_index("persona").reindex(levels)
        xs = np.arange(len(levels))
        usable = sub["converged"].astype(bool) & sub["coef"].notna() & sub["se"].notna()
        if usable.any():
            ax.errorbar(xs[usable.to_numpy()], sub.loc[usable, "coef"],
                        yerr=1.96 * sub.loc[usable, "se"], fmt="o-", color=colour,
                        markerfacecolor=WHITE, markeredgecolor=colour, markeredgewidth=1.0,
                        linewidth=1.4, capsize=2.5, zorder=3)
        for x, level, ok in zip(xs, levels, usable):
            if not bool(ok):
                ax.plot(x, 0, marker="x", color=colour, markersize=7, markeredgewidth=1.3)
        ax.axhline(0, color=INK, linewidth=0.8)
        ax.set_xticks(xs, labels, rotation=25, ha="right")
        ax.set_xlabel("Assigned risk-aware persona")
        panel_label(ax, "A" if model == expected_models[0] else "B", title)
        style_axis(ax)
    axes[0].set_ylabel("Position coefficient for Unsafe choice\n(logit; own gap minus others)")
    fig.text(0.50, 0.035,
             "Whiskers are 95% model-based intervals; x marks non-estimable cells.",
             ha="center", fontsize=8.0, color=MUTED)
    return save_publication_figure(fig, PAPER / "11_relative_position_grouped_bars", formats=("pdf", "png", "svg"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _manual_figure_outputs() -> dict[str, list[Path]]:
    missing = [
        str(path.relative_to(ROOT))
        for paths in MANUAL_FIGURE_FILES.values()
        for path in paths
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "Protected manual figure assets are missing: " + ", ".join(missing)
        )
    return MANUAL_FIGURE_FILES.copy()


def _fresh_frontier_outputs() -> dict[str, list[Path]]:
    if not FRESH_FRONTIER_INPUT.is_file():
        return {}
    from scripts.build_fresh_frontier_risk_profiles import build_frontier_risk_profiles

    return {"fresh_frontier_risk_profiles": build_frontier_risk_profiles()}


def build_scripted_opponent() -> list[Path]:
    """Regenerate the full five-route rival-conditioned figure."""

    subprocess.run([sys.executable, str(SCRIPTED_FIGURE_SCRIPT)],
                   cwd=ROOT, check=True)
    return [PAPER / "scripted_opponent.pdf", PAPER / "scripted_opponent.png"]


def main() -> None:
    configure_publication_style()
    PAPER.mkdir(parents=True, exist_ok=True)
    CLUSTER.mkdir(parents=True, exist_ok=True)
    frame = load_trajectories()
    outputs = {
        **_manual_figure_outputs(),
        **_fresh_frontier_outputs(),
        "figure_3_egt": build_egt(),
        "figure_4_rate": build_rate_figure(frame),
        "scripted_opponent": build_scripted_opponent(),
        "figure_5_archetypes": build_archetype_figure(),
        "figure_6_tsne": build_tsne(frame),
        "figure_7_distribution": build_distribution(),
    }
    from scripts.build_supplementary_figures import build_supplementary_figures

    outputs.update(build_supplementary_figures(frame))
    source_paths = [EGT_TABLE, ARCHETYPE_TABLE, PROJECTION_TABLE]
    source_paths += [ROOT / "references" / "source_study_dataset" / "airace_deidentified_long.csv"]
    source_paths += [SCRIPTED_FIGURE_SCRIPT, SCRIPTED_DERIVED]
    provenance = {
        "generator": "scripts/build_publication_figures.py",
        "style_module": "scripts/publication_style.py",
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "figure_outputs": {
            key: [str(path.relative_to(ROOT)) for path in paths] for key, paths in outputs.items()
        },
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
        "manual_figure_sha256": {
            key: {str(path.relative_to(ROOT)): _sha256(path) for path in paths}
            for key, paths in MANUAL_FIGURE_FILES.items()
        },
        "figure_status": {
            key: ("manual" if key in MANUAL_FIGURE_FILES else "generated")
            for key in outputs
        },
        "manual_figure_origin": {
            "commit": "f0dde0d",
            "note": "Restored author-supplied artwork; protected from automated writers.",
        },
        "counts": {
            "first_five_trajectories": int(len(frame)),
            "archetype_human_participants": int(pd.read_csv(ARCHETYPE_TABLE)["n"].sum()),
            "nplayer_position_rows": int(len(pd.read_csv(POSITION_TABLE))),
        },
        "figure_5_change": "replaced unrecoverable HDBSCAN artwork with the maintained four-archetype human-reference projection",
        "figure_4_rate_uncertainty": {
            "ci_method": "percentile cluster bootstrap over independent experimental units",
            "statistic": "unweighted mean Unsafe rate over player-race trajectories, rounds 1 to 5",
            "point_estimate_unchanged": True,
            "resamples": BOOTSTRAP_RESAMPLES,
            "base_seed": BOOTSTRAP_BASE_SEED,
            "seed_rule": "base_seed + population index in BASELINE_ORDER",
            "cluster_unit": {
                "llm_populations": "game_id (race); both seats of a race share the sampled horizon and setback draw under the common-random-number design",
                "human_population": "participant_id (one independent subject per participant)",
            },
            "populations": RATE_FIGURE_CLUSTERS,
            "supersedes": "row-level bootstrap over player-race trajectories, which treated the two seats of a race as independent samples",
        },
    }
    out = DATA / "publication_figure_set_provenance.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"figures": len(outputs), "trajectories": len(frame), "provenance": str(out)}, indent=2))


if __name__ == "__main__":
    main()
