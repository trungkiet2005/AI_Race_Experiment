#!/usr/bin/env python3
"""Create polished Matplotlib visuals for the FH analysis storyline."""

from __future__ import annotations

import textwrap
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
VIS_DIR = FIGURES_DIR / "visual_storyboard"

BLUE = "#2F6B9A"
ORANGE = "#D9822B"
GOLD = "#C9A227"
OLIVE = "#6B7D3D"
PINK = "#B45A7C"
INK = "#263238"
MUTED = "#6B7280"
GRID = "#E6E8EB"
PAPER = "#FBFBF8"
WHITE = "#FFFFFF"

FAMILY_COLORS = {
    "family_chatgpt": BLUE,
    "family_gemini": ORANGE,
}
MODEL_COLORS = {
    "gpt-5-nano": BLUE,
    "gpt-5.4-nano": PINK,
    "google-gemini-3-flash-preview": ORANGE,
    "google-gemini-3.1-flash-lite-preview": GOLD,
    "google-gemini-3.5-flash-lite": OLIVE,
}
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google-gemini-3-flash-preview": "Gemini 3 Flash",
    "google-gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google-gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}
FAMILY_LABELS = {
    "family_chatgpt": "ChatGPT",
    "family_gemini": "Gemini",
}
MODEL_ORDER = list(MODEL_LABELS)


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": WHITE,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 10,
            "axes.edgecolor": "#D7DADF",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    path = VIS_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def pct_axis(ax: plt.Axes) -> None:
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{x:.0%}" for x in np.linspace(0, 1, 6)])


def add_subtitle(ax: plt.Axes, subtitle: str) -> None:
    ax.text(
        0,
        1.015,
        subtitle,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.5,
        color=MUTED,
    )


def title(ax: plt.Axes, text: str) -> None:
    ax.set_title(text, loc="left", pad=24, fontweight="bold")


def label_bars(ax: plt.Axes, bars, fmt="{:.0%}", dy=0.012) -> None:
    for bar in bars:
        height = bar.get_height()
        if np.isnan(height):
            continue
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + dy,
            fmt.format(height),
            ha="center",
            va="bottom",
            fontsize=9,
            color=INK,
        )


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DERIVED_DIR / name)


def plot_family_protocol() -> Path:
    experiment = load_csv("family_experiment_summary.csv")
    pivot = experiment.pivot_table(
        index="experiment_mode", columns="family", values="unsafe_rate", aggfunc="mean"
    ).reindex(["mode_baseline", "mode_risk_matrix", "mode_strategy_persona"])
    labels = ["Baseline", "Risk matrix", "Strategy persona"]

    fig, ax = plt.subplots(figsize=(9.5, 5.4))
    x = np.arange(len(pivot.index))
    width = 0.34
    for offset, family in [(-width / 2, "family_chatgpt"), (width / 2, "family_gemini")]:
        bars = ax.bar(
            x + offset,
            pivot[family],
            width=width,
            color=FAMILY_COLORS[family],
            label=FAMILY_LABELS[family],
            edgecolor="#1F2937",
            linewidth=0.4,
        )
        label_bars(ax, bars)
    pct_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Unsafe decisions")
    title(ax, "Unsafe Rate By Family And Protocol")
    add_subtitle(ax, "Completed non-duplicate turns; baseline compared with risk-aware/persona protocols.")
    ax.legend(loc="upper right", ncol=2)
    return save(fig, "01_family_protocol_unsafe.png")


def plot_model_first_later() -> Path:
    first_later = load_csv("model_first_vs_later_summary.csv")
    pivot = first_later.pivot_table(index="model_slug", columns="turn_phase", values="unsafe_rate")
    pivot = pivot.reindex(MODEL_ORDER)
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    x = np.arange(len(pivot.index))
    width = 0.34
    bars1 = ax.bar(
        x - width / 2,
        pivot["round_1"],
        width,
        color="#D7DDC6",
        edgecolor=OLIVE,
        linewidth=1.1,
        label="Round 1",
    )
    bars2 = ax.bar(
        x + width / 2,
        pivot["round_2plus"],
        width,
        color=[MODEL_COLORS[m] for m in pivot.index],
        edgecolor="#1F2937",
        linewidth=0.4,
        label="Round 2+",
    )
    label_bars(ax, bars1)
    label_bars(ax, bars2)
    pct_axis(ax)
    ax.set_xticks(x)
    ax.set_xticklabels([MODEL_LABELS[m] for m in pivot.index], rotation=22, ha="right")
    ax.set_ylabel("Unsafe decisions")
    title(ax, "First-Turn Saturation Versus Later-Turn Behavior")
    add_subtitle(ax, "Baseline only. Gemini starts at 100% unsafe in round 1, then remains high later.")
    ax.legend(loc="upper left", ncol=2)
    return save(fig, "02_model_first_vs_later.png")


def plot_model_lag_heatmap() -> Path:
    lag = load_csv("model_lag_summary.csv")
    pivot = lag.pivot_table(index="model_slug", columns="lag_profile", values="unsafe_rate")
    pivot = pivot.reindex(MODEL_ORDER)[["0/0", "0/1", "1/0", "1/1"]]

    fig, ax = plt.subplots(figsize=(8.6, 5.5))
    image = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(pivot.columns)))
    ax.set_xticklabels(["Both safe\n0/0", "Opponent unsafe\n0/1", "Own unsafe\n1/0", "Both unsafe\n1/1"])
    ax.set_yticks(np.arange(len(pivot.index)))
    ax.set_yticklabels([MODEL_LABELS[m] for m in pivot.index])
    ax.grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            value = pivot.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.0%}",
                ha="center",
                va="center",
                color=INK if value < 0.72 else WHITE,
                fontsize=10,
                fontweight="bold",
            )
    title(ax, "Unsafe Rate By Previous-Round Lag Profile")
    add_subtitle(ax, "Baseline round 2+ decisions. Columns show own/opponent previous unsafe state.")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0%", "50%", "100%"])
    return save(fig, "03_model_lag_heatmap.png")


def plot_round_dynamics() -> Path:
    rounds = load_csv("model_round_dynamics.csv")
    fig, ax = plt.subplots(figsize=(10.5, 5.7))
    for model in MODEL_ORDER:
        sub = rounds[rounds["model_slug"] == model].sort_values("round")
        ax.plot(
            sub["round"],
            sub["unsafe_rate"],
            marker="o",
            markersize=4,
            linewidth=2.0,
            label=MODEL_LABELS[model],
            color=MODEL_COLORS[model],
        )
    pct_axis(ax)
    ax.set_xlabel("Round")
    ax.set_ylabel("Unsafe decisions")
    title(ax, "Baseline Unsafe Dynamics By Model")
    add_subtitle(ax, "Later rounds have smaller support after games stop; read tail points with care.")
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=9)
    return save(fig, "04_round_dynamics.png")


def plot_tree_performance_roots() -> Path:
    roots = load_csv("model_tree_roots.csv")
    metrics = load_csv("model_tree_cv_metrics.csv")
    perf = (
        metrics.groupby(["family", "model_slug"], dropna=False)
        .agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"))
        .reset_index()
    )
    merged = roots.merge(perf, on=["family", "model_slug"], how="left")
    merged["model_slug"] = pd.Categorical(merged["model_slug"], categories=MODEL_ORDER, ordered=True)
    merged = merged.sort_values("model_slug")

    fig, (ax1, ax2) = plt.subplots(
        1,
        2,
        figsize=(12.5, 5.5),
        gridspec_kw={"width_ratios": [1.1, 1.25]},
    )
    y = np.arange(len(merged))
    ax1.hlines(y, 0.5, merged["roc_auc"], color=GRID, linewidth=5)
    ax1.scatter(
        merged["roc_auc"],
        y,
        s=150,
        color=[MODEL_COLORS[m] for m in merged["model_slug"].astype(str)],
        edgecolor=INK,
        linewidth=0.6,
        zorder=3,
    )
    for yi, auc in zip(y, merged["roc_auc"]):
        ax1.text(auc + 0.012, yi, f"{auc:.2f}", va="center", fontsize=10, color=INK)
    ax1.set_yticks(y)
    ax1.set_yticklabels([MODEL_LABELS[m] for m in merged["model_slug"].astype(str)])
    ax1.set_xlim(0.45, 0.96)
    ax1.set_xlabel("Cross-validated ROC-AUC")
    ax1.set_title("Tree Predictive Strength")
    ax1.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax1.grid(axis="x")
    ax1.grid(axis="y", visible=False)

    root_counts = merged["root_feature"].value_counts().sort_values()
    bars = ax2.barh(root_counts.index, root_counts.values, color=[BLUE, ORANGE, OLIVE, PINK][: len(root_counts)])
    for bar in bars:
        ax2.text(
            bar.get_width() + 0.05,
            bar.get_y() + bar.get_height() / 2,
            f"{int(bar.get_width())}",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
    ax2.set_xlim(0, max(root_counts.values) + 1)
    ax2.set_xlabel("Models using feature as tree root")
    ax2.set_title("Root Feature Frequency")
    ax2.grid(axis="x")
    ax2.grid(axis="y", visible=False)

    fig.suptitle("Model-Level Decision Trees", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.02,
        0.93,
        "Shallow trees on baseline round 2+ decisions; root feature shows the first split each model uses.",
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])
    return save(fig, "05_tree_performance_roots.png")


def plot_logit_forest() -> Path:
    coeff = load_csv("model_logit_coefficients.csv")
    coeff = coeff[coeff["term"].isin(["own_prev_unsafe", "opponent_prev_unsafe", "progress_gap_before"])].copy()
    coeff["model_slug"] = pd.Categorical(coeff["model_slug"], categories=MODEL_ORDER, ordered=True)
    coeff = coeff.sort_values(["model_slug", "term"])
    terms = {
        "own_prev_unsafe": "Own previous unsafe",
        "opponent_prev_unsafe": "Opponent previous unsafe",
        "progress_gap_before": "Progress gap before",
    }
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 5.4), sharey=True)
    for ax, term in zip(axes, terms):
        sub = coeff[coeff["term"] == term].sort_values("model_slug")
        y = np.arange(len(sub))
        x = sub["coef"].clip(-4, 4)
        low = sub["ci95_low"].clip(-4, 4)
        high = sub["ci95_high"].clip(-4, 4)
        ax.hlines(y, low, high, color=GRID, linewidth=5)
        ax.scatter(
            x,
            y,
            s=110,
            color=[MODEL_COLORS[m] for m in sub["model_slug"].astype(str)],
            edgecolor=INK,
            linewidth=0.5,
            zorder=3,
        )
        ax.axvline(0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(terms[term], fontsize=12)
        ax.set_xlim(-4, 4)
        ax.set_xlabel("Logit coefficient\n(clipped at +/-4)")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        if ax is axes[0]:
            ax.set_yticks(y)
            ax.set_yticklabels([MODEL_LABELS[m] for m in sub["model_slug"].astype(str)])
        else:
            ax.set_yticks(y)
            ax.tick_params(axis="y", left=False, labelleft=False)
    fig.suptitle("Model-Level Logit Checks", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(
        0.02,
        0.92,
        "Baseline round 2+ decisions. Extreme Gemini gap coefficients are clipped so signs remain readable.",
        ha="left",
        color=MUTED,
        fontsize=10,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    return save(fig, "06_model_logit_forest.png")


def plot_gap_thresholds() -> Path:
    gap = load_csv("robustness_gap_threshold_scan.csv")
    gap = gap[gap["scope"].isin(["family_chatgpt_baseline_round2plus", "family_gemini_baseline_round2plus"])]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    for scope, color, label in [
        ("family_chatgpt_baseline_round2plus", BLUE, "ChatGPT baseline"),
        ("family_gemini_baseline_round2plus", ORANGE, "Gemini baseline"),
    ]:
        sub = gap[gap["scope"] == scope].sort_values("threshold")
        ax.plot(
            sub["threshold"],
            sub["behind_minus_middle"],
            marker="o",
            linewidth=2.2,
            color=color,
            label=label,
        )
    ax.axhline(0, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlabel("Behind threshold")
    ax.set_ylabel("Unsafe-rate gap: behind minus middle")
    title(ax, "Progress-Gap Threshold Sensitivity")
    add_subtitle(ax, "Baseline round 2+. Positive values mean being behind is associated with more unsafe choices.")
    ax.legend(loc="upper right")
    ax.grid(axis="y")
    return save(fig, "07_gap_threshold_sensitivity.png")


def draw_card(ax: plt.Axes, xy, wh, title: str, value: str, detail: str, color: str) -> None:
    x, y = xy
    w, h = wh
    patch = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.018,rounding_size=0.025",
        linewidth=1,
        edgecolor="#E2E5E9",
        facecolor=WHITE,
    )
    ax.add_patch(patch)
    ax.text(x + 0.035, y + h - 0.055, title, fontsize=10, color=MUTED, ha="left", va="top")
    value_display = value.replace("_", " ")
    if len(value_display) > 17:
        value_display = textwrap.fill(value_display, 18)
    value_size = 25 if len(value) <= 8 else 20 if len(value) <= 14 else 17
    ax.text(
        x + 0.035,
        y + h * 0.52,
        value_display,
        fontsize=value_size,
        color=color,
        ha="left",
        va="center",
        fontweight="bold",
        linespacing=0.9,
    )
    ax.text(
        x + 0.035,
        y + 0.045,
        textwrap.fill(detail, 32),
        fontsize=8.4,
        color=INK,
        ha="left",
        va="bottom",
        linespacing=1.25,
    )


def plot_executive_summary() -> Path:
    family = load_csv("family_baseline_summary.csv")
    model = load_csv("model_first_vs_later_summary.csv")
    tree = load_csv("model_tree_roots.csv")
    gem_base = family.loc[family["family"] == "family_gemini", "unsafe_rate"].iloc[0]
    chat_base = family.loc[family["family"] == "family_chatgpt", "unsafe_rate"].iloc[0]
    gpt_low = model[(model["model_slug"] == "gpt-5-nano") & (model["turn_phase"] == "round_2plus")][
        "unsafe_rate"
    ].iloc[0]
    gpt_high = model[(model["model_slug"] == "gpt-5.4-nano") & (model["turn_phase"] == "round_2plus")][
        "unsafe_rate"
    ].iloc[0]
    root_text = tree["root_feature"].value_counts().idxmax()

    fig, ax = plt.subplots(figsize=(12.5, 6.2))
    ax.set_axis_off()
    ax.text(0.02, 0.96, "FH Analysis Visual Summary", fontsize=24, fontweight="bold", ha="left", va="top")
    ax.text(
        0.02,
        0.89,
        "Completed baseline behavior: family gap, model split, first-turn saturation, and tree/state diagnostics.",
        fontsize=11,
        color=MUTED,
        ha="left",
        va="top",
    )
    draw_card(
        ax,
        (0.02, 0.50),
        (0.30, 0.31),
        "Family baseline gap",
        f"{gem_base - chat_base:+.1%}",
        f"Gemini {gem_base:.1%} vs ChatGPT {chat_base:.1%}.",
        ORANGE,
    )
    draw_card(
        ax,
        (0.35, 0.50),
        (0.30, 0.31),
        "ChatGPT model split",
        f"{gpt_high - gpt_low:+.1%}",
        f"Round 2+: GPT-5.4 {gpt_high:.1%} vs GPT-5 {gpt_low:.1%}.",
        PINK,
    )
    draw_card(
        ax,
        (0.68, 0.50),
        (0.30, 0.31),
        "Most frequent model-tree root",
        root_text,
        "Per-model trees use different roots.",
        BLUE,
    )
    draw_card(
        ax,
        (0.02, 0.15),
        (0.30, 0.29),
        "Gemini first turn",
        "100%",
        "All Gemini baseline first-turn cells.",
        ORANGE,
    )
    draw_card(
        ax,
        (0.35, 0.15),
        (0.30, 0.29),
        "Best model-tree AUC",
        "0.91",
        "Gemini 3.1 Flash Lite tree fit.",
        GOLD,
    )
    draw_card(
        ax,
        (0.68, 0.15),
        (0.30, 0.29),
        "Use in paper",
        "model first",
        "Show models before family averages.",
        OLIVE,
    )
    return save(fig, "00_visual_summary.png")


def create_contact_sheet(paths: list[Path]) -> Path:
    images = [mpimg.imread(path) for path in paths]
    ncols = 2
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 5.8 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, img, path in zip(axes, images, paths):
        ax.imshow(img)
        ax.set_axis_off()
    for ax in axes[len(images) :]:
        ax.set_axis_off()
    fig.suptitle("FH Visual Storyboard", fontsize=24, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return save(fig, "fh_visual_storyboard_contact_sheet.png")


def main() -> None:
    setup_style()
    paths = [
        plot_executive_summary(),
        plot_family_protocol(),
        plot_model_first_later(),
        plot_model_lag_heatmap(),
        plot_round_dynamics(),
        plot_tree_performance_roots(),
        plot_logit_forest(),
        plot_gap_thresholds(),
    ]
    contact_sheet = create_contact_sheet(paths)
    print(f"Wrote visual storyboard to {contact_sheet}")


if __name__ == "__main__":
    main()
