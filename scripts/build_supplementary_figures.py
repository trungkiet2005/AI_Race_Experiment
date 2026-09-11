#!/usr/bin/env python3
"""Build the source-backed supplementary figure set.

The main-paper builder owns the visual system, and this module extends the
same system to the six figures used by ``paper/supplementary.tex``.  Every
plot is rebuilt from a checked-in table, an admitted raw run, or the exact
coordinates exported by the main t-SNE build.  No legacy bitmap is used as a
source for a canonical supplementary figure.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.cm import ScalarMappable
from matplotlib.patches import FancyArrowPatch, Patch
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
    CATEGORICAL,
    FULL_WIDTH_IN,
    GOLD,
    GREEN,
    GRID,
    INK,
    LINE,
    MODEL_COLOURS,
    MODEL_MARKERS,
    MUTED,
    RED,
    WHITE,
    configure_publication_style,
    panel_label,
    save_publication_figure,
    style_axis,
)

PAPER = ROOT / "figures" / "paper"
CLUSTER = PAPER / "llm_human_clustering"
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
EGT = ROOT / "results" / "frontier" / "egt_frontier_comparison_v2"
HUMAN_CSV = ROOT / "references" / "source_study_dataset" / "airace_deidentified_long.csv"
SHAP_JSON = DATA / "feature_importance_results.json"
PERSONA_TABLE = DATA / "persona_role_gradient_extended.csv"
TSNE_TABLE = DATA / "tsne_coordinates.csv"

MODEL_ORDER = ["human"] + list(BASELINE_INPUTS)
SUPP_MODEL_ORDER = [
    "human",
    "gpt-5-nano",
    "gpt-5.4-nano",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.5-flash-lite",
    "claude-opus-5",
    "claude-sonnet-5",
]
PERSONA_MODEL_ORDER = [
    "gpt-5-nano",
    "gpt-5.4-nano",
    "google/gemini-3-flash-preview",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "claude-opus-5",
    "claude-sonnet-5",
]
PERSONA_LABELS = {
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(model: str) -> str:
    return LABELS.get(model, PERSONA_LABELS.get(model, model))


def _percent_heatmap(ax: plt.Axes, values: np.ndarray, row_labels: list[str],
                     col_labels: list[str], *, title: str, cmap: str,
                     vmin: float, vmax: float, fmt: str | None = None,
                     norm=None):
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=None if norm else vmin,
                      vmax=None if norm else vmax, norm=norm)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=38, ha="right")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    panel_label(ax, title[0], title[1:])
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if fmt:
        # A fixed size runs the values together once a panel has many columns,
        # and the widest string is what has to fit, not the average one.
        widest = max(len(fmt.format(v)) for v in values.ravel())
        per_tile_pt = 72.0 * ax.get_window_extent().width / ax.figure.dpi / values.shape[1]
        # The house floor is 8.0 pt drawn and this figure prints at 96%, so
        # the size is fixed and the panel is widened to fit it rather than
        # the type being shrunk to fit the panel.
        size = 8.0
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                text = fmt.format(value)
                red, green, blue, _ = image.cmap(image.norm(value))
                # Rec. 709 relative luminance; the 0.55 split is where white
                # type stops clearing a 4.5:1 contrast ratio on these maps.
                luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
                colour = WHITE if luminance < 0.55 else INK
                ax.text(j, i, text, ha="center", va="center", fontsize=size,
                        color=colour, weight="bold")
    return image


def build_egt_insights() -> list[Path]:
    from scripts.build_egt_frontier_insights import build_derived, read_csv

    rows = build_derived(
        read_csv(EGT / "theory_llm_comparison.csv"),
        read_csv(EGT / "llm_strategy_summary_primary_t0.csv"),
    )
    risks = np.array([0.1, 0.6, 0.9])
    by_model = {
        model: sorted((row for row in rows if row["model"] == model),
                      key=lambda row: float(row["max_private_risk"]))
        for model in sorted({str(row["model"]) for row in rows})
    }
    model_colours = {"claude": RED, "gemini": BLUE}
    strategy_colours = {"AS": CATEGORICAL[0], "AU": CATEGORICAL[1],
                        "CS": CATEGORICAL[2], "CAS": CATEGORICAL[3]}
    fig, axes = plt.subplots(2, 2, figsize=(FULL_WIDTH_IN, 4.8),
                             gridspec_kw={"hspace": 0.62, "wspace": 0.40})
    fig.subplots_adjust(left=0.10, right=0.98, top=0.90, bottom=0.17)

    ax = axes[0, 0]
    reference = {round(float(row["max_private_risk"]), 1): row for row in rows
                 if row["model"] == "claude"}
    ax.plot(risks, [100 * float(reference[x]["theory_main_reference"]) for x in risks],
            color=INK, marker="o", linewidth=1.5, label="EGT main reference")
    ax.plot(risks, [100 * float(reference[x]["theory_reported_best_fit"]) for x in risks],
            color=GOLD, marker="s", linewidth=1.4, linestyle="--", label="EGT reported best fit")
    for model, model_rows in by_model.items():
        ax.plot([float(row["max_private_risk"]) for row in model_rows],
                [100 * float(row["frontier_unsafe_rate"]) for row in model_rows],
                color=model_colours[model], marker="D", linewidth=1.6,
                label=f"Frontier: {model.title()}")
    ax.set(xticks=risks, xticklabels=["10%", "60%", "90%"],
           xlabel="Maximum private risk", ylabel="Unsafe rate (%)", ylim=(-3, 103))
    style_axis(ax); panel_label(ax, "A", "Risk response")
    ax.legend(frameon=False, fontsize=8.0, loc="lower left")

    ax = axes[0, 1]
    offsets = {"claude": -0.06, "gemini": 0.06}
    for model, model_rows in by_model.items():
        ax.plot(risks + offsets[model],
                [float(row["frontier_minus_theory_main_pp"]) for row in model_rows],
                color=model_colours[model], marker="o", linewidth=1.4,
                label=model.title())
    ax.axhline(0, color=INK, linewidth=0.8)
    ax.set(xticks=risks, xticklabels=["10%", "60%", "90%"],
           xlabel="Maximum private risk", ylabel="Frontier minus EGT (pp)")
    style_axis(ax); panel_label(ax, "B", "Departure from EGT")
    ax.legend(frameon=False, fontsize=8.0)

    ax = axes[1, 0]
    x = np.arange(len(risks))
    width = 0.30
    for model, model_rows in by_model.items():
        bottom = np.zeros(len(risks))
        for strategy in ("AS", "AU", "CS", "CAS"):
            values = np.array([float(row[f"fractional_nearest_{strategy}"])
                               for row in model_rows])
            ax.bar(x + offsets[model] * 2.0, values, width=width, bottom=bottom,
                   color=strategy_colours[strategy], edgecolor=WHITE, linewidth=0.6,
                   label=strategy if model == sorted(by_model)[0] else "_nolegend_")
            bottom += values
    ax.set(xticks=x, xticklabels=["10%", "60%", "90%"],
           xlabel="Maximum private risk", ylabel="Classified fraction", ylim=(0, 1.05))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style_axis(ax); panel_label(ax, "C", "Nearest-rule composition")
    fig.legend(
        [Patch(facecolor=strategy_colours[name], edgecolor=WHITE, label=name)
         for name in ("AS", "AU", "CS", "CAS")],
        ["AS", "AU", "CS", "CAS"], frameon=False, ncol=4, fontsize=8.0,
        loc="lower center", bbox_to_anchor=(0.5, 0.012),
    )

    ax = axes[1, 1]
    for model, model_rows in by_model.items():
        colour = model_colours[model]
        ax.scatter([100 * float(row["unique_classification_rate"]) for row in model_rows],
                   [100 * float(row["mean_minimum_mismatch_rate"]) for row in model_rows],
                   s=28, color=colour, marker="o", edgecolor=WHITE, linewidth=0.4,
                   label=model.title())
        for row in model_rows:
            ax.annotate(f"{int(100 * float(row['max_private_risk']))}%",
                        (100 * float(row["unique_classification_rate"]),
                         100 * float(row["mean_minimum_mismatch_rate"])),
                        xytext=(3, 3), textcoords="offset points", fontsize=8.0)
    ax.set(xlabel="Unique nearest-rule classification (%)",
           ylabel="Mean minimum mismatch (%)", xlim=(0, 105), ylim=(0, 35))
    style_axis(ax); panel_label(ax, "D", "Labels are a diagnostic lens")
    ax.legend(frameon=False, fontsize=8.0)
    fig.text(0.50, -0.028,
             "Frontier trajectories are descriptive; nearest-rule matches do not recover latent strategies.",
             ha="center", fontsize=8.0, color=MUTED)
    return save_publication_figure(fig, PAPER / "egt_frontier_insights",
                                   formats=("pdf", "png", "svg"))


def build_egt_invasion() -> list[Path]:
    # The numerical EGTtools run is already admitted and archived in this
    # directory.  This builder redraws its CSV output, so a Windows build
    # does not silently substitute a local proxy for the compiled EGTtools
    # implementation used to obtain the evidence.
    invasion = pd.read_csv(EGT / "egt_frontier_invasion.csv")
    required = {"max_private_risk", "focal_strategy", "opponent_strategy",
                "fixation_probability", "stationary_share", "above_neutral_drift"}
    if not required.issubset(invasion.columns):
        raise RuntimeError(f"EGTtools invasion artifact is missing {required - set(invasion.columns)}")
    strategies = ["AS", "AU", "CS", "CAS"]
    risks = [0.1, 0.6, 0.9]
    drift = 0.01
    colours = [CATEGORICAL[0], RED, BLUE, CATEGORICAL[3]]
    fig, axes = plt.subplots(1, 3, figsize=(FULL_WIDTH_IN, 2.95))
    fig.subplots_adjust(wspace=0.05, top=0.82, bottom=0.20)
    positions = {"AS": np.array([0.50, 0.82]), "AU": np.array([0.18, 0.30]),
                 "CS": np.array([0.82, 0.30]), "CAS": np.array([0.50, 0.12])}
    for axis, risk, label in zip(axes, risks, "ABC"):
        sub = invasion[invasion["max_private_risk"].round(1).eq(risk)]
        shares = sub.groupby("focal_strategy")["stationary_share"].first().reindex(strategies)
        if shares.isna().any():
            raise RuntimeError(f"Incomplete EGTtools node shares for risk={risk}")
        axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.set_aspect("equal"); axis.axis("off")
        # Draw only invasions above neutral drift. Curved arrows keep reciprocal
        # edges legible and make the directionality explicit.
        for row in sub.itertuples(index=False):
            if row.focal_strategy == row.opponent_strategy or not bool(row.above_neutral_drift):
                continue
            start = positions[row.opponent_strategy]
            end = positions[row.focal_strategy]
            delta = end - start
            normal = np.array([-delta[1], delta[0]])
            bend = 0.08 if strategies.index(row.focal_strategy) < strategies.index(row.opponent_strategy) else -0.08
            midpoint = (start + end) / 2 + bend * normal / (np.linalg.norm(normal) or 1.0)
            path = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=7,
                                   connectionstyle=f"arc3,rad={bend:.2f}",
                                   linewidth=0.7 + 1.6 * min(float(row.fixation_probability), 1.0),
                                   color=LINE, alpha=0.75, zorder=1)
            axis.add_patch(path)
            axis.text(midpoint[0], midpoint[1], f"{float(row.fixation_probability):.2f}",
                      fontsize=8.0, color=MUTED, ha="center", va="center",
                      bbox={"boxstyle": "round,pad=0.08", "facecolor": WHITE,
                            "edgecolor": "none", "alpha": 0.85}, zorder=3)
        for strategy, colour, share in zip(strategies, colours, shares.to_numpy(float)):
            point = positions[strategy]
            axis.scatter([point[0]], [point[1]], s=360 + 1300 * share,
                         color=colour, edgecolor=INK, linewidth=0.8, zorder=4)
            axis.text(point[0], point[1] + 0.005, strategy, ha="center", va="center",
                      fontsize=8.0, weight="bold", color=WHITE if strategy != "AS" else INK,
                      zorder=5)
            axis.text(point[0], point[1] - 0.105, f"{share:.0%}", ha="center", va="top",
                      fontsize=8.0, color=INK, zorder=5)
        axis.text(0.0, 1.06, label, transform=axis.transAxes, fontsize=9.2,
                  weight="bold", va="bottom", color=INK)
        axis.text(0.12, 1.06, rf"$r_{{\max}}={risk:.1f}$", transform=axis.transAxes,
                  fontsize=8.8, va="bottom", color=INK)
        axis.text(0.12, 1.00, f"dominant {strategies[int(np.argmax(shares.to_numpy()))]} ({shares.max():.0%})",
                  transform=axis.transAxes, fontsize=8.0, va="top", color=MUTED)
    fig.text(0.50, 0.035,
             r"Node area: finite-mutation main-reference chain ($\beta=2$, $\mu=0.02$); arrows: archived EGTtools fixation above neutral drift $1/Z$.",
             ha="center", fontsize=8.0, color=MUTED)
    return save_publication_figure(fig, PAPER / "egt_frontier_invasion",
                                   formats=("pdf", "png", "svg"))


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_own_risk_dependence() -> list[Path]:
    human = pd.read_csv(HUMAN_CSV)
    per_player = human.groupby(["participant_id", "risk_gamble_choice"], as_index=False)["decision"].mean()
    human_rows = []
    for choice in range(6):
        values = per_player.loc[per_player["risk_gamble_choice"].eq(choice), "decision"].to_numpy(float)
        mean, low, high = _bootstrap(values, 20260908 + choice)
        human_rows.append((choice, len(values), mean, low, high))
    persona = pd.read_csv(PERSONA_TABLE)
    persona = persona[persona["role"].isin([f"R{i}" for i in range(1, 7)])]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN, 3.35),
                             gridspec_kw={"width_ratios": [1.15, 1.52], "wspace": 0.32})
    fig.subplots_adjust(left=0.11, right=0.99, top=0.88, bottom=0.25)
    ax = axes[0]
    x = np.array([row[0] for row in human_rows])
    means = np.array([row[2] for row in human_rows])
    low = np.array([row[3] for row in human_rows])
    high = np.array([row[4] for row in human_rows])
    ax.errorbar(x, means, yerr=[means - low, high - means], fmt="o", color=INK,
                markerfacecolor=GREEN, markeredgecolor=INK, linewidth=1.2,
                capsize=2.5, label="Human")
    ax.plot(x, means, color=INK, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.7)
    ax.set(xticks=x,
           xticklabels=[f"{choice}\n(n={n})" for choice, n, *_ in human_rows],
           xlabel="Eckel--Grossman choice", ylabel="Unsafe rate", ylim=(0, 1.08))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style_axis(ax); panel_label(ax, "A", "Measured human disposition")

    ax = axes[1]
    for model in PERSONA_MODEL_ORDER:
        sub = persona[persona["model"].eq(model)].copy()
        if len(sub) != 6:
            raise RuntimeError(f"Expected six persona levels for {model}, found {len(sub)}")
        sub["level"] = sub["role"].str.replace("R", "", regex=False).astype(int)
        sub = sub.sort_values("level")
        dashed = model in {"gpt-5.6-luna", "gpt-5.6-terra"}
        ax.plot(sub["level"], sub["mean_unsafe_rate"], color=MODEL_COLOURS.get(model, CATEGORICAL[0]),
                marker=MODEL_MARKERS.get(model, "o"), markersize=4.2, linewidth=1.2,
                linestyle="--" if dashed else "-", label=_label(model))
    ax.set(xticks=np.arange(1, 7), xlabel="Assigned risk-aware persona",
           ylim=(0, 1.08))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style_axis(ax); panel_label(ax, "B", "Prompted model response")
    ax.legend(frameon=False, fontsize=8.0, loc="upper left", ncol=2)
    fig.text(0.50, 0.035,
             "Whiskers are participant-bootstrap 95% intervals; dashed lines mark routes outside the main baseline roster.",
             ha="center", fontsize=8.0, color=MUTED)
    return save_publication_figure(fig, CLUSTER / "09_human_vs_llm_own_risk_dependence",
                                   formats=("pdf", "png", "svg"))


def build_shap_heatmap() -> list[Path]:
    results = json.loads(SHAP_JSON.read_text(encoding="utf-8"))
    models = ["human"] + list(BASELINE_INPUTS) + ["gpt-5.6-luna", "gpt-5.6-terra"]
    # The JSON contains Claude routes in addition to the seven main baseline
    # inputs; preserve its complete, explicitly fitted population roster.
    models = [name for name in results if name != "human_with_demographics"]
    feature_order = ["own_prev_unsafe", "opponent_prev_unsafe", "progress_gap",
                     "max_private_risk", "round_number"]
    labels = {
        "own_prev_unsafe": "Own previous\nUnsafe",
        "opponent_prev_unsafe": "Opponent previous\nUnsafe",
        "progress_gap": "Progress\ngap",
        "max_private_risk": "Risk\ntreatment",
        "round_number": "Round\nnumber",
    }
    values = []
    for model in models:
        raw = results[model]["mean_abs_shap"]
        total = sum(float(raw[name]) for name in feature_order) or 1.0
        values.append([float(raw[name]) / total for name in feature_order])
    values = np.asarray(values)
    row_labels = [_label(model) for model in models]
    fig, ax = plt.subplots(figsize=(FULL_WIDTH_IN, 3.85))
    fig.subplots_adjust(left=0.25, right=0.98, top=0.84, bottom=0.26)
    image = _percent_heatmap(ax, values, row_labels, [labels[name] for name in feature_order],
                             title="A Predictive feature share", cmap="Blues", vmin=0,
                             vmax=float(values.max()), fmt="{:.0%}")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Share of mean |SHAP|", fontsize=8.0)
    cbar.ax.tick_params(labelsize=8.0)
    fig.text(0.50, 0.035,
             "Associational feature use by separately fitted forests; it is not a causal attribution.",
             ha="center", fontsize=8.0, color=MUTED)
    return save_publication_figure(fig, CLUSTER / "feature_importance_shap_heatmap",
                                   formats=("pdf", "png", "svg"))


def _ensure_tsne_table(frame: pd.DataFrame) -> pd.DataFrame:
    if TSNE_TABLE.is_file():
        coordinates = pd.read_csv(TSNE_TABLE)
        if len(coordinates) == len(frame) and {"tsne_x", "tsne_y"}.issubset(coordinates):
            return coordinates
    features = StandardScaler().fit_transform(flatten_features(frame))
    embedding = TSNE(n_components=2, perplexity=30, random_state=20260908,
                     init="pca", learning_rate="auto", max_iter=1500).fit_transform(features)
    coordinates = frame[["population", "player_id", "unsafe_rate"]].copy()
    coordinates["tsne_x"] = embedding[:, 0]
    coordinates["tsne_y"] = embedding[:, 1]
    DATA.mkdir(parents=True, exist_ok=True)
    coordinates.to_csv(TSNE_TABLE, index=False)
    return coordinates


def build_tsne_outcome(frame: pd.DataFrame) -> list[Path]:
    coordinates = _ensure_tsne_table(frame)
    x = coordinates["tsne_x"].to_numpy(float)
    y = coordinates["tsne_y"].to_numpy(float)
    rates = coordinates["unsafe_rate"].to_numpy(float)
    xpad = max(0.1, np.ptp(x) * 0.04); ypad = max(0.1, np.ptp(y) * 0.04)
    xlim = (x.min() - xpad, x.max() + xpad); ylim = (y.min() - ypad, y.max() + ypad)
    norm = matplotlib.colors.Normalize(0, 1)
    cmap = matplotlib.colormaps.get_cmap("RdYlGn_r")
    fig = plt.figure(figsize=(FULL_WIDTH_IN, 4.45))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.4, 1, 1, 1, 1], wspace=0.20,
                          hspace=0.45, left=0.03, right=0.985, top=0.91, bottom=0.09)
    placements = {
        "human": gs[:, 0],
        "gpt-5-nano": gs[0, 1], "gpt-5.4-nano": gs[0, 2],
        "google/gemini-3-flash-preview": gs[0, 3],
        "google/gemini-3.1-flash-lite-preview": gs[0, 4],
        "google/gemini-3.5-flash-lite": gs[1, 1],
        "claude-opus-5": gs[1, 2], "claude-sonnet-5": gs[1, 3],
    }
    for model, slot in placements.items():
        ax = fig.add_subplot(slot)
        focus = coordinates["population"].eq(model).to_numpy()
        if model == "human":
            ax.scatter(x, y, c=rates, cmap=cmap, norm=norm, s=13, edgecolor=WHITE,
                       linewidth=0.2, rasterized=True)
        else:
            ax.scatter(x[~focus], y[~focus], s=6, color=GRID, alpha=0.34,
                       linewidths=0, rasterized=True)
            ax.scatter(x[focus], y[focus], c=rates[focus], cmap=cmap, norm=norm,
                       s=16, marker=MODEL_MARKERS[model], edgecolor=WHITE,
                       linewidth=0.25, rasterized=True)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
        ax.text(0.01, 1.02, _label(model), transform=ax.transAxes, fontsize=8.0,
                weight="bold", va="bottom", ha="left")
    axes = fig.axes
    fig.text(0.03, 0.965, "A  Pooled outcome", fontsize=9.2, weight="bold", color=INK)
    fig.text(0.25, 0.965, "B  Population-specific panels", fontsize=9.2, weight="bold", color=INK)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.01, aspect=28)
    cbar.set_label("Own Unsafe rate", fontsize=8.0)
    cbar.ax.tick_params(labelsize=8.0)
    return save_publication_figure(fig, CLUSTER / "06_tsne_safe_unsafe",
                                   formats=("pdf", "png", "svg"))


def build_round_profiles(frame: pd.DataFrame) -> list[Path]:
    action_labels = [f"Own R{i}" for i in range(1, 6)] + [f"Opp R{i}" for i in range(1, 6)]
    gap_labels = [f"R{i}" for i in range(2, 6)]
    populations = ["human"] + list(BASELINE_INPUTS)
    action_values = []
    gap_values = []
    for population in populations:
        sub = frame[frame["population"].eq(population)]
        own = np.vstack(sub["own"].to_numpy())
        opponent = np.vstack(sub["opponent"].to_numpy())
        gaps = np.vstack(sub["gap"].to_numpy())[:, 1:]
        action_values.append(np.concatenate([own.mean(axis=0), opponent.mean(axis=0)]))
        # Signed gaps cancel when both focal-player roles are pooled.  The
        # absolute gap is the informative shared feature: how far apart the
        # two players are entering the next round.
        gap_values.append(np.abs(gaps).mean(axis=0))
    action_values = np.asarray(action_values)
    gap_values = np.asarray(gap_values)
    row_labels = [
        "Human", "GPT-5 nano", "GPT-5.4 nano", "Gemini 3 Flash",
        "Gemini 3.1", "Gemini 3.5", "Claude Opus", "Claude Sonnet",
    ]
    # The widened gutter grows the tight bounding box, and a figure that saves
    # wider than the column is one LaTeX will shrink, so the declared width is
    # reduced to land the saved file at the placement width.
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN - 0.75, 3.55),
                             gridspec_kw={"width_ratios": [3.05, 1.0], "wspace": 0.78})
    fig.subplots_adjust(left=0.145, right=0.985, top=0.84, bottom=0.27)
    image = _percent_heatmap(axes[0], action_values, row_labels, action_labels,
                             title="A Action profile", cmap="RdYlGn_r", vmin=0,
                             vmax=1, fmt="{:.0%}")
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.02)
    cbar.set_label("Unsafe probability", fontsize=8.0); cbar.ax.tick_params(labelsize=8.0)
    image = _percent_heatmap(axes[1], gap_values, row_labels, gap_labels,
                             title="B Absolute progress gap", cmap="Blues", vmin=0,
                             vmax=1.5, fmt="{:.1f}")
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.08, pad=0.05)
    cbar.set_label("Absolute step gap", fontsize=8.0); cbar.ax.tick_params(labelsize=8.0)
    fig.text(0.50, 0.035,
             "Each row is a population mean over complete first-five trajectories; absolute gap is measured entering rounds 2--5.",
             ha="center", fontsize=8.0, color=MUTED)
    return save_publication_figure(fig, CLUSTER / "03_radar_small_multiples_by_group_no_gap_r1",
                                   formats=("pdf", "png", "svg"))


def build_supplementary_figures(frame: pd.DataFrame | None = None) -> dict[str, list[Path]]:
    configure_publication_style()
    if frame is None:
        frame = load_trajectories()
    outputs = {
        "supplement_figure_1_egt_insights": build_egt_insights(),
        "supplement_figure_2_egt_invasion": build_egt_invasion(),
        "supplement_figure_3_own_risk": build_own_risk_dependence(),
        "supplement_figure_4_shap": build_shap_heatmap(),
        "supplement_figure_5_tsne_outcome": build_tsne_outcome(frame),
        "supplement_figure_6_round_profiles": build_round_profiles(frame),
    }
    source_paths = [
        EGT / "theory_llm_comparison.csv",
        EGT / "llm_strategy_summary_primary_t0.csv",
        EGT / "egt_expected_payoff_matrices.csv",
        EGT / "egt_stationary_summary.csv",
        EGT / "egt_frontier_invasion.csv",
        EGT / "egt_frontier_invasion.json",
        HUMAN_CSV,
        PERSONA_TABLE,
        SHAP_JSON,
        TSNE_TABLE,
    ]
    source_paths = [path for path in source_paths if path.is_file()]
    provenance = {
        "generator": "scripts/build_supplementary_figures.py",
        "style_module": "scripts/publication_style.py",
        "generated_utc": pd.Timestamp.utcnow().isoformat(),
        "figure_outputs": {key: [str(path.relative_to(ROOT)) for path in paths]
                           for key, paths in outputs.items()},
        "source_sha256": {str(path.relative_to(ROOT)): _sha256(path) for path in source_paths},
        "figure_notes": {
            "supplement_figure_1_egt_insights": "redrawn from validated frontier-v2 comparison tables",
            "supplement_figure_2_egt_invasion": "redrawn from archived egt_frontier_invasion.csv generated by EGTtools; numerical source is unchanged",
            "supplement_figure_3_own_risk": "participant-level human bootstrap and persona_role_gradient_extended.csv",
            "supplement_figure_4_shap": "redrawn from feature_importance_results.json",
            "supplement_figure_5_tsne_outcome": "same coordinates exported by the canonical main t-SNE builder",
            "supplement_figure_6_round_profiles": "redrawn from complete first-five trajectory table assembled by the canonical loader",
        },
    }
    out = DATA / "supplementary_figure_set_provenance.json"
    out.write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return outputs


def main() -> None:
    build_supplementary_figures()
    print(json.dumps({"status": "complete", "output": "figures/paper"}, indent=2))


if __name__ == "__main__":
    main()
