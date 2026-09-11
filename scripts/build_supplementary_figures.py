#!/usr/bin/env python3
"""Build the source-backed supplementary figure set.

The main-paper builder owns the visual system, and this module extends the
same system to the six figures used by ``paper/supplementary.tex``.  Every
plot is rebuilt from a checked-in table, an admitted raw run, or the exact
coordinates exported by the main t-SNE build.  No legacy bitmap is used as a
source for a canonical supplementary figure.

Four of the six are drawn here.  The two EGT figures are drawn by
``build_egt_frontier_insights.py`` and ``build_egt_frontier_invasion.py``,
which own them; this module only runs those builders so that a full
publication build still produces the complete set.  It used to keep its own
copies, and because every publication build reaches this module, each build
overwrote the maintained figures with the copies.

Route colours, markers and names come from ``figstyle``.  A second identity
table used to live here, and it collided with the main paper's: the same hue
and the same glyph meant different routes in the two halves of the same
document.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
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
    FULL_WIDTH_IN,
    GREEN,
    GRID,
    INK,
    MUTED,
    WHITE,
    configure_publication_style,
    panel_label,
    save_publication_figure,
    style_axis,
)

# Route identity comes from the main-paper style module and not from a second
# table kept here.  The two tables had drifted onto each other: grey meant
# GPT-5.4 nano in the main paper and Claude Sonnet 5 here, and the diamond meant
# Gemini 3 Flash there and Gemini 3.1 Flash Lite here, so a reader who learnt the
# routes in the paper read every supplementary figure wrong.  ``figstyle`` sets
# its own rcParams at import, which is a different policy from the one these
# figures are drawn under, so the import is fenced.
_RC_BEFORE_FIGSTYLE = matplotlib.rcParams.copy()
from scripts import figstyle as ROUTES  # noqa: E402

matplotlib.rcParams.update(_RC_BEFORE_FIGSTYLE)

PAPER = ROOT / "figures" / "paper"
CLUSTER = PAPER / "llm_human_clustering"
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
EGT = ROOT / "results" / "frontier" / "egt_frontier_comparison_v2"
HUMAN_CSV = ROOT / "references" / "source_study_dataset" / "airace_deidentified_long.csv"
SHAP_JSON = DATA / "feature_importance_results.json"
PERSONA_TABLE = DATA / "persona_role_gradient_extended.csv"
TSNE_TABLE = DATA / "tsne_coordinates.csv"

PERSONA_MODEL_ORDER = [
    "gpt-5-nano",
    "gpt-5.4-nano",
    "google/gemini-3-flash-preview",
    "gpt-5.6-luna",
    "gpt-5.6-terra",
    "claude-opus-5",
    "claude-sonnet-5",
]

def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _label(model: str) -> str:
    """The one name this route is called, everywhere in the paper."""
    return ROUTES.ROUTE_LABEL.get(ROUTES.route_id(model), LABELS.get(model, model))


def _colour(model: str) -> str:
    return ROUTES.ROUTE_C[ROUTES.route_id(model)]


def _marker(model: str) -> str:
    return ROUTES.ROUTE_M[ROUTES.route_id(model)]


# A tile label that does not fit inside its tile is the defect that shipped: a
# 100% printed as "00%" because the leading digit fell off the panel edge, and a
# row of gap values printed as "0.30.30.40.4" because neighbouring labels met.
# Neither is visible to the geometry gate or to the printed-type gate, so the
# check lives here, and it fails the build rather than warning.
TILE_GUTTER_PT = 3.0
# Where white and black type give a tile the same WCAG contrast ratio:
# (L + 0.05)^2 = 1.05 * 0.05, so a label placed by this rule clears 4.58:1
# whichever side of it the tile falls.
LUMINANCE_CROSSOVER = (1.05 * 0.05) ** 0.5 - 0.05


def _relative_luminance(red: float, green: float, blue: float) -> float:
    """WCAG relative luminance, which needs the sRGB channels linearised first.

    Weighting the raw 0-1 channels instead, as this module used to, reports a
    dark violet tile at 0.21 when its real luminance is 0.04, which is how a
    label came to be set in the colour that gives it 1.9:1.
    """
    def channel(value: float) -> float:
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return (0.2126 * channel(red) + 0.7152 * channel(green) + 0.0722 * channel(blue))


def _footer(fig: plt.Figure, text: str, *, y: float = 0.028, size: float = 8.0):
    """A bottom note wrapped to the figure's own width.

    One long ``fig.text`` does not wrap, and under ``bbox_inches='tight'`` it
    pushes the saved bounding box out past the figure, so the file on disk ends
    up wider than the placement width and LaTeX quietly rescales every label.
    """
    import textwrap

    characters = int(fig.get_size_inches()[0] * 0.97 * 72.0 / (size * 0.50))
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), characters))
    return fig.text(0.5, y, wrapped, ha="center", va="bottom", fontsize=size,
                    color=MUTED, linespacing=1.35)


def _assert_tiles_fit(ax: plt.Axes, texts: list[plt.Text], columns: int,
                      *, name: str) -> None:
    figure = ax.figure
    figure.canvas.draw()
    tile_px = ax.get_window_extent().width / columns
    widest = max(text.get_window_extent().width for text in texts)
    gutter_px = TILE_GUTTER_PT * figure.dpi / 72.0
    if widest + gutter_px > tile_px:
        offender = max(texts, key=lambda t: t.get_window_extent().width).get_text()
        raise RuntimeError(
            f"{name}: the widest tile label {offender!r} needs "
            f"{widest + gutter_px:.1f} px inside a {tile_px:.1f} px tile, so it "
            f"would be clipped or run into its neighbour. Widen the panel or "
            f"drop a column; do not shrink the type."
        )


def _percent_heatmap(ax: plt.Axes, values: np.ndarray, row_labels: list[str],
                     col_labels: list[str], *, title: str, cmap: str,
                     vmin: float, vmax: float, fmt: str | None = None,
                     norm=None, name: str = "heatmap",
                     row_label_colours: list[str] | None = None,
                     rotation: float = 38):
    image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=None if norm else vmin,
                      vmax=None if norm else vmax, norm=norm)
    ax.set_xticks(np.arange(len(col_labels)), col_labels, rotation=rotation,
                  ha="right" if rotation else "center")
    ax.set_yticks(np.arange(len(row_labels)), row_labels)
    panel_label(ax, title[0], title[1:])
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(values.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(values.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=WHITE, linewidth=1.8)
    ax.tick_params(which="minor", bottom=False, left=False)
    if row_label_colours:
        for tick, colour in zip(ax.get_yticklabels(), row_label_colours):
            tick.set_color(colour)
    if fmt:
        size = 8.0
        texts = []
        for i in range(values.shape[0]):
            for j in range(values.shape[1]):
                value = values[i, j]
                text = fmt.format(value)
                red, green, blue, _ = image.cmap(image.norm(value))
                luminance = _relative_luminance(red, green, blue)
                colour = WHITE if luminance < LUMINANCE_CROSSOVER else INK
                texts.append(ax.text(j, i, text, ha="center", va="center",
                                     fontsize=size, color=colour, weight="bold"))
        _assert_tiles_fit(ax, texts, values.shape[1], name=name)
    return image


def _bootstrap(values: np.ndarray, seed: int) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(4000, len(values)), replace=True).mean(axis=1)
    return float(values.mean()), float(np.quantile(draws, 0.025)), float(np.quantile(draws, 0.975))


def build_own_risk_dependence() -> list[Path]:
    """Measured human disposition beside an assigned model label.

    Two things were wrong beyond the roster rule.  The panels exist to be
    compared and shared a y-axis, but panel A was indexed 0-5 and panel B 1-6,
    so the same six ordinal levels sat one step out of register side by side;
    both now run on a common 1-6 index and each panel names its own coding.
    And panel A carried intervals and per-bin counts while panel B carried
    neither.  The checked-in persona artifact holds a cell mean and a player
    count and no player-level rates, so a participant-bootstrap interval is not
    derivable for panel B; the counts are therefore shown and the absence of
    intervals is stated rather than left to be inferred from their absence.
    """
    human = pd.read_csv(HUMAN_CSV)
    per_player = human.groupby(["participant_id", "risk_gamble_choice"], as_index=False)["decision"].mean()
    human_rows = []
    for choice in range(6):
        values = per_player.loc[per_player["risk_gamble_choice"].eq(choice), "decision"].to_numpy(float)
        mean, low, high = _bootstrap(values, 20260908 + choice)
        human_rows.append((choice, len(values), mean, low, high))
    persona = pd.read_csv(PERSONA_TABLE)
    persona = persona[persona["role"].isin([f"R{i}" for i in range(1, 7)])]
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN - 0.10, 4.55),
                             gridspec_kw={"width_ratios": [1.15, 1.52], "wspace": 0.20},
                             sharey=True)
    fig.subplots_adjust(left=0.105, right=0.99, top=0.92, bottom=0.40)
    ax = axes[0]
    # Both panels index the same six ordered levels from 1, so the reader is not
    # asked to align a 0-5 axis against a 1-6 axis by eye.
    x = np.array([row[0] for row in human_rows]) + 1
    means = np.array([row[2] for row in human_rows])
    low = np.array([row[3] for row in human_rows])
    high = np.array([row[4] for row in human_rows])
    ax.errorbar(x, means, yerr=[means - low, high - means], fmt="o", color=INK,
                markerfacecolor=GREEN, markeredgecolor=INK, linewidth=1.2,
                capsize=2.5, label="Human")
    ax.plot(x, means, color=INK, linewidth=1.0, linestyle=(0, (2, 2)), alpha=0.7)
    ax.set(xticks=x,
           xticklabels=[f"{level}\n(n={n})" for level, (_, n, *_) in zip(x, human_rows)],
           xlabel="Elicited Eckel-Grossman gamble, levels 1 to 6",
           ylabel="Unsafe rate", ylim=(0, 1.08))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    style_axis(ax); panel_label(ax, "A", "Measured human disposition")

    ax = axes[1]
    handles = []
    counts = []
    for model in PERSONA_MODEL_ORDER:
        sub = persona[persona["model"].eq(model)].copy()
        if len(sub) != 6:
            raise RuntimeError(f"Expected six persona levels for {model}, found {len(sub)}")
        sub["level"] = sub["role"].str.replace("R", "", regex=False).astype(int)
        sub = sub.sort_values("level")
        counts.append(sub["n_players"].to_numpy(int))
        # Membership of the nine audited routes is a property of the route and
        # is asked of the route table.  Hard-coding it as two names left GPT-5
        # nano, which is equally outside the roster, drawn as if it were inside.
        dashed = not ROUTES.in_roster(model)
        line, = ax.plot(sub["level"], sub["mean_unsafe_rate"], color=_colour(model),
                        marker=_marker(model), markersize=4.2, linewidth=1.2,
                        linestyle="--" if dashed else "-", label=_label(model))
        handles.append(line)
    counts = np.vstack(counts)
    modal = int(np.bincount(counts.ravel()).argmax())
    thin = {_label(model): (int(row.min()), int(row.max()))
            for model, row in zip(PERSONA_MODEL_ORDER, counts) if row.min() != modal}
    ax.set(xticks=np.arange(1, 7),
           xticklabels=[f"{level}\n(R{level})" for level in range(1, 7)],
           xlabel="Assigned risk-aware persona, levels 1 to 6",
           ylim=(0, 1.08))
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.tick_params(labelleft=True)
    style_axis(ax); panel_label(ax, "B", "Prompted model response")
    outside = [_label(m) for m in PERSONA_MODEL_ORDER if not ROUTES.in_roster(m)]
    fig.legend(handles, [h.get_label() for h in handles], frameon=False, fontsize=8.0,
               ncol=4, loc="lower center", bbox_to_anchor=(0.5, 0.200))
    counts_note = "; ".join(f"{name} rests on {lo} to {hi}" for name, (lo, hi) in thin.items())
    _footer(fig,
            "Both panels index the same six ordered levels from 1, and both run from most risk-averse at 1 to least "
            "risk-averse at 6: panel A's source file codes those levels 0 to 5, panel B's level k is persona Rk. "
            "Panel A whiskers are participant-bootstrap 95% "
            "intervals; panel B has none, because the persona artifact records a cell mean and a player count, not "
            f"the player-level rates a bootstrap needs. Panel-B points are means over {modal} players "
            f"({counts_note}). Dashed lines mark the {len(outside)} routes outside the nine-route audited roster "
            f"({', '.join(outside)}), which carry no admission verdict.",
            y=0.005)
    return save_publication_figure(fig, CLUSTER / "09_human_vs_llm_own_risk_dependence",
                                   formats=("pdf", "png", "svg"))


def _has_skill(entry: dict) -> bool:
    """Did this forest learn anything, or is its row a share of noise?

    A row of the heatmap normalises to 100% whether or not the forest predicts
    anything, so every row is drawn at the same visual weight.  Two of them are
    not comparable with the rest: the checkpoint plays so close to one action
    that the classifier recovers the rare class no better than chance, and the
    supplement's own text already says so for GPT-5-nano.  The test is the
    out-of-fold balanced accuracy, which is the number that exposes it, backed
    by the raw accuracy against the majority-class baseline.
    """
    return not (float(entry["balanced_accuracy"]) <= 0.55
                and float(entry["test_accuracy"]) <= float(entry["majority_baseline_accuracy"]))


def build_shap_heatmap() -> list[Path]:
    results = json.loads(SHAP_JSON.read_text(encoding="utf-8"))
    # The JSON contains Claude and GPT-5.6 routes in addition to the seven main
    # baseline inputs; preserve its complete, explicitly fitted population roster.
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
    auc = np.array([float(results[model]["roc_auc"]) for model in models])
    auc_sd = np.array([float(results[model]["roc_auc_std"]) for model in models])
    skill = np.array([_has_skill(results[model]) for model in models])
    row_labels = [_label(model) for model in models]
    row_colours = [INK if ok else MUTED for ok in skill]

    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN - 0.12, 4.45),
                             gridspec_kw={"width_ratios": [2.55, 1.0], "wspace": 0.45},
                             sharey=True)
    fig.subplots_adjust(left=0.225, right=0.975, top=0.86, bottom=0.33)
    image = _percent_heatmap(axes[0], values, row_labels,
                             [labels[name] for name in feature_order],
                             title="A Predictive feature share", cmap="Blues", vmin=0,
                             vmax=float(values.max()), fmt="{:.0%}",
                             name="SHAP shares", row_label_colours=row_colours)
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.030, pad=0.02)
    cbar.set_label("Share of mean |SHAP|", fontsize=8.0)
    cbar.ax.tick_params(labelsize=8.0)
    # The tiles were labelled in per cent while the bar was labelled 0.0 to 0.5.
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))

    ax = axes[1]
    rows = np.arange(len(models))
    ax.axvline(0.5, color=MUTED, linewidth=0.8, linestyle=(0, (3, 2)), zorder=1)
    ax.errorbar(auc[skill], rows[skill], xerr=auc_sd[skill], fmt="o", color=INK,
                markerfacecolor=BLUE, markeredgecolor=INK, markersize=4.6,
                linewidth=1.0, capsize=2.0, linestyle="none", zorder=3)
    if (~skill).any():
        ax.errorbar(auc[~skill], rows[~skill], xerr=auc_sd[~skill], fmt="o",
                    color=MUTED, markerfacecolor=WHITE, markeredgecolor=MUTED,
                    markersize=4.6, linewidth=1.0, capsize=2.0, linestyle="none",
                    zorder=3)
        for row in rows[~skill]:
            ax.annotate("no skill", (auc[row], row), xytext=(6, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=8.0, color=MUTED)
    ax.set(xlim=(0.35, 1.12), xticks=[0.5, 0.75, 1.0],
           xticklabels=["0.50\nchance", "0.75", "1.00"],
           xlabel="Out-of-fold ROC AUC", ylim=(len(models) - 0.4, -0.6))
    style_axis(ax, grid_axis="x")
    ax.tick_params(left=False)
    panel_label(ax, "B", "Forest skill")

    _footer(fig,
            "Associational feature use by separately fitted forests; it is not a causal attribution. Panel A "
            "normalises every row to 100% whether or not the forest predicts anything, so panel B carries the skill "
            "each row rests on: whiskers are the cross-validated standard deviation, and an open marker flags a "
            "forest whose balanced accuracy is at chance and whose accuracy does not beat the majority class.",
            y=0.005)
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


def _panel_titles(axes_titles: list[tuple[plt.Axes, plt.Text]], *, name: str) -> None:
    """Refuse two subplot titles that touch.

    The defect this catches shipped in two figures at once: "Gemini 3.5 Flash
    Lite" and "Claude Opus 5" ran together into one string because a
    left-aligned title is free to overrun its own panel and nothing measures it.
    """
    figure = axes_titles[0][0].figure
    figure.canvas.draw()
    rows: dict[int, list[tuple[float, float, str]]] = {}
    for ax, text in axes_titles:
        box = text.get_window_extent()
        rows.setdefault(round(box.y0), []).append((box.x0, box.x1, text.get_text()))
    for boxes in rows.values():
        boxes.sort()
        for (_, left_x1, left), (right_x0, _, right) in zip(boxes, boxes[1:]):
            if right_x0 - left_x1 < 6.0:
                raise RuntimeError(
                    f"{name}: subplot titles {left!r} and {right!r} are "
                    f"{right_x0 - left_x1:.1f} px apart and read as one string. "
                    f"Shorten, wrap, or widen the panels."
                )


def _wrap_title(label: str, limit: int = 12) -> str:
    """One line if it fits, otherwise the most balanced two-line break.

    Greedy wrapping breaks "Claude Opus 5" after "Opus", which reads as a
    stranded digit.  Choosing the split that minimises the longer line gives
    "Claude / Opus 5" and "Claude / Sonnet 5", which break in the same place.
    """
    if len(label) <= limit:
        return label
    words = label.split()
    best = min(
        range(1, len(words)),
        # Balance first, and on a tie keep the version token up with its family
        # name rather than pushing it down onto the second line.
        key=lambda k: (max(len(" ".join(words[:k])), len(" ".join(words[k:]))), -k),
    )
    return " ".join(words[:best]) + "\n" + " ".join(words[best:])


def build_tsne_outcome(frame: pd.DataFrame) -> list[Path]:
    coordinates = _ensure_tsne_table(frame)
    x = coordinates["tsne_x"].to_numpy(float)
    y = coordinates["tsne_y"].to_numpy(float)
    rates = coordinates["unsafe_rate"].to_numpy(float)
    xpad = max(0.1, np.ptp(x) * 0.04); ypad = max(0.1, np.ptp(y) * 0.04)
    xlim = (x.min() - xpad, x.max() + xpad); ylim = (y.min() - ypad, y.max() + ypad)
    norm = matplotlib.colors.Normalize(0, 1)
    # The outcome is the only thing this figure encodes, and it was encoded in
    # hue alone, on RdYlGn.  That map is not monotonic in luminance: its
    # lightest colour sits at the MIDPOINT, so a player at 0.5 was the most
    # visually salient point on the page, and under deuteranopia the two ends
    # collapsed to olive and rust while the middle stayed pale.  Viridis, which
    # five other figures in this paper already use, rises monotonically in
    # luminance, so the encoding survives greyscale and colour-vision
    # deficiency; marker area rises with the rate as an explicit second channel.
    cmap = matplotlib.colormaps.get_cmap("viridis")

    def sizes(values: np.ndarray, low: float, high: float) -> np.ndarray:
        return low + (high - low) * np.clip(values, 0.0, 1.0)

    fig = plt.figure(figsize=(FULL_WIDTH_IN - 0.50, 4.65))
    gs = fig.add_gridspec(2, 5, width_ratios=[1.4, 1, 1, 1, 1], wspace=0.22,
                          hspace=0.52, left=0.03, right=0.985, top=0.88, bottom=0.08)
    placements = {
        "human": gs[:, 0],
        "gpt-5-nano": gs[0, 1], "gpt-5.4-nano": gs[0, 2],
        "google/gemini-3-flash-preview": gs[0, 3],
        "google/gemini-3.1-flash-lite-preview": gs[0, 4],
        "google/gemini-3.5-flash-lite": gs[1, 1],
        "claude-opus-5": gs[1, 2], "claude-sonnet-5": gs[1, 3],
    }
    titles = []
    for model, slot in placements.items():
        ax = fig.add_subplot(slot)
        focus = coordinates["population"].eq(model).to_numpy()
        if model == "human":
            ax.scatter(x, y, c=rates, cmap=cmap, norm=norm, s=sizes(rates, 8, 22),
                       edgecolor=WHITE, linewidth=0.2, rasterized=True)
        else:
            ax.scatter(x[~focus], y[~focus], s=6, color=GRID, alpha=0.34,
                       linewidths=0, rasterized=True)
            ax.scatter(x[focus], y[focus], c=rates[focus], cmap=cmap, norm=norm,
                       s=sizes(rates[focus], 11, 28), marker=_marker(model),
                       edgecolor=WHITE, linewidth=0.25, rasterized=True)
        ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.axis("off")
        title = ax.text(0.5 if model != "human" else 0.01, 1.02,
                        _wrap_title(_label(model)), transform=ax.transAxes,
                        fontsize=8.0, weight="bold", va="bottom",
                        ha="center" if model != "human" else "left",
                        linespacing=1.15)
        titles.append((ax, title))
    axes = fig.axes
    fig.text(0.03, 0.955, "A  Pooled outcome", fontsize=9.2, weight="bold", color=INK)
    fig.text(0.275, 0.955, "B  Population-specific panels", fontsize=9.2, weight="bold", color=INK)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.018, pad=0.01, aspect=28)
    cbar.set_label("Own Unsafe rate", fontsize=8.0)
    cbar.ax.tick_params(labelsize=8.0)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _footer(fig,
            "Colour and marker area both rise with the player's own Unsafe rate; marker shape is the route, "
            "carried over from the main paper. Panel A pools all 760 trajectories.",
            y=0.005)
    _panel_titles(titles, name="06 population panels")
    return save_publication_figure(fig, CLUSTER / "06_tsne_safe_unsafe",
                                   formats=("pdf", "png", "svg"))


SEAT_TOLERANCE = 1e-12


def build_round_profiles(frame: pd.DataFrame) -> list[Path]:
    """Round-by-round Unsafe rate and progress gap, one row per population.

    Panel A used to carry ten columns, Own R1-R5 beside Opp R1-R5.  Five of them
    were the same numbers drawn twice.  Every model population plays itself with
    counterbalanced seats, so pooling over both seats makes the opponent
    marginal the *same multiset* as the own marginal: the two halves agreed to
    the last bit on all seven model rows.  Ten columns of five numbers left each
    tile too narrow for its own label, which is why the three rows that reach
    100% printed as ``00%``.  So the model rows now carry one marginal, and the
    humans, who played dyads rather than self-play and therefore have two
    genuinely separate seat marginals, carry both and say so.
    """
    round_labels = [f"R{i}" for i in range(1, 6)]
    gap_labels = [f"R{i}" for i in range(2, 6)]
    models = list(BASELINE_INPUTS)

    human = frame[frame["population"].eq("human")]
    human_own = np.vstack(human["own"].to_numpy()).mean(axis=0)
    human_opp = np.vstack(human["opponent"].to_numpy()).mean(axis=0)

    action_values = [human_own, human_opp]
    row_keys = ["human", "human"]
    row_labels = [f"{_label('human')} (own seat)", f"{_label('human')} (partner seat)"]
    for population in models:
        sub = frame[frame["population"].eq(population)]
        own = np.vstack(sub["own"].to_numpy()).mean(axis=0)
        opponent = np.vstack(sub["opponent"].to_numpy()).mean(axis=0)
        seat_gap = float(np.abs(own - opponent).max())
        if seat_gap > SEAT_TOLERANCE:
            raise RuntimeError(
                f"{population}: the two seat marginals differ by {seat_gap:.6g}, so "
                f"they are no longer the same measurement and panel A may not "
                f"collapse them. Draw both seats for this population."
            )
        action_values.append(own)
        row_keys.append(population)
        row_labels.append(_label(population))
    action_values = np.asarray(action_values)

    gap_values = np.asarray([
        # Signed gaps cancel when both focal-player roles are pooled.  The
        # absolute gap is the informative shared feature: how far apart the two
        # players are entering the next round.
        np.abs(np.vstack(frame[frame["population"].eq(key)]["gap"].to_numpy())[:, 1:]).mean(axis=0)
        for key in row_keys
    ])
    # The old panel B ran its colour scale to 1.50 while the largest value in it
    # was 0.65, so nine tenths of the panel sat in the first third of the map and
    # the colour carried nothing.  The scale now comes from the data.
    gap_max = float(gap_values.max())
    human_seat_gap_pp = 100.0 * float(np.abs(human_own - human_opp).max())
    row_colours = [_colour(key) for key in row_keys]

    # The saved width is what LaTeX places, so the canvas is sized to land the
    # tight bounding box on the full text width rather than be rescaled there.
    fig, axes = plt.subplots(1, 2, figsize=(FULL_WIDTH_IN - 0.62, 3.65),
                             gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.40})
    fig.subplots_adjust(left=0.215, right=0.985, top=0.85, bottom=0.20)
    image = _percent_heatmap(axes[0], action_values, row_labels, round_labels,
                             title="A Unsafe rate by round", cmap="viridis", vmin=0,
                             vmax=1, fmt="{:.0%}", name="03 panel A",
                             row_label_colours=row_colours, rotation=0)
    cbar = fig.colorbar(image, ax=axes[0], fraction=0.035, pad=0.03)
    cbar.set_label("Unsafe probability", fontsize=8.0); cbar.ax.tick_params(labelsize=8.0)
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    image = _percent_heatmap(axes[1], gap_values, [""] * len(row_labels), gap_labels,
                             title="B Absolute progress gap", cmap="Blues", vmin=0,
                             vmax=gap_max, fmt="{:.2f}", name="03 panel B",
                             rotation=0)
    axes[1].tick_params(left=False)
    cbar = fig.colorbar(image, ax=axes[1], fraction=0.055, pad=0.03)
    cbar.set_label("Absolute step gap", fontsize=8.0); cbar.ax.tick_params(labelsize=8.0)
    _footer(fig,
            "Population means over complete first-five trajectories. Counterbalanced self-play makes a model's two "
            "seat marginals identical, so each model row carries one marginal; the humans played dyads, and their two "
            f"seats differ by at most {human_seat_gap_pp:.1f} pp. Gap is measured entering rounds 2 to 5.")
    return save_publication_figure(fig, CLUSTER / "03_radar_small_multiples_by_group_no_gap_r1",
                                   formats=("pdf", "png", "svg"))


def _delegate_egt_figure(module_name: str, stem: str) -> list[Path]:
    """Run one of the EGT builders and report what it wrote.

    This module used to keep its own copy of both EGT figures.  The copies were
    reached from ``build_publication_figures.py`` and so from every full
    publication build, which meant each build silently overwrote the repaired
    figures with the old drawings.  Worse than a style regression: the old
    invasion drawing read ``focal_strategy``/``opponent_strategy`` from schema
    v1, where the EGTtools row (the resident) had been written out as the focal
    strategy, so its arrows ran from the invader to the population it invades.
    The owning modules are now the only writers.

    ``argparse`` in one of those ``main`` functions reads the caller's argv, and
    both re-apply the ``figstyle`` rcParams, so the call is fenced on both.
    """
    import importlib

    module = importlib.import_module(module_name)
    saved_argv = sys.argv
    try:
        sys.argv = [saved_argv[0]]
        module.main()
    finally:
        sys.argv = saved_argv
        configure_publication_style()
    written = [PAPER / f"{stem}.{suffix}" for suffix in ("pdf", "png", "svg")]
    missing = [path for path in written if not path.is_file()]
    if missing:
        raise RuntimeError(f"{module_name} did not write {[p.name for p in missing]}")
    return written


def build_supplementary_figures(frame: pd.DataFrame | None = None) -> dict[str, list[Path]]:
    configure_publication_style()
    if frame is None:
        frame = load_trajectories()
    outputs = {
        "supplement_figure_3_own_risk": build_own_risk_dependence(),
        "supplement_figure_4_shap": build_shap_heatmap(),
        "supplement_figure_5_tsne_outcome": build_tsne_outcome(frame),
        "supplement_figure_6_round_profiles": build_round_profiles(frame),
    }
    # Last, because both modules re-apply the figstyle rcParams while they draw.
    outputs["supplement_figure_1_egt_insights"] = _delegate_egt_figure(
        "scripts.build_egt_frontier_insights", "egt_frontier_insights")
    outputs["supplement_figure_2_egt_invasion"] = _delegate_egt_figure(
        "scripts.build_egt_frontier_invasion", "egt_frontier_invasion")
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
            "supplement_figure_1_egt_insights": "drawn by scripts/build_egt_frontier_insights.py, which owns it; this module only runs it",
            "supplement_figure_2_egt_invasion": "drawn by scripts/build_egt_frontier_invasion.py, which owns it; this module only runs it",
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
