#!/usr/bin/env python3
"""Load clustering data and preserve the historical generator entry point.

The publication figure set now lives in ``build_publication_figures.py``. This
module remains the maintained source loader for that builder, while its direct
CLI is delegated so an older command cannot recreate retired legacy artwork.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import OrderedDict, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

PAPER_FIGURES = ROOT / "figures" / "paper"
CLUSTER_FIGURES = PAPER_FIGURES / "llm_human_clustering"
DATA_OUT = ROOT / "results" / "cross_model_pilot_synthesis" / "data"

PROTECTED_MANUAL_FIGURE_STEMS = frozenset(
    {
        "figures/paper/AIRaceOverview",
        "figures/paper/ExpOverview",
        "figures/paper/llm_human_clustering/05b_unsafe_rate_by_group",
        "figures/paper/11_relative_position_grouped_bars",
    }
)
HUMAN_CSV = ROOT / "references" / "source_study_dataset" / "airace_deidentified_long.csv"
# The manuscript Figure 4 includes both the primary and T=.7 sensitivity
# frontier summaries; the complete source table set is the open-source EGT
# reconstruction output.  The frontier-only v2 directory intentionally omits
# the sensitivity summary and is therefore not the manuscript source.
EGT_OUTPUT = ROOT / "results" / "open_source" / "egt_reproduction"

WHITE = "#FFFFFF"
INK = "#17212B"
MUTED = "#4B5865"
GRID = "#D9DEE8"
OTHER = "#C9CBC7"

# The order is the order used in the manuscript paragraph and Figure 5.
BASELINE_INPUTS = OrderedDict(
    [
        ("gpt-5-nano", "results/frontier/openai/baseline/gpt-5-nano"),
        ("gpt-5.4-nano", "results/frontier/openai/baseline/gpt-5.4-nano"),
        (
            "google/gemini-3-flash-preview",
            "results/frontier/baseline/google-gemini-3-flash-preview",
        ),
        (
            "google/gemini-3.1-flash-lite-preview",
            "results/frontier/baseline/google-gemini-3.1-flash-lite-preview",
        ),
        (
            "google/gemini-3.5-flash-lite",
            "results/frontier/baseline/google-gemini-3.5-flash-lite",
        ),
        (
            "claude-opus-5",
            "results/frontier/bedrock/baseline/us.anthropic.claude-opus-5",
        ),
        (
            "claude-sonnet-5",
            "results/frontier/bedrock/baseline/us.anthropic.claude-sonnet-5",
        ),
    ]
)

LABELS = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
    "human": "Human",
}

# Fixed meanings: model identity is carried by the same marker and colour in
# the two panels; Human is the external reference population.
FIGURE_COLORS = {
    "gpt-5-nano": "#4A90D9",
    "gpt-5.4-nano": "#EB7B4D",
    "google/gemini-3-flash-preview": "#42B692",
    "google/gemini-3.1-flash-lite-preview": "#F0A500",
    "google/gemini-3.5-flash-lite": "#E58BB2",
    "claude-opus-5": "#2A9B2A",
    "claude-sonnet-5": "#6354B5",
    "human": "#E66364",
}
FIGURE_MARKERS = {
    "gpt-5-nano": "o",
    "gpt-5.4-nano": "s",
    "google/gemini-3-flash-preview": "^",
    "google/gemini-3.1-flash-lite-preview": "D",
    "google/gemini-3.5-flash-lite": "v",
    "claude-opus-5": "P",
    "claude-sonnet-5": "X",
    "human": "*",
}
FIGURE8_LABELS = {
    "human": "Human",
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1\nFlash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5\nFlash Lite",
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
}


def configure_plot() -> None:
    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.edgecolor": GRID,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def save_figure(fig: plt.Figure, stem: Path) -> None:
    relative = stem.resolve().relative_to(ROOT).as_posix()
    if relative in PROTECTED_MANUAL_FIGURE_STEMS:
        raise RuntimeError(
            f"Refusing to overwrite author-supplied artwork: {relative}. "
            "Use a new generated stem for an analysis variant."
        )
    stem.parent.mkdir(parents=True, exist_ok=True)
    fig.set_facecolor(WHITE)
    for axis in fig.axes:
        axis.set_facecolor(WHITE)
    common_kwargs = {
        "facecolor": WHITE,
        "edgecolor": WHITE,
        "transparent": False,
        "bbox_inches": "tight",
        "pad_inches": 0.04,
    }
    pdf_kwargs = {
        **common_kwargs,
        "metadata": {
            "Title": stem.stem,
            "Creator": "AI_Race_Experiment figure generator",
            "Subject": "Human versus LLM AI-race trajectories",
        },
    }
    # SVG's metadata schema is narrower than PDF's; keep the searchable PDF
    # metadata while avoiding backend-specific keys in the editable export.
    fig.savefig(stem.with_suffix(".png"), dpi=600, **common_kwargs)
    fig.savefig(stem.with_suffix(".pdf"), **pdf_kwargs)
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, **common_kwargs)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)


def load_human_first_five() -> tuple[pd.DataFrame, dict[str, list[float]]]:
    df = pd.read_csv(HUMAN_CSV)
    needed = set(range(1, 6))
    rows: list[dict[str, object]] = []
    by_id: dict[str, list[float]] = {}
    for player_id, group in df.groupby("participant_id", sort=True):
        first = group[group["round_number"].isin(needed)].sort_values("round_number")
        if set(first["round_number"].astype(int)) != needed or len(first) != 5:
            continue
        own = first["decision"].astype(float).to_numpy()
        opponent = first["decision_opponent"].astype(float).to_numpy()
        gap = first["delta_steps_lag"].astype(float).to_numpy()
        rows.append(
            {
                "population": "human",
                "player_id": str(player_id),
                "own": own,
                "opponent": opponent,
                "gap": gap,
                "unsafe_rate": float(own.mean()),
            }
        )
        by_id[str(player_id)] = own.tolist()
    out = pd.DataFrame(rows)
    if len(out) != 340:
        raise RuntimeError(f"Expected 340 complete human first-five trajectories, found {len(out)}")
    return out, by_id


def load_llm_first_five(model: str, relative_root: str) -> pd.DataFrame:
    path = ROOT / relative_root / "turns.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"Missing admitted baseline source: {path}")
    records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    action_lookup = {
        (str(row["game_id"]), int(row["round"]), str(row["player"])): int(row["unsafe"])
        for row in records
    }
    grouped: dict[tuple[str, str], list[dict[str, object]]] = defaultdict(list)
    for row in records:
        grouped[(str(row["game_id"]), str(row["player"]))].append(row)

    rows: list[dict[str, object]] = []
    for (game_id, player), group in sorted(grouped.items()):
        first = sorted((row for row in group if int(row["round"]) <= 5), key=lambda row: int(row["round"]))
        if [int(row["round"]) for row in first] != [1, 2, 3, 4, 5]:
            continue
        own = np.asarray([int(row["unsafe"]) for row in first], dtype=float)
        opponent = np.asarray(
            [
                action_lookup[(game_id, int(row["round"]), str(row["opponent"]))]
                for row in first
            ],
            dtype=float,
        )
        gap = np.asarray([float(row["progress_gap_before"]) for row in first], dtype=float)
        rows.append(
            {
                "population": model,
                "player_id": f"{game_id}::{player}",
                "own": own,
                "opponent": opponent,
                "gap": gap,
                "unsafe_rate": float(own.mean()),
            }
        )
    out = pd.DataFrame(rows)
    if len(out) != 60:
        raise RuntimeError(f"Expected 60 complete baseline trajectories for {model}, found {len(out)}")
    return out


def load_trajectories() -> pd.DataFrame:
    human, _ = load_human_first_five()
    frames = [human]
    for model, relative_root in BASELINE_INPUTS.items():
        frames.append(load_llm_first_five(model, relative_root))
    out = pd.concat(frames, ignore_index=True)
    if len(out) != 760:
        raise RuntimeError(f"Expected pooled N=760 trajectories, found {len(out)}")
    counts = out["population"].value_counts().to_dict()
    expected = {"human": 340, **{model: 60 for model in BASELINE_INPUTS}}
    if counts != expected:
        raise RuntimeError(f"Unexpected trajectory counts: {counts}; expected {expected}")
    return out


def flatten_features(frame: pd.DataFrame) -> np.ndarray:
    vectors = []
    for row in frame.itertuples(index=False):
        vectors.append(np.concatenate([row.own, row.opponent, row.gap]).astype(float))
    array = np.vstack(vectors)
    if array.shape != (len(frame), 15):
        raise RuntimeError(f"Expected 15 raw features per trajectory, got {array.shape}")
    return array


def redraw_figure_4(frame: pd.DataFrame) -> dict[str, float]:
    order = list(BASELINE_INPUTS) + ["human"]
    rates = {population: float(frame.loc[frame.population == population, "unsafe_rate"].mean()) for population in order}
    expected = {
        "gpt-5-nano": 0.17,
        "gpt-5.4-nano": 0.5366666667,
        "google/gemini-3-flash-preview": 0.73,
        "google/gemini-3.1-flash-lite-preview": 0.8266666667,
        "google/gemini-3.5-flash-lite": 0.7433333333,
        "claude-opus-5": 1 / 3,
        "claude-sonnet-5": 0.2166666667,
        "human": 0.56,
    }
    for population in order:
        if not np.isclose(rates[population], expected[population], atol=0.001):
            raise RuntimeError(
                f"Figure 5 source changed for {population}: {rates[population]:.6f}; "
                f"paper expects {expected[population]:.6f}"
            )

    fig, axis = plt.subplots(figsize=(10.5, 5.9), facecolor=WHITE)
    values = [rates[population] for population in order]
    axis.bar(
        np.arange(len(order)),
        values,
        color=[FIGURE_COLORS[population] for population in order],
        edgecolor=WHITE,
        linewidth=1.1,
        hatch="//",
        width=0.62,
        zorder=3,
    )
    axis.set_facecolor(WHITE)
    axis.set_ylim(0, 1)
    axis.set_ylabel("Mean unsafe rate (rounds 1–5)", fontsize=12)
    axis.set_xticks(np.arange(len(order)), [LABELS[population] for population in order])
    axis.tick_params(axis="x", labelrotation=28, labelsize=10, pad=8)
    axis.tick_params(axis="y", labelsize=10)
    axis.grid(axis="y", color=GRID, linewidth=0.8, zorder=0)
    axis.set_axisbelow(True)
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.spines["left"].set_color(GRID)
    axis.spines["bottom"].set_color(GRID)
    axis.set_title("Player-level mean unsafe rate, rounds 1–5", fontsize=16, pad=12)
    fig.tight_layout()
    save_figure(fig, CLUSTER_FIGURES / "05b_unsafe_rate_by_group")
    return rates


def redraw_tsne_auxiliary(frame: pd.DataFrame) -> dict[str, object]:
    features = StandardScaler().fit_transform(flatten_features(frame))
    tsne_config = {
        "n_components": 2,
        "perplexity": 30,
        "random_state": 20260908,
        "init": "pca",
        "learning_rate": "auto",
        "max_iter": 1500,
    }
    embedding = TSNE(**tsne_config).fit_transform(features)
    x_pad = max(0.1, float(np.ptp(embedding[:, 0])) * 0.04)
    y_pad = max(0.1, float(np.ptp(embedding[:, 1])) * 0.04)
    xlim = (float(embedding[:, 0].min() - x_pad), float(embedding[:, 0].max() + x_pad))
    ylim = (float(embedding[:, 1].min() - y_pad), float(embedding[:, 1].max() + y_pad))

    positions = {
        "human": (0.03, 0.08, 0.29, 0.84),
        "gpt-5-nano": (0.35, 0.56, 0.14, 0.34),
        "gpt-5.4-nano": (0.50, 0.56, 0.14, 0.34),
        "google/gemini-3-flash-preview": (0.65, 0.56, 0.14, 0.34),
        "google/gemini-3.1-flash-lite-preview": (0.80, 0.56, 0.17, 0.34),
        "google/gemini-3.5-flash-lite": (0.35, 0.08, 0.14, 0.34),
        "claude-opus-5": (0.50, 0.08, 0.14, 0.34),
        "claude-sonnet-5": (0.65, 0.08, 0.14, 0.34),
    }
    fig = plt.figure(figsize=(9.4, 3.46), facecolor=WHITE)
    for population, (left, bottom, width, height) in positions.items():
        axis = fig.add_axes([left, bottom, width, height], facecolor=WHITE)
        focus = frame["population"].to_numpy() == population
        axis.scatter(
            embedding[~focus, 0],
            embedding[~focus, 1],
            s=17 if population == "human" else 15,
            c=OTHER,
            alpha=0.48,
            linewidths=0,
            marker="o",
            rasterized=True,
            zorder=1,
        )
        axis.scatter(
            embedding[focus, 0],
            embedding[focus, 1],
            s=38 if population == "human" else 39,
            c=FIGURE_COLORS[population],
            alpha=0.92,
            linewidths=0.4,
            edgecolors=WHITE,
            marker=FIGURE_MARKERS[population],
            rasterized=True,
            zorder=2,
        )
        axis.set_xlim(*xlim)
        axis.set_ylim(*ylim)
        axis.axis("off")
        axis.set_title(
            FIGURE8_LABELS[population],
            fontsize=15 if population == "human" else 11.5,
            fontweight="normal",
            pad=3,
        )

    save_figure(fig, CLUSTER_FIGURES / "01_tsne_hero_human_left")
    return {"counts": frame["population"].value_counts().sort_index().to_dict(), "tsne": tsne_config}


def redraw_egt_figure_auxiliary() -> None:
    required = [
        EGT_OUTPUT / "egt_stationary_summary.csv",
        EGT_OUTPUT / "llm_strategy_summary_primary_t0.csv",
        EGT_OUTPUT / "llm_strategy_summary_sensitivity_t07.csv",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing EGT source tables: " + ", ".join(missing))
    from scripts import reproduce_egt_model as egt

    egt._configure_matplotlib()
    chain_summary = egt._read_chain_summary(required[0])
    llm_summary = egt._read_numeric_csv(required[1])
    sensitivity_summary = egt._read_numeric_csv(required[2])
    egt.plot_theory_llm_comparison(chain_summary, llm_summary, sensitivity_summary, PAPER_FIGURES)


def normalize_legacy_figure(source: Path, stem: Path) -> dict[str, object]:
    """Make an opaque-white copy without changing legacy plot content.

    The old artwork uses a uniform off-white canvas. Only pixels within a
    small RGB distance of the corner colour are changed, preserving coloured
    marks, labels, and gridlines. PDF/SVG exports remain honest raster wrappers
    because the only recoverable source in this repository is the PNG itself.
    """
    relative = stem.resolve().relative_to(ROOT).as_posix()
    if relative in PROTECTED_MANUAL_FIGURE_STEMS:
        raise RuntimeError(
            f"Refusing to overwrite author-supplied artwork: {relative}. "
            "Use a new generated stem for an analysis variant."
        )
    source_hash = sha256(source)
    image = Image.open(source).convert("RGBA")
    array = np.asarray(image).copy()
    background = array[0, 0, :3].astype(np.int16)
    distance = np.max(np.abs(array[:, :, :3].astype(np.int16) - background), axis=2)
    mask = distance <= 2
    array[mask, :3] = 255
    array[:, :, 3] = 255
    normalized = Image.fromarray(array, mode="RGBA")
    stem.parent.mkdir(parents=True, exist_ok=True)
    normalized.save(stem.with_suffix(".png"), dpi=(600, 600))

    # Keep the export boundary white and preserve the legacy pixel dimensions.
    fig = plt.figure(
        figsize=(image.width / 150, image.height / 150),
        dpi=150,
        facecolor=WHITE,
    )
    axis = fig.add_axes([0, 0, 1, 1], facecolor=WHITE)
    axis.imshow(normalized)
    axis.axis("off")
    fig.savefig(
        stem.with_suffix(".pdf"),
        facecolor=WHITE,
        edgecolor=WHITE,
        transparent=False,
        metadata={"Title": stem.stem, "Creator": "AI_Race_Experiment figure generator"},
    )
    svg_path = stem.with_suffix(".svg")
    fig.savefig(svg_path, facecolor=WHITE, edgecolor=WHITE, transparent=False)
    svg_path.write_text(
        "\n".join(line.rstrip() for line in svg_path.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(fig)
    return {
        "source_asset": str(source.relative_to(ROOT)),
        "source_sha256_before_normalization": source_hash,
        "source_corner_rgb": [int(value) for value in background],
        "normalized_pixels": int(mask.sum()),
        "output_shape": [image.height, image.width],
        "operation": "canvas-only RGB normalization; legacy plotted content retained",
    }


def redraw_figure_5() -> dict[str, object]:
    raise RuntimeError(
        "Legacy Figure 5 normalization is retired; run "
        "scripts/build_publication_figures.py for the source-backed archetype figure."
    )


def redraw_figure_8() -> dict[str, object]:
    source = PAPER_FIGURES / "11_relative_position_grouped_bars.png"
    if not source.is_file():
        raise FileNotFoundError(f"Missing legacy Figure 8 source asset: {source}")
    return normalize_legacy_figure(source, PAPER_FIGURES / "11_relative_position_grouped_bars")


def main() -> None:
    from scripts.build_publication_figures import main as publication_main

    publication_main()


if __name__ == "__main__":
    main()
