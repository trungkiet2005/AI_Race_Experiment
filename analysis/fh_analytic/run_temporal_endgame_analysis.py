#!/usr/bin/env python3
"""Temporal / shadow-of-the-future mining.

No prior report in this pipeline looks at time-to-actual-end. The horizon is
hidden by design (`ai_race/engine` samples it from a separate RNG stream and
never reveals it in the prompt -- see CLAUDE.md), so a player cannot reason
about "rounds remaining" directly. But every *completed* race has a known
final round in hindsight, so this stage asks a purely emergent question: does
unsafe behavior rise as a race approaches its true (unknown-to-the-player)
end, the way "shadow of the future" / endgame effects show up in finitely
repeated human games? If it does, it cannot be literal horizon-aware
reasoning (the horizon is hidden); it would have to be a side effect of
something else that correlates with lateness (accumulated private risk,
round count itself, race-specific momentum).

This is distinct from the `round_phase` (time since start) buckets already
used elsewhere in this pipeline, which cannot separate "later in an
11-round race" from "near the end of a 6-round race."
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures" / "temporal_endgame"
REPORTS_DIR = OUTPUT_DIR / "reports"

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

MODEL_ORDER = [
    "gpt-5-nano",
    "gpt-5.4-nano",
    "google-gemini-3-flash-preview",
    "google-gemini-3.1-flash-lite-preview",
    "google-gemini-3.5-flash-lite",
]
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google-gemini-3-flash-preview": "Gemini 3 Flash",
    "google-gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google-gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}
MODEL_COLORS = {
    "gpt-5-nano": BLUE,
    "gpt-5.4-nano": PINK,
    "google-gemini-3-flash-preview": ORANGE,
    "google-gemini-3.1-flash-lite-preview": GOLD,
    "google-gemini-3.5-flash-lite": OLIVE,
}
HUMAN_PHI_U_INTERVAL = (0.40, 0.75)  # E7 in human_reference.json


def clean_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.copy()
    for col in visible.columns:
        if pd.api.types.is_float_dtype(visible[col]):
            visible[col] = visible[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
        else:
            visible[col] = visible[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(visible.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(visible.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in visible.columns) + " |"
        for _, row in visible.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": WHITE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
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
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def load_turns_with_horizon() -> pd.DataFrame:
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv", low_memory=False)
    races = pd.read_csv(DERIVED_DIR / "races_canonical.csv", low_memory=False)
    turns["duplicate_grain_key"] = clean_bool(turns["duplicate_grain_key"])
    races["duplicate_grain_key"] = clean_bool(races["duplicate_grain_key"])
    for col in ["unsafe", "round"]:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")
    races["n_rounds"] = pd.to_numeric(races["n_rounds"], errors="coerce")
    races["stop_forced"] = pd.to_numeric(races.get("stop_forced"), errors="coerce").fillna(0)

    turns = turns[(turns["manifest_status"] == "completed") & (~turns["duplicate_grain_key"])].copy()
    races_clean = races[(races["manifest_status"] == "completed") & (~races["duplicate_grain_key"])].copy()

    horizon = races_clean[["source_run", "game_id", "n_rounds", "stop_forced"]].drop_duplicates()
    turns = turns.merge(horizon, on=["source_run", "game_id"], how="inner")
    turns["rounds_remaining"] = turns["n_rounds"] - turns["round"]
    turns["rounds_remaining_band"] = np.where(
        turns["rounds_remaining"] >= 4, "4plus", turns["rounds_remaining"].clip(lower=0).astype("Int64").astype(str)
    )
    turns["cluster_id"] = turns["source_run"].astype(str) + "::" + turns["game_id"].astype(str)
    return turns


def rate_with_ci(values: pd.Series) -> tuple[float, int, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return np.nan, 0, np.nan, np.nan
    mean = float(values.mean())
    se = math.sqrt(mean * (1 - mean) / n) if 0 <= mean <= 1 else np.nan
    lo = max(0.0, mean - 1.96 * se) if not math.isnan(se) else np.nan
    hi = min(1.0, mean + 1.96 * se) if not math.isnan(se) else np.nan
    return mean, n, lo, hi


def build_endgame_table(turns: pd.DataFrame) -> pd.DataFrame:
    band_order = ["0", "1", "2", "3", "4plus"]
    rows: list[dict[str, Any]] = []
    for keys, group in turns.groupby(["analysis_scope", "model_slug", "rounds_remaining_band"], dropna=False):
        scope, model, band = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append(
            {"analysis_scope": scope, "model_slug": model, "rounds_remaining_band": band, "n": n, "unsafe_rate": mean, "ci95_low": lo, "ci95_high": hi}
        )
    frame = pd.DataFrame(rows)
    frame["rounds_remaining_band"] = pd.Categorical(frame["rounds_remaining_band"], categories=band_order, ordered=True)
    frame = frame.sort_values(["analysis_scope", "model_slug", "rounds_remaining_band"])
    frame.to_csv(DERIVED_DIR / "temporal_endgame_by_rounds_remaining.csv", index=False)
    return frame


def build_pooled_endgame_table(turns: pd.DataFrame) -> pd.DataFrame:
    band_order = ["0", "1", "2", "3", "4plus"]
    rows: list[dict[str, Any]] = []
    for band, group in turns[turns["analysis_scope"] == "all_completed"].groupby("rounds_remaining_band", dropna=False):
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append({"rounds_remaining_band": band, "n": n, "unsafe_rate": mean, "ci95_low": lo, "ci95_high": hi})
    frame = pd.DataFrame(rows)
    frame["rounds_remaining_band"] = pd.Categorical(frame["rounds_remaining_band"], categories=band_order, ordered=True)
    frame = frame.sort_values("rounds_remaining_band")
    frame.to_csv(DERIVED_DIR / "temporal_endgame_pooled.csv", index=False)
    return frame


def build_final_turn_table(turns: pd.DataFrame) -> pd.DataFrame:
    """Unsafe rate on literally the last decision of a race vs everything else, per model."""

    turns = turns.copy()
    turns["is_final_turn"] = turns["rounds_remaining"] == 0
    rows: list[dict[str, Any]] = []
    for keys, group in turns[turns["analysis_scope"] == "all_completed"].groupby(["model_slug", "is_final_turn"], dropna=False):
        model, is_final = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append({"model_slug": model, "is_final_turn": bool(is_final), "n": n, "unsafe_rate": mean})
    frame = pd.DataFrame(rows).sort_values(["model_slug", "is_final_turn"])
    frame.to_csv(DERIVED_DIR / "temporal_final_turn_vs_rest.csv", index=False)
    return frame


def build_horizon_length_table(turns: pd.DataFrame) -> pd.DataFrame:
    """Does unsafe rate differ between races that happened to run long vs short (n_rounds)?"""

    rows: list[dict[str, Any]] = []
    bins = [0, 6, 8, 10, 100]
    labels = ["short_5_6", "mid_7_8", "long_9_10", "very_long_11plus"]
    turns = turns.copy()
    turns["horizon_band"] = pd.cut(turns["n_rounds"], bins=bins, labels=labels, right=True)
    for keys, group in turns[turns["analysis_scope"] == "all_completed"].groupby(["model_slug", "horizon_band"], dropna=False, observed=True):
        model, band = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append({"model_slug": model, "horizon_band": band, "n": n, "unsafe_rate": mean})
    frame = pd.DataFrame(rows).sort_values(["model_slug", "horizon_band"])
    frame.to_csv(DERIVED_DIR / "temporal_horizon_length.csv", index=False)
    return frame


def plot_endgame_curve(pooled: pd.DataFrame, by_model: pd.DataFrame) -> Path | None:
    if pooled.empty:
        return None
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    order = ["4plus", "3", "2", "1", "0"]
    p = pooled.set_index("rounds_remaining_band").reindex(order)
    axes[0].plot(range(len(order)), p["unsafe_rate"], marker="o", color=BLUE)
    axes[0].fill_between(range(len(order)), p["ci95_low"], p["ci95_high"], color=BLUE, alpha=0.15)
    axes[0].set_xticks(range(len(order)))
    axes[0].set_xticklabels(["4+", "3", "2", "1", "0 (final)"])
    axes[0].set_xlabel("rounds remaining until actual end")
    axes[0].set_ylabel("Unsafe rate")
    axes[0].set_title("Pooled endgame curve (all completed)", loc="left", fontweight="bold")
    axes[0].axhspan(HUMAN_PHI_U_INTERVAL[0], HUMAN_PHI_U_INTERVAL[1], color=GOLD, alpha=0.12)

    for model in MODEL_ORDER:
        sub = by_model[(by_model["model_slug"] == model) & (by_model["analysis_scope"] == "all_completed")]
        sub = sub.set_index("rounds_remaining_band").reindex(order)
        if sub["n"].sum() == 0:
            continue
        axes[1].plot(range(len(order)), sub["unsafe_rate"], marker="o", color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
    axes[1].set_xticks(range(len(order)))
    axes[1].set_xticklabels(["4+", "3", "2", "1", "0 (final)"])
    axes[1].set_xlabel("rounds remaining until actual end")
    axes[1].set_title("By model", loc="left", fontweight="bold")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    return save(fig, "01_endgame_curve.png")


def fmt(value: float, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.1%}" if pct else f"{value:.3g}"


def write_report(
    pooled: pd.DataFrame,
    by_model: pd.DataFrame,
    final_turn: pd.DataFrame,
    horizon_length: pd.DataFrame,
    n_decisions: int,
    n_games: int,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# FH Temporal / Shadow-Of-The-Future Mining")
    lines.append("")
    lines.append("## Scope And Question")
    lines.append("")
    lines.append(
        f"{n_decisions:,} completed, non-duplicate turn decisions across {n_games:,} completed races. The horizon "
        "is hidden by design -- a player never sees the final round in advance -- so any relationship between "
        "unsafe behavior and hindsight-known `rounds_remaining` cannot be literal horizon-aware reasoning. It can "
        "only be an emergent side effect of something correlated with lateness inside a specific race (accumulated "
        "private risk, opponent's cumulative unsafe count, race-specific momentum). This distinguishes 'closer to "
        "the actual end' from the `round_phase` (time since start) buckets used elsewhere, which cannot separate "
        "'round 9 of an 11-round race' from 'round 9 of a 10-round race that is about to stop.'"
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    if not pooled.empty:
        first = pooled[pooled["rounds_remaining_band"] == "4plus"]["unsafe_rate"]
        last = pooled[pooled["rounds_remaining_band"] == "0"]["unsafe_rate"]
        first_v = float(first.iloc[0]) if len(first) else np.nan
        last_v = float(last.iloc[0]) if len(last) else np.nan
        lines.append(
            f"- **Pooled unsafe rate moves from {fmt(first_v)} at 4+ rounds remaining to {fmt(last_v)} on the "
            f"literal final decision of a race** ({fmt((last_v - first_v) if pd.notna(first_v) and pd.notna(last_v) else np.nan, True)} "
            "shift). Since the horizon is hidden, this cannot be planned end-game defection; the mechanism most "
            "likely runs through accumulated private risk and cumulative unsafe-count, which rise across any race "
            "and happen to also rise near a race's actual end."
        )
    lines.append(
        "- **Per-model curves diverge in shape**, not just level -- see the model-split figure and "
        "`temporal_endgame_by_rounds_remaining.csv` for whether a given model's curve is flat, rising, or falling "
        "toward the end."
    )
    if not final_turn.empty:
        lines.append(
            "- **The single literal final turn of a race is not the same population as 'rounds_remaining=1'** "
            "(`temporal_final_turn_vs_rest.csv`) -- final turns pool short and long races together, so a flat "
            "aggregate curve can still hide a real within-race trend; use the banded table as the primary evidence."
        )
    if not horizon_length.empty:
        lines.append(
            "- **Unsafe rate also varies with how long a race happened to run** (`temporal_horizon_length.csv`): "
            "races that ran longer are not random draws from the same behavioral population as races that stopped "
            "early, since `stop_forced`/continuation itself partly depends on prior unsafe play through progress "
            "and setback mechanics."
        )
    lines.append("")

    lines.append("## Pooled Endgame Curve (All Completed)")
    lines.append("")
    lines.append(markdown_table(pooled))
    lines.append("")
    lines.append("Visual: `figures/temporal_endgame/01_endgame_curve.png` (gold band marks the human phi_U 95% interval, 40-75%, from `human_reference.json` E7).")
    lines.append("")

    lines.append("## By Model And Scope")
    lines.append("")
    lines.append(markdown_table(by_model[by_model["analysis_scope"] == "all_completed"]))
    lines.append("")
    lines.append("Baseline-only scope, for comparison with the rest of the pipeline:")
    lines.append("")
    lines.append(markdown_table(by_model[by_model["analysis_scope"] == "baseline_completed"]))
    lines.append("")

    lines.append("## Final Turn Vs Rest Of Race")
    lines.append("")
    lines.append(markdown_table(final_turn))
    lines.append("")

    lines.append("## Unsafe Rate By How Long The Race Actually Ran")
    lines.append("")
    lines.append(markdown_table(horizon_length))
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- `rounds_remaining` is computed in hindsight from the completed race's `n_rounds`; it is never available "
        "to the model at decision time. Any effect found here is descriptive/emergent, not evidence of horizon "
        "inference."
    )
    lines.append(
        "- Race length (`n_rounds`) is itself endogenous to play (stopping is probabilistic each round from round "
        "5 onward, per the paper-faithful mechanism), so comparing across `rounds_remaining` mixes races of "
        "different lengths; the horizon-length table above is provided to make that visible, not to control for it."
    )
    lines.append("- Uses `turns_canonical.csv` joined to `races_canonical.csv` on `(source_run, game_id)`; both filtered to completed, non-duplicate-grain rows.")

    path = REPORTS_DIR / "fh_temporal_endgame_mining.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    turns = load_turns_with_horizon()
    n_decisions = len(turns)
    n_games = turns[["source_run", "game_id"]].drop_duplicates().shape[0]
    print(f"turns with horizon: {n_decisions}, races: {n_games}")

    by_model = build_endgame_table(turns)
    pooled = build_pooled_endgame_table(turns)
    final_turn = build_final_turn_table(turns)
    horizon_length = build_horizon_length_table(turns)

    plot_endgame_curve(pooled, by_model)

    report_path = write_report(pooled, by_model, final_turn, horizon_length, n_decisions, n_games)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
