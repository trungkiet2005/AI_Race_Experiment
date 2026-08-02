#!/usr/bin/env python3
"""Fit an expanded deterministic strategy library to player trajectories.

The goal is to test whether strategies beyond AS/AU/CS/CAS cover enough
observed LLM behavior to be treated as meaningful candidate strategy classes.
"""

from __future__ import annotations

import json
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
EXPANDED_FIG_DIR = FIGURES_DIR / "expanded_strategy_library"
REPORTS_DIR = OUTPUT_DIR / "reports"

BLUE = "#2F6B9A"
ORANGE = "#D9822B"
GOLD = "#C9A227"
OLIVE = "#6B7D3D"
PINK = "#B45A7C"
TEAL = "#3A8F8A"
PURPLE = "#7C4D79"
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
FAMILY_COLORS = {
    "canonical": BLUE,
    "pd_memory": TEAL,
    "positional_gap": ORANGE,
    "hybrid_gap_memory": PINK,
    "sequence_motif": GOLD,
    "uncovered": MUTED,
}


@dataclass(frozen=True)
class StrategyRule:
    name: str
    family: str
    description: str
    predictor: Callable[[pd.DataFrame], np.ndarray]


def ensure_dirs() -> None:
    for path in [DERIVED_DIR, EXPANDED_FIG_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": WHITE,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, filename: str) -> Path:
    path = EXPANDED_FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def strategy_key(row: pd.Series) -> str:
    return f"{row['source_run']}|{row['game_id']}|p{int(row['player_index'])}"


def safe_first(frame: pd.DataFrame) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=int)
    return pred


def unsafe_first(frame: pd.DataFrame) -> np.ndarray:
    pred = np.ones(len(frame), dtype=int)
    return pred


def opponent_prev(frame: pd.DataFrame, default: int = 0) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=int)
    pred[0] = default
    if len(frame) > 1:
        pred[1:] = frame["opponent_prev_unsafe"].iloc[1:].fillna(0).astype(int).to_numpy()
    return pred


def anti_opponent_prev(frame: pd.DataFrame, default: int = 0) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=int)
    pred[0] = default
    if len(frame) > 1:
        pred[1:] = 1 - frame["opponent_prev_unsafe"].iloc[1:].fillna(0).astype(int).to_numpy()
    return pred


def wsls(frame: pd.DataFrame, initial: int = 0) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=int)
    pred[0] = initial
    for i in range(1, len(frame)):
        opp_prev = int(frame["opponent_prev_unsafe"].iloc[i]) if pd.notna(frame["opponent_prev_unsafe"].iloc[i]) else 0
        own_prev = pred[i - 1]
        pred[i] = 0 if own_prev == opp_prev else 1
    return pred


def grim(frame: pd.DataFrame, initial: int = 0) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=int)
    triggered = bool(initial)
    pred[0] = initial
    for i in range(1, len(frame)):
        opp_prev = int(frame["opponent_prev_unsafe"].iloc[i]) if pd.notna(frame["opponent_prev_unsafe"].iloc[i]) else 0
        triggered = triggered or bool(opp_prev)
        pred[i] = 1 if triggered else 0
    return pred


def two_tits_for_tat(frame: pd.DataFrame, initial: int = 0) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=int)
    pred[0] = initial
    for i in range(1, len(frame)):
        if i == 1:
            pred[i] = 0
        else:
            last1 = int(frame["opponent_prev_unsafe"].iloc[i]) if pd.notna(frame["opponent_prev_unsafe"].iloc[i]) else 0
            last2 = int(frame["opponent_prev_unsafe"].iloc[i - 1]) if pd.notna(frame["opponent_prev_unsafe"].iloc[i - 1]) else 0
            pred[i] = 1 if last1 and last2 else 0
    return pred


def one_shot_punisher(frame: pd.DataFrame, initial: int = 0) -> np.ndarray:
    pred = np.zeros(len(frame), dtype=int)
    pred[0] = initial
    for i in range(1, len(frame)):
        opp_prev = int(frame["opponent_prev_unsafe"].iloc[i]) if pd.notna(frame["opponent_prev_unsafe"].iloc[i]) else 0
        pred[i] = 1 if opp_prev and pred[i - 1] == 0 else 0
    return pred


def gap_rule(frame: pd.DataFrame, mode: str) -> np.ndarray:
    gap = frame["progress_gap_before"].fillna(0).astype(float).to_numpy()
    if mode == "ahead":
        return (gap > 0).astype(int)
    if mode == "ahead_strict":
        return (gap > 0.5).astype(int)
    if mode == "behind":
        return (gap < 0).astype(int)
    if mode == "behind_strict":
        return (gap < -0.5).astype(int)
    if mode == "uneven":
        return (np.abs(gap) > 0.5).astype(int)
    if mode == "safe_if_far_ahead":
        return (gap <= 1.0).astype(int)
    if mode == "unsafe_if_not_behind":
        return (gap >= 0).astype(int)
    raise ValueError(mode)


def hybrid_rule(frame: pd.DataFrame, mode: str) -> np.ndarray:
    gap = frame["progress_gap_before"].fillna(0).astype(float).to_numpy()
    opp = opponent_prev(frame, default=0)
    if mode == "leader_or_retaliate":
        return ((gap > 0) | (opp == 1)).astype(int)
    if mode == "leader_and_not_exploited":
        return ((gap > 0) & (opp == 0)).astype(int)
    if mode == "catchup_or_retaliate":
        return ((gap < 0) | (opp == 1)).astype(int)
    if mode == "defensive_safe_when_behind":
        return ((gap >= 0) & (opp == 1)).astype(int)
    if mode == "bully_until_far_ahead":
        return ((gap <= 1.0) | (opp == 1)).astype(int)
    raise ValueError(mode)


def alternating(frame: pd.DataFrame, initial: int = 0) -> np.ndarray:
    return np.array([(initial + i) % 2 for i in range(len(frame))], dtype=int)


def delayed_attack(frame: pd.DataFrame, safe_rounds: int) -> np.ndarray:
    return np.array([0 if i < safe_rounds else 1 for i in range(len(frame))], dtype=int)


def cooldown(frame: pd.DataFrame, unsafe_rounds: int) -> np.ndarray:
    return np.array([1 if i < unsafe_rounds else 0 for i in range(len(frame))], dtype=int)


def build_library() -> list[StrategyRule]:
    return [
        StrategyRule("AS", "canonical", "always safe", lambda f: safe_first(f)),
        StrategyRule("AU", "canonical", "always unsafe", lambda f: unsafe_first(f)),
        StrategyRule("CS", "canonical", "safe first, then copy opponent previous action", lambda f: opponent_prev(f, 0)),
        StrategyRule("CAS", "canonical", "unsafe first, then copy opponent previous action", lambda f: opponent_prev(f, 1)),
        StrategyRule("ANTI_CS", "pd_memory", "safe first, then do the opposite of opponent previous action", lambda f: anti_opponent_prev(f, 0)),
        StrategyRule("ANTI_CAS", "pd_memory", "unsafe first, then do the opposite of opponent previous action", lambda f: anti_opponent_prev(f, 1)),
        StrategyRule("WSLS_S", "pd_memory", "Pavlov / win-stay lose-shift, safe first", lambda f: wsls(f, 0)),
        StrategyRule("WSLS_U", "pd_memory", "Pavlov / win-stay lose-shift, unsafe first", lambda f: wsls(f, 1)),
        StrategyRule("GRIM_S", "pd_memory", "safe until opponent ever plays unsafe, then always unsafe", lambda f: grim(f, 0)),
        StrategyRule("GRIM_U", "pd_memory", "unsafe first and remains unsafe after any opponent unsafe", lambda f: grim(f, 1)),
        StrategyRule("TF2T_S", "pd_memory", "unsafe only after two consecutive opponent unsafe actions", lambda f: two_tits_for_tat(f, 0)),
        StrategyRule("ONE_SHOT_PUNISHER", "pd_memory", "retaliate one turn, then force cooldown", lambda f: one_shot_punisher(f, 0)),
        StrategyRule("LEADER_ATTACK", "positional_gap", "unsafe when ahead, safe otherwise", lambda f: gap_rule(f, "ahead")),
        StrategyRule("STRICT_LEADER_ATTACK", "positional_gap", "unsafe only when ahead by more than 0.5", lambda f: gap_rule(f, "ahead_strict")),
        StrategyRule("CATCHUP_ATTACK", "positional_gap", "unsafe when behind, safe otherwise", lambda f: gap_rule(f, "behind")),
        StrategyRule("DESPERATE_CATCHUP", "positional_gap", "unsafe only when behind by more than 0.5", lambda f: gap_rule(f, "behind_strict")),
        StrategyRule("UNEVEN_ATTACK", "positional_gap", "unsafe when the race is not close", lambda f: gap_rule(f, "uneven")),
        StrategyRule("BULLY_UNTIL_FAR_AHEAD", "positional_gap", "unsafe unless already ahead by more than 1", lambda f: gap_rule(f, "safe_if_far_ahead")),
        StrategyRule("PRESS_WHEN_NOT_BEHIND", "positional_gap", "unsafe when tied or ahead", lambda f: gap_rule(f, "unsafe_if_not_behind")),
        StrategyRule("LEADER_OR_RETALIATE", "hybrid_gap_memory", "unsafe when ahead or after opponent unsafe", lambda f: hybrid_rule(f, "leader_or_retaliate")),
        StrategyRule("OPPORTUNIST", "hybrid_gap_memory", "unsafe when ahead and opponent was previously safe", lambda f: hybrid_rule(f, "leader_and_not_exploited")),
        StrategyRule("CATCHUP_OR_RETALIATE", "hybrid_gap_memory", "unsafe when behind or after opponent unsafe", lambda f: hybrid_rule(f, "catchup_or_retaliate")),
        StrategyRule("DEFENSIVE_RETALIATOR", "hybrid_gap_memory", "retaliate only when not behind", lambda f: hybrid_rule(f, "defensive_safe_when_behind")),
        StrategyRule("BULLY_RETALIATOR", "hybrid_gap_memory", "unsafe until far ahead, also retaliates", lambda f: hybrid_rule(f, "bully_until_far_ahead")),
        StrategyRule("ALT_SAFE_FIRST", "sequence_motif", "alternate safe/unsafe starting safe", lambda f: alternating(f, 0)),
        StrategyRule("ALT_UNSAFE_FIRST", "sequence_motif", "alternate unsafe/safe starting unsafe", lambda f: alternating(f, 1)),
        StrategyRule("DELAYED_ATTACK_R2", "sequence_motif", "safe once, then always unsafe", lambda f: delayed_attack(f, 1)),
        StrategyRule("DELAYED_ATTACK_R3", "sequence_motif", "safe twice, then always unsafe", lambda f: delayed_attack(f, 2)),
        StrategyRule("PROBE_COOLDOWN_R2", "sequence_motif", "unsafe once, then always safe", lambda f: cooldown(f, 1)),
        StrategyRule("PROBE_COOLDOWN_R3", "sequence_motif", "unsafe twice, then always safe", lambda f: cooldown(f, 2)),
    ]


def classify_players(turns: pd.DataFrame, library: list[StrategyRule]) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    detail_rows: list[dict[str, object]] = []
    work = turns.copy()
    work["sequence_id"] = work.apply(strategy_key, axis=1)
    work = work.sort_values(["source_run", "game_id", "player_index", "round"])
    canonical_names = {"AS", "AU", "CS", "CAS"}

    for sequence_id, frame in work.groupby("sequence_id", sort=False):
        frame = frame.sort_values("round").copy()
        actual = frame["unsafe"].fillna(0).astype(int).to_numpy()
        first = frame.iloc[0]
        candidate_rows: list[dict[str, object]] = []
        for rule in library:
            pred = rule.predictor(frame)
            mismatches = int(np.sum(pred != actual))
            rate = mismatches / len(actual)
            exact = mismatches == 0
            one_mismatch = mismatches <= 1
            near_10pct = rate <= 0.10
            candidate_rows.append(
                {
                    "sequence_id": sequence_id,
                    "strategy": rule.name,
                    "strategy_family": rule.family,
                    "description": rule.description,
                    "mismatches": mismatches,
                    "mismatch_rate": rate,
                    "exact_fit": exact,
                    "one_mismatch_fit": one_mismatch,
                    "near_10pct_fit": near_10pct,
                }
            )
        detail_rows.extend(candidate_rows)
        detail = pd.DataFrame(candidate_rows)
        canon_min = int(detail[detail["strategy"].isin(canonical_names)]["mismatches"].min())
        expanded_min = int(detail["mismatches"].min())
        exact = detail[detail["mismatches"].eq(expanded_min)].copy()
        exact["priority"] = exact["strategy_family"].map(
            {"canonical": 0, "pd_memory": 1, "positional_gap": 2, "hybrid_gap_memory": 3, "sequence_motif": 4}
        )
        best = exact.sort_values(["priority", "strategy"]).iloc[0]
        exact_expanded = detail[detail["exact_fit"]]
        exact_canonical = exact_expanded[exact_expanded["strategy"].isin(canonical_names)]
        exact_new = exact_expanded[~exact_expanded["strategy"].isin(canonical_names)]
        near_expanded = detail[detail["one_mismatch_fit"]]
        near_canonical = near_expanded[near_expanded["strategy"].isin(canonical_names)]
        rows.append(
            {
                "sequence_id": sequence_id,
                "source_run": first["source_run"],
                "game_id": first["game_id"],
                "player_index": int(first["player_index"]),
                "model_slug": first["model_slug"],
                "family": first["family"],
                "provider": first["provider"],
                "persona_mode": first["persona_mode"],
                "condition": first["condition"],
                "analysis_scope": first["analysis_scope"],
                "n_turns": len(actual),
                "unsafe_turns": int(actual.sum()),
                "unsafe_rate": float(actual.mean()),
                "action_pattern": "".join("U" if x else "S" for x in actual),
                "canonical_min_mismatches": canon_min,
                "canonical_min_mismatch_rate": canon_min / len(actual),
                "canonical_exact_any": bool(canon_min == 0),
                "canonical_near_one_mismatch": bool(canon_min <= 1),
                "expanded_min_mismatches": expanded_min,
                "expanded_min_mismatch_rate": expanded_min / len(actual),
                "expanded_best_strategy": best["strategy"],
                "expanded_best_family": best["strategy_family"],
                "expanded_exact_any": bool(expanded_min == 0),
                "expanded_near_one_mismatch": bool(expanded_min <= 1),
                "exact_fit_strategies": "|".join(exact_expanded["strategy"].sort_values().tolist()),
                "exact_new_strategies": "|".join(exact_new["strategy"].sort_values().tolist()),
                "new_exact_beyond_canonical": bool(canon_min > 0 and len(exact_new) > 0),
                "near_new_beyond_canonical": bool(canon_min > 1 and len(near_expanded[~near_expanded["strategy"].isin(canonical_names)]) > 0),
                "best_is_new_exact_beyond_canonical": bool(canon_min > 0 and expanded_min == 0),
                "canonical_exact_strategy_count": int(len(exact_canonical)),
                "expanded_exact_strategy_count": int(len(exact_expanded)),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(detail_rows)


def summarize_strategy_coverage(players: pd.DataFrame, detail: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    exact_counts = (
        detail[detail["exact_fit"]]
        .merge(players[["sequence_id", "canonical_exact_any"]], on="sequence_id", how="left")
        .assign(exact_new_beyond_canonical=lambda d: ~d["canonical_exact_any"] & ~d["strategy"].isin(["AS", "AU", "CS", "CAS"]))
        .groupby(["strategy", "strategy_family", "description"], observed=True)
        .agg(
            exact_fit_sequences=("sequence_id", "nunique"),
            exact_new_beyond_canonical=("exact_new_beyond_canonical", "sum"),
        )
        .reset_index()
        .sort_values(["exact_new_beyond_canonical", "exact_fit_sequences"], ascending=False)
    )
    near_counts = (
        detail[detail["one_mismatch_fit"]]
        .merge(players[["sequence_id", "canonical_near_one_mismatch"]], on="sequence_id", how="left")
        .assign(near_new_beyond_canonical=lambda d: ~d["canonical_near_one_mismatch"] & ~d["strategy"].isin(["AS", "AU", "CS", "CAS"]))
        .groupby(["strategy", "strategy_family", "description"], observed=True)
        .agg(
            near_one_mismatch_sequences=("sequence_id", "nunique"),
            near_new_beyond_canonical=("near_new_beyond_canonical", "sum"),
        )
        .reset_index()
        .sort_values(["near_new_beyond_canonical", "near_one_mismatch_sequences"], ascending=False)
    )
    family_summary = (
        players.groupby(["expanded_best_family"], observed=True)
        .agg(
            best_fit_sequences=("sequence_id", "size"),
            exact_any=("expanded_exact_any", "sum"),
            new_exact_beyond_canonical=("new_exact_beyond_canonical", "sum"),
            near_one_mismatch=("expanded_near_one_mismatch", "sum"),
        )
        .reset_index()
    )
    return exact_counts, near_counts, family_summary


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100 * value:.1f}%"


def plot_coverage_lift(players: pd.DataFrame) -> Path:
    rows = [
        ("4 canonical\nexact", players["canonical_exact_any"].mean()),
        ("Expanded\nexact", players["expanded_exact_any"].mean()),
        ("New exact\nbeyond 4", players["new_exact_beyond_canonical"].mean()),
        ("4 canonical\n<=1 mismatch", players["canonical_near_one_mismatch"].mean()),
        ("Expanded\n<=1 mismatch", players["expanded_near_one_mismatch"].mean()),
        ("New near\nbeyond 4", players["near_new_beyond_canonical"].mean()),
    ]
    labels, values = zip(*rows)
    colors = [BLUE, TEAL, ORANGE, BLUE, TEAL, ORANGE]
    fig, ax = plt.subplots(figsize=(10.5, 6.3))
    bars = ax.bar(np.arange(len(values)), values, color=colors, edgecolor=WHITE, linewidth=1)
    ax.set_ylim(0, min(1.0, max(values) * 1.22))
    ax.set_xticks(np.arange(len(values)))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Share of player trajectories")
    ax.set_title("Coverage Lift from Expanded Strategy Library")
    ax.text(
        0,
        1.04,
        "Exact means zero mismatches over the full observed trajectory. Near means at most one mismatch.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, pct(value), ha="center", va="bottom", fontsize=10)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    fig.subplots_adjust(left=0.12, right=0.98, top=0.84, bottom=0.18)
    return savefig(fig, "01_coverage_lift.png")


def plot_new_exact_strategies(exact_counts: pd.DataFrame) -> Path:
    data = exact_counts[exact_counts["exact_new_beyond_canonical"].gt(0)].head(12).copy()
    if data.empty:
        data = exact_counts.head(12).copy()
    data = data.sort_values("exact_new_beyond_canonical")
    fig, ax = plt.subplots(figsize=(12, max(6, len(data) * 0.42)))
    colors = [FAMILY_COLORS.get(fam, MUTED) for fam in data["strategy_family"]]
    bars = ax.barh(np.arange(len(data)), data["exact_new_beyond_canonical"], color=colors, edgecolor=WHITE, linewidth=0.8)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["strategy"])
    ax.set_xlabel("Player trajectories exactly fit beyond the 4 canonical rules")
    ax.set_title("New Exact-Coverage Strategies")
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            f"{int(row['exact_new_beyond_canonical'])}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.28, right=0.93, top=0.86, bottom=0.14)
    return savefig(fig, "02_new_exact_strategies.png")


def plot_best_family_by_model(players: pd.DataFrame) -> Path:
    data = players.copy()
    data["best_family_plot"] = np.where(data["expanded_exact_any"], data["expanded_best_family"], "uncovered")
    counts = (
        data.groupby(["model_slug", "best_family_plot"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    counts["share"] = counts["n"] / counts.groupby("model_slug")["n"].transform("sum")
    plot = counts.pivot(index="model_slug", columns="best_family_plot", values="share").fillna(0)
    order = [m for m in MODEL_ORDER if m in plot.index] + [m for m in plot.index if m not in MODEL_ORDER]
    plot = plot.loc[order]
    cols = [c for c in ["canonical", "pd_memory", "positional_gap", "hybrid_gap_memory", "sequence_motif", "uncovered"] if c in plot.columns]

    fig, ax = plt.subplots(figsize=(12.5, 7))
    left = np.zeros(len(plot))
    y = np.arange(len(plot))
    for col in cols:
        values = plot[col].to_numpy()
        ax.barh(y, values, left=left, color=FAMILY_COLORS.get(col, MUTED), edgecolor=WHITE, linewidth=0.8, label=col)
        for i, value in enumerate(values):
            if value >= 0.08:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", color=WHITE, fontsize=8)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in plot.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of player trajectories")
    ax.set_title("Expanded Exact-Fit Strategy Family by Model")
    ax.text(
        0,
        1.04,
        "Family is assigned only when the expanded library exactly fits the full trajectory; otherwise uncovered.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.22)
    return savefig(fig, "03_exact_family_by_model.png")


def plot_near_family_by_model(players: pd.DataFrame) -> Path:
    data = players.copy()
    data["near_family_plot"] = np.where(data["expanded_near_one_mismatch"], data["expanded_best_family"], "uncovered")
    counts = (
        data.groupby(["model_slug", "near_family_plot"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    counts["share"] = counts["n"] / counts.groupby("model_slug")["n"].transform("sum")
    plot = counts.pivot(index="model_slug", columns="near_family_plot", values="share").fillna(0)
    order = [m for m in MODEL_ORDER if m in plot.index] + [m for m in plot.index if m not in MODEL_ORDER]
    plot = plot.loc[order]
    cols = [c for c in ["canonical", "pd_memory", "positional_gap", "hybrid_gap_memory", "sequence_motif", "uncovered"] if c in plot.columns]

    fig, ax = plt.subplots(figsize=(12.5, 7))
    left = np.zeros(len(plot))
    y = np.arange(len(plot))
    for col in cols:
        values = plot[col].to_numpy()
        ax.barh(y, values, left=left, color=FAMILY_COLORS.get(col, MUTED), edgecolor=WHITE, linewidth=0.8, label=col)
        for i, value in enumerate(values):
            if value >= 0.08:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", color=WHITE, fontsize=8)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in plot.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of player trajectories")
    ax.set_title("Expanded Near-Fit Strategy Family by Model")
    ax.text(
        0,
        1.04,
        "Near fit means at most one action mismatch over the full trajectory.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.22)
    return savefig(fig, "04_near_family_by_model.png")


def plot_new_by_length(players: pd.DataFrame) -> Path:
    summary = (
        players.groupby("n_turns")
        .agg(
            sequences=("sequence_id", "size"),
            canonical_exact=("canonical_exact_any", "mean"),
            expanded_exact=("expanded_exact_any", "mean"),
            new_exact=("new_exact_beyond_canonical", "mean"),
            expanded_near=("expanded_near_one_mismatch", "mean"),
        )
        .reset_index()
    )
    summary = summary[summary["sequences"].ge(20)]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.plot(summary["n_turns"], summary["canonical_exact"], marker="o", linewidth=2, color=BLUE, label="4 canonical exact")
    ax.plot(summary["n_turns"], summary["expanded_exact"], marker="o", linewidth=2, color=TEAL, label="expanded exact")
    ax.plot(summary["n_turns"], summary["expanded_near"], marker="o", linewidth=2, color=ORANGE, label="expanded <=1 mismatch")
    ax.set_ylim(0, 1)
    ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    ax.set_xlabel("Observed trajectory length")
    ax.set_ylabel("Coverage share")
    ax.set_title("Coverage by Game Length")
    ax.text(0, 1.04, "Longer trajectories are harder to fit exactly; near-fit coverage is the robustness check.", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.1, right=0.97, top=0.84, bottom=0.14)
    return savefig(fig, "05_coverage_by_length.png")


def make_contact_sheet(paths: list[Path]) -> Path:
    images = [mpimg.imread(path) for path in paths]
    fig, axes = plt.subplots(3, 2, figsize=(16, 19))
    axes = axes.ravel()
    for ax, image, path in zip(axes, images, paths):
        ax.imshow(image)
        ax.set_title(path.stem.replace("_", " ").title(), fontsize=11, pad=8)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.suptitle("Expanded Strategy Library Coverage", fontsize=22, y=0.995)
    fig.subplots_adjust(top=0.965, hspace=0.08, wspace=0.04)
    path = EXPANDED_FIG_DIR / "fh_expanded_strategy_library_contact_sheet.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    players: pd.DataFrame,
    exact_counts: pd.DataFrame,
    near_counts: pd.DataFrame,
    figures: list[Path],
    contact_sheet: Path,
) -> Path:
    n = len(players)
    canonical_exact = players["canonical_exact_any"].mean()
    expanded_exact = players["expanded_exact_any"].mean()
    new_exact = players["new_exact_beyond_canonical"].mean()
    canonical_near = players["canonical_near_one_mismatch"].mean()
    expanded_near = players["expanded_near_one_mismatch"].mean()
    new_near = players["near_new_beyond_canonical"].mean()

    top_exact_lines = []
    for _, row in exact_counts[exact_counts["exact_new_beyond_canonical"].gt(0)].head(12).iterrows():
        top_exact_lines.append(
            f"- `{row['strategy']}` ({row['strategy_family']}): +{int(row['exact_new_beyond_canonical'])} exact trajectories beyond canonical; "
            f"{int(row['exact_fit_sequences'])} exact total. Meaning: {row['description']}."
        )
    top_near_lines = []
    for _, row in near_counts[near_counts["near_new_beyond_canonical"].gt(0)].head(12).iterrows():
        top_near_lines.append(
            f"- `{row['strategy']}` ({row['strategy_family']}): +{int(row['near_new_beyond_canonical'])} near trajectories beyond canonical; "
            f"{int(row['near_one_mismatch_sequences'])} near total. Meaning: {row['description']}."
        )

    model_summary = (
        players.groupby("model_slug")
        .agg(
            sequences=("sequence_id", "size"),
            canonical_exact=("canonical_exact_any", "mean"),
            expanded_exact=("expanded_exact_any", "mean"),
            new_exact=("new_exact_beyond_canonical", "mean"),
            expanded_near=("expanded_near_one_mismatch", "mean"),
        )
        .reset_index()
        .sort_values("new_exact", ascending=False)
    )
    model_lines = [
        f"- {MODEL_LABELS.get(row['model_slug'], row['model_slug'])}: canonical exact {pct(row['canonical_exact'])}, expanded exact {pct(row['expanded_exact'])}, new exact {pct(row['new_exact'])}, expanded near {pct(row['expanded_near'])}."
        for _, row in model_summary.iterrows()
    ]

    body = f"""# Expanded Strategy Library Fit

## Question

The original paper assumes four deterministic strategies: AS, AU, CS, CAS. This analysis tests whether a larger, interpretable strategy library can explain additional LLM behavior with exact or near-exact trajectory fits.

## Coverage Result

- Player trajectories: {n:,}
- 4-strategy exact coverage: {pct(canonical_exact)}
- Expanded-library exact coverage: {pct(expanded_exact)}
- New exact coverage beyond the 4 strategies: {pct(new_exact)}
- 4-strategy near coverage, at most one mismatch: {pct(canonical_near)}
- Expanded-library near coverage, at most one mismatch: {pct(expanded_near)}
- New near coverage beyond the 4 strategies: {pct(new_near)}

## Model-Level Coverage

{chr(10).join(model_lines)}

## Candidate New Exact Strategies

{chr(10).join(top_exact_lines) if top_exact_lines else "- No noncanonical strategy added exact coverage beyond the original four."}

## Candidate New Near-Fit Strategies

{chr(10).join(top_near_lines) if top_near_lines else "- No noncanonical strategy added near-fit coverage beyond the original four."}

## Interpretation

Exact coverage is the strict test. Near-fit coverage is the behavioral test: it asks whether a rule captures almost all moves in the observed sequence. If a new rule adds many exact or near trajectories beyond AS/AU/CS/CAS and has a simple interpretation, it is a candidate publishable strategy class.

## Caveat

Gap-based rules use the observed race position at each turn. They are descriptive policy fits, not full counterfactual simulations of what the game state would have been if the strategy had been played from the start.

## Deliverables

- Contact sheet: `{contact_sheet}`
- Figures: {", ".join(f"`{path.name}`" for path in figures)}
- Tables: `expanded_strategy_player_fits.csv`, `expanded_strategy_fit_detail.csv`, `expanded_strategy_exact_counts.csv`, `expanded_strategy_near_counts.csv`, `expanded_strategy_family_summary.csv`
"""
    path = REPORTS_DIR / "fh_expanded_strategy_library_fit.md"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    configure_matplotlib()
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv", low_memory=False)
    turns = turns[turns["manifest_status"].eq("completed") & ~turns["duplicate_grain_key"].fillna(False).astype(bool)].copy()
    turns = turns[turns["unsafe"].notna()].copy()
    for col in ["unsafe", "round", "player_index", "progress_gap_before", "opponent_prev_unsafe"]:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")

    library = build_library()
    players, detail = classify_players(turns, library)
    exact_counts, near_counts, family_summary = summarize_strategy_coverage(players, detail)

    players.to_csv(DERIVED_DIR / "expanded_strategy_player_fits.csv", index=False)
    detail.to_csv(DERIVED_DIR / "expanded_strategy_fit_detail.csv", index=False)
    exact_counts.to_csv(DERIVED_DIR / "expanded_strategy_exact_counts.csv", index=False)
    near_counts.to_csv(DERIVED_DIR / "expanded_strategy_near_counts.csv", index=False)
    family_summary.to_csv(DERIVED_DIR / "expanded_strategy_family_summary.csv", index=False)

    figures = [
        plot_coverage_lift(players),
        plot_new_exact_strategies(exact_counts),
        plot_best_family_by_model(players),
        plot_near_family_by_model(players),
        plot_new_by_length(players),
    ]
    contact_sheet = make_contact_sheet(figures)
    report = write_report(players, exact_counts, near_counts, figures, contact_sheet)
    print(
        json.dumps(
            {
                "player_trajectories": int(len(players)),
                "canonical_exact_share": float(players["canonical_exact_any"].mean()),
                "expanded_exact_share": float(players["expanded_exact_any"].mean()),
                "new_exact_beyond_canonical_share": float(players["new_exact_beyond_canonical"].mean()),
                "canonical_near_one_mismatch_share": float(players["canonical_near_one_mismatch"].mean()),
                "expanded_near_one_mismatch_share": float(players["expanded_near_one_mismatch"].mean()),
                "new_near_beyond_canonical_share": float(players["near_new_beyond_canonical"].mean()),
                "report": str(report),
                "contact_sheet": str(contact_sheet),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
