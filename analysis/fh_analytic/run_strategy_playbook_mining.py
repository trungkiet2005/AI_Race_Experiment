#!/usr/bin/env python3
"""Mine strategic playbook patterns beyond canonical strategy labels."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
PLAYBOOK_FIG_DIR = FIGURES_DIR / "strategy_playbook"
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
MODEL_COLORS = {
    "gpt-5-nano": BLUE,
    "gpt-5.4-nano": PINK,
    "google-gemini-3-flash-preview": ORANGE,
    "google-gemini-3.1-flash-lite-preview": GOLD,
    "google-gemini-3.5-flash-lite": OLIVE,
}

MOTIF_ORDER = [
    "all_safe",
    "all_unsafe",
    "alternating",
    "late_escalation",
    "cooldown",
    "safe_then_attack",
    "probe_then_cool",
    "mixed_adaptive",
]
MOTIF_COLORS = {
    "all_safe": BLUE,
    "all_unsafe": PINK,
    "alternating": PURPLE,
    "late_escalation": ORANGE,
    "cooldown": TEAL,
    "safe_then_attack": GOLD,
    "probe_then_cool": OLIVE,
    "mixed_adaptive": MUTED,
}
DYAD_COLORS = {
    "mutual_safe": BLUE,
    "mutual_unsafe": PINK,
    "asymmetric_exploitation": ORANGE,
    "mutual_escalation": GOLD,
    "mutual_deescalation": TEAL,
    "alternating_collision": PURPLE,
    "mixed": MUTED,
}


def ensure_dirs() -> None:
    for path in [DERIVED_DIR, PLAYBOOK_FIG_DIR, REPORTS_DIR]:
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
    path = PLAYBOOK_FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def strategy_key(row: pd.Series) -> str:
    return f"{row['source_run']}|{row['game_id']}|p{int(row['player_index'])}"


def classify_motif(pattern: str) -> str:
    n = len(pattern)
    unsafe = np.array([1 if char == "U" else 0 for char in pattern], dtype=int)
    unsafe_rate = float(unsafe.mean()) if n else np.nan
    switches = int(np.sum(unsafe[1:] != unsafe[:-1])) if n > 1 else 0
    switch_rate = switches / max(n - 1, 1)
    first_half = unsafe[: max(1, n // 2)]
    second_half = unsafe[max(1, n // 2) :]

    if unsafe_rate == 0:
        return "all_safe"
    if unsafe_rate == 1:
        return "all_unsafe"
    if n >= 4 and switch_rate >= 0.65:
        return "alternating"
    if len(second_half) and first_half.mean() <= 0.25 and second_half.mean() >= 0.55:
        return "late_escalation"
    if len(second_half) and first_half.mean() >= 0.55 and second_half.mean() <= 0.25:
        return "cooldown"
    if pattern[0] == "S" and "U" in pattern[1:]:
        return "safe_then_attack"
    if pattern[0] == "U" and unsafe_rate <= 0.45 and "S" in pattern[1:]:
        return "probe_then_cool"
    return "mixed_adaptive"


def seq_switches(pattern: str) -> int:
    return sum(pattern[i] != pattern[i - 1] for i in range(1, len(pattern)))


def add_strategy_context(turns: pd.DataFrame, players: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    players = players.copy()
    players["play_motif"] = players["action_pattern"].map(classify_motif)
    players["switches"] = players["action_pattern"].map(seq_switches)
    players["switch_rate"] = players["switches"] / (players["n_turns"].astype(float) - 1).clip(lower=1)
    players["opens_unsafe"] = players["action_pattern"].str[0].eq("U")
    players["ever_switches"] = players["switches"].gt(0)
    players["strategy_label"] = np.where(
        players["is_exact_single_canonical"].astype(bool),
        players["strategy_best"],
        players["residual_signature"],
    )

    turns = turns.copy()
    turns["sequence_id"] = turns.apply(strategy_key, axis=1)
    join_cols = [
        "sequence_id",
        "action_pattern",
        "play_motif",
        "switches",
        "switch_rate",
        "opens_unsafe",
        "ever_switches",
        "strategy_label",
        "is_exact_single_canonical",
        "residual_signature",
    ]
    turns = turns.merge(players[join_cols], on="sequence_id", how="left")
    turns["gap_zone"] = np.select(
        [turns["progress_gap_before"] < -0.5, turns["progress_gap_before"] > 0.5],
        ["behind", "ahead"],
        default="tied",
    )
    for col in ["unsafe", "own_prev_unsafe", "opponent_prev_unsafe", "round", "progress_gap_before"]:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")
    state = turns[["own_prev_unsafe", "opponent_prev_unsafe"]].fillna(-1).astype(int).astype(str)
    turns["prev_pair"] = state["own_prev_unsafe"].map({"0": "S", "1": "U", "-1": "?"}) + state[
        "opponent_prev_unsafe"
    ].map({"0": "S", "1": "U", "-1": "?"})
    turns["switch_to_unsafe"] = turns["unsafe"].eq(1) & turns["own_prev_unsafe"].eq(0)
    turns["switch_to_safe"] = turns["unsafe"].eq(0) & turns["own_prev_unsafe"].eq(1)
    return turns, players


def safe_rate(frame: pd.DataFrame, mask: pd.Series, col: str = "unsafe", min_n: int = 30) -> tuple[float, int]:
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    return float(frame.loc[mask, col].mean()), n


def mine_levers(turns: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    round2 = turns[turns["round"].ge(2)].copy()
    scopes: list[tuple[str, list[str]]] = [
        ("overall", []),
        ("model", ["model_slug"]),
        ("family", ["family"]),
        ("motif", ["play_motif"]),
        ("strategy_label", ["strategy_label"]),
    ]
    rows: list[dict[str, Any]] = []
    for scope, keys in scopes:
        iterator = [((), round2)] if not keys else round2.groupby(keys, dropna=False)
        for values, frame in iterator:
            if not isinstance(values, tuple):
                values = (values,)
            dims = dict(zip(keys, values))
            seqs = players
            for key, value in dims.items():
                if key in seqs.columns:
                    seqs = seqs[seqs[key].eq(value)]
            base_n = len(frame)
            if base_n < 40:
                continue

            own_s = frame["own_prev_unsafe"].eq(0)
            own_u = frame["own_prev_unsafe"].eq(1)
            opp_s = frame["opponent_prev_unsafe"].eq(0)
            opp_u = frame["opponent_prev_unsafe"].eq(1)
            ahead = frame["gap_zone"].eq("ahead")
            behind = frame["gap_zone"].eq("behind")
            tied_or_ahead = frame["gap_zone"].isin(["tied", "ahead"])
            tied_or_behind = frame["gap_zone"].isin(["tied", "behind"])

            p_retaliate, n_retaliate = safe_rate(frame, own_s & opp_u)
            p_calm_after_safe, n_calm = safe_rate(frame, own_s & opp_s)
            p_opportunistic, n_oppatk = safe_rate(frame, ahead & opp_s)
            p_not_ahead, n_not_ahead = safe_rate(frame, tied_or_behind & opp_s)
            p_catchup, n_catch = safe_rate(frame, behind & opp_s)
            p_not_behind, n_not_behind = safe_rate(frame, tied_or_ahead & opp_s)
            p_forgive_s, n_forgive = safe_rate(frame.assign(safe=1 - frame["unsafe"]), own_u & opp_s, col="safe")
            p_mutual_stick, n_stick = safe_rate(frame, own_u & opp_u)

            rows.append(
                {
                    "scope": scope,
                    **dims,
                    "turns_round2plus": base_n,
                    "player_sequences": len(seqs),
                    "unsafe_rate": float(frame["unsafe"].mean()),
                    "opening_unsafe_rate": float(seqs["opens_unsafe"].mean()) if len(seqs) else np.nan,
                    "ever_switch_rate": float(seqs["ever_switches"].mean()) if len(seqs) else np.nan,
                    "mean_switch_rate": float(seqs["switch_rate"].mean()) if len(seqs) else np.nan,
                    "retaliation_rate": p_retaliate,
                    "calm_after_mutual_safe_rate": p_calm_after_safe,
                    "retaliation_lift": p_retaliate - p_calm_after_safe
                    if pd.notna(p_retaliate) and pd.notna(p_calm_after_safe)
                    else np.nan,
                    "n_retaliation_state": n_retaliate,
                    "n_mutual_safe_state": n_calm,
                    "opportunistic_rate": p_opportunistic,
                    "not_ahead_rate": p_not_ahead,
                    "opportunistic_lift": p_opportunistic - p_not_ahead
                    if pd.notna(p_opportunistic) and pd.notna(p_not_ahead)
                    else np.nan,
                    "n_opportunistic_state": n_oppatk,
                    "catchup_rate": p_catchup,
                    "not_behind_rate": p_not_behind,
                    "catchup_lift": p_catchup - p_not_behind
                    if pd.notna(p_catchup) and pd.notna(p_not_behind)
                    else np.nan,
                    "n_catchup_state": n_catch,
                    "forgiveness_rate": p_forgive_s,
                    "n_forgiveness_state": n_forgive,
                    "mutual_unsafe_stickiness": p_mutual_stick,
                    "n_mutual_unsafe_state": n_stick,
                }
            )
    return pd.DataFrame(rows)


def mine_response_matrix(turns: pd.DataFrame) -> pd.DataFrame:
    round2 = turns[turns["round"].ge(2) & turns["prev_pair"].isin(["SS", "SU", "US", "UU"])].copy()
    return (
        round2.groupby(["model_slug", "prev_pair", "gap_zone"], observed=True)
        .agg(n=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )


def mine_motifs(players: pd.DataFrame) -> pd.DataFrame:
    return (
        players.groupby(["model_slug", "play_motif"], observed=True)
        .agg(player_sequences=("sequence_id", "size"), unsafe_rate=("unsafe_rate", "mean"), switch_rate=("switch_rate", "mean"))
        .reset_index()
    )


def classify_dyad(group: pd.DataFrame) -> str:
    if len(group) != 2:
        return "mixed"
    motifs = set(group["play_motif"])
    rates = group["unsafe_rate"].astype(float).to_numpy()
    patterns = set(group["action_pattern"])
    if motifs == {"all_safe"}:
        return "mutual_safe"
    if motifs == {"all_unsafe"}:
        return "mutual_unsafe"
    if rates.max() - rates.min() >= 0.6:
        return "asymmetric_exploitation"
    if motifs.issubset({"late_escalation", "safe_then_attack"}) and rates.mean() >= 0.45:
        return "mutual_escalation"
    if motifs.issubset({"cooldown", "probe_then_cool"}) and rates.mean() <= 0.45:
        return "mutual_deescalation"
    if "alternating" in motifs or len(patterns) == 1 and any("SU" in pat or "US" in pat for pat in patterns):
        return "alternating_collision"
    return "mixed"


def mine_dyads(players: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    keys = ["source_run", "game_id"]
    for (source_run, game_id), group in players.groupby(keys, dropna=False):
        first = group.iloc[0]
        rows.append(
            {
                "source_run": source_run,
                "game_id": game_id,
                "model_slug": first["model_slug"],
                "family": first["family"],
                "dyad_style": classify_dyad(group),
                "mean_unsafe_rate": float(group["unsafe_rate"].mean()),
                "abs_unsafe_gap_between_players": float(group["unsafe_rate"].max() - group["unsafe_rate"].min()),
                "same_pattern": bool(group["action_pattern"].nunique() == 1),
                "player_patterns": " / ".join(group["action_pattern"].astype(str).tolist()),
            }
        )
    return pd.DataFrame(rows)


def plot_levers(levers: pd.DataFrame) -> Path:
    model = levers[levers["scope"].eq("model")].copy()
    model["model_label"] = model["model_slug"].map(MODEL_LABELS).fillna(model["model_slug"])
    order = [MODEL_LABELS[m] for m in MODEL_ORDER if m in set(model["model_slug"])]
    model = model.set_index("model_label").loc[order].reset_index()
    metrics = [
        ("retaliation_lift", "Retaliation\nopp U after own S"),
        ("opportunistic_lift", "Ahead attack\nopp was S"),
        ("catchup_lift", "Catch-up attack\nbehind, opp was S"),
        ("forgiveness_rate", "Forgiveness\nown U, opp S -> S"),
        ("mutual_unsafe_stickiness", "Stickiness\nafter UU"),
    ]
    data = model[[metric for metric, _ in metrics]].to_numpy(dtype=float)

    fig, ax = plt.subplots(figsize=(12.5, 6.5))
    vmax = np.nanmax(np.abs(data[:, :3]))
    im = ax.imshow(data, aspect="auto", cmap="PuOr_r", vmin=-max(vmax, 0.5), vmax=max(vmax, 0.5))
    ax.set_yticks(np.arange(len(model)))
    ax.set_yticklabels(model["model_label"])
    ax.set_xticks(np.arange(len(metrics)))
    ax.set_xticklabels([label for _, label in metrics], fontsize=9)
    ax.set_title("Strategic Levers by Model")
    ax.text(
        0,
        1.04,
        "Lift metrics compare conditional unsafe rates; forgiveness is P(safe) after own unsafe while opponent was safe.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, f"{data[i, j]:+.0%}" if j < 3 else f"{data[i, j]:.0%}", ha="center", va="center", fontsize=9, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Rate or lift")
    fig.subplots_adjust(left=0.23, right=0.95, top=0.85, bottom=0.18)
    return savefig(fig, "01_strategic_levers_by_model.png")


def plot_response_matrix(response: pd.DataFrame) -> Path:
    overall = (
        response.groupby(["prev_pair", "gap_zone"], observed=True)
        .apply(lambda g: pd.Series({"n": g["n"].sum(), "unsafe_rate": np.average(g["unsafe_rate"], weights=g["n"])}))
        .reset_index()
    )
    row_order = ["SS", "SU", "US", "UU"]
    col_order = ["behind", "tied", "ahead"]
    matrix = overall.pivot(index="prev_pair", columns="gap_zone", values="unsafe_rate").reindex(row_order)[col_order]
    counts = overall.pivot(index="prev_pair", columns="gap_zone", values="n").reindex(row_order)[col_order]

    fig, ax = plt.subplots(figsize=(8.5, 6.4))
    im = ax.imshow(matrix.to_numpy(), cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(col_order)))
    ax.set_xticklabels(["Behind", "Tied", "Ahead"])
    ax.set_yticks(np.arange(len(row_order)))
    ax.set_yticklabels(["prev SS", "prev SU", "prev US", "prev UU"])
    ax.set_title("Unsafe Rate by Previous State and Race Position")
    ax.text(
        0,
        1.05,
        "Rows are previous own/opponent actions from the player's view; U=unsafe, S=safe.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = matrix.iloc[i, j]
            n = counts.iloc[i, j]
            ax.text(j, i, f"{value:.0%}\nn={int(n)}", ha="center", va="center", fontsize=10, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("P(unsafe)")
    fig.subplots_adjust(left=0.18, right=0.94, top=0.84, bottom=0.12)
    return savefig(fig, "02_prev_state_gap_response_matrix.png")


def plot_motif_mix(motifs: pd.DataFrame) -> Path:
    totals = motifs.groupby("model_slug")["player_sequences"].transform("sum")
    motifs = motifs.assign(share=motifs["player_sequences"] / totals)
    plot = motifs.pivot(index="model_slug", columns="play_motif", values="share").fillna(0)
    order = [m for m in MODEL_ORDER if m in plot.index] + [m for m in plot.index if m not in MODEL_ORDER]
    plot = plot.loc[order]
    cols = [m for m in MOTIF_ORDER if m in plot.columns] + [m for m in plot.columns if m not in MOTIF_ORDER]

    fig, ax = plt.subplots(figsize=(12.5, 7))
    left = np.zeros(len(plot))
    y = np.arange(len(plot))
    for motif in cols:
        values = plot[motif].to_numpy()
        ax.barh(y, values, left=left, color=MOTIF_COLORS.get(motif, MUTED), edgecolor=WHITE, linewidth=0.8, label=motif)
        for i, value in enumerate(values):
            if value >= 0.08:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color=WHITE)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in plot.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of player trajectories")
    ax.set_title("Action Motif Mix by Model")
    ax.text(
        0,
        1.04,
        "Motifs summarize the full sequence, independent of canonical AS/AU/CS/CAS labels.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28), fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.27)
    return savefig(fig, "03_action_motif_mix_by_model.png")


def plot_dyads(dyads: pd.DataFrame) -> Path:
    counts = dyads.groupby(["model_slug", "dyad_style"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby("model_slug")["n"].transform("sum")
    plot = counts.pivot(index="model_slug", columns="dyad_style", values="share").fillna(0)
    order = [m for m in MODEL_ORDER if m in plot.index] + [m for m in plot.index if m not in MODEL_ORDER]
    plot = plot.loc[order]
    cols = [c for c in DYAD_COLORS if c in plot.columns] + [c for c in plot.columns if c not in DYAD_COLORS]

    fig, ax = plt.subplots(figsize=(12.5, 6.7))
    left = np.zeros(len(plot))
    y = np.arange(len(plot))
    for style in cols:
        values = plot[style].to_numpy()
        ax.barh(y, values, left=left, color=DYAD_COLORS.get(style, MUTED), edgecolor=WHITE, linewidth=0.8, label=style)
        for i, value in enumerate(values):
            if value >= 0.08:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", fontsize=8, color=WHITE)
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in plot.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of games")
    ax.set_title("Dyad-Level Play Styles by Model")
    ax.text(
        0,
        1.04,
        "Classified from both players' full action sequences within the same game.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    ax.legend(ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.28), fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.27)
    return savefig(fig, "04_dyad_play_styles_by_model.png")


def plot_round_switching(turns: pd.DataFrame) -> Path:
    round2 = turns[turns["round"].between(2, 12)].copy()
    summary = (
        round2.groupby(["round", "model_slug"], observed=True)
        .agg(
            switch_to_unsafe=("switch_to_unsafe", "mean"),
            switch_to_safe=("switch_to_safe", "mean"),
            n=("unsafe", "size"),
        )
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharey=True)
    for ax, metric, title in [
        (axes[0], "switch_to_unsafe", "Switch Into Unsafe"),
        (axes[1], "switch_to_safe", "Switch Back To Safe"),
    ]:
        for model in MODEL_ORDER:
            frame = summary[summary["model_slug"].eq(model)]
            if frame.empty:
                continue
            ax.plot(
                frame["round"],
                frame[metric],
                marker="o",
                linewidth=2,
                color=MODEL_COLORS.get(model, BLUE),
                label=MODEL_LABELS.get(model, model),
            )
        ax.set_title(title)
        ax.set_xlabel("Round")
        ax.set_ylim(0, max(0.8, summary[["switch_to_unsafe", "switch_to_safe"]].max().max() * 1.12))
        ax.yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    axes[0].set_ylabel("Share of round-2+ turns")
    axes[1].legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, fontsize=8)
    fig.suptitle("Timing of Strategic Switching", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.84, bottom=0.25, wspace=0.12)
    return savefig(fig, "05_round_switching_profiles.png")


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
    fig.suptitle("Strategy Playbook Mining Storyboard", fontsize=22, y=0.995)
    fig.subplots_adjust(top=0.965, hspace=0.08, wspace=0.04)
    path = PLAYBOOK_FIG_DIR / "fh_strategy_playbook_storyboard_contact_sheet.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def fmt(value: float, signed: bool = False) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{value:+.1%}" if signed else f"{value:.1%}"


def write_report(
    levers: pd.DataFrame,
    motifs: pd.DataFrame,
    dyads: pd.DataFrame,
    response: pd.DataFrame,
    figures: list[Path],
    contact_sheet: Path,
) -> Path:
    overall = levers[levers["scope"].eq("overall")].iloc[0]
    model = levers[levers["scope"].eq("model")].copy()
    motif_overall = (
        motifs.groupby("play_motif")
        .agg(player_sequences=("player_sequences", "sum"))
        .assign(share=lambda d: d["player_sequences"] / d["player_sequences"].sum())
        .sort_values("share", ascending=False)
    )
    dyad_overall = (
        dyads.groupby("dyad_style")
        .size()
        .rename("games")
        .to_frame()
        .assign(share=lambda d: d["games"] / d["games"].sum())
        .sort_values("share", ascending=False)
    )

    model_lines = []
    for _, row in model.sort_values("retaliation_lift", ascending=False).iterrows():
        model_lines.append(
            f"- {MODEL_LABELS.get(row['model_slug'], row['model_slug'])}: retaliation lift {fmt(row['retaliation_lift'], True)}, "
            f"opportunistic lift {fmt(row['opportunistic_lift'], True)}, catch-up lift {fmt(row['catchup_lift'], True)}, "
            f"forgiveness {fmt(row['forgiveness_rate'])}, UU stickiness {fmt(row['mutual_unsafe_stickiness'])}."
        )

    motif_lines = [
        f"- `{idx}`: {fmt(row['share'])} of player trajectories ({int(row['player_sequences'])})."
        for idx, row in motif_overall.head(8).iterrows()
    ]
    dyad_lines = [
        f"- `{idx}`: {fmt(row['share'])} of games ({int(row['games'])})."
        for idx, row in dyad_overall.head(7).iterrows()
    ]

    body = f"""# Strategy Playbook Mining

## New Strategic Levers

Across all completed non-duplicate turns, the mined playbook says:

- Retaliation lift: {fmt(overall['retaliation_lift'], True)}. This is P(unsafe | own previous safe, opponent previous unsafe) minus P(unsafe | both previously safe).
- Opportunistic lift: {fmt(overall['opportunistic_lift'], True)}. This is extra unsafe when ahead and opponent was previously safe.
- Catch-up lift: {fmt(overall['catchup_lift'], True)}. This is extra unsafe when behind and opponent was previously safe.
- Forgiveness rate: {fmt(overall['forgiveness_rate'])}. This is P(safe | own previous unsafe, opponent previous safe).
- Mutual-unsafe stickiness: {fmt(overall['mutual_unsafe_stickiness'])}. This is P(unsafe | both previously unsafe).

## Model Playbook

{chr(10).join(model_lines)}

## Sequence Motifs

{chr(10).join(motif_lines)}

## Dyad-Level Styles

{chr(10).join(dyad_lines)}

## Interpretation

The extra strategic signal is not just AU/AS/CS/CAS mismatch. The data shows three reusable moves: retaliation after being exploited, opportunistic attack when already ahead, and cooldown/forgiveness after unilateral unsafe. Different models combine those moves with different baseline aggression.

## Deliverables

- Contact sheet: `{contact_sheet}`
- Figures: {", ".join(f"`{path.name}`" for path in figures)}
- Tables: `strategy_playbook_levers.csv`, `strategy_playbook_response_matrix.csv`, `strategy_playbook_motifs.csv`, `strategy_playbook_dyads.csv`
"""
    path = REPORTS_DIR / "fh_strategy_playbook_mining.md"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    configure_matplotlib()
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv", low_memory=False)
    turns = turns[turns["manifest_status"].eq("completed") & ~turns["duplicate_grain_key"].fillna(False).astype(bool)].copy()
    players = pd.read_csv(DERIVED_DIR / "strategy_residual_player_classification.csv", low_memory=False)

    for frame in [turns, players]:
        for col in ["unsafe", "round", "player_index", "progress_gap_before", "own_prev_unsafe", "opponent_prev_unsafe"]:
            if col in frame.columns:
                frame[col] = pd.to_numeric(frame[col], errors="coerce")

    turns, players = add_strategy_context(turns, players)
    levers = mine_levers(turns, players)
    response = mine_response_matrix(turns)
    motifs = mine_motifs(players)
    dyads = mine_dyads(players)

    levers.to_csv(DERIVED_DIR / "strategy_playbook_levers.csv", index=False)
    response.to_csv(DERIVED_DIR / "strategy_playbook_response_matrix.csv", index=False)
    motifs.to_csv(DERIVED_DIR / "strategy_playbook_motifs.csv", index=False)
    dyads.to_csv(DERIVED_DIR / "strategy_playbook_dyads.csv", index=False)

    figures = [
        plot_levers(levers),
        plot_response_matrix(response),
        plot_motif_mix(motifs),
        plot_dyads(dyads),
        plot_round_switching(turns),
    ]
    contact_sheet = make_contact_sheet(figures)
    report = write_report(levers, motifs, dyads, response, figures, contact_sheet)
    overall = levers[levers["scope"].eq("overall")].iloc[0]
    print(
        json.dumps(
            {
                "turns": int(len(turns)),
                "player_sequences": int(len(players)),
                "games": int(len(dyads)),
                "retaliation_lift": float(overall["retaliation_lift"]),
                "opportunistic_lift": float(overall["opportunistic_lift"]),
                "catchup_lift": float(overall["catchup_lift"]),
                "forgiveness_rate": float(overall["forgiveness_rate"]),
                "mutual_unsafe_stickiness": float(overall["mutual_unsafe_stickiness"]),
                "report": str(report),
                "contact_sheet": str(contact_sheet),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
