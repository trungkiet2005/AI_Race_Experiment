#!/usr/bin/env python3
"""Deep-dive on first unsafe timing, escalation cascades, and strategy fitness."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
DEEP_FIG_DIR = FIGURES_DIR / "strategy_mechanism_deep_dive"
REPORTS_DIR = OUTPUT_DIR / "reports"
RANDOM_SEED = 260726

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
MACRO_COLORS = {
    "stable_safe": BLUE,
    "probe_cooldown": GOLD,
    "stable_unsafe": PINK,
    "catchup_attack": ORANGE,
    "mixed_adaptive": TEAL,
}
MACRO_LABELS = {
    "stable_safe": "Stable safe",
    "probe_cooldown": "Probe/cooldown",
    "stable_unsafe": "Stable unsafe",
    "catchup_attack": "Catch-up attack",
    "mixed_adaptive": "Mixed adaptive",
}


def ensure_dirs() -> None:
    for path in [DERIVED_DIR, DEEP_FIG_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def setup_style() -> None:
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
    path = DEEP_FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100 * value:.1f}%"


def strategy_key(row: pd.Series) -> str:
    return f"{row['source_run']}|{row['game_id']}|p{int(row['player_index'])}"


def cluster_macro(cluster: int) -> str:
    if cluster in {1, 5}:
        return "stable_safe"
    if cluster in {0, 4}:
        return "probe_cooldown"
    if cluster in {2, 3, 6}:
        return "stable_unsafe"
    if cluster == 7:
        return "catchup_attack"
    return "mixed_adaptive"


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv", low_memory=False)
    turns = turns[turns["manifest_status"].eq("completed") & ~turns["duplicate_grain_key"].fillna(False).astype(bool)].copy()
    turns = turns[turns["unsafe"].notna()].copy()
    for col in [
        "unsafe",
        "round",
        "player_index",
        "progress_gap_before",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
        "own_progress_after",
        "opponent_progress_after",
        "progress_gap_after",
        "cumulative_stage_payoff_after",
        "round_payoff",
    ]:
        if col in turns.columns:
            turns[col] = pd.to_numeric(turns[col], errors="coerce")
    turns["sequence_id"] = turns.apply(strategy_key, axis=1)
    turns["gap_zone"] = np.select(
        [turns["progress_gap_before"] < -0.5, turns["progress_gap_before"] > 0.5],
        ["behind", "ahead"],
        default="tied",
    )
    players = pd.read_csv(DERIVED_DIR / "strategy_synthesis_player_embeddings.csv", low_memory=False)
    players["cluster"] = pd.to_numeric(players["cluster"], errors="coerce").astype(int)
    players["strategy_macro"] = players["cluster"].map(cluster_macro)
    players["strategy_macro_label"] = players["strategy_macro"].map(MACRO_LABELS)
    keep = [
        "sequence_id",
        "cluster",
        "strategy_macro",
        "strategy_macro_label",
        "expanded_best_strategy",
        "expanded_best_family",
        "expanded_exact_any",
        "new_exact_beyond_canonical",
    ]
    turns = turns.merge(players[keep], on="sequence_id", how="left")
    return turns, players


def player_outcomes(turns: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sequence_id, frame in turns.sort_values(["sequence_id", "round"]).groupby("sequence_id", sort=False):
        frame = frame.sort_values("round")
        y = frame["unsafe"].astype(int)
        first = frame.iloc[0]
        last = frame.iloc[-1]
        unsafe_rounds = frame.loc[y.eq(1), "round"]
        rows.append(
            {
                "sequence_id": sequence_id,
                "source_run": first["source_run"],
                "game_id": first["game_id"],
                "player_index": int(first["player_index"]),
                "model_slug": first["model_slug"],
                "family": first["family"],
                "strategy_macro": first["strategy_macro"],
                "strategy_macro_label": first["strategy_macro_label"],
                "expanded_best_strategy": first.get("expanded_best_strategy"),
                "n_turns": len(frame),
                "unsafe_rate": float(y.mean()),
                "first_unsafe_round": int(unsafe_rounds.min()) if len(unsafe_rounds) else np.nan,
                "ever_unsafe": bool(len(unsafe_rounds)),
                "final_progress": float(last["own_progress_after"]),
                "final_progress_gap": float(last["progress_gap_after"]),
                "final_stage_payoff": float(last["cumulative_stage_payoff_after"]),
            }
        )
    out = pd.DataFrame(rows)
    opp = out[["source_run", "game_id", "player_index", "final_progress", "final_stage_payoff", "strategy_macro"]].copy()
    opp["player_index"] = 1 - opp["player_index"].astype(int)
    opp = opp.rename(
        columns={
            "final_progress": "opponent_final_progress",
            "final_stage_payoff": "opponent_final_stage_payoff",
            "strategy_macro": "opponent_strategy_macro",
        }
    )
    out = out.merge(opp, on=["source_run", "game_id", "player_index"], how="left")
    out["progress_advantage"] = out["final_progress"] - out["opponent_final_progress"]
    out["payoff_advantage"] = out["final_stage_payoff"] - out["opponent_final_stage_payoff"]
    out["win_progress"] = out["progress_advantage"].gt(0)
    out["tie_progress"] = out["progress_advantage"].eq(0)
    return out


def build_hazard_rows(turns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sequence_id, frame in turns.sort_values(["sequence_id", "round"]).groupby("sequence_id", sort=False):
        seen_unsafe = False
        for _, row in frame.sort_values("round").iterrows():
            if seen_unsafe:
                break
            event = int(row["unsafe"] == 1)
            rows.append(
                {
                    "sequence_id": sequence_id,
                    "model_slug": row["model_slug"],
                    "family": row["family"],
                    "strategy_macro": row["strategy_macro"],
                    "strategy_macro_label": row["strategy_macro_label"],
                    "round": int(row["round"]),
                    "gap": float(row["progress_gap_before"]),
                    "gap_zone": row["gap_zone"],
                    "opponent_prev_unsafe": 0.0 if pd.isna(row["opponent_prev_unsafe"]) else float(row["opponent_prev_unsafe"]),
                    "event_first_unsafe": event,
                }
            )
            if event:
                seen_unsafe = True
    return pd.DataFrame(rows)


def km_survival(hazard: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for group, frame in hazard.groupby(group_col, dropna=False):
        survival = 1.0
        for round_number in sorted(frame["round"].unique()):
            risk = frame[frame["round"].eq(round_number)]
            n_risk = len(risk)
            events = int(risk["event_first_unsafe"].sum())
            hazard_rate = events / n_risk if n_risk else np.nan
            survival *= 1 - hazard_rate if n_risk else 1
            rows.append(
                {
                    group_col: group,
                    "round": round_number,
                    "n_at_risk": n_risk,
                    "events": events,
                    "hazard": hazard_rate,
                    "survival_no_unsafe": survival,
                }
            )
    return pd.DataFrame(rows)


def hazard_summaries(hazard: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_state = (
        hazard.groupby(["gap_zone", "opponent_prev_unsafe"], observed=True)
        .agg(n=("event_first_unsafe", "size"), first_unsafe_hazard=("event_first_unsafe", "mean"))
        .reset_index()
    )
    model = (
        hazard.groupby("model_slug", observed=True)
        .agg(n=("event_first_unsafe", "size"), first_unsafe_hazard=("event_first_unsafe", "mean"))
        .reset_index()
    )
    x = pd.get_dummies(
        hazard[["round", "gap", "opponent_prev_unsafe", "model_slug"]].fillna(0),
        columns=["model_slug"],
        drop_first=True,
        dtype=float,
    )
    y = hazard["event_first_unsafe"].astype(int).to_numpy()
    clf = LogisticRegression(C=1e6, max_iter=5000, solver="lbfgs")
    clf.fit(x, y)
    coefs = pd.DataFrame(
        {
            "feature": x.columns,
            "coef": clf.coef_[0],
            "odds_ratio": np.exp(clf.coef_[0]),
        }
    ).sort_values("coef", ascending=False)
    return pd.concat([by_state.assign(scope="state"), model.assign(scope="model")], ignore_index=True), coefs


def build_dyad_rounds(turns: pd.DataFrame) -> pd.DataFrame:
    base_cols = ["source_run", "game_id", "model_slug", "family", "round"]
    pivot = (
        turns.pivot_table(index=base_cols, columns="player_index", values="unsafe", aggfunc="first")
        .reset_index()
        .rename(columns={0: "p0_unsafe", 1: "p1_unsafe"})
    )
    pivot = pivot.dropna(subset=["p0_unsafe", "p1_unsafe"]).copy()
    pivot["joint_state"] = np.where(
        pivot["p0_unsafe"].eq(0) & pivot["p1_unsafe"].eq(0),
        "SS",
        np.where(
            pivot["p0_unsafe"].eq(1) & pivot["p1_unsafe"].eq(1),
            "UU",
            np.where(pivot["p0_unsafe"].eq(1), "US", "SU"),
        ),
    )
    pivot["joint_macro"] = np.where(pivot["joint_state"].eq("SS"), "mutual_safe", np.where(pivot["joint_state"].eq("UU"), "mutual_unsafe", "unilateral_unsafe"))
    return pivot.sort_values(["source_run", "game_id", "round"])


def cascade_analysis(dyad: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    transitions: list[dict[str, Any]] = []
    game_rows: list[dict[str, Any]] = []
    for (source_run, game_id), frame in dyad.groupby(["source_run", "game_id"], sort=False):
        frame = frame.sort_values("round")
        states = frame["joint_state"].tolist()
        macros = frame["joint_macro"].tolist()
        first = frame.iloc[0]
        for i in range(len(frame) - 1):
            transitions.append(
                {
                    "source_run": source_run,
                    "game_id": game_id,
                    "model_slug": first["model_slug"],
                    "family": first["family"],
                    "round": int(frame.iloc[i]["round"]),
                    "from_state": states[i],
                    "to_state": states[i + 1],
                    "from_macro": macros[i],
                    "to_macro": macros[i + 1],
                }
            )
        has_ss = "SS" in states
        has_unilateral = any(state in {"SU", "US"} for state in states)
        has_uu = "UU" in states
        ss_to_unilateral_to_uu = False
        for i, state in enumerate(states):
            if state == "SS" and any(s in {"SU", "US"} for s in states[i + 1 :]):
                j = next((idx for idx in range(i + 1, len(states)) if states[idx] in {"SU", "US"}), None)
                if j is not None and "UU" in states[j + 1 :]:
                    ss_to_unilateral_to_uu = True
                    break
        game_rows.append(
            {
                "source_run": source_run,
                "game_id": game_id,
                "model_slug": first["model_slug"],
                "family": first["family"],
                "has_ss": has_ss,
                "has_unilateral": has_unilateral,
                "has_uu": has_uu,
                "ss_to_unilateral_to_uu": ss_to_unilateral_to_uu,
                "start_state": states[0],
                "end_state": states[-1],
            }
        )
    trans = pd.DataFrame(transitions)
    matrix = (
        trans.groupby(["from_macro", "to_macro"], observed=True)
        .size()
        .rename("n")
        .reset_index()
    )
    matrix["transition_rate"] = matrix["n"] / matrix.groupby("from_macro")["n"].transform("sum")
    metrics = (
        trans.groupby("model_slug", observed=True)
        .apply(
            lambda g: pd.Series(
                {
                    "ss_to_unilateral": ((g["from_macro"].eq("mutual_safe")) & (g["to_macro"].eq("unilateral_unsafe"))).sum()
                    / max(1, g["from_macro"].eq("mutual_safe").sum()),
                    "unilateral_to_uu": ((g["from_macro"].eq("unilateral_unsafe")) & (g["to_macro"].eq("mutual_unsafe"))).sum()
                    / max(1, g["from_macro"].eq("unilateral_unsafe").sum()),
                    "unilateral_to_ss": ((g["from_macro"].eq("unilateral_unsafe")) & (g["to_macro"].eq("mutual_safe"))).sum()
                    / max(1, g["from_macro"].eq("unilateral_unsafe").sum()),
                    "uu_stays_uu": ((g["from_macro"].eq("mutual_unsafe")) & (g["to_macro"].eq("mutual_unsafe"))).sum()
                    / max(1, g["from_macro"].eq("mutual_unsafe").sum()),
                }
            )
        )
        .reset_index()
    )
    games = pd.DataFrame(game_rows)
    game_metrics = (
        games.groupby("model_slug", observed=True)
        .agg(
            games=("game_id", "size"),
            any_unilateral=("has_unilateral", "mean"),
            any_uu=("has_uu", "mean"),
            ss_to_unilateral_to_uu=("ss_to_unilateral_to_uu", "mean"),
        )
        .reset_index()
    )
    metrics = metrics.merge(game_metrics, on="model_slug", how="left")
    return matrix, metrics, games


def fitness_summary(outcomes: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    by_macro = (
        outcomes.groupby("strategy_macro", observed=True)
        .agg(
            player_sequences=("sequence_id", "size"),
            unsafe_rate=("unsafe_rate", "mean"),
            final_progress=("final_progress", "mean"),
            final_stage_payoff=("final_stage_payoff", "mean"),
            progress_advantage=("progress_advantage", "mean"),
            payoff_advantage=("payoff_advantage", "mean"),
            win_rate=("win_progress", "mean"),
            tie_rate=("tie_progress", "mean"),
        )
        .reset_index()
    )
    by_strategy = (
        outcomes.groupby("expanded_best_strategy", observed=True)
        .agg(
            player_sequences=("sequence_id", "size"),
            unsafe_rate=("unsafe_rate", "mean"),
            final_progress=("final_progress", "mean"),
            final_stage_payoff=("final_stage_payoff", "mean"),
            progress_advantage=("progress_advantage", "mean"),
            payoff_advantage=("payoff_advantage", "mean"),
            win_rate=("win_progress", "mean"),
        )
        .reset_index()
        .sort_values("player_sequences", ascending=False)
    )
    return by_macro, by_strategy


def plot_survival(km: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    for macro in ["stable_safe", "probe_cooldown", "stable_unsafe", "catchup_attack", "mixed_adaptive"]:
        label = MACRO_LABELS[macro]
        frame = km[km["strategy_macro"].eq(macro)]
        if frame.empty:
            continue
        ax.step(frame["round"], frame["survival_no_unsafe"], where="post", linewidth=2.4, color=MACRO_COLORS[macro], label=label)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Round")
    ax.set_ylabel("P(no unsafe yet)")
    ax.set_title("Survival Until First Unsafe Action")
    ax.text(0, 1.04, "Kaplan-Meier style survival by discovered strategy macro.", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.legend(loc="upper right")
    fig.subplots_adjust(left=0.1, right=0.98, top=0.84, bottom=0.13)
    return savefig(fig, "01_first_unsafe_survival.png")


def plot_hazard_state(hazard_summary: pd.DataFrame) -> Path:
    state = hazard_summary[hazard_summary["scope"].eq("state")].copy()
    state["label"] = state["gap_zone"].str.title() + " | opp_prevU=" + state["opponent_prev_unsafe"].astype(int).astype(str)
    state = state.sort_values("first_unsafe_hazard")
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    bars = ax.barh(np.arange(len(state)), state["first_unsafe_hazard"], color=ORANGE, edgecolor=WHITE, linewidth=0.8)
    ax.set_yticks(np.arange(len(state)))
    ax.set_yticklabels(state["label"])
    ax.set_xlabel("First-unsafe hazard")
    ax.set_title("First Unsafe Trigger by Position and Opponent History")
    for bar, (_, row) in zip(bars, state.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{pct(row['first_unsafe_hazard'])}, n={int(row['n'])}", va="center", fontsize=9)
    ax.set_xlim(0, min(1, state["first_unsafe_hazard"].max() * 1.25))
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.28, right=0.93, top=0.86, bottom=0.13)
    return savefig(fig, "02_first_unsafe_hazard_by_state.png")


def plot_cascade_matrix(matrix: pd.DataFrame) -> Path:
    rows = ["mutual_safe", "unilateral_unsafe", "mutual_unsafe"]
    cols = rows
    mat = matrix.pivot(index="from_macro", columns="to_macro", values="transition_rate").reindex(rows)[cols].fillna(0)
    fig, ax = plt.subplots(figsize=(8.8, 6.5))
    im = ax.imshow(mat.to_numpy(), cmap="YlOrBr", vmin=0, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(["To SS", "To one-U", "To UU"])
    ax.set_yticks(np.arange(len(rows)))
    ax.set_yticklabels(["From SS", "From one-U", "From UU"])
    ax.set_title("Dyad Escalation Transition Matrix")
    ax.text(0, 1.05, "Joint states are compressed to mutual safe, unilateral unsafe, and mutual unsafe.", transform=ax.transAxes, color=MUTED, fontsize=10)
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            ax.text(j, i, pct(mat.iloc[i, j]), ha="center", va="center", fontsize=11, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cbar.set_label("Transition probability")
    fig.subplots_adjust(left=0.17, right=0.94, top=0.84, bottom=0.12)
    return savefig(fig, "03_cascade_transition_matrix.png")


def plot_cascade_by_model(metrics: pd.DataFrame) -> Path:
    metrics = metrics.copy()
    order = [m for m in MODEL_ORDER if m in set(metrics["model_slug"])]
    metrics["model_label"] = metrics["model_slug"].map(MODEL_LABELS)
    metrics = metrics.set_index("model_slug").loc[order].reset_index()
    cols = ["ss_to_unilateral", "unilateral_to_uu", "unilateral_to_ss", "uu_stays_uu", "ss_to_unilateral_to_uu"]
    labels = ["SS->one-U", "one-U->UU", "one-U->SS", "UU->UU", "cascade path"]
    data = metrics[cols].to_numpy()
    fig, ax = plt.subplots(figsize=(12, 6.5))
    im = ax.imshow(data, aspect="auto", cmap="YlGnBu", vmin=0, vmax=1)
    ax.set_yticks(np.arange(len(metrics)))
    ax.set_yticklabels(metrics["model_label"])
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_title("Escalation Cascade Metrics by Model")
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, pct(data[i, j]), ha="center", va="center", fontsize=9, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Probability")
    fig.subplots_adjust(left=0.22, right=0.95, top=0.86, bottom=0.16)
    return savefig(fig, "04_cascade_by_model.png")


def plot_fitness_macro(fitness: pd.DataFrame) -> Path:
    data = fitness.copy()
    data["label"] = data["strategy_macro"].map(MACRO_LABELS)
    fig, ax = plt.subplots(figsize=(9.8, 6.8))
    for _, row in data.iterrows():
        macro = row["strategy_macro"]
        ax.scatter(row["unsafe_rate"], row["win_rate"], s=max(120, row["player_sequences"] / 6), color=MACRO_COLORS.get(macro, MUTED), alpha=0.8, edgecolor=WHITE, linewidth=1)
        ax.text(row["unsafe_rate"] + 0.008, row["win_rate"], row["label"], va="center", fontsize=9)
    ax.axhline(0.5, color=GRID, linewidth=1.2)
    ax.set_xlabel("Mean unsafe rate")
    ax.set_ylabel("Progress win rate vs opponent")
    ax.set_title("Strategy Fitness: Unsafe Rate vs Win Rate")
    ax.text(0, 1.04, "Bubble size is number of player trajectories.", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.set_xlim(-0.02, min(1.02, data["unsafe_rate"].max() + 0.14))
    ax.set_ylim(0, min(1.02, data["win_rate"].max() + 0.18))
    fig.subplots_adjust(left=0.12, right=0.96, top=0.84, bottom=0.13)
    return savefig(fig, "05_strategy_fitness_frontier.png")


def plot_payoff_advantage(fitness: pd.DataFrame) -> Path:
    data = fitness.sort_values("payoff_advantage").copy()
    fig, ax = plt.subplots(figsize=(9.8, 6.2))
    colors = [MACRO_COLORS.get(m, MUTED) for m in data["strategy_macro"]]
    bars = ax.barh(np.arange(len(data)), data["payoff_advantage"], color=colors, edgecolor=WHITE, linewidth=0.8)
    ax.axvline(0, color=INK, linewidth=1)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(data["strategy_macro"].map(MACRO_LABELS))
    ax.set_xlabel("Mean payoff advantage vs opponent")
    ax.set_title("Payoff Advantage by Strategy Macro")
    for bar, (_, row) in zip(bars, data.iterrows()):
        x = bar.get_width()
        ax.text(x + (0.02 if x >= 0 else -0.02), bar.get_y() + bar.get_height() / 2, f"{x:+.2f}", va="center", ha="left" if x >= 0 else "right", fontsize=9)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.24, right=0.94, top=0.86, bottom=0.13)
    return savefig(fig, "06_payoff_advantage_by_strategy.png")


def make_contact_sheet(paths: list[Path]) -> Path:
    images = [mpimg.imread(path) for path in paths]
    fig, axes = plt.subplots(3, 2, figsize=(16, 19))
    axes = axes.ravel()
    for ax, image, path in zip(axes, images, paths):
        ax.imshow(image)
        ax.set_title(path.stem.replace("_", " ").title(), fontsize=11, pad=8)
        ax.axis("off")
    fig.suptitle("Strategy Mechanism Deep Dive", fontsize=22, y=0.995)
    fig.subplots_adjust(top=0.965, hspace=0.08, wspace=0.04)
    path = DEEP_FIG_DIR / "fh_strategy_mechanism_deep_dive_contact_sheet.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    outcomes: pd.DataFrame,
    km_macro: pd.DataFrame,
    hazard_summary: pd.DataFrame,
    hazard_coefs: pd.DataFrame,
    cascade_matrix: pd.DataFrame,
    cascade_metrics: pd.DataFrame,
    fitness_macro: pd.DataFrame,
    fitness_strategy: pd.DataFrame,
    figures: list[Path],
    contact_sheet: Path,
) -> Path:
    top_hazard = hazard_summary[hazard_summary["scope"].eq("state")].sort_values("first_unsafe_hazard", ascending=False)
    hazard_lines = [
        f"- {row['gap_zone']} / opponent_prev_unsafe={int(row['opponent_prev_unsafe'])}: first-unsafe hazard {pct(row['first_unsafe_hazard'])}, n={int(row['n'])}."
        for _, row in top_hazard.iterrows()
    ]
    cascade_lines = [
        f"- {MODEL_LABELS.get(row['model_slug'], row['model_slug'])}: SS->one-U {pct(row['ss_to_unilateral'])}, one-U->UU {pct(row['unilateral_to_uu'])}, UU->UU {pct(row['uu_stays_uu'])}, cascade path {pct(row['ss_to_unilateral_to_uu'])}."
        for _, row in cascade_metrics.iterrows()
    ]
    fitness_lines = [
        f"- {MACRO_LABELS.get(row['strategy_macro'], row['strategy_macro'])}: n={int(row['player_sequences'])}, unsafe {pct(row['unsafe_rate'])}, win {pct(row['win_rate'])}, payoff advantage {row['payoff_advantage']:+.2f}."
        for _, row in fitness_macro.sort_values("win_rate", ascending=False).iterrows()
    ]
    coef_lines = [
        f"- `{row['feature']}`: coef {row['coef']:+.2f}, odds ratio {row['odds_ratio']:.2f}."
        for _, row in hazard_coefs.head(8).iterrows()
    ]
    body = f"""# Strategy Mechanism Deep Dive

## Scope

This analysis extends the strategy synthesis with three mechanism tests: time to first unsafe action, dyad escalation cascades, and payoff/fitness by discovered strategy macro.

## First Unsafe Timing

{chr(10).join(hazard_lines)}

Top discrete hazard model coefficients:

{chr(10).join(coef_lines)}

## Escalation Cascades

{chr(10).join(cascade_lines)}

## Fitness

{chr(10).join(fitness_lines)}

## Interpretation

The strongest mechanism is not merely a fixed unsafe preference. Unsafe often enters through timing and dyad transitions: some strategies delay the first unsafe action, unilateral unsafe can either cool back to mutual safety or tip into mutual unsafe, and the payoff frontier separates stable-safe, probe/cooldown, catch-up, and unsafe-sticky regimes.

## Deliverables

- Contact sheet: `{contact_sheet}`
- Figures: {", ".join(f"`{path.name}`" for path in figures)}
- Tables: `strategy_deep_player_outcomes.csv`, `strategy_deep_first_unsafe_hazard_rows.csv`, `strategy_deep_first_unsafe_km.csv`, `strategy_deep_hazard_summary.csv`, `strategy_deep_hazard_coefficients.csv`, `strategy_deep_cascade_matrix.csv`, `strategy_deep_cascade_metrics.csv`, `strategy_deep_fitness_macro.csv`, `strategy_deep_fitness_strategy.csv`
"""
    path = REPORTS_DIR / "fh_strategy_mechanism_deep_dive.md"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    setup_style()
    turns, players = load_inputs()
    outcomes = player_outcomes(turns, players)
    hazard = build_hazard_rows(turns)
    km_macro = km_survival(hazard, "strategy_macro")
    hazard_summary, hazard_coefs = hazard_summaries(hazard)
    dyad = build_dyad_rounds(turns)
    cascade_matrix, cascade_metrics, cascade_games = cascade_analysis(dyad)
    fitness_macro, fitness_strategy = fitness_summary(outcomes)

    outcomes.to_csv(DERIVED_DIR / "strategy_deep_player_outcomes.csv", index=False)
    hazard.to_csv(DERIVED_DIR / "strategy_deep_first_unsafe_hazard_rows.csv", index=False)
    km_macro.to_csv(DERIVED_DIR / "strategy_deep_first_unsafe_km.csv", index=False)
    hazard_summary.to_csv(DERIVED_DIR / "strategy_deep_hazard_summary.csv", index=False)
    hazard_coefs.to_csv(DERIVED_DIR / "strategy_deep_hazard_coefficients.csv", index=False)
    cascade_matrix.to_csv(DERIVED_DIR / "strategy_deep_cascade_matrix.csv", index=False)
    cascade_metrics.to_csv(DERIVED_DIR / "strategy_deep_cascade_metrics.csv", index=False)
    cascade_games.to_csv(DERIVED_DIR / "strategy_deep_cascade_games.csv", index=False)
    fitness_macro.to_csv(DERIVED_DIR / "strategy_deep_fitness_macro.csv", index=False)
    fitness_strategy.to_csv(DERIVED_DIR / "strategy_deep_fitness_strategy.csv", index=False)

    figures = [
        plot_survival(km_macro),
        plot_hazard_state(hazard_summary),
        plot_cascade_matrix(cascade_matrix),
        plot_cascade_by_model(cascade_metrics),
        plot_fitness_macro(fitness_macro),
        plot_payoff_advantage(fitness_macro),
    ]
    contact_sheet = make_contact_sheet(figures)
    report = write_report(
        outcomes,
        km_macro,
        hazard_summary,
        hazard_coefs,
        cascade_matrix,
        cascade_metrics,
        fitness_macro,
        fitness_strategy,
        figures,
        contact_sheet,
    )
    print(
        json.dumps(
            {
                "player_sequences": int(len(outcomes)),
                "hazard_rows": int(len(hazard)),
                "games": int(len(cascade_games)),
                "top_first_unsafe_hazard": hazard_summary[hazard_summary["scope"].eq("state")]
                .sort_values("first_unsafe_hazard", ascending=False)
                .head(3)
                .to_dict(orient="records"),
                "fitness_macro": fitness_macro.to_dict(orient="records"),
                "report": str(report),
                "contact_sheet": str(contact_sheet),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
