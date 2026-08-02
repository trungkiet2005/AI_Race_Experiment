#!/usr/bin/env python3
"""Cluster human participants into behavioral archetypes, then ask which
archetype each LLM checkpoint's players land nearest to.

Five features per entity, mirroring the human study's own predictors:
overall Unsafe rate, reciprocity (own choice conditional on opponent's last
action), position sensitivity (own choice conditional on being ahead/behind),
own-action autocorrelation (persistence vs. alternation), and first-round
choice. Missing conditioning states (e.g. a participant who was never both
ahead and behind) are mean-imputed with the human population's own mean,
consistent with how the LLM points are later projected into the same space.

KMeans is fit once, on standardized human data only; LLM player-races are
standardized with the *human* mean/SD and assigned to the nearest human
centroid -- this asks "which human archetype does this look like," not "what
clusters exist in the pooled data."
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
HUMAN_CSV = ROOT / "public_dataset" / "airace_deidentified_long.csv"
OUTPUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUTPUT / "figures"
DATA = OUTPUT / "data"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "slate": "#64748B", "grid": "#DCE3ED",
}
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}
NEUTRAL_INPUTS = {
    "gpt-5-nano": ["results/frontier/openai/baseline/gpt-5-nano", "results/frontier/openai/persona/R0_neutral/gpt-5-nano"],
    "gpt-5.4-nano": ["results/frontier/openai/baseline/gpt-5.4-nano", "results/frontier/openai/persona/R0_neutral/gpt-5.4-nano"],
    "google/gemini-3-flash-preview": [
        "results/frontier/baseline/google-gemini-3-flash-preview",
        "results/frontier/persona/R0_neutral/google-gemini-3-flash-preview",
    ],
    "google/gemini-3.1-flash-lite-preview": ["results/frontier/baseline/google-gemini-3.1-flash-lite-preview"],
    "google/gemini-3.5-flash-lite": ["results/frontier/baseline/google-gemini-3.5-flash-lite"],
    "gpt-5.6-luna": ["results/frontier/bedrock_mantle/luna/baseline", "results/frontier/bedrock_mantle/luna/persona/R0_neutral"],
    "gpt-5.6-terra": ["results/frontier/bedrock_mantle/terra/baseline", "results/frontier/bedrock_mantle/terra/persona/R0_neutral"],
    "claude-opus-5": ["results/frontier/bedrock/baseline", "results/frontier/bedrock/persona/R0_neutral"],
    "claude-sonnet-5": ["results/frontier/bedrock/baseline", "results/frontier/bedrock/persona/R0_neutral"],
}
MODEL_SUBDIR = {"gpt-5.6-luna": "openai.gpt-5.6-luna", "gpt-5.6-terra": "openai.gpt-5.6-terra",
                "claude-opus-5": "us.anthropic.claude-opus-5", "claude-sonnet-5": "us.anthropic.claude-sonnet-5"}
FEATURES = ["overall_unsafe_rate", "reciprocity", "position_sensitivity", "own_autocorrelation", "first_round_unsafe"]
K = 4
SEED = 0


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.labelsize": 10, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _safe_diff(df: pd.DataFrame, col: str, cond_true, cond_false) -> float:
    a = df.loc[cond_true, col]
    b = df.loc[cond_false, col]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(a.mean() - b.mean())


def human_features() -> pd.DataFrame:
    df = pd.read_csv(HUMAN_CSV)
    rows = []
    for pid, g in df.groupby("participant_id"):
        later = g[g["round_number"] > 1]
        row = {
            "participant_id": pid,
            "overall_unsafe_rate": g["decision"].mean(),
            "reciprocity": _safe_diff(later, "decision", later["decision_opponent_lag"] == 1, later["decision_opponent_lag"] == 0),
            "position_sensitivity": _safe_diff(later, "decision", later["delta_steps_lag"] > 0, later["delta_steps_lag"] < 0),
            "own_autocorrelation": _safe_diff(later, "decision", later["decision_lag"] == 1, later["decision_lag"] == 0),
            "first_round_unsafe": float(g.loc[g["round_number"] == 1, "decision"].iloc[0]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def llm_player_features(model: str) -> pd.DataFrame:
    per_player: dict[str, list[dict]] = {}
    for d in NEUTRAL_INPUTS[model]:
        subdir = MODEL_SUBDIR.get(model, "")
        p = ROOT / d / subdir / "turns.jsonl"
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                key = f"{r['game_id']}::{r['player']}"
                per_player.setdefault(key, []).append(r)
    rows = []
    for key, turns in per_player.items():
        turns = sorted(turns, key=lambda r: r["round"])
        g = pd.DataFrame(turns)
        later = g[g["round"] > 1].copy()
        later["own_prev_unsafe"] = (later["own_prev_action"] == "unsafe").astype(int)
        later["opponent_prev_unsafe"] = (later["opponent_prev_action"] == "unsafe").astype(int)
        row = {
            "player_key": key,
            "overall_unsafe_rate": g["unsafe"].mean(),
            "reciprocity": _safe_diff(later, "unsafe", later["opponent_prev_unsafe"] == 1, later["opponent_prev_unsafe"] == 0),
            "position_sensitivity": _safe_diff(later, "unsafe", later["progress_gap_before"] > 0, later["progress_gap_before"] < 0),
            "own_autocorrelation": _safe_diff(later, "unsafe", later["own_prev_unsafe"] == 1, later["own_prev_unsafe"] == 0),
            "first_round_unsafe": float(g.loc[g["round"] == 1, "unsafe"].iloc[0]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    human = human_features()
    human_imputed = human.copy()
    for feat in FEATURES:
        human_imputed[feat] = human_imputed[feat].fillna(human_imputed[feat].mean())
    mean = human_imputed[FEATURES].mean()
    std = human_imputed[FEATURES].std()
    human_z = (human_imputed[FEATURES] - mean) / std

    km = KMeans(n_clusters=K, random_state=SEED, n_init=10)
    human_labels = km.fit_predict(human_z)
    human_imputed["cluster"] = human_labels

    cluster_summary = human_imputed.groupby("cluster")[FEATURES].mean()
    cluster_summary["n"] = human_imputed.groupby("cluster").size()
    cluster_summary.to_csv(DATA / "human_cluster_summary.csv")
    print(cluster_summary)

    projection_rows = []
    for model in NEUTRAL_INPUTS:
        llm = llm_player_features(model)
        for feat in FEATURES:
            llm[feat] = llm[feat].fillna(mean[feat])
        llm_z = (llm[FEATURES] - mean) / std
        labels = km.predict(llm_z)
        for c in range(K):
            projection_rows.append({"model": model, "cluster": c, "share": float((labels == c).mean()), "n": len(labels)})

    projection = pd.DataFrame(projection_rows)
    projection.to_csv(DATA / "llm_human_cluster_projection.csv", index=False)
    print(projection.pivot(index="model", columns="cluster", values="share"))

    # figures/llm_human_cluster_projection.png is built by
    # build_llm_human_cluster_projection_v2.py, not here. That version adds a
    # literal "Human (reference)" bar and uses the verified archetype *names*
    # rather than generic "Cluster N" labels (the number->name mapping is easy
    # to get wrong when written up separately -- it was, once). This script
    # still owns the underlying tables both consume: human_cluster_summary.csv
    # and llm_human_cluster_projection.csv.
    print("tables written; run build_llm_human_cluster_projection_v2.py for the figure")


if __name__ == "__main__":
    main()
