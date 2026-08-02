#!/usr/bin/env python3
"""Unified human-archetype-projection figure: all seven LLM checkpoints in one chart.

Supersedes the split between analyze_behavioral_clustering.py's original
5-checkpoint figure and build_gen56_full_extension.py's separate GPT-5.6-only
companion chart -- one figure, one legend, using the actual archetype names
(not generic "Cluster N" labels, to avoid the name/number mismatch that
happened when this was written up in prose across two passes).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUT / "figures"
DATA = OUT / "data"
HUMAN_CSV = ROOT / "public_dataset" / "airace_deidentified_long.csv"
BEDROCK_MANTLE_ROOT = ROOT / "results" / "frontier" / "bedrock_mantle"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "purple": "#7C3AED", "rose": "#DB2777",
    "slate": "#64748B", "grid": "#DCE3ED",
}
FEATURES = ["overall_unsafe_rate", "reciprocity", "position_sensitivity", "own_autocorrelation", "first_round_unsafe"]

NEUTRAL_INPUTS = {
    "gpt-5-nano": ["results/frontier/openai/baseline/gpt-5-nano", "results/frontier/openai/persona/R0_neutral/gpt-5-nano"],
    "gpt-5.4-nano": ["results/frontier/openai/baseline/gpt-5.4-nano", "results/frontier/openai/persona/R0_neutral/gpt-5.4-nano"],
    "google/gemini-3-flash-preview": [
        "results/frontier/baseline/google-gemini-3-flash-preview",
        "results/frontier/persona/R0_neutral/google-gemini-3-flash-preview",
    ],
    "google/gemini-3.1-flash-lite-preview": ["results/frontier/baseline/google-gemini-3.1-flash-lite-preview"],
    "google/gemini-3.5-flash-lite": ["results/frontier/baseline/google-gemini-3.5-flash-lite"],
    # GPT-5.6 Luna/Terra gained a real baseline + R0_neutral lane, so they are now
    # projected from the same neutral condition as every other checkpoint instead of
    # from a persona-pooled proxy. That change matters a lot here -- see INSIGHTS.md F2.
    "gpt-5.6-luna": ["results/frontier/bedrock_mantle/luna/baseline/openai.gpt-5.6-luna",
                      "results/frontier/bedrock_mantle/luna/persona/R0_neutral/openai.gpt-5.6-luna"],
    "gpt-5.6-terra": ["results/frontier/bedrock_mantle/terra/baseline/openai.gpt-5.6-terra",
                       "results/frontier/bedrock_mantle/terra/persona/R0_neutral/openai.gpt-5.6-terra"],
    "claude-opus-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-opus-5",
                       "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-opus-5"],
    "claude-sonnet-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-sonnet-5",
                         "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-sonnet-5"],
}
MODEL_ORDER = ["human", "gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview",
               "google/gemini-3.1-flash-lite-preview", "google/gemini-3.5-flash-lite",
               "gpt-5.6-luna", "gpt-5.6-terra", "claude-opus-5", "claude-sonnet-5"]
MODEL_LABELS = {
    "human": "Human\n(reference)", "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1\nFlash Lite", "google/gemini-3.5-flash-lite": "Gemini 3.5\nFlash Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}
# Cluster ID -> archetype name, verified against the original Part F table using
# gpt-5-nano (cluster2=99.2%="Cautious starter") and Gemini 3 Flash
# (cluster1=78.3%="Aggressive/reciprocator", cluster3=20.0%="Reciprocal catch-up")
# as anchors -- see the human_cluster_summary.csv row each name is keyed to.
CLUSTER_NAMES = {0: "Persister", 1: "Aggressive starter /\nreciprocator", 2: "Cautious starter", 3: "Reciprocal catch-up"}
CLUSTER_COLORS = {0: PALETTE["blue"], 1: PALETTE["teal"], 2: PALETTE["amber"], 3: PALETTE["red"]}


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.labelsize": 10, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def _safe_diff(df: pd.DataFrame, col: str, cond_true, cond_false) -> float:
    a = df.loc[cond_true, col]
    b = df.loc[cond_false, col]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(a.mean() - b.mean())


def fit_human_clusters() -> tuple[KMeans, pd.Series, pd.Series, np.ndarray]:
    df = pd.read_csv(HUMAN_CSV)
    rows = []
    for pid, g in df.groupby("participant_id"):
        later = g[g["round_number"] > 1]
        rows.append({
            "overall_unsafe_rate": g["decision"].mean(),
            "reciprocity": _safe_diff(later, "decision", later["decision_opponent_lag"] == 1, later["decision_opponent_lag"] == 0),
            "position_sensitivity": _safe_diff(later, "decision", later["delta_steps_lag"] > 0, later["delta_steps_lag"] < 0),
            "own_autocorrelation": _safe_diff(later, "decision", later["decision_lag"] == 1, later["decision_lag"] == 0),
            "first_round_unsafe": float(g.loc[g["round_number"] == 1, "decision"].iloc[0]),
        })
    human = pd.DataFrame(rows)
    for feat in FEATURES:
        human[feat] = human[feat].fillna(human[feat].mean())
    mean = human[FEATURES].mean()
    std = human[FEATURES].std()
    human_z = (human[FEATURES] - mean) / std
    km = KMeans(n_clusters=4, random_state=0, n_init=10)
    km.fit(human_z)
    return km, mean, std, km.labels_


def llm_neutral_features(model: str) -> pd.DataFrame:
    per_player: dict[str, list[dict]] = {}
    for d in NEUTRAL_INPUTS[model]:
        p = ROOT / d / "turns.jsonl"
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                key = f"{r['game_id']}::{r['player']}"
                per_player.setdefault(key, []).append(r)
    return _features_from_turns(per_player)


def _features_from_turns(per_player: dict[str, list[dict]]) -> pd.DataFrame:
    rows = []
    for key, turns in per_player.items():
        turns = sorted(turns, key=lambda r: r["round"])
        g = pd.DataFrame(turns)
        later = g[g["round"] > 1].copy()
        later["own_prev_unsafe"] = (later["own_prev_action"] == "unsafe").astype(int)
        later["opponent_prev_unsafe"] = (later["opponent_prev_action"] == "unsafe").astype(int)
        rows.append({
            "overall_unsafe_rate": g["unsafe"].mean(),
            "reciprocity": _safe_diff(later, "unsafe", later["opponent_prev_unsafe"] == 1, later["opponent_prev_unsafe"] == 0),
            "position_sensitivity": _safe_diff(later, "unsafe", later["progress_gap_before"] > 0, later["progress_gap_before"] < 0),
            "own_autocorrelation": _safe_diff(later, "unsafe", later["own_prev_unsafe"] == 1, later["own_prev_unsafe"] == 0),
            "first_round_unsafe": float(g.loc[g["round"] == 1, "unsafe"].iloc[0]),
        })
    return pd.DataFrame(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    km, mean, std, human_labels = fit_human_clusters()

    shares: dict[str, dict[int, float]] = {}
    ns: dict[str, int] = {}
    shares["human"] = {c: float((human_labels == c).mean()) for c in range(4)}
    ns["human"] = len(human_labels)
    print("human", "n=", ns["human"], shares["human"])
    for model in MODEL_ORDER:
        if model == "human":
            continue
        llm = llm_neutral_features(model)
        for feat in FEATURES:
            llm[feat] = llm[feat].fillna(mean[feat])
        llm_z = (llm[FEATURES] - mean) / std
        labels = km.predict(llm_z)
        shares[model] = {c: float((labels == c).mean()) for c in range(4)}
        ns[model] = len(labels)
        print(model, "n=", ns[model], shares[model])

    rows = [{"model": m, "cluster": c, "share": s, "n": ns[m]} for m, cs in shares.items() for c, s in cs.items()]
    pd.DataFrame(rows).to_csv(DATA / "llm_human_cluster_projection_unified.csv", index=False)

    setup_plot()
    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    bottom = np.zeros(len(MODEL_ORDER))
    for c in range(4):
        vals = np.array([shares[m][c] for m in MODEL_ORDER]) * 100
        ax.bar([MODEL_LABELS[m] for m in MODEL_ORDER], vals, bottom=bottom, color=CLUSTER_COLORS[c],
               label=f"{CLUSTER_NAMES[c]} (human {shares['human'][c] * 100:.1f}%)", width=0.62)
        bottom += vals
    # Set off the human reference bar from the seven LLM bars with a vertical divider.
    ax.axvline(0.5, color=PALETTE["navy"], linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.6)
    ax.set_ylabel("Share of player-races nearest to each human archetype (%)")
    ax.set_title("Which human behavioral archetype does each checkpoint resemble?", pad=14)
    ax.set_ylim(0, 100)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.22), ncol=2, fontsize=9)
    plt.setp(ax.get_xticklabels(), rotation=12, ha="right", fontsize=9)
    fig.tight_layout(rect=[0, 0.16, 1, 1])
    fig.savefig(FIGURES / "llm_human_cluster_projection.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "llm_human_cluster_projection.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote", FIGURES / "llm_human_cluster_projection.png")

    old_gen56_png = FIGURES / "llm_human_cluster_projection_gen56.png"
    old_gen56_pdf = FIGURES / "llm_human_cluster_projection_gen56.pdf"
    for p in (old_gen56_png, old_gen56_pdf):
        if p.exists():
            p.unlink()
            print("removed superseded", p)


if __name__ == "__main__":
    main()
