#!/usr/bin/env python3
"""Does playing more Unsafe actually pay off -- for humans, and for each LLM checkpoint?

The human public dataset has no payoff column, so this reconstructs one from
the documented mechanism (ai_race/configs/game/*.json: safeSafe=1.0,
safeUnsafe=0.6, unsafeSafe=2.4, unsafeUnsafe=2.0, racePrize=100, tie=50), which
this project's own documentation states is a paper-faithful copy of the human
study's mechanism. Prize/tie is determined from each participant's own final-
round acc_steps vs. acc_steps_opponent (not from the `won_race` field, which
cannot distinguish a win from a tie -- see the Part A data-quality note on that
field). The human file has no per-round setback/risk-draw field, so the human
"total payoff" reconstructed here is stage payoff + prize only -- an upper-bound
proxy that omits the setback penalty entirely. LLM data has the real
`final_payoff` (including the actual setback draw) in players.csv, so both a
matching proxy (stage+prize only) and the true realised payoff are reported for
LLMs, to show how much the omitted setback term matters.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
HUMAN_CSV = ROOT / "public_dataset" / "airace_deidentified_long.csv"
OUTPUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUTPUT / "figures"
DATA = OUTPUT / "data"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "slate": "#64748B", "grid": "#DCE3ED",
}
MODEL_ORDER = [
    "gpt-5-nano", "gpt-5.4-nano",
    "google/gemini-3-flash-preview", "google/gemini-3.1-flash-lite-preview", "google/gemini-3.5-flash-lite",
    "gpt-5.6-luna", "gpt-5.6-terra", "claude-opus-5", "claude-sonnet-5",
]
MODEL_COLORS = [PALETTE["blue"], PALETTE["cyan"], PALETTE["teal"], PALETTE["amber"], PALETTE["red"],
                "#7C3AED", "#DB2777", "#0F766E", "#B45309"]
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
PAYOFF = {(0, 0): 1.0, (0, 1): 0.6, (1, 0): 2.4, (1, 1): 2.0}  # (own, opponent) -> own round payoff
RACE_PRIZE = 100.0


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


def human_payoffs() -> pd.DataFrame:
    df = pd.read_csv(HUMAN_CSV)
    df["own_pay"] = df.apply(lambda r: PAYOFF[(int(r["decision"]), int(r["decision_opponent"]))], axis=1)
    rows = []
    for pid, g in df.groupby("participant_id"):
        stage_payoff = g["own_pay"].sum()
        last = g.loc[g["round_number"].idxmax()]
        if last["acc_steps"] > last["acc_steps_opponent"]:
            prize = RACE_PRIZE
        elif last["acc_steps"] == last["acc_steps_opponent"]:
            prize = RACE_PRIZE / 2
        else:
            prize = 0.0
        rows.append({
            "participant_id": pid, "overall_unsafe_rate": g["decision"].mean(),
            "stage_payoff": stage_payoff, "prize": prize, "proxy_total_payoff": stage_payoff + prize,
            "max_private_risk": g["max_private_risk"].iloc[0],
        })
    return pd.DataFrame(rows)


def llm_payoffs(model: str) -> pd.DataFrame:
    rows = []
    for d in NEUTRAL_INPUTS[model]:
        subdir = MODEL_SUBDIR.get(model, "")
        p = ROOT / d / subdir / "players.csv"
        with open(p) as f:
            for row in csv.DictReader(f):
                rows.append({
                    "overall_unsafe_rate": float(row["unsafe_frequency"]),
                    "stage_payoff": float(row["stage_payoff"]),
                    "prize": float(row["prize"]),
                    "proxy_total_payoff": float(row["stage_payoff"]) + float(row["prize"]),
                    "true_final_payoff": float(row["final_payoff"]),
                    "setback": int(row["setback"]),
                    "max_private_risk": float(row["max_private_risk"]),
                })
    return pd.DataFrame(rows)


def corr_stats(x: pd.Series, y: pd.Series) -> dict:
    r, p = stats.pearsonr(x, y)
    rho, ps = stats.spearmanr(x, y)
    slope, intercept, _, _, se = stats.linregress(x, y)[:5]
    return {"pearson_r": float(r), "pearson_p": float(p), "spearman_rho": float(rho), "spearman_p": float(ps),
            "ols_slope": float(slope), "ols_slope_se": float(se), "n": int(len(x))}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    human = human_payoffs()
    human_corr = corr_stats(human["overall_unsafe_rate"], human["proxy_total_payoff"])
    print("human (proxy, no setback):", human_corr)

    results = {"human_proxy": human_corr}
    llm_frames = {}
    for model in NEUTRAL_INPUTS:
        llm = llm_payoffs(model)
        llm_frames[model] = llm
        proxy_corr = corr_stats(llm["overall_unsafe_rate"], llm["proxy_total_payoff"])
        true_corr = corr_stats(llm["overall_unsafe_rate"], llm["true_final_payoff"])
        results[f"{model}_proxy"] = proxy_corr
        results[f"{model}_true"] = true_corr
        setback_rate = float(llm["setback"].mean())
        mean_gap = float((llm["proxy_total_payoff"] - llm["true_final_payoff"]).mean())
        print(model, "proxy:", proxy_corr, "true:", true_corr, "setback_rate:", setback_rate, "mean proxy-minus-true gap:", mean_gap)
        results[f"{model}_setback_rate"] = setback_rate
        results[f"{model}_proxy_minus_true_mean_gap"] = mean_gap

    with open(DATA / "payoff_welfare_correlations.json", "w") as f:
        json.dump(results, f, indent=2)

    setup_plot()
    n_panels = 1 + len(MODEL_ORDER)  # human + one per checkpoint
    n_cols = 4
    n_rows = -(-n_panels // n_cols)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4.6 * n_cols, 4.2 * n_rows), sharex=False)
    axes = axes.flatten()
    for spare in axes[n_panels:]:
        spare.axis("off")

    ax = axes[0]
    ax.scatter(human["overall_unsafe_rate"] * 100, human["proxy_total_payoff"], s=14, alpha=0.5, color=PALETTE["slate"])
    z = np.polyfit(human["overall_unsafe_rate"], human["proxy_total_payoff"], 1)
    xs = np.linspace(human["overall_unsafe_rate"].min(), human["overall_unsafe_rate"].max(), 50)
    ax.plot(xs * 100, np.poly1d(z)(xs), color=PALETTE["navy"], linewidth=2)
    ax.set_title(f"Human (proxy, no setback)\nr={human_corr['pearson_r']:.2f}, p={human_corr['pearson_p']:.3f}", fontsize=10.5)
    ax.set_xlabel("Unsafe rate (%)")
    ax.set_ylabel("Reconstructed payoff (stage + prize)")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color=PALETTE["grid"], linewidth=0.6)

    for i, (color, model) in enumerate(zip(MODEL_COLORS, MODEL_ORDER)):
        ax = axes[i + 1]
        llm = llm_frames[model]
        ax.scatter(llm["overall_unsafe_rate"] * 100, llm["true_final_payoff"], s=14, alpha=0.5, color=color, label="True (w/ setback)")
        ax.scatter(llm["overall_unsafe_rate"] * 100, llm["proxy_total_payoff"], s=14, alpha=0.3, color=PALETTE["slate"], marker="x", label="Proxy (no setback)")
        z = np.polyfit(llm["overall_unsafe_rate"], llm["true_final_payoff"], 1)
        llm_xs = np.linspace(llm["overall_unsafe_rate"].min(), llm["overall_unsafe_rate"].max(), 50)
        ax.plot(llm_xs * 100, np.poly1d(z)(llm_xs), color=color, linewidth=2)
        r = results[f"{model}_true"]["pearson_r"]
        p = results[f"{model}_true"]["pearson_p"]
        ax.set_title(f"{MODEL_LABELS[model]}\nr={r:.2f}, p={p:.3f} (true payoff)", fontsize=10.5)
        ax.set_xlabel("Unsafe rate (%)")
        if (i + 1) % n_cols == 0:
            ax.set_ylabel("Payoff")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(color=PALETTE["grid"], linewidth=0.6)
        if i == 0:
            ax.legend(frameon=False, fontsize=7.5, loc="best")

    fig.suptitle("Does playing more Unsafe pay off? Per-player Unsafe rate vs. realised payoff", fontsize=15, y=1.02)
    fig.tight_layout()
    save_figure(fig, FIGURES / "payoff_welfare_unsafe_vs_payoff")
    print("wrote", FIGURES / "payoff_welfare_unsafe_vs_payoff.png")


if __name__ == "__main__":
    main()
