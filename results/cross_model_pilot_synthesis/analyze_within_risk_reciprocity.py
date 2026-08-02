#!/usr/bin/env python3
"""Reciprocity measured the way the human study measures it: conditional on risk.

Part D ranks features by SHAP share and finds no LLM checkpoint leads with
opponent-reciprocity. That ranking is easy to misread, for two reasons this
script removes:

1. **Collinearity.** When a checkpoint's policy is close to deterministic given
   the risk treatment, the opponent's previous action becomes a near-perfect
   proxy for risk, and a tree model can split credit between them arbitrarily.
   Claude Opus 5 is the extreme case: it plays Unsafe on 100% of risk-0.1
   decisions and ~0% of risk-0.6/0.9 decisions, so "opponent played Unsafe" and
   "risk is 0.1" are the same event. Its 43% opponent-SHAP share is an artifact.
2. **Aggregation.** A gap computed over pooled risk levels mixes a genuine
   within-condition response with the between-condition difference in base rate.

The fix is the estimand the source paper actually uses: the change in P(Unsafe)
when the opponent's previous action was Unsafe rather than Safe, **holding the
risk treatment fixed**. Estimated per (population x risk level) as a linear
probability model with cluster-robust SEs, clustered on the CRN block so
matched repetitions are not treated as independent.

A cell is reported only where both arms exist with enough support; where a
checkpoint's own determinism removes all variation in the opponent's action,
the effect is **not identified** and is labelled as such rather than estimated
from a handful of observations.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUT / "figures"
DATA = OUT / "data"
HUMAN_CSV = ROOT / "public_dataset" / "airace_deidentified_long.csv"

PALETTE = {
    "navy": "#0B132B", "slate": "#64748B", "grid": "#DCE3ED", "human": "#3a3a38",
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
}
RISKS = [0.1, 0.6, 0.9]
RISK_LABELS = {0.1: "Risk cap 10%", 0.6: "Risk cap 60%", 0.9: "Risk cap 90%"}
MIN_ARM = 15  # minimum observations in each arm before a cell is estimated

NEUTRAL_INPUTS = {
    "gpt-5-nano": ["results/frontier/openai/baseline/gpt-5-nano", "results/frontier/openai/persona/R0_neutral/gpt-5-nano"],
    "gpt-5.4-nano": ["results/frontier/openai/baseline/gpt-5.4-nano", "results/frontier/openai/persona/R0_neutral/gpt-5.4-nano"],
    "google/gemini-3-flash-preview": [
        "results/frontier/baseline/google-gemini-3-flash-preview",
        "results/frontier/persona/R0_neutral/google-gemini-3-flash-preview",
    ],
    "google/gemini-3.1-flash-lite-preview": ["results/frontier/baseline/google-gemini-3.1-flash-lite-preview"],
    "google/gemini-3.5-flash-lite": ["results/frontier/baseline/google-gemini-3.5-flash-lite"],
    "gpt-5.6-luna": ["results/frontier/bedrock_mantle/luna/baseline/openai.gpt-5.6-luna",
                      "results/frontier/bedrock_mantle/luna/persona/R0_neutral/openai.gpt-5.6-luna"],
    "gpt-5.6-terra": ["results/frontier/bedrock_mantle/terra/baseline/openai.gpt-5.6-terra",
                       "results/frontier/bedrock_mantle/terra/persona/R0_neutral/openai.gpt-5.6-terra"],
    "claude-opus-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-opus-5",
                       "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-opus-5"],
    "claude-sonnet-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-sonnet-5",
                         "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-sonnet-5"],
}
POP_ORDER = ["human"] + list(NEUTRAL_INPUTS)
LABELS = {
    "human": "Human", "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def human_frame() -> pd.DataFrame:
    df = pd.read_csv(HUMAN_CSV)
    df = df[df["round_number"] > 1]
    return pd.DataFrame({
        "risk": df["max_private_risk"].astype(float),
        "opp_prev_unsafe": df["decision_opponent_lag"].astype(int),
        "unsafe": df["decision"].astype(int),
        "cluster": df["participant_id"].astype(str),
    })


def llm_frame(model: str) -> pd.DataFrame:
    rows = []
    for d in NEUTRAL_INPUTS[model]:
        p = ROOT / d / "turns.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r["round"] <= 1 or r.get("opponent_prev_action") is None:
                    continue
                rows.append({
                    "risk": float(r["max_private_risk"]),
                    "opp_prev_unsafe": 1 if r["opponent_prev_action"] == "unsafe" else 0,
                    "unsafe": int(r["unsafe"]),
                    "cluster": str(r.get("rep", r["game_id"])),
                })
    return pd.DataFrame(rows)


def cell_effect(df: pd.DataFrame) -> dict:
    """Cluster-robust LPM of unsafe on opponent's previous action, one risk cell."""
    n_u = int((df["opp_prev_unsafe"] == 1).sum())
    n_s = int((df["opp_prev_unsafe"] == 0).sum())
    if n_u < MIN_ARM or n_s < MIN_ARM:
        return {"status": "not_identified", "n_opp_unsafe": n_u, "n_opp_safe": n_s}
    if df["unsafe"].nunique() < 2:
        return {"status": "no_outcome_variance", "n_opp_unsafe": n_u, "n_opp_safe": n_s}
    fit = smf.ols("unsafe ~ opp_prev_unsafe", data=df).fit(
        cov_type="cluster", cov_kwds={"groups": df["cluster"].values})
    ci = fit.conf_int()
    return {
        "status": "ok", "n_opp_unsafe": n_u, "n_opp_safe": n_s,
        "n_clusters": int(df["cluster"].nunique()),
        "effect_pp": float(fit.params["opp_prev_unsafe"]) * 100,
        "ci_low_pp": float(ci.loc["opp_prev_unsafe", 0]) * 100,
        "ci_high_pp": float(ci.loc["opp_prev_unsafe", 1]) * 100,
        "p_value": float(fit.pvalues["opp_prev_unsafe"]),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}
    for pop in POP_ORDER:
        df = human_frame() if pop == "human" else llm_frame(pop)
        per_risk = {}
        for risk in RISKS:
            per_risk[str(risk)] = cell_effect(df[df["risk"] == risk].copy())
        results[pop] = per_risk
        cells = " | ".join(
            f"{risk}: " + (f"{c['effect_pp']:+.1f}pp" if c["status"] == "ok" else c["status"])
            for risk, c in zip(RISKS, per_risk.values()))
        print(f"{LABELS[pop]:24s} {cells}")

    with open(DATA / "within_risk_reciprocity.json", "w") as f:
        json.dump(results, f, indent=2)

    fig_reciprocity(results)


def fig_reciprocity(results: dict) -> None:
    setup_plot()
    pops = POP_ORDER
    ys = np.arange(len(pops))
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 5.6), sharey=True)

    for ax, risk in zip(axes, RISKS):
        for yi, pop in enumerate(pops):
            c = results[pop][str(risk)]
            is_human = pop == "human"
            if c["status"] != "ok":
                ax.text(2, yi, "not identified — no variation\nin opponent's action",
                        va="center", ha="left", fontsize=7.4, color=PALETTE["slate"], style="italic")
                continue
            eff, lo, hi = c["effect_pp"], c["ci_low_pp"], c["ci_high_pp"]
            color = PALETTE["human"] if is_human else (
                PALETTE["aqua"] if lo > 0 else PALETTE["orange"] if hi < 0 else PALETTE["slate"])
            ax.barh(yi, eff, height=0.6, color=color)
            ax.plot([lo, hi], [yi, yi], color=PALETTE["navy"], linewidth=1.3)
            ax.text(eff + (2.5 if eff >= 0 else -2.5), yi, f"{eff:+.0f}",
                    va="center", ha="left" if eff >= 0 else "right",
                    fontsize=8.4, weight="bold", color=PALETTE["navy"])
        ax.axvline(0, color=PALETTE["navy"], linewidth=1.0)
        # The human effect as a reference band across every panel.
        hc = results["human"][str(risk)]
        if hc["status"] == "ok":
            ax.axvspan(hc["ci_low_pp"], hc["ci_high_pp"], color=PALETTE["human"], alpha=0.10, zorder=0)
        ax.set_title(RISK_LABELS[risk], fontsize=12)
        ax.set_xlim(-30, 100)
        ax.set_xlabel("Change in P(Unsafe) when opponent\nplayed Unsafe last round (pp)")
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)
    axes[0].set_yticks(ys, [LABELS[p] for p in pops], fontsize=9)
    axes[0].invert_yaxis()

    fig.suptitle("Reciprocity, holding the risk treatment fixed: Claude Sonnet 5 far exceeds humans, Claude Opus 5 has none to measure",
                  fontsize=13, y=0.98)
    fig.subplots_adjust(left=0.155, right=0.99, top=0.86, bottom=0.13, wspace=0.08)
    fig.savefig(FIGURES / "within_risk_reciprocity.png", dpi=220, facecolor="white")
    fig.savefig(FIGURES / "within_risk_reciprocity.pdf", facecolor="white")
    plt.close(fig)
    print("\nwrote", FIGURES / "within_risk_reciprocity.png")


if __name__ == "__main__":
    main()
