#!/usr/bin/env python3
"""Round-by-round Unsafe trajectory: humans and every LLM checkpoint, one figure.

Part E of INSIGHTS.md described these shapes in prose plus a GPT-5.6-only
companion figure that pooled persona conditions (the only option before those
checkpoints had a neutral lane). Now that every checkpoint has a neutral/
no-persona lane, all eight populations can be drawn on one comparable axis.

Two things make this figure honest rather than merely pretty:

* N shrinks with round number, because races stop stochastically (min 5 rounds,
  then 20%/round). A late-round mean over three surviving races is noise. Points
  are therefore dropped below a minimum-N floor, and the surviving-N curve is
  drawn underneath so the reader can see the support thinning.
* Survivorship: races that run long are not a random subset. A long-horizon-only
  robustness check is reported in INSIGHTS.md rather than silently pooled here.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUT / "figures"
DATA = OUT / "data"
HUMAN_CSV = ROOT / "public_dataset" / "airace_deidentified_long.csv"

PALETTE = {
    "navy": "#0B132B", "slate": "#64748B", "grid": "#DCE3ED", "human": "#3a3a38",
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "green": "#008300", "violet": "#4a3aa7", "magenta": "#e87ba4",
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
    "gpt-5.6-luna": ["results/frontier/bedrock_mantle/luna/baseline/openai.gpt-5.6-luna",
                      "results/frontier/bedrock_mantle/luna/persona/R0_neutral/openai.gpt-5.6-luna"],
    "gpt-5.6-terra": ["results/frontier/bedrock_mantle/terra/baseline/openai.gpt-5.6-terra",
                       "results/frontier/bedrock_mantle/terra/persona/R0_neutral/openai.gpt-5.6-terra"],
    "claude-opus-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-opus-5",
                       "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-opus-5"],
    "claude-sonnet-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-sonnet-5",
                         "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-sonnet-5"],
}
LABELS = {
    "human": "Human", "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}
COLORS = {
    "human": PALETTE["human"], "gpt-5-nano": PALETTE["blue"], "gpt-5.4-nano": PALETTE["orange"],
    "google/gemini-3-flash-preview": PALETTE["aqua"], "google/gemini-3.1-flash-lite-preview": PALETTE["yellow"],
    "google/gemini-3.5-flash-lite": PALETTE["green"], "gpt-5.6-luna": PALETTE["violet"],
    "gpt-5.6-terra": PALETTE["magenta"],
    # Hues are reused across facet groups; no two of these share an axes, so the
    # categorical palette is never asked to separate more than 5 series at once.
    "claude-opus-5": PALETTE["blue"], "claude-sonnet-5": PALETTE["orange"],
}
GROUPS = [
    ("Original baseline checkpoints", ["gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview"]),
    ("Lite / newer-generation checkpoints",
     ["google/gemini-3.1-flash-lite-preview", "google/gemini-3.5-flash-lite", "gpt-5.6-luna", "gpt-5.6-terra"]),
    ("Claude (Bedrock)", ["claude-opus-5", "claude-sonnet-5"]),
]
MIN_N = 20  # below this the per-round mean is too noisy to plot


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 12, "axes.labelsize": 10, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def human_trajectory() -> pd.DataFrame:
    df = pd.read_csv(HUMAN_CSV)
    g = df.groupby("round_number")["decision"].agg(["mean", "size"]).reset_index()
    return g.rename(columns={"round_number": "round", "mean": "unsafe", "size": "n"})


def llm_trajectory(model: str) -> pd.DataFrame:
    by_round: dict[int, list[int]] = {}
    for d in NEUTRAL_INPUTS[model]:
        p = ROOT / d / "turns.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                rec = json.loads(line)
                by_round.setdefault(rec["round"], []).append(rec["unsafe"])
    rows = [{"round": r, "unsafe": float(np.mean(v)), "n": len(v)} for r, v in sorted(by_round.items())]
    return pd.DataFrame(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    traj = {"human": human_trajectory()}
    for m in NEUTRAL_INPUTS:
        traj[m] = llm_trajectory(m)

    tidy = pd.concat([t.assign(model=k) for k, t in traj.items()], ignore_index=True)
    tidy.to_csv(DATA / "round_trajectory_all_checkpoints.csv", index=False)
    for k, t in traj.items():
        keep = t[t["n"] >= MIN_N]
        print(f"{LABELS[k]:24s} rounds 1-{int(keep['round'].max())} (n>={MIN_N}), "
              f"r1={100*t.loc[t['round'] == 1, 'unsafe'].iloc[0]:.1f}% n1={int(t.loc[t['round'] == 1, 'n'].iloc[0])}")

    setup_plot()
    fig, axes = plt.subplots(2, len(GROUPS), figsize=(6.4 * len(GROUPS), 8.0), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08, "wspace": 0.16})

    for col, (group_label, members) in enumerate(GROUPS):
        ax, ax_n = axes[0, col], axes[1, col]
        for key in ["human"] + members:
            t = traj[key][traj[key]["n"] >= MIN_N]
            is_human = key == "human"
            ax.plot(t["round"], t["unsafe"] * 100, marker="o", markersize=4,
                    linewidth=3.0 if is_human else 1.9, color=COLORS[key],
                    label=LABELS[key], zorder=5 if is_human else 3)
            ax_n.plot(traj[key]["round"], traj[key]["n"], linewidth=1.4, color=COLORS[key],
                      alpha=0.9 if is_human else 0.75)
        ax.set_title(group_label, fontsize=12)
        ax.set_ylim(0, 105)
        ax.set_xlim(1, 16)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, fontsize=8.6, loc="lower right", ncol=1)

        ax_n.axhline(MIN_N, color=PALETTE["slate"], linewidth=0.9, linestyle=(0, (3, 3)))
        ax_n.set_yscale("log")
        ax_n.set_xlabel("Round number")
        ax_n.grid(axis="y", color=PALETTE["grid"], linewidth=0.6)
        ax_n.spines[["top", "right"]].set_visible(False)
        if col == 0:
            ax.set_ylabel("Mean Unsafe rate (%)")
            ax_n.set_ylabel("Surviving\ndecisions (log)", fontsize=8.5)

    fig.suptitle("No checkpoint reproduces the human round-by-round shape\n"
                  "(neutral/no-persona lane; human drawn in both panels as the reference)",
                  fontsize=13.5, y=0.99)
    fig.subplots_adjust(left=0.075, right=0.985, top=0.87, bottom=0.09)
    fig.savefig(FIGURES / "round_trajectory.png", dpi=220, facecolor="white")
    fig.savefig(FIGURES / "round_trajectory.pdf", facecolor="white")
    plt.close(fig)
    print("wrote", FIGURES / "round_trajectory.png")

    # The persona-pooled GPT-5.6-only companion is superseded by this figure.
    for stale in ("round_trajectory_gen56.png", "round_trajectory_gen56.pdf",
                   "feature_importance_shap_heatmap_gen56.png", "feature_importance_shap_heatmap_gen56.pdf"):
        p = FIGURES / stale
        if p.exists():
            p.unlink()
            print("removed superseded", p)


if __name__ == "__main__":
    main()
