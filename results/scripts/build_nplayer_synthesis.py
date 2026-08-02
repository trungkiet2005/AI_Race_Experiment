#!/usr/bin/env python3
"""N-player synthesis figure: peer-composition (social-framing) deterrence effect.

Reads the OpenAI N=3 coopadv sweep (results/nplayer/openai/coopadv/{AAA..CCC}/{model})
directly from players.csv -- no intermediate derived table -- and plots each
player's own framing role (adversarial/cooperative) against how many of their two
groupmates are also framed adversarial. See results/cross_model_pilot_synthesis/
INSIGHTS.md, N-player Insight 3, for the interpretation and caveats.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "cross_model_pilot_synthesis" / "figures"
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
COOPADV_ROOT = ROOT / "results" / "nplayer" / "openai" / "coopadv"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "slate": "#64748B", "grid": "#DCE3ED",
}
CODES = ["AAA", "AAC", "ACA", "ACC", "CAA", "CAC", "CCA", "CCC"]
MODEL_LABELS = {"gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano"}


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


def load_peer_composition() -> pd.DataFrame:
    frames = []
    for model in MODEL_LABELS:
        for code in CODES:
            p = COOPADV_ROOT / code / model / "players.csv"
            df = pd.read_csv(p)
            df["code"] = code
            frames.append(df)
    all_df = pd.concat(frames, ignore_index=True)
    n_adv_total = all_df["code"].str.count("A")
    is_own_adv = (all_df["persona_role"] == "adversarial").astype(int)
    all_df["peer_adv"] = n_adv_total - is_own_adv
    return all_df


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df = load_peer_composition()
    summary = df.groupby(["model", "persona_role", "peer_adv"])["unsafe_frequency"].agg(["mean", "count"]).reset_index()
    summary.to_csv(DATA / "nplayer_peer_composition.csv", index=False)

    setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 5.2), sharey=True)
    styles = {"adversarial": (PALETTE["red"], "o-"), "cooperative": (PALETTE["teal"], "o--")}
    for ax, model in zip(axes, MODEL_LABELS):
        sub = summary[summary["model"] == model]
        for role, (color, fmt) in styles.items():
            rows = sub[sub["persona_role"] == role].sort_values("peer_adv")
            ax.plot(rows["peer_adv"], 100 * rows["mean"], fmt, color=color, linewidth=2.4, markersize=8,
                    label=f"Own role: {role}")
            for _, r in rows.iterrows():
                ax.annotate(f"n={int(r['count'])}", (r["peer_adv"], 100 * r["mean"]), textcoords="offset points",
                            xytext=(0, 9), fontsize=7.5, color=PALETTE["slate"], ha="center")
        ax.set_title(MODEL_LABELS[model], fontsize=12)
        ax.set_xlabel("Number of the other 2 seats framed 'adversarial'")
        ax.set_xticks([0, 1, 2])
        ax.set_ylim(-5, 105)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean player-level Unsafe rate (%)")
    axes[0].legend(frameon=False, loc="center left")
    fig.suptitle("Adversarially-framed players pull back slightly as more groupmates are\nalso adversarial; cooperatively-framed players are unaffected by peers", y=1.1, fontsize=13.5)
    fig.text(0.01, -0.06,
              "N=3 races, OpenAI coopadv sweep (8 seat-composition codes x 2 models, 30 races/code, 90-180 players/point).\n"
              "Own-framing effect (adversarial vs cooperative) replicates the 2-player S_AA/S_CC finding; the peer-count slope is the new result.",
              transform=fig.transFigure, fontsize=8, color=PALETTE["slate"])
    save_figure(fig, FIGURES / "nplayer_peer_composition_effect")
    print("wrote", FIGURES / "nplayer_peer_composition_effect.png")


if __name__ == "__main__":
    main()
