#!/usr/bin/env python3
"""Classify humans and every LLM checkpoint against the paper's four canonical strategies.

The reduced strategy set in Fernandez Domingos and Han (2026):

    AS   always Safe
    AU   always Unsafe
    CS   Safe in round 1, thereafter copy the opponent's previous action
    CAS  Unsafe in round 1, thereafter copy the opponent's previous action

Part F of INSIGHTS.md clusters behaviour bottom-up (KMeans on human feature
vectors). This is the complementary top-down view: it asks how well each
population is described by the strategy vocabulary the paper itself defines.
Classification reuses `strategy_analysis.classify` rather than reimplementing
Hamming matching, so the tie-retention rule there applies here too.

Two corrections are essential, and without them the headline percentages are
close to meaningless:

**1. A chance baseline.** Median realised horizon is ~8 rounds. Over 8 binary
choices, an agent with no strategy at all still matches *some* canonical
strategy fairly well by luck, and an agent that simply always plays Unsafe
matches AU perfectly by definition -- which is a description of its base rate,
not evidence that it is running AU as a contingent policy. Each trajectory is
therefore compared against its own null: B Bernoulli sequences drawn at that
player's *own* realised Unsafe rate, classified identically. This preserves the
marginal rate while destroying contingent structure, so AS/AU matches at
extreme base rates correctly fail to beat their null, while genuine
copy-the-opponent behaviour (CS/CAS) does.

**2. An identifiability check.** CS and CAS differ only in round 1. If the
opponent's realised history makes two candidate strategies predict the same
trajectory -- e.g. an all-Safe opponent makes CS identical to AS -- then the
data cannot distinguish them no matter how clean the match is. The share of
trajectories where the strategy set is degenerate is reported, not hidden
inside a tie-break.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from strategy_analysis.classify import (  # noqa: E402
    CANONICAL_STRATEGIES,
    classify_trajectory,
    predict_strategy,
)

OUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUT / "figures"
DATA = OUT / "data"
HUMAN_CSV = ROOT / "public_dataset" / "airace_deidentified_long.csv"

PALETTE = {
    "navy": "#0B132B", "slate": "#64748B", "grid": "#DCE3ED", "human": "#3a3a38",
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
    "green": "#008300", "violet": "#4a3aa7", "magenta": "#e87ba4",
}
STRATEGY_COLORS = {
    "AS": PALETTE["blue"], "AU": PALETTE["orange"],
    "CS": PALETTE["aqua"], "CAS": PALETTE["violet"],
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
POP_ORDER = ["human"] + list(NEUTRAL_INPUTS)
LABELS = {
    "human": "Human", "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}
N_NULL = 400
ALPHA = 0.05
SEED = 0


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 12.5, "axes.labelsize": 10, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def human_trajectories() -> list[dict]:
    df = pd.read_csv(HUMAN_CSV)
    out = []
    for pid, g in df.sort_values("round_number").groupby("participant_id"):
        out.append({
            "id": str(pid),
            "own": g["decision"].astype(int).tolist(),
            "opp": g["decision_opponent"].astype(int).tolist(),
        })
    return out


def llm_trajectories(model: str) -> list[dict]:
    """One trajectory per seat, with the opponent's contemporaneous actions.

    Both seats of a race are in the same file, so the opponent's action in round
    k is simply the other seat's own action in round k -- read directly rather
    than reconstructed from the lagged `opponent_prev_action` field (which would
    also cost the final round). Spot-checked against the lag field: they agree.
    """
    per_game: dict[str, dict[str, list[dict]]] = {}
    for d in NEUTRAL_INPUTS[model]:
        p = ROOT / d / "turns.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                per_game.setdefault(r["game_id"], {}).setdefault(r["player"], []).append(r)

    out = []
    for game_id, seats in per_game.items():
        if len(seats) != 2:  # the engine guarantees exactly two players
            continue
        a, b = sorted(seats)
        acts = {p: [int(t["unsafe"]) for t in sorted(seats[p], key=lambda r: r["round"])]
                for p in (a, b)}
        n = min(len(acts[a]), len(acts[b]))
        if n < 2:
            continue
        for own_p, opp_p in ((a, b), (b, a)):
            out.append({"id": f"{game_id}::{own_p}",
                        "own": acts[own_p][:n], "opp": acts[opp_p][:n]})
    return out


def degenerate_strategies(opp: list[int]) -> list[tuple[str, str]]:
    """Pairs of canonical strategies that predict identical actions for this opponent."""
    preds = {s: predict_strategy(s, opp) for s in CANONICAL_STRATEGIES}
    dupes = []
    names = list(CANONICAL_STRATEGIES)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            if preds[a] == preds[b]:
                dupes.append((a, b))
    return dupes


def analyse(traj: list[dict], rng: np.random.Generator) -> dict:
    rows = []
    for t in traj:
        own, opp = t["own"], t["opp"]
        res = classify_trajectory(own, opp)
        best_rate = min(m.mismatch_rate for m in res.matches)
        base_rate = float(np.mean(own))
        n = len(own)

        # Null: same length, same marginal Unsafe rate, no contingent structure.
        null_best = np.empty(N_NULL)
        for b in range(N_NULL):
            sim = rng.binomial(1, base_rate, size=n).tolist()
            sim_res = classify_trajectory(sim, opp)
            null_best[b] = min(m.mismatch_rate for m in sim_res.matches)
        # One-sided: is the real trajectory closer to a canonical strategy than chance?
        p = float((null_best <= best_rate).mean())

        dupes = degenerate_strategies(opp)
        rows.append({
            "id": t["id"], "horizon": n, "base_rate": base_rate,
            "best_strategies": list(res.best_strategies),
            "unique_best": res.unique_best_strategy,
            "best_mismatch_rate": best_rate,
            "null_median_mismatch": float(np.median(null_best)),
            "p_value": p,
            "beats_chance": bool(p < ALPHA),
            "n_degenerate_pairs": len(dupes),
            "degenerate": bool(dupes),
        })
    return {"rows": rows}


def summarise(name: str, rows: list[dict]) -> dict:
    n = len(rows)
    unique = [r for r in rows if r["unique_best"]]
    beats = [r for r in rows if r["beats_chance"]]
    beats_unique = [r for r in beats if r["unique_best"]]

    def share(counter: Counter, denom: int) -> dict:
        return {s: round(100 * counter.get(s, 0) / denom, 1) for s in CANONICAL_STRATEGIES} if denom else {}

    return {
        "population": name,
        "n_trajectories": n,
        "median_horizon": float(np.median([r["horizon"] for r in rows])),
        "pct_unique_best": round(100 * len(unique) / n, 1),
        "pct_degenerate_strategy_set": round(100 * sum(r["degenerate"] for r in rows) / n, 1),
        "pct_beats_chance": round(100 * len(beats) / n, 1),
        "mean_best_mismatch_rate": round(float(np.mean([r["best_mismatch_rate"] for r in rows])), 4),
        "mean_null_median_mismatch": round(float(np.mean([r["null_median_mismatch"] for r in rows])), 4),
        "share_all_unique": share(Counter(r["unique_best"] for r in unique), len(unique)),
        "share_beats_chance_unique": share(Counter(r["unique_best"] for r in beats_unique), len(beats_unique)),
        "n_beats_chance_unique": len(beats_unique),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    summaries, all_rows = [], []
    for pop in POP_ORDER:
        traj = human_trajectories() if pop == "human" else llm_trajectories(pop)
        res = analyse(traj, rng)
        for r in res["rows"]:
            r["population"] = pop
        all_rows.extend(res["rows"])
        s = summarise(pop, res["rows"])
        summaries.append(s)
        print(f"\n=== {LABELS[pop]} === n={s['n_trajectories']} median horizon={s['median_horizon']:.0f}")
        print(f"  unique nearest strategy: {s['pct_unique_best']}%   degenerate strategy set: {s['pct_degenerate_strategy_set']}%")
        print(f"  mean best mismatch {s['mean_best_mismatch_rate']:.3f} vs null median {s['mean_null_median_mismatch']:.3f}")
        print(f"  beats own chance null: {s['pct_beats_chance']}%   (unique & beats chance: n={s['n_beats_chance_unique']})")
        print(f"  mix, all unique-best:        {s['share_all_unique']}")
        print(f"  mix, unique & beats chance:  {s['share_beats_chance_unique']}")

    pd.DataFrame(all_rows).to_csv(DATA / "canonical_strategy_classification.csv", index=False)
    with open(DATA / "canonical_strategy_summary.json", "w") as f:
        json.dump(summaries, f, indent=2)

    fig_strategies(summaries)


def fig_strategies(summaries: list[dict]) -> None:
    setup_plot()
    pops = [s["population"] for s in summaries]
    labels = [LABELS[p] for p in pops]
    xs = np.arange(len(pops))
    fig, axes = plt.subplots(1, 3, figsize=(15.8, 6.4),
                              gridspec_kw={"width_ratios": [1.25, 1.0, 1.0], "wspace": 0.42})
    fig.subplots_adjust(left=0.065, right=0.985, top=0.83, bottom=0.24)

    # Panel 1: the tally a careless reading would report.
    ax = axes[0]
    bottom = np.zeros(len(pops))
    for strat in CANONICAL_STRATEGIES:
        vals = np.array([s["share_all_unique"].get(strat, 0) for s in summaries], dtype=float)
        ax.bar(xs, vals, bottom=bottom, color=STRATEGY_COLORS[strat], width=0.62,
               label=strat, edgecolor="white", linewidth=0.6)
        bottom += vals
    ax.set_title("What a naive tally reports\n(nearest strategy, unique matches only)", fontsize=11.5)
    ax.set_ylim(0, 100)
    ax.set_ylabel("Share of classified trajectories (%)")
    ax.set_xticks(xs, labels, rotation=30, ha="right", fontsize=8.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
    ax.legend(frameon=False, ncol=4, fontsize=9, loc="upper center", bbox_to_anchor=(0.5, -0.24))

    # Panel 2: almost none of it survives a base-rate-matched null.
    ax = axes[1]
    beats = [s["pct_beats_chance"] for s in summaries]
    ns = [s["n_beats_chance_unique"] for s in summaries]
    ax.barh(xs, beats, height=0.6,
            color=[PALETTE["human"] if p == "human" else PALETTE["slate"] for p in pops])
    for xi, (v, s) in enumerate(zip(beats, summaries)):
        n_hit = round(v / 100 * s["n_trajectories"])
        ax.text(max(v, 0) + 1.2, xi, f"{v:.1f}%  ({n_hit}/{s['n_trajectories']})",
                va="center", fontsize=8.6, color=PALETTE["navy"], weight="bold")
    ax.set_yticks(xs, labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 26)
    ax.set_xlabel("% beating their own chance null")
    ax.set_title("What actually survives\na base-rate-matched null", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)

    # Panel 3: and often the four strategies are not even distinguishable.
    ax = axes[2]
    deg = [s["pct_degenerate_strategy_set"] for s in summaries]
    ax.barh(xs, deg, height=0.6, color=PALETTE["orange"])
    for xi, v in zip(xs, deg):
        ax.text(v + 1.2, xi, f"{v:.0f}%", va="center", fontsize=8.8,
                color=PALETTE["navy"], weight="bold")
    ax.set_yticks(xs, labels, fontsize=8.6)
    ax.invert_yaxis()
    ax.set_xlim(0, 100)
    ax.set_xlabel("% where >=2 strategies predict\nidentical actions")
    ax.set_title("...and how often the strategies\naren't even distinguishable", fontsize=11.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.7)

    fig.suptitle("The paper's four canonical strategies do not describe any population here better than chance",
                  fontsize=13.5, y=0.955)
    fig.savefig(FIGURES / "canonical_strategy_classification.png", dpi=220, facecolor="white")
    fig.savefig(FIGURES / "canonical_strategy_classification.pdf", facecolor="white")
    plt.close(fig)
    print("\nwrote", FIGURES / "canonical_strategy_classification.png")


if __name__ == "__main__":
    main()
