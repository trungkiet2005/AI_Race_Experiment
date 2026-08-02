#!/usr/bin/env python3
"""Does the N-player position effect flip sign across the risk-persona sweep?

Fits `unsafe ~ C(max_private_risk) + gap_continuous` (own progress minus the mean
of the other N-1 players' progress, generalizing the 2-player progress_gap), one
logit per persona level (none/R1..R6), for gpt-5-nano and gpt-5.4-nano, clustered
by `rep`. See results/cross_model_pilot_synthesis/INSIGHTS.md, N-player position
insight, for interpretation.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "cross_model_pilot_synthesis" / "figures"
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
NPLAYER_ROOT = ROOT / "results" / "nplayer" / "openai"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "slate": "#64748B", "grid": "#DCE3ED",
}
MODEL_LABELS = {"gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano"}
PERSONA_DIRS = {
    "none": "baseline_n3",
    "R1": "risk/R1", "R2": "risk/R2", "R3": "risk/R3",
    "R4": "risk/R4", "R5": "risk/R5", "R6": "risk/R6",
}
PERSONA_ORDER = ["none", "R1", "R2", "R3", "R4", "R5", "R6"]


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


def load_turns(model: str, persona: str) -> pd.DataFrame:
    p = NPLAYER_ROOT / PERSONA_DIRS[persona] / model / "turns.jsonl"
    rows = []
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            others = r["others_progress_before"]
            rows.append({
                "unsafe": r["unsafe"], "round": r["round"], "rep": r["rep"],
                "max_private_risk": r["max_private_risk"],
                "gap": r["own_progress_before"] - (sum(others) / len(others)),
            })
    return pd.DataFrame(rows)


def fit_one(model: str, persona: str) -> dict:
    df = load_turns(model, persona)
    df = df[df["round"] > 1]
    n_risk = df["max_private_risk"].nunique()
    formula = "unsafe ~ C(max_private_risk) + gap" if n_risk > 1 else "unsafe ~ gap"
    try:
        fit = smf.logit(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["rep"]}, disp=0)
        converged = fit.mle_retvals.get("converged", True)
        coef, se, p = float(fit.params["gap"]), float(fit.bse["gap"]), float(fit.pvalues["gap"])
    except Exception:
        converged, coef, se, p = False, np.nan, np.nan, np.nan
    return {
        "model": model, "persona": persona, "n": len(df), "unsafe_rate": float(df["unsafe"].mean()),
        "coef": coef, "se": se, "p": p, "converged": bool(converged),
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    rows = []
    for model in MODEL_LABELS:
        for persona in PERSONA_ORDER:
            r = fit_one(model, persona)
            rows.append(r)
            print(r)
    results = pd.DataFrame(rows)
    results.to_csv(DATA / "nplayer_position_effect_by_persona.csv", index=False)

    setup_plot()
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = list(range(len(PERSONA_ORDER)))
    for color, model in zip([PALETTE["blue"], PALETTE["cyan"]], MODEL_LABELS):
        sub = results[results["model"] == model].set_index("persona").loc[PERSONA_ORDER]
        usable = sub["converged"] & sub["se"].notna() & (sub["se"] < 3) & (sub["unsafe_rate"].between(0.05, 0.95))
        xs = [xi for xi, ok in zip(x, usable) if ok]
        ys = sub.loc[usable, "coef"].values
        errs = 1.96 * sub.loc[usable, "se"].values
        ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=color, linewidth=2, markersize=7, capsize=4,
                    label=MODEL_LABELS[model])
        dropped = [xi for xi, ok in zip(x, usable) if not ok]
        for xi in dropped:
            ax.plot(xi, 0, marker="x", color=color, markersize=9, alpha=0.5)
    ax.axhline(0, color=PALETTE["slate"], linewidth=1, linestyle="--")
    ax.axhline(-0.295, color=PALETTE["red"], linewidth=1.4, linestyle=":", label="Human study coefficient (-0.296)")
    ax.set_xticks(x, PERSONA_ORDER)
    ax.set_xlabel("Risk-attitude persona (none = no framing, R1 = most risk-averse ... R6 = most risk-seeking)")
    ax.set_ylabel("Logit coefficient: own-minus-others progress gap -> P(Unsafe)")
    ax.set_title("The position effect flips sign as risk-seeking framing increases\n(x markers = floor/ceiling or non-converged, not estimable)", pad=14)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left")
    save_figure(fig, FIGURES / "nplayer_position_effect_sign_flip")
    print("wrote", FIGURES / "nplayer_position_effect_sign_flip.png")


if __name__ == "__main__":
    main()
