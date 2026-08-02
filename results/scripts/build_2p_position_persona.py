#!/usr/bin/env python3
"""Does the 2-player position effect also flip sign under persona framing?

Mirrors build_nplayer_position_persona.py's question, but for the 2-player
engine's own risk-persona sweep. Fits `unsafe ~ C(max_private_risk) +
progress_gap_before` (the direct 2-player analogue of the N-player `gap`),
one logit per *own* seat_persona_role (R1..R6 from the risk_matrix, plus the
no-persona baseline and R0-neutral for reference), cluster-robust by `rep`,
for gpt-5-nano and gpt-5.4-nano. Pools over the opponent's persona role, since
the question here is "does MY OWN assigned framing change how position drives
MY behavior," not the peer-composition question already answered in the
N-player data.
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
OPENAI_ROOT = ROOT / "results" / "frontier" / "openai"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "slate": "#64748B", "grid": "#DCE3ED",
}
MODEL_LABELS = {"gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano"}
ROLE_MAP = {f"risk-{i}": f"R{i}" for i in range(1, 7)}
ROLE_MAP.update({"neutral": "R0", "": "none"})
LEVEL_ORDER = ["none", "R0", "R1", "R2", "R3", "R4", "R5", "R6"]

# All directories under results/frontier/openai/persona/ plus the no-persona
# baseline; every one of these was already confirmed complete/clean (0 parse
# failures, manifest counts matching file contents) in the earlier audit pass.
PERSONA_DIRS = [
    "baseline",
    "persona/R0_neutral",
    "persona/Rminus_risk_averse",
    "persona/Rplus_risk_seeking",
    "persona/S_AA_adv_adv",
    "persona/S_AC_adv_coop",
    "persona/S_CA_coop_adv",
    "persona/S_CC_coop_coop",
] + [f"persona/risk_matrix/R{i}_R{j}" for i in range(1, 7) for j in range(1, 7)]


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


def load_rows(model: str) -> pd.DataFrame:
    rows = []
    for d in PERSONA_DIRS:
        p = OPENAI_ROOT / d / model / "turns.jsonl"
        if not p.exists():
            continue
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r["round"] <= 1 or r["own_prev_action"] is None:
                    continue
                role = r.get("seat_persona_role", "") or ""
                level = ROLE_MAP.get(role)
                if level is None:
                    continue  # adversarial/cooperative handled separately if needed
                rows.append({
                    "level": level, "rep": r["rep"], "max_private_risk": r["max_private_risk"],
                    "gap": r["progress_gap_before"], "unsafe": r["unsafe"],
                })
    return pd.DataFrame(rows)


def fit_one(df: pd.DataFrame, level: str) -> dict:
    sub = df[df["level"] == level]
    if len(sub) == 0:
        return {"level": level, "n": 0, "unsafe_rate": np.nan, "coef": np.nan, "se": np.nan, "p": np.nan, "converged": False}
    n_risk = sub["max_private_risk"].nunique()
    formula = "unsafe ~ C(max_private_risk) + gap" if n_risk > 1 else "unsafe ~ gap"
    try:
        fit = smf.logit(formula, data=sub).fit(cov_type="cluster", cov_kwds={"groups": sub["rep"]}, disp=0)
        converged = bool(fit.mle_retvals.get("converged", True))
        coef, se, p = float(fit.params["gap"]), float(fit.bse["gap"]), float(fit.pvalues["gap"])
    except Exception:
        converged, coef, se, p = False, np.nan, np.nan, np.nan
    return {"level": level, "n": len(sub), "unsafe_rate": float(sub["unsafe"].mean()), "coef": coef, "se": se, "p": p, "converged": converged}


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for model in MODEL_LABELS:
        df = load_rows(model)
        print(f"{model}: {len(df)} round>=2 decisions loaded, levels present: {sorted(df['level'].unique())}")
        for level in LEVEL_ORDER:
            r = fit_one(df, level)
            r["model"] = model
            all_rows.append(r)
            print(r)
    results = pd.DataFrame(all_rows)
    results.to_csv(DATA / "2p_position_effect_by_persona.csv", index=False)

    setup_plot()
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    x = list(range(len(LEVEL_ORDER)))
    for color, model in zip([PALETTE["blue"], PALETTE["cyan"]], MODEL_LABELS):
        sub = results[results["model"] == model].set_index("level").loc[LEVEL_ORDER]
        usable = sub["converged"] & sub["se"].notna() & (sub["se"] < 3) & (sub["n"] > 0) & (sub["unsafe_rate"].between(0.05, 0.95))
        xs = [xi for xi, ok in zip(x, usable) if ok]
        ys = sub.loc[usable, "coef"].values
        errs = 1.96 * sub.loc[usable, "se"].values
        if len(xs):
            ax.errorbar(xs, ys, yerr=errs, fmt="o-", color=color, linewidth=2, markersize=7, capsize=4, label=MODEL_LABELS[model])
        dropped = [xi for xi, ok in zip(x, usable) if not ok and sub.iloc[xi]["n"] > 0]
        for xi in dropped:
            ax.plot(xi, 0, marker="x", color=color, markersize=9, alpha=0.5)
    ax.axhline(0, color=PALETTE["slate"], linewidth=1, linestyle="--")
    ax.axhline(-0.295, color=PALETTE["red"], linewidth=1.4, linestyle=":", label="Human study coefficient (-0.296)")
    ax.set_xticks(x, LEVEL_ORDER)
    ax.set_xlabel("Own seat's risk-attitude persona (none/R0 = no or neutral framing, R1 = most risk-averse ... R6 = most risk-seeking)")
    ax.set_ylabel("Logit coefficient: progress_gap_before -> P(Unsafe)")
    ax.set_title("2-player position effect under persona framing\n(x = floor/ceiling or non-converged, not estimable)", pad=14)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.text(0.01, -0.08,
              "2-player OpenAI persona sweep, cluster-robust by rep. Pooled over the opponent's own persona role.\n"
              "Negative = human-matching direction (behind -> more Unsafe); positive = reversed (ahead -> more Unsafe).",
              fontsize=8, color=PALETTE["slate"])
    save_figure(fig, FIGURES / "2p_position_effect_by_persona")
    print("wrote", FIGURES / "2p_position_effect_by_persona.png")


if __name__ == "__main__":
    main()
