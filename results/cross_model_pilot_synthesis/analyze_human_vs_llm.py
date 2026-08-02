#!/usr/bin/env python3
"""Direct human-vs-LLM comparison using the paper's own de-identified raw data.

Earlier synthesis work (results/cross_model_pilot_synthesis/) compared LLM
behavior against a small set of frozen human summary statistics
(results/scripts/human_reference.json: E1-E8). This script goes one level
deeper: it reproduces the human paper's own sample-construction recipe on the
raw participant-round data (public_dataset/), refits the
paper's dynamic specification directly (rather than trusting a single frozen
coefficient), and builds real distribution-vs-distribution comparisons against
LLM player-level data already produced by build_cross_model_pilot_synthesis.py.

It never edits or re-derives the DO_NOT_DEPOSIT id-mapping file, which is not
copied into this repository at all.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
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
MODEL_COLORS = [PALETTE["blue"], PALETTE["cyan"], PALETTE["teal"], PALETTE["amber"], PALETTE["red"]]
MODEL_ORDER = [
    "gpt-5-nano", "gpt-5.4-nano",
    "google/gemini-3-flash-preview", "google/gemini-3.1-flash-lite-preview", "google/gemini-3.5-flash-lite",
]
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}
AGE_BIN_MIDPOINT = {
    "18-22": 20, "23-27": 25, "28-32": 30, "33-37": 35, "38-42": 40,
    "43-47": 45, "48-52": 50, "53-57": 55, "58-62": 60, "68-72": 70,
}


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


# --- 1. load and validate the human raw data ---------------------------------------


def load_human() -> pd.DataFrame:
    df = pd.read_csv(HUMAN_CSV)
    return df


def fig2a_replication(df: pd.DataFrame) -> dict:
    """Per-participant mean decision across ALL rounds, by treatment. Raw, unfiltered set."""
    per_participant = df.groupby(["participant_id", "max_private_risk"], as_index=False)["decision"].mean()
    groups = {r: g["decision"].values for r, g in per_participant.groupby("max_private_risk")}
    n_by_risk = {r: len(v) for r, v in groups.items()}
    out = {"n_by_risk": n_by_risk, "mean_by_risk": {r: float(np.mean(v)) for r, v in groups.items()}}
    pairs = [(0.1, 0.6), (0.6, 0.9), (0.1, 0.9)]
    contrasts = {}
    for a, b in pairs:
        x, y = groups[a], groups[b]
        t, p = stats.ttest_ind(x, y)
        pooled_sd = np.sqrt(((len(x) - 1) * x.var(ddof=1) + (len(y) - 1) * y.var(ddof=1)) / (len(x) + len(y) - 2))
        d = (x.mean() - y.mean()) / pooled_sd
        contrasts[f"{a}_vs_{b}"] = {"cohens_d": float(d), "t": float(t), "p": float(p), "p_bonferroni": float(min(1.0, p * 3))}
    out["contrasts"] = contrasts
    out["per_participant_by_risk"] = {r: v.tolist() for r, v in groups.items()}
    return out


def build_analysed_sample(df: pd.DataFrame) -> pd.DataFrame:
    """Reproduce the README's exact round>=2 covariate-complete sample for Table 1."""
    d = df[df["round_number"] > 1].copy()
    d = d[d["sex"] != "CONSENT_REVOKED"]
    d = d[~d["nationality_group"].isin(["DATA_EXPIRED", "CONSENT_REVOKED"])]
    d["age_numeric"] = d["age"].map(AGE_BIN_MIDPOINT)
    required = ["decision", "decision_lag", "decision_opponent_lag", "delta_steps_lag", "risk_gamble_choice", "age_numeric"]
    d = d.dropna(subset=required)
    d["delta_steps_lag_c"] = d["delta_steps_lag"] - d["delta_steps_lag"].mean()
    d["age_c"] = d["age_numeric"] - d["age_numeric"].mean()
    first_round = df[df["round_number"] == 1].set_index("participant_id")["decision"]
    d["first_round_unsafe"] = d["participant_id"].map(first_round)
    return d


def fit_human_logit(analysed: pd.DataFrame) -> dict:
    formula = (
        "decision ~ C(max_private_risk) + first_round_unsafe "
        "+ decision_lag * decision_opponent_lag * delta_steps_lag_c "
        "+ C(sex) + age_c + C(nationality_group) + risk_gamble_choice"
    )
    model = smf.logit(formula, data=analysed)
    fit = model.fit(cov_type="cluster", cov_kwds={"groups": analysed["group_id"]}, disp=0)
    wanted = {
        "own_prev_unsafe": "decision_lag",
        "opponent_prev_unsafe": "decision_opponent_lag",
        "progress_gap_before": "delta_steps_lag_c",
        "first_round_unsafe": "first_round_unsafe",
    }
    out = {}
    for label, term in wanted.items():
        out[label] = {"coef": float(fit.params[term]), "se": float(fit.bse[term]), "p": float(fit.pvalues[term])}
    out["n_obs"] = int(fit.nobs)
    out["n_clusters"] = int(analysed["group_id"].nunique())
    out["n_participants"] = int(analysed["participant_id"].nunique())
    return out


def fig_s1(df: pd.DataFrame) -> dict:
    per_game = df.drop_duplicates("group_id")[["group_id", "num_rounds"]]
    return {"n_games": int(len(per_game)), "mean": float(per_game["num_rounds"].mean()), "median": float(per_game["num_rounds"].median()), "values": per_game["num_rounds"].tolist()}


def fig2c_winner_loser(df: pd.DataFrame) -> dict:
    # `won_race` is NOT constant across a participant's rows despite the README's
    # description: it reads 0 on every round except the final one, where it is set
    # to the true outcome (confirmed by inspecting individual participants' round
    # sequences). Use the final round's value as the participant's real outcome.
    final_round = df.sort_values("round_number").groupby("participant_id")["won_race"].last()
    mean_decision = df.groupby("participant_id")["decision"].mean()
    outcome = pd.DataFrame({"won_race": final_round, "mean_decision": mean_decision})
    winners = outcome.loc[outcome["won_race"] == 1, "mean_decision"]
    losers = outcome.loc[outcome["won_race"] == 0, "mean_decision"]
    return {
        "winner_mean": float(winners.mean()), "n_winners": int(len(winners)),
        "loser_mean": float(losers.mean()), "n_losers": int(len(losers)),
        "note": "won_race taken from each participant's final round_number row (see comment); the raw field is not constant per participant as the README states.",
    }


# --- 2. pull LLM player-level distributions from already-audited clean runs --------

NEUTRAL_INPUTS = {
    "gpt-5-nano": [
        "results/frontier/openai/baseline/gpt-5-nano",
        "results/frontier/openai/persona/R0_neutral/gpt-5-nano",
    ],
    "gpt-5.4-nano": [
        "results/frontier/openai/baseline/gpt-5.4-nano",
        "results/frontier/openai/persona/R0_neutral/gpt-5.4-nano",
    ],
    "google/gemini-3-flash-preview": [
        "results/frontier/baseline/google-gemini-3-flash-preview",
        "results/frontier/persona/R0_neutral/google-gemini-3-flash-preview",
        # api_5games_allrisk excluded: shares game_id/game_seed with 15 of the 30
        # baseline races (same realised horizon/stopping draws under this project's
        # CRN design), but is an independently-sampled re-run, not a duplicated log --
        # turn-by-turn actions differ in most risk-0.6/0.9 games (risk-0.1 matches
        # exactly only because that cell is ~100% Unsafe either way). Pooling would
        # violate CRN-independence; already flagged "superseded overlapping pilot" in
        # analyze_two_player_paper_figures.py for what is presumably this same reason.
    ],
    "google/gemini-3.1-flash-lite-preview": ["results/frontier/baseline/google-gemini-3.1-flash-lite-preview"],
    "google/gemini-3.5-flash-lite": ["results/frontier/baseline/google-gemini-3.5-flash-lite"],
}


def llm_horizons() -> list[int]:
    horizons: list[int] = []
    for dirs in NEUTRAL_INPUTS.values():
        for d in dirs:
            p = ROOT / d / "races.csv"
            if not p.exists():
                continue
            with open(p) as f:
                for row in csv.DictReader(f):
                    horizons.append(int(float(row["n_rounds"])))
    return horizons


def llm_winner_loser() -> dict:
    winner_rates, loser_rates = [], []
    for dirs in NEUTRAL_INPUTS.values():
        for d in dirs:
            p = ROOT / d / "players.csv"
            if not p.exists():
                continue
            with open(p) as f:
                for row in csv.DictReader(f):
                    rate = float(row["unsafe_frequency"])
                    if row["outcome"] == "winner":
                        winner_rates.append(rate)
                    elif row["outcome"] == "loser":
                        loser_rates.append(rate)
    return {
        "winner_mean": float(np.mean(winner_rates)), "n_winners": len(winner_rates),
        "loser_mean": float(np.mean(loser_rates)), "n_losers": len(loser_rates),
    }


# --- figures -------------------------------------------------------------------------


# NOTE: the distribution-overlay figure that used to be built here (a jittered
# dot-strip stacked below a human histogram on a shared axis) was hard to read
# -- the LLM rows' vertical position was fake jitter sharing an axis with a
# real percentage, which looked like negative values. It has been replaced by
# a small-multiples grid (one row per population, all non-negative, all
# sharing the same 0-100% axis) that also adds GPT-5.6 Luna/Terra; see
# build_human_vs_llm_distribution_v2.py, which is now the canonical generator
# for figures/human_vs_llm_distribution.png. Do not resurrect the old version
# here without also porting that fix.


def fig_dynamic_coefficients(human_fit: dict, llm_scorecard_coefs: dict) -> None:
    setup_plot()
    effects = [
        ("opponent_prev_unsafe", "Opponent's previous\naction was Unsafe"),
        ("progress_gap_before", "Own progress minus\nopponent's progress"),
        ("first_round_unsafe", "Chose Unsafe\nin round 1"),
        ("own_prev_unsafe", "Own previous\naction was Unsafe"),
    ]
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    rows = [MODEL_LABELS["gpt-5.4-nano"]] + ["Human (this refit)"]
    y_positions = {}
    y = 0
    yticks, yticklabels = [], []
    for effect_key, effect_label in effects:
        for source in ["human", "gpt-5.4-nano"]:
            if source == "human":
                c = human_fit[effect_key]
                color = PALETTE["slate"]
            else:
                c = llm_scorecard_coefs.get(effect_key)
                color = PALETTE["cyan"]
                if c is None:
                    continue
            ax.errorbar(c["coef"], y, xerr=1.96 * c["se"], fmt="o", color=color, markersize=7, capsize=4, linewidth=1.6)
            yticks.append(y)
            yticklabels.append(f"{'Human (this refit)' if source=='human' else 'GPT-5.4 nano'}")
            y -= 1
        y -= 0.6
    ax.axvline(0, color=PALETTE["slate"], linewidth=1, linestyle="--")
    ax.set_yticks(yticks, yticklabels, fontsize=8.5)
    ax2 = ax.twinx()
    group_centers = [-(i * 2.6 + 0.5) for i in range(len(effects))]
    ax2.set_yticks(group_centers)
    ax2.set_yticklabels([lbl for _, lbl in effects], fontsize=9)
    ax2.set_ylim(ax.get_ylim())
    ax2.tick_params(length=0)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Logistic-regression coefficient on choosing Unsafe (95% CI)")
    ax.set_title("Human dynamics, refit directly on raw participant data,\nversus the one LLM checkpoint where the same fit converges", pad=14)
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    save_figure(fig, FIGURES / "human_vs_llm_dynamic_coefficients")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)
    df = load_human()

    fig2a = fig2a_replication(df)
    analysed = build_analysed_sample(df)
    human_fit = fit_human_logit(analysed)
    s1 = fig_s1(df)
    c2c = fig2c_winner_loser(df)

    llm_hz = llm_horizons()
    llm_wl = llm_winner_loser()

    # gpt-5.4-nano is the only checkpoint whose Model-6-equivalent fit converges
    # (see results/cross_model_pilot_synthesis/INSIGHTS.md, Insight 3); pull its
    # coefficients straight from that lane's own analyzer output for the figure.
    gpt54_dir = OUTPUT / "_lane_outputs" / "gpt-5.4-nano"
    llm_coefs = {}
    hc_path = gpt54_dir / "human_comparison.csv"
    if hc_path.exists():
        with open(hc_path) as f:
            for row in csv.DictReader(f):
                name = row["name"]
                if row["llm_value"]:
                    llm_coefs[name] = {"coef": float(row["llm_value"]), "se": float(row["llm_se"]) if row["llm_se"] else 0.0}

    # figures/human_vs_llm_distribution.png is now built by
    # build_human_vs_llm_distribution_v2.py, not here -- see the note above
    # fig_distribution_overlay (removed).
    fig_dynamic_coefficients(human_fit, llm_coefs)

    validation = {
        "fig2a_n_by_risk": fig2a["n_by_risk"],
        "fig2a_mean_by_risk": fig2a["mean_by_risk"],
        "fig2a_contrasts": fig2a["contrasts"],
        "readme_expected_n_by_risk": {"0.1": 98, "0.6": 105, "0.9": 138},
        "readme_expected_d": {"0.1_vs_0.6": 0.341, "0.6_vs_0.9": -0.027, "0.1_vs_0.9": 0.322},
        "table1_analysed_sample": {
            "n_obs": human_fit["n_obs"], "n_clusters": human_fit["n_clusters"], "n_participants": human_fit["n_participants"],
        },
        "readme_expected_table1_sample": {"n_obs": 2888, "n_clusters": 172, "n_participants": 338},
        "table1_coefficients_this_refit": {k: v for k, v in human_fit.items() if k not in ("n_obs", "n_clusters", "n_participants")},
        "human_reference_json_values": {
            "opponent_prev_unsafe": 0.607, "progress_gap_before": -0.296, "first_round_unsafe": 0.217, "own_prev_unsafe": -0.193,
        },
        "fig_s1_rounds_per_game": {"n_games": s1["n_games"], "mean": s1["mean"], "median": s1["median"]},
        "llm_realised_horizon": {"n_races": len(llm_hz), "mean": float(np.mean(llm_hz)), "median": float(np.median(llm_hz))},
        "fig2c_winner_loser_human": c2c,
        "fig2c_winner_loser_llm_pooled": llm_wl,
    }
    with open(DATA / "human_vs_llm_validation.json", "w") as f:
        json.dump(validation, f, indent=2)

    print(json.dumps(validation, indent=2))


if __name__ == "__main__":
    main()
