#!/usr/bin/env python3
"""Does a checkpoint react to *who it is told the opponent is*, before the opponent acts?

The persona sweep analysed elsewhere in this synthesis varies a seat's own
risk-attitude framing (R1..R6). A second, previously unanalysed persona axis
exists in the same runs and asks a different question: each seat is framed as
either "adversarial" or "cooperative", and the four cells
(S_AA / S_AC / S_CA / S_CC) cross own framing with the *opponent's* framing.

That crossing makes a clean 2x2 factorial:

    own_adversarial x opponent_adversarial -> P(Unsafe)

The opponent-role main effect is the interesting one. The human study's
strongest dynamic effect (E1) is reciprocity -- humans condition on what the
opponent *did*. Here we can instead ask whether a model conditions on what it
was *told the opponent is*, which is a belief about disposition rather than a
response to behaviour. Restricting to round 1 sharpens this into a pure
prior-belief test: in round 1 no action has been observed by anyone, so any
opponent-role effect there cannot be reciprocity in the human sense.

Estimation mirrors the rest of this project: a logistic regression with
cluster-robust SEs, clustered on the CRN block (`rep`), since matched
repetitions share horizon/setback draws by design. Cells with no variance or
too few independent races stay descriptive -- a fitted coefficient on a
constant outcome is not evidence, and is reported as non-estimable instead.
"""

from __future__ import annotations

import csv
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

PALETTE = {
    "navy": "#0B132B", "slate": "#64748B", "grid": "#DCE3ED",
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a",
    "yellow": "#eda100", "red": "#e34948", "violet": "#4a3aa7", "magenta": "#e87ba4",
}
CELLS = ["S_AA_adv_adv", "S_AC_adv_coop", "S_CA_coop_adv", "S_CC_coop_coop"]
MODEL_DIRS = {
    "gpt-5-nano": "results/frontier/openai/persona/{cell}/gpt-5-nano",
    "gpt-5.4-nano": "results/frontier/openai/persona/{cell}/gpt-5.4-nano",
    "google/gemini-3-flash-preview": "results/frontier/persona/{cell}/google-gemini-3-flash-preview",
    "gpt-5.6-luna": "results/frontier/bedrock_mantle/luna/persona/{cell}/openai.gpt-5.6-luna",
    "gpt-5.6-terra": "results/frontier/bedrock_mantle/terra/persona/{cell}/openai.gpt-5.6-terra",
    "claude-opus-5": "results/frontier/bedrock/persona/{cell}/us.anthropic.claude-opus-5",
    "claude-sonnet-5": "results/frontier/bedrock/persona/{cell}/us.anthropic.claude-sonnet-5",
}
MODEL_ORDER = list(MODEL_DIRS)
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}
MODEL_COLORS = {
    "gpt-5-nano": PALETTE["blue"], "gpt-5.4-nano": PALETTE["orange"],
    "google/gemini-3-flash-preview": PALETTE["aqua"],
    "gpt-5.6-luna": PALETTE["yellow"], "gpt-5.6-terra": PALETTE["red"],
    "claude-opus-5": PALETTE["violet"], "claude-sonnet-5": PALETTE["magenta"],
}
MIN_RACES_FOR_INFERENCE = 5


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 12.5, "axes.labelsize": 10, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def load_turns(model: str) -> pd.DataFrame:
    """Per-decision rows tagged with own and opponent social framing."""
    rows = []
    for cell in CELLS:
        d = ROOT / MODEL_DIRS[model].format(cell=cell)
        turns_p, players_p = d / "turns.jsonl", d / "players.csv"
        if not turns_p.exists() or not players_p.exists():
            continue
        # players.csv carries persona_role per (game_id, player); turns.jsonl does not.
        role_of: dict[tuple[str, str], str] = {}
        with open(players_p) as f:
            for r in csv.DictReader(f):
                role_of[(r["game_id"], r["player"])] = r["persona_role"]
        with open(turns_p) as f:
            for line in f:
                rec = json.loads(line)
                own = role_of.get((rec["game_id"], rec["player"]))
                if own is None:
                    continue
                # The opponent is the other seat in the same race.
                opp_role = next(
                    (v for (g, pl), v in role_of.items() if g == rec["game_id"] and pl != rec["player"]),
                    None,
                )
                if opp_role is None:
                    continue
                rows.append({
                    "model": model, "cell": cell, "game_id": rec["game_id"], "rep": rec.get("rep"),
                    "round": rec["round"], "unsafe": int(rec["unsafe"]),
                    "own_adversarial": int(own == "adversarial"),
                    "opponent_adversarial": int(opp_role == "adversarial"),
                    "max_private_risk": float(rec["max_private_risk"]),
                })
    return pd.DataFrame(rows)


def cell_means(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["own_adversarial", "opponent_adversarial"])
    out = g["unsafe"].agg(["mean", "size"]).reset_index()
    out["n_races"] = g["game_id"].nunique().values
    return out


def fit_factorial(df: pd.DataFrame, label: str) -> dict:
    """Cluster-robust logit of unsafe on the own x opponent framing factorial."""
    n_races = df["game_id"].nunique()
    if df["unsafe"].nunique() < 2:
        return {"status": "non_estimable", "reason": "no variance in outcome", "n_races": int(n_races)}
    if n_races < MIN_RACES_FOR_INFERENCE:
        return {"status": "descriptive_only", "reason": f"only {n_races} independent races", "n_races": int(n_races)}
    # Cluster on rep where available (the CRN block); fall back to game_id.
    cluster = df["rep"] if df["rep"].notna().all() else df["game_id"]
    try:
        fit = smf.logit("unsafe ~ own_adversarial * opponent_adversarial", data=df).fit(
            disp=0, cov_type="cluster", cov_kwds={"groups": cluster.astype(str).values})
    except Exception as exc:  # separation / singular covariance
        return {"status": "non_estimable", "reason": f"{type(exc).__name__}: {exc}", "n_races": int(n_races)}
    conf = fit.conf_int()
    out = {"status": "ok", "n_obs": int(len(df)), "n_races": int(n_races), "n_clusters": int(cluster.nunique()),
           "terms": {}}
    for nm in fit.params.index:
        coef, se, p = float(fit.params[nm]), float(fit.bse[nm]), float(fit.pvalues[nm])
        # A separated cell yields an enormous coefficient with an enormous SE --
        # numerically "a fit", statistically meaningless. Flag rather than report.
        out["terms"][nm] = {
            "coef": coef, "se": se,
            "ci_low": float(conf.loc[nm, 0]), "ci_high": float(conf.loc[nm, 1]),
            "p_value": p,
            "unstable": bool(abs(coef) > 15 or se > 15),
        }
    if any(t["unstable"] for t in out["terms"].values()):
        out["status"] = "unstable_separation"
    return out


def fit_lpm(df: pd.DataFrame) -> dict:
    """Player-level linear probability model -- the estimate that survives separation.

    Several checkpoints play Unsafe exactly 0% of the time in the
    cooperative/cooperative cell, which is a real result but makes the logit
    perfectly separated (huge coefficients, meaningless SEs). Aggregating to
    one Unsafe *rate* per player and fitting OLS with cluster-robust SEs keeps
    a well-defined estimate, and its coefficients read directly in percentage
    points -- which is what the accompanying figure plots.
    """
    per_player = (df.groupby(["game_id", "rep", "own_adversarial", "opponent_adversarial"], dropna=False)["unsafe"]
                    .mean().reset_index(name="unsafe_rate"))
    n_races = per_player["game_id"].nunique()
    if n_races < MIN_RACES_FOR_INFERENCE or per_player["unsafe_rate"].nunique() < 2:
        return {"status": "descriptive_only", "n_races": int(n_races)}
    cluster = per_player["rep"] if per_player["rep"].notna().all() else per_player["game_id"]
    fit = smf.ols("unsafe_rate ~ own_adversarial * opponent_adversarial", data=per_player).fit(
        cov_type="cluster", cov_kwds={"groups": cluster.astype(str).values})
    conf = fit.conf_int()
    terms = {}
    for nm in fit.params.index:
        se_pp = float(fit.bse[nm]) * 100
        # When own framing alone determines every round-1 action, the model fits
        # perfectly and the residual variance is 0. statsmodels then divides ~0 by
        # ~0 and can return an arbitrarily small p-value for a coefficient that is
        # exactly zero. Report the estimate but refuse to call it significant.
        degenerate = se_pp < 1e-6
        terms[nm] = {
            "coef_pp": float(fit.params[nm]) * 100, "se_pp": se_pp,
            "ci_low_pp": float(conf.loc[nm, 0]) * 100, "ci_high_pp": float(conf.loc[nm, 1]) * 100,
            "p_value": None if degenerate else float(fit.pvalues[nm]),
            "degenerate_zero_variance": degenerate,
        }
    return {
        "status": "ok", "n_players": int(len(per_player)), "n_races": int(n_races),
        "n_clusters": int(cluster.nunique()), "terms": terms,
    }


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    all_results: dict[str, dict] = {}
    all_cells: list[pd.DataFrame] = []

    for model in MODEL_ORDER:
        df = load_turns(model)
        if df.empty:
            print(f"{model}: no data")
            continue
        cm = cell_means(df)
        cm.insert(0, "model", model)
        all_cells.append(cm)

        r1 = df[df["round"] == 1]
        res = {
            "n_obs": int(len(df)),
            "n_races": int(df["game_id"].nunique()),
            "cell_means": cm.drop(columns=["model"]).to_dict("records"),
            "all_rounds_fit": fit_factorial(df, model),
            "round1_only_fit": fit_factorial(r1, f"{model} r1"),
            "round1_cell_means": cell_means(r1).to_dict("records"),
            "lpm_all_rounds": fit_lpm(df),
            "lpm_round1": fit_lpm(r1),
        }
        all_results[model] = res

        print(f"\n=== {MODEL_LABELS[model]} === n_obs={res['n_obs']} n_races={res['n_races']}")
        for row in res["cell_means"]:
            own = "adv" if row["own_adversarial"] else "coop"
            opp = "adv" if row["opponent_adversarial"] else "coop"
            print(f"  own={own:4s} opp={opp:4s}  unsafe={row['mean']:.3f}  (n_decisions={row['size']}, n_races={row['n_races']})")
        for key in ("all_rounds_fit", "round1_only_fit"):
            f = res[key]
            if "terms" not in f:
                print(f"  {key}: {f['status']} ({f.get('reason')})")
                continue
            opp_t = f["terms"].get("opponent_adversarial", {})
            own_t = f["terms"].get("own_adversarial", {})
            flag = "" if f["status"] == "ok" else f"  [{f['status']}]"
            print(f"  {key}: own_adv coef={own_t.get('coef'):+.3f} (p={own_t.get('p_value'):.4f}) | "
                  f"opp_adv coef={opp_t.get('coef'):+.3f} (p={opp_t.get('p_value'):.4f}){flag}")
        for key in ("lpm_all_rounds", "lpm_round1"):
            f = res[key]
            if f.get("status") != "ok":
                print(f"  {key}: {f['status']} (n_races={f.get('n_races')})")
                continue
            own_t, opp_t = f["terms"]["own_adversarial"], f["terms"]["opponent_adversarial"]
            def _fmt(x):
                if x["degenerate_zero_variance"]:
                    return f"{x['coef_pp']:+.1f}pp (exact, zero residual variance)"
                return f"{x['coef_pp']:+.1f}pp [{x['ci_low_pp']:+.1f},{x['ci_high_pp']:+.1f}] p={x['p_value']:.4f}"
            print(f"  {key}: own_adv {_fmt(own_t)} | opp_adv {_fmt(opp_t)}")

    pd.concat(all_cells, ignore_index=True).to_csv(DATA / "social_persona_cell_means.csv", index=False)
    with open(DATA / "social_persona_axis.json", "w") as f:
        json.dump(all_results, f, indent=2)

    fig_social_axis(all_results)


def _lpm_term(results: dict, model: str, key: str, term: str) -> tuple[float, float, float]:
    """(coef_pp, half-width of 95% CI, p) for one LPM term, or NaNs if not estimable."""
    f = results[model].get(key, {})
    if f.get("status") != "ok" or term not in f.get("terms", {}):
        return float("nan"), float("nan"), float("nan")
    t = f["terms"][term]
    half = (t["ci_high_pp"] - t["ci_low_pp"]) / 2
    if not np.isfinite(half):  # degenerate/zero-variance fit: draw the point, no error bar
        half = 0.0
    p = t["p_value"]
    return t["coef_pp"], half, float("nan") if p is None else p


def fig_social_axis(results: dict) -> None:
    setup_plot()
    models = [m for m in MODEL_ORDER if m in results]
    xs = np.arange(len(models))
    fig, axes = plt.subplots(1, 2, figsize=(13.6, 5.8), sharey=True)

    # Panel 1: own framing -- a direct instruction effect, large everywhere.
    ax = axes[0]
    vals = [_lpm_term(results, m, "lpm_all_rounds", "own_adversarial") for m in models]
    ax.bar(xs, [v[0] for v in vals], yerr=[v[1] for v in vals], capsize=4,
           color=[MODEL_COLORS[m] for m in models], width=0.62,
           error_kw={"ecolor": PALETTE["navy"], "elinewidth": 1.1})
    for xi, (v, _, _) in zip(xs, vals):
        ax.text(xi, v + 2.5, f"{v:+.0f}", ha="center", va="bottom", fontsize=9.5,
                color=PALETTE["navy"], weight="bold")
    ax.set_title("Being told YOU are adversarial\n(vs cooperative)", fontsize=12)
    ax.set_ylabel("Change in Unsafe rate (percentage points)")

    # Panel 2: opponent framing, all rounds vs round 1 -- the actual finding.
    ax = axes[1]
    w = 0.36
    all_r = [_lpm_term(results, m, "lpm_all_rounds", "opponent_adversarial") for m in models]
    r1 = [_lpm_term(results, m, "lpm_round1", "opponent_adversarial") for m in models]
    ax.bar(xs - w / 2, [v[0] for v in all_r], yerr=[v[1] for v in all_r], capsize=4, width=w,
           color=[MODEL_COLORS[m] for m in models], label="All rounds",
           error_kw={"ecolor": PALETTE["navy"], "elinewidth": 1.1})
    ax.bar(xs + w / 2, [v[0] for v in r1], yerr=[v[1] for v in r1], capsize=4, width=w,
           color="white", edgecolor=[MODEL_COLORS[m] for m in models], linewidth=1.8, hatch="///",
           label="Round 1 only (no action seen yet)",
           error_kw={"ecolor": PALETTE["navy"], "elinewidth": 1.1})
    for xi, (v, _, _) in zip(xs, all_r):
        ax.text(xi - w / 2, v + 2.5, f"{v:+.0f}", ha="center", va="bottom", fontsize=9.5,
                color=PALETTE["navy"], weight="bold")
    ax.set_title("Being told the OPPONENT is adversarial\n(vs cooperative)", fontsize=12)
    ax.legend(frameon=False, fontsize=8.8, loc="upper left")

    for ax in axes:
        ax.axhline(0, color=PALETTE["navy"], linewidth=1.0)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.7)
        ax.set_xticks(xs, [MODEL_LABELS[m] for m in models], rotation=18, ha="right", fontsize=9)
    axes[0].set_ylim(-12, 112)

    fig.suptitle("A model obeys its own assigned disposition immediately — but only reacts to the opponent's after seeing it act",
                  fontsize=13, y=1.005)
    fig.tight_layout()
    fig.savefig(FIGURES / "social_persona_axis.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(FIGURES / "social_persona_axis.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("\nwrote", FIGURES / "social_persona_axis.png")


if __name__ == "__main__":
    main()
