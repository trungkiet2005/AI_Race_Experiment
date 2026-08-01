"""Rigorous statistical layer for the N-player pilot runs.

Extends ``analyze_nplayer_results.py``'s descriptive pass with the pieces
that make ``analysis/frontier/`` (2-player) more than a means-and-proportions
dashboard: nearest-strategy classification, a cluster-robust panel logistic
regression, a theory-fit search over the selection intensity beta, and
Wilson confidence intervals on every reported proportion. Mirrors, wherever
the N-player schema allows it, ``results/scripts/analyze_ai_race.py``'s
``_fit_clustered_logit``/``_build_theory_comparison`` and
``strategy_analysis/classify.py``'s nearest-strategy classifier -- see each
section below for exactly what differs and why.

Writes CSV/JSON outputs to ``analysis/nplayer/derived/`` (mirroring
``analysis/frontier/derived/``'s role: derived tables consumed by the
dashboard and report, not the raw run data itself, which stays under
``results/nplayer/``).

Scope, deliberately: only the neutral baseline (``nplayer_nonpersona/``, 60
races, 1584 turns) supports inferential statistics here. The persona sweep
(``nplayer-riskaware/``, 2 reps/cell) is far too small -- most persona cells
have zero variance in the outcome (0% or 100% Unsafe), which would make a
logistic fit either fail outright (perfect separation) or report a
meaningless coefficient. Persona stays strictly descriptive, as already
argued in ``ANALYSIS_qwen2.5-14b-instruct.md``.

Usage::

    python results/nplayer/rigorous_analysis.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent
DERIVED_DIR = REPO_ROOT / "analysis" / "nplayer" / "derived"
sys.path.insert(0, str(REPO_ROOT / "N-Player"))
sys.path.insert(0, str(RESULTS_DIR))
sys.path.insert(0, str(REPO_ROOT))

from analyze_nplayer_results import load_all, risk_from_game_id  # noqa: E402
from strategy_analysis.classify_nplayer import classify_trajectory_nplayer  # noqa: E402

RISKS: tuple[float, ...] = (0.1, 0.6, 0.9)

# ---------------------------------------------------------------------------
# 1. Nearest-strategy classification (mirrors strategy_analysis/classify.py,
#    applied to real turns instead of a synthetic dataset).
# ---------------------------------------------------------------------------


def classify_all_races(turns: pd.DataFrame) -> pd.DataFrame:
    """One row per (game_id, player): nearest AS/AU/CS strategy by Hamming distance.

    ``unique_best_strategy`` is ``None`` on a tie (kept, not broken -- same
    convention as the two-player classifier).
    """

    rows: list[dict[str, Any]] = []
    for game_id, race_turns in turns.groupby("game_id"):
        pivot = race_turns.sort_values("round").pivot(
            index="round", columns="player", values="unsafe"
        )
        seat_names = list(pivot.columns)
        risk = float(race_turns["risk"].iloc[0])
        persona_condition = race_turns["persona_condition"].iloc[0]
        for player in seat_names:
            own = tuple(int(v) for v in pivot[player].tolist())
            others = [
                tuple(int(v) for v in pivot[other].tolist())
                for other in seat_names
                if other != player
            ]
            result = classify_trajectory_nplayer(own, others)
            rows.append(
                {
                    "game_id": game_id,
                    "player": player,
                    "risk": risk,
                    "persona_condition": persona_condition,
                    "horizon": result.horizon,
                    "best_strategies": "|".join(result.best_strategies),
                    "unique_best_strategy": result.unique_best_strategy,
                    "n_tied": len(result.best_strategies),
                }
            )
    return pd.DataFrame(rows)


def strategy_mix_table(classifications: pd.DataFrame, *, group_cols: list[str]) -> pd.DataFrame:
    """Share of player-races nearest each strategy, "Tie/Other" for ties.

    Matches ``analysis/frontier/visualizations``' chart-5 category set
    (``AS, AU, CS, Tie/Other`` -- no CAS here, since the N-player engine and
    its source paper appendix do not define one).
    """

    working = classifications.copy()
    working["category"] = working["unique_best_strategy"].fillna("Tie/Other")
    categories = ["AS", "AU", "CS", "Tie/Other"]
    counts = (
        working.groupby(group_cols + ["category"])
        .size()
        .unstack("category", fill_value=0)
        .reindex(columns=categories, fill_value=0)
    )
    totals = counts.sum(axis=1)
    shares = counts.div(totals, axis=0)
    shares["total"] = totals
    return shares.reset_index()


# ---------------------------------------------------------------------------
# 2. Cluster-robust panel logistic regression on the baseline (mirrors
#    results/scripts/analyze_ai_race.py::_fit_clustered_logit).
# ---------------------------------------------------------------------------

# N-player analogue of the two-player's 6 nested specs: `opponent_prev_unsafe`
# (a single opponent) has no direct counterpart with >1 co-player, so it is
# replaced by `others_prev_unsafe_fraction` (share of co-players Unsafe last
# round); `progress_gap_before` keeps its meaning unchanged (own minus the
# *leading* co-player, already computed by the N-player engine).
NPLAYER_LOGIT_SPECIFICATIONS: tuple[tuple[str, str], ...] = (
    ("1", "unsafe ~ C(max_private_risk)"),
    (
        "2",
        "unsafe ~ C(max_private_risk) + own_prev_unsafe + "
        "others_prev_unsafe_fraction + progress_gap_before",
    ),
    (
        "3",
        "unsafe ~ C(max_private_risk) + own_prev_unsafe * "
        "others_prev_unsafe_fraction * progress_gap_before",
    ),
    ("4", "unsafe ~ C(max_private_risk) + first_round_unsafe"),
    (
        "5",
        "unsafe ~ C(max_private_risk) + first_round_unsafe + own_prev_unsafe + "
        "others_prev_unsafe_fraction + progress_gap_before",
    ),
    (
        "6",
        "unsafe ~ C(max_private_risk) + first_round_unsafe + own_prev_unsafe * "
        "others_prev_unsafe_fraction * progress_gap_before",
    ),
)


def build_logit_model_data(baseline_turns: pd.DataFrame) -> pd.DataFrame:
    """Round >= 2 estimation sample with the lag/position covariates.

    Cluster key is ``rep``: the neutral baseline's 20 repetitions are the
    CRN block here (same role as the two-player analyser's
    ``source_run::model::rep``) -- verified in
    ``ANALYSIS_qwen2.5-14b-instruct.md``'s CRN check that every repetition's
    realised horizon is identical across risk treatments for a fixed rep.
    """

    first_round = (
        baseline_turns.loc[baseline_turns["round"] == 1, ["game_id", "player", "unsafe"]]
        .rename(columns={"unsafe": "first_round_unsafe"})
    )
    merged = baseline_turns.merge(first_round, on=["game_id", "player"], how="left")
    model_data = merged.loc[merged["round"] >= 2].copy()
    model_data["unsafe"] = model_data["unsafe"].astype(int)
    model_data["own_prev_unsafe"] = (model_data["own_prev_action"] == "unsafe").astype(int)
    model_data["others_prev_unsafe_fraction"] = model_data["others_prev_actions"].map(
        lambda actions: sum(1 for a in actions if a == "unsafe") / len(actions)
    )
    model_data["first_round_unsafe"] = model_data["first_round_unsafe"].astype(int)
    return model_data


def fit_clustered_logit(model_data: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Fit the 6 nested specs, cluster-robust SE by ``rep``.

    A spec that cannot be estimated (here: any spec containing
    ``first_round_unsafe``, which is a constant in this sample -- see below)
    is recorded as ``not_estimable`` with a concrete reason rather than
    silently dropped or allowed to crash the whole run.
    """

    import statsmodels.formula.api as smf
    from statsmodels.tools.sm_exceptions import PerfectSeparationError

    coefficient_frames: list[pd.DataFrame] = []
    spec_metadata: dict[str, Any] = {}

    variance = model_data["first_round_unsafe"].var()
    first_round_is_constant = bool(variance == 0.0)

    for specification, formula in NPLAYER_LOGIT_SPECIFICATIONS:
        if first_round_is_constant and "first_round_unsafe" in formula:
            spec_metadata[specification] = {
                "formula": formula,
                "converged": False,
                "not_estimable_reason": (
                    "first_round_unsafe has zero variance in this sample "
                    "(every recorded race opens with UNSAFE, "
                    f"first_round_unsafe == 1 for all {len(model_data.drop_duplicates(['game_id','player']))} "
                    "player-races) -- perfectly collinear with the intercept."
                ),
            }
            continue
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("error")
                result = smf.logit(formula, data=model_data).fit(
                    disp=False,
                    cov_type="cluster",
                    cov_kwds={"groups": model_data["rep"]},
                )
        except (PerfectSeparationError, np.linalg.LinAlgError, Warning) as exc:
            spec_metadata[specification] = {
                "formula": formula,
                "converged": False,
                "not_estimable_reason": f"{type(exc).__name__}: {exc}",
            }
            continue

        confidence = result.conf_int()
        coefficients = pd.DataFrame(
            {
                "specification": specification,
                "term": result.params.index,
                "coefficient": result.params.values,
                "cluster_robust_se": result.bse.values,
                "z": result.tvalues.values,
                "p_value": result.pvalues.values,
                "ci_95_low": confidence.iloc[:, 0].values,
                "ci_95_high": confidence.iloc[:, 1].values,
            }
        )
        coefficients["odds_ratio"] = np.exp(coefficients["coefficient"])
        coefficient_frames.append(coefficients)
        spec_metadata[specification] = {
            "formula": formula,
            "converged": bool(result.mle_retvals.get("converged", True)),
            "log_likelihood": float(result.llf),
            "pseudo_r_squared": float(result.prsquared),
        }

    coefficients_table = (
        pd.concat(coefficient_frames, ignore_index=True)
        if coefficient_frames
        else pd.DataFrame(
            columns=[
                "specification", "term", "coefficient", "cluster_robust_se",
                "z", "p_value", "ci_95_low", "ci_95_high", "odds_ratio",
            ]
        )
    )
    metadata = {
        "n_observations": int(len(model_data)),
        "n_clusters": int(model_data["rep"].nunique()),
        "cluster_key": "rep",
        "cluster_rationale": (
            "Each repetition shares the same game_seed-derived horizon and "
            "setback draws across all three risk treatments (common random "
            "numbers) -- verified directly in this run's data; rep is the "
            "unit that must be treated as non-independent."
        ),
        "first_round_unsafe_is_constant": first_round_is_constant,
        "specifications": spec_metadata,
    }
    return coefficients_table, metadata


# ---------------------------------------------------------------------------
# 3. Theory-fit: search over beta (small-mutation limit) for the closest
#    joint match to the empirical baseline unsafe rate across all 3
#    treatments (mirrors, and extends, analyze_ai_race.py's
#    _build_theory_comparison -- that function checks two fixed (beta, mu)
#    points; mu is not modelled at all here (nor is it in the two-player
#    small_mutation_stationary), consistent with that module's own documented
#    scope limit).
# ---------------------------------------------------------------------------

THEORY_MECHANISM_PARAMS = dict(n=3, s=1.5, b=4.0, c=1.0, B=100.0, W=9.0, z=100)


def theory_beta_fit(empirical_by_risk: dict[float, float]) -> pd.DataFrame:
    from theory.stationary import au_frequency

    betas = np.logspace(-4, 1, 200)
    rows = []
    for beta in betas:
        predictions = {
            risk: au_frequency(pr=risk, beta=float(beta), **THEORY_MECHANISM_PARAMS)
            for risk in RISKS
        }
        sse = sum((predictions[risk] - empirical_by_risk[risk]) ** 2 for risk in RISKS)
        rows.append({"beta": float(beta), "sse": float(sse), **{
            f"predicted_{risk}": predictions[risk] for risk in RISKS
        }})
    table = pd.DataFrame(rows).sort_values("sse").reset_index(drop=True)
    return table


# ---------------------------------------------------------------------------
# 4. Wilson confidence intervals for every reported proportion.
# ---------------------------------------------------------------------------


def wilson_ci_table(turns: pd.DataFrame, *, group_cols: list[str], outcome: str = "unsafe") -> pd.DataFrame:
    from statsmodels.stats.proportion import proportion_confint

    grouped = turns.groupby(group_cols)[outcome].agg(["sum", "count"]).reset_index()
    grouped = grouped.rename(columns={"sum": "successes", "count": "n"})
    low, high = proportion_confint(grouped["successes"], grouped["n"], method="wilson")
    grouped["proportion"] = grouped["successes"] / grouped["n"]
    grouped["ci_95_low"] = low
    grouped["ci_95_high"] = high
    return grouped


def main() -> None:
    races, players, turns = load_all()
    for df in (races, players, turns):
        df["risk"] = df["game_id"].map(risk_from_game_id)
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 70, "\n1. STRATEGY CLASSIFICATION (AS/AU/CS, Hamming distance)\n", "=" * 70, sep="")
    classifications = classify_all_races(turns)
    classifications.to_csv(DERIVED_DIR / "strategy_classifications.csv", index=False)
    baseline_mask = classifications["persona_condition"] == "none"
    baseline_mix = strategy_mix_table(classifications[baseline_mask], group_cols=["risk"])
    baseline_mix.to_csv(DERIVED_DIR / "strategy_mix_baseline.csv", index=False)
    print(baseline_mix.to_string(index=False))

    persona_mix = strategy_mix_table(
        classifications[~baseline_mask], group_cols=["persona_condition", "risk"]
    )
    persona_mix.to_csv(DERIVED_DIR / "strategy_mix_persona.csv", index=False)

    print("\n" + "=" * 70, "\n2. CLUSTER-ROBUST LOGISTIC REGRESSION (baseline, round >= 2)\n", "=" * 70, sep="")
    baseline_turns = turns[turns["persona_condition"] == "none"].copy()
    model_data = build_logit_model_data(baseline_turns)
    coefficients_table, logit_metadata = fit_clustered_logit(model_data)
    coefficients_table.to_csv(DERIVED_DIR / "clustered_logit_coefficients.csv", index=False)
    (DERIVED_DIR / "clustered_logit_metadata.json").write_text(
        json.dumps(logit_metadata, indent=2, ensure_ascii=False)
    )
    print(f"n_observations={logit_metadata['n_observations']}, n_clusters={logit_metadata['n_clusters']}")
    for spec, meta in logit_metadata["specifications"].items():
        status = "OK" if meta.get("converged") else f"NOT ESTIMABLE ({meta.get('not_estimable_reason')})"
        print(f"  spec {spec}: {status}")
    if not coefficients_table.empty:
        print(coefficients_table[coefficients_table["specification"] == "3"].to_string(index=False))

    print("\n" + "=" * 70, "\n3. THEORY-FIT: search over beta (small-mutation limit)\n", "=" * 70, sep="")
    empirical_by_risk = {
        risk: float(baseline_turns.loc[baseline_turns["risk"] == risk, "unsafe"].mean())
        for risk in RISKS
    }
    beta_fit = theory_beta_fit(empirical_by_risk)
    beta_fit.to_csv(DERIVED_DIR / "theory_beta_fit.csv", index=False)
    best = beta_fit.iloc[0]
    print(f"best-fit beta = {best['beta']:.5f} (SSE = {best['sse']:.4f})")
    for risk in RISKS:
        print(
            f"  pr={risk}: theory={best[f'predicted_{risk}']:.3f}  "
            f"empirical={empirical_by_risk[risk]:.3f}"
        )
    (DERIVED_DIR / "theory_beta_fit_metadata.json").write_text(
        json.dumps(
            {
                "mechanism_params": THEORY_MECHANISM_PARAMS,
                "mutation_regime": "small_mutation_limit",
                "mutation_regime_caveat": (
                    "The finite-mutation rate mu is not applied anywhere in "
                    "N-Player/theory (same documented scope limit as "
                    "ai_race/theory/evolution.py) -- only the selection "
                    "intensity beta is searched here."
                ),
                "best_fit_beta": float(best["beta"]),
                "best_fit_sse": float(best["sse"]),
                "empirical_unsafe_rate_by_risk": empirical_by_risk,
                "search_grid": "logspace(-4, 1, 200)",
            },
            indent=2,
        )
    )

    print("\n" + "=" * 70, "\n4. WILSON 95% CI for unsafe rate by risk x condition\n", "=" * 70, sep="")
    conditions = turns.copy()
    conditions["condition"] = conditions["persona_condition"].where(
        conditions["persona_condition"] != "none", "baseline"
    )
    ci_table = wilson_ci_table(conditions, group_cols=["condition", "risk"])
    ci_table.to_csv(DERIVED_DIR / "unsafe_rate_ci.csv", index=False)
    print(ci_table.to_string(index=False))

    print(f"\nWrote derived tables to {DERIVED_DIR}")


if __name__ == "__main__":
    main()
