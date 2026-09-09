#!/usr/bin/env python3
"""A formal statistical test for cross-model heterogeneity, replacing "it looks
different" with a likelihood-ratio test.

Fits nested logits on the pooled neutral-lane data of a named checkpoint roster
(round >= 1, since the risk main effect doesn't need the round>=2 lag features):
  A (restricted): unsafe ~ C(max_private_risk)                      -- one common risk effect
  B (unrestricted): unsafe ~ C(max_private_risk) * C(model)          -- model-specific risk effects
  C (level only):  unsafe ~ C(model)                                  -- model differences in level alone
LR = 2*(loglik_B - loglik_A) ~ chi2(df_B - df_A) tests whether letting the risk
effect differ by model improves fit beyond what model-level differences alone
would predict. This does not use cluster-robust SEs (the LR test compares two
maximum-likelihood fits on the same data; a naive LR chi-square is a standard,
simple heterogeneity screen here, not a claim about individual coefficient
significance -- those are already reported with cluster-robust SEs elsewhere).

The roster is selectable so that every historically reported version of this
test is regenerable:

  --roster nine   (default) all nine neutral-lane checkpoints
  --roster seven  the nine minus the two Claude checkpoints
  --roster five   the five original checkpoints (nine minus Claude, minus GPT-5.6)
  --roster all    run every named roster above in one pass

Degrees of freedom, for reference, with K models and 3 risk levels: model A has
3 parameters, model C has K, model B has 3K. So B vs A has 3K-3 df, C vs
intercept has K-1 df, and B vs C has 3K-K = 2K df (the interaction alone is
(K-1)(3-1) df, but the B-vs-C comparison additionally frees the two risk main
effects, hence 2K rather than 2K-2).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import warnings
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import scipy
import statsmodels
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
OUT_PATH = DATA / "cross_model_heterogeneity_test.json"

# ---------------------------------------------------------------------------
# Neutral-lane input directories, declared once per checkpoint.
# ---------------------------------------------------------------------------
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

# ---------------------------------------------------------------------------
# Named rosters. Membership is declared here and nowhere else, and is copied
# verbatim into the output artifact.
# ---------------------------------------------------------------------------
_FIVE = [
    "gpt-5-nano",
    "gpt-5.4-nano",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.5-flash-lite",
]
_SEVEN = _FIVE + ["gpt-5.6-luna", "gpt-5.6-terra"]
_NINE = _SEVEN + ["claude-opus-5", "claude-sonnet-5"]

ROSTERS: dict[str, dict] = {
    "five": {
        "models": _FIVE,
        "description": "The five original LLM checkpoints: two OpenAI nano and three Google Gemini. "
                       "Excludes the GPT-5.6 (Luna/Terra) generation and both Claude checkpoints.",
    },
    "seven": {
        "models": _SEVEN,
        "description": "The five original checkpoints plus the GPT-5.6 generation (Luna, Terra). "
                       "Excludes both Claude checkpoints.",
    },
    "nine": {
        "models": _NINE,
        "description": "All nine neutral-lane checkpoints, including Claude Opus 5 and Claude Sonnet 5. "
                       "This is the default and the currently published roster.",
    },
}
DEFAULT_ROSTER = "nine"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load(models: list[str]) -> tuple[pd.DataFrame, dict[str, str]]:
    rows = []
    sources: dict[str, str] = {}
    for model in models:
        for d in NEUTRAL_INPUTS[model]:
            subdir = MODEL_SUBDIR.get(model, "")
            path = ROOT / d / subdir / "turns.jsonl"
            sources[str(Path(d) / subdir / "turns.jsonl").replace("\\", "/")] = sha256(path)
            with open(path) as f:
                for line in f:
                    r = json.loads(line)
                    rows.append({
                        "model": model,
                        "max_private_risk": r["max_private_risk"],
                        "unsafe": r["unsafe"],
                        "race_id": f"{model}|{r['game_id']}",
                    })
    return pd.DataFrame(rows), sources


def lr_test(loglik_restricted: float, loglik_full: float, df_restricted: int, df_full: int) -> dict:
    lr_stat = 2 * (loglik_full - loglik_restricted)
    df = df_full - df_restricted
    p = float(stats.chi2.sf(lr_stat, df))
    return {"lr_stat": float(lr_stat), "df": int(df), "p_value": p}


def fit_logit(formula: str, data: pd.DataFrame) -> tuple[object, list[str]]:
    """Fit and capture any convergence warning verbatim."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = smf.logit(formula, data=data).fit(disp=0)
        messages = [f"{w.category.__name__}: {w.message}" for w in caught]
    return fit, messages


def run_roster(name: str) -> dict:
    spec = ROSTERS[name]
    models = spec["models"]
    df, sources = load(models)
    print(f"\n=== roster {name!r}: {len(models)} models, {len(df)} decisions, "
          f"{df['race_id'].nunique()} races ===")

    fit_a, warn_a = fit_logit("unsafe ~ C(max_private_risk)", df)
    fit_b, warn_b = fit_logit("unsafe ~ C(max_private_risk) * C(model)", df)
    fit_c, warn_c = fit_logit("unsafe ~ C(model)", df)
    fit_intercept, warn_i = fit_logit("unsafe ~ 1", df)

    result_b_vs_a = lr_test(fit_a.llf, fit_b.llf, fit_a.df_model + 1, fit_b.df_model + 1)
    result_c_vs_intercept = lr_test(fit_intercept.llf, fit_c.llf, 1, fit_c.df_model + 1)
    result_b_vs_c = lr_test(fit_c.llf, fit_b.llf, fit_c.df_model + 1, fit_b.df_model + 1)

    print("Model A (unsafe ~ risk only): loglik=", fit_a.llf, "df=", fit_a.df_model + 1, "pseudo-R2=", fit_a.prsquared)
    print("Model B (unsafe ~ risk * model): loglik=", fit_b.llf, "df=", fit_b.df_model + 1, "pseudo-R2=", fit_b.prsquared)
    print("Model C (unsafe ~ model only): loglik=", fit_c.llf, "df=", fit_c.df_model + 1, "pseudo-R2=", fit_c.prsquared)
    print("LR B vs A (model-specific risk slope beats common slope):", result_b_vs_a)
    print("LR C vs intercept (models differ in level at all):", result_c_vs_intercept)
    print("LR B vs C (risk-response SHAPE differs beyond level):", result_b_vs_c)
    for label, msgs in (("A", warn_a), ("B", warn_b), ("C", warn_c), ("intercept", warn_i)):
        for m in msgs:
            print(f"  warning[{label}]: {m}")

    return {
        "roster": name,
        "roster_description": spec["description"],
        "models": list(models),
        "n_decisions": int(len(df)),
        "n_models": int(df["model"].nunique()),
        "n_races": int(df["race_id"].nunique()),
        "n_risk_levels": int(df["max_private_risk"].nunique()),
        "n_decisions_per_model": {k: int(v) for k, v in df["model"].value_counts().sort_index().items()},
        "loglik_A_risk_only": float(fit_a.llf), "loglik_B_risk_x_model": float(fit_b.llf),
        "loglik_C_model_only": float(fit_c.llf), "loglik_intercept_only": float(fit_intercept.llf),
        "pseudo_r2_A": float(fit_a.prsquared), "pseudo_r2_B": float(fit_b.prsquared), "pseudo_r2_C": float(fit_c.prsquared),
        "n_params_A": int(fit_a.df_model + 1), "n_params_B": int(fit_b.df_model + 1),
        "n_params_C": int(fit_c.df_model + 1), "n_params_intercept": 1,
        "lr_test_B_vs_A_slope_heterogeneity_given_common_slope_baseline": result_b_vs_a,
        "lr_test_C_vs_intercept_level_heterogeneity": result_c_vs_intercept,
        "lr_test_B_vs_C_slope_heterogeneity_beyond_level": result_b_vs_c,
        "converged": {"A": bool(fit_a.mle_retvals["converged"]), "B": bool(fit_b.mle_retvals["converged"]),
                      "C": bool(fit_c.mle_retvals["converged"]), "intercept": bool(fit_intercept.mle_retvals["converged"])},
        "fit_warnings": {"A": warn_a, "B": warn_b, "C": warn_c, "intercept": warn_i},
        "source_sha256": sources,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--roster", default=DEFAULT_ROSTER, choices=sorted(ROSTERS) + ["all"],
                    help=f"named checkpoint roster to fit (default: {DEFAULT_ROSTER})")
    ap.add_argument("--out", default=str(OUT_PATH), help="output JSON path")
    args = ap.parse_args()

    names = sorted(ROSTERS, key=lambda n: len(ROSTERS[n]["models"])) if args.roster == "all" else [args.roster]

    out_path = Path(args.out)
    existing = {}
    if out_path.exists():
        try:
            existing = json.loads(out_path.read_text())
        except json.JSONDecodeError:
            existing = {}

    rosters_out = dict(existing.get("rosters", {}))
    for name in names:
        rosters_out[name] = run_roster(name)

    # Backward compatibility: the top level keeps the shape of the original
    # single-roster artifact, for whichever roster is the published default.
    flat_source = rosters_out.get(DEFAULT_ROSTER) or rosters_out[names[-1]]
    results = {
        "default_roster": DEFAULT_ROSTER,
        "n_decisions": flat_source["n_decisions"],
        "n_models": flat_source["n_models"],
        "loglik_A_risk_only": flat_source["loglik_A_risk_only"],
        "loglik_B_risk_x_model": flat_source["loglik_B_risk_x_model"],
        "loglik_C_model_only": flat_source["loglik_C_model_only"],
        "loglik_intercept_only": flat_source["loglik_intercept_only"],
        "pseudo_r2_A": flat_source["pseudo_r2_A"],
        "pseudo_r2_B": flat_source["pseudo_r2_B"],
        "pseudo_r2_C": flat_source["pseudo_r2_C"],
        "lr_test_B_vs_A_slope_heterogeneity_given_common_slope_baseline":
            flat_source["lr_test_B_vs_A_slope_heterogeneity_given_common_slope_baseline"],
        "lr_test_C_vs_intercept_level_heterogeneity":
            flat_source["lr_test_C_vs_intercept_level_heterogeneity"],
        "lr_test_B_vs_C_slope_heterogeneity_beyond_level":
            flat_source["lr_test_B_vs_C_slope_heterogeneity_beyond_level"],
        "roster_definitions": {k: {"models": v["models"], "description": v["description"]} for k, v in ROSTERS.items()},
        "rosters": rosters_out,
        "provenance": {
            "script": "results/cross_model_pilot_synthesis/analyze_heterogeneity_test.py",
            "script_sha256": sha256(Path(__file__).resolve()),
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "seed": None,
            "seed_note": "Deterministic: Newton/BFGS maximum-likelihood fits on a fixed table, no randomness.",
            "python_version": platform.python_version(),
            "statsmodels_version": statsmodels.__version__,
            "scipy_version": scipy.__version__,
            "pandas_version": pd.__version__,
            "rosters_run_this_pass": names,
        },
    }
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", out_path)


if __name__ == "__main__":
    main()
