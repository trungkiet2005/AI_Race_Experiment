#!/usr/bin/env python3
"""A formal statistical test for cross-model heterogeneity, replacing "it looks
different" with a likelihood-ratio test.

Fits nested logits on the pooled 7-checkpoint neutral-lane data (round >= 1,
since the risk main effect doesn't need the round>=2 lag features):
  A (restricted): unsafe ~ C(max_private_risk)                      -- one common risk effect
  B (unrestricted): unsafe ~ C(max_private_risk) * C(model)          -- model-specific risk effects
  C (level only):  unsafe ~ C(model)                                  -- model differences in level alone
LR = 2*(loglik_B - loglik_A) ~ chi2(df_B - df_A) tests whether letting the risk
effect differ by model improves fit beyond what model-level differences alone
would predict. This does not use cluster-robust SEs (the LR test compares two
maximum-likelihood fits on the same data; a naive LR chi-square is a standard,
simple heterogeneity screen here, not a claim about individual coefficient
significance -- those are already reported with cluster-robust SEs elsewhere).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
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


def load() -> pd.DataFrame:
    rows = []
    for model, dirs in NEUTRAL_INPUTS.items():
        for d in dirs:
            subdir = MODEL_SUBDIR.get(model, "")
            with open(ROOT / d / subdir / "turns.jsonl") as f:
                for line in f:
                    r = json.loads(line)
                    rows.append({"model": model, "max_private_risk": r["max_private_risk"], "unsafe": r["unsafe"]})
    return pd.DataFrame(rows)


def lr_test(loglik_restricted: float, loglik_full: float, df_restricted: int, df_full: int) -> dict:
    lr_stat = 2 * (loglik_full - loglik_restricted)
    df = df_full - df_restricted
    p = float(stats.chi2.sf(lr_stat, df))
    return {"lr_stat": float(lr_stat), "df": int(df), "p_value": p}


def main() -> None:
    df = load()
    print("n decisions:", len(df), "models:", df["model"].nunique())

    fit_a = smf.logit("unsafe ~ C(max_private_risk)", data=df).fit(disp=0)
    fit_b = smf.logit("unsafe ~ C(max_private_risk) * C(model)", data=df).fit(disp=0)
    fit_c = smf.logit("unsafe ~ C(model)", data=df).fit(disp=0)
    fit_intercept = smf.logit("unsafe ~ 1", data=df).fit(disp=0)

    result_b_vs_a = lr_test(fit_a.llf, fit_b.llf, fit_a.df_model + 1, fit_b.df_model + 1)
    result_c_vs_intercept = lr_test(fit_intercept.llf, fit_c.llf, 1, fit_c.df_model + 1)
    result_b_vs_c = lr_test(fit_c.llf, fit_b.llf, fit_c.df_model + 1, fit_b.df_model + 1)

    print("\nModel A (unsafe ~ risk only): loglik=", fit_a.llf, "df=", fit_a.df_model + 1, "pseudo-R2=", fit_a.prsquared)
    print("Model B (unsafe ~ risk * model): loglik=", fit_b.llf, "df=", fit_b.df_model + 1, "pseudo-R2=", fit_b.prsquared)
    print("Model C (unsafe ~ model only): loglik=", fit_c.llf, "df=", fit_c.df_model + 1, "pseudo-R2=", fit_c.prsquared)
    print("\nLR test, does model-specific risk-slope (B) beat common risk-slope (A):", result_b_vs_a)
    print("LR test, does model (C) beat intercept-only (do models differ in level at all):", result_c_vs_intercept)
    print("LR test, does model-specific risk-slope (B) beat model-level-only (C) -- i.e. heterogeneity in SLOPE beyond level:", result_b_vs_c)

    results = {
        "n_decisions": len(df),
        "n_models": int(df["model"].nunique()),
        "loglik_A_risk_only": float(fit_a.llf), "loglik_B_risk_x_model": float(fit_b.llf),
        "loglik_C_model_only": float(fit_c.llf), "loglik_intercept_only": float(fit_intercept.llf),
        "pseudo_r2_A": float(fit_a.prsquared), "pseudo_r2_B": float(fit_b.prsquared), "pseudo_r2_C": float(fit_c.prsquared),
        "lr_test_B_vs_A_slope_heterogeneity_given_common_slope_baseline": result_b_vs_a,
        "lr_test_C_vs_intercept_level_heterogeneity": result_c_vs_intercept,
        "lr_test_B_vs_C_slope_heterogeneity_beyond_level": result_b_vs_c,
    }
    with open(DATA / "cross_model_heterogeneity_test.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote", DATA / "cross_model_heterogeneity_test.json")


if __name__ == "__main__":
    main()
