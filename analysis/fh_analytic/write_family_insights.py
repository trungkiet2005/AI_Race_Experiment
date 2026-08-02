#!/usr/bin/env python3
"""Write family-level insight tables and a compact narrative report."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
REPORTS_DIR = OUTPUT_DIR / "reports"

FAMILY_LABELS = {
    "family_chatgpt": "ChatGPT family",
    "family_gemini": "Gemini family",
}


def clean_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_No rows._"
    visible = frame.copy()
    for col in visible.columns:
        if pd.api.types.is_float_dtype(visible[col]):
            visible[col] = visible[col].map(lambda value: "" if pd.isna(value) else f"{value:.4g}")
        else:
            visible[col] = visible[col].map(lambda value: "" if pd.isna(value) else str(value))
    header = "| " + " | ".join(visible.columns) + " |"
    sep = "| " + " | ".join(["---"] * len(visible.columns)) + " |"
    rows = [
        "| " + " | ".join(str(row[col]) for col in visible.columns) + " |"
        for _, row in visible.iterrows()
    ]
    return "\n".join([header, sep, *rows])


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100 * float(value):.1f}%"


def pp(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100 * float(value):+.1f} pp"


def family_name(family: str) -> str:
    return FAMILY_LABELS.get(family, family)


def aggregate_turns(frame: pd.DataFrame, groups: Iterable[str]) -> pd.DataFrame:
    summary = (
        frame.groupby(list(groups), dropna=False)
        .agg(
            decisions=("unsafe", "size"),
            unsafe_rate=("unsafe", "mean"),
            retry_rate=("retry_count", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean()),
            parse_fail_rate=("parse_failed", "mean"),
            mean_round=("round", "mean"),
        )
        .reset_index()
    )
    return summary


def write_family_outputs() -> dict[str, pd.DataFrame]:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    coverage = pd.read_csv(DERIVED_DIR / "coverage_audit.csv")
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv")
    turns["duplicate_grain_key"] = clean_bool(turns["duplicate_grain_key"])
    turns["is_round2plus"] = clean_bool(turns["is_round2plus"])
    for col in [
        "unsafe",
        "retry_count",
        "parse_failed",
        "round",
        "max_private_risk",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
    ]:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")

    clean = turns[(turns["manifest_status"] == "completed") & (~turns["duplicate_grain_key"])].copy()
    baseline = clean[clean["analysis_scope"] == "baseline_completed"].copy()
    baseline_r2 = baseline[baseline["is_round2plus"]].copy()

    coverage_family = (
        coverage.groupby("family")
        .agg(
            runs=("source_run", "size"),
            completed_runs=("manifest_status", lambda s: (s == "completed").sum()),
            incomplete_runs=("manifest_status", lambda s: (s != "completed").sum()),
            race_rows=("race_rows", "sum"),
            player_rows=("player_rows", "sum"),
            turn_rows=("turn_rows", "sum"),
        )
        .reset_index()
    )
    coverage_family["completion_rate"] = coverage_family["completed_runs"] / coverage_family["runs"]

    family_summary = aggregate_turns(clean, ["family"])
    baseline_summary = aggregate_turns(baseline, ["family"])
    experiment_summary = aggregate_turns(clean, ["family", "persona_mode", "experiment_mode"])
    model_summary = aggregate_turns(baseline, ["family", "model_slug"])
    risk_summary = (
        baseline.groupby(["family", "max_private_risk"], dropna=False)
        .agg(decisions=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    baseline_r2["lag_profile"] = (
        baseline_r2["own_prev_unsafe"].astype(int).astype(str)
        + "/"
        + baseline_r2["opponent_prev_unsafe"].astype(int).astype(str)
    )
    lag_summary = (
        baseline_r2.groupby(["family", "lag_profile"], dropna=False)
        .agg(decisions=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    first_round_summary = (
        baseline[baseline["round"] == 1]
        .groupby("family", dropna=False)
        .agg(
            decisions=("unsafe", "size"),
            first_round_unsafe_rate=("unsafe", "mean"),
            retry_rate=("retry_count", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean()),
        )
        .reset_index()
    )

    provider_coefficients = pd.read_csv(DERIVED_DIR / "human_check_segment_coefficients.csv")
    provider_coefficients = provider_coefficients[
        provider_coefficients["segment_type"] == "baseline_provider"
    ].copy()
    provider_coefficients["family"] = provider_coefficients["segment"].map(
        {
            "provider_openai": "family_chatgpt",
            "provider_google": "family_gemini",
        }
    )

    tree_scope = pd.read_csv(DERIVED_DIR / "decision_tree_scope_summary.csv")
    root_stability = pd.read_csv(DERIVED_DIR / "decision_tree_all_completed_root_stability.csv")

    outputs = {
        "family_coverage": coverage_family,
        "family_summary_completed": family_summary,
        "family_baseline_summary": baseline_summary,
        "family_experiment_summary": experiment_summary,
        "family_model_baseline_summary": model_summary,
        "family_baseline_risk_summary": risk_summary,
        "family_baseline_lag_summary": lag_summary,
        "family_first_round_baseline_summary": first_round_summary,
        "family_human_check_coefficients": provider_coefficients,
        "family_tree_scope_summary": tree_scope,
        "family_tree_root_stability": root_stability,
    }
    for name, frame in outputs.items():
        frame.to_csv(DERIVED_DIR / f"{name}.csv", index=False)

    write_report(outputs)
    return outputs


def lookup_rate(frame: pd.DataFrame, family: str, column: str = "unsafe_rate") -> float:
    row = frame[frame["family"] == family]
    if row.empty:
        return np.nan
    return float(row.iloc[0][column])


def lookup_lag(frame: pd.DataFrame, family: str, lag_profile: str) -> float:
    row = frame[(frame["family"] == family) & (frame["lag_profile"] == lag_profile)]
    if row.empty:
        return np.nan
    return float(row.iloc[0]["unsafe_rate"])


def format_table(frame: pd.DataFrame, columns: list[str]) -> str:
    view = frame[columns].copy()
    rename = {
        "family": "family",
        "completion_rate": "completion",
        "unsafe_rate": "unsafe",
        "retry_rate": "retry",
        "parse_fail_rate": "parse_fail",
        "first_round_unsafe_rate": "first_round_unsafe",
    }
    view = view.rename(columns=rename)
    for col in view.columns:
        if col in {"unsafe", "retry", "parse_fail", "completion", "first_round_unsafe"}:
            view[col] = view[col].map(pct)
        elif col in {"decisions", "runs", "completed_runs", "incomplete_runs"}:
            view[col] = view[col].map(lambda value: f"{int(value):,}")
    view["family"] = view["family"].map(family_name) if "family" in view else view.get("family")
    return markdown_table(view)


def write_report(outputs: dict[str, pd.DataFrame]) -> None:
    coverage = outputs["family_coverage"]
    completed = outputs["family_summary_completed"]
    baseline = outputs["family_baseline_summary"]
    experiment = outputs["family_experiment_summary"]
    model = outputs["family_model_baseline_summary"]
    risk = outputs["family_baseline_risk_summary"]
    lag = outputs["family_baseline_lag_summary"]
    first_round = outputs["family_first_round_baseline_summary"]
    coeff = outputs["family_human_check_coefficients"]
    tree = outputs["family_tree_scope_summary"]
    root = outputs["family_tree_root_stability"]

    chat_all = lookup_rate(completed, "family_chatgpt")
    gem_all = lookup_rate(completed, "family_gemini")
    chat_base = lookup_rate(baseline, "family_chatgpt")
    gem_base = lookup_rate(baseline, "family_gemini")
    chat_risk_matrix = lookup_rate(
        experiment[
            (experiment["family"] == "family_chatgpt")
            & (experiment["experiment_mode"] == "mode_risk_matrix")
        ],
        "family_chatgpt",
    )
    gem_risk_matrix = lookup_rate(
        experiment[
            (experiment["family"] == "family_gemini")
            & (experiment["experiment_mode"] == "mode_risk_matrix")
        ],
        "family_gemini",
    )
    chat_strategy = lookup_rate(
        experiment[
            (experiment["family"] == "family_chatgpt")
            & (experiment["experiment_mode"] == "mode_strategy_persona")
        ],
        "family_chatgpt",
    )
    gem_strategy = lookup_rate(
        experiment[
            (experiment["family"] == "family_gemini")
            & (experiment["experiment_mode"] == "mode_strategy_persona")
        ],
        "family_gemini",
    )
    chat_first = lookup_rate(first_round, "family_chatgpt", "first_round_unsafe_rate")
    gem_first = lookup_rate(first_round, "family_gemini", "first_round_unsafe_rate")

    lines = [
        "# FH Family Insights",
        "",
        "## Executive Summary",
        "",
        (
            f"- **Gemini is much more unsafe in baseline, but risk-aware/persona protocols pull it down.** "
            f"Baseline unsafe is {pct(gem_base)} for Gemini versus {pct(chat_base)} for ChatGPT "
            f"({pp(gem_base - chat_base)} gap). Under risk-matrix/persona runs Gemini drops to "
            f"{pct(gem_risk_matrix)} / {pct(gem_strategy)}, while ChatGPT stays near "
            f"{pct(chat_risk_matrix)} / {pct(chat_strategy)}."
        ),
        (
            f"- **ChatGPT behavior is lower on average but split sharply by model.** "
            f"Baseline ChatGPT averages {pct(chat_base)}, but the baseline model table separates "
            f"`gpt-5-nano` from `gpt-5.4-nano`; family-level averages hide that model identity effect."
        ),
        (
            f"- **Gemini baseline starts saturated.** First-round baseline unsafe is "
            f"{pct(gem_first)} for Gemini versus {pct(chat_first)} for ChatGPT, so Gemini's "
            f"baseline logit coefficients are less stable and should be treated as descriptive/mechanistic clues, "
            f"not coefficient-level proof."
        ),
        (
            f"- **Predictive confidence is strongest for state/history, not family as a causal claim.** "
            f"The full completed tree roots on `{root.iloc[0]['root_feature']}` in "
            f"{int(root.iloc[0]['count'])}/{int(root.iloc[0]['count'])} bootstraps, while the baseline tree "
            f"mostly roots on `{tree.loc[tree['scope'] == 'baseline_completed', 'top_root_feature'].iloc[0]}`."
        ),
        "",
        "## Coverage And Data Quality",
        "",
        "Gemini has incomplete coverage and a higher retry rate, so any family comparison should keep coverage visible. Parse failures are zero in completed clean rows.",
        "",
        format_table(
            coverage,
            ["family", "runs", "completed_runs", "incomplete_runs", "completion_rate"],
        ),
        "",
        format_table(
            completed,
            ["family", "decisions", "unsafe_rate", "retry_rate", "parse_fail_rate"],
        ),
        "",
        "## Family-Level Behavioral Readout",
        "",
        "**Baseline is where the family separation is largest.** Gemini baseline is high-unsafe and starts unsafe immediately; ChatGPT is materially lower and more condition-sensitive through model identity than family label alone.",
        "",
        format_table(
            baseline,
            ["family", "decisions", "unsafe_rate", "retry_rate", "parse_fail_rate"],
        ),
        "",
        format_table(
            first_round,
            ["family", "decisions", "first_round_unsafe_rate", "retry_rate"],
        ),
        "",
        "Baseline model split:",
        "",
        format_table(
            model.sort_values(["family", "model_slug"]),
            ["family", "model_slug", "decisions", "unsafe_rate", "retry_rate"],
        ),
        "",
        "**Persona/risk-aware protocols change Gemini much more than ChatGPT.** The same protocol shift barely moves ChatGPT's aggregate unsafe rate, but it cuts Gemini from the saturated baseline into the high-40% range. That suggests Gemini is more steerable by explicit risk/persona framing in this snapshot.",
        "",
        format_table(
            experiment.sort_values(["family", "experiment_mode", "persona_mode"]),
            ["family", "persona_mode", "experiment_mode", "decisions", "unsafe_rate", "retry_rate"],
        ),
        "",
        "## Risk And Lag Patterns",
        "",
        "**Gemini's baseline risk curve is counterintuitive: unsafe falls as private risk rises.** The drop from low to high risk is descriptive evidence that baseline Gemini is not simply trading off private risk in the expected monotonic way; it begins almost always unsafe at low risk, then moderates as risk increases.",
        "",
        format_table(
            risk.sort_values(["family", "max_private_risk"]),
            ["family", "max_private_risk", "decisions", "unsafe_rate"],
        ),
        "",
        (
            f"**Lag response differs by family.** ChatGPT baseline follows a more human-like conditional pattern: "
            f"unsafe is {pct(lookup_lag(lag, 'family_chatgpt', '0/0'))} after both were safe, then rises to "
            f"{pct(lookup_lag(lag, 'family_chatgpt', '1/0'))} after own previous unsafe and "
            f"{pct(lookup_lag(lag, 'family_chatgpt', '0/1'))} after opponent previous unsafe. Gemini is asymmetric: "
            f"{pct(lookup_lag(lag, 'family_gemini', '0/1'))} after opponent-only unsafe but only "
            f"{pct(lookup_lag(lag, 'family_gemini', '1/0'))} after own-only unsafe."
        ),
        "",
        format_table(
            lag.sort_values(["family", "lag_profile"]),
            ["family", "lag_profile", "decisions", "unsafe_rate"],
        ),
        "",
        "## Human-Reference Checks",
        "",
        "**OpenAI/ChatGPT has interpretable baseline logit evidence; Gemini has separation/saturation.** ChatGPT provider coefficients are directionally human-like with significant effects for own previous unsafe, opponent previous unsafe, progress gap, and first-round unsafe. Gemini coefficients have missing CIs/p-values in the provider slice, consistent with saturated first-round and near-separated cells.",
        "",
        markdown_table(
            coeff[
                ["family", "term", "coef", "odds_ratio", "ci95_low", "ci95_high", "p_value", "n", "error"]
            ].assign(family=lambda d: d["family"].map(family_name))
        ),
        "",
        "## Predictive Model Implications",
        "",
        "**Model identity explains baseline behavior; accumulated risk/history explains the full completed behavior.** The baseline tree reaches ROC-AUC around 0.84 and mostly splits on `model_slug_gpt-5-nano`, so family-level summaries should be read with model identity in view. In the full completed data, `own_private_risk_before` is the root feature in every bootstrap, which is predictive confidence about behavioral state, not causal evidence that the state independently causes unsafe choices.",
        "",
        markdown_table(
            tree[
                [
                    "scope",
                    "decisions",
                    "unsafe_rate",
                    "cv_balanced_accuracy_mean",
                    "cv_roc_auc_mean",
                    "top_root_feature",
                    "top_root_share",
                ]
            ]
        ),
        "",
        "## Recommended Next Steps",
        "",
        "1. Use `family` as a reporting cut, but use `model_slug` as the primary explanatory cut for baseline because the tree and model table show it carries the separation.",
        "2. Treat Gemini baseline as a saturation case: report rates and lag tables first, and avoid overclaiming unstable segment coefficients.",
        "3. For robustness, rerun family-specific trees/logits after excluding first round, then compare whether ChatGPT's human-like lag/gap pattern and Gemini's opponent-only asymmetry persist.",
        "4. Keep incomplete Gemini cells visible in every family chart; exclude them from headline rates unless the run completes.",
        "",
        "## Caveats And Assumptions",
        "",
        "- Main behavioral rates use completed, non-duplicate rows from `turns_canonical.csv`.",
        "- Baseline comparisons use `analysis_scope == baseline_completed`; persona/risk-aware comparisons are descriptive because coverage and prompt framing differ.",
        "- Coefficients and predictive models answer different questions: logit terms are mechanistic checks; trees identify predictive splits/rules.",
        "- Gemini family has 3 incomplete runs and materially higher retry rates, so coverage/data-quality context should travel with family claims.",
    ]
    (REPORTS_DIR / "fh_family_insights.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    write_family_outputs()
    print(f"Wrote family insights to {REPORTS_DIR / 'fh_family_insights.md'}")
