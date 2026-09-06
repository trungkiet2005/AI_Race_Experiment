#!/usr/bin/env python3
"""Robustness analysis after the family-level readout.

This stage excludes first-round decisions, incomplete/running runs, and duplicate
grain rows. It then checks whether lag, gap, family, model, and persona patterns
hold across narrower scopes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
REPORTS_DIR = OUTPUT_DIR / "reports"
RANDOM_SEED = 260726

CORE_TERMS = ["own_prev_unsafe", "opponent_prev_unsafe", "progress_gap_before"]
FAMILY_LABELS = {
    "family_chatgpt": "ChatGPT",
    "family_gemini": "Gemini",
}


@dataclass(frozen=True)
class Scope:
    scope: str
    label: str
    mask: pd.Series


def clean_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def pct(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{100 * float(value):.1f}%"


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


def safe_import_statsmodels() -> tuple[Any, Any] | None:
    try:
        import statsmodels.api as sm  # type: ignore
        import statsmodels.formula.api as smf  # type: ignore

        return smf, sm
    except Exception:
        return None


def safe_import_sklearn() -> dict[str, Any] | None:
    try:
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            roc_auc_score,
        )
        from sklearn.model_selection import GroupKFold  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import OneHotEncoder  # type: ignore
        from sklearn.tree import DecisionTreeClassifier, export_text  # type: ignore

        return {
            "ColumnTransformer": ColumnTransformer,
            "accuracy_score": accuracy_score,
            "balanced_accuracy_score": balanced_accuracy_score,
            "brier_score_loss": brier_score_loss,
            "roc_auc_score": roc_auc_score,
            "GroupKFold": GroupKFold,
            "Pipeline": Pipeline,
            "OneHotEncoder": OneHotEncoder,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "export_text": export_text,
        }
    except Exception:
        return None


def safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def load_turns() -> pd.DataFrame:
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv")
    turns["duplicate_grain_key"] = clean_bool(turns["duplicate_grain_key"])
    turns["is_round2plus"] = clean_bool(turns["is_round2plus"])
    numeric_columns = [
        "unsafe",
        "round",
        "max_private_risk",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
        "progress_gap_before",
        "own_private_risk_before",
        "opponent_private_risk_before",
        "retry_count",
        "parse_failed",
    ]
    for col in numeric_columns:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")
    turns["cluster_id"] = turns["source_run"].astype(str) + "::" + turns["game_id"].astype(str)
    turns["lag_profile"] = (
        turns["own_prev_unsafe"].fillna(-1).astype(int).astype(str)
        + "/"
        + turns["opponent_prev_unsafe"].fillna(-1).astype(int).astype(str)
    )
    return turns


def make_base_frame(turns: pd.DataFrame) -> pd.DataFrame:
    return turns[
        (turns["manifest_status"] == "completed")
        & (~turns["duplicate_grain_key"])
        & (turns["is_round2plus"])
    ].dropna(
        subset=[
            "unsafe",
            "own_prev_unsafe",
            "opponent_prev_unsafe",
            "progress_gap_before",
            "max_private_risk",
            "own_private_risk_before",
            "opponent_private_risk_before",
        ]
    ).copy()


def build_scopes(frame: pd.DataFrame) -> list[Scope]:
    scopes = [
        Scope("all_round2plus", "All completed, round 2+", pd.Series(True, index=frame.index)),
        Scope(
            "baseline_round2plus",
            "Baseline completed, round 2+",
            frame["analysis_scope"].eq("baseline_completed"),
        ),
    ]
    for family in sorted(frame["family"].dropna().unique()):
        scopes.append(
            Scope(
                f"{family}_all_round2plus",
                f"{FAMILY_LABELS.get(family, family)} all, round 2+",
                frame["family"].eq(family),
            )
        )
        scopes.append(
            Scope(
                f"{family}_baseline_round2plus",
                f"{FAMILY_LABELS.get(family, family)} baseline, round 2+",
                frame["family"].eq(family) & frame["analysis_scope"].eq("baseline_completed"),
            )
        )
    for mode in sorted(frame["experiment_mode"].dropna().unique()):
        scopes.append(
            Scope(
                f"experiment_{mode}_round2plus",
                f"{mode}, round 2+",
                frame["experiment_mode"].eq(mode),
            )
        )
    return scopes


def summarize_scopes(frame: pd.DataFrame, scopes: list[Scope]) -> pd.DataFrame:
    rows = []
    for scope in scopes:
        sub = frame[scope.mask]
        if sub.empty:
            continue
        rows.append(
            {
                "scope": scope.scope,
                "label": scope.label,
                "decisions": len(sub),
                "clusters": sub["cluster_id"].nunique(),
                "unsafe_rate": sub["unsafe"].mean(),
                "retry_rate": (sub["retry_count"].fillna(0) > 0).mean(),
                "parse_fail_rate": sub["parse_failed"].mean(),
                "families": sub["family"].nunique(),
                "models": sub["model_slug"].nunique(),
            }
        )
    summary = pd.DataFrame(rows)
    summary.to_csv(DERIVED_DIR / "robustness_scope_summary.csv", index=False)
    return summary


def summarize_lag(frame: pd.DataFrame) -> pd.DataFrame:
    lag = (
        frame.groupby(["family", "experiment_mode", "lag_profile"], dropna=False)
        .agg(decisions=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    lag.to_csv(DERIVED_DIR / "robustness_lag_by_family_experiment.csv", index=False)
    return lag


def summarize_model(frame: pd.DataFrame) -> pd.DataFrame:
    model = (
        frame.groupby(["family", "experiment_mode", "model_slug"], dropna=False)
        .agg(
            decisions=("unsafe", "size"),
            unsafe_rate=("unsafe", "mean"),
            retry_rate=("retry_count", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean()),
            mean_own_private_risk_before=("own_private_risk_before", "mean"),
        )
        .reset_index()
    )
    model.to_csv(DERIVED_DIR / "robustness_model_summary.csv", index=False)
    return model


def scan_gap_thresholds(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    thresholds = [0.25, 0.5, 1.0, 1.5, 2.0]
    groups = [
        ("all_round2plus", pd.Series(True, index=frame.index)),
        ("baseline_round2plus", frame["analysis_scope"].eq("baseline_completed")),
    ]
    for family in sorted(frame["family"].dropna().unique()):
        groups.append((f"{family}_all_round2plus", frame["family"].eq(family)))
        groups.append(
            (
                f"{family}_baseline_round2plus",
                frame["family"].eq(family) & frame["analysis_scope"].eq("baseline_completed"),
            )
        )
    for group_name, mask in groups:
        sub = frame[mask]
        if sub.empty:
            continue
        for threshold in thresholds:
            behind = sub["progress_gap_before"] <= -threshold
            ahead = sub["progress_gap_before"] >= threshold
            middle = ~(behind | ahead)
            rows.append(
                {
                    "scope": group_name,
                    "threshold": threshold,
                    "behind_decisions": int(behind.sum()),
                    "behind_unsafe_rate": sub.loc[behind, "unsafe"].mean() if behind.any() else np.nan,
                    "middle_decisions": int(middle.sum()),
                    "middle_unsafe_rate": sub.loc[middle, "unsafe"].mean() if middle.any() else np.nan,
                    "ahead_decisions": int(ahead.sum()),
                    "ahead_unsafe_rate": sub.loc[ahead, "unsafe"].mean() if ahead.any() else np.nan,
                    "behind_minus_middle": (
                        sub.loc[behind, "unsafe"].mean() - sub.loc[middle, "unsafe"].mean()
                        if behind.any() and middle.any()
                        else np.nan
                    ),
                    "behind_minus_ahead": (
                        sub.loc[behind, "unsafe"].mean() - sub.loc[ahead, "unsafe"].mean()
                        if behind.any() and ahead.any()
                        else np.nan
                    ),
                }
            )
    scan = pd.DataFrame(rows)
    scan.to_csv(DERIVED_DIR / "robustness_gap_threshold_scan.csv", index=False)
    return scan


def fit_logit_scopes(frame: pd.DataFrame, scopes: list[Scope]) -> tuple[pd.DataFrame, pd.DataFrame]:
    imported = safe_import_statsmodels()
    if imported is None:
        skipped = pd.DataFrame([{"stage": "robustness_logit", "reason": "statsmodels not installed"}])
        skipped.to_csv(DERIVED_DIR / "robustness_logit_skipped.csv", index=False)
        return pd.DataFrame(), skipped
    smf, sm = imported
    formula = "unsafe ~ own_prev_unsafe + opponent_prev_unsafe + progress_gap_before + C(max_private_risk)"
    coefficient_rows = []
    metadata_rows = []
    for scope in scopes:
        sub = frame[scope.mask].copy()
        sub = sub.dropna(subset=["unsafe", *CORE_TERMS, "max_private_risk", "cluster_id"])
        if len(sub) < 100 or sub["unsafe"].nunique() < 2:
            metadata_rows.append(
                {
                    "scope": scope.scope,
                    "n": len(sub),
                    "clusters": sub["cluster_id"].nunique(),
                    "status": "skipped",
                    "error": "too few rows or one-class target",
                }
            )
            continue
        try:
            fit = smf.glm(formula=formula, data=sub, family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={"groups": sub["cluster_id"]},
                maxiter=200,
            )
            conf = fit.conf_int()
            for term in CORE_TERMS:
                coefficient_rows.append(
                    {
                        "scope": scope.scope,
                        "label": scope.label,
                        "term": term,
                        "coef": fit.params.get(term, np.nan),
                        "odds_ratio": float(np.exp(fit.params.get(term, np.nan))),
                        "ci95_low": conf.loc[term, 0] if term in conf.index else np.nan,
                        "ci95_high": conf.loc[term, 1] if term in conf.index else np.nan,
                        "p_value": fit.pvalues.get(term, np.nan),
                        "n": len(sub),
                        "clusters": sub["cluster_id"].nunique(),
                    }
                )
            metadata_rows.append(
                {
                    "scope": scope.scope,
                    "n": len(sub),
                    "clusters": sub["cluster_id"].nunique(),
                    "status": "fit",
                    "error": "",
                    "aic": fit.aic,
                    "pseudo_r2_mcfadden": 1 - (fit.llf / fit.llnull) if fit.llnull else np.nan,
                }
            )
        except Exception as exc:
            metadata_rows.append(
                {
                    "scope": scope.scope,
                    "n": len(sub),
                    "clusters": sub["cluster_id"].nunique(),
                    "status": "error",
                    "error": str(exc),
                }
            )
    coefficients = pd.DataFrame(coefficient_rows)
    metadata = pd.DataFrame(metadata_rows)
    coefficients.to_csv(DERIVED_DIR / "robustness_logit_coefficients.csv", index=False)
    metadata.to_csv(DERIVED_DIR / "robustness_logit_metadata.csv", index=False)
    return coefficients, metadata


def make_encoder(sk: dict[str, Any]) -> Any:
    try:
        return sk["OneHotEncoder"](handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return sk["OneHotEncoder"](handle_unknown="ignore", sparse=False)


def get_feature_names(pipeline: Any, numeric_features: list[str], categorical_features: list[str]) -> np.ndarray:
    prep = pipeline.named_steps["prep"]
    try:
        cat = prep.named_transformers_["cat"].get_feature_names_out(categorical_features)
        return np.array([*numeric_features, *cat])
    except Exception:
        return np.array(numeric_features)


def fit_tree_scopes(frame: pd.DataFrame, scopes: list[Scope]) -> tuple[pd.DataFrame, pd.DataFrame]:
    sk = safe_import_sklearn()
    if sk is None:
        skipped = pd.DataFrame([{"stage": "robustness_tree", "reason": "scikit-learn not installed"}])
        skipped.to_csv(DERIVED_DIR / "robustness_tree_skipped.csv", index=False)
        return pd.DataFrame(), pd.DataFrame()

    metric_rows = []
    root_rows = []
    for scope in scopes:
        sub = frame[scope.mask].copy()
        if len(sub) < 300 or sub["unsafe"].nunique() < 2 or sub["cluster_id"].nunique() < 4:
            continue

        numeric_features = [
            "own_prev_unsafe",
            "opponent_prev_unsafe",
            "progress_gap_before",
            "own_private_risk_before",
            "opponent_private_risk_before",
            "round",
        ]
        categorical_candidates = [
            "family",
            "model_slug",
            "persona_mode",
            "experiment_mode",
            "condition",
            "max_private_risk",
        ]
        categorical_features = [
            col for col in categorical_candidates if col in sub and sub[col].nunique(dropna=True) > 1
        ]
        X = sub[numeric_features + categorical_features].copy()
        X[categorical_features] = X[categorical_features].astype(str)
        y = sub["unsafe"].astype(int)
        groups = sub["cluster_id"].astype(str)

        pipeline = sk["Pipeline"](
            steps=[
                (
                    "prep",
                    sk["ColumnTransformer"](
                        transformers=[
                            ("num", "passthrough", numeric_features),
                            ("cat", make_encoder(sk), categorical_features),
                        ]
                    ),
                ),
                (
                    "tree",
                    sk["DecisionTreeClassifier"](
                        max_depth=3,
                        min_samples_leaf=max(30, min(150, len(sub) // 25)),
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )

        n_splits = min(5, groups.nunique())
        cv = sk["GroupKFold"](n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipeline.predict(X.iloc[test_idx])
            proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            fold_y = y.iloc[test_idx]
            metric_rows.append(
                {
                    "scope": scope.scope,
                    "label": scope.label,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "accuracy": sk["accuracy_score"](fold_y, pred),
                    "balanced_accuracy": sk["balanced_accuracy_score"](fold_y, pred),
                    "roc_auc": sk["roc_auc_score"](fold_y, proba) if fold_y.nunique() == 2 else np.nan,
                    "brier_score": sk["brier_score_loss"](fold_y, proba),
                }
            )

        pipeline.fit(X, y)
        feature_names = get_feature_names(pipeline, numeric_features, categorical_features)
        tree = pipeline.named_steps["tree"]
        root_index = tree.tree_.feature[0]
        root_feature = feature_names[root_index] if root_index >= 0 else ""
        root_rows.append(
            {
                "scope": scope.scope,
                "label": scope.label,
                "root_feature": root_feature,
                "root_threshold": tree.tree_.threshold[0] if root_index >= 0 else np.nan,
                "decisions": len(sub),
                "clusters": groups.nunique(),
            }
        )
        rules = sk["export_text"](tree, feature_names=list(feature_names), decimals=3, max_depth=3)
        (DERIVED_DIR / f"robustness_tree_rules__{scope.scope}.txt").write_text(rules, encoding="utf-8")

    metrics = pd.DataFrame(metric_rows)
    roots = pd.DataFrame(root_rows)
    metrics.to_csv(DERIVED_DIR / "robustness_tree_cv_metrics.csv", index=False)
    roots.to_csv(DERIVED_DIR / "robustness_tree_roots.csv", index=False)
    return metrics, roots


def plot_robustness(model_summary: pd.DataFrame, lag_summary: pd.DataFrame) -> None:
    plt = safe_import_matplotlib()
    if plt is None:
        return

    baseline_models = model_summary[model_summary["experiment_mode"] == "mode_baseline"].copy()
    if not baseline_models.empty:
        baseline_models = baseline_models.sort_values("unsafe_rate")
        fig, ax = plt.subplots(figsize=(9, 4.8))
        colors = baseline_models["family"].map(
            {"family_chatgpt": "#3b78c2", "family_gemini": "#c27a2c"}
        ).fillna("#777777")
        ax.barh(baseline_models["model_slug"], baseline_models["unsafe_rate"], color=colors)
        ax.set_xlabel("Unsafe rate, round 2+")
        ax.set_title("Baseline unsafe rate by model after excluding round 1")
        ax.set_xlim(0, 1)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "robustness_baseline_model_unsafe.png", dpi=180)
        plt.close(fig)

    lag_base = lag_summary[lag_summary["experiment_mode"] == "mode_baseline"].copy()
    if not lag_base.empty:
        pivot = lag_base.pivot(index="lag_profile", columns="family", values="unsafe_rate")
        pivot = pivot.reindex(["0/0", "0/1", "1/0", "1/1"])
        fig, ax = plt.subplots(figsize=(7, 4.5))
        x = np.arange(len(pivot.index))
        width = 0.35
        for offset, family in zip([-width / 2, width / 2], ["family_chatgpt", "family_gemini"]):
            if family in pivot:
                ax.bar(x + offset, pivot[family], width=width, label=FAMILY_LABELS.get(family, family))
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Lag profile: own_prev/opponent_prev unsafe")
        ax.set_ylabel("Unsafe rate")
        ax.set_title("Baseline lag response by family after excluding round 1")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "robustness_baseline_lag_by_family.png", dpi=180)
        plt.close(fig)


def compact_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    return (
        metrics.groupby(["scope", "label"], dropna=False)
        .agg(
            folds=("fold", "size"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            brier_score=("brier_score", "mean"),
        )
        .reset_index()
    )


def write_report(
    scope_summary: pd.DataFrame,
    lag_summary: pd.DataFrame,
    model_summary: pd.DataFrame,
    gap_scan: pd.DataFrame,
    logit_coefficients: pd.DataFrame,
    logit_metadata: pd.DataFrame,
    tree_metrics: pd.DataFrame,
    tree_roots: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    tree_compact = compact_metrics(tree_metrics)
    baseline_models = model_summary[model_summary["experiment_mode"] == "mode_baseline"].copy()
    baseline_models = baseline_models.sort_values(["family", "unsafe_rate"])
    baseline_lag = lag_summary[lag_summary["experiment_mode"] == "mode_baseline"].copy()
    baseline_gap = gap_scan[gap_scan["scope"].str.endswith("baseline_round2plus")].copy()
    provider_terms = logit_coefficients[
        logit_coefficients["scope"].isin(
            ["family_chatgpt_baseline_round2plus", "family_gemini_baseline_round2plus"]
        )
    ].copy()
    failed_logit = logit_metadata[logit_metadata["status"] != "fit"].copy()

    tree_root_lookup = {
        row["scope"]: row["root_feature"]
        for _, row in tree_roots.iterrows()
    } if not tree_roots.empty else {}

    lines = [
        "# FH Robustness Analysis",
        "",
        "## Executive Summary",
        "",
        "- **After excluding round 1, the family story still holds: Gemini remains higher-unsafe in baseline, and ChatGPT remains more model-split than family-split.** The baseline model table is the cleanest robustness cut because it removes the saturated first move and keeps model identity visible.",
        "- **The lag pattern persists but should be read as behavioral state, not causal evidence.** ChatGPT still shows a smoother increase after unsafe history; Gemini remains opponent-sensitive and asymmetric in baseline.",
        "- **Predictive splits remain dominated by model/state variables.** The robustness trees root on model identity in baseline scopes and on accumulated private-risk/state in wider scopes, matching the earlier full pipeline.",
        "- **Coefficient stability is uneven by segment.** ChatGPT baseline coefficients are easier to interpret; Gemini baseline fits after dropping round 1, but the gap coefficient remains extreme, so rate/lag evidence is more trustworthy than exact Gemini magnitudes.",
        "",
        "## Scope Gate",
        "",
        "All robustness tables exclude incomplete/running runs, duplicate-grain rows, and first-round decisions. This makes the analysis about response dynamics after the initial move.",
        "",
        markdown_table(
            scope_summary[
                [
                    "scope",
                    "decisions",
                    "clusters",
                    "unsafe_rate",
                    "retry_rate",
                    "families",
                    "models",
                ]
            ].head(12)
        ),
        "",
        "## Model Identity Still Carries The Baseline Split",
        "",
        "**Once round 1 is removed, Gemini remains high-unsafe and ChatGPT still splits sharply by model.** This supports using `model_slug` before family-level averages when explaining baseline behavior.",
        "",
        markdown_table(
            baseline_models[
                ["family", "model_slug", "decisions", "unsafe_rate", "retry_rate", "mean_own_private_risk_before"]
            ]
        ),
        "",
        "Visual: `figures/robustness_baseline_model_unsafe.png`.",
        "",
        "## Lag Response Is The Strongest Mechanistic Check",
        "",
        "**ChatGPT's baseline lag response is smoother; Gemini's is asymmetric.** This survives the round-1 exclusion, so it is not only a first-turn artifact.",
        "",
        markdown_table(
            baseline_lag[["family", "lag_profile", "decisions", "unsafe_rate"]].sort_values(
                ["family", "lag_profile"]
            )
        ),
        "",
        "Visual: `figures/robustness_baseline_lag_by_family.png`.",
        "",
        "## Gap Thresholds Are Directional, Not A Single Universal Cutoff",
        "",
        "**Being behind often raises unsafe relative to middle/ahead states, but the threshold is scope-dependent.** Treat these thresholds as diagnostic bins rather than a universal rule.",
        "",
        markdown_table(
            baseline_gap[
                [
                    "scope",
                    "threshold",
                    "behind_decisions",
                    "behind_unsafe_rate",
                    "middle_unsafe_rate",
                    "ahead_unsafe_rate",
                    "behind_minus_middle",
                    "behind_minus_ahead",
                ]
            ]
        ),
        "",
        "## Logit Robustness",
        "",
        "**The robust logit check agrees most clearly for ChatGPT/OpenAI-style baseline dynamics.** Gemini now fits after the round-1 exclusion, but its gap coefficient is still extreme, so the coefficient table is useful as a sign check, not a headline proof.",
        "",
        markdown_table(
            provider_terms[
                [
                    "scope",
                    "term",
                    "coef",
                    "odds_ratio",
                    "ci95_low",
                    "ci95_high",
                    "p_value",
                    "n",
                    "clusters",
                ]
            ]
        ),
        "",
    ]
    if not failed_logit.empty:
        lines.extend(
            [
                "Skipped/error logit scopes:",
                "",
                markdown_table(failed_logit[["scope", "n", "clusters", "status", "error"]]),
                "",
            ]
        )
    lines.extend(
        [
            "## Predictive Robustness",
            "",
            (
                "**Tree performance remains useful but exploratory.** "
                f"Baseline root: `{tree_root_lookup.get('baseline_round2plus', 'NA')}`; "
                f"ChatGPT baseline root: `{tree_root_lookup.get('family_chatgpt_baseline_round2plus', 'NA')}`; "
                f"Gemini baseline root: `{tree_root_lookup.get('family_gemini_baseline_round2plus', 'NA')}`."
            ),
            "",
            markdown_table(
                tree_compact[
                    [
                        "scope",
                        "balanced_accuracy",
                        "roc_auc",
                        "brier_score",
                    ]
                ].head(15)
            )
            if not tree_compact.empty
            else "_Tree metrics unavailable._",
            "",
            markdown_table(tree_roots[["scope", "root_feature", "root_threshold", "decisions", "clusters"]])
            if not tree_roots.empty
            else "_Tree roots unavailable._",
            "",
            "## What To Run Next",
            "",
            "1. Run per-model robustness reports for the highest-contrast models: `gpt-5-nano`, `gpt-5.4-nano`, and the three Gemini baseline models.",
            "2. For Gemini, add a first-turn-specific analysis instead of forcing one pooled logit, because first-round saturation is itself the signal.",
            "3. Add a mixed-effects or GEE specification if the paper needs a stronger repeated-game inference layer.",
            "",
            "## Caveats",
            "",
            "- These outputs deliberately exclude round 1; they answer response-dynamics questions, not initial-position questions.",
            "- Trees compress behavior into rules; they are not causal estimates.",
            "- Narrow segment logits can still hit separation or unstable covariance, especially in saturated Gemini cells.",
        ]
    )
    report_path = REPORTS_DIR / "fh_robustness_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "report": str(report_path.relative_to(REPO_ROOT)),
        "derived_outputs": [
            "robustness_scope_summary.csv",
            "robustness_model_summary.csv",
            "robustness_lag_by_family_experiment.csv",
            "robustness_gap_threshold_scan.csv",
            "robustness_logit_coefficients.csv",
            "robustness_logit_metadata.csv",
            "robustness_tree_cv_metrics.csv",
            "robustness_tree_roots.csv",
        ],
        "figures": [
            "robustness_baseline_model_unsafe.png",
            "robustness_baseline_lag_by_family.png",
        ],
    }
    (DERIVED_DIR / "robustness_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    turns = load_turns()
    frame = make_base_frame(turns)
    scopes = build_scopes(frame)
    scope_summary = summarize_scopes(frame, scopes)
    lag_summary = summarize_lag(frame)
    model_summary = summarize_model(frame)
    gap_scan = scan_gap_thresholds(frame)
    logit_coefficients, logit_metadata = fit_logit_scopes(frame, scopes)
    tree_metrics, tree_roots = fit_tree_scopes(frame, scopes)
    plot_robustness(model_summary, lag_summary)
    write_report(
        scope_summary,
        lag_summary,
        model_summary,
        gap_scan,
        logit_coefficients,
        logit_metadata,
        tree_metrics,
        tree_roots,
    )
    print(f"Wrote robustness analysis to {REPORTS_DIR / 'fh_robustness_analysis.md'}")


if __name__ == "__main__":
    main()
