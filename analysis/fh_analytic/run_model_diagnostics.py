#!/usr/bin/env python3
"""Per-model diagnostics after family and robustness stages.

This stage focuses on the models that drive the family-level findings:
first-turn saturation, round-by-round dynamics, lag profiles, and model-level
logit/tree robustness.
"""

from __future__ import annotations

import json
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
MODEL_ORDER = [
    "gpt-5-nano",
    "gpt-5.4-nano",
    "google-gemini-3.5-flash-lite",
    "google-gemini-3-flash-preview",
    "google-gemini-3.1-flash-lite-preview",
]


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
    return turns[
        (turns["manifest_status"] == "completed")
        & (~turns["duplicate_grain_key"])
    ].copy()


def summarize_baseline_models(turns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    baseline = turns[turns["analysis_scope"] == "baseline_completed"].copy()
    baseline_r2 = baseline[baseline["is_round2plus"]].copy()

    first_turn = (
        baseline[baseline["round"] == 1]
        .groupby(["family", "model_slug", "max_private_risk"], dropna=False)
        .agg(
            decisions=("unsafe", "size"),
            unsafe_rate=("unsafe", "mean"),
            retry_rate=("retry_count", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean()),
        )
        .reset_index()
    )
    first_turn.to_csv(DERIVED_DIR / "model_first_turn_by_risk.csv", index=False)

    first_vs_later = (
        baseline.assign(turn_phase=lambda d: np.where(d["round"] == 1, "round_1", "round_2plus"))
        .groupby(["family", "model_slug", "turn_phase"], dropna=False)
        .agg(decisions=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    first_vs_later.to_csv(DERIVED_DIR / "model_first_vs_later_summary.csv", index=False)

    round_dynamics = (
        baseline.groupby(["family", "model_slug", "round"], dropna=False)
        .agg(decisions=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    round_dynamics.to_csv(DERIVED_DIR / "model_round_dynamics.csv", index=False)

    lag = (
        baseline_r2.groupby(["family", "model_slug", "lag_profile"], dropna=False)
        .agg(decisions=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    lag.to_csv(DERIVED_DIR / "model_lag_summary.csv", index=False)

    gap_bin = (
        baseline_r2.groupby(["family", "model_slug", "gap_bin"], dropna=False)
        .agg(decisions=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    gap_bin.to_csv(DERIVED_DIR / "model_gap_bin_summary.csv", index=False)

    risk_later = (
        baseline_r2.groupby(["family", "model_slug", "max_private_risk"], dropna=False)
        .agg(decisions=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    risk_later.to_csv(DERIVED_DIR / "model_round2plus_by_risk.csv", index=False)

    return {
        "baseline": baseline,
        "baseline_r2": baseline_r2,
        "first_turn": first_turn,
        "first_vs_later": first_vs_later,
        "round_dynamics": round_dynamics,
        "lag": lag,
        "gap_bin": gap_bin,
        "risk_later": risk_later,
    }


def fit_model_logits(baseline_r2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    imported = safe_import_statsmodels()
    if imported is None:
        skipped = pd.DataFrame([{"stage": "model_logit", "reason": "statsmodels not installed"}])
        skipped.to_csv(DERIVED_DIR / "model_logit_skipped.csv", index=False)
        return pd.DataFrame(), skipped
    smf, sm = imported
    formula = "unsafe ~ own_prev_unsafe + opponent_prev_unsafe + progress_gap_before + C(max_private_risk)"
    coefficient_rows = []
    metadata_rows = []
    for model_slug, sub in baseline_r2.groupby("model_slug", dropna=False):
        sub = sub.dropna(subset=["unsafe", *CORE_TERMS, "max_private_risk", "cluster_id"]).copy()
        if len(sub) < 100 or sub["unsafe"].nunique() < 2:
            metadata_rows.append(
                {
                    "model_slug": model_slug,
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
                        "model_slug": model_slug,
                        "family": sub["family"].iloc[0],
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
                    "model_slug": model_slug,
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
                    "model_slug": model_slug,
                    "n": len(sub),
                    "clusters": sub["cluster_id"].nunique(),
                    "status": "error",
                    "error": str(exc),
                }
            )
    coefficients = pd.DataFrame(coefficient_rows)
    metadata = pd.DataFrame(metadata_rows)
    coefficients.to_csv(DERIVED_DIR / "model_logit_coefficients.csv", index=False)
    metadata.to_csv(DERIVED_DIR / "model_logit_metadata.csv", index=False)
    return coefficients, metadata


def fit_model_trees(baseline_r2: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sk = safe_import_sklearn()
    if sk is None:
        skipped = pd.DataFrame([{"stage": "model_tree", "reason": "scikit-learn not installed"}])
        skipped.to_csv(DERIVED_DIR / "model_tree_skipped.csv", index=False)
        return pd.DataFrame(), pd.DataFrame()
    metric_rows = []
    root_rows = []
    for model_slug, sub in baseline_r2.groupby("model_slug", dropna=False):
        sub = sub.dropna(
            subset=[
                "unsafe",
                "own_prev_unsafe",
                "opponent_prev_unsafe",
                "progress_gap_before",
                "own_private_risk_before",
                "opponent_private_risk_before",
                "round",
                "max_private_risk",
            ]
        ).copy()
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
        categorical_features = [
            col for col in ["max_private_risk", "gap_bin"] if sub[col].nunique(dropna=True) > 1
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
                        min_samples_leaf=max(20, len(sub) // 20),
                        class_weight="balanced",
                        random_state=RANDOM_SEED,
                    ),
                ),
            ]
        )
        cv = sk["GroupKFold"](n_splits=min(5, groups.nunique()))
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipeline.predict(X.iloc[test_idx])
            proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            fold_y = y.iloc[test_idx]
            metric_rows.append(
                {
                    "model_slug": model_slug,
                    "family": sub["family"].iloc[0],
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
                "model_slug": model_slug,
                "family": sub["family"].iloc[0],
                "root_feature": root_feature,
                "root_threshold": tree.tree_.threshold[0] if root_index >= 0 else np.nan,
                "decisions": len(sub),
                "clusters": groups.nunique(),
            }
        )
        rules = sk["export_text"](tree, feature_names=list(feature_names), decimals=3, max_depth=3)
        safe_model = str(model_slug).replace("/", "_").replace("\\", "_")
        (DERIVED_DIR / f"model_tree_rules__{safe_model}.txt").write_text(rules, encoding="utf-8")

    metrics = pd.DataFrame(metric_rows)
    roots = pd.DataFrame(root_rows)
    metrics.to_csv(DERIVED_DIR / "model_tree_cv_metrics.csv", index=False)
    roots.to_csv(DERIVED_DIR / "model_tree_roots.csv", index=False)
    return metrics, roots


def plot_outputs(first_vs_later: pd.DataFrame, round_dynamics: pd.DataFrame, lag: pd.DataFrame) -> None:
    plt = safe_import_matplotlib()
    if plt is None:
        return

    fv = first_vs_later.copy()
    fv["model_slug"] = pd.Categorical(fv["model_slug"], categories=MODEL_ORDER, ordered=True)
    pivot = fv.pivot(index="model_slug", columns="turn_phase", values="unsafe_rate").reindex(MODEL_ORDER)
    pivot = pivot.dropna(how="all")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(10, 4.8))
        x = np.arange(len(pivot.index))
        width = 0.36
        if "round_1" in pivot:
            ax.bar(x - width / 2, pivot["round_1"], width=width, label="Round 1", color="#606c38")
        if "round_2plus" in pivot:
            ax.bar(x + width / 2, pivot["round_2plus"], width=width, label="Round 2+", color="#bc6c25")
        ax.set_xticks(x)
        ax.set_xticklabels(pivot.index, rotation=25, ha="right")
        ax.set_ylim(0, 1)
        ax.set_ylabel("Unsafe rate")
        ax.set_title("Baseline first-turn vs later-turn unsafe by model")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "model_first_vs_later_unsafe.png", dpi=180)
        plt.close(fig)

    rd = round_dynamics.copy()
    if not rd.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        for model_slug, sub in rd.groupby("model_slug", sort=False):
            sub = sub.sort_values("round")
            ax.plot(sub["round"], sub["unsafe_rate"], marker="o", linewidth=1.7, label=model_slug)
        ax.set_ylim(0, 1)
        ax.set_xlabel("Round")
        ax.set_ylabel("Unsafe rate")
        ax.set_title("Baseline unsafe dynamics by model")
        ax.legend(fontsize=7, loc="best")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "model_round_dynamics.png", dpi=180)
        plt.close(fig)

    lag_view = lag.copy()
    lag_view["model_slug"] = pd.Categorical(lag_view["model_slug"], categories=MODEL_ORDER, ordered=True)
    pivot = lag_view.pivot(index="model_slug", columns="lag_profile", values="unsafe_rate").reindex(MODEL_ORDER)
    pivot = pivot[["0/0", "0/1", "1/0", "1/1"]].dropna(how="all")
    if not pivot.empty:
        fig, ax = plt.subplots(figsize=(7, 4.6))
        image = ax.imshow(pivot.to_numpy(), cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
        ax.set_xticks(np.arange(len(pivot.columns)))
        ax.set_xticklabels(pivot.columns)
        ax.set_yticks(np.arange(len(pivot.index)))
        ax.set_yticklabels(pivot.index)
        for i in range(pivot.shape[0]):
            for j in range(pivot.shape[1]):
                value = pivot.iloc[i, j]
                if not pd.isna(value):
                    ax.text(j, i, f"{value:.0%}", ha="center", va="center", fontsize=8)
        ax.set_xlabel("Lag profile: own_prev/opponent_prev unsafe")
        ax.set_title("Baseline lag profile unsafe rate by model")
        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / "model_lag_heatmap.png", dpi=180)
        plt.close(fig)


def compact_tree_metrics(metrics: pd.DataFrame) -> pd.DataFrame:
    if metrics.empty:
        return metrics
    return (
        metrics.groupby(["family", "model_slug"], dropna=False)
        .agg(
            folds=("fold", "size"),
            balanced_accuracy=("balanced_accuracy", "mean"),
            roc_auc=("roc_auc", "mean"),
            brier_score=("brier_score", "mean"),
        )
        .reset_index()
    )


def write_report(
    tables: dict[str, pd.DataFrame],
    logit_coefficients: pd.DataFrame,
    logit_metadata: pd.DataFrame,
    tree_metrics: pd.DataFrame,
    tree_roots: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    first_vs_later = tables["first_vs_later"]
    first_turn = tables["first_turn"]
    round_dynamics = tables["round_dynamics"]
    lag = tables["lag"]
    gap_bin = tables["gap_bin"]
    risk_later = tables["risk_later"]
    tree_compact = compact_tree_metrics(tree_metrics)

    fv_pivot = first_vs_later.pivot(index="model_slug", columns="turn_phase", values="unsafe_rate")
    model_gap = pd.DataFrame(
        {
            "model_slug": fv_pivot.index,
            "round_1_unsafe": fv_pivot.get("round_1"),
            "round_2plus_unsafe": fv_pivot.get("round_2plus"),
        }
    ).reset_index(drop=True)
    model_gap["drop_after_round1"] = model_gap["round_2plus_unsafe"] - model_gap["round_1_unsafe"]
    model_gap["model_slug"] = pd.Categorical(model_gap["model_slug"], categories=MODEL_ORDER, ordered=True)
    model_gap = model_gap.sort_values("model_slug")

    lines = [
        "# FH Model Diagnostics",
        "",
        "## Executive Summary",
        "",
        "- **The baseline family split is mostly a model story once you zoom in.** `gpt-5-nano` is the low-unsafe anchor, `gpt-5.4-nano` is much higher, and all three Gemini baseline models remain high even after round 1.",
        "- **Gemini's first-turn saturation is model-wide, not one Gemini outlier.** All Gemini baseline models are unsafe on every first-turn decision across the tested risk levels.",
        "- **Later-turn Gemini is still high but differentiates by model.** `google-gemini-3.1-flash-lite-preview` remains the highest later-turn Gemini model, while `google-gemini-3.5-flash-lite` is the lowest Gemini baseline model.",
        "- **Lag profiles explain more than static risk bins for model behavior.** The heatmap makes the core contrast visible: ChatGPT models have lower `0/0` rates, while Gemini models show very high opponent-triggered unsafe rates.",
        "",
        "## First-Turn Saturation Versus Later-Turn Behavior",
        "",
        "The first-turn table confirms the initial-position signal. Gemini baseline models are saturated at round 1, while ChatGPT models start lower and then diverge by model in later rounds.",
        "",
        markdown_table(model_gap[["model_slug", "round_1_unsafe", "round_2plus_unsafe", "drop_after_round1"]]),
        "",
        "Visual: `figures/model_first_vs_later_unsafe.png`.",
        "",
        "First-turn by risk:",
        "",
        markdown_table(
            first_turn[["family", "model_slug", "max_private_risk", "decisions", "unsafe_rate", "retry_rate"]]
        ),
        "",
        "## Round Dynamics",
        "",
        "The round-by-round view separates two mechanisms: initial unsafe propensity and persistence/response dynamics after interaction history exists.",
        "",
        markdown_table(
            round_dynamics[["family", "model_slug", "round", "decisions", "unsafe_rate"]].head(60)
        ),
        "",
        "Visual: `figures/model_round_dynamics.png`.",
        "",
        "## Lag Profiles By Model",
        "",
        "Lag profiles are the clearest mechanistic diagnostic. `0/0` means both players were safe in the previous round; `0/1` means only the opponent was unsafe; `1/0` means only the current player was unsafe; `1/1` means both were unsafe.",
        "",
        markdown_table(lag[["family", "model_slug", "lag_profile", "decisions", "unsafe_rate"]]),
        "",
        "Visual: `figures/model_lag_heatmap.png`.",
        "",
        "## Risk And Gap Checks",
        "",
        "Risk and progress-gap summaries are useful as descriptive checks, but model-level lag/history remains the stronger explanatory layer.",
        "",
        "Round 2+ by risk:",
        "",
        markdown_table(risk_later[["family", "model_slug", "max_private_risk", "decisions", "unsafe_rate"]]),
        "",
        "Round 2+ by gap bin:",
        "",
        markdown_table(gap_bin[["family", "model_slug", "gap_bin", "decisions", "unsafe_rate"]]),
        "",
        "## Model-Level Logit Checks",
        "",
        "Per-model logits are fit on baseline round 2+ decisions. Use them as sign checks because several models still have sharp separation patterns.",
        "",
        markdown_table(
            logit_coefficients[
                [
                    "family",
                    "model_slug",
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
        )
        if not logit_coefficients.empty
        else "_No model-level logit coefficients._",
        "",
        markdown_table(logit_metadata[["model_slug", "n", "clusters", "status", "error"]]),
        "",
        "## Model-Level Tree Checks",
        "",
        "The model-level trees compress later-turn behavior into shallow rules. They are useful for finding which state variable each model uses first, not for causal interpretation.",
        "",
        markdown_table(tree_compact[["family", "model_slug", "balanced_accuracy", "roc_auc", "brier_score"]])
        if not tree_compact.empty
        else "_No model-level tree metrics._",
        "",
        markdown_table(tree_roots[["family", "model_slug", "root_feature", "root_threshold", "decisions", "clusters"]])
        if not tree_roots.empty
        else "_No model-level tree roots._",
        "",
        "## Decision Implications",
        "",
        "1. In the paper/report, present baseline model diagnostics before family averages. The family average is real, but the model contrast is sharper and less likely to overgeneralize.",
        "2. Treat Gemini first-turn saturation as its own empirical result. Do not hide it by only reporting round 2+ robustness.",
        "3. Use lag-profile heatmaps as the primary mechanistic evidence, then use logits/tree roots as secondary confirmation.",
        "4. For the next stage, run persona-condition diagnostics inside Gemini to identify whether the risk-aware prompt reduces first-turn unsafe or mainly changes later-turn recovery.",
        "",
        "## Caveats",
        "",
        "- Model diagnostics here use completed, non-duplicate baseline rows unless stated otherwise.",
        "- First-turn and later-turn analyses answer different questions; do not collapse them into one causal interpretation.",
        "- Some model-level logits have wide or unstable intervals because the underlying behavior is nearly deterministic in specific cells.",
    ]
    report_path = REPORTS_DIR / "fh_model_diagnostics.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "report": str(report_path.relative_to(REPO_ROOT)),
        "derived_outputs": [
            "model_first_turn_by_risk.csv",
            "model_first_vs_later_summary.csv",
            "model_round_dynamics.csv",
            "model_lag_summary.csv",
            "model_gap_bin_summary.csv",
            "model_round2plus_by_risk.csv",
            "model_logit_coefficients.csv",
            "model_logit_metadata.csv",
            "model_tree_cv_metrics.csv",
            "model_tree_roots.csv",
        ],
        "figures": [
            "model_first_vs_later_unsafe.png",
            "model_round_dynamics.png",
            "model_lag_heatmap.png",
        ],
    }
    (DERIVED_DIR / "model_diagnostics_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )


def main() -> None:
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    turns = load_turns()
    tables = summarize_baseline_models(turns)
    logit_coefficients, logit_metadata = fit_model_logits(tables["baseline_r2"])
    tree_metrics, tree_roots = fit_model_trees(tables["baseline_r2"])
    plot_outputs(tables["first_vs_later"], tables["round_dynamics"], tables["lag"])
    write_report(tables, logit_coefficients, logit_metadata, tree_metrics, tree_roots)
    print(f"Wrote model diagnostics to {REPORTS_DIR / 'fh_model_diagnostics.md'}")


if __name__ == "__main__":
    main()
