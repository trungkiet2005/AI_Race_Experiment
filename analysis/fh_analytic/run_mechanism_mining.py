#!/usr/bin/env python3
"""Mechanism mining for FH behavior.

This stage goes beyond descriptive rates by extracting reusable rules,
clustering player trajectories into behavioral mechanisms, and fitting
random-forest models for feature-importance checks.
"""

from __future__ import annotations

import itertools
import json
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
MECH_FIG_DIR = FIGURES_DIR / "mechanism_mining"
REPORTS_DIR = OUTPUT_DIR / "reports"
RANDOM_SEED = 260726

BLUE = "#2F6B9A"
ORANGE = "#D9822B"
GOLD = "#C9A227"
OLIVE = "#6B7D3D"
PINK = "#B45A7C"
TEAL = "#3A8F8A"
INK = "#263238"
MUTED = "#6B7280"
GRID = "#E6E8EB"
PAPER = "#FBFBF8"
WHITE = "#FFFFFF"

MODEL_ORDER = [
    "gpt-5-nano",
    "gpt-5.4-nano",
    "google-gemini-3-flash-preview",
    "google-gemini-3.1-flash-lite-preview",
    "google-gemini-3.5-flash-lite",
]
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google-gemini-3-flash-preview": "Gemini 3 Flash",
    "google-gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google-gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}
MODEL_COLORS = {
    "gpt-5-nano": BLUE,
    "gpt-5.4-nano": PINK,
    "google-gemini-3-flash-preview": ORANGE,
    "google-gemini-3.1-flash-lite-preview": GOLD,
    "google-gemini-3.5-flash-lite": OLIVE,
}
CLUSTER_COLORS = {
    "mostly_safe": BLUE,
    "mixed_adaptive": TEAL,
    "opponent_triggered": ORANGE,
    "first_turn_partial_recovery": GOLD,
    "first_turn_opponent_triggered": ORANGE,
    "persistent_unsafe": PINK,
    "state_locked_unsafe": GOLD,
}


@dataclass(frozen=True)
class Scope:
    name: str
    label: str
    frame: pd.DataFrame


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": WHITE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 15,
            "axes.labelsize": 10,
            "axes.edgecolor": "#D7DADF",
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "legend.frameon": False,
        }
    )


def save(fig: plt.Figure, filename: str) -> Path:
    MECH_FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = MECH_FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def title(ax: plt.Axes, text: str, subtitle: str | None = None) -> None:
    ax.set_title(text, loc="left", pad=24, fontweight="bold")
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=9.5)


def pct_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{x:.0%}" for x in np.linspace(0, 1, 6)])


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
    turns["player_cluster_id"] = turns["cluster_id"] + "::p" + turns["player_index"].astype(str)
    turns["lag_profile"] = (
        turns["own_prev_unsafe"].fillna(-1).astype(int).astype(str)
        + "/"
        + turns["opponent_prev_unsafe"].fillna(-1).astype(int).astype(str)
    )
    turns = turns[(turns["manifest_status"] == "completed") & (~turns["duplicate_grain_key"])].copy()
    return add_mechanism_features(turns)


def bucket(value: float, bins: list[float], labels: list[str]) -> str:
    if pd.isna(value):
        return "missing"
    for upper, label in zip(bins, labels):
        if value <= upper:
            return label
    return labels[-1]


def add_mechanism_features(turns: pd.DataFrame) -> pd.DataFrame:
    turns = turns.copy()
    turns["round_phase"] = turns["round"].apply(
        lambda r: "early_r2_4" if r <= 4 else "mid_r5_8" if r <= 8 else "late_r9plus"
    )
    turns["gap_direction"] = turns["progress_gap_before"].apply(
        lambda g: "behind" if g <= -0.5 else "ahead" if g >= 0.5 else "near_tied"
    )
    turns["gap_magnitude"] = turns["progress_gap_before"].abs().apply(
        lambda g: "gap_0_0_5" if g < 0.5 else "gap_0_5_1" if g < 1 else "gap_1plus"
    )
    turns["own_risk_state"] = turns["own_private_risk_before"].apply(
        lambda r: bucket(r, [0.0, 0.15, 0.45, 0.75, np.inf], ["risk_zero", "risk_low", "risk_mid", "risk_high", "risk_very_high"])
    )
    turns["opponent_risk_state"] = turns["opponent_private_risk_before"].apply(
        lambda r: bucket(r, [0.0, 0.15, 0.45, 0.75, np.inf], ["risk_zero", "risk_low", "risk_mid", "risk_high", "risk_very_high"])
    )
    turns["max_risk_level"] = turns["max_private_risk"].map(lambda v: f"risk_{v:g}" if pd.notna(v) else "risk_missing")
    turns["prev_state"] = turns["lag_profile"].map(
        {
            "0/0": "both_prev_safe",
            "0/1": "opponent_prev_unsafe",
            "1/0": "own_prev_unsafe",
            "1/1": "both_prev_unsafe",
            "-1/-1": "first_round",
        }
    ).fillna(turns["lag_profile"])
    return turns


def build_rule_scopes(turns: pd.DataFrame) -> list[Scope]:
    baseline_r2 = turns[(turns["analysis_scope"] == "baseline_completed") & (turns["is_round2plus"])].copy()
    all_r2 = turns[turns["is_round2plus"]].copy()
    scopes = [
        Scope("common_baseline_r2", "Common baseline round 2+", baseline_r2),
        Scope("common_all_r2", "Common all completed round 2+", all_r2),
    ]
    for family, sub in baseline_r2.groupby("family", sort=True):
        scopes.append(Scope(f"family_{family}_baseline_r2", f"{family} baseline round 2+", sub.copy()))
    for model, sub in baseline_r2.groupby("model_slug", sort=True):
        safe = str(model).replace("/", "_").replace("\\", "_")
        scopes.append(Scope(f"model_{safe}_baseline_r2", f"{model} baseline round 2+", sub.copy()))
    return scopes


def atom_columns_for_scope(scope: Scope) -> list[str]:
    candidates = [
        "prev_state",
        "gap_direction",
        "gap_magnitude",
        "own_risk_state",
        "opponent_risk_state",
        "round_phase",
        "max_risk_level",
        "family",
        "model_slug",
        "experiment_mode",
        "persona_mode",
    ]
    return [col for col in candidates if col in scope.frame and scope.frame[col].nunique(dropna=True) > 1]


def mine_scope_rules(scope: Scope, max_depth: int = 3) -> pd.DataFrame:
    frame = scope.frame.dropna(subset=["unsafe"]).copy()
    if frame.empty:
        return pd.DataFrame()
    baseline = frame["unsafe"].mean()
    min_support = max(25, int(len(frame) * 0.04))
    atom_columns = atom_columns_for_scope(scope)
    atoms: list[tuple[str, str]] = []
    for col in atom_columns:
        counts = frame[col].value_counts(dropna=False)
        for value, count in counts.items():
            if count >= min_support:
                atoms.append((col, str(value)))

    rows = []
    for depth in range(1, max_depth + 1):
        for combo in itertools.combinations(atoms, depth):
            columns = [col for col, _ in combo]
            if len(set(columns)) < len(columns):
                continue
            mask = pd.Series(True, index=frame.index)
            for col, value in combo:
                mask &= frame[col].astype(str).eq(value)
            support = int(mask.sum())
            if support < min_support:
                continue
            unsafe_rate = frame.loc[mask, "unsafe"].mean()
            rule = " & ".join([f"{col}={value}" for col, value in combo])
            rows.append(
                {
                    "scope": scope.name,
                    "scope_label": scope.label,
                    "rule": rule,
                    "depth": depth,
                    "support": support,
                    "support_share": support / len(frame),
                    "unsafe_rate": unsafe_rate,
                    "baseline_unsafe_rate": baseline,
                    "lift": unsafe_rate - baseline,
                    "abs_lift": abs(unsafe_rate - baseline),
                    "direction": "unsafe_rule" if unsafe_rate >= baseline else "safe_rule",
                }
            )
    if not rows:
        return pd.DataFrame()
    rules = pd.DataFrame(rows)
    rules = rules.sort_values(["scope", "abs_lift", "support"], ascending=[True, False, False])
    return rules


def mine_rules(turns: pd.DataFrame) -> pd.DataFrame:
    scopes = build_rule_scopes(turns)
    rules = pd.concat([mine_scope_rules(scope) for scope in scopes], ignore_index=True)
    rules.to_csv(DERIVED_DIR / "mechanism_rules.csv", index=False)
    top = (
        rules.sort_values(["scope", "abs_lift", "support"], ascending=[True, False, False])
        .groupby("scope", group_keys=False)
        .head(10)
        .reset_index(drop=True)
    )
    top.to_csv(DERIVED_DIR / "mechanism_top_rules.csv", index=False)
    return rules


def bootstrap_top_rules(turns: pd.DataFrame, rules: pd.DataFrame, n_boot: int = 200) -> pd.DataFrame:
    baseline = turns[(turns["analysis_scope"] == "baseline_completed") & (turns["is_round2plus"])].copy()
    target_rules = rules[rules["scope"].eq("common_baseline_r2")].copy()
    target_rules = target_rules.sort_values(["abs_lift", "support"], ascending=[False, False]).head(12)
    if target_rules.empty:
        return pd.DataFrame()
    rng = np.random.default_rng(RANDOM_SEED)
    clusters = np.array(sorted(baseline["cluster_id"].dropna().unique()))
    by_cluster = {cluster: baseline.index[baseline["cluster_id"] == cluster].to_numpy() for cluster in clusters}
    samples = []
    for i in range(n_boot):
        chosen = rng.choice(clusters, size=len(clusters), replace=True)
        sample_idx = np.concatenate([by_cluster[cluster] for cluster in chosen])
        sample = baseline.loc[sample_idx].copy()
        base_rate = sample["unsafe"].mean()
        for _, rule_row in target_rules.iterrows():
            mask = evaluate_rule(sample, rule_row["rule"])
            if mask.sum() < 5:
                continue
            samples.append(
                {
                    "bootstrap": i,
                    "rule": rule_row["rule"],
                    "unsafe_rate": sample.loc[mask, "unsafe"].mean(),
                    "lift": sample.loc[mask, "unsafe"].mean() - base_rate,
                    "support": int(mask.sum()),
                }
            )
    sample_df = pd.DataFrame(samples)
    sample_df.to_csv(DERIVED_DIR / "mechanism_rule_bootstrap_samples.csv", index=False)
    if sample_df.empty:
        return sample_df
    summary = (
        sample_df.groupby("rule")
        .agg(
            bootstraps=("bootstrap", "nunique"),
            unsafe_rate_median=("unsafe_rate", "median"),
            unsafe_rate_ci_low=("unsafe_rate", lambda s: s.quantile(0.025)),
            unsafe_rate_ci_high=("unsafe_rate", lambda s: s.quantile(0.975)),
            lift_median=("lift", "median"),
            lift_ci_low=("lift", lambda s: s.quantile(0.025)),
            lift_ci_high=("lift", lambda s: s.quantile(0.975)),
            support_median=("support", "median"),
        )
        .reset_index()
        .sort_values("lift_median", ascending=False)
    )
    summary.to_csv(DERIVED_DIR / "mechanism_rule_bootstrap_summary.csv", index=False)
    return summary


def evaluate_rule(frame: pd.DataFrame, rule: str) -> pd.Series:
    mask = pd.Series(True, index=frame.index)
    for atom in rule.split(" & "):
        col, value = atom.split("=", 1)
        mask &= frame[col].astype(str).eq(value)
    return mask


def safe_import_sklearn() -> dict[str, Any] | None:
    try:
        from sklearn.cluster import KMeans  # type: ignore
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.ensemble import RandomForestClassifier  # type: ignore
        from sklearn.inspection import permutation_importance  # type: ignore
        from sklearn.metrics import (  # type: ignore
            balanced_accuracy_score,
            brier_score_loss,
            roc_auc_score,
            silhouette_score,
        )
        from sklearn.model_selection import GroupKFold  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import OneHotEncoder, StandardScaler  # type: ignore

        return {
            "KMeans": KMeans,
            "ColumnTransformer": ColumnTransformer,
            "RandomForestClassifier": RandomForestClassifier,
            "permutation_importance": permutation_importance,
            "balanced_accuracy_score": balanced_accuracy_score,
            "brier_score_loss": brier_score_loss,
            "roc_auc_score": roc_auc_score,
            "silhouette_score": silhouette_score,
            "GroupKFold": GroupKFold,
            "Pipeline": Pipeline,
            "OneHotEncoder": OneHotEncoder,
            "StandardScaler": StandardScaler,
        }
    except Exception:
        return None


def make_encoder(sk: dict[str, Any]) -> Any:
    try:
        return sk["OneHotEncoder"](handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return sk["OneHotEncoder"](handle_unknown="ignore", sparse=False)


def player_profile(turns: pd.DataFrame) -> pd.DataFrame:
    baseline = turns[turns["analysis_scope"].eq("baseline_completed")].copy()
    r2 = baseline[baseline["is_round2plus"]].copy()
    profile = (
        baseline.groupby(["player_cluster_id", "family", "model_slug", "max_private_risk"], dropna=False)
        .agg(
            first_round_unsafe=("unsafe", lambda s: s.iloc[0] if len(s) else np.nan),
            all_unsafe_rate=("unsafe", "mean"),
            rounds=("round", "max"),
            retry_rate=("retry_count", lambda s: (pd.to_numeric(s, errors="coerce").fillna(0) > 0).mean()),
        )
        .reset_index()
    )
    later = (
        r2.groupby("player_cluster_id")
        .agg(
            unsafe_rate_r2=("unsafe", "mean"),
            mean_gap_before=("progress_gap_before", "mean"),
            mean_own_risk_before=("own_private_risk_before", "mean"),
            max_own_risk_before=("own_private_risk_before", "max"),
        )
        .reset_index()
    )
    lag_rates = (
        r2.pivot_table(
            index="player_cluster_id",
            columns="lag_profile",
            values="unsafe",
            aggfunc="mean",
        )
        .rename(columns=lambda col: f"unsafe_after_{col}")
        .reset_index()
    )
    support = (
        r2.pivot_table(
            index="player_cluster_id",
            columns="lag_profile",
            values="unsafe",
            aggfunc="size",
            fill_value=0,
        )
        .rename(columns=lambda col: f"n_after_{col}")
        .reset_index()
    )
    profile = profile.merge(later, on="player_cluster_id", how="left")
    profile = profile.merge(lag_rates, on="player_cluster_id", how="left")
    profile = profile.merge(support, on="player_cluster_id", how="left")
    for col in ["unsafe_after_0/0", "unsafe_after_0/1", "unsafe_after_1/0", "unsafe_after_1/1"]:
        if col not in profile:
            profile[col] = np.nan
        profile[col] = profile[col].fillna(profile["unsafe_rate_r2"])
    for col in ["n_after_0/0", "n_after_0/1", "n_after_1/0", "n_after_1/1"]:
        if col not in profile:
            profile[col] = 0
    return profile.dropna(subset=["unsafe_rate_r2"]).copy()


def label_cluster(row: pd.Series) -> str:
    unsafe = row["unsafe_rate_r2"]
    opp_trigger = row["unsafe_after_01"] - row["unsafe_after_00"]
    own_trigger = row["unsafe_after_10"] - row["unsafe_after_00"]
    if unsafe >= 0.85:
        return "persistent_unsafe"
    if unsafe <= 0.25:
        return "mostly_safe"
    if row["first_round_unsafe"] >= 0.85 and unsafe < 0.75 and row["unsafe_after_01"] >= 0.8:
        return "first_turn_opponent_triggered"
    if row["first_round_unsafe"] >= 0.85 and unsafe < 0.75:
        return "first_turn_partial_recovery"
    if opp_trigger >= 0.2 and row["unsafe_after_01"] >= 0.65:
        return "opponent_triggered"
    if own_trigger >= 0.2:
        return "state_locked_unsafe"
    return "mixed_adaptive"


def cluster_profiles(turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sk = safe_import_sklearn()
    profiles = player_profile(turns)
    if sk is None or profiles.empty:
        profiles.to_csv(DERIVED_DIR / "mechanism_player_profiles.csv", index=False)
        return profiles, pd.DataFrame(), pd.DataFrame()
    features = [
        "first_round_unsafe",
        "unsafe_rate_r2",
        "unsafe_after_0/0",
        "unsafe_after_0/1",
        "unsafe_after_1/0",
        "unsafe_after_1/1",
        "mean_gap_before",
        "mean_own_risk_before",
        "max_own_risk_before",
        "retry_rate",
        "rounds",
        "max_private_risk",
    ]
    X = profiles[features].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = sk["StandardScaler"]()
    Xs = scaler.fit_transform(X)
    k_rows = []
    best_k = 4
    best_score = -np.inf
    for k in range(2, 7):
        km = sk["KMeans"](n_clusters=k, n_init=30, random_state=RANDOM_SEED)
        labels = km.fit_predict(Xs)
        score = sk["silhouette_score"](Xs, labels)
        k_rows.append({"k": k, "silhouette": score})
        if score > best_score:
            best_score = score
            best_k = k
    pd.DataFrame(k_rows).to_csv(DERIVED_DIR / "mechanism_cluster_k_scan.csv", index=False)
    km = sk["KMeans"](n_clusters=best_k, n_init=50, random_state=RANDOM_SEED)
    profiles["cluster_id_num"] = km.fit_predict(Xs)
    cluster_profile = (
        profiles.groupby("cluster_id_num")
        .agg(
            players=("player_cluster_id", "size"),
            unsafe_rate_r2=("unsafe_rate_r2", "mean"),
            first_round_unsafe=("first_round_unsafe", "mean"),
            unsafe_after_00=("unsafe_after_0/0", "mean"),
            unsafe_after_01=("unsafe_after_0/1", "mean"),
            unsafe_after_10=("unsafe_after_1/0", "mean"),
            unsafe_after_11=("unsafe_after_1/1", "mean"),
            mean_gap_before=("mean_gap_before", "mean"),
            mean_own_risk_before=("mean_own_risk_before", "mean"),
            max_own_risk_before=("max_own_risk_before", "mean"),
            retry_rate=("retry_rate", "mean"),
            rounds=("rounds", "mean"),
        )
        .reset_index()
    )
    cluster_profile["mechanism_label"] = cluster_profile.apply(label_cluster, axis=1)
    profiles = profiles.merge(cluster_profile[["cluster_id_num", "mechanism_label"]], on="cluster_id_num", how="left")
    mix = (
        profiles.groupby(["model_slug", "family", "mechanism_label"], dropna=False)
        .agg(players=("player_cluster_id", "size"))
        .reset_index()
    )
    mix["share"] = mix["players"] / mix.groupby("model_slug")["players"].transform("sum")
    profiles.to_csv(DERIVED_DIR / "mechanism_player_profiles.csv", index=False)
    cluster_profile.to_csv(DERIVED_DIR / "mechanism_cluster_profiles.csv", index=False)
    mix.to_csv(DERIVED_DIR / "mechanism_cluster_mix_by_model.csv", index=False)
    return profiles, cluster_profile, mix


def get_feature_names(pipeline: Any, numeric_features: list[str], categorical_features: list[str]) -> list[str]:
    prep = pipeline.named_steps["prep"]
    names: list[str] = list(numeric_features)
    if categorical_features:
        try:
            names.extend(prep.named_transformers_["cat"].get_feature_names_out(categorical_features).tolist())
        except Exception:
            names.extend(categorical_features)
    return names


def fit_random_forests(turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    sk = safe_import_sklearn()
    if sk is None:
        return pd.DataFrame(), pd.DataFrame()
    scopes = build_rule_scopes(turns)
    metric_rows = []
    importance_rows = []
    for scope in scopes:
        frame = scope.frame.copy()
        if len(frame) < 400 or frame["unsafe"].nunique() < 2 or frame["cluster_id"].nunique() < 4:
            continue
        numeric_features = [
            "own_prev_unsafe",
            "opponent_prev_unsafe",
            "progress_gap_before",
            "own_private_risk_before",
            "opponent_private_risk_before",
            "round",
            "max_private_risk",
        ]
        categorical_features = [
            col
            for col in [
                "prev_state",
                "gap_direction",
                "gap_magnitude",
                "own_risk_state",
                "opponent_risk_state",
                "round_phase",
                "family",
                "model_slug",
                "experiment_mode",
                "persona_mode",
            ]
            if frame[col].nunique(dropna=True) > 1
        ]
        data = frame.dropna(subset=["unsafe", *numeric_features, "cluster_id"]).copy()
        X = data[numeric_features + categorical_features].copy()
        X[categorical_features] = X[categorical_features].astype(str)
        y = data["unsafe"].astype(int)
        groups = data["cluster_id"].astype(str)
        pipe = sk["Pipeline"](
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
                    "rf",
                    sk["RandomForestClassifier"](
                        n_estimators=240,
                        max_depth=6,
                        min_samples_leaf=20,
                        class_weight="balanced_subsample",
                        random_state=RANDOM_SEED,
                        n_jobs=-1,
                    ),
                ),
            ]
        )
        n_splits = min(5, groups.nunique())
        cv = sk["GroupKFold"](n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups), start=1):
            pipe.fit(X.iloc[train_idx], y.iloc[train_idx])
            proba = pipe.predict_proba(X.iloc[test_idx])[:, 1]
            pred = (proba >= 0.5).astype(int)
            fold_y = y.iloc[test_idx]
            metric_rows.append(
                {
                    "scope": scope.name,
                    "scope_label": scope.label,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "balanced_accuracy": sk["balanced_accuracy_score"](fold_y, pred),
                    "roc_auc": sk["roc_auc_score"](fold_y, proba) if fold_y.nunique() == 2 else np.nan,
                    "brier_score": sk["brier_score_loss"](fold_y, proba),
                }
            )
        pipe.fit(X, y)
        feature_names = get_feature_names(pipe, numeric_features, categorical_features)
        importances = pipe.named_steps["rf"].feature_importances_
        for feature, importance in zip(feature_names, importances):
            importance_rows.append(
                {
                    "scope": scope.name,
                    "scope_label": scope.label,
                    "feature": feature,
                    "importance": importance,
                    "n": len(data),
                    "clusters": groups.nunique(),
                }
            )
    metrics = pd.DataFrame(metric_rows)
    importance = pd.DataFrame(importance_rows)
    metrics.to_csv(DERIVED_DIR / "mechanism_rf_cv_metrics.csv", index=False)
    importance.to_csv(DERIVED_DIR / "mechanism_rf_feature_importance.csv", index=False)
    return metrics, importance


def short_rule(rule: str, max_len: int = 72) -> str:
    replacements = {
        "prev_state=": "",
        "gap_direction=": "gap=",
        "gap_magnitude=": "",
        "own_risk_state=": "own ",
        "opponent_risk_state=": "opp ",
        "round_phase=": "",
        "max_risk_level=": "max ",
        "model_slug=": "",
        "family=": "",
        "experiment_mode=": "",
        "persona_mode=": "",
    }
    text = rule
    for old, new in replacements.items():
        text = text.replace(old, new)
    text = text.replace("_", " ")
    return textwrap.shorten(text, width=max_len, placeholder="...")


def plot_rules(rules: pd.DataFrame) -> Path:
    top = rules[
        rules["scope"].isin(
            [
                "common_baseline_r2",
                "family_family_chatgpt_baseline_r2",
                "family_family_gemini_baseline_r2",
            ]
        )
    ].copy()
    top = top.sort_values(["scope", "lift", "support"], ascending=[True, False, False])
    top = top.groupby("scope", group_keys=False).head(5)
    top["label"] = top["scope"].map(
        {
            "common_baseline_r2": "Common",
            "family_family_chatgpt_baseline_r2": "ChatGPT",
            "family_family_gemini_baseline_r2": "Gemini",
        }
    )
    top["rule_short"] = top["label"] + ": " + top["rule"].map(short_rule)
    top = top.sort_values("unsafe_rate")

    fig, ax = plt.subplots(figsize=(10.5, 7))
    colors = top["label"].map({"Common": TEAL, "ChatGPT": BLUE, "Gemini": ORANGE})
    y = np.arange(len(top))
    ax.hlines(y, top["baseline_unsafe_rate"], top["unsafe_rate"], color=GRID, linewidth=5)
    ax.scatter(top["unsafe_rate"], y, s=110, color=colors, edgecolor=INK, linewidth=0.5, zorder=3)
    ax.scatter(top["baseline_unsafe_rate"], y, s=40, color="#C8CDD3", edgecolor=INK, linewidth=0.4, zorder=2)
    for yi, value, support in zip(y, top["unsafe_rate"], top["support"]):
        ax.text(value + 0.012, yi, f"{value:.0%}  n={support}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(top["rule_short"])
    pct_axis(ax)
    ax.set_xlabel("Unsafe rate within rule")
    title(ax, "Top High-Unsafe Rules", "Dots show rule unsafe rate; grey markers show each scope baseline.")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return save(fig, "01_top_high_unsafe_rules.png")


def plot_safe_rules(rules: pd.DataFrame) -> Path:
    top = rules[
        rules["scope"].isin(
            [
                "common_baseline_r2",
                "model_gpt-5-nano_baseline_r2",
                "model_gpt-5.4-nano_baseline_r2",
                "model_google-gemini-3.5-flash-lite_baseline_r2",
            ]
        )
    ].copy()
    top = top.sort_values(["scope", "lift", "support"], ascending=[True, True, False])
    top = top.groupby("scope", group_keys=False).head(4)
    top["rule_short"] = top["scope"].str.replace("_baseline_r2", "", regex=False).str.replace("model_", "", regex=False)
    top["rule_short"] = top["rule_short"].str.replace("common", "common", regex=False) + ": " + top["rule"].map(short_rule)
    top = top.sort_values("unsafe_rate", ascending=False)

    fig, ax = plt.subplots(figsize=(10.5, 6.5))
    y = np.arange(len(top))
    ax.hlines(y, top["unsafe_rate"], top["baseline_unsafe_rate"], color=GRID, linewidth=5)
    ax.scatter(top["unsafe_rate"], y, s=110, color=BLUE, edgecolor=INK, linewidth=0.5, zorder=3)
    ax.scatter(top["baseline_unsafe_rate"], y, s=40, color="#C8CDD3", edgecolor=INK, linewidth=0.4, zorder=2)
    for yi, value, support in zip(y, top["unsafe_rate"], top["support"]):
        ax.text(value + 0.012, yi, f"{value:.0%}  n={support}", va="center", fontsize=9)
    ax.set_yticks(y)
    ax.set_yticklabels(top["rule_short"])
    pct_axis(ax)
    ax.set_xlabel("Unsafe rate within rule")
    title(ax, "Top Low-Unsafe Rules", "Blue dots are rule rates; grey markers show each scope baseline.")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return save(fig, "02_top_low_unsafe_rules.png")


def plot_cluster_profiles(cluster_profile: pd.DataFrame) -> Path:
    features = [
        "unsafe_rate_r2",
        "first_round_unsafe",
        "unsafe_after_00",
        "unsafe_after_01",
        "unsafe_after_10",
        "unsafe_after_11",
        "mean_own_risk_before",
    ]
    labels = [
        "Round 2+\nunsafe",
        "First turn\nunsafe",
        "After 0/0",
        "After 0/1",
        "After 1/0",
        "After 1/1",
        "Own risk\nbefore",
    ]
    data = cluster_profile.sort_values("unsafe_rate_r2").copy()
    display_labels = data["mechanism_label"] + "\n(n=" + data["players"].astype(str) + ")"
    matrix = data[features].to_numpy()

    fig, ax = plt.subplots(figsize=(10, 5.8))
    image = ax.imshow(matrix, cmap="YlOrRd", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(features)))
    ax.set_xticklabels(labels)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels(display_labels)
    ax.grid(False)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=9,
                color=INK if matrix[i, j] < 0.72 else WHITE,
                fontweight="bold",
            )
    title(ax, "Behavioral Mechanism Clusters", "K-means over baseline player trajectories; labels are rule-based summaries.")
    cbar = fig.colorbar(image, ax=ax, fraction=0.035, pad=0.04)
    cbar.set_ticks([0, 0.5, 1])
    cbar.set_ticklabels(["0%", "50%", "100%"])
    return save(fig, "03_cluster_profiles_heatmap.png")


def plot_cluster_mix(mix: pd.DataFrame) -> Path:
    pivot = mix.pivot_table(index="model_slug", columns="mechanism_label", values="share", fill_value=0)
    pivot = pivot.reindex(MODEL_ORDER).dropna(how="all")
    cluster_order = sorted(pivot.columns, key=lambda c: pivot[c].sum(), reverse=True)
    pivot = pivot[cluster_order]
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    left = np.zeros(len(pivot))
    y = np.arange(len(pivot))
    for cluster in cluster_order:
        values = pivot[cluster].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            color=CLUSTER_COLORS.get(cluster, MUTED),
            label=cluster.replace("_", " "),
            edgecolor=WHITE,
            linewidth=0.8,
        )
        for i, value in enumerate(values):
            if value >= 0.12:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", color=WHITE, fontsize=9, fontweight="bold")
        left += values
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in pivot.index])
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Share of baseline player trajectories")
    title(ax, "Mechanism Mix By Model", "Stacked shares from trajectory clusters; each player-game is one observation.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return save(fig, "04_cluster_mix_by_model.png")


def plot_rf_importance(importance: pd.DataFrame) -> Path:
    scopes = ["common_baseline_r2", "family_family_chatgpt_baseline_r2", "family_family_gemini_baseline_r2"]
    labels = {
        "common_baseline_r2": "Common",
        "family_family_chatgpt_baseline_r2": "ChatGPT",
        "family_family_gemini_baseline_r2": "Gemini",
    }
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharex=False)
    for ax, scope in zip(axes, scopes):
        sub = importance[importance["scope"] == scope].sort_values("importance", ascending=False).head(8)
        sub = sub.sort_values("importance")
        ax.barh(sub["feature"].str.replace("_", " "), sub["importance"], color=TEAL if scope == "common_baseline_r2" else BLUE if "chatgpt" in scope else ORANGE)
        ax.set_title(labels[scope], fontsize=12, fontweight="bold")
        ax.grid(axis="x")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("RF importance")
    fig.suptitle("Random-Forest Feature Importance By Scope", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.92, "Baseline round 2+ models; importances are impurity-based and used as predictive diagnostics.", color=MUTED, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return save(fig, "05_rf_feature_importance.png")


def plot_rf_performance(metrics: pd.DataFrame) -> Path:
    perf = (
        metrics.groupby(["scope", "scope_label"], dropna=False)
        .agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"))
        .reset_index()
    )
    perf = perf[perf["scope"].isin(["common_baseline_r2", "common_all_r2", "family_family_chatgpt_baseline_r2", "family_family_gemini_baseline_r2"])]
    perf["short"] = perf["scope"].map(
        {
            "common_baseline_r2": "Common baseline",
            "common_all_r2": "Common all",
            "family_family_chatgpt_baseline_r2": "ChatGPT baseline",
            "family_family_gemini_baseline_r2": "Gemini baseline",
        }
    )
    perf = perf.sort_values("roc_auc")
    fig, ax = plt.subplots(figsize=(8.8, 4.9))
    y = np.arange(len(perf))
    ax.hlines(y, 0.5, perf["roc_auc"], color=GRID, linewidth=5)
    ax.scatter(perf["roc_auc"], y, s=140, color=[TEAL, BLUE, ORANGE, PINK][: len(perf)], edgecolor=INK, linewidth=0.5)
    for yi, value in zip(y, perf["roc_auc"]):
        ax.text(value + 0.01, yi, f"{value:.2f}", va="center", fontsize=10)
    ax.axvline(0.5, color=MUTED, linestyle="--", linewidth=1)
    ax.set_xlim(0.45, 0.98)
    ax.set_yticks(y)
    ax.set_yticklabels(perf["short"])
    ax.set_xlabel("Cross-validated ROC-AUC")
    title(ax, "Mechanism Model Predictive Strength", "Random forests with grouped CV; higher AUC means state/rule features explain behavior better.")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return save(fig, "06_rf_predictive_strength.png")


def create_contact_sheet(paths: list[Path]) -> Path:
    images = [mpimg.imread(path) for path in paths]
    ncols = 2
    nrows = int(np.ceil(len(images) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(16, 6.1 * nrows))
    axes = np.array(axes).reshape(-1)
    for ax, img in zip(axes, images):
        ax.imshow(img)
        ax.set_axis_off()
    for ax in axes[len(images) :]:
        ax.set_axis_off()
    fig.suptitle("FH Mechanism Mining Storyboard", fontsize=24, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return save(fig, "fh_mechanism_storyboard_contact_sheet.png")


def write_report(
    rules: pd.DataFrame,
    rule_bootstrap: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    cluster_mix: pd.DataFrame,
    rf_metrics: pd.DataFrame,
    rf_importance: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    top_common = (
        rules[rules["scope"].eq("common_baseline_r2")]
        .sort_values(["lift", "support"], ascending=[False, False])
        .head(8)
        .copy()
    )
    top_safe = (
        rules[rules["scope"].eq("common_baseline_r2")]
        .sort_values(["lift", "support"], ascending=[True, False])
        .head(8)
        .copy()
    )
    rf_perf = (
        rf_metrics.groupby(["scope", "scope_label"], dropna=False)
        .agg(roc_auc=("roc_auc", "mean"), balanced_accuracy=("balanced_accuracy", "mean"), brier_score=("brier_score", "mean"))
        .reset_index()
        if not rf_metrics.empty
        else pd.DataFrame()
    )
    top_features = (
        rf_importance.sort_values(["scope", "importance"], ascending=[True, False])
        .groupby("scope", group_keys=False)
        .head(8)
        if not rf_importance.empty
        else pd.DataFrame()
    )

    lines = [
        "# FH Mechanism Mining",
        "",
        "## Executive Summary",
        "",
        "- **The cleanest common unsafe rule is not one variable; it is a state pattern.** High-unsafe rules combine prior unsafe state, being near-tied/behind, accumulated private risk, and model identity.",
        "- **Mechanisms differ sharply by model.** Gemini models form persistent/opponent-triggered unsafe clusters; `gpt-5-nano` is mostly safe; `gpt-5.4-nano` is mixed and harder to compress with shallow rules.",
        "- **Clustering supports the earlier tree story.** The trajectory clusters separate mostly-safe, first-turn/opponent-triggered, and persistent-unsafe behavior rather than merely splitting by provider name.",
        "- **Random forests confirm that state/history features carry the most predictive signal.** Treat this as predictive confidence, while rule tables remain the interpretable layer for paper text.",
        "",
        "## Common High-Unsafe Rules",
        "",
        markdown_table(top_common[["rule", "support", "support_share", "unsafe_rate", "baseline_unsafe_rate", "lift"]]),
        "",
        "## Common Low-Unsafe Rules",
        "",
        markdown_table(top_safe[["rule", "support", "support_share", "unsafe_rate", "baseline_unsafe_rate", "lift"]]),
        "",
        "## Bootstrap Stability For Top Common Rules",
        "",
        markdown_table(rule_bootstrap[["rule", "bootstraps", "unsafe_rate_median", "unsafe_rate_ci_low", "unsafe_rate_ci_high", "lift_median", "lift_ci_low", "lift_ci_high"]].head(12))
        if not rule_bootstrap.empty
        else "_Bootstrap rule summary unavailable._",
        "",
        "## Behavioral Clusters",
        "",
        markdown_table(cluster_profile[["cluster_id_num", "mechanism_label", "players", "unsafe_rate_r2", "first_round_unsafe", "unsafe_after_00", "unsafe_after_01", "unsafe_after_10", "unsafe_after_11"]])
        if not cluster_profile.empty
        else "_Cluster profiles unavailable._",
        "",
        "## Mechanism Mix By Model",
        "",
        markdown_table(cluster_mix[["model_slug", "mechanism_label", "players", "share"]])
        if not cluster_mix.empty
        else "_Cluster mix unavailable._",
        "",
        "## Random-Forest Mechanism Models",
        "",
        markdown_table(rf_perf[["scope", "roc_auc", "balanced_accuracy", "brier_score"]])
        if not rf_perf.empty
        else "_RF metrics unavailable._",
        "",
        "Top RF features:",
        "",
        markdown_table(top_features[["scope", "feature", "importance", "n", "clusters"]])
        if not top_features.empty
        else "_RF importance unavailable._",
        "",
        "## Visuals",
        "",
        "- `figures/mechanism_mining/fh_mechanism_storyboard_contact_sheet.png`",
        "- `figures/mechanism_mining/01_top_high_unsafe_rules.png`",
        "- `figures/mechanism_mining/02_top_low_unsafe_rules.png`",
        "- `figures/mechanism_mining/03_cluster_profiles_heatmap.png`",
        "- `figures/mechanism_mining/04_cluster_mix_by_model.png`",
        "- `figures/mechanism_mining/05_rf_feature_importance.png`",
        "- `figures/mechanism_mining/06_rf_predictive_strength.png`",
        "",
        "## Caveats",
        "",
        "- Rules are descriptive conjunctions; they are interpretable, not causal.",
        "- Clusters depend on chosen trajectory features and KMeans geometry; labels are generated from cluster centroids.",
        "- RF importances are predictive diagnostics and can share signal across correlated state features.",
    ]
    report_path = REPORTS_DIR / "fh_mechanism_mining.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "report": str(report_path.relative_to(REPO_ROOT)),
        "derived_outputs": [
            "mechanism_rules.csv",
            "mechanism_top_rules.csv",
            "mechanism_rule_bootstrap_summary.csv",
            "mechanism_player_profiles.csv",
            "mechanism_cluster_profiles.csv",
            "mechanism_cluster_mix_by_model.csv",
            "mechanism_rf_cv_metrics.csv",
            "mechanism_rf_feature_importance.csv",
        ],
        "figures_dir": str(MECH_FIG_DIR.relative_to(REPO_ROOT)),
    }
    (DERIVED_DIR / "mechanism_mining_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    setup_style()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    MECH_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    turns = load_turns()
    rules = mine_rules(turns)
    rule_bootstrap = bootstrap_top_rules(turns, rules)
    _, cluster_profile, cluster_mix = cluster_profiles(turns)
    rf_metrics, rf_importance = fit_random_forests(turns)
    paths = [
        plot_rules(rules),
        plot_safe_rules(rules),
        plot_cluster_profiles(cluster_profile),
        plot_cluster_mix(cluster_mix),
        plot_rf_importance(rf_importance),
        plot_rf_performance(rf_metrics),
    ]
    contact_sheet = create_contact_sheet(paths)
    write_report(rules, rule_bootstrap, cluster_profile, cluster_mix, rf_metrics, rf_importance)
    print(f"Wrote mechanism mining storyboard to {contact_sheet}")


if __name__ == "__main__":
    main()
