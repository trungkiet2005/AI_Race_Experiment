#!/usr/bin/env python3
"""Explain why the AI chooses SAFE vs UNSAFE from turn logs.

This script builds an automatic vectorized feature matrix from game-state + prompt
fields, fits an interpretable surrogate classifier, and exports:

- global feature attribution (coefficient/SHAP-like ranking),
- permutation robustness checks,
- local explanations for representative turns,
- diagnostic summaries for prompt-surface and text-response effects.

It is designed for exploratory audit work, not causal claims. The goal is:
mechanistic attribution, not absolute causality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    log_loss,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


PRESERVE_COLUMNS = [
    "run_root",
    "run_group",
    "run_treatment",
    "run_root_name",
    "lane",
    "game_id",
    "model",
    "max_private_risk",
    "prompt_version",
    "run_phase",
    "rep",
    "persona_condition",
    "seat_persona_role",
    "prompt_template_hash",
    "prompt",
    "raw_response",
    "attempts",
    "prompt_chars",
    "response_chars",
    "round",
    "own_progress_before",
    "opponent_progress_before",
    "progress_gap_before",
    "own_stage_payoff_before",
    "opponent_stage_payoff_before",
    "own_private_risk_before",
    "opponent_private_risk_before",
    "own_prev_action",
    "opponent_prev_action",
    "retry_count",
    "parse_failed",
    "latency_ms",
    "round_payoff",
    "step_increment",
    "stopped",
    "stop_draw",
    "unsafe",
]


NUMERIC_FEATURES = [
    "round",
    "max_private_risk",
    "rep",
    "prompt_chars",
    "response_chars",
    "attempts",
    "own_progress_before",
    "opponent_progress_before",
    "progress_gap_before",
    "own_stage_payoff_before",
    "opponent_stage_payoff_before",
    "own_private_risk_before",
    "opponent_private_risk_before",
    "retry_count",
    "round_payoff",
    "step_increment",
    "stop_draw",
]


BOOL_FEATURES = ["parse_failed", "stopped"]


CATEGORICAL_FEATURES = [
    "run_group",
    "run_treatment",
    "lane",
    "model",
    "run_phase",
    "persona_condition",
    "seat_persona_role",
    "prompt_version",
    "prompt_template_hash",
    "own_prev_action",
    "opponent_prev_action",
]


TEXT_FEATURES = [
    "prompt",
]
OPTIONAL_RESPONSE_TEXT_FEATURE = "raw_response"


PREFERRED_CONTEXT_COLUMNS = [
    "game_id",
    "player",
    "opponent",
    "round",
    "unsafe",
    "prompt_version",
    "run_root_name",
]


def parse_bool(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return float(value)
    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return float(value)
    if isinstance(value, str):
        lower = value.strip().lower()
        if lower in {"true", "1", "yes", "y", "t"}:
            return 1.0
        if lower in {"false", "0", "no", "n", "f"}:
            return 0.0
    return np.nan


def parse_unsafe(row: dict[str, Any]) -> float:
    raw = row.get("unsafe", None)
    parsed = parse_bool(raw)
    if not pd.isna(parsed):
        return parsed
    action = str(row.get("action", "")).strip().lower()
    if action in {"unsafe", "u"}:
        return 1.0
    if action in {"safe", "s"}:
        return 0.0
    return np.nan


def sanitise_prompt_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    # Stable placeholders avoid leaking raw IDs in lexical attribution.
    text = re.sub(r"Company_\d+", "COMPANY", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAI Race\b", "GAME", text, flags=re.IGNORECASE)
    return text.strip()


def prompt_hash(text: str) -> str:
    if not text:
        return "hash__missing"
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]
    return digest


def discover_turn_files(run_root: Path) -> list[Path]:
    return sorted(run_root.rglob("turns.jsonl"))


def load_turns_from_root(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for run in discover_turn_files(run_root):
        rows: list[dict[str, Any]] = []
        with run.open("r", encoding="utf-8") as handle:
            for line_no, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    rows.append(
                        {
                            "run_root": str(run_root),
                            "run_root_name": run_root.name,
                            "run_group": "decode_error",
                            "run_treatment": "decode_error",
                            "lane": "decode_error",
                            "parse_error": f"json_decode_error:{line_no}",
                        }
                    )
                    continue

                # Capture contextual IDs from path so we can compare treatments
                # across files even if schemas differ.
                rel = run.relative_to(run_root).parts
                lane = rel[0] if rel else "lane_unknown"
                treatment = rel[1] if len(rel) > 1 else rel[0]

                row["run_root"] = str(run_root)
                row["run_root_name"] = run_root.name
                row["run_group"] = lane
                row["run_treatment"] = treatment
                row["lane"] = lane
                row["prompt"] = sanitise_prompt_text(row.get("prompt"))
                row["raw_response"] = str(row.get("raw_response", "")).strip()
                row["attempts"] = len(row.get("attempt_history", []) or [])
                row["prompt_chars"] = len(row["prompt"])
                row["response_chars"] = len(row["raw_response"])
                row["prompt_template_hash"] = prompt_hash(row["prompt"])
                row["parse_failed"] = parse_bool(row.get("parse_failed"))
                row["latency_ms"] = row.get("latency_ms")
                row["attempt_count"] = row["attempts"]
                row["unsafe"] = parse_unsafe(row)
                rows.append(row)

                if pd.isna(row["unsafe"]):
                    continue

        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_turns(input_roots: list[Path]) -> pd.DataFrame:
    frames = [load_turns_from_root(root) for root in input_roots]
    tables = [table for table in frames if not table.empty]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def coerce_action_history(row: pd.Series) -> float:
    own = row.get("own_prev_action")
    opp = row.get("opponent_prev_action")
    own_v = parse_bool(own)
    opp_v = parse_bool(opp)
    if not np.isnan(own_v):
        return own_v
    if str(own).strip().lower() in {"unsafe", "u"}:
        return 1.0
    if str(own).strip().lower() in {"safe", "s"}:
        return 0.0
    if not np.isnan(opp_v):
        return opp_v
    if str(opp).strip().lower() in {"unsafe", "u"}:
        return 1.0
    if str(opp).strip().lower() in {"safe", "s"}:
        return 0.0
    return np.nan


def prepare_feature_frame(
    turns: pd.DataFrame,
    include_response_text: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    turns = turns.copy()
    turns = turns[~turns["unsafe"].isna()].copy()
    turns["unsafe"] = turns["unsafe"].astype(float)

    # Fill missing columns used by vectorizers/encoders with explicit missing values.
    for col in PRESERVE_COLUMNS:
        if col not in turns.columns:
            turns[col] = np.nan

    turns["prev_signal"] = turns.apply(coerce_action_history, axis=1)
    turns["run_phase"] = turns["run_phase"].fillna("pilot")
    turns["persona_condition"] = turns["persona_condition"].fillna("none")
    turns["seat_persona_role"] = turns["seat_persona_role"].fillna("none")
    turns["own_prev_action"] = turns["own_prev_action"].where(
        ~turns["own_prev_action"].isna(),
        other="none",
    )
    turns["opponent_prev_action"] = turns["opponent_prev_action"].where(
        ~turns["opponent_prev_action"].isna(),
        other="none",
    )

    turns["run_root_name"] = turns["run_root_name"].fillna("root")
    turns["lane"] = turns["lane"].fillna("unknown")
    turns["run_group"] = turns["run_group"].fillna("unknown")
    turns["run_treatment"] = turns["run_treatment"].fillna("unknown")
    turns["prompt_version"] = turns["prompt_version"].fillna("unknown")
    turns["prompt_template_hash"] = turns["prompt_template_hash"].fillna("unknown")
    turns["model"] = turns["model"].fillna("unknown")
    turns["stopped"] = turns["stopped"].fillna(False)
    turns["stop_draw"] = turns["stop_draw"].fillna(-1.0)

    turns["attempts"] = turns["attempts"].fillna(0)
    turns["prompt"] = turns["prompt"].fillna("")
    turns["raw_response"] = turns["raw_response"].fillna("")
    for numeric in NUMERIC_FEATURES:
        if numeric == "stop_draw":
            # stop_draw is often NaN and should remain numeric.
            turns[numeric] = pd.to_numeric(turns[numeric], errors="coerce")
        else:
            turns[numeric] = pd.to_numeric(turns[numeric], errors="coerce")
    turns["prompt_chars"] = turns["prompt_chars"].fillna(0)
    turns["response_chars"] = turns["response_chars"].fillna(0)
    turns["retry_count"] = pd.to_numeric(turns["retry_count"], errors="coerce").fillna(0)
    turns["rep"] = pd.to_numeric(turns["rep"], errors="coerce").fillna(0).astype(int)

    # Keep an explanatory context table for local cards.
    context = turns[PREFERRED_CONTEXT_COLUMNS].copy()
    context["prompt_template_hash"] = turns["prompt_template_hash"]
    context["player"] = turns["player"]
    context["opponent"] = turns["opponent"]
    context["player_index"] = turns["player_index"] if "player_index" in turns.columns else -1
    context["prompt_template_hash"] = turns["prompt_template_hash"]
    context["unsafe"] = turns["unsafe"]

    target = turns["unsafe"].copy()
    turns["prev_signal"] = turns["prev_signal"].fillna(-1.0)
    selected_features = (
        NUMERIC_FEATURES
        + ["prev_signal"]
        + BOOL_FEATURES
        + CATEGORICAL_FEATURES
        + TEXT_FEATURES
    )
    if include_response_text:
        selected_features = selected_features + [OPTIONAL_RESPONSE_TEXT_FEATURE]

    turns = turns[selected_features].copy()

    return turns, target, context


def build_models(
    turn_features: pd.DataFrame,
    target: pd.Series,
    *,
    max_tfidf_features: int,
    random_state: int,
    include_response_text: bool = False,
):
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=False)),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    prompt_transformer = TfidfVectorizer(
        max_features=max_tfidf_features,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
    )

    response_transformer = TfidfVectorizer(max_features=64, ngram_range=(1, 2), min_df=1)

    text_transformers = [("prompt", prompt_transformer, "prompt")]
    if include_response_text:
        text_transformers.append(("response", response_transformer, OPTIONAL_RESPONSE_TEXT_FEATURE))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES + ["prev_signal"] + BOOL_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            *text_transformers,
        ],
        sparse_threshold=0.3,
    )

    classifier = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    model = Pipeline(steps=[("preprocessor", preprocessor), ("clf", classifier)])

    X_train, X_test, y_train, y_test = train_test_split(
        turn_features,
        target,
        test_size=0.2,
        random_state=random_state,
        stratify=target,
    )
    model.fit(X_train, y_train)
    return model, X_train, X_test, y_train, y_test


def evaluate_model(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
    y_prob = model.predict_proba(X_test)[:, 1]
    y_hat = (y_prob >= 0.5).astype(int)
    auc = float(roc_auc_score(y_test, y_prob))
    acc = float(accuracy_score(y_test, y_hat))
    nll = float(log_loss(y_test, y_prob, labels=[0.0, 1.0]))
    report = classification_report(y_test, y_hat, output_dict=True, zero_division=0)
    return {
        "n_test": int(len(y_test)),
        "roc_auc": auc,
        "accuracy": acc,
        "log_loss": nll,
        "class_0_precision": float(report["0.0"]["precision"]),
        "class_1_precision": float(report["1.0"]["precision"]),
        "class_1_recall": float(report["1.0"]["recall"]),
    }


def _coalesce_feature_importance(model: Pipeline, feature_names: np.ndarray) -> pd.DataFrame:
    coefs = model.named_steps["clf"].coef_[0]
    return pd.DataFrame(
        {
            "feature": feature_names,
            "coef": coefs.astype(float),
            "abs_coef": np.abs(coefs),
            "direction": np.sign(coefs),
        }
    ).sort_values("abs_coef", ascending=False).reset_index(drop=True)


def _permutation_importance(
    model: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    feature_names: np.ndarray,
) -> pd.DataFrame:
    X_transformed = model.named_steps["preprocessor"].transform(X_test)
    X_dense = X_transformed.toarray() if hasattr(X_transformed, "toarray") else X_transformed
    clf = model.named_steps["clf"]
    result = permutation_importance(
        clf,
        X_dense,
        y_test,
        n_repeats=12,
        random_state=123,
        scoring="roc_auc",
        n_jobs=1,
    )
    return pd.DataFrame(
        {
            "feature": feature_names,
            "perm_importance": result.importances_mean,
            "perm_importance_std": result.importances_std,
        }
    ).sort_values("perm_importance", ascending=False).reset_index(drop=True)


def _local_attribution_table(
    model: Pipeline,
    turns: pd.DataFrame,
    context: pd.DataFrame,
    feature_names: np.ndarray,
    top_k: int,
) -> tuple[pd.DataFrame, list[str], str]:
    # Focus on borderline points; where the model is uncertain, attribution is most
    # interpretably useful for debugging surface-protocol sensitivity.
    y_prob = model.predict_proba(turns)[:, 1]
    uncertainty = np.abs(y_prob - 0.5)
    idx = np.argsort(uncertainty)[:top_k]

    transformed = model.named_steps["preprocessor"].transform(turns)
    # The transformed matrix can be sparse; densify a small slice only.
    X_sel = transformed[idx].toarray() if hasattr(transformed[idx], "toarray") else np.asarray(
        transformed[idx]
    )
    coefficients = model.named_steps["clf"].coef_[0]
    local_scores = X_sel * coefficients
    rows: list[dict[str, Any]] = []

    for i, row_ix in enumerate(idx):
        contrib = local_scores[i]
        top_ix = np.argsort(np.abs(contrib))[::-1][:12]
        top_pairs = [
            (feature_names[j], float(contrib[j]))
            for j in top_ix
            if np.isfinite(contrib[j]) and np.abs(float(contrib[j])) > 0.0
        ]
        row = dict(context.iloc[int(row_ix)])
        row.update(
            {
                "local_rank": int(i + 1),
                "predicted_prob_unsafe": float(y_prob[int(row_ix)]),
                "top_feature_count": len(top_pairs),
                "top_features": json.dumps(top_pairs, ensure_ascii=False),
            }
        )
        rows.append(row)

    method_desc = "linear_coefficient_attribution"
    return pd.DataFrame(rows), [], method_desc


def _shape_summary(turns: pd.DataFrame) -> pd.DataFrame:
    if "prompt_template_hash" not in turns.columns:
        return pd.DataFrame()
    frame = (
        turns.assign(
            unsafe=turns["unsafe"].astype(float),
            parse_failed=turns["parse_failed"].astype(float),
        )
        .groupby(["run_group", "run_treatment", "prompt_template_hash", "run_root_name"], dropna=False)
        .agg(
            n_rows=("unsafe", "size"),
            unsafe_rate=("unsafe", "mean"),
            parse_fail_rate=("parse_failed", "mean"),
            mean_prompt_chars=("prompt_chars", "mean"),
            mean_response_chars=("response_chars", "mean"),
            unique_games=("game_id", pd.Series.nunique),
        )
        .reset_index()
        .sort_values("unsafe_rate", ascending=False)
    )
    return frame


def _to_text(value: Any) -> str:
    return str(value) if value is not None else ""


def write_markdown_summary(
    output_dir: Path,
    model_metrics: dict[str, Any],
    global_importance: pd.DataFrame,
    permutation: pd.DataFrame,
    surface_summary: pd.DataFrame,
    local: pd.DataFrame,
    explainability_method: str,
) -> Path:
    text = [
        "# XAI Action Explainability (Auto Vector Encoder)",
        "",
        f"- Trained model: logistic regression on engineered + auto-vectorized prompt/response fields.",
        f"- Method used for local attribution: `{explainability_method}`.",
        f"- Test AUC: **{model_metrics['roc_auc']:.4f}**",
        f"- Test accuracy: **{model_metrics['accuracy']:.4f}**",
        f"- Test log-loss: **{model_metrics['log_loss']:.4f}**",
        "",
        "## Top global driving features",
        "",
        "| rank | feature | direction | |coef| |",
        "|---:|---|---:|---:|",
    ]
    for i, row in global_importance.head(20).iterrows():
        direction = "unsafe+" if row["direction"] > 0 else "safe+"
        text.append(
            f"| {i + 1} | `{row['feature']}` | {direction} | {row['abs_coef']:.4f} |"
        )

    text.append("")
    text.append("## Permutation summary (validation split)")
    text.append("")
    text.append("| rank | feature | perm_importance | std |")
    text.append("|---:|---|---:|---:|")
    for i, row in permutation.head(20).iterrows():
        text.append(
            f"| {i + 1} | `{_to_text(row['feature'])}` | {float(row['perm_importance']):.5f} | {float(row['perm_importance_std']):.5f} |"
        )

    text.append("")
    text.append("## Prompt-template summary")
    text.append("")
    text.append("| run_group | run_treatment | prompt_hash | n_rows | unsafe_rate | parse_fail |")
    text.append("|---|---|---|---:|---:|---:|")
    for _, row in surface_summary.head(30).iterrows():
        text.append(
            f"| {_to_text(row.get('run_group'))} | {_to_text(row.get('run_treatment'))} | {_to_text(row.get('prompt_template_hash'))} | "
            f"{int(row['n_rows'])} | {float(row['unsafe_rate']):.3f} | {float(row['parse_fail_rate']):.3f} |"
        )

    text.append("")
    text.append("## Representative local explanations")
    text.append("")
    text.append("| rank | game_id | round | prob_unsafe | top_features (feature, signed weight) |")
    text.append("|---:|---|---:|---:|---|")
    for _, row in local.head(20).iterrows():
        text.append(
            f"| {int(row['local_rank'])} | {row.get('game_id','')} | {row.get('round','')} | {float(row['predicted_prob_unsafe']):.3f} | {row.get('top_features','')} |"
        )

    output_path = output_dir / "xai_markdown_summary.md"
    output_path.write_text("\n".join(text) + "\n", encoding="utf-8")
    return output_path


def build_and_write(
    turns: pd.DataFrame,
    output_dir: Path,
    *,
    max_tfidf_features: int,
    top_local_examples: int,
    random_state: int,
    include_response_text: bool = False,
) -> dict[str, Any]:
    features, target, context = prepare_feature_frame(
        turns, include_response_text=include_response_text
    )
    model, X_train, X_test, y_train, y_test = build_models(
        features,
        target,
        max_tfidf_features=max_tfidf_features,
        random_state=random_state,
        include_response_text=include_response_text,
    )

    metrics = evaluate_model(model, X_test, y_test)
    model.fit(features, target)

    preprocessor = model.named_steps["preprocessor"]
    feature_names = np.array(preprocessor.get_feature_names_out())
    global_importance = _coalesce_feature_importance(model, feature_names)
    permutation = _permutation_importance(model, X_test, y_test, feature_names)

    local_rows, _, explainability_method = _local_attribution_table(
        model,
        X_test,
        context.loc[X_test.index].reset_index(drop=True),
        feature_names,
        top_k=top_local_examples,
    )

    prompt_surface = _shape_summary(turns.assign(unsafe=turns["unsafe"]))
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = output_dir / "xai_surrogate_pipeline.pkl"
    # Keep a lightweight metadata trail; do not serialize model by default because
    # this repository mainly tracks research artifacts and reproducible CSV/JSON.
    metrics["n_rows"] = int(len(target))
    metrics["n_features"] = int(features.shape[1])
    metrics["n_vectorized_features"] = int(len(feature_names))
    metrics["random_state"] = random_state
    metrics["top_tfidf_features"] = max_tfidf_features

    global_importance.head(200).to_csv(
        output_dir / "xai_global_importance.csv", index=False
    )
    permutation.head(200).to_csv(
        output_dir / "xai_permutation_importance.csv", index=False
    )
    local_rows.to_csv(output_dir / "xai_local_explanations.csv", index=False)
    prompt_surface.to_csv(output_dir / "xai_prompt_surface_summary.csv", index=False)

    with (output_dir / "xai_model_metadata.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2, ensure_ascii=False)

    with (output_dir / "xai_target_distribution.json").open("w", encoding="utf-8") as file:
        json.dump(
            {
                "n_unsafe": int((target == 1.0).sum()),
                "n_safe": int((target == 0.0).sum()),
                "unsafe_rate": float(target.mean()),
            },
            file,
            indent=2,
            ensure_ascii=False,
        )

    summary_path = write_markdown_summary(
        output_dir,
        metrics,
        global_importance,
        permutation,
        prompt_surface,
        local_rows,
        explainability_method,
    )

    if turns is not None and not turns.empty:
        turns.to_csv(output_dir / "xai_input_snapshot.csv", index=False)

    return {
        "model_metadata_path": str((output_dir / "xai_model_metadata.json").resolve()),
        "global_importance_path": str((output_dir / "xai_global_importance.csv").resolve()),
        "prompt_surface_path": str((output_dir / "xai_prompt_surface_summary.csv").resolve()),
        "summary_path": str(summary_path.resolve()),
        "local_path": str((output_dir / "xai_local_explanations.csv").resolve()),
        "method": explainability_method,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a post-hoc explainability audit for SAFE/UNSAFE policy decisions."
    )
    parser.add_argument(
        "--input-root",
        action="append",
        required=True,
        help="Root directory containing one or more turn logs (turns.jsonl). Repeatable.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write explainability artifacts.",
    )
    parser.add_argument(
        "--max-tfidf-features",
        type=int,
        default=300,
        help="Max prompt TF-IDF feature size.",
    )
    parser.add_argument(
        "--top-local-examples",
        type=int,
        default=20,
        help="How many local explanations to export.",
    )
    parser.add_argument(
        "--include-response-text",
        action="store_true",
        help="Include raw_response text in featureization (highly informative but target-leaky).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_roots = [Path(root) for root in args.input_root]
    turns = load_turns(input_roots)

    if turns.empty:
        raise SystemExit("No valid turns found from provided --input-root paths.")

    output_dir = Path(args.output_dir)
    report = build_and_write(
        turns,
        output_dir,
        max_tfidf_features=args.max_tfidf_features,
        top_local_examples=args.top_local_examples,
        random_state=args.random_state,
        include_response_text=args.include_response_text,
    )

    print("XAI report written to:")
    for key, value in report.items():
        print(f"- {key}: {value}")


if __name__ == "__main__":
    main()
