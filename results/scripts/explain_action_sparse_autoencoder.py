#!/usr/bin/env python3
"""Interpret SAFE/UNSAFE logs with a sparse dictionary-learning surrogate.

This script operates on logged prompt/state features.  It is intentionally not
described as a neuron-level sparse autoencoder: no internal model activations are
read here.  Splits are made before feature fitting and are grouped by race by
default so repeated turns from one endogenous trajectory cannot leak across the
evaluation boundary.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import warnings
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import issparse

from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.decomposition import MiniBatchDictionaryLearning
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


TEXT_FEATURES = ["prompt"]
OPTIONAL_RESPONSE_TEXT_FEATURE = "raw_response"

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


PREFERRED_CONTEXT_COLUMNS = [
    "run_root",
    "run_group",
    "run_treatment",
    "game_id",
    "player",
    "opponent",
    "round",
    "unsafe",
    "prompt_version",
    "run_root_name",
]


def _race_group_key(turns: pd.DataFrame) -> pd.Series:
    """Return a globally unique race key across pooled result roots."""
    required = ["run_root", "run_group", "run_treatment", "game_id"]
    missing = [column for column in required if column not in turns.columns]
    if missing:
        raise ValueError(f"Race-group split requires columns: {missing}")
    return turns[required].fillna("missing").astype(str).agg("::".join, axis=1)


def split_turns(
    turns: pd.DataFrame,
    *,
    split_unit: str,
    random_state: int,
    test_size: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Split raw rows before any imputation, scaling, one-hot, or TF-IDF fit."""
    clean = turns[~turns["unsafe"].isna()].copy().reset_index(drop=True)
    clean["unsafe"] = clean["unsafe"].astype(float)
    if split_unit == "row":
        train, test = train_test_split(
            clean,
            test_size=test_size,
            random_state=random_state,
            stratify=clean["unsafe"],
        )
        group_overlap = None
        n_groups_train = None
        n_groups_test = None
    else:
        if split_unit == "race":
            groups = _race_group_key(clean)
        elif split_unit == "prompt_hash":
            groups = clean["prompt_template_hash"].fillna("hash__missing").astype(str)
        else:
            raise ValueError(f"Unknown split_unit={split_unit!r}")
        splitter = GroupShuffleSplit(
            n_splits=1,
            test_size=test_size,
            random_state=random_state,
        )
        train_idx, test_idx = next(splitter.split(clean, clean["unsafe"], groups))
        train = clean.iloc[train_idx]
        test = clean.iloc[test_idx]
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        overlap = train_groups & test_groups
        if overlap:
            raise AssertionError(f"Grouped split leaked {len(overlap)} groups")
        group_overlap = 0
        n_groups_train = len(train_groups)
        n_groups_test = len(test_groups)
    if train["unsafe"].nunique() != 2 or test["unsafe"].nunique() != 2:
        raise ValueError(
            "Both train and test partitions must contain SAFE and UNSAFE labels; "
            f"got train={sorted(train['unsafe'].unique())}, test={sorted(test['unsafe'].unique())}"
        )
    return (
        train.reset_index(drop=True),
        test.reset_index(drop=True),
        {
            "split_unit": split_unit,
            "test_size_requested": test_size,
            "n_groups_train": n_groups_train,
            "n_groups_test": n_groups_test,
            "group_overlap": group_overlap,
        },
    )


def _parse_bool(value: Any) -> float:
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, np.integer)):
        if value in (0, 1):
            return float(value)
    if isinstance(value, (float, np.floating)):
        if value in (0.0, 1.0):
            return float(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"true", "1", "yes", "y", "t"}:
            return 1.0
        if v in {"false", "0", "no", "n", "f"}:
            return 0.0
    return np.nan


def _parse_unsafe(row: dict[str, Any]) -> float:
    unsafe = _parse_bool(row.get("unsafe"))
    if not pd.isna(unsafe):
        return unsafe
    action = str(row.get("action", "")).strip().lower()
    if action in {"unsafe", "u"}:
        return 1.0
    if action in {"safe", "s"}:
        return 0.0
    return np.nan


def _sanitize_prompt_text(text: Any) -> str:
    if not isinstance(text, str):
        return ""
    text = re.sub(r"Company_\d+", "COMPANY", text, flags=re.IGNORECASE)
    text = re.sub(r"\bAI Race\b", "GAME", text, flags=re.IGNORECASE)
    return text.strip()


def _prompt_hash(text: str) -> str:
    if not text:
        return "hash__missing"
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:12]


def discover_turn_files(run_root: Path) -> list[Path]:
    return sorted(run_root.rglob("turns.jsonl"))


def load_turns_from_root(run_root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for turns_path in discover_turn_files(run_root):
        rows: list[dict[str, Any]] = []
        with turns_path.open("r", encoding="utf-8") as handle:
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

                rel = turns_path.relative_to(run_root).parts
                lane = rel[0] if rel else "lane_unknown"
                treatment = rel[1] if len(rel) > 1 else lane

                row["run_root"] = str(run_root)
                row["run_root_name"] = run_root.name
                row["run_group"] = lane
                row["run_treatment"] = treatment
                row["lane"] = lane
                row["prompt"] = _sanitize_prompt_text(row.get("prompt"))
                row["raw_response"] = str(row.get("raw_response", "")).strip()
                row["attempts"] = len(row.get("attempt_history", []) or [])
                row["prompt_chars"] = len(row["prompt"])
                row["response_chars"] = len(row["raw_response"])
                row["prompt_template_hash"] = _prompt_hash(row["prompt"])
                row["parse_failed"] = _parse_bool(row.get("parse_failed"))
                row["attempt_count"] = row["attempts"]
                row["unsafe"] = _parse_unsafe(row)
                rows.append(row)

        if rows:
            frames.append(pd.DataFrame(rows))

    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def load_turns(input_roots: list[Path]) -> pd.DataFrame:
    tables = [load_turns_from_root(root) for root in input_roots]
    tables = [t for t in tables if not t.empty]
    if not tables:
        return pd.DataFrame()
    return pd.concat(tables, ignore_index=True)


def _normalize_feature_frame(turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Normalize schema without learning any statistics from the rows."""
    turns = turns.copy()
    turns = turns[~turns["unsafe"].isna()].copy()
    turns["unsafe"] = turns["unsafe"].astype(float)

    # Fill schema.
    for c in [
        "run_root",
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
        "prompt",
        "raw_response",
    ]:
        if c not in turns.columns:
            turns[c] = "none"

    turns["run_phase"] = turns["run_phase"].fillna("pilot")
    turns["persona_condition"] = turns["persona_condition"].fillna("none")
    turns["seat_persona_role"] = turns["seat_persona_role"].fillna("none")
    turns["own_prev_action"] = turns["own_prev_action"].fillna("none")
    turns["opponent_prev_action"] = turns["opponent_prev_action"].fillna("none")
    turns["run_root_name"] = turns["run_root_name"].fillna("root")
    turns["stopped"] = turns["stopped"].fillna(False)
    turns["stop_draw"] = pd.to_numeric(turns["stop_draw"], errors="coerce")

    for col in NUMERIC_FEATURES:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")
    turns["prompt_chars"] = turns["prompt_chars"].fillna(0)
    turns["response_chars"] = turns["response_chars"].fillna(0)
    turns["attempts"] = turns["attempts"].fillna(0)
    turns["retry_count"] = pd.to_numeric(turns["retry_count"], errors="coerce").fillna(0)
    turns["rep"] = pd.to_numeric(turns["rep"], errors="coerce").fillna(0).astype(int)
    turns["stop_draw"] = turns["stop_draw"].fillna(-1.0)
    turns["parse_failed"] = turns["parse_failed"].fillna(0.0)

    context_columns = [column for column in dict.fromkeys(
        PREFERRED_CONTEXT_COLUMNS + ["prompt_template_hash"]
    ) if column in turns.columns]
    context = turns[context_columns].copy()
    context["player_index"] = turns.get("player_index", pd.Series(-1, index=turns.index))
    return turns, turns["unsafe"].astype(float), context


def prepare_features(
    train_turns: pd.DataFrame,
    test_turns: pd.DataFrame,
    *,
    include_response_text: bool,
    max_tfidf_features: int = 300,
    max_learner_features: int = 300,
) -> tuple[Any, Any, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame, np.ndarray]:
    train, y_train, context_train = _normalize_feature_frame(train_turns)
    test, y_test, context_test = _normalize_feature_frame(test_turns)

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
    response_transformer = TfidfVectorizer(
        max_features=128,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.98,
    )

    text_transformers = [("prompt", prompt_transformer, "prompt")]
    if include_response_text:
        text_transformers.append(("response", response_transformer, OPTIONAL_RESPONSE_TEXT_FEATURE))

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, NUMERIC_FEATURES),
            ("cat", categorical_transformer, CATEGORICAL_FEATURES),
            *text_transformers,
        ],
        sparse_threshold=0.3,
    )

    train_matrix = preprocessor.fit_transform(train)
    test_matrix = preprocessor.transform(test)
    feature_names = preprocessor.get_feature_names_out()

    # Select learner features using training variance only, then apply the frozen
    # column set to test.  This prevents held-out rows from influencing the model.
    max_features = max(100, int(max_learner_features))
    if train_matrix.shape[1] > max_features:
        feature_names = np.asarray(feature_names)
        if issparse(train_matrix):
            mean = np.asarray(train_matrix.mean(axis=0)).ravel()
            mean_sq = np.asarray(train_matrix.multiply(train_matrix).mean(axis=0)).ravel()
            variances = mean_sq - mean * mean
            top_idx = np.argsort(np.array(variances))[::-1][:max_features]
            train_matrix = train_matrix[:, top_idx]
            test_matrix = test_matrix[:, top_idx]
            feature_names = feature_names[top_idx]
        else:
            variances = np.var(train_matrix, axis=0)
            top_idx = np.argsort(np.array(variances))[::-1][:max_features]
            train_matrix = train_matrix[:, top_idx]
            test_matrix = test_matrix[:, top_idx]
            feature_names = feature_names[top_idx]

    return (
        train_matrix,
        test_matrix,
        y_train.reset_index(drop=True),
        y_test.reset_index(drop=True),
        context_train.reset_index(drop=True),
        context_test.reset_index(drop=True),
        feature_names,
    )


def build_sparse_code_and_head(
    X_train,
    X_test,
    y_train,
    y_test,
    *,
    code_dim: int,
    alpha: float,
    random_state: int,
    max_iter: int,
) -> dict[str, Any]:
    n_components = min(code_dim, X_train.shape[1])
    n_components = max(1, int(n_components))
    dict_learn = MiniBatchDictionaryLearning(
        n_components=n_components,
        alpha=alpha,
        transform_alpha=alpha,
        fit_algorithm="lars",
        transform_algorithm="omp",
        transform_n_nonzero_coefs=max(1, n_components // 4),
        batch_size=min(512, max(32, len(y_train))),
        # n_jobs=1 is deterministic and avoids environment-specific joblib/psutil
        # process-discovery failures seen on Windows research workstations.
        n_jobs=1,
        random_state=random_state,
        max_iter=max_iter,
    )

    X_train_dense = (
        X_train.toarray().astype(np.float32)
        if hasattr(X_train, "toarray")
        else np.asarray(X_train, dtype=np.float32)
    )
    X_test_dense = (
        X_test.toarray().astype(np.float32)
        if hasattr(X_test, "toarray")
        else np.asarray(X_test, dtype=np.float32)
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        train_code = dict_learn.fit_transform(X_train_dense)
        test_code = dict_learn.transform(X_test_dense)
    convergence_warning_count = sum(
        issubclass(item.category, ConvergenceWarning) for item in caught
    )

    clf = LogisticRegression(
        max_iter=1000,
        solver="liblinear",
        class_weight="balanced",
        random_state=random_state,
    )
    clf.fit(train_code, y_train)
    y_prob = clf.predict_proba(test_code)[:, 1]
    y_hat = (y_prob >= 0.5).astype(int)

    # decode reconstruction for reconstruction quality.
    X_rec = train_code @ dict_learn.components_
    recon_mse = float(np.mean((X_train_dense - X_rec) ** 2))

    report = classification_report(y_test, y_hat, output_dict=True, zero_division=0)
    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_features_input": int(X_train.shape[1]),
        "n_code": int(n_components),
        "roc_auc": float(roc_auc_score(y_test, y_prob)),
        "accuracy": float(accuracy_score(y_test, y_hat)),
        "log_loss": float(log_loss(y_test, y_prob, labels=[0.0, 1.0])),
        "class_0_precision": float(report["0.0"]["precision"]),
        "class_1_precision": float(report["1.0"]["precision"]),
        "class_1_recall": float(report["1.0"]["recall"]),
        "class_0_recall": float(report["0.0"]["recall"]),
        "recon_mse": recon_mse,
        "convergence_warning_count": int(convergence_warning_count),
        "n_rows": int(len(y_train) + len(y_test)),
    }

    return {
        "dict_learn": dict_learn,
        "clf": clf,
        "train_code": train_code,
        "test_code": test_code,
        "y_test_prob": y_prob,
        "y_test_pred": y_hat,
        "metrics": metrics,
    }


def build_code_feature_table(dict_learn: DictionaryLearning, feature_names: np.ndarray, top_k: int = 12) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    components = dict_learn.components_
    for k, comp in enumerate(components):
        top_idx = np.argsort(np.abs(comp))[::-1][:top_k]
        for rank, idx in enumerate(top_idx, start=1):
            rows.append(
                {
                    "code": f"z{k:03d}",
                    "rank": rank,
                    "feature": str(feature_names[int(idx)]),
                    "weight": float(comp[int(idx)]),
                    "abs_weight": float(abs(comp[int(idx)])),
                }
            )
    return pd.DataFrame(rows)


def build_global_importance(
    dict_learn: DictionaryLearning,
    clf: LogisticRegression,
    train_code: np.ndarray,
    feature_names: np.ndarray,
    train_labels: pd.Series,
) -> pd.DataFrame:
    coef = clf.coef_[0]
    coef_abs = np.abs(coef)
    mean_code = train_code.mean(axis=0)
    mean_abs_code = np.abs(train_code).mean(axis=0)
    sparse_ratio = (np.mean(np.abs(train_code) > 1e-6, axis=0)).tolist()
    rows = []
    for i, w in enumerate(coef):
        rows.append(
            {
                "code": f"z{i:03d}",
                "coef": float(w),
                "abs_coef": float(coef_abs[i]),
                "direction": "unsafe+" if w > 0 else "safe+",
                "mean_code": float(mean_code[i]),
                "mean_abs_code": float(mean_abs_code[i]),
                "sparsity_ratio": float(sparse_ratio[i]),
            }
        )
    table = pd.DataFrame(rows).sort_values("abs_coef", ascending=False).reset_index(drop=True)
    table["rank"] = np.arange(1, len(table) + 1, dtype=int)
    return table


def _to_float(v: Any) -> float:
    return float(v) if pd.notna(v) else 0.0


def build_local_explanations(
    code: np.ndarray,
    probs: np.ndarray,
    context: pd.DataFrame,
    clf: LogisticRegression,
    top_k: int = 20,
) -> pd.DataFrame:
    uncertainty = np.abs(probs - 0.5)
    idx = np.argsort(uncertainty)[:top_k]
    coef = clf.coef_[0]
    rows: list[dict[str, Any]] = []
    for i in idx:
        contrib = code[int(i)] * coef
        top_ix = np.argsort(np.abs(contrib))[::-1][:12]
        top_pairs = []
        for j in top_ix:
            if np.isfinite(contrib[j]) and abs(float(contrib[j])) > 0.0:
                top_pairs.append((f"z{int(j):03d}", float(contrib[j])))
        row = dict(context.iloc[int(i)])
        row["local_rank"] = int(len(rows) + 1)
        row["predicted_prob_unsafe"] = float(probs[int(i)])
        row["top_code_contrib"] = json.dumps(top_pairs, ensure_ascii=False)
        rows.append(row)
    return pd.DataFrame(rows)


def write_summary(
    output_dir: Path,
    metrics: dict[str, Any],
    global_importance: pd.DataFrame,
    code_top_features: pd.DataFrame,
    local: pd.DataFrame,
) -> Path:
    lines = [
        "# Sparse Dictionary Action Audit",
        "",
        "> Scope: this feature-space dictionary-learning surrogate uses logged prompts and states. It does not inspect model neurons or establish a causal mechanism.",
        "",
        f"- Evaluation split: **{metrics.get('split_unit', 'unknown')}** (fit after split; group overlap: {metrics.get('group_overlap')})",
        f"- Samples: **{metrics['n_rows']}**",
        f"- Sparse code units: **{metrics['n_code']}**",
        f"- Test AUC: **{metrics['roc_auc']:.4f}**",
        f"- Test accuracy: **{metrics['accuracy']:.4f}**",
        f"- Test log-loss: **{metrics['log_loss']:.4f}**",
        f"- Reconstruction MSE: **{metrics['recon_mse']:.6f}**",
        "",
        "## Global code importance",
        "",
        "| rank | code | direction | coef | mean_abs_code | sparsity_ratio |",
        "|---:|---|---|---:|---:|---:|",
    ]
    for _, row in global_importance.head(30).iterrows():
        lines.append(
            f"| {int(row['rank'])} | `{row['code']}` | {row['direction']} | "
            f"{_to_float(row['coef']):.4f} | {_to_float(row['mean_abs_code']):.4f} | {_to_float(row['sparsity_ratio']):.3f} |"
        )

    lines.extend(
        [
            "",
            "## Top input features for each sparse code",
            "",
            "| code | rank | feature | weight |",
            "|---|---:|---|---:|",
        ]
    )
    for _, row in code_top_features.head(180).iterrows():
        lines.append(
            f"| `{row['code']}` | {int(row['rank'])} | `{row['feature']}` | { _to_float(row['weight']) :.4f} |"
        )

    lines.extend(
        [
            "",
            "## Representative local explanations",
            "",
            "| rank | game_id | round | prob_unsafe | top code contributions (code, signed) |",
            "|---:|---|---:|---:|---|",
        ]
    )
    for _, row in local.iterrows():
        lines.append(
            f"| {int(row['local_rank'])} | {row.get('game_id','')} | {row.get('round','')} | "
            f"{_to_float(row['predicted_prob_unsafe']):.3f} | {row.get('top_code_contrib', '')} |"
        )

    out = output_dir / "xai_sparse_autoencoder_summary.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run sparse-autoencoder-style explainability on SAFE/UNSAFE action logs."
    )
    parser.add_argument("--input-root", action="append", required=True, help="Run directory containing turns.jsonl. Repeatable.")
    parser.add_argument("--output-dir", required=True, help="Folder for outputs.")
    parser.add_argument("--code-dim", type=int, default=24, help="Dictionary size / latent dimensions.")
    parser.add_argument("--sparsity-alpha", type=float, default=1.0, help="DictionaryLearning alpha / transform_alpha.")
    parser.add_argument("--dict-iter", type=int, default=200, help="Dictionary learning iterations.")
    parser.add_argument("--top-local-examples", type=int, default=20, help="Local explanation examples.")
    parser.add_argument("--include-response-text", action="store_true", help="Include raw_response in featureization.")
    parser.add_argument("--random-state", type=int, default=42, help="Random seed.")
    parser.add_argument(
        "--split-unit",
        choices=["race", "prompt_hash", "row"],
        default="race",
        help=(
            "Evaluation split unit. 'race' is the primary leakage-resistant default; "
            "'prompt_hash' is stricter; 'row' is retained only as a leakage diagnostic."
        ),
    )
    parser.add_argument("--max-tfidf-features", type=int, default=300, help="Max TF-IDF prompt vocabulary size.")
    parser.add_argument(
        "--max-learner-features",
        type=int,
        default=300,
        help="Maximum feature count given to DictionaryLearning.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    turns = load_turns([Path(r) for r in args.input_root])
    if turns.empty:
        raise SystemExit("No valid turns found from the input roots.")

    train_turns, test_turns, split_metadata = split_turns(
        turns,
        split_unit=args.split_unit,
        random_state=args.random_state,
    )
    (
        x_train,
        x_test,
        y_train,
        y_test,
        ctx_train,
        ctx_test,
        feature_names,
    ) = prepare_features(
        train_turns,
        test_turns,
        include_response_text=args.include_response_text,
        max_tfidf_features=args.max_tfidf_features,
        max_learner_features=args.max_learner_features,
    )
    fit = build_sparse_code_and_head(
        x_train,
        x_test,
        y_train.to_numpy().astype(float),
        y_test.to_numpy().astype(float),
        code_dim=args.code_dim,
        alpha=args.sparsity_alpha,
        random_state=args.random_state,
        max_iter=args.dict_iter,
    )

    global_importance = build_global_importance(
        fit["dict_learn"],
        fit["clf"],
        fit["train_code"],
        feature_names,
        y_train,
    )
    code_top_features = build_code_feature_table(fit["dict_learn"], feature_names, top_k=12)
    local = build_local_explanations(
        fit["test_code"],
        fit["y_test_prob"],
        ctx_test.reset_index(drop=True),
        fit["clf"],
        top_k=args.top_local_examples,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    global_importance.to_csv(output_dir / "xai_sparse_autoencoder_global_importance.csv", index=False)
    code_top_features.to_csv(output_dir / "xai_sparse_autoencoder_code_features.csv", index=False)
    local.to_csv(output_dir / "xai_sparse_autoencoder_local_explanations.csv", index=False)
    with (output_dir / "xai_sparse_autoencoder_input_snapshot.csv").open("w", encoding="utf-8") as f:
        turns.to_csv(f, index=False)

    with (output_dir / "xae_target_distribution.json").open("w", encoding="utf-8") as f:
        json.dump(
            {
                "n_unsafe": int((turns["unsafe"] == 1.0).sum()),
                "n_safe": int((turns["unsafe"] == 0.0).sum()),
                "unsafe_rate": float(turns["unsafe"].mean()),
                "method": "feature_space_minibatch_dictionary_learning_surrogate",
            },
            f,
            indent=2,
        )

    feature_names_list = [str(v) for v in feature_names]
    model_report = fit["metrics"]
    model_report.update(
        {
            "feature_names_count": int(len(feature_names_list)),
            "sparsity_alpha": args.sparsity_alpha,
            "code_dim_requested": args.code_dim,
            "dict_iterations": args.dict_iter,
            "include_response_text": bool(args.include_response_text),
            "random_state": args.random_state,
            "max_tfidf_features": args.max_tfidf_features,
            "max_learner_features": args.max_learner_features,
            "representation_scope": "logged_prompt_and_state_features_not_internal_activations",
            "dictionary_solver": "MiniBatchDictionaryLearning(fit=lars,transform=omp,n_jobs=1)",
            "code_nonzero_budget": max(1, int(model_report["n_code"]) // 4),
            **split_metadata,
        }
    )
    with (output_dir / "xai_sparse_autoencoder_metadata.json").open("w", encoding="utf-8") as f:
        json.dump(model_report, f, indent=2)

    summary_path = write_summary(output_dir, model_report, global_importance, code_top_features, local)
    print("Sparse autoencoder audit artifacts:")
    print(f"- summary: {summary_path.resolve()}")
    print(f"- global_importance: {(output_dir / 'xai_sparse_autoencoder_global_importance.csv').resolve()}")
    print(f"- code_features: {(output_dir / 'xai_sparse_autoencoder_code_features.csv').resolve()}")
    print(f"- local: {(output_dir / 'xai_sparse_autoencoder_local_explanations.csv').resolve()}")


if __name__ == "__main__":
    main()
