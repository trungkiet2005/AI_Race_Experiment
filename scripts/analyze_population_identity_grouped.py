#!/usr/bin/env python3
"""Audit whether first-five-round trajectories identify their source population.

This is a descriptive diagnostic, not a claim that the trajectory embedding
recovers an intrinsic model identity.  The split keeps both players from an
LLM race together and uses class-balanced metrics because the pooled source
has 340 human trajectories and 60 trajectories for each of seven checkpoints.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.tree import DecisionTreeClassifier

from build_manuscript_clustering_figures import BASELINE_INPUTS, flatten_features, load_trajectories

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
SEED = 20260909
N_SPLITS = 5
N_PERMUTATIONS = 200


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def groups_for(frame: pd.DataFrame) -> np.ndarray:
    # Human player_id is the participant id.  LLM player_id is game_id::seat;
    # collapse the two seats so a race cannot straddle train and test.
    return np.asarray([
        str(pid).split("::", 1)[0] if pop != "human" else str(pid)
        for pid, pop in zip(frame["player_id"], frame["population"])
    ])


def fit_metrics(X: np.ndarray, y: np.ndarray, train: np.ndarray, test: np.ndarray, seed: int) -> dict:
    clf = DecisionTreeClassifier(
        max_depth=3, min_samples_leaf=8, class_weight="balanced", random_state=seed
    )
    clf.fit(X[train], y[train])
    pred = clf.predict(X[test])
    return {
        "accuracy": float(accuracy_score(y[test], pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
        "macro_f1": float(f1_score(y[test], pred, average="macro", zero_division=0)),
        "n_test": int(len(test)),
    }


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    frame = load_trajectories()
    X = flatten_features(frame)
    labels = sorted(frame["population"].unique())
    label_to_int = {label: i for i, label in enumerate(labels)}
    y = frame["population"].map(label_to_int).to_numpy()
    groups = groups_for(frame)

    splitter = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    splits = list(splitter.split(X, y, groups))
    rows = []
    for fold, (train, test) in enumerate(splits):
        metrics = fit_metrics(X, y, train, test, SEED + fold)
        metrics["fold"] = fold
        rows.append(metrics)
    observed = pd.DataFrame(rows)

    # A permutation null preserves the grouped folds and the observed class
    # counts.  It answers whether the tree exceeds chance under this exact
    # class-imbalanced pooled design without treating rows as independent.
    rng = np.random.default_rng(SEED)
    null = {"accuracy": [], "balanced_accuracy": [], "macro_f1": []}
    for rep in range(N_PERMUTATIONS):
        y_perm = rng.permutation(y)
        for fold, (train, test) in enumerate(splits):
            m = fit_metrics(X, y_perm, train, test, SEED + 10_000 + rep + fold)
            for key in null:
                null[key].append(m[key])

    summary = {}
    for key in ("accuracy", "balanced_accuracy", "macro_f1"):
        values = observed[key].to_numpy(float)
        null_values = np.asarray(null[key], dtype=float)
        summary[key] = {
            "mean": float(values.mean()),
            "std": float(values.std(ddof=1)),
            "null_mean": float(null_values.mean()),
            "null_std": float(null_values.std(ddof=1)),
            "null_95": [float(np.quantile(null_values, 0.025)), float(np.quantile(null_values, 0.975))],
            "permutation_p": float((1 + np.sum(null_values >= values.mean())) / (len(null_values) + 1)),
        }

    human_n = int((frame["population"] == "human").sum())
    model_n = {label: int((frame["population"] == label).sum()) for label in labels if label != "human"}
    source = ROOT / "references" / "source_study_dataset" / "airace_deidentified_long.csv"
    source_hashes = {"references/source_study_dataset/airace_deidentified_long.csv": sha256(source)}
    for model, relative_root in BASELINE_INPUTS.items():
        source_path = ROOT / relative_root / "turns.jsonl"
        source_hashes[f"{relative_root}/turns.jsonl"] = sha256(source_path)
    payload = {
        "schema_version": "ai-race-population-identity-grouped-v1",
        "evidence_class": "diagnostic",
        "estimand": "first-five-round population classification under race-grouped CV",
        "n_trajectories": int(len(frame)),
        "n_groups": int(len(np.unique(groups))),
        "class_counts": {label: int((frame["population"] == label).sum()) for label in labels},
        "human_trajectories": human_n,
        "llm_trajectories_per_checkpoint": sorted(set(model_n.values())),
        "feature_count": int(X.shape[1]),
        "tree": {"max_depth": 3, "min_samples_leaf": 8, "class_weight": "balanced"},
        "split": {"method": "StratifiedGroupKFold", "n_splits": N_SPLITS, "group_definition": "race for LLM; participant for human", "seed": SEED},
        "permutation_null": {"n_permutations": N_PERMUTATIONS, "seed": SEED},
        "metrics": summary,
        "fold_metrics": rows,
        "source_hashes": source_hashes,
        "interpretation": "Use balanced accuracy and macro-F1; raw accuracy is descriptive only because the pooled classes are imbalanced.",
    }
    (DATA / "population_identity_grouped.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    observed.to_csv(DATA / "population_identity_grouped_folds.csv", index=False)
    print(json.dumps(summary, indent=2))
    print("wrote", DATA / "population_identity_grouped.json")


if __name__ == "__main__":
    main()
