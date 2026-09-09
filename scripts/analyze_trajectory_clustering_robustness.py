#!/usr/bin/env python3
"""Stress-test trajectory clustering and embedding choices.

The manuscript's trajectory figures are descriptive, so this script does not
turn a particular HDBSCAN partition or t-SNE map into a causal result. It
reconstructs the same first-five-round feature matrix from the checked-in raw
run tables, sweeps the clustering and embedding controls named by the review,
and records cluster counts, noise fractions, adjusted Rand agreement, t-SNE
trustworthiness, and neighbourhood overlap. Raw inputs and every setting are
recorded beside the outputs so the analysis can be rerun without guessing the
source data.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import json
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE, trustworthiness
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

try:
    from scripts.build_manuscript_clustering_figures import (
        BASELINE_INPUTS,
        flatten_features,
        load_trajectories,
    )
except ModuleNotFoundError:
    from build_manuscript_clustering_figures import (
        BASELINE_INPUTS,
        flatten_features,
        load_trajectories,
    )


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
CLUSTER_SIZES = (10, 15, 20, 25)
MIN_SAMPLES = (3, 6, 9)
PERPLEXITIES = (5, 15, 30, 50)
TSNE_SEEDS = (20260908, 20260909, 20260910)
TSNE_BASE_SEED = TSNE_SEEDS[0]
K_NEIGHBOURS = 15


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def cluster_summary(labels: np.ndarray, populations: pd.Series) -> dict[str, object]:
    labels = np.asarray(labels, dtype=int)
    non_noise = labels >= 0
    counts = pd.Series(labels[non_noise]).value_counts()
    by_population: dict[str, dict[str, float]] = {}
    for population in sorted(populations.unique()):
        mask = populations.to_numpy() == population
        pop_labels = labels[mask]
        by_population[population] = {
            "n": int(mask.sum()),
            "noise_fraction": float(np.mean(pop_labels < 0)),
            "clustered_fraction": float(np.mean(pop_labels >= 0)),
        }
    return {
        "n_clusters": int(counts.size),
        "noise_fraction": float(np.mean(~non_noise)),
        "largest_cluster_fraction": float(counts.iloc[0] / len(labels)) if len(counts) else 0.0,
        "by_population": by_population,
    }


def neighbourhood_sets(embedding: np.ndarray) -> list[set[int]]:
    neighbours = NearestNeighbors(n_neighbors=K_NEIGHBOURS + 1).fit(embedding)
    indices = neighbours.kneighbors(return_distance=False)[:, 1:]
    return [set(row.tolist()) for row in indices]


def mean_jaccard(left: list[set[int]], right: list[set[int]]) -> float:
    values = []
    for a, b in zip(left, right):
        union = a | b
        values.append(len(a & b) / len(union) if union else 1.0)
    return float(np.mean(values))


def main() -> None:
    frame = load_trajectories()
    features = StandardScaler().fit_transform(flatten_features(frame))
    populations = frame["population"].reset_index(drop=True)

    cluster_rows: list[dict[str, object]] = []
    cluster_labels: dict[str, np.ndarray] = {}
    for min_cluster_size in CLUSTER_SIZES:
        for min_samples in MIN_SAMPLES:
            key = f"mcs{min_cluster_size}_ms{min_samples}"
            labels = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric="euclidean",
                cluster_selection_method="eom",
                prediction_data=False,
            ).fit_predict(features)
            cluster_labels[key] = labels
            summary = cluster_summary(labels, populations)
            cluster_rows.append(
                {
                    "analysis": "hdbscan",
                    "setting": key,
                    "min_cluster_size": min_cluster_size,
                    "min_samples": min_samples,
                    **{k: v for k, v in summary.items() if k != "by_population"},
                    "population_summary_json": json.dumps(summary["by_population"], sort_keys=True),
                }
            )

    baseline_labels = cluster_labels["mcs20_ms6"]
    for row in cluster_rows:
        labels = cluster_labels[str(row["setting"])]
        row["ari_vs_baseline"] = float(adjusted_rand_score(baseline_labels, labels))

    embeddings: dict[str, np.ndarray] = {}
    tsne_rows: list[dict[str, object]] = []
    baseline_neighbours: list[set[int]] | None = None
    for perplexity in PERPLEXITIES:
        for seed in TSNE_SEEDS:
            key = f"p{perplexity}_seed{seed}"
            embedding = TSNE(
                n_components=2,
                perplexity=perplexity,
                random_state=seed,
                # Random initialization makes the seed sweep a real
                # stability test. The manuscript's descriptive figure keeps
                # its separate PCA initialization and is not overwritten.
                init="random",
                learning_rate="auto",
                max_iter=1500,
            ).fit_transform(features)
            embeddings[key] = embedding
            neighbours = neighbourhood_sets(embedding)
            if perplexity == 30 and seed == TSNE_BASE_SEED:
                baseline_neighbours = neighbours
            tsne_rows.append(
                {
                    "analysis": "tsne",
                    "setting": key,
                    "perplexity": perplexity,
                    "seed": seed,
                    "trustworthiness": float(trustworthiness(features, embedding, n_neighbors=K_NEIGHBOURS)),
                    "neighbourhood_jaccard_vs_baseline": (
                        float(mean_jaccard(neighbours, baseline_neighbours))
                        if baseline_neighbours is not None
                        else None
                    ),
                }
            )

    # Fill the baseline-relative overlap for runs encountered before the
    # baseline in the deterministic grid.
    base = embeddings[f"p30_seed{TSNE_BASE_SEED}"]
    base_neighbours = neighbourhood_sets(base)
    for row in tsne_rows:
        embedding = embeddings[str(row["setting"])]
        row["neighbourhood_jaccard_vs_baseline"] = mean_jaccard(
            neighbourhood_sets(embedding), base_neighbours
        )

    OUT.mkdir(parents=True, exist_ok=True)
    cluster_path = OUT / "trajectory_hdbscan_robustness.csv"
    tsne_path = OUT / "trajectory_tsne_robustness.csv"
    pd.DataFrame(cluster_rows).to_csv(cluster_path, index=False)
    pd.DataFrame(tsne_rows).to_csv(tsne_path, index=False)

    source_files = [ROOT / "references/source_study_dataset/airace_deidentified_long.csv"]
    for _, raw_path in BASELINE_INPUTS.items():
        path = Path(raw_path) if Path(raw_path).is_absolute() else ROOT / Path(raw_path)
        source_files.append(path / "all_results.csv" if path.is_dir() else path)
    provenance = {
        "schema_version": "ai-race-trajectory-robustness-v1",
        "evidence_class": "diagnostic",
        "source_description": "First-five-round trajectories reconstructed from raw baseline tables; no HDBSCAN or t-SNE output is used as a causal estimand.",
        "n_trajectories": int(len(frame)),
        "n_features": int(features.shape[1]),
        "hdbscan_grid": {"min_cluster_size": list(CLUSTER_SIZES), "min_samples": list(MIN_SAMPLES)},
        "tsne_grid": {"perplexities": list(PERPLEXITIES), "seeds": list(TSNE_SEEDS), "init": "random", "n_neighbors": K_NEIGHBOURS},
        "packages": {name: package_version(name) for name in ("hdbscan", "scikit-learn", "numpy", "pandas")},
        "source_hashes": {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files},
        "outputs": {"hdbscan": cluster_path.name, "tsne": tsne_path.name},
    }
    (OUT / "trajectory_clustering_robustness.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(json.dumps({"trajectories": len(frame), "hdbscan_rows": len(cluster_rows), "tsne_rows": len(tsne_rows)}, indent=2))


if __name__ == "__main__":
    main()
