#!/usr/bin/env python3
"""Answer the reviewer question the archetype analysis currently leaves open:
why k=4, and does the substantive claim survive a different k?

The manuscript fixes k=4 with no selection criterion and no stability evidence.
Two things are therefore needed. First, a k-sweep with internal validity indices
(silhouette, Calinski-Harabasz, Davies-Bouldin, inertia) so the choice of k is
answerable from a table rather than from assertion. Second, a bootstrap
stability curve, because internal indices reward geometric compactness and say
nothing about whether the same partition would reappear in a fresh sample of
participants -- and a partition that does not reappear cannot support a claim
about human behavioural *types*. Agreement is measured with the adjusted Rand
index on the participants present in both the resample and the full sample, so
the score is comparable across k (ARI corrects for the chance agreement that
grows with the number of clusters).

The design constraint that matters most: the feature matrix must be the exact
341 x 5 standardized matrix the paper clusters, otherwise the sweep answers a
different question. It is therefore imported from the generator itself
(``results/cross_model_pilot_synthesis/analyze_behavioral_clustering.py``)
rather than replicated -- ``human_features``, ``llm_player_features``,
``FEATURES`` and ``SEED`` come from that module, and its ``main`` is guarded, so
importing it is side-effect free apart from matplotlib rcParams.

The projection half of the sweep exists because the paper's Panel B claim is
comparative, not about k: human trajectories spread across several archetypes
while most single checkpoints concentrate in one. That claim is only k-dependent
if the number of archetypes each population occupies changes with k, so it is
recomputed at every k with the paper's own nearest-centroid projection (LLM
player-races standardized with the *human* mean/SD, assigned to the nearest
human centroid).
"""

from __future__ import annotations

import hashlib
import json
import os
import sys
from pathlib import Path

# 2,500 k-means fits on a 341-row matrix are dominated by BLAS/OpenMP thread
# launch overhead, not arithmetic; single-threaded is ~15x faster here.
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import pandas as pd
import sklearn
from sklearn.cluster import KMeans
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)

ROOT = Path(__file__).resolve().parents[1]
GENERATOR_DIR = ROOT / "results" / "cross_model_pilot_synthesis"
DATA = GENERATOR_DIR / "data"

sys.path.insert(0, str(GENERATOR_DIR))
import analyze_behavioral_clustering as abc  # noqa: E402

K_RANGE = range(2, 7)
N_BOOTSTRAP = 500
SEED = abc.SEED
OCCUPANCY_THRESHOLD = 0.05
FEATURES = abc.FEATURES

ESTIMAND = (
    "Sensitivity of the four human behavioural archetypes to the number of k-means "
    "clusters: internal validity indices and bootstrap partition stability for k=2..6 "
    "on the same 341 x 5 standardized human descriptor matrix, plus the "
    "nearest-human-archetype projection share of every population at each k."
)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def source_hashes() -> dict[str, str]:
    hashes = {abc.HUMAN_CSV.relative_to(ROOT).as_posix(): sha256(abc.HUMAN_CSV)}
    for model, dirs in abc.NEUTRAL_INPUTS.items():
        subdir = abc.MODEL_SUBDIR.get(model, "")
        for d in dirs:
            p = ROOT / d / subdir / "turns.jsonl"
            hashes[p.relative_to(ROOT).as_posix()] = sha256(p)
    return hashes


def human_matrix() -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    human = abc.human_features()
    for feat in FEATURES:
        human[feat] = human[feat].fillna(human[feat].mean())
    mean = human[FEATURES].mean()
    std = human[FEATURES].std()
    return (human[FEATURES] - mean) / std, mean, std


def bootstrap_ari(human_z: pd.DataFrame, reference: np.ndarray, k: int) -> np.ndarray:
    x = human_z.to_numpy()
    n = len(x)
    rng = np.random.default_rng(SEED + k)
    scores = []
    for _ in range(N_BOOTSTRAP):
        idx = rng.integers(0, n, size=n)
        labels = KMeans(n_clusters=k, random_state=SEED, n_init=10).fit_predict(x[idx])
        first = {}
        for pos, participant in enumerate(idx):
            first.setdefault(int(participant), pos)
        present = np.array(sorted(first))
        boot = np.array([labels[first[p]] for p in present])
        scores.append(adjusted_rand_score(reference[present], boot))
    return np.asarray(scores)


def main() -> None:
    DATA.mkdir(parents=True, exist_ok=True)
    human_z, mean, std = human_matrix()
    assert human_z.shape == (341, 5), f"expected the paper's 341 x 5 matrix, got {human_z.shape}"

    llm_z = {}
    for model in abc.NEUTRAL_INPUTS:
        llm = abc.llm_player_features(model)
        for feat in FEATURES:
            llm[feat] = llm[feat].fillna(mean[feat])
        llm_z[model] = (llm[FEATURES] - mean) / std

    sweep_rows = []
    projection_rows = []
    for k in K_RANGE:
        km = KMeans(n_clusters=k, random_state=SEED, n_init=10)
        reference = km.fit_predict(human_z)
        sizes = np.bincount(reference, minlength=k)
        ari = bootstrap_ari(human_z, reference, k)
        sweep_rows.append({
            "k": k,
            "silhouette": float(silhouette_score(human_z, reference)),
            "calinski_harabasz": float(calinski_harabasz_score(human_z, reference)),
            "davies_bouldin": float(davies_bouldin_score(human_z, reference)),
            "inertia": float(km.inertia_),
            "cluster_sizes": "|".join(str(int(s)) for s in sizes),
            "smallest_cluster_n": int(sizes.min()),
            "ari_mean": float(ari.mean()),
            "ari_p2_5": float(np.percentile(ari, 2.5)),
            "ari_p97_5": float(np.percentile(ari, 97.5)),
            "n_bootstrap": N_BOOTSTRAP,
            "seed": SEED,
        })

        populations = {"human": reference}
        for model, z in llm_z.items():
            populations[model] = km.predict(z)
        for population, labels in populations.items():
            shares = [float((labels == c).mean()) for c in range(k)]
            row = {"k": k, "population": population, "n": int(len(labels))}
            for c in range(k):
                row[f"share_c{c}"] = shares[c]
            row["n_archetypes_occupied"] = int(sum(s >= OCCUPANCY_THRESHOLD for s in shares))
            row["n_archetypes_share_ge_0p01"] = int(sum(s >= 0.01 for s in shares))
            row["n_archetypes_nonzero"] = int(sum(s > 0.0 for s in shares))
            row["max_share"] = max(shares)
            projection_rows.append(row)

    sweep = pd.DataFrame(sweep_rows)
    sweep.to_csv(DATA / "human_archetype_k_sensitivity.csv", index=False)
    projection = pd.DataFrame(projection_rows)
    share_cols = [c for c in projection.columns if c.startswith("share_c")]
    projection = projection[
        ["k", "population", "n"] + share_cols
        + ["max_share", "n_archetypes_occupied", "n_archetypes_share_ge_0p01", "n_archetypes_nonzero"]
    ]
    projection.to_csv(DATA / "human_archetype_k_projection.csv", index=False)

    provenance = {
        "schema_version": "ai-race-human-archetype-k-sensitivity-v1",
        "evidence_class": "diagnostic",
        "estimand": ESTIMAND,
        "seed": SEED,
        "n_bootstrap": N_BOOTSTRAP,
        "k_range": list(K_RANGE),
        "n_participants": int(human_z.shape[0]),
        "features": FEATURES,
        "feature_matrix_source": "imported verbatim from results/cross_model_pilot_synthesis/analyze_behavioral_clustering.py (human_features, llm_player_features)",
        "standardization": "human-only mean/SD (pandas ddof=1), LLM players projected into the same human z-space",
        "bootstrap_scheme": "participant-level resample with replacement (n=341), k-means refit per draw in the fixed human z-space, adjusted Rand index against the full-sample reference labelling on the unique participants drawn (first occurrence per participant)",
        "bootstrap_rng": "numpy default_rng(seed + k)",
        "occupancy_threshold": OCCUPANCY_THRESHOLD,
        "occupancy_threshold_note": "an archetype counts as occupied by a population when at least 5 percent of that population's entities project to it; the 1 percent and strictly-positive counts are reported alongside so the claim can be read at any threshold",
        "packages": {
            "scikit-learn": sklearn.__version__,
            "numpy": np.__version__,
            "pandas": pd.__version__,
        },
        "source_hashes": source_hashes(),
        "outputs": {
            "sweep": "human_archetype_k_sensitivity.csv",
            "projection": "human_archetype_k_projection.csv",
        },
    }
    with open(DATA / "human_archetype_k_sensitivity.json", "w") as f:
        json.dump(provenance, f, indent=2)

    print(sweep.to_string(index=False))
    print()
    print(projection.pivot(index="population", columns="k", values="n_archetypes_occupied").to_string())


if __name__ == "__main__":
    main()
