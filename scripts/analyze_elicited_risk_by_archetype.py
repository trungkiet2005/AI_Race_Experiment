#!/usr/bin/env python3
"""Test whether the pre-game elicited risk preference differs across the four
human behavioural archetypes.

The supplement asserts a Kruskal--Wallis result for this comparison. No code or
artefact in the repository ever computed it, so this script computes it from the
public human table and writes the answer, the exclusion accounting and the
provenance to results/cross_model_pilot_synthesis/data/elicited_risk_by_archetype.json.

The clustering is not re-specified here. analyze_behavioral_clustering.py is
loaded by path and its human_features(), FEATURES, K and SEED are reused
verbatim, so the archetypes tested are the published archetypes; the script
refuses to continue if the label sizes are not the published multiset
{19, 170, 133, 19}.

The elicited measure is the incentivised Eckel--Grossman gamble choice, carried
on the long table as risk_gamble_choice (0 = most risk-averse, 5 = least
risk-averse); it is constant within a participant, so it is taken as the
participant's first value. Its identity is checked by reproducing the
participant-level correlation the supplement reports for the same variable
against overall Unsafe rate (r = -0.015, p = 0.79, n = 341).
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
from scipy import stats
from sklearn.cluster import KMeans

ROOT = Path(__file__).resolve().parents[1]
CLUSTERING_SRC = ROOT / "results" / "cross_model_pilot_synthesis" / "analyze_behavioral_clustering.py"
OUT = ROOT / "results" / "cross_model_pilot_synthesis" / "data" / "elicited_risk_by_archetype.json"

RISK_COL = "risk_gamble_choice"
PUBLISHED_SIZES = sorted([19, 170, 133, 19])
ALPHA = 0.05


def load_clustering_module():
    spec = importlib.util.spec_from_file_location("_behavioral_clustering_src", CLUSTERING_SRC)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def holm(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    n = len(pvals)
    adjusted = np.empty(n)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * pvals[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted.tolist()


def dunn(values: np.ndarray, labels: np.ndarray, keys: list) -> list[dict]:
    """Dunn's post-hoc test on the pooled midranks, with the standard tie
    correction, Holm-adjusted across the six pairs."""
    n = len(values)
    ranks = stats.rankdata(values)
    _, counts = np.unique(values, return_counts=True)
    tie_term = (counts ** 3 - counts).sum()
    sigma_sq = (n * (n + 1) / 12.0) - tie_term / (12.0 * (n - 1))

    mean_rank = {k: ranks[labels == k].mean() for k in keys}
    size = {k: int((labels == k).sum()) for k in keys}

    rows, raw = [], []
    for a, b in combinations(keys, 2):
        se = np.sqrt(sigma_sq * (1.0 / size[a] + 1.0 / size[b]))
        z = (mean_rank[a] - mean_rank[b]) / se
        p = 2.0 * stats.norm.sf(abs(z))
        rows.append({"group_a": a, "group_b": b, "mean_rank_a": float(mean_rank[a]),
                     "mean_rank_b": float(mean_rank[b]), "z": float(z), "p_raw": float(p)})
        raw.append(float(p))
    for row, adj in zip(rows, holm(raw)):
        row["p_holm"] = float(adj)
        row["significant_holm_05"] = bool(adj < ALPHA)
    return rows


def mann_whitney(values: np.ndarray, labels: np.ndarray, keys: list) -> list[dict]:
    rows, raw = [], []
    for a, b in combinations(keys, 2):
        xa, xb = values[labels == a], values[labels == b]
        u, p = stats.mannwhitneyu(xa, xb, alternative="two-sided")
        # rank-biserial correlation as the paired-with-U effect size
        rb = 2.0 * u / (len(xa) * len(xb)) - 1.0
        rows.append({"group_a": a, "group_b": b, "n_a": int(len(xa)), "n_b": int(len(xb)),
                     "u": float(u), "p_raw": float(p), "rank_biserial": float(rb)})
        raw.append(float(p))
    for row, adj in zip(rows, holm(raw)):
        row["p_holm"] = float(adj)
        row["significant_holm_05"] = bool(adj < ALPHA)
    return rows


def main() -> int:
    bc = load_clustering_module()

    long_df = pd.read_csv(bc.HUMAN_CSV)
    n_participants_in_table = int(long_df["participant_id"].nunique())

    risk = long_df.groupby("participant_id")[RISK_COL].first()
    within_participant_variation = int((long_df.groupby("participant_id")[RISK_COL].nunique() > 1).sum())
    if within_participant_variation:
        print(f"FAIL: {RISK_COL} varies within {within_participant_variation} participants", file=sys.stderr)
        return 1

    features = bc.human_features().set_index("participant_id")
    missing_before_imputation = {f: int(features[f].isna().sum()) for f in bc.FEATURES}
    rows_with_any_missing_descriptor = int(features[bc.FEATURES].isna().any(axis=1).sum())

    imputed = features.copy()
    for feat in bc.FEATURES:
        imputed[feat] = imputed[feat].fillna(imputed[feat].mean())
    mean, std = imputed[bc.FEATURES].mean(), imputed[bc.FEATURES].std()
    z = (imputed[bc.FEATURES] - mean) / std

    km = KMeans(n_clusters=bc.K, random_state=bc.SEED, n_init=10)
    imputed["cluster"] = km.fit_predict(z)
    imputed["elicited_risk"] = risk

    sizes = Counter(int(c) for c in imputed["cluster"])
    if sorted(sizes.values()) != PUBLISHED_SIZES:
        print(f"FAIL: archetype sizes {sorted(sizes.values())} != published {PUBLISHED_SIZES}; "
              "these are not the paper's clusters, refusing to test", file=sys.stderr)
        return 1
    print(f"archetype size check PASS: {dict(sorted(sizes.items()))}")

    # Identity check on the elicited variable: reproduce the supplement's
    # participant-level correlation of elicited risk against Unsafe rate.
    fit = stats.linregress(imputed["elicited_risk"].to_numpy(float),
                           imputed["overall_unsafe_rate"].to_numpy(float))
    # The supplement's quoted slope is the trend across the six bin means (the
    # dotted line in the figure), not the participant-level slope; both are
    # recorded so the discrepancy is visible rather than silently reconciled.
    bin_means = imputed.groupby("elicited_risk")["overall_unsafe_rate"].mean()
    bin_fit = stats.linregress(bin_means.index.to_numpy(float), bin_means.to_numpy(float))
    identity_check = {
        "reported_in_supplement": {"slope": -0.004, "r": -0.015, "p": 0.79, "n": 341},
        "participant_level": {"slope": float(fit.slope), "r": float(fit.rvalue),
                              "p": float(fit.pvalue), "n": int(len(imputed))},
        "six_bin_mean_trend": {"slope": float(bin_fit.slope), "r": float(bin_fit.rvalue),
                               "p": float(bin_fit.pvalue), "n": int(len(bin_means))},
        "r_p_n_reproduce": bool(abs(fit.rvalue + 0.015) < 5e-4 and abs(fit.pvalue - 0.79) < 5e-3
                                and len(imputed) == 341),
        "note": ("r, p and n reproduce at the participant level; the quoted slope -0.004 "
                 "matches the six-bin-mean trend line, not the participant-level slope."),
    }
    print("correlation identity check:", json.dumps(identity_check["participant_level"]))
    print("  six-bin-mean trend:", json.dumps(identity_check["six_bin_mean_trend"]))
    print("  r/p/n reproduce:", identity_check["r_p_n_reproduce"])

    keys = sorted(int(c) for c in imputed["cluster"].unique())
    values = imputed["elicited_risk"].to_numpy(float)
    labels = imputed["cluster"].to_numpy(int)

    groups = [values[labels == k] for k in keys]
    H, p = stats.kruskal(*groups)
    n_test = int(len(values))
    k = len(keys)
    epsilon_sq = float((H - k + 1) / (n_test - k))
    eta_sq = float((H - k + 1) / (n_test - k))  # identical algebra; reported once, named twice
    eta_sq_h = float(H / (n_test - 1))

    descriptives = []
    for key in keys:
        v = values[labels == key]
        q1, med, q3 = np.percentile(v, [25, 50, 75])
        descriptives.append({
            "cluster": key,
            "n": int(len(v)),
            "median": float(med),
            "q1": float(q1),
            "q3": float(q3),
            "iqr": float(q3 - q1),
            "mean": float(v.mean()),
            "sd": float(v.std(ddof=1)),
            "min": float(v.min()),
            "max": float(v.max()),
            "descriptor_centroid": {f: float(imputed.loc[imputed["cluster"] == key, f].mean())
                                    for f in bc.FEATURES},
        })

    print(f"Kruskal-Wallis H={H:.4f} df={k - 1} p={p:.6g} n={n_test} epsilon_sq={epsilon_sq:.4f}")
    for d in descriptives:
        print(f"  cluster {d['cluster']}: n={d['n']} median={d['median']:.1f} "
              f"IQR=[{d['q1']:.1f}, {d['q3']:.1f}] mean={d['mean']:.3f}")

    payload = {
        "estimand": ("Does the pre-game incentivised Eckel-Grossman elicited risk score "
                     "(0-5) differ in distribution across the four K-means behavioural "
                     "archetypes fitted to five standardised trajectory descriptors of the "
                     "human two-player race participants?"),
        "omnibus_test": {
            "test": "Kruskal-Wallis H (scipy.stats.kruskal, tie-corrected)",
            "variable": RISK_COL,
            "grouping": "cluster label from analyze_behavioral_clustering.py (K=4, SEED=0)",
            "H": float(H),
            "df": int(k - 1),
            "p": float(p),
            "n": n_test,
            "n_groups": k,
            "significant_at_05": bool(p < ALPHA),
            "conclusion": ("no detectable difference in elicited risk score across archetypes"
                           if p >= ALPHA else "elicited risk score differs across archetypes"),
        },
        "effect_size": {
            "epsilon_squared": epsilon_sq,
            "eta_squared_from_H": eta_sq,
            "eta_squared_H_over_n_minus_1": eta_sq_h,
            "formula_epsilon_squared": "(H - k + 1) / (n - k)",
            "formula_eta_squared_H_over_n_minus_1": "H / (n - 1)",
        },
        "group_descriptives": descriptives,
        "pairwise": {
            "dunn_holm": dunn(values, labels, keys),
            "mann_whitney_holm": mann_whitney(values, labels, keys),
            "correction": "Holm-Bonferroni across all 6 unordered archetype pairs, alpha=0.05",
        },
        "elicited_variable_identity_check": identity_check,
        "exclusion_accounting": {
            "participants_in_source_table": n_participants_in_table,
            "participants_with_descriptors": int(len(features)),
            "participants_with_cluster_label": int(len(imputed)),
            "participants_with_elicited_risk": int(risk.notna().sum()),
            "participants_missing_elicited_risk": int(risk.isna().sum()),
            "n_used_by_kruskal_wallis": n_test,
            "excluded": 0,
            "missing_descriptor_counts_before_imputation": missing_before_imputation,
            "participants_with_at_least_one_missing_descriptor": rows_with_any_missing_descriptor,
            "explanation": (
                "Nothing is excluded. Every one of the 341 human participants in "
                "airace_deidentified_long.csv contributes a cluster label and an elicited "
                "risk score, so the Kruskal-Wallis test runs on n=341. The elicitation is "
                "complete: risk_gamble_choice is non-missing for all 341 participants and is "
                "constant within a participant. Descriptors are incomplete for some "
                "participants (position_sensitivity is undefined for 268 participants who "
                "were never observed both ahead and behind, reciprocity and "
                "own_autocorrelation for 38), but analyze_behavioral_clustering.py mean-imputes "
                "those cells rather than dropping the participant, exactly as the published "
                "clustering does, so no participant is lost at the clustering step either. "
                "There is therefore no subset of this dataset of size 286, and the "
                "supplement's n=286 cannot be reconstructed from any documented step of this "
                "pipeline."
            ),
            "prior_claim_in_supplement": {"H": 21.95, "p": 0.001, "n": 286,
                                          "reproduced": False,
                                          "source_found_in_repository": False},
        },
        "provenance": {
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "script": "scripts/analyze_elicited_risk_by_archetype.py",
            "clustering_reused_from": "results/cross_model_pilot_synthesis/analyze_behavioral_clustering.py",
            "sources": [
                {"path": str(bc.HUMAN_CSV.relative_to(ROOT)).replace("\\", "/"),
                 "sha256": sha256(bc.HUMAN_CSV), "bytes": bc.HUMAN_CSV.stat().st_size},
                {"path": str(CLUSTERING_SRC.relative_to(ROOT)).replace("\\", "/"),
                 "sha256": sha256(CLUSTERING_SRC), "bytes": CLUSTERING_SRC.stat().st_size},
            ],
            "seed": int(bc.SEED),
            "k_clusters": int(bc.K),
            "features": list(bc.FEATURES),
            "kmeans_n_init": 10,
            "versions": {
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "pandas": pd.__version__,
                "scipy": scipy.__version__,
                "scikit-learn": sklearn.__version__,
            },
        },
    }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")
    print(f"wrote {OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
