#!/usr/bin/env python3
"""Extend every applicable INSIGHTS.md section to GPT-5.6 Luna/Terra.

Part J (build_gen56_persona_gradient.py) covered the persona gradient and
within-role risk sensitivity. This script covers the rest of what the data
supports: feature-importance/SHAP (Part D style), round-by-round trajectory
(Part E style), behavioral-archetype projection (Part F style), and the
payoff/welfare check (Part G style).

One structural caveat threads through all of it and is repeated in each
section rather than left implicit: Luna and Terra have no neutral/no-persona
baseline lane yet (only the full risk_matrix persona sweep plus Rminus/Rplus),
so every number here pools across persona conditions rather than holding
framing constant the way the original five checkpoints' Part D/E/F numbers
do. That makes these extensions descriptive companions to the originals, not
a like-for-like replication -- and is exactly why persona itself is not one
of the five SHAP features below (it varies here in a way it didn't for the
neutral-lane fits).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from scipy import stats
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUT / "figures"
DATA = OUT / "data"
BEDROCK_MANTLE_ROOT = ROOT / "results" / "frontier" / "bedrock_mantle"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "slate": "#64748B", "grid": "#DCE3ED",
}
MODEL_DIRS = {"gpt-5.6-luna": ("luna", "openai.gpt-5.6-luna"), "gpt-5.6-terra": ("terra", "openai.gpt-5.6-terra")}
MODEL_LABELS = {"gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra"}
MODEL_COLORS = {"gpt-5.6-luna": PALETTE["amber"], "gpt-5.6-terra": PALETTE["red"]}
CORE_FEATURES = ["own_prev_unsafe", "opponent_prev_unsafe", "progress_gap", "max_private_risk", "round_number"]


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.labelsize": 10, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def all_turns_dirs(model_key: str) -> list[Path]:
    model_dir, model_name = MODEL_DIRS[model_key]
    base = BEDROCK_MANTLE_ROOT / model_dir / "persona"
    dirs = []
    for cell in sorted((base / "risk_matrix").glob("*")):
        p = cell / model_name
        if (p / "turns.jsonl").exists():
            dirs.append(p)
    for single in ["Rminus_risk_averse", "Rplus_risk_seeking"]:
        p = base / single / model_name
        if (p / "turns.jsonl").exists():
            dirs.append(p)
    return dirs


# --- Part D extension: feature importance / SHAP -----------------------------------


def llm_core_frame(model_key: str) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for d in all_turns_dirs(model_key):
        with open(d / "turns.jsonl") as f:
            for line in f:
                rec = json.loads(line)
                if rec["round"] <= 1 or rec["own_prev_action"] is None:
                    continue
                rows.append({
                    "own_prev_unsafe": 1.0 if rec["own_prev_action"] == "unsafe" else 0.0,
                    "opponent_prev_unsafe": 1.0 if rec["opponent_prev_action"] == "unsafe" else 0.0,
                    "progress_gap": float(rec["progress_gap_before"]),
                    "max_private_risk": float(rec["max_private_risk"]),
                    "round_number": float(rec["round"]),
                    "unsafe": int(rec["unsafe"]),
                })
    frame = pd.DataFrame(rows)
    return frame[CORE_FEATURES], frame["unsafe"]


def fit_and_explain(X: pd.DataFrame, y: pd.Series, feature_names: list[str], seed: int = 0) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y if y.nunique() > 1 else None)
    clf = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=10, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    majority_baseline = max(float(y_test.mean()), 1 - float(y_test.mean()))
    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)
    auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None
    bal_acc = float(balanced_accuracy_score(y_test, pred)) if y_test.nunique() > 1 else None
    impurity = dict(zip(feature_names, clf.feature_importances_.tolist()))
    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_test)
    sv_arr = np.asarray(sv)
    sv_pos = sv_arr[:, :, 1] if sv_arr.ndim == 3 else (sv[1] if isinstance(sv, list) else sv_arr)
    mean_abs_shap = dict(zip(feature_names, np.abs(sv_pos).mean(axis=0).tolist()))
    return {
        "n": int(len(X)), "test_accuracy": float(clf.score(X_test, y_test)), "majority_baseline_accuracy": majority_baseline,
        "roc_auc": auc, "balanced_accuracy": bal_acc, "base_rate": float(y.mean()),
        "impurity_importance": impurity, "mean_abs_shap": mean_abs_shap,
    }


def part_d_extension() -> dict:
    results = {}
    for model_key in MODEL_DIRS:
        X, y = llm_core_frame(model_key)
        r = fit_and_explain(X, y, CORE_FEATURES)
        results[model_key] = r
        print(f"[D] {model_key}: n={r['n']} base_rate={r['base_rate']:.3f} AUC={r['roc_auc']:.3f} bal_acc={r['balanced_accuracy']:.3f}")
    with open(DATA / "gen56_feature_importance.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def fig_d_extension(gen56_results: dict) -> None:
    original = json.loads((DATA / "feature_importance_results.json").read_text())
    setup_plot()
    keys = ["human", "gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview",
            "google/gemini-3.1-flash-lite-preview", "google/gemini-3.5-flash-lite", "gpt-5.6-luna", "gpt-5.6-terra"]
    labels = {
        "human": "Human", "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
        "google/gemini-3-flash-preview": "Gemini 3 Flash", "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
        "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite", "gpt-5.6-luna": "GPT-5.6 Luna (persona-pooled)",
        "gpt-5.6-terra": "GPT-5.6 Terra (persona-pooled)",
    }
    feature_labels = {
        "own_prev_unsafe": "Own previous\naction Unsafe", "opponent_prev_unsafe": "Opponent's previous\naction Unsafe",
        "progress_gap": "Progress gap\n(own - opponent)", "max_private_risk": "Risk treatment\n(max_private_risk)",
        "round_number": "Round number",
    }
    grid = []
    for k in keys:
        imp = gen56_results[k]["mean_abs_shap"] if k in gen56_results else original[k]["mean_abs_shap"]
        total = sum(imp.values()) or 1.0
        grid.append([imp[f] / total for f in CORE_FEATURES])
    grid = np.array(grid)
    fig, ax = plt.subplots(figsize=(9.8, 0.8 * len(keys) + 2))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max(), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                    color="white" if val > grid.max() * 0.55 else PALETTE["navy"], fontsize=9, weight="bold")
    ax.set_xticks(range(len(CORE_FEATURES)), [feature_labels[f] for f in CORE_FEATURES], fontsize=8.5)
    ax.set_yticks(range(len(keys)), [labels[k] for k in keys], fontsize=9)
    ax.set_title("Feature-importance heatmap, extended with GPT-5.6\n(persona-pooled, not a neutral-lane fit -- see caveat)", pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(CORE_FEATURES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(keys))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(0.01, -0.02,
              "Human + first 5 rows: neutral-lane fits (see feature_importance_shap_heatmap.png). GPT-5.6 rows: fit on the full\n"
              "persona-risk-matrix sweep pooled (round >= 2), since no neutral lane exists yet for these two checkpoints.",
              fontsize=8, color=PALETTE["slate"])
    save_figure(fig, FIGURES / "feature_importance_shap_heatmap_gen56")
    print("wrote", FIGURES / "feature_importance_shap_heatmap_gen56.png")


# --- Part E extension: round-by-round trajectory ------------------------------------


def part_e_extension() -> dict:
    results = {}
    for model_key in MODEL_DIRS:
        by_round: dict[int, list[int]] = {}
        for d in all_turns_dirs(model_key):
            with open(d / "turns.jsonl") as f:
                for line in f:
                    rec = json.loads(line)
                    by_round.setdefault(rec["round"], []).append(rec["unsafe"])
        rows = []
        for r in sorted(by_round):
            vals = by_round[r]
            if len(vals) < 10:
                continue
            rows.append({"round": r, "mean_unsafe": float(np.mean(vals)), "n": len(vals)})
        results[model_key] = rows
        print(f"[E] {model_key}: rounds 1-{rows[-1]['round']} (n>=10 cutoff), round1 n={rows[0]['n']}, round1 rate={rows[0]['mean_unsafe']:.3f}")
    with open(DATA / "gen56_round_trajectory.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def fig_e_extension(trajectories: dict) -> None:
    setup_plot()
    fig, ax = plt.subplots(figsize=(9.8, 5.6))
    for model_key, rows in trajectories.items():
        xs = [r["round"] for r in rows]
        ys = [100 * r["mean_unsafe"] for r in rows]
        ax.plot(xs, ys, marker="o", markersize=4, linewidth=2, color=MODEL_COLORS[model_key], label=MODEL_LABELS[model_key] + " (persona-pooled)")
    ax.set_title("GPT-5.6 round-by-round trajectory (persona-pooled)\ncompare shape only, not level, against the neutral-lane checkpoints", pad=14)
    ax.set_xlabel("Round number")
    ax.set_ylabel("Mean Unsafe rate (%)")
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="best")
    fig.text(0.01, -0.08,
              "Pooled across the full persona-risk-matrix sweep (no neutral lane exists yet for these checkpoints), so the\n"
              "*level* here reflects the persona mix, not a comparable baseline; only the round-to-round *shape* is informative.",
              fontsize=8, color=PALETTE["slate"])
    save_figure(fig, FIGURES / "round_trajectory_gen56")
    print("wrote", FIGURES / "round_trajectory_gen56.png")


# --- Part F extension: behavioral archetype projection ------------------------------


def _safe_diff(df: pd.DataFrame, col: str, cond_true, cond_false) -> float:
    a = df.loc[cond_true, col]
    b = df.loc[cond_false, col]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    return float(a.mean() - b.mean())


def llm_player_features(model_key: str) -> pd.DataFrame:
    per_player: dict[str, list[dict]] = {}
    for d in all_turns_dirs(model_key):
        with open(d / "turns.jsonl") as f:
            for line in f:
                r = json.loads(line)
                key = f"{r['game_id']}::{r['player']}"
                per_player.setdefault(key, []).append(r)
    rows = []
    for key, turns in per_player.items():
        turns = sorted(turns, key=lambda r: r["round"])
        g = pd.DataFrame(turns)
        later = g[g["round"] > 1].copy()
        later["own_prev_unsafe"] = (later["own_prev_action"] == "unsafe").astype(int)
        later["opponent_prev_unsafe"] = (later["opponent_prev_action"] == "unsafe").astype(int)
        row = {
            "player_key": key,
            "overall_unsafe_rate": g["unsafe"].mean(),
            "reciprocity": _safe_diff(later, "unsafe", later["opponent_prev_unsafe"] == 1, later["opponent_prev_unsafe"] == 0),
            "position_sensitivity": _safe_diff(later, "unsafe", later["progress_gap_before"] > 0, later["progress_gap_before"] < 0),
            "own_autocorrelation": _safe_diff(later, "unsafe", later["own_prev_unsafe"] == 1, later["own_prev_unsafe"] == 0),
            "first_round_unsafe": float(g.loc[g["round"] == 1, "unsafe"].iloc[0]),
        }
        rows.append(row)
    return pd.DataFrame(rows)


def part_f_extension() -> dict:
    human_summary = pd.read_csv(DATA / "human_cluster_summary.csv", index_col=0)
    features = ["overall_unsafe_rate", "reciprocity", "position_sensitivity", "own_autocorrelation", "first_round_unsafe"]
    human_full = human_summary.copy()

    # Refit KMeans on the human data exactly as analyze_behavioral_clustering.py did,
    # so predict() below uses centroids consistent with human_cluster_summary.csv
    # (mean/std for standardization are recomputed the same way).
    human_csv = ROOT / "public_dataset" / "airace_deidentified_long.csv"
    df = pd.read_csv(human_csv)
    rows = []
    for pid, g in df.groupby("participant_id"):
        later = g[g["round_number"] > 1]
        rows.append({
            "overall_unsafe_rate": g["decision"].mean(),
            "reciprocity": _safe_diff(later, "decision", later["decision_opponent_lag"] == 1, later["decision_opponent_lag"] == 0),
            "position_sensitivity": _safe_diff(later, "decision", later["delta_steps_lag"] > 0, later["delta_steps_lag"] < 0),
            "own_autocorrelation": _safe_diff(later, "decision", later["decision_lag"] == 1, later["decision_lag"] == 0),
            "first_round_unsafe": float(g.loc[g["round_number"] == 1, "decision"].iloc[0]),
        })
    human = pd.DataFrame(rows)
    for feat in features:
        human[feat] = human[feat].fillna(human[feat].mean())
    mean = human[features].mean()
    std = human[features].std()
    human_z = (human[features] - mean) / std
    km = KMeans(n_clusters=4, random_state=0, n_init=10)
    km.fit(human_z)

    results = {}
    for model_key in MODEL_DIRS:
        llm = llm_player_features(model_key)
        for feat in features:
            llm[feat] = llm[feat].fillna(mean[feat])
        llm_z = (llm[features] - mean) / std
        labels = km.predict(llm_z)
        shares = {int(c): float((labels == c).mean()) for c in range(4)}
        results[model_key] = {"n": len(labels), "cluster_shares": shares}
        print(f"[F] {model_key}: n={len(labels)} cluster shares={shares}")
    with open(DATA / "gen56_cluster_projection.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def fig_f_extension(cluster_results: dict) -> None:
    original = pd.read_csv(DATA / "llm_human_cluster_projection.csv")
    setup_plot()
    models = ["gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview",
              "google/gemini-3.1-flash-lite-preview", "google/gemini-3.5-flash-lite", "gpt-5.6-luna", "gpt-5.6-terra"]
    labels = {
        "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano", "google/gemini-3-flash-preview": "Gemini 3 Flash",
        "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite", "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
        "gpt-5.6-luna": "GPT-5.6 Luna\n(persona-pooled)", "gpt-5.6-terra": "GPT-5.6 Terra\n(persona-pooled)",
    }
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    colors = [PALETTE["blue"], PALETTE["teal"], PALETTE["amber"], PALETTE["red"]]
    bottom = np.zeros(len(models))
    for c in range(4):
        vals = []
        for m in models:
            if m in cluster_results:
                vals.append(cluster_results[m]["cluster_shares"].get(str(c), cluster_results[m]["cluster_shares"].get(c, 0)) * 100)
            else:
                row = original[(original["model"] == m) & (original["cluster"] == c)]
                vals.append(float(row["share"].iloc[0]) * 100 if len(row) else 0)
        ax.bar([labels[m] for m in models], vals, bottom=bottom, color=colors[c], label=f"Cluster {c}", width=0.6)
        bottom += np.array(vals)
    ax.set_ylabel("Share of player-races nearest to each human cluster (%)")
    ax.set_title("Human-archetype projection, extended with GPT-5.6\n(persona-pooled -- includes players deliberately framed at every extreme)", pad=14)
    ax.set_ylim(0, 100)
    ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right", fontsize=8.5)
    fig.text(0.01, -0.42,
              "GPT-5.6 columns pool every persona cell (R1-R6 matrix + Rminus/Rplus), so they necessarily touch more of the human\n"
              "archetype space than a single neutral condition would -- not a like-for-like comparison with the other five columns.",
              fontsize=8, color=PALETTE["slate"])
    save_figure(fig, FIGURES / "llm_human_cluster_projection_gen56")
    print("wrote", FIGURES / "llm_human_cluster_projection_gen56.png")


# --- Part G extension: payoff / welfare ---------------------------------------------


def part_g_extension() -> dict:
    results = {}
    for model_key in MODEL_DIRS:
        rows = []
        for d in all_turns_dirs(model_key):
            p = d / "players.csv"
            with open(p) as f:
                for row in csv.DictReader(f):
                    rows.append({
                        "unsafe_rate": float(row["unsafe_frequency"]),
                        "true_payoff": float(row["final_payoff"]),
                        "setback": int(row["setback"]),
                    })
        df = pd.DataFrame(rows)
        r, p_val = stats.pearsonr(df["unsafe_rate"], df["true_payoff"])
        setback_rate = float(df["setback"].mean())
        results[model_key] = {"n": len(df), "pearson_r": float(r), "pearson_p": float(p_val), "setback_rate": setback_rate}
        print(f"[G] {model_key}: n={len(df)} r={r:.3f} p={p_val:.4f} setback_rate={setback_rate:.3f}")
    with open(DATA / "gen56_payoff_welfare.json", "w") as f:
        json.dump(results, f, indent=2)
    return results


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    d_results = part_d_extension()
    fig_d_extension(d_results)

    e_results = part_e_extension()
    fig_e_extension(e_results)

    f_results = part_f_extension()
    fig_f_extension(f_results)

    g_results = part_g_extension()

    print("\n=== summary ===")
    print(json.dumps({"D": {k: {"n": v["n"], "AUC": v["roc_auc"], "bal_acc": v["balanced_accuracy"]} for k, v in d_results.items()},
                       "G": g_results}, indent=2))


if __name__ == "__main__":
    main()
