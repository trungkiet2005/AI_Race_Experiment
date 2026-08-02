#!/usr/bin/env python3
"""What predicts an Unsafe choice, for humans versus each LLM checkpoint?

Fits a Random Forest classifier separately on the human raw dataset and on each
of the five 2-player neutral-lane LLM pilots, using the same core mechanical
feature set on both sides (own previous action, opponent's previous action,
progress gap, risk treatment, round number). Reports impurity-based importance,
permutation importance, and mean |SHAP value| for each population, plus a
demographics-only supplementary fit on the human side (sex/age/nationality/
risk_gamble_choice) since those have no LLM counterpart.

This is a descriptive comparison of association structure (which inputs a
classifier leans on to predict the observed choice), not a causal or mechanistic
claim about either population's decision process.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[2]
HUMAN_CSV = ROOT / "public_dataset" / "airace_deidentified_long.csv"
OUTPUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUTPUT / "figures"
DATA = OUTPUT / "data"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "slate": "#64748B", "grid": "#DCE3ED",
}
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}
NEUTRAL_INPUTS = {
    "gpt-5-nano": ["results/frontier/openai/baseline/gpt-5-nano", "results/frontier/openai/persona/R0_neutral/gpt-5-nano"],
    "gpt-5.4-nano": ["results/frontier/openai/baseline/gpt-5.4-nano", "results/frontier/openai/persona/R0_neutral/gpt-5.4-nano"],
    "google/gemini-3-flash-preview": [
        "results/frontier/baseline/google-gemini-3-flash-preview",
        "results/frontier/persona/R0_neutral/google-gemini-3-flash-preview",
        # api_5games_allrisk deliberately excluded: it shares game_id/game_seed with 15
        # of the 30 baseline races (so it shares realised horizon and stopping draws
        # under this project's CRN design), but is an independently-sampled re-run, not
        # a duplicated log -- turn-by-turn actions differ in most risk-0.6/0.9 games
        # (only risk-0.1 matches exactly, trivially, since that cell is ~100% Unsafe
        # either way). Pooling it with baseline would violate the CRN-independence
        # assumption the analyzer's clustering relies on, which is presumably why
        # analyze_two_player_paper_figures.py already excludes it as a "superseded
        # overlapping pilot" -- not because the log is literally duplicated.
    ],
    "google/gemini-3.1-flash-lite-preview": ["results/frontier/baseline/google-gemini-3.1-flash-lite-preview"],
    "google/gemini-3.5-flash-lite": ["results/frontier/baseline/google-gemini-3.5-flash-lite"],
}
CORE_FEATURES = ["own_prev_unsafe", "opponent_prev_unsafe", "progress_gap", "max_private_risk", "round_number"]
AGE_BIN_MIDPOINT = {
    "18-22": 20, "23-27": 25, "28-32": 30, "33-37": 35, "38-42": 40,
    "43-47": 45, "48-52": 50, "53-57": 55, "58-62": 60, "68-72": 70,
}


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


def fit_and_explain(X: pd.DataFrame, y: pd.Series, feature_names: list[str], seed: int = 0) -> dict:
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=seed, stratify=y if y.nunique() > 1 else None)
    clf = RandomForestClassifier(n_estimators=400, max_depth=6, min_samples_leaf=10, random_state=seed, n_jobs=-1)
    clf.fit(X_train, y_train)
    test_acc = float(clf.score(X_test, y_test))
    base_rate = float(y.mean())
    majority_baseline = max(float(y_test.mean()), 1 - float(y_test.mean()))
    proba = clf.predict_proba(X_test)[:, 1]
    pred = clf.predict(X_test)
    # AUC/balanced-accuracy need both classes present in the held-out fold.
    auc = float(roc_auc_score(y_test, proba)) if y_test.nunique() > 1 else None
    bal_acc = float(balanced_accuracy_score(y_test, pred)) if y_test.nunique() > 1 else None

    impurity = dict(zip(feature_names, clf.feature_importances_.tolist()))

    perm = permutation_importance(clf, X_test, y_test, n_repeats=20, random_state=seed, n_jobs=-1)
    perm_imp = dict(zip(feature_names, perm.importances_mean.tolist()))

    explainer = shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_test)
    # sklearn RF binary classifier: shap_values may be (n,features,2) or a list of 2 arrays depending on version
    sv_arr = np.asarray(sv)
    if sv_arr.ndim == 3:
        sv_pos = sv_arr[:, :, 1]
    elif isinstance(sv, list):
        sv_pos = sv[1]
    else:
        sv_pos = sv_arr
    mean_abs_shap = dict(zip(feature_names, np.abs(sv_pos).mean(axis=0).tolist()))

    return {
        "n_train": int(len(X_train)), "n_test": int(len(X_test)), "test_accuracy": test_acc,
        "majority_baseline_accuracy": majority_baseline, "roc_auc": auc, "balanced_accuracy": bal_acc,
        "base_rate": base_rate, "impurity_importance": impurity, "permutation_importance": perm_imp,
        "mean_abs_shap": mean_abs_shap,
    }


def human_core_frame() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(HUMAN_CSV)
    d = df[df["round_number"] > 1].copy()
    X = pd.DataFrame({
        "own_prev_unsafe": d["decision_lag"].astype(float),
        "opponent_prev_unsafe": d["decision_opponent_lag"].astype(float),
        "progress_gap": d["delta_steps_lag"].astype(float),
        "max_private_risk": d["max_private_risk"].astype(float),
        "round_number": d["round_number"].astype(float),
    })
    y = d["decision"].astype(int)
    return X, y


def human_demographics_frame() -> tuple[pd.DataFrame, pd.Series]:
    df = pd.read_csv(HUMAN_CSV)
    d = df[df["round_number"] > 1].copy()
    d = d[d["sex"] != "CONSENT_REVOKED"]
    d = d[~d["nationality_group"].isin(["DATA_EXPIRED", "CONSENT_REVOKED"])]
    d["age_numeric"] = d["age"].map(AGE_BIN_MIDPOINT)
    d = d.dropna(subset=["age_numeric", "risk_gamble_choice"])
    X = pd.DataFrame({
        "own_prev_unsafe": d["decision_lag"].astype(float),
        "opponent_prev_unsafe": d["decision_opponent_lag"].astype(float),
        "progress_gap": d["delta_steps_lag"].astype(float),
        "max_private_risk": d["max_private_risk"].astype(float),
        "round_number": d["round_number"].astype(float),
        "sex_male": (d["sex"] == "Male").astype(float),
        "age": d["age_numeric"].astype(float),
        "nationality_south_africa": (d["nationality_group"] == "South Africa").astype(float),
        "nationality_poland": (d["nationality_group"] == "Poland").astype(float),
        "risk_gamble_choice": d["risk_gamble_choice"].astype(float),
    })
    y = d["decision"].astype(int)
    return X, y


def llm_core_frame(model: str) -> tuple[pd.DataFrame, pd.Series]:
    rows = []
    for d in NEUTRAL_INPUTS[model]:
        p = ROOT / d / "turns.jsonl"
        with open(p) as f:
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


def fig_importance_heatmap(results: dict[str, dict]) -> None:
    setup_plot()
    labels = ["Human"] + [MODEL_LABELS[m] for m in NEUTRAL_INPUTS]
    keys = ["human"] + list(NEUTRAL_INPUTS.keys())
    feature_labels = {
        "own_prev_unsafe": "Own previous\naction Unsafe", "opponent_prev_unsafe": "Opponent's previous\naction Unsafe",
        "progress_gap": "Progress gap\n(own - opponent/others)", "max_private_risk": "Risk treatment\n(max_private_risk)",
        "round_number": "Round number",
    }
    grid = []
    for k in keys:
        imp = results[k]["mean_abs_shap"]
        total = sum(imp.values()) or 1.0
        grid.append([imp[f] / total for f in CORE_FEATURES])
    grid = np.array(grid)

    fig, ax = plt.subplots(figsize=(9.6, 0.85 * len(keys) + 1.8))
    im = ax.imshow(grid, cmap="Blues", vmin=0, vmax=grid.max(), aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            val = grid[i, j]
            ax.text(j, i, f"{val*100:.0f}%", ha="center", va="center",
                    color="white" if val > grid.max() * 0.55 else PALETTE["navy"], fontsize=9.5, weight="bold")
    ax.set_xticks(range(len(CORE_FEATURES)), [feature_labels[f] for f in CORE_FEATURES], fontsize=8.5)
    ax.set_yticks(range(len(labels)), labels, fontsize=9.5)
    ax.set_title("Share of predictive weight (mean |SHAP value|) per feature,\nRandom Forest fit separately on each population's own decisions", pad=14)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(CORE_FEATURES))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(labels))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    fig.text(0.01, -0.02,
              "Same 5 mechanical features fit on both sides (round >= 2 decisions only); shares are normalized per row so they sum to 100%.\n"
              "This describes what a classifier leans on to reproduce each population's choices -- not a causal or mechanistic claim.",
              fontsize=8, color=PALETTE["slate"])
    save_figure(fig, FIGURES / "feature_importance_shap_heatmap")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    results: dict[str, dict] = {}

    X, y = human_core_frame()
    results["human"] = fit_and_explain(X, y, CORE_FEATURES)
    r = results["human"]
    print("human core: n=", len(X), "base_rate", r["base_rate"], "test_acc", r["test_accuracy"], "majority_baseline", r["majority_baseline_accuracy"], "AUC", r["roc_auc"], "bal_acc", r["balanced_accuracy"])

    for model in NEUTRAL_INPUTS:
        X, y = llm_core_frame(model)
        results[model] = fit_and_explain(X, y, CORE_FEATURES)
        r = results[model]
        print(model, "n=", len(X), "base_rate", r["base_rate"], "test_acc", r["test_accuracy"], "majority_baseline", r["majority_baseline_accuracy"], "AUC", r["roc_auc"], "bal_acc", r["balanced_accuracy"])

    demo_features = ["own_prev_unsafe", "opponent_prev_unsafe", "progress_gap", "max_private_risk", "round_number",
                      "sex_male", "age", "nationality_south_africa", "nationality_poland", "risk_gamble_choice"]
    Xd, yd = human_demographics_frame()
    results["human_with_demographics"] = fit_and_explain(Xd, yd, demo_features)
    print("human+demographics: n=", len(Xd), "test_acc", results["human_with_demographics"]["test_accuracy"])

    with open(DATA / "feature_importance_results.json", "w") as f:
        json.dump(results, f, indent=2)

    fig_importance_heatmap(results)
    print("wrote", FIGURES / "feature_importance_shap_heatmap.png")


if __name__ == "__main__":
    main()
