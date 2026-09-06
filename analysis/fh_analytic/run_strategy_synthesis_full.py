#!/usr/bin/env python3
"""Full strategy synthesis with position, trees, clustering, and HMM embeddings."""

from __future__ import annotations

import json
import math
import textwrap
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier, _tree


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
SYNTH_FIG_DIR = FIGURES_DIR / "strategy_synthesis_full"
REPORTS_DIR = OUTPUT_DIR / "reports"
RANDOM_SEED = 260726

BLUE = "#2F6B9A"
ORANGE = "#D9822B"
GOLD = "#C9A227"
OLIVE = "#6B7D3D"
PINK = "#B45A7C"
TEAL = "#3A8F8A"
PURPLE = "#7C4D79"
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
PALETTE = [BLUE, ORANGE, GOLD, OLIVE, PINK, TEAL, PURPLE, "#8A7A5C", "#4E6E5D", "#A15C38"]


@dataclass
class CategoricalHMM:
    k: int
    n_symbols: int
    startprob: np.ndarray
    transmat: np.ndarray
    emissions: np.ndarray
    loglik: float
    bic: float
    iterations: int
    converged: bool


def ensure_dirs() -> None:
    for path in [DERIVED_DIR, SYNTH_FIG_DIR, REPORTS_DIR]:
        path.mkdir(parents=True, exist_ok=True)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "axes.facecolor": WHITE,
            "savefig.facecolor": PAPER,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.8,
            "grid.alpha": 1.0,
            "legend.frameon": False,
        }
    )


def savefig(fig: plt.Figure, filename: str) -> Path:
    path = SYNTH_FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100 * value:.1f}%"


def strategy_key(row: pd.Series) -> str:
    return f"{row['source_run']}|{row['game_id']}|p{int(row['player_index'])}"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv", low_memory=False)
    turns = turns[turns["manifest_status"].eq("completed") & ~turns["duplicate_grain_key"].fillna(False).astype(bool)].copy()
    turns = turns[turns["unsafe"].notna()].copy()
    for col in [
        "unsafe",
        "round",
        "player_index",
        "progress_gap_before",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
        "own_private_risk_before",
        "max_private_risk",
        "unsafe_fraction_after",
        "first_round_unsafe",
    ]:
        if col in turns.columns:
            turns[col] = pd.to_numeric(turns[col], errors="coerce")
    turns["sequence_id"] = turns.apply(strategy_key, axis=1)
    turns["gap_zone"] = np.select(
        [turns["progress_gap_before"] < -0.5, turns["progress_gap_before"] > 0.5],
        ["behind", "ahead"],
        default="tied",
    )
    prev_own = turns["own_prev_unsafe"].fillna(-1).astype(int).map({0: "S", 1: "U", -1: "?"})
    prev_opp = turns["opponent_prev_unsafe"].fillna(-1).astype(int).map({0: "S", 1: "U", -1: "?"})
    turns["prev_pair"] = prev_own + prev_opp
    turns["obs_token"] = (
        "a"
        + turns["unsafe"].astype(int).astype(str)
        + "_g"
        + turns["gap_zone"].astype(str)
        + "_p"
        + turns["prev_pair"].astype(str)
    )
    expanded_path = DERIVED_DIR / "expanded_strategy_player_fits.csv"
    if expanded_path.exists():
        expanded = pd.read_csv(expanded_path, low_memory=False)
    else:
        expanded = pd.DataFrame()
    return turns, expanded


def rate_or_nan(values: pd.Series) -> float:
    values = values.dropna()
    return float(values.mean()) if len(values) else np.nan


def player_feature_rows(turns: pd.DataFrame, expanded: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for sequence_id, frame in turns.sort_values(["sequence_id", "round"]).groupby("sequence_id", sort=False):
        frame = frame.sort_values("round")
        y = frame["unsafe"].astype(int).to_numpy()
        first = frame.iloc[0]
        gap = frame["progress_gap_before"].astype(float)
        ahead = frame["gap_zone"].eq("ahead")
        behind = frame["gap_zone"].eq("behind")
        tied = frame["gap_zone"].eq("tied")
        opp_u = frame["opponent_prev_unsafe"].eq(1)
        opp_s = frame["opponent_prev_unsafe"].eq(0)
        own_u = frame["own_prev_unsafe"].eq(1)
        own_s = frame["own_prev_unsafe"].eq(0)
        switches = int(np.sum(y[1:] != y[:-1])) if len(y) > 1 else 0
        alt_safe = np.array([i % 2 for i in range(len(y))], dtype=int)
        alt_unsafe = 1 - alt_safe
        early = y[: max(1, len(y) // 2)].mean()
        late = y[max(1, len(y) // 2) :].mean() if len(y) > 1 else early
        rows.append(
            {
                "sequence_id": sequence_id,
                "model_slug": first["model_slug"],
                "family": first["family"],
                "provider": first["provider"],
                "persona_mode": first["persona_mode"],
                "condition": first["condition"],
                "analysis_scope": first["analysis_scope"],
                "n_turns": len(y),
                "unsafe_rate": float(y.mean()),
                "first_unsafe": int(y[0]),
                "last_unsafe": int(y[-1]),
                "early_unsafe_rate": float(early),
                "late_unsafe_rate": float(late),
                "late_minus_early": float(late - early),
                "switch_count": switches,
                "switch_rate": switches / max(1, len(y) - 1),
                "alt_safe_match": float((y == alt_safe).mean()),
                "alt_unsafe_match": float((y == alt_unsafe).mean()),
                "mean_gap": float(gap.mean()),
                "unsafe_when_ahead": rate_or_nan(frame.loc[ahead, "unsafe"]),
                "unsafe_when_tied": rate_or_nan(frame.loc[tied, "unsafe"]),
                "unsafe_when_behind": rate_or_nan(frame.loc[behind, "unsafe"]),
                "n_ahead": int(ahead.sum()),
                "n_tied": int(tied.sum()),
                "n_behind": int(behind.sum()),
                "retaliation_rate": rate_or_nan(frame.loc[own_s & opp_u, "unsafe"]),
                "calm_rate": rate_or_nan(frame.loc[own_s & opp_s, "unsafe"]),
                "forgiveness_rate": rate_or_nan(1 - frame.loc[own_u & opp_s, "unsafe"]),
                "stickiness_uu": rate_or_nan(frame.loc[own_u & opp_u, "unsafe"]),
                "opportunistic_rate": rate_or_nan(frame.loc[ahead & opp_s, "unsafe"]),
                "catchup_rate": rate_or_nan(frame.loc[behind & opp_s, "unsafe"]),
                "action_pattern": "".join("U" if value else "S" for value in y),
            }
        )
    players = pd.DataFrame(rows)
    players["positional_delta"] = players["unsafe_when_ahead"] - players["unsafe_when_behind"]
    players["retaliation_lift"] = players["retaliation_rate"] - players["calm_rate"]
    if not expanded.empty:
        keep = [
            "sequence_id",
            "canonical_exact_any",
            "expanded_exact_any",
            "new_exact_beyond_canonical",
            "expanded_best_strategy",
            "expanded_best_family",
            "expanded_min_mismatch_rate",
            "expanded_near_one_mismatch",
        ]
        players = players.merge(expanded[[c for c in keep if c in expanded.columns]], on="sequence_id", how="left")
    return players


def tree_leaf_rules(
    model: DecisionTreeClassifier,
    feature_names: list[str],
    x: np.ndarray,
    y: np.ndarray,
    scope: str,
    label: str,
) -> list[dict[str, Any]]:
    tree = model.tree_
    rows: list[dict[str, Any]] = []

    def recurse(node: int, path: list[str]) -> None:
        if tree.feature[node] != _tree.TREE_UNDEFINED:
            name = feature_names[tree.feature[node]]
            threshold = tree.threshold[node]
            recurse(tree.children_left[node], path + [f"{name} <= {threshold:.3f}"])
            recurse(tree.children_right[node], path + [f"{name} > {threshold:.3f}"])
        else:
            support = int(tree.n_node_samples[node])
            values = tree.value[node][0]
            pred_class = int(np.argmax(values))
            unsafe_rate = float(values[1] / values.sum()) if values.sum() else np.nan
            rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "rule": " AND ".join(path) if path else "ALL",
                    "support_turns": support,
                    "unsafe_rate": unsafe_rate,
                    "predicted_action": "unsafe" if pred_class == 1 else "safe",
                    "tree_depth": len(path),
                }
            )

    recurse(0, [])
    total = len(y)
    for row in rows:
        row["support_share"] = row["support_turns"] / total
        row["auc_in_sample"] = roc_auc_score(y, model.predict_proba(x)[:, 1]) if len(np.unique(y)) > 1 else np.nan
    return rows


def fit_tree_rules(turns: pd.DataFrame) -> pd.DataFrame:
    features = [
        "round",
        "progress_gap_before",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
        "own_private_risk_before",
        "max_private_risk",
        "first_round_unsafe",
    ]
    scopes: list[tuple[str, str, pd.DataFrame]] = [("overall", "all", turns)]
    for family, frame in turns.groupby("family"):
        if len(frame) >= 500:
            scopes.append(("family", family, frame))
    for model_slug, frame in turns.groupby("model_slug"):
        if len(frame) >= 300:
            scopes.append(("model", model_slug, frame))

    rows: list[dict[str, Any]] = []
    for scope, label, frame in scopes:
        work = frame.copy()
        x = work[features].fillna(-1).to_numpy(dtype=float)
        y = work["unsafe"].astype(int).to_numpy()
        if len(np.unique(y)) < 2:
            continue
        min_leaf = max(50, int(len(work) * 0.025))
        tree = DecisionTreeClassifier(max_depth=3, min_samples_leaf=min_leaf, random_state=RANDOM_SEED)
        tree.fit(x, y)
        rows.extend(tree_leaf_rules(tree, features, x, y, scope, label))
    out = pd.DataFrame(rows)
    return out.sort_values(["scope", "label", "support_share"], ascending=[True, True, False])


def sequences_from_turns(turns: pd.DataFrame) -> tuple[list[np.ndarray], list[str], dict[str, int], dict[int, str]]:
    token_to_id = {token: i for i, token in enumerate(sorted(turns["obs_token"].unique()))}
    id_to_token = {v: k for k, v in token_to_id.items()}
    sequences: list[np.ndarray] = []
    ids: list[str] = []
    for sequence_id, frame in turns.sort_values(["sequence_id", "round"]).groupby("sequence_id", sort=False):
        ids.append(sequence_id)
        sequences.append(frame["obs_token"].map(token_to_id).to_numpy(dtype=int))
    return sequences, ids, token_to_id, id_to_token


def forward_backward(
    obs: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissions: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    k = len(startprob)
    t_len = len(obs)
    alpha = np.zeros((t_len, k))
    beta = np.zeros((t_len, k))
    scale = np.zeros(t_len)
    alpha[0] = startprob * emissions[:, obs[0]]
    scale[0] = alpha[0].sum()
    if scale[0] <= 0:
        scale[0] = 1e-300
    alpha[0] /= scale[0]
    for t in range(1, t_len):
        alpha[t] = (alpha[t - 1] @ transmat) * emissions[:, obs[t]]
        scale[t] = alpha[t].sum()
        if scale[t] <= 0:
            scale[t] = 1e-300
        alpha[t] /= scale[t]
    beta[-1] = 1.0
    for t in range(t_len - 2, -1, -1):
        beta[t] = transmat @ (emissions[:, obs[t + 1]] * beta[t + 1])
        beta[t] /= scale[t + 1]
    gamma = alpha * beta
    gamma /= gamma.sum(axis=1, keepdims=True)
    xi_sum = np.zeros((k, k))
    for t in range(t_len - 1):
        xi = alpha[t][:, None] * transmat * (emissions[:, obs[t + 1]] * beta[t + 1])[None, :]
        denom = xi.sum()
        if denom > 0:
            xi_sum += xi / denom
    loglik = float(np.sum(np.log(scale)))
    return loglik, gamma, xi_sum


def fit_hmm_once(sequences: list[np.ndarray], k: int, n_symbols: int, seed: int, max_iter: int = 45) -> CategoricalHMM:
    rng = np.random.default_rng(seed)
    startprob = rng.dirichlet(np.ones(k))
    transmat = np.vstack([rng.dirichlet(np.ones(k) + np.eye(k)[i] * 3.0) for i in range(k)])
    emissions = np.vstack([rng.dirichlet(np.ones(n_symbols)) for _ in range(k)])
    prev_ll = -np.inf
    converged = False
    total_obs = sum(len(seq) for seq in sequences)
    for iteration in range(1, max_iter + 1):
        start_counts = np.full(k, 1e-3)
        trans_counts = np.full((k, k), 1e-3)
        emit_counts = np.full((k, n_symbols), 1e-3)
        ll = 0.0
        for obs in sequences:
            seq_ll, gamma, xi_sum = forward_backward(obs, startprob, transmat, emissions)
            ll += seq_ll
            start_counts += gamma[0]
            trans_counts += xi_sum
            for symbol in range(n_symbols):
                mask = obs == symbol
                if mask.any():
                    emit_counts[:, symbol] += gamma[mask].sum(axis=0)
        startprob = start_counts / start_counts.sum()
        transmat = trans_counts / trans_counts.sum(axis=1, keepdims=True)
        emissions = emit_counts / emit_counts.sum(axis=1, keepdims=True)
        if abs(ll - prev_ll) < 1e-4 * (1 + abs(prev_ll)):
            converged = True
            break
        prev_ll = ll
    params = (k - 1) + k * (k - 1) + k * (n_symbols - 1)
    bic = -2 * ll + params * math.log(total_obs)
    return CategoricalHMM(k, n_symbols, startprob, transmat, emissions, ll, bic, iteration, converged)


def fit_hmm_scan(sequences: list[np.ndarray], n_symbols: int) -> tuple[CategoricalHMM, pd.DataFrame]:
    fits: list[CategoricalHMM] = []
    for k in range(2, 7):
        best: CategoricalHMM | None = None
        for restart in range(2):
            fit = fit_hmm_once(sequences, k, n_symbols, RANDOM_SEED + k * 100 + restart)
            if best is None or fit.loglik > best.loglik:
                best = fit
        assert best is not None
        fits.append(best)
    scan = pd.DataFrame(
        [
            {
                "k": fit.k,
                "loglik": fit.loglik,
                "bic": fit.bic,
                "iterations": fit.iterations,
                "converged": fit.converged,
            }
            for fit in fits
        ]
    )
    best = min(fits, key=lambda item: item.bic)
    return best, scan


def hmm_embeddings(
    sequences: list[np.ndarray],
    sequence_ids: list[str],
    fit: CategoricalHMM,
    id_to_token: dict[int, str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for sequence_id, obs in zip(sequence_ids, sequences):
        _, gamma, xi_sum = forward_backward(obs, fit.startprob, fit.transmat, fit.emissions)
        record: dict[str, Any] = {"sequence_id": sequence_id}
        occ = gamma.mean(axis=0)
        first = gamma[0]
        last = gamma[-1]
        for state in range(fit.k):
            record[f"hmm_occ_s{state}"] = float(occ[state])
            record[f"hmm_first_s{state}"] = float(first[state])
            record[f"hmm_last_s{state}"] = float(last[state])
        record["hmm_dominant_state"] = int(np.argmax(occ))
        record["hmm_state_entropy"] = float(-np.sum(occ * np.log(np.clip(occ, 1e-12, 1))))
        rows.append(record)

    state_rows: list[dict[str, Any]] = []
    for state in range(fit.k):
        probs = fit.emissions[state]
        unsafe_prob = 0.0
        ahead_prob = 0.0
        behind_prob = 0.0
        tied_prob = 0.0
        opp_prev_u_prob = 0.0
        own_prev_u_prob = 0.0
        top_tokens = []
        for symbol, prob in enumerate(probs):
            token = id_to_token[symbol]
            unsafe_prob += prob if token.startswith("a1") else 0
            ahead_prob += prob if "_gahead_" in token else 0
            behind_prob += prob if "_gbehind_" in token else 0
            tied_prob += prob if "_gtied_" in token else 0
            own_prev_u_prob += prob if token.endswith("pU?") or "_pU" in token else 0
            opp_prev_u_prob += prob if token.endswith("U") else 0
        for symbol in np.argsort(probs)[-6:][::-1]:
            top_tokens.append(f"{id_to_token[int(symbol)]}:{probs[int(symbol)]:.2f}")
        state_rows.append(
            {
                "state": state,
                "unsafe_emission_prob": float(unsafe_prob),
                "ahead_emission_prob": float(ahead_prob),
                "tied_emission_prob": float(tied_prob),
                "behind_emission_prob": float(behind_prob),
                "own_prev_unsafe_prob": float(own_prev_u_prob),
                "opponent_prev_unsafe_prob": float(opp_prev_u_prob),
                "self_transition_prob": float(fit.transmat[state, state]),
                "top_tokens": "; ".join(top_tokens),
            }
        )
    return pd.DataFrame(rows), pd.DataFrame(state_rows)


def choose_strategy_name(row: pd.Series) -> str:
    unsafe = row["unsafe_rate"]
    pos = row["positional_delta"]
    ret = row["retaliation_lift"]
    switch = row["switch_rate"]
    alt = max(row["alt_safe_match"], row["alt_unsafe_match"])
    late = row["late_minus_early"]
    stick = row["stickiness_uu"]
    forgive = row["forgiveness_rate"]
    if unsafe <= 0.12:
        return "ổn định an toàn"
    if unsafe >= 0.88:
        return "ổn định unsafe"
    if alt >= 0.82 or switch >= 0.70:
        return "luân phiên / anti-copy"
    if pd.notna(pos) and pos >= 0.30:
        return "đánh khi đang dẫn"
    if pd.notna(pos) and pos <= -0.30:
        return "đánh để gỡ khi tụt"
    if pd.notna(ret) and ret >= 0.25 and pd.notna(stick) and stick >= 0.60:
        return "trả đũa và leo thang"
    if pd.notna(forgive) and forgive >= 0.65 and unsafe < 0.55:
        return "đánh thử rồi hạ nhiệt"
    if late >= 0.30:
        return "trì hoãn rồi tấn công"
    if late <= -0.30:
        return "mở unsafe rồi hạ nhiệt"
    return "thích nghi hỗn hợp"


def cluster_embeddings(players: pd.DataFrame, hmm_embed: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    data = players.merge(hmm_embed, on="sequence_id", how="left")
    feature_cols = [
        "unsafe_rate",
        "first_unsafe",
        "last_unsafe",
        "late_minus_early",
        "switch_rate",
        "alt_safe_match",
        "alt_unsafe_match",
        "positional_delta",
        "retaliation_lift",
        "forgiveness_rate",
        "stickiness_uu",
        "opportunistic_rate",
        "catchup_rate",
        "expanded_min_mismatch_rate",
    ] + [c for c in data.columns if c.startswith("hmm_occ_s")]
    x = data[feature_cols].copy()
    for col in x.columns:
        x[col] = pd.to_numeric(x[col], errors="coerce")
        x[col] = x[col].fillna(x[col].median())
    x_scaled = StandardScaler().fit_transform(x)
    scan_rows = []
    best_k = 0
    best_score = -np.inf
    best_labels: np.ndarray | None = None
    for k in range(3, 10):
        km = KMeans(n_clusters=k, random_state=RANDOM_SEED, n_init=30)
        labels = km.fit_predict(x_scaled)
        score = silhouette_score(x_scaled, labels)
        scan_rows.append({"k": k, "silhouette": float(score), "inertia": float(km.inertia_)})
        if score > best_score:
            best_score = score
            best_k = k
            best_labels = labels
    assert best_labels is not None
    data["cluster"] = best_labels
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    coords = pca.fit_transform(x_scaled)
    data["embed_x"] = coords[:, 0]
    data["embed_y"] = coords[:, 1]
    scan = pd.DataFrame(scan_rows)
    profile = (
        data.groupby("cluster")
        .agg(
            player_sequences=("sequence_id", "size"),
            unsafe_rate=("unsafe_rate", "mean"),
            first_unsafe=("first_unsafe", "mean"),
            switch_rate=("switch_rate", "mean"),
            alt_match=("alt_safe_match", "mean"),
            positional_delta=("positional_delta", "mean"),
            retaliation_lift=("retaliation_lift", "mean"),
            forgiveness_rate=("forgiveness_rate", "mean"),
            stickiness_uu=("stickiness_uu", "mean"),
            late_minus_early=("late_minus_early", "mean"),
            alt_safe_match=("alt_safe_match", "mean"),
            alt_unsafe_match=("alt_unsafe_match", "mean"),
            expanded_exact=("expanded_exact_any", "mean"),
            new_exact=("new_exact_beyond_canonical", "mean"),
        )
        .reset_index()
    )
    profile["share"] = profile["player_sequences"] / len(data)
    profile["strategy_name"] = profile.apply(choose_strategy_name, axis=1)
    data = data.merge(profile[["cluster", "strategy_name"]], on="cluster", how="left")
    return data, profile, scan


def plot_tree_rules(tree_rules: pd.DataFrame) -> Path:
    data = tree_rules.copy()
    data = data[data["support_share"].ge(0.05)]
    data["distance"] = (data["unsafe_rate"] - 0.5).abs()
    data = data.sort_values(["distance", "support_share"], ascending=False).head(14)
    data = data.sort_values("distance")
    labels = data.apply(lambda r: f"{r['scope']}:{r['label']}\n{r['predicted_action']} | {r['rule']}", axis=1)
    colors = [ORANGE if value >= 0.5 else BLUE for value in data["unsafe_rate"]]
    fig, ax = plt.subplots(figsize=(13, max(7, len(data) * 0.42)))
    bars = ax.barh(np.arange(len(data)), data["unsafe_rate"], color=colors, edgecolor=WHITE, linewidth=0.8)
    ax.axvline(0.5, color=INK, linewidth=1)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels([textwrap.fill(label, 55) for label in labels], fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Unsafe rate inside tree leaf")
    ax.set_title("Reusable Decision-Tree Rules by Scope")
    ax.text(0, 1.04, "Rules are extracted from depth-3 trees fit on turn-level behavior.", transform=ax.transAxes, color=MUTED, fontsize=10)
    for bar, (_, row) in zip(bars, data.iterrows()):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{pct(row['unsafe_rate'])}, n={int(row['support_turns'])}", va="center", fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.48, right=0.95, top=0.86, bottom=0.12)
    return savefig(fig, "01_tree_rules.png")


def plot_hmm_states(state_profile: pd.DataFrame) -> Path:
    cols = [
        "unsafe_emission_prob",
        "ahead_emission_prob",
        "tied_emission_prob",
        "behind_emission_prob",
        "opponent_prev_unsafe_prob",
        "own_prev_unsafe_prob",
        "self_transition_prob",
    ]
    data = state_profile.set_index("state")[cols]
    fig, ax = plt.subplots(figsize=(10.5, 5.8))
    im = ax.imshow(data.to_numpy(), cmap="YlGnBu", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n").replace("prob", "") for c in cols], fontsize=8)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels([f"State {s}" for s in data.index])
    ax.set_title("HMM Latent State Profiles")
    ax.text(0, 1.05, "Categorical HMM emissions over action, position, and previous-state tokens.", transform=ax.transAxes, color=MUTED, fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.iloc[i, j]:.0%}", ha="center", va="center", fontsize=8, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cbar.set_label("Probability")
    fig.subplots_adjust(left=0.12, right=0.95, top=0.84, bottom=0.18)
    return savefig(fig, "02_hmm_state_profiles.png")


def plot_cluster_embedding(clustered: pd.DataFrame, profile: pd.DataFrame) -> Path:
    sample = clustered.sample(min(3500, len(clustered)), random_state=RANDOM_SEED)
    fig, ax = plt.subplots(figsize=(10.5, 7.2))
    for cluster in sorted(sample["cluster"].unique()):
        frame = sample[sample["cluster"].eq(cluster)]
        name = profile.loc[profile["cluster"].eq(cluster), "strategy_name"].iloc[0]
        ax.scatter(frame["embed_x"], frame["embed_y"], s=16, alpha=0.55, color=PALETTE[int(cluster) % len(PALETTE)], label=f"C{cluster}: {name}")
    ax.set_xlabel("Embedding PC1")
    ax.set_ylabel("Embedding PC2")
    ax.set_title("Trajectory Embedding Clusters")
    ax.text(0, 1.04, "Embedding combines sequence features with HMM posterior state occupancy.", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8)
    fig.subplots_adjust(left=0.1, right=0.72, top=0.86, bottom=0.12)
    return savefig(fig, "03_embedding_clusters.png")


def plot_cluster_profiles(profile: pd.DataFrame) -> Path:
    cols = [
        "share",
        "unsafe_rate",
        "first_unsafe",
        "switch_rate",
        "positional_delta",
        "retaliation_lift",
        "forgiveness_rate",
        "stickiness_uu",
        "late_minus_early",
        "new_exact",
    ]
    data = profile.set_index("strategy_name")[cols].copy()
    fig, ax = plt.subplots(figsize=(12.5, max(6, len(data) * 0.45)))
    im = ax.imshow(data.to_numpy(), aspect="auto", cmap="PuOr_r", vmin=-1, vmax=1)
    ax.set_xticks(np.arange(len(cols)))
    ax.set_xticklabels([c.replace("_", "\n") for c in cols], fontsize=8)
    ax.set_yticks(np.arange(len(data)))
    ax.set_yticklabels([f"{idx} (C{profile.iloc[i]['cluster']})" for i, idx in enumerate(data.index)], fontsize=9)
    ax.set_title("Cluster Strategy Profiles")
    ax.text(0, 1.04, "Signed cells show how each discovered group behaves by position, memory, and sequence shape.", transform=ax.transAxes, color=MUTED, fontsize=10)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            ax.text(j, i, f"{data.iloc[i, j]:.0%}", ha="center", va="center", fontsize=8, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Rate / lift")
    fig.subplots_adjust(left=0.26, right=0.95, top=0.86, bottom=0.17)
    return savefig(fig, "04_cluster_profiles.png")


def plot_cluster_mix(clustered: pd.DataFrame) -> Path:
    counts = clustered.groupby(["model_slug", "strategy_name"]).size().rename("n").reset_index()
    counts["share"] = counts["n"] / counts.groupby("model_slug")["n"].transform("sum")
    plot = counts.pivot(index="model_slug", columns="strategy_name", values="share").fillna(0)
    order = [m for m in MODEL_ORDER if m in plot.index] + [m for m in plot.index if m not in MODEL_ORDER]
    plot = plot.loc[order]
    cols = plot.sum().sort_values(ascending=False).index.tolist()
    fig, ax = plt.subplots(figsize=(12.5, 7))
    left = np.zeros(len(plot))
    y = np.arange(len(plot))
    for idx, col in enumerate(cols):
        vals = plot[col].to_numpy()
        ax.barh(y, vals, left=left, color=PALETTE[idx % len(PALETTE)], edgecolor=WHITE, linewidth=0.8, label=col)
        for i, val in enumerate(vals):
            if val >= 0.08:
                ax.text(left[i] + val / 2, i, f"{val:.0%}", ha="center", va="center", color=WHITE, fontsize=8)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in plot.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of player trajectories")
    ax.set_title("Discovered Strategy Mix by Model")
    ax.text(0, 1.04, "Cluster labels are inferred from HMM+behavior embeddings, not assigned from the original four strategies.", transform=ax.transAxes, color=MUTED, fontsize=10)
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.30), fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.24, right=0.98, top=0.86, bottom=0.28)
    return savefig(fig, "05_cluster_mix_by_model.png")


def plot_hmm_k_scan(scan: pd.DataFrame, cluster_scan: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    axes[0].plot(scan["k"], scan["bic"], marker="o", color=BLUE, linewidth=2)
    axes[0].set_title("HMM State Count Scan")
    axes[0].set_xlabel("HMM states")
    axes[0].set_ylabel("BIC")
    axes[1].plot(cluster_scan["k"], cluster_scan["silhouette"], marker="o", color=ORANGE, linewidth=2)
    axes[1].set_title("Embedding Cluster Count Scan")
    axes[1].set_xlabel("Clusters")
    axes[1].set_ylabel("Silhouette")
    fig.suptitle("Model Selection Diagnostics", fontsize=16, y=0.98)
    fig.subplots_adjust(left=0.08, right=0.98, top=0.82, bottom=0.15, wspace=0.25)
    return savefig(fig, "06_model_selection_scans.png")


def make_contact_sheet(paths: list[Path]) -> Path:
    images = [mpimg.imread(path) for path in paths]
    fig, axes = plt.subplots(3, 2, figsize=(16, 19))
    axes = axes.ravel()
    for ax, image, path in zip(axes, images, paths):
        ax.imshow(image)
        ax.set_title(path.stem.replace("_", " ").title(), fontsize=11, pad=8)
        ax.axis("off")
    for ax in axes[len(images) :]:
        ax.axis("off")
    fig.suptitle("Full Strategy Synthesis", fontsize=22, y=0.995)
    fig.subplots_adjust(top=0.965, hspace=0.08, wspace=0.04)
    path = SYNTH_FIG_DIR / "fh_strategy_synthesis_full_contact_sheet.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    profile: pd.DataFrame,
    state_profile: pd.DataFrame,
    hmm_scan: pd.DataFrame,
    cluster_scan: pd.DataFrame,
    tree_rules: pd.DataFrame,
    clustered: pd.DataFrame,
    figures: list[Path],
    contact_sheet: Path,
) -> Path:
    cluster_lines = []
    for _, row in profile.sort_values("share", ascending=False).iterrows():
        cluster_lines.append(
            f"- C{int(row['cluster'])} `{row['strategy_name']}`: share {pct(row['share'])}, unsafe {pct(row['unsafe_rate'])}, "
            f"positional_delta {pct(row['positional_delta'])}, retaliation_lift {pct(row['retaliation_lift'])}, "
            f"forgiveness {pct(row['forgiveness_rate'])}, stickiness_UU {pct(row['stickiness_uu'])}."
        )
    state_lines = []
    for _, row in state_profile.sort_values("unsafe_emission_prob").iterrows():
        state_lines.append(
            f"- State {int(row['state'])}: unsafe emission {pct(row['unsafe_emission_prob'])}, ahead {pct(row['ahead_emission_prob'])}, "
            f"behind {pct(row['behind_emission_prob'])}, self-transition {pct(row['self_transition_prob'])}; top tokens: {row['top_tokens']}."
        )
    top_rules = tree_rules.copy()
    top_rules["distance"] = (top_rules["unsafe_rate"] - 0.5).abs()
    top_rules = top_rules[top_rules["support_share"].ge(0.05)].sort_values(["distance", "support_share"], ascending=False).head(10)
    rule_lines = [
        f"- {row['scope']} `{row['label']}`: {row['predicted_action']} {pct(row['unsafe_rate'])}, support {pct(row['support_share'])}; rule: `{row['rule']}`."
        for _, row in top_rules.iterrows()
    ]
    model_mix = (
        clustered.groupby(["model_slug", "strategy_name"]).size().rename("n").reset_index()
        .assign(share=lambda d: d["n"] / d.groupby("model_slug")["n"].transform("sum"))
    )
    model_lines = []
    for model in MODEL_ORDER:
        frame = model_mix[model_mix["model_slug"].eq(model)].sort_values("share", ascending=False).head(3)
        if frame.empty:
            continue
        parts = [f"{row['strategy_name']} {pct(row['share'])}" for _, row in frame.iterrows()]
        model_lines.append(f"- {MODEL_LABELS.get(model, model)}: " + ", ".join(parts) + ".")

    body = f"""# Full Strategy Synthesis

## What Was Tested

This stage combines four views of strategy: expanded deterministic rules, position-conditioned behavior, decision-tree rules by model/family, and unsupervised HMM+clustering embeddings.

## HMM and Cluster Model Selection

- Best HMM state count by BIC: {int(hmm_scan.sort_values('bic').iloc[0]['k'])}
- Best embedding cluster count by silhouette: {int(cluster_scan.sort_values('silhouette', ascending=False).iloc[0]['k'])}

## Latent HMM States

{chr(10).join(state_lines)}

## Discovered Strategy Groups

{chr(10).join(cluster_lines)}

## Model Mix

{chr(10).join(model_lines)}

## Reusable Tree Rules

{chr(10).join(rule_lines)}

## Synthesis

The data supports treating strategy as a small set of behavioral regimes rather than only AS/AU/CS/CAS. The strongest additional axes are position-conditioned aggression, anti-copy/alternation, delayed/probe patterns, and escalation stickiness after mutual unsafe states. Gap-based rules are especially useful as mechanisms, while exact deterministic coverage is carried more by anti-copy and sequence motifs.

## Deliverables

- Contact sheet: `{contact_sheet}`
- Figures: {", ".join(f"`{path.name}`" for path in figures)}
- Tables: `strategy_synthesis_player_embeddings.csv`, `strategy_synthesis_cluster_profiles.csv`, `strategy_synthesis_hmm_state_profiles.csv`, `strategy_synthesis_hmm_k_scan.csv`, `strategy_synthesis_cluster_k_scan.csv`, `strategy_synthesis_tree_rules.csv`
"""
    path = REPORTS_DIR / "fh_strategy_synthesis_full.md"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    configure_matplotlib()
    turns, expanded = load_data()
    players = player_feature_rows(turns, expanded)
    tree_rules = fit_tree_rules(turns)
    sequences, sequence_ids, token_to_id, id_to_token = sequences_from_turns(turns)
    hmm_fit, hmm_scan = fit_hmm_scan(sequences, len(token_to_id))
    hmm_embed, state_profile = hmm_embeddings(sequences, sequence_ids, hmm_fit, id_to_token)
    clustered, cluster_profile, cluster_scan = cluster_embeddings(players, hmm_embed)

    clustered.to_csv(DERIVED_DIR / "strategy_synthesis_player_embeddings.csv", index=False)
    cluster_profile.to_csv(DERIVED_DIR / "strategy_synthesis_cluster_profiles.csv", index=False)
    state_profile.to_csv(DERIVED_DIR / "strategy_synthesis_hmm_state_profiles.csv", index=False)
    hmm_scan.to_csv(DERIVED_DIR / "strategy_synthesis_hmm_k_scan.csv", index=False)
    cluster_scan.to_csv(DERIVED_DIR / "strategy_synthesis_cluster_k_scan.csv", index=False)
    tree_rules.to_csv(DERIVED_DIR / "strategy_synthesis_tree_rules.csv", index=False)

    figures = [
        plot_tree_rules(tree_rules),
        plot_hmm_states(state_profile),
        plot_cluster_embedding(clustered, cluster_profile),
        plot_cluster_profiles(cluster_profile),
        plot_cluster_mix(clustered),
        plot_hmm_k_scan(hmm_scan, cluster_scan),
    ]
    contact_sheet = make_contact_sheet(figures)
    report = write_report(cluster_profile, state_profile, hmm_scan, cluster_scan, tree_rules, clustered, figures, contact_sheet)
    print(
        json.dumps(
            {
                "players": int(len(players)),
                "turns": int(len(turns)),
                "hmm_best_k": int(hmm_fit.k),
                "hmm_best_bic": float(hmm_fit.bic),
                "cluster_best_k": int(cluster_scan.sort_values("silhouette", ascending=False).iloc[0]["k"]),
                "cluster_profile": cluster_profile[["cluster", "strategy_name", "share", "unsafe_rate"]].to_dict(orient="records"),
                "report": str(report),
                "contact_sheet": str(contact_sheet),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
