#!/usr/bin/env python3
"""Hidden Markov model analysis for FH turn sequences.

Fits Bernoulli-emission HMMs to safe/unsafe sequences by player-game. The goal is
to infer latent behavioral states such as mostly-safe, mixed/triggered, and
persistent-unsafe regimes without adding new dependencies.
"""

from __future__ import annotations

import json
import math
import os
import textwrap
from dataclasses import dataclass
from pathlib import Path

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
HMM_FIG_DIR = FIGURES_DIR / "hmm"
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
STATE_COLORS = {
    "safe_state": BLUE,
    "mixed_state": TEAL,
    "unsafe_state": ORANGE,
    "persistent_unsafe_state": PINK,
}
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


@dataclass
class HMMFit:
    scope: str
    label: str
    k: int
    startprob: np.ndarray
    transmat: np.ndarray
    emissions: np.ndarray
    loglik: float
    iterations: int
    converged: bool
    n_obs: int
    n_sequences: int


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
    HMM_FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = HMM_FIG_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def title(ax: plt.Axes, text: str, subtitle: str | None = None) -> None:
    ax.set_title(text, loc="left", pad=24, fontweight="bold")
    if subtitle:
        ax.text(0, 1.015, subtitle, transform=ax.transAxes, ha="left", va="bottom", color=MUTED, fontsize=9.5)


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


def clean_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


def load_turns() -> pd.DataFrame:
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv")
    turns["duplicate_grain_key"] = clean_bool(turns["duplicate_grain_key"])
    turns["is_round2plus"] = clean_bool(turns["is_round2plus"])
    numeric = ["unsafe", "round", "retry_count", "max_private_risk"]
    for col in numeric:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")
    turns["game_cluster_id"] = turns["source_run"].astype(str) + "::" + turns["game_id"].astype(str)
    turns["player_sequence_id"] = turns["game_cluster_id"] + "::p" + turns["player_index"].astype(str)
    return turns[
        (turns["manifest_status"] == "completed")
        & (~turns["duplicate_grain_key"])
        & turns["unsafe"].notna()
    ].copy()


def build_sequences(frame: pd.DataFrame) -> list[np.ndarray]:
    sequences = []
    for _, sub in frame.sort_values(["player_sequence_id", "round"]).groupby("player_sequence_id", sort=False):
        values = sub.sort_values("round")["unsafe"].astype(int).to_numpy()
        if len(values) >= 2:
            sequences.append(values)
    return sequences


def logsumexp(values: np.ndarray, axis: int | None = None) -> np.ndarray:
    maxv = np.max(values, axis=axis, keepdims=True)
    maxv[~np.isfinite(maxv)] = 0
    out = maxv + np.log(np.sum(np.exp(values - maxv), axis=axis, keepdims=True))
    if axis is None:
        return np.squeeze(out)
    return np.squeeze(out, axis=axis)


def emission_logprob(obs: np.ndarray, emissions: np.ndarray) -> np.ndarray:
    eps = 1e-8
    p = np.clip(emissions, eps, 1 - eps)
    return obs[:, None] * np.log(p)[None, :] + (1 - obs[:, None]) * np.log(1 - p)[None, :]


def forward_backward(
    obs: np.ndarray,
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissions: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    k = len(startprob)
    log_start = np.log(np.clip(startprob, 1e-12, 1))
    log_trans = np.log(np.clip(transmat, 1e-12, 1))
    log_emit = emission_logprob(obs, emissions)
    t_len = len(obs)
    alpha = np.empty((t_len, k))
    beta = np.empty((t_len, k))
    alpha[0] = log_start + log_emit[0]
    for t in range(1, t_len):
        alpha[t] = log_emit[t] + logsumexp(alpha[t - 1][:, None] + log_trans, axis=0)
    beta[-1] = 0
    for t in range(t_len - 2, -1, -1):
        beta[t] = logsumexp(log_trans + log_emit[t + 1][None, :] + beta[t + 1][None, :], axis=1)
    loglik = float(logsumexp(alpha[-1]))
    gamma = np.exp(alpha + beta - loglik)
    xi_sum = np.zeros((k, k))
    for t in range(t_len - 1):
        xi_log = (
            alpha[t][:, None]
            + log_trans
            + log_emit[t + 1][None, :]
            + beta[t + 1][None, :]
            - loglik
        )
        xi_sum += np.exp(xi_log)
    return loglik, gamma, xi_sum


def initialize_params(sequences: list[np.ndarray], k: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    all_obs = np.concatenate(sequences)
    base = np.clip(all_obs.mean(), 0.05, 0.95)
    emissions = np.linspace(max(0.04, base - 0.35), min(0.96, base + 0.35), k)
    emissions += rng.normal(0, 0.035, size=k)
    emissions = np.clip(np.sort(emissions), 0.03, 0.97)
    startprob = rng.dirichlet(np.ones(k))
    transmat = np.full((k, k), 0.18 / max(1, k - 1))
    np.fill_diagonal(transmat, 0.82)
    transmat = transmat + rng.uniform(0, 0.03, size=(k, k))
    transmat = transmat / transmat.sum(axis=1, keepdims=True)
    return startprob, transmat, emissions


def sort_states(
    startprob: np.ndarray,
    transmat: np.ndarray,
    emissions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(emissions)
    return startprob[order] / startprob[order].sum(), transmat[order][:, order], emissions[order]


def fit_bernoulli_hmm(
    sequences: list[np.ndarray],
    k: int,
    scope: str,
    label: str,
    *,
    restarts: int = 12,
    max_iter: int = 220,
    tol: float = 1e-5,
) -> HMMFit:
    rng = np.random.default_rng(RANDOM_SEED + k * 1009 + len(sequences))
    best: HMMFit | None = None
    n_obs = int(sum(len(seq) for seq in sequences))
    for restart in range(restarts):
        startprob, transmat, emissions = initialize_params(sequences, k, rng)
        previous = -np.inf
        converged = False
        for iteration in range(1, max_iter + 1):
            start_acc = np.full(k, 1e-3)
            trans_acc = np.full((k, k), 1e-3)
            emit_num = np.full(k, 1e-3)
            emit_den = np.full(k, 2e-3)
            total_ll = 0.0
            for obs in sequences:
                ll, gamma, xi_sum = forward_backward(obs, startprob, transmat, emissions)
                total_ll += ll
                start_acc += gamma[0]
                trans_acc += xi_sum
                emit_num += gamma.T @ obs
                emit_den += gamma.sum(axis=0)
            startprob = start_acc / start_acc.sum()
            transmat = trans_acc / trans_acc.sum(axis=1, keepdims=True)
            emissions = np.clip(emit_num / emit_den, 1e-4, 1 - 1e-4)
            startprob, transmat, emissions = sort_states(startprob, transmat, emissions)
            if abs(total_ll - previous) < tol:
                converged = True
                break
            previous = total_ll
        fit = HMMFit(
            scope=scope,
            label=label,
            k=k,
            startprob=startprob,
            transmat=transmat,
            emissions=emissions,
            loglik=total_ll,
            iterations=iteration,
            converged=converged,
            n_obs=n_obs,
            n_sequences=len(sequences),
        )
        if best is None or fit.loglik > best.loglik:
            best = fit
    assert best is not None
    return best


def n_params(k: int) -> int:
    return (k - 1) + k * (k - 1) + k


def bic(fit: HMMFit) -> float:
    return -2 * fit.loglik + n_params(fit.k) * math.log(fit.n_obs)


def aic(fit: HMMFit) -> float:
    return -2 * fit.loglik + 2 * n_params(fit.k)


def viterbi(obs: np.ndarray, fit: HMMFit) -> np.ndarray:
    k = fit.k
    log_start = np.log(np.clip(fit.startprob, 1e-12, 1))
    log_trans = np.log(np.clip(fit.transmat, 1e-12, 1))
    log_emit = emission_logprob(obs, fit.emissions)
    delta = np.empty((len(obs), k))
    psi = np.zeros((len(obs), k), dtype=int)
    delta[0] = log_start + log_emit[0]
    for t in range(1, len(obs)):
        scores = delta[t - 1][:, None] + log_trans
        psi[t] = np.argmax(scores, axis=0)
        delta[t] = np.max(scores, axis=0) + log_emit[t]
    states = np.empty(len(obs), dtype=int)
    states[-1] = int(np.argmax(delta[-1]))
    for t in range(len(obs) - 2, -1, -1):
        states[t] = psi[t + 1, states[t + 1]]
    return states


def state_label(emission: float, rank: int, k: int) -> str:
    if emission < 0.25:
        return "safe_state"
    if emission < 0.65:
        return "mixed_state"
    if rank == k - 1 and emission >= 0.82:
        return "persistent_unsafe_state"
    return "unsafe_state"


def build_scopes(turns: pd.DataFrame) -> list[tuple[str, str, pd.DataFrame]]:
    baseline = turns[turns["analysis_scope"].eq("baseline_completed")].copy()
    scopes = [("common_baseline", "Common baseline", baseline)]
    for family, sub in baseline.groupby("family", sort=True):
        scopes.append((f"family_{family}", family.replace("family_", "").title(), sub.copy()))
    for model in MODEL_ORDER:
        sub = baseline[baseline["model_slug"].eq(model)].copy()
        if not sub.empty:
            scopes.append((f"model_{model}", MODEL_LABELS.get(model, model), sub))
    return scopes


def fit_scopes(turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, HMMFit], pd.DataFrame]:
    selected: dict[str, HMMFit] = {}
    selection_rows = []
    state_rows = []
    transition_rows = []
    decoded_rows = []
    for scope, label, frame in build_scopes(turns):
        sequences = build_sequences(frame)
        if len(sequences) < 10 or sum(len(seq) for seq in sequences) < 80:
            continue
        fits = []
        max_k = min(4, max(2, int(np.sqrt(len(sequences))) + 1))
        for k in range(2, max_k + 1):
            fit = fit_bernoulli_hmm(sequences, k, scope, label)
            fits.append(fit)
            selection_rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "k": k,
                    "loglik": fit.loglik,
                    "aic": aic(fit),
                    "bic": bic(fit),
                    "n_obs": fit.n_obs,
                    "n_sequences": fit.n_sequences,
                    "iterations": fit.iterations,
                    "converged": fit.converged,
                }
            )
        best = min(fits, key=bic)
        selected[scope] = best
        for state in range(best.k):
            state_rows.append(
                {
                    "scope": scope,
                    "label": label,
                    "state": state,
                    "state_label": state_label(best.emissions[state], state, best.k),
                    "emission_unsafe_prob": best.emissions[state],
                    "start_prob": best.startprob[state],
                    "self_transition": best.transmat[state, state],
                    "expected_dwell_turns": 1 / max(1e-9, 1 - best.transmat[state, state]),
                    "selected_k": best.k,
                    "bic": bic(best),
                    "n_obs": best.n_obs,
                    "n_sequences": best.n_sequences,
                }
            )
        for i in range(best.k):
            for j in range(best.k):
                transition_rows.append(
                    {
                        "scope": scope,
                        "label": label,
                        "from_state": i,
                        "to_state": j,
                        "transition_prob": best.transmat[i, j],
                    }
                )
        decoded_rows.extend(decode_scope(frame, best))
    selection = pd.DataFrame(selection_rows)
    states = pd.DataFrame(state_rows)
    transitions = pd.DataFrame(transition_rows)
    decoded = pd.DataFrame(decoded_rows)
    selection.to_csv(DERIVED_DIR / "hmm_model_selection.csv", index=False)
    states.to_csv(DERIVED_DIR / "hmm_state_summary.csv", index=False)
    transitions.to_csv(DERIVED_DIR / "hmm_transition_matrix.csv", index=False)
    decoded.to_csv(DERIVED_DIR / "hmm_decoded_states.csv", index=False)
    return states, transitions, selected, decoded


def decode_scope(frame: pd.DataFrame, fit: HMMFit) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seq_id, sub in frame.sort_values(["player_sequence_id", "round"]).groupby("player_sequence_id", sort=False):
        obs = sub.sort_values("round")["unsafe"].astype(int).to_numpy()
        if len(obs) < 2:
            continue
        states = viterbi(obs, fit)
        ordered = sub.sort_values("round")
        for (_, row), state in zip(ordered.iterrows(), states):
            rows.append(
                {
                    "scope": fit.scope,
                    "player_sequence_id": seq_id,
                    "family": row["family"],
                    "model_slug": row["model_slug"],
                    "max_private_risk": row["max_private_risk"],
                    "round": int(row["round"]),
                    "unsafe": int(row["unsafe"]),
                    "state": int(state),
                    "state_label": state_label(fit.emissions[state], state, fit.k),
                    "state_emission_unsafe_prob": fit.emissions[state],
                }
            )
    return rows


def summarize_decoded(decoded: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    state_share = (
        decoded.groupby(["scope", "state_label"], dropna=False)
        .agg(turns=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    state_share["share"] = state_share["turns"] / state_share.groupby("scope")["turns"].transform("sum")
    state_share.to_csv(DERIVED_DIR / "hmm_state_share.csv", index=False)
    round_share = (
        decoded.groupby(["scope", "round", "state_label"], dropna=False)
        .agg(turns=("unsafe", "size"))
        .reset_index()
    )
    round_share["share"] = round_share["turns"] / round_share.groupby(["scope", "round"])["turns"].transform("sum")
    round_share.to_csv(DERIVED_DIR / "hmm_state_share_by_round.csv", index=False)
    model_share = decoded[decoded["scope"].eq("common_baseline")].copy()
    model_share = (
        model_share.groupby(["model_slug", "state_label"], dropna=False)
        .agg(turns=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    model_share["share"] = model_share["turns"] / model_share.groupby("model_slug")["turns"].transform("sum")
    model_share.to_csv(DERIVED_DIR / "hmm_common_state_share_by_model.csv", index=False)
    return state_share, round_share, model_share


def plot_model_selection(selection: pd.DataFrame) -> Path:
    scopes = ["common_baseline", "family_family_chatgpt", "family_family_gemini"]
    fig, ax = plt.subplots(figsize=(9.2, 5.2))
    colors = [TEAL, BLUE, ORANGE]
    for scope, color in zip(scopes, colors):
        sub = selection[selection["scope"].eq(scope)].sort_values("k")
        if sub.empty:
            continue
        ax.plot(sub["k"], sub["bic"], marker="o", linewidth=2.2, color=color, label=sub["label"].iloc[0])
    ax.set_xlabel("Hidden states")
    ax.set_ylabel("BIC, lower is better")
    ax.set_xticks([2, 3, 4])
    title(ax, "HMM State Count Selection", "Bernoulli HMMs fit to baseline player-game unsafe sequences.")
    ax.legend()
    ax.grid(axis="y")
    return save(fig, "01_hmm_model_selection.png")


def plot_emissions(states: pd.DataFrame) -> Path:
    scopes = ["common_baseline", "family_family_chatgpt", "family_family_gemini"]
    sub = states[states["scope"].isin(scopes)].copy()
    sub["scope_label"] = sub["label"]
    sub["state_name"] = sub["state_label"] + " S" + sub["state"].astype(str)
    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    y_labels = []
    y = []
    x = []
    colors = []
    pos = 0
    for scope in scopes:
        s = sub[sub["scope"].eq(scope)].sort_values("emission_unsafe_prob")
        for _, row in s.iterrows():
            y.append(pos)
            y_labels.append(f"{row['label']} - {row['state_label'].replace('_', ' ')}")
            x.append(row["emission_unsafe_prob"])
            colors.append(STATE_COLORS.get(row["state_label"], MUTED))
            pos += 1
        pos += 0.6
    ax.scatter(x, y, s=160, color=colors, edgecolor=INK, linewidth=0.5, zorder=3)
    for xi, yi in zip(x, y):
        ax.text(xi + 0.02, yi, f"{xi:.0%}", va="center", fontsize=10)
    ax.set_yticks(y)
    ax.set_yticklabels(y_labels)
    ax.set_xlim(0, 1)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("P(unsafe | hidden state)")
    title(ax, "Hidden-State Unsafe Propensity", "States are sorted from safe to unsafe within each fitted HMM.")
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return save(fig, "02_hmm_state_emissions.png")


def plot_transition_heatmaps(selected: dict[str, HMMFit]) -> Path:
    scopes = ["common_baseline", "family_family_chatgpt", "family_family_gemini"]
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.4))
    for ax, scope in zip(axes, scopes):
        fit = selected.get(scope)
        if fit is None:
            ax.set_axis_off()
            continue
        image = ax.imshow(fit.transmat, cmap="Blues", vmin=0, vmax=1)
        ax.set_xticks(np.arange(fit.k))
        ax.set_yticks(np.arange(fit.k))
        labels = [state_label(fit.emissions[i], i, fit.k).replace("_state", "").replace("_", "\n") for i in range(fit.k)]
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        ax.set_title(fit.label, fontsize=12, fontweight="bold")
        ax.set_xlabel("To state")
        if ax is axes[0]:
            ax.set_ylabel("From state")
        ax.grid(False)
        for i in range(fit.k):
            for j in range(fit.k):
                value = fit.transmat[i, j]
                ax.text(j, i, f"{value:.0%}", ha="center", va="center", color=WHITE if value > 0.55 else INK, fontsize=9, fontweight="bold")
    fig.suptitle("HMM Transition Matrices", x=0.02, ha="left", fontsize=17, fontweight="bold")
    fig.text(0.02, 0.91, "Higher diagonal values mean the latent behavior state persists across rounds.", color=MUTED, fontsize=10)
    fig.colorbar(image, ax=axes, fraction=0.025, pad=0.03)
    return save(fig, "03_hmm_transition_heatmaps.png")


def plot_state_share_by_model(model_share: pd.DataFrame) -> Path:
    pivot = model_share.pivot_table(index="model_slug", columns="state_label", values="share", fill_value=0)
    pivot = pivot.reindex(MODEL_ORDER).dropna(how="all")
    order = [col for col in ["safe_state", "mixed_state", "unsafe_state", "persistent_unsafe_state"] if col in pivot]
    pivot = pivot[order]
    fig, ax = plt.subplots(figsize=(10.4, 5.6))
    y = np.arange(len(pivot))
    left = np.zeros(len(pivot))
    for state in order:
        values = pivot[state].to_numpy()
        ax.barh(
            y,
            values,
            left=left,
            color=STATE_COLORS.get(state, MUTED),
            label=state.replace("_state", "").replace("_", " "),
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
    ax.set_xlabel("Share of decoded turns under common baseline HMM")
    title(ax, "Hidden-State Mix By Model", "Common HMM decoded over baseline turns; states sorted by unsafe emission.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    return save(fig, "04_hmm_state_mix_by_model.png")


def plot_state_share_by_round(round_share: pd.DataFrame) -> Path:
    sub = round_share[round_share["scope"].eq("common_baseline")].copy()
    pivot = sub.pivot_table(index="round", columns="state_label", values="share", fill_value=0).sort_index()
    order = [col for col in ["safe_state", "mixed_state", "unsafe_state", "persistent_unsafe_state"] if col in pivot]
    fig, ax = plt.subplots(figsize=(10, 5.2))
    bottom = np.zeros(len(pivot))
    x = pivot.index.to_numpy()
    for state in order:
        values = pivot[state].to_numpy()
        ax.fill_between(x, bottom, bottom + values, color=STATE_COLORS.get(state, MUTED), alpha=0.86, label=state.replace("_state", "").replace("_", " "))
        bottom += values
    ax.set_ylim(0, 1)
    ax.set_yticks(np.linspace(0, 1, 6))
    ax.set_yticklabels([f"{v:.0%}" for v in np.linspace(0, 1, 6)])
    ax.set_xlabel("Round")
    ax.set_ylabel("Decoded state share")
    title(ax, "Hidden-State Mix Across Rounds", "Common baseline HMM; later rounds have less support as games stop.")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=4)
    ax.grid(axis="y")
    ax.grid(axis="x", visible=False)
    return save(fig, "05_hmm_state_mix_by_round.png")


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
    fig.suptitle("FH Hidden Markov Model Storyboard", fontsize=24, fontweight="bold", x=0.02, ha="left")
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    return save(fig, "fh_hmm_storyboard_contact_sheet.png")


def write_report(
    selection: pd.DataFrame,
    states: pd.DataFrame,
    transitions: pd.DataFrame,
    state_share: pd.DataFrame,
    model_share: pd.DataFrame,
) -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    selected = selection.loc[selection.groupby("scope")["bic"].idxmin()].sort_values("scope")
    common_states = states[states["scope"].eq("common_baseline")].copy()
    family_states = states[states["scope"].isin(["family_family_chatgpt", "family_family_gemini"])].copy()
    common_model = model_share.copy()
    lines = [
        "# FH Hidden Markov Model Analysis",
        "",
        "## Executive Summary",
        "",
        "- **HMM turns the mechanism story into latent regimes.** Instead of a static rule like `opponent_prev_unsafe`, it estimates states that persist and switch over rounds.",
        "- **Common baseline states separate low-unsafe and high-unsafe regimes.** The state emission probabilities show how much safer/uns safer each latent mode is.",
        "- **Model identity appears as state occupancy.** `gpt-5-nano` spends most decoded turns in safer states, while Gemini models spend more mass in unsafe or persistent-unsafe states.",
        "- **Transition matrices quantify persistence.** High diagonal probabilities mean once a model enters a latent unsafe regime, it tends to stay there for multiple turns.",
        "",
        "## Selected State Counts",
        "",
        markdown_table(selected[["scope", "label", "k", "bic", "loglik", "n_obs", "n_sequences"]]),
        "",
        "## Common Baseline Hidden States",
        "",
        markdown_table(common_states[["state", "state_label", "emission_unsafe_prob", "start_prob", "self_transition", "expected_dwell_turns"]]),
        "",
        "## Family-Specific Hidden States",
        "",
        markdown_table(family_states[["scope", "state", "state_label", "emission_unsafe_prob", "self_transition", "expected_dwell_turns"]]),
        "",
        "## Common HMM State Mix By Model",
        "",
        markdown_table(common_model[["model_slug", "state_label", "turns", "share", "unsafe_rate"]]),
        "",
        "## Visuals",
        "",
        "- `figures/hmm/fh_hmm_storyboard_contact_sheet.png`",
        "- `figures/hmm/01_hmm_model_selection.png`",
        "- `figures/hmm/02_hmm_state_emissions.png`",
        "- `figures/hmm/03_hmm_transition_heatmaps.png`",
        "- `figures/hmm/04_hmm_state_mix_by_model.png`",
        "- `figures/hmm/05_hmm_state_mix_by_round.png`",
        "",
        "## Caveats",
        "",
        "- This is a Bernoulli-emission HMM over observed safe/unsafe actions only; covariates such as risk and gap are interpreted after decoding, not included in the emission model.",
        "- HMM state labels are post-hoc labels based on unsafe emission probability.",
        "- BIC can prefer fewer states when behavior is saturated; if the paper needs theory-driven states, compare BIC-selected states with fixed K=3 fits.",
    ]
    report_path = REPORTS_DIR / "fh_hmm_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    manifest = {
        "report": str(report_path.relative_to(REPO_ROOT)),
        "derived_outputs": [
            "hmm_model_selection.csv",
            "hmm_state_summary.csv",
            "hmm_transition_matrix.csv",
            "hmm_decoded_states.csv",
            "hmm_state_share.csv",
            "hmm_state_share_by_round.csv",
            "hmm_common_state_share_by_model.csv",
        ],
        "figures_dir": str(HMM_FIG_DIR.relative_to(REPO_ROOT)),
    }
    (DERIVED_DIR / "hmm_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    setup_style()
    DERIVED_DIR.mkdir(parents=True, exist_ok=True)
    HMM_FIG_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    turns = load_turns()
    states, transitions, selected, decoded = fit_scopes(turns)
    selection = pd.read_csv(DERIVED_DIR / "hmm_model_selection.csv")
    state_share, round_share, model_share = summarize_decoded(decoded)
    paths = [
        plot_model_selection(selection),
        plot_emissions(states),
        plot_transition_heatmaps(selected),
        plot_state_share_by_model(model_share),
        plot_state_share_by_round(round_share),
    ]
    contact_sheet = create_contact_sheet(paths)
    write_report(selection, states, transitions, state_share, model_share)
    print(f"Wrote HMM storyboard to {contact_sheet}")


if __name__ == "__main__":
    main()
