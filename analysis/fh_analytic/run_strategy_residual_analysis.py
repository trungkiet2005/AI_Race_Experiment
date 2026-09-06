#!/usr/bin/env python3
"""Infer gap-based rules for strategies outside AS/AU/CS/CAS.

This stage reclassifies every player trajectory against the four canonical
strategies, isolates trajectories that are not an exact single canonical match,
then fits interpretable gap formulas to those residual behaviors.
"""

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
from sklearn.linear_model import LogisticRegression


REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures"
STRAT_FIG_DIR = FIGURES_DIR / "strategy_residual"
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
MODEL_COLORS = {
    "gpt-5-nano": BLUE,
    "gpt-5.4-nano": PINK,
    "google-gemini-3-flash-preview": ORANGE,
    "google-gemini-3.1-flash-lite-preview": GOLD,
    "google-gemini-3.5-flash-lite": OLIVE,
}
FORMULA_COLORS = {
    "constant": MUTED,
    "linear_gap": BLUE,
    "quadratic_gap": TEAL,
    "abs_gap": GOLD,
    "direction_bins": ORANGE,
    "step_low_gap": PINK,
    "step_high_gap": OLIVE,
    "gap_plus_lag": "#7C4D79",
}

CANONICAL = ("AS", "AU", "CS", "CAS")


@dataclass(frozen=True)
class FitResult:
    formula: str
    n: int
    events: int
    loglik: float
    bic: float
    aic: float
    params: int
    payload: dict[str, Any]


def ensure_dirs() -> None:
    for path in [DERIVED_DIR, STRAT_FIG_DIR, REPORTS_DIR]:
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
    path = STRAT_FIG_DIR / filename
    fig.savefig(path, dpi=180, bbox_inches="tight")
    plt.close(fig)
    return path


def sigmoid(z: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-np.asarray(z)))


def clip_prob(p: np.ndarray | float) -> np.ndarray | float:
    return np.clip(p, 1e-6, 1 - 1e-6)


def loglik_from_prob(y: np.ndarray, p: np.ndarray) -> float:
    p = clip_prob(p)
    return float(np.sum(y * np.log(p) + (1 - y) * np.log(1 - p)))


def canonical_predictions(strategy: str, frame: pd.DataFrame) -> np.ndarray:
    n = len(frame)
    if strategy == "AS":
        return np.zeros(n, dtype=int)
    if strategy == "AU":
        return np.ones(n, dtype=int)
    pred = np.zeros(n, dtype=int)
    first_value = 0 if strategy == "CS" else 1
    pred[0] = first_value
    if n > 1:
        lag = frame["opponent_prev_unsafe"].iloc[1:].fillna(0).astype(int).to_numpy()
        pred[1:] = lag
    return pred


def strategy_key(row: pd.Series) -> str:
    return f"{row['source_run']}|{row['game_id']}|p{int(row['player_index'])}"


def classify_players(turns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    work = turns.copy()
    work["sequence_id"] = work.apply(strategy_key, axis=1)
    sort_cols = ["source_run", "game_id", "player_index", "round"]
    work = work.sort_values(sort_cols)

    meta_cols = [
        "source_run",
        "game_id",
        "player_index",
        "model_slug",
        "family",
        "provider",
        "persona_mode",
        "condition",
        "experiment_mode",
        "max_private_risk",
        "analysis_scope",
    ]
    for sequence_id, frame in work.groupby("sequence_id", sort=False):
        frame = frame.sort_values("round")
        y = frame["unsafe"].astype(int).to_numpy()
        mismatches: dict[str, int] = {}
        rates: dict[str, float] = {}
        for strategy in CANONICAL:
            pred = canonical_predictions(strategy, frame)
            mismatch = int(np.sum(pred != y))
            mismatches[strategy] = mismatch
            rates[strategy] = mismatch / len(y)

        minimum = min(mismatches.values())
        tied = [name for name in CANONICAL if mismatches[name] == minimum]
        first = frame.iloc[0]
        record = {col: first.get(col) for col in meta_cols}
        record.update(
            {
                "sequence_id": sequence_id,
                "n_turns": len(frame),
                "unsafe_turns": int(y.sum()),
                "unsafe_rate": float(y.mean()),
                "action_pattern": "".join("U" if value else "S" for value in y),
                "strategy_tied": "|".join(tied),
                "strategy_best": tied[0] if len(tied) == 1 else "",
                "strategy_min_mismatches": minimum,
                "strategy_min_mismatch_rate": minimum / len(y),
                "is_exact_single_canonical": bool(minimum == 0 and len(tied) == 1),
                "residual_signature": ("near_" + tied[0]) if len(tied) == 1 else ("ambiguous_" + "|".join(tied)),
            }
        )
        for strategy in CANONICAL:
            record[f"mismatch_{strategy.lower()}"] = mismatches[strategy]
            record[f"mismatch_rate_{strategy.lower()}"] = rates[strategy]
        rows.append(record)

    return pd.DataFrame(rows)


def design_matrix(frame: pd.DataFrame, formula: str) -> tuple[np.ndarray, list[str]]:
    gap = frame["progress_gap_before"].astype(float).to_numpy()
    if formula == "linear_gap":
        return gap.reshape(-1, 1), ["gap"]
    if formula == "quadratic_gap":
        return np.column_stack([gap, gap**2]), ["gap", "gap_sq"]
    if formula == "abs_gap":
        return np.abs(gap).reshape(-1, 1), ["abs_gap"]
    if formula == "direction_bins":
        return np.column_stack([(gap < -0.5).astype(float), (gap > 0.5).astype(float)]), ["behind", "ahead"]
    if formula == "gap_plus_lag":
        own_lag = frame["own_prev_unsafe"].fillna(0).astype(float).to_numpy()
        opp_lag = frame["opponent_prev_unsafe"].fillna(0).astype(float).to_numpy()
        return np.column_stack([gap, opp_lag, own_lag]), ["gap", "opp_prev_unsafe", "own_prev_unsafe"]
    raise ValueError(f"Unknown formula: {formula}")


def fit_constant(y: np.ndarray) -> FitResult:
    n = len(y)
    p = float((y.sum() + 0.5) / (n + 1.0))
    loglik = loglik_from_prob(y, np.full(n, p))
    params = 1
    return FitResult(
        formula="constant",
        n=n,
        events=int(y.sum()),
        loglik=loglik,
        bic=-2 * loglik + params * math.log(n),
        aic=-2 * loglik + 2 * params,
        params=params,
        payload={"p": p},
    )


def fit_logistic(frame: pd.DataFrame, y: np.ndarray, formula: str) -> FitResult | None:
    n = len(y)
    if len(np.unique(y)) < 2:
        return None
    x, names = design_matrix(frame, formula)
    if np.all(np.nanstd(x, axis=0) < 1e-12):
        return None
    model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=5000, random_state=RANDOM_SEED)
    model.fit(x, y)
    p = model.predict_proba(x)[:, 1]
    loglik = loglik_from_prob(y, p)
    params = x.shape[1] + 1
    payload = {
        "intercept": float(model.intercept_[0]),
        "features": names,
        "coef": [float(value) for value in model.coef_[0]],
    }
    return FitResult(
        formula=formula,
        n=n,
        events=int(y.sum()),
        loglik=loglik,
        bic=-2 * loglik + params * math.log(n),
        aic=-2 * loglik + 2 * params,
        params=params,
        payload=payload,
    )


def candidate_thresholds(gap: np.ndarray) -> list[float]:
    values = sorted(set(float(v) for v in gap if np.isfinite(v)))
    if len(values) <= 1:
        return []
    mids = [(values[i] + values[i + 1]) / 2 for i in range(len(values) - 1)]
    anchors = [-2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0]
    lo, hi = min(values), max(values)
    return sorted(set(round(v, 6) for v in mids + [a for a in anchors if lo < a < hi]))


def fit_step(frame: pd.DataFrame, y: np.ndarray, direction: str) -> FitResult | None:
    gap = frame["progress_gap_before"].astype(float).to_numpy()
    thresholds = candidate_thresholds(gap)
    if not thresholds:
        return None

    best: dict[str, Any] | None = None
    for threshold in thresholds:
        left = gap <= threshold if direction == "low" else gap >= threshold
        if left.sum() == 0 or (~left).sum() == 0:
            continue
        p_left = float((y[left].sum() + 0.5) / (left.sum() + 1.0))
        p_right = float((y[~left].sum() + 0.5) / ((~left).sum() + 1.0))
        p = np.where(left, p_left, p_right)
        loglik = loglik_from_prob(y, p)
        if best is None or loglik > best["loglik"]:
            best = {
                "threshold": float(threshold),
                "p_left": p_left,
                "p_right": p_right,
                "loglik": loglik,
                "n_left": int(left.sum()),
                "n_right": int((~left).sum()),
            }
    if best is None:
        return None
    params = 3
    formula = "step_low_gap" if direction == "low" else "step_high_gap"
    return FitResult(
        formula=formula,
        n=len(y),
        events=int(y.sum()),
        loglik=float(best["loglik"]),
        bic=-2 * float(best["loglik"]) + params * math.log(len(y)),
        aic=-2 * float(best["loglik"]) + 2 * params,
        params=params,
        payload=best,
    )


def predict_formula(formula: str, payload: dict[str, Any], frame_or_gap: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(frame_or_gap, pd.DataFrame):
        frame = frame_or_gap
        gap = frame["progress_gap_before"].astype(float).to_numpy()
    else:
        frame = None
        gap = np.asarray(frame_or_gap, dtype=float)

    if formula == "constant":
        return np.full(len(gap), float(payload["p"]))
    if formula in {"linear_gap", "quadratic_gap", "abs_gap", "direction_bins"}:
        if frame is None:
            fake = pd.DataFrame({"progress_gap_before": gap})
            x, _ = design_matrix(fake, formula)
        else:
            x, _ = design_matrix(frame, formula)
        z = float(payload["intercept"]) + x @ np.asarray(payload["coef"], dtype=float)
        return np.asarray(sigmoid(z), dtype=float)
    if formula == "gap_plus_lag":
        if frame is None:
            fake = pd.DataFrame(
                {
                    "progress_gap_before": gap,
                    "opponent_prev_unsafe": np.zeros(len(gap)),
                    "own_prev_unsafe": np.zeros(len(gap)),
                }
            )
            x, _ = design_matrix(fake, formula)
        else:
            x, _ = design_matrix(frame, formula)
        z = float(payload["intercept"]) + x @ np.asarray(payload["coef"], dtype=float)
        return np.asarray(sigmoid(z), dtype=float)
    if formula == "step_low_gap":
        mask = gap <= float(payload["threshold"])
        return np.where(mask, float(payload["p_left"]), float(payload["p_right"]))
    if formula == "step_high_gap":
        mask = gap >= float(payload["threshold"])
        return np.where(mask, float(payload["p_left"]), float(payload["p_right"]))
    raise ValueError(formula)


def formula_text(formula: str, payload: dict[str, Any]) -> str:
    if formula == "constant":
        return f"P(U) = {payload['p']:.2f}"
    if formula == "linear_gap":
        a, b = payload["intercept"], payload["coef"][0]
        return f"logit P(U) = {a:.2f} + {b:.2f}*gap"
    if formula == "quadratic_gap":
        a, b, c = payload["intercept"], payload["coef"][0], payload["coef"][1]
        return f"logit P(U) = {a:.2f} + {b:.2f}*gap + {c:.2f}*gap^2"
    if formula == "abs_gap":
        a, b = payload["intercept"], payload["coef"][0]
        return f"logit P(U) = {a:.2f} + {b:.2f}*|gap|"
    if formula == "direction_bins":
        a, b, c = payload["intercept"], payload["coef"][0], payload["coef"][1]
        return f"logit P(U) = {a:.2f} + {b:.2f}*behind + {c:.2f}*ahead"
    if formula == "gap_plus_lag":
        a = payload["intercept"]
        b, c, d = payload["coef"]
        return f"logit P(U) = {a:.2f} + {b:.2f}*gap + {c:.2f}*opp_prevU + {d:.2f}*own_prevU"
    if formula == "step_low_gap":
        sign = "<="
        return f"P(U)= {payload['p_left']:.2f} if gap {sign} {payload['threshold']:.2f}, else {payload['p_right']:.2f}"
    if formula == "step_high_gap":
        sign = ">="
        return f"P(U)= {payload['p_left']:.2f} if gap {sign} {payload['threshold']:.2f}, else {payload['p_right']:.2f}"
    return formula


def archetype(formula: str, payload: dict[str, Any]) -> str:
    if formula == "constant":
        p = payload["p"]
        if p >= 0.7:
            return "residual is mostly unconditional unsafe"
        if p <= 0.3:
            return "residual is mostly unconditional safe"
        return "mixed residual not explained by gap"
    if formula in {"linear_gap", "quadratic_gap", "abs_gap", "direction_bins", "gap_plus_lag"}:
        p_behind = float(predict_formula(formula, payload, np.array([-2.0]))[0])
        p_tied = float(predict_formula(formula, payload, np.array([0.0]))[0])
        p_ahead = float(predict_formula(formula, payload, np.array([2.0]))[0])
        if formula == "abs_gap":
            return "unsafe rises when the race is uneven" if p_behind > p_tied + 0.08 else "gap magnitude is weak"
        if p_ahead - p_behind > 0.12:
            return "leader-pressure rule: unsafe rises when ahead"
        if p_behind - p_ahead > 0.12:
            return "catch-up rule: unsafe rises when behind"
        if max(p_behind, p_ahead) - p_tied > 0.12:
            return "polarized rule: unsafe at larger absolute gaps"
        return "gap effect is present but shallow"
    if formula == "step_low_gap":
        high_side = "behind/low-gap side" if payload["p_left"] >= payload["p_right"] else "above-threshold side"
        return f"threshold rule concentrated on {high_side}"
    if formula == "step_high_gap":
        high_side = "ahead/high-gap side" if payload["p_left"] >= payload["p_right"] else "below-threshold side"
        return f"threshold rule concentrated on {high_side}"
    return ""


def fit_group(frame: pd.DataFrame) -> list[FitResult]:
    y = frame["unsafe"].astype(int).to_numpy()
    results: list[FitResult] = [fit_constant(y)]
    for formula in ["linear_gap", "quadratic_gap", "abs_gap", "direction_bins", "gap_plus_lag"]:
        fit = fit_logistic(frame, y, formula)
        if fit is not None:
            results.append(fit)
    for direction in ["low", "high"]:
        fit = fit_step(frame, y, direction)
        if fit is not None:
            results.append(fit)
    return sorted(results, key=lambda item: item.bic)


def build_formula_tables(noncanonical_turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    best_rows: list[dict[str, Any]] = []

    group_specs: list[tuple[str, list[str], int]] = [
        ("signature", ["residual_signature"], 80),
        ("signature_model", ["residual_signature", "model_slug"], 35),
        ("signature_family", ["residual_signature", "family"], 35),
        ("model", ["model_slug"], 50),
        ("family", ["family"], 50),
        ("overall", [], 80),
    ]
    for scope, keys, min_n in group_specs:
        iterator = [((), noncanonical_turns)] if not keys else noncanonical_turns.groupby(keys, dropna=False)
        for values, frame in iterator:
            if len(frame) < min_n:
                continue
            if not isinstance(values, tuple):
                values = (values,)
            dims = dict(zip(keys, values))
            fits = fit_group(frame)
            best = fits[0]
            second_bic = fits[1].bic if len(fits) > 1 else np.nan
            gap_only_fits = [fit for fit in fits if fit.formula != "gap_plus_lag"]
            gap_only_best = gap_only_fits[0] if gap_only_fits else best
            for fit in fits:
                rows.append(
                    {
                        "scope": scope,
                        **dims,
                        "n": fit.n,
                        "events": fit.events,
                        "unsafe_rate": fit.events / fit.n,
                        "formula": fit.formula,
                        "params": fit.params,
                        "loglik": fit.loglik,
                        "bic": fit.bic,
                        "aic": fit.aic,
                        "payload_json": json.dumps(fit.payload, sort_keys=True),
                        "formula_text": formula_text(fit.formula, fit.payload),
                        "archetype": archetype(fit.formula, fit.payload),
                    }
                )
            p_neg2 = float(predict_formula(best.formula, best.payload, np.array([-2.0]))[0])
            p_zero = float(predict_formula(best.formula, best.payload, np.array([0.0]))[0])
            p_pos2 = float(predict_formula(best.formula, best.payload, np.array([2.0]))[0])
            gap_p_neg2 = float(predict_formula(gap_only_best.formula, gap_only_best.payload, np.array([-2.0]))[0])
            gap_p_zero = float(predict_formula(gap_only_best.formula, gap_only_best.payload, np.array([0.0]))[0])
            gap_p_pos2 = float(predict_formula(gap_only_best.formula, gap_only_best.payload, np.array([2.0]))[0])
            best_rows.append(
                {
                    "scope": scope,
                    **dims,
                    "n": best.n,
                    "events": best.events,
                    "unsafe_rate": best.events / best.n,
                    "best_formula": best.formula,
                    "gap_only_formula": gap_only_best.formula,
                    "best_bic": best.bic,
                    "gap_only_bic": gap_only_best.bic,
                    "lag_bic_lift": gap_only_best.bic - best.bic,
                    "second_bic": second_bic,
                    "bic_margin": second_bic - best.bic if np.isfinite(second_bic) else np.nan,
                    "formula_text": formula_text(best.formula, best.payload),
                    "gap_only_formula_text": formula_text(gap_only_best.formula, gap_only_best.payload),
                    "archetype": archetype(best.formula, best.payload),
                    "gap_only_archetype": archetype(gap_only_best.formula, gap_only_best.payload),
                    "p_gap_neg2": p_neg2,
                    "p_gap_0": p_zero,
                    "p_gap_pos2": p_pos2,
                    "effect_pos_minus_neg": p_pos2 - p_neg2,
                    "gap_only_p_gap_neg2": gap_p_neg2,
                    "gap_only_p_gap_0": gap_p_zero,
                    "gap_only_p_gap_pos2": gap_p_pos2,
                    "gap_only_effect_pos_minus_neg": gap_p_pos2 - gap_p_neg2,
                    "payload_json": json.dumps(best.payload, sort_keys=True),
                    "gap_only_payload_json": json.dumps(gap_only_best.payload, sort_keys=True),
                }
            )

    bins = [-np.inf, -2, -1, -0.5, 0.5, 1, 2, np.inf]
    labels = ["<=-2", "-2..-1", "-1..-0.5", "-0.5..0.5", "0.5..1", "1..2", ">=2"]
    binned = noncanonical_turns.copy()
    binned["gap_bucket"] = pd.cut(binned["progress_gap_before"], bins=bins, labels=labels, include_lowest=True)
    bin_summary = (
        binned.groupby(["residual_signature", "model_slug", "gap_bucket"], observed=True)
        .agg(n=("unsafe", "size"), unsafe_rate=("unsafe", "mean"))
        .reset_index()
    )
    return pd.DataFrame(rows), pd.DataFrame(best_rows), bin_summary


def format_pct(value: float) -> str:
    if pd.isna(value):
        return "n/a"
    return f"{100 * value:.1f}%"


def plot_noncanonical_mix(players: pd.DataFrame) -> Path:
    non = players[~players["is_exact_single_canonical"]].copy()
    counts = (
        non.groupby(["model_slug", "residual_signature"])
        .size()
        .rename("n")
        .reset_index()
    )
    totals = counts.groupby("model_slug")["n"].transform("sum")
    counts["share"] = counts["n"] / totals
    top_signatures = (
        counts.groupby("residual_signature")["n"].sum().sort_values(ascending=False).head(7).index.tolist()
    )
    counts["signature_plot"] = np.where(counts["residual_signature"].isin(top_signatures), counts["residual_signature"], "other")
    plot = (
        counts.groupby(["model_slug", "signature_plot"])["share"]
        .sum()
        .reset_index()
        .pivot(index="model_slug", columns="signature_plot", values="share")
        .fillna(0)
    )
    order = [m for m in MODEL_ORDER if m in plot.index] + [m for m in plot.index if m not in MODEL_ORDER]
    plot = plot.loc[order]
    sig_order = [sig for sig in top_signatures if sig in plot.columns] + (["other"] if "other" in plot.columns else [])
    colors = [BLUE, ORANGE, GOLD, OLIVE, PINK, TEAL, "#8E7CC3", MUTED]

    fig, ax = plt.subplots(figsize=(12.5, 7.2))
    left = np.zeros(len(plot))
    y = np.arange(len(plot))
    for idx, sig in enumerate(sig_order):
        vals = plot[sig].to_numpy()
        ax.barh(y, vals, left=left, color=colors[idx % len(colors)], edgecolor=WHITE, linewidth=0.8, label=sig)
        for i, value in enumerate(vals):
            if value >= 0.09:
                ax.text(left[i] + value / 2, i, f"{value:.0%}", ha="center", va="center", color=WHITE, fontsize=8)
        left += vals
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in plot.index])
    ax.set_xlim(0, 1)
    ax.set_xlabel("Share of non-exact-canonical player trajectories")
    ax.set_title("Noncanonical Strategy Signatures by Model")
    ax.text(
        0,
        1.04,
        "A trajectory is noncanonical if it does not match one exact AS/AU/CS/CAS rule across all turns.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    ax.legend(ncol=2, loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8)
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.22, right=0.98, top=0.86, bottom=0.24)
    return savefig(fig, "01_noncanonical_strategy_mix.png")


def plot_formula_winners(best: pd.DataFrame) -> Path:
    scope = best[best["scope"].eq("signature_model")].copy()
    if scope.empty:
        scope = best[best["scope"].eq("signature")].copy()
        scope["model_slug"] = "overall"
    scope = scope.sort_values(["n"], ascending=False).head(30)
    scope["label"] = scope.apply(
        lambda r: f"{MODEL_LABELS.get(r.get('model_slug', 'overall'), r.get('model_slug', 'overall'))}\n{r.get('residual_signature', '')}",
        axis=1,
    )
    formulas = ["constant", "linear_gap", "quadratic_gap", "abs_gap", "direction_bins", "step_low_gap", "step_high_gap"]
    matrix = pd.DataFrame(0.0, index=scope["label"], columns=formulas)
    for _, row in scope.iterrows():
        label = row["label"]
        matrix.loc[label, row["gap_only_formula"]] = abs(float(row["gap_only_effect_pos_minus_neg"])) * 100

    fig, ax = plt.subplots(figsize=(13, max(7, len(matrix) * 0.34)))
    im = ax.imshow(matrix.to_numpy(), aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(5, matrix.to_numpy().max()))
    ax.set_yticks(np.arange(len(matrix)))
    ax.set_yticklabels(matrix.index, fontsize=8)
    ax.set_xticks(np.arange(len(formulas)))
    ax.set_xticklabels([f.replace("_", "\n") for f in formulas], fontsize=8)
    ax.set_title("Best Pure-Gap Formula by Noncanonical Signature and Model")
    ax.text(
        0,
        1.03,
        "Darker cells mean the selected pure-gap formula changes predicted unsafe probability more strongly from gap -2 to +2.",
        transform=ax.transAxes,
        color=MUTED,
        fontsize=10,
    )
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix.iloc[i, j] > 0:
                ax.text(j, i, f"{matrix.iloc[i, j]:.0f}pp", ha="center", va="center", fontsize=7, color=INK)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label("|P(U at +2) - P(U at -2)|, percentage points")
    fig.subplots_adjust(left=0.36, right=0.94, top=0.88, bottom=0.16)
    return savefig(fig, "02_formula_winner_heatmap.png")


def plot_gap_curves(noncanonical_turns: pd.DataFrame, best: pd.DataFrame) -> Path:
    candidates = best[best["scope"].eq("signature_model")].copy()
    candidates["gap_abs_effect"] = candidates["gap_only_effect_pos_minus_neg"].abs()
    candidates = candidates[candidates["n"].ge(35)].sort_values(["gap_abs_effect", "n"], ascending=[False, False]).head(6)
    if candidates.empty:
        candidates = best[best["scope"].eq("signature")].copy()
        candidates["gap_abs_effect"] = candidates["gap_only_effect_pos_minus_neg"].abs()
        candidates = candidates.sort_values(["gap_abs_effect", "n"], ascending=[False, False]).head(6)

    n_panels = max(1, len(candidates))
    ncols = 3 if n_panels > 2 else n_panels
    nrows = int(math.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5.2 * ncols, 4.1 * nrows), squeeze=False)
    grid = np.linspace(-3.0, 3.0, 121)

    for ax, (_, row) in zip(axes.ravel(), candidates.iterrows()):
        frame = noncanonical_turns[noncanonical_turns["residual_signature"].eq(row.get("residual_signature"))].copy()
        if row["scope"] == "signature_model":
            frame = frame[frame["model_slug"].eq(row["model_slug"])].copy()
        payload = json.loads(row["gap_only_payload_json"])
        formula = row["gap_only_formula"]
        yhat = predict_formula(formula, payload, grid)

        bins = np.linspace(-3, 3, 13)
        frame["gap_bin_mid"] = pd.cut(frame["progress_gap_before"].clip(-3, 3), bins=bins, include_lowest=True)
        obs = (
            frame.groupby("gap_bin_mid", observed=True)
            .agg(n=("unsafe", "size"), unsafe_rate=("unsafe", "mean"), gap=("progress_gap_before", "mean"))
            .reset_index()
        )
        obs = obs[obs["n"].ge(3)]
        ax.plot(grid, yhat, color=FORMULA_COLORS.get(formula, BLUE), linewidth=2.4, label=formula.replace("_", " "))
        ax.scatter(
            obs["gap"],
            obs["unsafe_rate"],
            s=np.clip(obs["n"], 16, 140),
            color=INK,
            alpha=0.72,
            edgecolor=WHITE,
            linewidth=0.8,
            label="observed bins",
        )
        ax.axvline(0, color=GRID, linewidth=1.2)
        ax.set_ylim(-0.03, 1.03)
        ax.set_xlim(-3.05, 3.05)
        ax.set_xlabel("progress gap before turn")
        ax.set_ylabel("P(unsafe)")
        title = f"{row.get('residual_signature', '')}"
        if row["scope"] == "signature_model":
            title = f"{MODEL_LABELS.get(row['model_slug'], row['model_slug'])}: {title}"
        ax.set_title(textwrap.fill(title, 32), fontsize=11)
        ax.text(
            0.02,
            0.05,
            textwrap.fill(row["gap_only_formula_text"], 45),
            transform=ax.transAxes,
            fontsize=8,
            color=MUTED,
            bbox=dict(facecolor=WHITE, edgecolor=GRID, boxstyle="round,pad=0.3"),
        )
        ax.legend(loc="upper left", fontsize=8)

    for ax in axes.ravel()[len(candidates) :]:
        ax.axis("off")
    fig.suptitle("Best-Fit Pure-Gap Rules for High-Signal Noncanonical Groups", fontsize=16, y=0.99)
    fig.subplots_adjust(top=0.88, hspace=0.42, wspace=0.28)
    return savefig(fig, "03_best_gap_formula_curves.png")


def plot_archetype_effects(best: pd.DataFrame) -> Path:
    scope = best[best["scope"].eq("signature_model")].copy()
    scope = scope[scope["n"].ge(35)]
    if scope.empty:
        scope = best[best["scope"].eq("signature")].copy()
    scope = scope.assign(abs_effect=scope["gap_only_effect_pos_minus_neg"].abs())
    scope = scope.sort_values(["abs_effect", "n"], ascending=[False, False]).head(18)
    scope["label"] = scope.apply(
        lambda r: (
            f"{MODEL_LABELS.get(r.get('model_slug', 'overall'), r.get('model_slug', 'overall'))} | "
            f"{r.get('residual_signature', '')}"
        ),
        axis=1,
    )
    y = np.arange(len(scope))
    colors = [ORANGE if v > 0 else BLUE for v in scope["effect_pos_minus_neg"]]
    colors = [ORANGE if v > 0 else BLUE for v in scope["gap_only_effect_pos_minus_neg"]]
    fig, ax = plt.subplots(figsize=(13, max(7, len(scope) * 0.34)))
    ax.axvline(0, color=INK, linewidth=1)
    ax.hlines(y, 0, scope["gap_only_effect_pos_minus_neg"], color=GRID, linewidth=2)
    ax.scatter(scope["gap_only_effect_pos_minus_neg"], y, s=90, color=colors, edgecolor=WHITE, linewidth=0.8, zorder=3)
    for yi, (_, row) in enumerate(scope.iterrows()):
        x = row["gap_only_effect_pos_minus_neg"]
        ha = "left" if x >= 0 else "right"
        ax.text(x + (0.015 if x >= 0 else -0.015), yi, row["gap_only_formula"].replace("_", " "), va="center", ha=ha, fontsize=8, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(scope["label"], fontsize=8)
    ax.set_xlabel("Predicted P(unsafe at gap +2) - P(unsafe at gap -2)")
    ax.set_title("Pure-Gap Direction Effect in Noncanonical Strategies")
    ax.text(
        0,
        1.04,
        "Positive means more unsafe when ahead; negative means more unsafe when behind.",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
    )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.42, right=0.97, top=0.88, bottom=0.12)
    return savefig(fig, "04_gap_rule_archetypes.png")


def plot_noncanonical_rates(players: pd.DataFrame) -> Path:
    summary = (
        players.assign(noncanonical=~players["is_exact_single_canonical"])
        .groupby("model_slug")
        .agg(player_sequences=("sequence_id", "size"), noncanonical_share=("noncanonical", "mean"), unsafe_rate=("unsafe_rate", "mean"))
        .reset_index()
    )
    order = [m for m in MODEL_ORDER if m in set(summary["model_slug"])]
    summary["order"] = summary["model_slug"].map({m: i for i, m in enumerate(order)}).fillna(999)
    summary = summary.sort_values("order")
    y = np.arange(len(summary))

    fig, ax = plt.subplots(figsize=(11.5, 6.4))
    bars = ax.barh(
        y,
        summary["noncanonical_share"],
        color=[MODEL_COLORS.get(m, BLUE) for m in summary["model_slug"]],
        edgecolor=WHITE,
        linewidth=0.9,
    )
    ax.set_yticks(y)
    ax.set_yticklabels([MODEL_LABELS.get(m, m) for m in summary["model_slug"]])
    ax.set_xlim(0, min(1.0, max(0.25, summary["noncanonical_share"].max() * 1.22)))
    ax.set_xlabel("Share of player trajectories")
    ax.set_title("How Much Behavior Falls Outside Exact AS/AU/CS/CAS")
    ax.text(
        0,
        1.04,
        "Each bar counts player-level trajectories, not individual turns.",
        transform=ax.transAxes,
        fontsize=10,
        color=MUTED,
    )
    for bar, (_, row) in zip(bars, summary.iterrows()):
        ax.text(
            bar.get_width() + 0.01,
            bar.get_y() + bar.get_height() / 2,
            f"{row['noncanonical_share']:.1%}  n={int(row['player_sequences'])}",
            va="center",
            fontsize=9,
            color=INK,
        )
    ax.grid(axis="x")
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(left=0.25, right=0.93, top=0.86, bottom=0.14)
    return savefig(fig, "05_noncanonical_rate_by_model.png")


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
    fig.suptitle("Strategy Residual Gap-Rule Storyboard", fontsize=22, y=0.995)
    fig.subplots_adjust(top=0.965, hspace=0.08, wspace=0.04)
    path = STRAT_FIG_DIR / "fh_strategy_residual_storyboard_contact_sheet.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def write_report(
    players: pd.DataFrame,
    noncanonical_turns: pd.DataFrame,
    fits: pd.DataFrame,
    best: pd.DataFrame,
    bin_summary: pd.DataFrame,
    figure_paths: list[Path],
    contact_sheet: Path,
) -> Path:
    exact_share = players["is_exact_single_canonical"].mean()
    non_share = 1 - exact_share
    overall = best[best["scope"].eq("overall")].iloc[0]
    sig_best = best[best["scope"].eq("signature")].sort_values("n", ascending=False)
    model_best = best[best["scope"].eq("signature_model")].sort_values(["bic_margin", "n"], ascending=[False, False])

    top_signature_lines = []
    for _, row in sig_best.head(10).iterrows():
        top_signature_lines.append(
            f"- `{row['residual_signature']}`: n={int(row['n'])} turns, unsafe={format_pct(row['unsafe_rate'])}; "
            f"gap-only `{row['gap_only_formula']}` -> `{row['gap_only_formula_text']}` ({row['gap_only_archetype']}); "
            f"extended `{row['best_formula']}` -> `{row['formula_text']}`."
        )

    top_model_lines = []
    for _, row in model_best.head(12).iterrows():
        top_model_lines.append(
            f"- {MODEL_LABELS.get(row['model_slug'], row['model_slug'])} / `{row['residual_signature']}`: "
            f"n={int(row['n'])}, unsafe={format_pct(row['unsafe_rate'])}; "
            f"gap-only `{row['gap_only_formula']}` -> `{row['gap_only_formula_text']}`; "
            f"extended `{row['best_formula']}` improves BIC by {row['lag_bic_lift']:.1f}."
        )

    canonical_counts = (
        players.assign(noncanonical=~players["is_exact_single_canonical"])
        .groupby("model_slug")
        .agg(sequences=("sequence_id", "size"), noncanonical_share=("noncanonical", "mean"))
        .reset_index()
        .sort_values("noncanonical_share", ascending=False)
    )
    model_lines = [
        f"- {MODEL_LABELS.get(row['model_slug'], row['model_slug'])}: {format_pct(row['noncanonical_share'])} noncanonical over {int(row['sequences'])} player trajectories."
        for _, row in canonical_counts.iterrows()
    ]

    body = f"""# Strategy Residual Gap Rules

## Scope

This analysis reclassifies each player trajectory against AS, AU, CS, and CAS. A trajectory is treated as residual/noncanonical if it does not exactly match one single canonical strategy across all observed turns.

- Player trajectories analyzed: {len(players):,}
- Exact single-canonical share: {format_pct(exact_share)}
- Residual/noncanonical share: {format_pct(non_share)}
- Residual round-2+ turns used for gap formulas: {len(noncanonical_turns):,}

## Overall Fit

For the pooled residual behavior, the best pure-gap formula is `{overall['gap_only_formula']}`:

`{overall['gap_only_formula_text']}`

Gap-only interpretation: {overall['gap_only_archetype']}. Gap-only predicted unsafe changes from {format_pct(overall['gap_only_p_gap_neg2'])} at gap=-2 to {format_pct(overall['gap_only_p_gap_pos2'])} at gap=+2.

If lag/memory is allowed, the best extended formula is `{overall['best_formula']}`:

`{overall['formula_text']}`

Extended interpretation: {overall['archetype']}. Predicted unsafe changes from {format_pct(overall['p_gap_neg2'])} at gap=-2 to {format_pct(overall['p_gap_pos2'])} at gap=+2, and improves BIC by {overall['lag_bic_lift']:.1f} versus the best pure-gap formula.

## Noncanonical Coverage by Model

{chr(10).join(model_lines)}

## Top Residual Signatures

{chr(10).join(top_signature_lines)}

## Strong Model-Specific Rules

{chr(10).join(top_model_lines)}

## Deliverables

- Contact sheet: `{contact_sheet}`
- Figures: {", ".join(f"`{path.name}`" for path in figure_paths)}
- Tables: `strategy_residual_player_classification.csv`, `strategy_residual_turns.csv`, `strategy_gap_formula_fits.csv`, `strategy_gap_best_formulas.csv`, `strategy_gap_bin_summary.csv`

## Caveats

Gap means `own_progress_before - opponent_progress_before`; positive values mean the player is ahead before choosing the current action. First turns are excluded from formula fitting because their gap is mechanically zero and lag fields are absent.
"""
    path = REPORTS_DIR / "fh_strategy_residual_gap_rules.md"
    path.write_text(body, encoding="utf-8")
    return path


def main() -> None:
    ensure_dirs()
    configure_matplotlib()
    turns_path = DERIVED_DIR / "turns_canonical.csv"
    if not turns_path.exists():
        raise FileNotFoundError(f"Missing canonical turns table: {turns_path}")

    turns = pd.read_csv(turns_path, low_memory=False)
    turns = turns[turns["manifest_status"].eq("completed") & ~turns["duplicate_grain_key"].fillna(False).astype(bool)].copy()
    turns = turns[turns["unsafe"].notna()].copy()
    for col in ["unsafe", "round", "player_index", "progress_gap_before", "own_prev_unsafe", "opponent_prev_unsafe"]:
        if col in turns.columns:
            turns[col] = pd.to_numeric(turns[col], errors="coerce")

    players = classify_players(turns)
    sequence_cols = [
        "sequence_id",
        "strategy_tied",
        "strategy_best",
        "strategy_min_mismatches",
        "strategy_min_mismatch_rate",
        "is_exact_single_canonical",
        "residual_signature",
        "action_pattern",
    ]
    turns_with_strategy = turns.copy()
    turns_with_strategy["sequence_id"] = turns_with_strategy.apply(strategy_key, axis=1)
    turns_with_strategy = turns_with_strategy.merge(players[sequence_cols], on="sequence_id", how="left")

    noncanonical_turns = turns_with_strategy[
        ~turns_with_strategy["is_exact_single_canonical"].fillna(False)
        & turns_with_strategy["is_round2plus"].fillna(False).astype(bool)
        & turns_with_strategy["progress_gap_before"].notna()
    ].copy()

    fits, best, bin_summary = build_formula_tables(noncanonical_turns)

    players.to_csv(DERIVED_DIR / "strategy_residual_player_classification.csv", index=False)
    noncanonical_turns.to_csv(DERIVED_DIR / "strategy_residual_turns.csv", index=False)
    fits.to_csv(DERIVED_DIR / "strategy_gap_formula_fits.csv", index=False)
    best.to_csv(DERIVED_DIR / "strategy_gap_best_formulas.csv", index=False)
    bin_summary.to_csv(DERIVED_DIR / "strategy_gap_bin_summary.csv", index=False)

    figure_paths = [
        plot_noncanonical_rates(players),
        plot_noncanonical_mix(players),
        plot_formula_winners(best),
        plot_gap_curves(noncanonical_turns, best),
        plot_archetype_effects(best),
    ]
    contact_sheet = make_contact_sheet(figure_paths)
    report = write_report(players, noncanonical_turns, fits, best, bin_summary, figure_paths, contact_sheet)

    exact_share = players["is_exact_single_canonical"].mean()
    overall = best[best["scope"].eq("overall")].iloc[0]
    print(
        json.dumps(
            {
                "players": int(len(players)),
                "exact_single_canonical_share": float(exact_share),
                "noncanonical_turns_round2plus": int(len(noncanonical_turns)),
                "overall_best_formula": overall["best_formula"],
                "overall_formula_text": overall["formula_text"],
                "overall_archetype": overall["archetype"],
                "report": str(report),
                "contact_sheet": str(contact_sheet),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
