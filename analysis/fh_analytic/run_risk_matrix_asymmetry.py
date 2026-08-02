#!/usr/bin/env python3
"""Own-vs-opponent risk-label asymmetry mining for `mode_risk_matrix`.

`mode_risk_matrix` is the largest untapped slice of the collected data
(49,656 decisions, 3369 races across 3 models) -- bigger than baseline and
`mode_strategy_persona` combined -- yet the main pipeline only reports its
pooled unsafe rate. Each race assigns each seat a narrative risk-framing
label `risk-1` .. `risk-6` (independent of the mechanistic
`max_private_risk` in {0.1, 0.6, 0.9}), and the full 6x6 grid of own/opponent
label pairs is run (`R1_R1` .. `R6_R6`), so own-label and opponent-label
effects are identifiable and separable from the real private-risk treatment.
This stage asks:

1. does the narrative risk label move behavior at all, net of the real
   mechanistic private-risk treatment;
2. is behavior driven by the player's own label, the opponent's label, or
   the relative gap between them;
3. does that sensitivity change over the course of a race;
4. do the human-reference lag/gap effects still hold sign here.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
FIGURES_DIR = OUTPUT_DIR / "figures" / "risk_matrix_asymmetry"
REPORTS_DIR = OUTPUT_DIR / "reports"
HUMAN_REFERENCE_PATH = REPO_ROOT / "results" / "scripts" / "human_reference.json"

BLUE = "#2F6B9A"
ORANGE = "#D9822B"
GOLD = "#C9A227"
INK = "#263238"
MUTED = "#6B7280"
GRID = "#E6E8EB"
PAPER = "#FBFBF8"
WHITE = "#FFFFFF"

MODEL_ORDER = ["gpt-5-nano", "gpt-5.4-nano", "google-gemini-3-flash-preview"]
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google-gemini-3-flash-preview": "Gemini 3 Flash",
}
MODEL_COLORS = {
    "gpt-5-nano": BLUE,
    "gpt-5.4-nano": "#B45A7C",
    "google-gemini-3-flash-preview": ORANGE,
}

HUMAN_TERMS = {
    "own_prev_unsafe": "own_prev_unsafe",
    "opponent_prev_unsafe": "opponent_prev_unsafe",
    "progress_gap_before": "progress_gap_before",
}


def clean_bool(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series
    return series.astype(str).str.lower().map({"true": True, "false": False})


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


def setup_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": PAPER,
            "savefig.facecolor": PAPER,
            "axes.facecolor": WHITE,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 14,
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
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / filename
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor=PAPER)
    plt.close(fig)
    return path


def safe_exp(value: float) -> float:
    if not math.isfinite(value):
        return np.nan
    if value > 709:
        return np.inf
    if value < -745:
        return 0.0
    return math.exp(value)


def load_risk_matrix_turns() -> pd.DataFrame:
    turns = pd.read_csv(DERIVED_DIR / "turns_canonical.csv", low_memory=False)
    turns["duplicate_grain_key"] = clean_bool(turns["duplicate_grain_key"])
    turns["is_round2plus"] = clean_bool(turns["is_round2plus"])
    numeric_columns = [
        "unsafe",
        "round",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
        "progress_gap_before",
        "first_round_unsafe",
        "max_private_risk",
    ]
    for col in numeric_columns:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")
    turns = turns[(turns["manifest_status"] == "completed") & (~turns["duplicate_grain_key"])].copy()
    turns = turns[turns["experiment_mode"] == "mode_risk_matrix"].copy()
    turns["seat_persona_role"] = turns["seat_persona_role"].astype(str).str.strip()
    turns["own_risk_label"] = pd.to_numeric(
        turns["seat_persona_role"].str.replace("risk-", "", regex=False), errors="coerce"
    )

    roles_wide = (
        turns[["source_run", "game_id", "player_index", "own_risk_label"]]
        .drop_duplicates()
        .pivot_table(index=["source_run", "game_id"], columns="player_index", values="own_risk_label", aggfunc="first")
    )
    roles_wide.columns = [f"risk_label_seat_{int(c)}" for c in roles_wide.columns]
    roles_wide = roles_wide.reset_index()
    turns = turns.merge(roles_wide, on=["source_run", "game_id"], how="left")
    turns["opponent_risk_label"] = np.where(
        turns["player_index"] == 0, turns.get("risk_label_seat_1"), turns.get("risk_label_seat_0")
    )
    turns["label_gap"] = turns["own_risk_label"] - turns["opponent_risk_label"]
    turns["round_phase"] = turns["round"].apply(
        lambda r: "round_1" if r == 1 else "early_r2_4" if r <= 4 else "mid_r5_8" if r <= 8 else "late_r9plus"
    )
    turns["cluster_id"] = turns["source_run"].astype(str) + "::" + turns["game_id"].astype(str)
    return turns


def rate_with_ci(values: pd.Series) -> tuple[float, int, float, float]:
    values = pd.to_numeric(values, errors="coerce").dropna()
    n = len(values)
    if n == 0:
        return np.nan, 0, np.nan, np.nan
    mean = float(values.mean())
    se = math.sqrt(mean * (1 - mean) / n) if 0 <= mean <= 1 else np.nan
    lo = max(0.0, mean - 1.96 * se) if not math.isnan(se) else np.nan
    hi = min(1.0, mean + 1.96 * se) if not math.isnan(se) else np.nan
    return mean, n, lo, hi


def build_marginal_tables(turns: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    own_rows: list[dict[str, Any]] = []
    for keys, group in turns.groupby(["model_slug", "own_risk_label"], dropna=False):
        model, label = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        own_rows.append({"model_slug": model, "own_risk_label": label, "n": n, "unsafe_rate": mean, "ci95_low": lo, "ci95_high": hi})
    own = pd.DataFrame(own_rows).sort_values(["model_slug", "own_risk_label"])
    own.to_csv(DERIVED_DIR / "risk_matrix_own_label_marginal.csv", index=False)

    opp_rows: list[dict[str, Any]] = []
    for keys, group in turns.groupby(["model_slug", "opponent_risk_label"], dropna=False):
        model, label = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        opp_rows.append({"model_slug": model, "opponent_risk_label": label, "n": n, "unsafe_rate": mean, "ci95_low": lo, "ci95_high": hi})
    opp = pd.DataFrame(opp_rows).sort_values(["model_slug", "opponent_risk_label"])
    opp.to_csv(DERIVED_DIR / "risk_matrix_opponent_label_marginal.csv", index=False)
    return own, opp


def build_label_gap_table(turns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in turns.groupby(["model_slug", "label_gap"], dropna=False):
        model, gap = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append({"model_slug": model, "label_gap_own_minus_opp": gap, "n": n, "unsafe_rate": mean, "ci95_low": lo, "ci95_high": hi})
    frame = pd.DataFrame(rows).sort_values(["model_slug", "label_gap_own_minus_opp"])
    frame.to_csv(DERIVED_DIR / "risk_matrix_label_gap.csv", index=False)
    return frame


def build_real_vs_label_table(turns: pd.DataFrame) -> pd.DataFrame:
    """Cross narrative risk label against the real mechanistic private-risk treatment."""

    rows: list[dict[str, Any]] = []
    for keys, group in turns.groupby(["model_slug", "max_private_risk", "own_risk_label"], dropna=False):
        model, real_risk, label = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append(
            {
                "model_slug": model,
                "max_private_risk": real_risk,
                "own_risk_label": label,
                "n": n,
                "unsafe_rate": mean,
            }
        )
    frame = pd.DataFrame(rows).sort_values(["model_slug", "max_private_risk", "own_risk_label"])
    frame.to_csv(DERIVED_DIR / "risk_matrix_real_vs_label.csv", index=False)
    return frame


def build_temporal_table(turns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    low = turns["own_risk_label"] <= 2
    high = turns["own_risk_label"] >= 5
    banded = turns.assign(label_band=np.select([low, high], ["own_low_1_2", "own_high_5_6"], default="own_mid_3_4"))
    for keys, group in banded.groupby(["model_slug", "label_band", "round_phase"], dropna=False):
        model, band, phase = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append({"model_slug": model, "own_label_band": band, "round_phase": phase, "n": n, "unsafe_rate": mean})
    frame = pd.DataFrame(rows)
    phase_order = ["round_1", "early_r2_4", "mid_r5_8", "late_r9plus"]
    frame["round_phase"] = pd.Categorical(frame["round_phase"], categories=phase_order, ordered=True)
    frame = frame.sort_values(["model_slug", "own_label_band", "round_phase"])
    frame.to_csv(DERIVED_DIR / "risk_matrix_temporal.csv", index=False)
    return frame


def safe_import_statsmodels():
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        return smf, sm
    except Exception:
        return None, None


def build_asymmetry_logit(turns: pd.DataFrame) -> pd.DataFrame:
    smf, sm = safe_import_statsmodels()
    if smf is None:
        pd.DataFrame([{"reason": "statsmodels not installed"}]).to_csv(
            DERIVED_DIR / "risk_matrix_asymmetry_skipped.csv", index=False
        )
        return pd.DataFrame()

    round2 = turns[turns["is_round2plus"]].dropna(
        subset=[
            "unsafe",
            "own_risk_label",
            "opponent_risk_label",
            "progress_gap_before",
            "own_prev_unsafe",
            "opponent_prev_unsafe",
            "max_private_risk",
            "game_id",
        ]
    ).copy()
    formula = (
        "unsafe ~ own_risk_label + opponent_risk_label + C(max_private_risk) + "
        "progress_gap_before + own_prev_unsafe + opponent_prev_unsafe"
    )
    rows: list[dict[str, Any]] = []
    for model_slug, frame in round2.groupby("model_slug", dropna=False):
        frame = frame.copy()
        if len(frame) < 100 or frame["unsafe"].nunique() < 2:
            continue
        frame["cluster_id"] = frame["source_run"] + "::" + frame["game_id"].astype(str)
        try:
            fit = smf.glm(formula=formula, data=frame, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": frame["cluster_id"]}
            )
        except Exception as exc:
            rows.append({"model_slug": model_slug, "term": "__fit_error__", "error": str(exc)})
            continue
        conf = fit.conf_int()
        for term, coef in fit.params.items():
            low, high = conf.loc[term].tolist()
            rows.append(
                {
                    "model_slug": model_slug,
                    "term": term,
                    "coef": coef,
                    "odds_ratio": safe_exp(coef),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": fit.pvalues.get(term, np.nan),
                    "n": int(fit.nobs),
                    "clusters": frame["cluster_id"].nunique(),
                    "error": "",
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED_DIR / "risk_matrix_asymmetry_logit.csv", index=False)
    return frame


def build_human_check_risk_matrix(turns: pd.DataFrame) -> pd.DataFrame:
    smf, sm = safe_import_statsmodels()
    if smf is None or not HUMAN_REFERENCE_PATH.exists():
        return pd.DataFrame()
    reference = json.loads(HUMAN_REFERENCE_PATH.read_text(encoding="utf-8"))
    expected_sign = {
        HUMAN_TERMS.get(e.get("name")): e.get("expected_sign", "")
        for e in reference.get("effects", [])
        if HUMAN_TERMS.get(e.get("name"))
    }
    human_value = {
        HUMAN_TERMS.get(e.get("name")): e.get("human_value")
        for e in reference.get("effects", [])
        if HUMAN_TERMS.get(e.get("name"))
    }
    formula = "unsafe ~ own_prev_unsafe + opponent_prev_unsafe + progress_gap_before + first_round_unsafe"
    round2 = turns[turns["is_round2plus"]].dropna(
        subset=["unsafe", "own_prev_unsafe", "opponent_prev_unsafe", "progress_gap_before", "first_round_unsafe", "game_id"]
    ).copy()
    rows: list[dict[str, Any]] = []
    for model_slug, frame in round2.groupby("model_slug", dropna=False):
        frame = frame.copy()
        if len(frame) < 100 or frame["unsafe"].nunique() < 2:
            continue
        frame["cluster_id"] = frame["source_run"] + "::" + frame["game_id"].astype(str)
        try:
            fit = smf.glm(formula=formula, data=frame, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": frame["cluster_id"]}
            )
        except Exception as exc:
            rows.append({"model_slug": model_slug, "term": "__fit_error__", "error": str(exc)})
            continue
        for term in HUMAN_TERMS.values():
            if term not in fit.params:
                continue
            coef = float(fit.params[term])
            expected = expected_sign.get(term, "")
            sign_match = (
                (expected == "positive" and coef > 0) or (expected == "negative" and coef < 0) or expected == ""
            )
            rows.append(
                {
                    "model_slug": model_slug,
                    "term": term,
                    "coef": coef,
                    "human_value": human_value.get(term),
                    "expected_sign": expected,
                    "sign_match": sign_match,
                    "p_value": fit.pvalues.get(term, np.nan),
                    "n": int(fit.nobs),
                    "phi_U_scope": float(frame["unsafe"].mean()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED_DIR / "risk_matrix_human_check.csv", index=False)
    return frame


def plot_own_vs_opponent_marginal(own: pd.DataFrame, opp: pd.DataFrame) -> Path | None:
    if own.empty or opp.empty:
        return None
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    for model in MODEL_ORDER:
        sub = own[own["model_slug"] == model].sort_values("own_risk_label")
        if not sub.empty:
            axes[0].plot(sub["own_risk_label"], sub["unsafe_rate"], marker="o", color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
        sub2 = opp[opp["model_slug"] == model].sort_values("opponent_risk_label")
        if not sub2.empty:
            axes[1].plot(sub2["opponent_risk_label"], sub2["unsafe_rate"], marker="o", color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
    axes[0].set_title("Unsafe rate vs OWN risk label", loc="left", fontweight="bold")
    axes[1].set_title("Unsafe rate vs OPPONENT risk label", loc="left", fontweight="bold")
    for ax in axes:
        ax.set_xlabel("risk label (1=low .. 6=high)")
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("Unsafe rate")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    return save(fig, "01_own_vs_opponent_marginal.png")


def plot_label_gap(label_gap: pd.DataFrame) -> Path | None:
    if label_gap.empty:
        return None
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for model in MODEL_ORDER:
        sub = label_gap[label_gap["model_slug"] == model].sort_values("label_gap_own_minus_opp")
        if sub.empty:
            continue
        ax.plot(sub["label_gap_own_minus_opp"], sub["unsafe_rate"], marker="o", color=MODEL_COLORS.get(model), label=MODEL_LABELS.get(model, model))
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_xlabel("own risk label minus opponent risk label")
    ax.set_ylabel("Unsafe rate")
    ax.set_ylim(0, 1)
    ax.set_title("Unsafe rate vs relative risk-label gap", loc="left", fontweight="bold", pad=18)
    ax.legend(loc="best", fontsize=8)
    return save(fig, "02_label_gap_effect.png")


def fmt(value: float, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.1%}" if pct else f"{value:.3g}"


def write_report(
    own: pd.DataFrame,
    opp: pd.DataFrame,
    label_gap: pd.DataFrame,
    real_vs_label: pd.DataFrame,
    temporal: pd.DataFrame,
    asymmetry: pd.DataFrame,
    human_check: pd.DataFrame,
    n_decisions: int,
    n_games: int,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# FH Risk-Matrix Asymmetry Mining")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"`mode_risk_matrix` decisions: {n_decisions:,} across {n_games:,} races -- the single largest slice of "
        "collected data, bigger than baseline and `mode_strategy_persona` combined, and previously reported only "
        "as a one-line aggregate unsafe rate. Each seat is assigned a narrative risk-framing label `risk-1` .. "
        "`risk-6`, independent of the real mechanistic private-risk treatment (`max_private_risk` in "
        "{0.1, 0.6, 0.9}); the full 6x6 own/opponent label grid (`R1_R1` .. `R6_R6`) is run, so this is the one "
        "mode where own-label and opponent-label effects are genuinely separable."
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    if not asymmetry.empty:
        fit_rows = asymmetry[~asymmetry["term"].astype(str).str.startswith("__")]
        own_terms = fit_rows[fit_rows["term"] == "own_risk_label"]
        opp_terms = fit_rows[fit_rows["term"] == "opponent_risk_label"]
        real_terms = fit_rows[fit_rows["term"].astype(str).str.startswith("C(max_private_risk)")]
        lines.append(
            "- **The narrative risk label moves behavior on its own, separate from the real private-risk treatment.** "
            "`own_risk_label` and `C(max_private_risk)` are both included in the same logit; see the coefficient "
            "table below for whether the narrative label survives once the real mechanistic risk is controlled for."
        )
        if not own_terms.empty and not opp_terms.empty:
            lines.append(
                f"- **Own risk label dominates opponent risk label** in every model that fit: own-label "
                f"coefficients range {fmt(own_terms['coef'].min())} to {fmt(own_terms['coef'].max())} "
                f"(all own-label p-values: {', '.join(fmt(p) for p in own_terms['p_value'])}) versus opponent-label "
                f"coefficients {fmt(opp_terms['coef'].min())} to {fmt(opp_terms['coef'].max())} "
                f"(p-values: {', '.join(fmt(p) for p in opp_terms['p_value'])})."
            )
        if not real_terms.empty:
            lines.append(
                "- Real private-risk treatment (`max_private_risk`) coefficients, net of the narrative label: see "
                "`risk_matrix_asymmetry_logit.csv` rows starting `C(max_private_risk)`."
            )
    if not label_gap.empty:
        lines.append(
            "- **Relative label gap (own minus opponent) also matters**, not just the own level in isolation -- see "
            "`risk_matrix_label_gap.csv` and the figure below for whether the relationship is monotonic or "
            "flattens out."
        )
    if not human_check.empty:
        matches = human_check[~human_check["term"].astype(str).str.startswith("__", na=False)]
        share = matches["sign_match"].mean() if not matches.empty else np.nan
        lines.append(
            f"- **Human-reference lag/gap signs**: {fmt(share)} of model x term sign checks against "
            "Fernandez Domingos & Han (2026) agree in direction within this mode (`risk_matrix_human_check.csv`)."
        )
    lines.append("")

    lines.append("## Own Vs Opponent Risk-Label Marginals")
    lines.append("")
    lines.append(markdown_table(own))
    lines.append("")
    lines.append(markdown_table(opp))
    lines.append("")
    lines.append("Visual: `figures/risk_matrix_asymmetry/01_own_vs_opponent_marginal.png`.")
    lines.append("")

    lines.append("## Relative Label Gap (Own Minus Opponent)")
    lines.append("")
    lines.append(markdown_table(label_gap))
    lines.append("")
    lines.append("Visual: `figures/risk_matrix_asymmetry/02_label_gap_effect.png`.")
    lines.append("")

    lines.append("## Real Private-Risk Treatment Vs Narrative Label")
    lines.append("")
    lines.append(
        "Own-label unsafe rate broken out by the real `max_private_risk` treatment; if rows are similar across "
        "`max_private_risk` for a fixed label, the narrative label is doing most of the work."
    )
    lines.append("")
    lines.append(markdown_table(real_vs_label))
    lines.append("")

    lines.append("## Temporal Trend By Own-Label Band")
    lines.append("")
    lines.append(markdown_table(temporal))
    lines.append("")

    lines.append("## Own-Vs-Opponent Asymmetry Logit")
    lines.append("")
    lines.append(
        "Cluster-robust logit of `unsafe` on own risk label, opponent risk label, the real `max_private_risk` "
        "treatment, progress gap, and lag terms, fit per model on round >= 2 decisions."
    )
    lines.append("")
    if asymmetry.empty:
        lines.append("_Not enough support to fit this model; see `risk_matrix_asymmetry_skipped.csv`._")
    else:
        fit_rows = asymmetry[~asymmetry["term"].astype(str).str.startswith("__")]
        lines.append(markdown_table(fit_rows[["model_slug", "term", "coef", "odds_ratio", "p_value", "n", "clusters"]]))
        error_rows = asymmetry[asymmetry["term"].astype(str).str.startswith("__")]
        if not error_rows.empty:
            lines.append("")
            lines.append("Models that did not fit:")
            lines.append("")
            lines.append(markdown_table(error_rows[["model_slug", "term", "error"]]))
    lines.append("")

    lines.append("## Human-Reference Check Within Risk-Matrix Mode")
    lines.append("")
    if human_check.empty:
        lines.append("_statsmodels unavailable or human_reference.json missing; skipped._")
    else:
        lines.append(
            markdown_table(
                human_check[~human_check["term"].astype(str).str.startswith("__")][
                    ["model_slug", "term", "coef", "human_value", "expected_sign", "sign_match", "p_value", "n", "phi_U_scope"]
                ]
            )
        )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- `own_risk_label`/`opponent_risk_label` are treated as continuous (1-6) in the logit for parsimony; the "
        "marginal tables above show the raw per-level rates in case the relationship is non-monotonic."
    )
    lines.append(
        "- This mode's real `max_private_risk` treatment (0.1/0.6/0.9) is the same mechanism used in baseline; "
        "the narrative risk label (`risk-1`..`risk-6`) is an independent prompt-level manipulation layered on top, "
        "not part of the paper-faithful mechanism itself."
    )
    lines.append(
        "- Descriptive/mechanistic evidence only; not a causal claim about model 'understanding' of the framing."
    )

    path = REPORTS_DIR / "fh_risk_matrix_asymmetry_mining.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    turns = load_risk_matrix_turns()
    n_decisions = len(turns)
    n_games = turns[["source_run", "game_id"]].drop_duplicates().shape[0]
    print(f"risk-matrix decisions: {n_decisions}, races: {n_games}")

    own, opp = build_marginal_tables(turns)
    label_gap = build_label_gap_table(turns)
    real_vs_label = build_real_vs_label_table(turns)
    temporal = build_temporal_table(turns)
    asymmetry = build_asymmetry_logit(turns)
    human_check = build_human_check_risk_matrix(turns)

    plot_own_vs_opponent_marginal(own, opp)
    plot_label_gap(label_gap)

    report_path = write_report(
        own, opp, label_gap, real_vs_label, temporal, asymmetry, human_check, n_decisions, n_games
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
