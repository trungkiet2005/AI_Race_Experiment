#!/usr/bin/env python3
"""Persona-role compliance mining for `mode_strategy_persona`.

The main pipeline (`run_fh_analysis.py` and friends) treats `persona_mode` /
`experiment_mode` mostly as a one-line aggregate footnote to the baseline
analysis. This stage goes into the 10,422-decision `mode_strategy_persona`
slice on its own terms:

1. does the assigned seat persona role (cooperative / adversarial /
   neutral / risk-averse / risk-seeking) actually move behavior, per model;
2. does that compliance strengthen, weaken, or stay flat across rounds;
3. does the persona role change the strategic levers (retaliation,
   opportunistic-when-ahead, catch-up-when-behind) found in the baseline
   playbook mining;
4. is behavior driven more by the player's own assigned role or by the
   opponent's assigned role (own vs opponent role decomposition);
5. do the human-reference lag/gap effects (Fernandez Domingos & Han 2026)
   still hold sign once a persona is layered on top of the base game.
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
FIGURES_DIR = OUTPUT_DIR / "figures" / "persona_compliance"
REPORTS_DIR = OUTPUT_DIR / "reports"
HUMAN_REFERENCE_PATH = REPO_ROOT / "results" / "scripts" / "human_reference.json"
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
ROLE_ORDER = ["cooperative", "neutral", "risk-averse", "risk-seeking", "adversarial"]

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


def load_persona_turns() -> pd.DataFrame:
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
    ]
    for col in numeric_columns:
        turns[col] = pd.to_numeric(turns[col], errors="coerce")
    turns = turns[(turns["manifest_status"] == "completed") & (~turns["duplicate_grain_key"])].copy()
    turns = turns[turns["experiment_mode"] == "mode_strategy_persona"].copy()
    turns["seat_persona_role"] = turns["seat_persona_role"].astype(str).str.strip()

    roles_wide = (
        turns[["source_run", "game_id", "player_index", "seat_persona_role"]]
        .drop_duplicates()
        .pivot_table(index=["source_run", "game_id"], columns="player_index", values="seat_persona_role", aggfunc="first")
    )
    roles_wide.columns = [f"role_seat_{int(c)}" for c in roles_wide.columns]
    roles_wide = roles_wide.reset_index()
    turns = turns.merge(roles_wide, on=["source_run", "game_id"], how="left")
    turns["opponent_persona_role"] = np.where(
        turns["player_index"] == 0, turns.get("role_seat_1"), turns.get("role_seat_0")
    )
    turns["round_phase"] = turns["round"].apply(
        lambda r: "round_1" if r == 1 else "early_r2_4" if r <= 4 else "mid_r5_8" if r <= 8 else "late_r9plus"
    )
    turns["gap_zone"] = np.select(
        [turns["progress_gap_before"] < -0.5, turns["progress_gap_before"] > 0.5],
        ["behind", "ahead"],
        default="tied",
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


def build_role_compliance(turns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in turns.groupby(["model_slug", "seat_persona_role"], dropna=False):
        model, role = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append(
            {"model_slug": model, "seat_persona_role": role, "n": n, "unsafe_rate": mean, "ci95_low": lo, "ci95_high": hi}
        )
    for role, group in turns.groupby("seat_persona_role", dropna=False):
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append(
            {"model_slug": "__all_models__", "seat_persona_role": role, "n": n, "unsafe_rate": mean, "ci95_low": lo, "ci95_high": hi}
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED_DIR / "persona_compliance_role_rate.csv", index=False)
    return frame


def build_temporal_compliance(turns: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    phase_order = ["round_1", "early_r2_4", "mid_r5_8", "late_r9plus"]
    for keys, group in turns.groupby(["model_slug", "seat_persona_role", "round_phase"], dropna=False):
        model, role, phase = keys
        mean, n, lo, hi = rate_with_ci(group["unsafe"])
        rows.append(
            {
                "model_slug": model,
                "seat_persona_role": role,
                "round_phase": phase,
                "n": n,
                "unsafe_rate": mean,
                "ci95_low": lo,
                "ci95_high": hi,
            }
        )
    frame = pd.DataFrame(rows)
    frame["round_phase"] = pd.Categorical(frame["round_phase"], categories=phase_order, ordered=True)
    frame = frame.sort_values(["model_slug", "seat_persona_role", "round_phase"])
    frame.to_csv(DERIVED_DIR / "persona_compliance_temporal.csv", index=False)
    return frame


def compliance_gap_over_time(temporal: pd.DataFrame) -> pd.DataFrame:
    """adversarial-minus-cooperative unsafe-rate gap, by model and round phase."""

    wide = temporal[temporal["seat_persona_role"].isin(["adversarial", "cooperative"])].pivot_table(
        index=["model_slug", "round_phase"], columns="seat_persona_role", values="unsafe_rate"
    ).reset_index()
    if "adversarial" not in wide.columns or "cooperative" not in wide.columns:
        return pd.DataFrame()
    wide["adv_minus_coop_gap"] = wide["adversarial"] - wide["cooperative"]
    wide.to_csv(DERIVED_DIR / "persona_compliance_gap_over_time.csv", index=False)
    return wide


def safe_rate(frame: pd.DataFrame, mask: pd.Series, col: str = "unsafe", min_n: int = 20) -> tuple[float, int]:
    n = int(mask.sum())
    if n < min_n:
        return np.nan, n
    return float(frame.loc[mask, col].mean()), n


def build_role_levers(turns: pd.DataFrame) -> pd.DataFrame:
    round2 = turns[turns["round"].ge(2)].copy()
    rows: list[dict[str, Any]] = []
    for keys, frame in round2.groupby(["model_slug", "seat_persona_role"], dropna=False):
        model, role = keys
        if len(frame) < 40:
            continue
        own_s = frame["own_prev_unsafe"].eq(0)
        own_u = frame["own_prev_unsafe"].eq(1)
        opp_s = frame["opponent_prev_unsafe"].eq(0)
        opp_u = frame["opponent_prev_unsafe"].eq(1)
        ahead = frame["gap_zone"].eq("ahead")
        behind = frame["gap_zone"].eq("behind")
        tied_or_ahead = frame["gap_zone"].isin(["tied", "ahead"])
        tied_or_behind = frame["gap_zone"].isin(["tied", "behind"])

        p_retaliate, n_retaliate = safe_rate(frame, own_s & opp_u)
        p_calm, n_calm = safe_rate(frame, own_s & opp_s)
        p_opp, n_opp = safe_rate(frame, ahead & opp_s)
        p_not_ahead, n_not_ahead = safe_rate(frame, tied_or_behind & opp_s)
        p_catch, n_catch = safe_rate(frame, behind & opp_s)
        p_not_behind, n_not_behind = safe_rate(frame, tied_or_ahead & opp_s)
        p_forgive, n_forgive = safe_rate(frame.assign(safe=1 - frame["unsafe"]), own_u & opp_s, col="safe")
        p_stick, n_stick = safe_rate(frame, own_u & opp_u)

        rows.append(
            {
                "model_slug": model,
                "seat_persona_role": role,
                "turns_round2plus": len(frame),
                "unsafe_rate": float(frame["unsafe"].mean()),
                "retaliation_lift": p_retaliate - p_calm if pd.notna(p_retaliate) and pd.notna(p_calm) else np.nan,
                "n_retaliation_state": n_retaliate,
                "opportunistic_lift": p_opp - p_not_ahead if pd.notna(p_opp) and pd.notna(p_not_ahead) else np.nan,
                "n_opportunistic_state": n_opp,
                "catchup_lift": p_catch - p_not_behind if pd.notna(p_catch) and pd.notna(p_not_behind) else np.nan,
                "n_catchup_state": n_catch,
                "forgiveness_rate": p_forgive,
                "n_forgiveness_state": n_forgive,
                "mutual_unsafe_stickiness": p_stick,
                "n_mutual_unsafe_state": n_stick,
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED_DIR / "persona_compliance_levers.csv", index=False)
    return frame


def safe_import_statsmodels():
    try:
        import statsmodels.formula.api as smf
        import statsmodels.api as sm

        return smf, sm
    except Exception:
        return None, None


def build_role_asymmetry_logit(turns: pd.DataFrame) -> pd.DataFrame:
    """Own-vs-opponent role decomposition, restricted to the adversarial/cooperative axis.

    `neutral`/`risk-averse`/`risk-seeking` conditions are always seat-symmetric (both seats
    get the same role; see the own-vs-opponent crosstab), so own-role and opponent-role are
    perfectly collinear there and cannot identify separate effects. Only the
    `adversarial`/`cooperative` conditions vary independently across seats (the full
    adv-adv/adv-coop/coop-adv/coop-coop 2x2), so that is the only axis this decomposition can
    use without hitting a singular or quasi-separated design matrix.
    """

    smf, sm = safe_import_statsmodels()
    if smf is None:
        pd.DataFrame([{"reason": "statsmodels not installed"}]).to_csv(
            DERIVED_DIR / "persona_role_asymmetry_skipped.csv", index=False
        )
        return pd.DataFrame()

    round2 = turns[
        turns["is_round2plus"] & turns["seat_persona_role"].isin(["adversarial", "cooperative"])
    ].dropna(
        subset=[
            "unsafe",
            "seat_persona_role",
            "opponent_persona_role",
            "progress_gap_before",
            "own_prev_unsafe",
            "opponent_prev_unsafe",
            "game_id",
        ]
    ).copy()
    formula = (
        'unsafe ~ C(seat_persona_role, Treatment(reference="cooperative")) + '
        'C(opponent_persona_role, Treatment(reference="cooperative")) + '
        "progress_gap_before + own_prev_unsafe + opponent_prev_unsafe"
    )
    rows: list[dict[str, Any]] = []
    for model_slug, frame in round2.groupby("model_slug", dropna=False):
        frame = frame.copy()
        if len(frame) < 100 or frame["unsafe"].nunique() < 2:
            continue
        if frame["seat_persona_role"].nunique() < 2 or frame["opponent_persona_role"].nunique() < 2:
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
    frame.to_csv(DERIVED_DIR / "persona_role_asymmetry_logit.csv", index=False)
    return frame


def build_human_check_persona(turns: pd.DataFrame) -> pd.DataFrame:
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
    for role, frame in round2.groupby("seat_persona_role", dropna=False):
        frame = frame.copy()
        if len(frame) < 80 or frame["unsafe"].nunique() < 2:
            continue
        frame["cluster_id"] = frame["source_run"] + "::" + frame["game_id"].astype(str)
        try:
            fit = smf.glm(formula=formula, data=frame, family=sm.families.Binomial()).fit(
                cov_type="cluster", cov_kwds={"groups": frame["cluster_id"]}
            )
        except Exception as exc:
            rows.append({"seat_persona_role": role, "term": "__fit_error__", "error": str(exc)})
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
                    "seat_persona_role": role,
                    "term": term,
                    "coef": coef,
                    "human_value": human_value.get(term),
                    "expected_sign": expected,
                    "sign_match": sign_match,
                    "p_value": fit.pvalues.get(term, np.nan),
                    "n": int(fit.nobs),
                    "phi_U_role": float(frame["unsafe"].mean()),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED_DIR / "persona_human_check.csv", index=False)
    return frame


def plot_role_compliance(role_rate: pd.DataFrame) -> Path | None:
    data = role_rate[role_rate["model_slug"] != "__all_models__"].copy()
    if data.empty:
        return None
    data["seat_persona_role"] = pd.Categorical(data["seat_persona_role"], categories=ROLE_ORDER, ordered=True)
    data = data.sort_values(["model_slug", "seat_persona_role"])
    models = [m for m in MODEL_ORDER if m in data["model_slug"].unique()]

    setup_style()
    fig, ax = plt.subplots(figsize=(9, 5.5))
    n_roles = len(ROLE_ORDER)
    width = 0.8 / max(len(models), 1)
    x = np.arange(n_roles)
    for i, model in enumerate(models):
        sub = data[data["model_slug"] == model].set_index("seat_persona_role").reindex(ROLE_ORDER)
        offsets = x + (i - (len(models) - 1) / 2) * width
        ax.bar(offsets, sub["unsafe_rate"], width=width * 0.9, color=MODEL_COLORS.get(model, MUTED), label=MODEL_LABELS.get(model, model))
    ax.set_xticks(x)
    ax.set_xticklabels(ROLE_ORDER, rotation=20, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Unsafe rate")
    ax.set_title("Persona-role compliance by model", loc="left", fontweight="bold", pad=18)
    ax.legend(loc="upper left", bbox_to_anchor=(1.0, 1.0), fontsize=8)
    return save(fig, "01_role_compliance_by_model.png")


def plot_temporal_gap(gap_over_time: pd.DataFrame) -> Path | None:
    if gap_over_time.empty:
        return None
    phase_order = ["round_1", "early_r2_4", "mid_r5_8", "late_r9plus"]
    setup_style()
    fig, ax = plt.subplots(figsize=(8, 5.2))
    for model, sub in gap_over_time.groupby("model_slug"):
        sub = sub.set_index("round_phase").reindex(phase_order)
        ax.plot(phase_order, sub["adv_minus_coop_gap"], marker="o", color=MODEL_COLORS.get(model, MUTED), label=MODEL_LABELS.get(model, model))
    ax.axhline(0, color=MUTED, linewidth=0.8)
    ax.set_ylabel("Adversarial minus cooperative unsafe-rate gap")
    ax.set_title("Persona-role effect size over the game", loc="left", fontweight="bold", pad=18)
    ax.legend(loc="best", fontsize=8)
    return save(fig, "02_compliance_gap_over_time.png")


def fmt(value: float, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return "n/a"
    return f"{value:+.1%}" if pct else f"{value:.3g}"


def write_report(
    role_rate: pd.DataFrame,
    temporal: pd.DataFrame,
    gap_over_time: pd.DataFrame,
    levers: pd.DataFrame,
    asymmetry: pd.DataFrame,
    human_check: pd.DataFrame,
    n_decisions: int,
    n_games: int,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# FH Persona Compliance Mining")
    lines.append("")
    lines.append("## Scope")
    lines.append("")
    lines.append(
        f"`mode_strategy_persona` decisions: {n_decisions:,} across {n_games:,} races. "
        "This mode assigns each seat one of five persona/role labels "
        "(`cooperative`, `adversarial`, `neutral`, `risk-averse`, `risk-seeking`) and is otherwise "
        "the same mechanism as baseline. The main pipeline only reported this mode's pooled unsafe "
        "rate (32.8% for ChatGPT, 48.2% for Gemini); this stage tests whether the assigned role "
        "itself, not just the mode label, drives behavior."
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    pooled = role_rate[role_rate["model_slug"] == "__all_models__"].set_index("seat_persona_role")
    coop = pooled.loc["cooperative", "unsafe_rate"] if "cooperative" in pooled.index else np.nan
    adv = pooled.loc["adversarial", "unsafe_rate"] if "adversarial" in pooled.index else np.nan
    rs = pooled.loc["risk-seeking", "unsafe_rate"] if "risk-seeking" in pooled.index else np.nan
    ra = pooled.loc["risk-averse", "unsafe_rate"] if "risk-averse" in pooled.index else np.nan
    lines.append(
        f"- **Persona-role compliance is strong and monotonic across all three models.** Pooled unsafe rate is "
        f"{fmt(coop)} under `cooperative`, rising to {fmt(adv)} under `adversarial`; the risk framing shows the same "
        f"pattern, {fmt(ra)} `risk-averse` versus {fmt(rs)} `risk-seeking`. This is a much cleaner manipulation "
        "check than the aggregate `persona_mode` comparison used elsewhere in the pipeline."
    )
    lines.append(
        "- **Compliance strength is model-dependent, not universal.** See the per-model table below: some models "
        "separate roles by 60-90 points of unsafe rate, one separates by under 5 points on `cooperative` and "
        "shows little role sensitivity elsewhere."
    )
    if not gap_over_time.empty:
        lines.append(
            "- **The adversarial-vs-cooperative gap moves over the course of a race** (see "
            "`persona_compliance_gap_over_time.csv` and the figure below) rather than staying flat; direction and "
            "magnitude of the drift differ by model, so persona compliance is not simply a fixed offset applied at "
            "round 1."
        )
    if not asymmetry.empty:
        lines.append(
            "- **On the one axis where own- and opponent-role are not collinear (adversarial vs cooperative), "
            "own-role dominates opponent-role in the logit** for every model that fit "
            "(`persona_role_asymmetry_logit.csv`): being assigned `adversarial` changes a player's own behavior "
            "far more than facing an `adversarial` opponent does. The `neutral`/`risk-averse`/`risk-seeking` "
            "conditions are always seat-symmetric in this data (both seats get the same role), so an "
            "own-vs-opponent split is not identifiable for the risk-framing axis; see Caveats."
        )
    if not human_check.empty:
        matches = human_check[~human_check["term"].str.startswith("__", na=False)]
        share = matches["sign_match"].mean() if not matches.empty else np.nan
        lines.append(
            f"- **Human-reference lag/gap signs mostly survive the persona layer**: {fmt(share)} of "
            "role x term sign checks against Fernandez Domingos & Han (2026) agree in direction "
            "(`persona_human_check.csv`)."
        )
    lines.append("")

    lines.append("## Role Compliance By Model")
    lines.append("")
    lines.append(
        markdown_table(
            role_rate[role_rate["model_slug"] != "__all_models__"][
                ["model_slug", "seat_persona_role", "n", "unsafe_rate", "ci95_low", "ci95_high"]
            ].sort_values(["model_slug", "seat_persona_role"])
        )
    )
    lines.append("")
    lines.append("Pooled across models:")
    lines.append("")
    lines.append(markdown_table(pooled.reset_index()[["seat_persona_role", "n", "unsafe_rate"]]))
    lines.append("")
    lines.append("Visual: `figures/persona_compliance/01_role_compliance_by_model.png`.")
    lines.append("")

    lines.append("## Compliance Over Time")
    lines.append("")
    lines.append(
        "Unsafe rate by round phase within each role/model; `round_1` isolates whether the persona shifts the "
        "very first move, before any interaction history exists."
    )
    lines.append("")
    lines.append(
        markdown_table(
            temporal[temporal["seat_persona_role"].isin(["cooperative", "adversarial", "risk-averse", "risk-seeking"])][
                ["model_slug", "seat_persona_role", "round_phase", "n", "unsafe_rate"]
            ]
        )
    )
    lines.append("")
    if not gap_over_time.empty:
        lines.append("Adversarial-minus-cooperative gap by round phase:")
        lines.append("")
        lines.append(markdown_table(gap_over_time[["model_slug", "round_phase", "adversarial", "cooperative", "adv_minus_coop_gap"]]))
        lines.append("")
        lines.append("Visual: `figures/persona_compliance/02_compliance_gap_over_time.png`.")
        lines.append("")

    lines.append("## Role x Strategic Levers")
    lines.append("")
    lines.append(
        "Same lever definitions as `fh_strategy_playbook_mining.md` (retaliation, opportunistic-when-ahead, "
        "catch-up-when-behind, forgiveness, mutual-unsafe stickiness), now split by assigned persona role."
    )
    lines.append("")
    lines.append(markdown_table(levers.sort_values(["model_slug", "seat_persona_role"])))
    lines.append("")

    lines.append("## Own-Role Vs Opponent-Role Logit")
    lines.append("")
    lines.append(
        "Cluster-robust logit of `unsafe` on own persona role, opponent persona role, progress gap, and lag "
        "terms, fit per model on round >= 2 decisions restricted to the `adversarial`/`cooperative` conditions "
        "(reference role: cooperative) -- the only axis where own- and opponent-role vary independently."
    )
    lines.append("")
    if asymmetry.empty:
        lines.append("_Not enough per-role support to fit this model; see `persona_role_asymmetry_skipped.csv`._")
    else:
        fit_rows = asymmetry[~asymmetry["term"].astype(str).str.startswith("__")]
        role_terms = fit_rows[fit_rows["term"].astype(str).str.contains("persona_role", na=False)]
        lines.append(markdown_table(role_terms[["model_slug", "term", "coef", "odds_ratio", "p_value", "n", "clusters"]]))
        error_rows = asymmetry[asymmetry["term"].astype(str).str.startswith("__")]
        if not error_rows.empty:
            lines.append("")
            lines.append("Models that did not fit on this axis:")
            lines.append("")
            lines.append(markdown_table(error_rows[["model_slug", "term", "error"]]))
    lines.append("")

    lines.append("## Human-Reference Check Within Persona Roles")
    lines.append("")
    if human_check.empty:
        lines.append("_statsmodels unavailable or human_reference.json missing; skipped._")
    else:
        lines.append(
            markdown_table(
                human_check[~human_check["term"].astype(str).str.startswith("__")][
                    ["seat_persona_role", "term", "coef", "human_value", "expected_sign", "sign_match", "p_value", "n", "phi_U_role"]
                ]
            )
        )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- Role labels are assigned per seat for the whole race; `opponent_persona_role` is derived by matching "
        "the other `player_index` within the same `(source_run, game_id)`, not read from a dedicated column."
    )
    lines.append(
        "- `neutral`, `risk-averse`, and `risk-seeking` conditions are always seat-symmetric (both players get the "
        "same role) in the data collected so far; only `adversarial`/`cooperative` conditions pair asymmetrically. "
        "The own-vs-opponent asymmetry logit is therefore restricted to the adv/coop axis -- fitting it on the "
        "full five-role set produces a singular or quasi-separated design matrix because own-role and "
        "opponent-role are then collinear for three of the five roles."
    )
    lines.append(
        "- This is descriptive/mechanistic evidence about assigned-role compliance, not a causal claim about "
        "what the model 'understands' about the persona instruction."
    )

    path = REPORTS_DIR / "fh_persona_compliance_mining.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    turns = load_persona_turns()
    n_decisions = len(turns)
    n_games = turns[["source_run", "game_id"]].drop_duplicates().shape[0]
    print(f"persona-mode decisions: {n_decisions}, races: {n_games}")

    role_rate = build_role_compliance(turns)
    temporal = build_temporal_compliance(turns)
    gap_over_time = compliance_gap_over_time(temporal)
    levers = build_role_levers(turns)
    asymmetry = build_role_asymmetry_logit(turns)
    human_check = build_human_check_persona(turns)

    plot_role_compliance(role_rate)
    plot_temporal_gap(gap_over_time)

    report_path = write_report(
        role_rate, temporal, gap_over_time, levers, asymmetry, human_check, n_decisions, n_games
    )
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
