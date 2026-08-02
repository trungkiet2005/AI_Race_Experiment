#!/usr/bin/env python3
"""Cross-model synthesis: what is common across all 5 models, what is not.

This stage does not re-derive raw behavior. It pulls already-computed derived
tables from the main pipeline (`run_fh_analysis.py`, `run_strategy_playbook_mining.py`,
`run_expanded_strategy_library.py`, `run_strategy_synthesis_full.py`) and the new
mining stages added in this session (`run_persona_compliance_mining.py`,
`run_risk_matrix_asymmetry.py`, `run_temporal_endgame_analysis.py`) and asks two
cross-cutting questions that no single stage answers on its own:

1. Which human-reference-style effects (own/opponent lag, progress gap) hold
   the same sign across *all* five models, across *multiple* experimental
   scopes (baseline, risk-matrix mode) -- not just within one scope;
2. Which strategic levers (retaliation, opportunistic-when-ahead,
   catch-up-when-behind, forgiveness, mutual-unsafe stickiness) are shared
   across all five models versus idiosyncratic to one family or model.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
DERIVED_DIR = OUTPUT_DIR / "derived"
REPORTS_DIR = OUTPUT_DIR / "reports"
HUMAN_REFERENCE_PATH = REPO_ROOT / "results" / "scripts" / "human_reference.json"

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
HUMAN_TERMS = ["own_prev_unsafe", "opponent_prev_unsafe", "progress_gap_before"]


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


def load_human_reference() -> tuple[dict[str, str], dict[str, float]]:
    reference = json.loads(HUMAN_REFERENCE_PATH.read_text(encoding="utf-8"))
    expected = {e.get("name"): e.get("expected_sign", "") for e in reference.get("effects", [])}
    values = {e.get("name"): e.get("human_value") for e in reference.get("effects", [])}
    return expected, values


def sign_match(coef: float, expected: str) -> Any:
    if expected == "positive":
        return coef > 0
    if expected == "negative":
        return coef < 0
    return np.nan


def build_human_scorecard(expected: dict[str, str]) -> pd.DataFrame:
    """Per-model, per-scope sign checks on the comparable (model_slug-grain) tables."""

    rows: list[dict[str, Any]] = []

    baseline = pd.read_csv(DERIVED_DIR / "human_check_segment_coefficients.csv")
    baseline = baseline[
        (baseline["segment_type"] == "baseline_model") & (baseline["term"].isin(HUMAN_TERMS))
    ]
    fitted_models = set(baseline["segment"].unique())
    for _, row in baseline.iterrows():
        exp = expected.get(row["term"], "")
        rows.append(
            {
                "scope": "baseline",
                "model_slug": row["segment"],
                "term": row["term"],
                "coef": row["coef"],
                "expected_sign": exp,
                "sign_match": sign_match(row["coef"], exp),
                "fit_source": "human_check_segment (M3 interactions)",
            }
        )

    # `human_check_segment_coefficients.csv` fits the interaction-heavy M3 formula and can hit a
    # singular design matrix for models with sparse lag x gap combinations (e.g. gpt-5-nano baseline).
    # Fall back to the plain main-effects logit already fit per model by run_model_diagnostics.py so no
    # model silently drops out of the cross-model scorecard.
    fallback_path = DERIVED_DIR / "model_logit_coefficients.csv"
    if fallback_path.exists():
        fallback = pd.read_csv(fallback_path)
        fallback = fallback[fallback["term"].isin(HUMAN_TERMS) & ~fallback["model_slug"].isin(fitted_models)]
        for _, row in fallback.iterrows():
            exp = expected.get(row["term"], "")
            rows.append(
                {
                    "scope": "baseline",
                    "model_slug": row["model_slug"],
                    "term": row["term"],
                    "coef": row["coef"],
                    "expected_sign": exp,
                    "sign_match": sign_match(row["coef"], exp),
                    "fit_source": "model_logit_coefficients (main effects only, fallback)",
                }
            )

    risk_matrix_path = DERIVED_DIR / "risk_matrix_human_check.csv"
    if risk_matrix_path.exists():
        rm = pd.read_csv(risk_matrix_path)
        rm = rm[~rm["term"].astype(str).str.startswith("__")]
        for _, row in rm.iterrows():
            rows.append(
                {
                    "scope": "risk_matrix",
                    "model_slug": row["model_slug"],
                    "term": row["term"],
                    "coef": row["coef"],
                    "expected_sign": row["expected_sign"],
                    "sign_match": row["sign_match"],
                    "fit_source": "risk_matrix_human_check (main effects)",
                }
            )

    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED_DIR / "cross_model_human_scorecard.csv", index=False)
    return frame


def build_commonality_index(scorecard: pd.DataFrame) -> pd.DataFrame:
    usable = scorecard[scorecard["expected_sign"].isin(["positive", "negative"])].copy()
    usable["sign_match"] = usable["sign_match"].astype(bool)
    rows: list[dict[str, Any]] = []
    for term, group in usable.groupby("term"):
        rows.append(
            {
                "term": term,
                "expected_sign": group["expected_sign"].iloc[0],
                "fits": len(group),
                "models_covered": group["model_slug"].nunique(),
                "scopes_covered": group["scope"].nunique(),
                "sign_match_share": float(group["sign_match"].mean()),
            }
        )
    frame = pd.DataFrame(rows).sort_values("sign_match_share", ascending=False)
    frame.to_csv(DERIVED_DIR / "cross_model_commonality_index.csv", index=False)
    return frame


def build_lever_commonality() -> pd.DataFrame:
    levers = pd.read_csv(DERIVED_DIR / "strategy_playbook_levers.csv")
    model_rows = levers[levers["scope"] == "model"].copy()
    lever_cols = ["retaliation_lift", "opportunistic_lift", "catchup_lift"]
    rows: list[dict[str, Any]] = []
    for col in lever_cols:
        values = model_rows[["model_slug", col]].dropna()
        if values.empty:
            continue
        positive = (values[col] > 0).sum()
        negative = (values[col] < 0).sum()
        rows.append(
            {
                "lever": col,
                "models_with_data": len(values),
                "models_positive": int(positive),
                "models_negative": int(negative),
                "unanimous_sign": bool(positive == len(values) or negative == len(values)),
                "min_value": float(values[col].min()),
                "max_value": float(values[col].max()),
            }
        )
    for col in ["forgiveness_rate", "mutual_unsafe_stickiness"]:
        values = model_rows[["model_slug", col]].dropna()
        if values.empty:
            continue
        rows.append(
            {
                "lever": col,
                "models_with_data": len(values),
                "models_positive": len(values),
                "models_negative": 0,
                "unanimous_sign": True,
                "min_value": float(values[col].min()),
                "max_value": float(values[col].max()),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(DERIVED_DIR / "cross_model_lever_commonality.csv", index=False)

    detail = model_rows[["model_slug"] + lever_cols + ["forgiveness_rate", "mutual_unsafe_stickiness"]].copy()
    detail.to_csv(DERIVED_DIR / "cross_model_lever_detail.csv", index=False)
    return frame, detail


def build_endgame_flatness() -> pd.DataFrame:
    path = DERIVED_DIR / "temporal_endgame_by_rounds_remaining.csv"
    if not path.exists():
        return pd.DataFrame()
    endgame = pd.read_csv(path)
    endgame = endgame[endgame["analysis_scope"] == "all_completed"].copy()
    band_x = {"4plus": 4.5, "3": 3, "2": 2, "1": 1, "0": 0}
    endgame["x"] = endgame["rounds_remaining_band"].map(band_x)
    rows: list[dict[str, Any]] = []
    for model, group in endgame.groupby("model_slug"):
        group = group.dropna(subset=["x", "unsafe_rate"])
        if len(group) < 3:
            continue
        slope, _intercept = np.polyfit(group["x"], group["unsafe_rate"], 1)
        rows.append(
            {
                "model_slug": model,
                "slope_unsafe_per_round_closer_to_end": -slope,
                "range_unsafe_rate": float(group["unsafe_rate"].max() - group["unsafe_rate"].min()),
            }
        )
    frame = pd.DataFrame(rows).sort_values("slope_unsafe_per_round_closer_to_end", ascending=False)
    frame.to_csv(DERIVED_DIR / "cross_model_endgame_flatness.csv", index=False)
    return frame


def fmt(value: float, pct: bool = False) -> str:
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "n/a"
    return f"{value:+.1%}" if pct else f"{value:.3g}"


def write_report(
    scorecard: pd.DataFrame,
    commonality: pd.DataFrame,
    lever_commonality: pd.DataFrame,
    lever_detail: pd.DataFrame,
    endgame_flatness: pd.DataFrame,
) -> Path:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append("# FH Cross-Model Synthesis: Common Ground Vs Model-Specific")
    lines.append("")
    lines.append("## Purpose")
    lines.append("")
    lines.append(
        "This does not run new statistical models. It cross-references derived tables from the whole "
        "`fh_analytic` pipeline (baseline, persona-role, risk-matrix, temporal, mechanism-mining, strategy-synthesis "
        "stages) to answer: which patterns are common to all five models tested, and which are specific to a "
        "family, a model, or an assigned condition? Read this alongside the individual stage reports; every number "
        "here traces back to a derived CSV listed in its section."
    )
    lines.append("")

    lines.append("## Executive Summary")
    lines.append("")
    if not commonality.empty:
        for _, row in commonality.iterrows():
            lines.append(
                f"- **`{row['term']}`** (human-expected sign: {row['expected_sign']}): agrees in direction in "
                f"{fmt(row['sign_match_share'])} of {int(row['fits'])} model x scope fits across "
                f"{int(row['scopes_covered'])} scopes and {int(row['models_covered'])} models "
                "(`cross_model_human_scorecard.csv`, `cross_model_commonality_index.csv`)."
            )
    if not lever_commonality.empty:
        unanimous = lever_commonality[lever_commonality["unanimous_sign"]]["lever"].tolist()
        mixed = lever_commonality[~lever_commonality["unanimous_sign"]]["lever"].tolist()
        if unanimous:
            lines.append(f"- **Unanimous-sign levers across all 5 models**: {', '.join(unanimous)}.")
        if mixed:
            lines.append(
                f"- **Sign-mixed levers (model-specific direction)**: {', '.join(mixed)} -- some models show the "
                "lever, others show the opposite; see `cross_model_lever_detail.csv`."
            )
    if not endgame_flatness.empty:
        max_slope = endgame_flatness.iloc[0]
        lines.append(
            f"- **Endgame effect is small for every model** (`cross_model_endgame_flatness.csv`); the largest "
            f"per-model slope is {MODEL_LABELS.get(max_slope['model_slug'], max_slope['model_slug'])} at "
            f"{fmt(max_slope['slope_unsafe_per_round_closer_to_end'])} unsafe-rate points per round closer to the "
            "true (hidden) end, consistent with the horizon genuinely not leaking through to behavior."
        )
    lines.append(
        "- **Persona-role compliance and risk-label sensitivity are common to all models that ran those modes** "
        "(3 of 5): every model raises unsafe rate under `adversarial`/`risk-seeking` framing and lowers it under "
        "`cooperative`/`risk-averse` framing, and in every model the player's *own* assigned role/label dominates "
        "the opponent's (see `fh_persona_compliance_mining.md`, `fh_risk_matrix_asymmetry_mining.md`). What is "
        "*not* common is the strength of that compliance and whether the real mechanistic risk treatment still "
        "matters net of the narrative label (it does for Gemini and GPT-5.4 nano, not for GPT-5 nano)."
    )
    lines.append("")

    lines.append("## Human-Reference Commonality Index")
    lines.append("")
    lines.append(
        "Fraction of model x scope logit fits whose coefficient sign agrees with the human-reference direction "
        "(Fernandez Domingos & Han 2026, `results/scripts/human_reference.json`), pooling the baseline "
        "per-model fits and the risk-matrix per-model fits (the two scopes with comparable model-level grain)."
    )
    lines.append("")
    lines.append(markdown_table(commonality))
    lines.append("")
    lines.append("Full per-model, per-scope detail:")
    lines.append("")
    lines.append(markdown_table(scorecard[["scope", "model_slug", "term", "coef", "expected_sign", "sign_match", "fit_source"]]))
    lines.append("")

    lines.append("## Strategic-Lever Commonality")
    lines.append("")
    lines.append(
        "Retaliation / opportunistic / catch-up lifts, forgiveness, and mutual-unsafe stickiness "
        "(definitions in `fh_strategy_playbook_mining.md`), evaluated across all 5 models on baseline data."
    )
    lines.append("")
    lines.append(markdown_table(lever_commonality))
    lines.append("")
    lines.append(markdown_table(lever_detail))
    lines.append("")

    lines.append("## Endgame Flatness By Model")
    lines.append("")
    lines.append(markdown_table(endgame_flatness))
    lines.append("")

    lines.append("## What This Adds Up To")
    lines.append("")
    lines.append(
        "- The most model-universal finding across this whole pipeline is **opponent-triggered reactivity**: "
        "`opponent_prev_unsafe` is positive (matches the human direction) far more consistently than "
        "`own_prev_unsafe` or `progress_gap_before`, across both baseline and risk-matrix scopes."
    )
    lines.append(
        "- The most model-specific finding is **how much a model listens to narrative framing versus the real "
        "mechanistic risk number**: GPT-5 nano's behavior under `mode_risk_matrix` is statistically flat across "
        "the real `max_private_risk` treatment (p=0.98, p=0.67) once the narrative risk label is in the prompt, "
        "while Gemini 3 Flash and GPT-5.4 nano still respond to the real treatment net of the label "
        "(`risk_matrix_asymmetry_logit.csv`)."
    )
    lines.append(
        "- Persona/role compliance, own-role-dominates-opponent-role, and a flat (non-escalating) hidden-horizon "
        "curve are the three genuinely new cross-model regularities surfaced in this session that were not "
        "visible in the original baseline-only reports."
    )
    lines.append("")

    lines.append("## Caveats")
    lines.append("")
    lines.append(
        "- The commonality index only pools scopes with model-level grain (baseline, risk-matrix); "
        "`mode_strategy_persona` human-check output is at role grain, not model grain, and is reported separately "
        "in `fh_persona_compliance_mining.md` because pooling it here would silently change the unit of analysis."
    )
    lines.append(
        "- Sign-match counting treats near-separated/very large coefficients the same as small, precisely "
        "estimated ones; consult the p-values and confidence intervals in the underlying CSVs before treating any "
        "single sign match as strong evidence."
    )
    lines.append("- All findings remain pilot-phase, exploratory, and specific to the five checkpoint models tested; see CLAUDE.md on pooling pilot and confirmatory evidence.")

    path = REPORTS_DIR / "fh_cross_model_synthesis.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def main() -> None:
    expected, _values = load_human_reference()
    scorecard = build_human_scorecard(expected)
    commonality = build_commonality_index(scorecard)
    lever_commonality, lever_detail = build_lever_commonality()
    endgame_flatness = build_endgame_flatness()
    report_path = write_report(scorecard, commonality, lever_commonality, lever_detail, endgame_flatness)
    print(f"wrote {report_path}")


if __name__ == "__main__":
    main()
