#!/usr/bin/env python3
"""Sequential analysis for the copied frontier/persona data layout.

The runner is intentionally staged from simple to richer analyses:

1. coverage and completeness audit;
2. descriptive summaries and plots;
3. human-reference checks via turn-level logistic models;
4. exploratory decision-tree/rule models.

All inferential outputs are exploratory unless the input snapshot and target scope
are frozen elsewhere.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "experiments" / "players_2"
DEFAULT_OUTPUT = REPO_ROOT / "analysis" / "fh_analytic" / "outputs"
HUMAN_REFERENCE_PATH = REPO_ROOT / "results" / "scripts" / "human_reference.json"

HUMAN_CHECK_FORMULA = (
    "unsafe ~ C(max_private_risk) + first_round_unsafe + own_prev_unsafe * "
    "opponent_prev_unsafe * progress_gap_before"
)
LOGIT_SPECS: tuple[tuple[str, str], ...] = (
    ("M0_risk_only", "unsafe ~ C(max_private_risk)"),
    (
        "M1_human_core",
        "unsafe ~ C(max_private_risk) + own_prev_unsafe + "
        "opponent_prev_unsafe + progress_gap_before",
    ),
    (
        "M2_first_round",
        "unsafe ~ C(max_private_risk) + first_round_unsafe + own_prev_unsafe + "
        "opponent_prev_unsafe + progress_gap_before",
    ),
    (
        "M3_interactions",
        HUMAN_CHECK_FORMULA,
    ),
    (
        "M4_model_fixed_effects",
        "unsafe ~ C(max_private_risk) + first_round_unsafe + own_prev_unsafe * "
        "opponent_prev_unsafe * progress_gap_before + C(model_slug)",
    ),
)

HUMAN_TERMS = {
    "own_prev_unsafe": "own_prev_unsafe",
    "opponent_prev_unsafe": "opponent_prev_unsafe",
    "progress_gap_before": "progress_gap_before",
    "first_round_unsafe": "first_round_unsafe",
}

TEXT_HEAVY_TURN_COLUMNS = {
    "prompt",
    "raw_response",
    "reasoning",
    "attempt_history",
    "logprobs",
}

GAP_BIN_ORDER = [
    "behind_gt2",
    "behind_1_2",
    "behind_0_5_1",
    "tied_or_slight_behind",
    "ahead_0_0_5",
    "ahead_0_5_1",
    "ahead_1_2",
    "ahead_gt2",
]


@dataclass(frozen=True)
class RunFiles:
    raw_dir: Path
    provider: str
    family: str
    persona_mode: str
    experiment_mode: str
    condition: str
    model_slug: str
    races: Path | None
    players: Path | None
    turns: Path | None
    manifest: Path | None

    @property
    def source_run(self) -> str:
        return str(self.raw_dir.relative_to(REPO_ROOT)).replace("\\", "/")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=200,
        help="Bootstrap iterations for decision-tree rule stability.",
    )
    parser.add_argument(
        "--logit-bootstrap",
        type=int,
        default=100,
        help="Cluster bootstrap iterations for baseline human-check logit terms.",
    )
    parser.add_argument("--random-seed", type=int, default=260726)
    return parser.parse_args()


def discover_runs(input_root: Path) -> list[RunFiles]:
    runs: list[RunFiles] = []
    for raw_dir in sorted(input_root.rglob("raw")):
        rel = raw_dir.relative_to(input_root)
        parts = rel.parts
        if len(parts) < 7:
            continue
        provider, family, persona_mode, experiment_mode, condition, model_slug, leaf = parts[-7:]
        if leaf != "raw":
            continue
        races = raw_dir / "races.csv"
        players = raw_dir / "players.csv"
        turns = raw_dir / "turns.jsonl"
        manifest = raw_dir / "run_manifest.json"
        runs.append(
            RunFiles(
                raw_dir=raw_dir,
                provider=provider,
                family=family,
                persona_mode=persona_mode,
                experiment_mode=experiment_mode,
                condition=condition,
                model_slug=model_slug,
                races=races if races.exists() else None,
                players=players if players.exists() else None,
                turns=turns if turns.exists() else None,
                manifest=manifest if manifest.exists() else None,
            )
        )
    return runs


def add_metadata(frame: pd.DataFrame, run: RunFiles) -> pd.DataFrame:
    frame = frame.copy()
    frame["source_run"] = run.source_run
    frame["provider"] = run.provider
    frame["family"] = run.family
    frame["persona_mode"] = run.persona_mode
    frame["experiment_mode"] = run.experiment_mode
    frame["condition"] = run.condition
    frame["model_slug"] = run.model_slug
    frame["n_players"] = 2
    return frame


def read_manifest(path: Path | None) -> dict[str, Any]:
    if path is None:
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:  # pragma: no cover - reported in output table
        return {"manifest_read_error": str(exc)}


def load_tables(runs: list[RunFiles]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    coverage_rows: list[dict[str, Any]] = []
    races_parts: list[pd.DataFrame] = []
    players_parts: list[pd.DataFrame] = []
    turns_parts: list[pd.DataFrame] = []

    for run in runs:
        manifest = read_manifest(run.manifest)
        status = manifest.get("status", "__missing__")
        backend = (manifest.get("experiment") or {}).get("backend", "__missing__")
        created_utc = manifest.get("created_utc", "")

        race_rows = player_rows = turn_rows = 0
        if run.races:
            races = add_metadata(pd.read_csv(run.races), run)
            races_parts.append(races)
            race_rows = len(races)
        if run.players:
            players = add_metadata(pd.read_csv(run.players), run)
            players_parts.append(players)
            player_rows = len(players)
        if run.turns:
            turns = pd.read_json(run.turns, lines=True)
            turns = turns.drop(
                columns=[col for col in TEXT_HEAVY_TURN_COLUMNS if col in turns.columns],
                errors="ignore",
            )
            turns = add_metadata(turns, run)
            turns_parts.append(turns)
            turn_rows = len(turns)

        coverage_rows.append(
            {
                "source_run": run.source_run,
                "provider": run.provider,
                "family": run.family,
                "persona_mode": run.persona_mode,
                "experiment_mode": run.experiment_mode,
                "condition": run.condition,
                "model_slug": run.model_slug,
                "manifest_status": status,
                "backend": backend,
                "created_utc": created_utc,
                "has_races": run.races is not None,
                "has_players": run.players is not None,
                "has_turns": run.turns is not None,
                "race_rows": race_rows,
                "player_rows": player_rows,
                "turn_rows": turn_rows,
            }
        )

    races_all = pd.concat(races_parts, ignore_index=True) if races_parts else pd.DataFrame()
    players_all = pd.concat(players_parts, ignore_index=True) if players_parts else pd.DataFrame()
    turns_all = pd.concat(turns_parts, ignore_index=True) if turns_parts else pd.DataFrame()
    coverage = pd.DataFrame(coverage_rows)
    return races_all, players_all, turns_all, coverage


def as_binary(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(float)
    text = series.astype("string").str.lower().str.strip()
    mapped = text.map(
        {
            "1": 1.0,
            "true": 1.0,
            "yes": 1.0,
            "unsafe": 1.0,
            "0": 0.0,
            "false": 0.0,
            "no": 0.0,
            "safe": 0.0,
        }
    )
    numeric = pd.to_numeric(series, errors="coerce")
    return mapped.fillna(numeric)


def safe_exp(value: float) -> float:
    if not math.isfinite(value):
        return np.nan
    if value > 709:
        return np.inf
    if value < -745:
        return 0.0
    return math.exp(value)


def add_turn_features(turns: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    turns = turns.copy()
    turns["unsafe"] = as_binary(turns["unsafe"])
    turns["round"] = pd.to_numeric(turns["round"], errors="coerce").astype("Int64")
    turns["max_private_risk"] = pd.to_numeric(turns["max_private_risk"], errors="coerce")
    turns["progress_gap_before"] = pd.to_numeric(turns["progress_gap_before"], errors="coerce")
    turns["own_private_risk_before"] = pd.to_numeric(
        turns.get("own_private_risk_before"), errors="coerce"
    )
    turns["opponent_private_risk_before"] = pd.to_numeric(
        turns.get("opponent_private_risk_before"), errors="coerce"
    )
    turns["retry_count"] = pd.to_numeric(turns.get("retry_count"), errors="coerce").fillna(0)
    turns["parse_failed"] = as_binary(turns.get("parse_failed", pd.Series(False, index=turns.index))).fillna(0)

    turns["own_prev_unsafe"] = as_binary(turns.get("own_prev_action", pd.Series(pd.NA, index=turns.index)))
    turns["opponent_prev_unsafe"] = as_binary(
        turns.get("opponent_prev_action", pd.Series(pd.NA, index=turns.index))
    )

    player_col = "player_index" if "player_index" in turns.columns else "player"
    first = (
        turns.sort_values(["source_run", "game_id", player_col, "round"])
        .groupby(["source_run", "game_id", player_col], dropna=False)["unsafe"]
        .first()
        .rename("first_round_unsafe")
        .reset_index()
    )
    turns = turns.merge(first, on=["source_run", "game_id", player_col], how="left")

    turns["is_round2plus"] = turns["round"].astype(float) >= 2
    turns["gap_state"] = np.select(
        [
            turns["progress_gap_before"] < 0,
            turns["progress_gap_before"] > 0,
        ],
        ["behind", "ahead"],
        default="tied",
    )
    turns["gap_bin"] = pd.cut(
        turns["progress_gap_before"],
        bins=[-np.inf, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, np.inf],
        labels=[
            "behind_gt2",
            "behind_1_2",
            "behind_0_5_1",
            "tied_or_slight_behind",
            "ahead_0_0_5",
            "ahead_0_5_1",
            "ahead_1_2",
            "ahead_gt2",
        ],
        include_lowest=True,
        right=False,
    )

    status = coverage[["source_run", "manifest_status", "backend"]].drop_duplicates()
    turns = turns.merge(status, on="source_run", how="left")
    turn_key = (
        ["source_run", "game_id", "player_index", "round"]
        if "player_index" in turns.columns
        else ["source_run", "game_id", "player", "round"]
    )
    turns["duplicate_grain_key"] = turns.duplicated(turn_key, keep=False)
    turns["analysis_scope"] = np.where(
        (turns["persona_mode"] == "persona_none")
        & (turns["experiment_mode"] == "mode_baseline")
        & (turns["manifest_status"] == "completed"),
        "baseline_completed",
        np.where(turns["manifest_status"] == "completed", "all_completed", "incomplete_or_running"),
    )
    return turns


def add_player_features(players: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    players = players.copy()
    for col in ["unsafe_frequency", "final_payoff", "private_risk", "setback", "stage_payoff"]:
        if col in players.columns:
            players[col] = pd.to_numeric(players[col], errors="coerce")
    status = coverage[["source_run", "manifest_status", "backend"]].drop_duplicates()
    players = players.merge(status, on="source_run", how="left")
    player_key = (
        ["source_run", "game_id", "player_index"]
        if "player_index" in players.columns
        else ["source_run", "game_id", "player"]
    )
    players["duplicate_grain_key"] = players.duplicated(player_key, keep=False)
    return players


def add_race_features(races: pd.DataFrame, coverage: pd.DataFrame) -> pd.DataFrame:
    races = races.copy()
    for col in ["max_private_risk", "n_rounds", "parse_failures"]:
        if col in races.columns:
            races[col] = pd.to_numeric(races[col], errors="coerce")
    status = coverage[["source_run", "manifest_status", "backend"]].drop_duplicates()
    races = races.merge(status, on="source_run", how="left")
    races["duplicate_grain_key"] = races.duplicated(["source_run", "game_id"], keep=False)
    return races


def build_data_quality_findings(
    races: pd.DataFrame,
    players: pd.DataFrame,
    turns: pd.DataFrame,
    coverage: pd.DataFrame,
    derived_dir: Path,
) -> pd.DataFrame:
    findings: list[dict[str, Any]] = []

    def add_finding(
        *,
        check: str,
        severity: str,
        status: str,
        rows_checked: int,
        rows_affected: int,
        finding: str,
        impact: str,
        recommendation: str,
    ) -> None:
        rate = rows_affected / rows_checked if rows_checked else np.nan
        findings.append(
            {
                "check": check,
                "severity": severity,
                "status": status,
                "rows_checked": rows_checked,
                "rows_affected": rows_affected,
                "affected_rate": rate,
                "finding": finding,
                "impact": impact,
                "recommendation": recommendation,
            }
        )

    incomplete = coverage[coverage["manifest_status"] != "completed"]
    add_finding(
        check="run_completeness",
        severity="medium" if not incomplete.empty else "high",
        status="pass" if incomplete.empty else "warning",
        rows_checked=len(coverage),
        rows_affected=len(incomplete),
        finding=f"{len(incomplete)} run(s) are not marked completed.",
        impact="Incomplete runs can bias condition/model summaries if pooled silently.",
        recommendation="Use `manifest_status == completed` for primary analysis; show incomplete cells only in coverage tables.",
    )

    key_specs = [
        ("races_grain", races, ["source_run", "game_id"]),
        (
            "players_grain",
            players,
            ["source_run", "game_id", "player_index"]
            if "player_index" in players.columns
            else ["source_run", "game_id", "player"],
        ),
        (
            "turns_grain",
            turns,
            ["source_run", "game_id", "player_index", "round"]
            if "player_index" in turns.columns
            else ["source_run", "game_id", "player", "round"],
        ),
    ]
    for check, frame, keys in key_specs:
        duplicate_mask = frame.duplicated(keys, keep=False) if set(keys).issubset(frame.columns) else pd.Series(False, index=frame.index)
        duplicate_count = int(duplicate_mask.sum())
        if duplicate_count:
            duplicate_examples = (
                frame.loc[duplicate_mask, [col for col in keys if col in frame.columns]]
                .drop_duplicates()
                .head(50)
                .copy()
            )
            duplicate_examples.insert(0, "check", check)
            duplicate_examples.to_csv(
                derived_dir / f"data_quality_{check}_duplicate_examples.csv",
                index=False,
            )
        add_finding(
            check=check,
            severity="critical" if duplicate_count else "low",
            status="fail" if duplicate_count else "pass",
            rows_checked=len(frame),
            rows_affected=duplicate_count,
            finding=f"{duplicate_count} duplicated row(s) at intended grain {keys}.",
            impact="Duplicate grain would overcount races, players, or decisions.",
            recommendation="Primary analysis excludes duplicate-grain rows; inspect duplicate examples before citing affected cells.",
        )

    required_turn_fields = [
        "unsafe",
        "round",
        "max_private_risk",
        "progress_gap_before",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
        "first_round_unsafe",
    ]
    round2 = turns[turns["is_round2plus"]].copy()
    missing_required = 0
    for col in required_turn_fields:
        if col in round2.columns:
            missing_required += int(round2[col].isna().sum())
        else:
            missing_required += len(round2)
    add_finding(
        check="turn_required_fields_round2plus",
        severity="high" if missing_required else "low",
        status="fail" if missing_required else "pass",
        rows_checked=len(round2) * len(required_turn_fields),
        rows_affected=missing_required,
        finding=f"{missing_required} missing required turn-feature value(s) for round>=2 modeling.",
        impact="Missing lag/state fields reduce model rows or distort comparability.",
        recommendation="Keep model fitting on rows with complete required fields; audit source fields if this becomes nonzero.",
    )

    final_parse_failures = int(pd.to_numeric(turns["parse_failed"], errors="coerce").fillna(0).sum())
    add_finding(
        check="final_parse_failures",
        severity="high" if final_parse_failures else "low",
        status="fail" if final_parse_failures else "pass",
        rows_checked=len(turns),
        rows_affected=final_parse_failures,
        finding=f"{final_parse_failures} final turn parse failure(s).",
        impact="Unparseable decisions cannot be treated as behavioral evidence.",
        recommendation="Exclude final parse failures from behavioral analysis if any appear.",
    )

    retry_positive = int((pd.to_numeric(turns["retry_count"], errors="coerce").fillna(0) > 0).sum())
    add_finding(
        check="retry_rate",
        severity="medium" if retry_positive else "low",
        status="warning" if retry_positive else "pass",
        rows_checked=len(turns),
        rows_affected=retry_positive,
        finding=f"{retry_positive} decision(s) required at least one parse retry.",
        impact="Retry behavior can indicate prompt/model compliance differences and should be stratified.",
        recommendation="Report retry rates by model/protocol before making strong behavioral claims.",
    )

    retry_by_model = (
        turns.assign(retry_positive=(pd.to_numeric(turns["retry_count"], errors="coerce").fillna(0) > 0))
        .groupby(["provider", "experiment_mode", "model_slug"], dropna=False)
        .agg(decisions=("unsafe", "size"), retry_rate=("retry_positive", "mean"))
        .reset_index()
        .sort_values("retry_rate", ascending=False)
    )
    retry_by_model.to_csv(derived_dir / "data_quality_retry_by_model.csv", index=False)

    dq = pd.DataFrame(findings)
    dq.to_csv(derived_dir / "data_quality_findings.csv", index=False)
    return dq


def rate_summary(frame: pd.DataFrame, group_cols: list[str], value_col: str = "unsafe") -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for keys, group in frame.groupby(group_cols, dropna=False, observed=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        values = pd.to_numeric(group[value_col], errors="coerce").dropna()
        n = len(values)
        mean = float(values.mean()) if n else np.nan
        se = math.sqrt(mean * (1 - mean) / n) if n and 0 <= mean <= 1 else np.nan
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "n": n,
                "mean": mean,
                "ci95_low": max(0.0, mean - 1.96 * se) if not math.isnan(se) else np.nan,
                "ci95_high": min(1.0, mean + 1.96 * se) if not math.isnan(se) else np.nan,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return ""
    columns = list(frame.columns)
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join(["---"] * len(columns)) + " |",
    ]
    for _, row in frame.iterrows():
        values = []
        for column in columns:
            value = row[column]
            if isinstance(value, float):
                values.append(f"{value:.4g}")
            else:
                values.append(str(value))
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def build_descriptive_outputs(
    turns: pd.DataFrame,
    players: pd.DataFrame,
    derived_dir: Path,
) -> dict[str, pd.DataFrame]:
    clean_turns = turns[
        (turns["manifest_status"] == "completed") & (~turns["duplicate_grain_key"])
    ].copy()
    round2 = clean_turns[clean_turns["is_round2plus"]].copy()
    outputs = {
        "unsafe_by_risk_model_turn": rate_summary(
            clean_turns,
            [
                "provider",
                "family",
                "persona_mode",
                "experiment_mode",
                "condition",
                "model_slug",
                "max_private_risk",
            ],
        ),
        "unsafe_by_gap_bin_turn": rate_summary(
            round2,
            ["analysis_scope", "provider", "experiment_mode", "model_slug", "gap_bin"],
        ),
        "unsafe_by_lag_profile_turn": rate_summary(
            round2,
            [
                "analysis_scope",
                "provider",
                "experiment_mode",
                "model_slug",
                "own_prev_unsafe",
                "opponent_prev_unsafe",
            ],
        ),
        "first_round_persistence_turn": rate_summary(
            round2,
            ["analysis_scope", "provider", "experiment_mode", "model_slug", "first_round_unsafe"],
        ),
    }
    if not players.empty:
        clean_players = players[
            (players["manifest_status"] == "completed")
            & (~players["duplicate_grain_key"])
        ].copy()
        player_group = [
            "provider",
            "family",
            "persona_mode",
            "experiment_mode",
            "condition",
            "model_slug",
            "max_private_risk",
        ]
        outputs["unsafe_by_risk_model_player"] = (
            clean_players
            .groupby(player_group, dropna=False)
            .agg(
                players=("player", "count"),
                unsafe_frequency_mean=("unsafe_frequency", "mean"),
                unsafe_frequency_sd=("unsafe_frequency", "std"),
                final_payoff_mean=("final_payoff", "mean"),
                final_payoff_sd=("final_payoff", "std"),
                setback_rate=("setback", "mean"),
            )
            .reset_index()
        )
    for name, frame in outputs.items():
        frame.to_csv(derived_dir / f"{name}.csv", index=False)
    build_gap_threshold_scan(turns, derived_dir)
    return outputs


def build_gap_threshold_scan(turns: pd.DataFrame, derived_dir: Path) -> pd.DataFrame:
    """Scan simple progress-gap thresholds as interpretable cut points."""

    rows: list[dict[str, Any]] = []
    candidates = [-2.0, -1.5, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]
    base = turns[
        turns["is_round2plus"]
        & (turns["manifest_status"] == "completed")
        & (~turns["duplicate_grain_key"])
        & turns["progress_gap_before"].notna()
        & turns["unsafe"].notna()
    ].copy()
    groupings = [
        ("baseline_completed", ["analysis_scope"]),
        ("by_model", ["analysis_scope", "provider", "experiment_mode", "model_slug"]),
    ]
    for grouping_name, group_cols in groupings:
        data = base
        if grouping_name == "baseline_completed":
            data = data[data["analysis_scope"] == "baseline_completed"]
        for keys, group in data.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            key_values = {col: value for col, value in zip(group_cols, keys)}
            for threshold in candidates:
                low = group[group["progress_gap_before"] <= threshold]["unsafe"].astype(float)
                high = group[group["progress_gap_before"] > threshold]["unsafe"].astype(float)
                if len(low) < 20 or len(high) < 20:
                    continue
                low_mean = float(low.mean())
                high_mean = float(high.mean())
                se = math.sqrt(
                    low_mean * (1 - low_mean) / len(low)
                    + high_mean * (1 - high_mean) / len(high)
                )
                diff = low_mean - high_mean
                rows.append(
                    {
                        "grouping": grouping_name,
                        **key_values,
                        "threshold_rule": f"progress_gap_before <= {threshold:g}",
                        "threshold": threshold,
                        "n_low_or_equal": len(low),
                        "n_above": len(high),
                        "unsafe_low_or_equal": low_mean,
                        "unsafe_above": high_mean,
                        "diff_low_minus_above": diff,
                        "diff_ci95_low": diff - 1.96 * se,
                        "diff_ci95_high": diff + 1.96 * se,
                    }
                )
    scan = pd.DataFrame(rows)
    scan.to_csv(derived_dir / "gap_threshold_scan.csv", index=False)
    return scan


def safe_import_statsmodels():
    try:
        import statsmodels.formula.api as smf  # type: ignore
        import statsmodels.api as sm  # type: ignore

        return smf, sm
    except Exception:
        return None, None


def fit_human_check(
    turns: pd.DataFrame,
    derived_dir: Path,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    smf, sm = safe_import_statsmodels()
    if smf is None or sm is None:
        skipped = pd.DataFrame(
            [{"stage": "human_check", "reason": "statsmodels is not installed"}]
        )
        skipped.to_csv(derived_dir / "human_check_skipped.csv", index=False)
        return pd.DataFrame(), skipped

    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    scopes = {
        "baseline_completed": turns[
            (turns["analysis_scope"] == "baseline_completed")
            & turns["is_round2plus"]
            & (~turns["duplicate_grain_key"])
        ].copy(),
        "all_completed": turns[
            (turns["manifest_status"] == "completed")
            & turns["is_round2plus"]
            & (~turns["duplicate_grain_key"])
        ].copy(),
    }
    for scope_name, scope in scopes.items():
        model_frame = scope.dropna(
            subset=[
                "unsafe",
                "max_private_risk",
                "own_prev_unsafe",
                "opponent_prev_unsafe",
                "progress_gap_before",
                "first_round_unsafe",
                "model_slug",
                "game_id",
            ]
        ).copy()
        metadata_rows.append(
            {
                "scope": scope_name,
                "decisions": len(model_frame),
                "clusters": model_frame[["source_run", "game_id"]].drop_duplicates().shape[0],
                "models": model_frame["model_slug"].nunique(),
            }
        )
        if len(model_frame) < 50 or model_frame["unsafe"].nunique() < 2:
            continue
        model_frame["cluster_id"] = model_frame["source_run"] + "::" + model_frame["game_id"].astype(str)
        for spec_name, formula in LOGIT_SPECS:
            try:
                fit = smf.glm(formula=formula, data=model_frame, family=sm.families.Binomial()).fit(
                    cov_type="cluster",
                    cov_kwds={"groups": model_frame["cluster_id"]},
                )
            except Exception as exc:
                rows.append(
                    {
                        "scope": scope_name,
                        "spec": spec_name,
                        "term": "__fit_error__",
                        "error": str(exc),
                    }
                )
                continue
            conf = fit.conf_int()
            for term, coef in fit.params.items():
                low, high = conf.loc[term].tolist()
                rows.append(
                    {
                        "scope": scope_name,
                        "spec": spec_name,
                        "term": term,
                        "coef": coef,
                        "odds_ratio": safe_exp(coef),
                        "ci95_low": low,
                        "ci95_high": high,
                        "or_ci95_low": safe_exp(low),
                        "or_ci95_high": safe_exp(high),
                        "p_value": fit.pvalues.get(term, np.nan),
                        "n": int(fit.nobs),
                        "pseudo_r2_mcfadden": 1 - (fit.llf / fit.llnull)
                        if fit.llnull
                        else np.nan,
                        "error": "",
                    }
                )

    coefficients = pd.DataFrame(rows)
    metadata = pd.DataFrame(metadata_rows)
    coefficients.to_csv(derived_dir / "human_check_logit_coefficients.csv", index=False)
    metadata.to_csv(derived_dir / "human_check_logit_metadata.csv", index=False)
    build_human_ledger(coefficients, derived_dir)
    build_human_check_segments(turns, derived_dir, smf, sm)
    bootstrap_baseline_human_check(turns, derived_dir, smf, sm, bootstrap=bootstrap, seed=seed)
    build_baseline_prediction_grid(turns, derived_dir, smf, sm)
    return coefficients, metadata


def build_human_check_segments(
    turns: pd.DataFrame,
    derived_dir: Path,
    smf: Any,
    sm: Any,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Fit the core human-style model within readable segments.

    This is not a substitute for the pooled cluster-robust model. It answers a
    simpler stability question: do the human-reference terms keep the same sign
    across model/provider slices?
    """

    rows: list[dict[str, Any]] = []
    metadata_rows: list[dict[str, Any]] = []
    formula = (
        "unsafe ~ C(max_private_risk) + first_round_unsafe + own_prev_unsafe * "
        "opponent_prev_unsafe * progress_gap_before"
    )
    base = turns[
        (turns["analysis_scope"] == "baseline_completed")
        & turns["is_round2plus"]
        & (~turns["duplicate_grain_key"])
    ].dropna(
        subset=[
            "unsafe",
            "max_private_risk",
            "own_prev_unsafe",
            "opponent_prev_unsafe",
            "progress_gap_before",
            "first_round_unsafe",
            "model_slug",
            "game_id",
        ]
    )
    segments: list[tuple[str, str, pd.DataFrame]] = [
        ("baseline_provider", str(provider), group.copy())
        for provider, group in base.groupby("provider", dropna=False)
    ]
    segments.extend(
        [
            ("baseline_model", str(model), group.copy())
            for model, group in base.groupby("model_slug", dropna=False)
        ]
    )
    for segment_type, segment, frame in segments:
        clusters = frame[["source_run", "game_id"]].drop_duplicates().shape[0]
        metadata_rows.append(
            {
                "segment_type": segment_type,
                "segment": segment,
                "decisions": len(frame),
                "clusters": clusters,
                "unsafe_rate": frame["unsafe"].mean() if len(frame) else np.nan,
            }
        )
        if len(frame) < 100 or frame["unsafe"].nunique() < 2 or clusters < 10:
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment": segment,
                    "term": "__skipped__",
                    "error": "insufficient variation or clusters",
                }
            )
            continue
        frame = frame.copy()
        frame["cluster_id"] = frame["source_run"] + "::" + frame["game_id"].astype(str)
        try:
            fit = smf.glm(formula=formula, data=frame, family=sm.families.Binomial()).fit(
                cov_type="cluster",
                cov_kwds={"groups": frame["cluster_id"]},
            )
        except Exception as exc:
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment": segment,
                    "term": "__fit_error__",
                    "error": str(exc),
                }
            )
            continue
        conf = fit.conf_int()
        for term in HUMAN_TERMS.values():
            if term not in fit.params:
                continue
            low, high = conf.loc[term].tolist()
            coef = float(fit.params[term])
            rows.append(
                {
                    "segment_type": segment_type,
                    "segment": segment,
                    "term": term,
                    "coef": coef,
                    "odds_ratio": safe_exp(coef),
                    "ci95_low": low,
                    "ci95_high": high,
                    "p_value": fit.pvalues.get(term, np.nan),
                    "n": int(fit.nobs),
                    "error": "",
                }
            )
    segment_coefficients = pd.DataFrame(rows)
    segment_metadata = pd.DataFrame(metadata_rows)
    segment_coefficients.to_csv(derived_dir / "human_check_segment_coefficients.csv", index=False)
    segment_metadata.to_csv(derived_dir / "human_check_segment_metadata.csv", index=False)
    build_human_sign_stability(segment_coefficients, derived_dir)
    return segment_coefficients, segment_metadata


def build_human_sign_stability(segment_coefficients: pd.DataFrame, derived_dir: Path) -> pd.DataFrame:
    if segment_coefficients.empty or not HUMAN_REFERENCE_PATH.exists():
        return pd.DataFrame()
    reference = json.loads(HUMAN_REFERENCE_PATH.read_text(encoding="utf-8"))
    expected_by_term = {
        HUMAN_TERMS.get(effect.get("name")): effect.get("expected_sign", "")
        for effect in reference.get("effects", [])
        if HUMAN_TERMS.get(effect.get("name"))
    }
    rows: list[dict[str, Any]] = []
    usable = segment_coefficients[
        (segment_coefficients["segment_type"] == "baseline_model")
        & (~segment_coefficients["term"].astype(str).str.startswith("__"))
    ].copy()
    for term, group in usable.groupby("term"):
        expected = expected_by_term.get(term, "")
        if expected == "positive":
            matches = group["coef"].astype(float) > 0
        elif expected == "negative":
            matches = group["coef"].astype(float) < 0
        else:
            matches = pd.Series([np.nan] * len(group), index=group.index)
        rows.append(
            {
                "term": term,
                "expected_sign": expected,
                "segments": len(group),
                "sign_match_count": int(matches.sum()) if expected else np.nan,
                "sign_match_share": float(matches.mean()) if expected else np.nan,
                "median_coef": float(group["coef"].median()),
                "min_coef": float(group["coef"].min()),
                "max_coef": float(group["coef"].max()),
                "large_abs_coef_count": int((group["coef"].abs() > 10).sum()),
                "stability_note": "large coefficients suggest separation; use sign stability only"
                if (group["coef"].abs() > 10).any()
                else "",
            }
        )
    stability = pd.DataFrame(rows)
    stability.to_csv(derived_dir / "human_check_sign_stability.csv", index=False)
    return stability


def baseline_human_model_frame(turns: pd.DataFrame) -> pd.DataFrame:
    required = [
        "unsafe",
        "max_private_risk",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
        "progress_gap_before",
        "first_round_unsafe",
        "game_id",
    ]
    return turns[
        (turns["analysis_scope"] == "baseline_completed")
        & turns["is_round2plus"]
        & (~turns["duplicate_grain_key"])
    ].dropna(subset=required).copy()


def bootstrap_baseline_human_check(
    turns: pd.DataFrame,
    derived_dir: Path,
    smf: Any,
    sm: Any,
    *,
    bootstrap: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if bootstrap <= 0:
        return pd.DataFrame(), pd.DataFrame()
    model_frame = baseline_human_model_frame(turns)
    if len(model_frame) < 100 or model_frame["unsafe"].nunique() < 2:
        return pd.DataFrame(), pd.DataFrame()

    rng = np.random.default_rng(seed)
    model_frame["cluster_id"] = model_frame["source_run"] + "::" + model_frame["game_id"].astype(str)
    clusters = np.array(sorted(model_frame["cluster_id"].unique()))
    by_cluster = {cluster: group for cluster, group in model_frame.groupby("cluster_id", sort=False)}
    rows: list[dict[str, Any]] = []
    terms = list(HUMAN_TERMS.values())
    for iteration in range(bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample = pd.concat([by_cluster[cluster] for cluster in sampled], ignore_index=True)
        if sample["unsafe"].nunique() < 2:
            continue
        try:
            fit = smf.glm(
                formula=HUMAN_CHECK_FORMULA,
                data=sample,
                family=sm.families.Binomial(),
            ).fit()
        except Exception as exc:
            rows.append(
                {
                    "bootstrap": iteration,
                    "term": "__fit_error__",
                    "coef": np.nan,
                    "error": str(exc),
                }
            )
            continue
        for term in terms:
            if term not in fit.params:
                continue
            rows.append(
                {
                    "bootstrap": iteration,
                    "term": term,
                    "coef": float(fit.params[term]),
                    "error": "",
                }
            )
    samples = pd.DataFrame(rows)
    samples.to_csv(derived_dir / "human_check_baseline_bootstrap_coefficients.csv", index=False)
    usable = samples[~samples["term"].astype(str).str.startswith("__")].copy()
    if usable.empty:
        return samples, pd.DataFrame()

    expected_by_term = {}
    if HUMAN_REFERENCE_PATH.exists():
        reference = json.loads(HUMAN_REFERENCE_PATH.read_text(encoding="utf-8"))
        expected_by_term = {
            HUMAN_TERMS.get(effect.get("name")): effect.get("expected_sign", "")
            for effect in reference.get("effects", [])
            if HUMAN_TERMS.get(effect.get("name"))
        }
    summary_rows: list[dict[str, Any]] = []
    for term, group in usable.groupby("term"):
        coef = pd.to_numeric(group["coef"], errors="coerce").dropna()
        expected = expected_by_term.get(term, "")
        if coef.empty:
            continue
        if expected == "positive":
            sign_matches = coef > 0
        elif expected == "negative":
            sign_matches = coef < 0
        else:
            sign_matches = pd.Series([np.nan] * len(coef), index=coef.index)
        summary_rows.append(
            {
                "scope": "baseline_completed",
                "spec": "M3_interactions",
                "term": term,
                "bootstrap_successes": len(coef),
                "coef_median": float(coef.median()),
                "coef_ci95_low": float(coef.quantile(0.025)),
                "coef_ci95_high": float(coef.quantile(0.975)),
                "odds_ratio_median": safe_exp(float(coef.median())),
                "or_ci95_low": safe_exp(float(coef.quantile(0.025))),
                "or_ci95_high": safe_exp(float(coef.quantile(0.975))),
                "expected_sign": expected,
                "sign_match_share": float(sign_matches.mean()) if expected else np.nan,
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(derived_dir / "human_check_baseline_bootstrap_summary.csv", index=False)
    return samples, summary


def build_baseline_prediction_grid(
    turns: pd.DataFrame,
    derived_dir: Path,
    smf: Any,
    sm: Any,
) -> pd.DataFrame:
    model_frame = baseline_human_model_frame(turns)
    if len(model_frame) < 100 or model_frame["unsafe"].nunique() < 2:
        return pd.DataFrame()
    try:
        fit = smf.glm(
            formula=HUMAN_CHECK_FORMULA,
            data=model_frame,
            family=sm.families.Binomial(),
        ).fit()
    except Exception:
        return pd.DataFrame()

    gaps = np.arange(-2.0, 2.01, 0.25)
    risk_values = sorted(model_frame["max_private_risk"].dropna().unique())
    first_round_values = [0.0, 1.0]
    rows: list[dict[str, Any]] = []
    for risk in risk_values:
        for first_round in first_round_values:
            for own_prev in [0.0, 1.0]:
                for opponent_prev in [0.0, 1.0]:
                    for gap in gaps:
                        rows.append(
                            {
                                "max_private_risk": risk,
                                "first_round_unsafe": first_round,
                                "own_prev_unsafe": own_prev,
                                "opponent_prev_unsafe": opponent_prev,
                                "progress_gap_before": gap,
                                "lag_profile": f"own{int(own_prev)}_opp{int(opponent_prev)}",
                            }
                        )
    grid = pd.DataFrame(rows)
    pred = fit.get_prediction(grid).summary_frame(alpha=0.05)
    grid["predicted_unsafe"] = pred["mean"].to_numpy()
    grid["pred_ci95_low"] = pred["mean_ci_lower"].to_numpy()
    grid["pred_ci95_high"] = pred["mean_ci_upper"].to_numpy()
    grid.to_csv(derived_dir / "human_check_baseline_prediction_grid.csv", index=False)
    return grid


def build_human_ledger(coefficients: pd.DataFrame, derived_dir: Path) -> pd.DataFrame:
    if coefficients.empty or not HUMAN_REFERENCE_PATH.exists():
        return pd.DataFrame()
    reference = json.loads(HUMAN_REFERENCE_PATH.read_text(encoding="utf-8"))
    effects = reference.get("effects", [])
    rows: list[dict[str, Any]] = []
    spec = coefficients[
        (coefficients["scope"] == "baseline_completed")
        & (coefficients["spec"] == "M3_interactions")
    ].copy()
    for effect in effects:
        name = effect.get("name")
        term = HUMAN_TERMS.get(name)
        if not term:
            continue
        hit = spec[spec["term"] == term]
        if hit.empty:
            rows.append(
                {
                    "effect_id": effect.get("id"),
                    "effect_name": name,
                    "status": "missing_term",
                    "human_reference": effect.get("human_value"),
                    "expected_sign": effect.get("expected_sign", ""),
                }
            )
            continue
        row = hit.iloc[0].to_dict()
        coef = float(row.get("coef", np.nan))
        expected = effect.get("expected_sign", "")
        sign_match = (
            (expected == "positive" and coef > 0)
            or (expected == "negative" and coef < 0)
            or expected == ""
        )
        rows.append(
            {
                "effect_id": effect.get("id"),
                "effect_name": name,
                "human_reference": effect.get("human_value"),
                "expected_sign": expected,
                "llm_coef": coef,
                "llm_odds_ratio": row.get("odds_ratio"),
                "llm_ci95_low": row.get("ci95_low"),
                "llm_ci95_high": row.get("ci95_high"),
                "llm_p_value": row.get("p_value"),
                "sign_match": sign_match,
                "scope": "baseline_completed",
                "spec": "M3_interactions",
                "status": "estimated",
            }
        )
    ledger = pd.DataFrame(rows)
    ledger.to_csv(derived_dir / "human_reference_ledger.csv", index=False)
    return ledger


def safe_import_sklearn():
    try:
        from sklearn.compose import ColumnTransformer  # type: ignore
        from sklearn.metrics import (  # type: ignore
            accuracy_score,
            balanced_accuracy_score,
            brier_score_loss,
            roc_auc_score,
        )
        from sklearn.model_selection import GroupKFold  # type: ignore
        from sklearn.pipeline import Pipeline  # type: ignore
        from sklearn.preprocessing import OneHotEncoder  # type: ignore
        from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree  # type: ignore

        return {
            "ColumnTransformer": ColumnTransformer,
            "accuracy_score": accuracy_score,
            "balanced_accuracy_score": balanced_accuracy_score,
            "brier_score_loss": brier_score_loss,
            "roc_auc_score": roc_auc_score,
            "GroupKFold": GroupKFold,
            "Pipeline": Pipeline,
            "OneHotEncoder": OneHotEncoder,
            "DecisionTreeClassifier": DecisionTreeClassifier,
            "export_text": export_text,
            "plot_tree": plot_tree,
        }
    except Exception:
        return None


def fit_decision_tree(turns: pd.DataFrame, derived_dir: Path, figures_dir: Path, bootstrap: int, seed: int) -> None:
    sk = safe_import_sklearn()
    if sk is None:
        pd.DataFrame([{"stage": "decision_tree", "reason": "scikit-learn is not installed"}]).to_csv(
            derived_dir / "decision_tree_skipped.csv", index=False
        )
        return

    complete_frame = turns[
        (turns["manifest_status"] == "completed")
        & turns["is_round2plus"]
        & (~turns["duplicate_grain_key"])
    ].dropna(
        subset=[
            "unsafe",
            "progress_gap_before",
            "own_prev_unsafe",
            "opponent_prev_unsafe",
            "first_round_unsafe",
            "own_private_risk_before",
            "opponent_private_risk_before",
            "round",
            "max_private_risk",
            "model_slug",
            "provider",
            "persona_mode",
            "experiment_mode",
            "condition",
            "game_id",
        ]
    )
    scope_frames = {
        "baseline_completed": complete_frame[
            complete_frame["analysis_scope"] == "baseline_completed"
        ].copy(),
        "all_completed": complete_frame.copy(),
    }
    summary_rows: list[dict[str, Any]] = []
    for scope_name, frame in scope_frames.items():
        if frame.empty or frame["unsafe"].nunique() < 2:
            continue
        summary_rows.extend(
            fit_decision_tree_scope(
                scope_name=scope_name,
                frame=frame,
                derived_dir=derived_dir,
                figures_dir=figures_dir,
                bootstrap=bootstrap,
                seed=seed,
                sk=sk,
                write_legacy_aliases=(scope_name == "all_completed"),
            )
        )
    pd.DataFrame(summary_rows).to_csv(derived_dir / "decision_tree_scope_summary.csv", index=False)


def fit_decision_tree_scope(
    *,
    scope_name: str,
    frame: pd.DataFrame,
    derived_dir: Path,
    figures_dir: Path,
    bootstrap: int,
    seed: int,
    sk: dict[str, Any],
    write_legacy_aliases: bool = False,
) -> list[dict[str, Any]]:
    if frame.empty or frame["unsafe"].nunique() < 2:
        return []

    numeric_features = [
        "progress_gap_before",
        "own_prev_unsafe",
        "opponent_prev_unsafe",
        "first_round_unsafe",
        "own_private_risk_before",
        "opponent_private_risk_before",
        "round",
        "max_private_risk",
    ]
    categorical_features = [
        "model_slug",
        "provider",
        "persona_mode",
        "experiment_mode",
        "condition",
    ]
    X = frame[numeric_features + categorical_features]
    y = frame["unsafe"].astype(int)
    groups = frame["source_run"] + "::" + frame["game_id"].astype(str)

    preprocessor = sk["ColumnTransformer"](
        transformers=[
            ("num", "passthrough", numeric_features),
            ("cat", sk["OneHotEncoder"](handle_unknown="ignore"), categorical_features),
        ]
    )
    tree = sk["DecisionTreeClassifier"](
        max_depth=4,
        min_samples_leaf=75,
        random_state=seed,
        class_weight="balanced",
    )
    pipeline = sk["Pipeline"]([("prep", preprocessor), ("tree", tree)])

    n_splits = min(5, groups.nunique())
    metrics_rows: list[dict[str, Any]] = []
    if n_splits >= 2:
        gkf = sk["GroupKFold"](n_splits=n_splits)
        for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups), start=1):
            pipeline.fit(X.iloc[train_idx], y.iloc[train_idx])
            pred = pipeline.predict(X.iloc[test_idx])
            proba = pipeline.predict_proba(X.iloc[test_idx])[:, 1]
            fold_y = y.iloc[test_idx]
            metrics_rows.append(
                {
                    "scope": scope_name,
                    "fold": fold,
                    "n_train": len(train_idx),
                    "n_test": len(test_idx),
                    "accuracy": sk["accuracy_score"](fold_y, pred),
                    "balanced_accuracy": sk["balanced_accuracy_score"](fold_y, pred),
                    "roc_auc": sk["roc_auc_score"](fold_y, proba)
                    if fold_y.nunique() == 2
                    else np.nan,
                    "brier_score": sk["brier_score_loss"](fold_y, proba),
                }
            )
    metrics = pd.DataFrame(metrics_rows)
    metrics.to_csv(derived_dir / f"decision_tree_{scope_name}_cv_metrics.csv", index=False)
    if write_legacy_aliases:
        metrics.to_csv(derived_dir / "decision_tree_cv_metrics.csv", index=False)

    pipeline.fit(X, y)
    feature_names = get_feature_names(pipeline, numeric_features, categorical_features)
    text_rules = sk["export_text"](
        pipeline.named_steps["tree"],
        feature_names=list(feature_names),
        decimals=3,
        max_depth=4,
    )
    (derived_dir / f"decision_tree_{scope_name}_rules.txt").write_text(text_rules, encoding="utf-8")
    if write_legacy_aliases:
        (derived_dir / "decision_tree_rules.txt").write_text(text_rules, encoding="utf-8")
    importances = pd.DataFrame(
        {
            "feature": feature_names,
            "importance": pipeline.named_steps["tree"].feature_importances_,
        }
    ).sort_values("importance", ascending=False)
    importances.to_csv(derived_dir / f"decision_tree_{scope_name}_feature_importance.csv", index=False)
    if write_legacy_aliases:
        importances.to_csv(derived_dir / "decision_tree_feature_importance.csv", index=False)
    leaf_summary = build_tree_leaf_summary(pipeline, X, y, frame, derived_dir, scope_name)
    if write_legacy_aliases:
        leaf_summary.to_csv(derived_dir / "decision_tree_leaf_summary.csv", index=False)

    transformed = pipeline.named_steps["prep"].transform(X)
    root_stability = bootstrap_tree_rules(
        transformed,
        y.to_numpy(),
        groups.to_numpy(),
        feature_names,
        derived_dir,
        bootstrap,
        seed,
        sk,
        scope_name,
    )
    if write_legacy_aliases and root_stability is not None:
        root_stability.to_csv(derived_dir / "decision_tree_root_stability.csv", index=False)
        roots = pd.read_csv(derived_dir / f"decision_tree_{scope_name}_bootstrap_roots.csv")
        roots.to_csv(derived_dir / "decision_tree_bootstrap_roots.csv", index=False)
    plot_decision_tree_visual(pipeline, feature_names, figures_dir, scope_name, sk)
    plot_tree_importance(importances, figures_dir, scope_name)
    if write_legacy_aliases:
        plot_decision_tree_visual(pipeline, feature_names, figures_dir, "all", sk)
        plot_tree_importance(importances, figures_dir, "all")
    summary: list[dict[str, Any]] = []
    if not metrics.empty:
        mean_metrics = metrics.drop(columns=["fold", "scope"], errors="ignore").mean(numeric_only=True)
        summary.append(
            {
                "scope": scope_name,
                "decisions": len(frame),
                "clusters": groups.nunique(),
                "unsafe_rate": float(y.mean()),
                "cv_accuracy_mean": mean_metrics.get("accuracy", np.nan),
                "cv_balanced_accuracy_mean": mean_metrics.get("balanced_accuracy", np.nan),
                "cv_roc_auc_mean": mean_metrics.get("roc_auc", np.nan),
                "cv_brier_score_mean": mean_metrics.get("brier_score", np.nan),
                "top_importance_feature": importances.iloc[0]["feature"] if not importances.empty else "",
                "top_importance": importances.iloc[0]["importance"] if not importances.empty else np.nan,
                "top_root_feature": root_stability.iloc[0]["root_feature"]
                if root_stability is not None and not root_stability.empty
                else "",
                "top_root_share": root_stability.iloc[0]["share"]
                if root_stability is not None and not root_stability.empty
                else np.nan,
            }
        )
    return summary


def build_tree_leaf_summary(
    pipeline: Any,
    X: pd.DataFrame,
    y: pd.Series,
    frame: pd.DataFrame,
    derived_dir: Path,
    scope_name: str,
) -> pd.DataFrame:
    transformed = pipeline.named_steps["prep"].transform(X)
    leaves = pipeline.named_steps["tree"].apply(transformed)
    summary = (
        pd.DataFrame(
            {
                "leaf_id": leaves,
                "unsafe": y.to_numpy(),
                "provider": frame["provider"].to_numpy(),
                "experiment_mode": frame["experiment_mode"].to_numpy(),
                "model_slug": frame["model_slug"].to_numpy(),
                "progress_gap_before": frame["progress_gap_before"].to_numpy(),
            }
        )
        .groupby("leaf_id")
        .agg(
            n=("unsafe", "size"),
            unsafe_rate=("unsafe", "mean"),
            provider_mode_count=("provider", "nunique"),
            experiment_mode_count=("experiment_mode", "nunique"),
            model_count=("model_slug", "nunique"),
            gap_min=("progress_gap_before", "min"),
            gap_max=("progress_gap_before", "max"),
        )
        .reset_index()
        .sort_values(["unsafe_rate", "n"], ascending=[False, False])
    )
    total = summary["n"].sum()
    summary["support_share"] = summary["n"] / total if total else np.nan
    summary.insert(0, "scope", scope_name)
    summary.to_csv(derived_dir / f"decision_tree_{scope_name}_leaf_summary.csv", index=False)
    return summary


def get_feature_names(pipeline: Any, numeric_features: list[str], categorical_features: list[str]) -> np.ndarray:
    prep = pipeline.named_steps["prep"]
    try:
        cat = prep.named_transformers_["cat"].get_feature_names_out(categorical_features)
        return np.array([*numeric_features, *cat])
    except Exception:
        return np.array(numeric_features)


def bootstrap_tree_rules(
    transformed_X: Any,
    y: np.ndarray,
    cluster_ids: np.ndarray,
    feature_names: np.ndarray,
    derived_dir: Path,
    bootstrap: int,
    seed: int,
    sk: dict[str, Any],
    scope_name: str,
) -> pd.DataFrame | None:
    if bootstrap <= 0:
        return None
    rng = np.random.default_rng(seed)
    clusters = np.array(sorted(pd.Series(cluster_ids).unique()))
    by_cluster = {
        cluster: np.flatnonzero(cluster_ids == cluster)
        for cluster in clusters
    }
    root_rows: list[dict[str, Any]] = []
    for i in range(bootstrap):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        sample_idx = np.concatenate([by_cluster[cluster] for cluster in sampled])
        sample_y = y[sample_idx].astype(int)
        if len(np.unique(sample_y)) < 2:
            continue
        tree = sk["DecisionTreeClassifier"](
            max_depth=3,
            min_samples_leaf=75,
            random_state=seed + i,
            class_weight="balanced",
        )
        tree.fit(transformed_X[sample_idx], sample_y)
        root_feature_index = tree.tree_.feature[0]
        if root_feature_index >= 0:
            root_rows.append(
                {
                    "bootstrap": i,
                    "root_feature": feature_names[root_feature_index],
                    "root_threshold": tree.tree_.threshold[0],
                }
            )
    root = pd.DataFrame(root_rows)
    root.to_csv(derived_dir / f"decision_tree_{scope_name}_bootstrap_roots.csv", index=False)
    if not root.empty:
        stability = (
            root.groupby("root_feature")
            .agg(count=("root_feature", "size"), mean_threshold=("root_threshold", "mean"))
            .assign(share=lambda d: d["count"] / len(root))
            .sort_values("share", ascending=False)
            .reset_index()
        )
        stability.to_csv(derived_dir / f"decision_tree_{scope_name}_root_stability.csv", index=False)
        return stability
    return root


def safe_import_matplotlib():
    try:
        import matplotlib.pyplot as plt  # type: ignore

        return plt
    except Exception:
        return None


def plot_outputs(outputs: dict[str, pd.DataFrame], figures_dir: Path) -> None:
    plt = safe_import_matplotlib()
    if plt is None:
        return

    risk = outputs.get("unsafe_by_risk_model_turn", pd.DataFrame()).copy()
    if not risk.empty:
        base = risk[
            (risk["persona_mode"] == "persona_none")
            & (risk["experiment_mode"] == "mode_baseline")
        ].copy()
        if not base.empty:
            base["risk_label"] = base["max_private_risk"].map(lambda value: f"{float(value):.1f}")
            base["label"] = base["model_slug"] + " | risk " + base["risk_label"]
            base = base.sort_values(["model_slug", "max_private_risk"])
            fig, ax = plt.subplots(figsize=(10, 7))
            y_pos = np.arange(len(base))
            ax.barh(y_pos, base["mean"], color="#3f7f8c")
            ax.errorbar(
                base["mean"],
                y_pos,
                xerr=[base["mean"] - base["ci95_low"], base["ci95_high"] - base["mean"]],
                fmt="none",
                ecolor="#263238",
                capsize=3,
            )
            ax.set_xlabel("Unsafe rate")
            ax.set_ylabel("")
            ax.set_yticks(y_pos, base["label"])
            ax.set_title("Baseline unsafe rate by model and max private risk")
            ax.set_xlim(0, 1)
            ax.invert_yaxis()
            fig.tight_layout()
            fig.savefig(figures_dir / "baseline_unsafe_by_risk_model.png", dpi=180)
            plt.close(fig)

    gap = outputs.get("unsafe_by_gap_bin_turn", pd.DataFrame()).copy()
    if not gap.empty:
        base_gap = gap[gap["analysis_scope"] == "baseline_completed"].copy()
        if not base_gap.empty:
            fig, ax = plt.subplots(figsize=(11, 5))
            x_positions = {label: i for i, label in enumerate(GAP_BIN_ORDER)}
            for model, group in base_gap.groupby("model_slug"):
                group = group[group["gap_bin"].astype(str).isin(GAP_BIN_ORDER)].copy()
                group["gap_order"] = group["gap_bin"].astype(str).map(x_positions)
                group = group.sort_values("gap_order")
                ax.scatter(group["gap_order"], group["mean"], s=55, label=model)
            ax.set_ylabel("Unsafe rate")
            ax.set_xlabel("Progress gap bin before decision")
            ax.set_title("Baseline unsafe rate by progress gap")
            ax.set_xticks(range(len(GAP_BIN_ORDER)), GAP_BIN_ORDER)
            ax.tick_params(axis="x", rotation=45)
            ax.set_ylim(0, 1)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures_dir / "baseline_unsafe_by_gap_bin.png", dpi=180)
            plt.close(fig)

    threshold_path = figures_dir.parent / "derived" / "gap_threshold_scan.csv"
    if threshold_path.exists():
        scan = pd.read_csv(threshold_path)
        base_scan = scan[
            (scan["grouping"] == "by_model")
            & (scan["analysis_scope"] == "baseline_completed")
        ].copy()
        if not base_scan.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for model, group in base_scan.groupby("model_slug"):
                group = group.sort_values("threshold")
                ax.plot(
                    group["threshold"],
                    group["diff_low_minus_above"],
                    marker="o",
                    label=model,
                )
            ax.axhline(0, color="#555555", linewidth=1)
            ax.set_ylabel("Unsafe rate difference")
            ax.set_xlabel("Threshold: progress_gap_before <= x")
            ax.set_title("Baseline gap-threshold unsafe difference")
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figures_dir / "baseline_gap_threshold_scan.png", dpi=180)
            plt.close(fig)

    lag = outputs.get("unsafe_by_lag_profile_turn", pd.DataFrame()).copy()
    if not lag.empty:
        base_lag = lag[lag["analysis_scope"] == "baseline_completed"].copy()
        if not base_lag.empty:
            pivot = base_lag.pivot_table(
                index="own_prev_unsafe",
                columns="opponent_prev_unsafe",
                values="mean",
                aggfunc="mean",
            )
            fig, ax = plt.subplots(figsize=(5.5, 4.5))
            image = ax.imshow(pivot.values, vmin=0, vmax=1, cmap="viridis")
            ax.set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
            ax.set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
            ax.set_xlabel("Opponent previous unsafe")
            ax.set_ylabel("Own previous unsafe")
            ax.set_title("Baseline lag-action unsafe matrix")
            for y in range(pivot.shape[0]):
                for x in range(pivot.shape[1]):
                    ax.text(x, y, f"{pivot.values[y, x]:.2f}", ha="center", va="center", color="white")
            fig.colorbar(image, ax=ax, label="Unsafe rate")
            fig.tight_layout()
            fig.savefig(figures_dir / "baseline_lag_action_heatmap.png", dpi=180)
            plt.close(fig)


def plot_human_prediction_grid(derived_dir: Path, figures_dir: Path) -> None:
    plt = safe_import_matplotlib()
    if plt is None:
        return
    grid_path = derived_dir / "human_check_baseline_prediction_grid.csv"
    if not grid_path.exists():
        return
    grid = pd.read_csv(grid_path)
    if grid.empty:
        return
    risk_values = sorted(grid["max_private_risk"].dropna().unique())
    if not risk_values:
        return
    target_risk = min(risk_values, key=lambda value: abs(float(value) - 0.6))
    plot_data = grid[
        (grid["max_private_risk"] == target_risk)
        & (grid["first_round_unsafe"] == 0.0)
    ].copy()
    if plot_data.empty:
        return
    fig, ax = plt.subplots(figsize=(8.5, 5))
    labels = {
        "own0_opp0": "own SAFE, opponent SAFE",
        "own0_opp1": "own SAFE, opponent UNSAFE",
        "own1_opp0": "own UNSAFE, opponent SAFE",
        "own1_opp1": "own UNSAFE, opponent UNSAFE",
    }
    for profile, group in plot_data.groupby("lag_profile"):
        group = group.sort_values("progress_gap_before")
        ax.plot(
            group["progress_gap_before"],
            group["predicted_unsafe"],
            label=labels.get(profile, profile),
            linewidth=2,
        )
        ax.fill_between(
            group["progress_gap_before"],
            group["pred_ci95_low"],
            group["pred_ci95_high"],
            alpha=0.12,
        )
    ax.axvline(0, color="#555555", linewidth=1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Progress gap before decision")
    ax.set_ylabel("Predicted unsafe probability")
    ax.set_title("Baseline predicted unsafe by progress gap")
    ax.text(
        0.01,
        0.02,
        f"risk={float(target_risk):.1f}, first_round_unsafe=0",
        transform=ax.transAxes,
        fontsize=9,
        color="#444444",
    )
    ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(figures_dir / "baseline_predicted_unsafe_by_gap.png", dpi=180)
    plt.close(fig)


def plot_tree_importance(importances: pd.DataFrame, figures_dir: Path, scope_name: str) -> None:
    plt = safe_import_matplotlib()
    if plt is None or importances.empty:
        return
    top = importances.head(15).sort_values("importance")
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(top["feature"], top["importance"], color="#8a6f2a")
    ax.set_xlabel("Decision tree impurity importance")
    ax.set_title(f"Exploratory tree feature importance ({scope_name})")
    fig.tight_layout()
    filename = (
        "decision_tree_feature_importance.png"
        if scope_name == "all"
        else f"decision_tree_{scope_name}_feature_importance.png"
    )
    fig.savefig(figures_dir / filename, dpi=180)
    plt.close(fig)


def plot_decision_tree_visual(
    pipeline: Any,
    feature_names: np.ndarray,
    figures_dir: Path,
    scope_name: str,
    sk: dict[str, Any],
) -> None:
    plt = safe_import_matplotlib()
    if plt is None:
        return
    fig, ax = plt.subplots(figsize=(18, 9))
    sk["plot_tree"](
        pipeline.named_steps["tree"],
        feature_names=list(feature_names),
        class_names=["safe", "unsafe"],
        filled=True,
        impurity=False,
        proportion=True,
        rounded=True,
        fontsize=7,
        max_depth=3,
        ax=ax,
    )
    ax.set_title(f"Exploratory unsafe decision tree ({scope_name})")
    fig.tight_layout()
    filename = "decision_tree.png" if scope_name == "all" else f"decision_tree_{scope_name}.png"
    fig.savefig(figures_dir / filename, dpi=180)
    plt.close(fig)


def copy_if_exists(source: Path, target: Path) -> None:
    if source.exists():
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def write_requested_output_contract(output_dir: Path) -> None:
    """Write compact aliases matching the original requested artifact layout."""

    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    aliases = {
        "turns_canonical.csv": "turn_features.csv",
        "human_check_logit_coefficients.csv": "human_check_coefficients.csv",
        "decision_tree_root_stability.csv": "bootstrap_rule_stability.csv",
    }
    for source_name, target_name in aliases.items():
        copy_if_exists(derived_dir / source_name, derived_dir / target_name)

    metric_parts: list[pd.DataFrame] = []
    for path in sorted(derived_dir.glob("decision_tree_*_cv_metrics.csv")):
        if path.name == "decision_tree_cv_metrics.csv":
            continue
        metric_parts.append(pd.read_csv(path))
    if metric_parts:
        pd.concat(metric_parts, ignore_index=True).to_csv(
            derived_dir / "model_performance.csv", index=False
        )
    else:
        copy_if_exists(derived_dir / "decision_tree_cv_metrics.csv", derived_dir / "model_performance.csv")

    rule_rows = []
    for path in sorted(derived_dir.glob("decision_tree_*_rules.txt")):
        scope = path.name.removeprefix("decision_tree_").removesuffix("_rules.txt")
        rule_rows.append({"scope": scope, "rules": path.read_text(encoding="utf-8")})
    if rule_rows:
        pd.DataFrame(rule_rows).to_csv(derived_dir / "decision_tree_rules.csv", index=False)

    figure_aliases = {
        "baseline_unsafe_by_risk_model.png": "unsafe_by_risk_model.png",
        "baseline_unsafe_by_gap_bin.png": "unsafe_by_gap_bin.png",
        "baseline_lag_action_heatmap.png": "lag_action_heatmap.png",
        "baseline_predicted_unsafe_by_gap.png": "predicted_unsafe_by_gap.png",
        "decision_tree_feature_importance.png": "feature_importance.png",
    }
    for source_name, target_name in figure_aliases.items():
        copy_if_exists(figures_dir / source_name, figures_dir / target_name)
    if not (figures_dir / "decision_tree.png").exists():
        copy_if_exists(figures_dir / "decision_tree_all_completed.png", figures_dir / "decision_tree.png")

    report_path = output_dir / "report.md"
    if report_path.exists():
        copy_if_exists(report_path, reports_dir / "fh_analysis_report.md")


def write_report(
    output_dir: Path,
    coverage: pd.DataFrame,
    outputs: dict[str, pd.DataFrame],
    logit_coefficients: pd.DataFrame,
) -> None:
    output_dir = output_dir.resolve()
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"
    complete = coverage[coverage["manifest_status"] == "completed"]
    running = coverage[coverage["manifest_status"] != "completed"]
    lines = [
        "# FH Analytic Report",
        "",
        "## Stage 1: Coverage",
        "",
        f"- Raw runs discovered: {len(coverage)}",
        f"- Completed runs: {len(complete)}",
        f"- Incomplete/running runs: {len(running)}",
        f"- Race rows: {int(coverage['race_rows'].sum())}",
        f"- Player rows: {int(coverage['player_rows'].sum())}",
        f"- Turn rows: {int(coverage['turn_rows'].sum())}",
        "",
        "Coverage table: `derived/coverage_audit.csv`.",
        "Data-quality findings: `derived/data_quality_findings.csv`; retry rates by model: "
        "`derived/data_quality_retry_by_model.csv`.",
        "Canonical tables retain duplicate-grain rows for audit, but descriptive/model stages "
        "exclude rows flagged with `duplicate_grain_key == True`.",
        "",
    ]
    dq_path = derived_dir / "data_quality_findings.csv"
    if dq_path.exists():
        dq = pd.read_csv(dq_path)
        visible_dq = dq[
            ["check", "severity", "status", "rows_affected", "affected_rate", "finding"]
        ].copy()
        lines.extend(["Quality gate summary:", "", markdown_table(visible_dq), ""])
    lines.extend(
        [
            "## Stage 2: Descriptive Visuals",
            "",
            "- Baseline unsafe by risk/model: `figures/baseline_unsafe_by_risk_model.png`",
            "- Baseline unsafe by progress gap: `figures/baseline_unsafe_by_gap_bin.png`",
            "- Baseline gap-threshold scan: `figures/baseline_gap_threshold_scan.png`",
            "- Baseline lag-action heatmap: `figures/baseline_lag_action_heatmap.png`",
            "",
            "Core descriptive tables are in `derived/unsafe_by_*`. Use these before fitting models.",
            "",
            "## Stage 3: Human-Reference Checks",
            "",
            "Human-check logit outputs: `derived/human_check_logit_coefficients.csv` and "
            "`derived/human_reference_ledger.csv`.",
            "Baseline logit bootstrap CI: `derived/human_check_baseline_bootstrap_summary.csv`.",
            "Baseline predicted gap curves: `figures/baseline_predicted_unsafe_by_gap.png`.",
            "Segment stability outputs: `derived/human_check_segment_coefficients.csv` and "
            "`derived/human_check_sign_stability.csv`.",
            "Segment models can show separation in small/saturated slices; use sign stability as a "
            "screening diagnostic and rely on pooled baseline coefficients for headline estimates.",
        ]
    )
    if not logit_coefficients.empty:
        core = logit_coefficients[
            (logit_coefficients["scope"] == "baseline_completed")
            & (logit_coefficients["spec"] == "M3_interactions")
            & (logit_coefficients["term"].isin(HUMAN_TERMS.values()))
        ][["term", "coef", "odds_ratio", "ci95_low", "ci95_high", "p_value"]]
        if not core.empty:
            lines.extend(["", "Baseline M3 core terms:", ""])
            lines.append(markdown_table(core))
    bootstrap_summary_path = derived_dir / "human_check_baseline_bootstrap_summary.csv"
    if bootstrap_summary_path.exists():
        bootstrap_summary = pd.read_csv(bootstrap_summary_path)
        visible_bootstrap = bootstrap_summary[
            [
                "term",
                "bootstrap_successes",
                "coef_median",
                "coef_ci95_low",
                "coef_ci95_high",
                "odds_ratio_median",
                "sign_match_share",
            ]
        ].copy()
        lines.extend(["", "Baseline cluster bootstrap summary:", ""])
        lines.append(markdown_table(visible_bootstrap))
    lines.extend(
        [
            "",
            "## Stage 4: Exploratory Fit",
            "",
            "Tree outputs are split by scope so the analysis stays sequential:",
            "",
            "- Baseline tree CV metrics: `derived/decision_tree_baseline_completed_cv_metrics.csv`",
            "- Baseline tree rules: `derived/decision_tree_baseline_completed_rules.txt`",
            "- Baseline tree leaf support/confidence: `derived/decision_tree_baseline_completed_leaf_summary.csv`",
            "- Full completed tree CV metrics: `derived/decision_tree_all_completed_cv_metrics.csv`",
            "- Full completed tree rules: `derived/decision_tree_all_completed_rules.txt`",
            "- Full completed root stability: `derived/decision_tree_all_completed_root_stability.csv`",
            "- Scope summary: `derived/decision_tree_scope_summary.csv`",
            "",
            "Interpret tree/rule outputs as exploratory compression of behaviour, not causal effects.",
        ]
    )
    scope_summary_path = derived_dir / "decision_tree_scope_summary.csv"
    if scope_summary_path.exists():
        scope_summary = pd.read_csv(scope_summary_path)
        visible = scope_summary[
            [
                "scope",
                "decisions",
                "clusters",
                "unsafe_rate",
                "cv_balanced_accuracy_mean",
                "cv_roc_auc_mean",
                "top_root_feature",
                "top_root_share",
            ]
        ].copy()
        lines.extend(["", "Tree scope summary:", "", markdown_table(visible)])
    lines.extend(
        [
            "",
            "## Suggested Next Refinements",
            "",
            "- Decide whether incomplete Gemini cells are excluded or shown with an incomplete flag.",
            "- Add bootstrap CIs for logit coefficients if cluster-robust CIs are not enough.",
            "- Add shallow trees per provider/model to see whether the same rule appears across families.",
        ]
    )
    (output_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    # Keep paths visible for automation that wants a manifest.
    manifest = {
        "output_dir": str(output_dir.relative_to(REPO_ROOT)),
        "derived_dir": str(derived_dir.relative_to(REPO_ROOT)),
        "figures_dir": str(figures_dir.relative_to(REPO_ROOT)),
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    write_requested_output_contract(output_dir)


def main() -> None:
    args = parse_args()
    input_root = args.input.resolve()
    output_dir = args.output.resolve()
    derived_dir = output_dir / "derived"
    figures_dir = output_dir / "figures"
    derived_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    runs = discover_runs(input_root)
    races, players, turns, coverage = load_tables(runs)
    coverage.to_csv(derived_dir / "coverage_audit.csv", index=False)
    if races.empty or players.empty or turns.empty:
        raise SystemExit("No complete race/player/turn tables were found under the input root.")

    races = add_race_features(races, coverage)
    players = add_player_features(players, coverage)
    turns = add_turn_features(turns, coverage)
    races.to_csv(derived_dir / "races_canonical.csv", index=False)
    players.to_csv(derived_dir / "players_canonical.csv", index=False)
    turns.to_csv(derived_dir / "turns_canonical.csv", index=False)
    build_data_quality_findings(races, players, turns, coverage, derived_dir)

    outputs = build_descriptive_outputs(turns, players, derived_dir)
    plot_outputs(outputs, figures_dir)

    logit_coefficients, _ = fit_human_check(
        turns,
        derived_dir,
        bootstrap=args.logit_bootstrap,
        seed=args.random_seed,
    )
    plot_human_prediction_grid(derived_dir, figures_dir)
    fit_decision_tree(
        turns,
        derived_dir,
        figures_dir,
        bootstrap=args.bootstrap,
        seed=args.random_seed,
    )
    write_report(output_dir, coverage, outputs, logit_coefficients)
    print(f"Wrote FH analysis outputs to {output_dir}")


if __name__ == "__main__":
    main()
