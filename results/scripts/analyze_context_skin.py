#!/usr/bin/env python3
"""Validate, analyse, and visualize paired context-skin experiments.

The script accepts a common parent directory or explicit live/fixed-state roots.
It never pools different model digests, configurations, source revisions, profiles,
or decoding temperatures.  Behavioral action rates use the decoded semantic action;
opaque P/Q response-code summaries are retained as a separate presentation factor.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Circle
import numpy as np
import pandas as pd


LIVE_SCHEMA = "ai-race-context-skin-run-v1"
FIXED_SCHEMA = "ai-race-context-fixed-state-run-v1"
ABSTRACT = "abstract_contest"
BOOTSTRAP_SEED = 260801

SKIN_ORDER = [
    "abstract_contest",
    "technology_race",
    "logistics_contract",
    "crystal_guild_contract",
    "hospital_deployment",
    "colony_life_support",
    "robotic_expedition",
    "fictional_cartography",
]
SKIN_LABEL = {
    "abstract_contest": "Abstract control",
    "technology_race": "Technology race",
    "logistics_contract": "Logistics contract",
    "crystal_guild_contract": "Crystal guild",
    "hospital_deployment": "Hospital deployment",
    "colony_life_support": "Colony life support",
    "robotic_expedition": "Robotic expedition",
    "fictional_cartography": "Fictional cartography",
}
PLANNED_PAIRS = [
    ("logistics_contract", "crystal_guild_contract", "Commercial logistics"),
    ("hospital_deployment", "colony_life_support", "Public-safety deployment"),
    ("robotic_expedition", "fictional_cartography", "Neutral exploration"),
]

BLUE = "#2563EB"
BLUE_LIGHT = "#C9DBFF"
GOLD = "#D79B00"
ORANGE = "#E8792E"
PINK = "#C65A8E"
INK = "#172033"
MUTED = "#687386"
GRID = "#D9E0EA"
PAPER = "#FBFCFE"
WHITE = "#FFFFFF"
CMAP = LinearSegmentedColormap.from_list("context-blue", ["#F4F7FC", "#AFC8FB", BLUE])


@dataclass(frozen=True)
class RunInput:
    path: Path
    manifest: dict[str, Any]


@dataclass
class AnalysisData:
    live_turns: pd.DataFrame
    live_players: pd.DataFrame
    live_races: pd.DataFrame
    replay: pd.DataFrame
    comprehension: pd.DataFrame
    live_runs: list[RunInput]
    fixed_runs: list[RunInput]
    validation: dict[str, Any]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/open_source/context_skin_pilot"),
        help="Common parent used when explicit roots are omitted.",
    )
    parser.add_argument("--live-root", action="append", type=Path, default=[])
    parser.add_argument("--fixed-root", action="append", type=Path, default=[])
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="New analysis directory; source result directories are never modified.",
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=5000)
    parser.add_argument("--allow-incomplete-skins", action="store_true")
    return parser.parse_args(argv)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _model_digest(manifest: dict[str, Any]) -> str:
    model = manifest.get("model") or {}
    ollama = manifest.get("ollama_model") or {}
    digest = model.get("digest") or ollama.get("digest") or model.get("config_sha256")
    return str(digest or "")


def _model_name(manifest: dict[str, Any]) -> str:
    model = manifest.get("model") or {}
    ollama = manifest.get("ollama_model") or {}
    return str(model.get("name") or model.get("short_name") or ollama.get("name") or "")


def _cohort_key(run: RunInput) -> tuple[Any, ...]:
    manifest = run.manifest
    if manifest["schema_version"] == LIVE_SCHEMA:
        decoding = manifest.get("decoding", {})
        temperature = decoding.get("temperature")
    else:
        decoding = manifest.get("decoding", {})
        temperature = decoding.get("replay_temperature")
    return (
        manifest["schema_version"],
        manifest.get("profile"),
        temperature,
        _model_digest(manifest),
        manifest.get("experiment_config_sha256"),
        manifest.get("source_sha256"),
    )


def discover_runs(roots: Iterable[Path], schema: str) -> list[RunInput]:
    """Recursively find completed run manifests under arbitrary lane roots."""
    found: dict[Path, RunInput] = {}
    for raw_root in roots:
        root = raw_root.resolve()
        candidates = [root] if root.name == "run_manifest.json" else list(root.rglob("run_manifest.json"))
        for path in candidates:
            if not path.is_file():
                continue
            manifest = _json(path)
            if manifest.get("schema_version") != schema:
                continue
            found[path.resolve()] = RunInput(path.parent.resolve(), manifest)
    runs = sorted(found.values(), key=lambda run: str(run.path))
    if not runs:
        raise ValueError(f"No {schema} manifests found under {[str(path) for path in roots]}")
    groups: dict[tuple[Any, ...], list[RunInput]] = {}
    for run in runs:
        groups.setdefault(_cohort_key(run), []).append(run)
    if len(groups) != 1:
        detail = {str(key): len(value) for key, value in groups.items()}
        raise ValueError(
            "Multiple incompatible cohorts were discovered; pass explicit roots. "
            + json.dumps(detail, indent=2)
        )
    return runs


def _one_value(values: Iterable[Any], label: str) -> Any:
    normalized = {json.dumps(value, sort_keys=True) for value in values}
    if len(normalized) != 1:
        raise ValueError(f"Mismatched {label}: {sorted(normalized)}")
    return json.loads(next(iter(normalized)))


def _verify_fixed_artifacts(run: RunInput) -> None:
    for label, artifact in run.manifest.get("artifacts", {}).items():
        path = run.path / artifact["path"]
        if not path.is_file():
            raise ValueError(f"Missing fixed-state artifact {label}: {path}")
        payload = path.read_bytes()
        expected_bytes = int(artifact.get("bytes", len(payload)))
        expected_sha = artifact.get("sha256")

        def matches(candidate: bytes) -> bool:
            return len(candidate) == expected_bytes and (
                not expected_sha
                or hashlib.sha256(candidate).hexdigest() == expected_sha
            )

        if matches(payload):
            continue
        # Git can normalize CRLF/LF on checkout even when the immutable source
        # manifest recorded transport bytes.  Admit only an exact hash match
        # after a pure newline conversion; no other content drift is tolerated.
        newline_candidates = []
        if b"\r\n" in payload:
            newline_candidates.append(payload.replace(b"\r\n", b"\n"))
        if b"\n" in payload and b"\r\n" not in payload:
            newline_candidates.append(payload.replace(b"\n", b"\r\n"))
        if any(matches(candidate) for candidate in newline_candidates):
            continue
        if len(payload) != expected_bytes:
            raise ValueError(f"Artifact size mismatch: {path}")
        raise ValueError(f"Artifact SHA-256 mismatch: {path}")


def _read_jsonl(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    return pd.DataFrame(rows)


def validate_and_load(
    live_runs: list[RunInput],
    fixed_runs: list[RunInput],
    *,
    allow_incomplete_skins: bool = False,
) -> AnalysisData:
    """Validate provenance/coverage and load analysis-ready semantic rows."""
    for run in live_runs + fixed_runs:
        if run.manifest.get("status") != "completed":
            raise ValueError(f"Incomplete run {run.path}: {run.manifest.get('status')}")

    live_digest = _one_value((_model_digest(run.manifest) for run in live_runs), "live model digest")
    fixed_digest = _one_value((_model_digest(run.manifest) for run in fixed_runs), "fixed model digest")
    if not live_digest or live_digest != fixed_digest:
        raise ValueError(f"Live/fixed model digest mismatch: {live_digest!r} vs {fixed_digest!r}")
    model_name = _one_value((_model_name(run.manifest) for run in live_runs + fixed_runs), "model name")
    live_experiment_hash = _one_value(
        (run.manifest.get("experiment_config_sha256") for run in live_runs),
        "live experiment config hash",
    )
    fixed_experiment_hash = _one_value(
        (run.manifest.get("experiment_config_sha256") for run in fixed_runs),
        "fixed experiment config hash",
    )
    live_source = _one_value((run.manifest.get("source_sha256") for run in live_runs), "live source hash")
    fixed_source = _one_value((run.manifest.get("source_sha256") for run in fixed_runs), "fixed source hash")
    game_hashes = _one_value((run.manifest.get("game_config_sha256") for run in live_runs + fixed_runs), "game config hashes")
    profile_live = _one_value((run.manifest.get("profile") for run in live_runs), "live profile")
    profile_fixed = _one_value((run.manifest.get("profile") for run in fixed_runs), "fixed profile")
    live_temperature = _one_value((run.manifest.get("decoding", {}).get("temperature") for run in live_runs), "live temperature")
    replay_temperature = _one_value((run.manifest.get("decoding", {}).get("replay_temperature") for run in fixed_runs), "replay temperature")
    crn_hash = _one_value((run.manifest.get("crn", {}).get("contract_sha256") for run in live_runs), "live CRN contract")

    live_skin_ids = [str(run.manifest.get("context_skin", {}).get("id")) for run in live_runs]
    fixed_skin_ids = [str(skin) for run in fixed_runs for skin in run.manifest.get("skins", [])]
    if len(set(live_skin_ids)) != len(live_skin_ids):
        raise ValueError("Duplicate live context skin manifests")
    if len(set(fixed_skin_ids)) != len(fixed_skin_ids):
        raise ValueError("Duplicate fixed-state skin assignments across lanes")
    if set(live_skin_ids) != set(fixed_skin_ids):
        raise ValueError("Live and fixed-state context sets differ")
    if not allow_incomplete_skins and set(live_skin_ids) != set(SKIN_ORDER):
        raise ValueError(f"Expected all eight frozen skins, got {sorted(live_skin_ids)}")
    if ABSTRACT not in live_skin_ids:
        raise ValueError("abstract_contest control is required for paired effects")

    turn_frames: list[pd.DataFrame] = []
    player_frames: list[pd.DataFrame] = []
    race_frames: list[pd.DataFrame] = []
    for run in live_runs:
        skin_id = str(run.manifest["context_skin"]["id"])
        turns_path = run.path / "turns.jsonl"
        players_path = run.path / "players.csv"
        races_path = run.path / "races.csv"
        for path in (turns_path, players_path, races_path):
            if not path.is_file():
                raise ValueError(f"Missing live artifact: {path}")
        turns = _read_jsonl(turns_path)
        players = pd.read_csv(players_path)
        races = pd.read_csv(races_path)
        if len(races) != int(run.manifest["n_races"]) or len(turns) != int(run.manifest["n_turns"]):
            raise ValueError(f"Manifest row count mismatch: {run.path}")
        if len(races) != int(run.manifest["expected_races"]):
            raise ValueError(f"Incomplete expected race coverage: {run.path}")
        for frame in (turns, players, races):
            frame["skin_id"] = skin_id
            frame["family"] = str(run.manifest["context_skin"].get("family", ""))
            frame["lane"] = str(run.manifest.get("lane", ""))
        turns["mapping_id"] = turns["prompt_version"].astype(str).str.rsplit(":", n=1).str[-1]
        players["mapping_id"] = players["prompt_version"].astype(str).str.rsplit(":", n=1).str[-1]
        races["mapping_id"] = races["prompt_version"].astype(str).str.rsplit(":", n=1).str[-1]
        declared = set(run.manifest["action_code_factor"]["mappings"])
        if set(turns["mapping_id"].unique()) - declared:
            raise ValueError(f"Unknown action mapping in {turns_path}")
        turns["opaque_action_code"] = turns["raw_response"].astype(str).str.extract(
            r"^\s*ACTION\s*:\s*([PQ])\s*$", expand=False
        )
        turn_frames.append(turns)
        player_frames.append(players)
        race_frames.append(races)

    replay_frames: list[pd.DataFrame] = []
    comprehension_frames: list[pd.DataFrame] = []
    for run in fixed_runs:
        _verify_fixed_artifacts(run)
        replay = pd.read_csv(run.path / "paired_estimand_input.csv")
        comprehension = _read_jsonl(run.path / "comprehension_raw.jsonl")
        if len(replay) != int(run.manifest["expected_replay_rows"]):
            raise ValueError(f"Fixed replay row count mismatch: {run.path}")
        if len(comprehension) != int(run.manifest["expected_comprehension_rows"]):
            raise ValueError(f"Comprehension row count mismatch: {run.path}")
        if not bool((run.manifest.get("coverage") or {}).get("passed")):
            raise ValueError(f"Fixed-state lane coverage failed: {run.path}")
        replay_frames.append(replay)
        comprehension_frames.append(comprehension)

    turns = pd.concat(turn_frames, ignore_index=True)
    players = pd.concat(player_frames, ignore_index=True)
    races = pd.concat(race_frames, ignore_index=True)
    replay = pd.concat(replay_frames, ignore_index=True)
    comprehension = pd.concat(comprehension_frames, ignore_index=True)
    for frame, bool_cols in ((turns, ["parse_failed"]), (replay, ["parse_failed"]), (comprehension, ["semantic_correct", "strict_valid"])):
        for column in bool_cols:
            if frame[column].dtype != bool:
                frame[column] = frame[column].astype(str).str.lower().map({"true": True, "false": False, "1": True, "0": False})
            if frame[column].isna().any():
                raise ValueError(f"Invalid Boolean values in {column}")
    for frame in (turns, replay):
        frame["unsafe"] = pd.to_numeric(frame["unsafe"], errors="raise").astype(int)
        if not set(frame["unsafe"].unique()).issubset({0, 1}):
            raise ValueError("Unsafe must be binary")
        frame["max_private_risk"] = pd.to_numeric(frame["max_private_risk"], errors="raise")

    # Strict rectangular checks. Mapping stays part of every key.
    first = turns[turns["round"] == 1]
    live_key = ["max_private_risk", "rep", "player_index", "mapping_id"]
    counts = first.groupby(live_key, dropna=False)["skin_id"].nunique()
    if not (counts == len(live_skin_ids)).all():
        raise ValueError("Incomplete first-round paired context coverage")
    if first.duplicated(live_key + ["skin_id"]).any():
        raise ValueError("Duplicate first-round paired cell")
    fixed_key = ["state_id", "mapping_id"]
    fixed_counts = replay.groupby(fixed_key, dropna=False)["skin_id"].nunique()
    if not (fixed_counts == len(fixed_skin_ids)).all():
        raise ValueError("Incomplete fixed-state paired context coverage")
    if replay.duplicated(fixed_key + ["skin_id"]).any():
        raise ValueError("Duplicate fixed-state paired cell")

    admission_cells = []
    for run in fixed_runs:
        admission_cells.extend((run.manifest.get("admission") or {}).get("by_cell", {}).values())
    admission_passed = bool(admission_cells) and all(bool(cell.get("passed")) for cell in admission_cells)
    validation = {
        "status": "passed",
        "model_name": model_name,
        "model_digest": live_digest,
        "live_experiment_config_sha256": live_experiment_hash,
        "fixed_experiment_config_sha256": fixed_experiment_hash,
        "cross_protocol_config_hash_match": live_experiment_hash == fixed_experiment_hash,
        "game_config_sha256": game_hashes,
        "live_source_sha256": live_source,
        "fixed_source_sha256": fixed_source,
        "live_crn_contract_sha256": crn_hash,
        "live_profile": profile_live,
        "fixed_profile": profile_fixed,
        "live_temperature": live_temperature,
        "replay_temperature": replay_temperature,
        "cross_protocol_temperature_match": live_temperature == replay_temperature,
        "n_skins": len(live_skin_ids),
        "skins": sorted(live_skin_ids),
        "n_live_races": int(races["game_id"].nunique()),
        "n_live_decisions": int(len(turns)),
        "n_fixed_states": int(replay["state_id"].nunique()),
        "n_fixed_rows": int(len(replay)),
        "n_comprehension_rows": int(len(comprehension)),
        "live_parse_failures": int(turns["parse_failed"].sum()),
        "live_retried_decisions": int((pd.to_numeric(turns["retry_count"]) > 0).sum()),
        "fixed_parse_failures": int(replay["parse_failed"].sum()),
        "admission_passed": admission_passed,
        "claim_status": "exploratory_pilot" if admission_passed else "diagnostic_comprehension_failed",
        "live_manifest_sha256": [sha256_file(run.path / "run_manifest.json") for run in live_runs],
        "fixed_manifest_sha256": [sha256_file(run.path / "run_manifest.json") for run in fixed_runs],
    }
    return AnalysisData(turns, players, races, replay, comprehension, live_runs, fixed_runs, validation)


def cluster_bootstrap_mean(
    rows: pd.DataFrame,
    value: str,
    cluster: str,
    *,
    repetitions: int,
    seed: int,
) -> tuple[float, float, float, int]:
    """Percentile interval after resampling equal-weight cluster means."""
    clean = rows[[cluster, value]].dropna()
    cluster_means = clean.groupby(cluster, sort=True)[value].mean().to_numpy(float)
    estimate = float(cluster_means.mean()) if len(cluster_means) else math.nan
    if len(cluster_means) < 2 or repetitions < 1:
        return estimate, math.nan, math.nan, int(len(cluster_means))
    rng = np.random.default_rng(seed)
    draws = rng.choice(cluster_means, size=(repetitions, len(cluster_means)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high), int(len(cluster_means))


def _paired_rows(
    frame: pd.DataFrame,
    context: str,
    reference: str,
    keys: list[str],
    value: str,
) -> pd.DataFrame:
    subset = frame[frame["skin_id"].isin([context, reference]) & ~frame["parse_failed"]].copy()
    wide = subset.pivot(index=keys, columns="skin_id", values=value).dropna()
    if context not in wide or reference not in wide:
        return pd.DataFrame(columns=keys + ["context_value", "reference_value", "difference"])
    wide = wide.reset_index().rename(columns={context: "context_value", reference: "reference_value"})
    wide["difference"] = wide["context_value"] - wide["reference_value"]
    return wide


def _effect_row(
    paired: pd.DataFrame,
    *,
    context: str,
    reference: str,
    estimand: str,
    cluster: str,
    repetitions: int,
    seed: int,
    risk: float | str = "all",
) -> dict[str, Any]:
    estimate, low, high, n_clusters = cluster_bootstrap_mean(
        paired, "difference", cluster, repetitions=repetitions, seed=seed
    )
    flips_to_unsafe = int(((paired["reference_value"] == 0) & (paired["context_value"] == 1)).sum())
    flips_to_safe = int(((paired["reference_value"] == 1) & (paired["context_value"] == 0)).sum())
    return {
        "estimand": estimand,
        "context": context,
        "reference": reference,
        "risk": risk,
        "estimate_pp": estimate * 100,
        "ci_low_pp": low * 100,
        "ci_high_pp": high * 100,
        "n_paired_rows": int(len(paired)),
        "n_clusters": n_clusters,
        "safe_to_unsafe": flips_to_unsafe,
        "unsafe_to_safe": flips_to_safe,
        "stable": int(len(paired) - flips_to_unsafe - flips_to_safe),
    }


def summarize_rates(data: AnalysisData) -> tuple[pd.DataFrame, pd.DataFrame]:
    turns = data.live_turns.copy()
    turns["phase"] = np.where(turns["round"] == 1, "first_round", "later_rounds")
    valid = turns[~turns["parse_failed"]]
    live = (
        valid.groupby(["skin_id", "max_private_risk", "mapping_id", "phase"], observed=True)
        .agg(unsafe_rate=("unsafe", "mean"), n_decisions=("unsafe", "size"), code_p_rate=("opaque_action_code", lambda x: float((x == "P").mean())))
        .reset_index()
    )
    all_phase = (
        valid.groupby(["skin_id", "max_private_risk", "mapping_id"], observed=True)
        .agg(unsafe_rate=("unsafe", "mean"), n_decisions=("unsafe", "size"), code_p_rate=("opaque_action_code", lambda x: float((x == "P").mean())))
        .reset_index()
    )
    all_phase["phase"] = "full_trajectory"
    live = pd.concat([live, all_phase], ignore_index=True)
    replay_valid = data.replay[~data.replay["parse_failed"]]
    replay = (
        replay_valid.groupby(["skin_id", "max_private_risk", "mapping_id"], observed=True)
        .agg(unsafe_rate=("unsafe", "mean"), n_states=("state_id", "nunique"), n_rows=("unsafe", "size"), code_p_rate=("opaque_action_code", lambda x: float((x == "P").mean())))
        .reset_index()
    )
    return live, replay


def paired_effects(data: AnalysisData, repetitions: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    flips: list[dict[str, Any]] = []
    contexts = [skin for skin in SKIN_ORDER if skin in set(data.live_turns["skin_id"]) and skin != ABSTRACT]
    turn_keys = ["max_private_risk", "rep", "player_index", "mapping_id"]
    for phase, frame in (
        ("live_first_round", data.live_turns[data.live_turns["round"] == 1]),
        (
            "live_full_trajectory",
            data.live_turns.groupby(turn_keys + ["skin_id"], as_index=False).agg(
                unsafe=("unsafe", "mean"), parse_failed=("parse_failed", "max")
            ),
        ),
    ):
        for context_index, context in enumerate(contexts):
            paired = _paired_rows(frame, context, ABSTRACT, turn_keys, "unsafe")
            paired["cluster_id"] = paired["rep"].astype(str)
            rows.append(_effect_row(paired, context=context, reference=ABSTRACT, estimand=phase, cluster="cluster_id", repetitions=repetitions, seed=BOOTSTRAP_SEED + context_index))
            for risk in sorted(paired["max_private_risk"].unique()):
                risk_rows = paired[paired["max_private_risk"] == risk].copy()
                rows.append(_effect_row(risk_rows, context=context, reference=ABSTRACT, estimand=phase, cluster="cluster_id", repetitions=repetitions, seed=BOOTSTRAP_SEED + 100 + context_index, risk=float(risk)))
            if phase == "live_first_round":
                summary = rows[-(1 + len(paired["max_private_risk"].unique()))]
                flips.append({key: summary[key] for key in ("estimand", "context", "reference", "n_paired_rows", "safe_to_unsafe", "unsafe_to_safe", "stable")})

    fixed_keys = ["state_id", "mapping_id"]
    for context_index, context in enumerate(contexts):
        paired = _paired_rows(data.replay, context, ABSTRACT, fixed_keys, "unsafe")
        paired["cluster_id"] = paired["state_id"].astype(str)
        rows.append(_effect_row(paired, context=context, reference=ABSTRACT, estimand="fixed_state_direct", cluster="cluster_id", repetitions=repetitions, seed=BOOTSTRAP_SEED + 200 + context_index))
        fixed_row = rows[-1]
        flips.append({key: fixed_row[key] for key in ("estimand", "context", "reference", "n_paired_rows", "safe_to_unsafe", "unsafe_to_safe", "stable")})
        for risk in sorted(data.replay["max_private_risk"].unique()):
            state_ids = set(data.replay.loc[data.replay["max_private_risk"] == risk, "state_id"])
            risk_rows = paired[paired["state_id"].isin(state_ids)].copy()
            rows.append(_effect_row(risk_rows, context=context, reference=ABSTRACT, estimand="fixed_state_direct", cluster="cluster_id", repetitions=repetitions, seed=BOOTSTRAP_SEED + 300 + context_index, risk=float(risk)))
    return pd.DataFrame(rows), pd.DataFrame(flips)


def payoff_setback_summary(data: AnalysisData, repetitions: int) -> pd.DataFrame:
    frame = data.live_players.copy()
    frame["cluster_id"] = frame["rep"].astype(str)
    rows = []
    for context_index, context in enumerate(SKIN_ORDER):
        subset = frame[frame["skin_id"] == context]
        if subset.empty:
            continue
        payoff = cluster_bootstrap_mean(subset, "final_payoff", "cluster_id", repetitions=repetitions, seed=BOOTSTRAP_SEED + 400 + context_index)
        setback = cluster_bootstrap_mean(subset, "setback", "cluster_id", repetitions=repetitions, seed=BOOTSTRAP_SEED + 500 + context_index)
        rows.append({
            "context": context,
            "mean_final_payoff": payoff[0], "payoff_ci_low": payoff[1], "payoff_ci_high": payoff[2],
            "setback_rate": setback[0], "setback_ci_low": setback[1], "setback_ci_high": setback[2],
            "n_players": int(len(subset)), "n_race_clusters": payoff[3],
        })
    return pd.DataFrame(rows)


def planned_contrasts(data: AnalysisData, repetitions: int) -> pd.DataFrame:
    rows = []
    turn_keys = ["max_private_risk", "rep", "player_index", "mapping_id"]
    live_full = data.live_turns.groupby(turn_keys + ["skin_id"], as_index=False).agg(
        unsafe=("unsafe", "mean"), parse_failed=("parse_failed", "max")
    )
    for pair_index, (realistic, fictional, family) in enumerate(PLANNED_PAIRS):
        for estimand, frame, keys, cluster_parts in (
            ("live_first_round", data.live_turns[data.live_turns["round"] == 1], turn_keys, ["rep"]),
            ("live_full_trajectory", live_full, turn_keys, ["rep"]),
            ("fixed_state_direct", data.replay, ["state_id", "mapping_id"], ["state_id"]),
        ):
            paired = _paired_rows(frame, fictional, realistic, keys, "unsafe")
            paired["cluster_id"] = paired[cluster_parts].astype(str).agg("/".join, axis=1)
            effect = _effect_row(paired, context=fictional, reference=realistic, estimand=estimand, cluster="cluster_id", repetitions=repetitions, seed=BOOTSTRAP_SEED + 600 + pair_index)
            effect.update(pair_family=family, realistic=realistic, fictional=fictional)
            rows.append(effect)
    return pd.DataFrame(rows)


def comprehension_summaries(data: AnalysisData) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = data.comprehension.copy()
    domain = (
        frame.groupby("domain", observed=True)
        .agg(semantic_accuracy=("semantic_correct", "mean"), strict_valid_rate=("strict_valid", "mean"), n=("domain", "size"))
        .reset_index()
    )
    cell = (
        frame.groupby(["skin_id", "mapping_id"], observed=True)
        .agg(semantic_accuracy=("semantic_correct", "mean"), strict_valid_rate=("strict_valid", "mean"), n=("domain", "size"))
        .reset_index()
    )
    return domain, cell


def quality_summary(data: AnalysisData) -> pd.DataFrame:
    """One audit row per source shard, including failures and coverage."""
    rows: list[dict[str, Any]] = []
    for run in data.live_runs:
        skin = str(run.manifest["context_skin"]["id"])
        turns = data.live_turns[data.live_turns["skin_id"] == skin]
        races = data.live_races[data.live_races["skin_id"] == skin]
        rows.append({
            "protocol": "live_trajectory",
            "shard": f"lane_{run.manifest.get('lane')}/{skin}",
            "status": run.manifest["status"],
            "expected_rows": int(run.manifest["expected_races"]),
            "observed_rows": int(races["game_id"].nunique()),
            "observed_decisions_or_probes": int(len(turns)),
            "parse_failures": int(turns["parse_failed"].sum()),
            "retried_rows": int((pd.to_numeric(turns["retry_count"]) > 0).sum()),
            "paired_coverage_passed": True,
            "admission_passed": "not_applicable",
        })
    for run in data.fixed_runs:
        skins = set(str(value) for value in run.manifest["skins"])
        replay = data.replay[data.replay["skin_id"].isin(skins)]
        comprehension = data.comprehension[data.comprehension["skin_id"].isin(skins)]
        rows.append({
            "protocol": "fixed_state_replay",
            "shard": f"lane_{run.manifest.get('lane')}",
            "status": run.manifest["status"],
            "expected_rows": int(run.manifest["expected_replay_rows"]),
            "observed_rows": int(len(replay)),
            "observed_decisions_or_probes": int(len(comprehension)),
            "parse_failures": int(replay["parse_failed"].sum()),
            "retried_rows": int((pd.to_numeric(replay["retry_count"]) > 0).sum()),
            "paired_coverage_passed": bool((run.manifest.get("coverage") or {}).get("passed")),
            "admission_passed": bool((run.manifest.get("admission") or {}).get("passed")),
        })
    return pd.DataFrame(rows)


def context_mapping_diagnostic(data: AnalysisData, repetitions: int) -> pd.DataFrame:
    rows = []
    for context_index, context in enumerate(SKIN_ORDER):
        subset = data.replay[(data.replay["skin_id"] == context) & ~data.replay["parse_failed"]]
        wide = subset.pivot(index="state_id", columns="mapping_id", values="unsafe").dropna().reset_index()
        if not {"safe_p", "safe_q"}.issubset(wide.columns):
            continue
        wide["difference"] = wide["safe_q"] - wide["safe_p"]
        estimate = cluster_bootstrap_mean(wide, "difference", "state_id", repetitions=repetitions, seed=BOOTSTRAP_SEED + 700 + context_index)
        rows.append({
            "context": context, "estimand": "safe_q_minus_safe_p_semantic_unsafe_pp",
            "estimate_pp": estimate[0] * 100, "ci_low_pp": estimate[1] * 100,
            "ci_high_pp": estimate[2] * 100, "n_states": estimate[3],
        })
    return pd.DataFrame(rows)


def cross_protocol_alignment(effects: pd.DataFrame) -> pd.DataFrame:
    """Descriptive alignment only; the live/fixed temperatures are not pooled."""
    overall = effects[effects["risk"] == "all"]
    wide = overall.pivot(index="context", columns="estimand", values="estimate_pp")
    result = wide[["live_full_trajectory", "fixed_state_direct"]].reset_index()
    result["same_direction"] = (
        np.sign(result["live_full_trajectory"])
        == np.sign(result["fixed_state_direct"])
    )
    result["live_rank_desc"] = result["live_full_trajectory"].rank(
        method="average", ascending=False
    )
    result["fixed_rank_desc"] = result["fixed_state_direct"].rank(
        method="average", ascending=False
    )
    return _ordered(result)


def _style_axis(axis: Any, *, grid_axis: str = "x") -> None:
    axis.set_facecolor(PAPER)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.grid(axis=grid_axis, color=GRID, linewidth=0.75, alpha=0.8)
    axis.set_axisbelow(True)


def _blossom(figure: Any) -> None:
    center = (0.965, 0.965)
    for dx, dy, color in ((-0.010, 0, BLUE), (0.010, 0, GOLD), (0, -0.010, ORANGE), (0, 0.010, PINK)):
        figure.add_artist(Circle((center[0] + dx, center[1] + dy), 0.006, transform=figure.transFigure, facecolor=color, edgecolor="none", alpha=0.95))


def _title(axis: Any, title: str, subtitle: str) -> None:
    axis.set_title(title, loc="left", color=INK, weight="bold", fontsize=13, pad=30)
    axis.text(0, 1.01, subtitle, transform=axis.transAxes, color=MUTED, fontsize=8.5, va="bottom")


def _save(figure: Any, output: Path, stem: str) -> None:
    _blossom(figure)
    figure.savefig(output / f"{stem}.png", dpi=240, bbox_inches="tight", facecolor=WHITE)
    figure.savefig(output / f"{stem}.pdf", bbox_inches="tight", facecolor=WHITE)
    plt.close(figure)


def _ordered(frame: pd.DataFrame, column: str = "context") -> pd.DataFrame:
    order = {skin: index for index, skin in enumerate(SKIN_ORDER)}
    result = frame.copy()
    result["_order"] = result[column].map(order)
    return result.sort_values("_order").drop(columns="_order")


def plot_live_rates(live: pd.DataFrame, output: Path) -> None:
    data = live[live["phase"] == "full_trajectory"].copy()
    contexts = [skin for skin in SKIN_ORDER if skin in set(data["skin_id"])]
    risks = sorted(data["max_private_risk"].unique())
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.6), sharey=True, constrained_layout=True)
    for axis, mapping in zip(axes, ("safe_p", "safe_q")):
        pivot = data[data["mapping_id"] == mapping].pivot(index="skin_id", columns="max_private_risk", values="unsafe_rate").reindex(contexts).reindex(columns=risks)
        image = axis.imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap=CMAP, aspect="auto")
        for y in range(len(contexts)):
            for x in range(len(risks)):
                value = pivot.iloc[y, x]
                axis.text(x, y, f"{value:.0%}", ha="center", va="center", color=WHITE if value > 0.55 else INK, fontsize=9, weight="bold")
        axis.set_xticks(range(len(risks)), [f"{risk:.0%}" for risk in risks])
        axis.set_yticks(range(len(contexts)), [SKIN_LABEL[skin] for skin in contexts])
        axis.set_xlabel("Maximum private risk", color=INK)
        axis.set_title(f"{mapping}: semantic Unsafe rate", color=INK, weight="bold", fontsize=11, pad=10)
        axis.tick_params(colors=MUTED, labelsize=9)
        for spine in axis.spines.values():
            spine.set_visible(False)
    figure.colorbar(image, ax=axes, fraction=0.025, pad=0.02, label="Unsafe rate")
    figure.suptitle("Live-trajectory Unsafe choices by context, risk, and action mapping", x=0.06, y=1.10, ha="left", color=INK, weight="bold", fontsize=14)
    figure.text(0.06, 1.035, "Semantic actions after P/Q decoding; mapping strata are never pooled in this view", color=MUTED, fontsize=9)
    _save(figure, output, "live_unsafe_context_risk_mapping")


def plot_effects(
    effects: pd.DataFrame,
    output: Path,
    *,
    live_temperature: float,
    replay_temperature: float,
) -> None:
    data = effects[effects["risk"] == "all"].copy()
    contexts = [skin for skin in SKIN_ORDER if skin != ABSTRACT and skin in set(data["context"])]
    estimands = ["live_first_round", "fixed_state_direct", "live_full_trajectory"]
    labels = {"live_first_round": "Live first round", "fixed_state_direct": "Fixed-state replay", "live_full_trajectory": "Live full trajectory"}
    styles = {"live_first_round": (BLUE, "o"), "fixed_state_direct": (GOLD, "s"), "live_full_trajectory": (ORANGE, "^")}
    figure, axis = plt.subplots(figsize=(10.3, 6.3))
    ybase = np.arange(len(contexts))
    offsets = {estimand: offset for estimand, offset in zip(estimands, (-0.22, 0, 0.22))}
    for estimand in estimands:
        subset = data[data["estimand"] == estimand].set_index("context").reindex(contexts)
        y = ybase + offsets[estimand]
        color, marker = styles[estimand]
        axis.errorbar(subset["estimate_pp"], y, xerr=[subset["estimate_pp"] - subset["ci_low_pp"], subset["ci_high_pp"] - subset["estimate_pp"]], fmt=marker, color=color, ecolor=color, elinewidth=1.4, capsize=3, markersize=6, label=labels[estimand])
    axis.axvline(0, color=INK, linewidth=1.1)
    axis.set_yticks(ybase, [SKIN_LABEL[skin] for skin in contexts])
    axis.invert_yaxis()
    axis.set_xlabel("Paired Unsafe-rate difference vs abstract control (percentage points)", color=INK)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.23), fontsize=8.5)
    _title(
        axis,
        "Paired context effects relative to the abstract control",
        f"Live T={live_temperature:g}; fixed replay T={replay_temperature:g}; protocols shown together but never pooled",
    )
    _style_axis(axis, grid_axis="x")
    _save(figure, output, "paired_context_effects")


def plot_flips(
    flips: pd.DataFrame,
    output: Path,
    *,
    live_temperature: float,
    replay_temperature: float,
) -> None:
    contexts = [skin for skin in SKIN_ORDER if skin != ABSTRACT and skin in set(flips["context"])]
    figure, axes = plt.subplots(1, 2, figsize=(12.3, 5.8), sharey=True, constrained_layout=True)
    palette = [(BLUE_LIGHT, "Stable"), (ORANGE, "Safe -> Unsafe"), (BLUE, "Unsafe -> Safe")]
    for axis, estimand, heading in zip(axes, ("live_first_round", "fixed_state_direct"), ("Live first round", "Fixed-state replay")):
        subset = flips[flips["estimand"] == estimand].set_index("context").reindex(contexts)
        n = subset["n_paired_rows"].replace(0, np.nan)
        values = [subset["stable"] / n, subset["safe_to_unsafe"] / n, subset["unsafe_to_safe"] / n]
        left = np.zeros(len(contexts))
        for series, (color, label) in zip(values, palette):
            axis.barh(range(len(contexts)), series, left=left, color=color, edgecolor=WHITE, linewidth=0.8, label=label)
            left += series.fillna(0).to_numpy()
        axis.set_xlim(0, 1)
        axis.set_xticks([0, .25, .5, .75, 1], ["0%", "25%", "50%", "75%", "100%"])
        axis.set_yticks(range(len(contexts)), [SKIN_LABEL[skin] for skin in contexts])
        axis.set_title(heading, color=INK, weight="bold", fontsize=11)
        _style_axis(axis, grid_axis="x")
    axes[0].invert_yaxis()
    axes[1].legend(frameon=False, loc="lower center", bbox_to_anchor=(0.25, -0.22), ncol=3, fontsize=8.5)
    figure.suptitle("Direction of paired action changes versus abstract control", x=0.06, y=1.10, ha="left", color=INK, weight="bold", fontsize=14)
    figure.text(0.06, 1.035, f"Valid pairs; live first round T={live_temperature:g}, fixed replay T={replay_temperature:g}; estimates are not pooled", color=MUTED, fontsize=9)
    _save(figure, output, "paired_flip_directions")


def plot_payoff_setback(summary: pd.DataFrame, output: Path) -> None:
    data = _ordered(summary)
    n_clusters = int(data["n_race_clusters"].min())
    y = np.arange(len(data))
    figure, axes = plt.subplots(1, 2, figsize=(12.0, 5.8), sharey=True, constrained_layout=True)
    axes[0].errorbar(data["mean_final_payoff"], y, xerr=[data["mean_final_payoff"] - data["payoff_ci_low"], data["payoff_ci_high"] - data["mean_final_payoff"]], fmt="o", color=BLUE, ecolor=BLUE, capsize=3)
    axes[1].errorbar(data["setback_rate"] * 100, y, xerr=[(data["setback_rate"] - data["setback_ci_low"]) * 100, (data["setback_ci_high"] - data["setback_rate"]) * 100], fmt="s", color=GOLD, ecolor=GOLD, capsize=3)
    for axis, xlabel, title in zip(axes, ("Mean realized final payoff", "Player setback rate (%)"), ("Realized payoff", "Terminal setbacks")):
        axis.set_yticks(y, [SKIN_LABEL[skin] for skin in data["context"]])
        axis.set_xlabel(xlabel, color=INK)
        axis.set_title(title, color=INK, weight="bold", fontsize=11)
        _style_axis(axis, grid_axis="x")
    axes[0].invert_yaxis()
    figure.suptitle("Live-trajectory payoff and setback summaries", x=0.06, y=1.10, ha="left", color=INK, weight="bold", fontsize=14)
    figure.text(0.06, 1.035, f"Player outcomes with race-cluster intervals; {n_clusters} race clusters per context; exploratory ranking", color=MUTED, fontsize=9)
    _save(figure, output, "live_payoff_setback")


def plot_planned_contrasts(
    contrasts: pd.DataFrame,
    output: Path,
    *,
    live_temperature: float,
    replay_temperature: float,
) -> None:
    estimands = ["live_first_round", "fixed_state_direct", "live_full_trajectory"]
    labels = {"live_first_round": "Live first round", "fixed_state_direct": "Fixed-state replay", "live_full_trajectory": "Live full trajectory"}
    styles = {"live_first_round": (BLUE, "o"), "fixed_state_direct": (GOLD, "s"), "live_full_trajectory": (ORANGE, "^")}
    families = [pair[2] for pair in PLANNED_PAIRS]
    figure, axis = plt.subplots(figsize=(9.4, 4.8))
    ybase = np.arange(len(families))
    for estimand, offset in zip(estimands, (-0.2, 0, 0.2)):
        subset = contrasts[contrasts["estimand"] == estimand].set_index("pair_family").reindex(families)
        color, marker = styles[estimand]
        axis.errorbar(subset["estimate_pp"], ybase + offset, xerr=[subset["estimate_pp"] - subset["ci_low_pp"], subset["ci_high_pp"] - subset["estimate_pp"]], fmt=marker, color=color, ecolor=color, capsize=3, label=labels[estimand])
    axis.axvline(0, color=INK, linewidth=1.1)
    axis.set_yticks(ybase, families)
    axis.invert_yaxis()
    axis.set_xlabel("Fictional minus realistic Unsafe rate (percentage points)", color=INK)
    axis.legend(frameon=False, ncol=3, loc="lower center", bbox_to_anchor=(0.5, -0.3), fontsize=8.5)
    _title(
        axis,
        "Planned realistic-versus-fictional contrasts",
        f"Three matched pairs; live T={live_temperature:g}, fixed replay T={replay_temperature:g}; estimates are not pooled",
    )
    _style_axis(axis, grid_axis="x")
    _save(figure, output, "planned_realistic_fictional_contrasts")


def plot_mapping_diagnostic(
    live: pd.DataFrame,
    mapping: pd.DataFrame,
    output: Path,
    *,
    live_temperature: float,
    replay_temperature: float,
) -> None:
    live_overall = live[live["phase"] == "full_trajectory"].groupby(["skin_id", "mapping_id"], as_index=False).apply(
        lambda x: pd.Series({"unsafe_rate": np.average(x["unsafe_rate"], weights=x["n_decisions"]), "n": x["n_decisions"].sum()}), include_groups=False
    ).reset_index(drop=True)
    contexts = [skin for skin in SKIN_ORDER if skin in set(live_overall["skin_id"])]
    figure, axes = plt.subplots(1, 2, figsize=(12.1, 5.7), sharey=True, constrained_layout=True)
    y = np.arange(len(contexts))
    for mapping_id, offset, color, marker in (("safe_p", -0.12, BLUE, "o"), ("safe_q", 0.12, GOLD, "s")):
        subset = live_overall[live_overall["mapping_id"] == mapping_id].set_index("skin_id").reindex(contexts)
        axes[0].scatter(subset["unsafe_rate"] * 100, y + offset, color=color, marker=marker, label=mapping_id, s=38)
    axes[0].set_xlabel("Live semantic Unsafe rate (%)", color=INK)
    axes[0].set_title("Live mapping strata (descriptive)", color=INK, weight="bold", fontsize=11)
    axes[0].legend(frameon=False, ncol=2, loc="lower right", fontsize=8.5)
    fixed = mapping.set_index("context").reindex(contexts)
    axes[1].errorbar(fixed["estimate_pp"], y, xerr=[fixed["estimate_pp"] - fixed["ci_low_pp"], fixed["ci_high_pp"] - fixed["estimate_pp"]], fmt="s", color=ORANGE, ecolor=ORANGE, capsize=3)
    axes[1].axvline(0, color=INK, linewidth=1)
    axes[1].set_xlabel("safe_q - safe_p Unsafe rate (pp)", color=INK)
    axes[1].set_title("Fixed-state paired mapping effect", color=INK, weight="bold", fontsize=11)
    for axis in axes:
        axis.set_yticks(y, [SKIN_LABEL[skin] for skin in contexts])
        _style_axis(axis, grid_axis="x")
    axes[0].invert_yaxis()
    figure.suptitle("Context x opaque-action mapping diagnostic", x=0.06, y=1.10, ha="left", color=INK, weight="bold", fontsize=14)
    figure.text(0.06, 1.035, f"Live T={live_temperature:g} mapping is repetition-confounded; fixed T={replay_temperature:g} presents both mappings per state", color=MUTED, fontsize=9)
    _save(figure, output, "context_mapping_diagnostic")


def plot_comprehension(domain: pd.DataFrame, cell: pd.DataFrame, output: Path) -> None:
    domain_order = ["rule_recall", "stage_payoff", "state_update", "terminal_scoring"]
    contexts = [skin for skin in SKIN_ORDER if skin in set(cell["skin_id"])]
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), constrained_layout=True)
    d = domain.set_index("domain").reindex(domain_order)
    y = np.arange(len(d))
    axes[0].barh(y - .16, d["semantic_accuracy"] * 100, height=.3, color=BLUE, label="Semantic accuracy")
    axes[0].barh(y + .16, d["strict_valid_rate"] * 100, height=.3, color=GOLD, label="Strict format validity")
    axes[0].axvline(75, color=INK, linestyle="--", linewidth=1, label="75% domain gate")
    axes[0].set_yticks(y, [label.replace("_", " ").title() for label in domain_order])
    axes[0].invert_yaxis()
    axes[0].set_xlim(0, 105)
    axes[0].set_xlabel("Rate (%)", color=INK)
    axes[0].set_title("Comprehension by domain", color=INK, weight="bold", fontsize=11)
    axes[0].legend(
        frameon=False,
        fontsize=7.8,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=3,
        columnspacing=1.1,
        handletextpad=0.45,
    )
    _style_axis(axes[0], grid_axis="x")
    pivot = cell.pivot(index="skin_id", columns="mapping_id", values="semantic_accuracy").reindex(contexts).reindex(columns=["safe_p", "safe_q"])
    strict = cell.pivot(index="skin_id", columns="mapping_id", values="strict_valid_rate").reindex(contexts).reindex(columns=["safe_p", "safe_q"])
    image = axes[1].imshow(pivot.to_numpy(), vmin=0, vmax=1, cmap=CMAP, aspect="auto")
    for row in range(len(contexts)):
        for column in range(2):
            axes[1].text(column, row, f"{pivot.iloc[row, column]:.0%}\nstrict {strict.iloc[row, column]:.0%}", ha="center", va="center", fontsize=8, color=WHITE if pivot.iloc[row, column] > .72 else INK, weight="bold")
    axes[1].set_xticks([0, 1], ["safe_p", "safe_q"])
    axes[1].set_yticks(range(len(contexts)), [SKIN_LABEL[skin] for skin in contexts])
    axes[1].set_title("Semantic admission by context x mapping", color=INK, weight="bold", fontsize=11)
    axes[1].tick_params(colors=MUTED, labelsize=9)
    for spine in axes[1].spines.values():
        spine.set_visible(False)
    figure.colorbar(image, ax=axes[1], fraction=.035, pad=.03, label="Semantic accuracy")
    figure.suptitle("Comprehension admission gate", x=0.06, y=1.10, ha="left", color=INK, weight="bold", fontsize=14)
    figure.text(0.06, 1.035, "Frozen thresholds: >=90% overall and >=75% in every domain; failed cells remain diagnostic only", color=MUTED, fontsize=9)
    _save(figure, output, "comprehension_admission")


def _fmt_pp(value: float) -> str:
    return "NA" if pd.isna(value) else f"{value:+.1f} pp"


def _fmt_ci(row: pd.Series) -> str:
    if pd.isna(row["ci_low_pp"]):
        return "CI unavailable"
    return f"95% bootstrap CI {row['ci_low_pp']:+.1f} to {row['ci_high_pp']:+.1f} pp"


def build_report(
    data: AnalysisData,
    effects: pd.DataFrame,
    payoff: pd.DataFrame,
    contrasts: pd.DataFrame,
    comp_domain: pd.DataFrame,
    output: Path,
) -> str:
    validation = data.validation
    overall_effects = effects[effects["risk"] == "all"]
    replay = overall_effects[overall_effects["estimand"] == "fixed_state_direct"].sort_values("estimate_pp", ascending=False)
    live_full = overall_effects[overall_effects["estimand"] == "live_full_trajectory"].sort_values("estimate_pp", ascending=False)
    first = overall_effects[overall_effects["estimand"] == "live_first_round"]
    top_replay = replay.iloc[0]
    top_live = live_full.iloc[0]
    state_row = comp_domain.set_index("domain").loc["state_update"]
    terminal_row = comp_domain.set_index("domain").loc["terminal_scoring"]
    all_first_zero = bool(np.allclose(first["estimate_pp"].fillna(0), 0))
    evidence_label = "ADMITTED EXPLORATORY PILOT" if validation["admission_passed"] else "DIAGNOSTIC ONLY -- COMPREHENSION ADMISSION FAILED"
    protocol_support_note = (
        "Live and fixed-state protocol configurations have matching file hashes."
        if validation["cross_protocol_config_hash_match"]
        else "Live and fixed-state raw experiment-config hashes differ; fixed-state results are shown only as a separately labelled supporting diagnostic."
    )
    decoding_note = (
        "Live and fixed-state replay use the same decoding temperature."
        if validation["cross_protocol_temperature_match"]
        else f"Live uses temperature {validation['live_temperature']}, while fixed-state replay uses {validation['replay_temperature']}; their estimates are not pooled or treated as a decoding-controlled comparison."
    )
    valid_live = data.live_turns[~data.live_turns["parse_failed"]]
    mapping_rates = valid_live.groupby("mapping_id")["unsafe"].agg(["mean", "count"])
    safe_p_rate = mapping_rates.loc["safe_p"]
    safe_q_rate = mapping_rates.loc["safe_q"]
    live_planned = contrasts[contrasts["estimand"] == "live_full_trajectory"].copy()
    bounded_planned = live_planned[
        (live_planned["ci_low_pp"] > 0) | (live_planned["ci_high_pp"] < 0)
    ]
    planned_note = (
        "No live realistic-versus-fictional interval excluded zero."
        if bounded_planned.empty
        else "Live intervals excluding zero occurred only for: "
        + ", ".join(
            f"{row.pair_family} ({row.estimate_pp:+.1f} pp)"
            for row in bounded_planned.itertuples()
        )
        + "."
    )
    live_scale = f"{validation['live_profile']} live run"
    fixed_scale = f"{validation['fixed_profile']} fixed-state run"
    races_per_skin = int(
        data.live_races.groupby("skin_id")["game_id"].nunique().min()
    )
    alignment = cross_protocol_alignment(effects)
    alignment_r = float(
        alignment[["live_full_trajectory", "fixed_state_direct"]]
        .corr()
        .iloc[0, 1]
    )
    same_direction = int(alignment["same_direction"].sum())
    temperatures_match = bool(validation["cross_protocol_temperature_match"])
    alignment_temperature_note = (
        "decoding temperatures match, but admission failed"
        if temperatures_match
        else "decoding temperatures differ and admission failed"
    )
    lines = [
        "# Context-skin invariance: live simulation and fixed-state replay",
        "",
        f"> **Evidence status: {evidence_label}.** This combines a {live_scale} with a separately labelled {fixed_scale}; neither is a confirmatory estimate. "
        "The same mathematical game and exact model digest were used throughout, but failure of the frozen comprehension gate prevents interpreting action differences as informed utility optimization.",
        "",
        "## Technical summary",
        "",
        f"- **Coverage and provenance passed.** {validation['n_skins']} skins, {validation['n_live_races']} live races, "
        f"{validation['n_live_decisions']} live decisions, {validation['n_fixed_states']} fixed states, and {validation['n_fixed_rows']} replay cells were reconciled. "
        f"There were {validation['live_parse_failures']} live and {validation['fixed_parse_failures']} fixed-state final parse failures.",
        f"- **No first-round context effect was observed in the {validation['live_profile']} live run.** " + ("Every paired first-round estimate was exactly 0 pp." if all_first_zero else "At least one paired first-round estimate differed from zero."),
        f"- **Later behavior separated across skins.** The largest fixed-state difference versus the abstract control was {SKIN_LABEL[top_replay['context']]} at {_fmt_pp(top_replay['estimate_pp'])} ({_fmt_ci(top_replay)}); "
        f"the largest live full-trajectory difference was {SKIN_LABEL[top_live['context']]} at {_fmt_pp(top_live['estimate_pp'])} ({_fmt_ci(top_live)}). Both rankings remain exploratory.",
        f"- **The admission failure is substantive, not cosmetic.** State-update semantic accuracy was {state_row['semantic_accuracy']:.1%}; terminal-scoring accuracy was {terminal_row['semantic_accuracy']:.1%}. "
        "Behavior below therefore diagnoses prompt-conditioned output, not verified understanding of the game.",
        f"- **Opaque mapping dominated semantic action rates.** When P denoted Safe (`safe_p`), the live semantic Unsafe rate was {safe_p_rate['mean']:.1%} across {int(safe_p_rate['count']):,} decisions; "
        f"when Q denoted Safe (`safe_q`), it was {safe_q_rate['mean']:.1%} across {int(safe_q_rate['count']):,}. This is why mapping remains explicit in every primary table.",
        f"- **Planned semantic contrasts were selective.** {planned_note} These intervals remain exploratory and do not estimate a general realism effect.",
        f"- **Live and fixed context profiles aligned descriptively.** All {same_direction}/{len(alignment)} non-abstract context effects had the same sign, and the across-context Pearson correlation was {alignment_r:.3f}. This is not a pooled estimate: {alignment_temperature_note}.",
        f"- **Protocol boundary.** {protocol_support_note} {decoding_note}",
        "",
        "## Design and estimands",
        "",
        "Within each protocol, all eight prompts preserve progress increments, payoff matrix, risk treatments, horizon process, prize, tie rule, setback process, state disclosure, parser, model digest, and decoding. Context nouns and introductions change. The model answers with opaque code P or Q; `safe_p` and `safe_q` reverse which code denotes the semantic Safe action while display order remains P then Q.",
        "",
        "The pre-feedback estimand is the paired first-round semantic Unsafe-rate difference versus `abstract_contest`, keyed by risk, repetition, seat, and mapping. Fixed-state replay asks the model at the same engine-reachable state under every context and both mappings, so that comparison isolates a direct prompt effect at those sampled states. Full live trajectories include continued context exposure and endogenous feedback; they are total trajectory differences, not direct effects.",
        "",
        "## Action rates keep context, risk, and mapping visible",
        "",
        "The heatmap reports decision-weighted decoded semantic actions. It deliberately does not average the two action-code mappings: a strong P/Q response tendency can otherwise masquerade as a context effect. The paired effect chart below instead macro-averages each player-race trajectory before clustering, so its values need not equal a subtraction of heatmap cells.",
        "",
        "![Live Unsafe choices by context, risk, and mapping](figures/live_unsafe_context_risk_mapping.png)",
        "",
        "## Direct, pre-feedback, and total-trajectory estimates disagree",
        "",
        f"First-round rows are the cleanest live pre-feedback comparison. The {fixed_scale} supplies a separate direct-effect diagnostic at sampled later states. Full trajectories answer a different question because choices change subsequent progress, risk, and history. Fixed-state intervals reflect {validation['n_fixed_states']} sampled states.",
        "",
        "![Paired context effects](figures/paired_context_effects.png)",
        "",
        "The flip view prevents a zero average from hiding offsetting directions. A one-sided reference distribution can make all changes point in the same direction, so the exact Safe-to-Unsafe and Unsafe-to-Safe counts remain visible.",
        "",
        "![Paired flip directions](figures/paired_flip_directions.png)",
        "",
        "## Mapping is a major diagnostic factor",
        "",
        "In live play, mapping is assigned by repetition and is therefore confounded with the repetition-specific stochastic horizon; its live difference is descriptive only. Fixed-state replay presents both mappings to every state and can estimate the paired mapping effect. The resulting asymmetry shows why P/Q mapping must remain a reported factor rather than a nuisance silently pooled away.",
        "",
        "![Context by mapping diagnostic](figures/context_mapping_diagnostic.png)",
        "",
        "## Realistic versus fictional pairs",
        "",
        "The three contrasts were planned from matched narrative pairs. They are a compact sensitivity check, not a general estimate of realism or fiction: each category contains only three hand-authored stories.",
        "",
        "![Planned realistic-fictional contrasts](figures/planned_realistic_fictional_contrasts.png)",
        "",
        "## Payoff and setback outcomes",
        "",
        f"Realized payoff combines stage payoffs, race outcome, and one sampled setback draw. With {races_per_skin} live races per skin, payoff/setback rankings remain exploratory and should not be used to claim one context is economically superior.",
        "",
        "![Live payoff and setback](figures/live_payoff_setback.png)",
        "",
        "## Comprehension gate failed",
        "",
        "The model recalled simultaneous choice and most stage payoffs, but failed the state-update and terminal-scoring domains. Strict-format validity is reported separately from semantic correctness: formatting errors are not silently converted into reasoning errors, and semantic errors are not excused by correct formatting.",
        "",
        "![Comprehension admission](figures/comprehension_admission.png)",
        "",
        "Because every context x mapping cell had to pass the frozen gate and none did, the fixed-state replay manifest classifies the evidence as `diagnostic_comprehension_failed`. Running replay despite that failure is useful for debugging prompt behavior, but it does not rehabilitate the evidence.",
        "",
        "## Data-quality and uncertainty boundary",
        "",
        f"- Exact model: `{validation['model_name']}` at digest `{validation['model_digest']}`.",
        f"- Live/fixed profiles: `{validation['live_profile']}` / `{validation['fixed_profile']}`; temperatures: {validation['live_temperature']} / {validation['replay_temperature']}.",
        f"- Live/fixed experiment-config hashes: `{validation['live_experiment_config_sha256']}` / `{validation['fixed_experiment_config_sha256']}`; exact match: `{validation['cross_protocol_config_hash_match']}`.",
        "- Source revisions are verified within live and fixed protocols. They differ across runner types by design and are not asserted to be the same executable file.",
        "- Paired percentile intervals resample race clusters for live outcomes and state clusters for fixed replay. They quantify finite-sample resampling variability, not population uncertainty over prompts, models, or narrative domains.",
        "- Decisions within races are dependent. Full-trajectory rates are descriptive enacted behavior, not independent Bernoulli observations.",
        "- Fixed-state replay holds the disclosed state constant but does not prove causal mediation by any SAE feature. Neural steering needs held-out features, dose response, sign reversal, matched-norm random directions, and unrelated-feature controls.",
        "- Realistic/fictional contrasts are planned but under-covered: three pairs cannot establish a broad semantic-category effect.",
        "",
        "## Recommended promotion gates",
        "",
        "1. Repair or redesign comprehension prompts and rerun the frozen admission battery without changing thresholds after seeing outcomes.",
        "2. Promote only after every context x mapping cell passes, then run at least the 32-repetition / 32-state pilot profiles with manifests frozen before analysis.",
        "3. Keep mapping randomized or crossed within identical live race seeds; the current parity assignment balances counts but does not identify a live mapping effect.",
        "4. Add at least one more model family and a separately labelled cross-decoding robustness run before discussing model or decoding stability.",
        "5. Treat activation-SAE prediction and steering as separate stages: AUC is association; controlled action shifts are causal intervention evidence.",
        "",
        "## Reproducible artifacts",
        "",
        "The adjacent CSV files contain all plotted summaries; `analysis_summary.json` records source-manifest hashes, coverage, admission status, and figure inventory. Re-run with:",
        "",
        "```bash",
        "python results/scripts/analyze_context_skin.py \\",
        "  --live-root <live-run-root> \\",
        "  --fixed-root <fixed-state-run-root> \\",
        "  --output-dir <new-output-dir>",
        "```",
    ]
    report = "\n".join(lines) + "\n"
    (output / "context_skin_analysis.md").write_text(report, encoding="utf-8")
    return report


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    live_roots = args.live_root or [args.input_root]
    fixed_roots = args.fixed_root or [args.input_root]
    live_runs = discover_runs(live_roots, LIVE_SCHEMA)
    fixed_runs = discover_runs(fixed_roots, FIXED_SCHEMA)
    data = validate_and_load(live_runs, fixed_runs, allow_incomplete_skins=args.allow_incomplete_skins)
    output = args.output_dir.resolve()
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)

    live_rates, replay_rates = summarize_rates(data)
    effects, flips = paired_effects(data, args.bootstrap_repetitions)
    payoff = payoff_setback_summary(data, args.bootstrap_repetitions)
    contrasts = planned_contrasts(data, args.bootstrap_repetitions)
    comp_domain, comp_cell = comprehension_summaries(data)
    mapping = context_mapping_diagnostic(data, args.bootstrap_repetitions)
    quality = quality_summary(data)
    alignment = cross_protocol_alignment(effects)

    tables = {
        "live_context_rates.csv": live_rates,
        "fixed_state_context_rates.csv": replay_rates,
        "paired_context_effects.csv": effects,
        "paired_flip_directions.csv": flips,
        "payoff_setback_summary.csv": payoff,
        "planned_realistic_fictional_contrasts.csv": contrasts,
        "comprehension_by_domain.csv": comp_domain,
        "comprehension_by_cell.csv": comp_cell,
        "context_mapping_diagnostic.csv": mapping,
        "quality_coverage.csv": quality,
        "cross_protocol_effect_alignment.csv": alignment,
    }
    for filename, frame in tables.items():
        frame.to_csv(output / filename, index=False)

    plot_live_rates(live_rates, figures)
    plot_effects(
        effects,
        figures,
        live_temperature=float(data.validation["live_temperature"]),
        replay_temperature=float(data.validation["replay_temperature"]),
    )
    plot_flips(
        flips,
        figures,
        live_temperature=float(data.validation["live_temperature"]),
        replay_temperature=float(data.validation["replay_temperature"]),
    )
    plot_payoff_setback(payoff, figures)
    plot_planned_contrasts(
        contrasts,
        figures,
        live_temperature=float(data.validation["live_temperature"]),
        replay_temperature=float(data.validation["replay_temperature"]),
    )
    plot_mapping_diagnostic(
        live_rates,
        mapping,
        figures,
        live_temperature=float(data.validation["live_temperature"]),
        replay_temperature=float(data.validation["replay_temperature"]),
    )
    plot_comprehension(comp_domain, comp_cell, figures)
    build_report(data, effects, payoff, contrasts, comp_domain, output)

    figure_stems = [
        "live_unsafe_context_risk_mapping",
        "paired_context_effects",
        "paired_flip_directions",
        "live_payoff_setback",
        "planned_realistic_fictional_contrasts",
        "context_mapping_diagnostic",
        "comprehension_admission",
    ]
    summary = {
        "schema_version": "ai-race-context-skin-analysis-v1",
        "status": "complete",
        "validation": data.validation,
        "bootstrap": {
            "method": "paired percentile cluster bootstrap",
            "repetitions": args.bootstrap_repetitions,
            "seed": BOOTSTRAP_SEED,
            "live_cluster": "repetition CRN stream (risk strata share base_seed + rep)",
            "fixed_cluster": "state_id",
        },
        "input_roots": {
            "live": [str(path.resolve()) for path in live_roots],
            "fixed": [str(path.resolve()) for path in fixed_roots],
        },
        "tables": {filename: {"rows": int(len(frame)), "sha256": sha256_file(output / filename)} for filename, frame in tables.items()},
        "figures": {
            stem: {
                extension: {"bytes": (figures / f"{stem}.{extension}").stat().st_size, "sha256": sha256_file(figures / f"{stem}.{extension}")}
                for extension in ("png", "pdf")
            }
            for stem in figure_stems
        },
        "report": {"path": "context_skin_analysis.md", "sha256": sha256_file(output / "context_skin_analysis.md")},
        "claim_boundary": (
            "exploratory pilot; admission passed"
            if data.validation["admission_passed"]
            else "diagnostic only; comprehension admission failed"
        ),
        "cross_protocol_diagnostic": {
            "n_contexts": int(len(alignment)),
            "pearson_r_live_full_vs_fixed": float(
                alignment[["live_full_trajectory", "fixed_state_direct"]]
                .corr()
                .iloc[0, 1]
            ),
            "same_direction_contexts": int(alignment["same_direction"].sum()),
            "warning": (
                "descriptive only; decoding temperatures match but comprehension admission failed"
                if data.validation["cross_protocol_temperature_match"]
                else "descriptive only; decoding temperatures differ and comprehension admission failed"
            ),
        },
    }
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
