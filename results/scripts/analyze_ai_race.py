#!/usr/bin/env python3
"""Build descriptive AI Race tables from Kaggle experiment outputs.

The analyser expects each discovered run directory to contain ``turns.jsonl``,
``races.csv``, and ``players.csv``. It does not contain reference results or
synthetic observations: every output row is derived from supplied run logs. Primary
behavioural estimands use only parse-clean, canonical-mechanism races and never pool
prompt versions or decoding/seed/mechanism contracts implicitly.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "derived" / "ai_race_analysis"
# Frozen before any model output is inspected, so scoring the LLM against the
# human effects stays mechanical rather than chosen after seeing the estimates.
HUMAN_REFERENCE_PATH = Path(__file__).resolve().parent / "human_reference.json"
LOGIT_FORMULA = (
    "unsafe ~ C(max_private_risk) + first_round_unsafe + "
    "own_prev_unsafe * opponent_prev_unsafe * progress_gap_before"
)
# The six nested specifications of Table 1 in the source paper. Reporting only the
# saturated model would hide how unstable a coefficient is across specifications —
# in the human data the race-position term is significant in the interaction model
# and not in the additive one, and that instability is itself a finding.
LOGIT_SPECIFICATIONS: tuple[tuple[str, str], ...] = (
    ("1", "unsafe ~ C(max_private_risk)"),
    (
        "2",
        "unsafe ~ C(max_private_risk) + own_prev_unsafe + opponent_prev_unsafe "
        "+ progress_gap_before",
    ),
    (
        "3",
        "unsafe ~ C(max_private_risk) + own_prev_unsafe * opponent_prev_unsafe "
        "* progress_gap_before",
    ),
    ("4", "unsafe ~ C(max_private_risk) + first_round_unsafe"),
    (
        "5",
        "unsafe ~ C(max_private_risk) + first_round_unsafe + own_prev_unsafe "
        "+ opponent_prev_unsafe + progress_gap_before",
    ),
    ("6", LOGIT_FORMULA),
)
MISSING_PROMPT_VERSION = "__MISSING_PROMPT_VERSION__"
MISSING_RUN_PHASE = "__MISSING_RUN_PHASE__"
MISSING_RUN_STATUS = "__MISSING_RUN_STATUS__"
MISSING_PERSONA_CONDITION = "__MISSING_PERSONA_CONDITION__"
PROTOCOL_SIGNATURE_SCHEMA = "ai-race-analysis-protocol-signature-v1"
CANONICAL_PROMPT_VERSION = "ai-race-fairgame-v3"
# One template file per language carries the same promptVersion label, so the
# canonical check accepts any of the frozen template hashes rather than a single
# scalar. Editing a template — including whitespace — still requires a new
# promptVersion and a new entry here; the gate never accepts modified text
# relabelled as an existing version.
CANONICAL_PROMPT_SHA256_BY_TEMPLATE: dict[str, str] = {
    "ai_race_en": (
        "27086bd80378c25e859d03527a5ae55c1046f231ef7b914db9cb3c3b4fb2df3e"
    ),
    "ai_race_vi": (
        "a6d3f738cf58043ae0dadc351cac12da07bd60778317b0566d743f5e40a77510"
    ),
}
CANONICAL_PROMPT_SHA256S = frozenset(CANONICAL_PROMPT_SHA256_BY_TEMPLATE.values())
SUPPORTED_PROTOCOL_MANIFEST_SCHEMAS = {
    "ai-race-kaggle-run-v1",
    "ai-race-kbench-run-v1",
}

TURN_ALIASES: dict[str, tuple[str, ...]] = {
    "game_id": ("race_id",),
    "player_id": ("player", "participant_id", "agent_id", "seat"),
    "opponent_id": ("opponent", "opponent_player", "opponent_player_id"),
    "round": ("round_number", "round_index", "turn"),
    "model": ("model_id", "model_name"),
    "prompt_version": ("protocol_version", "prompt_protocol_version"),
    "run_phase": ("analysis_phase", "study_phase"),
    "max_private_risk": (
        "risk",
        "risk_level",
        "risk_probability",
        "private_risk_max",
        "p_max",
    ),
    "unsafe": ("is_unsafe",),
    "action": ("choice", "decision", "parsed_action"),
    "parse_failed": ("parse_fail", "parsing_failed"),
    "retry_count": ("retries", "n_retries"),
    "progress_gap_before": ("pre_decision_progress_gap", "gap_before"),
    "own_progress_before": ("player_progress_before",),
    "opponent_progress_before": ("other_progress_before",),
    "own_progress_after": ("progress_after", "player_progress_after"),
    "opponent_progress_after": ("other_progress_after",),
}

RACE_ALIASES: dict[str, tuple[str, ...]] = {
    "game_id": ("race_id",),
    "model": ("model_id", "model_name"),
    "prompt_version": ("protocol_version", "prompt_protocol_version"),
    "run_phase": ("analysis_phase", "study_phase"),
    "max_private_risk": (
        "risk",
        "risk_level",
        "risk_probability",
        "private_risk_max",
        "p_max",
    ),
    "n_rounds": ("rounds_played", "horizon"),
    "tie": ("tied", "is_tie"),
}

PLAYER_ALIASES: dict[str, tuple[str, ...]] = {
    "game_id": ("race_id",),
    "player_id": ("player", "participant_id", "agent_id", "seat"),
    "model": ("model_id", "model_name"),
    "prompt_version": ("protocol_version", "prompt_protocol_version"),
    "run_phase": ("analysis_phase", "study_phase"),
    "max_private_risk": (
        "risk",
        "risk_level",
        "risk_probability",
        "private_risk_max",
        "p_max",
    ),
    "outcome": ("race_outcome", "result"),
    "unsafe_frequency": ("unsafe_rate",),
}

RUN_KEY = ["source_run"]
RACE_KEY = [*RUN_KEY, "game_id"]
PLAYER_KEY = [*RACE_KEY, "player_id"]
TURN_KEY = [*PLAYER_KEY, "round"]
BASE_CONTEXT = ["model", "max_private_risk"]
CONTEXT = [
    *BASE_CONTEXT,
    # Persona does not change the prompt template, so it never shows up in
    # prompt_version or protocol_signature. It has to stratify every table on its
    # own, or a persona run and the neutral baseline would be averaged together.
    "persona_condition",
    "prompt_version",
    "protocol_signature",
    "run_phase",
    "run_status",
]


def _rename_aliases(
    frame: pd.DataFrame,
    aliases: dict[str, tuple[str, ...]],
) -> pd.DataFrame:
    """Rename the first available alias for each absent canonical column."""

    frame = frame.copy()
    renames: dict[str, str] = {}
    for canonical, candidates in aliases.items():
        if canonical in frame.columns:
            continue
        present = [candidate for candidate in candidates if candidate in frame.columns]
        if present:
            renames[present[0]] = canonical
    return frame.rename(columns=renames)


def _require_columns(frame: pd.DataFrame, required: Iterable[str], *, table: str) -> None:
    missing = sorted(set(required) - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing required columns: {', '.join(missing)}")


def _normalise_identifier(series: pd.Series, *, name: str) -> pd.Series:
    if series.isna().any():
        raise ValueError(f"{name} contains missing identifiers")
    result = series.astype(str).str.strip()
    if result.eq("").any():
        raise ValueError(f"{name} contains empty identifiers")
    return result


def _binary_value(value: Any, *, name: str, allow_action_words: bool = False) -> float:
    if pd.isna(value):
        return np.nan
    if isinstance(value, (bool, np.bool_)):
        return float(int(value))
    if isinstance(value, (int, np.integer)) and value in (0, 1):
        return float(value)
    if isinstance(value, (float, np.floating)) and math.isfinite(float(value)):
        if float(value) in (0.0, 1.0):
            return float(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        false_values = {"0", "false", "no"}
        true_values = {"1", "true", "yes"}
        if allow_action_words:
            false_values |= {"s", "safe"}
            true_values |= {"u", "unsafe"}
        if normalised in false_values:
            return 0.0
        if normalised in true_values:
            return 1.0
    raise ValueError(f"{name} must be binary; received {value!r}")


def _normalise_binary(
    series: pd.Series,
    *,
    name: str,
    allow_missing: bool = False,
    allow_action_words: bool = False,
) -> pd.Series:
    result = series.map(
        lambda value: _binary_value(
            value,
            name=name,
            allow_action_words=allow_action_words,
        )
    )
    if not allow_missing and result.isna().any():
        raise ValueError(f"{name} contains missing values")
    return result.astype("Float64")


def _normalise_numeric(
    series: pd.Series,
    *,
    name: str,
    allow_missing: bool = False,
) -> pd.Series:
    result = pd.to_numeric(series, errors="coerce")
    invalid = series.notna() & result.isna()
    if invalid.any():
        example = series.loc[invalid].iloc[0]
        raise ValueError(f"{name} contains a non-numeric value: {example!r}")
    finite = result.dropna().map(lambda value: math.isfinite(float(value)))
    if not finite.all():
        raise ValueError(f"{name} contains a non-finite value")
    if not allow_missing and result.isna().any():
        raise ValueError(f"{name} contains missing values")
    return result.astype(float)


def _normalise_optional_rep(frame: pd.DataFrame, *, table: str) -> pd.DataFrame:
    """Normalise a common-random-number repetition index when it is logged."""

    frame = frame.copy()
    if "rep" not in frame:
        return frame
    rep = _normalise_numeric(
        frame["rep"],
        name=f"{table}.rep",
        allow_missing=True,
    )
    present = rep.notna()
    if (
        (rep.loc[present] < 0).any()
        or not np.equal(
            rep.loc[present],
            np.floor(rep.loc[present]),
        ).all()
    ):
        raise ValueError(f"{table}.rep must contain non-negative integers")
    frame["rep"] = rep.astype("Int64")
    return frame


def _discover_run_directories(inputs: Sequence[Path]) -> list[Path]:
    runs: set[Path] = set()
    for raw_path in inputs:
        path = raw_path.expanduser().resolve()
        if not path.exists():
            continue
        if path.is_file():
            if path.name != "turns.jsonl":
                raise ValueError(
                    f"input file must be turns.jsonl, not {path.name!r}: {path}"
                )
            runs.add(path.parent)
            continue
        if (path / "turns.jsonl").is_file():
            runs.add(path)
        runs.update(candidate.parent for candidate in path.rglob("turns.jsonl"))
    if not runs:
        rendered = ", ".join(str(path) for path in inputs)
        raise FileNotFoundError(
            "no run directory containing turns.jsonl was found under: "
            f"{rendered}. Run the experiment on Kaggle before analysing results."
        )
    return sorted(runs, key=lambda path: path.as_posix())


def _read_turns(path: Path, *, source_run: str) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            records.append(
                {
                    **record,
                    "source_run": source_run,
                    "source_line": line_number,
                }
            )
    if not records:
        raise ValueError(f"{path} contains no decision records")
    return pd.DataFrame.from_records(records)


def _manifest_text(
    manifest: dict[str, Any],
    key: str,
    *,
    path: Path,
) -> str:
    value = manifest.get(key)
    text = str(value).strip() if value is not None else ""
    if not text:
        raise ValueError(f"{path} is missing non-empty manifest field {key!r}")
    return text


def _manifest_mapping(
    manifest: dict[str, Any],
    key: str,
    *,
    path: Path,
) -> dict[str, Any]:
    value = manifest.get(key)
    if not isinstance(value, dict) or not value:
        raise ValueError(f"{path} must contain a non-empty object at {key!r}")
    return value


def _require_mapping_keys(
    value: dict[str, Any],
    required: Iterable[str],
    *,
    path: Path,
    field: str,
) -> None:
    missing = sorted(set(required) - set(value))
    if missing:
        raise ValueError(
            f"{path} field {field!r} is missing protocol provenance keys: "
            f"{', '.join(missing)}"
        )


def _is_canonical_prompt(version: Any, sha256: Any) -> bool:
    """Accept only a frozen template hash under the canonical version label."""

    return (
        version == CANONICAL_PROMPT_VERSION
        and isinstance(sha256, str)
        and sha256 in CANONICAL_PROMPT_SHA256S
    )


def _canonical_protocol_signature(payload: dict[str, Any], *, path: Path) -> str:
    try:
        canonical = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{path} contains protocol provenance that cannot be canonicalised"
        ) from exc
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _verified_protocol_payload(
    manifest: dict[str, Any],
    *,
    path: Path,
) -> tuple[dict[str, Any], str]:
    """Build the exact run contract used to prevent silent protocol pooling."""

    schema_version = _manifest_text(manifest, "schema_version", path=path)
    if schema_version not in SUPPORTED_PROTOCOL_MANIFEST_SCHEMAS:
        raise ValueError(
            f"{path} uses unsupported schema_version={schema_version!r}; "
            "primary analysis accepts the provenance-rich offline Kaggle or "
            "Kaggle Benchmark manifests"
        )

    source_sha256 = _manifest_text(manifest, "source_sha256", path=path)
    prompt_version = _manifest_text(manifest, "prompt_version", path=path)
    prompt_sha256 = _manifest_text(manifest, "prompt_sha256", path=path)
    decoding = _manifest_mapping(manifest, "decoding", path=path)
    package_versions = _manifest_mapping(manifest, "package_versions", path=path)

    if schema_version == "ai-race-kaggle-run-v1":
        model = _manifest_mapping(manifest, "model", path=path)
        _require_mapping_keys(
            model,
            ("short_name", "path", "engine", "config_sha256"),
            path=path,
            field="model",
        )
        model_label = str(model["short_name"]).strip()
        model_path = str(model["path"]).strip()
        model_engine = str(model["engine"]).strip()
        model_config_sha256 = (
            str(model["config_sha256"]).strip()
            if model["config_sha256"] is not None
            else ""
        )
        if not all(
            (model_label, model_path, model_engine, model_config_sha256)
        ):
            raise ValueError(
                f"{path} offline model provenance requires non-empty short_name, "
                "path, engine, and config_sha256"
            )
        _require_mapping_keys(
            decoding,
            (
                "temperature",
                "max_tokens",
                "logprobs",
                "logprobs_enabled",
                "max_parse_retries",
            ),
            path=path,
            field="decoding",
        )
        experiment = _manifest_mapping(manifest, "experiment", path=path)
        _require_mapping_keys(
            experiment,
            ("games", "seed"),
            path=path,
            field="experiment",
        )
        games = experiment["games"]
        if (
            not isinstance(games, list)
            or not games
            or any(not str(game).strip() for game in games)
        ):
            raise ValueError(
                f"{path} experiment.games must be a non-empty list of identifiers"
            )
        game_config_sha256 = _manifest_mapping(
            manifest,
            "game_config_sha256",
            path=path,
        )
        missing_game_hashes = sorted(
            str(game) for game in games if str(game) not in game_config_sha256
        )
        empty_game_hashes = sorted(
            str(game)
            for game in games
            if not str(game_config_sha256.get(str(game), "")).strip()
        )
        if missing_game_hashes or empty_game_hashes:
            raise ValueError(
                f"{path} game_config_sha256 does not fully identify experiment.games"
            )
        model_identity: dict[str, Any] = {
            "label": model_label,
            "path": model_path,
            "engine": model_engine,
            "config_sha256": model_config_sha256,
            "route": None,
        }
        seed_contract: dict[str, Any] = {
            "provenance": (
                "ai-race-kaggle-run-v1 deterministic per-decision seed contract"
            ),
            "base_seed_field": "experiment.seed",
            "base_seed_value_in_signature": False,
            "seed_forwarded_to_backend": True,
            "backend": model_engine,
        }
        mechanism: dict[str, Any] = {
            "games": [str(game) for game in games],
            "game_config_sha256": game_config_sha256,
        }
    else:
        model_value = manifest.get("model")
        model_label = str(model_value).strip() if model_value is not None else ""
        model_route = _manifest_text(manifest, "model_route", path=path)
        if not model_label:
            raise ValueError(f"{path} is missing a non-empty model label")
        _require_mapping_keys(
            decoding,
            (
                "temperature_requested",
                "temperature_forwarded_by_sdk",
                "temperature_effective",
                "temperature_effective_confirmed",
                "temperature_status",
                "output_token_limit_parameter",
                "output_token_limit",
                "max_parse_retries",
                "max_transport_retries",
            ),
            path=path,
            field="decoding",
        )
        seed_contract = _manifest_mapping(
            manifest,
            "sampling_seed_provenance",
            path=path,
        )
        _require_mapping_keys(
            seed_contract,
            (
                "requested",
                "forwarded_by_sdk",
                "applied",
                "applied_known",
                "status",
                "strip_detection",
            ),
            path=path,
            field="sampling_seed_provenance",
        )
        mechanism = _manifest_mapping(manifest, "mechanism", path=path)
        _require_mapping_keys(
            mechanism,
            (
                "minimum_rounds",
                "stop_probability",
                "risk_levels",
                "race_prize",
                "stage_payoff",
                "progress",
            ),
            path=path,
            field="mechanism",
        )
        model_identity = {
            "label": model_label,
            "path": None,
            "engine": manifest.get("llm_backend_mro"),
            "config_sha256": None,
            "route": model_route,
        }

    payload = {
        "signature_schema": PROTOCOL_SIGNATURE_SCHEMA,
        "manifest_schema": schema_version,
        "source_sha256": source_sha256,
        "prompt": {
            "version": prompt_version,
            "sha256": prompt_sha256,
            "canonical_version": CANONICAL_PROMPT_VERSION,
            # The accepted hash set is deliberately not part of the signature: it
            # is analyser configuration, and adding a language later must not
            # silently rewrite the signatures of already-analysed runs.
            "canonical_match": _is_canonical_prompt(prompt_version, prompt_sha256),
        },
        "model": model_identity,
        "decoding": decoding,
        "seed_contract": seed_contract,
        "mechanism": mechanism,
        "runtime": {
            "package_versions": package_versions,
        },
    }
    return payload, model_label


def _protocol_contract_from_manifest(
    manifest: dict[str, Any],
    *,
    path: Path,
    source_run: str,
    allow_unverified: bool,
) -> tuple[str, dict[str, Any], str | None]:
    try:
        payload, model_label = _verified_protocol_payload(manifest, path=path)
    except ValueError as exc:
        if not allow_unverified:
            raise
        model_value = manifest.get("model")
        if isinstance(model_value, dict):
            fallback_label = str(model_value.get("short_name", "")).strip()
        else:
            fallback_label = str(model_value).strip() if model_value is not None else ""
        payload = {
            "signature_schema": PROTOCOL_SIGNATURE_SCHEMA,
            "verification": "unverified_protocol_audit_only",
            "source_run": source_run,
            "provenance_error": str(exc),
            "available_manifest": manifest,
        }
        model_label = fallback_label or None
    return (
        _canonical_protocol_signature(payload, path=path),
        payload,
        model_label,
    )


def _read_run_manifest(
    run_directory: Path,
    *,
    allow_nonfinal_runs: bool,
) -> tuple[dict[str, Any], str, str | None]:
    path = run_directory / "run_manifest.json"
    if not path.is_file():
        if not allow_nonfinal_runs:
            raise FileNotFoundError(
                f"{run_directory} has no run_manifest.json; refusing an unverified "
                "or possibly partial run. Use --allow-nonfinal-runs only for an "
                "explicit protocol-health audit."
            )
        return {}, MISSING_RUN_STATUS, None
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON") from exc
    if not isinstance(manifest, dict):
        raise ValueError(f"{path}: expected a JSON object")

    status_value = manifest.get("status")
    status = str(status_value).strip().lower() if status_value is not None else ""
    if not status:
        if not allow_nonfinal_runs:
            raise ValueError(
                f"{path} has no status; refusing an unverified run. Use "
                "--allow-nonfinal-runs only for a protocol-health audit."
            )
        status = MISSING_RUN_STATUS
    if status != "completed" and not allow_nonfinal_runs:
        raise ValueError(
            f"{path} reports status={status!r}, not 'completed'. Failed, running, "
            "or protocol-failed runs are excluded from primary analysis; use "
            "--allow-nonfinal-runs only for an explicitly stratified health audit."
        )

    phase_value = manifest.get("run_phase")
    if phase_value is None and isinstance(manifest.get("experiment"), dict):
        phase_value = manifest["experiment"].get("runPhase")
    phase = str(phase_value).strip().lower() if phase_value is not None else None
    return manifest, status, phase or None


def _read_run_tables(
    run_directories: Sequence[Path],
    *,
    allow_nonfinal_runs: bool,
    allow_mixed_protocols: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, dict[str, Any]]]:
    turn_frames: list[pd.DataFrame] = []
    race_frames: list[pd.DataFrame] = []
    player_frames: list[pd.DataFrame] = []
    protocol_payloads: dict[str, dict[str, Any]] = {}

    for run_directory in run_directories:
        turns_path = run_directory / "turns.jsonl"
        races_path = run_directory / "races.csv"
        players_path = run_directory / "players.csv"
        missing = [
            path.name
            for path in (turns_path, races_path, players_path)
            if not path.is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"{run_directory} is missing required sibling files: {', '.join(missing)}"
            )

        source_run = run_directory.resolve().as_posix()
        manifest_path = run_directory / "run_manifest.json"
        manifest, run_status, manifest_run_phase = _read_run_manifest(
            run_directory,
            allow_nonfinal_runs=allow_nonfinal_runs,
        )
        (
            protocol_signature,
            protocol_payload,
            manifest_model_label,
        ) = _protocol_contract_from_manifest(
            manifest,
            path=manifest_path,
            source_run=source_run,
            allow_unverified=(
                allow_nonfinal_runs and allow_mixed_protocols
            ),
        )
        existing_payload = protocol_payloads.get(protocol_signature)
        if existing_payload is not None and existing_payload != protocol_payload:
            raise ValueError(
                "protocol signature collision while reading "
                f"{manifest_path}; refusing ambiguous provenance"
            )
        protocol_payloads[protocol_signature] = protocol_payload
        turns = _read_turns(turns_path, source_run=source_run)

        races = pd.read_csv(races_path)
        if races.empty:
            raise ValueError(f"{races_path} contains no races")
        races["source_run"] = source_run

        players = pd.read_csv(players_path)
        if players.empty:
            raise ValueError(f"{players_path} contains no player rows")
        players["source_run"] = source_run

        expected_counts = {
            "n_turns": len(turns),
            "n_races": len(races),
            "n_players": len(players),
        }
        required_count_fields = {"n_turns", "n_races"}
        manifest_counts_verified = bool(manifest) and required_count_fields <= set(
            manifest
        )
        missing_required_counts = sorted(required_count_fields - set(manifest))
        if run_status == "completed" and missing_required_counts:
            raise ValueError(
                f"{run_directory / 'run_manifest.json'} is completed but missing "
                f"required output counts: {', '.join(missing_required_counts)}"
            )
        for field, observed in expected_counts.items():
            if field not in manifest:
                continue
            try:
                expected = int(manifest[field])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"{run_directory / 'run_manifest.json'} has invalid {field}"
                ) from exc
            if expected != observed:
                manifest_counts_verified = False
                if run_status == "completed" or not allow_nonfinal_runs:
                    raise ValueError(
                        f"{run_directory / 'run_manifest.json'} reports "
                        f"{field}={expected}, but the result files contain {observed}"
                    )

        for table in (turns, races, players):
            table["run_status"] = run_status
            table["manifest_run_phase"] = manifest_run_phase
            table["manifest_prompt_version"] = manifest.get("prompt_version")
            table["manifest_persona_condition"] = manifest.get("persona_condition")
            table["manifest_base_seed"] = (manifest.get("experiment") or {}).get("seed")
            table["manifest_model_label"] = manifest_model_label
            table["protocol_signature"] = protocol_signature
            table["manifest_counts_verified"] = manifest_counts_verified
        turn_frames.append(turns)
        race_frames.append(races)
        player_frames.append(players)

    return (
        pd.concat(turn_frames, ignore_index=True, sort=False),
        pd.concat(race_frames, ignore_index=True, sort=False),
        pd.concat(player_frames, ignore_index=True, sort=False),
        protocol_payloads,
    )


def _prepare_turns(frame: pd.DataFrame) -> pd.DataFrame:
    turns = _rename_aliases(frame, TURN_ALIASES)
    _require_columns(
        turns,
        [
            *TURN_KEY,
            *BASE_CONTEXT,
            "unsafe",
            "action",
            "parse_failed",
            "progress_gap_before",
            "own_progress_before",
            "opponent_progress_before",
            "own_stage_payoff_before",
            "opponent_stage_payoff_before",
            "own_private_risk_before",
            "opponent_private_risk_before",
            "round_payoff",
            "step_increment",
            "own_progress_after",
            "opponent_progress_after",
            "progress_gap_after",
            "cumulative_stage_payoff_after",
            "current_private_risk_after",
            "stopped",
            "stop_draw",
        ],
        table="turns.jsonl",
    )

    for column in ("source_run", "game_id", "player_id", "model"):
        turns[column] = _normalise_identifier(
            turns[column],
            name=f"turns.jsonl.{column}",
        )
    if "opponent_id" in turns:
        turns["opponent_id"] = _normalise_identifier(
            turns["opponent_id"],
            name="turns.jsonl.opponent",
        )

    turns["round"] = _normalise_numeric(
        turns["round"],
        name="turns.jsonl.round",
    )
    if not np.equal(turns["round"], np.floor(turns["round"])).all():
        raise ValueError("turns.jsonl.round must contain integers")
    turns["round"] = turns["round"].astype(int)
    if turns["round"].lt(1).any():
        raise ValueError("turns.jsonl.round must start at 1")

    turns["max_private_risk"] = _normalise_numeric(
        turns["max_private_risk"],
        name="turns.jsonl.max_private_risk",
    )
    if (~turns["max_private_risk"].between(0, 1)).any():
        raise ValueError("turns.jsonl.max_private_risk must lie in [0, 1]")

    turns["unsafe"] = _normalise_binary(
        turns["unsafe"],
        name="turns.jsonl.unsafe",
        allow_missing=True,
        allow_action_words=True,
    )
    action_binary = _normalise_binary(
        turns["action"],
        name="turns.jsonl.action",
        allow_missing=True,
        allow_action_words=True,
    )
    disagreement = turns["unsafe"].notna() & action_binary.notna()
    disagreement &= turns["unsafe"].ne(action_binary)
    if disagreement.any():
        row = turns.loc[disagreement, [*TURN_KEY, "unsafe", "action"]].iloc[0]
        raise ValueError(
            "turns.jsonl action and unsafe disagree at "
            f"{tuple(row[column] for column in TURN_KEY)}"
        )

    turns["parse_failed"] = _normalise_binary(
        turns["parse_failed"],
        name="turns.jsonl.parse_failed",
    ).astype(bool)
    if (turns["unsafe"].isna() | action_binary.isna()).any():
        raise ValueError(
            "turns.jsonl must log both action and unsafe for every decision, "
            "including the enacted Safe fallback after a parse failure"
        )
    if "retry_count" in turns:
        turns["retry_count"] = _normalise_numeric(
            turns["retry_count"],
            name="turns.jsonl.retry_count",
        )
        if (turns["retry_count"] < 0).any():
            raise ValueError("turns.jsonl.retry_count must be non-negative")
        if not np.equal(
            turns["retry_count"],
            np.floor(turns["retry_count"]),
        ).all():
            raise ValueError("turns.jsonl.retry_count must contain integers")
    else:
        turns["retry_count"] = 0.0

    for logged_column in ("own_prev_action", "opponent_prev_action"):
        if logged_column in turns:
            turns[f"logged_{logged_column}_unsafe"] = _normalise_binary(
                turns[logged_column],
                name=f"turns.jsonl.{logged_column}",
                allow_missing=True,
                allow_action_words=True,
            )

    turns["progress_gap_before"] = _normalise_numeric(
        turns["progress_gap_before"],
        name="turns.jsonl.progress_gap_before",
    )
    if {"own_progress_before", "opponent_progress_before"} <= set(turns.columns):
        own = _normalise_numeric(
            turns["own_progress_before"],
            name="turns.jsonl.own_progress_before",
        )
        opponent = _normalise_numeric(
            turns["opponent_progress_before"],
            name="turns.jsonl.opponent_progress_before",
        )
        turns["own_progress_before"] = own
        turns["opponent_progress_before"] = opponent
        derived_gap = own - opponent
        comparable = turns["progress_gap_before"].notna() & derived_gap.notna()
        if not np.isclose(
            turns.loc[comparable, "progress_gap_before"],
            derived_gap.loc[comparable],
            atol=1e-9,
            rtol=0,
        ).all():
            raise ValueError(
                "turns.jsonl.progress_gap_before does not match "
                "own_progress_before - opponent_progress_before"
            )

    for numeric_column in (
        "own_stage_payoff_before",
        "opponent_stage_payoff_before",
        "own_private_risk_before",
        "opponent_private_risk_before",
        "round_payoff",
        "step_increment",
        "own_progress_after",
        "opponent_progress_after",
        "progress_gap_after",
        "cumulative_stage_payoff_after",
        "current_private_risk_after",
    ):
        if numeric_column in turns:
            turns[numeric_column] = _normalise_numeric(
                turns[numeric_column],
                name=f"turns.jsonl.{numeric_column}",
            )
    if "stopped" in turns:
        turns["stopped"] = _normalise_binary(
            turns["stopped"],
            name="turns.jsonl.stopped",
        ).astype(bool)
    if "stop_draw" in turns:
        turns["stop_draw"] = _normalise_numeric(
            turns["stop_draw"],
            name="turns.jsonl.stop_draw",
            allow_missing=True,
        )
        present_draw = turns["stop_draw"].dropna()
        if ((present_draw < 0) | (present_draw >= 1)).any():
            raise ValueError("turns.jsonl.stop_draw must lie in [0, 1)")

    if turns.duplicated(TURN_KEY).any():
        duplicate = turns.loc[turns.duplicated(TURN_KEY, keep=False), TURN_KEY].iloc[0]
        raise ValueError(
            "duplicate player-round record: "
            f"{tuple(duplicate[column] for column in TURN_KEY)}"
        )
    return _normalise_optional_rep(turns, table="turns.jsonl")


def _prepare_races(
    frame: pd.DataFrame,
    *,
    allow_noncanonical_mechanism: bool,
) -> pd.DataFrame:
    races = _rename_aliases(frame, RACE_ALIASES)
    _require_columns(
        races,
        [
            *RACE_KEY,
            *BASE_CONTEXT,
            "n_rounds",
            "stop_forced",
            "tie",
            "parse_failures",
        ],
        table="races.csv",
    )
    for column in ("source_run", "game_id", "model"):
        races[column] = _normalise_identifier(
            races[column],
            name=f"races.csv.{column}",
        )
    races["max_private_risk"] = _normalise_numeric(
        races["max_private_risk"],
        name="races.csv.max_private_risk",
    )
    if (~races["max_private_risk"].between(0, 1)).any():
        raise ValueError("races.csv.max_private_risk must lie in [0, 1]")
    canonical_risk = races["max_private_risk"].map(
        lambda value: any(
            math.isclose(float(value), expected, abs_tol=1e-12)
            for expected in (0.1, 0.6, 0.9)
        )
    )
    races["canonical_risk_treatment"] = canonical_risk.astype(bool)
    if (~canonical_risk).any() and not allow_noncanonical_mechanism:
        raise ValueError(
            "canonical analysis requires max_private_risk in {0.1, 0.6, 0.9}; "
            "pass --allow-noncanonical-mechanism only for a labelled audit; "
            "such races remain excluded from behavioral estimands"
        )
    races["n_rounds"] = _normalise_numeric(
        races["n_rounds"],
        name="races.csv.n_rounds",
    )
    if (
        (races["n_rounds"] < 1).any()
        or not np.equal(races["n_rounds"], np.floor(races["n_rounds"])).all()
    ):
        raise ValueError("races.csv.n_rounds must contain positive integers")
    races["canonical_minimum_horizon"] = races["n_rounds"].ge(5)
    if races["n_rounds"].lt(5).any() and not allow_noncanonical_mechanism:
        raise ValueError(
            "canonical AI Race horizons require n_rounds >= 5; pass "
            "--allow-noncanonical-mechanism only for an explicitly labelled "
            "audit; such races remain excluded from behavioral estimands"
        )
    if races.duplicated(RACE_KEY).any():
        raise ValueError("races.csv contains duplicate source_run/game_id rows")
    races["stop_forced"] = _normalise_binary(
        races["stop_forced"],
        name="races.csv.stop_forced",
    ).astype(bool)
    if "tie" in races:
        races["tie"] = _normalise_binary(
            races["tie"],
            name="races.csv.tie",
        ).astype(bool)
    if "parse_failures" in races:
        races["parse_failures"] = _normalise_numeric(
            races["parse_failures"],
            name="races.csv.parse_failures",
        )
        if (
            (races["parse_failures"] < 0).any()
            or not np.equal(
                races["parse_failures"],
                np.floor(races["parse_failures"]),
            ).all()
        ):
            raise ValueError("races.csv.parse_failures must contain non-negative integers")
        races["parse_failures"] = races["parse_failures"].astype(int)
    return _normalise_optional_rep(races, table="races.csv")


def _prepare_players(frame: pd.DataFrame) -> pd.DataFrame:
    players = _rename_aliases(frame, PLAYER_ALIASES)
    _require_columns(
        players,
        [
            *PLAYER_KEY,
            *BASE_CONTEXT,
            "outcome",
            "n_rounds",
            "progress",
            "stage_payoff",
            "unsafe_count",
            "unsafe_frequency",
            "private_risk",
            "prize",
            "setback_eligible",
            "setback_draw",
            "setback",
            "final_payoff",
        ],
        table="players.csv",
    )
    for column in ("source_run", "game_id", "player_id", "model"):
        players[column] = _normalise_identifier(
            players[column],
            name=f"players.csv.{column}",
        )
    players["max_private_risk"] = _normalise_numeric(
        players["max_private_risk"],
        name="players.csv.max_private_risk",
    )
    if (~players["max_private_risk"].between(0, 1)).any():
        raise ValueError("players.csv.max_private_risk must lie in [0, 1]")
    players["outcome"] = (
        players["outcome"].astype(str).str.strip().str.lower()
    )
    valid_outcomes = {"winner", "loser", "tie"}
    invalid_outcomes = sorted(set(players["outcome"]) - valid_outcomes)
    if invalid_outcomes:
        raise ValueError(
            "players.csv.outcome must be winner, loser, or tie; found "
            f"{invalid_outcomes}"
        )
    if players.duplicated(PLAYER_KEY).any():
        raise ValueError("players.csv contains duplicate source_run/game_id/player rows")
    for numeric_column in (
        "n_rounds",
        "progress",
        "stage_payoff",
        "unsafe_count",
        "unsafe_frequency",
        "private_risk",
        "prize",
        "final_payoff",
    ):
        if numeric_column in players:
            players[numeric_column] = _normalise_numeric(
                players[numeric_column],
                name=f"players.csv.{numeric_column}",
            )
    for binary_column in ("setback_eligible", "setback"):
        if binary_column in players:
            players[binary_column] = _normalise_binary(
                players[binary_column],
                name=f"players.csv.{binary_column}",
            ).astype(bool)
    if "setback_draw" in players:
        players["setback_draw"] = _normalise_numeric(
            players["setback_draw"],
            name="players.csv.setback_draw",
        )
        if (
            (players["setback_draw"] < 0)
            | (players["setback_draw"] >= 1)
        ).any():
            raise ValueError("players.csv.setback_draw must lie in [0, 1)")
    return _normalise_optional_rep(players, table="players.csv")


def _resolve_prompt_versions(
    turns: pd.DataFrame,
    races: pd.DataFrame,
    players: pd.DataFrame,
    *,
    allow_mixed_protocols: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Resolve one prompt protocol per race from any of the three logged tables."""

    cleaned_tables: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    for table_name, table in (
        ("turns.jsonl", turns),
        ("races.csv", races),
        ("players.csv", players),
    ):
        table = table.copy()
        if "prompt_version" not in table:
            table["prompt_version"] = pd.Series(
                pd.NA,
                index=table.index,
                dtype="string",
            )
        else:
            version = table["prompt_version"].astype("string").str.strip()
            table["prompt_version"] = version.mask(version.eq(""))
        if "manifest_prompt_version" not in table:
            table["manifest_prompt_version"] = pd.Series(
                pd.NA,
                index=table.index,
                dtype="string",
            )
        else:
            manifest_version = (
                table["manifest_prompt_version"].astype("string").str.strip()
            )
            table["manifest_prompt_version"] = manifest_version.mask(
                manifest_version.eq("")
            )
        cleaned_tables.append(table)
        candidates.extend(
            [
                table[[*RACE_KEY, "prompt_version"]]
                .dropna(subset=["prompt_version"])
                .assign(source_table=table_name),
                table[[*RACE_KEY, "manifest_prompt_version"]]
                .dropna(subset=["manifest_prompt_version"])
                .rename(
                    columns={
                        "manifest_prompt_version": "prompt_version",
                    }
                )
                .assign(source_table=f"{table_name}:run_manifest.json"),
            ]
        )

    candidate_versions = pd.concat(candidates, ignore_index=True)
    if candidate_versions.empty:
        version_lookup = pd.DataFrame(columns=[*RACE_KEY, "prompt_version"])
    else:
        version_counts = candidate_versions.groupby(
            RACE_KEY,
            observed=True,
        )["prompt_version"].nunique()
        conflicts = version_counts.loc[version_counts.gt(1)]
        if not conflicts.empty:
            conflict_key = conflicts.index[0]
            raise ValueError(
                "prompt_version conflicts across turns/races/players for race "
                f"{conflict_key}; mixed labels within one race cannot be overridden"
            )
        version_lookup = candidate_versions[
            [*RACE_KEY, "prompt_version"]
        ].drop_duplicates(RACE_KEY)

    resolved_lookup = races[RACE_KEY].merge(
        version_lookup,
        on=RACE_KEY,
        how="left",
        validate="one_to_one",
    )
    n_missing = int(resolved_lookup["prompt_version"].isna().sum())
    if n_missing and not allow_mixed_protocols:
        raise ValueError(
            f"prompt_version is missing for {n_missing} race(s). Refusing to pool "
            "unknown protocols; pass --allow-mixed-protocols only for an explicitly "
            "labelled sensitivity analysis."
        )
    if n_missing:
        resolved_lookup["prompt_version"] = resolved_lookup[
            "prompt_version"
        ].fillna(MISSING_PROMPT_VERSION)

    versions = sorted(
        str(value)
        for value in resolved_lookup["prompt_version"].dropna().unique().tolist()
    )
    if len(versions) > 1 and not allow_mixed_protocols:
        raise ValueError(
            "multiple prompt_version values were discovered "
            f"({versions}). Refusing to pool incompatible protocols; analyse one "
            "version at a time or pass --allow-mixed-protocols for an explicitly "
            "stratified sensitivity analysis."
        )

    resolved_tables: list[pd.DataFrame] = []
    for table in cleaned_tables:
        resolved = table.drop(
            columns=["prompt_version", "manifest_prompt_version"],
        ).merge(
            resolved_lookup,
            on=RACE_KEY,
            how="left",
            validate="many_to_one",
        )
        resolved_tables.append(resolved)
    return (*resolved_tables, versions)


def _resolve_persona_conditions(
    turns: pd.DataFrame,
    races: pd.DataFrame,
    players: pd.DataFrame,
    *,
    allow_missing_persona_condition: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Resolve one persona condition per race from the tables and the manifest.

    Unlike prompt version or run phase, multiple persona conditions in one input
    set are expected: the persona study compares them. What must never happen is
    an *unlabelled* race silently joining a labelled one, so a missing condition
    is an error rather than a sentinel fill.
    """

    cleaned_tables: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    for table in (turns, races, players):
        table = table.copy()
        for column in ("persona_condition", "manifest_persona_condition"):
            if column not in table:
                table[column] = pd.Series(pd.NA, index=table.index, dtype="string")
            else:
                value = table[column].astype("string").str.strip()
                table[column] = value.mask(value.eq(""))
        cleaned_tables.append(table)
        candidates.extend(
            [
                table[[*RACE_KEY, "persona_condition"]]
                .dropna(subset=["persona_condition"])
                .rename(columns={"persona_condition": "candidate_condition"}),
                table[[*RACE_KEY, "manifest_persona_condition"]]
                .dropna(subset=["manifest_persona_condition"])
                .rename(
                    columns={"manifest_persona_condition": "candidate_condition"}
                ),
            ]
        )

    candidate_conditions = pd.concat(candidates, ignore_index=True)
    if candidate_conditions.empty:
        condition_lookup = pd.DataFrame(columns=[*RACE_KEY, "persona_condition"])
    else:
        condition_counts = candidate_conditions.groupby(
            RACE_KEY,
            observed=True,
        )["candidate_condition"].nunique()
        conflicts = condition_counts.loc[condition_counts.gt(1)]
        if not conflicts.empty:
            raise ValueError(
                "persona_condition conflicts across rows/manifest for race "
                f"{conflicts.index[0]}; a race has exactly one persona condition"
            )
        condition_lookup = candidate_conditions.rename(
            columns={"candidate_condition": "persona_condition"}
        )[[*RACE_KEY, "persona_condition"]].drop_duplicates(RACE_KEY)

    resolved_lookup = races[RACE_KEY].merge(
        condition_lookup,
        on=RACE_KEY,
        how="left",
        validate="one_to_one",
    )
    n_missing = int(resolved_lookup["persona_condition"].isna().sum())
    if n_missing and not allow_missing_persona_condition:
        raise ValueError(
            f"persona_condition is missing for {n_missing} race(s). A run recorded "
            "before persona labelling cannot be distinguished from the neutral "
            "baseline, and persona changes behaviour without changing the prompt "
            "hash. Re-run with a labelled agents configuration, or pass "
            "--allow-missing-persona-condition for an explicitly labelled audit."
        )
    if n_missing:
        resolved_lookup["persona_condition"] = resolved_lookup[
            "persona_condition"
        ].fillna(MISSING_PERSONA_CONDITION)

    conditions = sorted(
        str(value)
        for value in resolved_lookup["persona_condition"].dropna().unique().tolist()
    )

    resolved_tables: list[pd.DataFrame] = []
    for table in cleaned_tables:
        resolved = table.drop(
            columns=["persona_condition", "manifest_persona_condition"],
        ).merge(
            resolved_lookup,
            on=RACE_KEY,
            how="left",
            validate="many_to_one",
        )
        resolved_tables.append(resolved)
    return (*resolved_tables, conditions)


def _resolve_run_phases(
    turns: pd.DataFrame,
    races: pd.DataFrame,
    players: pd.DataFrame,
    *,
    allow_nonconfirmatory_runs: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    """Resolve and enforce the pilot/confirmatory analysis phase per race."""

    if (
        "manifest_run_phase" not in races
        or races["manifest_run_phase"].isna().any()
    ) and not allow_nonconfirmatory_runs:
        raise ValueError(
            "a completed primary-analysis run_manifest.json is missing run_phase. "
            "The table-level label is not a substitute for the run-level protocol "
            "declaration; use --allow-nonconfirmatory-runs only for an explicit "
            "audit."
        )

    cleaned_tables: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    for table in (turns, races, players):
        table = table.copy()
        if "run_phase" not in table:
            table["run_phase"] = pd.Series(pd.NA, index=table.index, dtype="string")
        else:
            phase = table["run_phase"].astype("string").str.strip().str.lower()
            table["run_phase"] = phase.mask(phase.eq(""))
        if "manifest_run_phase" in table:
            manifest_phase = (
                table["manifest_run_phase"].astype("string").str.strip().str.lower()
            )
            table["manifest_run_phase"] = manifest_phase.mask(manifest_phase.eq(""))
        else:
            table["manifest_run_phase"] = pd.Series(
                pd.NA,
                index=table.index,
                dtype="string",
            )
        cleaned_tables.append(table)
        candidates.extend(
            [
                table[[*RACE_KEY, "run_phase"]]
                .dropna(subset=["run_phase"])
                .rename(columns={"run_phase": "candidate_phase"}),
                table[[*RACE_KEY, "manifest_run_phase"]]
                .dropna(subset=["manifest_run_phase"])
                .rename(columns={"manifest_run_phase": "candidate_phase"}),
            ]
        )

    candidate_phases = pd.concat(candidates, ignore_index=True)
    invalid = sorted(
        set(candidate_phases["candidate_phase"].astype(str))
        - {"pilot", "confirmatory"}
    )
    if invalid:
        raise ValueError(
            "run_phase must be 'pilot' or 'confirmatory'; found "
            f"{invalid}"
        )

    if candidate_phases.empty:
        phase_lookup = pd.DataFrame(columns=[*RACE_KEY, "run_phase"])
    else:
        phase_counts = candidate_phases.groupby(
            RACE_KEY,
            observed=True,
        )["candidate_phase"].nunique()
        conflicts = phase_counts.loc[phase_counts.gt(1)]
        if not conflicts.empty:
            raise ValueError(
                "run_phase conflicts across rows/manifest for race "
                f"{conflicts.index[0]}; phase conflicts cannot be overridden"
            )
        phase_lookup = (
            candidate_phases.rename(columns={"candidate_phase": "run_phase"})[
                [*RACE_KEY, "run_phase"]
            ].drop_duplicates(RACE_KEY)
        )

    resolved_lookup = races[RACE_KEY].merge(
        phase_lookup,
        on=RACE_KEY,
        how="left",
        validate="one_to_one",
    )
    n_missing = int(resolved_lookup["run_phase"].isna().sum())
    if n_missing and not allow_nonconfirmatory_runs:
        raise ValueError(
            f"run_phase is missing for {n_missing} race(s). Primary analysis requires "
            "an explicit confirmatory label; use --allow-nonconfirmatory-runs only "
            "for a stratified pilot/protocol audit."
        )
    if n_missing:
        resolved_lookup["run_phase"] = resolved_lookup["run_phase"].fillna(
            MISSING_RUN_PHASE
        )

    phases = sorted(
        str(value)
        for value in resolved_lookup["run_phase"].dropna().unique().tolist()
    )
    if len(phases) > 1 and not allow_nonconfirmatory_runs:
        raise ValueError(
            f"multiple run_phase values were discovered ({phases}); refusing to pool "
            "pilot and confirmatory runs"
        )
    nonconfirmatory = [phase for phase in phases if phase != "confirmatory"]
    if nonconfirmatory and not allow_nonconfirmatory_runs:
        raise ValueError(
            f"primary analysis accepts run_phase='confirmatory' only; found "
            f"{nonconfirmatory}. Use --allow-nonconfirmatory-runs only for a "
            "stratified sensitivity/protocol audit."
        )

    resolved_tables: list[pd.DataFrame] = []
    for table in cleaned_tables:
        resolved = table.drop(
            columns=["run_phase", "manifest_run_phase"],
        ).merge(
            resolved_lookup,
            on=RACE_KEY,
            how="left",
            validate="many_to_one",
        )
        resolved_tables.append(resolved)
    return (*resolved_tables, phases)


def _resolve_protocol_signatures(
    turns: pd.DataFrame,
    races: pd.DataFrame,
    players: pd.DataFrame,
    *,
    protocol_payloads: dict[str, dict[str, Any]],
    allow_mixed_protocols: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Validate one manifest contract per run and one contract per model by default."""

    resolved_tables: list[pd.DataFrame] = []
    run_contract_candidates: list[pd.DataFrame] = []
    for table_name, table in (
        ("turns.jsonl", turns),
        ("races.csv", races),
        ("players.csv", players),
    ):
        _require_columns(
            table,
            ("source_run", "model", "protocol_signature"),
            table=table_name,
        )
        table = table.copy()
        table["protocol_signature"] = _normalise_identifier(
            table["protocol_signature"],
            name=f"{table_name}.protocol_signature",
        )
        if "manifest_model_label" not in table:
            table["manifest_model_label"] = pd.Series(
                pd.NA,
                index=table.index,
                dtype="string",
            )
        else:
            label = table["manifest_model_label"].astype("string").str.strip()
            table["manifest_model_label"] = label.mask(label.eq(""))

        verified_model = table["manifest_model_label"].notna()
        model_mismatch = verified_model & table["model"].ne(
            table["manifest_model_label"]
        )
        if model_mismatch.any():
            example = table.loc[
                model_mismatch,
                ["source_run", "model", "manifest_model_label"],
            ].iloc[0]
            raise ValueError(
                "model label disagrees with run_manifest.json for "
                f"{example['source_run']}: rows use {example['model']!r}, "
                f"manifest uses {example['manifest_model_label']!r}"
            )

        run_contract_candidates.append(
            table[
                [
                    "source_run",
                    "model",
                    "protocol_signature",
                    "manifest_model_label",
                ]
            ].drop_duplicates()
        )
        resolved_tables.append(table)

    run_contracts = pd.concat(run_contract_candidates, ignore_index=True)
    signature_counts = run_contracts.groupby(
        "source_run",
        observed=True,
    )["protocol_signature"].nunique()
    conflicting_runs = signature_counts.loc[signature_counts.ne(1)]
    if not conflicting_runs.empty:
        raise ValueError(
            "a source run resolved to multiple protocol signatures: "
            f"{conflicting_runs.index[0]}"
        )

    observed_signatures = sorted(
        run_contracts["protocol_signature"].astype(str).unique()
    )
    missing_payloads = [
        signature
        for signature in observed_signatures
        if signature not in protocol_payloads
    ]
    if missing_payloads:
        raise ValueError(
            "protocol payload registry is missing observed signature "
            f"{missing_payloads[0]}"
        )
    noncanonical_prompts: dict[str, dict[str, Any]] = {}
    for signature in observed_signatures:
        payload = protocol_payloads[signature]
        prompt = payload.get("prompt")
        if not isinstance(prompt, dict) or not _is_canonical_prompt(
            prompt.get("version"),
            prompt.get("sha256"),
        ):
            noncanonical_prompts[signature] = (
                prompt
                if isinstance(prompt, dict)
                else {"verification": payload.get("verification", "missing")}
            )
    if noncanonical_prompts and not allow_mixed_protocols:
        raise ValueError(
            "primary analysis requires canonical prompt "
            f"{CANONICAL_PROMPT_VERSION!r} with one of the frozen template "
            f"SHA-256 values {sorted(CANONICAL_PROMPT_SHA256S)}; found "
            f"noncanonical protocol signatures "
            f"{noncanonical_prompts}. Pass --allow-mixed-protocols only for an "
            "explicit prompt/signature-stratified sensitivity audit."
        )

    model_counts = (
        run_contracts[["model", "protocol_signature"]]
        .drop_duplicates()
        .groupby("model", observed=True)["protocol_signature"]
        .nunique()
    )
    incompatible_models = model_counts.loc[model_counts.gt(1)]
    if not incompatible_models.empty and not allow_mixed_protocols:
        details = {
            str(model): int(count)
            for model, count in incompatible_models.items()
        }
        raise ValueError(
            "multiple manifest protocol signatures were found within the same "
            f"model label ({details}). This means prompt hash, exact source/model "
            "revision, decoding, seed handling, mechanism, or runtime provenance "
            "changed. Analyse contracts separately or pass --allow-mixed-protocols "
            "for an explicitly signature-stratified sensitivity analysis."
        )

    return resolved_tables[0], resolved_tables[1], resolved_tables[2]


def _shared_base_seed(races: pd.DataFrame) -> bool:
    """Report whether every run drew its horizons from the same base seed.

    ``game_seed = base_seed + rep`` is independent of both the risk treatment and
    the agents configuration, so when the base seed is shared a repetition index
    identifies one horizon/setback draw across every run directory — including
    persona cells, which live in separate directories and therefore have separate
    ``source_run`` values. Widening the cluster in that case keeps one random draw
    in one block instead of splitting it. The base seed is deliberately not part
    of the protocol signature (independent replication batches may reseed), so it
    is read from the manifest here rather than from the run contract.
    """

    if "manifest_base_seed" not in races:
        return False
    seeds = races["manifest_base_seed"]
    if seeds.empty or seeds.isna().any():
        # One unlabelled run is enough to make a shared draw unverifiable.
        return False
    return int(seeds.nunique()) == 1


def _resolve_repetition_blocks(
    turns: pd.DataFrame,
    races: pd.DataFrame,
    players: pd.DataFrame,
    *,
    share_blocks_across_runs: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Derive the CRN repetition index and block ID from any logged table."""

    tables: list[pd.DataFrame] = []
    candidates: list[pd.DataFrame] = []
    for table in (turns, races, players):
        table = table.copy()
        if "rep" not in table:
            table["rep"] = pd.Series(pd.NA, index=table.index, dtype="Int64")
        tables.append(table)
        candidates.append(table[[*RACE_KEY, "rep"]].dropna(subset=["rep"]))

    candidate_reps = pd.concat(candidates, ignore_index=True)
    if candidate_reps.empty:
        rep_lookup = races[RACE_KEY].copy()
        rep_lookup["rep"] = pd.Series(pd.NA, index=rep_lookup.index, dtype="Int64")
    else:
        rep_counts = candidate_reps.groupby(RACE_KEY, observed=True)["rep"].nunique()
        conflicts = rep_counts.loc[rep_counts.gt(1)]
        if not conflicts.empty:
            raise ValueError(
                "rep conflicts across turns/races/players for race "
                f"{conflicts.index[0]}"
            )
        rep_lookup = (
            races[RACE_KEY]
            .merge(
                candidate_reps.drop_duplicates(RACE_KEY),
                on=RACE_KEY,
                how="left",
                validate="one_to_one",
            )
        )
        rep_lookup["rep"] = rep_lookup["rep"].astype("Int64")

    resolved_tables: list[pd.DataFrame] = []
    for table in tables:
        resolved = table.drop(columns=["rep"]).merge(
            rep_lookup,
            on=RACE_KEY,
            how="left",
            validate="many_to_one",
        )
        resolved["randomization_block_id"] = pd.Series(
            pd.NA,
            index=resolved.index,
            dtype="string",
        )
        has_rep = resolved["rep"].notna()
        prefix = (
            ""
            if share_blocks_across_runs
            else resolved.loc[has_rep, "source_run"] + "::"
        )
        resolved.loc[has_rep, "randomization_block_id"] = (
            prefix
            + resolved.loc[has_rep, "model"]
            + "::rep"
            + resolved.loc[has_rep, "rep"].astype(str)
        )
        resolved_tables.append(resolved)
    return tuple(resolved_tables)


def _validate_join_keys(
    turns: pd.DataFrame,
    races: pd.DataFrame,
    players: pd.DataFrame,
) -> None:
    race_lookup = races[RACE_KEY].drop_duplicates().assign(_in_races=True)
    missing_races = (
        turns[RACE_KEY]
        .drop_duplicates()
        .merge(race_lookup, on=RACE_KEY, how="left")
        .loc[lambda data: data["_in_races"].isna(), RACE_KEY]
    )
    if not missing_races.empty:
        raise ValueError(
            "turns.jsonl contains a game absent from races.csv: "
            f"{tuple(missing_races.iloc[0])}"
        )

    player_lookup = players[PLAYER_KEY].drop_duplicates().assign(_in_players=True)
    missing_players = (
        turns[PLAYER_KEY]
        .drop_duplicates()
        .merge(player_lookup, on=PLAYER_KEY, how="left")
        .loc[lambda data: data["_in_players"].isna(), PLAYER_KEY]
    )
    if not missing_players.empty:
        raise ValueError(
            "turns.jsonl contains a player absent from players.csv: "
            f"{tuple(missing_players.iloc[0])}"
        )

    for table_name, table, key in (
        ("races.csv", races, RACE_KEY),
        ("players.csv", players, PLAYER_KEY),
    ):
        context = table[[*key, *CONTEXT]]
        joined = turns[[*key, *CONTEXT]].merge(
            context,
            on=key,
            how="left",
            suffixes=("_turn", "_table"),
        )
        model_mismatch = joined["model_turn"].ne(joined["model_table"])
        prompt_mismatch = joined["prompt_version_turn"].ne(
            joined["prompt_version_table"]
        )
        risk_mismatch = ~np.isclose(
            joined["max_private_risk_turn"],
            joined["max_private_risk_table"],
            atol=1e-12,
            rtol=0,
        )
        if model_mismatch.any() or prompt_mismatch.any() or risk_mismatch.any():
            raise ValueError(
                "model/max_private_risk/prompt_version disagree between "
                f"turns.jsonl and {table_name}"
            )

    roster_sizes = players.groupby(RACE_KEY, observed=True)["player_id"].nunique()
    if not roster_sizes.eq(2).all():
        bad = roster_sizes.loc[~roster_sizes.eq(2)]
        raise ValueError(
            "AI Race analysis requires exactly two players per game; "
            f"first invalid game {bad.index[0]} has {int(bad.iloc[0])}"
        )

    decision_counts = turns.groupby([*RACE_KEY, "round"], observed=True).size()
    if not decision_counts.eq(2).all():
        bad = decision_counts.loc[~decision_counts.eq(2)]
        raise ValueError(
            "every game-round must contain two simultaneous decisions; "
            f"first invalid game-round {bad.index[0]} has {int(bad.iloc[0])}"
        )

    observed_horizons = (
        turns.groupby(PLAYER_KEY, observed=True)["round"].agg(["min", "max", "nunique"])
    )
    invalid_sequences = (
        observed_horizons["min"].ne(1)
        | observed_horizons["nunique"].ne(observed_horizons["max"])
    )
    if invalid_sequences.any():
        raise ValueError(
            "each player trajectory must contain consecutive rounds starting at 1"
        )

    race_horizons = turns.groupby(RACE_KEY, observed=True)["round"].max().rename(
        "observed_n_rounds"
    )
    horizon_check = races.set_index(RACE_KEY)[["n_rounds"]].join(race_horizons)
    if not np.isclose(
        horizon_check["n_rounds"],
        horizon_check["observed_n_rounds"],
        atol=0,
        rtol=0,
    ).all():
        raise ValueError("races.csv.n_rounds disagrees with turns.jsonl")


def _add_race_quality(
    turns: pd.DataFrame,
    races: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Derive race-level parse and canonical-mechanism inclusion flags."""

    derived = (
        turns.groupby(RACE_KEY, observed=True)["parse_failed"]
        .agg(derived_parse_failures="sum")
        .reset_index()
    )
    derived["derived_parse_failures"] = derived["derived_parse_failures"].astype(int)
    race_quality = races.merge(
        derived,
        on=RACE_KEY,
        how="left",
        validate="one_to_one",
    )
    if race_quality["derived_parse_failures"].isna().any():
        raise ValueError("races.csv contains a race with no turns.jsonl decisions")
    if "parse_failures" in race_quality:
        mismatch = race_quality["parse_failures"].ne(
            race_quality["derived_parse_failures"]
        )
        if mismatch.any():
            bad = race_quality.loc[mismatch, RACE_KEY].iloc[0]
            raise ValueError(
                "races.csv.parse_failures disagrees with turns.jsonl for "
                f"{tuple(bad)}"
            )
    race_quality["parse_failures"] = race_quality[
        "derived_parse_failures"
    ].astype(int)
    race_quality = race_quality.drop(columns=["derived_parse_failures"])
    race_quality["any_parse_failure"] = race_quality["parse_failures"].gt(0)
    race_quality["parse_clean"] = ~race_quality["any_parse_failure"]
    race_quality["canonical_mechanism"] = (
        race_quality["canonical_risk_treatment"]
        & race_quality["canonical_minimum_horizon"]
        & ~race_quality["stop_forced"]
    )
    race_quality["included_in_behavioral_estimands"] = (
        race_quality["parse_clean"] & race_quality["canonical_mechanism"]
    )

    def exclusion_reason(row: pd.Series) -> str:
        reasons: list[str] = []
        if bool(row["any_parse_failure"]):
            reasons.append("any_parse_failure")
        if not bool(row["canonical_risk_treatment"]):
            reasons.append("noncanonical_risk_treatment")
        if not bool(row["canonical_minimum_horizon"]):
            reasons.append("below_five_round_minimum")
        if bool(row["stop_forced"]):
            reasons.append("forced_safety_cap_stop")
        return "|".join(reasons) if reasons else "included"

    race_quality["behavioral_exclusion_reason"] = race_quality.apply(
        exclusion_reason,
        axis=1,
    )

    turns = turns.merge(
        race_quality[
            [
                *RACE_KEY,
                "parse_failures",
                "any_parse_failure",
                "canonical_mechanism",
                "included_in_behavioral_estimands",
                "behavioral_exclusion_reason",
            ]
        ],
        on=RACE_KEY,
        how="left",
        validate="many_to_one",
    )
    return turns, race_quality


def _add_dynamic_columns(turns: pd.DataFrame) -> pd.DataFrame:
    """Add strict valid-action lags and the opponent's same-round action."""

    turns = turns.sort_values(TURN_KEY).reset_index(drop=True).copy()
    turns["valid_unsafe"] = turns["unsafe"].where(~turns["parse_failed"])

    opponent = turns[
        [
            *RACE_KEY,
            "round",
            "player_id",
            "unsafe",
            "valid_unsafe",
            "parse_failed",
        ]
    ].rename(
        columns={
            "player_id": "opponent_player_id_derived",
            "unsafe": "opponent_current_unsafe_enacted",
            "valid_unsafe": "opponent_current_unsafe",
            "parse_failed": "opponent_current_parse_failed",
        }
    )
    paired = turns.merge(opponent, on=[*RACE_KEY, "round"], how="left")
    paired = paired.loc[
        paired["player_id"].ne(paired["opponent_player_id_derived"])
    ].copy()
    if len(paired) != len(turns):
        raise ValueError("could not derive exactly one opponent decision per player-round")
    if "opponent_id" in paired:
        mismatch = paired["opponent_id"].ne(paired["opponent_player_id_derived"])
        if mismatch.any():
            raise ValueError("turns.jsonl.opponent does not match the paired player")

    paired = paired.sort_values(TURN_KEY).reset_index(drop=True)
    grouped = paired.groupby(PLAYER_KEY, sort=False, observed=True)
    paired["own_prev_unsafe_enacted"] = grouped["unsafe"].shift(1)
    paired["opponent_prev_unsafe_enacted"] = grouped[
        "opponent_current_unsafe_enacted"
    ].shift(1)
    paired["own_prev_unsafe"] = grouped["valid_unsafe"].shift(1)
    paired["opponent_prev_unsafe"] = grouped["opponent_current_unsafe"].shift(1)
    paired["first_round_unsafe"] = grouped["valid_unsafe"].transform(
        lambda values: values.iloc[0]
    )

    for logged_column, derived_column in (
        ("logged_own_prev_action_unsafe", "own_prev_unsafe_enacted"),
        ("logged_opponent_prev_action_unsafe", "opponent_prev_unsafe_enacted"),
    ):
        if logged_column not in paired:
            continue
        comparable = (
            paired["round"].ge(2)
            & paired[logged_column].notna()
            & paired[derived_column].notna()
        )
        if paired.loc[comparable, logged_column].ne(
            paired.loc[comparable, derived_column]
        ).any():
            raise ValueError(
                f"turns.jsonl.{logged_column.removeprefix('logged_').removesuffix('_unsafe')} "
                "disagrees with the previous-round trajectory"
            )

    # Unsafe counts entering the decision, and their difference. In this mechanism
    # progress = round + 0.5 * unsafe_count for both seats, so the race gap is
    # exactly half this difference. Recording it makes that identity checkable and
    # lets the collinearity between "being behind" and "having been safer than the
    # opponent" be reported rather than hidden inside the gap coefficient.
    paired["own_unsafe_count_before"] = (
        grouped["valid_unsafe"].cumsum().sub(paired["valid_unsafe"]).astype("Float64")
    )
    paired["opponent_unsafe_count_before"] = (
        grouped["opponent_current_unsafe"]
        .cumsum()
        .sub(paired["opponent_current_unsafe"])
        .astype("Float64")
    )
    paired["unsafe_count_diff_before"] = (
        paired["own_unsafe_count_before"] - paired["opponent_unsafe_count_before"]
    )

    gap = paired["progress_gap_before"]
    paired["race_state"] = np.select(
        [gap < -1e-9, gap > 1e-9, gap.notna()],
        ["behind", "ahead", "tied"],
        default=None,
    )
    # The gap moves in steps of 0.5 and rarely leaves [-2, 2] over a nine-round
    # race, so a magnitude bin separates "one round behind" from "well behind"
    # without inventing empty cells.
    paired["gap_bin"] = np.select(
        [
            gap <= -1.0 + 1e-9,
            gap < -1e-9,
            gap.abs() <= 1e-9,
            gap < 1.0 - 1e-9,
            gap.notna(),
        ],
        ["<=-1.0", "-0.5", "0.0", "+0.5", ">=+1.0"],
        default=None,
    )
    # Named seat_index rather than seat: "seat" is a documented input alias for
    # player_id, and reusing it here would be ambiguous in the emitted tables.
    paired["seat_index"] = (
        _normalise_numeric(paired["player_index"], name="turns.jsonl.player_index")
        if "player_index" in paired
        else pd.Series(pd.NA, index=paired.index, dtype="Float64")
    )
    return paired


def _validate_mechanics_and_terminal_state(
    turns: pd.DataFrame,
    races: pd.DataFrame,
    players: pd.DataFrame,
) -> list[str]:
    """Fail on logged mechanism/terminal contradictions; return checks performed."""

    checks: list[str] = []

    def require_close(
        observed: pd.Series,
        expected: pd.Series,
        *,
        label: str,
    ) -> None:
        if not np.isclose(
            observed.astype(float),
            expected.astype(float),
            atol=1e-9,
            rtol=0,
        ).all():
            raise ValueError(f"mechanism validation failed: {label}")
        checks.append(label)

    if "step_increment" in turns:
        require_close(
            turns["step_increment"],
            1.0 + 0.5 * turns["unsafe"].astype(float),
            label="step_increment matches Safe=1.0/Unsafe=1.5",
        )

    if "round_payoff" in turns:
        own = turns["unsafe"].astype(int)
        opponent = turns["opponent_current_unsafe_enacted"].astype(int)
        expected_payoff = pd.Series(
            np.select(
                [
                    own.eq(0) & opponent.eq(0),
                    own.eq(0) & opponent.eq(1),
                    own.eq(1) & opponent.eq(0),
                    own.eq(1) & opponent.eq(1),
                ],
                [1.0, 0.6, 2.4, 2.0],
                default=np.nan,
            ),
            index=turns.index,
        )
        require_close(
            turns["round_payoff"],
            expected_payoff,
            label="round_payoff matches the canonical 2x2 stage game",
        )

    if {"own_progress_before", "own_progress_after", "step_increment"} <= set(
        turns.columns
    ):
        require_close(
            turns["own_progress_after"],
            turns["own_progress_before"] + turns["step_increment"],
            label="own_progress_after matches pre-state plus step increment",
        )

        ordered_progress = turns.sort_values(TURN_KEY).copy()
        first_round = ordered_progress["round"].eq(1)
        require_close(
            ordered_progress.loc[first_round, "own_progress_before"],
            pd.Series(
                0.0,
                index=ordered_progress.index[first_round],
            ),
            label="each trajectory starts from zero progress",
        )
        prior_progress = ordered_progress.groupby(
            PLAYER_KEY,
            sort=False,
            observed=True,
        )["own_progress_after"].shift(1)
        later_round = ordered_progress["round"].ge(2)
        require_close(
            ordered_progress.loc[later_round, "own_progress_before"],
            prior_progress.loc[later_round],
            label="progress state is continuous across rounds",
        )

    if {
        "own_progress_after",
        "opponent_progress_after",
        "progress_gap_after",
    } <= set(turns.columns):
        require_close(
            turns["progress_gap_after"],
            turns["own_progress_after"] - turns["opponent_progress_after"],
            label="progress_gap_after matches focal minus opponent progress",
        )

    if {
        "own_progress_before",
        "own_progress_after",
        "opponent_progress_before",
        "opponent_progress_after",
    } <= set(turns.columns):
        opponent_progress = turns[
            [
                *RACE_KEY,
                "round",
                "player_id",
                "own_progress_before",
                "own_progress_after",
            ]
        ].rename(
            columns={
                "player_id": "candidate_opponent_id",
                "own_progress_before": "derived_opponent_progress_before",
                "own_progress_after": "derived_opponent_progress_after",
            }
        )
        paired_progress = turns.merge(
            opponent_progress,
            on=[*RACE_KEY, "round"],
            how="left",
        )
        paired_progress = paired_progress.loc[
            paired_progress["player_id"].ne(
                paired_progress["candidate_opponent_id"]
            )
        ]
        if len(paired_progress) != len(turns):
            raise ValueError(
                "mechanism validation failed: could not pair opponent progress"
            )
        require_close(
            paired_progress["opponent_progress_before"],
            paired_progress["derived_opponent_progress_before"],
            label="opponent_progress_before matches the paired trajectory",
        )
        require_close(
            paired_progress["opponent_progress_after"],
            paired_progress["derived_opponent_progress_after"],
            label="opponent_progress_after matches the paired trajectory",
        )

    trajectory_order = turns.sort_values(TURN_KEY).copy()
    if {
        "own_stage_payoff_before",
        "round_payoff",
        "cumulative_stage_payoff_after",
    } <= set(turns.columns):
        derived_cumulative_payoff = trajectory_order.groupby(
            PLAYER_KEY,
            sort=False,
            observed=True,
        )["round_payoff"].cumsum()
        derived_payoff_before = (
            derived_cumulative_payoff - trajectory_order["round_payoff"]
        )
        require_close(
            trajectory_order["own_stage_payoff_before"],
            derived_payoff_before,
            label=(
                "own accumulated stage payoff before the decision matches prior "
                "round payoffs"
            ),
        )
        require_close(
            trajectory_order["cumulative_stage_payoff_after"],
            derived_cumulative_payoff,
            label="cumulative stage payoff matches the round-payoff trajectory",
        )
        require_close(
            trajectory_order["cumulative_stage_payoff_after"],
            trajectory_order["own_stage_payoff_before"]
            + trajectory_order["round_payoff"],
            label="accumulated stage payoff advances by the current round payoff",
        )
    if {
        "own_private_risk_before",
        "current_private_risk_after",
    } <= set(turns.columns):
        cumulative_unsafe = trajectory_order.groupby(
            PLAYER_KEY,
            sort=False,
            observed=True,
        )["unsafe"].cumsum()
        cumulative_unsafe_before = cumulative_unsafe - trajectory_order["unsafe"]
        completed_before = trajectory_order["round"] - 1
        expected_risk_before = (
            trajectory_order["max_private_risk"]
            * cumulative_unsafe_before
            / completed_before.where(completed_before.gt(0), 1)
        )
        expected_risk_before = expected_risk_before.where(
            completed_before.gt(0),
            0.0,
        )
        require_close(
            trajectory_order["own_private_risk_before"],
            expected_risk_before,
            label=(
                "own private risk before the decision matches prior Unsafe "
                "frequency"
            ),
        )
        require_close(
            trajectory_order["current_private_risk_after"],
            trajectory_order["max_private_risk"]
            * cumulative_unsafe
            / trajectory_order["round"],
            label=(
                "current private risk matches max_private_risk times cumulative "
                "Unsafe frequency"
            ),
        )

    if {
        "own_stage_payoff_before",
        "opponent_stage_payoff_before",
        "own_private_risk_before",
        "opponent_private_risk_before",
    } <= set(turns.columns):
        opponent_observed_state = turns[
            [
                *RACE_KEY,
                "round",
                "player_id",
                "own_stage_payoff_before",
                "own_private_risk_before",
            ]
        ].rename(
            columns={
                "player_id": "candidate_state_opponent_id",
                "own_stage_payoff_before": "derived_opponent_stage_payoff_before",
                "own_private_risk_before": "derived_opponent_private_risk_before",
            }
        )
        paired_state = turns.merge(
            opponent_observed_state,
            on=[*RACE_KEY, "round"],
            how="left",
        )
        paired_state = paired_state.loc[
            paired_state["player_id"].ne(
                paired_state["candidate_state_opponent_id"]
            )
        ]
        if len(paired_state) != len(turns):
            raise ValueError(
                "mechanism validation failed: could not pair opponent payoff/risk "
                "state"
            )
        require_close(
            paired_state["opponent_stage_payoff_before"],
            paired_state["derived_opponent_stage_payoff_before"],
            label=(
                "opponent accumulated stage payoff before the decision matches "
                "the paired trajectory"
            ),
        )
        require_close(
            paired_state["opponent_private_risk_before"],
            paired_state["derived_opponent_private_risk_before"],
            label=(
                "opponent private risk before the decision matches the paired "
                "trajectory"
            ),
        )

    if "stopped" in turns:
        final_round = turns.groupby(RACE_KEY, observed=True)["round"].transform("max")
        expected_stopped = turns["round"].eq(final_round)
        if not turns["stopped"].eq(expected_stopped).all():
            raise ValueError(
                "mechanism validation failed: stopped must be true only for both "
                "decisions in the terminal round"
            )
        checks.append("stopped marks exactly the terminal game-round")

    if "stop_draw" in turns:
        stop_rounds = (
            turns.groupby([*RACE_KEY, "round"], observed=True)
            .agg(
                n_decisions=("stop_draw", "size"),
                n_stop_draws=("stop_draw", "count"),
                minimum_stop_draw=("stop_draw", "min"),
                maximum_stop_draw=("stop_draw", "max"),
            )
            .reset_index()
            .merge(
                races[
                    [
                        *RACE_KEY,
                        "n_rounds",
                        "max_private_risk",
                        "stop_forced",
                        "randomization_block_id",
                    ]
                ],
                on=RACE_KEY,
                how="left",
                validate="many_to_one",
            )
        )
        partial_draw = stop_rounds["n_stop_draws"].ne(0) & stop_rounds[
            "n_stop_draws"
        ].ne(stop_rounds["n_decisions"])
        if partial_draw.any():
            raise ValueError(
                "mechanism validation failed: both players must log the same "
                "race-level stopping draw for a game-round"
            )
        present_draw = stop_rounds["n_stop_draws"].gt(0)
        if not np.isclose(
            stop_rounds.loc[present_draw, "minimum_stop_draw"],
            stop_rounds.loc[present_draw, "maximum_stop_draw"],
            atol=0,
            rtol=0,
        ).all():
            raise ValueError(
                "mechanism validation failed: player rows disagree on stop_draw"
            )

        before_minimum = stop_rounds["round"].lt(5)
        eligible_round = ~before_minimum
        if stop_rounds.loc[before_minimum, "n_stop_draws"].ne(0).any():
            raise ValueError(
                "mechanism validation failed: stop_draw must be absent before "
                "the canonical fifth round"
            )
        if stop_rounds.loc[eligible_round, "n_stop_draws"].ne(
            stop_rounds.loc[eligible_round, "n_decisions"]
        ).any():
            raise ValueError(
                "mechanism validation failed: every round from round 5 onward "
                "must log one shared stopping draw"
            )

        terminal = stop_rounds["round"].eq(stop_rounds["n_rounds"])
        nonterminal_eligible = eligible_round & ~terminal
        if stop_rounds.loc[
            nonterminal_eligible,
            "minimum_stop_draw",
        ].lt(0.2).any():
            raise ValueError(
                "mechanism validation failed: a nonterminal post-round-5 draw "
                "is below the canonical 0.2 stopping threshold"
            )
        stochastic_terminal = terminal & eligible_round & ~stop_rounds["stop_forced"]
        forced_terminal = terminal & eligible_round & stop_rounds["stop_forced"]
        if stop_rounds.loc[
            stochastic_terminal,
            "minimum_stop_draw",
        ].ge(0.2).any():
            raise ValueError(
                "mechanism validation failed: an unforced terminal stopping draw "
                "must be below 0.2"
            )
        if stop_rounds.loc[
            forced_terminal,
            "minimum_stop_draw",
        ].lt(0.2).any():
            raise ValueError(
                "mechanism validation failed: stop_forced contradicts a terminal "
                "draw already below 0.2"
            )
        checks.append(
            "stop_draw follows the shared minimum-5-round, p=0.2 stopping rule"
        )

        block_stop_rows = stop_rounds.loc[
            stop_rounds["randomization_block_id"].notna()
        ]
        block_risk_counts = block_stop_rows.groupby(
            "randomization_block_id",
            observed=True,
        )["max_private_risk"].nunique()
        matched_blocks = block_risk_counts.loc[block_risk_counts.ge(2)].index
        block_stop_rows = block_stop_rows.loc[
            block_stop_rows["randomization_block_id"].isin(matched_blocks)
        ]
        if not block_stop_rows.empty:
            block_horizon_counts = block_stop_rows.groupby(
                "randomization_block_id",
                observed=True,
            )["n_rounds"].nunique()
            if block_horizon_counts.gt(1).any():
                raise ValueError(
                    "CRN validation failed: matched risk-treatment races in a "
                    "source_run/model/rep block have different horizons"
                )
            post_minimum_draws = block_stop_rows.loc[
                block_stop_rows["round"].ge(5)
            ]
            block_draw_ranges = post_minimum_draws.groupby(
                ["randomization_block_id", "round"],
                observed=True,
            ).agg(
                minimum_block_draw=("minimum_stop_draw", "min"),
                maximum_block_draw=("maximum_stop_draw", "max"),
            )
            if not np.isclose(
                block_draw_ranges["minimum_block_draw"],
                block_draw_ranges["maximum_block_draw"],
                atol=0,
                rtol=0,
            ).all():
                raise ValueError(
                    "CRN validation failed: matched risk-treatment races do not "
                    "reuse the same stopping-draw stream"
                )
            checks.append(
                "horizon and stopping draws are fixed within each CRN repetition block"
            )

    derived = (
        turns.groupby(PLAYER_KEY, observed=True)
        .agg(
            derived_n_rounds=("round", "size"),
            derived_unsafe_count=("unsafe", "sum"),
        )
        .reset_index()
    )
    if "round_payoff" in turns:
        stage = (
            turns.groupby(PLAYER_KEY, observed=True)["round_payoff"]
            .sum()
            .rename("derived_stage_payoff")
            .reset_index()
        )
        derived = derived.merge(stage, on=PLAYER_KEY, validate="one_to_one")
    if "own_progress_after" in turns:
        terminal = (
            turns.sort_values(TURN_KEY)
            .groupby(PLAYER_KEY, observed=True)
            .tail(1)[[*PLAYER_KEY, "own_progress_after"]]
            .rename(columns={"own_progress_after": "derived_progress"})
        )
        derived = derived.merge(terminal, on=PLAYER_KEY, validate="one_to_one")

    terminal_check = players.merge(
        derived,
        on=PLAYER_KEY,
        how="left",
        validate="one_to_one",
    )
    if terminal_check["derived_n_rounds"].isna().any():
        raise ValueError("terminal validation failed: players.csv row has no trajectory")
    if "n_rounds" in terminal_check:
        require_close(
            terminal_check["n_rounds"],
            terminal_check["derived_n_rounds"],
            label="players.csv n_rounds matches trajectory length",
        )
    if "unsafe_count" in terminal_check:
        require_close(
            terminal_check["unsafe_count"],
            terminal_check["derived_unsafe_count"],
            label="players.csv unsafe_count matches enacted trajectory",
        )
    if "unsafe_frequency" in terminal_check:
        require_close(
            terminal_check["unsafe_frequency"],
            terminal_check["derived_unsafe_count"]
            / terminal_check["derived_n_rounds"],
            label="players.csv unsafe_frequency matches enacted trajectory",
        )
    if {"progress", "derived_progress"} <= set(terminal_check.columns):
        require_close(
            terminal_check["progress"],
            terminal_check["derived_progress"],
            label="players.csv progress matches terminal turn state",
        )
    if {"stage_payoff", "derived_stage_payoff"} <= set(terminal_check.columns):
        require_close(
            terminal_check["stage_payoff"],
            terminal_check["derived_stage_payoff"],
            label="players.csv stage_payoff matches summed round payoffs",
        )

    require_close(
        terminal_check["private_risk"],
        terminal_check["max_private_risk"]
        * terminal_check["derived_unsafe_count"]
        / terminal_check["derived_n_rounds"],
        label=(
            "players.csv private_risk matches max_private_risk times realised "
            "Unsafe frequency"
        ),
    )

    expected_prize = pd.Series(
        np.select(
            [
                terminal_check["outcome"].eq("winner"),
                terminal_check["outcome"].eq("tie"),
                terminal_check["outcome"].eq("loser"),
            ],
            [100.0, 50.0, 0.0],
            default=np.nan,
        ),
        index=terminal_check.index,
    )
    require_close(
        terminal_check["prize"],
        expected_prize,
        label="players.csv prize matches canonical winner=100/tie=50/loser=0",
    )

    expected_eligible = terminal_check["outcome"].isin(["winner", "tie"])
    if not terminal_check["setback_eligible"].eq(expected_eligible).all():
        raise ValueError(
            "terminal validation failed: setback eligibility must be restricted "
            "to winners and tied winners"
        )
    checks.append("setback eligibility matches winner/tied-winner status")

    expected_setback = expected_eligible & terminal_check["setback_draw"].lt(
        terminal_check["private_risk"]
    )
    if not terminal_check["setback"].eq(expected_setback).all():
        raise ValueError(
            "terminal validation failed: setback flag contradicts eligibility, "
            "fixed-seat draw, or realised private risk"
        )
    checks.append("setback flags match eligibility, draw, and private risk")

    expected_final_payoff = (
        terminal_check["stage_payoff"] + terminal_check["prize"]
    ).where(~terminal_check["setback"], 0.0)
    require_close(
        terminal_check["final_payoff"],
        expected_final_payoff,
        label=(
            "players.csv final_payoff is zero after setback and otherwise "
            "stage_payoff plus prize"
        ),
    )

    if (
        "randomization_block_id" in terminal_check
        and terminal_check["randomization_block_id"].notna().any()
    ):
        block_rows = terminal_check.loc[
            terminal_check["randomization_block_id"].notna()
        ]
        block_risk_counts = block_rows.groupby(
            "randomization_block_id",
            observed=True,
        )["max_private_risk"].nunique()
        matched_blocks = block_risk_counts.loc[block_risk_counts.ge(2)].index
        block_rows = block_rows.loc[
            block_rows["randomization_block_id"].isin(matched_blocks)
        ]
        if not block_rows.empty:
            seat_draw_counts = block_rows.groupby(
                ["randomization_block_id", "player_id"],
                observed=True,
            )["setback_draw"].nunique()
            if seat_draw_counts.gt(1).any():
                raise ValueError(
                    "terminal validation failed: a source_run/model/rep/player seat "
                    "has different setback draws across matched risk treatments"
                )
            checks.append(
                "setback draws are fixed by player seat within each CRN "
                "repetition block"
            )

    if "progress" in players:
        maximum = players.groupby(RACE_KEY, observed=True)["progress"].transform("max")
        tied = players.groupby(RACE_KEY, observed=True)["progress"].transform(
            lambda values: int(np.isclose(values, values.max(), atol=1e-9).sum()) > 1
        )
        expected_outcome = np.where(
            tied,
            "tie",
            np.where(np.isclose(players["progress"], maximum), "winner", "loser"),
        )
        if not players["outcome"].eq(expected_outcome).all():
            raise ValueError(
                "terminal validation failed: players.csv outcome contradicts progress"
            )
        checks.append("players.csv outcomes match terminal progress")

    if "tie" in races:
        derived_tie = (
            players.groupby(RACE_KEY, observed=True)["outcome"]
            .apply(lambda values: bool(values.eq("tie").all()))
            .rename("derived_tie")
            .reset_index()
        )
        tie_check = races.merge(
            derived_tie,
            on=RACE_KEY,
            how="left",
            validate="one_to_one",
        )
        if not tie_check["tie"].eq(tie_check["derived_tie"]).all():
            raise ValueError(
                "terminal validation failed: races.csv tie contradicts players.csv"
            )
        checks.append("races.csv tie matches player outcomes")

    return checks


def _binary_summary(
    frame: pd.DataFrame,
    *,
    groups: Sequence[str],
    value: str,
    count_name: str = "n_decisions",
) -> pd.DataFrame:
    data = frame.loc[frame[value].notna()].copy()
    summary = (
        data.groupby(list(groups), dropna=False, observed=True)[value]
        .agg(**{count_name: "count", "unsafe_count": "sum", "unsafe_rate": "mean"})
        .reset_index()
    )
    summary["unsafe_count"] = summary["unsafe_count"].astype(int)
    return summary.sort_values(list(groups)).reset_index(drop=True)


def _mean_summary(
    frame: pd.DataFrame,
    *,
    groups: Sequence[str],
    value: str,
    prefix: str,
) -> pd.DataFrame:
    data = frame.loc[frame[value].notna()].copy()
    summary = (
        data.groupby(list(groups), dropna=False, observed=True)[value]
        .agg(n_players="count", mean="mean", sd="std")
        .reset_index()
    )
    return summary.rename(
        columns={
            "mean": f"mean_{prefix}",
            "sd": f"descriptive_sd_{prefix}",
        }
    ).sort_values(list(groups)).reset_index(drop=True)


def _build_player_metrics(
    turns: pd.DataFrame,
    players: pd.DataFrame,
    *,
    include_exploratory_behind: bool,
) -> pd.DataFrame:
    group_columns = [*PLAYER_KEY, *CONTEXT]
    grouped = turns.groupby(group_columns, sort=True, observed=True)
    metrics = grouped.agg(
        n_rounds=("round", "size"),
        n_valid_actions=("valid_unsafe", "count"),
        unsafe_count=("valid_unsafe", "sum"),
        unsafe_rate=("valid_unsafe", "mean"),
        parse_failures=("parse_failed", "sum"),
        parse_failure_rate=("parse_failed", "mean"),
        retry_count=("retry_count", "sum"),
        first_round_unsafe=("first_round_unsafe", "first"),
    ).reset_index()
    metrics["unsafe_count"] = metrics["unsafe_count"].fillna(0).astype(int)
    metrics["parse_failures"] = metrics["parse_failures"].astype(int)

    later = turns.loc[turns["round"].ge(2)]
    later_metrics = (
        later.groupby(group_columns, observed=True)["valid_unsafe"]
        .agg(n_later_valid_actions="count", later_unsafe_rate="mean")
        .reset_index()
    )
    metrics = metrics.merge(later_metrics, on=group_columns, how="left")

    response = turns.loc[
        turns["round"].ge(2)
        & turns["valid_unsafe"].notna()
        & turns["opponent_prev_unsafe"].notna()
    ]
    response_grouped = (
        response.groupby(
            [*group_columns, "opponent_prev_unsafe"],
            observed=True,
        )["valid_unsafe"]
        .agg(n="count", rate="mean")
        .reset_index()
    )
    for previous, label in ((0.0, "safe"), (1.0, "unsafe")):
        subset = response_grouped.loc[
            response_grouped["opponent_prev_unsafe"].eq(previous),
            [*group_columns, "n", "rate"],
        ].rename(
            columns={
                "n": f"n_after_opponent_{label}",
                "rate": f"unsafe_after_opponent_{label}",
            }
        )
        metrics = metrics.merge(subset, on=group_columns, how="left")
    metrics["opponent_response_difference"] = (
        metrics["unsafe_after_opponent_unsafe"]
        - metrics["unsafe_after_opponent_safe"]
    )

    state_rows = turns.loc[
        turns["valid_unsafe"].notna() & turns["race_state"].notna()
    ]
    state_grouped = (
        state_rows.groupby([*group_columns, "race_state"], observed=True)[
            "valid_unsafe"
        ]
        .agg(n="count", rate="mean")
        .reset_index()
    )
    for state in ("ahead", "tied", "behind"):
        subset = state_grouped.loc[
            state_grouped["race_state"].eq(state),
            [*group_columns, "n", "rate"],
        ].rename(
            columns={
                "n": f"n_{state}",
                "rate": f"unsafe_while_{state}",
            }
        )
        metrics = metrics.merge(subset, on=group_columns, how="left")
    metrics["behind_minus_ahead"] = (
        metrics["unsafe_while_behind"] - metrics["unsafe_while_ahead"]
    )

    roster = players[[*PLAYER_KEY, "outcome"]]
    metrics = metrics.merge(roster, on=PLAYER_KEY, how="left", validate="one_to_one")

    classifications = _classify_player_trajectories(
        turns,
        include_exploratory_behind=include_exploratory_behind,
    )
    metrics = metrics.merge(
        classifications,
        on=PLAYER_KEY,
        how="left",
        validate="one_to_one",
    )
    return metrics.sort_values(PLAYER_KEY).reset_index(drop=True)


def _classify_player_trajectories(
    turns: pd.DataFrame,
    *,
    include_exploratory_behind: bool,
) -> pd.DataFrame:
    if str(REPOSITORY_ROOT) not in sys.path:
        sys.path.insert(0, str(REPOSITORY_ROOT))
    from strategy_analysis.classify import classify_trajectory

    rows: list[dict[str, Any]] = []
    for key, trajectory in turns.groupby(PLAYER_KEY, sort=True, observed=True):
        trajectory = trajectory.sort_values("round")
        complete = (
            trajectory["valid_unsafe"].notna().all()
            and trajectory["opponent_current_unsafe"].notna().all()
        )
        base = dict(zip(PLAYER_KEY, key))
        if not complete:
            rows.append(
                {
                    **base,
                    "strategy_classifiable": False,
                    "strategy_best": None,
                    "strategy_tied": None,
                    "strategy_min_mismatches": np.nan,
                    "strategy_min_mismatch_rate": np.nan,
                }
            )
            continue

        result = classify_trajectory(
            trajectory["valid_unsafe"].astype(int).tolist(),
            trajectory["opponent_current_unsafe"].astype(int).tolist(),
            include_exploratory_behind=include_exploratory_behind,
            progress_gaps_before=(
                trajectory["progress_gap_before"].tolist()
                if include_exploratory_behind
                else None
            ),
        )
        minimum = min(match.mismatches for match in result.matches)
        minimum_rate = min(match.mismatch_rate for match in result.matches)
        rows.append(
            {
                **base,
                "strategy_classifiable": True,
                "strategy_best": result.unique_best_strategy,
                "strategy_tied": "|".join(result.best_strategies),
                "strategy_min_mismatches": minimum,
                "strategy_min_mismatch_rate": minimum_rate,
                **{
                    f"mismatch_rate_{match.strategy.lower()}": match.mismatch_rate
                    for match in result.matches
                },
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                *PLAYER_KEY,
                "strategy_classifiable",
                "strategy_best",
                "strategy_tied",
                "strategy_min_mismatches",
                "strategy_min_mismatch_rate",
            ]
        )
    return pd.DataFrame.from_records(rows)


def _welch_contrast(left: pd.Series, right: pd.Series) -> dict[str, float]:
    """Welch t, Cohen's d, and (when SciPy is available) a two-sided p-value.

    Reported alongside the raw group means so a contrast can still be read when
    the optional analysis extra is not installed.
    """

    left = left.dropna().astype(float)
    right = right.dropna().astype(float)
    n_left, n_right = len(left), len(right)
    result: dict[str, float] = {
        "n_left": n_left,
        "n_right": n_right,
        "mean_left": float(left.mean()) if n_left else float("nan"),
        "mean_right": float(right.mean()) if n_right else float("nan"),
        "t": float("nan"),
        "df": float("nan"),
        "p_value": float("nan"),
        "cohens_d": float("nan"),
    }
    if n_left < 2 or n_right < 2:
        return result

    var_left = float(left.var(ddof=1))
    var_right = float(right.var(ddof=1))
    standard_error = math.sqrt(var_left / n_left + var_right / n_right)
    if standard_error <= 0:
        return result
    result["t"] = (result["mean_left"] - result["mean_right"]) / standard_error
    numerator = (var_left / n_left + var_right / n_right) ** 2
    denominator = (var_left / n_left) ** 2 / (n_left - 1) + (
        var_right / n_right
    ) ** 2 / (n_right - 1)
    result["df"] = numerator / denominator if denominator > 0 else float("nan")

    pooled_variance = (
        (n_left - 1) * var_left + (n_right - 1) * var_right
    ) / (n_left + n_right - 2)
    if pooled_variance > 0:
        result["cohens_d"] = (
            result["mean_left"] - result["mean_right"]
        ) / math.sqrt(pooled_variance)

    try:
        from scipy import stats
    except ImportError:
        # p stays NaN: t and d are still informative, and refusing to emit the
        # table at all would hide the descriptive contrast.
        return result
    if math.isfinite(result["df"]):
        result["p_value"] = float(
            2.0 * stats.t.sf(abs(result["t"]), result["df"])
        )
    return result


def _pairwise_contrasts(
    frame: pd.DataFrame,
    *,
    strata: Sequence[str],
    factor: str,
    value: str,
) -> pd.DataFrame:
    """Every within-stratum pairwise contrast of ``factor`` levels on ``value``.

    Bonferroni correction uses the number of comparisons inside one stratum, which
    is what the source paper reports for its three treatment contrasts.
    """

    rows: list[dict[str, Any]] = []
    usable = frame.loc[frame[value].notna()]
    strata = [column for column in strata if column in usable.columns]
    for stratum, group in usable.groupby(list(strata), dropna=False, observed=True):
        levels = sorted(group[factor].dropna().unique().tolist())
        pairs = [
            (left, right)
            for index, left in enumerate(levels)
            for right in levels[index + 1 :]
        ]
        for left, right in pairs:
            contrast = _welch_contrast(
                group.loc[group[factor].eq(left), value],
                group.loc[group[factor].eq(right), value],
            )
            corrected = contrast["p_value"] * len(pairs)
            rows.append(
                {
                    **dict(zip(strata, stratum if isinstance(stratum, tuple) else (stratum,))),
                    "factor": factor,
                    "level_left": left,
                    "level_right": right,
                    "n_comparisons_in_stratum": len(pairs),
                    **contrast,
                    "p_value_bonferroni": min(corrected, 1.0)
                    if math.isfinite(corrected)
                    else float("nan"),
                }
            )
    if not rows:
        return pd.DataFrame(
            columns=[
                *strata,
                "factor",
                "level_left",
                "level_right",
                "n_comparisons_in_stratum",
                "n_left",
                "n_right",
                "mean_left",
                "mean_right",
                "t",
                "df",
                "p_value",
                "cohens_d",
                "p_value_bonferroni",
            ]
        )
    return pd.DataFrame.from_records(rows)


def _winner_loser_pairs(player_metrics: pd.DataFrame) -> pd.DataFrame:
    """One row per decided race: the winner's and the loser's Unsafe frequency.

    Ties are excluded because the paper's winner/loser comparison is undefined for
    them; they are counted separately in the correlation table.
    """

    columns = [*RACE_KEY, *CONTEXT, "player_id", "outcome", "unsafe_rate"]
    usable = player_metrics.loc[
        player_metrics["outcome"].isin(["winner", "loser"]),
        [column for column in columns if column in player_metrics],
    ].copy()
    if usable.empty:
        return pd.DataFrame(columns=[*RACE_KEY, *CONTEXT, "winner_unsafe_rate", "loser_unsafe_rate"])

    wide = usable.pivot_table(
        index=[*RACE_KEY, *CONTEXT],
        columns="outcome",
        values="unsafe_rate",
        aggfunc="first",
        observed=True,
    ).reset_index()
    wide = wide.rename(
        columns={"winner": "winner_unsafe_rate", "loser": "loser_unsafe_rate"}
    )
    for column in ("winner_unsafe_rate", "loser_unsafe_rate"):
        if column not in wide:
            wide[column] = np.nan
    complete = wide.loc[
        wide["winner_unsafe_rate"].notna() & wide["loser_unsafe_rate"].notna()
    ].copy()
    complete["winner_minus_loser"] = (
        complete["winner_unsafe_rate"] - complete["loser_unsafe_rate"]
    )
    return complete.sort_values([*RACE_KEY]).reset_index(drop=True)


def _build_tables(
    turns: pd.DataFrame,
    player_metrics: pd.DataFrame,
    *,
    all_turns: pd.DataFrame,
    race_quality: pd.DataFrame,
) -> dict[str, pd.DataFrame]:
    by_context = [*CONTEXT]
    tables: dict[str, pd.DataFrame] = {}

    tables["unsafe_by_risk_model_turn.csv"] = _binary_summary(
        turns.loc[~turns["parse_failed"]],
        groups=by_context,
        value="valid_unsafe",
    )
    tables["unsafe_by_risk_model_player.csv"] = _mean_summary(
        player_metrics,
        groups=by_context,
        value="unsafe_rate",
        prefix="player_unsafe_rate",
    )

    response = turns.loc[
        turns["round"].ge(2)
        & turns["valid_unsafe"].notna()
        & turns["opponent_prev_unsafe"].notna()
    ].copy()
    response["opponent_previous_action"] = response["opponent_prev_unsafe"].map(
        {0.0: "safe", 1.0: "unsafe"}
    )
    tables["opponent_response_turn.csv"] = _binary_summary(
        response,
        groups=[*by_context, "opponent_previous_action"],
        value="valid_unsafe",
    )
    tables["opponent_response_player.csv"] = _mean_summary(
        player_metrics,
        groups=by_context,
        value="opponent_response_difference",
        prefix="unsafe_after_opponent_unsafe_minus_safe",
    )

    state = turns.loc[
        turns["valid_unsafe"].notna() & turns["race_state"].notna()
    ]
    tables["unsafe_by_race_state_turn.csv"] = _binary_summary(
        state,
        groups=[*by_context, "race_state"],
        value="valid_unsafe",
    )
    tables["race_state_player.csv"] = _mean_summary(
        player_metrics,
        groups=by_context,
        value="behind_minus_ahead",
        prefix="unsafe_behind_minus_ahead",
    )

    persistence = turns.loc[
        turns["round"].ge(2)
        & turns["valid_unsafe"].notna()
        & turns["first_round_unsafe"].notna()
    ].copy()
    persistence["first_round_action"] = persistence["first_round_unsafe"].map(
        {0.0: "safe", 1.0: "unsafe"}
    )
    tables["first_round_persistence_turn.csv"] = _binary_summary(
        persistence,
        groups=[*by_context, "first_round_action"],
        value="valid_unsafe",
    )
    player_persistence = player_metrics.loc[
        player_metrics["first_round_unsafe"].notna()
    ].copy()
    player_persistence["first_round_action"] = player_persistence[
        "first_round_unsafe"
    ].map({0.0: "safe", 1.0: "unsafe"})
    tables["first_round_persistence_player.csv"] = _mean_summary(
        player_persistence,
        groups=[*by_context, "first_round_action"],
        value="later_unsafe_rate",
        prefix="later_player_unsafe_rate",
    )

    parse_failures = (
        all_turns.groupby(by_context, observed=True)
        .agg(
            n_decisions=("parse_failed", "size"),
            parse_failures=("parse_failed", "sum"),
            parse_failure_rate=("parse_failed", "mean"),
            total_retries=("retry_count", "sum"),
            mean_retries=("retry_count", "mean"),
        )
        .reset_index()
        .sort_values(by_context)
        .reset_index(drop=True)
    )
    parse_failures["parse_failures"] = parse_failures["parse_failures"].astype(int)
    race_parse_health = (
        race_quality.groupby(by_context, observed=True)
        .agg(
            n_races=("game_id", "size"),
            contaminated_races=("any_parse_failure", "sum"),
            parse_clean_races=("parse_clean", "sum"),
            canonical_mechanism_races=("canonical_mechanism", "sum"),
            forced_stop_races=("stop_forced", "sum"),
            behavioral_races=("included_in_behavioral_estimands", "sum"),
        )
        .reset_index()
    )
    race_parse_health["noncanonical_mechanism_races"] = (
        race_parse_health["n_races"]
        - race_parse_health["canonical_mechanism_races"]
    )
    for column in (
        "contaminated_races",
        "parse_clean_races",
        "canonical_mechanism_races",
        "noncanonical_mechanism_races",
        "forced_stop_races",
        "behavioral_races",
    ):
        race_parse_health[column] = race_parse_health[column].astype(int)
    parse_failures = parse_failures.merge(
        race_parse_health,
        on=by_context,
        how="outer",
        validate="one_to_one",
    )
    tables["parse_failures.csv"] = parse_failures

    quality_columns = [
        *RACE_KEY,
        *CONTEXT,
        "rep",
        "randomization_block_id",
        "manifest_counts_verified",
        "n_rounds",
        "canonical_risk_treatment",
        "canonical_minimum_horizon",
        "stop_forced",
        "canonical_mechanism",
        "parse_failures",
        "any_parse_failure",
        "parse_clean",
        "included_in_behavioral_estimands",
        "behavioral_exclusion_reason",
    ]
    tables["race_quality.csv"] = race_quality[
        [column for column in quality_columns if column in race_quality]
    ].sort_values(RACE_KEY).reset_index(drop=True)

    tables["outcome_player.csv"] = _mean_summary(
        player_metrics,
        groups=[*by_context, "outcome"],
        value="unsafe_rate",
        prefix="player_unsafe_rate",
    )

    strategy_rows = player_metrics.loc[
        player_metrics["strategy_classifiable"].eq(True)
    ].copy()
    if not strategy_rows.empty:
        strategy_rows["nearest_strategy_set"] = strategy_rows["strategy_tied"]
        strategy_summary = (
            strategy_rows.groupby(
                [*by_context, "nearest_strategy_set"],
                observed=True,
            )
            .agg(
                n_players=("player_id", "size"),
                mean_min_mismatch_rate=("strategy_min_mismatch_rate", "mean"),
            )
            .reset_index()
            .sort_values([*by_context, "nearest_strategy_set"])
            .reset_index(drop=True)
        )
    else:
        strategy_summary = pd.DataFrame(
            columns=[
                *by_context,
                "nearest_strategy_set",
                "n_players",
                "mean_min_mismatch_rate",
            ]
        )
    tables["strategy_summary_player.csv"] = strategy_summary
    tables["player_metrics.csv"] = player_metrics

    # --- Figure-2A analogue: pairwise treatment contrasts -------------------
    risk_strata = [column for column in by_context if column != "max_private_risk"]
    tables["treatment_contrasts.csv"] = _pairwise_contrasts(
        player_metrics,
        strata=risk_strata,
        factor="max_private_risk",
        value="unsafe_rate",
    )
    persona_strata = [column for column in by_context if column != "persona_condition"]
    tables["persona_contrasts.csv"] = _pairwise_contrasts(
        player_metrics,
        strata=persona_strata,
        factor="persona_condition",
        value="unsafe_rate",
    )

    # --- Figure-2B analogue: lagged action profile and race position --------
    lagged = turns.loc[
        turns["round"].ge(2)
        & turns["valid_unsafe"].notna()
        & turns["own_prev_unsafe"].notna()
        & turns["opponent_prev_unsafe"].notna()
    ].copy()
    lagged["own_previous_action"] = lagged["own_prev_unsafe"].map(
        {0.0: "safe", 1.0: "unsafe"}
    )
    lagged["opponent_previous_action"] = lagged["opponent_prev_unsafe"].map(
        {0.0: "safe", 1.0: "unsafe"}
    )
    # This is also the empirical transition matrix P(Unsafe_t | own_{t-1},
    # opponent_{t-1}); a separate transition table would repeat these numbers.
    tables["unsafe_by_lag_profile_turn.csv"] = _binary_summary(
        lagged,
        groups=[*by_context, "own_previous_action", "opponent_previous_action"],
        value="valid_unsafe",
    )
    binned = turns.loc[turns["valid_unsafe"].notna() & turns["gap_bin"].notna()]
    tables["unsafe_by_gap_bin_turn.csv"] = _binary_summary(
        binned,
        groups=[*by_context, "gap_bin"],
        value="valid_unsafe",
    )
    tables["unsafe_by_gap_lag_turn.csv"] = _binary_summary(
        lagged.loc[lagged["gap_bin"].notna()],
        groups=[
            *by_context,
            "gap_bin",
            "own_previous_action",
            "opponent_previous_action",
        ],
        value="valid_unsafe",
    )

    # --- Figure-2C analogue: winner versus loser Unsafe frequency -----------
    pairs = _winner_loser_pairs(player_metrics)
    tables["winner_loser_pairs.csv"] = pairs
    if pairs.empty:
        tables["winner_loser_correlation.csv"] = pd.DataFrame(
            columns=[
                *by_context,
                "n_decided_races",
                "mean_winner_unsafe_rate",
                "mean_loser_unsafe_rate",
                "mean_winner_minus_loser",
                "pearson_r",
            ]
        )
    else:
        correlation = (
            pairs.groupby(by_context, observed=True)
            .apply(
                lambda group: pd.Series(
                    {
                        "n_decided_races": int(len(group)),
                        "mean_winner_unsafe_rate": group[
                            "winner_unsafe_rate"
                        ].mean(),
                        "mean_loser_unsafe_rate": group["loser_unsafe_rate"].mean(),
                        "mean_winner_minus_loser": group[
                            "winner_minus_loser"
                        ].mean(),
                        "pearson_r": (
                            group["winner_unsafe_rate"].corr(
                                group["loser_unsafe_rate"]
                            )
                            if len(group) > 1
                            else float("nan")
                        ),
                    }
                ),
                include_groups=False,
            )
            .reset_index()
        )
        tables["winner_loser_correlation.csv"] = correlation

    # --- Figure-S1 analogue: realised horizon -------------------------------
    horizon_source = race_quality.loc[
        race_quality["behavioral_exclusion_reason"].eq("included")
        if "behavioral_exclusion_reason" in race_quality
        else slice(None)
    ]
    if "n_rounds" in horizon_source and not horizon_source.empty:
        horizon_groups = [
            column for column in by_context if column in horizon_source.columns
        ]
        tables["horizon_distribution.csv"] = (
            horizon_source.groupby([*horizon_groups, "n_rounds"], observed=True)
            .size()
            .reset_index(name="n_races")
            .sort_values([*horizon_groups, "n_rounds"])
            .reset_index(drop=True)
        )
    else:
        tables["horizon_distribution.csv"] = pd.DataFrame(
            columns=[*by_context, "n_rounds", "n_races"]
        )

    # --- LLM-only diagnostics -----------------------------------------------
    seat_rows = turns.loc[turns["valid_unsafe"].notna() & turns["seat_index"].notna()]
    tables["seat_balance.csv"] = _binary_summary(
        seat_rows,
        groups=[*by_context, "seat_index"],
        value="valid_unsafe",
    )

    # progress_gap_before is 0.5 * (own Unsafe count - opponent Unsafe count) by
    # construction, so "being behind" and "having been safer than the opponent"
    # are one variable. Emitting the identity residual and the correlation keeps
    # that from being read as two independent predictors.
    collinearity_rows: list[dict[str, Any]] = []
    gap_source = turns.loc[
        turns["progress_gap_before"].notna()
        & turns["unsafe_count_diff_before"].notna()
    ]
    for key, group in gap_source.groupby(by_context, observed=True):
        implied = 0.5 * group["unsafe_count_diff_before"].astype(float)
        residual = (group["progress_gap_before"].astype(float) - implied).abs()
        collinearity_rows.append(
            {
                **dict(zip(by_context, key if isinstance(key, tuple) else (key,))),
                "n_decisions": int(len(group)),
                "max_abs_identity_residual": float(residual.max()),
                "pearson_r_gap_vs_unsafe_diff": float(
                    group["progress_gap_before"].corr(
                        group["unsafe_count_diff_before"]
                    )
                )
                if len(group) > 1
                else float("nan"),
            }
        )
    tables["gap_collinearity.csv"] = (
        pd.DataFrame.from_records(collinearity_rows)
        if collinearity_rows
        else pd.DataFrame(
            columns=[
                *by_context,
                "n_decisions",
                "max_abs_identity_residual",
                "pearson_r_gap_vs_unsafe_diff",
            ]
        )
    )
    return tables


def _logit_formula_for(frame: pd.DataFrame) -> str:
    """Add protocol controls without duplicating model and signature fixed effects."""

    prompt_versions = frame["prompt_version"].astype(str).nunique()
    models = frame["model"].astype(str).nunique()
    run_phases = frame["run_phase"].astype(str).nunique()
    run_statuses = frame["run_status"].astype(str).nunique()
    signatures_per_model = frame.groupby(
        "model",
        observed=True,
    )["protocol_signature"].nunique()
    mixed_contract_within_model = bool(signatures_per_model.gt(1).any())

    formula = LOGIT_FORMULA
    if mixed_contract_within_model:
        # A signature contains model identity, prompt hash, exact source revision,
        # decoding, seed, mechanism, and runtime provenance. Adding C(model) or
        # C(prompt_version) beside it would be redundant or rank-deficient.
        formula += " + C(protocol_signature)"
    else:
        if prompt_versions > 1:
            formula += " + C(prompt_version)"
        if models > 1:
            formula += " + C(model)"
    if run_phases > 1:
        formula += " + C(run_phase)"
    if run_statuses > 1:
        formula += " + C(run_status)"
    # Persona leaves prompt_version and protocol_signature untouched, so pooling
    # persona cells without a control would load the persona effect onto the
    # treatment and lagged-action coefficients. It is only added when it is
    # identified; see _persona_identification for what "absorbed" means here.
    identification = _persona_identification(frame)
    if identification["identified"]:
        formula += " + C(persona_condition)"
    return formula


def _persona_identification(frame: pd.DataFrame) -> dict[str, Any]:
    """Decide whether a persona effect is separable from the protocol signature.

    Each persona cell is its own experiment configuration and therefore its own
    run directory. If those runs also differ in source revision, decoding, or
    package versions — anything the protocol signature covers — then persona
    varies *between* signatures and never within one. The two sets of dummies are
    then identical columns: the fit is singular, and even where it is not, no
    contrast can attribute a difference to persona rather than to the batch.

    The only way to identify a persona effect is to run every cell in one batch,
    so that the cells share a signature and persona varies inside it.
    """

    conditions = sorted(frame["persona_condition"].astype(str).unique().tolist())
    within_signature = frame.groupby(
        "protocol_signature",
        observed=True,
    )["persona_condition"].nunique()
    varies_within_signature = bool(within_signature.gt(1).any())
    return {
        "persona_conditions": conditions,
        "n_persona_conditions": len(conditions),
        "varies_within_protocol_signature": varies_within_signature,
        "confounded_with_protocol_signature": (
            len(conditions) > 1 and not varies_within_signature
        ),
        "identified": len(conditions) > 1 and varies_within_signature,
        "remedy": (
            "Run every persona cell in one batch so they share source revision, "
            "decoding, and package versions; persona then varies inside a single "
            "protocol signature and its coefficient is estimable."
        ),
    }


def _fit_clustered_logit(
    turns: pd.DataFrame,
    *,
    output_directory: Path,
    allow_mixed_protocols: bool = False,
) -> list[str]:
    try:
        import statsmodels.formula.api as smf
    except ImportError as exc:
        raise RuntimeError(
            "--fit-logit requires statsmodels; install the project requirements"
        ) from exc

    model_data = turns.loc[
        turns["round"].ge(2)
        & turns[
            [
                "valid_unsafe",
                "max_private_risk",
                "first_round_unsafe",
                "own_prev_unsafe",
                "opponent_prev_unsafe",
                "progress_gap_before",
            ]
        ].notna().all(axis=1)
    ].copy()
    if "randomization_block_id" not in model_data:
        raise ValueError(
            "clustered logit requires rep so a source_run/model/rep common-random-"
            "number block can be constructed"
        )
    if model_data["randomization_block_id"].isna().any():
        raise ValueError(
            "clustered logit refused: one or more included races have no valid rep/"
            "randomization_block_id"
        )
    model_data["unsafe"] = model_data["valid_unsafe"].astype(int)
    model_data["first_round_unsafe"] = model_data["first_round_unsafe"].astype(int)
    model_data["own_prev_unsafe"] = model_data["own_prev_unsafe"].astype(int)
    model_data["opponent_prev_unsafe"] = model_data[
        "opponent_prev_unsafe"
    ].astype(int)

    if model_data.empty:
        raise ValueError("no complete round-2+ observations are available for the logit")
    if model_data["unsafe"].nunique() < 2:
        raise ValueError("clustered logit requires both Safe and Unsafe outcomes")
    block_risk_counts = model_data.groupby(
        "randomization_block_id",
        observed=True,
    )["max_private_risk"].nunique()
    if block_risk_counts.lt(2).any():
        raise ValueError(
            "clustered logit refused: every included source_run/model/rep block "
            "must span at least two risk treatments so common-random-number "
            "dependence can be accounted for. Keep matched risk treatments in the "
            "same run directory and audit race-level exclusions."
        )
    n_clusters = model_data["randomization_block_id"].nunique()
    if n_clusters < 2:
        raise ValueError("clustered logit requires at least two CRN repetition blocks")

    prompt_versions = sorted(model_data["prompt_version"].astype(str).unique())
    models = sorted(model_data["model"].astype(str).unique())
    protocol_signatures = sorted(
        model_data["protocol_signature"].astype(str).unique()
    )
    run_phases = sorted(model_data["run_phase"].astype(str).unique())
    run_statuses = sorted(model_data["run_status"].astype(str).unique())
    persona_identification = _persona_identification(model_data)
    if persona_identification["confounded_with_protocol_signature"]:
        message = (
            "persona_condition varies across "
            f"{persona_identification['n_persona_conditions']} cells "
            f"({persona_identification['persona_conditions']}) but never within a "
            "protocol signature, so persona is perfectly confounded with the run "
            "batch. Its coefficient is not estimable and the treatment and "
            "lagged-action coefficients carry any persona difference. "
            + persona_identification["remedy"]
        )
        if not allow_mixed_protocols:
            raise ValueError(message)
        print(f"WARNING: {message}", file=sys.stderr)

    protocol_controls = _logit_formula_for(model_data).removeprefix(LOGIT_FORMULA)

    coefficient_frames: list[pd.DataFrame] = []
    specification_fits: list[dict[str, Any]] = []
    for specification, base_formula in LOGIT_SPECIFICATIONS:
        fitted_formula = base_formula + protocol_controls
        result = smf.logit(fitted_formula, data=model_data).fit(
            disp=False,
            cov_type="cluster",
            cov_kwds={"groups": model_data["randomization_block_id"]},
        )
        confidence = result.conf_int()
        coefficients = pd.DataFrame(
            {
                "specification": specification,
                "term": result.params.index,
                "coefficient": result.params.values,
                "cluster_robust_se": result.bse.values,
                "z": result.tvalues.values,
                "p_value": result.pvalues.values,
                "ci_95_low": confidence.iloc[:, 0].values,
                "ci_95_high": confidence.iloc[:, 1].values,
            }
        )
        coefficients["odds_ratio"] = np.exp(coefficients["coefficient"])
        coefficients["odds_ratio_ci_95_low"] = np.exp(coefficients["ci_95_low"])
        coefficients["odds_ratio_ci_95_high"] = np.exp(coefficients["ci_95_high"])
        coefficient_frames.append(coefficients)
        specification_fits.append(
            {
                "specification": specification,
                "formula": fitted_formula,
                "converged": bool(result.mle_retvals.get("converged", False)),
                "log_likelihood": float(result.llf),
                "pseudo_r_squared": float(result.prsquared),
            }
        )

    coefficient_name = "clustered_logit_coefficients.csv"
    pd.concat(coefficient_frames, ignore_index=True).to_csv(
        output_directory / coefficient_name,
        index=False,
    )

    metadata = {
        "specifications": specification_fits,
        "specification_rationale": (
            "The six nested specifications of Table 1 in the source paper. "
            "Comparing them shows whether a coefficient is stable or an artefact "
            "of one particular specification."
        ),
        "protocol_controls": protocol_controls or "(none required)",
        "n_observations": int(len(model_data)),
        "n_games": int(model_data[[*RACE_KEY]].drop_duplicates().shape[0]),
        "n_crn_repetition_blocks": int(n_clusters),
        "prompt_versions": prompt_versions,
        "persona_identification": persona_identification,
        "models": models,
        "protocol_signatures": protocol_signatures,
        "run_phases": run_phases,
        "run_statuses": run_statuses,
        "covariance": "cluster-robust by source_run::model::rep",
        "cluster_rationale": (
            "The same repetition reuses horizon/setback random streams across risk "
            "treatments; game-only clustering would not capture that dependence."
        ),
        "interpretation": (
            "Conditional associations in an endogenous repeated interaction; "
            "not causal effects."
        ),
        "excluded_covariates": (
            "Human demographics and elicited risk preference are not defined for "
            "these LLM agents."
        ),
    }
    metadata_name = "clustered_logit_metadata.json"
    (output_directory / metadata_name).write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return [coefficient_name, metadata_name]


def _tost_equivalent(
    estimate: float,
    standard_error: float,
    bound: float,
    alpha: float,
) -> tuple[bool, float]:
    """Two one-sided tests for equivalence to zero within +/- ``bound``.

    A null result in the human data must be matched by evidence of a *small*
    effect, not by a failure to reject. Returns the decision and the larger of the
    two one-sided p-values.
    """

    if not (math.isfinite(estimate) and math.isfinite(standard_error)):
        return False, float("nan")
    if standard_error <= 0:
        return abs(estimate) < bound, 0.0 if abs(estimate) < bound else 1.0
    try:
        from scipy import stats
    except ImportError:
        return False, float("nan")
    lower_p = float(stats.norm.sf((estimate - (-bound)) / standard_error))
    upper_p = float(stats.norm.cdf((estimate - bound) / standard_error))
    p_value = max(lower_p, upper_p)
    return p_value < alpha, p_value


def _build_human_comparison(
    *,
    coefficients: pd.DataFrame | None,
    treatment_contrasts: pd.DataFrame,
    player_metrics: pd.DataFrame,
    reference_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Score each preregistered human effect against the LLM estimate.

    The criteria live in ``human_reference.json`` rather than in this function so
    they can be frozen before any model output is inspected; scoring them here
    afterwards is then mechanical.
    """

    reference = json.loads(reference_path.read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []

    def coefficient_for(term: str, specification: str) -> tuple[float, float, float]:
        if coefficients is None or coefficients.empty:
            return float("nan"), float("nan"), float("nan")
        match = coefficients.loc[
            coefficients["term"].eq(term)
            & coefficients["specification"].astype(str).eq(specification)
        ]
        if match.empty:
            return float("nan"), float("nan"), float("nan")
        row = match.iloc[0]
        return (
            float(row["coefficient"]),
            float(row["cluster_robust_se"]),
            float(row["p_value"]),
        )

    for effect in reference["effects"]:
        row: dict[str, Any] = {
            "effect_id": effect["id"],
            "name": effect["name"],
            "kind": effect["kind"],
            "test": effect["test"],
            "human_value": effect.get("human_value"),
            "llm_value": float("nan"),
            "llm_se": float("nan"),
            "llm_p_value": float("nan"),
            "criterion": "",
            "verdict": "inconclusive",
            "description": effect["description"],
        }

        if effect["kind"] == "coefficient":
            estimate, standard_error, p_value = coefficient_for(
                effect["term"],
                str(effect["specification"]),
            )
            row.update(llm_value=estimate, llm_se=standard_error, llm_p_value=p_value)
            if effect["test"] == "directional":
                sign_ok = (
                    estimate > 0
                    if effect["expected_sign"] == "positive"
                    else estimate < 0
                )
                row["criterion"] = (
                    f"{effect['expected_sign']} and p < {effect['alpha']}"
                )
                if math.isfinite(estimate) and math.isfinite(p_value):
                    row["verdict"] = (
                        "replicated"
                        if sign_ok and p_value < effect["alpha"]
                        else "not_replicated"
                    )
            else:
                bound = float(effect["equivalence_bound"])
                equivalent, tost_p = _tost_equivalent(
                    estimate,
                    standard_error,
                    bound,
                    float(effect["alpha"]),
                )
                row["criterion"] = f"TOST |beta| < {bound}"
                row["llm_p_value"] = tost_p
                if math.isfinite(tost_p):
                    row["verdict"] = (
                        "replicated" if equivalent else "not_replicated"
                    )

        elif effect["kind"] == "contrast":
            match = treatment_contrasts.loc[
                treatment_contrasts["factor"].eq(effect["factor"])
                & treatment_contrasts["level_left"].astype(float).eq(
                    float(effect["level_left"])
                )
                & treatment_contrasts["level_right"].astype(float).eq(
                    float(effect["level_right"])
                )
            ]
            if not match.empty:
                # Pool the strata by averaging Cohen's d: the criteria are stated
                # for one overall contrast, and per-stratum rows stay available in
                # treatment_contrasts.csv.
                effect_size = float(match["cohens_d"].mean())
                row["llm_value"] = effect_size
                if effect["test"] == "equivalence":
                    row["criterion"] = f"|d| < {effect['equivalence_bound']}"
                    if math.isfinite(effect_size):
                        row["verdict"] = (
                            "replicated"
                            if abs(effect_size) < float(effect["equivalence_bound"])
                            else "not_replicated"
                        )
                else:
                    minimum = float(effect["minimum_absolute_effect"])
                    sign_ok = (
                        effect_size > 0
                        if effect["expected_sign"] == "positive"
                        else effect_size < 0
                    )
                    row["criterion"] = (
                        f"{effect['expected_sign']} and |d| > {minimum}"
                    )
                    if math.isfinite(effect_size):
                        row["verdict"] = (
                            "replicated"
                            if sign_ok and abs(effect_size) > minimum
                            else "not_replicated"
                        )

        elif effect["kind"] == "level":
            observed = float(player_metrics["unsafe_rate"].mean())
            low, high = effect["interval"]
            row["llm_value"] = observed
            row["criterion"] = f"within [{low}, {high}]"
            if math.isfinite(observed):
                row["verdict"] = (
                    "replicated" if low <= observed <= high else "not_replicated"
                )

        elif effect["kind"] == "strategy_share":
            unique = player_metrics.loc[player_metrics["strategy_best"].notna()]
            if len(unique):
                share = float(unique["strategy_best"].eq(effect["strategy"]).mean())
                row["llm_value"] = share
                row["criterion"] = f"share < {effect['maximum']}"
                row["verdict"] = (
                    "replicated"
                    if share < float(effect["maximum"])
                    else "not_replicated"
                )

        rows.append(row)

    metadata = {
        "reference_source": reference["source"],
        "reference_schema": reference["schema_version"],
        "human_sample": reference["human_sample"],
        "notes": reference["notes"],
        "scoring": (
            "Criteria are frozen in human_reference.json and applied mechanically. "
            "'inconclusive' means the LLM estimate was unavailable, not that the "
            "effect was absent."
        ),
        "pooling_warning": (
            "Verdicts pool every persona condition, model, and treatment present "
            "in the input. Score one condition at a time when the comparison is "
            "meant to be about a specific cell."
        ),
    }
    return pd.DataFrame.from_records(rows), metadata


def _write_outputs(
    *,
    tables: dict[str, pd.DataFrame],
    output_directory: Path,
    run_directories: Sequence[Path],
    turns: pd.DataFrame,
    all_turns: pd.DataFrame,
    race_quality: pd.DataFrame,
    races: pd.DataFrame,
    players: pd.DataFrame,
    fit_logit: bool,
    prompt_versions: Sequence[str],
    persona_conditions: Sequence[str],
    protocol_payloads: dict[str, dict[str, Any]],
    allow_mixed_protocols: bool,
    allow_nonconfirmatory_runs: bool,
    allow_nonfinal_runs: bool,
    allow_noncanonical_mechanism: bool,
    mechanics_checks: Sequence[str],
) -> None:
    output_directory.mkdir(parents=True, exist_ok=True)
    behavioral_prompt_versions = sorted(
        turns["prompt_version"].astype(str).unique().tolist()
    )
    behavioral_models = sorted(turns["model"].astype(str).unique().tolist())
    protocol_signatures = sorted(
        all_turns["protocol_signature"].astype(str).unique().tolist()
    )
    behavioral_protocol_signatures = sorted(
        turns["protocol_signature"].astype(str).unique().tolist()
    )
    run_phases = sorted(all_turns["run_phase"].astype(str).unique().tolist())
    behavioral_run_phases = sorted(
        turns["run_phase"].astype(str).unique().tolist()
    )
    run_statuses = sorted(all_turns["run_status"].astype(str).unique().tolist())
    behavioral_run_statuses = sorted(
        turns["run_status"].astype(str).unique().tolist()
    )
    written: list[str] = []
    for filename, table in tables.items():
        table.to_csv(output_directory / filename, index=False)
        written.append(filename)

    fitted_coefficients: pd.DataFrame | None = None
    if fit_logit:
        written.extend(
            _fit_clustered_logit(
                turns,
                output_directory=output_directory,
                allow_mixed_protocols=allow_mixed_protocols,
            )
        )
        fitted_coefficients = pd.read_csv(
            output_directory / "clustered_logit_coefficients.csv"
        )

    comparison, comparison_metadata = _build_human_comparison(
        coefficients=fitted_coefficients,
        treatment_contrasts=tables["treatment_contrasts.csv"],
        player_metrics=tables["player_metrics.csv"],
        reference_path=HUMAN_REFERENCE_PATH,
    )
    comparison.to_csv(output_directory / "human_comparison.csv", index=False)
    written.append("human_comparison.csv")
    (output_directory / "human_comparison_metadata.json").write_text(
        json.dumps(comparison_metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    written.append("human_comparison_metadata.json")

    manifest = {
        "source_runs": [path.resolve().as_posix() for path in run_directories],
        "n_source_runs": len(run_directories),
        "n_races_total": int(len(races)),
        "n_races_behavioral": int(
            race_quality["included_in_behavioral_estimands"].sum()
        ),
        "n_races_excluded_total": int(
            (~race_quality["included_in_behavioral_estimands"]).sum()
        ),
        "n_races_excluded_any_parse_failure": int(
            race_quality["any_parse_failure"].sum()
        ),
        "n_races_excluded_noncanonical_mechanism": int(
            (~race_quality["canonical_mechanism"]).sum()
        ),
        "n_races_excluded_forced_safety_cap_stop": int(
            race_quality["stop_forced"].sum()
        ),
        "n_player_races_total": int(len(players)),
        "n_player_races_behavioral": int(
            turns[PLAYER_KEY].drop_duplicates().shape[0]
        ),
        "n_player_rounds_total": int(len(all_turns)),
        "n_player_rounds_behavioral": int(len(turns)),
        "n_parse_failures": int(all_turns["parse_failed"].sum()),
        "all_manifest_counts_verified": bool(
            race_quality["manifest_counts_verified"].all()
        ),
        "behavior_filter": (
            "A race is included in behavioral, strategy, and logit estimands only "
            "when every decision parsed and the race used a canonical risk "
            "treatment, reached the five-round minimum, and did not terminate at "
            "a forced safety cap. All races remain in protocol-health and "
            "accounting outputs."
        ),
        "prompt_versions": list(prompt_versions),
        "behavioral_prompt_versions": behavioral_prompt_versions,
        "persona_conditions": list(persona_conditions),
        "behavioral_persona_conditions": sorted(
            turns["persona_condition"].astype(str).unique().tolist()
        ),
        # Recorded whether or not the logit ran: a reader must be able to see that
        # a persona comparison in the descriptive tables is confounded with the
        # run batch without having to notice a missing regression term.
        "persona_identification": _persona_identification(turns),
        "behavioral_models": behavioral_models,
        "protocol_signatures": protocol_signatures,
        "behavioral_protocol_signatures": behavioral_protocol_signatures,
        "protocol_signature_payloads": {
            signature: protocol_payloads[signature]
            for signature in protocol_signatures
        },
        "allow_mixed_protocols": allow_mixed_protocols,
        "run_phases": run_phases,
        "behavioral_run_phases": behavioral_run_phases,
        "run_statuses": run_statuses,
        "behavioral_run_statuses": behavioral_run_statuses,
        "allow_nonconfirmatory_runs": allow_nonconfirmatory_runs,
        "allow_nonfinal_runs": allow_nonfinal_runs,
        "allow_noncanonical_mechanism": allow_noncanonical_mechanism,
        "mechanics_checks_passed": list(mechanics_checks),
        "protocol_grouping": (
            "Every descriptive table is stratified by prompt_version, "
            "protocol_signature, run_phase, and run_status. By default each model "
            "label may have only one exact manifest contract; overrides are "
            "explicit sensitivity/protocol audits."
        ),
        "player_level_note": (
            "Player-level tables average within-player trajectory rates so longer "
            "realised horizons do not automatically receive more weight."
        ),
        "inference_cluster": (
            "source_run/model/rep common-random-number repetition block; the logit "
            "refuses inputs without valid blocks spanning risk treatments"
        ),
        "logit_requested": fit_logit,
        "logit_formula": _logit_formula_for(turns) if fit_logit else None,
        "outputs": sorted(written),
    }
    manifest_name = "analysis_manifest.json"
    (output_directory / manifest_name).write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(written) + 1} derived files to {output_directory}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        type=Path,
        dest="inputs",
        help=(
            "run directory, turns.jsonl, or ancestor directory to search; repeat to "
            "combine sources (default: results/open_source and results/frontier)"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"derived output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--fit-logit",
        action="store_true",
        help=(
            "fit the planned statsmodels logistic regression with standard errors "
            "clustered by source_run/model/rep CRN blocks"
        ),
    )
    parser.add_argument(
        "--include-exploratory-behind",
        action="store_true",
        help=(
            "include the explicitly exploratory behind-responsive rule in nearest-"
            "strategy classification"
        ),
    )
    parser.add_argument(
        "--allow-mixed-protocols",
        action="store_true",
        help=(
            "explicit sensitivity-analysis opt-in for multiple or missing "
            "prompt versions or multiple exact manifest contracts within one "
            "model; descriptive outputs remain prompt/signature-stratified and "
            "the logit controls for protocol_signature when needed"
        ),
    )
    parser.add_argument(
        "--allow-missing-persona-condition",
        action="store_true",
        help=(
            "explicit audit opt-in for runs recorded before persona labelling; "
            "unlabelled races are kept in a separate persona_condition stratum "
            "and never merged into the neutral baseline"
        ),
    )
    parser.add_argument(
        "--allow-nonconfirmatory-runs",
        action="store_true",
        help=(
            "explicit audit/sensitivity opt-in for pilot, mixed, or missing "
            "run_phase values; outputs remain phase-stratified"
        ),
    )
    parser.add_argument(
        "--allow-nonfinal-runs",
        action="store_true",
        help=(
            "explicit protocol-health opt-in for missing, running, failed, or "
            "protocol-failed run manifests; outputs remain status-stratified"
        ),
    )
    parser.add_argument(
        "--allow-noncanonical-mechanism",
        "--allow-noncanonical-horizons",
        dest="allow_noncanonical_mechanism",
        action="store_true",
        help=(
            "load noncanonical risk treatments or horizons for a labelled "
            "mechanism audit; these races remain outside behavioral estimands "
            "and this flag does not waive mechanics-consistency validation"
        ),
    )
    args = parser.parse_args(argv)

    inputs = args.inputs or [
        REPOSITORY_ROOT / "results" / "open_source",
        REPOSITORY_ROOT / "results" / "frontier",
    ]
    run_directories = _discover_run_directories(inputs)
    raw_turns, raw_races, raw_players, protocol_payloads = _read_run_tables(
        run_directories,
        allow_nonfinal_runs=args.allow_nonfinal_runs,
        allow_mixed_protocols=args.allow_mixed_protocols,
    )
    turns = _prepare_turns(raw_turns)
    races = _prepare_races(
        raw_races,
        allow_noncanonical_mechanism=args.allow_noncanonical_mechanism,
    )
    players = _prepare_players(raw_players)
    turns, races, players, prompt_versions = _resolve_prompt_versions(
        turns,
        races,
        players,
        allow_mixed_protocols=args.allow_mixed_protocols,
    )
    turns, races, players, _run_phases = _resolve_run_phases(
        turns,
        races,
        players,
        allow_nonconfirmatory_runs=args.allow_nonconfirmatory_runs,
    )
    turns, races, players, persona_conditions = _resolve_persona_conditions(
        turns,
        races,
        players,
        allow_missing_persona_condition=args.allow_missing_persona_condition,
    )
    turns, races, players = _resolve_protocol_signatures(
        turns,
        races,
        players,
        protocol_payloads=protocol_payloads,
        allow_mixed_protocols=args.allow_mixed_protocols,
    )
    turns, races, players = _resolve_repetition_blocks(
        turns,
        races,
        players,
        share_blocks_across_runs=_shared_base_seed(races),
    )
    _validate_join_keys(turns, races, players)
    turns, race_quality = _add_race_quality(turns, races)
    turns = _add_dynamic_columns(turns)
    mechanics_checks = _validate_mechanics_and_terminal_state(
        turns,
        races,
        players,
    )
    behavioral_turns = turns.loc[
        turns["included_in_behavioral_estimands"]
    ].copy()
    player_metrics = _build_player_metrics(
        behavioral_turns,
        players,
        include_exploratory_behind=args.include_exploratory_behind,
    )
    tables = _build_tables(
        behavioral_turns,
        player_metrics,
        all_turns=turns,
        race_quality=race_quality,
    )
    _write_outputs(
        tables=tables,
        output_directory=args.output.resolve(),
        run_directories=run_directories,
        turns=behavioral_turns,
        all_turns=turns,
        race_quality=race_quality,
        races=races,
        players=players,
        fit_logit=args.fit_logit,
        prompt_versions=prompt_versions,
        persona_conditions=persona_conditions,
        protocol_payloads=protocol_payloads,
        allow_mixed_protocols=args.allow_mixed_protocols,
        allow_nonconfirmatory_runs=args.allow_nonconfirmatory_runs,
        allow_nonfinal_runs=args.allow_nonfinal_runs,
        allow_noncanonical_mechanism=args.allow_noncanonical_mechanism,
        mechanics_checks=mechanics_checks,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
