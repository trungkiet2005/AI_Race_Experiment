"""Shared fail-closed helpers for the frozen impact follow-up analyses."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd


BOOTSTRAP_SEED = 260802
MIN_BOOTSTRAP_REPETITIONS = 1_000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover_runs(root: Path, schema: str) -> list[tuple[Path, dict[str, Any]]]:
    runs: list[tuple[Path, dict[str, Any]]] = []
    for path in sorted(root.rglob("run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") == schema:
            runs.append((path.parent, manifest))
    if not runs:
        raise ValueError(f"No {schema!r} run manifests found under {root}")
    return runs


def require_one_value(manifests: Iterable[dict[str, Any]], field: str) -> Any:
    values = [manifest.get(field) for manifest in manifests]
    encoded = {json.dumps(value, sort_keys=True) for value in values}
    if len(encoded) != 1 or values[0] is None:
        raise ValueError(f"Incompatible or missing manifest field {field!r}")
    return values[0]


def validate_common_manifests(manifests: list[dict[str, Any]]) -> dict[str, Any]:
    if any(manifest.get("status") != "completed" for manifest in manifests):
        raise ValueError("Every run manifest must have status='completed'")
    if any(
        manifest.get("experiment", {}).get("runPhase") != "pilot"
        for manifest in manifests
    ):
        raise ValueError("Follow-up evidence must remain pilot-labelled")

    common = {
        field: require_one_value(manifests, field)
        for field in ("experiment_config_sha256", "source_sha256", "model", "decoding")
    }
    model = common["model"]
    if not isinstance(model, dict) or not model.get("config_sha256"):
        raise ValueError("An exact non-empty model digest is required")
    decoding = common["decoding"]
    if not isinstance(decoding, dict):
        raise ValueError("Decoding provenance must be an object")
    if float(decoding.get("temperature", float("nan"))) != 0.0:
        raise ValueError("Frozen follow-up decoding temperature must equal 0")
    if not decoding.get("seed_requested") or not decoding.get(
        "seed_probe_exact_match"
    ):
        raise ValueError("Fixed-seed request and reproducibility probe must pass")
    ollama_digests = {
        manifest.get("ollama_model", {}).get("digest") for manifest in manifests
    }
    if len(ollama_digests) != 1 or None in ollama_digests or "" in ollama_digests:
        raise ValueError("Exact Ollama digest is missing or differs across runs")
    if next(iter(ollama_digests)) != model["config_sha256"]:
        raise ValueError("Model digest fields disagree inside the manifests")
    return common


def read_run_artifacts(
    run_dir: Path, manifest: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[Path]]:
    races_path = run_dir / "races.csv"
    players_path = run_dir / "players.csv"
    turns_path = run_dir / "turns.jsonl"
    manifest_path = run_dir / "run_manifest.json"
    paths = [races_path, players_path, turns_path, manifest_path]
    missing = [str(path) for path in paths if not path.is_file()]
    if missing:
        raise ValueError(f"Missing source artifacts: {missing}")
    races = pd.read_csv(races_path)
    players = pd.read_csv(players_path)
    turns = pd.read_json(turns_path, lines=True)
    expected_races = int(manifest["expected_races"])
    if len(races) != expected_races or len(races) != int(manifest["n_races"]):
        raise ValueError(f"Race coverage disagrees with manifest in {run_dir}")
    if len(turns) != int(manifest["n_turns"]):
        raise ValueError(f"Turn coverage disagrees with manifest in {run_dir}")
    if len(players) != 2 * len(races):
        raise ValueError(f"Expected exactly two player rows per race in {run_dir}")
    if int(races["parse_failures"].sum()) != int(turns["parse_failed"].sum()):
        raise ValueError(f"Parse-failure accounting mismatch in {run_dir}")
    if races["game_id"].duplicated().any():
        raise ValueError(f"Duplicate race id in {run_dir}")
    if players.duplicated(["game_id", "player_index"]).any():
        raise ValueError(f"Duplicate player-race row in {run_dir}")
    if turns.duplicated(["game_id", "player_index", "round"]).any():
        raise ValueError(f"Duplicate turn row in {run_dir}")
    return races, players, turns, paths


def holm(p_values: list[float]) -> list[float]:
    if not p_values:
        return []
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * float(p_values[index]))
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def clustered_values(rows: pd.DataFrame, value: str) -> np.ndarray:
    """Return equal-weight means for independent CRN repetition streams.

    Follow-up runners reuse ``base_seed + rep`` across all risk treatments.
    Risk strata within a repetition are dependent views of the same random
    stream, not independent clusters.
    """
    if "rep" not in rows.columns:
        raise ValueError("CRN inference requires the repetition identifier 'rep'")
    clusters = (
        rows.groupby("rep", observed=True)[value]
        .mean()
        .to_numpy(dtype=float)
    )
    if len(clusters) == 0:
        raise ValueError("No complete CRN clusters remain for inference")
    return clusters


def bootstrap_ci(
    rows: pd.DataFrame, value: str, repetitions: int, *, seed_offset: int = 0
) -> tuple[float, float]:
    if repetitions < MIN_BOOTSTRAP_REPETITIONS:
        raise ValueError(
            f"At least {MIN_BOOTSTRAP_REPETITIONS} frozen bootstrap draws are required"
        )
    clusters = clustered_values(rows, value)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    draws = rng.choice(
        clusters, size=(repetitions, len(clusters)), replace=True
    ).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def sign_flip_p(
    rows: pd.DataFrame, value: str, repetitions: int, *, seed_offset: int = 0
) -> float:
    if repetitions < MIN_BOOTSTRAP_REPETITIONS:
        raise ValueError(
            f"At least {MIN_BOOTSTRAP_REPETITIONS} frozen randomization draws are required"
        )
    clusters = clustered_values(rows, value)
    observed = abs(float(clusters.mean()))
    rng = np.random.default_rng(BOOTSTRAP_SEED + 10_000 + seed_offset)
    signs = rng.choice((-1.0, 1.0), size=(repetitions, len(clusters)))
    null = np.abs((signs * clusters).mean(axis=1))
    return float((np.count_nonzero(null >= observed) + 1) / (repetitions + 1))


def write_analysis_manifest(
    *, output: Path, schema: str, repetitions: int, sources: list[Path]
) -> None:
    payload = {
        "schema_version": schema,
        "status": "complete",
        "bootstrap_repetitions": repetitions,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sources": {str(path): sha256(path) for path in sorted(set(sources))},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in sorted(output.iterdir())
            if path.is_file() and path.name != "analysis_manifest.json"
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
