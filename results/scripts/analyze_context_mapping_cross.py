#!/usr/bin/env python3
"""Fail-closed analysis for the fully crossed context × mapping follow-up."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


SCHEMA = "ai-race-context-mapping-cross-run-v1"
SKINS = (
    "abstract_contest",
    "technology_race",
    "logistics_contract",
    "crystal_guild_contract",
    "hospital_deployment",
    "colony_life_support",
    "robotic_expedition",
    "fictional_cartography",
)
MAPPINGS = ("safe_p", "safe_q")
RISKS = (0.1, 0.6, 0.9)
BOOTSTRAP_SEED = 260802


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def discover(root: Path) -> list[tuple[Path, dict[str, Any]]]:
    runs = []
    for path in sorted(root.rglob("run_manifest.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") == SCHEMA:
            runs.append((path.parent, manifest))
    if len(runs) != len(SKINS):
        raise ValueError(f"Expected {len(SKINS)} completed skin runs, found {len(runs)}")
    return runs


def load_and_validate(root: Path) -> tuple[pd.DataFrame, dict[str, Any], list[Path]]:
    runs = discover(root)
    manifests = [manifest for _, manifest in runs]
    if any(manifest.get("status") != "completed" for manifest in manifests):
        raise ValueError("Every mapping-cross manifest must be completed")
    if any(manifest.get("experiment", {}).get("runPhase") != "pilot" for manifest in manifests):
        raise ValueError("Mapping-cross evidence must remain pilot-labelled")
    if {
        manifest.get("context_skin", {}).get("id") for manifest in manifests
    } != set(SKINS):
        raise ValueError("Frozen eight-skin coverage failed")
    for field in ("experiment_config_sha256", "source_sha256", "model"):
        values = {json.dumps(manifest.get(field), sort_keys=True) for manifest in manifests}
        if len(values) != 1:
            raise ValueError(f"Incompatible manifest field {field}")

    frames, source_paths = [], []
    total_races = total_turns = 0
    for run_dir, manifest in runs:
        players_path = run_dir / "players.csv"
        races_path = run_dir / "races.csv"
        turns_path = run_dir / "turns.jsonl"
        for path in (players_path, races_path, turns_path, run_dir / "run_manifest.json"):
            if not path.is_file():
                raise ValueError(f"Missing source artifact {path}")
            source_paths.append(path)
        players = pd.read_csv(players_path)
        races = pd.read_csv(races_path)
        turns = pd.read_json(turns_path, lines=True)
        if len(races) != int(manifest["expected_races"]):
            raise ValueError(f"Incomplete races in {run_dir}")
        if len(races) != int(manifest["n_races"]) or len(turns) != int(manifest["n_turns"]):
            raise ValueError(f"Manifest counts do not match artifacts in {run_dir}")
        if int(races["parse_failures"].sum()) != int(turns["parse_failed"].sum()):
            raise ValueError(f"Parse-failure accounting mismatch in {run_dir}")
        players["skin_id"] = manifest["context_skin"]["id"]
        players["mapping_id"] = players["prompt_version"].astype(str).str.rsplit(":", n=1).str[-1]
        frames.append(players)
        total_races += len(races)
        total_turns += len(turns)

    players = pd.concat(frames, ignore_index=True)
    players["max_private_risk"] = pd.to_numeric(players["max_private_risk"])
    players["rep"] = pd.to_numeric(players["rep"]).astype(int)
    players["player_index"] = pd.to_numeric(players["player_index"]).astype(int)
    players["unsafe_frequency"] = pd.to_numeric(players["unsafe_frequency"])
    players["final_payoff"] = pd.to_numeric(players["final_payoff"])
    players["n_rounds"] = pd.to_numeric(players["n_rounds"]).astype(int)
    players["game_seed"] = pd.to_numeric(players["game_seed"]).astype(int)
    if set(players["mapping_id"]) != set(MAPPINGS):
        raise ValueError("Both opaque mappings are required")
    if set(players["max_private_risk"].round(6)) != set(RISKS):
        raise ValueError("Frozen risk coverage failed")
    key = ["skin_id", "max_private_risk", "rep", "player_index", "mapping_id"]
    if players.duplicated(key).any():
        raise ValueError("Duplicate player-level crossed cell")
    expected_rows = len(SKINS) * len(RISKS) * 32 * 2 * len(MAPPINGS)
    if len(players) != expected_rows:
        raise ValueError(f"Expected {expected_rows} player rows, found {len(players)}")
    mapping_counts = players.groupby(key[:-1], dropna=False)["mapping_id"].nunique()
    if not (mapping_counts == len(MAPPINGS)).all():
        raise ValueError("A paired block is missing an opaque mapping")
    invariants = players.groupby(key[:-1], dropna=False).agg(
        seeds=("game_seed", "nunique"), horizons=("n_rounds", "nunique")
    )
    if not (invariants == 1).all().all():
        raise ValueError("Game seed or hidden horizon differs across paired mappings")
    audit = {
        "status": "passed",
        "evidence_class": "preregistered_diagnostic_pilot",
        "n_runs": len(runs),
        "n_races": total_races,
        "n_player_races": len(players),
        "n_decisions": total_turns,
        "n_contexts": players["skin_id"].nunique(),
        "n_mappings": players["mapping_id"].nunique(),
        "n_risks": players["max_private_risk"].nunique(),
        "parse_failures": int(sum(pd.read_csv(path / "races.csv")["parse_failures"].sum() for path, _ in runs)),
        "claim_boundary": "Checkpoint-scoped mapping/context interaction; not game understanding or cross-model generality.",
    }
    return players, audit, source_paths


def paired_rows(players: pd.DataFrame) -> pd.DataFrame:
    index = ["max_private_risk", "rep", "player_index", "mapping_id"]
    wide = players.pivot(index=index, columns="skin_id", values="unsafe_frequency")
    if wide.isna().any().any():
        raise ValueError("Incomplete context pairing")
    rows = []
    contexts = [context for context in SKINS if context != "abstract_contest" and context in wide]
    for context in contexts:
        delta = (wide[context] - wide["abstract_contest"]).rename("context_delta")
        frame = delta.reset_index()
        pivot = frame.pivot(
            index=["max_private_risk", "rep", "player_index"],
            columns="mapping_id",
            values="context_delta",
        ).reset_index()
        pivot["interaction_did"] = pivot["safe_p"] - pivot["safe_q"]
        pivot["context"] = context
        rows.append(pivot)
    return pd.concat(rows, ignore_index=True)


def cluster_bootstrap(values: pd.DataFrame, repetitions: int) -> tuple[float, float]:
    cluster = (
        values.groupby(["max_private_risk", "rep"], observed=True)["interaction_did"]
        .mean()
        .to_numpy()
    )
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    draws = rng.choice(cluster, size=(repetitions, len(cluster)), replace=True).mean(axis=1)
    return tuple(np.quantile(draws, [0.025, 0.975]).tolist())


def sign_flip_p(values: pd.DataFrame, repetitions: int) -> float:
    cluster = (
        values.groupby(["max_private_risk", "rep"], observed=True)["interaction_did"]
        .mean()
        .to_numpy()
    )
    observed = abs(float(cluster.mean()))
    rng = np.random.default_rng(BOOTSTRAP_SEED + 1)
    extreme = 0
    for _ in range(repetitions):
        signs = rng.choice((-1.0, 1.0), size=len(cluster))
        extreme += abs(float((cluster * signs).mean())) >= observed
    return (extreme + 1) / (repetitions + 1)


def holm(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        candidate = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted.tolist()


def summarize(rows: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    results = []
    for context, subset in rows.groupby("context", sort=False):
        low, high = cluster_bootstrap(subset, repetitions)
        results.append(
            {
                "context": context,
                "n_player_blocks": len(subset),
                "safe_p_context_delta": subset["safe_p"].mean(),
                "safe_q_context_delta": subset["safe_q"].mean(),
                "interaction_did": subset["interaction_did"].mean(),
                "ci95_low": low,
                "ci95_high": high,
                "sign_flip_p": sign_flip_p(subset, repetitions),
            }
        )
    result = pd.DataFrame(results)
    result["holm_p"] = holm(result["sign_flip_p"].tolist())
    result["replicated_direction"] = result["interaction_did"] > 0
    result["promotion_passed"] = (
        result["replicated_direction"]
        & (result["ci95_low"] > 0)
        & (result["holm_p"] < 0.05)
    )
    return result


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    ordered = summary.sort_values("interaction_did")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.axvline(0, color="#64748b", linewidth=1)
    ax.errorbar(
        ordered["interaction_did"] * 100,
        y,
        xerr=np.vstack(
            [
                (ordered["interaction_did"] - ordered["ci95_low"]) * 100,
                (ordered["ci95_high"] - ordered["interaction_did"]) * 100,
            ]
        ),
        fmt="o",
        color="#2563eb",
        ecolor="#93c5fd",
        capsize=4,
    )
    ax.set_yticks(y, ordered["context"].str.replace("_", " ").str.title())
    ax.set_xlabel("Context × mapping difference-in-differences (percentage points)")
    ax.set_title("Fully crossed action mapping tests the pilot's largest validity gap", loc="left", weight="bold")
    ax.text(
        0,
        1.02,
        "Positive values mean the context effect is larger when P denotes Safe; cluster-bootstrap 95% intervals",
        transform=ax.transAxes,
        color="#64748b",
        fontsize=9,
    )
    ax.grid(axis="x", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output / "context_mapping_interaction.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "context_mapping_interaction.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repetitions", type=int, default=5000)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    players, audit, sources = load_and_validate(args.input_root.resolve())
    rows = paired_rows(players)
    summary = summarize(rows, args.repetitions)
    rows.to_csv(output / "paired_player_blocks.csv", index=False)
    summary.to_csv(output / "interaction_summary.csv", index=False)
    plot_summary(summary, output)
    (output / "quality_audit.json").write_text(
        json.dumps(audit, indent=2) + "\n", encoding="utf-8"
    )
    manifest = {
        "schema_version": "ai-race-context-mapping-cross-analysis-v1",
        "status": "complete",
        "bootstrap_repetitions": args.repetitions,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "sources": {str(path): sha256(path) for path in sources},
        "outputs": {
            path.name: {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in output.iterdir()
            if path.is_file() and path.name != "analysis_manifest.json"
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps({"status": "complete", "audit": audit}, indent=2))


if __name__ == "__main__":
    main()
