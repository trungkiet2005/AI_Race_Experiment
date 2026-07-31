"""Validate and summarize paired AI Race prompt surface-sensitivity runs."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def clustered_rate_ci(
    counts_by_rep: dict[int, tuple[int, int]], *, seed_label: str, draws: int = 5000
) -> tuple[float, float]:
    """Percentile CI from a repetition-block bootstrap.

    A repetition is the independent randomization unit shared across risk cells,
    seats, horizons, and prompt variants. Resampling individual decisions would
    therefore produce spuriously narrow intervals.
    """
    reps = sorted(counts_by_rep)
    if not reps or any(total <= 0 for _, total in counts_by_rep.values()):
        raise ValueError("Cluster bootstrap requires non-empty repetition blocks")
    seed = int(hashlib.sha256(seed_label.encode("utf-8")).hexdigest()[:16], 16)
    rng = random.Random(seed)
    estimates: list[float] = []
    for _ in range(draws):
        sampled = [reps[rng.randrange(len(reps))] for _ in reps]
        successes = sum(counts_by_rep[rep][0] for rep in sampled)
        total = sum(counts_by_rep[rep][1] for rep in sampled)
        estimates.append(successes / total)
    estimates.sort()
    return estimates[int(0.025 * draws)], estimates[int(0.975 * draws) - 1]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write empty table: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def completed_manifests(lane_roots: Iterable[Path]) -> list[tuple[Path, dict[str, Any]]]:
    found: list[tuple[Path, dict[str, Any]]] = []
    for lane_root in lane_roots:
        for path in sorted(lane_root.glob("*/run_manifest.json")):
            manifest = json.loads(path.read_text(encoding="utf-8"))
            if manifest.get("status") != "completed":
                raise ValueError(f"Incomplete shard: {path} ({manifest.get('status')})")
            found.append((path, manifest))
    if not found:
        raise ValueError("No completed surface-sensitivity manifests found")
    return found


def assert_single_contract(manifests: list[tuple[Path, dict[str, Any]]]) -> None:
    fields = [
        ("source_sha256", lambda m: m["source_sha256"]),
        ("model digest", lambda m: m["model"]["config_sha256"]),
        ("base_seed", lambda m: m["base_seed"]),
        ("decoding", lambda m: json.dumps(m["decoding"], sort_keys=True)),
        ("repetitions", lambda m: m["experiment"]["repetitions"]),
        ("games", lambda m: tuple(m["experiment"]["games"])),
    ]
    for label, getter in fields:
        values = {getter(manifest) for _, manifest in manifests}
        if len(values) != 1:
            raise ValueError(f"Mixed {label} across sensitivity shards: {values}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane-root", action="append", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    manifests = completed_manifests(args.lane_root)
    assert_single_contract(manifests)
    variants = [manifest["prompt_variant"]["id"] for _, manifest in manifests]
    if len(set(variants)) != len(variants):
        raise ValueError(f"Prompt variants must occur exactly once, got {variants}")
    if "canonical" not in variants:
        raise ValueError("Paired analysis requires the canonical control")

    turns: list[dict[str, Any]] = []
    races: list[dict[str, Any]] = []
    metadata: dict[str, dict[str, Any]] = {}
    raw_files: list[dict[str, str]] = []
    for manifest_path, manifest in manifests:
        variant = manifest["prompt_variant"]["id"]
        metadata[variant] = manifest["prompt_variant"]
        turn_path = manifest_path.parent / "turns.jsonl"
        race_path = manifest_path.parent / "races.csv"
        shard_turns = read_jsonl(turn_path)
        with race_path.open("r", newline="", encoding="utf-8") as handle:
            shard_races = list(csv.DictReader(handle))
        if len(shard_turns) != int(manifest["n_turns"]):
            raise ValueError(f"Turn count mismatch for {variant}")
        if len(shard_races) != int(manifest["n_races"]):
            raise ValueError(f"Race count mismatch for {variant}")
        for row in shard_turns:
            row["variant"] = variant
            row["family"] = metadata[variant]["family"]
        for row in shard_races:
            row["variant"] = variant
        turns.extend(shard_turns)
        races.extend(shard_races)
        raw_files.extend(
            [
                {"path": str(turn_path), "sha256": sha256_file(turn_path)},
                {"path": str(race_path), "sha256": sha256_file(race_path)},
                {"path": str(manifest_path), "sha256": sha256_file(manifest_path)},
            ]
        )

    for row in turns:
        if int(row["unsafe"]) not in (0, 1):
            raise ValueError(f"Invalid unsafe value: {row['unsafe']!r}")
    parse_failures = sum(int(bool(row["parse_failed"])) for row in turns)

    # Horizon is generated from a seed stream independent of model actions.  Equal
    # lengths across variants therefore audit common random numbers and mechanics.
    horizons: dict[tuple[float, int], set[int]] = defaultdict(set)
    for row in races:
        horizons[(float(row["max_private_risk"]), int(row["rep"]))].add(
            int(row["n_rounds"])
        )
    mismatched_horizons = {key: values for key, values in horizons.items() if len(values) != 1}
    if mismatched_horizons:
        raise ValueError(f"Paired horizons differ across prompt variants: {mismatched_horizons}")

    canonical_first = {
        (float(row["max_private_risk"]), int(row["rep"]), int(row["player_index"])):
        int(row["unsafe"])
        for row in turns
        if row["variant"] == "canonical" and int(row["round"]) == 1
    }
    if not canonical_first:
        raise ValueError("Canonical first-round decisions are missing")

    variant_rows: list[dict[str, Any]] = []
    risk_rows: list[dict[str, Any]] = []
    for variant in sorted(metadata):
        selected = [row for row in turns if row["variant"] == variant]
        first = [row for row in selected if int(row["round"]) == 1]
        flips = 0
        safe_to_unsafe = 0
        unsafe_to_safe = 0
        for row in first:
            key = (
                float(row["max_private_risk"]), int(row["rep"]),
                int(row["player_index"]),
            )
            if key not in canonical_first:
                raise ValueError(f"Unmatched first-round decision: {variant} {key}")
            control = canonical_first[key]
            observed = int(row["unsafe"])
            flips += int(control != observed)
            safe_to_unsafe += int(control == 0 and observed == 1)
            unsafe_to_safe += int(control == 1 and observed == 0)
        unsafe_count = sum(int(row["unsafe"]) for row in selected)
        unsafe_by_rep: dict[int, tuple[int, int]] = {}
        flip_by_rep: dict[int, tuple[int, int]] = {}
        for rep in sorted({int(row["rep"]) for row in selected}):
            rep_rows = [row for row in selected if int(row["rep"]) == rep]
            unsafe_by_rep[rep] = (
                sum(int(row["unsafe"]) for row in rep_rows), len(rep_rows)
            )
            rep_first = [row for row in first if int(row["rep"]) == rep]
            rep_flips = sum(
                int(
                    int(row["unsafe"])
                    != canonical_first[
                        (float(row["max_private_risk"]), rep, int(row["player_index"]))
                    ]
                )
                for row in rep_first
            )
            flip_by_rep[rep] = (rep_flips, len(rep_first))
        unsafe_low, unsafe_high = clustered_rate_ci(
            unsafe_by_rep, seed_label=f"unsafe:{variant}"
        )
        flip_low, flip_high = clustered_rate_ci(
            flip_by_rep, seed_label=f"flip:{variant}"
        )
        variant_rows.append(
            {
                "variant": variant,
                "family": metadata[variant]["family"],
                "interpretation": metadata[variant]["interpretation"],
                "n_decisions": len(selected),
                "unsafe_rate": unsafe_count / len(selected),
                "unsafe_rate_cluster_bootstrap_ci95_low": unsafe_low,
                "unsafe_rate_cluster_bootstrap_ci95_high": unsafe_high,
                "n_first_round": len(first),
                "first_round_flip_rate_vs_canonical": flips / len(first),
                "first_round_flip_cluster_bootstrap_ci95_low": flip_low,
                "first_round_flip_cluster_bootstrap_ci95_high": flip_high,
                "first_round_safe_to_unsafe": safe_to_unsafe,
                "first_round_unsafe_to_safe": unsafe_to_safe,
                "parse_failures": sum(int(bool(row["parse_failed"])) for row in selected),
            }
        )
        risks = sorted({float(row["max_private_risk"]) for row in selected})
        for risk in risks:
            cell = [row for row in selected if float(row["max_private_risk"]) == risk]
            cell_first = [row for row in cell if int(row["round"]) == 1]
            cell_flips = sum(
                int(int(row["unsafe"]) != canonical_first[(risk, int(row["rep"]), int(row["player_index"]))])
                for row in cell_first
            )
            risk_rows.append(
                {
                    "variant": variant,
                    "family": metadata[variant]["family"],
                    "max_private_risk": risk,
                    "n_decisions": len(cell),
                    "unsafe_rate": sum(int(row["unsafe"]) for row in cell) / len(cell),
                    "n_first_round": len(cell_first),
                    "first_round_flip_rate_vs_canonical": cell_flips / len(cell_first),
                }
            )

    output_dir = args.output_dir
    write_csv(output_dir / "variant_summary.csv", variant_rows)
    write_csv(output_dir / "risk_variant_summary.csv", risk_rows)
    canonical_rate = next(row["unsafe_rate"] for row in variant_rows if row["variant"] == "canonical")
    report = {
        "schema_version": "ai-race-surface-sensitivity-analysis-v1",
        "n_variants": len(metadata),
        "n_races": len(races),
        "n_decisions": len(turns),
        "parse_failures": parse_failures,
        "paired_horizon_cells": len(horizons),
        "canonical_unsafe_rate": canonical_rate,
        "unsafe_rate_range": [
            min(row["unsafe_rate"] for row in variant_rows),
            max(row["unsafe_rate"] for row in variant_rows),
        ],
        "max_first_round_flip_rate_vs_canonical": max(
            row["first_round_flip_rate_vs_canonical"] for row in variant_rows
        ),
        "note": (
            "First-round flips compare identical states and matched sampling seeds. "
            "Whole-trajectory unsafe-rate differences include feedback from earlier actions. "
            "Intervals resample repetition blocks, not dependent decisions."
        ),
        "raw_files": raw_files,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "analysis.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
