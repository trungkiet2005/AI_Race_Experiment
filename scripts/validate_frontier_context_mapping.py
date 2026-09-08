"""Validate and summarize downloaded frontier context/mapping runs.

The validator fails closed. It accepts a route only when the completed run has
all 120 expected cells, exactly ten races per cell, no parse failures, and
matching race/turn counts. The derived CSV is generated from raw ``races.csv``
and is the only input intended for a manuscript table or figure.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


CONTEXTS = ("technology_race", "abstract_game")
MAPPINGS = ("P_SAFE_Q_UNSAFE", "Q_SAFE_P_UNSAFE")
RISKS = (0.1, 0.6, 0.9)
EXPECTED_REPETITIONS = 10
EXPECTED_RACES = len(CONTEXTS) * len(MAPPINGS) * len(RISKS) * EXPECTED_REPETITIONS


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def find_route(root: Path) -> Path:
    candidates = sorted(root.glob("**/results/ai_race_context_mapping/*/summary.json"))
    if not candidates:
        candidates = sorted(root.glob("**/ai_race_context_mapping/*/summary.json"))
    if not candidates:
        raise FileNotFoundError(f"no context-mapping summary found below {root}")
    if len(candidates) != 1:
        raise ValueError(f"expected one route summary below {root}, found {len(candidates)}")
    return candidates[0].parent


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(route: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    summary = load_json(route / "summary.json")
    manifest = load_json(route / "run_manifest.json")
    with (route / "races.csv").open(encoding="utf-8", newline="") as handle:
        races = list(csv.DictReader(handle))
    with (route / "turns.jsonl").open(encoding="utf-8") as handle:
        turns = [json.loads(line) for line in handle if line.strip()]

    keys = [
        (row["context"], row["mapping"], round(float(row["max_private_risk"]), 1))
        for row in races
    ]
    expected_keys = [
        (context, mapping, risk)
        for context in CONTEXTS
        for mapping in MAPPINGS
        for risk in RISKS
    ]
    counts = Counter(keys)
    missing = [key for key in expected_keys if counts[key] != EXPECTED_REPETITIONS]
    duplicate_or_wrong = [key for key, count in counts.items() if key not in expected_keys or count != EXPECTED_REPETITIONS]
    parse_failures = sum(int(bool(row.get("parse_failed"))) for row in turns)
    errors = []
    if summary.get("status") != "completed" or manifest.get("status") != "completed":
        errors.append("summary or manifest is not completed")
    if len(races) != EXPECTED_RACES:
        errors.append(f"expected {EXPECTED_RACES} races, found {len(races)}")
    if len(turns) != int(summary.get("n_turns", -1)):
        errors.append("turn count does not match summary")
    if int(summary.get("n_races", -1)) != len(races):
        errors.append("race count does not match summary")
    if parse_failures or int(summary.get("parse_failures", -1)) != 0:
        errors.append("parse failures are non-zero")
    errors.extend(f"missing or incomplete cell: {key}" for key in missing)
    errors.extend(f"unexpected or duplicate cell: {key}" for key in duplicate_or_wrong if key not in missing)

    derived: list[dict[str, object]] = []
    by_cell: dict[tuple[str, str, float], list[dict[str, str]]] = {}
    for row in races:
        key = (row["context"], row["mapping"], round(float(row["max_private_risk"]), 1))
        by_cell.setdefault(key, []).append(row)
    for (context, mapping, risk), rows in sorted(by_cell.items()):
        decisions = sum(2 * int(row["n_rounds"]) for row in rows)
        unsafe = sum(int(row["unsafe_actions"]) for row in rows)
        derived.append(
            {
                "model_route": summary.get("model_route"),
                "context": context,
                "mapping": mapping,
                "max_private_risk": risk,
                "n_races": len(rows),
                "n_decisions": decisions,
                "unsafe_actions": unsafe,
                "unsafe_rate": unsafe / decisions if decisions else None,
                "mean_rounds": sum(int(row["n_rounds"]) for row in rows) / len(rows),
            }
        )
    audit = {
        "schema_version": "frontier-context-mapping-validation-v1",
        "status": "passed" if not errors else "failed",
        "route": str(route),
        "model_route": summary.get("model_route"),
        "protocol_id": summary.get("protocol_id"),
        "summary": summary,
        "manifest_sha256": sha256(route / "run_manifest.json"),
        "races_sha256": sha256(route / "races.csv"),
        "turns_sha256": sha256(route / "turns.jsonl"),
        "n_races": len(races),
        "n_turns": len(turns),
        "n_cells": len(by_cell),
        "parse_failures": parse_failures,
        "errors": errors,
    }
    if errors:
        raise ValueError(json.dumps(audit, indent=2))
    return audit, derived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path, help="Downloaded task output root")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    route = find_route(args.root.resolve())
    audit, rows = validate(route)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / f"{route.name}_validation.json").write_text(json.dumps(audit, indent=2) + "\n", encoding="utf-8")
    with (args.output / f"{route.name}_cells.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "passed", "route": str(route), "cells": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
