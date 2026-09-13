"""Audit the existing scripted-opponent evidence after excluding one legacy cell.

This is a read-only sensitivity audit.  It does not call a model, contact
Kaggle, inspect account credentials, or rewrite a raw or derived result.  The
legacy-contract fact is recorded in
``docs/scripted-opponent-opus-contract-harmonization-amendment-2026-09-13.md``;
this script deliberately does not infer decoding metadata from the rate CSVs.

The sensitivity is defined before reading the values:

* exclude Claude Opus 5, Always Safe, risk 0.6;
* remove the one AU-minus-AS rival contrast that needs that AS arm; and
* remove the one route-by-risk ordering cell that needs all four arms.

The script reports the full-grid and excluded-cell denominators, positive
rival-contrast counts, and weak/strict AS <= CS <= CAS <= AU ordering counts.
It fails closed if the five expected tables are incomplete, duplicated, or
contain malformed cell metadata.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DERIVED = ROOT / "results" / "derived" / "scripted_opponent_campaign"
RIVAL_CONTRASTS = DERIVED / "risk_versus_rival.json"

# The root table is the original campaign route; the other tables use the
# route tags emitted by scripts/analyze_scripted_opponent.py.
ROUTE_TABLES = {
    "google/gemini-3-flash-preview": DERIVED / "scripted_opponent_rates.csv",
    "anthropic/claude-opus-5@default": DERIVED / "claude-opus-5-default" / "scripted_opponent_rates.csv",
    "anthropic/claude-sonnet-5@default": DERIVED / "claude-sonnet-5-default" / "scripted_opponent_rates.csv",
    "openai/gpt-5.4-2026-03-05": DERIVED / "gpt-5.4-2026-03-05" / "scripted_opponent_rates.csv",
    "openai/gpt-5.5-2026-04-23": DERIVED / "gpt-5.5-2026-04-23" / "scripted_opponent_rates.csv",
}

STRATEGIES = ("AS", "CS", "CAS", "AU")
RISKS = (0.1, 0.6, 0.9)
EXPECTED_ROWS = len(STRATEGIES) * len(RISKS)
EXPECTED_RACES = 10
EXPECTED_DECISIONS = 93
EPS = 1e-12

LEGACY_ROUTE = "anthropic/claude-opus-5@default"
LEGACY_STRATEGY = "AS"
LEGACY_RISK = 0.6

REQUIRED_COLUMNS = {
    "opponent_strategy",
    "max_private_risk",
    "n_races",
    "n_route_decisions",
    "unsafe_rate",
    "ci95_low",
    "ci95_high",
    "inference_supported",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def as_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"expected True/False, got {value!r}")


def load_tables() -> tuple[dict[str, dict[tuple[str, float], dict]], dict[str, str]]:
    tables: dict[str, dict[tuple[str, float], dict]] = {}
    hashes: dict[str, str] = {}

    for route, path in ROUTE_TABLES.items():
        if not path.is_file():
            raise SystemExit(f"REFUSED: missing derived table: {path.relative_to(ROOT)}")
        hashes[route] = sha256(path)
        with path.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            columns = set(reader.fieldnames or ())
            missing = REQUIRED_COLUMNS - columns
            if missing:
                raise SystemExit(
                    f"REFUSED: {path.relative_to(ROOT)} missing columns {sorted(missing)}"
                )
            rows = list(reader)

        if len(rows) != EXPECTED_ROWS:
            raise SystemExit(
                f"REFUSED: {route} has {len(rows)} rows; expected {EXPECTED_ROWS}"
            )

        cells: dict[tuple[str, float], dict] = {}
        for row in rows:
            try:
                strategy = row["opponent_strategy"]
                risk = float(row["max_private_risk"])
                parsed = {
                    "strategy": strategy,
                    "risk": risk,
                    "n_races": int(row["n_races"]),
                    "n_route_decisions": int(row["n_route_decisions"]),
                    "unsafe_rate": float(row["unsafe_rate"]),
                    "ci95_low": float(row["ci95_low"]),
                    "ci95_high": float(row["ci95_high"]),
                    "inference_supported": as_bool(row["inference_supported"]),
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise SystemExit(
                    f"REFUSED: malformed row in {path.relative_to(ROOT)}: {exc}"
                ) from exc

            if strategy not in STRATEGIES or risk not in RISKS:
                raise SystemExit(f"REFUSED: unexpected cell in {route}: {strategy}@{risk}")
            key = (strategy, risk)
            if key in cells:
                raise SystemExit(f"REFUSED: duplicate cell in {route}: {strategy}@{risk}")
            if parsed["n_races"] != EXPECTED_RACES:
                raise SystemExit(f"REFUSED: {route} {strategy}@{risk} has wrong race count")
            if parsed["n_route_decisions"] != EXPECTED_DECISIONS:
                raise SystemExit(f"REFUSED: {route} {strategy}@{risk} has wrong decision count")
            if not parsed["inference_supported"]:
                raise SystemExit(f"REFUSED: {route} {strategy}@{risk} is not inference-supported")
            if not all(math.isfinite(parsed[name]) for name in ("unsafe_rate", "ci95_low", "ci95_high")):
                raise SystemExit(f"REFUSED: non-finite rate or interval in {route} {strategy}@{risk}")
            if not 0.0 <= parsed["ci95_low"] <= parsed["unsafe_rate"] <= parsed["ci95_high"] <= 1.0:
                raise SystemExit(f"REFUSED: invalid rate interval in {route} {strategy}@{risk}")
            cells[key] = parsed

        expected = {(strategy, risk) for strategy in STRATEGIES for risk in RISKS}
        if set(cells) != expected:
            raise SystemExit(f"REFUSED: {route} does not contain the expected 12-cell grid")
        tables[route] = cells

    return tables, hashes


def load_paired_rival_contrasts() -> tuple[list[dict], str]:
    """Load the published paired AU-minus-AS contrasts, without recomputing them.

    The CSV rate difference is a pooled descriptive difference.  The campaign's
    rival estimand is instead differenced inside each repetition, so this audit
    reads the already-derived paired contrast table and checks its shape/sign
    against the CSV grid.
    """
    if not RIVAL_CONTRASTS.is_file():
        raise SystemExit(f"REFUSED: missing derived contrast table: {RIVAL_CONTRASTS.relative_to(ROOT)}")
    try:
        payload = json.loads(RIVAL_CONTRASTS.read_text(encoding="utf-8"))
        rows = payload["rival_contrast"]["cells"]
    except (OSError, KeyError, TypeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"REFUSED: malformed derived contrast table: {exc}") from exc

    expected = {(route, risk) for route in ROUTE_TABLES for risk in RISKS}
    seen = set()
    for row in rows:
        try:
            key = (str(row["model_route"]), float(row["max_private_risk"]))
            difference = float(row["mean_difference"])
            n_blocks = int(row["n_blocks"])
            pairing_verified = bool(row["pairing_verified"])
        except (KeyError, TypeError, ValueError) as exc:
            raise SystemExit(f"REFUSED: malformed paired contrast row: {exc}") from exc
        if key in seen:
            raise SystemExit(f"REFUSED: duplicate paired contrast: {key}")
        if key not in expected:
            raise SystemExit(f"REFUSED: unexpected paired contrast: {key}")
        if not math.isfinite(difference) or n_blocks != EXPECTED_RACES or not pairing_verified:
            raise SystemExit(f"REFUSED: paired contrast failed integrity checks: {key}")
        seen.add(key)
    if seen != expected:
        raise SystemExit(
            f"REFUSED: paired contrast table has {len(seen)} valid cells; expected {len(expected)}"
        )
    return rows, sha256(RIVAL_CONTRASTS)


def weakly_ordered(cells: dict[tuple[str, float], dict], risk: float) -> bool:
    values = [cells[(strategy, risk)]["unsafe_rate"] for strategy in STRATEGIES]
    return all(left <= right + EPS for left, right in zip(values, values[1:]))


def strictly_ordered(cells: dict[tuple[str, float], dict], risk: float) -> bool:
    values = [cells[(strategy, risk)]["unsafe_rate"] for strategy in STRATEGIES]
    return all(left < right - EPS for left, right in zip(values, values[1:]))


def audit(
    tables: dict[str, dict[tuple[str, float], dict]],
    hashes: dict[str, str],
    paired_rival_contrasts: list[dict],
) -> dict:
    if LEGACY_ROUTE not in tables:
        raise SystemExit(f"REFUSED: legacy route is not present: {LEGACY_ROUTE}")
    legacy_key = (LEGACY_STRATEGY, LEGACY_RISK)
    if legacy_key not in tables[LEGACY_ROUTE]:
        raise SystemExit("REFUSED: declared legacy cell is absent from the derived table")

    route_risk_cells = [
        (route, risk)
        for route in ROUTE_TABLES
        for risk in RISKS
    ]
    weak_cells = []
    strict_cells = []
    csv_signs = {}
    for route, risk in route_risk_cells:
        cells = tables[route]
        csv_signs[(route, risk)] = cells[("AU", risk)]["unsafe_rate"] - cells[("AS", risk)]["unsafe_rate"]
        weak_cells.append({"route": route, "risk": risk, "pass": weakly_ordered(cells, risk)})
        strict_cells.append({"route": route, "risk": risk, "pass": strictly_ordered(cells, risk)})

    rival_contrasts = []
    for row in paired_rival_contrasts:
        route = str(row["model_route"])
        risk = float(row["max_private_risk"])
        difference = float(row["mean_difference"])
        if difference * csv_signs[(route, risk)] < -EPS:
            raise SystemExit(f"REFUSED: paired contrast sign disagrees with CSV rates: {route}@{risk}")
        rival_contrasts.append({"route": route, "risk": risk, "difference": difference})

    excluded_rival = [
        row for row in rival_contrasts
        if row["route"] == LEGACY_ROUTE and row["risk"] == LEGACY_RISK
    ]
    excluded_ordering = [
        row for row in weak_cells
        if row["route"] == LEGACY_ROUTE and row["risk"] == LEGACY_RISK
    ]
    if len(excluded_rival) != 1 or len(excluded_ordering) != 1:
        raise SystemExit("REFUSED: exclusion did not identify exactly one rival and ordering cell")

    retained_rival = [row for row in rival_contrasts if row not in excluded_rival]
    retained_weak = [row for row in weak_cells if row not in excluded_ordering]
    retained_strict = [row for row in strict_cells if not (
        row["route"] == LEGACY_ROUTE and row["risk"] == LEGACY_RISK
    )]

    return {
        "scope": {
            "input_type": "read-only derived scripted-opponent rate and paired-contrast tables",
            "model_runs": 0,
            "kaggle_runs": 0,
            "raw_or_derived_replacements": 0,
            "routes": len(ROUTE_TABLES),
            "route_risk_cells": len(route_risk_cells),
            "strategy_risk_cells": len(ROUTE_TABLES) * EXPECTED_ROWS,
            "paired_rival_contrast_cells": len(rival_contrasts),
        },
        "excluded_cell": {
            "route": LEGACY_ROUTE,
            "strategy": LEGACY_STRATEGY,
            "risk": LEGACY_RISK,
            "strategy_risk_cells_removed": 1,
            "route_risk_ordering_cells_removed": len(excluded_ordering),
            "au_minus_as_contrasts_removed": len(excluded_rival),
            "note": "Legacy decoding is recorded in the amendment; the derived rate CSV does not encode decoding metadata.",
        },
        "rival_contrast": {
            "definition": "AU minus AS unsafe-rate contrast at each fixed route and risk",
            "full": {
                "denominator": len(rival_contrasts),
                "positive": sum(row["difference"] > EPS for row in rival_contrasts),
                "non_positive": sum(row["difference"] <= EPS for row in rival_contrasts),
            },
            "excluding_legacy_cell": {
                "denominator": len(retained_rival),
                "positive": sum(row["difference"] > EPS for row in retained_rival),
                "non_positive": sum(row["difference"] <= EPS for row in retained_rival),
            },
            "excluded_difference": excluded_rival[0]["difference"],
        },
        "weak_ordering": {
            "definition": "AS <= CS <= CAS <= AU by unsafe rate at each fixed route and risk",
            "full": {
                "denominator": len(weak_cells),
                "pass": sum(row["pass"] for row in weak_cells),
                "fail": sum(not row["pass"] for row in weak_cells),
            },
            "excluding_legacy_cell": {
                "denominator": len(retained_weak),
                "pass": sum(row["pass"] for row in retained_weak),
                "fail": sum(not row["pass"] for row in retained_weak),
            },
            "excluded_cell_passed": excluded_ordering[0]["pass"],
        },
        "strict_ordering": {
            "definition": "AS < CS < CAS < AU by unsafe rate at each fixed route and risk",
            "full": {
                "denominator": len(strict_cells),
                "pass": sum(row["pass"] for row in strict_cells),
                "fail": sum(not row["pass"] for row in strict_cells),
            },
            "excluding_legacy_cell": {
                "denominator": len(retained_strict),
                "pass": sum(row["pass"] for row in retained_strict),
                "fail": sum(not row["pass"] for row in retained_strict),
            },
        },
        "input_sha256": {**hashes, "paired_rival_contrasts": sha256(RIVAL_CONTRASTS)},
    }


def print_report(result: dict) -> None:
    scope = result["scope"]
    excluded = result["excluded_cell"]
    rival = result["rival_contrast"]
    weak = result["weak_ordering"]
    strict = result["strict_ordering"]
    print("READ-ONLY OPUS LEGACY-CELL SENSITIVITY AUDIT")
    print(f"routes={scope['routes']} route_risk_cells={scope['route_risk_cells']} strategy_risk_cells={scope['strategy_risk_cells']}")
    print(f"model_runs={scope['model_runs']} kaggle_runs={scope['kaggle_runs']} replacements={scope['raw_or_derived_replacements']}")
    print(
        "excluded="
        f"{excluded['route']} {excluded['strategy']}@{excluded['risk']} "
        f"strategy_risk_cells={excluded['strategy_risk_cells_removed']} "
        f"route_risk_ordering_cells={excluded['route_risk_ordering_cells_removed']} "
        f"au_minus_as_contrasts={excluded['au_minus_as_contrasts_removed']}"
    )
    print(
        "rival AU-AS positive: "
        f"full {rival['full']['positive']}/{rival['full']['denominator']} "
        f"({rival['full']['non_positive']} non-positive); "
        f"excluding legacy {rival['excluding_legacy_cell']['positive']}/"
        f"{rival['excluding_legacy_cell']['denominator']} "
        f"({rival['excluding_legacy_cell']['non_positive']} non-positive)"
    )
    print(
        "weak ordering AS<=CS<=CAS<=AU: "
        f"full {weak['full']['pass']}/{weak['full']['denominator']} "
        f"({weak['full']['fail']} fail); "
        f"excluding legacy {weak['excluding_legacy_cell']['pass']}/"
        f"{weak['excluding_legacy_cell']['denominator']} "
        f"({weak['excluding_legacy_cell']['fail']} fail)"
    )
    print(
        "strict ordering AS<CS<CAS<AU: "
        f"full {strict['full']['pass']}/{strict['full']['denominator']} "
        f"({strict['full']['fail']} fail); "
        f"excluding legacy {strict['excluding_legacy_cell']['pass']}/"
        f"{strict['excluding_legacy_cell']['denominator']} "
        f"({strict['excluding_legacy_cell']['fail']} fail)"
    )
    print(f"excluded AU-AS difference={100 * rival['excluded_difference']:.6f} percentage points")
    print("input_sha256:")
    for route, digest in result["input_sha256"].items():
        print(f"  {route}: {digest}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="print the machine-readable audit result")
    args = parser.parse_args()
    tables, hashes = load_tables()
    paired_rival_contrasts, contrast_hash = load_paired_rival_contrasts()
    hashes["paired_rival_contrasts"] = contrast_hash
    result = audit(tables, hashes, paired_rival_contrasts)
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
    else:
        print_report(result)


if __name__ == "__main__":
    main()
