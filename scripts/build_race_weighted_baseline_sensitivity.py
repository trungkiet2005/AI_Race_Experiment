"""Package the existing baseline's race-weighted sensitivity analysis.

This script does not collect or resample new model output.  The baseline
analysis already computes decision-weighted rates and equal-weight-per-race
cluster-bootstrap intervals; this command exposes those fields in a small,
auditable artifact for the supplement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import date
from pathlib import Path


EXPECTED_ADMITTED = {
    "google/gemini-3-flash-preview",
    "anthropic/claude-opus-5@default",
    "openai/gpt-5.4-2026-03-05",
    "openai/gpt-5.5-2026-04-23",
    "anthropic/claude-sonnet-5@default",
}
RISKS = ("0p1", "0p6", "0p9")


def finite_float(row: dict[str, str], key: str) -> float:
    value = row.get(key, "")
    if value in {"", "None", "nan", "NaN"}:
        raise ValueError(f"missing numeric field {key!r} for {row.get('route')}")
    number = float(value)
    if number != number or number in {float("inf"), float("-inf")}:
        raise ValueError(f"non-finite numeric field {key!r} for {row.get('route')}")
    return number


def build(input_path: Path) -> dict:
    source_bytes = input_path.read_bytes()
    with input_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    admitted = [row for row in rows if row.get("admitted_for_gameplay") == "True"]
    routes = {row.get("route") for row in admitted}
    if routes != EXPECTED_ADMITTED:
        raise ValueError(
            "baseline admission roster drifted: "
            f"expected {sorted(EXPECTED_ADMITTED)}, got {sorted(routes)}"
        )
    if len(admitted) != len(EXPECTED_ADMITTED):
        raise ValueError("expected one baseline row per admitted route")

    records = []
    for row in admitted:
        rates = {}
        for risk in RISKS:
            rates[risk] = {
                "decision_weighted_rate_pp": finite_float(
                    row, f"unsafe_rate_risk_{risk}"
                ) * 100.0,
                "race_weighted_rate_pp": finite_float(
                    row, f"unsafe_rate_race_mean_risk_{risk}"
                ) * 100.0,
                "race_weighted_ci_low_pp": finite_float(
                    row, f"unsafe_race_mean_ci_low_risk_{risk}"
                ) * 100.0,
                "race_weighted_ci_high_pp": finite_float(
                    row, f"unsafe_race_mean_ci_high_risk_{risk}"
                ) * 100.0,
                "n_races": int(row[f"n_races_risk_{risk}"]),
                "n_decisions": int(row[f"n_decisions_risk_{risk}"]),
            }
        records.append(
            {
                "route": row["route"],
                "short_name": row["short_name"],
                "rates": rates,
                "decision_weighted_response_pp": finite_float(
                    row, "risk_response_pp"
                ),
                "decision_weighted_response_ci_low_pp": finite_float(
                    row, "risk_response_ci_low_pp"
                ),
                "decision_weighted_response_ci_high_pp": finite_float(
                    row, "risk_response_ci_high_pp"
                ),
                "race_weighted_response_pp": finite_float(
                    row, "risk_response_race_mean_pp"
                ),
                "race_weighted_response_ci_low_pp": finite_float(
                    row, "risk_response_race_mean_ci_low_pp"
                ),
                "race_weighted_response_ci_high_pp": finite_float(
                    row, "risk_response_race_mean_ci_high_pp"
                ),
            }
        )

    return {
        "schema_version": "ai-race-baseline-race-weighted-sensitivity-v1",
        "generated": str(date.today()),
        "evidence_class": "diagnostic",
        "source_csv": str(input_path).replace("\\", "/"),
        "source_sha256": hashlib.sha256(source_bytes).hexdigest(),
        "estimand": (
            "Endpoint-level Unsafe rates, reported both with equal weight per "
            "decision and equal weight per race; race intervals are percentile "
            "cluster-bootstrap ranges with both seats of a race resampled together."
        ),
        "uncertainty": {
            "method": "percentile bootstrap over game_id race clusters",
            "n_resamples": 5000,
            "seed": 20260909,
            "races_per_risk_cell": 10,
            "independent_unit": "race",
        },
        "rows": records,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(
            "results/frontier/baseline_campaign_v6/derived/"
            "audit_versus_behaviour.csv"
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/derived/baseline_race_weighted_sensitivity.json"),
    )
    args = parser.parse_args()
    artifact = build(args.input)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {args.output} ({len(artifact['rows'])} admitted routes)")


if __name__ == "__main__":
    main()
