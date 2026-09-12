"""Fit the existing EGT sensitivity sweep to all five admitted routes.

This is an analysis-only extension.  It reads the already-computed 90-cell EGT
sensitivity table and the canonical nine-route baseline turns; it does not run
new LLM requests or new evolutionary chains.  The older two-route fit remains
untouched because it is the direct comparison used by the existing manuscript
paragraph.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import figdata


INPUT_SWEEP = ROOT / "results" / "open_source" / "egt_reproduction" / "egt_beta_sensitivity.csv"
OUTPUT_DIR = ROOT / "results" / "open_source" / "egt_reproduction"
OUTPUT_FITS = OUTPUT_DIR / "egt_admitted_route_fits.csv"
OUTPUT_SUMMARY = OUTPUT_DIR / "egt_admitted_route_summary.csv"
OUTPUT_JSON = OUTPUT_DIR / "egt_admitted_route_fits.json"

RISKS = (0.1, 0.6, 0.9)
REFERENCE = ("beta_over_Z", 2.0, 0.02)
WEAK_SELECTION_MAX_BETA = 0.03
MIXING_SUSPECT_RANGE = 0.05

LABELS = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "anthropic/claude-opus-5@default": "Claude Opus 5",
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.5-2026-04-23": "GPT-5.5",
    "anthropic/claude-sonnet-5@default": "Claude Sonnet 5",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_sweep() -> dict[tuple[str, float, float, float], dict[str, str]]:
    with INPUT_SWEEP.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    cells = {}
    for row in rows:
        key = (
            row["mutation_rule"],
            float(row["beta"]),
            float(row["mutation"]),
            float(row["max_private_risk"]),
        )
        cells[key] = row
    if len(cells) != 90:
        raise ValueError(f"expected 90 EGT sensitivity rows, found {len(cells)}")
    return cells


def admitted_profiles() -> dict[str, dict[float, dict[str, float]]]:
    admission = figdata.admission()
    admitted = set(admission.loc[admission["admitted_for_gameplay"], "route"])
    if set(LABELS) != admitted:
        raise ValueError(
            "admitted roster changed: "
            f"expected {sorted(LABELS)}, found {sorted(admitted)}"
        )

    turns = figdata.baseline_turns()
    profiles: dict[str, dict[float, dict[str, float]]] = {}
    for route in sorted(admitted):
        subset = turns[turns["model_route"] == route]
        profiles[route] = {}
        for risk in RISKS:
            cell = subset[subset["max_private_risk"] == risk]
            if len(cell) != 186:
                raise ValueError(f"{route} risk {risk} has {len(cell)} decisions, not 186")
            profiles[route][risk] = {
                "unsafe_rate": float(cell["unsafe"].mean()),
                "decisions": int(len(cell)),
            }
    return profiles


def observed_shape(values: dict[float, float]) -> str:
    low, middle, high = (values[risk] for risk in RISKS)
    if low >= 0.85 and high <= 0.10:
        return "near-step switch"
    if low > middle > high and high > 0.10:
        return "graded decline"
    return "other monotone profile"


def cell_profile(
    sweep: dict[tuple[str, float, float, float], dict[str, str]],
    key_prefix: tuple[str, float, float],
) -> tuple[np.ndarray, bool]:
    values = []
    suspect = False
    for risk in RISKS:
        row = sweep[key_prefix + (risk,)]
        values.append(float(row["unsafe_frequency_mean"]))
        suspect = suspect or float(row["unsafe_frequency_between_chain_range"]) > MIXING_SUSPECT_RANGE
    return np.asarray(values, dtype=float), suspect


def fit_row(
    route: str,
    observed: np.ndarray,
    sweep: dict[tuple[str, float, float, float], dict[str, str]],
    mutation_rule: str,
    beta: float,
    mutation: float,
) -> dict[str, object]:
    predicted, suspect = cell_profile(sweep, (mutation_rule, beta, mutation))
    deviation = predicted - observed
    return {
        "route": route,
        "route_label": LABELS[route],
        "mutation_rule": mutation_rule,
        "beta": beta,
        "mutation": mutation,
        "rmse_percentage_points": float(np.sqrt(np.mean(deviation**2)) * 100),
        "max_absolute_deviation_percentage_points": float(np.max(np.abs(deviation)) * 100),
        "profile_mixing_suspect": bool(suspect),
        "well_mixed": bool(not suspect),
        "theory_r0p1": float(predicted[0]),
        "theory_r0p6": float(predicted[1]),
        "theory_r0p9": float(predicted[2]),
        "observed_r0p1": float(observed[0]),
        "observed_r0p6": float(observed[1]),
        "observed_r0p9": float(observed[2]),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    sweep = read_sweep()
    profiles = admitted_profiles()

    fit_rows: list[dict[str, object]] = []
    keys = sorted({(rule, beta, mutation) for rule, beta, mutation, _ in sweep})
    for route, cells in profiles.items():
        observed = np.asarray([cells[risk]["unsafe_rate"] for risk in RISKS], dtype=float)
        for rule, beta, mutation in keys:
            row = fit_row(route, observed, sweep, rule, beta, mutation)
            row["is_reference"] = (rule, beta, mutation) == REFERENCE
            row["is_weak_selection_candidate"] = beta <= WEAK_SELECTION_MAX_BETA
            fit_rows.append(row)

    summary_rows: list[dict[str, object]] = []
    best: dict[str, dict[str, object]] = {}
    for route in sorted(profiles):
        route_rows = [row for row in fit_rows if row["route"] == route]
        reference = next(row for row in route_rows if row["is_reference"])
        well_mixed = [row for row in route_rows if row["well_mixed"]]
        weak = [row for row in well_mixed if row["is_weak_selection_candidate"]]
        if not weak:
            raise ValueError(f"no well-mixed weak-selection cells for {route}")
        best_overall = min(well_mixed, key=lambda row: row["rmse_percentage_points"])
        best_weak = min(weak, key=lambda row: row["rmse_percentage_points"])
        observed = profiles[route]
        summary = {
            "route": route,
            "route_label": LABELS[route],
            "observed_shape": observed_shape({risk: observed[risk]["unsafe_rate"] for risk in RISKS}),
            "observed_r0p1": observed[0.1]["unsafe_rate"],
            "observed_r0p6": observed[0.6]["unsafe_rate"],
            "observed_r0p9": observed[0.9]["unsafe_rate"],
            "reference_rmse_percentage_points": reference["rmse_percentage_points"],
            "reference_max_absolute_deviation_percentage_points": reference[
                "max_absolute_deviation_percentage_points"
            ],
            "best_well_mixed_rule": best_overall["mutation_rule"],
            "best_well_mixed_beta": best_overall["beta"],
            "best_well_mixed_mutation": best_overall["mutation"],
            "best_well_mixed_rmse_percentage_points": best_overall["rmse_percentage_points"],
            "best_weak_rule": best_weak["mutation_rule"],
            "best_weak_beta": best_weak["beta"],
            "best_weak_mutation": best_weak["mutation"],
            "best_weak_rmse_percentage_points": best_weak["rmse_percentage_points"],
            "weak_closer_than_reference": best_weak["rmse_percentage_points"] < reference[
                "rmse_percentage_points"
            ],
            "best_weak_max_absolute_deviation_percentage_points": best_weak[
                "max_absolute_deviation_percentage_points"
            ],
        }
        summary_rows.append(summary)
        best[route] = summary

    write_csv(OUTPUT_FITS, fit_rows)
    write_csv(OUTPUT_SUMMARY, summary_rows)

    payload = {
        "analysis": "analysis-only fit of the existing EGT sensitivity sweep to all five admitted frontier routes",
        "evidence_boundary": (
            "No new LLM requests or evolutionary chains were run. Theory values are read from the existing "
            "90-cell seeded-chain sensitivity artifact; route values are decision-weighted rates from the "
            "canonical nine-route neutral baseline, restricted to the five routes admitted by the gameplay gate."
        ),
        "inputs": {
            "egt_sensitivity_csv": {"path": str(INPUT_SWEEP.relative_to(ROOT)), "sha256": sha256(INPUT_SWEEP)},
            "baseline_loader": "scripts/figdata.py::baseline_turns",
            "admission_csv": str(figdata.ADMISSION_CSV.relative_to(ROOT)),
        },
        "fit_definition": {
            "distance": "RMSE in percentage points over the three matched risk levels 0.1, 0.6 and 0.9",
            "reference_cell": {"mutation_rule": REFERENCE[0], "beta": REFERENCE[1], "mutation": REFERENCE[2]},
            "weak_selection_definition": f"beta <= {WEAK_SELECTION_MAX_BETA} and not mixing-suspect",
            "mixing_suspect_threshold": MIXING_SUSPECT_RANGE,
            "primary_selection": "best well-mixed cell; ties are resolved by first sorted cell",
        },
        "route_count": len(summary_rows),
        "route_labels": LABELS,
        "summaries": best,
        "counts": {
            "sweep_cells": len(keys),
            "fit_rows": len(fit_rows),
            "routes_closer_to_weak_than_reference": sum(
                bool(row["weak_closer_than_reference"]) for row in summary_rows
            ),
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "outputs": [str(path.relative_to(ROOT)) for path in (OUTPUT_FITS, OUTPUT_SUMMARY, OUTPUT_JSON)],
    }
    OUTPUT_JSON.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    print(json.dumps({"summary": summary_rows, "outputs": payload["outputs"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
