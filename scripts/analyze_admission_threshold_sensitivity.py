"""Recompute the endpoint-admission verdict under alternative gate thresholds.

The declared gate is conjunctive: a route is admitted only when its overall
semantic accuracy, its state-reconstruction accuracy and its terminal-scoring
accuracy all clear their own threshold. This script asks what the admitted set
would have been had those three numbers been declared elsewhere.

Two properties of the instrument shape the answer and are computed rather than
assumed.

  1. Each gated quantity is scored over a small, fixed number of probe answers,
     so it can only take values on a discrete grid. Overall accuracy rests on
     60 retained answers per route and moves in steps of one sixtieth;
     state reconstruction and terminal scoring rest on 15 each and move in steps
     of one fifteenth. A threshold sweep finer than that grid reports
     differences the battery cannot resolve, so every sweep here runs on the
     grid and the report names the achievable levels.

  2. Because the three conditions are conjunctive, a route can be refused by a
     condition other than the one being swept. Every row of every output
     therefore carries which condition binds, not only the verdict.

Route labels come from an explicit mapping that is asserted to cover exactly the
routes present in the campaign. No label is derived by substring search: `mini`
is a substring of `gemini`, so a substring rule silently mislabels two Gemini
routes, and this file refuses to contain such a rule.

Inputs
  results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv
      per-route, per-domain accuracies and the declared verdicts
  results/frontier/admission_campaign_v6/derived/admission_campaign_v6.json
      per-domain probe-answer counts, which give each grid its denominator, and
      the declared thresholds

Outputs
  results/derived/admission_threshold_sensitivity/
      admission_threshold_sensitivity.json
      one_at_a_time_bands.csv
      binding_conditions.csv
      report.md

Usage
  python scripts/analyze_admission_threshold_sensitivity.py
"""

from __future__ import annotations

import csv
import datetime as dt
import json
from fractions import Fraction
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN_DERIVED = REPO_ROOT / "results" / "frontier" / "admission_campaign_v6" / "derived"
CSV_PATH = CAMPAIGN_DERIVED / "admission_campaign_v6.csv"
JSON_PATH = CAMPAIGN_DERIVED / "admission_campaign_v6.json"
OUT_DIR = REPO_ROOT / "results" / "derived" / "admission_threshold_sensitivity"

TOLERANCE = 1e-9

# The three conjunctive gate conditions, in the order the protocol declares
# them, each paired with the campaign domain whose accuracy it reads.
GATES = ("overall_accuracy", "state_reconstruction", "terminal_scoring")

# Reader-facing route labels. Asserted below to cover exactly the campaign's
# routes, which is what makes a substring rule unnecessary.
ROUTE_LABEL = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "anthropic/claude-opus-5@default": "Claude Opus 5",
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.5-2026-04-23": "GPT-5.5",
    "anthropic/claude-sonnet-5@default": "Claude Sonnet 5",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "openai/gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "openai/gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
}

ESTIMAND = (
    "Which of the nine audited endpoint routes the conjunctive admission gate "
    "would have admitted, recomputed over the discrete grid of thresholds the "
    "frozen probe battery can actually resolve. This is a re-reading of one "
    "completed campaign under alternative entry conditions, not a new "
    "measurement and not an estimate of any route's comprehension."
)


class RefusedError(RuntimeError):
    """Raised when an input does not satisfy a structural check."""


def _load() -> tuple[list[dict], dict[str, int], dict[str, float]]:
    """Return per-route records, per-gate denominators, and declared thresholds."""
    if not CSV_PATH.exists() or not JSON_PATH.exists():
        raise RefusedError(f"campaign derived tables missing under {CAMPAIGN_DERIVED}")

    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    declared = {
        "overall_accuracy": float(payload["thresholds_applied"]["overall_accuracy_min"]),
        "state_reconstruction": float(
            payload["thresholds_applied"]["state_reconstruction_accuracy_min"]
        ),
        "terminal_scoring": float(
            payload["thresholds_applied"]["terminal_scoring_accuracy_min"]
        ),
    }
    if not payload["thresholds_applied"].get("expected_payoff_is_diagnostic_only"):
        raise RefusedError("expected payoff is no longer diagnostic only; the gate has changed")

    counts = {r["route"]: r["by_domain_counts"] for r in payload["routes"]}
    n_rows = {r["route"]: int(r["n_rows"]) for r in payload["routes"]}

    records: list[dict] = []
    with CSV_PATH.open(encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            route = row["route"]
            if route not in ROUTE_LABEL:
                raise RefusedError(f"no reader-facing label declared for route {route}")
            if route not in counts:
                raise RefusedError(f"route {route} is in the table but not in the campaign record")
            values = {
                "overall_accuracy": float(row["overall_accuracy"]),
                "state_reconstruction": float(row["accuracy_state_reconstruction"]),
                "terminal_scoring": float(row["accuracy_terminal_scoring"]),
            }
            denominators = {
                "overall_accuracy": n_rows[route],
                "state_reconstruction": int(counts[route]["state_reconstruction"]["rows"]),
                "terminal_scoring": int(counts[route]["terminal_scoring"]["rows"]),
            }
            scored = {}
            for gate in GATES:
                exact = values[gate] * denominators[gate]
                nearest = round(exact)
                if abs(exact - nearest) > 1e-6:
                    raise RefusedError(
                        f"{route} {gate} accuracy {values[gate]} is not a whole number of "
                        f"answers out of {denominators[gate]}"
                    )
                scored[gate] = nearest
            records.append(
                {
                    "route": route,
                    "label": ROUTE_LABEL[route],
                    "declared_admitted": row["admitted_for_gameplay"] == "True",
                    "accuracy": values,
                    "correct": scored,
                    "rows": denominators,
                }
            )

    if len(records) != len(ROUTE_LABEL):
        raise RefusedError(f"expected {len(ROUTE_LABEL)} routes, found {len(records)}")

    # One denominator per gate across the whole campaign, which is what lets a
    # single grid be swept for all nine routes at once.
    shared: dict[str, int] = {}
    for gate in GATES:
        seen = {rec["rows"][gate] for rec in records}
        if len(seen) != 1:
            raise RefusedError(f"{gate} is scored over different denominators across routes: {seen}")
        shared[gate] = seen.pop()

    return records, shared, declared


def _admits(records: list[dict], thresholds: dict[str, float]) -> list[str]:
    """Routes admitted under `thresholds`, in campaign order."""
    out = []
    for rec in records:
        if all(rec["accuracy"][gate] + TOLERANCE >= thresholds[gate] for gate in GATES):
            out.append(rec["label"])
    return out


def _binding(rec: dict, thresholds: dict[str, float]) -> list[str]:
    """The gate conditions that refuse this route under `thresholds`."""
    return [gate for gate in GATES if rec["accuracy"][gate] + TOLERANCE < thresholds[gate]]


def _grid(denominator: int) -> list[Fraction]:
    """Every accuracy a route can score on a domain of this size."""
    return [Fraction(k, denominator) for k in range(denominator + 1)]


def _bands(records: list[dict], gate: str, declared: dict[str, float],
           denominator: int) -> list[dict]:
    """Sweep one gate over its achievable grid, holding the other two declared.

    A threshold that is not itself achievable still partitions the routes, and
    the partition changes only when the threshold crosses an achievable score.
    Each band below is therefore a half-open interval of thresholds: everything
    above the previous achievable score, up to and including this one.
    """
    levels = _grid(denominator)
    bands: list[dict] = []
    for index, level in enumerate(levels):
        thresholds = dict(declared)
        thresholds[gate] = float(level)
        admitted = _admits(records, thresholds)
        lower_exclusive = float(levels[index - 1]) if index else None
        if bands and bands[-1]["admitted"] == admitted:
            bands[-1]["threshold_at_most_pct"] = round(100 * float(level), 4)
            continue
        bands.append(
            {
                "gate": gate,
                "threshold_above_pct": (
                    None if lower_exclusive is None else round(100 * lower_exclusive, 4)
                ),
                "threshold_at_most_pct": round(100 * float(level), 4),
                "admitted": admitted,
                "n_admitted": len(admitted),
            }
        )
    # Which condition refuses a route can change inside a band even though the
    # verdict does not, so this is recorded at the strictest threshold the band
    # covers and the column name says so.
    for band in bands:
        band["binding_for_refused_at_band_top"] = {
            rec["label"]: _binding(rec, {**declared, gate: band["threshold_at_most_pct"] / 100})
            for rec in records
            if rec["label"] not in band["admitted"]
        }
    return bands


def _binding_by_level(records: list[dict], gate: str, declared: dict[str, float],
                      denominator: int) -> list[dict]:
    """Verdict and binding conditions for every route at every achievable level."""
    rows = []
    for level in _grid(denominator):
        thresholds = {**declared, gate: float(level)}
        for rec in records:
            binding = _binding(rec, thresholds)
            rows.append(
                {
                    "gate_swept": gate,
                    "threshold_pct": round(100 * float(level), 4),
                    "label": rec["label"],
                    "admitted": not binding,
                    "binding_conditions": binding,
                }
            )
    return rows


def _maximal_box(records: list[dict], declared: dict[str, float],
                 denominators: dict[str, int]) -> dict:
    """Largest independent range per gate over which the admitted set is unchanged.

    The region on which the verdict is unchanged need not be a box in general,
    because a refused route needs only one of the three conditions to refuse it.
    The box below is therefore verified exhaustively against the full grid
    before it is reported.
    """
    declared_set = _admits(records, declared)
    box: dict[str, dict] = {}
    for gate in GATES:
        levels = _grid(denominators[gate])
        keep = [lv for lv in levels if _admits(records, {**declared, gate: float(lv)}) == declared_set]
        if not keep:
            raise RefusedError(f"the declared verdict is not reproduced anywhere on the {gate} grid")
        below = [lv for lv in levels if float(lv) < min(keep)]
        box[gate] = {
            "declared_pct": round(100 * declared[gate], 4),
            "highest_unchanged_pct": round(100 * float(max(keep)), 4),
            "lowest_unchanged_pct": round(100 * float(min(keep)), 4),
            "opens_below_pct": round(100 * float(max(below)), 4) if below else None,
            "step_pct": round(100 / denominators[gate], 4),
            "headroom_above_declared_pp": round(100 * (float(max(keep)) - declared[gate]), 4),
            "slack_below_declared_pp": (
                None if not below else round(100 * (declared[gate] - float(max(below))), 4)
            ),
        }

    # Exhaustive check that the box really is a region of constant verdict.
    checked = 0
    for a in _grid(denominators["overall_accuracy"]):
        if not box["overall_accuracy"]["lowest_unchanged_pct"] <= 100 * float(a) <= box["overall_accuracy"]["highest_unchanged_pct"]:
            continue
        for b in _grid(denominators["state_reconstruction"]):
            if not box["state_reconstruction"]["lowest_unchanged_pct"] <= 100 * float(b) <= box["state_reconstruction"]["highest_unchanged_pct"]:
                continue
            for c in _grid(denominators["terminal_scoring"]):
                if not box["terminal_scoring"]["lowest_unchanged_pct"] <= 100 * float(c) <= box["terminal_scoring"]["highest_unchanged_pct"]:
                    continue
                checked += 1
                thresholds = {
                    "overall_accuracy": float(a),
                    "state_reconstruction": float(b),
                    "terminal_scoring": float(c),
                }
                if _admits(records, thresholds) != declared_set:
                    raise RefusedError(f"the admitted set changes inside the reported box at {thresholds}")
    box["grid_points_verified"] = checked
    return box


def _single_condition_verdicts(records: list[dict], declared: dict[str, float]) -> dict:
    """What each condition would admit on its own, at its declared value."""
    out = {}
    for gate in GATES:
        alone = {g: (declared[g] if g == gate else 0.0) for g in GATES}
        dropped = {g: (0.0 if g == gate else declared[g]) for g in GATES}
        out[gate] = {
            "admitted_by_this_condition_alone": _admits(records, alone),
            "admitted_with_this_condition_dropped": _admits(records, dropped),
        }
    return out


def main() -> int:
    records, denominators, declared = _load()
    declared_set = _admits(records, declared)
    recorded_set = [rec["label"] for rec in records if rec["declared_admitted"]]
    if declared_set != recorded_set:
        raise RefusedError(
            f"recomputed admitted set {declared_set} does not match the recorded verdicts {recorded_set}"
        )

    bands = {gate: _bands(records, gate, declared, denominators[gate]) for gate in GATES}
    by_level = {
        gate: _binding_by_level(records, gate, declared, denominators[gate]) for gate in GATES
    }
    box = _maximal_box(records, declared, denominators)
    single = _single_condition_verdicts(records, declared)

    binding_declared = {
        rec["label"]: _binding(rec, declared) for rec in records if not rec["declared_admitted"]
    }
    # The gate that is nearest to changing its answer in each direction.
    tighten = []
    for gate in GATES:
        head = box[gate]["headroom_above_declared_pp"]
        after = [b for b in bands[gate] if b["threshold_at_most_pct"] > box[gate]["highest_unchanged_pct"]]
        leaves = sorted(set(declared_set) - set(after[0]["admitted"])) if after else []
        tighten.append({"gate": gate, "headroom_pp": head, "route_that_leaves_first": leaves})
    loosen = []
    for gate in GATES:
        slack = box[gate]["slack_below_declared_pp"]
        if slack is None:
            loosen.append({"gate": gate, "slack_pp": None, "route_that_enters_first": []})
            continue
        opened = _admits(records, {**declared, gate: box[gate]["opens_below_pct"] / 100})
        loosen.append(
            {
                "gate": gate,
                "slack_pp": slack,
                "route_that_enters_first": sorted(set(opened) - set(declared_set)),
            }
        )

    summary = {
        "schema_version": 1,
        "generated_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "estimand": ESTIMAND,
        "campaign": "ai-race-frontier-admission-v6",
        "sources": {
            "csv": str(CSV_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
            "json": str(JSON_PATH.relative_to(REPO_ROOT)).replace("\\", "/"),
        },
        "declared_thresholds_pct": {g: round(100 * declared[g], 4) for g in GATES},
        "declared_admitted": declared_set,
        "resolution": {
            gate: {
                "answers_scored_per_route": denominators[gate],
                "step_pct": round(100 / denominators[gate], 4),
                "achievable_levels_pct": [
                    round(100 * float(lv), 4) for lv in _grid(denominators[gate])
                ],
                "declared_threshold_is_achievable": any(
                    abs(float(lv) - declared[gate]) < 1e-9 for lv in _grid(denominators[gate])
                ),
            }
            for gate in GATES
        },
        "one_at_a_time_bands": bands,
        "binding_by_threshold_level": by_level,
        "unchanged_box": box,
        "single_condition_verdicts": single,
        "binding_conditions_at_declared_thresholds": binding_declared,
        "nearest_change_by_tightening": tighten,
        "nearest_change_by_loosening": loosen,
    }

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "admission_threshold_sensitivity.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    with (OUT_DIR / "one_at_a_time_bands.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["gate", "threshold_above_pct", "threshold_at_most_pct", "n_admitted",
             "admitted", "refused_with_binding_conditions"]
        )
        for gate in GATES:
            for band in bands[gate]:
                writer.writerow(
                    [
                        gate,
                        "" if band["threshold_above_pct"] is None else band["threshold_above_pct"],
                        band["threshold_at_most_pct"],
                        band["n_admitted"],
                        "; ".join(band["admitted"]),
                        "; ".join(
                            f"{label} [{'+'.join(cond) if cond else 'none'}]"
                            for label, cond in band["binding_for_refused_at_band_top"].items()
                        ),
                    ]
                )

    with (OUT_DIR / "binding_by_threshold.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["gate_swept", "threshold_pct", "route", "admitted", "binding_conditions"])
        for gate in GATES:
            for row in by_level[gate]:
                writer.writerow(
                    [
                        row["gate_swept"],
                        row["threshold_pct"],
                        row["label"],
                        row["admitted"],
                        "; ".join(row["binding_conditions"]) or "none",
                    ]
                )

    with (OUT_DIR / "binding_conditions.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["route", "label", "overall_accuracy_pct", "state_reconstruction_pct",
             "terminal_scoring_pct", "declared_admitted", "binding_conditions_at_declared"]
        )
        for rec in records:
            writer.writerow(
                [
                    rec["route"],
                    rec["label"],
                    round(100 * rec["accuracy"]["overall_accuracy"], 1),
                    round(100 * rec["accuracy"]["state_reconstruction"], 1),
                    round(100 * rec["accuracy"]["terminal_scoring"], 1),
                    rec["declared_admitted"],
                    "; ".join(_binding(rec, declared)) or "none",
                ]
            )

    lines = [
        "# Admission-gate threshold sensitivity",
        "",
        f"Campaign `ai-race-frontier-admission-v6`. Estimand: {ESTIMAND}",
        "",
        "## Resolution of the instrument",
        "",
        "| Gate condition | Answers scored per route | Step | Declared threshold achievable? |",
        "|---|---|---|---|",
    ]
    for gate in GATES:
        res = summary["resolution"][gate]
        lines.append(
            f"| `{gate}` | {res['answers_scored_per_route']} | {res['step_pct']:.2f} pp | "
            f"{'yes' if res['declared_threshold_is_achievable'] else 'no'} |"
        )
    lines += ["", "## Bands over which the admitted set is constant", ""]
    for gate in GATES:
        lines.append(f"### `{gate}`")
        lines.append("")
        lines.append("| Threshold band | Admitted |")
        lines.append("|---|---|")
        for band in bands[gate]:
            low = band["threshold_above_pct"]
            span = (
                f"at most {band['threshold_at_most_pct']:.1f}%"
                if low is None
                else f"above {low:.1f}% and at most {band['threshold_at_most_pct']:.1f}%"
            )
            lines.append(f"| {span} | {band['n_admitted']}: {', '.join(band['admitted']) or 'none'} |")
        lines.append("")
    lines += ["## Binding condition at the declared thresholds", ""]
    for rec in records:
        binding = _binding(rec, declared)
        lines.append(f"- {rec['label']}: {'; '.join(binding) if binding else 'admitted, no condition binds'}")
    lines.append("")
    (OUT_DIR / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"declared admitted set ({len(declared_set)}): {', '.join(declared_set)}")
    for gate in GATES:
        b = box[gate]
        print(
            f"{gate}: declared {b['declared_pct']:.1f}%, unchanged up to {b['highest_unchanged_pct']:.1f}% "
            f"(+{b['headroom_above_declared_pp']:.1f} pp), opens below "
            f"{b['opens_below_pct'] if b['opens_below_pct'] is None else format(b['opens_below_pct'], '.1f')}"
        )
    print(f"grid points verified inside the unchanged box: {box['grid_points_verified']}")
    for entry in tighten:
        print(f"tighten {entry['gate']}: first out {entry['route_that_leaves_first'] or 'none'}")
    for entry in loosen:
        print(f"loosen {entry['gate']}: first in {entry['route_that_enters_first'] or 'none'}")
    for gate in GATES:
        print(f"{gate} alone admits: {single[gate]['admitted_by_this_condition_alone']}")
        print(f"{gate} dropped leaves: {single[gate]['admitted_with_this_condition_dropped']}")
    print(f"wrote {OUT_DIR.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
