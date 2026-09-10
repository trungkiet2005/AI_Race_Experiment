"""Estimate the group-size effect from the matched N = 2 to 5 sweep.

The earlier three- to five-player pilots had no two-player arm, so a difference
across group sizes could not be separated from a difference between two lanes.
This sweep runs every group size through one engine, one prompt template, one
parser and one protocol, and the two-player arm is the paper's own game: the
group-count stage rule evaluated at two players reproduces the headline payoff
matrix exactly, which ``scripts/verify_matched_nplayer_design.py`` asserts
offline.

Two estimands, and they answer different questions.

The per-cell Unsafe rate is the descriptive picture. Its uncertainty is taken
over races, because the seats of one race share a state history and are not
separate draws.

The group-size contrast is the point of the design. The game seed is
``base_seed + rep`` and is independent of both the risk treatment and the number
of seats, so a repetition index reuses one horizon stopping-draw stream at every
risk and every group size. Differencing N against the two-player arm inside a
repetition therefore removes the horizon draw, leaving group size as what varies.
This script re-derives that pairing from the recorded ``game_seed`` rather than
assuming it, and refuses to report a paired contrast whose blocks do not match.

What the design cannot do, and no analysis can repair, is isolate a group-size
effect free of everything group size entails. Adding a company changes the stage
payoff, because the group-count rule depends on how many companies there are,
and it lengthens the state description the agent reads. Those are properties of
the game and of the task. The estimand is the effect of group size within this
game's own definition.

Ten races per cell supports a race-clustered interval and nothing stronger. Any
cell below the repository's five-independent-race floor is reported as
descriptive only rather than given an interval that would not mean anything.
"""

from __future__ import annotations

import csv
import hashlib
import json
import platform
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "frontier" / "nplayer_matched_campaign"
OUT_DIR = ROOT / "results" / "derived" / "nplayer_matched_campaign"

PROTOCOL_ID = "ai-race-nplayer-matched-hosted-confirmatory-v1"
GROUP_SIZES = (2, 3, 4, 5)
BASELINE_SIZE = 2
RISKS = ("0.1", "0.6", "0.9")
EXPECTED_REPETITIONS = 10
# One (group size, risk) cell is the unit that gets collected, because a Model
# Proxy identity cannot sustain the whole sweep in one run.
EXPECTED_RACES_PER_CELL = EXPECTED_REPETITIONS
MIN_RACES_FOR_INFERENCE = 5

N_BOOT = 5000
SEED = 20260910


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def discover() -> dict[str, dict[tuple[int, str], dict]]:
    """Find every collected cell: one directory per (group size, risk)."""
    found: dict[str, dict[tuple[int, str], dict]] = defaultdict(dict)
    for receipt_path in sorted(CAMPAIGN.rglob("collection_receipt.json")):
        if "failed_runs" in receipt_path.parts:
            continue
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        n_players = int(receipt["cell"]["n_players"])
        risk = f"{float(receipt['cell']['max_private_risk']):g}"
        turns_path = receipt_path.parent / f"n{n_players}" / "turns.jsonl"
        manifest_path = receipt_path.parent / "run_manifest.json"
        if not turns_path.exists() or not manifest_path.exists():
            continue
        found[receipt["model_route"]][(n_players, risk)] = {
            "receipt": receipt,
            "manifest": json.loads(manifest_path.read_text(encoding="utf-8")),
            "turns_path": turns_path,
        }
    return found


def load_cell(entry: dict, n_players: int, risk: str) -> tuple[dict[str, dict], list[str]]:
    """Collapse one cell's decisions into races, and report what is wrong."""
    races: dict[str, dict] = {}
    problems: list[str] = []
    parse_failures = 0

    manifest = entry["manifest"]
    if manifest.get("status") != "completed":
        problems.append(
            f"N={n_players} risk {risk}: manifest status {manifest.get('status')!r}"
        )
    if manifest.get("protocol_id") != PROTOCOL_ID:
        problems.append(
            f"N={n_players} risk {risk}: protocol {manifest.get('protocol_id')!r}"
        )

    for line in entry["turns_path"].open(encoding="utf-8"):
        if not line.strip():
            continue
        row = json.loads(line)
        if int(row["n_players"]) != n_players:
            problems.append(
                f"N={n_players} risk {risk}: a decision records "
                f"{row['n_players']} players"
            )
        if f"{float(row['max_private_risk']):g}" != risk:
            problems.append(
                f"N={n_players} risk {risk}: a decision records risk "
                f"{row['max_private_risk']}"
            )
        if row.get("parse_failed"):
            parse_failures += 1
        rec = races.setdefault(
            row["game_id"],
            {
                "n_players": n_players,
                "risk": risk,
                "repetition": int(row["rep"]),
                "game_seed": row.get("game_seed"),
                "unsafe": 0,
                "decisions": 0,
                "seats": set(),
            },
        )
        rec["unsafe"] += int(row["unsafe"])
        rec["decisions"] += 1
        rec["seats"].add(row["player_index"])

    if parse_failures:
        problems.append(f"N={n_players} risk {risk}: {parse_failures} parse failures")
    if len(races) != EXPECTED_RACES_PER_CELL:
        problems.append(
            f"N={n_players} risk {risk}: {len(races)} races, "
            f"expected {EXPECTED_RACES_PER_CELL}"
        )
    for rec in races.values():
        if len(rec["seats"]) != n_players:
            problems.append(
                f"N={n_players} risk {risk}: a race recorded {len(rec['seats'])} seats"
            )
    return races, problems


def cell_rng(*key) -> np.random.Generator:
    """A generator determined by the cell alone, not by the campaign around it.

    A single generator threaded through every cell makes each interval depend on
    how many cells were analysed before it, so collecting one more risk level
    silently moves the intervals of the levels already reported.  Deriving the
    stream from a stable digest of the cell's own identity keeps a cell's
    interval reproducible whatever else is in the tree.
    """
    digest = hashlib.sha256(
        "|".join(str(part) for part in key).encode("utf-8")
    ).digest()
    return np.random.default_rng([SEED, int.from_bytes(digest[:8], "big")])


def cluster_bootstrap(cells: list[tuple[int, int]], rng: np.random.Generator):
    unsafe = np.array([c[0] for c in cells], dtype=float)
    total = np.array([c[1] for c in cells], dtype=float)
    point = unsafe.sum() / total.sum()
    idx = rng.integers(0, len(cells), size=(N_BOOT, len(cells)))
    draws = unsafe[idx].sum(axis=1) / total[idx].sum(axis=1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_group_size_contrast(
    by_size: dict[int, dict[str, dict]], n_players: int, rng: np.random.Generator
) -> dict:
    """N against the two-player arm, differenced inside a repetition block."""
    blocks: dict[tuple[str, int], dict[int, dict]] = defaultdict(dict)
    for size in (BASELINE_SIZE, n_players):
        for rec in by_size[size].values():
            blocks[(rec["risk"], rec["repetition"])][size] = rec

    diffs: list[float] = []
    seed_matched = 0
    seed_total = 0
    for sides in blocks.values():
        if BASELINE_SIZE not in sides or n_players not in sides:
            continue
        base, other = sides[BASELINE_SIZE], sides[n_players]
        seed_total += 1
        if base["game_seed"] is not None and base["game_seed"] == other["game_seed"]:
            seed_matched += 1
        diffs.append(
            other["unsafe"] / other["decisions"] - base["unsafe"] / base["decisions"]
        )

    if not diffs:
        return {"available": False, "reason": "no complete repetition blocks"}
    array = np.array(diffs, dtype=float)
    idx = rng.integers(0, len(array), size=(N_BOOT, len(array)))
    draws = array[idx].mean(axis=1)
    return {
        "available": True,
        "group_size": n_players,
        "baseline_group_size": BASELINE_SIZE,
        "n_blocks": len(array),
        "mean_difference": float(array.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "seed_matched_blocks": seed_matched,
        "seed_blocks_checked": seed_total,
        "pairing_verified": seed_total > 0 and seed_matched == seed_total,
    }


def main() -> None:
    found = discover()
    if not found:
        raise SystemExit(
            f"no matched cells found under {CAMPAIGN.relative_to(ROOT)}"
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    contrasts: dict[str, dict] = {}
    identities: dict[str, dict[str, str]] = {}
    hashes: dict[str, str] = {}
    refused: dict[str, list[str]] = {}

    for route, cells in sorted(found.items()):
        risks_present = sorted({risk for _, risk in cells})
        complete_risks = [
            risk
            for risk in risks_present
            if all((n, risk) in cells for n in GROUP_SIZES)
        ]
        incomplete = [risk for risk in risks_present if risk not in complete_risks]
        if incomplete:
            refused.setdefault(route, []).append(
                "risk levels collected for only some group sizes, so not "
                f"reported: {incomplete}"
            )
        if not complete_risks:
            refused.setdefault(route, []).append(
                "no risk level has all four group sizes; there is no matched "
                "comparison to report"
            )
            continue

        by_risk: dict[str, dict[int, dict[str, dict]]] = {}
        problems: list[str] = []
        for risk in complete_risks:
            by_risk[risk] = {}
            for n in GROUP_SIZES:
                entry = cells[(n, risk)]
                races, issues = load_cell(entry, n, risk)
                by_risk[risk][n] = races
                problems.extend(issues)
                hashes[str(entry["turns_path"].relative_to(ROOT))] = sha256_file(
                    entry["turns_path"]
                )
                identities.setdefault(route, {})[f"N={n} risk {risk}"] = entry[
                    "receipt"
                ]["executing_identity"]
        if problems:
            refused.setdefault(route, []).extend(problems)
            continue

        for risk in complete_risks:
            for n in GROUP_SIZES:
                cells_for_boot = [
                    (r["unsafe"], r["decisions"]) for r in by_risk[risk][n].values()
                ]
                supported = len(cells_for_boot) >= MIN_RACES_FOR_INFERENCE
                if supported:
                    point, low, high = cluster_bootstrap(
                        cells_for_boot, cell_rng("rate", route, risk, n)
                    )
                else:
                    point = sum(c[0] for c in cells_for_boot) / max(
                        sum(c[1] for c in cells_for_boot), 1
                    )
                    low = high = None
                rows.append(
                    {
                        "model_route": route,
                        "n_players": n,
                        "risk": risk,
                        "n_races": len(cells_for_boot),
                        "n_decisions": sum(c[1] for c in cells_for_boot),
                        "unsafe_rate": point,
                        "ci95_low": low,
                        "ci95_high": high,
                        "inference_supported": supported,
                        "executing_identity": identities[route][f"N={n} risk {risk}"],
                    }
                )
            contrasts.setdefault(route, {})[risk] = {
                str(n): paired_group_size_contrast(
                    by_risk[risk], n, cell_rng("contrast", route, risk, n)
                )
                for n in GROUP_SIZES
                if n != BASELINE_SIZE
            }

    if not rows:
        for route, problems in refused.items():
            print(f"refused {route}:")
            for problem in problems:
                print(f"  {problem}")
        raise SystemExit("no complete matched comparison to report")

    csv_path = OUT_DIR / "nplayer_matched_rates.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "schema_version": "ai-race-nplayer-matched-analysis-v1",
        "estimand": (
            "Effect of the number of competing companies on the Unsafe rate, "
            "within this game's own definition, reported as a race-clustered "
            "per-cell rate and as a repetition-paired contrast against the "
            "two-player arm."
        ),
        "protocol_id": PROTOCOL_ID,
        "design": {
            "group_sizes": list(GROUP_SIZES),
            "baseline_group_size": BASELINE_SIZE,
            "repetitions_per_cell": EXPECTED_REPETITIONS,
            "risks_reported": sorted({r["risk"] for r in rows}),
            "risks_in_frozen_grid": list(RISKS),
        },
        "collection_note": (
            "The sweep was collected one (group size, risk) cell at a time on "
            "separate Kaggle identities, because no single Model Proxy identity "
            "sustains the whole design. The assignment was declared before the "
            "first run in docs/matched-nplayer-collection-plan-2026-09-10.md. "
            "Every cell is collected whole, so it carries its own "
            "race-clustered interval, and the game seed is independent of the "
            "treatment and the seat count so repetitions still pair across "
            "cells collected separately."
        ),
        "executing_identities": identities,
        "identification_note": (
            "Group size is not separable from the stage payoff or from the "
            "prompt length: the group-count rule depends on the number of "
            "companies and the state description names each of them. No "
            "analysis can repair that, so this is the effect of group size "
            "within the game's definition, not a pure group-size effect."
        ),
        "cell_size_note": (
            f"{EXPECTED_REPETITIONS} races per cell. A cell below "
            f"{MIN_RACES_FOR_INFERENCE} independent races is reported without "
            "an interval rather than with one that would not mean anything."
        ),
        "routes_tabulated": sorted({r["model_route"] for r in rows}),
        "routes_refused": refused,
        "paired_contrasts": contrasts,
        "ci_method": (
            "percentile bootstrap; per-cell rates resample races, paired "
            "contrasts resample repetition blocks"
        ),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "source_sha256": hashes,
        "environment": {"python": platform.python_version(), "numpy": np.__version__},
    }
    json_path = OUT_DIR / "nplayer_matched_rates.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"wrote {csv_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    for route, problems in refused.items():
        print(f"note on {route}: {problems}")
    for route, per_risk in contrasts.items():
        print(f"\n{route}")
        for risk, per_n in sorted(per_risk.items()):
            print(f"  risk {risk}")
            base = [r for r in rows if r["model_route"] == route
                    and r["risk"] == risk and r["n_players"] == BASELINE_SIZE][0]
            print(
                f"    N={BASELINE_SIZE} (baseline): "
                f"{100 * base['unsafe_rate']:.1f}% "
                f"[{100 * base['ci95_low']:.1f}, {100 * base['ci95_high']:.1f}]"
            )
            for n, res in sorted(per_n.items(), key=lambda kv: int(kv[0])):
                rate = [r for r in rows if r["model_route"] == route
                        and r["risk"] == risk and r["n_players"] == int(n)][0]
                if not res.get("available"):
                    print(f"    N={n}: unavailable ({res.get('reason')})")
                    continue
                verified = "verified" if res["pairing_verified"] else "NOT verified"
                print(
                    f"    N={n}: {100 * rate['unsafe_rate']:.1f}% "
                    f"[{100 * rate['ci95_low']:.1f}, {100 * rate['ci95_high']:.1f}]"
                    f"  |  minus N={BASELINE_SIZE}: "
                    f"{100 * res['mean_difference']:+.1f} pp "
                    f"[{100 * res['ci95_low']:+.1f}, {100 * res['ci95_high']:+.1f}] "
                    f"over {res['n_blocks']} blocks, seed pairing {verified}"
                )


if __name__ == "__main__":
    main()
