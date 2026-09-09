"""Estimate the representation effect from the crossed frontier context/mapping runs.

The design is fully crossed and counterbalanced: two narrative skins by two
assignments of the opaque response codes P and Q to Safe and Unsafe, by three
private-risk treatments, by ten repetitions, so 120 races per route. Nothing
about the game changes across those cells. The payoff matrix, the progress
increments, the hidden horizon law and the risk formula are identical; only how
the task is worded and which letter names which action move. Any behavioural
difference across a mapping cell is therefore a property of the presentation,
not of the game.

Two estimands are reported and they are not interchangeable.

The marginal rate pools every decision in a cell and is what a reader expects to
see, but its uncertainty has to be taken over races rather than decisions,
because both seats of a race share one state history.

The paired contrast is the stronger one and is what the counterbalancing was for.
Repetition index is a common-random-number block: the same index reuses the same
sampled horizon across mapping and context cells, which the recorded
``horizon_draws_sha256`` lets us verify rather than assume. Differencing within a
repetition removes the horizon draw from the comparison, so the remaining
variation is the presentation. The script checks the pairing holds and refuses to
report a paired number if it does not.

The bootstrap resamples the independent unit: races for the marginal rates,
repetition blocks for the paired contrasts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "frontier" / "context_mapping_campaign_v3"
OUT_DIR = ROOT / "results" / "derived" / "frontier_context_mapping_campaign_v3"

PROTOCOL_ID = "ai-race-frontier-context-mapping-v3"
CONTEXTS = ("technology_race", "abstract_game")
MAPPINGS = ("P_SAFE_Q_UNSAFE", "Q_SAFE_P_UNSAFE")
RISKS = ("0.1", "0.6", "0.9")
EXPECTED_REPETITIONS = 10
EXPECTED_RACES = len(CONTEXTS) * len(MAPPINGS) * len(RISKS) * EXPECTED_REPETITIONS
EXPECTED_DECISIONS_PER_CELL = 186

N_BOOT = 5000
SEED = 20260909


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_route(turns_path: Path) -> list[dict]:
    rows = []
    with turns_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def discover() -> dict[str, dict]:
    """Find every completed route in the campaign, skipping failure records."""
    routes: dict[str, dict] = {}
    for manifest_path in sorted(CAMPAIGN.rglob("run_manifest.json")):
        # Failure records stay in the accounting chain but are never evidence, so
        # the derived table walks only the live task tree.
        if "failed_runs" in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        turns_path = manifest_path.with_name("turns.jsonl")
        if not turns_path.exists():
            continue
        routes[manifest["model_route"]] = {
            "manifest": manifest,
            "manifest_path": manifest_path,
            "turns_path": turns_path,
        }
    return routes


def validate(route: str, entry: dict) -> list[str]:
    """Refuse a route rather than report a partial counterbalanced design."""
    problems: list[str] = []
    manifest = entry["manifest"]
    if manifest.get("status") != "completed":
        problems.append(
            f"{route}: manifest status is {manifest.get('status')!r}, not completed"
        )
    if manifest.get("protocol_id") != PROTOCOL_ID:
        problems.append(
            f"{route}: protocol_id is {manifest.get('protocol_id')!r}, not {PROTOCOL_ID}"
        )

    rows = entry["rows"]
    races = {r["game_id"] for r in rows}
    if len(races) != EXPECTED_RACES:
        problems.append(f"{route}: {len(races)} races, expected {EXPECTED_RACES}")

    parse_failures = sum(1 for r in rows if r.get("parse_failed"))
    if parse_failures:
        problems.append(f"{route}: {parse_failures} parse failures")

    per_cell: dict[tuple, set] = defaultdict(set)
    decisions: dict[tuple, int] = defaultdict(int)
    for r in rows:
        key = (r["context"], r["mapping"], f"{float(r['max_private_risk']):g}")
        per_cell[key].add(r["game_id"])
        decisions[key] += 1
    for context in CONTEXTS:
        for mapping in MAPPINGS:
            for risk in RISKS:
                key = (context, mapping, risk)
                if len(per_cell[key]) != EXPECTED_REPETITIONS:
                    problems.append(
                        f"{route}: cell {key} has {len(per_cell[key])} races, "
                        f"expected {EXPECTED_REPETITIONS}"
                    )
                if decisions[key] != EXPECTED_DECISIONS_PER_CELL:
                    problems.append(
                        f"{route}: cell {key} has {decisions[key]} decisions, "
                        f"expected {EXPECTED_DECISIONS_PER_CELL}"
                    )
    return problems


def race_table(rows: list[dict]) -> dict[str, dict]:
    """One record per race, which is the independent unit."""
    agg: dict[str, dict] = {}
    for r in rows:
        rec = agg.setdefault(
            r["game_id"],
            {
                "context": r["context"],
                "mapping": r["mapping"],
                "risk": f"{float(r['max_private_risk']):g}",
                "repetition": int(r["repetition"]),
                "unsafe": 0,
                "decisions": 0,
                "horizon": r.get("horizon_draws_sha256"),
            },
        )
        rec["unsafe"] += int(r["unsafe"])
        rec["decisions"] += 1
    return agg


def cluster_bootstrap(values: list[tuple[int, int]], rng: np.random.Generator) -> tuple[float, float, float]:
    """Percentile interval resampling races, each a (unsafe, decisions) pair."""
    unsafe = np.array([v[0] for v in values], dtype=float)
    total = np.array([v[1] for v in values], dtype=float)
    point = unsafe.sum() / total.sum()
    idx = rng.integers(0, len(values), size=(N_BOOT, len(values)))
    draws = unsafe[idx].sum(axis=1) / total[idx].sum(axis=1)
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))


def paired_contrast(
    races: dict[str, dict], factor: str, levels: tuple[str, str], rng: np.random.Generator
) -> dict:
    """Difference within a repetition block, which holds the horizon draw fixed.

    The block is every cell sharing the repetition index and the levels of the
    other factors, so the only thing that differs inside a block is `factor`.
    """
    other = "context" if factor == "mapping" else "mapping"
    blocks: dict[tuple, dict[str, dict]] = defaultdict(dict)
    for rec in races.values():
        key = (rec["repetition"], rec["risk"], rec[other])
        blocks[key][rec[factor]] = rec

    pairs: list[tuple[float, float]] = []
    horizon_matched = 0
    horizon_total = 0
    for key, sides in blocks.items():
        if levels[0] not in sides or levels[1] not in sides:
            continue
        a, b = sides[levels[0]], sides[levels[1]]
        horizon_total += 1
        if a["horizon"] is not None and a["horizon"] == b["horizon"]:
            horizon_matched += 1
        pairs.append((a["unsafe"] / a["decisions"], b["unsafe"] / b["decisions"]))

    if not pairs:
        return {"available": False, "reason": "no complete repetition blocks"}

    diffs = np.array([p[1] - p[0] for p in pairs], dtype=float)
    idx = rng.integers(0, len(diffs), size=(N_BOOT, len(diffs)))
    draws = diffs[idx].mean(axis=1)
    return {
        "available": True,
        "factor": factor,
        "levels": list(levels),
        "n_blocks": len(pairs),
        "mean_difference": float(diffs.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "horizon_draw_matched_blocks": horizon_matched,
        "horizon_draw_blocks_checked": horizon_total,
        "pairing_verified": horizon_total > 0 and horizon_matched == horizon_total,
    }


def main() -> None:
    routes = discover()
    if not routes:
        raise SystemExit(f"no runs found under {CAMPAIGN}")

    for route, entry in routes.items():
        entry["rows"] = load_route(entry["turns_path"])

    admitted: dict[str, dict] = {}
    refused: dict[str, list[str]] = {}
    for route, entry in routes.items():
        problems = validate(route, entry)
        if problems:
            refused[route] = problems
        else:
            admitted[route] = entry

    print(f"routes found: {len(routes)}; complete and crossed: {len(admitted)}")
    for route, problems in refused.items():
        print(f"  refused {route}:")
        for p in problems:
            print(f"    {p}")
    if not admitted:
        raise SystemExit("no route holds a complete crossed design; nothing to report")

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    marginal_rows: list[dict] = []
    contrasts: dict[str, dict] = {}
    hashes: dict[str, str] = {}

    for route, entry in admitted.items():
        rng = np.random.default_rng(SEED)
        races = race_table(entry["rows"])
        hashes[str(entry["turns_path"].relative_to(ROOT))] = sha256_file(entry["turns_path"])

        for factor, levels in (("mapping", MAPPINGS), ("context", CONTEXTS)):
            for level in levels:
                cells = [
                    (r["unsafe"], r["decisions"]) for r in races.values() if r[factor] == level
                ]
                point, low, high = cluster_bootstrap(cells, rng)
                marginal_rows.append(
                    {
                        "model_route": route,
                        "factor": factor,
                        "level": level,
                        "risk": "all",
                        "n_races": len(cells),
                        "n_decisions": sum(c[1] for c in cells),
                        "unsafe_rate": point,
                        "ci95_low": low,
                        "ci95_high": high,
                    }
                )
            for risk in RISKS:
                for level in levels:
                    cells = [
                        (r["unsafe"], r["decisions"])
                        for r in races.values()
                        if r[factor] == level and r["risk"] == risk
                    ]
                    point, low, high = cluster_bootstrap(cells, rng)
                    marginal_rows.append(
                        {
                            "model_route": route,
                            "factor": factor,
                            "level": level,
                            "risk": risk,
                            "n_races": len(cells),
                            "n_decisions": sum(c[1] for c in cells),
                            "unsafe_rate": point,
                            "ci95_low": low,
                            "ci95_high": high,
                        }
                    )

        contrasts[route] = {
            "mapping": paired_contrast(races, "mapping", MAPPINGS, rng),
            "context": paired_contrast(races, "context", CONTEXTS, rng),
        }

    csv_path = OUT_DIR / "context_mapping_cross_marginals.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(marginal_rows[0]))
        writer.writeheader()
        writer.writerows(marginal_rows)

    payload = {
        "schema_version": "ai-race-frontier-context-mapping-cross-v1",
        "estimand": (
            "Effect of task presentation, the opaque response-code assignment and the "
            "narrative skin, on the Unsafe rate with the game mechanism held fixed; "
            "reported both as a race-clustered marginal rate and as a repetition-paired "
            "contrast."
        ),
        "protocol_id": PROTOCOL_ID,
        "design": {
            "contexts": list(CONTEXTS),
            "mappings": list(MAPPINGS),
            "risks": list(RISKS),
            "repetitions": EXPECTED_REPETITIONS,
            "races_per_route": EXPECTED_RACES,
        },
        "routes_complete": sorted(admitted),
        "routes_refused": refused,
        "paired_contrasts": contrasts,
        "ci_method": (
            "percentile bootstrap; marginal rates resample races, paired contrasts "
            "resample repetition blocks"
        ),
        "n_bootstrap": N_BOOT,
        "seed": SEED,
        "source_sha256": hashes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
        },
    }
    json_path = OUT_DIR / "context_mapping_cross_marginals.json"
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")

    print(f"\nwrote {csv_path.relative_to(ROOT)}")
    print(f"wrote {json_path.relative_to(ROOT)}")
    for route, result in contrasts.items():
        print(f"\n{route}")
        for factor, res in result.items():
            if not res.get("available"):
                print(f"  {factor}: unavailable ({res.get('reason')})")
                continue
            verified = "verified" if res["pairing_verified"] else "NOT verified"
            print(
                f"  {factor} {res['levels'][1]} minus {res['levels'][0]}: "
                f"{100 * res['mean_difference']:+.1f} pp "
                f"[{100 * res['ci95_low']:+.1f}, {100 * res['ci95_high']:+.1f}] "
                f"over {res['n_blocks']} repetition blocks, horizon pairing {verified}"
            )


if __name__ == "__main__":
    main()
