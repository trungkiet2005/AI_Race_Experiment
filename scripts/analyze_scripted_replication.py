#!/usr/bin/env python3
"""Replication blocks and refused routes for the scripted-rival design.

Declared in docs/ijcai-revision-collection-plan-2026-11-23.md. Block A is the
canonical campaign; blocks B and C are pre-declared replication blocks with new
seed bases on the AS and AU rivals only. A route enters a block only when all
six of its (rival, risk) cells exist and pass the loader's checks; partial
blocks are listed, never tabulated. Repetitions pair across risks inside a
block, so the pooled analysis clusters on (block, repetition).

Refused routes (block A, AS and AU) are descriptive and never pooled with the
admitted routes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_scripted_opponent as aso  # noqa: E402
from scripts import analyze_scripted_interaction_robust as rob  # noqa: E402

FRONTIER = ROOT / "results" / "frontier"
SOURCES = {
    "A": FRONTIER / "scripted_opponent_campaign",
    "B": FRONTIER / "scripted_opponent_replication" / "block_B",
    "C": FRONTIER / "scripted_opponent_replication" / "block_C",
}
REFUSED = FRONTIER / "scripted_opponent_refused"
OUT = ROOT / "results" / "derived" / "scripted_replication"
ADMITTED = [
    "google/gemini-3-flash-preview",
    "openai/gpt-5.4-2026-03-05",
    "openai/gpt-5.5-2026-04-23",
    "anthropic/claude-sonnet-5@default",
    "anthropic/claude-opus-5@default",
]
RIVALS = ("AS", "AU")
LOW, HIGH = 0.1, 0.9
SEED = 20261123


def load_blocks(campaign: Path, route: str):
    if not campaign.exists():
        return None, ["campaign directory absent"]
    aso.CAMPAIGN = campaign
    turns, _, problems = aso.load_cells(route)
    if turns.empty:
        return None, problems or ["no cells"]
    blocks = aso.repetition_blocks(turns)
    missing = [f"{s}@{r}" for s in RIVALS for r in aso.RISKS if (s, r) not in blocks]
    if missing:
        return None, problems + [f"missing {missing}"]
    return {k: v for k, v in blocks.items() if k[0] in RIVALS}, problems


def interaction_values(blocks):
    reps = sorted(set.intersection(*(set(blocks[(s, r)]) for s in RIVALS for r in (LOW, HIGH))))
    return np.array([(blocks[("AU", HIGH)][k]["rate"] - blocks[("AS", HIGH)][k]["rate"])
                     - (blocks[("AU", LOW)][k]["rate"] - blocks[("AS", LOW)][k]["rate"]) for k in reps])


def contrast_values(blocks, risk):
    reps = sorted(set(blocks[("AU", risk)]) & set(blocks[("AS", risk)]))
    return np.array([blocks[("AU", risk)][k]["rate"] - blocks[("AS", risk)][k]["rate"] for k in reps])


def summarise(vals, rng):
    lo, hi = rob.signflip_interval(vals)
    b_lo, b_hi, _, _ = rob.bca_interval(vals, rng, B=50_000)
    return {"n_blocks": int(len(vals)), "estimate_pp": 100 * float(vals.mean()),
            "signflip_ci_pp": [100 * lo, 100 * hi], "signflip_p": float(rob.signflip_p(vals)),
            "bca_ci_pp": [100 * b_lo, 100 * b_hi]}


def logit_rows(blocks, tag):
    rows = []
    for (s, r), blk in blocks.items():
        if r in (LOW, HIGH):
            for rep, b in blk.items():
                rows.append((1.0 if s == "AU" else 0.0, r, f"{tag}-{rep}", b["unsafe"], b["decisions"]))
    return rows


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    report = {"plan": "docs/ijcai-revision-collection-plan-2026-11-23.md", "routes": {}, "excluded": {}}
    pooled_p = []
    for route in ADMITTED:
        per_block, merged, rows = {}, {}, []
        for tag, path in SOURCES.items():
            blocks, problems = load_blocks(path, route)
            if blocks is None:
                report["excluded"][f"{route} block {tag}"] = problems
                continue
            per_block[tag] = {
                "interaction": summarise(interaction_values(blocks), rng),
                "au_minus_as": {str(r): summarise(contrast_values(blocks, r), rng) for r in aso.RISKS},
            }
            for key, blk in blocks.items():
                merged.setdefault(key, {}).update({f"{tag}-{rep}": b for rep, b in blk.items()})
            rows += logit_rows(blocks, tag)
        if not per_block:
            continue
        entry = {"blocks_present": sorted(per_block), "per_block": per_block}
        if len(per_block) > 1:
            entry["pooled"] = {
                "interaction": summarise(interaction_values(merged), rng),
                "au_minus_as": {str(r): summarise(contrast_values(merged, r), rng) for r in aso.RISKS},
                "logodds_interaction": rob.cluster_logit(rows, (LOW, HIGH)),
            }
            pooled_p.append((route, entry["pooled"]["interaction"]["signflip_p"]))
        report["routes"][route] = entry
    if pooled_p:
        adj = rob.holm(np.array([p for _, p in pooled_p]))
        for (route, _), a in zip(pooled_p, adj):
            report["routes"][route]["pooled"]["interaction"]["holm_p"] = float(a)

    refused = {}
    if REFUSED.exists():
        aso.CAMPAIGN = REFUSED
        for route in aso.routes_present():
            blocks, problems = load_blocks(REFUSED, route)
            if blocks is None:
                report["excluded"][f"{route} refused"] = problems
                continue
            refused[route] = {
                "rates_pct": {f"{s}@{r}": 100 * sum(b["unsafe"] for b in blocks[(s, r)].values())
                              / sum(b["decisions"] for b in blocks[(s, r)].values())
                              for s in RIVALS for r in aso.RISKS},
                "au_minus_as": {str(r): summarise(contrast_values(blocks, r), rng) for r in aso.RISKS},
                "interaction": summarise(interaction_values(blocks), rng),
            }
    report["refused_routes"] = refused
    (OUT / "scripted_replication.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for route, e in report["routes"].items():
        line = ", ".join(f"{t}: {e['per_block'][t]['interaction']['estimate_pp']:+.1f}" for t in e["blocks_present"])
        pooled = e.get("pooled", {}).get("interaction")
        tail = (f" | pooled {pooled['estimate_pp']:+.1f} [{pooled['signflip_ci_pp'][0]:+.1f}, "
                f"{pooled['signflip_ci_pp'][1]:+.1f}] n={pooled['n_blocks']}" if pooled else "")
        print(f"{route}: {line}{tail}")
    for k, v in report["excluded"].items():
        print(f"excluded {k}: {v}")
    print(f"wrote {OUT / 'scripted_replication.json'}")


if __name__ == "__main__":
    main()
