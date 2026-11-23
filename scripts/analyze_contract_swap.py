#!/usr/bin/env python3
"""E4 of docs/ijcai-revision-collection-plan-2026-11-23.md (amendments 3 and 5).

Claude Sonnet 5 under the GPT-5.5 decoding contract (reasoning low, 1,024
tokens) on AS and AU at risks 0.1 and 0.9, block-A seeds; and the Claude Opus 5
AS@0.6 cell re-collected under the active contract beside its legacy cell.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import analyze_scripted_opponent as aso  # noqa: E402
from scripts import analyze_scripted_interaction_robust as rob  # noqa: E402
from scripts.analyze_scripted_replication import interaction_values, contrast_values  # noqa: E402

SWAP = ROOT / "results/frontier/scripted_opponent_contract_swap"
OUT = ROOT / "results/derived/contract_swap"
SONNET = "anthropic/claude-sonnet-5@default"
OPUS = "anthropic/claude-opus-5@default"


def blocks_of(campaign: Path, route: str):
    aso.CAMPAIGN = campaign
    turns, receipts, problems = aso.load_cells(route)
    if problems:
        raise SystemExit(f"{route} in {campaign}: {problems}")
    return aso.repetition_blocks(turns), turns


def rate(blk):
    return 100 * sum(b["unsafe"] for b in blk.values()) / sum(b["decisions"] for b in blk.values())


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(20261123)
    swap, turns = blocks_of(SWAP / "sonnet_low", SONNET)
    reasoning = sorted(set(json.loads(p.read_text())["decoding"]["reasoning_requested"] or "none"
                           for p in (SWAP / "sonnet_low").glob("*/*/run_manifest.json")))
    orig, _ = blocks_of(ROOT / "results/frontier/scripted_opponent_campaign", SONNET)
    out = {"sonnet_low": {"reasoning_requested": reasoning,
                          "rates_pct": {f"{s}@{r}": rate(b) for (s, r), b in sorted(swap.items())},
                          "original_rates_pct": {f"{s}@{r}": rate(orig[(s, r)]) for (s, r) in sorted(swap)}}}
    for name, blk in (("swapped", swap), ("original", {k: orig[k] for k in swap})):
        v = interaction_values(blk)
        lo, hi = rob.signflip_interval(v)
        out["sonnet_low"][f"interaction_{name}"] = {"estimate_pp": 100 * float(v.mean()),
                                                     "signflip_ci_pp": [100 * lo, 100 * hi],
                                                     "signflip_p": float(rob.signflip_p(v)), "n_blocks": int(len(v))}
    out["sonnet_low"]["au_minus_as_swapped"] = {str(r): 100 * float(contrast_values(swap, r).mean()) for r in (0.1, 0.9)}
    opus, _ = blocks_of(SWAP / "opus_active", OPUS)
    oorig, _ = blocks_of(ROOT / "results/frontier/scripted_opponent_campaign", OPUS)
    out["opus_AS_0.6"] = {"active_contract_rate_pct": rate(opus[("AS", 0.6)]),
                          "legacy_cell_rate_pct": rate(oorig[("AS", 0.6)])}
    (OUT / "contract_swap.json").write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
