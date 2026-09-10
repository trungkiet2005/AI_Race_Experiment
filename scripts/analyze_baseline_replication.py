"""Compare a whole baseline run against an independent repeat of the same cell.

Every route-level number in the paper comes from one 30-race run, so a reader is
entitled to ask how much of it is the route and how much is the run.  One route
was collected twice by accident of scheduling: ``google/gemini-3-flash-preview``
ran at task version 3 on 2026-09-08 and again at task version 4 on 2026-09-09,
same frozen protocol, same seed structure, same prompt hash, different day and a
different billing identity.  That pair is a free run-to-run reproducibility
check, and this script is what makes it a measurement rather than an anecdote.

The repeat is deliberately stored outside ``baseline_campaign_v6/ai-race-baseline``
because both campaign analysers key their results by ``model_route``: a second
run of an already-represented route sitting inside that tree would not raise an
error, it would quietly replace the run the manuscript reports.

Fail-closed.  It refuses unless both runs carry the same protocol, prompt hash,
mechanism and seed, both completed, both are free of parse failures, and both
cover the same set of (game seed, risk) blocks -- because the comparison is only
a reproducibility statement if the two runs were asked the same question.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REFERENCE = (
    ROOT / "results" / "frontier" / "baseline_campaign_v6" / "ai-race-baseline" / "3"
    / "gemini-3-flash-preview" / "1395291" / "results" / "ai_race_baseline"
)
REPEAT = ROOT / "results" / "frontier" / "baseline_replication" / "gemini-3-flash-preview"
OUT = ROOT / "results" / "derived" / "baseline_replication.json"
ROUTE = "google/gemini-3-flash-preview"
PROTOCOL = "ai-race-frontier-baseline-v3"
RISKS = (0.1, 0.6, 0.9)
EXPECTED_DECISIONS = 558
EXPECTED_RACES = 30

INVARIANT = ("protocol_id", "prompt_sha256", "prompt_version", "seed", "model_route",
             "n_races", "repetitions_per_risk", "run_phase")


def load(run: Path) -> dict:
    manifest = json.loads((run / "run_manifest.json").read_text(encoding="utf-8"))
    if manifest.get("status") != "completed":
        raise SystemExit(f"{run} did not complete: {manifest.get('status')}")
    if manifest.get("protocol_id") != PROTOCOL:
        raise SystemExit(f"{run} is protocol {manifest.get('protocol_id')!r}, not {PROTOCOL}")
    if manifest.get("model_route") != ROUTE:
        raise SystemExit(f"{run} is route {manifest.get('model_route')!r}, not {ROUTE}")

    counts: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    blocks: set[tuple[int, float]] = set()
    parse_failures = 0
    for line in (run / "turns.jsonl").read_text(encoding="utf-8").splitlines():
        turn = json.loads(line)
        risk = float(turn["max_private_risk"])
        counts[risk][1] += 1
        if str(turn.get("action", "")).upper().startswith("UNSAFE"):
            counts[risk][0] += 1
        if turn.get("parse_failed"):
            parse_failures += 1
        blocks.add((turn["game_seed"], risk))
    if parse_failures:
        raise SystemExit(f"{run} carries {parse_failures} parse failures")
    decisions = sum(c[1] for c in counts.values())
    if decisions != EXPECTED_DECISIONS or sorted(counts) != list(RISKS):
        raise SystemExit(f"{run} has {decisions} decisions over risks {sorted(counts)}")
    return {
        "manifest": manifest,
        "rates": {r: 100.0 * counts[r][0] / counts[r][1] for r in RISKS},
        "decisions": {r: counts[r][1] for r in RISKS},
        "blocks": blocks,
    }


def main() -> None:
    ref, rep = load(REFERENCE), load(REPEAT)

    for field in INVARIANT:
        if ref["manifest"].get(field) != rep["manifest"].get(field):
            raise SystemExit(
                f"runs disagree on {field!r}: {ref['manifest'].get(field)!r} vs "
                f"{rep['manifest'].get(field)!r}; they are not repeats of one cell"
            )
    if ref["manifest"]["mechanism"] != rep["manifest"]["mechanism"]:
        raise SystemExit("runs disagree on the mechanism, so they are not repeats of one cell")
    if ref["blocks"] != rep["blocks"]:
        raise SystemExit(
            f"runs cover different (game seed, risk) blocks: "
            f"{len(ref['blocks'] ^ rep['blocks'])} blocks differ"
        )
    if len(ref["blocks"]) != EXPECTED_RACES:
        raise SystemExit(f"expected {EXPECTED_RACES} blocks, found {len(ref['blocks'])}")

    per_risk = {
        f"{r}": {
            "reference_pct": round(ref["rates"][r], 4),
            "repeat_pct": round(rep["rates"][r], 4),
            "difference_pp": round(rep["rates"][r] - ref["rates"][r], 4),
            "decisions": ref["decisions"][r],
        }
        for r in RISKS
    }
    resp_ref = ref["rates"][0.1] - ref["rates"][0.9]
    resp_rep = rep["rates"][0.1] - rep["rates"][0.9]
    largest = max(abs(v["difference_pp"]) for v in per_risk.values())

    result = {
        "schema_version": "baseline-replication-v1",
        "route": ROUTE,
        "protocol_id": PROTOCOL,
        "what_is_held_fixed": sorted(INVARIANT) + ["mechanism", "game seed blocks"],
        "what_differs": [
            "task version 3 versus 4",
            "collection date 2026-09-08 versus 2026-09-09",
            "sampling is not reproducible on this route: the SDK strips the seed "
            "request, which the manifest records as "
            f"{ref['manifest']['sampling_seed_provenance']['status']!r}",
        ],
        "reference_run": str(REFERENCE.relative_to(ROOT)).replace("\\", "/"),
        "repeat_run": str(REPEAT.relative_to(ROOT)).replace("\\", "/"),
        "reference_completed_utc": ref["manifest"]["completed_utc"],
        "repeat_completed_utc": rep["manifest"]["completed_utc"],
        "races_per_risk_level": EXPECTED_RACES // len(RISKS),
        "per_risk": per_risk,
        "largest_absolute_difference_pp": round(largest, 4),
        "risk_response_reference_pp": round(resp_ref, 4),
        "risk_response_repeat_pp": round(resp_rep, 4),
        "risk_response_difference_pp": round(resp_rep - resp_ref, 4),
        "caveat": (
            "One route, one repeat.  This bounds run-to-run variation for this "
            "route at this sample size; it does not license the same bound for "
            "routes that were collected once."
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")

    print(f"route {ROUTE}, {EXPECTED_DECISIONS} decisions per run, seed blocks identical")
    for r in RISKS:
        v = per_risk[f"{r}"]
        print(f"  risk {r}: {v['reference_pct']:6.1f}% -> {v['repeat_pct']:6.1f}%  "
              f"({v['difference_pp']:+.1f} pp over {v['decisions']} decisions)")
    print(f"  largest per-risk difference {largest:.1f} pp")
    print(f"  risk response {resp_ref:.1f} pp -> {resp_rep:.1f} pp "
          f"({resp_rep - resp_ref:+.1f} pp)")
    print(f"  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
