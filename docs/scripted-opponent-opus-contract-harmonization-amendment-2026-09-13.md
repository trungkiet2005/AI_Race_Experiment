# Scripted-opponent Opus contract-harmonization amendment, 2026-09-13

This amendment declares one bounded replication of the Claude Opus 5 cell
collected under the legacy decoding contract. It is written before the new
request is made. The existing cell, its receipt, its raw output and its
legacy contract remain unchanged and are not overwritten.

## Purpose

The complete Claude Opus 5 scripted-opponent grid contains eleven cells under
the active contract and one cell, `AS@0.6`, under the earlier contract. The
replication asks the same route and the same frozen cell under the active
contract so the route-level result can be reported without mixing decoding
settings inside one route.

This is a contract-harmonization replication, not a replacement. If it fails,
the legacy cell remains valid evidence under its recorded contract and the
five-route campaign continues to disclose the heterogeneity. If it succeeds,
the new cell is kept in a separate provenance branch until the raw output,
cell audit and derived tables all pass.

## Frozen design

| Field | Declared value |
|---|---|
| Route | `anthropic/claude-opus-5@default` |
| Cell | `AS@0.6` |
| Protocol | `ai-race-scripted-opponent-v1` |
| Prompt version | `ai-race-fairgame-v3` |
| Repetitions | 10 races |
| Route decisions | 93 expected |
| Seat allocation | five races in each seat |
| Active decoding contract | `reasoning=low`, 1024-token cap |
| Account | `trungkiet` |

The game, prompt, parser, risk level, scripted rival, seed streams, stopping
rule, race count and seat allocation are unchanged. The only changed field is
the decoding contract, which is the object being harmonized.

## Collection rule

The baked one-cell source is generated from the canonical task by
`scripts/bake_scripted_opponent_cell.py`. A single diagnostic probe is allowed
before the full cell and is not evidence. The full cell is attempted only on
the declared account. A quota refusal retires this amendment and is recorded
as infrastructure failure; the cell is not moved to another account. A
transport congestion failure may be retried at most twice on the same account
with the same source.

The run is ingested only if it completes ten races, has zero parse failures,
uses both seats five times, carries the frozen mechanism hash for risk 0.6,
and passes scripted-rival replay. The old and new cells are then analysed side
by side before either can be used to revise a manuscript claim.

## Outcome

The declared task push reached task version 10 under `trungkiet`, but validation
returned HTTP 403 because the maximum estimated operation cost exceeded the
account's available quota. No race and no model output were produced. The
failure is recorded in
`results/failed_runs/scripted_opponent_opus_contract_harmonization_20260913.json`;
the existing legacy cell remains unchanged and is not silently replaced.
