# Scripted-opponent expanded account-pool amendment, 2026-09-12

This is a new amendment, written before the next model-output request. It
extends the resumption only to five additional Kaggle Benchmark identities
whose raw API tokens under `infra/kaggle_for_research/kaggle-api-3` were
successfully exchanged for Benchmark credentials. It does not silently rotate
any cell inside the earlier five-profile amendment. Earlier failures and
successful cells remain unchanged.

## Verified identities

| Token file | Benchmark identity | Status |
|---|---|---|
| `acc06.txt` | `minh2duy` | eligible |
| `acc07.txt` | `boymagic` | eligible |
| `acc08.txt` | `trngthtnhi` | eligible |
| `acc09.txt` | `osduyminh` | eligible |
| `acc10.txt` | `trnnguynchis` | excluded: Benchmark auth returned HTTP 403 for missing verification or stale credentials |
| `acc11.txt` | `trngbotrn` | eligible |

The raw tokens and generated `benchmark.env` files stay outside the repository.
No token, key, or expiry secret may enter a receipt, log, commit, or paper.

## Frozen contract

The game, prompt bytes, parser, scripted rival, seed streams, risk grid,
repetitions, seat balance and protocol remain those in
`ai-race-scripted-opponent-v1`. The active decoding contract remains 1024
OpenAI-compatible `max_tokens` with `reasoning="low"` for both missing routes.
Each uploaded source is one baked cell generated from the canonical task by
`scripts/bake_scripted_opponent_cell.py`.

This amendment changes the collecting identity and therefore is a provenance
amendment, not an invisible equivalent of the earlier collection. New cells
are kept under the same route/cell evidence tree with an identity receipt, and
all route-level comparisons disclose the identity boundary and the decoding
contract.

## Fixed assignment for the remaining cells

The assignment is fixed before the first probe/full-cell attempt under this
amendment. A 403 retires the identity for this amendment and leaves its cells
missing. No cell moves to another identity after a refusal. A 429 or timeout
may be retried twice on the same assigned identity only.

### Claude Opus 5

| Cell | Assigned identity |
|---|---|
| `AS@0.9` | `minh2duy` |
| `AU@0.1` | `boymagic` |
| `CAS@0.1` | `trngthtnhi` |

### GPT-5.5

| Cell | Assigned identity |
|---|---|
| `AS@0.6` | `osduyminh` |
| `AU@0.6` | `trngbotrn` |
| `CS@0.6` | `minh2duy` |
| `AU@0.9` | `boymagic` |
| `CS@0.9` | `trngthtnhi` |
| `CAS@0.9` | `osduyminh` |
| `AU@0.1` | `trngbotrn` |
| `CS@0.1` | `minh2duy` |
| `CAS@0.1` | `boymagic` |

The five eligible identities are first tested with the workload-compatible
structured-output probe for each route they own. A passing probe authorises an
attempt on its preassigned cells but is not evidence. A route is upgraded to a
complete grid only after all twelve cells pass the existing fail-closed ingest
audit.

`acc10` is not used as a fallback: its 403 is an account-verification failure,
and no raw token-only file is treated as a Benchmark identity without a valid
Benchmark auth exchange.
