# Scripted-opponent resumption assignment ledger, 2026-09-12

This ledger makes the deterministic assignment in the resumption amendment
explicit. It is an operational record, not a reclassification of any result.
The identity order is `foundnotkiet`, `kit567`, `hunhtrungkit`, `tnkiet`,
`trungkiet`. For GPT-5.5 the cell position uses offset 0; for Claude Opus 5 it
uses offset 1. Positions are ordered by risk blocks `0.6`, `0.9`, `0.1`, and
within a block `AS`, `AU`, `CS`, `CAS`.

## GPT-5.5

| Risk | AS | AU | CS | CAS |
|---|---|---|---|---|
| 0.6 | foundnotkiet | kit567 | hunhtrungkit | tnkiet |
| 0.9 | trungkiet | foundnotkiet | kit567 | hunhtrungkit |
| 0.1 | tnkiet | trungkiet | foundnotkiet | kit567 |

## Claude Opus 5

| Risk | AS | AU | CS | CAS |
|---|---|---|---|---|
| 0.6 | kit567 | hunhtrungkit | tnkiet | trungkiet |
| 0.9 | foundnotkiet | kit567 | hunhtrungkit | tnkiet |
| 0.1 | trungkiet | foundnotkiet | kit567 | hunhtrungkit |

The pre-existing Opus `AS@0.6` cell was not recollected. The eight new Opus
cells already ingested match this ledger. `foundnotkiet` was retired after a
workload-compatible 1024-token probe returned HTTP 403. `hunhtrungkit` was
retired after the assigned `CAS@0.1` task returned HTTP 403. Their assigned
cells remain missing; no cell is moved to another identity within this ledger.

The raw token-only files under `infra/kaggle_for_research/kaggle-api-3` are not
Benchmark profiles and are excluded because they contain no username,
`benchmark.env`, or auditable account mapping.
