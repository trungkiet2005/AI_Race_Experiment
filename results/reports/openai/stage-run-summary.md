# OpenAI stage run summary — 2026-08-01

This is the durable audit summary for the completed run formerly represented by
the repository-root `logs_openai_stage/` directory and `run_openai_stage.log`.
Raw console logs now live locally under `results/frontier/openai/_logs/stage/`
and are intentionally ignored by Git.

The maintained entry point is [`scripts/run_openai_stage.py`](../../../scripts/run_openai_stage.py).

## Outcome

- Configured jobs: 44.
- Already complete and skipped: 3 (`openai_baseline`,
  `openai_persona_baseline_neutral`, and
  `openai_persona_baseline_risk_averse`).
- Executed jobs: 41.
- Successful jobs: 41.
- Failed jobs: 0.
- Maximum parallel jobs: 2.
- Per-job duration: 3.9–37.0 minutes; median 5.1 minutes.

## Retry diagnostics

The raw logs contained 402 `rate-limited (429)` retry messages across 27 jobs.
All affected jobs ultimately completed. The two longest jobs were
`openai_persona_baseline_risk_4_2` (36.7 minutes) and
`openai_persona_baseline_risk_4_3` (37.0 minutes), consistent with extended
rate-limit backoff.

The retry-message count is a console-line count, not a count of distinct API
requests. Use the canonical run manifests and result tables for experimental
status and observations.

## Artifact policy

Canonical experiment artifacts remain under `results/frontier/openai/`, including
the run manifests, turns, races, players, and convenience result tables. Console
logs are useful for transient debugging but are not inputs to analysis and are
not required in a clean clone.
