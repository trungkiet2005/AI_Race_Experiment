# Hosted route reachability diagnostic (2026-09-12)

This directory contains the raw outputs and source notebooks downloaded from
Kaggle Benchmark task `connectivity-ping`, task version 7, owned by the
`foundnotkiet` research profile.

The diagnostic completed for all five admitted model routes:

- Gemini 3 Flash (`gemini-3-flash-preview`)
- Claude Opus 5 (`claude-opus-5-default`)
- GPT-5.4 (`gpt-5.4-2026-03-05`)
- GPT-5.5 (`gpt-5.5-2026-04-23`)
- Claude Sonnet 5 (`claude-sonnet-5-default`)

The task sends only a one-prompt `PONG` request. It establishes route
reachability and preserves provenance; it is not an AI Race run and must not
be pooled with behavioural evidence or cited as a behavioural result. The
scripted-opponent completion remains limited to its three admitted route
grids, as documented in `docs/scripted-opponent-completion-plan-2026-09-11.md`.
