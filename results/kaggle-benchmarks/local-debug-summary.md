# Local Kaggle Benchmark smoke summary — 2026-07-31 to 2026-08-01

This file preserves the useful outcome of the Kaggle CLI `.task.json` and
`.run.json` artifacts formerly stored at the repository root. The raw exports
remain available locally under `_local-debug/2026-08-01/` and are ignored by Git.

| Task | Model route | Outcome | Interpretation |
|---|---|---|---|
| `debug-llm-type` | `google/gemini-3-flash-preview` | Completed | Minimal backend/type probe succeeded. |
| `connectivity-ping` | `google/gemini-3-flash-preview` | Completed | Hosted model returned a valid response. |
| `connectivity-ping` | `llm` | Errored | Local Model Proxy credentials were missing (`MODEL_PROXY_URL`, `MODEL_PROXY_API_KEY`). |
| `ai-race-baseline` | `llm` | Errored | Backend was unknown, so the task correctly refused to run without an explicit 256-token-limit parameter. |

These are connectivity and contract diagnostics, not admitted behavioural results.
Canonical Kaggle Benchmark outputs belong in the versioned task/model directories
beside this summary.
