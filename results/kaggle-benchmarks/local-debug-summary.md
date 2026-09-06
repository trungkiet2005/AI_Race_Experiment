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


## 2026-08-02 batch

A second batch of 61 CLI artifacts had accumulated at the repository root under
the same filenames. They are now under `_local-debug/2026-08-02/`, ignored by Git
like the first batch. Of the 40 `.run.json` exports, 23 completed and 17 errored;
every failure is infrastructure, not a result:

| Task | Completed | Errored | What the failures were |
|---|---|---|---|
| `mtbench-all-models-schema-smoke-v1` | 2 | 16 | provider-side, see below |
| `mtbench-all-models-target-lane0-v1` (+ 8 recovery shards) | 9 | 1 | provider-side |
| `mtbench-all-models-target-lane1-v1` (+ 8 recovery shards) | 9 | 0 | — |
| `mtbench-all-models-scientific-smoke-v1` | 2 | 0 | — |
| `connectivity-ping` | 1 | 0 | — |

Terminal exception across all 17 failures:

| Count | Exception |
|---|---|
| 12 | `openai.InternalServerError` 503, "The requested model is currently unavailable" |
| 2 | `openai.RateLimitError` 429, "The model is currently experiencing heavy load" |
| 2 | `openai.LengthFinishReasonError`, response hit the completion-token limit |
| 1 | `openai.AuthenticationError` 401, "Authorization failed: expired token" |

The `recovery-s00`..`s07` shards are re-runs of the two target lanes after those
503s, which is why the lane tasks appear once per shard. None of this is admitted
behavioural evidence; it is transport diagnostics for the benchmark harness.
