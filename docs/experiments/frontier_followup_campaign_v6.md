# Frontier follow-up campaign v6

Status: protocol amendment prepared before rerun. This campaign does not pool
with the v5 admission task or with the five-model baseline until every route
has an independently downloaded, validated admission artifact.

## Purpose

Close the per-endpoint task-validity limitation for the behavioural endpoints
used in `frontier_full_20260908/baseline_v3`. The v5 task exposed two classes of
failure that must be separated from model behaviour:

- some routes reject `reasoning="none"` and require an explicit low or high
  reasoning budget;
- successful server-side runs were not retrievable through the downloader.

The v6 source records the route-specific reasoning contract in each manifest.
The downloader failure remains a blocking artifact-retrieval issue and cannot
be repaired by relabelling a run.

## Frozen scope

The admission probe bank, prompt text, parser, thresholds, repetitions, and
temperature remain unchanged from v5. Only the documented route contract is
amended. The protocol identifier is therefore `ai-race-frontier-admission-v6`.

The behavioural baseline remains the separately frozen
`ai-race-frontier-baseline-v3` protocol. Its five locally complete routes are:

- Gemini 3 Flash Preview
- GPT-5.4 nano
- GPT-5.4 mini
- GPT-5.4
- GPT-5.5

## Route contract

The v6 task records the following requested reasoning values:

| Route family | Requested reasoning | Reason |
|---|---|---|
| Gemini 3.1 Pro | `high` | zero budget is rejected by the route |
| Qwen 3 Next Thinking | `high` | thinking route requires reasoning |
| Qwen 3 Next, Qwen 3 235B, GLM-5 | `low` | literal `none` is rejected |
| Gemma | omitted | route rejects the reasoning parameter |
| Other routes | `none` | preserve the v5 observable contract |

These are requested settings only. The manifest must continue to distinguish
requested, forwarded, and effective provider settings.

## Admission and promotion gates

An endpoint is eligible for behavioural promotion only when all of the
following are present in its own downloaded directory:

1. completed task manifest and run receipt;
2. complete raw responses for all frozen probes and repetitions;
3. valid and correct rates meeting the frozen domain thresholds;
4. no unaccounted transport, schema, or parse failure;
5. matching prompt, probe-bank, and protocol hashes;
6. an independent local validation receipt.

A server status without downloadable raw output is `blocked`, not admitted.
Transport errors, unsupported routes, empty structured output, and contract
errors are retained as failure records and never enter behavioural summaries.

## Storage contract

Every v6 download goes to a new immutable directory:

```text
results/kaggle-benchmarks/frontier_full_20260908/admission_v6/<route-tag>/
```

Derived admission tables go to:

```text
results/kaggle-benchmarks/frontier_full_20260908/derived/admission_v6/
```

The directory must retain the task source or source receipt, task and run
metadata, raw responses, admission summary, validation receipt, and SHA256
manifest. No file from v5 or baseline_v3 is overwritten. The campaign index
and results catalog are regenerated only after local validation completes.

## Stopping rule

If the downloader again omits successful run artifacts, stop promotion and
report the routes as blocked. Do not create a third protocol by changing the
probe bank, thresholds, parser, or output interpretation to make the campaign
look complete.
