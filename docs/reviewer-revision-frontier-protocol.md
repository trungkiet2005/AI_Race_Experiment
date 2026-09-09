# Reviewer revision protocol: frontier-only evidence

## Purpose

This protocol addresses the two first-round reviews without silently promoting
old open-weight results. New behavioural claims use only frontier endpoints
queried through Kaggle Benchmarks. Local and open-weight runs remain historical
diagnostics and are not pooled with the revision evidence.

The existing mechanism, canonical prompt, parser, seed streams, and outcome
definitions stay fixed. A changed prompt, decoding contract, model revision, or
analysis rule creates a new protocol and cannot be merged with the earlier runs.

## Review-to-evidence matrix

| Review concern | Evidence required before a claim | Planned artefact |
|---|---|---|
| Audit was run on one checkpoint | Per-endpoint rule, state, terminal, and expected-payoff audit | `ai-race-frontier-admission` task outputs |
| Opaque P/Q mapping was confounded | Mapping x context x repetition counterbalance, fixed-state replay, comprehension gate | frontier context task and paired replay tables |
| Rank effects are selected, not causal | Condition on prior own action and progress within rank strata; label as association | N-player conditional analysis |
| Small confirmatory cells | Frozen race count, complete-cell receipt, Wilson intervals, and race-level clusters | frontier baseline and N-player manifests |
| HDBSCAN/t-SNE sensitivity | Prespecified parameter grid and seed-stability summaries | trajectory robustness report |
| Decoding/model drift | Route, revision, effective temperature, token cap, seed status, and prompt hash in every manifest | task manifests and admission ledger |
| Missing cells/model naming | One canonical model registry used by tables and figures | `docs/frontier-model-registry.json` |
| Persona versus formatting | Neutral placebo, persona text hashes, and within-batch persona contrasts | frontier persona task |
| Qualitative EGT comparison | Same horizon law, same payoff rule, and a parameter sweep against theory | EGT comparison tables and figure |
| Parse failures and fallback policy | Raw responses, retry counts, contaminated-race counts, and no silent fallback | run manifests and QA report |
| Language dependence | English audit and, only if the route supports it, matched translated audit with the same probes | language sensitivity appendix |
| Reproducibility | Source hash, prompt hash, exact task source, raw outputs, and clean-room analysis | release manifest |

## Gate ladder

1. Local syntax and synthetic mechanism tests.
2. One representative frontier probe per route.
3. Frontier admission task for every route used in behavioural analysis.
4. Frontier neutral baseline with matched environment seeds.
5. Frontier representation and N-player tasks only for admitted routes.
6. Independent download validation and analysis.
7. Manuscript and figure rewrite from the new derived tables.

An endpoint that fails admission is retained as a failure record and excluded
from behavioural headline claims. A transport error, empty response, or
incomplete cell is not model evidence.

## Frontier endpoint set

The initial candidate set is the exact route list advertised to the active
Kaggle Benchmark identity: `openai/gpt-5.4-nano-2026-03-17`,
`anthropic/claude-sonnet-5@default`, `google/gemini-3-flash-preview`,
`google/gemini-3.1-flash-lite-preview`, `deepseek-ai/deepseek-r1-0528`,
`qwen/qwen3-next-80b-a3b-instruct`, `openai/gpt-oss-120b`, and
`ibm/granite-4.0-h-small`. The exact route selected for each run is recorded in
the task manifest. The route list is a candidate set, not a promise that all
routes are reachable or admitted.

## Account policy

The local environment currently exposes one Kaggle Benchmark identity,
`daosyduyminh`. No collaborator credential root is configured. Runs therefore
remain on that identity until explicitly authorised credential files are
provided. Account rotation must never be used to bypass a quota or private-input
restriction; separate authorised accounts must receive separate task versions
and separate output directories.

## Protocol amendments

Amendments are recorded here rather than applied silently, so a reader can see
what changed, when, and which artefacts each version covers.

### 2026-09-09 - collaborator identities authorised, with per-identity separation

The account policy above was written when only `daosyduyminh` was configured and
it therefore said no collaborator credential root existed. Five collaborator
credentials are in fact available, and their use for this project is now
explicitly authorised by the author.

The prohibition that matters is unchanged and is not about which account runs a
job. It is that a quota refusal must never be laundered into evidence: a run
that a quota stopped is a failure record, and a second identity may not be used
to make an incomplete sample look complete. Concretely:

- A run records its executing identity in the manifest, and any table that
  pools runs from more than one identity must say so.
- A second identity receives its own task version and its own output directory.
  Artefacts from different identities are never written into the same campaign
  directory as though they were one sample.
- A route whose run failed on quota under one identity and completed under
  another keeps BOTH records. The failure is not deleted because a later
  attempt succeeded.
- Rotating until a run happens to pass is not permitted. The repetition count,
  the risk grid and the exclusion rules are fixed before the run, and a rerun
  uses the same frozen numbers.

The reason to allow a second identity at all is that a per-account token
reservation limit is a property of the billing account, not of the model or the
protocol, so refusing to switch would let an accounting limit decide which
scientific questions get answered.

### 2026-09-09 - route-resolved reasoning budget

Both frontier tasks previously passed `reasoning="none"` to every route. Two
route families reject the reasoning-budget *argument itself* rather than its
value: naming `reasoning` at all returns HTTP 400 "Request contains an invalid
argument" before a single probe or decision is sampled. That is a transport
contract mismatch, not evidence about the model, and it silently removed
`google/gemini-3.5-flash-lite` from the audit even though the same route answers
the identical probe bank in other tasks on this identity.

The tasks now resolve the reasoning budget per route and omit the parameter
entirely for the routes that refuse it, listed in the task source. Nothing else
changed: the mechanism, canonical prompt, probe bank, parser, seed streams,
temperature, and token cap are untouched.

Scope of the amendment:

- For every route whose resolved contract is still `reasoning="none"`, the
  outgoing request is byte-identical to the earlier runs, so those artefacts
  stay poolable with the earlier task versions.
- For a route whose resolved contract is `reasoning=null`, the manifest records
  that value. The difference is therefore visible in the artefact rather than
  inferred, and any table that pools such a route must say so.

Affected task versions: `ai-race-frontier-admission` version 8 and
`ai-race-baseline` version 4 onward.

## Stopping rule

The revision does not rewrite the paper around a failed or partial campaign.
If the frontier admission, completeness, or independent verification gate
fails, the affected claim remains exploratory or is removed. The paper may
retain a limitation when the proposed experiment cannot identify the requested
effect.
