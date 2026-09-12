# Frontier evidence collection protocol

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

- The ingestion receipt records the operator-supplied executing identity; the
  task manifest cannot server-attest that field. Any table that pools runs
  from more than one identity must say so.
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

## Matched group-size sweep: specified, verified, and quota-blocked

The three- to five-player pilots lack a matched two-player condition, which is a
stated limitation of the manuscript. The design that would remove it exists and
is frozen:

- Task `kaggle/benchmarks/ai_race_nplayer_matched.py`, protocol
  `ai-race-nplayer-matched-hosted-confirmatory-v1`, group sizes 2 to 5 crossed
  with the three risk treatments at ten repetitions per cell.
- Verified offline by `scripts/verify_matched_nplayer_design.py`: the mechanism
  outside the group-count rule is identical at every group size; the two-player
  arm reproduces the headline payoff matrix exactly at 1.0 / 0.6 / 2.4 / 2.0, so
  it is the paper's own game rather than a near neighbour; and one repetition
  index shares a single horizon stopping-draw stream across every risk and every
  group size, so the contrast can be differenced within a repetition.
- Analysed by `scripts/analyze_nplayer_matched.py`, written before any data
  existed so the estimand was fixed in advance, and fail-closed on a missing
  arm, a parse failure, an unbalanced cell or a wrong seat count.

It has produced no data. Both authorised identities refused it with HTTP 403 on
the same day, `daosyduyminh` immediately and `trungkiet` after about 48 minutes
of sequential requests, so an identity here carries on the order of 800 to 1,000
requests while the frozen sweep needs roughly 7,800. The two failure records are
`results/failed_runs/nplayer_matched_daosyduyminh_20260910.json` and
`results/failed_runs/nplayer_matched_trungkiet_20260910.json`.

Three further authorised identities exist and were deliberately not tried.
Rotating until one happens to complete selects on the outcome and would build a
single sample out of several billing accounts mid-run, which the
collaborator-identity amendment above forbids. Shrinking the design to fit is
not available either: at two or three repetitions every cell falls below the
five-independent-race floor, which makes the cells descriptive only and so
cannot remove the limitation the design exists to remove.

Until it runs, the manuscript keeps the limitation as written. A specified and
unexecuted design is not evidence, and must never be reported as an attempted or
inconclusive result.

## 2026-09-10 - the matched sweep, resumed by partitioning it

The section above records the sweep as blocked, and that record stands as
written: it was true when written, and the two failure records it names remain
in the tree. What follows supersedes its conclusion, not its account.

The blockage was resolved without rotating on failure. Rather than retrying the
whole sweep until an identity happened to survive it, the workload was
partitioned into the analysis unit itself, one (group size, risk) cell of ten
repetitions, roughly 370 to 930 requests, which fits inside a single identity's
quota. Each cell is therefore complete in itself and carries its own
race-clustered interval; no cell is stitched together from more than one
account. The assignment of cells to identities was fixed in a plan committed
before the first cell was started
(`docs/matched-nplayer-collection-plan-2026-09-10.md`, extended by
`docs/matched-nplayer-collection-plan-2026-09-10-extension.md`), and the
extension rotates the assignment so that no identity is confounded with a group
size. Every cell records its collecting identity in a
`collection_receipt.json`, because the benchmark server exposes no identity
variable to task code.

What this produced: `maxPrivateRisk = 0.6` and `0.1` both complete at all four
group sizes, reported in the supplement. The three-player cell at risk 0.1 first
failed twice with HTTP 429 route congestion, retained as
`results/failed_runs/nplayer_matched_n3_risk0p1_hunhtrungkit_20260910.json`, and
was later collected on its assigned identity once the route was no longer
congested. That is a retry of the same cell on the same declared identity, not a
rotation: nothing about the assignment moved because a run failed. Risk 0.9 was then collected the same way and
completed the grid: 12 cells, 120 races, 3,696 decisions, zero parse failures,
every cell whole on a single declared identity.

The grid answers a question a single risk level could not. The three- and
four-company steps are +11.1 and +28.2 points at risk 0.6 and +11.0 and +28.1
at risk 0.9, agreeing to about a tenth of a point across conditions collected on
different days and different identities. The five-company step differs, +31.0
against +40.0, because the risk-0.6 five-company cell is at 100 per cent and its
contrast is truncated by the ceiling; the risk-0.9 figure is the one to quote for
that step.

Two facts about risk 0.1 belong in the record rather than only in the paper.
Every cell sits at 100 per cent unsafe, so every contrast is exactly zero and
every bootstrap interval has zero width. That is an absence of variation in the
sample, not precision, and it means the route is already at the ceiling before a
competitor is added rather than that group size stops mattering at low risk.
The same route reaches 98.9 and 100.0 per cent unsafe at risk 0.1 in the two
independent neutral-baseline runs, collected under a different protocol on a
different engine, so the ceiling is a property of the route.

One reproducibility defect was found and fixed while collecting this level. The
analyser threaded a single bootstrap generator through every cell, so each
interval depended on how many cells had been analysed before it, and adding risk
0.1 moved the already-reported risk-0.6 intervals by up to 0.2 points. Each cell
now derives its generator from a digest of its own identity (`cell_rng`), so a
cell's interval is reproducible whatever else is in the tree, and the supplement
carries the corrected values.

## 2026-09-10 - archive reconciliation

Every Kaggle Benchmark task version on every authorised identity was downloaded
and reconciled against the repository by content hash. All matched-sweep cells
were already stored. Two completed runs were not:

- A second thirty-race baseline run of `google/gemini-3-flash-preview` at task
  version 4, which rode along with the reasoning-budget amendment push. It is a
  genuine independent repeat of a reported cell, since sampling on these routes
  is not reproducible, and is now reported as a run-to-run reproducibility check
  in the supplement. It is stored at `results/frontier/baseline_replication/`,
  **outside** the campaign tree, because both campaign analysers key results by
  model route and would silently displace the reported run instead of raising.
- A nine-race pilot from 2026-08-01 whose manifest carries no `protocol_id` at
  all. It is retained under `results/frontier/pilots/` as provenance and is read
  by no analyser.

## 2026-09-10 - the scripted-opponent campaign

The largest gap in this study was never a missing route or a missing risk level.
It was that every gameplay result was self-play, so a route responding to its
rival could not be distinguished from a route running through a phase of its
own, and the evolutionary lane's four reduced strategies had never been played
against by anything.

The campaign puts `google/gemini-3-flash-preview` against those four strategies,
executed by the task file rather than called, which also means the rival costs no
requests: one cell is 93 route decisions and the whole grid is 1,116. Twelve
cells, each collected whole on one declared identity under
`docs/scripted-opponent-collection-plan-2026-09-10.md`, with the assignment
rotated so no identity is confounded with a strategy.

Three properties were asserted offline before any push, by
`scripts/verify_scripted_opponent_design.py`, and all fifteen of its checks pass.
The prompt is byte-identical to the neutral baseline's and never says the rival
is scripted, so races here are comparable with races there. The route's seat is
counterbalanced five and five, because this study has measured a seat effect on
prompts differing only in which of two names is whose. And a conditional strategy
answers the route rather than itself, which is the check that separates a rival
from a mirror; a strategy that answered its own history would be a constant and
the whole conditional arm would measure nothing.

The rival is code, so it cannot be audited by reading a response. It is audited
by replay: `scripts/ingest_scripted_cell.py` recomputes every rival move from the
route's own moves and refuses the cell on a single deviation. All twelve cells
replayed clean. That is the one failure mode that would have left the data
looking perfectly healthy while answering a different question.

What it found is in `CLAUDE.md` and in the supplement. The part that bears on the
rest of the manuscript is that a self-play rate is not a measurement of how a
route treats risk. It is an equilibrium of two copies of one policy escalating
each other, and this campaign is what separates the two.

## Stopping rule

The revision does not rewrite the paper around a failed or partial campaign.
If the frontier admission, completeness, or independent verification gate
fails, the affected claim remains exploratory or is removed. The paper may
retain a limitation when the proposed experiment cannot identify the requested
effect.
