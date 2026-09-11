# Scripted-opponent campaign: declared extension to three routes, 2026-09-11

Written and committed **before** any cell below is started, for the same reason
every other plan in this study was: once results begin to arrive, the assignment
must not be adjustable.

## What already exists, and what is wrong with it

`docs/scripted-opponent-collection-plan-2026-09-10.md` collected the whole
scripted-opponent grid, twelve cells, on one endpoint:
`google/gemini-3-flash-preview`. What it found is the strongest single result in
the study. Against a rival that always plays Safe the route plays Unsafe on
24.7, 17.2 and 14.0 per cent of its decisions at risk 0.1, 0.6 and 0.9; against a
rival that always plays Unsafe, on 100.0, 89.2 and 88.2 per cent. The rival's
stance is worth +73.5, +67.4 and +71.7 points, differenced inside a repetition so
the horizon draw is removed.

That finding currently rests on one checkpoint. A single endpoint cannot tell a
property of how language models play this game from a property of one commercial
route sampled on one afternoon, and the manuscript has to say so in every
sentence that cites it. Two more admitted routes remove that particular bound,
or fail to, and either outcome is reportable.

## What is being collected

The same frozen design, on two further routes admitted by
`results/frontier/admission_campaign_v6/`:

- `anthropic/claude-sonnet-5@default`, admitted at 85.0 per cent overall;
- `openai/gpt-5.4-2026-03-05`, admitted at 90.0 per cent overall.

Nothing about the design changes. Protocol stays `ai-race-scripted-opponent-v1`;
the four reduced strategies stay `AS`, `AU`, `CS`, `CAS`; the three risk levels
stay 0.1, 0.6 and 0.9; ten repetitions per cell; the route's seat counterbalanced
five and five; the prompt byte-identical to the neutral baseline's and still
silent about the rival being scripted. `scripts/verify_scripted_opponent_design.py`
was re-run before this plan was written and all fifteen of its checks pass.

Both routes accept the reasoning-budget argument. Their neutral-baseline
manifests in `results/frontier/baseline_campaign_v6/` record
`decoding.reasoning_requested = "none"`, so neither falls under the 2026-09-09
amendment that omits the parameter, and their outgoing requests carry the same
decoding contract as the Gemini cells.

One consequence of the task's own hashing rule is worth stating in advance,
because it is a check rather than a coincidence. When the benchmark server
executes a task it does not expose it as a file, so `task_source_sha256()` falls
back to hashing the canonical contract: prompt template, minimum rounds, stop
probability, the collected risk level, prize, progress, stage payoffs and prompt
version. That value depends on the risk level and on nothing else. Every cell
collected here must therefore carry exactly the `source_sha256` its Gemini
counterpart already carries:

| Risk | Required `source_sha256` |
|---|---|
| 0.1 | `d0a4d70f78c107263557675688171835597372c2c76454a4e414f6f33594fff4` |
| 0.6 | `4a460bac224e3f7c6e6292a531bb0b0cf3da5e6305ab684006323a67b344189a` |
| 0.9 | `6e6b1c92eda1d32677cb49fc2f00bedeeb473ca5b8d4619bfcf377a15a059683` |

A mismatch means the mechanism moved, and the cell is a failure record rather
than a result.

## Cost, measured rather than estimated

From the artefacts already in the tree: one cell is 10 races and **93 route
decisions**, and one route is 12 cells, 2,232 turn rows of which **1,116 are
route decisions**. The rival is executed by the task file rather than called, so
it costs no requests. Parse retries add a few requests at most; every Gemini cell
recorded zero parse failures and zero retries.

Two routes are therefore about 2,232 requests. A Model Proxy identity carries on
the order of 800 to 1,400 requests, so **one route is roughly one identity's
whole budget** and the extension does not fit on one account. As everywhere else
in this study, the workload is partitioned into the analysis unit itself, one
(route, strategy, risk) cell, and each cell is collected whole on one declared
identity.

## Quota probe, run before this plan was written

Every identity was probed with the one-request `connectivity-ping` task on both
target routes, 2026-09-11, before any cell was assigned:

| Identity | `claude-sonnet-5-default` | `gpt-5.4-2026-03-05` |
|---|---|---|
| `foundnotkiet` | completed | completed |
| `kit567` | completed | completed |
| `hunhtrungkit` | completed | completed |
| `tnkiet` | completed | completed |
| `trungkiet` | completed | completed |

`daosyduyminh` is not probed and not used: no credential for it is configured on
this workstation. Five identities are available, which is why the assignment
below is a five-way rotation rather than the four-way rotation the Gemini plan
used.

## Assignment

Fixed now, before the first push. Twenty-four cells, five identities, five cells
each except `trungkiet` which takes four. About 465 requests per identity, 372
for `trungkiet`.

### `anthropic/claude-sonnet-5@default`

| Cell | risk 0.1 | risk 0.6 | risk 0.9 |
|---|---|---|---|
| Always Safe (`AS`) | `foundnotkiet` | `kit567` | `hunhtrungkit` |
| Always Unsafe (`AU`) | `tnkiet` | `trungkiet` | `foundnotkiet` |
| Conditional Safe (`CS`) | `kit567` | `hunhtrungkit` | `tnkiet` |
| Conditional Unsafe (`CAS`) | `trungkiet` | `foundnotkiet` | `kit567` |

### `openai/gpt-5.4-2026-03-05`

| Cell | risk 0.1 | risk 0.6 | risk 0.9 |
|---|---|---|---|
| Always Safe (`AS`) | `hunhtrungkit` | `tnkiet` | `trungkiet` |
| Always Unsafe (`AU`) | `foundnotkiet` | `kit567` | `hunhtrungkit` |
| Conditional Safe (`CS`) | `tnkiet` | `trungkiet` | `foundnotkiet` |
| Conditional Unsafe (`CAS`) | `kit567` | `hunhtrungkit` | `tnkiet` |

The two tables are the same rotation shifted by two positions, which is what
makes them jointly interpretable. Within a route, every strategy is collected on
three different identities and every identity sees three different strategies, so
an account effect would appear as an inconsistency between cells rather than
hide inside a strategy contrast. Across routes, no (strategy, risk) cell is
collected on the same identity for both routes, so the route comparison is not
confounded with the account either. That second property is the one the Gemini
plan had no need of and this one does.

The identity remains a billing boundary and not a model boundary: route, prompt,
parser, protocol, temperature request and seed structure are identical in every
cell. The rotation makes that assumption checkable instead of merely asserted.

Every cell records its collecting identity in a `collection_receipt.json`,
because the benchmark server exposes neither `KAGGLE_USERNAME` nor
`KAGGLE_KERNEL_RUN_OWNER` to task code and the run manifest's own field reads
`unrecorded`.

## Order of collection

By risk level, each level complete across both routes before the next begins:

1. risk 0.6, four strategies on Claude Sonnet 5, then four on GPT-5.4;
2. risk 0.9, the same;
3. risk 0.1, the same.

The order is chosen so that an early stop still leaves something whole. After
the first block both new routes have a complete four-strategy grid at the middle
risk level, which is enough to say whether the rival's stance moves them, and the
analyser reports a risk level that is complete without waiting for the ones that
are not. Risk 0.6 is the middle of the frozen grid and is named here for that
reason and no other.

## Where the results go, and why not into the existing directory

Each route gets **its own directory and its own derived artefact**. Cells land in
`results/frontier/scripted_opponent_campaign/<STRATEGY>_risk<RISK>/<route-tag>/`,
beside the Gemini cells rather than on top of them, and the derived tables for a
new route are written to
`results/derived/scripted_opponent_campaign/<route-tag>/`.

This is not tidiness. `scripts/analyze_scripted_opponent.py` keyed its receipts
by `(strategy, risk)` alone, so a second route dropped into the same tree would
have been silently pooled with the first: a table reading 24 cells where the
manuscript says 12, and one route's rate averaged with another's under a single
heading. The same hazard is already recorded for the neutral baseline, which is
why `results/frontier/baseline_replication/` sits outside its campaign tree.
Rather than move the data, the analyser now takes one route at a time and
refuses to mix them.

`scripts/analyze_scripted_opponent.py` with no argument continues to report the
Gemini route alone, writing the same file with the same numbers, so the twelve
cells the manuscript cites cannot move because this extension exists. Each cell's
bootstrap generator is derived from that cell alone, and the Gemini route's
generator keys are frozen exactly as they were.

## Stopping rule

Unchanged from the rest of this study, and it binds.

- A cell refused on quota, HTTP 403, is recorded as a failure record under
  `results/failed_runs/` and is **not** reattempted on another identity. The
  identity that refused is retired for the rest of this collection and its
  remaining cells are recorded as not collected.
- A cell that fails on transport congestion, HTTP 429 or a timeout, may be
  retried on **its own declared identity**, at most twice. That is a retry and
  not a rotation: nothing about the assignment moves because a run failed.
- A cell that completes but fails ingestion, on a parse failure, a wrong seat
  balance, a wrong race count, a wrong protocol id, or a single deviation by the
  scripted rival, is a failure record and is not repaired by hand.
- A route missing any cell is reported as the cells that exist, naming the ones
  that do not. A partial route is never presented as a grid.

## Two results named in advance

Neither can then be presented as a surprise.

If the new routes reproduce the Gemini pattern, a low rate against the safe rival
and a high one against the unsafe rival with the difference excluding zero, then
the finding is a property of more than one endpoint and the manuscript can say
so with two more checkpoints behind it.

If a new route plays at similar rates against Always Safe and Always Unsafe, then
its behaviour is not driven by the rival's stance, the Gemini result does not
generalise across endpoints, and the manuscript reports the failure to replicate
as the finding it is. A route that plays Unsafe almost always, or almost never,
whatever the rival does, is also possible: a degenerate policy is a result about
that route and not a broken cell, and it is reported as a rate with an interval
of zero width described as an absence of variation rather than as precision.

A third possibility is worth naming because it is the easiest to mishandle. The
routes may agree in direction and differ in level. That is a replication of the
mechanism and not of the number, and the manuscript must say which of the two it
is claiming.

## Naming

The arm in which the rival always plays Safe is **not** called exploitation, here
or in any table this plan produces. On the Gemini route it carries the lowest
rates in the campaign, well below self-play, so it measures restraint kept rather
than an opportunity taken. The contrast is named for what it compares: how far
the rival's stance moves the route.
