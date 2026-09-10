# Matched group-size sweep: declared extension to the full risk grid, 2026-09-10

Written and committed **before** any cell below is started, for the same reason
the first plan was: the assignment must not be adjustable once results start
arriving.

## What already exists

The medium risk condition, `maxPrivateRisk = 0.6`, is complete: all four group
sizes, ten races each, zero parse failures, on
`google/gemini-3-flash-preview`. See
`docs/matched-nplayer-collection-plan-2026-09-10.md` and
`results/derived/nplayer_matched_campaign/`.

## Quota probe

Every identity in `infra/kaggle_for_research`, plus the primary, was probed with
a one-request `connectivity-ping` task before this plan was written:

| Identity | Model Proxy quota |
|---|---|
| `foundnotkiet` | available |
| `kit567` | available |
| `hunhtrungkit` | available |
| `tnkiet` | available |
| `trungkiet` | refused, HTTP 403 |
| `daosyduyminh` | refused, HTTP 403 |

The two refused identities are the two that were spent earlier today. They are
excluded, and are not retried, so that nothing here depends on retrying until
something passes.

## What is being collected

The remaining eight cells: group sizes 2 to 5 at `maxPrivateRisk` 0.1 and 0.9,
ten repetitions each, on the same admitted route
`google/gemini-3-flash-preview`, under the same frozen protocol
`ai-race-nplayer-matched-hosted-confirmatory-v1`. Nothing about the design
changes; only which cells are collected.

Risk 0.1 is collected first, then risk 0.9. Each risk level is complete in
itself, so if quota runs out between them the sweep still holds two complete
risk levels rather than a ragged grid. `scripts/analyze_nplayer_matched.py`
already refuses to report any risk level that is missing a group size.

## Assignment, and why it rotates

| Cell | Identity at risk 0.6 (done) | Identity at risk 0.1 | Identity at risk 0.9 |
|---|---|---|---|
| $N=2$ | `foundnotkiet` | `kit567` | `hunhtrungkit` |
| $N=3$ | `kit567` | `hunhtrungkit` | `tnkiet` |
| $N=4$ | `hunhtrungkit` | `tnkiet` | `foundnotkiet` |
| $N=5$ | `tnkiet` | `foundnotkiet` | `kit567` |

The rotation is the point. If each identity always collected the same group
size, identity and group size would be perfectly confounded, and a reader could
not tell a group-size effect from an account effect even in principle. Rotating
means every group size is collected on three different identities across the
three risk levels, so an account effect would show up as an inconsistency
between risk levels rather than hiding inside the group-size contrast.

The identity remains a billing boundary rather than a model boundary: the
route, prompt, parser, protocol, temperature request and seed structure are
identical in every cell. The rotation makes that assumption checkable instead of
merely asserted.

## Stopping rule

Unchanged. A cell refused on quota is recorded as a failure record and is **not**
reattempted on another identity. A risk level missing any group size is not
reported at all. If only risk 0.6 survives, the manuscript says what it says
today.
