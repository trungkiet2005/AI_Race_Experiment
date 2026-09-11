# Matched group-size sweep: declared collection plan, 2026-09-10

This plan is written **before** any of the runs below is started, and is committed
before the first push. That order is the point of the document: the cells, the
identity that collects each one, and the stopping rule are fixed in advance, so
nothing here can be chosen after seeing a result.

## Why the design is being collected in pieces

The frozen sweep is four group sizes by three risk treatments at ten repetitions
per cell, which is 120 races and roughly 7,800 model requests. Two authorised
Kaggle identities refused it with HTTP 403 on 2026-09-09 and 2026-09-10:
`daosyduyminh` immediately, `trungkiet` after about 48 minutes of sequential
requests. An identity here sustains on the order of one thousand requests. The
whole sweep does not fit on one identity, and it does not fit on the identities
available today either.

`kaggle benchmarks tasks run` has no way to pass an environment variable to the
server, so which cells a run collects has to be baked into the pushed source. A
cell is therefore one push and one run.

## What is being collected, and what is not

**Scope reduced before running, not after:** the group-size comparison is
collected at **one risk level, the medium condition `maxPrivateRisk = 0.6`**, for
all four group sizes. That is 4 cells, 40 races, roughly 2,600 requests, which
fits across four fresh identities at about one cell each.

The medium condition is chosen because it is the middle of the frozen grid, and
for no other reason. It is named here before any of these cells has been run.

The other two risk levels are **not** being collected today. The resulting claim
is therefore a matched group-size comparison **at risk 0.6**, and it must be
written that way. It does not become a claim about the risk grid by being
extended later; if the remaining eight cells are collected, that is a separate,
larger claim and this document is amended rather than quietly superseded.

## Cell to identity assignment

Fixed now. Each identity collects one cell, whole, with all ten repetitions.

| Cell | Group size | Risk | Identity | Approximate requests |
|---|---|---|---|---|
| A | 2 | 0.6 | `foundnotkiet` | 372 |
| B | 3 | 0.6 | `kit567` | 558 |
| C | 4 | 0.6 | `hunhtrungkit` | 744 |
| D | 5 | 0.6 | `tnkiet` | 930 |

`daosyduyminh` and `trungkiet` are excluded because both were refused today; they
are not retried in order to avoid selecting on the outcome.

Model route for every cell: `google/gemini-3-flash-preview`, an admitted route
under the endpoint-admission campaign. One route across all four cells, so the
group-size contrast is not confounded with the model.

## Why this is a partition and not identity rotation

The collaborator-identity amendment in
`docs/frontier-evidence-collection-protocol.md` forbids rotating identities until a
run happens to pass, because that selects on the outcome and stitches one sample
out of several billing accounts mid-run. This is a different thing:

- the assignment is declared before the first run, not adjusted as results arrive;
- each cell is collected **whole**, so it supports its own race-clustered
  interval and is never a fragment;
- the game seed is `base_seed + rep` and is independent of the risk treatment and
  of the seat count, so a repetition index pairs across cells no matter which
  identity collected them, which
  `scripts/verify_matched_nplayer_design.py` asserts offline;
- every manifest records its executing identity, and any table that pools cells
  must name the identities.

The identity is a billing boundary, not a model boundary: the route, prompt,
parser, protocol, temperature request and seed structure are identical across
cells. That it is nevertheless a difference between runs is disclosed rather than
argued away.

## Stopping rule

If a cell is refused on quota, that cell is **not** reattempted on a different
identity today. It is recorded as a failure record and the sweep stays
incomplete. `scripts/analyze_nplayer_matched.py` refuses an incomplete grid, so
an unfinished collection cannot be reported as a result.

If fewer than four cells complete, there is no matched group-size comparison and
the manuscript keeps its existing limitation unchanged.
