# Scripted-opponent campaign: declared collection plan, 2026-09-10

Written and committed **before** any cell below is started, for the same reason
every other plan in this study was: the assignment must not be adjustable once
results begin to arrive.

## Why this experiment exists

Every gameplay result in this paper is self-play. Both companies in a race are
the same endpoint, so a route that plays Unsafe after its rival did cannot be
told apart from a route that is simply in an unsafe phase, because the rival is
the route. The Mantel-Haenszel reciprocity estimates already in hand condition on
the route's own previous move, which removes the arithmetic part of that
confound, but it cannot remove the design part: there is still only one policy in
the room.

The paper also has two lanes that never meet. The evolutionary lane is built on
four reduced strategies from the source paper, Always Safe, Always Unsafe,
Conditional Safe and Conditional Unsafe, and the empirical lane has never once
played against them. This campaign plays the audited route against exactly those
four.

Against a fixed and known rival, two quantities self-play confounds come apart:

- **exploitation**, playing Unsafe against a rival that always plays Safe;
- **retaliation**, playing Unsafe against a rival that always plays Unsafe.

The two conditional strategies then ask whether the route conditions on history
at all, and because they differ only in their opening move, the pair separates
answering the rival's first move from answering the rival's pattern.

## The design, frozen

- Route: `google/gemini-3-flash-preview`, the highest-scoring admitted endpoint
  and the one the rest of this study reports most often.
- Rivals: `AS`, `AU`, `CS`, `CAS`, executed by the task file rather than called,
  so the rival costs no requests.
- Risk: `0.1`, `0.6`, `0.9`, the same three levels as every other campaign.
- Ten repetitions per cell, one horizon stream shared with the neutral baseline,
  so a repetition here can be differenced against the same repetition there.
- Protocol `ai-race-scripted-opponent-v1`.
- The prompt is byte-identical to the neutral baseline's and **never says the
  rival is scripted**. The rival is a strategy, not a disclosure.
- The route's seat is counterbalanced: seat one on even repetitions, seat two on
  odd. This study has measured a seat effect in the neutral baseline, on prompts
  differing only in which of two names is whose, so a fixed seat would fold that
  effect into every number here.
- Checked offline before any push by `scripts/verify_scripted_opponent_design.py`,
  which asserts that each strategy behaves as its name says, that a conditional
  strategy answers the route and not itself, that every mechanism field and the
  prompt hash match the neutral baseline, that the seat balance is 5 and 5, and
  that the horizons are the baseline's own. All fifteen checks pass.

## Cost, and why it is partitioned

One cell is ten races and 93 route decisions. The full grid is twelve cells and
1,116 route decisions. A Model Proxy identity carries roughly 800 to 1,400
requests, so the grid does not comfortably fit on one identity, and this study
does not stitch a single sample out of several billing accounts mid-run.

So the workload is partitioned into the analysis unit itself, one (strategy,
risk) cell, and each cell is collected whole on one identity. Each cell then
carries its own race-clustered interval, and because the game seed is
independent of the treatment, repetitions still pair across cells collected on
different accounts.

## Assignment, and why it rotates

| Cell | risk 0.1 | risk 0.6 | risk 0.9 |
|---|---|---|---|
| Always Safe (`AS`) | `kit567` | `hunhtrungkit` | `tnkiet` |
| Always Unsafe (`AU`) | `hunhtrungkit` | `tnkiet` | `foundnotkiet` |
| Conditional Safe (`CS`) | `tnkiet` | `foundnotkiet` | `kit567` |
| Conditional Unsafe (`CAS`) | `foundnotkiet` | `kit567` | `hunhtrungkit` |

Three cells per identity, 279 route decisions each, comfortably inside a single
identity's quota.

The rotation is the point. If one identity always collected the same rival, the
identity and the treatment would be perfectly confounded and a reader could not
tell a strategy effect from an account effect even in principle. Under this
assignment every strategy is collected on three different identities and every
identity sees three different strategies at three different risk levels, so an
account effect would appear as an inconsistency across cells rather than hide
inside the contrast.

The identity remains a billing boundary and not a model boundary: route, prompt,
parser, protocol, temperature request and seed structure are identical in every
cell. The rotation makes that assumption checkable instead of merely asserted.

Every cell records its collecting identity in a `collection_receipt.json`,
because the benchmark server exposes neither `KAGGLE_USERNAME` nor
`KAGGLE_KERNEL_RUN_OWNER` to task code and the run manifest's own field therefore
reads `unrecorded`.

## Stopping rule

Unchanged from the rest of this study. A cell refused on quota is recorded as a
failure record and is **not** reattempted on another identity; a cell that fails
on transport congestion may be retried on **its own declared identity**, because
that is a retry and not a rotation. A strategy missing any risk level, or a risk
level missing any strategy, is not reported as a grid. If only part survives, the
supplement reports the cells that exist and says which do not.

Two results are worth stating in advance so that neither can be presented as a
surprise afterwards. If the route plays Unsafe at similar rates against Always
Safe and Always Unsafe, its earlier apparent reciprocity was a property of
self-play rather than of the route. If it plays Unsafe far more against Always
Unsafe, the reciprocity is real and this campaign is what establishes it. Both
outcomes are reportable and the manuscript will report whichever occurs.
