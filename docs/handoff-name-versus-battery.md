# Does the battery measure anything a regex on the model name does not?

Written 2026-09-12 for the ACL audit paper. This is the paper's existential
question, so the answer belongs early in its results rather than in a limitation.

Recomputed from `results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv`.
Size token matched as a hyphen or dot delimited segment (`mini`, `nano`, `lite`),
which matters: the naive substring test matches `mini` inside `gemini` and
reports 8 of 9 instead of 9 of 9.

## The problem, stated at full strength

The binary verdict is reproduced perfectly by the string match. Nine of nine,
exact one-sided permutation p = 1/C(9,4) = 1/126 = 0.0079. A sixty-call battery
whose verdict a regex reproduces has not been shown to measure anything.

## What the continuous scores add

A threshold discards information, so ask the same question of the per-domain
scores rather than the verdict.

| domain | no size token | size token | name separates? |
|---|---|---|---|
| rule recall | 1.000 | 0.938 | overlaps |
| stage payoff | 1.000 | 1.000 | overlaps |
| **state reconstruction** | **0.933** | **0.550** | **separates completely** |
| state transition | 1.000 | 0.667 | overlaps |
| terminal scoring | 0.960 | 0.733 | overlaps |
| expected payoff | 0.267 | 0.042 | overlaps |

And within the five routes that carry **no** size token, the battery still
resolves differences: state reconstruction spans 0.800 to 1.000, terminal
scoring 0.800 to 1.000, expected payoff 0.000 to 0.500.

## The honest reading, which is not a rescue

Two things are true at once and the paper must say both.

The battery resolves structure the name does not: five of six domains overlap
across the name split, and there is real spread inside the name-identical group.

But the one domain the name separates completely is **the domain that gates**.
So the decision still rests exactly where a regex can reach, and that is the
claim a reviewer will press.

Do not write "five of six domains show the battery measures something else"
without the next sentence. Three of those five overlap largely because they sit
at the ceiling (rule recall 1.000 vs 0.938, stage payoff 1.000 vs 1.000, state
transition 1.000 vs 0.667). Overlap by saturation is not discriminating power,
and a draft that counts it as such is making exactly the clean generalisation
its own table contradicts.

## What this tells the new collection to do

The discriminant-validity problem is **concentrated in the gating domain**, not
spread across the battery. So the roster to collect is not "more models"; it is
models where **state reconstruction and the size token come apart** — a
large-named route that reconstructs state poorly, or a small-named route that
reconstructs it well.

This is also the argument for the further task families. A game whose state to
track is the opponent's history, rather than an accumulating private quantity,
creates a new axis of variation in precisely the domain that currently has none.

## Reproduce

`scripts/` has no module for this yet. The calculation is a group-by over the
six `accuracy_*` columns of the admission CSV plus a token-boundary regex; it
should be written into the ACL paper's analysis code rather than left in a
scratch file.
