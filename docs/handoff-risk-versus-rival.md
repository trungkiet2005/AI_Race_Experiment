# The rival against the danger: suggested wording

Handoff only. Nothing here has been written into `paper/main.tex` or
`paper/supplementary.tex`. Every number below is produced by
`scripts/analyze_scripted_opponent.py --all-routes` and pinned by
`scripts/verify_manuscript_claims.py`.

## What changed in the analysis

The two halves of the new headline were not being produced the same way. The
rival contrast was a difference taken inside a repetition block, resampled over
repetitions, carrying a percentile interval and a re-derived check that the two
arms really shared a horizon draw. The risk contrast was a subtraction of two
cell rates, with no interval and no pairing. Putting those two in one sentence
invites the objection that the sentence exists to answer.

The risk contrast is now computed exactly as the rival contrast is. The game
seed is `base + repetition` and names neither the rival nor the risk, so one
repetition index reuses a single horizon stopping draw in every cell of every
endpoint. That is what lets the two risk levels be differenced inside a
repetition, and the analyser re-derives the pairing from the recorded seed
rather than assuming it. All twelve risk contrasts are paired over ten blocks
with the pairing verified.

Each risk contrast is emitted per route and per rival strategy, so the effect
can be read at a fixed rival instead of pooled across rivals. Pooling would
have hidden the one cell a reader needs to see.

Scope. The analysis covers the three endpoints whose grids are complete at
twelve of twelve cells: Gemini 3 Flash, Claude Sonnet 5 and GPT-5.4. A fourth
endpoint has one collected cell and is named in the scope block of the artifact
and kept out of every range, every span and every comparison in it. The code
was exercised against a fourth and a fifth partially collected endpoint and
reports each on its own without entering any range, leaving the three complete
grids bit for bit unchanged.

## The table

Stated risk 0.1 minus stated risk 0.9, at a fixed rival, differenced inside a
repetition, 95 per cent percentile interval over ten repetitions.

| route | vs Always Safe | vs Always Unsafe | vs Conditional Safe | vs Conditional Unsafe |
|---|---|---|---|---|
| Gemini 3 Flash | +10.4 [+5.4, +15.7] | +12.2 [+8.2, +16.1] | +18.0 [+12.4, +23.1] | +31.5 [+23.1, +38.1] |
| Claude Sonnet 5 | +12.7 [+8.5, +16.6] | **+31.5 [+27.2, +35.3]** | +19.4 [+14.3, +23.5] | +41.0 [+37.7, +43.8] |
| GPT-5.4 | +12.3 [+6.2, +18.6] | +10.0 [+5.9, +14.2] | +21.3 [+14.0, +28.3] | +24.4 [+21.3, +28.5] |

Always Unsafe minus Always Safe, at a fixed stated risk, same construction,
runs from +45.0 to +73.5 points across the nine cells.

One note for anyone reconciling these against the rates table. A paired
contrast averages the ten repetition rates, while the rates table pools the 93
decisions, and the two differ by a point or so when repetitions run to
different lengths. The heights quoted below are the paired ones, so that a
contrast and the heights it was taken between are read off the same
construction.

## The exception, and what it is

Claude Sonnet 5 against the always-unsafe rival gives up 31.5 points. That is
2.49 times the next largest cell on those two arms and 2.73 times the average
of the other five. It is two and a half times, not three times, and the draft
should say two and a half.

The question is whether that cell is a property of the route or of the height
its rates start from. It is a real question, because an arm already playing
Unsafe in every round of every repetition has nowhere left to record a
response.

It is a property of the route. Claude Sonnet 5 and GPT-5.4 begin that arm at
the same height: 88.7 per cent against 87.8 per cent at risk 0.1, a paired gap
of +0.9 points with an interval of [+0.0, +2.7], and neither arm is saturated,
with one repetition block at the ceiling for Claude Sonnet 5 and none for
GPT-5.4. From that level start their risk effects still differ by 21.5 points
with an interval of [+13.3, +29.0]. Two arms at the same height and the same
headroom cannot be separated by their height.

The ceiling does bite, but somewhere else, and in the opposite direction to the
one a reader would guess. Gemini 3 Flash plays Unsafe in 100 per cent of
decisions in all ten repetition blocks of that arm at risk 0.1, so its +12.2 is
a lower bound on a response the rate scale cannot record. On the log odds
scale, which is not compressed at the boundary, that same arm moves furthest of
the three, 3.26 against 1.91 for Claude Sonnet 5 and 0.67 for GPT-5.4. Its
small difference of rates is therefore not evidence of a small response, and
the paper must not read it as one. The Gemini against GPT-5.4 comparison, which
puts a saturated arm beside an unsaturated one, comes back at +2.2 points with
an interval of [-4.2, +8.0] and is reported as inconclusive rather than as a
null.

## The sentence

> Holding the stated probability of catastrophe fixed, replacing an always-safe
> rival with an always-unsafe one moves a route by 45 to 74 points, whereas
> holding the rival fixed and raising that probability from 0.1 to 0.9 moves it
> by 10 to 13 on the same two arms, with a single exception: Claude Sonnet 5
> gives up 31.5 points [27.2, 35.3] against the always-unsafe rival, two and a
> half times the next largest cell, and a property of that route rather than of
> where its rates sit, because it and GPT-5.4 begin that arm level and still
> differ by 21.5 points [13.3, 29.0].

## The companion sentence, which is not optional

The sentence above is scoped to the always-safe and always-unsafe arms, because
those are the two arms the rival contrast is taken between. On those arms the
two sets of intervals never meet: the largest risk upper bound is 35.3 and the
smallest rival lower bound is 35.8. Widen the risk side to the conditional
rivals and they do meet, because the risk effect there reaches 41.0 points with
an upper bound of 43.8. So the claim that the rival matters several times more
than the danger is true of the arms it names and not of the grid as a whole,
and the manuscript needs a sentence saying so:

> Against a rival that copies the route's own previous move the stated risk
> matters more than this, up to 41.0 points [37.7, 43.8], so the comparison
> above is a statement about the two unconditional rivals rather than about the
> whole design.

## What the draft should not say

Three things in the table contradict shorter versions of the claim, and each
has a guard in `scripts/verify_manuscript_claims.py` that fails if a later
edit reintroduces it.

- Not "three times every other cell". The ratio is 2.49.
- Not "the rival matters more than the danger" without naming the arms. Claude
  Sonnet 5's largest risk contrast is +41.0 against the conditional-unsafe
  rival, above the +31.5 the sentence calls exceptional, and it overlaps the
  smallest rival contrast.
- Not "Gemini 3 Flash responds least to risk against an unsafe rival". Its arm
  is at the ceiling in all ten blocks and its log odds shift is the largest of
  the three.

## Where the numbers live

- `results/derived/scripted_opponent_campaign/risk_versus_rival.json`, the two
  contrasts side by side with the scope block and the exception test.
- `results/derived/scripted_opponent_campaign/**/scripted_opponent_rates.json`,
  the per-route grids, now carrying `paired_risk_contrasts` with a ceiling
  diagnostic on each entry, and `grid_complete`.
- `scripts/verify_manuscript_claims.py`, which pins the point estimate and both
  interval endpoints for each route on each of the two arms, the exception
  cell, the level-start test, the saturation finding, and the three guards
  above.
