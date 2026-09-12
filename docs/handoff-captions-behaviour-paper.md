# Figure handoff for the behaviour paper

Handoff only. Nothing here has been written into `paper/main.tex` or
`paper/supplementary.tex`; the figure lane does not own those files. Every
number below was recomputed in this session by the generator that draws it, and
each generator prints the value it drew and refuses to draw when the value it
finds no longer supports the sentence on the panel.

* * *

## 1. Two placement changes the paper will not build correctly without

**a. The opening figure uses the protected author artwork.** Figure 1 is
`AIRaceOverview.pdf`, the hand-drawn mechanism overview. It keeps the repeated
race, the two-player payoff matrix and the $N$-player payoff in one visual
language. Same placement, full width:

    \includegraphics[width=\textwidth]{../figures/paper/AIRaceOverview.pdf}

**b. The scripted-rival figure uses the taller full-width composition.** It
contains three risk facets above the rival-stance contrast and the self-play
comparison, so the labels remain readable at publication size. The canonical
file is `scripted_opponent.pdf`:

    \includegraphics[width=\textwidth]{../figures/paper/scripted_opponent.pdf}

A full-width float keeps the five panels legible and is inside the page budget.

**c. Two supplement variants exist and nothing includes them yet.**
`human_versus_model_all_routes.pdf` draws all nine routes and
`theory_versus_behaviour_all_routes.pdf` draws the nine-route roster against the
model. The main paper's versions lead with the five screened routes and humans,
which is what the brief asks for, so the nine-route versions belong in the
supplement. Both are full width.

* * *

## 2. Captions

House rules applied: the caption carries what the panel cannot, every term used
on a panel is defined once here, and no value is repeated from a table. Where a
panel deliberately underclaims, the caption carries the count.

### Figure 1, opening, full width. `AIRaceOverview.pdf`

> **The race and its two payoff views.** (a) A repeated development race in
> which each round advances progress by the action-dependent amount and the
> winning company receives the prize after the race ends. (b) The two-player
> stage-payoff matrix, where Unsafe is attractive within a round while private
> setback risk supplies the safety cost. (c) The $N$-player extension, where the
> payoff is divided across the leading companies and group size changes the
> payoff structure. Nothing in this figure is a result; the race is an idealised
> game with two actions, a fixed prize and a stated risk.

Term defined here and used by every later figure: *route*, one commercial model
endpoint at one fixed setting.

### Figure 2, risk, one column. `risk_response.pdf`

> **Every admitted route plays Unsafe less as the stated danger rises, and four
> of the five do it by nearly the same amount.** Pooled Unsafe play for the five
> screened routes at each stated maximum private risk, ten races and 186
> decisions per cell, no parse failures. The right-hand column is the drop from
> the lowest to the highest risk in percentage points; it is a difference of two
> published cell rates and carries no interval, so it is a central value rather
> than an estimate with uncertainty attached. Four routes drop within a band
> nineteen points wide. Claude Opus 5 does not trade off at all: it plays Unsafe
> on every decision at the lowest risk and on none above it, and is reported on
> its own rather than inside an average, because the five-route mean is one
> route's scale. An outlined cell sits at nought or at one hundred and therefore
> has no room beside it, so any contrast it enters is truncated. The bottom row
> is not a sixth route: it is one of the five administered the same frozen
> protocol a second time, and the two administrations are at most 1.1 points
> apart at any risk level. Only three risk levels were run, so behaviour between
> them is unobserved, which matters most for the route whose entire change
> happens in the first gap. Every rate here is self-play, which Figure 3 shows
> is not a measurement of how a route treats risk. The four routes the screen
> refused are drawn beside these five in the supplementary material.

### Figure 3, rivalry, full width. `scripted_opponent.pdf`

> **What the rival is doing moves a route several times further than how
> dangerous the race is.** Three admitted routes play the same race against four
> rivals that are code rather than models, executed by the task file: Always
> Safe, Conditional Safe, Conditional Unsafe and Always Unsafe. The two
> unconditional rivals ignore the route entirely, the two conditional ones
> answer only the route's own last move, and no route is told its rival is
> scripted, so a contrast across rivals is causal by construction. Thirty-six
> cells, 360 races, 3,348 route decisions, no parse failures, and no rival
> deviation in a replay of every recorded turn; seats are counterbalanced five
> and five in every cell. (a) Unsafe play rises from the always-safe to the
> always-unsafe rival in all nine route-by-risk cells, strictly in eight: at the
> lowest risk one route is at one hundred per cent against both unsafe rivals,
> and two cells with no room above them cannot be put in an order. (b) Both
> families of contrast are differenced inside a repetition, so the hidden
> stopping time is removed from each. Changing the rival is worth three and a
> half to seven times what changing the stated danger is worth, and no interval
> of either kind falls in the shaded corridor between them. That comparison
> belongs to the two unconditional rivals: against a rival that copies the
> route's own last move the danger is worth up to 41 points, which overlaps the
> smallest rival contrast, and the panel says so rather than leaving it to the
> supplement. (c) Against a rival that always plays Safe, every route plays
> Unsafe less than it does against a second copy of itself, in all nine cells.
> A self-play rate is therefore a property of a policy meeting itself, not a
> measurement of how a route treats risk. Three routes, one game, one prompt
> version and ten races per cell; the four rivals are reduced strategies rather
> than a sample of opponents.

### Figure 4, theory, full width. `theory_versus_behaviour.pdf`

> **The evolutionary benchmark and the route profiles share a risk coordinate.**
> Panel (a) shows the reduced model solved on 201 risk levels at both declared
> selection strengths, beside the five admitted routes at the three tested
> levels. The strong-selection model is step-like, the weak-selection model is
> graded, four routes move gradually, and Claude Opus 5 changes sharply between
> the two lowest tested levels. Panel (b) shows the corresponding finite-
> mutation strategy composition at the reference and reported best-fit
> settings. The dominant strategy shifts from Always Unsafe to the conditional
> strategy that starts Unsafe and then to the conditional strategy that starts
> Safe as risk rises. Nothing here is fitted to route behaviour.

The curves in panel (a) are the small-mutation limit. The strategy shares in
panel (b) come from the archived finite-mutation chains, so the figure keeps the
two evolutionary layers explicit rather than treating them as one object. The
reported best-fit regime is the one used in the paper's finite-mutation sweep.

### Figure 5, humans, full width. `human_versus_model.pdf`

> **Human participants use nearly the whole policy space and every screened
> route uses a sliver of it.** The measure is the number of distinct five-round
> action sequences among twenty paired trajectories, so one means all twenty are
> identical and twenty means all twenty differ. (a) The human reference is
> 20,000 draws of twenty participants from the source study's 340 complete
> trajectories, redrawn at each risk level; the dashed rule is the smallest
> count any draw produced. All fifteen screened route-by-risk cells fall below
> every human draw at every risk level. Against a stricter reference matched on
> independence as well as on size, built from ten complete human pairs, thirteen
> of the fifteen still do. (b) Pooled over the three risk levels, the human
> median uses 57 of a possible 60 sequences and the narrowest route uses two,
> the same sequence repeated by all twenty trajectories in every risk cell. The
> figure discloses what the screen costs: the only cells in this study that
> reach the human range belong to two routes the screen refused, and they are
> reported here rather than omitted. Human and model trajectories are matched on
> count and on round structure but not on incentive, and twenty is a small
> sample for a measure whose ceiling is twenty.

* * *

## 3. Things the writers should know before quoting a number

**The theory comparison now uses the canonical baseline values.** An earlier
claim-map entry copied numbers from an independent repeat and an obsolete
comparison. Recomputed from
`results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.csv` and
from the decision records independently, the screened-route values are Gemini
3 Flash at 98.9, 73.1, 60.2 and Claude Sonnet 5 at 89.2, 48.4, 32.3. The
claim map and both manuscript sources now use these values; the independent
repeat remains 100.0, 74.2, 59.7 in its own subsection.

**"Four to seven times" is half a point generous.** The recomputed ratio of the
rival contrast to the risk contrast across the nine cells runs from 3.6 to 7.0.
Write three and a half to seven, which is what the figure draws.

**The risk half of the headline is now the paired contrast, not the difference
of two cell rates.** Against a rival held at Always Safe the values are 10.4,
12.3 and 12.7 points with intervals, not the 10.75, 10.75 and 11.83 the claim
map carries, which were unpaired differences of published cell rates. The
figure and `results/derived/scripted_opponent_campaign/risk_versus_rival.json`
agree to machine precision, and the generator fails if they ever stop agreeing.

**Do not write that the rival outweighs the danger without naming the arm.** On
the conditional-unsafe arm the danger is worth up to 41.0 points with an
interval reaching 43.8, and the smallest rival contrast has a lower bound of
35.8, so the two families meet there. The panel title and the note under it both
name the arm; a sentence in the body that drops the qualifier contradicts the
figure standing beside it.

**Unit.** Both figures that report a change now write it as points. Do not
reintroduce "pp" in a caption or an axis: one unit under two names costs the
reader attention that belongs on the number.

* * *

## 4. What changed in the figure code this session

- `scripts/figstyle.py`: the numeral on a heat tile now takes its ink from the
  tile's own luminance instead of from a fixed cut in the value. The fixed cut
  assumed a colormap that darkens as the value rises; viridis brightens, so it
  was printing white type on the yellow end and dark type on the purple end,
  which is the weakest type available at both ends. The defect was visible in
  the shipped risk-response and audit figures, and two other scripts had already
  patched it locally after the fact. `textcolor_flip` still overrides, for the
  one caller that measured its own crossing point on a different colormap.
- `scripts/figures/fig_risk_response.py`: the band width in the title is read
  off the drops instead of typed; the repeat note now reads "at most 1.1 points
  apart" rather than "never 1.1 points off", which asserted the opposite of the
  finding; the outline channel has a key; the column header says points.
- `scripts/figures/fig_scripted_opponent.py`: the rival-versus-danger panel
  names the arm its claim holds on and carries the widest risk contrast in the
  grid, which refuses the unqualified reading; the recomputed risk contrast is
  now cross-checked against the analyser's published table and the figure
  refuses to draw if they disagree; panel titles say what the nine are.
