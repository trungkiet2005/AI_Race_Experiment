# Figure handoff for the behaviour paper

Handoff only. Nothing here has been written into `paper/main.tex` or
`paper/supplementary.tex`; the figure lane does not own those files. Every
number below was recomputed in this session by the generator that draws it, and
each generator prints the value it drew and refuses to draw when the value it
finds no longer supports the sentence on the panel.

---

## 1. Two placement changes the paper will not build correctly without

**a. The opening figure is a different file now.** The mechanism figure
`AIRaceOverview.pdf` is no longer part of the set. It drew the mechanism three
times, spent a third of its width on the group-size rule that this paper now
reports in the supplementary material, and was drawn in a second visual
language at 19 inches wide, so on the page its lettering printed at nearly ten
points beside the six-point lettering of every other figure. Its replacement is
`delegation_overview.pdf`, which encodes the delegation scenario, the two
actions, the stage payoffs and the three lenses. Same placement, full width:

    \includegraphics[width=\textwidth]{../figures/paper/delegation_overview.pdf}

**b. The scripted-rival figure is now a full-width figure and must be placed as
one.** It currently sits in a `figure` at `\columnwidth`, and the geometry gate
fails on it: the file is 6.99 in wide, the column is 3.34 in, so the page
rescales it to 48 per cent and its 7 pt labels print at 3.3 pt. It grew a third
panel because that panel carries the paper's headline, the rival against the
danger on one axis, which did not previously exist as a picture. The fix is one
environment change:

    \begin{figure*}[t]
      \includegraphics[width=\textwidth]{../figures/paper/scripted_opponent_main.pdf}

A full-width float is charged twice its height and nothing for its width, so
this costs about two and a half column inches over the column version, which is
inside the budget the claim map gives this subsection. Both gate scripts pass
once the environment changes; until it changes, both fail on this one file.

**c. Two supplement variants exist and nothing includes them yet.**
`human_versus_model_all_routes.pdf` draws all nine routes and
`theory_versus_behaviour_all_routes.pdf` draws the nine-route roster against the
model. The main paper's versions lead with the five screened routes and humans,
which is what the brief asks for, so the nine-route versions belong in the
supplement. Both are full width.

---

## 2. Captions

House rules applied: the caption carries what the panel cannot, every term used
on a panel is defined once here, and no value is repeated from a table. Where a
panel deliberately underclaims, the caption carries the count.

### Figure 1, opening, full width. `delegation_overview.pdf`

> **The race, and the three lenses this paper applies to it.** (a) A decision in
> a race between two developers is handed to a language model, which is given
> the rules, the stated danger, the rival's last move and the score, and returns
> one of two actions. The paper is prospective throughout: it asks what a
> delegate would choose, never who delegates today. (b) Safe advances one race
> step and adds no risk; Unsafe advances one and a half and raises the chance of
> a setback, which is the stated maximum multiplied by the share of the player's
> own moves that were Unsafe, and which wipes out the whole payoff. A setback is
> applied only to a player who wins or ties for first. (c) The stage payoffs,
> parsed from the prompt every player was sent. Unsafe earns more than Safe
> against either rival move and mutual Unsafe earns more than mutual Safe, so
> the round in isolation carries no tension; the entire cost of racing is the
> setback in (b). (d) The three lenses, in the order the results take them. Nine
> routes played the risk grid and five entered the behavioural panel under the
> comprehension screen described in the methods. Nothing in this figure is a
> result. The race is an idealised game with two actions, a fixed prize and a
> stated risk, and models no real development programme.

Term defined here and used by every later figure: *route*, one commercial model
endpoint at one fixed setting.

### Figure 2, risk, one column. `risk_response.pdf`

> **Every screened route plays Unsafe less as the stated danger rises, and four
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

### Figure 3, rivalry, full width. `scripted_opponent_main.pdf`

> **What the rival is doing moves a route several times further than how
> dangerous the race is.** Three screened routes play the same race against four
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

### Figure 4, theory, one column. `theory_versus_behaviour.pdf`

> **The evolutionary benchmark is a step and the screened routes are slopes.**
> The reduced evolutionary model solved on 201 risk levels at both declared
> selection strengths, against the five screened routes at the three levels the
> protocol ran. Selection strength, written as beta, is how sharply an imitating
> player prefers the better-performing strategy; both curves are the
> vanishing-mutation limit in a population of one hundred, which is the limit
> the model is usually reported in and not a fit to any behaviour here. At the
> reference strength the model stays at the ceiling, falls almost its whole
> height inside a single risk step of 0.005, and stays on the floor; a step
> function has no gradient, so there is nothing for a route to be calibrated
> against. Weakening selection gives the model a slope of its own, falling at
> most five points per risk step. Four of the five routes walk a slope; Claude
> Opus 5 has a cliff of its own, and it is in the wrong place, somewhere inside
> the gap between the two lowest levels where the protocol ran nothing. Its two
> saturated cells are 186 decisions on the ceiling and 372 on the floor.
> Nothing here is fitted and no parameter is estimated from behaviour, so this
> is not a test of the model.

**Read this before writing the subsection.** The two curves drawn here are the
vanishing-mutation limit. The claim that a weak selection strength brings the
model close to the routes comes from a different object: the finite-mutation
stationary distribution at a mutation rate of 0.05, which predicts 87.3, 63.9
and 38.0 per cent at the three levels and does land inside the band of routes.
The drawn curve at the same strength passes near 100, 97 and 5. If the body
quotes the fit while the page shows the curve, a reader comparing the two finds
a contradiction. Either write the fit as belonging to the finite-mutation
sweep, in words that say so, or leave it in the supplement. The figure lane
considered adding the three finite-mutation points to this panel and did not:
at column width the panel already carries two model curves, a five-route band,
the exception route, a cliff leader and two boundary rules, and a third model
object labelled with the same strength as one of the curves would be read as a
contradiction rather than as a second regime. If the body needs the fit on the
page, say so and it can have a second column-width panel.

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

---

## 3. Things the writers should know before quoting a number

**The claim map's Figure 4 numbers are wrong and the figures are right.** Item
B13 of `docs/behaviour-paper-claim-map-2026-09-12.md` gives Gemini 3 Flash as
98.9, 74.2, 59.1 and Claude Sonnet 5 as 89.2, 46.2, 37.6. Recomputed from
`results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.csv` and
from the decision records independently, the values are 98.9, 73.1, 60.2 and
89.2, 48.4, 32.3. The 74.2 in that line is the independent repeat's value, not
the route's, and 59.1, 46.2 and 37.6 appear in no artifact. Four of six numbers
in that line are wrong. Do not copy it.

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

---

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
