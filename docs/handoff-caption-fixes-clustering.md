# Caption handoff: the LLM/human clustering supplement figures

Four figures under `figures/paper/llm_human_clustering/` were redrawn on
2026-09-11 by `scripts/build_supplementary_figures.py`. Their captions in
`paper/supplementary.tex` now describe figures that no longer exist, and three
of them describe an encoding that was never safe to describe that way. The
caption file is owned by another agent, so the replacements are written out
here.

Every correction below is settled by a named artifact or by the redrawn figure
itself. Where a caption and an artifact disagree, the artifact wins.

Figures covered:

| figure | label | section |
|---|---|---|
| `06_tsne_safe_unsafe` | `fig:human-tsne-outcome` | 1 |
| `03_radar_small_multiples_by_group_no_gap_r1` | `fig:human-radar` | 2 |
| `09_human_vs_llm_own_risk_dependence` | `fig:own-risk-dependence` | 3 |
| `feature_importance_shap_heatmap` | `fig:feature-importance` | 4 |

Section 5 is a defect in a figure this work did not own and could not fix.

---

## 1. `fig:human-tsne-outcome` (`06_tsne_safe_unsafe`, figure included at line 1911)

### 1a. The colour scale is no longer green-to-red

**Current caption text**

> recoloured by each player's own mean \Unsafe{} rate over rounds 1--5
> (green = \Safe{}, red = \Unsafe{}) instead of population identity.

**Current Description text**

> on a green-to-red scale where green is Safe and red is Unsafe. ... The
> embedding separates almost entirely by Unsafe rate: a solid green region, a
> solid red region, and a mixed band between them, with very few red points
> inside the green region or the reverse.

**Why it has to change.** The figure used `RdYlGn_r`. On that map hue is the
only channel that carries the encoding, because position is the t-SNE
coordinate and shape is the route. `RdYlGn` is not monotonic in luminance: its
*lightest* colour sits at the midpoint, so a player at 0.5 was the most visually
salient point on the page, and under deuteranopia the two ends collapse to dark
olive and rust while the middle stays pale. The figure now uses `viridis`, which
five other figures in this manuscript already use and which rises monotonically
in luminance, and marker area now rises with the rate as an explicit second
channel. Low Unsafe is dark purple and small; high Unsafe is yellow and large.

**Replacement caption**

```latex
    \caption{The same t-SNE coordinates as the population-identity embedding in the main paper,
    recoloured by each player's own mean \Unsafe{} rate over rounds 1--5
    instead of by population identity. Colour runs along viridis from dark
    purple at all-\Safe{} to yellow at all-\Unsafe{}, and marker area rises with
    the same rate, so the encoding survives greyscale and colour-vision
    deficiency; marker shape is the route, and it is the shape each route
    carries everywhere else in the paper. Left: all 760 trajectories pooled.
    Right: the same colouring broken out per population.}
```

**Replacement Description**

```latex
    \Description{The same t-SNE coordinates as the previous figure,
    recoloured by each player's own mean Unsafe rate over rounds 1 to 5
    rather than by which population produced it. Colour and marker size both
    increase with the Unsafe rate, from small dark purple points at all-Safe to
    large yellow points at all-Unsafe. The left panel pools all 760
    trajectories and the right panel breaks the same colouring out per
    population. The embedding separates almost entirely by Unsafe rate: a solid
    dark region, a solid yellow region, and a mixed band between them, with very
    few high-rate points inside the low-rate region or the reverse.}
```

### 1b. Nothing else in this caption changes

The claim about separation by Unsafe rate is unaffected; only the words naming
the colours are.

---

## 2. `fig:human-radar` (`03_radar_small_multiples_by_group_no_gap_r1`, figure included at line 1941)

### 2a. Panel A no longer has Own and Opp columns, because they were the same
numbers drawn twice

**Current caption text**

> Panel A reports the probability of an \Unsafe{} action for each player's own
> and opponent action at rounds 1--5. Panel B reports the mean absolute
> progress gap entering rounds 2--5, so it measures how far apart the two
> players are after pooling focal-player roles. The eight rows are the human
> reference and the seven baseline model populations.

**Current Description text**

> The first heatmap gives Unsafe-action probabilities for own and
> opponent actions at five rounds. ... Rows represent Human and seven
> model populations; red indicates more Unsafe action in the first panel,
> while darker blue indicates a larger separation in the second.

**Why it is wrong.** Panel A carried ten columns, `Own R1`-`Own R5` beside
`Opp R1`-`Opp R5`. Five of them were duplicates by construction. Every model
population plays itself with counterbalanced seats, so pooling over both seats
makes the opponent marginal the same multiset as the own marginal. Checked
against the trajectory table the canonical loader assembles
(`scripts/build_manuscript_clustering_figures.load_trajectories`, 760
trajectories): for all seven model populations the largest absolute difference
between the own and the opponent marginal is **exactly 0.000000** at every one
of the five rounds. The humans are the single exception, because they played
dyads rather than self-play; their two seats differ by at most **0.29 pp**
(0.002941, one participant in 340), at rounds 1, 3, 4 and 5, and by exactly zero
at round 2.

The ten columns of five numbers left each tile too narrow for its own label,
which is why the three rows that reach 100% printed as `00%` with the leading
digit clipped off at the tile edge. The redrawn panel A has five columns,
`R1`-`R5`, and nine rows: the two human seats shown separately and labelled, and
one row per model carrying the single marginal.

The colour map also changed from `RdYlGn_r` to `viridis`, for the reason given
in section 1a, and panel B's scale now ends at its own data maximum of 0.65
rather than at a hard-coded 1.50 that left the whole panel in the first third of
the ramp.

**Replacement caption**

```latex
    \caption{Round-by-round profile for each population, shown as two
    source-backed heatmaps. Panel A reports the probability of an \Unsafe{}
    action at rounds 1--5. Counterbalanced self-play makes a model's two seat
    marginals the same measurement, and they agree exactly for all seven model
    populations, so each model row carries one marginal rather than the same
    five numbers twice; the humans played dyads, so their two seats are drawn
    separately and differ by at most 0.3 percentage points. Panel B reports the
    mean absolute progress gap entering rounds 2--5, so it measures how far
    apart the two players are after pooling focal-player roles, on a scale that
    ends at its own maximum of 0.65. Row labels carry each route's colour from
    the main paper.}
```

**Replacement Description**

```latex
    \Description{Two heatmaps summarising the shared trajectory features. The
    first gives Unsafe-action probabilities at five rounds, one row for each of
    the two human seats and one row for each of seven model populations. The
    second gives mean absolute progress gaps entering rounds two through five
    for the same rows. In the first panel a yellow tile is near-certain Unsafe
    play and a dark purple tile is near-certain Safe play; in the second, darker
    blue is a larger separation between the two players.}
```

### 2b. The row count in any body text that says "eight"

The figure now has nine rows. Any sentence in the surrounding text that counts
the rows needs the same change.

---

## 3. `fig:own-risk-dependence` (`09_human_vs_llm_own_risk_dependence`, figure included at line 1441)

### 3a. Three routes are outside the roster, not two

**Current caption text**

> GPT-5.6 Luna and Terra (dashed) are the two routes outside the
> main baseline roster.

**Current Description text**

> the two GPT-5.6 lines are dashed because they are outside the main baseline
> roster.

**Why it is wrong.** The generator hard-coded the dashed set as the two literal
names `gpt-5.6-luna` and `gpt-5.6-terra`. Membership of the audited roster is a
property of the route, and the roster is the nine routes listed as `ROUTE_ORDER`
in `scripts/figstyle.py`:

| route | in roster |
|---|---|
| `google/gemini-3-flash-preview` | yes |
| `anthropic/claude-opus-5@default` | yes |
| `anthropic/claude-sonnet-5@default` | yes |
| `openai/gpt-5.4-nano-2026-03-17` | yes |
| `openai/gpt-5-nano` | **no** |
| `openai/gpt-5.6-luna` | **no** |
| `openai/gpt-5.6-terra` | **no** |

GPT-5 nano is a different checkpoint from GPT-5.4 nano and is not one of the
nine audited routes, so like the two GPT-5.6 routes it can never receive an
admission verdict. It was nevertheless drawn solid, which told the reader the
opposite. The rule is now computed from `figstyle.in_roster()` rather than
written out, so it cannot drift again, and three lines are dashed.

### 3b. The two panels no longer sit one index apart

**Current Description text**

> plots human mean Unsafe rate against the elicited Eckel-Grossman gamble
> choice from 0 to 5

**Why it has to change.** The panels share a y-axis and exist to be compared,
but panel A ran 0 to 5 while panel B ran 1 to 6 for the same six ordered levels.
Both now run 1 to 6. The supplement's own body text at line 1376 says the
elicitation is coded 0 = most risk-averse to 5 = least risk-averse, and the
persona sweep runs R1 most risk-averse to R6 most risk-seeking, so the two
axes are now aligned in direction as well as in index. Panel A's tick labels
keep the participants-per-bin counts; panel B's carry the persona name `R1` to
`R6` under each level so the recoding is visible in the figure.

**Note for the body text.** Lines 1376 and 1422 describe the instrument on its
native 0 to 5 coding. Those sentences are about the elicitation and remain
correct; the figure's footer states the re-indexing explicitly, so the caption
does not have to. If the caption owner prefers the body and the figure to agree
literally, the change belongs in the body, not in the figure, because the figure
must align with panel B.

### 3c. Panel B still has no intervals, and that is now stated rather than left
to be noticed

`results/cross_model_pilot_synthesis/data/persona_role_gradient_extended.csv`
carries exactly four columns: `model`, `role`, `mean_unsafe_rate`,
`n_players`. It holds a cell mean and a player count and no player-level rates,
so a participant-bootstrap interval of the kind panel A carries is **not
derivable from it**, and none was invented. The figure now says so in its
footer. The counts are also not uniform: every panel-B point is a mean over 360
players except Gemini 3 Flash, whose six levels rest on 330, 300, 150, 180, 270
and 90 players. That is stated too.

### 3d. Replacement caption and Description

```latex
    \caption{A measured human disposition against an assigned model label.
    Left: mean \Unsafe{} rate by elicited Eckel--Grossman gamble level,
    with 95\% confidence intervals and participants per bin; the dotted
    line is the trend across the six group means. Right: mean \Unsafe{}
    rate by assigned risk-persona level, for every model with a six-level
    sweep; GPT-5 nano, GPT-5.6 Luna and GPT-5.6 Terra (dashed) are the three
    routes outside the nine-route audited roster and carry no admission
    verdict. Both panels index the same six ordered levels from 1, running
    from most to least risk-averse, so the shapes can be read against each
    other; the alignment is visual only, because the left is a disposition
    measured once before play and the right an instruction re-assigned every
    race. The right panel carries no intervals: the archived persona table
    records a cell mean and a player count and not the player-level rates a
    bootstrap needs.}
```

```latex
    \Description{Two panels sharing a vertical Unsafe-rate axis and a
    six-step horizontal axis indexed 1 to 6. The left panel plots human mean
    Unsafe rate against the elicited Eckel-Grossman gamble level, with 95
    percent confidence intervals and the number of participants beneath
    each point; the six points stay between 0.49 and 0.62 and a dotted
    trend line through them is almost flat. The right panel plots mean
    Unsafe rate against the assigned risk-persona level R1 to R6 for seven
    models, each with its own colour and marker shape, and carries no
    intervals. Every model line climbs steeply from left to right, from near
    the floor or lower middle at level 1 to between 0.52 and 1.0 at level 6;
    the GPT-5 nano and the two GPT-5.6 lines are dashed because they are
    outside the audited roster.}
```

---

## 4. `fig:feature-importance` (`feature_importance_shap_heatmap`, figure included at line 1594)

### 4a. The figure now has two panels, and the second is the forests' skill

**Current caption text**

> Population-specific predictive structure for round-$t\geq2$
> \Unsafe{} choices. Each row reports the share of mean absolute SHAP
> value assigned to the same five pre-decision features by a random
> forest fitted separately to that population. SHAP magnitudes describe
> how the fitted classifier uses observed features; they are neither
> causal effects nor evidence of the agents' internal reasoning.

**Why it has to change.** Every row normalises to 100% whether or not the forest
predicts anything, so a row of noise was drawn at the same visual weight as a row
of signal. The supplement's own body text already says this about one row: it
calls GPT-5-nano's out-of-fold AUC of 0.80 "misleading on its own, since
balanced accuracy is exactly 0.50". The figure said nothing. A second panel now
carries out-of-fold ROC AUC with its cross-validated standard deviation and a
chance rule at 0.50, and marks with an open marker and a muted row label any
forest whose balanced accuracy is at chance and whose accuracy does not beat the
majority class.

Read from `results/cross_model_pilot_synthesis/data/feature_importance_results.json`,
two rows are flagged and only two:

| population | ROC AUC | balanced accuracy | accuracy | majority baseline | flagged |
|---|---|---|---|---|---|
| Humans | 0.628 | 0.581 | 0.618 | 0.588 | no |
| GPT-5 nano | 0.795 | 0.500 | 0.923 | 0.923 | **yes** |
| GPT-5.4 nano | 0.541 | 0.517 | 0.532 | 0.544 | **yes** |
| Gemini 3 Flash | 0.918 | 0.773 | 0.878 | 0.747 | no |
| Gemini 3.1 Flash Lite | 0.948 | 0.747 | 0.904 | 0.815 | no |
| Gemini 3.5 Flash Lite | 0.830 | 0.705 | 0.797 | 0.677 | no |
| GPT-5.6 Luna | 0.908 | 0.775 | 0.860 | 0.734 | no |
| GPT-5.6 Terra | 0.925 | 0.853 | 0.854 | 0.581 | no |
| Claude Opus 5 | 1.000 | 0.996 | 0.998 | 0.665 | no |
| Claude Sonnet 5 | 0.974 | 0.932 | 0.933 | 0.480 | no |

These AUC and balanced-accuracy figures are the same numbers already tabulated
in `tab:feature-importance`, so the figure and the table now agree; the table
omits GPT-5.6 Luna and Terra, which the figure carries.

**One correction to an audit note that circulated with this work.** The audit
described Claude Opus 5's row as "a share of a near-zero total" from a
deterministic policy. The artifact does not support that. Claude Opus 5's forest
has ROC AUC 1.000 and balanced accuracy 0.996, the highest in the set, and its
summed mean absolute SHAP is 0.444, the third largest of the ten, behind Claude Sonnet 5 at 0.507 and GPT-5.6 Terra at 0.461. The rows that
are shares of near-noise are GPT-5 nano and GPT-5.4 nano. No caption should
repeat the Opus 5 claim.

### 4b. The colourbar and the tiles now use the same unit

Tiles were labelled in per cent while the colourbar ran 0.0 to 0.5. The
colourbar is now in per cent.

### 4c. Replacement caption and Description

```latex
    \caption{Population-specific predictive structure for round-$t\geq2$
    \Unsafe{} choices. Panel A: each row reports the share of mean absolute
    SHAP value assigned to the same five pre-decision features by a random
    forest fitted separately to that population. A row sums to 100\% whether
    or not the forest predicts anything, so panel B reports the skill each
    row rests on: out-of-fold ROC AUC with its cross-validated standard
    deviation, against a chance rule at 0.50. An open marker and a muted row
    label mark a forest whose balanced accuracy is at chance and whose
    accuracy does not beat the majority class, which is true of GPT-5 nano and
    GPT-5.4 nano and of no other population; their feature shares are shares of
    a predictor that has not learnt the rare \Unsafe{} class and should not be
    read beside the others. SHAP magnitudes describe how the fitted classifier
    uses observed features; they are neither causal effects nor evidence of the
    agents' internal reasoning.}
```

```latex
    \Description{Two panels sharing one row per population. The left panel is a
    heatmap of the share of mean absolute SHAP value that a random forest
    assigns to each of five pre-decision features: own previous action,
    opponent's previous action, progress gap, assigned risk treatment, and
    round number. The human row is dominated by the opponent's previous action
    at 56 percent. GPT-5 nano is dominated instead by progress gap and round
    number at 41 and 38 percent, the two Gemini Flash checkpoints by the risk
    treatment at 33 and 37 percent, and Claude Sonnet 5 by the opponent's
    previous action at 48 percent. The right panel plots each forest's
    out-of-fold ROC AUC with a whisker for its cross-validated standard
    deviation and a dashed chance line at 0.50. Eight forests sit between 0.63
    and 1.00 with filled markers; GPT-5 nano at 0.80 and GPT-5.4 nano at 0.54
    carry open markers and the label "no skill".}
```

---

## 5. A defect this work could not fix: `01_tsne_hero_human_left`

`fig:human-tsne-population` (figure included at line 1880) renders the subplot titles
"Gemini 3.5 Flash Lite" and "Claude Opus 5" touching, so the reader sees
`Gemini 3.5 Flash LiteClaude Opus 5` as one string. It is the same defect that
was in `06_tsne_safe_unsafe`, from the same left-aligned unmeasured title, and
it is still present after a full `scripts/build_publication_figures.py` run.

That figure is drawn by `scripts/build_publication_figures.py`, not by
`scripts/build_supplementary_figures.py`, so it was outside what this work was
allowed to edit. The fix is the same one applied in the supplement builder:
centre each title on its own panel, wrap a label longer than about twelve
characters onto two balanced lines, and then measure the drawn title boxes and
refuse the figure if two of them in the same row come within six pixels. The
helpers `_wrap_title` and `_panel_titles` in
`scripts/build_supplementary_figures.py` are the implementation.

The same figure also still uses a third route palette, `FIGURE_COLORS` and
`FIGURE_MARKERS` in `scripts/build_manuscript_clustering_figures.py`, which
agrees with neither `scripts/figstyle.py` nor the supplement builder. Whoever
owns `build_publication_figures.py` should point it at `figstyle` the way the
supplement builder now does, via `figstyle.route_id`, `ROUTE_C`, `ROUTE_M` and
`ROUTE_LABEL`.

No caption change is proposed for this figure, because the caption is not what
is wrong with it.

---

## 6. Route naming

The same route was being called up to four different things across these
figures: "Gemini 3.5", "Gemini 3.5 Flash Lite", "G3.5 FL" and the raw key
`google/gemini-3.5-flash-lite`. All four supplement figures now take their names
from one table, `ROUTE_LABEL` in `scripts/figstyle.py`, which is the table the
main paper's figures already read. Two consequences for the caption text:

- The human population is labelled **Humans** in these figures, following
  `ROUTE_LABEL["human"]`, not "Human".
- The three Gemini routes are written in full as "Gemini 3 Flash",
  "Gemini 3.1 Flash Lite" and "Gemini 3.5 Flash Lite" everywhere. Any caption or
  body sentence that shortens one of them to "Gemini 3.1" or "Gemini 3.5" now
  disagrees with the figure it describes.
