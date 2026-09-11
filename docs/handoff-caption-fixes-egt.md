# Caption handoff: the two EGT supplement figures

The figures `egt_frontier_insights` and `egt_frontier_invasion` were redrawn on
2026-09-11. Their captions in `paper/supplementary.tex` now describe a figure
that no longer exists, and one of them was already wrong about the paper's own
admission gate before the redraw. The captions live in a file this work was not
allowed to touch, so the replacements are written out here.

Every correction below is settled by a named artifact. Where the caption and the
artifact disagree, the artifact wins.

---

## 1. `fig:egt-frontier-insights` (`paper/supplementary.tex`, caption at the
figure beginning near line 752)

### 1a. "the two admitted frontier routes" is factually wrong

**Current text**

> Boundary analysis for the reconstructed evolutionary model and the two
> admitted frontier routes.

**Why it is wrong.** Five routes are admitted, not two.
`results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv` has
`admitted_for_gameplay = True` on exactly five rows:

| route | overall accuracy | admitted |
|---|---|---|
| `google/gemini-3-flash-preview` | 0.933 | True |
| `anthropic/claude-opus-5@default` | 0.917 | True |
| `openai/gpt-5.4-2026-03-05` | 0.900 | True |
| `openai/gpt-5.5-2026-04-23` | 0.900 | True |
| `anthropic/claude-sonnet-5@default` | 0.850 | True |
| `google/gemini-3.1-flash-lite-preview` | 0.800 | False |
| `openai/gpt-5.4-mini-2026-03-17` | 0.750 | False |
| `google/gemini-3.5-flash-lite` | 0.650 | False |
| `openai/gpt-5.4-nano-2026-03-17` | 0.517 | False |

The same five are listed as `ADMITTED` in `scripts/figstyle.py`, which is the
module the rest of the paper's figures already read.

The figure does not show the admitted routes. It shows the two routes that were
run through the evolutionary comparison protocol, which is a different and
smaller set. `results/frontier/egt_frontier_comparison_v2/reconstruction_manifest.json`,
key `llm_comparison.frontier_primary.model_routes`, names them:
`anthropic-claude-sonnet-5-default` and `google-gemini-3-flash-preview`. The
`context` column of `llm_strategy_summary_primary_t0.csv` abbreviates those to
`claude` and `gemini`, which is where the unidentifiable "Claude" and "Gemini"
series labels came from.

### 1b. The panel letters and the panel names changed

The figure was rebuilt on `scripts/figstyle.py`, so it now uses lowercase panel
letters a-d like the rest of the manuscript, writes the risk cap as 0.1 / 0.6 /
0.9 rather than 10% / 60% / 90%, and says "Unsafe play" rather than "Unsafe
rate". Each panel also carries its claim above the axes, so the caption no
longer has to repeat what each panel is.

### 1c. Uncertainty is now drawn, and its asymmetry has to be declared

The two evolutionary curves carry a shaded band. It is the minimum-to-maximum
range across the four independent seeded chains recorded in
`egt_stationary_summary.csv` (`unsafe_frequency_min`, `unsafe_frequency_max`),
which `reconstruction_manifest.json` calls
`between_independent_chain_range_diagnostic_not_confidence_interval`. It is not
a confidence interval and the caption must not call it one.

The route points carry no band. `llm_strategy_summary_primary_t0.csv` holds one
rate per route and risk and no interval of any kind. A binomial interval over
the 186 decisions is available and would be wrong, because those decisions are
10 races of repeated self-play rather than 186 independent draws. Nothing was
invented to fill the gap.

### 1d. Replacement caption and Description

```latex
    \caption{The reduced evolutionary model against the two routes that were
    run through the comparison protocol,
    \path{anthropic/claude-sonnet-5@default} and
    \path{google/gemini-3-flash-preview}. These are two of the five routes the
    comprehension gate admits, not the admitted set. (a) At its reference
    parameter point the model is a step in risk, holding \Unsafe{} play near
    the ceiling through $p_r^{\max}=0.6$ and dropping it to 1.9\% at 0.9; both
    routes are ramps and both stay far above the model at 0.9. The band on each
    model curve is the range across four independent seeded chains, which is a
    convergence diagnostic and not a confidence interval; the route points carry
    no interval because the archived summary holds none. (b) The same contrast
    signed against the reference, with each point on the risk it was run at.
    (c) Which of the four rules each route's trajectories sit nearest, one stack
    per route. (d) Why (c) is a lens and never a finding: the labels are least
    unique exactly where the distance to the nearest rule is largest.}
    \Description{Four panels comparing a reconstructed evolutionary model with
    self-play by Claude Sonnet 5 and Gemini 3 Flash. The first panel plots
    unsafe play against maximum private risk, showing the model falling from
    99 percent to 2 percent while both routes decline gradually. The second
    plots the signed difference from the model. The third stacks the four
    nearest-rule fractions for each route at each risk. The fourth plots mean
    distance to the nearest rule against the share of trajectories with a
    unique nearest rule.}
```

---

## 2. `fig:egt-frontier-invasion` (`paper/supplementary.tex`, caption at the
figure beginning near line 777)

### 2a. "directed edges" and "arrows show" described something that was not drawn

**Current caption text**

> directed edges show EGTtools \texttt{PairwiseComparison} fixation
> probabilities above neutral drift.

**Current Description text**

> Node areas show finite-mutation stationary shares, and arrows show EGTtools
> fixation probabilities above neutral drift.

**Why it was wrong.** The shipped figure had no arrowheads anywhere. Every edge
was a plain grey line. The generator did request an arrow style, but it drew the
arrow from node centre to node centre underneath discs drawn at a higher
z-order, so every head was buried inside the target node. In an invasion diagram
direction is the entire content, so the caption promised the one thing the
figure did not deliver. The redrawn figure stops each arrow short of the target
disc, by the disc's own radius, and colours it with the invader's colour, so the
direction is now visible and the caption sentence is now true.

### 2b. The direction the old figure would have shown was backwards

This is the more serious finding, and it needs a sentence in the body text as
well as a corrected caption.

EGTtools returns `fixation_probabilities[i, j]` meaning the probability that
strategy **j** fixates in a resident population of strategy **i**. The archived
upstream source in this repository says so directly,
`tmp/EGTTools-docs-upstream/src/egttools/analytical/sed_analytical.py`:

```python
def fixation_probability(self, invader: int, resident: int, beta: float, ...)   # line 571
...
fp = self.fixation_probability(second, first, beta, *args)                      # line 719
fixation_probabilities[first, second] = fp                                      # line 720
```

Row is the resident, column is the invader. The first version of
`scripts/build_egt_frontier_invasion.py` wrote that row out as `focal_strategy`
and that column as `opponent_strategy` in `egt_frontier_invasion.csv`, which
reads as the exact opposite. `scripts/build_supplementary_figures.py` then drew
each edge from `opponent_strategy` to `focal_strategy`, that is, from the true
invader to the true resident. Every arrow in the figure would have pointed the
wrong way, and only the buried arrowheads kept that invisible.

The numbers settle it without appeal to the source. At $p_r^{\max}=0.1$ the
finite-mutation chain sits on Always Unsafe 91.9\% of the time, and the cell
that equals 1.0 there is row AS, column AU. The only reading consistent with AU
dominance is "AU takes over an AS population", so the column is the invader.

`egt_frontier_invasion.csv` now carries explicit `resident_strategy` and
`invader_strategy` columns that cannot be read two ways, and keeps
`focal_strategy` and `opponent_strategy` as aliases with the meaning their names
imply: `focal_strategy` is the invader. The nine invasions that clear neutral
drift are:

| $p_r^{\max}$ | invasion | fixation |
|---|---|---|
| 0.1 | AU invades AS | 1.000 |
| 0.1 | CAS invades AS | 1.000 |
| 0.1 | AU invades CS | 1.000 |
| 0.1 | CAS invades CS | 1.000 |
| 0.6 | CAS invades AS | 1.000 |
| 0.6 | CAS invades CS | 0.997 |
| 0.9 | CAS invades AS | 1.000 |
| 0.9 | CS invades AU | 1.000 |
| 0.9 | CS invades CAS | 1.000 |

### 2c. The per-edge numbers are gone on purpose

Eight of the nine drawn edges are at 1.0000 and the ninth is at 0.9973, so the
old figure carried three or four labels all reading "1.00" stacked over the
middle of the panel, where no reader could assign any of them to an edge. Each
panel now states how many invasions clear drift and what the smallest fixation
probability among them is. The full matrix stays in
`egt_frontier_invasion.csv`.

### 2d. Replacement caption and Description

```latex
    \caption{Which strategy takes the population, at each configured risk. An
    arrow runs from the resident population to the strategy that invades it and
    fixates with probability above neutral drift $1/Z=0.01$, and carries the
    invader's colour. Disc area is that strategy's share of the independently
    simulated finite-mutation chain ($\beta=2$, $\mu=0.02$). The two layers
    answer different questions: the arrows are evaluated in the EGTtools
    small-mutation limit, whose stationary vector is not unique here because the
    transition matrix has more than one absorbing class, while the disc areas
    come from the finite-mutation chains. Strategies are Always Safe, Always
    Unsafe, and the two conditional rules that copy the opponent's last action
    after opening Safe (CS) or Unsafe (CAS). No language-model route plays this
    population process, and this is not a latent-strategy diagram for any of
    them.}
    \Description{Three network diagrams, one for each maximum private risk. Four
    discs are placed at the corners of a square, sized by each strategy's share
    of the population. Coloured arrows run from each resident strategy to the
    strategy that invades and takes over. At risk 0.1 Always Unsafe takes over
    and holds 92 percent. At 0.6 the conditional rule that opens Unsafe takes
    over and holds 96 percent. At 0.9 the conditional rule that opens Safe takes
    over and holds 94 percent.}
```

### 2e. One body sentence also needs changing

`paper/supplementary.tex`, just above the float:

> Its directed edges are computed by the archive-compatible EGTtools
> implementation.

EGTtools publishes no wheel for the CPython this repository runs, so the default
build redraws the archived numerical output rather than re-running the solver.
Suggested replacement:

> Its edges are the fixation probabilities produced by the archive-compatible
> EGTtools run whose output is stored in `egt_frontier_invasion.csv`; the figure
> redraws that archived output rather than re-running the solver, and the
> builder's `--recompute` flag re-runs it where the EGTtools environment is
> available.

---

## 3. A blocker the caption owner has to know about

`scripts/build_supplementary_figures.py` contains its own private copies of
these two figures, `build_egt_insights()` and `build_egt_invasion()`, registered
as `supplement_figure_1_egt_insights` and `supplement_figure_2_egt_invasion`.
`scripts/build_publication_figures.py` calls `build_supplementary_figures()`,
and `scripts/build_publication.py` calls that, so **any full build overwrites
`figures/paper/egt_frontier_insights.*` and `figures/paper/egt_frontier_invasion.*`
with the old broken drawings.** This happened twice during the redraw, minutes
apart, from another session's build.

Those two builders reproduce every defect listed above: no visible arrowheads,
share labels printed on top of their own discs, clustered "1.00" edge labels,
AS and CS in the same blue, uppercase panel letters, "10%/60%/90%", "Unsafe
rate", and a private brick-red and steel-blue pair used nowhere else. They must
be removed and delegated. The replacement inside
`build_supplementary_figures()` is:

```python
from scripts.build_egt_frontier_insights import main as build_egt_insights_main
from scripts.build_egt_frontier_invasion import main as build_egt_invasion_main
```

with the two entries in the `outputs` dict calling those and the two local
`build_egt_insights` / `build_egt_invasion` functions deleted. Note that both
modules write to `figures/paper/` themselves and re-apply `figstyle` rcParams,
so they should be called last, or `configure_publication_style()` should be
called again after them.

Until that is done the repair is not in the manuscript, whatever the working
tree says at any given moment.
