# Submission-readiness stress test — 2026-09-08

This note reviews the current manuscript, supplementary material, figure provenance,
analysis artifacts, and the repository publication checks.  It is intentionally
conservative: a result is treated as headline-ready only when its unit of analysis,
endpoint admission, provenance, and robustness support the claim being made.

## Verdict

**Not submission-ready yet.**  The paper has a strong audit-first core and unusually
good evidence-boundary language, but four blockers remain:

1. **The title currently overstates the diversity result.**  A sample-size-matched,
   clustering-free stress test of paired first-five-round trajectories finds that
   most checkpoints are much less diverse than humans, but GPT-5.4 nano is a clear
   exception.  At every risk cap its Hill-q=1 effective trajectory count is close to
   the rarefied human reference.  Therefore "Humans Are More Diverse" is not safe as
   an unqualified cross-model title.
2. **Seven-model headline language is stronger than the endpoint-admission evidence.**
   Only Gemini 3 Flash and Claude Sonnet 5 pass the current frontier gameplay admission.
   The other five checkpoints are useful descriptive pilots, but should not be treated
   as equivalent confirmatory evidence unless they receive endpoint-specific admission.
3. **Clustering/embedding robustness is incomplete.**  The HDBSCAN sensitivity sweep
   is still absent, and t-SNE is only a visualization.  Main-text claims should not
   depend on a single HDBSCAN setting or t-SNE geometry.
4. **Publication QA and layout still need a final gate.**  The current PDF uses the
   AAMAS '26 header and a placeholder submission ID.  The current main PDF has nine
   content pages before references.  If the target venue/year uses an eight-content-page
   review limit, one page still has to be removed.  Run the official target-year
   template and the repository publication checker immediately before submission.

## Important new stress-test result

New script:
`scripts/analyze_trajectory_diversity_rarefaction.py`

New data:
`results/cross_model_pilot_synthesis/data/trajectory_diversity_rarefaction.csv`

The analysis avoids HDBSCAN and t-SNE.  It compares the exact paired five-round action
trajectory (focal actions + opponent actions) within each risk condition.  Because the
two-player speed increments are fixed, the pre-round progress-gap path is deterministic
from those paired actions.  Human cells are rarefied to the LLM cell size (n=20) and
diversity is reported as Hill q=0 and q=1.

Current q=1 effective trajectory counts:

| Population | 10% | 60% | 90% |
|---|---:|---:|---:|
| Human, rarefied n=20 | 18.81 | 19.67 | 19.30 |
| GPT-5 nano | 10.17 | 7.19 | 10.00 |
| GPT-5.4 nano | **18.66** | **18.66** | **18.66** |
| Gemini 3 Flash | 1.00 | 9.33 | 10.72 |
| Gemini 3.1 Flash Lite | 1.00 | 2.77 | 2.87 |
| Gemini 3.5 Flash Lite | 3.63 | 11.49 | 11.49 |
| Claude Opus 5 | 1.00 | 1.00 | 1.00 |
| Claude Sonnet 5 | 4.50 | 2.74 | 1.48 |

This strengthens the *model-specific policy* claim and weakens the blanket *humans are
more diverse than LLMs* claim.  The cleanest title direction is therefore something like:

> **Audit-First Evaluation Reveals Model-Specific Policies in Idealised AI Development Races**

or

> **Aggregate Safety Rates Hide Model-Specific Policies in Idealised AI Development Races**

The human-diversity result can remain a substantive secondary finding: most tested
checkpoints are highly compressed relative to humans, with GPT-5.4 nano as an explicit
exception.

## Main-text figure triage

### Keep / strengthen

**Figure 1 — game mechanism.**
Keep it.  It is the highest-value explanatory figure.  Standardize the visual semantics:
Safe = blue-green/teal, Unsafe = vermillion, theory/reference = charcoal/black.  Never
encode Safe/Unsafe with red-vs-green alone; preserve labels/shapes.

**Current EGT-vs-frontier figure.**
Keep the basic comparison in the main paper, but make the estimand boundary impossible
to miss: EGT is a population-evolution benchmark; LLM points are prompted self-play.
If space is tight, use only the risk-response and frontier-minus-EGT panels from the
supplementary boundary figure.

**Human-vs-LLM predictive structure (SHAP).**
Promote the SHAP heatmap from supplementary material.  It is more informative than a
mean-rate bar chart because it directly shows that similar aggregate action rates can
come from different observable state dependence.

### Remove from main or demote

**Persona cartoon/taxonomy figure.**
Move out of the main paper.  It consumes substantial space while communicating a prompt
taxonomy that can be explained in 2–3 lines or a compact supplementary table.

**Mean Unsafe-rate bar chart.**
Remove from main.  The exact values are already in prose and the chart duplicates them.
Use the space for a robustness result.

**HDBSCAN composition figure.**
Demote until the clustering sweep is regenerated and archived.  The current repository
provenance states that the original clustering table/generator for this legacy artwork
is not present, so the current redraw is canvas normalization rather than a scientific
reconstruction.

**Population t-SNE figure.**
Demote to supplementary unless a stability panel across seeds/perplexities is supplied.
Do not make a scientific conclusion depend on apparent t-SNE separation.

**Current smoothed distribution figure.**
The idea is strong but the 3x3/log-KDE presentation is too dense at paper size and mixes
off-roster GPT-5.6 Luna/Terra into a seven-model main-paper story.  Redraw it using only
the declared roster or replace it with the direct rarefaction figure above.

**Rank-by-persona grouped bars.**
Keep only if the first sentence of the caption says that rank is observed/endogenous.
A stronger main-text replacement is an adjusted/fixed-state contrast; the detailed
grouped bars can live in supplementary material.

## Supplementary figures worth promoting

Priority order:

1. **SHAP heatmap** — promote to main.
2. **Human measured risk vs assigned LLM risk-persona** — strong anti-anthropomorphism
   result; promote if one additional main figure fits.
3. **EGT boundary analysis panels A/B** — promote if the EGT comparison is central.
4. **EGTtools invasion topology** — keep supplementary.  It is useful and attractive,
   but it is a hybrid diagnostic and should not be read as latent LLM strategy.
5. **Radar-style summaries** — do not promote; heatmaps/interval plots are easier to read.

## Experiments required before submission

### P0 — do these before calling the paper submission-ready

**1. Direct diversity robustness.**
The new rarefaction analysis is the first step.  Add a race/pair-aware uncertainty check
and, if the title keeps a diversity claim, report at least one embedding-free metric in
the main text.  Explicitly report the GPT-5.4 nano exception.

**2. HDBSCAN sensitivity sweep.**
Vary at minimum:
- `min_cluster_size in {10, 15, 20, 25}`
- `min_samples in {3, 6, 9}`

Report number of clusters, unclustered fraction, ARI/NMI or co-clustering stability, and
whether the population-level conclusion changes.  Archive the source table and generator.

**3. t-SNE stability / de-emphasis.**
Run multiple seeds and perplexities only as a visualization stress test.  Prefer a PCA
or original-space dispersion result for inference.

**4. Recover scientific provenance for the legacy HDBSCAN and rank figures.**
The current repository explicitly says their original source tables/generators are not
present.  Rebuild them from raw artifacts before treating them as publication-grade
scientific redraws.

**5. Final publication gate.**
Run:
`python scripts/build_publication.py`
`python scripts/check_publication.py --allow-placeholder-id`
during drafting, then without `--allow-placeholder-id` for the final package.  Resolve
all undefined references, overfull boxes, Type-3 fonts, and target-year template issues.

### P1 — high-value if API/compute budget allows

**Endpoint admission for every model used in a headline cross-model claim.**
Otherwise scope the seven-checkpoint results as descriptive historical pilots.

**Fully crossed representation robustness.**
Cross action-code mapping x narrative skin x endpoint x matched state/seed and retain the
comprehension gate.  The existing confounded context pilot should remain audit evidence,
not a causal headline.

**N-player rank deconfounding.**
Use fixed-state replay or condition on lagged Unsafe history, round, risk, and group size.
Observed rank is partly produced by earlier Unsafe choices, so raw rank contrasts are
selection/composition effects as well as strategic responses.

**Identical-prompt repeat reliability.**
At representative fixed states, repeat the same prompt multiple times and report flip
rate / action entropy.  This is especially important because some hosted routes do not
honor requested seeds or temperature in the same way.

**Race-level uncertainty.**
Use race/pair bootstrap or cluster-robust inference for key dynamic coefficients and
model x risk interactions.

## EGTtools presentation

For the main paper, prefer a simple risk-conditioned line/point plot:
risk cap -> theoretical stationary strategy mass or expected Unsafe rate, with frontier
self-play overlaid but clearly labeled as a different process.

For supplementary material, use three fixed-layout network panels (10%, 60%, 90%):
- fixed AS/AU/CS/CAS node positions across panels;
- node area = finite-mutation stationary mass;
- edge width = fixation probability above neutral drift;
- same edge scale across all panels;
- caption: "population-evolution benchmark; not a latent-strategy diagram for LLMs."

A higher-value stress test than another network is a **risk x selection-strength phase
map** showing the dominant evolutionary strategy or expected Unsafe rate across beta.
That would reveal whether the EGT reference conclusion depends on one selection-strength
choice.

## Visual system

Use one semantic palette consistently:

- Safe: `#009E73` (blue-green)
- Unsafe: `#D55E00` (vermillion)
- Human/reference: `#3A3A38`
- Theory: near-black/navy, often dashed
- Grid/secondary text: neutral gray

For signed coefficients, use a diverging blue/orange scale centered at zero.
For nonnegative SHAP shares, use a single sequential scale.  Avoid rainbow palettes.
Use marker shape/line style in addition to color, minimum 7–8 pt final-print text, and
direct labels when possible.

## Recommended main-paper visual stack after revision

A compact submission version should aim for approximately five scientific figures:

1. Game mechanism.
2. Validity/representation robustness summary.
3. EGT vs frontier risk response.
4. Sample-size-matched trajectory diversity +/or SHAP predictive structure.
5. N-player adjusted/fixed-state effect.

Put persona taxonomy, t-SNE, HDBSCAN composition, full distribution densities, detailed
rank/persona cells, EGT invasion networks, and full model tables in supplementary material.

This layout keeps the main paper aligned with its strongest defensible novelty:
**audit first, then compare model-specific strategic trajectories without equating
aggregate action rates with understanding, human-likeness, or general safety.**
