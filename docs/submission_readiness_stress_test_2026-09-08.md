# Submission-readiness stress test - 2026-09-08, with a 2026-09-09 follow-up

This file is kept as a dated pair. The 2026-09-08 note is the original stress
test; the 2026-09-09 follow-up below records what closed each of its four
blockers and names the artifact that closed it. The original findings are
retained further down, unedited apart from a superseded marker, because the
reasoning behind them is still the reason the checks exist.

**Read the follow-up first.** The 2026-09-08 blocker list no longer describes the
repository: it says the seven-model language exceeds the admission evidence when
nine routes are now audited, and it says the PDF carries an AAMAS '26 header when
both manuscript sources now use the AAMAS 2027 template.

## Follow-up, 2026-09-09: what closed each blocker

| 2026-09-08 blocker | Status on 2026-09-09 | Artifact that settles it |
|---|---|---|
| 1. Title overstates the diversity result | **Addressed, with one wording caveat** | `results/cross_model_pilot_synthesis/data/trajectory_diversity_rarefaction.csv`, `scripts/build_diversity_figure.py`, `figures/paper/trajectory_diversity_main.pdf`, and the scoping paragraphs in `paper/main.tex` |
| 2. Seven-model language exceeds the admission evidence | **Superseded** | `results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv` and `results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.csv` |
| 3. Clustering and embedding robustness incomplete | **Closed** | `results/cross_model_pilot_synthesis/data/human_archetype_k_sensitivity.csv` and `.json`, `trajectory_hdbscan_robustness.csv`, `trajectory_tsne_robustness.csv`, `trajectory_clustering_robustness.json` |
| 4. AAMAS '26 header and nine content pages | **Closed on the template and the page budget; the placeholder submission ID is still open** | `paper/main.tex` and `paper/supplementary.tex` preamble, and the built `paper/ai_race_paper.pdf` |

### 1. The diversity claim is now quantitative, and it is scoped

The diversity numbers are no longer an assertion about an embedding. Every cell
in `results/cross_model_pilot_synthesis/data/trajectory_diversity_rarefaction.csv`
is rarefied to 20 trajectories and carries a cluster-bootstrap interval on both
statistics it reports, Hill q=1 and mean pairwise Hamming distance, with races as
the resampling cluster for the checkpoints and participants for the humans; the
columns are `q1_ci_low` / `q1_ci_high` and `hamming_ci_low` / `hamming_ci_high`.
`scripts/build_diversity_figure.py` draws the main-paper panel from that table
alone, so a presentation change can never move a published value.

The substance the 2026-09-08 note worried about is now stated rather than
smoothed over. `paper/main.tex` reports the GPT-5.4 nano exception explicitly, at
an effective 18.7 distinct sequences at every risk level with intervals that
overlap the human reference, and immediately records that this is also the
checkpoint that fails the comprehension gate most severely, reconstructing the
state in 20 percent of probes at 51.7 percent overall. The claim it makes is the
scoped one: among the checkpoints that pass the gate, every one is strictly less
diverse than the human sample at every risk level.

The caveat, stated plainly rather than declared closed: the title's lead clause is
still "Humans Are More Diverse", now qualified by "Audited Frontier LLMs" in
`paper/main.tex`. The scoping lives in that qualifier and in the body text, not in
the lead clause, which is a judgement call for the manuscript owners rather than a
missing artifact. Note also that the built `paper/ai_race_paper.pdf` still shows
the earlier variant of the title without "Audited", so it is behind the source.

### 2. Nine routes are audited, five admitted, eight have gameplay

The blocker asked for endpoint-specific admission before any cross-model headline
language. That evidence now exists.
`results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv`
records nine routes under protocol `ai-race-frontier-admission-v6`, each with 60
retained rows from 20 frozen probes at three repetitions, and five admitted:
`google/gemini-3-flash-preview` 0.9333, `anthropic/claude-opus-5@default` 0.9167,
`openai/gpt-5.4-2026-03-05` 0.9000, `openai/gpt-5.5-2026-04-23` 0.9000, and
`anthropic/claude-sonnet-5@default` 0.8500. Four are refused:
`google/gemini-3.1-flash-lite-preview` 0.8000, `openai/gpt-5.4-mini-2026-03-17`
0.7500, `google/gemini-3.5-flash-lite` 0.6500, and
`openai/gpt-5.4-nano-2026-03-17` 0.5167.

`results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.csv` then
gives eight of those routes a matched gameplay baseline under
`ai-race-frontier-baseline-v3`, 30 races and 558 decisions and zero parse
failures each, and joins each route's measured probe accuracy to its measured
risk response over the eight. `google/gemini-3.5-flash-lite` failed in transport
with zero races and is retained as a failure record, so it appears in the audit
and in no behavioural table.

Two limits carry forward rather than closing. The route-availability limit is
real: `gpt-5-nano` is one of the seven checkpoints in the diversity comparison and
is no longer offered on the audited identity, so six of those seven carry a
verdict and the seventh never can. And the admission campaign's own
`expected_payoff` domain is the weakest domain on all nine routes, which is why it
is frozen as diagnostic and never gates admission; no claim may rest on a route's
expected-payoff arithmetic.

### 3. The sweeps the note asked for exist

- **k-means archetypes.** `human_archetype_k_sensitivity.csv` sweeps k from 2 to 6
  on the exact 341-by-5 standardized matrix the paper clusters, imported from the
  clustering generator rather than reimplemented, and reports silhouette,
  Calinski-Harabasz, Davies-Bouldin and inertia together with a 500-draw
  participant-level bootstrap adjusted-Rand stability interval at every k. The
  published k=4 has the best silhouette in the sweep, 0.316, with a mean ARI of
  0.626 and a 95 percent interval of 0.391 to 0.976;
  `human_archetype_k_projection.csv` recomputes the paper's occupancy comparison
  at every k, because that claim is comparative rather than about k.
- **HDBSCAN.** `trajectory_hdbscan_robustness.csv` covers exactly the grid the
  note demanded, `min_cluster_size` in {10, 15, 20, 25} crossed with
  `min_samples` in {3, 6, 9}, reporting cluster count, noise fraction,
  largest-cluster fraction, per-population clustered fraction, and adjusted Rand
  against the baseline setting.
- **t-SNE.** `trajectory_tsne_robustness.csv` runs perplexities {5, 15, 30, 50}
  across three seeds with trustworthiness and neighbourhood-Jaccard agreement
  against the baseline layout. `trajectory_clustering_robustness.json` carries the
  grids, package versions and source hashes, and records the evidence class as
  diagnostic, so no inference depends on the embedding geometry.

### 4. Template and page budget

Both `paper/main.tex` and `paper/supplementary.tex` now declare
`\documentclass[sigconf,anonymous]{aamas}` with
`\acmConference[AAMAS '27]{... (AAMAS 2027)}{May 3 -- 7, 2027}{Hanoi, Vietnam}`
and `\copyrightyear{2027}`. The AAMAS '26 header is gone.

The page budget is met as built. `paper/ai_race_paper.pdf` is 9 pages and the
`REFERENCES` heading begins on page 8, so the main content occupies 8 pages and
the references run into the ninth. AAMAS 2027 allows at most 8 pages of main
content with unlimited additional reference pages, so no page has to be removed
at the moment. A reflow that pushes content onto page 9 would break that, so the
count has to be rechecked after every content edit.

What is still open here: the submission ID is a placeholder, so
`scripts/check_publication.py` still needs `--allow-placeholder-id` and must be
rerun without it once AAMAS assigns the anonymous ID. The built PDF is also
behind the manuscript source, as noted above.

### Still open after this follow-up

1. Placeholder submission ID, and a final `build_publication.py` plus
   `check_publication.py` pass without the placeholder flag.
2. Rebuild the PDFs from the current sources; the committed PDF predates the
   current title and abstract wording.
3. The fully crossed representation robustness under P1 is complete for one route
   only, `google/gemini-3-flash-preview`, at 120 races and 2,232 decisions with a
   passing independent validator. The matched Claude run is not present in this
   repository as an artifact.
4. The N-player frontier rerun is still not admitted; its attempts are retained
   under `results/failed_runs/`.
5. The figure triage, palette and visual-stack recommendations in the original
   note were not part of this follow-up and have not been re-audited against the
   current figure set.

---

## Original note, 2026-09-08 (superseded, retained as history)

Everything below is the 2026-09-08 note as written. Its blocker list is
superseded by the follow-up above; read it for the reasoning, not for the status.

This note reviews the current manuscript, supplementary material, figure provenance,
analysis artifacts, and the repository publication checks.  It is intentionally
conservative: a result is treated as headline-ready only when its unit of analysis,
endpoint admission, provenance, and robustness support the claim being made.

### Verdict

**Not submission-ready yet.** *(Superseded: see the 2026-09-09 follow-up above.)*  The paper has a strong audit-first core and unusually
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

### Important new stress-test result

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

### Main-text figure triage

#### Keep / strengthen

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

#### Remove from main or demote

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

### Supplementary figures worth promoting

Priority order:

1. **SHAP heatmap** — promote to main.
2. **Human measured risk vs assigned LLM risk-persona** — strong anti-anthropomorphism
   result; promote if one additional main figure fits.
3. **EGT boundary analysis panels A/B** — promote if the EGT comparison is central.
4. **EGTtools invasion topology** — keep supplementary.  It is useful and attractive,
   but it is a hybrid diagnostic and should not be read as latent LLM strategy.
5. **Radar-style summaries** — do not promote; heatmaps/interval plots are easier to read.

### Experiments required before submission

#### P0 — do these before calling the paper submission-ready

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

#### P1 — high-value if API/compute budget allows

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

### EGTtools presentation

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

### Visual system

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

### Recommended main-paper visual stack after revision

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
