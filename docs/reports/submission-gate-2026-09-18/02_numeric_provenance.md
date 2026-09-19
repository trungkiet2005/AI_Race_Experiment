# 02 Numeric provenance: paper/main.tex

Date: 2026-09-18. Read-only audit. No repository file was edited and nothing was rebuilt.

Snapshot audited (unchanged from start to end of the audit):

| file | sha256 |
|---|---|
| `paper/main.tex` | `6a9a3490c06482aa8aec52719d2d9ccfbad7be0979e47d8813b00534db5a2025` |
| `paper/supplementary.tex` | `5da2e5248563e9d63e5962a1984e90b5a47004622233d29c9d61f7175fc133ed` |

Row-level evidence is in `02_numeric_provenance.csv`, with one row per number in main.tex (N001 to N199) in order of appearance. Line numbers refer to the snapshot above.

## 1. Verifier outcome

`python scripts/verify_manuscript_claims.py` ran cleanly with exit code 0:

> 247 of 247 claims verified against their artifact

There were no FAIL lines. One stale note: CLAUDE.md still says the verifier "recomputes all 68 headline numbers", but it now runs 247 checks.

How far the verifier reaches: it recomputes values from artifacts and pins them to constants that are hard-coded in its own source. It does **not** parse main.tex. The only exception is one check that looks for the `risk_response.pdf` include and label. So a "covered" number is one whose value the verifier asserts independently. If the prose changed and the verifier constant did not, the verifier would not catch it. For three numbers the verifier prints the value but does not assert it (GPT-5.4 Always-Safe rates 36.6/32.3/25.8; Sonnet 5 at 32.3; the location of the 87.0/25.3 extremes). This audit marks those as not covered and checked them separately.

## 2. Scope of extraction

Every number in the body, abstract, figure captions and equations of main.tex is included. That covers counts written as words ("thirteen of fifteen", "Four limitations"), game parameters and design constants. Three kinds of numbers are left out: the template preamble (float fractions, conference name and dates), the author block (emails), and LaTeX comments. main.tex currently has no tables, because the risk-response display is now a figure.

## 3. Counts

| status | covered by verifier | not covered | total |
|---|---|---|---|
| MATCH | 128 | 37 | 165 |
| ROUNDING | 2 | 1 | 3 |
| MISMATCH | 2 | 3 | 5 |
| UNTRACEABLE | 0 | 0 | 0 |
| NOT_A_RESULT | 0 | 26 | 26 |
| **total** | **132** | **67** | **199** |

The three ROUNDING rows are all acceptable at the stated precision:
- "19-point band" is 18.8 (N047, N049).
- "loses 98 points" is 97.90 (N115).

UNTRACEABLE is zero, but two sources sit outside `results/` and `data/`. The human counts (N137 to N143, N195) trace to `references/source_study_dataset/airace_deidentified_long.csv`, read through `scripts/analyze_trajectory_diversity_rarefaction.load_human`. The beta=2 cliff (N115, N116) is not stored in any artifact. It exists only inside `scripts/figures/fig_theory_versus_behaviour.py`, and I recomputed it in memory with that module's `compute()` and `report()`. The result is 97.90 points between risk 0.635 and 0.640, where 0.005 is the 201-point grid spacing, so the true cliff can only be narrower.

## 4. MISMATCH list, with proposed corrections

**M1. N098, L815.** Quote: "The fixed-safe and self-play designs differ in thirteen of fifteen cells".
Artifact: self-play is higher than the Always-Safe arm in 13 cells, lower in 1 (Opus at 0.6: 0.0% against 1.1%) and equal in 1 (Opus at 0.9: 0.0% against 0.0%). So the two designs differ in 14 of 15 cells. The verifier asserts "self-play exceeds the fixed-safe rival in 13 of 15", which is a different statement.
Proposed: "Self-play exceeds the fixed-safe rate in thirteen of fifteen cells;"

**M2. N099, L816.** Quote: "Claude Opus 5 reverses the direction at risks 0.6 and 0.9".
Artifact: only 0.6 is a reversal. At 0.9 both designs are exactly 0.0% (0 of 186 self-play decisions; 0 of 93 against Always Safe).
Proposed: "the exceptions are Claude Opus 5 at risk 0.6, where self-play is lower (0.0% against 1.1%), and at 0.9, where both are 0.0%; both rates are at or near the floor."

**M3. N189, L996.** Quote: "Exploratory classifier and feature analyses ... use a seven-checkpoint roster containing neither GPT-5.4 nor GPT-5.5".
Artifact: the classifier (`population_identity_grouped.json`, class_counts) uses 7 checkpoints plus human. The feature analysis (`feature_importance_results.json`, and supplementary Table `tab:feature-importance`) uses 9 checkpoints plus human, adding GPT-5.6 Luna and GPT-5.6 Terra. The "neither GPT-5.4 nor GPT-5.5" part is correct for both.
Proposed: "use pilot rosters of seven (classifier) and nine (feature analysis) checkpoints, containing neither GPT-5.4 nor GPT-5.5,"

**M4. N196, L1031.** Quote: "Four further limits are worth stating."
The paragraph actually lists five limits:
1. replay against live play;
2. two crossed routes;
3. the one-route group-size grid;
4. seed forwarding;
5. the private trade-off only.

Proposed: "Five further limits are worth stating." Alternatively, merge two of the sentences.

**M5. N199, L1038-1039.** Quote: "Seed forwarding is provider-specific: one route's SDK stripped the requested seed and another forwarded it unconfirmed."
Artifact: `sampling_seed_provenance.status` in the nine `baseline_campaign_v6` run manifests reads `not_applied_sdk_stripped_for_route` on the 3 Google routes and `forwarded_to_provider_application_unconfirmed` on the 6 OpenAI and Anthropic routes. This is leftover two-route wording, and it contradicts main.tex's own L589-591, which is correct.
Proposed: "Seed forwarding is provider-specific: the SDK stripped the requested seed on the three Google routes and forwarded it, unconfirmed, on the six OpenAI and Anthropic routes."

## 5. Main text against supplement

These numbers agree between main.tex and supplementary.tex:
- admission thresholds, 20/6/3 probes, 75.0 and 51.7;
- 30 races and 558 decisions;
- the 38.2 to 57.0 band (supplement race-weighting table);
- 1.1, 186, 38.7 and 40.3, and the 93.0 spread (supplement gives 7.0 to 100.0);
- mapping effects 8.3 and 9.5;
- the scripted campaign: 60 cells, 600 races, 5,580 decisions, 14/15 and 12/15 ordering, 97.8 and 100.0, +16.5, the 25.3 to 87.0 extremes and which cells they sit in, 10.4 to 69.9, GPT-5.4 at 36.6/32.3/25.8, Gemini near +70, GPT-5.4 near +50;
- EGT: 99.2/98.0/1.9, 87.3/63.9/38.0, 10.28 and 15.13, 90 cells, the 5.1 to 15.4 and 32.9 to 36.6 RMSE ranges, the 0.005 window;
- Gemini and Sonnet profiles;
- human flow 341/340/338, 2,888 observations and 172 clusters, dyads 48+52+68=168;
- 20,000 draws;
- dyad table 5+5+3=13 admitted and 7+7+4=18 of 27, floors 12/15/13;
- confirmatory diversity values 18.9/19.7/19.3, 0.467/0.495/0.497, 12 against 13, 0.414 against 0.269, the mini and nano counts;
- the two-checkpoint position analysis (supplement L2233).

Where they disagree or fail to support each other:
1. **The self-play against fixed-safe count (M1 and M2) is wrong in both documents in the same way.** Supplement L1650-1653 says "differ in thirteen of fifteen cells ... Claude Opus 5 is lower at risks 0.6 and 0.9". The supplement's own Table `tab:scripted-opponent` prints Opus self-play 0.0 and Always Safe 0.0 at 0.9, which contradicts "lower". Fix both documents together.
2. **Roster size (M3).** Main says seven checkpoints. The supplement's feature-importance table and archetype-coverage caption use the nine-checkpoint pilot. The supplement is internally consistent: it states seven for the classifier and nine for the feature analysis.
3. **Main L620-628 says the persona placebo, the "eight narrative skins", and the paired first-round, fixed-state replay and live-trajectory readings are "described in full in the supplementary material".** The supplement never mentions eight skins. It describes only the two-skin crossed rerun. The persona placebo appears only in a figure caption, and a text search finds no fixed-state replay or live-trajectory section. The count of 8 itself matches `ai_race/prompts/context_skins.py` (N027, MATCH), but the pointer to the supplement is not honoured.
4. **Seed handling (M5).** Main L1038 disagrees with main L589-591. Supplement L694-696 is correct because it is scoped to the two-route block. Supplement L879-881 says "every run manifest records `not_applied_sdk_stripped_for_route`". That is true only for the Gemini runs that paragraph discusses, and it reads as a general statement. Suggest "every run manifest for this route".

## 6. Abstract and introduction against results

There are no disagreements:
- five scripted-rival endpoints (abstract L305, results L753, discussion L1004);
- four graded profiles plus one switch (abstract L314, intro L461-462, results L657 and L673);
- 13 of 15 admitted cells below every dyad draw (abstract L317, results L660, L928, L956 and the Figure caption L969).

The abstract's qualitative wording, "can overlap the effect of changing the stated risk", matches results L783-787 and verifier check 219.

## 7. Other findings (not numeric mismatches)

- **N083, L783-786.** The sentence says the two effect families overlap "once the conditional rivals are included". But the two numbers it then cites, +16.5 as the smallest stance lower bound and +69.9 as the largest matched-arm risk contrast, are both without conditional rivals, and the matched-arm spans already overlap (25.3 to 87.0 against 10.4 to 69.9). Either drop the clause or cite the conditional-arm figure, +98.1 [94.4, 100.0] on Opus 5 against Conditional Safe.
- **N150, L928.** "Eighteen of 27" is one draw of a finite null. Verifier check 247 shows 16 to 20 over 40 redraws, and the verifier's own comment says "the body reports a range and not a floor", but the body prints only the point value. "13 of 15" is the published artifact value. Consider stating the redraw range, at least in the supplement.
- **N023, supplement L741-744.** The supplement's shares (91.9/96.3/93.9) come from `results/frontier/egt_frontier_comparison_v2/egt_stationary_summary.csv`. `results/open_source/egt_reproduction/egt_stationary_summary.csv` gives 92.4/96.5/94.2. These are two independently seeded chain sets with the same modal strategy, so the main text's AU, CAS, CS ordering holds either way.
- **Supplement-only error, L1158.** "sit six points apart on the admission audit": Gemini 3 Flash 93.3 against Claude Sonnet 5 85.0 overall is 8.3 points. The two are 6.7 apart only on state reconstruction (93.3 against 100.0).
- **N126, L880.** "beta=0.01 and mu=0.05 are the values the source study reports as its own best fit to human play" is a claim about the cited paper. It cannot be checked from the artifacts and should be checked against the source.
- **Typography, L728 and L779.** `Unsafe{}` is missing its backslash, so it prints as plain "Unsafe" rather than the \Unsafe{} small caps.
