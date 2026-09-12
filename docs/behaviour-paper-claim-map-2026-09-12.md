# Behaviour paper: thesis and claim map

Written 2026-09-12 and superseded by the complete five-route scripted-opponent
campaign completed 2026-09-13. The historical three-route values below remain
as an audit trail, but the active manuscript claims must use the five-route
values in the dated amendment and the derived campaign tables.

Every number below was recomputed in this session from the artifact named
beside it. `scripts/verify_manuscript_claims.py` passes 179 of 179 against the
current manuscript. The handful of main-paper numbers that verifier does not
cover were recomputed separately and are marked. Two disagreements with the
current text are reported in section 6; neither is a wrong number, both are
scope statements that are missing.

---

## 1. Thesis and title

### The thesis, one sentence

**If decisions in a competitive race between AI developers were delegated to
frontier language models, the safety of those decisions would depend on both
the stated danger and what the rival is doing, with the magnitude varying by
checkpoint, and the population of delegates would converge on a handful of
near-deterministic policies that human participants never confine themselves
to.**

The supporting shape, which is what makes it one paper and not three findings:

1. Screened routes do respond to catastrophic risk, and they respond in a
   narrow band. Moving the stated maximum private risk from 0.1 to 0.9 against
   a rival that always plays Safe changes Unsafe play by 10.8 to 11.8 points
   on all three routes measured that way.
2. Changing the rival at fixed risk changes it by 45.0 to 73.5 points. The
   rival is worth roughly four to seven times what the risk is worth, and that
   contrast is causal by construction because the rival is code the route
   cannot influence and is never told about.
3. Because the rival dominates, a self-play rate is not a measurement of how a
   route treats risk. Self-play sits above the fixed-safe-rival rate in all
   nine route-by-risk cells.
4. And the delegate population is not a population. Twenty human participants
   produce effectively 18.9 to 19.7 distinct five-round action sequences out of
   twenty; every admitted route falls below every one of 20,000 matched human
   draws, and one produces a single sequence, repeated twenty times, at every
   risk level.

The safety framing is prospective and stays prospective. The paper says "as
strategic decisions are increasingly mediated by AI systems, this behaviour
becomes the behaviour that matters", never "companies delegate safety decisions
to language models". The game is an idealised race and models nothing else.

### Title

Current: *More Than the Risk: Frontier LLM Behavior in AI Development Races Depends on the Rival*

What is genuinely strong and must survive: it leads with an active finding
rather than a topic, it names the comparison population, and "AI Development
Races" is the search phrase the field indexes on. What has to go: "Audited"
puts the screen in the title of the behaviour paper, "Idealised" is a hedge
spending title space, and "Extreme Policies" is vague enough that a reader
cannot cite it as a fact.

Three alternatives, all built as active finding plus mechanism plus scope
phrase:

**A. Answering the Rival, Not the Risk: Frontier LLM Agents Collapse to a Few
Safety Policies in AI Development Races**

**B. The Rival Outweighs the Risk: How Frontier LLM Agents Would Behave if
Safety Decisions in AI Development Races Were Delegated to Them**

**C. One Policy Where Humans Use Twenty: Opponent-Driven Safety Behaviour of
Frontier LLM Agents in AI Development Races**

**Selected: More Than the Risk.** It states the opponent-dependence finding
without implying that routes ignore stated risk, which the data do not show. It
drops the audit from the title while keeping the paper's central mechanism
visible. The older alternatives remain below as discarded title candidates.
B is the most faithful to the delegation framing but is long and its conditional
mood reads as speculative in a title. C is the most vivid and
the best for talks, but "One Policy" is true of one route and an overstatement
of the other four, so it fails the evidence gate as a title.

One honest limit on title engineering here: the house standard asks for a
universal scope phrase such as "across languages". This repository cannot carry
one. Every confirmatory frontier run is a single prompt version,
`ai-race-fairgame-v3`, and no language field is recorded in any
`results/frontier/baseline_campaign_v6/**/run_manifest.json`. Do not put a
cross-lingual or multi-prompt scope phrase in the title. The available scope
phrase is the game and the human comparison, which is what A uses.

---

## 2. Claim inventory

RQ1 Risk: how frontier agents trade speed against safety as catastrophic risk
rises. RQ2 Rivalry: whether those choices are intrinsic or emerge from the
opponent. RQ3 Human and theory: whether agents reproduce what evolutionary game
theory predicts or what humans display.

### Claims that move into the behaviour paper

| # | Claim in plain language | Artifact | Number | RQ |
|---|---|---|---|---|
| B1 | Every admitted route plays Unsafe less as the stated catastrophic risk rises, and the four that respond gradually do so within a narrow band. | `results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.csv` | Risk response 38.2, 38.7, 40.3 and 57.0 points for GPT-5.5, Gemini 3 Flash, GPT-5.4 and Claude Sonnet 5 | RQ1 |
| B2 | One admitted route does not trade off at all; it switches. | same | Claude Opus 5 plays Unsafe on all 186 decisions at risk 0.1 and Safe on all 372 above it, a 100.0-point step | RQ1 |
| B3 | Even at the highest risk, admitted routes keep playing Unsafe about a third to a half of the time in self-play. | same | Risk-0.9 self-play rates 60.2, 55.4, 46.8, 32.3 percent for Gemini 3 Flash, GPT-5.5, GPT-5.4, Claude Sonnet 5 | RQ1 |
| B4 | The behavioural baseline is clean, so none of this is a parsing artifact. | `results/frontier/baseline_campaign_v6/**/run_manifest.json` | 30 races, 558 decisions, 0 parse failures on each of nine routes; 10 races per risk cell | RQ1 |
| B5 | Rerunning one route under the same frozen protocol reproduces its profile, so a route's number is not a run. | `results/frontier/baseline_replication/gemini-3-flash-preview` | 100.0 / 74.2 / 59.7 against 98.9 / 73.1 / 60.2; largest per-risk difference 1.1 points; risk response 38.7 to 40.3 | RQ1 |
| B6 | What the rival does moves play substantially, but the magnitude varies by checkpoint and overlaps the risk effect in the full set. | `results/derived/scripted_opponent_campaign/**/scripted_opponent_rates.csv` plus the paired contrasts in the same tree | Rival stance 25.3 to 87.0 points; risk 0.1 to 0.9 against a fixed Safe rival 10.4 to 69.9 points | RQ2 |
| B7 | Responding to the rival is general across the five screened routes tested, with one saturated boundary reversal. | `results/derived/scripted_opponent_campaign/` | Unsafe play follows the weak ordering in 14 of 15 route-by-risk cells, strictly in 12; all 15 paired lower bounds are positive, smallest +16.5 points | RQ2 |
| B8 | How strongly a route answers its rival is a property of the checkpoint, not of the game. | same | Rival stance at risk 0.9: Gemini 3 Flash 71.7 [65.6, 77.5], Claude Opus 5 59.3 [51.2, 66.5], GPT-5.4 49.8 [35.8, 60.5], GPT-5.5 61.7 [53.5, 69.1], Claude Sonnet 5 45.0 [42.0, 48.2] | RQ2 |
| B9 | Even the rival's first move alone changes the rest of the race, and most where risk is cheapest. | same | Opening-move contrast 25.1 [20.9, 29.7], 15.8 [8.3, 23.5], 11.5 [4.5, 19.7] points on Gemini 3 Flash | RQ2 |
| B10 | A self-play rate is an interaction outcome, not a stand-alone measurement of risk attitude. | same, against `audit_versus_behaviour.csv` | Self-play exceeds the fixed-safe rate in 13 of 15 cells; Claude Opus 5 reverses at risks 0.6 and 0.9 | RQ2 |
| B11 | One route keeps taking real risk against a rival that never does. | `results/derived/scripted_opponent_campaign/gpt-5.4-2026-03-05/` | GPT-5.4 plays Unsafe 36.6, 32.3 and 25.8 percent against Always Safe, the highest of the three at every risk level | RQ2 |
| B12 | The design fixes the rival and audits its realised moves, so the within-design direction is not a model-response artefact. | `results/derived/scripted_opponent_campaign/`, ingestion replay | 60 of 60 cells, 600 races, 5,580 route decisions and 5,580 scripted rival moves, 0 parse failures, 0 rival deviations, seat counterbalanced five and five in every cell | RQ2 |
| B13 | The evolutionary benchmark predicts a switch and the routes deliver a gradient, so read at its usual setting the theory does not describe them. | `scripts/reproduce_egt_model.py` outputs | At selection strength 2 the model predicts 99.2, 98.0 and 1.9 percent Unsafe; Gemini 3 Flash plays 98.9, 73.1, 60.2 and Claude Sonnet 5 89.2, 48.4, 32.3 | RQ3 |
| B14 | That mismatch is a property of the setting, not of the game, and the setting that fits the routes is the one the source study fits to its own humans. | `scripts/analyze_egt_beta_sensitivity.py` outputs | 90 stationary cells swept; at selection strength 0.01 with mutation 0.05 the model predicts 87.3, 63.9 and 38.0 percent, within 10.3 points of Claude Sonnet 5 and 15.1 of Gemini 3 Flash in root mean square error | RQ3 |
| B14a | The all-five analysis-only extension preserves the weak-selection reading for the four graded routes while Claude Opus 5 remains a shape exception. | `results/open_source/egt_reproduction/egt_admitted_route_summary.csv` and `.json` | The best well-mixed weak-selection cell is beta 0.01 with fixed mutation 0.05; graded-route RMSE is 5.1--15.4 points versus 32.9--36.6 at the reference cell, while Opus 5 is a near-step switch | RQ3 |
| B15 | Human participants use nearly the whole policy space and every screened route uses a sliver of it. | `results/derived/trajectory_diversity_confirmatory/trajectory_diversity_confirmatory.csv` | Humans reach an effective 18.9, 19.7 and 19.3 sequences out of 20 at mean pairwise distances 0.47, 0.50 and 0.50; the largest screened upper bound is 12.3 against a human lower bound of 14.8 | RQ3 |
| B16 | The compression is severe rather than marginal, and in one case total. | same | Claude Opus 5 produces one sequence, repeated by all 20 trajectories, at every risk level; Claude Sonnet 5 uses two at risk 0.1 | RQ3 |
| B17 | No admitted route reaches the human range even once, against a matched null. | same | All 15 admitted route-by-risk cells fall below every one of 20,000 matched human draws; human null minima 15, 17 and 16 distinct sequences out of 20 | RQ3 |
| B18 | The result survives a stricter null matched on independence, not only on sample size. | same | 18 of 27 cells fall below a ten-dyad null, never fewer than 16 over forty redraws | RQ3 |
| B19 | Two refused routes do reach the human range, which the paper discloses rather than omits. | same | GPT-5.4 mini 20, 17, 17 and GPT-5.4 nano 16, 19, 16 distinct sequences, inside the null at every risk level | RQ3, scope |
| B20 | Matching a human mean is not matching a human population, because the response rules differ. | `results/cross_model_pilot_synthesis/data/feature_importance_results.json` | Human play is organised by the opponent's previous action, 56.0 percent of total SHAP magnitude; Claude Sonnet 5 leads with the same feature at 48.0 percent; the two Gemini Flash checkpoints lead with assigned risk at 32.5 and 36.6 percent and GPT-5-nano with relative position at 41.2 percent; human model ROC AUC 0.628 against 0.974 for Claude Sonnet 5 | RQ3 |
| B21 | Trajectories carry recoverable population signal, but the populations overlap. | `results/cross_model_pilot_synthesis/data/population_identity_grouped.json` | Balanced accuracy 42.3 percent plus or minus 4.3 against a grouped permutation null mean of 12.4 percent, permutation p = 0.001 | RQ3 |
| B22 | A measured human risk preference does not predict human play, while a prompted risk persona rewrites it, so a persona is a policy instruction and not a preference. | `results/cross_model_pilot_synthesis/data/elicited_risk_by_archetype.json` and `data/persona_role_gradient.csv` | Elicited risk against Unsafe play r = -0.015, p = 0.79, n = 341; the persona sweep from its lowest to its highest level moves Unsafe play by 50.5 to 98.3 points across seven checkpoints | RQ3, discussion |
| B23 | An equivalent presentation of the same mechanism changes play, so a single reported rate is presentation-bound. | `scripts/analyze_frontier_context_mapping_cross.py` outputs | Swapping which opaque code denotes Safe raises Unsafe play 9.5 [5.6, 13.8] points on Gemini 3 Flash and 8.3 [5.4, 12.2] on Claude Sonnet 5, over 60 paired blocks each; narrative skin 10.8 [6.0, 15.6] and 3.4 [0.9, 7.4] | scope on RQ1 and RQ2 |
| B24 | Adding competitors raises Unsafe play wherever play is off the boundary. | `scripts/analyze_nplayer_matched.py` outputs | At risk 0.6: 68.8, 80.3, 98.3, 100.0 percent for two to five companies, paired contrasts +11.1, +28.2, +31.0; at risk 0.9: 58.0, 68.9, 83.8, 96.1 percent, contrasts +11.0, +28.1, +40.0 | RQ2, supplement |

### Claims that CANNOT move, because they belong to the audit story

These are the evaluation paper's material. None of them may be a result in the
behaviour paper. Items marked "screen" appear in the behaviour paper only in
the methods screen sentences of section 4, as a declared entry condition, and
in full in the supplement.

| # | Claim | Artifact | Number | Why it cannot move |
|---|---|---|---|---|
| A1 | Every route reads the stage-payoff matrix perfectly while state reconstruction ranges from 100 to 20 percent. | `results/frontier/admission_campaign_v6/**/admission.json` | 9 of 9 perfect on payoffs, 8 of 9 on rule recall; state reconstruction 1.00 to 0.20 | This is the finding that comprehension is uneven across subtasks. It is the evaluation paper's headline, not a behavioural result. |
| A2 | Format compliance does not discriminate comprehension. | same, plus baseline manifests | Parser satisfied on all 540 probe outputs and all 5,022 baseline decisions while state reconstruction spans the full range | Same. It is a claim about the measuring instrument. |
| A3 | No route computes a closed-form expected payoff, and five score zero. | same | Maximum 0.50, five routes at 0.00 | Diagnostic domain of the battery. No behavioural claim rests on it. |
| A4 | Measured comprehension is descriptively associated with the risk response. | `results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.csv` | Spearman rho 0.87, exact permutation p = 0.004, n = 9; 0.92, p = 0.003, n = 8 without Claude Opus 5 | This is the whole second audit subsection, and it is the evaluation paper's central evidence that the battery predicts anything. Moving it would rebuild the audit story inside the behaviour paper. |
| A5 | The screen carries no information the route name does not. | `figures/paper/audit_reads_the_name.pdf` and its source | 9 of 9 agreement with a size word in the name, exact permutation p = 1/126 = 0.008; 8 of 9 and p = 0.040 if `flash` counts | Audit self-criticism. Behaviour paper states the bare fact in the screen sentences (see section 4) and nothing more. |
| A6 | The screen moves between identical administrations. | `results/frontier/admission_campaign_v5` and `_v6` | State reconstruction moved 13.3 points, two answers in fifteen, on two of four routes; no verdict changed | screen, one clause |
| A7 | One route's verdict has moved, across a token-cap change. | `admission_campaign` v1, v5, v6 | Claude Sonnet 5 refused at 76.7 percent, then admitted at 81.7 and 85.0 | Evaluation paper. Behaviour paper does not need it and cannot afford it. |
| A8 | The verdict is stable over a wide threshold band and unstable in exactly one direction. | `scripts/verify_manuscript_claims.py` threshold sweep | Same admitted set at all 676 combinations inside the stable bands; 5.0 points of headroom on each condition; one achievable step down on state reconstruction admits Gemini 3.1 Flash-Lite | screen, one clause |
| A9 | Only one of the three gate conditions binds. | same | State reconstruction alone at 75 percent reproduces the admitted set; the other two refuse no route the first does not | screen, one clause |
| A10 | The screen does not partition behaviour. | `audit_versus_behaviour.csv` | Refused Gemini 3.1 Flash-Lite responds 37.6 points against an admitted minimum of 38.2, with overlapping intervals; refused mean 19.4 against admitted mean 54.8 | This is the audit paper's honest caveat. In the behaviour paper it would invite the reader to re-derive the audit analysis from the supplement. Keep it in the supplement and in the evaluation paper. |
| A11 | The routes flattest in risk are the ones that cannot rebuild the state, and their rates look ordinary. | same | GPT-5.4 nano 9.7 points and GPT-5.4 mini 7.0 points, at 20.0 and 66.7 percent state reconstruction, with Unsafe rates near 60 percent | Audit payoff argument. |
| A12 | One refused route's two symmetric seats disagree. | `scripts/analyze_seat_confound.py` outputs | GPT-5.4 nano's Unsafe rate differs 19.4 points [12.1, 25.9] between byte-identical seats, 30.0 points on the opening move | Belongs with the evaluation paper as a validity failure. The behaviour paper's own seat counterbalancing is stated in methods instead. |

---

## 3. Section outline and page budget

Eight content pages, hard. Measured from the built PDF, a page of this class
holds about 6,100 characters of rendered text, and the current body runs to
page 8 with references starting 4,803 characters down that page. The audit
material now occupying the main paper is about 1.8 pages: 0.38 pages of prose
plus the admission table in the first subsection, and 0.65 pages of prose plus
a full-width three-panel figure in the second. The methods screen costs back
about 0.15. **Net recovered: roughly 1.5 pages.** That is the budget the four
results subsections below are funded from. Do not spend it on longer sentences.

| Section | Pages | Contents |
|---|---|---|
| Title, abstract, keywords | 0.35 | Structured abstract, delegation framing in the first two sentences, at most one anchor number |
| 1. Introduction | 1.10 | The delegation scenario as motivation, the race as the controlled instrument, the three questions, the contribution, the scope sentence |
| 2. The race and what theory predicts | 0.85 | Two-player mechanism, displayed payoff matrix and risk equation, the four benchmark strategies as a formal table, N-player in two sentences pointing at the supplement. Carries the mechanism figure, full width. |
| 3. Design, units and the validity screen | 0.90 | Agent protocol, sealed simultaneous rounds, race as the unit, parse-failure contamination rule, decoding contract honesty, evidence strata table, and the eight screen sentences of section 4 below |
| 4.1 Risk response | 1.00 | B1 to B5. Needs a new column-width or full-width figure built from panels a and c of the existing audit figure, with the audit panel removed |
| 4.2 Opponent dependence | 1.10 | B6 to B12. The paper's only causal-by-construction design and its longest subsection. Carries the scripted-rival figure |
| 4.3 Theory comparison | 0.75 | B13, B14. Carries the theory figure |
| 4.4 Human diversity | 1.15 | B15 to B21, with B19 as an explicit disclosure. Carries the human-versus-model figure, full width |
| 5. Discussion, limitations, conclusion | 0.80 | B22 and B23 as discussion, the limitation block, the closing argument |
| **Total** | **8.00** | |

Figures: mechanism (full width), risk response (new), scripted rival (column),
theory (column), human versus model (full width). That is five. The house
budget is six to eight, so there is room for one more, and the obvious
candidate is a single panel putting the rival contrast beside the risk contrast
on one axis, which is the thesis in one picture and does not currently exist.
Tables: evidence strata, the benchmark-strategy definition table, and the
disclosed-arithmetic table if it survives the space contest.

Two structural notes for the writers.

The results order is fixed: risk, then opponent, then theory, then humans. It
is not the order the work was done in and must not be described as one. It is
the order of the argument: establish that the routes respond to the stated
danger at all, show that the rival matters several times more, ask whether
theory or humans predict either shape, and finish on the population result,
which is the finding a reader should leave with.

The admission table itself does not appear in the main paper. If a reviewer
needs to see which routes entered, the screen sentences name all nine.

---

## 4. The load-bearing question

**If the audit leaves the main paper, what must remain so that a reviewer can
see the behavioural panel was chosen by a pre-specified validity screen rather
than cherry-picked?**

### The argument

Five of nine routes are analysed. A reviewer's first question is why those
five, and the honest answer has to survive three specific attacks.

*Attack one: you picked the five that gave you the result.* The defence is not
a claim of good faith, it is the structure of the screen. The screen scores
answers to twenty frozen comprehension questions against ground truth. It never
reads a race, a rate, a trajectory or an outcome. A selection rule that cannot
see the outcome variable cannot be tuned to it, whatever order things were run
in. This is the strongest available defence and it must be stated explicitly,
because it does not depend on chronology.

It also has to be stated carefully, because the chronology is genuinely mixed
and a reviewer with the artifacts can reconstruct it. Comparing `completed_utc`
across every `run_manifest.json` under `results/frontier/admission_campaign_v5`,
`admission_campaign_v6` and `baseline_campaign_v6`: four routes were screened
before their gameplay baseline completed, and five completed gameplay first. Of
those five, two already carried a verdict from the earlier administration on
2026-09-07, so **three routes received their first screen verdict after their
own gameplay baseline had completed**, and two of the three are admitted,
GPT-5.4 and GPT-5.5. That is the exact exposure. It is small and it is not
fatal, because the screen scores probe answers against ground truth and never
reads a race, but it is why the defence must be structural rather than
chronological and why the word "pre-registered" cannot be used. What is true
and checkable is that the probe bank, the three conditions and their threshold
values were fixed before the campaign, that the same bank hash was used at
every administration, and that the earlier administration reached the same
verdict on all four routes it covered.

*Attack two: the threshold was chosen after seeing which routes it kept.* The
defence is the threshold sweep, and a one-clause version of it belongs in the
main paper rather than only in the supplement. The verdict is the same at all
676 threshold combinations inside a band five points wide on each condition. A
rule that only produces this answer at one exact setting would be a rule chosen
to produce it.

*Attack three: you discarded the inconvenient routes.* The defence is that
nothing was discarded. All nine routes played the same confirmatory baseline
with identical race counts and zero parse failures, and every refused route's
behaviour is reported in the supplement beside the five. The paper must say
this, and it must also say the one thing a writer will want to leave out:
among the refused routes are the only two that reach human-level trajectory
diversity. A paper that screens routes out and then omits that the screened-out
routes carry the most flattering-looking result on one measure is doing exactly
what the screen exists to prevent. Disclosing it is what makes the screen
credible; it also costs nothing, because the paper's claim is about screened
routes and stays true.

*What the screen must not be allowed to claim.* It must not be presented as a
measurement of understanding, because the repository's own evidence refuses
that reading on two counts: only one of the three conditions refuses anything,
and the admitted and refused sets coincide exactly with whether the route name
carries a size word. Both belong in the main paper's methods, not only in the
supplement, because a reviewer who finds a limitation in the supplement first
assumes there are more.

### The minimum sentences, verbatim

Place these in the methods section, immediately before the results. They are
written to be dropped in as a paragraph.

> Not every endpoint can track this game well enough for its choices to be read
> as strategy, so the set of routes analysed below was fixed by a comprehension
> screen that never looks at play. The screen is a frozen bank of twenty
> questions asked three times each, covering rule recall, reading the stage
> payoffs, rebuilding the current state from a short history, applying one state
> change, and scoring a finished race. A route enters the behavioural panel only
> if it answers at least 80 percent of its sixty questions correctly, and at
> least 75 percent within state reconstruction and within terminal scoring; the
> bank, the three conditions and their values were fixed before the campaign and
> the verdict is unchanged across every threshold combination within five points
> of each of them. Nine routes were screened and five entered: Gemini 3 Flash,
> Claude Opus 5, GPT-5.4, GPT-5.5 and Claude Sonnet 5. The four that did not
> enter were not discarded; all nine played the same confirmatory baseline, and
> every refused route's behaviour is reported beside the admitted five in the
> supplementary material, including the two refused routes that are the only
> ones in this study to reach human-level trajectory diversity. Three limits on
> the screen belong here rather than only in the supplement: rebuilding the
> current state is the only one of the three conditions that refuses any route,
> loosening it by one answer in fifteen would admit a sixth, and the admitted
> and refused sets coincide exactly with whether a route's name carries a size
> word, which nine endpoints cannot separate from comprehension. The screen is
> therefore a declared entry condition and not a measurement of understanding.
> No analysis below ranks routes by their screen scores, interpolates between
> them, or treats admitted against refused as a measured contrast.

Eight sentences. Every number in them is verified: the bank of twenty at three
repetitions and the three thresholds are in every
`results/frontier/admission_campaign_v6/**/admission.json`; the five names and
676 combinations are recomputed by `scripts/verify_manuscript_claims.py`; the
one-binding-condition, one-step-down and route-string facts are checks
"state reconstruction alone at 75% reproduces the admitted set", "one step down
on state reconstruction admits Gemini 3.1 Flash-Lite and nobody else" and the
9-of-9 permutation in the supplement's `audit_reads_the_name` material.

Do not write "pre-specified" or "pre-registered" anywhere near this paragraph.
The thresholds were fixed before the campaign, which is what the sentences say,
but three routes received their first verdict after their own gameplay, and a
reviewer who reads the manifests will find that. The sentences above are
defensible exactly as written; a stronger word would not be.

---

## 5. What moves out, and where it lands

| Moving out | Lands in | Notes |
|---|---|---|
| Results subsection "Validity gate: can agents follow the race?" (0.38 pages prose) | Supplement, existing section on the nine-route admission campaign | Already written there in more detail. No new writing needed. |
| The nine-route admission table | Supplement, already present as the admission table | Remove from main paper entirely. |
| Results subsection "Is measured comprehension associated with the risk response?" (0.65 pages) | Supplement, and it becomes the seed of the evaluation paper | The rho = 0.87 result has no home in the supplement yet in narrative form. Someone must write the supplement version, not just delete the main-paper one. **This is the one piece of real new writing the split requires.** |
| Figure `audit_versus_behaviour_v2.pdf`, panel b | Deleted from the main paper, kept in the supplement | Panel b is the audit scatter. |
| Figure `audit_versus_behaviour_v2.pdf`, panels a and c | **Stay, redrawn** as a new behaviour-only figure for section 4.1 | Panel a is the risk-response grid and panel c is the Claude Opus 5 step. Those are RQ1's evidence, currently living inside the audit figure. The figure generator must be changed, not the exported file. |
| The route-string finding, `audit_reads_the_name` | Supplement only, plus one clause in the screen paragraph | |
| Gate test-retest and threshold-sweep subsections | Already supplement-only; unchanged | One clause of each is promoted into the screen paragraph. |
| The seat-confound result for GPT-5.4 nano | Supplement | It is a validity failure of a refused route. |
| The two audit-framing paragraphs in the discussion, beginning "The lesson is not that an audited route understands the game" and the first of the three limitations | Rewritten around the behavioural claims | Do not simply delete. The limitation they carry, that endpoints differ in more than their probe scores, becomes a limitation about comparing checkpoints at all. |

### Scope note for the ARR evaluation paper

**What it would contain from this repository.** The nine-route frozen battery
over six comprehension domains and its full probe-level outputs. The finding
that reading the payoff matrix is solved by everyone while rebuilding the
current state ranges from 100 to 20 percent, which is the paper's actual
claim: comprehension in a repeated game is not one thing and the easy subtasks
carry no information. The finding that format compliance discriminates nothing,
measured over 540 probe outputs and 5,022 gameplay decisions with zero parse
failures. The threshold sensitivity, the test-retest movement of 13.3 points on
identical administrations, and the one verdict that moved across a token-cap
change. The descriptive association with the risk response, rho = 0.87 over
nine routes. The seat-asymmetry failure on a refused route.

**Why that is not yet a paper.** Its own supplement contains the refutation.
Sorting the nine routes on whether `mini`, `nano` or `lite` appears as a whole
word in the route name reproduces the battery's verdict exactly, 9 of 9, with
an exact permutation p of 1/126. On this sample the battery carries no
information that reading the model name does not already carry. An evaluation
paper whose instrument is empirically indistinguishable from a string match
will be rejected, and correctly. Everything else in the list is a property of
one battery on one game.

**What new work it needs.** Three things, in order of how much they buy.

1. **Routes where name and comprehension disagree.** The battery earns its cost
   only on an endpoint whose size word and whose state tracking point in
   opposite directions: a large route that fails state reconstruction, or a
   small route that passes it. Finding one is an empirical question and it may
   fail, in which case the honest paper is "comprehension screening in this
   family is not separable from model tier", which is also publishable but is a
   different paper. This must be attempted before the framing is chosen.
2. **New task families.** One game cannot support a claim about comprehension
   screening. The battery needs at least two more repeated-game environments
   with different state structures, so that a route's state-reconstruction
   score can be shown to transfer or not to transfer. **This repository does
   not have them.** There is a Public Goods Game in the cited prior work and an
   N-player variant of this race in the engine, but no frozen probe bank exists
   for either, and no route has been administered one. This is new collection,
   not new analysis.
3. **A comparison against cheaper screens.** The paper must show what the
   sixty-call battery buys over a single state-reconstruction question, over
   the route name, and over reported parse health. The repository already shows
   that two of the three gate conditions bind on no route, so the honest
   starting position is that most of the battery is redundant on this sample.

Until items 1 and 2 exist, the evaluation paper is offcuts. Say so to the
author now rather than after the behaviour paper ships and the audit material
has nowhere to go.

---

## 6. Where the split weakens the paper, and what to do

This is the section that does not agree with the plan.

**6.1 The paper loses its single most quotable sentence.** Right now the
strongest line in the manuscript is that the only two routes reaching human
trajectory diversity are the two the gate refuses. It is surprising, it is a
result about evaluation practice rather than about one game, and it is the kind
of sentence that gets cited as an established fact. The behaviour paper keeps
the result as a disclosure but loses the argument, and what replaces it, "the
five screened routes are all below every human draw", is safer and duller.
*What to do:* accept the loss on RQ3 and pay for it on RQ2. The scripted-rival
campaign is causal by construction, 36 of 36 cells, zero rival deviations, and
it currently gets 0.49 pages. Give it 1.10 and make the rival-versus-risk
comparison the paper's quotable line. That comparison, roughly eleven points
for the catastrophic risk against forty-five to seventy-four for the rival,
does not exist as a sentence anywhere in the current manuscript, and it is
better than the line being lost because it is causal rather than descriptive.

**6.2 The paper loses its only novel method and must replace it.** Stripped of
the audit, the behaviour paper is a known game, a known human dataset, a known
evolutionary benchmark, and frontier models. The methodological contribution has
to become the scripted-rival design and the mirror-match critique that follows
from it: a demonstration that every self-play rate in this literature, including
the ones in the papers this one cites, is an equilibrium of a policy meeting
itself and not a measurement of risk attitude. That claim is supported here on
three routes and nine cells and it generalises past this game. *What to do:*
state it as a contribution in the introduction, not as a caveat in the results.

**6.3 The rival-versus-risk comparison needs one small new analysis before it
can be the headline.** The rival contrasts are paired bootstrap contrasts with
intervals. The risk drop at a fixed rival, 10.75, 10.75 and 11.83 points, I
computed here as a raw difference of two cell rates from
`scripts/analyze_scripted_opponent.py`'s rates table. It has no interval and it
is not paired within a repetition block the way the rival contrasts are.
*What to do:* have the analysis lane add a paired within-repetition risk
contrast to the scripted campaign analyser, so the two halves of the headline
are computed the same way. This is new analysis over existing data, not new
collection. Until it exists, the comparison may be stated as a contrast of
magnitudes with the asymmetry flagged, never as two intervals side by side.

**6.4 The safety framing promises more than the results deliver.** The thesis
is a safety-delegation question; the results are about behavioural measurement.
A reviewer will notice that no result in this paper is about harm, about a
safety outcome, or about anything a company would do. *What to do:* keep the
delegation scenario to one paragraph of the introduction and one of the
discussion, and make the claims themselves behavioural throughout. Never write
that these models are unsafe or safe. The strongest defensible safety sentence
is that a delegate whose choices are set by the rival rather than by the stated
danger is a delegate whose behaviour cannot be predicted from the danger, which
is a statement about predictability and is fully supported.

**6.5 Human diversity now carries more weight on a single source.** With the
audit gone, RQ3 rests entirely on 340 complete trajectories from one published
human study of one game. That dependency existed before and was diluted by the
audit material; it is now exposed. *What to do:* state it in the limitations in
the same paragraph as the diversity claim, not in a threats section the reader
may skip, and do not add a second human dataset that was collected differently.

**6.6 A scope defect that the split makes worse.** The trajectory classifier
(B21) and the SHAP comparison (B20) are computed on a seven-checkpoint pilot
roster: `claude-opus-5`, `claude-sonnet-5`, `gemini-3-flash-preview`,
`gemini-3.1-flash-lite-preview`, `gemini-3.5-flash-lite`, `gpt-5-nano`,
`gpt-5.4-nano`, per
`results/cross_model_pilot_synthesis/data/population_identity_grouped.json` and
`data/feature_importance_results.json`. **That roster contains neither GPT-5.4
nor GPT-5.5**, which are two of the five screened routes, and it contains
`gpt-5-nano`, which is not in the nine at all. In the current manuscript these
two claims sit inside the confirmatory diversity subsection with no roster
statement, directly after nine-route confirmatory numbers. In the behaviour
paper, where RQ3 carries more weight, that is a real defect. *What to do:*
either say in the sentence which roster it is on, or move both to the
supplement. Do not leave them next to confirmatory numbers unlabelled.

**6.7 A cohort distinction to preserve.** The human sample is 340 complete
paired five-round trajectories for the diversity comparison and 341
participants for the elicited-risk correlation and the archetype clustering.
Both are correct and they are different cohorts. Keep them distinct; do not
unify them in the rewrite.

**6.8 A framing correction the split makes available and should take.** The
admitted mean risk response of 54.8 points averages one step policy with four
graded ones. Claude Opus 5 goes from 100 percent to 0 percent between risk 0.1
and 0.6 and contributes a 100.0-point "response" that is not a trade-off at
all. In the combined paper that average sat inside an audit argument where it
did no harm. As RQ1's headline it would be misleading. *What to do:* report the
four graded routes as a band, 38.2 to 57.0 points, and report Claude Opus 5
separately as a switch. This is more honest and it is also a better result,
because a 19-point band across four frontier routes from three vendors is a
tighter finding than an average that one route dominates.
