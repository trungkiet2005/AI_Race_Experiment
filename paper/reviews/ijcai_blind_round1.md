# IJCAI 2027 main track: blind simulated review, round 1

Date 2026-11-22. Paper: "Self-Play Can Mask Frontier LLM Risk Responses in AI Development Races" (`paper/main.tex`, AAMAS-format body, plus `paper/supplementary.tex`).

## 0. Conditions of this review

- **Blind.** I did not open the contents of `paper/reviews/` (only listed its existence), `STATE.md`, `docs/` or any earlier review or fix list. The harness injected the project `CLAUDE.md` into context after I started. I ignored its framing. I used it only to learn which runners exist (section 5 below). Every criticism below comes from `main.tex` and `supplementary.tex` as text. I read `main.tex` in full and used the supplement's roadmap, scripted-opponent design paragraphs and section list. I did **not** read the whole supplement (3,449 lines), so "not in the supplement" claims apply to what I read and need a grep before anyone acts on them.
- **Criteria used.** The CFP (https://2027.ijcai.org/ijcai-2027-call-for-papers-main-track/, fetched today) states the selection criteria: novelty and originality of ideas, correctness, clarity, significance of results, potential impact, quality of presentation. It also gives: 7 body pages with figures and tables, plus 2 pages of references only; overlength is rejected without review; Phase 1 sends each paper to two PC members and summarily rejects papers with two insufficiently positive reviews, with no author response; the reproducibility checklist is mandatory; LLMs may only polish text the authors wrote, and LLM-generated text in methodology or analysis must be labelled and described. The reviewer form itself is not public. I therefore used the known IJCAI-style form: relevance, originality/novelty, technical quality/soundness, significance, clarity, reproducibility, overall, confidence. Per-criterion scores are 1 to 5. Overall is on a 1 to 10 scale where 1 to 3 is reject, 4 to 5 is borderline or weak reject, 6 is weak accept and 7 or more is accept. This scale is my approximation, not the official form.
- **Page-count fact.** The AAMAS body is 8 pages, the IJCAI body is 7. The one-line premise "content unchanged" is therefore an issue in itself; see section 6.

## 1. Review A: MAS and game-theory reviewer

**Summary.** The paper studies a two-player repeated "AI race" (stage-dominant Unsafe, private setback risk that bites only the winner). Five commercial LLM endpoints that pass a probe screen are run (i) in self-play at three risk levels and (ii) against four code-defined rivals (AS, CS, CAS, AU), ten races per cell. It claims that the self-play risk response is "opponent-conditioned". The AU-minus-AS contrast is positive in 15 of 15 cells, and the rival-by-risk interaction grows for Opus 5 and GPT-5.5, shrinks for Sonnet 5 and is unresolved for Gemini 3 Flash and GPT-5.4 (Table 2). Two supporting checks follow. Graded profiles are closer to a weak-selection EGT benchmark than a strong-selection one. Endpoint samples occupy fewer five-round trajectories than matched human dyads in 13 of 15 cells.

**Strengths.**
1. The identification question is clear and the fixed-rival design is the right instrument for it. Rival is exogenous, seat counterbalanced, horizon differenced out by common random numbers, and all 5,580 rival moves were replayed. This is better-audited than most LLM-game papers.
2. Scoping is honest: boundary cells are flagged, "unresolved" is not read as zero, contract heterogeneity is admitted, four graded endpoints are reported as a band rather than averaged with the switch.
3. The self-play versus fixed-rival gap for Gemini 3 Flash (98.9/73.1/60.2 against 24.7/17.2/14.0) is a vivid, correct illustration.

**Weaknesses.**
1. **No normative baseline, so the headline is close to tautological.** In this game the best reply plainly depends on the rival. Against AU, playing Safe forfeits the prize but also escapes the winner-only setback draw. Against AS, Unsafe wins the race but puts the endpoint at risk. A positive AU-minus-AS contrast is what a perfectly rational agent would produce. The paper never computes the expected-payoff-optimal policy against each fixed rival. It cannot say whether an endpoint over- or under-reacts, or whether the endpoint-specific interaction signs are rational adaptation or prompt artefact. The rival is fixed and the horizon distribution is known to the authors, so this is a small dynamic program. Without it, "opponent-conditioned" is true of any agent that responds to incentives, and the novelty reduces to a measurement-design recommendation.
2. **Self-play versus scripted rival is not a like-for-like contrast.** The mirror opponent is an adaptive policy that is not in the four-point set. "Self-play exceeds fixed-safe in 13 of 15 cells" is nearly mechanical, because a rival that sometimes plays Unsafe invites Unsafe. The natural comparators are cross-play against a different LLM, a mixed pool, or an adaptive best-response rival. The paper lists these as future work (Discussion, last paragraphs). They are the experiments that would show self-play "masks" anything.
3. **The title and abstract overclaim "mask".** Self-play is non-increasing in risk on all five endpoints and so is the fixed-safe curve. No result shows a qualitatively different risk response that self-play hides. What is shown is that levels and the AU-minus-AS interaction differ by rival, which is a weaker statement than "mask".
4. **EGT section is weak as evidence.** "Closer to weak selection than strong" compares a step-like model to graded data. Almost any graded profile is closer to the graded model. The best cell was picked from 90 by RMSE over three points per endpoint. The defence ("also the source study's human-fitted values") is useful but it makes the benchmark a human-fit comparison, not evidence about LLMs. At beta near 0.01 the stationary distribution is close to drift, so the match says little about payoff-driven behaviour. The text says the cell was "not selected solely to fit" yet also calls it "the closest cell in the entire sweep"; the logic is muddled.
5. **Four rivals are four chosen points.** The interaction claim ("endpoint-specific shape") is conditional on AU and AS, the two extremes, where saturation is most likely. The paper itself notes a ceiling case (GPT-5.5 at risk 0.1, CAS 100.0 versus AU 97.8).

**Questions.** What is the exact best-response policy to each rival at each risk, and where do the five endpoints sit relative to it? Does the interaction sign follow from the best response? Why is no cross-play or mixed-pool cell run when the Discussion says that is the correct target?

**Scores (1-5).** Relevance 4. Originality 2. Soundness 3. Significance 2. Clarity 3. Reproducibility 3. **Overall 5/10** (borderline, leaning reject). **Confidence 4/5.**

## 2. Review B: LLM-evaluation and NLP reviewer

**Summary.** As above, read as an LLM-evaluation methodology paper: fix and audit the opponent, report trajectories, screen for task validity.

**Strengths.**
1. Careful provenance: strict parser, zero parse failures, whole race dropped on one failure, raw prompts and responses retained, a replay audit, a claim-recomputing checker.
2. The crossed narrative-by-code presentation check (Table 1) is a welcome control. It shows 8 to 11 point shifts from semantically equivalent renderings. That is itself an honest warning about prompt dependence.
3. Contract limits (temperature not forwarded, seed handling, GPT-5.5 decoding, one legacy Opus cell) are disclosed rather than hidden.

**Weaknesses.**
1. **Prompt artefact risk is not closed for the central claim.** The presentation check covers self-play on two endpoints only. The headline result, the endpoint-specific rival-by-risk interaction, is measured under one prompt version and one rendering. Table 1 shows rendering effects of up to about 11 points, which is larger than the unresolved interactions (-1.8, +2.3) and comparable to the contrast size. No scripted-rival cell is crossed with a rendering, so "endpoint-specific shape" could be "endpoint-by-rendering shape".
2. **Endpoint effects are confounded with decoding contract.** GPT-5.5 ran with reasoning low and a 1,024-token cap, the others mostly with reasoning none and 256 tokens, and one Opus cell on a legacy contract. GPT-5.5 is one of the two endpoints whose interaction is largest and positive. "Endpoint-specific" cannot be separated from "reasoning-budget-specific", and the paper concedes cross-endpoint comparison is "descriptive". The claim "three distinct shapes" is a cross-endpoint comparison, so it is not supported beyond description.
3. **Hosted endpoints, unconfirmed temperature and seed, no open-weight model in the main text.** Preview models get deprecated, so the reproducibility checklist cannot really be met. The "released checker recomputes quantities" helps but the data cannot be regenerated. The trajectory-concentration claim in particular depends on decoding temperature, which was not controlled or confirmed.
4. **Diversity comparison is apples to oranges.** One endpoint sampled 20 times at provider-default decoding is compared with many different people. Fewer distinct paths from one fixed policy than from a population is expected from any single agent, human or LLM. The fair comparison is within-person repeated play, or a pool across endpoints or personas against humans. The paper frames this as "a second failure of aggregate evaluation", which the design cannot support. The excluded mini and nano endpoints are the only ones inside the human null, and the paper dismisses them because "the screen refuses both". The screen was defined after gameplay for 3 of 9 routes, so exclusion of exactly the counter-examples is a forking-paths concern.
5. **The screen is weakly justified.** The admitted/refused split exactly matches the absence/presence of "mini", "nano" or "lite" in the name (the paper says so). Two admitted endpoints clear one domain at the minimum passing score. It is not preregistered, and it flips with one probe answer (Gemini 3.1 Flash-Lite at 73.3%). Everything then rests on the five largest endpoints, so claims about "frontier LLM" generalise from a size-selected set.
6. **No related-work section in the body.** Positioning against LLM game-theory benchmarks and LLM-versus-LLM and LLM-versus-fixed-opponent studies is spread across the introduction. Novelty is hard to assess.
7. The AI-assistance statement (supplement end) says exact model versions and histories were not retained and that Claude Code and Codex did "manuscript editing". IJCAI allows LLMs only to polish text authors wrote, and requires labelled description of LLM use. Needs to be in the main paper and match the policy.

**Questions.** Does the interaction pattern survive reasoning-budget harmonisation, rendering changes and an open-weight model with controlled temperature? Is concentration just low effective temperature? Do the four refused endpoints show the same scripted-rival pattern?

**Scores (1-5).** Relevance 4. Originality 3. Soundness 3. Significance 3. Clarity 3. Reproducibility 2. **Overall 5/10.** **Confidence 4/5.**

## 3. Review C: hostile statistics and methods reviewer

**Summary.** Evidence is 10 races per endpoint-risk-rival cell, with bootstrap intervals over ten matched repetitions, interpreted as three distinct interaction shapes across five endpoints.

**Strengths.** Paired CRN design is right; clustering by race is right; the "unresolved is not zero" caveat is stated; headline counts (13/15, 14/15, 12/15) are internally consistent in the text I checked.

**Weaknesses.**
1. **n is too small for the interpretive weight.** The interaction is a difference of differences from ten blocks per cell, and the paper builds a three-way taxonomy on it. Percentile bootstrap with 10 clusters is anti-conservative; the true coverage of a nominal 95% interval is well below 95%. The two "unresolved" endpoints are exactly where more data would decide whether the taxonomy is three shapes or two. The paper says so itself.
2. **Scale dependence of the interaction.** The interaction is on the raw rate scale, bounded in [0,1]. Opus 5 and GPT-5.5 sit at the ceiling at low risk (GPT-5.5: CAS 100.0, AU 97.8 at risk 0.1; Opus 100% Unsafe at 0.1 against self). A ceiling-compressed AU-minus-AS contrast at 0.1 mechanically produces a positive risk interaction. On a log-odds scale it may vanish or flip. "Rival effect grows with risk" is not interpretable without a scale-free analysis or a model-based (cluster-robust logit with rival-by-risk) estimate.
3. **Multiplicity.** 15 contrasts, 5 interactions, 4 presentation contrasts, 27 trajectory cells and a 90-cell EGT sweep, with no correction or global statement. The smallest AU-minus-AS lower bound (+16.5) is safe, but the interaction and presentation claims are not (Sonnet abstract-contest +3.4 [+0.9, +7.4] is marginal).
4. **Forking paths in the EGT fit.** Best of 90 cells by RMSE on three points per endpoint is selected on the outcome. The reported "within 9.6 and 15.4 points" is an in-sample minimum.
5. **Diversity statistic.** "Fewer distinct paths than the minimum of 20,000 resampled dyad draws" is an extreme-order-statistic test whose threshold depends on the number of draws, not a p-value or effect size. Richness (distinct counts) is sample-size sensitive. Trajectories are two seat-views per race, so n is 10 independent races per risk, not 20. The text does mention equal sample size and the cluster-aware null; the headline should still be given as an effect size with an interval, not as "13 of 15 below the floor".
6. **Confirmatory label.** The paper calls the campaigns confirmatory. The screen verdicts for 3 of 9 routes came after gameplay, GPT-5.5's cells ran under a different contract, one Opus cell is legacy, and the paper does not report a time-stamped preregistration of the interaction hypothesis. The word "confirmatory" is stronger than the record shown.
7. **Unconfirmed decoding.** Temperature 0.7 was requested but never forwarded on any baseline endpoint, effective seed unconfirmed. Replicability of any single number is limited to "same provider, same period". The stability evidence (Gemini repeat, within 1.1 points) is one endpoint.
8. **Mild inconsistencies to fix** (not fatal): abstract says "four graded profiles closer to weak selection", Figure 3 caption says all five; "20 of its trajectories at every risk level" reads as 20 in total; "ten matched repetitions per endpoint, rival and risk condition" while seats are 5/5, so each seat sees five.

**Questions.** What are the interval widths after a small-sample correction? What is the interaction on the log-odds scale? What is the family-wise error across the claims?

**Scores (1-5).** Relevance 3. Originality 2. Soundness 2. Significance 2. Clarity 3. Reproducibility 2. **Overall 3/10** (reject). **Confidence 5/5.**

### Score table

| | Relev. | Orig. | Sound. | Signif. | Clarity | Repro. | Overall (1-10) | Conf. |
|---|---|---|---|---|---|---|---|---|
| A (MAS/GT) | 4 | 2 | 3 | 2 | 3 | 3 | 5 | 4 |
| B (LLM eval) | 4 | 3 | 3 | 3 | 3 | 2 | 5 | 4 |
| C (stats) | 3 | 2 | 2 | 2 | 3 | 2 | 3 | 5 |
| mean | 3.7 | 2.3 | 2.7 | 2.3 | 3.0 | 2.3 | **4.3** | 4.3 |

## 4. Area Chair meta-review (hostile)

**Recommendation: reject.** The paper's identification instinct is good and its audit trail is better than average, but the core claim is either obvious or under-supported, depending on reading.

- If "opponent-conditioned" means the response to risk changes with the rival, the claim is expected from game theory, and the paper gives no best-response baseline to show deviation from rational adaptation (Review A).
- If it means endpoints differ in the shape of that dependence, the evidence is 10 races per cell, a different decoding contract on one of the two endpoints with the largest effect, one rendering, an anti-conservative bootstrap, and a bounded-scale interaction (Reviews B, C). Those do not support a three-way taxonomy.
- The two "supporting checks" are not supporting. The EGT match is a best-of-90 fit to three points and the diversity comparison confounds single-policy sampling with population heterogeneity. Both consume body space the central claim needs.
- Title and abstract say "mask"; nothing in the results shows a masked qualitative response.
- The reproducibility story rests on closed, preview, temperature-unconfirmed endpoints.
- Integrity: no number in the text I checked contradicts another (e.g. 38.7 to 40.3, 13/15 and 21/27, 12 and 13 distinct trajectories, 18/27 versus 21/27, the 54.8 average). Remaining concerns are claims without support: "mask", "second failure of aggregate evaluation", "minimum reporting package" (the paper did not test that the package works), "confirmatory" for post-hoc screen and amended contracts, and the AI-assistance statement being inadequate for IJCAI policy.
- Phase 1 risk: two borderline-to-negative PC reviews are likely, which means summary rejection with no author response.

**Prediction.** Reject. I put acceptance at about 15 to 20 percent as it stands (mean overall 4.3 of 10, confidence 4.3). With must-priority fixes below the realistic ceiling is a borderline weak-accept (about 40 to 45 percent), because originality stays modest.

## 5. Weaknesses turned into fixes

Line numbers refer to the current `paper/main.tex` (AAMAS body). They will shift in the IJCAI port.

### 5a. TEXT fixes

| # | Objection | Location | Change |
|---|---|---|---|
| T1 | "mask" overclaim | title l.89; abstract l.303-328; intro l.414-421 and l.423-425 | Retitle to what is shown, e.g. "Self-Play Rates Are Not Endpoint Risk Profiles: Fixed-Rival Audits of Frontier LLMs in an AI Race Game". Replace "mask" by "does not identify". Delete "alter or conceal the treatment-response curve" unless a qualitatively different curve is shown. |
| T2 | Obviousness | after l.397 and l.475-488 (game section) | Add a half-page paragraph stating, per rival, the exact best reply and the sign the rational model predicts for the interaction (once E1 is run). State plainly that a positive AU-AS contrast is predicted; the contribution is how far and in what shape each endpoint departs from it. |
| T3 | Novelty unclear | introduction l.387-398 | Add a 5-line "related work and gap" paragraph: fixed-opponent LLM game studies exist (Akata-style repeated games, Payne, Lekeas); name what they did not do (paired rival-by-risk design with an audited replay) and mention LLM game-theory benchmarks. Verify every citation before adding. |
| T4 | Contract confound | Results l.791-795 and abstract l.316-318 | Say "within-endpoint, within-contract" every time the three shapes are named. Add the GPT-5.5 (reasoning low, 1,024 tokens) and Opus-legacy-cell facts to Table 2's caption. Weaken "three distinct shapes" to "three observed shapes under non-identical contracts" until E4 lands. |
| T5 | Ceiling/scale | Table 2 caption l.764-768 and l.785-795 | State that the interaction is on the rate scale and that GPT-5.5 and Opus are at or near ceiling at risk 0.1. Add the log-odds interaction beside it (E3). |
| T6 | Interval validity | l.793-795; Table 2 | Report the interval method and the number of clusters; replace percentile bootstrap by BCa or permutation. Add a "no multiplicity correction; Holm-adjusted" column or sentence. |
| T7 | "Self-play exceeds fixed-safe in 13 of 15" is near-mechanical | abstract l.318-319; l.807-809 | Drop from the abstract. Keep in Results only with the explanation that the mirror rival is not a fixed-safe rival. |
| T8 | EGT logic | l.842-877; Figure 3 caption | Remove "not selected solely to fit"; say "best of 90 cells over three points; the source study's human-fitted values coincide with it". Replace "closer to weak than strong" by the RMSE numbers with a note that a drift-dominated regime is graded by construction. Consider moving the section to the supplement (section 6). |
| T9 | Diversity framing | l.893-946; abstract l.324-327; Discussion l.988-991 | State the one-policy-versus-population asymmetry in the first paragraph. Replace "second failure of aggregate evaluation" by "a descriptive contrast". Report an effect size and interval, not "13/15 below every draw". Move the mini/nano paragraph (l.936-946) up front so the counter-examples are not read as footnotes, and drop "the screen refuses both of them" as justification. |
| T10 | Screen post-hoc | l.596-610 | Add one sentence: the scripted-rival and diversity results for the four refused endpoints (E8) show whether the exclusion matters. State that the screen is a size proxy. |
| T11 | "Confirmatory" | l.569-582 and every "confirmatory" in the results | Define "confirmatory (protocol frozen at run time, not preregistered)". Do not call the screen's three post-hoc verdicts or the amended GPT-5.5 cells confirmatory. |
| T12 | "Minimum reporting package" untested | l.1018-1025 | Phrase as a recommendation; say it was not validated beyond this game. |
| T13 | Repro | l.1068-1073 | Add what is and is not regenerable (closed endpoints); name the open-weight replication (E6) as the reproducible core. |
| T14 | AI-use policy | supplement end (l.3435-3441) and main paper | Move a 3-line statement to the main paper or ethics slot. Describe exactly what the assistants did. Make sure no LLM generated analysis or text that is not polishing, as IJCAI allows. |
| T15 | Small inconsistencies | abstract l.322-323 versus Figure 3 caption l.831-832; l.928-930 ("20 trajectories ... at every risk level"); l.793 | Align "four graded" versus "all five closer to weak"; write "20 trajectories per risk level"; state seats are 5/5 within each cell. |
| T16 | Group-size paragraph | l.1007-1016 | Supports no claim, so cut it to one sentence (section 6). |

### 5b. EXPERIMENTS

Runner reuse (from the repo tree): hosted API through Kaggle Benchmarks tasks `kaggle/benchmarks/ai_race_scripted_opponent.py` (scripted rival, seat counterbalanced, replay audit), `ai_race_baseline.py` (self-play), `ai_race_frontier_context_mapping.py` (rendering crossing); open-weight GPU through `kaggle/experiments/baseline.py` and `baseline_transformers.py`; analysers `scripts/analyze_scripted_opponent.py`, `analyze_trajectory_diversity_dyad.py`, `analyze_egt_beta_sensitivity.py`; theory in `ai_race/theory/`. Cost unit: one race against a scripted rival is about 9.3 endpoint calls (E[T]=9); one self-play race is about 18.6. The completed scripted campaign is 600 races and 5,580 calls, which is the yardstick.

| ID | Experiment | Answers / closes | How it runs | Cost | Priority |
|---|---|---|---|---|---|
| E1 | **Best-response baseline.** Solve the single-agent MDP against each fixed rival (state: round, progress gap, own Unsafe count; hidden geometric horizon; risk in {0.1, 0.6, 0.9}). Report optimal Unsafe rate per rival and the predicted interaction sign, and place the five endpoints against it. | A1 (tautology), A5; makes T2 possible. | CPU only, `ai_race/theory/` and `ai_race/engine/scoring.py` for payoffs; a script plus a figure. | 0 API, under 1 day of coding | **Must** |
| E2 | **More repetitions where it matters.** Add 20 new CRN repetitions (new seed block) of AS and AU at all three risks on all five endpoints, giving 30 per cell. | C1, C3, B (unresolved shapes). | `ai_race_scripted_opponent.py`, new `base_seed` block, analyse with block as a factor. | 5 endpoints x 2 rivals x 3 risks x 20 = 600 races, about 5.6k calls (about one campaign) | **Must** |
| E3 | **Scale-free and small-sample analysis.** Cluster-robust logit with rival x risk, log-odds interaction, BCa/permutation intervals, Holm across families, ceiling flags. | C2, C3, T5, T6. | Extend `analyze_scripted_opponent.py`; no new data. | 0 API | **Must** |
| E4 | **Decoding-contract harmonisation.** GPT-5.5 under the others' contract (reasoning none, 256 tokens) if the route accepts it, and Sonnet 5 under GPT-5.5's contract (reasoning low, 1,024 tokens); rerun the legacy Opus cell on the current contract. AS and AU at risks 0.1 and 0.9, 20 races. | B2 (contract confound), T4. | Same task with `AI_RACE_*` contract env vars; record the resolved contract in the manifest. | About 2 endpoints x 2 rivals x 2 risks x 20 + 40 = 360 races, about 3.3k calls | **Must** |
| E5 | **Rendering robustness of the scripted-rival interaction.** Cross the narrative-by-code renderings (4 variants) with AS/AU at risks 0.1 and 0.9 on two endpoints (Gemini 3 Flash, Sonnet 5, the cheapest). | B1 (prompt artefact), T4. | Port the context-mapping renderer into the scripted task; replay audit unchanged. | 2 x 4 x 2 x 2 x 10 = 320 races, about 3k calls. Needs one code change | **Must** (if the port is hard, drop to one endpoint) |
| E6 | **Open-weight replication with controlled decoding.** Qwen-class and Llama-class 8 to 14B (and one larger if the GPU allows) at fixed temperature and seed: scripted AS/AU and self-play at three risks, 30 races each. | B3, reproducibility checklist, T13. | `kaggle/experiments/baseline.py` plus a scripted-rival hook (the replay logic exists in `ai_race/audit/context_replay.py`; verify). Kaggle GPU pool, not the Mac. | 3 models x (2 rivals + self) x 3 risks x 30 = about 810 races, a few GPU-hours in total | **Should** (Must if the contract confound in E4 cannot be resolved on the hosted routes) |
| E7 | **Decoding-temperature sweep for trajectory concentration.** One or two open-weight models at temperatures 0.3, 0.7, 1.0, plus a pooled-across-models sample compared with the human null. | B4, C5, T9 (is concentration just effective temperature?). | Same open-weight runner; the pooled version is analysis only with existing data. | 1 to 2 models x 3 temps x 3 risks x 30 self-play = about 540 races, a few GPU-hours | **Should** |
| E8 | **Scripted-rival run on the four refused endpoints.** AS and AU at three risks, 10 races each. | B4, B5 (screen excludes the counter-examples), T10. | `ai_race_scripted_opponent.py`; routes exist in the baseline campaign. | 4 x 2 x 3 x 10 = 240 races, about 2.2k calls | **Should** |
| E9 | **Cross-play and a sampled rival set.** (a) Each endpoint against two other endpoints at three risks, 10 races. (b) Eight rivals sampled from the memory-1 deterministic policy space (not just the four named). | A2, A5, T7; gives "self-play masks" an actual comparator. | (a) Needs a two-endpoint task (the repo has a heterogeneous-dyad runner for open weights; hosted needs new code). (b) Scripted task with a policy table. | (a) 5 x 2 x 3 x 10 = 300 races, about 5.6k calls. (b) 2 endpoints x 8 x 3 x 10 = 480 races, about 4.5k calls | **Should** (a), **Nice** (b) |
| E10 | **Denser risk grid for the Opus 5 switch.** Risks 0.2, 0.3, 0.4, 0.5, 20 self-play races each. | Locates the unmeasured transition (l.657-662). | `ai_race_baseline.py`. | 4 x 20 x 18.6 = about 1.5k calls | **Nice** |
| E11 | A second game (different payoff matrix or horizon). | Generality. | New engine config only if `GameConfig` allows; otherwise out of scope. | Large | Not in 5 weeks; state as limit |

**Suggested order.** E1 and E3 first (zero API, week 1). E2, E4, E5 in weeks 1 to 3. E6, E7, E8 in weeks 2 to 4 on Kaggle GPU. E9a only if quota remains. Freeze the analysis plan for E2 and E4 in the repo before the first push, and report the new blocks as a separate pre-declared replication, never pooled silently with the old block. Keep hosted quota in mind: the five-endpoint scripted campaign already needed a reallocation amendment, so budget about 20k calls across E2, E4, E5, E8 with an account-rotation plan.

**What would move the verdict.** E1 plus E2 plus E4 plus E5 are the minimum that addresses the three most damaging objections (tautology, n, contract/prompt confound). They cost about 12.5k hosted calls and one script. E6 and E7 are what make the reproducibility checklist defensible.

## 6. Page pressure: 8 pages now, 7 for IJCAI

The text is already dense; losing one page means roughly 600 to 700 words plus figure space. Candidates in order of lowest value to the central claim:

1. **Section 5.3 (EGT benchmark, l.842-877) and Figure 3 (full-width).** Move the figure and the sweep description to the supplement; keep two sentences and one number. Saves about 0.5 page. This is also the weakest evidence (A4, C4).
2. **Table 1 (presentation robustness) and its paragraph (l.680-704).** Keep one sentence ("equivalent renderings shift rates by 3 to 11 points; crossed table in S4"). Saves about 0.25 page. If E5 is run, one line of E5 replaces it.
3. **Group-size paragraph (l.1007-1016)** and the sentence in the intro (l.432-433). Reduce to one clause. About 0.15 page.
4. **Human-side reconstruction paragraphs (l.905-908 and l.999-1005).** One sentence each. About 0.15 page.
5. **Section 3.3 screen detail (l.596-610).** Keep thresholds and the one-probe fragility; the 676-combination detail goes to the supplement. About 0.1 page.
6. **Discussion (l.1042-1073):** the "evaluation target", "boundary behaviour" and "next tests" paragraphs overlap with each other and with the intro. Merge into one paragraph. About 0.3 page.
7. **Figure 1 (full-width overview) and Table `benchmark-strategies`.** Reduce Figure 1 to a column-width panel or put Table 1's four strategies in one sentence; they are the cheapest visual elements. About 0.2 to 0.3 page.
8. **Diversity section (5.4):** if kept, drop Figure 4 panels (b) and (c) to the supplement and keep panel (a). About 0.2 page.

Do **not** move: Table 2 (interaction), Figure 2 (scripted-rival), the self-play figure (Figure `risk-response`), the contract-heterogeneity caveats, and the best-response comparison from E1, which should take the space freed by items 1 and 2.

Because IJCAI says "7 pages including all figures and tables", the AAMAS two-column float layout needs a real recompile in the IJCAI style, not just trimming; the full-width `figure*` floats are the likely overflow. Note that AAMAS supplementary is also separate and not read by reviewers by obligation. Nothing a claim depends on (the replay audit, the interval method, the contract table) should live only there.

## 7. Integrity flags (summary)

No direct number contradiction found in the main text. Flags are unsupported or under-supported claims:

1. "Mask" (title, abstract): no result shows a hidden qualitative response.
2. "Second failure of aggregate evaluation" and "evaluation should therefore fix and audit the opponent": the evaluation package is argued, not tested.
3. "Confirmatory" for campaigns with post-hoc screen verdicts, an amended contract on GPT-5.5 and a legacy Opus cell.
4. EGT "not selected solely to fit" next to "closest cell in the entire sweep".
5. Diversity result excludes (by a post-hoc screen) the only endpoints that fall inside the human null.
6. AI-assistance disclosure is in the supplement only and states that exact versions and histories were not retained; IJCAI policy asks for labelled, described LLM use.
7. Minor wording inconsistencies listed in Review C item 8.
