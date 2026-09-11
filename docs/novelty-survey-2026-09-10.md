# Novelty survey, 2026-09-10

This is the record of a five lane novelty survey run against the AI_Race_Experiment
manuscript at HEAD `ed3863c`. It is written for someone who was not in the session.
It exists so that nobody has to redo the search, and so that no contribution sentence
reaches a reviewer in a form the literature already refutes.

**Target of the survey.** `paper/main.tex` (60 KB, AAMAS 2026 submission) and
`paper/supplementary.tex` (125 KB), both as of 2026-09-10 22:23.

**Coverage.** Five lanes, run in parallel:

| Lane | Question it searched |
|---|---|
| llm-repeated-games | LLMs in repeated games, especially against fixed or scripted opponents |
| ai-race-egt | AI development race games, EGT benchmarks for LLM agents, group size |
| behavioural-audit | Comprehension audits and admission gates for LLM behavioural studies |
| diversity-and-representation | Human versus LLM behavioural diversity, label and framing robustness |
| inside-the-repo | The repository against its own manuscript |

**How to read the verdicts.**

* **COLLIDES** means a named prior work did the thing the paper presents as new. The
  claim has to be rewritten, not merely hedged.
* **NARROWER THAN IT LOOKS** means the general form of the claim has a predecessor
  but a specific, stateable version survives. The paper keeps the finding and loses
  the framing.
* **UNCLAIMED** means no lane found a predecessor. Where a lane could not reach a
  source, that is recorded as an open question rather than as clearance.

**Two kinds of novelty, kept apart throughout.** *Mathematical or methodological
novelty* is a new estimator, model, or identification strategy. *Audit or reporting
novelty* is a known instrument applied to a claim nobody has scoped it to before,
plus the discipline of reporting what it refuses. Almost every surviving contribution
in this paper is the second kind. That is a real contribution. Calling it the first
kind is what gets caught in review.

**Verification status tags on every source.** `[full]` a lane read the full text.
`[html]` a lane read an HTML rendering only, PDF extraction failed. `[abstract]` a
lane saw only the abstract or a search snippet; treat every number from it as
unverified. `[second-hand]` a number reported through another paper, never checked
against the original.

---

## C0. Running the Fernandez Domingos and Han AI race with LLM agents

**Kind:** empirical, not methodological. This is a new subject placed in an existing
mechanism.

### What the paper claims

> "We reproduce the two-player game studied by \citet{falling_behind_unsafe} and add
> a compatible multiplayer version based on \citet{to_regulate_or_not}. The game
> engine, not the LLM, resolves every simultaneous round, and we save every prompt,
> response, parsed action, retry, state change, configuration, random seed and final
> outcome." (`main.tex:429`)

### Closest prior work

1. **Fernandez Domingos, E. and Han, T. A. (2026). "Falling Behind Drives Unsafe
   Development in an Idealised AI Race Experiment." arXiv:2607.26034v1.** `[full]`
   The source study, human subjects. oTree plus Prolific, 471 recruited and 340
   completed, 173 games, realised mean 9.56 rounds. Payoffs 1.0 / 0.6 / 2.4 / 2.0,
   progress 1.0 versus 1.5, horizon T = 5 + Geom(0.2) - 1, prize 100 ECU, terminal
   risk q_i = p_r^max times n_i^U / T applied only to winners and tied winners,
   p_r^max in {0.10, 0.60, 0.90}. The preregistered 0.60 versus 0.90 comparison is
   null (d = -0.027, t = -0.206, corrected p = 1). Exploratory results: opponent's
   preceding Unsafe beta = 0.607 (p = 0.002), being ahead after mutual Safe
   beta = -0.296 (p = 0.048), and a first-round Unsafe choice predicting later Unsafe
   play with strength that becomes marginal in the fullest specification. The paper
   states explicitly that it is not an evaluation of LLM agents.
2. **Buscemi, A., Han, T. A. et al. (2025). "Do LLMs trust AI regulation? Emerging
   behaviour of game-theoretic LLM agents." arXiv:2504.08640.** `[html]`
   Three actor types (users trust or not, developers comply or defect, regulators
   enforce or lenient), risk factor epsilon in (-inf, 1], enforcement cost
   c_R = 0.5 low and 5 high, sample points epsilon = -0.1 and 0.2. Two models only,
   GPT-4o and Mistral Large, never mixed inside one game. One-shot and 10-round
   repeated variants, personality-trait conditions. Headline: LLM agents are more
   pessimistic, that is more distrustful and defective, than pure game-theoretic
   agents. It is not a race game: no speed or progress dimension, no stochastic
   horizon, no winner-take-all prize, no accumulated private setback risk. The lane
   could not open the PDF, so the exact method for comparing LLM output with the EGT
   prediction is an open question.
3. **Balabanova, N., Bashir, A., Bova, P., Buscemi, A., Cimpeanu, T., Correia da
   Fonseca, H., Di Stefano, A., Duong, M. H., Fernandez Domingos, E., Fernandes, A.,
   Han, T. A., Krellner, M., Ogbo, N. B., Powers, S. T., Proverbio, D., Santos, F. P.,
   Shamszaman, Z. U., Song, Z. (2025). "Media and responsible AI governance: a
   game-theoretic and LLM analysis." arXiv:2503.09858; Phil. Trans. R. Soc. A, in
   press.** `[html]` Four populations (commentariat, users, developers, regulators),
   Fermi pairwise comparison, N_U = N_C = N_R = 100, beta = 0.1, small-mutation Markov
   chain plus four-population replicator dynamics. The LLM arm is GPT-4o and Mistral
   Large through FAIRGAME, one-shot, personalities set to None. The EGT comparison is
   qualitative and side by side, with no matched numeric contrast.
4. **Gruetzemacher, R., Avin, S. et al. (2025). "Strategic Insights from Simulation
   Gaming of AI Race Dynamics." arXiv:2410.03092; Futures.** `[abstract]` Already in
   `references.bib` as `gruetzemacher2025strategic`. 43 facilitated Intelligence
   Rising games over four years, with human players, qualitative thematic analysis,
   no formal payoff matrix, no LLM players, no statistics.
5. **The group's own preprint, arXiv:2608.01193, "Humans Are More Diverse: Frontier
   LLMs Show Extreme Policies in Idealised AI Development Races," posted 2 Aug 2026.**
   `[abstract]` Not a competitor, it is this paper minus the scripted campaign, with
   the full author list public. See the anonymity note in the collision ledger.

### Verdict: UNCLAIMED, and the cleanest thing in the paper

Nobody outside this group has run the Fernandez Domingos and Han race mechanism with
LLM agents. The Han lab's LLM plus EGT lane (items 2 and 3) sits on the regulation
and trust game, which has no progress increments, no hidden stopping time, and no
winner-conditioned risk realisation. The one prior study that calls itself an AI race
simulation (item 4) uses humans and no formal game.

The claim that must **not** be written is "first LLM study of AI race dynamics."
Items 2 and 3 self-describe that way and both carry Han and Fernandez Domingos as
coauthors. A reviewer will read the overclaim as the authors ignoring their own lab.

### Narrowest true form

> The AI-race stage game has been studied as an evolutionary model
> \citep{to_regulate_or_not,artificial_intelligence_development_races} and once as a
> human behavioural experiment \citep{falling_behind_unsafe}; LLM agents have been
> placed in adjacent AI-governance games by the same community, but in a
> three-population trust-and-regulation game with two models and no progress race, no
> stochastic horizon and no winner-conditioned setback risk. We run the race
> mechanism itself, with simultaneous sealed actions, progress 1.0 against 1.5, a
> hidden five-round-minimum horizon and a winner-only risk draw, with LLM agents, and
> we gate the resulting behaviour behind a comprehension audit.

---

## C1. The audit gate: a frozen comprehension battery placed before any strategic reading

**Kind:** audit and reporting novelty. There is no new statistic here at all. The
battery is accuracy scored, the gate is a threshold.

### What the paper claims

> "Our contribution is the combination: an auditable race mechanism, a validity gate
> applied before any strategic reading, and a comparison against game-theory and
> human benchmarks made only for the routes that pass it." (`main.tex:483`)

> "Admission requires 80\% overall accuracy together with 75\% on state
> reconstruction and 75\% on terminal scoring. The closed-form expected-payoff probes
> are recorded but do not gate admission." (`main.tex:645`)

### Closest prior work

**On the instrument, that is the battery itself:**

1. **Piatti, G., Jin, Z., Kleiman-Weiner, M., Scholkopf, B., Sachan, M., Mihalcea, R.
   (2024). "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society
   of LLM Agents." NeurIPS 2024, arXiv:2404.16698v4 (GovSim).** `[full]` Builds four
   sub-skill tests, each 150 procedurally generated problems, so 600 items per model,
   scored as accuracy against ground truth, across 15 LLMs. The four domains are
   (a) basic understanding of simulation dynamics and simple reasoning,
   (b) individually sustainable choices without group interaction, (c) calculation of
   the sustainability threshold assuming all participants harvest equally, and
   (d) the same threshold via a belief about the other agents' actions. That is close
   to a one-to-one match with this paper's rule recall, state reconstruction,
   terminal scoring and expected payoff. Their bank is 30 times larger per model.
2. **Cacioli, J.-P. (2026). "Screen Before You Interpret: A Portable Validity
   Protocol for Benchmark-Based LLM Confidence Signals." arXiv:2604.17714v1,
   20 Apr 2026.** `[full]` 524 items across 20 frontier LLMs, an explicitly frozen
   index set, thresholds transferred from the MMPI-3 and PAI clinical validity scales.
3. **Li, H. (2026). "Targeted Tests for LLM Reasoning: An Audit-Constrained
   Protocol." arXiv:2605.11599v3, 30 Jun 2026.** `[full]` A versioned frozen probe
   bank (12 tasks in v1 and v2, 18 in v3) plus a checkpointed artifact-and-audit
   contract. Hash-pinned protocol freezing already exists as a named methodological
   object.
4. **Zhu and Zhang (2026). "Clean Engineering, Unstable Measurement: A Preregistered
   Reliability Failure of Black-Box LLM Observers on Shared Endpoints."
   arXiv:2609.04198v1.** `[full]` Design constants sealed in a run-local hash-pinned
   configuration snapshot before execution, with preregistered numeric gates.

**On the gate, that is letting the comprehension result decide what gets interpreted:**

5. **Fernandez Domingos and Han (2026), arXiv:2607.26034.** `[full]` The source study
   already gates on comprehension. Its exclusion criteria include failing the
   comprehension test after five attempts, with per-treatment exclusion counts in
   Table S3. Gating a behavioural report on a comprehension result is the standard
   human-subjects rule, and this paper is transporting it, not inventing it.
6. **Akata, E., Schulz, L., Coda-Forno, J., Oh, S. J., Bethge, M., Schulz, E. (2025).
   "Playing repeated games with large language models." Nature Human Behaviour
   9(7):1380-1390.** `[full]` The comprehension questionnaire, "participants were
   required to complete a comprehension questionnaire. Only upon responding correctly
   to all questions, they could proceed to the main part of the experiment," applies
   to **human participants only**. No comprehension screening, payoff-computation
   check or exclusion criterion is applied to any of GPT-4, Claude 2, Llama-2-70B,
   text-davinci-003 or text-davinci-002. This is the single best sentence available
   in defence of the gate: the flagship LLM repeated-games study gates humans and does
   not gate models.
7. **Huang, C., Chen, C., Lin, C., Lyu, H., Xu, X., Luo, J. (2026). "Probing
   Outcome-Level Resemblance and Mechanism-Level Alignment in LLM Risk Decisions:
   Evidence from the St. Petersburg Game." arXiv:2606.04978v1, 3 Jun 2026.** `[full]`
   Appendix E.4, verbatim: "These models are excluded from the final model pool
   because they fail to correctly compute the expected value of the St. Petersburg
   game, which serves as a basic validity check for inclusion." Excluded: OLMo-2-7B
   and Instruct, DeepSeek LLM 67B Base and Chat, Gemma 2 27B Base and Instruct,
   Gemma 3 1B PT and IT, SmolLM2 1.7B Base and Instruct, Mistral 7B v0.3 Base and
   Instruct, Mixtral 8x22B v0.1 Base and Instruct, Falcon3-7B Base and Instruct,
   Llama 3 70B Base and Instruct. 28 models survive. This is
   comprehension-result-gates-inclusion in an economic decision study, three months
   before this submission. Its weaknesses: a single arithmetic probe, run in
   preliminary trials, no stated threshold, no repetition count, no per-domain table,
   and the excluded models' behaviour is fully suppressed with no diagnostic retention.
8. **Cacioli (2026), again.** `[full]` States the two-stage architecture abstractly:
   "Stage A (validity screening) determines whether the confidence signal is
   interpretable. Stage B (substantive analysis) proceeds only if Stage A is passed."
   Validated on 20 frontier LLMs and 524 items, yielding 4 Invalid, 2 Indeterminate
   and 14 Valid, with a failure-mode demonstration that unscreened invalid models give
   AUROC at chance, 0.499 to 0.509. It also prescribes the retention rule this paper
   uses: "If Stage A classifies as Invalid, substantive metrics may be computed for
   completeness but must be flagged as potentially uninformative."

### Verdict: COLLIDES on the instrument, COLLIDES on the gate, survives only as a combination in a strategic setting

Every separable component has a named 2024 to 2026 predecessor. A frozen hash-pinned
probe bank is Li 2026 and Zhu and Zhang 2026. A four-domain accuracy-scored
comprehension battery for an LLM social dilemma is GovSim, at 30 times the item
count. Fail-the-check-lose-your-row is Huang et al. 2026. Screen-then-interpret with
flagged retention of the refused rows is Cacioli 2026, prescribed rather than merely
practised.

Two corrections are forced by the survey and are not optional:

* **The paper overstates its own practice.** Section `sec:results-audit-behaviour`
  reports all nine routes' gameplay, solid against dashed and filled against hollow.
  The gate does not decide whether a route's behaviour is reported. It decides which
  routes enter the strategic comparison against the human and EGT benchmarks. That is
  exactly Cacioli's middle position and should be described as such.
* **20 probes at 3 repetitions is small.** The supplement already concedes that four
  of six domains move only in steps of 16.7 percentage points. Against GovSim's 600
  items per model this is a reviewer-visible weakness, and it compounds with the
  test-retest instability the repository records (see the repository section, item
  R1).

### Narrowest true form

> Gating on comprehension is standard for human participants and is used in the human
> race experiment we reproduce; a single-probe expected-value check has been used to
> exclude models from an economic-decision study (Huang et al., 2026), and a two-stage
> screen-then-interpret protocol with flagged retention of refused endpoints has been
> formalised for confidence signals (Cacioli, 2026), while the closest comprehension
> battery for an LLM social dilemma is GovSim's four sub-skill tests, 600 procedurally
> generated items per model over 15 models (Piatti et al., 2024). What we add is the
> combination in a strategic-interaction setting: thresholds on three named
> comprehension domains, declared before the campaign and applied per endpoint to a
> hash-pinned probe bank, deciding which endpoints enter the strategic comparison,
> with refused endpoints retained and reported as diagnostic evidence. The flagship
> LLM repeated-games study gates its human participants on comprehension and applies
> no such check to any model \citep{playing_repeated_games_with_llms}.

---

## C2. Measured comprehension ranks with strategic sensitivity, rho = 0.87

**Kind:** empirical association. The statistic is Spearman's rho with an exact
permutation test. Nothing about it is new.

### What the paper claims

> "Across the nine routes, the rank correlation between risk response and
> state-reconstruction accuracy is $\rho=0.87$ (exact permutation $p=0.004$, $n=9$),
> rising to $\rho=0.92$ ($p=0.003$, $n=8$) when Claude Opus 5 is set aside."
> (`main.tex:720`)

### Closest prior work

1. **Piatti et al. (2024), GovSim, section 3.7 and Figure 5.** `[full]` OLS of
   survival time on each sub-skill accuracy across 15 models: simulation dynamics
   R^2 = 0.69, sustainable action R^2 = 0.92, sustainability threshold under the
   equal-harvest assumption R^2 = 0.76, and the belief-based threshold R^2 = 0.82,
   all p < 0.001. The abstract headline is that the ability to form beliefs about
   other agents correlates 0.83 with community survival time. GovSim also states the
   tier pattern plainly: larger models such as GPT-4o show better survival time and
   total gain, while smaller models such as Llama-3-8B often fail to manage any
   resource sustainably.
2. **Ruan, Y., Maddison, C. J., Hashimoto, T. (2024). "Observational Scaling Laws and
   the Predictability of Language Model Performance." NeurIPS 2024,
   arXiv:2405.10938.** `[full]` Over roughly 100 public models the benchmark-model
   matrix is low dimensional: the top 3 principal components explain about 97% of the
   variance, PC-1 represents general capability, and within a family PC-1 correlates
   linearly with log training FLOPs. Any new accuracy-scored battery is, a priori,
   mostly a noisy read of PC-1.

### Verdict: COLLIDES HARD, against a larger and better powered predecessor

GovSim is 15 models with 600 probe items each and four separate correlations at
p < 0.001. This paper is 9 commercial endpoints, 20 probes, one rank correlation.
The paper's related-work section currently cites GovSim nowhere.

There is one real difference and it is worth defending in the text rather than
leaving for a reviewer to notice its absence. GovSim regresses an **outcome level**,
survival time, on comprehension. This paper correlates comprehension with a
**treatment slope**, the Unsafe-rate difference between assigned risk 0.1 and 0.9.
A model can score high on an outcome level with a constant policy. It cannot score
high on a risk response without conditioning on the manipulated variable. That
distinction is the whole of the surviving contribution.

The repository adds two internal problems that make this the most exposed claim in
the paper. The x-axis moves on repeat administration of the identical battery, and
the y-axis is a self-play quantity that the paper's own scripted-opponent appendix
says is not a measurement of how a route treats risk. See repository items R1 and R2.

### Narrowest true form

> GovSim already establishes that sub-skill accuracy predicts outcomes in an LLM
> social dilemma, with $R^2$ from 0.69 to 0.92 over 15 models and 600 probe items each
> (Piatti et al., 2024). We report the analogous association for a treatment slope
> rather than an outcome level: state-reconstruction accuracy ranks with the
> sensitivity of self-play to an experimentally assigned risk parameter,
> $\rho=0.87$, exact permutation $p=0.004$, $n=9$. Like GovSim's, this is descriptive
> across commercial endpoints and identifies no effect; benchmark accuracy is
> additionally known to be dominated by a single general-capability component that
> tracks scale (Ruan et al., 2024), so nothing here separates comprehension from
> provider, scale or training.

---

## C3. The gate's partition is recoverable from the route name

**Kind:** reporting novelty, and only that. The underlying phenomenon is expected.

### What the paper claims

The self-audit is currently carried by the admission table rather than by a sentence:
admitted are Gemini 3 Flash, Claude Opus 5, GPT-5.4, GPT-5.5 and Claude Sonnet 5;
refused are Gemini 3.1 Flash-Lite, GPT-5.4 mini, Gemini 3.5 Flash-Lite and GPT-5.4
nano (`main.tex`, Table `tab:task-audit`). Nine of nine are predicted by the presence
of "mini", "nano" or "lite" in the route string, at zero cost.

### Closest prior work

1. **Ruan et al. (2024).** `[full]` As above. A dominant general-capability component
   linear in log FLOPs makes tier collinearity the prior expectation for any
   accuracy-scored battery.
2. **Ilic, D. and Gignac, G. E. (2024). Intelligence.** `[second-hand]` A dominant
   g-factor across 12 benchmarks, reported as 65.6% of benchmark variance and
   correlating 0.70 with parameter count. The lane read this through Kearns's thesis
   and did not verify the numbers against the original. **Verify before citing.**
3. **Kearns, R. O. (2025). "Quantifying construct validity in large language model
   evaluations." MSc thesis, University of Oxford, arXiv:2602.15532.** `[full]`
   4,395 models from OpenLLM Leaderboard v2 across 19 BBH subtasks; standard latent
   factor models produce a dominant factor functioning essentially as a model-size
   proxy. An MSc thesis: cite as supporting, not as authority.
4. **Burnell, R. et al. (2023). "Revealing the structure of language model
   capabilities." arXiv:2306.10062.** `[abstract]` 29 LLMs, 27 tasks, three factors
   rather than one. The counterweight showing the collapse to one factor is not
   inevitable.
5. **Cacioli (2026), arXiv:2604.17714.** `[full]` The counter-example that removes the
   easy defence. Cacioli's validity gate over 20 frontier LLMs does **not** track
   tier: Invalid includes DeepSeek-R1, Gemini 3.1 Pro, Qwen 80B Think and Gemma 3 1B;
   Indeterminate includes GPT-5.4 nano and Gemma 3 12B; the remaining 14 are Valid. A
   frontier Pro model fails and a nano model does not. So "any validity gate would
   just recover scale" is not available as an argument. Tier collinearity is a
   property of *this* battery, accuracy-scored comprehension items, not of gating in
   general.
6. Genre precedent for the move itself: **"An Embarrassingly Simple Graph Heuristic
   Reveals Shortcut-Solvable Benchmarks for Sequential Recommendation,"
   arXiv:2605.07125.** `[abstract]` A trivial heuristic matching an expensive
   evaluation is an established, named finding in ML, though not in LLM behavioural
   gating.

### Verdict: UNCLAIMED as an act of self-reporting, COLLIDES as a phenomenon

No lane found a prior LLM behavioural or agent study that runs an admission gate, uses
it to decide what to interpret, and then audits its own gate against a zero-cost
naming heuristic. That specific act is unclaimed. The phenomenon it reports is not,
and the paper must never phrase it as a discovery that comprehension tracks model tier.

### Narrowest true form

> The collinearity is expected rather than discovered: benchmark accuracy is dominated
> by a single general-capability component that tracks scale (Ruan et al., 2024, top
> three principal components about 97\% of variance, PC-1 linear in log-FLOPs), and
> GovSim reports the same size pattern in its sub-skill tests. We report instead that
> our own admission decisions add no partition information beyond the endpoint name,
> because a gate must be justified by what else it yields, here the per-domain profile
> and the parser-health record, and because validity gates over frontier models need
> not behave this way: a 2026 confidence-signal screen classifies a frontier Pro model
> Invalid and a nano model only Indeterminate (Cacioli, 2026).

---

## C4. Representation robustness: narrative skin crossed with opaque code assignment

**Kind:** partly audit novelty, partly a replication in a new regime. The one exact
mathematical statement in it, the counterbalancing identity, is elementary algebra and
should be presented as a diagnostic, not a result.

### What the paper claims

> "swapping which code denotes \Safe{} raises \Unsafe{} play by 9.5 percentage points
> on Gemini 3 Flash (95\% interval 5.6 to 13.8) and by 8.3 on Claude Sonnet 5 (5.4 to
> 12.2), over 60 paired blocks each. The narrative skin matters too, but
> route-specifically: retelling the same game as an abstract contest rather than a
> technology race raises \Unsafe{} play by 10.8 points on Gemini 3 Flash (6.0 to 15.6)
> and by 3.4 on Claude Sonnet 5 (0.9 to 7.4)." (`main.tex:821`)

> "A relabelling that a correct model of the game should ignore therefore changes the
> action sequence." (`supplementary.tex`, `app:context-mapping`)

### Closest prior work

1. **Zheng, C., Zhou, H., Meng, F., Zhou, J., Huang, M. (2024). "Large Language Models
   Are Not Robust Multiple Choice Selectors." ICLR 2024 Spotlight,
   arXiv:2309.03882.** `[full]` 20 LLMs, three benchmarks. Defines selection bias as
   preferring specific option IDs and attributes it to token bias, a prior mass on the
   ID tokens themselves, measured by the standard deviation of recalls across option
   IDs. Answer-moving attack on MMLU zero-shot: gpt-3.5-turbo accuracy swings 60.9% to
   74.2%, llama-30B 41.2% to 68.2%, falcon-inst-40B 38.3% to 69.1%. On 1,000
   controlled MMLU samples llama-30B picks A/B/C/D at 34.6/27.3/22.3/15.8% and
   gpt-3.5-turbo at 22.5/25.6/32.3/19.6%. Their PriDe method estimates the ID prior by
   permuting option contents on a small subset and averaging into a global prior, then
   divides it out. That permute-and-average operation is the same operation as this
   paper's counterbalance; the difference is that PriDe removes the prior while this
   paper reports it as a factor.
2. **Won, H.-I., Jang, J., Kim, H. (2026). "When Counterbalancing Hides the Bias:
   Access-Conditioned Position Lock in Forced-Choice LLM Evaluation." arXiv:2607.10202
   (v1 11 Jul 2026, v2 8 Aug 2026).** `[full]` The most important find in this lane.
   18 binary dilemmas, N = 40 responses per model-item, counterbalanced label
   orientations, forced single-letter answers, nine models including Claude Opus,
   Sonnet, Haiku and Fable 5, GPT-5.5, GPT-5.4, Gemini-2.5-flash, DeepSeek V4-Flash
   and Grok-3, that is the same model generation as this paper. Result:
   counterbalancing maps a position lock, returning the same letter regardless of
   content, onto the same near-0.5 signature as genuine neutrality, so the
   concentration index is not identifiable at its low end; the fraction of
   position-locked items tracks the index at r = -0.986. Under access paths that permit
   reasoning, DeepSeek moves 0.06 to 0.63 extremity while lock falls 1.00 to 0.22.
   They stop short of an identity, offering the bound extremity <= 1 - lock_fraction.
   This paper's design does what Won et al. prescribe, reporting the letter-conditioned
   quantity instead of averaging it away.
3. **Robinson, I. and Burden, J. (2025). "Framing the Game: How Context Shapes LLM
   Decision-Making." arXiv:2503.04840.** `[full]` Already cited as
   `robinsonBurdenFraming2025`, but the paper's citation-audit row understates it.
   They manipulate topic (10 categories), world type (real or imaginary) and actor type
   (allies, enemies, neutral), 100 vignettes per combination, GPT-4o, Claude 3.5
   Sonnet and Llama-70B plus 14 more in the appendix. Critically: "Each generated
   scenario is presented to the LLM twice, varying the mapping of A and B to Defect and
   Cooperate to account for any ordinal or token bias," and they report 15% to 21%
   inconsistency rates when labels are swapped. Cooperation ranges from 26% (sporting
   events, Llama) to 75% (21st-century global politics); allies 54% to 72% against
   enemies 40% to 47%; one context shift drops cooperation from 98% to 37%. So the
   2 by 2 crossing of narrative framing with a counterbalanced action-label mapping in
   a game already exists. The difference is what is done with it: they treat mapping as
   nuisance and report an unsigned inconsistency rate, this paper treats it as a factor
   with a signed, race-paired estimate.
4. **Herr, N., Acero, F., Raileanu, R., Perez-Ortiz, M., Li, Z. (2024).
   arXiv:2407.04467.** `[full]` Already cited as `herrStrategicBias2024`. Positional
   bias, payoff bias and behavioural bias defined separately. Stag Hunt, answer-only:
   GPT-3.5 selects A at 67.1% when A is stated first against 0.0% when B is first,
   GPT-4o 25.0% against 73.9%, Llama-3-8B 99.9% against 50.0%, GPT-4-Turbo 25.5%
   against 45.6%. **Calibration point:** those swings are 20 to 67 percentage points
   against this paper's 8 to 10. The honest reading is that they are different
   manipulations, presentation order versus which arbitrary symbol denotes which
   action, and that this paper's effect is measured with the horizon differenced out
   over 60 paired blocks, which a one-shot design cannot do. Make that comparison in
   the text rather than leaving it to a reviewer.
5. **Ancestry to cite once:** Zhao, Wallace, Feng, Klein, Singh, ICML 2021,
   "Calibrate Before Use," arXiv:2102.09690 `[abstract]`, naming majority-label,
   recency and common-token bias and introducing contextual calibration with
   content-free inputs, improving average accuracy by up to 30.0 points absolute;
   Holtzman et al., EMNLP 2021, "Surface Form Competition" `[abstract]`; Reif, Y. and
   Schwartz, R., NAACL 2024, "Beyond Performance: Quantifying and Mitigating Label
   Bias in LLMs," arXiv:2405.02743 `[abstract]`, 279 classification tasks and ten LLMs
   with substantial label bias persisting after debiasing.
6. **Weaker neighbours.** Lore, N. and Heydari, B. (2024), Scientific Reports 14,
   s41598-024-69032-z `[html]`: four games by five framings, 300 initialisations per
   LLM for each of 20 game-by-context combinations, but they use "C" and "D"
   throughout and never relabel, so framing and symbol are not separated. Piche, D. et
   al. (2025), arXiv:2511.19405 `[html]`: relabels Cooperate and Defect to A and B as a
   memorisation control, with no counterbalancing, no run counts and no quantitative
   relabelled-versus-standard comparison.

**Verified algebra.** With u_k = P(Unsafe | mapping k) and q_k = P(emit Q | mapping k),
u_1 = q_1 and u_2 = 1 - q_2, so the contrast u_2 - u_1 = 1 - (q_1 + q_2) = 2(0.5 - q_bar).
Under a balanced counterbalance this is an exact identity, independent of the data. The
figure script asserts |gap| < 1e-9 and its docstring already says the identity explains
nothing on its own. That is the correct reading and it must stay.

### Verdict: COLLIDES on the phenomenon and on the design element; survives as a replication in a new regime

The letter anchor is a five-year-old named literature under selection bias, token bias
and label bias, with a canonical paper whose debiasing method performs the same
permute-and-average operation as this counterbalance. A 2 by 2 crossing of narrative
framing with a counterbalanced action-label mapping in a game already exists. And a
2026 paper on the same model generation is specifically about what counterbalancing
hides at the letter level.

Nothing here kills the contribution. The current wording, "a relabelling that a
correct model of the game should ignore therefore changes the action sequence," reads
as a discovery and will be marked down as one.

The strongest defensive sentence available and currently unused: Gemini 3 Flash emits
Q on 77% of turns when Q means Unsafe and on 11% when Q means Safe. That is the
opposite of a position lock in the Won et al. sense, and it is exactly the diagnostic
their paper asks for.

### Narrowest true form

> Letter and option-ID anchoring is established for single-turn forced-choice and
> multiple-choice settings (Zhao et al., 2021; Zheng et al., 2024, with 13 to 31 point
> accuracy swings under answer-moving attacks; Reif and Schwartz, 2024), and one prior
> game study counterbalances the action-label mapping across framing conditions and
> reports a 15\% to 21\% inconsistency rate \citep{robinsonBurdenFraming2025}. We
> measure the signed effect of the code assignment in a repeated game with accumulating
> state, differenced within common-random-number repetition blocks so the sampled
> horizon is held fixed, on two routes that had already passed a comprehension gate,
> and we report both the counterbalancing identity that makes the code contrast exactly
> twice the marginal letter preference and the per-mapping letter shares, which show the
> anchoring is state-conditioned rather than a fixed letter habit (Won et al., 2026).

---

## C5. Human participants are more diverse than every admitted checkpoint

**Kind:** claim-scoped use of a known statistic. Hill numbers and rarefaction are
ecology, already imported into LLM evaluation twice. The novelty is the object measured
and the null, not the estimator.

### What the paper claims

> "Humans produce close to the maximum possible variety, an effective 18.9, 19.7 and
> 19.3 distinct sequences out of 20 at risk 0.1, 0.6 and 0.9 ... Claude Opus 5 produces
> a single sequence, repeated by all 20 of its trajectories, at every risk level."
> (`main.tex:948`)

> "Every checkpoint that passes the gate is strictly less diverse than the human sample
> at every risk level, with no exception." (`main.tex:966`)

### Closest prior work

1. **Wright, D., Masud, S., Moore, J., Yadav, S., Antoniak, M., Ebert Christensen, P.,
   Park, C. Y., Augenstein, I. (2026). "What and Whose Knowledge? Measuring Epistemic
   Diversity in Large Language Models." EMNLP 2026, arXiv:2510.04226.** `[full]`
   27 LLMs by 155 topics by 200 prompt variations, 1.7M responses decomposed into
   69.5M atomic claims. The diversity index is **Hill diversity, citing Hill (1973)
   and Jost (2006)**, with Hill-Shannon D_S = exp(-sum p_i ln p_i). They use
   **rarefaction and coverage standardisation**, estimating coverage with Chao and Jost
   (2012) and downsampling every model to the minimum coverage across comparisons,
   which is the same sample-size-fairness move as rarefying to 20. Headline: nearly all
   models are less epistemically diverse than a plain web-search baseline; model size
   has a significant negative effect, beta = -228.47, p far below 1e-3, and RAG a
   positive one, beta = +739.19.
2. **Hodel, D. and West, J. D. (2025). "Epistemic diversity across language models
   mitigates knowledge collapse." arXiv:2512.15011, 30 pp, 17 Dec 2025.** `[full]`
   Uses Hill-Shannon Diversity as "the effective number of equally frequent, diverse AI
   models," segments fixed training data across an increasing number of LMs and runs ten
   self-training iterations, finding that diversity improves long-run performance while
   monoculture accelerates collapse. This is the second independent prior application of
   Hill numbers to LLMs, and it uses q = 1 as an effective number of policies, which is
   closer in spirit to this paper's reading than Wright et al.'s claim-level use.
3. **Wenger, E. and Kenett, Y. (2025). "We're Different, We're the Same: Creative
   Homogeneity Across LLMs." arXiv:2501.19361.** `[full]` 102 human participants after
   screening 114, and 22 LLMs with 7 families used for the statistics, on three
   standardised divergent-thinking tests: Alternative Uses (5 objects), Forward Flow
   (5 seed words), Divergent Association (10 words). Statistic is mean pairwise cosine
   distance between sentence embeddings, LLM-versus-LLM against human-versus-human.
   AUT 0.459 against 0.738, effect size 2.2; FF 0.534 against 0.835, d = 2.0;
   DAT 0.665 against 0.819, d = 1.4. Collides on the framing, a population of LLMs is
   far less diverse than a matched human population on a standardised task. Does not
   collide on domain, statistic, rarefaction or null.
4. **del Rio-Chanona, R. M., Pangallo, M., Hommes, C. (2025). "Can Generative AI Agents
   Behave Like Humans? Evidence from Laboratory Market Experiments."
   arXiv:2505.07457.** `[html]` Already cited as `delRioChanonaMarkets2025`. What the
   paper's citation-audit row does not say: their heterogeneity comparison is
   **qualitative and visual**. They fit per-agent behavioural parameters by OLS and plot
   the 3D parameter cloud, writing that "Human subjects exhibit greater heterogeneity
   than LLM agents, with their estimated parameters more widely dispersed across the
   parameter space." No variance statistic, no entropy, no dispersion coefficient is
   computed. Six human subjects per experiment against six LLM agents. This is the
   paper's strongest supporting predecessor: the contribution here is precisely the
   quantification they left undone, and saying so is a cheap framing win.
5. **Anwar, A. and Georgalos, K. (2026). "Playing Against the Machine: Cooperation,
   Communication, and Strategy Heterogeneity in Repeated Prisoner's Dilemma."
   arXiv:2603.15852.** `[html]` 126 human participants play repeated PD, delta = 0.80,
   7 supergames, against GPT-5.2, benchmarked against 108 human-human pairs from
   Dvorak and Fehrler (2024). Strategy heterogeneity is estimated with the Strategy
   Frequency Estimation Method (Dal Bo and Frechette, 2011) over
   {ALLD, ALLC, GRIM, TFT, WSLS, T2}, a finite-mixture MLE returning a distribution
   over strategies and a tremble. Pre-Play human-AI 55.6% GRIM and 13.7% ALLC against
   human-human 53.3% GRIM and 37.9% TFT; Repeated human-AI 40.1% ALLC against
   human-human 70.6% ALLC; Wald tests p = 0.045 and p = 0.068. **A reviewer will ask
   why SFEM was not used.** The answer to put in the paper: SFEM projects onto a
   hand-specified strategy library and cannot represent a population that collapses to
   one literal sequence, and Hill q = 0 equal to 1.0 with Hamming 0.000 is not
   expressible as a mixture vector.
6. **Secondary, one line each.** Doshi and Hauser, Science Advances 10(28), eadn5290
   (2024) `[abstract]`, AI-assisted stories more similar to each other than human-only
   stories. Anderson, Shah and Kreminski, C&C '24, 413-425 `[abstract]`, 36
   participants, ChatGPT users produce less semantically distinct ideas. Kleinberg and
   Raghavan, arXiv:2101.05853, PNAS `[abstract]`, origin of the monoculture framing.
   Bommasani et al., NeurIPS 2022, arXiv:2211.13972 `[abstract]`, outcome
   homogenization. Zhang, arXiv:2609.04373 `[html]`, pairwise correlation of eight LLM
   traders' actions rising monotonically with capability, t = 3.94, p = 0.001,
   R^2 = 0.475 against ELO, with no human baseline. Pasarkar and Dieng, AISTATS 2024,
   arXiv:2310.12952 `[abstract]`, "Cousins of the Vendi Score" generalises the Vendi
   Score as a similarity-aware Hill number of order q, so the ML community already owns
   the Hill-number vocabulary.

### Verdict: COLLIDES on the statistic and on the framing; UNCLAIMED on the object and the null

Hill numbers with rarefaction applied to LLM output are not new as of October 2025, and
two independent groups have done it. "A population of LLMs is less diverse than a
matched human population" is published for creativity tasks with a large effect size.
What no lane found is the diversity of the **literal paired action sequence of a
repeated game**, rarefied to a common size with a cluster bootstrap over races, against
a **matched human resampling null drawn from the same mechanism**, with the reading
conditioned on a comprehension gate.

`app:diversity` currently introduces Hill q = 0 and q = 1 as if from ecology first
principles. It must cite Hill (1973), Jost (2006) and Chao et al. (2014) for
rarefaction and extrapolation with Hill numbers, and it should acknowledge Wright et al.
and Hodel and West as prior LLM applications. Never write "nobody has applied Hill
numbers to LLM behaviour."

### Narrowest true form

> The closest work applies Hill diversity with rarefaction and coverage standardisation
> to the claims in LLM text against a web-search baseline (Wright et al., 2026; Hodel
> and West, 2025), compares LLM and human population diversity on creativity tests with
> embedding cosine distance (Wenger and Kenett, 2025), or estimates a mixture over a
> hand-specified strategy library in a repeated game (Anwar and Georgalos, 2026). We
> measure the diversity of the literal paired action sequence of a repeated game,
> rarefied to a common size with a cluster bootstrap over races, against a matched human
> resampling null from the same mechanism, and we condition the reading on a
> comprehension gate. Only \citet{delRioChanonaMarkets2025} compare human and LLM
> strategic heterogeneity in a repeated economic setting, and their comparison is a
> visual inspection of a fitted parameter cloud with no diversity statistic.

---

## C6. The evolutionary benchmark: comparison, selection-strength sweep, and risk-axis inversion

**Kind:** three separate claims of three different kinds. The comparison is a
replication with LLMs substituted for humans. The sweep is a sensitivity analysis. The
inversion is a standard operation whose *diagnosis* here is an identifiability result.

### What the paper claims

> "a claim that LLM agents contradict the evolutionary benchmark cannot rest on one
> setting of the selection strength, because the sign of the high-risk gap depends on
> it." (`main.tex:865`)

> "Panel b inverts the model, asking which risk it would need to emit each observed
> rate; the configured range of $0.80$ returns as a band of $0.12$."
> (`main.tex:880`)

### Closest prior work, comparison half

1. **Fernandez Domingos and Han (2026), reduced evolutionary model.** `[full]` Four
   strategies AS, AU, CS, CAS; 10^4 Monte Carlo races per ordered matchup for
   conditional pairs, closed form otherwise; finite-population pairwise comparison with
   Fermi imitation; stationary distribution over strategies. **They already compare a
   real behavioural population against that stationary distribution** in their Figure
   3B. The theory-versus-behaviour comparison in `sec:results-theory` is therefore a
   re-run of the source study's own comparison with LLMs substituted for humans.
2. **arXiv:2602.16662, "Evaluating Collective Behaviour of Hundreds of LLM Agents"
   (2026).** `[html]` Three social dilemmas (Public Goods, Collective Risk Dilemma,
   Common Pool Resource); Claude Haiku 4.5, Gemini 3.1 Flash Lite, GPT-5.4 Mini;
   512 strategies per model per attitude per game, 9,216 strategies total; Fermi
   pairwise comparison at inverse temperature beta = 1, 2,000 generations with analysis
   on the final 100, 100 independent runs per game and group size, 200 self-play
   repetitions per composition. The direction is the reverse of this paper: they put
   LLM-authored strategies into an evolutionary dynamic rather than reading LLM play
   against a stationary distribution.
3. **arXiv:2501.16173, "Will Systems of LLM Agents Cooperate: An Investigation into a
   Social Dilemma" (2025).** `[html]` IPD payoffs 3/5/1; ChatGPT-4o and Claude 3.5
   Sonnet asked to emit complete strategies in natural language, hand-transcribed to
   Python; Moran process at population n = 12, all-play-all 1,000-round matches, run to
   fixation. ChatGPT-4o under the Default prompt converged to the aggressive equilibrium
   66% of the time from a 4:1:1 start. Same reverse direction.
4. **Wang, Y., Chen, X., Wang, Z. (2017). "Testability of evolutionary game dynamics
   models based on experimental economics data." Physica A 486:455-464.** `[abstract]`
   Pre-LLM precedent for the general move: RPS lab data, angular momentum and speed as
   measurement variables, goodness of fit between experimental and theoretical dynamic
   patterns. Comparing an observed population to an EGT prediction and scoring the match
   is a twenty-year-old programme in experimental economics.

### Closest prior work, inversion half

5. **Fernandez Domingos and Han (2026), again, and this is the named predecessor.**
   `[full]` Reference point beta = 2 with mu = beta/Z = 0.02; the best-fit point is
   "obtained when fixing a higher mutation rate, mu = 0.05, and weak selection,
   beta = 0.01," motivated as capturing higher behavioural noise in the experiment. So
   the source study **already inverts this exact model against observed behaviour**, on
   the (beta, mu) axis, against human play. It does **not** invert on the risk axis and
   never asks which p_r^max would produce an observed Unsafe rate.
6. **Traulsen, A., Semmann, D., Sommerfeld, R. D., Krambeck, H.-J., Milinski, M. (2010).
   "Human strategy updating in evolutionary games." PNAS 107(7):2962-2966.**
   `[abstract]` The canonical inversion of an EGT update rule against observed
   behaviour. Human subjects on a virtual spatial lattice, from which the strategy update
   rule itself is measured; estimated spontaneous strategy-change probabilities
   mu_C = 0.28 +/- 0.07 and mu_D = 0.25 +/- 0.01, the headline being that exploration is
   far higher than models assume. **The numbers came from a search snippet, not the PDF.
   Read the PDF before citing them.** This is the sixteen-year-old precedent for this
   paper's conclusion that behaviour sits closer to a near-neutral, high-exploration
   population.
7. **Pechon-Elkins, M. and Chun, J. (2026). "Auditing Game-Theoretic Measures of
   Strategic Reasoning in LLMs." arXiv:2603.10029 (v1 Feb 2026, v2 Aug 2026).**
   `[abstract]` The most uncomfortable hit in this lane, because it pairs the same two
   moves this paper makes: an audit of a game-theoretic LLM benchmark plus a
   fitted-parameter inversion against LLM choice data. 1,855 interactions, seven models.
   Found a 1,024-token limit producing empty responses that the parser replaced with
   default actions, invalidating prior Kimi K2 results, which is structurally the same
   failure this project's parse-failure rule is built to catch. Corrected a signalling
   game bluff target from 0.340 to a conditional 2/3. Bluffing propensity 0.109 to 0.732
   across models. Model-specific fitted propensities predicted choices better than
   equilibrium predictions, and they treat fitted values as "payoff- and
   model-dependent summaries instead of structural rationality parameters," which is
   this paper's own caveat that panel b is an inversion and not an estimate of anybody's
   perceived risk, written down first. **PDF extraction failed. Read it before citing,
   and read it before submitting.**
8. **The inversion is long known under other names.** Solving a structural model for the
   parameter that reproduces an observed moment is calibration or indirect inference
   (Gourieroux, Monfort and Renault 1993, J. Applied Econometrics; Dridi, Guay and
   Renault 2007, J. Econometrics, which explicitly interprets calibration as estimation
   by simulation) `[abstract]`; in finance it is implied volatility; in behavioural game
   theory it is estimating the QRE precision lambda from observed choice frequencies
   (McKelvey and Palfrey 1995) `[abstract]`, with the identification hazards catalogued
   in "Quantal response equilibrium as a structural model for estimation: the missing
   manual," Games and Economic Behavior, 2025 `[abstract]`, which notes that lambda is
   interpretable only relative to payoff units. The Fermi beta in this model is
   mathematically the same logit precision.

### Verdict

* **Comparison: COLLIDES.** It is what the source study does with humans, and the LLM
  plus EGT literature was doing it in both directions by early 2026. Drop any framing
  that the comparison itself is new.
* **Beta sweep: UNCLAIMED.** No lane found prior work running a selection-strength
  sensitivity check on an LLM-versus-EGT comparison. 90 stationary-distribution cells
  over beta in [0.001, 10] crossed with three mutation rules, with the sign of the
  high-risk gap reversing inside that range, is the piece to keep.
* **Inversion: NARROWER THAN IT LOOKS.** The operation is thirty years standard, and the
  source study already inverts this very model on a different axis. What survives is not
  the inversion but its diagnosis: the risk axis is compressed to the point of
  non-identification.

There is an internal tension to fix before review. `sec:results-theory` already concedes
that beta = 0.01 and mu = 0.05 "are the values the source study reports as its own best
fit." The paper is therefore standing on the source's inversion while presenting an
inversion as its own move.

### Narrowest true forms

> **Comparison and sweep.** \citet{falling_behind_unsafe} compare human play with their
> reduced model's stationary distribution, and recent work embeds LLM-authored
> strategies inside Fermi or Moran dynamics. What has not been done is to ask whether
> such a comparison is stable in the parameter that governs it: we sweep 90
> stationary-distribution cells over $\beta \in [0.001, 10]$ crossed with three mutation
> rules and show that the sign of the high-risk gap between benchmark and behaviour
> reverses within that range, so no single-$\beta$ verdict that LLM agents contradict
> the evolutionary benchmark is well posed.

> **Inversion.** Solving a behavioural model for the parameter that reproduces an
> observed rate is standard practice, as calibration and indirect inference in
> econometrics, implied volatility in finance, and precision estimation in quantal
> response equilibrium, and the source study already inverts this very model against
> human play on the $(\beta, \mu)$ axis, reporting $\beta=0.01$, $\mu=0.05$ as its best
> fit. Our contribution is not the inversion but its diagnosis: inverted on the risk
> axis at the source's own weak-selection point, three configured risk levels spanning
> $0.80$ of the axis return inside a band of width $0.12$, so the benchmark's risk
> parameter is not identified from an aggregate \Unsafe{} rate. That bounds what any
> Unsafe-rate-versus-risk comparison against this model can establish, ours included.

---

## C7. The matched group-size grid, two to five companies by three risk levels

**Kind:** identification-design novelty, small in scope. The crossing is the new part;
the group-size question is not.

### What the paper claims

> "\Unsafe{} play rises monotonically with the number of competing companies at every
> private-risk condition that leaves room for it to rise, and all six of those contrasts
> exclude zero." (`supplementary.tex:2175`)

Contrasts against the two-player arm: +11.1, +28.2, +31.0 percentage points at
p_r^max = 0.6, and +11.0, +28.1, +40.0 at 0.9. Every cell at p_r^max = 0.1 is at 100%.

### Closest prior work

1. **arXiv:2510.22422, "Group size effects and collective misalignment in LLM
   multi-agent systems" (2026); PNAS.** `[html]` The biggest name in this space. A
   naming game with two word options from biased pairs, +100 for coordination and -50
   for miscoordination, memory of recent interactions. Group sizes N = 2, 6, 12, 24, 48,
   96, 192, 384, 768, 1536, 3072 and 10,000. Four models: Qwen QwQ-32B, Phi-4, GPT-4o,
   Llama 3.1 70B Instruct. 1,000 runs per condition, up to 1,000 population rounds,
   temperature 0.5. Collective preference for a convention increases with population
   size until convergence becomes deterministic, with the threshold ranging from N = 2 to
   about 10^4 depending on condition. The protocol is uniform across sizes and group size
   is **not crossed with any risk or incentive parameter**.
2. **arXiv:2602.16662 (2026), again.** `[html]` Group sizes n in {4, 16, 64, 256} for
   self-play robustness and {4, 64} for cultural evolution, across three dilemmas on
   three frontier models. Relevant finding: Gemini's Common Pool Resource strategies fail
   catastrophically in large groups while holding up at small sizes, a group-size by
   model interaction. Welfare normalised to [0, 1], no risk-parameter crossing.
3. **arXiv:2604.04782, "Cheap Talk, Empty Promise: Frontier LLMs easily break public
   promises for self-interest" (2026).** `[abstract]` **Author list not verified, detail
   from a search snippet, needs a direct read.** Extends to groups of 3 to 10 agents
   across Volunteer's Dilemma, El Farol Bar and Diner's Dilemma, and reports **no
   systematic trend across group sizes**, with Diner's Dilemma most stable at 4.2 points
   of variation across the full range. It contradicts the direction found here, in a
   different game. Cite it: "our monotone group-size effect is not universal across LLM
   social dilemmas" is stronger and more honest than silence.
4. **"Analysing public goods games using reinforcement learning: effect of increasing
   group size on cooperation." Royal Society Open Science 11:241195 (2024).**
   `[abstract]` Pre-LLM. Deep Q-learning agents in a PGG at varying group size;
   cooperation becomes more likely as group size increases, because Q-values for
   cooperative and non-cooperative actions converge in large groups.
5. **Han, T. A. et al., "To regulate or not: a social dynamics analysis of an idealised
   AI race," JAIR 69:881-921 (2020), and "Artificial intelligence development races in
   heterogeneous settings," Sci. Rep. 12:1723 (2022).** `[full]` Both already in
   `references.bib`. These are N-player EGT with no agents at all: the group-count payoff
   rule D = k + s(N - k) used in Figure 1c is theirs, and the group-size dependence of
   race outcomes is already analysed analytically there. This grid is the empirical arm
   of an existing theoretical claim.

### Verdict: NARROWER THAN IT LOOKS, but the surviving part is a real design contribution

"Group size changes LLM multi-agent behaviour" is settled and published in PNAS at group
sizes reaching 10^4, against a maximum of 5 here. "Group size in an N-player AI race
changes safety outcomes" is Han et al.'s own prior theoretical result. What no lane found
is any LLM group-size study that **crosses group size with a risk treatment**: all three
hold the incentive structure fixed while varying N. The matched design here, one route,
same engine at N = 2 with the two-player matrix verified to return exactly, shared horizon
draw stream inside a repetition, is a stronger identification design than any of the three.

The honesty constraint the supplement already states must survive into any summary: the
grid has 12 cells and **7 of them are off the response boundary**. Four cells at
p_r^max = 0.1 carry no information at all, and the N = 5 cell at 0.6 is truncated. Ten
races per cell, one route.

### Narrowest true form

> Group size is known to change LLM multi-agent outcomes, over $N=2$ to $10^4$ in a
> convention game and $n=4$ to $256$ across three social dilemmas, though with no
> systematic trend in three binary-action group games at $N=3$ to $10$, and the
> group-size dependence of the $N$-player AI race is an existing analytical result
> \citep{to_regulate_or_not}. In every one of those LLM studies the incentive structure
> is held fixed while $N$ varies. We report a group-size sweep in which group size is
> crossed with an assigned risk treatment, on a single admitted route, with the
> two-player arm run through the same $N$-player engine and the horizon stopping-draw
> stream shared within a repetition so the contrast is paired. \Unsafe{} play rises
> monotonically with the number of companies in the seven of twelve cells that are off
> the response boundary, by $+11.0$ to $+40.0$ points against the two-player arm; the
> remaining five cells are at $100\%$ and carry no group-size information.

---

## C8. The scripted-opponent campaign

This is three claims and they have very different exposure. Handled separately.

**Kind:** C8a is a standard design, C8b is an empirical contrast, C8c is a clean
identification, C8d is a methodological claim with a named predecessor.

### C8a. The scripted-opponent design itself

#### What the paper claims

> "The route admitted with the highest score ... played the same game against each of
> those four strategies at each of the three private-risk levels: twelve cells, ten
> repetitions each, 120 races and 1,116 route decisions with zero parse failures."
> (`supplementary.tex`, `app:scripted-opponent`)

#### Closest prior work

1. **Akata et al. (2025), Nature Human Behaviour 9(7):1380-1390.** `[full]` Phase 2 is
   the fixed-opponent arm, in exactly two games. Prisoner's Dilemma: three fixed
   strategies, Always Cooperate, Always Defect, and Defect Once (defects in round 1,
   cooperates thereafter). There is no tit-for-tat in their PD. Battle of the Sexes: two
   singleton strategies plus one alternating turn-taking strategy, started on the
   opponent's preferred option "for courtesy." Findings: "GPT-4 never cooperates again
   when playing with an agent that defects once but then cooperates on every round
   thereafter," and when told explicitly that the opponent would defect once and
   otherwise cooperate, "this resulted in GPT-4 choosing to defect throughout all
   rounds"; in BoS, "GPT-4 performs poorly when playing with an alternating pattern ...
   an instance of a behavioural flaw." PD payoffs 8/0/10/5, BoS 10,7 / 0,0 / 0,0 / 7,10,
   known finite horizon of 10 rounds, no stochastic stopping, no terminal prize, no risk.
   Phase 1 is 1,224 games over 144 2x2 games in six families, every engine against every
   engine including itself, so self-play sits on the diagonal of an 8x8 heatmap and
   cross-play off it, never contrasted as an estimand.
2. **Payne, K. and Alloui-Cros, B. (2025). "Strategic Intelligence in Large Language
   Models: Evidence from evolutionary Game Theory." arXiv:2507.02618, 3 Jul 2025.**
   `[full]` **The closest published design in the whole literature and it is not in the
   bibliography.** "The first ever series of evolutionary IPD tournaments, pitting
   canonical strategies (e.g., Tit-for-Tat, Grim Trigger) against agents from OpenAI,
   Google, and Anthropic." Ten hand-coded canonical strategies including both Tit for Tat
   **and Suspicious Tit for Tat**, plus Grim, Generous TFT, WSLS, Prober, Gradual,
   Random. Stochastic horizon with per-round termination probability p in
   {0.10, 0.25, 0.75}, hard cap 30 rounds, expected match lengths 9 +/- 3, 3 +/- 1 and
   1.3 +/- 0.1. 24-agent population, 276 round-robin matches per phase, five phases per
   tournament, evolutionary reproduction between phases, seven tournaments including a
   three-way LLM Showdown. Payoffs R = 3, S = 0, T = 5, P = 1. Nearly 32,000 prose
   rationales coded.
3. **Fontana, N., Pierri, F., Aiello, L. M. (2025). "Nicer Than Humans: How do Large
   Language Models Behave in the Prisoner's Dilemma?" Proc. ICWSM 19:522-535
   (arXiv:2406.13605).** `[full]` The closest pure fixed-opponent design.
   Llama2-70B-chat, Llama3-70B-Instruct, GPT-3.5, temperature 0.7, N = 100 rounds per
   game, k = 100 repetitions per cell, 95% CIs. The opponent is a scripted Unfair Random
   URND_alpha, cooperating with probability alpha swept across [0, 1], where alpha = 1 is
   Always Cooperate and alpha = 0 is Always Defect. That is structurally the same
   instrument as a four-point rival axis, at much higher resolution. Llama3 keeps
   p_coop below 0.3 for every alpha < 1 and jumps to about 100% only at alpha = 1;
   Llama2 is sigmoidal with inflection between 0.6 and 0.7; GPT-3.5 stays between 0.2 and
   0.4 for alpha < 0.5 and never reaches full cooperation. **No self-play anywhere.**
4. **Ye, Cao, Chen, Ferrara (2026). "Stop Drawing Scientific Claims from LLM Social
   Simulations Without Robustness Audits." arXiv:2605.18890, 17 May 2026.** `[full]`
   10-round PD, payoffs 3/0/5/1, 30 simulations per condition. Single-agent mode: LLM
   against four fixed policies, TitForTat, Random, AlwaysCooperate, AlwaysDefect.
   Two-agent mode: two identically configured LLM agents. Models gpt-5.2,
   claude-haiku-4-5, gemini-2.5-flash, deepseek-v3. They cite Akata, Fontana and the Han
   group as "common practice" for running both modes. Their estimand is prompt
   perturbation: persona format alone moves cooperation by 76 points in gpt-5.2, 77 in
   claude-haiku, 36 in gemini-2.5-flash and 1 in deepseek-v3.

#### Verdict: COLLIDES on design, does not collide on instance

Playing an LLM against Always-Cooperate, Always-Defect and a mirroring conditional
strategy is standard practice with at least four independent precedents, one of them
already in the bibliography and one of them explicitly calling it common practice. Worse
for the framing: **CS and CAS are Tit-for-Tat and Suspicious Tit-for-Tat under different
names**, and Payne and Alloui-Cros ran exactly that pair as live opponents against
frontier models under a stochastic horizon. Nothing resembling "we introduce a
scripted-opponent design" can be written.

Three properties are genuinely uncommon and each is defensible on its own:
non-disclosure with a byte-identical prompt (Akata's most informative fixed-opponent
probe *told* GPT-4 the opponent's strategy); replay verification of the scripted rival,
which no lane found in any of the eight papers read; and seat counterbalancing against a
measured seat effect (Akata ran both positions for coverage, not against a measured
effect).

#### Narrowest true form

> Playing an LLM against fixed strategies is standard practice in the iterated
> Prisoner's Dilemma \citep{playing_repeated_games_with_llms} and elsewhere (Fontana et
> al., 2025; Payne and Alloui-Cros, 2025; Ye et al., 2026). We run it in the
> AI-development-race game of \citet{falling_behind_unsafe}, where the rival set is not
> borrowed from Axelrod but is the four-strategy reduced model that the source study
> itself fitted to its human data, where the rival is not disclosed as scripted, the
> route's seat is counterbalanced five and five, and every rival move is replayed from
> the route's own recorded moves before the cell is admitted.

### C8b. The rival's stance moves the route about seventy points

#### What the paper claims

> "The rival's stance moves this route by about seventy points, and the stated risk
> barely changes that number." (`supplementary.tex:2291`) Contrasts +73.5, +67.4, +71.7.

#### Closest prior work

1. **Fontana et al. (2025).** `[full]` The alpha sweep **is** this contrast, at higher
   resolution. Llama3 below 0.3 for all alpha < 1 and about 100% at alpha = 1 is a
   rival-stance effect of roughly 70 points, published a year earlier. Their game has no
   risk parameter, so they cannot compare the rival axis against a treatment axis.
2. **Akata et al. (2025).** `[full]` GPT-4's defection is near-total against a defector
   and it will not return to cooperation after Defect-Once, so the dependence exists
   there too, but it is reported as a pairwise heatmap and the AC-versus-AD difference is
   never differenced or given an interval.
3. **Payne and Alloui-Cros (2025), Table 13.** `[full]` Gemini cooperation falls 0.852 to
   0.022 as termination probability rises from 10% to 75%; OpenAI 0.876 to 0.957 over the
   same change. Their manipulated axis is the horizon and it moves behaviour more than 80
   points for Gemini. They pool across opponents, so a per-opponent contrast is not
   recoverable.
4. **Ye et al. (2026).** `[full]` The 76-point persona-format effect is the number a
   hostile reviewer will hold against +73.5.

#### Verdict: COLLIDES in kind, survives in the comparison

"The rival's behaviour matters a lot" is established. What no lane found is a **matched
contrast of the rival axis against the game's own experimental treatment axis**, rival
stance against the private-risk cap, inside one design where both are manipulated and
both are differenced within a repetition against a shared horizon draw.

Two honesty notes. Ye et al. force the paper to say why +73.5 is a finding and 76 points
of persona formatting is a fragility; the answer is that a rival's stance is the game's
own state variable and a bullet-list persona is not, but it must be said, because the
numbers are the same size. And ten repetition blocks is thin: Fontana ran k = 100, Payne
276 matches per phase.

#### Narrowest true form

> The dependence of an LLM's play on a fixed rival's stance is documented in the iterated
> Prisoner's Dilemma, most finely by Fontana et al. (2025), whose $\alpha$ sweep moves
> Llama3 from under 30\% cooperation to near 100\%. What has not been reported is that
> dependence measured against the game's own manipulated treatment in the same design:
> here the rival's stance is worth $+73.5$, $+67.4$ and $+71.7$ points while the
> private-risk cap moves the same route far less, with the two differenced inside a
> repetition so the horizon stopping draw drops out.

### C8c. The rival's opening move alone

#### What the paper claims

> "The rival's opening move alone is worth $+25.1$, $+15.8$ and $+11.5$ points ...
> One move in round one is enough to fix the rest of the race."
> (`supplementary.tex:2306`)

#### Closest prior work

1. **Payne and Alloui-Cros (2025).** `[full]` Has both Tit-for-Tat and Suspicious
   Tit-for-Tat in the same population, so the contrast that isolates the opening move is
   **latent in their data and they never take it**. They report pooled strategic
   fingerprints, P(C|CC), P(C|CD), P(C|DC), P(C|DD), across all opponents, so TFT and
   STFT are absorbed into the field.
2. **Akata et al. (2025).** `[full]` The Defect-Once agent is an opening-move probe and
   produces their most-cited finding. But Defect-Once is unconditional after round 1, so
   it isolates the LLM's response to a single defection, not the opening move of an
   otherwise identical conditional rival.
3. **Fernandez Domingos and Han (2026).** `[full]` In humans, a first-round Unsafe choice
   predicted a greater later tendency toward Unsafe play, although its strength became
   marginal in the fullest specification. First-move path dependence in humans is theirs,
   exploratory, and already the motivation for this paper's trajectory framing.
4. **Fontana et al. (2025).** `[full]` Defines Suspicious TFT, never plays it.

#### Verdict: UNCLAIMED. The cleanest result in the paper

A pair of rivals identical from round two onward, differing only in round one,
differenced within a seed, is a clean identification of the opening move. No lane found
it in any of the eight papers read in this area.

The parallel with the source study, a first-round Unsafe choice carrying forward in
humans and a rival's first Unsafe move fixing the whole race for the LLM, is the best
unused narrative bridge in the campaign.

#### Narrowest true form

> Suspicious Tit-for-Tat has been played against frontier LLMs alongside ordinary
> Tit-for-Tat (Payne and Alloui-Cros, 2025), and a defect-once-then-cooperate probe has
> been used to test forgiveness \citep{playing_repeated_games_with_llms}, but no prior
> study differences a pair of rivals that are identical from round two onward in order to
> isolate the rival's opening move. Doing so here, inside a repetition so the horizon draw
> cancels, attributes $+25.1$, $+15.8$ and $+11.5$ points to the rival's first action
> alone, and at the lowest risk level a rival that merely opens \Unsafe{} and then mirrors
> drives the route to \Unsafe{} on all 93 of its decisions, the same as a rival that is
> \Unsafe{} throughout. \citet{falling_behind_unsafe} report the human analogue: a
> first-round \Unsafe{} choice predicts later \Unsafe{} play.

### C8d. Self-play misrepresents the policy

#### What the paper claims

> "A self-play rate is not a measurement of how a route treats risk. ... Every self-play
> number in this paper should be read that way." (`supplementary.tex:2322`)

#### Closest prior work

1. **Lekeas, P. V. and Stamatopoulos, G. (2026). "What Suppresses Nash Equilibrium Play
   in Large Language Models? Mechanistic Evidence and Causal Control." arXiv:2604.27167v2,
   4 May 2026.** `[full]` **The named predecessor, and it is not in the bibliography.**
   Four open models, Llama-3-8B and 70B Instruct and Qwen2.5-32B and 72B, four canonical
   2x2 games (PD, Stag Hunt, Chicken, Matching Pennies), 50 rounds, three reasoning modes
   (Direct, CoT, Scratchpad), metric Nash distance. They run self-play plus all 12 ordered
   cross-play pairings, and the framing sentence is: "Self-play tells us how each model
   behaves against a copy of itself, but what happens when models from different families
   and scales meet each other cannot be predicted from self-play alone." The abstract is
   harder still: "The cross-play experiments reveal three phenomena invisible in
   self-play." Their instance, the "8B defection unlock": Llama-8B cooperates most of the
   time against itself in Direct mode but defects 98% of the time in every cross-play
   pairing, and any large model paired with Llama-8B defects even though those same models
   cooperate 100% in self-play; PD Direct cross-play Nash distances are 0.063 for every
   pairing involving L-8B against 0.000 for large-model pairs. **Direction of their
   effect is the mirror image of this paper's:** for them self-play is the cooperative
   reading and contact with a different agent breaks it; here self-play is the escalated
   reading and contact with a fixed safe rival collapses it. That difference is real and
   usable, but it does not rescue conceptual priority.
2. **Pal, S., Mallela, A., Hilbe, C., Pracher, L., Wei, C., Fu, F., Schnell, S.,
   Nowak, M. A. (2026). "Strategies of cooperation and defection in five large language
   models." arXiv:2601.09849; published as "Large language models instantiate
   evolutionarily robust strategies of cooperation," PNAS Nexus, 2026.** `[full]`
   Already in the bibliography as `palCooperation2026`, and currently cited for something
   weaker than what it says. claude-sonnet-4, gemini-2.5-pro, gpt-4o, gpt-5,
   llama-3.3-70b, about 40,000 API calls. They do not play games to measure behaviour;
   they **elicit the memory-1 strategy directly**, asking each model 50 times what it
   would do in the first round and after each of LL, LR, RL, RR, then compute
   analytically how that inferred strategy fares against all 10^6 sampled memory-1
   opponents. Their motivation is the strongest existing statement of this paper's
   problem: prior work reports "realized outcomes of individual experiments, rather than
   the underlying LLM strategies that generate these outcomes," and eliciting the strategy
   "allows us to ask how LLMs would fare against any other (hitherto unobserved) opponent
   strategies." They mark which strategies are Nash equilibria during self-play and
   separately report the fraction of random memory-1 opponents that beat the self-payoff.
   The self-play-versus-arbitrary-opponent distinction is therefore formalised there,
   analytically, in January 2026.
3. **Long, O. and Teplica, C. (2025). "The AI in the Mirror: LLM Self-Recognition in an
   Iterated Public Goods Game." arXiv:2508.18467, 25 Aug 2025.** `[abstract]` **The lane
   could not get the full text; treat this as unverified.** Four reasoning and
   non-reasoning models in an iterated public goods game, told either that they play
   another AI agent or that they play themselves; reported finding is that telling LLMs
   they are playing themselves significantly changes their tendency to cooperate. Useful
   as a contrast: this paper's scripted prompt is byte-identical to the baseline and
   discloses nothing, so its divergence cannot be a self-recognition effect.
4. **Ye et al. (2026).** `[full]` Holds both interaction modes over the same game and
   models, so the matched self-play versus fixed-opponent measurement physically exists in
   their figures. They never draw the contrast; mode is a robustness axis for them. Someone
   will do this deliberately within a year.
5. **Payne and Alloui-Cros (2025), Table 13.** `[full]` Gemini's cooperation rises from
   0.852 in the mixed field to 0.925 in the LLM-only Showdown, read as adaptive
   intelligence. Same phenomenon class, opposite reading, and confounded because the
   opponent pool changes with the condition.

#### Verdict: NARROWER THAN IT LOOKS, with a named predecessor that must be cited in the same sentence

As written, "a self-play rate is not a measurement of how a route treats risk" is a
general methodological claim, and the general claim has a predecessor: Lekeas and
Stamatopoulos said in May 2026 that self-play cannot predict behaviour against a
different opponent, gave a 98-point instance, and named three phenomena as invisible in
self-play. Pal et al. formalised the same distinction analytically in January 2026, and
that paper is already in this bibliography.

What survives, and it is worth a lot:

* Their comparator is another LLM. Here it is a fixed, known, uninfluenceable strategy,
  so the direction of the effect is not in question.
* Their divergence is a Nash distance in games with no risk treatment. Here it is the
  response to **the treatment the study is about**: against Always Safe the route falls
  monotonically with stated risk, 24.7%, 17.2%, 14.0%, while in self-play the same route
  reads 98.9%, 73.1%, 60.2%. The self-play reading does not merely differ in level, it
  **hides the risk response**. That is sharper than "differs."
* Their finding is about heterogeneous model pairings; this one is one route against a
  strategy, so model identity confounds nothing.
* No prior work says this about an **AI-safety-relevant escalation rate**. Every prior
  self-play versus cross-play divergence is about cooperation in a PD.

Never write "we are the first to show that self-play misrepresents behaviour."

#### Narrowest true form

> That an LLM's self-play behaviour need not predict its behaviour against a different
> agent has been shown for cross-play between models: Lekeas and Stamatopoulos (2026)
> report phenomena invisible in self-play, including a model that cooperates with a copy
> of itself but defects in 98\% of rounds in every cross-play pairing, and
> \citet{palCooperation2026} reach the same distinction analytically by eliciting
> memory-1 strategies and evaluating them against $10^6$ sampled opponents rather than
> against copies of themselves. What has not been shown is the version that matters for
> an AI-race safety reading: that a route's self-play rate can hide the treatment
> response entirely. Against a fixed Always-Safe rival this route's \Unsafe{} rate falls
> monotonically with the stated private risk, $24.7\%$, $17.2\%$ and $14.0\%$; in
> self-play under a byte-identical prompt the same route reads $98.9\%$, $73.1\%$ and
> $60.2\%$. The self-play figure is an equilibrium of two copies of one policy escalating
> each other, and reporting it as the policy's response to risk overstates it by up to
> seventy-four points.

---

## Collision ledger, ordered by damage

Damage is judged by what a reviewer who knows the predecessor would conclude about the
authors, not by how interesting the predecessor is.

| # | Collision | Damage | What the paper should do |
|---|---|---|---|
| 1 | **The group's own preprint arXiv:2608.01193** is this paper minus the scripted campaign, public and de-anonymised, with a title one word from the submission title. | Fatal to anonymity, not to novelty. A double-blind reviewer who searches the title finds 13 named authors in one click. | Decide the anonymity handling now: cite it in third person as related work with no author names, or remove title overlap, or both. This is an AAMAS policy question, not a novelty question. It also means the scripted-opponent arm is the only part of the submission that is new relative to the public record. |
| 2 | **Piatti et al., GovSim (NeurIPS 2024)** does the comprehension-battery-predicts-behaviour claim with 15 models, 600 items each, four correlations at p < 0.001. The paper's related work cites it nowhere. | Very high. It is the flagship prior result for C1 and C2 at once, at 30 times the item count, and its absence reads as an undone survey. | Cite it in `sec:results-audit-behaviour` in the same paragraph as rho = 0.87, and state the slope-versus-level distinction there. Cite it again for the battery in `app:frontier-admission`. |
| 3 | **Payne and Alloui-Cros (2025), arXiv:2507.02618** runs Tit-for-Tat and Suspicious Tit-for-Tat as live opponents against frontier models under a stochastic horizon. That is the closest published design to the scripted campaign, and CS and CAS are those two strategies renamed. | Very high, and it is the riskiest single omission in the bibliography. | Add to `references.bib`. Say in `app:scripted-opponent` that the scripted-opponent design is standard, name Akata, Fontana, Payne and Ye, and claim only non-disclosure, replay verification and seat counterbalancing. |
| 4 | **Lekeas and Stamatopoulos (2026), arXiv:2604.27167** published the general claim that self-play cannot predict behaviour against a different opponent, with a 98-point instance and the phrase "invisible in self-play." | High. The supplement's "every self-play number in this paper should be read that way" is currently an uncited general claim. | Add to `references.bib` and cite it in the same sentence. Reframe the contribution as the treatment-response version: self-play hides the risk response, in a safety-relevant escalation rate, against an uninfluenceable rival. |
| 5 | **Huang et al. (2026), arXiv:2606.04978** and **Cacioli (2026), arXiv:2604.17714** are same-year predecessors for fail-the-check-lose-your-row and for screen-then-interpret-with-flagged-retention. | High. The introduction's "a validity gate applied before any strategic reading" is currently uncited against both. | Add both. Beat Huang on rigour rather than omitting them, and cite Cacioli where the gate's retention rule is described. Add the Akata asymmetry, humans gated and models not, as the justification sentence. |
| 6 | **The source study already inverts this exact model** on the (beta, mu) axis and reports beta = 0.01, mu = 0.05 as its own best fit, which `sec:results-theory` already quotes. | High, and internal. The paper presents an inversion as its own move while standing on the source's. | Reframe as identifiability: the contribution is the diagnosis that the risk axis is not identified, not the inversion. Add the calibration, indirect inference and QRE ancestry in one sentence. |
| 7 | **Hill numbers with rarefaction on LLM output** are already published twice: Wright et al. (EMNLP 2026, arXiv:2510.04226) and Hodel and West (arXiv:2512.15011). `app:diversity` currently introduces them from ecology first principles. | Moderate to high. Easy to fix, embarrassing if not fixed. | Cite Hill (1973), Jost (2006), Chao et al. (2014) for the method and both LLM applications for precedent. Keep the object and the null as the contribution. |
| 8 | **Zheng et al. (ICLR 2024, arXiv:2309.03882)** owns the letter anchor as token bias, and PriDe's permute-and-average is the same operation as the counterbalance; **Robinson and Burden (2025)** already cross narrative framing with a counterbalanced action-label mapping in a game; **Won et al. (2026, arXiv:2607.10202)** is specifically about what counterbalancing hides at the letter level on the same model generation. | Moderate. The finding survives as a replication in a new regime; only the wording is exposed. | Reframe `app:context-mapping` as a replication in a repeated-game regime with a signed, horizon-paired estimate. Cite Won et al. as the reason the letter-conditioned quantity is reported rather than differenced out, and add the per-mapping letter shares (Q on 77\% versus 11\%) as the position-lock diagnostic. |
| 9 | **Akata, Fontana, Ye** make fixed-opponent play standard practice, with Ye explicitly calling it common practice. | Moderate. Only harmful if the design is presented as introduced. | One sentence in `app:scripted-opponent`. |
| 10 | **Group size is settled**: PNAS at N up to 10^4 (arXiv:2510.22422), n up to 256 (arXiv:2602.16662), and a null trend at N = 3 to 10 (arXiv:2604.04782); and the N-player race group-size result is Han et al.'s own theory. | Moderate. The crossing with risk survives; the group-size claim alone does not. | Claim the crossing and the matched identification, not the effect. State that 7 of 12 cells are off the boundary. Cite the null-trend paper as the honest counterweight. |
| 11 | **Buscemi et al. (2504.08640) and Balabanova et al. (2503.09858)** are LLM studies of AI governance games with Han and Fernandez Domingos as coauthors. | Moderate, and specifically about self-citation. | Add both. Never write "first LLM study of AI race dynamics." |
| 12 | **Pechon-Elkins and Chun (2026, arXiv:2603.10029)** pairs a benchmark audit with a fitted-parameter inversion for LLM game benchmarks, and reaches the fitted-parameters-are-summaries caveat first. | Moderate. Six months before this submission. | Read the PDF, then cite. The lane could not extract it, so its exact content is an open question. |
| 13 | **Traulsen et al. (PNAS 2010)** is the canonical inversion of an EGT update rule against observed behaviour, reporting exploration rates far above what models assume, which is this paper's own weak-selection, high-exploration conclusion sixteen years earlier. | Moderate, and a Han-adjacent reviewer will notice immediately. | Read the PDF, verify mu_C and mu_D, then cite in `sec:results-theory`. |
| 14 | **Ruan et al. (2024)** makes tier collinearity the prior expectation, and **Cacioli (2026)** removes the "any gate would recover scale" defence by exhibiting a gate that does not. | Low to moderate. Only harmful if the collinearity is framed as a discovery. | Report it as an instrument limitation with both citations. |
| 15 | **Ye et al. (2026)** report a 76-point persona-format effect, the same size as the +73.5 rival-stance contrast. | Low, but a hostile reviewer will use it. | Answer it in the text: a rival's stance is the game's own state variable, a bullet-list persona is not. |
| 16 | **Anwar and Georgalos (2026)** use SFEM for exactly this question class. | Low. It is a "why not that method" question, not a priority claim. | Answer pre-emptively: SFEM cannot represent a population that collapses to one literal sequence. |
| 17 | **Wenger and Kenett (2025)** already report LLM populations far less diverse than matched humans, on creativity tests, with d from 1.4 to 2.2. | Low. Different domain and statistic. | One-line citation in `sec:results-diversity`. |

### Citation-accuracy defect found during the survey

`main.tex` lines 415 to 419 state that Huynh et al. "finds that cooperation can move in
the opposite direction from its evolutionary benchmark." The lane's reads of
arXiv:2512.07462 v1 and v2 and arXiv:2601.19082 report **no EGT benchmark in either
paper**: no replicator dynamics, no stationary distribution, no beta, no population size.
Both compare LLM trajectories against four canonical repeated-game strategies (ALLC,
ALLD, TFT, WSLS) via a supervised LSTM classifier trained on synthetic data, and against
backward-induction predictions. Reported numbers include PD payoff scaling
lambda in {0.1, 1.0, 10.0}, PGG r in {1.1, 2.0, 2.9}, Vietnamese-versus-English
cooperation gaps up to 29 points, Claude ALLC 31.7% and Llama WSLS 46.5%. Two coauthors
of this paper wrote those papers, so this is a one-sentence fix, but as written it is a
mis-citation that props up the evolutionary-benchmark framing. **PDF extraction failed on
both, so the lane read HTML renderings; confirm against the PDFs before rewriting.**

Related: the lane also confirmed that Huynh et al. (arXiv:2512.07462) does **not** script
opponents, contrary to what a search summary suggests. ALLC, ALLD, TFT and WSLS there are
classifier training labels, synthetic trajectories with noise in {0, 0.05} used to train
Logistic Regression, Random Forest, NN and LSTM, then applied to FAIRGAME logs of
LLM-versus-LLM play. No LLM in that paper ever faces a scripted rival. Since seven
authors are shared, one clarifying sentence both removes the risk and lets the scripted
arm be claimed as new relative to the group's own work.

---

## What the repository's own history says

The repository is unusually honest: retained failure records, `superseded_by.json`
pointers, an open protocol-amendment log, a fail-closed claim verifier passing 105 of 105
at HEAD `ed3863c`. Most of what follows is not sloppiness. It is material the history
handled correctly and the manuscript then failed to carry.

### R1. The admission gate has been administered three times and its verdicts flip

Identical `probe_bank_sha256 = 2953fb47...`, identical `rules_context_sha256 = b80981a5e8...`,
3 repetitions, temperature 0, in all three campaigns. The only decoding change on file is
the output cap, 128 to 256, between v1 and v5. **v5 and v6 have no recorded contract
difference at all.**

| Route | overall v1, v5, v6 | state recon. v1, v5, v6 | verdict across administrations |
|---|---|---|---|
| `anthropic/claude-sonnet-5@default` | 76.7, 81.7, 85.0 | 86.7, 86.7, 100.0 | **refused, admitted, admitted** |
| `google/gemini-3.1-flash-lite-preview` | 81.7, 76.7, 80.0 | 66.7, 60.0, 73.3 | refused throughout, but clears the 80\% overall bar in v1 |
| `google/gemini-3-flash-preview` | 95.0, 93.3, 93.3 | 100.0, 93.3, 93.3 | admitted throughout |
| `openai/gpt-5.4-nano-2026-03-17` | 53.3, 51.7, 51.7 | 20.0, 20.0, 20.0 | refused throughout |

The manuscript's central x-axis moves by +13.3 points on Claude Sonnet 5 and by +13.3
then -6.7 on Gemini 3.1 Flash-Lite, with no protocol change recorded between v5 and v6.
`sec:results-audit-behaviour` writes "Gemini 3.1 Flash-Lite misses admission on state
reconstruction, 73.3\% against the 75\% threshold" as if 73.3 were a fixed quantity; the
repository says the same route scored 60.0 on the same bank three days earlier. And
Claude Sonnet 5, an admitted route carrying the entire representation-robustness
contribution, was **refused** by this same gate at v1.

Hiding this is the wrong move. It is the strongest available answer to round-1 Reviewer
Q7, which asked for per-inference reliability metrics quantifying volatility across
repeated identical prompts, and which the manuscript still does not answer. Reported, it
converts the survey's most dangerous omission into a contribution: this is what an audit
gate's reliability actually looks like. Paths:
`results/frontier/admission_campaign/`, `results/frontier/admission_campaign_v5/`.

### R2. The supplement says every self-play number must be re-read; the main paper never tells the reader

`supplementary.tex:2311` states it. Commit `f61e6ff` is titled "First two
scripted-opponent cells overturn the reading of the baseline." `main.tex` contains the
string "self-play" exactly twice, both about the EGT estimand boundary at lines 563 and
867. It never says the gameplay is mirror-match self-play, and it builds on that
gameplay: the rho = 0.87 headline is a self-play risk response, the theory-versus-behaviour
section uses self-play trajectories, and the closing sentence that every route passing
the gate responds strongly to the assigned risk is what the scripted campaign shows to be
an escalation-spiral artefact.

Under this project's own AAMAS rule, reviewers are not obliged to read the supplement, so
nothing a claim depends on may live only there. This correction is in the wrong document.
**Fix first.**

### R3. Table 2 in `main.tex` is from the two-route era

`main.tex:615-628`, the evidence-strata table:

| Row as printed | What the paper now reports |
|---|---|
| Frontier endpoint admission, **two admitted routes**, 120 probe responses | nine routes, 540 retained outputs, five admitted |
| Five-checkpoint baseline, 150 races, 2,790 decisions | `sec:results-diversity` uses **seven** checkpoints, 420 trajectories |
| Context pilot (temperature 0), 768 races, 13,680 decisions | `sec:results-sensitivity` excludes it and keeps it only in the audit trail |
| Fixed-state replay, 1,536 state cells | appears nowhere else in either document; no result reported |

The 120-response row survives from `results/frontier/admission_campaign_v5/`, two routes
by 60 rows, which commit `ccba3eb` superseded. `verify_manuscript_claims.py` does not
cover this table.

### R4. The supplement carries a superseded admission table beside the current one

`supplementary.tex:438-475`, `tab:frontier-admission-registry`, one column set:

| Row | Actual source | Rows behind it |
|---|---|---|
| `gemini-3-flash-preview` 93.3 / 93.3 / 100 | v5 confirmatory | 60 |
| `claude-sonnet-5@default` 81.7 / 86.7 / 80 | v5 confirmatory | 60 |
| `gemini-3.1-flash-lite-preview` 75 / 60 / 80 | **`admission_smoke`** | **20** |
| `gpt-5.4-nano` 50 / 20 / 60 | **`admission_smoke`** | **20** |

Only the two refused rows come from the 20-answer smoke, and the caption does not mark
which rows are which. Claude Sonnet 5's own smoke was 75.0, which would refuse, against
its 81.7 confirmatory. A reviewer comparing this table with main-paper Table 1 sees
`gemini-3.1-flash-lite` at 75/60/80 in one place and 80.0/73.3/80.0 in the other, with no
sentence saying they are different campaigns.

### R5. The disclosed-arithmetic table is a main-paper result from an unnamed checkpoint that fails the paper's own gate

`main.tex:782-808` reports canonical 52.0% against decision card 60.8%. The source is
`results/open_source/game_understanding_pilot/`, that is **Qwen2.5-7B-Instruct-fp16**,
whose own audit in the same directory reads state reconstruction 37.0%, state transition
22.2%, terminal scoring 53.3%, expected payoff 16.7%, overall semantic accuracy 59.1%
across 41 probes and 685 outputs. The strings "Qwen2.5" and "Qwen2.5-7B" appear nowhere in
either `.tex`. `sec:design` states the governing rule, that a comprehension check gates
all of them, and the paper applies it to the context-skin pilot but not here, in a
subsection placed immediately after the nine-route frontier gate where every reader will
assume a frontier route.

Two further problems with the same table:

* **Two live interval versions.** `docs/game-understanding-audit-results.md` and
  `results/open_source/game_understanding_pilot/admission.json` give canonical 48.0 to
  55.9 and card 51.2 to 67.9, an overlap of 4.7 points.
  `results/cross_model_pilot_synthesis/data/disclosed_arithmetic_race_bootstrap.csv`,
  which the verifier reads, gives 48.9 to 55.4 and 55.1 to 65.3, an overlap of **0.3
  points**. Neither file marks the other as superseded.
* **The headline reading is inside the repository's own documented Monte Carlo noise.**
  "The two race-clustered intervals still touch" is 55.06 against 55.35, from a csv using
  4,000 resamples with **different bootstrap seeds per arm** in a paired comparison.
  Commit `47f900b` established for this repository that ten-block percentile endpoints
  carry a few tenths of movement even at 200,000 draws. A 0.3-point margin at 4,000 draws
  is not resolvable.

### R6. Numeric and provenance slips the verifier does not cover

* `main.tex:743`, "the largest per-cell difference is 5.3 for Claude Sonnet 5."
  Recomputed from raw `turns.jsonl`: v2 Claude 89.2473 / 46.2366 / 37.6344 against v6
  89.2473 / 48.3871 / 32.2581, largest absolute difference 5.376, so **5.4**. Gemini's
  1.1 checks out.
* The same sentence attributes the Gemini figures to "the earlier confirmatory block."
  The verifier pins 1.1 and 38.7 to `results/derived/baseline_replication.json`, which is
  a **later** run (2026-09-09, task version 4, different identity) held deliberately
  outside the campaign tree, while the Claude 5.4 can only come from v2 against v6. One
  sentence, two different comparisons, one label.
* `main.tex:1085`, "GPT-5.4-nano has no single strong predictor."
  `feature_importance_results.json` gives it progress-gap 36.5%, exceeding Gemini 3
  Flash's 32.5% that the same paragraph calls largest. The supplement gets this right by
  adding the AUC 0.54 and balanced-accuracy 0.52 justification; the main text drops the
  justification and keeps the conclusion.
* `docs/reviewer-revision-frontier-protocol.md` says "Two route families reject the
  reasoning-budget argument"; CLAUDE.md and the v6 manifests identify exactly one,
  `google/gemini-3.5-flash-lite`, task version 8.

### R7. The shipped deliverables are behind the source

`paper/ai_race_paper.pdf` and `paper/ai_race_supplementary.pdf` (both 10:32) and their
mirrors in `results/artifacts/publication/` predate commits `ad85038` (21:50), `47f900b`
(21:52) and `ed3863c` (22:03). The shipped supplement is **18 pages and contains no
scripted-opponent appendix at all**: zero occurrences of "exploitation", "retaliation",
"rival's stance" or "Every self-play number". The source supplement is 20 pages with
`app:scripted-opponent`. `paper/main.pdf` (22:20) is **corrupt**: pypdf fails with
"Stream has ended unexpectedly." CLAUDE.md's rule that final PDFs must always be written
to `paper/ai_race_paper.pdf` is currently violated.

### R8. `app:context-mapping` contradicts itself

Lines about 808 to 830 state the crossed rerun "has now completed on both admitted
routes," naming Gemini 3 Flash run 1373757 and Claude Sonnet 5 run 1504455, 120 races and
2,232 decisions each. The closing paragraph at about 927 to 935 states "the design is
crossed for one route only," "The matched rerun on `anthropic/claude-sonnet-5@default`
did not complete," and "we ... make no mapping claim."

The data agree with the first version:
`results/derived/frontier_context_mapping_campaign_v3/context_mapping_cross_marginals.csv`
carries `anthropic/claude-sonnet-5@default` rows at 60 races and 1,116 decisions per
mapping level, the full 120. The closing paragraph is stale text from before the Claude
rerun landed, and it **directly negates the two-route claim in the main paper**. Delete or
rewrite it. Minor, same section: the text points to "the derived table named at the end of
this subsection" for the second route's twelve cells, but only
`google-gemini-3-flash-preview_cells.csv` exists in the derived directory.

### R9. Abandoned and superseded work, with the reason on file

Retained failure records, all with reasons: `nplayer_baseline_n3_20260908.json`
(non-retryable model request failure), `nplayer_baseline_n3_v5_20260908.json` (HTTP 403
quota refusal at task-creation validation, 0 races), `nplayer_matched_daosyduyminh_20260910.json`
and `nplayer_matched_trungkiet_20260910.json` (403 per-account token reservation; the
second establishes roughly 800 to 1,000 requests per identity against roughly 7,800
needed), `nplayer_matched_n3_risk0p1_hunhtrungkit_20260910.json` (HTTP 429 route
congestion, retried on the same declared identity rather than rotated),
`context_mapping_claude_1373758_20260909.json` (403 at 106 of 120 races, unbalanced
fragment, never tabulated), `context_mapping_gemini_1373757_download.json` (Kaggle output
download failed, log-only values refused), `comprehension_reaudit_v2_greennode_66f1029/`
(PyTorch/NVML allocator error at batch 8 on a 20 GB MIG, retry at batch 1 succeeded),
`kaggle_crossmodel_scaffold_smoke_v1` through `v4` (four consecutive kernel ERROR
terminations, zero admitted requests), the v6 `gemini-3.5-flash-lite` transport failure
superseded by the task-version-8 rerun with the reasoning parameter omitted, and
`results/frontier/nplayer_failed_attempt_1377133/` (request errors, no gameplay).

Routes audited and dropped: `deepseek-ai/deepseek-r1-0528`, `ibm/granite-4.0-h-small`,
`openai/gpt-oss-120b`, `qwen/qwen3-next-80b-a3b-instruct`, all with bounded transport
retries exhausted, all reported in the supplement. **`google/gemma-4-26b-a4b` was also
attempted in v1 and v5, both failed, and appears in neither table.** A fifth unreachable
route, unreported.

Whole programmes frozen and never executed: positive payoff-scale invariance (P0, 384
races; mechanical contract passed 1,048,512 terminal comparisons at max normalised error
1.4e-14; GPU runner blocked on SSH auth) which is **the third leg of the paper's own
declared invariance triple**, narrative, code and utility unit, of which two shipped;
transition-by-terminal computation scaffold (P0, 768 races, GPU blocked); opponent
belief and action coherence (P1, 768 races, runner not admitted); capacity and family
selectivity (frozen protocol, no model output admitted, GreenNode smoke failed closed on
GPU offload in BF16); replay-to-fork feedback, per-turn opaque remapping and SAE causal
promotion, all conditional and never run; context-recognition and contamination audit v1
rejected after pilot for a contradictory response contract, v2 written and not run.

**The entire SAE and XAI lane was cut from the manuscript.**
`paper/CITATION_CHANGELOG.md` still carries copy-ready LaTeX for a main-text SAE section,
including the negative result that with only six held-out race clusters, none of the 12
target-minus-control intervals excluding zero is a failure to establish feature
specificity. Six analysers and a runner survive, plus `results/open_source/activation_sae/`.
"SAE" and "sparse autoencoder" now appear zero times in either `.tex`. The audit that
justifies the cut is `results/impact_upgrade/xai_claim_audit.md`: decodability without
control, double-held-out AUC 0.985 at layer 20, 0 action flips in 1,312 intervention rows
per layer, 0 of 12 preregistered intervals excluding zero.

Empty shells with no failure record: `results/frontier/context_mapping_campaign_v1/`,
`context_mapping_campaign_v2/`, `results/frontier/nplayer_failed_attempt_claude_20260908/`
are untracked zero-file directories never present in git. The v3 Claude attempt is
retained exemplarily; v1 and v2 left nothing at all. This is the one hole in an otherwise
complete accounting chain.

### R10. Live results in the repository the paper is not using and should

1. **The gate's own test-retest** (R1). Answers Reviewer Q7 directly.
2. **Exogenous position endowment**, `results/open_source/position_endowment_greennode_e3cf825/`,
   commits `e3cf825` and `15e4bc1`. An engine-scored exogenous progress adjustment applied
   after one common four-round history, which is precisely the randomised-rank design
   Reviewer Q5 asked for and which `docs/reviewer-revision-frontier-protocol.md` lists in
   its review-to-evidence matrix. 192 rows per block, 0 parse failures, both mappings,
   numeric-only and verified-rank-label arms. Primary contrasts from the numeric-only arm,
   block 1: Qwen2.5-7B behind minus ahead in the two-player game **+0.0**, last minus
   leader at N = 3 +41.7, last minus middle +8.3; Mistral-7B **+0.0** on all three. That is
   a **null on the source study's title effect** in the two-player game under exogenous
   position, with a large effect only at N = 3 on one checkpoint. Both checkpoints fail the
   comprehension gate, so it cannot be a headline, but the paper's limitation that no pure
   group-size effect is identifiable would be far stronger beside it, and the SHAP
   disclaimer currently reads as if no causal design was ever attempted.
3. **Temperature-zero reproducibility, three independent measurements.**
   `results/open_source/heterogeneous_dyad_greennode_ba2906a/`, lane-counterbalanced
   replication, exact per-decision agreement 98.6% (Qwen 97.5%, Mistral 99.7%) across H100
   MIG lane assignments at T = 0; position endowment block 2, matched-probe action
   agreement 94.8% (Qwen) and 100.0% (Mistral); `docs/game-understanding-audit-results.md`,
   all five T = 0 repetitions byte-identical in every item-condition cell. The decoding
   caveats, temperature not confirmed and seed stripped for `google/` routes, would land
   much harder with a measured number attached.
4. **Mechanical validity, level 1 of the paper's own four-level audit, never reported.**
   `main.tex:437` promises four levels, the first being that the game engine applies the
   stated rules correctly. Neither document reports a single mechanical-validity result.
   The repository has them: `results/derived/payoff_scale_contract/summary.json`, 21,844
   joint action sequences by 4 payoff scales by 3 risks by 4 setback-draw pairs, that is
   1,048,512 terminal comparisons at max normalised final-payoff error 1.42e-14;
   `scripts/verify_matched_nplayer_design.py`, asserting the N = 2 arm reproduces the
   headline matrix exactly; a 118-test pytest suite; the frozen prompt SHA-256 contract;
   and all joint action histories through length four matching the reference calculator.
   One sentence and a supplementary paragraph close a promise the paper currently makes and
   does not keep.
5. **The scripted-opponent campaign belongs in the main paper.** It is the only
   non-self-play gameplay evidence, the only design the repository calls causal by
   construction, the only place the evolutionary lane's four reduced strategies are ever
   actually played against, and the result that most changes how
   `sec:results-audit-behaviour` should be read. It is also the only part of the submission
   that is new relative to the public preprint arXiv:2608.01193. It currently lives entirely
   in a document reviewers are not obliged to open.
6. **`results/reports/frontier/report.md`, a live, unmarked, contradicting report.** Its
   section 7 headline is "4/8 replicated" on the E1 to E8 human-effect comparison (E1, E3,
   E6, E7 replicated). The current derived artifact
   `results/derived/agent_check_frontier_baseline/human_comparison.csv` gives E1 to E4
   inconclusive with no logit fitted, and E7 not replicated, phi_U = 0.771 outside the
   declared [0.4, 0.75] band. Neither file points at the other. **E2, `progress_gap_before`,
   the source study's own title effect, is not replicated or inconclusive in every
   version**, which together with item 2 gives the repository two independent readings that
   "falling behind" does not transfer to these agents. The paper reports neither.
7. **Two off-roster checkpoints with complete SHAP profiles**, `gpt-5.6-luna` (round number
   41.3%) and `gpt-5.6-terra` (risk 32.3%, own-previous 31.4%), sit in
   `feature_importance_results.json` and are excluded from the seven-checkpoint story. The
   2026-09-08 stress test flagged mixing them in as a defect, so the exclusion is
   deliberate, but nothing in either `.tex` records that two further checkpoints were
   measured and set aside.

### R11. An unclosed judgement call the history raised twice

`docs/submission_readiness_stress_test_2026-09-08.md` contains the original rarefaction
table, where the same result was read as a **blocker against the title**: "This
strengthens the model-specific policy claim and weakens the blanket
humans-are-more-diverse claim," with two proposed replacement titles. The live table,
`results/cross_model_pilot_synthesis/data/trajectory_diversity_rarefaction.csv`, is
unchanged in substance, with GPT-5.4 nano at 18.66 at all three risks, and commit
`47f900b` now prints both nulls, 18 of 21 and 15 of 21. The resolution recorded in the
follow-up was explicitly "a judgement call for the manuscript owners rather than a missing
artifact." That is an open item, not a closed one. The body sentence at `main.tex:966` is
already the correct, gated form; the title's unqualified lead clause is what the history
objects to.

---

## Bibliography check

Checked against `paper/references.bib`, the only `.bib` file in `paper/`. It holds 50
entries.

### Cited in this survey and already present

`falling_behind_unsafe`, `to_regulate_or_not`, `artificial_intelligence_development_races`,
`playing_repeated_games_with_llms` (Akata), `palCooperation2026` (Pal et al. is **already
in the bib**, contrary to one lane's note, though the intro currently cites it for
something weaker than what it says), `herrStrategicBias2024`, `robinsonBurdenFraming2025`,
`delRioChanonaMarkets2025`, `huynhUnderstanding2025`, `huynhPayoffScaling2026`,
`buscemiFAIRGAME2025`, `gruetzemacher2025strategic`, `weiSelectionBias2024`,
`pezeshkpourOptionOrder2024`, `sclarPromptFormatting2024`, `salinasButterfly2024`,
`zhengPersona2024`, `wuCounterfactual2024`, `synthetic_replacement_human`,
`anthisSocialSimulation2025`, `codaFornoCogBench2024`, `wangSocioBench2025`,
`suhSurveyDistributions2025`, `wangHumanSubjectivity2024`, `zhengBoundedRationality2025`,
`yaoCompetition2026`, `luMultiTurn2026`, `han2019modelling`, `han2022voluntary`,
`fernandezDomingos2020Egttools`.

### Must be added before submission

Ordered by the damage the omission does. Suggested keys in brackets.

1. `[payneAlloui2025]` Payne, K. and Alloui-Cros, B. "Strategic Intelligence in Large
   Language Models: Evidence from evolutionary Game Theory." arXiv:2507.02618, 2025.
2. `[piattiGovSim2024]` Piatti, G., Jin, Z., Kleiman-Weiner, M., Scholkopf, B., Sachan, M.,
   Mihalcea, R. "Cooperate or Collapse: Emergence of Sustainable Cooperation in a Society
   of LLM Agents." NeurIPS 2024, arXiv:2404.16698.
3. `[lekeasStamatopoulos2026]` Lekeas, P. V. and Stamatopoulos, G. "What Suppresses Nash
   Equilibrium Play in Large Language Models? Mechanistic Evidence and Causal Control."
   arXiv:2604.27167, 2026.
4. `[huangStPetersburg2026]` Huang, C., Chen, C., Lin, C., Lyu, H., Xu, X., Luo, J.
   "Probing Outcome-Level Resemblance and Mechanism-Level Alignment in LLM Risk Decisions:
   Evidence from the St. Petersburg Game." arXiv:2606.04978, 2026.
5. `[cacioliScreen2026]` Cacioli, J.-P. "Screen Before You Interpret: A Portable Validity
   Protocol for Benchmark-Based LLM Confidence Signals." arXiv:2604.17714, 2026.
6. `[fontanaNicer2025]` Fontana, N., Pierri, F., Aiello, L. M. "Nicer Than Humans: How do
   Large Language Models Behave in the Prisoner's Dilemma?" Proc. ICWSM 19:522-535, 2025
   (arXiv:2406.13605).
7. `[yeRobustnessAudits2026]` Ye, Cao, Chen, Ferrara. "Stop Drawing Scientific Claims from
   LLM Social Simulations Without Robustness Audits." arXiv:2605.18890, 2026.
8. `[wrightEpistemic2026]` Wright, D., Masud, S., Moore, J., Yadav, S., Antoniak, M.,
   Ebert Christensen, P., Park, C. Y., Augenstein, I. "What and Whose Knowledge? Measuring
   Epistemic Diversity in Large Language Models." EMNLP 2026, arXiv:2510.04226.
9. `[hodelWest2025]` Hodel, D., West, J. D. "Epistemic diversity across language models
   mitigates knowledge collapse." arXiv:2512.15011, 2025.
10. `[hill1973]` Hill, M. O. "Diversity and evenness: a unifying notation and its
    consequences." Ecology 54(2):427-432, 1973. `[abstract]` **Standard method citation,
    carried from the lane's identification of the required ancestry rather than read.**
11. `[jost2006]` Jost, L. "Entropy and diversity." Oikos 113(2):363-375, 2006.
    `[abstract]` Same caveat.
12. `[chaoRarefaction2014]` Chao, A. et al. "Rarefaction and extrapolation with Hill
    numbers: a framework for sampling and estimation in species diversity studies."
    Ecological Monographs 84(1):45-67, 2014. `[abstract]` Same caveat.
13. `[zhengSelectors2024]` Zheng, C., Zhou, H., Meng, F., Zhou, J., Huang, M. "Large
    Language Models Are Not Robust Multiple Choice Selectors." ICLR 2024,
    arXiv:2309.03882.
14. `[wonCounterbalancing2026]` Won, H.-I., Jang, J., Kim, H. "When Counterbalancing Hides
    the Bias: Access-Conditioned Position Lock in Forced-Choice LLM Evaluation."
    arXiv:2607.10202, 2026.
15. `[traulsenUpdating2010]` Traulsen, A., Semmann, D., Sommerfeld, R. D., Krambeck, H.-J.,
    Milinski, M. "Human strategy updating in evolutionary games." PNAS 107(7):2962-2966,
    2010. **Read the PDF before quoting mu_C and mu_D.**
16. `[pechonElkinsChun2026]` Pechon-Elkins, M., Chun, J. "Auditing Game-Theoretic Measures
    of Strategic Reasoning in LLMs." arXiv:2603.10029, 2026. **PDF not extracted; read
    before citing.**
17. `[buscemiTrustRegulation2025]` Buscemi, A., Han, T. A. et al. "Do LLMs trust AI
    regulation? Emerging behaviour of game-theoretic LLM agents." arXiv:2504.08640, 2025.
18. `[balabanovaMedia2025]` Balabanova, N. et al. "Media and responsible AI governance: a
    game-theoretic and LLM analysis." arXiv:2503.09858, 2025; Phil. Trans. R. Soc. A.
19. `[ruanObservational2024]` Ruan, Y., Maddison, C. J., Hashimoto, T. "Observational
    Scaling Laws and the Predictability of Language Model Performance." NeurIPS 2024,
    arXiv:2405.10938.
20. `[groupSizeMisalignment2026]` "Group size effects and collective misalignment in LLM
    multi-agent systems." arXiv:2510.22422, 2026; PNAS. **Author list not captured by the
    lane; complete it before citing.**
21. `[hundredsOfAgents2026]` "Evaluating Collective Behaviour of Hundreds of LLM Agents."
    arXiv:2602.16662, 2026. **Author list not captured; complete before citing.**
22. `[cheapTalk2026]` "Cheap Talk, Empty Promise: Frontier LLMs easily break public
    promises for self-interest." arXiv:2604.04782, 2026. **Snippet only; author list and
    numbers unverified.**
23. `[anwarGeorgalos2026]` Anwar, A., Georgalos, K. "Playing Against the Machine:
    Cooperation, Communication, and Strategy Heterogeneity in Repeated Prisoner's
    Dilemma." arXiv:2603.15852, 2026.
24. `[wengerKenett2025]` Wenger, E., Kenett, Y. "We're Different, We're the Same: Creative
    Homogeneity Across LLMs." arXiv:2501.19361, 2025.

### Optional, add if the corresponding sentence is written

`[zhaoCalibrate2021]` Zhao et al., ICML 2021, arXiv:2102.09690;
`[reifSchwartz2024]` Reif and Schwartz, NAACL 2024, arXiv:2405.02743;
`[liTargetedTests2026]` Li, arXiv:2605.11599;
`[zhuZhang2026]` Zhu and Zhang, arXiv:2609.04198;
`[burnellStructure2023]` Burnell et al., arXiv:2306.10062;
`[llmMoranDilemma2025]` arXiv:2501.16173;
`[wangTestability2017]` Wang, Chen and Wang, Physica A 486:455-464, 2017;
`[loreHeydari2024]` Lore and Heydari, Scientific Reports 14, 2024;
`[pasarkarDieng2024]` Pasarkar and Dieng, AISTATS 2024, arXiv:2310.12952;
`[kleinbergRaghavan2021]` Kleinberg and Raghavan, arXiv:2101.05853;
`[bommasaniHomogenization2022]` Bommasani et al., NeurIPS 2022, arXiv:2211.13972;
`[doshiHauser2024]` Doshi and Hauser, Science Advances 10(28):eadn5290, 2024;
`[longTeplica2025]` Long and Teplica, arXiv:2508.18467 **(abstract only; do not cite
until read)**;
`[kearnsConstruct2025]` Kearns, arXiv:2602.15532 (MSc thesis, supporting only);
`[ilicGignac2024]` Ilic and Gignac, Intelligence, 2024 **(numbers second-hand; verify
before citing)**.

### Cannot be cited normally

`arXiv:2608.01193`, the group's own preprint. It is prior art for everything in this
paper except the scripted-opponent campaign, and citing it by title de-anonymises the
submission. Decide the handling with the corresponding authors before the AAMAS deadline.

### Open questions this survey could not close

* The exact comparison method in Buscemi et al. (arXiv:2504.08640): PDF extraction failed.
* The full content of Pechon-Elkins and Chun (arXiv:2603.10029): PDF extraction failed,
  and it is the closest audit-plus-inversion predecessor.
* The author list and exact protocol of arXiv:2604.04782 and arXiv:2510.22422.
* Whether Long and Teplica (arXiv:2508.18467) says what its abstract says.
* Whether Traulsen et al.'s reported exploration rates are the numbers quoted here.
* Whether the Ilic and Gignac variance and correlation figures are as reported
  second-hand through Kearns.

Nothing in the list above was treated as clearance. Each is an unclosed question, not an
absence of prior work.
