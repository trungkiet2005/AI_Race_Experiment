# Related work sources for the ARR evaluation paper

Built 2026-09-19 for `paper/acl/main.tex`, section `sec:related`. The BibTeX for
every key below is in `custom_candidates.bib` next to this file. Nothing here has
been moved into `custom.bib` or cited in `main.tex`.

**How each entry was checked.** Title, authors, year and venue were read at a
primary record: the publisher's DOI record (Crossref), the ACL Anthology page,
the NeurIPS, ICLR or PMLR proceedings page, or the arXiv abstract page. Where a
published version exists it is the one in the .bib. The text was then read:
"abstract" means only the abstract was read; "full text" means the named sections
were opened and read. The sentence on what a work shows is paraphrased from that
reading, never from memory.

**Reading tags.** `[abstract]`, `[full text: sections]`, `[metadata only]`.

**Paper context.** The ARR paper treats a comprehension screen as a measurement
instrument and asks (RQ1) whether comprehension is one ability or fractionated
by subtask, (RQ2) whether the screen is reliable, (RQ3) whether it carries
information beyond cheap signals such as the model name or tier, (RQ4) whether a
state-tracking score transfers across a two-player race, an iterated social
dilemma and a four-player public goods game, and (RQ5) whether admission
predicts behaviour.

Counts: A 18, B 6 read plus 1 metadata only, C 5, D 4, E 6. Total 39 read, 40
entries.

---

## A. Comprehension, manipulation and attention checks; LLMs as simulated subjects; LLM game-theory evaluation

### A1. Checks in human experiments

**`oppenheimer2009instructional`**
Source: https://doi.org/10.1016/j.jesp.2009.03.009 `[full text: abstract, introduction, General discussion, p. 870]`
Shows: introduces the instructional manipulation check, a question that detects
participants who do not read instructions, and reports failure rates from 14% to
46% across its samples; the authors say that removing failers, or better, making
them reread, raises statistical power.
Bears on the paper: positions. The comprehension screen is the model analogue of
an IMC, and Oppenheimer et al. already note that an IMC only works if failing it
predicts failing the real task, which is exactly the convergent-validity
question the paper asks of its screen (RQ5).

**`berinsky2014separating`**
Source: https://doi.org/10.1111/ajps.12081 `[abstract]`
Shows: a single attention screener is not the best way to measure attention,
several items work better, and passing the screener correlates with politically
relevant respondent characteristics, so excluding failers limits
generalisability; the authors recommend reporting results conditional on
attention level rather than dropping failers.
Bears on the paper: supports, and is the closest human analogue of RQ3. A human
screener whose verdict tracks respondent traits is the same problem as a model
screen whose verdict tracks model tier; "multiple items beat one" also speaks to
the single-probe baseline.

**`aronow2019note`**
Source: https://doi.org/10.1017/pan.2019.5 `[abstract]`
Shows: dropping subjects who fail a post-treatment manipulation check can bias
estimates seriously; bounds for the always-passers can be wide or infinite, and
the authors recommend changing the design instead of discarding subjects.
Bears on the paper: supports the choice to retain refused routes and report their
behaviour rather than delete them; it also gives the formal reason why a gate
that selects on an outcome-related trait can change what the behavioural
analysis estimates.

**`chou2009control`**
Source: https://doi.org/10.1007/s10683-008-9206-4 `[abstract]`
Shows: in a two-person guessing game many subjects fail to choose the weakly
dominant strategy because they do not recognise the game form (the mapping from
choices to outcomes and payoffs); recognition depends on how the game is
presented, and without it experimental control over game theory tests is lost.
Bears on the paper: supports the premise that comprehension of rules and payoffs
is a precondition for reading behaviour as strategy, and that comprehension can
be presentation specific, which motivates the transfer question (RQ4).

**`burtonchellew2016conditional`**
Source: https://doi.org/10.1073/pnas.1509740113 `[abstract and significance statement]`
Shows: in public goods games, variation in behaviour is better explained by
variation in understanding than by variation in fairness preferences, and
misunderstanding leads to cooperation.
Bears on the paper: supports gating in the F3 public goods family specifically;
in humans, confusion is not noise but produces apparent cooperation, so an
unscreened model's "cooperation" in F3 is uninterpretable.

### A2. LLMs as simulated subjects

**`aher2023using`**
Source: https://proceedings.mlr.press/v202/aher23a.html `[abstract]`
Shows: proposes Turing Experiments, in which an LLM simulates a sample of
participants in classic studies (Ultimatum Game, garden path sentences, Milgram,
wisdom of crowds); three replicate, the fourth reveals a "hyper-accuracy
distortion" in newer models.
Bears on the paper: positions. The simulated-subject programme asks whether
models behave like humans; it does not ask whether models understood the task,
which is the step the paper audits.

**`argyle2023out`**
Source: https://doi.org/10.1017/pan.2023.2 `[abstract]`
Shows: conditioning GPT-3 on sociodemographic backstories reproduces the response
distributions of human subgroups ("algorithmic fidelity", "silicon samples").
Bears on the paper: positions. Fidelity is judged at the outcome level; the paper
argues that an outcome-level match says nothing about whether the model
comprehended the instrument.

**`filippas2024large`**
Source: https://doi.org/10.1145/3670865.3673513 (EC '24, pp. 614-615); arXiv 2301.07543 `[abstract, arXiv page]`
Shows: treats LLMs as "homo silicus" that can be given endowments, information
and preferences, and replicates classic behavioural economics experiments with
qualitatively similar results.
Bears on the paper: positions. This is the work usually cited as "Horton (2023)";
see the naming note under "Not found / not verified". Endowing a model with a
role presupposes it has understood the role, which is untested there.

**`dillion2023can`**
Source: https://doi.org/10.1016/j.tics.2023.04.008 `[abstract, PubMed 37173156]`
Shows: a short opinion piece reviewing whether and when language models might
replace human participants in psychology, with a theoretical model and caveats.
Bears on the paper: positions; a citation for the general "AI as participant"
framing. The abstract is short, so do not cite it for any specific claim beyond
that framing without reading the article.

### A3. LLM game-theory evaluation

**`akata2025playing`**
Source: https://doi.org/10.1038/s41562-025-02172-y; arXiv 2305.16867v2 `[full text: Methods (Human-LLM interactions, Design); Results (Prisoner's Dilemma robustness checks)]`
Shows: LLMs play finitely repeated 2x2 games; they do well in self-interested games
such as the iterated Prisoner's Dilemma and poorly in coordination games such as
Battle of the Sexes. Human participants had to answer a comprehension
questionnaire correctly before playing ("Only upon responding correctly to all
questions, they could proceed"); the five LLMs (GPT-4, text-davinci-003 and 002,
Claude 2, Llama 2 70B) received no comprehension or rule check, only prompt
robustness variations (option order, relabelling, currency).
Bears on the paper: supports the gap. The flagship repeated-games study gates
humans on comprehension and does not gate models, which is the asymmetry the
paper starts from. Correction to the task brief: Akata et al. do not gate on rule
understanding.

**`duan2024gtbench`**
Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/3191170938b6102e5c203b036b7c16dd-Abstract-Conference.html; arXiv 2402.12348 `[abstract; full text: Appendix A1, A5.6, Section 4.4]`
Shows: GTBench, 10 game-theoretic tasks over a taxonomy of information,
dynamics and chance; LLMs fail in complete-information deterministic games and
compete in probabilistic ones. There is no comprehension test before play; a
"completion rate" sanity check counts matches with only legal moves, a prompt
adapter converts outputs into legal actions, and the error analysis finds
"misinterpretation" errors about piece ownership.
Bears on the paper: contrasts. Rule failures are observed after the fact and
partly repaired by the harness, rather than measured by a screen. The proceedings
title says "Capabilities"; the arXiv title says "Limitations". This is the "Duan
et al." of the brief.

**`huang2025competing`**
Source: https://proceedings.iclr.cc/paper_files/paper/2025/hash/a46adbe2f0ca0e16ef8857e188991ad7-Abstract-Conference.html; arXiv 2403.11807v7 `[abstract; full text: Section 4.1]`
Shows: GAMA-Bench, eight multi-player classical games with a dynamic score; 13
LLMs from 6 families, Gemini-1.5-Pro highest at 69.8/100. Section 4.1 reruns
games five times, varies temperature from 0 to 1 and paraphrases prompts; prompt
wording can change scores considerably. There is no rule-understanding test.
Bears on the paper: contrasts. It checks the stability of the gameplay score,
not of any comprehension measure. The ICLR 2025 published title differs from
the arXiv title ("How Far Are We on the Decision-Making of LLMs?"); cite the
published one.

**`fan2024can`**
Source: https://doi.org/10.1609/aaai.v38i16.29751 `[abstract]`
Shows: decomposes rationality into building a desire, refining a belief and
acting optimally, and tests these in the dictator game, Rock-Paper-Scissors and a
ring-network game; even GPT-4 falls short of humans, for example failing to
refine beliefs from simple patterns, and the authors urge caution about using
LLMs in game experiments.
Bears on the paper: supports RQ1 in spirit. Rationality is already decomposed
into separable components that fail differently, which is the fractionation
claim applied to strategic reasoning rather than to comprehension.

**`lore2024strategic`**
Source: https://doi.org/10.1038/s41598-024-69032-z `[abstract]`
Shows: GPT-3.5, GPT-4 and LLaMa-2 differ in how much their choices respond to
game structure versus contextual framing; GPT-3.5 is highly context sensitive,
GPT-4 attends to structure but distinguishes game types only coarsely.
Bears on the paper: supports. Whether a model is responding to the payoff
structure at all is a comprehension question, and the answer varies by model.

**`herr2024are`**
Source: https://arxiv.org/abs/2407.04467 (arXiv only, v3 Oct 2024; no venue found) `[abstract]`
Shows: in Stag Hunt and Prisoner's Dilemma, GPT-3.5, GPT-4-Turbo, GPT-4o and
Llama-3-8B show positional, payoff and behavioural biases, and lose 16% to 34% of
performance when the game configuration is misaligned with those biases; GPT-4o
drops the most.
Bears on the paper: supports. A bias that masquerades as strategy is the reason a
screen is needed, and the fact that the strongest general model suffers the most
is a warning against assuming tier predicts rule use.

**`piatti2024cooperate`**
Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/ca9567d8ef6b2ea2da0d7eed57b933ee-Abstract-Conference.html; arXiv 2404.16698v4 `[full text: Section 3.7, Figure 5]`
Shows: GovSim, a common-pool resource society of LLM agents; highest survival rate
below 54%. Section 3.7 builds four sub-skill tests of 150 procedurally generated
problems each (simulation dynamics, sustainable action, threshold under an
equal-harvest assumption, threshold from beliefs about others) and regresses
survival time on each: R^2 = 0.69, 0.92, 0.76 and 0.82.
Bears on the paper: closest prior battery. It is a subskill comprehension battery
for an LLM social dilemma with a predictive-validity check, but it reports no
reliability, no discriminant test against model size (it notes larger models do
better), and one environment only.

**`huang2026probing`**
Source: https://arxiv.org/abs/2606.04978 (arXiv only) `[full text: Appendix E.4]`
Shows: 28 LLMs play the St. Petersburg game; outcome-level resemblance to human
bids hides mechanism-level differences. Appendix E.4 excludes 20 base and
instruct models (OLMo-2 7B, DeepSeek LLM 67B, Gemma 2 27B, Gemma 3 1B, SmolLM2
1.7B, Mistral 7B, Mixtral 8x22B, Falcon3-7B, Llama 3 70B) "because they fail to
correctly compute the expected value", with no threshold, repetition count or
reliability reported.
Bears on the paper: direct precedent for a comprehension gate on LLMs in an
economic decision study, and a clean example of an unvalidated one.

**`cacioli2026screen`**
Source: https://arxiv.org/abs/2604.17714 (arXiv only) `[full text: Sections 1.2, 2.1, 3.4, 6.5]`
Shows: a two-stage protocol, Stage A validity screening then Stage B substantive
analysis, adapted from MMPI-3 and PAI validity scales and applied to LLM
confidence signals over 20 models and 524 items: 4 Invalid, 2 Indeterminate, 14
Valid. Invalid includes DeepSeek-R1, Gemini 3.1 Pro, Qwen 80B Think and Gemma 3
1B, so validity does not track size (Section 1.2 calls it a property of the
model, probe and task). Invalid metrics may be computed but must be flagged
(Section 2.1). Its own limitations say "Single administration. No test-retest
reliability. Prompt sensitivity was not tested" (Section 6.5).
Bears on the paper: closest prior protocol. It formalises screen-then-interpret
with flagged retention, and it is the counter-example to "any gate just recovers
scale". It validates the screen against a criterion but not its reliability,
which is RQ2.

---

## B. Construct validity and measurement theory for AI evaluation

**`raji2021ai`**
Source: https://datasets-benchmarks-proceedings.neurips.cc/paper/2021/hash/084b6fbb10729ed4da8c3d3f5a3ae7c9-Abstract-round2.html `[abstract]`
Shows: position paper arguing that "general" benchmarks are stand-ins for broad
anointed problems and that treating them as general measures of progress has
construct validity problems.
Bears on the paper: positions. A screen score read as "the model understands the
game" is a small instance of the same over-reading. Author order follows the
proceedings page (Raji, Denton, Bender, Hanna, Paullada), which differs from the
arXiv order.

**`bean2025measuring`**
Source: https://proceedings.neurips.cc/paper_files/paper/2025/hash/1967e0fc3aa6cbbace562f5cb8e3954e-Abstract-Datasets_and_Benchmarks_Track.html; arXiv 2511.04703 `[full text: Sections 2, 3.1 to 3.8, Appendix D]`
Shows: a systematic review of 445 LLM benchmarks by 29 reviewers; 78.2% define
the phenomenon, only 16.0% use uncertainty estimates or statistical tests, 53.4%
give construct validity evidence; Section 3.2 ("measure the phenomenon and only
the phenomenon") flags that 21.1% impose output formats that themselves affect
performance. Eight recommendations follow.
Bears on the paper: supports and frames the method. The paper's parse-health
baseline answers recommendation 3.2 and its per-domain analysis answers 3.1
(sub-components). Use it as the main citation for construct validity in LLM
benchmarks.

**`liao2023rethinking`**
Source: https://arxiv.org/abs/2306.03100 (arXiv only, v4 Jan 2025) `[abstract]`
Shows: argues that evaluation of general-purpose LLMs should assess whether
human needs in downstream use cases are met (the socio-technical gap), drawing on
realism lessons from social science, HCI and XAI.
Bears on the paper: positions, weakly. Useful only for the realism-versus-cost
trade-off; it does not address screens or construct validity quantitatively.

**`wallach2025position`**
Source: https://proceedings.mlr.press/v267/wallach25a.html `[abstract]`
Shows: argues that evaluating generative AI is a social science measurement
challenge and proposes a four-level framework (background concept, systematised
concept, measurement instrument, instance-level measurements) with lenses for
interrogating validity.
Bears on the paper: frames the paper. "Comprehension" is the systematised
concept, the 20-probe battery the instrument, the admit/refuse verdict the
measurement; RQ1 and RQ3 test whether the instrument matches the concept. There
is an earlier arXiv version (2411.10939) without "Position:"; cite the ICML one.

**`salaudeen2025measurement`**
Source: https://arxiv.org/abs/2505.10573 (arXiv only, v4 Jun 2025) `[full text: Section 1 Table 2, Section 4 Figure 2, Section 5.3]`
Shows: a validity-centred framework that maps evaluative claims to the evidence
they need, using content, criterion, construct, external and consequential
validity; within construct validity it separates structural, convergent and
discriminant validity (Section 5.3), and works through GPQA as a case study.
Bears on the paper: supports the RQ structure directly: RQ1 is structural
validity, RQ3 discriminant, RQ4 external, RQ5 convergent or criterion.

**`jacobs2021measurement`**
Source: https://doi.org/10.1145/3442188.3445901; arXiv 1912.05511 `[abstract]`
Shows: brings measurement modelling from the social sciences to fairness,
arguing that unobservable constructs are operationalised through measurement
models whose assumptions create mismatches, and gives fairness-oriented
definitions of construct reliability and construct validity.
Bears on the paper: supports. It is the standard ML citation for treating
reliability and validity as separate properties of an operationalisation, which
RQ2 and RQ3 test separately.

**`campbell1959convergent`** `[metadata only]`
Source: https://doi.org/10.1037/h0046016; PubMed 13634291
Shows: not read. Title, authors, journal, volume, issue and pages confirmed at
Crossref and PubMed; the article is paywalled and no index carries an abstract.
Bears on the paper: the origin of the convergent and discriminant validity terms
and the multitrait-multimethod matrix. Before citing it for anything more than
the terms, open the article. Excluded from the "read" count.

---

## C. State tracking and world-state reasoning

**`kim2023entity`**
Source: https://aclanthology.org/2023.acl-long.213/ `[full text: Sections 3.2, 4.1, 4.4, Figure 2]`
Shows: a boxes task (7 boxes, 12 operations per scenario) that asks for an
entity's final state; against a strong random baseline, only GPT-3.5
text-davinci-003 tracks entities non-trivially, staying above baseline after up
to 7 operations on a box, while GPT-3 davinci and Flan-T5 mostly repeat the
initial state; the authors attribute the difference to code pretraining.
Bears on the paper: supports RQ1 and RQ4. State tracking is a separable ability
that varies sharply between models of similar scale, and it degrades with the
number of state-changing operations, which is what the F2 and F3 histories add.

**`li2021implicit`**
Source: https://aclanthology.org/2021.acl-long.143/ `[abstract]`
Shows: BART and T5 contain contextual representations that behave like dynamic
models of entities and situations, support a linear readout of each entity's
current state, and can be edited with predictable effects on generation.
Bears on the paper: positions. It gives the representational side of "state
reconstruction"; the paper measures the behavioural side only.

**`toshniwal2022chess`**
Source: https://doi.org/10.1609/aaai.v36i10.21390 `[abstract]`
Shows: chess notation allows direct probing of board state; transformers trained
on move sequences learn to track pieces and predict legal moves, and success
depends on attention over the whole game history.
Bears on the paper: supports the design choice of games with a reference engine,
where the true state is known exactly and a state-reconstruction answer can be
scored without ambiguity.

**`merrill2024illusion`**
Source: https://proceedings.mlr.press/v235/merrill24a.html `[abstract]`
Shows: state-space models, like transformers, cannot express computation outside
TC^0, so they cannot solve state-tracking problems such as permutation
composition, tracking chess moves, evaluating code or tracking entities in a
long narrative.
Bears on the paper: positions. A formal reason why state tracking over many turns
is expected to fail, independently of scale, which is why it is a plausible
screen domain and a plausible candidate to fractionate.

**`laban2026llms`**
Source: https://proceedings.iclr.cc/paper_files/paper/2026/hash/59f6421e64707225fdf5b28840679a07-Abstract-Conference.html; arXiv 2505.06120 `[abstract]`
Shows: top open and closed models lose on average 39% of performance when a
fully specified task is revealed over several turns instead of one; the loss is
mostly a large rise in unreliability, as models commit early to assumptions and
do not recover.
Bears on the paper: supports the concern that single-turn probe accuracy may not
carry over to multi-turn play, and that unreliability rather than capability is
the main multi-turn failure, which bears on RQ2 and RQ5.

---

## D. Scale and tier confounds

**`ruan2024observational`**
Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/1cded4f97cf5f01a284c574110b7e3b9-Abstract-Conference.html; arXiv 2405.10938 `[full text: Sections 3.2, 3.3, Figures 2 and 3]`
Shows: benchmark performance across public models lies in a low-dimensional
capability space; the top 3 principal components explain about 97% of variance,
PC-1 alone nearly 80%, PC-1 acts as general capability, and within a family PC-1
is linear in log training FLOPs (R^2 > 0.9). Section 3.2 reports 77 models from
21 families; the abstract says about 100 models.
Bears on the paper: supports RQ3's prior. Any accuracy-scored battery is expected
to be mostly a read of general capability, so a screen that reproduces the tier
ordering is the default outcome, not a discovery.

**`ren2024safetywashing`**
Source: https://proceedings.neurips.cc/paper_files/paper/2024/hash/7ebcdd0de471c027e67a11959c666d74-Abstract-Datasets_and_Benchmarks_Track.html; arXiv 2407.21792 `[full text: Sections 3, 4.1, 7]`
Shows: a capabilities score is the first principal component of 12 capability
benchmarks over 27 base and 26 chat models; many safety benchmarks correlate
highly with it (for example TruthfulQA MC1 81.2%, ETHICS 82.2%) while others do
not (sycophancy, jailbreak robustness); Section 7 recommends that new safety
evaluations report their correlation with capabilities.
Bears on the paper: the closest method analogue for RQ3. The name baseline is a
zero-cost version of their capabilities-correlation test, and their
recommendation is the criterion the paper applies to its own screen.

**`ilic2024evidence`**
Source: https://doi.org/10.1016/j.intell.2024.101858; arXiv 2310.11616 `[abstract]`
Shows: over 591 LLMs and 12 tests, scores show a positive manifold and a general
ability factor plus a knowledge and reading-writing group factor; parameter
count correlates positively with both factor scores, with diminishing returns.
Bears on the paper: supports RQ3's prior from psychometrics. Do not cite the
"65.6% of variance" or "r = 0.70 with parameters" figures that circulate in the
repository's novelty survey; they are not in the abstract and were not verified.

**`burnell2023revealing`**
Source: https://arxiv.org/abs/2306.10062 (arXiv only) `[abstract]`
Shows: factor analysis of 29 models on 27 tasks finds three factors (reasoning,
comprehension, core language modelling) rather than one, with different
relations to size and instruction tuning.
Bears on the paper: contrasts with Ruan and Ilic. Capabilities need not collapse
to one factor, so fractionation of comprehension (RQ1) is plausible a priori.

---

## E. Reliability, nondeterminism and sensitivity of LLM evaluation

**`atil2025nondeterminism`**
Source: https://aclanthology.org/2025.eval4nlp-1.12/; arXiv 2408.04667v5 `[full text: abstract, Sections 1 to 3 (PDF pp. 135-137); Sections 5.1, 7.1, 8 (arXiv v5)]`
Shows: five API-hosted models (GPT-3.5 Turbo, GPT-4o, Llama-3-70B and 8B,
Mixtral-8x7B) at temperature 0 over eight MMLU and BBH tasks and 10 runs vary in
accuracy by up to 15%, with up to 70% between best and worst possible
performance; parsed-answer agreement (TARa) is much higher than raw-string
agreement (TARr); locally run models are stable; the authors recommend reporting
maximum and minimum across runs.
Bears on the paper: supports RQ2 directly. Temperature 0 on a hosted endpoint is
not a deterministic instrument, so screen verdicts must be re-administered
before they can be trusted. The published title adds "System ... in Hosted
Environments".

**`ouyang2025empirical`**
Source: https://doi.org/10.1145/3697010 `[abstract]`
Shows: over 829 code generation problems, ChatGPT is highly non-deterministic;
at the default setting, 75.76%, 51.00% and 47.56% of tasks (CodeContests, APPS,
HumanEval) give no two identical test outputs across requests, and temperature 0
reduces but does not remove it.
Bears on the paper: supports RQ2 as an independent replication of temperature 0
nondeterminism in a different task domain.

**`sclar2024quantifying`**
Source: https://proceedings.iclr.cc/paper_files/paper/2024/hash/6c0e99d736da621403018ca7b32b1a4d-Abstract-Conference.html `[abstract]`
Shows: meaning-preserving changes to prompt formatting shift few-shot accuracy
by up to 76 points (LLaMA-2-13B); sensitivity survives scale, more shots and
instruction tuning, and format performance correlates only weakly between
models, so single-format comparisons are questionable.
Bears on the paper: supports treating parse health and format as a confound to
report (plan strand c), and warns that the screen's single fixed format may
favour some routes.

**`mizrahi2024state`**
Source: https://aclanthology.org/2024.tacl-1.52/ `[abstract]`
Shows: across 6.5M instances, 20 LLMs and 39 tasks, different paraphrases of
the instruction change both absolute scores and model rankings; proposes
multi-prompt metrics.
Bears on the paper: supports. The screen is administered with one wording per
probe, so a ranking-level claim (RQ3, RQ4) inherits this sensitivity; name it as
a limitation or add paraphrases.

**`zheng2024large`**
Source: https://openreview.net/forum?id=shr9PXz7T0; arXiv 2309.03882 (venue "ICLR 2024 Spotlight" read on the arXiv page) `[abstract]`
Shows: over 20 LLMs and 3 benchmarks, models prefer particular option IDs in
multiple-choice questions ("selection bias"), mainly from token bias toward
letters such as A to D; proposes the PriDe debiasing method.
Bears on the paper: supports checking whether any lettered probes in the screen
are answered by position rather than content.

**`zhu2026clean`**
Source: https://arxiv.org/abs/2609.04198 (arXiv only, 3 Sep 2026) `[abstract]`
Shows: a preregistered audit of 52,988 requests to black-box LLM judges on shared
endpoints; same-window repeat rankings reach Spearman 0.400 against a
preregistered 0.90, next-day identical replays 0.78 against 0.99; the authors
conclude that on a shared endpoint "a model name is not a frozen instrument" and
that an evaluation must measure its instrument before freezing a gate on it.
Bears on the paper: supports RQ2 and the preregistration design, and is very
recent; check for a revised version before submission.

---

## Closest prior work to "validating the comprehension screen itself"

No work found validates a comprehension screen for LLM agents as an instrument.
The closest is **Cacioli (2026)**, which builds a two-stage screen-then-interpret
protocol for LLM confidence signals, applies it to 20 frontier models, shows
that its validity classes do not follow model size, and prescribes flagged
retention of invalid rows; but its screen is a validity check on confidence, not
a comprehension battery, it runs once with no test-retest reliability and no
prompt sensitivity (its own Section 6.5), it is not tested against a zero-cost
baseline such as the model name, and it has no task-family transfer. **Piatti et
al. (2024, GovSim)** is the closest comprehension battery: four subskill tests in
an LLM social dilemma whose accuracies predict survival (R^2 from 0.69 to 0.92),
which is a convergent-validity result; but it reports no reliability, no
discriminant test against size, although it states that larger models do better,
and a single environment, so neither fractionation as a measurement problem nor
transfer is examined. **Huang et al. (2026)** use a comprehension gate, an
expected-value check, to exclude 20 models from an LLM economic decision study,
with no threshold, repetition or validation reported. **Ren et al. (2024)**
supply the method the paper borrows for RQ3, testing whether a benchmark carries
information beyond general capability, but on safety benchmarks rather than on
an admission gate. **Berinsky et al. (2014)** is the human precedent: it
validates attention screeners themselves, finds that several items beat one and
that passing correlates with respondent characteristics. That is the human
version of the name confound. What none of these does, and what the ARR paper
adds, is to administer the same comprehension screen repeatedly (reliability),
break it down by domain (fractionation), test it against the model name and
single-probe baselines (discriminant validity) and carry a state-tracking score
across three game families (transfer). **Akata et al. (2025)** motivates the
gap: humans are gated on comprehension, models are not.

Five closest, in order: `cacioli2026screen`, `piatti2024cooperate`,
`huang2026probing`, `ren2024safetywashing`, `berinsky2014separating`.

---

## Not found / not verified

Items the brief named or the repository's novelty survey suggested that are not
in the .bib, or are in it with a caveat:

1. **"Horton 2023" as a single-author paper.** Not confirmed. The arXiv record
   2301.07543 now lists Horton, Filippas and Manning, and Crossref lists the same
   three authors for NBER Working Paper w31122 (DOI 10.3386/w31122, April 2023).
   The .bib uses the published EC '24 version (`filippas2024large`). If the
   paper must cite "Horton (2023)" as single author, check the NBER PDF first.
2. **"Duan et al." as a work separate from GTBench.** No separate work was
   identified; the brief's Duan et al. is taken to be GTBench (Jinhao Duan first
   author), `duan2024gtbench`.
3. **LLM game-theory papers that gate models on rule understanding.** Akata et
   al. 2025, GTBench and GAMA-Bench were checked in full text and none gates
   models on a comprehension test. The only model gate found is Huang et al.
   2026. The brief's premise that these papers gate on rule understanding is not
   supported.
4. **Campbell and Fiske (1959).** Metadata verified, text not read (paywalled, no
   abstract indexed). In the .bib with a flag, excluded from the read count.
5. **Cronbach and Meehl (1955), "Construct validity in psychological tests",
   Psychological Bulletin 52(4):281-302, DOI 10.1037/h0040957.** Metadata
   verified at Crossref and PubMed 13245896; text not read. Not in the .bib.
6. **Ilic and Gignac numbers "65.6% of variance" and "r = 0.70 with parameter
   count"** (from `docs/novelty-survey-2026-09-10.md`, second-hand through a
   thesis). Not in the abstract, not verified. The work itself is verified.
7. **Kearns (2025) MSc thesis, arXiv 2602.15532.** Not checked in this pass.
8. **Liao, Taori, Raji and Schmidt (2021), "Are we learning yet?"** Not checked;
   the brief's "Liao & Xiao" was resolved to Liao and Xiao (2023).
9. **Li (2026) arXiv 2605.11599, Pechon-Elkins and Chun (2026) arXiv
   2603.10029, Fernandez Domingos and Han (2026) arXiv 2607.26034.** Leads from
   the novelty survey, not checked in this pass. The last one is the source
   study of the concurrent submission; handle its citation under the ARR
   anonymity rules (plan section 6).
10. **Verified but left out to keep the list under 40:** Hauser, Ellsworth and
    Gonzalez (2018) "Are Manipulation Checks Necessary?", Frontiers in
    Psychology 9:998, DOI 10.3389/fpsyg.2018.00998 (full text read; argues
    checks act as interventions and suggests separate pilot validation);
    Andreoni (1995) "Cooperation in Public-Goods Experiments: Kindness or
    Confusion?", AER 85(4):891-904 (abstract read on EconPapers; about half of
    cooperation comes from subjects who understand free-riding); Song, Wang, Li
    and Lin (2025) "The Good, The Bad, and The Greedy", NAACL 2025,
    2025.naacl-long.211 (abstract read); Li et al. (2023) "Emergent World
    Representations", ICLR 2023, arXiv 2210.13382 (abstract read). Any of these
    can be added on request.

**Venue notes to respect when citing.** GTBench and GAMA-Bench have published
titles that differ from their arXiv titles. The Atil et al. published title
differs from the arXiv title. Raji et al. author order differs between the
proceedings and arXiv; the .bib follows the proceedings. Herr et al., Salaudeen
et al., Liao and Xiao, Burnell et al., Huang et al. 2026, Cacioli 2026 and Zhu
and Zhang 2026 are arXiv only as of 2026-09-19; re-check for published versions
before submission.
