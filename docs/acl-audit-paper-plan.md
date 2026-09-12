# The ARR evaluation paper: plan, claim inventory, and what is missing

Written 2026-09-12, alongside the creation of `paper/acl/`. This is the planning
document for the **second** paper, the one about the comprehension screen. The
behaviour paper stays in `paper/` as the AAMAS submission. The combined
manuscript that held both stories is frozen at the git tag
`combined-manuscript-2026-09-12`.

Companion document: `docs/behaviour-paper-claim-map-2026-09-12.md`. Its section 2
lists the audit claims A1 to A12 as the ones that cannot move into the behaviour
paper, and its section 5 carries the original scope note for this paper. Every
number that document attributes to an artifact has been re-read from that
artifact in this session. Section 4 below records the two places where the
artifacts and the claim map disagree. **Where they disagree, the artifact wins.**

The one sentence a reader of this document needs first: **this paper is not
submittable today, and the reason is not writing, it is missing data.**

---

## 1. Thesis

A comprehension screen is a measurement instrument that selects which models are
allowed into a behavioural analysis, and therefore it decides what the
behavioural analysis is about. Nobody validates it. This paper treats the screen
as an instrument and asks what it measures, and the answer on the evidence
currently in this repository is uncomfortable: comprehension fractionates so
sharply that an aggregate screen score is carried by one subtask, and the
screen's admit or refuse verdict is reproduced exactly by a regular expression
over the model name.

That refutation is the paper's premise, not its limitation. A screen earns its
cost only by beating the cheap signals it is confounded with, and the paper's
contribution is the criterion for that, plus the evidence of whether this
battery meets it.

---

## 2. Research questions

**RQ1. Is comprehension one thing?** Does an aggregate screen score describe a
single ability, or does it average subtasks that behave independently across
endpoints?

**RQ2. Is the screen reliable?** Does the same battery, administered twice under
an identical protocol, give the same verdict? Does the verdict survive a band of
thresholds? Which conditions actually bind?

**RQ3. Does the screen carry information a cheaper signal does not?** The
discriminant-validity question, against three baselines: the model name, a
single probe, and reported parse health. This is the question the paper exists
to ask.

**RQ4. Does a screen score transfer across task families?** If state
reconstruction on a two player race predicts state reconstruction on an iterated
social dilemma and on a public goods game, the screen measures a construct. If
it does not, screens must be rebuilt per environment, which is also a finding.

**RQ5. Does admission predict downstream behaviour?** Convergent validity,
reported as descriptive, over nine endpoints, and reported together with the
fact that it is confounded with RQ3's name baseline by construction.

RQ1, RQ2 and RQ5 can be answered from artifacts on disk today. RQ3 can be
*asked* from artifacts on disk today and can only be *answered* with new data.
RQ4 has no data at all.

---

## 3. Claim inventory: what this repository already supports

Every row was recomputed in this session directly from the artifact named. The
"status" column is about evidence, not about writing.

### 3.1 Fractionation (RQ1)

| # | Claim | Artifact | Number | Status |
|---|---|---|---|---|
| E1 | Reading the stage payoff matrix is solved by every endpoint | `results/frontier/admission_campaign_v6/**/admission.json` | 9 of 9 routes at accuracy 1.000 in domain `stage_payoff` | verified |
| E2 | Rule recall is nearly solved | same | 8 of 9 at 1.000; GPT-5.4 nano at 0.750 | verified |
| E3 | State reconstruction is the domain that separates endpoints | same | spans 1.000 (Claude Sonnet 5, GPT-5.4) down to 0.200 (GPT-5.4 nano); intermediate values 0.933, 0.933, 0.800, 0.733, 0.667, 0.600 | verified |
| E4 | No endpoint computes a closed form expected payoff | same | maximum 0.500 (Gemini 3 Flash, GPT-5.5); five routes at 0.000 (Claude Sonnet 5, GPT-5.4, GPT-5.4 mini, GPT-5.4 nano, Gemini 3.5 Flash Lite) | verified |
| E5 | The aggregate hides this | same | overall accuracy spans 0.517 to 0.933 while `stage_payoff` has zero variance across all nine | verified |

### 3.2 Format compliance (RQ1, instrument hygiene)

| # | Claim | Artifact | Number | Status |
|---|---|---|---|---|
| E6 | Gameplay parsing is clean across the whole comprehension range | `results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.csv`, column `parse_failures` | 9 routes, 30 races and 558 decisions each, 5,022 decisions total, 0 parse failures | verified |
| E7 | Probe answers are almost but not entirely well formed | `results/frontier/admission_campaign_v6/**/raw_responses.jsonl`, field `semantic_valid` | 540 probe rows, 535 valid, 5 invalid: 1 on Claude Opus 5 and 4 on Claude Sonnet 5, all five in the `expected_payoff` domain; 1 row additionally carries a transport error | **corrects the claim map**, see section 4 |

### 3.3 Reliability and sensitivity (RQ2)

| # | Claim | Artifact | Number | Status |
|---|---|---|---|---|
| E8 | The same battery administered twice moves | `results/frontier/admission_campaign_v5/` against `_v6/` | state reconstruction moved +13.3 points, two answers in fifteen, on 2 of the 4 routes both campaigns cover (Claude Sonnet 5 0.867 to 1.000, Gemini 3.1 Flash Lite 0.600 to 0.733); the other two unchanged; no verdict changed | verified |
| E9 | One verdict has moved across protocol versions | `results/frontier/admission_campaign/` (v1), `_v5`, `_v6` | Claude Sonnet 5 overall accuracy 0.767 refused, then 0.817 admitted, then 0.850 admitted | verified |
| E10 | Only one of the three gate conditions binds | `admission_campaign_v6/**/admission.json` | state reconstruction at 0.75 alone reproduces the admitted set exactly: the five at or above 0.75 are exactly the five admitted | verified |
| E11 | One achievable step down admits a sixth route and nobody else | same | Gemini 3.1 Flash Lite sits at 0.733, one answer in fifteen below the threshold; the next route below is at 0.667 | verified |
| E12 | The verdict survives a threshold band | `scripts/verify_manuscript_claims.py` | identical admitted set at all 676 combinations within five points on each condition | **not re-run this session**, re-run before use |

### 3.4 The name baseline (RQ3)

| # | Claim | Artifact | Number | Status |
|---|---|---|---|---|
| E13 | The battery's verdict is reproduced exactly by a size word in the route name | the nine verdicts in `admission_campaign_v6`; figure `figures/paper/audit_reads_the_name.pdf`, generator `scripts/figures/fig_audit_reads_the_name.py` | the four refused routes are exactly the four whose names contain `mini`, `nano` or `lite`: GPT-5.4 mini, GPT-5.4 nano, Gemini 3.1 Flash Lite, Gemini 3.5 Flash Lite. With 5 admitted of 9 there are C(9,4) = 126 labellings, so the exact one sided permutation probability of perfect agreement is 1/126 = 0.0079 | verified, including the arithmetic |
| E14 | A single probe baseline and a parse health baseline | `raw_responses.jsonl` holds the probe level data | not computed | **NEEDS ANALYSIS** |

### 3.5 Convergent validity (RQ5)

| # | Claim | Artifact | Number | Status |
|---|---|---|---|---|
| E15 | Screen state reconstruction is associated with the behavioural risk response | `results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.json`, `correlations` block | Spearman rho = 0.8740, exact permutation p = 0.00377 over all 9! = 362,880 relabellings, n = 9 | verified |
| E16 | Stronger without the one step policy route | same, `excluding_boundary_case` | rho = 0.9222, p = 0.00268, n = 8 | verified |
| E17 | Weaker against the aggregate score than against the binding domain | same | overall accuracy: rho = 0.7699, p = 0.01927, n = 9 | verified |
| E18 | The screen does not partition behaviour | `audit_versus_behaviour.json`, `admission_contrast` | refused maximum 37.63 points against admitted minimum 38.17: the sets touch. Refused mean 19.35 against admitted mean 54.84 | verified |
| E19 | The flattest responders are the weakest at state reconstruction | `audit_versus_behaviour.csv` | GPT-5.4 mini 6.99 points at 0.667; GPT-5.4 nano 9.68 points at 0.200 | verified |

### 3.6 A validity failure the screen catches

| # | Claim | Artifact | Number | Status |
|---|---|---|---|---|
| E20 | One refused route's two byte identical seats disagree | `results/derived/seat_confound/seat_gap.csv`, `seat_gap_leave_one_out.csv`, `seat_confound.json` | GPT-5.4 nano: seat 1 rate 0.491 against seat 2 rate 0.685, gap 19.35 points [12.09, 25.88] over 30 races and 558 decisions; pooled all route gap 2.31 points [1.09, 3.59]; removing this one route drops the pooled gap to 0.18 points [-0.53, 0.90] | verified |
| E21 | The same route's opening move gap | `seat_gap_opening.csv` | **not supported per route.** The opening file carries only pooled rows: all routes 9.26 points [5.56, 12.96]. There is no per route opening row anywhere in `results/derived/seat_confound/` | **corrects the claim map**, see section 4 |

---

## 4. Where the artifacts disagree with the claim map

Two places. Neither is a wrong analysis; both are statements that are stronger
than the files behind them. They are recorded here so the ARR paper does not
inherit them, and they should be corrected in the claim map too.

**4.1 Probe level parse health.** Claim A2 in
`docs/behaviour-paper-claim-map-2026-09-12.md` states "Parser satisfied on all
540 probe outputs and all 5,022 baseline decisions". The gameplay half is exact.
The probe half is not: 535 of 540 probe rows carry `semantic_valid: true`, and
five do not. All five are in the `expected_payoff` domain, one on Claude Opus 5
and four on Claude Sonnet 5, and one row additionally records a transport error.
The underlying claim survives, because the parser is still satisfied across the
whole comprehension range and still fails to track it, but it must be written on
the real denominator and must name the domain. Note also that
`expected_payoff` is the domain the battery itself flags diagnostic only
(`"expected_payoff_is_diagnostic_only": true` in every `admission.json`), which
is the natural way to report it.

**4.2 The per route opening move seat gap.** Claim A12 states GPT-5.4 nano's
seats differ by "30.0 points on the opening move". `seat_gap_opening.csv`
contains four rows, all pooled over every route: an overall row at 9.26 points
and three per risk rows. No per route opening row exists in the seat confound
tree. Either `scripts/analyze_seat_confound.py` is extended to emit per route
opening rows and the number is recomputed, or the claim is dropped. The route
level gap of 19.35 points [12.09, 25.88] is fully supported and is enough on its
own.

---

## 5. What must be collected before this paper can be submitted

In the order of how much each one buys. Items 1 and 2 are the difference between
a paper and a pile of offcuts. Neither is analysis. Both are new runs.

### 5.1 Routes where the name and the comprehension score disagree

**Why.** E13 is the paper's central problem. As long as the admitted set is
exactly the set without a size word in the name, no reviewer has to believe the
battery measures anything. The battery earns its cost the moment one endpoint
breaks the tie: a large tier route that fails state reconstruction, or a small
tier route that passes it.

**What to collect.** The frozen v6 bank, unchanged, administered to a widened
roster chosen to stress the tie rather than to extend it. Candidates: open
weight models whose names carry no tier word at all, older generations of the
same families, reasoning configurations of an endpoint already in the roster
where the route string is constant and the behaviour is not, and small models
with a reputation for strong state tracking.

**The honest branch.** This may fail. If no endpoint breaks the tie, the paper
is not dead but it is a different paper: "comprehension screening in current
commercial families is not separable from model tier", which is publishable and
is a warning to everyone using a screen, but it cannot claim the battery
measures comprehension. **Decide the framing after this run, not before.**

**Status.** Not started. No roster fixed, no quota plan.

### 5.2 Two further task families

**Why.** One game cannot support a claim about comprehension screening, because
state reconstruction on one state structure is a task and not a construct. The
transfer question is what separates the two.

**What to collect.** A frozen probe bank for each of:

- **F2, an iterated social dilemma.** The state a route must track is the
  opponent's history. Probes mirror the six v6 domains: rule recall, stage
  payoff, state reconstruction from a short history, one state transition,
  terminal scoring, expected payoff.
- **F3, a public goods or common pool game.** The state a route must track is an
  aggregate over all players, which is a different kind of memory from a history
  of one opponent. Same six domains.

Both administered to the same roster as F1, three repetitions per probe, same
scoring rule, bank hash recorded, so that a per route per domain score is
comparable across families.

**What exists.** F1 only. An N player variant of the race exists in the engine
under `ai_race/`, and a public goods game appears in the cited prior work, but
**no frozen probe bank exists for either family and no endpoint has been
administered one.** Searched: `results/frontier/`, `results/derived/`, `docs/`.

**Status.** Not started. This is the single largest item of work in the plan.

### 5.3 Cheaper screen baselines

**Why.** The paper must state what a sixty call battery buys over each cheap
alternative. E10 already shows that two of the three gate conditions refuse
nobody, so the honest starting position is that most of the battery is redundant
on this sample.

**What to compute.** Against the same nine verdicts: the route name regex
(done, E13), a single state reconstruction probe drawn at random with the
distribution over draws reported, each domain alone, and reported parse health.
`raw_responses.jsonl` holds everything needed.

**Status.** Data on disk, analysis not written. This is the one cheap item.

### 5.4 Smaller items

- Re-run `scripts/verify_manuscript_claims.py` for the 676 combination threshold
  sweep before E12 is written down.
- Either emit per route opening seat rows or drop E21 (section 4.2).
- A per domain by per route accuracy figure. Plotting only.
- A fresh, verified `paper/acl/custom.bib`. Do not reuse `paper/references.bib`:
  it belongs to the concurrent submission and is being edited by another
  workflow. Every entry verified at the source, title, authors, venue, year and
  DOI, per the evidence verification gate.

---

## 6. ARR rules that govern this split

Fetched from the ACL Rolling Review call for papers on 2026-09-12. Re-check
before submission, because ARR revises this text between cycles.

**Dual submission.** ARR will not consider a paper that is under review at a
journal or another conference at the time of submission, and papers must not
have significant content overlap with work published or accepted elsewhere.
Violations are desk rejected. Two *different* papers from the same authors are
permitted. The rule bites on **content overlap**, not on the number of
submissions, which is exactly why section 7 below matters.

**Concurrent related work.** Concurrently submitted papers on a related topic
with an overlapping author set **must cite each other and discuss the
differences in the related work section**, and anonymised versions should be
uploaded as supplementary material. The recommended citation form is
`Anonymous (2026). Paper title. Under review.`

**Thin slicing.** ARR states that the mutual citation requirement exists to
discourage thinly sliced contributions. A submission judged to be a slice may be
desk rejected.

**Limitations.** A Limitations section is mandatory. A paper without one is desk
rejected. It sits after the conclusion and before the references and does not
count against the page limit.

**Length.** Long paper: 8 pages of content, unlimited references. Short paper: 4
pages. Unlimited space after the conclusion for limitations and ethics.

**Anonymity.** Two way anonymisation. No author names or affiliations. Self
citations phrased in the third person: "Smith previously showed", never "we
previously showed". Since 15 February 2024 there is no anonymity period
restricting non anonymous preprints during review, but anonymous submissions are
still favoured for awards and for borderline decisions.

### What this means concretely, in both directions

1. **This paper cites the behaviour paper**, anonymously, in the related work
   section, with one sentence on what it claims and one sentence on the
   difference: that the screen appears there as a declared entry condition in
   the methods and that no result there rests on a screen score.
2. **The behaviour paper cites this one**, the same way, in its related work.
   That edit belongs to the workflow that owns `paper/main.tex`. It is not
   optional and it must land before the AAMAS submission goes out.
3. **Both anonymised PDFs go up as supplementary material** on the other's
   submission.
4. **Neither paper may be the same paper.** See section 7.
5. Check the AAMAS side for the mirror image of the ARR rule. AAMAS has its own
   concurrent submission policy and the obligation is symmetric.

---

## 7. The overlap risk, stated plainly

The material in section 3 was collected for one manuscript and was written up
inside it until 2026-09-12. If the ARR paper is that material re-sectioned, then
the two submissions share a dataset, a roster, a protocol and a results table,
and the honest description of the pair is one paper submitted twice. Under the
ARR rules in section 6 that is a desk reject, and it is the correct outcome.

The damage is not symmetric and it is not confined to the ARR paper.

- **The ARR paper is rejected**, and the audit material has nowhere to go,
  because it has now been cut out of the behaviour paper.
- **The behaviour paper is damaged too.** Its own related work section will, by
  then, cite this paper as concurrent work. A reviewer who reads both and sees
  one contribution split across two venues has a reason to distrust the pair,
  and AAMAS has its own overlap policy to apply.
- **The repository's credibility carries the cost**, and it is the same
  repository that has to submit the next thing.

The test to apply before submitting, in one sentence: **remove sections 5.1 and
5.2 from the ARR paper and ask whether what remains is a contribution.** Today
the answer is no, and that is why this paper is blocked on data and not on
writing. The boundary that makes it a real second paper is the transfer evidence
across task families and the name breaking roster. Neither exists yet.

A second test, for the behaviour paper's side of the boundary: the screen may
appear there only as a declared entry condition, in the eight methods sentences
specified in the claim map's section 4, with no analysis resting on screen
scores. If the behaviour paper starts arguing from the screen again, the
boundary has collapsed from the other direction.

---

## 8. Template provenance

Fetched 2026-09-12 from the official ACL style files distribution maintained by
the ACL at `https://github.com/acl-org/acl-style-files`, raw files from the
`master` branch. Nothing was hand written and nothing was modified.

| file | sha256 |
|---|---|
| `acl.sty` | `19dfeddc2c0e448f3926a0bef048a9db3f3611b46265b760caabd7ada4f361de` |
| `acl_natbib.bst` | `6fbb306202290f4b68e74ac1460a8b27398500cb6dfeb4492e74c457eae7cd1e` |
| `acl_latex.tex` | `339c9ee9705c1767d44ef24b85365c0bb8619ecc3fcf66172bf1356292389614` |
| `acl_lualatex.tex` | `0d9987ba833331a996f9abcd1a03eebba7d7ae331145793e7ddb6b5ae8db27b3` |
| `custom.bib` | `d76ccb30ddceb70c9e1ad0be3f43dfe301ad5765aab2c960fc830ef67bf8232a` |
| `anthology.bib.txt` | `2b78d2d9aeda62e14c4e46099e8225b5fc116387d8e0a54aad776485e249ceff` |

One note on the last file. Upstream ships it as `anthology.bib.txt`, and it is
**not** a bibliography: it is a short text explaining that the ACL Anthology
bibliography is now too large for a single file and is served as shards from
`https://aclanthology.org/anthology-1.bib` and `anthology-2.bib`. It is kept
under its upstream name so that nobody points `\bibliography` at it. Fetch the
shards only if the paper actually needs Anthology keys.

`paper/acl/main.tex` builds clean with MiKTeX pdflatex, eight pages in `[review]`
mode with line numbers, and `python scripts/anonymity_scan.py paper/acl/main.pdf`
reports clean. The scaffold's `\nocite{*}` has been removed; the document now
contains only the one citation whose entry is in `paper/acl/custom.bib`.

---

## 9. Moving the AAMAS paper to `paper/aamas/` later

**Not done, deliberately.** `paper/main.tex` and `paper/supplementary.tex` are
being rewritten by another workflow right now, and moving them would break the
build in the middle of that. The ARR paper went into `paper/acl/` instead, which
required no change to anything.

This section is the complete list of what a later move has to update. It was
produced by grepping the working tree on 2026-09-12 for `paper/main.tex`,
`paper/supplementary.tex`, `paper/ai_race_paper.pdf`,
`paper/ai_race_supplementary.pdf`, `paper/references.bib`,
`paper/submission_id.tex`, `paper/aamas.cls`, `ROOT / "paper"` and `PAPER_DIR`.
Line numbers are as of that date; re-grep before editing, because the rewrite
workflow is changing `main.tex` and `supplementary.tex` continuously.

**Read this first.** `figures/paper/` is a **different thing**: it is the figure
output directory, not the manuscript directory, and it does not move. Several
dozen grep hits are for that path. They are excluded below and must stay
excluded.

### 9.1 The one that will actually break the build

`paper/main.tex` and `paper/supplementary.tex` reach their figures with
`../figures/paper/...`, resolved from the document's own directory because
`scripts/build_publication.py` compiles with `cwd=document.parent`. Moving the
sources one level deeper makes every one of those resolve to
`paper/figures/paper/...`, which does not exist.

| file | count | change |
|---|---|---|
| `paper/main.tex` | 4 `\includegraphics` (lines 501, 829, 970, 1049) | `../figures/` becomes `../../figures/` |
| `paper/supplementary.tex` | 16 `\includegraphics` | same |

`\input{submission_id.tex}` and `\documentclass{aamas}` resolve from the
document's directory and survive the move, provided `submission_id.tex`,
`aamas.cls`, `ACM-Reference-Format.bst` and `references.bib` move with the
sources.

### 9.2 Python scripts

| file | lines | what is there |
|---|---|---|
| `scripts/build_publication.py` | 37 | `PAPER_DIR = ROOT / "paper"` |
| | 77, 82, 83 | `latex("paper/main.tex", jobname="ai_race_paper")` |
| | 80, 91 | `bib_dir=ROOT / "paper"` |
| | 88, 93, 94 | `latex("paper/supplementary.tex", ...)` |
| | 192 | `target = PAPER_DIR / product.name`, where the final PDFs are copied |
| | 5, 9, 115 | docstring and comment text naming `paper/` and `main.tex` |
| `scripts/build_release_manifest.py` | 71, 72, 73, 74, 75, 76 | `ROOT / "paper" / ...` for `main.tex`, `supplementary.tex`, `references.bib`, `submission_id.tex`, `ai_race_paper.pdf`, `ai_race_supplementary.pdf` |
| `scripts/build_submission_bundle.py` | 32 | `PAPER_DIR = ROOT / "paper"` |
| | 46, 47 | the two PDFs |
| | 57, 58 | `PAPER_DIR / "main.tex"`, `PAPER_DIR / "supplementary.tex"` |
| | 139, 142 | the README text written into the bundle |
| `scripts/check_publication.py` | 20 | `PAPER_DIR = ROOT / "paper"` |
| | 22, 96 | the two PDF names checked in `PAPER_DIR` |
| `scripts/check_figure_gallery.py` | 72 | `((ROOT / "paper", "main.tex"), (ROOT / "paper", "supplementary.tex"), ...)` |
| | 95 | walks `"paper"` recursively; it already walks `paper/acl/` today, which is harmless, but confirm after the move |
| `scripts/check_figure_geometry.py` | 36, 37 | `PAPER = ROOT / "paper"`, `SOURCES = ("main.tex", "supplementary.tex")` |
| | 148 | the error message naming `paper/` |
| `scripts/check_page_budget.py` | 33 | `PAPER = ROOT / "paper" / "main.pdf"` |
| | 77 | stem rewriting that maps `ai_race_paper` to `main` |
| `scripts/check_printed_type.py` | 27, 91 | `PAPER = ROOT / "paper"` and the two source names |
| `scripts/verify_manuscript_claims.py` | 1157, 1209 | `open("paper/supplementary.tex", ...)`, read relative to the working directory, so this one also depends on where it is run from |
| `scripts/build_supplementary_figures.py` | 5 | docstring only, no path resolution |
| `scripts/build_results_catalog.py` | 215 | prose in the generated catalog saying the manuscript PDFs live in `paper/` |

`scripts/check_anonymity_gate.py` and `scripts/anonymity_scan.py` take their
target as an argument and need no change.

### 9.3 Configuration

| file | lines | what is there |
|---|---|---|
| `.gitignore` | 114 to 121 | `paper/*.aux`, `*.bbl`, `*.blg`, `*.fdb_latexmk`, `*.fls`, `*.log`, `*.out`, `paper/main.pdf`. A single level glob, so it stops matching after the move |
| `.gitignore` | 122 to 129 | the equivalent `paper/acl/` rules, added 2026-09-12. These stay as they are |
| `.gitignore` | 147 | `paper/*.txt`, the pdflatex console dumps |
| `.gitignore` | 149 | `/paper/legacy/` |

No GitHub Actions workflows exist in this repository, so there is no CI path to
update.

### 9.4 Documentation to re-point

Prose only, no build impact, but a stale path in a document a fresh agent reads
first is how the last drift started.

`CLAUDE.md` lines 9, 51, 53, 54, 55. `README.md` line 16. `paper/README.md`
line 65. `paper/CITATION_CHANGELOG.md` lines 3, 9 to 17, 78, 81.
`results/artifacts/submission/README.txt` line 13.
`results/impact_upgrade/release_manifest.json` lines 82, 86, 90, 94, 98, 102.
`results/PAPER_FIGURE_CANDIDATES.md` lines 3, 68, 90.
`figures/gallery/SELECTED_FOR_PAPER.md` lines 12, 19.
`docs/` references in `handoff-caption-fixes-clustering.md` line 5,
`handoff-caption-fixes-egt.md` lines 4, 14, 105, 213,
`handoff-risk-versus-rival.md` lines 3, 4,
`novelty-survey-2026-09-10.md` lines 8, 9, 1333, 1338, 1340, 1486,
`submission_readiness_stress_test_2026-09-08.md` lines 18, 21, 35, 45, 47, 104,
109.

`paper/legacy/` was excluded from the sweep; it is historical and is gitignored.

### 9.5 Suggested order for the move

1. Wait for the AAMAS rewrite to land and the build to be green.
2. `git mv` the sources, `aamas.cls`, `ACM-Reference-Format.bst`,
   `references.bib` and `submission_id.tex` into `paper/aamas/`.
3. Fix the 20 `\includegraphics` prefixes (section 9.1) in the same commit, or
   the build fails and the failure looks like a missing figure rather than a
   move.
4. Update `.gitignore` before the first build, or the build artifacts land in
   `git status`.
5. Update the scripts in section 9.2, then run, in order:
   `scripts/build_publication.py`, `scripts/check_publication.py`,
   `scripts/check_page_budget.py`, `scripts/check_figure_geometry.py`,
   `scripts/check_printed_type.py`, `scripts/check_figure_gallery.py`,
   `scripts/verify_manuscript_claims.py`.
6. Update the documentation in section 9.4 last, in one commit, so the diff is
   readable.
