# Gate 06: cross-paper overlap and AAMAS internal consistency

Audit date 2026-09-18. Read-only. Line numbers refer to these exact file states
(sha1), which were unchanged at the start and end of the audit:

| file | sha1 | mtime |
|---|---|---|
| `paper/main.tex` (AAMAS) | `8fee4127...` | 2026-09-14 16:00 |
| `paper/supplementary.tex` (AAMAS supp) | `71041cc9...` | 2026-09-14 16:36 |
| `paper/acl/main.tex` (ARR) | `e4c5dd40...` | 2026-09-12 12:15 |

Another agent is editing `paper/main.tex`; re-grep the quoted text before
acting on a line number.

Authorities: `docs/behaviour-paper-claim-map-2026-09-12.md` (A1-A12, B1-B24,
sec. 4 and 5), `docs/acl-audit-paper-plan.md` (sec. 6 and 7).

---

## 0. Venue rules that decide what overlap is allowed

**ARR** (fetched 2026-09-18 from https://aclrollingreview.org/cfp):

- "ARR will not consider any paper that is under review in a journal or another
  conference at the time of submission" and "we will not consider any paper that
  overlaps significantly in content or results with papers that will be (or have
  been) published elsewhere, without exception."
- "any concurrently submitted papers on a related topic with an overlapping set
  of authors must cite each other and discuss the differences in the related
  work section." "Anonymized versions of such papers should be uploaded as
  supplementary material (in the 'data' field)." Recommended form:
  `Anonymous (2026). Paper title. Under review.`
- Self-citations in third person; no names or affiliations. Limitations section
  mandatory (desk reject otherwise).

**AAMAS 2027** (https://warwick.ac.uk/fac/sci/dcs/aamas2027/guidelines-and-policies/instructions/,
https://warwick.ac.uk/fac/sci/dcs/aamas2027/calls/call-for-main-track/):

- "Authors must not submit substantially similar work to AAMAS 2027 and another
  archival venue at the same time."
- "reviewers are not required to look at supplementary material" and "Any
  information that is essential for understanding or evaluating your paper must
  be included in the paper itself."
- Paper deadline 8 Oct 2026, notification 21 Dec 2026. "Papers that are not
  selected will be automatically considered for publication in the Findings of
  AAMAS 2027 ... unless the authors opt out."

**ARR dates** (https://aclrollingreview.org/dates): October 2026 cycle submits
12 Oct 2026 and ends 20 Dec 2026, i.e. fully inside the AAMAS review window.

**What this permits.** Two different papers sharing a roster, game and
collection protocol are allowed. What is not allowed: (a) either paper failing to
cite the other when both are under review; (b) the ARR paper overlapping
"significantly in content or results" with the AAMAS paper, which will almost
certainly be published (accepted, or Findings by default unless opted out);
(c) the pair reading as one contribution sliced in two. Because the ARR paper
must upload the anonymised AAMAS PDF, ARR reviewers will see any verbatim
overlap with the AAMAS supplement directly.

---

## 1. A1-A12 in the AAMAS paper (main.tex + supplementary.tex)

Verdict key: **OK-screen** = methods-level entry condition, permitted;
**OK-supp** = permitted by claim map sec. 2/5 as supplement record;
**VIOLATION** = appears as a result or argument the claim map forbids.

| Claim | Where | Quoted text | Verdict |
|---|---|---|---|
| A1 per-domain fractionation | main: absent as result. main 645-646 | "Two admitted routes pass one screening domain at the minimum attainable score" | OK-screen (bare fact). |
| | supp 464-496 `tab:admission-v6` | three gated columns, state recon. 20.0-100.0 | OK-supp (screen record). Caption supp 473-474 says "The per-domain results in full are retained in this supplement" but only 3 of 6 domains are shown anywhere in the supplement: false pointer. |
| | supp 580-588 (open-weight battery, a different instrument) | "What the battery shows is a gap between stating a rule and using it ... everything that requires carrying the state forward or doing the arithmetic collapses" | Borderline: the A1 finding type stated as a result on a second instrument. Low risk, but it is the ARR paper's thesis. |
| A2 format compliance | main 635 | "An action that parses is not evidence that the agent followed the race." | OK-screen (motivation). |
| | supp 593-596 | "strict format compliance, 32.1\% ... far below the semantic parse rate, which was 100\%" | Borderline result on the open-weight battery; overlaps ARR sec:res-format thesis. |
| A3 expected payoff | supp 472-473 | "no route reaches 75\% on it" | OK-supp. |
| A4 rho = 0.87 | absent from main and supp | none | OK. But residue: supp 898-902 "the risk response, the quantity the admission analysis in \S\ref{app:admission-v6} actually uses ... the between-route spread of $7.0$ to $100.0$ points that the same analysis reports". app:admission-v6 no longer uses or reports either; dangling pointer to the deleted A4 analysis. |
| A5 name coincidence | main 646-647 | "on this nine-route roster the admitted/refused split coincides with endpoint tier naming" | OK-screen. |
| | supp 534-549 | full permutation argument, "$1/126$, or $0.008$" | OK-supp by claim map, but near-verbatim with ARR 590-618 (see sec. 3). |
| A6 test-retest | supp 509-513 | "two of the four routes present in both moved, each by two answers in fifteen" | OK-supp; verbatim with ARR 495-498. |
| A7 token-cap verdict move | supp 514-518 | "Claude Sonnet 5 was refused under the 128-token cap and admitted under 256" | **VIOLATION**. Claim map A7: "Evaluation paper. Behaviour paper does not need it and cannot afford it." Also verbatim with ARR 505-508. |
| A8 threshold band | main: **absent** | none | Gap. Claim map sec. 4 (attack two): "a one-clause version of it belongs in the main paper rather than only in the supplement." |
| | supp 498-507 | "the admitted set is the same at all 676 threshold combinations" | OK-supp; verbatim with ARR 512-519. |
| A9 one binding condition | main: absent; supp 501-503 | "two of the three conditions can be loosened to zero without admitting anyone new" | OK-supp; claim map wanted it in the main methods too. |
| A10 no partition (37.6 vs 38.2) | absent from main and supp | none | OK. |
| A11 flattest = weakest state tracking | main 720-722 | "against a 93.0-point spread in risk response across the nine routes" | Partial leak: 93.0 = 100.0 (Opus) minus 7.0 (GPT-5.4 mini, A11 number); refused-route risk responses are tabulated nowhere, and main 663-664 says a refused route "appears below only where it is named as refused". |
| | supp 858-860, 3143-3147 | "GPT-5.4 mini and GPT-5.4 nano ... are also the two whose \Unsafe{} rate barely moves with the stated risk"; "the route that responds to the name instead of to the state is a route the comprehension gate refused ... its risk response of $9.7$ pp" | Partial A11 leak in supplement (link between refusal and flat response). |
| A12 seat asymmetry | main: absent (seat counterbalancing in methods, main 766) | "The route's seat is counterbalanced five and five inside each cell" | OK. |
| | supp 3117-3162 | "So the screen separates the routes on the whole-race asymmetry and does not separate them at the opening, and the design protection the paper actually relies on is the seat counterbalancing rather than the screen." | **VIOLATION of the boundary in spirit**: claim map sec. 5 lets the GPT-5.4 nano seat result sit in the supplement, but this paragraph evaluates whether the screen separates routes, which is ARR sec:res-seat (ARR 748-775) almost point for point. |
| (screen as result, main body) | main 957-961 | "Both refused routes are routes the quality-control screen of \S\ref{sec:design-screen} refused, at 75.0\% and 51.7\% overall probe accuracy. That is why the screen is reported in the body: without it, these two rows would read as the finding that small models match human strategic diversity." | **VIOLATION**. Screen scores are printed in Results and the admitted/refused split is used to reinterpret a behavioural result; claim map sec. 4 last sentence: no analysis "treats admitted against refused as a measured contrast". "small models" is the tier reading the paper itself says it cannot separate. Keep the B19 disclosure; drop the scores and the "That is why ..." sentence. |
| (screen as contribution) | main 481-485 | "Third, an auditable race mechanism ... entered through a comprehension screen that scores answers to frozen questions against ground truth and never reads a race" | Overlap risk: the AAMAS paper claims the screen as a contribution while ARR 884 says the two papers "do not share a contribution" and the screen is ARR's object of study. Recast contribution 3 as the engine-resolved mechanism only. |
| (screen named as comprehension) | main 482; supp 714, 1164, 3144 | "comprehension screen"; "the comprehension gate admits"; "had already passed the comprehension gate" | Conflicts with main 647-648 "never as a capability axis" and claim map sec. 4 "must not be presented as a measurement of understanding". |
| (screen score comparison) | supp 1156-1158 | "even though the two routes differ by provider and sit six points apart on the admission audit" | VIOLATION of supp 423-424 ("Nothing in this supplement ranks routes by their probe scores"). The number also matches no column: Gemini 3 Flash vs Claude Sonnet 5 differ by 8.3 overall, 6.7 state recon., 20.0 terminal. |

Main-methods completeness against claim map sec. 4 (the eight required sentences):
present: never reads a race (644), 20 probes x 3 (636-637), thresholds (638-639),
5 of 9 entered (639), refused still played (639-642), tier coincidence (646-647),
not a capability axis (647-648). Missing: the 676-combination band, "only one
condition refuses any route", "loosening it by one answer in fifteen would admit
a sixth", "fixed before the campaign", and the names of all nine routes (the two
Gemini Flash-Lite refused routes are never named in main.tex, so a main-only
reader cannot check the tier-naming sentence).

---

## 2. B1-B24 in the ARR paper (paper/acl/main.tex)

B5-B11, B13-B22, B24 do not appear. The ones that do:

| Claim | Where | Quoted text | Verdict |
|---|---|---|---|
| B1 risk response | ARR 682-684, 697-699 | "each endpoint's behavioural sensitivity is summarised by how far its play shifts between the lowest and the highest stated danger"; "37.6 points against 38.2" | Used as the outcome variable of a convergent-validity analysis, not as a behavioural finding. Permitted by the plan (RQ5), but 38.2 is GPT-5.5's B1 value and must cite the concurrent paper at this point; it does not. |
| B2 step policy | ARR 717-719 | "the one endpoint whose play is a switch rather than a gradient" | Descriptive mention; cite concurrent paper. |
| B4 clean baseline | ARR 441-443, 681-682 | "5{,}022 decisions across 270 games with no parse failure"; "thirty games and 558 decisions each" | Collection facts reused as A2 evidence. Acceptable; cite. |
| B23 presentation-bound | ARR 831-837 | "swapping which of two opaque response codes means safe moves the endpoint's play by between 24.0 and 57.3 percentage points ... A relabelling that a correct model of the game must ignore rewrites the action sequence." | Different endpoint from B23 (refused open-weight pilot, not Gemini/Sonnet), but the same manipulation and near-verbatim conclusion as AAMAS supp 1218-1219. Medium overlap. Worse: this pilot's fixed-state replays and eight retellings are what AAMAS main 620-628 describes as its own design, and they exist only in the ARR paper (see sec. 5, item 5.2). |

---

## 3. Verbatim and near-verbatim text shared across the two papers

Method: 12-word shingle match after stripping LaTeX and comments (exact), plus
sentence-level difflib ratio >= 0.55 (near). Script:
scratchpad `overlap.py`. All hits are AAMAS supplement vs ARR, except two.

Exact runs of 12+ words:

| AAMAS | ARR | words | shared text |
|---|---|---|---|
| main 635 | 130-131 | 12 | "an action that parses is not evidence that the agent followed the" |
| main 1017-1018; supp 3363-3364 | 690-691 | 12 | "nine commercial endpoints are not a sample from a population of models" |
| supp 498-500 | 512-514 | 24 | "Each declared value carries 5.0 points of headroom, and the three ranges hold together rather than one at a time: the admitted set is" |
| supp 512-514 | 496-498 | 24 | "moved, each by two answers in fifteen on state reconstruction, and the other two did not move at all. No verdict changed. Across the" |
| supp 515-517 | 506-507 | 23 | "was refused under the 128-token cap and admitted under 256. A truncated answer is scored wrong and a low cap can truncate, so" |
| supp 517-518 | 507-508 | 13 | "is not a clean repeat and we do not present it as one" |
| supp 527-528 | 531-532 | 17 | "pass margin can therefore be narrower than the movement the same instrument shows between two identical administrations" |
| supp 535-536 | 590-591 | 13 | "small when mini, nano or lite appears as a whole word of its" |
| supp 538-540 | 594-595 | 21 | "Enumerating all $C(9,4)=126$ ways of relabelling which four names carry a size word gives an exact one-sided permutation" |

Near-verbatim (ratio in brackets):

| AAMAS | ARR | text (AAMAS form) |
|---|---|---|
| supp 426-427 | 268-269 (0.75) | "The screen is a frozen bank of 20 probes over six domains, asked three times each, giving 60 scored answers per route" |
| supp 448-451 | 276-279 (0.86, 0.79) | "All nine answered the same probe bank, whose SHA-256 hash begins 2953fb47, and read the same rules context, whose hash begins b80981a5; the analysis script fails closed and refuses to tabulate a route whose hashes differ." |
| supp 484-493 | 399-408 (0.74) | the admission table rows repeat the same nine route strings and the same overall / state recon. / terminal values |
| supp 499-501 | 513-515 (0.88) | "the admitted set is the same at all 676 threshold combinations inside them" |
| supp 504-507 | 516-519 (0.59) | one achievable step down to 73.3\% "admits ... and nobody else" |
| supp 520-527 | 521-531 (0.58, 0.73) | "clears state reconstruction at exactly that score ... One answer otherwise would put GPT-5.5 at 73.3\%, which is the score on which Gemini 3.1 Flash-Lite is refused." |
| supp 534-538 | 590-593 (0.79, 0.74) | "Calling a route small when mini, nano or lite appears as a whole word of its name splits these nine endpoints into exactly the two sets the screen produced" |
| supp 538-540 | 594-595 (0.96) | "... gives an exact one-sided permutation $p$ of $1/126$, or $0.008$." |
| supp 540-546 | 613-618 (0.57, 0.83) | "on this roster the screen draws no line that the route name does not already draw, so no claim resting on the admitted against refused split is separable from a claim about scale" |
| supp 604-606 | 473-474 (0.65) | "a separate instrument ... its scores are never compared/pooled with those" |
| supp 1218-1219 | 834-835 (0.84) | "A relabelling that a correct model of the game should ignore therefore changes the action sequence." |
| supp 3156-3157 | 770-771 (0.85) | "Four of the nine routes open \Unsafe{} from both seats in every race and can show no gap at all." |
| supp 3157-3159 | 772-774 (paraphrase) | "the screen separates the routes on the whole-race asymmetry and does not separate them at the opening" |

**Reading.** Every ARR result in sec:res-stability (ARR 478-568) and the name
paragraph of sec:res-name (ARR 590-596) is already in the AAMAS supplement, in
mostly the same sentences; sec:res-seat (ARR 725-804) duplicates supp 3117-3162.
The ARR source comment at ARR line 18 itself says "Do not paste prose from the concurrent
manuscript into this file". Rewrite on one side: the cleanest cut is to shrink
supp `app:admission-v6` to the verdict table plus the claim-map screen sentences,
and move A6, A7, A8-detail, the margin-zero paragraph, the permutation test and
the seat-screen argument out of the AAMAS supplement.

**Cross-paper numeric mismatch on the same object** (GPT-5.4 nano whole-race seat
gap, nine-route neutral baseline): supp 3126-3129 "$+19.5$ pp $[+12.9,+25.7]$ ...
falls to $+0.5$ pp $[-0.4,+1.4]$; over the five admitted routes it is $-0.5$ pp
$[-1.3,+0.4]$" with pooled "$+2.6$ pp" (supp 3032-3033); ARR 756-760 and Table
tab:seat 785-792 give $+19.4$, $+0.2$, $-0.6$, pooled $+2.3$. Supp 3146 itself
switches to "$19.4$ pp" decision-weighted. A reader holding both PDFs sees two
sets of numbers for one analysis. Pick one weighting and use it in both.

---

## 4. Mutual citation and anonymity

| Check | Result |
|---|---|
| AAMAS cites ARR | **No.** No text in main.tex or supplementary.tex, and `paper/references.bib` has no Anonymous / under-review entry. `docs/acl-audit-paper-plan.md` sec. 6 item 2: "It is not optional and it must land before the AAMAS submission goes out." |
| ARR cites AAMAS | Yes: ARR 879-889, `\citep{anonymous-concurrent}`. |
| ARR citation placed in Related Work (ARR rule) | **No.** It is in sec. 8 "Contribution Boundary" (ARR 846). The Related Work section is a placeholder: ARR 199-200 "\pending{this section. ...}". |
| ARR citation title matches AAMAS | **No.** `paper/acl/custom.bib` 15-16 "More Than the Risk: Frontier LLM Safety Policies Depend on the Rival" vs main.tex 81 "More Than the Risk: Frontier LLM Behaviour in AI Development Races Depends on the Rival". |
| ARR citation form | `author = {Anonymous}`, year 2026, note "Under review. Concurrent submission by the same authors." Matches the ARR-recommended form and reveals no identity. Fine once the title is fixed. |
| ARR boundary text accurate | **No.** ARR 887-889: "the per-domain results, the threshold sweep, the reliability analysis and the comparison against cheaper signals appear only here." The AAMAS supplement contains the gated-domain table (supp 464-496), the threshold sweep (498-507), the reliability analysis and token-cap move (509-518), and the name baseline with the same permutation test (534-549). ARR 884 "they do not share a contribution" is contradicted by AAMAS contribution 3 (main 481-485). A false boundary statement in the paragraph that exists to prove non-overlap is the single most dangerous sentence in the pair. |
| Anonymity of a future AAMAS->ARR citation | Use `Anonymous. 2026. <exact ARR title>. Under review.` in third person; do not write "our" / "we previously". Naming the other venue is not identifying. |
| Incidental identifiers (outside the brief, noted only) | supp 1098 "run 1373757 ... run 1504455" (provider run IDs), supp 890 "Kaggle archive", supp 2923 "Model Proxy identity". Not names, but check with the anonymity gate. |

---

## 5. AAMAS internal consistency

### 5.1 Pointers from main.tex to the supplement

main.tex has no "Supplement Section X", "Table S#" or "Figure S#" pointers; all
13 pointers are generic "(see the) supplementary material". All `\ref` targets
in main.tex resolve to labels inside main.tex.

| main line | promise | status in supplementary.tex |
|---|---|---|
| 486-487 | exploratory multiplayer results | present, supp 2230-2308, 2899-3115 |
| 556-558 | "the rule, its parameters and its results are in the supplementary material" | **Rule and parameters absent.** The group-count payoff rule is never stated; supp 2193 "Eq.~(2)" and 2263 "Eq.~(3)--(4)" cite equations that do not exist in the supplement (it has no numbered equations). |
| 612-618 | paired decision-card diagnostic on the published-weight route | **No result anywhere.** supp 563-564 and 631-634 describe it ("30 races and 558 decisions in each of the two presentations") but report no contrast. |
| 620-628 | "eight narrative skins", "a paired first-round comparison", "a \emph{fixed-state replay}", "a \emph{live trajectory}" | **Absent from AAMAS supp**, which has two skins and live play only (supp 1090-1234). Eight retellings and fixed-state replays exist only in the ARR paper (ARR 823-837). main 1031-1033 then states a limitation about fixed-state vs live effect sizes the AAMAS paper never reports. |
| 651 | full screening results and robustness checks | present, supp 417-556 |
| 665-666 | "full model tables, predictive diagnostics and robustness checks" | present |
| 677-678 | "race-weighted reanalysis preserves the non-increasing response on every admitted route" | Partial: supp tab:baseline-race-sensitivity (673-692) gives only the 0.1-minus-0.9 contrast, which cannot show non-increase across three levels. |
| 720-722 | "a 93.0-point spread in risk response across the nine routes" | **Unsupported.** No nine-route risk-response table exists; supp 898-902 cites it to an analysis that is no longer in app:admission-v6. |
| 729 | 8.3--9.5 point code-swap effect | present, supp tab:context-mapping-paired 1190-1191 |
| 776 | contract provenance | present, supp 1413-1425 |
| 841, 887 | full EGT fit, five-route fit | present, supp 912-1006 |
| 922 | human refit | present, supp 1236-1273 |
| 935 | Hill q=1 and Hamming checks | present, supp tab:diversity-confirmatory 2785-2840 |
| 985-987 | "Raw player-level \Unsafe{} rates, plotted per risk condition in the supplementary material, spread across nearly the whole available range for humans at every risk level while each checkpoint occupies a narrow band" | **No such plot.** supp fig:unsafe-rate-by-group (2411-2443) is one mean per population with a CI, not per risk condition and not a distribution, and on the seven-checkpoint pilot roster (no GPT-5.4 or GPT-5.5). The main-text claim has no support. |
| 990-992 | embedding | present, supp 2310-2375 (pilot roster) |
| 995-997 | classifier and feature analyses | present, correctly scoped |

Claims in main.tex whose only support lives in the supplement (AAMAS reviewers
need not read it): the risk-at-Safe contrast 10.4-69.9 (main 780-781; the main
figure 3b shows only the rival contrast; the table is supp 1531-1577), the
race-weighted robustness (677), the contract heterogeneity (773-776) and the
group-size grid (1036). These are stated with numbers in main and are
acceptable. The unsupported ones are 722 and 985-987 above.

Reverse stale pointers (supplement claiming main-text content that is gone):
supp 1341 "the persona and measured risk results in the main paper"; supp
2136-2137 "the part of this analysis that the main paper's diversity section
points at: human winners play \Unsafe{} 16.0 percentage points more often";
supp 2187-2188 "referenced throughout the two-player strategic diversity results
in the main paper"; supp 2195-2196 "pooled aggregate rates already reported in
the main text". None of these is in main.tex.

### 5.2 Number and object checks (same object verified before calling a conflict)

1. **Self-play vs fixed-safe cell count.** main 815-817: "The fixed-safe and
   self-play designs differ in thirteen of fifteen cells; Claude Opus 5 reverses
   the direction at risks $0.6$ and $0.9$". Same claim at supp 1650-1653. Object:
   the five-route scripted campaign, Always Safe column vs self-play column of
   supp tab:scripted-opponent. Row supp 1562: Opus 5 at 0.9 is AS 0.0, self-play
   0.0, a tie, not a reversal; at 0.6 it is 1.1 vs 0.0, a reversal. So self-play
   is higher in 13 cells, lower in 1, equal in 1: the designs differ in 14, and
   Opus reverses at 0.6 only.
2. **EGT RMSE.** main 874-876: at $\beta=0.01$, $\mu=0.05$ "the model predicts
   87.3\%, 63.9\% and 38.0\% ... within 10.3 percentage points of Claude Sonnet 5
   and 15.1 of Gemini 3 Flash". From the profiles printed in main.tex itself (Sonnet
   89.2/48.4/32.3 at 836-837; Gemini 98.9/73.1/60.2) the RMSE is 9.6 and 15.4,
   which is exactly what supp tab:egt-five-route-fit prints for the same cell
   (supp 999, 1002; supp 983-984 confirms the same cell). supp 969-972 repeats
   10.28 / 15.13 and largest misses 17.70 / 21.14, whereas the printed profiles
   give 15.5 and 22.2. Same routes, same cell, two numbers. Recompute from the
   artifact and use one.
3. **"Highest" Always-Safe rate.** supp 1617-1618: "GPT-5.4 has the highest
   \emph{Always Safe} rates, $36.6\%$, $32.3\%$ and $25.8\%$". False at risk 0.1
   in the five-route table: GPT-5.5 72.0, Claude Opus 5 63.4 (supp 1560, 1568).
   Carried over from the three-route subset. main 807-809 correctly avoids "highest".
4. **Seed handling.** main 589-591: "the SDK stripped the requested seed on the
   Google routes and forwarded it, application unconfirmed, on the OpenAI and
   Anthropic ones." main 1038-1039: "one route's SDK stripped the requested seed
   and another forwarded it unconfirmed" (stale two-route wording). supp 879-881:
   "the SDK strips the seed request before it reaches the provider, which every
   run manifest records as \texttt{not\_applied\_sdk\_stripped\_for\_route}"
   contradicts both main 590-591 and supp 695-696 for the Claude manifest.
5. **Figure 1 panel pointers.** Caption main 507-515 puts the stage-payoff matrix
   in panel (c) and the risk/horizon mechanics in (b). Text main 530 cites the
   matrix as "Figure~\ref{fig:mechanism}b"; main 537 cites the hidden horizon as
   panel a; main 542 cites the prize and setback asymmetry as panel a. All three
   are off by one panel relative to the caption and the Description (517-523).
6. **Rival-vs-risk overlap sentence.** main 782-784: "Both contrasts use the same
   ten matched blocks, so the absolute comparison is the load-bearing result, but
   the two effect families overlap once the conditional rivals are included." The
   rival contrast is Always Unsafe minus Always Safe (main 737-738, supp 1538);
   conditional rivals do not enter it, so "once the conditional rivals are
   included" has no referent. The overlap actually comes from Opus 5 (+69.9 risk)
   and GPT-5.5 (+48.3 risk) against their own rival cells (supp 1560-1570).
   The same sentence is in claim map B6.
7. **Table arithmetic readers will try.** supp tab:scripted-opponent prints rates
   and paired contrasts side by side; AU minus AS from the printed rates gives
   75.3 for Gemini 3 Flash at 0.1 against a printed "+73.5", and 63.4 vs "+69.9"
   for Opus 5 risk-at-Safe. Presumably race-weighted contrasts over
   decision-weighted rates; the caption (1538-1542) does not say so.
8. Verified consistent (no action): 54.8 average; 186/372 Opus decisions; 38.7 to
   40.3 replication; 5,580 decisions; 18/27, 13/15 and 21/27 diversity counts
   against supp 2704-2706 and 2806-2837; dyads 48+52+68 = 168; effective counts
   18.9/19.7/19.3; 12 vs 13 trajectories and 0.41 vs 0.27 Hamming; EGT five-route
   ranges 5.1-15.4 and 32.9-36.6.
9. Worth a look: in supp tab:diversity-confirmatory the Gemini 3 Flash (admitted)
   and Gemini 3.1 Flash Lite (refused) rows at risk 0.1 (supp 2807, 2812) are
   identical in every value and interval, and identical to the pilot Claude
   Sonnet 5 row at 0.9 (supp 2763). Possible for near-deterministic routes, but
   check against the artifact for a copy error.
10. supp 1837 "Of the seven tested checkpoints, five have such a sweep" and 1880-1881
    "available for five of the seven tested checkpoints" vs tab:persona-effect
    (1861-1867), which lists seven checkpoints with sweeps.

### 5.3 Research questions and contributions

| Item | Abstract (294-323) | Introduction | Results | Conclusion (999-1041) |
|---|---|---|---|---|
| RQ Risk | "risk responses share a qualitative direction but differ in shape" (302-304) | 455-456 | sec:results-risk 668-729 | 1002-1004, answered |
| RQ Rivalry | "strongly opponent-dependent ... can overlap the effect of changing the stated risk" (304-311) | 456-457 | sec:results-scripted 750-826 | 1004-1009, answered |
| RQ Humans and theory: theory | "four graded profiles closer in response shape to the near-neutral regime" (313-315) | 458-459; 466-467 **"Neither external reference shows that shape: the evolutionary model predicts a switch where the agents deliver a gradient"** | 867-876 "As $\beta$ falls towards neutrality ... the model's own risk response becomes graded, which is the shape the routes show" | **Not concluded.** Only the limitation at 1026-1029. |
| RQ Humans and theory: humans | "13 of 15 admitted cells fall below every draw" (317-318) | 467-469 | sec:results-diversity 896-997 | **Not concluded.** 1011-1015 discusses what a human reference is, not what was found. Claim map sec. 3 calls this "the finding a reader should leave with". |
| Contribution 1, scripted-rival design and self-play critique | implied (309-311) | 471-478 | 813-820 | 1004-1009 partly; the self-play critique is not restated |
| Contribution 2, trajectory-level human comparison | 316-318 | 478-481 | delivered | not restated |
| Contribution 3, auditable mechanism + comprehension screen | absent from abstract | 481-485 | methods only, by design | absent |
| Promised: "We therefore test wording, answer order, arithmetic disclosure, narrative framing and opaque response codes separately" | absent | 421-424 | only narrative x code (726-729). Wording and answer order: no gameplay result anywhere. Arithmetic disclosure: no result anywhere (decision card, 5.1). | absent |
| Delivered, not promised | Gemini replication (718-724); "the response to the stated risk survives the change of design" (799-801) | | | Fine, robustness only |

Main contradiction: intro 466-467 vs results 867-876 and abstract 313-315 are
about the same object (the shape of the EGT model's risk response): the intro says
the theory predicts a switch, the results say it predicts a switch only at strong
selection and a gradient near neutrality. Fix the intro to "at its usual
selection strength".

Tone mismatch on the headline: results opener main 656-657 "their behaviour is
especially sensitive to the rival" vs main 786-787 "The contrast is therefore a
context-sensitive descriptive comparison, not a universal ordering of rival
effects over risk." Same object (rival stance vs risk contrast). Drop "especially".

### 5.4 Terminology (counts: main / supp; prose only)

| Concept | Variants (main / supp) | Recommended |
|---|---|---|
| Hosted model unit | route 110 / 338; endpoint 11 / 29; checkpoint 8 / 78; model 33 / 87; agent 30 / 13; delegate 4 / 0 | **route** for measured units; "agent" only for the conceptual player. Supp 365-373 defines endpoint = route with a verdict and checkpoint = dated version, but main 384-386 defines route as an endpoint, main 305 says "five frontier-model endpoints", main 803 "varies markedly across checkpoints", main 949 "two checkpoints are the only exceptions" about routes. |
| The screen | quality-control screen 3 / 4; task-validity screen 2 / 0; comprehension screen 1 (482, split line) / 0; comprehension gate 0 / 1; gate 0 / 9; gate-admitted 0 / 2; admission audit 0 / 1; battery 2 / 19; bare screen 17 / 17 | **task-validity screen** (matches "never as a capability axis"); retire "comprehension screen/gate" and "admission audit". |
| Which routes are "audited" | "audited routes" 0 / 8, meaning 2 routes (supp 961 "both audited routes", 1233), 5 routes (supp 1346 "Five audited routes"), 9 routes (supp 1052, 1889, 2695, 2787 "all nine audited routes") | **screened** (9), **admitted** (5); never "audited". |
| Opponent design | scripted rival 2 / 5; scripted-rival 5 / 6; scripted-opponent 0 / 13 in prose (campaign, figure titles); fixed opponent 1 / 0; fixed-safe 2 / 3; scripted code 1 / 0 | **scripted rival**; keep file names but change prose "scripted-opponent campaign" (supp 1431, 1532). |
| Self-play | self-play 6 / 27; mirror-match 5 / 1 | pick one; main uses both within one paragraph (813-819). Recommend **self-play** with one gloss. |
| Outcome | Unsafe rate 10 / 44; Unsafe play 9 / 18; risk response 6 / 7 and risk-response 4 / 5 (the 0.1-minus-0.9 drop); "risk-response drops" (697) | **Unsafe rate** for the level; **risk response** for the 0.1 minus 0.9 difference, defined once in 668-679 (currently never defined in main). |
| Risk treatment | stated risk 6 / 2; stated danger 5 / 0; maximum private risk 3 / 12; catastrophe/catastrophic 3 / 0; risk level/condition 15 / 43 | **stated maximum risk** ($p_r^{\max}$); "catastrophe" only in the framing. |
| Conditional strategies | "Conditional Unsafe (CAS)" main 570; "conditional Always Unsafe (CAS)" supp 759; "the conditional rule that opens \Unsafe{}" supp 742-743 | **Conditional Unsafe (CU)**, or keep CAS but name it consistently. |
| Game | race 38 / 153; game 34 / 40 (ARR uses "game") | race for one play-through, game for the mechanism; fine as is. |

### 5.5 Scope claims that go past the evidence

| Where | Quote | Problem |
|---|---|---|
| main 899-900 | "Twenty people asked to run this race do not behave alike; twenty copies of a frontier route very nearly do." | Admitted routes reach 11-14 distinct trajectories of 20 (supp 2810, 2820-2821, 2831-2832), and two refused frontier-vendor routes reach 16-20. "very nearly do" holds for Opus 5 and Sonnet 5 only. |
| main 901-903 | "so a route can match the human mean without resembling the human population at all" | "at all" is stronger than 13 of 15 cells below a null. |
| main 896 (heading); main 312 | "Frontier agents are behaviourally concentrated"; "Frontier agents also show a distinctive relationship" | Unscoped; say "admitted routes". |
| main 831-833 | "their response shapes differ because of an assumption inside the theory" | Causal "because": the sweep shows the model's shape depends on $\beta$, not why the routes differ. |
| main 656-657 | "especially sensitive to the rival" | Comparative claim the section disowns (786-787). |
| main 959-961 | "these two rows would read as the finding that small models match human strategic diversity" | Model-scale inference from 2 routes, and a tier reading the paper says it cannot separate. |
| main 1002 | "These agents incorporate both the stated risk and the rival's stance." | "incorporate" implies an internal process; "respond to" is what was measured. |
| main 295-296 | "Large language models are increasingly used to advise and act on consequential decisions." | Uncited empirical claim in the first sentence. |
| supp 1663-1665 | "what self-play hides is the \emph{risk response} itself ... and it hides it on every route we have run this design on" | Opus 5 self-play reveals, not hides, its step; stated as universal over 5. |
| rho = 0.87 | absent from main and supp | No causal-language risk on the AAMAS side. |

---

## 6. Findings ranked by severity

**Desk-reject risk**

1. AAMAS paper does not cite the ARR paper (main.tex, supplementary.tex, references.bib: no entry). ARR requires both to cite each other in related work; the ARR citation sits outside Related Work (ARR 846, Related Work is `\pending`, ARR 199-200) and carries the wrong AAMAS title (custom.bib 15-16 vs main 81).
2. ARR boundary statement is false: ARR 887-889 "the per-domain results, the threshold sweep, the reliability analysis and the comparison against cheaper signals appear only here" vs AAMAS supp 464-549; ARR 884 "they do not share a contribution" vs main 481-485 contribution 3. Nine 12-to-24-word verbatim runs (sec. 3) make ARR sec:res-stability and the name paragraph of sec:res-name duplicates of AAMAS supp app:admission-v6, and ARR sec:res-seat duplicates supp 3117-3162. With AAMAS papers going to Findings by default, ARR's "overlaps significantly ... with papers that will be published" rule bites.
3. Claim-map violations in the AAMAS paper: main 957-961 prints screen scores (75.0\%, 51.7\%) in Results and uses the refused/admitted split to reinterpret a behavioural result; supp 514-518 carries A7 (forbidden outright); supp 3143-3159 evaluates whether the screen separates routes (the ARR result); supp 1158 compares routes on screen score.
4. AAMAS main 620-628 and 1031-1033 describe eight narrative skins, first-round paired comparisons and fixed-state replays that exist only in the ARR paper (ARR 823-837), and 612-618 a decision-card diagnostic reported nowhere. An AAMAS reviewer reading the ARR PDF finds the AAMAS design section describing ARR results.

**Reviewer-visible error**

5. main 815-817 / supp 1650-1653: "differ in thirteen of fifteen cells; Claude Opus 5 reverses ... at 0.6 and 0.9"; at 0.9 both rates are 0.0 (supp 1562). Correct: differ in 14, reversal at 0.6 only.
6. main 874-876 RMSE "10.3 ... 15.1" vs 9.6 and 15.4 recomputed from main's own printed profiles and printed in supp 999/1002 for the same cell.
7. Intro main 466-467 "the evolutionary model predicts a switch where the agents deliver a gradient" contradicts results 871-873 and abstract 313-315.
8. Unsupported or broken main-to-supp pointers: 720-722 (93.0-point nine-route spread; no table, and it rests on A11 numbers), 985-987 (per-risk player-level plot does not exist), 556-558 (N-player rule and parameters absent; supp 2193, 2263 cite nonexistent Eq. (2)-(4)).
9. main 782-784 "the two effect families overlap once the conditional rivals are included" has no referent; main 656-657 "especially sensitive to the rival" contradicts 786-787; supp 1617-1618 "GPT-5.4 has the highest Always Safe rates" is false at risk 0.1.
10. Conclusion (999-1041) never states the theory or the human-diversity finding; the promise at main 421-424 to test wording, answer order and arithmetic disclosure is not delivered.

**Polish** (report only): seed-handling contradictions (main 589-591 vs 1038-1039 vs supp 879-881); Figure 1 panel letters off by one (main 530, 537, 542); cross-paper seat-gap numbers (supp 3126-3133 vs ARR 756-792); terminology table 5.4; supp stale "main paper" pointers (1341, 2136-2137, 2187-2188, 2195-2196); supp 473-474 "per-domain results in full" not present; supp 1837 vs tab:persona-effect checkpoint count; identical diversity rows to verify (supp 2807, 2812, 2763); refused Gemini Flash-Lite routes never named in main.
