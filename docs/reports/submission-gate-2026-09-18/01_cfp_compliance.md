# Gate 01: AAMAS 2027 CFP compliance

Checked 2026-09-18, 15:52 UTC (22:52 Hanoi). This check was read only: nothing was built or edited.

**Sources.** The AAMAS 2027 call **is published**, so every rule below is a 2027 rule. No AAMAS 2026 rule is used anywhere in this report.

- The official site is `https://warwick.ac.uk/fac/sci/dcs/aamas2027/`. Two things confirm it is the official site. The official template's own `AAMAS_2027_sample.tex` names it. The OpenReview venue group `ifaamas.org/AAMAS/2027/Conference` belongs to the IFAAMAS namespace and gives the same contact, `aamas2027pcs@gmail.com`. `ifaamas.org` itself still links only AAMAS 2026.
- Pages read in full:
  - `.../calls/call-for-main-track/`
  - `.../guidelines-and-policies/instructions/`
  - `.../guidelines-and-policies/qa/`
  - `.../guidelines-and-policies/findings/`
  - `.../guidelines-and-policies/reciprocal-reviewer-policy/`
  - `.../guidelines-and-policies/reviewer-guidelines/`
  - `.../calls/call-for-blue-sky-ideas/`
  - `.../calls/call-for-aaai-fast-track/`
- Also read:
  - the template `.../aamas_2027_template.zip`, dated 2026-07-22
  - the live OpenReview submission form, `api2.openreview.net/invitations?id=ifaamas.org/AAMAS/2027/Conference/-/Submission`
- ARR rules come from `https://aclrollingreview.org/cfp` and `https://aclrollingreview.org/dates`.

**Manuscript state checked.**

| file | details |
|---|---|
| `paper/main.tex` | 1,060 lines, mtime 2026-09-14 16:00 |
| `paper/supplementary.tex` | 3,375 lines |
| `paper/submission_id.tex` | |
| `paper/aamas.cls` | |
| `paper/ai_race_paper.pdf` | 9 pages, built 2026-09-14 16:38, sha256 `ad867d95…` |
| `paper/ai_race_supplementary.pdf` | 31 pages |
| `results/artifacts/submission/supplementary.zip` | 2.54 MB |

Another agent is editing `main.tex` right now. Every PDF finding below is about the 2026-09-14 build, which may no longer match the source.

## Rule-by-rule table

In the table, the base URL `https://warwick.ac.uk/fac/sci/dcs/aamas2027/` is shortened to `W/`.

| rule | official source (URL + exact quote) | manuscript status | evidence | fix needed |
|---|---|---|---|---|
| **R1 Deadlines and timezone** | `W/calls/call-for-main-track/`: "Author registration on OpenReview: 17 Sep 2026 / Abstract submission: 1 Oct 2026 / Paper submission: 8 Oct 2026 … All deadlines are at the end of the specified day, Anywhere on Earth (UTC-12)." The OpenReview venue record agrees: "Abstract Registration: Oct 02 2026 11:59AM UTC-0, Submission Deadline: Oct 09 2026 11:59AM UTC-0" | UNKNOWN (these are process deadlines) | The repo records no date (CLAUDE.md:44-45) | Put the dates below into the repo's plan. |
| **R2 Every author has an OpenReview account** | `W/guidelines-and-policies/instructions/`: "every author on a paper must have an account on OpenReview; the deadline for this is two weeks before abstract submission." The form also makes every submitter confirm: "I confirm that all authors have up-to-date OpenReview profiles, including their full publication name, current position, institution-affiliated email address, and DBLP URL if available." | **UNKNOWN, and the deadline has already passed** | End of 17 Sep AoE was 18 Sep 11:59 UTC (18:59 Hanoi). The check ran about 4 hours after that. The camera-ready block lists 12 authors (main.tex:225-291). Profile lookup needs a login, so I could not check it. | Confirm today that all 12 authors have active profiles. For any author without one, create it now and write to aamas2027pcs@gmail.com. The email is my suggestion; the CFP does not describe a late process. |
| **R3 Submission system** | `W/guidelines-and-policies/instructions/`: "Please register your abstract and submit your paper on OpenReview here: https://openreview.net/group?id=ifaamas.org/AAMAS/2027/Conference" | UNKNOWN (not yet submitted) | n/a | Register the abstract on OpenReview by 1 Oct AoE. |
| **R4 Page limit** | `W/guidelines-and-policies/instructions/`: "Papers submitted to the main track must be at most eight (8) pages long, with any number of additional pages containing bibliographic references. Excessive use of typesetting tricks to make everything fit into 8 pages is not admissible." | **PASS** for the current build, with the same caveat as R6 | `ai_race_paper.pdf` has 9 pages. The main content ends on p. 8, with the last line of §5 ("…or uncertainty about real capabilities."). About a quarter of the right-hand column is left empty. References start at the top of p. 9 ("REFERENCES", forced by `\clearpage` at main.tex:1052). | Recheck after R6 and R7 are fixed: restoring the copyright block and reference line takes space on p. 1. |
| **R5 Appendix, acknowledgements, ethics toward the limit** | `W/guidelines-and-policies/qa/`: "An appendix included in the main paper is considered part of the paper and therefore counts towards the 8-page limit." No page says how acknowledgements or an ethics statement count. Template: "Note that the text of your acknowledgments will be omitted if you compile your document with the `anonymous` option." | **PASS** | main.tex has no appendix. The acknowledgements are commented out (main.tex:1043-1050). | None now. For the camera-ready, use the `acks` environment. Do not restore the old `\section*` block. |
| **R6 Template version, style files unmodified** | `W/guidelines-and-policies/instructions/`: "conform to our formatting guidelines: Download [aamas_2027_template.zip] … Please do not modify the style files or any of the layout parameters. The use of LaTeX is mandatory." The official `aamas.cls` inside the zip is `[2026/06/27 v2.19 …]`. | **FAIL** | `paper/aamas.cls:40` is `[2021/05/01 v1.78 …]`, which is 5 years older than the official file (sha256 `f0af39a7…`; official is `e88c8e3e…`). The PDF metadata also says "aamas 2021/05/01 v1.78". The `ACM-Reference-Format.bst` file also differs from the one in the zip. The old class has no `\submissionType`, so the page-1 header "Research Paper Track" and the "AAMAS 2027, 3–7 May 2027, Hanoi" line are missing. Headings print in the 2021 style ("KEYWORDS", "1 INTRODUCTION"), not in the 2027 sample's style ("Keywords", "1 Introduction"). | Replace `aamas.cls` and `ACM-Reference-Format.bst` with the files from `aamas_2027_template.zip`. Add `\submissionType{Research Paper Track}`. Rebuild and recheck R4. |
| **R7 Copyright block and ACM reference format** | Template `AAMAS_2027_sample.tex`: "%%% AAMAS-2027 copyright block (do not change!)" followed by the CC-BY `\@copyrightpermission` block and `\acmConference[AAMAS 2027]{…}{3--7 May 2027}{Hanoi, Vietnam}{M.~Baldoni, F.~Fang, W.~Yeoh, N.~Yorke-Smith (eds.)}`. Its text says: "Make sure your paper includes the correct copyright information and the correct specification of the ACM Reference Format." | **FAIL** | main.tex:338 `\settopmatter{printacmref=false}` removes the ACM Reference Format. main.tex:339 `\renewcommand\footnotetextcopyrightpermission[1]{}` removes the copyright and licence block. main.tex:44-54 has the wrong block: "AAMAS '27", "May 3 -- 7, 2027", "AAMAS 2027 editors", and no CC-BY minipage. Page 1 of the PDF shows neither item (checked visually). The same two lines are in supplementary.tex:303-304. | Paste the official copyright block exactly as written and delete lines 338-339. These removals also save space on p. 1, so reviewers could read them as the "typesetting tricks" R4 forbids. |
| **R8 No changes to layout parameters** | `W/guidelines-and-policies/instructions/`: "Please do not modify the style files or any of the layout parameters." Template: "Modifying the template---e.g., by changing margins, typeface sizes, line spacing, paragraph or list definitions---or making excessive use of the `\vspace` command … is not allowed." | **UNKNOWN (at risk)** | main.tex:34-40 redefine `\textfraction`, `\topfraction`, `\bottomfraction`, `\floatpagefraction` and the `topnumber`/`bottomnumber`/`totalnumber` counters. main.tex:346-347 turn on `\pagestyle{fancy}\fancyhead{}`, which the 2027 sample leaves commented out, so page headers are removed. main.tex has no `\vspace`. | Float fractions are not on the template's list of examples, but they are layout parameters. The safe course is to delete lines 34-40 and 346-347, then rebuild and check R4. |
| **R9 Double-blind anonymity** | `W/guidelines-and-policies/instructions/`: "Papers should be … prepared for double-blind reviewing". Reviewer guidelines: desk rejection if papers "are not properly anonymized (e.g., include author names or identifying information in the main paper or supplementary material)" and "While it may be possible to guess the authors based on citations or prior work, this does not constitute a violation unless the submission explicitly reveals the authors' identities." | **PASS**, with one minor leak | `\documentclass[sigconf,anonymous]` is set (main.tex:19; supplementary.tex:19). The anonymity receipts are clean for the paper PDF, the supplementary PDF and the zip (scanner v3, 2026-09-14). The PDF Author field is empty. There is no URL in either source apart from a comment (main.tex:16). Self-citations are in the third person ("the two-player game studied by Domingos and Han [14]"). One minor leak: p. 1 prints "Submission Id: TBD∗∗∗∗∗∗∗∗†", i.e. the anonymised author-note markers from `\authornotemark`/`\authornote` (main.tex:226-289). This shows that the paper has 8+ equal-contribution notes and a corresponding-author note. | Optional: comment out the `\authornote`/`\authornotemark` lines while the paper is anonymous. |
| **R10 Self-citation, code links, arXiv** | Q&A #5 at `W/guidelines-and-policies/qa/`: "Posting a paper on a preprint server such as arXiv does not count as publication in an archival venue and is therefore permitted. However, the paper must not be simultaneously submitted to another archival venue". No page states a rule on code links. The Supplementary Material section only says "you must ensure that your supplementary material does not compromise the anonymity of your submission." | **PASS** | There are no code or data links. Kaggle accounts are anonymised in the supplement (supplementary.tex:1504-1507). | None |
| **R11 Supplementary material format** | `W/guidelines-and-policies/instructions/`: "Supplementary material should be submitted as a single zip file and should not exceed 25MB." Also: "Do not use supplementary material to submit an extended or corrected version of your paper". | **PASS** | `results/artifacts/submission/supplementary.zip` is 2.54 MB and holds one file, `supplementary.pdf` (31 pp.). Its sections are extended results A to E (supplementary.tex:376-3261), not a second version of the paper. | Rebuild the zip after the main paper is rebuilt, because the zip dates from 2026-09-14. |
| **R12 Reviewers may skip the supplement; essential content goes in the paper** | `W/guidelines-and-policies/instructions/`: "Note that reviewers are not required to look at supplementary material. … Any information that is essential for understanding or evaluating your paper must be included in the paper itself." Reviewer guidelines: "papers are expected to be self-contained and should not rely on appendices or supplementary material for core contributions." | **UNKNOWN (judgement)** | The main paper sends several things to the supplement: the persona, skin and code manipulations (main.tex:620-627), the full screen results (650-651), the contract provenance (775-776), the five-route EGT fit (841, 886-887) and the representation-robustness numbers (726-729). The core claims (risk, rival, theory, diversity) have their numbers and intervals in the main text. | Check that no headline claim is supported only by the supplement. The strongest candidate is the phrase "race-weighted reanalysis preserves the non-increasing response" (main.tex:677-678). |
| **R13 Archival supplement at camera-ready** | `W/guidelines-and-policies/instructions/`: "you should make your (suitably revised) supplementary material openly available in archival form at the time of publication of your paper, and … include a reference to the supplementary material in the camera-ready version". | UNKNOWN (applies after acceptance) | n/a | Plan a Zenodo deposit for the camera-ready (deadline 25 Jan 2027). |
| **R14 Dual and concurrent submission** | `W/guidelines-and-policies/instructions/`: "Authors must not submit substantially similar work to AAMAS 2027 and another archival venue at the same time, or submit the same work elsewhere while it remains under review at AAMAS 2027. … Submissions that constitute 'thin slicing' or whose publication would make another overlapping submission too incremental may be rejected." The OpenReview form makes authors confirm: "I confirm that neither this manuscript nor a substantially similar version of it is currently under review at another archival venue." and "I confirm that any other relevant simultaneous submissions by any author of this manuscript are cited in this manuscript in anonymous form as \"under review\" work." | **UNKNOWN: depends on when the ARR paper is submitted** (details in the section after this table) | main.tex has no anonymous "under review" citation (grep for "concurrent\|under review\|anonymous" in main.tex and references.bib finds none). The ARR paper does cite the AAMAS paper (`paper/acl/custom.bib:13-19`), but under an **outdated title**, "More Than the Risk: Frontier LLM Safety Policies Depend on the Rival". The current title is "…Frontier LLM Behaviour in AI Development Races Depends on the Rival" (main.tex:81). | See the concurrent-submission section. Fix the title in `paper/acl/custom.bib`. |
| **R15 Policy on AI-assisted writing and research** | `W/guidelines-and-policies/instructions/`: "It is permissible to use AI-assisted technologies for the polishing and formatting of text and in the creation of code and scripts … Where used in the creation of hypotheses or methodologies, including experimental design, detailed information should be provided in the paper or in the supplementary material, including the prompt used, as well as the AI tool and its version. … Area Chairs may choose to desk reject papers if assistive technology is used inappropriately, e.g., hallucinated citations." Q&A #3: "Do not reconstruct or guess information that was not retained. Instead, disclose this explicitly". | **UNKNOWN (likely FAIL)** | Neither main.tex nor supplementary.tex contains a statement on the authors' own use of AI tools (grep for "assist\|ChatGPT\|Claude Code\|Codex\|generative AI" finds none). The repository is run through Claude Code agents (CLAUDE.md, `.agents/skills/`). Whether that use reached "hypotheses or methodologies, including experimental design" is for the authors to say. | Authors decide the scope of use. If any design or hypothesis work was AI-assisted, add a disclosure section to the supplement: the tool, its version, and the prompts, or a clear statement that the prompts were not retained. Tick the fourth form confirmation only after a citation audit (see Could not verify). |
| **R16 AI-generated images** | `W/guidelines-and-policies/instructions/`: "AI-generated images and other multimedia are only permitted if generative AI is the topic of the paper, and such images or multimedia are provided as qualitative evidence of the research output." | **PASS** (main paper); **UNKNOWN** (supplement) | The five main-paper figures are drawn by code: `scripts/figures/fig_overview.py` makes `delegation_overview`, and the other four are data plots. The supplement includes `figures/paper/ExpOverview.pdf` (Creator: Canva; it contains 308×308 and 205×205 raster icons whose origin is unknown). | Confirm that the Canva icons are not generative-AI images. If they are, redraw them or remove the figure. |
| **R17 Abstract registration and keywords** | `W/guidelines-and-policies/instructions/`: "registering an abstract of your paper (of around 100-300 words in plain text) is required one week before the paper submission deadline, and you will be asked to provide some additional information at this time, such as keywords characterizing your paper." | **PASS** (content ready) | The abstract is 271 words (main.tex:294-323). There are 6 keywords (main.tex:329). | Paste the abstract as plain text. The paper has no maths in it. |
| **R18 Area and topic selection** | `W/guidelines-and-policies/instructions/`: "Papers must be associated with one of the AAMAS areas. Area Chairs will check that abstracts are within scope for the area; if not we will attempt to identify a more appropriate area but will desk reject the paper if this can not be done." The form requires `primary_subject_area` (exactly one), allows up to two optional `secondary_subject_areas`, and requires `area_topics` ("Select at least one and no more than three topics"). The GAAI scope reads: "Advances in language models, prompt engineering, generic tool use, or arbitrary generative tasks are outside the scope of this area unless they make a clear contribution to autonomous agents or multiagent systems." | UNKNOWN (not chosen yet) | n/a | Suggestion, not a rule: primary area GAAI; secondary GTEP (+ SIM). Topics: "GAAI: Modeling and analysis of generative AI agents", "GTEP: Behavioural game theory", "GTEP: Evolutionary game theory". |
| **R19 Ethics statement, broader impact, reproducibility checklist, CCS concepts** | None of the official pages or the OpenReview form asks for any of these. The 2027 sample `.tex` has no CCS block. | **PASS** (not required) | There is no CCSXML in main.tex, which is consistent with the 2027 sample. | None |
| **R20 Reciprocal reviewing** | `W/guidelines-and-policies/reciprocal-reviewer-policy/`: "Each submission must designate at least one qualified author who agrees to serve as a reviewer for AAMAS. … Submissions that fail to satisfy this requirement may be desk-rejected." The reviewer guidelines define who qualifies: "(1) hold a PhD in Computer Science or a closely related field; or (2) be a PhD student in their third year or later with at least 3 peer-reviewed publications". The form has the fields `reciprocal_reviewer_nomination` and `reciprocal_reviewer_nomination_confirmation`. | UNKNOWN (nothing in the repo) | At least the senior authors should qualify: Fernández Domingos, Le Hong Trang and The Anh Han (main.tex:277-291). I could not confirm their consent. | Before 1 Oct, get one senior author to agree and name them in the form. Reviewing runs until 13 Nov, with rebuttal acknowledgement by 3 Dec. |
| **R21 Findings opt-out** | `W/calls/call-for-main-track/`: "Papers that are not selected will be automatically considered for publication in the Findings of AAMAS 2027 under a CC-BY licence, unless the authors opt out of this option in the submission form". The form field `findings_of_AAMAS_2027` (Yes/No) is required. | UNKNOWN (a decision) | n/a | Decide before submitting. Findings is archival, so accepting it affects any later reuse of this material (see R14). |
| **R22 Submission ID on the PDF** | Template: "you will be assigned a submission number when you register the abstract of your paper on OpenReview. Include this number in your document using the `\acmSubmissionID` command." | **FAIL** (expected at this stage) | `paper/submission_id.tex:2` is `\newcommand{\SubmissionID}{TBD}`. The PDF p. 1 prints "Submission Id: TBD". | After abstract registration, set the ID and rebuild without `--allow-placeholder-id`. |
| **R23 Author list frozen** | `W/guidelines-and-policies/instructions/`: "altering the author list of a paper, or changing the order of authors, after acceptance is not allowed." | UNKNOWN | main.tex has two author blocks with different orders: the commented block at 84-179 and the active block at 225-291. The comment at 223 says "listed alphabetically", but the active order is not alphabetical. | Fix the final list and order before registering the abstract. |
| **R24 Student paper flag** | `W/guidelines-and-policies/instructions/`: "If the primary author of your paper, i.e., first author or co-first author, is a student, then please register your paper as a student paper by checking the 'the primary author is a student' checkbox." | UNKNOWN (form field) | The co-first authors use student emails (main.tex:228-250). | Tick the box. |
| **R25 Main track or Blue Sky** | `W/calls/call-for-blue-sky-ideas/`: "Submissions are limited to 4 pages in length in the AAMAS 2027 format … Abstract submission: 5 Nov 2026 / Paper submission: 12 Nov 2026". Blue Sky also requires two extra form fields: "Rationale for Blue Sky" and "Author Track Record". | **PASS** (main track fits) | This is an 8-page empirical research paper, so it belongs in the main track, which the 2027 class calls "Research Paper Track". | Use `\submissionType{Research Paper Track}` (needs R6). |

## Concurrent submission: what AAMAS and ARR each require

**AAMAS 2027** (sources quoted in R14):

1. You may not submit substantially similar work to AAMAS and another archival venue at the same time. You may not submit this work elsewhere while it is under review at AAMAS, which runs from 8 Oct to 21 Dec 2026.
2. A paper can be rejected for thin slicing, or when its publication "would make another overlapping submission too incremental".
3. At submission the authors must confirm that every relevant simultaneous submission by any author "is cited in this manuscript in anonymous form as 'under review' work."
4. AAMAS does **not** ask for the other paper to be uploaded. I found no such text.

**ARR** (`https://aclrollingreview.org/cfp`):

- "There can be no overlap in stated contributions. … any concurrently submitted papers on a related topic with an overlapping set of authors must cite each other and discuss the differences in the related work section. Anonymized versions of such papers should be uploaded as supplementary material (in the 'data' field in the submission form). Recommended citation format in the bibliography: Anonymous (2026). Paper title. Under review."
- "ARR will not consider any paper that is under review in a journal or another conference at the time of submission … we will not consider any paper that overlaps significantly in content or results with papers that will be (or have been) published elsewhere".
- The page also points to a text-reuse limit, "no more than 10% total tokens".

**The timing problem the plan misses.** `docs/acl-audit-paper-plan.md` §6 item 2 says the AAMAS paper must cite the ARR paper "before the AAMAS submission goes out". That is only correct if the ARR paper is really under review by 8 Oct 2026.

- The plan says the ARR paper is "not submittable today" because data is missing (§0, §7).
- The next ARR deadline is **12 Oct 2026**. The dates page lists no later cycle except "ACL 2027: January, 2027".
- If the ARR paper is not submitted by 8 Oct, citing it in the AAMAS PDF as "under review" is false. It would also make the third form confirmation untrue.
- If the ARR paper goes in on 12 Oct, it will be under review alongside AAMAS. But the AAMAS PDF cannot be changed after 8 Oct AoE, so the AAMAS side has no citation.

In that case the ARR side still must cite the AAMAS paper and upload its anonymised PDF in the `data` field. It must also use the **current** title; `custom.bib` still has the old one. Writing to the AAMAS PCs is my suggestion, not a written rule.

If the ARR paper waits until January 2027, the AAMAS paper will have a decision by then. If AAMAS accepts it, into the Proceedings or into Findings (which is archival), ARR's rule on significant overlap with published work applies to the whole shared roster, protocol and screen data. This is a second reason to decide on the Findings opt-out (R21) with the ARR plan in mind.

On the AAMAS side, main.tex:645-648 ("the admitted/refused split coincides with endpoint tier naming") and main.tex:956-961 argue from the screen. The claim map allows the first as a bare fact (A5). Both need rereading against the thin-slicing clause once the ARR scope is fixed.

## Deadlines

All dates are official 2027 dates. Hanoi time is ICT, UTC+7, so the end of a day AoE is 18:59 ICT on the following day.

| event | official date | end of day AoE, in UTC | in Hanoi |
|---|---|---|---|
| OpenReview accounts for all authors | 17 Sep 2026 | 18 Sep 11:59 UTC | 18 Sep 18:59: **passed** |
| Abstract registration (main track) | 1 Oct 2026 | 2 Oct 11:59 UTC (also in the OpenReview record) | 2 Oct 18:59 |
| Paper and supplementary submission | 8 Oct 2026 | 9 Oct 11:59 UTC (also in the OpenReview record) | 9 Oct 18:59 |
| Reviews due (reviewer duty, R20) | 13 Nov 2026 | | |
| Rebuttal | 20-24 Nov 2026 | | |
| Notification | 21 Dec 2026 | | |
| Camera-ready | 25 Jan 2027 | | |
| Blue Sky: abstract / paper | 5 Nov / 12 Nov 2026 | | |
| AAAI Fast Track | 11 Dec 2026 | | |
| Conference, JW Marriott Hanoi | 3-7 May 2027 | | |
| ARR October 2026 cycle | submission 12 Oct 2026, cycle ends 20 Dec 2026 | | |

## Could not verify

- **Whether all 12 authors have OpenReview profiles.** Profile search on OpenReview needs a login and returns 403 to a guest. The deadline passed about 4 hours before this check.
- **Whether a qualified author has agreed to review** (R20), and whether the designated person meets the PhD rule.
- **Page count under the official v2.19 class** with the copyright block and ACM reference line restored and the float overrides removed. That needs a rebuild, which this gate did not allow. p. 8 has about a quarter of a column free now, so overflow is possible.
- **Whether the 2026-09-14 PDF matches the current `main.tex`**, because another agent is editing it.
- **How far AI tools were used** in hypotheses, methodology or experimental design (R15). Only the authors know this.
- **Whether the Canva icons in `ExpOverview.pdf` are generative-AI images** (R16).
- **Whether any citation is hallucinated.** This gate did not audit citations, although the form asks authors to confirm there are none.
- **The review status of Fernández Domingos and Han, arXiv 2607.26034.** Two of its authors are authors here. If it is under review at an archival venue, the form's "simultaneous submissions" confirmation may cover it. It is not substantially similar work, so the dual-submission rule itself is not at issue.
- **How acknowledgements, ethics statements and extra pages count at camera-ready.** No official page states this.
- **ARR cycles after October 2026.** The dates page lists only "ACL 2027: January, 2027". The repo records no target ARR cycle.
- **The official AAMAS 2027 FAQ.** The main-track call says "a FAQ page will be posted here soon". The existing Q&A page has 5 entries, and all were read.

## Three most urgent actions

1. **Today: OpenReview accounts.** The author-account deadline passed at 18:59 Hanoi on 18 Sep. Confirm now that each of the 12 authors has an active and complete OpenReview profile (name, position, institutional email, DBLP). Create any that are missing and tell aamas2027pcs@gmail.com. In the same pass, get a PhD-holding author to agree to be the reciprocal reviewer; the nomination is due with the abstract on 1 Oct.
2. **Before any more layout work: switch to the official 2027 template.**
   - Replace `paper/aamas.cls` (v1.78, 2021) and `ACM-Reference-Format.bst` with the files from `aamas_2027_template.zip`.
   - Paste the official copyright block, including `\acmConference`, the editors and the CC-BY minipage.
   - Add `\submissionType{Research Paper Track}`.
   - Delete main.tex:34-40 (float overrides), 338-339 (reference line and copyright removed) and 346-347 (headers removed).
   - Rebuild and confirm the main content still ends by p. 8.

   Any page fitting done on the current build is not reliable, because R6 and R7 are both page-1 fails.
3. **Settle the statements before 1 Oct.**
   - Choose the ARR timing and write the AAMAS citation to match it. Do not cite the ARR paper as "under review" unless it is submitted by 8 Oct. Fix the stale title in `paper/acl/custom.bib`.
   - Decide the AI-use disclosure for the supplement, and the Findings opt-in or opt-out.
   - Choose the primary area and 1-3 topics.
   - After abstract registration, put the OpenReview ID into `paper/submission_id.tex`.
