# 04 Build QA (clean-room rebuild of committed HEAD)

Date: 2026-09-18. Tree audited: `D:\aiqa`, a `git archive` export of the committed
HEAD (no `.git`, no `data/`). The export matches the live repository's HEAD
`6f54ecd` (2026-09-14) for every tracked file under `paper/` except the PDFs and
receipts the build rewrote, checked with `git hash-object` against
`git ls-tree HEAD`. Nothing was fixed. Every result below comes from a command
that was run, with its output read.

Toolchain: Python 3.14.0, MiKTeX-pdfTeX 4.26.0 (1.40.29), MiKTeX bibtex,
poppler `pdffonts`/`pdfinfo`/`pdftotext`/`pdftoppm`.

`data/` was not needed by any step. No test, build step or checker failed
because it was missing.

## Results

| # | Check | Command (cwd `D:\aiqa`) | Exit | Verdict |
|---|---|---|---|---|
| 1 | Test suite | `python -m pytest -q` | 1 | **FAIL (environmental)**: 543 passed, 2 failed, 5 skipped, 248 s |
| 2 | Paper + supplement build | `python scripts/build_publication.py --paper-only` | 0 | PASS: both PDFs built, bibtex found `references.bib`, anonymity scan clean on both |
| 3a | Publication QA (placeholder allowed) | `python scripts/check_publication.py --allow-placeholder-id` | 0 | PASS: "publication QA passed" |
| 3b | Publication QA (strict) | `python scripts/check_publication.py` | 1 | FAIL, expected: "submission ID is still TBD" in both PDFs |
| 3c | Anonymity gate self-test | `python scripts/check_anonymity_gate.py` | 0 | PASS: 16/16 PASS lines, "anonymity gate check passed" |
| 3d | Page budget | `python scripts/check_page_budget.py` | 0 | PASS: 9 pages, references begin at top of p. 9, 8 content pages vs limit 8 |
| 3e | Cross-references | `python scripts/check_paper_crossrefs.py` | 0 | PASS: main 20 labels/13 refs, supplement 67/64, ARR 25/17, none unresolved |
| 3f | House style | `python scripts/check_house_style.py` | 0 | PASS: 0 problems in main.tex and supplementary.tex |
| 3g | Figure geometry | `python scripts/check_figure_geometry.py` | 0 | PASS: 20 figures at drawn size (ExpOverview.pdf is protected ART, printed at 65.1%) |
| 3h | Printed type | `python scripts/check_printed_type.py` | 0 | PASS: every vector figure clears the 6.0 pt body floor; one raster (01_tsne_hero_human_left.png) not measurable |
| 3i | Symmetry (no args) | `python scripts/check_symmetry.py` | 2 | N/A: argparse error, `--input` required. It is a pilot run check, not a paper check |
| 3i' | Symmetry (on baseline) | `python scripts/check_symmetry.py --input results/frontier/baseline_campaign_v6` | 0 | PASS: tied-throughout share 26.7%, "OK: enough races separated" |
| 3j | Figure gallery | `python scripts/check_figure_gallery.py` | 1 | **FAIL**: 77 missing figure sources + 37 hash mismatches (see Non-blocking) |
| 4 | Independent PDF inspection | pdfinfo / pdffonts / pdftotext / LaTeX logs | n/a | PASS apart from the items listed under Blocking |
| 4' | Committed PDF staleness | text diff and 60 dpi page-by-page pixel diff, committed vs rebuilt | n/a | PASS: committed PDFs are **not stale** (text identical, every page pixel-identical) |
| 5 | Submission bundle | `python scripts/build_submission_bundle.py` (no dry-run flag exists; it only writes locally to `results/artifacts/submission/` inside `D:\aiqa`, uploads nothing) | 0 | PASS: zip 2.43 MB vs 25 MB cap, anonymity clean. It does **not** refuse a TBD submission ID |

### Step 1 detail: pytest

Failing tests (both same cause):

- `ai_race/tests/test_comprehension_reaudit.py::test_coordinator_retains_complete_raw_evidence_and_fail_closed_manifest`
- `ai_race/tests/test_comprehension_reaudit.py::test_bad_response_envelope_marks_manifest_failed`

Verbatim tail:

```
E               subprocess.CalledProcessError: Command '['git', 'rev-parse', 'HEAD']' returned non-zero exit status 128.
C:\Python314\Lib\subprocess.py:577: CalledProcessError
```

Cause: `kaggle/experiments/greennode_comprehension_reaudit.py:424` calls
`git rev-parse HEAD` with `check=True`; the export has no `.git`. **Neither
failure is due to the missing `data/`.** They are artefacts of auditing a
non-git export; they would likely pass in a git checkout (not run here). The
tests do make the suite depend on being inside a git work tree.

Skipped (5): `ai_race/tests/test_models_factory.py:306, 336, 369, 395, 441`,
"the Bedrock backend needs the api extra". None skip for missing data.

### Step 2 detail: build

- `build_publication_figures.py` ran first and regenerated the generated
  figure set without error.
- bibtex: `BIBINPUTS` is set by the script from a native Windows `Path`
  (`D:\aiqa\paper`), and both `.blg` files report `Database file #1:
  references.bib`. One warning each: "page numbers missing in both pages and
  numpages fields in sclarPromptFormatting2024". No `[?]` and no `??` in either
  PDF's text layer.
- Final logs: 0 undefined references, 0 undefined citations, 0 multiply
  defined labels, 0 "Rerun" requests, 0 missing characters, **0 overfull
  boxes** in either document.
- Outputs: `paper/ai_race_paper.pdf` (9 pp, 781,244 B) and
  `paper/ai_race_supplementary.pdf` (31 pp, 2,771,236 B), receipts written.

### Step 4 detail: PDF inspection

| | Main paper | Supplement |
|---|---|---|
| Pages | 9 | 31 |
| Content ends / references start | p. 8 ends in Limitations; `REFERENCES` is the first line of p. 9, p. 9 is references only | n/a |
| Page size | US Letter 612 x 792 | US Letter |
| Type 3 fonts | none | none |
| Non-embedded fonts | none (all `emb yes`) | none |
| Font set | Linux Libertine/Biolinum, newtx math (Type 1); Arial (CID TrueType) from matplotlib figures | same, 24 distinct |
| Author / Subject | absent / empty | absent / empty |
| Title | the paper title | "Supplementary Material: " + title |
| Creator | `LaTeX with aamas 2021/05/01 v1.78 ... hyperref 2026-01-29 v7.01p` | same |
| Producer | `MiKTeX-pdfTeX 4.26.0 (1.40.29)` | same |
| XMP | Keywords, `dc:source ai_race_paper.tex`, timestamps with `+07:00` offset; no names | same pattern |
| Page-one byline | "Anonymous Author(s) Submission Id: TBD" | same |
| Names, emails, affiliations on p. 1 | none | none |

Independent raw-stream scan (all streams zlib-inflated) for local paths,
the 12 Kaggle identities printed by the figure builder, `hcmus`, `hcmut`,
`github.com`, `kaggle.com`, `D:/`, `/home/`: zero hits in either PDF. The only
hit for "Minh" is the author list of references [18] and [19].

Committed vs rebuilt: page counts equal (9 and 31), `pdftotext` output
identical for both documents, and every page pixel-identical at 60 dpi. The
byte difference (776,263 to 781,244) comes from regenerated figure files, not
from visible content.

## Blocking

1. **Submission ID is `TBD` and printed on page 1 of both PDFs**
   ("Submission Id: TBD"). Source: `paper/submission_id.tex:2`
   (`\newcommand{\SubmissionID}{TBD}`). `check_publication.py` without the
   placeholder flag fails on exactly this. Known and expected until AAMAS
   assigns the ID, but the paper must not be uploaded in this state.
2. **`build_submission_bundle.py` builds a bundle with `TBD` in it.** It checks
   staleness and anonymity but never the submission ID, so the gate that
   produces the upload files passes a PDF that fails strict
   `check_publication.py`. Run strict `check_publication.py` before uploading;
   the bundle's exit 0 is not sufficient.
3. **Template/format compliance (needs a decision; cross-reference
   `01_cfp_compliance.md`).** The build uses `aamas.cls` v1.78 dated
   2021/05/01 (PDF Creator string). `paper/main.tex:338-339` sets
   `\settopmatter{printacmref=false}` and blanks
   `\footnotetextcopyrightpermission`, and page 1 of the built PDF carries no
   AAMAS proceedings/copyright block (no "Proc.", "IFAAMAS" or "©" in the
   p. 1 text). The class itself warns at the end of the run:
   "ACM reference format is mandatory for papers over one page" and "CCS
   concepts are mandatory for papers over two pages". The page budget passes
   only in this configuration (8 content pages exactly); restoring the block
   may push content onto p. 9. I did not rebuild with the official class, so
   whether this overflows is unverified.

Nothing else found would visibly break the PDF: no `??`, no `[?]`, no overfull
boxes, no Type 3 or unembedded fonts, no identifying metadata.

## Non-blocking

- `ai_race/tests/test_comprehension_reaudit.py` (2 tests) require a git work
  tree via `kaggle/experiments/greennode_comprehension_reaudit.py:424`; fails in
  any archive export.
- `scripts/check_figure_gallery.py` exit 1, two unrelated causes:
  - 77 "missing figure source" entries in `figures/manifest.json` point into
    gitignored paths (`.gitignore:22 /results/**`): 52 in
    `results/cross_model_pilot_synthesis/deep_insight_gallery_20260802/`, 14 in
    `.../figure_candidates_20260802/`, 10 in `.../top5_story_figures_20260803/`,
    and 1 temp file
    `results/impact_upgrade/impact_report.html.tmp-10184-...-verification-failure.png`.
    These exist in the live working tree but not in any clean checkout, so this
    check can never pass from git alone.
  - 37 hash mismatches under `figures/paper/`. For all 37, the **committed HEAD
    blob** already differs from the sha256 recorded in `figures/manifest.json`
    (checked with `git show HEAD:<path>` against the manifest), so the manifest
    was stale before this rebuild, not only after it.
    `build_figure_gallery.py` was not run (it would modify the tree).
- Regenerating figures rewrites 29 tracked files under `figures/paper/`
  (PDF/SVG bytes differ from HEAD) with no visible change in the manuscript
  (pixel-identical pages). Likely embedded timestamps; the generators are not
  byte-reproducible.
- `paper/references.bib`, entry `sclarPromptFormatting2024`: no pages/numpages
  (bibtex warning in both `.blg`).
- `paper/main.tex:544` (`\subsection{$N$-player game}`): hyperref "Token not
  allowed in a PDF string", math shift dropped from the bookmark (twice).
  Supplement has 4 of the same warning.
- `paper/main.tex:341` microtype "Unable to apply patch `footnote'"
  (supplementary.tex:306 same). Harmless.
- Main paper p. 6 is a float-only page ("Text page 6 contains only floats",
  log lines 1239, 1246).
- `\balance` called in the second column ("Columns might not be balanced").
- Supplement: 4 "`h` float specifier changed to `ht`".
- XMP timestamps carry `+07:00` (Indochina time zone). Weak location hint,
  normal for any PDF built in Vietnam.
- Self-citations [18] Huynh et al. 2025 and [19] Huynh et al. 2026 list six
  people who also appear in the commented camera-ready author block
  (`paper/main.tex:84-150`). The text cites them in the third person
  ("later work adds payoff scaling ... [18, 19]"), which is the normal
  double-blind practice; noted only because author overlap is visible to a
  reviewer who compares names.
- No anonymous concurrent-work citation to the ARR companion paper exists in
  `main.tex`, `supplementary.tex` or `references.bib` (grep for
  "concurrent"/"companion"/"Anonymous" finds nothing relevant), although
  `CLAUDE.md` says both papers must cite each other. Decide whether that
  applies to this submission, given the ARR paper is not submittable.
- `scripts/build_publication_figures.py` prints the 12 Kaggle identity names to
  stdout during the build. They do not reach either PDF (raw-stream scan clean),
  but they land in any captured build log.
- The commented author block with student emails lives in `paper/main.tex`.
  The bundle ships only PDFs, so this matters only if sources are ever uploaded
  (for example to arXiv or as supplementary source).

## Overfull boxes

None. `grep -n Overfull` on
`results/_build/latex/current/ai_race_paper.log` and
`ai_race_supplementary.log` returns zero lines for the final pass.

For completeness, the underfull boxes (all non-blocking):

Main paper (`ai_race_paper.log`): 12 `Underfull \vbox (badness 10000) while
\output is active`; `Underfull \hbox` in the bibliography at `.bbl` lines
403-412 (Payne and Alloui-Cros, badness 2809, 3158) and 458-470 (Sclar et al.,
badness 3039, 2310, 1902).

Supplement (`ai_race_supplementary.log`), `\hbox`, with `supplementary.tex`
line:

| line | badness | text |
|---|---|---|
| 338 | 1845 | roadmap table "C. Scripted-rival intervention" |
| 349 | 1127 | roadmap table "E. Exploratory interaction scope" |
| 355-356 | 1221 | "Scoped extensions and the remaining reproducibility record" |
| 724 | 1005 | caption ending "...distance to the nearest rule is largest." |
| 1120 | 10000, 1028, 10000 | Table 7 caption with `google/gemini-3-flash-preview` in monospace |
| 1295 | 10000 | related-work table "Pezeshkpour and Hruschka" |
| 1302 | 10000 | "Salinas and Morstatter" |
| 1322 | 10000 | "Robinson and Burden" |
| 2965 | 10000, 1009, 3240 | caption ending "...five-company effect against no boundary." |
| `.bbl` 179-188, 234-246 | 2809, 3158, 3039, 2310, 1902 | same two bibliography entries as the main paper |

plus 26 `Underfull \vbox` during output.

## Step 5 detail: bundle

Output (verbatim):

```
anonymity scan: clean
anonymity scan: supplementary.zip
    clean
paper           : 0.75 MB
supplementary   : 2.43 MB  (limit 25 MB)
bundle written to results\artifacts\submission
```

`supplementary.zip` contents: one member, `supplementary.pdf`, 2,771,236 B
uncompressed, 2,549,754 B compressed; zip file 2,549,886 B. Bundle folder also
holds `paper.pdf`, `supplementary.pdf`, three `*.anonymity.json` receipts and
`README.txt`. No refusal. The script has no `--dry-run`; it writes only to
`results/artifacts/submission/` and uploads nothing. The live checkout was not
touched except for this file.
