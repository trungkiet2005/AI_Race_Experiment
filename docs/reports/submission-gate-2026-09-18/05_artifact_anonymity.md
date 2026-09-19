# 05 Artifact anonymity and reviewer reproducibility

Audit date 2026-09-18. Read-only audit. Nothing was edited, built, committed or deleted.
Identifying strings are redacted to their first three characters plus `***`.

Artefacts audited, with their state at audit time:

| file | bytes | SHA-256 (prefix) | anonymity receipt |
|---|---|---|---|
| `results/artifacts/submission/paper.pdf` (9 pp) | 776,263 | `ad867d95` | v3, clean, matches bytes |
| `results/artifacts/submission/supplementary.pdf` (31 pp) | 2,765,949 | `c032ca3c` | v3, clean, matches bytes |
| `results/artifacts/submission/supplementary.zip` | 2,538,950 | n/a | v3, clean |
| `paper/main.tex`, `paper/supplementary.tex` | source | n/a | n/a |
| `figures/paper/**` (20 files the two PDFs include) | n/a | n/a | n/a |

`paper/ai_race_paper.pdf` and `paper/ai_race_supplementary.pdf` are byte-identical to the bundle copies.
Both PDFs are newer than their `.tex` sources (PDF 2026-09-14 16:38, `main.tex` 16:00, `supplementary.tex` 16:36).
Another agent is editing the paper in parallel, so everything below must be re-checked after the next rebuild.

## 1. Bundle contents and size estimate

`scripts/build_submission_bundle.py` ships only the following:

| upload | contents | size |
|---|---|---|
| `paper.pdf` | the manuscript, uploaded as its own file | 0.74 MiB |
| `supplementary.zip` | one member: `supplementary.pdf` | 2.42 MiB (limit 25 MiB) |

These files are not uploaded and are local only: `README.txt`, `supplementary.pdf` (a reading copy) and the `*.anonymity.json` receipts.

`scripts/build_release_manifest.py` is not part of the submission. It writes a SHA-256 list of about 60 internal files to `results/impact_upgrade/release_manifest.json`. That delivery surface includes `kaggle/impact_kernel/kernel-metadata.json` and `dataset-metadata.json`, which carry `"id": "dao***/ai-race-impact-admission"` and `"dao***/ai-race-admission-source"`. It is a latent leak if that surface is ever packaged. Today it is not.

**No code or data artefact is shipped or linked.** The paper contains no link to GitHub, Kaggle, Hugging Face, Zenodo or OSF. All 36 link targets in the paper and all 17 in the supplement point to arxiv.org, doi.org, proceedings.mlr.press or proceedings.iclr.cc.

## 2. Leaks found

### 2a. In the shipped files (paper.pdf, supplementary.pdf inside the zip)

No author name appears as an author. No email, compute handle, GitHub handle, institution, grant, local path or credential appears in the text layer, the inflated object graph, the Info dictionary or the XMP. XMP `dc:creator` = "Anonymous Author(s)". Neither PDF has an `/Author` key. The remaining findings are indirect:

| # | file | location | string (redacted) | severity |
|---|---|---|---|---|
| L1 | supplementary.pdf | raw byte offsets 395427 and 395439, in object 502, the `PTEX.InfoDict` of `figures/paper/ExpOverview.pdf` | `/Keywords (DAH***,BAE***)`: a Canva design ID and a Canva brand/team ID. They sit next to `/Creator (Canva)` and `/Title (h)`. The source figure's XMP also has `dc:language="vi-VN"`, but that XMP is not carried into the supplement. | Medium. The IDs are tied to a Canva account, and the design URL opens only if it was shared. |
| L2 | supplementary.pdf, text layer | pdftotext char offsets about 39512, 62271 and 167876; supplementary.tex lines 885, 892, 907, 1200-1203, 1392, 1677, 1987, 2914, 2934, 3112, 3120-3123 | Protocol IDs `ai-***-frontier-baseline-v3`, `ai-***-frontier-context-mapping-v3` and `ai-***-nplayer-matched-hosted-confirmatory-v1`, plus about 12 repository paths (`scr***/analyze_baseline_replication.py`, `res***/cross_model_pilot_synthesis/data/`, `res***/derived/seat_confound/`, `scr***/verify_matched_nplayer_design.py`, ...). Each string occurs in tracked files of the public repository (44, 18 and 40 tracked files respectively for the three protocol IDs). The protocol IDs also contain the Kaggle Benchmark task slugs `ai-***-frontier-context-mapping` and `ai-***-nplayer-matched`. | Medium while the repository is public. One web or GitHub search goes from the supplement to the repository, and the repository holds the author list (R1). If the Kaggle tasks are published, their owner handle is visible too. **I did not verify whether the tasks are published.** |
| L3 | paper.pdf, references | main.tex:431-432, `\citep{huynhUnderstanding2025,huynhPayoffScaling2026}` | Two 16-author self-citations (arXiv 2512.07462 and 2601.19082). Their author lists overlap 8 of the 12 hidden authors (Huy***, Dao***, Ngu***-Lam, Pha***, Tra***, Le ***, Tra*** Le Hong, Han***). The source study by Fer*** Domingos & Han*** is cited as well. | Low. All of these are cited in the third person ("later work adds ..."), which double-blind rules allow. Keep "we"/"our prior work" wording out of any future revision. |
| L4 | both PDFs, object graph and XMP | XMP `CreateDate ...+07:00`; every embedded Matplotlib `PTEX.InfoDict` `/CreationDate (D:2026091...+07'00')` | Timezone UTC+7 | Low. It reveals the region only. |
| L5 | both PDFs | figure InfoDicts; XMP `dc:source` | `/Creator (AI *** publication figure generator)`, `/Subject (AI *** empirical research figure)`, `dc:source ai_***_paper.tex` | Low. This is the repository's vocabulary, not the repository name. |
| L6 | both PDFs, page 1 | header | `Submission Id: TBD` (from `paper/submission_id.tex`) | Not an anonymity leak. It is a submission-gate item. |

### 2b. Outside the bundle but reachable from it (public repository)

`gh repo view` confirms that `github.com/tru***/AI_Race_Experiment` has visibility **PUBLIC**. The local `origin/main` is `6f54ecd`.

| # | file (tracked, public) | location | string (redacted) | severity |
|---|---|---|---|---|
| R1 | `paper/main.tex` | lines 84-179 (commented), **225-290 (active, hidden only by the `anonymous` class option)**, 1048-1050 (commented acknowledgements) | Full author list with emails `231***@student.hcm***.edu.vn` (5), `tru***@hcm***.edu.vn` and 3 more HCMUT addresses, `elias.fer***@vub.be`, `lht***@hcm***.edu.vn`, `T.H***@tees.ac.uk`; grant `EPS***` `EP/***`. The exact submitted PDFs (same SHA-256 `ad867d95...` and `c032ca3c...`) are tracked in the same repository, so searching the title "Mor*** Than the Risk" finds the repository. | **High** |
| R2 | `docs/scripted-opponent-account-map.md`, `docs/kaggle-account-inventory-audit-2026-09-13.md` | whole files | The Account A-E to handle map (`kit***`, `hun***`, `tnk***`, `tru***`, `fou***`) and 20 Kaggle identities (`min***`, `boy***`, `trn***` x3, `osd***`, `dao***`, `chi***` x3, `chu***`, `kak***`, `ton***`, `tru***`, `vin***`, ...). | **High** for the double-blind review. Also, supplementary.tex:1506 ("the mapping is retained privately") is untrue, although that table sits inside `\begin{comment}` and is not rendered. |
| R3 | `results/frontier/scripted_opponent_campaign/**`, `results/derived/scripted_opponent_campaign/**`, `results/derived/nplayer_matched_campaign/**`, `results/failed_runs/scripted_opponent_completion_20260912.json` | receipts and derived JSON/CSV | Raw handles. Counts in `frontier/scripted_opponent_campaign`: `tnk***` 12, `hun***` 11, `kit***` 10, `fou***` 8, `tru***` 7, `trn***` 4, `boy***` 3, `min***` 2. The rendered supplement (supplementary.tex:1401) says the assignment "is recorded in anonymised receipts". | Medium. The claim is inaccurate against the repository, and the receipts de-anonymise. |
| R4 | `LICENSE`; git history | LICENSE line 3; commit authors | `Copyright (c) 2026 tru***`. Commit identities: `tec***` <`duy***@gmail.com`>, `231***` <`...+tru***@users.noreply.github.com`>, `Phu***` <`...@MacBook-Pro-cua-Phu-Quy.local`>, `Ngu***` <`ngu***@gmail.com`>, `Phạ***` <`pha***@gmail.com`>, `Đào***` <`...+tec***@users.noreply.github.com`>, `lph***` <`lep***@gmail.com`> | High while the repository is public. It cannot be fixed without rewriting history. |
| R5 | `kaggle/impact_kernel/kernel-metadata.json`, `dataset-metadata.json` | line 2, line 12 | `dao***/ai-race-impact-admission`, `dao***/ai-race-admission-source` | Medium, latent (see §1) |
| R6 | `scripts/` (7 `tru***`, 5 `dao***`, 3 `PhD***` path strings, 2 `hcm***`), `results/derived/frontier_context_mapping_campaign_v3` (2 `PhD***` paths, untracked), `results/frontier/{admission,baseline}_campaign_v6` READMEs and reports (`dao***`, 3 each), `ai_race/` (1 `fou***`) | various | handles and local paths | Medium, and only if code is shipped. A sanitiser is required first (see §5). |

### 2c. Checked and clean (confirms or corrects the brief)

- **GCP key.** No file matches `AIza[0-9A-Za-z_-]{35}` in this working tree, and `git log --all -G` finds no such string in any commit. `results/frontier/gemini/` does not exist in this tree and is ignored at `.gitignore:168`. The key described in the brief lives in some other tree, if anywhere; I could not verify it from here.
- **`_id_mapping_DO_NOT_DEPOSIT.csv`**, and its `__MACOSX/._` twin, are under `results/public_dataset/`. They are untracked, were never committed, and are not in the bundle. They are excluded only by the blanket rule `/results/**` (`.gitignore:22`) and by a name filter in `scripts/build_results_catalog.py:92,147`. A future `!/results/public_dataset/**` negation would publish them.
- The `hf_` hit at `vendor/FAIRGAME/kaggle_notebook_online.py:60` is a placeholder made of one repeated character. It is a false positive.
- The run artefacts contain no `MODEL_PROXY_*` values, bearer tokens or `api_key`/`token` fields.
- `PTEX.FileName` entries are relative (`../figures/paper/*.pdf`). There are no embedded files. The one PNG carries only `Software` and `dpi`.
- The Account A-E table in the supplement (lines 1498-1510) is inside `\begin{comment}` and is not rendered.

## 3. Scanner gaps (`scripts/anonymity_scan.py`, SCANNER_VERSION 3)

| # | gap | effect today |
|---|---|---|
| G1 | 14 of the 20 inventoried Kaggle handles are not listed: `min***`, `boy***`, `trn***`(thtnhi), `osd***`, `trn***`(nguynchis), `trn***`(gbotrn), `chi***`(sboiz), `chu***`, `kak***`, `ton***`, `tru***`(nkdabest), `vin***`, `chi***`(boiz), `chi***`(nguyentran). Email local parts with no domain (`duy***`, `pha***`, `lep***`) are also not listed. | Would miss a relabelled table row or a receipt quoted in an appendix. |
| G2 | There are no patterns for `tees.ac.uk`, `vub.be`, "Tee***side", "Vri*** Universiteit", "Ho Chi Minh City University of (Science\|Technology)", bare `HCMUT`/`HCMUS`, or student IDs `2312\d{4}`. Author surnames are excluded on purpose, and nothing replaces them with a positive check that the byline rendered anonymously. | A switch to the non-anonymous `\documentclass` would be caught only through the `hcmus`/`hcmut` email domains. The Teesside and VUB authors would pass. |
| G3 | There are no credential patterns (`AIza`, `sk-`, `ghp_`, `github_pat_`, `hf_`, `KGAT_`, `AKIA`, `-----BEGIN ... PRIVATE KEY`). | Harmless while the bundle is PDF-only. It becomes a real gap once code or data is added to the zip. |
| G4 | Metadata keys are not policed. A non-empty `/Author`, `/Keywords` or `/Subject` in an embedded figure's `PTEX.InfoDict` passes, and timezone offsets are not checked. | L1 (the Canva IDs) passed. |
| G5 | The object surface is decoded as latin-1 only. hyperref writes Info strings as UTF-16BE octal escapes (`\376\377\000M...`), and hex strings `<FEFF...>` are not decoded either. I confirmed that `/Title` and `/Creator` in both PDFs use the octal form. | A name in a UTF-16 `/Author`, or in a figure Info dictionary, would evade the object scan. |
| G6 | The text layer is read with `pdftotext -layout` and never normalised. A handle split by hyphenation or a line break passes. The supplement already uses `\discretionary` inside paths. | Split identifiers evade. |
| G7 | There is no repository-fingerprint check. Strings that are unique to the public repository (protocol IDs `ai-race-*-v\d`, `scripts/*.py`, `results/...` paths, the paper title) are not compared against tracked files. | L2 passed. |
| G8 | Zip scan: a non-UTF-8 text member is only noted, not failed. Nested archives and binary formats (parquet, xlsx) are not opened. Notes never change the verdict. | Matters once data ships. |
| G9 | The pattern list covers the paper artefacts only. There is no mode that scans a source tree, such as a code snapshot for an anonymous mirror. | There is no gate for R3 and R6. |

## 4. Reproducibility from the reviewer's side

- **With only the ZIP, a reviewer can re-run nothing.** The ZIP holds one PDF and no code or data, and neither PDF gives a link. The main paper has no code or data availability statement. The supplement refers about 10 times to "the artifact" (supplementary.tex lines 1409, 1507, 1783, 2593, 2931, 2943, ...). It also prints `scripts/...` and `results/...` paths the reviewer cannot open, so it promises material the reviewer cannot reach.
- **The checker works on the authors' tree.** I ran `scripts/verify_manuscript_claims.py` in place with `PYTHONDONTWRITEBYTECODE=1`: **247 of 247 claims verified**.
- **What the checker needs.** Its inputs are the 16 result locations it names (`results/frontier/{admission_campaign_v6, baseline_campaign_v6, context_mapping_campaign_v3, scripted_opponent_campaign}`, `results/derived/*`, `results/open_source/{egt_reproduction, game_understanding_pilot}`, `results/cross_model_pilot_synthesis/data`, `results/failed_runs/...json`), plus `references/source_study_dataset/airace_deidentified_long.csv` (tracked and de-identified), `scripts/` and `ai_race/`. That is **93.9 MB raw and about 5.4 MB deflated**. Together with the 2.4 MiB PDF it fits well inside the 25 MB limit.
- **Gaps a fresh copy would hit:**
  1. `results/derived/frontier_context_mapping_campaign_v3/` and `results/derived/trajectory_diversity_confirmatory/` are gitignored. The checker reads them at lines 268 and 547, so a clean clone stops with `FileNotFoundError` unless `analyze_trajectory_diversity_confirmatory.py` and `analyze_frontier_context_mapping_cross.py` are run first. Nothing documents that order.
  2. The checker crashes on a missing input instead of reporting FAIL. In a committed-tree copy extracted to a deep directory, line 48 raised `ValueError: max() iterable argument is empty`.
  3. The longest tracked path is 211 characters, and 474 paths exceed 150. With Windows `LongPathsEnabled=0` (this machine), 116 files failed to extract into a deep directory. The run-directory names need shortening before shipping.
  4. The README has no reproduce entry point. `requirements.txt` gives version ranges only, and the Python version the checker was run under is not recorded.
  5. The inputs contain handles and local paths (R3, R6), so they cannot ship without pseudonymisation. The pseudonymisation must stay consistent with the Account A-E labels, because some checks count accounts.
- **Human data.** `references/source_study_dataset/` is the source study's released, de-identified dataset. It can ship. The re-identification key never appears in it.

## 5. Recommended actions (ranked)

1. **Make `tru***/AI_Race_Experiment` private for the review period** (GitHub setting, reversible, no history rewrite). This removes R1, R2, R4 and R5 and turns L2 from Medium to Low in one step. Deleting files from `main` would not help, because git history keeps the author block, LICENSE and commit identities. The authors must decide this. It is the only change that closes the High items.
2. **Ship the code and data inside `supplementary.zip`** (about 8 MB total) as the primary route, so no external link is needed. Build it from `git archive HEAD`, then:
   - add the two gitignored derived directories;
   - pseudonymise every handle to the existing Account A-E scheme, extended to the other 15 identities;
   - strip `PhD***`/`C:\Users` paths;
   - exclude `docs/kaggle-account-*`, `docs/scripted-opponent-account-map.md`, `kaggle/impact_kernel/*metadata.json`, `results/public_dataset/`, `.git`, `paper/reviews/`, and every `*DO_NOT_DEPOSIT*`;
   - replace the LICENSE copyright line with "Anonymous authors";
   - scan every member with the extended patterns (step 4).

   Optional mirror: push the same sanitised snapshot to a **new** repository with a neutral name, and publish it through **anonymous.4open.science** with a term list covering all names, handles and institutions. Link it as `https://anonymous.4open.science/r/<id>`. Never use the original repository as the source, because its name and owner are identifying.
3. **Add an availability paragraph to the main paper**, for example: *"Code, the de-identified run records and a checker that recomputes all 247 reported quantities from them are provided in the supplementary archive (and at an anonymised repository, URL). They will be released publicly under an open licence with the camera-ready version."* Point the supplement's "retained with the artifact" wording at that archive. Correct supplementary.tex:1401 ("anonymised receipts") so it describes the shipped, pseudonymised receipts.
4. **Extend `anonymity_scan.py`** to close G1-G8:
   - the full handle list and the institution and student-ID patterns;
   - a positive check that page 1 reads "Anonymous Author(s)" and that XMP `dc:creator` is anonymous;
   - credential patterns;
   - UTF-16 and hex string decoding;
   - rejection of non-empty `/Author`, `/Keywords` or `/Subject` in any Info dictionary;
   - dehyphenation and whitespace-joined matching;
   - a repository-fingerprint pass (`git grep -F` over tracked files outside `paper/` for every protocol ID and path in the text layer);
   - a source-tree mode for the code snapshot.

   Then bump `SCANNER_VERSION` so the old receipts stop counting.
5. **Remove metadata from the PDFs** (L1, L4, L5): add `\pdfsuppressptexinfo=-1`, `\pdfinfoomitdate=1` and `\pdftrailerid{}` to both preambles. Re-export `figures/paper/ExpOverview.pdf` without the Canva keywords. Pass `metadata={"CreationDate": None, "Creator": None, "Subject": None}` to Matplotlib `savefig`.
6. **If the repository stays public anyway,** replace the protocol IDs and `scripts/`/`results/` paths in the supplement with neutral labels (L2). Also confirm with the Kaggle CLI that no `ai-race-*` Benchmark task or kernel is published.
7. **Harden the checker:** fail cleanly on missing inputs, document the order analysers → checker in the README, and shorten deep run paths to under 150 characters.
8. **Fill `\SubmissionID`** in `paper/submission_id.tex` (L6), rebuild through `scripts/build_publication.py`, and rebuild the bundle, because the parallel edits will change both PDFs.
