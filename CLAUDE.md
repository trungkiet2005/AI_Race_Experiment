# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

**Venue:** AAMAS 2027, Hanoi, Vietnam, 3 to 7 May 2027.

`paper/main.tex` and `paper/supplementary.tex` are both built on the AAMAS 2027
sigconf template: `\documentclass[sigconf,anonymous]{aamas}` with
`\acmConference[AAMAS '27]{... (AAMAS 2027)}{May 3 -- 7, 2027}{Hanoi, Vietnam}`
and `\copyrightyear{2027}`. AAMAS 2027 allows at most **8 pages of main
content**, with unlimited *additional* pages for references only. Supplementary
material is a single ZIP of at most 25 MB, and reviewers are not obliged to read
it, so nothing a claim depends on may live only there. No submission date is
recorded in this repository, so check the call before planning against one.

A research codebase that runs the two-player repeated "AI race" of Fernández Domingos and Han (2026) with LLM agents instead of human participants. Two model-controlled companies simultaneously choose SAFE or UNSAFE each round; UNSAFE advances faster and pays more now but accumulates a private setback risk that only bites if you win or tie.

[README.md](README.md) states the canonical mechanism; [PROJECT.md](PROJECT.md) is the research protocol (estimands, validation gates, planned sequence). Both are binding on code changes — the engine is meant to be *paper-faithful*, so changing payoffs, horizons, or risk accounting is a protocol change, not a refactor.

Diagnostic pilots have been run. **Nine endpoint routes are now audited and five are admitted** (`results/frontier/admission_campaign_v6/`, protocol `ai-race-frontier-admission-v6`, 20 frozen probes at three repetitions, 60 retained rows per route). Admitted: `google/gemini-3-flash-preview` 93.3%, `anthropic/claude-opus-5@default` 91.7%, `openai/gpt-5.4-2026-03-05` 90.0%, `openai/gpt-5.5-2026-04-23` 90.0%, `anthropic/claude-sonnet-5@default` 85.0%. Refused: `google/gemini-3.1-flash-lite-preview` 80.0%, `openai/gpt-5.4-mini-2026-03-17` 75.0%, `google/gemini-3.5-flash-lite` 65.0%, `openai/gpt-5.4-nano-2026-03-17` 51.7%. Admission needs all three gates (overall ≥ 0.80, `state_reconstruction` ≥ 0.75, `terminal_scoring` ≥ 0.75), which is why Gemini 3.1 Flash-Lite is refused at exactly 0.80 overall: its `state_reconstruction` is 0.7333. `expected_payoff` is recorded but diagnostic-only and never gates; it is the weakest domain on every one of the nine routes, so no analysis may lean on a route's expected-payoff arithmetic.

A matched gameplay baseline now exists for **all nine** routes (`results/frontier/baseline_campaign_v6/`, protocol `ai-race-frontier-baseline-v3`, `run_phase = confirmatory`): 30 races, 558 decisions, zero parse failures and zero parse retries each, 10 races per risk cell. `google/gemini-3.5-flash-lite` first failed at the transport layer with 0 races; that attempt is retained under `failed_runs/` as a failure record, never as behaviour, and the successful re-run sits at task version 4 with the reasoning-budget argument omitted per the 2026-09-09 amendment. It is the one route in this campaign whose decoding contract differs, so any table that puts it beside the others must say so. Two contract facts must travel with any citation of this campaign, because both are written in the manifests: the effective temperature is **not confirmed** (0.7 requested, not forwarded by the SDK), and the sampling seed is confirmed only as *requested* — stripped by the SDK for `google/` routes, forwarded but unconfirmed for the OpenAI and Anthropic routes. `results/frontier/baseline_campaign_v6/derived/audit_versus_behaviour.csv` joins admission accuracy to risk response over all nine routes as a descriptive statement, not a population effect: Spearman rho = 0.87 against state-reconstruction accuracy (exact permutation p = 0.004, n = 9), 0.92 with the degenerate Claude Opus 5 step policy set aside. Nine commercial endpoints are not a sample from a population of models, and the admitted and refused groups overlap, so never write this as a causal or estimated effect.

**The fully crossed context/mapping design is now complete for both admitted routes**, `google/gemini-3-flash-preview` (run 1373757) and `anthropic/claude-sonnet-5@default` (run 1504455), at 120 races and 2,232 decisions each with zero parse failures. Because the repetition index is a common-random-number block, each contrast is taken within a repetition and the recorded `horizon_draws_sha256` confirms all 60 paired blocks share one sampled horizon, so the horizon is differenced out. Run `python scripts/analyze_frontier_context_mapping_cross.py`; it refuses any route whose 120 races, ten-per-cell balance, or parse-failure count does not check out, and it never descends into `failed_runs/`. Swapping which opaque code denotes Safe raises Unsafe play by 9.5 points on Gemini 3 Flash (95% interval 5.6 to 13.8) and 8.3 on Claude Sonnet 5 (5.4 to 12.2); the narrative skin raises it by 10.8 (6.0 to 15.6) and 3.4 (0.9 to 7.4) respectively, so the code effect replicates at a similar size while the skin effect is route-specific. All four intervals exclude zero. Two audited routes is enough to say the effect replicates, not to characterise endpoints in general. The earlier Claude attempt that stopped at 106 of 120 races on an HTTP 403 quota refusal is retained, never tabulated, at `results/frontier/context_mapping_campaign_v3/failed_runs/.../1373758` with a `superseded_by.json` pointer, plus `results/failed_runs/context_mapping_claude_1373758_20260909.json`; its 106 races are an unbalanced fragment and are never a mapping result. **Self-play was hiding the mechanism, and the scripted-opponent campaign is what shows it.** Against a rival that always plays Safe, `google/gemini-3-flash-preview` plays Unsafe on 24.7%, 17.2% and 14.0% of its decisions at risk 0.1/0.6/0.9; against a rival that always plays Unsafe, 100.0%, 89.2% and 88.2%. Differenced inside a repetition so the horizon draw is removed, the rival's stance is worth +73.5 [+67.8, +78.7], +67.4 [+57.1, +76.6] and +71.7 [+65.6, +77.5] points. **Do not call the safe-rival arm exploitation.** At 24.7/17.2/14.0% it carries the lowest rates in the campaign, well below self-play, so the route is keeping restraint rather than taking an opportunity; the contrast measures how far the rival's stance moves the route and is named for what it compares. Interval endpoints carry a few tenths of Monte Carlo movement because ten blocks make the resampling distribution coarse, so the verifier pins the point estimate exactly and the endpoints to half a point. The rival's opening move alone, isolated because Conditional Safe and Conditional Unsafe are the same strategy after round one, is worth +25.1 [+20.9, +29.7], +15.8 [+8.3, +23.5] and +11.5 [+4.5, +19.7] points, largest where risk is cheapest. All nine intervals exclude zero and every contrast is paired over ten blocks with the seed pairing verified. **The consequence for the rest of the paper is that the self-play rate is not a measurement of how this route treats risk.** In self-play the same route reads 98.9%, 73.1% and 60.2%, which is an equilibrium of two copies escalating each other; once the rival is fixed and safe, the route's response to stated risk is monotone and its level is low. Bound it honestly: one route, one game, one prompt version, and a scripted rival establishes what the route does against that rival rather than against a population of opponents. The design is causal by construction because the rival cannot be influenced by the route, the route never learns the rival is scripted, and the seat is counterbalanced five and five; ingestion replays every rival move from the route's own history and refuses the cell on a single deviation. **The matched group-size grid is complete: four group sizes by three risk levels, 12 cells, 120 races, 3,696 decisions, 0 parse failures.** Unsafe play rises monotonically with the number of competitors on `google/gemini-3-flash-preview` wherever play is off the boundary. At risk 0.6: 68.8 / 80.3 / 98.3 / 100.0%, paired contrasts +11.1, +28.2, +31.0. At risk 0.9: 58.0 / 68.9 / 83.8 / 96.1%, paired contrasts +11.0, +28.1, +40.0. Every interval excludes zero and the horizon draw is verified shared across all ten blocks. **The three- and four-company steps agree to about a tenth of a point across the two risk levels** (+11.1 vs +11.0, +28.2 vs +28.1) even though those levels were collected on different days and different identities; that agreement is the strongest internal evidence the sweep carries. The five-company contrast differs (+31.0 vs +40.0) only because the risk-0.6 cell is at 100.0% and its contrast is therefore truncated by the ceiling, so quote the risk-0.9 figure for the five-company step. At risk 0.1 every cell is at 100.0% and every contrast is exactly zero: the route is already at the ceiling before a competitor is added, which means the design has no room in which an effect could appear, **not** that group size stops mattering at low risk. Report zero-width intervals as an absence of variation, never as precision. Two things bound it and must travel with any citation: it is one route; group size is not separable from the stage payoff or the prompt length, because the group-count rule and the state description both depend on how many companies there are; and each cell was collected whole on one of several Kaggle identities because no identity sustains the whole grid, by an assignment declared in `docs/matched-nplayer-collection-plan-2026-09-10.md` and its `-extension.md` before the first run rather than by retrying until something passed, with the assignment rotated across risk levels so no identity is confounded with a group size. Each cell's bootstrap generator is derived from that cell alone (`cell_rng`), so collecting one more risk level cannot move an interval already reported. The two full-sweep attempts that quota refused are `results/failed_runs/nplayer_matched_daosyduyminh_20260910.json` and `nplayer_matched_trungkiet_20260910.json`. The N=3 frontier rerun is still not admitted; the retained records are `results/failed_runs/nplayer_baseline_n3_20260908.json` (non-retryable model request failures, no admissible raw output) and `results/failed_runs/nplayer_baseline_n3_v5_20260908.json` (HTTP 403 Model Proxy quota refusal at task-creation validation, 0 races). A quota or transport refusal is never model evidence. Persona and language reruns are not part of the admitted frontier campaign. Six of the seven checkpoints in the human-diversity comparison now carry an admission verdict; `gpt-5-nano` is no longer offered on the audited identity and can never receive one, which bounds that comparison by route availability rather than by design. Every admitted artifact must trace to a completed manifest, immutable raw logs, and a fail-closed analyzer. Never pool pilot and confirmatory evidence or generalize checkpoint-scoped audits into claims about subjective understanding, stable preferences, or all LLMs.

## Commands

```bash
pytest                                   # full suite (testpaths=ai_race/tests, pythonpath=. and vendor)
pytest ai_race/tests/test_scoring.py     # one file
pytest ai_race/tests/test_scoring.py::test_name -v
pytest -k "seed"                         # one pattern

pip install -e ".[dev]"                  # engine + pytest
pip install -e ".[analysis]"             # scipy/statsmodels/matplotlib for the analyser
pip install -e ".[api,kaggle-benchmark]" # hosted-model + Kaggle Benchmark paths
```

Paper build outputs:

```bash
python scripts/build_publication_figures.py
python scripts/build_supplementary_figures.py  # optional: supplement only
python scripts/build_publication.py --paper-only
python scripts/check_publication.py --allow-placeholder-id
python scripts/build_figure_gallery.py
```

The manuscript sources are `paper/main.tex` and `paper/supplementary.tex`.
The shared anonymous submission ID is configured once in
`paper/submission_id.tex`.
Their final PDFs must always be written to `paper/ai_race_paper.pdf` and
`paper/ai_race_supplementary.pdf`. LaTeX auxiliary files belong under the
ignored `results/_build/latex/current/` directory. The build script also keeps
copies under `results/artifacts/publication/` because submission and release
tools read that mirror. The QA command should be run after every build; omit
`--allow-placeholder-id` once AAMAS assigns the anonymous submission ID.

`scripts/build_publication.py --paper-only` validates the four protected
author-supplied artworks and regenerates the remaining slots in the canonical
fourteen-figure set (eight main-paper and six supplementary) before compiling
the paper and supplementary PDF. The final PDFs therefore
stay in `paper/` for easy discovery, while the mirrored copies remain under
`results/artifacts/publication/` for release tooling. Do not edit exported
figure files by hand; change the generator or its checked-in source table and
regenerate the generated set. The protected artworks are restored from their
recorded source revision when needed and are never targets of automated
writers.

All project figures have one browsing entry point at `figures/`. Use
`figures/paper/` for assets referenced by the manuscript and deck,
`figures/gallery/` for the full candidate and redraw collection, and
`figures/diagnostics/` for hash-tracked copies of additional analysis exports.
Run `scripts/build_figure_gallery.py` after generating figures. The original
files under `results/` and `analysis/` remain the provenance source, and a
gallery copy never upgrades an exploratory or diagnostic evidence class.

`vendor/FAIRGAME/unit_tests/` is vendored upstream and excluded from `testpaths`; it is not part of this project's suite.

Local dry run without any model backend (deterministic mock responses):

```bash
python -m ai_race.runner.run_experiment ai_race/configs/experiment/baseline.json --mock random --output /tmp/smoke
```

Analysis over completed run directories (requires the `analysis` extra):

```bash
python scripts/analyze_ai_race.py --input <run-root> --output <derived-dir> [--fit-logit]
```

Campaign and reviewer-question analysers added with the 2026-09-09 revision. Each
one owns exactly one derived artifact and is fail-closed or provenance-stamped;
run the generator, never hand-edit its output:

```bash
python scripts/analyze_frontier_admission_campaign.py      # 9-route admission table, refuses a row whose artefacts fail a structural check
python scripts/analyze_audit_versus_behaviour.py           # joins admission accuracy to gameplay risk response over the 8 baseline routes
python scripts/analyze_egt_beta_sensitivity.py             # sweeps EGT selection strength beta x mutation rule x risk against the routes
python scripts/analyze_human_archetype_k_sensitivity.py    # k-sweep validity indices plus a bootstrap ARI stability curve for the human archetypes
python scripts/analyze_elicited_risk_by_archetype.py       # Kruskal-Wallis on elicited Eckel-Grossman risk across the four published archetypes
python scripts/build_diversity_figure.py                   # draws the main-paper trajectory-diversity panel from the rarefaction table
python scripts/analyze_frontier_context_mapping_cross.py   # repetition-paired presentation effects on both crossed routes, fail-closed on an unbalanced design
python scripts/analyze_nplayer_matched.py                  # matched group-size comparison, refuses a risk level missing any group size
python scripts/verify_matched_nplayer_design.py            # offline design check; run before pushing any matched cell
python scripts/verify_manuscript_claims.py                 # recomputes all 68 headline numbers from artifacts; exits non-zero on any mismatch
python results/cross_model_pilot_synthesis/analyze_heterogeneity_test.py --roster five
```

The last one regenerates the nested-logit heterogeneity statistic under any named
checkpoint roster, so every historically reported version of that test stays
reproducible instead of being re-derived by hand.

## Execution policy

Experiments do not run on this workstation — there is no GPU and no configured API path. Static review and `pytest` are what happens locally; behavioral and GPU validation happen on Kaggle:

- Open-weight models: [kaggle/experiments/baseline.py](kaggle/experiments/baseline.py), a `# %%`-celled GPU notebook that imports `ai_race` and `FAIRGAME` from the mounted repo.
- Frontier/API models: [kaggle/benchmarks/ai_race_baseline.py](kaggle/benchmarks/ai_race_baseline.py), a self-contained `kaggle_benchmarks` task (slug `ai-race-baseline`) that reimplements the mechanism without importing the package.

Kaggle Benchmark push/run/download are **checkpointed** operations: run one command, show output, stop, and wait for the user before the next. See [kaggle/benchmarks/README.md](kaggle/benchmarks/README.md) and the `write-kaggle-benchmarks` skill in [.agents/skills/](.agents/skills/). Do not invent a kernel slug when no `kernel-metadata.json` exists.

## Architecture

Data flows config → games → lockstep batch → journal → analyser.

**Engine** ([ai_race/engine/](ai_race/engine/)) is pure and backend-agnostic.
- [state.py](ai_race/engine/state.py) — `Action`, `GameConfig` (validates the mechanism in `__post_init__`), and the two flat output records `TurnRecord` / `GameResult`. JSON configs are camelCase; Python attributes are snake_case; `GameConfig.from_dict` is the only bridge.
- [game.py](ai_race/engine/game.py) — `AIRaceGame` exposes a *stepwise* API (`build_round_prompts()` → `apply_round_responses()`) so many races advance in lockstep against one batched backend. Both prompts in a round render from the same pre-action snapshot; that is what makes choices simultaneous.
- [scoring.py](ai_race/engine/scoring.py) — pure functions for stage payoffs, effective private risk (`max_private_risk * unsafe_count / rounds_played`), prize allocation, and winner-only setback.
- [round.py](ai_race/engine/round.py) — prompt assembly plus a deliberately strict parser: a response must be exactly one `ACTION: SAFE|UNSAFE` line. It never rescues an answer by scanning prose; a failure falls back to Safe and is logged as `parse_failed`.
- [prompt.py](ai_race/engine/prompt.py), [strategies.py](ai_race/engine/strategies.py) — template rendering and the canonical AS/AU/CS/CAS reference strategies.

**Runner** ([ai_race/runner/](ai_race/runner/)). `build_games_for_model` expands treatment × language × repetition into `AIRaceGame`s; `run_games_batched` advances every unfinished race one round at a time, retries only the parse-failed prompts from the *unchanged* prompt with a fresh deterministic retry seed, and calls `on_round_complete`. `run_experiment` wires in the backend, one result directory per model.

**Backends** ([ai_race/models/factory.py](ai_race/models/factory.py)) return a uniform `send_batch(prompts, seeds=None)`. Offline goes to `FAIRGAME.src.llm_connectors.local_vllm_connector` (vLLM or transformers); hosted goes to `ChatModelFactory`. `--mock` bypasses both. The `seeds=` keyword is passed unconditionally and a `TypeError` is allowed to surface — silently dropping seeds would make the recorded provenance false.

**Recording** ([ai_race/dataio/recorder.py](ai_race/dataio/recorder.py)). `RunJournal` appends after every completed round, so an interrupted run leaves incomplete races visible in `turns.jsonl` with no matching terminal CSV row. Output per model: `turns.jsonl` (one row per player decision), `races.csv`, `players.csv`, `run_manifest.json`.

**Analysis.** [ai_race/analysis/metrics.py](ai_race/analysis/metrics.py) is a dependency-free descriptive layer used in tests/notebooks. [scripts/analyze_ai_race.py](scripts/analyze_ai_race.py) (~3.2k lines) is the real analyser: it validates joins, mechanism arithmetic, CRN blocks, and protocol signatures before emitting any table. [analysis/strategy/classify.py](analysis/strategy/classify.py) does nearest-strategy Hamming classification and keeps ties rather than forcing a unique label. Alongside it sit six single-purpose analysers, one derived artifact each: [analyze_frontier_admission_campaign.py](scripts/analyze_frontier_admission_campaign.py) derives the nine-route admission table and refuses to tabulate a route whose row count, per-domain sums, recomputed accuracy, recomputed admit flag, `protocol_id`, or probe-bank and rules-context hashes do not check out; [analyze_audit_versus_behaviour.py](scripts/analyze_audit_versus_behaviour.py) joins measured probe accuracy to measured risk response route by route, clustering the bootstrap on the race; [analyze_egt_beta_sensitivity.py](scripts/analyze_egt_beta_sensitivity.py) layers a selection-strength and mutation-rule sweep on the existing EGT reconstruction so the theory-versus-LLM gap can be checked for dependence on beta=2; [analyze_human_archetype_k_sensitivity.py](scripts/analyze_human_archetype_k_sensitivity.py) answers "why k=4" with internal validity indices plus a bootstrap adjusted-Rand stability curve on the paper's own 341x5 standardized matrix, imported from the clustering generator rather than reimplemented; [analyze_elicited_risk_by_archetype.py](scripts/analyze_elicited_risk_by_archetype.py) computes the elicited-risk-by-archetype test the supplement had asserted without code, reusing the published labels and refusing to run if the label sizes are not the published multiset; and [build_diversity_figure.py](scripts/build_diversity_figure.py) draws the diversity panel from the rarefaction table only, so presentation can never move a published value.

**vendor/FAIRGAME/** is vendored upstream (Apache-2.0, LIST/SOM Research Lab) and reused only for its LLM connectors. Treat it as a dependency: don't refactor it to match project style.

## Invariants worth preserving

- **Seed streams are deliberately separated and derived without Python's salted `hash`.** `_stream_seed` in [game.py](ai_race/engine/game.py) gives horizon (stream 17) and fixed-seat setback (stream 29) disjoint streams; `sampling_seed()` is a third. Game seed is `base_seed + rep` and *independent of the treatment name*, so matched repetitions share horizon and setback draws across the 0.1/0.6/0.9 risk conditions. That common-random-number structure is what the analyser clusters on (`source_run/model/rep`) and what it verifies. Breaking it invalidates the inference.
- **The prompt is hashed.** Under `prompt_version = ai-race-fairgame-v3`, `ai_race/prompts/ai_race_en.txt` must hash to `27086bd80378c25e859d03527a5ae55c1046f231ef7b914db9cb3c3b4fb2df3e` and `ai_race_vi.txt` to `a6d3f738cf58043ae0dadc351cac12da07bd60778317b0566d743f5e40a77510`; the pair is frozen in `CANONICAL_PROMPT_SHA256_BY_TEMPLATE` and enforced by `ai_race/tests/test_prompt_contract.py`. Any edit — including whitespace — requires a new `promptVersion` and a new table entry; the analyser rejects modified text relabelled as v3. Runs under different prompt versions cannot be pooled.
- **The prompt must never reveal the terminal round.** The horizon is pre-sampled from a separate RNG and is hidden by design.
- **`run_phase` gates pooling.** The checked-in baseline is `pilot`. Primary analysis accepts only a single `confirmatory` phase with `run_manifest.status="completed"`; the audit overrides (`--allow-nonconfirmatory-runs`, `--allow-nonfinal-runs`, `--allow-mixed-protocols`, `--allow-noncanonical-mechanism`) stratify output but never promote pilot data to evidence.
- **Parse failures contaminate the whole race.** One `parse_failed=true` decision excludes the entire race from every behavioural estimand, because the Safe fallback propagates into later state. Don't loosen `parse_action` to raise the apparent success rate.
- Exactly two players everywhere. `GameConfig` rejects `nPlayers != 2`, and scoring/recording assume two-element lists.

## Analyzing LLM behavioral results

When asked to "analyze" or "visualize" a run under `results/` (2-player or N-player), match the
rigor already established in `results/reports/frontier/` (`scripts/analyze_ai_race.py`), not just
means/proportions. That baseline includes, wherever the sample size allows it: a cluster-robust
panel logistic regression (cluster on the CRN block — pair or `rep`, whichever repetitions share
common-random-number draws), nearest-strategy classification by Hamming distance (AS/AU/CS[/CAS]),
confidence intervals on every reported proportion (not point estimates alone), and an explicit
theory-vs-experiment comparison against `ai_race/theory/` or `ai_race/theory_nplayer/` — search over the
free parameter (e.g. beta) for the best qualitative fit rather than checking one arbitrarily-picked
value. Persona/small-N cells that can't support inference (zero variance, <5 independent races)
stay strictly descriptive — say so explicitly rather than fitting a model that will silently fail
or mean nothing. See `results/reports/nplayer/report.md`'s "Đối chiếu độ sâu" table for a worked example
of holding an N-player pilot to this same bar, and what stayed descriptive-only and why.

### Two previously reported statistics, resolved 2026-09-09

Both were unreproducible when checked, and they resolved in opposite directions.
Do not re-derive either one by hand:

- **The nested-logit heterogeneity statistic is real.** chi-squared(10) = 354.67,
  p = 4.1e-70, is the B-versus-C likelihood-ratio test on the five-checkpoint
  neutral-lane roster (4,464 decisions, 240 races). It is regenerable with
  `python results/cross_model_pilot_synthesis/analyze_heterogeneity_test.py --roster five`
  and lands in `results/cross_model_pilot_synthesis/data/cross_model_heterogeneity_test.json`.
  The default roster there is now `nine`, whose same test is 2,317.38 on 18 df, so
  always name the roster with the number.
- **The Kruskal-Wallis statistic H = 21.95 with n = 286 had no source and is
  withdrawn.** The correct recomputation is a null:
  H = 4.540, df = 3, p = 0.209, n = 341, epsilon-squared = 0.0046, in
  `results/cross_model_pilot_synthesis/data/elicited_risk_by_archetype.json`.
  Elicited Eckel-Grossman risk does not differ detectably across the four
  archetypes. Report the null; do not restore the old number.

## Conventions

- Python ≥3.10, `from __future__ import annotations`, dataclasses for records, module docstrings that state *why* a design is the way it is. Comments explain non-obvious protocol reasoning, not mechanics.
- Adding a treatment = a new `ai_race/configs/game/*.json`, not a code change. Adding an experiment = a new `configs/experiment/*.json`.
- New `TurnRecord`/`GameResult` fields must be reflected in [results/README.md](results/README.md)'s schema section and in the analyser's validation, or completed runs will fail their audit.
- Some docs under `kaggle/` are written in Vietnamese; match the language of the file you are editing.

## Reviewer revision frontier campaign

The per-endpoint admission task is
[`kaggle/benchmarks/ai_race_frontier_admission.py`](kaggle/benchmarks/ai_race_frontier_admission.py).
It must complete the frozen rule, state, terminal, and expected-payoff audit
before a hosted route enters behavioural claims. The candidate route registry is
[`docs/frontier-model-registry.json`](docs/frontier-model-registry.json), and the
review-to-evidence matrix is
[`docs/frontier-evidence-collection-protocol.md`](docs/frontier-evidence-collection-protocol.md).

Current campaign artifacts, all on the single configured Kaggle identity
`daosyduyminh`:

| Campaign | Protocol | What it holds |
|---|---|---|
| `results/frontier/admission_campaign_v6/` | `ai-race-frontier-admission-v6` | 9 routes audited, 5 admitted; 60 retained rows per route; `derived/admission_campaign_v6.csv`, `.json`, `report.md`; one retained failed attempt |
| `results/frontier/baseline_campaign_v6/` | `ai-race-frontier-baseline-v3` | 9 routes of gameplay, 30 races and 558 decisions each, 0 parse failures; `derived/audit_versus_behaviour.csv`, `.json`; one retained failed attempt whose re-run succeeded |
| `results/frontier/context_mapping_campaign_v3/` | `ai-race-frontier-context-mapping-v3` | both admitted routes, crossed and complete, 120 races and 2,232 decisions each; `derived/context_mapping_cross_marginals.csv`, `.json`; one retained superseded attempt |
| `results/frontier/nplayer_matched_campaign/` | `ai-race-nplayer-matched-hosted-confirmatory-v1` | matched group sizes 2 to 5 at all three risk levels on `google/gemini-3-flash-preview`, 10 races per cell, 0 parse failures; `derived/nplayer_matched_rates.csv`, `.json`; each cell carries a `collection_receipt.json` naming the identity that collected it |
| `results/frontier/baseline_replication/` | `ai-race-frontier-baseline-v3` | an independent repeat of the `google/gemini-3-flash-preview` baseline cell, same protocol, prompt hash, mechanism and 30 seed blocks, different day and identity; largest per-risk difference 1.1 points, risk response 38.7 to 40.3. **Kept outside `baseline_campaign_v6/` on purpose**: both campaign analysers key by `model_route`, so a second run of a represented route placed in that tree would silently displace the reported one rather than raise. Read it with `scripts/analyze_baseline_replication.py` |
| `results/frontier/pilots/` | none | pre-protocol pilots kept as provenance; the 2026-08-01 nine-race run has no `protocol_id` at all, which is exactly why it can never be pooled with or compared against confirmatory runs. No analyser reads this directory |
| `results/frontier/scripted_opponent_campaign/` | `ai-race-scripted-opponent-v1` | the audited route against the four reduced strategies rather than itself; 12 cells (4 strategies x 3 risks), 120 races, 1,116 route decisions, 0 parse failures; `derived/scripted_opponent_rates.csv`, `.json`. **Only rows with `is_route_decision` true are the model's own choices**; the others are the script, and pooling them would report the strategy's behaviour as the route's. Every cell carries a `collection_receipt.json` naming its identity and recording that the rival replayed clean |

Three things about these directories are load-bearing:

- **The protocol was amended in the open on 2026-09-09.** Both frontier tasks used
  to pass `reasoning="none"` to every route. Some routes reject the
  reasoning-budget *argument itself* — naming `reasoning` at all returns HTTP 400
  before a single probe is sampled — which is a transport contract mismatch, not
  model evidence. The tasks now **resolve the budget per route and omit the
  parameter entirely for routes that refuse it**, and the manifest records the
  omission. A route whose resolved contract is still `reasoning="none"` sends a
  byte-identical request and stays poolable with the earlier task versions; a
  route whose parameter was dropped has a different contract, and any table that
  puts it beside the others must say so. In the v6 admission campaign that is
  exactly one route, `google/gemini-3.5-flash-lite` at task version 8. The
  amendment is recorded under "Protocol amendments" in
  `docs/frontier-evidence-collection-protocol.md`; amend there, never silently.
- **`baseline_campaign_v2` and `baseline_campaign_v6` are two separate samples and
  must never be pooled or swapped.** They carry the same `protocol_id` but
  different task versions and different run ids, and the same route gives
  different numbers: `google/gemini-3-flash-preview` plays Unsafe 0.7419 at risk
  0.6 in v2 (run 1371960) and 0.7312 in v6 (run 1395291);
  `anthropic/claude-sonnet-5@default` plays 0.3763 at risk 0.9 in v2 (run 1371961)
  and 0.3226 in v6 (run 1494092). Any table must name the run it read.
- **Failure records stay in the accounting chain.** A failed or superseded attempt
  is kept under the campaign's `failed_runs/` and is never deleted, never
  substituted with a different route, and never tabulated. The derived analysers
  walk only the live task tree and never descend into `failed_runs/`.

## Current execution note

The workstation now has a configured hosted Kaggle Benchmark path for bounded
frontier smoke and confirmatory runs. Heavy model workloads should use Kaggle
or the managed H100 pods. A successful request is not evidence by itself:
retain the raw responses, completed manifest, parser-failure accounting, and
integrity audit before promoting a run into the manuscript.
