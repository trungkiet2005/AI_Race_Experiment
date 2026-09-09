# AI Race LLM Experiment

This repository studies how large language models behave in a repeated, idealised
AI-development race. Two model-controlled companies repeatedly choose between
**Safe** and **Unsafe** development. Unsafe development is faster and pays more in
the current round, but it accumulates a private setback risk for a race winner.

The environment is adapted from Fernández Domingos and Han (2026), _Falling Behind
Drives Unsafe Development in an Idealised AI Race Experiment_ (arXiv:2607.26034).
The source study used human participants; this project is an LLM adaptation and does
not present its future model outputs as a replication of human cognition.

The manuscript targets **AAMAS 2027** (Hanoi, Vietnam, 3 to 7 May 2027): at most
8 pages of main content, unlimited additional pages for references only, and one
supplementary ZIP of at most 25 MB that reviewers are not obliged to read. Both
`paper/main.tex` and `paper/supplementary.tex` use the AAMAS 2027 sigconf
template and copyright block.

## Setup

```bash
python -m venv .venv && . .venv/Scripts/activate   # POSIX: . .venv/bin/activate
pip install -e ".[analysis,dev]"                    # add ",api" for hosted models
pytest                                              # 51 test modules; API tests skip without their extras
```

Runs against hosted providers read credentials from a `.env` file. Copy
[`.env.example`](.env.example) to `.env` and fill in only the providers you use;
`.env` is gitignored. The offline engine needs no credentials at all.

## Canonical game

- Two players choose simultaneously in every round.
- Safe advances `1.0` step; Unsafe advances `1.5` steps.
- Stage payoff matrix (row = own action, column = opponent action):

  | own \ opponent | Safe | Unsafe |
  | -------------- | ---: | -----: |
  | Safe           |  1.0 |    0.6 |
  | Unsafe         |  2.4 |    2.0 |

- The race lasts at least five rounds. After every completed round from round 5
  onward, it stops with probability `0.2`, so the expected duration is 9 rounds.
- The progress leader receives 100 ECU; a tie gives 50 ECU to each player.
- Treatments set maximum private risk to `0.1`, `0.6`, or `0.9`.
- Before each decision, both agents observe both accumulated stage payoffs,
  both current private risks, both race positions, and the preceding revealed
  action profile; same-round choices remain simultaneous and hidden.
- A winner's effective setback probability is
  `max_private_risk * unsafe_actions / rounds_played`. A setback removes that
  player's entire race payoff. A loser keeps stage payoffs and receives no prize.

## Repository layout

```text
ai_race/                 Core engine, prompts, configs, runners, tests, and metrics
                         (theory/ is the two-player model, theory_nplayer/ the N-player one)
analysis/
  fh_analytic/           Human-vs-LLM analytic pipeline and its derived outputs
  strategy/              AS/AU/CS/CAS trajectory classification (importable package)
data/                    Analysis-ready copied view of selected runs (see data/README.md)
kaggle/
  experiments/           GPU notebook source for the offline baseline
  benchmarks/            Kaggle Benchmark task for frontier/API models
  setup/                 Offline-wheel preparation notes and script
references/
  papers/                Curated literature: markdown notes, PDFs, arXiv source bundles
  _citation_audit/       arXiv sources retained while auditing the manuscript citations
  source_study_dataset/  The source study's published de-identified participant table
scripts/                 All operational and analysis scripts: stage runners,
                         analysers, catalog and manifest builders
results/                 Canonical home for raw runs, analyses, reports, QA, and publication PDF mirrors
docs/                    Experiment protocols and run guides
paper/                   Manuscript source, final PDFs, and bounded pilot audit evidence
figures/                 Canonical publication assets and the complete figure-selection gallery
slides/                  AI Race presentation outline, deck source, and design reference
web/                     Browser front-ends: the simulator and the trajectory lab
vendor/FAIRGAME/         Vendored upstream connectors (Apache-2.0, not our code)
```

Legacy Collective Risk outputs and trained strategy artifacts are preserved locally
under `.archive/collective_risk/` and excluded from Git. They are not mixed with the
AI Race result schema.

## Baseline configuration

The paper-faithful baseline is
[`ai_race/configs/experiment/baseline.json`](ai_race/configs/experiment/baseline.json).
It sweeps the three private-risk treatments using neutral companies and a hidden
stochastic horizon. The checked-in configuration is explicitly a three-repetition
`pilot`; it must not be pooled with confirmatory data. Freeze the protocol, set
`runPhase` to `confirmatory`, and choose the preregistered sample size before a full
run.

Expected output for each model:

```text
turns.jsonl      one row per player decision
races.csv        one row per two-player race
players.csv      one row per player-race
run_manifest.json
```

## Hosted-model (API) runs

[`ai_race/configs/experiment/api_baseline.json`](ai_race/configs/experiment/api_baseline.json)
runs the identical treatments through the Kaggle model proxy instead of a local
GPU. It selects `"backend": "proxy"`, and `models` must name routes listed in
`LLMS_AVAILABLE`.

```bash
kaggle benchmarks auth            # refresh MODEL_PROXY_* in .env; the token is short-lived
python -m ai_race.runner.run_experiment ai_race/configs/experiment/api_baseline.json
```

The proxy backend retries transport failures with backoff and then raises. A failed
call is never converted into a Safe action, so an expired token stops the run and
writes a `failed` manifest instead of contaminating the panel. `samplingSeedApplied`
stays `false` because the proxy does not confirm that a forwarded seed was applied.
`"backend": "api"` still routes to the FAIRGAME provider SDK connectors, which need
`API_KEY_OPENAI` / `API_KEY_ANTHROPIC` / `API_KEY_MISTRAL` instead.

## Prompt templates

Prompts in [`ai_race/prompts/`](ai_race/prompts/) follow the FAIRGAME
`resources/game_templates` convention: camelCase placeholders (`{currentPlayerName}`,
`{opponent1}`, `{strategy1}`, `{weight1}`…`{weight4}`, `{history}`, `{currentRound}`)
plus optional blocks written as `{blockName}: [ ... ]`. `intro` is kept only when the
seat carries a persona; `gameLength`, `opponentIntro`, and `communicate` are deleted
in this design, so the horizon stays hidden. Templates keep the strict
`ACTION: SAFE` / `ACTION: UNSAFE` output contract, because parse failures are a
recorded protocol-health measure rather than a rescued free-text answer.

The main behavioral outcomes are Unsafe frequency, response to the opponent's
previous action, pre-decision progress gap, first-round momentum, and winner/loser
Unsafe frequency.

## Run policy

Step-by-step commands for every path — staging the dataset, the two Kaggle routes,
the local proxy route, and the local analysis — are in
[`docs/running-the-experiment.md`](docs/running-the-experiment.md).

The current workstation is not used to execute experiments. Run the offline baseline
from [`kaggle/experiments/baseline.py`](kaggle/experiments/baseline.py) on Kaggle
with GPU enabled. The script is organized with `# %%` cells, copies the read-only
input repository to `/kaggle/working`, loads model inputs sequentially, and writes a
zip archive to Kaggle Output. It imports `ai_race` and `FAIRGAME` from that input, so
the repository must be staged as a Kaggle Dataset and added as an input; the Kaggle
Benchmark task is self-contained and is pushed as a single file instead.

Kaggle Benchmark publication and remote runs are checkpointed operations. See
[`kaggle/benchmarks/README.md`](kaggle/benchmarks/README.md) for the exact
push/status/log commands; do not invent a kernel slug when no
`kernel-metadata.json` is present.

## Research status

- AI Race game mechanics, state, scoring, prompt, logging schema, and baseline
  configuration are implemented.
- CRSD-specific documents, experiments, results, manuscript figures, and slides have
  been removed from the active project.
- **Endpoint admission: nine hosted routes audited, five admitted.** The audit
  (`results/frontier/admission_campaign_v6/`, protocol
  `ai-race-frontier-admission-v6`) puts 20 frozen probes at three repetitions,
  60 retained rows, through every route and requires overall accuracy at least
  0.80 together with state-reconstruction and terminal-scoring accuracy at least
  0.75. Admitted: Gemini 3 Flash 93.3%, Claude Opus 5 91.7%, GPT-5.4 90.0%,
  GPT-5.5 90.0%, Claude Sonnet 5 85.0%. Refused: Gemini 3.1 Flash-Lite 80.0%,
  GPT-5.4 mini 75.0%, Gemini 3.5 Flash-Lite 65.0%, GPT-5.4 nano 51.7%. Expected
  payoff is recorded but never gates admission, and it is the weakest domain on
  every route. A refused route is kept as evidence about the route, not deleted.
- **Gameplay: eight of those routes have a matched confirmatory baseline**
  (`results/frontier/baseline_campaign_v6/`, protocol
  `ai-race-frontier-baseline-v3`): 30 races, 558 decisions and zero parse
  failures each. Gemini 3.5 Flash-Lite failed at the transport layer with zero
  races and is retained as a failure record rather than reported as a result. The
  requested temperature was not forwarded by the SDK and the sampling seed is
  confirmed only as requested, so both are cited as unconfirmed.
- One fully crossed context-and-mapping run is complete, on Gemini 3 Flash, with
  120 races and 2,232 decisions and a passing independent validator. The N-player
  frontier rerun is not admitted; its attempts are kept under
  `results/failed_runs/`.
- Deterministic checks and analysis run locally. Model evidence must run in a
  declared GPU environment or on the audited Kaggle Benchmark identity, with exact
  source, model, decoding, and hardware provenance; the earlier understanding
  pilot used two GreenNode H100 lanes. Open-weight smoke and diagnostic pilots for
  persona sensitivity, prompt-surface sensitivity, and game understanding remain
  historical diagnostics and are never pooled with the frontier campaigns.

See [`PROJECT.md`](PROJECT.md) for research questions, estimands, and validation
criteria.

## Interactive visualization and presentation

The repository includes a responsive protocol website with a deterministic,
educational AI Race simulator. It does not call a model and does not create research
data. Start a local server from the repository root:

```bash
python -m http.server 8000
```

Then open `http://localhost:8000/web/`. The simulator implements simultaneous
Safe/Unsafe decisions, the exact payoff matrix, progress updates, the hidden
stochastic horizon, private terminal risk, a progress chart, and an auditable round
ledger.

The canonical presentation source is
[`slides/ai_race_research_deck.tex`](slides/ai_race_research_deck.tex). The compiled
26-frame PDF is written to `results/artifacts/publication/ai_race_research_deck.pdf`; see
[`slides/README.md`](slides/README.md) for build commands. A browser-based companion
deck is also available at `http://localhost:8000/slides/`.
The versioned multi-prompt robustness protocol, evidence survey, treatment
taxonomy, and paired estimands are documented in
[`docs/prompt-sensitivity-survey.md`](docs/prompt-sensitivity-survey.md). Surface
variants are deliberately excluded from the canonical primary-mechanism pool and
are analyzed with `scripts/analyze_surface_sensitivity.py`.

The complete public GPU handoff contains 11 hash-verified raw/analysis archives
plus expanded derived tables. See
[`results/open_source/gpu_run_archive/`](results/open_source/gpu_run_archive/) and
verify a clean clone with `python scripts/audit_gpu_archives.py
--archive-dir results/open_source/gpu_run_archive`.

## Game-understanding audit

The admitted Qwen2.5 7B F16 pilot contains 685 atomic rule/arithmetic probe outputs
and 60 paired canonical/calculator-card races (1,116 decisions). Rule recall and
stage-payoff lookup were strong, while state-transition and expected-payoff
accuracy were weak. This is checkpoint-scoped pilot evidence, not proof of an
internal world model and not confirmatory evidence about treatment effects.

- protocol and admission gates: [`docs/game-understanding-audit.md`](docs/game-understanding-audit.md)
- bounded results and provenance: [`docs/game-understanding-audit-results.md`](docs/game-understanding-audit-results.md)
- admitted tables and hashes: [`results/open_source/game_understanding_pilot/`](results/open_source/game_understanding_pilot/)
- publication figures and selection gallery: [`figures/`](figures/)
