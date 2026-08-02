# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

A research codebase that runs the two-player repeated "AI race" of Fernández Domingos and Han (2026) with LLM agents instead of human participants. Two model-controlled companies simultaneously choose SAFE or UNSAFE each round; UNSAFE advances faster and pays more now but accumulates a private setback risk that only bites if you win or tie.

[README.md](README.md) states the canonical mechanism; [PROJECT.md](PROJECT.md) is the research protocol (estimands, validation gates, planned sequence). Both are binding on code changes — the engine is meant to be *paper-faithful*, so changing payoffs, horizons, or risk accounting is a protocol change, not a refactor.

Diagnostic pilots have been run, but confirmatory AI Race inference remains pending. Admitted pilot artifacts must trace to a completed manifest, immutable raw logs, and a fail-closed analyzer. Never pool pilot and confirmatory evidence or generalize checkpoint-scoped audits into claims about subjective understanding, stable preferences, or all LLMs.

## Commands

```bash
pytest                                   # full suite (testpaths=ai_race/tests, pythonpath=.)
pytest ai_race/tests/test_scoring.py     # one file
pytest ai_race/tests/test_scoring.py::test_name -v
pytest -k "seed"                         # one pattern

pip install -e ".[dev]"                  # engine + pytest
pip install -e ".[analysis]"             # scipy/statsmodels/matplotlib for the analyser
pip install -e ".[api,kaggle-benchmark]" # hosted-model + Kaggle Benchmark paths
```

`FAIRGAME/unit_tests/` is vendored upstream and excluded from `testpaths`; it is not part of this project's suite.

Local dry run without any model backend (deterministic mock responses):

```bash
python -m ai_race.runner.run_experiment ai_race/configs/experiment/baseline.json --mock random --output /tmp/smoke
```

Analysis over completed run directories (requires the `analysis` extra):

```bash
python results/scripts/analyze_ai_race.py --input <run-root> --output <derived-dir> [--fit-logit]
```

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

**Analysis.** [ai_race/analysis/metrics.py](ai_race/analysis/metrics.py) is a dependency-free descriptive layer used in tests/notebooks. [results/scripts/analyze_ai_race.py](results/scripts/analyze_ai_race.py) (~3.2k lines) is the real analyser: it validates joins, mechanism arithmetic, CRN blocks, and protocol signatures before emitting any table. [strategy_analysis/classify.py](strategy_analysis/classify.py) does nearest-strategy Hamming classification and keeps ties rather than forcing a unique label.

**FAIRGAME/** is vendored upstream (Apache-2.0, LIST/SOM Research Lab) and reused only for its LLM connectors. Treat it as a dependency: don't refactor it to match project style.

## Invariants worth preserving

- **Seed streams are deliberately separated and derived without Python's salted `hash`.** `_stream_seed` in [game.py](ai_race/engine/game.py) gives horizon (stream 17) and fixed-seat setback (stream 29) disjoint streams; `sampling_seed()` is a third. Game seed is `base_seed + rep` and *independent of the treatment name*, so matched repetitions share horizon and setback draws across the 0.1/0.6/0.9 risk conditions. That common-random-number structure is what the analyser clusters on (`source_run/model/rep`) and what it verifies. Breaking it invalidates the inference.
- **The prompt is hashed.** Under `prompt_version = ai-race-fairgame-v3`, `ai_race/prompts/ai_race_en.txt` must hash to `27086bd80378c25e859d03527a5ae55c1046f231ef7b914db9cb3c3b4fb2df3e` and `ai_race_vi.txt` to `a6d3f738cf58043ae0dadc351cac12da07bd60778317b0566d743f5e40a77510`; the pair is frozen in `CANONICAL_PROMPT_SHA256_BY_TEMPLATE` and enforced by `ai_race/tests/test_prompt_contract.py`. Any edit — including whitespace — requires a new `promptVersion` and a new table entry; the analyser rejects modified text relabelled as v3. Runs under different prompt versions cannot be pooled.
- **The prompt must never reveal the terminal round.** The horizon is pre-sampled from a separate RNG and is hidden by design.
- **`run_phase` gates pooling.** The checked-in baseline is `pilot`. Primary analysis accepts only a single `confirmatory` phase with `run_manifest.status="completed"`; the audit overrides (`--allow-nonconfirmatory-runs`, `--allow-nonfinal-runs`, `--allow-mixed-protocols`, `--allow-noncanonical-mechanism`) stratify output but never promote pilot data to evidence.
- **Parse failures contaminate the whole race.** One `parse_failed=true` decision excludes the entire race from every behavioural estimand, because the Safe fallback propagates into later state. Don't loosen `parse_action` to raise the apparent success rate.
- Exactly two players everywhere. `GameConfig` rejects `nPlayers != 2`, and scoring/recording assume two-element lists.

## Analyzing LLM behavioral results

When asked to "analyze" or "visualize" a run under `results/` (2-player or N-player), match the
rigor already established in `results/reports/frontier/` (`results/scripts/analyze_ai_race.py`), not just
means/proportions. That baseline includes, wherever the sample size allows it: a cluster-robust
panel logistic regression (cluster on the CRN block — pair or `rep`, whichever repetitions share
common-random-number draws), nearest-strategy classification by Hamming distance (AS/AU/CS[/CAS]),
confidence intervals on every reported proportion (not point estimates alone), and an explicit
theory-vs-experiment comparison against `ai_race/theory/` or `N-Player/theory/` — search over the
free parameter (e.g. beta) for the best qualitative fit rather than checking one arbitrarily-picked
value. Persona/small-N cells that can't support inference (zero variance, <5 independent races)
stay strictly descriptive — say so explicitly rather than fitting a model that will silently fail
or mean nothing. See `results/reports/nplayer/report.md`'s "Đối chiếu độ sâu" table for a worked example
of holding an N-player pilot to this same bar, and what stayed descriptive-only and why.

## Conventions

- Python ≥3.10, `from __future__ import annotations`, dataclasses for records, module docstrings that state *why* a design is the way it is. Comments explain non-obvious protocol reasoning, not mechanics.
- Adding a treatment = a new `ai_race/configs/game/*.json`, not a code change. Adding an experiment = a new `configs/experiment/*.json`.
- New `TurnRecord`/`GameResult` fields must be reflected in [results/README.md](results/README.md)'s schema section and in the analyser's validation, or completed runs will fail their audit.
- Some docs under `kaggle/` are written in Vietnamese; match the language of the file you are editing.
