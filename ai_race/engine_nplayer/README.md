# N-player AI Race engine

Extends the paper-faithful two-player mechanism in `ai_race/engine/` to any
`n_players >= 2`, following Appendix B of the reference paper
(`jair-12225/JAIR-12225-ArticlePDF-25030-1-10-20201122.md`, Han et al. 2020,
"N-player AI Race Definition"). Built as a fully separate package on purpose
— see "Why separate" below — with configs, prompts, and tests of its own.
Nothing under `ai_race/engine/`, `ai_race/dataio/recorder.py`,
`ai_race/runner/`, `ai_race/dataio/config_loader.py`'s `validate_game`/
`validate_agents`, or `results/scripts/analyze_ai_race.py` is touched or
depended on by this module.

## Mechanism

Each round, for a group of `n_players` where `k` chose SAFE:

```
pi_SAFE(k)   = -cost + benefit / (k + speed*(N-k))     if 1 <= k < N
             = -cost + benefit / N                      if k = N
pi_UNSAFE(k) =  speed * benefit / (k + speed*(N-k))     for 0 <= k < N
```

`speed` is both the Unsafe progress multiplier (an Unsafe player advances
`speed` times as fast as a Safe one) and the payoff weight in the formula
above — one number does both jobs, matching the paper. Progress, the hidden
stochastic horizon (`min_rounds` + geometric stop lottery), the race prize,
and private setback risk (`max_private_risk * unsafe_count / rounds_played`,
applied only to whoever ends up in the lead) all keep the exact same shape as
the two-player engine — only the *group size the payoff is computed over*
changes. When multiple players share the lead at the end, the prize is split
evenly among them (2 for a pairwise tie, up to all `N`).

**Verified backward-compatible at N=2**: with `cost=1, benefit=4, speed=1.5`,
this formula reproduces the shipped two-player payoff matrix (`safeSafe=1.0,
safeUnsafe=0.6, unsafeSafe=2.4, unsafeUnsafe=2.0`) exactly — see
`ai_race/tests/test_nplayer_scoring.py`.

## Deliberately out of scope

- **`p_fo` (found-out probability)**: the paper's N-player appendix also
  includes a term for Unsafe players being caught and losing that round's
  benefit. Dropped per instructor's direction; only private setback risk
  (`max_private_risk`) is modelled.
- **Population / evolutionary dynamics**: fixation probabilities, stationary
  distributions, and multivariate-hypergeometric group sampling from a
  finite population of size `Z` are the paper's EGT analysis of *many* races
  over evolutionary time, already implemented for the two-player game in
  `ai_race/theory/evolution.py` / `equilibria.py`. This module only plays a
  single race at a time between `n_players` agents; it does not port that
  analysis.
- **CAS strategy**: `ai_race/engine/strategies.py`'s four canonical
  strategies include CAS, but the paper's N-player appendix only defines
  AS/AU/CS for the group game. `strategies.py` here only generalises those
  three rather than guessing an N-player CAS.
- **Vietnamese prompt template / other languages**: only
  `ai_race/engine_nplayer/prompts/ai_race_nplayer_en.txt` exists.
- **Packaging**: `pyproject.toml`'s `package-data` for `ai_race` only globs
  `prompts/*.txt` (top-level) and `configs/**/*.json` (recursive — this
  already covers `configs/game_nplayer/`/`configs/agents_nplayer/` for free).
  It does not declare `engine_nplayer/prompts/*.txt`, so a non-editable
  `pip install .` (building a real wheel) would ship without this template.
  Not touched here since every documented workflow in this repo either runs
  from source (`pip install -e ...`) or copies the raw repo tree (Kaggle) —
  neither is affected — but a real package release would need that glob added.

## Layout

| File | Role |
|---|---|
| `state.py` | `NPlayerGameConfig`, `NPlayerTurnRecord`, `NPlayerGameResult` |
| `scoring.py` | Payoff formula, race outcomes (winner/tie/loser by comparing to the group max), terminal scoring |
| `strategies.py` | AS/AU/CS generalised to a list of co-player histories |
| `game.py` | `NPlayerAIRaceGame` — same stepwise `build_round_prompts()`/`apply_round_responses()` shape as `ai_race.engine.game.AIRaceGame` |
| `round.py` / `prompt.py` | Prompt rendering; the "other companies" state, the per-`k` payoff table, and round history are pre-joined into multi-line blocks in Python since `str.format` has no loop construct |
| `recorder.py` | `NPlayerRunJournal` — long format only (`turns.jsonl`, `races.csv`, `players.csv`); no wide `player_1_*`/`player_2_*` columns, since those don't generalise to arbitrary N |
| `runner.py` | `build_games_for_model` / `run_games_batched` / `run_experiment`, mirroring `ai_race/runner/`, fixing its `cursor : cursor + 2` response-slicing (hardcoded to two players) to slice by each active game's actual agent count |

Configs live in `ai_race/configs/game_nplayer/*.json` and
`ai_race/configs/agents_nplayer/*.json` (separate from `configs/game/` and
`configs/agents/`, which are validated elsewhere as strictly two-player).

## Reused unchanged from the rest of the repo

`ai_race.engine.state.Action`, `ai_race.engine.agent.RaceAgent`,
`ai_race.engine.round.{parse_action, extract_reasoning, response_text}`,
`ai_race.engine.prompt.apply_optional_blocks`,
`ai_race.dataio.config_loader.{load_json, validate_experiment,
personas_sha256}`, `ai_race.dataio.recorder.{append_turns_jsonl,
write_turns_jsonl}`, `ai_race.models.factory`, `ai_race.paths`,
`ai_race.runner.run_experiment.{model_slug, make_mock_send_batch}`. None of
these depend on the number of players.
