# FH Data Quality: Duplicate-Grain Root Cause

## Finding

The critical `races_grain` / `players_grain` / `turns_grain` failures in `fh_analysis_report.md`
(48 / 96 / 336 duplicated rows, ~1.4% of races) all trace to exactly **two raw run directories**:

- `data/experiments/players_2/provider_google/family_gemini/persona_risk_aware/mode_risk_matrix/condition_R3_R2/google-gemini-3-flash-preview/raw/`
- `data/experiments/players_2/provider_google/family_gemini/persona_risk_aware/mode_risk_matrix/condition_R3_R3/google-gemini-3-flash-preview/raw/`

Each of these `races.csv` files contains 24 duplicated `game_id` rows (out of 42 total races in the
file). For a duplicated pair, `game_id`, `game_seed`, and the recorded `stop_draws` RNG sequence are
**identical**, but the *outcome* differs: e.g. one copy of
`ai_race_risk_10__google-gemini-3-flash-preview__en__persona_risk_3_2__rep0005` ends with
`winner=Company_1, tie=0, progress=13.5/13.0, unsafe_count=7/6`, while the other copy of the same
`game_id` ends with `tie=1, progress=12.5/12.5, unsafe_count=5/5`.

## Interpretation

Two things are happening at once, and it is worth keeping them separate:

1. **Recorder/resume hygiene**: the same `game_id` was recorded twice in the same `races.csv`. The
   most likely cause is a run that was interrupted and resumed without the resume path skipping
   already-completed games, so `RunJournal` appended a second terminal row for a game index that had
   already finished. This is a pipeline/harness issue, not a modeling one, and is fully mitigated by
   the existing `duplicate_grain_key` exclusion used everywhere in `fh_analytic`.
2. **Within-model non-determinism given a fixed `game_seed`**: the two recorded playthroughs of the
   *same* `game_id` produced *different* action sequences even though `game_seed` (which drives the
   horizon-length and setback RNG streams, per `_stream_seed` in `ai_race/engine/game.py`) and the
   logged `stop_draws` are identical. `game_seed` does not control LLM sampling; that is a separate
   `sampling_seed()` stream, and these turns have `sampling_seed_applied=False`. So this is expected,
   not a bug: the backend is not seeded for these rows, and Gemini's own sampling varies run to run.
   It is an incidental confirmation that `google-gemini-3-flash-preview` responses are not
   reproducible turn-for-turn under a fixed `game_seed` alone, which is worth remembering if a future
   confirmatory run wants bit-for-bit replayability -- `sampling_seed_applied` would need to be `True`.

## Recommendation

No data was lost or double-counted in any of the mining reports in this session or the prior
baseline pipeline: every stage filters on `duplicate_grain_key == False`. Before a confirmatory run,
worth checking `RunJournal`'s resume path so a restart cannot append a second terminal row for a
game index that already has one, and confirming `sampling_seed_applied` is `True` wherever
replayability matters.
