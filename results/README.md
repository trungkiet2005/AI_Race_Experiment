# AI Race results

This tree is intentionally empty until an AI Race run completes on Kaggle. It does
not retain Collective Risk outputs and it does not contain placeholder observations,
effect sizes, or figures.

```text
results/
  open_source/     outputs from offline/open-weight Kaggle runs
  frontier/        outputs from frontier/API or Kaggle Benchmark runs
  scripts/         analysis code tracked in Git
  derived/         generated tables (ignored; created by the analyser)
```

Keep one self-contained directory per run. A run directory must contain
`turns.jsonl`, `races.csv`, `players.csv`, and `run_manifest.json` so completion,
phase, counts, and joins can be audited.

## Expected schema

`turns.jsonl` has one JSON object per player decision. The canonical logger writes
the following analysis fields:

- identifiers/context: `game_id`, `model`, `max_private_risk`,
  `prompt_version`, `run_phase`, `rep`, `round`, `player`, `player_index`, and
  `opponent`;
- action/protocol health: `action` (`"safe"` or `"unsafe"`), `unsafe` (`0` or
  `1`), `parse_failed`, and `retry_count`;
- dynamic state: `own_prev_action`, `opponent_prev_action`,
  `own_progress_before`, `opponent_progress_before`, and
  `progress_gap_before`, both players' accumulated stage payoffs, and both
  players' current private risks before the decision;
- canonical-mechanism audit: `round_payoff`, `step_increment`,
  `own_progress_after`, `opponent_progress_after`, `progress_gap_after`,
  `cumulative_stage_payoff_after`, `current_private_risk_after`, `stop_draw`,
  and `stopped`.

Full prompts, raw responses, attempt histories, and seed fields remain in the run
logs for reproducibility but are not copied into descriptive tables.
`progress_gap_before` is always focal progress minus opponent progress, measured
before the simultaneous round decision.

`races.csv` has one row per race and includes at least `game_id`, `model`,
`max_private_risk`, `prompt_version`, `run_phase`, `rep`, `n_rounds`, and
`parse_failures`. `stop_forced` and `tie` are required audit fields. The canonical
file also records seeds, final progress, Unsafe frequencies, setbacks, and final
payoffs.

`players.csv` has one row per player-race and includes at least `game_id`, `model`,
`max_private_risk`, `prompt_version`, `run_phase`, `rep`, `player`, and `outcome`
(`winner`, `loser`, or `tie`). `risk` is accepted as a legacy alias for
`max_private_risk`. The terminal audit fields are required: rounds, progress, stage
payoff, Unsafe count/frequency, private risk, prize, setback
eligibility/draw/outcome, and final payoff.

`run_manifest.json` must report `status`, `run_phase`, and output counts.
`status="completed"` is required for primary analysis. Completed manifests must
include matching `n_turns` and `n_races`; `n_players` is also checked when the
runner supplies it. Primary analysis additionally requires one of the two
provenance-rich Kaggle schemas: `ai-race-kaggle-run-v1` for the offline/open-weight
runner or `ai-race-kbench-run-v1` for Kaggle Benchmarks. Each must identify the
exact source and prompt hashes, model route/revision, decoding contract, sampling
seed handling, mechanism/configuration, and relevant package versions. The lean
`ai-race-results-v1` development-runner manifest is useful for smoke tests but is
deliberately insufficient for primary pooled analysis.

The analyser accepts a small set of documented legacy aliases, but it validates
unique keys, two players per race, two decisions per game-round, consecutive
variable horizons, cross-file model/risk/prompt agreement, repetition-block
consistency, race-level parse counts, the logged race horizon, and all canonical
mechanism audit fields. It fails loudly on malformed JSON, broken joins, missing
audit fields, or contradictory mechanics. Both the core runner and Kaggle
Benchmark schemas satisfy the row-level requirements; primary protocol pooling
uses the two provenance-rich Kaggle manifests described above.

The canonical risk treatments are 0.1, 0.6, and 0.9 and the minimum horizon is five
rounds. Noncanonical risk/horizon smoke tests are rejected unless
`--allow-noncanonical-mechanism` is explicitly supplied. The old
`--allow-noncanonical-horizons` spelling remains an alias. This audit flag loads
those races into health/accounting outputs but never admits them to behavioural
estimands, and it does not waive mechanics-consistency checks.

Before producing behavioural results, the analyser checks Safe/Unsafe step
increments, the 2x2 stage payoff matrix, pre/post progress arithmetic, shared
stopping draws under the minimum-5-round and \(p=0.2\) rule, terminal `stopped`
markers, cross-round progress continuity, paired-opponent state, both players'
logged pre-decision accumulated payoff/private risk, cumulative stage payoff/private
risk, trajectory totals, winner/tie status, terminal private-risk
calculation, the 100/50/0 prize allocation, winner/tied-winner setback eligibility,
setback flags, fixed-seat setback draws across CRN-matched risk treatments, and
final payoff.
For logged repetition blocks it also verifies that matched risk treatments reuse
the same realised horizon and stopping-draw stream.

## Analysis

Run this on Kaggle after downloading or mounting completed experiment outputs:

```bash
python results/scripts/analyze_ai_race.py
```

The default discovery roots are `results/open_source/` and `results/frontier/`.
Additional roots can be combined explicitly:

```bash
python results/scripts/analyze_ai_race.py \
  --input /kaggle/input/open-weight-runs \
  --input /kaggle/input/frontier-runs \
  --output /kaggle/working/ai-race-analysis
```

The script produces:

- decision-weighted and player-weighted Unsafe rates by model, maximum private
  risk, and prompt version;
- Unsafe rates after the opponent's previous Safe versus Unsafe action;
- Unsafe rates while the focal player is ahead, tied, or behind;
- later Unsafe rates split by the first-round action;
- parse-failure and retry rates plus per-race inclusion accounting;
- per-player trajectory metrics and winner/loser/tie comparisons;
- nearest canonical AS/AU/CS/CAS mismatch summaries.

Player-level tables first calculate each player's trajectory rate and then average
those rates. They complement the decision-level summaries because a longer realised
horizon otherwise contributes more rows. Conditional contrasts are reported only
for players observed in both relevant states; missing exposure is not imputed.
Reported player-level SDs are descriptive dispersion across trajectories. No
`SD/sqrt(n_players)` standard error is emitted because the two players in a race and
the matched treatment races within a repetition block are not independent.

The contamination and mechanism rules operate at the race level. If either player
has any `parse_failed=true` decision, the **entire race** is excluded from every
behavioural table, player comparison, strategy classification, and panel model.
This prevents the Safe fallback and its downstream state changes from entering an
estimand through otherwise valid later rows. A race is also excluded from those
estimands if it uses a noncanonical risk treatment, ends below round 5, or has
`stop_forced=1`; a safety-cap termination is not treated as a draw from the
paper's stochastic horizon. All races remain in
`parse_failures.csv`, `race_quality.csv`, and manifest accounting. There is no
override that promotes an excluded race into a primary behavioural estimand.

## Prompt/protocol compatibility

`prompt_version` is resolved per race from `run_manifest.json` and any of
`turns.jsonl`, `races.csv`, or `players.csv`, then checked across all four sources.
A conflicting version within one race is always an error. By default, missing
versions or multiple versions across pooled inputs are also errors, so incompatible
protocols cannot be combined silently.

The canonical primary prompt is `ai-race-paper-v2`, SHA-256
`6180d4f699813a602a53cf4290b972aa4df4bf02ff1c646a85ab09d80d7729ff`.
Both Kaggle paths hash the same prompt text. A different version or hash—including
modified text relabelled as v2—is rejected from primary analysis and requires the
explicit mixed-protocol sensitivity override below.

The analyser also canonicalises the complete manifest protocol payload and hashes
it as `protocol_signature`. The payload covers the manifest schema, exact source
revision, prompt version and SHA-256, model identity/route, every decoding field,
sampling-seed provenance, mechanism or game-config hashes, and package versions.
The numeric base seed is intentionally not part of compatibility: independent
replication batches may use different seed values, but they must use the same
documented seed-application contract.

Different models may have different signatures and can be compared normally.
Within one model label, however, primary analysis permits exactly one signature.
Thus reusing a model label after changing temperature, token limit, seed forwarding,
prompt text, source/model revision, mechanics, or relevant runtime versions fails
before any table is written. `analysis_manifest.json` records the full
signature-to-payload map rather than only an opaque digest.

For an explicitly labelled sensitivity analysis, allow multiple or unknown
protocols with:

```bash
python results/scripts/analyze_ai_race.py --allow-mixed-protocols
```

Descriptive outputs remain stratified by both `prompt_version` and
`protocol_signature`; missing prompt labels appear as
`__MISSING_PROMPT_VERSION__`. If one model contains multiple signatures under this
explicit override, the optional logit adds `C(protocol_signature)` (which already
contains model and prompt identity) rather than pretending the decoding contracts
are interchangeable. Runs with missing or legacy manifest provenance are admitted
only when both `--allow-mixed-protocols` and `--allow-nonfinal-runs` mark the output
as an audit; each unverified source receives its own non-pooling signature.

`run_phase` is a second required protocol dimension with allowed values `pilot` and
`confirmatory`. Primary analysis accepts only a single, non-missing
`run_phase="confirmatory"` and `run_manifest.status="completed"`. Pilot,
mixed-phase, missing-phase, failed, running, and `protocol_failed` inputs are
rejected by default.

Two explicit audit-only overrides exist:

```bash
python results/scripts/analyze_ai_race.py \
  --allow-nonconfirmatory-runs \
  --allow-nonfinal-runs
```

Outputs remain stratified by `run_phase` and `run_status`; these flags do not turn a
pilot or failed run into confirmatory evidence. A missing phase/status receives an
explicit sentinel rather than being silently merged.

Derived files default to `results/derived/ai_race_analysis/` and are ignored by Git.
Only the scripts, README, and empty source-directory sentinels belong in version
control.

## Optional panel model

The paper-style association model is opt-in:

```bash
python results/scripts/analyze_ai_race.py --fit-logit
```

It uses `statsmodels` and rounds 2 onward:

```text
unsafe
  ~ C(max_private_risk)
  + first_round_unsafe
  + own_prev_unsafe * opponent_prev_unsafe * progress_gap_before
```

The risk treatments reuse horizon and setback random streams for the same
repetition. Consequently, game-only clustered standard errors are insufficient.
The analyser clusters by the common-random-number block
`source_run/model/rep`, which groups matched risk-treatment races together. The
logit refuses to run if `rep` cannot be resolved, if fewer than two blocks exist, or
if any included block fails to span at least two risk treatments. The mechanism
gate separately checks that logged CRN blocks reuse their horizon, stopping draws,
and fixed-seat setback draws across risk treatments.

When explicitly combining more than one model, prompt version, run phase, or run
status, the sensitivity model adds the corresponding categorical fixed effect
(`C(model)`, `C(prompt_version)`, `C(run_phase)`, or `C(run_status)`). Descriptive
tables are always stratified by these protocol dimensions. If a model contains
multiple exact protocol signatures under the explicit override, a single
`C(protocol_signature)` effect replaces the model/prompt effects to avoid redundant
columns. Descriptive tables also remain stratified by the exact signature.

Coefficients are conditional associations in an endogenous repeated interaction,
not causal effects. Human demographics and elicited risk-preference covariates from
the source paper are deliberately absent: they are not defined for LLM agents.

The optional `--include-exploratory-behind` flag adds
`BEHIND_UNSAFE_EXPLORATORY` to strategy-distance tables. It is not one of the
paper's canonical strategies and must remain labelled exploratory.
