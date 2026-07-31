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
  `prompt_version`, `run_phase`, `persona_condition`, `seat_persona_role`, `rep`,
  `round`, `player`, `player_index`, and `opponent`;
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
`max_private_risk`, `prompt_version`, `run_phase`, `persona_condition`,
`player_1_persona_role`, `player_2_persona_role`, `rep`, `n_rounds`, and
`parse_failures`. `stop_forced` and `tie` are required audit fields. The canonical
file also records seeds, final progress, Unsafe frequencies, setbacks, and final
payoffs.

`players.csv` has one row per player-race and includes at least `game_id`, `model`,
`max_private_risk`, `prompt_version`, `run_phase`, `persona_condition`,
`persona_role`, `rep`, `player`, and `outcome`
(`winner`, `loser`, or `tie`). `risk` is accepted as a legacy alias for
`max_private_risk`. The terminal audit fields are required: rounds, progress, stage
payoff, Unsafe count/frequency, private risk, prize, setback
eligibility/draw/outcome, and final payoff.

`all_results.csv` has one row per race in the FAIRGAME `all_results` shape: the
per-round action and payoff sequences as JSON list columns, plus terminal progress,
Unsafe counts, private risk, prize, setback, and final payoff for each seat. It also
carries `persona_condition` and `playerN_persona_role`. It is a convenience view for
skimming outcomes and is not read by the analyser; prompts and raw responses stay in
`turns.jsonl`, which remains the audit surface.

`persona_condition` labels the seat/persona cell (`none` for the neutral baseline).
Injecting a persona fills an optional block that is already part of the frozen
template, so it leaves `prompt_version`, the prompt hash, and `protocol_signature`
completely unchanged. `persona_condition` is therefore the only thing separating a
persona race from a neutral one, and the analyser stratifies every table by it and
refuses unlabelled races unless `--allow-missing-persona-condition` is supplied.
`seat_persona_role`/`persona_role` name the role of that specific seat, so an
asymmetric cell such as adversarial-versus-cooperative can be split by seat.

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

Prompt surface-sensitivity runs use a separate paired analysis so modified prompt
hashes can never enter the canonical primary pool accidentally:

```bash
python results/scripts/analyze_surface_sensitivity.py \
  --lane-root /path/to/lane-a --lane-root /path/to/lane-b \
  --output-dir results/derived/surface-sensitivity
```

It checks a single model/source/decoding contract, one completed shard per
variant, manifest/file counts, common-random-number horizons, and exact first-round
pairing. The first-round flip rate is the direct surface-sensitivity estimand;
whole-trajectory Unsafe-rate differences also contain state feedback after an
earlier action changes. Reported intervals resample complete repetition blocks,
not dependent decision rows.

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
- nearest canonical AS/AU/CS/CAS mismatch summaries;
- `sample_summary.csv`, one row per analysis cell pooling sample sizes, mean and
  median Unsafe frequency, realised horizons, and parse-failure accounting;
- pairwise treatment contrasts on **two** analysis windows, `treatment_contrasts.csv`
  over every round and `treatment_contrasts_round2plus.csv` over the panel sample,
  with the persona equivalents; the two disagree whenever round 1 differs from later
  rounds, and `human_reference.json` names which table each effect is scored on;
- `theory_vs_experiment.csv`, described under *Theory outputs* below.

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

The canonical primary prompt version is `ai-race-fairgame-v3`. One template file per
language carries that label, so the analyser accepts any of the frozen template
hashes:

| template | SHA-256 |
|---|---|
| `ai_race/prompts/ai_race_en.txt` | `27086bd80378c25e859d03527a5ae55c1046f231ef7b914db9cb3c3b4fb2df3e` |
| `ai_race/prompts/ai_race_vi.txt` | `a6d3f738cf58043ae0dadc351cac12da07bd60778317b0566d743f5e40a77510` |

A different version, or a hash outside that set—including modified text relabelled
as v3—is rejected from primary analysis and requires the explicit mixed-protocol
sensitivity override below.

Both Kaggle paths hash the same prompt text. `kaggle/benchmarks/ai_race_baseline.py`
is self-contained by design and does not import the package, so it holds a
byte-for-byte copy of `ai_race_en.txt`; `ai_race/tests/test_prompt_contract.py`
compares that copy against the shipped file, because a copy that drifted by one
character would record a hash outside the canonical set and lose every race to the
prompt gate.

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

## Optional robustness refits

```bash
python results/scripts/analyze_ai_race.py --fit-logit-robustness
```

Off by default because it is one fit per common-random-number block. It refits the
saturated specification once per block, each time omitting that block, and writes
`logit_robustness_jackknife.csv` plus `logit_robustness_metadata.json`.

| column | meaning |
|---|---|
| `variant` | `full`, `exclude_retried_races`, or `exclude_min_horizon` |
| `term` | regression term |
| `n_observations`, `n_blocks`, `n_blocks_refitted` | sample and refit accounting |
| `coefficient_full` | estimate on the whole variant sample |
| `coefficient_min`, `coefficient_max` | leave-one-block-out range |
| `max_abs_shift`, `block_of_max_shift` | largest displacement and the block causing it |
| `sign_stable` | full-sample and every leave-one-out estimate agree in sign |
| `negligible_at_full_sample` | the estimate is numerically zero, so it has no sign |

Exclusions are applied at race level, never at decision level: the lagged
predictors are built from the race trajectory, so dropping single rows would leave
later lags pointing at rounds no longer in the sample. `exclude_retried_races`
drops any race where a decision needed a generation retry — the closest analogue
here to the source paper's dropped pair — and `exclude_min_horizon` drops races
that stopped at the five-round minimum and therefore carry the least history.

A block whose removal makes the model unidentified or non-convergent is listed in
`skipped_blocks` rather than recorded as a zero coefficient. The leave-one-out
spread is a sensitivity diagnostic, not a standard error, and has no p-value.

## Theory outputs

```bash
python results/scripts/build_theory_tables.py
```

Writes to `results/derived/ai_race_theory/`. This script reads **no run output**.
Every number is a property of the game defined by `ai_race/configs/game/*.json` and
is identical for every model, persona condition, and run.

| file | contents |
|---|---|
| `theory_payoff_matrix.csv` | expected payoff for each ordered strategy pair per treatment, with the `method` that produced it |
| `theory_equilibria.csv` | stage-game class, social-dilemma threshold, pure Nash profiles, and the AS/AU closed-form boundaries |
| `theory_stationary_distribution.csv` | predicted population composition under evolutionary dynamics |
| `theory_expected_unsafe.csv` | Unsafe frequency implied by that composition |
| `theory_metadata.json` | parameters, method notes, and the caveats below |

Payoffs are computed by exact enumeration over the horizon distribution by default.
`--payoff-method monte_carlo` reproduces the source paper's construction (closed
form on the four AS/AU pairs, 10⁴ replications elsewhere) and exists for
cross-checking. The equilibrium and evolutionary tables always use the exact
matrix: `Pi(CAS, AU)` and `Pi(AU, AU)` are the same number in the game, sampling
noise makes them differ by roughly 0.04, and an exhaustive best-response search
reads that difference as a strict preference and drops real equilibria.

Two things these files are not:

- **`theory_stationary_distribution.csv` is not `strategy_summary_player.csv`.**
  The first is a predicted population composition; the second classifies observed
  LLM trajectories against the nearest canonical strategy. Different questions.
- **`theory_vs_experiment.csv` is not a fit.** The analyser emits it alongside the
  behavioural tables, comparing each cell's observed *median* Unsafe frequency —
  the statistic Figure 3B of the source paper uses — against the model prediction.
  `predicted_phi_U` depends only on `max_private_risk`, so it is identical for every
  model and persona cell. `difference` measures how far an LLM sits from the game
  theory; a small difference is a property of the game, not evidence about a model.
  The warning is repeated in `theory_vs_experiment_metadata.json`.

The evolutionary tables are the **small-mutation limit**, where the population is
monomorphic and the chain reduces to fixation probabilities between the four
strategies. `nominal_mu` records the mutation rate of the source paper's parameter
point that a row approximates; it is not applied, and `mu` is 0. Two consequences
are recorded in the metadata rather than left to be discovered: AU and CAS are
exactly payoff-equivalent against each other in this limit, so the paper's AU-to-CAS
transition near `p_r^max = 0.2` cannot appear and their stationary mass splits
evenly instead; and a finite mutation rate spreads mass into mixed populations that
the limit cannot represent. What the limit does reproduce is the CS takeover above
roughly `p_r^max = 0.6` and the negligible stationary mass of Always Safe at every
treatment.
