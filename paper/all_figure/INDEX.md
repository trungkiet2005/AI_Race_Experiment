# AI Race -- all figures, gathered for paper selection

**99 indexed figures total**: 47 new exploratory (never published anywhere) + 52 already-admitted figures copied in from `results/open_source/*`. `[NEW]` sections are descriptive-only, generated outside the repo's admission process. `[ADMITTED]` sections passed the project's own hash/manifest admission gates -- see each pilot's README under `results/open_source/` for the exact evidence-class wording before using one in the paper.

Folder layout mirrors the sections below (`NEW_01_persona/`, `ADMITTED_context_skin_pilot/`, ...); every file in this folder is a plain copy, safe to delete or move independently.

## 0. Cross-pilot overview  [NEW]

### Cross-pilot orientation: canonical Unsafe rate vs risk

![Cross-pilot orientation: canonical Unsafe rate vs risk](NEW_00_overview/cross_pilot_canonical_unsafe_rate.png)

Each pilot's own baseline/canonical/no-persona condition, side by side purely for orientation. These are four separate pilot studies with different prompt surfaces and coverage; CLAUDE.md's pooling prohibition applies -- do not read this as one combined estimate.

## 1. Persona-sensitivity pilot (prompt_sensitivity_pilot/)  [NEW]

### Unsafe rate by persona x risk (Wilson 95% CI)

![Unsafe rate by persona x risk (Wilson 95% CI)](NEW_01_persona/persona_unsafe_rate_by_risk.png)

Decision-level Unsafe rate for each risk-attitude persona framing (R0/R-/S_AA/S_AC/S_CA) versus the no-persona baseline, split by the three max-private-risk treatments. Wilson 95% CIs on raw proportions. Pilot, descriptive only.

### Persona pairwise contrasts (Cohen's d), by risk

![Persona pairwise contrasts (Cohen's d), by risk](NEW_01_persona/persona_contrasts_effect_sizes.png)

Every pairwise persona contrast (t-test on player-level mean Unsafe frequency), one panel per risk treatment. Bars in red survive Bonferroni correction across the stratum; grey bars do not. n_comparisons_in_stratum is small per pair (pilot), so treat as descriptive screening, not confirmatory.

### Risk-treatment contrasts (Cohen's d), by persona

![Risk-treatment contrasts (Cohen's d), by persona](NEW_01_persona/treatment_contrasts_effect_sizes.png)

0.1-vs-0.6 and 0.6-vs-0.9 max-private-risk contrasts on player-level Unsafe frequency, computed separately within each persona condition. Compare to the paper's own contrast (human_comparison E5/E6): the paper reports a small negative 0.6-vs-0.9 effect and a larger positive 0.1-vs-0.6 effect.

### Cluster-robust logit forest plot, spec 6

![Cluster-robust logit forest plot, spec 6](NEW_01_persona/clustered_logit_forest_full_spec.png)

Coefficients (log-odds) with cluster-robust 95% CI for the richest fitted specification of P(Unsafe) ~ risk + persona + own/opponent previous action + progress gap (+ interactions), clustered on race. Pre-computed by the project's own analyser; this figure only visualises it.

### Key coefficient stability across the 6 nested specifications

![Key coefficient stability across the 6 nested specifications](NEW_01_persona/clustered_logit_across_specifications.png)

Tracks own/opponent previous-action, progress-gap, and first-round coefficients as controls are added specification by specification. A term that swings sign or loses its CI-exclusion-of-zero as controls are added is a fragile effect, not a robust one.

### Jackknife (leave-one-block-out) coefficient ranges

![Jackknife (leave-one-block-out) coefficient ranges](NEW_01_persona/logit_jackknife_robustness.png)

Bars span [coefficient_min, coefficient_max] across leave-one-cluster-block-out refits; dot marks the full-sample coefficient. Three variants: full sample, excluding races with a parse retry, excluding min-horizon races. A term whose range crosses zero is not sign-stable under resampling.

### Unsafe rate by progress-gap bin x risk

![Unsafe rate by progress-gap bin x risk](NEW_01_persona/gap_bin_unsafe_rate.png)

Pooled across persona conditions: how far ahead/behind a player is before deciding, versus the probability the decision is Unsafe. Shaded band is a Wilson 95% CI on the pooled proportion.

### Own x opponent previous-action -> Unsafe rate heatmap

![Own x opponent previous-action -> Unsafe rate heatmap](NEW_01_persona/lag_profile_heatmap.png)

Conditions the current-round Unsafe rate on the joint (own, opponent) action pair from the previous round, one panel per risk treatment. The opponent_prev_unsafe row/column contrast is the same effect tested in human_comparison effect E1.

### Opponent-reciprocity effect by persona x risk

![Opponent-reciprocity effect by persona x risk](NEW_01_persona/opponent_response_effect.png)

Player-level mean of (own Unsafe rate after opponent played Unsafe) minus (after opponent played Safe). Error bars are +-1/2 descriptive SD across players (not a CI). Positive values indicate tit-for-tat-like escalation.

### Unsafe rate by race state x risk

![Unsafe rate by race state x risk](NEW_01_persona/race_state_unsafe.png)

Tests the falling-behind mechanism the paper's title names: is Unsafe play more common when a player is behind? Pooled across personas, Wilson 95% CI.

### First-round choice -> later Unsafe rate

![First-round choice -> later Unsafe rate](NEW_01_persona/first_round_persistence.png)

Round-2-plus Unsafe rate conditioned on the same player's round-1 action. Corresponds to human_comparison effect E3 (first_round_unsafe), which the paper finds significant only at the 10% level.

### Winner vs loser Unsafe rate, per race

![Winner vs loser Unsafe rate, per race](NEW_01_persona/winner_loser_scatter.png)

Points above the diagonal are races the winner played more Unsafe than the loser. Overall Pearson r values are in winner_loser_correlation.csv (pooled here only for visual orientation, not restated as a single number).

### Nearest-strategy classification shares by persona x risk

![Nearest-strategy classification shares by persona x risk](NEW_01_persona/strategy_classification_shares.png)

Hamming-distance nearest-strategy label per player (ties kept and joined with '|', per the project's classify.py convention). Shares are of classifiable players only.

### LLM observed vs EGT-predicted Unsafe fraction, by persona

![LLM observed vs EGT-predicted Unsafe fraction, by persona](NEW_01_persona/theory_vs_experiment_persona.png)

Bars are the persona pilot's observed mean phi_U; black dashes are the reduced evolutionary-game prediction at the paper's main-text reference point (beta=2) and best-fit point (beta=0.01). Complements egt_reproduction's own theory-vs-LLM figure, sliced by persona condition.

### Human vs LLM replication scorecard (8 pre-registered effects)

![Human vs LLM replication scorecard (8 pre-registered effects)](NEW_01_persona/human_comparison_scorecard.png)

Each panel is one pre-registered human-paper effect (Table 1 / Table S3 / Figure 2A of the source paper) against the persona pilot's LLM estimate. Green title = replicated by the project's own verdict rule, red = not replicated. Read alongside each effect's 'description' text in human_comparison.csv -- several are equivalence/TOST tests, not simple sign matches.

### Sample composition, exclusions, and seat balance

![Sample composition, exclusions, and seat balance](NEW_01_persona/sample_quality_overview.png)

Left: races per persona x risk cell (design coverage check). Middle: reasons a race is excluded from behavioural estimands (parse failure, forced stop, non-canonical mechanism, etc; race_quality.csv). Right: Unsafe rate by seat index (Participant_1 vs Participant_2) -- large gaps would flag a seat-order confound. All parse_failures.csv rates were 0 in this pilot.

## 2. Surface-sensitivity pilot + smoke  [NEW]

### Surface-variant Unsafe rate forest plot (pilot)

![Surface-variant Unsafe rate forest plot (pilot)](NEW_02_surface/surface_variant_unsafe_rate_forest.png)

18 surface-wording variants of the same canonical prompt (paraphrase, formatting, ordering, emphasis, framing, boundary/position edits), each run on the same coverage as the canonical control. Dashed line is the canonical variant's own rate. Cluster-bootstrap 95% CI per variant_summary.csv.

### Smoke vs pilot Unsafe rate per surface variant

![Smoke vs pilot Unsafe rate per surface variant](NEW_02_surface/surface_variant_pilot_vs_smoke.png)

Compares the small protocol-development smoke run against the larger admitted pilot for every surface variant. Large jumps flag variants whose estimate was unstable at smoke scale; both stages are still diagnostic, never confirmatory, per the pilot README.

### First-round flip rate vs canonical, by surface variant

![First-round flip rate vs canonical, by surface variant](NEW_02_surface/surface_variant_first_round_flip_forest.png)

First decision (round 1, identical pre-action state across variants) that differs from the canonical wording's decision on the matched replay. The framing and emphasis families show the largest flip rates; several formatting/order variants show none.

### Surface variant x risk heatmap (pilot)

![Surface variant x risk heatmap (pilot)](NEW_02_surface/surface_variant_by_risk_heatmap.png)

Same 18 variants, split by max-private-risk treatment. Rows ordered by the overall pooled Unsafe rate (figure 1 in this section).

### Unsafe-rate spread by surface-variant family

![Unsafe-rate spread by surface-variant family](NEW_02_surface/surface_family_boxplot.png)

Each dot is one variant's pooled Unsafe rate; boxes group variants by editing family (formatting, framing, order, emphasis, ...). families with n=1 variant show as a degenerate box -- read the dot, not the box, in that case.

### Direction of round-1 flips by surface variant

![Direction of round-1 flips by surface variant](NEW_02_surface/surface_first_round_direction_stacked.png)

Counts (out of 60 matched round-1 decisions per variant) that flip toward Unsafe (red, right) vs. toward Safe (blue, left) relative to the canonical wording on the same pre-action state.

## 3. Game-understanding pilot  [NEW]

### Unsafe rate: canonical vs. calculator-decision-card prompt

![Unsafe rate: canonical vs. calculator-decision-card prompt](NEW_03_game_understanding/behavior_unsafe_rate_by_condition_risk.png)

canonical is the paper-faithful prompt; calculator_decision_card additionally discloses computed expected payoffs for each action before the decision. Cluster-robust (race-level) 95% CI.

### Setback rate and mean final payoff by condition x risk

![Setback rate and mean final payoff by condition x risk](NEW_03_game_understanding/behavior_setback_payoff_panels.png)

Setback rate is the fraction of races where the eventual winner (or tied winner) drew the private setback and lost the terminal payoff. Both conditions share the same horizon/seed coverage (30 races each, 10 per risk cell).

### Probe semantic accuracy and strict-format rate by variant

![Probe semantic accuracy and strict-format rate by variant](NEW_03_game_understanding/probe_accuracy_heatmap.png)

Semantic accuracy credits any response the parser can map to the intended meaning; strict-format rate requires the exact expected answer format. calculator discloses the numeric payoff computation before asking the probe question -- compare its accuracy to the unaided direct/paraphrase variants.

### Strict vs semantic probe accuracy, per condition x domain

![Strict vs semantic probe accuracy, per condition x domain](NEW_03_game_understanding/probe_strict_vs_semantic.png)

Points far above the diagonal answer correctly but not in the exact required format -- a formatting gap, not a knowledge gap. Domains include rule_recall, stage_payoff, expected_payoff, state_reconstruction, state_transition, terminal_scoring.

### Answer-flip rate under order/paraphrase perturbation, by domain

![Answer-flip rate under order/paraphrase perturbation, by domain](NEW_03_game_understanding/probe_flip_rate_by_domain.png)

For each probe item asked twice (forward/reverse order, or direct/paraphrase wording), the fraction of the 5 repetitions where the semantic answer changed. High flip rates under paraphrase alone indicate the probed 'knowledge' is wording-dependent rather than stable.

## 4. Context-skin pilot -- new supplementary views  [NEW]

### Unsafe-rate trajectory by round, per context x risk (T=0)

![Unsafe-rate trajectory by round, per context x risk (T=0)](NEW_04_context_supplement/context_skin_round_trajectory.png)

Round-level Unsafe rate pooled across the 32 repetitions per context x risk cell, up to round 9 (coverage thins beyond it because the horizon is stochastic and hidden from the model). New figure from raw turns.jsonl; not duplicated among the 27 already-admitted context_skin_pilot figures.

### Outcome rates by context: seat-1 win, tie, and setback

![Outcome rates by context: seat-1 win, tie, and setback](NEW_04_context_supplement/context_skin_outcome_rates.png)

Sanity/coverage check across the 8 context skins: seat-order win-rate imbalance would show as Participant_1 win rate far from ~0.5 - tie_rate/2, holding all else fixed by the shared CRN design.

### T=0 vs T=0.7 Unsafe rate, one point per context x risk cell

![T=0 vs T=0.7 Unsafe rate, one point per context x risk cell](NEW_04_context_supplement/context_skin_t0_vs_t07_scatter.png)

Compact single-panel companion to the 5 detailed figures already under analysis_temperature_robustness/. Points above the diagonal got more Unsafe under stochastic decoding at the same context x risk cell.

### Fixed-state paired context effect on Unsafe rate

![Fixed-state paired context effect on Unsafe rate](NEW_04_context_supplement/context_skin_fixed_state_paired_context.png)

Uses the fixed_state_pilot_t0 design: 96 engine-reachable decision states are replayed unchanged under each of the 8 context skins and both opaque action-code mappings, so round/progress/gap/risk are held exactly constant and only the surface context text differs.

### Opaque action-code position (P vs Q = Safe) effect on Unsafe rate

![Opaque action-code position (P vs Q = Safe) effect on Unsafe rate](NEW_04_context_supplement/context_skin_action_map_effect.png)

The engine always presents two opaque codes; which code maps to the mechanistically Safe action is balanced across an action-map manipulation. A gap between the two bars for the same context is a position/label bias independent of context content.

## 5. Context-skin pilot -- admitted figures  [ADMITTED]

### comprehension admission

![comprehension admission](ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__comprehension_admission.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### context mapping diagnostic

![context mapping diagnostic](ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__context_mapping_diagnostic.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### live payoff setback

![live payoff setback](ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__live_payoff_setback.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### live unsafe context risk mapping

![live unsafe context risk mapping](ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__live_unsafe_context_risk_mapping.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### paired context effects

![paired context effects](ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__paired_context_effects.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### paired flip directions

![paired flip directions](ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__paired_flip_directions.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### planned realistic fictional contrasts

![planned realistic fictional contrasts](ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__planned_realistic_fictional_contrasts.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### comprehension admission

![comprehension admission](ADMITTED_context_skin_pilot/analysis_live_pilot_t07__figures__comprehension_admission.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### context mapping diagnostic

![context mapping diagnostic](ADMITTED_context_skin_pilot/analysis_live_pilot_t07__figures__context_mapping_diagnostic.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### live payoff setback

![live payoff setback](ADMITTED_context_skin_pilot/analysis_live_pilot_t07__figures__live_payoff_setback.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### live unsafe context risk mapping

![live unsafe context risk mapping](ADMITTED_context_skin_pilot/analysis_live_pilot_t07__figures__live_unsafe_context_risk_mapping.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### paired context effects

![paired context effects](ADMITTED_context_skin_pilot/analysis_live_pilot_t07__figures__paired_context_effects.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### paired flip directions

![paired flip directions](ADMITTED_context_skin_pilot/analysis_live_pilot_t07__figures__paired_flip_directions.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### planned realistic fictional contrasts

![planned realistic fictional contrasts](ADMITTED_context_skin_pilot/analysis_live_pilot_t07__figures__planned_realistic_fictional_contrasts.png)

Primary T=0 live pilot (8 contexts x 3 risk cells, 768 races) -- the report's accepted headline figure set.

### comprehension admission

![comprehension admission](ADMITTED_context_skin_pilot/analysis_smoke_t0__figures__comprehension_admission.png)

Protocol-development smoke run (small N) -- retained as development evidence, not a primary estimate.

### context mapping diagnostic

![context mapping diagnostic](ADMITTED_context_skin_pilot/analysis_smoke_t0__figures__context_mapping_diagnostic.png)

Protocol-development smoke run (small N) -- retained as development evidence, not a primary estimate.

### live payoff setback

![live payoff setback](ADMITTED_context_skin_pilot/analysis_smoke_t0__figures__live_payoff_setback.png)

Protocol-development smoke run (small N) -- retained as development evidence, not a primary estimate.

### live unsafe context risk mapping

![live unsafe context risk mapping](ADMITTED_context_skin_pilot/analysis_smoke_t0__figures__live_unsafe_context_risk_mapping.png)

Protocol-development smoke run (small N) -- retained as development evidence, not a primary estimate.

### paired context effects

![paired context effects](ADMITTED_context_skin_pilot/analysis_smoke_t0__figures__paired_context_effects.png)

Protocol-development smoke run (small N) -- retained as development evidence, not a primary estimate.

### paired flip directions

![paired flip directions](ADMITTED_context_skin_pilot/analysis_smoke_t0__figures__paired_flip_directions.png)

Protocol-development smoke run (small N) -- retained as development evidence, not a primary estimate.

### planned realistic fictional contrasts

![planned realistic fictional contrasts](ADMITTED_context_skin_pilot/analysis_smoke_t0__figures__planned_realistic_fictional_contrasts.png)

Protocol-development smoke run (small N) -- retained as development evidence, not a primary estimate.

### context effect temperature change

![context effect temperature change](ADMITTED_context_skin_pilot/analysis_temperature_robustness__figures__context_effect_temperature_change.png)

Paired T=0 vs T=0.7 robustness analysis, 10,000 race-cluster bootstrap replicates.

### context effect temperature stability

![context effect temperature stability](ADMITTED_context_skin_pilot/analysis_temperature_robustness__figures__context_effect_temperature_stability.png)

Paired T=0 vs T=0.7 robustness analysis, 10,000 race-cluster bootstrap replicates.

### temperature mapping interaction heatmap

![temperature mapping interaction heatmap](ADMITTED_context_skin_pilot/analysis_temperature_robustness__figures__temperature_mapping_interaction_heatmap.png)

Paired T=0 vs T=0.7 robustness analysis, 10,000 race-cluster bootstrap replicates.

### temperature trajectory agreement

![temperature trajectory agreement](ADMITTED_context_skin_pilot/analysis_temperature_robustness__figures__temperature_trajectory_agreement.png)

Paired T=0 vs T=0.7 robustness analysis, 10,000 race-cluster bootstrap replicates.

### temperature unsafe delta by context

![temperature unsafe delta by context](ADMITTED_context_skin_pilot/analysis_temperature_robustness__figures__temperature_unsafe_delta_by_context.png)

Paired T=0 vs T=0.7 robustness analysis, 10,000 race-cluster bootstrap replicates.

### context recognition audit

![context recognition audit](ADMITTED_context_skin_pilot/context_recognition_t0_pilot__figures__context_recognition_audit.png)

Frozen context-recognition audit (does the model name the real-world domain it's role-playing?).

## 6. EGT reconstruction -- admitted figures  [ADMITTED]

### egt chain diagnostics

![egt chain diagnostics](ADMITTED_egt_reproduction/egt_chain_diagnostics.png)

Reduced evolutionary-game-theory reconstruction of the source paper, and its LLM comparison.

### egt expected payoff matrices

![egt expected payoff matrices](ADMITTED_egt_reproduction/egt_expected_payoff_matrices.png)

Reduced evolutionary-game-theory reconstruction of the source paper, and its LLM comparison.

### egt stationary strategy composition

![egt stationary strategy composition](ADMITTED_egt_reproduction/egt_stationary_strategy_composition.png)

Reduced evolutionary-game-theory reconstruction of the source paper, and its LLM comparison.

### egt strategy lens vs llm

![egt strategy lens vs llm](ADMITTED_egt_reproduction/egt_strategy_lens_vs_llm.png)

Reduced evolutionary-game-theory reconstruction of the source paper, and its LLM comparison.

### egt theory vs llm unsafe

![egt theory vs llm unsafe](ADMITTED_egt_reproduction/egt_theory_vs_llm_unsafe.png)

Reduced evolutionary-game-theory reconstruction of the source paper, and its LLM comparison.

## 7. Activation-level SAE pilot -- admitted figures  [ADMITTED]

### association selected features

![association selected features](ADMITTED_activation_sae/causal_selfplay__fast-sae-pilot-L12-v1__analysis__figures__association_selected_features.png)

Native-runtime causal self-play SAE audit (dose/sign/reconstruction/matched-random controls, layer 12).

### fixed state dose response

![fixed state dose response](ADMITTED_activation_sae/causal_selfplay__fast-sae-pilot-L12-v1__analysis__figures__fixed_state_dose_response.png)

Native-runtime causal self-play SAE audit (dose/sign/reconstruction/matched-random controls, layer 12).

### fixed state intervention diagnostics

![fixed state intervention diagnostics](ADMITTED_activation_sae/causal_selfplay__fast-sae-pilot-L12-v1__analysis__figures__fixed_state_intervention_diagnostics.png)

Native-runtime causal self-play SAE audit (dose/sign/reconstruction/matched-random controls, layer 12).

### fixed state target minus controls

![fixed state target minus controls](ADMITTED_activation_sae/causal_selfplay__fast-sae-pilot-L12-v1__analysis__figures__fixed_state_target_minus_controls.png)

Native-runtime causal self-play SAE audit (dose/sign/reconstruction/matched-random controls, layer 12).

### live direct comparable flips

![live direct comparable flips](ADMITTED_activation_sae/causal_selfplay__fast-sae-pilot-L12-v1__analysis__figures__live_direct_comparable_flips.png)

Native-runtime causal self-play SAE audit (dose/sign/reconstruction/matched-random controls, layer 12).

### live endogenous payoff effects

![live endogenous payoff effects](ADMITTED_activation_sae/causal_selfplay__fast-sae-pilot-L12-v1__analysis__figures__live_endogenous_payoff_effects.png)

Native-runtime causal self-play SAE audit (dose/sign/reconstruction/matched-random controls, layer 12).

### causal steering controls

![causal steering controls](ADMITTED_activation_sae/context_fast_sae_analysis__causal_steering_controls.png)

Layer-12/layer-20 context screen: held-out probe AUC and causal-steering controls.

### context shift descriptives

![context shift descriptives](ADMITTED_activation_sae/context_fast_sae_analysis__context_shift_descriptives.png)

Layer-12/layer-20 context screen: held-out probe AUC and causal-steering controls.

### heldout action probe

![heldout action probe](ADMITTED_activation_sae/context_fast_sae_analysis__heldout_action_probe.png)

Layer-12/layer-20 context screen: held-out probe AUC and causal-steering controls.

### sae token position robustness

![sae token position robustness](ADMITTED_activation_sae/figures__sae_token_position_robustness.png)

Primary token-position comparison across the pre-action vs prompt-last capture positions.

### sae feature confirmation

![sae feature confirmation](ADMITTED_activation_sae/surface_n600_strict_pre_action__figures__sae_feature_confirmation.png)

SAE fidelity/probe/feature-confirmation figures, pre-action capture position (n=600 decisions).

### sae fidelity by layer

![sae fidelity by layer](ADMITTED_activation_sae/surface_n600_strict_pre_action__figures__sae_fidelity_by_layer.png)

SAE fidelity/probe/feature-confirmation figures, pre-action capture position (n=600 decisions).

### sae probe by layer

![sae probe by layer](ADMITTED_activation_sae/surface_n600_strict_pre_action__figures__sae_probe_by_layer.png)

SAE fidelity/probe/feature-confirmation figures, pre-action capture position (n=600 decisions).

### sae feature confirmation

![sae feature confirmation](ADMITTED_activation_sae/surface_n600_strict_prompt_last__figures__sae_feature_confirmation.png)

Same views at the stricter prompt-last capture position, before response boilerplate.

### sae fidelity by layer

![sae fidelity by layer](ADMITTED_activation_sae/surface_n600_strict_prompt_last__figures__sae_fidelity_by_layer.png)

Same views at the stricter prompt-last capture position, before response boilerplate.

### sae probe by layer

![sae probe by layer](ADMITTED_activation_sae/surface_n600_strict_prompt_last__figures__sae_probe_by_layer.png)

Same views at the stricter prompt-last capture position, before response boilerplate.

## 8. Persona pilot -- surrogate XAI figures  [ADMITTED]

### xai top global coefficients

![xai top global coefficients](ADMITTED_prompt_sensitivity_pilot/xai_auto_vector_encoder__xai_top_global_coefficients.png)

Surrogate logged-feature XAI model (global coefficients / permutation importance), full prompt features.

### xai top permutation importance

![xai top permutation importance](ADMITTED_prompt_sensitivity_pilot/xai_auto_vector_encoder__xai_top_permutation_importance.png)

Surrogate logged-feature XAI model (global coefficients / permutation importance), full prompt features.

### xai top global coefficients

![xai top global coefficients](ADMITTED_prompt_sensitivity_pilot/xai_auto_vector_encoder_no_response__xai_top_global_coefficients.png)

Same surrogate XAI model with response-derived features removed.

### xai top permutation importance

![xai top permutation importance](ADMITTED_prompt_sensitivity_pilot/xai_auto_vector_encoder_no_response__xai_top_permutation_importance.png)

Same surrogate XAI model with response-derived features removed.

## 9. Two-player paper figure bank  [NEW / EXPLORATORY]

Fourteen publication-formatted figure families from the consolidated N=2 analysis. Each figure is available in PDF, SVG, 600-dpi PNG, and 600-dpi TIFF under `NEW_05_two_player_paper_analysis/`. There are no confirmatory gameplay runs in this evidence snapshot, so all behavioral comparisons remain exploratory. Separate API, KBench, and local-model protocols are shown as robustness lanes rather than pooled estimates.

### Baseline risk response

![Baseline risk response](NEW_05_two_player_paper_analysis/fig01_baseline_risk_response.png)

Balanced five-model comparison across 10%, 60%, and 90% maximum private risk, with repetition-block uncertainty and paired high-minus-low contrasts.

### Cross-protocol baseline robustness

![Cross-protocol baseline robustness](NEW_05_two_player_paper_analysis/fig01b_protocol_robustness_baselines.png)

Descriptive comparison of the balanced frontier cohort with the smaller clean Claude retry and the separate local-Qwen protocol.

### Initialization and round dynamics

![Initialization and round dynamics](NEW_05_two_player_paper_analysis/fig02_initialization_and_dynamics.png)

Separates first-round behavior from later trajectories so model-specific initialization is not hidden by all-round averages.

### Conditional dynamics

![Conditional dynamics](NEW_05_two_player_paper_analysis/fig03_conditional_dynamics.png)

Lagged-action and race-state summaries; these are descriptive endogenous associations, not causal response estimates.

### Strategy composition

![Strategy composition](NEW_05_two_player_paper_analysis/fig04_strategy_composition.png)

Player-race archetype composition from All Safe through All Unsafe, split by model and risk treatment.

### Safety-payoff frontier

![Safety-payoff frontier](NEW_05_two_player_paper_analysis/fig05_safety_payoff_frontier.png)

Joint view of behavioral safety and realized payoff, with payoff components separated for interpretation.

### Persona effects

![Persona effects](NEW_05_two_player_paper_analysis/fig06_persona_effects.png)

Risk-averaged persona shifts relative to the no-persona baseline, shown separately by model and protocol.

### Persona role asymmetry

![Persona role asymmetry](NEW_05_two_player_paper_analysis/fig07_persona_role_asymmetry.png)

Shows how assigning persona instructions to different seats changes Unsafe behavior.

### Complete GPT risk-persona surfaces

![Complete GPT risk-persona surfaces](NEW_05_two_player_paper_analysis/fig08_gpt_risk_persona_surfaces.png)

Full 36-of-36 persona-pair factorial surfaces for GPT-5 nano and GPT-5.4 nano.

### Partial Gemini risk-persona surface

![Partial Gemini risk-persona surface](NEW_05_two_player_paper_analysis/fig09_gemini_risk_persona_partial.png)

Only the 14-of-36 clean Gemini cells are displayed; duplicate, partial, running, and empty cells are masked rather than imputed.

### Prompt-surface sensitivity

![Prompt-surface sensitivity](NEW_05_two_player_paper_analysis/fig10_surface_sensitivity.png)

Robustness screen showing large behavioral variation across prompt variants with unchanged game semantics.

### Context and temperature diagnostic

![Context and temperature diagnostic](NEW_05_two_player_paper_analysis/fig11_context_temperature_diagnostic.png)

Diagnostic-only context-skin result retained with its failed frozen-comprehension admission warning.

### Evidence inventory

![Evidence inventory](NEW_05_two_player_paper_analysis/fig12_evidence_inventory.png)

Audit view of available, admitted, incomplete, duplicate, protocol-failed, and out-of-scope artifacts.

### Repeat-run stability

![Repeat-run stability](NEW_05_two_player_paper_analysis/fig13_repeat_run_stability.png)

Compares overlapping runs without pooling shared identifiers or treating stochastic API calls as exact replications.
