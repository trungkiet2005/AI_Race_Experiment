# FH Analytic Report

## Stage 1: Coverage

- Raw runs discovered: 118
- Completed runs: 115
- Incomplete/running runs: 3
- Race rows: 3393
- Player rows: 6786
- Turn rows: 62868

Coverage table: `derived/coverage_audit.csv`.
Data-quality findings: `derived/data_quality_findings.csv`; retry rates by model: `derived/data_quality_retry_by_model.csv`.
Canonical tables retain duplicate-grain rows for audit, but descriptive/model stages exclude rows flagged with `duplicate_grain_key == True`.

Quality gate summary:

| check | severity | status | rows_affected | affected_rate | finding |
| --- | --- | --- | --- | --- | --- |
| run_completeness | medium | warning | 3 | 0.02542 | 3 run(s) are not marked completed. |
| races_grain | critical | fail | 48 | 0.01415 | 48 duplicated row(s) at intended grain ['source_run', 'game_id']. |
| players_grain | critical | fail | 96 | 0.01415 | 96 duplicated row(s) at intended grain ['source_run', 'game_id', 'player_index']. |
| turns_grain | critical | fail | 336 | 0.005345 | 336 duplicated row(s) at intended grain ['source_run', 'game_id', 'player_index', 'round']. |
| turn_required_fields_round2plus | low | pass | 0 | 0 | 0 missing required turn-feature value(s) for round>=2 modeling. |
| final_parse_failures | low | pass | 0 | 0 | 0 final turn parse failure(s). |
| retry_rate | medium | warning | 776 | 0.01234 | 776 decision(s) required at least one parse retry. |

## Stage 2: Descriptive Visuals

- Baseline unsafe by risk/model: `figures/baseline_unsafe_by_risk_model.png`
- Baseline unsafe by progress gap: `figures/baseline_unsafe_by_gap_bin.png`
- Baseline gap-threshold scan: `figures/baseline_gap_threshold_scan.png`
- Baseline lag-action heatmap: `figures/baseline_lag_action_heatmap.png`

Core descriptive tables are in `derived/unsafe_by_*`. Use these before fitting models.

## Stage 3: Human-Reference Checks

Human-check logit outputs: `derived/human_check_logit_coefficients.csv` and `derived/human_reference_ledger.csv`.
Baseline logit bootstrap CI: `derived/human_check_baseline_bootstrap_summary.csv`.
Baseline predicted gap curves: `figures/baseline_predicted_unsafe_by_gap.png`.
Segment stability outputs: `derived/human_check_segment_coefficients.csv` and `derived/human_check_sign_stability.csv`.
Segment models can show separation in small/saturated slices; use sign stability as a screening diagnostic and rely on pooled baseline coefficients for headline estimates.

Baseline M3 core terms:

| term | coef | odds_ratio | ci95_low | ci95_high | p_value |
| --- | --- | --- | --- | --- | --- |
| first_round_unsafe | 1.635 | 5.128 | 1.151 | 2.118 | 3.351e-11 |
| own_prev_unsafe | -0.3624 | 0.696 | -0.8526 | 0.1277 | 0.1473 |
| opponent_prev_unsafe | 1.083 | 2.955 | 0.6414 | 1.525 | 1.558e-06 |
| progress_gap_before | -1.067 | 0.3442 | -1.57 | -0.5632 | 3.275e-05 |

Baseline cluster bootstrap summary:

| term | bootstrap_successes | coef_median | coef_ci95_low | coef_ci95_high | odds_ratio_median | sign_match_share |
| --- | --- | --- | --- | --- | --- | --- |
| first_round_unsafe | 100 | 1.707 | 1.245 | 2.169 | 5.511 | 1 |
| opponent_prev_unsafe | 100 | 1.071 | 0.6204 | 1.542 | 2.919 | 1 |
| own_prev_unsafe | 100 | -0.4048 | -0.8555 | 0.006209 | 0.6671 | nan |
| progress_gap_before | 100 | -1.067 | -1.66 | -0.7269 | 0.344 | 1 |

## Stage 4: Exploratory Fit

Tree outputs are split by scope so the analysis stays sequential:

- Baseline tree CV metrics: `derived/decision_tree_baseline_completed_cv_metrics.csv`
- Baseline tree rules: `derived/decision_tree_baseline_completed_rules.txt`
- Baseline tree leaf support/confidence: `derived/decision_tree_baseline_completed_leaf_summary.csv`
- Full completed tree CV metrics: `derived/decision_tree_all_completed_cv_metrics.csv`
- Full completed tree rules: `derived/decision_tree_all_completed_rules.txt`
- Full completed root stability: `derived/decision_tree_all_completed_root_stability.csv`
- Scope summary: `derived/decision_tree_scope_summary.csv`

Interpret tree/rule outputs as exploratory compression of behaviour, not causal effects.

Tree scope summary:

| scope | decisions | clusters | unsafe_rate | cv_balanced_accuracy_mean | cv_roc_auc_mean | top_root_feature | top_root_share |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_completed | 2490 | 150 | 0.5851 | 0.7645 | 0.84 | model_slug_gpt-5-nano | 0.93 |
| all_completed | 55518 | 3369 | 0.3891 | 0.7766 | 0.8446 | own_private_risk_before | 1 |

## Suggested Next Refinements

- Decide whether incomplete Gemini cells are excluded or shown with an incomplete flag.
- Add bootstrap CIs for logit coefficients if cluster-robust CIs are not enough.
- Add shallow trees per provider/model to see whether the same rule appears across families.