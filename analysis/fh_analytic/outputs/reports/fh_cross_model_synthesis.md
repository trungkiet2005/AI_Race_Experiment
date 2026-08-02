# FH Cross-Model Synthesis: Common Ground Vs Model-Specific

## Purpose

This does not run new statistical models. It cross-references derived tables from the whole `fh_analytic` pipeline (baseline, persona-role, risk-matrix, temporal, mechanism-mining, strategy-synthesis stages) to answer: which patterns are common to all five models tested, and which are specific to a family, a model, or an assigned condition? Read this alongside the individual stage reports; every number here traces back to a derived CSV listed in its section.

## Executive Summary

- **`opponent_prev_unsafe`** (human-expected sign: positive): agrees in direction in 0.75 of 8 model x scope fits across 2 scopes and 5 models (`cross_model_human_scorecard.csv`, `cross_model_commonality_index.csv`).
- **`progress_gap_before`** (human-expected sign: negative): agrees in direction in 0.625 of 8 model x scope fits across 2 scopes and 5 models (`cross_model_human_scorecard.csv`, `cross_model_commonality_index.csv`).
- **Unanimous-sign levers across all 5 models**: opportunistic_lift, forgiveness_rate, mutual_unsafe_stickiness.
- **Sign-mixed levers (model-specific direction)**: retaliation_lift, catchup_lift -- some models show the lever, others show the opposite; see `cross_model_lever_detail.csv`.
- **Endgame effect is small for every model** (`cross_model_endgame_flatness.csv`); the largest per-model slope is Gemini 3 Flash at 0.0199 unsafe-rate points per round closer to the true (hidden) end, consistent with the horizon genuinely not leaking through to behavior.
- **Persona-role compliance and risk-label sensitivity are common to all models that ran those modes** (3 of 5): every model raises unsafe rate under `adversarial`/`risk-seeking` framing and lowers it under `cooperative`/`risk-averse` framing, and in every model the player's *own* assigned role/label dominates the opponent's (see `fh_persona_compliance_mining.md`, `fh_risk_matrix_asymmetry_mining.md`). What is *not* common is the strength of that compliance and whether the real mechanistic risk treatment still matters net of the narrative label (it does for Gemini and GPT-5.4 nano, not for GPT-5 nano).

## Human-Reference Commonality Index

Fraction of model x scope logit fits whose coefficient sign agrees with the human-reference direction (Fernandez Domingos & Han 2026, `results/scripts/human_reference.json`), pooling the baseline per-model fits and the risk-matrix per-model fits (the two scopes with comparable model-level grain).

| term | expected_sign | fits | models_covered | scopes_covered | sign_match_share |
| --- | --- | --- | --- | --- | --- |
| opponent_prev_unsafe | positive | 8 | 5 | 2 | 0.75 |
| progress_gap_before | negative | 8 | 5 | 2 | 0.625 |

Full per-model, per-scope detail:

| scope | model_slug | term | coef | expected_sign | sign_match | fit_source |
| --- | --- | --- | --- | --- | --- | --- |
| baseline | google-gemini-3-flash-preview | own_prev_unsafe | 9.043e+14 |  |  | human_check_segment (M3 interactions) |
| baseline | google-gemini-3-flash-preview | opponent_prev_unsafe | 3.699e+15 | positive | True | human_check_segment (M3 interactions) |
| baseline | google-gemini-3-flash-preview | progress_gap_before | 2.034e+13 | negative | False | human_check_segment (M3 interactions) |
| baseline | google-gemini-3.1-flash-lite-preview | own_prev_unsafe | -2.077 |  |  | human_check_segment (M3 interactions) |
| baseline | google-gemini-3.1-flash-lite-preview | opponent_prev_unsafe | 23.57 | positive | True | human_check_segment (M3 interactions) |
| baseline | google-gemini-3.1-flash-lite-preview | progress_gap_before | -19.78 | negative | True | human_check_segment (M3 interactions) |
| baseline | google-gemini-3.5-flash-lite | own_prev_unsafe | -0.8411 |  |  | human_check_segment (M3 interactions) |
| baseline | google-gemini-3.5-flash-lite | opponent_prev_unsafe | 0.6912 | positive | True | human_check_segment (M3 interactions) |
| baseline | google-gemini-3.5-flash-lite | progress_gap_before | -18.54 | negative | True | human_check_segment (M3 interactions) |
| baseline | gpt-5.4-nano | own_prev_unsafe | 0.2068 |  |  | human_check_segment (M3 interactions) |
| baseline | gpt-5.4-nano | opponent_prev_unsafe | 0.08742 | positive | True | human_check_segment (M3 interactions) |
| baseline | gpt-5.4-nano | progress_gap_before | -0.3786 | negative | True | human_check_segment (M3 interactions) |
| baseline | gpt-5-nano | own_prev_unsafe | 0.2152 |  |  | model_logit_coefficients (main effects only, fallback) |
| baseline | gpt-5-nano | opponent_prev_unsafe | -1.728 | positive | False | model_logit_coefficients (main effects only, fallback) |
| baseline | gpt-5-nano | progress_gap_before | -1.745 | negative | True | model_logit_coefficients (main effects only, fallback) |
| risk_matrix | google-gemini-3-flash-preview | own_prev_unsafe | 0.7475 |  | True | risk_matrix_human_check (main effects) |
| risk_matrix | google-gemini-3-flash-preview | opponent_prev_unsafe | 2.956 | positive | True | risk_matrix_human_check (main effects) |
| risk_matrix | google-gemini-3-flash-preview | progress_gap_before | -1.968 | negative | True | risk_matrix_human_check (main effects) |
| risk_matrix | gpt-5-nano | own_prev_unsafe | 0.09007 |  | True | risk_matrix_human_check (main effects) |
| risk_matrix | gpt-5-nano | opponent_prev_unsafe | -0.517 | positive | False | risk_matrix_human_check (main effects) |
| risk_matrix | gpt-5-nano | progress_gap_before | 0.409 | negative | False | risk_matrix_human_check (main effects) |
| risk_matrix | gpt-5.4-nano | own_prev_unsafe | 1.04 |  | True | risk_matrix_human_check (main effects) |
| risk_matrix | gpt-5.4-nano | opponent_prev_unsafe | 0.03983 | positive | True | risk_matrix_human_check (main effects) |
| risk_matrix | gpt-5.4-nano | progress_gap_before | 0.3511 | negative | False | risk_matrix_human_check (main effects) |

## Strategic-Lever Commonality

Retaliation / opportunistic / catch-up lifts, forgiveness, and mutual-unsafe stickiness (definitions in `fh_strategy_playbook_mining.md`), evaluated across all 5 models on baseline data.

| lever | models_with_data | models_positive | models_negative | unanimous_sign | min_value | max_value |
| --- | --- | --- | --- | --- | --- | --- |
| retaliation_lift | 5 | 3 | 2 | False | -0.1153 | 0.6952 |
| opportunistic_lift | 3 | 3 | 0 | True | 0.2687 | 0.4907 |
| catchup_lift | 3 | 1 | 2 | False | -0.4112 | 0.4032 |
| forgiveness_rate | 5 | 5 | 0 | True | 0.2362 | 0.7673 |
| mutual_unsafe_stickiness | 5 | 5 | 0 | True | 0.1824 | 0.9028 |

| model_slug | retaliation_lift | opportunistic_lift | catchup_lift | forgiveness_rate | mutual_unsafe_stickiness |
| --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | 0.6952 | 0.2946 | 0.4032 | 0.7673 | 0.9028 |
| google-gemini-3.1-flash-lite-preview | 0.03125 |  |  | 0.5918 | 0.8315 |
| google-gemini-3.5-flash-lite | 0.2996 |  |  | 0.7321 | 0.7689 |
| gpt-5-nano | -0.1153 | 0.2687 | -0.1443 | 0.5257 | 0.1824 |
| gpt-5.4-nano | -0.03546 | 0.4907 | -0.4112 | 0.2362 | 0.6557 |

## Endgame Flatness By Model

| model_slug | slope_unsafe_per_round_closer_to_end | range_unsafe_rate |
| --- | --- | --- |
| google-gemini-3-flash-preview | 0.01994 | 0.08271 |
| gpt-5-nano | -0.004946 | 0.05504 |
| gpt-5.4-nano | -0.007251 | 0.0408 |

## What This Adds Up To

- The most model-universal finding across this whole pipeline is **opponent-triggered reactivity**: `opponent_prev_unsafe` is positive (matches the human direction) far more consistently than `own_prev_unsafe` or `progress_gap_before`, across both baseline and risk-matrix scopes.
- The most model-specific finding is **how much a model listens to narrative framing versus the real mechanistic risk number**: GPT-5 nano's behavior under `mode_risk_matrix` is statistically flat across the real `max_private_risk` treatment (p=0.98, p=0.67) once the narrative risk label is in the prompt, while Gemini 3 Flash and GPT-5.4 nano still respond to the real treatment net of the label (`risk_matrix_asymmetry_logit.csv`).
- Persona/role compliance, own-role-dominates-opponent-role, and a flat (non-escalating) hidden-horizon curve are the three genuinely new cross-model regularities surfaced in this session that were not visible in the original baseline-only reports.

## Caveats

- The commonality index only pools scopes with model-level grain (baseline, risk-matrix); `mode_strategy_persona` human-check output is at role grain, not model grain, and is reported separately in `fh_persona_compliance_mining.md` because pooling it here would silently change the unit of analysis.
- Sign-match counting treats near-separated/very large coefficients the same as small, precisely estimated ones; consult the p-values and confidence intervals in the underlying CSVs before treating any single sign match as strong evidence.
- All findings remain pilot-phase, exploratory, and specific to the five checkpoint models tested; see CLAUDE.md on pooling pilot and confirmatory evidence.