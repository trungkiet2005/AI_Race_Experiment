# FH Robustness Analysis

## Executive Summary

- **After excluding round 1, the family story still holds: Gemini remains higher-unsafe in baseline, and ChatGPT remains more model-split than family-split.** The baseline model table is the cleanest robustness cut because it removes the saturated first move and keeps model identity visible.
- **The lag pattern persists but should be read as behavioral state, not causal evidence.** ChatGPT still shows a smoother increase after unsafe history; Gemini remains opponent-sensitive and asymmetric in baseline.
- **Predictive splits remain dominated by model/state variables.** The robustness trees root on model identity in baseline scopes and on accumulated private-risk/state in wider scopes, matching the earlier full pipeline.
- **Coefficient stability is uneven by segment.** ChatGPT baseline coefficients are easier to interpret; Gemini baseline fits after dropping round 1, but the gap coefficient remains extreme, so rate/lag evidence is more trustworthy than exact Gemini magnitudes.

## Scope Gate

All robustness tables exclude incomplete/running runs, duplicate-grain rows, and first-round decisions. This makes the analysis about response dynamics after the initial move.

| scope | decisions | clusters | unsafe_rate | retry_rate | families | models |
| --- | --- | --- | --- | --- | --- | --- |
| all_round2plus | 55518 | 3369 | 0.3891 | 0.01014 | 2 | 5 |
| baseline_round2plus | 2490 | 150 | 0.5851 | 0.007631 | 2 | 5 |
| family_chatgpt_all_round2plus | 43824 | 2640 | 0.3572 | 0.0005476 | 1 | 2 |
| family_chatgpt_baseline_round2plus | 996 | 60 | 0.3494 | 0 | 1 | 2 |
| family_gemini_all_round2plus | 11694 | 729 | 0.5086 | 0.04609 | 1 | 3 |
| family_gemini_baseline_round2plus | 1494 | 90 | 0.7423 | 0.01272 | 1 | 3 |
| experiment_mode_baseline_round2plus | 2490 | 150 | 0.5851 | 0.007631 | 2 | 5 |
| experiment_mode_risk_matrix_round2plus | 43740 | 2652 | 0.3822 | 0.009488 | 2 | 3 |
| experiment_mode_strategy_persona_round2plus | 9288 | 567 | 0.3692 | 0.01389 | 2 | 3 |

## Model Identity Still Carries The Baseline Split

**Once round 1 is removed, Gemini remains high-unsafe and ChatGPT still splits sharply by model.** This supports using `model_slug` before family-level averages when explaining baseline behavior.

| family | model_slug | decisions | unsafe_rate | retry_rate | mean_own_private_risk_before |
| --- | --- | --- | --- | --- | --- |
| family_chatgpt | gpt-5-nano | 498 | 0.1386 | 0 | 0.07355 |
| family_chatgpt | gpt-5.4-nano | 498 | 0.5602 | 0 | 0.2619 |
| family_gemini | google-gemini-3.5-flash-lite | 498 | 0.6767 | 0 | 0.3807 |
| family_gemini | google-gemini-3-flash-preview | 498 | 0.7349 | 0.03815 | 0.3539 |
| family_gemini | google-gemini-3.1-flash-lite-preview | 498 | 0.8153 | 0 | 0.409 |

Visual: `figures/robustness_baseline_model_unsafe.png`.

## Lag Response Is The Strongest Mechanistic Check

**ChatGPT's baseline lag response is smoother; Gemini's is asymmetric.** This survives the round-1 exclusion, so it is not only a first-turn artifact.

| family | lag_profile | decisions | unsafe_rate |
| --- | --- | --- | --- |
| family_chatgpt | 0/0 | 498 | 0.241 |
| family_chatgpt | 0/1 | 169 | 0.4615 |
| family_chatgpt | 1/0 | 169 | 0.503 |
| family_chatgpt | 1/1 | 160 | 0.4062 |
| family_gemini | 0/0 | 122 | 0.6967 |
| family_gemini | 0/1 | 229 | 0.9301 |
| family_gemini | 1/0 | 229 | 0.3057 |
| family_gemini | 1/1 | 914 | 0.8107 |

Visual: `figures/robustness_baseline_lag_by_family.png`.

## Gap Thresholds Are Directional, Not A Single Universal Cutoff

**Being behind often raises unsafe relative to middle/ahead states, but the threshold is scope-dependent.** Treat these thresholds as diagnostic bins rather than a universal rule.

| scope | threshold | behind_decisions | behind_unsafe_rate | middle_unsafe_rate | ahead_unsafe_rate | behind_minus_middle | behind_minus_ahead |
| --- | --- | --- | --- | --- | --- | --- | --- |
| baseline_round2plus | 0.25 | 380 | 0.6605 | 0.622 | 0.3421 | 0.03856 | 0.3184 |
| baseline_round2plus | 0.5 | 380 | 0.6605 | 0.622 | 0.3421 | 0.03856 | 0.3184 |
| baseline_round2plus | 1 | 109 | 0.5229 | 0.5907 | 0.5321 | -0.06773 | -0.009174 |
| baseline_round2plus | 1.5 | 55 | 0.6 | 0.5874 | 0.4727 | 0.01261 | 0.1273 |
| baseline_round2plus | 2 | 20 | 0.7 | 0.5845 | 0.55 | 0.1155 | 0.15 |
| family_chatgpt_baseline_round2plus | 0.25 | 255 | 0.4941 | 0.1996 | 0.4902 | 0.2945 | 0.003922 |
| family_chatgpt_baseline_round2plus | 0.5 | 255 | 0.4941 | 0.1996 | 0.4902 | 0.2945 | 0.003922 |
| family_chatgpt_baseline_round2plus | 1 | 109 | 0.5229 | 0.2995 | 0.5321 | 0.2234 | -0.009174 |
| family_chatgpt_baseline_round2plus | 1.5 | 55 | 0.6 | 0.3262 | 0.4727 | 0.2738 | 0.1273 |
| family_chatgpt_baseline_round2plus | 2 | 20 | 0.7 | 0.3379 | 0.55 | 0.3621 | 0.15 |
| family_gemini_baseline_round2plus | 0.25 | 125 | 1 | 0.787 | 0.04 | 0.213 | 0.96 |
| family_gemini_baseline_round2plus | 0.5 | 125 | 1 | 0.787 | 0.04 | 0.213 | 0.96 |
| family_gemini_baseline_round2plus | 1 | 0 |  | 0.7423 |  |  |  |
| family_gemini_baseline_round2plus | 1.5 | 0 |  | 0.7423 |  |  |  |
| family_gemini_baseline_round2plus | 2 | 0 |  | 0.7423 |  |  |  |

## Logit Robustness

**The robust logit check agrees most clearly for ChatGPT/OpenAI-style baseline dynamics.** Gemini now fits after the round-1 exclusion, but its gap coefficient is still extreme, so the coefficient table is useful as a sign check, not a headline proof.

| scope | term | coef | odds_ratio | ci95_low | ci95_high | p_value | n | clusters |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| family_chatgpt_baseline_round2plus | own_prev_unsafe | 0.633 | 1.883 | 0.2679 | 0.998 | 0.0006777 | 996 | 60 |
| family_chatgpt_baseline_round2plus | opponent_prev_unsafe | 0.3605 | 1.434 | -0.0202 | 0.7411 | 0.06346 | 996 | 60 |
| family_chatgpt_baseline_round2plus | progress_gap_before | -0.09634 | 0.9082 | -0.2671 | 0.07437 | 0.2687 | 996 | 60 |
| family_gemini_baseline_round2plus | own_prev_unsafe | -0.874 | 0.4173 | -1.281 | -0.4665 | 2.629e-05 | 1494 | 90 |
| family_gemini_baseline_round2plus | opponent_prev_unsafe | 0.4835 | 1.622 | 0.1176 | 0.8495 | 0.009604 | 1494 | 90 |
| family_gemini_baseline_round2plus | progress_gap_before | -8.937 | 0.0001314 | -10.97 | -6.91 | 5.713e-18 | 1494 | 90 |

## Predictive Robustness

**Tree performance remains useful but exploratory.** Baseline root: `model_slug_gpt-5-nano`; ChatGPT baseline root: `model_slug_gpt-5-nano`; Gemini baseline root: `own_private_risk_before`.

| scope | balanced_accuracy | roc_auc | brier_score |
| --- | --- | --- | --- |
| all_round2plus | 0.7643 | 0.8144 | 0.1713 |
| baseline_round2plus | 0.7279 | 0.7903 | 0.1737 |
| experiment_mode_baseline_round2plus | 0.7279 | 0.7903 | 0.1737 |
| experiment_mode_risk_matrix_round2plus | 0.762 | 0.8078 | 0.1739 |
| experiment_mode_strategy_persona_round2plus | 0.8077 | 0.8694 | 0.1424 |
| family_chatgpt_all_round2plus | 0.7582 | 0.8176 | 0.1725 |
| family_chatgpt_baseline_round2plus | 0.7356 | 0.7594 | 0.1925 |
| family_gemini_all_round2plus | 0.8646 | 0.9243 | 0.0998 |
| family_gemini_baseline_round2plus | 0.7235 | 0.8365 | 0.1577 |

| scope | root_feature | root_threshold | decisions | clusters |
| --- | --- | --- | --- | --- |
| all_round2plus | own_private_risk_before | 0.02404 | 55518 | 3369 |
| baseline_round2plus | model_slug_gpt-5-nano | 0.5 | 2490 | 150 |
| family_chatgpt_all_round2plus | own_private_risk_before | 0.02404 | 43824 | 2640 |
| family_chatgpt_baseline_round2plus | model_slug_gpt-5-nano | 0.5 | 996 | 60 |
| family_gemini_all_round2plus | opponent_prev_unsafe | 0.5 | 11694 | 729 |
| family_gemini_baseline_round2plus | own_private_risk_before | 0.511 | 1494 | 90 |
| experiment_mode_baseline_round2plus | model_slug_gpt-5-nano | 0.5 | 2490 | 150 |
| experiment_mode_risk_matrix_round2plus | own_private_risk_before | 0.02404 | 43740 | 2652 |
| experiment_mode_strategy_persona_round2plus | own_private_risk_before | 0.02071 | 9288 | 567 |

## What To Run Next

1. Run per-model robustness reports for the highest-contrast models: `gpt-5-nano`, `gpt-5.4-nano`, and the three Gemini baseline models.
2. For Gemini, add a first-turn-specific analysis instead of forcing one pooled logit, because first-round saturation is itself the signal.
3. Add a mixed-effects or GEE specification if the paper needs a stronger repeated-game inference layer.

## Caveats

- These outputs deliberately exclude round 1; they answer response-dynamics questions, not initial-position questions.
- Trees compress behavior into rules; they are not causal estimates.
- Narrow segment logits can still hit separation or unstable covariance, especially in saturated Gemini cells.