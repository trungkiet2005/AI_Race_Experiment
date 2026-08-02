# FH Mechanism Mining

## Executive Summary

- **The cleanest common unsafe rule is not one variable; it is a state pattern.** High-unsafe rules combine prior unsafe state, being near-tied/behind, accumulated private risk, and model identity.
- **Mechanisms differ sharply by model.** Gemini models form persistent/opponent-triggered unsafe clusters; `gpt-5-nano` is mostly safe; `gpt-5.4-nano` is mixed and harder to compress with shallow rules.
- **Clustering supports the earlier tree story.** The trajectory clusters separate mostly-safe, first-turn/opponent-triggered, and persistent-unsafe behavior rather than merely splitting by provider name.
- **Random forests confirm that state/history features carry the most predictive signal.** Treat this as predictive confidence, while rule tables remain the interpretable layer for paper text.

## Common High-Unsafe Rules

| rule | support | support_share | unsafe_rate | baseline_unsafe_rate | lift |
| --- | --- | --- | --- | --- | --- |
| own_risk_state=risk_low & model_slug=google-gemini-3-flash-preview | 166 | 0.06667 | 1 | 0.5851 | 0.4149 |
| own_risk_state=risk_low & model_slug=google-gemini-3.1-flash-lite-preview | 166 | 0.06667 | 1 | 0.5851 | 0.4149 |
| opponent_risk_state=risk_low & model_slug=google-gemini-3-flash-preview | 166 | 0.06667 | 1 | 0.5851 | 0.4149 |
| opponent_risk_state=risk_low & model_slug=google-gemini-3.1-flash-lite-preview | 166 | 0.06667 | 1 | 0.5851 | 0.4149 |
| max_risk_level=risk_0.1 & model_slug=google-gemini-3-flash-preview | 166 | 0.06667 | 1 | 0.5851 | 0.4149 |
| max_risk_level=risk_0.1 & model_slug=google-gemini-3.1-flash-lite-preview | 166 | 0.06667 | 1 | 0.5851 | 0.4149 |
| prev_state=both_prev_unsafe & own_risk_state=risk_low & model_slug=google-gemini-3-flash-preview | 166 | 0.06667 | 1 | 0.5851 | 0.4149 |
| prev_state=both_prev_unsafe & own_risk_state=risk_low & model_slug=google-gemini-3.1-flash-lite-preview | 166 | 0.06667 | 1 | 0.5851 | 0.4149 |

## Common Low-Unsafe Rules

| rule | support | support_share | unsafe_rate | baseline_unsafe_rate | lift |
| --- | --- | --- | --- | --- | --- |
| gap_direction=ahead & family=family_gemini | 125 | 0.0502 | 0.04 | 0.5851 | -0.5451 |
| gap_direction=ahead & gap_magnitude=gap_0_5_1 & family=family_gemini | 125 | 0.0502 | 0.04 | 0.5851 | -0.5451 |
| prev_state=own_prev_unsafe & gap_direction=ahead & family=family_gemini | 122 | 0.049 | 0.04098 | 0.5851 | -0.5442 |
| prev_state=own_prev_unsafe & gap_magnitude=gap_0_5_1 & family=family_gemini | 122 | 0.049 | 0.04098 | 0.5851 | -0.5442 |
| round_phase=late_r9plus & model_slug=gpt-5-nano | 120 | 0.04819 | 0.05 | 0.5851 | -0.5351 |
| round_phase=late_r9plus & family=family_chatgpt & model_slug=gpt-5-nano | 120 | 0.04819 | 0.05 | 0.5851 | -0.5351 |
| prev_state=both_prev_safe & round_phase=late_r9plus & model_slug=gpt-5-nano | 106 | 0.04257 | 0.0566 | 0.5851 | -0.5285 |
| gap_direction=near_tied & own_risk_state=risk_low & model_slug=gpt-5-nano | 110 | 0.04418 | 0.06364 | 0.5851 | -0.5215 |

## Bootstrap Stability For Top Common Rules

| rule | bootstraps | unsafe_rate_median | unsafe_rate_ci_low | unsafe_rate_ci_high | lift_median | lift_ci_low | lift_ci_high |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gap_direction=near_tied & round_phase=mid_r5_8 & model_slug=gpt-5-nano | 200 | 0.06378 | 0.03326 | 0.1125 | -0.5168 | -0.5779 | -0.4651 |
| gap_direction=near_tied & own_risk_state=risk_low & model_slug=gpt-5-nano | 200 | 0.06397 | 0.0258 | 0.1364 | -0.5174 | -0.576 | -0.4553 |
| gap_magnitude=gap_0_0_5 & opponent_risk_state=risk_low & model_slug=gpt-5-nano | 200 | 0.06397 | 0.0258 | 0.1364 | -0.5174 | -0.576 | -0.4553 |
| gap_direction=near_tied & opponent_risk_state=risk_low & model_slug=gpt-5-nano | 200 | 0.06397 | 0.0258 | 0.1364 | -0.5174 | -0.576 | -0.4553 |
| gap_magnitude=gap_0_0_5 & own_risk_state=risk_low & model_slug=gpt-5-nano | 200 | 0.06397 | 0.0258 | 0.1364 | -0.5174 | -0.576 | -0.4553 |
| prev_state=both_prev_safe & round_phase=late_r9plus & model_slug=gpt-5-nano | 200 | 0.05951 | 0.009233 | 0.1275 | -0.5233 | -0.5945 | -0.4533 |
| round_phase=late_r9plus & model_slug=gpt-5-nano | 200 | 0.05283 | 0.008895 | 0.1 | -0.532 | -0.5952 | -0.4703 |
| round_phase=late_r9plus & family=family_chatgpt & model_slug=gpt-5-nano | 200 | 0.05283 | 0.008895 | 0.1 | -0.532 | -0.5952 | -0.4703 |
| prev_state=own_prev_unsafe & gap_magnitude=gap_0_5_1 & family=family_gemini | 200 | 0.04016 | 0.01105 | 0.07149 | -0.5441 | -0.605 | -0.4872 |
| prev_state=own_prev_unsafe & gap_direction=ahead & family=family_gemini | 200 | 0.04016 | 0.01105 | 0.07149 | -0.5441 | -0.605 | -0.4872 |
| gap_direction=ahead & family=family_gemini | 200 | 0.03895 | 0.01093 | 0.06781 | -0.5457 | -0.6062 | -0.4886 |
| gap_direction=ahead & gap_magnitude=gap_0_5_1 & family=family_gemini | 200 | 0.03895 | 0.01093 | 0.06781 | -0.5457 | -0.6062 | -0.4886 |

## Behavioral Clusters

| cluster_id_num | mechanism_label | players | unsafe_rate_r2 | first_round_unsafe | unsafe_after_00 | unsafe_after_01 | unsafe_after_10 | unsafe_after_11 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | first_turn_opponent_triggered | 15 | 0.7197 | 1 | 0.7165 | 0.9444 | 0.4889 | 0.7267 |
| 1 | mostly_safe | 80 | 0.2205 | 0.075 | 0.2862 | 0.2131 | 0.151 | 0.1172 |
| 2 | first_turn_opponent_triggered | 135 | 0.6294 | 0.9185 | 0.7221 | 0.8403 | 0.3993 | 0.5747 |
| 3 | persistent_unsafe | 70 | 0.8981 | 0.8714 | 0.9049 | 0.9581 | 0.8333 | 0.8857 |

## Mechanism Mix By Model

| model_slug | mechanism_label | players | share |
| --- | --- | --- | --- |
| google-gemini-3-flash-preview | first_turn_opponent_triggered | 44 | 0.7333 |
| google-gemini-3-flash-preview | persistent_unsafe | 16 | 0.2667 |
| google-gemini-3.1-flash-lite-preview | first_turn_opponent_triggered | 40 | 0.6667 |
| google-gemini-3.1-flash-lite-preview | persistent_unsafe | 20 | 0.3333 |
| google-gemini-3.5-flash-lite | first_turn_opponent_triggered | 40 | 0.6667 |
| google-gemini-3.5-flash-lite | persistent_unsafe | 20 | 0.3333 |
| gpt-5-nano | mostly_safe | 60 | 1 |
| gpt-5.4-nano | first_turn_opponent_triggered | 26 | 0.4333 |
| gpt-5.4-nano | mostly_safe | 20 | 0.3333 |
| gpt-5.4-nano | persistent_unsafe | 14 | 0.2333 |

## Random-Forest Mechanism Models

| scope | roc_auc | balanced_accuracy | brier_score |
| --- | --- | --- | --- |
| common_all_r2 | 0.8749 | 0.7871 | 0.1483 |
| common_baseline_r2 | 0.8854 | 0.7821 | 0.1393 |
| family_family_chatgpt_baseline_r2 | 0.7763 | 0.7133 | 0.1904 |
| family_family_gemini_baseline_r2 | 0.8977 | 0.7989 | 0.1309 |
| model_google-gemini-3-flash-preview_baseline_r2 | 0.9231 | 0.8315 | 0.1266 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | 0.9444 | 0.8333 | 0.1291 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | 0.8273 | 0.7472 | 0.1622 |
| model_gpt-5-nano_baseline_r2 | 0.7652 | 0.7297 | 0.1803 |
| model_gpt-5.4-nano_baseline_r2 | 0.5703 | 0.5311 | 0.2484 |

Top RF features:

| scope | feature | importance | n | clusters |
| --- | --- | --- | --- | --- |
| common_all_r2 | own_private_risk_before | 0.2151 | 55518 | 3369 |
| common_all_r2 | own_risk_state_risk_zero | 0.1723 | 55518 | 3369 |
| common_all_r2 | progress_gap_before | 0.0653 | 55518 | 3369 |
| common_all_r2 | own_prev_unsafe | 0.05855 | 55518 | 3369 |
| common_all_r2 | round | 0.03934 | 55518 | 3369 |
| common_all_r2 | model_slug_gpt-5-nano | 0.03756 | 55518 | 3369 |
| common_all_r2 | prev_state_both_prev_safe | 0.03755 | 55518 | 3369 |
| common_all_r2 | gap_direction_ahead | 0.03345 | 55518 | 3369 |
| common_baseline_r2 | model_slug_gpt-5-nano | 0.1134 | 2490 | 150 |
| common_baseline_r2 | family_family_chatgpt | 0.1107 | 2490 | 150 |
| common_baseline_r2 | family_family_gemini | 0.08755 | 2490 | 150 |
| common_baseline_r2 | own_private_risk_before | 0.08739 | 2490 | 150 |
| common_baseline_r2 | opponent_private_risk_before | 0.07423 | 2490 | 150 |
| common_baseline_r2 | opponent_prev_unsafe | 0.07337 | 2490 | 150 |
| common_baseline_r2 | progress_gap_before | 0.04412 | 2490 | 150 |
| common_baseline_r2 | model_slug_gpt-5.4-nano | 0.04088 | 2490 | 150 |
| family_family_chatgpt_baseline_r2 | model_slug_gpt-5.4-nano | 0.2176 | 996 | 60 |
| family_family_chatgpt_baseline_r2 | model_slug_gpt-5-nano | 0.2033 | 996 | 60 |
| family_family_chatgpt_baseline_r2 | gap_direction_near_tied | 0.07949 | 996 | 60 |
| family_family_chatgpt_baseline_r2 | gap_magnitude_gap_0_0_5 | 0.07918 | 996 | 60 |
| family_family_chatgpt_baseline_r2 | round | 0.071 | 996 | 60 |
| family_family_chatgpt_baseline_r2 | opponent_private_risk_before | 0.05545 | 996 | 60 |
| family_family_chatgpt_baseline_r2 | own_private_risk_before | 0.037 | 996 | 60 |
| family_family_chatgpt_baseline_r2 | round_phase_early_r2_4 | 0.03475 | 996 | 60 |
| family_family_gemini_baseline_r2 | own_private_risk_before | 0.1867 | 1494 | 90 |
| family_family_gemini_baseline_r2 | progress_gap_before | 0.1038 | 1494 | 90 |
| family_family_gemini_baseline_r2 | opponent_private_risk_before | 0.09264 | 1494 | 90 |
| family_family_gemini_baseline_r2 | prev_state_own_prev_unsafe | 0.07099 | 1494 | 90 |
| family_family_gemini_baseline_r2 | gap_direction_ahead | 0.06711 | 1494 | 90 |
| family_family_gemini_baseline_r2 | opponent_prev_unsafe | 0.05806 | 1494 | 90 |
| family_family_gemini_baseline_r2 | max_private_risk | 0.05602 | 1494 | 90 |
| family_family_gemini_baseline_r2 | opponent_risk_state_risk_low | 0.04686 | 1494 | 90 |
| model_google-gemini-3-flash-preview_baseline_r2 | own_private_risk_before | 0.2454 | 498 | 30 |
| model_google-gemini-3-flash-preview_baseline_r2 | opponent_private_risk_before | 0.1365 | 498 | 30 |
| model_google-gemini-3-flash-preview_baseline_r2 | max_private_risk | 0.1069 | 498 | 30 |
| model_google-gemini-3-flash-preview_baseline_r2 | own_risk_state_risk_low | 0.08605 | 498 | 30 |
| model_google-gemini-3-flash-preview_baseline_r2 | opponent_risk_state_risk_low | 0.07908 | 498 | 30 |
| model_google-gemini-3-flash-preview_baseline_r2 | opponent_prev_unsafe | 0.05083 | 498 | 30 |
| model_google-gemini-3-flash-preview_baseline_r2 | round | 0.04256 | 498 | 30 |
| model_google-gemini-3-flash-preview_baseline_r2 | progress_gap_before | 0.04036 | 498 | 30 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | own_private_risk_before | 0.277 | 498 | 30 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | opponent_private_risk_before | 0.1611 | 498 | 30 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | max_private_risk | 0.09646 | 498 | 30 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | own_risk_state_risk_low | 0.08182 | 498 | 30 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | opponent_risk_state_risk_low | 0.07621 | 498 | 30 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | round | 0.05103 | 498 | 30 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | own_prev_unsafe | 0.04271 | 498 | 30 |
| model_google-gemini-3.1-flash-lite-preview_baseline_r2 | own_risk_state_risk_mid | 0.03594 | 498 | 30 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | progress_gap_before | 0.1433 | 498 | 30 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | prev_state_own_prev_unsafe | 0.1364 | 498 | 30 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | opponent_prev_unsafe | 0.1202 | 498 | 30 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | gap_direction_ahead | 0.115 | 498 | 30 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | own_private_risk_before | 0.09457 | 498 | 30 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | opponent_private_risk_before | 0.05141 | 498 | 30 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | gap_direction_behind | 0.05105 | 498 | 30 |
| model_google-gemini-3.5-flash-lite_baseline_r2 | prev_state_opponent_prev_unsafe | 0.0355 | 498 | 30 |
| model_gpt-5-nano_baseline_r2 | round | 0.2628 | 498 | 30 |
| model_gpt-5-nano_baseline_r2 | round_phase_early_r2_4 | 0.1065 | 498 | 30 |
| model_gpt-5-nano_baseline_r2 | own_private_risk_before | 0.08834 | 498 | 30 |
| model_gpt-5-nano_baseline_r2 | round_phase_late_r9plus | 0.07064 | 498 | 30 |
| model_gpt-5-nano_baseline_r2 | opponent_private_risk_before | 0.06352 | 498 | 30 |
| model_gpt-5-nano_baseline_r2 | gap_direction_behind | 0.05557 | 498 | 30 |
| model_gpt-5-nano_baseline_r2 | progress_gap_before | 0.04567 | 498 | 30 |
| model_gpt-5-nano_baseline_r2 | own_risk_state_risk_zero | 0.04299 | 498 | 30 |
| model_gpt-5.4-nano_baseline_r2 | opponent_private_risk_before | 0.121 | 498 | 30 |
| model_gpt-5.4-nano_baseline_r2 | own_private_risk_before | 0.1125 | 498 | 30 |
| model_gpt-5.4-nano_baseline_r2 | progress_gap_before | 0.08338 | 498 | 30 |
| model_gpt-5.4-nano_baseline_r2 | prev_state_both_prev_unsafe | 0.08252 | 498 | 30 |
| model_gpt-5.4-nano_baseline_r2 | round | 0.07705 | 498 | 30 |
| model_gpt-5.4-nano_baseline_r2 | max_private_risk | 0.05649 | 498 | 30 |
| model_gpt-5.4-nano_baseline_r2 | gap_magnitude_gap_0_5_1 | 0.04698 | 498 | 30 |
| model_gpt-5.4-nano_baseline_r2 | own_prev_unsafe | 0.04527 | 498 | 30 |

## Visuals

- `figures/mechanism_mining/fh_mechanism_storyboard_contact_sheet.png`
- `figures/mechanism_mining/01_top_high_unsafe_rules.png`
- `figures/mechanism_mining/02_top_low_unsafe_rules.png`
- `figures/mechanism_mining/03_cluster_profiles_heatmap.png`
- `figures/mechanism_mining/04_cluster_mix_by_model.png`
- `figures/mechanism_mining/05_rf_feature_importance.png`
- `figures/mechanism_mining/06_rf_predictive_strength.png`

## Caveats

- Rules are descriptive conjunctions; they are interpretable, not causal.
- Clusters depend on chosen trajectory features and KMeans geometry; labels are generated from cluster centroids.
- RF importances are predictive diagnostics and can share signal across correlated state features.