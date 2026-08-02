# FH Risk-Matrix Asymmetry Mining

## Scope

`mode_risk_matrix` decisions: 49,080 across 2,670 races -- the single largest slice of collected data, bigger than baseline and `mode_strategy_persona` combined, and previously reported only as a one-line aggregate unsafe rate. Each seat is assigned a narrative risk-framing label `risk-1` .. `risk-6`, independent of the real mechanistic private-risk treatment (`max_private_risk` in {0.1, 0.6, 0.9}); the full 6x6 own/opponent label grid (`R1_R1` .. `R6_R6`) is run, so this is the one mode where own-label and opponent-label effects are genuinely separable.

## Executive Summary

- **The narrative risk label moves behavior on its own, separate from the real private-risk treatment.** `own_risk_label` and `C(max_private_risk)` are both included in the same logit; see the coefficient table below for whether the narrative label survives once the real mechanistic risk is controlled for.
- **Own risk label dominates opponent risk label** in every model that fit: own-label coefficients range 0.929 to 2.03 (all own-label p-values: 2.64e-127, 0, 0) versus opponent-label coefficients -0.114 to 0.252 (p-values: 0.000208, 0.435, 1.96e-05).
- Real private-risk treatment (`max_private_risk`) coefficients, net of the narrative label: see `risk_matrix_asymmetry_logit.csv` rows starting `C(max_private_risk)`.
- **Relative label gap (own minus opponent) also matters**, not just the own level in isolation -- see `risk_matrix_label_gap.csv` and the figure below for whether the relationship is monotonic or flattens out.
- **Human-reference lag/gap signs**: 0.667 of model x term sign checks against Fernandez Domingos & Han (2026) agree in direction within this mode (`risk_matrix_human_check.csv`).

## Own Vs Opponent Risk-Label Marginals

| model_slug | own_risk_label | n | unsafe_rate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | 1 | 2790 | 0.3079 | 0.2908 | 0.325 |
| google-gemini-3-flash-preview | 2 | 2469 | 0.3256 | 0.3072 | 0.3441 |
| google-gemini-3-flash-preview | 3 | 1620 | 0.5827 | 0.5587 | 0.6067 |
| google-gemini-3-flash-preview | 4 | 909 | 0.6348 | 0.6035 | 0.6661 |
| google-gemini-3-flash-preview | 5 | 558 | 0.862 | 0.8334 | 0.8906 |
| google-gemini-3-flash-preview | 6 | 558 | 0.9355 | 0.9151 | 0.9559 |
| gpt-5-nano | 1 | 3348 | 0.01762 | 0.01317 | 0.02208 |
| gpt-5-nano | 2 | 3348 | 0.01344 | 0.00954 | 0.01734 |
| gpt-5-nano | 3 | 3348 | 0.1655 | 0.1529 | 0.1781 |
| gpt-5-nano | 4 | 3348 | 0.4546 | 0.4377 | 0.4715 |
| gpt-5-nano | 5 | 3348 | 0.4982 | 0.4813 | 0.5151 |
| gpt-5-nano | 6 | 3348 | 0.5185 | 0.5016 | 0.5354 |
| gpt-5.4-nano | 1 | 3348 | 0.003883 | 0.001776 | 0.00599 |
| gpt-5.4-nano | 2 | 3348 | 0.002389 | 0.0007356 | 0.004043 |
| gpt-5.4-nano | 3 | 3348 | 0.3462 | 0.3301 | 0.3623 |
| gpt-5.4-nano | 4 | 3348 | 0.4719 | 0.455 | 0.4888 |
| gpt-5.4-nano | 5 | 3348 | 0.7861 | 0.7723 | 0.8 |
| gpt-5.4-nano | 6 | 3348 | 0.9848 | 0.9806 | 0.9889 |

| model_slug | opponent_risk_label | n | unsafe_rate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | 1 | 2790 | 0.3817 | 0.3637 | 0.3997 |
| google-gemini-3-flash-preview | 2 | 2469 | 0.3827 | 0.3636 | 0.4019 |
| google-gemini-3-flash-preview | 3 | 1620 | 0.5389 | 0.5146 | 0.5632 |
| google-gemini-3-flash-preview | 4 | 909 | 0.5435 | 0.5111 | 0.5758 |
| google-gemini-3-flash-preview | 5 | 558 | 0.724 | 0.6869 | 0.7611 |
| google-gemini-3-flash-preview | 6 | 558 | 0.7276 | 0.6907 | 0.7645 |
| gpt-5-nano | 1 | 3348 | 0.3014 | 0.2858 | 0.3169 |
| gpt-5-nano | 2 | 3348 | 0.2939 | 0.2785 | 0.3093 |
| gpt-5-nano | 3 | 3348 | 0.2769 | 0.2617 | 0.292 |
| gpt-5-nano | 4 | 3348 | 0.2655 | 0.2506 | 0.2805 |
| gpt-5-nano | 5 | 3348 | 0.2676 | 0.2526 | 0.2826 |
| gpt-5-nano | 6 | 3348 | 0.2625 | 0.2476 | 0.2774 |
| gpt-5.4-nano | 1 | 3348 | 0.4812 | 0.4643 | 0.4981 |
| gpt-5.4-nano | 2 | 3348 | 0.4791 | 0.4622 | 0.496 |
| gpt-5.4-nano | 3 | 3348 | 0.4498 | 0.433 | 0.4667 |
| gpt-5.4-nano | 4 | 3348 | 0.4188 | 0.402 | 0.4355 |
| gpt-5.4-nano | 5 | 3348 | 0.3871 | 0.3706 | 0.4036 |
| gpt-5.4-nano | 6 | 3348 | 0.3793 | 0.3629 | 0.3958 |

Visual: `figures/risk_matrix_asymmetry/01_own_vs_opponent_marginal.png`.

## Relative Label Gap (Own Minus Opponent)

| model_slug | label_gap_own_minus_opp | n | unsafe_rate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | -5 | 279 | 0.6667 | 0.6114 | 0.722 |
| google-gemini-3-flash-preview | -4 | 558 | 0.7384 | 0.7019 | 0.7748 |
| google-gemini-3-flash-preview | -3 | 837 | 0.5795 | 0.546 | 0.6129 |
| google-gemini-3-flash-preview | -2 | 837 | 0.4337 | 0.4001 | 0.4673 |
| google-gemini-3-flash-preview | -1 | 1146 | 0.2469 | 0.222 | 0.2719 |
| google-gemini-3-flash-preview | 0 | 1590 | 0.239 | 0.218 | 0.26 |
| google-gemini-3-flash-preview | 1 | 1146 | 0.2784 | 0.2524 | 0.3043 |
| google-gemini-3-flash-preview | 2 | 837 | 0.5102 | 0.4763 | 0.544 |
| google-gemini-3-flash-preview | 3 | 837 | 0.6894 | 0.658 | 0.7207 |
| google-gemini-3-flash-preview | 4 | 558 | 0.8943 | 0.8688 | 0.9198 |
| google-gemini-3-flash-preview | 5 | 279 | 0.9176 | 0.8853 | 0.9498 |
| gpt-5-nano | -5 | 558 | 0.03405 | 0.019 | 0.0491 |
| gpt-5-nano | -4 | 1116 | 0.02151 | 0.01299 | 0.03002 |
| gpt-5-nano | -3 | 1674 | 0.08961 | 0.07592 | 0.1033 |
| gpt-5-nano | -2 | 2232 | 0.155 | 0.14 | 0.17 |
| gpt-5-nano | -1 | 2790 | 0.2147 | 0.1995 | 0.2299 |
| gpt-5-nano | 0 | 3348 | 0.253 | 0.2383 | 0.2677 |
| gpt-5-nano | 1 | 2790 | 0.3032 | 0.2862 | 0.3203 |
| gpt-5-nano | 2 | 2232 | 0.397 | 0.3767 | 0.4173 |
| gpt-5-nano | 3 | 1674 | 0.5335 | 0.5096 | 0.5574 |
| gpt-5-nano | 4 | 1116 | 0.5806 | 0.5517 | 0.6096 |
| gpt-5-nano | 5 | 558 | 0.5842 | 0.5433 | 0.6251 |
| gpt-5.4-nano | -5 | 558 | 0.007168 | 0.0001686 | 0.01417 |
| gpt-5.4-nano | -4 | 1116 | 0.00448 | 0.0005619 | 0.008399 |
| gpt-5.4-nano | -3 | 1674 | 0.08303 | 0.06982 | 0.09625 |
| gpt-5.4-nano | -2 | 2232 | 0.1478 | 0.1331 | 0.1626 |
| gpt-5.4-nano | -1 | 2790 | 0.2735 | 0.2569 | 0.29 |
| gpt-5.4-nano | 0 | 3348 | 0.4182 | 0.4015 | 0.4349 |
| gpt-5.4-nano | 1 | 2790 | 0.5459 | 0.5274 | 0.5644 |
| gpt-5.4-nano | 2 | 2232 | 0.7151 | 0.6963 | 0.7338 |
| gpt-5.4-nano | 3 | 1674 | 0.822 | 0.8037 | 0.8403 |
| gpt-5.4-nano | 4 | 1116 | 0.9005 | 0.883 | 0.9181 |
| gpt-5.4-nano | 5 | 558 | 0.9821 | 0.9711 | 0.9931 |

Visual: `figures/risk_matrix_asymmetry/02_label_gap_effect.png`.

## Real Private-Risk Treatment Vs Narrative Label

Own-label unsafe rate broken out by the real `max_private_risk` treatment; if rows are similar across `max_private_risk` for a fixed label, the narrative label is doing most of the work.

| model_slug | max_private_risk | own_risk_label | n | unsafe_rate |
| --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | 0.1 | 1 | 930 | 0.414 |
| google-gemini-3-flash-preview | 0.1 | 2 | 823 | 0.3888 |
| google-gemini-3-flash-preview | 0.1 | 3 | 540 | 0.7389 |
| google-gemini-3-flash-preview | 0.1 | 4 | 303 | 0.8119 |
| google-gemini-3-flash-preview | 0.1 | 5 | 186 | 0.9839 |
| google-gemini-3-flash-preview | 0.1 | 6 | 186 | 0.9946 |
| google-gemini-3-flash-preview | 0.6 | 1 | 930 | 0.2806 |
| google-gemini-3-flash-preview | 0.6 | 2 | 823 | 0.3171 |
| google-gemini-3-flash-preview | 0.6 | 3 | 540 | 0.5722 |
| google-gemini-3-flash-preview | 0.6 | 4 | 303 | 0.604 |
| google-gemini-3-flash-preview | 0.6 | 5 | 186 | 0.8441 |
| google-gemini-3-flash-preview | 0.6 | 6 | 186 | 0.9409 |
| google-gemini-3-flash-preview | 0.9 | 1 | 930 | 0.229 |
| google-gemini-3-flash-preview | 0.9 | 2 | 823 | 0.271 |
| google-gemini-3-flash-preview | 0.9 | 3 | 540 | 0.437 |
| google-gemini-3-flash-preview | 0.9 | 4 | 303 | 0.4884 |
| google-gemini-3-flash-preview | 0.9 | 5 | 186 | 0.7581 |
| google-gemini-3-flash-preview | 0.9 | 6 | 186 | 0.871 |
| gpt-5-nano | 0.1 | 1 | 1116 | 0.0233 |
| gpt-5-nano | 0.1 | 2 | 1116 | 0.01523 |
| gpt-5-nano | 0.1 | 3 | 1116 | 0.1523 |
| gpt-5-nano | 0.1 | 4 | 1116 | 0.4498 |
| gpt-5-nano | 0.1 | 5 | 1116 | 0.5063 |
| gpt-5-nano | 0.1 | 6 | 1116 | 0.5161 |
| gpt-5-nano | 0.6 | 1 | 1116 | 0.01434 |
| gpt-5-nano | 0.6 | 2 | 1116 | 0.01344 |
| gpt-5-nano | 0.6 | 3 | 1116 | 0.1711 |
| gpt-5-nano | 0.6 | 4 | 1116 | 0.4615 |
| gpt-5-nano | 0.6 | 5 | 1116 | 0.4749 |
| gpt-5-nano | 0.6 | 6 | 1116 | 0.5242 |
| gpt-5-nano | 0.9 | 1 | 1116 | 0.01523 |
| gpt-5-nano | 0.9 | 2 | 1116 | 0.01165 |
| gpt-5-nano | 0.9 | 3 | 1116 | 0.1729 |
| gpt-5-nano | 0.9 | 4 | 1116 | 0.4525 |
| gpt-5-nano | 0.9 | 5 | 1116 | 0.5134 |
| gpt-5-nano | 0.9 | 6 | 1116 | 0.5152 |
| gpt-5.4-nano | 0.1 | 1 | 1116 | 0.002688 |
| gpt-5.4-nano | 0.1 | 2 | 1116 | 0.0008961 |
| gpt-5.4-nano | 0.1 | 3 | 1116 | 0.3566 |
| gpt-5.4-nano | 0.1 | 4 | 1116 | 0.4892 |
| gpt-5.4-nano | 0.1 | 5 | 1116 | 0.8109 |
| gpt-5.4-nano | 0.1 | 6 | 1116 | 0.9892 |
| gpt-5.4-nano | 0.6 | 1 | 1116 | 0.002688 |
| gpt-5.4-nano | 0.6 | 2 | 1116 | 0.002688 |
| gpt-5.4-nano | 0.6 | 3 | 1116 | 0.3665 |
| gpt-5.4-nano | 0.6 | 4 | 1116 | 0.4892 |
| gpt-5.4-nano | 0.6 | 5 | 1116 | 0.8118 |
| gpt-5.4-nano | 0.6 | 6 | 1116 | 0.9884 |
| gpt-5.4-nano | 0.9 | 1 | 1116 | 0.006272 |
| gpt-5.4-nano | 0.9 | 2 | 1116 | 0.003584 |
| gpt-5.4-nano | 0.9 | 3 | 1116 | 0.3154 |
| gpt-5.4-nano | 0.9 | 4 | 1116 | 0.4373 |
| gpt-5.4-nano | 0.9 | 5 | 1116 | 0.7357 |
| gpt-5.4-nano | 0.9 | 6 | 1116 | 0.9767 |

## Temporal Trend By Own-Label Band

| model_slug | own_label_band | round_phase | n | unsafe_rate |
| --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | own_high_5_6 | round_1 | 120 | 1 |
| google-gemini-3-flash-preview | own_high_5_6 | early_r2_4 | 360 | 0.775 |
| google-gemini-3-flash-preview | own_high_5_6 | mid_r5_8 | 396 | 0.947 |
| google-gemini-3-flash-preview | own_high_5_6 | late_r9plus | 240 | 0.9542 |
| google-gemini-3-flash-preview | own_low_1_2 | round_1 | 570 | 0 |
| google-gemini-3-flash-preview | own_low_1_2 | early_r2_4 | 1710 | 0.3029 |
| google-gemini-3-flash-preview | own_low_1_2 | mid_r5_8 | 1881 | 0.3806 |
| google-gemini-3-flash-preview | own_low_1_2 | late_r9plus | 1098 | 0.3907 |
| google-gemini-3-flash-preview | own_mid_3_4 | round_1 | 330 | 0.9879 |
| google-gemini-3-flash-preview | own_mid_3_4 | early_r2_4 | 810 | 0.3728 |
| google-gemini-3-flash-preview | own_mid_3_4 | mid_r5_8 | 891 | 0.6061 |
| google-gemini-3-flash-preview | own_mid_3_4 | late_r9plus | 498 | 0.7088 |
| gpt-5-nano | own_high_5_6 | round_1 | 720 | 0.07639 |
| gpt-5-nano | own_high_5_6 | early_r2_4 | 2160 | 0.6806 |
| gpt-5-nano | own_high_5_6 | mid_r5_8 | 2376 | 0.529 |
| gpt-5-nano | own_high_5_6 | late_r9plus | 1440 | 0.4319 |
| gpt-5-nano | own_low_1_2 | round_1 | 720 | 0 |
| gpt-5-nano | own_low_1_2 | early_r2_4 | 2160 | 0.01065 |
| gpt-5-nano | own_low_1_2 | mid_r5_8 | 2376 | 0.02567 |
| gpt-5-nano | own_low_1_2 | late_r9plus | 1440 | 0.01389 |
| gpt-5-nano | own_mid_3_4 | round_1 | 720 | 0.02778 |
| gpt-5-nano | own_mid_3_4 | early_r2_4 | 2160 | 0.4 |
| gpt-5-nano | own_mid_3_4 | mid_r5_8 | 2376 | 0.3304 |
| gpt-5-nano | own_mid_3_4 | late_r9plus | 1440 | 0.2826 |
| gpt-5.4-nano | own_high_5_6 | round_1 | 720 | 1 |
| gpt-5.4-nano | own_high_5_6 | early_r2_4 | 2160 | 0.8917 |
| gpt-5.4-nano | own_high_5_6 | mid_r5_8 | 2376 | 0.8674 |
| gpt-5.4-nano | own_high_5_6 | late_r9plus | 1440 | 0.8486 |
| gpt-5.4-nano | own_low_1_2 | round_1 | 720 | 0 |
| gpt-5.4-nano | own_low_1_2 | early_r2_4 | 2160 | 0.002315 |
| gpt-5.4-nano | own_low_1_2 | mid_r5_8 | 2376 | 0.004209 |
| gpt-5.4-nano | own_low_1_2 | late_r9plus | 1440 | 0.004167 |
| gpt-5.4-nano | own_mid_3_4 | round_1 | 720 | 0.6958 |
| gpt-5.4-nano | own_mid_3_4 | early_r2_4 | 2160 | 0.4023 |
| gpt-5.4-nano | own_mid_3_4 | mid_r5_8 | 2376 | 0.3721 |
| gpt-5.4-nano | own_mid_3_4 | late_r9plus | 1440 | 0.3368 |

## Own-Vs-Opponent Asymmetry Logit

Cluster-robust logit of `unsafe` on own risk label, opponent risk label, the real `max_private_risk` treatment, progress gap, and lag terms, fit per model on round >= 2 decisions.

| model_slug | term | coef | odds_ratio | p_value | n | clusters |
| --- | --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | Intercept | -6.074 | 0.002303 | 4.749e-151 | 7884 | 492 |
| google-gemini-3-flash-preview | C(max_private_risk)[T.0.6000000000000001] | -0.9973 | 0.3689 | 4.45e-18 | 7884 | 492 |
| google-gemini-3-flash-preview | C(max_private_risk)[T.0.9] | -1.585 | 0.205 | 2.102e-43 | 7884 | 492 |
| google-gemini-3-flash-preview | own_risk_label | 2.029 | 7.605 | 2.642e-127 | 7884 | 492 |
| google-gemini-3-flash-preview | opponent_risk_label | 0.2521 | 1.287 | 0.0002081 | 7884 | 492 |
| google-gemini-3-flash-preview | progress_gap_before | -4.451 | 0.01166 | 3.669e-22 | 7884 | 492 |
| google-gemini-3-flash-preview | own_prev_unsafe | 0.1043 | 1.11 | 0.4888 | 7884 | 492 |
| google-gemini-3-flash-preview | opponent_prev_unsafe | 2.104 | 8.2 | 6.932e-78 | 7884 | 492 |
| gpt-5-nano | Intercept | -4.001 | 0.0183 | 4.696e-275 | 17928 | 1080 |
| gpt-5-nano | C(max_private_risk)[T.0.6000000000000001] | 0.001534 | 1.002 | 0.9821 | 17928 | 1080 |
| gpt-5-nano | C(max_private_risk)[T.0.9] | 0.0293 | 1.03 | 0.6655 | 17928 | 1080 |
| gpt-5-nano | own_risk_label | 0.9294 | 2.533 | 0 | 17928 | 1080 |
| gpt-5-nano | opponent_risk_label | 0.01512 | 1.015 | 0.4352 | 17928 | 1080 |
| gpt-5-nano | progress_gap_before | -0.08059 | 0.9226 | 0.01006 | 17928 | 1080 |
| gpt-5-nano | own_prev_unsafe | -0.7061 | 0.4936 | 5.264e-36 | 17928 | 1080 |
| gpt-5-nano | opponent_prev_unsafe | -1.074 | 0.3418 | 1.985e-66 | 17928 | 1080 |
| gpt-5.4-nano | Intercept | -5.22 | 0.005405 | 0 | 17928 | 1080 |
| gpt-5.4-nano | C(max_private_risk)[T.0.6000000000000001] | 0.03073 | 1.031 | 0.6681 | 17928 | 1080 |
| gpt-5.4-nano | C(max_private_risk)[T.0.9] | -0.3156 | 0.7294 | 3.943e-06 | 17928 | 1080 |
| gpt-5.4-nano | own_risk_label | 1.471 | 4.354 | 0 | 17928 | 1080 |
| gpt-5.4-nano | opponent_risk_label | -0.1135 | 0.8927 | 1.958e-05 | 17928 | 1080 |
| gpt-5.4-nano | progress_gap_before | 0.06559 | 1.068 | 0.01546 | 17928 | 1080 |
| gpt-5.4-nano | own_prev_unsafe | -0.06221 | 0.9397 | 0.3522 | 17928 | 1080 |
| gpt-5.4-nano | opponent_prev_unsafe | -0.4236 | 0.6547 | 4.403e-08 | 17928 | 1080 |

## Human-Reference Check Within Risk-Matrix Mode

| model_slug | term | coef | human_value | expected_sign | sign_match | p_value | n | phi_U_scope |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | own_prev_unsafe | 0.7475 | -0.193 |  | True | 1.285e-15 | 7884 | 0.4745 |
| google-gemini-3-flash-preview | opponent_prev_unsafe | 2.956 | 0.607 | positive | True | 2.648e-195 | 7884 | 0.4745 |
| google-gemini-3-flash-preview | progress_gap_before | -1.968 | -0.296 | negative | True | 5.846e-14 | 7884 | 0.4745 |
| gpt-5-nano | own_prev_unsafe | 0.09007 | -0.193 |  | True | 0.1084 | 17928 | 0.3073 |
| gpt-5-nano | opponent_prev_unsafe | -0.517 | 0.607 | positive | False | 8.694e-19 | 17928 | 0.3073 |
| gpt-5-nano | progress_gap_before | 0.409 | -0.296 | negative | False | 1.176e-139 | 17928 | 0.3073 |
| gpt-5.4-nano | own_prev_unsafe | 1.04 | -0.193 |  | True | 3.677e-62 | 17928 | 0.4166 |
| gpt-5.4-nano | opponent_prev_unsafe | 0.03983 | 0.607 | positive | True | 0.5508 | 17928 | 0.4166 |
| gpt-5.4-nano | progress_gap_before | 0.3511 | -0.296 | negative | False | 5.014e-56 | 17928 | 0.4166 |

## Caveats

- `own_risk_label`/`opponent_risk_label` are treated as continuous (1-6) in the logit for parsimony; the marginal tables above show the raw per-level rates in case the relationship is non-monotonic.
- This mode's real `max_private_risk` treatment (0.1/0.6/0.9) is the same mechanism used in baseline; the narrative risk label (`risk-1`..`risk-6`) is an independent prompt-level manipulation layered on top, not part of the paper-faithful mechanism itself.
- Descriptive/mechanistic evidence only; not a causal claim about model 'understanding' of the framing.