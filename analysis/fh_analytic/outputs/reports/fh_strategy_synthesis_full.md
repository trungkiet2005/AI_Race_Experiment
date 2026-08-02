# Full Strategy Synthesis

## What Was Tested

This stage combines four views of strategy: expanded deterministic rules, position-conditioned behavior, decision-tree rules by model/family, and unsupervised HMM+clustering embeddings.

## HMM and Cluster Model Selection

- Best HMM state count by BIC: 6
- Best embedding cluster count by silhouette: 9

## Latent HMM States

- State 2: unsafe emission 0.0%, ahead 0.0%, behind 0.0%, self-transition 87.2%; top tokens: a0_gtied_pSS:0.90; a0_gtied_pSU:0.10; a0_gtied_pUU:0.00; a0_gtied_pUS:0.00; a1_gtied_pSS:0.00; a1_gbehind_pSU:0.00.
- State 0: unsafe emission 14.4%, ahead 0.0%, behind 99.6%, self-transition 99.4%; top tokens: a0_gbehind_pSU:0.57; a0_gbehind_pSS:0.21; a1_gbehind_pSU:0.07; a0_gbehind_pUU:0.06; a1_gbehind_pSS:0.04; a1_gbehind_pUU:0.03.
- State 4: unsafe emission 34.5%, ahead 0.0%, behind 0.0%, self-transition 0.0%; top tokens: a0_gtied_p??:0.65; a1_gtied_p??:0.35; a1_gbehind_pUS:0.00; a0_gahead_pSU:0.00; a0_gtied_pSU:0.00; a0_gtied_pUS:0.00.
- State 1: unsafe emission 52.9%, ahead 0.8%, behind 1.2%, self-transition 77.7%; top tokens: a1_gtied_pSS:0.23; a0_gtied_pUU:0.17; a0_gtied_pUS:0.17; a1_gtied_pSU:0.14; a1_gtied_pUS:0.10; a0_gtied_pSS:0.07.
- State 3: unsafe emission 66.8%, ahead 99.6%, behind 0.0%, self-transition 99.2%; top tokens: a1_gahead_pUS:0.45; a0_gahead_pUS:0.19; a1_gahead_pSS:0.14; a0_gahead_pSS:0.10; a1_gahead_pUU:0.07; a0_gahead_pUU:0.03.
- State 5: unsafe emission 99.0%, ahead 0.2%, behind 0.0%, self-transition 87.0%; top tokens: a1_gtied_pUU:0.88; a1_gtied_pUS:0.09; a1_gtied_pSU:0.02; a0_gtied_pUU:0.00; a0_gtied_pUS:0.00; a0_gahead_pUS:0.00.

## Discovered Strategy Groups

- C5 `ổn định an toàn`: share 21.8%, unsafe 1.7%, positional_delta n/a, retaliation_lift 1.0%, forgiveness 89.4%, stickiness_UU 11.6%.
- C1 `ổn định an toàn`: share 16.9%, unsafe 5.3%, positional_delta n/a, retaliation_lift 1.1%, forgiveness 95.4%, stickiness_UU 12.0%.
- C0 `đánh thử rồi hạ nhiệt`: share 14.0%, unsafe 42.8%, positional_delta n/a, retaliation_lift -14.0%, forgiveness 73.4%, stickiness_UU 20.4%.
- C4 `đánh thử rồi hạ nhiệt`: share 12.5%, unsafe 52.1%, positional_delta n/a, retaliation_lift 35.3%, forgiveness 71.4%, stickiness_UU 33.9%.
- C7 `đánh để gỡ khi tụt`: share 11.5%, unsafe 53.2%, positional_delta -50.0%, retaliation_lift -16.7%, forgiveness 53.8%, stickiness_UU 53.7%.
- C2 `ổn định unsafe`: share 10.5%, unsafe 89.1%, positional_delta n/a, retaliation_lift -0.7%, forgiveness 8.8%, stickiness_UU 82.0%.
- C3 `ổn định unsafe`: share 8.7%, unsafe 93.1%, positional_delta n/a, retaliation_lift 12.1%, forgiveness 28.7%, stickiness_UU 96.6%.
- C8 `thích nghi hỗn hợp`: share 3.6%, unsafe 32.3%, positional_delta 0.0%, retaliation_lift -37.2%, forgiveness 63.7%, stickiness_UU 19.3%.
- C6 `ổn định unsafe`: share 0.5%, unsafe 100.0%, positional_delta n/a, retaliation_lift n/a, forgiveness n/a, stickiness_UU n/a.

## Model Mix

- GPT-5 nano: ổn định an toàn 44.2%, đánh thử rồi hạ nhiệt 30.3%, đánh để gỡ khi tụt 15.4%.
- GPT-5.4 nano: ổn định an toàn 40.2%, ổn định unsafe 28.8%, đánh thử rồi hạ nhiệt 16.0%.
- Gemini 3 Flash: đánh thử rồi hạ nhiệt 37.0%, ổn định unsafe 29.5%, ổn định an toàn 28.5%.
- Gemini 3.1 Flash Lite: ổn định unsafe 56.7%, đánh thử rồi hạ nhiệt 43.3%.
- Gemini 3.5 Flash Lite: đánh thử rồi hạ nhiệt 70.0%, ổn định unsafe 30.0%.

## Reusable Tree Rules

- model `google-gemini-3.1-flash-lite-preview`: unsafe 100.0%, support 49.1%; rule: `own_private_risk_before <= 0.668 AND own_private_risk_before <= 0.495 AND round <= 8.500`.
- model `gpt-5.4-nano`: unsafe 100.0%, support 5.9%; rule: `first_round_unsafe > 0.500 AND progress_gap_before > -0.250 AND own_private_risk_before <= 0.005`.
- model `google-gemini-3-flash-preview`: safe 0.7%, support 34.6%; rule: `opponent_prev_unsafe <= 0.500 AND first_round_unsafe <= 0.500 AND progress_gap_before > -0.250`.
- family `family_gemini`: safe 0.7%, support 31.7%; rule: `opponent_prev_unsafe <= 0.500 AND first_round_unsafe <= 0.500 AND progress_gap_before > -0.250`.
- model `google-gemini-3.5-flash-lite`: unsafe 98.1%, support 18.6%; rule: `progress_gap_before <= 0.250 AND own_private_risk_before <= 0.507 AND own_private_risk_before <= 0.077`.
- model `google-gemini-3.1-flash-lite-preview`: unsafe 95.5%, support 12.0%; rule: `own_private_risk_before <= 0.668 AND own_private_risk_before <= 0.495 AND round > 8.500`.
- family `family_chatgpt`: safe 5.0%, support 30.6%; rule: `first_round_unsafe <= 0.500 AND own_private_risk_before <= 0.019 AND round > 2.500`.
- model `gpt-5.4-nano`: safe 5.2%, support 32.4%; rule: `first_round_unsafe <= 0.500 AND own_private_risk_before <= 0.023 AND round > 1.500`.
- overall `all`: safe 5.2%, support 28.8%; rule: `first_round_unsafe <= 0.500 AND own_private_risk_before <= 0.019 AND round > 2.500`.
- model `gpt-5-nano`: safe 6.2%, support 33.8%; rule: `progress_gap_before <= 0.250 AND own_private_risk_before <= 0.019 AND round > 2.500`.

## Synthesis

The data supports treating strategy as a small set of behavioral regimes rather than only AS/AU/CS/CAS. The strongest additional axes are position-conditioned aggression, anti-copy/alternation, delayed/probe patterns, and escalation stickiness after mutual unsafe states. Gap-based rules are especially useful as mechanisms, while exact deterministic coverage is carried more by anti-copy and sequence motifs.

## Deliverables

- Contact sheet: `C:\Users\admin\Downloads\AI_Race_Experiment\analysis\fh_analytic\outputs\figures\strategy_synthesis_full\fh_strategy_synthesis_full_contact_sheet.png`
- Figures: `01_tree_rules.png`, `02_hmm_state_profiles.png`, `03_embedding_clusters.png`, `04_cluster_profiles.png`, `05_cluster_mix_by_model.png`, `06_model_selection_scans.png`
- Tables: `strategy_synthesis_player_embeddings.csv`, `strategy_synthesis_cluster_profiles.csv`, `strategy_synthesis_hmm_state_profiles.csv`, `strategy_synthesis_hmm_k_scan.csv`, `strategy_synthesis_cluster_k_scan.csv`, `strategy_synthesis_tree_rules.csv`
