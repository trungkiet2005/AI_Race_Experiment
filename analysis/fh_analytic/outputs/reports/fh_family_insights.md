# FH Family Insights

## Executive Summary

- **Gemini is much more unsafe in baseline, but risk-aware/persona protocols pull it down.** Baseline unsafe is 77.0% for Gemini versus 33.5% for ChatGPT (+43.5 pp gap). Under risk-matrix/persona runs Gemini drops to 47.0% / 48.2%, while ChatGPT stays near 35.5% / 32.8%.
- **ChatGPT behavior is lower on average but split sharply by model.** Baseline ChatGPT averages 33.5%, but the baseline model table separates `gpt-5-nano` from `gpt-5.4-nano`; family-level averages hide that model identity effect.
- **Gemini baseline starts saturated.** First-round baseline unsafe is 100.0% for Gemini versus 21.7% for ChatGPT, so Gemini's baseline logit coefficients are less stable and should be treated as descriptive/mechanistic clues, not coefficient-level proof.
- **Predictive confidence is strongest for state/history, not family as a causal claim.** The full completed tree roots on `own_private_risk_before` in 200/200 bootstraps, while the baseline tree mostly roots on `model_slug_gpt-5-nano`.

## Coverage And Data Quality

Gemini has incomplete coverage and a higher retry rate, so any family comparison should keep coverage visible. Parse failures are zero in completed clean rows.

| family | runs | completed_runs | incomplete_runs | completion |
| --- | --- | --- | --- | --- |
| ChatGPT family | 88 | 88 | 0 | 100.0% |
| Gemini family | 30 | 27 | 3 | 90.0% |

| family | decisions | unsafe | retry | parse_fail |
| --- | --- | --- | --- | --- |
| ChatGPT family | 49,104 | 35.0% | 0.3% | 0.0% |
| Gemini family | 13,188 | 51.1% | 4.7% | 0.0% |

## Family-Level Behavioral Readout

**Baseline is where the family separation is largest.** Gemini baseline is high-unsafe and starts unsafe immediately; ChatGPT is materially lower and more condition-sensitive through model identity than family label alone.

| family | decisions | unsafe | retry | parse_fail |
| --- | --- | --- | --- | --- |
| ChatGPT family | 1,116 | 33.5% | 0.0% | 0.0% |
| Gemini family | 1,674 | 77.0% | 1.4% | 0.0% |

| family | decisions | first_round_unsafe | retry |
| --- | --- | --- | --- |
| ChatGPT family | 120 | 21.7% | 0.0% |
| Gemini family | 180 | 100.0% | 2.8% |

Baseline model split:

| family | model_slug | decisions | unsafe | retry |
| --- | --- | --- | --- | --- |
| ChatGPT family | gpt-5-nano | 558 | 12.4% | 0.0% |
| ChatGPT family | gpt-5.4-nano | 558 | 54.7% | 0.0% |
| Gemini family | google-gemini-3-flash-preview | 558 | 76.3% | 4.3% |
| Gemini family | google-gemini-3.1-flash-lite-preview | 558 | 83.5% | 0.0% |
| Gemini family | google-gemini-3.5-flash-lite | 558 | 71.1% | 0.0% |

**Persona/risk-aware protocols change Gemini much more than ChatGPT.** The same protocol shift barely moves ChatGPT's aggregate unsafe rate, but it cuts Gemini from the saturated baseline into the high-40% range. That suggests Gemini is more steerable by explicit risk/persona framing in this snapshot.

| family | persona_mode | experiment_mode | decisions | unsafe | retry |
| --- | --- | --- | --- | --- | --- |
| ChatGPT family | persona_none | mode_baseline | 1,116 | 33.5% | 0.0% |
| ChatGPT family | persona_risk_aware | mode_risk_matrix | 40,176 | 35.5% | 0.3% |
| ChatGPT family | persona_risk_aware | mode_strategy_persona | 7,812 | 32.8% | 0.3% |
| Gemini family | persona_none | mode_baseline | 1,674 | 77.0% | 1.4% |
| Gemini family | persona_risk_aware | mode_risk_matrix | 8,904 | 47.0% | 5.2% |
| Gemini family | persona_risk_aware | mode_strategy_persona | 2,610 | 48.2% | 5.3% |

## Risk And Lag Patterns

**Gemini's baseline risk curve is counterintuitive: unsafe falls as private risk rises.** The drop from low to high risk is descriptive evidence that baseline Gemini is not simply trading off private risk in the expected monotonic way; it begins almost always unsafe at low risk, then moderates as risk increases.

| family | max_private_risk | decisions | unsafe |
| --- | --- | --- | --- |
| ChatGPT family | 0.1 | 372 | 35.2% |
| ChatGPT family | 0.6 | 372 | 30.6% |
| ChatGPT family | 0.9 | 372 | 34.7% |
| Gemini family | 0.1 | 558 | 94.1% |
| Gemini family | 0.6 | 558 | 74.2% |
| Gemini family | 0.9 | 558 | 62.7% |

**Lag response differs by family.** ChatGPT baseline follows a more human-like conditional pattern: unsafe is 24.1% after both were safe, then rises to 50.3% after own previous unsafe and 46.2% after opponent previous unsafe. Gemini is asymmetric: 93.0% after opponent-only unsafe but only 30.6% after own-only unsafe.

| family | lag_profile | decisions | unsafe |
| --- | --- | --- | --- |
| ChatGPT family | 0/0 | 498 | 24.1% |
| ChatGPT family | 0/1 | 169 | 46.2% |
| ChatGPT family | 1/0 | 169 | 50.3% |
| ChatGPT family | 1/1 | 160 | 40.6% |
| Gemini family | 0/0 | 122 | 69.7% |
| Gemini family | 0/1 | 229 | 93.0% |
| Gemini family | 1/0 | 229 | 30.6% |
| Gemini family | 1/1 | 914 | 81.1% |

## Human-Reference Checks

**OpenAI/ChatGPT has interpretable baseline logit evidence; Gemini has separation/saturation.** ChatGPT provider coefficients are directionally human-like with significant effects for own previous unsafe, opponent previous unsafe, progress gap, and first-round unsafe. Gemini coefficients have missing CIs/p-values in the provider slice, consistent with saturated first-round and near-separated cells.

| family | term | coef | odds_ratio | ci95_low | ci95_high | p_value | n | error |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Gemini family | own_prev_unsafe | -0.8284 | 0.4368 |  |  |  | 1494 |  |
| Gemini family | opponent_prev_unsafe | 0.5701 | 1.769 |  |  |  | 1494 |  |
| Gemini family | progress_gap_before | -19.01 | 5.562e-09 |  |  |  | 1494 |  |
| Gemini family | first_round_unsafe | 1.806 | 6.084 |  |  |  | 1494 |  |
| ChatGPT family | own_prev_unsafe | 0.9143 | 2.495 | 0.3203 | 1.508 | 0.002553 | 996 |  |
| ChatGPT family | opponent_prev_unsafe | 0.6977 | 2.009 | 0.1016 | 1.294 | 0.0218 | 996 |  |
| ChatGPT family | progress_gap_before | -0.9934 | 0.3703 | -1.532 | -0.4549 | 0.0003001 | 996 |  |
| ChatGPT family | first_round_unsafe | 0.7181 | 2.051 | 0.3049 | 1.131 | 0.0006577 | 996 |  |

## Predictive Model Implications

**Model identity explains baseline behavior; accumulated risk/history explains the full completed behavior.** The baseline tree reaches ROC-AUC around 0.84 and mostly splits on `model_slug_gpt-5-nano`, so family-level summaries should be read with model identity in view. In the full completed data, `own_private_risk_before` is the root feature in every bootstrap, which is predictive confidence about behavioral state, not causal evidence that the state independently causes unsafe choices.

| scope | decisions | unsafe_rate | cv_balanced_accuracy_mean | cv_roc_auc_mean | top_root_feature | top_root_share |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_completed | 2490 | 0.5851 | 0.7645 | 0.84 | model_slug_gpt-5-nano | 0.93 |
| all_completed | 55518 | 0.3891 | 0.7766 | 0.8446 | own_private_risk_before | 1 |

## Recommended Next Steps

1. Use `family` as a reporting cut, but use `model_slug` as the primary explanatory cut for baseline because the tree and model table show it carries the separation.
2. Treat Gemini baseline as a saturation case: report rates and lag tables first, and avoid overclaiming unstable segment coefficients.
3. For robustness, rerun family-specific trees/logits after excluding first round, then compare whether ChatGPT's human-like lag/gap pattern and Gemini's opponent-only asymmetry persist.
4. Keep incomplete Gemini cells visible in every family chart; exclude them from headline rates unless the run completes.

## Caveats And Assumptions

- Main behavioral rates use completed, non-duplicate rows from `turns_canonical.csv`.
- Baseline comparisons use `analysis_scope == baseline_completed`; persona/risk-aware comparisons are descriptive because coverage and prompt framing differ.
- Coefficients and predictive models answer different questions: logit terms are mechanistic checks; trees identify predictive splits/rules.
- Gemini family has 3 incomplete runs and materially higher retry rates, so coverage/data-quality context should travel with family claims.