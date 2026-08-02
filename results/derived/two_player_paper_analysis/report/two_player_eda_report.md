# Two-player AI Race: exploratory data analysis and paper figure bank

Generated: 2026-08-02T00:33:00.041054+00:00

## Executive boundary

- Confirmatory gameplay runs available: **0**.
- Completed parse-clean directory runs admitted for exploratory analysis: **113**.
- Every behavioral result in this report is pilot or diagnostic evidence. Model comparisons are exploratory because backend, decoding, and seed-forwarding contracts differ across evidence lanes.
- The headline balanced baseline uses five frontier models, 10 repetition blocks × 3 risks × 2 players per model. Claude (3 blocks/risk) and local Qwen are shown only in separate-protocol robustness figures.

## Primary estimand and uncertainty

The primary estimand is the mean player-level trajectory Unsafe fraction. Each player–race contributes equally, so long realized horizons do not receive extra weight. Percentile 95% intervals resample whole source-run × repetition blocks, preserving both seats and the three common-random-number risk treatments. The primary within-model contrast is 90% minus 10% maximum private risk.

## Baseline findings

| Model | Unsafe @10% | Unsafe @60% | Unsafe @90% | 90%−10% (95% CI) | blocks |
|---|---:|---:|---:|---:|---:|
| Gemini 3 Flash | 100.0% | 72.3% | 53.9% | -46.1% [-51.4%, -40.5%] | 10 |
| Gemini 3.1 Flash-Lite | 100.0% | 80.1% | 69.9% | -30.1% [-33.1%, -27.0%] | 10 |
| Gemini 3.5 Flash-Lite | 83.8% | 70.8% | 62.6% | -21.2% [-28.8%, -13.7%] | 10 |
| GPT-5 nano | 12.3% | 15.1% | 14.5% | +2.2% [-5.2%, +10.8%] | 10 |
| GPT-5.4 nano | 57.7% | 49.6% | 56.7% | -1.0% [-6.5%, +3.8%] | 10 |

The robust descriptive pattern is heterogeneity, not one universal LLM response: the Gemini pilots reduce Unsafe play strongly as risk increases; GPT-5 nano remains low and nearly flat; GPT-5.4 nano is non-monotone. Round 1 must be separated from later play because several Gemini cells initialize at 100% Unsafe while GPT-5 nano initializes at 0%.

## Persona and prompt sensitivity

- Gemini 3 Flash: largest available risk-averaged persona shift is Coop–coop, -75.4% versus the no-persona baseline (95% CI -77.7% to -72.9%).
- GPT-5 nano: largest available risk-averaged persona shift is Risk-seeking, +27.4% versus the no-persona baseline (95% CI +23.7% to +31.0%).
- GPT-5.4 nano: largest available risk-averaged persona shift is Coop–coop, -52.5% versus the no-persona baseline (95% CI -56.8% to -48.4%).
- Qwen2.5 7B: largest available risk-averaged persona shift is Risk-averse, -24.3% versus the no-persona baseline (95% CI -30.4% to -19.2%).
- The Qwen prompt-surface pilot spans 8.4%–89.2% Unsafe across 18 prompt variants despite unchanged game semantics. This is a major robustness result, not a nuisance detail.
- The GPT risk-persona matrix is complete (36/36 cells for both GPT models). Gemini has only 14 clean cells; duplicate/partial cells are masked and must not be imputed.
- Context-skin behavior remains diagnostic only because the frozen comprehension admission gate failed; its figure intentionally retains that warning.

## Recommended paper shortlist

1. **Main:** Fig. 01 baseline risk response + paired high-vs-low forest.
2. **Main:** Fig. 02 initialization and round dynamics, because it prevents an all-round average from hiding model-specific first actions.
3. **Main or Results appendix:** Fig. 05 safety–payoff plane and payoff decomposition.
4. **Persona result:** Fig. 06 persona effects or Fig. 07 role asymmetry; use Fig. 08 for the complete factorial surface.
5. **Robustness:** Fig. 10 prompt-surface sensitivity. It is unusually strong and should be discussed even if placed in the supplement.
6. **Supplement/QC:** Figs. 01b, 03, 04, 09, 11–13.

## Figure catalog

- `fig01_baseline_risk_response`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig01_baseline_risk_response_source.csv` when applicable.
- `fig01b_protocol_robustness_baselines`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig01b_protocol_robustness_baselines_source.csv` when applicable.
- `fig02_initialization_and_dynamics`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig02_initialization_and_dynamics_source.csv` when applicable.
- `fig03_conditional_dynamics`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig03_conditional_dynamics_source.csv` when applicable.
- `fig04_strategy_composition`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig04_strategy_composition_source.csv` when applicable.
- `fig05_safety_payoff_frontier`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig05_safety_payoff_frontier_source.csv` when applicable.
- `fig06_persona_effects`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig06_persona_effects_source.csv` when applicable.
- `fig07_persona_role_asymmetry`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig07_persona_role_asymmetry_source.csv` when applicable.
- `fig08_gpt_risk_persona_surfaces`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig08_gpt_risk_persona_surfaces_source.csv` when applicable.
- `fig09_gemini_risk_persona_partial`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig09_gemini_risk_persona_partial_source.csv` when applicable.
- `fig10_surface_sensitivity`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig10_surface_sensitivity_source.csv` when applicable.
- `fig11_context_temperature_diagnostic`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig11_context_temperature_diagnostic_source.csv` when applicable.
- `fig12_evidence_inventory`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig12_evidence_inventory_source.csv` when applicable.
- `fig13_repeat_run_stability`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/fig13_repeat_run_stability_source.csv` when applicable.

## Interpretation limits

- Pilot CIs quantify repetition-block variability in this experiment snapshot; they do not represent a population of prompts or model versions.
- Lag, position, winner, and payoff associations are post-action/endogenous summaries and are not causal mechanism estimates.
- Unseeded API outputs mean repeated calls are independent stochastic attempts, not exact replications. Overlapping game identifiers are never pooled without `source_run`.
- Failed, running, protocol-failed, duplicate-key, partial, smoke, and superseded artifacts remain visible in `dataset_inventory.csv` but are excluded from headline behavior.

## Reproduction

```bash
python results/scripts/analyze_two_player_paper_figures.py
```
