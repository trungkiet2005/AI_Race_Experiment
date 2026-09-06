# Strategy Residual Gap Rules

## Scope

This analysis reclassifies each player trajectory against AS, AU, CS, and CAS. A trajectory is treated as residual/noncanonical if it does not exactly match one single canonical strategy across all observed turns.

- Player trajectories analyzed: 6,774
- Exact single-canonical share: 23.7%
- Residual/noncanonical share: 76.3%
- Residual round-2+ turns used for gap formulas: 42,963

## Overall Fit

For the pooled residual behavior, the best pure-gap formula is `step_low_gap`:

`P(U)= 0.33 if gap <= 0.00, else 0.56`

Gap-only interpretation: threshold rule concentrated on above-threshold side. Gap-only predicted unsafe changes from 33.2% at gap=-2 to 55.9% at gap=+2.

If lag/memory is allowed, the best extended formula is `gap_plus_lag`:

`logit P(U) = -1.03 + 0.39*gap + 0.98*opp_prevU + 0.53*own_prevU`

Extended interpretation: leader-pressure rule: unsafe rises when ahead. Predicted unsafe changes from 14.0% at gap=-2 to 43.8% at gap=+2, and improves BIC by 2279.0 versus the best pure-gap formula.

## Noncanonical Coverage by Model

- Gemini 3.1 Flash Lite: 95.0% noncanonical over 60 player trajectories.
- Gemini 3.5 Flash Lite: 88.3% noncanonical over 60 player trajectories.
- GPT-5 nano: 84.5% noncanonical over 2640 player trajectories.
- Gemini 3 Flash: 79.3% noncanonical over 1374 player trajectories.
- GPT-5.4 nano: 65.8% noncanonical over 2640 player trajectories.

## Top Residual Signatures

- `ambiguous_AS|CS`: n=12300 turns, unsafe=7.8%; gap-only `step_low_gap` -> `P(U)= 0.05 if gap <= 0.00, else 0.32` (threshold rule concentrated on above-threshold side); extended `step_low_gap` -> `P(U)= 0.05 if gap <= 0.00, else 0.32`.
- `near_AS`: n=10454 turns, unsafe=26.5%; gap-only `abs_gap` -> `logit P(U) = -0.87 + -0.18*|gap|` (gap magnitude is weak); extended `gap_plus_lag` -> `logit P(U) = -0.32 + 0.03*gap + -1.06*opp_prevU + -1.03*own_prevU`.
- `near_AU`: n=9462 turns, unsafe=72.2%; gap-only `step_low_gap` -> `P(U)= 0.76 if gap <= 1.00, else 0.66` (threshold rule concentrated on behind/low-gap side); extended `gap_plus_lag` -> `logit P(U) = 2.15 + -0.28*gap + -0.96*opp_prevU + -0.75*own_prevU`.
- `ambiguous_AU|CAS`: n=3980 turns, unsafe=82.4%; gap-only `abs_gap` -> `logit P(U) = 2.11 + -1.33*|gap|` (gap magnitude is weak); extended `gap_plus_lag` -> `logit P(U) = 0.07 + 0.38*gap + 2.65*opp_prevU + -0.61*own_prevU`.
- `near_CAS`: n=2645 turns, unsafe=48.1%; gap-only `step_low_gap` -> `P(U)= 0.69 if gap <= 0.00, else 0.34` (threshold rule concentrated on behind/low-gap side); extended `gap_plus_lag` -> `logit P(U) = -0.34 + -0.15*gap + 2.11*opp_prevU + -0.86*own_prevU`.
- `near_CS`: n=2092 turns, unsafe=55.2%; gap-only `step_low_gap` -> `P(U)= 0.73 if gap <= -0.50, else 0.31` (threshold rule concentrated on behind/low-gap side); extended `gap_plus_lag` -> `logit P(U) = -0.54 + -0.30*gap + 1.80*opp_prevU + -0.49*own_prevU`.
- `ambiguous_AU|CS`: n=553 turns, unsafe=71.8%; gap-only `linear_gap` -> `logit P(U) = 0.96 + -0.62*gap` (catch-up rule: unsafe rises when behind); extended `linear_gap` -> `logit P(U) = 0.96 + -0.62*gap`.
- `ambiguous_AS|CAS`: n=489 turns, unsafe=23.5%; gap-only `step_low_gap` -> `P(U)= 0.46 if gap <= -0.50, else 0.17` (threshold rule concentrated on behind/low-gap side); extended `step_low_gap` -> `P(U)= 0.46 if gap <= -0.50, else 0.17`.
- `ambiguous_AS|AU|CS`: n=444 turns, unsafe=55.2%; gap-only `abs_gap` -> `logit P(U) = 0.94 + -0.60*|gap|` (gap magnitude is weak); extended `gap_plus_lag` -> `logit P(U) = 1.26 + -0.60*gap + -1.04*opp_prevU + -0.50*own_prevU`.
- `ambiguous_AS|AU`: n=395 turns, unsafe=53.4%; gap-only `constant` -> `P(U) = 0.53` (mixed residual not explained by gap); extended `gap_plus_lag` -> `logit P(U) = 1.97 + -0.67*gap + -2.53*opp_prevU + -1.15*own_prevU`.

## Strong Model-Specific Rules

- GPT-5 nano / `near_AS`: n=6888, unsafe=28.7%; gap-only `abs_gap` -> `logit P(U) = -0.75 + -0.25*|gap|`; extended `gap_plus_lag` improves BIC by 856.8.
- GPT-5 nano / `near_AU`: n=4064, unsafe=69.1%; gap-only `abs_gap` -> `logit P(U) = 1.33 + -0.42*|gap|`; extended `gap_plus_lag` improves BIC by 647.8.
- Gemini 3 Flash / `near_CAS`: n=1291, unsafe=51.4%; gap-only `linear_gap` -> `logit P(U) = 0.92 + -2.82*gap`; extended `gap_plus_lag` improves BIC by 256.6.
- GPT-5 nano / `ambiguous_AS|AU`: n=315, unsafe=54.3%; gap-only `linear_gap` -> `logit P(U) = 0.34 + -0.32*gap`; extended `gap_plus_lag` improves BIC by 117.4.
- Gemini 3 Flash / `ambiguous_AU|CAS`: n=1634, unsafe=90.9%; gap-only `linear_gap` -> `logit P(U) = 3.12 + -3.72*gap`; extended `gap_plus_lag` improves BIC by 109.1.
- GPT-5.4 nano / `near_AU`: n=4373, unsafe=74.4%; gap-only `step_low_gap` -> `P(U)= 0.77 if gap <= 1.75, else 0.70`; extended `gap_plus_lag` improves BIC by 76.5.
- Gemini 3 Flash / `near_CS`: n=1162, unsafe=57.8%; gap-only `quadratic_gap` -> `logit P(U) = -1.62 + -6.02*gap + -2.11*gap^2`; extended `gap_plus_lag` improves BIC by 46.7.
- GPT-5.4 nano / `near_AS`: n=3234, unsafe=21.6%; gap-only `constant` -> `P(U) = 0.22`; extended `gap_plus_lag` improves BIC by 38.7.
- GPT-5.4 nano / `ambiguous_AU|CAS`: n=1916, unsafe=75.1%; gap-only `abs_gap` -> `logit P(U) = 1.64 + -0.90*|gap|`; extended `gap_plus_lag` improves BIC by 35.8.
- GPT-5 nano / `near_CS`: n=555, unsafe=47.9%; gap-only `step_low_gap` -> `P(U)= 0.71 if gap <= -0.50, else 0.39`; extended `gap_plus_lag` improves BIC by 32.8.
- GPT-5.4 nano / `near_CS`: n=375, unsafe=57.9%; gap-only `step_low_gap` -> `P(U)= 0.67 if gap <= -0.50, else 0.45`; extended `gap_plus_lag` improves BIC by 31.6.
- Gemini 3 Flash / `ambiguous_AU|CS`: n=260, unsafe=78.1%; gap-only `linear_gap` -> `logit P(U) = 0.03 + -1.92*gap`; extended `gap_plus_lag` improves BIC by 28.7.

## Deliverables

- Contact sheet: `C:\Users\admin\Downloads\AI_Race_Experiment\analysis\fh_analytic\outputs\figures\strategy_residual\fh_strategy_residual_storyboard_contact_sheet.png`
- Figures: `05_noncanonical_rate_by_model.png`, `01_noncanonical_strategy_mix.png`, `02_formula_winner_heatmap.png`, `03_best_gap_formula_curves.png`, `04_gap_rule_archetypes.png`
- Tables: `strategy_residual_player_classification.csv`, `strategy_residual_turns.csv`, `strategy_gap_formula_fits.csv`, `strategy_gap_best_formulas.csv`, `strategy_gap_bin_summary.csv`

## Caveats

Gap means `own_progress_before - opponent_progress_before`; positive values mean the player is ahead before choosing the current action. First turns are excluded from formula fitting because their gap is mechanically zero and lag fields are absent.
