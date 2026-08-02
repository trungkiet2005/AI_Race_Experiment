# Expanded Strategy Library Fit

## Question

The original paper assumes four deterministic strategies: AS, AU, CS, CAS. This analysis tests whether a larger, interpretable strategy library can explain additional LLM behavior with exact or near-exact trajectory fits.

## Coverage Result

- Player trajectories: 6,774
- 4-strategy exact coverage: 45.5%
- Expanded-library exact coverage: 50.5%
- New exact coverage beyond the 4 strategies: 5.0%
- 4-strategy near coverage, at most one mismatch: 58.9%
- Expanded-library near coverage, at most one mismatch: 72.6%
- New near coverage beyond the 4 strategies: 13.7%

## Model-Level Coverage

- GPT-5 nano: canonical exact 36.2%, expanded exact 42.4%, new exact 6.2%, expanded near 67.6%.
- GPT-5.4 nano: canonical exact 48.6%, expanded exact 53.4%, new exact 4.8%, expanded near 73.3%.
- Gemini 3 Flash: canonical exact 59.1%, expanded exact 62.5%, new exact 3.4%, expanded near 82.2%.
- Gemini 3.5 Flash Lite: canonical exact 18.3%, expanded exact 20.0%, new exact 1.7%, expanded near 55.0%.
- Gemini 3.1 Flash Lite: canonical exact 38.3%, expanded exact 38.3%, new exact 0.0%, expanded near 61.7%.

## Candidate New Exact Strategies

- `ANTI_CS` (pd_memory): +88 exact trajectories beyond canonical; 272 exact total. Meaning: safe first, then do the opposite of opponent previous action.
- `ALT_SAFE_FIRST` (sequence_motif): +80 exact trajectories beyond canonical; 110 exact total. Meaning: alternate safe/unsafe starting safe.
- `PROBE_COOLDOWN_R2` (sequence_motif): +49 exact trajectories beyond canonical; 93 exact total. Meaning: unsafe once, then always safe.
- `ANTI_CAS` (pd_memory): +45 exact trajectories beyond canonical; 245 exact total. Meaning: unsafe first, then do the opposite of opponent previous action.
- `DELAYED_ATTACK_R2` (sequence_motif): +36 exact trajectories beyond canonical; 70 exact total. Meaning: safe once, then always unsafe.
- `ALT_UNSAFE_FIRST` (sequence_motif): +35 exact trajectories beyond canonical; 102 exact total. Meaning: alternate unsafe/safe starting unsafe.
- `WSLS_U` (pd_memory): +27 exact trajectories beyond canonical; 227 exact total. Meaning: Pavlov / win-stay lose-shift, unsafe first.
- `DELAYED_ATTACK_R3` (sequence_motif): +26 exact trajectories beyond canonical; 26 exact total. Meaning: safe twice, then always unsafe.
- `DESPERATE_CATCHUP` (positional_gap): +12 exact trajectories beyond canonical; 1245 exact total. Meaning: unsafe only when behind by more than 0.5.
- `UNEVEN_ATTACK` (positional_gap): +12 exact trajectories beyond canonical; 1245 exact total. Meaning: unsafe when the race is not close.
- `WSLS_S` (pd_memory): +10 exact trajectories beyond canonical; 1185 exact total. Meaning: Pavlov / win-stay lose-shift, safe first.
- `TF2T_S` (pd_memory): +8 exact trajectories beyond canonical; 1353 exact total. Meaning: unsafe only after two consecutive opponent unsafe actions.

## Candidate New Near-Fit Strategies

- `ANTI_CS` (pd_memory): +298 near trajectories beyond canonical; 929 near total. Meaning: safe first, then do the opposite of opponent previous action.
- `ALT_SAFE_FIRST` (sequence_motif): +286 near trajectories beyond canonical; 455 near total. Meaning: alternate safe/unsafe starting safe.
- `DELAYED_ATTACK_R2` (sequence_motif): +165 near trajectories beyond canonical; 982 near total. Meaning: safe once, then always unsafe.
- `ANTI_CAS` (pd_memory): +156 near trajectories beyond canonical; 735 near total. Meaning: unsafe first, then do the opposite of opponent previous action.
- `ALT_UNSAFE_FIRST` (sequence_motif): +145 near trajectories beyond canonical; 321 near total. Meaning: alternate unsafe/safe starting unsafe.
- `WSLS_U` (pd_memory): +113 near trajectories beyond canonical; 500 near total. Meaning: Pavlov / win-stay lose-shift, unsafe first.
- `CATCHUP_ATTACK` (positional_gap): +96 near trajectories beyond canonical; 1651 near total. Meaning: unsafe when behind, safe otherwise.
- `DELAYED_ATTACK_R3` (sequence_motif): +90 near trajectories beyond canonical; 312 near total. Meaning: safe twice, then always unsafe.
- `PROBE_COOLDOWN_R2` (sequence_motif): +86 near trajectories beyond canonical; 2345 near total. Meaning: unsafe once, then always safe.
- `CATCHUP_OR_RETALIATE` (hybrid_gap_memory): +59 near trajectories beyond canonical; 1918 near total. Meaning: unsafe when behind or after opponent unsafe.
- `BULLY_RETALIATOR` (hybrid_gap_memory): +52 near trajectories beyond canonical; 724 near total. Meaning: unsafe until far ahead, also retaliates.
- `WSLS_S` (pd_memory): +51 near trajectories beyond canonical; 1453 near total. Meaning: Pavlov / win-stay lose-shift, safe first.

## Interpretation

Exact coverage is the strict test. Near-fit coverage is the behavioral test: it asks whether a rule captures almost all moves in the observed sequence. If a new rule adds many exact or near trajectories beyond AS/AU/CS/CAS and has a simple interpretation, it is a candidate publishable strategy class.

## Caveat

Gap-based rules use the observed race position at each turn. They are descriptive policy fits, not full counterfactual simulations of what the game state would have been if the strategy had been played from the start.

## Deliverables

- Contact sheet: `C:\Users\admin\Downloads\AI_Race_Experiment\analysis\fh_analytic\outputs\figures\expanded_strategy_library\fh_expanded_strategy_library_contact_sheet.png`
- Figures: `01_coverage_lift.png`, `02_new_exact_strategies.png`, `03_exact_family_by_model.png`, `04_near_family_by_model.png`, `05_coverage_by_length.png`
- Tables: `expanded_strategy_player_fits.csv`, `expanded_strategy_fit_detail.csv`, `expanded_strategy_exact_counts.csv`, `expanded_strategy_near_counts.csv`, `expanded_strategy_family_summary.csv`
