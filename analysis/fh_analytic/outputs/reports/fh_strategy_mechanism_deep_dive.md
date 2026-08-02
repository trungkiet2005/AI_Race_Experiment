# Strategy Mechanism Deep Dive

## Scope

This analysis extends the strategy synthesis with three mechanism tests: time to first unsafe action, dyad escalation cascades, and payoff/fitness by discovered strategy macro.

## First Unsafe Timing

- tied / opponent_prev_unsafe=1: first-unsafe hazard 19.6%, n=1897.
- tied / opponent_prev_unsafe=0: first-unsafe hazard 19.2%, n=20371.
- behind / opponent_prev_unsafe=0: first-unsafe hazard 8.3%, n=1791.
- behind / opponent_prev_unsafe=1: first-unsafe hazard 4.8%, n=4586.

Top discrete hazard model coefficients:

- `model_slug_google-gemini-3.1-flash-lite-preview`: coef +9.02, odds ratio 8298.42.
- `model_slug_google-gemini-3.5-flash-lite`: coef +9.02, odds ratio 8298.42.
- `model_slug_gpt-5.4-nano`: coef -0.17, odds ratio 0.85.
- `model_slug_gpt-5-nano`: coef -0.55, odds ratio 0.58.
- `round`: coef -0.56, odds ratio 0.57.
- `gap`: coef -0.63, odds ratio 0.53.
- `opponent_prev_unsafe`: coef -0.75, odds ratio 0.47.

## Escalation Cascades

- Gemini 3 Flash: SS->one-U 11.8%, one-U->UU 17.7%, UU->UU 84.1%, cascade path 13.1%.
- Gemini 3.1 Flash Lite: SS->one-U 6.2%, one-U->UU 40.8%, UU->UU 75.5%, cascade path 23.3%.
- Gemini 3.5 Flash Lite: SS->one-U 44.4%, one-U->UU 22.3%, UU->UU 62.2%, cascade path 23.3%.
- GPT-5 nano: SS->one-U 32.5%, one-U->UU 8.3%, UU->UU 3.5%, cascade path 29.5%.
- GPT-5.4 nano: SS->one-U 32.2%, one-U->UU 13.1%, UU->UU 44.4%, cascade path 12.9%.

## Fitness

- Catch-up attack: n=781, unsafe 53.2%, win 93.2%, payoff advantage +6.97.
- Stable unsafe: n=1333, unsafe 91.2%, win 64.9%, payoff advantage +6.28.
- Probe/cooldown: n=1798, unsafe 47.2%, win 37.9%, payoff advantage +0.23.
- Stable safe: n=2618, unsafe 3.2%, win 1.7%, payoff advantage -4.95.
- Mixed adaptive: n=244, unsafe 32.3%, win 0.4%, payoff advantage -5.25.

## Interpretation

The strongest mechanism is not merely a fixed unsafe preference. Unsafe often enters through timing and dyad transitions: some strategies delay the first unsafe action, unilateral unsafe can either cool back to mutual safety or tip into mutual unsafe, and the payoff frontier separates stable-safe, probe/cooldown, catch-up, and unsafe-sticky regimes.

## Deliverables

- Contact sheet: `C:\Users\admin\Downloads\AI_Race_Experiment\analysis\fh_analytic\outputs\figures\strategy_mechanism_deep_dive\fh_strategy_mechanism_deep_dive_contact_sheet.png`
- Figures: `01_first_unsafe_survival.png`, `02_first_unsafe_hazard_by_state.png`, `03_cascade_transition_matrix.png`, `04_cascade_by_model.png`, `05_strategy_fitness_frontier.png`, `06_payoff_advantage_by_strategy.png`
- Tables: `strategy_deep_player_outcomes.csv`, `strategy_deep_first_unsafe_hazard_rows.csv`, `strategy_deep_first_unsafe_km.csv`, `strategy_deep_hazard_summary.csv`, `strategy_deep_hazard_coefficients.csv`, `strategy_deep_cascade_matrix.csv`, `strategy_deep_cascade_metrics.csv`, `strategy_deep_fitness_macro.csv`, `strategy_deep_fitness_strategy.csv`
