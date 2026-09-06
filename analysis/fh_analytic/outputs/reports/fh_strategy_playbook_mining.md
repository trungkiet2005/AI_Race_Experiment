# Strategy Playbook Mining

## New Strategic Levers

Across all completed non-duplicate turns, the mined playbook says:

- Retaliation lift: +4.7%. This is P(unsafe | own previous safe, opponent previous unsafe) minus P(unsafe | both previously safe).
- Opportunistic lift: +41.9%. This is extra unsafe when ahead and opponent was previously safe.
- Catch-up lift: -20.3%. This is extra unsafe when behind and opponent was previously safe.
- Forgiveness rate: 42.1%. This is P(safe | own previous unsafe, opponent previous safe).
- Mutual-unsafe stickiness: 65.1%. This is P(unsafe | both previously unsafe).

## Model Playbook

- Gemini 3 Flash: retaliation lift +69.5%, opportunistic lift +29.5%, catch-up lift +40.3%, forgiveness 76.7%, UU stickiness 90.3%.
- Gemini 3.5 Flash Lite: retaliation lift +30.0%, opportunistic lift n/a, catch-up lift n/a, forgiveness 73.2%, UU stickiness 76.9%.
- Gemini 3.1 Flash Lite: retaliation lift +3.1%, opportunistic lift n/a, catch-up lift n/a, forgiveness 59.2%, UU stickiness 83.2%.
- GPT-5.4 nano: retaliation lift -3.5%, opportunistic lift +49.1%, catch-up lift -41.1%, forgiveness 23.6%, UU stickiness 65.6%.
- GPT-5 nano: retaliation lift -11.5%, opportunistic lift +26.9%, catch-up lift -14.4%, forgiveness 52.6%, UU stickiness 18.2%.

## Sequence Motifs

- `all_safe`: 31.2% of player trajectories (2114).
- `safe_then_attack`: 20.3% of player trajectories (1374).
- `alternating`: 18.9% of player trajectories (1283).
- `mixed_adaptive`: 11.8% of player trajectories (796).
- `all_unsafe`: 10.5% of player trajectories (714).
- `probe_then_cool`: 3.7% of player trajectories (251).
- `late_escalation`: 2.0% of player trajectories (133).
- `cooldown`: 1.6% of player trajectories (109).

## Dyad-Level Styles

- `mixed`: 27.7% of games (937).
- `alternating_collision`: 27.4% of games (929).
- `asymmetric_exploitation`: 21.7% of games (736).
- `mutual_safe`: 17.2% of games (582).
- `mutual_unsafe`: 4.4% of games (149).
- `mutual_escalation`: 1.2% of games (40).
- `mutual_deescalation`: 0.4% of games (14).

## Interpretation

The extra strategic signal is not just AU/AS/CS/CAS mismatch. The data shows three reusable moves: retaliation after being exploited, opportunistic attack when already ahead, and cooldown/forgiveness after unilateral unsafe. Different models combine those moves with different baseline aggression.

## Deliverables

- Contact sheet: `C:\Users\admin\Downloads\AI_Race_Experiment\analysis\fh_analytic\outputs\figures\strategy_playbook\fh_strategy_playbook_storyboard_contact_sheet.png`
- Figures: `01_strategic_levers_by_model.png`, `02_prev_state_gap_response_matrix.png`, `03_action_motif_mix_by_model.png`, `04_dyad_play_styles_by_model.png`, `05_round_switching_profiles.png`
- Tables: `strategy_playbook_levers.csv`, `strategy_playbook_response_matrix.csv`, `strategy_playbook_motifs.csv`, `strategy_playbook_dyads.csv`
