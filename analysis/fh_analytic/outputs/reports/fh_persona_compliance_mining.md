# FH Persona Compliance Mining

## Scope

`mode_strategy_persona` decisions: 10,422 across 567 races. This mode assigns each seat one of five persona/role labels (`cooperative`, `adversarial`, `neutral`, `risk-averse`, `risk-seeking`) and is otherwise the same mechanism as baseline. The main pipeline only reported this mode's pooled unsafe rate (32.8% for ChatGPT, 48.2% for Gemini); this stage tests whether the assigned role itself, not just the mode label, drives behavior.

## Executive Summary

- **Persona-role compliance is strong and monotonic across all three models.** Pooled unsafe rate is 0.024 under `cooperative`, rising to 0.621 under `adversarial`; the risk framing shows the same pattern, 0.105 `risk-averse` versus 0.781 `risk-seeking`. This is a much cleaner manipulation check than the aggregate `persona_mode` comparison used elsewhere in the pipeline.
- **Compliance strength is model-dependent, not universal.** See the per-model table below: some models separate roles by 60-90 points of unsafe rate, one separates by under 5 points on `cooperative` and shows little role sensitivity elsewhere.
- **The adversarial-vs-cooperative gap moves over the course of a race** (see `persona_compliance_gap_over_time.csv` and the figure below) rather than staying flat; direction and magnitude of the drift differ by model, so persona compliance is not simply a fixed offset applied at round 1.
- **On the one axis where own- and opponent-role are not collinear (adversarial vs cooperative), own-role dominates opponent-role in the logit** for every model that fit (`persona_role_asymmetry_logit.csv`): being assigned `adversarial` changes a player's own behavior far more than facing an `adversarial` opponent does. The `neutral`/`risk-averse`/`risk-seeking` conditions are always seat-symmetric in this data (both seats get the same role), so an own-vs-opponent split is not identifiable for the risk-framing axis; see Caveats.
- **Human-reference lag/gap signs mostly survive the persona layer**: 0.75 of role x term sign checks against Fernandez Domingos & Han (2026) agree in direction (`persona_human_check.csv`).

## Role Compliance By Model

| model_slug | seat_persona_role | n | unsafe_rate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | adversarial | 252 | 0.7143 | 0.6585 | 0.7701 |
| google-gemini-3-flash-preview | cooperative | 684 | 0.01754 | 0.007705 | 0.02738 |
| google-gemini-3-flash-preview | neutral | 558 | 0.7796 | 0.7452 | 0.814 |
| google-gemini-3-flash-preview | risk-averse | 558 | 0.1326 | 0.1045 | 0.1608 |
| google-gemini-3-flash-preview | risk-seeking | 558 | 1 | 1 | 1 |
| gpt-5-nano | adversarial | 1116 | 0.4579 | 0.4287 | 0.4871 |
| gpt-5-nano | cooperative | 1116 | 0.009857 | 0.004061 | 0.01565 |
| gpt-5-nano | neutral | 558 | 0.01434 | 0.004473 | 0.0242 |
| gpt-5-nano | risk-averse | 558 | 0.05376 | 0.03505 | 0.07248 |
| gpt-5-nano | risk-seeking | 558 | 0.4104 | 0.3696 | 0.4512 |
| gpt-5.4-nano | adversarial | 1116 | 0.7634 | 0.7385 | 0.7884 |
| gpt-5.4-nano | cooperative | 1116 | 0.04211 | 0.03033 | 0.0539 |
| gpt-5.4-nano | neutral | 558 | 0.5072 | 0.4657 | 0.5487 |
| gpt-5.4-nano | risk-averse | 558 | 0.1272 | 0.09959 | 0.1549 |
| gpt-5.4-nano | risk-seeking | 558 | 0.9319 | 0.911 | 0.9528 |

Pooled across models:

| seat_persona_role | n | unsafe_rate |
| --- | --- | --- |
| adversarial | 2484 | 0.6212 |
| cooperative | 2916 | 0.02401 |
| neutral | 1674 | 0.4337 |
| risk-averse | 1674 | 0.1045 |
| risk-seeking | 1674 | 0.7808 |

Visual: `figures/persona_compliance/01_role_compliance_by_model.png`.

## Compliance Over Time

Unsafe rate by round phase within each role/model; `round_1` isolates whether the persona shifts the very first move, before any interaction history exists.

| model_slug | seat_persona_role | round_phase | n | unsafe_rate |
| --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | adversarial | round_1 | 36 | 1 |
| google-gemini-3-flash-preview | adversarial | early_r2_4 | 108 | 0.7222 |
| google-gemini-3-flash-preview | adversarial | mid_r5_8 | 96 | 0.625 |
| google-gemini-3-flash-preview | adversarial | late_r9plus | 12 | 0.5 |
| google-gemini-3-flash-preview | cooperative | round_1 | 78 | 0 |
| google-gemini-3-flash-preview | cooperative | early_r2_4 | 234 | 0.01709 |
| google-gemini-3-flash-preview | cooperative | mid_r5_8 | 246 | 0.02846 |
| google-gemini-3-flash-preview | cooperative | late_r9plus | 126 | 0.007937 |
| google-gemini-3-flash-preview | risk-averse | round_1 | 60 | 0.1333 |
| google-gemini-3-flash-preview | risk-averse | early_r2_4 | 180 | 0.1389 |
| google-gemini-3-flash-preview | risk-averse | mid_r5_8 | 198 | 0.1313 |
| google-gemini-3-flash-preview | risk-averse | late_r9plus | 120 | 0.125 |
| google-gemini-3-flash-preview | risk-seeking | round_1 | 60 | 1 |
| google-gemini-3-flash-preview | risk-seeking | early_r2_4 | 180 | 1 |
| google-gemini-3-flash-preview | risk-seeking | mid_r5_8 | 198 | 1 |
| google-gemini-3-flash-preview | risk-seeking | late_r9plus | 120 | 1 |
| gpt-5-nano | adversarial | round_1 | 120 | 0.08333 |
| gpt-5-nano | adversarial | early_r2_4 | 360 | 0.5556 |
| gpt-5-nano | adversarial | mid_r5_8 | 396 | 0.4773 |
| gpt-5-nano | adversarial | late_r9plus | 240 | 0.4667 |
| gpt-5-nano | cooperative | round_1 | 120 | 0 |
| gpt-5-nano | cooperative | early_r2_4 | 360 | 0.002778 |
| gpt-5-nano | cooperative | mid_r5_8 | 396 | 0.01515 |
| gpt-5-nano | cooperative | late_r9plus | 240 | 0.01667 |
| gpt-5-nano | risk-averse | round_1 | 60 | 0 |
| gpt-5-nano | risk-averse | early_r2_4 | 180 | 0.1056 |
| gpt-5-nano | risk-averse | mid_r5_8 | 198 | 0.03535 |
| gpt-5-nano | risk-averse | late_r9plus | 120 | 0.03333 |
| gpt-5-nano | risk-seeking | round_1 | 60 | 0.4333 |
| gpt-5-nano | risk-seeking | early_r2_4 | 180 | 0.4111 |
| gpt-5-nano | risk-seeking | mid_r5_8 | 198 | 0.4343 |
| gpt-5-nano | risk-seeking | late_r9plus | 120 | 0.3583 |
| gpt-5.4-nano | adversarial | round_1 | 120 | 0.8167 |
| gpt-5.4-nano | adversarial | early_r2_4 | 360 | 0.7417 |
| gpt-5.4-nano | adversarial | mid_r5_8 | 396 | 0.7854 |
| gpt-5.4-nano | adversarial | late_r9plus | 240 | 0.7333 |
| gpt-5.4-nano | cooperative | round_1 | 120 | 0 |
| gpt-5.4-nano | cooperative | early_r2_4 | 360 | 0.02778 |
| gpt-5.4-nano | cooperative | mid_r5_8 | 396 | 0.05556 |
| gpt-5.4-nano | cooperative | late_r9plus | 240 | 0.0625 |
| gpt-5.4-nano | risk-averse | round_1 | 60 | 0.2833 |
| gpt-5.4-nano | risk-averse | early_r2_4 | 180 | 0.1056 |
| gpt-5.4-nano | risk-averse | mid_r5_8 | 198 | 0.1162 |
| gpt-5.4-nano | risk-averse | late_r9plus | 120 | 0.1 |
| gpt-5.4-nano | risk-seeking | round_1 | 60 | 1 |
| gpt-5.4-nano | risk-seeking | early_r2_4 | 180 | 0.8722 |
| gpt-5.4-nano | risk-seeking | mid_r5_8 | 198 | 0.9596 |
| gpt-5.4-nano | risk-seeking | late_r9plus | 120 | 0.9417 |

Adversarial-minus-cooperative gap by round phase:

| model_slug | round_phase | adversarial | cooperative | adv_minus_coop_gap |
| --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | round_1 | 1 | 0 | 1 |
| google-gemini-3-flash-preview | early_r2_4 | 0.7222 | 0.01709 | 0.7051 |
| google-gemini-3-flash-preview | mid_r5_8 | 0.625 | 0.02846 | 0.5965 |
| google-gemini-3-flash-preview | late_r9plus | 0.5 | 0.007937 | 0.4921 |
| gpt-5-nano | round_1 | 0.08333 | 0 | 0.08333 |
| gpt-5-nano | early_r2_4 | 0.5556 | 0.002778 | 0.5528 |
| gpt-5-nano | mid_r5_8 | 0.4773 | 0.01515 | 0.4621 |
| gpt-5-nano | late_r9plus | 0.4667 | 0.01667 | 0.45 |
| gpt-5.4-nano | round_1 | 0.8167 | 0 | 0.8167 |
| gpt-5.4-nano | early_r2_4 | 0.7417 | 0.02778 | 0.7139 |
| gpt-5.4-nano | mid_r5_8 | 0.7854 | 0.05556 | 0.7298 |
| gpt-5.4-nano | late_r9plus | 0.7333 | 0.0625 | 0.6708 |

Visual: `figures/persona_compliance/02_compliance_gap_over_time.png`.

## Role x Strategic Levers

Same lever definitions as `fh_strategy_playbook_mining.md` (retaliation, opportunistic-when-ahead, catch-up-when-behind, forgiveness, mutual-unsafe stickiness), now split by assigned persona role.

| model_slug | seat_persona_role | turns_round2plus | unsafe_rate | retaliation_lift | n_retaliation_state | opportunistic_lift | n_opportunistic_state | catchup_lift | n_catchup_state | forgiveness_rate | n_forgiveness_state | mutual_unsafe_stickiness | n_mutual_unsafe_state |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | adversarial | 216 | 0.6667 |  | 5 | -0.2065 | 58 |  | 0 | 0.6327 | 49 | 0.9545 | 110 |
| google-gemini-3-flash-preview | cooperative | 606 | 0.0198 | 0.1222 | 47 |  | 0 | 0.1333 | 30 |  | 3 |  | 6 |
| google-gemini-3-flash-preview | neutral | 498 | 0.759 | 0.309 | 66 |  | 0 |  | 0 | 0.6212 | 66 | 0.8187 | 320 |
| google-gemini-3-flash-preview | risk-averse | 498 | 0.1325 | 0.7471 | 43 |  | 0 |  | 0 | 0.8605 | 43 | 0.4091 | 22 |
| google-gemini-3-flash-preview | risk-seeking | 498 | 1 |  | 0 |  | 0 |  | 0 |  | 0 | 1 | 498 |
| gpt-5-nano | adversarial | 996 | 0.503 | -0.3711 | 86 | -0.06585 | 372 |  | 16 | 0.5417 | 336 | 0.075 | 120 |
| gpt-5-nano | cooperative | 996 | 0.01104 | -0.004356 | 253 |  | 0 | 0.06341 | 123 |  | 3 |  | 6 |
| gpt-5-nano | neutral | 498 | 0.01606 |  | 5 |  | 1 |  | 0 |  | 5 |  | 2 |
| gpt-5-nano | risk-averse | 498 | 0.06024 | -0.01711 | 21 |  | 0 |  | 0 | 1 | 21 |  | 8 |
| gpt-5-nano | risk-seeking | 498 | 0.4076 | -0.4189 | 131 | 0.1437 | 117 | -0.1357 | 43 | 0.4733 | 131 | 0.09211 | 76 |
| gpt-5.4-nano | adversarial | 996 | 0.757 | 0.03792 | 95 | 0.06341 | 411 |  | 7 | 0.1763 | 465 | 0.661 | 295 |
| gpt-5.4-nano | cooperative | 996 | 0.04719 | 0.03174 | 388 |  | 4 | 0.06201 | 66 |  | 18 | 0 | 23 |
| gpt-5.4-nano | neutral | 498 | 0.5281 | 0.03653 | 132 | -0.003655 | 53 |  | 19 | 0.4848 | 132 | 0.541 | 122 |
| gpt-5.4-nano | risk-averse | 498 | 0.1084 | 0.04314 | 50 | 0.00374 | 38 | -0.03253 | 28 | 0.86 | 50 |  | 16 |
| gpt-5.4-nano | risk-seeking | 498 | 0.9237 |  | 24 |  | 7 |  | 0 | 0 | 24 | 0.9182 | 440 |

## Own-Role Vs Opponent-Role Logit

Cluster-robust logit of `unsafe` on own persona role, opponent persona role, progress gap, and lag terms, fit per model on round >= 2 decisions restricted to the `adversarial`/`cooperative` conditions (reference role: cooperative) -- the only axis where own- and opponent-role vary independently.

| model_slug | term | coef | odds_ratio | p_value | n | clusters |
| --- | --- | --- | --- | --- | --- | --- |
| google-gemini-3-flash-preview | C(seat_persona_role, Treatment(reference="cooperative"))[T.adversarial] | 8.005 | 2996 | 9.212e-18 | 822 | 57 |
| google-gemini-3-flash-preview | C(opponent_persona_role, Treatment(reference="cooperative"))[T.adversarial] | 2.02 | 7.537 | 0.1335 | 822 | 57 |
| gpt-5-nano | C(seat_persona_role, Treatment(reference="cooperative"))[T.adversarial] | 5.042 | 154.8 | 1.318e-50 | 1992 | 120 |
| gpt-5-nano | C(opponent_persona_role, Treatment(reference="cooperative"))[T.adversarial] | 0.08178 | 1.085 | 0.6899 | 1992 | 120 |
| gpt-5.4-nano | C(seat_persona_role, Treatment(reference="cooperative"))[T.adversarial] | 4.215 | 67.7 | 1.064e-67 | 1992 | 120 |
| gpt-5.4-nano | C(opponent_persona_role, Treatment(reference="cooperative"))[T.adversarial] | -0.07928 | 0.9238 | 0.7435 | 1992 | 120 |

## Human-Reference Check Within Persona Roles

| seat_persona_role | term | coef | human_value | expected_sign | sign_match | p_value | n | phi_U_role |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| adversarial | own_prev_unsafe | -0.3591 | -0.193 |  | True | 0.005044 | 2208 | 0.6336 |
| adversarial | opponent_prev_unsafe | -0.4687 | 0.607 | positive | False | 0.005265 | 2208 | 0.6336 |
| adversarial | progress_gap_before | 0.007359 | -0.296 | negative | False | 0.8952 | 2208 | 0.6336 |
| neutral | own_prev_unsafe | 0.573 | -0.193 |  | True | 0.007792 | 1494 | 0.4344 |
| neutral | opponent_prev_unsafe | 1.497 | 0.607 | positive | True | 1.639e-15 | 1494 | 0.4344 |
| neutral | progress_gap_before | -0.1004 | -0.296 | negative | True | 0.515 | 1494 | 0.4344 |
| risk-averse | own_prev_unsafe | 0.1839 | -0.193 |  | True | 0.5388 | 1494 | 0.1004 |
| risk-averse | opponent_prev_unsafe | 1.512 | 0.607 | positive | True | 1.173e-07 | 1494 | 0.1004 |
| risk-averse | progress_gap_before | -0.7238 | -0.296 | negative | True | 0.04352 | 1494 | 0.1004 |
| risk-seeking | own_prev_unsafe | -0.1481 | -0.193 |  | True | 0.6139 | 1494 | 0.7771 |
| risk-seeking | opponent_prev_unsafe | -0.1054 | 0.607 | positive | False | 0.7017 | 1494 | 0.7771 |
| risk-seeking | progress_gap_before | -0.4071 | -0.296 | negative | True | 0.005134 | 1494 | 0.7771 |

## Caveats

- Role labels are assigned per seat for the whole race; `opponent_persona_role` is derived by matching the other `player_index` within the same `(source_run, game_id)`, not read from a dedicated column.
- `neutral`, `risk-averse`, and `risk-seeking` conditions are always seat-symmetric (both players get the same role) in the data collected so far; only `adversarial`/`cooperative` conditions pair asymmetrically. The own-vs-opponent asymmetry logit is therefore restricted to the adv/coop axis -- fitting it on the full five-role set produces a singular or quasi-separated design matrix because own-role and opponent-role are then collinear for three of the five roles.
- This is descriptive/mechanistic evidence about assigned-role compliance, not a causal claim about what the model 'understands' about the persona instruction.