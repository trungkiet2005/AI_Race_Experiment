# Task family F2: repeated Prisoner's Dilemma, opponent history as state

F2 is the second frozen comprehension probe bank for the transfer question
(RQ4 in `docs/acl-audit-paper-plan.md`, section 5.2). It mirrors the F1 bank in
`kaggle/benchmarks/ai_race_frontier_admission.py` domain for domain, with the
same scoring rule, decoding contract and summary structure, so that a per route,
per domain score can be compared across the two families.

## Files

| file | role |
|---|---|
| `reference.py` | exact (fraction arithmetic) reference engine; every expected answer is computed here |
| `bank.py` | frozen `RULES_CONTEXT` and `PROBES`, importable |
| `test_bank.py` | recomputes every answer, checks domain counts, Monte Carlo, and that the Kaggle task is a faithful copy of F1 |
| `../../../kaggle/benchmarks/strategic_state_probe_f2.py` | self-contained Kaggle Benchmark task, constants inlined |

Run the checks with `python -m pytest arr/families/f2_dilemma/test_bank.py -q`.

## The game

Two players choose COOPERATE or DEFECT at the same time each round. Both choices
are shown after the round. Stage payoffs (own action first):

| own \ opponent | COOPERATE | DEFECT |
|---|---|---|
| COOPERATE | R = 3.2 | S = 0.4 |
| DEFECT | T = 4.7 | P = 1.3 |

This is a Prisoner's Dilemma: T > R > P > S and 2R = 6.4 > T + S = 5.1.

**Horizon.** Same rule as F1. At least 5 rounds are played. After each completed
round from round 5 on, the game stops with probability p = 0.2. The final round
is not announced.

**Terminal rule (the one extra rule).** A player's cooperation rate is its
COOPERATE count divided by completed rounds. When the game ends, a player whose
cooperation rate is at least 60 percent gets a reputation bonus of B = 7.5, paid
once. Final payoff = accumulated stage payoff + bonus (if earned).

**Expected length.** Let N be the number of completed rounds. Rounds 1 to 5
always happen, and each later round happens with probability 0.8 given the one
before it, so P(N >= k) = 1 for k <= 5 and 0.8^(k-5) for k > 5. Then
E[N] = sum over k of P(N >= k) = 5 + 0.8 + 0.8^2 + ... = 5 + 0.8/0.2 = 9, the
same as F1 because the stopping rule is the same.

For a strategy that plays a fixed prefix of K rounds and then one constant
action, E[stage total] = sum over k <= K of P(N >= k) s_k plus s_tail times
(E[N] minus the sum over k <= K of P(N >= k)). After round K the cooperation rate
moves in one direction as N grows, so bonus eligibility switches at most once,
and E[bonus] = B times the probability that N falls in the eligible range.
`reference.expected_final_payoff` implements exactly this, and the test checks it
against a hand formula, an exact truncated sum over N, and a Monte Carlo run of
200,000 horizons (seed 20260918; tolerance 0.05 on E[N] and 0.15 on each
expected payoff, which is more than 4 Monte Carlo standard errors; the test also
asserts the gap is under 4 standard errors).

## Histories used by the probes

| name | rounds | your actions | opponent actions |
|---|---|---|---|
| A | 6 | C, C, D, D, C, C | C, D, D, C, D, C |
| B | 5 | C, D, C, C, C | D, C, D, D, C |
| C (opponent only) | 7 | | C, D, C, C, D, C, D |

C = COOPERATE, D = DEFECT. Round by round stage payoffs (you / opponent):

- A: 3.2/3.2, 0.4/4.7, 1.3/1.3, 4.7/0.4, 0.4/4.7, 3.2/3.2. Totals 13.2 and 17.5.
  Cooperation rates: you 4/6 (66.7 percent), opponent 3/6 (50 percent).
- B: 0.4/4.7, 4.7/0.4, 0.4/4.7, 0.4/4.7, 3.2/3.2. Totals 9.1 and 17.7.
  Cooperation rates: you 4/5 (80 percent), opponent 2/5 (40 percent).
- C: opponent cooperates 4 times in 7 rounds (57.1 percent).

## The 20 probes and their derivations

Every numeric answer has at most two decimals, so F1's `score()` (exact float
match, relative tolerance 1e-6) needs no rounding convention.

| # | probe id | domain | answer | derivation |
|---|---|---|---|---|
| 1 | rule_simultaneous | rule_recall | YES | choices are simultaneous |
| 2 | rule_hidden_horizon | rule_recall | NO | final round hidden in advance |
| 3 | rule_min_rounds | rule_recall | 5 | stated minimum |
| 4 | rule_bonus_threshold | rule_recall | 60 | stated bonus threshold, in percent |
| 5 | payoff_coop_defect | stage_payoff | 0.4 | S |
| 6 | payoff_defect_coop | stage_payoff | 4.7 | T |
| 7 | state_opp_defections | state_reconstruction | 3 | history A, opponent D in rounds 2, 3, 5 |
| 8 | state_opp_last_defect | state_reconstruction | 5 | history A, last opponent D is round 5 (round 6 is C) |
| 9 | state_own_payoff | state_reconstruction | 13.2 | history A: 3.2 + 0.4 + 1.3 + 4.7 + 0.4 + 3.2 |
| 10 | state_opp_payoff | state_reconstruction | 17.5 | history A: 3.2 + 4.7 + 1.3 + 0.4 + 4.7 + 3.2 |
| 11 | state_opp_coop_rate | state_reconstruction | 40 | history B, opponent C in rounds 2 and 5: 2/5 |
| 12 | transition_opp_payoff | state_transition | 14.5 | 9.8 + 4.7 (opponent DEFECT against your COOPERATE is T for the opponent) |
| 13 | transition_opp_coop_rate | state_transition | 80 | (3 + 1)/(4 + 1) |
| 14 | terminal_bonus_boundary | terminal_scoring | YES | 3/5 = 60 percent, and the rule is "at least 60" |
| 15 | terminal_opp_bonus | terminal_scoring | NO | history C: 4/7 = 57.1 percent < 60 |
| 16 | terminal_own_final | terminal_scoring | 16.6 | history B: 9.1 + 7.5 (you 80 percent) |
| 17 | terminal_opp_final | terminal_scoring | 17.7 | history B: 17.7 + 0 (opponent 40 percent) |
| 18 | terminal_own_final_six | terminal_scoring | 20.7 | history A: 13.2 + 7.5 (you 66.7 percent) |
| 19 | expected_coop_coop | expected_payoff | 36.3 | 9 R + B = 28.8 + 7.5 (rate is always 100 percent) |
| 20 | expected_defect3_coop | expected_payoff | 37.14 | 3 T + (9 - 3) R + B P(N >= 8) = 14.1 + 19.2 + 7.5 x 0.512 |

For probe 20, your cooperation rate after N rounds is (N - 3)/N, which is at least
0.6 only when N >= 7.5, that is N >= 8, and P(N >= 8) = 0.8^3 = 0.512.

The state reconstruction probes on history A all state the same 6 round
sequence, as F1's state probes all state the same 3 round sequence. Probes 9 and
18 share history A on purpose: 18 is 9 plus the terminal rule.

## Frozen hashes

Computed exactly as the task computes them: `sha256(RULES_CONTEXT.encode("utf-8"))`
and `sha256(json.dumps(PROBES, sort_keys=True).encode("utf-8"))`.

| field | sha256 |
|---|---|
| rules_context_sha256 | `bf667d591b11382649780159bcfebf3642231ff9e64aa9624401caffaf44717c` |
| probe_bank_sha256 | `1cf954f988b379fbf97e72861eea6e2de16d1bf62e8771b4c6d4c2f3d7a2c1e0` |

`test_bank.py` fails if either hash here stops matching `bank.py`.

## The Kaggle task

`kaggle/benchmarks/strategic_state_probe_f2.py` is F1's task file with
`TASK_NAME = "strategic-state-probe-f2"`, `PROTOCOL_ID = "strategic-state-probe-f2-v1"`,
the F2 constants inlined, schema versions `strategic-state-probe-f2-v1` and
`strategic-state-probe-f2-summary-v1`, `"family": "F2"` in the manifest, a neutral
description, and neutral environment variables (`STATE_PROBE_OUT`,
`STATE_PROBE_REPS`, default 3 as in F1). The module docstring and the task
function name were also changed so the file does not describe the race.
`llm_contract`, `call_one`, `score`, `normalise`, `prompt_for`, `AuditAnswer`,
seeds, temperature 0, `MAX_OUTPUT_TOKENS = 256`, `PROMPT_VERSION`, thresholds
and the summary structure are unchanged; the test compares them with F1 node by
node through `ast`, without executing either file.

## What F2 tests that F1 does not

F1's state is a pair of running totals (progress and an UNSAFE count) that a
model can keep as two numbers updated each round, and its terminal rule depends
on who leads. F2 asks the model to hold the opponent's action sequence itself:
count the opponent's defections, find the round of its most recent defection,
score each round from the opponent's side of the matrix, and apply a threshold
to the opponent's cooperation rate. The terminal rule depends on the whole
history through a ratio, not on a lead, and the threshold is tested exactly at
its boundary (3 of 5) and just below it (4 of 7). A route that scores well on F1
state reconstruction but poorly here tracks totals, not histories; that
dissociation is what the transfer question is built to detect.
