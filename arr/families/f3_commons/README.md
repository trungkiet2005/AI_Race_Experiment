# F3: repeated four-player public goods game

F3 is the third task family for the transfer question (RQ4 in
`docs/acl-audit-paper-plan.md`, section 5.2). It uses the same six probe domains,
the same domain counts, the same scorer, seeds, decoding contract and thresholds
as the F1 admission bank, so a per-route, per-domain score can be compared across
families. What changes is the state the model must track.

| file | role |
|---|---|
| `reference.py` | exact reference engine (fractions); renders the rules text and every probe, and computes every answer |
| `bank.py` | the frozen `RULES_CONTEXT` and `PROBES`, plus their hashes |
| `test_bank.py` | recomputes every answer, checks the counts, checks the Kaggle copy against `bank.py` through the AST, checks the shared contract against F1 |
| `../../../kaggle/benchmarks/strategic_state_probe_f3.py` | self-contained Kaggle Benchmark task |

Run the checks from the repository root:

```
python -m pytest arr/families/f3_commons/test_bank.py -q
```

## The game

Four players. Each round every player receives 10 tokens and all four choose at
the same time how much to put into a common pool: 0, 5 or 10. The pool is
multiplied by 1.6 and split equally, so each token contributed returns 0.4 to
every player, including the one who gave it. A player's stage payoff in a round is

    10 - own contribution + 0.4 x (group total contribution in that round)

The horizon is the F1 horizon: at least 5 completed rounds, then after each
completed round from round 5 on the game stops with probability 20 percent. The
final round is not announced.

**Terminal rule (threshold bonus).** When the game ends, if the four players'
contributions summed over all rounds are at least 150 tokens, every player
receives a bonus of 40, whatever that player contributed. Otherwise nobody
receives it. Final payoff = accumulated stage payoff + bonus (if any).

## Task identity and hashes

| field | value |
|---|---|
| task name | `strategic-state-probe-f3` |
| protocol id | `strategic-state-probe-f3-v1` |
| manifest `family` | `F3` |
| `rules_context_sha256` | `5aaea179cb23e62cda29ad0ee9dd92cfc0b6c05074a7b657073f35cc1a14b94f` |
| `probe_bank_sha256` | `31ff7c47875bd700f126e42392d6769c1ab4a75d9d4c09fccb71c83033410b5b` |

Both hashes are computed the way the task computes them:
`sha256(RULES_CONTEXT.encode("utf-8"))` and
`sha256(json.dumps(PROBES, sort_keys=True).encode("utf-8"))`. They are stored in
`bank.py` and asserted by the tests against the constants inlined in the task
file.

The task file differs from F1 only in: the module docstring, `TASK_NAME`,
`PROTOCOL_ID`, the two environment variable names (`STATE_PROBE_F3_REPS`,
`STATE_PROBE_F3_OUT`, default output `results/strategic_state_probe_f3`), the
rules text and probes, the task description, the task function name, the two
`schema_version` strings, and the added manifest key `"family": "F3"`. The
tests assert that everything else (scorer, prompt, decoding contract, seeds,
thresholds, manifest and summary structure, task body) is AST-identical to F1.
`PROMPT_VERSION` is kept at F1's value because the prompt template is unchanged.

## Horizon results used by the expected-payoff probes

Let L be the number of completed rounds. L = 5 + K, where K counts the rounds in
which the game continued, P(K = k) = 0.8^k x 0.2. So

- E[L] = 5 + 0.8 / 0.2 = 9;
- P(L >= n) = 1 for n <= 5, and 0.8^(n - 5) for n > 5.

For a profile played every round with stage payoff s and group total G per round,
the bonus is paid exactly when G x L >= 150, that is when L >= ceil(150 / G). So

    E[final payoff] = s x 9 + 40 x P(L >= ceil(150 / G))

This is exact. The tests check it two further ways: against a sum over horizon
lengths 1 to 400 using the round-by-round engine (agreement within 1e-9), and
against a Monte Carlo simulation of the stopping rule with 200,000 draws (fixed
seed 20260918), requiring |mean - exact| <= 4 standard errors and a standard
error below 0.25. Observed: 183.964 (SE 0.160) against 184, and 137.455 (SE
0.164) against 137.48.

## The 20 probes and their answers

The shared history used by the five state_reconstruction probes (you, P2, P3, P4):
round 1 (10, 10, 5, 0), round 2 (5, 10, 5, 0), round 3 (0, 5, 5, 0),
round 4 (5, 10, 0, 0). Group totals per round: 25, 20, 10, 15; sum 70.

| # | probe id | domain | answer | derivation |
|---|---|---|---|---|
| 1 | `rule_simultaneous` | rule_recall | YES | rules: all four choose simultaneously |
| 2 | `rule_hidden_horizon` | rule_recall | NO | rules: the final round is hidden in advance |
| 3 | `rule_multiplier` | rule_recall | 1.6 | rules |
| 4 | `rule_group_target` | rule_recall | 150 | rules: bonus if the summed contributions reach at least 150 |
| 5 | `payoff_free_ride` | stage_payoff | 22 | 10 - 0 + 0.4 x 30 |
| 6 | `payoff_sole_contributor` | stage_payoff | 4 | 10 - 10 + 0.4 x 10 |
| 7 | `state_own_payoff` | state_reconstruction | 48 | (0 + 10) + (5 + 8) + (10 + 4) + (5 + 6) |
| 8 | `state_other_payoff` | state_reconstruction | 68 | P4 keeps 10 each round: 4 x 10 + 0.4 x 70 |
| 9 | `state_group_average` | state_reconstruction | 17.5 | 70 / 4 |
| 10 | `state_rounds_above` | state_reconstruction | 3 | round totals above 12: 25, 20, 15 |
| 11 | `state_target_gap` | state_reconstruction | 80 | 150 - 70 |
| 12 | `transition_target_gap` | state_transition | 55 | round 5 (10, 10, 5, 0) adds 25: 80 - 25 |
| 13 | `transition_own_payoff` | state_transition | 58 | 48 + (10 - 10 + 0.4 x 25) |
| 14 | `terminal_target_missed` | terminal_scoring | NO | 25 + 30 + 20 + 30 + 20 + 20 = 145 < 150 |
| 15 | `terminal_free_rider_bonus` | terminal_scoring | YES | the bonus goes to every player whatever it contributed |
| 16 | `terminal_payoff_bonus` | terminal_scoring | 164 | 160 >= 150: 124 + 40 |
| 17 | `terminal_payoff_no_bonus` | terminal_scoring | 124 | 135 < 150: 124 + 0 |
| 18 | `terminal_payoff_from_rounds` | terminal_scoring | 112 | 30 + 35 + 25 + 30 + 35 = 155 >= 150: 72 + 40 |
| 19 | `expected_all_ten` | expected_payoff | 184 | s = 16, G = 40, target reached by round 4 so the bonus is certain: 16 x 9 + 40 |
| 20 | `expected_all_five` | expected_payoff | 137.48 | s = 13, G = 20, needs L >= 8, P = 0.8^3 = 0.512: 13 x 9 + 40 x 0.512 |

Domain counts: rule_recall 4 (two YES/NO, two numeric), stage_payoff 2,
state_reconstruction 5, state_transition 2, terminal_scoring 5 (two YES/NO,
three numeric), expected_payoff 2. These match F1 exactly. Every numeric answer
has at most two decimals, and the tests check that under the shared scorer the
answer scores correct and a value 0.01 or 1 away scores wrong.

The stated terminal figures are reachable game states, which the tests check:
a stage payoff of 124 arises after 8 rounds with own contributions 20 and group
total 160, or own 10 and group total 135; a stage payoff of 72 arises after 5
rounds with own contributions 40 and group total 155.

## What F3 tests that F1 and an opponent-history game do not

In F1 the relevant state is a handful of counters, own progress, rival progress
and own UNSAFE count, each updated from one player's own action. In a game whose
state is one opponent's history, the model has to hold a single sequence and
read it back. F3 needs neither kind of memory alone. Every quantity it asks about
is a sum over a players-by-rounds table: one's own payoff depends on what the
three others put into the pool, a free-riding player's payoff depends on the
whole group, and the terminal outcome depends on a group total that no single
player controls. The model must aggregate across both axes of the table,
compare the result with a target, and in the expected-payoff probes turn the
round at which the target is crossed into a probability under the hidden
horizon. A route that tracks its own counters or one rival's moves well can
still fail here, which is what makes F3 useful for asking whether a
state-tracking score is a construct or a property of one state structure. As in
F1, the history is given in the prompt, so F3 measures reconstruction from a
stated table, not memory across a live multi-turn game.
