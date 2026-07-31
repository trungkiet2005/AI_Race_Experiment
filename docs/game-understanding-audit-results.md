# Game-understanding audit: admitted pilot results

Protocol: `ai-race-game-understanding-v2`

Source revision: `1e96ce20fd5a`

Model: `qwen2.5:7b-instruct-fp16`

Model digest: `59805ce4a4046be2d8f63231a78daacd2e66f5dccf1a64d0d138ebeeb26ff16c`

## Outcome first

The checkpoint reliably recalled the public rules and stage payoff matrix, but it
did not reliably calculate the evolving game. Therefore, a plausible SAFE/UNSAFE
choice is not evidence that the agent represented the complete mechanism.

Across 685 probe outputs, semantic accuracy was 59.1%. Unaided conditions reached
52.1%; the disclosed-calculator condition reached 75.6%. Strict compliance with
the requested `ANSWER: <value>` format was 32.1%, while every response contained a
semantically parseable scalar under the frozen v2 parser.

| Domain | Outputs | Semantic accuracy |
|---|---:|---:|
| Rule recall | 190 | 97.4% |
| Stage payoff | 60 | 100.0% |
| State reconstruction | 135 | 37.0% |
| State transition | 90 | 22.2% |
| Terminal scoring | 150 | 53.3% |
| Expected payoff | 60 | 16.7% |

All five temperature-zero repetitions returned the same raw response in every
item-condition cell. This supports repeatability under one fixed decoding contract;
it does not supply 685 independent observations or establish task validity.

## Prompt robustness

The tested categorical answer-order reversals produced zero correctness flips.
Paraphrasing produced correctness flips for one of nine state-reconstruction items
and two of eight terminal-scoring items. Stable repetition and paraphrase
sensitivity therefore coexist: they answer different reliability questions.

Representative preserved errors include:

- expected payoff 59 reported as 5;
- post-transition progress 5.5 reported as 7;
- winner final payoff 110 reported as 50; and
- reconstructed opponent progress 3.5 reported as 0 or 2.4 depending on wording.

## Calculator-aided behavior

Thirty risk-by-repetition race cells were paired across the canonical and
calculator-card conditions. The hidden horizon matched in all 30 cells. Each
condition produced 30 races and 558 decisions, with zero parse failures.

| Condition | UNSAFE rate | Cluster interval | Mean final payoff | Setback rate |
|---|---:|---:|---:|---:|
| Canonical | 52.0% | 48.0--55.9% | 42.77 | 26.7% |
| Calculator decision card | 60.8% | 51.2--67.9% | 42.21 | 21.7% |

The paired first-round action flip rate was only 3.3%, so most divergence developed
later in the histories. The pilot does not identify a confirmatory causal effect,
but it rules out the simple story that supplying correct local arithmetic must make
sampled play safer or more profitable.

## Audit gates passed

- Python engine, independent reference oracle, and browser fixtures passed.
- All joint action histories through length four matched the reference calculator.
- Non-finite mechanism values and impossible terminal states fail closed.
- Probe and behavior manifests share one source hash and one exact model digest.
- Every logged probe prompt was exactly re-rendered and every raw response rescored.
- Every calculator-card fragment was recomputed from its logged pre-turn state.
- Canonical and aided races had complete turns, terminal records, and paired horizons.
- Manual inspection covered correct and wrong outputs across every available
  domain-condition cell; cells without errors were inspected as all-correct cells.

## Provenance and claim boundary

The full raw bundle remains on the authorized persistent GPU volume and is
intentionally excluded from Git. Its exact handoff path is retained in the
private compute log.

The admitted local tables and hashes are in
`results/open_source/game_understanding_pilot/`. Raw prompts, responses, action
attempts, turns, races, manifests, and logs remain on the persistent shared volume.

These results support performance claims only for the tested checkpoint, English
prompt pool, mechanism, decoding settings, source revision, and hardware contract.
They do not establish subjective understanding, intent, a stable risk preference,
an internal world model, or generalization to other checkpoints or settings.
