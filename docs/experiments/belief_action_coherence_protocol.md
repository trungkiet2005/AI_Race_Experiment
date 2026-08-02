# Opponent-belief and action-coherence protocol

## Status

**Prepared design; runner not yet admitted.** This experiment follows after the
mapping, payoff-scale, and state-scaffold grids because it changes the opponent
population and response contract. It must never be pooled with self-play.

## Question

Can the agent predict a deterministic reference opponent's next action, and is
its own action coherent with that stated belief and the disclosed payoff table?
This separates three capabilities that a single observed move conflates:
belief formation, payoff use, and action selection.

## Design

- opponents: canonical AS, AU, CS, and CAS policies;
- elicitation: action-only versus pre-action `BELIEF_OPPONENT_Q: 0..100`
  followed by one strict P/Q action;
- both opaque mappings, three private-risk levels, and 16 common-random-number
  repetitions;
- `4 × 2 × 2 × 3 × 16 = 768` races;
- the scripted opponent never receives the model's private text and acts from
  the same pre-round state, preserving simultaneous choice;
- the model is not told the strategy label.

## Endpoints

1. Belief Brier score and calibration against the opponent's deterministic next
   action.
2. Belief-update accuracy after informative opponent actions.
3. One-stage belief/action regret using the model's stated probability and the
   public payoff table, clearly labelled myopic rather than full-game regret.
4. Realized payoff, Unsafe rate, and exploitability by opponent strategy.
5. The causal effect of belief elicitation on action, estimated within the paired
   seed/mapping/opponent block.

Prediction accuracy does not prove strategic understanding. Belief/action
coherence is a stronger construct check, but the full repeated game still
contains horizon and terminal-risk trade-offs not captured by myopic regret.

## Admission gates

- strict parse coverage ≥ 99% for both response contracts;
- no action leakage from strategy names or hidden opponent state;
- exact scripted-opponent replay and simultaneous prompt construction;
- both mappings within every block;
- comprehension domains reported before payoff or rationality language;
- uncertainty clustered at the independently seeded race.
