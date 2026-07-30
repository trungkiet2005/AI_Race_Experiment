# AI Race LLM research protocol

## Scope

The project asks whether LLM agents reproduce the dynamic safety pressures observed
in the human AI-race experiment of Fernández Domingos and Han (2026). It reuses the
existing config-driven, lockstep LLM infrastructure, but the domain model is a
two-player technological race rather than a collective-risk public-goods game.

The first study is a model-behavior experiment. It does not elicit human risk
preferences, demographics, or beliefs, so those covariates from the source paper are
not applicable.

## Research questions

1. Does maximum private risk (`0.1`, `0.6`, `0.9`) change the probability of an
   Unsafe action?
2. Is Unsafe play more likely after the opponent played Unsafe in the previous
   round?
3. Does falling behind increase Unsafe play, and does being ahead reduce it?
4. Does a first-round Unsafe choice predict later Unsafe behavior?
5. Which canonical reduced strategy—AS, AU, CS, or CAS—is closest to each LLM
   trajectory?
6. How stable are these effects across model families and model scale?

Questions 2–5 should be labeled confirmatory only if they are preregistered before
the first AI Race model output is inspected. Until then, this document describes
planned estimands rather than a preregistration.

## Experimental unit and treatment

- Experimental unit: a two-agent race (`game_id`).
- Decision unit: player-round, nested within race.
- Between-race treatment: maximum private setback risk.
- Both seats use the same model in the baseline; asymmetric-model races are a future
  extension.
- Actions within a round are simultaneous and generated from the same pre-action
  snapshot.
- The stopping horizon is hidden from agents and drawn from a separate deterministic
  RNG stream.

The baseline uses common environment seeds by repetition across risk treatments.
Horizon and fixed-seat setback streams are deterministic, separated, and logged.
Offline generation also receives per-decision seeds. Hosted runs record seed
request, SDK forwarding, and confirmed application separately; sampling common
random numbers are not claimed when the provider does not confirm seed support.

## Primary estimands

- Treatment-level mean Unsafe frequency.
- Conditional Unsafe frequency after opponent Safe versus opponent Unsafe.
- Conditional Unsafe frequency while ahead, tied, or behind.
- Association between first-round Unsafe and later Unsafe.
- Winner-versus-loser mean Unsafe frequency.
- Parse-failure rate and retry count as protocol health measures.

The paper-style panel model for rounds `t >= 2` is:

```text
unsafe
  ~ C(max_private_risk)
  + first_round_unsafe
  + own_prev_unsafe * opponent_prev_unsafe * progress_gap_before
```

Inference uses the common-random-number block `source_run + model + rep`, which
contains the matched risk-treatment races and therefore subsumes within-race
dependence. Coefficients describe conditional associations, not causal effects:
lagged actions summarize an endogenous interaction history.

Every output row carries `run_phase` (`pilot` or `confirmatory`) and
`prompt_version`. Pilot and confirmatory observations, or incompatible prompt
versions, must not be pooled silently. The checked-in baseline remains `pilot`
until the prompt, configurations, exclusions, and analysis plan are frozen.

## Canonical reduced strategies

- **AS**: Safe in every round.
- **AU**: Unsafe in every round.
- **CS**: Safe in round 1, then copy the opponent's previous action.
- **CAS**: Unsafe in round 1, then copy the opponent's previous action.

A rule such as "play Unsafe when behind" is scientifically relevant but is an
exploratory extension, not one of the paper's four reduced strategies.

## Validation gates on Kaggle

Before interpreting behavior:

1. Confirm the task uses exactly two independent player contexts.
2. Confirm both prompts in a round use the identical pre-action state.
3. Confirm all responses parse to Safe or Unsafe; report retries and failures.
4. Confirm realised horizons follow the minimum-5, stop-probability-0.2 mechanism
   and average approximately 9 only over a sufficiently large diagnostic sample.
5. Confirm winner/tie prize allocation and winner-only private setback handling.
6. Confirm no prompt reveals the pre-sampled terminal round.
7. Confirm `turns.jsonl`, `races.csv`, and `players.csv` join on stable identifiers.

No local execution is part of the current refactor. Passing source review is not a
substitute for these runtime checks.

## Planned sequence

1. Run a small Kaggle smoke configuration for one open-source model.
2. Inspect parsing, prompt fidelity, simultaneous decisions, horizons, and terminal
   scoring.
3. Freeze prompt/config versions.
4. Run the full three-treatment baseline for each selected model.
5. Run the analysis script without changing outcome definitions after seeing results.
6. Only then populate manuscript Results and slide figures.
