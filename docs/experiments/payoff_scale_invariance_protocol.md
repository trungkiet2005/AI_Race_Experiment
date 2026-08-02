# Positive payoff-scale invariance diagnostic

## Research question

Does an LLM agent preserve its action trajectory when every utility in the AI
Race is expressed in a different positive numerical unit? Multiplying all four
stage payoffs and the terminal prize by the same positive factor leaves the
strategic problem unchanged. Progress, private-risk accumulation, setback
draws, and the stochastic horizon are untouched.

This is a **diagnostic pilot**, not evidence of a stable risk preference. A
failure shows sensitivity to payoff representation under the tested checkpoint
and protocol; it does not by itself identify the internal cause.

## Frozen design

- factors: `0.1`, `1`, `10`, and `100`;
- three maximum-private-risk treatments;
- 32 common-random-number repetitions;
- 384 races total, with all four factors inside every risk-by-seed block;
- temperature 0, exact model digest, strict one-line parser, and unchanged-prompt
  retries;
- stage payoffs and race prize are scaled together; no additive shift is used
  because terminal setback resets payoff to zero and would break affine
  equivalence.
- both GPU lanes run all four scales on disjoint even/odd repetition shards;
  effects are estimated within lane before pooling.

## Endpoints

1. Paired first-round disagreement against factor `1`.
2. Probability and round of first complete-trajectory divergence.
3. Unsafe-rate difference after normalizing by risk and player role.
4. Rescaled final-payoff equality among pairs whose action trajectories remain
   identical.

Intervals resample the `(risk, repetition)` block. The three non-reference
factors use Holm correction. Decision rows are never treated as independent
trials.

## Behavioral analysis admission

`results/scripts/analyze_payoff_scale_behavior.py` fails closed unless it finds
all 384 races and all four scales in every risk-by-repetition block. It also
requires completed manifests, one exact non-empty model digest, identical
temperature-zero decoding, matching source/config hashes, identical horizons,
stop draws, game seeds, and seat-specific setback draws, plus unique race,
player, and turn keys. If one decision has a parse failure, the complete
four-scale `(risk, repetition)` block is excluded and the loss is reported.

The analyzer writes paired player-level rows, race-block bootstrap intervals,
and Holm-adjusted sign-flip tests for the three signed Unsafe-rate contrasts.
Disagreement rates are reported directly: a non-significant contrast is never
used to claim equivalence. Run it with at least 1,000 frozen resamples:

```bash
python -m results.scripts.analyze_payoff_scale_behavior \
  --input-root <downloaded-payoff-scale-root> \
  --output-dir results/derived/payoff_scale_behavior \
  --bootstrap-repetitions 5000
```

## Mechanical admission test

`results/scripts/analyze_payoff_scale_contract.py` exhaustively checks every
two-player joint-action sequence through seven rounds, all three risk levels,
four setback-draw pairs, and all four scales. Non-payoff terminal fields must
match exactly and every scaled payoff must equal `factor × reference` within
floating-point tolerance before any model run is admitted.

## GPU launch

```bash
python -m kaggle.experiments.greennode_payoff_scale \
  --lane a --profile pilot --repo-root /network-volume/icse27/AI_Race_Experiment \
  --output-root /network-volume/icse27/AI_Race_Experiment/results/open_source/payoff_scale/lane_a \
  --required-model-digest <exact-ollama-digest>

python -m kaggle.experiments.greennode_payoff_scale \
  --lane b --profile pilot --repo-root /network-volume/icse27/AI_Race_Experiment \
  --output-root /network-volume/icse27/AI_Race_Experiment/results/open_source/payoff_scale/lane_b \
  --required-model-digest <exact-ollama-digest>
```
