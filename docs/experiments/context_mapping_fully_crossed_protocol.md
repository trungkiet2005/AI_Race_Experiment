# Fully crossed context × action-mapping follow-up

## Status and claim boundary

This is a **preregistered diagnostic pilot**, not a confirmatory estimate of
game understanding or stable preference. The source checkpoint failed the
existing comprehension admission gate, so no behavioral result from this run
may be promoted beyond checkpoint-scoped prompt-conditioned behavior. It must
not be pooled with confirmatory evidence.

## Why this run is next

The completed context pilot balanced opaque action-code mapping by repetition
parity. That was adequate for its planned average context contrast, but the
post-hoc trajectory audit found that all observed divergence occurred when
`P` denoted Safe. Mapping and seed were therefore not crossed within a paired
unit. This follow-up closes that specific identification gap with the least new
compute.

## Frozen design

- Model: exact Ollama model name and digest recorded at launch.
- Decoding: temperature 0, maximum 16 output tokens, fixed sampling seed, at
  most three unchanged-prompt parse retries.
- Mechanism: the three frozen risk games (0.1, 0.6, 0.9); no payoff, progress,
  horizon, information, or setback change.
- Prompt factors: eight frozen context skins × two frozen P/Q mappings.
- Pairing: 32 repetitions. For every risk/context/repetition block, both
  mappings share the same game seed and player-round sampling seeds.
- Unit counts: 8 × 2 × 3 × 32 = 1,536 races before exclusions.
- Exclusion: one final parse failure contaminates and excludes the entire race.
- Execution: both GPU lanes run all eight contexts on disjoint even/odd
  repetition shards; context is never assigned by hardware lane. Each lane
  writes a resumable manifest to the shared persistent disk.

## Frozen estimands

1. **Primary interaction estimand:** difference-in-differences in semantic
   Unsafe rate,
   `(context − abstract) at Safe=P − (context − abstract) at Safe=Q`, reported
   separately by context and pooled across risk only after showing risk strata.
2. **Primary trajectory estimand:** paired probability of any action divergence
   from the abstract control by round, using a Kaplan–Meier estimator with race
   termination treated as censoring.
3. **Secondary estimands:** first-round action disagreement, final payoff
   difference, progress difference, and setback incidence; each reported by
   mapping and player role.

Uncertainty is clustered by the independent common-random-number repetition
stream. All three risk strata reuse `base_seed + repetition`, so risk must not
be counted as an independent cluster. Both player roles are retained. Seven non-control context
interactions use Holm correction. Raw decision rows are not treated as
independent Bernoulli trials.

## Admission and integrity gates

- exact source/config/model/prompt/mechanism hashes present;
- 1,536 completed races and the expected 16 prompt cells per risk;
- identical seed and mechanism signatures across both mappings in every paired
  block;
- no silent seed dropping and a passing repeated fixed-seed probe;
- parse failures and retries reported, never rescued from prose;
- rule-recall, state-update, terminal-scoring, and expected-payoff comprehension
  reported before gameplay claims;
- any failed comprehension domain keeps the result diagnostic.

## Promotion rule

The mapping interaction is promoted from “pilot target” to “replicated
diagnostic” only if its direction matches the original audit and the Holm-
adjusted interval excludes zero. It remains non-causal evidence about internal
understanding. A confirmatory behavioral claim additionally requires an admitted
checkpoint and independent model-family replication under the same frozen grid.

## Launch commands

Run one lane per GPU after replacing the current GreenNode SSH NodePorts:

```bash
python -m kaggle.experiments.greennode_context_mapping_cross \
  --lane a --profile pilot --repo-root /network-volume/icse27/AI_Race_Experiment \
  --output-root /network-volume/icse27/AI_Race_Experiment/results/open_source/context_mapping_cross/lane_a \
  --required-model-digest <exact-ollama-digest>

python -m kaggle.experiments.greennode_context_mapping_cross \
  --lane b --profile pilot --repo-root /network-volume/icse27/AI_Race_Experiment \
  --output-root /network-volume/icse27/AI_Race_Experiment/results/open_source/context_mapping_cross/lane_b \
  --required-model-digest <exact-ollama-digest>
```
