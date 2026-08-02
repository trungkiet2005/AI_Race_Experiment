# Heterogeneous Qwen–Mistral dyad diagnostic

**Evidence class: diagnostic, unadmitted.** Both checkpoints failed the frozen
state-update and terminal-scoring comprehension gates. These results measure
prompt-conditioned enacted actions; they do not establish strategic
understanding, expected-payoff optimization, or model-family universals.

## What was run

- 2 lane-counterbalanced blocks; **384 races** and
  **4992 decisions** total.
- Exact BF16 checkpoints: Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.1.
- Same-checkpoint controls plus Qwen→Mistral and Mistral→Qwen seat reversal.
- 2×2 self-identity × opponent-identity disclosure, neutral/competitive role,
  risks 0.1/0.6/0.9, temperature 0.
- Block 2 swapped the models across the 20GB and 40GB H100 MIG lanes.

## Validation

- Validation passed: **True**.
- Final parse failures: **0**.
- Lane-matched actions: **2496**;
  agreement **98.6%**
  (35 mismatches).
- By checkpoint: Qwen agreement
  **97.5%**;
  Mistral agreement
  **99.7%**.
- Source commit: `ba2906ae0f32fdd1af69d9d68e1c8f26b00012d4`.

Block 2 is a technical lane replication and is not pooled into behavioral
rates. All rates below use block 1 only.

## Main diagnostic observations

1. Overall Unsafe rate was **75.2%** for Qwen and
   **93.3%** for Mistral under this factorial.
2. The largest raw opponent-label contrast was
   **-41.7 percentage points** for
   `mistral7_01` / `cross_family` /
   `neutral` / `round 1`
   (n=24 accurate-label and
   n=24 not-disclosed decisions).
   This is a surface-label effect in a smoke diagnostic, not a confidence-bounded
   population estimate.
3. Live-race position is endogenous. For two players,
   `progress_gap_before = 0.5 × (own prior Unsafe count − opponent prior Unsafe count)`;
   therefore the position figure is association only. A randomized progress
   endowment / matched fork is required for a causal first/middle/last claim.

## Position rates used in the descriptive figure

| Model | Persona | Behind | Tied | Ahead |
|---|---|---:|---:|---:|
| Mistral 7B | competitive | 100.0% | 96.4% | 100.0% |
| Mistral 7B | neutral | 100.0% | 84.3% | 100.0% |
| Qwen 2.5 7B | competitive | 86.0% | 57.1% | 100.0% |
| Qwen 2.5 7B | neutral | 93.9% | 72.5% | 100.0% |


## Figures

![](figures/identity_disclosure_matrix.png)

![](figures/risk_response_same_vs_cross.png)

![](figures/endogenous_position_response.png)

## Robustness boundary and next experiment

- Temperature zero gives deterministic checkpoint behavior for a fixed prompt;
  repeated horizons are not independent model samples. The 1.4% cross-lane
  mismatch shows greedy GPU inference was not bitwise invariant to lane/runtime;
  this is reported as a robustness result, not averaged away.
- Accurate and not-disclosed arms differ in tokens, so the estimand is label
  disclosure, not hidden recognition of the opponent's family.
- Persona is a prompt-conditioned role, not a stable personality.
- The next causal position experiment should apply an engine-scored randomized
  progress endowment after a common prehistory, query the immediate action, and
  roll matched branches forward with identical random streams.
- N=3 must record `n_ahead` and ties explicitly: strict leader/middle/last are
  `n_ahead = 0/1/2` only when no other player is tied.
