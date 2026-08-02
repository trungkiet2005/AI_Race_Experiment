# Exogenous position-endowment diagnostic

Status: **completed unadmitted behavioral diagnostic**. Block 1 is the only primary behavioral block; block 2 is retained exclusively as a lane-reproducibility audit.

Both checkpoints failed the separate comprehension admission gate. These actions therefore show how the named checkpoint-template stacks responded to frozen prompts; they do not demonstrate game understanding, expected-payoff optimization, or a family-wide strategic trait.

## Frozen design and validation

- Source commit: `e3cf82523bd2a342f9cfb62db2fd445682390756`
- Rows: 192 per block = 96 probe IDs × 2 checkpoints
- Parse failures: 0 in each validated block
- Primary: block 1; lane audit only: block 2
- Temperature: 0; native sampling seeds were recorded but not applied
- Intervention: an engine-scored exogenous progress adjustment after one common four-round history
- Surface controls: both opaque P/Q mappings and numeric-only versus verified rank-label prompts
- Mailbox audit: experiment requests use `ai-race-position-endowment-v1`; the shared worker envelope uses `ai-race-heterogeneous-dyad-v1` and is separately hash/request-ID validated

## Primary direct effects

The table reports exact percentage-point differences from the numeric-only arm in block 1. It does not attach sampling confidence intervals because there is one deterministic response per frozen prompt and only one common history.

| Checkpoint | Behind − ahead (2P) | Last − leader (N=3) | Last − middle (N=3) |
|---|---:|---:|---:|
| Qwen2.5-7B | +0.0 pp | +41.7 pp | +8.3 pp |
| Mistral-7B | +0.0 pp | +0.0 pp | +0.0 pp |

![Primary position response](primary_position_response.png)

![Primary direct contrasts](primary_direct_contrasts.png)

## Lane reproducibility

Exact matched-probe action agreement was 94.8% for Qwen2.5-7B and 100.0% for Mistral-7B. Block 2 is not pooled with block 1: identical design cells on a second lane test runtime reproducibility, not an independent behavioral population.

![Lane reproducibility](lane_reproducibility.png)

## Causal and interpretation boundary

The progress adjustment is exogenous and explicitly engine-scored, so a matched rank contrast has a causal **direct fixed-state prompt** interpretation within this frozen state bank. It estimates the immediate response to displayed and payoff-relevant position while prior actions, stage payoff, private risk, and the decision round are held fixed.

It is not the total effect of falling behind in a live game. A live intervention can change the opponent's later actions, the focal model's future prompts, accumulated risk, stopping opportunities, and terminal outcomes. Estimating that total feedback effect requires replay-to-fork live trajectories with common future environment streams. The present direct effect and a future live total effect answer different questions and must not be pooled.

Further boundaries:

- The bank contains one common four-round history, so it does not establish generality across histories, rounds, or gap magnitudes.
- The verified-label arm changes both numeric state and an explicit lexical label; numeric-only is primary and label differences are surface-sensitivity evidence.
- Temperature-zero exact repetitions do not create independent model samples.
- Qwen2.5-7B and Mistral-7B differ in weights, tokenizer, chat template, and training. Their difference is checkpoint-template heterogeneity, not a universal model-family effect.
- Because admission failed, no result should be described as rational adaptation or a learned world model.

## Files

- `primary_position_rates.csv`: block-1 rates by checkpoint, game size, label, mapping, and position
- `primary_direct_contrasts.csv`: prespecified direct position contrasts
- `lane_reproducibility_summary.csv`: block-level rate and exact-action agreement
- `probe_level_lane_comparison.csv`: one-to-one block comparison
- `mailbox_validation.csv`: request/response hashes, routes, counts, and the explicit transport/experiment protocol split
- `block_validation.csv` and `quality_audit.json`: provenance, admission-digest, and coverage checks
