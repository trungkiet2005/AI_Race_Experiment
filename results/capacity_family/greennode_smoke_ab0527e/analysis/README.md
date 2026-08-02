# GreenNode capacity/family admission smoke

Status: **diagnostic comprehension failure; gameplay not admitted**.

Every checkpoint below completed 160/160 unique probes with greedy decoding, BF16 parameters, and CUDA-only placement. All failed the frozen admission rule, so no live-game behavior was launched.

| Model | Strict parse | Rule recall | Stage payoff | State update | Terminal scoring | Admission |
|---|---:|---:|---:|---:|---:|---|
| qwen2.5-7b-instruct | 47.5% | 97.5% | 100.0% | 2.5% | 0.0% | **FAIL** |
| qwen2.5-14b-instruct | 98.1% | 100.0% | 100.0% | 12.5% | 0.0% | **FAIL** |
| mistral-7b-instruct-v0.1 | 100.0% | 95.0% | 100.0% | 0.0% | 2.5% | **FAIL** |

## Main diagnostic

Moving from 7B to 14B changed strict parsing from 47.5% to 98.1%, but did not solve the substantive bottleneck: state-update accuracy was 2.5% versus 12.5%, and terminal scoring was 0.0% versus 0.0%. Both models were perfect on rule recall and stage payoff. This is a two-checkpoint smoke, not a scaling estimate.

All disclosed arithmetic-tool outputs matched the engine and no hidden-information leak was detected. That verifies the scaffold interface, not internal game understanding.

## Family replication

Mistral-7B achieved 100.0% strict parsing, 95.0% rule recall, 100.0% stage-payoff accuracy, 0.0% state-update accuracy, and 2.5% terminal-scoring accuracy. This adds a checkpoint-template replication, not a family-wide estimate.

## Provenance and integrity

- Source commit: `ab0527eba990dea2620bc03a37b2c33673e58949`
- Shared request bank: `8f85c813283251593a1a5dc89df241805c9fe85da6b72dae006c5287838f3b24`
- Download archive SHA-256: `2159d1d5aba0e2e2cacca250f2e5d9ba74d8397486472e8aae249f0aa0871717`
- Download archive: `ai_race_capacity_ab0527e_smoke_3models.tar.gz`
- Raw evidence: `results/*/smoke/comprehension_raw.jsonl`
- Runtime receipts: `results/*/smoke/run_manifest.json`

## Interpretation boundary

The run has one repetition per scaffold cell and only three named checkpoints: two Qwen sizes and one Mistral checkpoint-template stack. It supports debugging and model admission decisions, not confirmatory model ranking, a universal scaling law, or any claim about gameplay behavior.
