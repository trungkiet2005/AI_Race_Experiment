# GreenNode heterogeneous Qwen–Mistral diagnostic

This artifact contains the first reproducible, seat-routed heterogeneous-model
AI Race run in the repository.

- Source commit: `ba2906ae0f32fdd1af69d9d68e1c8f26b00012d4`
- Checkpoints: Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.1
- Hardware: two H100 MIG lanes, then model-to-lane assignment swapped
- Coverage: 384 races, 4,992 decisions, 0 final parse failures
- Design: same/cross checkpoint, cross-family seat reversal, 2×2 self/opponent
  identity-label disclosure, neutral/competitive role, and three risk levels
- Evidence class: **diagnostic, unadmitted**; both checkpoints failed the frozen
  state-update and terminal-scoring comprehension gates

Read the illustrated findings and validity boundary in
[`analysis/README.md`](analysis/README.md). The directly usable raw records and
manifests are under [`ai_race_hetero_ba2906a/results/`](ai_race_hetero_ba2906a/results/).

The full persistent-pod artifact—including mailbox request/response files,
worker receipts, logs, and stopped-worker receipts—is preserved as
`ai_race_hetero_ba2906a_full.tar.gz` with SHA-256:

```text
908f1d678fd5ea2c3244534c68cc0975c97f23dfc69441edda56a8707ed606a5
```

Block 2 is a lane-counterbalance robustness replication. It is not pooled into
the behavioral rates in the report. Exact per-decision agreement across lane
assignments was 98.6% (Qwen 97.5%, Mistral 99.7%), which is itself an important
temperature-zero reproducibility warning.
