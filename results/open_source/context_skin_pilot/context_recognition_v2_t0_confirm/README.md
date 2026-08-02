# Context-recognition audit v2

This is the clean rerun of `ai-race-context-recognition-audit-v2` after the v1
pilot exposed a contradictory candidate contract. Version 1 remains archived
separately and was not rescored.

The temperature-zero confirmatory matrix contains 16 requests: eight context
skins crossed with both P/Q mappings. All 16 responses were strictly valid on
the first attempt. Every response selected
`generic_structural_resemblance`, high confidence, and `candidate: null`; none
reported a specific named game or benchmark. Recognition class agreed across
the two mappings for all eight contexts.

This is a model self-report, not evidence of training-data contamination,
memorisation, latent understanding, or causal recognition. Conversely, the
absence of a named candidate does not establish the absence of contamination.
The audit was isolated from gameplay and from the comprehension admission gate.

Both lanes ran sequentially on one NVIDIA H100 with the same exact
`qwen2.5:7b-instruct-fp16` digest
`59805ce4a4046be2d8f63231a78daacd2e66f5dccf1a64d0d138ebeeb26ff16c`.
See each lane's `run_manifest.json`, `recognition_rows.jsonl`, and
`recognition_summary.json` for prompts, raw outputs, seeds, hashes, and full
provenance.
