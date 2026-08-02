# Kaggle cross-model scaffold smoke v1 — failed run receipt

- Kernel: `daosyduyminh/ai-race-impact-admission`, version 1
- URL: https://www.kaggle.com/code/daosyduyminh/ai-race-impact-admission
- Observed terminal status: `KernelWorkerStatus.ERROR`
- Intended scope: Qwen2.5-7B, English, smoke profile, 160 comprehension requests
- Evidence status: **failed; zero admitted requests and zero gameplay claims**

The output/log download endpoint returned HTTP 429 during the audit window, so
no raw failure log is claimed here. Version 1 cloned the full public repository
at runtime. Version 2 replaces that fragile dependency with the private,
versioned `daosyduyminh/ai-race-admission-source/1` Dataset and keeps the same
admission-only scientific scope.

This directory is deliberately under `results/failed_runs/`. Catalog and paper
analyses must not count it as completed evidence.
