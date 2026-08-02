# Native-Qwen context mediation with FAST SAE

## Scope

This protocol asks a narrower question than the live context simulation: when
the numerical game state is identical, which activation changes accompany a
change in the pinned model's Safe/Unsafe preference after only the cover story
changes? It uses the native
`Qwen/Qwen2.5-7B-Instruct@a09a35458c702b33eeacc393d103063234e8bc28`
checkpoint and
`Geaming/Qwen2.5-7B-Instruct_SAEs@5a7ecabe1401bf4de11a0e6da1f7c36bbb46a464`.
It never imports actions produced by Ollama or a previous run.

The runner is
`kaggle/interpretability/greennode_context_fast_sae.py`. It uses the canonical
engine to create reachable states and the frozen eight-skin renderer. For every
matched `(state_id, skin_id, P/Q mapping)` cell, it scores the complete
sequences `ACTION: P` and `ACTION: Q`, including EOS, and then decodes the
chosen opaque code to canonical Safe/Unsafe. The activation capture is the
final prompt token before either action response begins.

## Leakage controls

Both source trajectories and context pairs are held out.

- Discovery pairs: technology vs abstract; logistics vs crystal-guild.
- Evaluation pairs: hospital vs lunar-colony; robotic expedition vs fictional
  cartography.
- A whole `(game treatment, source trajectory)` is assigned to discovery or
  evaluation; its rounds, seats, mappings, and context renders cannot cross.
- Feature ranking and dose scale use only discovery trajectories crossed with
  discovery context pairs.
- Confirmation and intervention use evaluation trajectories crossed with the
  two unseen evaluation pairs.

The action probe reports four quadrants, so generalization to unseen states and
unseen contexts is visible separately. Its AUC is predictive evidence only.
Feature mining ranks discovery-only SAE code deltas by standardized separation
between action-flip and stable matched pairs, with association to the matched
change in Unsafe-vs-Safe log odds as the tie-break. If discovery contains no
flips, the runner records that the ranking has fallen back to continuous
log-odds association and it must not be reported as flip-feature mining.

## Intervention admission

Selected decoder directions are evaluated only on double-held-out prompts. The
frozen control family is:

- zero intervention;
- full SAE reconstruction replacement;
- current target-feature ablation;
- positive and negative target-feature doses;
- equal-norm seeded random directions orthogonal to the target;
- an active but discovery-unrelated SAE feature.

Do not describe a feature as causal mediation unless the target direction
exceeds matched controls, changes monotonically with dose, reverses with sign,
and is not explained by poor SAE reconstruction. Even if those checks pass,
the conclusion is local to this checkpoint, layer, SAE, state bank, and context
set. The fixed-state runner does not estimate terminal payoff effects; those
belong to the separate live-trajectory runner.

## GreenNode commands

The shared `/network-volume` mount is rejected by code. Run under
`/home/jovyan`, one heavy process per pod, then SCP immutable artifacts back.

```bash
# Pod A: earlier-middle layer
python -m kaggle.interpretability.greennode_context_fast_sae \
  --stage all --profile smoke --layer 12 \
  --output-dir /home/jovyan/ai_race_runs/context_fast_sae_smoke_l12

# Pod B: later-middle layer
python -m kaggle.interpretability.greennode_context_fast_sae \
  --stage all --profile smoke --layer 20 \
  --output-dir /home/jovyan/ai_race_runs/context_fast_sae_smoke_l20
```

Inspect both manifests and raw prompt/action rows before promoting the exact
same commands to `--profile pilot`. Pilot is 32 reachable states per risk, all
eight skins, and both counterbalanced P/Q mappings. Use `--resume` only with an
identical configuration fingerprint. `capture`, `analyze`, and `steer` can be
run as separate stages for recovery; analysis is CPU-compatible after the GPU
capture completes.

Required promotion checks:

1. exact model and SAE revisions plus `sae-lens==5.3.3` are in the manifest;
2. every state has exactly 16 context/mapping cells;
3. raw prompts contain no canonical `SAFE` or `UNSAFE` action labels;
4. trajectory and context discovery/evaluation sets have zero overlap;
5. baseline steering replay error is within the frozen tolerance;
6. all artifact hashes validate after SCP.
