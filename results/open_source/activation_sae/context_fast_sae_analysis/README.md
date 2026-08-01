# Context FAST-SAE smoke audit

## Decision

Promote **Layer 20 capture + analysis only** to the larger exploratory pilot. Hold Layer 12, and do not promote causal steering for either layer yet. Layer 20 has stronger double-held-out probe association (AUC 0.985, accuracy 0.929, n=56) than Layer 12 (AUC 0.922, accuracy 0.893). This prioritization is operational, not a claim that Layer 20 explains the decision.

Both runs passed artifact hashes, exact baseline replay, 12-state × 8-context × 2-mapping coverage, opaque-label checks, and whole-trajectory/context-pair separation. The model checkpoint, state bank, prompts, baseline action scores, and context-pair rows are byte/exact-value aligned across layers.

## 1. Context-shift descriptives

![Context-shift descriptives](context_shift_descriptives.png)

Changing only the story changed continuous action preference on fixed states. The largest held-out contrast was `robotic_expedition__vs__fictional_cartography` with mean |Δ Unsafe log odds| 1.214. Held-out context pairs produced only 1 discrete flip across 28 matched pairs. These are model-behavior descriptives shared by both layer runs; SAE-code distances are layer-specific and should not be read as directly comparable calibrated magnitudes.

## 2. Held-out probe association

![Held-out action probe](heldout_action_probe.png)

The discovery-only linear probe generalized to unseen trajectories and unseen context families. Layer 20 is the stronger screening candidate, but AUC measures decodability/association. It does not identify why the model chose P/Q and does not establish mediation by any selected SAE feature.

## 3. Causal steering versus controls

![Causal steering controls](causal_steering_controls.png)

Causal admission fails at both layers. Discovery feature ranking is dominated by **one action flip among 20 discovery pairs**. Across all held-out target, random, unrelated, reconstruction, ablation, and sign-control interventions, the action-flip rate is 0. Target dose curves do not show a reliable sign-reversing pattern beyond controls; mean shifts are sparse while medians remain near numerical zero (maximum absolute intervention median: Layer 12 9.54e-07, Layer 20 5.36e-07). Therefore these results do not support a causal or neuron-level “reason” claim.

## Promotion boundary

- **Layer 20:** run `capture`, then `analyze`, at pilot scale. Require at least 10 discovery flips before running or interpreting `steer`; otherwise switch feature selection to a preregistered continuous log-odds target.
- **Layer 12:** hold as a smoke robustness point; do not spend a full pilot lane unless cross-layer replication becomes the primary question.
- **Neither layer:** no causal mediation claim and no steering promotion from this smoke.

This is an exploratory smoke audit of one checkpoint, two SAE layers, four context contrasts, and a small state bank. The observed effect sizes are not confirmatory estimates.
