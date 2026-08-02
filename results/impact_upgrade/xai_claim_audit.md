# XAI claim audit: paper-safe evidence, causal boundaries, and next experiments

Audit date: 2026-08-01
Scope: the impact-upgrade synthesis, the cross-runtime surface FAST-SAE probes, the native context FAST-SAE screen at layers 12 and 20, and the native layer-12 actual-self-play intervention pilot.
Decision rule: a result is called *causal* only when the intervention is replay-exact, target-specific relative to frozen controls, sign/dose coherent, behaviorally neutral under reconstruction, and supported on held-out race clusters.

## Bottom line

The strongest defensible XAI result is a **decodability-without-control dissociation**. SAE-coded residual activations predict Safe/Unsafe-associated scores on held-out prompts and races, including a double-held-out context/state AUC of 0.985 at layer 20. However, neither context-screen layer produced an action flip in 1,312 intervention rows per layer, and the self-play audit found 0 of 12 preregistered strongest-dose target-minus-control intervals excluding zero. The repository therefore supports a representation-association result and a well-controlled negative causal result; it does **not** support a claim that a discovered SAE feature is the model's reason for choosing Unsafe or a causal strategy controller.

The new impact synthesis adds a compelling behavioral target for future mechanistic work: context-induced trajectory divergence is perfectly gated by the opaque Safe=P versus Safe=Q mapping in this diagnostic pilot. Because mapping was assigned by repetition parity, that interaction is a preregistered hypothesis target, not a clean causal mapping estimate.

## 1. Evidence and integrity audit

| Evidence block | Intended grain | Machine-readable denominator | Integrity result | Admissible class |
|---|---|---:|---|---|
| Impact synthesis | Qwen decision / race / paired player trajectory | 13,680 decisions; 768 races; 1,344 context-reference trajectory pairs | All 8 listed inputs and all 22 listed analytical outputs match SHA-256 after regeneration | Diagnostic synthesis with verified analytical provenance |
| Cross-runtime surface FAST-SAE | Decision activation | 600 decisions; 480 train and 120 grouped evaluation; five layers | Complete manifests; zero cross-split exact-prefix overlap; model runtime mismatch is explicit | Exploratory association only |
| Native context FAST-SAE | Fixed-state prompt/action cell and intervention row | 192 captured cells and 1,312 intervention rows per layer; 32 held-out prompts per layer | Run complete at both layers; 11/11 derived outputs match SHA-256; all six stage artifacts per layer are present and match their manifest hashes | Exploratory fixed-state association/intervention screen |
| Native self-play FAST-SAE | Whole race, replay-exact decision, and live steered trajectory | 18 source races / 300 decisions; 24 fixed held-out decisions / 984 intervention rows; 114 live trajectories over six unique seeds | 41 runner-recorded sources verified; 23/23 derived artifacts match SHA-256; baseline replay error 0.0 | Exploratory causal audit with negative specificity result |

### Integrity qualifications

1. The two referenced `context_skin_invariance.json` inputs have been restored from the canonical repository config; each is 822 bytes and matches the manifest SHA-256 `1961aadf…` exactly.
2. `context_fast_sae_analysis/summary.json` now hashes LF-normalized text outputs, and `impact_upgrade/analysis_manifest.json` was regenerated against that frozen summary; all eight inputs pass.
3. `release_manifest.json` hashes the report, demo, paper, deck, audits, analytical tables, and figures as a separate delivery-layer receipt. It should be regenerated only after the final release edit.
4. Native FAST-SAE manifests pin the model, SAE, package version, runner hash, and configuration fingerprint, but `git_revision` is null. Source hashes and the release manifest substantially mitigate this, yet the final release commit should still be recorded.

## 2. Exact paper-safe claims

The sentences below can be used in the paper with their scope qualifiers intact.

### Behavioral target for XAI

> In a diagnostic Qwen2.5-7B-Instruct context-skin experiment with fixed payoff mechanics, all 1,344 paired player trajectories agreed in round 1, but six of seven non-reference contexts later diverged from the abstract reference under one opaque action mapping. Under Safe=P, all 576 pairs belonging to those six contexts diverged (576/672 across all seven tested contexts); under Safe=Q, none of 672 context-reference pairs diverged. Because mapping was assigned by repetition parity, this pattern identifies a mapping-by-context replication target rather than a mapping-causal effect.

> Fixed-state replay detected direct context-associated Unsafe-rate differences up to 16.7 percentage points, while live trajectory contrasts reached 34.0 points. The live-minus-fixed gap is descriptive: it combines repeated prompt exposure with endogenous state feedback and is not a causal mediation estimate.

### Cross-runtime surface representation

> On 600 logged decisions split by connected components of whole-race membership and exact causal-prefix identity (480 train, 120 evaluation), linear probes over pretrained FAST-SAE codes achieved mean held-out ROC-AUC 0.872 across five layers immediately after the common `ACTION:` prefix, compared with a mean shuffled-label AUC of 0.496 over 100 layer-shuffle fits.

> At the stricter final-prompt-token position before response boilerplate, mean held-out ROC-AUC remained 0.842 versus a shuffled-label mean of 0.492. Removing the common response prefix reduced mean AUC by 0.030, with a positive reduction at all five audited layers.

Required qualifier: the logged actions came from an Ollama GGUF F16 runtime, whereas the activations came from the pinned native Hugging Face checkpoint. These results show cross-runtime linear recoverability, not a causal mechanism or a faithful attribution of the sampled action.

### Native fixed-state context representation

> With the pinned native Qwen2.5-7B-Instruct checkpoint and FAST SAE, a layer-20 probe evaluated on both unseen state trajectories and unseen context pairs achieved ROC-AUC 0.985 and accuracy 92.9% on 56 prompt/action cells (17 Unsafe, 39 Safe). The analogous layer-12 result was AUC 0.922 and accuracy 89.3% on the same denominator.

> Despite this held-out decodability, no action changed in 1,312 intervention rows at layer 12 or 1,312 intervention rows at layer 20, each built from 32 held-out prompts. Thus the context screen found predictive representations but no action-level evidence that the selected SAE directions controlled the decision under the tested interventions.

Required qualifier: this was a smoke profile with 12 reachable states per layer. Feature discovery had only one context-induced action flip in discovery and one in the double-held-out split. The 1,312 rows per layer are repeated intervention conditions over 32 prompts, not 1,312 independent decisions.

### Native actual-self-play association and intervention

> In native constrained-policy self-play, feature selection used 12 discovery races (206 decisions) and evaluation used six held-out races (94 decisions). The strongest held-out oriented action AUC was 0.880 for feature 16320, with an Unsafe-score correlation of -0.693. Feature 8505 had held-out AUC 0.819 and correlation +0.692; feature 1803 had oriented AUC 0.727 and correlation -0.700. These are association statistics.

> The fixed-state audit replayed 24 held-out decisions exactly and produced 984 intervention rows. At the preregistered strongest dose, 0 of 12 target-minus-control cluster-bootstrap intervals excluded zero, comparing each selected target with matched-random and unrelated active-feature controls. This pilot therefore did not establish target-specific causal control.

> Across the 12 live feature-by-sign-by-control contrasts, target and control action sequences matched exactly in 68.1% of paired races on average. Live runs used only six common-random-number seeds per condition; payoff changes are exploratory endogenous trajectory effects, not stable feature-specific payoff effects.

> Full SAE reconstruction changed the action in 3 of 24 fixed prompts (12.5%) and produced a mean absolute Unsafe log-odds change of 0.097. Reconstruction was therefore not behaviorally neutral, which further blocks a clean feature-ablation interpretation.

### Concise headline suitable for the abstract or discussion

> Sparse-autoencoder codes made action-associated information highly decodable on held-out states and contexts (maximum AUC 0.985), but controlled steering did not establish a feature-specific action mechanism: context steering yielded no flips, and 0/12 strongest-dose target-minus-control intervals excluded zero in self-play. This separates representational availability from behavioral control.

## 3. Strongest numerical insights

| Rank | Insight | Exact denominator | Why it matters | Claim boundary |
|---:|---|---:|---|---|
| 1 | Layer-20 double-held-out context/state probe AUC 0.9849; accuracy 0.9286 | 56 cells: 17 Unsafe, 39 Safe | Action-associated information survives both state and context holdout | Association only; no cluster CI |
| 2 | Zero action flips in context steering | 1,312 rows / 32 prompts at layer 12 and the same at layer 20 | Directly falsifies the simplest “high AUC implies controllable feature” story under tested doses | Local negative intervention result |
| 3 | 0/12 strongest-dose target-minus-control CIs exclude zero | 24 fixed decisions in six held-out race clusters per contrast | Target directions do not beat matched-random or unrelated-feature controls | Exploratory negative specificity result, not proof of no effect |
| 4 | Mean exact target/control live sequence match 68.1% | 12 contrast cells, six paired races each | Much live behavior is reproducible by controls, weakening feature-specific attribution | Descriptive; cells are small |
| 5 | SAE reconstruction itself flips 12.5% | 3/24 fixed held-out prompts | Reconstruction error is behaviorally material and can masquerade as feature effect | Calibration warning |
| 6 | Context divergence is gated by action mapping | Safe=P: 576/672 context-reference pairs diverge overall and 576/576 among six non-reference contexts; Safe=Q: 0/672 | Gives a concrete mechanism target linking surface form to trajectory feedback | Mapping is parity-confounded |
| 7 | Cross-runtime probe signal survives removal of response boilerplate | Mean AUC 0.842 vs shuffled 0.492 across five layers; 120 eval decisions | The association is not entirely the common `ACTION:` prefix | Runtime-mismatched association only |

One secondary diagnostic must not be promoted: 2 of 24 all-dose, unadjusted target-control intervals excluded zero, both for feature 16320 at alpha -1. These were not the frozen strongest-dose endpoint, arise inside a selected/multiple-contrast family, and lack monotone sign/dose support. They are hypothesis-generating only.

## 4. Claims that are negative, blocked, or prohibited

### Explicitly unsupported

- “Feature 16320/8505/1803 is an Unsafe neuron,” “safety feature,” “risk preference,” or monosemantic strategy variable.
- “The SAE explains why the model chose Safe or Unsafe.” It identifies decodable correlates; it does not identify a unique reason.
- “Steering the selected feature reliably changes behavior.” Target effects did not reliably exceed controls, and context steering caused no action flips.
- “The intervention improves payoff.” The largest target-condition mean payoff increase versus zero was +5.13 over six races with a 95% bootstrap interval of [-9.80, +25.20]; controls changed payoff on the same scale.
- “Context effects are mediated by the selected SAE feature.” Mediation admission was false at both audited context layers.
- “The mechanism generalizes across models, layers, contexts, or decoding policies.” Native causal evidence covers one checkpoint, one SAE family, one self-play layer, two context layers, and a constrained exact-sequence policy.
- “Zero flips prove the feature has no causal role.” The smoke bank, doses, intervention site, and limited action margins bound the negative result.
- “Live payoff differences are fixed-state causal effects.” Once a decision changes, later prompts and states are endogenous.

### Blocked until remediation

- The context-run invariance archive and parent impact manifest are now restored and hash-complete; future edits must keep both receipts synchronized.
- A causal context-by-mapping claim is blocked until mapping is fully crossed within seed blocks rather than assigned by repetition parity.
- A confirmatory SAE controller claim is blocked until reconstruction is behaviorally neutral and the frozen target outperforms a sufficiently large empirical random-direction null on held-out race clusters.

## 5. Reviewer risks and remediation

| Severity | Risk | Evidence | Consequence | Smallest remediation |
|---|---|---|---|---|
| High | Mapping and repetition are not fully crossed in the live context pilot | All divergence is concentrated under Safe=P; mapping uses repetition parity | The most dramatic behavioral interaction may include seed/block structure | Re-run both mappings inside every seed/context/risk block |
| High | Context feature discovery is nearly flip-free | One discovery flip and one double-held-out flip | Feature ranking and AUC can reflect broad state/action score structure rather than the rare context-induced decision change | Expand the state bank; require at least 10 discovery flips before feature selection or steering |
| High | SAE reconstruction is behaviorally non-neutral in self-play | 3/24 fixed actions flip under full reconstruction | Ablation/decoder-direction effects can be reconstruction artifacts | Gate on action neutrality plus log-odds tolerance before interpreting feature edits |
| High | Intervention specificity is not demonstrated | 0/12 extreme target-control CIs exclude zero; mean live exact match 68.1% | Causal feature language is not defensible | Freeze target/sign/dose and compare against many matched random directions on more held-out race clusters |
| Medium | Effective sample size is race/state limited | Self-play evaluation has six race clusters; context steering has 32 prompts/layer | Decision-row counts overstate independent information; CIs are unstable or absent | Increase clusters/states, report cluster bootstrap intervals and hierarchical sensitivity |
| Medium | Context probe lacks uncertainty at the correct grouped grain | AUC 0.985 is reported on 56 nested cells | Reviewers may read a point estimate as precise | Bootstrap whole trajectory/state groups and report interval plus class counts |
| Medium | Surface probe is cross-runtime and SAE codes are dense | Mean AUC remains high, but decision and attribution runtimes differ; prior audit reports 46%–63% active features | Weakens mechanistic and monosemantic interpretation | Keep as appendix association baseline; prioritize native-runtime results and report L0/fidelity |
| Medium | Constrained action policy limits external validity | Native self-play uses exact full-sequence likelihood and argmax | Results may not transfer to sampled free generation | Replicate the frozen intervention under a parser-audited free-generation policy |
| Medium | One checkpoint and one pretrained SAE family | All native XAI evidence is Qwen2.5-7B-Instruct + FAST | No cross-model mechanism generality | Replicate the locked protocol on a second open model with a pinned pretrained SAE |
| Low/operational | Native run `git_revision` is null; working tree is not yet a release commit | Manifest audit above | Residual release-traceability objection | Commit the verified tree, then regenerate the release manifest against that revision |

## 6. Three highest-value next XAI experiments

### Experiment 1 — Causal localization of the context × opaque-mapping gate

**Question:** Is the dramatic mapping-gated context effect carried by action-code token processing, narrative context processing, or their interaction?

**Design:** Fully cross the same state, context, risk, seat, seed, and both Safe=P/Safe=Q mappings. Add a third arm that randomizes the opaque mapping independently on every decision. Score exact P/Q sequences at temperature 0. Capture a layer sweep at the final prompt token and around the action-ID tokens. Perform paired residual activation patching from reference to alternate context and from Safe=P to Safe=Q, plus position-matched and random-vector controls.

**Primary endpoint:** paired change in Unsafe-minus-Safe sequence log-odds at fixed state.
**Secondary endpoint:** action flip only for prompts whose baseline margin lies inside a preregistered band.
**Admission:** identical engine state, full seed-block crossing, zero parser ambiguity, baseline replay error below tolerance, target patch exceeds control patches, and cluster interval excludes zero.
**Impact:** directly attacks the paper's most visually striking behavioral phenomenon and can distinguish semantic context from token/position bias without assuming SAE monosemanticity.

### Experiment 2 — Properly powered, frozen SAE specificity test

**Question:** Does a discovery-selected SAE direction exert target-specific control beyond generic residual perturbation?

**Design:** Promote layer-20 context capture to the frozen pilot bank (32 reachable states per risk, eight skins, both mappings), but do not steer until discovery contains at least 10 context-induced action flips. Freeze one layer, feature, orientation, and dose using discovery only. On held-out race clusters, compare the target against zero, reconstruction, ablation, unrelated active features, and at least 32 equal-norm orthogonal random directions. Use the target's percentile in the empirical random-direction null as the specificity statistic.

**Primary endpoint:** target-minus-random-null change in fixed-state Unsafe log-odds.
**Hard gate:** full SAE reconstruction must preserve every action and stay within a preregistered score-error tolerance.
**Multiplicity:** one frozen target endpoint; all other layers/features remain exploratory.
**Impact:** converts the current informative negative pilot into a reviewer-resistant causal test and prevents cherry-picking a direction that merely behaves like generic noise.

### Experiment 3 — Locked cross-model and policy transfer

**Question:** Is the identified representation/intervention phenomenon checkpoint-specific, and does it survive realistic generation?

**Design:** Without reselecting features on evaluation, replicate the same fixed-state factorial and intervention controls on (a) the pinned Qwen checkpoint under free generation and (b) a second open checkpoint with a pinned compatible pretrained SAE. Use opaque codes, exact parser logs, matched states, whole-race splits, and identical admission rules. Report native model-specific features rather than asserting feature-ID correspondence.

**Primary endpoints:** grouped held-out AUC, target percentile among random controls, direct comparable action-flip rate, and parse-failure rate.
**Transfer rule:** a mechanism is replicated only if directionality and specificity pass independently in both policies/models; do not pool feature IDs or decision rows.
**Impact:** closes the two largest external-validity objections: one checkpoint and one constrained decoding policy.

## 7. Recommended paper placement

- **Main text:** use the concise headline and one figure pairing held-out AUC with controlled intervention outcomes. The intellectual contribution is the separation of decodable strategy-associated information from demonstrated behavioral control.
- **Behavior section:** present the mapping-gated context divergence as the mechanism target, explicitly labeled diagnostic and parity-confounded.
- **Appendix:** place the layer-wise surface probes, reconstruction/sparsity diagnostics, all target-control intervals, live payoff plots, and the integrity table.
- **Limitations:** state that the context admission gate failed, mapping was not fully crossed within seed, self-play evaluation had six clusters, reconstruction was not neutral, and native XAI covers one checkpoint.

## 8. Machine-readable sources checked

- `results/impact_upgrade/analysis_manifest.json`
- `results/impact_upgrade/data_quality_audit.json`
- `results/impact_upgrade/data/context_direct_vs_live.csv`
- `results/impact_upgrade/data/context_mapping_interaction.csv`
- `results/impact_upgrade/data/trajectory_divergence_summary.csv`
- `results/open_source/activation_sae/figures/token_position_robustness.csv`
- `results/open_source/activation_sae/context_fast_sae_analysis/summary.json`
- `results/open_source/activation_sae/context_fast_sae_analysis/heldout_probe_summary.csv`
- `results/open_source/activation_sae/context_fast_sae_analysis/layer_promotion_decisions.csv`
- `results/open_source/activation_sae/context_fast_sae_smoke_l12/context_steering_rows.jsonl`
- `results/open_source/activation_sae/context_fast_sae_smoke_l20/context_steering_rows.jsonl`
- `results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/analysis_summary.json`
- `results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/tables/selected_feature_associations.csv`
- `results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/tables/fixed_state_condition_summary.csv`
- `results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/tables/fixed_state_target_control_contrasts.csv`
- `results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/tables/live_condition_summary.csv`
- `results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/tables/live_target_control_contrasts.csv`
