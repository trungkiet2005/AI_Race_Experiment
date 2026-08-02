# Activation-level SAE protocol

This protocol explains SAFE/UNSAFE propensity from internal residual-stream
activations. It is a different estimand from the repository's earlier
`explain_action_sparse_autoencoder.py`, which learns a sparse representation of
prompt text and engineered game-state columns and is therefore only a surrogate.

## Pinned framework and artifacts

The default preset uses SAELens 5.3.3 and pretrained FAST JumpReLU SAEs for the
exact `Qwen/Qwen2.5-7B-Instruct` revision
`a09a35458c702b33eeacc393d103063234e8bc28`. The SAE repository is pinned to
`Geaming/Qwen2.5-7B-Instruct_SAEs` revision
`5a7ecabe1401bf4de11a0e6da1f7c36bbb46a464`. Available residual-post layers are
4, 12, 18, 20, and 25; each dictionary maps 3,584 residual dimensions to 28,672
JumpReLU features. These values are machine-readable in
`ai_race/xai/presets/qwen25_7b_instruct_fast.json`.

The pipeline uses TransformerLens `from_pretrained_no_processing` because the SAE
configuration was trained on `blocks.{layer}.hook_resid_post` without centered
writing weights. Capturing Hugging Face tensors and assuming coordinate parity is
not accepted silently.

## Identification contract

The primary activation is the last token immediately before the model emits the
parsed SAFE/UNSAFE label. The label itself is excluded. `prompt_last` is a stricter
robustness condition that stops at the assistant generation marker. Decisions are
split by `game_id`, never by turn, because turns within a race are dependent.

The exact model digest that generated the decisions is mandatory. If it differs
from the pinned base model revision, the script refuses to run unless
`--allow-model-provenance-mismatch` is supplied. Such a run is marked
"cross-model exploratory association only" and must not be presented as an
internal explanation of the decision-producing model. This matters especially
for Ollama/GGUF quantizations.

At nonzero temperature, the pre-action residual predicts decision propensity; it
cannot contain the random sampler draw. SAE feature labels are hypotheses, not
proof that a feature is monosemantic or causally used.

## GPU commands

Install the pinned optional environment once on each pod:

```bash
python -m pip install -e '.[activation-xai]'
```

Validate the dataset and race-level split without loading weights:

```bash
python results/scripts/run_activation_sae.py \
  --input-root /path/to/exact-qwen-turns \
  --output-dir /home/jovyan/ai-race/activation-sae/dry-run \
  --dry-run
```

Run disjoint layer lanes on two GPUs. Each process loads one base model; the
shared output directory must **not** be used concurrently, so give each lane its
own directory and merge CSV rows only after both manifests report `complete`.

```bash
# Pod A
python results/scripts/run_activation_sae.py \
  --input-root /path/to/exact-qwen-turns \
  --output-dir /home/jovyan/ai-race/activation-sae/lane-a \
  --layers 4 12 18 \
  --decision-model-digest a09a35458c702b33eeacc393d103063234e8bc28

# Pod B
python results/scripts/run_activation_sae.py \
  --input-root /path/to/exact-qwen-turns \
  --output-dir /home/jovyan/ai-race/activation-sae/lane-b \
  --layers 20 25 \
  --decision-model-digest a09a35458c702b33eeacc393d103063234e8bc28
```

Use `--max-samples 8 --layers 12 --n-label-shuffles 2` for the first GPU smoke
test. A full run should use all decisions and at least 20 label shuffles.

The GreenNode NFS mount was unresponsive during the 2026-08-01 run, so both
lanes deliberately used pod-local `/home/jovyan` storage and were copied back
independently. Do not touch `/network-volume` while its mount is in an
uninterruptible state. The visualization merge script rejects lanes whose
sample IDs, labels, splits, capture hashes, model revisions, SAE revisions, or
decision-model provenance differ.

## Artifact schema

Every output directory is self-contained:

- `manifest.json`: source-file hashes, exact model/SAE revisions, package
  versions, capture semantics, split audit, command, claim scope, and artifact
  hashes.
- `samples.csv`: one row per decision, with provenance keys, group split, prompt
  hashes, token count, and truncation indicator; raw prompts/responses are omitted.
- `sparse_codes_topk.csv`: the strongest positive features per decision.
- `reconstruction_metrics.csv`: normalized MSE, cosine similarity, and mean L0 by
  layer and split.
- `probe_metrics.csv`: grouped-eval SAFE/UNSAFE metrics for the frozen SAE codes.
- `negative_controls.csv`: identical probes after shuffling train labels.
- `feature_action_associations.csv`: train and eval feature associations reported
  separately so discovery cannot masquerade as confirmation.

Required paper checks are: both labels in both splits; no race crosses splits;
exact decision/attribution model match for primary claims; reconstruction quality
reported; real-label probe compared with shuffled-label controls; train-discovered
features checked independently on eval; and any cross-split duplicate prompt
prefixes disclosed.

## Confirmatory and causal extension gates

The current runner is an activation-association pilot. A confirmatory study must
add state-only, prompt TF-IDF, and raw-residual probes alongside the SAE probe;
teacher-forced log probability for the complete `ACTION: SAFE` and
`ACTION: UNSAFE` strings; cluster-bootstrap intervals; and BH-FDR correction.
Feature IDs, layer, token position, sign, and endpoint must be frozen on discovery
data before final evaluation.

Causal language additionally requires decoder-direction intervention with a
predeclared dose grid and zero, encode-decode reconstruction, matched-norm random
direction, unrelated-feature, and sign-reversal controls. Report action log-odds,
choice flips, parse validity, and realized payoff. A stable controlled
dose-response supports only the narrow claim that this representation can shift
choice propensity in this checkpoint and protocol.

Planned figures are SAE fidelity by layer; held-out probe performance against
label-shuffle controls; train-discovered versus eval-confirmed feature effects;
matched surface-prompt feature shifts; token-position trajectories; and controlled
intervention dose-response. UMAP is descriptive supplement only, never mechanism
evidence.
