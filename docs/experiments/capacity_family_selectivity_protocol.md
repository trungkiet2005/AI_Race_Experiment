# Capacity, family, and strategic-selectivity protocol

Status: frozen protocol; no model output has been admitted under this design.

## Contribution spine

The experiment tests whether model capacity and model family improve three
separate capabilities at the same rate: understanding the game, ignoring
payoff-irrelevant presentation changes, and adapting when incentives really
change. It does not treat a larger checkpoint as automatically more strategic.

This design follows GAMEBoT's use of rule-generated intermediate ground truth
([Lin et al., ACL 2025](https://aclanthology.org/2025.acl-long.378/)) and the
separation of label perturbations from payoff counterfactuals in
[Georgousis et al., GEM 2026](https://aclanthology.org/2026.gem-main.31/).
Repeated-game baselines such as
[Akata et al.](https://www.nature.com/articles/s41562-025-02172-y) already test
option relabeling, payoff units, scripted opponents, and predictions. The new
claim here is therefore a validity decomposition across controlled checkpoint
axes, not simply that several LLMs played a game.

## Two axes, not one mixed leaderboard

| Axis | Frozen checkpoints | Supported interpretation |
|---|---|---|
| Within-family capacity | Qwen2.5 7B, 14B, 32B Instruct | Capacity comparison inside one released series |
| Matched-scale family | Qwen2.5 7B, Gemma-2 9B IT, Mistral-7B Instruct v0.1 | Replication across three tested checkpoint-template stacks |

Qwen7 is shared by both axes. Three sizes do not support a scaling-law claim.
The existing Qwen72 configuration is excluded because it changes weight
quantization and KV-cache precision. Llama-3.1-8B remains a prespecified
replacement for Mistral only if the official gated source becomes accessible;
it must not be substituted with an unpinned mirror after results are seen.

All primary cells use Transformers, native chat templates, BF16 weights and
compute, no quantization, greedy decoding, one GPU, neutral Participant IDs,
and the exact strict opaque-code parser. Native chat templates are hashed and
reported, so the family estimand is explicitly the checkpoint plus its native
template stack. A common-template study would be a separate robustness stratum.

## Staged experiment

### Stage 0: hardware and provenance

Each Kaggle version contains one checkpoint only. Before loading weights it
must verify the expected RTX PRO 6000 name, one GPU, compute capability at
least 8.0, BF16 support, and model-specific VRAM. A failed check writes a
`blocked_hardware` receipt with zero requests and exits normally. It is not a
model result.

The manifest binds model-source version, weight-file sizes, tokenizer and chat
template, source Dataset, Git revision, prompt bank, decoding, and effective
batch size. Qwen32 requires at least 72 GiB free device memory under this
protocol; its official BF16 attachment is about 61 GiB before runtime overhead.

### Stage 1: comprehension admission

For each checkpoint, evaluate 96 unique cells:

- three risk levels: 0.1, 0.6, 0.9;
- two opaque mappings: Safe=P and Safe=Q;
- sixteen engine-scored probes: four each for rule recall, stage payoff, state
  update, and terminal scoring.

The behavioral cell is admitted only if coverage is exact, strict parse is at
least 95%, every domain is at least 80%, and state update plus terminal scoring
are each at least 90%. A failed model remains a reported diagnostic result; it
is not silently replaced or allowed into live gameplay.

### Stage 2: targeted diagnostic gameplay

Only admitted stacks run the three prespecified contexts:

- `abstract_contest`: reference;
- `logistics_contract`: positive-control replication selected from the pilot;
- `technology_race`: low-effect/null-control replication selected from the
  same frozen pilot.

The grid fully crosses context, both mappings, three risks, and 32 independent
CRN repetition streams. For five stacks this is 2,880 races and approximately
51,840 decisions at the nine-round expected horizon. Resampling and clustered
inference use repetition streams; risk and player observations stay inside the
same block.

This 32-stream run is diagnostic. Confirmatory wording requires a separate
96-stream run with fresh streams and a frozen model roster after admission.

The generic `ai_race.runner.run_experiment` does not expand context skins or
opaque mappings and therefore must not execute this configuration. Its 480-race
base expansion is only a schema smoke, not the 2,880-race targeted design. The
configuration fails closed with `genericRunnerCompatible: false`; Stage 1 uses
the dedicated Kaggle or GreenNode admission runner, and Stage 2 requires its
dedicated context/mapping adapter after admission passes.

### Stage 3: invariance versus incentive adaptation

At identical engine-reachable states, compare:

1. the reference prompt;
2. a positive payoff-unit transformation with unchanged preferences;
3. a payoff counterfactual whose engine-computed action-value optimum changes.

Both opaque mappings are crossed. The primary quantities are nuisance flip
rate, oracle-correct adaptation rate, and selectivity, defined as adaptation
minus nuisance flips. The oracle, action-value margin, state bank, and threshold
must be generated and hashed before model inference. This stage is not yet
implemented and stays protocol-only.

## Promotion and stopping rules

- Do not pool BF16 and quantized outputs. If hardware forces AWQ, rerun the
  entire Qwen capacity axis under one AWQ contract and include a Qwen7
  BF16/AWQ bridge as a separate condition.
- Do not continue from 32 to 96 streams based on the observed p-value. The
  independent 96-stream target is a new fixed-N run.
- Report model-level results even when a pooled interaction is available.
- A correct comprehension response is task performance, not proof of an
  internal world model.
- A family result is scoped to the named checkpoint-template stacks.
- A capacity result is scoped to Qwen2.5 7B, 14B, and 32B; never call it a
  universal scaling law.

## Compute priority

1. Qwen7 160-request scaffold smoke to validate the runtime and artifact path.
2. Qwen14 and Qwen32 comprehension-only smokes on verified sufficient VRAM.
3. Gemma9 and Mistral7 comprehension-only smokes.
4. The 96-cell risk-by-mapping admission bank for every surviving stack.
5. Targeted 32-stream gameplay only for admitted stacks.

Persona matrices, brand-role prompts, more temperature sweeps, N=4/N=5, and
additional SAE discovery are explicitly out of scope for this GPU budget.
