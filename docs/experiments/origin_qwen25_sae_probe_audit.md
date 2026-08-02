# Audit of the original Qwen2.5 SAE probe

Audit target: `origin/main:kaggle/interpretability/qwen25_sae_probe.py` at
commit `776d45fef60ac228c8a5bf09ea6c1058cdaff6ed`.

## What the original runner really does

The race loop constructs `AIRaceGame` objects with `build_games_for_model`,
renders both prompts before applying either response, and sends the two parsed
responses back through `apply_round_responses`. Therefore the engine, rather
than the notebook, owns progress, stage payoff, hidden horizon, private risk,
setback, and final payoff calculations.

The reported Cohen's d values are only feature-action associations. They are
not game simulations, payoff validation, or causal explanations. The steering
stage was intended to add a decoder direction during generation, but it has
not produced admitted GPU evidence in this repository.

## Run and provenance blockers

1. `MODEL_NAME` has no immutable Hugging Face revision. `SAE_RELEASE` is a
   mutable SAELens registry alias and the SAE weight revision/checksum is not
   recorded.
2. The engine is mislabeled as `qwen3-8b-interp-selfplay` even though the code
   loads Qwen2.5-7B-Instruct. This contaminates game IDs and recorded model
   provenance.
3. The documented `andyrdt` artifact is not the project's pinned FAST artifact.
   There is no startup check for an expected model revision, SAE revision,
   training hook, architecture, SAELens version, or weight checksum.
4. Cache validity is decided by file existence/count only. A layer, model,
   prompt, seed, or code change can silently reuse stale `.pt` and JSON files.
5. Race `.pt` files and JSON summaries are written directly rather than
   atomically. There is no append-only ledger, per-shard hash, source hash,
   hardware record, completion manifest, expected-cell gate, or checksum audit.
6. Kaggle setup depends on a separately built wheel/weight dataset whose
   manifest is not sufficient to prove the exact runtime files. On GreenNode,
   the unavailable `/network-volume` path must not be used for code, cache,
   logs, or output.

## Capture and selection problems

1. Capture is the final prompt token, which is a valid pre-decision position,
   but behavior is generated in a separate pass and the exact action-sequence
   log-odds are not recorded. The analysis therefore cannot distinguish a
   strong decision preference from a sampled or formatting-dependent output.
2. All completed decisions are used to rank features. The steering holdout is
   then sampled from the same races and labels. This is feature-selection and
   state leakage, not held-out confirmation.
3. Decisions within a race are treated as independent. There is no whole-race
   split, clustered uncertainty, treatment balance check, or separate
   discovery/evaluation statistic.
4. Ranking only by absolute Cohen's d can fail or yield NaNs when one action is
   absent. It also ignores the continuous Safe-vs-Unsafe preference available
   from sequence log-likelihoods.
5. The notebook records no SAE reconstruction fidelity, sparsity, dead-feature
   rate, raw-residual comparator, label-shuffle control, or duplicate-prefix
   audit.

## Why the original steering is not a sufficient causal test

The hook adds one direction to every sequence position on every generation
forward pass. That changes the prompt, generated prefix, and later tokens at
once, so a flip can arise from broad generation or output-format disruption
rather than the pre-action decision state. The intervention size is an
arbitrary multiple of a unit vector, not calibrated to observed feature scale.

The original design also omits an exact unedited replay gate, SAE
reconstruction control, matched-norm random direction, unrelated active
feature, discovery/evaluation separation, and explicit sign reversal. Parse
failures are counted but divide the nominal holdout denominator, confounding
decision changes with protocol failures.

## Replacement runner and evidence boundary

`kaggle/interpretability/greennode_fast_sae_selfplay.py` is a separate
GreenNode runner. It pins Qwen2.5-7B-Instruct revision
`a09a35458c702b33eeacc393d103063234e8bc28` and FAST SAE revision
`5a7ecabe1401bf4de11a0e6da1f7c36bbb46a464`, validates the FAST metadata and
SAELens 5.3.3, and hashes the SAE weights at runtime.

It first runs actual self-play through `AIRaceGame`. The model chooses between
the two exact admissible response sequences using their complete
teacher-forced likelihoods, including EOS. This constrained policy eliminates
parser-format noise and is explicitly recorded as a different behavioral
policy from free generation. It captures and encodes the final prompt token,
writes atomic whole-race shards containing every engine turn and terminal
payoff, and splits whole races within treatment into discovery and evaluation.
The normalized model-pair probability is `sigmoid(Unsafe log-likelihood - Safe
log-likelihood)`. At temperature zero the enacted decision policy is an argmax,
so its probability is separately recorded as exactly zero or one. The latter
must not be presented as the model's calibrated belief.

Feature selection uses only discovery races. Steering replays evaluation
states and edits only the final prompt position. The primary endpoint is the
change in Unsafe-vs-Safe sequence log-odds. Zero edit, SAE reconstruction,
matched-norm orthogonal random direction, unrelated active feature, and
positive/negative doses are all recorded. An exact baseline replay mismatch is
a hard failure. Fixed-state evaluation also ablates an active target feature by
subtracting its current encoded value times its original SAE decoder row.

The separate `steered_play` stage rebuilds held-out races from the same game
seeds and runs complete target, sign-reversal, matched-random, unrelated, and
zero trajectories through `AIRaceGame`. Thus hidden horizon draws and
fixed-seat setback draws use common random numbers. It records action counts,
unsafe rates, rounds, final payoffs, setbacks, and paired payoff changes. A
decision is called a direct action flip only while its complete prompt still
matches the zero trajectory. Once an action differs, later state and history
are endogenous; later behavior and terminal payoff differences are valid live
trajectory effects but not fixed-state causal effects.

Even a successful run supports only a direct, checkpoint- and layer-specific
intervention result under this constrained action policy. It does not identify
a human-readable concept automatically, prove that one feature is the sole
reason for an action, or establish how an online steered race would evolve
through endogenous feedback.
