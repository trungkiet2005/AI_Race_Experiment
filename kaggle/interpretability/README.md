# Qwen2.5-7B-Instruct SAE feature mining + steering (exploratory)

**Status: exploratory mechanistic-interpretability probe, not part of the
confirmatory AI Race protocol described in [PROJECT.md](../../PROJECT.md).**
It reuses `ai_race.engine` unmodified to generate races (same payoffs, horizon
distribution, and strict `ACTION: SAFE|UNSAFE` parser as every other run in
this project), so races produced here are paper-faithful. What is new and
exploratory is everything layered on top: reading and editing the model's
internal residual-stream activations while it decides.

Every race this script generates is self-play (both seats are the same
hooked model) and tagged `run_phase="pilot"`. **Never pool this data with
confirmatory cross-model AI Race results**, and do not describe an SAE
feature as "the reason" the model chose Unsafe beyond what Stage 3's
steering flip-rate actually shows on this one checkpoint at one layer.

## Why Qwen2.5-7B-Instruct + the `andyrdt` SAE

The other Kaggle notebooks in this project's sibling repos (e.g.
`trungkiet/small-riskgame`, for the separate CRSD/Milinski game) already run
Qwen2.5-Instruct via vLLM and have verified it follows this project's strict
prompt format. This track needs two things vLLM cannot give us: per-token
residual-stream activations, and a pretrained sparse autoencoder to interpret
them.

- **Activations**: vLLM is a batched black-box inference engine; it does not
  expose intermediate activations. This script uses
  [TransformerLens](https://github.com/TransformerLensOrg/TransformerLens)'s
  `HookedTransformer` instead, which is slower (no continuous batching, one
  race at a time) but exposes every intermediate tensor — the same tradeoff
  `kaggle/experiments/baseline.py`'s `ENGINE_PROFILE="transformers"` branch
  already documents for this project's non-interpretability track.
- **Pretrained SAE**: [SAELens](https://github.com/jbloomAus/SAELens)'
  registry (`sae_lens/pretrained_saes.yaml`, release
  `qwen2.5-7b-instruct-andyrdt`, HF repo `andyrdt/saes-qwen2.5-7b-instruct`)
  has SAEs trained **directly on Qwen2.5-7B-Instruct**'s own residual stream,
  at layers 3/7/11/15/19/23/27. Training a bespoke SAE instead would need a
  large, broad-corpus activation dataset (a few hundred short AI Race races
  is not enough for an SAE to converge); reusing this pretrained SAE avoids
  that GPU cost, and — because it was trained on the exact checkpoint this
  script runs, not a related base model — avoids the base-vs-instruct
  transfer approximation an earlier draft of this script would have needed.
  (SAELens also ships "Qwen Scope" SAEs, but only for Qwen3/Qwen3.5 *-Base*
  checkpoints, which would have forced that tradeoff; this project's own
  Qwen2.5 track was found afterward and is strictly better for this use
  case.)

Each SAE layer has a companion [Neuronpedia](https://neuronpedia.org) id
(`qwen2.5-7b-it/{layer}-resid-post-aa`) if a feature's automated or
human-written description is useful context when writing up Stage 2/3
results — treat those descriptions as another model's guess, not ground
truth, same as any other exploratory evidence in this track.

## Offline (competition-gated GPU) vs online (plain notebook) install

`kaggle/kernel-metadata.json` (this project's existing baseline kernel) gets
its GPU via `competition_sources` (`arc-prize-2026-arc-agi-3`), which is
generally paired with `enable_internet: false`. This probe needs sae-lens,
transformer-lens, and the SAE weight files, none of which ship on a stock
Kaggle image. Two ways to satisfy that, both handled automatically by
`qwen25_sae_probe.py`'s Cell 2/Cell 5 (no manual switch needed):

- **Prebuilt (competition-gated GPU, Internet OFF)**: run
  `setup_download_sae_assets.py` once in a separate Internet-ON, no-GPU
  kernel (see that file's docstring), save its `/kaggle/working/sae_probe_assets`
  output as a Kaggle Dataset, and `+ Add Input` that dataset to the probe
  kernel alongside the repo input. Cell 2 auto-detects the wheelhouse
  (`find_wheels_dir`, same pattern the sibling CRSD notebook uses) and the
  SAE asset cache (`manifest.json` with a `sae_release_repo_id` key) under
  `/kaggle/input`, installs from the wheelhouse with `--no-index`, and points
  `HF_HOME`/`HF_HUB_OFFLINE=1` at the pre-populated SAE cache so
  `SAE.from_pretrained` resolves locally. Qwen2.5-7B-Instruct itself comes
  from `MODEL_LOCAL_PATH` (the same Kaggle Model mount
  `kaggle/kernel-metadata.json` already lists under `model_sources`), loaded
  via `transformers.AutoModelForCausalLM` and handed to
  `HookedTransformer.from_pretrained(..., hf_model=...)` so TransformerLens
  never needs to fetch weights itself.
- **Plain notebook (Internet ON, not competition-gated)**: skip the setup
  kernel. With no offline wheelhouse/SAE-cache/model-mount found, Cell 2 pip
  installs directly and Cell 5 falls back to a live HF Hub download for both
  the model and the SAE. Simpler, but likely loses the competition-gated
  `NvidiaRtxPro6000` machine shape in favour of whatever GPU a non-competition
  kernel gets.

## Pipeline and where it stops right now

See the module docstring in
[`qwen25_sae_probe.py`](qwen25_sae_probe.py) for the full per-cell breakdown.
Summary: **Stage 1** (self-play races + activation capture) and **Stage 2**
(SAE feature mining, ranked by Cohen's d between SAFE and UNSAFE decisions)
are implemented end to end. **Stage 3** (causal steering: add each top
feature's decoder direction to the residual stream and see how often the
parsed action flips) is implemented but **has not been run** — nothing in
this repository has executed it on a GPU yet. **Stage 4** (visualisation +
zip for the Kaggle Output tab) reads Stage 2/3's cached JSON and does not
need a GPU.

Every stage after the first checks its cache file on disk and skips
recomputation unless `FORCE_RECOMPUTE = True`, so a Kaggle session that runs
out of GPU-hours mid-pipeline does not lose earlier stages' work.

## Before pushing to Kaggle

This project's Kaggle Benchmark/notebook operations are **checkpointed**:
run one step, look at the output, then decide the next step — see the root
[CLAUDE.md](../../CLAUDE.md) and [kaggle/benchmarks/README.md](../benchmarks/README.md).
Concretely, that means:

1. Smoke-test Stage 1 with `N_RACES` small (e.g. 2) before committing a full
   GPU session to 60 races.
2. `LAYER` must be one of `{3, 7, 11, 15, 19, 23, 27}` — the only layers the
   `qwen2.5-7b-instruct-andyrdt` release covers. The default (15) has not
   been run; confirm the SAE loads and `sae.cfg.d_in == model.cfg.d_model`
   in Cell 5 before trusting it.
3. Only after Stage 1/2 produce a sane feature ranking should Stage 3
   (steering) be run — it re-runs generation `TOP_K_FEATURES_TO_STEER ×
   len(STEERING_ALPHAS) × STEERING_HOLDOUT_N` times and is the most
   GPU-expensive stage per decision.
