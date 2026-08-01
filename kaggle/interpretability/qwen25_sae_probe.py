# %%
"""Qwen2.5-7B-Instruct SAE feature mining + steering on AI Race decisions —
Kaggle GPU notebook.

⚠️ EXPLORATORY, NOT PART OF THE CONFIRMATORY AI RACE PROTOCOL. This is a
mechanistic-interpretability probe layered on top of the paper-faithful engine
(``ai_race.engine`` is used unmodified to build races and score rounds); it does
not change payoffs, horizons, or risk accounting. Every race here is self-play
(both seats are the same hooked Qwen2.5-7B-Instruct) and every race is tagged
``run_phase="pilot"`` — never pool this with confirmatory cross-model AI Race
data, and never claim these SAE features explain "why" the model chose
SAFE/UNSAFE beyond a correlational-then-causal (steering) result on this one
checkpoint. See ``kaggle/interpretability/README.md`` for the full rationale.

Model/SAE choice: SAELens' registry (``sae_lens/pretrained_saes.yaml``,
release ``qwen2.5-7b-instruct-andyrdt``, HF repo ``andyrdt/saes-qwen2.5-7b-instruct``)
has SAEs trained **directly on Qwen2.5-7B-Instruct**'s own residual stream at
layers 3/7/11/15/19/23/27 — the same chat model already used elsewhere in
this project's Kaggle notebooks (e.g. the sibling CRSD ``small-riskgame``
notebook), already known to follow the strict ``ACTION: SAFE|UNSAFE`` output
format reliably. This avoids the base-vs-instruct transfer approximation that
a Qwen3-Base-trained "Qwen Scope" SAE would have required if paired with a
Qwen3 chat model instead (an earlier draft of this script used that path;
switched once this exact-match SAE release was found in the registry).

This track cannot reuse the project's vLLM offline backend
(``ai_race.models.factory`` / ``FAIRGAME.src.llm_connectors``): vLLM is a
black-box batched inference engine and does not expose per-token residual
stream activations. Model access here goes through TransformerLens's
``HookedTransformer`` instead, which is slower (no continuous batching) but
lets us read and edit any intermediate activation — see
``kaggle/experiments/baseline.py``'s ``ENGINE_PROFILE="transformers"`` branch
for the project's existing precedent of a slower, unbatched fallback path.

Pipeline (each stage is independently resumable from the previous stage's
cached output on disk, so a Kaggle session that runs out of GPU-hours does not
lose earlier work — see ``OUTPUT_DIR`` layout below):

  1. RACES     — self-play Qwen2.5-7B-Instruct races via the real ai_race
                 engine; for each decision, teacher-force the prompt to
                 capture the residual stream at
                 ``blocks.{LAYER}.hook_resid_post`` (last prompt token, i.e.
                 the state right before the model commits to an action), then
                 greedily generate the one-line ``ACTION: SAFE|UNSAFE``
                 response and parse it with the project's own strict parser.
                 Cached to ``OUTPUT_DIR/activations/<race_id>.pt``.
  2. FEATURES  — encode every cached residual vector through the pretrained
                 SAE (pure CPU/tensor work, no model needed) and rank SAE
                 features by association with the SAFE/UNSAFE label. Cached to
                 ``OUTPUT_DIR/feature_ranking.json``.
  3. STEERING  — for the top-K ranked features, add/subtract the feature's SAE
                 decoder direction to the residual stream during generation on
                 a held-out sample of decision points, and measure how often
                 the parsed action flips relative to unsteered generation at
                 the same prompt. Cached to ``OUTPUT_DIR/steering_results.json``.
  4. VISUALIZE — matplotlib figures from the two JSON caches above (no model,
                 no GPU) saved as PNGs, then everything is zipped for the
                 Output tab, mirroring the zip step in the other Kaggle
                 notebooks in this project.

Re-running this file with ``FORCE_RECOMPUTE = False`` (the default) skips any
stage whose cache file already exists, so mining/steering/visualising can be
iterated on without repeating the GPU-heavy race generation.

CÁCH CHẠY: GPU ON, Internet ON (pip install sae-lens/transformer-lens and
download Qwen2.5-7B-Instruct + the pretrained SAE weights from Hugging Face). + Add Input:
this repo (contains ``ai_race/`` — no ``FAIRGAME/`` GPU connector needed for
this track). Run cells 1 → 8 in order the first time; on a resumed session,
cells 5 (model/SAE load) and later can be skipped if their outputs are only
needed for a stage whose cache already exists.
"""

# %%
# CELL 1: CẤU HÌNH — sửa ở đây
import os
from pathlib import Path

MODEL_NAME = "Qwen/Qwen2.5-7B-Instruct"
# Local Kaggle Model mount, same convention as kaggle/kernel-metadata.json's
# model_sources ("qwen-lm/qwen2.5/Transformers/7b-instruct/1" -> mounted at
# /kaggle/input/models/<owner>/<model>/<framework>/<variation>/<version>).
# None = fall back to a live HF Hub download of MODEL_NAME (needs Internet ON).
MODEL_LOCAL_PATH = "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
SAE_RELEASE = "qwen2.5-7b-instruct-andyrdt"       # SAE trained directly on this checkpoint
LAYER = 15                                        # must be one of {3,7,11,15,19,23,27} -- the only layers this release covers
SAE_ID = f"resid_post_layer_{LAYER}_trainer_1"
HOOK_NAME = f"blocks.{LAYER}.hook_resid_post"

TREATMENT_GAME = "ai_race_risk_60"                # matches the mid-risk paper treatment
# Smoke-test override so a Kaggle session can start tiny (e.g. AI_RACE_SAE_N_RACES=2)
# before committing GPU-hours to the full N_RACES, without editing/re-pushing code.
N_RACES = int(os.environ.get("AI_RACE_SAE_N_RACES", "2"))  # SMOKE TEST default -- raise to 60 after this checkpoint
BASE_SEED = 20260801                              # independent of any confirmatory run's seed
MAX_NEW_TOKENS = 12                                # response must be exactly "ACTION: SAFE|UNSAFE"
GENERATION_TEMPERATURE = 0.0                       # greedy: this is a mechanistic probe, not a
                                                    # behavioural-distribution estimate

TOP_K_FEATURES_TO_STEER = 8
STEERING_ALPHAS = [-8.0, -4.0, 4.0, 8.0]           # multiples of each feature's decoder-direction unit norm
STEERING_HOLDOUT_N = 40                            # decision points sampled (with replacement across races) for steering

OUTPUT_DIR = Path("/kaggle/working/qwen25_sae_probe")
ACTIVATIONS_DIR = OUTPUT_DIR / "activations"
FEATURE_RANKING_PATH = OUTPUT_DIR / "feature_ranking.json"
STEERING_RESULTS_PATH = OUTPUT_DIR / "steering_results.json"
FIGURES_DIR = OUTPUT_DIR / "figures"

FORCE_RECOMPUTE = False  # True re-runs every stage even if its cache file exists

for _d in (ACTIVATIONS_DIR, FIGURES_DIR):
    _d.mkdir(parents=True, exist_ok=True)

print(
    f"model={MODEL_NAME} sae_release={SAE_RELEASE} sae_id={SAE_ID} layer={LAYER} hook={HOOK_NAME}\n"
    f"treatment={TREATMENT_GAME} n_races={N_RACES} output={OUTPUT_DIR}"
)

# %%
# CELL 2: Cài sae-lens + transformer-lens.
#
# Two paths, auto-detected (see kaggle/interpretability/setup_download_sae_assets.py):
#   - OFFLINE: a Dataset built by that setup script is attached as Kaggle Input.
#     Installs from the local wheelhouse -- no Internet needed, so this can run
#     attached to a competition (competition_sources gate GPU access on this
#     project's machine_shape, and competitions typically require Internet OFF).
#   - ONLINE fallback: no such Dataset found -- pip install directly. Needs
#     Internet ON and therefore (per this project's Kaggle setup) a plain,
#     non-competition kernel rather than the competition-gated GPU shape.
import importlib.util
import subprocess
import sys
from pathlib import Path


def find_wheels_dir(package_names: list[str], root: str = "/kaggle/input", max_depth: int = 6) -> Path | None:
    root_path = Path(root)
    if not root_path.is_dir():
        return None
    patterns = [name.replace("-", "_") + "*.whl" for name in package_names]
    stack = [(root_path, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            if all(any(d.glob(pat)) for pat in patterns):
                return d
        except OSError:
            pass
        if depth < max_depth:
            try:
                for c in sorted(d.iterdir()):
                    if c.is_dir() and not c.name.startswith("."):
                        stack.append((c, depth + 1))
            except OSError:
                pass
    return None


_pip_packages = ["sae-lens", "transformer-lens"]
_need = [
    pkg
    for pkg, mod in zip(_pip_packages, ["sae_lens", "transformer_lens"])
    if importlib.util.find_spec(mod) is None
]
OFFLINE_WHEELS_DIR = find_wheels_dir(_pip_packages)

if not _need:
    print("sae-lens and transformer-lens already present.")
elif OFFLINE_WHEELS_DIR is not None:
    print(f"installing {_need} offline from {OFFLINE_WHEELS_DIR} ...")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--no-index",
         f"--find-links={OFFLINE_WHEELS_DIR}", *_need],
        check=True,
    )
else:
    print(f"no offline wheelhouse found under /kaggle/input -- installing {_need} via Internet")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", *_need], check=True)

# Set by Cell 5 if an offline SAE asset Dataset (from setup_download_sae_assets.py)
# is found; None means Cell 5 falls back to SAE.from_pretrained's live HF download.
OFFLINE_SAE_DIR = None
for _candidate_root in ("/kaggle/input",):
    _p = Path(_candidate_root)
    if _p.is_dir():
        for _sub in _p.rglob("manifest.json"):
            try:
                import json as _json
                _manifest = _json.loads(_sub.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            if "sae_release_repo_id" in _manifest:
                OFFLINE_SAE_DIR = _sub.parent / "sae"
                break
    if OFFLINE_SAE_DIR is not None:
        break
print(f"offline SAE asset dir: {OFFLINE_SAE_DIR or '(none -- will use live HF download)'}")

# %%
# CELL 3: Bootstrap repo (import ai_race) — cùng pattern dò input với các notebook khác
import shutil
import sys as _sys
from pathlib import Path as _Path


def find_repo_input(root: str = "/kaggle/input", max_depth: int = 6) -> _Path | None:
    root_path = _Path(root)
    if not root_path.is_dir():
        return None
    stack = [(root_path, 0)]
    while stack:
        d, depth = stack.pop()
        try:
            if (d / "ai_race").is_dir():
                return d
        except OSError:
            pass
        if depth < max_depth:
            try:
                for c in sorted(d.iterdir()):
                    if c.is_dir() and not c.name.startswith("."):
                        stack.append((c, depth + 1))
            except OSError:
                pass
    return None


WORK_COPY = _Path("/kaggle/working/ai_race_repo")
REPO_INPUT = find_repo_input()
if REPO_INPUT is None:
    raise FileNotFoundError("No '+ Add Input' dataset containing ai_race/ was found under /kaggle/input")

if WORK_COPY.exists():
    shutil.rmtree(WORK_COPY)
shutil.copytree(REPO_INPUT, WORK_COPY, ignore=shutil.ignore_patterns(".git"))
if str(WORK_COPY) not in _sys.path:
    _sys.path.insert(0, str(WORK_COPY))
os.chdir(WORK_COPY)
print(f"repo input: {REPO_INPUT} -> working copy: {WORK_COPY}")

# %%
# CELL 4: Imports that need the repo on sys.path
import json  # noqa: E402

import torch  # noqa: E402
from sae_lens import SAE  # noqa: E402
from transformer_lens import HookedTransformer  # noqa: E402
from transformers import AutoModelForCausalLM, AutoTokenizer  # noqa: E402

from ai_race.dataio.config_loader import validate_experiment  # noqa: E402
from ai_race.engine.round import parse_action  # noqa: E402
from ai_race.engine.state import Action  # noqa: E402
from ai_race.runner.run_experiment import build_games_for_model  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
if DEVICE == "cpu":
    print("WARNING: no CUDA device visible — this notebook expects a Kaggle GPU accelerator.")

# %%
# CELL 5: Nạp model + SAE (bỏ qua nếu chỉ đang chạy lại Cell 7/8 từ cache có sẵn)
_need_model = FORCE_RECOMPUTE or (
    len(list(ACTIVATIONS_DIR.glob("*.pt"))) < N_RACES
) or not FEATURE_RANKING_PATH.exists()

if _need_model:
    if MODEL_LOCAL_PATH and Path(MODEL_LOCAL_PATH).is_dir():
        # Load weights from the Kaggle Model mount (Internet OFF safe), then
        # hand the already-loaded HF model to TransformerLens so it only
        # needs MODEL_NAME for its architecture/weight-conversion mapping,
        # not to fetch weights itself.
        print(f"loading HF checkpoint from local mount: {MODEL_LOCAL_PATH}")
        hf_model = AutoModelForCausalLM.from_pretrained(MODEL_LOCAL_PATH, torch_dtype=torch.bfloat16)
        hf_tokenizer = AutoTokenizer.from_pretrained(MODEL_LOCAL_PATH)
        model = HookedTransformer.from_pretrained(
            MODEL_NAME, hf_model=hf_model, tokenizer=hf_tokenizer, device=DEVICE, dtype="bfloat16"
        )
    else:
        print(f"MODEL_LOCAL_PATH not found -- falling back to a live HF Hub download of {MODEL_NAME}")
        model = HookedTransformer.from_pretrained(MODEL_NAME, device=DEVICE, dtype="bfloat16")
    model.eval()
    print(f"loaded {MODEL_NAME}: n_layers={model.cfg.n_layers} d_model={model.cfg.d_model}")
    if not (0 <= LAYER < model.cfg.n_layers):
        raise ValueError(f"LAYER={LAYER} out of range for n_layers={model.cfg.n_layers}")

    if OFFLINE_SAE_DIR is not None:
        # setup_download_sae_assets.py populates OFFLINE_SAE_DIR as a standard
        # HF hub cache layout; pointing HF_HOME there and forcing offline mode
        # makes SAE.from_pretrained resolve locally with no network call.
        os.environ["HF_HOME"] = str(OFFLINE_SAE_DIR)
        os.environ["HF_HUB_OFFLINE"] = "1"
        print(f"using offline SAE cache: {OFFLINE_SAE_DIR}")
    sae = SAE.from_pretrained(
        release=SAE_RELEASE,
        sae_id=SAE_ID,
        device=DEVICE,
    )
    if hasattr(sae, "cfg") and sae.cfg.d_in != model.cfg.d_model:
        raise ValueError(
            f"SAE d_in={sae.cfg.d_in} does not match model d_model={model.cfg.d_model} "
            "-- wrong layer/checkpoint pairing"
        )
    print(f"loaded SAE {SAE_RELEASE}/{SAE_ID}: d_sae={sae.cfg.d_sae}")
else:
    model = None
    sae = None
    print("skipping model/SAE load: activation cache and feature ranking already complete")

# %%
# CELL 6: Helpers — capture-and-generate, driving one AI Race decision at a time
import numpy as np  # noqa: E402


@torch.no_grad()
def generate_with_capture(prompt: str, seed: int) -> tuple[str, torch.Tensor]:
    """Return (raw_response_text, resid_post[HOOK_NAME] at the last prompt token).

    The activation is captured with one teacher-forced forward pass over the
    prompt (this is the state the model is in immediately before it commits to
    its first output token). Generation itself is a separate, greedy pass —
    the response format is a strict one-line ``ACTION: SAFE|UNSAFE``, so
    sampling temperature is not needed to get a well-formed answer.
    """
    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    chat_text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = model.to_tokens(chat_text)

    _, cache = model.run_with_cache(tokens, names_filter=HOOK_NAME)
    resid = cache[HOOK_NAME][0, -1, :].detach().to("cpu", dtype=torch.float32).clone()

    output_tokens = model.generate(
        tokens,
        max_new_tokens=MAX_NEW_TOKENS,
        do_sample=GENERATION_TEMPERATURE > 0.0,
        temperature=max(GENERATION_TEMPERATURE, 1e-6),
        verbose=False,
    )
    new_tokens = output_tokens[0, tokens.shape[1] :]
    response_text = model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response_text, resid


def steered_generate(prompt: str, seed: int, direction: torch.Tensor, alpha: float) -> str:
    """Like generate_with_capture, but adds alpha * direction to resid at every
    forward pass during generation (prompt included)."""

    def _hook(value, hook):  # noqa: ANN001
        return value + alpha * direction.to(value.dtype).to(value.device)

    torch.manual_seed(seed)
    messages = [{"role": "user", "content": prompt}]
    chat_text = model.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )
    tokens = model.to_tokens(chat_text)
    with model.hooks(fwd_hooks=[(HOOK_NAME, _hook)]):
        output_tokens = model.generate(
            tokens,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            verbose=False,
        )
    new_tokens = output_tokens[0, tokens.shape[1] :]
    return model.tokenizer.decode(new_tokens, skip_special_tokens=True).strip()

# %%
# CELL 7a: STAGE 1 — self-play races, caching one activation record per decision
def run_races() -> None:
    exp = {
        "name": "interp_probe",
        "games": [TREATMENT_GAME],
        "models": ["qwen3-8b-interp-selfplay"],
        "languages": ["en"],
        "repetitions": N_RACES,
        "seed": BASE_SEED,
        "agents": "companies_default",
        "runPhase": "pilot",
        "samplingSeedApplied": True,
    }
    validate_experiment(exp)
    games = build_games_for_model(exp, model="qwen3-8b-interp-selfplay")

    for game in games:
        cache_path = ACTIVATIONS_DIR / f"{game.game_id}.pt"
        if cache_path.exists() and not FORCE_RECOMPUTE:
            continue

        records: list[dict] = []
        while not game.is_finished:
            prompts = game.build_round_prompts()
            responses: list[str] = []
            resids: list[torch.Tensor] = []
            seeds: list[int] = []
            for player_index, prompt in enumerate(prompts):
                seed = game.sampling_seed(player_index, game.current_round)
                text, resid = generate_with_capture(prompt, seed)
                responses.append(text)
                resids.append(resid)
                seeds.append(seed)
            game.apply_round_responses(responses)

            # apply_round_responses appended exactly one TurnRecord per player,
            # in player-index order, for the round just completed.
            for turn, resid, prompt, seed in zip(game.turns[-2:], resids, prompts, seeds):
                records.append(
                    {
                        "race_id": game.game_id,
                        "round": turn.round,
                        "player_index": (
                            0 if turn.player == game.agents[0].name else 1
                        ),
                        "action": turn.action,
                        "parse_failed": turn.parse_failed,
                        "own_prev_action": turn.own_prev_action,
                        "opponent_prev_action": turn.opponent_prev_action,
                        "progress_gap_before": turn.progress_gap_before,
                        "own_private_risk_before": turn.own_private_risk_before,
                        "prompt": prompt,
                        "seed": seed,
                        "resid": resid,
                    }
                )

        # Paper-faithful invariant this project already enforces for confirmatory
        # analysis: one parse_failed decision contaminates the whole race, so it
        # is excluded here too rather than mixed into the feature-mining sample.
        any_parse_failed = any(r["parse_failed"] for r in records)
        torch.save(
            {
                "race_id": game.game_id,
                "any_parse_failed": any_parse_failed,
                "records": records,
            },
            cache_path,
        )
        status = "PARSE_FAILED (excluded downstream)" if any_parse_failed else "ok"
        print(f"[{game.game_id}] {len(records)} decisions cached -> {cache_path} ({status})")


if _need_model:
    run_races()
else:
    print("skipping race generation: activation cache already complete")

# %%
# CELL 7b: STAGE 2 — SAE feature mining (CPU tensor work only, no model needed)
def load_cached_decisions() -> list[dict]:
    decisions = []
    for cache_path in sorted(ACTIVATIONS_DIR.glob("*.pt")):
        payload = torch.load(cache_path, map_location="cpu")
        if payload["any_parse_failed"]:
            continue
        decisions.extend(payload["records"])
    return decisions


def mine_features() -> dict:
    decisions = load_cached_decisions()
    if not decisions:
        raise RuntimeError("No parse-clean decisions cached -- run Stage 1 first")

    resid_matrix = torch.stack([d["resid"] for d in decisions]).to(DEVICE, dtype=torch.float32)
    labels = np.array([1 if d["action"] == Action.UNSAFE.value else 0 for d in decisions])

    with torch.no_grad():
        feats = sae.encode(resid_matrix).to("cpu", dtype=torch.float32).numpy()

    safe_mask = labels == 0
    unsafe_mask = labels == 1
    mean_safe = feats[safe_mask].mean(axis=0)
    mean_unsafe = feats[unsafe_mask].mean(axis=0)
    pooled_std = feats.std(axis=0) + 1e-8
    cohens_d = (mean_unsafe - mean_safe) / pooled_std

    ranking = sorted(
        (
            {
                "feature_index": int(i),
                "cohens_d": float(cohens_d[i]),
                "mean_activation_safe": float(mean_safe[i]),
                "mean_activation_unsafe": float(mean_unsafe[i]),
            }
            for i in range(feats.shape[1])
            if pooled_std[i] > 1e-6  # drop dead/constant features
        ),
        key=lambda row: abs(row["cohens_d"]),
        reverse=True,
    )

    summary = {
        "model": MODEL_NAME,
        "sae_release": SAE_RELEASE,
        "sae_id": SAE_ID,
        "layer": LAYER,
        "hook_name": HOOK_NAME,
        "n_decisions": len(decisions),
        "n_races": len({d["race_id"] for d in decisions}),
        "n_safe": int(safe_mask.sum()),
        "n_unsafe": int(unsafe_mask.sum()),
        "top_features": ranking[:200],
    }
    FEATURE_RANKING_PATH.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(
        f"mined {feats.shape[1]} features over {len(decisions)} decisions "
        f"({summary['n_safe']} safe / {summary['n_unsafe']} unsafe) -> {FEATURE_RANKING_PATH}"
    )
    return summary


if FORCE_RECOMPUTE or not FEATURE_RANKING_PATH.exists():
    feature_summary = mine_features()
else:
    feature_summary = json.loads(FEATURE_RANKING_PATH.read_text(encoding="utf-8"))
    print(f"loaded cached feature ranking from {FEATURE_RANKING_PATH}")

# %%
# CELL 7c: STAGE 3 — steering: does nudging a top feature causally flip the action?
def run_steering() -> dict:
    """For each top feature x alpha, regenerate a held-out sample of decisions
    with the feature's (unit-normalised) SAE decoder direction added to the
    residual stream at every forward pass, and compare the parsed action
    against the ORIGINAL unsteered action recorded during Stage 1 (same
    prompt, same seed -- greedy decoding, so the only difference is the hook).

    This is the causal check: Stage 2's Cohen's d is purely correlational
    (the feature and the action could share a common cause upstream); a high
    flip rate here is evidence the feature's direction is doing causal work
    in the model's decision, not just co-occurring with it.
    """
    decisions = load_cached_decisions()
    rng = np.random.default_rng(BASE_SEED)
    holdout_idx = rng.choice(
        len(decisions), size=min(STEERING_HOLDOUT_N, len(decisions)), replace=False
    )
    holdout = [decisions[i] for i in holdout_idx]

    top_features = feature_summary["top_features"][:TOP_K_FEATURES_TO_STEER]
    results = []
    for feature_row in top_features:
        idx = feature_row["feature_index"]
        direction = sae.W_dec[idx].detach()
        direction = direction / (direction.norm() + 1e-8)

        per_alpha = []
        for alpha in STEERING_ALPHAS:
            n_flipped_to_unsafe = 0
            n_flipped_to_safe = 0
            n_parse_failed = 0
            for decision in holdout:
                steered_text = steered_generate(
                    decision["prompt"], decision["seed"], direction, alpha
                )
                steered_action, steered_parse_failed = parse_action(steered_text)
                if steered_parse_failed:
                    n_parse_failed += 1
                    continue
                original_unsafe = decision["action"] == Action.UNSAFE.value
                steered_unsafe = steered_action is Action.UNSAFE
                if steered_unsafe and not original_unsafe:
                    n_flipped_to_unsafe += 1
                elif not steered_unsafe and original_unsafe:
                    n_flipped_to_safe += 1
            per_alpha.append(
                {
                    "alpha": alpha,
                    "n_holdout": len(holdout),
                    "n_flipped_to_unsafe": n_flipped_to_unsafe,
                    "n_flipped_to_safe": n_flipped_to_safe,
                    "flip_rate": (n_flipped_to_unsafe + n_flipped_to_safe) / len(holdout),
                    "n_parse_failed": n_parse_failed,
                }
            )
        results.append(
            {
                "feature_index": idx,
                "cohens_d": feature_row["cohens_d"],
                "per_alpha": per_alpha,
            }
        )
        print(f"steered feature {idx} (Cohen's d={feature_row['cohens_d']:.2f}): {per_alpha}")

    STEERING_RESULTS_PATH.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"wrote steering results -> {STEERING_RESULTS_PATH}")
    return {"results": results}


RUN_STEERING = os.environ.get("AI_RACE_SAE_RUN_STEERING", "0") != "0"  # SMOKE TEST default off -- re-enable after this checkpoint
if RUN_STEERING and _need_model and (FORCE_RECOMPUTE or not STEERING_RESULTS_PATH.exists()):
    steering_summary = run_steering()
elif STEERING_RESULTS_PATH.exists():
    steering_summary = {"results": json.loads(STEERING_RESULTS_PATH.read_text(encoding="utf-8"))}
    print(f"loaded cached steering results from {STEERING_RESULTS_PATH}")
else:
    steering_summary = None
    print("skipping steering: model not loaded and no cached results -- set _need_model or run Cell 5 manually")

# %%
# CELL 8: VISUALIZE + package (matplotlib only, no GPU needed)
import matplotlib.pyplot as plt  # noqa: E402

summary = json.loads(FEATURE_RANKING_PATH.read_text(encoding="utf-8"))
top20 = summary["top_features"][:20]

fig, ax = plt.subplots(figsize=(8, 6))
ax.barh(
    [str(row["feature_index"]) for row in reversed(top20)],
    [row["cohens_d"] for row in reversed(top20)],
)
ax.set_xlabel("Cohen's d (UNSAFE mean - SAFE mean, SAE feature activation)")
ax.set_ylabel("SAE feature index")
ax.set_title(
    f"Top 20 SAE features by |Cohen's d| — {MODEL_NAME}, layer {LAYER}\n"
    f"n={summary['n_decisions']} decisions ({summary['n_safe']} safe / {summary['n_unsafe']} unsafe)"
)
fig.tight_layout()
fig.savefig(FIGURES_DIR / "top_features_cohens_d.png", dpi=150)
plt.close(fig)
print(f"saved {FIGURES_DIR / 'top_features_cohens_d.png'}")

if STEERING_RESULTS_PATH.exists():
    steering_rows = json.loads(STEERING_RESULTS_PATH.read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(8, 6))
    for row in steering_rows:
        alphas = [a["alpha"] for a in row["per_alpha"]]
        flip_rates = [a["flip_rate"] for a in row["per_alpha"]]
        ax.plot(alphas, flip_rates, marker="o", label=f"feature {row['feature_index']}")
    ax.axhline(0.0, color="grey", linewidth=0.5)
    ax.set_xlabel("steering alpha (multiples of unit decoder-direction norm)")
    ax.set_ylabel("action flip rate vs unsteered generation")
    ax.set_title(f"Causal steering effect — top {len(steering_rows)} features, layer {LAYER}")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "steering_flip_rates.png", dpi=150)
    plt.close(fig)
    print(f"saved {FIGURES_DIR / 'steering_flip_rates.png'}")
else:
    print("no steering results cached yet -- skipping steering figure")

import zipfile  # noqa: E402

zip_path = Path("/kaggle/working/qwen25_sae_probe.zip")
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
    for fp in OUTPUT_DIR.rglob("*"):
        if fp.is_file():
            z.write(fp, fp.relative_to(OUTPUT_DIR.parent))
print(f"{zip_path} ({zip_path.stat().st_size / 1024 / 1024:.2f} MB) — download from Output tab.")
