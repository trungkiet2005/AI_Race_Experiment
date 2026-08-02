"""GreenNode AI Race self-play, FAST-SAE mining, and held-out steering.

This is an exploratory mechanistic-interpretability runner.  Unlike a probe
over old logs, Stage 1 actually drives :class:`ai_race.engine.game.AIRaceGame`
with the pinned Qwen checkpoint.  The model chooses between the two admissible
complete responses by their teacher-forced sequence likelihood.  The strict
engine parser then applies the chosen response, so all state transitions,
payoffs, stopping, and terminal risk remain owned by the canonical engine.

Stages are separated deliberately:

* ``selfplay`` writes one atomic JSON + NPZ shard per completed race;
* ``mine`` selects SAE features on whole discovery races only;
* ``steer`` replays held-out decision states and reports direct log-odds shifts
  under target-feature, sign-reversal, reconstruction, matched-random, and
  unrelated-feature controls.
* ``steered_play`` re-runs held-out whole-race seeds under frozen interventions
  and records behavioral trajectories and terminal engine outcomes.

The runner pins both the base model and the pretrained FAST JumpReLU SAE.  It
refuses to write to the currently unreliable GreenNode NFS mount; use a path
under ``/home/jovyan`` and copy the completed immutable run back afterward.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Callable, Sequence

import numpy as np

# Direct ``python kaggle/interpretability/...py`` execution otherwise puts only
# this file's directory on sys.path.  Resolve the immutable staged repo root
# before importing the canonical engine.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_race.runner.run_experiment import build_games_for_model


MODEL_REPO = "Qwen/Qwen2.5-7B-Instruct"
MODEL_REVISION = "a09a35458c702b33eeacc393d103063234e8bc28"
SAE_REPO = "Geaming/Qwen2.5-7B-Instruct_SAEs"
SAE_REVISION = "5a7ecabe1401bf4de11a0e6da1f7c36bbb46a464"
SAE_LENS_VERSION = "5.3.3"
SUPPORTED_LAYERS = (4, 12, 18, 20, 25)
SAE_ID_TEMPLATE = "FAST/blocks_{layer}_hook_resid_post_8X_2048_jumprelu"
SCHEMA_VERSION = "ai-race.greennode-fast-sae-selfplay.v1"
ACTION_COMPLETIONS = {"safe": "ACTION: SAFE", "unsafe": "ACTION: UNSAFE"}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    return _sha256_bytes(
        json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    )


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _safe_slug(value: str) -> str:
    return "".join(c if c.isalnum() or c in "._-" else "-" for c in value).strip("-")


def _seeded_unit_random(width: int, seed: int, target: Any) -> Any:
    """Create a deterministic matched-norm random control orthogonal to target."""
    import torch

    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    direction = torch.randn(width, generator=generator, dtype=torch.float32)
    target_cpu = target.detach().float().cpu()
    target_cpu = target_cpu / target_cpu.norm().clamp_min(1e-12)
    direction = direction - torch.dot(direction, target_cpu) * target_cpu
    return direction / direction.norm().clamp_min(1e-12)


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(float(value)) else None


def choose_from_log_odds(log_odds: float, *, temperature: float, seed: int) -> tuple[str, float]:
    """Return action and decision-policy probability, not the model pair probability.

    At temperature zero the policy is deterministic, so this probability is
    exactly zero or one.  The model's normalized probability within the two
    candidate sequences is always ``sigmoid(raw_log_odds)`` and is recorded
    separately by the callers.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    if temperature == 0:
        probability_unsafe = float(log_odds > 0.0)
        return ("unsafe" if log_odds > 0.0 else "safe"), probability_unsafe
    probability_unsafe = _sigmoid(log_odds / temperature)
    return (
        "unsafe" if random.Random(int(seed)).random() < probability_unsafe else "safe"
    ), probability_unsafe


def grouped_race_split(
    records: Sequence[dict[str, Any]], *, eval_fraction: float, seed: int
) -> dict[str, str]:
    """Split whole races, stratified by treatment, without decision leakage."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    strata: dict[str, set[str]] = {}
    for row in records:
        treatment = str(row["treatment"])
        strata.setdefault(treatment, set()).add(str(row["game_id"]))
    assignment: dict[str, str] = {}
    for treatment, race_ids_set in sorted(strata.items()):
        race_ids = sorted(
            race_ids_set,
            key=lambda race_id: _sha256_bytes(f"{seed}|{treatment}|{race_id}".encode()),
        )
        if len(race_ids) < 2:
            n_eval = 0
        else:
            n_eval = min(len(race_ids) - 1, max(1, round(len(race_ids) * eval_fraction)))
        eval_ids = set(race_ids[:n_eval])
        assignment.update({race_id: ("eval" if race_id in eval_ids else "discovery") for race_id in race_ids})
    return assignment


def _correlation_columns(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = matrix.astype(np.float64, copy=False)
    y = target.astype(np.float64, copy=False)
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean()
    numerator = np.sum(x_centered * y_centered[:, None], axis=0)
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _standardized_action_difference(matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    safe = matrix[labels == 0]
    unsafe = matrix[labels == 1]
    if not len(safe) or not len(unsafe):
        return np.full(matrix.shape[1], np.nan, dtype=np.float64)
    pooled = matrix.std(axis=0, dtype=np.float64)
    difference = unsafe.mean(axis=0) - safe.mean(axis=0)
    return np.divide(difference, pooled, out=np.zeros_like(pooled), where=pooled > 0)


def rank_features(
    codes: np.ndarray,
    records: Sequence[dict[str, Any]],
    split: dict[str, str],
    *,
    min_prevalence: float,
) -> list[dict[str, Any]]:
    """Discover on train races and report confirmation statistics separately."""
    discovery = np.asarray([split[str(row["game_id"])] == "discovery" for row in records])
    evaluation = ~discovery
    if not discovery.any() or not evaluation.any():
        raise ValueError("feature mining requires at least one discovery and one eval race")
    log_odds = np.asarray([float(row["unsafe_log_odds"]) for row in records])
    labels = np.asarray([int(row["action"] == "unsafe") for row in records])
    discovery_codes = codes[discovery]
    eval_codes = codes[evaluation]
    discovery_corr = _correlation_columns(discovery_codes, log_odds[discovery])
    eval_corr = _correlation_columns(eval_codes, log_odds[evaluation])
    discovery_d = _standardized_action_difference(discovery_codes, labels[discovery])
    eval_d = _standardized_action_difference(eval_codes, labels[evaluation])
    prevalence = np.mean(discovery_codes > 0, axis=0)
    scales = np.std(discovery_codes, axis=0, dtype=np.float64)
    eligible = np.flatnonzero((prevalence >= min_prevalence) & (scales > 0))
    order = eligible[np.argsort(np.abs(discovery_corr[eligible]))[::-1]]
    return [
        {
            "feature_id": int(feature),
            "discovery_corr_log_odds": float(discovery_corr[feature]),
            "eval_corr_log_odds": float(eval_corr[feature]),
            "discovery_action_cohens_d": _finite_or_none(discovery_d[feature]),
            "eval_action_cohens_d": _finite_or_none(eval_d[feature]),
            "discovery_prevalence": float(prevalence[feature]),
            "discovery_scale": float(scales[feature]),
        }
        for feature in order
    ]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--stage", choices=("all", "selfplay", "mine", "steer", "steered_play"), default="all"
    )
    parser.add_argument("--layer", type=int, required=True, choices=SUPPORTED_LAYERS)
    parser.add_argument(
        "--treatments", nargs="+", default=["ai_race_risk_10", "ai_race_risk_60", "ai_race_risk_90"]
    )
    parser.add_argument("--repetitions", type=int, default=6)
    parser.add_argument("--base-seed", type=int, default=20260801)
    parser.add_argument("--decision-temperature", type=float, default=0.0)
    parser.add_argument("--eval-fraction", type=float, default=0.34)
    parser.add_argument("--top-features", type=int, default=3)
    parser.add_argument("--ranking-limit", type=int, default=500)
    parser.add_argument("--min-feature-prevalence", type=float, default=0.05)
    parser.add_argument("--steering-max-decisions", type=int, default=24)
    parser.add_argument("--steering-alphas", nargs="+", type=float, default=[-2.0, -1.0, 1.0, 2.0])
    parser.add_argument("--live-alpha", type=float, default=2.0)
    parser.add_argument("--steered-play-max-races-per-treatment", type=int, default=2)
    parser.add_argument("--score-replay-tolerance", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--prompt-variant", default="canonical")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args(argv)


def _run_config(args: argparse.Namespace) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "model_repo": MODEL_REPO,
        "model_revision": MODEL_REVISION,
        "sae_repo": SAE_REPO,
        "sae_revision": SAE_REVISION,
        "sae_lens_version": SAE_LENS_VERSION,
        "sae_id": SAE_ID_TEMPLATE.format(layer=args.layer),
        "layer": args.layer,
        "hook_name": f"blocks.{args.layer}.hook_resid_post",
        "capture_position": "final prompt token before any action-label token",
        "action_policy": "teacher-forced exact-sequence likelihood over ACTION: SAFE vs ACTION: UNSAFE, including EOS",
        "treatments": list(args.treatments),
        "repetitions": args.repetitions,
        "base_seed": args.base_seed,
        "decision_temperature": args.decision_temperature,
        "prompt_variant": args.prompt_variant,
        "eval_fraction": args.eval_fraction,
        "top_features": args.top_features,
        "ranking_limit": args.ranking_limit,
        "min_feature_prevalence": args.min_feature_prevalence,
        "steering_max_decisions": args.steering_max_decisions,
        "steering_alphas": list(args.steering_alphas),
        "live_alpha": args.live_alpha,
        "steered_play_max_races_per_treatment": args.steered_play_max_races_per_treatment,
        "score_replay_tolerance": args.score_replay_tolerance,
        "dtype": args.dtype,
        "device": args.device,
    }


def _guard_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if str(resolved).replace("\\", "/").startswith("/network-volume"):
        raise ValueError("GreenNode /network-volume is unavailable; use /home/jovyan and SCP results back")
    return resolved


def _prepare_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output = _guard_output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _run_config(args)
    script_path = Path(__file__).resolve()
    runner_sha256 = _sha256_file(script_path)
    fingerprint = _canonical_sha256({"config": config, "runner_sha256": runner_sha256})
    manifest_path = output / "manifest.json"
    if manifest_path.exists():
        prior = json.loads(manifest_path.read_text(encoding="utf-8"))
        if prior.get("config_fingerprint") != fingerprint:
            raise ValueError("output directory belongs to a different immutable run configuration")
        if not args.resume and args.stage in {"all", "selfplay"}:
            raise FileExistsError("run exists; pass --resume only for the identical configuration")
        return manifest_path, prior
    engine_files = {
        relative: _sha256_file(REPO_ROOT / relative)
        for relative in (
            "ai_race/engine/game.py",
            "ai_race/engine/scoring.py",
            "ai_race/engine/round.py",
            "ai_race/runner/run_experiment.py",
        )
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": _utc_now(),
        "config": config,
        "config_fingerprint": fingerprint,
        "git_revision": _git_revision(),
        "runner_sha256": runner_sha256,
        "engine_source_sha256": engine_files,
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformers": _package_version("transformers"),
            "transformer_lens": _package_version("transformer-lens"),
            "sae_lens": _package_version("sae-lens"),
        },
        "evidence_class": "exploratory-pilot",
        "claim_boundary": "checkpoint-, layer-, policy-, and game-context-specific direct interventions",
        "stages": {},
    }
    _atomic_json(manifest_path, manifest)
    return manifest_path, manifest


def _load_model_and_sae(args: argparse.Namespace, manifest: dict[str, Any]) -> tuple[Any, Any, dict[str, Any]]:
    import torch
    from huggingface_hub import snapshot_download
    from sae_lens import SAE
    from transformer_lens import HookedTransformer

    if _package_version("sae-lens") != SAE_LENS_VERSION:
        raise RuntimeError(
            f"install exact SAE Lens version: pip install 'sae-lens=={SAE_LENS_VERSION}'; "
            f"found {_package_version('sae-lens')!r}"
        )
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    cache_dir = str(args.cache_dir) if args.cache_dir else None
    model_snapshot = snapshot_download(
        MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    sae_id = SAE_ID_TEMPLATE.format(layer=args.layer)
    sae_snapshot = snapshot_download(
        SAE_REPO,
        revision=SAE_REVISION,
        allow_patterns=[f"{sae_id}/*"],
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
    )
    sae_dir = Path(sae_snapshot) / sae_id
    cfg_path = sae_dir / "cfg.json"
    weights_path = sae_dir / "sae_weights.safetensors"
    if not cfg_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"incomplete pinned SAE snapshot: {sae_dir}")
    sae_cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    expected_hook = f"blocks.{args.layer}.hook_resid_post"
    required = {
        "model_name": MODEL_REPO,
        "hook_name": expected_hook,
        "hook_layer": args.layer,
        "architecture": "jumprelu",
        "d_in": 3584,
        "d_sae": 28672,
        "sae_lens_version": SAE_LENS_VERSION,
        "prepend_bos": False,
    }
    mismatches = {key: (sae_cfg.get(key), value) for key, value in required.items() if sae_cfg.get(key) != value}
    if mismatches:
        raise ValueError(f"pinned SAE metadata mismatch: {mismatches}")

    dtype = getattr(torch, args.dtype)
    model = HookedTransformer.from_pretrained_no_processing(
        MODEL_REPO,
        revision=MODEL_REVISION,
        cache_dir=cache_dir,
        local_files_only=args.local_files_only,
        device=args.device,
        dtype=dtype,
        trust_remote_code=False,
    )
    model.eval()
    sae = SAE.load_from_pretrained(str(sae_dir), device=args.device)
    sae.eval()
    if int(model.cfg.d_model) != int(sae.cfg.d_in):
        raise ValueError(f"model d_model={model.cfg.d_model} != SAE d_in={sae.cfg.d_in}")
    artifact = {
        "model_snapshot": str(Path(model_snapshot).resolve()),
        "sae_snapshot": str(Path(sae_snapshot).resolve()),
        "sae_cfg_sha256": _sha256_file(cfg_path),
        "sae_weights_sha256": _sha256_file(weights_path),
    }
    if torch.cuda.is_available():
        device_index = torch.cuda.current_device()
        artifact["cuda_device"] = torch.cuda.get_device_name(device_index)
        artifact["cuda_capability"] = list(torch.cuda.get_device_capability(device_index))
    manifest["resolved_artifacts"] = artifact
    return model, sae, artifact


def _render_chat(tokenizer: Any, prompt: str) -> str:
    messages = [{"role": "user", "content": str(prompt)}]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
    except TypeError:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def _completion_tokens(model: Any, completion: str) -> Any:
    import torch

    ids = model.tokenizer.encode(completion, add_special_tokens=False)
    if model.tokenizer.eos_token_id is not None:
        ids = [*ids, int(model.tokenizer.eos_token_id)]
    return torch.tensor(ids, device=model.cfg.device, dtype=torch.long).unsqueeze(0)


def _score_completion(
    model: Any,
    prefix_tokens: Any,
    completion_tokens: Any,
    *,
    hook_name: str,
    edit: Callable[[Any, int], Any] | None = None,
    capture: bool = False,
) -> tuple[float, Any | None]:
    import torch

    tokens = torch.cat([prefix_tokens, completion_tokens], dim=-1)
    prefix_index = int(prefix_tokens.shape[-1] - 1)

    def edit_hook(value: Any, hook: Any) -> Any:  # noqa: ARG001
        return edit(value, prefix_index) if edit is not None else value

    with torch.inference_mode():
        if capture:
            if edit is not None:
                with model.hooks(fwd_hooks=[(hook_name, edit_hook)]):
                    logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook_name)
            else:
                logits, cache = model.run_with_cache(tokens, names_filter=lambda name: name == hook_name)
            activation = cache[hook_name][0, prefix_index, :].detach()
        else:
            hooks = [(hook_name, edit_hook)] if edit is not None else []
            with model.hooks(fwd_hooks=hooks):
                logits = model(tokens)
            activation = None
        prediction_logits = logits[0, prefix_index : tokens.shape[-1] - 1, :]
        log_probs = torch.log_softmax(prediction_logits.float(), dim=-1)
        selected = log_probs.gather(1, completion_tokens[0].unsqueeze(1)).sum()
    return float(selected.cpu()), activation


def score_action_pair(
    model: Any,
    prompt: str,
    *,
    hook_name: str,
    edit_factory: Callable[[int], Callable[[Any, int], Any]] | None = None,
    capture: bool = False,
) -> tuple[dict[str, float], Any | None]:
    rendered = _render_chat(model.tokenizer, prompt)
    prefix = model.to_tokens(rendered, prepend_bos=False)
    if int(prefix.shape[-1]) >= 2048:
        raise ValueError(f"prompt has {prefix.shape[-1]} tokens; FAST SAE context limit is 2048")
    scores: dict[str, float] = {}
    captured = None
    for index, (action, completion) in enumerate(ACTION_COMPLETIONS.items()):
        completion_ids = _completion_tokens(model, completion)
        edit = edit_factory(int(prefix.shape[-1] - 1)) if edit_factory is not None else None
        score, activation = _score_completion(
            model,
            prefix,
            completion_ids,
            hook_name=hook_name,
            edit=edit,
            capture=capture and index == 0,
        )
        scores[action] = score
        if activation is not None:
            captured = activation
    return scores, captured


def _selfplay(args: argparse.Namespace, model: Any, sae: Any, manifest_path: Path, manifest: dict[str, Any]) -> None:
    import torch

    output = manifest_path.parent
    shard_dir = output / "race_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    model_label = f"{MODEL_REPO}@{MODEL_REVISION[:12]}+constrained-sequence"
    experiment = {
        "name": "greennode_fast_sae_selfplay",
        "games": list(args.treatments),
        "models": [model_label],
        "languages": ["en"],
        "repetitions": int(args.repetitions),
        "seed": int(args.base_seed),
        "agents": "companies_default",
        "runPhase": "pilot",
        "samplingSeedApplied": True,
        "promptVariant": args.prompt_variant,
    }
    games = build_games_for_model(experiment, model=model_label)
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    completed: list[dict[str, Any]] = []
    for game_index, game in enumerate(games, start=1):
        slug = _safe_slug(game.game_id)
        json_path = shard_dir / f"{slug}.json"
        npz_path = shard_dir / f"{slug}.npz"
        if json_path.exists() and npz_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("config_fingerprint") != manifest["config_fingerprint"]:
                raise ValueError(f"stale race shard has different config: {json_path}")
            completed.append(
                {"game_id": game.game_id, "json_sha256": _sha256_file(json_path), "npz_sha256": _sha256_file(npz_path)}
            )
            print(f"[{game_index}/{len(games)}] resume {game.game_id}", flush=True)
            continue
        records: list[dict[str, Any]] = []
        codes: list[np.ndarray] = []
        while not game.is_finished:
            prompts = game.build_round_prompts()
            responses: list[dict[str, Any]] = []
            round_xai: list[dict[str, Any]] = []
            for player_index, prompt in enumerate(prompts):
                scores, activation = score_action_pair(model, prompt, hook_name=hook_name, capture=True)
                if activation is None:
                    raise RuntimeError("activation capture failed")
                unsafe_log_odds = scores["unsafe"] - scores["safe"]
                seed = game.sampling_seed(player_index, game.current_round)
                action, probability_unsafe = choose_from_log_odds(
                    unsafe_log_odds, temperature=args.decision_temperature, seed=seed
                )
                with torch.inference_mode():
                    code = sae.encode(activation.unsqueeze(0)).squeeze(0)
                    reconstruction = sae.decode(code.unsqueeze(0)).squeeze(0)
                residual = activation.detach().float()
                reconstructed = reconstruction.detach().float()
                normalized_mse = float(
                    torch.square(residual - reconstructed).sum()
                    / torch.square(residual).sum().clamp_min(1e-12)
                )
                cosine = float(torch.nn.functional.cosine_similarity(residual, reconstructed, dim=0))
                codes.append(code.detach().float().cpu().numpy())
                response = {
                    "text": ACTION_COMPLETIONS[action],
                    "cumulative_logprob": scores[action],
                    "top_alternatives": [
                        {"text": ACTION_COMPLETIONS[name], "cumulative_logprob": value}
                        for name, value in scores.items()
                    ],
                }
                responses.append(response)
                round_xai.append(
                    {
                        "game_id": game.game_id,
                        "treatment": game.config.name,
                        "max_private_risk": game.config.max_private_risk,
                        "round": game.current_round,
                        "player_index": player_index,
                        "sampling_seed": seed,
                        "prompt": prompt,
                        "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                        "rendered_chat_sha256": _sha256_bytes(_render_chat(model.tokenizer, prompt).encode("utf-8")),
                        "action": action,
                        "safe_sequence_logprob": scores["safe"],
                        "unsafe_sequence_logprob": scores["unsafe"],
                        "unsafe_log_odds": unsafe_log_odds,
                        "model_pair_probability_unsafe": _sigmoid(unsafe_log_odds),
                        "decision_policy_probability_unsafe": probability_unsafe,
                        "sae_l0": int(torch.count_nonzero(code).cpu()),
                        "sae_normalized_mse": normalized_mse,
                        "sae_cosine_similarity": cosine,
                    }
                )
            game.apply_round_responses(responses, prompts=prompts)
            for xai_row, turn in zip(round_xai, game.turns[-2:]):
                xai_row["engine_turn"] = turn.to_dict()
                records.append(xai_row)
        if game.result is None:
            raise RuntimeError("finished game lacks terminal result")
        np_codes = np.stack(codes).astype(np.float32)
        _atomic_npz(npz_path, sae_codes=np_codes)
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": manifest["config_fingerprint"],
            "game_id": game.game_id,
            "records": records,
            "engine_result": game.result.to_dict(),
            "npz_file": npz_path.name,
            "npz_sha256": _sha256_file(npz_path),
        }
        _atomic_json(json_path, payload)
        completed.append(
            {"game_id": game.game_id, "json_sha256": _sha256_file(json_path), "npz_sha256": _sha256_file(npz_path)}
        )
        manifest["stages"]["selfplay"] = {
            "status": "running",
            "n_expected_races": len(games),
            "n_completed_races": len(completed),
            "updated_at_utc": _utc_now(),
        }
        _atomic_json(manifest_path, manifest)
        print(f"[{game_index}/{len(games)}] complete {game.game_id}: {len(records)} decisions", flush=True)
    manifest["stages"]["selfplay"] = {
        "status": "complete",
        "n_expected_races": len(games),
        "n_completed_races": len(completed),
        "shards": completed,
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)


def _load_shards(output: Path, fingerprint: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    codes: list[np.ndarray] = []
    json_paths = sorted((output / "race_shards").glob("*.json"))
    if not json_paths:
        raise FileNotFoundError("no self-play race shards; run --stage selfplay first")
    for json_path in json_paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("config_fingerprint") != fingerprint:
            raise ValueError(f"mixed run fingerprint in {json_path}")
        npz_path = json_path.with_suffix(".npz")
        if _sha256_file(npz_path) != payload["npz_sha256"]:
            raise ValueError(f"NPZ checksum mismatch for {npz_path}")
        with np.load(npz_path) as arrays:
            race_codes = np.asarray(arrays["sae_codes"], dtype=np.float32)
        if race_codes.shape[0] != len(payload["records"]):
            raise ValueError(f"record/code row mismatch in {json_path}")
        records.extend(payload["records"])
        codes.append(race_codes)
    return records, np.concatenate(codes, axis=0)


def _mine(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    output = manifest_path.parent
    records, codes = _load_shards(output, manifest["config_fingerprint"])
    split = grouped_race_split(records, eval_fraction=args.eval_fraction, seed=args.base_seed)
    ranking = rank_features(codes, records, split, min_prevalence=args.min_feature_prevalence)
    if len(ranking) < args.top_features + 1:
        raise ValueError("not enough active variable SAE features for target and unrelated controls")
    selected = ranking[: args.top_features]
    eligible_control = ranking[args.top_features :]
    unrelated = min(eligible_control, key=lambda row: abs(row["discovery_corr_log_odds"]))
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": manifest["config_fingerprint"],
        "split_unit": "whole race, stratified by treatment",
        "n_decisions": len(records),
        "n_races": len(split),
        "n_discovery_races": sum(value == "discovery" for value in split.values()),
        "n_eval_races": sum(value == "eval" for value in split.values()),
        "race_split": split,
        "selection_target": "correlation with continuous Unsafe-vs-Safe sequence log-odds",
        "selected_features": selected,
        "unrelated_control_feature": unrelated,
        "ranking": ranking[: args.ranking_limit],
        "limitations": [
            "Feature association is predictive, not causal.",
            "Causal claims require the separate held-out steering stage and its controls.",
            "Feature names or semantic interpretations are not inferred automatically.",
        ],
    }
    ranking_path = output / "feature_mining.json"
    _atomic_json(ranking_path, summary)
    manifest["stages"]["mine"] = {
        "status": "complete",
        "artifact": ranking_path.name,
        "sha256": _sha256_file(ranking_path),
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)
    print(f"mined {len(ranking)} eligible features; selected {[row['feature_id'] for row in selected]}", flush=True)
    return summary


def _edit_add(delta: Any) -> Callable[[Any, int], Any]:
    def edit(value: Any, prefix_index: int) -> Any:
        changed = value.clone()
        changed[:, prefix_index, :] = changed[:, prefix_index, :] + delta.to(value.device, value.dtype)
        return changed

    return edit


def _edit_replace(replacement: Any) -> Callable[[Any, int], Any]:
    def edit(value: Any, prefix_index: int) -> Any:
        changed = value.clone()
        changed[:, prefix_index, :] = replacement.to(value.device, value.dtype)
        return changed

    return edit


def feature_ablation_delta(feature_activation: float, decoder_row: Any) -> Any:
    """Remove exactly one SAE reconstruction term from the residual stream."""
    return -float(feature_activation) * decoder_row


def _steer(
    args: argparse.Namespace,
    model: Any,
    sae: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    feature_summary: dict[str, Any],
) -> None:
    import torch

    output = manifest_path.parent
    records, _ = _load_shards(output, manifest["config_fingerprint"])
    split = feature_summary["race_split"]
    eval_rows = [row for row in records if split[str(row["game_id"])] == "eval"]
    eval_rows.sort(key=lambda row: _sha256_bytes(f"{args.base_seed}|{row['game_id']}|{row['round']}|{row['player_index']}".encode()))
    eval_rows = eval_rows[: args.steering_max_decisions]
    if not eval_rows:
        raise ValueError("no held-out decisions available for steering")
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    unrelated_id = int(feature_summary["unrelated_control_feature"]["feature_id"])
    unrelated_direction = sae.W_dec[unrelated_id].detach().float()
    unrelated_direction = unrelated_direction / unrelated_direction.norm().clamp_min(1e-12)
    output_rows: list[dict[str, Any]] = []
    max_replay_error = 0.0
    for sample_index, row in enumerate(eval_rows):
        baseline_scores, activation = score_action_pair(model, row["prompt"], hook_name=hook_name, capture=True)
        if activation is None:
            raise RuntimeError("steering baseline activation capture failed")
        baseline_log_odds = baseline_scores["unsafe"] - baseline_scores["safe"]
        replay_error = abs(baseline_log_odds - float(row["unsafe_log_odds"]))
        max_replay_error = max(max_replay_error, replay_error)
        if replay_error > args.score_replay_tolerance:
            raise RuntimeError(
                f"baseline replay mismatch {replay_error:.6g} exceeds tolerance; model/runtime is not identical"
            )
        with torch.inference_mode():
            code = sae.encode(activation.unsqueeze(0)).squeeze(0)
            reconstruction = sae.decode(code.unsqueeze(0)).squeeze(0)

        def add_result(
            condition: str,
            target_feature_id: int | None,
            direction_feature_id: int | None,
            alpha: float | None,
            residual_dose: float,
            scores: dict[str, float],
        ) -> None:
            steered_log_odds = scores["unsafe"] - scores["safe"]
            output_rows.append(
                {
                    "sample_index": sample_index,
                    "game_id": row["game_id"],
                    "round": row["round"],
                    "player_index": row["player_index"],
                    "condition": condition,
                    "target_feature_id": target_feature_id,
                    "direction_feature_id": direction_feature_id,
                    "alpha": alpha,
                    "residual_dose": residual_dose,
                    "baseline_unsafe_log_odds": baseline_log_odds,
                    "steered_unsafe_log_odds": steered_log_odds,
                    "delta_unsafe_log_odds": steered_log_odds - baseline_log_odds,
                    "baseline_action": "unsafe" if baseline_log_odds > 0 else "safe",
                    "steered_action": "unsafe" if steered_log_odds > 0 else "safe",
                    "action_flipped": int((baseline_log_odds > 0) != (steered_log_odds > 0)),
                    "safe_sequence_logprob": scores["safe"],
                    "unsafe_sequence_logprob": scores["unsafe"],
                }
            )

        add_result("zero", None, None, 0.0, 0.0, baseline_scores)
        reconstruction_scores, _ = score_action_pair(
            model,
            row["prompt"],
            hook_name=hook_name,
            edit_factory=lambda _index: _edit_replace(reconstruction),
        )
        add_result("sae_reconstruction", None, None, 0.0, 0.0, reconstruction_scores)
        for feature_row in feature_summary["selected_features"]:
            feature_id = int(feature_row["feature_id"])
            scale = max(float(feature_row["discovery_scale"]), 1e-8)
            target_decoder = sae.W_dec[feature_id].detach().float()
            target_direction = target_decoder / target_decoder.norm().clamp_min(1e-12)
            feature_activation = float(code[feature_id].detach().float().cpu())
            ablation_delta = feature_ablation_delta(feature_activation, target_decoder)
            ablated_scores, _ = score_action_pair(
                model,
                row["prompt"],
                hook_name=hook_name,
                edit_factory=lambda _index, delta=ablation_delta: _edit_add(delta),
            )
            add_result(
                "target_feature_ablation",
                feature_id,
                feature_id,
                None,
                float(ablation_delta.norm().cpu()),
                ablated_scores,
            )
            output_rows[-1]["ablated_feature_activation"] = feature_activation
            output_rows[-1]["decoder_row_norm"] = float(target_decoder.norm().cpu())
            random_direction = _seeded_unit_random(
                int(target_direction.numel()), args.base_seed + feature_id, target_direction
            ).to(target_direction.device)
            for alpha in args.steering_alphas:
                dose = float(alpha) * scale
                for condition, direction, direction_feature in (
                    ("target_feature", target_direction, feature_id),
                    ("matched_random", random_direction, None),
                    ("unrelated_feature", unrelated_direction, unrelated_id),
                ):
                    scores, _ = score_action_pair(
                        model,
                        row["prompt"],
                        hook_name=hook_name,
                        edit_factory=lambda _index, delta=dose * direction: _edit_add(delta),
                    )
                    add_result(
                        condition,
                        feature_id,
                        direction_feature,
                        float(alpha),
                        dose,
                        scores,
                    )
        print(f"steered {sample_index + 1}/{len(eval_rows)} held-out decisions", flush=True)

    jsonl_path = output / "steering_rows.jsonl"
    temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in output_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, jsonl_path)
    conditions = sorted({row["condition"] for row in output_rows})
    summaries = []
    for condition in conditions:
        target_features = sorted(
            {row["target_feature_id"] for row in output_rows if row["condition"] == condition},
            key=lambda value: (-1 if value is None else int(value)),
        )
        for target_feature_id in target_features:
            alphas = sorted(
                {
                    row["alpha"]
                    for row in output_rows
                    if row["condition"] == condition
                    and row["target_feature_id"] == target_feature_id
                },
                key=lambda value: (value is None, 0.0 if value is None else value),
            )
            for alpha in alphas:
                subset = [
                    row for row in output_rows
                    if row["condition"] == condition
                    and row["target_feature_id"] == target_feature_id
                    and (
                        (row["alpha"] is None and alpha is None) or row["alpha"] == alpha
                    )
                ]
                deltas = np.asarray([row["delta_unsafe_log_odds"] for row in subset])
                summaries.append(
                    {
                        "condition": condition,
                        "target_feature_id": target_feature_id,
                        "direction_feature_id": subset[0]["direction_feature_id"],
                        "alpha": alpha,
                        "n": len(subset),
                        "mean_delta_unsafe_log_odds": float(deltas.mean()),
                        "median_delta_unsafe_log_odds": float(np.median(deltas)),
                        "flip_rate": float(np.mean([row["action_flipped"] for row in subset])),
                    }
                )
    summary_path = output / "steering_summary.json"
    _atomic_json(
        summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": manifest["config_fingerprint"],
            "n_eval_decisions": len(eval_rows),
            "max_baseline_replay_error": max_replay_error,
            "intervention_position": "only final prompt token at pinned resid_post hook",
            "dose_units": (
                "alpha doses use discovery-set SD of target SAE activation; ablation uses "
                "minus the current held-out feature activation times the original decoder row"
            ),
            "controls": [
                "zero",
                "sae_reconstruction",
                "target_feature_ablation",
                "matched_random",
                "unrelated_feature",
                "negative alpha sign reversal",
            ],
            "summaries": summaries,
        },
    )
    manifest["stages"]["steer"] = {
        "status": "complete",
        "rows": len(output_rows),
        "n_eval_decisions": len(eval_rows),
        "max_baseline_replay_error": max_replay_error,
        "artifacts": {
            jsonl_path.name: _sha256_file(jsonl_path),
            summary_path.name: _sha256_file(summary_path),
        },
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)


def _live_experiment(args: argparse.Namespace, model_label: str) -> dict[str, Any]:
    return {
        "name": "greennode_fast_sae_steered_play",
        "games": list(args.treatments),
        "models": [model_label],
        "languages": ["en"],
        "repetitions": int(args.repetitions),
        "seed": int(args.base_seed),
        "agents": "companies_default",
        "runPhase": "pilot",
        "samplingSeedApplied": True,
        "promptVariant": args.prompt_variant,
    }


def _condition_directions(
    args: argparse.Namespace, sae: Any, feature_summary: dict[str, Any]
) -> list[dict[str, Any]]:
    """Freeze target and control directions before any live outcome is seen."""
    conditions: list[dict[str, Any]] = [
        {
            "condition_id": "zero",
            "condition": "zero",
            "target_feature_id": None,
            "direction_feature_id": None,
            "alpha": 0.0,
            "residual_dose": 0.0,
            "direction": None,
        }
    ]
    unrelated_id = int(feature_summary["unrelated_control_feature"]["feature_id"])
    unrelated = sae.W_dec[unrelated_id].detach().float()
    unrelated = unrelated / unrelated.norm().clamp_min(1e-12)
    for row in feature_summary["selected_features"]:
        feature_id = int(row["feature_id"])
        scale = max(float(row["discovery_scale"]), 1e-8)
        target = sae.W_dec[feature_id].detach().float()
        target = target / target.norm().clamp_min(1e-12)
        matched_random = _seeded_unit_random(
            int(target.numel()), args.base_seed + feature_id, target
        ).to(target.device)
        for sign_name, sign in (("positive", 1.0), ("negative", -1.0)):
            alpha = sign * float(args.live_alpha)
            dose = alpha * scale
            for condition, direction, direction_feature_id in (
                ("target_feature", target, feature_id),
                ("matched_random", matched_random, None),
                ("unrelated_feature", unrelated, unrelated_id),
            ):
                conditions.append(
                    {
                        "condition_id": f"feature-{feature_id}__{condition}__{sign_name}",
                        "condition": condition,
                        "target_feature_id": feature_id,
                        "direction_feature_id": direction_feature_id,
                        "alpha": alpha,
                        "residual_dose": dose,
                        "direction": direction,
                    }
                )
    return conditions


def _run_live_trajectory(
    args: argparse.Namespace,
    model: Any,
    game: Any,
    condition: dict[str, Any],
    *,
    baseline_decisions: dict[tuple[int, int], dict[str, Any]] | None,
) -> dict[str, Any]:
    """Run one complete engine trajectory under a frozen residual edit."""
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    direction = condition["direction"]
    dose = float(condition["residual_dose"])
    edit_factory = None
    if direction is not None and dose != 0.0:
        edit_factory = lambda _index, delta=dose * direction: _edit_add(delta)
    decisions: list[dict[str, Any]] = []
    while not game.is_finished:
        prompts = game.build_round_prompts()
        responses: list[dict[str, Any]] = []
        round_rows: list[dict[str, Any]] = []
        for player_index, prompt in enumerate(prompts):
            scores, _ = score_action_pair(
                model,
                prompt,
                hook_name=hook_name,
                edit_factory=edit_factory,
            )
            log_odds = scores["unsafe"] - scores["safe"]
            sampling_seed = game.sampling_seed(player_index, game.current_round)
            action, policy_probability = choose_from_log_odds(
                log_odds,
                temperature=args.decision_temperature,
                seed=sampling_seed,
            )
            responses.append(
                {
                    "text": ACTION_COMPLETIONS[action],
                    "cumulative_logprob": scores[action],
                    "top_alternatives": [
                        {"text": ACTION_COMPLETIONS[name], "cumulative_logprob": value}
                        for name, value in scores.items()
                    ],
                }
            )
            prompt_sha256 = _sha256_bytes(prompt.encode("utf-8"))
            baseline = (
                baseline_decisions.get((game.current_round, player_index))
                if baseline_decisions is not None
                else None
            )
            state_comparable = baseline is not None and baseline["prompt_sha256"] == prompt_sha256
            round_rows.append(
                {
                    "round": game.current_round,
                    "player_index": player_index,
                    "sampling_seed": sampling_seed,
                    "prompt": prompt,
                    "prompt_sha256": prompt_sha256,
                    "action": action,
                    "safe_sequence_logprob": scores["safe"],
                    "unsafe_sequence_logprob": scores["unsafe"],
                    "unsafe_log_odds": log_odds,
                    "model_pair_probability_unsafe": _sigmoid(log_odds),
                    "decision_policy_probability_unsafe": policy_probability,
                    "state_comparable_to_zero": bool(state_comparable),
                    "matched_zero_action": baseline["action"] if state_comparable else None,
                    "action_flipped_if_comparable": (
                        int(action != baseline["action"]) if state_comparable else None
                    ),
                }
            )
        game.apply_round_responses(responses, prompts=prompts)
        for row, turn in zip(round_rows, game.turns[-2:]):
            row["engine_turn"] = turn.to_dict()
            decisions.append(row)
    if game.result is None:
        raise RuntimeError("live steered game finished without engine result")
    return {
        "condition": {
            key: value for key, value in condition.items() if key != "direction"
        },
        "game_id": game.game_id,
        "treatment": game.config.name,
        "rep": game.rep,
        "game_seed": game.seed,
        "decisions": decisions,
        "engine_result": game.result.to_dict(),
    }


def _steered_play(
    args: argparse.Namespace,
    model: Any,
    sae: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    feature_summary: dict[str, Any],
) -> None:
    """Run paired live trajectories with common horizon and setback RNG streams.

    Once an intervention changes an action, subsequent prompts are endogenous
    and can differ from zero.  Therefore action flips are only paired while the
    full current prompt still matches the zero trajectory.  Terminal outcomes
    remain valid trajectory-level treatment results, but are not direct
    fixed-state effects.
    """
    output = manifest_path.parent
    model_label = f"{MODEL_REPO}@{MODEL_REVISION[:12]}+constrained-sequence"
    experiment = _live_experiment(args, model_label)
    all_games = build_games_for_model(experiment, model=model_label)
    eval_ids = {
        race_id for race_id, split in feature_summary["race_split"].items() if split == "eval"
    }
    selected_ids: set[str] = set()
    per_treatment: dict[str, int] = {}
    for game in all_games:
        if game.game_id not in eval_ids:
            continue
        count = per_treatment.get(game.config.name, 0)
        if count >= args.steered_play_max_races_per_treatment:
            continue
        selected_ids.add(game.game_id)
        per_treatment[game.config.name] = count + 1
    if not selected_ids:
        raise ValueError("no held-out whole-race seeds selected for live steering")

    conditions = _condition_directions(args, sae, feature_summary)
    serialized_rows: list[dict[str, Any]] = []
    baseline_by_id: dict[str, dict[tuple[int, int], dict[str, Any]]] = {}
    baseline_result_by_id: dict[str, dict[str, Any]] = {}
    for condition_index, condition in enumerate(conditions, start=1):
        # Rebuild from scratch so every condition receives identical game seeds,
        # hidden horizon draws, fixed-seat setback draws, and initial state.
        games = [
            game
            for game in build_games_for_model(experiment, model=model_label)
            if game.game_id in selected_ids
        ]
        for game_index, game in enumerate(games, start=1):
            baseline = baseline_by_id.get(game.game_id)
            trajectory = _run_live_trajectory(
                args,
                model,
                game,
                condition,
                baseline_decisions=baseline,
            )
            if condition["condition_id"] == "zero":
                baseline_by_id[game.game_id] = {
                    (int(row["round"]), int(row["player_index"])): row
                    for row in trajectory["decisions"]
                }
                baseline_result_by_id[game.game_id] = trajectory["engine_result"]
                for row in trajectory["decisions"]:
                    row["state_comparable_to_zero"] = True
                    row["matched_zero_action"] = row["action"]
                    row["action_flipped_if_comparable"] = 0
            zero_result = baseline_result_by_id[game.game_id]
            result = trajectory["engine_result"]
            trajectory["paired_zero"] = {
                "n_rounds": zero_result["n_rounds"],
                "stop_draws_identical": result["stop_draws"] == zero_result["stop_draws"],
                "setback_draws_identical": result["setback_draws"] == zero_result["setback_draws"],
                "total_final_payoff": float(sum(zero_result["final_payoffs"])),
                "delta_total_final_payoff": float(
                    sum(result["final_payoffs"]) - sum(zero_result["final_payoffs"])
                ),
            }
            if not trajectory["paired_zero"]["stop_draws_identical"]:
                raise RuntimeError("common-random-number horizon stream changed across conditions")
            if not trajectory["paired_zero"]["setback_draws_identical"]:
                raise RuntimeError("fixed-seat setback draws changed across conditions")
            serialized_rows.append(trajectory)
            print(
                f"live condition {condition_index}/{len(conditions)} race "
                f"{game_index}/{len(games)}: {condition['condition_id']} {game.game_id}",
                flush=True,
            )

    jsonl_path = output / "steered_play_races.jsonl"
    temporary = jsonl_path.with_suffix(jsonl_path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in serialized_rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, jsonl_path)

    summaries: list[dict[str, Any]] = []
    condition_ids = sorted({row["condition"]["condition_id"] for row in serialized_rows})
    for condition_id in condition_ids:
        subset = [row for row in serialized_rows if row["condition"]["condition_id"] == condition_id]
        decisions = [decision for row in subset for decision in row["decisions"]]
        comparable = [row for row in decisions if row["state_comparable_to_zero"]]
        final_payoffs = [payoff for row in subset for payoff in row["engine_result"]["final_payoffs"]]
        summaries.append(
            {
                **subset[0]["condition"],
                "n_races": len(subset),
                "n_decisions": len(decisions),
                "unsafe_count": sum(row["action"] == "unsafe" for row in decisions),
                "unsafe_rate": float(np.mean([row["action"] == "unsafe" for row in decisions])),
                "n_state_comparable_decisions": len(comparable),
                "comparable_action_flip_rate": (
                    float(np.mean([row["action_flipped_if_comparable"] for row in comparable]))
                    if comparable
                    else None
                ),
                "mean_rounds": float(np.mean([row["engine_result"]["n_rounds"] for row in subset])),
                "mean_player_final_payoff": float(np.mean(final_payoffs)),
                "setback_count": sum(
                    setback for row in subset for setback in row["engine_result"]["setbacks"]
                ),
                "mean_delta_total_final_payoff_vs_zero": float(
                    np.mean([row["paired_zero"]["delta_total_final_payoff"] for row in subset])
                ),
            }
        )
    summary_path = output / "steered_play_summary.json"
    _atomic_json(
        summary_path,
        {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": manifest["config_fingerprint"],
            "n_unique_race_seeds": len(selected_ids),
            "common_random_numbers": {
                "game_seed": True,
                "hidden_horizon_stream": True,
                "fixed_seat_setback_stream": True,
            },
            "interpretation_boundary": (
                "Direct action flips are evaluated only while the complete prompt matches zero. "
                "After the first action divergence, later states are endogenous; terminal payoff "
                "differences are live-trajectory effects, not fixed-state direct effects."
            ),
            "summaries": summaries,
        },
    )
    manifest["stages"]["steered_play"] = {
        "status": "complete",
        "n_unique_race_seeds": len(selected_ids),
        "n_conditions": len(conditions),
        "n_trajectories": len(serialized_rows),
        "artifacts": {
            jsonl_path.name: _sha256_file(jsonl_path),
            summary_path.name: _sha256_file(summary_path),
        },
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.repetitions < 2:
        raise ValueError("at least two repetitions are required for discovery/eval race separation")
    manifest_path, manifest = _prepare_manifest(args)
    needs_model = args.stage in {"all", "selfplay", "steer", "steered_play"}
    model = sae = None
    if needs_model:
        model, sae, _ = _load_model_and_sae(args, manifest)
        _atomic_json(manifest_path, manifest)
    if args.stage in {"all", "selfplay"}:
        _selfplay(args, model, sae, manifest_path, manifest)
    feature_summary: dict[str, Any] | None = None
    if args.stage in {"all", "mine"}:
        feature_summary = _mine(args, manifest_path, manifest)
    if args.stage in {"all", "steer"}:
        if feature_summary is None:
            feature_path = manifest_path.parent / "feature_mining.json"
            if not feature_path.is_file():
                raise FileNotFoundError("feature_mining.json missing; run --stage mine first")
            feature_summary = json.loads(feature_path.read_text(encoding="utf-8"))
        _steer(args, model, sae, manifest_path, manifest, feature_summary)
    if args.stage in {"all", "steered_play"}:
        if feature_summary is None:
            feature_path = manifest_path.parent / "feature_mining.json"
            if not feature_path.is_file():
                raise FileNotFoundError("feature_mining.json missing; run --stage mine first")
            feature_summary = json.loads(feature_path.read_text(encoding="utf-8"))
        _steered_play(args, model, sae, manifest_path, manifest, feature_summary)
    required = {"selfplay", "mine", "steer", "steered_play"}
    if required.issubset(
        {name for name, value in manifest.get("stages", {}).items() if value.get("status") == "complete"}
    ):
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _utc_now()
        _atomic_json(manifest_path, manifest)
    print(f"run status={manifest['status']} output={manifest_path.parent}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
