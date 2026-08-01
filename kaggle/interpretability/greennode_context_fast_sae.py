"""Fixed-state context mediation with pinned Qwen2.5 and FAST SAE.

This runner is deliberately separate from both the Ollama context pilot and
the live-trajectory SAE runner.  It renders every context skin from the same
engine-reachable decision states, scores the two complete opaque responses
(``ACTION: P`` and ``ACTION: Q``), and captures the final prompt token before
either response is emitted.  Consequently, all labels in this run come from
the pinned native Hugging Face checkpoint used for the activation capture.

The analysis holds out both whole source trajectories and complete context
pairs.  Feature selection is discovery-only.  The steering stage is evaluated
on the double-held-out quadrant and includes zero, SAE-reconstruction,
target-ablation, matched-random, unrelated-feature, and sign-reversal controls.
It is an exploratory mediation test, not a claim that an SAE feature is a
human-readable or unique cause of the model's decision.
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
import subprocess
import sys
from typing import Any, Callable, Iterable, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from ai_race.audit.context_replay import FrozenDecisionState, generate_reachable_states, render_replay_prompt
from ai_race.dataio.config_loader import load_game_config, load_json
from ai_race.prompts.context_skins import ACTION_CODE_MAPPINGS, SKINS, get_action_code_mapping
from kaggle.interpretability.greennode_fast_sae_selfplay import (
    MODEL_REPO,
    MODEL_REVISION,
    SAE_ID_TEMPLATE,
    SAE_LENS_VERSION,
    SAE_REPO,
    SAE_REVISION,
    SUPPORTED_LAYERS,
    _completion_tokens,
    _edit_add,
    _edit_replace,
    _load_model_and_sae,
    _render_chat,
    _score_completion,
    _seeded_unit_random,
)


SCHEMA_VERSION = "ai-race.context-fast-sae.v1"
PROFILE_STATES_PER_RISK = {"smoke": 4, "pilot": 32}
OPAQUE_COMPLETIONS = {"P": "ACTION: P", "Q": "ACTION: Q"}

# Pairs, rather than individual prompts, are held out.  The discovery pairs
# span recognizable/decontextualized and realistic/fictional contrasts.  The
# evaluation pairs come from two unseen domains.
DISCOVERY_CONTEXT_PAIRS = (
    ("technology_race", "abstract_contest"),
    ("logistics_contract", "crystal_guild_contract"),
)
EVALUATION_CONTEXT_PAIRS = (
    ("hospital_deployment", "colony_life_support"),
    ("robotic_expedition", "fictional_cartography"),
)
DISCOVERY_SKINS = tuple(dict.fromkeys(item for pair in DISCOVERY_CONTEXT_PAIRS for item in pair))
EVALUATION_SKINS = tuple(dict.fromkeys(item for pair in EVALUATION_CONTEXT_PAIRS for item in pair))


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
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    with temporary.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temporary, path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
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


def _guard_output_path(path: Path) -> Path:
    resolved = path.expanduser().resolve()
    if str(resolved).replace("\\", "/").startswith("/network-volume"):
        raise ValueError("GreenNode /network-volume is unavailable; use /home/jovyan and SCP afterward")
    return resolved


def _sigmoid(value: float) -> float:
    if value >= 0:
        return 1.0 / (1.0 + math.exp(-value))
    exp_value = math.exp(value)
    return exp_value / (1.0 + exp_value)


def trajectory_group_id(state: FrozenDecisionState | dict[str, Any]) -> str:
    """Group both seats and all rounds of a source trajectory together."""
    if isinstance(state, dict):
        return f"{state['game_name']}|{state['trajectory_id']}"
    return f"{state.game_name}|{state.trajectory_id}"


def grouped_trajectory_split(
    states: Sequence[FrozenDecisionState | dict[str, Any]], *, eval_fraction: float, seed: int
) -> dict[str, str]:
    """Deterministic risk-stratified split with no trajectory crossing."""
    if not 0.0 < eval_fraction < 1.0:
        raise ValueError("eval_fraction must be in (0, 1)")
    strata: dict[str, set[str]] = {}
    for state in states:
        if isinstance(state, dict):
            stratum = str(state["game_name"])
        else:
            stratum = state.game_name
        strata.setdefault(stratum, set()).add(trajectory_group_id(state))
    assignment: dict[str, str] = {}
    for stratum, groups in sorted(strata.items()):
        ordered = sorted(
            groups,
            key=lambda group: _sha256_bytes(f"{seed}|{stratum}|{group}".encode("utf-8")),
        )
        if len(ordered) < 2:
            raise ValueError(f"stratum {stratum!r} has fewer than two trajectories")
        n_eval = min(len(ordered) - 1, max(1, round(len(ordered) * eval_fraction)))
        eval_groups = set(ordered[:n_eval])
        assignment.update(
            {group: ("evaluation" if group in eval_groups else "discovery") for group in ordered}
        )
    return assignment


def canonical_action_scores(scores: dict[str, float], mapping_id: str) -> dict[str, Any]:
    """Decode P/Q sequence scores into a canonical Safe/Unsafe decision."""
    mapping = get_action_code_mapping(mapping_id)
    if set(scores) != {"P", "Q"}:
        raise ValueError("opaque action scores must contain exactly P and Q")
    emitted_code = "P" if float(scores["P"]) >= float(scores["Q"]) else "Q"
    action = mapping.decode(emitted_code).lower()
    safe_score = float(scores[mapping.safe_code])
    unsafe_score = float(scores[mapping.unsafe_code])
    unsafe_log_odds = unsafe_score - safe_score
    return {
        "emitted_code": emitted_code,
        "action": action,
        "unsafe": int(action == "unsafe"),
        "safe_sequence_logprob": safe_score,
        "unsafe_sequence_logprob": unsafe_score,
        "unsafe_log_odds": unsafe_log_odds,
        "model_pair_probability_unsafe": _sigmoid(unsafe_log_odds),
    }


def _score_opaque_pair(
    model: Any,
    prompt: str,
    *,
    hook_name: str,
    edit: Callable[[Any, int], Any] | None = None,
    capture: bool = False,
) -> tuple[dict[str, float], Any | None]:
    rendered = _render_chat(model.tokenizer, prompt)
    prefix = model.to_tokens(rendered, prepend_bos=False)
    if int(prefix.shape[-1]) >= 2048:
        raise ValueError(f"prompt has {prefix.shape[-1]} tokens; FAST SAE limit is 2048")
    scores: dict[str, float] = {}
    captured = None
    for index, (code, completion) in enumerate(OPAQUE_COMPLETIONS.items()):
        completion_ids = _completion_tokens(model, completion)
        score, activation = _score_completion(
            model,
            prefix,
            completion_ids,
            hook_name=hook_name,
            edit=edit,
            capture=capture and index == 0,
        )
        scores[code] = score
        if activation is not None:
            captured = activation
    return scores, captured


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
        "capture_position": "final prompt token before ACTION:P/Q",
        "action_policy": "argmax full-sequence likelihood over ACTION: P/Q including EOS",
        "profile": args.profile,
        "states_per_risk": args.states_per_risk or PROFILE_STATES_PER_RISK[args.profile],
        "base_seed": args.base_seed,
        "eval_fraction": args.eval_fraction,
        "discovery_context_pairs": DISCOVERY_CONTEXT_PAIRS,
        "evaluation_context_pairs": EVALUATION_CONTEXT_PAIRS,
        "top_features": args.top_features,
        "min_feature_prevalence": args.min_feature_prevalence,
        "steering_alphas": args.steering_alphas,
        "steering_max_rows": args.steering_max_rows,
        "score_replay_tolerance": args.score_replay_tolerance,
        "device": args.device,
        "dtype": args.dtype,
    }


def _prepare_manifest(args: argparse.Namespace) -> tuple[Path, dict[str, Any]]:
    output = _guard_output_path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    config = _run_config(args)
    runner_sha = _sha256_file(__file__)
    fingerprint = _canonical_sha256({"config": config, "runner_sha256": runner_sha})
    path = output / "manifest.json"
    if path.exists():
        prior = json.loads(path.read_text(encoding="utf-8"))
        if prior.get("config_fingerprint") != fingerprint:
            raise ValueError("output belongs to a different immutable run configuration")
        if not args.resume and args.stage in {"all", "capture"}:
            raise FileExistsError("run exists; use --resume only for the identical configuration")
        return path, prior
    source_files = (
        "ai_race/audit/context_replay.py",
        "ai_race/prompts/context_skins.py",
        "ai_race/engine/game.py",
        "ai_race/engine/prompt.py",
        "kaggle/interpretability/greennode_fast_sae_selfplay.py",
    )
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "status": "running",
        "created_at_utc": _utc_now(),
        "config": config,
        "config_fingerprint": fingerprint,
        "git_revision": _git_revision(),
        "runner_sha256": runner_sha,
        "source_sha256": {name: _sha256_file(REPO_ROOT / name) for name in source_files},
        "environment": {
            "hostname": platform.node(),
            "python": platform.python_version(),
            "torch": _package_version("torch"),
            "transformer_lens": _package_version("transformer-lens"),
            "sae_lens": _package_version("sae-lens"),
            "scikit_learn": _package_version("scikit-learn"),
        },
        "label_provenance": (
            "native pinned HF checkpoint full-sequence P/Q likelihood; no Ollama or historical labels"
        ),
        "evidence_class": "exploratory-fixed-state-context-mediation",
        "claim_boundary": (
            "checkpoint-, SAE-, layer-, context-pair-, state-bank-, and intervention-specific"
        ),
        "stages": {},
    }
    _atomic_json(path, manifest)
    return path, manifest


def _load_states(args: argparse.Namespace) -> tuple[list[FrozenDecisionState], dict[str, str], Path]:
    config_path = args.config if args.config.is_absolute() else REPO_ROOT / args.config
    experiment = load_json(config_path)
    if set(experiment["contextSkins"]) != set(SKINS):
        raise ValueError("experiment config must contain exactly the eight registered context skins")
    configs = [
        load_game_config(
            REPO_ROOT / "ai_race" / "configs" / "game" / f"{name}.json",
            model=f"{MODEL_REPO}@{MODEL_REVISION[:12]}",
        )
        for name in experiment["games"]
    ]
    states = generate_reachable_states(
        configs,
        states_per_config=args.states_per_risk or PROFILE_STATES_PER_RISK[args.profile],
        base_seed=args.base_seed,
    )
    split = grouped_trajectory_split(states, eval_fraction=args.eval_fraction, seed=args.base_seed)
    return states, split, config_path


def _capture(
    args: argparse.Namespace, model: Any, sae: Any, manifest_path: Path, manifest: dict[str, Any]
) -> None:
    import torch

    states, split, config_path = _load_states(args)
    output = manifest_path.parent
    experiment = load_json(config_path)
    game_config_hashes = {
        name: _sha256_file(
            REPO_ROOT / "ai_race" / "configs" / "game" / f"{name}.json"
        )
        for name in experiment["games"]
    }
    state_path = output / "reachable_states.jsonl"
    _atomic_jsonl(state_path, [state.to_dict() for state in states])
    shard_dir = output / "state_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    expected_rows = len(states) * len(SKINS) * len(ACTION_CODE_MAPPINGS)
    completed: list[dict[str, Any]] = []
    for state_index, state in enumerate(states, start=1):
        json_path = shard_dir / f"{state.state_id}.json"
        npz_path = shard_dir / f"{state.state_id}.npz"
        if json_path.exists() and npz_path.exists():
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            if payload.get("config_fingerprint") != manifest["config_fingerprint"]:
                raise ValueError(f"mixed fingerprint in {json_path}")
            if _sha256_file(npz_path) != payload.get("npz_sha256"):
                raise ValueError(f"checksum mismatch in {npz_path}")
            completed.append(
                {
                    "state_id": state.state_id,
                    "json_sha256": _sha256_file(json_path),
                    "npz_sha256": _sha256_file(npz_path),
                }
            )
            continue
        rows: list[dict[str, Any]] = []
        codes: list[np.ndarray] = []
        for skin_id in sorted(SKINS):
            for mapping_id in sorted(ACTION_CODE_MAPPINGS):
                prompt = render_replay_prompt(state, skin_id=skin_id, mapping_id=mapping_id)
                scores, activation = _score_opaque_pair(
                    model, prompt, hook_name=hook_name, capture=True
                )
                if activation is None:
                    raise RuntimeError("final-prompt-token activation capture failed")
                with torch.inference_mode():
                    code = sae.encode(activation.unsqueeze(0)).squeeze(0)
                    reconstruction = sae.decode(code.unsqueeze(0)).squeeze(0)
                decoded = canonical_action_scores(scores, mapping_id)
                residual = activation.detach().float()
                reconstructed = reconstruction.detach().float()
                row = {
                    "state_id": state.state_id,
                    "trajectory_id": state.trajectory_id,
                    "trajectory_group_id": trajectory_group_id(state),
                    "state_split": split[trajectory_group_id(state)],
                    "game_name": state.game_name,
                    "max_private_risk": state.max_private_risk,
                    "source_seed": state.source_seed,
                    "round": state.round_number,
                    "player_index": state.player_index,
                    "skin_id": skin_id,
                    "context_split": "discovery" if skin_id in DISCOVERY_SKINS else "evaluation",
                    "mapping_id": mapping_id,
                    "sampling_seed": state.sampling_seed,
                    "prompt": prompt,
                    "prompt_sha256": _sha256_bytes(prompt.encode("utf-8")),
                    "rendered_chat_sha256": _sha256_bytes(
                        _render_chat(model.tokenizer, prompt).encode("utf-8")
                    ),
                    "p_sequence_logprob": float(scores["P"]),
                    "q_sequence_logprob": float(scores["Q"]),
                    "raw_response": OPAQUE_COMPLETIONS[decoded["emitted_code"]],
                    **decoded,
                    "sae_l0": int(torch.count_nonzero(code).cpu()),
                    "sae_normalized_mse": float(
                        torch.square(residual - reconstructed).sum()
                        / torch.square(residual).sum().clamp_min(1e-12)
                    ),
                    "sae_cosine_similarity": float(
                        torch.nn.functional.cosine_similarity(residual, reconstructed, dim=0)
                    ),
                }
                rows.append(row)
                codes.append(code.detach().float().cpu().numpy())
        _atomic_npz(npz_path, sae_codes=np.stack(codes).astype(np.float32))
        payload = {
            "schema_version": SCHEMA_VERSION,
            "config_fingerprint": manifest["config_fingerprint"],
            "state_id": state.state_id,
            "records": rows,
            "npz_file": npz_path.name,
            "npz_sha256": _sha256_file(npz_path),
        }
        _atomic_json(json_path, payload)
        completed.append(
            {
                "state_id": state.state_id,
                "json_sha256": _sha256_file(json_path),
                "npz_sha256": _sha256_file(npz_path),
            }
        )
        manifest["stages"]["capture"] = {
            "status": "running",
            "n_expected_states": len(states),
            "n_completed_states": len(completed),
            "updated_at_utc": _utc_now(),
        }
        _atomic_json(manifest_path, manifest)
        print(f"[{state_index}/{len(states)}] captured {state.state_id}", flush=True)
    manifest["stages"]["capture"] = {
        "status": "complete",
        "n_states": len(states),
        "n_rows": expected_rows,
        "n_contexts": len(SKINS),
        "n_mappings": len(ACTION_CODE_MAPPINGS),
        "state_split": split,
        "state_shards": completed,
        "game_config_sha256": game_config_hashes,
        "artifacts": {
            state_path.name: _sha256_file(state_path),
            config_path.name: _sha256_file(config_path),
        },
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)


def _load_capture(output: Path, fingerprint: str) -> tuple[list[dict[str, Any]], np.ndarray]:
    records: list[dict[str, Any]] = []
    code_arrays: list[np.ndarray] = []
    paths = sorted((output / "state_shards").glob("*.json"))
    if not paths:
        raise FileNotFoundError("no context state shards; run --stage capture first")
    for json_path in paths:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        if payload.get("config_fingerprint") != fingerprint:
            raise ValueError(f"mixed fingerprint in {json_path}")
        npz_path = json_path.with_suffix(".npz")
        if _sha256_file(npz_path) != payload.get("npz_sha256"):
            raise ValueError(f"NPZ checksum mismatch in {npz_path}")
        with np.load(npz_path) as arrays:
            codes = np.asarray(arrays["sae_codes"], dtype=np.float32)
        if len(payload["records"]) != codes.shape[0]:
            raise ValueError(f"record/code mismatch in {json_path}")
        records.extend(payload["records"])
        code_arrays.append(codes)
    codes = np.concatenate(code_arrays, axis=0)
    expected_cells = {(skin, mapping) for skin in SKINS for mapping in ACTION_CODE_MAPPINGS}
    by_state: dict[str, set[tuple[str, str]]] = {}
    for row in records:
        by_state.setdefault(str(row["state_id"]), set()).add(
            (str(row["skin_id"]), str(row["mapping_id"]))
        )
    incomplete = [state_id for state_id, cells in by_state.items() if cells != expected_cells]
    if incomplete:
        raise ValueError(f"incomplete context cells for states: {incomplete[:3]}")
    return records, codes


def build_context_pair_examples(
    records: Sequence[dict[str, Any]],
    codes: np.ndarray,
    pairs: Sequence[tuple[str, str]],
    *,
    required_state_split: str,
) -> tuple[list[dict[str, Any]], np.ndarray]:
    """Construct matched context deltas; no row from another split is admitted."""
    if len(records) != codes.shape[0]:
        raise ValueError("record/code length mismatch")
    lookup = {
        (str(row["state_id"]), str(row["skin_id"]), str(row["mapping_id"])): index
        for index, row in enumerate(records)
        if row["state_split"] == required_state_split
    }
    state_ids = sorted(
        {str(row["state_id"]) for row in records if row["state_split"] == required_state_split}
    )
    examples: list[dict[str, Any]] = []
    deltas: list[np.ndarray] = []
    for left_skin, right_skin in pairs:
        for state_id in state_ids:
            for mapping_id in sorted(ACTION_CODE_MAPPINGS):
                left_key = (state_id, left_skin, mapping_id)
                right_key = (state_id, right_skin, mapping_id)
                if left_key not in lookup or right_key not in lookup:
                    raise ValueError(f"missing matched context cell {left_key} or {right_key}")
                left_index, right_index = lookup[left_key], lookup[right_key]
                left, right = records[left_index], records[right_index]
                log_odds_delta = float(right["unsafe_log_odds"]) - float(left["unsafe_log_odds"])
                examples.append(
                    {
                        "state_id": state_id,
                        "trajectory_group_id": left["trajectory_group_id"],
                        "state_split": required_state_split,
                        "pair_id": f"{left_skin}__vs__{right_skin}",
                        "left_skin": left_skin,
                        "right_skin": right_skin,
                        "mapping_id": mapping_id,
                        "left_action": left["action"],
                        "right_action": right["action"],
                        "action_flip": int(left["action"] != right["action"]),
                        "left_unsafe_log_odds": float(left["unsafe_log_odds"]),
                        "right_unsafe_log_odds": float(right["unsafe_log_odds"]),
                        "delta_unsafe_log_odds": log_odds_delta,
                    }
                )
                deltas.append(codes[right_index] - codes[left_index])
    return examples, np.stack(deltas).astype(np.float32)


def _column_correlation(matrix: np.ndarray, target: np.ndarray) -> np.ndarray:
    x = matrix.astype(np.float64, copy=False)
    y = target.astype(np.float64, copy=False)
    x_centered = x - x.mean(axis=0)
    y_centered = y - y.mean()
    numerator = np.sum(x_centered * y_centered[:, None], axis=0)
    denominator = np.sqrt(np.sum(x_centered**2, axis=0) * np.sum(y_centered**2))
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def _flip_effect(matrix: np.ndarray, flips: np.ndarray) -> np.ndarray:
    if len(np.unique(flips)) < 2:
        return np.zeros(matrix.shape[1], dtype=np.float64)
    flipped = np.abs(matrix[flips == 1])
    stable = np.abs(matrix[flips == 0])
    scale = np.abs(matrix).std(axis=0, dtype=np.float64)
    return np.divide(
        flipped.mean(axis=0) - stable.mean(axis=0),
        scale,
        out=np.zeros_like(scale),
        where=scale > 0,
    )


def _feature_ranking(
    discovery_rows: Sequence[dict[str, Any]],
    discovery_delta: np.ndarray,
    evaluation_rows: Sequence[dict[str, Any]],
    evaluation_delta: np.ndarray,
    *,
    min_prevalence: float,
) -> list[dict[str, Any]]:
    target = np.asarray([row["delta_unsafe_log_odds"] for row in discovery_rows], dtype=np.float64)
    flips = np.asarray([row["action_flip"] for row in discovery_rows], dtype=np.int8)
    eval_target = np.asarray([row["delta_unsafe_log_odds"] for row in evaluation_rows], dtype=np.float64)
    eval_flips = np.asarray([row["action_flip"] for row in evaluation_rows], dtype=np.int8)
    prevalence = np.mean(discovery_delta != 0, axis=0)
    scales = discovery_delta.std(axis=0, dtype=np.float64)
    corr = _column_correlation(discovery_delta, target)
    flip_d = _flip_effect(discovery_delta, flips)
    eval_corr = _column_correlation(evaluation_delta, eval_target)
    eval_flip_d = _flip_effect(evaluation_delta, eval_flips)
    eligible = np.flatnonzero((prevalence >= min_prevalence) & (scales > 0))
    order = sorted(
        eligible.tolist(), key=lambda feature: (-abs(flip_d[feature]), -abs(corr[feature]), feature)
    )
    return [
        {
            "feature_id": int(feature),
            "discovery_corr_context_delta_vs_log_odds_delta": float(corr[feature]),
            "discovery_flip_abs_delta_cohens_d": float(flip_d[feature]),
            "evaluation_corr_context_delta_vs_log_odds_delta": float(eval_corr[feature]),
            "evaluation_flip_abs_delta_cohens_d": float(eval_flip_d[feature]),
            "discovery_prevalence": float(prevalence[feature]),
            "discovery_delta_scale": float(scales[feature]),
        }
        for feature in order
    ]


def _safe_auc(y_true: np.ndarray, probability: np.ndarray) -> float | None:
    if len(np.unique(y_true)) < 2:
        return None
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(y_true, probability))


def _action_probe(records: Sequence[dict[str, Any]], codes: np.ndarray, seed: int) -> dict[str, Any]:
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler

    train = np.asarray(
        [row["state_split"] == "discovery" and row["context_split"] == "discovery" for row in records]
    )
    y = np.asarray([row["unsafe"] for row in records], dtype=np.int8)
    variance = codes[train].var(axis=0)
    n_variable = int(np.count_nonzero(variance))
    if n_variable == 0 or len(np.unique(y[train])) < 2:
        return {
            "available": False,
            "reason": "discovery quadrant has no variable features or only one action class",
            "n_train": int(train.sum()),
        }
    selected = np.argsort(variance)[-min(4096, n_variable):]
    scaler = StandardScaler().fit(codes[train][:, selected])
    classifier = LogisticRegression(C=1.0, max_iter=2000, random_state=seed).fit(
        scaler.transform(codes[train][:, selected]), y[train]
    )
    quadrants = {}
    for state_split in ("discovery", "evaluation"):
        for context_split in ("discovery", "evaluation"):
            mask = np.asarray(
                [
                    row["state_split"] == state_split and row["context_split"] == context_split
                    for row in records
                ]
            )
            probability = classifier.predict_proba(scaler.transform(codes[mask][:, selected]))[:, 1]
            truth = y[mask]
            key = f"{state_split}_states__{context_split}_contexts"
            quadrants[key] = {
                "n": int(mask.sum()),
                "unsafe_rate": float(truth.mean()),
                "accuracy": float(np.mean((probability >= 0.5) == truth)),
                "roc_auc": _safe_auc(truth, probability),
            }
    return {
        "available": True,
        "probe": "L2 logistic regression on discovery-only variance-selected SAE features",
        "n_train": int(train.sum()),
        "n_features": int(selected.size),
        "selected_feature_ids_sha256": _canonical_sha256(selected.tolist()),
        "quadrants": quadrants,
        "interpretation": "predictive association only; AUC is not a causal or semantic explanation",
    }


def _pair_summary(rows: Sequence[dict[str, Any]], delta: np.ndarray) -> list[dict[str, Any]]:
    summaries = []
    for pair_id in sorted({str(row["pair_id"]) for row in rows}):
        indices = [index for index, row in enumerate(rows) if row["pair_id"] == pair_id]
        subset = [rows[index] for index in indices]
        delta_norm = np.linalg.norm(delta[indices], axis=1)
        summaries.append(
            {
                "pair_id": pair_id,
                "n": len(indices),
                "action_flip_rate": float(np.mean([row["action_flip"] for row in subset])),
                "mean_delta_unsafe_log_odds": float(
                    np.mean([row["delta_unsafe_log_odds"] for row in subset])
                ),
                "mean_abs_delta_unsafe_log_odds": float(
                    np.mean(np.abs([row["delta_unsafe_log_odds"] for row in subset]))
                ),
                "mean_sae_code_delta_l2": float(delta_norm.mean()),
            }
        )
    return summaries


def _analyze(args: argparse.Namespace, manifest_path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    output = manifest_path.parent
    records, codes = _load_capture(output, manifest["config_fingerprint"])
    discovery_rows, discovery_delta = build_context_pair_examples(
        records, codes, DISCOVERY_CONTEXT_PAIRS, required_state_split="discovery"
    )
    evaluation_rows, evaluation_delta = build_context_pair_examples(
        records, codes, EVALUATION_CONTEXT_PAIRS, required_state_split="evaluation"
    )
    ranking = _feature_ranking(
        discovery_rows,
        discovery_delta,
        evaluation_rows,
        evaluation_delta,
        min_prevalence=args.min_feature_prevalence,
    )
    if len(ranking) < args.top_features + 1:
        raise ValueError("not enough eligible features for targets and unrelated control")
    selected = ranking[: args.top_features]
    unrelated = min(
        ranking[args.top_features :],
        key=lambda row: (
            abs(row["discovery_corr_context_delta_vs_log_odds_delta"])
            + abs(row["discovery_flip_abs_delta_cohens_d"])
        ),
    )
    pair_rows_path = output / "context_pair_rows.jsonl"
    _atomic_jsonl(pair_rows_path, [*discovery_rows, *evaluation_rows])
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": manifest["config_fingerprint"],
        "split_unit": "whole engine source trajectory and whole preregistered context pair",
        "selection_data": "discovery trajectories x discovery context pairs only",
        "confirmation_data": "evaluation trajectories x evaluation context pairs only",
        "n_records": len(records),
        "n_states": len({row["state_id"] for row in records}),
        "discovery_context_pairs": DISCOVERY_CONTEXT_PAIRS,
        "evaluation_context_pairs": EVALUATION_CONTEXT_PAIRS,
        "context_shift_metrics": {
            "discovery": _pair_summary(discovery_rows, discovery_delta),
            "double_heldout": _pair_summary(evaluation_rows, evaluation_delta),
        },
        "action_probe": _action_probe(records, codes, args.base_seed),
        "action_flip_feature_selection": {
            "target": (
                "discovery-only standardized separation of absolute context-code deltas in "
                "action-flip versus stable pairs; matched log-odds-delta correlation is the tie-break"
            ),
            "n_discovery_pairs": len(discovery_rows),
            "n_discovery_action_flips": int(
                sum(int(row["action_flip"]) for row in discovery_rows)
            ),
            "fallback_if_no_discovery_flips": (
                "ranking reduces to discovery log-odds-delta correlation; this must not be labeled flip mining"
            ),
            "selected_features": selected,
            "unrelated_control_feature": unrelated,
            "ranking": ranking[: args.ranking_limit],
        },
        "limitations": [
            "Probe performance is association, not causal XAI.",
            "Feature selection and dose calibration never use held-out context pairs or trajectories.",
            "A steering effect supports a local causal role only if it exceeds matched controls and reverses with sign/dose.",
            "A fixed-state effect does not estimate downstream live-trajectory payoff effects.",
        ],
        "artifacts": {pair_rows_path.name: _sha256_file(pair_rows_path)},
    }
    path = output / "context_analysis.json"
    _atomic_json(path, summary)
    manifest["stages"]["analyze"] = {
        "status": "complete",
        "selected_feature_ids": [row["feature_id"] for row in selected],
        "artifacts": {path.name: _sha256_file(path), pair_rows_path.name: _sha256_file(pair_rows_path)},
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)
    return summary


def _steer(
    args: argparse.Namespace,
    model: Any,
    sae: Any,
    manifest_path: Path,
    manifest: dict[str, Any],
    analysis: dict[str, Any],
) -> None:
    import torch

    records, _ = _load_capture(manifest_path.parent, manifest["config_fingerprint"])
    heldout_all = [
        row
        for row in records
        if row["state_split"] == "evaluation" and row["context_split"] == "evaluation"
    ]
    heldout_state_ids = sorted(
        {str(row["state_id"]) for row in heldout_all},
        key=lambda state_id: _sha256_bytes(f"{args.base_seed}|{state_id}".encode("utf-8")),
    )
    rows_per_state = len(EVALUATION_SKINS) * len(ACTION_CODE_MAPPINGS)
    n_states = max(1, args.steering_max_rows // rows_per_state)
    selected_state_ids = set(heldout_state_ids[:n_states])
    heldout = [row for row in heldout_all if str(row["state_id"]) in selected_state_ids]
    heldout.sort(key=lambda row: (str(row["state_id"]), str(row["skin_id"]), str(row["mapping_id"])))
    if not heldout:
        raise ValueError("double-held-out steering set is empty")
    if len(heldout) != len(selected_state_ids) * rows_per_state:
        raise ValueError("held-out steering selection lost a matched context/mapping cell")
    selection = analysis["action_flip_feature_selection"]
    targets = selection["selected_features"]
    unrelated_id = int(selection["unrelated_control_feature"]["feature_id"])
    unrelated = sae.W_dec[unrelated_id].detach().float()
    unrelated = unrelated / unrelated.norm().clamp_min(1e-12)
    hook_name = f"blocks.{args.layer}.hook_resid_post"
    output_rows: list[dict[str, Any]] = []
    max_replay_error = 0.0

    def record_result(
        source: dict[str, Any],
        baseline: dict[str, Any],
        condition: str,
        scores: dict[str, float],
        *,
        target_id: int | None = None,
        direction_id: int | None = None,
        alpha: float | None = None,
        residual_dose: float | None = None,
    ) -> None:
        decoded = canonical_action_scores(scores, str(source["mapping_id"]))
        output_rows.append(
            {
                "state_id": source["state_id"],
                "trajectory_group_id": source["trajectory_group_id"],
                "skin_id": source["skin_id"],
                "mapping_id": source["mapping_id"],
                "condition": condition,
                "target_feature_id": target_id,
                "direction_feature_id": direction_id,
                "alpha": alpha,
                "residual_dose": residual_dose,
                "baseline_action": baseline["action"],
                "steered_action": decoded["action"],
                "action_flipped": int(baseline["action"] != decoded["action"]),
                "baseline_unsafe_log_odds": baseline["unsafe_log_odds"],
                "steered_unsafe_log_odds": decoded["unsafe_log_odds"],
                "delta_unsafe_log_odds": decoded["unsafe_log_odds"] - baseline["unsafe_log_odds"],
                "p_sequence_logprob": float(scores["P"]),
                "q_sequence_logprob": float(scores["Q"]),
            }
        )

    for sample_index, row in enumerate(heldout, start=1):
        baseline_scores, activation = _score_opaque_pair(
            model, row["prompt"], hook_name=hook_name, capture=True
        )
        if activation is None:
            raise RuntimeError("steering baseline activation capture failed")
        baseline = canonical_action_scores(baseline_scores, str(row["mapping_id"]))
        replay_error = abs(float(baseline["unsafe_log_odds"]) - float(row["unsafe_log_odds"]))
        max_replay_error = max(max_replay_error, replay_error)
        if replay_error > args.score_replay_tolerance:
            raise RuntimeError(
                f"baseline replay mismatch {replay_error:.6g}; runtime/model is not identical"
            )
        with torch.inference_mode():
            code = sae.encode(activation.unsqueeze(0)).squeeze(0)
            reconstruction = sae.decode(code.unsqueeze(0)).squeeze(0)
        record_result(row, baseline, "zero", baseline_scores, alpha=0.0, residual_dose=0.0)
        reconstruction_scores, _ = _score_opaque_pair(
            model,
            row["prompt"],
            hook_name=hook_name,
            edit=_edit_replace(reconstruction),
        )
        record_result(row, baseline, "sae_reconstruction", reconstruction_scores)
        for target_row in targets:
            feature_id = int(target_row["feature_id"])
            decoder = sae.W_dec[feature_id].detach().float()
            direction = decoder / decoder.norm().clamp_min(1e-12)
            feature_activation = float(code[feature_id].detach().float().cpu())
            ablation_delta = -feature_activation * decoder
            ablation_scores, _ = _score_opaque_pair(
                model,
                row["prompt"],
                hook_name=hook_name,
                edit=_edit_add(ablation_delta),
            )
            record_result(
                row,
                baseline,
                "target_feature_ablation",
                ablation_scores,
                target_id=feature_id,
                direction_id=feature_id,
                residual_dose=float(ablation_delta.norm().cpu()),
            )
            output_rows[-1]["ablated_feature_activation"] = feature_activation
            random_direction = _seeded_unit_random(
                int(direction.numel()), args.base_seed + feature_id, direction
            ).to(direction.device)
            scale = max(float(target_row["discovery_delta_scale"]), 1e-8)
            for alpha in args.steering_alphas:
                dose = float(alpha) * scale
                for condition, test_direction, direction_id in (
                    ("target_feature", direction, feature_id),
                    ("matched_random", random_direction, None),
                    ("unrelated_feature", unrelated, unrelated_id),
                ):
                    scores, _ = _score_opaque_pair(
                        model,
                        row["prompt"],
                        hook_name=hook_name,
                        edit=_edit_add(dose * test_direction),
                    )
                    record_result(
                        row,
                        baseline,
                        condition,
                        scores,
                        target_id=feature_id,
                        direction_id=direction_id,
                        alpha=float(alpha),
                        residual_dose=dose,
                    )
        print(f"[{sample_index}/{len(heldout)}] steered held-out context prompt", flush=True)
    rows_path = manifest_path.parent / "context_steering_rows.jsonl"
    _atomic_jsonl(rows_path, output_rows)
    summaries = []
    keys = sorted(
        {
            (row["condition"], row["target_feature_id"], row["alpha"])
            for row in output_rows
        },
        key=lambda item: (item[0], -1 if item[1] is None else item[1], -99 if item[2] is None else item[2]),
    )
    for condition, feature_id, alpha in keys:
        subset = [
            row
            for row in output_rows
            if (row["condition"], row["target_feature_id"], row["alpha"])
            == (condition, feature_id, alpha)
        ]
        summaries.append(
            {
                "condition": condition,
                "target_feature_id": feature_id,
                "alpha": alpha,
                "n": len(subset),
                "mean_delta_unsafe_log_odds": float(
                    np.mean([row["delta_unsafe_log_odds"] for row in subset])
                ),
                "median_delta_unsafe_log_odds": float(
                    np.median([row["delta_unsafe_log_odds"] for row in subset])
                ),
                "action_flip_rate": float(np.mean([row["action_flipped"] for row in subset])),
            }
        )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "config_fingerprint": manifest["config_fingerprint"],
        "evaluation_quadrant": "evaluation trajectories x evaluation context pairs",
        "n_heldout_prompts": len(heldout),
        "max_baseline_replay_error": max_replay_error,
        "intervention_position": "final prompt token only at pinned resid_post hook",
        "controls": [
            "zero",
            "sae_reconstruction",
            "target_feature_ablation",
            "matched_random",
            "unrelated_feature",
            "negative-alpha sign reversal",
        ],
        "summaries": summaries,
        "causal_admission_rule": (
            "do not interpret a target as causal mediation unless held-out target effects exceed "
            "matched controls, show dose/sign behavior, and preserve the reconstruction baseline"
        ),
    }
    summary_path = manifest_path.parent / "context_steering_summary.json"
    _atomic_json(summary_path, summary)
    manifest["stages"]["steer"] = {
        "status": "complete",
        "n_rows": len(output_rows),
        "n_heldout_prompts": len(heldout),
        "max_baseline_replay_error": max_replay_error,
        "artifacts": {
            rows_path.name: _sha256_file(rows_path),
            summary_path.name: _sha256_file(summary_path),
        },
        "completed_at_utc": _utc_now(),
    }
    _atomic_json(manifest_path, manifest)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stage", choices=("all", "capture", "analyze", "steer"), default="all")
    parser.add_argument("--profile", choices=tuple(PROFILE_STATES_PER_RISK), default="smoke")
    parser.add_argument("--states-per-risk", type=int)
    parser.add_argument("--layer", type=int, required=True, choices=SUPPORTED_LAYERS)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ai_race/configs/experiment/context_skin_invariance.json"),
    )
    parser.add_argument("--base-seed", type=int, default=260726)
    parser.add_argument("--eval-fraction", type=float, default=0.34)
    parser.add_argument("--top-features", type=int, default=3)
    parser.add_argument("--ranking-limit", type=int, default=500)
    parser.add_argument("--min-feature-prevalence", type=float, default=0.05)
    parser.add_argument("--steering-max-rows", type=int, default=32)
    parser.add_argument("--steering-alphas", nargs="+", type=float, default=[-2.0, -1.0, 1.0, 2.0])
    parser.add_argument("--score-replay-tolerance", type=float, default=2e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("bfloat16", "float16", "float32"), default="bfloat16")
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    if args.states_per_risk is not None and args.states_per_risk < 2:
        parser.error("--states-per-risk must be at least 2 for a trajectory split")
    minimum_steering_rows = len(EVALUATION_SKINS) * len(ACTION_CODE_MAPPINGS)
    if args.steering_max_rows < minimum_steering_rows:
        parser.error(
            f"--steering-max-rows must be at least {minimum_steering_rows} to preserve one matched state block"
        )
    if not any(alpha < 0 for alpha in args.steering_alphas) or not any(
        alpha > 0 for alpha in args.steering_alphas
    ):
        parser.error("--steering-alphas must include both negative and positive sign controls")
    return args


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if set(DISCOVERY_SKINS).intersection(EVALUATION_SKINS):
        raise RuntimeError("context discovery/evaluation sets overlap")
    if set((*DISCOVERY_SKINS, *EVALUATION_SKINS)) != set(SKINS):
        raise RuntimeError("context pair split does not cover exactly all registered skins")
    manifest_path, manifest = _prepare_manifest(args)
    needs_model = args.stage in {"all", "capture", "steer"}
    model = sae = None
    if needs_model:
        model, sae, _ = _load_model_and_sae(args, manifest)
        _atomic_json(manifest_path, manifest)
    try:
        if args.stage in {"all", "capture"}:
            _capture(args, model, sae, manifest_path, manifest)
        analysis = None
        if args.stage in {"all", "analyze"}:
            analysis = _analyze(args, manifest_path, manifest)
        if args.stage in {"all", "steer"}:
            if analysis is None:
                analysis_path = manifest_path.parent / "context_analysis.json"
                analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
                if analysis.get("config_fingerprint") != manifest["config_fingerprint"]:
                    raise ValueError("analysis belongs to another run")
            _steer(args, model, sae, manifest_path, manifest, analysis)
        manifest["status"] = "complete"
        manifest["completed_at_utc"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        return 0
    except Exception as error:
        manifest["status"] = "failed"
        manifest["error"] = {"type": type(error).__name__, "message": str(error)}
        manifest["failed_at_utc"] = _utc_now()
        _atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    raise SystemExit(main())
