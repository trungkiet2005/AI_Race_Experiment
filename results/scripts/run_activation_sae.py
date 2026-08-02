#!/usr/bin/env python3
"""Activation-level SAFE/UNSAFE analysis with pinned pretrained FAST SAEs.

This is intentionally separate from ``explain_action_sparse_autoencoder.py``:
that script factorizes engineered/text features, whereas this script replays the
causal pre-action prefix through the exact base LLM and encodes residual-stream
activations with pretrained SAELens JumpReLU dictionaries.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import subprocess
import sys
from typing import Any, Sequence

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_race.xai.activation_sae import (
    ACTIVATION_SAE_SCHEMA_VERSION,
    build_capture_prefix,
    cross_split_duplicate_audit,
    grouped_train_eval_split,
    load_decision_examples,
    race_prefix_components,
    sha256_file,
    sha256_text,
)


DEFAULT_PRESET = ROOT / "ai_race" / "xai" / "presets" / "qwen25_7b_instruct_fast.json"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", action="append", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--preset", default=str(DEFAULT_PRESET))
    parser.add_argument("--layers", nargs="*", type=int)
    parser.add_argument("--capture-position", choices=("pre_action", "prompt_last"), default="pre_action")
    parser.add_argument("--eval-fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--top-features-per-sample", type=int, default=64)
    parser.add_argument("--n-label-shuffles", type=int, default=20)
    parser.add_argument(
        "--probe-max-features",
        type=int,
        default=4096,
        help="Train-only variance screen before linear probes; <=0 keeps every SAE feature.",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dtype", choices=("float32", "float16", "bfloat16"), default="bfloat16")
    parser.add_argument("--cache-dir")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--fail-on-cross-split-duplicates",
        action="store_true",
        help="Optional stricter generalization test; repeated race states often create legitimate duplicate prompts.",
    )
    parser.add_argument(
        "--decision-model-digest",
        help="Exact digest/revision of the model that generated turns.jsonl (required unless --dry-run).",
    )
    parser.add_argument(
        "--allow-model-provenance-mismatch",
        action="store_true",
        help="Exploratory only: labels may come from a different revision/quantization than the attribution model.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate inputs/split/provenance without loading a model.")
    return parser.parse_args(argv)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _git_revision() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _version(name: str) -> str | None:
    try:
        from importlib.metadata import version

        return version(name)
    except Exception:  # noqa: BLE001
        return None


def _load_preset(path: str | Path) -> dict[str, Any]:
    preset_path = Path(path)
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    required = {
        "model_repo",
        "model_revision",
        "sae_repo",
        "sae_revision",
        "sae_lens_version",
        "layers",
        "sae_id_template",
        "activation_width",
        "feature_width",
    }
    missing = sorted(required - set(preset))
    if missing:
        raise ValueError(f"preset missing keys: {missing}")
    preset["preset_path"] = str(preset_path.resolve())
    preset["preset_sha256"] = sha256_file(preset_path)
    return preset


def _validate_provenance(args: argparse.Namespace, preset: dict[str, Any]) -> dict[str, Any]:
    supplied = str(args.decision_model_digest or "").strip()
    exact = supplied == str(preset["model_revision"])
    if not args.dry_run and not supplied:
        raise ValueError(
            "--decision-model-digest is required: activation attribution is invalid if labels "
            "were generated by an unrecorded model revision/quantization"
        )
    if supplied and not exact and not args.allow_model_provenance_mismatch:
        raise ValueError(
            "decision-model digest does not match the pinned attribution model revision; "
            "pass --allow-model-provenance-mismatch only for an explicitly exploratory replay"
        )
    return {
        "decision_model_digest": supplied or None,
        "attribution_model_revision": preset["model_revision"],
        "exact_model_match": exact,
        "exploratory_mismatch_override": bool(args.allow_model_provenance_mismatch),
        "claim_scope": "same-model activation attribution" if exact else "cross-model exploratory association only",
    }


def _load_saes(preset: dict[str, Any], layers: list[int], args: argparse.Namespace) -> dict[int, Any]:
    from huggingface_hub import snapshot_download
    from sae_lens import SAE

    installed = _version("sae-lens")
    if installed != str(preset["sae_lens_version"]):
        raise RuntimeError(
            f"SAELens must be exactly {preset['sae_lens_version']} for this artifact; installed={installed!r}"
        )
    loaded: dict[int, Any] = {}
    for layer in layers:
        sae_id = str(preset["sae_id_template"]).format(layer=layer)
        snapshot = snapshot_download(
            repo_id=str(preset["sae_repo"]),
            revision=str(preset["sae_revision"]),
            allow_patterns=[f"{sae_id}/*"],
            cache_dir=args.cache_dir,
            local_files_only=args.local_files_only,
        )
        artifact_dir = Path(snapshot) / sae_id
        if not (artifact_dir / "cfg.json").is_file() or not (artifact_dir / "sae_weights.safetensors").is_file():
            raise FileNotFoundError(f"incomplete pretrained SAE artifact: {artifact_dir}")
        sae = SAE.load_from_pretrained(str(artifact_dir), device=args.device)
        cfg = json.loads((artifact_dir / "cfg.json").read_text(encoding="utf-8"))
        expected_hook = f"blocks.{layer}.hook_resid_post"
        if cfg.get("hook_name") != expected_hook:
            raise ValueError(f"SAE hook mismatch: {cfg.get('hook_name')!r} != {expected_hook!r}")
        if int(cfg.get("d_in", -1)) != int(preset["activation_width"]):
            raise ValueError("SAE activation width does not match pinned preset")
        loaded[layer] = sae
    return loaded


def _load_model(preset: dict[str, Any], args: argparse.Namespace) -> Any:
    import torch
    from transformer_lens import HookedTransformer

    dtype = getattr(torch, args.dtype)
    kwargs = {
        "revision": str(preset["model_revision"]),
        "dtype": dtype,
        "device": args.device,
        "cache_dir": args.cache_dir,
        "local_files_only": args.local_files_only,
        "trust_remote_code": False,
    }
    # No folding/centering: the pretrained SAE cfg records
    # model_from_pretrained_kwargs.center_writing_weights=false.
    model = HookedTransformer.from_pretrained_no_processing(
        str(preset["model_repo"]), **kwargs
    )
    model.eval()
    if int(model.cfg.d_model) != int(preset["activation_width"]):
        raise ValueError("base model d_model does not match pretrained SAE d_in")
    return model


def _tokenize(model: Any, text: str, context_size: int) -> tuple[Any, bool]:
    tokens = model.to_tokens(text, prepend_bos=False)
    truncated = int(tokens.shape[-1]) > context_size
    if truncated:
        tokens = tokens[:, -context_size:]
    return tokens, truncated


def _capture_and_encode(
    model: Any,
    saes: dict[int, Any],
    examples: list[Any],
    capture_texts: list[str],
    splits: list[str],
    preset: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[int, np.ndarray]]:
    import torch

    layers = sorted(saes)
    hooks = [f"blocks.{layer}.hook_resid_post" for layer in layers]
    sample_rows: list[dict[str, Any]] = []
    sparse_rows: list[dict[str, Any]] = []
    reconstruction_rows: list[dict[str, Any]] = []
    dense_codes: dict[int, list[np.ndarray]] = {layer: [] for layer in layers}
    squared_error: dict[tuple[int, str], float] = {}
    squared_input: dict[tuple[int, str], float] = {}
    cosine_sum: dict[tuple[int, str], float] = {}
    l0_sum: dict[tuple[int, str], float] = {}
    counts: dict[tuple[int, str], int] = {}

    for index, (example, text, split) in enumerate(zip(examples, capture_texts, splits)):
        tokens, truncated = _tokenize(model, text, int(preset["context_size"]))
        with torch.inference_mode():
            _, cache = model.run_with_cache(tokens, names_filter=lambda name: name in hooks)
        sample_row = example.metadata()
        sample_row.update(
            {
                "split": split,
                "capture_position": args.capture_position,
                "rendered_capture_sha256": sha256_text(text),
                "n_tokens": int(tokens.shape[-1]),
                "left_truncated": int(truncated),
            }
        )
        sample_rows.append(sample_row)
        for layer in layers:
            hook_name = f"blocks.{layer}.hook_resid_post"
            activation = cache[hook_name][0, -1, :].to(args.device)
            if int(activation.numel()) != int(preset["activation_width"]):
                raise ValueError(f"layer {layer}: unexpected activation width {activation.numel()}")
            with torch.inference_mode():
                code = saes[layer].encode(activation.unsqueeze(0)).squeeze(0)
                reconstruction = saes[layer].decode(code.unsqueeze(0)).squeeze(0)
            code_cpu = code.detach().float().cpu().numpy()
            dense_codes[layer].append(code_cpu)
            activation_f = activation.detach().float()
            reconstruction_f = reconstruction.detach().float()
            key = (layer, split)
            error = float(torch.square(activation_f - reconstruction_f).sum().cpu())
            energy = float(torch.square(activation_f).sum().cpu())
            cosine = float(torch.nn.functional.cosine_similarity(activation_f, reconstruction_f, dim=0).cpu())
            l0 = int(torch.count_nonzero(code).cpu())
            squared_error[key] = squared_error.get(key, 0.0) + error
            squared_input[key] = squared_input.get(key, 0.0) + energy
            cosine_sum[key] = cosine_sum.get(key, 0.0) + cosine
            l0_sum[key] = l0_sum.get(key, 0.0) + l0
            counts[key] = counts.get(key, 0) + 1
            top_n = min(args.top_features_per_sample, l0, code_cpu.size)
            if top_n:
                indices = np.argpartition(code_cpu, -top_n)[-top_n:]
                indices = indices[np.argsort(code_cpu[indices])[::-1]]
                for rank, feature_id in enumerate(indices, start=1):
                    value = float(code_cpu[feature_id])
                    if value <= 0:
                        continue
                    sparse_rows.append(
                        {
                            "sample_id": example.sample_id,
                            "layer": layer,
                            "feature_id": int(feature_id),
                            "activation": value,
                            "rank": rank,
                            "label_unsafe": example.label_unsafe,
                            "split": split,
                        }
                    )
        del cache
        if (index + 1) % 25 == 0:
            print(f"captured {index + 1}/{len(examples)} decisions", flush=True)

    for (layer, split), count in sorted(counts.items()):
        reconstruction_rows.append(
            {
                "layer": layer,
                "split": split,
                "n": count,
                "normalized_mse": squared_error[(layer, split)] / max(squared_input[(layer, split)], 1e-12),
                "mean_cosine_similarity": cosine_sum[(layer, split)] / count,
                "mean_l0": l0_sum[(layer, split)] / count,
            }
        )
    return sample_rows, sparse_rows, reconstruction_rows, {
        layer: np.stack(rows) for layer, rows in dense_codes.items()
    }


def _probe_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import average_precision_score, balanced_accuracy_score, roc_auc_score

    predicted = (scores >= 0.0).astype(int)
    return {
        "roc_auc": float(roc_auc_score(y_true, scores)),
        "average_precision": float(average_precision_score(y_true, scores)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, predicted)),
    }


def _fit_probe(
    train_x: np.ndarray, train_y: np.ndarray, eval_x: np.ndarray, seed: int
) -> tuple[np.ndarray, dict[str, Any]]:
    from sklearn.linear_model import RidgeClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler(with_mean=False)
    train_scaled = scaler.fit_transform(train_x)
    eval_scaled = scaler.transform(eval_x)
    model = RidgeClassifier(
        alpha=1.0,
        class_weight="balanced",
        solver="lsqr",
        tol=1e-4,
    )
    del seed
    model.fit(train_scaled, train_y)
    diagnostics = {
        "solver": "ridge_lsqr",
        "alpha": 1.0,
        "tolerance": 1e-4,
        "convergence_warning_count": 0,
    }
    return np.asarray(model.decision_function(eval_scaled)), diagnostics


def _run_probes(
    codes: dict[int, np.ndarray], labels: np.ndarray, splits: list[str], args: argparse.Namespace
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    train = np.asarray([value == "train" for value in splits])
    evaluation = ~train
    if len(np.unique(labels[train])) < 2 or len(np.unique(labels[evaluation])) < 2:
        raise ValueError("both grouped splits must contain SAFE and UNSAFE labels")
    metric_rows: list[dict[str, Any]] = []
    control_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(args.seed)
    for layer, matrix in sorted(codes.items()):
        if args.probe_max_features > 0 and matrix.shape[1] > args.probe_max_features:
            train_variance = np.var(matrix[train], axis=0, dtype=np.float64)
            selected_features = np.argpartition(
                train_variance, -args.probe_max_features
            )[-args.probe_max_features :]
        else:
            selected_features = np.arange(matrix.shape[1])
        train_matrix = matrix[train][:, selected_features]
        eval_matrix = matrix[evaluation][:, selected_features]
        print(
            f"probing layer {layer} with {len(selected_features)}/{matrix.shape[1]} "
            "train-selected features",
            flush=True,
        )
        probabilities, diagnostics = _fit_probe(
            train_matrix, labels[train], eval_matrix, args.seed
        )
        metric_rows.append(
            {
                "layer": layer,
                "representation": "pretrained_sae",
                "n_probe_features": int(len(selected_features)),
                "feature_screen": "train_only_variance",
                **diagnostics,
                **_probe_metrics(labels[evaluation], probabilities),
            }
        )
        for shuffle_index in range(args.n_label_shuffles):
            shuffled = labels[train].copy()
            rng.shuffle(shuffled)
            try:
                null_probabilities, null_diagnostics = _fit_probe(
                    train_matrix, shuffled, eval_matrix, args.seed + shuffle_index + 1
                )
                metrics = _probe_metrics(labels[evaluation], null_probabilities)
            except ValueError:
                metrics = {"roc_auc": float("nan"), "average_precision": float("nan"), "balanced_accuracy": float("nan")}
                null_diagnostics = {
                    "solver": "ridge_lsqr",
                    "alpha": 1.0,
                    "tolerance": 1e-4,
                    "convergence_warning_count": 0,
                }
            control_rows.append(
                {
                    "layer": layer,
                    "control": "shuffled_train_labels",
                    "iteration": shuffle_index,
                    "n_probe_features": int(len(selected_features)),
                    "feature_screen": "train_only_variance",
                    **null_diagnostics,
                    **metrics,
                }
            )
        print(f"completed probes for layer {layer}", flush=True)
    return metric_rows, control_rows


def _feature_associations(codes: dict[int, np.ndarray], labels: np.ndarray, splits: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    split_values = np.asarray(splits)
    for layer, matrix in sorted(codes.items()):
        for split in ("train", "eval"):
            subset = split_values == split
            safe = matrix[subset & (labels == 0)]
            unsafe = matrix[subset & (labels == 1)]
            if not len(safe) or not len(unsafe):
                continue
            safe_mean = safe.mean(axis=0)
            unsafe_mean = unsafe.mean(axis=0)
            difference = unsafe_mean - safe_mean
            nonzero = np.flatnonzero((safe_mean > 0) | (unsafe_mean > 0))
            order = nonzero[np.argsort(np.abs(difference[nonzero]))[::-1]][:500]
            for rank, feature in enumerate(order, start=1):
                rows.append(
                    {
                        "layer": layer,
                        "split": split,
                        "rank_abs_mean_difference": rank,
                        "feature_id": int(feature),
                        "mean_safe": float(safe_mean[feature]),
                        "mean_unsafe": float(unsafe_mean[feature]),
                        "unsafe_minus_safe": float(difference[feature]),
                        "prevalence_safe": float(np.mean(safe[:, feature] > 0)),
                        "prevalence_unsafe": float(np.mean(unsafe[:, feature] > 0)),
                    }
                )
    return rows


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    preset = _load_preset(args.preset)
    layers = sorted(args.layers if args.layers else [int(value) for value in preset["layers"]])
    unsupported = sorted(set(layers) - {int(value) for value in preset["layers"]})
    if unsupported:
        raise ValueError(f"no pinned pretrained FAST SAE for layers {unsupported}")
    provenance = _validate_provenance(args, preset)
    examples, input_audit = load_decision_examples(args.input_root)
    if args.max_samples:
        rng = random.Random(args.seed)
        selected = sorted(rng.sample(range(len(examples)), min(args.max_samples, len(examples))))
        examples = [examples[index] for index in selected]
    split_hashes = [
        item.prompt_sha256
        if args.capture_position == "prompt_last"
        else item.capture_prefix_sha256
        for item in examples
    ]
    split_groups = race_prefix_components(
        [item.game_id for item in examples], split_hashes
    )
    splits = grouped_train_eval_split(
        [item.label_unsafe for item in examples],
        split_groups,
        eval_fraction=args.eval_fraction,
        seed=args.seed,
    )
    duplicate_audit = cross_split_duplicate_audit(
        split_hashes, splits
    )
    if duplicate_audit["n_cross_split_duplicate_prefixes"] and args.fail_on_cross_split_duplicates:
        raise ValueError(
            "identical causal prefixes cross train/eval; regroup the experiment or disable "
            "--fail-on-cross-split-duplicates and disclose the robustness limitation"
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    base_manifest = {
        "schema_version": ACTIVATION_SAE_SCHEMA_VERSION,
        "status": "validated" if args.dry_run else "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_revision": _git_revision(),
        "command": sys.argv,
        "preset": preset,
        "provenance": provenance,
        "input_audit": input_audit,
        "split_audit": {
            "group_key": "connected_component(game_id, exact_causal_prefix_sha256)",
            "eval_fraction_requested": args.eval_fraction,
            "n_train": splits.count("train"),
            "n_eval": splits.count("eval"),
            "n_train_games": len({item.game_id for item, split in zip(examples, splits) if split == "train"}),
            "n_eval_games": len({item.game_id for item, split in zip(examples, splits) if split == "eval"}),
            "n_connected_components": len(set(split_groups)),
            **duplicate_audit,
        },
        "capture": {
            "position": args.capture_position,
            "target_label_excluded_from_prefix": True,
            "layers": layers,
            "semantics": preset["capture_semantics"],
            "context_size": preset["context_size"],
        },
        "probe": {
            "max_features": args.probe_max_features,
            "feature_screen": "train_only_variance",
            "screen_fit_partition": "train_only",
            "n_label_shuffles": args.n_label_shuffles,
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "torch": _version("torch"),
            "transformer_lens": _version("transformer-lens"),
            "sae_lens": _version("sae-lens"),
            "transformers": _version("transformers"),
        },
        "limitations": [
            "SAE features are hypotheses, not automatically monosemantic concepts.",
            "Association/probe accuracy does not establish a causal decision mechanism.",
            "At temperature > 0, a pre-action activation represents propensity, not the sampled RNG draw.",
            "Feature discovery on train and confirmation on grouped eval must be reported separately.",
        ],
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
    if args.dry_run:
        print(json.dumps(base_manifest["split_audit"], indent=2))
        return 0

    model = _load_model(preset, args)
    tokenizer = model.tokenizer
    capture_texts = [
        build_capture_prefix(
            tokenizer,
            item.prompt,
            item.response_prefix,
            position=args.capture_position,
        )
        for item in examples
    ]
    saes = _load_saes(preset, layers, args)
    sample_rows, sparse_rows, reconstruction_rows, codes = _capture_and_encode(
        model, saes, examples, capture_texts, splits, preset, args
    )
    labels = np.asarray([item.label_unsafe for item in examples], dtype=np.int64)
    probe_rows, control_rows = _run_probes(codes, labels, splits, args)
    association_rows = _feature_associations(codes, labels, splits)
    _write_csv(output_dir / "samples.csv", sample_rows)
    _write_csv(output_dir / "sparse_codes_topk.csv", sparse_rows)
    _write_csv(output_dir / "reconstruction_metrics.csv", reconstruction_rows)
    _write_csv(output_dir / "probe_metrics.csv", probe_rows)
    _write_csv(output_dir / "negative_controls.csv", control_rows)
    _write_csv(output_dir / "feature_action_associations.csv", association_rows)
    base_manifest["status"] = "complete"
    base_manifest["completed_at_utc"] = datetime.now(timezone.utc).isoformat()
    base_manifest["artifacts"] = {
        name: {"sha256": sha256_file(output_dir / name), "bytes": (output_dir / name).stat().st_size}
        for name in (
            "samples.csv",
            "sparse_codes_topk.csv",
            "reconstruction_metrics.csv",
            "probe_metrics.csv",
            "negative_controls.csv",
            "feature_action_associations.csv",
        )
    }
    manifest_path.write_text(json.dumps(base_manifest, indent=2), encoding="utf-8")
    print(f"wrote activation-level SAE artifacts to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
