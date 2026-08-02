"""Run one BF16 scaffold-admission job on a GreenNode GPU pod.

This wrapper deliberately stops before live gameplay.  It verifies the
allocated GPU, resolves one official Hugging Face revision into the shared
persistent cache, and reuses the evidence-complete Kaggle admission runner.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from kaggle.experiments import kaggle_crossmodel_scaffold_admission as admission


DEFAULT_OUTPUT = Path(
    "/network-volume/icse27/ai_race_results/capacity_family_admission"
)
DEFAULT_CACHE = Path("/network-volume/icse27/huggingface")
MIN_COMPUTE_CAPABILITY = (8, 0)

GREENNODE_MODELS: dict[str, dict[str, Any]] = {
    "qwen25_7b": {
        "repo_id": "Qwen/Qwen2.5-7B-Instruct",
        "revision": "a09a35458c702b33eeacc393d103063234e8bc28",
        "family": "Qwen2.5",
        "short_name": "qwen2.5-7b-instruct",
        "min_free_vram_gib": 18.0,
        "batch_size": 1,
    },
    "qwen25_14b": {
        "repo_id": "Qwen/Qwen2.5-14B-Instruct",
        "revision": "cf98f3b3bbb457ad9e2bb7baf9a0125b6b88caa8",
        "family": "Qwen2.5",
        "short_name": "qwen2.5-14b-instruct",
        "min_free_vram_gib": 36.0,
        "batch_size": 2,
    },
    "qwen25_32b": {
        "repo_id": "Qwen/Qwen2.5-32B-Instruct",
        "revision": "5ede1c97bbab6ce5cda5812749b4c0bdf79b18dd",
        "family": "Qwen2.5",
        "short_name": "qwen2.5-32b-instruct",
        "min_free_vram_gib": 72.0,
        "batch_size": 2,
    },
    "gemma2_9b": {
        "repo_id": "google/gemma-2-9b-it",
        "revision": "11c9b309abf73637e4b6f9a3fa1e92e615547819",
        "family": "Gemma-2",
        "short_name": "gemma-2-9b-it",
        "min_free_vram_gib": 22.0,
        "batch_size": 1,
        "gated": True,
    },
    "mistral7_01": {
        "repo_id": "mistralai/Mistral-7B-Instruct-v0.1",
        "revision": "ec5deb64f2c6e6fa90c1abf74a91d5c93a9669ca",
        "family": "Mistral-7B",
        "short_name": "mistral-7b-instruct-v0.1",
        "min_free_vram_gib": 18.0,
        "batch_size": 1,
    },
}


def evaluate_greennode_hardware_gate(
    model_key: str,
    *,
    cuda_available: bool,
    gpu_name: str,
    gpu_count: int,
    free_vram_bytes: int,
    total_vram_bytes: int,
    compute_capability: tuple[int, int],
    bf16_supported: bool,
) -> dict[str, Any]:
    """Return a fail-closed receipt before model download or inference."""
    if model_key not in GREENNODE_MODELS:
        raise ValueError(f"Unknown model key {model_key!r}")
    model = GREENNODE_MODELS[model_key]
    free_gib = free_vram_bytes / (1024**3)
    total_gib = total_vram_bytes / (1024**3)
    checks = {
        "cuda_available": bool(cuda_available),
        "single_gpu": gpu_count == 1,
        "free_vram": free_gib >= float(model["min_free_vram_gib"]),
        "compute_capability": tuple(compute_capability) >= MIN_COMPUTE_CAPABILITY,
        "bf16_supported": bool(bf16_supported),
    }
    return {
        "passed": all(checks.values()),
        "model_key": model_key,
        "model_short_name": model["short_name"],
        "checks": checks,
        "required": {
            "gpu_count": 1,
            "min_free_vram_gib": model["min_free_vram_gib"],
            "min_compute_capability": list(MIN_COMPUTE_CAPABILITY),
            "bf16": True,
        },
        "observed": {
            "cuda_available": bool(cuda_available),
            "gpu_name": gpu_name,
            "gpu_count": gpu_count,
            "free_vram_gib": free_gib,
            "total_vram_gib": total_gib,
            "compute_capability": list(compute_capability),
            "bf16_supported": bool(bf16_supported),
        },
    }


def runtime_hardware_gate(model_key: str) -> dict[str, Any]:
    import torch

    available = bool(torch.cuda.is_available())
    if available:
        name = str(torch.cuda.get_device_name(0))
        count = int(torch.cuda.device_count())
        properties = torch.cuda.get_device_properties(0)
        total = int(properties.total_memory)
        free, _ = map(int, torch.cuda.mem_get_info(0))
        capability = tuple(map(int, torch.cuda.get_device_capability(0)))
        bf16 = bool(torch.cuda.is_bf16_supported())
    else:
        name, count, free, total, capability, bf16 = "", 0, 0, 0, (0, 0), False
    return {
        **evaluate_greennode_hardware_gate(
            model_key,
            cuda_available=available,
            gpu_name=name,
            gpu_count=count,
            free_vram_bytes=free,
            total_vram_bytes=total,
            compute_capability=capability,
            bf16_supported=bf16,
        ),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
    }


def repo_commit(repo: Path) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def resolve_model(model_key: str, cache: Path, local_path: Path | None) -> Path:
    if local_path is not None:
        resolved = local_path.resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(f"Local model path does not exist: {resolved}")
        return resolved
    from huggingface_hub import snapshot_download

    model = GREENNODE_MODELS[model_key]
    return Path(
        snapshot_download(
            repo_id=model["repo_id"],
            revision=model["revision"],
            cache_dir=cache,
            allow_patterns=(
                "*.json",
                "*.model",
                "*.safetensors",
                "*.safetensors.index.json",
                "tokenizer*",
                "special_tokens_map.json",
            ),
        )
    )


def blocked_receipt(
    output: Path, model_key: str, hardware: dict[str, Any]
) -> Path:
    model = GREENNODE_MODELS[model_key]
    target = output / model["short_name"] / "hardware_gate.json"
    admission.write_json(
        target,
        {
            "schema_version": "ai-race-greennode-hardware-gate-v1",
            "status": "blocked_hardware",
            "evidence_class": "blocked",
            "generated_utc": datetime.now(timezone.utc).isoformat(),
            "model": {
                "repo_id": model["repo_id"],
                "revision": model["revision"],
            },
            "hardware": hardware,
            "n_requests": 0,
        },
    )
    return target


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-key", required=True, choices=sorted(GREENNODE_MODELS))
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cache-dir", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--local-model-path", type=Path)
    parser.add_argument("--run-profile", choices=("smoke", "pilot"), default="smoke")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo = args.repo_root.resolve()
    if not (repo / "ai_race").is_dir():
        raise FileNotFoundError(f"Repo root lacks ai_race package: {repo}")
    hardware = runtime_hardware_gate(args.model_key)
    if not hardware["passed"]:
        path = blocked_receipt(args.output_root, args.model_key, hardware)
        print(json.dumps({"status": "blocked_hardware", "receipt": str(path)}, indent=2))
        return 2

    model_path = resolve_model(args.model_key, args.cache_dir, args.local_model_path)
    model = GREENNODE_MODELS[args.model_key]
    admission.MODELS[args.model_key] = {
        "source": f"hf://{model['repo_id']}@{model['revision']}",
        "path": model_path,
        "family": model["family"],
        "short_name": model["short_name"],
        "min_vram_gib": model["min_free_vram_gib"],
        "batch_size": model["batch_size"],
    }
    admission.OUTPUT_ROOT = args.output_root.resolve()
    admission.RUN_PROFILE = args.run_profile
    admission.REPO_COMMIT = repo_commit(repo)
    admission.SOURCE_DATASET = "greennode-local-checkout"
    result = admission.run_one(repo, args.model_key, hardware)
    print(json.dumps(result, indent=2))
    return 0 if result.get("status") == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
