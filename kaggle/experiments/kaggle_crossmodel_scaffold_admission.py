"""Kaggle RTX scaffold-comprehension admission for protocol-matched checkpoints.

The kernel is intentionally admission-first: it writes complete raw evidence
for every tested checkpoint and never launches gameplay.  A separate kernel
version may scale from ``smoke`` to ``pilot`` only after this artifact passes
coverage, parser, provenance, and GPU checks.
"""
from __future__ import annotations

import gc
import hashlib
import json
import os
import platform
import shutil
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence


# Frozen kernel-version controls. Change these only in a new Kaggle version.
RUN_PROFILE = "smoke"
MODEL_KEYS = ("qwen25_7b",)
REPO_URL = "https://github.com/trungkiet2005/AI_Race_Experiment.git"
REPO_COMMIT = "42102749c4c9615f8dd54bafe229e3d8e32c625a"
SOURCE_DATASET = "daosyduyminh/ai-race-admission-source/1"
SOURCE_ROOT_CANDIDATES = (
    Path("/kaggle/input/datasets/daosyduyminh/ai-race-admission-source"),
    Path("/kaggle/input/ai-race-admission-source"),
)
OUTPUT_ROOT = Path("/kaggle/working/ai_race_results/crossmodel_scaffold_admission")
BATCH_SIZE = 8
MAX_NEW_TOKENS = 16
TEMPERATURE = 0.0
PROFILE_REPETITIONS = {"smoke": 1, "pilot": 5}

MODELS = {
    "qwen25_7b": {
        "source": "qwen-lm/qwen2.5/Transformers/7b-instruct/1",
        "path": Path(
            "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1"
        ),
        "family": "Qwen2.5",
        "short_name": "qwen2.5-7b-instruct",
    },
    "qwen25_14b": {
        "source": "qwen-lm/qwen2.5/Transformers/14b-instruct/1",
        "path": Path(
            "/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b-instruct/1"
        ),
        "family": "Qwen2.5",
        "short_name": "qwen2.5-14b-instruct",
    },
    "gemma2_9b": {
        "source": "google/gemma-2/Transformers/gemma-2-9b-it/2",
        "path": Path(
            "/kaggle/input/models/google/gemma-2/transformers/gemma-2-9b-it/2"
        ),
        "family": "Gemma-2",
        "short_name": "gemma-2-9b-it",
    },
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def write_jsonl(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        "".join(
            json.dumps(row, sort_keys=True, ensure_ascii=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
        newline="\n",
    )
    os.replace(temporary, path)


def artifact(path: Path, root: Path) -> dict[str, Any]:
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def locate_source_repo() -> Path:
    candidates = list(SOURCE_ROOT_CANDIDATES)
    input_root = Path("/kaggle/input")
    if input_root.is_dir():
        candidates.extend(
            path.parent for path in input_root.rglob("ai_race.zip")
        )
        candidates.extend(
            path.parent for path in input_root.rglob("ai_race/__init__.py")
        )
    for candidate in dict.fromkeys(candidates):
        if (candidate / "ai_race").is_dir():
            sys.path.insert(0, str(candidate))
            return candidate
        # Kaggle expands a directory-mode Dataset and strips the uploaded
        # top-level folder, leaving package contents directly at the mount.
        if (candidate / "__init__.py").is_file() and (candidate / "audit").is_dir():
            extracted = Path("/kaggle/working/ai_race_source")
            package = extracted / "ai_race"
            if extracted.exists():
                shutil.rmtree(extracted)
            shutil.copytree(candidate, package)
            sys.path.insert(0, str(extracted))
            return extracted
        archive = candidate / "ai_race.zip"
        if archive.is_file():
            extracted = Path("/kaggle/working/ai_race_source")
            if extracted.exists():
                shutil.rmtree(extracted)
            extracted.mkdir(parents=True)
            with zipfile.ZipFile(archive) as handle:
                handle.extractall(extracted)
            if not (extracted / "ai_race").is_dir():
                raise RuntimeError("Source Dataset archive lacks ai_race package")
            sys.path.insert(0, str(extracted))
            return extracted
    mounted = []
    if input_root.is_dir():
        mounted = sorted(path.as_posix() for path in input_root.iterdir())
    raise FileNotFoundError(
        "Pinned ai-race-admission-source Kaggle Dataset is not mounted; "
        f"input roots={mounted}"
    )


def model_provenance(model: dict[str, Any]) -> dict[str, Any]:
    root = Path(model["path"])
    if not root.is_dir():
        raise FileNotFoundError(f"Kaggle model mount is missing: {root}")
    metadata = {}
    for name in (
        "config.json",
        "generation_config.json",
        "tokenizer_config.json",
        "tokenizer.json",
        "model.safetensors.index.json",
    ):
        path = root / name
        if path.is_file():
            metadata[name] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    weights = [
        {"name": path.name, "bytes": path.stat().st_size}
        for path in sorted(root.glob("*.safetensors"))
    ]
    if not weights:
        raise RuntimeError(f"No safetensor weights found in {root}")
    identity = {
        "kaggle_model_source": model["source"],
        "family": model["family"],
        "short_name": model["short_name"],
        "metadata": metadata,
        "weight_files": weights,
    }
    return {**identity, "digest": canonical_sha256(identity)}


class TransformersGreedyBackend:
    def __init__(self, model_path: Path) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_path, local_files_only=True, trust_remote_code=False
        )
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            local_files_only=True,
            trust_remote_code=False,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            low_cpu_mem_usage=True,
        )
        self.model.eval()

    def _chat(self, prompt: str) -> str:
        messages = [{"role": "user", "content": prompt}]
        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

    def __call__(
        self, prompts: Sequence[str], seeds: Sequence[int] | None = None
    ) -> list[str]:
        if seeds is not None and len(seeds) != len(prompts):
            raise ValueError("Prompt/seed batch lengths differ")
        rendered = [self._chat(prompt) for prompt in prompts]
        encoded = self.tokenizer(
            rendered, return_tensors="pt", padding=True, truncation=False
        )
        device = next(self.model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        input_width = int(encoded["input_ids"].shape[1])
        with self.torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=MAX_NEW_TOKENS,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )
        return self.tokenizer.batch_decode(
            generated[:, input_width:], skip_special_tokens=True
        )

    def close(self) -> None:
        del self.model
        del self.tokenizer
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def run_one(repo: Path, model_key: str) -> dict[str, Any]:
    import torch

    from ai_race.audit.scaffold_comprehension import (
        build_scaffold_probe_requests,
        request_bank_sha256,
        run_scaffold_comprehension,
        scaffold_admission_summary,
    )
    from ai_race.audit.state_scaffold import SCAFFOLD_CONDITIONS
    from ai_race.dataio.config_loader import load_game_config, load_json, validate_experiment
    from ai_race.prompts.context_skins import ACTION_CODE_MAPPINGS

    model = MODELS[model_key]
    provenance = model_provenance(model)
    output = OUTPUT_ROOT / model["short_name"] / RUN_PROFILE
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest: dict[str, Any] = {
        "schema_version": "ai-race-kaggle-crossmodel-scaffold-admission-run-v1",
        "status": "running",
        "evidence_class": "protocol",
        "started_utc": utc_now(),
        "run_profile": RUN_PROFILE,
        "repo_url": REPO_URL,
        "repo_commit": REPO_COMMIT,
        "source_dataset": SOURCE_DATASET,
        "model": provenance,
        "language": "en",
        "decoding": {
            "temperature": TEMPERATURE,
            "max_new_tokens": MAX_NEW_TOKENS,
            "do_sample": False,
            "batch_size": BATCH_SIZE,
            "sampling_seed_applied": False,
            "seed_note": "Greedy decoding; request seeds retained as CRN identifiers only.",
        },
        "hardware": {
            "hostname": platform.node(),
            "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "",
            "gpu_count": torch.cuda.device_count(),
            "python": platform.python_version(),
            "torch": torch.__version__,
        },
        "artifacts": {},
        "error": None,
    }
    write_json(manifest_path, manifest)
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required")
        config_path = repo / "ai_race/configs/experiment/state_scaffold_factorial.json"
        experiment = validate_experiment(load_json(config_path))
        conditions = list(experiment["scaffoldConditions"])
        mappings = list(experiment["actionCodeMappings"])
        if conditions != list(SCAFFOLD_CONDITIONS):
            raise RuntimeError("Scaffold condition order drifted")
        if mappings != list(ACTION_CODE_MAPPINGS):
            raise RuntimeError("Action-code mapping order drifted")
        game = load_game_config(
            repo / "ai_race/configs/game/ai_race_risk_60.json",
            model=model["short_name"],
        )
        repetitions = PROFILE_REPETITIONS[RUN_PROFILE]
        requests = build_scaffold_probe_requests(
            game,
            condition_ids=conditions,
            mapping_ids=mappings,
            repetitions=repetitions,
            seed=int(experiment["seed"]),
        )
        expected = len(conditions) * len(mappings) * repetitions * 16
        if len(requests) != expected:
            raise RuntimeError(f"Request-bank coverage mismatch: {len(requests)} != {expected}")
        backend = TransformersGreedyBackend(Path(model["path"]))
        try:
            probe = "Return exactly one line: ANSWER: YES"
            if backend([probe])[0] != backend([probe])[0]:
                raise RuntimeError("Greedy reproducibility probe failed")
            rows = run_scaffold_comprehension(
                requests, backend, batch_size=BATCH_SIZE
            )
        finally:
            backend.close()
        summary = scaffold_admission_summary(
            rows,
            game,
            condition_ids=conditions,
            mapping_ids=mappings,
            repetitions=repetitions,
        )
        raw_path = output / "comprehension_raw.jsonl"
        write_jsonl(raw_path, rows)
        admission_path = output / "admission.json"
        admission = {
            "schema_version": "ai-race-kaggle-crossmodel-scaffold-admission-v1",
            "status": "complete",
            "protocol": "ai-race-state-scaffold-comprehension-v1",
            "generated_utc": utc_now(),
            "model": provenance,
            "language": "en",
            "request_bank_sha256": request_bank_sha256(requests),
            "experiment_config_sha256": file_sha256(config_path),
            "decoding": manifest["decoding"],
            **summary,
            "artifacts": {"comprehension_raw": artifact(raw_path, output)},
        }
        write_json(admission_path, admission)
        manifest.update(
            status="completed",
            completed_utc=utc_now(),
            evidence_class=(
                "admitted" if admission["passed"]
                else "diagnostic_comprehension_failed"
            ),
            expected_requests=expected,
            n_requests=len(rows),
            admission_passed=bool(admission["passed"]),
            request_bank_sha256=admission["request_bank_sha256"],
            artifacts={
                "comprehension_raw": artifact(raw_path, output),
                "admission": artifact(admission_path, output),
            },
        )
        write_json(manifest_path, manifest)
    except Exception as error:
        manifest.update(
            status="failed",
            evidence_class="failed",
            completed_utc=utc_now(),
            error=f"{type(error).__name__}: {error}",
        )
        write_json(manifest_path, manifest)
        raise
    return manifest


def main() -> None:
    if RUN_PROFILE not in PROFILE_REPETITIONS:
        raise ValueError(f"Unknown RUN_PROFILE={RUN_PROFILE!r}")
    unknown = set(MODEL_KEYS) - set(MODELS)
    if unknown:
        raise ValueError(f"Unknown model keys: {sorted(unknown)}")
    repo = locate_source_repo()
    summaries = [run_one(repo, model_key) for model_key in MODEL_KEYS]
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    write_json(
        OUTPUT_ROOT / "suite_manifest.json",
        {
            "schema_version": "ai-race-kaggle-crossmodel-scaffold-suite-v1",
            "status": "completed",
            "run_profile": RUN_PROFILE,
            "repo_commit": REPO_COMMIT,
            "source_dataset": SOURCE_DATASET,
            "models": [summary["model"] for summary in summaries],
            "admission_passed": {
                summary["model"]["short_name"]: summary["admission_passed"]
                for summary in summaries
            },
        },
    )
    shutil.make_archive(
        "/kaggle/working/ai_race_crossmodel_scaffold_admission",
        "zip",
        OUTPUT_ROOT.parent,
        OUTPUT_ROOT.name,
    )
    print(json.dumps({"status": "completed", "models": len(summaries)}, indent=2))


if __name__ == "__main__":
    main()
