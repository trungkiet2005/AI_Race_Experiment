"""Run the isolated context-recognition audit on GreenNode Ollama pods.

This runner never invokes gameplay or comprehension.  It writes a separate
artifact tree so recognition questions cannot enter agent prompts or admission
gates.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
from pathlib import Path
from typing import Any, Iterable, Optional

from ai_race.audit.context_recognition import (
    CONTEXT_RECOGNITION_PROTOCOL,
    run_recognition_matrix,
    summarize_recognition,
)
from ai_race.dataio.config_loader import load_game_config, load_json
from ai_race.prompts.context_skins import (
    ACTION_CODE_MAPPINGS,
    ACTION_CODE_PROTOCOL,
    CONTEXT_SKIN_PROTOCOL,
    SKINS,
    context_skin_sha256,
)
from kaggle.experiments.greennode_prompt_sensitivity import (
    DEFAULT_ENDPOINT,
    DEFAULT_MODEL,
    OllamaBatchBackend,
    atomic_json,
    gpu_name,
    model_provenance,
    request_json,
    sha256_file,
    source_tree_sha256,
    utc_now,
)


LANE_SKINS = {
    "a": [
        "technology_race",
        "logistics_contract",
        "hospital_deployment",
        "robotic_expedition",
    ],
    "b": [
        "abstract_contest",
        "crystal_guild_contract",
        "colony_life_support",
        "fictional_cartography",
    ],
}
PROFILE_REPETITIONS = {"confirm": 1, "smoke": 2, "pilot": 20}


def _atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    _atomic_text(
        path,
        "".join(
            json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n" for row in rows
        ),
    )


def _artifact(path: Path) -> dict[str, Any]:
    return {"path": path.name, "bytes": path.stat().st_size, "sha256": sha256_file(path)}


class MockRecognitionBackend:
    """Deterministic schema smoke backend; its outputs are never empirical evidence."""

    def __call__(
        self, prompts: list[str], seeds: Optional[list[int]] = None
    ) -> list[str]:
        resolved = list(seeds or range(len(prompts)))
        if len(resolved) != len(prompts):
            raise ValueError("Mock backend requires one seed per prompt")
        categories = (
            ("no_recognition", None),
            ("uncertain", None),
            ("generic_structural_resemblance", None),
        )
        outputs = []
        for seed in resolved:
            category, candidate = categories[int(seed) % len(categories)]
            outputs.append(
                json.dumps(
                    {
                        "recognition_class": category,
                        "candidate": candidate,
                        "confidence": "low",
                    },
                    separators=(",", ":"),
                )
            )
        return outputs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lane", choices=sorted(LANE_SKINS), required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_REPETITIONS), default="smoke")
    parser.add_argument("--backend", choices=("ollama", "mock"), default="ollama")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("ai_race/configs/experiment/context_skin_invariance.json"),
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=96)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-parse-retries", type=int, default=2)
    parser.add_argument("--required-gpu", default="H100")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    config_path = args.config if args.config.is_absolute() else root / args.config
    experiment = load_json(config_path)
    configured_skins = list(experiment.get("contextSkins", []))
    partition = LANE_SKINS["a"] + LANE_SKINS["b"]
    if set(configured_skins) != set(SKINS) or set(partition) != set(SKINS):
        raise RuntimeError("Config and recognition lanes must cover all skins")
    if set(LANE_SKINS["a"]) & set(LANE_SKINS["b"]):
        raise RuntimeError("Recognition lanes overlap")

    game_names = list(experiment["games"])
    game_configs = [
        load_game_config(
            root / "ai_race" / "configs" / "game" / f"{name}.json",
            model=args.model,
        )
        for name in game_names
    ]
    recognition_config = min(
        game_configs, key=lambda item: abs(item.max_private_risk - 0.6)
    )
    repetitions = PROFILE_REPETITIONS[args.profile]
    run_spec = {
        "protocol": CONTEXT_RECOGNITION_PROTOCOL,
        "lane": args.lane,
        "profile": args.profile,
        "backend": args.backend,
        "skins": LANE_SKINS[args.lane],
        "repetitions": repetitions,
        "seed": int(experiment["seed"]),
        "model": args.model,
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "max_parse_retries": args.max_parse_retries,
    }
    run_spec_sha256 = hashlib.sha256(
        json.dumps(run_spec, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    manifest_path = output_root / "run_manifest.json"
    if manifest_path.is_file() and not args.no_resume:
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            if previous.get("run_spec_sha256") != run_spec_sha256:
                raise RuntimeError("Completed output belongs to another run specification")
            print(f"[resume] completed recognition audit: {output_root}", flush=True)
            return

    core_path = root / "ai_race" / "audit" / "context_recognition.py"
    skins_path = root / "ai_race" / "prompts" / "context_skins.py"
    runner_path = Path(__file__).resolve()
    source_files = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in (core_path, skins_path, runner_path, config_path)
    }
    source_files.update(
        {
            f"ai_race/configs/game/{name}.json": sha256_file(
                root / "ai_race" / "configs" / "game" / f"{name}.json"
            )
            for name in game_names
        }
    )

    if args.backend == "ollama":
        detected_gpu = gpu_name()
        if args.required_gpu.lower() not in detected_gpu.lower():
            raise RuntimeError(
                f"GPU mismatch: required {args.required_gpu!r}, detected {detected_gpu!r}"
            )
        model_info = model_provenance(args.endpoint, args.model)
        ollama_version = request_json(args.endpoint, "/api/version").get("version")
        backend: Any = OllamaBatchBackend(
            endpoint=args.endpoint,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            workers=args.workers,
        )
        evidence_class = "exploratory_model_self_report"
    else:
        detected_gpu = "not_used"
        ollama_version = None
        model_info = {
            "name": "deterministic-mock-recognition-v1",
            "digest": hashlib.sha256(b"deterministic-mock-recognition-v1").hexdigest(),
            "size": 0,
            "details": {"synthetic": True},
        }
        backend = MockRecognitionBackend()
        evidence_class = "synthetic_smoke_only"

    expected_rows = len(LANE_SKINS[args.lane]) * len(ACTION_CODE_MAPPINGS) * repetitions
    manifest: dict[str, Any] = {
        "schema_version": "ai-race-context-recognition-run-v1",
        "status": "running",
        "evidence_class": evidence_class,
        "evidence_boundary": (
            "Recognition is a separate model self-report audit and is not proof of "
            "training contamination, memorisation, or causal recognition."
        ),
        "started_utc": utc_now(),
        "completed_utc": None,
        "run_spec": run_spec,
        "run_spec_sha256": run_spec_sha256,
        "source_tree_sha256": source_tree_sha256(
            root, (runner_path, core_path, skins_path)
        ),
        "source_files_sha256": source_files,
        "experiment_config_sha256": sha256_file(config_path),
        "game_config_sha256": {
            name: source_files[f"ai_race/configs/game/{name}.json"] for name in game_names
        },
        "scenario_game_config": recognition_config.name,
        "scenario_game_config_max_private_risk": recognition_config.max_private_risk,
        "template_sha256": {
            f"{skin_id}/{mapping_id}": context_skin_sha256(skin_id, mapping_id)
            for skin_id in LANE_SKINS[args.lane]
            for mapping_id in ACTION_CODE_MAPPINGS
        },
        "model": {
            "name": model_info.get("name"),
            "digest": model_info.get("digest"),
            "size": model_info.get("size"),
            "details": model_info.get("details", {}),
            "engine": args.backend,
        },
        "hardware": {
            "hostname": platform.node(),
            "gpu_name": detected_gpu,
            "ollama_version": ollama_version,
            "python": platform.python_version(),
        },
        "decoding": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "workers": args.workers,
            "batch_size": args.batch_size,
            "max_parse_retries": args.max_parse_retries,
            "retry_prompt_policy": "exact_prompt_reissue_without_correction_or_hint",
            "seed_requested": True,
        },
        "expected_rows": expected_rows,
        "n_rows": 0,
        "artifacts": {},
        "error": None,
    }
    atomic_json(manifest_path, manifest)
    try:
        rows = run_recognition_matrix(
            recognition_config,
            LANE_SKINS[args.lane],
            backend,
            repetitions=repetitions,
            seed=int(experiment["seed"]),
            batch_size=args.batch_size,
            max_parse_retries=args.max_parse_retries,
        )
        if len(rows) != expected_rows:
            raise RuntimeError(f"Expected {expected_rows} rows, received {len(rows)}")
        results_path = output_root / "recognition_rows.jsonl"
        summary_path = output_root / "recognition_summary.json"
        _atomic_jsonl(results_path, rows)
        atomic_json(summary_path, summarize_recognition(rows))
        manifest.update(
            {
                "status": "completed",
                "completed_utc": utc_now(),
                "n_rows": len(rows),
                "artifacts": {
                    "rows": _artifact(results_path),
                    "summary": _artifact(summary_path),
                },
            }
        )
        atomic_json(manifest_path, manifest)
        print(
            f"[completed] lane={args.lane} profile={args.profile} rows={len(rows)}",
            flush=True,
        )
    except Exception as error:
        manifest.update(
            {
                "status": "failed",
                "completed_utc": utc_now(),
                "error": f"{type(error).__name__}: {error}",
            }
        )
        atomic_json(manifest_path, manifest)
        raise


if __name__ == "__main__":
    main()
