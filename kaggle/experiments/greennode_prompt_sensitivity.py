"""Run disjoint AI Race prompt-sensitivity shards on GreenNode Ollama pods.

The two pods share persistent storage, so each lane owns distinct experiment
directories and writes atomic manifests. Completed experiment shards are skipped
on resume; an interrupted shard is rerun from scratch without mixing partial rows.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_race.dataio.config_loader import load_json, personas_sha256
from ai_race.dataio.recorder import RunJournal
from ai_race.runner.batch import run_games_batched
from ai_race.runner.run_experiment import build_games_for_model


LANE_EXPERIMENTS = {
    "a": [
        "baseline",
        "persona_baseline_neutral",
        "persona_baseline_risk_averse",
        "persona_baseline_coop_coop",
        "persona_baseline_adv_coop",
    ],
    "b": [
        "baseline_swapped",
        "persona_baseline_risk_seeking",
        "persona_baseline_adv_adv",
        "persona_baseline_coop_adv",
    ],
}
PROFILE_REPETITIONS = {"smoke": 2, "pilot": 10}
DEFAULT_MODEL = "qwen2.5:7b-instruct-fp16"
DEFAULT_ENDPOINT = "http://localhost:11434"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256(root: Path) -> str:
    digest = hashlib.sha256()
    roots = [root / "ai_race", root / "FAIRGAME" / "src"]
    files = sorted(
        path
        for source_root in roots
        for path in source_root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".txt"}
    )
    for path in files:
        digest.update(path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".tmp-{os.getpid()}")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, path)


def request_json(endpoint: str, route: str, payload: Optional[dict] = None) -> dict:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        endpoint.rstrip("/") + route,
        data=data,
        headers={"Content-Type": "application/json"},
        method="GET" if data is None else "POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read().decode("utf-8"))


def gpu_name() -> str:
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip().splitlines()[0] if result.stdout.strip() else ""


def model_provenance(endpoint: str, model: str) -> dict[str, Any]:
    tags = request_json(endpoint, "/api/tags")
    matches = [item for item in tags.get("models", []) if item.get("name") == model]
    if len(matches) != 1:
        raise RuntimeError(f"Ollama model {model!r} is missing or ambiguous")
    item = matches[0]
    return {
        "name": model,
        "digest": item.get("digest"),
        "size": item.get("size"),
        "details": item.get("details", {}),
    }


class OllamaBatchBackend:
    def __init__(
        self,
        *,
        endpoint: str,
        model: str,
        temperature: float,
        max_tokens: int,
        workers: int,
    ) -> None:
        self.endpoint = endpoint
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.workers = workers

    def one(self, prompt: str, seed: int) -> str:
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "keep_alive": "30m",
            "options": {
                "temperature": self.temperature,
                "seed": int(seed),
                "num_predict": self.max_tokens,
                "num_ctx": 4096,
            },
        }
        response = request_json(self.endpoint, "/api/generate", payload)
        if not response.get("done"):
            raise RuntimeError("Ollama returned an incomplete non-streaming response")
        return str(response.get("response", ""))

    def __call__(
        self, prompts: list[str], seeds: Optional[list[int]] = None
    ) -> list[str]:
        resolved_seeds = list(seeds or range(len(prompts)))
        if len(resolved_seeds) != len(prompts):
            raise ValueError("Ollama batch requires one seed per prompt")
        with ThreadPoolExecutor(max_workers=min(self.workers, len(prompts))) as pool:
            return list(pool.map(self.one, prompts, resolved_seeds))


def agents_provenance(root: Path, experiment: dict[str, Any]) -> dict[str, Any]:
    name = str(experiment.get("agents", "companies_default"))
    path = root / "ai_race" / "configs" / "agents" / f"{name}.json"
    config = load_json(path)
    return {
        "agents_name": name,
        "agents_config_sha256": sha256_file(path),
        "persona_condition": str(config.get("personaCondition", "none")),
        "persona_roles": list(config.get("personaRoles", ["", ""])),
        "persona_sha256": personas_sha256(config.get("personas", {}) or {}),
    }


def run_experiment_shard(
    *,
    root: Path,
    output_root: Path,
    experiment_name: str,
    repetitions: int,
    model: str,
    backend: OllamaBatchBackend,
    common: dict[str, Any],
    resume: bool,
) -> dict[str, Any]:
    output_dir = output_root / experiment_name
    manifest_path = output_dir / "run_manifest.json"
    if resume and manifest_path.is_file():
        previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        if previous.get("status") == "completed":
            print(f"[resume] skip completed shard {experiment_name}", flush=True)
            return previous

    config_dir = root / "ai_race" / "configs"
    experiment_path = config_dir / "experiment" / f"{experiment_name}.json"
    experiment = load_json(experiment_path)
    experiment["repetitions"] = repetitions
    experiment["runPhase"] = "pilot"
    experiment["samplingSeedApplied"] = True
    games = build_games_for_model(experiment, model)
    expected_races = len(games)
    journal = RunJournal(output_dir, reset=True)
    manifest = {
        "schema_version": "ai-race-greennode-run-v1",
        "status": "running",
        "started_utc": utc_now(),
        "completed_utc": None,
        **common,
        "experiment_name": experiment_name,
        "experiment": experiment,
        "experiment_config_sha256": sha256_file(experiment_path),
        "effective_experiment_sha256": hashlib.sha256(
            json.dumps(experiment, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "expected_races": expected_races,
        **agents_provenance(root, experiment),
        "n_races": 0,
        "n_turns": 0,
        "error": None,
    }
    atomic_json(manifest_path, manifest)
    try:
        results = run_games_batched(
            games,
            backend,
            verbose=False,
            max_parse_retries=int(experiment.get("maxParseRetries", 3)),
            on_round_complete=journal.record_round,
        )
        if len(results) != expected_races or journal.race_count != expected_races:
            raise RuntimeError(
                f"Incomplete coverage: expected={expected_races}, "
                f"results={len(results)}, journal={journal.race_count}"
            )
    except Exception as error:
        manifest.update(
            status="failed",
            completed_utc=utc_now(),
            n_races=journal.race_count,
            n_turns=journal.turn_count,
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(manifest_path, manifest)
        raise
    manifest.update(
        status="completed",
        completed_utc=utc_now(),
        n_races=journal.race_count,
        n_turns=journal.turn_count,
    )
    atomic_json(manifest_path, manifest)
    print(
        f"[completed] {experiment_name}: {journal.race_count} races, "
        f"{journal.turn_count} decisions",
        flush=True,
    )
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--lane", choices=sorted(LANE_EXPERIMENTS), required=True)
    parser.add_argument("--profile", choices=sorted(PROFILE_REPETITIONS), default="smoke")
    parser.add_argument("--repo-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--endpoint", default=DEFAULT_ENDPOINT)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=16)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--required-gpu", default="H100")
    parser.add_argument("--no-resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.repo_root.resolve()
    output_root = args.output_root.resolve()
    experiments = LANE_EXPERIMENTS[args.lane]
    repetitions = PROFILE_REPETITIONS[args.profile]
    detected_gpu = gpu_name()
    if args.required_gpu.lower() not in detected_gpu.lower():
        raise RuntimeError(
            f"GPU mismatch: required {args.required_gpu!r}, detected {detected_gpu!r}"
        )
    model_info = model_provenance(args.endpoint, args.model)
    ollama_version = request_json(args.endpoint, "/api/version").get("version")
    backend = OllamaBatchBackend(
        endpoint=args.endpoint,
        model=args.model,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        workers=args.workers,
    )

    # A fixed-seed probe catches connectors that silently ignore the option before
    # any behavioral shard is recorded as sampling-seeded.
    probe_prompt = "Reply with exactly ACTION: SAFE or ACTION: UNSAFE."
    probe_a = backend.one(probe_prompt, 260726)
    probe_b = backend.one(probe_prompt, 260726)
    if probe_a != probe_b:
        raise RuntimeError("Ollama fixed-seed reproducibility probe failed")

    source_hash = source_tree_sha256(root)
    prompt_path = root / "ai_race" / "prompts" / "ai_race_en.txt"
    common = {
        "profile": args.profile,
        "lane": args.lane,
        "hostname": platform.node(),
        "gpu_name": detected_gpu,
        "ollama_version": ollama_version,
        "model": model_info,
        "source_sha256": source_hash,
        "prompt_version": "ai-race-fairgame-v3",
        "prompt_sha256": sha256_file(prompt_path),
        "decoding": {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
            "num_ctx": 4096,
            "workers": args.workers,
            "seed_requested": True,
            "seed_probe_exact_match": True,
        },
        "base_seed": 260726,
    }
    lane_manifest_path = output_root / "lane_manifest.json"
    lane_manifest = {
        "schema_version": "ai-race-greennode-lane-v1",
        "status": "running",
        "started_utc": utc_now(),
        **common,
        "experiments": experiments,
        "repetitions": repetitions,
        "runs": [],
    }
    atomic_json(lane_manifest_path, lane_manifest)
    started = time.monotonic()
    try:
        for experiment_name in experiments:
            lane_manifest["runs"].append(
                run_experiment_shard(
                    root=root,
                    output_root=output_root,
                    experiment_name=experiment_name,
                    repetitions=repetitions,
                    model=args.model,
                    backend=backend,
                    common=common,
                    resume=not args.no_resume,
                )
            )
            atomic_json(lane_manifest_path, lane_manifest)
    except Exception as error:
        lane_manifest.update(
            status="failed",
            completed_utc=utc_now(),
            elapsed_seconds=round(time.monotonic() - started, 3),
            error=f"{type(error).__name__}: {error}",
        )
        atomic_json(lane_manifest_path, lane_manifest)
        raise
    lane_manifest.update(
        status="completed",
        completed_utc=utc_now(),
        elapsed_seconds=round(time.monotonic() - started, 3),
        error=None,
    )
    atomic_json(lane_manifest_path, lane_manifest)
    print(f"Lane {args.lane.upper()} completed in {lane_manifest['elapsed_seconds']}s")


if __name__ == "__main__":
    main()
