# %%
"""AI Race baseline — Kaggle GPU notebook (Internet OFF).

Notebook này dùng đúng runner của project:

* ``ai_race.runner.run_experiment.build_games_for_model``
* ``ai_race.runner.batch.run_games_batched``
* ``ai_race.dataio.recorder`` cho turns/races/players
* ``ai_race.models.factory`` cho backend vLLM/Transformers

Repo input được tự dò bằng hai marker ``ai_race/`` và ``FAIRGAME/``, sau đó copy
vào ``/kaggle/working/ai_race_repo`` để Python có thể import và ghi cache. Các
model được nạp, chạy, giải phóng tuần tự. Không cần Internet nếu Kaggle image đã
có vLLM hoặc một Dataset wheels đã được add làm input.
"""

# %%
# Cấu hình người dùng — sửa path theo các input đã add trong Kaggle.
import os
from pathlib import Path

MODELS = [
    {
        "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/7b-instruct/1",
        "short_name": "qwen2.5-7b-instruct",
        "engine": "vllm",
        # Các override dưới đây là tùy chọn:
        # "temperature": 0.7,
        # "max_tokens": 256,
        # "logprobs": 5,  # opt-in; keep 0/off for the behavioral baseline
        # "max_model_len": 4096,
        # "engine_overrides": {"quantization": "awq", "dtype": "auto"},
    },
    # Thêm model khác tại đây; notebook sẽ chạy lần lượt, không nạp đồng thời.
]

# Dataset đã stage: https://www.kaggle.com/datasets/nguyenlamphuquy/ai-race-experiment
# Kaggle mount dataset theo *slug*, không kèm username, nên path là
# /kaggle/input/ai-race-experiment. Username chỉ nằm trong dataset id dùng cho CLI
# (nguyenlamphuquy/ai-race-experiment). Để None thì notebook tự dò input nào có
# đồng thời ai_race/ và FAIRGAME/.
REPO_INPUT_DIR = "/kaggle/input/ai-race-experiment"

# Mọi cell persona phải chạy trong CÙNG một session. protocol_signature gồm source
# revision, decoding và package versions; chạy lệch session thì persona trùng khít
# với batch và analyser sẽ từ chối ước lượng hệ số persona.
PROMPT_SENSITIVITY_EXPERIMENTS = [
    "baseline",                       # none  — đối chứng trung tính
    "baseline_swapped",               # none  — đảo ghế, đo artefact vị trí
    "persona_baseline_neutral",       # R0    — placebo cùng độ dài
    "persona_baseline_risk_averse",   # R-
    "persona_baseline_risk_seeking",  # R+
    "persona_baseline_coop_coop",     # S_CC
    "persona_baseline_adv_adv",       # S_AA
    "persona_baseline_adv_coop",      # S_AC  — cell bất đối xứng
    "persona_baseline_coop_adv",      # S_CA  — mirror bắt buộc của S_AC
]

# Kernel bootstrap có thể chọn profile mà không sửa source đã hash. Mọi arm của
# prompt-sensitivity phải chạy trong cùng session để source/model/decoding giống hệt.
RUN_PROFILE = os.environ.get("AI_RACE_RUN_PROFILE", "baseline").strip().lower()
PROFILE_EXPERIMENTS = {
    "baseline": ["baseline"],
    "prompt_sensitivity_smoke": PROMPT_SENSITIVITY_EXPERIMENTS,
    "prompt_sensitivity_pilot": PROMPT_SENSITIVITY_EXPERIMENTS,
}
if RUN_PROFILE not in PROFILE_EXPERIMENTS:
    raise ValueError(
        f"Unknown AI_RACE_RUN_PROFILE={RUN_PROFILE!r}; expected one of "
        f"{sorted(PROFILE_EXPERIMENTS)}"
    )
EXPERIMENTS = list(PROFILE_EXPERIMENTS[RUN_PROFILE])

_repetition_default = {
    "baseline": 10,
    "prompt_sensitivity_smoke": 2,
    "prompt_sensitivity_pilot": 10,
}[RUN_PROFILE]
_repetition_env = os.environ.get("AI_RACE_REPETITIONS_OVERRIDE")
REPETITIONS_OVERRIDE = (
    int(_repetition_env) if _repetition_env is not None else _repetition_default
)
# Smoke = 2 rep/arm; pilot = 10 rep/arm. Chỉ scale sau khi coverage, parser và
# symmetry gates đều đạt. Config gốc vẫn giữ 50 rep cho confirmatory sau freeze.
RUN_PHASE_OVERRIDE = None  # "pilot" hoặc "confirmatory"; None = dùng config

REQUIRED_GPU_NAME = os.environ.get("AI_RACE_REQUIRED_GPU", "").strip()
MIN_GPU_VRAM_GIB = float(os.environ.get("AI_RACE_MIN_GPU_VRAM_GIB", "0"))

DEFAULT_ENGINE = "vllm"
TEMPERATURE = 0.7
MAX_TOKENS = 256
LOGPROBS = 0  # Opt-in only; values >0 add substantial decode memory/work.
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.90
TENSOR_PARALLEL_SIZE = 1
ENFORCE_EAGER = True
BATCH_SIZE = 128
MAX_PARSE_RETRIES_OVERRIDE = None  # None = dùng maxParseRetries trong experiment
FAIL_ON_INCOMPLETE_RUN = True

INSTALL_VLLM_IF_MISSING = True
VLLM_WHEELS_DIR = None  # Bắt buộc điền Dataset wheelhouse đã audit nếu cần cài.

WORK_COPY = Path("/kaggle/working/ai_race_repo")
OUTPUT_DIR = Path("/kaggle/working/ai_race_results")
ZIP_PATH = Path("/kaggle/working/ai_race_results.zip")
RESET_OUTPUT_DIR = True


# %%
# Helpers tự dò input. Tìm có giới hạn độ sâu để tránh quét toàn bộ model weights.
import sys


def find_directory(root, predicate, max_depth=6):
    root = Path(root)
    if not root.is_dir():
        return None
    stack = [(root, 0)]
    while stack:
        directory, depth = stack.pop()
        try:
            if predicate(directory):
                return directory.resolve()
        except OSError:
            pass
        if depth >= max_depth:
            continue
        try:
            children = [
                child
                for child in directory.iterdir()
                if child.is_dir() and not child.name.startswith(".")
            ]
        except OSError:
            continue
        stack.extend((child, depth + 1) for child in reversed(sorted(children)))
    return None


def is_repo_input(directory):
    directory = Path(directory)
    return (directory / "ai_race").is_dir() and (directory / "FAIRGAME").is_dir()


def find_repo_input(root="/kaggle/input"):
    """Prefer the configured dataset mount, then fall back to discovery.

    Naming the expected path turns "wrong dataset added" into an explicit error
    instead of a silent fallback onto some other input that happens to contain the
    two marker directories — which would run a different source revision than the
    one the manifest is about to claim.
    """
    configured = globals().get("REPO_INPUT_DIR")
    if configured:
        configured = Path(configured)
        if is_repo_input(configured):
            return configured.resolve()
        discovered = find_directory(root, is_repo_input)
        raise FileNotFoundError(
            f"REPO_INPUT_DIR={configured} does not contain both ai_race/ and "
            "FAIRGAME/. Add the dataset "
            "'nguyenlamphuquy/ai-race-experiment' as a notebook input, or set "
            "REPO_INPUT_DIR=None to auto-discover. "
            + (
                f"A usable repo input was found at {discovered}."
                if discovered
                else "No usable repo input was found under /kaggle/input."
            )
        )
    return find_directory(root, is_repo_input)


# %%
# Cài vLLM offline nếu model yêu cầu và image hiện tại chưa có.
import hashlib
import importlib.util
import json
import subprocess


def validate_gpu_runtime():
    """Fail closed before loading weights when the assigned GPU is wrong."""

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; refusing to run a GPU experiment")
    properties = torch.cuda.get_device_properties(0)
    name = str(properties.name)
    total_vram_gib = float(properties.total_memory) / 1024**3
    if REQUIRED_GPU_NAME and REQUIRED_GPU_NAME.lower() not in name.lower():
        raise RuntimeError(
            f"GPU mismatch: required name containing {REQUIRED_GPU_NAME!r}, got {name!r}"
        )
    if total_vram_gib + 1e-9 < MIN_GPU_VRAM_GIB:
        raise RuntimeError(
            f"GPU VRAM mismatch: require >= {MIN_GPU_VRAM_GIB:.1f} GiB, "
            f"got {total_vram_gib:.1f} GiB"
        )
    runtime = {
        "gpu_name": name,
        "gpu_vram_gib": round(total_vram_gib, 3),
        "cuda_version": str(torch.version.cuda),
        "torch_cuda_device_count": int(torch.cuda.device_count()),
        "required_gpu_name": REQUIRED_GPU_NAME or None,
        "minimum_gpu_vram_gib": MIN_GPU_VRAM_GIB,
    }
    print(f"GPU runtime: {json.dumps(runtime, sort_keys=True)}")
    return runtime


GPU_RUNTIME = validate_gpu_runtime()

needs_vllm = any(
    model.get("engine", DEFAULT_ENGINE).lower() == "vllm" for model in MODELS
)
has_vllm = importlib.util.find_spec("vllm") is not None

if needs_vllm and not has_vllm:
    if not INSTALL_VLLM_IF_MISSING:
        raise RuntimeError(
            "MODELS yêu cầu vLLM nhưng image chưa có và INSTALL_VLLM_IF_MISSING=False."
        )
    if VLLM_WHEELS_DIR is None:
        raise FileNotFoundError(
            "Set VLLM_WHEELS_DIR to the exact audited wheelhouse Dataset path. "
            "Automatic wheel discovery is disabled."
        )
    wheels_dir = Path(VLLM_WHEELS_DIR)
    if wheels_dir is None or not wheels_dir.is_dir():
        raise FileNotFoundError(
            f"Không tìm thấy vLLM wheelhouse tại {wheels_dir}."
        )
    wheel_manifest_path = wheels_dir / "manifest.json"
    if not wheel_manifest_path.is_file():
        raise FileNotFoundError(
            f"Wheelhouse thiếu manifest SHA-256: {wheel_manifest_path}"
        )
    wheel_manifest = json.loads(wheel_manifest_path.read_text(encoding="utf-8"))
    manifest_files = wheel_manifest.get("files")
    requested_specs = wheel_manifest.get("requested")
    if not isinstance(manifest_files, list) or not manifest_files:
        raise ValueError("Wheelhouse manifest.files must be a non-empty list")
    if not isinstance(requested_specs, list) or not requested_specs:
        raise ValueError("Wheelhouse manifest.requested must be a non-empty list")
    if any(
        not isinstance(spec, str) or "==" not in spec
        for spec in requested_specs
    ):
        raise ValueError(
            "Wheelhouse manifest.requested must contain only exact == requirements"
        )
    declared_wheels = set()
    for entry in manifest_files:
        if not isinstance(entry, dict) or not {"name", "sha256"} <= set(entry):
            raise ValueError("Each wheel manifest entry requires name and sha256")
        declared_name = str(entry["name"])
        if Path(declared_name).name != declared_name or not declared_name.endswith(".whl"):
            raise ValueError(f"Invalid wheel filename in manifest: {declared_name!r}")
        if declared_name in declared_wheels:
            raise ValueError(f"Duplicate wheel filename in manifest: {declared_name}")
        declared_wheels.add(declared_name)
        wheel_path = wheels_dir / declared_name
        if not wheel_path.is_file():
            raise FileNotFoundError(f"Wheel declared in manifest is missing: {wheel_path}")
        if "bytes" in entry and wheel_path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"Wheel byte-size mismatch: {wheel_path.name}")
        digest = hashlib.sha256()
        with wheel_path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != str(entry["sha256"]).lower():
            raise RuntimeError(f"Wheel SHA-256 mismatch: {wheel_path.name}")
    actual_wheels = {path.name for path in wheels_dir.glob("*.whl")}
    if actual_wheels != declared_wheels:
        raise RuntimeError(
            "Wheelhouse contents do not exactly match manifest.files; "
            f"undeclared={sorted(actual_wheels - declared_wheels)}, "
            f"missing={sorted(declared_wheels - actual_wheels)}"
        )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "--no-index",
            "--only-binary=:all:",
            f"--find-links={wheels_dir}",
            *requested_specs,
        ]
    )
    importlib.invalidate_caches()
    print(f"Installed vLLM offline from {wheels_dir}")
elif needs_vllm:
    print("vLLM is already available in this Kaggle image.")
else:
    print("No configured model requires vLLM.")


# %%
# Copy source từ input read-only sang /kaggle/working.
import shutil

repo_input = find_repo_input()
if repo_input is None:
    raise FileNotFoundError(
        "Không tìm thấy repo chứa đồng thời ai_race/ và FAIRGAME/ dưới /kaggle/input. "
        "Hãy Add Input repo này vào notebook."
    )

if WORK_COPY.exists():
    shutil.rmtree(WORK_COPY)
shutil.copytree(repo_input, WORK_COPY, ignore=shutil.ignore_patterns(".git", "__pycache__"))

REPO_ROOT = WORK_COPY
os.chdir(REPO_ROOT)
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

required_paths = [
    REPO_ROOT / "ai_race",
    REPO_ROOT / "ai_race" / "configs",
    REPO_ROOT / "FAIRGAME",
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise FileNotFoundError(f"Repo copy thiếu thành phần bắt buộc: {missing}")

print(f"Repo input : {repo_input}")
print(f"Working copy: {REPO_ROOT}")


# %%
# Import API framework sau khi working copy đã vào sys.path.
from ai_race.dataio.config_loader import load_json
from ai_race.dataio.recorder import RunJournal
from ai_race.models import factory
from ai_race.runner.batch import run_games_batched
from ai_race.runner.run_experiment import build_games_for_model

CONFIG_DIR = REPO_ROOT / "ai_race" / "configs"


def load_experiment(name):
    candidates = [
        CONFIG_DIR / "experiment" / f"{name}.json",
        CONFIG_DIR / f"{name}.json",
    ]
    path = next((candidate for candidate in candidates if candidate.is_file()), None)
    if path is None:
        raise FileNotFoundError(f"Missing experiment config; checked: {candidates}")
    experiment = load_json(path)
    if REPETITIONS_OVERRIDE is not None:
        experiment["repetitions"] = int(REPETITIONS_OVERRIDE)
    if RUN_PHASE_OVERRIDE is not None:
        experiment["runPhase"] = str(RUN_PHASE_OVERRIDE)
    return experiment, path


def offline_settings_for(model):
    logprobs = int(model.get("logprobs", LOGPROBS))
    if logprobs < 0:
        raise ValueError("model.logprobs must be a non-negative integer")
    if (
        model.get("engine", DEFAULT_ENGINE).lower() != "vllm"
        and logprobs > 0
    ):
        raise ValueError("logprobs > 0 is supported only by the vLLM engine")
    return {
        "backend": model.get("engine", DEFAULT_ENGINE),
        "sampling": {
            "temperature": float(model.get("temperature", TEMPERATURE)),
            "maxTokens": int(model.get("max_tokens", MAX_TOKENS)),
            "logprobs": logprobs,
        },
        "engine": {
            "maxModelLen": int(model.get("max_model_len", MAX_MODEL_LEN)),
            "gpuMemoryUtilization": float(
                model.get("gpu_memory_utilization", GPU_MEMORY_UTILIZATION)
            ),
            "tensorParallelSize": int(
                model.get("tensor_parallel_size", TENSOR_PARALLEL_SIZE)
            ),
            "enforceEager": bool(model.get("enforce_eager", ENFORCE_EAGER)),
            **dict(model.get("engine_overrides", {})),
        },
    }

# %%
# Helpers chạy/lưu một experiment.
import csv
import gc
import hashlib
import importlib.metadata
import json
import traceback
from datetime import datetime, timezone


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_tree_sha256():
    digest = hashlib.sha256()
    roots = [REPO_ROOT / "ai_race", REPO_ROOT / "FAIRGAME" / "src"]
    files = sorted(
        path
        for root in roots
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".py", ".json", ".txt"}
    )
    for path in files:
        relative = path.relative_to(REPO_ROOT).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def package_versions():
    versions = {}
    for package in ("numpy", "pandas", "torch", "transformers", "vllm"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


SOURCE_SHA256 = source_tree_sha256()
PACKAGE_VERSIONS = package_versions()


def load_model(model):
    preset = {
        "modelPath": str(model["path"]),
        "engine": dict(model.get("engine_overrides", {})),
    }
    factory.init_offline_backend(offline_settings_for(model), preset, force=True)
    return factory.get_send_batch(
        model["short_name"], offline=True, batch_size=int(model.get("batch_size", BATCH_SIZE))
    )


def run_one_experiment(model, experiment_name, send_batch):
    experiment, experiment_path = load_experiment(experiment_name)
    out_dir = OUTPUT_DIR / model["short_name"] / experiment_name
    journal = RunJournal(out_dir, reset=True)
    max_parse_retries = int(
        MAX_PARSE_RETRIES_OVERRIDE
        if MAX_PARSE_RETRIES_OVERRIDE is not None
        else experiment.get("maxParseRetries", 3)
    )
    game_config_hashes = {
        game_name: sha256_file(CONFIG_DIR / "game" / f"{game_name}.json")
        for game_name in experiment["games"]
    }
    prompt_path = REPO_ROOT / "ai_race" / "prompts" / "ai_race_en.txt"
    model_config_path = Path(model["path"]) / "config.json"
    agents_name = str(experiment.get("agents", "companies_default"))
    agents_path = CONFIG_DIR / "agents" / f"{agents_name}.json"
    agents_cfg = json.loads(agents_path.read_text(encoding="utf-8"))
    run_manifest = {
        "schema_version": "ai-race-kaggle-run-v1",
        "status": "running",
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "completed_utc": None,
        "source_sha256": SOURCE_SHA256,
        "run_profile": RUN_PROFILE,
        "repetitions_override": REPETITIONS_OVERRIDE,
        "gpu_runtime": GPU_RUNTIME,
        "experiment_name": experiment_name,
        "run_phase": str(experiment.get("runPhase", "pilot")),
        "experiment": experiment,
        "experiment_config_sha256": sha256_file(experiment_path),
        "effective_experiment_sha256": hashlib.sha256(
            json.dumps(
                experiment,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "game_config_sha256": game_config_hashes,
        # Persona is an agents-configuration factor, so it never shows up in the
        # prompt hash. Without these fields a persona run and the neutral baseline
        # are indistinguishable to the analyser.
        "agents_name": agents_name,
        "agents_config_sha256": sha256_file(agents_path),
        "persona_condition": str(agents_cfg.get("personaCondition", "none")).strip(),
        "persona_roles": [str(role) for role in agents_cfg.get("personaRoles", ["", ""])],
        "persona_sha256": hashlib.sha256(
            json.dumps(
                agents_cfg.get("personas", {}) or {},
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=False,
            ).encode("utf-8")
        ).hexdigest(),
        "prompt_version": "ai-race-fairgame-v3",
        "prompt_sha256": sha256_file(prompt_path),
        "model": {
            "short_name": model["short_name"],
            "path": str(model["path"]),
            "engine": model.get("engine", DEFAULT_ENGINE),
            "config_sha256": (
                sha256_file(model_config_path) if model_config_path.is_file() else None
            ),
        },
        "decoding": {
            "temperature": float(model.get("temperature", TEMPERATURE)),
            "max_tokens": int(model.get("max_tokens", MAX_TOKENS)),
            "logprobs": int(model.get("logprobs", LOGPROBS)),
            "logprobs_enabled": (
                model.get("engine", DEFAULT_ENGINE).lower() == "vllm"
                and int(model.get("logprobs", LOGPROBS)) > 0
            ),
            "max_parse_retries": max_parse_retries,
        },
        "package_versions": PACKAGE_VERSIONS,
        "n_races": 0,
        "n_turns": 0,
        "error": None,
    }
    manifest_path = out_dir / "run_manifest.json"

    def write_run_manifest():
        manifest_path.write_text(
            json.dumps(run_manifest, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    write_run_manifest()
    try:
        games = build_games_for_model(
            experiment,
            model["short_name"],
        )
        expected_races = len(games)
        run_manifest["expected_races"] = expected_races
        write_run_manifest()
        results = run_games_batched(
            games,
            send_batch,
            verbose=True,
            max_parse_retries=max_parse_retries,
            on_round_complete=journal.record_round,
        )
        if len(results) != expected_races or journal.race_count != expected_races:
            raise RuntimeError(
                "Incomplete experiment coverage: "
                f"expected={expected_races}, results={len(results)}, "
                f"journal={journal.race_count}"
            )
    except Exception as error:
        run_manifest.update(
            {
                "status": "failed",
                "completed_utc": datetime.now(timezone.utc).isoformat(),
                "n_races": journal.race_count,
                "n_turns": journal.turn_count,
                "error": f"{type(error).__name__}: {error}",
            }
        )
        write_run_manifest()
        raise

    run_manifest.update(
        {
            "status": "completed",
            "completed_utc": datetime.now(timezone.utc).isoformat(),
            "n_races": journal.race_count,
            "n_turns": journal.turn_count,
        }
    )
    write_run_manifest()
    print(
        f"[{model['short_name']}/{experiment_name}] "
        f"{journal.race_count} races, {journal.turn_count} decisions -> {out_dir}"
    )
    return {
        "model": model["short_name"],
        "experiment": experiment_name,
        "status": "completed",
        "n_races": journal.race_count,
        "n_turns": journal.turn_count,
        "output_dir": str(out_dir),
        "run_manifest": str(manifest_path),
    }


def free_model():
    """Giải phóng singleton backend trước khi nạp checkpoint kế tiếp."""
    try:
        free_from_factory = getattr(factory, "free_offline_backend", None)
        if callable(free_from_factory):
            free_from_factory()
        else:
            from FAIRGAME.src.llm_connectors import local_vllm_connector

            local_vllm_connector.free_local_llm()
    except Exception as error:
        print(f"Backend cleanup warning: {error}")
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def merge_csv_files(filename, destination):
    sources = sorted(
        path
        for path in OUTPUT_DIR.rglob(filename)
        if path.resolve() != destination.resolve()
    )
    rows = []
    fields = []
    for source in sources:
        run_manifest_path = source.parent / "run_manifest.json"
        if not run_manifest_path.is_file():
            print(f"[aggregate skip] no sibling run_manifest.json: {source}")
            continue
        try:
            source_manifest = json.loads(
                run_manifest_path.read_text(encoding="utf-8")
            )
        except (OSError, json.JSONDecodeError) as error:
            raise RuntimeError(
                f"Cannot verify aggregate source manifest: {run_manifest_path}"
            ) from error
        run_status = str(source_manifest.get("status", "")).strip().lower()
        if run_status != "completed":
            print(
                f"[aggregate skip] status={run_status or 'missing'}: {source}"
            )
            continue
        relative = source.relative_to(OUTPUT_DIR)
        source_run = relative.parent.as_posix()
        with source.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row.setdefault("model", relative.parts[0] if len(relative.parts) > 0 else "")
                row.setdefault(
                    "experiment", relative.parts[1] if len(relative.parts) > 1 else ""
                )
                row["source_run"] = source_run
                row["run_status"] = run_status
                for field in row:
                    if field not in fields:
                        fields.append(field)
                rows.append(row)
    if not rows:
        return 0
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def write_prompt_sensitivity_summary(players_path, destination):
    """Write decision-weighted arm effects relative to the neutral baseline.

    This is a descriptive smoke/pilot diagnostic, not the confirmatory estimator.
    The repository analyser remains authoritative for clustered inference.
    """

    if not players_path.is_file():
        return 0
    aggregates = {}
    with players_path.open("r", newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (
                str(row.get("model", "")),
                str(row.get("experiment", "")),
                str(row.get("persona_condition", "")),
                float(row["max_private_risk"]),
            )
            bucket = aggregates.setdefault(
                key, {"unsafe": 0, "decisions": 0, "trajectories": 0}
            )
            bucket["unsafe"] += int(float(row["unsafe_count"]))
            bucket["decisions"] += int(float(row["n_rounds"]))
            bucket["trajectories"] += 1

    baseline_rates = {}
    for (model, experiment, _condition, risk), values in aggregates.items():
        if experiment == "baseline" and values["decisions"]:
            baseline_rates[(model, risk)] = values["unsafe"] / values["decisions"]

    rows = []
    for (model, experiment, condition, risk), values in sorted(aggregates.items()):
        unsafe_rate = values["unsafe"] / values["decisions"]
        baseline_rate = baseline_rates.get((model, risk))
        rows.append(
            {
                "model": model,
                "experiment": experiment,
                "persona_condition": condition,
                "max_private_risk": risk,
                "player_trajectories": values["trajectories"],
                "decisions": values["decisions"],
                "unsafe_rate": unsafe_rate,
                "baseline_unsafe_rate": baseline_rate,
                "unsafe_rate_delta_vs_baseline": (
                    unsafe_rate - baseline_rate if baseline_rate is not None else ""
                ),
            }
        )
    if not rows:
        return 0
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


# %%
# Chạy model tuần tự. Manifest được cập nhật sau từng model để giữ kết quả đã xong.
if RESET_OUTPUT_DIR and OUTPUT_DIR.exists():
    shutil.rmtree(OUTPUT_DIR)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

manifest = {
    "repo_input": str(repo_input),
    "run_profile": RUN_PROFILE,
    "repetitions_override": REPETITIONS_OVERRIDE,
    "experiments": list(EXPERIMENTS),
    "source_sha256": SOURCE_SHA256,
    "package_versions": PACKAGE_VERSIONS,
    "gpu_runtime": GPU_RUNTIME,
    "models": [],
    "runs": [],
}


def write_manifest():
    (OUTPUT_DIR / "run_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


for model in MODELS:
    model_path = Path(model["path"])
    manifest["models"].append(
        {
            "short_name": model["short_name"],
            "path": str(model_path),
            "engine": model.get("engine", DEFAULT_ENGINE),
        }
    )
    if not model_path.exists():
        print(f"[skip] {model['short_name']}: missing input {model_path}")
        manifest["runs"].append(
            {
                "model": model["short_name"],
                "status": "skipped",
                "error": f"Model input does not exist: {model_path}",
            }
        )
        write_manifest()
        continue

    try:
        print(f"\nLoading {model['short_name']} from {model_path}")
        send_batch = load_model(model)
        for experiment_name in EXPERIMENTS:
            manifest["runs"].append(
                run_one_experiment(model, experiment_name, send_batch)
            )
    except Exception as error:
        traceback.print_exc()
        manifest["runs"].append(
            {
                "model": model["short_name"],
                "status": "failed",
                "error": f"{type(error).__name__}: {error}",
            }
        )
    finally:
        free_model()
        write_manifest()


# %%
# Gộp bảng so sánh chéo model và đóng gói output để tải từ Kaggle.
import zipfile

n_race_rows = merge_csv_files(
    "races.csv", OUTPUT_DIR / "ai_race_all_models.csv"
)
n_player_rows = merge_csv_files(
    "players.csv", OUTPUT_DIR / "ai_race_players_all_models.csv"
)
print(f"Merged {n_race_rows} race rows and {n_player_rows} player rows.")
summary_rows = write_prompt_sensitivity_summary(
    OUTPUT_DIR / "ai_race_players_all_models.csv",
    OUTPUT_DIR / "prompt_sensitivity_summary.csv",
)
print(f"Prompt-sensitivity diagnostic: {summary_rows} rows.")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT_DIR.parent))

print(f"Results: {OUTPUT_DIR}")
print(f"Archive: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024**2:.2f} MiB)")

incomplete_runs = [
    run for run in manifest["runs"] if run.get("status") != "completed"
]
completed_runs = [
    run for run in manifest["runs"] if run.get("status") == "completed"
]
if FAIL_ON_INCOMPLETE_RUN and (incomplete_runs or not completed_runs):
    raise RuntimeError(
        "Kaggle run is incomplete; inspect run_manifest.json and notebook logs. "
        f"completed={len(completed_runs)}, incomplete={len(incomplete_runs)}"
    )
