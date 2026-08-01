# %%
"""AI Race N-player variant: cooperative/adversarial persona, full 2^3 matrix x 3 companies -- Kaggle GPU notebook (Internet OFF).

Quet du 8 to hop hop tac/doi dau tren 3 ghe (2^3, day du -- khac voi risk vi day chi la bien nhi phan nen to hop van kha thi). Gop ca 8 cell vao MOT EXPERIMENTS de chay chung mot session.

Sinh ra song song với kaggle/experiments/baseline.py nhưng dùng
ai_race.engine_nplayer (module N-player riêng, xem
ai_race/engine_nplayer/README.md) -- không import ai_race.runner/
ai_race.dataio.recorder của bản 2-player. Model cố định là Qwen2.5-14B-Instruct
qua backend transformers, không cần quantization (14B ~28GB bf16, vừa thoải
mái trong 96GB RTX PRO 6000) nên không có cell cài vLLM/bitsandbytes.

DEBUG_DUMP_RACE mặc định True: in lại TOÀN BỘ prompt + raw response + action
đã parse của mọi vòng, mọi race trong lần chạy này, để soát xem cơ chế N-player
có hoạt động đúng không trước khi scale lên.
"""

# %%
import os
from pathlib import Path

MODELS = [
    {
        "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b-instruct/1",
        "short_name": "qwen2.5-14b-instruct",
        "engine": "transformers",
    },
    # Thêm model khác tại đây; notebook chạy lần lượt, không nạp đồng thời.
]

# Dataset đã stage: https://www.kaggle.com/datasets/nguyenlamphuquy/ai-race-experiment
REPO_INPUT_DIRS = [
    "/kaggle/input/datasets/nguyenlamphuquy/ai-race-experiment",
    "/kaggle/input/ai-race-experiment",
]

EXPERIMENTS = [
    "persona_nplayer_baseline_coopadv_CCC_n3",
    "persona_nplayer_baseline_coopadv_CCA_n3",
    "persona_nplayer_baseline_coopadv_CAC_n3",
    "persona_nplayer_baseline_coopadv_CAA_n3",
    "persona_nplayer_baseline_coopadv_ACC_n3",
    "persona_nplayer_baseline_coopadv_ACA_n3",
    "persona_nplayer_baseline_coopadv_AAC_n3",
    "persona_nplayer_baseline_coopadv_AAA_n3"
]

_repetition_env = os.environ.get("AI_RACE_REPETITIONS_OVERRIDE")
REPETITIONS_OVERRIDE = int(_repetition_env) if _repetition_env is not None else 2
RUN_PHASE_OVERRIDE = None  # "pilot" hoac "confirmatory"; None = dung config
RUN_PROFILE = "nplayer_persona_coopadv"  # chi de ghi vao manifest cho de tra cuu

REQUIRED_GPU_NAME = os.environ.get("AI_RACE_REQUIRED_GPU", "").strip()
MIN_GPU_VRAM_GIB = float(os.environ.get("AI_RACE_MIN_GPU_VRAM_GIB", "0"))

DEFAULT_ENGINE = "transformers"
TEMPERATURE = 0.7
MAX_TOKENS = 256
LOGPROBS = 0

MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.90
TENSOR_PARALLEL_SIZE = 1
ENFORCE_EAGER = True

BATCH_SIZE = 128
MAX_PARSE_RETRIES_OVERRIDE = None
FAIL_ON_INCOMPLETE_RUN = True

# In lai TOAN BO prompt/response cua MOI race sau khi chay xong.
DEBUG_DUMP_RACE = True
DEBUG_MAX_PROMPT_CHARS = None

WORK_COPY = Path("/kaggle/working/ai_race_repo")
OUTPUT_DIR = Path("/kaggle/working/ai_race_nplayer_results")
ZIP_PATH = Path("/kaggle/working/ai_race_nplayer_results.zip")
RESET_OUTPUT_DIR = True


# %%
# Repo input discovery -- identical logic to kaggle/experiments/baseline.py.
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
    configured = globals().get("REPO_INPUT_DIRS")
    if configured:
        if isinstance(configured, (str, Path)):
            configured = [configured]
        candidates = [Path(candidate) for candidate in configured]
        for candidate in candidates:
            if is_repo_input(candidate):
                return candidate.resolve()
        discovered = find_directory(root, is_repo_input)
        listed = ", ".join(str(candidate) for candidate in candidates)
        raise FileNotFoundError(
            f"None of REPO_INPUT_DIRS ({listed}) contains both ai_race/ and "
            "FAIRGAME/. Add the dataset "
            "'nguyenlamphuquy/ai-race-experiment' as a notebook input, or set "
            "REPO_INPUT_DIRS=None to auto-discover. "
            + (
                f"A usable repo input was found at {discovered}; add it to "
                "REPO_INPUT_DIRS if that is the intended dataset."
                if discovered
                else "No usable repo input was found under /kaggle/input."
            )
        )
    return find_directory(root, is_repo_input)


# %%
import hashlib
import importlib.util
import json


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
print("Model dung transformers, khong quantize -- khong can cai vLLM/bitsandbytes.")


# %%
# Copy source tu input read-only sang /kaggle/working.
import shutil

repo_input = find_repo_input()
if repo_input is None:
    raise FileNotFoundError(
        "Khong tim thay repo chua dong thoi ai_race/ va FAIRGAME/ duoi /kaggle/input. "
        "Hay Add Input repo nay vao notebook."
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
    REPO_ROOT / "ai_race" / "engine_nplayer",
    REPO_ROOT / "FAIRGAME",
]
missing = [str(path) for path in required_paths if not path.exists()]
if missing:
    raise FileNotFoundError(f"Repo copy thieu thanh phan bat buoc: {missing}")

print(f"Repo input : {repo_input}")
print(f"Working copy: {REPO_ROOT}")


# %%
# Import framework N-player sau khi working copy da vao sys.path.
from ai_race.dataio.config_loader import load_json
from ai_race.engine_nplayer.debug import dump_race_prompts, list_race_ids
from ai_race.engine_nplayer.recorder import NPlayerRunJournal
from ai_race.engine_nplayer.runner import build_games_for_model, run_games_batched
from ai_race.models import factory

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
    if model.get("engine", DEFAULT_ENGINE).lower() != "vllm" and logprobs > 0:
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
import csv
import gc
import importlib.metadata
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
    for package in ("numpy", "pandas", "torch", "transformers"):
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
    journal = NPlayerRunJournal(out_dir, reset=True)
    max_parse_retries = int(
        MAX_PARSE_RETRIES_OVERRIDE
        if MAX_PARSE_RETRIES_OVERRIDE is not None
        else experiment.get("maxParseRetries", 3)
    )
    agents_name = str(experiment.get("agents", "companies_nplayer_default_n3"))
    agents_path = CONFIG_DIR / "agents_nplayer" / f"{agents_name}.json"
    agents_cfg = json.loads(agents_path.read_text(encoding="utf-8"))
    run_manifest = {
        "schema_version": "ai-race-nplayer-kaggle-run-v1",
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
        "agents_name": agents_name,
        "agents_config_sha256": sha256_file(agents_path),
        "persona_condition": str(agents_cfg.get("personaCondition", "none")).strip(),
        "persona_roles": [str(role) for role in agents_cfg.get("personaRoles", [])],
        "model": {
            "short_name": model["short_name"],
            "path": str(model["path"]),
            "engine": model.get("engine", DEFAULT_ENGINE),
        },
        "decoding": {
            "temperature": float(model.get("temperature", TEMPERATURE)),
            "max_tokens": int(model.get("max_tokens", MAX_TOKENS)),
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
            json.dumps(run_manifest, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    write_run_manifest()
    try:
        games = build_games_for_model(experiment, model["short_name"])
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
        source_manifest = json.loads(run_manifest_path.read_text(encoding="utf-8"))
        run_status = str(source_manifest.get("status", "")).strip().lower()
        if run_status != "completed":
            print(f"[aggregate skip] status={run_status or 'missing'}: {source}")
            continue
        relative = source.relative_to(OUTPUT_DIR)
        source_run = relative.parent.as_posix()
        with source.open("r", newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row.setdefault("model", relative.parts[0] if len(relative.parts) > 0 else "")
                row.setdefault("experiment", relative.parts[1] if len(relative.parts) > 1 else "")
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


# %%
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
            manifest["runs"].append(run_one_experiment(model, experiment_name, send_batch))
    except Exception as error:
        traceback.print_exc()
        manifest["runs"].append(
            {"model": model["short_name"], "status": "failed", "error": f"{type(error).__name__}: {error}"}
        )
    finally:
        free_model()
        write_manifest()


# %%
# In lai TOAN BO prompt/response cua MOI race chay xong trong lan nay.
if DEBUG_DUMP_RACE:
    all_race_ids = list_race_ids(OUTPUT_DIR)
    print(f"[debug] {len(all_race_ids)} race trong output. In toan bo:")
    for race_id, (model_name, risk, rep, _path) in all_race_ids:
        print(f"   {race_id}   ({model_name}, p_max={risk}, rep={rep})")
        dump_race_prompts(
            race_id,
            OUTPUT_DIR,
            max_prompt_chars=DEBUG_MAX_PROMPT_CHARS,
        )


# %%
import zipfile

n_race_rows = merge_csv_files("races.csv", OUTPUT_DIR / "ai_race_nplayer_all_models.csv")
n_player_rows = merge_csv_files("players.csv", OUTPUT_DIR / "ai_race_nplayer_players_all_models.csv")
print(f"Merged {n_race_rows} race rows and {n_player_rows} player rows.")

with zipfile.ZipFile(ZIP_PATH, "w", zipfile.ZIP_DEFLATED) as archive:
    for path in sorted(OUTPUT_DIR.rglob("*")):
        if path.is_file():
            archive.write(path, path.relative_to(OUTPUT_DIR.parent))

print(f"Results: {OUTPUT_DIR}")
print(f"Archive: {ZIP_PATH} ({ZIP_PATH.stat().st_size / 1024**2:.2f} MiB)")

incomplete_runs = [run for run in manifest["runs"] if run.get("status") != "completed"]
completed_runs = [run for run in manifest["runs"] if run.get("status") == "completed"]
if FAIL_ON_INCOMPLETE_RUN and (incomplete_runs or not completed_runs):
    raise RuntimeError(
        "Kaggle run is incomplete; inspect run_manifest.json and notebook logs. "
        f"completed={len(completed_runs)}, incomplete={len(incomplete_runs)}"
    )
