# %%
"""AI Race baseline — Kaggle GPU notebook (Internet OFF).

Notebook này dùng đúng runner của project:

* ``ai_race.runner.run_experiment.build_games_for_model``
* ``ai_race.runner.batch.run_games_batched``
* ``ai_race.dataio.recorder`` cho turns/races/players
* ``ai_race.models.factory`` cho backend vLLM/Transformers

Repo input được tự dò bằng hai marker ``ai_race/`` và ``FAIRGAME/``, sau đó copy
vào ``/kaggle/working/ai_race_repo`` để Python có thể import và ghi cache. Các
model được nạp, chạy, giải phóng tuần tự.

Hai chế độ chạy được gói vào một switch duy nhất, ``ENGINE_PROFILE`` ngay dưới đây —
đổi máy/teammate thì chỉ sửa một dòng, không phải lục lại từng model:

* ``"vllm"``         — cần wheelhouse offline đã audit (``VLLM_WHEELS_DIR``), batch
  thật (``BATCH_SIZE`` prompt/lượt), throughput cao. Dùng khi đã có wheelhouse.
* ``"transformers"`` — dùng thẳng image Kaggle, không cài gì, không cần wheelhouse.
  Vì runner LUÔN truyền seed riêng cho mỗi quyết định (invariant của repo), backend
  này buộc phải sinh TỪNG PROMPT MỘT — chậm hơn vllm rất nhiều (~1 generation/lượt,
  không phải ``BATCH_SIZE``). Dùng khi chưa có wheelhouse hoặc máy không có vLLM.
"""

# %%
# Cấu hình người dùng — sửa path theo các input đã add trong Kaggle.
import os
from pathlib import Path

# Switch duy nhất giữa 2 chế độ — xem so sánh ở docstring trên đầu file.
ENGINE_PROFILE = "vllm"  # đổi thành "transformers" nếu chưa có wheelhouse vLLM

MODELS = [
    {
        "path": "/kaggle/input/models/qwen-lm/qwen2.5/transformers/72b-instruct/1",
        "short_name": "qwen2.5-72b-instruct",
        "engine": ENGINE_PROFILE,
        # 72B KHÔNG vừa bf16: 72,7 tỉ tham số × 2 byte ≈ 145 GB, trong khi RTX PRO
        # 6000 có 96 GB. Nếu chạy bằng "transformers" thì bắt buộc quantize. NF4 ≈
        # 0,55 byte/tham số → ≈ 40 GB, còn dư chỗ cho KV cache. bnb-8bit (≈ 73 GB)
        # cũng vừa nhưng sát trần. Khi ENGINE_PROFILE="vllm" thì bỏ qua engine_overrides
        # này (vLLM tự đọc quantization_config trong checkpoint nếu có, hoặc set
        # "quantization" tường minh trong engine_overrides theo format vLLM: awq/gptq/
        # fp8/bitsandbytes — KHÔNG dùng "bnb-4bit" ở đây, đó là tên riêng của transformers).
        #
        # LƯU Ý: quantization phải nằm trong engine_overrides. Đặt "quantization" ở
        # cấp trên cùng của dict này sẽ bị BỎ QUA IM LẶNG — offline_settings_for()
        # chỉ đọc engine_overrides, và model sẽ cố nạp bf16 rồi OOM.
        "engine_overrides": {"quantization": "bnb-4bit"} if ENGINE_PROFILE == "transformers" else {},
        # Các override tùy chọn khác:
        # "temperature": 0.7,
        # "max_tokens": 256,
    },
    # Thêm model khác tại đây; notebook sẽ chạy lần lượt, không nạp đồng thời
    # (mỗi model được free_model() giải phóng VRAM trước khi nạp checkpoint kế tiếp).
    #
    # ĐỪNG thêm một entry thứ hai cho cùng checkpoint: short_name quyết định thư mục
    # output, nên hai entry trùng short_name sẽ ghi đè lên nhau (RunJournal mở với
    # reset=True). Một entry riêng có thể override "engine" khác ENGINE_PROFILE nếu
    # cần trộn 2 backend trong cùng lần chạy.
]

# Dataset đã stage: https://www.kaggle.com/datasets/nguyenlamphuquy/ai-race-experiment
#
# Kaggle dùng hai layout mount tuỳ notebook: chỉ *slug* (không kèm username), hoặc
# datasets/<owner>/<slug>. Liệt kê cả hai và lấy cái nào tồn tại — pin đúng một
# chuỗi thì đổi notebook là hỏng, còn bỏ trống hoàn toàn thì mất cái chốt chặn
# "add nhầm dataset". Ứng viên đầu tiên khớp sẽ được dùng.
#
# Username chỉ nằm trong dataset id dùng cho CLI (nguyenlamphuquy/ai-race-experiment).
# Đặt REPO_INPUT_DIRS = None để bỏ chốt chặn và tự dò mọi input.
REPO_INPUT_DIRS = [
    "/kaggle/input/datasets/nguyenlamphuquy/ai-race-experiment",
    "/kaggle/input/ai-race-experiment",
]

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

# Cùng switch với ENGINE_PROFILE ở trên — model nào không khai "engine" riêng sẽ
# dùng giá trị này.
DEFAULT_ENGINE = ENGINE_PROFILE
TEMPERATURE = 0.7
MAX_TOKENS = 256
LOGPROBS = 0  # Backend transformers không hỗ trợ; >0 sẽ raise ngay khi build config.

# --- Bốn hằng dưới đây CHỈ có tác dụng với backend vllm ---------------------
# `transformers_init_kwargs` không đọc chúng, nên ở cấu hình hiện tại chúng bị bỏ
# qua hoàn toàn. Giữ lại để đổi ngược về vLLM không phải viết lại. Backend
# transformers luôn nạp bf16 với `device_map="auto"`, tự trải qua GPU đang có.
MAX_MODEL_LEN = 4096
GPU_MEMORY_UTILIZATION = 0.90
TENSOR_PARALLEL_SIZE = 1
ENFORCE_EAGER = True
# ---------------------------------------------------------------------------

# Chỉ còn tác dụng khi seeds=None. Runner LUÔN truyền seed cho từng quyết định
# (đó là invariant của repo, không phải tuỳ chọn), nên backend transformers rơi
# về sinh TỪNG PROMPT MỘT: một forward pass gộp dùng chung một torch RNG nên
# không thể tôn trọng seed riêng của mỗi (rep, round, agent). Nghĩa là throughput
# ở đây là ~1 generation/lượt, không phải 128.
BATCH_SIZE = 128
MAX_PARSE_RETRIES_OVERRIDE = None  # None = dùng maxParseRetries trong experiment
FAIL_ON_INCOMPLETE_RUN = True

# Chỉ chạy khi có model nào khai engine="vllm". Với cấu hình transformers hiện tại
# thì cell cài vLLM tự bỏ qua.
INSTALL_VLLM_IF_MISSING = True
# Dataset wheelhouse đã audit: vllm==0.26.0, build bởi pip download (cp312,
# manylinux_2_28_x86_64, cu130) từ máy local, xem
# nguyenlamphuquy/vllm-0-26-0-wheelhouse-cu130-py312 (184 wheels, manifest
# SHA-256 tại manifest.json).
VLLM_WHEELS_DIR = "/kaggle/input/datasets/nguyenlamphuquy/vllm-0-26-0-wheelhouse-cu130-py312"

# Wheelhouse bitsandbytes cho quantization bnb-4bit/bnb-8bit ở backend transformers.
# Dataset: nguyenlamphuquy/ai-race-bnb-wheels (private) — bitsandbytes==0.50.0,
# wheel py3-none-manylinux_2_24_x86_64 nên độc lập phiên bản Python; chứa sẵn
# libbitsandbytes_cuda128.so khớp torch 2.10.0+cu128 của image. Kaggle tự giải nén
# nên wheel nằm ở gốc dataset. Liệt kê cả hai layout mount như REPO_INPUT_DIRS.
BNB_WHEELS_DIRS = [
    "/kaggle/input/datasets/nguyenlamphuquy/ai-race-bnb-wheels",
    "/kaggle/input/ai-race-bnb-wheels",
]

# Debug: in lại chuỗi prompt của một race sau khi chạy xong (đọc từ turns.jsonl,
# không gọi lại model). False = tắt · True = lấy race đầu tiên · "<game_id>" =
# chỉ đích danh. Cell debug in sẵn danh sách game_id để copy vào đây.
DEBUG_DUMP_RACE = False
# Ghế 1 chỉ khác ghế 0 ở hoán vị danh tính, nên mặc định chỉ in phần khác biệt.
DEBUG_DIFF_SECOND_SEAT = True
DEBUG_MAX_PROMPT_CHARS = None  # None = in trọn prompt; đặt số để cắt bớt

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
    """Prefer a configured dataset mount, then fall back to discovery.

    Naming the expected paths turns "wrong dataset added" into an explicit error
    instead of a silent fallback onto some other input that happens to contain the
    two marker directories — which would run a different source revision than the
    one the manifest is about to claim.

    Several candidates are allowed because Kaggle mounts a dataset either under its
    bare slug or under datasets/<owner>/<slug> depending on the notebook. Both name
    the same dataset, so trying them in order costs nothing and keeps the guard.
    """
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
# Cài bitsandbytes offline nếu có model xin quantize bnb-* và image chưa có.
#
# 72B không vừa bf16 trên 96 GB nên phải quantize, mà backend transformers chỉ
# nhận bnb-4bit/bnb-8bit — tức bắt buộc có wheel bitsandbytes. Internet OFF nên
# nó phải đến từ một Dataset đã stage sẵn.


def verify_wheelhouse(wheels_dir, label):
    """Đối chiếu từng wheel với SHA-256 trong manifest trước khi cài.

    Một wheelhouse chứa shared object CUDA; cài nhầm bản là hỏng theo kiểu khó
    truy, nên kiểm hash là bắt buộc chứ không phải cẩn thận thừa.
    """
    wheels_dir = Path(wheels_dir)
    if not wheels_dir.is_dir():
        raise FileNotFoundError(f"Không tìm thấy {label} wheelhouse tại {wheels_dir}")
    manifest_path = wheels_dir / "manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"{label} wheelhouse thiếu manifest SHA-256: {manifest_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entries = manifest.get("files")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"{label} manifest.files phải là list không rỗng")
    declared = set()
    for entry in entries:
        if not isinstance(entry, dict) or not {"name", "sha256"} <= set(entry):
            raise ValueError(f"{label}: mỗi entry cần name và sha256")
        name = str(entry["name"])
        if Path(name).name != name or not name.endswith(".whl"):
            raise ValueError(f"{label}: tên wheel không hợp lệ: {name!r}")
        if name in declared:
            raise ValueError(f"{label}: tên wheel trùng: {name}")
        declared.add(name)
        path = wheels_dir / name
        if not path.is_file():
            raise FileNotFoundError(f"{label}: wheel khai trong manifest nhưng thiếu: {path}")
        if "bytes" in entry and path.stat().st_size != int(entry["bytes"]):
            raise RuntimeError(f"{label}: lệch kích thước: {path.name}")
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        if digest.hexdigest().lower() != str(entry["sha256"]).lower():
            raise RuntimeError(f"{label}: lệch SHA-256: {path.name}")
    actual = {p.name for p in wheels_dir.glob("*.whl")}
    if actual != declared:
        raise RuntimeError(
            f"{label}: nội dung wheelhouse không khớp manifest; "
            f"thừa={sorted(actual - declared)}, thiếu={sorted(declared - actual)}"
        )
    return manifest


def find_wheelhouse(candidates):
    for candidate in candidates or []:
        path = Path(candidate)
        if (path / "manifest.json").is_file():
            return path
    return None


needs_bnb = any(
    str((model.get("engine_overrides") or {}).get("quantization", "")).lower().startswith("bnb")
    for model in MODELS
)
has_bnb = importlib.util.find_spec("bitsandbytes") is not None

if needs_bnb and not has_bnb:
    bnb_dir = find_wheelhouse(BNB_WHEELS_DIRS)
    if bnb_dir is None:
        raise FileNotFoundError(
            "MODELS xin quantization bnb-* nhưng image chưa có bitsandbytes và không "
            f"tìm thấy wheelhouse ở {BNB_WHEELS_DIRS}. Add Dataset "
            "'nguyenlamphuquy/ai-race-bnb-wheels' vào notebook, hoặc bỏ "
            "engine_overrides.quantization (khi đó 72B sẽ OOM ở bf16)."
        )
    bnb_manifest = verify_wheelhouse(bnb_dir, "bitsandbytes")
    subprocess.check_call(
        [
            sys.executable, "-m", "pip", "install",
            "--no-index", "--only-binary=:all:", f"--find-links={bnb_dir}",
            "bitsandbytes",
        ]
    )
    importlib.invalidate_caches()
    import bitsandbytes  # noqa: F401  — fail ngay tại đây thay vì lúc nạp weight
    print(f"Installed bitsandbytes offline from {bnb_dir}")
    print(f"  target runtime: {bnb_manifest.get('target_runtime')}")
elif needs_bnb:
    print("bitsandbytes is already available in this Kaggle image.")
else:
    print("No configured model requests bnb quantization.")


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
# Debug: in lại toàn bộ chuỗi prompt của MỘT race.
#
# Đọc từ turns.jsonl đã ghi, không chạy lại model — nên rẻ, và xem được đúng cái
# prompt đã thực sự gửi đi chứ không phải cái ta nghĩ là đã gửi.
#
# Mặc định chỉ in đầy đủ prompt của ghế 0 và phần KHÁC BIỆT của ghế 1. Hai prompt
# trong cùng một vòng chỉ khác nhau ở hoán vị danh tính (~2.100 ký tự giống hệt),
# nên in cả hai bản đầy đủ chỉ làm trôi log. Đặt DEBUG_DIFF_SECOND_SEAT = False
# nếu muốn xem trọn vẹn cả hai.
import difflib


def list_race_ids(output_dir=None, limit=None):
    """Liệt kê game_id có trong output, kèm model và mức rủi ro."""
    output_dir = Path(output_dir or OUTPUT_DIR)
    seen = {}
    for path in sorted(output_dir.rglob("turns.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen.setdefault(
                row["game_id"],
                (row.get("model"), row.get("max_private_risk"), row.get("rep"), path),
            )
    items = list(seen.items())
    return items[:limit] if limit else items


def _load_race_turns(game_id=None, output_dir=None):
    """Trả về (game_id, các lượt đã sắp xếp). game_id=None -> lấy race đầu tiên."""
    output_dir = Path(output_dir or OUTPUT_DIR)
    rows = []
    for path in sorted(output_dir.rglob("turns.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if game_id is None or row["game_id"] == game_id:
                rows.append(row)
        if rows and game_id is None:
            # Chốt luôn race đầu tiên gặp được, đừng gom cả thư mục.
            game_id = rows[0]["game_id"]
            rows = [r for r in rows if r["game_id"] == game_id]
            break
        if rows:
            break
    rows.sort(key=lambda r: (int(r["round"]), int(r.get("player_index", 0))))
    return game_id, rows


def dump_race_prompts(
    game_id=None,
    output_dir=None,
    diff_second_seat=True,
    max_prompt_chars=None,
    show_response=True,
):
    """In từng vòng của một race: prompt gửi đi, phản hồi thô, hành động đã parse."""
    game_id, rows = _load_race_turns(game_id, output_dir)
    if not rows:
        print(f"[debug] không tìm thấy lượt nào cho game_id={game_id!r}")
        return

    head = rows[0]
    print("=" * 78)
    print(f"RACE   {game_id}")
    print(f"model  {head.get('model')}   p_max={head.get('max_private_risk')}   "
          f"rep={head.get('rep')}   seed={head.get('game_seed')}")
    print(f"prompt_version={head.get('prompt_version')}  "
          f"persona={head.get('persona_condition')}  "
          f"vòng={max(int(r['round']) for r in rows)}")
    print("=" * 78)

    for round_number in sorted({int(r["round"]) for r in rows}):
        seats = [r for r in rows if int(r["round"]) == round_number]
        print(f"\n{'─' * 78}\nVÒNG {round_number}\n{'─' * 78}")
        reference = None
        for seat in seats:
            label = f"[{seat.get('player')} idx={seat.get('player_index')}]"
            print(f"\n{label}  seed={seat.get('sampling_seed')}  "
                  f"gap_before={seat.get('progress_gap_before')}")
            prompt = seat.get("prompt") or ""
            if reference is None or not diff_second_seat:
                shown = prompt if max_prompt_chars is None else prompt[:max_prompt_chars]
                print(shown)
                if max_prompt_chars is not None and len(prompt) > max_prompt_chars:
                    print(f"... (cắt bớt {len(prompt) - max_prompt_chars} ký tự)")
                reference = prompt
            else:
                diff = [
                    line
                    for line in difflib.unified_diff(
                        reference.split("\n"), prompt.split("\n"), lineterm="", n=0
                    )
                    if line.startswith(("+", "-"))
                    and not line.startswith(("+++", "---"))
                ]
                print(f"(khác ghế trước {len(diff)} dòng; đặt "
                      f"diff_second_seat=False để in đầy đủ)")
                for line in diff:
                    print("   " + line)
            if show_response:
                print(f"  -> raw     : {seat.get('raw_response')!r}")
                print(f"  -> action  : {seat.get('action')}   "
                      f"parse_failed={seat.get('parse_failed')}   "
                      f"retry={seat.get('retry_count')}")

    print(f"\n{'=' * 78}\nQUỸ ĐẠO")
    for player in dict.fromkeys(r.get("player") for r in rows):
        trail = [
            r["action"][0].upper()
            for r in rows
            if r.get("player") == player
        ]
        print(f"  {player:12s} {' '.join(trail)}")
    print("=" * 78)


if DEBUG_DUMP_RACE:
    target = None if DEBUG_DUMP_RACE is True else str(DEBUG_DUMP_RACE)
    available = list_race_ids(limit=8)
    print(f"[debug] {len(list_race_ids())} race trong output; 8 cái đầu:")
    for race_id, (model_name, risk, rep, _) in available:
        print(f"   {race_id}   ({model_name}, p_max={risk}, rep={rep})")
    print()
    dump_race_prompts(
        target,
        diff_second_seat=DEBUG_DIFF_SECOND_SEAT,
        max_prompt_chars=DEBUG_MAX_PROMPT_CHARS,
    )


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
