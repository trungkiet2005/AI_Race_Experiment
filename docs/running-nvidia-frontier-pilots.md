# Chạy pilot DeepSeek qua NVIDIA API Catalog (backend `"nvidia"`)

Hướng dẫn thao tác để lắp API key + chạy 8 config
`ai_race/configs/experiment/nvidia_*.json` (baseline + 7 persona core) đã tạo sẵn, dùng model
`deepseek-ai/deepseek-v4-flash` và `deepseek-ai/deepseek-v4-pro` qua endpoint OpenAI-compatible
của NVIDIA (`https://integrate.api.nvidia.com/v1`). Cùng mẫu với đường OpenAI trực tiếp —
xem [running-openai-frontier-pilots.md](running-openai-frontier-pilots.md) để so sánh. Việc gọi
API nằm trong [ai_race/models/nvidia_direct.py](../ai_race/models/nvidia_direct.py).

## 0. Môi trường

Dùng lại `.venv-kaggle` — cùng gói `openai` (NVIDIA dùng chung SDK, chỉ đổi `base_url`):

```bash
cd AI_Race_Experiment
.venv-kaggle/bin/python -c "import openai, dotenv; print('deps ok', openai.__version__)"
```

## 1. Lắp API key

Thêm một dòng vào `.env` ở gốc repo (đã có trong `.gitignore`, không commit):

```
NVIDIA_API_KEY=nvapi-...
```

Kiểm tra nhanh:

```bash
.venv-kaggle/bin/python -c "
from dotenv import load_dotenv; import os
load_dotenv()
assert os.getenv('NVIDIA_API_KEY'), 'chưa thấy NVIDIA_API_KEY trong .env'
print('key loaded, độ dài:', len(os.getenv('NVIDIA_API_KEY')))
"
```

## 2. Smoke test rẻ — trước khi chạy pilot thật

```bash
.venv-kaggle/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ['NVIDIA_API_KEY'], base_url='https://integrate.api.nvidia.com/v1')
r = client.chat.completions.create(
    model='deepseek-ai/deepseek-v4-flash',
    messages=[{'role': 'user', 'content': 'Reply with exactly one word: OK'}],
    max_tokens=16,
)
print(r.choices[0].message.content)
"
```

Rồi chạy 1 config, 1 treatment, 1 rep qua đúng engine của project:

```bash
.venv-kaggle/bin/python - <<'EOF'
import json
from pathlib import Path

path = Path("ai_race/configs/experiment/nvidia_baseline.json")
data = json.loads(path.read_text(encoding="utf-8"))
data["games"] = ["ai_race_risk_60"]
data["repetitions"] = 1
data["models"] = data["models"][:1]
Path("/tmp/nvidia_smoke.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
EOF

.venv-kaggle/bin/python -m ai_race.runner.run_experiment \
  /tmp/nvidia_smoke.json --output /tmp/nvidia_smoke_out
```

Đọc `turns.jsonl` trong `/tmp/nvidia_smoke_out/<model-slug>/` — xác nhận response parse được
`ACTION: SAFE|UNSAFE`, không có `parse_failed`, trước khi chạy tiếp.

## 3. Về `thinking`

DeepSeek V4 hỗ trợ chế độ suy luận ẩn (`chat_template_kwargs.thinking`). 8 config này để
**`"thinking": false` mặc định** — bật `thinking: true` có rủi ro giống hệt gpt-5-series ở
đường OpenAI: model có thể tiêu hết `max_tokens` cho suy luận ẩn rồi trả về message rỗng, tính
là `parse_failed` (và một parse failure làm hỏng cả race, xem `CLAUDE.md`). Nếu muốn bật
`thinking`, tăng `max_tokens` lên rõ rệt trước, và chạy smoke test lại.

## 4. Chạy pilot đầy đủ — 8 config, tuần tự, resumable

Copy đoạn dưới thành file, sửa nếu cần, chạy `.venv-kaggle/bin/python run_nvidia_stage.py`. Cùng
mẫu `JOBS` như [running-proxy-pilots.md](running-proxy-pilots.md) và
[running-openai-frontier-pilots.md](running-openai-frontier-pilots.md), không cần bước
`kaggle benchmarks auth`:

```python
"""Chạy 8 config nvidia_*.json tuần tự. Tự bỏ qua config đã completed."""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = REPO / ".venv-kaggle/bin/python"

JOBS = [
    ("ai_race/configs/experiment/nvidia_baseline.json", "results/frontier/nvidia/baseline"),
    ("ai_race/configs/experiment/nvidia_persona_baseline_neutral.json", "results/frontier/nvidia/persona/R0_neutral"),
    ("ai_race/configs/experiment/nvidia_persona_baseline_risk_averse.json", "results/frontier/nvidia/persona/Rminus_risk_averse"),
    ("ai_race/configs/experiment/nvidia_persona_baseline_risk_seeking.json", "results/frontier/nvidia/persona/Rplus_risk_seeking"),
    ("ai_race/configs/experiment/nvidia_persona_baseline_coop_coop.json", "results/frontier/nvidia/persona/S_CC_coop_coop"),
    ("ai_race/configs/experiment/nvidia_persona_baseline_adv_adv.json", "results/frontier/nvidia/persona/S_AA_adv_adv"),
    ("ai_race/configs/experiment/nvidia_persona_baseline_adv_coop.json", "results/frontier/nvidia/persona/S_AC_adv_coop"),
    ("ai_race/configs/experiment/nvidia_persona_baseline_coop_adv.json", "results/frontier/nvidia/persona/S_CA_coop_adv"),
]


def model_slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower() or "model"


summary = []
for index, (config_rel, output_rel) in enumerate(JOBS, start=1):
    config_path = REPO / config_rel
    output_root = REPO / output_rel
    exp = json.loads(config_path.read_text(encoding="utf-8"))
    models = exp["models"]

    all_done = all(
        (output_root / model_slug(m) / "run_manifest.json").is_file()
        and json.loads((output_root / model_slug(m) / "run_manifest.json").read_text()).get("status") == "completed"
        for m in models
    )
    if all_done:
        print(f"\n[{index}/{len(JOBS)}] {exp['name']}: đã completed, bỏ qua.", flush=True)
        summary.append((exp["name"], "SKIP", 0.0))
        continue

    print(f"\n{'=' * 70}\n[{index}/{len(JOBS)}] {exp['name']}  (models={models})\n{'=' * 70}", flush=True)
    started = time.time()
    result = subprocess.run(
        [str(PY), "-m", "ai_race.runner.run_experiment", str(config_path), "--output", str(output_root)],
        cwd=REPO,
    )
    elapsed = time.time() - started
    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    summary.append((exp["name"], status, elapsed))
    print(f"\n--> {exp['name']}: {status} sau {elapsed / 60:.1f} phút", flush=True)

print(f"\n{'=' * 70}\nTỔNG KẾT\n{'=' * 70}")
for name, status, elapsed in summary:
    print(f"{status:20s} {elapsed / 60:5.1f} phút  {name}")

failed = [n for n, s, _ in summary if s.startswith("FAIL")]
sys.exit(1 if failed else 0)
```

## Bẫy đã biết trước (chưa xác nhận thực nghiệm — chưa có key để test)

| Bẫy | Hậu quả | Giảm thiểu |
|---|---|---|
| `thinking: true` làm model tiêu hết `max_tokens` cho suy luận ẩn | Message rỗng, `parse_failed`, hỏng cả race | Giữ `thinking: false`; nếu cần bật thì tăng mạnh `max_tokens` và smoke test trước |
| Chưa rõ NVIDIA có hỗ trợ `max_tokens` chuẩn hay cần tên tham số khác cho từng model | Lỗi 400 ngay từ request đầu | Chạy đúng bước 2 (smoke test) trước khi tin bất kỳ config nào |
| Chi phí thật (không phải Kaggle free-tier) | Full grid tốn tiền ngoài dự kiến | Freeze sau pilot nhỏ, không chạy full 8 config × N rep lớn trước khi xác nhận |
| Trộn pilot với confirmatory | Mất tính preregistered | `run_phase` gate như mọi đường khác |
