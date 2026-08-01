# Chạy pilot qua Kaggle Model Proxy (đường C)

Hướng dẫn ngắn để chạy 1 hoặc nhiều experiment config qua proxy (không cần GPU,
không cần push Kaggle Benchmark). Xem `docs/running-the-experiment.md` để so
sánh với đường A (GPU notebook) và đường B (Kaggle Benchmark).

## Chuẩn bị (một lần)

```bash
cd AI_Race_Experiment
python3 -m pip install --user uv
uv python install 3.12
uv venv --python 3.12 .venv-kaggle
uv pip install --python .venv-kaggle/bin/python -e ".[dev,api,analysis]"
```

Token Model Proxy sống ngắn hạn (~1 giờ), phải refresh trước mỗi lần chạy dài:

```bash
.venv-kaggle/bin/kaggle benchmarks auth -y
```

## Chạy 1 config

```bash
.venv-kaggle/bin/python -m ai_race.runner.run_experiment \
  ai_race/configs/experiment/<TÊN_CONFIG>.json \
  --output results/frontier/<THƯ_MỤC_MUỐN_LƯU>
```

Kết quả nằm ở `results/frontier/<THƯ_MỤC_MUỐN_LƯU>/<model-slug>/`:
`turns.jsonl`, `races.csv`, `players.csv`, `run_manifest.json`. Đọc field
`status` trong `run_manifest.json` để biết chạy xong (`completed`), đang chạy
(`running`), hay lỗi (`failed`).

## Chạy nhiều config nối tiếp — sửa `JOBS` rồi chạy

Copy đoạn dưới thành file (ví dụ `run_stage.py`), sửa list `JOBS`, chạy
`.venv-kaggle/bin/python run_stage.py`. Script tự refresh token trước mỗi
config, và **tự bỏ qua** config nào đã có `run_manifest.json` với
`status == "completed"` — chạy lại an toàn nếu bị đứt giữa chừng, không tốn
tiền chạy lại phần đã xong.

```python
"""Chạy nhiều experiment config tuần tự, mỗi cái refresh token riêng."""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent  # sửa nếu đặt file ở chỗ khác
PY = REPO / ".venv-kaggle/bin/python"
KAGGLE = REPO / ".venv-kaggle/bin/kaggle"

# (đường dẫn config, thư mục output) — SỬA DANH SÁCH NÀY MỖI LẦN MUỐN CHẠY KHÁC.
JOBS = [
    ("ai_race/configs/experiment/api_baseline_flashlite35.json", "results/frontier/baseline"),
    ("ai_race/configs/experiment/api_persona_baseline_neutral.json", "results/frontier/persona/R0_neutral"),
    # Thêm dòng ở đây cho config mới, ví dụ:
    # ("ai_race/configs/experiment/api_persona_baseline_XXX.json", "results/frontier/persona/XXX"),
]


def model_slug(name: str) -> str:
    import re
    return re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-").lower() or "model"


summary = []
for index, (config_rel, output_rel) in enumerate(JOBS, start=1):
    config_path = REPO / config_rel
    output_root = REPO / output_rel
    exp = json.loads(config_path.read_text(encoding="utf-8"))
    model = exp["models"][0]

    manifest_path = output_root / model_slug(model) / "run_manifest.json"
    if manifest_path.is_file():
        try:
            existing_status = json.loads(manifest_path.read_text(encoding="utf-8")).get("status")
        except json.JSONDecodeError:
            existing_status = None
        if existing_status == "completed":
            print(f"\n[{index}/{len(JOBS)}] {exp['name']}: đã completed, bỏ qua.", flush=True)
            summary.append((exp["name"], "SKIP", 0.0))
            continue

    print(f"\n{'=' * 70}", flush=True)
    print(f"[{index}/{len(JOBS)}] {exp['name']}  (model={model})", flush=True)
    print("=" * 70, flush=True)

    subprocess.run([str(KAGGLE), "benchmarks", "auth", "-y"], cwd=REPO, check=False)

    started = time.time()
    result = subprocess.run(
        [str(PY), "-m", "ai_race.runner.run_experiment", str(config_path), "--output", str(output_root)],
        cwd=REPO,
    )
    elapsed = time.time() - started
    status = "OK" if result.returncode == 0 else f"FAIL(rc={result.returncode})"
    summary.append((exp["name"], status, elapsed))
    print(f"\n--> {exp['name']}: {status} sau {elapsed / 60:.1f} phút", flush=True)

print(f"\n{'=' * 70}\nTỔNG KẾT STAGE\n{'=' * 70}")
for name, status, elapsed in summary:
    print(f"{status:20s} {elapsed / 60:5.1f} phút  {name}")

failed = [n for n, s, _ in summary if s.startswith("FAIL")]
sys.exit(1 if failed else 0)
```

**Chạy nền + xem log liên tục:**

```bash
.venv-kaggle/bin/python run_stage.py > run_stage.log 2>&1 &
tail -f run_stage.log
```

## Tạo config mới (persona khác, model khác)

Copy 1 file có sẵn trong `ai_race/configs/experiment/api_persona_baseline_*.json`
làm mẫu, đổi 2 chỗ:

- `"agents"`: tên file trong `ai_race/configs/agents/` (không có đuôi `.json`) —
  danh sách persona có sẵn: `persona_neutral`, `persona_risk_averse`,
  `persona_risk_seeking`, `persona_coop_coop`, `persona_adv_adv`,
  `persona_adv_coop`, `persona_coop_adv`.
- `"models"`: route đầy đủ có prefix hãng, ví dụ `"google/gemini-3-flash-preview"`.
  Xem model nào đang khả dụng bằng `.venv-kaggle/bin/kaggle b t models`.

**Giữ nguyên `"seed": 260726`** ở mọi config — đổi seed là mất khả năng so sánh
matched-pairs giữa các điều kiện (xem `CLAUDE.md`, mục Invariants).

Nếu hay gặp lỗi `429` (quá tải) khi chạy nhiều model song song, hạ
`proxyOptions.concurrency` xuống `2` và tăng `max_transport_retries` lên `6` —
đã xác nhận công thức này ổn định hơn `concurrency: 8` mặc định.

## Sau khi chạy: phân tích

```bash
.venv-kaggle/bin/python results/scripts/analyze_ai_race.py \
  --input results/frontier/<thư-mục-vừa-chạy> \
  --output results/derived/<tên-tuỳ-chọn> \
  --allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs \
  --allow-missing-persona-condition
```

4 flag `--allow-*` là bắt buộc cho runner local (manifest schema
`ai-race-results-v1` chưa đủ giàu thông tin cho primary analysis — xem
`docs/running-the-experiment.md`). **Không dùng 4 flag này để công bố kết quả
chính thức**, chỉ dùng cho pilot/audit.

Kiểm tra soi gương trước khi tin bất kỳ số liệu nào về "dẫn trước/bị bỏ lại":

```bash
.venv-kaggle/bin/python results/scripts/check_symmetry.py --input results/frontier/<thư-mục-vừa-chạy>
```
