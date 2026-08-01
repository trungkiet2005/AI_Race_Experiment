# Chạy pilot frontier GPT qua OpenAI API trực tiếp (backend `"openai"`)

Hướng dẫn thao tác để lắp API key + model rồi chạy 8 config
`ai_race/configs/experiment/openai_*.json` (baseline + 7 persona core) đã tạo sẵn.
Xem lý do và phạm vi ở [frontier-gpt-experiment-plan.md](frontier-gpt-experiment-plan.md);
so sánh với đường proxy ở [running-proxy-pilots.md](running-proxy-pilots.md). Khác biệt
quan trọng nhất: đường này **không qua Kaggle**, gọi thẳng `api.openai.com`, nên không có
bước `kaggle benchmarks auth`, không có token hết hạn theo giờ — nhưng cũng không có
concurrency/retry sẵn (xem mục "Bẫy" cuối file).

## 0. Môi trường

Dùng lại `.venv-kaggle` — đã có sẵn `openai`, `python-dotenv`, `pytest`, không cần tạo venv
mới:

```bash
cd AI_Race_Experiment
.venv-kaggle/bin/python -c "import openai, dotenv; print('deps ok', openai.__version__)"
```

Nếu báo thiếu module, cài lại:

```bash
uv pip install --python .venv-kaggle/bin/python -e ".[dev,api,analysis]"
```

## 1. Lắp API key

Thêm một dòng vào `.env` ở gốc repo (file này đã có trong `.gitignore`, không commit):

```
API_KEY_OPENAI=sk-...
```

`FAIRGAME`'s `OpenAIConnector` đọc biến này qua `os.getenv` sau khi
`llm_factory_connector.py` gọi `load_dotenv()` (không cần export tay ở shell, chỉ cần
`.env` nằm ở thư mục chạy lệnh — tức gốc repo). Kiểm tra nhanh:

```bash
.venv-kaggle/bin/python -c "
from dotenv import load_dotenv; import os
load_dotenv()
assert os.getenv('API_KEY_OPENAI'), 'chưa thấy API_KEY_OPENAI trong .env'
print('key loaded, độ dài:', len(os.getenv('API_KEY_OPENAI')))
"
```

## 2. Lắp model — chọn model id thật

**Không đoán tên model.** Sau khi có key, liệt kê model thật sự khả dụng cho key đó:

```bash
.venv-kaggle/bin/python -c "
from dotenv import load_dotenv; load_dotenv()
import os
from openai import OpenAI
client = OpenAI(api_key=os.environ['API_KEY_OPENAI'])
for m in client.models.list().data:
    print(m.id)
" | sort
```

Chọn ra các model id GPT muốn benchmark (ví dụ nhiều bản frontier khác nhau để trả lời
RQ6 — ổn định qua model scale/family).

### Điền vào cả 8 config cùng lúc

Không cần sửa tay từng file. Mỗi config nhận một **danh sách** model trong `"models"` —
runner tự tạo một thư mục kết quả riêng cho mỗi model trong danh sách đó (xem
`run_experiment` trong [run_experiment.py](../ai_race/runner/run_experiment.py)), nên chỉ
cần điền đúng một danh sách model vào cả 8 file:

```bash
.venv-kaggle/bin/python - <<'EOF'
import json
from pathlib import Path

# Sửa danh sách này theo model id thật đã xác nhận ở bước trên.
MODEL_IDS = [
    "gpt-4o",
    "gpt-4.1",
]

config_dir = Path("ai_race/configs/experiment")
targets = sorted(config_dir.glob("openai_*.json"))
assert targets, "không tìm thấy config openai_*.json nào"

for path in targets:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("backend") != "openai":
        continue
    data["models"] = MODEL_IDS
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("updated", path.name, "->", MODEL_IDS)
EOF
```

Kiểm tra lại không còn placeholder nào sót:

```bash
grep -l REPLACE_ME_openai_model_id ai_race/configs/experiment/openai_*.json && echo "CÒN SÓT" || echo "OK, hết placeholder"
```

## 3. Smoke test rẻ — trước khi tốn tiền cho pilot thật

Chạy 1 config, 1 treatment, 1 rep để xác nhận connectivity + parsing, chưa cần đụng đến
cả 8 config:

```bash
.venv-kaggle/bin/python - <<'EOF'
import json
from pathlib import Path

path = Path("ai_race/configs/experiment/openai_baseline.json")
data = json.loads(path.read_text(encoding="utf-8"))
data["games"] = ["ai_race_risk_60"]
data["repetitions"] = 1
data["models"] = data["models"][:1]  # chỉ model đầu tiên cho smoke test
Path("/tmp/openai_smoke.json").write_text(json.dumps(data, indent=2), encoding="utf-8")
EOF

.venv-kaggle/bin/python -m ai_race.runner.run_experiment \
  /tmp/openai_smoke.json --output /tmp/openai_smoke_out
```

Đọc `turns.jsonl` trong `/tmp/openai_smoke_out/<model-slug>/` — xác nhận response parse
được `ACTION: SAFE|UNSAFE`, không có `parse_failed`. Xong bước này mới sang pilot thật.

## 4. Chạy pilot đầy đủ — 8 config, tuần tự, resumable

Copy đoạn dưới thành `run_openai_stage.py` ở gốc repo rồi chạy. Cùng mẫu với script
`JOBS` trong [running-proxy-pilots.md](running-proxy-pilots.md#chạy-nhiều-config-nối-tiếp--sửa-jobs-rồi-chạy),
bỏ bước `kaggle benchmarks auth` vì không cần:

```python
"""Chạy 8 config openai_*.json tuần tự. Tự bỏ qua config đã completed."""
import json
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent
PY = REPO / ".venv-kaggle/bin/python"

JOBS = [
    ("ai_race/configs/experiment/openai_baseline.json", "results/frontier/openai/baseline"),
    ("ai_race/configs/experiment/openai_persona_baseline_neutral.json", "results/frontier/openai/persona/R0_neutral"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_averse.json", "results/frontier/openai/persona/Rminus_risk_averse"),
    ("ai_race/configs/experiment/openai_persona_baseline_risk_seeking.json", "results/frontier/openai/persona/Rplus_risk_seeking"),
    ("ai_race/configs/experiment/openai_persona_baseline_coop_coop.json", "results/frontier/openai/persona/S_CC_coop_coop"),
    ("ai_race/configs/experiment/openai_persona_baseline_adv_adv.json", "results/frontier/openai/persona/S_AA_adv_adv"),
    ("ai_race/configs/experiment/openai_persona_baseline_adv_coop.json", "results/frontier/openai/persona/S_AC_adv_coop"),
    ("ai_race/configs/experiment/openai_persona_baseline_coop_adv.json", "results/frontier/openai/persona/S_CA_coop_adv"),
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
        print(f"\n[{index}/{len(JOBS)}] {exp['name']}: mọi model đã completed, bỏ qua.", flush=True)
        summary.append((exp["name"], "SKIP", 0.0))
        continue

    print(f"\n{'=' * 70}", flush=True)
    print(f"[{index}/{len(JOBS)}] {exp['name']}  (models={models})", flush=True)
    print("=" * 70, flush=True)

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

Chạy nền + xem log liên tục (khuyến nghị, vì gọi API tuần tự từng prompt sẽ chậm hơn
đường proxy — xem mục "Bẫy"):

```bash
.venv-kaggle/bin/python run_openai_stage.py > run_openai_stage.log 2>&1 &
tail -f run_openai_stage.log
```

Đứt giữa chừng (hết quota, mất mạng) thì chạy lại y nguyên lệnh trên — script tự bỏ qua
model/config đã `completed`, không tốn tiền chạy lại phần đã xong. **Lưu ý:** logic
skip ở đây kiểm tra theo *từng model trong một config*, khác bản proxy chỉ kiểm tra
model đầu tiên — vì một config `openai_*.json` giờ có thể chứa nhiều model GPT cùng lúc.

## 5. Soi gương trước khi tin bất kỳ số liệu race-position nào

```bash
for d in results/frontier/openai/baseline results/frontier/openai/persona/*/; do
  echo "=== $d ==="
  .venv-kaggle/bin/python results/scripts/check_symmetry.py --input "$d"
done
```

Script exit 1 nếu >40% race bị symmetry collapse (hai seat luôn chọn giống nhau). Gặp
vậy thì **dừng lại**, đừng scale lên rep nhiều hơn — thử persona bất đối xứng
(`S_AC`/`S_CA`) đã có sẵn để xác nhận, hoặc xem lại `temperature` trong `proxyOptions`
của config (đang là `0.7`, xem cảnh báo ở mục "Bẫy").

## 6. Phân tích (audit, chưa phải kết quả cuối)

```bash
.venv-kaggle/bin/python results/scripts/analyze_ai_race.py \
  --input results/frontier/openai --output results/derived/openai_pilot_audit --fit-logit \
  --allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs \
  --allow-missing-persona-condition
```

Đọc theo thứ tự ở
[running-the-experiment.md](running-the-experiment.md#bước-3--đọc-output): trước tiên
`analysis_manifest.json` rồi `parse_failures.csv` — khác 0 thì dừng, đừng diễn giải hành
vi gì cả. **4 flag `--allow-*` chỉ dùng cho pilot/audit**, không dùng cho kết quả công bố.

## 7. Freeze rồi chạy confirmatory

Sau khi pilot sạch (symmetry ổn, parse failure = 0):

1. Đổi `"runPhase": "pilot"` → `"confirmatory"` trong cả 8 file `openai_*.json`.
2. Sửa `"repetitions"` nếu muốn tăng số rep cho bản chính thức (khác pilot 10).
3. Chạy lại đúng `run_openai_stage.py` với output root khác (ví dụ
   `results/frontier/openai_confirmatory/...`) để không lẫn với dữ liệu pilot.
4. Phân tích chính thức, **bỏ 4 flag `--allow-*`**:
   ```bash
   .venv-kaggle/bin/python results/scripts/analyze_ai_race.py \
     --input results/open_source --input results/frontier \
     --output results/derived/ai_race_analysis --fit-logit
   ```

## Bẫy riêng của đường `"openai"`

| Bẫy | Hậu quả | Cách tránh |
|---|---|---|
| Gọi API tuần tự, không `ThreadPoolExecutor` | Pilot 8 config × nhiều model chậm hơn hẳn đường proxy | Chạy nền (`&` + `tail -f`), ước lượng thời gian từ smoke test trước khi chạy full |
| Không có `max_transport_retries`/`timeout` cấu hình được | Một lỗi transport giữa chừng làm cả model đó `FAIL`, phải chạy lại (script tự skip phần đã `completed` nên không tốn tiền chạy lại) | Theo dõi log; nếu 429 lặp lại liên tục, dừng, đợi rồi chạy lại `run_openai_stage.py` |
| Quên đổi `MODEL_IDS` ở bước 2 trước khi chạy pilot thật | Chạy nhầm placeholder `REPLACE_ME_openai_model_id`, lỗi ngay từ request đầu (rẻ, nhưng phí thời gian) | `grep -l REPLACE_ME` trước khi chạy stage script |
| `temperature` 0.7 chưa từng kiểm chứng riêng cho backend này (chỉ mới kiểm chứng ở đường proxy) | Có thể symmetry collapse khác | Luôn chạy `check_symmetry.py` (bước 5) trước khi tin kết quả |
| Chạy `openai_baseline` hôm nay, các persona cell hôm khác | `protocol_signature` khác nhau → persona trùng khít với batch, analyser từ chối | Chạy hết 8 config trong cùng một lần gọi `run_openai_stage.py` |
| Trộn output pilot (`results/frontier/openai/`) với confirmatory | Mất tính preregistered, phải phân biệt bằng `run_phase` | Dùng thư mục output khác nhau cho pilot và confirmatory (bước 7) |
