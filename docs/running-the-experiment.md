# Hướng dẫn chạy AI Race

Trả lời ngắn cho câu hỏi "dataset hay self-contained": **cả hai, tùy đường chạy.**

| Đường | Dùng cho | Source lên Kaggle kiểu gì |
|---|---|---|
| **A. GPU notebook** — `kaggle/experiments/baseline.py` | model open-weight (Qwen, Llama, Gemma) | **Phải stage repo thành Kaggle Dataset** rồi Add Input. Notebook `import ai_race` và `FAIRGAME` từ đó |
| **B. Kaggle Benchmark** — `kaggle/benchmarks/ai_race_baseline.py` | model frontier qua Model Proxy | **Self-contained**, push một file duy nhất, không cần dataset |
| **C. Local + Model Proxy** — `configs/experiment/api_baseline.json` | model hosted, chạy từ máy bạn | Không lên Kaggle gì cả, chỉ cần `.env` |

Phân tích thì **luôn chạy local**, không có bước nào trên Kaggle.

---

## Chuẩn bị local (một lần)

```bash
pip install -e ".[dev,analysis]"
python3 -m pytest          # 89 passed
```

`dev` cho pytest, `analysis` cho pandas/scipy/statsmodels mà analyser cần.

Chạy thử không cần model nào, để chắc pipeline sạch:

```bash
python3 -m ai_race.runner.run_experiment \
  ai_race/configs/experiment/baseline.json --mock random --output /tmp/smoke
```

---

## Đường A — GPU notebook cho model open-weight

### A1. Vì sao phải stage source

`kaggle/experiments/baseline.py` **không** self-contained. Nó gọi thẳng runner của project:

```python
from ai_race.runner.run_experiment import build_games_for_model
from ai_race.runner.batch import run_games_batched
from ai_race.dataio.recorder import RunJournal
from ai_race.models import factory
```

Hàm `find_repo_input()` quét `/kaggle/input` tìm thư mục chứa **đồng thời** `ai_race/` và
`FAIRGAME/`, rồi copy sang `/kaggle/working/ai_race_repo` (vì `/kaggle/input` là read-only,
Python cần chỗ ghi `__pycache__`). Không có input đó thì notebook dừng ngay:

> `Không tìm thấy repo chứa đồng thời ai_race/ và FAIRGAME/ dưới /kaggle/input.`

### A2. Dataset (đã tạo)

**Đã stage sẵn:** <https://www.kaggle.com/datasets/nguyenlamphuquy/ai-race-experiment>
(private, ~4.9 MB, 205 file).

Đường mount trong notebook là **`/kaggle/input/datasets/nguyenlamphuquy/ai-race-experiment`**
hoặc **`/kaggle/input/ai-race-experiment`** (slug trần, không kèm username) tuỳ
notebook. Notebook khai cả hai và lấy cái nào có thật; xem [A3](#a3-inputs-cần-add-vào-notebook).
Username chỉ bắt buộc trong dataset id dùng cho CLI (`nguyenlamphuquy/ai-race-experiment`).

Script tái tạo / cập nhật:

```bash
STAGE=/tmp/ai_race_dataset
rsync -a --delete-excluded \
  --exclude='.git' --exclude='arXiv-2607.26034v1' --exclude='__pycache__' \
  --exclude='*.pyc' --exclude='.venv' --exclude='*.egg-info' --exclude='.env' \
  --exclude='results/derived' --exclude='.pytest_cache' \
  ./ "$STAGE"/

cat > "$STAGE/dataset-metadata.json" <<'JSON'
{
  "title": "AI Race Experiment",
  "id": "nguyenlamphuquy/ai-race-experiment",
  "licenses": [{"name": "Apache 2.0"}]
}
JSON

kaggle datasets version -p "$STAGE" -m "sync <git-sha>" --dir-mode zip
# lần đầu tiên thì dùng: kaggle datasets create -p "$STAGE" --dir-mode zip
```

`--dir-mode zip` nén từng thư mục top-level để upload nhanh; Kaggle **tự giải nén** nên
cấu trúc thư mục giữ nguyên khi mount.

`arXiv-2607.26034v1/` bị loại (3.4 MB PDF, không dùng khi chạy). **Bắt buộc giữ:**
`ai_race/` (gồm `configs/` và `prompts/`) và `FAIRGAME/` — notebook assert cả ba.

Kiểm tra dataset sau khi upload khớp với repo local:

```bash
kaggle datasets files nguyenlamphuquy/ai-race-experiment -v | head
```

> Mỗi lần sửa prompt, config, hay engine là **phải** tạo dataset version mới. Dataset cũ =
> code cũ, và `source_sha256` trong manifest sẽ ghi lại đúng phiên bản cũ đó — sai lệch
> giữa cái bạn nghĩ mình chạy và cái thực sự chạy là loại lỗi khó phát hiện nhất.
> `source_sha256` hash mọi file `.py/.json/.txt` dưới `ai_race/` và `FAIRGAME/src/`, nên
> sửa `docs/` hay `kaggle/` không đổi nó.

### A3. Inputs cần add vào notebook

| Input | Bắt buộc | Mount path |
|---|---|---|
| Dataset `nguyenlamphuquy/ai-race-experiment` | có | `/kaggle/input/datasets/nguyenlamphuquy/ai-race-experiment` hoặc `/kaggle/input/ai-race-experiment` |
| Kaggle Model — `qwen-lm/qwen2.5/transformers/14b-instruct` | có | `/kaggle/input/models/qwen-lm/qwen2.5/transformers/14b-instruct/1` |
| Kaggle Model — `google/gemma-3/transformers/gemma-3-12b-it` | có | `/kaggle/input/models/google/gemma-3/transformers/gemma-3-12b-it/1` |
| Dataset wheelhouse vLLM | chỉ khi image chưa có vLLM | điền `VLLM_WHEELS_DIR`; phải có `manifest.json` liệt kê SHA-256 từng wheel |

Notebook pin `REPO_INPUT_DIRS` là **danh sách** cả hai path trên và lấy cái nào tồn
tại. Kaggle mount dataset lúc theo slug trần, lúc theo `datasets/<owner>/<slug>` tuỳ
notebook — pin đúng một chuỗi thì đổi notebook là hỏng. Nếu không path nào khớp, nó
raise kèm gợi ý nơi tìm thấy repo thật, thay vì âm thầm dùng input khác: chạy nhầm
source revision là lỗi mà manifest sẽ ghi lại sai. Đặt `REPO_INPUT_DIRS = None` để bỏ
chốt chặn và tự dò (tìm tới độ sâu 6, đủ cho cả hai layout).

Internet OFF. Auto-discovery wheel bị tắt cố ý — phải chỉ đúng path wheelhouse đã audit.

### A4. Chạy

Sửa cell cấu hình đầu file:

```python
MODELS = [
    {"path": ".../qwen-lm/qwen2.5/transformers/14b-instruct/1",
     "short_name": "qwen2.5-14b-instruct", "engine": "transformers"},
    {"path": ".../google/gemma-3/transformers/gemma-3-12b-it/1",
     "short_name": "gemma-3-12b-it", "engine": "transformers"},
]
EXPERIMENTS = ["baseline"]      # hoặc thêm các persona_baseline_*
REPETITIONS_OVERRIDE = 10       # pilot; None = dùng config (50)
RUN_PHASE_OVERRIDE = None       # "confirmatory" khi đã freeze
TEMPERATURE = 0.7               # KHÔNG dùng 0, xem mục "Bẫy" bên dưới
```

Accelerator: **RTX PRO 6000, 96 GB**. Tải `ai_race_results.zip` trong output về
`results/open_source/`.

#### Backend: transformers có sẵn trong image

Cấu hình mặc định dùng `engine="transformers"`, tức thư viện `transformers` đã có
trong image Kaggle. Không cài gì, không cần wheelhouse vLLM, Internet vẫn OFF. Cell
cài vLLM tự bỏ qua và in `No configured model requires vLLM.`; `run_manifest.json`
sẽ ghi `packageVersions.vllm = null`, đúng với thực tế.

VRAM không phải ràng buộc ở đây: Qwen2.5-14B bf16 ≈ 29,5 GB và Gemma-3-12B ≈ 24,4 GB,
đều thoải mái trên 96 GB, chạy tuần tự từng model một. Giữ `TENSOR_PARALLEL_SIZE = 1`
và không cần quantize.

**Cái phải đánh đổi là tốc độ.** Runner luôn cấp seed riêng cho từng quyết định — đó
là invariant của repo. Một forward pass gộp lại dùng chung một torch RNG nên không
thể tôn trọng seed riêng của từng `(rep, round, agent)`, vì vậy backend transformers
**rơi về sinh từng prompt một** khi có seeds. `BATCH_SIZE = 128` do đó không có tác
dụng, throughput là ~1 generation mỗi lượt.

Ước lượng thô cho `REPETITIONS_OVERRIDE = 10`: 3 treatment × 10 rep = 30 race, dài
trung bình 9 vòng, 2 người chơi ≈ **540 generation/model/experiment**. Bỏ override
(50 rep) thì thành ≈ 2.700. Nhân với thời gian sinh 256 token của model 12–14B để ra
wall-clock, rồi đối chiếu giới hạn phiên của Kaggle trước khi chạy bản đầy đủ.

Nếu cần nhanh hơn thì đổi `engine` của từng model về `"vllm"`; bốn hằng
`MAX_MODEL_LEN` / `GPU_MEMORY_UTILIZATION` / `TENSOR_PARALLEL_SIZE` / `ENFORCE_EAGER`
vẫn còn nguyên trong notebook và chỉ có tác dụng ở backend đó. Đổi backend làm đổi
`packageVersions` nên `protocol_signature` khác đi — run vLLM và run transformers
**không pool chung được** ở primary mode.

#### Gemma-3 12B là model multimodal

Đây là rủi ro cụ thể cần kiểm trước khi chạy dài. `_init_transformers_engine` nạp
bằng `AutoModelForCausalLM`. Với Gemma-3, chỉ bản 1B là text-only
(`Gemma3ForCausalLM`); các bản 4B/12B/27B dùng `Gemma3ForConditionalGeneration` và
config của chúng **không** nằm trong bảng ánh xạ của `AutoModelForCausalLM`. Khả năng
cao lệnh nạp raise `Unrecognized configuration class ... for AutoModelForCausalLM`.

Chạy thử nạp riêng Gemma-3 trước, hoặc chạy Qwen trước để có kết quả chắc chắn.
Notebook ghi lỗi của từng model vào manifest rồi đi tiếp, nhưng
`FAIL_ON_INCOMPLETE_RUN = True` sẽ báo hỏng ở cuối. Ba cách xử lý, theo thứ tự ưu
tiên: dùng `engine="vllm"` cho riêng Gemma-3 (vLLM xử lý được kiến trúc này); hoặc
đổi sang checkpoint text-only; hoặc sửa connector để lùi về
`AutoModelForImageTextToText` — nhưng `FAIRGAME/` là vendored, sửa nó là tạo nhánh
riêng phải tự bảo trì.

Gemma-3 cũng cần `transformers ≥ 4.50`; kiểm tra `packageVersions.transformers` trong
manifest sau lần chạy đầu.

### A5. Persona: chạy tất cả cell trong CÙNG một session

Đây là ràng buộc thật, không phải khuyến nghị. `protocol_signature` gồm source revision,
decoding, và package versions. Nếu bạn chạy `none` hôm nay và `S_AA` tuần sau với image
Kaggle khác, hai run có signature khác nhau, và persona trở thành **trùng khít với batch** —
không tách được. Analyser sẽ raise:

> `persona_condition varies across N cells (...) but never within a protocol signature, so
> persona is perfectly confounded with the run batch.`

Cách làm đúng: liệt kê hết vào một lần chạy.

```python
EXPERIMENTS = [
    "baseline",                      # none  — đối chứng trung tính
    "persona_baseline_neutral",      # R0    — placebo cùng độ dài
    "persona_baseline_risk_averse",  # R-
    "persona_baseline_risk_seeking", # R+
    "persona_baseline_coop_coop",    # S_CC
    "persona_baseline_adv_adv",      # S_AA
    "persona_baseline_adv_coop",     # S_AC  — cell bất đối xứng
    "persona_baseline_coop_adv",     # S_CA  — mirror của S_AC
]
```

Thêm `"baseline_swapped"` nếu muốn đo luôn hiệu ứng ghế (chỉ đảo thứ tự tên công ty,
persona rỗng — so với `baseline` sẽ ra artefact vị trí thuần túy).

---

## Đường B — Kaggle Benchmark cho model frontier

**Không cần dataset.** `kaggle/benchmarks/ai_race_baseline.py` reimplement toàn bộ cơ chế
trong một file, không import `ai_race`. Push thẳng file đó.

Đây là **thao tác checkpointed**: chạy một lệnh, xem output, dừng, rồi mới lệnh kế.
Không nối chuỗi.

```bash
kaggle b init -y
```
```bash
kaggle b t push ai-race-baseline -f kaggle/benchmarks/ai_race_baseline.py --wait
```
```bash
kaggle b auth -y          # ngay trước khi run: token ngắn hạn, hết hạn là 401
```
```bash
kaggle b t run ai-race-baseline -m <model-slug> --wait
```
```bash
kaggle b t download ai-race-baseline -m <model-slug> -o results/frontier
```

Model slug lấy từ `kaggle b t models`, đừng đoán.

**Cái giá của self-contained:** file này giữ một **bản sao byte-for-byte** của
`ai_race/prompts/ai_race_en.txt`. Sửa template mà quên đồng bộ bản sao thì hash lệch và
toàn bộ race bị loại khỏi primary analysis. `ai_race/tests/test_prompt_contract.py` so
trực tiếp hai bên nên `pytest` sẽ đỏ ngay — chạy test trước khi push.

Đường B không có persona (`PERSONA_CONDITION = "none"` cố định). Muốn persona trên model
frontier thì dùng đường C.

---

## Đường C — Local + Model Proxy

Chạy trên máy bạn, không GPU, gọi model hosted qua endpoint mà `kaggle_benchmarks` dùng.

```bash
kaggle benchmarks auth        # ghi MODEL_PROXY_URL / MODEL_PROXY_API_KEY vào .env
python3 -m ai_race.runner.run_experiment \
  ai_race/configs/experiment/api_baseline.json \
  --output results/frontier/api_baseline
```

`api_baseline.json` đặt `"backend": "proxy"` và `"samplingSeedApplied": false` — proxy
không xác nhận seed nên **không được coi là CRN cho sampling**. CRN của horizon và setback
vẫn giữ (chúng đến từ RNG local, không phụ thuộc provider).

Token hết hạn nhanh, chạy lại `kaggle benchmarks auth` trước mỗi phiên dài.

Đường này dùng đúng agents config như đường A, nên **có persona đầy đủ** — chỉ cần đổi
`"agents"` trong experiment config.

---

## Phân tích (luôn local)

### Bước 1 — kiểm tra symmetry collapse TRƯỚC khi scale

```bash
python3 results/scripts/check_symmetry.py --input results/open_source/<run>
```

Hai seat cùng model đọc prompt gần đối xứng. Nếu chúng chọn giống nhau mọi vòng thì
`ΔS ≡ 0` suốt race, trạng thái luôn là `tied`, và **toàn bộ trục race-position biến mất**.
Script exit 1 khi >40% race bị vậy. Gặp trường hợp đó thì tăng temperature, hoặc dùng cell
persona bất đối xứng (`S_AC`), chứ đừng scale lên 50 rep.

### Bước 2 — chạy analyser

Primary (chỉ dùng khi đã có manifest Kaggle đầy đủ, `run_phase = confirmatory`):

```bash
python3 results/scripts/analyze_ai_race.py \
  --input results/open_source --input results/frontier \
  --output results/derived/ai_race_analysis \
  --fit-logit
```

Audit trên pilot hoặc trên output mock local:

```bash
python3 results/scripts/analyze_ai_race.py \
  --input /tmp/smoke --output /tmp/derived --fit-logit \
  --allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs
```

Manifest của runner local là `ai-race-results-v1`, **cố ý không đủ** cho primary analysis —
nó thiếu source hash, decoding contract, seed provenance. Đó là lý do smoke test luôn cần
ba flag audit. Đừng dùng ba flag đó cho kết quả thật.

### Bước 3 — đọc output

28 file trong thư mục output. Đọc theo thứ tự này:

1. `analysis_manifest.json` — bao nhiêu race bị loại và vì sao; `persona_identification`
2. `parse_failures.csv` — khác 0 thì **đừng diễn giải hành vi gì cả**
3. `race_quality.csv`, `horizon_distribution.csv` — cơ chế có chạy đúng không (mean ≈ 9)
4. `gap_collinearity.csv` — phải thấy `pearson_r = 1.0`, xác nhận ΔS = 0.5 × hiệu Unsafe
5. `unsafe_by_risk_model_player.csv`, `treatment_contrasts.csv` — Fig 2A
6. `unsafe_by_lag_profile_turn.csv`, `unsafe_by_gap_lag_turn.csv` — Fig 2B
7. `winner_loser_correlation.csv` — Fig 2C
8. `clustered_logit_coefficients.csv` — Table 1, cột `specification` 1–6
9. `human_comparison.csv` — bảng 8 dòng E1–E8, verdict replicated / not / inconclusive

---

## Bẫy đã gặp

| Bẫy | Hậu quả | Cách tránh |
|---|---|---|
| `temperature = 0` cho arm hành vi | hai seat chọn giống hệt nhau, trục race-position biến mất | dùng 0.7; temp 0 chỉ để đo determinism, chạy riêng |
| Persona cell chạy lệch batch | persona trùng khít với batch, hệ số không ước lượng được | chạy hết trong một session |
| Đổi `seed` giữa các experiment config | mất CRN, mất matched-pairs qua persona | giữ `260726` ở **mọi** config |
| Sửa prompt mà quên bump `promptVersion` | analyser từ chối toàn bộ run | `pytest` bắt được ngay |
| Sửa prompt mà quên đồng bộ bản sao trong benchmark | output đường B bị loại khỏi primary | `pytest` bắt được ngay |
| Dùng dataset version cũ | chạy code cũ mà tưởng code mới | tạo version mới sau mỗi thay đổi |
| Diễn giải kết quả khi `parse_failures > 0` | một parse failure làm hỏng **cả race** (Safe fallback lan sang vòng sau) | sửa parsing trước, đừng nới lỏng `parse_action` |
| Pool pilot với confirmatory | mất tính preregistered | `run_phase` gate; đừng dùng `--allow-nonconfirmatory-runs` cho kết quả thật |

---

## Thứ tự chạy đề xuất

1. Local: `pytest` + mock smoke.
2. Stage dataset (đường A) hoặc push task (đường B).
3. **Pilot 10 rep, một model, 2 điều kiện.**
4. `check_symmetry.py` → không đạt thì dừng, chỉnh, chạy lại pilot.
5. Analyser với flag audit; xem parse health và cơ chế.
6. Freeze: prompt, config, persona text, `human_reference.json`, số rep. Đổi `runPhase` →
   `confirmatory`.
7. Full grid, tất cả persona cell trong một session.
8. Analyser **một lần**, không đổi định nghĩa outcome sau khi đã thấy kết quả.
9. Điền `paper/` và `slides/`.
