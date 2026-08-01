# Plan: chạy AI Race trên các frontier model GPT (OpenAI API trực tiếp)

## Bối cảnh — vì sao không dùng đường có sẵn

Repo đã có ba đường chạy model (xem [running-the-experiment.md](running-the-experiment.md)).
Đường C (`api_baseline_crossmodel.json`, `backend: "proxy"`) **đã** chạy được
`openai/gpt-5.4-nano-2026-03-17` qua Kaggle Model Proxy — nhưng đó là một model GPT
duy nhất, chọn theo route nào proxy tình cờ expose, không phải danh sách frontier GPT
do ta chủ động chọn. Yêu cầu lần này là chạy **nhiều** frontier model GPT bằng **API
key OpenAI riêng**, nên cần một đường khác: gọi thẳng OpenAI API, không qua proxy.

Đường đó đã có khung sẵn nhưng **chưa từng được dùng thật**:

- [ai_race/models/factory.py](../ai_race/models/factory.py) có `backend="api"` →
  gọi `FAIRGAME.src.llm_connectors.llm_factory_connector.ChatModelFactory`.
- [FAIRGAME/src/llm_connectors/openai_connector.py](../FAIRGAME/src/llm_connectors/openai_connector.py)
  đọc `API_KEY_OPENAI` từ env, nhận **bất kỳ chuỗi `provider_model` nào** — bản thân
  connector không hardcode model.
- Nhưng `MODEL_PROVIDER_MAP` trong
  [llm_factory_connector.py](../FAIRGAME/src/llm_connectors/llm_factory_connector.py)
  chỉ có một entry: `"OpenAIGPT4o" -> "gpt-4o"`. Không config nào trong
  `ai_race/configs/experiment/` dùng `"backend": "api"` — **grep xác nhận 0 kết quả**.
  Đây là đường chưa được thực chiến, không phải đường đã kiểm chứng.

Vậy có hai việc: (1) vá một khoảng hở nhỏ trong `factory.py` để không phải sửa file
vendor cho mỗi model GPT mới, và (2) thực sự chạy pilot → confirmatory theo đúng
trình tự ở [PROJECT.md](../PROJECT.md).

## Việc cần làm ngay — không cần API key

### 1. Thêm nhánh backend nhận model id trực tiếp

Không sửa `MODEL_PROVIDER_MAP` (file vendor, mỗi model mới lại phải thêm một dòng ở
đó). Thay vào đó, thêm một nhánh trong `get_send_batch()`
([factory.py](../ai_race/models/factory.py)) gọi thẳng `OpenAIConnector` với
`model_name` nguyên văn — connector đã hỗ trợ sẵn, chỉ là chưa có đường gọi tới nó
ngoài `MODEL_PROVIDER_MAP`:

```python
if backend == "openai":
    from FAIRGAME.src.llm_connectors.openai_connector import OpenAIConnector

    options = dict(proxy_options or {})
    model = OpenAIConnector(model_name, temperature=float(options.get("temperature", 0.7)))

    def send_openai(prompts, seeds=None):
        del seeds  # OpenAIConnector không nhận seed — ghi rõ trong manifest, đừng ngụy trang thành CRN
        return [model.send_prompt(p) for p in prompts]

    return send_openai
```

Nhờ vậy, `"models"` trong experiment config có thể là chuỗi model OpenAI thật
(`"gpt-4o"`, `"gpt-4.1"`, `"gpt-5"`, …) giống hệt cách đường proxy dùng slug thật,
thay vì phải đăng ký từng abstract name trong file vendor.

- **Đổi mặc định temperature:** `OpenAIConnector.__init__` mặc định `temperature=1.0`.
  Bẫy đã ghi trong [running-the-experiment.md](running-the-experiment.md#bẫy-đã-gặp):
  temperature thấp/0 làm hai seat hội tụ giống hệt nhau, trục race-position biến
  mất — nhưng 1.0 lại chưa từng được kiểm chứng cho game này. Truyền tường minh
  `0.7` qua option, đừng dựa vào default của connector.
- **Không có concurrency.** Vòng lặp hiện tại gọi API tuần tự từng prompt
  (`[model.send_prompt(p) for p in prompts]`). Với lockstep batching (nhiều race
  cùng tiến một vòng), điều này sẽ chậm và có thể chạm rate limit sớm hơn cần thiết.
  Cân nhắc thêm `concurrency` bằng `ThreadPoolExecutor` giống tinh thần
  `proxyOptions.concurrency` của đường proxy — nhưng **chỉ làm nếu pilot chứng minh
  cần**, đừng tối ưu sớm.
- **Không có retry/timeout theo transport.** Đường proxy có
  `max_transport_retries`, `timeout`. Đường OpenAI SDK trực tiếp hiện không có gì
  tương đương — SDK OpenAI có retry mặc định riêng, nhưng nên xác nhận hành vi đó
  thay vì giả định.
- **Không sửa gì trong `FAIRGAME/`** — đúng nguyên tắc "treat FAIRGAME as a
  dependency" trong [CLAUDE.md](../CLAUDE.md).

### 2. Tạo experiment config mới

Nhân bản [api_baseline_crossmodel.json](../ai_race/configs/experiment/api_baseline_crossmodel.json)
thành `ai_race/configs/experiment/openai_frontier_baseline.json`:

```json
{
  "name": "openai_frontier_baseline",
  "runPhase": "pilot",
  "description": "Frontier GPT models via direct OpenAI API (own key), neutral baseline, persona=none.",
  "games": ["ai_race_risk_10", "ai_race_risk_60", "ai_race_risk_90"],
  "agents": "companies_default",
  "languages": ["en"],
  "models": ["<điền model id thật ở đây>"],
  "useOffline": false,
  "backend": "openai",
  "samplingSeedApplied": false,
  "proxyOptions": {"temperature": 0.7},
  "repetitions": 10,
  "seed": 260726,
  "maxParseRetries": 3,
  "verbose": true
}
```

`seed` giữ nguyên `260726` như mọi config khác — đây là điều kiện để CRN của
horizon/setback khớp với các run model khác, cho phép so sánh across-model (RQ6).

### 3. Danh sách model GPT cụ thể — **cần bạn xác nhận, không đoán**

Tôi sẽ không tự điền model id vào config vì đoán sai tên model là loại lỗi im lặng
nguy hiểm nhất ở đây (chi phí là tiền thật, không phải Kaggle credit). Trước khi
chạy, cần bạn liệt kê chính xác các model id sẽ dùng — kiểm bằng
`curl https://api.openai.com/v1/models -H "Authorization: Bearer $API_KEY_OPENAI"`
sau khi có key, đối chiếu với tài liệu OpenAI hiện hành.

### 4. Cài đặt local

```bash
pip install -e ".[api,dev]"   # openai SDK + python-dotenv + pytest
pytest                        # đảm bảo chưa gãy gì trước khi đụng code
```

## Việc cần làm khi có API key

1. Thêm vào `.env` (đã có trong `.gitignore`, không commit):
   ```
   API_KEY_OPENAI=sk-...
   ```
2. **Smoke test rẻ trước khi tốn tiền cho pilot thật:** 1 model, 1 risk treatment,
   `repetitions: 1`, xem log thô — xác nhận connectivity, format response, và
   `ACTION: SAFE|UNSAFE` có parse được không trước khi commit đến pilot 10 rep ×
   3 treatment.
3. Chạy pilot đầy đủ:
   ```bash
   python3 -m ai_race.runner.run_experiment \
     ai_race/configs/experiment/openai_frontier_baseline.json \
     --output results/frontier/openai_pilot
   ```
4. `python3 results/scripts/check_symmetry.py --input results/frontier/openai_pilot`
   — dừng và chỉnh nếu >40% race bị symmetry collapse (đúng bẫy nhiệt độ ở mục 1).
5. Audit bằng analyser với ba flag `--allow-*` (pilot, không phải kết quả cuối):
   ```bash
   python3 results/scripts/analyze_ai_race.py \
     --input results/frontier/openai_pilot --output /tmp/derived_openai --fit-logit \
     --allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs
   ```
   Đọc `parse_failures.csv` trước tiên — khác 0 thì dừng, đừng diễn giải hành vi.
6. Freeze: model list, prompt/config version, số rep, rồi đổi `runPhase` →
   `"confirmatory"` trong config.
7. Chạy full grid (tất cả model GPT đã chọn, ba treatment, số rep đã freeze) —
   mỗi model một thư mục con dưới `results/frontier/`.
8. Chạy analyser **một lần**, không đổi định nghĩa outcome sau khi đã thấy kết quả:
   ```bash
   python3 results/scripts/analyze_ai_race.py \
     --input results/open_source --input results/frontier \
     --output results/derived/ai_race_analysis --fit-logit
   ```

## Câu hỏi cần bạn quyết định trước khi chạy thật

- Danh sách model GPT cụ thể (xem mục 3 ở trên).
- Số rep cho pilot (mặc định đề xuất 10, theo đúng trình tự PROJECT.md) và cho full
  grid (baseline hiện dùng 50 cho open-source; với API trả phí, có thể muốn ít hơn).
- Có chạy các persona cell (`S_AA`, `S_AC`, …) cho GPT hay chỉ baseline neutral?
  Ảnh hưởng trực tiếp số lượng request × tiền.
- Có cần giới hạn ngân sách/rate limit cứng (ví dụ dừng nếu vượt N request) trước
  khi chạy full grid không, vì đây là API trả phí thật chứ không phải Kaggle quota.

## Rủi ro riêng của đường này

| Rủi ro | Hậu quả | Giảm thiểu |
|---|---|---|
| Model id sai/hết hạn | Toàn bộ run lỗi ngay từ prompt đầu | Xác nhận qua `/v1/models` trước khi điền config |
| `temperature` mặc định của `OpenAIConnector` là 1.0, chưa test cho game này | Có thể symmetry collapse khác hướng so với 0.7 đã dùng ở đường proxy | Luôn truyền tường minh, chạy `check_symmetry.py` trước khi scale |
| Gọi API tuần tự, không concurrency | Pilot chậm, có thể chạm rate limit giữa chừng | Thêm `ThreadPoolExecutor` nếu pilot cho thấy cần, đo trước khi tối ưu |
| Không có seed cho sampling | Không được coi là CRN cho sampling (giống đường proxy) | Giữ `"samplingSeedApplied": false`; CRN của horizon/setback vẫn nguyên vì đến từ RNG local |
| Chi phí thật, không phải Kaggle credit | Full grid chạy nhầm số rep/model có thể tốn ngoài dự kiến | Freeze sau pilot, không chạy full grid trước khi có xác nhận rõ ràng |
| Trộn pilot với confirmatory | Mất tính preregistered | `run_phase` gate như mọi đường khác; không dùng `--allow-nonconfirmatory-runs` cho kết quả thật |

## Thứ tự thực hiện

1. Vá `factory.py` (mục "Việc cần làm ngay" #1) + tạo config mới (#2) — làm ngay,
   không cần key.
2. Chờ API key + danh sách model GPT cụ thể từ bạn.
3. Điền key vào `.env`, điền model id vào config.
4. Smoke test 1 model/1 treatment/1 rep.
5. Pilot 10 rep, 3 treatment, tất cả model đã chọn.
6. `check_symmetry.py` → không đạt thì dừng, chỉnh temperature/prompt, chạy lại pilot.
7. Analyser với flag audit; đọc theo đúng thứ tự ở
   [running-the-experiment.md](running-the-experiment.md#bước-3--đọc-output).
8. Freeze, đổi `runPhase` → `confirmatory`, chạy full grid.
9. Analyser một lần, cập nhật `paper/` và `slides/`.
