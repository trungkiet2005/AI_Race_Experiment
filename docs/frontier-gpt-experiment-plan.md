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

## Phạm vi cấu hình game

`ai_race/configs/experiment/` hiện có **46 tổ hợp** treatment × persona (3 mức risk
0,1/0,6/0,9 là cố định trong mọi config; cái thay đổi là `agents`):

| Tầng | Số config | Đã từng chạy qua đường hosted/API nào chưa? |
|---|---|---|
| Baseline neutral (`agents: "companies_default"`, persona="none") | 1 | Có — `api_baseline_crossmodel.json` (qua proxy) |
| Seat-swap check (`baseline_swapped`) | 1 | Chưa — chỉ có bản offline |
| 8 persona core (R0, R-, R+, S_CC, S_AA, S_AC, S_CA + neutral kể trên) | 7 file `api_persona_baseline_*.json` | Có — qua proxy |
| Ma trận persona rủi ro 6×6 (`persona_baseline_risk_1_1` … `risk_6_6`) | 36 | **Chưa bao giờ**, kể cả qua proxy — chỉ tồn tại bản offline (`LocalQwen`) |

[docs/running-proxy-pilots.md](running-proxy-pilots.md) là tiền lệ đã thiết lập cho
các lần chạy hosted trước đây (Gemini qua proxy): phạm vi thực tế ở đó là **baseline
+ 8 persona core**, chạy bằng một script `JOBS` tuần tự, tự bỏ qua config đã
`completed`. Ma trận 6×6 chưa từng được đưa vào bất kỳ lần chạy hosted nào.

**Plan này mặc định theo đúng phạm vi tiền lệ đó — baseline + 7 persona core (8
config) — và để ngoài phạm vi:**

- Ma trận rủi ro 6×6 (36 config): giữ nguyên là local-only như hiện trạng, trừ khi
  bạn nói rõ muốn mở rộng đường API cho phần này (khối lượng request gấp ~5 lần).
- `baseline_swapped`: dễ thêm (chỉ đổi tên công ty), nhưng cần bạn xác nhận có muốn
  đo seat-position artefact cho GPT hay không — không tự động thêm vào scope.

Nếu bạn muốn phạm vi khác (chỉ baseline, hoặc mở rộng cả ma trận 6×6), báo lại và
tôi sửa phần "full grid" bên dưới.

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

### 2. Tạo 8 experiment config (baseline + 7 persona core)

Nhân bản từng file `api_baseline_crossmodel.json` và 7 file
`api_persona_baseline_*.json` thành 8 file `openai_*.json` tương ứng, chỉ đổi
`backend` từ `"proxy"` sang `"openai"`, đổi `proxyOptions` thành chỉ còn
`temperature`, và để `"models"` là danh sách model id GPT thật (điền ở bước 3).
`"agents"` giữ nguyên như bản gốc — đó chính là cái phân biệt 8 config này:

| File mới | `agents` (copy từ bản proxy tương ứng) | Persona |
|---|---|---|
| `openai_baseline.json` | `companies_default` | none |
| `openai_persona_baseline_neutral.json` | `persona_neutral` | R0 |
| `openai_persona_baseline_risk_averse.json` | `persona_risk_averse` | R- |
| `openai_persona_baseline_risk_seeking.json` | `persona_risk_seeking` | R+ |
| `openai_persona_baseline_coop_coop.json` | `persona_coop_coop` | S_CC |
| `openai_persona_baseline_adv_adv.json` | `persona_adv_adv` | S_AA |
| `openai_persona_baseline_adv_coop.json` | `persona_adv_coop` | S_AC |
| `openai_persona_baseline_coop_adv.json` | `persona_coop_adv` | S_CA |

Mẫu cho `openai_baseline.json`:

```json
{
  "name": "openai_baseline",
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
horizon/setback khớp với các run model khác, cho phép so sánh across-model (RQ6)
**và** giữa các persona cell (matched-pairs). **Persona đầy đủ phải chạy trong
cùng một session/batch** — xem cảnh báo `protocol_signature` ở
[running-the-experiment.md](running-the-experiment.md#a5-persona-chạy-tất-cả-cell-trong-cùng-một-session):
chạy `openai_baseline` hôm nay và `openai_persona_baseline_adv_adv` tuần sau (khác
package version SDK OpenAI chẳng hạn) sẽ làm persona trùng khít với run batch,
analyser sẽ từ chối join. Dùng script `JOBS` tuần tự ở mục kế tiếp để đảm bảo cả 9
config chạy liền nhau.

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
   3 treatment × 8 config.
3. Chạy pilot cho cả 8 config bằng script `JOBS` tuần tự, theo đúng mẫu ở
   [running-proxy-pilots.md](running-proxy-pilots.md#chạy-nhiều-config-nối-tiếp--sửa-jobs-rồi-chạy) —
   khác biệt duy nhất so với mẫu proxy là **không cần** dòng `kaggle benchmarks auth`
   (backend `openai` không qua Kaggle proxy, không có token hết hạn):
   ```python
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
   ```
   Script tự bỏ qua config đã `completed` — chạy lại an toàn nếu bị đứt giữa chừng
   (hết quota, mất mạng), không tốn tiền chạy lại phần đã xong.
4. `python3 results/scripts/check_symmetry.py --input results/frontier/openai/baseline`
   (và các thư mục persona) — dừng và chỉnh nếu >40% race bị symmetry collapse
   (đúng bẫy nhiệt độ ở mục 1).
5. Audit bằng analyser với các flag `--allow-*` (pilot, không phải kết quả cuối) —
   xem đúng bộ flag ở [running-proxy-pilots.md](running-proxy-pilots.md#sau-khi-chạy-phân-tích):
   ```bash
   python3 results/scripts/analyze_ai_race.py \
     --input results/frontier/openai --output /tmp/derived_openai --fit-logit \
     --allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs \
     --allow-missing-persona-condition
   ```
   Đọc `parse_failures.csv` trước tiên — khác 0 thì dừng, đừng diễn giải hành vi.
6. Freeze: model list, prompt/config version, số rep, rồi đổi `runPhase` →
   `"confirmatory"` trong cả 8 config.
7. Chạy full grid (tất cả model GPT đã chọn × 8 config, ba treatment, số rep đã
   freeze) bằng cùng script `JOBS`, chỉ đổi `repetitions` trong config và thư mục
   output.
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
- Có mở rộng thêm `baseline_swapped` (seat-position check) vào scope không? Mặc định
  plan này **không** bao gồm.
- Có mở rộng đường API cho ma trận persona rủi ro 6×6 (36 config, hiện chỉ chạy
  local) không? Mặc định plan này **không** bao gồm — khối lượng request sẽ tăng
  ~5 lần so với 8 config baseline+persona core.
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

1. Vá `factory.py` (mục "Việc cần làm ngay" #1) + tạo 8 config `openai_*.json`
   (#2) — làm ngay, không cần key.
2. Chờ API key + danh sách model GPT cụ thể từ bạn (và xác nhận/điều chỉnh phạm vi
   ở mục "Phạm vi cấu hình game" nếu muốn khác 8 config mặc định).
3. Điền key vào `.env`, điền model id vào cả 8 config.
4. Smoke test 1 model/1 treatment/1 rep trên `openai_baseline.json`.
5. Pilot 10 rep × 3 treatment × 8 config, chạy bằng script `JOBS` tuần tự.
6. `check_symmetry.py` cho từng thư mục output → không đạt thì dừng, chỉnh
   temperature/prompt, chạy lại pilot.
7. Analyser với flag audit; đọc theo đúng thứ tự ở
   [running-the-experiment.md](running-the-experiment.md#bước-3--đọc-output).
8. Freeze cả 8 config, đổi `runPhase` → `confirmatory`, chạy lại full grid bằng
   `JOBS`.
9. Analyser một lần, cập nhật `paper/` và `slides/`.
