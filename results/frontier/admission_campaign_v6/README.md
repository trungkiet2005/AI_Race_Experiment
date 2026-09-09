# Admission campaign `ai-race-frontier-admission-v6` (ingested 2026-09-09)

Cross-model endpoint-admission audit run before any frontier gameplay. Nine routes completed,
one earlier attempt failed and is kept as a failure record.

## Protocol

| Mục | Giá trị |
|---|---|
| `protocol_id` | `ai-race-frontier-admission-v6` |
| Task | `ai-race-frontier-admission`, Kaggle Benchmarks |
| Kaggle identity | `daosyduyminh` (đây là identity duy nhất được cấu hình cho campaign này) |
| Probe bank | 20 probe đóng băng, sáu domain: `rule_recall` (4 probe), `stage_payoff` (2), `state_reconstruction` (5), `state_transition` (2), `terminal_scoring` (5), `expected_payoff` (2) |
| Lặp lại | 3 lần, đúng như yêu cầu |
| Dòng giữ lại | 60 mỗi route (20 probe × 3 lần) |
| Decoding | temperature 0, giới hạn 256 token đầu ra, `prompt_version = ai-race-fairgame-v3` |
| Base seed | 260726; seed mỗi probe = `260726 + 1000 × rep + index` |
| `probe_bank_sha256` | `2953fb472dcba47511de108a47d4126b0f661e79bf60c96a3ab46f581abca8bc` (giống nhau trên cả 9 route) |
| `rules_context_sha256` | `b80981a5e888d04649300f79b7388ca7a755b958f239f0172efd9ca35aab2ae0` (giống nhau trên cả 9 route) |

Điều kiện admit — phải đạt **cả ba**:

- overall accuracy ≥ 0,80
- `state_reconstruction` accuracy ≥ 0,75
- `terminal_scoring` accuracy ≥ 0,75

`expected_payoff` cũng có ngưỡng 0,75 ghi trong artefact, nhưng nó **chỉ mang tính chẩn đoán**
(`expected_payoff_is_diagnostic_only = true`) và không bao giờ chặn admission. Một route được
gắn `paper-ready` khi vừa được admit vừa có ít nhất 3 lần lặp.

## Vì sao có hai task version

Task được đóng gói lại giữa hai lần chạy, nên artefact nằm dưới hai version:

- **version 7** — tám route. Ở version này tham số reasoning budget được gửi là
  `reasoning = "none"` cho mọi route.
- **version 8** — một route, `google/gemini-3.5-flash-lite`. Route này ở version 7 **thất bại**
  (xem `failed_runs/` bên dưới) vì nó từ chối chính *tham số* reasoning, không phải giá trị của
  nó: chỉ cần gọi tên `reasoning` là provider trả HTTP 400 "Request contains an invalid argument"
  trước khi có probe nào được lấy mẫu. Theo mục sửa đổi **2026-09-09 "route-resolved reasoning
  budget"** trong [docs/reviewer-revision-frontier-protocol.md](../../../docs/reviewer-revision-frontier-protocol.md),
  task từ version 8 trở đi **bỏ hẳn tham số** cho những route từ chối nó, và ghi lại việc bỏ đó
  trong manifest (`reasoning_requested = null`). Không có gì khác thay đổi: cơ chế, canonical
  prompt, probe bank, parser, seed stream, temperature và token cap đều nguyên vẹn.

Hệ quả cần nói rõ khi gộp bảng: tám route có `reasoning_requested = "none"` gửi request
**byte-identical** với các lần chạy trước, nên artefact của chúng vẫn poolable với task version
cũ. Route `google/gemini-3.5-flash-lite` có contract **khác** — tham số bị bỏ — và điều đó nhìn
thấy được trong artefact chứ không phải suy ra. Bảng nào đặt route này cạnh tám route còn lại
phải nói ra điều đó.

## Route list và kết quả

Sắp theo overall accuracy giảm dần. Xem `derived/` để có số theo từng domain.

| Route | Task ver | Run | Overall accuracy | Admitted | Evidence class |
|---|---|---|---|---|---|
| `google/gemini-3-flash-preview` | 7 | 1463707 | 0,9333 | có | `paper-ready` |
| `anthropic/claude-opus-5@default` | 7 | 1494033 | 0,9167 | có | `paper-ready` |
| `openai/gpt-5.4-2026-03-05` | 7 | 1494043 | 0,9000 | có | `paper-ready` |
| `openai/gpt-5.5-2026-04-23` | 7 | 1494045 | 0,9000 | có | `paper-ready` |
| `anthropic/claude-sonnet-5@default` | 7 | 1494039 | 0,8500 | có | `paper-ready` |
| `google/gemini-3.1-flash-lite-preview` | 7 | 1494040 | 0,8000 | **không** | `diagnostic` |
| `openai/gpt-5.4-mini-2026-03-17` | 7 | 1494044 | 0,7500 | **không** | `diagnostic` |
| `google/gemini-3.5-flash-lite` | 8 | 1494138 | 0,6500 | **không** | `diagnostic` |
| `openai/gpt-5.4-nano-2026-03-17` | 7 | 1494042 | 0,5167 | **không** | `diagnostic` |

**Được admit (5):** `google/gemini-3-flash-preview`, `anthropic/claude-opus-5@default`,
`openai/gpt-5.4-2026-03-05`, `openai/gpt-5.5-2026-04-23`, `anthropic/claude-sonnet-5@default`.

**Bị từ chối (4):** `google/gemini-3.1-flash-lite-preview` (overall vừa đủ 0,8000 nhưng
`state_reconstruction` = 0,7333 < 0,75), `openai/gpt-5.4-mini-2026-03-17`,
`google/gemini-3.5-flash-lite`, `openai/gpt-5.4-nano-2026-03-17`.

## Layout

Giống hệt convention của `admission_campaign/` và `admission_campaign_v5/` — mirror nguyên
cây tải về từ Kaggle:

```
ai-race-frontier-admission/<taskVersion>/<model-dir>/<runId>/
    ai-race-frontier-admission-run_id_Run_1_<route>.run.json
    ai-race-frontier-admission.task.json
    results/ai_race_frontier_admission/<model-tag>/
        admission.json
        raw_responses.jsonl
        run_manifest.json
```

`raw_responses.jsonl` và `run_manifest.json` được copy **byte-for-byte** (đã kiểm chứng bằng
SHA-256 hai đầu); không có file nào bị sửa hay chuẩn hoá lại.

## `failed_runs/` — attempt được giữ lại

```
failed_runs/ai-race-frontier-admission/7/gemini-3.5-flash-lite/1494041/
```

`run_manifest.json` ở đó ghi `status = "failed"`,
`status_reason = "RuntimeError: bounded transport retries exhausted"`, và **không có**
`admission.json` hay `raw_responses.jsonl` vì không probe nào được lấy mẫu. Nó được giữ lại
theo chính sách của repo: một attempt thất bại hoặc bị thay thế vẫn nằm trong chuỗi kế toán,
không bị xoá và không bị thay thế bằng một route khác. Lần chạy lại thành công là run 1494138
ở task version 8.

Run này nằm ngoài mọi bảng dẫn xuất. Script phân tích chỉ đi qua
`ai-race-frontier-admission/` và không bao giờ đi vào `failed_runs/`.

## Bảng dẫn xuất

Sinh bởi [scripts/analyze_frontier_admission_campaign.py](../../../scripts/analyze_frontier_admission_campaign.py):

```bash
python scripts/analyze_frontier_admission_campaign.py
```

- `derived/admission_campaign_v6.csv` — một dòng mỗi route
- `derived/admission_campaign_v6.json` — cùng nội dung, kèm provenance (SHA-256 của từng
  `admission.json` và `raw_responses.jsonl` đã đọc)
- `derived/report.md` — báo cáo ngắn

Script fail-closed: nó từ chối xuất một dòng khi `n_rows` khác 60, số dòng raw khác `n_rows`,
số theo domain không cộng đúng, overall accuracy không recompute được, cờ admit không recompute
được từ ba ngưỡng, `protocol_id` không giống nhau, hoặc `probe_bank_sha256` /
`rules_context_sha256` lệch giữa các route. Lệch hash nghĩa là route đó đang đo một probe bank
khác và phải bị báo là non-comparable thay vì đặt cạnh các route còn lại.
