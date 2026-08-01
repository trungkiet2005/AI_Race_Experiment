# Index kết quả `results/frontier/` (Kaggle Model Proxy, đường C)

Tất cả run ở đây chạy bằng:

```bash
kaggle benchmarks auth -y
python -m ai_race.runner.run_experiment ai_race/configs/experiment/<config>.json --output <output-thư-mục-dưới-đây>
```

Xem [docs/running-proxy-pilots.md](../../docs/running-proxy-pilots.md) để biết cách chạy 1 hoặc
nhiều config nối tiếp, và cách tạo config mới. Toàn bộ đều `runPhase: "pilot"`, **không phải
confirmatory** — không dùng để công bố kết luận cuối.

## Baseline — không persona, so sánh model (`baseline/`)

| Thư mục | Config | Model | Rep |
|---|---|---|---|
| `baseline/google-gemini-3-flash-preview/` | `api_baseline_crossmodel.json` | `google/gemini-3-flash-preview` | 10 |
| `baseline/google-gemini-3.1-flash-lite-preview/` | `api_baseline_crossmodel.json` | `google/gemini-3.1-flash-lite-preview` | 10 |
| `baseline/google-gemini-3.5-flash-lite/` | `api_baseline_flashlite35.json` | `google/gemini-3.5-flash-lite` | 10 |

Model đã thử nhưng **không dùng được** (không có thư mục ở đây): `deepseek-ai/deepseek-r1-0528`,
`openai/gpt-oss-120b`, `openai/gpt-5.4-nano-2026-03-17`, `qwen/qwen3-next-80b-a3b-instruct` (429
kéo dài); `google/gemini-2.5-flash`, `google/gemini-2.5-pro`, `google/gemini-3.1-pro-preview`,
`google/gemini-3.5-flash`, `google/gemini-3.6-flash`, `qwen/qwen3-next-80b-a3b-thinking`,
`qwen/qwen3-235b-a22b-instruct-2507`, `qwen/qwen3-coder-480b-a35b-instruct` (503, route không
khả dụng trên proxy). `anthropic/claude-sonnet-5`, `ibm/granite-4.0-h-small` chưa thử lại.

## Persona — 7 điều kiện gốc, tất cả trên `google/gemini-3-flash-preview`, 10 rep (`persona/`)

| Thư mục | Config | `personaCondition` | Ý nghĩa |
|---|---|---|---|
| `persona/S_AA_adv_adv/` | `api_persona_baseline_adv_adv.json` | S_AA | Cả 2 bên hung hăng |
| `persona/S_AC_adv_coop/` | `api_persona_baseline_adv_coop.json` | S_AC | Seat 1 hung hăng, seat 2 hợp tác |
| `persona/S_CA_coop_adv/` | `api_persona_baseline_coop_adv.json` | S_CA | Mirror của S_AC (đảo ghế) |
| `persona/S_CC_coop_coop/` | `api_persona_baseline_coop_coop.json` | S_CC | Cả 2 bên hợp tác |
| `persona/R0_neutral/` | `api_persona_baseline_neutral.json` | R0 | Placebo — có câu mô tả nhưng trung tính |
| `persona/Rminus_risk_averse/` | `api_persona_baseline_risk_averse.json` | R- | Cả 2 bên né rủi ro |
| `persona/Rplus_risk_seeking/` | `api_persona_baseline_risk_seeking.json` | R+ | Cả 2 bên thích rủi ro |

Điều kiện "không persona" (`agents: companies_default`) nằm trong `baseline/`, không lặp lại ở đây.

## Risk-preference matrix — 6×6 = 36 ô, đang chạy (`persona/R{i}_R{j}_risk_matrix/`)

Tái hiện thang đo Eckel-Grossman thật của paper gốc (SI Task 1): mỗi seat được gán 1 mức
risk-preference từ 1 (chắc chắn/không biến động) đến 6 (biến động nhất). `i` = mức của seat 1,
`j` = mức của seat 2 — bao gồm cả ô đối xứng (`i == j`) lẫn bất đối xứng.

- Config: `ai_race/configs/experiment/api_persona_baseline_risk_{i}_{j}.json` (36 file)
- Agents: `ai_race/configs/agents/persona_risk_{i}_{j}.json`
- Model: `google/gemini-3-flash-preview`, 10 rep mỗi ô
- Output: `persona/R{i}_R{j}_risk_matrix/google-gemini-3-flash-preview/`
- **Đang chạy tuần tự** qua script `run_risk_matrix.py` (không có trong repo, chỉ ở máy chạy —
  xem mẫu script trong `docs/running-proxy-pilots.md`). Kiểm tra `run_manifest.json` trong từng
  thư mục ô để biết ô nào `completed`/`running`/chưa chạy.

## Muốn mở rộng thêm

1. Tạo config mới trong `ai_race/configs/experiment/` (copy 1 file `api_persona_baseline_*.json`
   làm mẫu, đổi `"agents"` và/hoặc `"models"`).
2. Chạy theo lệnh ở đầu file này, đặt `--output` là một thư mục con mới, **rõ tên** dưới
   `baseline/` (nếu không persona) hoặc `persona/` (nếu có persona).
3. Thêm 1 dòng vào bảng tương ứng ở trên.
4. Giữ nguyên `"seed": 260726` ở mọi config để còn so sánh matched-pairs được.
