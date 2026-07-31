# Trạng thái triển khai plan phân tích hành vi LLM

Đối chiếu với [llm-behavior-analysis-plan.md](llm-behavior-analysis-plan.md).
Test: `python3 -m pytest` → **89 passed**.

## Đã xong

| Task | Nội dung | File |
|---|---|---|
| T0.1 | Prompt canonical đổi sang `ai-race-fairgame-v3`; hằng số scalar thay bằng bảng `{template: sha}` cho cả `en` và `vi`; thêm `_is_canonical_prompt()` | `results/scripts/analyze_ai_race.py`, `CLAUDE.md`, `results/README.md`, `ai_race/engine/state.py`, `kaggle/experiments/baseline.py` |
| T0.2 | Bỏ key chết `persona_block` (template mới dùng `{intro}`/`{personality}`) | `ai_race/engine/prompt.py` |
| T1.1 | `repetitions` 3 → 50 (≈2.700 player-round, xấp xỉ N của paper) | `configs/experiment/baseline.json` |
| T1.2 | Cấu hình đảo ghế để đo seat artefact | `configs/agents/companies_swapped.json`, `configs/experiment/baseline_swapped.json` |
| T1.3 | Script phát hiện symmetry collapse, exit 1 khi vượt ngưỡng | `results/scripts/check_symmetry.py` |
| T2.1 | 7 agents config + 7 experiment config cho các cell persona | `configs/agents/persona_*.json`, `configs/experiment/persona_baseline_*.json` |
| T2.2 | Test khoá luật viết persona | `ai_race/tests/test_personas.py` |
| T2.3 | `persona_condition` + `persona_role` chạy suốt engine → recorder → manifest → analyser; gate từ chối run không nhãn | 7 file, xem bên dưới |
| T3.1 | Cột dẫn xuất `own/opponent_unsafe_count_before`, `unsafe_count_diff_before`, `gap_bin`, `seat_index` | `analyze_ai_race.py:_add_dynamic_columns` |
| T3.2 | 10 bảng mô tả mới | `analyze_ai_race.py:_build_tables` |
| T3.3 | 6 đặc tả logit lồng nhau + vá CRN block pool qua persona | `analyze_ai_race.py:_fit_clustered_logit` |
| T3.4 | Tiêu chí người đóng băng trong JSON + bảng chấm điểm tự động | `results/scripts/human_reference.json`, `analyze_ai_race.py:_build_human_comparison` |

## Output analyser mới (28 file, trước là 18)

`treatment_contrasts.csv` · `persona_contrasts.csv` · `unsafe_by_lag_profile_turn.csv` ·
`unsafe_by_gap_bin_turn.csv` · `unsafe_by_gap_lag_turn.csv` · `winner_loser_pairs.csv` ·
`winner_loser_correlation.csv` · `horizon_distribution.csv` · `seat_balance.csv` ·
`gap_collinearity.csv` · `human_comparison.csv` · `human_comparison_metadata.json`

`unsafe_by_lag_profile_turn.csv` chính là ma trận chuyển trạng thái
`P(Unsafe_t | own_{t−1}, opp_{t−1})` — plan liệt kê riêng một bảng `transition_matrix.csv`
nhưng nó sẽ trùng số liệu hoàn toàn nên không sinh ra.

## Kiểm chứng bằng dữ liệu thật (mock)

- **Đẳng thức ΔS.** `gap_collinearity.csv` trên 902 quyết định/treatment:
  `max_abs_identity_residual = 0.0`, `pearson_r = 1.0`. Xác nhận bằng số rằng
  `progress_gap_before ≡ 0.5 × unsafe_count_diff_before` — hai biến này là một.
- **CRN qua persona.** Pool 3 run dir (`none`, `S_AA`, `S_AC`) → 450 race, 7.218 quan sát,
  **50** CRN block (`model::rep`), đúng như thiết kế vì cả ba dùng chung `seed: 260726`.
- **Gate persona.** Xoá nhãn khỏi turns/races/players/manifest → analyser raise
  `"persona_condition is missing for 150 race(s)"`. Có `--allow-missing-persona-condition`
  thì giữ ở stratum riêng, không trộn vào `none`.
- **Gate symmetry.** `--mock random` → 2% race degenerate, exit 0.
  `--mock safe` → 100% degenerate, exit 1 kèm hướng dẫn xử lý.
- **TOST.** Đối chiếu thủ công với `scipy.stats.norm`: khớp đến 1e-9. Điểm ước lượng nhỏ
  nhưng SE lớn → `not_replicated`, đúng ý nghĩa của equivalence test.

## Chưa làm

| Task | Lý do |
|---|---|
| T1.4 — backend đối thủ script (`ai_race/models/scripted.py`) | Plan đánh dấu tùy chọn; cần đổi routing trong `run_games_batched`, là thay đổi kiến trúc riêng |
| T2.4 — manipulation check (`persona_probe.py`) | Cần model thật, không chạy được trên máy này |
| WS4 — toàn bộ phần chạy | Không có GPU/API path ở đây; theo `CLAUDE.md` thì chạy trên Kaggle |

## Hai chỗ đã xử lý sau đó

### 1. `kaggle/benchmarks/ai_race_baseline.py` đã port sang v3

Task này cố ý không import `ai_race` mà reimplement cơ chế, nên nó phải giữ **bản sao
byte-for-byte** của template. Đã làm:

- `PROMPT_TEMPLATE` giờ là bản sao chính xác của `ai_race/prompts/ai_race_en.txt` →
  `sha256 = 27086bd8…`, khớp canonical.
- `PROMPT_VERSION = "ai-race-fairgame-v3"`.
- Thêm `apply_optional_blocks` (bản sao của engine) và render bằng bộ placeholder FAIRGAME.
- Thêm `PERSONA_CONDITION = "none"` vào cả ba loại row (turns/players/races) — thiếu nhãn
  thì analyser sẽ từ chối output của task này.

Kiểm chứng: prompt render ra **giống hệt từng ký tự** với `ai_race.engine.prompt.build_prompt`
(2.261 ký tự, cùng state). Khoá lại bằng 4 test trong `test_prompt_contract.py`, trong đó
một test so trực tiếp hằng số inline với file shipped — bản sao trôi một ký tự là test đỏ.

### 2. Persona bị hấp thụ giờ là lỗi, không phải im lặng

Thêm `_persona_identification()` trả về `identified` /
`confounded_with_protocol_signature`. Hành vi mới:

- **Primary mode:** persona có nhiều cell nhưng không biến thiên bên trong một
  `protocol_signature` → `_fit_clustered_logit` **raise**, kèm lý do và cách khắc phục.
- **Audit mode** (`--allow-mixed-protocols`): in WARNING ra stderr và vẫn chạy, nhưng ghi
  `persona_identification` vào `clustered_logit_metadata.json`.
- **Luôn luôn:** `analysis_manifest.json` mang `persona_identification`, kể cả khi không
  chạy logit — để người đọc thấy được bảng persona mô tả đang bị confound với batch mà
  không cần tự nhận ra một số hạng hồi quy bị thiếu.

Điều kiện thực tế để persona ước lượng được: **chạy tất cả cell persona trong cùng một
batch Kaggle** (cùng source revision, decoding, package versions). Thông điệp lỗi nói đúng
câu đó.
