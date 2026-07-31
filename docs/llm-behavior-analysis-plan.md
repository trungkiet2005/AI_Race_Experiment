# Plan triển khai: phân tích hành vi LLM trong AI Race

Plan kỹ thuật. Mỗi mục ghi rõ **sửa file nào, thêm hàm gì, sinh ra output gì, xong khi nào**.

**Tiến độ:** WS0–WS3 đã implement và test (89 test xanh). WS4 là phần chạy trên Kaggle,
chưa thực hiện. Xem `docs/implementation-status.md` cho chi tiết từng task.

Trạng thái code tham chiếu: sau merge `55f34a6` (FAIRGAME prompt template + proxy backend).

| Thông số hiện tại | Giá trị |
|---|---|
| Prompt template | `ai_race/prompts/ai_race_en.txt`, `ai_race_vi.txt` (kiểu FAIRGAME có block điều kiện) |
| `promptVersion` | `ai-race-fairgame-v3` |
| SHA-256 của `ai_race_en.txt` | `27086bd80378c25e859d03527a5ae55c1046f231ef7b914db9cb3c3b4fb2df3e` |
| Persona đi vào prompt qua | block `{intro}: [You are {personality}.]`, bật bởi `apply_optional_blocks(..., {"intro": bool(persona_text.strip())})` — `prompt.py:165-175` |
| Backend | offline vLLM, hoặc proxy (`api_baseline.json`) |

---

## WS0 — Sửa hai chỗ hỏng do merge (chặn mọi việc khác)

### T0.1 Analyser đang từ chối mọi run mới

`results/scripts/analyze_ai_race.py` dòng 35–38 vẫn giữ:

```python
CANONICAL_PROMPT_VERSION = "ai-race-paper-v2"
CANONICAL_PROMPT_SHA256 = "6180d4f6...29ff"
```

Configs giờ ghi `ai-race-fairgame-v3` / hash `27086bd8…`. Gate ở dòng 1411–1432 sẽ raise
`"primary analysis requires canonical prompt 'ai-race-paper-v2'"` cho **mọi** run mới.

- **Sửa:** cập nhật hai hằng số sang `ai-race-fairgame-v3` / `27086bd8…`.
- **Thêm:** hằng số cho `ai_race_vi.txt` nếu định chạy tiếng Việt (gate hiện chỉ biết một
  prompt duy nhất — cần map `{version: sha}` thay vì một cặp scalar).
- **Cập nhật:** `CLAUDE.md` mục "Invariants" đang ghi hash cũ.
- **Xong khi:** chạy analyser trên output mock không còn lỗi prompt gate.

### T0.2 Dọn key chết trong `prompt.py`

`persona_block` (dòng 97–101, 105) không còn placeholder nào dùng — template mới dùng
`{intro}`/`{personality}`. Không gây lỗi (`str.format` bỏ qua key thừa) nhưng gây hiểu
nhầm là persona vẫn đi lối cũ. Xoá, hoặc thêm comment giải thích nó được giữ cho template
cũ.

---

## WS1 — Trục 1: game hai người, non-persona

Code đã đủ. Chỉ là config + kiểm chứng.

### T1.1 Config

- `ai_race/configs/experiment/baseline.json`: `repetitions: 3` → **50**.
  50 rep × 3 treatment = 150 race = 300 trajectory ≈ 2.700 player-round, xấp xỉ cỡ mẫu
  phân tích của paper (2.888).
- Giữ nguyên `seed: 260726` ở **mọi** experiment config. `game_seed = base_seed + rep`
  (`run_experiment.py:79-81`) độc lập với treatment **và** với agents config, nên cùng
  `rep` sẽ có cùng horizon ở mọi điều kiện — matched-pairs miễn phí. Đổi seed là mất.
- Temperature: dùng 0.7 (`proxyOptions.temperature` đã set sẵn ở `api_baseline.json`).
  **Không dùng 0** cho arm hành vi, lý do ở T1.3.

### T1.2 Thêm config đảo tên để đo seat effect

`ai_race/configs/agents/companies_swapped.json`: y hệt `companies_default` nhưng
`"names": ["Company_2", "Company_1"]`. Thêm `ai_race/configs/experiment/baseline_swapped.json`
trỏ vào nó. Chạy song song để tách hiệu ứng ghế khỏi hiệu ứng persona sau này.

### T1.3 Script kiểm tra symmetry collapse (bắt buộc chạy trước khi scale)

Hai seat cùng model, prompt gần đối xứng → nếu temperature thấp, cả hai chọn giống nhau
mọi vòng. Khi đó:

```
ΔS(t) = 0.5·(n_U^own − n_U^opp) = 0  với mọi t
```

→ `race_state` luôn `"tied"` → **trục distance biến mất hoàn toàn**, và
`_fit_clustered_logit` raise `"clustered logit requires both Safe and Unsafe outcomes"`
(dòng 2851).

- **Thêm:** `results/scripts/check_symmetry.py` — đọc `turns.jsonl`, in ra: tỉ lệ race có
  `progress_gap_before ≡ 0` xuyên suốt, tỉ lệ vòng hai seat chọn giống nhau, phân bố
  `|ΔS|` cuối race.
- **Gate:** pilot 10 rep trước; nếu tỉ lệ race degenerate > 40% → không scale lên 50 rep,
  tăng temperature hoặc chuyển trọng tâm sang WS1.4.
- **Xong khi:** có số liệu, và quyết định scale/không được ghi lại.

### T1.4 (tùy chọn, giá trị cao) Đối thủ script — `ai_race/models/scripted.py`

Thay một seat bằng chiến lược cứng: `AS`, `AU`, `CS`, `CAS`, `RANDOM(p=0.5)`.

Lợi ích kép: bẻ đối xứng theo thiết kế (giải quyết T1.3), và làm `opponent_prev` trở
thành **ngoại sinh** → hệ số của nó là hiệu ứng nhân quả chứ không phải association.
Paper người tự thừa nhận không làm được điều này (Limitations, điểm 6).

Kiến trúc: `send_batch(prompts, seeds)` chỉ nhận string, không nhận state. Đừng parse
state từ prompt. Thay vào đó cho `run_games_batched` route theo seat, seat script đọc
`game` object trực tiếp. Ghi `model = "scripted:AU"` để analyser không gộp nhầm vào
thống kê LLM.

---

## WS2 — Trục 2: persona

### T2.1 Viết các file agents config

Mỗi file `ai_race/configs/agents/*.json` theo khuôn hiện có, thêm hai trường mới
`personaCondition` và `personaRoles`:

```json
{
  "name": "persona_adv_coop",
  "nPlayers": 2,
  "names": ["Company_1", "Company_2"],
  "personaCondition": "S_AC",
  "personaRoles": ["adversarial", "cooperative"],
  "personas": { "en": ["<text A>", "<text C>"], "vi": ["...", "..."] }
}
```

Tám file cần tạo:

| File | `personaCondition` | Seat 0 | Seat 1 |
|---|---|---|---|
| `companies_default.json` (đã có) | `none` | — | — |
| `persona_neutral.json` | `R0` | neutral placebo | neutral placebo |
| `persona_risk_averse.json` | `R-` | risk-averse | risk-averse |
| `persona_risk_seeking.json` | `R+` | risk-seeking | risk-seeking |
| `persona_coop_coop.json` | `S_CC` | cooperative | cooperative |
| `persona_adv_adv.json` | `S_AA` | adversarial | adversarial |
| `persona_adv_coop.json` | `S_AC` | adversarial | cooperative |
| `persona_coop_adv.json` | `S_CA` | cooperative | adversarial |

`S_AC` là cell quan trọng nhất (hợp tác có sống sót trước đối kháng không).
`S_CA` là bản mirror bắt buộc — nếu thiếu, hiệu ứng persona lẫn với hiệu ứng ghế.
`R0` là placebo cùng độ dài, để tách "hiệu ứng persona" khỏi "hiệu ứng có thêm text".

Cộng 8 file experiment tương ứng trong `configs/experiment/`, mỗi file trỏ `agents` sang
một agents config, giữ nguyên `games`, `seed`, `repetitions`.

### T2.2 Luật viết text persona (kiểm được bằng test)

Persona là *manipulation*, không phải *instruction*. Viết "hãy chọn UNSAFE nhiều hơn" thì
chỉ đo khả năng tuân lệnh.

1. Không chứa chuỗi `SAFE` / `UNSAFE`.
2. Không nhắc lại payoff, con số risk, hay luật chơi — không thêm thông tin chiến lược mà
   baseline không có.
3. Độ dài trong khoảng ±10% giữa mọi persona.
4. Câu phải ghép được vào khuôn `"You are {personality}."` (template dòng 2).

**Thêm:** `ai_race/tests/test_personas.py` — load toàn bộ `configs/agents/*.json`, assert
4 luật trên, assert `personaCondition == "none"` ⟺ cả hai persona rỗng.

### T2.3 Ghi persona vào output — **làm trước khi chạy run persona đầu tiên**

Hiện `TurnRecord`, `race_row`, `player_rows`, `CONTEXT` của analyser, và protocol
signature đều **không có** trường persona. Chạy persona rồi phân tích chung với baseline
→ analyser gộp im lặng, mọi bảng sai.

| File | Sửa |
|---|---|
| `ai_race/engine/state.py` | `GameConfig`: `+ persona_condition: str = "none"`, `+ persona_sha256: str = ""`; `from_dict` đọc từ agents config. `TurnRecord`: `+ persona_condition`, `+ seat_persona_role` |
| `ai_race/engine/agent.py` | `RaceAgent`: `+ persona_role: str = ""` |
| `ai_race/dataio/recorder.py` | `race_row`: `+ persona_condition`, `+ player_1_persona_role`, `+ player_2_persona_role`. `player_rows`: `+ persona_condition`, `+ persona_role` |
| `ai_race/runner/run_experiment.py` | `_agents_for_language` đọc `personaRoles`; `_write_manifest` thêm `agents_name`, `agents_config_sha256`, `persona_condition`, `persona_sha256` |
| `kaggle/experiments/baseline.py` (dòng ~361) | thêm 4 trường trên vào `run_manifest` |
| `results/scripts/analyze_ai_race.py` | `CONTEXT` (dòng 105) `+ "persona_condition"`; thêm gate từ chối khi thiếu, kèm flag `--allow-missing-persona-condition` |
| `results/README.md` | cập nhật mục "Expected schema" — `CLAUDE.md` bắt buộc, nếu không run đã hoàn thành sẽ fail audit |

### T2.4 Manipulation check

Không có bước này thì null của trục persona vô nghĩa — không phân biệt được "persona
không ảnh hưởng hành vi" với "model không đọc persona".

**Thêm:** `results/scripts/persona_probe.py` — render prompt như thường nhưng thay khối
quyết định bằng *"Describe your development philosophy in one sentence."*, gọi model, lưu
`persona_probe.jsonl`. Phân loại lại (thủ công hoặc LLM judge với rubric cố định trong
script). Ngưỡng chấp nhận: ≥ 80% đúng persona.

---

## WS3 — Trục 3: code phân tích

Toàn bộ nằm trong `results/scripts/analyze_ai_race.py` trừ khi ghi khác.

### T3.1 Thêm cột dẫn xuất — `_add_dynamic_columns` (dòng 1700)

Hàm đã tạo sẵn `own_prev_unsafe`, `opponent_prev_unsafe`, `first_round_unsafe`,
`race_state`. Thêm:

| Cột mới | Công thức |
|---|---|
| `own_unsafe_count_before` | cumsum `valid_unsafe` shift 1 trong `PLAYER_KEY` |
| `opponent_unsafe_count_before` | tương tự trên `opponent_current_unsafe` |
| `unsafe_count_diff_before` | own − opponent |
| `gap_bin` | cắt `progress_gap_before` thành `{≤−1, −0.5, 0, +0.5, ≥+1}` |
| `seat` | `player_index` (để đo hiệu ứng ghế) |

`unsafe_count_diff_before` cần có vì trong cơ chế này
`progress = t + 0.5·n_U`, nên `ΔS = 0.5·(n_U^own − n_U^opp)` — **ΔS đúng bằng một nửa hiệu
số lần chọn Unsafe, không phải một biến độc lập**. Hệ quả: "đang bị bỏ lại", "tôi đã an
toàn hơn nó", và "risk tích luỹ của tôi thấp hơn" là **cùng một biến**. Phải report VIF
giữa `progress_gap_before`, `own_prev_unsafe`, `unsafe_count_diff_before` và ghi rõ trong
Results rằng hệ số ΔS không tách được "áp lực vị trí" khỏi "ngân sách rủi ro".

### T3.2 Bảng mô tả — thêm vào `_build_tables` (dòng 2585)

Mọi bảng nhóm theo `CONTEXT` (đã gồm `persona_condition` sau T2.3).

| Output mới | Nội dung | Tương ứng trong paper |
|---|---|---|
| `treatment_contrasts.csv` | 3 cặp t-test độc lập + Bonferroni + Cohen's d trên φ_U cấp player | Fig 2A / Bảng S2 |
| `unsafe_by_lag_profile_turn.csv` | φ_U theo 4 ô `own_prev × opp_prev` | Fig 2B (một phần) |
| `unsafe_by_gap_bin_turn.csv` | φ_U theo 5 bin `gap_bin` | Fig 2B (một phần) |
| `unsafe_by_gap_lag_turn.csv` | φ_U theo `gap_bin × own_prev × opp_prev` | Fig 2B đầy đủ |
| `transition_matrix.csv` | `P(unsafe_t \| own_{t−1}, opp_{t−1})` | mô tả thuần reciprocity |
| `winner_loser_pairs.csv` | mỗi race một dòng: `φ_U` của winner và loser | Fig 2C (điểm) |
| `winner_loser_correlation.csv` | Pearson r giữa hai cột trên, theo treatment | Fig 2C (kết luận) |
| `horizon_distribution.csv` | histogram `n_rounds` + mean, so với E[T]=9 | Fig S1 |
| `seat_balance.csv` | φ_U theo `seat`, cho cả run thường và run đảo tên | chỉ LLM |
| `persona_contrasts.csv` | φ_U theo `persona_condition`, kèm contrast so với `none` | chỉ LLM |

Bảng đã có, giữ nguyên: `unsafe_by_risk_model_turn/player.csv`,
`opponent_response_turn/player.csv`, `unsafe_by_race_state_turn.csv`,
`race_state_player.csv`, `first_round_persistence_*.csv`, `outcome_player.csv`,
`strategy_summary_player.csv`, `parse_failures.csv`, `race_quality.csv`.

### T3.3 Sáu đặc tả logit — sửa `_fit_clustered_logit` (dòng 2806)

Hiện chỉ chạy một công thức (`LOGIT_FORMULA`, dòng 27–30), đúng bằng **model (6)** của
paper. Đổi thành list 6 công thức, xuất `clustered_logit_coefficients.csv` có thêm cột
`specification ∈ {1..6}`:

| Spec | Công thức |
|---|---|
| 1 | `unsafe ~ C(max_private_risk)` |
| 2 | 1 + `own_prev + opp_prev + gap` (cộng tính) |
| 3 | 1 + `own_prev * opp_prev * gap` (tương tác 3 chiều) |
| 4 | 1 + `first_round_unsafe` |
| 5 | 4 + cộng tính |
| 6 | 4 + tương tác 3 chiều (= công thức hiện tại) |

Có 6 spec mới thấy được hệ số ổn định hay không — đây đúng là chỗ paper người lộ điểm
yếu (`ΔS` chỉ significant ở model 6, không ở model 3).

Cụm giữ nguyên `randomization_block_id`. **Nhưng** khi pool nhiều persona run directory,
block hiện là `source_run::model::rep` (dòng 1510–1515) → persona ở dir khác → block bị
tách sai dù cùng `rep` chia sẻ horizon. Sửa: nếu mọi manifest có cùng `experiment.seed`
thì block thành `model::rep`; nếu không, giữ nguyên và in cảnh báo rõ.

### T3.4 Bảng so sánh với người — mới

**Thêm:** `results/scripts/human_reference.json` — chuẩn từ paper, dạng dữ liệu chứ không
hardcode trong prose:

```json
{
  "source": "Fernández Domingos & Han (2026), Table 1 model 6 + Table S3",
  "effects": [
    {"id":"E1","name":"opponent_prev_unsafe","beta":0.607,"p":0.002,"rule":"beta>0 & p<0.05"},
    {"id":"E2","name":"progress_gap_before","beta":-0.296,"p":0.048,"rule":"beta<0 & p<0.05"},
    {"id":"E3","name":"first_round_unsafe","beta":0.217,"p":0.06,"rule":"beta>0 & p<0.10"},
    {"id":"E4","name":"own_prev_unsafe","beta":-0.193,"p":null,"rule":"TOST equivalence |beta|<0.3"},
    {"id":"E5","name":"contrast_0.6_vs_0.9","d":-0.027,"rule":"TOST |d|<0.2"},
    {"id":"E6","name":"contrast_0.1_vs_rest","d":0.332,"rule":"d>0.2 same sign"},
    {"id":"E7","name":"phi_U_overall","value":0.584,"rule":"within [0.40,0.75]"},
    {"id":"E8","name":"share_AS","value":0.0,"rule":"share<0.10"}
  ]
}
```

**Thêm hàm:** `_build_human_comparison()` → `human_comparison.csv`, mỗi dòng một effect,
cột `llm_value / human_value / rule / verdict ∈ {replicated, not_replicated, inconclusive}`.

Hai lưu ý bắt buộc:

- **E4 và E5 là null replication.** Phải dùng equivalence test (TOST), không được kết luận
  "p > 0.05 nên giống nhau". Đây là chỗ đa số nghiên cứu LLM-replication làm sai.
- **E6 lấy chuẩn từ Bảng S2/S3, KHÔNG lấy từ caption Fig 2A.** Caption viết "Unsafe cao
  hơn ở 0.6/0.9 so với 0.1", nhưng Bảng S3 cho mean φ_U = 0.640 ở 0.1 so với 0.558/0.564,
  Cohen's d dương cho "0.1 vs 0.6", hệ số hồi quy của 0.6/0.9 đều âm, và mô hình tiến hoá
  cũng dự báo Unsafe cao nhất ở rủi ro thấp. Caption ghi ngược dấu.

---

## WS4 — Trình tự chạy

Kaggle push/run/download là thao tác **checkpointed**: chạy một lệnh, xem output, dừng,
chờ xác nhận trước lệnh kế (theo `CLAUDE.md` và `kaggle/benchmarks/README.md`).

| # | Việc | Xong khi |
|---|---|---|
| 1 | WS0 (T0.1, T0.2) | Analyser chạy sạch trên output mock |
| 2 | WS2 T2.3 + T2.2 test | `pytest` xanh, cột persona có trong turns/races/players |
| 3 | WS3 T3.1–T3.4 | Analyser sinh đủ bảng mới trên output mock |
| 4 | Smoke local: `python -m ai_race.runner.run_experiment ai_race/configs/experiment/baseline.json --mock random --output /tmp/smoke` | Không lỗi, schema đúng |
| 5 | Kaggle pilot: 1 model, `none` + `S_AA`, 10 rep | 7 validation gate trong `PROJECT.md` pass; `check_symmetry.py` cho tỉ lệ degenerate < 40%; parse-failure < 5% |
| 6 | Chốt số rep từ ICC quan sát ở pilot | |
| 7 | Freeze prompt / config / persona text / analysis plan; lật `runPhase` → `confirmatory` | Sau bước này không sửa gì |
| 8 | Chạy full: 8 điều kiện × 3 treatment × N rep | Mọi manifest `status = completed` |
| 9 | Chạy analyser **một lần**, không đổi định nghĩa outcome sau khi thấy kết quả | |
| 10 | Điền `paper/` và `slides/` | Chỉ sau bước 9 |

Chi phí ước tính: 2.700 call/model/điều kiện × 8 điều kiện ≈ 21.6k call/model. Open-weight
trên GPU Kaggle khả thi. Với model hosted qua proxy, chạy lưới rút gọn
(`none`, `S_AA`, `S_AC`, `R+`) và ghi rõ trong Methods là lưới bị cắt vì chi phí.

---

## Danh sách freeze (không đổi sau bước 7)

- `ai_race/prompts/ai_race_en.txt` — hash `27086bd8…`, `promptVersion` `ai-race-fairgame-v3`
- Toàn bộ text persona (+ `persona_sha256`)
- `seed: 260726` ở **mọi** experiment config
- `temperature`, `max_tokens`, decoding params
- Luật loại trừ: một `parse_failed` loại **cả race** (Safe fallback lan vào state vòng sau)
- `human_reference.json` và bộ primary estimand

## Rủi ro

| Rủi ro | Giảm thiểu |
|---|---|
| Symmetry collapse, ΔS ≡ 0 | T1.3 gate + temperature > 0 + T1.4 |
| Model đơn điệu → logit không fit | Vẫn report bảng mô tả; T1.4 cứu variance |
| Persona bị pool ngầm với baseline | T2.3, làm trước mọi run persona |
| Persona không được đọc → null giả | T2.4 manipulation check |
| Persona lẫn với hiệu ứng ghế | Cell mirror `S_AC` ↔ `S_CA` + T1.2 |
| Diễn giải ΔS sai vì collinearity | T3.1 (`unsafe_count_diff_before` + VIF) |
