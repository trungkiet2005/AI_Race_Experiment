# Trạng thái triển khai plan phân tích hành vi LLM

Đối chiếu với [llm-behavior-analysis-plan.md](llm-behavior-analysis-plan.md) (đợt 1)
và [remaining-analyses-plan.md](remaining-analyses-plan.md) (đợt 2).
Test: `python3 -m pytest` → **188 passed** (đợt 1 để lại 89).

Đợt 2 nằm ở [§Đợt 2](#đợt-2--các-phân-tích-còn-thiếu-so-với-paper) bên dưới.

## Đợt 1 — đã xong

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

---

# Đợt 2 — các phân tích còn thiếu so với paper

Theo [remaining-analyses-plan.md](remaining-analyses-plan.md), bước 1–5 của
§"Thứ tự thực hiện". Phủ **15/19** mục của
[paper-analyses-inventory.md](paper-analyses-inventory.md), **không thêm dependency
nào**.

## Đã xong

| Task | Mục paper | Nội dung | File |
|---|---|---|---|
| TA.1 | #3 | `treatment_contrasts_round2plus.csv` + bản persona; `human_reference.json` thêm trường `contrast_table`, E5/E6 chấm trên mẫu `t ≥ 2` | `analyze_ai_race.py`, `human_reference.json` |
| TA.2 | #8 | `_fit_logit_robustness` — jackknife theo CRN block + 3 biến thể loại trừ; cờ `--fit-logit-robustness` | `analyze_ai_race.py` |
| TA.3 | #10 | `sample_summary.csv` — một dòng mỗi ô `CONTEXT`, có `median_phi_U` | `analyze_ai_race.py` |
| TB.1 | #11 (một phần) | `expected_horizon`, `unconditional_expected_payoff` dạng đóng | `ai_race/theory/payoffs.py` |
| TB.2 | #15 | `stage_game_class`, `social_dilemma_threshold` | `ai_race/theory/equilibria.py` |
| TB.3 | #16 | `nash_equilibria`, `unconditional_nash_regions`, `theory_equilibria.csv` | `ai_race/theory/equilibria.py`, `build_theory_tables.py` |
| TC.1 | #11 | `simulate_matchup` — MC, gọi `strategy_action` luân phiên cả hai ghế | `ai_race/theory/payoffs.py` |
| TC.2 | #11 (đầy đủ) | `expected_payoff_matrix` 4×4, `theory_payoff_matrix.csv` | `ai_race/theory/payoffs.py` |
| TD.1 | #17, #18A | Giới hạn µ nhỏ: `fitness_in_population`, `fixation_probability`, `small_mutation_stationary` | `ai_race/theory/evolution.py` |
| WS-E | #13 | `_build_theory_comparison` → `theory_vs_experiment.csv` | `analyze_ai_race.py` |

## Ground truth đã khớp

Mọi con số dưới đây tính từ `GameConfig`, không hardcode, và khớp SI của paper:

```
E[W] = 9.0                                   test_theory_payoffs.py
pi = (1.0, 0.6, 2.4, 2.0)
Pi_AS,AS = 59.0 ; Pi_AS,AU = 5.4
Pi_AU,AS = (1-p)·121.6 ; Pi_AU,AU = (1-p)·68
stage game = deadlock, T>P>R>S               test_theory_equilibria.py
p* = 1 - 59/68 = 0.132353
Nash 2 chiến lược: 0.514803 / 0.920588
Nash 4 chiến lược: AU|CAS · AU|CAS · CS
AS không bao giờ là Nash (ở mọi treatment, mọi profile)
```

Hai test cấu trúc theo yêu cầu:
`test_the_four_au_cas_profiles_stand_or_fall_together` (bốn tổ hợp AU/CAS cùng là
cân bằng hoặc cùng không) và `test_monte_carlo_converges_to_the_closed_form`
(200.000 replication, dung sai tương đối 1%, cho cả 4 cặp AS/AU).

## Ba chỗ lệch khỏi plan — đã kiểm chứng, nói rõ lý do

### 1. Thêm đường tính **exact enumeration**, và nó là mặc định cho Nash + tiến hoá

Plan chỉ có hai đường: dạng đóng cho AS/AU và MC cho 12 cặp còn lại. Vấn đề: cả bốn
chiến lược đều **tất định** khi biết `W`, nên nguồn ngẫu nhiên duy nhất là `W`, mà
phân phối của `W` chỉ có < 100 nguyên tử. Cộng thẳng qua nguyên tử cho kết quả
**chính xác**, rẻ hơn lấy mẫu.

Điều này quan trọng, không phải tối ưu hoá:

- `Π(CAS,AU)` và `Π(AU,AU)` là **cùng một số** (CAS gặp đối thủ Unsafe từ vòng 1 thì
  chơi Unsafe mọi vòng). Dưới MC 20.000 replication chúng lệch ~0,043.
- Dò best-response vét cạn đọc chênh lệch 0,043 đó là "ưu thế chặt", nên
  `(AU,CAS)` và `(CAS,AU)` **bị loại khỏi tập cân bằng** — phá đúng tính chất cấu
  trúc mà plan yêu cầu test.
- Với enumeration, đẳng thức đúng đến 1e-12 và cả bốn tổ hợp đều là cân bằng.

`simulate_matchup` vẫn giữ nguyên như spec (đúng cách paper làm) và dùng cho test
hội tụ chéo. `test_monte_carlo_needs_a_wider_equilibrium_tolerance` khoá lại chính
hiện tượng này.

### 2. TD.1 **không thể** tái tạo chuyển pha AU→CAS ở 0,2 — đây là giới hạn toán học

Plan đặt acceptance: `< 0,2 → AU`, `0,2–0,6 → CAS`, `> 0,6 → CS`.

Thực tế trong giới hạn µ → 0:

```
Π(CAS,AU) = Π(AU,AU)  và  Π(AU,CAS) = Π(CAS,CAS)
⟹ ρ(AU→CAS) = ρ(CAS→AU) = 1/Z = 0,01 chính xác, với MỌI β và MỌI treatment
```

AU và CAS **trung tính hoàn toàn** với nhau, nên giới hạn chia đôi khối lượng
(0,5/0,5) ở mọi mức rủi ro dưới ngưỡng CS. Phân biệt được chúng đòi hỏi µ hữu hạn:
chỉ khi trong quần thể còn đột biến AS/CS thì thế đối xứng mới bị phá. Đó là TD.2
(EGTtools), chưa làm.

Fig S6 cũng vậy: `AU@0.1` và `CAS@0.6` không tách được, chỉ `CS@0.9` tái tạo được.

Những gì TD.1 **có** tái tạo, đã test:

| Kết quả | Trạng thái |
|---|---|
| Chuyển pha sang CS ở `p ≈ 0,637` (paper: "> 0,6") | ✅ |
| AS giữ khối lượng không đáng kể ở mọi treatment (< 0,05; Fig S8) | ✅ |
| Cặp {AU, CAS} giữ > 95% khối lượng ở 0,1 và 0,6, < 5% ở 0,9 | ✅ |
| β tăng → phân phối tập trung hơn (Fig S6 hướng) | ✅ |
| Σ tần suất = 1, mọi giá trị trong [0,1] | ✅ |
| Tách AU khỏi CAS (chuyển pha ở 0,2) | ❌ cần TD.2 |

Giới hạn này ghi trong docstring của `evolution.py`, trong
`theory_metadata.json:mutation_regime_caveat`, và trong
`theory_vs_experiment_metadata.json`.

### 3. WS-E phải làm **sau** TD.1, và acceptance của nó cũng bị giới hạn trên chặn

Plan xếp WS-E là bước 4, TD.1 là bước 5. Nhưng `theory_vs_experiment.csv` có cột
`predicted_phi_U, beta, mu, Z` — đó chính là output của phân phối dừng, tức TD.1.
Không có TD.1 thì không có gì để so. Đã làm TD.1 trước rồi mới WS-E; cả hai
deliverable đều giao đủ.

Acceptance của WS-E ("chênh nhỏ giữa 0,6 và 0,9, chênh lớn hơn so với 0,1") là hệ
quả của µ hữu hạn nên cũng không đạt được:

| `p_r^max` | `predicted_phi_U` (reference β=2) | (best_fit β=0,01) |
|---|---|---|
| 0,1 | 1,000 | 1,000 |
| 0,6 | 1,000 | 0,931 |
| 0,9 | 0,010 | 0,064 |

Hướng đúng (Unsafe giảm khi rủi ro tăng) nhưng khoảng cách 0,6–0,9 lớn chứ không
nhỏ. Cột `mu` ghi `0.0` (giá trị **thực sự** dùng), `nominal_mu` ghi 0,02 / 0,05 là
điểm tham số của paper mà dòng đó xấp xỉ, và `mutation_regime` ghi
`small_mutation_limit`. Metadata nói thẳng rằng khớp Fig 3B cần chuỗi µ hữu hạn.

## Output mới

Analyser (`analyze_ai_race.py`), 35 file thay vì 31:

`sample_summary.csv` · `treatment_contrasts_round2plus.csv` ·
`persona_contrasts_round2plus.csv` · `logit_robustness_jackknife.csv` ·
`logit_robustness_metadata.json` · `theory_vs_experiment.csv` ·
`theory_vs_experiment_metadata.json`

Script lý thuyết mới `results/scripts/build_theory_tables.py` → 5 file trong
`results/derived/ai_race_theory/`:

`theory_payoff_matrix.csv` · `theory_equilibria.csv` ·
`theory_stationary_distribution.csv` · `theory_expected_unsafe.csv` ·
`theory_metadata.json`

Tiền tố `theory_` là bắt buộc: `theory_stationary_distribution.csv` là phân phối
dừng **dự báo**, còn `strategy_summary_player.csv` phân loại quỹ đạo LLM **quan
sát được**. Hai câu hỏi khác nhau. Ghi rõ trong `results/README.md` và trong
`theory_metadata.json:naming_warning`.

## Kiểm chứng bằng dữ liệu thật (mock)

```bash
python3 -m ai_race.runner.run_experiment ai_race/configs/experiment/baseline.json \
  --mock random --output /tmp/smoke
python3 results/scripts/analyze_ai_race.py --input /tmp/smoke --output /tmp/derived \
  --fit-logit --fit-logit-robustness \
  --allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs
python3 results/scripts/build_theory_tables.py --output /tmp/theory
```

- **Hai cửa sổ phân tích thực sự khác nhau.** Trên 150 race mock:
  `treatment_contrasts.csv` cho mean 0,51484, `treatment_contrasts_round2plus.csv`
  cho 0,50000. Vòng 1 khác các vòng sau, đúng như lý do paper báo cáo hai bộ.
- **Jackknife chạy 50 block × 3 biến thể.** Mọi hệ số primary
  (`opponent_prev_unsafe`, `progress_gap_before`, `first_round_unsafe`,
  `own_prev_unsafe`) có `sign_stable = True`. Hai dummy treatment ở mức 1e-16 (mock
  cho ba treatment cùng quỹ đạo qua CRN) được đánh dấu
  `negligible_at_full_sample = True` thay vì bị báo là đổi dấu — nhiễu số học không
  phải bất ổn. `exclude_min_horizon` bỏ 9 block → 41 block, 2.190 quan sát.
- **`sample_summary.csv`** cho 3 dòng, `n_races` tổng 150, khớp `race_quality.csv`.
- **`theory_vs_experiment.csv`** cho 6 dòng (3 treatment × 2 điểm tham số);
  `predicted_phi_U` giống hệt nhau qua mọi model — đúng như thiết kế, và có test
  `test_theory_prediction_does_not_depend_on_the_model` khoá lại.

## Chưa làm (để lượt sau, theo phạm vi đã thống nhất)

| Task | Mục paper | Lý do |
|---|---|---|
| TA.4 | phần dư #2 | Mixed-effects logit (`BinomialBayesMixedGLM`), ngoài phạm vi đợt này |
| TD.2 | #12, #14, #18B, #19 | Cần thêm dependency EGTtools; cũng là thứ duy nhất tháo được hai giới hạn ở §2 và §3 trên |
