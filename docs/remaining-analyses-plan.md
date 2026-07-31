# Plan triển khai: các phân tích còn thiếu so với paper gốc

Đối chiếu: [paper-analyses-inventory.md](paper-analyses-inventory.md) liệt kê 19 phân tích.
Hiện code phủ 6, không áp dụng 2, thiếu 11. Plan này lấp 11 mục đó.

Mỗi task ghi rõ **file nào, hàm gì, output gì, xong khi nào**. Phần lớn acceptance test
lấy **số có sẵn trong paper** làm ground truth — đây là điểm mạnh nhất của công việc này:
nó kiểm chứng được, không phải "chạy ra số rồi tin".

| Nhóm | Mục | Dependency mới | Ưu tiên |
|---|---|---|---|
| **WS-A** khoảng trống thực nghiệm | #3, #8, #10, phần dư #2 | không | cao — rẻ |
| **WS-B** lý thuyết dạng đóng | #11 (một phần), #15, #16 | không | cao — rẻ, kết quả chắc chắn |
| **WS-C** Monte Carlo payoff | #11 (đầy đủ) | không | trung bình |
| **WS-D** động lực tiến hóa | #12, #14, #17, #18, #19 | EGTtools (một phần) | thấp — đắt nhất |
| **WS-E** cầu nối lý thuyết ↔ dữ liệu | #13 | phụ thuộc WS-C | cao — đây là chỗ nối hai nửa |

---

## §0 Nguyên tắc

**Lý thuyết phải suy ra từ `GameConfig`, không hardcode.** Nếu ai đó đổi `unsafeProgress`
trong config, mọi con số lý thuyết phải đi theo. Điều này biến "cơ chế bị sửa" thành một
test đỏ thay vì một bảng sai âm thầm.

**Không reimplement cơ chế.** Monte Carlo phải dùng lại
[`ai_race/engine/scoring.py`](../ai_race/engine/scoring.py) (`joint_round_outcome`,
`race_outcomes`, `effective_private_risk`). Viết lại luật thắng/hòa lần thứ hai là cách
chắc chắn nhất để hai nửa của dự án lệch nhau.

**Lý thuyết không đọc dữ liệu.** Mô hình tiến hóa là tính chất của *trò chơi*, không của
LLM — nó cho cùng một dự báo dù người chơi là ai. Chỉ WS-E mới chạm vào output run.

Vị trí: **`ai_race/theory/`**, module thuần, không import backend, không import pandas ở
phần dạng đóng.

---

## WS-A — Ba khoảng trống thực nghiệm

### TA.1 (#3) Effect size trên mẫu `t ≥ 2`

Paper báo cáo **hai** bộ kiểm định treatment: Fig 2A trên toàn bộ vòng, và Table S
`pairwise-comparisons` trên đúng mẫu `t ≥ 2` của Table 1. Hai bộ khác nhau (d = 0,341 vs
hiệu ứng thô), và paper nói rõ vì sao. Hiện ta chỉ có bộ thứ nhất.

`later_unsafe_rate` **đã có sẵn** trong `player_metrics.csv`
([analyze_ai_race.py](../results/scripts/analyze_ai_race.py), `_build_player_metrics`).
Chỉ thiếu một lời gọi.

- **Sửa:** `_build_tables` — thêm
  `tables["treatment_contrasts_round2plus.csv"] = _pairwise_contrasts(player_metrics, strata=risk_strata, factor="max_private_risk", value="later_unsafe_rate")`
  và bản persona tương ứng.
- **Sửa:** `human_reference.json` — E5/E6 nên trỏ vào bộ `t ≥ 2` (paper lấy d từ Table S,
  không phải từ Fig 2A). Thêm trường `"contrast_table": "treatment_contrasts_round2plus.csv"`
  và cho `_build_human_comparison` đọc đúng bảng.
- **Xong khi:** hai bảng contrast tồn tại, `human_comparison.csv` chấm E5/E6 trên bộ
  `t ≥ 2`, và test khẳng định hai bảng cho số khác nhau khi vòng 1 khác các vòng sau.

### TA.2 (#8) Robustness — cluster jackknife

Paper chạy lại toàn bộ Table 1 sau khi loại một cặp bất thường, để chứng minh kết quả không
do một cluster chi phối. Analogue LLM chính xác nhất là **jackknife theo CRN block**: bỏ lần
lượt từng block, xem hệ số dao động bao nhiêu.

- **Thêm:** `_fit_logit_robustness(turns, output_directory)` trong analyser.
  Với đặc tả 6, refit `n_blocks` lần, mỗi lần bỏ một `randomization_block_id`.
- **Output:** `logit_robustness_jackknife.csv` — cột
  `term, coefficient_full, coefficient_min, coefficient_max, max_abs_shift, sign_stable, block_of_max_shift`.
- **Thêm biến thể loại trừ** trong cùng file, cột `variant`:
  `full` · `exclude_retried_races` (race có `retry_count > 0` — analogue của dữ liệu không
  hoàn hảo) · `exclude_min_horizon` (race dừng đúng ở vòng 5).
- **Cờ CLI:** `--fit-logit-robustness` (mặc định tắt; nó chạy N lần fit).
- **Xong khi:** `sign_stable` đúng cho mọi hệ số primary trên dữ liệu mock, và test khẳng
  định jackknife phát hiện được một block bịa ra có ảnh hưởng lớn.

### TA.3 (#10) Bảng thống kê mô tả gộp

Số liệu đã có nhưng nằm rải ở `unsafe_by_risk_model_player.csv` và `race_quality.csv`.
Paper gộp thành một bảng để đọc một lần.

- **Thêm:** `tables["sample_summary.csv"]` — mỗi dòng một `CONTEXT`, cột:
  `n_players, n_races, n_decisions, n_decisions_round2plus, mean_phi_U, median_phi_U, sd_phi_U,
  mean_phi_U_round2plus, mean_n_rounds, median_n_rounds, parse_failure_rate, n_races_excluded`.
- `median_phi_U` là bắt buộc, không phải tùy chọn: Fig 3B của paper so **median**, không so
  mean. WS-E cần cột này.
- **Xong khi:** một dòng cho mỗi ô `model × risk × persona`, và tổng `n_races` khớp
  `race_quality.csv`.

### TA.4 (phần dư #2) Mixed-effects logit

Không phải để đưa risk preference vào — LLM không có. Mà vì paper báo cáo **hai** cách xử lý
phụ thuộc trong cặp: cluster-robust SE (Table 1) và random intercept (Table S prereg), và
hai cách cho **kết luận khác nhau về `a_i^{t-1}`** (−0,62 p<0,001 so với ≈ 0 n.s.). Sự bất
đồng đó là một phát hiện, không phải chi tiết kỹ thuật — và ta hiện chỉ có một cách.

- **Thêm:** `_fit_mixed_effects_logit()` dùng
  `statsmodels.genmod.bayes_mixed_glm.BinomialBayesMixedGLM` (đã có trong extra `analysis`;
  statsmodels không có GLMM logit tần suất), random intercept theo `randomization_block_id`.
- **Output:** `mixed_effects_logit.csv` + metadata ghi rõ đây là **variational Bayes**, không
  phải MLE, nên hệ số không so trực tiếp được với cột cluster-robust.
- **Cờ:** `--fit-mixed-effects`.
- **Xong khi:** chạy được trên mock, và metadata nói rõ giới hạn diễn giải.

---

## WS-B — Lý thuyết dạng đóng (không thêm dependency)

### TB.1 (#11 một phần) Payoff kỳ vọng dạng đóng — `ai_race/theory/payoffs.py`

```python
def expected_horizon(config: GameConfig) -> float:
    """E[W] = min_rounds + (1 - p)/p."""

def unconditional_expected_payoff(config, own: str, opponent: str) -> float:
    """Closed form for the AS/AU pairs only."""
```

Chỉ hợp lệ cho cặp AS/AU vì payoff **affine theo W**: stage payoff là `W·π`, và tỉ lệ Unsafe
`n_U/W` là hằng (0 hoặc 1) nên hệ số rủi ro không phụ thuộc W. Hàm phải **raise** nếu được
gọi với CS/CAS — im lặng trả số sai ở đó là cái bẫy dễ mắc nhất trong cả mảng này.

Quy tắc payoff kỳ vọng (khác `terminal_scoring`, vốn *bốc* setback):

```
E[payoff] = (1 - p_r) x (stage_payoff + prize),   p_r = 0 nếu không thắng/hòa
```

**Acceptance test — số lấy thẳng từ paper (SI §3.1):**

| Đại lượng | Giá trị phải ra |
|---|---|
| `E[W]` với p=0,2, min=5 | `9.0` |
| π từ `b=4, c=1, s_U=1.5` | `(1, 0.6, 2.4, 2.0)` |
| `Π̄_AS,AS` | `59` |
| `Π̄_AS,AU` | `5.4` |
| `Π̄_AU,AS` | `(1 − p_max)·121.6` |
| `Π̄_AU,AU` | `(1 − p_max)·68` |

### TB.2 (#15) Phân loại stage game + ngưỡng social dilemma — `ai_race/theory/equilibria.py`

```python
def stage_game_class(config) -> str:
    """Return 'prisoners_dilemma' | 'deadlock' | 'chicken' | 'harmony' | ..."""

def social_dilemma_threshold(config) -> float:
    """p* above which mutual Safe beats mutual Unsafe in total expected payoff."""
```

**Acceptance:** với config canonical, `stage_game_class` trả `"deadlock"` (T=2,4 > P=2 >
R=1 > S=0,6 — thiếu điều kiện `R > P` của PD), và
`social_dilemma_threshold ≈ 0.132` (`1 − 59/68`). Test phải khẳng định 0,6 và 0,9 vượt
ngưỡng còn 0,1 thì không — đó là câu chuyện "cấu trúc lặp biến Deadlock thành social
dilemma".

### TB.3 (#16) Cân bằng Nash

```python
def nash_equilibria(payoff_matrix, strategies) -> list[tuple[str, str]]:
    """Exhaustive best-response search over pure symmetric profiles."""

def unconditional_nash_regions(config) -> dict:
    """The AS/AU boundaries as closed-form p_max thresholds."""
```

**Acceptance — bảng `tab:si:pd-nash` và đoạn văn kèm theo:**

| Kiểm tra | Kỳ vọng |
|---|---|
| Biên 2 chiến lược | `(AU,AU)` duy nhất khi `p<0.515`; cả hai khi `0.515≤p≤0.921`; `(AS,AS)` duy nhất khi `p>0.921` |
| 4 chiến lược, `p=0.1` | AU và CAS (tương đương) |
| 4 chiến lược, `p=0.6` | AU và CAS (tương đương) |
| 4 chiến lược, `p=0.9` | CS duy nhất |
| Mọi treatment | **AS không bao giờ là cân bằng** |
| Cấu trúc | AU và CAS đồng thời là cân bằng hoặc đồng thời không |

Hai test cuối là kiểm tra cấu trúc, không phải kiểm tra số — chúng bắt được lỗi mà so sánh
số đơn thuần bỏ lọt.

**Output:** `results/scripts/` sinh `theory_equilibria.csv`
(`max_private_risk, stage_game_class, social_dilemma_threshold, nash_equilibria, as_is_nash`).

---

## WS-C — Monte Carlo cho cặp có chiến lược điều kiện (#11 đầy đủ)

### TC.1 Mô phỏng ván — `ai_race/theory/payoffs.py`

```python
def simulate_matchup(config, own: str, opponent: str, *, replications: int = 10_000,
                     seed: int) -> float:
    """Mean expected payoff over sampled horizons for any strategy pair."""
```

- Bốc `W ~ min_rounds + Geom(p) − 1`, chặn trên bằng `config.max_rounds_safety_cap`.
- Sinh hành động bằng `strategy_action(strategy, round_number, opponent_history)` trong
  [`ai_race/engine/strategies.py`](../ai_race/engine/strategies.py), gọi **luân phiên cho
  cả hai ghế theo từng vòng**.

  > Đừng dùng `strategy_trajectory` ở đây. Nó nhận **một chuỗi hành động đối thủ cố định**,
  > nên chỉ đúng khi đối thủ vô điều kiện. Với CS đấu CAS, hành động của mỗi bên phụ thuộc
  > bên kia ở vòng trước — dùng `strategy_trajectory` sẽ lặng lẽ cho ra một ván khác.
- Cộng dồn bằng `joint_round_outcome`; kết thúc bằng `race_outcomes` +
  `effective_private_risk`.
- **Lấy kỳ vọng của xổ số rủi ro theo giải tích** (`× (1 − p_r)`), **không bốc setback**.
  Bốc setback làm phương sai tăng vô ích và khiến MC không hội tụ về dạng đóng.
- `seed` bắt buộc, mặc định dẫn xuất từ config — determinism là invariant của repo.

### TC.2 Ma trận payoff 4×4

```python
def expected_payoff_matrix(config, *, replications=10_000, seed) -> dict[tuple[str,str], float]:
```

Dùng dạng đóng cho 4 cặp AS/AU, MC cho 12 cặp còn lại.

**Acceptance:**
- MC cho các cặp AS/AU hội tụ về dạng đóng trong sai số MC (dùng `replications=200_000`,
  dung sai tương đối 1%). Đây là **test chéo giữa hai con đường tính** — mạnh hơn nhiều so
  với so từng cái với hằng số.
- `Π(CAS, AU) == Π(AU, AU)` và `Π(AU, CAS) == Π(AU, AU)` trong dung sai MC: CAS không phân
  biệt được với AU khi đối thủ chơi Unsafe từ vòng 1 (paper nêu rõ, SI §3.4).
- Cùng `seed` → cùng ma trận, đến từng bit.

**Output:** `theory_payoff_matrix.csv` (một dòng mỗi `risk × own × opponent`, cột
`payoff, method ∈ {closed_form, monte_carlo}, replications, seed`).

---

## WS-D — Động lực tiến hóa

### TD.1 Giới hạn đột biến nhỏ — `ai_race/theory/evolution.py` (numpy, **không cần EGTtools**)

Đây là đường tắt đáng giá. Với µ → 0, quần thể gần như luôn đơn hình, nên phân phối dừng
rút về chuỗi Markov nhúng trên 4 trạng thái đơn hình, tính từ xác suất cố định:

```python
def fitness_in_population(matrix, resident, mutant, k, Z) -> tuple[float, float]:
    """f_A(k) = [(k-1)Π(A,A) + (Z-k)Π(A,B)] / (Z-1), và đối xứng cho B."""

def fixation_probability(matrix, mutant, resident, *, Z, beta) -> float:
    """ρ = 1 / (1 + Σ_{i=1}^{Z-1} Π_{j=1}^{i} exp(-β(f_B(j) - f_A(j))))"""

def small_mutation_stationary(matrix, strategies, *, Z, beta) -> dict[str, float]:
    """Left eigenvector of the embedded chain M[B][A] = ρ_{B→A} / (n - 1)."""
```

Chỉ cần numpy. Phủ được `#17` và `#18` panel A ở chế độ µ nhỏ — tức phần lớn giá trị khoa
học của mảng này.

**Acceptance — chuyển pha ở Fig S5 và thứ tự ở Fig S6:**

| `p_r^max` | Chiến lược trội (Z=100, β=2) |
|---|---|
| < 0,2 | AU |
| 0,2 – 0,6 | CAS |
| > 0,6 | CS |

Cộng thêm: `AS` giữ khối lượng dừng không đáng kể ở mọi treatment (Fig S8 kết luận vậy trên
cả 4 mặt). Và một test lành mạnh: `Σ frequencies == 1`, mọi giá trị trong `[0,1]`.

### TD.2 Đột biến hữu hạn — cần EGTtools

`#12` (quét µ×β), `#14` (simplex tứ diện), `#18` panel B, `#19` (4 mặt) đều cần phân phối
dừng của chuỗi đầy đủ. Không gian trạng thái là số cách chia Z vào 4 nhóm =
`C(Z+3,3) = 176.851` với Z=100. Tự viết được nhưng không nên — paper dùng EGTtools
(`domingos2023egttools`), dùng lại cho khớp.

- **Thêm extra:** `theory = ["egttools>=0.1"]` trong `pyproject.toml`.
- Import lười, `pytest.importorskip` trong test, và analyser bỏ qua sạch sẽ nếu thiếu.
- **Output:** `theory_stationary_distribution.csv` (`risk, beta, mu, strategy, frequency`),
  `theory_expected_unsafe.csv` (`risk, beta, mu, expected_unsafe_frequency`).
- **Hình** `#14`/`#19` chỉ vẽ khi có `matplotlib`; đây là hình, không phải estimand — ưu
  tiên thấp nhất trong cả plan.

**Acceptance:** ở µ = 1/Z, β = 2, phân phối dừng đầy đủ khớp giới hạn µ nhỏ của TD.1 trong
dung sai lỏng — nghĩa là hai cài đặt độc lập xác nhận lẫn nhau.

---

## WS-E — Cầu nối lý thuyết ↔ dữ liệu (#13)

Đây là chỗ hai nửa gặp nhau, và là lý do đáng làm WS-C/WS-D.

- **Thêm:** trong analyser, `_build_theory_comparison(sample_summary, config_by_risk)` →
  `theory_vs_experiment.csv`, cột
  `max_private_risk, model, persona_condition, observed_median_phi_U, predicted_phi_U,
   difference, beta, mu, Z`.
- Dùng **median** φ_U, không phải mean — paper Fig 3B so median.
- Chạy ở **hai** điểm tham số như paper: tham chiếu `β=2, µ=β/Z=0.02` và khớp tốt
  `β=0.01, µ=0.05`.
- **Cảnh báo bắt buộc trong metadata:** dự báo lý thuyết **không phụ thuộc model LLM**. Nó
  giống nhau cho mọi model và mọi persona cell. Cột `difference` là *khoảng cách của LLM so
  với lý thuyết trò chơi*, không phải một fit.
- **Xong khi:** hướng của hiệu ứng treatment trong cột `predicted_phi_U` khớp Fig 3B (chênh
  nhỏ giữa 0,6 và 0,9, chênh lớn hơn so với 0,1).

---

## Acceptance test tổng hợp

Toàn bộ số dưới đây lấy từ paper và phải khớp — đây là bộ test có giá trị nhất vì nó độc
lập với code của ta:

```
E[W] = 9.0                                  SI §3.1
π = (1.0, 0.6, 2.4, 2.0)                    SI §3.1
Π̄_AS,AS = 59.0 ; Π̄_AS,AU = 5.4             SI §3.1
Π̄_AU,AS = (1-p)·121.6 ; Π̄_AU,AU = (1-p)·68 SI §3.1
stage game = Deadlock (T>P>R>S)             SI §3.4a
p* = 1 - 59/68 = 0.132353                   SI §3.4b
Nash 2 chiến lược: 0.514803 / 0.920588      SI §3.4c
Nash 4 chiến lược: AU|CAS, AU|CAS, CS       Table S pd-nash
AS không bao giờ là Nash                    SI §3.4d
Chuyển pha: 0.2 và 0.6                      Fig S5
Trội theo β: AU@0.1, CAS@0.6, CS@0.9        Fig S6
```

Mọi con số ở trên đã được tính lại độc lập từ tham số gốc (`b=4, B=100, c=1, s_U=1.5,
p=0.2, min_rounds=5`) và khớp paper; paper làm tròn `0.132353 → 0.132`,
`0.514803 → 0.515`, `0.920588 → 0.921`. Test nên so với giá trị tính từ `GameConfig`, không
so với số đã làm tròn trong bài.

---

## Rủi ro

| Rủi ro | Hậu quả | Giảm thiểu |
|---|---|---|
| Dùng `terminal_scoring` (bốc setback) cho payoff kỳ vọng | MC không hội tụ về dạng đóng, sai lệch nhỏ khó thấy | Tính kỳ vọng giải tích `×(1−p_r)`; test hội tụ chéo TC.2 |
| Áp dạng đóng cho cặp CS/CAS | Sai số im lặng — payoff không affine theo W | Hàm raise thay vì trả số |
| Hardcode π thay vì suy từ `GameConfig` | Đổi cơ chế thì lý thuyết lệch thầm lặng | Mọi hàm nhận `GameConfig` |
| Hiểu `strategy_summary_player.csv` là `#17` | Nhầm phân loại quỹ đạo quan sát với phân phối dừng lý thuyết | Đặt tên file lý thuyết có tiền tố `theory_`; ghi rõ trong `results/README.md` |
| Đọc `theory_vs_experiment.csv` như một fit | Lý thuyết không dùng dữ liệu LLM; "khớp" không phải bằng chứng về model | Cảnh báo trong metadata |
| EGTtools không cài được / API đổi | WS-D tắc | TD.1 (µ nhỏ) không cần EGTtools và phủ phần lớn giá trị |
| Jackknife N lần fit chậm | Analyser lâu | Sau cờ `--fit-logit-robustness`, mặc định tắt |

---

## Thứ tự thực hiện

1. **WS-A** (TA.1 → TA.3) — rẻ nhất, không dependency, lấp ngay 3 mục.
2. **WS-B** — thuần đại số, cho ngay `#15` `#16` với ground truth chắc chắn.
3. **WS-C** — mở khóa `#11` đầy đủ; test hội tụ chéo với WS-B là cột mốc quan trọng.
4. **WS-E** — nối lý thuyết vào dữ liệu, dùng được ngay sau WS-C với TD.1.
5. **TD.1** — giới hạn µ nhỏ, phủ `#17` `#18A`, chỉ cần numpy.
6. **TA.4** — mixed-effects, độc lập, làm lúc nào cũng được.
7. **TD.2** — EGTtools, `#12` `#14` `#18B` `#19`. Đắt nhất, giá trị thêm thấp nhất.

Sau bước 4 thì 15/19 mục được phủ mà **không thêm một dependency nào**. Bốn mục còn lại đều
là hình minh họa động lực học, không phải estimand.
