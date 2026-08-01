# Phân tích run `frontier` — Gemini 3 family (Google, qua proxy Kaggle)

Nguồn dữ liệu: `results/frontier/` (`baseline/` — 3 model không persona; `persona/` — 6 cell
persona, 5/6 hoàn chỉnh, trên `google/gemini-3-flash-preview`).
Đối chiếu: [docs/paper-analyses-inventory.md](../../docs/paper-analyses-inventory.md) — 19 phân
tích của paper gốc; [docs/implementation-status.md](../../docs/implementation-status.md) — tình
trạng đã code trong `analyze_ai_race.py`.
Dashboard: [visualizations/dashboard.html](visualizations/dashboard.html).
Bảng dẫn xuất: [derived/](derived/) (35 file, sinh bởi `results/scripts/analyze_ai_race.py`).

> **Đây là PILOT, không phải confirmatory.** Mọi run có `run_phase = pilot` và manifest schema
> `ai-race-results-v1` — theo [results/README.md](../../results/README.md), schema này "cố ý
> không đủ cho phân tích chính gộp" (thiếu source hash, decoding contract đầy đủ, seed
> provenance đầy đủ so với hai schema Kaggle/Kaggle-Benchmark). Vì vậy analyser chỉ chạy được
> khi bật cả ba cờ audit: `--allow-mixed-protocols --allow-nonfinal-runs
> --allow-nonconfirmatory-runs --allow-missing-persona-condition`. Theo
> [docs/running-the-experiment.md](../../docs/running-the-experiment.md): **"Đừng dùng ba flag đó
> cho kết quả thật."** Tài liệu này đọc như một audit/exploratory pass trên dữ liệu pilot, không
> phải bằng chứng confirmatory cho PROJECT.md.

---

## 0. Các phân tích đã được code (đối chiếu `docs/`)

`results/scripts/analyze_ai_race.py` (~3.500 dòng) hiện phủ **15/19** phân tích định lượng của
bài báo gốc (theo bảng tra cứu trong `paper-analyses-inventory.md`), cộng thêm các phân tích
riêng cho hành vi LLM không có trong paper gốc (parse-quality, seat balance, CRN/persona
confounding). Việc còn thiếu (TD.2 — EGTtools, mixed-effects logit) cần thêm dependency và nằm
ngoài phạm vi báo cáo này.

| # | Phân tích | Trạng thái | Output |
|---|---|---|---|
| 1 | t-test cặp treatment (toàn vòng) | ✅ | `treatment_contrasts.csv` |
| 2 | Mixed-effects logistic tiền đăng ký (risk preference) | ❌ chưa làm (TA.4) | — |
| 3 | Effect size + t-test mẫu t ≥ 2 | ✅ | `treatment_contrasts_round2plus.csv` |
| 4 | φ_U theo ΔS × lag profile | ✅ | `unsafe_by_gap_lag_turn.csv`, `unsafe_by_lag_profile_turn.csv` |
| 5 | Tương quan φ_U thắng–thua | ✅ | `winner_loser_correlation.csv`, `winner_loser_pairs.csv` |
| 6 | **Hồi quy logistic panel cluster-robust, 6 đặc tả** | ✅ | `clustered_logit_coefficients.csv` |
| 7 | Covariate nhân khẩu + risk preference | N/A cho LLM | agent không có nhân khẩu/risk preference elicited |
| 8 | Robustness loại dropout / jackknife theo CRN block | ✅ | `logit_robustness_jackknife.csv` |
| 9 | Phân phối số vòng thực tế | ✅ | `horizon_distribution.csv` |
| 10 | Thống kê mô tả mẫu | ✅ | `sample_summary.csv` |
| 11 | Ma trận payoff kỳ vọng (đóng + Monte Carlo + exact enumeration) | ✅ (lý thuyết, không cần dữ liệu run) | `ai_race/theory/payoffs.py` |
| 12 | Quét (µ, β) × treatment | ❌ cần TD.2 (EGTtools) | — |
| 13 | So khớp median φ_U thực nghiệm vs mô hình | ✅ | `theory_vs_experiment.csv` |
| 14 | Simplex tứ diện 4 chiến lược | ❌ cần TD.2 | — |
| 15 | Deadlock + ngưỡng social dilemma | ✅ (lý thuyết) | `ai_race/theory/equilibria.py` |
| 16 | Cân bằng Nash (2 & 4 chiến lược) | ✅ (lý thuyết) | `theory_equilibria.csv` |
| 17 | Phân phối chiến lược theo `p_r^max` | ⚠️ một phần — tách được ngưỡng CS, không tách AU/CAS (giới hạn µ→0, xem TD.1) | `theory_stationary_distribution.csv` |
| 18 | Độ nhạy theo β, µ | ⚠️ một phần, cùng giới hạn | `theory_expected_unsafe.csv` |
| 19 | Động lực 4 mặt 3-chiến-lược | ❌ cần TD.2 | — |

Ngoài 19 mục trên, analyser còn có các phân tích **không nằm trong paper gốc** nhưng bắt buộc
cho dữ liệu LLM: cổng chất lượng parse (`parse_failures.csv`), đẳng thức cơ chế ΔS
(`gap_collinearity.csv`), cân bằng ghế (`seat_balance.csv`), phân loại chiến lược gần nhất
(`strategy_summary_player.csv`), và đối chiếu 8 hiệu ứng người–LLM (`human_comparison.csv`, mục
E1–E8 bên dưới) — các mục này không có tương đương trực tiếp trong bảng 19 mục vì paper gốc
không cần chúng (người tham gia không "parse fail", không có seat-order artefact do thiết kế
khác).

Chi tiết đầy đủ nằm trong hai file gốc; mục này chỉ tóm tắt để đối chiếu nhanh với báo cáo bên
dưới.

---

## 1. Provenance & cổng chất lượng

| Mục | Giá trị |
|---|---|
| Backend | `proxy` (Kaggle), `ai-race-results-v1` — không đủ điều kiện cho phân tích chính |
| `prompt_version` | `ai-race-fairgame-v3` (canonical, khớp hash) |
| Model | `google/gemini-3-flash-preview`, `google/gemini-3.1-flash-lite-preview`, `google/gemini-3.5-flash-lite` |
| Persona | `none` (baseline) · `R0` neutral · `R-` risk-averse · `S_AA` adversarial-adversarial · `S_AC`/`S_CA` adversarial-vs-cooperative (theo ghế) — tất cả trên `gemini-3-flash-preview` |
| Persona **bị loại** | `R+` (risk-seeking): `status="running"`, 0 race hoàn chỉnh — không có `races.csv`/`players.csv` để phân tích |
| Lưới | 3 risk treatment × (3 model baseline + 5 persona cell) × 10 rep (S_* chỉ 3 rep/seat) |
| Race | **177/177 giữ lại**, 0 bị loại vì parse/forced-stop/noncanonical mechanism |
| Quyết định | **3.168**, 354 player-race |
| Parse failure | **0** trên toàn bộ |
| Đẳng thức ΔS | `max_abs_identity_residual = 0,0`, `pearson_r = 1,0` mọi stratum — cơ chế state đúng |
| Số vòng trung bình | 8,95 (kỳ vọng lý thuyết 9 với `p = 0,2`) |
| CRN block | 30 block (`source_run/model/rep`), dùng cho cluster-robust SE |
| Persona × protocol | **Confound với batch** — mỗi persona cell chạy ở protocol signature riêng → hệ số persona trong logit **không ước lượng được** (xem §4) |

Mọi cổng cơ chế (step increment, stage payoff, prize, setback, stop rule…) đều pass — xem
`analysis_manifest.json:mechanics_checks_passed` (31 mục, tất cả `true`).

---

## 2. φ_U theo mức rủi ro (tương ứng Fig. 2A của paper)

Tần suất Unsafe trung bình mỗi người chơi (`unsafe_by_risk_model_player.csv`), model baseline
(không persona):

| Model | risk = 0.1 | risk = 0.6 | risk = 0.9 |
|---|---|---|---|
| gemini-3-flash-preview | 1,000 | 0,723 | 0,539 |
| gemini-3.1-flash-lite-preview | 1,000 | 0,801 | 0,699 |
| gemini-3.5-flash-lite | 0,838 | 0,708 | 0,626 |

**Khác biệt lớn với paper người:** ở người, φ_U trung bình dao động hẹp quanh 0,58 (0,64 / 0,56
/ 0,56 theo treatment) và 0,6-vs-0,9 **không có hiệu ứng** (H1.2 bị bác bỏ). Ở cả ba model LLM
này, φ_U **giảm đơn điệu và mạnh** theo risk, và **0,6 vs 0,9 vẫn có ý nghĩa thống kê**
(`treatment_contrasts_round2plus.csv`, baseline không persona: d = 1,03–1,87, p < 0,01 ở cả ba
model) — ngược hướng với
phát hiện null quan trọng nhất của paper gốc. Xem biểu đồ 1 trong dashboard.

Cả ba model **Unsafe gần như tuyệt đối ở risk = 0,1** (φ_U ≥ 0,84) — sát trần lý thuyết, khác
mức 0,64 quan sát ở người cùng treatment.

---

## 3. Hiệu ứng persona (chỉ mô tả — không suy luận nhân quả)

Vì mỗi persona cell chạy ở một protocol signature riêng (batch khác nhau), **không thể tách**
hiệu ứng persona khỏi hiệu ứng batch bằng hồi quy (§4 giải thích cơ chế). Bảng dưới chỉ mô tả
chênh lệch quan sát được, không phải ước lượng nhân quả:

| Persona | risk = 0.1 | risk = 0.6 | risk = 0.9 | Ghi chú |
|---|---|---|---|---|
| Baseline (không persona) | 1,000 | 0,723 | 0,539 | |
| R0 — trung lập tường minh | 0,992 | 0,714 | 0,616 | gần baseline |
| **R− — risk-averse** | 0,405 | **0,000** | **0,000** | sụp hoàn toàn về Safe ở risk cao |
| **S_AA — cả hai ghế adversarial** | 1,000 | 1,000 | 0,952 | gần như luôn Unsafe ở mọi risk |
| S_AC — ghế adversarial (đối thủ cooperative) | 0,294 | 0,279 | 0,151 | thấp hơn hẳn baseline dù được gán "adversarial" |
| S_CA — ghế cooperative (đối thủ adversarial) | 0,522 | 0,246 | 0,175 | |

Hai quan sát đáng chú ý:

1. **Persona risk-averse (R−) khoá cứng vào Safe** ngay khi risk vượt 0,1 — mạnh hơn nhiều so
   với null-result về risk preference của người (H2.1–H2.3 không được ủng hộ ở paper gốc).
   Prompt persona ở đây có hiệu lực hành vi rất lớn, khác biệt hẳn với "sở thích rủi ro elicited"
   (Eckel–Grossman) không dự báo được gì ở người.
2. **Nhãn "adversarial" không đơn điệu với Unsafe.** S_AA (cả hai ghế adversarial) gần 1,0 Unsafe
   như kỳ vọng, nhưng ghế adversarial trong cặp bất đối xứng (S_AC) lại có φ_U **thấp nhất**
   trong toàn bộ bảng (0,15–0,29) — thấp hơn cả ghế cooperative đối diện (S_CA, 0,18–0,52). Cần
   đọc lại prompt persona S_AC/S_CA hoặc coi đây là artefact của cỡ mẫu rất nhỏ (n = 6
   người/ô, 3 race/ô) trước khi diễn giải thêm.

`persona_contrasts.csv` trống — analyser không sinh được so sánh cặp vì không có hai persona
nào cùng protocol signature để so trực tiếp trong cùng risk treatment.

---

## 4. Hồi quy logistic panel cluster-robust (tương ứng Table 1)

Đặc tả (6) — bão hoà nhất, kiểm soát `protocol_signature` để tách nhiễu batch:

```
unsafe ~ C(risk) + first_round_unsafe + own(t-1) × opponent(t-1) × ΔS(t-1) + C(protocol_signature)
```

Cluster-robust SE theo `source_run/model/rep` (30 block), N = 2.814 quyết định (mẫu t ≥ 2, sau
khi bỏ vòng 1).

| Biến | β̂ | SE | p |
|---|---|---|---|
| risk = 0,6 (so với 0,1) | **−2,116** | 0,285 | < 0,001 |
| risk = 0,9 (so với 0,1) | **−2,791** | 0,352 | < 0,001 |
| Unsafe vòng 1 | **+1,674** | 0,509 | 0,001 |
| Own(t−1) = Unsafe | −0,595 | 0,364 | 0,102 |
| **Opponent(t−1) = Unsafe** | **+1,486** | 0,341 | **< 0,001** |
| Own(t−1) × Opp(t−1) | −0,984 | 0,493 | 0,046 |
| ΔS(t−1) (vị thế đua) | −0,224 | 0,312 | 0,472 |
| Own(t−1) × ΔS(t−1) | −3,462 | 1,992 | 0,082 |
| Opp(t−1) × ΔS(t−1) | −0,104 | 0,811 | 0,898 |
| Tương tác 3 chiều | +2,464 | 2,510 | 0,326 |

Pseudo R² = 0,450 (specification 6) — **cao hơn nhiều** so với paper người (0,006–0,040).

**Ba điểm khớp hướng với paper người:**
1. **Hành động vòng trước của đối thủ là dự báo mạnh nhất và có ý nghĩa nhất** — đúng phát hiện
   trung tâm của paper (β người model 6 = 0,607, p = 0,002; ở đây β = 1,486, p < 0,001) — cùng
   dấu, cùng thứ hạng ảnh hưởng, nhưng độ lớn hệ số **gấp ~2,4 lần**.
2. **Hành động vòng 1 dự báo hành vi về sau** — cùng dấu dương với paper (β người = 0,217, p <
   0,10; ở đây β = 1,674, p = 0,001) — "behavioural momentum" mạnh hơn hẳn ở LLM.
3. **Hành động riêng của mình (own(t−1)) không có ý nghĩa** khi đã kiểm soát đối thủ — khớp kết
   luận "không phải quán tính hành động đơn thuần" của paper.

**Hai điểm lệch hướng với paper người:**
- **Hệ số treatment có ý nghĩa mạnh** (p < 0,001 cả hai bậc risk) — trái ngược hoàn toàn với
  paper người, nơi hệ số treatment âm nhưng **không có ý nghĩa thống kê** trong panel t ≥ 2.
- **ΔS(t−1) không có ý nghĩa chính** (p = 0,47) ở đây, trong khi paper người tìm thấy hiệu ứng có
  ý nghĩa ở model 6 (β = −0,296, p = 0,048).

Robustness (`logit_robustness_jackknife.csv`) — jackknife 30 block × biến thể exclude/retry: cần
đọc trước khi tin các hệ số trên là ổn định qua từng block (chưa audit chi tiết trong báo cáo
này).

**Lưu ý nội sinh:** giống paper gốc, biến trễ không ngoại sinh chặt → đọc như liên hệ có điều
kiện, không phải hiệu ứng nhân quả. Điều này **nghiêm trọng hơn** ở đây vì mẫu pilot còn trộn
nhiều protocol signature khác nhau (persona × batch).

---

## 5. Tương quan φ_U thắng–thua trong cặp (tương ứng Fig. 2C)

`winner_loser_correlation.csv` — mỗi stratum (model × persona × risk) chỉ có 1–5 race quyết định
thắng-thua (loại race hoà), nên số liệu **chỉ mang tính minh hoạ**, không đủ mạnh để kiểm định.
Quan sát chung: người thắng có φ_U cao hơn người thua ở hầu hết stratum (đúng hướng paper), biên
độ chênh lệch (`mean_winner_minus_loser`) dao động 0,08–0,45. Không đủ N để bình luận về việc
tương quan có tăng theo risk hay không như paper người tìm thấy.

---

## 6. Phân loại chiến lược gần nhất (baseline, không persona)

`strategy_summary_player.csv`, gộp theo nearest-strategy set (Hamming) trên 3 model baseline:

| Risk | AS | AU | CS | CAS | Tie/Other |
|---|---|---|---|---|---|
| 0,1 | 0% | 5% | 0% | 13% | 82% |
| 0,6 | 0% | 32% | 0% | 45% | 23% |
| 0,9 | 3% | 37% | 0% | 40% | 20% |

Ở risk thấp phần lớn quỹ đạo là "Tie/Other" — hợp lý vì φ_U ≈ 1,0 sát trần khiến nhiều player
gần như luôn Unsafe (khớp AU) nhưng lệch đủ để không khớp Hamming tuyệt đối với bất kỳ chiến
lược thuần nào. Không quan sát **CS** đáng kể ở bất kỳ risk nào (0% mọi treatment) — khác paper
lý thuyết, nơi CS là cân bằng Nash duy nhất ở risk = 0,9 (`theory_equilibria.csv`). **AS gần như
vắng mặt** (0–3%) — khớp kết luận lý thuyết "AS không bao giờ là cân bằng" của paper gốc.

---

## 7. Đối chiếu 8 hiệu ứng người–LLM (E1–E8)

Từ `human_comparison.csv`, chấm tự động theo `results/scripts/human_reference.json`.

> **"Replicated" chỉ xét dấu (chiều hiệu ứng) và một ngưỡng ý nghĩa thống kê / effect size tối
> thiểu — không xét con số LLM có gần con số người hay không** (cột "Tiêu chí chấm" dưới đây là
> điều kiện thật sự dùng để chấm, lấy trực tiếp từ `human_comparison.csv:criterion`). Hệ số hồi
> quy của hai nghiên cứu khác nhau (người thật vs LLM, cỡ mẫu khác, prompt khác) không nằm trên
> cùng một thang đo tuyệt đối để so trực tiếp; cái so được là *có tồn tại cùng một quy luật hành
> vi hay không*, không phải *độ mạnh của quy luật đó có bằng nhau không*. Vì vậy ở E1/E3/E6, hệ
> số LLM lớn hơn hệ số người 2–9 lần **vẫn** được chấm "replicated" — khoảng cách độ lớn đó tự nó
> là một phát hiện (LLM áp dụng đúng quy luật hành vi của người nhưng mạnh và máy móc hơn hẳn),
> không phải lỗi chấm điểm.

| ID | Hiệu ứng | Giá trị người | Giá trị LLM | Tiêu chí chấm | Verdict |
|---|---|---|---|---|---|
| E1 | opponent_prev_unsafe | 0,607 | 1,486 | dương và p < 0,05 | **replicated** |
| E2 | progress_gap_before | −0,296 | −0,224 (p=0,47) | âm và p < 0,05 | not replicated |
| E3 | first_round_unsafe | 0,217 | 1,674 | dương và p < 0,1 | **replicated** |
| E4 | own_prev_unsafe ≈ 0 (equivalence/TOST) | −0,193 | −0,595 | TOST \|β\| < 0,3 | not replicated |
| E5 | contrast 0,6 vs 0,9 ≈ 0 (equivalence) | −0,027 | d = 1,085 | \|d\| < 0,2 | not replicated |
| E6 | contrast 0,1 vs 0,6 | d = 0,341 | d = 3,082 | dương và \|d\| > 0,2 | **replicated** (hướng đúng, độ lớn vượt xa) |
| E7 | φ_U tổng thể | 0,584 | 0,624 | trong khoảng [0,4; 0,75] | **replicated** |
| E8 | share AS | ~0,11 (near-absence) | 0,115 | share < 0,1 | not replicated (sát ngưỡng) |

**4/8 replicated** — cùng nhóm hiệu ứng "hướng đối thủ chi phối, momentum vòng 1, mức φ_U tổng
thể hợp lý" tái lập được ở LLM. **4/8 not replicated** — chủ yếu là các hiệu ứng null/equivalence
của người (own-action không dự báo được, 0,6-vs-0,9 không khác biệt) mà LLM lại cho hệ số lớn và
có ý nghĩa — tức LLM "quá nhạy" với treatment và có quán tính hành động riêng mạnh hơn người,
đúng như mục §2 và §4 đã nêu.

---

## 8. Giới hạn của báo cáo này

1. **Không phải confirmatory.** `run_phase = pilot`, manifest schema thiếu provenance đầy đủ.
   Không dùng số liệu này để kết luận về PROJECT.md RQ nào.
2. **Persona confound với batch protocol.** Mọi so sánh persona trong §3 là mô tả thô, không
   phải effect đã kiểm soát nhiễu batch (decoding, package version, source revision khác nhau
   giữa các cell).
3. **Cỡ mẫu rất lệch giữa cell.** Baseline/R0/R− có 20 người/ô; các cell S_* chỉ có 6 người/ô (3
   race). Winner/loser correlation ở §5 dựa trên 1–5 race/stratum — quá nhỏ để suy luận.
4. **Persona `R+` (risk-seeking) bị loại hoàn toàn** — run chưa hoàn tất, 0 race, không có
   `races.csv`/`players.csv`. Không có dữ liệu risk-seeking để đối chiếu với risk-averse.
5. **Chỉ 3 model cùng một họ (Gemini 3), một backend (proxy Kaggle).** Không kết luận được gì về
   khác biệt giữa các họ model.
6. **Biến trễ không ngoại sinh chặt** (như paper gốc) — hệ số logit đọc như liên hệ có điều kiện.

---

## Phụ lục — cách tái tạo

```bash
.venv-kaggle/bin/python3 results/scripts/analyze_ai_race.py \
  --input results/frontier/baseline/google-gemini-3-flash-preview \
  --input results/frontier/baseline/google-gemini-3.1-flash-lite-preview \
  --input results/frontier/baseline/google-gemini-3.5-flash-lite \
  --input results/frontier/persona/R0_neutral/google-gemini-3-flash-preview \
  --input results/frontier/persona/Rminus_risk_averse/google-gemini-3-flash-preview \
  --input results/frontier/persona/S_AA_adv_adv/google-gemini-3-flash-preview \
  --input results/frontier/persona/S_AC_adv_coop/google-gemini-3-flash-preview \
  --input results/frontier/persona/S_CA_coop_adv/google-gemini-3-flash-preview \
  --output analysis/frontier/derived \
  --fit-logit --fit-logit-robustness \
  --allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs \
  --allow-missing-persona-condition
```

`results/frontier/persona/Rplus_risk_seeking/` bị loại khỏi `--input` vì `run_manifest.json`
báo `status="running"`, 0 race, và thư mục thiếu `races.csv`/`players.csv` (analyser yêu cầu đủ
ba file `turns.jsonl` + `races.csv` + `players.csv` mỗi run dir).
