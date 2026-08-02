# Phân tích run `openai` — gpt-5-nano và gpt-5.4-nano (OpenAI API trực tiếp)

Nguồn dữ liệu: `results/frontier/openai/` (`baseline/` — không persona; `persona/` — 7 cell
persona core; `persona/risk_matrix/` — ma trận rủi ro 6×6 đầy đủ, 36 ô). Backend `"openai"`
(direct API, không qua Kaggle proxy) — xem [ai_race/models/openai_direct.py](../../ai_race/models/openai_direct.py)
và [docs/running-openai-frontier-pilots.md](../../docs/running-openai-frontier-pilots.md).
Đối chiếu: [docs/paper-analyses-inventory.md](../../docs/paper-analyses-inventory.md) — 19 phân
tích của paper gốc. Đối chiếu chéo với pilot Gemini: [analysis/frontier/report.md](../frontier/report.md).
Bảng dẫn xuất: [derived/](derived/) (35 file, sinh bởi `results/scripts/analyze_ai_race.py`).

> **Đây là PILOT, không phải confirmatory.** Mọi run có `run_phase = pilot` và manifest schema
> `ai-race-results-v1` — theo [results/README.md](../../results/README.md), schema này "cố ý
> không đủ cho phân tích chính gộp". Analyser chỉ chạy được khi bật cả bốn cờ audit:
> `--allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs
> --allow-missing-persona-condition`. Theo
> [docs/running-the-experiment.md](../../docs/running-the-experiment.md): **"Đừng dùng các flag
> đó cho kết quả thật."** Tài liệu này đọc như một audit/exploratory pass, không phải bằng chứng
> confirmatory cho PROJECT.md.

---

## 0. Phạm vi và khác biệt so với pilot Gemini

| | Pilot Gemini (`analysis/frontier/`) | Pilot OpenAI (tài liệu này) |
|---|---|---|
| Model | 3 (họ Gemini, qua proxy) | 2 (`gpt-5-nano`, `gpt-5.4-nano`, API trực tiếp) |
| Persona | 1 baseline + 5/6 cell core (1 bị loại vì chưa chạy xong) | 1 baseline + 7 cell core + **36 ô ma trận rủi ro 6×6 đầy đủ** |
| Race | 177 | **2.640** |
| Quyết định | 3.168 | **49.104** |
| Parse failure | 0 | 0 |
| `check_symmetry.py` | không nêu trong báo cáo | **PASS** — 20,0% tied-throughout (ngưỡng 40%) |

`results/scripts/analyze_ai_race.py` phủ cùng 15/19 phân tích định lượng của paper gốc như đã
liệt kê ở [analysis/frontier/report.md §0](../frontier/report.md); không lặp lại bảng đó ở đây.

---

## 1. Provenance & cổng chất lượng

| Mục | Giá trị |
|---|---|
| Backend | `openai` (API trực tiếp, `ai_race/models/openai_direct.py`), manifest `ai-race-results-v1` — không đủ điều kiện cho phân tích chính |
| `prompt_version` | `ai-race-fairgame-v3` (canonical, khớp hash) |
| Model | `gpt-5-nano`, `gpt-5.4-nano` |
| Persona | `none` · `R0` trung lập · `R-` risk-averse · `R+` risk-seeking · `S_CC`/`S_AA` (đối xứng) · `S_AC`/`S_CA` (bất đối xứng theo ghế) · **36 ô `R{i}-R{j}`, i,j = 1..6** (thang cược Eckel-Grossman, paper SI Task 1) |
| Lưới | 3 risk treatment × 44 cell persona × 10 rep × 2 model = **2.640 race** |
| Race giữ lại | **2.640/2.640** (100%), 0 bị loại vì parse/forced-stop/noncanonical mechanism |
| Quyết định | **49.104**, 5.280 player-race |
| Parse failure | **0/49.104** — nhưng `gpt-5-nano` cần tổng **125 lần retry parse** (0/49.104 vẫn là 0 sau retry; `gpt-5.4-nano` cần 0 retry). Không có race nào bị "contaminated" |
| Đẳng thức ΔS | `max_abs_identity_residual = 0,0` mọi stratum; `pearson_r = 1,0` (20/264 stratum NaN vì phương sai suy biến, không phải lỗi) |
| Số vòng trung bình | **9,3** (kỳ vọng lý thuyết 9,0 với p=0,2) |
| Cân bằng ghế | seat 0: 0,338 / seat 1: 0,363 — chênh 0,025, không có artefact vị trí đáng kể |
| Winner–loser N | trung vị **10 race quyết định thắng-thua/stratum** (so với 1–5 ở pilot Gemini) |
| `check_symmetry.py` | **PASS** — 20,0% race hoà từ đầu đến cuối (ngưỡng chặn 40%); một số ô persona **đối xứng hai ghế giống hệt nhau** (`S_CC`, và các ô đường chéo `R{i}-R{i}`) có tied-rate rất cao cục bộ — hợp lý về cơ chế, không phải lỗi (xem §3) |
| CRN block | 20 block (`source_run/model/rep`) |
| Persona × protocol | **Confound với `source_run`** — mọi hồi quy persona **không ước lượng được** (xem §5, nguyên nhân khác pilot Gemini) |

Mọi cổng cơ chế (step increment, stage payoff, prize, setback, stop rule…) đều pass — 31/31 mục
trong `analysis_manifest.json:mechanics_checks_passed`, không có mục nào fail.

### 1.1. Vì sao persona vẫn confound dù chạy trong CÙNG một session

Khác với pilot Gemini (persona confound vì mỗi cell chạy ở batch/thời điểm khác nhau), **44 config
OpenAI này chạy trong đúng một lần gọi `scripts/run_openai_stage.py`**, cùng source revision, cùng
package version. Nhưng analyser vẫn báo:

> `persona_condition varies across 44 cells (...) but never within a protocol signature`

Lý do nằm ở chính schema `ai-race-results-v1`: `analyze_ai_race.py` chỉ chấp nhận
`ai-race-kaggle-run-v1`/`ai-race-kbench-run-v1` làm bằng chứng đủ để khẳng định hai run *thật
sự* cùng protocol. Với schema local-runner, hàm `_protocol_contract_from_manifest` (khi được mở
khóa bằng `--allow-mixed-protocols`) rơi vào nhánh "unverified", và payload dùng để hash chữ ký
**gồm cả `source_run`** — tức đường dẫn thư mục output tuyệt đối, **luôn luôn khác nhau** giữa
các config vì mỗi config ghi vào một thư mục riêng. Do đó **88 protocol signature riêng biệt**
(44 config × 2 model), bất kể có chạy cùng session hay không. Đây là giới hạn cấu trúc của việc
dùng runner local (`ai_race.runner.run_experiment`) nói chung, không phải kỷ luật vận hành —
chỉ có thể khắc phục bằng cách chạy qua `kaggle_benchmarks` (schema `ai-race-kbench-run-v1`,
giàu provenance hơn).

**Điểm được giữ nguyên:** ba mức risk (0,1/0,6/0,9) nằm **trong cùng một run directory** (cùng
`source_run`), nên contrast risk trong §2 và hệ số risk trong logit ở §5 **không** bị confound
kiểu này — chỉ có so sánh **giữa các cell persona khác nhau** mới bị ảnh hưởng.

---

## 2. φ_U theo mức rủi ro — baseline không persona (tương ứng Fig. 2A)

| Model | risk = 0,1 | risk = 0,6 | risk = 0,9 | Ghi chú |
|---|---|---|---|---|
| gpt-5-nano | 0,123 | 0,151 | 0,145 | không đơn điệu, biên độ nhỏ |
| gpt-5.4-nano | 0,577 | 0,496 | 0,567 | hình chữ U, biên độ lớn hơn |

**Khác biệt căn bản với cả người và Gemini.** Ở người: φ_U dao động hẹp quanh 0,58, 0,6-vs-0,9
không có hiệu ứng. Ở Gemini: φ_U **giảm đơn điệu mạnh** theo risk (gần bão hoà 1,0 ở risk=0,1).
Ở hai model GPT-nano này: **không có xu hướng đơn điệu nào** — cả hai model đều Unsafe **ít
hơn** ở risk=0,6 so với hai đầu mút (0,1 và 0,9), hình chữ U thay vì đường dốc.

Kiểm định trên mẫu toàn vòng (`treatment_contrasts.csv`) không có cặp nào có ý nghĩa (p ≥ 0,12
cả hai model). Nhưng trên mẫu t ≥ 2 (`treatment_contrasts_round2plus.csv`, sau khi bỏ vòng 1 vốn
không có lịch sử để phản ứng), `gpt-5.4-nano` cho:

| Cặp | mean trái | mean phải | t | p | Cohen's d |
|---|---|---|---|---|---|
| 0,1 vs 0,6 | 0,611 | 0,476 | 2,093 | **0,043** | 0,662 |
| 0,6 vs 0,9 | 0,476 | 0,600 | −2,178 | **0,036** | −0,689 |
| 0,1 vs 0,9 | 0,611 | 0,600 | 0,178 | 0,860 | 0,056 |

Hình chữ U có ý nghĩa thống kê ở hai đầu (0,1↔0,6 và 0,6↔0,9) trong khi 0,1↔0,9 gần như bằng
nhau — tức risk=0,6 là một **điểm trũng cục bộ** thật, không phải nhiễu ngẫu nhiên ở một điểm.
`gpt-5-nano` không có cặp nào có ý nghĩa ở mẫu này (p ≥ 0,37).

Cả hai model **không hề bão hoà gần 1,0 ở risk thấp** như Gemini — φ_U của baseline luôn dưới
0,58 ở cả hai model, mức tổng thể gần với người hơn hẳn Gemini về **độ lớn** dù khác hẳn về
**hình dạng theo treatment**.

---

## 3. Tám cell persona core (chỉ mô tả — không suy luận nhân quả)

Vì mọi so sánh liên-cell bị confound với `source_run` (§1.1), bảng dưới chỉ mô tả chênh lệch
quan sát được:

| Persona | gpt-5-nano (0,1/0,6/0,9) | gpt-5.4-nano (0,1/0,6/0,9) |
|---|---|---|
| Baseline (`none`) | 0,123 / 0,151 / 0,145 | 0,577 / 0,496 / 0,567 |
| R0 — trung lập tường minh | **0,000 / 0,000 / 0,044** | 0,524 / 0,515 / 0,473 |
| R− — risk-averse | 0,042 / 0,076 / 0,039 | 0,121 / 0,186 / 0,115 |
| R+ — risk-seeking | 0,424 / 0,390 / 0,427 | 0,946 / 0,955 / 0,920 |
| S_CC — cả hai ghế cooperative | **0,000 / 0,000 / 0,000** | 0,018 / 0,038 / 0,009 |
| S_AA — cả hai ghế adversarial | 0,419 / 0,344 / 0,420 | 0,718 / 0,740 / 0,698 |
| S_AC — ghế adversarial (đối thủ cooperative) | 0,254 / 0,244 / 0,209 | 0,438 / 0,418 / 0,434 |
| S_CA — ghế cooperative (đối thủ adversarial) | 0,256 / 0,330 / 0,321 | 0,484 / 0,436 / 0,389 |

Ba quan sát đáng chú ý:

1. **Thứ hạng persona hợp lý và nhất quán ở cả hai model:** R− thấp nhất, R+ cao nhất,
   S_CC ≈ 0 (cả hai ghế được yêu cầu hợp tác), S_AA cao gần bằng R+. Đây là chiều hướng hợp lý
   nhất trong cả hai pilot — persona chi phối hành vi rõ ràng hơn cả sở thích rủi ro elicited ở
   người (H2.1–H2.3 null ở paper gốc).
2. **`gpt-5-nano` sụp hoàn toàn về Safe với BẤT KỲ persona nào**, kể cả R0 — placebo trung lập,
   không mang nội dung định hướng, vẫn kéo φ_U từ 0,12-0,15 (baseline) xuống 0,00-0,04. Đây là
   hiệu ứng **"có persona hay không"**, không phải hiệu ứng **hướng** của persona — chỉ riêng
   việc chèn một đoạn vai trò (dù trung lập) đã đủ thay đổi hành vi mạnh. `gpt-5.4-nano` không
   có hiệu ứng này (R0 gần baseline: 0,524/0,515/0,473 so với 0,577/0,496/0,567).
3. **Đúng như dự đoán từ `check_symmetry.py` (§1):** `S_CC` trên `gpt-5-nano` là 0,000 tuyệt đối
   ở cả ba risk — khớp với quan sát tied-throughout 100% cho cell này ở model đó.

`persona_contrasts.csv` trống (0 dòng) — analyser không sinh được so sánh cặp trực tiếp vì
không có hai persona nào chung protocol signature, đúng như phân tích ở §1.1.

---

## 4. Ma trận rủi ro 6×6 — persona theo thang Eckel-Grossman (MỚI so với pilot Gemini)

Đây là trục dữ liệu Gemini **chưa có**: mỗi ghế được gán độc lập một mức "đã từng chọn mức cược
thứ *i* trên thang sáu mức rủi ro tăng dần" (1 = né rủi ro nhất, 6 = ưa rủi ro nhất), đủ 36 tổ
hợp (i, j). φ_U trung bình mỗi player (gộp cả 3 risk treatment, `player_metrics.csv` phân theo
`player_id` để tách ghế "mình" khỏi ghế "đối thủ" qua `personaRoles` trong config agent):

**gpt-5-nano** — hàng = mức của MÌNH, cột = mức của ĐỐI THỦ:

| own\opp | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1** | 0,010 | 0,004 | 0,012 | 0,025 | 0,022 | 0,035 |
| **2** | 0,001 | 0,005 | 0,007 | 0,023 | 0,017 | 0,019 |
| **3** | 0,101 | 0,128 | 0,134 | 0,215 | 0,188 | 0,214 |
| **4** | 0,534 | 0,520 | 0,471 | 0,425 | 0,413 | 0,385 |
| **5** | 0,602 | 0,560 | 0,508 | 0,441 | 0,476 | 0,440 |
| **6** | 0,599 | 0,576 | 0,523 | 0,477 | 0,487 | 0,478 |

**gpt-5.4-nano:**

| own\opp | 1 | 2 | 3 | 4 | 5 | 6 |
|---|---|---|---|---|---|---|
| **1** | 0,002 | 0,004 | 0,002 | 0,000 | 0,008 | 0,006 |
| **2** | 0,000 | 0,000 | 0,003 | 0,006 | 0,003 | 0,002 |
| **3** | 0,479 | 0,463 | 0,385 | 0,300 | 0,252 | 0,250 |
| **4** | 0,628 | 0,581 | 0,506 | 0,446 | 0,390 | 0,345 |
| **5** | 0,833 | 0,869 | 0,834 | 0,796 | 0,729 | 0,702 |
| **6** | 0,982 | 0,982 | 0,983 | 0,991 | 0,985 | 0,988 |

Hai phát hiện, cả hai đều khớp giữa hai model độc lập:

1. **Mức rủi ro CỦA MÌNH dự báo hành vi gần như đơn điệu và rất mạnh** — từ ≈0 (mức 1-2) lên
   0,4-0,6 (`gpt-5-nano`) hoặc gần 1,0 (`gpt-5.4-nano`, mức 6). Đây là **dose-response** sạch
   hơn hẳn so với người: paper gốc dùng đúng thang Eckel-Grossman này để elicit risk preference
   và tìm thấy **không có** quan hệ dự báo nào với hành vi thật (H2.1–H2.3 không được ủng hộ).
   Ở LLM, "đã từng chọn mức cược cao" trong persona text chuyển thành hành vi Unsafe gần như
   quyết định luận, đặc biệt rõ ở `gpt-5.4-nano` (mức 6 → 0,98-0,99 bất kể đối thủ).
2. **Mức rủi ro CỦA ĐỐI THỦ có quan hệ NGƯỢC, ổn định nhưng nhỏ hơn nhiều:** trong mọi hàng
   (own_level cố định), φ_U **giảm dần** khi opp_level tăng — ví dụ `gpt-5-nano` hàng own=4:
   0,534 → 0,385 khi đối thủ đi từ mức 1 lên mức 6. Đối thủ càng "ưa rủi ro" (theo persona),
   mình càng thận trọng hơn — chiều **bù trừ**, không phải leo thang. Phát hiện này **khớp
   chính xác** với dấu âm của `opponent_prev_unsafe` trong hồi quy logit ở §5 — hai phân tích
   độc lập (mô tả theo persona tĩnh vs. hồi quy theo hành động động) cùng chỉ ra một cơ chế.

Chi tiết đầy đủ 36×3 risk × 2 model = 216 giá trị nằm trong `unsafe_by_risk_model_player.csv`
và `player_metrics.csv`; bảng trên gộp theo risk để dễ đọc.

---

## 5. Hồi quy logistic panel cluster-robust (tương ứng Table 1)

Đặc tả (6) — bão hoà nhất, kiểm soát `C(protocol_signature)` (88 mức) để tách nhiễu batch:

```
unsafe ~ C(risk) + first_round_unsafe + own(t-1) × opponent(t-1) × ΔS(t-1) + C(protocol_signature)
```

Cluster-robust SE theo `source_run::model::rep` (20 block), N = 43.824 (mẫu t ≥ 2).

> **CẢNH BÁO BẮT BUỘC ĐỌC TRƯỚC BẢNG:** statsmodels báo **`converged: False`** cho **cả 6 đặc
> tả**, không riêng đặc tả 6. Với 88 dummy `protocol_signature` cộng thêm một số cell hoàn toàn
> tách biệt (ví dụ `S_CC` trên `gpt-5-nano` = 0,000 tuyệt đối ở mọi risk — dummy đó dự báo
> Y=0 hoàn hảo), đây là dấu hiệu kinh điển của **quasi-complete separation**, khiến MLE không
> có nghiệm hữu hạn ổn định. `logit_robustness_jackknife.csv` — vốn cần refit hội tụ theo từng
> block — ra **0 dòng**, tức bước robustness đã tự bỏ cuộc vì lý do tương tự. Các hệ số dưới đây
> vẫn được statsmodels trả về (giá trị dừng ở vòng lặp cuối), nhưng **không nên coi là ước lượng
> MLE ổn định** — đọc như tín hiệu định hướng thô, mức độ không chắc chắn thấp hơn cả mức "pilot,
> chưa preregister" đã nêu ở đầu tài liệu.

| Biến | β̂ | SE | p |
|---|---|---|---|
| risk = 0,6 (so với 0,1) | −0,010 | 0,038 | 0,782 |
| risk = 0,9 (so với 0,1) | **−0,126** | 0,051 | **0,013** |
| Unsafe vòng 1 | **+0,635** | 0,070 | < 0,001 |
| Own(t−1) = Unsafe | +0,170 | 0,126 | 0,176 |
| **Opponent(t−1) = Unsafe** | **−1,016** | 0,051 | **< 0,001** |
| Own(t−1) × Opp(t−1) | −0,655 | 0,087 | < 0,001 |
| ΔS(t−1) (vị thế đua) | **+0,490** | 0,047 | < 0,001 |
| Own(t−1) × ΔS(t−1) | −0,255 | 0,039 | < 0,001 |
| Opp(t−1) × ΔS(t−1) | +0,174 | 0,065 | 0,008 |
| Tương tác 3 chiều | +0,259 | 0,088 | 0,003 |

Pseudo R² (đặc tả 6) = 0,347.

**Hai đảo dấu hoàn toàn so với cả người và Gemini — phát hiện nổi bật nhất của pilot này:**

1. **`opponent_prev_unsafe` ÂM** (−1,016, p<0,001), trong khi người: +0,607, Gemini: +1,486.
   Ở người và Gemini, đối thủ chơi Unsafe vòng trước làm **tăng** xác suất mình cũng chơi Unsafe
   (leo thang/trả đũa — phát hiện trung tâm của paper gốc). Ở hai model GPT-nano này, đối thủ
   Unsafe làm mình **giảm** xác suất Unsafe — phòng thủ/bù trừ thay vì leo thang. Khớp trực
   tiếp với phát hiện độc lập ở §4 (mức rủi ro đối thủ càng cao, mình càng thận trọng).
2. **`progress_gap_before` (ΔS) DƯƠNG** (+0,490, p<0,001), trong khi người: −0,296 (dẫn trước
   → thận trọng hơn), Gemini: −0,224 (cùng dấu người, không ý nghĩa). Ở đây: **dẫn trước làm
   tăng** xác suất Unsafe — củng cố lợi thế thay vì bảo toàn nó.

**Một điểm khớp hướng với cả người và Gemini:** hành động vòng 1 vẫn dự báo dương hành vi về
sau (+0,635, p<0,001) — "behavioural momentum" tái lập ở cả ba nguồn dữ liệu (người, Gemini,
GPT), dấu hiệu ổn định nhất qua mọi pilot.

**Lưu ý nội sinh:** giống mọi pilot khác, biến trễ không ngoại sinh chặt → đọc như liên hệ có
điều kiện. Ở đây rủi ro đọc-quá còn cao hơn vì thêm vấn đề hội tụ nêu trên.

---

## 6. Tương quan φ_U thắng–thua trong cặp (tương ứng Fig. 2C)

`winner_loser_correlation.csv`, 243 stratum (model × persona × risk), trung vị **10 race quyết
định thắng-thua/stratum** (so với 1–5 ở pilot Gemini — mẫu này khá hơn hẳn). Chênh lệch trung
bình người thắng trừ người thua (`mean_winner_minus_loser`) = **0,438** trên toàn bộ stratum
(khoảng 0,062–1,0), cùng hướng với người và Gemini: người thắng có φ_U cao hơn người thua.

---

## 7. Phân loại chiến lược gần nhất (baseline, không persona)

`strategy_summary_player.csv`:

- **`gpt-5-nano`**: gần như thuần **AS** (Always Safe) — 13-16/20 player mỗi risk (65-80%),
  phần còn lại rơi vào biên `AS|CS` (tie giữa AS và CS, không phân biệt được vì φ_U quá thấp).
  Không quan sát AU/CAS đáng kể nào.
- **`gpt-5.4-nano`**: phân bố đa dạng hơn hẳn — AS, AU, CS, **và CAS** đều xuất hiện với tỷ
  trọng đáng kể (ví dụ risk=0,6: AS 6, AU 5, CAS 4, CS 1, cộng vài cell biên) — khác hẳn
  Gemini, nơi **CS không xuất hiện ở bất kỳ risk nào** (0% mọi treatment).

**Đối chiếu với lý thuyết** (`theory_equilibria.csv`, đã có sẵn trong repo): paper gốc dự đoán
CS là cân bằng Nash duy nhất ở risk=0,9. `gpt-5.4-nano` là model **đầu tiên trong cả hai pilot**
dùng CS ở tỷ trọng khác 0 (baseline risk=0,9: 4/20 player) — dù chưa áp đảo. `gpt-5-nano` giống
Gemini ở điểm CS vắng mặt hoàn toàn.

---

## 8. Đối chiếu 8 hiệu ứng người–LLM (E1–E8)

Từ `human_comparison.csv`, chấm tự động theo `results/scripts/human_reference.json`, **gộp
toàn bộ 44 cell × 2 model** (không tách riêng theo model — đọc con số này như một trung bình
rất thô trên cả tập dữ liệu không đồng nhất).

| ID | Hiệu ứng | Giá trị người | Giá trị LLM | Tiêu chí chấm | Verdict |
|---|---|---|---|---|---|
| E1 | opponent_prev_unsafe | 0,607 | **−1,016** | dương và p < 0,05 | not replicated (đảo dấu) |
| E2 | progress_gap_before | −0,296 | **+0,490** | âm và p < 0,05 | not replicated (đảo dấu) |
| E3 | first_round_unsafe | 0,217 | 0,635 | dương và p < 0,1 | **replicated** |
| E4 | own_prev_unsafe ≈ 0 (TOST) | −0,193 | 0,170 | \|β\| < 0,3 | not replicated |
| E5 | contrast 0,6 vs 0,9 ≈ 0 | −0,027 | d = 0,031 | \|d\| < 0,2 | **replicated** |
| E6 | contrast 0,1 vs 0,6 | d = 0,341 | d = −0,003 | dương và \|d\| > 0,2 | not replicated (gần 0) |
| E7 | φ_U tổng thể | 0,584 | 0,353 | trong [0,4; 0,75] | not replicated (thấp hơn khoảng) |
| E8 | share AS | ~0,11 | **0,562** | share < 0,1 | not replicated (cao hơn nhiều) |

**Chỉ 2/8 replicated** — thấp hơn hẳn Gemini (4/8). Nhưng bức tranh định tính khác hẳn, không
chỉ là "kém hơn": E1/E2 **đảo dấu hoàn toàn** (không phải chỉ sai độ lớn như ở Gemini), và E8
lệch theo hướng ngược — Gemini gần như không dùng AS, GPT-nano dùng AS **áp đảo** (56%, chủ yếu
đến từ các cell persona low-risk và `gpt-5-nano` baseline). **Đọc E5/E6/E7 cẩn thận:** các con
số này gộp cả 44 cell rất khác nhau (một số cell φ_U ≈ 0, một số ≈ 1), nên "replicated" ở E5 rất
có thể là trung bình hoá hai xu hướng ngược nhau chứ không phải một hiệu ứng thật bằng 0 — xem
lại contrast theo *từng model, chỉ baseline* ở §2 để so sánh mức độ tin cậy hơn.

---

## 9. Giới hạn của báo cáo này

1. **Không phải confirmatory.** `run_phase = pilot`, manifest schema thiếu provenance đầy đủ.
   Không dùng số liệu này để kết luận về PROJECT.md RQ nào.
2. **Persona confound với `source_run`, không do kỷ luật vận hành mà do giới hạn cấu trúc của
   schema `ai-race-results-v1`** (xem §1.1) — kể cả khi mọi cell chạy chung một session, hồi
   quy persona vẫn không ước lượng được qua đường này.
3. **Hồi quy logit ở §5 không hội tụ** (cả 6 đặc tả). Đọc hệ số như tín hiệu định hướng, không
   phải ước lượng MLE ổn định. `--fit-logit-robustness` không sinh được output vì cùng nguyên
   nhân.
4. **Bảng E1-E8 (§8) gộp 44 cell rất không đồng nhất** — một số giá trị "replicated"/"not
   replicated" có thể là trung bình hoá của các xu hướng đối lập giữa các cell, không phải một
   hiệu ứng đơn nhất.
5. **Chỉ 2 model cùng họ (`gpt-5-nano`, `gpt-5.4-nano`), một backend (API OpenAI trực tiếp).**
   Không kết luận được gì về khác biệt giữa các họ model (GPT vs Gemini vs người) vượt ra ngoài
   so sánh mô tả.
6. **`gpt-5-nano` cần cơ chế `reasoning_effort` đặc thù** (xem
   [ai_race/models/openai_direct.py](../../ai_race/models/openai_direct.py)) để tránh response
   rỗng do đốt hết ngân sách token vào reasoning ẩn — dữ liệu cuối sạch (0 parse_failed) nhưng
   cơ chế sinh sinh ra nó khác `gpt-5.4-nano` (temperature effective cũng khác: `gpt-5-nano`
   dùng default 1,0 của model vì từ chối giá trị 0,7 tường minh, `gpt-5.4-nano` dùng đúng 0,7).
   Hai model không hoàn toàn "cùng điều kiện decoding" dù chạy qua cùng backend.
7. **Biến trễ không ngoại sinh chặt** (như mọi pilot khác) — hệ số logit đọc như liên hệ có
   điều kiện, không phải hiệu ứng nhân quả.

---

## Phụ lục — cách tái tạo

```bash
.venv-kaggle/bin/python3 results/scripts/analyze_ai_race.py \
  --input results/frontier/openai \
  --output analysis/openai/derived \
  --fit-logit --fit-logit-robustness \
  --allow-mixed-protocols --allow-nonfinal-runs --allow-nonconfirmatory-runs \
  --allow-missing-persona-condition
```

Không cần liệt kê từng thư mục con như pilot Gemini — cả 88 run directory (44 config × 2 model)
đều `status="completed"`, nên `--input results/frontier/openai` tự dò đủ qua
`_discover_run_directories`.

Soi gương (bắt buộc trước khi tin số liệu race-position):

```bash
.venv-kaggle/bin/python results/scripts/check_symmetry.py --input results/frontier/openai
```
