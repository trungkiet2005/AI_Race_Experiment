# Phân tích run `baseline` — qwen2.5-14b-instruct và gemma-3-12b-it

Nguồn: `results/ai_race_results.zip` (Kaggle, 2026-07-31, RTX PRO 6000).
Notebook: [kaggle/experiments/baseline.py](../../kaggle/experiments/baseline.py).
Đối chiếu: [docs/paper-analyses-inventory.md](../../docs/paper-analyses-inventory.md) — 19 phân tích của paper gốc.

> **Đây là PILOT.** `run_phase = pilot`, 10 repetition, 10 CRN block. Không một con số nào
> trong tài liệu này là bằng chứng confirmatory. Theo [PROJECT.md](../../PROJECT.md), primary
> analysis chỉ nhận một phase `confirmatory` duy nhất với manifest `status="completed"`.
> Mọi p-value dưới đây đọc như tín hiệu định hướng, không phải kiểm định.

---

## 0. Provenance

| Mục | Giá trị |
|---|---|
| `source_sha256` | `ef1eb67ef51d010cd2018c0b9ac3d149bc9f61b4fce002119029e4adcda93cee` |
| `prompt_version` | `ai-race-fairgame-v3` |
| `prompt_sha256` | `27086bd8…2df3e` — khớp canonical |
| `persona_condition` | `none` (ghế trung tính, `companies_default`) |
| Backend | `transformers` 5.0.0, bf16, `device_map="auto"` |
| `vllm` | `null` — xác nhận không dùng vLLM |
| torch / numpy / pandas | 2.10.0+cu128 / 2.0.2 / 2.3.3 |
| Repo input | `/kaggle/input/datasets/nguyenlamphuquy/ai-race-experiment` |
| Decoding | temperature 0,7 · max_tokens 256 · logprobs tắt |
| `historyMode` | **`previous_round`** — xem cảnh báo bên dưới |
| Lưới | 3 treatment × 1 ngôn ngữ × 10 rep × 2 model = 60 race |

> **Run này chạy dưới `historyMode: "previous_round"`.** Agent chỉ thấy vòng `t−1`, nên
> **không thể** nhìn lại nước đi vòng 1 của chính mình sau khi race đã đi tiếp. Các config
> đã đổi sang `first_and_previous` **sau** run này, nên `protocol_signature` khác nhau và
> analyser sẽ từ chối gộp run này với run tương lai ở primary mode — đúng như thiết kế.
> Chi tiết ở mục 5.1.

Cả hai model đều `status: completed`, 30 race, 558 quyết định.

---

## 1. Cổng chất lượng protocol

| Gate | Kết quả |
|---|---|
| Parse | **0 / 1.116 thất bại**, 0 retry, `attempt_history` dài đúng 1 |
| Prompt hash | canonical v3 cả hai model |
| Race bị loại | **0 / 60** |
| Manifest đối chiếu turns | khớp |
| Đẳng thức ΔS | `max_abs_identity_residual = 0,0`, `pearson_r = 1,0` |
| Seed CRN | 186 seed phân biệt / 558 = đúng 558/3 treatment |
| Đối xứng ghế (Qwen) | seat 0: 0,215 / seat 1: 0,247 — chênh 0,03, không có artefact vị trí |
| **`check_symmetry.py`** | **FAIL** — 65,0% race hoà từ đầu đến cuối, ngưỡng 40% |

Đường ống chạy đúng. Cổng duy nhất thất bại là symmetry, và nguyên nhân nằm hoàn toàn ở
Gemma (mục 3).

### Kiểm chứng CRN

Phân bố độ dài race (60 race, gộp 2 model): `{5:12, 7:6, 8:6, 9:12, 10:6, 11:6, 13:6, 16:6}`.

Mọi tần suất đều là **bội số của 6** = 2 model × 3 treatment. Đúng như thiết kế: `game_seed =
base_seed + rep` độc lập với tên treatment, nên 6 race cùng `rep` dùng chung một lần bốc
horizon. 10 rep → 10 horizon phân biệt. Đây là bằng chứng số cho thấy cấu trúc common random
number còn nguyên vẹn.

Trung bình **9,30 vòng** so với kỳ vọng lý thuyết **9,0** (paper gốc: 9,56 trên 173 race).

---

## 2. Thống kê mô tả mẫu (mục #10)

| Model | `p_max` | n người | n race | n quyết định | t≥2 | mean φ_U | median φ_U | SD | vòng TB |
|---|---|---|---|---|---|---|---|---|---|
| gemma-3-12b-it | 0,1 | 20 | 10 | 186 | 166 | **0,000** | 0,000 | 0,000 | 9,3 |
| gemma-3-12b-it | 0,6 | 20 | 10 | 186 | 166 | **0,000** | 0,000 | 0,000 | 9,3 |
| gemma-3-12b-it | 0,9 | 20 | 10 | 186 | 166 | **0,000** | 0,000 | 0,000 | 9,3 |
| qwen2.5-14b | 0,1 | 20 | 10 | 186 | 166 | 0,239 | 0,222 | 0,134 | 9,3 |
| qwen2.5-14b | 0,6 | 20 | 10 | 186 | 166 | 0,173 | 0,222 | 0,127 | 9,3 |
| qwen2.5-14b | 0,9 | 20 | 10 | 186 | 166 | 0,196 | 0,236 | 0,125 | 9,3 |

φ_U tổng của Qwen: **0,203** (trọng số người chơi) / **0,231** (trọng số quyết định).
Paper gốc trên người: **0,584**.

> **Cảnh báo đọc số.** `human_comparison.csv` báo E7 φ_U = 0,101. Đó là trung bình gộp
> 0,203 của Qwen với 0,000 của Gemma — **không mô tả model nào cả**. Đừng trích con số đó.

---

## 3. Gemma-3-12B-it: dữ liệu không dùng được

### Triệu chứng

**558/558 lượt gọi trả về đúng cùng một chuỗi ký tự: `'ACTION: SAFE'`.** Reasoning rỗng.
Unsafe = 0,000 ở cả ba mức rủi ro.

Với temperature 0,7 và **186 seed phân biệt**, việc ra cùng một token mọi lần nghĩa là
`P(SAFE) ≈ 1,0` tại mọi điểm quyết định.

### Hệ quả dây chuyền

Vì Gemma không bao giờ lệch quỹ đạo, trạng thái race trở nên tất định và không gian prompt
sụp đổ còn **96 prompt phân biệt** (Qwen: 180) = đúng `3 treatment × 16 vòng × 2 ghế`.
Model tự bóp mẫu của chính nó.

### Đường ống đã được loại trừ

| Khả năng | Kiểm chứng | Kết luận |
|---|---|---|
| Parse hỏng, fallback Safe che giấu | `parse_failed=0`, `retry=0`, `attempt_history=1` | loại |
| Seed không được áp | `sampling_seed_applied=True`, 186 seed phân biệt | loại |
| Prompt render sai | 2.143–2.248 ký tự, hoán vị hai ghế đúng | loại |
| Không sampling | temperature 0,7 > 0 → `do_sample=True`; `torch.manual_seed` mỗi lượt | loại |
| Model nạp lỗi | `status: completed`, 30 race, 558 lượt — khớp Qwen | loại |

Model **thật sự** nhận prompt hợp lệ, **thật sự** đang sampling, và vẫn ra cùng một token.

### Hai giả thuyết chưa phân biệt được

- **(A)** Gemma hiểu trò chơi và kết luận Safe là tối ưu.
- **(B)** Gemma từ chối token `UNSAFE` do safety tuning, chưa từng cân nhắc payoff.

Nghi ngờ nghiêng về (B): nhãn hành động **chính là chữ** `SAFE` / `UNSAFE`
([prompt.py:141-142](../../ai_race/engine/prompt.py#L141-L142)). Một model safety-tune nặng có
prior rất mạnh trên hai từ đó, độc lập với cấu trúc payoff. Nhưng dữ liệu hiện có **không
phân biệt được** — cả (A) và (B) đều sinh ra đúng bộ số này.

Bằng chứng gián tiếp: Qwen dùng **cùng prompt, cùng nhãn** vẫn chọn UNSAFE 23% và chọn có
cấu trúc. Nên prompt tự nó không chặn UNSAFE.

### Phép thử quyết định

1. **Đổi nhãn hành động** thành `OPTION_A` / `OPTION_B`. Gemma bắt đầu chọn B → là (B).
   Vẫn 100% A → là (A). Rẻ và dứt khoát. *Lưu ý: đổi template ⇒ phải cấp `promptVersion`
   mới, và run mới không pool chung với v3.*
2. **Bật logprobs** để biết `P(UNSAFE)` là 1e-3 hay 1e-12 — rất khác nhau về diễn giải.
   Cần chuyển sang backend vLLM; transformers không hỗ trợ.
3. **Cell persona `risk_seeking` / `adv_adv`** — nếu vai diễn hung hăng cũng không lay
   chuyển được thì prior về từ ngữ mạnh hơn cả hướng dẫn.

### Tác động lên phần còn lại

- Mọi estimand hành vi của Gemma thoái hoá; ba treatment cho kết quả giống hệt.
- **Logit gộp hai model crash** (`LinAlgError: Singular matrix`) vì Gemma tách hoàn toàn.
- Phân loại chiến lược vô nghĩa: cả 60 người chơi ra `AS|CS` với mismatch **0,000** — quỹ
  đạo toàn-Safe khớp *cả* AS *lẫn* CS hoàn hảo, không phân biệt được.
- `check_symmetry.py` FAIL: Gemma 100% race hoà, kéo tỉ lệ chung lên 65%.

**Kết luận: Gemma không đóng góp thông tin nào cho đợt này.** Nhưng đừng loại nó khỏi bài
trong im lặng — "một model từ chối tuyệt đối hành động mang nhãn UNSAFE ở mọi mức rủi ro"
là một kết quả, chỉ là chưa diễn giải được cho tới khi chạy phép thử đổi nhãn.

---

## 4. Qwen2.5-14B: phát hiện trung tâm

### 4.1. Chuyển pha ở đúng vòng 5

```
vòng:    1     2     3     4  |  5      6      7      8      9     10
unsafe: 0,00  0,00  0,00  0,00 | 0,283  0,521  0,354  0,690  0,194  0,583
```

Safe **tuyệt đối** ở vòng 1–4 trong cả 30 race, bật Unsafe từ vòng 5.

Vòng 5 là `minRounds` — vòng đầu tiên race **có thể** kết thúc. Prompt nói thẳng:

> *"The race lasts at least 5 rounds. After every completed round from round 5 onward, the
> race ends with probability 20%."*

Qwen đang đọc luật dừng và chơi endgame. Đây là chiến lược **có lý**: rủi ro hiệu dụng là
`p_max × n_U/W`, nên chỉ đánh cược khi lợi thế tiến độ có khả năng quyết định thắng thua,
thay vì gánh tỉ lệ đó suốt một race có thể còn rất dài.

Khối vòng 2–4 chiếm **180/498 = 36%** mẫu panel và tất định hoàn toàn.

### 4.2. Bị dẫn trước làm tăng Unsafe — mạnh (mục #4)

Đây là phát hiện trung tâm của paper gốc, và nó tái tạo rất rõ:

| `p_max` | dẫn trước (+0,5) | hoà (0,0) | bị dẫn (−0,5) |
|---|---|---|---|
| 0,1 | 0,385 | 0,157 | **0,769** |
| 0,6 | 0,125 | 0,123 | **0,938** |
| 0,9 | 0,300 | 0,137 | **0,800** |

Chênh lệch bị-dẫn trừ dẫn-trước: **+0,38 / +0,81 / +0,50**. Nhất quán và lớn ở cả ba
treatment. `|ΔS|` chưa bao giờ vượt 0,5 trong run này.

### 4.3. Ma trận chuyển trạng thái — luân phiên, không phải quán tính

`P(Unsafe_t | own_{t−1}, opp_{t−1})`, chỉ vòng ≥5 (bỏ khối tất định):

| own_prev \ opp_prev | Safe | Unsafe |
|---|---|---|
| **Safe** | 0,404 (n=146) | **0,892** (n=65) |
| **Unsafe** | 0,154 (n=65) | **0,048** (n=42) |

Đối thủ vừa Unsafe và mình vừa Safe → 89% mình Unsafe. Nhưng mình vừa Unsafe xong → gần như
chắc chắn lùi về Safe (4,8%). Đây là **chia lượt**, không phải momentum.

### 4.4. Hồi quy panel (mục #6)

**Chỉ đặc tả 1–2 ước lượng được.** Lý do ở mục 5.

Đặc tả 2, N=498, 10 CRN block, pseudo R² = 0,194:

| Biến | β | SE | p |
|---|---|---|---|
| `p_max = 0,6` | −0,642 | 0,257 | 0,013 |
| `p_max = 0,9` | −0,320 | 0,151 | 0,034 |
| `own_prev_unsafe` | **−2,234** | 0,359 | <0,001 |
| `opponent_prev_unsafe` | **+2,339** | 0,493 | <0,001 |
| `progress_gap_before` | −0,717 | 0,290 | 0,013 |

Giới hạn chỉ vòng ≥5 (N=318, loại khối tất định), hiệu ứng **vẫn còn**:
`own_prev` −2,745 (p<0,001) · `opponent_prev` +1,339 (p=0,011) · `gap` −0,664 (p=0,011).
Vậy các liên hệ lag không phải artefact cơ học của khối vòng 2–4.

> **10 cluster.** Cluster-robust SE với 10 cluster là lạc quan. Đọc p-value như tín hiệu.

### 4.5. Thắng–thua (mục #5)

| `p_max` | race phân định | φ_U người thắng | φ_U người thua | chênh |
|---|---|---|---|---|
| 0,1 | 5 | 0,340 | 0,214 | +0,126 |
| 0,6 | 2 | 0,343 | 0,221 | +0,121 |
| 0,9 | 2 | 0,380 | 0,279 | +0,101 |

Hướng khớp paper (người thắng Unsafe nhiều hơn). Nhưng **chỉ 9/30 race phân định được** —
21 race hoà. Cột `pearson_r` mà analyser xuất ra là 0,996–1,000; với n=2 thì tương quan
bằng 1 là tất yếu hình học, **không có nội dung**. Bỏ qua cột đó ở cỡ mẫu này.

### 4.6. Effect size treatment (mục #1 và #3)

| Cửa sổ | so sánh | mean L | mean R | t | Cohen's d | p (Bonf.) |
|---|---|---|---|---|---|---|
| Toàn vòng | 0,1 vs 0,6 | 0,239 | 0,173 | 1,617 | 0,512 | 0,342 |
| Toàn vòng | 0,1 vs 0,9 | 0,239 | 0,196 | 1,065 | 0,337 | 0,881 |
| Toàn vòng | 0,6 vs 0,9 | 0,173 | 0,196 | −0,578 | −0,183 | 1,000 |
| **t ≥ 2** | 0,1 vs 0,6 | 0,267 | 0,193 | 1,644 | **0,520** | 0,325 |
| **t ≥ 2** | 0,1 vs 0,9 | 0,267 | 0,217 | 1,108 | 0,351 | 0,824 |
| **t ≥ 2** | 0,6 vs 0,9 | 0,193 | 0,217 | −0,564 | **−0,178** | 1,000 |

So với người: d(0,1 vs 0,6) = 0,341 → Qwen 0,520. d(0,6 vs 0,9) = −0,027 → Qwen −0,178.

Hướng khớp, độ lớn hơn, nhưng **không có ý nghĩa thống kê sau Bonferroni** ở n=20/nhóm.

---

## 5. Cái KHÔNG ước lượng được, và tại sao

### 5.1. `first_round_unsafe` là hằng số → đặc tả 4–6 singular

**0/60 người chơi mở đầu Unsafe** ở *cả hai* model. Biến trùng khít với intercept.

Ma trận thiết kế đặc tả 6: **11 cột, rank 10** — thiếu đúng 1 bậc tự do. Đặc tả 3 (không có
`first_round_unsafe`): 10 cột, rank 10, đủ hạng.

Hệ quả: **đặc tả 6 — đặc tả chính của Table 1 — không chạy được**, và **mục #E3
(behavioural momentum vòng 1) không thể ước lượng** ở run này.

Có một lý do cấu trúc đứng sau, không chỉ là chuyện mẫu nhỏ. Dưới
`historyMode: "previous_round"`, agent **không nhìn thấy** hành động vòng 1 của chính mình
một khi race đã đi qua vòng 2. Nhưng CS và CAS **chỉ khác nhau đúng ở nước đi đầu**. Nghĩa
là phân biệt CS/CAS là thứ agent không thể thực hiện dù muốn — trong khi phân tích lại
điều kiện hoá trên chính biến đó. Đã thêm `historyMode: "first_and_previous"` (mang vòng 1
đi kèm vòng `t−1`) và bật cho ba game config sau run này; xem mục 0.

### 5.2. Đặc tả 3 không hội tụ

Hệ số −48,3 và −25,4 với **SE = NaN**, `ConvergenceWarning`. Tách hoàn toàn ở ô tương tác
ba chiều. Không báo cáo được.

### 5.3. Bảng jackknife rỗng

`--fit-logit-robustness` dựa trên `LOGIT_FORMULA` = đặc tả 6, vốn singular ở dữ liệu này.
Mọi lần refit trả `None` và bị ghi vào `skipped_blocks`.

### 5.4. Persona không ước lượng được

Chỉ có một `persona_condition` (`none`). Không có contrast nào để chạy.

---

## 6. Đối chiếu lý thuyết (mục #11, #13, #15, #16, #17)

### 6.1. Cấu trúc trò chơi — không phụ thuộc model

| `p_max` | stage game | ngưỡng p* | vượt ngưỡng? | Nash đối xứng | AS là Nash? | vùng AS/AU |
|---|---|---|---|---|---|---|
| 0,1 | deadlock `T>P>R>S` | 0,1324 | không | AU \| CAS | không | chỉ AU |
| 0,6 | deadlock `T>P>R>S` | 0,1324 | có | AU \| CAS | không | song ổn định |
| 0,9 | deadlock `T>P>R>S` | 0,1324 | có | **CS** | không | song ổn định |

Ma trận payoff kỳ vọng tại `p_max = 0,1` (hàng = mình):

| | AS | AU | CAS | CS |
|---|---|---|---|---|
| **AS** | 59,00 | 5,40 | 8,60 | 59,00 |
| **AU** | 109,44 | 61,20 | 61,20 | 106,56 |
| **CAS** | 108,96 | 61,20 | 61,20 | 86,75 |
| **CS** | 59,00 | 16,60 | 33,79 | 59,00 |

### 6.2. Lý thuyết dự báo Unsafe, thực tế cho Safe

| `p_max` | φ_U dự báo (β=2) | φ_U dự báo (β=0,01) | Qwen quan sát (median) | chênh |
|---|---|---|---|---|
| 0,1 | 1,000 | 1,000 | 0,222 | **−0,778** |
| 0,6 | 1,000 | 0,931 | 0,222 | **−0,708** |
| 0,9 | 0,010 | 0,064 | 0,236 | +0,226 |

> **`predicted_phi_U` không phụ thuộc model LLM.** Nó giống hệt nhau cho mọi model và mọi
> cell persona. Cột `difference` là *khoảng cách của LLM so với lý thuyết trò chơi*, không
> phải một fit. "Khớp" không phải bằng chứng về model.

### 6.3. Phân loại chiến lược quan sát so với phân phối dừng

Phân phối dừng **lý thuyết** (Z=100, β=2, giới hạn µ→0):

| `p_max` | AS | AU | CAS | CS |
|---|---|---|---|---|
| 0,1 | 0,000 | 0,500 | 0,500 | 0,000 |
| 0,6 | 0,000 | 0,500 | 0,500 | 0,000 |
| 0,9 | 0,010 | 0,000 | 0,010 | **0,981** |

Phân loại **quan sát** của Qwen (60 người chơi, gộp treatment):
`CS` 25 · `AS|CS` 23 · `AS` 12 · **`AU` 0 · `CAS` 0**.

Không một người chơi nào gần AU hay CAS. Lý thuyết dự báo hai chiến lược đó chiếm ~100%
khối lượng ở `p_max` 0,1 và 0,6. Đây là bất đồng lớn nhất giữa hai nửa dự án.

> **`theory_stationary_distribution.csv` ≠ `strategy_summary_player.csv`.** Cái thứ nhất là
> phân phối dừng **dự báo** của mô hình tiến hoá; cái thứ hai phân loại quỹ đạo LLM **quan
> sát được**. Hai câu hỏi khác nhau, không phải hai ước lượng của một đại lượng.

---

## 7. Bảng đối chiếu 19 mục của paper

| # | Phân tích | Trạng thái | Ghi chú |
|---|---|---|---|
| 1 | t-test treatment (toàn vòng) | ✅ | mục 4.6; không ý nghĩa sau Bonferroni |
| 2 | Mixed-effects prereg (risk preference) | ⛔ N/A | LLM không có sở thích rủi ro; TA.4 chưa cài |
| 3 | Effect size trên mẫu t≥2 | ✅ | mục 4.6; d = 0,520 / 0,351 / −0,178 |
| 4 | φ_U theo ΔS × lag | ✅ | mục 4.2–4.3; **tái tạo mạnh** |
| 5 | Tương quan thắng–thua | ⚠️ | mục 4.5; chỉ 9 race phân định, `pearson_r` vô nghĩa |
| 6 | **Logit panel 6 đặc tả** | ⚠️ **1–2 / 6** | đặc tả 3–6 singular hoặc không hội tụ |
| 7 | Covariate nhân khẩu | ⛔ N/A | không định nghĩa cho LLM |
| 8 | Robustness jackknife | ⛔ | phụ thuộc đặc tả 6, vốn singular |
| 9 | Phân phối số vòng | ✅ | mục 1; TB 9,30 vs lý thuyết 9,0 |
| 10 | Thống kê mô tả mẫu | ✅ | mục 2 |
| 11 | Ma trận payoff kỳ vọng | ✅ | mục 6.1; khớp SI của paper |
| 12 | Quét (µ, β) × treatment | ❌ | cần EGTtools (TD.2) |
| 13 | Median φ_U vs mô hình | ✅ | mục 6.2 |
| 14 | Simplex tứ diện | ❌ | cần EGTtools (TD.2) |
| 15 | Deadlock + ngưỡng dilemma | ✅ | mục 6.1; p* = 0,1324 |
| 16 | Cân bằng Nash | ✅ | mục 6.1; AS không bao giờ là Nash |
| 17 | Phân phối chiến lược theo `p_max` | ⚠️ | mục 6.3; giới hạn µ nhỏ không tách được AU/CAS |
| 18 | Độ nhạy β và µ | ⚠️ panel A | panel B cần EGTtools |
| 19 | Động lực 4 mặt simplex | ❌ | cần EGTtools (TD.2) |

**9 tái tạo đầy đủ · 4 một phần · 2 không áp dụng · 4 cần dependency mới.**

---

## 8. So sánh với người: khớp ở đâu, trái ngược ở đâu

| Hiệu ứng | Người (paper) | Qwen | Đánh giá |
|---|---|---|---|
| `a_{-i}^{t-1}` là dự báo mạnh nhất | +0,607 (p=0,002) | **+2,339** (p<0,001) | ✅ khớp hướng, mạnh hơn nhiều |
| Bị dẫn trước → Unsafe | ΔS −0,296 (p=0,048) | **−0,717** (p=0,013) | ✅ khớp, rõ hơn |
| `a_i^{t-1}` ≈ 0, không dự báo | −0,193 n.s. | **−2,234 (p<0,001)** | ❌ **TRÁI NGƯỢC** |
| Momentum vòng 1 | +0,217 (p=0,06) | không ước lượng được | ⛔ |
| 0,6 vs 0,9 không khác nhau | d = −0,027 | d = −0,178 | ✅ nhỏ, cùng hướng |
| φ_U tổng | 0,584 | 0,203 | ❌ thấp hơn nhiều |
| AS gần như vắng mặt | ~0 | **12/60 thuần AS + 23 mơ hồ** | ❌ trái ngược |

Điểm khác biệt sâu nhất: người có **quán tính hành động**, Qwen **chia lượt**. Và người chơi
Unsafe gần 3× nhiều hơn.

---

## 9. Việc cần làm

Xếp theo thứ tự chặn đường.

### Chặn đường trước khi tăng repetition

1. **Phép thử đổi nhãn hành động cho Gemma.** `OPTION_A`/`OPTION_B` thay cho
   `SAFE`/`UNSAFE`. Không có nó thì Gemma vẫn là 0 thông tin dù chạy 50 rep hay 500.
2. **Tạo biến thiên ở vòng 1.** Chừng nào mọi model còn mở đầu Safe, đặc tả 4–6 **luôn**
   singular và mục #6 không bao giờ đầy đủ. Đường trực tiếp: cell persona `risk_seeking`
   hoặc `adv_adv`.

Chạy 50 rep bây giờ chỉ nhân bản hai vấn đề này lên 5 lần.

### Sau khi gỡ được hai cái trên

3. Chạy **tất cả cell persona trong CÙNG một session Kaggle** — khác session thì
   `protocol_signature` khác nhau, persona trùng khít với batch, và analyser sẽ **raise**
   chứ không im lặng cho ra hệ số.
4. Bỏ `REPETITIONS_OVERRIDE` → 50 rep, để có 50 CRN block thay vì 10.
5. Đặt `RUN_PHASE_OVERRIDE = "confirmatory"` **chỉ sau khi** prompt, config, tiêu chí loại
   trừ và kế hoạch phân tích đã đóng băng.

### Không chặn đường

6. TA.4 — mixed-effects logit, để có hai cách xử lý phụ thuộc như paper.
7. TD.2 — EGTtools, mở khoá mục #12, #14, #18B, #19.
8. Cân nhắc `baseline_swapped` để đo artefact vị trí ghế — hiện chênh lệch chỉ 0,03 nên ưu
   tiên thấp.

---

## 10. Tệp dẫn xuất

Analyser (`--allow-nonconfirmatory-runs`, không `--fit-logit` vì singular):

```
sample_summary.csv · treatment_contrasts.csv · treatment_contrasts_round2plus.csv
unsafe_by_lag_profile_turn.csv · unsafe_by_gap_bin_turn.csv · unsafe_by_race_state_turn.csv
winner_loser_pairs.csv · winner_loser_correlation.csv · horizon_distribution.csv
strategy_summary_player.csv · player_metrics.csv · race_quality.csv · parse_failures.csv
seat_balance.csv · gap_collinearity.csv · human_comparison.csv
theory_vs_experiment.csv · analysis_manifest.json
```

Lý thuyết (`build_theory_tables.py`, không đọc dữ liệu run):

```
theory_payoff_matrix.csv · theory_equilibria.csv
theory_stationary_distribution.csv · theory_expected_unsafe.csv · theory_metadata.json
```

Lệnh tái tạo:

```bash
unzip results/ai_race_results.zip -d <tmp>
python3 results/scripts/check_symmetry.py --input <tmp>/ai_race_results
python3 results/scripts/analyze_ai_race.py --input <tmp>/ai_race_results \
  --output <out> --allow-nonconfirmatory-runs
python3 results/scripts/build_theory_tables.py --output <out-theory>
```

Logit đặc tả 1–2 cho riêng Qwen phải fit ngoài analyser, vì analyser chạy cả 6 đặc tả và
crash ở đặc tả 4.
