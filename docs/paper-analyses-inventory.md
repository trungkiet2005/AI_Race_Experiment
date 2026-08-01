# Danh mục các phân tích trong bài báo gốc

Nguồn: `references/papers/sources/arXiv-2607.26034v1/paper.tex` — Fernández Domingos & Han, *"Falling Behind Drives Unsafe
Development in an Idealised AI Race Experiment"* (arXiv:2607.26034v1).

Đây là bản kiểm kê **mọi phân tích định lượng** xuất hiện trong bài (main text + Supporting
Information), ghi rõ: phân tích gì, trên dữ liệu nào, ra kết quả gì, nằm ở hình/bảng nào.
Dùng để đối chiếu khi thiết kế phân tích tương ứng cho phiên bản LLM.

Ký hiệu: `φ_U` = tần suất chọn Unsafe; `ΔS_{t-1}` = chênh lệch bước đua vòng trước;
`a_i^{t-1}` = hành động vòng trước của chính người chơi; `a_{-i}^{t-1}` = của đối thủ;
`a_i^1` = hành động vòng 1; `p_r^max ∈ {0.1, 0.6, 0.9}` = mức rủi ro riêng tối đa.

---

## 0. Mẫu dữ liệu và các ràng buộc phân tích

| Hạng mục | Giá trị |
|---|---|
| Tuyển ban đầu | 471 người (147 / 128 / 196 theo `p_r^max` = 0.1 / 0.6 / 0.9) |
| Hoàn thành sau loại trừ | 340 người (97 / 104 / 139) |
| Mẫu phân tích panel | **N = 2.888 quan sát, 338 người, 172 cặp** (97 / 105 / 136) |
| Cửa sổ phân tích | từ **vòng t ≥ 2** (để có biến trễ) |
| Số ván | 173 ván, độ dài trung bình thực tế **9,56 vòng** (kỳ vọng lý thuyết 9) |

Hai điều chỉnh mẫu quan trọng: 2 người bị cờ `DATA_EXPIRED` (mất quốc tịch) bị loại khi thêm
biến quốc tịch; 1 người bỏ giữa chừng ở `p_r^max=0.6` vẫn được giữ lại vì đóng góp 3 quyết
định hợp lệ ở t ≥ 2.

---

## 1. Kiểm định tiền đăng ký (pre-registered)

### 1.1. Giả thuyết 1 — Ảnh hưởng của mức rủi ro tối đa lên tần suất Safe/Unsafe

**Phương pháp:** t-test hai mẫu độc lập trên `φ_U` trung bình của từng người (toàn bộ vòng),
hiệu chỉnh Bonferroni cho 3 so sánh cặp.

**Kết quả** (Fig. 2A, N = 98 / 105 / 138):

| So sánh | t | p (Bonferroni) |
|---|---|---|
| 0.1 vs 0.6 | 2,635 | 0,0272 * |
| 0.1 vs 0.9 | 2,811 | 0,0161 * |
| **0.6 vs 0.9** (giả thuyết tiền đăng ký) | **−0,0101** | **1,000 — không có hiệu ứng** |

→ **H1.2 không được ủng hộ.** Mức rủi ro 0.9 không làm tăng hành vi Safe so với 0.6.

### 1.2. Giả thuyết 2 — Sở thích rủi ro (risk preference) dự báo Unsafe

**Phương pháp:** hồi quy logistic hỗn hợp (mixed-effects) có random intercept theo nhóm; biến
sở thích rủi ro là lựa chọn gamble Eckel–Grossman (0–5, coi như thang liên tục); `p_r^max` vào
mô hình dưới dạng biến liên tục đã căn giữa.

**Kết quả** (Table S, `tab:si:prereg-mixed-model`):

| Hệ số cố định | Ước lượng | p |
|---|---|---|
| Intercept | 0,5949 | < 0,001 *** |
| Max private risk | −0,1234 | 0,690 |
| `a_i(t-1)`: Unsafe | −0,6203 | < 0,001 *** |
| `a_{-i}(t-1)`: Unsafe | 0,2820 | 0,057 † |
| `ΔS(t-1)` | −0,2863 | 0,064 † |
| **Risk preference** | **−0,0452** | **0,148 — không đáng kể** |
| Unsafe × Unsafe | 0,3046 | 0,125 |
| Unsafe × ΔS(t-1) | 0,3496 | 0,083 † |
| Opp. Unsafe × ΔS(t-1) | 0,3923 | 0,059 † |
| Tương tác ba chiều | −0,1588 | 0,538 |
| Risk pref × Max private risk | −0,1354 | 0,168 |

Random intercept SD = 0,8088.

→ **H2.1–H2.3 không được ủng hộ.** Sở thích rủi ro không dự báo hành vi Unsafe, cả tác động
chính lẫn tương tác với mức rủi ro.

### 1.3. Effect size cho so sánh tiền đăng ký

Trên mẫu phân tích t ≥ 2 (Table S, `tab:si:pairwise-comparisons`):

| So sánh | N₁ | N₂ | t | p (Bonf.) | Cohen's d |
|---|---|---|---|---|---|
| 0.1 vs 0.6 | 97 | 105 | 2,424 | 0,0487 | 0,341 |
| 0.1 vs 0.9 | 97 | 136 | 2,422 | 0,0486 | 0,323 |
| 0.6 vs 0.9 | 105 | 136 | −0,206 | 1,000 | **−0,027** |

Tác giả **từ chối báo cáo post-hoc power**, viện dẫn Hoenig & Heisey (2001): power tính từ
effect size quan sát chỉ là hàm xác định của p-value, không phải bằng chứng độc lập.

---

## 2. Phân tích khám phá (exploratory) — phần đóng góp chính

### 2.1. `φ_U` theo trạng thái cuộc đua (Fig. 2B)

Tần suất Unsafe được cắt theo ba biến trạng thái đồng thời: vị thế đua `ΔS_{t-1}`, hành động
trước của bản thân `a_i^{t-1}`, và hành động trước của đối thủ `a_{-i}^{t-1}`.

Phát hiện: người **đang bị dẫn trước (falling behind)** chọn Unsafe nhiều hơn người đang dẫn;
sau khi cả hai cùng chơi Unsafe, xu hướng tiếp tục Unsafe tăng; đối thủ chơi Safe làm tăng
khả năng chuyển sang Safe.

### 2.2. Tương quan `φ_U` của người thắng và người thua trong cặp (Fig. 2C)

Vẽ tương quan giữa `⟨φ_U^W⟩` (người thắng) và `⟨φ_U^L⟩` (người thua) cùng phân phối biên.

Phát hiện: người thắng có `φ_U` cao hơn, nhưng hai đại lượng **tương quan dương trong cặp**, và
**tương quan này tăng theo `p_r^max`** — rủi ro càng cao thì kết quả đua càng gắn với hồ sơ
Unsafe chung của cả cặp. Phân phối biên cho thấy mode của `φ_U` thấp nhất ở `p_r^max = 0.9`.

### 2.3. Hồi quy logistic panel cluster-robust (Table 1 — phân tích trung tâm)

**Thiết kế:** biến phụ thuộc = 1 nếu chọn Unsafe ở vòng t. Sai số chuẩn cluster ở **cấp cặp**
(172 cluster). Mẫu t ≥ 2, N = 2.888. Treatment tham chiếu: `p_r^max = 0.1`. `ΔS_{t-1}` căn giữa
theo trung bình mẫu. Mọi mô hình đều có biến kiểm soát nhân khẩu + sở thích rủi ro.

**Sáu đặc tả lồng nhau:**

| Mô hình | Nội dung |
|---|---|
| (1) | chỉ treatment dummy + covariates |
| (2) | thêm ba biến trễ cộng tính (`a_i^{t-1}`, `a_{-i}^{t-1}`, `ΔS_{t-1}`) |
| (3) | thay bằng tương tác ba chiều đầy đủ |
| (4)–(6) | lặp lại (1)–(3) sau khi thêm `a_i^1` |

**Hệ số chính:**

| Biến | (1) | (2) | (3) | (4) | (5) | (6) |
|---|---|---|---|---|---|---|
| `p_r^max=0.6` | −0,183 | −0,142 | −0,152 | −0,155 | −0,123 | −0,132 |
| `p_r^max=0.9` | −0,245 | −0,190 | −0,194 | −0,241 | −0,188 | −0,191 |
| `a_i^1` | | | | 0,291 * | 0,207 † | 0,217 † |
| `a_i^{t-1}` | | 0,022 | −0,173 | | 0,007 | −0,193 |
| `a_{-i}^{t-1}` ← mạnh nhất | | 0,863 (p<0,001) | 0,640 (p=0,001) | | 0,832 (p<0,001) | 0,607 (p=0,002) |
| `ΔS_{t-1}` | | 0,106 † | −0,238 | | 0,065 | −0,296 * |
| `a_i^{t-1} × a_{-i}^{t-1}` | | | 0,396 † | | | 0,400 † |
| `a_i^{t-1} × ΔS_{t-1}` | | | 0,448 * | | | 0,466 * |
| `a_{-i}^{t-1} × ΔS_{t-1}` | | | 0,201 | | | 0,218 |
| Tương tác ba chiều | | | −0,162 | | | −0,183 |
| Pseudo R² | 0,006 | 0,035 | 0,039 | 0,009 | 0,037 | 0,040 |

**Bốn kết luận rút ra:**

1. **Hành động trước của đối thủ là dự báo mạnh nhất và bền vững nhất** (model 3: β̂ = 0,640,
   p = 0,001; model 6: β̂ = 0,607, p = 0,002).
2. **Hành động trước của chính mình không dự báo được** khi đã kiểm soát đối thủ và vị thế đua
   → không phải quán tính hành động đơn thuần.
3. **Vị thế đua có tác dụng, nhưng phụ thuộc bối cảnh:** dẫn trước làm giảm Unsafe (model 6:
   β̂ = −0,296, p = 0,048; model 3 không đáng kể, p = 0,113), và tương tác
   `a_i^{t-1} × ΔS_{t-1}` dương có ý nghĩa (p = 0,016 / 0,011).
4. **Hành động vòng 1 dự báo hành vi về sau** — `a_i^1` có ý nghĩa ở model 4, cận ý nghĩa ở
   model 6 → "behavioural momentum" ban đầu.

Hệ số treatment âm nhưng **không có ý nghĩa thống kê** trong panel t ≥ 2.

**Lưu ý về pseudo R²:** giá trị nhỏ (0,006–0,040) được biện minh bằng McFadden (1974) — thang
0,2–0,4 đã là fit rất tốt; kiểm định liên quan là dấu và ý nghĩa của từng hệ số, không phải
phương sai giải thích.

**Cảnh báo nội sinh:** vì `a_i^t` phụ thuộc toàn bộ lịch sử chứ không chỉ vòng liền trước, các
biến trễ **không ngoại sinh chặt** → hệ số phải đọc là *liên hệ có điều kiện*, không phải hiệu
ứng nhân quả.

### 2.4. Hệ số covariate nhân khẩu và sở thích rủi ro (Table S, `tab:si:panel-covariates`)

Báo cáo phần bị lược khỏi Table 1: giới tính (nam so với nữ), tuổi (căn giữa), quốc tịch
(Nam Phi / Ba Lan so với phần còn lại — hai nguồn tuyển chính, 44% và 11%), sở thích rủi ro
(cộng tính và tương tác với treatment).

**Không biến nào có ý nghĩa thống kê** (p > 0,1 toàn bộ). Hệ số sở thích rủi ro **âm nhất quán**
qua cả 6 mô hình — nghĩa là người ít e ngại rủi ro lại chọn Unsafe *ít* hơn, ngược trực giác —
nhưng độ lớn nhỏ và không phân biệt được với 0.

### 2.5. Robustness: loại bỏ cặp bỏ giữa chừng (Table S, `tab:si:cluster_logit_unsafe_no_dropout`)

Chạy lại toàn bộ Table 1 sau khi loại 1 cặp có người timeout giữa ván (N = 2.885, 171 cluster).
**Mọi hệ số gần như không đổi** → kết quả không bị chi phối bởi cặp chưa hoàn thành.

### 2.6. Phân phối số vòng thực tế (Fig. S1)

Histogram số vòng của 173 ván. Trung bình quan sát 9,56 so với kỳ vọng lý thuyết 9. Dùng để
xử lý một **lỗi số học trong pre-registration**: hướng dẫn ghi "trung bình 10 vòng", giá trị
đúng là 5 + 1/0,2 − 1 = 9. Tác giả công khai lỗi, lập luận rằng nó không ảnh hưởng hành vi
(phát hiện sau khi thu dữ liệu xong) và nhỏ so với phương sai của phân phối hình học.

### 2.7. Thống kê mô tả mẫu (Table S, `tab:si:summary-stats`)

Theo từng treatment: số người / số cặp, cân bằng giới, tuổi (mean, SD), quốc tịch,
`φ_U` trung bình, và lựa chọn gamble.

| Chỉ số | 0.1 | 0.6 | 0.9 | Tổng |
|---|---|---|---|---|
| N người | 97 | 105 | 136 | 338 |
| N cặp | 49 | 53 | 70 | 172 |
| **Mean `φ_U`** | **0,640** | **0,558** | **0,564** | 0,584 |
| Mean gamble choice | 1,85 | 2,22 | 2,37 | 2,17 |
| Mode gamble choice | 0 | 2 | 2 | 2 |

---

## 3. Phân tích lý thuyết trò chơi / mô hình tiến hóa

### 3.1. Xây dựng ma trận payoff kỳ vọng (SI `si:expected-payoff-matrix`)

Tham số: `b = 4`, `B = 100`, `c = 1`, `s_U = 1,5`, `s_S = 1`, `p = 0,2` (δ = 0,8), E[W] = 9.

Payoff stage game suy ra từ tham số: π₁₁ = −c + b/2 = 1; π₁₂ = −c + b/(s_U+1) = 0,6;
π₂₁ = s_U·b/(s_U+1) = 2,4; π₂₂ = b/2 = 2.

Tiến độ đua: `R_i(W) = W·s_S + (s_U − s_S)·n_i(W)`. Rủi ro riêng hiệu dụng:
`p_r(W) = p_r^max · n_i(W)/W`, **chỉ áp cho người thắng hoặc hòa**.

**Hai chế độ tính payoff:**
- Cặp chiến lược **vô điều kiện** (AS/AU với nhau): có dạng đóng, thay trực tiếp E[W]:
  - `Π̄_{AS,AS} = B/2 + E[W]·π₁₁ = 59`
  - `Π̄_{AS,AU} = E[W]·π₁₂ = 5,4`
  - `Π̄_{AU,AS} = (1 − p_r^max)(B + E[W]·π₂₁) = (1 − p_r^max)·121,6`
  - `Π̄_{AU,AU} = (1 − p_r^max)(B/2 + E[W]·π₂₂) = (1 − p_r^max)·68`
- Cặp có **chiến lược điều kiện** (CS/CAS): payoff không affine theo W (phụ thuộc parity của W
  và tỉ lệ Unsafe thực tế) → **mô phỏng Monte Carlo, 10⁴ lần lặp cho mỗi cặp có thứ tự.**

### 3.2. Không gian bốn chiến lược rút gọn

| Ký hiệu | Tên | Hành vi |
|---|---|---|
| AS | Always Safe | luôn S |
| AU | Always Unsafe | luôn U |
| CS | Conditionally Safe | vòng 1 chơi S, từ vòng 2 copy hành động trước của đối thủ |
| CAS | Conditionally Antisocial Safe | vòng 1 chơi U, từ vòng 2 copy đối thủ |

Cả CS và CAS đều là **memory-one reactive / Tit-for-Tat**, chỉ khác nước đi đầu. Việc chọn tập
này được biện minh trực tiếp từ hồi quy: hiệu ứng mạnh của `a_{-i}^{t-1}` → cần chiến lược điều
kiện; hiệu ứng của `a_i^1` → cần tách CS khỏi CAS.

Tác giả thừa nhận đây là **xấp xỉ**: dữ liệu gợi ý hành vi phụ thuộc *khoảng cách* đua
(distance-dependent), nhưng không mô hình hóa rõ để giữ tính khả giải; CS/CAS được đọc là
proxy cho phản ứng bảo thủ vs. hung hăng.

### 3.3. Động lực tiến hóa quần thể hữu hạn (Fig. 3)

**Phương pháp:** quần thể well-mixed cỡ Z, pairwise comparison với hàm Fermi
`(1 + exp[−β(f_A − f_B)])^{-1}`, đột biến µ; tính **phân phối dừng của chuỗi Markov** rồi suy ra
tần suất Unsafe dự báo. Cài đặt bằng EGTtools.

**Ba phân tích con:**
- **Fig. 3A** — quét toàn bộ không gian tham số (µ, β) × treatment. Hai điểm được đánh dấu:
  điểm tham chiếu β = 2, µ = β/Z = 0,02 (chọn lọc mạnh, đột biến sát ngưỡng trung tính); và
  **điểm khớp tốt nhất** µ = 0,05, β = 0,01 (nhiễu hành vi cao, chọn lọc yếu).
- **Fig. 3B** — so khớp **median `φ_U` thực nghiệm** với dự báo mô hình theo treatment. Mô hình
  tái tạo đúng *hướng* của hiệu ứng treatment, gồm cả chênh lệch nhỏ giữa hai mức rủi ro cao và
  chênh lệch lớn hơn so với mức rủi ro thấp.
- **Fig. 3C** — **động lực simplex tứ diện** 4 chiến lược: mũi tên = gradient chọn lọc (màu = cường
  độ), quả cầu xám = phân phối dừng quần thể hữu hạn. Ở điểm tham chiếu, chiến lược trội phụ
  thuộc mạnh vào `p_r^max`: **AU khi rủi ro thấp → CAS ở rủi ro trung bình → CS ở rủi ro cao.**
  Ở vùng khớp tốt nhất, phân phối dừng khuếch tán về mặt và ruột simplex.

### 3.4. Quan hệ với Prisoner's Dilemma lặp và cân bằng Nash (SI `si:sec:pd-relationship`)

**a) Phân loại stage game.** Viết theo ký hiệu PD: T = 2,4; P = 2; R = 1; S = 0,6 → **T > P > R > S**,
tức là **Deadlock chứ không phải PD** (PD cần R > P). Unsafe trội chặt mỗi vòng như Defect,
nhưng cả hai cùng Unsafe *không* tệ hơn cả hai cùng Safe ở cấp stage game.

**b) Ngưỡng biến thành social dilemma.** Rủi ro tích lũy đảo ngược điều đó. Mutual Safe cho tổng
payoff cao hơn mutual Unsafe khi

```
p_r^max > p_r^max* = 1 − (B/2 + E[W]π₁₁)/(B/2 + E[W]π₂₂) = 1 − 59/68 ≈ 0,132
```

→ đúng ở treatment 0.6 và 0.9 (và gần đạt ở 0.1). **Cấu trúc lặp biến một trò chơi không phải
dilemma thành social dilemma ở cấp tổng payoff kỳ vọng.**

**c) Cân bằng Nash trong trò chơi hai chiến lược vô điều kiện (AS, AU):**

| Vùng `p_r^max` | Cân bằng |
|---|---|
| < 0,515 | (AU, AU) duy nhất |
| 0,515 – 0,921 | **cả (AS,AS) và (AU,AU)** — cấu trúc phối hợp, hai điểm nghỉ ổn định, cách nhau bởi hỗn hợp nội không ổn định |
| > 0,921 | (AS, AS) duy nhất |

**d) Cân bằng Nash của trò chơi bốn chiến lược đầy đủ** (dò best-response vét cạn trên chính ma
trận payoff dùng cho Fig. 3, Table S `tab:si:pd-nash`):

| `p_r^max` | Cân bằng Nash đối xứng | AS có phải cân bằng? |
|---|---|---|
| 0.1 | AU, CAS (tương đương) | Không |
| 0.6 | AU, CAS (tương đương) | Không |
| 0.9 | **CS (duy nhất)** | Không |

Hai sự kiện cấu trúc: (i) CAS **không phân biệt được với AU** khi gặp đối thủ chơi Unsafe từ
vòng đầu → 4 tổ hợp AU/CAS cùng là cân bằng hoặc cùng không; (ii) **AS không bao giờ là cân bằng**
— một đột biến CAS chơi Unsafe một lần rồi Safe mãi sẽ vượt AS đúng nửa bước mỗi ván và thắng
gần như không rủi ro. Ngược lại, CS thành cân bằng duy nhất ở rủi ro cao vì khi gặp đối thủ
Unsafe, CS chơi Safe vòng 1 rồi Unsafe → luôn chậm nửa bước → **không bao giờ dính xổ số rủi
ro cuối ván**; khi xổ số đó đủ đắt, cố tình thua lại an toàn hơn thắng.

**e) Đối chiếu với văn liệu PD lặp.** CS/CAS là các trường hợp riêng của TFT — họ chiến lược
được suy luận phổ biến nhất từ hành vi người trong thí nghiệm PD lặp (Dal Bó & Fréchette 2018;
Montero-Porras et al. 2022). Kết quả null về sở thích rủi ro cũng khớp với văn liệu này.

### 3.5. Phân phối chiến lược theo `p_r^max` (Fig. S5)

Quét `p_r^max` liên tục ở Z = 100, β = 2, µ = 1/Z. Panel A: tần suất từng chiến lược. Panel B:
tần suất Unsafe kỳ vọng.

**Hai chuyển pha:** `p_r^max < 0,2` → AU trội; `0,2 < p_r^max < 0,6` → CAS trội;
`p_r^max > 0,6` → CS trội. Ở β cao, ngay cả khi CAS trội thì tần suất Unsafe vẫn rất lớn, và
**tụt mạnh về 0 khi CS trội**.

### 3.6. Độ nhạy theo β và µ (Fig. S6, S7)

- **Panel A (Fig. S6):** tần suất dừng của 4 chiến lược theo β (log scale), mỗi treatment, µ = 1/Z.
  β tăng → quần thể tập trung vào một chiến lược trội: AU ở 0.1, CAS ở 0.6, CS ở 0.9.
- **Panel B (Fig. S7):** theo µ (log scale), β = 2, Z = 100. µ tăng → phân phối dừng tiến về
  đồng đều; µ giảm → phục hồi trật tự phụ thuộc treatment như panel A.

Kết luận độ nhạy: **với β > 1 và dải µ rộng, kết quả mô hình ổn định.**

### 3.7. Động lực trên bốn mặt 3-chiến-lược của simplex (Fig. S8)

Bốn trò chơi con thu được bằng cách bỏ lần lượt một chiến lược (AS-AU-CS bỏ CAS; AS-AU-CAS bỏ
CS; AS-CS-CAS bỏ AU; AU-CS-CAS bỏ AS) × 3 treatment. Streamline = gradient chọn lọc, nền xám =
phân phối dừng (Z = 100, β = 0,1, µ = 0,1).

Kết quả: **trên cả bốn mặt**, khối lượng dừng tập trung gần AU ở 0.1 → dịch về CAS ở 0.6 → dịch
về CS ở 0.9, phản chiếu simplex đầy đủ. **AS hút khối lượng dừng không đáng kể ở mọi mặt và mọi
treatment** → AS bị chiến lược điều kiện lấn át bất kể có chiến lược nào khác hiện diện.

---

## 4. Bảng tra nhanh: phân tích ↔ hình/bảng

| # | Phân tích | Vị trí |
|---|---|---|
| 1 | t-test cặp giữa các treatment (toàn vòng) | Fig. 2A |
| 2 | Mixed-effects logistic tiền đăng ký (risk preference) | Table S `prereg-mixed-model` |
| 3 | Effect size + t-test trên mẫu t ≥ 2 | Table S `pairwise-comparisons` |
| 4 | `φ_U` theo ΔS × `a_i^{t-1}` × `a_{-i}^{t-1}` | Fig. 2B |
| 5 | Tương quan `φ_U` thắng–thua trong cặp | Fig. 2C |
| 6 | **Hồi quy logistic panel cluster-robust, 6 đặc tả** | **Table 1** |
| 7 | Hệ số covariate nhân khẩu + risk preference | Table S `panel-covariates` |
| 8 | Robustness loại cặp dropout | Table S `no_dropout` |
| 9 | Phân phối số vòng thực tế | Fig. S1 |
| 10 | Thống kê mô tả mẫu | Table S `summary-stats` |
| 11 | Ma trận payoff kỳ vọng (dạng đóng + Monte Carlo 10⁴) | SI `expected-payoff-matrix` |
| 12 | Quét không gian (µ, β) × treatment | Fig. 3A |
| 13 | So khớp median `φ_U` thực nghiệm vs mô hình | Fig. 3B |
| 14 | Động lực simplex tứ diện 4 chiến lược | Fig. 3C |
| 15 | Phân loại Deadlock + ngưỡng social dilemma | SI `pd-relationship` |
| 16 | Cân bằng Nash (2 chiến lược và 4 chiến lược) | SI `pd-relationship`, Table S `pd-nash` |
| 17 | Phân phối chiến lược theo `p_r^max` (chuyển pha) | Fig. S5 |
| 18 | Độ nhạy theo β và µ | Fig. S6, S7 |
| 19 | Động lực trên 4 mặt 3-chiến-lược | Fig. S8 |

---

## 5. Những hạn chế tác giả tự nêu (ảnh hưởng tới cách đọc phân tích)

1. Cuộc đua bị **cách điệu hóa mạnh** và ngắn so với cạnh tranh công nghệ thực.
2. Mô hình rút gọn xấp xỉ hành vi phụ thuộc khoảng cách bằng vài chiến lược tất định.
3. Thiết kế tập trung vào **rủi ro riêng**, không phải rủi ro tập thể / hệ thống.
4. **Các phát hiện trung tâm (đối thủ, vị thế đua, momentum vòng 1) đều là khám phá, không tiền
   đăng ký.**
5. Ba treatment chỉ thay đổi `p_r^max`, giữ nguyên cấu trúc cạnh tranh.
6. Biến trễ **không ngoại sinh chặt** → không đọc được như hiệu ứng nhân quả.
