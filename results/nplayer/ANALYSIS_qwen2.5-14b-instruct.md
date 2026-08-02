# Phân tích kết quả pilot N-player (qwen2.5-14b-instruct)

Dữ liệu: hai run pilot (`run_phase="pilot"`, chưa phải confirmatory) dùng
`ai_race/engine_nplayer` (N=3), cùng model `qwen2.5-14b-instruct` (offline,
transformers), cùng cơ chế/seed/prompt:

- **`nplayer_nonpersona/baseline_nplayer_n3`**: neutral (không persona), 3 mức
  rủi ro `max_private_risk` ∈ {0.1, 0.6, 0.9}, 20 lần lặp/mức → 60 races, 1584
  lượt quyết định.
- **`nplayer-riskaware/persona_nplayer_baseline_risk_{1..6}_n3`**: persona rủi
  ro Eckel-Grossman (R1 = chọn mức cược an toàn nhất trong 6 mức, R6 = chọn
  mức cược rủi ro nhất), áp cho **cả 3 ghế giống nhau** (symmetric), cùng 3
  mức rủi ro, chỉ 2 lần lặp/mức → 6 races/persona × 6 persona = 36 races, 594
  lượt quyết định.

Tham số cơ chế (giống nhau cho cả 2 run): `n=3, s=1.5(speed), b=4(benefit),
c=1(cost), B=100(racePrize), minRounds=5, stopProbability=0.2` → horizon kỳ
vọng lý thuyết `E[W]=9`.

Script tái tạo số liệu mô tả: [analyze_nplayer_results.py](analyze_nplayer_results.py).
Script tái tạo số liệu suy luận (hồi quy, phân loại chiến lược, theory-fit):
[rigorous_analysis.py](rigorous_analysis.py) → ghi ra
[analysis/nplayer/derived/](../../analysis/nplayer/derived/).
Dashboard trực quan: [analysis/nplayer/visualizations/dashboard.html](../../analysis/nplayer/visualizations/dashboard.html).

## 1. Chất lượng dữ liệu

- **0% parse failure** ở cả 2 run — không có race nào bị loại vì lỗi parse
  hành động.
- **Kiểm tra CRN (common-random-number)**: với cùng `(risk, rep)`, `n_rounds`
  thực tế **giống hệt nhau tuyệt đối** giữa baseline và cả 6 persona (đã kiểm
  từng cặp rep 0/1 × 3 mức risk = 6 tổ, mỗi tổ chỉ có 1 giá trị `n_rounds`
  duy nhất). Xác nhận đúng thiết kế: horizon chỉ phụ thuộc `game_seed =
  base_seed + rep`, độc lập với persona/agents — matched-repetition hoạt động
  đúng như `CLAUDE.md` mô tả.
- Horizon trung bình thực tế của baseline (8.8 vòng, std 4.0, tối đa 17) khớp
  hợp lý với `E[W]=9` lý thuyết. Horizon của run persona (5.5 vòng trung
  bình) thấp hơn hẳn — **không phải bất thường**, chỉ vì run này dùng đúng 2
  rep (rep 0 và 1), mà 2 rep đó tình cờ rút được horizon ngắn (5 và 6 vòng);
  đây là biến thiên mẫu nhỏ, không phải lỗi cơ chế.

## 2. Baseline (không persona): tỷ lệ UNSAFE giảm dần theo rủi ro

| max_private_risk (pr) | Tỷ lệ UNSAFE | Setback rate (trong nhóm leader) | Full-tie rate |
|---|---|---|---|
| 0.1 | 69.1% | 3.6% | 5% |
| 0.6 | 57.2% | 36.7% | 10% |
| 0.9 | 46.6% | 42.9% | 10% |

Hướng đúng như kỳ vọng: risk càng cao → mô hình chơi UNSAFE càng ít. Nhưng
**hiệu ứng khá thoải (gradual)**: chỉ giảm ~22 điểm phần trăm trên toàn dải
0.1→0.9, và ngay cả ở pr=0.9 mô hình vẫn chơi UNSAFE gần một nửa số lượt.
Setback rate tăng theo pr là hệ quả cơ học trực tiếp của công thức
`effective_risk = max_private_risk × unsafe_fraction`, không phải phát hiện
hành vi.

Phân bố outcome (thắng/thua/hòa) và full-tie rate gần như không đổi qua 3 mức
risk — risk chủ yếu ảnh hưởng đến *lựa chọn hành động*, không ảnh hưởng nhiều
đến *cấu trúc kết quả cuộc đua*.

### Đối chiếu với lý thuyết N-Player (`N-Player/theory`)

Thay vì chốt một β tùy ý, `rigorous_analysis.py::theory_beta_fit` **quét lưới
β** (log-scale, 200 điểm từ 10⁻⁴ đến 10¹) và tìm β cực tiểu hóa tổng bình
phương sai khác so với 3 điểm thực nghiệm (giới hạn small-mutation, Z=100,
cùng tham số cơ chế n=3, s=1.5, b=4, c=1, B=100, W=9):

| pr | Lý thuyết @ β=0.1 (tùy ý) | Lý thuyết @ β tối ưu = 0.0016 | Thực nghiệm |
|---|---|---|---|
| 0.1 | 100% | 69.1% | 69.1% |
| 0.6 | 100% | 41.2% | 57.2% |
| 0.9 | 0.03% | 25.8% | 46.6% |

Ngay cả ở β tối ưu (chọn lọc rất yếu, gần trôi dạt trung tính), mô hình vẫn
**dự đoán độ dốc giảm mạnh hơn thực tế**: khớp gần đúng ở pr=0.1, nhưng thấp
hơn thực nghiệm rõ rệt ở pr=0.6 (41% dự đoán so với 57% thực tế) và pr=0.9
(26% so với 47%). Nói cách khác: **mô hình quần thể tiến hóa dự đoán LLM sẽ
giảm hành vi UNSAFE nhanh hơn theo rủi ro so với những gì thực sự quan sát
được** — LLM "cứng đầu" hơn, giữ UNSAFE cao hơn dự đoán khi rủi ro đã tăng.

Đây **không phải mâu thuẫn về nguyên tắc** — hai đại lượng đo hai thứ khác
nhau về bản chất: lý thuyết mô tả trạng thái mà một *quần thể* tác nhân hội
tụ tới sau rất nhiều thế hệ chọn lọc tự nhiên, còn dữ liệu là hành vi suy
luận trong-ngữ-cảnh của *một* LLM trong từng ván riêng lẻ — không có lý do
tiên nghiệm để hai thứ khớp về độ lớn hay hình dạng chuyển tiếp. Nhưng việc
quét β và vẫn không khớp được cả 3 điểm cùng lúc (ngay ở mức khớp tốt nhất)
là bằng chứng định lượng, không chỉ định tính, cho khoảng cách giữa hai mô
hình.

## 3. Hiệu ứng thời điểm và vị thế

- **Vòng 1: 100% UNSAFE ở cả 3 mức risk.** Mọi người chơi luôn mở màn bằng
  UNSAFE bất kể tham số rủi ro của ván — hợp lý vì ở vòng 1 chưa có tín hiệu
  nào từ đối thủ (progress = 0, chưa có gì để so sánh); tỷ lệ UNSAFE chỉ bắt
  đầu phân hóa theo risk **từ vòng 2 trở đi**.
- **Theo vị thế** (dẫn đầu / bị dẫn / hòa) tại pr=0.1: nhóm "hòa" UNSAFE
  nhiều nhất (76.7%), nhóm "bị dẫn" ít nhất (56.3%) — hơi phản trực giác nếu
  kỳ vọng "bị dẫn thì liều để đuổi kịp". Ở pr=0.9 thì gần như phẳng
  (45-52%), không còn phân hóa rõ theo vị thế.

## 4. Persona rủi ro (Eckel-Grossman R1-R6): hiệu ứng persona lấn át hiệu ứng cơ chế

| Persona | Mô tả | UNSAFE @pr=0.1 | @pr=0.6 | @pr=0.9 | Trung bình |
|---|---|---|---|---|---|
| R1 | chọn cược an toàn nhất (1/6) | 0% | 0% | 0% | **0%** |
| R2 | cược 2/6 | 0% | 0% | 0% | **0%** |
| R3 | cược giữa (3/6) | 78.8% | 63.6% | 48.5% | 63.6% |
| R4 | cược 4/6 | 100% | 100% | 100% | **100%** |
| R5 | cược 5/6 | 100% | 100% | 100% | **100%** |
| R6 | cược rủi ro nhất (6/6) | 100% | 100% | 100% | **100%** |

Đây là phát hiện rõ ràng nhất trong dữ liệu: **persona tạo ra một hàm bậc
thang gần như hoàn hảo** — R1/R2 luôn SAFE tuyệt đối, R4/R5/R6 luôn UNSAFE
tuyệt đối, **bất kể tham số rủi ro thực tế của ván đua (pr) là bao nhiêu**.
Chỉ riêng R3 (persona "trung tính giữa") mới thể hiện độ nhạy với pr, và độ
nhạy đó lặp lại đúng hình dạng giảm-dần-theo-risk của baseline không-persona
(78.8%→63.6%→48.5%, so với baseline 69.1%→57.2%→46.6% — cao hơn baseline một
chút nhưng cùng xu hướng).

Diễn giải: model có vẻ **neo (anchor) rất mạnh vào nhãn persona** thay vì
tích hợp thông số rủi ro định lượng nêu trong prompt, đối với 5/6 mức persona
— chỉ ở đúng mức "trung tính" model mới thực sự cân nhắc con số `pr` thực tế
của ván đấu.

**Giới hạn cần lưu ý nghiêm túc**: mỗi ô persona×risk chỉ có **2 race độc lập**
(2 rep × 3 người chơi × 5-6 vòng ≈ 33 lượt quyết định, nhưng đơn vị độc lập
thống kê thực sự là race, tức n=2). Đây là dữ liệu **pilot thăm dò**, một
model duy nhất, persona chỉ áp đối xứng (không có ma trận persona lệch giữa
các ghế). Không nên khái quát hóa thành "LLM có thiên kiến neo persona" nói
chung — chỉ là một quan sát mô tả trên checkpoint/run cụ thể này, cần chạy
confirmatory với nhiều rep hơn và nhiều model hơn để xác nhận.

## 5. Phân loại chiến lược gần nhất (Hamming distance, AS/AU/CS)

Áp `strategy_analysis/classify_nplayer.py` (bản N-player của
`strategy_analysis/classify.py`, chỉ 3 chiến lược AS/AU/CS — N-player không
có CAS) lên toàn bộ 180 player-race của baseline:

| risk | AS | AU | CS | Tie/Other |
|---|---|---|---|---|
| 0.1 | 1.7% | 81.7% | 0% | 16.7% |
| 0.6 | 18.3% | 60.0% | 0% | 21.7% |
| 0.9 | 45.0% | 41.7% | 0% | 13.3% |

Nhất quán với tỷ lệ UNSAFE thô: AS tăng dần, AU giảm dần theo risk. Đáng chú
ý: **không một player-race nào được phân vào CS** — chiến lược "an toàn có
điều kiện" (bắt chước nhóm) không xuất hiện trong dữ liệu thật ở mức có thể
nhận diện được bằng Hamming distance, dù đó là 1 trong 3 chiến lược tham
chiếu. Tỷ lệ hòa (Tie/Other, quỹ đạo không khớp duy nhất một chiến lược nào)
khá cao (13–22%), phản ánh việc LLM không chơi một chiến lược thuần túy,
xác định trước.

## 6. Hồi quy logistic panel cluster-robust (baseline, đặc tả 3)

`unsafe ~ C(risk) + own(t-1) × others(t-1)_tỷ_lệ_Unsafe × ΔS(t-1)`, cluster
theo `rep` (20 cluster — mỗi rep chia sẻ cùng horizon/setback draw qua cả 3
mức risk nhờ CRN, đã xác nhận ở mục 1). N = 1.404 quyết định (vòng ≥ 2).

| Biến | Hệ số | CI 95% | p |
|---|---|---|---|
| Risk = 0.6 (so với 0.1) | −1.065 | [−1.348, −0.783] | < 0.001 |
| Risk = 0.9 (so với 0.1) | −1.732 | [−2.058, −1.405] | < 0.001 |
| Own(t-1) = Unsafe | +1.101 | [0.560, 1.642] | < 0.001 |
| Others(t-1) tỷ lệ Unsafe | +1.472 | [0.607, 2.338] | < 0.001 |
| Own(t-1) × Others(t-1) | −3.989 | [−5.235, −2.743] | < 0.001 |
| ΔS(t-1) (vị thế đua) | +0.032 | [−0.328, 0.393] | 0.860 (n.s.) |
| Own(t-1) × ΔS(t-1) | +0.533 | [−0.399, 1.464] | 0.262 (n.s.) |
| Others(t-1) × ΔS(t-1) | +0.701 | [−0.012, 1.413] | 0.054 (cận biên) |
| Tương tác ba chiều | −0.444 | [−1.770, 0.882] | 0.511 (n.s.) |

**Ba điểm khác biệt so với paper gốc 2-player** (`docs/paper-analyses-inventory.md`
mục 2.3, Table 1):
1. Trong paper người, **hành động trước của chính mình không dự báo được**
   sau khi kiểm soát đối thủ. Ở đây, **cả hai đều dự báo mạnh** (own p<0.001,
   others p<0.001) — LLM có "quán tính hành động" mà con người không có.
2. Hệ số tương tác `own × others` **âm rất lớn** (−3.99, p<0.001): khi cả bản
   thân và đối thủ đều vừa chơi Unsafe, xác suất tiếp tục Unsafe **thấp hơn**
   tổng hai hiệu ứng riêng lẻ — một dạng bão hòa không thấy trong paper người.
3. **Vị thế đua (ΔS) hầu như không có tác dụng riêng lẻ** ở đây (p=0.86),
   trong khi ở paper người đây là một phát hiện trung tâm ("falling behind
   drives unsafe development" — chính là tên bài báo). Tương tác
   `others(t-1) × ΔS(t-1)` chỉ cận biên (p=0.054).

Đặc tả 4–6 (thêm `first_round_unsafe`) **không ước lượng được**: biến này là
hằng số (=1) trong toàn bộ 180 player-race của baseline — mọi race đều mở
màn bằng UNSAFE (mục 3) — nên cộng tuyến hoàn toàn với intercept. Đây tự nó
là một phát hiện (hành vi vòng 1 hoàn toàn không có phương sai để hồi quy),
không phải lỗi kỹ thuật.

**Cỡ mẫu và giới hạn**: 20 cluster là khá mỏng cho SE cluster-robust (paper
người có 172 cặp); nên đọc dấu và độ lớn tương đối, không đọc p-value như
ngưỡng cứng tuyệt đối. Persona không đưa vào hồi quy — với chỉ 2 race/ô,
nhiều ô có phương sai bằng 0 (R1/R2/R4/R5/R6), khiến logit hoặc không hội tụ
hoặc separation hoàn hảo; phần persona vẫn chỉ mô tả (mục 4).

## 7. Tóm tắt

1. Cơ chế hoạt động đúng thiết kế (CRN khớp tuyệt đối, setback rate tăng đúng
   theo `max_private_risk`, 0% parse failure).
2. Baseline: UNSAFE giảm dần khi rủi ro tăng (hồi quy xác nhận: risk 0.6 và
   0.9 đều giảm có ý nghĩa thống kê so với 0.1, p<0.001), nhưng thoải hơn
   nhiều so với dự đoán của mô hình lý thuyết quần thể — kể cả sau khi quét β
   tìm điểm khớp tốt nhất, lý thuyết vẫn dự đoán độ dốc giảm mạnh hơn thực tế
   quan sát được ở risk 0.6/0.9.
3. Mọi người chơi luôn mở màn UNSAFE ở vòng 1 (không có phương sai để hồi
   quy), bất kể risk; phân hóa hành vi chỉ xuất hiện từ vòng 2.
4. Hồi quy cho thấy **cả hành động trước của chính mình lẫn của đối thủ đều
   dự báo mạnh** (khác paper người, nơi chỉ đối thủ mới dự báo được), có bão
   hòa khi cả hai cùng Unsafe, và **vị thế đua gần như không có tác dụng** —
   trái ngược phát hiện trung tâm của paper người gốc.
5. Phân loại chiến lược: không player-race nào khớp CS; tỷ lệ AS tăng, AU
   giảm theo risk, khớp với tỷ lệ thô; 13–22% quỹ đạo không khớp chiến lược
   thuần túy nào.
6. Phát hiện nổi bật nhất: **persona rủi ro áp đảo hoàn toàn tín hiệu risk
   thực tế** ở 5/6 mức persona (hàm bậc thang 0%/100%), chỉ persona "trung
   tính" mới nhạy với `pr` thực tế — nhưng cỡ mẫu rất nhỏ (n=2 race/ô, không
   đưa vào hồi quy được), cần xác nhận lại với run confirmatory trước khi
   kết luận chắc chắn.
