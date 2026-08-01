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

Script tái tạo số liệu: [analyze_nplayer_results.py](analyze_nplayer_results.py).

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

Tính stationary distribution (giới hạn small-mutation, Z=100, β=0.1) với đúng
tham số cơ chế trên:

| pr | Lý thuyết: tần suất AU tại stationary | Thực nghiệm: tỷ lệ UNSAFE mỗi lượt |
|---|---|---|
| 0.1 | 100% | 69.1% |
| 0.6 | 100% | 57.2% |
| 0.9 | 0% | 46.6% |

Lý thuyết dự đoán một **bước nhảy sắc nét** (AU thống trị hoàn toàn ở pr≤0.6,
biến mất hoàn toàn ở pr=0.9), trong khi dữ liệu thực nghiệm cho thấy độ giảm
**tuyến tính/thoải dần**, không có bước nhảy tập trung giữa 0.6 và 0.9. Đây
**không phải mâu thuẫn** — hai đại lượng đo hai thứ khác nhau về bản chất:
lý thuyết mô tả trạng thái mà một *quần thể* tác nhân hội tụ tới sau rất
nhiều thế hệ chọn lọc tự nhiên (Z=100, β=0.1), còn dữ liệu là hành vi suy
luận trong-ngữ-cảnh của *một* LLM trong từng ván riêng lẻ — không có lý do
tiên nghiệm để hai thứ khớp về độ lớn hay hình dạng chuyển tiếp, chỉ có thể
kỳ vọng khớp về *hướng* (đã khớp).

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

## 5. Tóm tắt

1. Cơ chế hoạt động đúng thiết kế (CRN khớp tuyệt đối, setback rate tăng đúng
   theo `max_private_risk`, 0% parse failure).
2. Baseline: UNSAFE giảm dần khi rủi ro tăng, nhưng thoải hơn nhiều so với dự
   đoán "bước nhảy sắc nét" của mô hình lý thuyết quần thể — không mâu thuẫn,
   vì hai mô hình đo hai cơ chế khác nhau (chọn lọc tiến hóa dài hạn vs suy
   luận trong-ngữ-cảnh của một model).
3. Mọi người chơi luôn mở màn UNSAFE ở vòng 1, bất kể risk; phân hóa hành vi
   chỉ xuất hiện từ vòng 2.
4. Phát hiện nổi bật nhất: **persona rủi ro áp đảo hoàn toàn tín hiệu risk
   thực tế** ở 5/6 mức persona (hàm bậc thang 0%/100%), chỉ persona "trung
   tính" mới nhạy với `pr` thực tế — nhưng cỡ mẫu rất nhỏ (n=2 race/ô), cần
   xác nhận lại với run confirmatory trước khi kết luận chắc chắn.
