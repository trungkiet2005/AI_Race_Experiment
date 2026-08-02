# Kịch bản thuyết trình — `progress_1.tex`

Kịch bản đi theo **đúng thứ tự frame** trong `progress_1.tex` (41 trang: 32 frame nội dung
+ 9 trang chia mục tự động do metropolis sinh ra từ mỗi lệnh `\section{}`). Với mỗi frame:

- **Thành phần trên slide** — liệt kê từng thành phần thị giác xuất hiện (chart, bảng, chip,
  màu sắc) và ý nghĩa của nó, để người trình bày biết đang chỉ tay vào đâu.
- **Nói gì** — đoạn kịch bản nói tự nhiên, đọc thành tiếng vừa khớp thời lượng 1 slide.
- **Lưu ý** — mẹo trình bày / điểm dễ bị hỏi ngược, chỉ ghi khi cần.

Tổng thời lượng ước tính: **~25–28 phút nói** (không tính Q&A), phù hợp báo cáo tiến độ.
Có thể cắt bớt phần Backup (3 slide cuối) nếu chỉ có 20 phút.

---

## Mở đầu

### Trang 1 — Title slide
**Thành phần trên slide:**
- Nền chia đôi: mảng tối `RaceInk` bên trái, dải `RaceCyan` bên phải — mô-típ thị giác gốc từ
  deck protocol cũ, giữ nguyên bản sắc thương hiệu của dự án.
- Dòng eyebrow nhỏ màu cyan: *"LLM BEHAVIOR × STRATEGIC SAFETY — SO SÁNH XUYÊN MÔ HÌNH 4 MODEL"*.
- Tiêu đề lớn màu trắng: câu hỏi cốt lõi của cả bài nói.
- Subtitle mờ hơn: định vị đây là báo cáo tiến độ, không phải kết luận cuối.
- Chip đen `PILOT — CHƯA CONFIRMATORY` (nền đen, chữ `RaceLime`) — chip này sẽ lặp lại xuyên
  suốt bài, đây là lần xuất hiện đầu tiên, cần nói rõ ngay từ giây đầu.

**Nói gì:**
"Chào mọi người. Hôm nay mình báo cáo tiến độ dự án AI Race — câu hỏi mình đặt ra là: LLM có
hành xử như người khi bị đặt vào một cuộc đua tốc độ-vs-an toàn không? Mình đã có 4 lần chạy
pilot với 4 họ model khác nhau, và có kết quả để chia sẻ. Nhưng ngay từ đầu, xin nhấn mạnh chip
này: **tất cả đều là dữ liệu pilot, chưa phải confirmatory** — nghĩa là mọi con số hôm nay đọc
như tín hiệu định hướng, không phải bằng chứng cuối cùng cho bất kỳ kết luận nào."

**Lưu ý:** đặt kỳ vọng đúng ngay slide 1 — tránh để khán giả nghĩ đây là kết quả đã chốt.

---

### Trang 2 — Nội dung trình bày
**Thành phần trên slide:** `\tableofcontents` tự sinh từ 9 lệnh `\section` — danh sách 9 mục
lớn, đánh số, màu chữ `RaceInk` trên nền giấy `RacePaper`.

**Nói gì:**
"Cấu trúc bài nói hôm nay: đầu tiên nhắc nhanh khung thí nghiệm, rồi giới thiệu 4 pilot, đi
qua kết quả từng pilot một — baseline, frontier Gemini, OpenAI, N-player — rồi phần quan trọng
nhất là tổng hợp xuyên mô hình, và kết bằng giới hạn với việc cần làm tiếp."

---

## Section 1 — Nhắc nhanh khung thí nghiệm

*(Trang chia mục tự động — chỉ hiện tên mục "Nhắc nhanh khung thí nghiệm" với thanh progress
bar. Nói lướt 3–5 giây: "Phần này nhắc lại nhanh cơ chế game cho ai chưa quen.")*

### Trang 3 — Một vòng chơi, 5 bước, rủi ro riêng cho người thắng
**Thành phần trên slide:**
- Cột trái: sơ đồ TikZ 5 hộp nối bằng mũi tên — QUAN SÁT → CHỌN → CỬA (ẩn) → CẬP NHẬT → DỪNG?
  (hộp cuối viền cam `RaceAmber` để nhấn mạnh đây là điểm rẽ nhánh xác suất).
- Công thức $q_i = p_{\max}\cdot n_i^U/T$ ngay dưới sơ đồ — công thức rủi ro hiệu dụng.
- Chú thích nhỏ: rủi ro chỉ áp dụng cho người thắng/hòa thắng, tại thời điểm dừng.
- Cột phải: block liệt kê 4 câu hỏi nghiên cứu gốc (đánh số 1–4).
- Exampleblock cuối: trích dẫn nguồn paper gốc.

**Nói gì:**
"Nhắc nhanh luật chơi: mỗi vòng, hai bên cùng lúc quan sát trạng thái, chọn Safe hoặc Unsafe
trong khi hành động của nhau vẫn ẩn, rồi cả hai lộ diện cùng lúc, cập nhật tiến độ và điểm, và
sau vòng thứ 5 thì mỗi vòng có 20% khả năng dừng. Unsafe đi nhanh hơn nhưng cộng dồn rủi ro
riêng — công thức ở đây — và rủi ro đó chỉ 'nổ' vào người thắng hoặc hòa thắng lúc kết thúc.
Từ luật chơi này, paper gốc và mình đặt ra 4 câu hỏi: mức rủi ro có đổi hành vi không, đối thủ
Unsafe vòng trước có kéo mình theo không, bị dẫn trước có đẩy mình liều hơn không, và vòng 1 có
ảnh hưởng lâu dài không. Bốn câu hỏi này là khung xuyên suốt toàn bộ phần sau."

---

## Section 2 — Phạm vi 4 lần chạy pilot

*(Trang chia mục tự động.)*

### Trang 4 — 4 pilot, 4 họ model, quy mô rất khác nhau
**Thành phần trên slide:**
- 4 thẻ (beamercolorbox) xếp ngang, mỗi thẻ 1 pilot: `BASELINE`, `FRONTIER`, `OPENAI`,
  `N-PLAYER` — mỗi thẻ có chip tên ở đầu, tên model, số race lớn màu `RaceCyanDark`, số quyết
  định, và 1 dòng ghi chú nhỏ (loại thí nghiệm/backend).
- Alertblock cuối trang: nhắc lại `run_phase = pilot` cho cả 4.

**Nói gì:**
"Bốn pilot này rất khác nhau về quy mô. Baseline là 2 người chơi, Qwen đấu Gemma, 60 race.
Frontier là 3 model họ Gemini cộng 5 ô persona, 177 race. OpenAI là bộ dữ liệu lớn nhất —
2.640 race, gần 50 nghìn quyết định, vì có thêm ma trận persona rủi ro 6 nhân 6. Và N-player
là bản mở rộng sang 3 người chơi cùng lúc, vẫn dùng Qwen. Điểm chung: **cả 4 đều là pilot**,
chưa cái nào đạt trạng thái confirmatory."

**Lưu ý:** nếu bị hỏi "sao không gộp chung phân tích 4 cái này" — trả lời: prompt/protocol
khác nhau giữa các run nên analyser từ chối gộp, đây là thiết kế có chủ đích, không phải thiếu
sót.

### Trang 5 — Cổng chất lượng: đường ống chạy đúng thiết kế
**Thành phần trên slide:**
- Bảng 5 hàng × 4 cột (baseline/frontier/openai/nplayer): parse failure, đẳng thức ΔS, cân
  bằng ghế, `check_symmetry.py`, CRN block.
- Icon `\cmark`/`\xmark` màu xanh `\good`/đỏ `\bad` trước mỗi giá trị — quét nhanh bằng mắt.
- Duy nhất 1 ô đỏ: `check_symmetry.py` của baseline (65%, vượt ngưỡng 40%).
- Exampleblock giải thích: cơ chế đúng ở cả 4, cổng fail duy nhất có nguyên nhân đã biết.

**Nói gì:**
"Trước khi đọc số hành vi, phải kiểm tra đường ống có chạy đúng không. Bảng này là cổng chất
lượng: 0 lỗi parse ở cả 4 run, đẳng thức trạng thái ΔS khớp tuyệt đối, ghế cân bằng. Duy nhất
một ô đỏ — symmetry check của baseline, 65% race hòa từ đầu đến cuối, vượt ngưỡng 40%. Nhưng
đừng lo, nguyên nhân đã biết và sẽ giải thích ngay slide sau: hoàn toàn do Gemma, không phải
lỗi engine."

---

## Section 3 — Baseline 2-player: Qwen2.5-14B vs Gemma-3-12B

*(Trang chia mục tự động.)*

### Trang 6 — Gemma-3-12B: sụp hoàn toàn về Safe
**Thành phần trên slide:**
- Cột trái: alertblock "Triệu chứng" — con số 558/558 tô đỏ `\bad`; block "Đường ống đã loại
  trừ" — 5 dòng checklist xanh, mỗi dòng loại 1 nguyên nhân kỹ thuật khả dĩ.
- Cột phải: bar chart 3 cột (risk 0,1/0,6/0,9), cả 3 đều **bằng 0** — cố tình vẽ phẳng tuyệt
  đối để gây ấn tượng thị giác "không có gì để đo".
- Exampleblock dưới: 2 giả thuyết (A)/(B) đối lập, chưa phân biệt được, kèm phép thử đề xuất.

**Nói gì:**
"Bắt đầu bằng phát hiện gây bất ngờ nhất: Gemma-3-12B, trong 558 lượt gọi, trả về **đúng cùng
một câu trả lời** — 'ACTION: SAFE' — ở cả ba mức rủi ro. Biểu đồ bên phải vẽ phẳng ở 0 vì đúng
nghĩa đen là 0. Mình đã loại trừ 5 nguyên nhân kỹ thuật khả dĩ — parse lỗi, seed không áp dụng,
prompt sai, không sampling, model nạp lỗi — tất cả đều **không phải** nguyên nhân. Vậy còn lại
hai khả năng: hoặc Gemma thực sự hiểu luật chơi và kết luận Safe là tối ưu, hoặc nó từ chối
sinh ra token 'UNSAFE' vì safety-tuning, không hề cân nhắc payoff. Dữ liệu hiện tại **không đủ
để phân biệt** hai giả thuyết này — cần một phép thử rẻ tiền: đổi nhãn hành động thành
OPTION_A/OPTION_B, nếu Gemma bắt đầu chọn B thì đó là giả thuyết (B)."

**Lưu ý:** đây là finding "âm tính" nhưng đáng nói — đừng để nghe như thất bại, mà là một kết
quả chưa diễn giải được.

### Trang 7 — Qwen2.5-14B: chuyển pha đúng vòng 5
**Thành phần trên slide:**
- Bar chart 10 cột (vòng 1–10), 2 màu: xanh dương đậm cho vòng 1–4 (toàn 0), cam cho vòng 5–10
  (giá trị dao động 0,19–0,69).
- Đường kẻ đứt nét dọc ở giữa vòng 4 và 5 — đánh dấu ranh giới `minRounds=5`.
- Exampleblock diễn giải: Safe tuyệt đối là 36% mẫu panel, tất định hoàn toàn.

**Nói gì:**
"Ngược lại, Qwen cho một tín hiệu rất sạch. Nhìn biểu đồ: vòng 1 đến 4, tỷ lệ Unsafe là 0 —
tuyệt đối, ở cả 30 race. Rồi đúng vòng 5 — đường kẻ đứt này — tỷ lệ Unsafe bật lên. Vòng 5
chính là `minRounds`, vòng đầu tiên mà race **có thể** kết thúc theo luật dừng 20%. Qwen đang
đọc đúng luật chơi và trì hoãn rủi ro cho tới khi nó thực sự có ý nghĩa — đây là một chiến lược
endgame hợp lý, không phải ngẫu nhiên."

### Trang 8 — Bị dẫn trước ⇒ Unsafe mạnh
**Thành phần trên slide:**
- Grouped bar chart: 3 nhóm theo risk (0,1/0,6/0,9), mỗi nhóm 3 cột — dẫn trước (xanh), hòa
  (xám), bị dẫn (cam) — cột "bị dẫn" luôn cao vượt trội trong cả 3 nhóm.
- Alertblock: chênh lệch bị-dẫn trừ dẫn-trước, 3 con số đều tô đỏ `\bad` (dùng đỏ ở đây để nhấn
  mạnh độ lớn của hiệu ứng, không phải nghĩa "xấu").

**Nói gì:**
"Đây là phát hiện trung tâm của paper gốc, và nó tái tạo rất rõ ở Qwen. Ba nhóm cột theo mức
rủi ro; trong mỗi nhóm, cột cam — bị dẫn trước — luôn cao vượt hẳn cột xanh — dẫn trước. Chênh
lệch là 0,38, 0,81, và 0,50 điểm phần trăm, nhất quán ở cả ba mức rủi ro. Nói cách khác: khi
Qwen đang thua, nó liều hơn hẳn — đúng logic 'đằng nào cũng thua thì đánh cược' của paper
người."

### Trang 9 — Chia lượt, không phải quán tính
**Thành phần trên slide:**
- Ma trận 2×2 tô màu theo cường độ: hàng = hành động của mình vòng trước, cột = hành động của
  đối thủ vòng trước. Ô (Safe, Unsafe) tô cam đậm với giá trị 0,892 in đậm; ô (Unsafe, Unsafe)
  tô xanh rất nhạt với giá trị 0,048.
- Mỗi ô có thêm cỡ mẫu `n=` nhỏ bên dưới số chính.

**Nói gì:**
"Ma trận này trả lời câu hỏi: Qwen có 'quán tính' hành động không, hay phản ứng theo đối thủ?
Đọc ô cam đậm: khi mình vừa Safe và đối thủ vừa Unsafe, có tới 89,2% khả năng mình chuyển sang
Unsafe ở vòng sau. Nhưng nhìn ô cuối: khi mình vừa Unsafe xong, chỉ 4,8% khả năng mình tiếp tục
Unsafe — gần như chắc chắn lùi về Safe. Đây là **hành vi chia lượt** — đánh một nhịp rồi rút —
chứ không phải cứ liều là liều tiếp."

### Trang 10 — Hệ số hồi quy Qwen (đặc tả 2)
**Thành phần trên slide:**
- Horizontal bar chart (xbar) 5 biến, mỗi thanh có nhãn số ngay đầu thanh (`nodes near coords`).
  Màu đỏ cho hệ số âm, xanh lá cho hệ số dương duy nhất (`opponent(t-1)=U`, +2,339).
- Đường thẳng đứng tại 0 để mắt dễ so trục.
- Alertblock cảnh báo: chỉ 10 CRN cluster, và `own(t-1)` trái ngược paper người.

**Nói gì:**
"Hồi quy panel xác nhận lại câu chuyện định tính vừa rồi bằng con số: hệ số của
`opponent(t-1)=U` là dương rất mạnh, +2,339 — đối thủ Unsafe kéo mình theo. Nhưng chú ý thanh
đỏ `own(t-1)=U`, âm rất mạnh −2,234 — nghĩa là **hành động của chính mình vòng trước dự báo
NGƯỢC** cho vòng này, đúng với ma trận chia-lượt vừa xem. Điểm cần cẩn thận: chỉ có 10 cluster
CRN, nên sai số chuẩn ở đây lạc quan hơn thực tế — đọc dấu và độ lớn như tín hiệu, chưa phải
ước lượng chắc chắn."

---

## Section 4 — Frontier: họ Gemini

*(Trang chia mục tự động.)*

### Trang 11 — φ_U theo rủi ro — Gemini giảm đơn điệu mạnh, ngược người
**Thành phần trên slide:**
- Line chart 4 đường: đường xám nét đứt (Người, gần như phẳng ~0,58 — vẽ làm đường tham chiếu)
  và 3 đường liền màu khác nhau cho 3 model Gemini, cả 3 đều dốc xuống mạnh từ risk 0,1 đến 0,9.
- Legend dưới trục liệt kê tên từng model.

**Nói gì:**
"Đây là biểu đồ Fig 2A tương đương của paper, áp cho Gemini. Đường xám đứt nét là người —
gần như phẳng, vì phát hiện quan trọng của paper người là 0,6 với 0,9 không khác nhau. Nhưng
nhìn ba đường Gemini: **dốc xuống rất mạnh và đơn điệu** — càng rủi ro cao, càng ít Unsafe. Ở
risk thấp 0,1, cả ba model gần như bão hòa ở đỉnh 1,0. Đây là hướng **ngược hẳn** với phát hiện
null quan trọng nhất của paper gốc."

### Trang 12 — Persona: risk-averse khóa cứng, "adversarial" không đơn điệu
**Thành phần trên slide:**
- Horizontal bar chart 6 persona cell, sắp xếp tăng dần theo giá trị trung bình φ_U — từ
  `R−` (thấp nhất) đến `S_AA` (cao nhất, tô đỏ để nhấn mạnh cực trị).
- Mỗi thanh có nhãn số ở đầu.

**Nói gì:**
"Persona ảnh hưởng hành vi rất rõ — risk-averse thấp nhất, cả-hai-ghế-adversarial cao nhất, hợp
lý. Nhưng có một nghịch lý: `S_AC` — ghế được gán vai adversarial nhưng đối thủ lại cooperative
— lại **thấp hơn cả** `S_CA`, ghế cooperative đối diện. Chú ý: đây chỉ là mô tả thô, vì mỗi
persona chạy ở một protocol signature riêng, không tách được khỏi nhiễu batch — nên không kết
luận nhân quả từ biểu đồ này."

### Trang 13 — Hồi quy panel Gemini (đặc tả 6)
**Thành phần trên slide:**
- Xbar 6 biến chính, màu đỏ cho 2 hệ số treatment (risk 0,6 và 0,9, đều âm mạnh), xanh lá cho
  `Unsafe vòng 1` và `opponent(t-1)=U` (cả hai dương, khớp hướng người).

**Nói gì:**
"Hồi quy panel đầy đủ cho Gemini cho hai tin tốt và một tin cần chú ý. Tin tốt: `opponent(t-1)`
dương +1,486 — khớp hướng người (+0,607) nhưng mạnh gấp 2,4 lần; và hành động vòng 1 cũng dự
báo dương rất mạnh, khớp 'behavioural momentum' của paper. Tin cần chú ý — hai thanh đỏ: hệ số
treatment (risk 0,6 và 0,9) có ý nghĩa thống kê rất mạnh, ngược hẳn với paper người, nơi
treatment không có ý nghĩa."

### Trang 14 — 8 hiệu ứng người–LLM (E1–E8): 4/8 replicated
**Thành phần trên slide:**
- Lưới 2×4 "chip" màu — 4 chip xanh lá (E1, E3, E6, E7 — replicated), 4 chip đỏ (E2, E4, E5,
  E8 — not replicated). Mỗi chip có mã hiệu ứng, mô tả ngắn, kết quả.
- Exampleblock cảnh báo cách đọc: "replicated" chỉ xét dấu, không xét độ lớn.

**Nói gì:**
"Tổng kết lại bằng bảng điểm 8 hiệu ứng chuẩn hóa từ paper người — mã E1 đến E8. Quét nhanh
bằng màu: 4 xanh, 4 đỏ, đúng 4 trên 8. Nhưng lưu ý quan trọng: 'replicated' ở đây chỉ nghĩa là
**đúng dấu và đủ ý nghĩa thống kê**, không có nghĩa độ lớn bằng người — ba trong bốn ô xanh có
hệ số lớn hơn người từ 2 đến 9 lần."

---

## Section 5 — OpenAI: GPT-5-nano / GPT-5.4-nano

*(Trang chia mục tự động.)*

### Trang 15 — φ_U theo rủi ro — hình chữ U, không đơn điệu
**Thành phần trên slide:**
- Line chart 3 đường: người (xám đứt nét, tham chiếu), gpt-5-nano (xanh dương, biên độ nhỏ),
  gpt-5.4-nano (cam, biên độ lớn hơn, rõ hình chữ U — thấp nhất ở risk=0,6).

**Nói gì:**
"Sang bộ dữ liệu lớn nhất — OpenAI. Hình dạng ở đây khác hẳn cả người lẫn Gemini: **hình chữ
U**. Cả hai model GPT-nano đều Unsafe ít hơn ở risk=0,6 so với hai đầu mút 0,1 và 0,9. Với
`gpt-5.4-nano`, kiểm định trên mẫu từ vòng 2 cho thấy điểm trũng ở 0,6 này **có ý nghĩa thống
kê thật**, không phải nhiễu — nhưng khác Gemini, không có model nào bão hòa gần 1,0."

### Trang 16 — Persona: thứ hạng nhất quán trên 2 model độc lập
**Thành phần trên slide:**
- Grouped bar chart 8 persona cell × 2 model (xanh = gpt-5-nano, cam = gpt-5.4-nano), sắp theo
  thứ tự tăng dần hợp lý: S_CC thấp nhất → R+ cao nhất.

**Nói gì:**
"Thứ hạng persona ở đây hợp lý nhất trong toàn bộ 4 pilot, và quan trọng là **nhất quán trên cả
hai model độc lập**: risk-averse thấp nhất, risk-seeking cao nhất, cả-hai-ghế-cooperative gần
0. Nhưng có một hiệu ứng lạ: `gpt-5-nano` sụp về Safe với **bất kỳ** persona nào, kể cả R0
trung lập hoàn toàn — từ 0,14 baseline xuống 0,015. Đây là hiệu ứng 'có persona hay không', chứ
không phải hướng nội dung persona."

### Trang 17 — Ma trận rủi ro 6×6
**Thành phần trên slide:**
- Lưới 6×6 (+ 1 hàng/cột tiêu đề) tô màu gradient cam theo cường độ giá trị: góc trên-trái gần
  trắng (φ_U≈0), góc dưới gần cam đậm (φ_U≈0,98–0,99).
- Hai callout hai bên dưới: "Hàng = mức của MÌNH" (dose-response tăng mạnh) và "Cột = mức của
  ĐỐI THỦ" (giảm nhẹ khi đọc ngang một hàng).

**Nói gì:**
"Đây là trục dữ liệu chỉ OpenAI mới có — ma trận persona rủi ro 6 nhân 6 theo thang
Eckel-Grossman. Đọc theo hàng từ trên xuống: mức rủi ro của **chính mình** đẩy φ_U từ gần 0 lên
gần 1,0 — dose-response rất sạch, sạch hơn cả người, vì ở người thang đo này **không** dự báo
được gì. Nhưng đọc ngang một hàng, từ trái sang phải: khi đối thủ 'ưa rủi ro' hơn, màu nhạt dần
một chút — mình lại **thận trọng hơn**. Hai phát hiện độc lập này khớp chính xác với dấu âm của
hệ số `opponent_prev` ở slide tiếp theo."

### Trang 18 — Hồi quy panel GPT-nano: đảo dấu hoàn toàn
**Thành phần trên slide:**
- Xbar 5 biến, hai thanh đỏ nổi bật: `opponent(t-1)=U` (−1,016) và `ΔS` (+0,490) — cả hai đều
  ngược dấu so với slide tương ứng của Gemini/baseline.
- Alertblock cảnh báo hội tụ: `converged: False` cho cả 6 đặc tả.

**Nói gì:**
"Đây là phát hiện nổi bật nhất của toàn bộ đợt phân tích. Ở người, đối thủ Unsafe kéo mình
theo — dương 0,607. Ở Gemini cũng vậy, dương 1,486. Nhưng ở GPT-nano — thanh đỏ đầu tiên —
**âm 1,016**: đối thủ Unsafe làm mình **giảm** xác suất Unsafe, phòng thủ thay vì leo thang.
Thanh đỏ thứ hai — ΔS dương 0,490 — cũng đảo dấu: dẫn trước làm **tăng** Unsafe, tức củng cố
lợi thế thay vì bảo toàn nó, ngược hoàn toàn với người. Nhưng — quan trọng — statsmodels báo
`converged: False` cho cả 6 đặc tả, vì có 88 dummy protocol signature gây quasi-separation. Nên
đọc đây như **tín hiệu thô**, chưa phải hệ số ổn định."

**Lưu ý:** đây là slide dễ bị hỏi nhiều nhất — chuẩn bị sẵn câu trả lời cho "vậy kết luận gì
được từ đây?" → "chưa kết luận được, cần chạy lại với protocol sạch hơn."

### Trang 19 — 8 hiệu ứng người–LLM (E1–E8): chỉ 2/8
**Thành phần trên slide:**
- Cùng bố cục chip 2×4 như trang 14, nhưng chỉ 2 chip xanh (E3, E5), 6 chip đỏ — hai chip đầu
  (E1, E2) có thêm chữ "đảo dấu" bên trong để phân biệt với "not replicated" thường.

**Nói gì:**
"So bảng điểm với Gemini: chỉ 2 trên 8 xanh. Nhưng khác biệt không chỉ ở số lượng — hai ô đỏ
đầu tiên, E1 và E2, tôi ghi rõ 'đảo dấu' vì đó là loại sai khác nghiêm trọng hơn hẳn 'not
replicated' thông thường của Gemini, vốn vẫn đúng dấu chỉ là quá mạnh."

---

## Section 6 — N-player (N=3): Qwen2.5-14B

*(Trang chia mục tự động.)*

### Trang 20 — φ_U giảm theo rủi ro (69%→57%→47%)
**Thành phần trên slide:**
- Bar chart đơn giản 3 cột, mỗi cột có nhãn số ngay trên đầu.
- Exampleblock so với lý thuyết quần thể: dốc lý thuyết (sau khi quét β tối ưu) vẫn giảm mạnh
  hơn thực tế.

**Nói gì:**
"Chuyển sang N-player, 3 người chơi cùng lúc. φ_U giảm đơn điệu và có ý nghĩa thống kê mạnh
theo rủi ro — 69, 57, rồi 47%. Đáng chú ý: mình đã **quét** tham số β của mô hình lý thuyết
quần thể để tìm điểm khớp tốt nhất, chặt hơn cách chỉ chọn 2 điểm cố định như ở frontier — và
ngay cả vậy, lý thuyết vẫn dự báo dốc giảm **mạnh hơn** thực tế quan sát."

### Trang 21 — Persona Eckel-Grossman lấn át tín hiệu rủi ro thực
**Thành phần trên slide:**
- Sơ đồ TikZ minh họa dạng bậc thang (không phải chart số liệu chính xác) — đường đỏ nhảy giữa
  hai mức thấp/cao, chấm xanh đánh dấu ngoại lệ "mức trung tính".
- Alertblock nhắc: đây là minh họa định tính, cỡ mẫu chỉ 2 race/ô.

**Nói gì:**
"Slide này cố tình vẽ minh họa chứ không phải chart số liệu chính xác — vì cỡ mẫu chỉ 2
race/ô, không đưa vào hồi quy được. Nhưng xu hướng định tính rất rõ: 5 trên 6 mức persona
Eckel-Grossman cho hành vi gần như **bậc thang tuyệt đối** — 0% hoặc 100% — lấn át hoàn toàn
tín hiệu rủi ro thực `p_max`. Chỉ riêng mức 'trung tính' — chấm xanh này — vẫn còn nhạy với rủi
ro thực. Cần xác nhận lại bằng một run có đủ repetition."

---

## Section 7 — Tổng hợp xuyên mô hình

*(Trang chia mục tự động — nhấn giọng: "Đây là phần quan trọng nhất của bài nói.")*

### Trang 22 — Từ 4 mảnh rời rạc đến một câu hỏi chung
**Thành phần trên slide:**
- TikZ flow diagram: 4 hộp viền đỏ xếp dọc bên trái (tóm tắt 1 dòng mỗi pilot), tất cả mũi tên
  cam hội tụ vào 1 hộp xanh lá bên phải — câu hỏi tổng hợp.

**Nói gì:**
"Bốn pilot vừa xem, mỗi cái kể một câu chuyện khác nhau — Gemma tắt tiếng, Qwen chia lượt,
Gemini giảm đơn điệu, GPT-nano hình chữ U và đảo dấu, N-player thì persona lấn át risk thật.
Câu hỏi tự nhiên là: gộp lại, có tồn tại **một hành vi LLM chung** khi đối mặt rủi ro không?
Ba slide tiếp theo trả lời câu hỏi này bằng ba cách nhìn khác nhau — và câu trả lời sơ bộ là
không."

### Trang 23 — φ_U(risk) không cùng hình dạng giữa các nguồn
**Thành phần trên slide:**
- Line chart 5 đường chồng lên nhau: người (xám đứt nét, phẳng), Gemini (xanh dương, dốc),
  GPT-nano (cam, chữ U), Qwen 2-player (đỏ, thấp và tương đối phẳng), Qwen 3-player (xanh lá,
  giảm nhẹ).

**Nói gì:**
"Đây là hình gộp cả 5 nguồn dữ liệu vào một trục. Nhìn 5 đường: 5 hình dạng khác nhau hoàn
toàn — phẳng, dốc đơn điệu, chữ U, thấp-ổn định, giảm nhẹ. Không có hai đường nào giống nhau.
Thông điệp: chọn dùng model nào để mô phỏng 'tác nhân AI' trong một nghiên cứu là đang chọn cả
một **hình dạng hành vi** riêng biệt, không chỉ chọn một con số trung bình đại diện."

### Trang 24 — Dấu hệ số opponent_prev và ΔS đảo chiều theo model
**Thành phần trên slide:**
- Hai panel song song, mỗi panel là diverging bar chart quanh trục 0: trái là hệ số
  `opponent(t-1)`, phải là hệ số `ΔS`. Mỗi panel có 5 thanh (Người, Gemini, GPT-nano, Qwen 2p,
  Qwen 3p). Màu xanh lá = cùng dấu người, đỏ = đảo dấu, thanh "Người" luôn tô đen làm mốc.
- Alertblock: "Ảnh quan trọng nhất của đợt này."

**Nói gì:**
"Đây là hình tôi coi là quan trọng nhất buổi báo cáo hôm nay. Bên trái: hệ số của hành động
đối thủ vòng trước, 5 nguồn. Bốn trong năm thanh màu xanh — cùng dấu dương với người. Duy nhất
một thanh đỏ: GPT-nano, âm. Bên phải: hệ số ΔS — vị thế đua. Ba thanh xanh cùng dấu âm với
người, một thanh xám gần 0 không có ý nghĩa của Qwen 3-player, và một thanh đỏ — lại là
GPT-nano, dương. **Chỉ một model, GPT-nano, đảo dấu cả hai hiệu ứng trung tâm của paper gốc
cùng lúc** — nhưng như đã nói, hồi quy đó không hội tụ, nên đây là tín hiệu cần xác nhận lại,
chưa phải kết luận."

**Lưu ý:** đây là slide đáng dừng lâu nhất — cho khán giả thời gian đọc cả hai panel trước khi
nói tiếp.

### Trang 25 — Tỷ lệ tái tạo 8 hiệu ứng người
**Thành phần trên slide:**
- Bar chart 2 cột đơn giản: Gemini = 4, GPT-nano = 2, trên thang /8, có nhãn số trên đầu cột.

**Nói gì:**
"Tóm số lại: Gemini tái tạo được 4 trên 8 hiệu ứng chuẩn của người, GPT-nano chỉ 2 trên 8. Và
như đã nói ở hai slide trước đó, không chỉ khác nhau về **số lượng** — 4 chỗ không-khớp của
Gemini vẫn đúng dấu, chỉ là quá mạnh; còn GPT-nano có 2 trong 6 chỗ không-khớp là **đảo dấu
hoàn toàn**. Không thể nói 'LLM hành xử như người', mà cũng không thể nói 'các LLM hành xử
giống nhau'."

---

## Section 8 — Giới hạn & việc cần làm tiếp

*(Trang chia mục tự động.)*

### Trang 26 — Giới hạn cần giữ khi diễn giải
**Thành phần trên slide:**
- Hai cột alertblock song song: "Thống kê" (4 gạch đầu dòng) và "Đo lường" (3 gạch đầu dòng).
- Exampleblock cuối: nguyên tắc báo cáo — không dùng deck này làm bằng chứng cho PROJECT.md.

**Nói gì:**
"Trước khi sang việc cần làm, liệt kê rõ giới hạn để không ai mang số liệu hôm nay đi trích dẫn
sai. Về thống kê: tất cả vẫn là pilot; hồi quy OpenAI không hội tụ; baseline và N-player chỉ có
10 đến 20 cluster CRN nên sai số chuẩn lạc quan; persona N-player chỉ 2 race/ô. Về đo lường:
persona bị confound với protocol ở cả hai run frontier và openai — nhưng vì hai nguyên nhân cấu
trúc khác nhau; và biến `first_round_unsafe` là hằng số ở nhiều model, làm nhiều đặc tả hồi quy
bị singular. Nguyên tắc xuyên suốt: đây là tín hiệu pilot, không phải bằng chứng confirmatory
cho bất kỳ câu hỏi nghiên cứu nào trong PROJECT.md."

### Trang 27 — Việc cần làm tiếp
**Thành phần trên slide:**
- Danh sách đánh số 1–5, in đậm tên hành động đầu mỗi mục.
- Exampleblock cuối: giải thích thứ tự ưu tiên — 2 mục đầu chặn đường.

**Nói gì:**
"Năm việc cần làm, theo thứ tự ưu tiên. Một và hai chặn đường trước tiên: phép thử đổi nhãn
hành động cho Gemma, và tạo biến thiên ở vòng 1 để hết bị hằng số. Ba là chạy lại persona trong
cùng một session hoặc qua kaggle_benchmarks để gỡ confound protocol. Bốn là bổ sung
mixed-effects logit và EGTtools để mở khóa các mục còn thiếu. Và cuối cùng — chỉ sau khi bốn
bước trên xong — mới khóa protocol lại và chạy confirmatory thật. Chạy thêm repetition ngay bây
giờ, trước khi giải quyết mục 1 và 2, chỉ nhân bản vấn đề lên, không thêm thông tin gì mới."

---

## Section 9 — Kết

*(Trang chia mục tự động.)*

### Trang 28 — [standout] Bốn model, bốn hành vi khác nhau
**Thành phần trên slide:** frame `[standout]` của metropolis — chữ lớn giữa trang, không có
block/chart, chỉ 2 dòng: câu kết luận chính (dòng trên) và câu định hướng đọc kết quả (dòng
dưới, màu `RaceLime`).

**Nói gì:**
"Tóm lại bằng một câu: bốn model, bốn hành vi khác nhau — kể cả đảo dấu hiệu ứng trung tâm của
paper người. Chưa đủ dữ liệu confirmatory để kết luận điều gì chắc chắn — nhưng đã đủ để không
còn giả định rằng LLM hành xử như người, hay thậm chí các LLM hành xử giống nhau."

### Trang 29 — [standout] Cảm ơn
**Thành phần trên slide:** frame standout đơn giản, 1 dòng.

**Nói gì:** "Cảm ơn mọi người đã lắng nghe, rất mong nhận được góp ý và câu hỏi."

*(Dừng ở đây nếu chỉ có 20 phút. Ba slide sau là backup, chỉ mở khi có câu hỏi liên quan.)*

---

## Backup (chỉ trình bày khi được hỏi)

### Trang 30 — Backup 1: Hệ số hồi quy đầy đủ, cả 4 nguồn
**Thành phần trên slide:** bảng dày 6 hàng × 5 cột (Người, Gemini, GPT-nano, Qwen 2p, Qwen 3p),
thu nhỏ bằng `\resizebox`, dòng cuối ghi rõ nguồn từng cột.

**Dùng khi:** ai hỏi "vậy còn own(t-1) và Unsafe vòng 1 thì sao ở tất cả các nguồn?" — bảng này
có đủ số cho cả 5 biến chính, không chỉ 2 biến đã chiếu ở trang 24.

### Trang 31 — Backup 2: Provenance & lệnh tái tạo
**Thành phần trên slide:** liệt kê 4 file báo cáo nguồn, 1 block code lệnh tái tạo mẫu (openai),
alertblock "Freeze before you look" nhắc nguyên tắc khóa protocol.

**Dùng khi:** ai hỏi "số liệu này lấy từ đâu, tái tạo lại được không?" — trả lời bằng lệnh cụ
thể trên slide, không cần nhớ thuộc lòng.

### Trang 32 — Tham chiếu chính
**Thành phần trên slide:** card trích dẫn đầy đủ paper gốc (tên, tác giả, arXiv ID, link), và
1 dòng chỉ tới tài liệu dự án (`CLAUDE.md`, `PROJECT.md`, `results/`).

**Dùng khi:** kết thúc phần Q&A, hoặc ai hỏi "paper gốc là gì, đọc ở đâu."

---

## Ghi chú tổng thể cho người trình bày

- **Nhịp màu xuyên suốt:** xanh lá `\good` = khớp/lành mạnh, đỏ `\bad` = lệch/đảo dấu/vấn đề,
  cam `\hl` = con số cần chú ý (không hẳn tốt hay xấu). Nếu quên số chính xác, chỉ cần nói theo
  màu là khán giả vẫn theo kịp mạch.
- **Slide dễ bị hỏi sâu nhất:** trang 18 (đảo dấu GPT-nano, không hội tụ) và trang 24 (diverging
  bar tổng hợp) — nên chuẩn bị kỹ hai slide này nhất, có thể dừng lâu hơn mức trung bình.
- **Nếu bị cắt thời gian:** có thể bỏ trang 12, 16, 17, 21 (các slide persona chi tiết) mà
  không làm gãy mạch — mạch chính (φ_U theo risk → hồi quy → E1–E8) vẫn đủ để tới phần tổng hợp.
- **Không tự thêm số liệu ngoài deck** khi bị hỏi khó — nếu không có trong 4 báo cáo nguồn hoặc
  backup, trả lời "chưa có dữ liệu để trả lời chính xác, sẽ kiểm tra lại trong `results/`."
