# Plan: Progress-report slide deck (`progress_1.tex`)

## 0. Mục tiêu và phạm vi

Đây **không phải** bản deck "protocol / results pending" (`slides/ai_race_research_deck.tex`
hiện tại). Đó là bài trình bày trước khi chạy — mọi ô kết quả đều để trống `PENDING`. Bây giờ
4 pilot run đã chạy xong và có báo cáo trong `analysis/`, nên `progress_1.tex` là bản
**progress report**: trình bày những gì đã quan sát được, đối chiếu với paper người, và nêu rõ
giới hạn + việc cần làm tiếp — đúng tinh thần "kỷ luật báo cáo" đã đặt ra trong CLAUDE.md
("không dùng dữ liệu pilot làm bằng chứng confirmatory").

Không sửa `ai_race_research_deck.tex` cũ. Deck mới là file độc lập:

```
slides/progress/plan.md          (tài liệu này)
slides/progress/progress_1.tex   (nguồn LaTeX)
slides/progress/progress_1.pdf   (bản build)
```

## 1. Nguồn dữ liệu — 4 lần chạy pilot trong `analysis/`

| Run | File báo cáo | Model | Quy mô | Đặc điểm nổi bật |
|---|---|---|---|---|
| `baseline` (2-player) | `analysis/baseline.md` | qwen2.5-14b-instruct, gemma-3-12b-it | 60 race, 558 quyết định, 10 rep | Gemma sụp 100% Safe; Qwen có chuyển pha đúng vòng 5 |
| `frontier` (2-player) | `analysis/frontier/report.md` | 3× Gemini (proxy Kaggle) + 5 cell persona | 177 race, 3.168 quyết định | φ_U giảm đơn điệu mạnh theo risk — ngược null của người |
| `openai` (2-player) | `analysis/openai/report.md` | gpt-5-nano, gpt-5.4-nano (API trực tiếp) | 2.640 race, 49.104 quyết định, ma trận persona 6×6 | φ_U hình chữ U; **đảo dấu** cả `opponent_prev` lẫn `ΔS` so với người/Gemini |
| `nplayer` (N=3) | `analysis/nplayer/report.md` | qwen2.5-14b-instruct | 60 race baseline + persona Eckel-Grossman | UNSAFE giảm đơn điệu theo risk; persona lấn át hoàn toàn tín hiệu risk thật |

Tất cả đều `run_phase = pilot` — **không phải confirmatory**. Đây là ràng buộc xuyên suốt toàn
bộ deck, phải xuất hiện lặp lại bằng nhãn/chip trực quan, không chỉ một dòng chữ ở đầu.

Số liệu dùng trong slide lấy trực tiếp từ 4 file trên (đã đọc kỹ), không tính lại từ CSV thô —
nếu cần đối chiếu, các bảng số nằm trong `analysis/*/derived/*.csv`.

## 2. Ràng buộc kỹ thuật

- Không có TeX Live trên máy này → cài `basictex` qua Homebrew (đã được xác nhận), sau đó dùng
  `tlmgr` cài thêm các gói còn thiếu (metropolis + phụ thuộc, pgfplots, tikz libraries, v.v.)
- Engine: **xelatex** (dùng `fontspec` để lấy font Unicode hỗ trợ tiếng Việt sẵn có trên macOS,
  ví dụ Helvetica Neue/Arial — không phụ thuộc đường dẫn font cứng như trong `refs.tex` gốc, vì
  máy này không có font Fira ở `/usr/local/texlive/...`).
- Không dùng gói `babel[vietnamese]` (dễ thiếu hyphenation pattern trên basictex tối giản) — gõ
  tiếng Việt trực tiếp bằng UTF-8, tự đặt nhãn thủ công (giống cách `ai_race_research_deck.tex`
  đã làm với tiếng Anh).
- Toàn bộ biểu đồ dựng bằng **TikZ/pgfplots thuần** (giá trị số gõ trực tiếp từ bảng trong 4
  report, đã có sẵn, chính xác) — không sinh ảnh PNG qua Python/matplotlib. Lý do: giữ file tự
  chứa (self-contained), không phụ thuộc thêm pipeline build ảnh, và khớp cách `refs.tex` dựng
  toàn bộ visual bằng TikZ.
- Build 2 lần bằng xelatex để Beamer/metropolis resolve đúng progress bar & page count, giống
  quy ước trong `slides/README.md`.

## 3. Định hướng thiết kế (bám `refs.tex`, giữ bản sắc RaceInk)

`refs.tex` (đề tài Loss-of-Plasticity) là **template về cấu trúc trình bày**, không phải về
màu sắc — chủ đề khác hẳn. Những gì lấy từ `refs.tex`:

1. **Theme `metropolis`**, `progressbar=frametitle`, `block=fill`, section page có progress bar.
2. **Macro highlight ngữ nghĩa**: `\hl{}` (nhấn mạnh chính), `\tc{}` (điểm phụ), `\bad{}` (tín
   hiệu xấu/lệch), `\good{}` (tín hiệu tốt/khớp) — tô màu nhất quán xuyên suốt thay vì bold rời
   rạc.
3. **Khối `alertblock` / `exampleblock` / `block`** dùng có chủ đích: `alertblock` cho tín hiệu
   cảnh báo hoặc bất ngờ, `exampleblock` cho takeaway/diễn giải, `block` cho định nghĩa/bối cảnh
   trung tính.
4. **Bảng nén bằng `resizebox`** khi cần trình bày số dày đặc (hệ số hồi quy, ma trận chuyển
   trạng thái) — chấp nhận là "visualization" vì có tô màu `\bad`/`\good` theo hàng, không phải
   text thuần.
5. **TikZ flow diagram kiểu "trước → sau"** (như slide "Từ benchmark sang phân tích cơ chế") để
   đóng khung các bước chuyển góc nhìn — dùng lại cho slide "từ 4 pilot rời rạc → một câu chuyện
   xuyên mô hình".
6. **Slide "Giới hạn cần giữ khi diễn giải"** và **"Việc cần làm tiếp"** đặt gần cuối, tách biệt
   rõ khỏi phần phát hiện — giữ nguyên vị trí này.
7. Kết bằng `\begin{frame}[standout]`.

Những gì **giữ nguyên từ bản sắc RaceInk đã thiết lập** (không đổi sang deepblue của
`refs.tex`, vì đây là cùng một dự án/khán giả với `ai_race_research_deck.tex`):

- Palette: `RaceInk` (nền tối), `RacePaper` (nền sáng), `RaceCyan`/`RaceCyanDark` (Safe),
  `RaceAmber` (Unsafe/cảnh báo), `RaceLime` (điểm nhấn tích cực), `RaceMuted` (chú thích),
  `RaceRed` (lệch/mismatch nặng).
- Chip nhãn: tái dùng khái niệm `\pending`, `\sourcechip` ("SOURCE STUDY — HUMAN") — thêm chip
  mới `\pilotchip` ("PILOT — NOT CONFIRMATORY") xuất hiện trên **mọi** slide có số liệu LLM, và
  `\llmchip{MODEL}` để gắn tên model lên mỗi biểu đồ.
- Ánh xạ macro `\hl/\tc/\bad/\good` sang palette RaceInk: `\good` = RaceCyan/RaceLime (khớp
  paper người hoặc tín hiệu lành mạnh), `\bad` = RaceAmber/RaceRed (lệch/đảo dấu/không hội tụ).

## 4. Nguyên tắc "ít text, nhiều visualization" — áp dụng cụ thể

Với từng loại số liệu trong 4 report, chọn visual thay vì liệt kê:

| Loại số liệu nguồn | Visual chọn | Vì sao không dùng bảng/text thuần |
|---|---|---|
| φ_U theo risk (0,1/0,6/0,9), nhiều model | Line chart pgfplots, mỗi model 1 đường | So hình dạng (đơn điệu giảm vs. chữ U vs. phẳng) là điểm mấu chốt — mắt đọc hình dạng nhanh hơn đọc 3 số |
| Unsafe theo vòng (vòng 1→10, Qwen 2-player) | Bar/line chart theo vòng | "Chuyển pha đúng vòng 5" là hiệu ứng bậc thang — chart cho thấy ngay, bảng số cần đọc từng ô |
| Bị dẫn/hoà/dẫn trước × risk | Grouped bar chart 3 nhóm | So sánh độ lớn hiệu ứng ΔS trực quan hơn bảng |
| Ma trận chuyển trạng thái 2×2 | Heatmap tô màu theo cường độ + số | Làm nổi bật ô 89% vs 4,8% ngay lập tức |
| Persona ranking (nhiều cell) | Horizontal bar chart sắp theo giá trị | Thấy ngay thứ hạng đơn điệu R− → baseline → R+ |
| Ma trận rủi ro 6×6 (openai) | Heatmap 6×6, tô gradient theo φ_U | Bảng 6×6 số thô rất khó đọc trong 1 slide; heatmap lộ ngay gradient theo hàng (own) và cột (opponent) |
| Hệ số hồi quy (β, SE, p) nhiều biến | Coefficient/forest-style bar (dot + CI ngang hoặc bar có dấu) | Dấu (+/−) và độ lớn là thông điệp, không phải giá trị SE chính xác |
| Đối chiếu người–LLM E1–E8 | Chip grid 8 ô, xanh/cam theo verdict | Thay bảng text "replicated/not replicated" bằng lưới màu quét nhanh |
| So sánh dấu hệ số xuyên 5 nguồn (người, Gemini, GPT-nano, Qwen 2p, Qwen 3p) | Diverging bar chart (âm/dương hai phía trục 0) | Đây là phát hiện quan trọng nhất của cả đợt — cần 1 hình duy nhất kể hết câu chuyện "đảo dấu" |
| Cổng chất lượng (parse failure, ΔS residual, symmetry) | Checklist icon grid, không phải bảng | Đây là "đã pass hết", chỉ cần xác nhận nhanh, không cần đọc số |

Mỗi slide nội dung: **tối đa ~25 từ tiêu đề + 1 hình + 1 khối takeaway ngắn** (theo đúng mật độ
của `refs.tex`, ví dụ slide "F9-A"). Không có slide nào chỉ có bullet list dài quá 4 dòng.

## 5. Cấu trúc & outline chi tiết từng slide

Ước lượng **~27 slide chính + 3 backup**, chia theo section (section page có progress bar như
`refs.tex`).

### Section 0 — Mở đầu
1. **Title** — "AI Race: LLM có hành xử như người không?" / subtitle "Progress report — 4 pilot run, kết quả chưa confirmatory". Chip `\pilotchip` to giữa slide.
2. **Nội dung trình bày** — `\tableofcontents`.

### Section 1 — Nhắc nhanh khung thí nghiệm (1 slide, nén tối đa vì khán giả đã biết)
3. **Cơ chế & câu hỏi gốc** — 1 slide gộp: mini sơ đồ round (Observe→Choose→Commit→Update→Stop?) + công thức `q_i = p_max·n_U/T` + 4 câu hỏi nghiên cứu, lấy lại tinh thần slide 4/5/8 của `ai_race_research_deck.tex` nhưng nén thành 1 slide "recap", không làm lại cả bộ.

### Section 2 — Phạm vi 4 pilot run đã chạy
4. **Bản đồ 4 lần chạy** — bảng/kim tự tháp 4 ô (model, N race, N quyết định), tô màu theo họ model (Qwen/Gemma/Gemini/GPT). Nguồn: đầu mỗi report.md.
5. **Cổng chất lượng dữ liệu** — icon-checklist 4 cột (1 cột/run): parse failure, ΔS identity residual, seat balance, `check_symmetry.py`. Toàn bộ pass trừ 1 case (baseline symmetry FAIL do Gemma) → đánh dấu rõ bằng `\bad`.

### Section 3 — Baseline 2-player: Qwen2.5-14B vs Gemma-3-12B
6. **Gemma: sụp hoàn toàn về Safe** — visual: 558/558 cùng 1 token, φ_U = 0,000 cả 3 risk (bar phẳng ở 0) + đường ống loại trừ nguyên nhân (5 khả năng đã loại, dạng checklist gạch bỏ) + 2 giả thuyết A/B chưa phân biệt được.
7. **Qwen: chuyển pha đúng vòng 5** — line/bar chart unsafe rate theo vòng 1–10 (0,0,0,0 | 0,283,0,521,...) — đường kẻ dọc đánh dấu `minRounds=5`.
8. **Qwen: bị dẫn trước → Unsafe mạnh** — grouped bar (dẫn trước/hoà/bị dẫn) × 3 risk, số 0,769/0,938/0,800 nổi bật bằng `\bad` (đọc là tín hiệu mạnh, tái tạo đúng paper).
9. **Qwen: chia lượt, không phải quán tính** — heatmap 2×2 ma trận chuyển trạng thái (Safe|Unsafe)×(Safe|Unsafe), ô 0,892 và 0,048 làm nổi bật.
10. **Qwen: hệ số hồi quy đặc tả 2** — coefficient bar: `p_max=0,6` (−0,642), `p_max=0,9` (−0,320), `own_prev` (−2,234), `opponent_prev` (+2,339), `ΔS` (−0,717) — kèm cảnh báo "chỉ 10 cluster, đọc như tín hiệu".

### Section 4 — Frontier: họ Gemini
11. **φ_U theo risk — Gemini giảm đơn điệu mạnh** — line chart 3 model, kèm đường tham chiếu ngang "người ≈ 0,58 phẳng" để tương phản.
12. **Persona: risk-averse khoá cứng, "adversarial" không đơn điệu** — horizontal bar ranking 6 persona cell, đánh dấu nghịch lý S_AC thấp hơn cả S_CA.
13. **Hồi quy panel — khớp hướng người nhưng mạnh hơn nhiều** — coefficient bar đặc tả 6, tô `\good` cho các biến khớp dấu người (opponent_prev, first_round), `\bad` cho treatment có ý nghĩa (lệch paper).
14. **8 hiệu ứng người–LLM: 4/8 replicated** — chip grid 8 ô (E1–E8), xanh/cam.

### Section 5 — OpenAI: GPT-5-nano / GPT-5.4-nano (bộ dữ liệu lớn nhất)
15. **φ_U theo risk — hình chữ U, không đơn điệu** — line chart 2 model, tương phản trực tiếp với slide 11 (Gemini monotone) bằng cách đặt cạnh nhau nếu còn chỗ, hoặc nhắc lại hình Gemini mờ làm nền.
16. **Persona ranking nhất quán 2 model độc lập** — bar chart 8 persona × 2 model (double bar), thứ hạng hợp lý R− thấp nhất → R+ cao nhất.
17. **Ma trận rủi ro 6×6 — dose-response cực mạnh theo MÌNH, bù trừ theo ĐỐI THỦ** — heatmap 6×6 (1 cho mỗi model, hoặc chọn gpt-5.4-nano vì tương phản rõ nhất 0,002→0,988), chú thích 2 mũi tên: hàng tăng mạnh, cột giảm nhẹ.
18. **Hồi quy: đảo dấu hoàn toàn — phát hiện nổi bật nhất** — coefficient bar đặc tả 6, `opponent_prev` = −1,016 (tô đỏ, mũi tên "người/Gemini: dương") và `ΔS` = +0,490 (tô đỏ, mũi tên "người: âm"), kèm cảnh báo `converged: False` / quasi-separation bằng chip riêng.
19. **8 hiệu ứng người–LLM: chỉ 2/8, và 2 cái đảo dấu** — chip grid, tô đặc biệt E1/E2 (đảo dấu, khác "not replicated" thường).

### Section 6 — N-player (N=3): Qwen2.5-14B
20. **φ_U giảm theo risk (69→57→47%)** — bar chart 3 risk, kèm p<0,001 note nhỏ.
21. **Persona Eckel-Grossman: hàm bậc thang lấn át tín hiệu risk thật** — step chart hoặc bar 6 mức R1–R6 gần 0%/100%, chỉ persona "trung tính" nhạy với risk thật.

### Section 7 — Tổng hợp xuyên mô hình (phần "money slide")
22. **Từ 4 mảnh rời rạc → một câu hỏi chung** — TikZ flow diagram kiểu slide "Từ benchmark sang phân tích cơ chế" của `refs.tex`: 4 ô đỏ (4 pilot, số riêng) → 1 ô xanh ("có tồn tại một hành vi LLM chung không?").
23. **Hình dạng φ_U(risk) không giống nhau giữa các nguồn** — 1 chart duy nhất overlay 5 đường (người phẳng, Gemini giảm dốc, GPT-nano chữ U, Qwen-2p tăng nhẹ, Qwen-3p giảm) — thông điệp: "không có một 'hành vi LLM' phổ quát".
24. **Dấu hệ số `opponent_prev` và `ΔS` đảo chiều theo model** — diverging bar chart 2 nhóm × 5 nguồn quanh trục 0, đây là hình quan trọng nhất của deck.
25. **Tỉ lệ tái tạo 8 hiệu ứng người: 4/8 (Gemini) vs 2/8 (GPT-nano)** — grouped bar hoặc side-by-side chip grid rút gọn, dẫn vào kết luận "không model nào tái tạo đầy đủ, và không tái tạo theo cùng cách".

### Section 8 — Giới hạn & việc cần làm
26. **Giới hạn cần giữ khi diễn giải** — 2 cột: (trái) thống kê — pilot-only, N nhỏ, non-convergence ở openai, cluster ít; (phải) đo lường — persona confound với protocol/batch ở cả 2 run frontier lẫn openai (2 nguyên nhân khác nhau), `first_round_unsafe` hằng số làm singular nhiều đặc tả.
27. **Việc cần làm tiếp** — danh sách ưu tiên ngắn: (1) phép thử đổi nhãn action cho Gemma, (2) tạo biến thiên vòng 1, (3) chạy persona cùng session/qua kaggle_benchmarks để gỡ confound, (4) mixed-effects logit (TA.4), (5) EGTtools (TD.2), (6) confirmatory run sau khi đóng băng protocol.

### Section 9 — Kết
28. **Thông điệp chính** — 1 slide standout: "Bốn model, bốn hành vi khác nhau — kể cả đảo dấu hiệu ứng trung tâm của paper người. Chưa đủ confirmatory để kết luận, nhưng đủ để không giả định LLM hành xử giống người, hay giống nhau."
29. **Cảm ơn** — `[standout]`, giống slide cuối `refs.tex`.

### Backup (đóng băng, chỉ show nếu được hỏi)
B1. Bảng đầy đủ hệ số hồi quy đặc tả 6, cả 4 nguồn (dạng bảng dày, không cần visual hoá thêm).
B2. Chi tiết provenance/lệnh tái tạo mỗi run (giữ tinh thần "Freeze before you look" của deck cũ).
B3. Reference — trích dẫn Fernández Domingos & Han (2026), arXiv:2607.26034.

## 6. Trình tự thực hiện

1. Cài `basictex` (đang chạy nền) → `tlmgr` cài các gói còn thiếu: `metropolis`, `pgf`,
   `pgfplots`, `translations`, `pgfopts`, `xcolor`, `booktabs`, `multirow`, `microtype`,
   `appendixnumberbeamer`, `fontspec`, `unicode-math` (nếu cần).
2. Viết khung `progress_1.tex`: preamble (theme, màu, macro, chip) → section 0–2 (mở đầu, recap,
   phạm vi 4 run) → build thử sớm để bắt lỗi preamble/font trước khi viết hết 29 slide.
3. Viết tiếp section 3–6 (nội dung 4 pilot) — mỗi slide đối chiếu số liệu với report gốc trước
   khi gõ.
4. Viết section 7–9 (tổng hợp, giới hạn, kết) — đây là phần cần cẩn thận nhất vì tổng hợp số từ
   nhiều report khác nhau vào cùng 1 chart.
5. Build 2 lần bằng `xelatex`, sửa lỗi/overfull box, kiểm tra tất cả trang render đúng (đặc biệt
   heatmap 6×6 và diverging bar — dễ lỗi TikZ nhất).
6. Rà lại: mọi số trên slide có khớp nguyên văn report không; mọi slide có chip `PILOT` khi cần;
   không trộn ngôn ngữ; không vượt quá mật độ text đã đặt ra.

## 7. Rủi ro / điểm cần quyết định khi viết

- **Heatmap 6×6 openai**: chỉ đưa 1 model (gpt-5.4-nano, tương phản mạnh nhất 0,002→0,988) để
  giữ mật độ thấp; model kia nhắc bằng 1 câu trong takeaway, không vẽ heatmap thứ hai trừ khi
  còn dư chỗ.
- **Slide 23 (overlay 5 đường)**: 5 đường trên 1 pgfplots có thể rối — cân nhắc tách φ_U theo 2
  mức risk quan trọng nhất (0,1 và 0,9) thay vì cả 3, nếu bản nháp đầu quá rối mắt.
- **Ngôn ngữ**: toàn bộ tiếng Việt (khớp 4 report nguồn và `refs.tex`) — khác với
  `ai_race_research_deck.tex` cũ (tiếng Anh). Đây là 2 deck độc lập, không cần đồng bộ ngôn ngữ.
