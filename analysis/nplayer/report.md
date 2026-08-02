# Phân tích run `nplayer` — qwen2.5-14b-instruct (N=3)

Nguồn dữ liệu: `results/nplayer/` (`nplayer_nonpersona/` — baseline không persona, 3 mức rủi ro
riêng tối đa × 20 lần lặp; `nplayer-riskaware/` — persona rủi ro Eckel-Grossman R1–R6, đối xứng
cả 3 ghế, × 2 lần lặp). Cả hai đều `run_phase = pilot`.

Dashboard: [visualizations/dashboard.html](visualizations/dashboard.html).
Phân tích đầy đủ bằng văn xuôi (số liệu, diễn giải, giới hạn):
[results/nplayer/ANALYSIS_qwen2.5-14b-instruct.md](../../results/nplayer/ANALYSIS_qwen2.5-14b-instruct.md).
Script mô tả: [results/nplayer/analyze_nplayer_results.py](../../results/nplayer/analyze_nplayer_results.py).
Script suy luận (hồi quy, phân loại chiến lược, theory-fit):
[results/nplayer/rigorous_analysis.py](../../results/nplayer/rigorous_analysis.py) → [derived/](derived/).
Đối chiếu lý thuyết: [N-Player/theory/](../../N-Player/theory/) (điều kiện DSAI closed-form,
stationary distribution AS/AU/CS).

> **Đây là PILOT, không phải confirmatory.** Run persona chỉ có 2 lần lặp/ô (~2 race độc lập/ô) —
> các con số persona là quan sát mô tả, không phải ước lượng chắc chắn. Chỉ baseline (60 race,
> 1.404 quyết định vòng ≥2, 20 cluster CRN) đủ mẫu cho hồi quy/kiểm định thống kê.

## Đối chiếu độ sâu với `analysis/frontier/` (2-player)

`analysis/frontier/` đối chiếu với 19 phân tích của một **nghiên cứu con người thật**
(`arXiv-2607.26034v1`, xem `docs/paper-analyses-inventory.md`); N-player không có nghiên cứu
con người tương ứng nào (bài báo đó chỉ chơi 2 người). Bảng dưới so mức độ nghiêm ngặt của phần
**thực nghiệm LLM** (không tính phần lý thuyết N-Player thuần túy, đã đầy đủ ở `N-Player/theory/`):

| Hạng mục | frontier (2-player) | nplayer (N=3) |
|---|---|---|
| t-test / effect size giữa treatment | ✅ | ⚠️ chỉ Wilson CI, chưa t-test chính thức |
| Hồi quy logistic panel cluster-robust | ✅ 6 đặc tả, cluster theo cặp | ✅ 6 đặc tả (3 ước lượng được, 3 không do `first_round_unsafe` hằng số), cluster theo `rep` |
| Phân loại chiến lược (Hamming distance) | ✅ AS/AU/CS/CAS | ✅ AS/AU/CS (không CAS, N-player không định nghĩa) |
| So khớp lý thuyết tiến hóa quần thể | ✅ 2 điểm (β,µ) cố định | ✅ **quét β grid search** tìm điểm khớp tốt nhất (chặt hơn frontier) |
| Robustness (jackknife/dropout) | ✅ | ❌ chưa làm |
| CI/p-value trên biểu đồ | ✅ | ✅ (Wilson CI, forest plot) |

## Tóm tắt nhanh (chi tiết ở dashboard/ANALYSIS.md)

1. Cơ chế hoạt động đúng thiết kế: 0% parse failure, CRN (common-random-number) khớp tuyệt đối
   giữa baseline và mọi persona ở cùng rep, setback rate tăng đúng theo `max_private_risk`.
2. Baseline: tỷ lệ UNSAFE giảm dần khi rủi ro tăng (69%→57%→47%, hồi quy xác nhận p<0.001 cho cả
   hai contrast risk). Ngay cả sau khi **quét β tìm điểm khớp tốt nhất**, lý thuyết quần thể vẫn
   dự đoán độ dốc giảm mạnh hơn thực tế ở risk 0.6/0.9 — LLM giữ hành vi UNSAFE cao hơn dự đoán.
3. Mọi người chơi luôn mở màn bằng UNSAFE ở vòng 1 (biến này thành hằng số, không hồi quy được).
4. Hồi quy cluster-robust: khác paper người gốc ở 3 điểm — (a) hành động trước của *chính mình*
   cũng dự báo mạnh (paper người: không), (b) có bão hòa khi cả hai bên cùng vừa Unsafe, (c) vị
   thế đua (ΔS) gần như không có tác dụng riêng (paper người: đây là phát hiện trung tâm).
5. Phân loại chiến lược: không player-race nào khớp CS; 13–22% quỹ đạo không khớp chiến lược
   thuần túy nào (Tie/Other).
6. Phát hiện nổi bật nhất: persona rủi ro Eckel-Grossman áp đảo hoàn toàn tín hiệu risk thực tế ở
   5/6 mức persona (hàm bậc thang 0%/100%) — chỉ persona "trung tính" mới nhạy với `pr` thực tế.
   Cỡ mẫu rất nhỏ (n=2 race/ô, không đưa vào hồi quy được), cần xác nhận lại bằng confirmatory run.
