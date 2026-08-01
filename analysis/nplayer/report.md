# Phân tích run `nplayer` — qwen2.5-14b-instruct (N=3)

Nguồn dữ liệu: `results/nplayer/` (`nplayer_nonpersona/` — baseline không persona, 3 mức rủi ro
riêng tối đa × 20 lần lặp; `nplayer-riskaware/` — persona rủi ro Eckel-Grossman R1–R6, đối xứng
cả 3 ghế, × 2 lần lặp). Cả hai đều `run_phase = pilot`.

Dashboard: [visualizations/dashboard.html](visualizations/dashboard.html).
Phân tích đầy đủ bằng văn xuôi (số liệu, diễn giải, giới hạn):
[results/nplayer/ANALYSIS_qwen2.5-14b-instruct.md](../../results/nplayer/ANALYSIS_qwen2.5-14b-instruct.md).
Script sinh số liệu: [results/nplayer/analyze_nplayer_results.py](../../results/nplayer/analyze_nplayer_results.py).
Đối chiếu lý thuyết: [N-Player/theory/](../../N-Player/theory/) (điều kiện DSAI closed-form,
stationary distribution AS/AU/CS).

> **Đây là PILOT, không phải confirmatory.** Run persona chỉ có 2 lần lặp/ô (~2 race độc lập/ô) —
> các con số persona là quan sát mô tả, không phải ước lượng chắc chắn.

## Tóm tắt nhanh (chi tiết ở dashboard/ANALYSIS.md)

1. Cơ chế hoạt động đúng thiết kế: 0% parse failure, CRN (common-random-number) khớp tuyệt đối
   giữa baseline và mọi persona ở cùng rep, setback rate tăng đúng theo `max_private_risk`.
2. Baseline: tỷ lệ UNSAFE giảm dần khi rủi ro tăng (69%→57%→47%), nhưng thoải hơn nhiều so với
   bước nhảy sắc nét mà lý thuyết quần thể (`N-Player/theory`) dự đoán — không mâu thuẫn, hai mô
   hình đo hai cơ chế khác nhau (chọn lọc tiến hóa dài hạn vs suy luận trong-ngữ-cảnh một model).
3. Mọi người chơi luôn mở màn bằng UNSAFE ở vòng 1, bất kể risk.
4. Phát hiện nổi bật nhất: persona rủi ro Eckel-Grossman áp đảo hoàn toàn tín hiệu risk thực tế ở
   5/6 mức persona (hàm bậc thang 0%/100%) — chỉ persona "trung tính" mới nhạy với `pr` thực tế.
   Cỡ mẫu rất nhỏ (n=2 race/ô), cần xác nhận lại bằng confirmatory run.
