# Kaggle workflows for the AI Race

Thư mục này có hai đường chạy độc lập:

- `experiments/baseline.py`: notebook GPU/Internet OFF cho các checkpoint
  open-source. Notebook tái sử dụng trực tiếp package `ai_race` và FAIRGAME, nạp
  nhiều model lần lượt để tránh giữ nhiều model trên GPU.
- `benchmarks/ai_race_baseline.py`: Kaggle Benchmark self-contained cho model
  frontier qua `kaggle_benchmarks`; không cần import source của project.
- `setup/build_quant_wheels.py`: notebook Internet ON để tạo một Kaggle Dataset
  chứa vLLM wheels cho phiên chạy offline.

Output của notebook experiment nằm tại:

```text
/kaggle/working/ai_race_results/
  <model>/
    <experiment>/
      turns.jsonl
      races.csv
      players.csv
  ai_race_all_models.csv
  ai_race_players_all_models.csv
  run_manifest.json
```

`baseline.py` mặc định xóa đúng thư mục output trên ở đầu một lần chạy mới
(`RESET_OUTPUT_DIR = True`). Không có kết quả benchmark giả hoặc artifact chạy thử
nào được lưu trong Git. Hai file aggregate chỉ nhận run có sibling manifest
`status="completed"` và thêm `source_run`/`run_status` cho từng row; partial output
của run lỗi vẫn được giữ để audit nhưng không được gộp. vLLM logprobs mặc định tắt
để giảm decode memory/work; chỉ bật rõ ràng trong model config khi thật sự cần
detailed XAI output, và lựa chọn này được ghi trong run manifest.

## Thứ tự dùng trên Kaggle

1. Nếu Kaggle image chưa có vLLM, chọn một exact vLLM version đã audit, chạy
   `setup/build_quant_wheels.py` với Internet ON, rồi lưu wheelhouse + manifest
   SHA-256 thành Dataset.
2. Tạo notebook GPU với Internet OFF, add repo (phải chứa đồng thời `ai_race/` và
   `FAIRGAME/`), checkpoint model, và Dataset wheels nếu cần.
3. Copy `experiments/baseline.py` vào notebook, sửa `MODELS`; nếu cần cài vLLM,
   điền explicit `VLLM_WHEELS_DIR`. Sau đó Run All.
4. Tải `ai_race_results.zip` từ tab Output.

Nếu dùng Kaggle API cho notebook, copy `kernel-metadata.example.json` thành
`kernel-metadata.json`, thay `YOUR_KAGGLE_USERNAME`, rồi khai báo các input thật
trong `dataset_sources`/`kernel_sources`. File `.example.json` chỉ là template;
không chứa owner hay artifact đã được xác nhận.

Xem README trong từng thư mục để biết cấu hình chi tiết.
