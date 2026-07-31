# Kaggle workflows for the AI Race

Thư mục này có hai đường chạy độc lập:

- `experiments/baseline.py`: notebook GPU/Internet OFF cho các checkpoint
  open-source. Notebook tái sử dụng trực tiếp package `ai_race` và FAIRGAME, nạp
  nhiều model lần lượt để tránh giữ nhiều model trên GPU.
- `experiments/prompt_sensitivity.py`: entrypoint private, fail-closed cho RTX PRO
  6000 (>=80 GiB). Mặc định chạy smoke 2 repetition trên 9 arm trong cùng session:
  baseline, seat-swap, placebo, risk-averse/risk-seeking và bốn social-persona cell.
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
  prompt_sensitivity_summary.csv
  run_manifest.json
```

`baseline.py` mặc định xóa đúng thư mục output trên ở đầu một lần chạy mới
(`RESET_OUTPUT_DIR = True`). Không có kết quả benchmark giả hoặc artifact chạy thử
nào được lưu trong Git. Hai file aggregate chỉ nhận run có sibling manifest
`status="completed"` và thêm `source_run`/`run_status` cho từng row; partial output
của run lỗi vẫn được giữ để audit nhưng không được gộp. vLLM logprobs mặc định tắt
để giảm decode memory/work; chỉ bật rõ ràng trong model config khi thật sự cần
detailed XAI output, và lựa chọn này được ghi trong run manifest.

`prompt_sensitivity_summary.csv` là diagnostic mô tả theo arm/risk: số decision,
Unsafe rate và chênh lệch so với baseline trung tính. Không dùng bảng này thay cho
clustered inference của `results/scripts/analyze_ai_race.py`.

## Prompt-sensitivity profile

Entrypoint đặt `AI_RACE_RUN_PROFILE=prompt_sensitivity_smoke`, yêu cầu đúng tên GPU
`RTX PRO 6000`, VRAM tối thiểu 80 GiB, CUDA khả dụng, và coverage đủ số race trước
khi đánh dấu run `completed`. Sau khi smoke qua parser/coverage/symmetry gates, tạo
kernel version mới với `AI_RACE_RUN_PROFILE=prompt_sensitivity_pilot` để chạy 10
repetition/arm. Không sửa prompt, temperature, seed hoặc model giữa hai version.

Mọi arm dùng seed `260726`, ba risk treatment và prompt template v3 giống nhau.
Persona chỉ đi qua optional intro block, nên đây là matched prompt manipulation;
`baseline_swapped` tách artefact ghế và `R0` tách hiệu ứng thêm text khỏi nội dung.

## GreenNode hai lane

`experiments/greennode_prompt_sensitivity.py` dùng cùng engine/config nhưng gọi
Ollama nội bộ. Lane A và B là partition không giao nhau của 9 arm, nên hai GPU
không chạy trùng quan sát. Mỗi lane có output root riêng trên shared disk, manifest
được replace nguyên tử, và khi resume chỉ bỏ qua shard có `status="completed"`.

Runner khóa model name/digest, Ollama version, hostname, GPU, source hash, prompt
hash, decoding và fixed-seed probe. Smoke dùng 2 repetition/arm; pilot dùng 10 và
chỉ được launch sau khi merge output smoke qua coverage/parser/symmetry gates.

Nếu một persona đối xứng bão hòa ở một action và làm symmetry gate thất bại, giữ
cell đó như saturation diagnostic từ full smoke. Pilot dùng `--matrix identified`
để scale controls và các cell còn action/race-position variation; manifest ghi rõ
matrix nên hai tập không bao giờ bị pool âm thầm.

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
