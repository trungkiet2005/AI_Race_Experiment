# AI Race Kaggle Benchmark

`ai_race_baseline.py` là task self-contained có slug `ai-race-baseline`. Task chạy
hai agent có chat tách biệt trong repeated AI race của paper, quét
`pmax ∈ {0.1, 0.6, 0.9}`, và ghi:

- `turns.jsonl`: prompt, raw response, action, parse/retry và race state;
- `races.csv`: horizon, winner/tie, setback và payoff;
- `players.csv`: SAFE/UNSAFE rate, progress, prize, private risk và payoff;
- `summary.json`: thống kê theo risk cùng parse health;
- `run_manifest.json`: phase, prompt/source hash, model, decoding, seed, package và
  trạng thái terminal.

Mọi request đều có output cap 256 token. Task tự chọn `max_tokens` cho backend
OpenAI/Model Proxy và `max_output_tokens` cho backend GoogleGenAI trực tiếp; lựa
chọn cùng giá trị cap được ghi trong manifest, turn row và attempt history. Nếu
Kaggle bổ sung backend chưa nhận diện được, task sẽ dừng thay vì âm thầm bỏ cap;
chỉ override rõ ràng bằng
`AI_RACE_TOKEN_LIMIT_PARAMETER=max_tokens` hoặc
`AI_RACE_TOKEN_LIMIT_PARAMETER=max_output_tokens` sau khi kiểm tra API của
backend đó.

Common random numbers của horizon và fixed-seat setback draw được bảo đảm giữa
ba treatment risk trong cùng repetition. Sampling seed được ghi riêng theo các
trạng thái requested, SDK-forwarded và applied/known. Không diễn giải sampling
randomness là CRN đã được bảo đảm nếu provider không xác nhận seed; với các route
mà Kaggle SDK chủ động loại seed, `sampling_seed_applied` được ghi là `false`.
Tương tự, temperature được ghi tách thành requested, SDK-forwarded và effective;
không xem `0.7` là temperature đã áp dụng khi backend đặt
`support_temperature=false` hoặc provider không xác nhận giá trị effective.

Mặc định có 3 race cho mỗi risk và `RUN_PHASE="pilot"`. Trước confirmatory run,
freeze task source rồi đổi fallback `AI_RACE_RUN_PHASE` thành `confirmatory` và
đặt `AI_RACE_REPS`/`REPETITIONS` đúng cỡ mẫu đã preregister. Không download rồi
pool pilot với confirmatory; mọi row và manifest đều ghi phase.

## Workflow theo checkpoint

Chạy **một lệnh mỗi lần**, kiểm tra output rồi mới sang checkpoint kế tiếp. Không
chain các lệnh này trong một command dài.

Khởi tạo CLI/credentials:

```bash
kaggle b init -y
```

Push đúng task slug:

```bash
kaggle b t push ai-race-baseline -f kaggle/benchmarks/ai_race_baseline.py --wait
```

Ngay trước remote run, refresh token Model Proxy (token ngắn hạn; run cũ từng gặp
401 do token hết hạn):

```bash
kaggle b auth -y
```

Run một model (dùng canonical model slug do `kaggle b t models` trả về):

```bash
kaggle b t run ai-race-baseline -m <model-slug> --wait
```

Xem trạng thái hoặc log:

```bash
kaggle b t status ai-race-baseline -m <model-slug>
```

```bash
kaggle b t log ai-race-baseline -m <model-slug>
```

Tải output sau khi run terminal:

```bash
kaggle b t download ai-race-baseline -m <model-slug> -o results/kaggle-benchmarks
```

Nếu cần notebook source để debug, thêm `-f -s` vào lệnh download. Không diễn giải
SAFE/UNSAFE rate nếu `parse_failures` khác 0; task chỉ assert parse health, không
assert một kết quả hành vi mong muốn.
