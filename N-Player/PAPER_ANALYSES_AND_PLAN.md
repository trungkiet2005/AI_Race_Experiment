# N-Player DSAIR — Phân tích trong paper & kế hoạch code

Nguồn: Han, Pereira, Santos & Lenaerts, *"To Regulate or Not: A Social Dynamics
Analysis of an Idealised AI Race"*, JAIR 69 (2020) 881-921
([jair-12225/JAIR-12225-ArticlePDF-25030-1-10-20201122.md](../jair-12225/JAIR-12225-ArticlePDF-25030-1-10-20201122.md)).

Phạm vi: paper định nghĩa mô hình 2-player ở phần thân bài, rồi tổng quát hóa
lên **N-player (N ≥ 2)** trong **Appendix B**, và mở rộng thêm trong Appendix C
(risk cá nhân vs tập thể), D (hậu quả kéo dài khi bị phát hiện) và E (social
welfare) — các phần này đều nói rõ kết quả "vẫn đúng" hoặc "thay đổi thế nào"
khi chuyển từ 2 sang N. Repo hiện đã có `ai_race/engine_nplayer/` để **chạy**
một ván đua N-player (mock hoặc LLM thật), nhưng **chưa có lớp phân tích lý
thuyết (EGT) cho N-player** — đó là khoảng trống mà thư mục này sẽ lấp.

## 1. Danh mục các phân tích trong paper (N-player / Appendix B-E)

| # | Phân tích | Vị trí trong paper | Mô tả | Đã có trong repo? |
|---|---|---|---|---|
| 1 | Công thức payoff mỗi vòng theo k người chơi SAFE: `π_SAFE(k)`, `π_UNSAFE(k)` | Appendix B, đầu mục "N-player AI Race Definition" | Payoff phụ thuộc số người chơi SAFE `k` trong nhóm N, chia sẻ benefit `b` theo tốc độ phát triển | ✅ `ai_race/engine_nplayer/scoring.py` (đã có sẵn) + `theory/conditions.py::stage_payoff_safe/unsafe` (bản tổng quát có `pfo`) |
| 2 | Payoff trung bình cả ván đua cho AS/AU/CS gặp AU: `Π_AS,AU(k)`, `Π_AU,AS(k)`, `Π_CS,AU(k)`, `Π_AU,CS(k)` | Appendix B, ngay sau (1) | Tích hợp phần thưởng cuối `B` và tốc độ hoàn thành `W` hoặc `W/s` vào payoff trung bình mỗi vòng | ✅ `theory/welfare.py::average_payoff_{as_vs_au,au_vs_as,cs_vs_au,au_vs_cs}` |
| 3 | Điều kiện phúc lợi tập thể: khi nào toàn bộ SAFE tốt hơn toàn bộ UNSAFE, `Π_AS,AU(N) > Π_AU,AS(0)` | Eq. 21 | `pr > 1 − (B + W(b−Nc)) / (sB + W(1−pfo)b)` | ✅ `theory/conditions.py::welfare_condition_threshold` |
| 4 | Rút gọn cho **early DSAI** (B/W ≫ b) | Eq. 22 | `pr > 1 − 1/s` — giống hệt điều kiện 2-player, **không phụ thuộc N** | ✅ `theory/conditions.py::early_dsai_welfare_threshold` |
| 5 | Rút gọn cho **late DSAI** (B/W ≪ b) | Eq. 23 | `pr > 1 − (b − Nc)/((1−pfo) b)`; cần `b > Nc`; ngưỡng tăng theo N | ✅ `theory/conditions.py::late_dsai_welfare_threshold` |
| 6 | Điều kiện risk-dominance của AS, CS so với AU (tổng payoff qua mọi k) | Eq. 24, 25 | `Σ_{k=0}^{N-1} π_AU,AS(k) < Σ_{k=1}^{N} π_AS,AU(k)` (tương tự cho CS) | ✅ `theory/population.py::risk_dominant` (dạng tổng quát, nhận `payoff_of_k` bất kỳ — dùng chung cho Eq. 24/25 và Eq. 31) |
| 7 | Rút gọn risk-dominance cho early DSAI, dùng số điều hòa `H_N = Σ_{i=1}^N 1/i` | Eq. 26 | `pr > 1 − 1/(N·H_N·s)`; khi N→∞ ngưỡng → 1 | ✅ `theory/conditions.py::early_dsai_risk_dominance_threshold` — khớp số paper trích (N=5,s=1.5 → 0.94) |
| 8 | Ba vùng DSAI (compliance / dilemma / innovation) trong không gian `s`–`pr`, cho N-team | Fig. S7, S8 (early); Fig. S9 (late) | Biên (I)/(II)/(III) xác định bởi Eq. 22 & 26 (early) hoặc Eq. 27 & 28 (late); vùng dilemma (II) rộng ra khi N tăng | ✅ `theory/conditions.py::dsai_zone` |
| 9 | Risk-dominance AS/CS vs AU cho **late DSAI** | Eq. 27, 28 | `pr > 1 − Σπ_SAFE(i) / Σπ_UNSAFE(i)`; dạng rút gọn cho CS dùng `π(N)_SAFE` và `π(0)_UNSAFE` | ✅ `theory/conditions.py::late_dsai_risk_dominance_threshold_{as,cs}` |
| 10 | Lấy mẫu nhóm theo phân phối hypergeometric đa biến (multivariate hypergeometric) | Công thức `H(k,N,x,Z)` (Hauert et al., 2007) | Xác suất chọn k cá thể loại i và N−k loại j từ quần thể Z | ✅ `theory/population.py::hypergeometric_pmf` |
| 11 | Payoff trung bình của chiến lược i/j trong quần thể hữu hạn: `P_ij(x)`, `P_ji(x)` | Eq. 29 | Tổng có trọng số hypergeometric của `Π_ij(k)` qua mọi k khả dĩ — đây là bản N-player của `fitness_in_population` (2-player) | ✅ `theory/population.py::average_payoff_{i,j}` — đối chiếu chính xác với `ai_race/theory/evolution.py` tại N=2 |
| 12 | Xác suất chuyển trạng thái `T±(k)` (pairwise comparison / Fermi rule) cho quần thể N-team | Eq. 30 | Giống công thức 2-player nhưng payoff dùng `P_ij(x)`/`P_ji(x)` ở trên | ✅ Gộp vào `theory/population.py::fixation_probability` (log-sum-exp, giống `evolution.py`) |
| 13 | Xác suất cố định hóa (fixation probability) & phân phối dừng (stationary distribution) cho N-team | Suy ra từ Eq. 29-30 (cùng cơ chế Eq. 4 ở phần thân bài) | Bản N-player của `ai_race/theory/evolution.py` | ✅ `theory/population.py::fixation_probability`, `small_mutation_stationary` — khớp giới hạn neutral-drift `ρ=1/Z` khi `β=0` (paper phát biểu trực tiếp) |
| 14 | Điều kiện risk-dominance tổng quát (giới hạn Z lớn) | Eq. 31 | `Σ_{k=1}^N Π_ij(k) ≥ Σ_{k=0}^{N-1} Π_ji(k)` | ✅ `theory/population.py::risk_dominant` |
| 15 | Risk cá nhân vs tập thể (hệ số `γ`) áp dụng cho N-team | Appendix C (đoạn cuối) + Fig. S12 | Payoff AS/CS gặp AU nhân với `(1 − pr·γ)`; late DSAI: vùng innovation rộng ra khi γ tăng | ✅ Tham số `gamma` trong `theory/welfare.py::average_payoff_{as_vs_au,cs_vs_au}` |
| 16 | Social welfare / average population payoff qua các vùng DSAI, tổng quát hóa cho N | Appendix E (Fig. S13-S15, thảo luận) | So sánh phúc lợi trung bình giữa 3 vùng — cùng công thức payoff nhưng lấy trung bình quần thể tại trạng thái dừng | ✅ `theory/welfare.py::homogeneous_payoff`, `social_welfare` |
| 17 (phụ) | Phân tích dữ liệu hành vi thực nghiệm (LLM chơi N-player) — không phải lý thuyết trong paper nhưng là phần đối chiếu cần có | n/a (khoảng trống của repo, tương tự `results/scripts/analyze_ai_race.py`) | Thống kê mô tả trên output của `engine_nplayer` (turns.jsonl/races.csv/players.csv dạng long-format) | ❌ Chưa (không bắt buộc theo yêu cầu "phân tích trong paper", để phase phụ) |
| — | Vẽ lại hình minh họa (Fig. S7/S8, S9, S12) để review trực quan | — | Script matplotlib, không phải test tự động | ✅ `figures/reproduce_paper_figures.py` — 9 hình PNG trong `figures/output/`, khớp hình dạng paper (biên early-DSAI cong tăng theo `s`; late-DSAI vùng innovation mở rộng rõ rệt khi γ: 0→1, đúng như Appendix C mô tả) |

**Trạng thái tổng quan (2026-08-01):** Phase 1-4 đã code xong và có test — 62/62 test pass
(`N-Player/tests/`), cộng script vẽ hình đối chiếu trực quan (`figures/reproduce_paper_figures.py`,
9 hình PNG trong `figures/output/`). Còn lại: Phase 5 (bridge với dữ liệu LLM thực nghiệm,
việc lớn hơn nhiều, để riêng khi có nhu cầu).

Ghi chú: mục 1 (payoff cơ chế) **đã** được implement đúng và có test riêng
(`ai_race/tests/test_nplayer_scoring.py`), khớp lại được ma trận 2-player khi
N=2. Các mục 2–16 là phần lý thuyết EGT (Evolutionary Game Theory) mà
`ai_race/engine_nplayer/README.md` liệt kê rõ là **"deliberately out of
scope"** cho engine — đây chính là phần cần code trong thư mục `N-Player/`.

## 2. Kế hoạch code

### Nguyên tắc

- Tái sử dụng, không sao chép: dùng lại `ai_race.engine_nplayer.state.NPlayerGameConfig`
  và `ai_race.engine_nplayer.scoring` (đã có `π_SAFE(k)`/`π_UNSAFE(k)`) làm nguồn
  sự thật cho cơ chế; không viết lại công thức payoff cơ bản.
- Mirror kiến trúc đã có ở `ai_race/theory/` (`payoffs.py` → `equilibria.py` →
  `evolution.py`) nhưng tổng quát hóa theo N thay vì cố định 2 người chơi. Cùng
  quy ước: closed-form khi có thể (Eq. 21-28, 31), fallback sang tổng
  hypergeometric hữu hạn (Eq. 29-30) — không cần Monte Carlo vì mọi tổng đều hữu hạn.
- Mỗi công thức trong bảng trên → một hàm thuần (pure function), có test đối
  chiếu số liệu paper trích dẫn sẵn (vd: Fig. S7 caption nói N=5, s=1.5 →
  vùng (I) tại `pr > 0.94`, vùng (II) tại `0.94 > pr > 0.33`).

### Cấu trúc file (đã tạo trong `N-Player/`)

```
N-Player/
├── PAPER_ANALYSES_AND_PLAN.md   (file này)
├── conftest.py           # thêm N-Player/ và repo root vào sys.path cho pytest
├── theory/
│   ├── __init__.py
│   ├── conditions.py     # mục 3-9, 15: điều kiện đóng (closed-form), 3 vùng DSAI  ✅ done
│   ├── population.py     # mục 10-14: hypergeometric sampling, P_ij(x), fixation, stationary dist  ✅ done
│   ├── welfare.py        # mục 2, 16: payoff trung bình + social welfare tại stationary state  ✅ done
│   └── stationary.py     # ráp welfare.py + population.py thành stationary distribution AS/AU/CS thật  ✅ done
├── figures/
│   ├── reproduce_paper_figures.py   # script vẽ heatmap AU-frequency, cf. Fig S7-S9/S12  ✅ done
│   └── output/*.png                 # 9 hình đã render
└── tests/
    ├── test_conditions.py   # 20 test, pass
    ├── test_population.py   # 16 test, pass
    ├── test_stationary.py   # 8 test, pass — test tích hợp end-to-end
    └── test_welfare.py      # 18 test, pass
```

(Nếu về sau muốn thư mục này trở thành một phần chính thức của package
`ai_race` — ví dụ để `results/scripts/analyze_ai_race.py` gọi tới — có thể di
chuyển `theory/` vào `ai_race/theory_nplayer/` sau; giữ tách biệt lúc đầu để
không đụng tới vùng đã "đóng băng" theo `CLAUDE.md`.)

### Phase 1 — `theory/conditions.py` (mục 3-9)

Hàm chính, tất cả nhận `NPlayerGameConfig` (hoặc tham số rời `n, s, pr, pfo,
b, c, B, W`) và trả về `float`/`bool`/`str`:

- `welfare_condition_threshold(config) -> float` — vế phải Eq. 21 (ngưỡng `pr`
  đầy đủ, không rút gọn theo regime).
- `early_dsai_welfare_threshold(config) -> float` — Eq. 22 (`1 − 1/s`).
- `late_dsai_welfare_threshold(config) -> float` — Eq. 23.
- `harmonic_number(n) -> float` — `H_N`, dùng nội bộ.
- `early_dsai_risk_dominance_threshold(config) -> float` — Eq. 26.
- `late_dsai_risk_dominance_threshold(config, strategy: Literal["AS","CS"]) -> float`
  — Eq. 27/28.
- `collective_risk_scaled_threshold(config, gamma: float, strategy) -> float`
  — mục 15, tổng quát Eq. 35/36 của bản 2-player sang N (nhân `(1 − pr·γ)`).
- `dsai_zone(config, *, regime: Literal["early","late"]) -> Literal["compliance","dilemma","innovation"]`
  — phân loại theo 2 ngưỡng ở trên, dùng để tô lại Fig. S7/S9 (mục 8).

Test: cắm `N=5, s=1.5` → kỳ vọng ngưỡng vùng (I) ≈ 0.94, biên vùng (III) =
1 − 1/1.5 ≈ 0.333 — đúng số paper ghi trong caption Fig. S7.

### Phase 2 — `theory/population.py` (mục 10-14)

- `hypergeometric_pmf(k, n_trials, x, z) -> float` — công thức `H(k,N,x,Z)`.
- `average_payoff_i(config, payoff_i_of_k, x, z) -> float` — Eq. 29 (`P_ij(x)`),
  nhận một hàm `k -> Π_ij(k)` (lấy từ mục 2 hoặc trực tiếp từ
  `engine_nplayer.scoring`) và tổng theo hypergeometric.
- `transition_probability(config, payoff_i_of_k, payoff_j_of_k, x, z, beta) -> tuple[float,float]`
  — `T+(x)`, `T-(x)` (Eq. 30), dùng lại log-sum-exp pattern giống
  `ai_race/theory/evolution.py::fixation_probability` để tránh overflow.
- `fixation_probability(config, mutant_payoff_of_k, resident_payoff_of_k, *, z, beta) -> float`
  — cùng thuật toán Fermi/pairwise-comparison, nhưng payoff lấy từ
  `average_payoff_i`/`average_payoff_j` (Eq. 29) thay vì công thức 2-player
  trực tiếp.
- `small_mutation_stationary(config, strategies, payoff_lookup, *, z, beta) -> dict[str,float]`
  — bản N-player của hàm cùng tên trong `ai_race/theory/evolution.py`; ma trận
  chuyển 4×4 (hoặc 3×3 nếu bỏ CAS — xem mục "Quy ước" bên dưới) từ
  `fixation_probability`.
- `risk_dominant(config, i_payoff_of_k, j_payoff_of_k) -> bool` — Eq. 31, giới
  hạn Z lớn.

Quy ước chiến lược: paper Appendix B chỉ định nghĩa AS/AU/CS cho N-player
(không có CAS N-player), khớp với quyết định đã ghi trong
`ai_race/engine_nplayer/README.md`. Module này giữ nguyên quy ước đó — chỉ 3
chiến lược.

### Phase 3 — `theory/welfare.py` (mục 2, 16)

- `expected_payoff_matrix(config, strategies=("AS","AU","CS")) -> dict[tuple[str,str], float]`
  — closed-form Eq. trong mục 2 (payoff trung bình cả ván, dùng
  `expected_horizon` kiểu giống `ai_race/theory/payoffs.py::expected_horizon`).
- `social_welfare(stationary, payoff_matrix) -> float` — phúc lợi trung bình
  quần thể tại trạng thái dừng, dùng để tái tạo lại Fig. S13-S15 (so sánh giữa
  3 vùng DSAI).

### Phase 4 — `figures/reproduce_paper_figures.py`

Script (không phải test) vẽ lại các heatmap tần suất AU theo `(s, pr)` — Fig.
S7/S8 — và theo `(pfo, pr)` — Fig. S9/S12 — cho một vài giá trị N (3, 5, 10)
bằng matplotlib, dùng làm kiểm tra trực quan đối chiếu paper (không phải test
tự động, chỉ chạy thủ công khi cần review).

### Phase 5 (tùy chọn, phụ) — cầu nối với dữ liệu LLM thực nghiệm

Nếu sau này cần so khớp dự đoán lý thuyết ở trên với hành vi LLM thật chơi
N-player (giống cách `results/scripts/analyze_ai_race.py` so `_build_theory_comparison`
với dữ liệu 2-player), cần thêm một bước đọc `races.csv`/`players.csv` dạng
long-format của `engine_nplayer` — hiện chưa có analyzer nào đọc được. Đây là
việc lớn hơn nhiều (mirror toàn bộ `analyze_ai_race.py`), nên tách thành yêu
cầu riêng, không nằm trong phạm vi "phân tích lý thuyết trong paper" của kế
hoạch này.

### Thứ tự triển khai (đã làm Phase 1-3)

1. Phase 1 (`conditions.py`) — thuần công thức đóng, không phụ thuộc gì khác,
   có thể viết test ngay từ số liệu paper trích dẫn. **Xong**, 20 test pass.
2. Phase 2 (`population.py`) — phức tạp hơn (hypergeometric + fixation), phụ
   thuộc NumPy như `ai_race/theory/evolution.py`. **Xong**, 16 test pass.
3. Phase 3 (`welfare.py`) — cần Phase 1 (tái dùng `stage_payoff_safe/unsafe`)
   để dựng `Π_ij(k)` và γ-scaling. **Xong**, 18 test pass.
4. Phase 4 (`figures/reproduce_paper_figures.py`) — cần ráp Phase 2 + Phase 3
   lại với nhau để tính được stationary distribution thật (module mới
   `theory/stationary.py`, không nằm trong plan ban đầu nhưng cần thiết để
   nối hai phase lại — xem mục 3 "Ghi chú kỹ thuật" bên dưới về lỗi phát hiện
   được khi ráp). **Xong**, 8 test tích hợp pass + 9 hình PNG khớp hình dạng
   paper (biên early-DSAI cong đúng dạng `1-1/(N·H_N·s)`; vùng innovation
   late-DSAI mở rộng rõ khi γ tăng 0→1, đúng như Appendix C).
5. Phase 5 — để sau, khi có nhu cầu thực tế so khớp dữ liệu LLM.

## 3. Ghi chú kỹ thuật quan trọng khi implement

- **Quy ước "per-round rate" của `Π` trong Appendix B, không phải "total
  payoff" như `ai_race/theory/payoffs.py`.** Paper nói rõ mọi ô trong ma trận
  `Π` là tốc độ trung bình *mỗi vòng* ("obtaining on average B/2W per round").
  `ai_race/theory/payoffs.py` (dùng cho engine 2-player) lại tính **tổng tích
  lũy cả ván** (`horizon * stage_payoff + prize`, không chia W) — đây là một
  quy ước khác, hợp lý cho mục đích riêng của nó (khớp với `final_payoffs` mà
  engine thực sự ghi lại), nhưng **không** phải cùng đại lượng với `Π` trong
  paper. `theory/welfare.py` bám theo đúng chữ nghĩa paper (chia `NW` hoặc
  `W`), nên **không so khớp trực tiếp bằng số** với `ai_race/theory/payoffs.py`
  — thay vào đó test đối chiếu bằng cách rút gọn về N=2 rồi tính tay theo
  Eq. 2 nguyên văn (xem docstring đầu `theory/welfare.py` và
  `tests/test_welfare.py`).
- **OCR của paper bị mất một số ký hiệu công thức** (đã cảnh báo ngay đầu file
  markdown gốc). Cụ thể, hệ số sống sót `(1 − p_r)` trong `Π_AU,AS(k)` và
  `Π_AU,CS(k)` bị trích xuất thành một chữ "p" trơ trọi. `theory/welfare.py`
  đọc nó là `(1 − p_r)` — cách đọc duy nhất khớp với ô tương ứng ở Eq. 2 (bản
  2-player) và được test xác nhận.
- **Ý nghĩa của "k" trong `Π_CS,AU(k)`/`Π_AU,CS(k)` đã được xác định qua đối
  chiếu N=2**: `k` = số đồng đội kiểu CS (kể cả bản thân, với `Π_AS,AU`/
  `Π_CS,AU`) chơi SAFE — vì CS luôn chơi SAFE ở vòng 1, "k" cũng chính là số
  đối tác kiểu CS có mặt trong nhóm ở vòng đầu tiên.
- **Lỗi phát hiện khi ráp Phase 2 + Phase 3 (đã sửa)**: `population.py`'s
  `fitness_in_population` ban đầu nhận `mutant_payoff_of_k` và
  `resident_payoff_of_k` với **hai quy ước "k" khác nhau ngầm định** (mutant:
  tự đếm cả bản thân 1..n; resident: chỉ đếm đối tác không tính bản thân
  0..n-1). Test Phase 2 không bắt được vì chỉ dùng hàm hằng số/toy (bất biến
  với cách đếm). Khi ráp `welfare.py`'s hàm thật (bất đối xứng thật sự) vào
  `stationary.py`, cùng một hàm phải đóng vai trò mutant lẫn resident tùy
  chiều xâm nhập — lộ ra xung đột quy ước. Đã sửa bằng cách thống nhất một
  quy ước duy nhất (luôn tự-bao-gồm-bản-thân, 1..n) cho cả hai đối số, và để
  `fitness_in_population` tự quy đổi nội bộ cho phía resident
  (`resident_payoff_of_k(n - k)`). Bài học: khi một hàm nhận "payoff theo k"
  cho nhiều vai trò khác nhau, quy ước của "k" phải được cố định rõ ràng ở
  một chỗ duy nhất — kiểm thử bằng hàm hằng số/đối xứng sẽ không phát hiện
  được sai lệch quy ước, cần test bằng dữ liệu bất đối xứng thật
  (`tests/test_stationary.py` làm việc này).

## 4. Cách chạy test

Repo yêu cầu Python ≥3.10 (`pyproject.toml`) nhưng máy này chỉ có Python 3.9.6
theo mặc định; đã dùng Python 3.12 (`~/.local/bin/python3.12`, quản lý bởi
`uv`) trong một virtualenv cô lập (không phải `.venv` trong repo, để không
đụng tới việc resolve dependency đầy đủ của `pyproject.toml`):

```bash
uv venv --python 3.12 /path/to/isolated/venv
uv pip install --python /path/to/isolated/venv/bin/python pytest numpy
/path/to/isolated/venv/bin/python -m pytest N-Player/tests/ -v
```

`N-Player/conftest.py` tự thêm `N-Player/` và repo root vào `sys.path`, nên
không cần cài `ai_race` ở chế độ editable để import được `theory.*` hay
`ai_race.*` trong test.

Để chạy script vẽ hình (cần thêm `matplotlib`, không phải dependency của
repo):

```bash
uv pip install --python /path/to/isolated/venv/bin/python matplotlib
/path/to/isolated/venv/bin/python N-Player/figures/reproduce_paper_figures.py --resolution 40
```

Hình PNG sẽ được ghi vào `N-Player/figures/output/`.
