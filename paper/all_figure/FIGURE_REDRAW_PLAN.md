# Kế hoạch vẽ lại hình cho paper AAMAS

Thay thế phần "chọn hình" trong [`SELECTED_FOR_PAPER.md`](SELECTED_FOR_PAPER.md). File đó trả lời
*"hình nào đáng dùng"*; file này trả lời *"vẽ lại thế nào cho đúng loại chart, đúng số liệu, đúng
mục đích"*.

Mọi con số dưới đây đã được đối chiếu ngược lại CSV nguồn. Chỗ nào không tái tạo được đều được đánh
dấu 🚩.

**Kết quả rút gọn: 8 hình → 6 hình + 2 bảng.** Hai hình bị giết là hình bị vẽ sai loại chart, không
phải hình yếu nội dung.

---

## 0. Bốn vấn đề chặn, phải xử lý trước khi vẽ bất cứ hình nào

| # | Vấn đề | Ảnh hưởng |
|---|---|---|
| B1 | **`results/cross_provider/` không có `run_manifest.json` ở bất kỳ đâu** (đã `find` xác nhận). CLAUDE.md: *"Admitted pilot artifacts must trace to a completed manifest, immutable raw logs, and a fail-closed analyzer."* | Hình 04 + 05 hiện không đủ điều kiện admit. Xem §4/§5 — có nguồn thay thế **có** manifest SHA-256. |
| B2 | **Nguồn của hình 07 đã bị xoá khỏi repo.** `ai_race/results/_api_5games_allrisk/` được thêm ở `a81a8c8`, xoá ở `b07ae73` (tổ tiên của HEAD). `analyze_two_player_paper_figures.py` hard-code đường dẫn này → chạy lại hôm nay sẽ crash. `dataset_inventory.csv:94` vẫn liệt kê nó như artifact sống. | Không regenerate được cho camera-ready. Phải khôi phục thư mục hoặc bỏ hình. |
| B3 | **Mọi CI trong toàn bộ bộ hình đều dựa trên G = 10 cluster.** Cluster-robust SE với 10 cluster là anti-conservative; percentile bootstrap trên 10 block under-cover. | Reviewer AAMAS sẽ hỏi wild cluster bootstrap. Phải ghi G trong caption, không giấu. |
| B4 | **`ANALYSIS_REPORT.md` §7 sai sự thật:** *"Claude Haiku 4.5 luôn là model ít hung hăng nhất trong mọi bối cảnh"*. Trong *Gemini vs Haiku*, Haiku hung hăng **hơn** ở cả 3 mức risk (0.554/0.540/0.411 vs 0.494/0.456/0.401). | Không được mang câu này vào paper. Sửa report nguồn. |

Ngoài ra, [`SELECTED_FOR_PAPER.md:111-113`](SELECTED_FOR_PAPER.md) viết `canonical` "nằm gần giữa phân
phối" — **sai**. Canonical (0.5215) đứng **thứ 4 từ dưới lên trong 18** biến thể. Cách đọc đúng lại
*có lợi hơn* cho paper: wording đã freeze là bảo thủ so với không gian wording payoff-identical.

---

## 1. Hình 01 + 02 → **gộp thành 1 forest 6 dòng + 1 bảng**

### Chẩn đoán (đây chính là chỗ "lan man")

Hình 01 dùng **một dạng chart (bar từ gốc 0) cho năm loại test khác nhau**. Cột `test` trong
`human_comparison.csv` ghi rõ: `directional` (E1–E3), `equivalence` (E4, E5),
`directional_effect_size` (E6), `interval` (E7), `upper_bound` (E8).

Hậu quả cụ thể, không phải chuyện thẩm mỹ:

- **Panel E5 nói ngược lại kết luận của chính nó.** Tiêu chí là `|d| < 0.2` (tức *càng nhỏ càng
  đúng*). Hai bar là −0.027 và +0.037, trục y trải −0.03→0.04, nên khoảng cách 0.064 lấp đầy cả
  panel. Panel gán nhãn **"replicated"** trong khi vẽ ra cú vọt trông lớn nhất hình.
- **E7 có tiêu chí "nằm trong [0.40, 0.75]" nhưng dải đó không bao giờ được vẽ** → người đọc không
  thể thực hiện phép test mà panel tuyên bố báo cáo.
- **E8 không có giá trị human.** `human_value` **rỗng** trong CSV. Panel vẫn render verdict
  "not_replicated" đối chiếu với một con số chưa từng được công bố.
- **8 trục y, 4 đơn vị** (log-odds, Cohen's d, tỉ lệ, strategy share trên mẫu số khác) trong một
  lưới small-multiple — small multiple bắt buộc chung thang.
- 🚩 **Error bar trong hình 01 là ±1 SE, không phải 95% CI, và không chỗ nào ghi.** Đã dựng lại:
  E1 `0.0578 ± 0.2918` → bar vẽ [−0.234, +0.350]; E3 `4.6008 ± 1.5919` → [3.01, 6.19]. CI 95% thật
  trong `clustered_logit_coefficients.csv` rộng gần gấp đôi (E3: [1.481, 7.721]). **Hình 02 dùng CI
  95% cho đúng 4 hệ số đó** → hai hình trong cùng một bài đang mâu thuẫn nhau về cùng một ước lượng.
- E5–E8 có `llm_se` rỗng → vẽ bar trần, nhìn không phân biệt được với "độ bất định bằng 0".

**Không thể đưa 8 effect về một thang chung** — E5/E6 không có human SE, E8 không có human value.
Nên phương án "vẽ lại chung một thang" là *bất khả thi*, không phải không thích.

Hình 02 thì đúng dạng (forest cho hệ số) nhưng sai nội dung: 15 dòng gồm formula thô
(`C(persona_condition)[T.R0]`), reference level vô hình, `first_round_unsafe` với CI [1.48, 7.72]
kéo trục ra −4→8 làm hai hệ số risk-treatment (−0.367, −0.350) — đúng estimand chính — teo thành
vạch nhỏ. Bốn dòng persona không được phép ở đó: `seat_balance.csv` cho thấy S_AC và S_CA có một
ghế **0/93 Unsafe ở mọi mức risk** (279 quyết định tất định) — đó là policy kịch bản hoá, và
CLAUDE.md nói rõ cell zero-variance ở lại mức mô tả.

### Spec vẽ lại — forest ghép cặp, một thang log-odds

**Dạng:** dot-and-interval ngang, 2 mark/dòng ở block A, 1 mark ở block B. Một vùng vẽ, hai block
ngăn bằng hairline. Rộng `\linewidth`.

**Trục x:** "Cluster-robust logit coefficient (log-odds of Unsafe)", −2.5 → +5.0.

**Thứ tự sắp xếp (quy tắc cơ học, không cảm tính):** block A giảm dần theo
`|β_LLM − β_human| / SE_human` → 37.8, 9.75, 4.33, 2.86. Trùng với thứ tự sắp theo hiệu tuyệt đối,
nên lựa chọn không phụ thuộc thang.

**Block A — "Dynamic predictors: human study vs LLM pilot"**

| Dòng | Human β [95% CI] | LLM β [95% CI] |
|---|---|---|
| Chose Unsafe in round 1 | +0.217 [−0.010, +0.444] | **+4.601 [+1.481, +7.721]** |
| Own progress lead over opponent | −0.296 [−0.588, −0.004] | **+1.156 [+0.980, +1.332]** |
| Own previous action was Unsafe | −0.193 [−0.569, +0.183] | **−1.024 [−1.736, −0.313]** |
| Opponent's previous action was Unsafe | +0.607 [+0.231, +0.983] | **+0.058 [−0.514, +0.630]** |

CI human = β ± 1.96·SE, SE = 0.116 / 0.149 / 0.192 / 0.192.

**Block B — "Risk treatment (LLM pilot only)"**, reference `max private risk = 0.1`:
0.6 vs 0.1 → **−0.367 [−0.615, −0.119]**; 0.9 vs 0.1 → **−0.350 [−0.570, −0.129]**.

**Encoding:**
- LLM = màu nhấn, human = xám de-emphasis. Dot ≥8px, interval 2px không serif cap. Lệch dọc ±0.18
  trong mỗi band.
- **Bỏ hoàn toàn đỏ/xám theo p<.05.** CI cắt 0 đã mang thông tin đó rồi. Thay bằng: *fill* của mark
  LLM mã hoá `sign_stable` từ `logit_robustness_jackknife.csv` — đặc = dấu ổn định qua cả 10 lần
  refit leave-one-block-out, rỗng = đổi dấu. **Đúng một dòng rỗng: `opponent_prev_unsafe`** (dao
  động −0.094 → +0.237). Đây mới là thông tin reviewer cần và hiện đang bị giấu.
- Dòng 1 có cận trên 7.721 vượt trục → vẽ tới mép trục, kết thúc bằng mũi tên, chú thích
  `upper 95% = 7.72`. Point estimate (4.601) vẫn nằm trong trục.
- **Không đỏ/xanh lá ở bất kỳ đâu.**

**Bỏ khỏi hình 02:** intercept, 5 dummy persona, 4 số hạng tương tác — 10/16 dòng. Cả 16 dòng đi
vào bảng phụ lục (chính là `clustered_logit_coefficients.csv` lọc `specification == 6`).

**n:** 3 486 quyết định (round ≥ 2), 210 race, 420 player, **10 CRN block**, một checkpoint
(`qwen2.5:7b-instruct-fp16`), 6 persona condition, `ai-race-fairgame-v3`, parse_failure_rate = 0.0
ở cả 18 cell, `run_phase = pilot`. Human: 2 888 participant-round, 338 người, 172 pair cluster.

### Bảng thay cho E5–E8

| Effect | Loại test | Human | LLM pilot | Tiêu chí | Đạt? |
|---|---|---|---|---|---|
| E5 risk 0.6 vs 0.9, Cohen's d | equivalence | −0.027 | **+0.037** | \|d\| < 0.2 | có |
| E6 risk 0.1 vs 0.6, Cohen's d | direction + size | +0.341 | **+0.097** | d>0 và \|d\|>0.2 | không (đúng dấu, thiếu độ lớn) |
| E7 φ_U tổng thể | interval | 0.584 | **0.538** | trong [0.40, 0.75] | có |
| E8 share Always-Safe | upper bound | **không công bố** | **0.449** = 157/350 | share < 0.1 | không |

Hai điều bảng **bắt buộc** nói mà hình cũ giấu:
1. E5 và E6 là **trung bình của 6 giá trị Cohen's d theo persona mà dấu không thống nhất** (biên độ
   −0.465 → +0.433). Một cái bar đang che một khoảng như thế.
2. 🚩 **Mẫu số của E8 là 350, không phải 420.** 70 player có nhãn nearest-strategy hoà bị loại
   (`strategy_analysis/classify.py` giữ tie thay vì ép nhãn). Con số này không nằm trong metadata
   nào — phải ghi ra.

### Caption

> Cluster-robust logistic coefficients (log-odds of Unsafe) for the four dynamic predictors of
> Fernández Domingos and Han's Table 1 model 6 and the two risk-treatment contrasts, from a
> single-checkpoint pilot (Qwen2.5 7B Instruct F16; 210 races, 3 486 round-level decisions,
> 10 common-random-number blocks, zero parse failures, `run_phase = pilot`), beside the published
> human estimates (2 888 participant-rounds, 172 pair clusters). Intervals are 95% cluster-robust on
> the repetition block and are anti-conservative at ten clusters; no coefficient here is a
> confirmatory estimate.

### Vị trí

`\subsection{Human-reference comparison pilot}` **mới**, chèn sau `\FloatBarrier` dòng 222, **trước**
`\subsection{Maximum-risk treatment effects}` dòng 223 — anh em thứ 7 của các mục pilot đã có.
**Tuyệt đối không** đặt vào dòng 225/229/233: sáu dòng của hình này trả lời *đúng* ba câu hỏi
`\pending`, thả vào đó là biến pilot thành confirmatory qua cửa sau.

---

## 2. Hình 03 → **2 panel, sắp theo panel A**

### Chẩn đoán

- **Legend 12 màu cho 18 điểm là trang trí.** `family` có 12 level, **8 là singleton**. Thông tin đã
  nằm sẵn trong nhãn trục y (`format_*`, `order_*`). Bảng màu là lát `tab20` → control/lexical/
  position là ba pastel gần trùng, noise/order là cặp đỏ/hồng, syntax/whitespace là cặp nâu/nâu
  nhạt. Trượt CVD trước cả khi chạy validator. Legend lại nằm **trong** vùng vẽ.
- **Reference level không được neo.** `canonical` là chấm xanh nhạt ở dòng 15, cùng trọng số thị
  giác với 17 thứ đang được so *với nó*. Đường dashed không nhãn, không CI band.
- **Đại lượng đang vẽ không đỡ nổi claim của hình.** `analysis.json:14` nói thẳng: *"Whole-trajectory
  unsafe-rate differences include feedback from earlier actions."* Sau vòng 1 hai nhánh ở trạng thái
  game khác nhau. Trên metric này **11/17 CI loại trừ point estimate của canonical** → hình đang
  chứng minh *giúp* phản biện của reviewer, không phải chống lại nó.

### `order_actions_reversed` — **không phải artifact.** Đã kiểm ba đường

1. `parse_failures = 0` cho biến thể đó (cả 558 response khớp contract).
2. `ai_race/prompts/sensitivity.py:44-54` chỉ hoán vị *thứ tự* `{strategy1}`/`{strategy2}`; token
   không đổi. `ai_race/engine/round.py:49` parse bằng `Action.coerce(...)` — **theo nhãn, không có
   mapping theo vị trí ở bất kỳ đâu trong pipeline**.
3. Test quyết định: ở vòng 1, cùng state, cùng seed, nó lật **2/60** quyết định so với canonical.
   Nếu là đảo mapping nhãn thì flip rate vòng 1 phải ~0.80–0.97. Thực tế 0.033.

→ Toàn bộ khoảng cách −43.7 pp **mở ra sau vòng 1**. Đây là phân kỳ hành vi thật dưới prompt
payoff-identical, nhưng nằm trọn trong phần trajectory bị nhiễm feedback. Thiết kế 2 panel dưới đây
xử lý chuyện này bằng cấu trúc, không bằng chú thích.

### Pooling theo risk là hợp lệ (đã kiểm)

`risk_variant_summary.csv`, n=186/cell: biên độ trong-biến-thể qua 3 mức risk trung vị **5.4 pp**,
max 11.8 pp. Biên độ giữa-biến-thể ở risk cố định: **0.817 / 0.806 / 0.801**. Tỉ lệ ≈ 15:1, thứ hạng
giữ nguyên ở cả 3 mức. Không có tương tác variant × risk.

### Spec vẽ lại

**Dạng:** 2 panel dot-and-interval ngang, chung trục y, `figure*` (2 cột), ~7.0 × 4.2 in, 18 dòng.

**Panel A (trái — mang claim) — Round-1 paired flip rate vs canonical.** n = **60 quyết định vòng 1
ghép cặp** mỗi biến thể (30 race × 2 ghế), cùng state, cùng sampling seed. CI cluster-bootstrap.
→ **15/15 biến thể meaning-preserving đều ≤ 0.15**; chín biến thể ở 0.033 (2/60);
`position_risk_near_response` 0.017. Ngoại lệ duy nhất: **`emotional_importance` 0.833 (50/60)** —
đúng cái biến thể `behavioral_framing` được thiết kế để *không* bảo toàn nghĩa.
`order_actions_reversed` nằm lẫn trong khối phẳng ở 0.033.

**Panel B (phải) — Whole-trajectory Unsafe rate.** n = 558 quyết định/biến thể, dải 0.084 → 0.892.
Subtitle panel ghi thẳng: *"includes endogenous state feedback — not a controlled wording contrast."*

**Thứ tự (chung cả 2 panel): giảm dần theo Panel A.** Đây là toàn bộ ý đồ của hình:
`emotional_importance` đứng riêng trên đỉnh, `order_actions_reversed` chìm vào khối phẳng ở A nhưng
vẫn là cực trị ở B. Người đọc thấy sự phân ly như một *hình dạng*, không phải một câu văn.

**Neo canonical:** panel A — đường liền 1.5px tại 0, nhãn inline `canonical (reference)`. Panel B —
đường liền tại **0.5215** với band **[0.480, 0.560]** (chính CI của canonical), nhãn `canonical 0.52`.
Đường **liền**, không dashed. Dòng `canonical` vẫn nằm trong danh sách nhưng mực trung tính, marker
rỗng.

Panel B thêm **trục phụ trên cùng ghi Δ pp so với canonical** (−45 … +40) — đây là đổi nhãn tuyến
tính của *cùng một thang*, không phải dual-axis, nên hợp lệ.

**`family` → bỏ khỏi kênh thị giác.** Thay bằng cột `interpretation` đã có sẵn trong CSV, 3 mức:
`meaning_preserving` (15, chấm tròn) / `behavioral_framing` (1, hình thoi) / `robustness_perturbation`
(1, hình vuông) / `control` (canonical, xám, tròn rỗng). Ba series nằm trong ngưỡng thoải mái;
shape redundancy → an toàn khi in đen trắng. `family` chuyển sang bảng phụ lục.

**Nhãn:** thay ID snake_case bằng cụm dễ đọc (`position_risk_near_response` → "risk rules moved next
to the answer"). Ở bề rộng cột AAMAS, ID hiện tại không đọc được.

🚩 **Không có CI cho Δ ở bất kỳ đâu.** Các interval hiện tại là marginal và seed độc lập
(`seed_label=f"unsafe:{variant}"`, `analyze_surface_sensitivity.py:199-203`) → mỗi biến thể được
resample trên một draw rep-block *khác nhau*. Trừ chúng cho nhau sai hai lần. Cách xử lý ở trên
(vẽ rate thật + trục phụ Δ) là lối thoát; cách sửa đúng là **~10 dòng trong analyser**: bootstrap Δ
ghép cặp trên các `rep` chung với một common resample mỗi draw.

### Companion: giữ **đúng một** — `surface_first_round_direction_stacked` (vẽ lại)

Nó mang thông tin không cái nào khác có: **dấu** của các lần lật. Chuyện 50 flip của
`emotional_importance` là **48 Safe→Unsafe / 2 Unsafe→Safe** (cue cảm xúc làm model liều hơn — một
claim có hướng) không suy ra được từ flip rate. Diverging stacked bar quanh 0 đã đúng dạng. Sửa: kéo
trục x tới 60, khe 2px tại 0, legend ra ngoài, **giữ đúng thứ tự dòng của hình chính**.

**Loại ba cái còn lại:**
- `surface_family_boxplot` — **xoá.** 8/12 hộp vẽ từ một điểm dữ liệu. Boxplot của n=1 không phải
  phân phối.
- `surface_variant_by_risk_heatmap` — **hạ xuống bảng phụ lục.** Cả 54 ô đều in số → màu mã hoá lại
  thứ đã đọc được; lại dùng diverging map (RdYlBu_r) cho đại lượng độ lớn với midpoint neo vào
  không gì cả. Phát hiện của nó là một kết quả âm, đáng một câu.
- `surface_variant_pilot_vs_smoke` — **loại**, ngược với đề xuất cũ. (i) smoke chỉ 6 race / 72 quyết
  định / 12 quyết định vòng 1 mỗi biến thể; (ii) 🚩 **hai run dùng estimator khác nhau** — smoke là
  `unsafe_rate_ci95_*` (binomial trên quyết định phụ thuộc), pilot là
  `unsafe_rate_cluster_bootstrap_ci95_*` — và dumbbell lặng lẽ bỏ cả hai interval, đúng chỗ giấu sự
  lệch pha; (iii) ổn định thứ hạng chỉ đúng ở hai cực, giữa xáo mạnh (`format_numbered_state`
  hạng 13 → 6, `format_dense` hạng 8 → 15).

### Caption

> Under matched game state and matched sampling seeds, all 15 meaning-preserving surface rewrites of
> the frozen v3 prompt flip at most 9 of 60 paired first-round decisions (≤15%); the sole exception
> is `emotional_importance` (50/60), a deliberately non-meaning-preserving framing cue.
> Whole-trajectory Unsafe rates (right) diverge far more widely (0.084–0.892 against canonical 0.522,
> n=558 per variant) because the states themselves differ after round 1 — Qwen2.5-7B-Instruct,
> `run_phase="pilot"`, 30 races and 10 repetition blocks per variant, 0 parse failures.

---

## 3. Hình 04 → **slopegraph ghép cặp, facet theo risk, bỏ hết dòng N=3**

### Chẩn đoán

**V1 — hai cơ chế khác nhau trên cùng một trục xếp hạng. ĐÃ XÁC NHẬN.**
`full_analysis.py:453-461` append các dòng `N3_TURNS` với `context="3-way race"` vào **cùng** list
`rows` rồi `sort_values("mean")` xếp hạng cả 9 dòng. Turn N=3 mang `ai-race-nplayer-v1`,
`n_players: 3`; turn cặp mang `ai-race-fairgame-v3`, `n_players: 2` (đã kiểm trong `turns.jsonl`
thô). Payoff khác về cấu trúc, không chỉ về thang: prompt N=3 chia lợi ích thị trường cho ba
("If 0 of 3 companies choose SAFE: each UNSAFE company earns 1.333"). Limitation #4 của chính report
nói đúng điều này — mà hình headline thì vi phạm.

**V2 — pooling qua risk, trong khi risk có main effect lớn ngang effect đang xếp hạng.**
Biên độ theo đối thủ (pooled) +0.074 → +0.175. Biên độ theo risk trong một cell −0.093 → −0.180.
Cùng cỡ. Phân tầng theo risk **làm đổi 4/6 thứ hạng**. Xếp hạng pooled là artifact của việc trung
bình qua nhân tố bị bỏ quên.

**V3 — forest sắp xếp là sai dạng cho claim này; thiết kế có ghép cặp mà hình vứt bỏ ghép cặp.**
Ba matchup dùng chung `game_seed` mỗi rep (260801–260810) và chung horizon mỗi rep
(6,8,12,5,9,8,17,6,5,10) → **CRN block trải qua các matchup**. Claim "hung hăng phụ thuộc đối thủ,
không phải hằng số của model" là contrast *trong-model, giữa-đối-thủ*, và ước lượng ghép cặp rõ ràng:

| model | vs A | vs B | **paired Δ [95% t-CI], n=10 rep khớp** |
|---|---|---|---|
| GPT Luna | Haiku 0.462 | Gemini 0.630 | **+0.167 [+0.117, +0.218]** |
| Gemini 3.5 Flash-Lite | Haiku 0.450 | Luna 0.625 | **+0.175 [+0.130, +0.219]** |
| Claude Haiku 4.5 | Luna 0.428 | Gemini 0.502 | **+0.074 [+0.025, +0.122]** |

Cả ba đều loại trừ 0 — **phát biểu mạnh hơn hẳn** so với những gì forest sắp xếp đỡ được. Hình hiện
tại thay vào đó tính t-interval độc lập không ghép cặp mỗi dòng rồi mời người đọc so bằng mắt các
interval chồng nhau (63% vs 62%; 46%/45%/43% chồng lẫn nhau). Nó xếp hạng thứ nó không tách được,
đồng thời vứt đi contrast tách được sạch sẽ.

**V4 — lỗi encoding.** Model identity mã hoá **ba lần** (nhãn y, màu chấm, legend) → legend thừa
hoàn toàn. Nhãn `%` vẽ tại `hi + 0.02` (dòng 471) → vị trí x của nhãn biến thiên theo độ rộng CI,
quét cột nhãn là đọc nhiễu. `axvline(0.5)` là mốc không nhãn và **không có ý nghĩa gì trong game
này** — 50% không phải Nash, không phải ngưỡng, không phải null. `xlim(-0.02, 1.12)` trong khi dữ
liệu trải 0.35–0.75 → data chiếm ~36% trục, phần dư chỉ để đỗ nhãn.

### Spec vẽ lại

- **Dạng: slopegraph** (dot ghép cặp nối bằng đường). Không phải forest. Claim là contrast
  trong-model trên thiết kế khớp cặp → ghép cặp phải là đơn vị thị giác gốc.
- **3 panel cạnh nhau: risk 0.1 | 0.6 | 0.9.** Không bao giờ pool risk. Chung trục y 0→1.
- **Trục x: 2 vị trí ordinal** — `opponent = Claude Haiku 4.5` và `opponent = model frontier còn lại`.
  Ghi tên **đối thủ**, không phải chuỗi matchup (sửa V4: hiện "Gemini vs Luna" là nhãn context cho
  player "GPT Luna", bắt người đọc tự suy ra đối thủ là Gemini).
- Mỗi model một đường nối 2 context, 2px, endpoint dot ≥8px, vòng 2px màu nền chỗ đường cắt nhau.
  Màu = model, cố định xuyên suốt mọi hình trong paper.
- **Uncertainty: paired Δ là estimand** → đặt ở panel hẹp kề bên dưới dạng dot-and-interval trên
  thang *hiệu*, có đường 0. **Không** vẽ CI không ghép cặp trên các mức (đó chính là lỗi V3).
- **Bỏ:** đường mốc 0.5, padding trục tới 1.12, nhãn đặt theo độ rộng CI. **Bỏ toàn bộ dòng N=3.**

**Số liệu (mean của rep-mean; n = 10 rep khớp CRN mỗi cell; 258 quyết định mỗi matchup):**

| model | risk | vs Haiku | vs đối thủ kia | Δ |
|---|---|---|---|---|
| GPT Luna | 0.1 / 0.6 / 0.9 | 0.546 / 0.464 / 0.376 | 0.699 / 0.653 / 0.537 (Gemini) | +0.153 / +0.189 / +0.161 |
| Gemini 3.5 FL | 0.1 / 0.6 / 0.9 | 0.494 / 0.456 / 0.401 | 0.716 / 0.621 / 0.536 (Luna) | +0.222 / +0.165 / +0.135 |
| Claude Haiku | 0.1 / 0.6 / 0.9 | 0.502 / 0.429 / 0.354 (vs Luna) | 0.554 / 0.540 / 0.411 (Gemini) | +0.052 / +0.111 / +0.057 |

**Companion tuỳ chọn** (giữ được vật liệu N=3 một cách hợp lệ): một panel nhỏ **riêng, đóng khung
rõ**, caption ghi rõ là cơ chế khác, hiển thị rate pooled 3-way — Luna 0.692 [0.651, 0.734], Gemini
0.660 [0.634, 0.686], Haiku 0.615 [0.584, 0.646], n=10 rep. **Không bao giờ chung trục** với các
dòng 2 người.

### Caption

> Unsafe choice rate for each model against each of its two opponents, shown separately at each
> maximum-private-risk level, with the paired within-model difference at right; n = 10
> common-random-number-matched repetitions per cell (258 decisions per matchup), two-player mechanism
> only (`prompt_version = ai-race-fairgame-v3`). All three models are more Unsafe against the
> non-Haiku opponent (paired Δ +0.167, +0.175, +0.074; all 95% t-intervals exclude zero), so
> aggression is opponent-contingent rather than a per-model constant; pilot data (`run_phase =
> "pilot"`, no `run_manifest.json`), not confirmatory evidence.

---

## 4. Hình 05 → **thay hẳn bằng position effect 2 người**. Đây là thay đổi có giá trị nhất trong kế hoạch

### Chẩn đoán hình N=3 hiện tại

**V6 — bar "Ahead of both" gán nhãn sai, và phần lớn không hề "ahead". Đây là lỗi nghiêm trọng nhất
trong cả 8 hình.** `full_analysis.py:620-624` định nghĩa rank bằng `behind_of = sum(o > own for o in
others)` — so sánh **chặt** — rồi gán `0 → "Ahead of both"`. **Hoà rơi vào rank 0.** Phân rã 792
quyết định N=3:

| trạng thái thật | n | tỉ lệ | unsafe rate |
|---|---|---|---|
| hoà cả ba | 282 | 35.6% | 0.631 |
| hoà đầu với một người | 140 | 17.7% | 0.529 |
| **dẫn thật sự (ahead of both)** | **100** | **12.6%** | **0.380** |
| Middle | 163 | 20.6% | 0.896 |
| Behind both | 107 | 13.5% | 0.794 |

Cả **90 quyết định vòng 1** (mọi người ở progress 0) đều bị đếm là "Ahead of both". Rate dẫn-thật là
**0.380**, không phải ~0.55 như hình pool ra.

**V7 — hiệu ứng không đơn điệu → khung ordinal đang over-claim.** Pooled: Ahead 0.380 → Middle 0.896
→ Behind 0.794. **Middle > Behind.** Risk 0.1 và 0.6 đều không đơn điệu; chỉ risk 0.9 đơn điệu. Tức
là *không phải "đi sau"* dự báo liều lĩnh, mà *"không dẫn đầu"* mới dự báo. Điều này làm yếu đúng
cách đọc "falling behind" mà hình được chọn để minh hoạ.

**V8 — n không đều và không công bố; CI bị clamp mà hiển thị như chính xác.** n_rep mỗi cell là 8, 9
hoặc 10 chứ không hằng số (Behind@0.9 chỉ 8 cluster / 30 quyết định), trái với caption "95% CI across
reps". Bốn interval bị `min(1.0, ...)` kẹp tại 1.0 và render như thể kết thúc đúng trần.

**V9 — dynamite plot.** Bar + error bar cho tỉ lệ: thân bar neo ở 0 và khẳng định độ-lớn-từ-0, nhưng
toàn bộ tín hiệu nằm ở 0.45–0.95 → thân bar là mực chết, CI mới là nội dung. Nhãn `%` đặt tại
`m + 0.03` **đâm vào whisker trên ở cả 9 bar** (thấy rõ trong PNG). `DANGER3` mã hoá rank trùng lặp
với vị trí x, và **cùng bảng màu đó mang nghĩa "số người chơi unsafe (0/1/2)" trong `x10b`** — cùng
màu, khác thực thể, vi phạm "color follows the entity".

### Bản thay thế 2 người tồn tại, cùng 3 model, cùng corpus, và mạnh hơn hẳn

| | Hình 05 N=3 hiện tại | Bản thay 2 người |
|---|---|---|
| cơ chế | `ai-race-nplayer-v1` — **không phải** game của paper | `ai-race-fairgame-v3` — **đúng** game của paper |
| effect size | ~0.30–0.45 pp, và gán nhãn sai | **+0.79 pooled**, +0.74 → +0.86 theo risk |
| ghép cặp | không | **chính xác** — mỗi vòng bất đối xứng cho đúng 1 ahead + 1 behind (đã kiểm 96/96, 77/77, 83/83) |
| cluster/cell | 8, 9 hoặc 10 (không đều, không công bố) | **10 ở mọi cell** |
| đơn điệu | không | có, ở mọi mức risk |
| nhãn hạng | hỏng (V6) | trichotomy đúng — `analyze_ai_race.py:1974-1977` dùng `np.select` dấu chặt, có category `tied` thật |

**Số liệu — cross_provider 2 người, cả 3 matchup, n=10 rep/cell:**

Pooled qua risk và matchup (1 548 quyết định có state xác định):

| state | n | tỉ lệ | mean | 95% t-CI (10 rep) |
|---|---|---|---|---|
| Ahead | 256 | 16.5% | **0.190** | [0.120, 0.260] |
| Tied | 1 036 | 66.9% | **0.483** | [0.447, 0.520] |
| Behind | 256 | 16.5% | **0.983** | [0.962, 1.000] |

Theo risk:

| risk | Ahead (n) | Tied (n) | Behind (n) | **paired Δ behind−ahead [95% CI], n=10** |
|---|---|---|---|---|
| 0.1 | 0.183 (82) | 0.594 (352) | 0.988 (82) | **+0.856 [+0.755, +0.957]** |
| 0.6 | 0.250 (96) | 0.463 (324) | 0.979 (96) | **+0.739 [+0.600, +0.879]** |
| 0.9 | 0.154 (78) | 0.394 (360) | 0.949 (78) | **+0.806 [+0.704, +0.908]** |

**Phát hiện phụ đáng một câu:** hiệu ứng vị trí gần như bất biến theo risk (Behind 0.99→0.95, Ahead
0.18→0.15), trong khi **toàn bộ gradient risk sống trong trạng thái hoà** (0.594 → 0.463 → 0.394).
Vị trí lấn át risk; risk chỉ cắn khi cuộc đua đối xứng.

Theo model (pooled risk + matchup): Gemini 3.5 Flash-Lite có **Ahead 0.000 (n=62) và Behind 1.000
(n=111)** — zero variance. Theo CLAUDE.md, giữ nguyên mức mô tả, **không fit model** trên cell này,
và nói rõ là tất định.

### Nguồn tốt hơn nếu tập model cho phép

`results/derived/two_player_paper_analysis/tables/baseline_position_estimates.csv` — 5 checkpoint
frontier, 10 CRN block mỗi cái, percentile cluster bootstrap trên `source_run × repetition`
(5 000 rep, seed 260802), và **cả 5 đều có `run_manifest.json` băm SHA-256**. Đây là nguồn *có
provenance*, khác với toàn bộ corpus `cross_provider` (B1). Ưu tiên nguồn này nếu tập model của
paper cho phép.

### Ranh giới bắt buộc báo cáo, không được giấu

`results/open_source/prompt_sensitivity_pilot/unsafe_by_race_state_turn.csv` (qwen2.5:7b-instruct-fp16)
cho hiệu ứng **ngược lại**: Ahead 0.845 vs Behind 0.667 ở risk 0.1; Ahead 0.833 vs Behind 0.583 ở
risk 0.9. Và `gpt-5.4-nano` thì phẳng (0.621 vs 0.579, CI chồng). Nên claim phải phát biểu ở phạm vi
**checkpoint-scoped**, đúng theo lệnh cấm khái quát hoá "all LLMs" trong CLAUDE.md.

**Caveat nhân quả bắt buộc có trong text:** vị trí là biến **hậu-hành-động và nội sinh** — một
người đi sau *bởi vì* đã chơi Safe trước đó, nên điều kiện hoá theo nó là điều kiện hoá theo biến
post-treatment. `two_player_eda_report.md` nói rõ điều này; `ANALYSIS_REPORT.md` §5 thì **không**, và
còn gọi đây là "hiệu ứng phòng thủ vị trí dẫn đầu" — ngôn ngữ nhân quả mà thiết kế không đỡ được.

### Spec vẽ lại

Dot-and-interval (**không bar**), trục category ordinal Ahead → Tied → Behind, 3 panel theo risk,
chung trục x 0→1. CI t 95% cluster theo rep, n=10 mọi cell. **Đánh dấu rõ các cận bị clamp** bằng
cap rỗng thay vì để nó đọc như chính xác. Direct-label 3 điểm, bỏ nhãn `%` va chạm. Không dùng lại
ramp `DANGER3`. Thêm inset paired Δ (behind − ahead) — có nghĩa ở đây vì thiết kế 2 người ghép cặp
chính xác trong từng vòng.

### Verdict N=3

**Bỏ khỏi thân bài.** Nó không mua thêm gì mà dữ liệu 2 người không cho sạch hơn, và nó tốn của
paper một phản biện trộn-cơ-chế trên đúng cái hình dễ trích dẫn nhất. **Không** đưa vào thân bài kèm
disclaimer: disclaimer không sửa được một cái trục xếp hạng hai cơ chế, và nó mời đúng câu hỏi
reviewer mà dữ liệu pilot không trả lời nổi. Chuyển bản N=3 **đã sửa** (tách "Ahead of both" thành
hoà-ba-chiều / hoà-đầu / dẫn-thật) sang companion paper N-player hoặc phụ lục có rào rõ.

### Caption

> Unsafe choice rate by race position (ahead / tied / behind on progress before the decision), pooled
> over the three two-player matchups and shown separately at each risk level; n = 10 repetition
> clusters per cell, 1 548 decisions (ahead 256, tied 1 036, behind 256), `ai-race-fairgame-v3`, risk
> not pooled. Being behind raises the Unsafe rate by +0.79 [+0.71, +0.88] over being ahead, an effect
> essentially invariant to risk whereas the risk gradient appears only in the tied state
> (0.594→0.394); positions are post-action and endogenous, so this association is descriptive, not
> causal.

---

## 5. Hình 06 → **SWAP, nhưng ship bản vẽ lại, không phải bản ứng viên hiện tại**

### Chẩn đoán

- **Trục x không phải liên tục cũng không phải phân loại.** `set_xticks(RISK_ORDER)` vẽ tại giá trị
  thật 0.1/0.6/0.9 nhưng render ra ba tick **cách đều** → Δ=0.5 và Δ=0.3 vẽ cùng bề rộng. Chữ "V"
  của GPT-5.4 nano (57.7 → 49.6 → 56.7) bị phóng đại bởi khoảng cách sai đó.
- **Vi phạm thật nằm ở CI *band*.** `fill_between(...)` tô một vùng 95% liên tục qua những giá trị
  risk chưa từng chạy. Ba ước lượng rời rạc không cấp phép cho một vùng liên tục. Cái này nặng hơn
  đường nối.
- **Hai CI suy biến được vẽ lặng lẽ.** Gemini 3 Flash và 3.1 Flash-Lite ở risk 0.1 đều
  `estimate=1.0, ci95_low=1.0, ci95_high=1.0` — cả 10 block (20 player-race) đúng bằng 1.0, nên mọi
  bootstrap resample đều là 1.0. Interval suy biến **do cấu trúc**, không phải do chính xác. Vẽ ra
  thành band bề rộng 0 thì nó đọc như ước lượng chặt nhất hình — ngược hẳn sự thật. Hai series còn
  chồng gần khít tại đó → người đọc thấy một mark ở chỗ có hai.
- 🚩 **p-value không nên xuất hiện, và nếu xuất hiện thì hiện đang không dựng lại được.**
  `holm_p_high_vs_low` được hiệu chỉnh trên **họ 7 model**, không phải 5 model được vẽ (kiểm số học:
  0.001953125 × 7 = 0.013671875 ✓ và 0.00390625 × 5 = 0.01953125 ✓; họ gồm cả Claude Haiku 4.5
  n_blocks=3 và Qwen2.5 7B n_blocks=20, đều bị bỏ khỏi hình). Ngoài ra hai p Wilcoxon nhỏ nhất
  (0.001953125) **đang ở sàn** của signed-rank n=10 (2/2¹⁰) — chúng đo cỡ mẫu, không đo độ lớn hiệu
  ứng.

**A và B bổ sung nhau, không trùng lặp — có bằng chứng.** Marginal của GPT-5.4 nano chồng nhau nặng
(57.7 [52.0, 63.5] vs 56.7 [53.0, 60.7]) nhưng contrast **ghép cặp theo CRN block** là −1.0 pp
[−6.5, +3.8] — **hẹp hơn cả hai marginal**. Chính sự thu hẹp đó là toàn bộ phần thưởng của thiết kế
common-random-number, và mắt thường không đọc ra được từ panel A. Giữ cả hai.

### Spec vẽ lại

**Panel A:** dot-and-interval nối, **trục x tuyến tính thật** 0→1.0, tick chỉ tại 0.10/0.60/0.90.
**Thay `fill_between` bằng error bar dọc rời rạc** tại ba điểm đã lấy mẫu — bỏ band. Đường nối giữ
lại nhưng hạ xuống 1px ~40% opacity, và caption ghi rõ đó là guide đọc, không phải nội suy. 5 series
mỗi cái một màu + **một marker riêng** (composite encoding, an toàn CVD). **Cell bão hoà:** hai điểm
100% dùng marker rỗng, cap phẳng tại 1.00, một chú thích chung *"saturated: 20/20 player-races
Unsafe; bootstrap CI degenerate"*, và jitter ±0.006 trên x + vòng 2px màu nền để thấy được cả hai.

| model | risk 0.1 | risk 0.6 | risk 0.9 |
|---|---|---|---|
| Gemini 3 Flash | **100.0** [100.0, 100.0] † | 72.3 [69.1, 75.5] | 53.9 [48.6, 59.6] |
| Gemini 3.1 Flash-Lite | **100.0** [100.0, 100.0] † | 80.1 [77.4, 82.6] | 69.9 [66.7, 73.1] |
| Gemini 3.5 Flash-Lite | 83.8 [78.5, 89.9] | 70.8 [65.4, 76.4] | 62.6 [59.4, 66.2] |
| GPT-5 nano | 12.3 [7.1, 17.2] | 15.1 [8.6, 21.6] | 14.5 [8.4, 20.6] |
| GPT-5.4 nano | 57.7 [52.0, 63.5] | 49.6 [43.5, 55.5] | 56.7 [53.0, 60.7] |

† suy biến. Mọi cell: n_blocks = 10, n_observations = 20 player-race.

**Panel B:** giữ forest contrast, **bỏ p-value khỏi hình**. Đường 0 là hairline **liền** (dashed đọc
như "ngưỡng"). Thay chú thích `n=10` bằng sign-consistency `10 blocks · k/10 ↓` — đó mới là nội dung
phi tham số thực có ở n=10. Tiêu đề panel: "Paired within CRN repetition block".

| model | 90% − 10% | 95% CI |
|---|---|---|
| Gemini 3 Flash | −46.1 pp | [−51.4, −40.5] |
| Gemini 3.1 Flash-Lite | −30.1 pp | [−33.1, −27.0] |
| Gemini 3.5 Flash-Lite | −21.2 pp | [−28.8, −13.7] |
| GPT-5 nano | **+2.2 pp** | [−5.2, +10.8] |
| GPT-5.4 nano | −1.0 pp | [−6.5, +3.8] |

Nếu reviewer đòi p-value: đưa vào bảng, kèm ghi chú Holm đã hiệu chỉnh trên họ 7 model.

### Verdict swap

**SWAP.** Bản đương nhiệm `results/impact_upgrade/figures/cross_model_risk_response.pdf` có cùng
point estimate và: (1) **không có uncertainty ở bất kỳ đâu** — trong khi mục Outcomes của chính
main.tex cam kết *"Estimates will include uncertainty intervals"*; (2) **không có panel contrast**,
nên không đỡ nổi các claim trong-model mà prose đang phát biểu ("remained low and nearly flat",
"was non-monotone", "declined as risk increased"); (3) legend đè lên chuỗi dữ liệu ở ~13%;
(4) marker toàn hình tròn, cụm blue/cyan/teal kề nhau — rủi ro CVD; (5) tiêu đề tự khẳng định kết
luận ("qualitatively different risk response") mà không có gì chống lưng.

Cả hai đều dính lỗi trục cách đều; chỉ bản ứng viên có lỗi band, nhưng cũng chỉ bản ứng viên sửa
được thành thứ đỡ được prose.

### Caption

> Player-level Unsafe rate by maximum private setback risk for five checkpoints (A) and the paired
> within-repetition-block 90%-minus-10% contrast (B); each cell is ten races and twenty player
> trajectories, with 95% percentile bootstrap intervals clustered on the repetition block. Two Gemini
> cells are saturated at 100% so their intervals are degenerate rather than precise; all runs are
> exploratory pilots on non-protocol-matched provider routes and are juxtaposed descriptively.

---

## 6. Hình 07 → **giữ cấu trúc, vẽ lại panel B** (nếu B2 được gỡ)

### Chẩn đoán

- **Ba bar không có uncertainty, và "n=74" là mẫu số gây hiểu lầm.** Đã khôi phục nguồn đã xoá
  (`git show a81a8c8:ai_race/results/_api_5games_allrisk/.../races.csv`): pilot cũ là **5 rep**, seed
  260726–260730, horizon 5/7/9/11/5 = 37 vòng × 2 ghế = **đúng 74**. Nên mỗi bar dựa trên **5 race
  seed / 10 player-trajectory**, và *cùng* 5 seed đó lặp lại ở cả ba mức risk → ba bar cũng không
  độc lập với nhau. In "n=74" cạnh một cái bar mời gọi một Wilson interval ±10 pp, trong khi độ bất
  định cluster trung thực rộng hơn nhiều và đứng trên 5 block — đúng ngưỡng "descriptive only" của
  CLAUDE.md.
- **Bar 100% là artifact, đã xác nhận.** Cả hai run đều đúng 1.0 ở risk 0.1, và cả 5 race cũ đều có
  `player_1_unsafe_frequency = player_2_unsafe_frequency = 1.0`. Khi cả hai run đồng loạt Unsafe,
  agreement **bị ép** bằng 100%. Nó mang thông tin bằng 0 về reproducibility.
- **65% agreement chỉ nhỉnh hơn ngẫu nhiên — và đây mới là phát hiện mà hình đang chôn.** Chance
  agreement xấp xỉ từ marginal của chính hai run: risk 0.6 → ≈59.6% (quan sát 64.9%, κ ≈ 0.13);
  risk 0.9 → ≈50.6% (quan sát 64.9%, κ ≈ 0.29). "65%" đọc như "khá tái lập được"; κ ≈ 0.13 nói
  "hai lần rút độc lập từ cùng một đồng xu lệch".
  🚩 **Các κ này là suy ra bằng tay, không nằm trong CSV nào — phải tính lại từ 74 quyết định khớp
  trước khi dùng.**

**Panel A và B trên chung trục x là điểm mạnh thật**, không phải lỗi: "ổn định ở mức tổng hợp, bất
ổn ở mức quyết định" đúng là luận điểm phương pháp, và đặt chung trục risk là thứ làm mâu thuẫn đó
đọc được. Giữ cấu trúc, sửa encoding panel B.

### Spec vẽ lại

**Panel A:** giữ nội dung, áp dụng cùng cách sửa trục x tuyến tính, thêm số block vào legend —
"Earlier 5-rep pilot (10 player-races/cell)" và "Final 10-rep pilot (20 player-races/cell)". Giá trị
100.0 / 71.3 / 57.7 vs 100.0 / 72.3 / 53.9.

**Panel B: thay bar bằng dot-plot trên 5 race seed dùng chung.** Mỗi mức risk vẽ **5 tỉ lệ agreement
theo từng seed** dưới dạng chấm nhỏ (vòng 2px màu nền) cộng một crossbar đậm cho tỉ lệ pooled. Với 5
cluster, đây là encoding trung thực: nó *cho thấy* đơn vị lấy mẫu thay vì khẳng định một interval mà
bootstrap 5 block không ổn định nổi. Nếu bắt buộc phải có interval thì phải là **cluster bootstrap
trên 5 `game_id`**, tuyệt đối không Wilson trên n=74. Ghi trên trục: "5 shared race seeds · 74 matched
decisions". Chú thích trực tiếp risk 0.1: *"forced — both runs 100% Unsafe"*. Thêm **đường mốc
chance-agreement** mỗi mức risk để 64.9% được đọc đúng nền so sánh.

Giá trị: 74/74 = 100.0%; 48/74 = 64.9%; 48/74 = 64.9%.

### Caption

> The same Gemini 3 Flash protocol rerun without a fixed decoding seed reproduces the aggregate risk
> response (A: 100.0/71.3/57.7% versus 100.0/72.3/53.9%) but not the individual decisions (B: matched
> game-round-seat agreement of 74/74, 48/74 and 48/74 over five shared race seeds). Agreement at risk
> 0.1 is forced because both pilots were uniformly Unsafe there, so repeated unseeded API runs are
> reported as separate pilots, never as interchangeable replicates.

---

## 7. Hình 08 → **giết panel phải, giữ panel trái, chuyển xuống phụ lục**

### Chẩn đoán

- **Bar vàng của Rule Recall là số 0 THẬT, không phải thiếu dữ liệu.** `rule_recall,1.0,0.0,64` —
  strict format validity đúng bằng 0%. `barh` render bar dài 0 → không phân biệt được với một series
  bị bỏ. Nó đang giấu một phát hiện thật: **domain duy nhất model đúng 100% về ngữ nghĩa lại là
  domain nó không bao giờ phát đúng format `ANSWER: <value>` yêu cầu.**
- **Đường gate 75% vẽ ngang qua một series mà nó không quản.** Theo run manifest, gate là
  `minimum_overall_semantic_accuracy: 0.9` và `minimum_domain_semantic_accuracy: 0.75` — **chỉ áp cho
  semantic accuracy**. `strict_valid_rate` không bị gate. Một `axvline(75)` trải qua cả bar xanh lẫn
  bar vàng ngụ ý sai rằng strict format đang bị test ở 75%. Đây là lỗi data-logic, không phải style.
- **Gate 90% overall nêu trong subtitle nhưng không bao giờ vẽ, và giá trị overall (57.0%) không bao
  giờ xuất hiện.** (1.0 + 0.984375 + 0.125 + 0.171875)/4 = 0.5703125.
- **Heatmap đang mã hoá nhiễu thành trường màu.** 16 ô, n=16 mỗi ô, `vmin=0, vmax=1` trong khi dữ
  liệu trải 0.500–0.625 — **12.5% của thang màu**. Nhưng tệ hơn cả việc bị nén: biến thiên hoàn toàn
  là ±1 item ở n=4. `rule_recall` đạt 4/4 ở **16/16 ô**; `stage_payoff` 4/4 ở 15/16;
  `state_update` và `terminal_scoring` **luôn chỉ 0/4 hoặc 1/4**. Ô 50% nghĩa là cả hai domain khó
  đều 0; ô 62.5% nghĩa là cả hai đều 1. **Toàn bộ dải động của heatmap là hai lần tung đồng xu.** Nó
  còn hiển thị sai đại lượng: gate được đánh giá *theo domain trong từng cell* (n=4), nên accuracy
  tổng ở mức cell không phải thứ bị fail.
- **Artifact trang trí:** `_blossom()` (dòng 657) đóng một glyph 4 vòng tròn tại figure coord
  (0.965, 0.965) mỗi lần save — trong PNG nó nằm góc trên phải panel phải và đọc như một swatch
  legend lạc. Bỏ trước khi nộp.
- **Trùng lặp:** main.tex dòng 136 đã có `paper/figures/game_understanding_accuracy.pdf`, caption
  "Semantic accuracy by audit domain" — cùng dạng chart, cùng luận điểm construct-validity, trên
  audit anh em.

### Spec

**Giết hẳn panel phải.** Một heatmap 16 ô mà toàn bộ dải là hai item ở n=4 thì không báo cáo được ở
bất kỳ thang màu nào. Thay bằng đúng một câu: *"0 of 16 context × mapping cells passed; every cell
failed on state update (0–1 of 4) and terminal scoring (0–1 of 4)."*

**Giữ panel trái**, đổi thành dot-and-interval một cột, **chuyển cả hình xuống phụ lục** (hoặc gộp
làm facet thứ hai của `fig:understanding-audit` — cùng dạng, cùng claim, dataset anh em).
4 dòng domain × 2 metric, **Wilson 95% CI ở n=64 trên mọi điểm** (hiện đang không có). **Số 0 phải
được ghi nhãn, không vẽ thành bar vô hình**: Rule Recall strict = 0% dùng marker rỗng trên trục kèm
chữ "0%". **Chỉ series semantic mới có đường gate 75%** — tách facet cho strict format. Thêm dòng
overall: semantic **57.0%** (n=256) đối chiếu gate **90%**, strict 33.6%. Bỏ `_blossom()`.

| domain (n=64) | semantic | strict format |
|---|---|---|
| Rule recall | 100.0% | **0.0%** |
| Stage payoff | 98.4% | 34.4% |
| State update | **12.5%** | 7.8% |
| Terminal scoring | **17.2%** | 92.2% |
| *overall (n=256)* | *57.0%* | *33.6%* |

⚠ **Hai audit khác nhau, số gần giống nhau — không được lẫn.** Con số "59.1% semantic / 32.1% strict"
trong main.tex thuộc audit **game-understanding**. Audit context-skin sau hình 08 là **57.0% / 33.6%**.
Hai giá trị 12.5% và 17.2% mà mục "Payoff-preserving context robustness pilot" trích là của
context-skin và khớp `comprehension_by_domain.csv` chính xác.

### Caption

> Comprehension audit of the context-pilot checkpoint (Qwen2.5 7B Instruct F16; 256 probe items, 64
> per domain): public rule recall and one-stage payoff lookup were near-perfect (100% and 98.4%)
> while state update and terminal scoring reached only 12.5% and 17.2%. The frozen gate requires at
> least 90% overall and 75% per-domain semantic accuracy in every context-by-mapping cell; none of
> the sixteen cells passed, so the context pilot remains diagnostic only.

---

## 8. Bảng chốt

| # | Hình cũ | Quyết định | Dạng chart mới | Vị trí |
|---|---|---|---|---|
| 01 | human_comparison_scorecard | **Giết** — gộp vào 02 | — | — |
| 02 | clustered_logit_forest_full_spec | **Gộp + tỉa 10/16 dòng** | Paired forest 6 dòng, 1 thang log-odds | Mục pilot mới sau dòng 222 |
| — | *(mới)* | **Bảng E5–E8** | Bảng | Cùng mục |
| 03 | surface_variant_unsafe_rate_forest | **Vẽ lại 2 panel** | Dot-interval ×2, sắp theo panel A | Mục pilot mới cạnh "Calculator-aided" |
| — | surface_first_round_direction_stacked | **Thêm companion** | Diverging stacked bar | Cùng mục / phụ lục |
| 04 | x7_aggression_forest_plot | **Vẽ lại, bỏ dòng N=3** | Slopegraph ghép cặp, facet risk | Mục "Cross-provider matchup pilot" |
| 05 | x10a_n3_rank_response | **Thay hẳn bằng 2-player** | Dot-interval 3 panel + inset paired Δ | Cùng mục |
| 06 | fig01_baseline_risk_response | **Swap + vẽ lại** | Dot-interval rời rạc (bỏ band) + forest contrast | Thay dòng 160 |
| 07 | fig13_repeat_run_stability | **Giữ, vẽ lại panel B** ⚠ chặn bởi B2 | Panel B → dot-plot 5 seed + đường chance | "Reproducibility" / "Limitations" |
| 08 | comprehension_admission | **Giết panel phải, hạ phụ lục** | Dot-interval + Wilson CI, tách gate | Phụ lục |

**Nguyên tắc chung áp cho cả 6 hình:**
- Không bar cho tỉ lệ có CI → dot-and-interval.
- Không band liên tục trên nhân tố rời rạc → error bar rời rạc.
- Không đỏ/xanh-lá cho verdict; không mã hoá p<.05 bằng màu khi CI đã cắt 0.
- Không quá 3–4 series phân loại; `family` 12 mức → `interpretation` 3 mức.
- Màu theo thực thể, cố định xuyên suốt paper (Luna / Haiku / Gemini giữ nguyên hue ở mọi hình).
- Ghép cặp trong thiết kế phải hiện ra trong hình (slopegraph / paired Δ), không được thay bằng CI
  marginal.
- CI suy biến hoặc bị clamp phải được đánh dấu, không vẽ như ước lượng chặt.
- Mọi caption ghi n, số CRN block, `run_phase`, và cơ chế (`ai-race-fairgame-v3` vs
  `ai-race-nplayer-v1`). Tối đa 2 câu.
