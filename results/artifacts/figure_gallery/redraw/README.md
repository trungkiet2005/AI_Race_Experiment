# Hình đã vẽ lại — bàn giao

Thực thi [`../FIGURE_REDRAW_PLAN.md`](../FIGURE_REDRAW_PLAN.md). **8 hình cũ → 6 hình + 1 companion
+ 1 bảng.**

```
prepare_data.py   đọc run artifact gốc → tables/*.csv   (14 bảng nguồn)
draw_figures.py   đọc tables/*.csv     → figures/*.pdf + *.png
verify.py         đọc lại artifact gốc → assert khớp    (89/89 pass)
```

Chạy lại toàn bộ:

```bash
cd results/artifacts/figure_gallery/redraw
python prepare_data.py && python draw_figures.py && python verify.py
```

`draw_figures.py` **không** chạm vào thư mục run nào — nó chỉ đọc `tables/`. Nên mọi mark trong PDF
đều truy được về một CSV reviewer mở được, và đó cũng chính là "table view" mà quy tắc contrast của
palette đòi hỏi.

---

## Bảng màu

Dùng thứ tự categorical của design system, **đã chạy validator** chứ không chọn bằng mắt
(`scripts/validate_palette.js` của skill `dataviz`):

| Bộ | Dùng ở | CVD separation | Normal-vision floor |
|---|---|---|---|
| slot 1–5, adjacent | hình E, D | PASS, ΔE 9.1 (protan) | PASS, ΔE 19.6 |
| slot 1–3, **all-pairs** | hình C, B | PASS, ΔE 9.2 | PASS, ΔE 24.0 |
| 2 series | hình A | PASS, ΔE 24.7 | PASS, ΔE 33.6 |

Ba slot dưới 3:1 contrast trên nền trắng → áp **relief rule**: mọi hình đều có bảng nguồn trong
`tables/` và có direct label, nên identity không bao giờ chỉ nằm ở màu. Mọi hình nhiều series đều
đổi cả **shape marker**, nên đọc được khi in đen trắng.

Màu bám theo **thực thể**: một checkpoint giữ nguyên hue ở mọi hình nó xuất hiện.

---

## Sáu hình + companion + bảng

### `figA_human_reference` — thay hình 01 + 02

Forest ghép cặp, **một thang log-odds**, 6 dòng. Hình 01 cũ dùng một dạng chart (bar từ gốc 0) cho
**năm loại test** khác nhau trên 8 trục y — không có phép biến đổi nào đưa cả 8 về một thang trung
thực, nên 4 effect còn lại thành bảng.

Sửa được ba thứ hình cũ làm sai:
- **Interval giờ là CI 95%** (verify.py assert cả hai chiều: khớp `ci_95_low/high`, **và** rộng hơn
  1 SE). Hình 01 cũ vẽ ±1 SE trong khi hình 02 vẽ CI 95% cho *đúng bốn hệ số đó*.
- Bỏ mã hoá đỏ/xám theo `p<.05` (CI cắt 0 đã mang thông tin đó). Thay bằng **fill của marker = độ
  ổn định dấu qua jackknife leave-one-CRN-block-out**. Đúng một dòng rỗng: `opponent_prev_unsafe`.
- Dòng `first_round_unsafe` có cận trên 7.72 vượt trục → vẽ tới mép, mũi tên, ghi rõ giá trị. Không
  cắt câm.

> **Caption.** Cluster-robust logistic coefficients (log-odds of Unsafe) for the four dynamic
> predictors of Fernández Domingos and Han's Table 1 model 6 and the two risk-treatment contrasts,
> from a single-checkpoint pilot (Qwen2.5 7B Instruct F16; 210 races, 3 486 round-level decisions,
> 10 common-random-number blocks, zero parse failures, `run_phase = pilot`), beside the published
> human estimates (2 888 participant-rounds, 172 pair clusters). Intervals are 95% cluster-robust on
> the repetition block and are anti-conservative at ten clusters; no coefficient here is a
> confirmatory estimate.

**Bảng đi kèm** (`tables/figA_side_table.csv`) cho E5–E8. Hai điều bảng phải nói mà hình cũ giấu:
E5/E6 là **trung bình của 6 Cohen's d theo persona mà dấu không thống nhất** (E5: −0.135→+0.330;
E6: −0.465→+0.433); và **E8 không có giá trị human** (`human_value` rỗng upstream) với mẫu số 350
chứ không phải 420.

### `figB_surface_wording` — thay hình 03

2 panel dot-and-interval, **cùng thứ tự dòng, sắp theo panel A**.

- Panel A = contrast **có kiểm soát** (cùng state, cùng seed, n=60 ghép cặp). Panel B = trajectory
  rate, **bị nhiễu bởi state nội sinh** — subtitle nói thẳng.
- **`family` 12 mức → `interpretation` 3 mức.** 8/12 family là singleton; legend 12 màu là trang trí
  và trượt CVD.
- Canonical được neo bằng đường liền + band CI của chính nó, marker rỗng, nhãn dưới chân trục.
- Trục phụ trên panel B ghi Δ pp — đổi nhãn tuyến tính của *cùng* thang, không phải dual-axis.

Phát hiện: **cả 15 biến thể meaning-preserving lật ≤ 15% quyết định vòng 1**; ngoại lệ duy nhất là
`emotional_importance` (50/60) — biến thể được thiết kế để *không* bảo toàn nghĩa.

> **Caption.** Under matched game state and matched sampling seeds, all 15 meaning-preserving
> surface rewrites of the frozen v3 prompt flip at most 9 of 60 paired first-round decisions; the
> sole exception is `emotional_importance` (50/60), a deliberately non-meaning-preserving framing
> cue. Whole-trajectory Unsafe rates (right) diverge far more widely (0.084–0.892 against canonical
> 0.522, n=558 per variant) because the states themselves differ after round 1 — reversing the
> action order moves the trajectory by −43.7 pp yet flips only 2 of 60 round-1 decisions
> (Qwen2.5-7B-Instruct, `run_phase="pilot"`, 30 races and 10 repetition blocks per variant, 0 parse
> failures).

### `figB2_surface_flip_direction` — companion (phụ lục)

Diverging stacked bar cho **dấu** của các lần lật — thứ flip rate không thể mang. 50 flip của
`emotional_importance` là **48 Safe→Unsafe / 2 Unsafe→Safe**: cue cảm xúc làm model liều hơn, một
claim có hướng. Trục chạy tới trần 60 cả hai phía, không tới max dữ liệu.

*Ba companion khác bị loại:* `surface_family_boxplot` (8/12 hộp vẽ từ 1 điểm), `..._by_risk_heatmap`
(54 ô đều in số → màu thừa; diverging map cho đại lượng độ lớn), `..._pilot_vs_smoke` (hai run dùng
**estimator khác nhau** — binomial vs cluster-bootstrap — và dumbbell bỏ cả hai interval).

### `figC_opponent_contingency` — thay hình 04

**Slopegraph ghép cặp, facet theo risk**, cộng panel paired Δ.

- **Bỏ toàn bộ 3 dòng N=3.** verify.py assert mọi decision được vẽ đều là `ai-race-fairgame-v3`.
  Hình cũ xếp hạng chung trục hai cơ chế khác nhau.
- **Không pool risk** — pooling làm đổi 4/6 thứ hạng.
- Thiết kế vốn ghép cặp (3 matchup dùng chung 10 game seed — verify.py assert). Dùng paired Δ thì
  **cả ba đều loại trừ 0**: +0.167, +0.175, +0.074. Forest sắp xếp cũ vứt đi đúng phần này rồi mời
  người đọc so bằng mắt các interval chồng nhau.
- Bỏ `axvline(0.5)` (không phải Nash, không phải ngưỡng), bỏ padding trục tới 1.12, bỏ nhãn đặt theo
  độ rộng CI.

> **Caption.** Unsafe choice rate for each model against each of its two opponents, shown separately
> at each maximum-private-risk level, with the paired within-model difference at right; n = 10
> common-random-number-matched repetitions per cell (258 decisions per matchup), two-player
> mechanism only (`prompt_version = ai-race-fairgame-v3`). All three models are more Unsafe against
> the non-Haiku opponent (paired Δ +0.167, +0.175, +0.074; all 95% t-intervals exclude zero), so
> aggression is opponent-contingent rather than a per-model constant; pilot data (`run_phase =
> "pilot"`, no `run_manifest.json`), not confirmatory evidence.

### `figD_race_position` — thay hình 05

**Đổi nguồn, không chỉ đổi chart.** Hình N=3 cũ có bar "Ahead of both" đếm cả hoà (chỉ 100/522
quyết định thật sự dẫn đầu). Bản thay dùng đúng cơ chế 2 người của paper, **5 checkpoint đều có
`run_manifest.json` completed** — verify.py assert từng cái, và assert `results/cross_provider/`
thì không có cái nào.

Facet 5 panel nhỏ thay vì chồng: 3 series gần trùng đường, direct label đè lên nhau. Ahead/Behind
**ghép cặp chính xác** (n bằng nhau ở cả 5 model — verify.py assert). 5 ô tất định được đánh dấu
bằng marker rỗng, không vẽ như ước lượng chặt.

Cho thấy dị biệt thật: 3 Gemini đơn điệu mạnh (Ahead 0% → Behind 100%), GPT-5 nano yếu,
**GPT-5.4 nano phẳng/ngược** (0.621 vs 0.579). Nên claim là **checkpoint-scoped**.

> **Caption.** Unsafe choice rate by race position (own progress ahead of, level with, or behind the
> opponent before the decision) for five baseline checkpoints; 10 common-random-number repetition
> blocks per cell, ahead and behind exactly paired by construction, 95% percentile bootstrap
> intervals. Open markers are deterministic cells where all ten blocks agreed, so the interval is
> degenerate rather than precise; positions are post-action and endogenous, so these associations
> are descriptive pilot evidence, not causal estimates, and GPT-5.4 nano shows no positional effect
> at all.

### `figE_baseline_risk_response` — swap hình đang dùng ở `main.tex:160`

- **Bỏ `fill_between`.** Band 95% liên tục quét qua các mức risk chưa từng chạy. Thay bằng error bar
  rời rạc tại đúng 3 điểm đã lấy mẫu. Trục x tuyến tính thật (0.1/0.6/0.9 đúng khoảng cách).
- **Hai CI suy biến được đánh dấu** (20/20 player-race Unsafe → mọi resample = 1.0). Vẽ ra thành
  band bề rộng 0 thì nó đọc như ước lượng chặt nhất hình.
- **Bỏ p-value khỏi hình.** `holm_p_high_vs_low` upstream hiệu chỉnh trên **họ 7 model**, không phải
  5 model được vẽ — verify.py assert bằng số học. Ai tính lại Holm từ 5 dòng sẽ ra số khác.
- Giữ cả 2 panel: chúng bổ sung chứ không trùng. GPT-5.4 nano có marginal chồng nhau nặng nhưng
  contrast ghép cặp là −1.0 pp [−6.5, +3.8] — **hẹp hơn cả hai marginal**. Đó là toàn bộ phần
  thưởng của thiết kế CRN và mắt thường không đọc ra từ panel A.

> **Caption.** Player-level Unsafe rate by maximum private setback risk for five checkpoints (A) and
> the paired within-repetition-block 90%-minus-10% contrast (B); each cell is ten races and twenty
> player trajectories, with 95% percentile bootstrap intervals clustered on the repetition block.
> Two Gemini cells are saturated at 100% so their intervals are degenerate rather than precise; all
> runs are exploratory pilots on non-protocol-matched provider routes and are juxtaposed
> descriptively.

### `figF_repeat_run_stability` — giữ, vẽ lại panel B

**Chặn B2 đã gỡ.** Thư mục nguồn bị xoá ở commit `b07ae73` và
`analyze_two_player_paper_figures.py` vẫn hard-code đường dẫn chết đó → chạy lại sẽ crash.
`prepare_data.py` đọc blob từ `git show a81a8c8:...`. **Vẫn nên khôi phục thư mục** để analyser gốc
chạy được.

- Panel B cũ là 3 bar không có uncertainty nào, chú "n=74". Đơn vị lấy mẫu thật là **5 race seed**
  (verify.py assert), và cùng 5 seed đó lặp ở cả 3 mức risk. Giờ vẽ 5 chấm per-seed + crossbar
  pooled: *cho thấy* cluster thay vì khẳng định một interval mà bootstrap 5 block không ổn định nổi.
- **Bar 100% là artifact** — cả hai run đều toàn Unsafe ở risk 0.1, agreement bị *ép* bằng 100%
  (verify.py assert). Đánh dấu trực tiếp trên hình.
- Thêm **đường chance-agreement**: 65% đọc như "khá tái lập được", nhưng chance là 59.4%/51.3% →
  κ = 0.14 / 0.28. Đây mới là phát hiện, và hình cũ chôn nó.

> **Caption.** The same Gemini 3 Flash protocol rerun without a fixed decoding seed reproduces the
> aggregate risk response (A: 100.0/71.3/57.7% versus 100.0/72.3/53.9%) but not the individual
> decisions (B: matched game-round-seat agreement of 74/74, 48/74 and 48/74 over five shared race
> seeds, against chance agreement of 100%, 59% and 51%). Agreement at risk 0.1 is forced because
> both pilots were uniformly Unsafe there, so repeated unseeded API runs are reported as separate
> pilots, never as interchangeable replicates.

### `figG_comprehension_audit` — hình 08, giết panel phải, hạ phụ lục

- **Giết heatmap.** 16 ô, n=16/ô, `vmin=0/vmax=1` trong khi dữ liệu trải 0.500–0.625 (12.5% thang
  màu) — và toàn bộ dải động là **hai lần tung đồng xu** ở n=4. Thay bằng một câu: *0 trong 16 cell
  đạt gate; mọi cell trượt ở state update (0–1 / 4) và terminal scoring (0–1 / 4).*
- **Tách facet theo metric.** Gate chỉ áp cho semantic accuracy (≥90% overall, ≥75% per-domain);
  `strict_valid_rate` **không** bị gate. Một `axvline(75)` quét qua cả hai series là lỗi data-logic.
- **Wilson 95% CI ở n=64 cho mọi điểm** (upstream không có).
- **Rule recall strict = 0% là số 0 THẬT**, ghi nhãn chứ không vẽ bar vô hình. Nó giấu một phát hiện:
  domain duy nhất đúng 100% ngữ nghĩa lại là domain không bao giờ phát đúng format.
- Bỏ `_blossom()` — glyph 4 vòng tròn đóng ở góc mỗi lần save, đọc như swatch legend lạc.

> **Caption.** Comprehension audit of the context-pilot checkpoint (Qwen2.5 7B Instruct F16; 256
> probe items, 64 per domain, 95% Wilson intervals): rule recall and one-stage payoff lookup were
> near-perfect (100% and 98.4%) while state update and terminal scoring reached only 12.5% and
> 17.2%. The frozen gate applies to semantic accuracy alone (≥90% overall, ≥75% per domain) and none
> of the sixteen context-by-mapping cells passed, so the context pilot remains diagnostic only.

---

## Đưa vào `main.tex`

| Hình | Vị trí | Loại |
|---|---|---|
| `figA_human_reference` + bảng E5–E8 | `\subsection{Human-reference comparison pilot}` **mới**, sau `\FloatBarrier` dòng 222, **trước** dòng 223 | Thêm |
| `figB_surface_wording` | Mục pilot mới, cạnh "Calculator-aided behavioural pilot" | Thêm |
| `figB2_surface_flip_direction` | Phụ lục | Thêm |
| `figC_opponent_contingency` | `\subsection{Cross-provider matchup pilot}` mới | Thêm |
| `figD_race_position` | Cùng mục trên, hoặc "Cross-checkpoint baseline replication" | Thêm |
| `figE_baseline_risk_response` | **Thay** `results/impact_upgrade/figures/cross_model_risk_response.pdf` (dòng 160) | Swap |
| `figF_repeat_run_stability` | "Reproducibility and data provenance" hoặc Limitations | Thêm |
| `figG_comprehension_audit` | Phụ lục | Thêm |

**Tuyệt đối không** đặt figA/figC/figD vào ba mục `\pending` (dòng 225/229/233). Chúng trả lời *đúng*
ba câu hỏi đó, nên thả vào là biến pilot thành confirmatory qua cửa sau và phá cam kết editorial
paper đã giữ 5 lần.

---

## Còn tồn — cần bạn quyết

1. **`results/cross_provider/` không có `run_manifest.json` nào** (verify.py assert điều này như một
   dữ kiện). Trượt luật admit của `CLAUDE.md`. Ảnh hưởng **hình C**. Hình D đã được chuyển sang
   nguồn có manifest; hình C thì không có nguồn thay thế vì không corpus nào khác cho hai provider
   khác nhau đối đầu. Hoặc rào rõ trong text, hoặc chạy lại có manifest.
2. **Khôi phục `ai_race/results/_api_5games_allrisk/`.** Blob còn trong git nên hình F vẽ được, nhưng
   `analyze_two_player_paper_figures.py` vẫn crash và `dataset_inventory.csv:94` vẫn liệt kê nó như
   artifact sống.
3. **Sửa `results/cross_provider/ANALYSIS_REPORT.md` §7** — câu "Claude Haiku 4.5 luôn là model ít
   hung hăng nhất trong mọi bối cảnh" bị chính CSV của nó bác: trong *Gemini vs Haiku*, Haiku hung
   hăng hơn ở cả 3 mức risk (0.554/0.540/0.411 vs 0.494/0.456/0.401).
4. **Sửa `SELECTED_FOR_PAPER.md:111-113`** — canonical **không** "nằm gần giữa phân phối", nó đứng
   thứ 4 từ dưới lên trong 18 (verify.py assert). Cách đọc đúng có lợi hơn cho paper.
5. **G = 10 ở mọi nơi.** Mọi CI trong bộ hình đứng trên 10 CRN cluster (hình F: 5). Caption đã ghi.
   Reviewer sẽ hỏi wild cluster bootstrap.
6. **Chưa có CI cho Δ trong hình B.** Interval hiện tại là marginal, seed độc lập
   (`seed_label=f"unsafe:{variant}"`), trừ chúng cho nhau sai hai lần. Sửa đúng là ~10 dòng trong
   `analyze_surface_sensitivity.py`: bootstrap Δ ghép cặp trên các `rep` chung với một common
   resample mỗi draw. Nếu chạy, đổi trục chính panel B sang Δ với CI ghép cặp thật.
