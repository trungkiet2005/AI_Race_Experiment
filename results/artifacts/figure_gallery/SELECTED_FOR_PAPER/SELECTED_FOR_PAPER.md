# Chọn hình cho paper AAMAS — lý do & insight

**8 hình đã chọn nằm sẵn trong [`SELECTED_FOR_PAPER/`](SELECTED_FOR_PAPER/)** (đánh số 01–08 khớp
bảng ở mục 4), đã đổi tên rõ ràng và copy ra khỏi các subfolder gốc — không cần lục lại
`NEW_01_persona/`, `NEW_02_surface/`... nữa. Hình 06 và 07 có kèm bản `.pdf`/`.svg` để nhúng thẳng
vào `main.tex` (bản `.tiff` 600dpi gốc không copy sang vì quá nặng — cần thì lấy lại từ
`NEW_05_two_player_paper_analysis/`). Các đường dẫn trong phần phân tích bên dưới vẫn trỏ về vị trí
gốc (để giữ ngữ cảnh mục/nguồn), nhưng khi thao tác thực tế hãy dùng file trong `SELECTED_FOR_PAPER/`.

**Mục đích của file này:** từ 120 hình trong `results/artifacts/figure_gallery/` (99 hình đã có caption trong
[`INDEX.md`](INDEX.md) + 21 hình rời `cross_provider` chưa được index), chọn ra một bộ hình có
tiềm năng thật sự nâng chất lượng bản thảo `paper/main.tex`, kèm insight vì sao từng hình giúp
paper "dễ được accept" hơn — theo đúng khung đánh giá mà một reviewer AAMAS sẽ dùng: tính chặt chẽ
của claim, có preempt được câu hỏi hiển nhiên của reviewer không, và có vi phạm nguyên tắc bằng
chứng của chính dự án không (`CLAUDE.md`, `paper/README.md`).

**Đã đọc trước khi chọn:** toàn bộ `INDEX.md`, `results/cross_provider/ANALYSIS_REPORT.md`, và
`paper/main.tex` (structure, 6 hình đang dùng, và đặc biệt 3 mục kết quả đang `\pending`). Phần
lớn giá trị của file này nằm ở việc đối chiếu hình ứng viên với **đúng những gì `main.tex` đã cam
kết** — không phải chỉ "hình nào đẹp".

---

## 0. Hai phát hiện nền tảng, chi phối mọi lựa chọn dưới đây

### (a) 3/6 hình đang dùng trong `main.tex` đã có bản sao y hệt trong `all_figure/`

| Đang dùng trong `main.tex` | Bản sao trong `all_figure/` |
|---|---|
| `results/open_source/context_skin_pilot/analysis_temperature_robustness/figures/context_effect_temperature_stability.pdf` | `ADMITTED_context_skin_pilot/analysis_temperature_robustness__figures__context_effect_temperature_stability.png` |
| `results/open_source/activation_sae/.../fixed_state_target_minus_controls.pdf` | `ADMITTED_activation_sae/causal_selfplay__...__fixed_state_target_minus_controls.png` |
| `results/open_source/egt_reproduction/egt_theory_vs_llm_unsafe.pdf` | `ADMITTED_egt_reproduction/egt_theory_vs_llm_unsafe.png` |

→ Đừng đề xuất lại 3 hình này như "hình mới" — chúng đã ở trong bài. Chỉ nêu ở đây để bạn không mất công tìm lại.

### (b) `main.tex` có đúng 3 mục kết quả đang `\pending`, và **không nên lấp bằng hình pilot**

```
\subsection{Maximum-risk treatment effects}           → \pending
\subsection{Opponent response, race position, and first-round behaviour} → \pending
\subsection{Model and condition heterogeneity}        → \pending
```

Ngay bên dưới, `main.tex` viết rõ: *"Confirmatory claims about risk treatment, opponent response,
race position, or model-family differences remain pending until protocol-matched preregistered
runs pass comprehension and design gates."* Đây là một cam kết editorial mạnh, và **đúng là kiểu
cam kết mà reviewer làm behavioral game theory rất thích** — nó nói "chúng tôi biết phân biệt
pilot với confirmatory, và sẽ không tự lừa mình". Các hình mạnh nhất trong kho (persona
risk-response, opponent-reciprocity, cross-provider risk-response) đều **trả lời đúng 3 câu hỏi
này** — tức là dùng chúng để "lấp" 3 mục trên sẽ phá vỡ chính cam kết đó và biến pilot thành
confirmatory qua cửa sau.

**Nước đi đúng, và cũng là nước đi paper đã dùng 5 lần rồi:** thêm một **mục pilot mới, độc lập**,
đặt cạnh "Calculator-aided behavioural pilot" / "Cross-checkpoint baseline replication" / "Payoff-preserving
context robustness pilot" — cùng văn phong hedge ("diagnostic", "not a confirmatory estimate", số
liệu có CI, ghi rõ n nhỏ). Đây là khung mà toàn bộ gợi ý bên dưới tuân theo.

---

## 1. TOP PICK — nên thêm vào thân bài chính (mỗi hình 1 mục pilot mới)

### 1.1 Persona-sensitivity & human-replication pilot *(mục mới)*

**Hình chính:** [`NEW_01_persona/human_comparison_scorecard.png`](NEW_01_persona/human_comparison_scorecard.png)

**Insight — vì sao giúp accept:** Phần Introduction của `main.tex` đặt ra đúng 3 câu hỏi confirmatory
(risk treatment, opponent's preceding action, race position) và trích dẫn trực tiếp Fernández
Domingos & Han làm nguồn tham chiếu hành vi người. Hình này là **8 phép so sánh preregistration-target
person-vs-LLM song song**, đúng những effect mà Introduction đã nêu tên (E1 opponent_prev_unsafe,
E2 progress_gap, E3 first_round_unsafe, E5/E6 risk contrasts...). Cho reviewer thấy: nhóm tác giả
đã pilot **chính xác** các câu hỏi họ hứa sẽ confirm — đây là bằng chứng mạnh nhất có thể có rằng
protocol confirmatory là khả thi và đã được instrument đúng, mà không cần phá vỡ ranh giới
pilot/confirmatory. Bốn trong tám effect "not_replicated" — đừng ngại con số này, nó *tăng* độ tin
cậy vì cho thấy không có cherry-pick.

**Lưu ý khi dùng:** panel E8 (`share_AS`) chỉ có cột LLM, không có cột human — kiểm tra lại
`human_comparison.csv` trước khi trích số, có thể cần crop hoặc chú thích riêng panel này.

**Hình phụ nên đi kèm (appendix hoặc cùng mục, tuỳ dung lượng):**
[`NEW_01_persona/clustered_logit_forest_full_spec.png`](NEW_01_persona/clustered_logit_forest_full_spec.png)
— version thống kê chặt hơn của cùng câu chuyện (cluster-robust logit, 6 specification), cho
reviewer thấy `own_prev_unsafe`, `opponent_prev_unsafe` tương tác với `progress_gap_before` chứ
không đứng độc lập — đúng độ phức tạp mà một mô hình dynamic thật sự cần.

**Caption gợi ý (theo đúng văn phong hedge của `main.tex`):**
> *Persona-sensitivity pilot (n=210 races), juxtaposed against the eight preregistration-target
> effects reported by Fernández Domingos and Han. Green = same sign as the human study; red = not
> replicated. This pilot instruments the confirmatory questions posed in §1 but is not a
> confirmatory estimate: risk-treatment and pairing coverage are limited to one checkpoint and ten
> races per cell.*

---

### 1.2 Surface-wording sensitivity pilot *(mục mới)*

**Hình chính:** [`NEW_02_surface/surface_variant_unsafe_rate_forest.png`](NEW_02_surface/surface_variant_unsafe_rate_forest.png)

**Insight — vì sao giúp accept:** Đây là hình **preempt trước** câu phản biện phổ biến nhất mà bất
kỳ reviewer nào quen với LLM behavioral papers sẽ viết: *"kết quả của bạn có thể chỉ là do cách
diễn đạt prompt, không phải một hiện tượng ổn định."* `main.tex` §2.2 ("Prior work and scope") đã
trích dẫn đúng khung lý thuyết cho việc này — Robinson & Burden (context variability) và Mousavi
Davoudi et al. (invariance dưới payoff-preserving narrative change) — nhưng hiện tại **không có
hình nào minh hoạ khung đó cho chính wording của game** (context-robustness pilot hiện có chỉ đổi
narrative skin, không đổi cấu trúc câu lệnh/format/thứ tự). Forest plot 18 biến thể (Unsafe rate từ
9% đến 89%!) là bằng chứng định lượng trực tiếp cho đúng câu trích dẫn đó, và biến `canonical`
(biến thức tế paper dùng) nằm gần giữa phân phối — một điểm neo tốt cho thảo luận "kết quả của
paper không nằm ở cực đoan của không gian wording".

**Hình phụ cho appendix:** [`NEW_02_surface/surface_variant_pilot_vs_smoke.png`](NEW_02_surface/surface_variant_pilot_vs_smoke.png)
— cho thấy thứ hạng tương đối giữa smoke (n~72) và pilot (n~558) khá ổn định dù magnitude dao
động, một robustness-of-robustness check gọn.

---

### 1.3 Cross-provider matchup pilot *(mục mới — nội dung mới nhất, chưa từng vào paper)*

**Hình chính:** [`x7_aggression_forest_plot.png`](x7_aggression_forest_plot.png)
(nằm ở gốc `all_figure/`, cùng cấp `INDEX.md` — đây là 1 trong 21 hình rời chưa được `gather_all.py`
gom vào subfolder; nguồn gốc thật là `results/cross_provider/figures/x7_aggression_forest_plot.png`,
xem `results/cross_provider/ANALYSIS_REPORT.md` để biết đầy đủ phương pháp)

**Insight — vì sao giúp accept:** Toàn bộ pilot khác trong bài (calculator, context-skin, SAE,
cross-checkpoint) đều là **self-play cùng một checkpoint hoặc sweep checkpoint riêng lẻ** — chưa có
hình nào cho hai checkpoint **khác nhà cung cấp** thật sự chơi đối đầu nhau (GPT-5.6 Luna vs Claude
Haiku 4.5 vs Gemini 3.5 Flash-Lite). Đây là câu hỏi external-validity hiển nhiên tiếp theo mà một
reviewer AAMAS sẽ hỏi: *"liệu hiệu ứng có chỉ xuất hiện trong self-play?"* — và bộ dữ liệu này (dù
là pilot, n=10 race/ô, `run_phase="pilot"`, không có `run_manifest.json` theo đúng
`results/cross_provider/ANALYSIS_REPORT.md`) là câu trả lời sơ bộ trực tiếp. `x7` cho thấy độ hung
hăng **phụ thuộc mạnh vào đối thủ cụ thể** chứ không phải hằng số của model (Luna: 46% vs Haiku,
63% vs Gemini) — một phát hiện tinh tế, không tầm thường, đúng kiểu insight mà reviewer thích vì nó
không "quá gọn để tin".

**Hình đi kèm rất mạnh cho cùng mục:**
[`x10a_n3_rank_response.png`](x10a_n3_rank_response.png) (cũng ở gốc `all_figure/`) —
"đang dẫn đầu" giảm unsafe rate 30–45 điểm phần trăm so với không dẫn đầu, ở **mọi** mức risk. Đây
là effect size lớn nhất trong toàn bộ kho hình, và nó **operationalize chính xác chủ đề "falling
behind" mà tiêu đề paper gợi mở** (falling-behind-ai-race.md là tên file nguồn nhân bản). *Lưu ý:*
hình này dùng dữ liệu trận 3 người (`ai-race-nplayer-v1`), khác cơ chế với game 2 người
(`ai-race-fairgame-v3`) mà `main.tex` định nghĩa — nếu dùng, phải nói rõ trong 1 câu rằng đây là
minh hoạ từ một cơ chế N=3 riêng biệt, không phải cùng game 2-player (xem mục 3 bên dưới, "N-player
= không cùng cơ chế").

**Caption gợi ý:**
> *Pairwise cross-provider matchups (GPT-5.6 Luna, Claude Haiku 4.5, Gemini 3.5 Flash-Lite), 10
> races per matchup–risk cell, `run_phase="pilot"`, no confirmatory manifest. Aggression rank
> depends on the specific opponent, not a fixed per-model constant — a pattern the single-checkpoint
> baseline pilot cannot show because it never pairs two different providers.*

---

### 1.4 Nâng cấp hình "Cross-checkpoint baseline replication" đang dùng (swap, không phải thêm)

**Hình hiện tại:** `results/impact_upgrade/figures/cross_model_risk_response.pdf`
**Đề xuất thay bằng:** [`NEW_05_two_player_paper_analysis/fig01_baseline_risk_response.png`](NEW_05_two_player_paper_analysis/fig01_baseline_risk_response.png)
(có sẵn `.pdf`/`.svg`/600dpi `.tiff` — đúng chuẩn nộp bài AAMAS)

**Vì sao:** đối chiếu số liệu, đây là **cùng một tập 5-checkpoint baseline** (GPT-5 nano 12/15/15%,
GPT-5.4 nano 58/50/57%, ba Gemini giảm dần theo risk) mà §"Cross-checkpoint baseline replication"
đã report bằng số — chỉ khác là bản mới có thêm panel B (paired repetition-block contrast, 90%
trừ 10% risk, với CI riêng từng checkpoint) mà bản đang dùng không có. Đây không phải nội dung mới
cần thêm câu chữ mới, chỉ là **hình tốt hơn cho đúng đoạn văn đã viết sẵn** — chi phí sửa bài gần
bằng không, lợi ích visual/publication-format thì rõ.

---

### 1.5 Repeat-run stability — nên thêm vào "Reproducibility and data provenance" hoặc "Limitations"

**Hình:** [`NEW_05_two_player_paper_analysis/fig13_repeat_run_stability.png`](NEW_05_two_player_paper_analysis/fig13_repeat_run_stability.png)

**Insight — vì sao giúp accept:** Đây là hình "tự phản biện" hiếm và có giá trị cao. Panel A cho
thấy Unsafe-rate theo risk gần như **trùng khít** giữa hai lần chạy pilot độc lập, không seed
(5-rep sớm vs 10-rep cuối) — tức thống kê tổng hợp ổn định. Nhưng panel B cho thấy ở cấp **quyết
định khớp từng game/round/seat**, tỉ lệ trùng khớp tuyệt đối chỉ 65–100% — tức từng quyết định
riêng lẻ *không* deterministic dù phân phối tổng thể thì có. Đây trả lời **đồng thời hai câu hỏi
reviewer hay hỏi ngược nhau**: "làm sao biết số liệu pilot không phải nhiễu?" (panel A: không, ổn
định) và "làm sao biết bạn không đang báo cáo một lần chạy may mắn ở cấp quyết định?" (panel B:
đúng, đừng tin từng quyết định, chỉ tin phân phối). Rất hợp với đoạn Limitations hiện tại về
"Repeated calls to one checkpoint do not constitute a population sample".

---

### 1.6 Minh hoạ trực quan cho comprehension-gate failure đã có trong text

**Hình:** [`ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__comprehension_admission.png`](ADMITTED_context_skin_pilot/analysis_live_pilot_t0__figures__comprehension_admission.png)

**Insight:** §"Payoff-preserving context robustness pilot" đã viết bằng số ("state-update accuracy
was 12.5%, terminal-scoring accuracy was 17.2%") nhưng **không có hình minh hoạ** cho chính con số
này — hai hình hiện có (`context_direct_vs_live`, `context_mapping_gate`) chỉ nói về hành vi, không
nói về comprehension gate. Thêm hình này cho reviewer thấy trực quan tại sao claim bị giới hạn ở
"context-conditioned behavior", không phải "understanding" — củng cố đúng câu văn đã có sẵn, zero
rủi ro overclaim vì số liệu đã được report bằng lời.

---

## 2. Phụ lục / supplementary — củng cố độ tin cậy phương pháp, không cần trong thân bài

Nếu venue có supplementary/appendix (kiểm tra template AAMAS 2026 cho phép hay không trước khi
dùng), các hình sau rất đáng đưa vào — chúng không kể một "finding" mới mà kể **"chúng tôi đã kiểm
tra đúng cách"**, thứ mà reviewer thống kê-kỹ-tính rất coi trọng nhưng không cần chiếm chỗ hình
chính:

- [`NEW_01_persona/sample_quality_overview.png`](NEW_01_persona/sample_quality_overview.png) —
  coverage theo cell, lý do loại trừ race, seat-order balance. Trả lời "n mỗi cell là bao nhiêu,
  loại bao nhiêu, seat có lệch không" trong một hình.
- [`NEW_01_persona/logit_jackknife_robustness.png`](NEW_01_persona/logit_jackknife_robustness.png) —
  leave-one-block-out, cho thấy hệ số nào đổi dấu khi bỏ 1 cluster (fragile) vs hệ số nào không.
- [`NEW_01_persona/clustered_logit_across_specifications.png`](NEW_01_persona/clustered_logit_across_specifications.png) —
  6 specification lồng nhau, hệ số nào "sống sót" khi thêm control.
- [`NEW_02_surface/surface_family_boxplot.png`](NEW_02_surface/surface_family_boxplot.png) — gom
  18 biến thể theo họ chỉnh sửa (framing/format/order/...), câu chuyện "họ nào gây flip nhiều nhất".

---

## 3. Không nên dùng / cần cẩn trọng đặc biệt

| Nhóm hình | Vì sao cẩn trọng |
|---|---|
| `n3_*.png`, phần lớn `x10b`, `x11` (trận 3 người, `results/cross_provider/gemini_openai_claude_n3/`) | Dùng cơ chế/prompt **khác** (`ai-race-nplayer-v1`) so với game 2 người mà toàn bộ `main.tex` định nghĩa (`GameConfig` ép đúng 2 người — xem `CLAUDE.md`). Đưa vào paper 2-player sẽ cần cả một đoạn methods giải thích cơ chế N=3 riêng — không đáng, trừ khi bạn chủ động muốn mở một mục "N-player extension" ngắn. Để dành cho companion paper N-player (đã có `results/reports/nplayer/report.md`, `N-Player/theory/`). |
| `ADMITTED_activation_sae/.../surface_n600_..._sae_feature_confirmation.png` | Đây là hình "association" *trước khi* qua causal control — nếu dùng thay cho `fixed_state_target_minus_controls.png` (hình đang dùng, đã qua control và cho null result), bạn sẽ **vô tình đảo ngược đúng luận điểm cẩn trọng mà `main.tex` §"Discussion" đang giữ** ("Predictive AUC or feature–action correlation... does not answer whether that feature uniquely causes a decision"). Nếu thêm, chỉ thêm cạnh hình null-control hiện có, không thay thế. |
| 3 hình đã dùng (xem mục 0a) | Đừng đề xuất lại như hình mới — sẽ đọc như trùng lặp/padding trong mắt reviewer. |
| Bất kỳ hình nào từ nhóm persona/surface/cross-provider dùng để lấp 3 mục `\pending` | Xem mục 0b — vi phạm trực tiếp cam kết editorial hiện có của chính bài. |

---

## 4. Bảng tổng hợp nhanh (checklist khi soạn `main.tex`)

| # | File trong `SELECTED_FOR_PAPER/` | Nguồn gốc | Vị trí đề xuất trong `main.tex` | Loại thay đổi |
|---|---|---|---|---|
| 1 | `01_human_comparison_scorecard.png` | `NEW_01_persona/` | Mục pilot mới, sau "Evolutionary-game reconstruction..." | Thêm mới |
| 2 | `02_clustered_logit_forest_full_spec.png` | `NEW_01_persona/` | Cùng mục trên hoặc appendix | Thêm mới |
| 3 | `03_surface_variant_unsafe_rate_forest.png` | `NEW_02_surface/` | Mục pilot mới, cạnh "Calculator-aided behavioural pilot" | Thêm mới |
| 4 | `04_x7_aggression_forest_plot.png` | root `all_figure/` (← `results/cross_provider/`) | Mục pilot mới "Cross-provider matchup pilot" | Thêm mới |
| 5 | `05_x10a_n3_rank_response.png` | root `all_figure/` (← `results/cross_provider/`), gắn caveat N=3 | Cùng mục trên, kèm 1 câu disclaimer cơ chế | Thêm mới, có điều kiện |
| 6 | `06_fig01_baseline_risk_response.{png,pdf,svg}` | `NEW_05_two_player_paper_analysis/` | Thay `cross_model_risk_response.pdf` | Swap |
| 7 | `07_fig13_repeat_run_stability.{png,pdf,svg}` | `NEW_05_two_player_paper_analysis/` | "Reproducibility and data provenance" hoặc "Limitations" | Thêm mới |
| 8 | `08_comprehension_admission.png` | `ADMITTED_context_skin_pilot/` | "Payoff-preserving context robustness pilot" | Thêm mới, minh hoạ số đã có |

Tổng: **6–8 hình mới/swap** cho thân bài (tuỳ page budget), **~4 hình** cho appendix. Đủ để mở rộng
đáng kể bề rộng bằng chứng (persona, surface-wording, cross-provider) mà không phá cam kết
pilot-vs-confirmatory — đúng điểm mạnh nhất bài này đang có so với các paper LLM-behavioral khác:
kỷ luật bằng chứng, không phải quy mô mẫu.
