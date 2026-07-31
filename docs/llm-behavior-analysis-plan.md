# Plan phân tích hành vi LLM trong AI Race

**Trạng thái:** bản nháp đề xuất, chưa được phê duyệt. Chưa phải preregistration.
**Quan hệ với tài liệu khác:** [PROJECT.md](../PROJECT.md) là protocol nghiên cứu đang có hiệu lực;
tài liệu này đề xuất mở rộng nó theo ba trục. Mọi mâu thuẫn giữa hai file phải được
giải quyết về phía `PROJECT.md` cho tới khi plan này được duyệt và merge vào đó.

**Paper nguồn:** Fernández Domingos & Han (2026), *Falling Behind Drives Unsafe
Development in an Idealised AI Race Experiment* (`arXiv-2607.26034v1/paper.tex`).

---

## §0. Bốn phát hiện từ code quyết định toàn bộ thiết kế

Plan này được viết sau khi đọc engine, runner, recorder, analyser (~3.2k dòng),
classifier và Kaggle manifest. Bốn quan sát dưới đây thay đổi cách thiết kế phải viết.

### (1) Persona đã plumb sẵn end-to-end, và KHÔNG đụng vào prompt hash

- [`ai_race/prompts/ai_race_en.txt`](../ai_race/prompts/ai_race_en.txt) dòng 2 có sẵn
  placeholder `{persona_block}`.
- [`ai_race/engine/prompt.py`](../ai_race/engine/prompt.py) dòng 71–74 render nó thành
  `"\nAdditional role instruction:\n<text>\n"`, và bỏ trống khi persona rỗng.
- [`ai_race/runner/run_experiment.py`](../ai_race/runner/run_experiment.py) dòng 37–44
  đọc `personas[language][seat]` từ agents config.

Nghĩa là **persona là một trục agents-config, không phải một prompt version mới**.
File template không đổi một byte, hash `6180d4f6…` giữ nguyên, invariant trong
`CLAUDE.md` không bị vi phạm. Đây là ý đồ thiết kế có sẵn:
[`companies_default.json`](../ai_race/configs/agents/companies_default.json) cố ý set
`personas: {"en": ["", ""]}` và mô tả mình là "no safety or risk persona is injected".

**Hệ quả cho plan:** không cần bump `promptVersion` cho các arm persona. Bump
`promptVersion` thực ra là *phản tác dụng*, vì nó làm analyser hard-fail ở gate
canonical-prompt (`analyze_ai_race.py:1411-1432`) và ép mọi output xuống nhãn
"sensitivity audit".

### (2) Nhưng persona hiện KHÔNG được ghi vào bất kỳ output nào → nguy cơ pooling ngầm

- `TurnRecord` trong [`state.py`](../ai_race/engine/state.py) không có trường persona.
- `race_row` và `player_rows` trong [`recorder.py`](../ai_race/dataio/recorder.py) không có.
- `CONTEXT` của analyser (`analyze_ai_race.py:104-111`) chỉ gồm
  `model, max_private_risk, prompt_version, protocol_signature, run_phase, run_status`.
- Protocol signature (`analyze_ai_race.py:510-534`) gồm prompt / model / decoding /
  seed_contract / mechanism / runtime — **không có agents**.

Hệ quả: chạy persona và non-persona rồi phân tích chung thì analyser **không phát hiện
được**, gộp im lặng, mọi bảng sai. Đây là bug chờ xảy ra và phải vá **trước** khi chạy
run persona đầu tiên (§4.1).

Thông tin persona hiện chỉ tồn tại gián tiếp trong `game_id`, có dạng
`{game}__{model}__{lang}__{agents}__rep{NNNN}` (`run_experiment.py:82-85`). Phục hồi
bằng cách parse string là fragile và không được dùng làm nguồn chính thức.

### (3) CRN mạnh hơn dự kiến — và trải cả sang trục persona

`game_seed = base_seed + rep` (`run_experiment.py:79-81`), cố ý độc lập với tên
treatment **và cũng độc lập với agents config**. Nên nếu mọi experiment config
(baseline + tất cả persona cell) dùng chung `"seed": 260726`, thì **rep k trong mọi
điều kiện có cùng horizon và cùng setback draw**.

Đây là matched-pairs design gần như miễn phí, mạnh hơn hẳn thí nghiệm người — nơi mỗi
cặp participant chỉ chơi một treatment duy nhất.

- **Điều kiện bắt buộc:** không được đổi `seed` giữa các config.
- **Vấn đề cần vá:** `randomization_block_id = source_run::model::rep`
  (`analyze_ai_race.py:1510-1515`). Persona ở run directory khác → `source_run` khác →
  block bị tách sai. Xem §4.3.

### (4) ΔS không phải biến độc lập — nó là hiệu số hành vi Unsafe, đổi đơn vị

Đây là điểm phân tích sâu nhất trong toàn bộ plan. Trong cơ chế này:

```
progress_i(t) = 1·n_S + 1.5·n_U = (t − n_U) + 1.5·n_U = t + 0.5·n_U

⟹  ΔS(t) = progress_own − progress_opp = 0.5 · (n_U^own − n_U^opp)
```

**ΔS đúng bằng một nửa hiệu số lần chọn Unsafe.** Nó không phải "vị trí trong cuộc đua"
theo nghĩa một biến độc lập — nó *là* "tôi đã liều hơn đối thủ bao nhiêu lần". Hệ quả
dây chuyền:

- "Đang bị bỏ lại" ⟺ "tôi đã an toàn hơn nó" ⟺ "risk tích luỹ của tôi thấp hơn nó"
  ⟺ "payoff tích luỹ của tôi thấp hơn nó". Bốn thứ này **hoàn toàn collinear** trong
  cơ chế gốc.
- Vậy hệ số `ΔS_{t-1} = −0.296` của paper người **không phân biệt được** giữa hai
  câu chuyện:
  - **(a) Áp lực vị trí** — "tôi dẫn trước nên không cần liều nữa";
  - **(b) Ngân sách rủi ro** — "tôi đã liều nhiều nên risk cao, giờ phải phanh".
- Paper kể câu chuyện (a) trong Discussion và đặt tên nó là *fear of falling behind*,
  nhưng dữ liệu không loại được (b).

Với LLM ta **có thể** tách được, vì ta điều khiển được cơ chế. Đây là đóng góp khoa học
lớn nhất mà plan này có thể tạo ra, và nó nằm ở Arm D (§1.4).

---

## §1. TRỤC 1 — Two-player non-persona (baseline)

### 1.1 Trạng thái: đã implement xong, chỉ cần freeze + scale

[`baseline.json`](../ai_race/configs/experiment/baseline.json) +
[`companies_default.json`](../ai_race/configs/agents/companies_default.json) đã đúng:
hai seat trung tính, persona rỗng, ba treatment 0.1/0.6/0.9, `runPhase: "pilot"`.
**Không cần đổi code cho arm này.** Chỉ cần tăng `repetitions` và freeze.

### 1.2 Vấn đề chí mạng #1 — Symmetry collapse

Prompt của seat 0 và seat 1 ở vòng 1 chỉ khác nhau ở `Company_1` / `Company_2`. Cùng
model, temperature = 0 → **hai seat gần chắc chắn cho cùng một action**. Nếu model lại
có xu hướng copy đối thủ (như người), race khoá vào mirror play vĩnh viễn:

```
ΔS(t) = 0.5·(n_U^own − n_U^opp) = 0  với mọi t
```

→ `race_state` luôn `"tied"` → **toàn bộ trục distance biến mất**;
`unsafe_by_race_state_turn.csv` chỉ còn một dòng; hệ số `progress_gap_before` không ước
lượng được. Và `_fit_clustered_logit` sẽ raise
`"clustered logit requires both Safe and Unsafe outcomes"` (`analyze_ai_race.py:2851`)
nếu model quá đơn điệu.

**Đây là kịch bản thất bại có xác suất cao nhất của cả nghiên cứu.** Ba lớp phòng vệ,
áp dụng đồng thời:

| Lớp | Biện pháp | Ghi chú |
|---|---|---|
| L1 | `temperature ∈ {0.7, 1.0}`, per-decision `sampling_seed` (đã có qua `game.sampling_seed()`) | Không dùng temp = 0 cho arm hành vi. Temp = 0 chỉ dùng cho một run *determinism check* riêng |
| L2 | **Arm B — đối thủ ngoại sinh** (§1.3) | Bẻ đối xứng bằng thiết kế, không dựa vào noise |
| L3 | Gate tiền nghiệm: pilot 10 rep; nếu tỉ lệ race có `ΔS ≡ 0` xuyên suốt > 40% → **dừng, không scale**, dồn nguồn lực sang Arm B/C | Ghi ngưỡng này vào preregistration trước khi chạy |

**Bổ sung — name-permutation check.** Cho 50% số race hoán vị thứ tự tên thành
`["Company_2", "Company_1"]` để đo seat effect thuần tuý. Nếu φ_U của seat 0 ≠ seat 1
khi persona rỗng, đó là artifact vị trí và mọi kết luận persona sau này bắt buộc phải
counterbalance.

### 1.3 Arm B — Đối thủ ngoại sinh (giải quyết Limitation 6 của paper)

Paper thừa nhận (§Limitations, điểm 6): `a_{-i}^{t-1}` **không ngoại sinh**, nên hệ số
+0.607 chỉ là tương quan có điều kiện, không phải nhân quả. Với LLM ta sửa được: **thay
một seat bằng chiến lược lập trình cứng.**

- Seat 1 = một trong `AS`, `AU`, `CS`, `CAS`, `BEHIND_UNSAFE`, `RANDOM(p=0.5)`.
- Seat 0 = LLM.
- Chiến lược script không "nhìn" LLM theo nghĩa chiến lược, chỉ theo luật của nó →
  `a_{-i}^{t-1}` trở thành **ngoại sinh thật sự** → hệ số của nó là **hiệu ứng nhân
  quả**, không phải association.

Hai contrast quan trọng nhất:

- **`AS` vs `AU`**: cùng một LLM, đối thủ luôn-Safe vs luôn-Unsafe. Hiệu số φ_U là ước
  lượng nhân quả sạch của "reciprocal unsafe" — con số paper người không thể có.
- **`RANDOM(0.5)`**: arm tốt nhất về mặt thống kê, vì nó cho variance tối đa trên cả
  `opp_prev` lẫn ΔS mà vẫn ngoại sinh hoàn toàn.

Engine hiện chưa có backend "scripted player" — cần thêm, chi tiết ở §4.2. Đây là **arm
có giá trị khoa học cao nhất trên mỗi đồng chi phí** trong toàn bộ plan.

### 1.4 Arm D — Tách ΔS khỏi lịch sử Unsafe (exploratory, non-canonical)

Từ §0(4): trong cơ chế gốc không thể tách. Cách rẻ nhất để tách: **handicap ngoại
sinh** — cho seat 0 một `initialProgress` ≠ 0, bốc theo rep từ
`{−1.5, −1, 0, +1, +1.5}`.

Khi đó:

```
ΔS(t) = handicap + 0.5·Δn_U
```

Hai nguồn biến thiên tách rời, và ta hồi quy đồng thời **cả `ΔS` lẫn `Δn_U`** — điều
paper người không làm được:

- Nếu hệ số `ΔS` sống sót khi kiểm soát `Δn_U` → **câu chuyện vị trí đúng**, "fear of
  falling behind" là hiệu ứng thật.
- Nếu chỉ `Δn_U` sống sót → hiệu ứng thật là **ngân sách rủi ro**, và cách paper diễn
  giải cần được đặt lại.

Ràng buộc thủ tục: đây là mechanism change → cần trường `initialProgress` trong
`GameConfig`, config game riêng, `run_phase` riêng, và **bắt buộc chạy analyser với
`--allow-noncanonical-mechanism`**. Không bao giờ gộp vào primary analysis.

### 1.5 Cỡ mẫu

Baseline hiện `repetitions: 3` → 3 rep × 3 treatment = 9 race/model = 18 trajectory.
**Quá nhỏ, không dùng được cho bất kỳ inference nào.**

Mục tiêu: **khớp cỡ mẫu phân tích của paper người** (2.888 quan sát, 172 cụm).

```
50 rep × 3 treatment = 150 race / model / điều kiện
                     = 300 trajectory
                     ≈ 2.700 player-round (sau khi bỏ vòng 1)
                     = 50 CRN block, mỗi block trải đủ 3 treatment ✓
                       (thoả gate ở analyze_ai_race.py:2856)
                     ≈ 2.700 lệnh gọi LLM / model / điều kiện
```

Quy trình hai bước, tránh đốt compute mù:

1. **Pilot 10 rep** → đo ICC trong CRN block, tỉ lệ degenerate, parse-failure rate,
   phân bố ΔS thực tế.
2. Tính lại N từ design effect quan sát được. **50 rep là sàn, không phải mục tiêu.**
   Nếu ICC cao (LLM đồng nhất, mirror nhiều) có thể cần 100+ rep, hoặc kết luận rằng
   arm này không informative và dồn sang Arm B.

---

## §2. TRỤC 2 — Persona

### 2.1 Nguyên tắc viết persona (quan trọng hơn nội dung persona)

Persona là **manipulation**, không phải **instruction**. Nếu viết "hãy chọn UNSAFE
nhiều hơn" thì ta chỉ đo khả năng tuân lệnh, không đo hành vi chiến lược. Bốn ràng buộc
bắt buộc, tất cả đều kiểm tra được bằng test tự động:

1. **Không nhắc chữ SAFE/UNSAFE**, không nhắc bất kỳ hành động cụ thể nào.
2. **Không nhắc lại payoff, risk number, hay luật chơi** — không được thêm thông tin
   chiến lược mà baseline không có. Đây chính là cảnh báo trong
   [`references/papers/markdown/falling-behind-ai-race.md`](../references/papers/markdown/falling-behind-ai-race.md)
   dòng 83: personas là *experimental factor* vì chúng có thể thêm thông tin chiến lược
   không có trong task gốc.
3. **Cân độ dài** — mọi persona nằm trong ±10% số token của nhau, và có một persona
   `neutral` cùng độ dài làm placebo, để tách "hiệu ứng persona" khỏi "hiệu ứng có thêm
   một đoạn text".
4. **Hash và ghi lại** — `persona_sha256` vào run manifest.

Test cho (1) và (3) đặt ở `ai_race/tests/test_personas.py`.

### 2.2 Lưới điều kiện

Hai họ persona: risk preference (trục *dispositional*) và social orientation (trục
*chiến lược*).

**Họ R — Risk preference** (đối xứng, cả hai seat cùng persona):

| Cell | Seat 0 | Seat 1 | Vai trò |
|---|---|---|---|
| `R0` | neutral placebo | neutral placebo | Placebo cùng độ dài |
| `R−` | risk-averse | risk-averse | |
| `R+` | risk-seeking | risk-seeking | |

**Họ S — Social orientation** (cả đối xứng lẫn bất đối xứng):

| Cell | Seat 0 | Seat 1 | Vai trò |
|---|---|---|---|
| `S_CC` | cooperative | cooperative | Cả hai hợp tác |
| `S_AA` | adversarial | adversarial | Cả hai đối kháng |
| `S_AC` | adversarial | cooperative | **Bất đối xứng — cell quan trọng nhất** |
| `S_CA` | cooperative | adversarial | Counterbalance seat cho `S_AC` |

`S_AC`/`S_CA` là cell có giá trị nhất: nó đo **hợp tác có sống sót được trước đối kháng
không**, và nó bẻ đối xứng theo thiết kế nên né được vấn đề §1.2. Bắt buộc có cả hai
chiều, nếu không hiệu ứng persona lẫn với hiệu ứng seat.

Cộng baseline `none` → **8 điều kiện persona × 3 treatment risk = 24 cell**.

Nếu ngân sách chặt, cắt theo thứ tự ưu tiên: **giảm số model trước, không giảm số
cell**. Trong trường hợp buộc phải cắt cell, thứ tự bỏ là `R0` (mất placebo — chấp nhận
được nếu báo cáo rõ), rồi `S_CC`. Không bao giờ bỏ `S_CA` (mất counterbalance) hay bỏ cả
họ R (mất phần trả lời trực tiếp H2 của paper).

### 2.3 Vì sao trục risk-preference là đóng góp thật, không phải làm cho đủ

Paper người **đo** risk preference bằng Eckel–Grossman và ra null (hệ số −0.012 đến
−0.027, p > 0.1 ở mọi đặc tả). Nhưng đo lường có sai số, task elicitation có thể không
valid, và null của một biến *đo được* là bằng chứng yếu.

Với LLM ta **áp đặt** risk preference thay vì đo. Ba kết cục, cả ba đều báo cáo được:

- Persona risk-seeking/averse **không** dịch chuyển φ_U, nhưng persona
  adversarial/cooperative **có** → bằng chứng mạnh hơn paper gốc cho chính luận điểm của
  paper gốc: kênh chiến lược lấn át kênh khuynh hướng cá nhân. Manipulation > measurement.
- **Cả hai** đều dịch chuyển → LLM khác người ở chỗ nó phản ứng với framing tính cách, và
  đây là giới hạn của LLM như một mô hình cho hành vi người — phải nói rõ.
- Persona risk **có** dịch chuyển còn adversarial **không** → kết quả ngược paper, và cực
  kỳ đáng báo cáo.

Một thiết kế mà mọi kết cục đều informative là một thiết kế tốt.

### 2.4 Manipulation check

Không có kiểm tra này thì mọi null của trục persona đều vô nghĩa: không phân biệt được
"persona không ảnh hưởng hành vi" với "model không đọc persona". Hai check, chạy tách
khỏi game:

1. **Free-text probe** — cùng prompt, thay phần `CURRENT DECISION` bằng
   *"Describe your development philosophy in one sentence."* → coder mù (hoặc LLM judge
   với rubric cố định, ghi rõ rubric) phân loại về đúng persona. Yêu cầu ≥ 80% accuracy.
2. **Behavioural probe ngoài game** — một quyết định rủi ro đơn giản không liên quan đến
   race (bản LLM của Eckel–Grossman). Đây chính là bản LLM của Task 1 trong paper, và cho
   phép so trực tiếp: *khẩu vị rủi ro đo được* của LLM có dự báo φ_U không — song song
   hoàn hảo với H2 của paper.

### 2.5 Ba confound phải xử lý trước khi chạy

| Confound | Xử lý |
|---|---|
| Persona gắn với seat index | Counterbalance đầy đủ: mọi cell bất đối xứng có bản mirror (`S_AC` ↔ `S_CA`); đưa `seat` vào model như fixed effect |
| Persona = "thêm text" chứ không phải nội dung | Placebo `R0` cùng độ dài, cùng cấu trúc câu, nội dung vô thưởng vô phạt |
| Persona của đối thủ bị rò rỉ sang seat kia | Kiểm tra `build_prompt`: persona chỉ vào prompt của chính seat đó (đúng theo `prompt.py:71-79`), nhưng phải có test khẳng định — nếu rò rỉ thì `S_AC` biến thành trò chơi thông tin hoàn hảo và không còn so được với người |

---

## §3. TRỤC 3 — Analysis

### 3.1 Bản đồ paper → LLM → file analyser đã có

| Paper | Nội dung | Output analyser đã có | Còn thiếu |
|---|---|---|---|
| Fig 2A | φ_U theo treatment | `unsafe_by_risk_model_player.csv` | Không thiếu — nhưng **dùng Bảng S3 làm chuẩn so sánh, không dùng caption Fig 2A** (caption ghi ngược dấu, xem §3.4) |
| Fig 2B | Unsafe ~ ΔS × own_prev × opp_prev | `unsafe_by_race_state_turn.csv`, `opponent_response_turn.csv` | Thiếu **cell chéo 3 chiều** trong một bảng |
| Fig 2C | φ_U winner vs loser, tương quan trong cặp | `outcome_player.csv` | Thiếu **hệ số tương quan trong cặp** theo treatment |
| Table 1 | 6 model logit lồng nhau | `clustered_logit_coefficients.csv` — **chỉ 1 đặc tả** | Thiếu 5 đặc tả còn lại |
| Fig S1 | Phân bố số vòng | `race_quality.csv` | Cần histogram + kiểm tra mean ≈ 9 |
| Fig S5 | Phân bố chiến lược theo p_r^max | `strategy_summary_player.csv` | Đủ |

`LOGIT_FORMULA` hiện tại (`analyze_ai_race.py:27-30`) đúng bằng **model (6)** của paper.
Cần thêm 5 đặc tả kia để tái tạo cấu trúc nested và cho thấy hệ số ổn định hay không —
đây chính là chỗ paper người bộc lộ điểm yếu (`ΔS` chỉ significant ở model 6, không ở
model 3).

### 3.2 Xử lý ΔS: bin, centering, và bẫy collinearity

ΔS là **lattice**, chỉ nhận bội của 0.5, và với horizon trung bình 9 thì thực tế nằm
trong `[−4.5, +4.5]`, tập trung dày ở `{−1, −0.5, 0, +0.5, +1}`.

- Báo cáo **cả hai** dạng: ΔS liên tục (centered theo mean của **mẫu LLM**, ghi rõ giá
  trị centering — không dùng mean của mẫu người) và ΔS phân loại
  `{behind / tied / ahead}` (analyser đã tạo `race_state` ở `analyze_ai_race.py:1766-1770`).
- Thêm bin theo độ lớn: `{≤−1, −0.5, 0, +0.5, ≥+1}`. "Bị bỏ lại 0.5 bước" (lệch 1 vòng)
  khác về chất so với "bị bỏ lại 2 bước".
- **Bắt buộc report VIF** giữa `ΔS_{t-1}`, `own_prev_unsafe`, và risk tích luỹ. Từ
  §0(4), ΔS = 0.5·Δn_U nên ΔS tương quan cơ học với lịch sử Unsafe. Paper không report
  chỉ số này; ta nên.

### 3.3 Đơn vị cụm và multiplicity

- **Cụm:** `randomization_block_id` (`source_run::model::rep`). Analyser đã ép mỗi block
  phải trải ≥ 2 treatment (`analyze_ai_race.py:2856-2864`) — đúng, giữ nguyên. Khi pool
  cross-persona phải mở rộng, xem §4.3.
- **Primary vs secondary:** khai báo trước khi chạy. Đề xuất primary = ba hệ số
  `{opponent_prev, progress_gap, first_round}` trong đặc tả đầy đủ; Holm–Bonferroni trên
  3. Mọi thứ khác là exploratory và phải ghi nhãn rõ trong mọi bảng.
- **Loại trừ:** giữ nguyên luật của repo — một `parse_failed` là **loại cả race**, vì
  Safe fallback lan vào state các vòng sau. Không nới lỏng `parse_action` để làm đẹp tỉ
  lệ thành công.

### 3.4 Tiêu chí "replicate" — định nghĩa TRƯỚC khi nhìn kết quả

Đây là phần quyết định nghiên cứu có nói được gì hay không. Không có tiêu chí định trước
thì mọi kết quả đều trở thành "một phần giống người".

| # | Hiệu ứng người | Chuẩn người | Replicate nếu |
|---|---|---|---|
| E1 | Opponent prev Unsafe → Unsafe ↑ | β = +0.607, p = 0.002 | β > 0 và p < 0.05 sau Holm |
| E2 | Dẫn trước → Unsafe ↓ | β = −0.296, p = 0.048 | β < 0 và p < 0.05 |
| E3 | Vòng 1 Unsafe → sau Unsafe ↑ | β = +0.217, p < 0.1 | β > 0 và p < 0.10 |
| E4 | Own prev **không** dự báo | β = −0.193, n.s. | \|β\| < 0.3 **và** TOST kết luận tương đương |
| E5 | Treatment 0.6 vs 0.9 **không** khác | d = −0.027 | \|d\| < 0.2 qua TOST |
| E6 | 0.1 có φ_U **cao hơn** 0.6/0.9 | d ≈ +0.33 | d > 0.2, cùng dấu |
| E7 | φ_U tổng thể | 0.584 | Nằm trong [0.40, 0.75] |
| E8 | AS gần như không tồn tại | AS không là Nash ở mọi treatment | tỉ lệ trajectory phân loại AS < 10% |

**E4 và E5 là null replication** — phải dùng equivalence test (TOST), không phải
"p > 0.05 nên giống nhau". Đây là chỗ đa số paper LLM-replication làm sai.

**Cảnh báo về E6.** Caption Fig 2A của paper viết *"Unsafe play is significantly higher
in the 0.6 and 0.9 treatments than in the 0.1 treatment"*, nhưng Bảng S3 cho
mean φ_U = 0.640 (ở 0.1) vs 0.558 / 0.564; Cohen's d cho "0.1 vs 0.6" là **+0.341**
(dương ⇒ nhóm 0.1 cao hơn); hệ số hồi quy của 0.6/0.9 đều âm so với 0.1; và mô hình tiến
hoá cũng dự báo Unsafe cao nhất ở rủi ro thấp. Hướng đúng là **0.1 có Unsafe CAO hơn**.
Dùng Bảng S2/S3 làm chuẩn, không dùng câu văn trong caption.

Deliverable cuối cùng của trục này là **một bảng 8 dòng**: replicate / không / không kết
luận được, kèm hướng lệch. Rõ ràng hơn nhiều so với "LLM behaves similarly to humans".

### 3.5 Estimand chỉ LLM mới có (không có đối chứng người — báo cáo ở mục riêng)

- **Nhân quả thật của opponent action** (Arm B):
  `E[unsafe | opp = AU] − E[unsafe | opp = AS]`. Paper người không có con số này.
- **Tách vị trí vs ngân sách rủi ro** (Arm D): hệ số `ΔS` khi kiểm soát `Δn_U`.
- **Determinism / repeatability:** cùng seed, cùng prompt, chạy lại → tỉ lệ action trùng.
  Đây là **cận trên** của mọi hiệu ứng đo được và phải báo cáo trước mọi hệ số khác.
- **Tỉ số disposition/strategy:** effect size của persona chia cho effect size của
  opponent action. Đây là câu trả lời định lượng cho câu hỏi trung tâm của paper.
- **Sensitivity to seat/name:** artifact thuần, phải report để người sau biết.

### 3.6 Bảng mô tả tối thiểu phải sinh ra

Mọi bảng stratify theo `persona_condition × max_private_risk × model`:

1. φ_U theo treatment (turn-level và player-level) + t-test cặp + Cohen's d — bản LLM
   của Fig 2A / Bảng S2.
2. φ_U theo `opp_prev × own_prev` (4 ô) — bản LLM của Fig 2B.
3. φ_U theo `race_state × opp_prev` (6 ô) và theo bin ΔS 5 mức.
4. φ_U winner vs loser + tương quan trong cặp theo treatment — Fig 2C.
5. Ma trận chuyển trạng thái `P(action_t | own_{t−1}, opp_{t−1})` — 4×2, bản mô tả thuần
   của reciprocity, không cần model.
6. Phân bố nhãn chiến lược AS/AU/CS/CAS + tỉ lệ tie + mismatch rate — Fig S5. Giữ
   nguyên nguyên tắc của [`strategy_analysis/`](../strategy_analysis/README.md): không ép
   tie thành nhãn duy nhất, và `BEHIND_UNSAFE_EXPLORATORY` không bao giờ gộp với 4 nhãn
   canonical trong bảng confirmatory.
7. Phân bố horizon + kiểm tra mean ≈ 9 — Fig S1.
8. Protocol health: parse-failure rate, retry count, refusal rate, độ dài response.

---

## §4. Thay đổi code cần thiết (theo thứ tự bắt buộc)

### 4.1 Ghi persona vào output — chặn mọi run persona cho tới khi xong

| File | Thay đổi |
|---|---|
| [`state.py`](../ai_race/engine/state.py) | `GameConfig`: thêm `persona_condition: str = "none"`, `persona_sha256: str = ""`. `TurnRecord`: thêm `persona_condition`, `seat_persona_role` |
| [`recorder.py`](../ai_race/dataio/recorder.py) | `race_row` + `player_rows`: thêm `persona_condition` và `player_N_persona_role` / `persona_role` |
| [`run_experiment.py`](../ai_race/runner/run_experiment.py) | Đọc `personaCondition` + `personaRoles` từ agents config; hash nội dung persona |
| `analyze_ai_race.py:104` | Thêm `persona_condition` vào `CONTEXT` |
| `analyze_ai_race.py` | Gate mới: nếu `persona_condition` thiếu ở bất kỳ race nào → refuse, trừ khi có flag mới `--allow-missing-persona-condition`. Dùng cùng khuôn với `_resolve_prompt_versions` |
| [`kaggle/experiments/baseline.py`](../kaggle/experiments/baseline.py) (dòng 361) | Manifest: thêm `agents_name`, `agents_config_sha256`, `persona_condition`, `persona_sha256` |
| [`results/README.md`](../results/README.md) | Cập nhật mục "Expected schema" — `CLAUDE.md` yêu cầu, nếu không thì completed run sẽ fail audit |

**Gate consistency bắt buộc:** `persona_condition == "none"` ⟺ cả hai persona rỗng. Vi
phạm → raise. Đây là thứ ngăn được lỗi tệ nhất có thể xảy ra: chạy persona rồi báo cáo
như baseline.

### 4.2 Backend chiến lược script (cho Arm B)

`ai_race/models/scripted.py`: một callable tương thích `send_batch` trả
`ACTION: SAFE|UNSAFE` theo AS / AU / CS / CAS / BEHIND / RANDOM(seed).

Vấn đề kiến trúc: `send_batch(prompts, seeds)` chỉ nhận prompt string, không nhận state.
Hai lựa chọn:

- **(a)** parse state từ prompt — fragile, không nên;
- **(b)** cho `run_games_batched` route theo seat; seat scripted đọc `game` object trực
  tiếp.

**Chọn (b).** Và ghi `model = "scripted:AU"` để analyser thấy rõ đây không phải LLM và
không gộp nhầm vào thống kê model.

### 4.3 CRN block khi pool cross-persona

`randomization_block_id` hiện là `source_run::model::rep`. Khi pool nhiều persona run
directory, cùng `rep` chia sẻ horizon nhưng khác `source_run` → block bị tách sai.

Sửa: khi mọi run có cùng `experiment.seed` (đọc được từ manifest), block trở thành
`model::rep`; nếu seed khác nhau thì giữ nguyên và **cảnh báo rõ rằng CRN không trải qua
persona**. Điều kiện tiên quyết: **mọi experiment config dùng chung `"seed": 260726`**.

### 4.4 Sáu đặc tả logit lồng nhau

Đổi `_fit_clustered_logit` để chạy một list công thức thay vì một công thức duy nhất,
xuất `clustered_logit_coefficients.csv` có thêm cột `specification ∈ {1..6}`, khớp
Table 1 của paper.

---

## §5. Trình tự thực thi

Kaggle push/run/download là **checkpointed**: chạy một lệnh, hiện output, dừng, chờ
người dùng trước lệnh kế tiếp (theo `CLAUDE.md` và
[`kaggle/benchmarks/README.md`](../kaggle/benchmarks/README.md)).

| # | Bước | Gate để đi tiếp |
|---|---|---|
| 0 | Viết preregistration: primary estimand, tiêu chí §3.4, luật loại trừ, N, ngưỡng degenerate | Đóng băng **trước** khi nhìn output AI Race đầu tiên |
| 1 | Code §4.1 + §4.4 + tests; `pytest` xanh | Test persona-consistency và length-balance pass |
| 2 | Smoke local `--mock random`, 2 rep; kiểm tra cột persona xuất hiện đúng | Analyser chạy sạch trên output mock |
| 3 | **Kaggle pilot:** 1 model open-weight, điều kiện `none` + `S_AA`, 10 rep, temp 0.7 | 7 validation gate trong [PROJECT.md](../PROJECT.md) đều pass; tỉ lệ `ΔS ≡ 0` < 40%; parse-failure < 5% |
| 4 | Tính lại N từ ICC quan sát được ở pilot | Chốt số rep cuối cùng |
| 5 | Freeze prompt / config / agents / analysis plan; lật `runPhase` → `confirmatory` | Không sửa gì sau bước này |
| 6 | Chạy full lưới: 8 điều kiện × 3 treatment × N rep, 1–2 model open-weight | Manifest `status = completed` cho mọi run |
| 7 | Code §4.2, chạy Arm B (đối thủ ngoại sinh) | |
| 8 | Chạy analyser **một lần**, không đổi định nghĩa outcome sau khi thấy kết quả | |
| 9 | Arm D (non-canonical, `--allow-noncanonical-mechanism`), báo cáo tách riêng | |
| 10 | Điền Results vào `paper/` và figure vào `slides/` | Chỉ sau khi bước 8 xong |

Về chi phí: full lưới ≈ 2.700 call × 8 điều kiện ≈ 21.6k call / model. Với open-weight
trên GPU Kaggle là khả thi. Với frontier API, chạy lưới rút gọn (`none`, `S_AA`, `S_AC`,
`R+`) và nói rõ trong Methods rằng lưới bị cắt vì chi phí.

---

## §6. Danh sách freeze (không được đổi sau bước 5)

- `ai_race/prompts/ai_race_en.txt` — hash `6180d4f699813a602a53cf4290b972aa4df4bf02ff1c646a85ab09d80d7729ff`
- Nội dung mọi persona (+ `persona_sha256`)
- `seed: 260726` cho **mọi** experiment config (điều kiện để CRN trải qua persona)
- Temperature, max_tokens, decoding params
- Luật loại trừ (parse-failure loại cả race)
- Tiêu chí replicate §3.4 và bộ primary estimand §3.3
- Số rep

## §7. Rủi ro đã biết

| Rủi ro | Xác suất | Giảm thiểu |
|---|---|---|
| Symmetry collapse, ΔS ≡ 0 | **Cao** | §1.2 ba lớp L1/L2/L3 |
| Model quá đơn điệu (toàn AU hoặc toàn AS) → logit không fit | Trung bình | Report φ_U mô tả kể cả khi logit fail; Arm B cứu variance |
| Persona bị pool ngầm với baseline | Cao nếu không vá §4.1 | Gate consistency, bắt buộc trước run persona |
| Persona không được đọc → null giả | Trung bình | Manipulation check §2.4 |
| Persona confound với seat | Cao nếu không counterbalance | Mirror cell `S_AC` ↔ `S_CA` |
| Diễn giải ΔS sai vì collinearity | **Chắc chắn xảy ra nếu không xử lý** | §0(4), VIF ở §3.2, Arm D |
| Chi phí frontier API vượt ngân sách | Trung bình | Lưới rút gọn, khai báo rõ |
