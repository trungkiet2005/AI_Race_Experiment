# AI Race Pilot Visualization & Results Insight Bundle

Generated from artifact: `AI_Race_Experiment/docs/reports/pilot_insight_report/artifact.json`

## 1) Validation checklist and repository cleanup outcome
- Artifact generation script and visualization pipeline were rebuilt and validated against current tracked files.
- Hard checks passed for row/denominator consistency, parse-failure controls, and protocol provenance mapping.
- Duplicate folder-content candidates in `results/open_source` were checked by filename+size signatures; no exact duplicate folders were found except expected structural differences across pilot/smoke/multimodule experiments.

### Folder signature scan

| Folder | File count | Status |
|---|---:|---|
| `game_understanding_pilot` | 5 | kept as distinct workload domain |
| `gpu_run_archive` | 13 | kept as distinct workload domain |
| `prompt_sensitivity_pilot` | 28 | kept as distinct workload domain |
| `surface_sensitivity_pilot` | 5 | kept as distinct workload domain |
| `surface_sensitivity_smoke` | 3 | kept as distinct workload domain |

No exact-duplicate candidate folder signatures found.

## 2) Evidence dataset map from artifact

| Dataset | Rows | Purpose |
|---|---:|---|
| `headlines` | 4 | one-row KPI cards |
| `surface` | 18 | surface variant audit rows |
| `surface_noncontrol` | 17 | surface variants excluding canonical |
| `surface_ranked` | 18 | surface rows ordered by magnitude |
| `persona` | 6 | persona-condition unsafe rates |
| `asymmetric_roles` | 2 | role split in asymmetric persona conditions |
| `probes` | 18 | probe semantic/strict accuracy by condition/domain |
| `behavior` | 2 | behavioral paired calculator ablation |
| `evidence` | 3 | pilot evidence ledger |

## 3) Manifest visuals

| Type | ID | Dataset | Title |
|---|---|---|---|
| chart | surface_delta_chart | surface | Unsafe-rate change across surface variants |
| chart | surface_scatter_chart | surface_noncontrol | Entry-decision flips and full-trajectory shifts |
| chart | persona_chart | persona | Unsafe rate across persona conditions |
| chart | asymmetric_role_chart | asymmetric_roles | Role-specific behavior in asymmetric games |
| chart | probe_chart | probes | Semantic game-understanding accuracy |
| chart | behavior_chart | behavior | Unsafe behavior with and without a calculator card |
| table | surface_table | surface_ranked | Surface-variant audit detail |
| table | behavior_table | behavior | Calculator behavior audit |
| table | evidence_table | evidence | Pilot evidence ledger |
| metric-strip | surface_span_card | headlines | Surface span |
| metric-strip | persona_span_card | headlines | Persona span |
| metric-strip | probe_card | headlines | Probe semantic accuracy |
| metric-strip | health_card | headlines | Parse-failure health |

## 4) Core pilot findings (human-readable)

- Surface sensitivity span: **80.8%** from baseline 52.2%; max effect at `+37.1%` on `Position Risk Near Response` and min `-43.7%` on `Order Actions Reversed`.
- Surface unsafe rates range: **8.4%** (`Order Actions Reversed`) to **89.2%** (`Position Risk Near Response`).
- No-persona unsafe benchmark: **56.3%**; persona span **45.7%**.
- Highest-risk role gap: risk-averse was the most conservative; adversarial-vs-adversarial was the least conservative.
- Probe semantics: unaided semantic accuracy **52.1%**, calculator uplift **23.5%**.
- Game-understanding behavior: calculator condition rose unsafe from **52.0%** to **60.8%** while mean payoff remained statistically flat; tie rate dropped.
- Protocol health: parse failures **0** / audited observations **15751**.

### Lowest direct semantic probe domains (pilot output)
| Condition | Domain | Semantic Accuracy |
|---|---|---:|
| Direct | Expected payoff | 0.0% |
| Direct | State transition | 0.0% |

## 5) Outputs and where to find them

- Artifact: `docs/reports/pilot_insight_report/artifact.json`
- Single canonical insights file (this one): `results/visualization_insight_full.md`
- Rebuild command: `python results/scripts/build_pilot_insight_report.py`

## 6) Next cleanup action suggested
- Keep generated report assets under `docs/reports/pilot_insight_report/` and move any new analysis markdown snapshots to a single tracker file under `results/` (this file).
- If you want, we can next split this into a short technical appendix and a slide-ready insight digest file while preserving a single source of truth.

## 7) XAI auto-vector attribution audit (prompt-sensitivity turn logs)

The new explainability pass added `results/scripts/explain_action_xai.py`, which builds
surrogate, auto-vectorized classifiers for SAFE/UNSAFE actions and writes decision-level
interpretability artifacts for every requested run.

### 7.1 Inputs used

- Prompt logs:
  - `tmp/pilot_rebuild/pilot_identified_t1_0`
  - `results/frontier`
- Feature construction:
  - Numeric game-state and action-history fields (e.g., `step_increment`, `round_payoff`,
    `prompt_chars`, `response_chars`, lag states),
  - Categorical context (`run_group`, `run_treatment`, `model`, `persona_condition`,
    `seat_persona_role`, etc.),
  - Prompt text TF-IDF (`prompt`),
  - Optional raw response TF-IDF variant (`raw_response`) for leakage-aware comparison.

### 7.2 Current output sets

| variant | directory | target leakage note | status |
|---|---|---|---|
| Text-free prompt-only vectorization | `results/open_source/prompt_sensitivity_pilot/xai_auto_vector_encoder_no_response` | does not include `raw_response` | production-safe diagnostic baseline |
| Full prompt+response vectorization | `results/open_source/prompt_sensitivity_pilot/xai_auto_vector_encoder` | includes `raw_response`; high predictive performance can become proxy-to-leak driven | experimental/checking-only |

Both variants currently use the same 5,958 cleaned decision rows and share identical
unsafe rate (59.68%).

### 7.3 Key quantified findings

- **Performance (validation split):** AUC and accuracy are both near perfect (~1.0) in both variants.  
  This should be treated as an upper-bound *separability* signal, not a causal proof of a valid mechanism.
- **Dominant linear drivers (both variants):**
  - `step_increment` and `round_payoff` are the strongest positive unsafe-weighted features;
  - `prompt_chars` and `attempts` are substantial context/risk/format proxies;
  - `response_chars` appears strongly weighted when included.
- **Permutation robustness:** only a few low-order numeric fields produce non-zero
  permutation signal once the linear decision surface is estimated; text n-grams are mostly
  numerically near zero in this sample and should be read as unstable/unreliable feature ranking artifacts.
- **Surface protocol signal is confounded by hash/split granularity:** prompt-template groups are
  currently represented by short template hashes and split into many one-row groups; this collapses
  interpretable grouping, so protocol comparisons should be made on merged conditions first.

### 7.4 Files emitted

- `xai_model_metadata.json`
- `xai_global_importance.csv`
- `xai_permutation_importance.csv`
- `xai_local_explanations.csv`
- `xai_prompt_surface_summary.csv`
- `xai_input_snapshot.csv`
- `xai_target_distribution.json`
- `xai_markdown_summary.md`

### 7.5 Recommended next step

- Add a protocol-level aggregation pass before plotting (`run_treatment` + `persona_condition`
  + coarse text family), then regenerate top-attribution figures with fixed tokens to produce
  stable dashboard-grade visuals.

### 7.6 Visual artifacts generated

- `results/open_source/prompt_sensitivity_pilot/xai_auto_vector_encoder_no_response/xai_top_global_coefficients.png`
- `results/open_source/prompt_sensitivity_pilot/xai_auto_vector_encoder_no_response/xai_top_permutation_importance.png`
- `results/open_source/prompt_sensitivity_pilot/xai_auto_vector_encoder/xai_top_global_coefficients.png`
- `results/open_source/prompt_sensitivity_pilot/xai_auto_vector_encoder/xai_top_permutation_importance.png`

### 7.7 Reference XAI libraries for design traceability

- `tmp/xai_ref_sources/lime` (LIME reference implementation)
- `tmp/xai_ref_sources/shap` (SHAP reference implementation)

These reference repos are intentionally left in `/tmp` and not tracked by git;
they were used as design references for the prompt/text vectorization pipeline and
audit reporting style.
