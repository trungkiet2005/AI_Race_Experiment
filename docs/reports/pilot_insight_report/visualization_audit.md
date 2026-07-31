# Visualization Audit Report (Pilot)

Generated: `2026-07-31T09:46:10Z`  
Input artifact: [artifact.json](/AI_Race_Experiment/docs/reports/pilot_insight_report/artifact.json)

## 1) Objective

Audit all visualization outputs for the current pilot data:  
- surface-form prompt sensitivity  
- persona sensitivity  
- game-understanding probes  
- calculator behavior ablation  

The artifact drives an HTML report with 6 charts, 3 tables, and metric blocks.

## 2) Data + pipeline

- Script: [results/scripts/build_pilot_insight_report.py](/AI_Race_Experiment/results/scripts/build_pilot_insight_report.py)
- Input files:
  - `results/open_source/surface_sensitivity_pilot/variant_summary.csv`
  - `results/open_source/surface_sensitivity_pilot/risk_variant_summary.csv`
  - `results/open_source/prompt_sensitivity_pilot/unsafe_by_risk_model_turn.csv`
  - `results/open_source/prompt_sensitivity_pilot/seat_balance.csv`
  - `results/open_source/game_understanding_pilot/probe_summary.csv`
  - `results/open_source/game_understanding_pilot/behavior_summary.csv`
  - `results/open_source/game_understanding_pilot/behavior_by_risk.csv`
  - `results/open_source/game_understanding_pilot/admission.json`

## 3) Quality checks that were validated

The script asserts hard invariants before building output:
- Manifest status is completed; phase is pilot.
- Expected raw row counts: 18 surface rows, 54 surface-risk rows, 18 persona rows.
- Decision totals are internally consistent:
  - surface sum matches manifest coverage
  - persona total decisions = 3906
  - behavior total decisions = 1116
  - probe outputs = 685
- Parse failures = 0 for all audited data used here.
- Derived rates are recomputed and validated (e.g., `unsafe_count / n_decisions`).
- Risk-stratified and risk-behavior totals reconcile to overall totals.

Command run:

```bash
python AI_Race_Experiment/results/scripts/build_pilot_insight_report.py
```

Status from run:

```json
{
  "status": "ok",
  "output": "D:\\PhD_LetGoo\\PhD_Farming\\AI_Race\\AI_Race_Experiment\\docs\\reports\\pilot_insight_report\\artifact.json",
  "datasets": {
    "headlines": 4,
    "surface": 18,
    "surface_noncontrol": 17,
    "surface_ranked": 18,
    "persona": 6,
    "asymmetric_roles": 2,
    "probes": 18,
    "behavior": 2,
    "evidence": 3
  }
}
```

## 4) Artifact structure

| Dataset | Rows |
|---|---:|
| `headlines` | 4 |
| `surface` | 18 |
| `surface_noncontrol` | 17 |
| `surface_ranked` | 18 |
| `persona` | 6 |
| `asymmetric_roles` | 2 |
| `probes` | 18 |
| `behavior` | 2 |
| `evidence` | 3 |

## 5) Visualization inventory

### 5.1 Cards

| ID | Meaning |
|---|---|
| `surface_span_card` | Unsafe-rate span across surface variants |
| `persona_span_card` | Unsafe-rate span across persona arms |
| `probe_card` | Unaided semantic probe accuracy and calculator uplift |
| `health_card` | Parse failures and audited observations |

### 5.2 Charts

| ID | Type | Dataset | Title |
|---|---|---|---|
| `surface_delta_chart` | bar | `surface` | Unsafe-rate change across surface variants |
| `surface_scatter_chart` | scatter | `surface_noncontrol` | Entry-decision flips and full-trajectory shifts |
| `persona_chart` | bar | `persona` | Unsafe rate across persona conditions |
| `asymmetric_role_chart` | bar | `asymmetric_roles` | Role-specific behavior in asymmetric games |
| `probe_chart` | bar | `probes` | Semantic game-understanding accuracy |
| `behavior_chart` | bar | `behavior` | Unsafe behavior with and without a calculator card |

### 5.3 Tables

| ID | Dataset | Title |
|---|---|---|
| `surface_table` | `surface_ranked` | Surface-variant audit detail |
| `behavior_table` | `behavior` | Calculator behavior audit |
| `evidence_table` | `evidence` | Pilot evidence ledger |

## 6) Key findings (from generated artifact)

- Surface prompt sensitivity:
  - Canonical unsafe rate = **52.2%**
  - Min / Max unsafe across variants:
    - **8.4%** (`Order Actions Reversed`)
    - **89.2%** (`Position Risk Near Response`)
  - Span = **80.8 percentage points**
- Persona effects:
  - No-persona unsafe rate = **56.3%**
  - Risk-averse role = **31.2%** ( -25.1 pp vs no-persona )
  - Adversarial-vs-adversarial = **76.9%** ( +20.6 pp vs no-persona )
  - Persona arm span = **45.7 pp**
- Probe understanding:
  - Unaided semantic accuracy = **52.1%**
  - Calculator uplift = **+23.5 pp**
  - Hardest domains (semantic): expected payoff and state transition remain near 0% in unaided prompts.
- Calculator ablation:
  - Unsafe rate: **52.0% -> 60.8%** (+8.8 pp)
  - Mean final payoff: **42.77 -> 42.21**
  - Tie rate: **53.3% -> 33.3%**
- Protocol health:
  - Parse failures = **0 / 15,751** audited observations

## 7) Files to check

- Artifact: [artifact.json](/AI_Race_Experiment/docs/reports/pilot_insight_report/artifact.json)
- Candidate HTML: [candidate.html](/AI_Race_Experiment/docs/reports/pilot_insight_report/candidate.html)
- Last render-failure screenshots (if present):
  - `report.html.tmp-2912-afbd242c-25cc-4c8b-af9c-822fd891399f.validation-failure.png`
  - `report.html.tmp-6736-c34fe603-13a3-4ea3-a51e-cb10058634c2.validation-failure.png`

## 8) Current caveat

The visualization build logic and data reconciliation are valid. A known frontend check issue remains from previous verification runs: occasional `horizontal_overflow` in portable artifact rendering.  
That is a layout constraint issue, not a data-calculation bug.

## 9) Next step recommended

Run one more pass of layout tightening (label lengths / spacing / title sizes), then regenerate the report HTML for full pass in publish pipeline.
