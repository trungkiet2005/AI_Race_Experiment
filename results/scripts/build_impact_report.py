#!/usr/bin/env python3
"""Build the complete technical impact report as a portable artifact source."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "results" / "impact_upgrade"


def records(path: Path) -> list[dict]:
    frame = pd.read_csv(path)
    return json.loads(frame.to_json(orient="records"))


def source(source_id: str, label: str, path: str, description: str) -> tuple[dict, dict]:
    manifest = {"id": source_id, "label": label, "path": path}
    canonical = {
        "id": source_id,
        "query": {
            "engine": "python",
            "language": "sql",
            "sql": f"SELECT * FROM {Path(path).stem}",
            "description": description,
            "tables_used": [path],
            "filters": ["Evidence classes remain separate; no cross-protocol inferential pooling"],
            "metric_definitions": [
                "Unsafe rate is the within-player enacted Unsafe fraction unless a block says otherwise.",
                "Live context effects compare paired player trajectories under common random numbers.",
                "Fixed-state effects replay identical states and therefore isolate direct prompt response at those states.",
            ],
        },
    }
    return manifest, canonical


def build_markdown() -> str:
    return """# AI Race evidence synthesis and impact upgrade

## Technical summary

The upgraded evidence changes the paper's strongest defensible story. The result is not that one language model has a stable level of risk appetite. It is that **strategic behavior is jointly determined by model checkpoint, prompt surface, semantic framing, and endogenous trajectory feedback—even when payoff mechanics are unchanged**.

Five baseline checkpoints cover sharply different regimes: GPT-5 nano remains near 12–15% Unsafe, GPT-5.4 nano is U-shaped near 50–58%, while the three Gemini checkpoints begin at 84–100% Unsafe under 10% risk and decline as risk rises. These are cross-provider pilots and are never pooled inferentially, but they falsify any simple one-model generalization.

The Qwen context experiment provides a more controlled mechanism result. All paired first-round decisions agree, yet fixed-state replay detects direct context effects up to 16.7 percentage points and live trajectories reach a 34.0-point aggregate gap. The crucial new interaction is stronger: **the context effect is entirely gated by opaque action-code mapping in this pilot**. When Safe is code Q, the seven contexts produce no divergence from the abstract reference; when Safe is code P, six contexts diverge in every paired player trajectory, with the largest Unsafe-rate difference reaching 68.0 points. Mapping is balanced but assigned by repetition parity, so this is a high-priority replication target rather than a mapping-causal estimate.

The inference audit corrected an important dependency error before the follow-up launch. All three risk treatments reuse `base_seed + repetition`, so the project has **32 independent CRN repetition streams, not 96 risk-by-repetition clusters**. Historical context intervals and figures have been recomputed with risk strata inside the repetition cluster. The frozen 32-stream mapping grid therefore remains diagnostic. A prospective residual-bootstrap sensitivity uses the largest observed context-delta spread as a conservative variance proxy: 96 independent streams deliver estimated power 0.937 for a 15-point interaction after a seven-test multiplicity screen; this is a design calculation, not a behavioral result.

The validity boundary is equally important. Qwen passes rule recall (100%) and stage payoff (98.4%), but state updating is 12.5% and terminal scoring 17.2%; the preregistered comprehension admission gate therefore fails. SAE probes can predict action-associated representations (held-out AUC up to 0.985), but context steering causes no action flips in the context run, and the self-play target-minus-control intervention contrasts do not establish a reliable causal controller. Those negative results are retained, not hidden.

## Key findings and visual evidence

### 1. Cross-model heterogeneity is qualitative, not a scale shift

The same three risk caps produce monotone-decreasing Gemini curves, a low flat GPT-5 nano curve, and a U-shaped GPT-5.4 nano curve. Reporting only an aggregate model mean would erase the phenomenon the study is trying to measure. The paper should report checkpoint-level curves and frame cross-model evidence as replication of instability, not as one pooled treatment effect.

![Five checkpoint risk-response curves](figures/cross_model_risk_response.png)

### 2. Action-code position gates semantic framing

The mapping interaction is the most demo-worthy result because it is visible round by round and has a concrete experimental remedy. Opaque IDs were intended to neutralize Safe/Unsafe wording, yet which ID denotes Safe determines whether context can move behavior. A fully crossed diagnostic follow-up is now frozen: both mappings run inside every seed block instead of being assigned by repetition parity.

![Context sensitivity stratified by opaque action mapping](figures/context_mapping_gate.png)

### 3. Direct prompt response grows along live trajectories

Fixed-state replay measures the action change caused by context while holding the state constant. Live play repeats the context and allows earlier decisions to alter later states. Logistics has a 15.6-point fixed-state effect and a 34.0-point live effect; the 18.4-point difference is descriptive evidence consistent with amplification, not a causal mediation estimate because the analysis units differ.

![Fixed-state direct effects versus live trajectory effects](figures/context_direct_vs_live.png)

### 4. More aggressive progress does not improve realized payoff

Across all six Safe=P contexts that shifted behavior, Unsafe play increased while mean final payoff fell by 5.6 to 28.6 points relative to the paired abstract game. The largest behavioral shifts were also among the largest payoff losses (descriptive Pearson r=-0.81 across six context cells). This is not a universal welfare claim: the six points share one checkpoint and a parity-confounded mapping. It is nevertheless an important internal check—the framing effect changes consequential play, but does not make the agent better at the disclosed objective.

![Behavioral shift versus realized-payoff difference](figures/behavior_payoff_tradeoff.png)

### 5. No round-1 flip does not imply robustness

All 1,344 paired player-trajectory comparisons agree in round 1. Divergence begins later, then changes progress, private risk, setbacks, and terminal payoff. Entry-only audits therefore miss state-conditional sensitivity in repeated games.

![Accumulation of paired trajectory divergence over rounds](figures/trajectory_divergence_curve.png)

### 6. Mechanistic interpretability supplied a useful negative result

The FAST-SAE pipeline has pinned model/SAE revisions, held-out splits, matched random and unrelated-feature controls, and live self-play. It detects predictive representations, but the intervention evidence does not clear the causal bar. This distinguishes *decodable information* from *behavioral control* and makes the XAI section more credible.

![SAE decodability versus controlled intervention evidence](figures/xai_decodability_vs_control.png)

### 7. Power and stopping are now prospective

The 32-stream run is fixed as a diagnostic replication and cannot be promoted after seeing a favorable result. A separate 96-stream confirmatory target is frozen for a 15-point smallest effect of scientific interest, 80% target power, Holm family size seven, fixed N, and no optional continuation. Because the pilot mapping assignment is confounded, the power model is deliberately labelled a conservative sensitivity analysis.

![Prospective power sensitivity by independent CRN streams](power/context_mapping_power.png)

### 8. The evidence ladder prevents impressive diagnostics from becoming overclaims

The result families occupy different evidential levels: mechanical validation and EGT reconstruction verify implementations; context, identity, position, and cross-model runs diagnose behavior; SAE probes establish association; controlled interventions test but currently do not establish feature-specific causation. Keeping these lanes visible is itself a core result because it shows exactly which claims are ready for the main paper and which belong in an exploratory appendix.

![Evidence classes and their current promotion boundary](figures/evidence_ladder.png)

## Scope, data, and metric definitions

The synthesis covers 768 Qwen T=0 context races (13,680 decisions), a separate matched T=0.7 stratum of the same size, 2,640 OpenAI pilot races (49,104 decisions), 177 Gemini pilot races (3,168 decisions), an N=3 Qwen pilot, FAST-SAE representation and intervention audits, and an independent EGTTools transition validation. Temperature strata and provider protocols are not pooled.

The player-level Unsafe rate averages each player's trajectory before aggregation. Mapping-stratified live contrasts compare the same risk, repetition, player seat, and mapping against the abstract context. Kaplan–Meier curves treat race termination as censoring. The fixed-state estimand compares prompts on identical replayed states. Terminal payoff differences are realized, so they include setback draws fixed by common-random-number block.

## Methodology and quality gates

Every source is admitted through explicit checks: manifest status and phase, model and prompt identity, expected cell coverage, zero parse failures where claimed, player/turn count reconciliation, canonical mechanics, and common-random-number alignment. The impact script fails closed on missing contexts, risks, mappings, rows, or inconsistent horizons. Output tables and figures are hashed in `analysis_manifest.json`.

The EGT reconstruction is labelled faithful rather than bitwise because the original paper has not released its private code and seeds. The independent transition matrix matches the pinned EGTTools implementation to 1.11e-16 and its stationary distribution to 7.63e-15 in the validation case.

## Limitations, uncertainty, and robustness boundary

- Context findings remain **diagnostic** because the comprehension admission gate fails.
- Earlier context uncertainty used risk-by-repetition clusters. The corrected analysis clusters by repetition because risk strata share the same RNG stream; estimates are unchanged but intervals are wider where dependence matters.
- OpenAI and Gemini results are **pilots**; their local-run manifests do not identify persona effects cleanly across protocol signatures.
- Mapping is balanced but assigned by repetition parity in the completed live context runs. The mapping interaction is therefore a replication target, not a clean mapping main effect.
- Decisions within a race are dependent. Turn-weighted rates are descriptive, not independent Bernoulli trials.
- The live-minus-fixed gap combines repeated prompt exposure and endogenous state feedback; it is not a causal mediation effect.
- SAE feature association is not feature causation. The strongest controlled intervention contrasts do not support a reliable action controller.
- N=3 persona cells have two races each and belong in the demo/appendix, not the headline claim.

## Recommended next experiments

1. **Complete admission before gameplay, then run the fully crossed diagnostic.** For every seed, execute both Safe=P and Safe=Q across all eight contexts only after provenance and comprehension are recorded. The checked-in 1,536-race grid remains diagnostic; the independent 96-stream replication is the frozen confirmatory target.
2. **Admission-gated cross-family replication.** Run at least three model families at one fixed decoding setting; analyze gameplay only for model/configuration cells that pass rule recall, state transition, terminal scoring, and expected-payoff thresholds.
3. **Direct-versus-feedback replay.** Freeze a logged state sequence, replay all contexts at each state, then separately launch endogenous trajectories from the same first divergence. This creates commensurable direct and feedback estimands.
4. **Opaque-ID randomization per decision.** Randomize labels on every turn and decode after response to test whether position bias persists when a stable code policy cannot form.
5. **Causal SAE promotion rule.** Increase discovery action flips, pre-register feature selection on discovery races, and require target steering to exceed matched-random and unrelated-feature controls on held-out states and live races.

## Further questions

- Why does context have leverage only under one opaque mapping: token prior, recency, instruction position, or a learned code policy?
- Which comprehension failure best predicts later divergence after controlling for model and first-round choice?
- Are the Gemini risk slopes stable under the same action mapping and prompt template used for Qwen?
- Does trajectory feedback amplify context because of self-consistency, opponent imitation, progress-gap response, or terminal-risk calculation?
- Can a neutral verified state tool improve payoff without itself becoming a new framing treatment?

## Literature alignment

The result aligns with recent work showing that equivalent strategic games can change under narrative context ([Same Game, Different Story](https://arxiv.org/abs/2607.19670)) and that framing shapes LLM decisions ([Framing the Game](https://arxiv.org/abs/2503.04840)). It also addresses the warning that some apparent prompt sensitivity can be an evaluation artifact ([Hua et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1006/)): here actions are parsed as exact opaque codes, mechanics are recomputed from raw logs, and divergence changes enacted trajectories, so the main effect is not a fuzzy answer-matching artifact. The XAI boundary follows SAE work that separates representation discovery from controlled intervention ([SAIF](https://arxiv.org/abs/2502.11356), [FAST](https://arxiv.org/abs/2506.07691), [RouteSAE](https://arxiv.org/abs/2503.08200)).
"""


def build_artifact() -> dict:
    data_dir = OUT / "data"
    cross_model = records(data_dir / "cross_model_baseline_rates.csv")
    mapping = records(data_dir / "context_mapping_interaction.csv")
    tradeoff = records(data_dir / "context_behavior_payoff_tradeoff.csv")
    decomposition = records(data_dir / "context_direct_vs_live.csv")
    divergence = records(data_dir / "trajectory_divergence_summary.csv")
    ledger = records(data_dir / "evidence_ledger.csv")
    quality = json.loads((OUT / "data_quality_audit.json").read_text(encoding="utf-8"))
    power = records(OUT / "power" / "power_grid.csv")
    temperature = pd.read_csv(
        ROOT / "results/open_source/context_skin_pilot/analysis_temperature_robustness/tables/paired_overall_summary.csv"
    ).iloc[0]
    comprehension = records(
        ROOT / "results/open_source/context_skin_pilot/analysis_live_pilot_t0/comprehension_by_domain.csv"
    )

    max_mapping = max(row["mean_unsafe_delta"] for row in mapping if row["mapping"] == "safe_p")
    max_gap = max(row["live_minus_fixed_descriptive_gap_pp"] for row in decomposition)
    headlines = [
        {"id": "models", "value": 5, "secondary": 3},
        {"id": "mapping", "value": max_mapping, "secondary": 0.0},
        {"id": "temperature", "value": temperature["mean_action_agreement"], "secondary": temperature["exact_player_trajectory_rate"]},
        {"id": "quality", "value": quality["parse_failures"], "secondary": quality["n_turns"]},
    ]
    sources_spec = [
        ("cross_model", "Cross-model pilot baselines", "results/impact_upgrade/data/cross_model_baseline_rates.csv", "Player-level baseline Unsafe rates for two OpenAI and three Gemini checkpoints."),
        ("mapping", "Context × mapping interaction", "results/impact_upgrade/data/context_mapping_interaction.csv", "Paired Qwen T=0 trajectory comparisons against the abstract context, stratified by opaque mapping."),
        ("tradeoff", "Behavior and payoff trade-off", "results/impact_upgrade/data/context_behavior_payoff_tradeoff.csv", "Safe=P context-level behavioral shifts and paired realized-payoff differences."),
        ("decomposition", "Direct versus live context effects", "results/impact_upgrade/data/context_direct_vs_live.csv", "Fixed-state direct and live full-trajectory context contrasts."),
        ("divergence", "Trajectory divergence", "results/impact_upgrade/data/trajectory_divergence_summary.csv", "Player-trajectory divergence and payoff consequences."),
        ("evidence", "Evidence ledger", "results/impact_upgrade/data/evidence_ledger.csv", "Evidence class, coverage, quality gate, and permitted paper use."),
        ("comprehension", "Comprehension admission", "results/open_source/context_skin_pilot/analysis_live_pilot_t0/comprehension_by_domain.csv", "Semantic and strict-valid accuracy by game-understanding domain."),
        ("power", "Prospective power sensitivity", "results/impact_upgrade/power/power_grid.csv", "Monte Carlo design sensitivity by independent CRN repetition streams and smallest interaction."),
    ]
    manifest_sources, canonical_sources = [], []
    for spec in sources_spec:
        left, right = source(*spec)
        manifest_sources.append(left)
        canonical_sources.append(right)

    cards = [
        {"id": "model_card", "description": "Baseline checkpoints represented in the descriptive replication.", "dataset": "headlines", "sourceId": "cross_model", "filter": {"id": "models"}, "metrics": [{"label": "Model checkpoints", "field": "value", "format": "number"}, {"label": "Provider protocols", "field": "secondary", "format": "number"}]},
        {"id": "mapping_card", "description": "Largest context difference under Safe=P; all Safe=Q differences are zero.", "dataset": "headlines", "sourceId": "mapping", "filter": {"id": "mapping"}, "metrics": [{"label": "Largest Safe=P context shift", "field": "value", "format": "percent"}, {"label": "Largest Safe=Q shift", "field": "secondary", "format": "percent"}]},
        {"id": "temperature_card", "description": "T=0 versus T=0.7 matched action and exact-trajectory stability.", "dataset": "headlines", "sourceId": "divergence", "filter": {"id": "temperature"}, "metrics": [{"label": "Action agreement", "field": "value", "format": "percent"}, {"label": "Exact player trajectories", "field": "secondary", "format": "percent"}]},
        {"id": "quality_card", "description": "Raw T=0 context decisions admitted by the impact audit.", "dataset": "headlines", "sourceId": "evidence", "filter": {"id": "quality"}, "metrics": [{"label": "Parse failures", "field": "value", "format": "number"}, {"label": "Audited decisions", "field": "secondary", "format": "number"}]},
    ]
    charts = [
        {"id": "cross_model_chart", "title": "Five checkpoints occupy different behavioral regimes", "subtitle": "Pilot player-level Unsafe rates; provider protocols remain separate.", "type": "line", "dataset": "cross_model", "sourceId": "cross_model", "valueFormat": "percent", "encodings": {"x": {"field": "max_private_risk", "type": "quantitative", "label": "Maximum private risk"}, "y": {"field": "unsafe_rate", "type": "quantitative", "label": "Unsafe rate"}, "color": {"field": "model_label", "type": "nominal", "label": "Checkpoint"}, "tooltip": [{"field": "model_label", "type": "nominal", "label": "Checkpoint"}, {"field": "n_players", "type": "quantitative", "label": "Players"}]}},
        {"id": "mapping_chart", "title": "Action-code position gates context sensitivity", "subtitle": "Difference from abstract contest; 96 paired player-races per context × mapping.", "type": "bar", "dataset": "mapping", "sourceId": "mapping", "valueFormat": "percent", "encodings": {"x": {"field": "context", "type": "nominal", "label": "Context"}, "y": {"field": "mean_unsafe_delta", "type": "quantitative", "label": "Unsafe-rate difference"}, "color": {"field": "mapping", "type": "nominal", "label": "Mapping"}, "tooltip": [{"field": "ever_diverged_rate", "type": "quantitative", "label": "Ever diverged", "format": "percent"}, {"field": "mean_final_payoff_delta", "type": "quantitative", "label": "Mean payoff difference"}]}},
        {"id": "decomposition_chart", "title": "Direct effects and live trajectory effects", "subtitle": "The distance from the diagonal is descriptive amplification, not causal mediation.", "type": "scatter", "dataset": "decomposition", "sourceId": "decomposition", "encodings": {"x": {"field": "fixed_direct_effect_pp", "type": "quantitative", "label": "Fixed-state direct effect (pp)"}, "y": {"field": "live_effect_pp", "type": "quantitative", "label": "Live trajectory effect (pp)"}, "color": {"field": "context", "type": "nominal", "label": "Context"}, "tooltip": [{"field": "context", "type": "nominal", "label": "Context"}, {"field": "live_minus_fixed_descriptive_gap_pp", "type": "quantitative", "label": "Descriptive gap (pp)"}]}},
        {"id": "tradeoff_chart", "title": "Behavioral shift versus realized payoff", "subtitle": "Safe=P diagnostic cells; all six shifted contexts increase Unsafe play and reduce mean payoff.", "type": "scatter", "dataset": "tradeoff", "sourceId": "tradeoff", "encodings": {"x": {"field": "unsafe_delta_pp", "type": "quantitative", "label": "Unsafe-rate difference (pp)"}, "y": {"field": "payoff_delta", "type": "quantitative", "label": "Mean final-payoff difference"}, "color": {"field": "context", "type": "nominal", "label": "Context"}, "tooltip": [{"field": "context", "type": "nominal", "label": "Context"}, {"field": "n_paired_player_races", "type": "quantitative", "label": "Paired player-races"}]}},
        {"id": "comprehension_chart", "title": "Rule recall survives; state reasoning does not", "subtitle": "Semantic accuracy in the T=0 context comprehension audit.", "type": "bar", "dataset": "comprehension", "sourceId": "comprehension", "valueFormat": "percent", "encodings": {"x": {"field": "domain", "type": "nominal", "label": "Domain"}, "y": {"field": "semantic_accuracy", "type": "quantitative", "label": "Semantic accuracy"}, "tooltip": [{"field": "n", "type": "quantitative", "label": "Responses"}, {"field": "strict_valid_rate", "type": "quantitative", "label": "Strict-valid rate", "format": "percent"}]}},
        {"id": "power_chart", "title": "Power is governed by independent repetition streams", "subtitle": "Conservative pilot-residual sensitivity; familywise alpha 0.05 across seven contexts.", "type": "line", "dataset": "power", "sourceId": "power", "valueFormat": "percent", "encodings": {"x": {"field": "n_crn_repetition_streams", "type": "quantitative", "label": "Independent CRN streams"}, "y": {"field": "power_holm_single_step", "type": "quantitative", "label": "Estimated power"}, "color": {"field": "true_interaction", "type": "nominal", "label": "True interaction"}, "tooltip": [{"field": "true_interaction", "type": "quantitative", "label": "Interaction", "format": "percent"}, {"field": "monte_carlo_se", "type": "quantitative", "label": "Monte Carlo SE"}]}},
    ]
    tables = [
        {"id": "evidence_table", "title": "Evidence ledger", "subtitle": "Full quality gates and permitted uses are preserved in the source CSV.", "dataset": "evidence", "sourceId": "evidence", "columns": [{"field": "study", "label": "Study", "type": "text"}, {"field": "evidence_class", "label": "Class", "type": "text"}, {"field": "races", "label": "Races", "format": "number"}, {"field": "decisions_or_rows", "label": "Decisions / rows", "format": "number"}]},
        {"id": "context_table", "title": "Direct and live context contrasts", "subtitle": "Percentage-point contrasts against abstract contest.", "dataset": "decomposition", "sourceId": "decomposition", "columns": [{"field": "context", "label": "Context", "type": "text"}, {"field": "fixed_direct_effect_pp", "label": "Fixed direct (pp)", "format": "number"}, {"field": "live_effect_pp", "label": "Live (pp)", "format": "number"}, {"field": "live_minus_fixed_descriptive_gap_pp", "label": "Descriptive gap (pp)", "format": "number"}]},
    ]
    markdown = build_markdown()
    blocks = [
        {"id": "title", "type": "markdown", "body": "# AI Race evidence synthesis\n\n### Model × prompt × trajectory: a validity-first upgrade"},
        {"id": "summary", "type": "markdown", "body": markdown.split("## Key findings", 1)[0]},
        {"id": "metrics", "type": "metric-strip", "cardIds": ["model_card", "mapping_card", "temperature_card", "quality_card"]},
        {"id": "cross_model_text", "type": "markdown", "body": "## Key findings and visual evidence\n\n### Cross-model heterogeneity is qualitative, not a scale shift\n\nThe five pilot checkpoints show three different response shapes. This supports a checkpoint-level validity claim, not a pooled model-family effect."},
        {"id": "cross_model_block", "type": "chart", "chartId": "cross_model_chart"},
        {"id": "mapping_text", "type": "markdown", "body": "### Opaque action-code position gates semantic framing\n\nUnder **Safe=Q**, no paired live trajectory differs from the abstract reference. Under **Safe=P**, all 96 paired player-races diverge for six contexts, and the largest Unsafe shift is **68.0 points**. Because mapping follows repetition parity, this is the top replication target; a fully crossed 1,536-race diagnostic is now frozen and launch-ready."},
        {"id": "mapping_block", "type": "chart", "chartId": "mapping_chart"},
        {"id": "decomposition_text", "type": "markdown", "body": f"### Direct response grows along live trajectories\n\nThe largest live-minus-fixed descriptive gap is **{max_gap:.1f} points**. Fixed replay and live play use different units, so this is consistent with repeated exposure and feedback amplification but is not a mediation estimate."},
        {"id": "decomposition_block", "type": "chart", "chartId": "decomposition_chart"},
        {"id": "tradeoff_text", "type": "markdown", "body": "### Aggressive progress is not payoff improvement\n\nEvery shifted Safe=P context increases Unsafe play while reducing mean final payoff. Across six context cells, the descriptive Pearson correlation between behavioral and payoff differences is **-0.81**. This is consequential behavioral sensitivity, not improved optimization; the small, parity-confounded cell set prevents a general welfare claim."},
        {"id": "tradeoff_block", "type": "chart", "chartId": "tradeoff_chart"},
        {"id": "comprehension_text", "type": "markdown", "body": "### Comprehension is the admission bottleneck\n\nThe model can repeat rules and stage payoffs, but state update and terminal scoring fail. Gameplay therefore remains diagnostic even though code, parser, state transitions, and payoff accounting pass."},
        {"id": "comprehension_block", "type": "chart", "chartId": "comprehension_chart"},
        {"id": "power_text", "type": "markdown", "body": "### Power and stopping are prospective\n\nRisk strata reuse the same `base_seed + repetition` stream, so the independent unit is repetition. The 32-stream run remains diagnostic. A separate 96-stream replication is frozen for a 15-point interaction, fixed N, Holm family size seven, and no optional continuation; estimated design power is **93.7%** under the conservative pilot-residual sensitivity."},
        {"id": "power_block", "type": "chart", "chartId": "power_chart"},
        {"id": "rest", "type": "markdown", "body": "## Scope, data, methodology, limitations, next experiments, and further questions\n\n" + markdown.split("## Scope, data, and metric definitions", 1)[1]},
        {"id": "context_table_block", "type": "table", "tableId": "context_table"},
        {"id": "evidence_table_block", "type": "table", "tableId": "evidence_table"},
    ]
    generated = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "AI Race Evidence Synthesis and Impact Upgrade",
            "description": "Technical report covering cross-model pilots, context/mapping sensitivity, trajectory divergence, comprehension, XAI, and EGT validation.",
            "generatedAt": generated,
            "cards": cards,
            "charts": charts,
            "tables": tables,
            "sources": manifest_sources,
            "blocks": blocks,
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated,
            "status": "ready",
            "datasets": {
                "headlines": headlines,
                "cross_model": cross_model,
                "mapping": mapping,
                "tradeoff": tradeoff,
                "decomposition": decomposition,
                "divergence": divergence,
                "comprehension": comprehension,
                "power": power,
                "evidence": ledger,
            },
            "accessIssues": [],
        },
        "sources": canonical_sources,
        "package_info": {"originUrl": "artifact://ai-race-impact-upgrade", "controls": {"edit": False, "refresh": False}},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, default=OUT / "artifact.json")
    parser.add_argument("--markdown", type=Path, default=OUT / "impact_report.md")
    args = parser.parse_args()
    artifact = build_artifact()
    args.artifact.parent.mkdir(parents=True, exist_ok=True)
    args.artifact.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.markdown.write_text(build_markdown().rstrip() + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "artifact": str(args.artifact), "markdown": str(args.markdown)}, indent=2))


if __name__ == "__main__":
    main()
