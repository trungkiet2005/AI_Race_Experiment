#!/usr/bin/env python3
"""Build the audited pilot-synthesis artifact used by the portable HTML report."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def read_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def read_json(relative: str) -> dict:
    with (ROOT / relative).open(encoding="utf-8") as handle:
        return json.load(handle)


def number(row: dict[str, str], key: str) -> float:
    return float(row[key])


def integer(row: dict[str, str], key: str) -> int:
    return int(float(row[key]))


def close(left: float, right: float, tolerance: float = 1e-10) -> bool:
    return abs(left - right) <= tolerance


def source(source_id: str, label: str, path: str, description: str) -> tuple[dict, dict]:
    source_table = Path(path).name.replace(".", "_")
    manifest_source = {"id": source_id, "label": label, "path": path}
    canonical_source = {
        "id": source_id,
        "query": {
            "engine": "python",
            "language": "sql",
            "sql": f"SELECT * FROM {source_table}",
            "description": description,
            "tables_used": [path],
            "filters": ["Completed pilot rows only; smoke runs excluded"],
            "metric_definitions": [
                "Unsafe rate = enacted Unsafe decisions / all parsed player decisions.",
                "First-round flip rate = paired first-round decisions differing from canonical / paired first-round decisions.",
            ],
        },
    }
    return manifest_source, canonical_source


def build_artifact() -> dict:
    surface_path = "results/open_source/surface_sensitivity_pilot/variant_summary.csv"
    surface_risk_path = "results/open_source/surface_sensitivity_pilot/risk_variant_summary.csv"
    surface_manifest_path = "results/open_source/surface_sensitivity_pilot_manifest.json"
    persona_path = "results/open_source/prompt_sensitivity_pilot/unsafe_by_risk_model_turn.csv"
    seat_path = "results/open_source/prompt_sensitivity_pilot/seat_balance.csv"
    persona_manifest_path = "results/open_source/prompt_sensitivity_pilot/analysis_manifest.json"
    probe_path = "results/open_source/game_understanding_pilot/probe_summary.csv"
    behavior_path = "results/open_source/game_understanding_pilot/behavior_summary.csv"
    behavior_risk_path = "results/open_source/game_understanding_pilot/behavior_by_risk.csv"
    admission_path = "results/open_source/game_understanding_pilot/admission.json"

    surface_raw = read_csv(surface_path)
    surface_risk_raw = read_csv(surface_risk_path)
    surface_manifest = read_json(surface_manifest_path)
    persona_raw = read_csv(persona_path)
    seat_raw = read_csv(seat_path)
    persona_manifest = read_json(persona_manifest_path)
    probe_raw = read_csv(probe_path)
    behavior_raw = read_csv(behavior_path)
    behavior_risk_raw = read_csv(behavior_risk_path)
    admission = read_json(admission_path)

    # Data-quality gates: fail closed if a report would silently mix or misstate runs.
    assert surface_manifest["status"] == "completed"
    assert surface_manifest["run_phase"] == "pilot"
    assert len(surface_raw) == 18
    assert sum(integer(row, "n_decisions") for row in surface_raw) == surface_manifest["coverage"]["player_decisions"]
    assert sum(integer(row, "parse_failures") for row in surface_raw) == 0
    assert len(surface_risk_raw) == 54
    assert persona_manifest["all_manifest_counts_verified"] is True
    assert persona_manifest["n_races_total"] == persona_manifest["n_races_behavioral"] == 210
    assert persona_manifest["n_player_rounds_total"] == persona_manifest["n_player_rounds_behavioral"] == 3906
    assert persona_manifest["n_parse_failures"] == 0
    assert len(persona_raw) == 18
    assert sum(integer(row, "n_decisions") for row in persona_raw) == 3906
    for row in persona_raw:
        assert close(number(row, "unsafe_rate"), integer(row, "unsafe_count") / integer(row, "n_decisions"))
    assert admission["status"] == "completed"
    assert admission["evidence_class"] == "pilot"
    assert admission["probe_outputs"] == sum(integer(row, "n") for row in probe_raw if row["domain"] == "ALL") == 685
    assert sum(integer(row, "parse_failures") for row in behavior_raw) == 0
    assert sum(integer(row, "n_decisions") for row in behavior_raw) == 1116

    canonical = next(row for row in surface_raw if row["variant"] == "canonical")
    canonical_rate = number(canonical, "unsafe_rate")
    surface = []
    for row in surface_raw:
        delta = number(row, "unsafe_rate") - canonical_rate
        surface.append(
            {
                "variant": row["variant"],
                "label": row["variant"].replace("_", " ").title(),
                "family": row["family"],
                "interpretation": row["interpretation"].replace("_", " ").title(),
                "n_decisions": integer(row, "n_decisions"),
                "unsafe_rate": number(row, "unsafe_rate"),
                "ci_low": number(row, "unsafe_rate_cluster_bootstrap_ci95_low"),
                "ci_high": number(row, "unsafe_rate_cluster_bootstrap_ci95_high"),
                "unsafe_delta": delta,
                "absolute_delta": abs(delta),
                "first_round_flip": number(row, "first_round_flip_rate_vs_canonical"),
                "parse_failures": integer(row, "parse_failures"),
            }
        )
    surface.sort(key=lambda row: row["unsafe_delta"])

    # Reconcile risk-stratified summaries to the all-risk surface summary.
    for variant in {row["variant"] for row in surface_risk_raw}:
        strata = [row for row in surface_risk_raw if row["variant"] == variant]
        total_n = sum(integer(row, "n_decisions") for row in strata)
        weighted_rate = sum(integer(row, "n_decisions") * number(row, "unsafe_rate") for row in strata) / total_n
        overall = next(row for row in surface if row["variant"] == variant)
        assert total_n == overall["n_decisions"]
        assert close(weighted_rate, overall["unsafe_rate"])

    persona_order = ["R-", "S_CA", "S_AC", "none", "R0", "S_AA"]
    persona_labels = {
        "R-": "Risk-averse role",
        "R0": "Neutral role",
        "S_AA": "Adversarial vs adversarial",
        "S_AC": "Adversarial vs cooperative",
        "S_CA": "Cooperative vs adversarial",
        "none": "No persona",
    }
    persona = []
    for condition in persona_order:
        rows = [row for row in persona_raw if row["persona_condition"] == condition]
        decisions = sum(integer(row, "n_decisions") for row in rows)
        unsafe = sum(integer(row, "unsafe_count") for row in rows)
        persona.append(
            {
                "condition": condition,
                "label": persona_labels[condition],
                "n_decisions": decisions,
                "unsafe_count": unsafe,
                "unsafe_rate": unsafe / decisions,
                "delta_vs_no_persona": 0.0,
            }
        )
    no_persona = next(row["unsafe_rate"] for row in persona if row["condition"] == "none")
    for row in persona:
        row["delta_vs_no_persona"] = row["unsafe_rate"] - no_persona

    # In asymmetric games, seat 0 receives the first role and seat 1 the second.
    role_totals = {
        "Adversarial role": {"n_decisions": 0, "unsafe_count": 0},
        "Cooperative role": {"n_decisions": 0, "unsafe_count": 0},
    }
    for row in seat_raw:
        condition = row["persona_condition"]
        if condition not in {"S_AC", "S_CA"}:
            continue
        seat = integer(row, "seat_index")
        role = condition.split("_")[1][seat]
        label = "Adversarial role" if role == "A" else "Cooperative role"
        role_totals[label]["n_decisions"] += integer(row, "n_decisions")
        role_totals[label]["unsafe_count"] += integer(row, "unsafe_count")
    asymmetric_roles = []
    for label, values in role_totals.items():
        asymmetric_roles.append(
            {
                "role": label,
                **values,
                "unsafe_rate": values["unsafe_count"] / values["n_decisions"],
            }
        )

    condition_labels = {"direct": "Direct", "paraphrase": "Paraphrase", "calculator": "Calculator"}
    domain_labels = {
        "rule_recall": "Rule recall",
        "stage_payoff": "Stage payoff",
        "state_reconstruction": "State reconstruction",
        "state_transition": "State transition",
        "terminal_scoring": "Terminal scoring",
        "expected_payoff": "Expected payoff",
    }
    probes = []
    for row in probe_raw:
        if row["condition"] not in condition_labels or row["domain"] == "ALL":
            continue
        probes.append(
            {
                "condition": condition_labels[row["condition"]],
                "domain": domain_labels[row["domain"]],
                "n": integer(row, "n"),
                "semantic_accuracy": number(row, "semantic_accuracy"),
                "strict_accuracy": number(row, "strict_accuracy"),
            }
        )

    behavior_labels = {"canonical": "Canonical", "calculator_decision_card": "Calculator decision card"}
    behavior = []
    for row in behavior_raw:
        behavior.append(
            {
                "condition": behavior_labels[row["condition"]],
                "n_races": integer(row, "n_races"),
                "n_decisions": integer(row, "n_decisions"),
                "unsafe_rate": number(row, "unsafe_rate"),
                "ci_low": number(row, "unsafe_rate_cluster_ci95_low"),
                "ci_high": number(row, "unsafe_rate_cluster_ci95_high"),
                "mean_final_payoff": number(row, "mean_final_payoff"),
                "setback_rate": number(row, "setback_rate"),
                "tie_rate": number(row, "tie_rate"),
            }
        )

    # Verify risk-stratified behavior totals before exposing them.
    for condition in {row["condition"] for row in behavior_risk_raw}:
        strata = [row for row in behavior_risk_raw if row["condition"] == condition]
        overall = next(row for row in behavior_raw if row["condition"] == condition)
        assert sum(integer(row, "n_decisions") for row in strata) == integer(overall, "n_decisions")

    surface_span = max(row["unsafe_rate"] for row in surface) - min(row["unsafe_rate"] for row in surface)
    persona_span = max(row["unsafe_rate"] for row in persona) - min(row["unsafe_rate"] for row in persona)
    calculator_gain = admission["probe_calculator_semantic_accuracy"] - admission["probe_unaided_semantic_accuracy"]
    calculator_behavior_delta = behavior[1]["unsafe_rate"] - behavior[0]["unsafe_rate"]

    headlines = [
        {
            "id": "surface_span",
            "value": surface_span,
            "baseline": canonical_rate,
            "label": "Surface-form Unsafe span",
            "baseline_label": "Canonical Unsafe rate",
        },
        {
            "id": "persona_span",
            "value": persona_span,
            "baseline": no_persona,
            "label": "Persona-arm Unsafe span",
            "baseline_label": "No-persona Unsafe rate",
        },
        {
            "id": "probe_accuracy",
            "value": admission["probe_unaided_semantic_accuracy"],
            "baseline": calculator_gain,
            "label": "Unaided probe accuracy",
            "baseline_label": "Calculator uplift",
        },
        {
            "id": "protocol_health",
            "value": 0,
            "baseline": 15751,
            "label": "Parse failures",
            "baseline_label": "Audited decisions + probes",
        },
    ]

    evidence = [
        {
            "study": "Surface sensitivity pilot",
            "scope": "18 variants; 3 risk levels; 10 paired repetitions",
            "races": 540,
            "decisions_or_probes": 10044,
            "temperature": 0.7,
            "parse_failures": 0,
            "claim_class": "Exploratory descriptive",
        },
        {
            "study": "Persona sensitivity pilot",
            "scope": "6 analyzed persona arms; 3 risk levels",
            "races": 210,
            "decisions_or_probes": 3906,
            "temperature": 1.0,
            "parse_failures": 0,
            "claim_class": "Exploratory within protocol",
        },
        {
            "study": "Game-understanding audit",
            "scope": "41 probe items plus 2 behavior conditions",
            "races": 60,
            "decisions_or_probes": 1801,
            "temperature": 0.0,
            "parse_failures": 0,
            "claim_class": "Probe + behavioral audit",
        },
    ]

    manifest_sources = []
    canonical_sources = []
    for args in [
        ("surface", "Surface-form pilot summary", surface_path, "Loads audited surface-variant rates and paired first-round flips."),
        ("persona", "Persona pilot turn summary", persona_path, "Aggregates Unsafe decisions across risk strata within persona condition."),
        ("seats", "Asymmetric persona seat summary", seat_path, "Aggregates role-specific decisions after counterbalanced seat assignment."),
        ("probes", "Game-understanding probe summary", probe_path, "Loads strict and semantic accuracy by probe condition and domain."),
        ("behavior", "Calculator behavior ablation", behavior_path, "Loads paired behavioral outcomes for canonical and calculator decision-card conditions."),
        ("synthesis", "Pilot evidence ledger", "results/scripts/build_pilot_insight_report.py", "Reconciles manifests, denominators, rates, and report-ready derived comparisons."),
    ]:
        manifest_source, canonical_source = source(*args)
        manifest_sources.append(manifest_source)
        canonical_sources.append(canonical_source)

    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    manifest = {
        "version": 1,
        "surface": "report",
        "title": "AI Race Pilot Evidence: Competence, Prompt Sensitivity, and Behavioral Drift",
        "description": "An audited technical synthesis of the completed Qwen2.5-7B game-understanding, persona, and surface-form pilots.",
        "generatedAt": generated_at,
        "cards": [
            {
                "id": "surface_span_card",
                "description": "Range from the lowest to highest decision-weighted Unsafe rate across 18 surface variants.",
                "dataset": "headlines",
                "sourceId": "surface",
                "filter": {"id": "surface_span"},
                "metrics": [
                    {"label": "Surface-form Unsafe span", "field": "value", "format": "percent"},
                    {"label": "Canonical", "field": "baseline", "format": "percent"},
                ],
            },
            {
                "id": "persona_span_card",
                "description": "Range across analyzed persona arms, aggregated across risk levels.",
                "dataset": "headlines",
                "sourceId": "persona",
                "filter": {"id": "persona_span"},
                "metrics": [
                    {"label": "Persona-arm Unsafe span", "field": "value", "format": "percent"},
                    {"label": "No persona", "field": "baseline", "format": "percent"},
                ],
            },
            {
                "id": "probe_card",
                "description": "Semantic accuracy across all unaided probe variants, with calculator improvement as context.",
                "dataset": "headlines",
                "sourceId": "probes",
                "filter": {"id": "probe_accuracy"},
                "metrics": [
                    {"label": "Unaided semantic accuracy", "field": "value", "format": "percent"},
                    {"label": "Calculator uplift", "field": "baseline", "format": "percent", "signed": True},
                ],
            },
            {
                "id": "health_card",
                "description": "Protocol-health check across audited pilot decisions and probe outputs.",
                "dataset": "headlines",
                "sourceId": "synthesis",
                "filter": {"id": "protocol_health"},
                "metrics": [
                    {"label": "Parse failures", "field": "value", "format": "number"},
                    {"label": "Audited observations", "field": "baseline", "format": "number"},
                ],
            },
        ],
        "charts": [
            {
                "id": "surface_delta_chart",
                "title": "Unsafe-rate change across surface variants",
                "subtitle": "Decision-weighted change from the canonical prompt; 558 decisions per variant.",
                "type": "bar",
                "dataset": "surface",
                "sourceId": "surface",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "label", "type": "nominal", "label": "Prompt variant"},
                    "y": {"field": "unsafe_delta", "type": "quantitative", "label": "Change from canonical"},
                    "tooltip": [
                        {"field": "unsafe_rate", "type": "quantitative", "label": "Unsafe rate", "format": "percent"},
                        {"field": "first_round_flip", "type": "quantitative", "label": "First-round flip", "format": "percent"},
                    ],
                },
            },
            {
                "id": "surface_scatter_chart",
                "title": "Entry-decision flips and full-trajectory shifts",
                "subtitle": "Each point is one prompt variant relative to canonical; later rounds remain exposed to the variant.",
                "type": "scatter",
                "dataset": "surface_noncontrol",
                "sourceId": "surface",
                "encodings": {
                    "x": {"field": "first_round_flip", "type": "quantitative", "label": "First-round flip rate"},
                    "y": {"field": "unsafe_delta", "type": "quantitative", "label": "Unsafe-rate change"},
                    "color": {"field": "interpretation", "type": "nominal", "label": "Variant class"},
                    "tooltip": [
                        {"field": "label", "type": "nominal", "label": "Variant"},
                        {"field": "family", "type": "nominal", "label": "Family"},
                    ],
                },
            },
            {
                "id": "persona_chart",
                "title": "Unsafe rate across persona conditions",
                "subtitle": "Decision-weighted rates across three risk levels; the no-persona arm is the descriptive baseline.",
                "type": "bar",
                "dataset": "persona",
                "sourceId": "persona",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "label", "type": "nominal", "label": "Persona condition"},
                    "y": {"field": "unsafe_rate", "type": "quantitative", "label": "Unsafe rate"},
                    "tooltip": [
                        {"field": "delta_vs_no_persona", "type": "quantitative", "label": "Change vs no persona", "format": "percent"},
                        {"field": "n_decisions", "type": "quantitative", "label": "Decisions"},
                    ],
                },
            },
            {
                "id": "asymmetric_role_chart",
                "title": "Role-specific behavior in asymmetric games",
                "subtitle": "Seat-counterbalanced aggregation of S_AC and S_CA conditions.",
                "type": "bar",
                "dataset": "asymmetric_roles",
                "sourceId": "seats",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "role", "type": "nominal", "label": "Assigned role"},
                    "y": {"field": "unsafe_rate", "type": "quantitative", "label": "Unsafe rate"},
                    "tooltip": [{"field": "n_decisions", "type": "quantitative", "label": "Decisions"}],
                },
            },
            {
                "id": "probe_chart",
                "title": "Semantic game-understanding accuracy",
                "subtitle": "Repeated item-level probes; accuracy is semantic, not strict-format compliance.",
                "type": "bar",
                "dataset": "probes",
                "sourceId": "probes",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "domain", "type": "nominal", "label": "Probe domain"},
                    "y": {"field": "semantic_accuracy", "type": "quantitative", "label": "Semantic accuracy"},
                    "color": {"field": "condition", "type": "nominal", "label": "Prompt condition"},
                    "tooltip": [
                        {"field": "n", "type": "quantitative", "label": "Probe outputs"},
                        {"field": "strict_accuracy", "type": "quantitative", "label": "Strict accuracy", "format": "percent"},
                    ],
                },
            },
            {
                "id": "behavior_chart",
                "title": "Unsafe behavior with and without a calculator card",
                "subtitle": "Thirty races and 558 decisions per condition; intervals are race-cluster bootstrap 95% CIs.",
                "type": "bar",
                "dataset": "behavior",
                "sourceId": "behavior",
                "valueFormat": "percent",
                "encodings": {
                    "x": {"field": "condition", "type": "nominal", "label": "Decision condition"},
                    "y": {"field": "unsafe_rate", "type": "quantitative", "label": "Unsafe rate"},
                    "tooltip": [
                        {"field": "ci_low", "type": "quantitative", "label": "CI low", "format": "percent"},
                        {"field": "ci_high", "type": "quantitative", "label": "CI high", "format": "percent"},
                        {"field": "mean_final_payoff", "type": "quantitative", "label": "Mean final payoff"},
                    ],
                },
            },
        ],
        "tables": [
            {
                "id": "surface_table",
                "title": "Surface-variant audit detail",
                "subtitle": "All completed pilot variants, ranked by absolute change from canonical.",
                "dataset": "surface_ranked",
                "sourceId": "surface",
                "defaultSort": {"field": "unsafe_delta", "direction": "desc"},
                "columns": [
                    {"field": "label", "label": "Variant", "type": "text"},
                    {"field": "family", "label": "Family", "type": "text"},
                    {"field": "unsafe_rate", "label": "Unsafe rate", "format": "percent"},
                    {"field": "unsafe_delta", "label": "Change vs canonical", "format": "percent", "movement": True},
                    {"field": "first_round_flip", "label": "First-round flip", "format": "percent"},
                    {"field": "n_decisions", "label": "Decisions", "format": "number"},
                ],
            },
            {
                "id": "behavior_table",
                "title": "Calculator behavior audit",
                "subtitle": "Behavioral outcomes for the paired canonical and calculator decision-card runs.",
                "dataset": "behavior",
                "sourceId": "behavior",
                "defaultSort": {"field": "unsafe_rate", "direction": "desc"},
                "columns": [
                    {"field": "condition", "label": "Condition", "type": "text"},
                    {"field": "unsafe_rate", "label": "Unsafe rate", "format": "percent"},
                    {"field": "mean_final_payoff", "label": "Mean payoff", "format": "number"},
                    {"field": "setback_rate", "label": "Setback rate", "format": "percent"},
                    {"field": "tie_rate", "label": "Tie rate", "format": "percent"},
                    {"field": "n_races", "label": "Races", "format": "number"},
                ],
            },
            {
                "id": "evidence_table",
                "title": "Pilot evidence ledger",
                "subtitle": "Completed pilots only; smoke runs are intentionally excluded from reported effects.",
                "dataset": "evidence",
                "sourceId": "synthesis",
                "defaultSort": {"field": "decisions_or_probes", "direction": "desc"},
                "columns": [
                    {"field": "study", "label": "Study", "type": "text"},
                    {"field": "scope", "label": "Scope", "type": "text"},
                    {"field": "races", "label": "Races", "format": "number"},
                    {"field": "decisions_or_probes", "label": "Decisions / probes", "format": "number"},
                    {"field": "temperature", "label": "Temperature", "format": "number"},
                    {"field": "parse_failures", "label": "Parse failures", "format": "number"},
                    {"field": "claim_class", "label": "Claim class", "type": "text"},
                ],
            },
        ],
        "sources": manifest_sources,
        "blocks": [
            {"id": "title", "type": "markdown", "body": "# AI Race Pilot Evidence"},
            {
                "id": "technical_summary",
                "type": "markdown",
                "body": "## Technical summary\n\n**The central result is a competence–stability gap.** Qwen2.5-7B can recall the rules and calculate the one-round payoff table, yet it is weak on state transitions and expected payoff, and its enacted strategy changes sharply under superficially different prompts. Across completed pilots there were no parse failures, so these shifts are behavioral rather than parser artifacts. They remain exploratory: one model, pilot-scale repetitions, and different decoding temperatures across protocols prevent a pooled causal estimate.",
            },
            {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["surface_span_card", "persona_span_card", "probe_card", "health_card"]},
            {
                "id": "surface_result",
                "type": "markdown",
                "sourceId": "surface",
                "body": "## Meaning-preserving surface changes moved Unsafe behavior by up to tens of percentage points\n\nThe canonical Unsafe rate was **52.2%**. Moving risk information next to the response increased it to **89.2%** (+37.1 pp), while reversing the displayed action order reduced it to **8.4%** (-43.7 pp). Lexical synonyms, dense formatting, uppercase emphasis, compact boundaries, and several order changes also produced large trajectory-level shifts. This is descriptive sensitivity under a fixed game mechanism, not evidence that typography alone has a stable causal effect across models.",
            },
            {"id": "surface_delta_block", "type": "chart", "chartId": "surface_delta_chart"},
            {
                "id": "surface_dynamics",
                "type": "markdown",
                "sourceId": "surface",
                "body": "### Small entry flips can coexist with large trajectory divergence\n\nSeveral variants changed only **1.7–3.3%** of paired first-round choices but shifted the full-trajectory Unsafe rate by roughly 20–44 pp. Read this as repeated prompt exposure plus endogenous game-state feedback: the first-round flip metric isolates only the entry decision, whereas every later decision sees both the altered prompt and a potentially diverged history. Emotional framing is the counterexample—it flipped **83.3%** of first-round choices, showing a direct framing effect rather than a subtle downstream divergence.",
            },
            {"id": "surface_scatter_block", "type": "chart", "chartId": "surface_scatter_chart"},
            {
                "id": "persona_result",
                "type": "markdown",
                "sourceId": "persona",
                "body": "## Persona assignment dominated the tested risk manipulation\n\nAcross risk levels, Unsafe rates ranged from **31.2%** for the risk-averse role to **76.9%** for adversarial-versus-adversarial, compared with **56.3%** without a persona. Within-persona risk contrasts were generally much smaller than these between-persona gaps in the pilot tables. The appropriate interpretation is that role text is part of the treatment, not harmless flavor text.",
            },
            {"id": "persona_chart_block", "type": "chart", "chartId": "persona_chart"},
            {
                "id": "role_result",
                "type": "markdown",
                "sourceId": "seats",
                "body": "### Asymmetric roles acted as an almost deterministic policy switch\n\nAfter combining the seat-counterbalanced S_AC and S_CA games, adversarial-role decisions were **92.8% Unsafe**, while cooperative-role decisions were **0% Unsafe**. The aggregate S_AC/S_CA bars therefore conceal two polarized agents rather than two moderate policies. This is the cleanest evidence that analysis should remain at the player-role level before aggregating to a race condition.",
            },
            {"id": "role_chart_block", "type": "chart", "chartId": "asymmetric_role_chart"},
            {
                "id": "understanding_result",
                "type": "markdown",
                "sourceId": "probes",
                "body": "## Rule recall was strong, but multi-step game reasoning was the bottleneck\n\nUnder the direct prompt, semantic accuracy was **100%** for rule recall and stage payoff, but **0%** for state transition and expected payoff. Paraphrasing preserved the easy domains yet did not repair multi-step reasoning. A disclosed calculator raised overall semantic accuracy from **52.1% unaided to 75.6%**, with the largest practical gains in reconstruction, transition, terminal scoring, and expected payoff; it still did not make those domains perfect.",
            },
            {"id": "probe_chart_block", "type": "chart", "chartId": "probe_chart"},
            {
                "id": "calculator_result",
                "type": "markdown",
                "sourceId": "behavior",
                "body": "## Better arithmetic did not translate into safer or higher-payoff play\n\nThe calculator decision card increased Unsafe behavior from **52.0% to 60.8%** (+8.8 pp). Mean final payoff was nearly unchanged and slightly lower (**42.77 to 42.21**), while race-cluster confidence intervals for Unsafe behavior overlap. The ablation therefore shows a mechanism warning: improving access to correct calculations can change policy direction without demonstrating a payoff benefit.",
            },
            {"id": "behavior_chart_block", "type": "chart", "chartId": "behavior_chart"},
            {
                "id": "behavior_table_note",
                "type": "markdown",
                "sourceId": "behavior",
                "body": "### Behavioral condition-level audit\n\nThe table reports the same 30-race paired outputs (canonical vs calculator) used in the chart and keeps raw n-race/decision denominators side by side for reproducibility checks.",
            },
            {"id": "behavior_table_block", "type": "table", "tableId": "behavior_table"},
            {
                "id": "scope",
                "type": "markdown",
                "body": "## Scope, data, and metric definitions\n\n**Unsafe rate** is the number of enacted Unsafe actions divided by parsed player decisions; it is decision-weighted, so longer races contribute more turns. **First-round flip rate** compares paired entry choices with the canonical prompt under common seeds. **Semantic accuracy** accepts meaning-equivalent answers even when strict output formatting fails. Surface results cover 18 variants × 3 risk levels × 10 repetitions; persona results cover 210 races; the understanding audit contains 685 probe outputs plus 60 behavioral races. Smoke runs are excluded from all effect summaries.",
            },
            {"id": "evidence_table_block", "type": "table", "tableId": "evidence_table"},
            {
                "id": "methodology",
                "type": "markdown",
                "body": "## Experimental design and validation\n\nAll experiments use the same pinned Qwen2.5-7B-Instruct F16 model digest and the canonical AI Race mechanism. Surface variants use paired seeds and a fixed stochastic horizon; persona analysis verifies the full payoff, progress, risk, stopping, setback, and terminal-state mechanics before admitting a race. This report additionally reconciles all displayed rates against raw counts, risk-stratified summaries against overall summaries, and manifest totals against CSV row totals. Cluster-bootstrap intervals are shown where the source analysis supplies them; no new inferential test is introduced here.",
            },
            {"id": "surface_table_intro", "type": "markdown", "body": "### Exact surface audit table\n\nThe table preserves every completed variant and denominator. Sort by the absolute change column to inspect the strongest deviations, then compare the entry flip column to distinguish immediate response changes from longer-run divergence."},
            {"id": "surface_table_block", "type": "table", "tableId": "surface_table"},
            {
                "id": "limitations",
                "type": "markdown",
                "body": "## Limitations, uncertainty, and robustness boundary\n\n- These are **exploratory pilots**, not confirmatory estimates.\n- Results cover one open model and one exact model digest; cross-model generality is unknown.\n- Surface and behavior pilots used temperature 0.7, persona used 1.0, and probes used 0.0; comparisons are kept within protocol and should not be pooled.\n- Decisions within a race are dependent. Decision-weighted rates describe enacted behavior but are not independent Bernoulli trials.\n- The first-round analysis isolates entry sensitivity only; later trajectory gaps mix continued prompt exposure with endogenous state feedback.\n- Calculator outputs demonstrate access to disclosed arithmetic, not an internal world model or causal understanding.",
            },
            {
                "id": "next_steps",
                "type": "markdown",
                "body": "## Recommended next experiments\n\n1. **Run a preregistered confirmatory grid** across at least three model families, fixed decoding settings, and more common-random-number repetitions.\n2. **Factor surface changes independently**: action-label order, response position, whitespace, lexical choice, and framing should be crossed rather than bundled.\n3. **Add comprehension admission gates** before gameplay: rule recall, state update, terminal scoring, and expected-payoff thresholds.\n4. **Separate direct from feedback effects** by replaying a fixed logged state sequence, then compare with the endogenous live trajectory.\n5. **Report player-role and race-level estimands together** so polarized asymmetric roles cannot disappear inside aggregate condition rates.",
            },
            {
                "id": "questions",
                "type": "markdown",
                "body": "## Further questions\n\n- Does action-order sensitivity persist when choices are emitted as randomized opaque IDs and decoded after the response?\n- Are the large surface effects stable at temperature 0 and across quantizations?\n- Which reasoning failure—state tracking, terminal payoff, or uncertainty integration—best predicts Unsafe play?\n- Does a verified external game-state tool improve realized payoff once its outputs are constrained to be behaviorally neutral?",
            },
        ],
    }

    return {
        "surface": "report",
        "manifest": manifest,
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headlines": headlines,
                "surface": surface,
                "surface_noncontrol": [row for row in surface if row["variant"] != "canonical"],
                "surface_ranked": sorted(surface, key=lambda row: row["absolute_delta"], reverse=True),
                "persona": persona,
                "asymmetric_roles": asymmetric_roles,
                "probes": probes,
                "behavior": behavior,
                "evidence": evidence,
            },
            "accessIssues": [],
        },
        "sources": canonical_sources,
        "package_info": {
            "originUrl": "artifact://ai-race-pilot-evidence",
            "controls": {"edit": False, "refresh": False},
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT / "docs" / "reports" / "pilot_insight_report" / "artifact.json",
    )
    args = parser.parse_args()
    artifact = build_artifact()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(artifact, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"status": "ok", "output": str(args.output), "datasets": {key: len(value) for key, value in artifact["snapshot"]["datasets"].items()}}, indent=2))


if __name__ == "__main__":
    main()
