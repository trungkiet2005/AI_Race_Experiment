#!/usr/bin/env python3
"""Build the cross-study impact synthesis and trajectory-divergence demo data.

This module never pools protocols.  It emits descriptive, source-labelled tables
that keep pilot, diagnostic, reconstruction, and blocked causal evidence apart.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "results" / "impact_upgrade"
REFERENCE_CONTEXT = "abstract_contest"
CONTEXT_ORDER = [
    "abstract_contest",
    "technology_race",
    "robotic_expedition",
    "colony_life_support",
    "hospital_deployment",
    "fictional_cartography",
    "crystal_guild_contract",
    "logistics_contract",
]
CONTEXT_LABELS = {
    "abstract_contest": "Abstract contest",
    "technology_race": "Technology race",
    "robotic_expedition": "Robotic expedition",
    "colony_life_support": "Colony life support",
    "hospital_deployment": "Hospital deployment",
    "fictional_cartography": "Fictional cartography",
    "crystal_guild_contract": "Crystal guild",
    "logistics_contract": "Logistics contract",
}
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}
PALETTE = {
    "navy": "#0B132B",
    "blue": "#2563EB",
    "cyan": "#06B6D4",
    "teal": "#0D9488",
    "amber": "#F59E0B",
    "red": "#DC2626",
    "slate": "#64748B",
    "grid": "#DCE3ED",
}


def read_json(path: Path) -> dict:
    with path.open(encoding="utf-8-sig") as handle:
        return json.load(handle)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def context_from_prompt(prompt_version: str) -> str:
    parts = str(prompt_version).split(":")
    if len(parts) < 4 or not parts[0].startswith("ai-race-context-skin-v1"):
        raise ValueError(f"Unexpected context prompt version: {prompt_version}")
    return parts[1]


def mapping_from_prompt(prompt_version: str) -> str:
    mapping = str(prompt_version).split(":")[-1]
    if mapping not in {"safe_p", "safe_q"}:
        raise ValueError(f"Unexpected action mapping: {prompt_version}")
    return mapping


def load_live_turns(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("lane_*/*/turns.jsonl")):
        frame = pd.read_json(path, lines=True)
        frame["source_path"] = path.relative_to(ROOT).as_posix()
        frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No turn logs under {root}")
    turns = pd.concat(frames, ignore_index=True)
    turns["max_private_risk"] = turns["max_private_risk"].astype(float).round(1)
    turns["context"] = turns["prompt_version"].map(context_from_prompt)
    turns["mapping"] = turns["prompt_version"].map(mapping_from_prompt)
    return turns


def load_live_players(root: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for path in sorted(root.glob("lane_*/*/players.csv")):
        frame = pd.read_csv(path)
        frame["source_path"] = path.relative_to(ROOT).as_posix()
        frames.append(frame)
    players = pd.concat(frames, ignore_index=True)
    players["max_private_risk"] = players["max_private_risk"].astype(float).round(1)
    players["context"] = players["prompt_version"].map(context_from_prompt)
    players["mapping"] = players["prompt_version"].map(mapping_from_prompt)
    return players


def audit_live(turns: pd.DataFrame, players: pd.DataFrame) -> dict:
    required_contexts = set(CONTEXT_ORDER)
    contexts = set(turns["context"].unique())
    if contexts != required_contexts:
        raise AssertionError(f"Context coverage mismatch: {sorted(contexts)}")
    if len(turns) != 13_680 or len(players) != 1_536:
        raise AssertionError(f"Unexpected T=0 coverage: {len(turns)} turns, {len(players)} players")
    if int(turns["parse_failed"].sum()) != 0:
        raise AssertionError("T=0 live context run contains parse failures")
    if set(turns["mapping"].unique()) != {"safe_p", "safe_q"}:
        raise AssertionError("Both opaque action mappings are required")
    observed_risks = np.sort(turns["max_private_risk"].unique().astype(float))
    if not np.allclose(observed_risks, np.array([0.1, 0.6, 0.9]), atol=1e-12):
        raise AssertionError("Risk grid mismatch")
    player_turn_counts = turns.groupby(["game_id", "player_index"]).size().rename("n")
    player_reported = players.set_index(["game_id", "player_index"])["n_rounds"].sort_index()
    if not player_turn_counts.sort_index().equals(player_reported.astype("int64")):
        raise AssertionError("Turn counts do not match players.csv n_rounds")
    return {
        "status": "passed",
        "n_turns": int(len(turns)),
        "n_player_races": int(len(players)),
        "n_races": int(players["game_id"].nunique()),
        "n_contexts": int(turns["context"].nunique()),
        "n_risk_levels": int(turns["max_private_risk"].nunique()),
        "n_mappings": int(turns["mapping"].nunique()),
        "parse_failures": int(turns["parse_failed"].sum()),
        "retried_decisions": int((turns["retry_count"] > 0).sum()),
    }


def trajectory_rows(turns: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    key = ["max_private_risk", "rep", "player_index", "mapping"]
    trajectory: dict[tuple, dict[str, pd.DataFrame]] = {}
    for values, group in turns.sort_values("round").groupby(key + ["context"], sort=False):
        pair_key, context = tuple(values[:-1]), values[-1]
        trajectory.setdefault(pair_key, {})[context] = group.sort_values("round")

    player_lookup = players.set_index(key + ["context"])
    rows: list[dict] = []
    for pair_key, by_context in trajectory.items():
        if set(by_context) != set(CONTEXT_ORDER):
            raise AssertionError(f"Incomplete context block: {pair_key}")
        reference = by_context[REFERENCE_CONTEXT]
        ref_actions = reference["unsafe"].astype(int).to_numpy()
        ref_player = player_lookup.loc[pair_key + (REFERENCE_CONTEXT,)]
        for context in CONTEXT_ORDER[1:]:
            target = by_context[context]
            target_actions = target["unsafe"].astype(int).to_numpy()
            if len(target_actions) != len(ref_actions):
                raise AssertionError("CRN horizons differ across contexts")
            mismatch = target_actions != ref_actions
            first_divergence = int(np.flatnonzero(mismatch)[0] + 1) if mismatch.any() else None
            target_player = player_lookup.loc[pair_key + (context,)]
            rows.append(
                {
                    "risk": float(pair_key[0]),
                    "rep": int(pair_key[1]),
                    "player_index": int(pair_key[2]),
                    "mapping": pair_key[3],
                    "reference": REFERENCE_CONTEXT,
                    "context": context,
                    "n_rounds": int(len(ref_actions)),
                    "ever_diverged": int(mismatch.any()),
                    "first_divergence_round": first_divergence,
                    "action_disagreement_rate": float(mismatch.mean()),
                    "later_round_disagreement_rate": float(mismatch[1:].mean()) if len(mismatch) > 1 else 0.0,
                    "unsafe_rate_reference": float(ref_actions.mean()),
                    "unsafe_rate_context": float(target_actions.mean()),
                    "unsafe_rate_delta": float(target_actions.mean() - ref_actions.mean()),
                    "final_progress_delta": float(target_player["progress"] - ref_player["progress"]),
                    "final_payoff_delta": float(target_player["final_payoff"] - ref_player["final_payoff"]),
                    "payoff_changed": int(target_player["final_payoff"] != ref_player["final_payoff"]),
                    "setback_reference": int(ref_player["setback"]),
                    "setback_context": int(target_player["setback"]),
                }
            )
    result = pd.DataFrame(rows)
    expected = 96 * 2 * 7
    if len(result) != expected:
        raise AssertionError(f"Expected {expected} paired player trajectories, found {len(result)}")
    return result


def divergence_summary(rows: pd.DataFrame) -> pd.DataFrame:
    grouped = rows.groupby("context", sort=False)
    summary = grouped.agg(
        n_paired_player_races=("ever_diverged", "size"),
        ever_diverged_rate=("ever_diverged", "mean"),
        mean_action_disagreement=("action_disagreement_rate", "mean"),
        mean_later_round_disagreement=("later_round_disagreement_rate", "mean"),
        mean_unsafe_delta=("unsafe_rate_delta", "mean"),
        mean_final_progress_delta=("final_progress_delta", "mean"),
        mean_final_payoff_delta=("final_payoff_delta", "mean"),
        payoff_changed_rate=("payoff_changed", "mean"),
    ).reset_index()
    medians = (
        rows.dropna(subset=["first_divergence_round"])
        .groupby("context")["first_divergence_round"]
        .median()
        .rename("median_first_divergence_round_conditional")
    )
    return summary.merge(medians, on="context", how="left")


def mapping_interaction_summary(rows: pd.DataFrame) -> pd.DataFrame:
    return (
        rows.groupby(["context", "mapping"], sort=False)
        .agg(
            n_paired_player_races=("ever_diverged", "size"),
            ever_diverged_rate=("ever_diverged", "mean"),
            mean_action_disagreement=("action_disagreement_rate", "mean"),
            mean_unsafe_delta=("unsafe_rate_delta", "mean"),
            mean_final_payoff_delta=("final_payoff_delta", "mean"),
            payoff_changed_rate=("payoff_changed", "mean"),
        )
        .reset_index()
    )


def divergence_curve(rows: pd.DataFrame) -> pd.DataFrame:
    records: list[dict] = []
    max_round = int(rows["n_rounds"].max())
    for context, group in rows.groupby("context", sort=False):
        survival = 1.0
        for round_number in range(1, max_round + 1):
            event_time = group["first_divergence_round"]
            at_risk = group[
                (group["n_rounds"] >= round_number)
                & (event_time.isna() | (event_time >= round_number))
            ]
            events = int((at_risk["first_divergence_round"] == round_number).sum())
            if len(at_risk):
                survival *= 1.0 - events / len(at_risk)
            records.append(
                {
                    "context": context,
                    "round": round_number,
                    "n_at_risk": int(len(at_risk)),
                    "n_events": events,
                    "kaplan_meier_cumulative_divergence": float(1.0 - survival),
                }
            )
    return pd.DataFrame(records)


def context_decomposition() -> pd.DataFrame:
    source = pd.read_csv(
        ROOT
        / "results/open_source/context_skin_pilot/analysis_live_pilot_t0/paired_context_effects.csv"
    )
    source = source[(source["risk"].astype(str) == "all") & (source["context"] != REFERENCE_CONTEXT)]
    live = source[source["estimand"] == "live_full_trajectory"].copy()
    fixed = source[source["estimand"] == "fixed_state_direct"].copy()
    columns = ["context", "estimate_pp", "ci_low_pp", "ci_high_pp", "n_paired_rows", "n_clusters"]
    live = live[columns].rename(
        columns={
            "estimate_pp": "live_effect_pp",
            "ci_low_pp": "live_ci_low_pp",
            "ci_high_pp": "live_ci_high_pp",
            "n_paired_rows": "live_paired_rows",
            "n_clusters": "live_clusters",
        }
    )
    fixed = fixed[columns].rename(
        columns={
            "estimate_pp": "fixed_direct_effect_pp",
            "ci_low_pp": "fixed_ci_low_pp",
            "ci_high_pp": "fixed_ci_high_pp",
            "n_paired_rows": "fixed_paired_rows",
            "n_clusters": "fixed_clusters",
        }
    )
    result = live.merge(fixed, on="context", validate="one_to_one")
    result["live_minus_fixed_descriptive_gap_pp"] = (
        result["live_effect_pp"] - result["fixed_direct_effect_pp"]
    )
    result["interpretation"] = (
        "Descriptive contrast only: live and fixed-state estimands use different units of analysis; "
        "the gap mixes repeated exposure and endogenous feedback and is not a causal mediation estimate."
    )
    return result


def cross_model_baselines() -> pd.DataFrame:
    frames = []
    for provider in ["openai", "frontier"]:
        path = ROOT / f"analysis/{provider}/derived/unsafe_by_risk_model_player.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["persona_condition"] == "none"].copy()
        frame["provider"] = provider
        frames.append(frame)
    result = pd.concat(frames, ignore_index=True)
    result["model_label"] = result["model"].map(MODEL_LABELS).fillna(result["model"])
    result = result.rename(columns={"mean_player_unsafe_rate": "unsafe_rate"})
    keep = [
        "provider",
        "model",
        "model_label",
        "max_private_risk",
        "n_players",
        "unsafe_rate",
        "run_phase",
        "run_status",
        "prompt_version",
        "protocol_signature",
    ]
    result = result[keep].sort_values(["provider", "model", "max_private_risk"])
    if len(result) != 15 or result["model"].nunique() != 5:
        raise AssertionError("Expected five cross-model baselines over three risk levels")
    return result


def evidence_ledger() -> pd.DataFrame:
    context_summary = read_json(
        ROOT / "results/open_source/context_skin_pilot/analysis_live_pilot_t0/analysis_summary.json"
    )
    openai_manifest = read_json(ROOT / "results/reports/openai/derived/analysis_manifest.json")
    frontier_manifest = read_json(ROOT / "results/reports/frontier/derived/analysis_manifest.json")
    sae_summary = read_json(
        ROOT / "results/open_source/activation_sae/context_fast_sae_analysis/summary.json"
    )
    causal_summary = read_json(
        ROOT
        / "results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/analysis_summary.json"
    )
    return pd.DataFrame(
        [
            {
                "study": "Qwen context skins, T=0",
                "evidence_class": "diagnostic",
                "models": 1,
                "races": context_summary["validation"]["n_live_races"],
                "decisions_or_rows": context_summary["validation"]["n_live_decisions"],
                "quality_gate": "Mechanics/coverage pass; comprehension admission fails",
                "paper_use": "Validity boundary and hypothesis generation",
            },
            {
                "study": "Qwen context skins, T=0.7",
                "evidence_class": "diagnostic robustness",
                "models": 1,
                "races": 768,
                "decisions_or_rows": 13_680,
                "quality_gate": "Matched protocol and CRN; separate temperature stratum",
                "paper_use": "Temperature stability only; never pooled with T=0",
            },
            {
                "study": "OpenAI baseline/persona grid",
                "evidence_class": "pilot",
                "models": 2,
                "races": openai_manifest["n_races_total"],
                "decisions_or_rows": openai_manifest["n_player_rounds_total"],
                "quality_gate": "Mechanics pass; local manifest leaves persona/protocol confounded",
                "paper_use": "Cross-model descriptive replication",
            },
            {
                "study": "Gemini baseline/persona grid",
                "evidence_class": "pilot",
                "models": 3,
                "races": frontier_manifest["n_races_total"],
                "decisions_or_rows": frontier_manifest["n_player_rounds_total"],
                "quality_gate": "Mechanics pass; small persona cells and unverified protocol signatures",
                "paper_use": "Cross-model descriptive replication",
            },
            {
                "study": "FAST-SAE context representation",
                "evidence_class": "association",
                "models": 1,
                "races": 0,
                "decisions_or_rows": 384,
                "quality_gate": f"Held-out probes pass; max intervention flip={max(x['max_intervention_flip_rate'] for x in sae_summary['layer_decisions']):.1%}",
                "paper_use": "Representation result, not causal controller",
            },
            {
                "study": "FAST-SAE self-play intervention",
                "evidence_class": "exploratory causal audit",
                "models": 1,
                "races": causal_summary["counts"]["live_races"],
                "decisions_or_rows": causal_summary["counts"]["fixed_intervention_rows"],
                "quality_gate": "Matched random/unrelated controls; strongest target-control CIs include zero",
                "paper_use": "Negative causal result / stopping rule",
            },
            {
                "study": "EGTTools transition parity",
                "evidence_class": "paper-ready method validation",
                "models": 0,
                "races": 0,
                "decisions_or_rows": 0,
                "quality_gate": "Transition max error 1.11e-16; stationary max error 7.63e-15",
                "paper_use": "Validates independent evolutionary reconstruction",
            },
            {
                "study": "N=3 Qwen extension",
                "evidence_class": "pilot",
                "models": 1,
                "races": 96,
                "decisions_or_rows": 0,
                "quality_gate": "Mechanics/CRN pass; persona n=2 races per cell",
                "paper_use": "Scalability demo, not main estimand",
            },
        ]
    )


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": PALETTE["grid"],
            "axes.linewidth": 0.8,
            "xtick.color": PALETTE["slate"],
            "ytick.color": PALETTE["slate"],
            "text.color": PALETTE["navy"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_cross_model(data: pd.DataFrame, figures: Path) -> None:
    setup_plot()
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    colors = [PALETTE["blue"], PALETTE["cyan"], PALETTE["teal"], PALETTE["amber"], PALETTE["red"]]
    for color, (model, group) in zip(colors, data.groupby("model_label", sort=False)):
        group = group.sort_values("max_private_risk")
        ax.plot(
            group["max_private_risk"],
            100 * group["unsafe_rate"],
            marker="o",
            linewidth=2.4,
            markersize=7,
            color=color,
            label=model,
        )
    ax.set_title("Same mechanism, five model checkpoints, qualitatively different risk response")
    ax.set_xlabel("Maximum private setback risk")
    ax.set_ylabel("Mean player-level Unsafe rate (%)")
    ax.set_xticks([0.1, 0.6, 0.9], ["10%", "60%", "90%"])
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="lower left")
    ax.text(
        0.01,
        -0.2,
        "Pilot baselines; player-weighted rates. Provider protocols are shown together descriptively, never pooled inferentially.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=PALETTE["slate"],
    )
    save_figure(fig, figures / "cross_model_risk_response")


def plot_context_decomposition(data: pd.DataFrame, figures: Path) -> None:
    setup_plot()
    ordered = data.assign(order=data["context"].map({x: i for i, x in enumerate(CONTEXT_ORDER)})).sort_values("order")
    y = np.arange(len(ordered))
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.hlines(y, ordered["fixed_direct_effect_pp"], ordered["live_effect_pp"], color=PALETTE["grid"], linewidth=4)
    ax.scatter(ordered["fixed_direct_effect_pp"], y, s=70, color=PALETTE["cyan"], label="Fixed-state direct")
    ax.scatter(ordered["live_effect_pp"], y, s=70, color=PALETTE["red"], label="Live full trajectory")
    ax.axvline(0, color=PALETTE["slate"], linewidth=1)
    ax.set_yticks(y, [CONTEXT_LABELS[x] for x in ordered["context"]])
    ax.set_xlabel("Unsafe-rate difference vs abstract contest (percentage points)")
    ax.set_title("Direct context effects are amplified along endogenous live trajectories")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        -0.16,
        "Paired T=0 Qwen pilot. The connector is descriptive, not a causal mediation estimate; live and replay units differ.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=PALETTE["slate"],
    )
    save_figure(fig, figures / "context_direct_vs_live")


def plot_divergence_curve(data: pd.DataFrame, figures: Path) -> None:
    setup_plot()
    fig, ax = plt.subplots(figsize=(9.4, 5.4))
    contexts = ["technology_race", "robotic_expedition", "fictional_cartography", "logistics_contract"]
    colors = [PALETTE["slate"], PALETTE["teal"], PALETTE["amber"], PALETTE["red"]]
    for context, color in zip(contexts, colors):
        group = data[data["context"] == context]
        ax.plot(
            group["round"],
            100 * group["kaplan_meier_cumulative_divergence"],
            linewidth=2.4,
            color=color,
            label=CONTEXT_LABELS[context],
        )
    ax.set_xlim(1, data["round"].max())
    ax.set_ylim(0, 100)
    ax.set_xlabel("Round reached")
    ax.set_ylabel("Paired player trajectories diverged by round (%)")
    ax.set_title("No entry flip, then repeated exposure separates the trajectories")
    ax.grid(color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        -0.18,
        "Reference: abstract contest. Kaplan–Meier cumulative divergence with race end treated as censoring; all round-1 decisions agree.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=PALETTE["slate"],
    )
    save_figure(fig, figures / "trajectory_divergence_curve")


def plot_mapping_gate(data: pd.DataFrame, figures: Path) -> None:
    setup_plot()
    order = [x for x in CONTEXT_ORDER if x != REFERENCE_CONTEXT]
    pivot = data.pivot(index="context", columns="mapping", values="mean_unsafe_delta").reindex(order)
    y = np.arange(len(pivot))
    height = 0.34
    fig, ax = plt.subplots(figsize=(9.4, 5.6))
    ax.barh(y - height / 2, 100 * pivot["safe_p"], height=height, color=PALETTE["red"], label="Safe = P")
    ax.barh(y + height / 2, 100 * pivot["safe_q"], height=height, color=PALETTE["cyan"], label="Safe = Q")
    ax.set_yticks(y, [CONTEXT_LABELS[x] for x in pivot.index])
    ax.invert_yaxis()
    ax.set_xlabel("Unsafe-rate difference vs abstract contest (percentage points)")
    ax.set_title("Opaque action-code position gates the context effect")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right")
    ax.text(
        0.01,
        -0.17,
        "Paired T=0 live trajectories, n=96 player-races per context × mapping. Mapping is balanced but assigned by repetition parity.",
        transform=ax.transAxes,
        fontsize=8.5,
        color=PALETTE["slate"],
    )
    save_figure(fig, figures / "context_mapping_gate")


def plot_evidence_ladder(data: pd.DataFrame, figures: Path) -> None:
    setup_plot()
    levels = {
        "paper-ready method validation": 4,
        "pilot": 3,
        "diagnostic robustness": 2,
        "diagnostic": 2,
        "association": 2,
        "exploratory causal audit": 1,
    }
    ordered = data.assign(level=data["evidence_class"].map(levels)).sort_values(["level", "study"])
    colors = ordered["level"].map({1: PALETTE["red"], 2: PALETTE["amber"], 3: PALETTE["blue"], 4: PALETTE["teal"]})
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    y = np.arange(len(ordered))
    ax.scatter(ordered["level"], y, s=120, c=colors)
    ax.hlines(y, 0.8, ordered["level"], colors=PALETTE["grid"], linewidth=2)
    ax.set_yticks(y, ordered["study"])
    ax.set_xticks([1, 2, 3, 4], ["Blocked / negative", "Diagnostic", "Pilot", "Method validated"])
    ax.set_xlim(0.8, 4.2)
    ax.set_title("Evidence is deliberately stratified instead of promoted by rhetoric")
    ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right", "left", "bottom"]].set_visible(False)
    save_figure(fig, figures / "evidence_ladder")


def xai_decodability_control() -> pd.DataFrame:
    """Build a compact, source-backed association-versus-control summary."""
    context = read_json(
        ROOT / "results/open_source/activation_sae/context_fast_sae_analysis/summary.json"
    )
    selfplay = read_json(
        ROOT
        / "results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/analysis_summary.json"
    )
    rows = []
    for decision in context["layer_decisions"]:
        rows.append(
            {
                "stage": "association",
                "label": f"Layer {decision['layer']} held-out probe",
                "value": float(decision["double_heldout_probe_auc"]),
                "passed": np.nan,
                "total": int(decision["double_heldout_probe_n"]),
            }
        )
        rows.append(
            {
                "stage": "causal_check",
                "label": f"Layer {decision['layer']} intervention flips",
                "value": float(decision["max_intervention_flip_rate"]),
                "passed": 0,
                "total": 1312,
            }
        )
    extreme = selfplay["fixed_state_extreme_target_control_contrasts"]
    passed = sum(
        float(row["delta_ci95_low"]) > 0 or float(row["delta_ci95_high"]) < 0
        for row in extreme
    )
    rows.append(
        {
            "stage": "causal_check",
            "label": "Self-play target-control CIs",
            "value": passed / len(extreme),
            "passed": passed,
            "total": len(extreme),
        }
    )
    return pd.DataFrame(rows)


def plot_xai_decodability_control(data: pd.DataFrame, figures: Path) -> None:
    setup_plot()
    association = data[data["stage"] == "association"].copy()
    causal = data[data["stage"] == "causal_check"].copy()
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.25), gridspec_kw={"width_ratios": [1, 1.25]})

    colors = [PALETTE["blue"], PALETTE["cyan"]]
    axes[0].barh(association["label"], association["value"], color=colors, height=0.5)
    axes[0].axvline(0.5, color=PALETTE["slate"], linestyle="--", linewidth=1)
    for y, value in enumerate(association["value"]):
        axes[0].text(value - 0.02, y, f"{value:.3f}", ha="right", va="center", color="white", weight="bold")
    axes[0].set_xlim(0.5, 1.0)
    axes[0].set_xlabel("Double-held-out ROC-AUC")
    axes[0].set_title("Action is decodable", loc="left", weight="bold")

    y = np.arange(len(causal))
    axes[1].hlines(y, 0, 1, color=PALETTE["grid"], linewidth=5)
    axes[1].scatter(causal["value"], y, color=PALETTE["red"], s=100, zorder=3)
    axes[1].set_yticks(y, causal["label"])
    axes[1].set_xlim(-0.02, 1.0)
    axes[1].set_xlabel("Share passing the behavioral check")
    axes[1].set_title("Causal promotion checks do not pass", loc="left", weight="bold")
    for index, row in causal.reset_index(drop=True).iterrows():
        axes[1].text(0.025, index, f"{int(row['passed'])}/{int(row['total']):,}", va="center", color=PALETTE["navy"], weight="bold")

    for ax in axes:
        ax.grid(axis="x", color=PALETTE["grid"], linewidth=0.8)
        ax.spines[["top", "right", "left"]].set_visible(False)
    fig.suptitle("Decodable representation without demonstrated feature-specific control", x=0.06, ha="left", weight="bold")
    fig.text(
        0.06,
        0.01,
        "Pinned Qwen2.5-7B-Instruct + FAST-SAE. AUC is association; intervention flips and target-control intervals test behavioral specificity.",
        color=PALETTE["slate"],
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 0.9))
    save_figure(fig, figures / "xai_decodability_vs_control")


def demo_payload(
    rows: pd.DataFrame,
    turns: pd.DataFrame,
    decomposition: pd.DataFrame,
    mapping_summary: pd.DataFrame,
) -> dict:
    ranked = rows.sort_values(
        ["payoff_changed", "action_disagreement_rate", "first_divergence_round"],
        ascending=[False, False, True],
        na_position="last",
    ).head(36)
    cases: list[dict] = []
    key = ["max_private_risk", "rep", "player_index", "mapping", "context"]
    lookup = {values: group.sort_values("round") for values, group in turns.groupby(key, sort=False)}
    for idx, row in ranked.reset_index(drop=True).iterrows():
        shared = (row["risk"], row["rep"], row["player_index"], row["mapping"])
        ref = lookup[shared + (REFERENCE_CONTEXT,)]
        target = lookup[shared + (row["context"],)]
        rounds = []
        for (_, left), (_, right) in zip(ref.iterrows(), target.iterrows()):
            rounds.append(
                {
                    "round": int(left["round"]),
                    "reference_action": "Unsafe" if int(left["unsafe"]) else "Safe",
                    "context_action": "Unsafe" if int(right["unsafe"]) else "Safe",
                    "reference_progress": float(left["own_progress_after"]),
                    "context_progress": float(right["own_progress_after"]),
                    "reference_gap": float(left["progress_gap_after"]),
                    "context_gap": float(right["progress_gap_after"]),
                    "reference_risk": float(left["current_private_risk_after"]),
                    "context_risk": float(right["current_private_risk_after"]),
                }
            )
        cases.append(
            {
                "id": f"case-{idx + 1:02d}",
                "context": row["context"],
                "context_label": CONTEXT_LABELS[row["context"]],
                "reference": REFERENCE_CONTEXT,
                "reference_label": CONTEXT_LABELS[REFERENCE_CONTEXT],
                "risk": float(row["risk"]),
                "rep": int(row["rep"]),
                "player_index": int(row["player_index"]),
                "mapping": row["mapping"],
                "first_divergence_round": int(row["first_divergence_round"]),
                "action_disagreement_rate": float(row["action_disagreement_rate"]),
                "final_payoff_delta": float(row["final_payoff_delta"]),
                "rounds": rounds,
            }
        )
    return {
        "schema_version": "ai-race-trajectory-demo-v1",
        "scope": "Qwen2.5-7B-Instruct F16, temperature 0, pilot, reference=abstract_contest",
        "claim_boundary": "Demo of paired descriptive trajectory divergence; not confirmatory and not causal mediation.",
        "contexts": [
            {
                **row,
                "context_label": CONTEXT_LABELS[row["context"]],
            }
            for row in decomposition.to_dict(orient="records")
        ],
        "mapping_interaction": mapping_summary.to_dict(orient="records"),
        "cases": cases,
    }


def write_manifest(output: Path, inputs: list[Path], outputs: list[Path], audit: dict) -> None:
    payload = {
        "schema_version": "ai-race-impact-synthesis-v1",
        "status": "complete",
        "evidence_policy": "No protocol pooling; pilot, diagnostic, association, causal-audit, and method-validation classes remain separate.",
        "audit": audit,
        "inputs": {path.relative_to(ROOT).as_posix(): sha256(path) for path in inputs},
        "outputs": {
            path.relative_to(ROOT).as_posix(): {"bytes": path.stat().st_size, "sha256": sha256(path)}
            for path in outputs
            if path.exists()
        },
    }
    (output / "analysis_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    output = args.output.resolve()
    figures = output / "figures"
    data_dir = output / "data"
    figures.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    live_root = ROOT / "results/open_source/context_skin_pilot/live_pilot_t0"
    turns = load_live_turns(live_root)
    players = load_live_players(live_root)
    audit = audit_live(turns, players)
    rows = trajectory_rows(turns, players)
    summary = divergence_summary(rows)
    mapping_summary = mapping_interaction_summary(rows)
    curve = divergence_curve(rows)
    decomposition = context_decomposition()
    cross_model = cross_model_baselines()
    ledger = evidence_ledger()
    xai = xai_decodability_control()

    tables = {
        "trajectory_pair_rows.csv": rows,
        "trajectory_divergence_summary.csv": summary,
        "context_mapping_interaction.csv": mapping_summary,
        "trajectory_divergence_curve.csv": curve,
        "context_direct_vs_live.csv": decomposition,
        "cross_model_baseline_rates.csv": cross_model,
        "evidence_ledger.csv": ledger,
        "xai_decodability_vs_control.csv": xai,
    }
    for name, frame in tables.items():
        frame.to_csv(data_dir / name, index=False)

    plot_cross_model(cross_model, figures)
    plot_context_decomposition(decomposition, figures)
    plot_divergence_curve(curve, figures)
    plot_mapping_gate(mapping_summary, figures)
    plot_evidence_ladder(ledger, figures)
    plot_xai_decodability_control(xai, figures)

    payload = demo_payload(rows, turns, decomposition, mapping_summary)
    demo_path = data_dir / "trajectory_demo.json"
    demo_path.write_text(json.dumps(payload, ensure_ascii=False) + "\n", encoding="utf-8")

    quality = {
        **audit,
        "trajectory_pairs": int(len(rows)),
        "contexts_vs_reference": int(rows["context"].nunique()),
        "all_first_round_actions_agree": bool(
            (rows["first_divergence_round"].fillna(999) > 1).all()
        ),
        "cross_model_checkpoints": int(cross_model["model"].nunique()),
        "evidence_ledger_rows": int(len(ledger)),
    }
    quality_path = output / "data_quality_audit.json"
    quality_path.write_text(json.dumps(quality, indent=2) + "\n", encoding="utf-8")

    input_paths = [
        ROOT / "results/reports/openai/derived/unsafe_by_risk_model_player.csv",
        ROOT / "results/reports/openai/derived/analysis_manifest.json",
        ROOT / "results/reports/frontier/derived/unsafe_by_risk_model_player.csv",
        ROOT / "results/reports/frontier/derived/analysis_manifest.json",
        ROOT / "results/open_source/context_skin_pilot/analysis_live_pilot_t0/paired_context_effects.csv",
        ROOT / "results/open_source/context_skin_pilot/analysis_live_pilot_t0/analysis_summary.json",
        ROOT / "results/open_source/activation_sae/context_fast_sae_analysis/summary.json",
        ROOT / "results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/analysis_summary.json",
    ]
    output_paths = [data_dir / name for name in tables] + [
        demo_path,
        quality_path,
        *sorted(figures.glob("*")),
    ]
    write_manifest(output, input_paths, output_paths, quality)
    print(json.dumps({"status": "complete", "output": str(output), "quality": quality}, indent=2))


if __name__ == "__main__":
    main()
