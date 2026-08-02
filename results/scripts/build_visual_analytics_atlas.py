#!/usr/bin/env python3
"""Build the publication and demo visual atlas from admitted impact tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
IMPACT = ROOT / "results" / "impact_upgrade"
DATA = IMPACT / "data"
OUT = IMPACT / "visual_atlas"
CONTEXT = {
    "technology_race": "Technology",
    "robotic_expedition": "Robotic expedition",
    "colony_life_support": "Life support",
    "hospital_deployment": "Hospital",
    "fictional_cartography": "Cartography",
    "crystal_guild_contract": "Crystal guild",
    "logistics_contract": "Logistics",
}
NAVY, BLUE, CYAN, RED, AMBER, SLATE = "#0B132B", "#2563EB", "#06B6D4", "#DC2626", "#F59E0B", "#64748B"


def style() -> None:
    sns.set_theme(style="whitegrid", context="talk")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "text.color": NAVY,
            "axes.labelcolor": NAVY,
            "axes.titlecolor": NAVY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save(fig: plt.Figure, stem: str, dpi: int = 220) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=dpi, bbox_inches="tight", facecolor="white")
    fig.savefig(OUT / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def model_risk_heatmap(cross: pd.DataFrame) -> None:
    matrix = cross.pivot(index="model_label", columns="max_private_risk", values="unsafe_rate") * 100
    matrix = matrix.reindex(sorted(matrix.index, key=lambda x: matrix.loc[x, 0.1]))
    fig, ax = plt.subplots(figsize=(8.8, 4.8))
    sns.heatmap(
        matrix,
        annot=True,
        fmt=".1f",
        cmap=sns.color_palette("rocket_r", as_cmap=True),
        vmin=0,
        vmax=100,
        linewidths=1,
        cbar_kws={"label": "Player-level Unsafe rate (%)"},
        ax=ax,
    )
    ax.set_title("The same risk caps produce qualitatively different checkpoint policies")
    ax.set_xlabel("Maximum private setback risk")
    ax.set_ylabel("")
    ax.set_xticklabels([f"{100*x:.0f}%" for x in matrix.columns])
    fig.text(0.01, -0.02, "Pilot checkpoint rates; provider protocols are displayed together descriptively and are not pooled.", fontsize=9, color=SLATE)
    save(fig, "model_risk_heatmap")


def context_effect_forest(direct: pd.DataFrame) -> None:
    direct = direct.copy().sort_values("live_effect_pp")
    y = np.arange(len(direct))
    fig, ax = plt.subplots(figsize=(9.4, 5.8))
    ax.errorbar(
        direct["fixed_direct_effect_pp"], y - 0.12,
        xerr=[direct["fixed_direct_effect_pp"] - direct["fixed_ci_low_pp"], direct["fixed_ci_high_pp"] - direct["fixed_direct_effect_pp"]],
        fmt="o", color=CYAN, capsize=3, label="Fixed-state direct", zorder=3,
    )
    ax.errorbar(
        direct["live_effect_pp"], y + 0.12,
        xerr=[direct["live_effect_pp"] - direct["live_ci_low_pp"], direct["live_ci_high_pp"] - direct["live_effect_pp"]],
        fmt="o", color=RED, capsize=3, label="Live trajectory", zorder=3,
    )
    ax.axvline(0, color=SLATE, linewidth=1)
    ax.set_yticks(y, [CONTEXT[x] for x in direct["context"]])
    ax.set_xlabel("Unsafe-rate difference vs technology framing (percentage points)")
    ax.set_title("Context effects survive dependency-aware intervals and grow in live play")
    ax.legend(frameon=False, loc="lower right", fontsize=10, markerscale=0.75)
    sns.despine(ax=ax, left=True)
    fig.text(0.01, -0.02, "Live intervals cluster 32 independent CRN repetition streams; replay intervals use 96 state clusters. Estimands are not pooled.", fontsize=9, color=SLATE)
    save(fig, "context_effect_forest")


def divergence_heatmap(curve: pd.DataFrame) -> None:
    matrix = curve.pivot(index="context", columns="round", values="kaplan_meier_cumulative_divergence") * 100
    matrix = matrix.reindex([x for x in CONTEXT if x in matrix.index])
    fig, ax = plt.subplots(figsize=(10.6, 5.5))
    sns.heatmap(matrix, cmap=sns.light_palette(RED, as_cmap=True), vmin=0, vmax=100, cbar_kws={"label": "Diverged by round (%)"}, ax=ax)
    ax.set_yticklabels([CONTEXT[x] for x in matrix.index], rotation=0)
    ax.set_xlabel("Round")
    ax.set_ylabel("")
    ax.set_title("Round-1 agreement hides rapid trajectory separation")
    fig.text(0.01, -0.02, "Kaplan–Meier cumulative divergence; race termination is treated as censoring. Every paired round-1 action agrees.", fontsize=9, color=SLATE)
    save(fig, "trajectory_divergence_heatmap")


def executive_atlas(
    cross: pd.DataFrame,
    mapping: pd.DataFrame,
    direct: pd.DataFrame,
    tradeoff: pd.DataFrame,
    curve: pd.DataFrame,
    comprehension: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15.5, 16.5), constrained_layout=True)
    ax = axes[0, 0]
    sns.lineplot(data=cross, x="max_private_risk", y="unsafe_rate", hue="model_label", marker="o", linewidth=2.2, palette="tab10", ax=ax)
    ax.set(xlabel="Maximum private risk", ylabel="Unsafe rate", title="A. Checkpoint × risk response")
    ax.set_xticks([0.1, 0.6, 0.9], ["10%", "60%", "90%"])
    ax.set_ylim(0, 1.05)
    ax.legend(title="", fontsize=7, frameon=False)

    ax = axes[0, 1]
    mp = mapping[mapping.context != "technology_race"].pivot(index="context", columns="mapping", values="mean_unsafe_delta") * 100
    mp = mp.reindex(CONTEXT.keys()).dropna(how="all")
    sns.heatmap(mp, annot=True, fmt=".1f", cmap="vlag", center=0, cbar_kws={"label": "Unsafe Δ (pp)"}, ax=ax)
    ax.set_yticklabels([CONTEXT[x] for x in mp.index], rotation=0)
    ax.set(xlabel="Opaque mapping", ylabel="", title="B. Mapping gates context sensitivity")

    ax = axes[1, 0]
    ordered = direct.sort_values("live_effect_pp")
    y = np.arange(len(ordered))
    ax.hlines(y, ordered.fixed_direct_effect_pp, ordered.live_effect_pp, color="#CBD5E1", linewidth=4)
    ax.scatter(ordered.fixed_direct_effect_pp, y, color=CYAN, s=65, label="Fixed")
    ax.scatter(ordered.live_effect_pp, y, color=RED, s=65, label="Live")
    ax.set_yticks(y, [CONTEXT[x] for x in ordered.context])
    ax.set(xlabel="Unsafe difference (pp)", title="C. Direct response versus live feedback")
    ax.legend(frameon=False)

    ax = axes[1, 1]
    sns.regplot(data=tradeoff, x="unsafe_delta_pp", y="payoff_delta", ci=None, scatter_kws={"s": 70, "color": RED}, line_kws={"color": SLATE, "linestyle": "--"}, ax=ax)
    for row in tradeoff.itertuples(index=False):
        ax.annotate(CONTEXT[row.context], (row.unsafe_delta_pp, row.payoff_delta), xytext=(4, 4), textcoords="offset points", fontsize=7)
    ax.axhline(0, color=SLATE, linewidth=1)
    ax.set(xlabel="Unsafe difference (pp)", ylabel="Final-payoff difference", title="D. More Unsafe, lower realized payoff")

    ax = axes[2, 0]
    selected = curve[curve.context.isin(["technology_race", "robotic_expedition", "fictional_cartography", "logistics_contract"])]
    sns.lineplot(data=selected, x="round", y="kaplan_meier_cumulative_divergence", hue="context", linewidth=2.2, palette=[SLATE, BLUE, AMBER, RED], ax=ax)
    ax.set(xlabel="Round", ylabel="Diverged by round", title="E. Sensitivity accumulates after entry")
    ax.set_ylim(0, 1.0)
    labels = [CONTEXT.get(x, x) for x in selected.context.drop_duplicates()]
    handles, _ = ax.get_legend_handles_labels()
    ax.legend(handles, labels, title="", fontsize=7, frameon=False)

    ax = axes[2, 1]
    comp = comprehension.sort_values("semantic_accuracy")
    sns.barplot(data=comp, y="domain", x="semantic_accuracy", color=BLUE, ax=ax)
    ax.axvline(0.8, color=RED, linestyle="--", linewidth=1.5, label="80% admission threshold")
    for patch, value in zip(ax.patches, comp.semantic_accuracy):
        ax.text(value + 0.02, patch.get_y() + patch.get_height() / 2, f"{100*value:.1f}%", va="center", fontsize=8)
    ax.set(xlabel="Semantic accuracy", ylabel="", title="F. Comprehension is the validity bottleneck", xlim=(0, 1.08))
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle("AI Race visual analytics atlas — behavior, mechanism, and validity boundary", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(0.5, -0.012, "Exploratory pilots unless explicitly labelled method validation. Rates use player/race-aware denominators; protocols and temperatures are never pooled.", ha="center", fontsize=10, color=SLATE)
    save(fig, "executive_visual_atlas")


def fixed_vs_live_explainer() -> None:
    fig, ax = plt.subplots(figsize=(12.5, 5.4))
    ax.set_xlim(0, 12.5); ax.set_ylim(0, 5.4); ax.axis("off")
    ax.text(0.3, 4.9, "Fixed-state replay", fontsize=18, fontweight="bold", color=NAVY)
    ax.text(6.7, 4.9, "Live trajectory", fontsize=18, fontweight="bold", color=NAVY)
    def box(x, y, w, text, color):
        ax.add_patch(FancyBboxPatch((x, y), w, 1.05, boxstyle="round,pad=.16", fc="white", ec=color, lw=2))
        ax.text(x+w/2, y+.53, text, ha="center", va="center", fontsize=10, color=NAVY)
    box(.3, 2.5, 1.7, "Same frozen\nstate S", BLUE); box(2.5, 3.15, 1.7, "Context A", CYAN); box(2.5, 1.85, 1.7, "Context B", AMBER)
    box(4.7, 3.15, 1.25, "Action A", CYAN); box(4.7, 1.85, 1.25, "Action B", AMBER)
    for y in (3.67,2.37): ax.add_patch(FancyArrowPatch((2.0,3.02),(2.48,y),arrowstyle="->",mutation_scale=14,color=SLATE))
    ax.add_patch(FancyArrowPatch((4.2,3.67),(4.68,3.67),arrowstyle="->",mutation_scale=14,color=SLATE)); ax.add_patch(FancyArrowPatch((4.2,2.37),(4.68,2.37),arrowstyle="->",mutation_scale=14,color=SLATE))
    ax.text(.4,.65,"Direct prompt response",fontsize=13,fontweight="bold",color=BLUE); ax.text(.4,.25,"State, history, risk and opponents are held constant.",fontsize=10,color=SLATE)
    xs=[6.7,8.15,9.6,11.05]; labels=["Prompt A/B\nat S₁","Action\nchanges","New state\nS₂ differs","Later actions\nand payoff"]
    for x,t in zip(xs,labels): box(x,2.5,1.05,t,RED if x>8 else BLUE)
    for x in xs[:-1]: ax.add_patch(FancyArrowPatch((x+1.05,3.02),(x+1.42,3.02),arrowstyle="->",mutation_scale=14,color=SLATE))
    ax.text(6.8,.65,"Total live effect",fontsize=13,fontweight="bold",color=RED); ax.text(6.8,.25,"Direct response + repeated exposure + endogenous feedback; not a mediation estimate.",fontsize=10,color=SLATE)
    ax.set_title("Why fixed-state and live estimates answer different causal questions", fontsize=22, fontweight="bold", color=NAVY, pad=12)
    save(fig, "fixed_vs_live_explainer")


def extended_evidence_atlas() -> None:
    temp = pd.read_csv(ROOT / "results/open_source/context_skin_pilot/analysis_temperature_robustness/tables/context_effect_stability.csv")
    agreement = pd.read_csv(ROOT / "results/open_source/context_skin_pilot/analysis_temperature_robustness/tables/trajectory_agreement_by_context.csv")
    position = pd.read_csv(ROOT / "results/open_source/position_endowment_greennode_e3cf825/analysis/primary_position_rates.csv")
    identity = pd.read_csv(ROOT / "results/open_source/heterogeneous_dyad_greennode_ba2906a/analysis/data/opponent_identity_effects_aggregated.csv")
    theory = pd.read_csv(ROOT / "results/open_source/egt_reproduction/theory_llm_comparison.csv")
    assoc = pd.read_csv(ROOT / "results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/tables/selected_feature_associations.csv")
    controls = pd.read_csv(ROOT / "results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1/analysis/tables/fixed_state_target_control_contrasts.csv")
    fig, axes = plt.subplots(3, 2, figsize=(14.5, 15.2), constrained_layout=True)
    ax=axes[0,0]; sns.scatterplot(data=temp,x="full_effect_t0",y="full_effect_t07",hue="skin",s=90,ax=ax); lim=max(temp.full_effect_t0.max(),temp.full_effect_t07.max())*1.08; ax.plot([0,lim],[0,lim],ls="--",color=SLATE); ax.set(xlabel="Context effect, T=0",ylabel="Context effect, T=.7",title="A. Context ranks persist, magnitudes move",xlim=(-.01,lim),ylim=(-.01,lim)); ax.legend([],[],frameon=False)
    for r in temp.itertuples(): ax.annotate(CONTEXT.get(r.skin,r.skin),(r.full_effect_t0,r.full_effect_t07),xytext=(4,3),textcoords="offset points",fontsize=7)
    ax=axes[0,1]; sns.scatterplot(data=agreement,x="mean_decision_agreement",y="exact_joint_trajectory_rate",hue="skin",s=90,ax=ax); ax.plot([.8,1],[.8,1],ls="--",color=SLATE); ax.set(xlabel="Mean action agreement",ylabel="Exact joint trajectory rate",title="B. High local agreement still compounds"); ax.legend([],[],frameon=False)
    pos=position[(position.game_size==3)&(position.rank_label_condition=="numeric_only")].groupby(["model","position"],as_index=False).agg(unsafe_rate=("unsafe_rate","mean")); order=["leader","middle","last"]
    ax=axes[1,0]; sns.pointplot(data=pos,x="position",y="unsafe_rate",hue="model",order=order,markers="o",linestyles="-",palette=[BLUE,RED],ax=ax); ax.set(xlabel="Exogenous N=3 position",ylabel="Unsafe rate",title="C. Falling behind matters for Qwen, not Mistral",ylim=(-.05,1.05)); ax.legend(title="",fontsize=8,frameon=False)
    pivot=identity.pivot_table(index=["seat_model_key","persona_condition","round_phase"],columns="dyad_type",values="opponent_label_effect_pp").sort_index()
    ax=axes[1,1]; sns.heatmap(pivot,annot=True,fmt=".1f",cmap="vlag",center=0,cbar_kws={"label":"Disclosure effect (pp)"},ax=ax); ax.set(xlabel="Dyad type",ylabel="Model / persona / phase",title="D. Identity disclosure is strongest at entry")
    ax=axes[2,0]; x=100*theory.max_private_risk; ax.plot(x,100*theory.theory_unsafe_main_reference,marker="o",label="EGT strong selection",color=RED); ax.plot(x,100*theory.theory_unsafe_reported_best_fit,marker="o",label="EGT reported fit",color=AMBER); ax.plot(x,100*theory.llm_primary_t0_unsafe_technology_race,marker="o",label="LLM technology T=0",color=BLUE); ax.fill_between(x,100*theory.llm_primary_t0_unsafe_context_min,100*theory.llm_primary_t0_unsafe_context_max,color=CYAN,alpha=.2,label="LLM context range"); ax.set(xlabel="Maximum private risk (%)",ylabel="Unsafe rate (%)",title="E. Reproduced theory is not the prompt policy",ylim=(-3,103)); ax.legend(frameon=False,fontsize=8)
    ev=assoc[(assoc.split=="eval")&(assoc.feature_role=="selected")].copy(); ctrl=controls[controls.alpha.abs()==2].groupby("target_feature_id",as_index=False).agg(max_abs_control_delta=("mean_target_minus_control_delta",lambda s: s.abs().max())); ev=ev.merge(ctrl,left_on="feature_id",right_on="target_feature_id",how="left")
    ax=axes[2,1]; ax.scatter(ev.auc_oriented,ev.max_abs_control_delta,s=110,color=RED); [ax.annotate(str(r.feature_id),(r.auc_oriented,r.max_abs_control_delta),xytext=(5,4),textcoords="offset points",fontsize=8) for r in ev.itertuples()]; ax.axhline(0,color=SLATE,lw=1); ax.set(xlabel="Held-out oriented AUC",ylabel="Largest |target − control| mean delta",title="F. Predictive features do not clear causal controls")
    fig.suptitle("Extended evidence atlas — robustness, rank, identity, theory, and XAI",fontsize=22,fontweight="bold",color=NAVY)
    fig.text(.5,-.012,"Diagnostics remain checkpoint- and protocol-scoped. Position is fixed-state; identity is disclosed; theory and LLM trajectories are different processes.",ha="center",fontsize=10,color=SLATE)
    save(fig,"extended_evidence_atlas", dpi=160)


def write_readme() -> None:
    text = """# AI Race visual analytics atlas

This folder is the compact visualization layer for the admitted impact evidence.
Every chart is generated from `results/impact_upgrade/data/` or the frozen
comprehension table. PNG files target web/demo use; PDFs are vector exports for
the paper and slides.

## Open first

![Executive six-panel atlas](executive_visual_atlas.png)

## Detailed figures

![Checkpoint-by-risk heatmap](model_risk_heatmap.png)

![Dependency-aware context effect forest](context_effect_forest.png)

![Trajectory divergence heatmap](trajectory_divergence_heatmap.png)

## Fixed-state versus live estimands

![Fixed-state versus live explainer](fixed_vs_live_explainer.png)

`Fixed-state` compares prompts at the same frozen state and estimates direct
prompt response. `Live` reruns the game, so its contrast includes direct
response, repeated exposure, and endogenous state/opponent feedback. Their
difference is descriptive and is not a causal mediation estimate.

## Extended evidence

![Extended evidence atlas](extended_evidence_atlas.png)

- Context-effect ranks remain similar across T=0 and T=.7 (Spearman rho .857),
  but only 62.6% of complete player trajectories are identical.
- In the N=3 numeric-only fixed-state bank, Qwen rises from 50.0% Unsafe as
  leader to 91.7% as last; Mistral remains at 0% in every position cell.
- Disclosed opponent identity has its largest effects at round 1 (-41.7 to
  +25.0 percentage points); most later-round Mistral contrasts collapse to 0.
- The independently reproduced evolutionary phase pattern and the observed LLM
  prompt policy are not the same stochastic process or behavioral object.
- Selected FAST-SAE features retain held-out action information, but the
  target-minus-control interventions do not establish feature-specific control.

## Reading boundary

- Cross-provider curves are descriptive and never inferentially pooled.
- Live and fixed-state context effects answer different questions.
- Mapping follows repetition parity in this pilot, so mapping-conditioned results
  are replication targets rather than clean mapping-causal effects.
- The comprehension admission gate fails; behavior is not evidence of informed
  game-theoretic optimization.
- SAE decodability and feature causation remain separate claims.
"""
    (OUT / "README.md").write_text(text, encoding="utf-8", newline="\n")


def write_gallery() -> None:
    cards = [
        ("Executive visual atlas", "Six linked views: model heterogeneity, mapping, direct/live effects, payoff, trajectory, and comprehension.", "executive_visual_atlas.png", "executive_visual_atlas.pdf"),
        ("Checkpoint × risk heatmap", "Exact player-level Unsafe rates across five checkpoints and three risk caps.", "model_risk_heatmap.png", "model_risk_heatmap.pdf"),
        ("Dependency-aware context effects", "Fixed-state and live estimates with intervals at their correct clustering grains.", "context_effect_forest.png", "context_effect_forest.pdf"),
        ("Trajectory divergence by round", "Cumulative separation after identical round-1 decisions.", "trajectory_divergence_heatmap.png", "trajectory_divergence_heatmap.pdf"),
        ("Fixed-state versus live", "A visual estimand explainer: direct response at one frozen state versus total endogenous trajectory effect.", "fixed_vs_live_explainer.png", "fixed_vs_live_explainer.pdf"),
        ("Extended evidence atlas", "Temperature, rank, identity, evolutionary theory, and controlled SAE evidence.", "extended_evidence_atlas.png", "extended_evidence_atlas.pdf"),
    ]
    card_html = "\n".join(
        f'''<article><header><div><h2>{title}</h2><p>{caption}</p></div><a href="{pdf}">Vector PDF</a></header><a class="image" href="{png}"><img src="{png}" alt="{title}" loading="lazy"></a></article>'''
        for title, caption, png, pdf in cards
    )
    html = f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>AI Race Visual Analytics Atlas</title><style>
:root{{--ink:#0b132b;--muted:#64748b;--line:#dce3ed;--paper:#f8fafc;--blue:#2563eb}}*{{box-sizing:border-box}}
body{{margin:0;background:var(--paper);color:var(--ink);font:16px/1.55 Inter,ui-sans-serif,system-ui,sans-serif}}
main{{width:min(1500px,94vw);margin:auto;padding:56px 0 80px}}h1{{font-size:clamp(2rem,4vw,4.4rem);line-height:1.02;margin:0 0 18px}}
.lede{{max-width:900px;color:var(--muted);font-size:1.08rem;margin-bottom:40px}}.chips{{display:flex;gap:10px;flex-wrap:wrap;margin:24px 0 42px}}
.chips span{{border:1px solid var(--line);background:white;border-radius:999px;padding:7px 12px;font-size:.86rem}}
article{{background:white;border:1px solid var(--line);border-radius:20px;padding:22px;margin:24px 0;box-shadow:0 12px 36px #0b132b0a}}
header{{display:flex;justify-content:space-between;align-items:start;gap:24px;margin-bottom:18px}}h2{{margin:0 0 4px;font-size:1.35rem}}p{{margin:0;color:var(--muted)}}
header a{{white-space:nowrap;color:var(--blue);font-weight:700;text-decoration:none}}.image{{display:block;overflow:auto;border-radius:12px;background:#fff}}
img{{display:block;width:100%;height:auto}}footer{{color:var(--muted);font-size:.88rem;margin-top:36px}}
@media(max-width:700px){{main{{padding-top:32px}}article{{padding:12px}}header{{display:block}}header a{{display:inline-block;margin-top:10px}}}}
</style></head><body><main><h1>AI Race visual analytics atlas</h1>
<p class="lede">A validity-first visual synthesis of the admitted results. Each panel preserves its own protocol, denominator, estimand, and uncertainty boundary.</p>
<div class="chips"><span>Matplotlib 3.10</span><span>Seaborn 0.13</span><span>PNG 220 dpi</span><span>Vector PDF</span><span>No protocol pooling</span></div>
{card_html}
<footer>Exploratory pilots unless explicitly labelled method validation. Mapping-conditioned findings remain replication targets because mapping follows repetition parity.</footer>
</main></body></html>'''
    (OUT / "index.html").write_text(html, encoding="utf-8", newline="\n")


def main() -> None:
    style()
    cross = pd.read_csv(DATA / "cross_model_baseline_rates.csv")
    mapping = pd.read_csv(DATA / "context_mapping_interaction.csv")
    direct = pd.read_csv(DATA / "context_direct_vs_live.csv")
    tradeoff = pd.read_csv(DATA / "context_behavior_payoff_tradeoff.csv")
    curve = pd.read_csv(DATA / "trajectory_divergence_curve.csv")
    comprehension = pd.read_csv(ROOT / "results/open_source/context_skin_pilot/analysis_live_pilot_t0/comprehension_by_domain.csv")
    model_risk_heatmap(cross)
    context_effect_forest(direct)
    divergence_heatmap(curve)
    executive_atlas(cross, mapping, direct, tradeoff, curve, comprehension)
    fixed_vs_live_explainer()
    extended_evidence_atlas()
    write_readme()
    write_gallery()
    print(f"Wrote visual atlas to {OUT}")


if __name__ == "__main__":
    main()
