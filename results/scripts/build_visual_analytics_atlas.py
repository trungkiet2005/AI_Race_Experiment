#!/usr/bin/env python3
"""Build the publication and demo visual atlas from admitted impact tables."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
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


def save(fig: plt.Figure, stem: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{stem}.png", dpi=220, bbox_inches="tight", facecolor="white")
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
    write_readme()
    write_gallery()
    print(f"Wrote visual atlas to {OUT}")


if __name__ == "__main__":
    main()
