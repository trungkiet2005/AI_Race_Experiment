"""Draw the redrawn paper figures from the derived source tables.

Reads only ``tables/*.csv`` (written by prepare_data.py) so that every mark traces
to a table a reviewer can open. Rationale for each form choice is in
``FIGURE_REDRAW_PLAN.md``; the short version is recorded above each function.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from figstyle import (COL_W, FULL_W, GRID, INK, INK_2, MUTED, SURFACE, TABLE_DIR,
                      C_AQUA, C_BLUE, C_MAGENTA, C_ORANGE, MODEL_STYLE,
                      PROVIDER_STYLE, RISK_LABEL, dot, interval, save, strip,
                      use_paper_style, zero_rule)

HUMAN_GREY = "#9a9a95"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLE_DIR / f"{name}.csv")


# ---------------------------------------------------------------------------
# A. Human-reference comparison
# ---------------------------------------------------------------------------
# Form: paired dot-and-interval on one log-odds axis. The superseded scorecard put
# five different tests in one bar grid with eight y-scales; only the four dynamic
# coefficients share a scale with each other, so only they are plotted. The rest
# is a table.
def figure_a() -> None:
    d = load("figA_human_reference")
    a = d[d.block == "A"].reset_index(drop=True)
    b = d[d.block == "B"].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(FULL_W * 0.74, 2.75))
    n_a, n_b = len(a), len(b)
    # Row 0 at the top; a one-row gap separates the two blocks.
    y_a = np.arange(n_a)[::-1] + n_b + 1.35
    y_b = np.arange(n_b)[::-1]

    XMAX = 5.0
    off = 0.19
    for i, r in a.iterrows():
        y = y_a[i]
        interval(ax, y + off, r.human_ci_low, r.human_ci_high, color=HUMAN_GREY, lw=1.7)
        dot(ax, r.human_beta, y + off, color=HUMAN_GREY, marker="o", size=30)
        hi = min(r.llm_ci_high, XMAX)
        interval(ax, y - off, r.llm_ci_low, hi, color=C_BLUE)
        if r.llm_ci_high > XMAX:
            # Clipped, not truncated: an arrowhead plus the value, so the reader is
            # never told the interval ends at the axis.
            ax.annotate("", xy=(XMAX + 0.16, y - off), xytext=(hi, y - off),
                        arrowprops=dict(arrowstyle="-|>", color=C_BLUE, lw=1.6,
                                        shrinkA=0, shrinkB=0, mutation_scale=7))
            ax.annotate(f"upper 95% = {r.llm_ci_high:.2f}", xy=(XMAX + 0.2, y - off),
                        xytext=(0, -7.5), textcoords="offset points",
                        fontsize=6.3, color=INK_2, ha="right", va="top")
        dot(ax, r.llm_beta, y - off, color=C_BLUE, marker="o", size=34,
            filled=bool(r.sign_stable))

    for i, r in b.iterrows():
        y = y_b[i]
        interval(ax, y, r.llm_ci_low, r.llm_ci_high, color=C_BLUE)
        dot(ax, r.llm_beta, y, color=C_BLUE, marker="o", size=34,
            filled=bool(r.sign_stable))

    zero_rule(ax, 0.0)
    ax.set_yticks(list(y_a) + list(y_b))
    ax.set_yticklabels(list(a.label) + list(b.label))
    ax.set_ylim(-0.75, y_a[0] + 1.0)
    ax.set_xlim(-2.5, XMAX + 0.25)
    ax.set_xticks([-2, -1, 0, 1, 2, 3, 4, 5])
    ax.set_xlabel("Cluster-robust logit coefficient (log-odds of choosing Unsafe)")
    strip(ax)
    ax.tick_params(axis="y", length=0)

    sep = (y_a[-1] + y_b[0]) / 2
    ax.axhline(sep, color=GRID, lw=0.7, zorder=0)
    for text, y0 in (("Dynamic predictors: human study vs. LLM pilot", y_a[0]),
                     ("Risk treatment (LLM pilot only; no published human log-odds)", y_b[0])):
        ax.annotate(text, xy=(0.0, y0 + 0.68), xycoords=("axes fraction", "data"),
                    fontsize=7.0, color=INK_2, style="italic", va="center")

    handles = [
        Line2D([], [], color=HUMAN_GREY, marker="o", lw=1.7, ms=5.2,
               markeredgecolor=SURFACE, label="Human study (Table 1, model 6)"),
        Line2D([], [], color=C_BLUE, marker="o", lw=1.8, ms=5.6,
               markeredgecolor=SURFACE, label="LLM pilot"),
        Line2D([], [], color=C_BLUE, marker="o", lw=0, ms=5.6, markerfacecolor=SURFACE,
               markeredgecolor=C_BLUE, markeredgewidth=1.5,
               label="sign flips when any one repetition block is dropped"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.56, -0.14),
               handletextpad=0.5, labelspacing=0.32, columnspacing=1.6)
    save(fig, "figA_human_reference")


# ---------------------------------------------------------------------------
# B. Surface-wording sensitivity
# ---------------------------------------------------------------------------
# Form: two dot-and-interval panels sharing a row order. Left carries the claim
# (matched state, matched seed, so the contrast is controlled); right carries the
# trajectory rate, which is confounded by endogenous state and is labelled as such.
# Sorting both by the left panel makes the dissociation a shape rather than a caveat.
INTERP_STYLE = {
    "meaning_preserving": (C_BLUE, "o", "meaning-preserving rewrite"),
    "behavioral_framing": (C_ORANGE, "D", "behavioural framing cue (not meaning-preserving)"),
    "robustness_perturbation": (C_AQUA, "s", "robustness perturbation (typos)"),
    "control": (MUTED, "o", "canonical wording (reference)"),
}


def figure_b() -> None:
    """The finding is a *dissociation between two measures*, which is a position in two
    dimensions, not two columns of dots. Plotting the controlled contrast against the
    confounded one puts every variant in a quadrant whose meaning is the conclusion:
    almost everything sits in the left band -- the wording moved the trajectory without
    moving the decision."""
    d = load("figB_surface_variants").reset_index(drop=True)
    canonical = float(d.loc[d.variant == "canonical", "unsafe_rate"].iloc[0])
    can_lo = float(d.loc[d.variant == "canonical", "unsafe_rate_cluster_bootstrap_ci95_low"].iloc[0])
    can_hi = float(d.loc[d.variant == "canonical", "unsafe_rate_cluster_bootstrap_ci95_high"].iloc[0])

    fig, ax = plt.subplots(figsize=(FULL_W * 0.82, 3.5))

    # The 15% band is not a threshold, it is the observed ceiling of every
    # meaning-preserving rewrite; shading it names the region the claim lives in.
    ax.axvspan(-0.04, 0.15, color=C_BLUE, alpha=0.055, zorder=0, lw=0)
    ax.axhspan(can_lo, can_hi, color=MUTED, alpha=0.15, zorder=0, lw=0)
    ax.axhline(canonical, color=MUTED, lw=0.9, zorder=1)
    ax.axvline(0.0, color=MUTED, lw=0.9, zorder=1)

    for _, r in d.iterrows():
        color, marker, _ = INTERP_STYLE[r.interpretation]
        is_ctl = r.variant == "canonical"
        x, yv = r.first_round_flip_rate_vs_canonical, r.unsafe_rate
        ax.plot([r.first_round_flip_cluster_bootstrap_ci95_low,
                 r.first_round_flip_cluster_bootstrap_ci95_high], [yv, yv],
                color=color, lw=1.0, alpha=0.35, solid_capstyle="butt", zorder=2)
        ax.plot([x, x], [r.unsafe_rate_cluster_bootstrap_ci95_low,
                         r.unsafe_rate_cluster_bootstrap_ci95_high],
                color=color, lw=1.0, alpha=0.35, solid_capstyle="butt", zorder=2)
        dot(ax, x, yv, color=color, marker=marker, size=40, filled=not is_ctl)

    # Only the four points that carry an argument are labelled; the rest are the mass.
    for variant, dx, dy, ha, va in (
        ("canonical", -10, -9, "right", "top"),
        ("order_actions_reversed", 11, 0, "left", "center"),
        ("position_risk_near_response", 11, 0, "left", "center"),
        ("emotional_importance", -10, -10, "right", "top"),
    ):
        r = d[d.variant == variant].iloc[0]
        ax.annotate(r.label, xy=(r.first_round_flip_rate_vs_canonical, r.unsafe_rate),
                    xytext=(dx, dy), textcoords="offset points", fontsize=6.5,
                    color=INK_2, ha=ha, va=va)

    ax.annotate("15 of 18 rewrites land inside this band:\n"
                "the wording moved the trajectory, not the decision",
                xy=(0.15, 0.36), xytext=(0.29, 0.24), textcoords="data",
                fontsize=6.8, color=C_BLUE, ha="left", va="center",
                arrowprops=dict(arrowstyle="-", color=C_BLUE, lw=0.7, alpha=0.6,
                                shrinkA=2, shrinkB=2))

    ax.set_xlim(-0.04, 0.92)
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks([0, 0.15, 0.25, 0.5, 0.75])
    ax.set_xticklabels(["0", "15%", "25%", "50%", "75%"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "25%", "50%", "75%", "100%"])
    ax.set_xlabel("CONTROLLED $\\rightarrow$  round-1 decisions flipped vs. canonical\n"
                  "matched state and seed, $n$=60 paired")
    ax.set_ylabel("CONFOUNDED $\\rightarrow$  whole-trajectory Unsafe rate\n"
                  "$n$=558 decisions per variant")
    strip(ax, y=True, grid_axis="both")

    handles = [Line2D([], [], color=c, marker=m, lw=0, ms=5.4, markeredgecolor=SURFACE,
                      label=lab) for c, m, lab in INTERP_STYLE.values()]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, -0.14),
               handletextpad=0.5, columnspacing=1.6)
    save(fig, "figB_surface_wording")


def figure_b_companion() -> None:
    """Signed decomposition of the same paired round-1 flips. The flip rate cannot
    show direction, and direction is the behavioural claim."""
    d = load("figB_surface_variants").reset_index(drop=True)
    d = d[d.variant != "canonical"]
    y = np.arange(len(d))[::-1]

    fig, ax = plt.subplots(figsize=(COL_W * 1.35, 3.3))
    h = 0.62
    for i, (_, r) in enumerate(d.iterrows()):
        if r.first_round_safe_to_unsafe:
            ax.barh(y[i], r.first_round_safe_to_unsafe, height=h, left=1.0,
                    color=C_ORANGE, edgecolor=SURFACE, lw=1.0, zorder=3)
        if r.first_round_unsafe_to_safe:
            # Both sides start one unit off zero, so the two polarities never touch.
            ax.barh(y[i], r.first_round_unsafe_to_safe, height=h,
                    left=-(1.0 + r.first_round_unsafe_to_safe),
                    color=C_BLUE, edgecolor=SURFACE, lw=1.0, zorder=3)
    zero_rule(ax, 0.0)
    ax.set_yticks(y)
    ax.set_yticklabels(d.label)
    ax.set_ylim(-0.8, len(d) - 0.2)
    # The axis runs to the ceiling of 60 on both sides, not to the data maximum, so
    # the reader can see how much of the paired sample each variant actually moved.
    ax.set_xlim(-62, 62)
    ax.set_xticks([-60, -40, -20, 0, 20, 40, 60])
    ax.set_xticklabels(["60", "40", "20", "0", "20", "40", "60"])
    ax.set_xlabel("$\\leftarrow$ toward Safe    Round-1 decisions flipped, of 60 paired"
                  "    toward Unsafe $\\rightarrow$")
    strip(ax)
    ax.tick_params(axis="y", length=0)
    handles = [
        Line2D([], [], color=C_ORANGE, lw=6, label="canonical Safe $\\rightarrow$ Unsafe"),
        Line2D([], [], color=C_BLUE, lw=6, label="canonical Unsafe $\\rightarrow$ Safe"),
    ]
    ax.legend(handles=handles, loc="lower right", handlelength=1.2, handletextpad=0.5)
    save(fig, "figB2_surface_flip_direction")


# ---------------------------------------------------------------------------
# C. Cross-provider opponent contingency
# ---------------------------------------------------------------------------
# Form: slopegraph faceted by risk, plus a paired-contrast panel. The design pairs
# a model's two opponent contexts on matched CRN repetitions; a sorted forest of
# unpaired marginal intervals discards exactly that pairing.
PAIRS = [
    ("GPT Luna", "Claude Haiku 4.5", "Gemini 3.5 Flash-Lite"),
    ("Gemini 3.5 Flash-Lite", "Claude Haiku 4.5", "GPT Luna"),
    ("Claude Haiku 4.5", "GPT Luna", "Gemini 3.5 Flash-Lite"),
]
SHORT = {"GPT Luna": "Luna", "Claude Haiku 4.5": "Haiku",
         "Gemini 3.5 Flash-Lite": "Gemini"}


def figure_c() -> None:
    delta = load("figC_paired_delta")
    per_risk = delta[delta.max_private_risk != "pooled"].copy()
    per_risk["max_private_risk"] = per_risk.max_private_risk.astype(float)
    pooled = delta[delta.max_private_risk == "pooled"]

    fig, axes = plt.subplots(
        1, 4, figsize=(FULL_W, 2.5),
        gridspec_kw={"width_ratios": [1, 1, 1, 1.25], "wspace": 0.28})

    for ax, risk in zip(axes[:3], (0.1, 0.6, 0.9)):
        sub = per_risk[per_risk.max_private_risk == risk]
        for model, opp_a, opp_b in PAIRS:
            r = sub[sub.player == model].iloc[0]
            color, marker = PROVIDER_STYLE[model]
            ax.plot([0, 1], [r.rate_vs_a, r.rate_vs_b], color=color, lw=1.7, zorder=3)
            for x, v in ((0, r.rate_vs_a), (1, r.rate_vs_b)):
                dot(ax, x, v, color=color, marker=marker, size=32)
        ax.set_xlim(-0.32, 1.32)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["vs Haiku", "vs other"])
        ax.set_ylim(0.3, 0.8)
        ax.set_yticks([0.3, 0.4, 0.5, 0.6, 0.7, 0.8])
        ax.set_title(f"risk {RISK_LABEL[risk]}", loc="center", color=INK_2, pad=5)
        strip(ax, grid_axis="y")
        if risk != 0.1:
            ax.set_yticklabels([])
            ax.tick_params(axis="y", length=0)
    axes[0].set_ylabel("Unsafe choice rate")

    axd = axes[3]
    y = np.arange(len(pooled))[::-1]
    for i, (_, r) in enumerate(pooled.iterrows()):
        color, marker = PROVIDER_STYLE[r.player]
        interval(axd, y[i], r.ci_low, r.ci_high, color=color)
        dot(axd, r.paired_delta, y[i], color=color, marker=marker, size=34)
        axd.annotate(f"{r.paired_delta:+.3f}", xy=(r.ci_high, y[i]), xytext=(4, 0),
                     textcoords="offset points", fontsize=6.4, color=INK_2, va="center")
    zero_rule(axd, 0.0)
    axd.set_yticks(y)
    axd.set_yticklabels([SHORT[p] for p in pooled.player])
    axd.set_ylim(-0.7, len(pooled) - 0.3)
    axd.set_xlim(-0.03, 0.31)
    axd.set_xticks([0.0, 0.1, 0.2, 0.3])
    axd.set_xlabel("paired $\\Delta$, pooled over risk")
    axd.set_title("within-model, between-opponent", loc="center", color=INK_2, pad=5)
    strip(axd)

    # Haiku is its own series' reference opponent for the other two, so its own row has
    # no "vs Haiku" column; naming its two opponents in the legend keeps the shared
    # x-axis honest without an annotation that collides with the lines.
    legend_label = {"Claude Haiku 4.5": "Claude Haiku 4.5 (vs Luna, then vs Gemini)"}
    handles = [Line2D([], [], color=PROVIDER_STYLE[m][0], marker=PROVIDER_STYLE[m][1],
                      lw=1.7, ms=5.4, markeredgecolor=SURFACE,
                      label=legend_label.get(m, m))
               for m, _, _ in PAIRS]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.14),
               handletextpad=0.5, columnspacing=1.6)
    save(fig, "figC_opponent_contingency")


# ---------------------------------------------------------------------------
# D. Race position, five-checkpoint baseline
# ---------------------------------------------------------------------------
# Form: dot-and-interval over an ordinal position axis, one row per checkpoint.
# Replaces the N=3 rank figure, whose mechanism is not the paper's game and whose
# leading category counted ties as leads.
POSITIONS = ["Ahead", "Tied", "Behind"]


def figure_d() -> None:
    """Faceted rather than overlaid: three of the five checkpoints run along the same
    path and their direct labels collide on a shared panel. Small multiples also let
    each checkpoint's deterministic cells be marked without a shared legend."""
    d = load("figD_position")
    order = ["Gemini 3 Flash", "Gemini 3.1 Flash-Lite", "Gemini 3.5 Flash-Lite",
             "GPT-5 nano", "GPT-5.4 nano"]
    xpos = {p: i for i, p in enumerate(POSITIONS)}

    fig, axes = plt.subplots(1, 5, figsize=(FULL_W, 2.05), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    for ax, model in zip(axes, order):
        color, marker = MODEL_STYLE[model]
        sub = d[d.model_label == model].set_index("position")
        ax.plot([xpos[p] for p in POSITIONS], [sub.loc[p, "estimate"] for p in POSITIONS],
                color=color, lw=1.4, zorder=3, alpha=0.85)
        for p in POSITIONS:
            r = sub.loc[p]
            if not r.degenerate:
                ax.plot([xpos[p]] * 2, [r.ci95_low, r.ci95_high], color=color, lw=1.7,
                        solid_capstyle="butt", zorder=3)
            dot(ax, xpos[p], r.estimate, color=color, marker=marker, size=30,
                filled=not r.degenerate)
        n_pair = int(sub.loc["Ahead", "n_observations"])
        ax.set_title(model, loc="center", color=color, fontsize=7.2, pad=4)
        # Top-left is the one corner free in all five panels.
        ax.annotate(f"$n$={n_pair} ahead,\n{n_pair} behind", xy=(0.03, 0.97),
                    xycoords="axes fraction", ha="left", va="top", fontsize=6.1,
                    color=MUTED)
        ax.set_xticks(list(xpos.values()))
        ax.set_xticklabels(POSITIONS, fontsize=6.6)
        ax.set_xlim(-0.5, 2.5)
        ax.set_ylim(-0.10, 1.10)
        strip(ax, grid_axis="y")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axes[0].set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axes[0].set_ylabel("Unsafe choice rate")
    fig.supxlabel("Own progress relative to the opponent, before the decision",
                  fontsize=7.6, color=INK_2, y=-0.06)

    handles = [Line2D([], [], color=INK_2, marker="o", lw=0, ms=5.4,
                      markerfacecolor=SURFACE, markeredgecolor=INK_2, markeredgewidth=1.5,
                      label="deterministic cell: all ten blocks identical, interval degenerate")]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.19),
               handletextpad=0.5)
    save(fig, "figD_race_position")


# ---------------------------------------------------------------------------
# E. Baseline risk response
# ---------------------------------------------------------------------------
# Form: discrete dot-and-interval on a true linear risk axis, plus the paired
# within-block contrast. The superseded version painted a continuous confidence
# band across risk values that were never run, and plotted two degenerate
# intervals as though they were the tightest estimates on the chart.
def figure_e() -> None:
    rate = load("figE_risk_response")
    contrast = load("figE_contrast")
    order = ["Gemini 3 Flash", "Gemini 3.1 Flash-Lite", "Gemini 3.5 Flash-Lite",
             "GPT-5 nano", "GPT-5.4 nano"]

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FULL_W, 2.6),
        gridspec_kw={"width_ratios": [1.4, 1.0], "wspace": 0.46})

    for k, model in enumerate(order):
        color, marker = MODEL_STYLE[model]
        sub = rate[rate.model_label == model].sort_values("max_private_risk")
        # Jitter the saturated pair so two coincident marks remain two marks.
        jx = (k - 0.5) * 0.017 if sub.saturated.any() else 0.0
        x = sub.max_private_risk + jx
        axA.plot(x, sub.estimate, color=color, lw=1.0, alpha=0.42, zorder=2)
        for _, r in sub.iterrows():
            xi = r.max_private_risk + jx
            if not r.saturated:
                axA.plot([xi, xi], [r.ci95_low, r.ci95_high], color=color, lw=1.7,
                         solid_capstyle="butt", zorder=3)
            dot(axA, xi, r.estimate, color=color, marker=marker, size=32,
                filled=not r.saturated)

    axA.set_xlim(0.0, 1.0)
    axA.set_xticks([0.1, 0.6, 0.9])
    axA.set_xticklabels(["10%", "60%", "90%"])
    axA.set_ylim(-0.04, 1.12)
    axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axA.set_xlabel("Maximum private setback risk")
    axA.set_ylabel("Mean Unsafe fraction per player-race")
    axA.set_title("A   Risk response", loc="left", color=INK, pad=6)
    strip(axA, grid_axis="y")
    axA.annotate("saturated: 20/20 player-races Unsafe,\nbootstrap interval degenerate",
                 xy=(0.125, 1.0), xytext=(0.30, 1.10), textcoords="data",
                 fontsize=6.3, color=INK_2, va="top", ha="left",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.7,
                                 shrinkA=1, shrinkB=2))

    y = np.arange(len(order))[::-1]
    for i, model in enumerate(order):
        color, marker = MODEL_STYLE[model]
        r = contrast[contrast.model_label == model].iloc[0]
        interval(axB, y[i], r.ci95_low, r.ci95_high, color=color)
        dot(axB, r.estimate, y[i], color=color, marker=marker, size=34)
    zero_rule(axB, 0.0)
    axB.set_yticks(y)
    axB.set_yticklabels(order)
    axB.set_ylim(-0.7, len(order) - 0.3)
    axB.set_xlim(-0.56, 0.16)
    axB.set_xticks([-0.5, -0.25, 0.0])
    axB.set_xticklabels(["$-$50 pp", "$-$25 pp", "0"])
    axB.set_xlabel("Change in Unsafe fraction,\n90% risk $-$ 10% risk")
    axB.set_title("B   Paired within CRN repetition block", loc="left", color=INK, pad=6)
    strip(axB)

    handles = [Line2D([], [], color=MODEL_STYLE[m][0], marker=MODEL_STYLE[m][1],
                      lw=1.6, ms=5.2, markeredgecolor=SURFACE, label=m) for m in order]
    handles.append(Line2D([], [], color=INK_2, marker="o", lw=0, ms=5.2,
                          markerfacecolor=SURFACE, markeredgecolor=INK_2,
                          markeredgewidth=1.5, label="saturated / degenerate interval"))
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.30),
               handletextpad=0.5, columnspacing=1.5)
    save(fig, "figE_baseline_risk_response")


# ---------------------------------------------------------------------------
# F. Repeat-run stability
# ---------------------------------------------------------------------------
# Form: aggregate rates left, per-seed agreement right. Panel B was three bars with
# no uncertainty over what looked like n=74 independent decisions; the sampling unit
# is five shared race seeds, so the seeds are shown instead of an interval a
# five-block bootstrap cannot stabilise.
def figure_f() -> None:
    rates = load("figF_rates")
    pooled = load("figF_agreement_pooled")
    per_seed = load("figF_agreement_per_seed")

    fig, (axA, axB) = plt.subplots(
        1, 2, figsize=(FULL_W * 0.92, 2.5),
        gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.26})

    runs = [("Earlier 5-rep pilot (10 player-races/cell)", C_ORANGE, "o"),
            ("Final 10-rep pilot (20 player-races/cell)", C_BLUE, "s")]
    for label, color, marker in runs:
        sub = rates[rates.run == label.split(" (")[0]].sort_values("max_private_risk")
        axA.plot(sub.max_private_risk, sub.unsafe_rate, color=color, lw=1.5,
                 zorder=3, label=label)
        for _, r in sub.iterrows():
            dot(axA, r.max_private_risk, r.unsafe_rate, color=color, marker=marker, size=32)
    axA.set_xlim(0.0, 1.0)
    axA.set_xticks([0.1, 0.6, 0.9])
    axA.set_xticklabels(["10%", "60%", "90%"])
    axA.set_ylim(-0.04, 1.1)
    axA.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    axA.set_yticklabels(["0%", "25%", "50%", "75%", "100%"])
    axA.set_xlabel("Maximum private setback risk")
    axA.set_ylabel("Unsafe fraction")
    axA.set_title("A   Aggregate risk response agrees", loc="left", color=INK, pad=6)
    strip(axA, grid_axis="y")
    axA.legend(loc="lower left", handletextpad=0.6)

    # Panel B shows the 222 matched decisions themselves. Three summary heights with an
    # interval would be a claim about a population; a field of 222 tiles is the evidence,
    # and it makes the difference between a solid block and a speckled one immediate --
    # which is exactly the "stable in aggregate, unstable per decision" point.
    per_dec = load("figF_agreement_per_decision")
    per_dec = per_dec.sort_values(["max_private_risk", "game_seed", "round", "player"])
    seeds = sorted(per_dec.game_seed.unique())
    row_of = {0.1: 2, 0.6: 1, 0.9: 0}
    COLS = 26
    for risk, g_risk in per_dec.groupby("max_private_risk"):
        row = row_of[risk]
        col = 0
        for seed in seeds:
            g = g_risk[g_risk.game_seed == seed]
            for _, r in g.iterrows():
                x, ydelta = col % COLS, (col // COLS) * 0.30
                axB.add_patch(plt.Rectangle(
                    (x, row - ydelta - 0.11), 0.82, 0.22,
                    facecolor=C_BLUE if r.same_action else SURFACE,
                    edgecolor=C_BLUE if r.same_action else "#c4c3bf",
                    linewidth=0.5, zorder=3))
                col += 1
            col += 1  # a one-tile gap marks the seed boundary
    for risk, row in row_of.items():
        r = pooled[pooled.max_private_risk == risk].iloc[0]
        axB.annotate(f"risk {RISK_LABEL[risk]}", xy=(-1.2, row - 0.15), fontsize=7.0,
                     color=INK_2, ha="right", va="center")
        note = (f"{r.agreement:.0%} agree" if r.forced else
                f"{r.agreement:.0%} agree · {r.chance_agreement:.0%} by chance · "
                f"$\\kappa$={r.kappa:.2f}")
        axB.annotate(note, xy=(-1.2, row - 0.44), fontsize=6.3, color=MUTED,
                     ha="right", va="center")
    axB.annotate("forced: both runs are 100% Unsafe here, so agreement carries no information",
                 xy=(0, row_of[0.1] + 0.30), fontsize=6.3, color=INK_2, ha="left",
                 va="bottom")
    axB.set_xlim(-13.5, COLS + 0.5)
    axB.set_ylim(-0.85, 2.75)
    axB.set_xticks([])
    axB.set_yticks([])
    for side in ("top", "right", "bottom", "left"):
        axB.spines[side].set_visible(False)
    axB.set_title("B   Individual decisions do not", loc="left", color=INK, pad=6)
    axB.set_xlabel("one tile = one matched game–round–seat decision\n"
                   "74 per risk level, grouped by the five shared race seeds")
    handles = [
        Line2D([], [], color=C_BLUE, lw=0, marker="s", ms=5.0, label="same action"),
        Line2D([], [], color="#c4c3bf", lw=0, marker="s", ms=5.0,
               markerfacecolor=SURFACE, markeredgecolor="#c4c3bf",
               label="different action"),
    ]
    axB.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, -0.06),
               ncol=2, handletextpad=0.4, columnspacing=1.2)
    save(fig, "figF_repeat_run_stability")


# ---------------------------------------------------------------------------
# G. Comprehension audit (appendix)
# ---------------------------------------------------------------------------
# Form: dot-and-interval, faceted by metric because only one of the two is gated.
# The superseded version ran one 75% rule across both metrics, rendered a true zero
# as an invisible bar, and paired it with a 16-cell heatmap whose entire dynamic
# range was two items at n=4.
def figure_g() -> None:
    """The finding is a *crossing*: the one domain the model gets 100% right is the one
    it never formats correctly, and the domain it formats almost perfectly is one it
    gets wrong 83% of the time. Two columns of dots hide that; a slopegraph makes the
    lines cross, which is the result."""
    d = load("figG_comprehension")
    domains = ["rule_recall", "stage_payoff", "state_update", "terminal_scoring"]
    pretty = {"rule_recall": "Rule recall", "stage_payoff": "Stage payoff",
              "state_update": "State update", "terminal_scoring": "Terminal scoring"}
    # Warm where knowing is easy, cool where formatting is: the crossing pairs get the
    # two ends of the categorical order so the swap is legible at a glance.
    hue = {"rule_recall": C_BLUE, "stage_payoff": C_AQUA,
           "state_update": MUTED, "terminal_scoring": C_ORANGE}
    sem = d[d.metric == "semantic accuracy"].set_index("domain")
    fmt = d[d.metric == "strict format validity"].set_index("domain")

    fig, ax = plt.subplots(figsize=(FULL_W * 0.60, 3.0))

    # The gates live in the tick column rather than as annotated rules inside the plot:
    # at 90% and 100% the in-plot labels would sit on top of the domain labels.
    for value in (0.75, 0.90):
        ax.axhline(value, color=MUTED, lw=0.8, zorder=1)

    # rule_recall (100%) and stage_payoff (98%) sit 2pp apart -- direct labels there
    # would collide regardless of offset, so those two left-side labels are stacked in
    # a fixed reading order instead of pinned to their exact y.
    left_label_y = {"rule_recall": 1.00, "stage_payoff": 0.88,
                    "terminal_scoring": 0.17, "state_update": 0.05}
    for dom in domains:
        s, f = sem.loc[dom], fmt.loc[dom]
        color = hue[dom]
        ax.plot([0, 1], [s.value, f.value], color=color, lw=2.0, zorder=3,
                solid_capstyle="round")
        for x, r in ((0, s), (1, f)):
            ax.plot([x, x], [r.ci_low, r.ci_high], color=color, lw=1.5,
                    solid_capstyle="butt", zorder=3, alpha=0.7)
            dot(ax, x, r.value, color=color, marker="o", size=38)
        ax.annotate(f"{pretty[dom]}  {s.value:.0%}", xy=(0, left_label_y[dom]),
                    xytext=(-9, 0), textcoords="offset points", fontsize=6.8,
                    color=color, ha="right", va="center")
        label = "0%  (true zero, not missing)" if f.value == 0 else f"{f.value:.0%}"
        ax.annotate(label, xy=(1, f.value), xytext=(9, 0), textcoords="offset points",
                    fontsize=6.8, color=color, ha="left", va="center")

    ax.annotate("the model formats what it gets wrong\nand fumbles what it knows",
                xy=(0.5, 0.50), xytext=(0.5, 0.50), fontsize=6.8, color=INK_2,
                ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.35", facecolor=SURFACE,
                          edgecolor=GRID, linewidth=0.6))

    ax.set_xlim(-0.62, 1.52)
    ax.set_ylim(-0.06, 1.06)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Semantic accuracy\n(gated)", "Strict format validity\n(not gated)"])
    ax.set_yticks([0, 0.25, 0.5, 0.75, 0.90, 1.0])
    ax.set_yticklabels(["0%", "25%", "50%", "75%  per-domain gate",
                        "90%  overall gate", "100%"])
    ax.set_ylabel("Rate, 95% Wilson interval ($n$=64 per domain)")
    strip(ax, y=True, grid_axis="y")
    ax.tick_params(axis="x", length=0)
    for lab in ax.get_yticklabels():
        if "gate" in lab.get_text():
            lab.set_color(MUTED)
            lab.set_fontsize(6.4)
    save(fig, "figG_comprehension_audit")


if __name__ == "__main__":
    use_paper_style()
    figure_a()
    figure_b()
    figure_b_companion()
    figure_c()
    figure_d()
    figure_e()
    figure_f()
    figure_g()
