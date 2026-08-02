#!/usr/bin/env python3
"""Does the persona/framing-dominance finding replicate across model families?

Extends the original 3-checkpoint persona-role gradient
(build_cross_model_pilot_synthesis.py: gpt-5-nano / gpt-5.4-nano /
gemini-3-flash-preview, via direct provider APIs) with four checkpoints run
through AWS Bedrock: gpt-5.6-luna and gpt-5.6-terra (the "mantle" route,
results/frontier/bedrock_mantle/) and claude-opus-5 / claude-sonnet-5
(results/frontier/bedrock/). Seven checkpoints across three model families and
three backend routes now have a complete 6x6 persona risk_matrix.

The Bedrock route was substituted for the direct OpenAI route on the GPT-5.6
checkpoints after the latter hit two compounding problems: billing credit
exhaustion, and a reasoning-effort mitigation bug (probe_reasoning_effort() sent
reasoning_effort="minimal", invalid for gpt-5.6 -- valid values are
none/low/medium/high/xhigh -- which the harness misread as "this model rejects
the parameter entirely" and permanently disabled). Both Bedrock sweeps are
unaffected: every run directory is complete with zero parse failures.

Each checkpoint is anchored by its own neutral/no-persona lane at the left of
the gradient, so the framing swing is read against un-framed behaviour rather
than against R1 -- which is itself already an extreme framing.
"""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
FIGURES = ROOT / "results" / "cross_model_pilot_synthesis" / "figures"
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
BEDROCK_MANTLE_ROOT = ROOT / "results" / "frontier" / "bedrock_mantle"
BEDROCK_ROOT = ROOT / "results" / "frontier" / "bedrock"
ORIGINAL_GRADIENT_CSV = DATA / "persona_role_gradient.csv"

PALETTE = {
    "navy": "#0B132B", "slate": "#64748B", "grid": "#DCE3ED",
    # Validated categorical slots (see the dataviz palette reference); the old
    # ad hoc hues here had pairs too close to separate under CVD simulation.
    "blue": "#2a78d6", "orange": "#eb6834", "aqua": "#1baf7a", "yellow": "#eda100",
}
MODEL_ORDER = ["gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview",
               "gpt-5.6-luna", "gpt-5.6-terra", "claude-opus-5", "claude-sonnet-5"]
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}
# Faceted so no single axes carries more than four lines; hues are the validated
# categorical slots and are reused across facets, which never share an axes.
GRADIENT_FACETS = [
    ("OpenAI-lineage checkpoints", ["gpt-5-nano", "gpt-5.4-nano", "gpt-5.6-luna", "gpt-5.6-terra"]),
    ("Gemini and Claude", ["google/gemini-3-flash-preview", "claude-opus-5", "claude-sonnet-5"]),
]
FACET_COLORS = [PALETTE["blue"], PALETTE["orange"], PALETTE["aqua"], PALETTE["yellow"]]
ROLE_LABEL = {f"risk-{i}": f"R{i}" for i in range(1, 7)}
# key -> (persona sweep root, model subdirectory name)
SWEEP_ROOTS = {
    "gpt-5.6-luna": (BEDROCK_MANTLE_ROOT / "luna" / "persona", "openai.gpt-5.6-luna"),
    "gpt-5.6-terra": (BEDROCK_MANTLE_ROOT / "terra" / "persona", "openai.gpt-5.6-terra"),
    "claude-opus-5": (BEDROCK_ROOT / "persona", "us.anthropic.claude-opus-5"),
    "claude-sonnet-5": (BEDROCK_ROOT / "persona", "us.anthropic.claude-sonnet-5"),
}
# Neutral/no-persona lane per checkpoint, used as the "what does this model do when
# nobody tells it who to be" anchor at the left of the gradient. Every checkpoint on
# the figure now has one (GPT-5.6 Luna/Terra gained theirs in the latest batch), so
# the persona swing can be read against a real un-framed baseline rather than against
# R1 -- which is itself already an extreme (most risk-averse) framing.
NEUTRAL_INPUTS = {
    "gpt-5-nano": ["results/frontier/openai/baseline/gpt-5-nano", "results/frontier/openai/persona/R0_neutral/gpt-5-nano"],
    "gpt-5.4-nano": ["results/frontier/openai/baseline/gpt-5.4-nano", "results/frontier/openai/persona/R0_neutral/gpt-5.4-nano"],
    "google/gemini-3-flash-preview": [
        "results/frontier/baseline/google-gemini-3-flash-preview",
        "results/frontier/persona/R0_neutral/google-gemini-3-flash-preview",
    ],
    "gpt-5.6-luna": ["results/frontier/bedrock_mantle/luna/baseline/openai.gpt-5.6-luna",
                      "results/frontier/bedrock_mantle/luna/persona/R0_neutral/openai.gpt-5.6-luna"],
    "gpt-5.6-terra": ["results/frontier/bedrock_mantle/terra/baseline/openai.gpt-5.6-terra",
                       "results/frontier/bedrock_mantle/terra/persona/R0_neutral/openai.gpt-5.6-terra"],
    "claude-opus-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-opus-5",
                       "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-opus-5"],
    "claude-sonnet-5": ["results/frontier/bedrock/baseline/us.anthropic.claude-sonnet-5",
                         "results/frontier/bedrock/persona/R0_neutral/us.anthropic.claude-sonnet-5"],
}


def setup_plot() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titleweight": "bold",
        "axes.titlesize": 13, "axes.labelsize": 10, "axes.edgecolor": PALETTE["grid"],
        "axes.linewidth": 0.8, "xtick.color": PALETTE["slate"], "ytick.color": PALETTE["slate"],
        "text.color": PALETTE["navy"], "figure.facecolor": "white", "axes.facecolor": "white",
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_new_model_gradient(model_key: str) -> dict[str, tuple[float, int]]:
    sweep_root, model_name = SWEEP_ROOTS[model_key]
    agg: dict[str, list[float]] = {f"R{i}": [0.0, 0] for i in range(1, 7)}
    for cell in sorted((sweep_root / "risk_matrix").glob("*")):
        p = cell / model_name / "players.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                role = row["persona_role"]
                if role not in ROLE_LABEL:
                    continue
                r = ROLE_LABEL[role]
                agg[r][0] += float(row["unsafe_frequency"])
                agg[r][1] += 1
    return {r: (s / n, n) for r, (s, n) in agg.items()}


def load_new_model_by_risk(model_key: str) -> dict[str, dict[float, tuple[float, int]]]:
    sweep_root, model_name = SWEEP_ROOTS[model_key]
    agg: dict[str, dict[float, list[float]]] = {f"R{i}": {0.1: [0.0, 0], 0.6: [0.0, 0], 0.9: [0.0, 0]} for i in range(1, 7)}
    for cell in sorted((sweep_root / "risk_matrix").glob("*")):
        p = cell / model_name / "players.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                role = row["persona_role"]
                if role not in ROLE_LABEL:
                    continue
                r = ROLE_LABEL[role]
                risk = float(row["max_private_risk"])
                agg[r][risk][0] += float(row["unsafe_frequency"])
                agg[r][risk][1] += 1
    return {r: {risk: (s / n, n) for risk, (s, n) in cells.items()} for r, cells in agg.items()}


def load_neutral(model: str) -> tuple[float, int]:
    """Mean player-level Unsafe rate in the neutral/no-persona lane."""
    total, n = 0.0, 0
    for d in NEUTRAL_INPUTS[model]:
        p = ROOT / d / "players.csv"
        if not p.exists():
            continue
        with open(p) as f:
            for row in csv.DictReader(f):
                total += float(row["unsafe_frequency"])
                n += 1
    return (total / n, n) if n else (float("nan"), 0)


def load_single_persona(model_key: str, persona: str) -> tuple[float, int]:
    sweep_root, model_name = SWEEP_ROOTS[model_key]
    p = sweep_root / persona / model_name / "players.csv"
    with open(p) as f:
        rows = list(csv.DictReader(f))
    rate = sum(float(r["unsafe_frequency"]) for r in rows) / len(rows)
    return rate, len(rows)


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    original = pd.read_csv(ORIGINAL_GRADIENT_CSV)
    gradient: dict[str, dict[str, tuple[float, int]]] = {}
    for model in ["gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview"]:
        sub = original[original["model"] == model]
        gradient[model] = {row.role: (row.mean_unsafe_rate, int(row.n_players)) for row in sub.itertuples()}
    for model_key in SWEEP_ROOTS:
        gradient[model_key] = load_new_model_gradient(model_key)

    neutral = {m: load_neutral(m) for m in MODEL_ORDER}
    for model, (rate, n) in neutral.items():
        gradient[model]["none"] = (rate, n)
        print(f"{model}: neutral lane mean_unsafe={rate:.4f} n={n}")

    rows = []
    for model, roles in gradient.items():
        for role, (rate, n) in roles.items():
            rows.append({"model": model, "role": role, "mean_unsafe_rate": rate, "n_players": n})
    pd.DataFrame(rows).to_csv(DATA / "persona_role_gradient_extended.csv", index=False)

    by_risk = {m: load_new_model_by_risk(m) for m in SWEEP_ROOTS}
    risk_rows = []
    for model, roles in by_risk.items():
        for role, cells in roles.items():
            for risk, (rate, n) in cells.items():
                risk_rows.append({"model": model, "role": role, "max_private_risk": risk, "mean_unsafe_rate": rate, "n_players": n})
    pd.DataFrame(risk_rows).to_csv(DATA / "persona_by_risk_extended.csv", index=False)

    single_rows = []
    for model_key in SWEEP_ROOTS:
        for persona in ["Rminus_risk_averse", "Rplus_risk_seeking"]:
            rate, n = load_single_persona(model_key, persona)
            single_rows.append({"model": model_key, "persona": persona, "mean_unsafe_rate": rate, "n_players": n})
            print(model_key, persona, f"mean_unsafe={rate:.4f}", f"n={n}")
    pd.DataFrame(single_rows).to_csv(DATA / "single_persona_extended.csv", index=False)

    # --- Figure 1: persona gradient, faceted by model family ---
    setup_plot()
    roles_order = ["none"] + [f"R{i}" for i in range(1, 7)]
    x = list(range(len(roles_order)))
    fig, axes = plt.subplots(1, len(GRADIENT_FACETS), figsize=(13.6, 5.8), sharey=True)
    for ax, (facet_label, members) in zip(axes, GRADIENT_FACETS):
        for color, model in zip(FACET_COLORS, members):
            cells = gradient[model]
            ys = [100 * cells[r][0] for r in roles_order]
            ax.plot(x, ys, marker="o", linewidth=2.4, markersize=7, color=color,
                    label=MODEL_LABELS[model])
            # Ring the un-framed anchor so the eye separates "what it does by
            # default" from "what framing does to it".
            ax.plot(0, ys[0], marker="o", markersize=12, markerfacecolor="none",
                    markeredgecolor=color, markeredgewidth=1.6, zorder=5)
        ax.axvline(0.5, color=PALETTE["navy"], linewidth=1.1, linestyle=(0, (4, 3)), alpha=0.55)
        ax.set_title(facet_label, fontsize=12)
        ax.set_xticks(x, ["none\n(neutral)"] + [f"R{i}" for i in range(1, 7)])
        ax.set_ylim(0, 105)
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False, loc="upper left", fontsize=8.8)
    axes[0].set_ylabel("Mean player-level Unsafe rate (%)")
    fig.suptitle("Every checkpoint follows the same persona/framing gradient -- across three model families\n"
                  "and three backend routes -- but each starts from a very different un-framed default",
                  fontsize=13.5, y=1.02)
    fig.supxlabel("Assigned seat risk-attitude persona (none = un-framed baseline; R1 = most risk-averse ... R6 = most risk-seeking)",
                   fontsize=10, y=-0.02)
    save_figure(fig, FIGURES / "persona_role_gradient_extended")
    print("wrote", FIGURES / "persona_role_gradient_extended.png")

    # --- Figure 2: within-role risk sensitivity, old generation vs new ---
    setup_plot()
    sweep_models = list(SWEEP_ROOTS)
    fig, axes = plt.subplots(1, len(sweep_models), figsize=(3.5 * len(sweep_models) + 1.6, 5.2), sharey=True)
    persona_roles_order = [f"R{i}" for i in range(1, 7)]  # by_risk has no "none" anchor
    for ax, model_key in zip(axes, sweep_models):
        cells = by_risk[model_key]
        for i, role in enumerate(persona_roles_order):
            risk_cells = cells[role]
            xs = sorted(risk_cells)
            ys = [100 * risk_cells[r][0] for r in xs]
            ax.plot(xs, ys, marker="o", linewidth=1.8, markersize=5,
                     color=plt.cm.viridis(i / 5), label=role)
        ax.set_title(MODEL_LABELS[model_key].split(" (")[0], fontsize=11)
        ax.set_xlabel("Maximum private setback risk")
        ax.set_xticks([0.1, 0.6, 0.9], ["10%", "60%", "90%"])
        ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Mean player-level Unsafe rate (%)")
    axes[-1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, title="Persona")
    fig.suptitle("Do the newer checkpoints keep a real risk effect underneath the persona framing?", y=1.04, fontsize=13.5)
    save_figure(fig, FIGURES / "within_role_risk_sensitivity")
    print("wrote", FIGURES / "within_role_risk_sensitivity.png")


if __name__ == "__main__":
    main()
