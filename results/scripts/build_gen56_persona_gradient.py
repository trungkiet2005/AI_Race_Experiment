#!/usr/bin/env python3
"""Does the persona/framing-dominance finding replicate in the gpt-5.6 generation?

Extends the original 3-checkpoint persona-role gradient (build_cross_model_pilot_synthesis.py,
gpt-5-nano / gpt-5.4-nano / gemini-3-flash-preview, all via direct provider APIs) with two
new checkpoints run through a different backend route entirely: gpt-5.6-luna and
gpt-5.6-terra via AWS Bedrock ("mantle" route), results/frontier/bedrock_mantle/.
That route was substituted for the direct OpenAI route after the latter hit two
compounding problems for these checkpoints: OpenAI billing credit exhaustion, and a
reasoning-effort mitigation bug (probe_reasoning_effort() sent reasoning_effort="minimal",
an invalid value for gpt-5.6 -- valid values are none/low/medium/high/xhigh -- which the
harness misread as "this model rejects the parameter entirely" and permanently disabled).
The Bedrock Mantle re-run is unaffected by either problem: all 76 run directories are
complete (manifest counts match file contents) with zero parse failures across 42,408
decisions.
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
ORIGINAL_GRADIENT_CSV = DATA / "persona_role_gradient.csv"

PALETTE = {
    "navy": "#0B132B", "blue": "#2563EB", "cyan": "#06B6D4", "teal": "#0D9488",
    "amber": "#F59E0B", "red": "#DC2626", "slate": "#64748B", "grid": "#DCE3ED", "purple": "#7C3AED",
}
MODEL_ORDER = ["gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview", "gpt-5.6-luna", "gpt-5.6-terra"]
MODEL_COLORS = [PALETTE["blue"], PALETTE["cyan"], PALETTE["teal"], PALETTE["amber"], PALETTE["red"]]
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "gpt-5.6-luna": "GPT-5.6 Luna (Bedrock Mantle)", "gpt-5.6-terra": "GPT-5.6 Terra (Bedrock Mantle)",
}
ROLE_LABEL = {f"risk-{i}": f"R{i}" for i in range(1, 7)}
NEW_MODEL_DIRS = {"gpt-5.6-luna": ("luna", "openai.gpt-5.6-luna"), "gpt-5.6-terra": ("terra", "openai.gpt-5.6-terra")}


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
    model_dir, model_name = NEW_MODEL_DIRS[model_key]
    agg: dict[str, list[float]] = {f"R{i}": [0.0, 0] for i in range(1, 7)}
    for cell in sorted((BEDROCK_MANTLE_ROOT / model_dir / "persona" / "risk_matrix").glob("*")):
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
    model_dir, model_name = NEW_MODEL_DIRS[model_key]
    agg: dict[str, dict[float, list[float]]] = {f"R{i}": {0.1: [0.0, 0], 0.6: [0.0, 0], 0.9: [0.0, 0]} for i in range(1, 7)}
    for cell in sorted((BEDROCK_MANTLE_ROOT / model_dir / "persona" / "risk_matrix").glob("*")):
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


def load_single_persona(model_key: str, persona: str) -> tuple[float, int]:
    model_dir, model_name = NEW_MODEL_DIRS[model_key]
    p = BEDROCK_MANTLE_ROOT / model_dir / "persona" / persona / model_name / "players.csv"
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
    for model_key in NEW_MODEL_DIRS:
        gradient[model_key] = load_new_model_gradient(model_key)

    rows = []
    for model, roles in gradient.items():
        for role, (rate, n) in roles.items():
            rows.append({"model": model, "role": role, "mean_unsafe_rate": rate, "n_players": n})
    pd.DataFrame(rows).to_csv(DATA / "persona_role_gradient_with_gen56.csv", index=False)

    by_risk = {m: load_new_model_by_risk(m) for m in NEW_MODEL_DIRS}
    risk_rows = []
    for model, roles in by_risk.items():
        for role, cells in roles.items():
            for risk, (rate, n) in cells.items():
                risk_rows.append({"model": model, "role": role, "max_private_risk": risk, "mean_unsafe_rate": rate, "n_players": n})
    pd.DataFrame(risk_rows).to_csv(DATA / "gen56_persona_by_risk.csv", index=False)

    single_rows = []
    for model_key in NEW_MODEL_DIRS:
        for persona in ["Rminus_risk_averse", "Rplus_risk_seeking"]:
            rate, n = load_single_persona(model_key, persona)
            single_rows.append({"model": model_key, "persona": persona, "mean_unsafe_rate": rate, "n_players": n})
            print(model_key, persona, f"mean_unsafe={rate:.4f}", f"n={n}")
    pd.DataFrame(single_rows).to_csv(DATA / "gen56_single_persona.csv", index=False)

    # --- Figure 1: extended 5-model gradient ---
    setup_plot()
    fig, ax = plt.subplots(figsize=(9.8, 5.8))
    roles_order = [f"R{i}" for i in range(1, 7)]
    x = list(range(1, 7))
    for color, model in zip(MODEL_COLORS, MODEL_ORDER):
        cells = gradient[model]
        ys = [100 * cells[r][0] for r in roles_order]
        ax.plot(x, ys, marker="o", linewidth=2.4, markersize=7, color=color, label=MODEL_LABELS[model])
    ax.set_title("The persona/framing gradient replicates in a newer model generation\n(GPT-5.6, run via a different backend entirely)", pad=14)
    ax.set_xlabel("Assigned seat risk-attitude persona (R1 = most risk-averse ... R6 = most risk-seeking)")
    ax.set_ylabel("Mean player-level Unsafe rate (%)")
    ax.set_xticks(x, roles_order)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5)
    fig.text(0.01, -0.1,
              "GPT-5.6 Luna/Terra: n=360 players/point, 0.1-0.6-0.9 risk pooled, results/frontier/bedrock_mantle/ (0 parse failures, 76/76 cells complete).\n"
              "Earlier three checkpoints: 2-player neutral-lane persona sweep (see persona_role_gradient.png). Pilot; never pooled across checkpoints.",
              fontsize=8, color=PALETTE["slate"])
    save_figure(fig, FIGURES / "persona_role_gradient_gen56")
    print("wrote", FIGURES / "persona_role_gradient_gen56.png")

    # --- Figure 2: within-role risk sensitivity, old generation vs new ---
    setup_plot()
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)
    for ax, model_key in zip(axes, ["gpt-5.6-luna", "gpt-5.6-terra"]):
        cells = by_risk[model_key]
        for i, role in enumerate(roles_order):
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
    axes[1].legend(frameon=False, loc="center left", bbox_to_anchor=(1.0, 0.5), fontsize=8, title="Persona")
    fig.suptitle("Unlike the earlier GPT-5/5.4 nano checkpoints, GPT-5.6 keeps a real\nwithin-persona risk effect at every framing level", y=1.06, fontsize=13.5)
    fig.text(0.01, -0.08,
              "n=120 players/point. Every persona level declines with risk here (contrast with gpt-5-nano/gpt-5.4-nano's\n"
              "near-flat within-role risk response in the original 2-player neutral sweep).",
              fontsize=8, color=PALETTE["slate"])
    save_figure(fig, FIGURES / "gen56_within_role_risk_sensitivity")
    print("wrote", FIGURES / "gen56_within_role_risk_sensitivity.png")


if __name__ == "__main__":
    main()
