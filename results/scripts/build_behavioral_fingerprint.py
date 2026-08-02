"""Build a cross-checkpoint behavioral fingerprint from frozen synthesis tables.

The figure intentionally keeps estimands separate: risk response, conditional
reciprocity, persona swing, realized-payoff correlation, and canonical-strategy
fit are descriptive diagnostics and are not combined into a composite score.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
FIGURES = ROOT / "results" / "cross_model_pilot_synthesis" / "figures"
OUTPUT = DATA / "behavioral_fingerprint.csv"

DISPLAY = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
}


def load() -> pd.DataFrame:
    risk = pd.read_csv(DATA / "risk_response_by_model.csv")
    pivot = risk.pivot(index="model", columns="max_private_risk", values="mean_unsafe_rate")

    persona = pd.read_csv(DATA / "persona_role_gradient_extended.csv")
    persona = persona[persona["role"].isin(["R1", "R6"])].pivot(
        index="model", columns="role", values="mean_unsafe_rate"
    )

    reciprocity = json.loads((DATA / "within_risk_reciprocity.json").read_text())
    payoff = json.loads((DATA / "payoff_welfare_correlations.json").read_text())
    strategy = pd.read_csv(DATA / "canonical_strategy_classification.csv")

    rows = []
    for model in DISPLAY:
        effects = [
            cell["effect_pp"]
            for cell in reciprocity.get(model, {}).values()
            if cell.get("status") == "ok" and np.isfinite(cell.get("effect_pp", np.nan))
        ]
        model_strategy = strategy[strategy["population"] == model]
        beats = 100 * model_strategy["beats_chance"].mean() if len(model_strategy) else np.nan
        rows.append(
            {
                "model": model,
                "display_model": DISPLAY[model],
                "risk_response_pp": 100 * (pivot.loc[model, 0.9] - pivot.loc[model, 0.1]),
                "reciprocity_median_pp": float(np.median(effects)) if effects else np.nan,
                "persona_swing_pp": (
                    100 * (persona.loc[model, "R6"] - persona.loc[model, "R1"])
                    if model in persona.index
                    else np.nan
                ),
                "realized_payoff_r": payoff.get(f"{model}_true", {}).get("pearson_r", np.nan),
                "canonical_fit_above_chance_pct": beats,
            }
        )
    return pd.DataFrame(rows)


def render(df: pd.DataFrame) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="white", context="paper")
    fig, axes = plt.subplots(
        1, 2, figsize=(11.2, 5.8), gridspec_kw={"width_ratios": [1.45, 1]}, constrained_layout=True
    )

    left = df.set_index("display_model")[["risk_response_pp", "reciprocity_median_pp", "persona_swing_pp"]]
    left.columns = [
        "Risk response\nP(U|0.9) − P(U|0.1)",
        "Within-risk reciprocity\nmedian ΔP(U)",
        "Persona swing\nP(U|R6) − P(U|R1)",
    ]
    sns.heatmap(
        left,
        ax=axes[0],
        cmap=sns.diverging_palette(25, 240, as_cmap=True),
        center=0,
        vmin=-100,
        vmax=100,
        annot=True,
        fmt=".0f",
        linewidths=0.8,
        linecolor="white",
        cbar_kws={"label": "Percentage-point contrast", "shrink": 0.72},
    )
    axes[0].set_title("A. Behavioral contrasts are not one-dimensional", loc="left", weight="bold")
    axes[0].set_xlabel("")
    axes[0].set_ylabel("")
    axes[0].tick_params(axis="x", rotation=0)

    y = np.arange(len(df))
    axes[1].axvline(0, color="#5b6472", lw=1, zorder=0)
    axes[1].scatter(df["realized_payoff_r"], y, s=52, color="#286f9b", label="Unsafe–realized payoff r")
    ax2 = axes[1].twiny()
    ax2.scatter(
        df["canonical_fit_above_chance_pct"], y,
        s=52, marker="s", facecolors="white", edgecolors="#d17a22",
        label="Canonical fit above chance"
    )
    axes[1].set_yticks(y, labels=df["display_model"])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-1, 1)
    ax2.set_xlim(0, 10)
    axes[1].set_xlabel("Pearson r (Unsafe rate vs realized payoff)", color="#286f9b")
    ax2.set_xlabel("Player-races beating base-rate chance (%)", color="#d17a22")
    axes[1].set_title("B. Payoff coupling and strategy fit also separate", loc="left", weight="bold")
    axes[1].grid(axis="y", color="#e7e9ed", lw=0.7)
    axes[1].grid(axis="x", visible=False)
    ax2.grid(False)

    fig.suptitle(
        "Cross-checkpoint behavioral fingerprint",
        x=0.01, ha="left", fontsize=15, weight="bold"
    )
    fig.text(
        0.01, 0.005,
        "Neutral-lane pilots. Reciprocity is conditioned on risk; persona swing compares R6 with R1. "
        "Metrics remain separate diagnostics—not a composite model score.",
        fontsize=8.5, color="#4f5661"
    )
    for ext in ("png", "pdf"):
        fig.savefig(FIGURES / f"behavioral_fingerprint.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    df = load()
    df.to_csv(OUTPUT, index=False)
    render(df)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
