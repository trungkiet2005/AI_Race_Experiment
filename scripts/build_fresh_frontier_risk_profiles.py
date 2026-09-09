#!/usr/bin/env python3
"""Build a fresh multi-model risk-response figure from validated baseline logs."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd

try:
    from scripts.publication_style import (
        CATEGORICAL,
        MUTED,
        configure_publication_style,
        panel_label,
        save_publication_figure,
        style_axis,
    )
except ModuleNotFoundError:
    from publication_style import (
        CATEGORICAL,
        MUTED,
        configure_publication_style,
        panel_label,
        save_publication_figure,
        style_axis,
    )


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "results/kaggle-benchmarks/frontier_full_20260908/derived/ai_race_analysis/player_metrics.csv"
OUTPUT = ROOT / "figures/paper/frontier_many_model_risk_profiles"
PROVENANCE = ROOT / "results/kaggle-benchmarks/frontier_full_20260908/derived/frontier_many_model_risk_profiles.json"

MODEL_ORDER = [
    "google-gemini-3-flash-preview",
    "openai-gpt-5.4-nano-2026-03-17",
    "openai-gpt-5.4-mini-2026-03-17",
    "openai-gpt-5.4-2026-03-05",
    "openai-gpt-5.5-2026-04-23",
]
MODEL_LABELS = {
    "google-gemini-3-flash-preview": "Gemini 3 Flash",
    "openai-gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
    "openai-gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "openai-gpt-5.4-2026-03-05": "GPT-5.4",
    "openai-gpt-5.5-2026-04-23": "GPT-5.5",
}
RISK_ORDER = [0.1, 0.6, 0.9]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_frontier_risk_profiles() -> list[Path]:
    if not INPUT.is_file():
        raise FileNotFoundError(INPUT)
    frame = pd.read_csv(INPUT)
    required = {"model", "max_private_risk", "unsafe_rate", "run_status", "run_phase"}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    frame = frame[
        frame["run_status"].eq("completed")
        & frame["run_phase"].eq("confirmatory")
        & frame["model"].isin(MODEL_ORDER)
    ].copy()
    if set(frame["model"]) != set(MODEL_ORDER):
        raise RuntimeError("Fresh figure requires all five validated model routes")
    counts = frame.groupby(["model", "max_private_risk"]).size()
    if counts.min() != 20 or counts.max() != 20:
        raise RuntimeError("Expected 20 player trajectories per model and risk level")

    configure_publication_style()
    fig, ax = plt.subplots(figsize=(7.10, 3.25))
    x = np.arange(len(RISK_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        sub = frame[frame["model"].eq(model)]
        means = []
        lows = []
        highs = []
        for risk in RISK_ORDER:
            values = sub.loc[sub["max_private_risk"].eq(risk), "unsafe_rate"].to_numpy(float)
            mean = float(values.mean())
            se = float(values.std(ddof=1) / np.sqrt(len(values)))
            means.append(mean)
            lows.append(max(0.0, mean - 1.96 * se))
            highs.append(min(1.0, mean + 1.96 * se))
        colour = CATEGORICAL[index]
        ax.plot(x, means, marker="o", markersize=5.2, linewidth=1.7, color=colour,
                label=MODEL_LABELS[model], zorder=3)
        ax.fill_between(x, lows, highs, color=colour, alpha=0.10, linewidth=0, zorder=1)
    ax.set_xticks(x, ["10%", "60%", "90%"])
    ax.set_xlabel("Maximum private setback risk")
    ax.set_ylabel("Unsafe choice rate")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.22),
              handlelength=1.8, columnspacing=1.2)
    panel_label(ax, "A", "Fresh five-model baseline")
    ax.text(0.0, -0.31, "Points show player-level means; ribbons are 95% normal intervals (n = 20 players per cell).",
            transform=ax.transAxes, fontsize=8.0, color=MUTED, va="top")
    style_axis(ax)
    outputs = save_publication_figure(fig, OUTPUT, formats=("pdf", "png", "svg"))
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    PROVENANCE.write_text(
        json.dumps(
            {
                "figure": "frontier_many_model_risk_profiles",
                "evidence_class": "diagnostic",
                "status_note": "Fresh confirmatory baseline artifacts are valid; admission output retrieval is incomplete for this campaign.",
                "source": str(INPUT.relative_to(ROOT)).replace("\\", "/"),
                "source_sha256": sha256(INPUT),
                "models": MODEL_ORDER,
                "model_count": len(MODEL_ORDER),
                "risk_levels": RISK_ORDER,
                "players_per_cell": 20,
                "outputs": [str(path.relative_to(ROOT)).replace("\\", "/") for path in outputs],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    return outputs


def main() -> None:
    outputs = build_frontier_risk_profiles()
    print(json.dumps({"models": len(MODEL_ORDER), "outputs": [str(p) for p in outputs], "provenance": str(PROVENANCE)}))


if __name__ == "__main__":
    main()
