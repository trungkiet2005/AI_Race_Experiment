#!/usr/bin/env python3
"""Plot strict pre-action versus prompt-last activation-SAE robustness."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd


BLUE = "#2563EB"
GOLD = "#D79B00"
INK = "#172033"
GRID = "#D9E0EA"
MUTED = "#6B7280"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pre-action-summary", required=True)
    parser.add_argument("--prompt-last-summary", required=True)
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    pre = pd.read_csv(args.pre_action_summary).sort_values("layer")
    prompt = pd.read_csv(args.prompt_last_summary).sort_values("layer")
    if not pre["layer"].reset_index(drop=True).equals(prompt["layer"].reset_index(drop=True)):
        raise ValueError("capture positions do not cover identical layers")
    if set(pre["representation"]) != {"pretrained_sae"} or set(prompt["representation"]) != {"pretrained_sae"}:
        raise ValueError("unexpected representation")
    merged = pre.merge(prompt, on="layer", suffixes=("_pre_action", "_prompt_last"))
    merged["auc_delta_pre_action_minus_prompt_last"] = (
        merged["roc_auc_pre_action"] - merged["roc_auc_prompt_last"]
    )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    merged.to_csv(output_dir / "token_position_robustness.csv", index=False)

    layers = merged["layer"]
    figure, axis = plt.subplots(figsize=(8.2, 4.8))
    axis.axhline(0.5, color=INK, linestyle="--", linewidth=1, label="Chance AUC")
    axis.plot(
        layers,
        merged["roc_auc_pre_action"],
        marker="o",
        linewidth=2.2,
        color=BLUE,
        label="Before action label (after `ACTION:` prefix)",
    )
    axis.plot(
        layers,
        merged["roc_auc_prompt_last"],
        marker="s",
        linewidth=2.2,
        color=GOLD,
        label="Prompt last token (before response boilerplate)",
    )
    axis.set_title(
        "SAE action information across two causal token positions",
        loc="left",
        color=INK,
        weight="bold",
        pad=30,
    )
    axis.text(
        0,
        1.01,
        "Same 600 decisions and strict connected-component split; n=120 held-out",
        transform=axis.transAxes,
        color=MUTED,
        fontsize=9,
    )
    axis.set_xlabel("Transformer layer", color=INK)
    axis.set_ylabel("Grouped-eval ROC-AUC", color=INK)
    axis.set_xticks(layers)
    axis.set_ylim(0.48, 0.93)
    axis.spines[["top", "right"]].set_visible(False)
    axis.spines[["left", "bottom"]].set_color(GRID)
    axis.tick_params(colors=MUTED, labelsize=9)
    axis.grid(axis="y", color=GRID, linewidth=0.7)
    axis.set_axisbelow(True)
    axis.legend(
        frameon=False,
        fontsize=8,
        loc="center",
        bbox_to_anchor=(0.56, 0.20),
    )
    figure.savefig(output_dir / "sae_token_position_robustness.png", dpi=220, bbox_inches="tight", facecolor="white")
    figure.savefig(output_dir / "sae_token_position_robustness.pdf", bbox_inches="tight", facecolor="white")
    plt.close(figure)
    manifest = {
        "schema_version": "ai-race.activation-sae.position-comparison.v1",
        "status": "complete",
        "layers": [int(value) for value in layers],
        "n_eval": 120,
        "mean_auc_pre_action": float(merged["roc_auc_pre_action"].mean()),
        "mean_auc_prompt_last": float(merged["roc_auc_prompt_last"].mean()),
        "mean_auc_delta": float(merged["auc_delta_pre_action_minus_prompt_last"].mean()),
        "interpretation": "association persists before response boilerplate; no causal or same-runtime claim",
    }
    (output_dir / "token_position_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
