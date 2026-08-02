#!/usr/bin/env python3
"""Prospective power sensitivity for the context x mapping CRN design.

The completed parity-confounded pilot cannot identify the future mapping
interaction variance.  This script therefore uses the largest observed
repetition-level context-delta spread as an explicit conservative variance
proxy and reports a grid rather than pretending it is a direct estimate.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from results.scripts.analyze_context_skin import (
    FIXED_SCHEMA,
    LIVE_SCHEMA,
    _paired_rows,
    discover_runs,
    validate_and_load,
)


SEED = 260802
HOLM_FAMILY = 7
ALPHA = 0.05 / HOLM_FAMILY
DEFAULT_BLOCKS = (16, 32, 64, 96, 128, 160, 192)
DEFAULT_EFFECTS = (0.05, 0.10, 0.15, 0.20)


def variance_proxy(live_root: Path, fixed_root: Path) -> tuple[np.ndarray, pd.DataFrame]:
    data = validate_and_load(
        discover_runs([live_root], LIVE_SCHEMA),
        discover_runs([fixed_root], FIXED_SCHEMA),
    )
    keys = ["max_private_risk", "rep", "player_index", "mapping_id"]
    full = data.live_turns.groupby(keys + ["skin_id"], as_index=False).agg(
        unsafe=("unsafe", "mean"), parse_failed=("parse_failed", "max")
    )
    summaries = []
    residuals: dict[str, np.ndarray] = {}
    for context in sorted(set(full["skin_id"]) - {"abstract_contest"}):
        paired = _paired_rows(
            full, context, "abstract_contest", keys, "unsafe"
        )
        by_rep = paired.groupby("rep", observed=True)["difference"].mean()
        centered = (by_rep - by_rep.mean()).to_numpy(float)
        residuals[context] = centered
        summaries.append(
            {
                "context": context,
                "n_repetition_streams": len(centered),
                "mean_context_delta": float(by_rep.mean()),
                "sd_context_delta": float(by_rep.std(ddof=1)),
            }
        )
    frame = pd.DataFrame(summaries).sort_values("sd_context_delta", ascending=False)
    calibration = str(frame.iloc[0]["context"])
    return residuals[calibration], frame


def simulate(
    residuals: np.ndarray,
    *,
    block_grid: tuple[int, ...],
    effects: tuple[float, ...],
    simulations: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    for blocks in block_grid:
        sampled = rng.choice(residuals, size=(simulations, blocks), replace=True)
        standard_errors = sampled.std(axis=1, ddof=1) / np.sqrt(blocks)
        for effect in effects:
            estimates = sampled.mean(axis=1) + effect
            statistic = np.divide(
                estimates,
                standard_errors,
                out=np.zeros_like(estimates),
                where=standard_errors > 0,
            )
            p_values = 2 * stats.t.sf(np.abs(statistic), df=blocks - 1)
            rows.append(
                {
                    "n_crn_repetition_streams": blocks,
                    "true_interaction": effect,
                    "power_holm_single_step": float(np.mean(p_values < ALPHA)),
                    "monte_carlo_se": float(
                        np.sqrt(np.mean(p_values < ALPHA) * (1 - np.mean(p_values < ALPHA)) / simulations)
                    ),
                    "simulations": simulations,
                    "familywise_alpha": 0.05,
                    "holm_family_size": HOLM_FAMILY,
                    "screening_alpha": ALPHA,
                }
            )
    return pd.DataFrame(rows)


def plot_power(frame: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    colors = ("#64748b", "#2563eb", "#f59e0b", "#16a34a")
    for color, (effect, group) in zip(colors, frame.groupby("true_interaction")):
        ax.plot(
            group["n_crn_repetition_streams"],
            group["power_holm_single_step"],
            marker="o",
            linewidth=2,
            color=color,
            label=f"{effect * 100:.0f} pp interaction",
        )
    ax.axhline(0.8, color="#0f172a", linestyle="--", linewidth=1)
    ax.axvline(32, color="#94a3b8", linestyle=":", linewidth=1)
    ax.text(32, 0.04, "diagnostic grid", ha="center", color="#64748b", fontsize=9)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Independent CRN repetition streams")
    ax.set_ylabel("Estimated power")
    ax.set_title("Prospective context × mapping power sensitivity", loc="left", weight="bold")
    ax.text(
        0,
        1.02,
        "Pilot residual bootstrap; conservative variance proxy; α=0.05/7",
        transform=ax.transAxes,
        color="#64748b",
        fontsize=9,
    )
    ax.grid(alpha=0.18)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(output / "context_mapping_power.png", dpi=220, bbox_inches="tight")
    fig.savefig(output / "context_mapping_power.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--live-root", type=Path, required=True)
    parser.add_argument("--fixed-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--simulations", type=int, default=10_000)
    args = parser.parse_args()
    if args.simulations < 1_000:
        raise ValueError("At least 1,000 simulations are required")
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    residuals, proxy = variance_proxy(args.live_root.resolve(), args.fixed_root.resolve())
    power = simulate(
        residuals,
        block_grid=DEFAULT_BLOCKS,
        effects=DEFAULT_EFFECTS,
        simulations=args.simulations,
    )
    proxy.to_csv(output / "variance_proxy.csv", index=False)
    power.to_csv(output / "power_grid.csv", index=False)
    plot_power(power, output)
    recommendation = {
        "schema_version": "ai-race-context-mapping-power-sensitivity-v1",
        "status": "complete",
        "evidence_class": "prospective_design_sensitivity_not_behavioral_result",
        "seed": SEED,
        "simulations": args.simulations,
        "variance_proxy_context": str(proxy.iloc[0]["context"]),
        "variance_proxy_sd": float(proxy.iloc[0]["sd_context_delta"]),
        "se_soi": 0.15,
        "target_power": 0.80,
        "frozen_confirmatory_blocks": 96,
        "diagnostic_blocks": 32,
        "decision": (
            "Keep the 32-block run diagnostic. A future confirmatory replication "
            "uses 96 independent repetition streams to target a 15 pp interaction; "
            "re-estimation or optional continuation from the diagnostic result is prohibited."
        ),
        "limitations": [
            "The source pilot confounds mapping with repetition parity.",
            "Residual spread is a conservative context-delta proxy, not an identified mapping-interaction variance.",
            "The Monte Carlo rejection rule uses a multiplicity-adjusted t statistic; the final analysis remains the frozen sign-flip/bootstrap analysis.",
        ],
    }
    (output / "design_decision.json").write_text(
        json.dumps(recommendation, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(recommendation, indent=2))


if __name__ == "__main__":
    main()
