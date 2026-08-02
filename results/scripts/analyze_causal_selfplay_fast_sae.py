#!/usr/bin/env python3
"""Audit and visualize the actual-self-play FAST-SAE causal pilot.

The analysis deliberately keeps four evidence classes separate:

1. discovery/evaluation associations (correlation, action AUC),
2. fixed-state direct interventions on the final prompt token,
3. control comparisons and sign/dose diagnostics, and
4. live endogenous trajectory and payoff effects.

Raw experiment artifacts are read-only.  All derived tables, figures, provenance,
and the report are written below ``analysis/`` in the supplied run directory.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


BLUE = "#2563EB"
BLUE_LIGHT = "#AFC8FB"
GOLD = "#D79B00"
ORANGE = "#E8792E"
PINK = "#C65A8E"
INK = "#172033"
MUTED = "#687386"
GRID = "#D9E0EA"
PAPER = "#FBFCFE"
WHITE = "#FFFFFF"
BOOTSTRAP_SEED = 260801

CONDITION_LABEL = {
    "target_feature": "Target feature",
    "matched_random": "Matched random",
    "unrelated_feature": "Unrelated feature",
    "target_feature_ablation": "Target ablation",
    "sae_reconstruction": "SAE reconstruction",
    "zero": "Zero / baseline",
}
CONDITION_COLOR = {
    "target_feature": BLUE,
    "matched_random": GOLD,
    "unrelated_feature": ORANGE,
    "target_feature_ablation": PINK,
    "sae_reconstruction": MUTED,
    "zero": INK,
}
CONDITION_MARKER = {
    "target_feature": "o",
    "matched_random": "s",
    "unrelated_feature": "^",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "run_dir",
        type=Path,
        nargs="?",
        default=Path(
            "results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1"
        ),
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def validate_source_hashes(run_dir: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    """Verify every checksum recorded by the experiment runner."""
    expected: dict[Path, str] = {}
    for shard in manifest["stages"]["selfplay"]["shards"]:
        game_id = shard["game_id"]
        expected[run_dir / "race_shards" / f"{game_id}.json"] = shard["json_sha256"]
        expected[run_dir / "race_shards" / f"{game_id}.npz"] = shard["npz_sha256"]
    expected[run_dir / manifest["stages"]["mine"]["artifact"]] = manifest["stages"][
        "mine"
    ]["sha256"]
    for name, digest in manifest["stages"]["steer"]["artifacts"].items():
        expected[run_dir / name] = digest
    for name, digest in manifest["stages"]["steered_play"]["artifacts"].items():
        expected[run_dir / name] = digest

    failures: list[dict[str, str]] = []
    for path, expected_digest in expected.items():
        if not path.is_file():
            failures.append(
                {"path": path.relative_to(run_dir).as_posix(), "reason": "missing"}
            )
            continue
        observed = sha256_file(path)
        if observed != expected_digest:
            failures.append(
                {
                    "path": path.relative_to(run_dir).as_posix(),
                    "reason": "sha256_mismatch",
                    "expected": expected_digest,
                    "observed": observed,
                }
            )
    if failures:
        raise ValueError(f"source checksum validation failed: {failures}")
    return {"status": "pass", "n_verified_artifacts": len(expected)}


def auc_rank(y_true: np.ndarray, score: np.ndarray) -> float:
    """Binary AUC via average ranks; returns NaN for a one-class sample."""
    y = np.asarray(y_true, dtype=int)
    x = np.asarray(score, dtype=float)
    keep = np.isfinite(x)
    y, x = y[keep], x[keep]
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(x, kind="mergesort")
    sorted_x = x[order]
    ranks = np.empty(len(x), dtype=float)
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and sorted_x[end] == sorted_x[start]:
            end += 1
        ranks[order[start:end]] = (start + 1 + end) / 2.0
        start = end
    pos_rank_sum = float(ranks[y == 1].sum())
    return (pos_rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def percentile_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    low, high = np.quantile(array, [0.025, 0.975])
    return float(low), float(high)


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    repetitions: int,
    seed_offset: int = 0,
) -> tuple[float, float]:
    clusters = list(frame[cluster].drop_duplicates())
    if not clusters:
        return float("nan"), float("nan")
    by_cluster = {key: frame.loc[frame[cluster] == key, value].to_numpy(float) for key in clusters}
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    means = np.empty(repetitions, dtype=float)
    for i in range(repetitions):
        sampled = rng.choice(clusters, size=len(clusters), replace=True)
        means[i] = np.concatenate([by_cluster[key] for key in sampled]).mean()
    return percentile_ci(means)


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": PAPER,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_associations(
    run_dir: Path, mining: dict[str, Any]
) -> tuple[pd.DataFrame, pd.DataFrame]:
    selected = list(mining["selected_features"])
    unrelated = dict(mining["unrelated_control_feature"])
    selected_ids = [int(row["feature_id"]) for row in selected]
    feature_rows = selected + [unrelated]

    observations: list[dict[str, Any]] = []
    split_map = mining["race_split"]
    for meta_path in sorted((run_dir / "race_shards").glob("*.json")):
        meta = read_json(meta_path)
        npz_path = meta_path.with_suffix(".npz")
        with np.load(npz_path) as arrays:
            codes = np.asarray(arrays["sae_codes"], dtype=float)
        records = meta["records"]
        if len(records) != len(codes):
            raise ValueError(f"record/code mismatch in {meta_path.name}")
        for index, record in enumerate(records):
            row: dict[str, Any] = {
                "game_id": meta["game_id"],
                "split": split_map[meta["game_id"]],
                "action_unsafe": int(record["action"] == "unsafe"),
                "unsafe_log_odds": float(record["unsafe_log_odds"]),
            }
            for feature_id in selected_ids + [int(unrelated["feature_id"])]:
                row[f"feature_{feature_id}"] = float(codes[index, feature_id])
            observations.append(row)
    obs = pd.DataFrame(observations)

    summary: list[dict[str, Any]] = []
    for source in feature_rows:
        feature_id = int(source["feature_id"])
        selected_flag = feature_id in selected_ids
        for split in ("discovery", "eval"):
            part = obs.loc[obs["split"] == split]
            activations = part[f"feature_{feature_id}"].to_numpy(float)
            labels = part["action_unsafe"].to_numpy(int)
            log_odds = part["unsafe_log_odds"].to_numpy(float)
            corr = float(np.corrcoef(activations, log_odds)[0, 1])
            summary.append(
                {
                    "feature_id": feature_id,
                    "feature_role": "selected" if selected_flag else "unrelated_control",
                    "split": split,
                    "n_decisions": len(part),
                    "unsafe_decisions": int(labels.sum()),
                    "corr_unsafe_log_odds": corr,
                    "auc_unsafe_action": auc_rank(labels, activations),
                    "auc_oriented": max(auc_rank(labels, activations), 1 - auc_rank(labels, activations)),
                    "activation_prevalence": float(np.mean(activations != 0)),
                }
            )
    return pd.DataFrame(summary), obs


def summarize_fixed(
    steering: pd.DataFrame, repetitions: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups = ["condition", "target_feature_id", "alpha"]
    summaries: list[dict[str, Any]] = []
    for keys, part in steering.groupby(groups, dropna=False, sort=True):
        condition, feature, alpha = keys
        lo, hi = cluster_bootstrap_mean(
            part, "delta_unsafe_log_odds", "game_id", repetitions, len(summaries)
        )
        summaries.append(
            {
                "condition": condition,
                "target_feature_id": feature,
                "alpha": alpha,
                "n_decisions": len(part),
                "n_races": part["game_id"].nunique(),
                "mean_delta_unsafe_log_odds": part["delta_unsafe_log_odds"].mean(),
                "ci95_low": lo,
                "ci95_high": hi,
                "median_delta_unsafe_log_odds": part["delta_unsafe_log_odds"].median(),
                "mean_abs_delta_unsafe_log_odds": part["delta_unsafe_log_odds"].abs().mean(),
                "action_flip_rate": part["action_flipped"].mean(),
            }
        )
    summary = pd.DataFrame(summaries)

    controls: list[dict[str, Any]] = []
    target = steering.loc[steering["condition"] == "target_feature"]
    for control_name in ("matched_random", "unrelated_feature"):
        control = steering.loc[steering["condition"] == control_name]
        paired = target.merge(
            control,
            on=["sample_index", "game_id", "target_feature_id", "alpha"],
            suffixes=("_target", "_control"),
            validate="one_to_one",
        )
        paired["target_minus_control_delta"] = (
            paired["delta_unsafe_log_odds_target"] - paired["delta_unsafe_log_odds_control"]
        )
        paired["target_minus_control_abs_delta"] = (
            paired["delta_unsafe_log_odds_target"].abs()
            - paired["delta_unsafe_log_odds_control"].abs()
        )
        paired["target_minus_control_flip"] = (
            paired["action_flipped_target"] - paired["action_flipped_control"]
        )
        for (feature, alpha), part in paired.groupby(["target_feature_id", "alpha"]):
            lo, hi = cluster_bootstrap_mean(
                part,
                "target_minus_control_delta",
                "game_id",
                repetitions,
                1000 + len(controls),
            )
            abs_lo, abs_hi = cluster_bootstrap_mean(
                part,
                "target_minus_control_abs_delta",
                "game_id",
                repetitions,
                2000 + len(controls),
            )
            controls.append(
                {
                    "control": control_name,
                    "target_feature_id": int(feature),
                    "alpha": float(alpha),
                    "n_decisions": len(part),
                    "n_races": part["game_id"].nunique(),
                    "mean_target_minus_control_delta": part[
                        "target_minus_control_delta"
                    ].mean(),
                    "delta_ci95_low": lo,
                    "delta_ci95_high": hi,
                    "mean_target_minus_control_abs_delta": part[
                        "target_minus_control_abs_delta"
                    ].mean(),
                    "abs_delta_ci95_low": abs_lo,
                    "abs_delta_ci95_high": abs_hi,
                    "mean_target_minus_control_flip_rate": part[
                        "target_minus_control_flip"
                    ].mean(),
                }
            )
    contrast = pd.DataFrame(controls)

    sign_rows: list[dict[str, Any]] = []
    for (condition, feature), part in steering.loc[
        steering["condition"].isin(["target_feature", "matched_random", "unrelated_feature"])
    ].groupby(["condition", "target_feature_id"]):
        per_alpha = part.groupby("alpha")["delta_unsafe_log_odds"].mean().sort_index()
        x = per_alpha.index.to_numpy(float)
        y = per_alpha.to_numpy(float)
        slope = float(np.polyfit(x, y, 1)[0])
        odd = {float(k): float(v) for k, v in per_alpha.items()}
        antisymmetry_error = abs(odd.get(2.0, math.nan) + odd.get(-2.0, math.nan))
        sign_rows.append(
            {
                "condition": condition,
                "target_feature_id": int(feature),
                "linear_slope_per_alpha": slope,
                "antisymmetry_error_at_abs_alpha_2": antisymmetry_error,
                "positive_minus_negative_at_abs_alpha_2": odd.get(2.0, math.nan)
                - odd.get(-2.0, math.nan),
            }
        )
    return summary, contrast, pd.DataFrame(sign_rows)


def flatten_live(records: list[dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for record in records:
        condition = record["condition"]
        decisions = record["decisions"]
        comparable = [d for d in decisions if d["state_comparable_to_zero"]]
        result = record["engine_result"]
        action_sequence = ";".join(d["action"] for d in decisions)
        rows.append(
            {
                "condition_id": condition["condition_id"],
                "condition": condition["condition"],
                "target_feature_id": condition["target_feature_id"],
                "alpha": condition["alpha"],
                "game_id": record["game_id"],
                "treatment": record["treatment"],
                "game_seed": record["game_seed"],
                "n_decisions": len(decisions),
                "unsafe_rate": np.mean([d["action"] == "unsafe" for d in decisions]),
                "n_state_comparable_decisions": len(comparable),
                "comparable_flip_count": sum(
                    int(d["action_flipped_if_comparable"]) for d in comparable
                ),
                "comparable_flip_rate": (
                    np.mean([d["action_flipped_if_comparable"] for d in comparable])
                    if comparable
                    else float("nan")
                ),
                "n_rounds": result["n_rounds"],
                "total_final_payoff": sum(result["final_payoffs"]),
                "mean_player_final_payoff": np.mean(result["final_payoffs"]),
                "setback_count": sum(result["setbacks"]),
                "delta_total_final_payoff_vs_zero": record["paired_zero"][
                    "delta_total_final_payoff"
                ],
                "action_sequence_sha256": hashlib.sha256(
                    action_sequence.encode("utf-8")
                ).hexdigest(),
            }
        )
    frame = pd.DataFrame(rows)
    zero = frame.loc[frame["condition"] == "zero", ["game_id", "unsafe_rate"]].rename(
        columns={"unsafe_rate": "zero_unsafe_rate"}
    )
    frame = frame.merge(zero, on="game_id", how="left", validate="many_to_one")
    frame["delta_unsafe_rate_vs_zero"] = frame["unsafe_rate"] - frame["zero_unsafe_rate"]
    return frame


def summarize_live(
    live: pd.DataFrame, repetitions: int
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    summary_rows: list[dict[str, Any]] = []
    keys = ["condition", "target_feature_id", "alpha"]
    for group, part in live.groupby(keys, dropna=False, sort=True):
        condition, feature, alpha = group
        payoff_ci = percentile_ci(
            np.random.default_rng(BOOTSTRAP_SEED + len(summary_rows)).choice(
                part["delta_total_final_payoff_vs_zero"].to_numpy(float),
                size=(repetitions, len(part)),
                replace=True,
            ).mean(axis=1)
        )
        summary_rows.append(
            {
                "condition": condition,
                "target_feature_id": feature,
                "alpha": alpha,
                "n_races": len(part),
                "n_decisions": int(part["n_decisions"].sum()),
                "unsafe_rate": np.average(part["unsafe_rate"], weights=part["n_decisions"]),
                "mean_delta_unsafe_rate_vs_zero": part["delta_unsafe_rate_vs_zero"].mean(),
                "comparable_flip_rate": (
                    part["comparable_flip_count"].sum()
                    / part["n_state_comparable_decisions"].sum()
                ),
                "n_state_comparable_decisions": int(
                    part["n_state_comparable_decisions"].sum()
                ),
                "mean_delta_total_final_payoff_vs_zero": part[
                    "delta_total_final_payoff_vs_zero"
                ].mean(),
                "payoff_delta_ci95_low": payoff_ci[0],
                "payoff_delta_ci95_high": payoff_ci[1],
                "mean_setback_count": part["setback_count"].mean(),
            }
        )
    summary = pd.DataFrame(summary_rows)

    contrast_rows: list[dict[str, Any]] = []
    equivalence_rows: list[dict[str, Any]] = []
    target = live.loc[live["condition"] == "target_feature"]
    for control_name in ("matched_random", "unrelated_feature"):
        control = live.loc[live["condition"] == control_name]
        paired = target.merge(
            control,
            on=["game_id", "target_feature_id", "alpha"],
            suffixes=("_target", "_control"),
            validate="one_to_one",
        )
        paired["payoff_target_minus_control"] = (
            paired["delta_total_final_payoff_vs_zero_target"]
            - paired["delta_total_final_payoff_vs_zero_control"]
        )
        paired["unsafe_target_minus_control"] = (
            paired["unsafe_rate_target"] - paired["unsafe_rate_control"]
        )
        paired["same_action_sequence"] = (
            paired["action_sequence_sha256_target"]
            == paired["action_sequence_sha256_control"]
        ).astype(int)
        for (feature, alpha), part in paired.groupby(["target_feature_id", "alpha"]):
            payoff_boot = np.random.default_rng(
                BOOTSTRAP_SEED + 3000 + len(contrast_rows)
            ).choice(
                part["payoff_target_minus_control"].to_numpy(float),
                size=(repetitions, len(part)),
                replace=True,
            ).mean(axis=1)
            lo, hi = percentile_ci(payoff_boot)
            contrast_rows.append(
                {
                    "control": control_name,
                    "target_feature_id": int(feature),
                    "alpha": float(alpha),
                    "n_paired_races": len(part),
                    "mean_target_minus_control_payoff_delta": part[
                        "payoff_target_minus_control"
                    ].mean(),
                    "payoff_contrast_ci95_low": lo,
                    "payoff_contrast_ci95_high": hi,
                    "mean_target_minus_control_unsafe_rate": part[
                        "unsafe_target_minus_control"
                    ].mean(),
                    "exact_action_sequence_match_rate": part[
                        "same_action_sequence"
                    ].mean(),
                }
            )
            equivalence_rows.extend(
                {
                    "control": control_name,
                    "target_feature_id": int(feature),
                    "alpha": float(alpha),
                    "game_id": row.game_id,
                    "same_action_sequence": int(row.same_action_sequence),
                    "payoff_target_minus_control": row.payoff_target_minus_control,
                    "unsafe_target_minus_control": row.unsafe_target_minus_control,
                }
                for row in part.itertuples()
            )
    return summary, pd.DataFrame(contrast_rows), pd.DataFrame(equivalence_rows)


def association_figure(frame: pd.DataFrame, out: Path) -> None:
    selected = frame.loc[frame["feature_role"] == "selected"].copy()
    feature_ids = sorted(selected["feature_id"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.5))
    specs = [
        ("corr_unsafe_log_odds", "Pearson correlation", (-1, 1)),
        ("auc_unsafe_action", "Unsafe-action AUC", (0, 1)),
    ]
    for ax, (metric, title, limits) in zip(axes, specs):
        for i, feature in enumerate(feature_ids):
            part = selected.loc[selected["feature_id"] == feature].set_index("split")
            discovery = part.loc["discovery", metric]
            evaluation = part.loc["eval", metric]
            ax.plot([discovery, evaluation], [i, i], color=GRID, linewidth=2, zorder=1)
            ax.scatter(discovery, i, s=72, color=BLUE_LIGHT, edgecolor=BLUE, marker="o", zorder=2)
            ax.scatter(evaluation, i, s=72, color=BLUE, edgecolor=INK, marker="s", zorder=3)
        ax.set_yticks(range(len(feature_ids)), [f"Feature {f}" for f in feature_ids])
        ax.set_xlim(*limits)
        ax.set_title(title, loc="left", fontweight="bold")
        ax.axvline(0 if metric.startswith("corr") else 0.5, color=INK, linewidth=1, linestyle="--")
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("Association only; not a causal estimate")
    discovery_handle = axes[0].scatter(
        [], [], color=BLUE_LIGHT, edgecolor=BLUE, label="Discovery (12 races)"
    )
    eval_handle = axes[0].scatter(
        [], [], color=BLUE, edgecolor=INK, marker="s", label="Held-out eval (6 races)"
    )
    fig.suptitle("Selected FAST-SAE feature associations", x=0.08, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.08,
        0.91,
        "Layer 12, final prompt token; AUC direction is not re-oriented, so values below 0.5 denote inverse coding.",
        color=MUTED,
    )
    fig.legend(
        handles=[discovery_handle, eval_handle],
        loc="upper center",
        bbox_to_anchor=(0.5, 0.865),
        ncol=2,
    )
    fig.subplots_adjust(top=0.72, wspace=0.36)
    save_figure(fig, out / "association_selected_features")


def fixed_dose_figure(summary: pd.DataFrame, out: Path) -> None:
    selected = summary.loc[
        summary["condition"].isin(["target_feature", "matched_random", "unrelated_feature"])
        & summary["target_feature_id"].notna()
    ].copy()
    features = sorted(int(v) for v in selected["target_feature_id"].unique())
    fig, axes = plt.subplots(1, len(features), figsize=(14.2, 4.6), sharey=True)
    for ax, feature in zip(axes, features):
        part = selected.loc[selected["target_feature_id"] == feature]
        for condition in ("target_feature", "matched_random", "unrelated_feature"):
            line = part.loc[part["condition"] == condition].sort_values("alpha")
            ax.plot(
                line["alpha"],
                line["mean_delta_unsafe_log_odds"],
                color=CONDITION_COLOR[condition],
                marker=CONDITION_MARKER[condition],
                linewidth=2,
                markersize=6,
                label=CONDITION_LABEL[condition],
            )
            ax.fill_between(
                line["alpha"].to_numpy(float),
                line["ci95_low"].to_numpy(float),
                line["ci95_high"].to_numpy(float),
                color=CONDITION_COLOR[condition],
                alpha=0.10,
            )
        ax.axhline(0, color=INK, linewidth=1)
        ax.axvline(0, color=GRID, linewidth=1)
        ax.set_title(f"Feature {feature}", fontweight="bold")
        ax.set_xlabel("Steering alpha (feature-scale units)")
    axes[0].set_ylabel("Mean change in Unsafe log-odds")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Fixed-state steering dose response", x=0.07, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.07,
        0.91,
        "24 replay-exact decisions from 6 held-out races; bands are race-cluster bootstrap 95% intervals.",
        color=MUTED,
    )
    fig.subplots_adjust(top=0.79, right=0.84, wspace=0.18)
    save_figure(fig, out / "fixed_state_dose_response")


def fixed_control_figure(contrast: pd.DataFrame, out: Path) -> None:
    part = contrast.loc[contrast["alpha"].abs() == 2].copy()
    part["label"] = part.apply(
        lambda row: f"F{int(row.target_feature_id)}  {'+' if row.alpha > 0 else '-'}2 vs {CONDITION_LABEL[row.control]}",
        axis=1,
    )
    part = part.sort_values(["target_feature_id", "alpha", "control"]).reset_index(drop=True)
    y = np.arange(len(part))
    fig, ax = plt.subplots(figsize=(10.7, 7.0))
    colors = [GOLD if c == "matched_random" else ORANGE for c in part["control"]]
    ax.hlines(y, part["delta_ci95_low"], part["delta_ci95_high"], color=colors, linewidth=2)
    ax.scatter(part["mean_target_minus_control_delta"], y, color=colors, edgecolor=INK, s=62, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_yticks(y, part["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Target minus control: change in Unsafe log-odds")
    fig.suptitle(
        "Target-feature specificity at |alpha| = 2",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.94,
        "Intervals crossing zero do not establish target-specific influence (24 decisions; 6 race clusters).",
        color=MUTED,
    )
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(top=0.88)
    save_figure(fig, out / "fixed_state_target_minus_controls")


def fixed_diagnostic_figure(summary: pd.DataFrame, out: Path) -> None:
    conditions = [
        "target_feature",
        "matched_random",
        "unrelated_feature",
        "target_feature_ablation",
        "sae_reconstruction",
    ]
    rows: list[dict[str, Any]] = []
    for condition in conditions:
        part = summary.loc[summary["condition"] == condition]
        if condition in ("target_feature", "matched_random", "unrelated_feature"):
            part = part.loc[part["alpha"].abs() == 2]
        weights = part["n_decisions"].to_numpy(float)
        rows.append(
            {
                "condition": condition,
                "mean_abs": np.average(part["mean_abs_delta_unsafe_log_odds"], weights=weights),
                "flip_rate": np.average(part["action_flip_rate"], weights=weights),
            }
        )
    plot = pd.DataFrame(rows)
    labels = [CONDITION_LABEL[c] for c in plot["condition"]]
    y = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 5.0))
    for ax, metric, title, xlabel in [
        (axes[0], "mean_abs", "Perturbation magnitude", "Mean absolute change in Unsafe log-odds"),
        (axes[1], "flip_rate", "Discrete action instability", "Action flip rate"),
    ]:
        colors = [CONDITION_COLOR[c] for c in plot["condition"]]
        ax.barh(y, plot[metric], color=colors, edgecolor=INK, linewidth=0.7)
        ax.set_yticks(y, labels)
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel(xlabel)
        ax.grid(axis="y", visible=False)
        for yi, value in zip(y, plot[metric]):
            label = f"{value:.3f}" if metric == "mean_abs" else f"{100*value:.1f}%"
            ax.text(value, yi, f"  {label}", va="center", color=INK)
    fig.suptitle("Fixed-state intervention diagnostics", x=0.07, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.07,
        0.91,
        "Target/random/unrelated use |alpha|=2; ablation and full SAE reconstruction use their native interventions.",
        color=MUTED,
    )
    fig.subplots_adjust(top=0.78, wspace=0.55, right=0.95)
    save_figure(fig, out / "fixed_state_intervention_diagnostics")


def live_direct_figure(summary: pd.DataFrame, out: Path) -> None:
    part = summary.loc[summary["condition"].isin(["target_feature", "matched_random", "unrelated_feature"])]
    features = sorted(int(v) for v in part["target_feature_id"].dropna().unique())
    fig, axes = plt.subplots(1, len(features), figsize=(14.2, 4.5), sharey=True)
    for ax, feature in zip(axes, features):
        fp = part.loc[part["target_feature_id"] == feature]
        for condition in ("target_feature", "matched_random", "unrelated_feature"):
            cp = fp.loc[fp["condition"] == condition].sort_values("alpha")
            ax.plot(
                cp["alpha"],
                100 * cp["comparable_flip_rate"],
                color=CONDITION_COLOR[condition],
                marker=CONDITION_MARKER[condition],
                linewidth=2,
                label=CONDITION_LABEL[condition],
            )
        ax.set_title(f"Feature {feature}", fontweight="bold")
        ax.set_xticks([-2, 2])
        ax.set_xlabel("Live alpha")
        ax.set_ylim(0, max(6, 100 * part["comparable_flip_rate"].max() * 1.25))
    axes[0].set_ylabel("Comparable action flip rate (%)")
    axes[-1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Direct live action flips before state divergence", x=0.07, ha="left", fontsize=16, fontweight="bold")
    fig.text(
        0.07,
        0.91,
        "Only decisions whose complete prompt still matches zero are counted; later decisions are excluded.",
        color=MUTED,
    )
    fig.subplots_adjust(top=0.78, right=0.84, wspace=0.18)
    save_figure(fig, out / "live_direct_comparable_flips")


def live_payoff_figure(summary: pd.DataFrame, out: Path) -> None:
    part = summary.loc[summary["condition"].isin(["target_feature", "matched_random", "unrelated_feature"])]
    part = part.sort_values(["target_feature_id", "alpha", "condition"]).reset_index(drop=True)
    part["label"] = part.apply(
        lambda row: f"F{int(row.target_feature_id)} {'+' if row.alpha > 0 else '-'}2 - {CONDITION_LABEL[row.condition]}",
        axis=1,
    )
    y = np.arange(len(part))
    colors = [CONDITION_COLOR[c] for c in part["condition"]]
    fig, ax = plt.subplots(figsize=(10.8, 8.0))
    ax.hlines(y, part["payoff_delta_ci95_low"], part["payoff_delta_ci95_high"], color=colors, linewidth=2)
    ax.scatter(part["mean_delta_total_final_payoff_vs_zero"], y, color=colors, edgecolor=INK, s=58, zorder=3)
    ax.axvline(0, color=INK, linewidth=1.2)
    ax.set_yticks(y, part["label"])
    ax.invert_yaxis()
    ax.set_xlabel("Mean change in total final payoff vs zero")
    fig.suptitle(
        "Live endogenous payoff differences",
        x=0.08,
        y=0.98,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.94,
        "Six CRN-paired races per condition; intervals are race bootstrap and remain highly uncertain.",
        color=MUTED,
    )
    ax.grid(axis="y", visible=False)
    fig.subplots_adjust(top=0.91)
    save_figure(fig, out / "live_endogenous_payoff_effects")


def fmt(value: float, digits: int = 3) -> str:
    if value is None or not np.isfinite(float(value)):
        return "NA"
    return f"{float(value):.{digits}f}"


def build_report(
    manifest: dict[str, Any],
    associations: pd.DataFrame,
    fixed_summary: pd.DataFrame,
    fixed_contrast: pd.DataFrame,
    sign_summary: pd.DataFrame,
    live_summary: pd.DataFrame,
    live_contrast: pd.DataFrame,
) -> str:
    selected = associations.loc[associations["feature_role"] == "selected"]
    eval_assoc = selected.loc[selected["split"] == "eval"]
    extreme = fixed_contrast.loc[fixed_contrast["alpha"].abs() == 2]
    target_specific = extreme.loc[
        (extreme["delta_ci95_low"] > 0) | (extreme["delta_ci95_high"] < 0)
    ]
    exact_matches = live_contrast["exact_action_sequence_match_rate"].mean()
    live_target = live_summary.loc[live_summary["condition"] == "target_feature"]
    strongest = live_target.iloc[
        live_target["mean_delta_total_final_payoff_vs_zero"].abs().argmax()
    ]
    recon = fixed_summary.loc[fixed_summary["condition"] == "sae_reconstruction"].iloc[0]

    assoc_lines = "\n".join(
        f"| {int(row.feature_id)} | {row.corr_unsafe_log_odds:+.3f} | {row.auc_unsafe_action:.3f} | {row.activation_prevalence:.3f} |"
        for row in eval_assoc.itertuples()
    )
    live_lines = "\n".join(
        f"| {int(row.target_feature_id)} | {row.alpha:+.0f} | {100*row.comparable_flip_rate:.1f}% | {100*row.mean_delta_unsafe_rate_vs_zero:+.2f} pp | {row.mean_delta_total_final_payoff_vs_zero:+.2f} |"
        for row in live_target.sort_values(["target_feature_id", "alpha"]).itertuples()
    )
    live_control_lines = "\n".join(
        f"| {int(row.target_feature_id)} | {row.alpha:+.0f} | {CONDITION_LABEL[row.control]} | {100*row.exact_action_sequence_match_rate:.1f}% | {row.mean_target_minus_control_payoff_delta:+.2f} |"
        for row in live_contrast.sort_values(
            ["target_feature_id", "alpha", "control"]
        ).itertuples()
    )
    config = manifest["config"]
    return f"""# FAST-SAE actual-self-play causal audit

## Bottom line

The layer-12 FAST-SAE pilot found strong held-out **associations**, but it did **not** establish feature-specific causal control of SAFE/UNSAFE behavior. The three selected features retained large absolute held-out correlations (range {eval_assoc.corr_unsafe_log_odds.abs().min():.3f}-{eval_assoc.corr_unsafe_log_odds.abs().max():.3f}) and action-discrimination AUCs away from 0.5. However, at the configured strongest fixed-state dose (`|alpha|=2`), only {len(target_specific)} of {len(extreme)} target-minus-control contrasts had a race-cluster bootstrap interval excluding zero. This is an exploratory diagnostic with only six held-out race clusters, not a confirmatory null test.

Live target steering frequently reproduced control behavior: the mean exact target/control action-sequence match rate across feature/sign/control cells was {100*exact_matches:.1f}%. Consequently, trajectory and payoff changes cannot be attributed specifically to the selected SAE direction. The most extreme target-condition mean payoff difference was feature {int(strongest.target_feature_id)} at alpha {strongest.alpha:+.0f}: {strongest.mean_delta_total_final_payoff_vs_zero:+.2f} total payoff units versus zero across six races, with a bootstrap interval [{strongest.payoff_delta_ci95_low:+.2f}, {strongest.payoff_delta_ci95_high:+.2f}]. Controls produced changes on the same scale.

## What was actually run

- Model: `{config['model_repo']}` at revision `{config['model_revision']}`.
- SAE: `{config['sae_repo']}`, `{config['sae_id']}`, revision `{config['sae_revision']}`.
- Intervention site: layer {config['layer']} residual stream, final prompt token before any action-label token.
- Decision policy: full exact-sequence likelihood for `ACTION: SAFE` versus `ACTION: UNSAFE`, including EOS; temperature {config['decision_temperature']}.
- Self-play: {manifest['stages']['selfplay']['n_completed_races']} races, 300 decisions; split by whole race into 12 discovery and 6 held-out evaluation races.
- Fixed-state steering: {manifest['stages']['steer']['n_eval_decisions']} replay-exact decisions and {manifest['stages']['steer']['rows']} intervention rows. Maximum baseline replay error was {manifest['stages']['steer']['max_baseline_replay_error']}.
- Live steering: {manifest['stages']['steered_play']['n_trajectories']} trajectories over {manifest['stages']['steered_play']['n_unique_race_seeds']} common-random-number seeds.

## 1. Discovery and held-out association

Feature selection used discovery-race correlation with the continuous Unsafe-minus-Safe sequence log-odds. AUC is added here only as a descriptive action-discrimination metric; neither correlation nor AUC identifies a causal feature.

| Feature | Eval correlation | Eval Unsafe-action AUC | Activation prevalence |
|---:|---:|---:|---:|
{assoc_lines}

![Selected feature associations](figures/association_selected_features.png)

## 2. Fixed-state causal steering

All fixed-state rows use the identical held-out prompts and the baseline replay gate passed exactly. The intended target direction therefore has a clean *direct-effect* interpretation at this token, but specificity requires it to outperform matched-random and unrelated-feature directions.

![Fixed-state dose response](figures/fixed_state_dose_response.png)

The dose curves are not consistently monotone or antisymmetric. Target-direction slopes per alpha were {', '.join(f"F{int(r.target_feature_id)}={r.linear_slope_per_alpha:+.4f}" for r in sign_summary.loc[sign_summary.condition == 'target_feature'].itertuples())}. This weak sign/dose behavior is inconsistent with a simple one-dimensional causal controller.

![Target versus controls](figures/fixed_state_target_minus_controls.png)

Full SAE reconstruction itself changed the action on {100*recon.action_flip_rate:.1f}% of fixed prompts and had mean absolute log-odds change {recon.mean_abs_delta_unsafe_log_odds:.3f}. Because reconstruction is not behaviorally neutral, feature-ablation and SAE-space steering results require additional calibration against reconstruction artifacts.

![Intervention diagnostics](figures/fixed_state_intervention_diagnostics.png)

## 3. Live simulation: direct flips versus endogenous feedback

Only action changes observed before the first state divergence are directly comparable to the zero condition. Later decisions inherit altered histories and are trajectory effects.

| Feature | Alpha | Direct comparable flips | Unsafe-rate delta vs zero | Total-payoff delta vs zero |
|---:|---:|---:|---:|---:|
{live_lines}

![Comparable live flips](figures/live_direct_comparable_flips.png)

![Endogenous payoff effects](figures/live_endogenous_payoff_effects.png)

The live payoff chart is not a fixed-state causal estimate. It combines the initial direct perturbation, subsequent endogenous state feedback, and rare setback realization. Each point has only six races.

| Feature | Alpha | Control | Exact target/control action-sequence match | Target-minus-control payoff delta |
|---:|---:|---|---:|---:|
{live_control_lines}

Exact sequence matches are especially diagnostic here: identical action sequences under a target direction and a control direction cannot support a feature-specific behavioral interpretation, even when both differ from zero.

## 4. Claim boundary

Supported:

- The selected activations are associated with the model's Unsafe-vs-Safe score in held-out races.
- Residual-stream interventions can perturb scores and occasionally flip actions at replay-exact states.
- Interventions can alter full self-play trajectories under common random numbers.

Not supported:

- A selected SAE feature uniquely represents an Unsafe intention, safety preference, or game-theoretic strategy.
- Target-feature steering has a larger or more reliable effect than norm-matched random and unrelated-feature controls.
- Live payoff differences are stable, general, or beneficial.
- Feature semantics transfer across layers, checkpoints, contexts, or decoding policies.

## 5. Next confirmatory experiment

1. Increase held-out race clusters before increasing decision rows; uncertainty is currently race-limited.
2. Require behavioral-neutrality gates for SAE reconstruction and zero-dose hooks.
3. Select features on discovery races only, then freeze feature, sign, layer, alpha, and all controls.
4. Use multiple norm-matched random directions per feature and report the target's percentile in that empirical null.
5. Test opaque action labels and multiple context skins to separate action-token coding from strategy representation.
6. Replicate at another layer and checkpoint without re-selecting on the evaluation set.

## Reproducibility

Run from the repository root:

```bash
python results/scripts/analyze_causal_selfplay_fast_sae.py \\
  results/open_source/activation_sae/causal_selfplay/fast-sae-pilot-L12-v1
```

Derived CSV tables are in `tables/`, figures are in `figures/` as PNG and vector PDF, and `analysis_manifest.json` records SHA-256 provenance. Raw experiment files were not modified.
The analysis aborts if any of the 41 runner-recorded source checksums fails (36 race shard files plus five stage artifacts).
"""


def make_inventory(run_dir: Path, output_dir: Path) -> pd.DataFrame:
    files = [
        path
        for path in run_dir.rglob("*")
        if path.is_file() and output_dir not in path.parents
    ]
    return pd.DataFrame(
        [
            {
                "path": path.relative_to(run_dir).as_posix(),
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
            for path in sorted(files)
        ]
    )


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    run_dir = args.run_dir.resolve()
    output_dir = (args.output_dir or run_dir / "analysis").resolve()
    figures_dir = output_dir / "figures"
    tables_dir = output_dir / "tables"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    required = [
        "manifest.json",
        "feature_mining.json",
        "steering_rows.jsonl",
        "steering_summary.json",
        "steered_play_races.jsonl",
        "steered_play_summary.json",
    ]
    missing = [name for name in required if not (run_dir / name).is_file()]
    if missing:
        raise FileNotFoundError(f"missing required artifacts: {missing}")

    manifest = read_json(run_dir / "manifest.json")
    if manifest.get("status") != "complete":
        raise ValueError("refusing to analyze an incomplete run")
    source_hash_validation = validate_source_hashes(run_dir, manifest)
    mining = read_json(run_dir / "feature_mining.json")
    steering = pd.DataFrame(read_jsonl(run_dir / "steering_rows.jsonl"))
    live_records = read_jsonl(run_dir / "steered_play_races.jsonl")

    associations, _ = load_associations(run_dir, mining)
    fixed_summary, fixed_contrast, sign_summary = summarize_fixed(
        steering, args.bootstrap_repetitions
    )
    live = flatten_live(live_records)
    live_summary, live_contrast, live_equivalence = summarize_live(
        live, args.bootstrap_repetitions
    )

    save_csv(associations, tables_dir / "selected_feature_associations.csv")
    save_csv(fixed_summary, tables_dir / "fixed_state_condition_summary.csv")
    save_csv(fixed_contrast, tables_dir / "fixed_state_target_control_contrasts.csv")
    save_csv(sign_summary, tables_dir / "fixed_state_sign_dose_diagnostics.csv")
    save_csv(live, tables_dir / "live_race_level_metrics.csv")
    save_csv(live_summary, tables_dir / "live_condition_summary.csv")
    save_csv(live_contrast, tables_dir / "live_target_control_contrasts.csv")
    save_csv(live_equivalence, tables_dir / "live_target_control_equivalence.csv")

    configure_matplotlib()
    association_figure(associations, figures_dir)
    fixed_dose_figure(fixed_summary, figures_dir)
    fixed_control_figure(fixed_contrast, figures_dir)
    fixed_diagnostic_figure(fixed_summary, figures_dir)
    live_direct_figure(live_summary, figures_dir)
    live_payoff_figure(live_summary, figures_dir)

    report = build_report(
        manifest,
        associations,
        fixed_summary,
        fixed_contrast,
        sign_summary,
        live_summary,
        live_contrast,
    )
    (output_dir / "analysis_report.md").write_text(report, encoding="utf-8", newline="\n")

    inventory = make_inventory(run_dir, output_dir)
    save_csv(inventory, tables_dir / "source_artifact_inventory.csv")
    extreme = fixed_contrast.loc[fixed_contrast["alpha"].abs() == 2]
    summary_json = {
        "schema_version": "ai-race.causal-selfplay-analysis.v1",
        "source_run": run_dir.name,
        "source_config_fingerprint": manifest["config_fingerprint"],
        "evidence_class": manifest["evidence_class"],
        "counts": {
            "selfplay_races": manifest["stages"]["selfplay"]["n_completed_races"],
            "selfplay_decisions": mining["n_decisions"],
            "discovery_races": mining["n_discovery_races"],
            "eval_races": mining["n_eval_races"],
            "fixed_eval_decisions": manifest["stages"]["steer"]["n_eval_decisions"],
            "fixed_intervention_rows": len(steering),
            "live_races": len(live),
            "live_unique_seeds": manifest["stages"]["steered_play"]["n_unique_race_seeds"],
        },
        "validation": {
            "source_hashes": source_hash_validation,
            "max_baseline_replay_error": manifest["stages"]["steer"][
                "max_baseline_replay_error"
            ],
            "fixed_target_control_extreme_contrasts": len(extreme),
            "fixed_extreme_contrasts_ci_excludes_zero": int(
                ((extreme["delta_ci95_low"] > 0) | (extreme["delta_ci95_high"] < 0)).sum()
            ),
            "mean_live_exact_target_control_sequence_match_rate": live_contrast[
                "exact_action_sequence_match_rate"
            ].mean(),
            "target_specific_causal_claim_supported": False,
            "reason": "Target effects do not reliably exceed matched-random and unrelated-feature controls; live trajectories frequently coincide with controls.",
        },
        "selected_feature_associations": associations.loc[
            associations["feature_role"] == "selected"
        ].to_dict(orient="records"),
        "fixed_state_extreme_target_control_contrasts": extreme.to_dict(
            orient="records"
        ),
        "fixed_state_sign_dose_diagnostics": sign_summary.to_dict(orient="records"),
        "live_target_condition_summary": live_summary.loc[
            live_summary["condition"] == "target_feature"
        ].to_dict(orient="records"),
        "live_target_control_contrasts": live_contrast.to_dict(orient="records"),
        "chart_contracts": [
            {
                "figure": "association_selected_features",
                "question": "Do discovery associations persist on held-out races?",
                "family": "paired dot comparison",
                "claim_boundary": "association only",
            },
            {
                "figure": "fixed_state_dose_response",
                "question": "Are direct score changes monotone, signed, and target-specific?",
                "family": "faceted line with cluster-bootstrap intervals",
                "claim_boundary": "fixed-state direct effect",
            },
            {
                "figure": "fixed_state_target_minus_controls",
                "question": "Does target steering exceed controls at the strongest dose?",
                "family": "paired contrast interval plot",
                "claim_boundary": "feature specificity",
            },
            {
                "figure": "live_direct_comparable_flips",
                "question": "Does steering flip actions before state divergence?",
                "family": "faceted line",
                "claim_boundary": "direct comparable live decisions only",
            },
            {
                "figure": "live_endogenous_payoff_effects",
                "question": "How do complete steered trajectories differ in realized payoff?",
                "family": "interval plot",
                "claim_boundary": "endogenous trajectory effect",
            },
        ],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(to_jsonable(summary_json), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    derived = [path for path in output_dir.rglob("*") if path.is_file()]
    analysis_manifest = {
        "schema_version": "ai-race.causal-selfplay-analysis-manifest.v1",
        "source_config_fingerprint": manifest["config_fingerprint"],
        "analysis_script": "results/scripts/analyze_causal_selfplay_fast_sae.py",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "source_artifact_count": len(inventory),
        "derived_artifacts": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in sorted(derived)
            if path.name != "analysis_manifest.json"
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote causal FAST-SAE analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
