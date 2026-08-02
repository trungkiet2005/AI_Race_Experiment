"""Validate and visualize the GreenNode heterogeneous-dyad diagnostic."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


INK = "#172033"
MUTED = "#667085"
BLUE = "#2563EB"
AMBER = "#F59E0B"
TEAL = "#0F9D8A"
RED = "#DC4C64"
MODEL_SHORT = {
    "qwen25_7b": "Qwen 2.5 7B",
    "mistral7_01": "Mistral 7B",
}


def read_jsonl(path: Path) -> pd.DataFrame:
    return pd.DataFrame(
        [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    )


def load_block(path: Path, block: str) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    manifest = json.loads((path / "run_manifest.json").read_text(encoding="utf-8"))
    turns = read_jsonl(path / "turns.jsonl")
    races = read_jsonl(path / "races.jsonl")
    turns["lane_block"] = block
    races["lane_block"] = block
    return manifest, turns, races


def validate(
    manifests: dict[str, dict[str, Any]],
    turns: pd.DataFrame,
    races: pd.DataFrame,
) -> dict[str, Any]:
    failures: list[str] = []
    for block, manifest in manifests.items():
        if manifest.get("status") != "completed":
            failures.append(f"{block}: manifest is not completed")
        if manifest.get("evidence_class") != "diagnostic_unadmitted":
            failures.append(f"{block}: unexpected evidence class")
        if int(manifest.get("n_races", -1)) != 192:
            failures.append(f"{block}: expected 192 races")
        if int(manifest.get("n_turns", -1)) != 2496:
            failures.append(f"{block}: expected 2496 turns")
        if any(
            bool(receipt.get("passed"))
            for receipt in manifest.get("admission_receipts", {}).values()
        ):
            failures.append(f"{block}: admission unexpectedly passed")

    duplicated_games = int(
        races.groupby(["lane_block", "game_id"]).size().gt(1).sum()
    )
    parse_failures = int(turns["parse_failed"].astype(bool).sum())
    if duplicated_games:
        failures.append(f"{duplicated_games} duplicate block/game rows")
    if parse_failures:
        failures.append(f"{parse_failures} final parse failures")

    expected_identity = {
        ("not_disclosed", "not_disclosed"),
        ("not_disclosed", "accurate"),
        ("accurate", "not_disclosed"),
        ("accurate", "accurate"),
    }
    observed_identity = set(
        turns[["self_identity_condition", "opponent_identity_condition"]]
        .drop_duplicates()
        .itertuples(index=False, name=None)
    )
    if observed_identity != expected_identity:
        failures.append("2x2 self/opponent disclosure coverage mismatch")

    key = ["game_id", "round", "player_index"]
    wide = turns.pivot(index=key, columns="lane_block", values="action")
    lane_comparable = int(wide.dropna().shape[0])
    lane_action_mismatches = int((wide.dropna()["block1"] != wide.dropna()["block2"]).sum())
    if lane_comparable != 2496:
        failures.append(f"only {lane_comparable}/2496 decisions matched across lanes")

    source_commits = sorted(
        {str(manifest.get("source_commit")) for manifest in manifests.values()}
    )
    if len(source_commits) != 1:
        failures.append("lane blocks used different source commits")
    indexed = turns.set_index(key + ["lane_block"])
    model_wide = indexed["seat_model_key"].unstack("lane_block")
    mismatch_rows = pd.DataFrame(
        {
            "mismatch": wide.dropna()["block1"] != wide.dropna()["block2"],
            "model": model_wide.loc[wide.dropna().index, "block1"],
        }
    )
    mismatch_by_model = {
        str(model): {
            "n_mismatches": int(group["mismatch"].sum()),
            "n_compared": int(len(group)),
            "agreement": float(1.0 - group["mismatch"].mean()),
        }
        for model, group in mismatch_rows.groupby("model")
    }

    return {
        "passed": not failures,
        "failures": failures,
        "n_blocks": len(manifests),
        "n_races_total": int(len(races)),
        "n_turns_total": int(len(turns)),
        "parse_failures": parse_failures,
        "lane_comparable_decisions": lane_comparable,
        "lane_action_mismatches": lane_action_mismatches,
        "lane_action_agreement": 1.0 - lane_action_mismatches / max(lane_comparable, 1),
        "lane_action_by_model": mismatch_by_model,
        "source_commits": source_commits,
    }


def add_position(turns: pd.DataFrame) -> pd.DataFrame:
    turns = turns.copy()
    turns["persona_condition"] = turns["persona_condition"].replace(
        {"none": "neutral"}
    )
    turns["position"] = np.select(
        [turns["progress_gap_before"] > 1e-9, turns["progress_gap_before"] < -1e-9],
        ["ahead", "behind"],
        default="tied",
    )
    turns["round_phase"] = np.where(turns["round"] == 1, "round 1", "round 2+")
    turns["model_label"] = turns["seat_model_key"].map(MODEL_SHORT)
    return turns


def summaries(primary: pd.DataFrame) -> dict[str, pd.DataFrame]:
    core = (
        primary.groupby(
            [
                "seat_model_key",
                "dyad_type",
                "persona_condition",
                "self_identity_condition",
                "opponent_identity_condition",
                "max_private_risk",
            ],
            dropna=False,
        )["unsafe"]
        .agg([("unsafe_rate", "mean"), ("n_decisions", "size")])
        .reset_index()
    )
    position = (
        primary.groupby(
            ["seat_model_key", "persona_condition", "position"], dropna=False
        )["unsafe"]
        .agg([("unsafe_rate", "mean"), ("n_decisions", "size")])
        .reset_index()
    )
    round_one = (
        primary[primary["round"] == 1]
        .groupby(
            [
                "seat_model_key",
                "dyad_type",
                "self_identity_condition",
                "opponent_identity_condition",
            ]
        )["unsafe"]
        .agg([("unsafe_rate", "mean"), ("n_decisions", "size")])
        .reset_index()
    )
    later = (
        primary[primary["round"] >= 2]
        .groupby(
            [
                "seat_model_key",
                "dyad_type",
                "self_identity_condition",
                "opponent_identity_condition",
            ]
        )["unsafe"]
        .agg([("unsafe_rate", "mean"), ("n_decisions", "size")])
        .reset_index()
    )
    return {"cell": core, "position": position, "round_one": round_one, "later": later}


def identity_effects(primary: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "seat_model_key",
        "dyad_type",
        "persona_condition",
        "self_identity_condition",
        "max_private_risk",
        "round_phase",
    ]
    rates = (
        primary.groupby(grouping + ["opponent_identity_condition"])["unsafe"]
        .mean()
        .unstack("opponent_identity_condition")
        .reset_index()
    )
    rates["opponent_label_effect_pp"] = 100 * (
        rates["accurate"] - rates["not_disclosed"]
    )
    return rates


def aggregate_identity_effects(primary: pd.DataFrame) -> pd.DataFrame:
    grouping = [
        "seat_model_key",
        "dyad_type",
        "persona_condition",
        "round_phase",
    ]
    summary = (
        primary.groupby(grouping + ["opponent_identity_condition"])["unsafe"]
        .agg([("unsafe_rate", "mean"), ("n_decisions", "size")])
        .reset_index()
    )
    rates = summary.pivot(
        index=grouping,
        columns="opponent_identity_condition",
        values="unsafe_rate",
    ).reset_index()
    counts = summary.pivot(
        index=grouping,
        columns="opponent_identity_condition",
        values="n_decisions",
    ).reset_index()
    result = rates.copy()
    result["n_accurate"] = counts["accurate"]
    result["n_not_disclosed"] = counts["not_disclosed"]
    result["opponent_label_effect_pp"] = 100 * (
        result["accurate"] - result["not_disclosed"]
    )
    return result


def style_axis(axis: plt.Axes) -> None:
    axis.spines[["top", "right"]].set_visible(False)
    axis.tick_params(colors=MUTED)
    axis.title.set_color(INK)
    axis.xaxis.label.set_color(INK)
    axis.yaxis.label.set_color(INK)
    axis.grid(axis="y", color="#E7EAF0", linewidth=0.8, zorder=0)


def save_figure(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def plot_identity_matrix(primary: pd.DataFrame, output: Path) -> None:
    grouped = (
        primary.groupby(
            [
                "seat_model_key",
                "persona_condition",
                "self_identity_condition",
                "opponent_identity_condition",
            ]
        )["unsafe"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(2, 2, figsize=(9.2, 7.2), constrained_layout=True)
    for axis, (model, persona) in zip(
        axes.flat,
        [
            (model, persona)
            for model in ("qwen25_7b", "mistral7_01")
            for persona in ("neutral", "competitive")
        ],
    ):
        subset = grouped[
            (grouped["seat_model_key"] == model)
            & (grouped["persona_condition"] == persona)
        ]
        matrix = subset.pivot(
            index="self_identity_condition",
            columns="opponent_identity_condition",
            values="unsafe",
        ).reindex(index=["not_disclosed", "accurate"], columns=["not_disclosed", "accurate"])
        image = axis.imshow(matrix.values, vmin=0, vmax=1, cmap="Blues")
        for row in range(2):
            for col in range(2):
                value = float(matrix.iloc[row, col])
                axis.text(
                    col,
                    row,
                    f"{100 * value:.1f}%",
                    ha="center",
                    va="center",
                    color="white" if value > 0.55 else INK,
                    fontsize=12,
                    fontweight="bold",
                )
        axis.set_xticks([0, 1], ["not disclosed", "accurate"])
        axis.set_yticks([0, 1], ["not disclosed", "accurate"])
        axis.set_xlabel("Opponent identity label")
        axis.set_ylabel("Self identity label")
        axis.set_title(f"{MODEL_SHORT[model]} · {persona}", loc="left", fontweight="bold")
    fig.colorbar(image, ax=axes, label="Unsafe rate", shrink=0.76)
    fig.suptitle("Identity labels alter diagnostic play", fontsize=17, color=INK, fontweight="bold")
    save_figure(fig, output, "identity_disclosure_matrix")


def plot_risk_response(primary: pd.DataFrame, output: Path) -> None:
    subset = primary[
        (primary["persona_condition"] == "neutral")
        & (primary["self_identity_condition"] == "not_disclosed")
        & (primary["opponent_identity_condition"] == "not_disclosed")
    ]
    grouped = (
        subset.groupby(["seat_model_key", "dyad_type", "max_private_risk"])["unsafe"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, constrained_layout=True)
    for axis, model in zip(axes, ("qwen25_7b", "mistral7_01")):
        cell = grouped[grouped["seat_model_key"] == model]
        for dyad, color, marker in (
            ("same_family", TEAL, "o"),
            ("cross_family", AMBER, "s"),
        ):
            line = cell[cell["dyad_type"] == dyad].sort_values("max_private_risk")
            axis.plot(
                line["max_private_risk"],
                100 * line["unsafe"],
                marker=marker,
                linewidth=2.3,
                color=color,
                label=dyad.replace("_", " "),
            )
        style_axis(axis)
        axis.set_title(MODEL_SHORT[model], loc="left", fontweight="bold")
        axis.set_xlabel("Maximum private risk")
        axis.set_xticks([0.1, 0.6, 0.9])
    axes[0].set_ylabel("Unsafe decisions (%)")
    axes[1].legend(frameon=False, loc="best")
    fig.suptitle(
        "Actual opponent policy under identity-not-disclosed prompts",
        fontsize=16,
        color=INK,
        fontweight="bold",
    )
    save_figure(fig, output, "risk_response_same_vs_cross")


def plot_position(primary: pd.DataFrame, output: Path) -> None:
    order = ["behind", "tied", "ahead"]
    grouped = (
        primary.groupby(["seat_model_key", "persona_condition", "position"])["unsafe"]
        .mean()
        .reset_index()
    )
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), sharey=True, constrained_layout=True)
    for axis, model in zip(axes, ("qwen25_7b", "mistral7_01")):
        for persona, color, marker in (
            ("neutral", BLUE, "o"),
            ("competitive", RED, "s"),
        ):
            cell = grouped[
                (grouped["seat_model_key"] == model)
                & (grouped["persona_condition"] == persona)
            ].set_index("position").reindex(order)
            axis.plot(
                order,
                100 * cell["unsafe"],
                marker=marker,
                linewidth=2.3,
                color=color,
                label=persona,
            )
        style_axis(axis)
        axis.set_title(MODEL_SHORT[model], loc="left", fontweight="bold")
        axis.set_xlabel("Endogenous live-race position")
    axes[0].set_ylabel("Unsafe decisions (%)")
    axes[1].legend(frameon=False)
    fig.suptitle(
        "Position response is descriptive, not causal",
        fontsize=16,
        color=INK,
        fontweight="bold",
    )
    save_figure(fig, output, "endogenous_position_response")


def pct(value: float) -> str:
    return f"{100 * value:.1f}%"


def write_report(
    output: Path,
    validation: dict[str, Any],
    primary: pd.DataFrame,
    tables: dict[str, pd.DataFrame],
    effects: pd.DataFrame,
    aggregate_effects: pd.DataFrame,
) -> None:
    overall = primary.groupby("seat_model_key")["unsafe"].mean().to_dict()
    largest = aggregate_effects.iloc[
        aggregate_effects["opponent_label_effect_pp"].abs().argmax()
    ]
    position = tables["position"].pivot(
        index=["seat_model_key", "persona_condition"],
        columns="position",
        values="unsafe_rate",
    )
    report = f"""# Heterogeneous Qwen–Mistral dyad diagnostic

**Evidence class: diagnostic, unadmitted.** Both checkpoints failed the frozen
state-update and terminal-scoring comprehension gates. These results measure
prompt-conditioned enacted actions; they do not establish strategic
understanding, expected-payoff optimization, or model-family universals.

## What was run

- 2 lane-counterbalanced blocks; **{validation['n_races_total']} races** and
  **{validation['n_turns_total']} decisions** total.
- Exact BF16 checkpoints: Qwen2.5-7B-Instruct and Mistral-7B-Instruct-v0.1.
- Same-checkpoint controls plus Qwen→Mistral and Mistral→Qwen seat reversal.
- 2×2 self-identity × opponent-identity disclosure, neutral/competitive role,
  risks 0.1/0.6/0.9, temperature 0.
- Block 2 swapped the models across the 20GB and 40GB H100 MIG lanes.

## Validation

- Validation passed: **{validation['passed']}**.
- Final parse failures: **{validation['parse_failures']}**.
- Lane-matched actions: **{validation['lane_comparable_decisions']}**;
  agreement **{pct(validation['lane_action_agreement'])}**
  ({validation['lane_action_mismatches']} mismatches).
- By checkpoint: Qwen agreement
  **{pct(validation['lane_action_by_model']['qwen25_7b']['agreement'])}**;
  Mistral agreement
  **{pct(validation['lane_action_by_model']['mistral7_01']['agreement'])}**.
- Source commit: `{validation['source_commits'][0]}`.

Block 2 is a technical lane replication and is not pooled into behavioral
rates. All rates below use block 1 only.

## Main diagnostic observations

1. Overall Unsafe rate was **{pct(overall['qwen25_7b'])}** for Qwen and
   **{pct(overall['mistral7_01'])}** for Mistral under this factorial.
2. The largest raw opponent-label contrast was
   **{largest['opponent_label_effect_pp']:+.1f} percentage points** for
   `{largest['seat_model_key']}` / `{largest['dyad_type']}` /
   `{largest['persona_condition']}` / `{largest['round_phase']}`
   (n={int(largest['n_accurate'])} accurate-label and
   n={int(largest['n_not_disclosed'])} not-disclosed decisions).
   This is a surface-label effect in a smoke diagnostic, not a confidence-bounded
   population estimate.
3. Live-race position is endogenous. For two players,
   `progress_gap_before = 0.5 × (own prior Unsafe count − opponent prior Unsafe count)`;
   therefore the position figure is association only. A randomized progress
   endowment / matched fork is required for a causal first/middle/last claim.

## Position rates used in the descriptive figure

| Model | Persona | Behind | Tied | Ahead |
|---|---|---:|---:|---:|
"""
    for (model, persona), row in position.iterrows():
        report += (
            f"| {MODEL_SHORT[model]} | {persona} | {pct(row.get('behind', np.nan))} | "
            f"{pct(row.get('tied', np.nan))} | {pct(row.get('ahead', np.nan))} |\n"
        )
    report += """

## Figures

![](figures/identity_disclosure_matrix.png)

![](figures/risk_response_same_vs_cross.png)

![](figures/endogenous_position_response.png)

## Robustness boundary and next experiment

- Temperature zero gives deterministic checkpoint behavior for a fixed prompt;
  repeated horizons are not independent model samples. The 1.4% cross-lane
  mismatch shows greedy GPU inference was not bitwise invariant to lane/runtime;
  this is reported as a robustness result, not averaged away.
- Accurate and not-disclosed arms differ in tokens, so the estimand is label
  disclosure, not hidden recognition of the opponent's family.
- Persona is a prompt-conditioned role, not a stable personality.
- The next causal position experiment should apply an engine-scored randomized
  progress endowment after a common prehistory, query the immediate action, and
  roll matched branches forward with identical random streams.
- N=3 must record `n_ahead` and ties explicitly: strict leader/middle/last are
  `n_ahead = 0/1/2` only when no other player is tied.
"""
    (output / "README.md").write_text(report, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--block1", type=Path, required=True)
    parser.add_argument("--block2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    figures = args.output / "figures"
    data = args.output / "data"
    figures.mkdir(exist_ok=True)
    data.mkdir(exist_ok=True)

    loaded = {
        "block1": load_block(args.block1, "block1"),
        "block2": load_block(args.block2, "block2"),
    }
    manifests = {key: value[0] for key, value in loaded.items()}
    turns = pd.concat([value[1] for value in loaded.values()], ignore_index=True)
    races = pd.concat([value[2] for value in loaded.values()], ignore_index=True)
    validation = validate(manifests, turns, races)
    if not validation["passed"]:
        raise RuntimeError("; ".join(validation["failures"]))

    primary = add_position(turns[turns["lane_block"] == "block1"])
    tables = summaries(primary)
    effects = identity_effects(primary)
    aggregate_effects = aggregate_identity_effects(primary)
    for name, table in tables.items():
        table.to_csv(data / f"{name}_summary.csv", index=False)
    effects.to_csv(data / "opponent_identity_effects.csv", index=False)
    aggregate_effects.to_csv(
        data / "opponent_identity_effects_aggregated.csv", index=False
    )
    (data / "validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    plot_identity_matrix(primary, figures)
    plot_risk_response(primary, figures)
    plot_position(primary, figures)
    write_report(
        args.output,
        validation,
        primary,
        tables,
        effects,
        aggregate_effects,
    )


if __name__ == "__main__":
    main()
