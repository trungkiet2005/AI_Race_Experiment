#!/usr/bin/env python3
"""Sample-size-matched first-five-round trajectory diversity.

This analysis is deliberately independent of HDBSCAN and t-SNE.  A trajectory
is represented by the focal player's first five actions together with the
opponent's first five actions.  Under the fixed progress increments in the
two-player game, the pre-round progress-gap sequence is a deterministic
function of those paired actions, so adding it would not add new information.

For each private-risk condition we compare populations at the smallest cell
size (20 in the current seven-checkpoint baseline).  Populations with more
observations are rarefied by sampling without replacement.  Diversity is
reported as:
  * q=0: number of distinct paired trajectories;
  * q=1: exp(Shannon entropy), the effective number of paired trajectories.

The human reference is resampled 2,000 times with a fixed RNG seed.  Current
LLM cells have exactly 20 trajectories per risk condition and therefore enter
without resampling.  This analysis is descriptive: it measures observed
trajectory diversity in the tested samples, not latent policy entropy.
"""
from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

ROOT = Path(__file__).resolve().parents[1]
HUMAN_CSV = ROOT / "references" / "source_study_dataset" / "airace_deidentified_long.csv"
DATA_OUT = ROOT / "results" / "cross_model_pilot_synthesis" / "data"
FIG_OUT = ROOT / "figures" / "paper"

MODEL_INPUTS = [
    ("GPT-5 nano", ROOT / "results/frontier/openai/baseline/gpt-5-nano/all_results.csv"),
    ("GPT-5.4 nano", ROOT / "results/frontier/openai/baseline/gpt-5.4-nano/all_results.csv"),
    ("Gemini 3 Flash", ROOT / "results/frontier/baseline/google-gemini-3-flash-preview/all_results.csv"),
    ("Gemini 3.1 Flash Lite", ROOT / "results/frontier/baseline/google-gemini-3.1-flash-lite-preview/all_results.csv"),
    ("Gemini 3.5 Flash Lite", ROOT / "results/frontier/baseline/google-gemini-3.5-flash-lite/all_results.csv"),
    ("Claude Opus 5", ROOT / "results/frontier/bedrock/baseline/us.anthropic.claude-opus-5/all_results.csv"),
    ("Claude Sonnet 5", ROOT / "results/frontier/bedrock/baseline/us.anthropic.claude-sonnet-5/all_results.csv"),
]
RISKS = (0.1, 0.6, 0.9)
N_RESAMPLES = 2000
SEED = 20260908


def action_bits(value: str) -> tuple[int, ...]:
    actions = ast.literal_eval(value)
    return tuple(1 if str(action).upper() == "UNSAFE" else 0 for action in actions[:5])


def paired_key(own: tuple[int, ...], opponent: tuple[int, ...]) -> str:
    if len(own) != 5 or len(opponent) != 5:
        raise ValueError("Expected five actions for both players")
    return "".join(map(str, own)) + "|" + "".join(map(str, opponent))


def load_human() -> pd.DataFrame:
    raw = pd.read_csv(HUMAN_CSV)
    rows: list[dict[str, object]] = []
    for participant_id, group in raw.groupby("participant_id", sort=True):
        first = group[group["round_number"].isin(range(1, 6))].sort_values("round_number")
        if first["round_number"].astype(int).tolist() != [1, 2, 3, 4, 5]:
            continue
        own = tuple(first["decision"].astype(int).tolist())
        opponent = tuple(first["decision_opponent"].astype(int).tolist())
        rows.append(
            {
                "population": "Human",
                "unit": str(participant_id),
                "risk_cap": float(first["max_private_risk"].iloc[0]),
                "trajectory": paired_key(own, opponent),
            }
        )
    return pd.DataFrame(rows)


def load_model(label: str, path: Path) -> pd.DataFrame:
    raw = pd.read_csv(path)
    rows: list[dict[str, object]] = []
    for record in raw.itertuples(index=False):
        a1 = action_bits(record.player1_actions)
        a2 = action_bits(record.player2_actions)
        if len(a1) != 5 or len(a2) != 5:
            continue
        risk = float(record.max_private_risk)
        rows.append(
            {
                "population": label,
                "unit": f"{record.game_id}::p1",
                "risk_cap": risk,
                "trajectory": paired_key(a1, a2),
            }
        )
        rows.append(
            {
                "population": label,
                "unit": f"{record.game_id}::p2",
                "risk_cap": risk,
                "trajectory": paired_key(a2, a1),
            }
        )
    return pd.DataFrame(rows)


def hill_numbers(keys: list[str]) -> tuple[float, float]:
    counts = Counter(keys)
    q0 = float(len(counts))
    n = float(len(keys))
    probs = np.asarray([count / n for count in counts.values()], dtype=float)
    entropy = float(-(probs * np.log(probs)).sum())
    q1 = float(np.exp(entropy))
    return q0, q1


def rarefy(keys: list[str], target_n: int, rng: np.random.Generator) -> dict[str, float]:
    if len(keys) < target_n:
        raise ValueError(f"Cell has {len(keys)} rows but target_n={target_n}")
    if len(keys) == target_n:
        q0, q1 = hill_numbers(keys)
        return {
            "q0_mean": q0,
            "q0_low": q0,
            "q0_high": q0,
            "q1_mean": q1,
            "q1_low": q1,
            "q1_high": q1,
        }

    q0_values = np.empty(N_RESAMPLES, dtype=float)
    q1_values = np.empty(N_RESAMPLES, dtype=float)
    values = np.asarray(keys, dtype=object)
    for i in range(N_RESAMPLES):
        sample = rng.choice(values, size=target_n, replace=False).tolist()
        q0_values[i], q1_values[i] = hill_numbers(sample)
    return {
        "q0_mean": float(q0_values.mean()),
        "q0_low": float(np.quantile(q0_values, 0.025)),
        "q0_high": float(np.quantile(q0_values, 0.975)),
        "q1_mean": float(q1_values.mean()),
        "q1_low": float(np.quantile(q1_values, 0.025)),
        "q1_high": float(np.quantile(q1_values, 0.975)),
    }


def build_table(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    min_cell = int(
        frame.groupby(["population", "risk_cap"], observed=True).size().min()
    )
    target_n = min(20, min_cell)
    rows: list[dict[str, object]] = []
    order = ["Human"] + [label for label, _ in MODEL_INPUTS]
    for risk in RISKS:
        for population in order:
            cell = frame[
                frame["population"].eq(population)
                & np.isclose(frame["risk_cap"].astype(float), risk)
            ]
            summary = rarefy(cell["trajectory"].tolist(), target_n, rng)
            rows.append(
                {
                    "risk_cap": risk,
                    "population": population,
                    "source_n": len(cell),
                    "comparison_n": target_n,
                    **summary,
                }
            )
    return pd.DataFrame(rows)


def draw(table: pd.DataFrame) -> None:
    order = ["Human"] + [label for label, _ in MODEL_INPUTS]
    x = np.arange(len(order), dtype=float)
    offsets = {0.1: -0.18, 0.6: 0.0, 0.9: 0.18}
    markers = {0.1: "o", 0.6: "s", 0.9: "^"}
    colors = {0.1: "#0072B2", 0.6: "#E69F00", 0.9: "#CC79A7"}

    fig, ax = plt.subplots(figsize=(11.2, 5.2))
    for risk in RISKS:
        sub = (
            table[table["risk_cap"].eq(risk)]
            .set_index("population")
            .loc[order]
        )
        xx = x + offsets[risk]
        y = sub["q1_mean"].to_numpy(float)
        low = sub["q1_low"].to_numpy(float)
        high = sub["q1_high"].to_numpy(float)
        ax.scatter(
            xx,
            y,
            s=56,
            marker=markers[risk],
            color=colors[risk],
            edgecolor="black",
            linewidth=0.45,
            label=f"Risk cap {int(risk * 100)}%",
            zorder=3,
        )
        varying = np.where((high - low) > 1e-12)[0]
        if len(varying):
            ax.errorbar(
                xx[varying],
                y[varying],
                yerr=np.vstack([y[varying] - low[varying], high[varying] - y[varying]]),
                fmt="none",
                ecolor=colors[risk],
                elinewidth=1.5,
                capsize=3,
                zorder=2,
            )

    ax.axvline(0.5, color="#B8B8B8", linewidth=0.8)
    ax.set_xticks(x, order, rotation=28, ha="right")
    ax.set_ylabel("Effective number of paired 5-round trajectories (Hill q=1)")
    ax.set_xlabel("Population")
    ax.set_ylim(0, max(21, float(table["q1_high"].max()) * 1.04))
    ax.grid(axis="y", linewidth=0.55, alpha=0.35)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center")
    ax.set_title("Sample-size-matched paired trajectory diversity")
    fig.tight_layout()

    FIG_OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_OUT / "trajectory_diversity_rarefaction.pdf", bbox_inches="tight", facecolor="white")
    fig.savefig(FIG_OUT / "trajectory_diversity_rarefaction.png", dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    frames = [load_human()]
    for label, path in MODEL_INPUTS:
        frames.append(load_model(label, path))
    data = pd.concat(frames, ignore_index=True)

    table = build_table(data)
    DATA_OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(DATA_OUT / "trajectory_diversity_rarefaction.csv", index=False)
    draw(table)

    print(table.to_string(index=False))
    print("\nKey boundary: GPT-5.4 nano is close to the rarefied human reference in every risk cell;")
    print("do not use this analysis to claim that every tested LLM is less diverse than humans.")


if __name__ == "__main__":
    main()
