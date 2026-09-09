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
  * q=1: exp(Shannon entropy), the effective number of paired trajectories;
  * mean pairwise Hamming distance between the paired 10-bit action strings,
    divided by 10 so it lands in [0, 1].

Why a third statistic.  Hill numbers count distinct strings and are blind to
how far apart those strings are: a cell holding twenty trajectories that differ
in a single round scores the same q=0 as a cell holding twenty trajectories
that differ in all ten bits.  Mean pairwise Hamming distance does not cluster
or bin anything, so it separates "many nearly identical policies" from "many
genuinely different policies" without inheriting any tuning parameter.

Why the point estimates are computed the way they are.  The published point
values are the mean over the rarefaction draws (for a cell larger than the
comparison size) or the exact cell value (for a cell already at the comparison
size).  That definition is kept unchanged, and the draw order and RNG stream
that produce it are kept unchanged, so the point estimates in this table are
bit-identical to the ones the earlier version published.

Why the interval is a two-stage cluster bootstrap.  Rarefaction without
replacement quantifies only subsampling noise, so a cell whose source size
already equals the comparison size of 20 gets an interval of exactly zero
width.  Every LLM cell is in that position, which made the earlier intervals
uninformative rather than tight.  The honest question is sampling
uncertainty over the units that were actually drawn independently, so the
interval comes from resampling those units with replacement.

Why the cluster is the race, not the seat.  ``load_model`` emits two rows per
race, one per seat, because both seats of a race are trajectories.  They are
not independent draws: under the common-random-number design both seats of one
race share the horizon draw and the setback draw, so a race is the smallest
exchangeable unit.  Resampling seats independently would treat one experimental
unit as two and shrink the interval by roughly a factor of sqrt(2).  For the
human population the exchangeable unit is the participant, who contributes one
trajectory.  When a race is drawn, both of its seats come with it.

Why rarefaction happens inside each bootstrap draw and not around it.  The
comparison is defined at a fixed sample size of 20; if we rarefied once and
then bootstrapped the rarefied sample, the interval would describe uncertainty
in a single 20-trajectory subsample rather than in the population, and the
human cell would lose the very rarefaction step that makes it comparable.
Drawing clusters first and then rarefying to 20 inside the draw keeps every
bootstrap replicate on the same footing as the point estimate: each replicate
is itself a size-20 comparison.  A cell whose bootstrap replicate is already
of size 20 (every LLM cell: ten races, two seats each) skips the inner step,
and all of its interval width comes from the cluster stage.

A cell in which all twenty trajectories are identical has q=0 = q=1 = 1 and
mean pairwise Hamming 0 in every replicate, so its interval is a point by
construction, not by a defect in the interval.

This analysis is descriptive: it measures observed trajectory diversity in the
tested samples, not latent policy entropy.
"""
from __future__ import annotations

import ast
from collections import Counter
import hashlib
import json
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
TRAJECTORY_BITS = 10
CI_METHOD = (
    "two-stage cluster bootstrap over races/participants, then rarefaction to n=20"
)
CLUSTER_UNIT = {
    "Human": "participant_id (one trajectory per participant)",
    "LLM": "game_id (one race; both seats travel together because they share the horizon and setback draws)",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
                "cluster": str(participant_id),
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
                "cluster": str(record.game_id),
                "risk_cap": risk,
                "trajectory": paired_key(a1, a2),
            }
        )
        rows.append(
            {
                "population": label,
                "unit": f"{record.game_id}::p2",
                "cluster": str(record.game_id),
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


def to_bits(keys: list[str]) -> np.ndarray:
    """Return an (n, 10) 0/1 matrix for the paired action strings."""
    stripped = [key.replace("|", "") for key in keys]
    matrix = np.frombuffer("".join(stripped).encode("ascii"), dtype=np.uint8)
    matrix = matrix.reshape(len(stripped), TRAJECTORY_BITS) - ord("0")
    return matrix.astype(np.int64)


def mean_pairwise_hamming(keys: list[str]) -> float:
    """Mean pairwise Hamming distance over the 10 bits, normalised to [0, 1].

    For each bit position with ``k`` ones among ``n`` strings there are exactly
    ``k * (n - k)`` discordant pairs, so the mean over all ``C(n, 2)`` pairs is
    obtained in O(10) rather than O(n^2).
    """
    n = len(keys)
    if n < 2:
        return 0.0
    ones = to_bits(keys).sum(axis=0)
    discordant = float((ones * (n - ones)).sum())
    n_pairs = n * (n - 1) / 2.0
    return discordant / (n_pairs * TRAJECTORY_BITS)


def cell_statistics(keys: list[str]) -> tuple[float, float, float]:
    q0, q1 = hill_numbers(keys)
    return q0, q1, mean_pairwise_hamming(keys)


def rarefy(keys: list[str], target_n: int, rng: np.random.Generator) -> dict[str, float]:
    """Point estimates at the comparison size.

    The RNG draw sequence is exactly the one the earlier version used, so the
    published q=0 and q=1 point values do not move; the Hamming mean is
    computed on the same drawn subsamples rather than on new draws.
    """
    if len(keys) < target_n:
        raise ValueError(f"Cell has {len(keys)} rows but target_n={target_n}")
    if len(keys) == target_n:
        q0, q1, hamming = cell_statistics(keys)
        return {
            "q0_mean": q0,
            "q0_low": q0,
            "q0_high": q0,
            "q1_mean": q1,
            "q1_low": q1,
            "q1_high": q1,
            "mean_pairwise_hamming": hamming,
        }

    q0_values = np.empty(N_RESAMPLES, dtype=float)
    q1_values = np.empty(N_RESAMPLES, dtype=float)
    hamming_values = np.empty(N_RESAMPLES, dtype=float)
    values = np.asarray(keys, dtype=object)
    for i in range(N_RESAMPLES):
        sample = rng.choice(values, size=target_n, replace=False).tolist()
        q0_values[i], q1_values[i], hamming_values[i] = cell_statistics(sample)
    return {
        "q0_mean": float(q0_values.mean()),
        "q0_low": float(np.quantile(q0_values, 0.025)),
        "q0_high": float(np.quantile(q0_values, 0.975)),
        "q1_mean": float(q1_values.mean()),
        "q1_low": float(np.quantile(q1_values, 0.025)),
        "q1_high": float(np.quantile(q1_values, 0.975)),
        "mean_pairwise_hamming": float(hamming_values.mean()),
    }


def cluster_bootstrap(
    cell: pd.DataFrame, target_n: int, rng: np.random.Generator
) -> dict[str, float]:
    """Two-stage cluster bootstrap: resample clusters, then rarefy to target_n.

    Stage one draws the same number of clusters as the cell contains, with
    replacement, so a race enters as a whole (both seats) or not at all.  Stage
    two rarefies the pooled replicate down to the comparison size when it is
    larger, which is what keeps every replicate a size-``target_n`` comparison.
    """
    groups = [
        np.asarray(group["trajectory"].tolist(), dtype=object)
        for _, group in cell.groupby("cluster", sort=True)
    ]
    n_clusters = len(groups)
    if n_clusters == 0:
        raise ValueError("Empty cell")

    q0_values = np.empty(N_RESAMPLES, dtype=float)
    q1_values = np.empty(N_RESAMPLES, dtype=float)
    hamming_values = np.empty(N_RESAMPLES, dtype=float)
    for i in range(N_RESAMPLES):
        picks = rng.integers(0, n_clusters, size=n_clusters)
        pooled = np.concatenate([groups[j] for j in picks])
        if len(pooled) > target_n:
            pooled = rng.choice(pooled, size=target_n, replace=False)
        keys = pooled.tolist()
        q0_values[i], q1_values[i], hamming_values[i] = cell_statistics(keys)
    return {
        "n_clusters": n_clusters,
        "q0_ci_low": float(np.quantile(q0_values, 0.025)),
        "q0_ci_high": float(np.quantile(q0_values, 0.975)),
        "q1_ci_low": float(np.quantile(q1_values, 0.025)),
        "q1_ci_high": float(np.quantile(q1_values, 0.975)),
        "hamming_ci_low": float(np.quantile(hamming_values, 0.025)),
        "hamming_ci_high": float(np.quantile(hamming_values, 0.975)),
    }


def build_table(frame: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    boot_rng = np.random.default_rng([SEED, 1])
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
            interval = cluster_bootstrap(cell, target_n, boot_rng)
            rows.append(
                {
                    "risk_cap": risk,
                    "population": population,
                    "source_n": len(cell),
                    "comparison_n": target_n,
                    **summary,
                    **interval,
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
        low = sub["q1_ci_low"].to_numpy(float)
        high = sub["q1_ci_high"].to_numpy(float)
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
                yerr=np.vstack(
                    [
                        np.clip(y[varying] - low[varying], 0.0, None),
                        np.clip(high[varying] - y[varying], 0.0, None),
                    ]
                ),
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
    ax.set_ylim(0, max(21, float(table["q1_ci_high"].max()) * 1.04))
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
    source_files = [HUMAN_CSV]
    for _, raw_path in MODEL_INPUTS:
        source_path = Path(raw_path)
        source_files.append(
            source_path / "all_results.csv" if source_path.is_dir() else source_path
        )
    provenance = {
        "schema_version": "ai-race-trajectory-diversity-rarefaction-v3",
        "evidence_class": "diagnostic",
        "estimand": "Sample-size-matched observed diversity of paired first-five-round trajectories",
        "diversity_statistics": [
            "hill_q0_distinct_paired_trajectories",
            "hill_q1_effective_paired_trajectories",
            "mean_pairwise_hamming_over_10_bits",
        ],
        "point_estimate_method": (
            "mean over rarefaction draws without replacement for cells larger than "
            "the comparison size; exact cell value for cells already at it "
            "(unchanged from schema v2)"
        ),
        "ci_method": CI_METHOD,
        "ci_level": 0.95,
        "n_resamples": N_RESAMPLES,
        "seed": SEED,
        "bootstrap_seed": [SEED, 1],
        "cluster_unit": CLUSTER_UNIT,
        "comparison_n": int(table["comparison_n"].min()),
        "risk_caps": list(RISKS),
        "source_hashes": {path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files},
        "output": "trajectory_diversity_rarefaction.csv",
    }
    (DATA_OUT / "trajectory_diversity_rarefaction.json").write_text(
        json.dumps(provenance, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(table.to_string(index=False))
    print("\nKey boundary: GPT-5.4 nano is close to the rarefied human reference in every risk cell;")
    print("do not use this analysis to claim that every tested LLM is less diverse than humans.")


if __name__ == "__main__":
    main()
