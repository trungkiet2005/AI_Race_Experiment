#!/usr/bin/env python3
"""Create paper-inspired descriptive reports and figures for two-agent AI races.

This script is deliberately separate from ``analyze_ai_race.py``.  It is a compact
post-run report for one completed, homogeneous 150-race batch (50 repetitions x
three private-risk treatments), rather than a tool for pooling heterogeneous
studies or claiming a human replication.  The unit resampled for uncertainty is a
race, never an individual decision.

Example
-------
python results/scripts/report_two_agent_race.py \
  --input /kaggle/input/my-ai-race-run \
  --output /kaggle/working/two-agent-report
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable

import matplotlib

# Kaggle and CI do not provide a desktop/Tk display.  Select a file-rendering
# backend before importing pyplot so PNG generation remains portable.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from mpl_toolkits.axes_grid1.inset_locator import inset_axes


RISK_ORDER = (0.1, 0.6, 0.9)
RISK_LABELS = {0.1: "10%", 0.6: "60%", 0.9: "90%"}
COLORS = {0.1: "#4c78a8", 0.6: "#f58518", 0.9: "#54a24b"}


def _discover_runs(inputs: Iterable[Path]) -> list[Path]:
    runs: set[Path] = set()
    for source in inputs:
        source = source.resolve()
        if source.is_file():
            source = source.parent
        if (source / "turns.jsonl").exists():
            runs.add(source)
        if source.exists():
            runs.update(path.parent for path in source.rglob("turns.jsonl"))
    if not runs:
        raise ValueError("No turns.jsonl found under --input")
    return sorted(runs)


def _read_runs(run_dirs: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    turns, races, players = [], [], []
    for run_dir in run_dirs:
        required = [run_dir / name for name in ("turns.jsonl", "races.csv", "players.csv")]
        missing = [path.name for path in required if not path.exists()]
        if missing:
            raise ValueError(f"{run_dir} is missing required files: {', '.join(missing)}")
        turns.append(pd.read_json(run_dir / "turns.jsonl", lines=True).assign(source_run=str(run_dir)))
        races.append(pd.read_csv(run_dir / "races.csv").assign(source_run=str(run_dir)))
        players.append(pd.read_csv(run_dir / "players.csv").assign(source_run=str(run_dir)))
    return pd.concat(turns, ignore_index=True), pd.concat(races, ignore_index=True), pd.concat(players, ignore_index=True)


def _require(frame: pd.DataFrame, columns: set[str], table: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{table} is missing columns: {', '.join(missing)}")


def _bootstrap_mean(values: pd.DataFrame, value: str, *, draws: int, rng: np.random.Generator) -> tuple[float, float, float]:
    """Bootstrap a mean by resampling complete races, preserving two-player dependence."""
    grouped = values.groupby("race_key", sort=False)[value].mean().to_numpy(dtype=float)
    if not len(grouped):
        return (float("nan"), float("nan"), float("nan"))
    samples = rng.choice(grouped, size=(draws, len(grouped)), replace=True).mean(axis=1)
    return float(grouped.mean()), float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _treatment_summary(turns: pd.DataFrame, rng: np.random.Generator, draws: int) -> pd.DataFrame:
    rows = []
    for risk in RISK_ORDER:
        subset = turns.loc[turns.max_private_risk.eq(risk)]
        mean, low, high = _bootstrap_mean(subset, "unsafe", draws=draws, rng=rng)
        rows.append({"risk": risk, "unsafe_rate": mean, "ci_low": low, "ci_high": high,
                     "n_races": subset.race_key.nunique(), "n_decisions": len(subset)})
    return pd.DataFrame(rows)


def _welch(left: pd.Series, right: pd.Series) -> tuple[float, float, float]:
    left, right = left.dropna(), right.dropna()
    if len(left) < 2 or len(right) < 2:
        return float("nan"), float("nan"), float("nan")
    result = stats.ttest_ind(left, right, equal_var=False)
    pooled = np.sqrt(((left.var(ddof=1) + right.var(ddof=1)) / 2))
    d = (left.mean() - right.mean()) / pooled if pooled else float("nan")
    return float(left.mean() - right.mean()), float(result.pvalue), float(d)


def _hypotheses(turns: pd.DataFrame, players: pd.DataFrame) -> pd.DataFrame:
    """Paper-inspired associations; all are labelled descriptive for LLM agents."""
    player_rate = turns.groupby(["race_key", "player"], as_index=False).unsafe.mean()
    rows = []
    for left, right in ((0.1, 0.6), (0.1, 0.9), (0.6, 0.9)):
        estimate, p_value, d = _welch(
            player_rate.loc[player_rate.race_key.isin(turns.loc[turns.max_private_risk.eq(left), "race_key"]), "unsafe"],
            player_rate.loc[player_rate.race_key.isin(turns.loc[turns.max_private_risk.eq(right), "race_key"]), "unsafe"],
        )
        rows.append({"hypothesis": f"H1: Unsafe differs, risk {left} vs {right}", "estimate": estimate,
                     "p_value": p_value, "effect_size_d": d, "n_units": len(player_rate),
                     "interpretation": "player-race Unsafe-rate difference (left minus right)"})

    later = turns.loc[turns["round"].gt(1)].copy()
    for name, condition, positive_meaning in (
        ("H2: response to opponent Unsafe", "opponent_prev_action", "Unsafe after opponent Unsafe minus Safe"),
        ("H3: response to falling behind", "race_state", "Unsafe while behind minus ahead"),
    ):
        if name.startswith("H2"):
            summary = later.groupby(["race_key", "player", condition]).unsafe.mean().unstack(condition)
            if {"safe", "unsafe"}.issubset(summary.columns):
                differences = summary["unsafe"] - summary["safe"]
            else:
                differences = pd.Series(dtype=float)
        else:
            summary = later.groupby(["race_key", "player", condition]).unsafe.mean().unstack(condition)
            if {"behind", "ahead"}.issubset(summary.columns):
                differences = summary["behind"] - summary["ahead"]
            else:
                differences = pd.Series(dtype=float)
        test = stats.ttest_1samp(differences, 0.0) if len(differences) >= 2 else None
        rows.append({"hypothesis": name, "estimate": float(differences.mean()) if len(differences) else float("nan"),
                     "p_value": float(test.pvalue) if test else float("nan"), "effect_size_d": float(test.statistic / np.sqrt(len(differences))) if test else float("nan"),
                     "n_units": int(len(differences)), "interpretation": positive_meaning})

    first = turns.loc[turns["round"].eq(1), ["race_key", "player", "unsafe"]].rename(columns={"unsafe": "first_unsafe"})
    later_rates = later.groupby(["race_key", "player"], as_index=False).unsafe.mean().merge(first, on=["race_key", "player"])
    estimate, p_value, d = _welch(later_rates.loc[later_rates.first_unsafe.eq(1), "unsafe"], later_rates.loc[later_rates.first_unsafe.eq(0), "unsafe"])
    rows.append({"hypothesis": "H4: first-round momentum", "estimate": estimate, "p_value": p_value,
                 "effect_size_d": d, "n_units": len(later_rates),
                 "interpretation": "later Unsafe after first-round Unsafe minus Safe"})
    report = pd.DataFrame(rows)
    # The source paper reports all three private-risk pair contrasts together.
    # Keep that family visible and provide the simple Bonferroni audit column.
    report["p_value_bonferroni"] = np.nan
    h1 = report.hypothesis.str.startswith("H1:")
    report.loc[h1, "p_value_bonferroni"] = np.minimum(
        1.0, report.loc[h1, "p_value"].astype(float) * int(h1.sum())
    )
    return report


def _plot_bar(summary: pd.DataFrame, path: Path, title: str, ylabel: str) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    x = np.arange(len(summary))
    y = summary.iloc[:, 1].to_numpy(float)
    if {"ci_low", "ci_high"}.issubset(summary.columns):
        error = np.vstack([y - summary.ci_low, summary.ci_high - y])
        ax.errorbar(x, y, yerr=error, fmt="none", color="#222", capsize=4)
    ax.bar(x, y, color=[COLORS.get(float(r), "#777") for r in summary.risk], width=.62)
    ax.set_xticks(x, [RISK_LABELS.get(float(r), str(r)) for r in summary.risk])
    ax.set_xlabel("Maximum private risk")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_condition(turns: pd.DataFrame, column: str, labels: list[str], path: Path, title: str) -> None:
    data = turns.loc[turns["round"].gt(1)].groupby(column).unsafe.mean().reindex(labels)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.bar(labels, data.to_numpy(float), color="#4c78a8")
    ax.set_ylim(0, 1); ax.set_ylabel("Unsafe rate"); ax.set_title(title)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _gap_label(value: float) -> str:
    if value <= -1.0:
        return "<= -1.0"
    if value >= 1.0:
        return ">= +1.0"
    return f"{value:+.1f}" if value else "0.0"


def _plot_game_state(turns: pd.DataFrame, path: Path) -> None:
    """Paper Figure 2B analogue: both lagged actions and exact race position."""
    later = turns.loc[turns["round"].gt(1)].copy()
    later["gap_bin"] = later.progress_gap_before.astype(float).map(_gap_label)
    gap_order = ["<= -1.0", "-0.5", "0.0", "+0.5", ">= +1.0"]
    action_pairs = [("safe", "safe"), ("safe", "unsafe"), ("unsafe", "safe"), ("unsafe", "unsafe")]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for ax, (own, opponent) in zip(axes.flat, action_pairs):
        subset = later.loc[later.own_prev_action.eq(own) & later.opponent_prev_action.eq(opponent)]
        rates = subset.groupby("gap_bin").unsafe.mean().reindex(gap_order)
        counts = subset.groupby("gap_bin").size().reindex(gap_order, fill_value=0)
        x = np.arange(len(gap_order))
        ax.plot(x, rates, marker="o", color="#4c78a8")
        for index, (rate, count) in enumerate(zip(rates, counts)):
            if pd.notna(rate):
                ax.annotate(f"n={count}", (index, rate), xytext=(0, 7), textcoords="offset points", ha="center", fontsize=8)
        ax.set_title(f"Previous: own {own.upper()}, opponent {opponent.upper()}", fontsize=10)
        ax.set_ylim(0, 1); ax.grid(axis="y", alpha=.2)
        ax.spines[["top", "right"]].set_visible(False)
    for ax in axes[-1]:
        ax.set_xticks(np.arange(len(gap_order)), gap_order)
        ax.set_xlabel("Progress gap before decision (own - opponent)")
    for ax in axes[:, 0]:
        ax.set_ylabel("P(current action = UNSAFE)")
    fig.suptitle("Unsafe choices by previous actions and race position", y=.995)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def _plot_winner_loser_by_risk(players: pd.DataFrame, path: Path) -> None:
    """Paper Figure 2C analogue, preserving each race's winner-loser pair."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.6), sharex=True, sharey=True)
    for ax, risk in zip(axes, RISK_ORDER):
        subset = players.loc[players.max_private_risk.astype(float).round(1).eq(risk)]
        paired = subset.pivot_table(index="race_key", columns="outcome", values="unsafe_frequency", aggfunc="first")
        paired = paired.reindex(columns=["winner", "loser"])
        paired = paired.dropna(subset=["winner", "loser"], how="any")
        ax.scatter(paired.loser, paired.winner, alpha=.72, color=COLORS[risk], edgecolor="white", linewidth=.35)
        ax.axline((0, 0), slope=1, color="#666", linestyle="--", linewidth=1)
        if len(paired) >= 3 and paired.loser.nunique() > 1 and paired.winner.nunique() > 1:
            correlation, p_value = stats.pearsonr(paired.loser, paired.winner)
            fit = np.polyfit(paired.loser, paired.winner, 1)
            line = np.linspace(0, 1, 100)
            ax.plot(line, np.clip(np.polyval(fit, line), 0, 1), color="#222", linewidth=1)
            note = f"r = {correlation:.2f}\np = {p_value:.3f}\nn = {len(paired)}"
        else:
            note = f"n = {len(paired)}\ncorrelation unavailable"
        ax.text(.04, .95, note, transform=ax.transAxes, va="top", fontsize=9,
                bbox={"facecolor": "white", "alpha": .85, "edgecolor": "none"})
        top = inset_axes(ax, width="100%", height="18%", loc="upper center", borderpad=0)
        top.hist(paired.loser, bins=np.linspace(0, 1, 9), color=COLORS[risk], alpha=.55)
        top.set(xticks=[], yticks=[], xlim=(0, 1)); top.patch.set_alpha(0)
        right = inset_axes(ax, width="18%", height="100%", loc="center right", borderpad=0)
        right.hist(paired.winner, bins=np.linspace(0, 1, 9), orientation="horizontal", color=COLORS[risk], alpha=.55)
        right.set(xticks=[], yticks=[], ylim=(0, 1)); right.patch.set_alpha(0)
        ax.set_title(f"Risk {RISK_LABELS[risk]}")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Loser Unsafe frequency")
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("Winner Unsafe frequency")
    fig.suptitle("Unsafe frequency of winners and losers within decisive races", y=.99)
    # Inset marginal histograms are not compatible with matplotlib's tight_layout.
    fig.subplots_adjust(left=.07, right=.99, bottom=.13, top=.82, wspace=.18)
    fig.savefig(path, dpi=180); plt.close(fig)


def _plot_horizon_distribution(races: pd.DataFrame, path: Path) -> None:
    horizons = races.n_rounds.astype(int)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    bins = np.arange(4.5, horizons.max() + 1.6, 1)
    ax.hist(horizons, bins=bins, color="#4c78a8", edgecolor="white")
    ax.axvline(horizons.mean(), color="#111", linestyle="--", label=f"Observed mean = {horizons.mean():.2f}")
    ax.axvline(9, color="#777", linestyle=":", linewidth=2, label="Theoretical mean = 9")
    ax.set(xlabel="Rounds played in race", ylabel="Number of races", title="Distribution of realised race horizons")
    ax.legend(frameon=False); ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout(); fig.savefig(path, dpi=180); plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", action="append", type=Path, required=True, help="run directory or parent directory; repeatable")
    parser.add_argument("--output", type=Path, required=True, help="directory for CSV, JSON, and PNG outputs")
    parser.add_argument("--expected-races", type=int, default=30, help="fail unless this number of races is present; use 0 to disable")
    parser.add_argument("--bootstrap-draws", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260726)
    args = parser.parse_args()

    turns, races, players = _read_runs(_discover_runs(args.input))
    _require(turns, {"game_id", "player", "round", "unsafe", "max_private_risk", "own_prev_action", "opponent_prev_action", "progress_gap_before"}, "turns.jsonl")
    _require(races, {"game_id", "n_rounds", "stop_forced", "parse_failures", "max_private_risk"}, "races.csv")
    _require(players, {"game_id", "player", "outcome", "unsafe_frequency", "max_private_risk"}, "players.csv")
    for frame in (turns, races, players):
        frame["race_key"] = frame["source_run"].astype(str) + "::" + frame.game_id.astype(str)
    turns.max_private_risk = turns.max_private_risk.astype(float).round(1)
    races.max_private_risk = races.max_private_risk.astype(float).round(1)
    if set(races.max_private_risk) != set(RISK_ORDER):
        raise ValueError(f"Expected exactly risk treatments {RISK_ORDER}; found {sorted(set(races.max_private_risk))}")
    if args.expected_races and len(races) != args.expected_races:
        raise ValueError(f"Expected {args.expected_races} races, found {len(races)}")
    if args.expected_races:
        expected_per_risk = args.expected_races // len(RISK_ORDER)
        counts = races.max_private_risk.value_counts().to_dict()
        if args.expected_races % len(RISK_ORDER) or any(
            int(counts.get(risk, 0)) != expected_per_risk for risk in RISK_ORDER
        ):
            raise ValueError(
                f"Expected {expected_per_risk} races in each risk treatment; found {counts}"
            )
    if races.game_id.duplicated().any() and races.source_run.nunique() == 1:
        raise ValueError("races.csv contains duplicate game_id values")
    bad = races.loc[races.stop_forced.astype(int).eq(1) | races.parse_failures.astype(int).gt(0)]
    if len(bad):
        raise ValueError(f"{len(bad)} race(s) have forced stopping or parse failures; do not analyse them as clean behavior")
    turns["race_state"] = np.select([turns.progress_gap_before > 1e-9, turns.progress_gap_before < -1e-9], ["ahead", "behind"], default="tied")
    turns["own_prev_action"] = turns.own_prev_action.fillna("").str.lower()
    turns["opponent_prev_action"] = turns.opponent_prev_action.fillna("").str.lower()

    output = args.output.resolve(); output.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    treatment = _treatment_summary(turns, rng, args.bootstrap_draws)
    hypotheses = _hypotheses(turns, players)
    treatment.to_csv(output / "unsafe_by_risk.csv", index=False)
    hypotheses.to_csv(output / "hypotheses.csv", index=False)
    _plot_bar(treatment, output / "figure_1_unsafe_by_risk.png", "Unsafe choices by private-risk treatment", "Unsafe rate (race-bootstrap 95% CI)")
    _plot_game_state(turns, output / "figure_2b_game_state.png")
    _plot_winner_loser_by_risk(players, output / "figure_2c_winner_loser_by_risk.png")
    _plot_horizon_distribution(races, output / "figure_s1_horizon_distribution.png")
    summary = {"n_races": int(len(races)), "n_decisions": int(len(turns)), "risk_counts": {str(k): int(v) for k, v in races.max_private_risk.value_counts().sort_index().items()}, "figures": 4,
               "note": "Paper-inspired LLM behavioral associations, not a replication claim. Outputs cover Figure 2A, Figure 2B, Figure 2C, and Figure S1 analogues. Inference resamples races; hypotheses H2-H4 are descriptive conditional associations."}
    (output / "report_manifest.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote report for {len(races)} races to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
