"""Descriptive analysis of the two N-player pilot runs under this directory.

Not part of the N-player recording/analysis pipeline proper (see
``results/scripts/analyze_ai_race.py``'s docstring for why that analyzer is
two-player-only, and ``N-Player/PAPER_ANALYSES_AND_PLAN.md`` item 17 for the
still-open gap of a real N-player analyzer). This is a standalone script for
one specific pass: descriptive stats over ``nplayer_nonpersona/`` (neutral
baseline, N=3, risk treatments 0.1/0.6/0.9, 20 reps) and ``nplayer-riskaware/``
(Eckel-Grossman risk-persona sweep R1-R6, symmetric across all 3 seats, same
mechanism/seed, 2 reps), cross-checked qualitatively against
``N-Player/theory``'s stationary-distribution predictions.

Both runs are ``run_phase="pilot"`` -- descriptive only, not confirmatory
evidence (see the repo's ``CLAUDE.md``). Findings are written up in
``ANALYSIS_qwen2.5-14b-instruct.md`` next to this file (named for the one
model both runs use); run this script to regenerate the numbers.

Usage::

    python results/nplayer/analyze_nplayer_results.py
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(REPO_ROOT / "N-Player"))

RISK_RE = re.compile(r"n3_risk_(\d+)__")


def risk_from_game_id(game_id: str) -> float:
    match = RISK_RE.search(game_id)
    return int(match.group(1)) / 100.0


def _load_run_dirs() -> list[Path]:
    dirs = []
    for exp_group in RESULTS_DIR.iterdir():
        if not exp_group.is_dir():
            continue
        for model_dir in exp_group.glob("*/"):
            if not model_dir.is_dir():
                continue
            for run_dir in model_dir.glob("*/"):
                if (run_dir / "run_manifest.json").exists():
                    dirs.append(run_dir)
    return sorted(dirs)


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    races, players, turns = [], [], []
    for run_dir in _load_run_dirs():
        manifest = json.loads((run_dir / "run_manifest.json").read_text())
        source_group = run_dir.parent.parent.name
        r = pd.read_csv(run_dir / "races.csv")
        p = pd.read_csv(run_dir / "players.csv")
        t = pd.read_json(run_dir / "turns.jsonl", lines=True)
        for df in (r, p, t):
            df["source_group"] = source_group
            df["experiment"] = manifest["experiment_name"]
        races.append(r)
        players.append(p)
        turns.append(t)
    return (
        pd.concat(races, ignore_index=True),
        pd.concat(players, ignore_index=True),
        pd.concat(turns, ignore_index=True),
    )


def main() -> None:
    races, players, turns = load_all()
    for df in (races, players, turns):
        df["risk"] = df["game_id"].map(risk_from_game_id)

    print("=" * 70, "\nOVERVIEW\n", "=" * 70, sep="")
    print(f"races: {len(races)}, players: {len(players)}, turns: {len(turns)}")
    print(races.groupby(["source_group", "experiment"]).size())

    print("\n" + "=" * 70, "\nDATA QUALITY: parse failure rate\n", "=" * 70, sep="")
    print(turns.groupby("source_group")["parse_failed"].mean())

    print("\n" + "=" * 70, "\nHORIZON: realized n_rounds (theoretical E[W]=9)\n", "=" * 70, sep="")
    print(races.groupby("source_group")["n_rounds"].describe()[["mean", "std", "min", "max"]])

    print("\n" + "=" * 70, "\nCRN CHECK: n_rounds identical across persona conditions for the same rep+risk\n", "=" * 70, sep="")
    shared_reps = races[races["rep"].isin([0, 1])]
    check = shared_reps.groupby(["risk", "rep"])["n_rounds"].nunique()
    print("distinct n_rounds values per (risk, rep) -- should all be 1:")
    print(check)

    baseline_turns = turns[turns["persona_condition"] == "none"]
    baseline_players = players[players["persona_condition"] == "none"]
    baseline_races = races[races["persona_condition"] == "none"]

    print("\n" + "=" * 70, "\nUNSAFE RATE BY RISK TREATMENT (neutral baseline)\n", "=" * 70, sep="")
    print(baseline_turns.groupby("risk")["unsafe"].agg(["mean", "count"]))

    print("\n" + "=" * 70, "\nOUTCOME DISTRIBUTION BY RISK TREATMENT (neutral baseline)\n", "=" * 70, sep="")
    print(pd.crosstab(baseline_players["risk"], baseline_players["outcome"], normalize="index"))
    print("\nis_full_tie rate by risk:")
    print(baseline_races.groupby("risk")["is_full_tie"].mean())

    print("\n" + "=" * 70, "\nSETBACK RATE AMONG SETBACK-ELIGIBLE PLAYERS (neutral baseline)\n", "=" * 70, sep="")
    eligible = baseline_players[baseline_players["setback_eligible"] == 1]
    print(eligible.groupby("risk")["setback"].agg(["mean", "count"]))

    print("\n" + "=" * 70, "\nPERSONA SWEEP: unsafe rate by persona x risk treatment\n", "=" * 70, sep="")
    persona_turns = turns[turns["persona_condition"] != "none"]
    print(persona_turns.groupby(["persona_condition", "risk"])["unsafe"].agg(["mean", "count"]))
    print("\nPersona main effect (collapsed across risk):")
    print(persona_turns.groupby("persona_condition")["unsafe"].agg(["mean", "count"]))

    print("\n" + "=" * 70, "\nFIRST-ROUND MOMENTUM: unsafe rate, round 1 vs later (neutral baseline)\n", "=" * 70, sep="")
    baseline_turns = baseline_turns.copy()
    baseline_turns["is_round1"] = baseline_turns["round"] == 1
    print(baseline_turns.groupby(["risk", "is_round1"])["unsafe"].mean())

    print("\n" + "=" * 70, "\nPOSITION-DEPENDENT BEHAVIOUR (neutral baseline)\n", "=" * 70, sep="")
    def position(row):
        gap = row["progress_gap_before"]
        if abs(gap) < 1e-9:
            return "tied"
        return "ahead" if gap > 0 else "behind"
    baseline_turns["position"] = baseline_turns.apply(position, axis=1)
    print(baseline_turns.groupby(["risk", "position"])["unsafe"].agg(["mean", "count"]))

    print("\n" + "=" * 70, "\nTHEORY CROSS-CHECK: stationary AU frequency (n=3,s=1.5,b=4,c=1,B=100,W=9)\n", "=" * 70, sep="")
    from theory.stationary import stationary_distribution
    for pr in (0.1, 0.6, 0.9):
        stationary = stationary_distribution(
            n=3, s=1.5, b=4.0, c=1.0, B=100.0, W=9.0, z=100, beta=0.1, pr=pr
        )
        empirical = baseline_turns.loc[baseline_turns["risk"] == pr, "unsafe"].mean()
        print(
            f"pr={pr}: theory AU_stationary={stationary['AU']:.3f}  "
            f"empirical_unsafe_rate={empirical:.3f}"
        )


if __name__ == "__main__":
    main()
