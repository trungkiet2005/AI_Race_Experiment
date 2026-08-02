"""Exhaustively verify the positive payoff-scale invariance contract."""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_race.audit.payoff_scale import PAYOFF_SCALES, PAYOFF_SCALE_PROTOCOL
from ai_race.engine.scoring import terminal_scoring
from ai_race.engine.state import Action, GameConfig


DEFAULT_OUTPUT = ROOT / "results" / "derived" / "payoff_scale_contract"
JOINT_ACTIONS = tuple(itertools.product((Action.SAFE, Action.UNSAFE), repeat=2))
RISKS = (0.1, 0.6, 0.9)
DRAW_PAIRS = ((0.0, 0.0), (0.37, 0.83), (0.83, 0.37), (1.0, 1.0))


def trajectory_state(config: GameConfig, sequence: tuple[tuple[Action, Action], ...]):
    progress = [0.0, 0.0]
    stage = [0.0, 0.0]
    unsafe = [0, 0]
    for left, right in sequence:
        progress[0] += config.progress_for(left)
        progress[1] += config.progress_for(right)
        stage[0] += config.stage_payoff(left, right)
        stage[1] += config.stage_payoff(right, left)
        unsafe[0] += left.unsafe
        unsafe[1] += right.unsafe
    return progress, stage, unsafe


def analyze(max_rounds: int) -> tuple[list[dict], dict]:
    config = GameConfig(name="payoff-scale-contract")
    stats: dict[tuple[float, float], dict[str, float | int]] = defaultdict(
        lambda: {
            "comparisons": 0,
            "non_payoff_mismatches": 0,
            "max_normalized_prize_error": 0.0,
            "max_normalized_final_payoff_error": 0.0,
        }
    )
    sequence_count = 0
    for rounds in range(1, max_rounds + 1):
        for sequence in itertools.product(JOINT_ACTIONS, repeat=rounds):
            sequence_count += 1
            progress, stage, unsafe = trajectory_state(config, sequence)
            for risk in RISKS:
                for draws in DRAW_PAIRS:
                    reference = terminal_scoring(
                        stage_payoffs=stage,
                        progress=progress,
                        unsafe_counts=unsafe,
                        rounds_played=rounds,
                        max_private_risk=risk,
                        race_prize=config.race_prize,
                        setback_draws=draws,
                    )
                    for scale in PAYOFF_SCALES:
                        scaled = terminal_scoring(
                            stage_payoffs=[value * scale for value in stage],
                            progress=progress,
                            unsafe_counts=unsafe,
                            rounds_played=rounds,
                            max_private_risk=risk,
                            race_prize=config.race_prize * scale,
                            setback_draws=draws,
                        )
                        row = stats[(risk, scale)]
                        row["comparisons"] += 1
                        non_payoff_match = (
                            scaled.outcomes == reference.outcomes
                            and scaled.private_risks == reference.private_risks
                            and scaled.setback_eligible == reference.setback_eligible
                            and scaled.setback_draws == reference.setback_draws
                            and scaled.setbacks == reference.setbacks
                        )
                        if not non_payoff_match:
                            row["non_payoff_mismatches"] += 1
                        prize_error = max(
                            abs(value / scale - expected)
                            for value, expected in zip(scaled.prizes, reference.prizes)
                        )
                        payoff_error = max(
                            abs(value / scale - expected)
                            for value, expected in zip(
                                scaled.final_payoffs, reference.final_payoffs
                            )
                        )
                        row["max_normalized_prize_error"] = max(
                            float(row["max_normalized_prize_error"]), prize_error
                        )
                        row["max_normalized_final_payoff_error"] = max(
                            float(row["max_normalized_final_payoff_error"]),
                            payoff_error,
                        )
    rows = [
        {
            "max_private_risk": risk,
            "payoff_scale": scale,
            **stats[(risk, scale)],
            "admitted": (
                stats[(risk, scale)]["non_payoff_mismatches"] == 0
                and stats[(risk, scale)]["max_normalized_prize_error"] <= 1e-12
                and stats[(risk, scale)]["max_normalized_final_payoff_error"]
                <= 1e-12
            ),
        }
        for risk in RISKS
        for scale in PAYOFF_SCALES
    ]
    summary = {
        "schema_version": "ai-race.payoff-scale-contract.v1",
        "protocol": PAYOFF_SCALE_PROTOCOL,
        "max_rounds_exhaustive": max_rounds,
        "joint_action_sequences": sequence_count,
        "risk_levels": list(RISKS),
        "setback_draw_pairs": [list(pair) for pair in DRAW_PAIRS],
        "payoff_scales": list(PAYOFF_SCALES),
        "terminal_comparisons": sum(int(row["comparisons"]) for row in rows),
        "all_cells_admitted": all(bool(row["admitted"]) for row in rows),
        "max_normalized_final_payoff_error": max(
            float(row["max_normalized_final_payoff_error"]) for row in rows
        ),
    }
    return rows, summary


def write_outputs(output: Path, rows: list[dict], summary: dict) -> None:
    output.mkdir(parents=True, exist_ok=True)
    csv_path = output / "scale_contract.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary["scale_contract_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    status = "PASS" if summary["all_cells_admitted"] else "FAIL"
    (output / "README.md").write_text(
        "\n".join(
            [
                "# Payoff-scale mechanical contract",
                "",
                f"**Admission: {status}.**",
                "",
                f"- Exhaustive joint-action sequences: {summary['joint_action_sequences']:,}",
                f"- Terminal comparisons: {summary['terminal_comparisons']:,}",
                f"- Maximum normalized final-payoff error: {summary['max_normalized_final_payoff_error']:.3g}",
                "- Scope: all trajectories through the reported round bound, three risk levels, four draw pairs, and four positive scales.",
                "- Meaning: the code treatment preserves the game mechanically; it does not predict that an LLM will behave invariantly.",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-rounds", type=int, default=7)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.max_rounds < 1:
        raise SystemExit("--max-rounds must be positive")
    rows, summary = analyze(args.max_rounds)
    write_outputs(args.output.resolve(), rows, summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary["all_cells_admitted"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
