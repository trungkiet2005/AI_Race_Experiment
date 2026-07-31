#!/usr/bin/env python3
"""Report whether a pilot run collapsed into mirrored play before scaling it up.

Both seats are the same model reading structurally identical prompts, so at low
sampling temperature they can pick the same action every round. When that happens

    progress_gap(t) = 0.5 * (own Unsafe count - opponent Unsafe count) = 0

for the whole race: every decision is taken while tied, the race-position
estimand has no variation left to explain, and the panel logit has no Safe/Unsafe
contrast to fit. The failure is invisible in an aggregate Unsafe frequency, which
is exactly why it has to be checked on a small pilot rather than discovered after
the full grid has been paid for.

This script only describes the pilot; it never edits or filters a run.

    python results/scripts/check_symmetry.py --input <run-dir> [--max-degenerate 0.4]

Exit status is 1 when the degenerate share exceeds the threshold, so the check can
gate a scale-up step.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

TOLERANCE = 1e-9


def _iter_turn_files(inputs: Iterable[Path]) -> list[Path]:
    found: list[Path] = []
    for entry in inputs:
        if entry.is_file() and entry.name == "turns.jsonl":
            found.append(entry)
        elif entry.is_dir():
            found.extend(sorted(entry.rglob("turns.jsonl")))
    if not found:
        raise SystemExit(f"no turns.jsonl was found under {[str(p) for p in inputs]}")
    return found


def _load_races(paths: Iterable[Path]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    races: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        source_run = path.parent.resolve().as_posix()
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise SystemExit(f"{path}:{line_number}: invalid JSON: {exc}")
                races[(source_run, str(record.get("game_id", "")))].append(record)
    return races


def _summarise_race(turns: list[dict[str, Any]]) -> dict[str, Any]:
    by_round: dict[int, dict[int, dict[str, Any]]] = defaultdict(dict)
    for turn in turns:
        by_round[int(turn["round"])][int(turn["player_index"])] = turn

    complete_rounds = sorted(
        round_number
        for round_number, seats in by_round.items()
        if set(seats) == {0, 1}
    )
    matched = 0
    for round_number in complete_rounds:
        seats = by_round[round_number]
        if int(seats[0]["unsafe"]) == int(seats[1]["unsafe"]):
            matched += 1

    gaps = [
        float(turn.get("progress_gap_before") or 0.0)
        for turn in turns
        if turn.get("progress_gap_before") is not None
    ]
    final_gaps = [
        abs(float(by_round[round_number][0].get("progress_gap_after") or 0.0))
        for round_number in complete_rounds[-1:]
    ]
    return {
        "n_rounds": len(complete_rounds),
        "matched_rounds": matched,
        "always_tied": all(abs(gap) <= TOLERANCE for gap in gaps) if gaps else True,
        "final_abs_gap": final_gaps[0] if final_gaps else 0.0,
        "parse_failed": any(bool(turn.get("parse_failed")) for turn in turns),
        "condition": (
            str(turns[0].get("persona_condition", "")),
            str(turns[0].get("model", "")),
            float(turns[0].get("max_private_risk", float("nan"))),
        ),
    }


def _format_share(numerator: int, denominator: int) -> str:
    if denominator == 0:
        return "n/a"
    return f"{numerator}/{denominator} ({numerator / denominator:.1%})"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        action="append",
        dest="inputs",
        type=Path,
        required=True,
        help="run directory or turns.jsonl; repeat to combine sources",
    )
    parser.add_argument(
        "--max-degenerate",
        type=float,
        default=0.4,
        help=(
            "fail when this share of races stayed tied for their whole duration "
            "(default: 0.4)"
        ),
    )
    args = parser.parse_args(argv)

    races = _load_races(_iter_turn_files(args.inputs))
    summaries = {key: _summarise_race(turns) for key, turns in races.items()}
    if not summaries:
        raise SystemExit("no races were recovered from the supplied inputs")

    by_condition: dict[tuple[str, str, float], list[dict[str, Any]]] = defaultdict(list)
    for summary in summaries.values():
        by_condition[summary["condition"]].append(summary)

    print(f"races: {len(summaries)}")
    print(f"parse-contaminated races: {sum(s['parse_failed'] for s in summaries.values())}")
    print()
    print(f"{'persona':<10}{'model':<24}{'risk':>6}{'races':>8}{'tied-throughout':>18}{'mirrored rounds':>18}")
    for condition in sorted(by_condition):
        persona, model, risk = condition
        group = by_condition[condition]
        degenerate = sum(1 for summary in group if summary["always_tied"])
        rounds = sum(summary["n_rounds"] for summary in group)
        matched = sum(summary["matched_rounds"] for summary in group)
        print(
            f"{persona or '-':<10}{model[:23]:<24}{risk:>6.2f}{len(group):>8}"
            f"{_format_share(degenerate, len(group)):>18}"
            f"{_format_share(matched, rounds):>18}"
        )

    total_degenerate = sum(1 for summary in summaries.values() if summary["always_tied"])
    share = total_degenerate / len(summaries)
    print()
    print(f"overall tied-throughout share: {share:.1%}")

    final_gaps = sorted(summary["final_abs_gap"] for summary in summaries.values())
    if final_gaps:
        middle = final_gaps[len(final_gaps) // 2]
        print(f"median |final progress gap|: {middle:.2f}")

    if share > args.max_degenerate:
        print(
            f"\nFAIL: {share:.1%} of races never left a tie, above the "
            f"{args.max_degenerate:.1%} threshold. The race-position estimand has no "
            "variation to explain at this setting. Raise the sampling temperature, "
            "use an asymmetric persona cell, or run against a scripted opponent "
            "before scaling up.",
            file=sys.stderr,
        )
        return 1
    print("\nOK: enough races separated for the race-position estimand.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
