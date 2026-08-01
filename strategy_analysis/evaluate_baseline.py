"""Evaluate the nearest-strategy (Hamming distance) classifier against a
labelled synthetic dataset produced by ``generate_dataset.py``.

This is a robustness check on ``classify_trajectory`` itself, not a claim
about LLM behaviour: at ``noise_p = 0`` accuracy must be 1.0 by construction
(the dataset is generated from the same canonical rules the classifier
predicts), and degradation under noise quantifies how much per-round
execution noise the nearest-strategy rule can tolerate before ties or
misclassification appear.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from strategy_analysis.classify import classify_trajectory


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                records.append(json.loads(line))
    return records


def evaluate(records: list[dict[str, Any]]) -> dict[str, Any]:
    by_noise: dict[float, list[dict[str, Any]]] = defaultdict(list)

    for record in records:
        result = classify_trajectory(record["own_actions"], record["opponent_actions"])
        true_strategy = record["true_strategy"]
        tied = true_strategy in result.best_strategies
        exact = result.unique_best_strategy == true_strategy
        by_noise[record["noise_p"]].append(
            {
                "true_strategy": true_strategy,
                "best_strategies": result.best_strategies,
                "tied_correct": tied,
                "exact_correct": exact,
                "n_tied": len(result.best_strategies),
            }
        )

    summary: dict[str, Any] = {"noise_levels": {}}
    for noise_p, rows in sorted(by_noise.items()):
        n = len(rows)
        tied_correct = sum(r["tied_correct"] for r in rows)
        exact_correct = sum(r["exact_correct"] for r in rows)
        confusion: Counter[tuple[str, str]] = Counter()
        for row in rows:
            for predicted in row["best_strategies"]:
                confusion[(row["true_strategy"], predicted)] += 1
        per_strategy: dict[str, dict[str, float]] = {}
        for strategy in sorted({r["true_strategy"] for r in rows}):
            strategy_rows = [r for r in rows if r["true_strategy"] == strategy]
            per_strategy[strategy] = {
                "n": len(strategy_rows),
                "tied_accuracy": sum(r["tied_correct"] for r in strategy_rows)
                / len(strategy_rows),
                "exact_accuracy": sum(r["exact_correct"] for r in strategy_rows)
                / len(strategy_rows),
            }
        summary["noise_levels"][str(noise_p)] = {
            "n": n,
            "tied_accuracy": tied_correct / n,
            "exact_accuracy": exact_correct / n,
            "mean_tie_width": sum(r["n_tied"] for r in rows) / n,
            "per_strategy": per_strategy,
            "confusion": {f"{t}->{p}": c for (t, p), c in sorted(confusion.items())},
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset", type=Path, help="JSONL from generate_dataset.py")
    parser.add_argument("--output", type=Path, help="write JSON summary here (default: stdout)")
    args = parser.parse_args(argv)

    records = _read_jsonl(args.dataset)
    if not records:
        raise ValueError(f"{args.dataset}: no records found")
    summary = evaluate(records)

    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.output is None:
        print(text)
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
        print(f"wrote summary to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
