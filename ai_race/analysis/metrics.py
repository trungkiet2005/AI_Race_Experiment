"""Descriptive AI Race metrics with no pandas dependency."""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable


def _row(record: Any) -> dict[str, Any]:
    return record.to_dict() if hasattr(record, "to_dict") else dict(record)


def unsafe_rate(turns: Iterable[Any]) -> float:
    values = [int(_row(turn).get("unsafe", 0)) for turn in turns]
    return sum(values) / len(values) if values else 0.0


def unsafe_rate_by_risk(turns: Iterable[Any]) -> dict[float, float]:
    grouped: dict[float, list[int]] = defaultdict(list)
    for turn in turns:
        row = _row(turn)
        grouped[float(row["max_private_risk"])].append(int(row["unsafe"]))
    return {
        risk: sum(values) / len(values)
        for risk, values in sorted(grouped.items())
        if values
    }


def transition_unsafe_rates(turns: Iterable[Any]) -> dict[str, float]:
    """Unsafe frequency conditional on the opponent's previous action."""
    grouped: dict[str, list[int]] = defaultdict(list)
    for turn in turns:
        row = _row(turn)
        previous = row.get("opponent_prev_action")
        if previous in {"safe", "unsafe"}:
            grouped[str(previous)].append(int(row["unsafe"]))
    return {
        previous: sum(values) / len(values)
        for previous, values in sorted(grouped.items())
        if values
    }


def position_label(progress_gap_before: float, tolerance: float = 1e-9) -> str:
    gap = float(progress_gap_before)
    if gap > tolerance:
        return "ahead"
    if gap < -tolerance:
        return "behind"
    return "tied"


def position_unsafe_rates(turns: Iterable[Any]) -> dict[str, float]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for turn in turns:
        row = _row(turn)
        label = position_label(float(row["progress_gap_before"]))
        grouped[label].append(int(row["unsafe"]))
    return {
        label: sum(values) / len(values)
        for label, values in sorted(grouped.items())
        if values
    }


def first_round_momentum(turns: Iterable[Any]) -> dict[str, float]:
    """Later-round Unsafe rate grouped by each player's first action."""
    rows = [_row(turn) for turn in turns]
    first: dict[tuple[str, str], int] = {}
    for row in rows:
        if int(row["round"]) == 1:
            first[(str(row["game_id"]), str(row["player"]))] = int(row["unsafe"])
    grouped: dict[str, list[int]] = defaultdict(list)
    for row in rows:
        if int(row["round"]) < 2:
            continue
        initial = first.get((str(row["game_id"]), str(row["player"])))
        if initial is not None:
            grouped["first_unsafe" if initial else "first_safe"].append(int(row["unsafe"]))
    return {
        label: sum(values) / len(values)
        for label, values in sorted(grouped.items())
        if values
    }


def parse_failure_rate(turns: Iterable[Any]) -> float:
    failures = [int(bool(_row(turn).get("parse_failed", False))) for turn in turns]
    return sum(failures) / len(failures) if failures else 0.0

