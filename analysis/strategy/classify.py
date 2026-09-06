"""Nearest-strategy classification for repeated binary AI Race trajectories.

Safe is encoded as ``0`` and Unsafe as ``1``.  The four canonical strategies are
the reduced strategy set in Fernández Domingos and Han (2026):

* AS: always Safe.
* AU: always Unsafe.
* CS: Safe in round 1, then copy the opponent's previous action.
* CAS: Unsafe in round 1, then copy the opponent's previous action.

Classification is descriptive: it finds the strategy trajectory with the smallest
Hamming distance from the observed trajectory.  Ties are retained because short or
noisy variable-horizon trajectories need not identify a unique nearest strategy.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from numbers import Integral, Real
from pathlib import Path
from typing import Any, Iterable, Sequence


SAFE = 0
UNSAFE = 1
CANONICAL_STRATEGIES = ("AS", "AU", "CS", "CAS")

# This rule is useful for the LLM study but is not part of the paper's canonical
# four-strategy reduction.  Keeping "EXPLORATORY" in the public label prevents it
# from being mistaken for a preregistered or paper-defined strategy.
EXPLORATORY_BEHIND_STRATEGY = "BEHIND_UNSAFE_EXPLORATORY"


@dataclass(frozen=True)
class StrategyMatch:
    """Distance from one observed trajectory to one candidate strategy."""

    strategy: str
    predicted_actions: tuple[int, ...]
    mismatches: int
    mismatch_rate: float
    canonical: bool


@dataclass(frozen=True)
class ClassificationResult:
    """Complete nearest-strategy result, including all tied nearest strategies."""

    horizon: int
    best_strategies: tuple[str, ...]
    matches: tuple[StrategyMatch, ...]

    @property
    def unique_best_strategy(self) -> str | None:
        """Return the nearest strategy only when the minimum is unique."""

        return self.best_strategies[0] if len(self.best_strategies) == 1 else None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serialisable representation."""

        return {
            "horizon": self.horizon,
            "best_strategies": list(self.best_strategies),
            "unique_best_strategy": self.unique_best_strategy,
            "matches": [
                {
                    **asdict(match),
                    "predicted_actions": list(match.predicted_actions),
                }
                for match in self.matches
            ],
        }


def _as_binary_action(value: Any, *, field: str) -> int:
    """Normalise a Safe/Unsafe value without silently accepting other numbers."""

    if isinstance(value, bool):
        return int(value)
    if isinstance(value, Integral) and value in (SAFE, UNSAFE):
        return int(value)
    if isinstance(value, Real) and math.isfinite(float(value)) and value in (0.0, 1.0):
        return int(value)
    if isinstance(value, str):
        normalised = value.strip().lower()
        if normalised in {"0", "s", "safe"}:
            return SAFE
        if normalised in {"1", "u", "unsafe"}:
            return UNSAFE
    raise ValueError(
        f"{field} must contain only binary actions (0/Safe or 1/Unsafe); "
        f"received {value!r}"
    )


def normalise_actions(values: Iterable[Any], *, field: str = "actions") -> tuple[int, ...]:
    """Convert an iterable of binary Safe/Unsafe values to an immutable trajectory."""

    return tuple(
        _as_binary_action(value, field=f"{field}[{index}]")
        for index, value in enumerate(values)
    )


def _normalise_gaps(values: Iterable[Any], *, horizon: int) -> tuple[float, ...]:
    gaps: list[float] = []
    for index, value in enumerate(values):
        if isinstance(value, bool):
            raise ValueError(f"progress_gaps_before[{index}] must be numeric, not bool")
        try:
            gap = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"progress_gaps_before[{index}] must be numeric; received {value!r}"
            ) from exc
        if not math.isfinite(gap):
            raise ValueError(
                f"progress_gaps_before[{index}] must be finite; received {value!r}"
            )
        gaps.append(gap)
    if len(gaps) != horizon:
        raise ValueError(
            "progress_gaps_before must have the same length as the action "
            f"trajectories ({len(gaps)} != {horizon})"
        )
    return tuple(gaps)


def predict_strategy(
    strategy: str,
    opponent_actions: Sequence[Any],
    *,
    progress_gaps_before: Sequence[Any] | None = None,
) -> tuple[int, ...]:
    """Predict a strategy's actions over the opponent's observed horizon.

    ``progress_gaps_before`` is required only for the explicitly exploratory rule.
    A negative gap means that the focal player is behind before the decision.
    """

    opponent = normalise_actions(opponent_actions, field="opponent_actions")
    if not opponent:
        raise ValueError("a trajectory must contain at least one round")

    if strategy == "AS":
        return (SAFE,) * len(opponent)
    if strategy == "AU":
        return (UNSAFE,) * len(opponent)
    if strategy == "CS":
        return (SAFE, *opponent[:-1])
    if strategy == "CAS":
        return (UNSAFE, *opponent[:-1])
    if strategy == EXPLORATORY_BEHIND_STRATEGY:
        if progress_gaps_before is None:
            raise ValueError(
                f"{EXPLORATORY_BEHIND_STRATEGY} requires progress_gaps_before"
            )
        gaps = _normalise_gaps(progress_gaps_before, horizon=len(opponent))
        return tuple(UNSAFE if gap < 0 else SAFE for gap in gaps)
    valid = (*CANONICAL_STRATEGIES, EXPLORATORY_BEHIND_STRATEGY)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {valid}")


def classify_trajectory(
    own_actions: Sequence[Any],
    opponent_actions: Sequence[Any],
    *,
    include_exploratory_behind: bool = False,
    progress_gaps_before: Sequence[Any] | None = None,
) -> ClassificationResult:
    """Classify one variable-horizon trajectory by minimum mismatch rate.

    Raw mismatch counts and rates have the same ordering within a trajectory because
    all candidate strategies are evaluated on the same observed horizon.  The rate is
    also returned so distances can be compared descriptively across different horizons.
    """

    own = normalise_actions(own_actions, field="own_actions")
    opponent = normalise_actions(opponent_actions, field="opponent_actions")
    if not own:
        raise ValueError("a trajectory must contain at least one round")
    if len(own) != len(opponent):
        raise ValueError(
            "own_actions and opponent_actions must have the same length "
            f"({len(own)} != {len(opponent)})"
        )

    candidates = list(CANONICAL_STRATEGIES)
    if include_exploratory_behind:
        if progress_gaps_before is None:
            raise ValueError(
                "progress_gaps_before is required when "
                "include_exploratory_behind=True"
            )
        candidates.append(EXPLORATORY_BEHIND_STRATEGY)

    matches: list[StrategyMatch] = []
    for strategy in candidates:
        predicted = predict_strategy(
            strategy,
            opponent,
            progress_gaps_before=progress_gaps_before,
        )
        mismatches = sum(observed != expected for observed, expected in zip(own, predicted))
        matches.append(
            StrategyMatch(
                strategy=strategy,
                predicted_actions=predicted,
                mismatches=mismatches,
                mismatch_rate=mismatches / len(own),
                canonical=strategy in CANONICAL_STRATEGIES,
            )
        )

    minimum = min(match.mismatches for match in matches)
    best = tuple(match.strategy for match in matches if match.mismatches == minimum)
    return ClassificationResult(
        horizon=len(own),
        best_strategies=best,
        matches=tuple(matches),
    )


def _iter_trajectory_records(path: Path) -> Iterable[tuple[int, dict[str, Any]]]:
    """Yield one trajectory object per JSON or JSONL record."""

    if path.suffix.lower() == ".jsonl":
        with path.open(encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
                if not isinstance(record, dict):
                    raise ValueError(f"{path}:{line_number}: expected a JSON object")
                yield line_number, record
        return

    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    records = payload if isinstance(payload, list) else [payload]
    for index, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise ValueError(f"{path}: record {index} is not a JSON object")
        yield index, record


def _write_jsonl(records: Iterable[dict[str, Any]], output: Path | None) -> None:
    lines = [json.dumps(record, ensure_ascii=False, sort_keys=True) for record in records]
    text = "\n".join(lines) + ("\n" if lines else "")
    if output is None:
        print(text, end="")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    """Classify one or more trajectory records from JSON/JSONL."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "input",
        type=Path,
        help=(
            "JSON or JSONL containing own_actions and opponent_actions arrays; "
            "an optional trajectory_id is copied to output"
        ),
    )
    parser.add_argument("--output", type=Path, help="write JSONL here (default: stdout)")
    parser.add_argument(
        "--include-exploratory-behind",
        action="store_true",
        help=(
            "also evaluate BEHIND_UNSAFE_EXPLORATORY; each input record must include "
            "progress_gaps_before"
        ),
    )
    args = parser.parse_args(argv)

    output_records: list[dict[str, Any]] = []
    for record_number, record in _iter_trajectory_records(args.input):
        try:
            result = classify_trajectory(
                record["own_actions"],
                record["opponent_actions"],
                include_exploratory_behind=args.include_exploratory_behind,
                progress_gaps_before=record.get("progress_gaps_before"),
            )
        except KeyError as exc:
            raise ValueError(
                f"{args.input}: record {record_number} is missing {exc.args[0]!r}"
            ) from exc
        output_records.append(
            {
                "trajectory_id": record.get("trajectory_id", record_number),
                **result.to_dict(),
            }
        )

    _write_jsonl(output_records, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
