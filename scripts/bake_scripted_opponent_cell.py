"""Create a one-cell Kaggle task source from the canonical task.

Kaggle Benchmark task runs do not inherit the caller's environment variables.
The canonical task therefore exposes selectors for local checks, while a remote
cell must have the selected strategy and risk baked into the uploaded source.
This helper changes only those two selector assignments and refuses any source
that does not contain the expected canonical lines.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "kaggle" / "benchmarks" / "ai_race_scripted_opponent.py"
EXPECTED_STRATEGIES = 'STRATEGIES = _selected("AI_RACE_STRATEGIES", STRATEGIES_FROZEN, str)'
EXPECTED_RISKS = 'RISK_LEVELS = _selected("AI_RACE_RISKS", RISK_LEVELS_FROZEN, float)'
ALLOWED_STRATEGIES = {"AS", "AU", "CS", "CAS"}
ALLOWED_RISKS = {0.1, 0.6, 0.9}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=sorted(ALLOWED_STRATEGIES), required=True)
    parser.add_argument("--risk", type=float, choices=sorted(ALLOWED_RISKS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=CANONICAL)
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    if source.count(EXPECTED_STRATEGIES) != 1 or source.count(EXPECTED_RISKS) != 1:
        raise SystemExit(
            "canonical selector lines were not found exactly once; refusing to bake "
            "an unverified task source"
        )

    risk_literal = repr(float(args.risk))
    baked = source.replace(
        EXPECTED_STRATEGIES,
        f"STRATEGIES = ({args.strategy!r},)",
    ).replace(
        EXPECTED_RISKS,
        f"RISK_LEVELS = ({risk_literal},)",
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(baked, encoding="utf-8", newline="\n")
    print(f"baked {args.strategy} risk {args.risk:g} from {args.source}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
