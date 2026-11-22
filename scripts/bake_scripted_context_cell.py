"""Create a one-cell Kaggle task source from the scripted context-mapping task.

Kaggle Benchmark task runs do not inherit the caller's environment variables, so
a remote cell needs its strategy, risk, context and mapping baked into the
uploaded source.  This helper changes only those four selector assignments (and,
for a pre-declared replication, the seed line) and refuses any source that does
not contain each canonical line exactly once.
"""

from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CANONICAL = ROOT / "kaggle" / "benchmarks" / "ai_race_scripted_context_mapping.py"
EXPECTED_STRATEGIES = 'STRATEGIES = _selected("AI_RACE_STRATEGIES", STRATEGIES_FROZEN, str)'
EXPECTED_RISKS = 'RISK_LEVELS = _selected("AI_RACE_RISKS", RISK_LEVELS_FROZEN, float)'
EXPECTED_CONTEXT = 'CONTEXT = os.environ.get("AI_RACE_CONTEXT", "technology_race").strip()'
EXPECTED_MAPPING = 'MAPPING = os.environ.get("AI_RACE_MAPPING", "P_SAFE_Q_UNSAFE").strip()'
EXPECTED_SEED = 'BASE_SEED = int(os.environ.get("AI_RACE_SEED", "260726"))'
ALLOWED_STRATEGIES = {"AS", "AU", "CS", "CAS"}
ALLOWED_RISKS = {0.1, 0.6, 0.9}
ALLOWED_CONTEXTS = {"technology_race", "abstract_game"}
ALLOWED_MAPPINGS = {"P_SAFE_Q_UNSAFE", "Q_SAFE_P_UNSAFE"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=sorted(ALLOWED_STRATEGIES), required=True)
    parser.add_argument("--risk", type=float, choices=sorted(ALLOWED_RISKS), required=True)
    parser.add_argument("--context", choices=sorted(ALLOWED_CONTEXTS), required=True)
    parser.add_argument("--mapping", choices=sorted(ALLOWED_MAPPINGS), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--source", type=Path, default=CANONICAL)
    parser.add_argument("--base-seed", type=int, default=None,
                        help="Seed block for a pre-declared replication; omit to keep "
                             "the canonical block that pairs with every other campaign.")
    args = parser.parse_args()

    source = args.source.read_text(encoding="utf-8")
    expected = (EXPECTED_STRATEGIES, EXPECTED_RISKS, EXPECTED_CONTEXT, EXPECTED_MAPPING)
    if any(source.count(line) != 1 for line in expected):
        raise SystemExit(
            "canonical selector lines were not found exactly once; refusing to bake "
            "an unverified task source"
        )

    baked = (
        source.replace(EXPECTED_STRATEGIES, f"STRATEGIES = ({args.strategy!r},)")
        .replace(EXPECTED_RISKS, f"RISK_LEVELS = ({float(args.risk)!r},)")
        .replace(EXPECTED_CONTEXT, f"CONTEXT = {args.context!r}")
        .replace(EXPECTED_MAPPING, f"MAPPING = {args.mapping!r}")
    )
    if args.base_seed is not None:
        if baked.count(EXPECTED_SEED) != 1:
            raise SystemExit("canonical seed line not found exactly once; refusing to bake")
        baked = baked.replace(EXPECTED_SEED, f"BASE_SEED = {int(args.base_seed)}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(baked, encoding="utf-8", newline="\n")
    print(f"baked {args.strategy} risk {args.risk:g} {args.context} {args.mapping} from {args.source}")
    print(f"wrote {args.output}")


if __name__ == "__main__":
    main()
