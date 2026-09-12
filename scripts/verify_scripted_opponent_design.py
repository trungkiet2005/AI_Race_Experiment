"""Check the scripted-opponent task offline, before it costs a single request.

A benchmark task is the one piece of this study that cannot be debugged where it
runs: the server gives no environment back, a failed run still spends quota, and
an opponent that is subtly wrong produces data that looks fine and answers a
different question.  So the mechanism is exercised here, against a stub model,
and the properties that make the design interpretable are asserted rather than
inspected.

What is checked, and why each one matters.

The four strategies must be exactly the four the evolutionary lane of this study
is built on, and each must behave as its name says.  A conditional strategy in
particular must answer the ROUTE's previous move and not its own, because a
strategy that answered itself would be a constant and the whole conditional arm
would be measuring nothing.

The route's seat must alternate across repetitions and both seats must be used
equally, because this study has measured a seat effect and a fixed seat would
fold it into every number.

The horizon must be shared across strategies within a repetition, so that a
route's play against Always Safe and against Always Unsafe can be differenced
inside a repetition with the horizon draw removed from the comparison.

The mechanism must otherwise be the neutral baseline's, unchanged, or results
here cannot be set beside results there.
"""

from __future__ import annotations

import importlib.util
import os
import sys
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK = ROOT / "kaggle" / "benchmarks" / "ai_race_scripted_opponent.py"
BASELINE = ROOT / "kaggle" / "benchmarks" / "ai_race_baseline.py"

# Every game-mechanism field must agree with the neutral baseline.  Decoding
# belongs to a separate contract: the resumption amendment may change its
# output cap, but the manifest must expose that change rather than hiding it.
MECHANISM_FIELDS = (
    "SAFE", "UNSAFE", "N_PLAYERS", "PLAYER_NAMES", "PROMPT_VERSION",
    "MIN_ROUNDS", "STOP_PROBABILITY", "PRIZE", "PROGRESS", "STAGE_PAYOFF",
    "BASE_SEED", "HORIZON_STREAM", "SETBACK_STREAM", "RISK_LEVELS_FROZEN",
)

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not ok:
        failures.append(name)


def load(path: Path, name: str):
    """Import a task file without running its ``__main__`` section."""
    os.environ.setdefault("LLM_DEFAULT", "offline/stub")
    # The task deliberately ends in an unguarded call, because Kaggle executes
    # benchmark source as a module rather than as a script.  Reading the file
    # here means dropping that call, not adding a guard the server would then
    # have to work around.
    source = path.read_text(encoding="utf-8")
    marker = ".run(kbench.llm)"
    if marker in source:
        source = source[: source.rindex(marker)].rsplit(chr(10), 1)[0]
    module = importlib.util.module_from_spec(
        importlib.util.spec_from_loader(name, loader=None)
    )
    module.__file__ = str(path)
    sys.modules[name] = module

    # The SDK's task decorator inspects the return annotation and registers the
    # function with a server it cannot reach from here.  Standing in for it
    # leaves every constant and every helper exactly as the server will see
    # them, which is the only part this check is about.
    real = sys.modules.get("kaggle_benchmarks")
    sys.modules["kaggle_benchmarks"] = _StubSdk()
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        if real is None:
            sys.modules.pop("kaggle_benchmarks", None)
        else:
            sys.modules["kaggle_benchmarks"] = real
    return module


class _StubSdk(types.ModuleType):
    """Enough of the benchmark SDK to import a task without registering it."""

    llm = None

    def __init__(self) -> None:
        super().__init__("kaggle_benchmarks")

    @staticmethod
    def task(*_args, **_kwargs):
        def decorator(func):
            func.run = lambda *a, **k: None
            return func

        return decorator


def main() -> None:
    task = load(TASK, "scripted_task")
    baseline = load(BASELINE, "baseline_task")

    check("the task declares its own protocol",
          task.PROTOCOL_ID == "ai-race-scripted-opponent-v1", task.PROTOCOL_ID)

    check("the frozen strategies are the four reduced strategies",
          task.STRATEGIES_FROZEN == ("AS", "AU", "CS", "CAS"),
          str(task.STRATEGIES_FROZEN))

    # --- the strategies behave as their names say ---------------------------
    safe, unsafe = task.SAFE, task.UNSAFE
    always_safe = [task.scripted_action("AS", r, [unsafe] * (r - 1)) for r in range(1, 8)]
    check("Always Safe never plays Unsafe, whatever the route does",
          set(always_safe) == {safe}, f"{len(always_safe)} rounds")

    always_unsafe = [task.scripted_action("AU", r, [safe] * (r - 1)) for r in range(1, 8)]
    check("Always Unsafe never plays Safe, whatever the route does",
          set(always_unsafe) == {unsafe}, f"{len(always_unsafe)} rounds")

    check("the two conditional strategies differ only in the opening move",
          task.scripted_action("CS", 1, []) == safe
          and task.scripted_action("CAS", 1, []) == unsafe,
          "CS opens Safe, CAS opens Unsafe")

    route = [unsafe, safe, safe, unsafe, unsafe]
    for name in ("CS", "CAS"):
        replies = [task.scripted_action(name, r, route[: r - 1]) for r in range(2, 6)]
        check(f"{name} repeats the route's previous move from round two",
              replies == route[:4], f"{replies} against {route[:4]}")

    # A conditional strategy that answered its own history would be a constant,
    # so this is the check that separates a rival from a mirror.
    mirror = [task.scripted_action("CS", r, [safe] * (r - 1)) for r in range(1, 6)]
    against_unsafe = [task.scripted_action("CS", r, [unsafe] * (r - 1)) for r in range(1, 6)]
    check("a conditional strategy answers the route, not itself",
          mirror != against_unsafe, f"{mirror} against {against_unsafe}")

    # --- the mechanism is the baseline's ------------------------------------
    disagreements = []
    for field in MECHANISM_FIELDS:
        mine = getattr(task, field, "<missing>")
        theirs = getattr(baseline, field.replace("_FROZEN", ""), "<missing>")
        if mine != theirs:
            disagreements.append(f"{field}: {mine!r} against {theirs!r}")
    check("every game-mechanism field matches the neutral baseline",
          not disagreements, "; ".join(disagreements) or "all identical")
    check("the amended decoding cap is explicit and positive",
          isinstance(task.MAX_OUTPUT_TOKENS, int) and task.MAX_OUTPUT_TOKENS > 0,
          f"MAX_OUTPUT_TOKENS={task.MAX_OUTPUT_TOKENS}")

    check("the prompt template is byte-identical to the baseline's",
          task.PROMPT_TEMPLATE == baseline.PROMPT_TEMPLATE,
          "so a race here is comparable with a race there")

    # --- the seat is counterbalanced ----------------------------------------
    seats = [rep % task.N_PLAYERS for rep in range(task.REPETITIONS)]
    check("the route's seat alternates across repetitions",
          seats.count(0) == seats.count(1) == task.REPETITIONS // 2,
          f"seat one in {seats.count(0)} repetitions, seat two in {seats.count(1)}")

    # --- the horizon is shared across the treatment -------------------------
    horizons = {}
    for rep in range(task.REPETITIONS):
        seed = task.stream_seed(task.BASE_SEED + rep, task.HORIZON_STREAM)
        horizons[rep] = task.sample_horizon(seed)[0]
    check("one horizon stream serves every strategy and risk in a repetition",
          len(horizons) == task.REPETITIONS,
          "the game seed is base + rep and names neither treatment")

    baseline_horizons = {
        rep: baseline.sample_horizon(
            baseline.stream_seed(baseline.BASE_SEED + rep, baseline.HORIZON_STREAM)
        )[0]
        for rep in range(task.REPETITIONS)
    }
    check("the horizons are the neutral baseline's own horizons",
          horizons == baseline_horizons,
          f"{sorted(horizons.values())}")

    # --- the cell selector ---------------------------------------------------
    os.environ["AI_RACE_STRATEGIES"] = "AU,CS"
    os.environ["AI_RACE_RISKS"] = "0.6"
    reloaded = load(TASK, "scripted_task_subset")
    check("a run can be restricted to one declared cell",
          reloaded.STRATEGIES == ("AU", "CS") and reloaded.RISK_LEVELS == (0.6,),
          f"{reloaded.STRATEGIES} x {reloaded.RISK_LEVELS}")
    check("the frozen grid survives the restriction",
          reloaded.STRATEGIES_FROZEN == ("AS", "AU", "CS", "CAS")
          and reloaded.RISK_LEVELS_FROZEN == (0.1, 0.6, 0.9),
          "so a partial run still records the grid it is part of")
    for name in ("AI_RACE_STRATEGIES", "AI_RACE_RISKS"):
        os.environ.pop(name, None)

    # --- cost, so the plan can be declared before it is spent ---------------
    rounds = sum(horizons.values())
    per_cell = rounds
    total = per_cell * len(task.STRATEGIES_FROZEN) * len(task.RISK_LEVELS_FROZEN)
    print(f"\n  one cell is {task.REPETITIONS} races and {per_cell} route decisions")
    print(f"  the full grid is {len(task.STRATEGIES_FROZEN)} x "
          f"{len(task.RISK_LEVELS_FROZEN)} cells, {total} route decisions")
    print("  the rival costs nothing: it is executed here, not called")

    print()
    if failures:
        raise SystemExit(f"{len(failures)} design check(s) failed: {failures}")
    print(f"{len(sys.argv) and 'design verified'}: the task may be pushed")


if __name__ == "__main__":
    main()
