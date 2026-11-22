"""Check the scripted context-mapping task offline, before it costs a single request.

The task is the scripted-rival task with one thing changed: the route's prompt is
rendered by the context by mapping template of the self-play context task.  The
claim worth protecting is that nothing else moved, so a difference between a cell
here and the scripted campaign can be put down to presentation.  This verifier
therefore runs the real ``play_race`` and ``decide`` code against a mock model and
asserts the properties that make that claim true.

What is checked.

Each strategy behaves as named, and the two conditional strategies answer the
ROUTE's previous move, read out of the seat-indexed history, in either seat.

The route's seat alternates with the repetition and both seats are used five
times in ten, with the seat and the cell recorded on every row.

The horizon, stop draws and setback draws are the scripted task's own for the
same seeds, so a race here is paired with the matching race there.

The decoding contract, retry policy and mechanism constants are byte-for-byte or
value-for-value the scripted task's.

The prompt template, contexts and mappings are copies of the context-mapping
task's, and the rendered prompt for seat one equals that task's own rendering.
For seat two the only difference is that the history is told from the deciding
player's side.

The parser accepts exactly one of the cell's two codes and rejects prose, a code
inside a sentence and the legacy ``ACTION:`` line.

Finally it states, rather than assumes, whether any context and mapping pair
reproduces the ai-race-fairgame-v3 prompt of the scripted task.
"""

from __future__ import annotations

import contextlib
import difflib
import importlib.util
import os
import re
import sys
import tempfile
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BENCH = ROOT / "kaggle" / "benchmarks"
TASK = BENCH / "ai_race_scripted_context_mapping.py"
SCRIPTED = BENCH / "ai_race_scripted_opponent.py"
SELFPLAY = BENCH / "ai_race_frontier_context_mapping.py"

# The cell whose semantics are closest to the scripted prompt: the technology
# race narrative, with the code listed first standing for Safe.  It is the
# nearest cell, not an identical one; see check_canonical_prompt.
CANONICAL_CELL = ("technology_race", "P_SAFE_Q_UNSAFE")

SAME_AS_SCRIPTED = (
    "SAFE", "UNSAFE", "N_PLAYERS", "PLAYER_NAMES", "MIN_ROUNDS", "STOP_PROBABILITY",
    "PRIZE", "PROGRESS", "STAGE_PAYOFF", "BASE_SEED", "HORIZON_STREAM",
    "SETBACK_STREAM", "RISK_LEVELS_FROZEN", "STRATEGIES_FROZEN", "STRATEGY_DESCRIPTION",
    "TEMPERATURE", "REASONING", "REASONING_OMITTED_ROUTES", "REASONING_LOW_ROUTES",
    "MAX_OUTPUT_TOKENS", "MAX_PARSE_RETRIES", "MAX_TRANSPORT_RETRIES",
    "REQUEST_TIMEOUT_SECONDS", "SEED_MODULUS", "SAMPLING_SEED_STRIDE",
    "RETRY_SEED_STRIDE", "REPETITIONS", "PERSONA_CONDITION", "EXPECTED_ROUNDS",
    "SDK_SEED_STRIPPED_ROUTE_PREFIXES",
)


class _Chats:
    @staticmethod
    @contextlib.contextmanager
    def new(_name, orphan=True):
        yield


class _Assertions:
    @staticmethod
    def assert_equal(expected, actual, expectation=""):
        assert expected == actual, expectation


class _StubSdk(types.ModuleType):
    llm = None
    chats = _Chats
    assertions = _Assertions

    def __init__(self) -> None:
        super().__init__("kaggle_benchmarks")

    @staticmethod
    def task(*_args, **_kwargs):
        def decorator(func):
            func.run = lambda *a, **k: None
            return func

        return decorator


def load(path: Path, name: str, env: dict | None = None):
    """Import a task file without running its unguarded ``.run`` call."""
    saved_env = {key: os.environ.get(key) for key in (env or {})}
    os.environ.setdefault("LLM_DEFAULT", "offline/stub")
    os.environ.update(env or {})
    source = path.read_text(encoding="utf-8")
    marker = ".run(kbench.llm)"
    if marker in source:
        source = source[: source.rindex(marker)].rsplit("\n", 1)[0]
    module = importlib.util.module_from_spec(importlib.util.spec_from_loader(name, loader=None))
    module.__file__ = str(path)
    sys.modules[name] = module
    real = sys.modules.get("kaggle_benchmarks")
    sys.modules["kaggle_benchmarks"] = _StubSdk()
    try:
        exec(compile(source, str(path), "exec"), module.__dict__)
    finally:
        if real is None:
            sys.modules.pop("kaggle_benchmarks", None)
        else:
            sys.modules["kaggle_benchmarks"] = real
        for key, value in saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
    return module


class OpenAI:
    """A mock endpoint; its class name selects the OpenAI branch of the contract."""

    model = "mock/stub"

    def __init__(self, policy, prose_first=0, prose_always=False):
        self.policy = policy
        self.prose_first = prose_first
        self.prose_always = prose_always
        self.calls = 0
        self.prompts: list[str] = []
        self._attempt = 0

    def prompt(self, text, schema=None, **_kwargs):
        self.calls += 1
        self.prompts.append(text)
        if self.prose_always or self._attempt < self.prose_first:
            self._attempt += 1
            return "I will choose P because it is safe."
        self._attempt = 0
        safe = re.search(r"SAFE means (\w)", text).group(1)
        unsafe = re.search(r"UNSAFE means (\w)", text).group(1)
        round_number = int(re.search(r"before round (\d+)", text).group(1))
        code = unsafe if self.policy(round_number) else safe
        return schema(action=code)


def alternating(round_number: int) -> bool:
    return round_number % 2 == 0


def _src_block(path: Path, start: str, end: str) -> str:
    text = path.read_text(encoding="utf-8")
    i = text.index(start)
    return text[i: text.index(end, i)]


def run_checks(verbose: bool = True) -> list[str]:
    failures: list[str] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        if verbose:
            print(f"{'PASS' if ok else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
        if not ok:
            failures.append(name)

    out = tempfile.mkdtemp(prefix="scripted_context_")
    os.environ["AI_RACE_BENCHMARK_OUT"] = out
    try:
        task = load(TASK, "ctx_task")
        scripted = load(SCRIPTED, "ctx_scripted")
        selfplay = load(SELFPLAY, "ctx_selfplay")
    finally:
        os.environ.pop("AI_RACE_BENCHMARK_OUT", None)

    check("the task declares its own protocol, task name and prompt version",
          task.PROTOCOL_ID == "ai-race-scripted-context-mapping-v1"
          and task.TASK_NAME == "ai-race-scripted-context-mapping"
          and task.PROMPT_VERSION == "ai-race-context-mapping-v1",
          f"{task.PROTOCOL_ID} / {task.TASK_NAME} / {task.PROMPT_VERSION}")
    check("the selector defaults are the canonical seed and ten repetitions",
          task.BASE_SEED == 260726 and task.REPETITIONS == 10)

    # --- strategies ---------------------------------------------------------
    safe, unsafe = task.SAFE, task.UNSAFE
    check("Always Safe never plays Unsafe",
          {task.scripted_action("AS", r, [unsafe] * (r - 1)) for r in range(1, 8)} == {safe})
    check("Always Unsafe never plays Safe",
          {task.scripted_action("AU", r, [safe] * (r - 1)) for r in range(1, 8)} == {unsafe})
    check("CS opens Safe and CAS opens Unsafe",
          task.scripted_action("CS", 1, []) == safe and task.scripted_action("CAS", 1, []) == unsafe)
    route = [unsafe, safe, safe, unsafe, unsafe]
    for name in ("CS", "CAS"):
        replies = [task.scripted_action(name, r, route[: r - 1]) for r in range(2, 6)]
        check(f"{name} repeats the route's previous move", replies == route[:4])
    check("the scripted_action function is the scripted task's own",
          _src_block(TASK, "def scripted_action", "# %%\ndef sample_horizon")
          == _src_block(SCRIPTED, "def scripted_action", "# %%\ndef sample_horizon"))

    # --- constants and decoding contract -----------------------------------
    drift = [f for f in SAME_AS_SCRIPTED
             if getattr(task, f, "<missing>") != getattr(scripted, f, "<missing>")]
    check("mechanism, decoding and retry constants equal the scripted task's",
          not drift, ", ".join(drift) or f"{len(SAME_AS_SCRIPTED)} fields identical")
    check("resolve_llm_contract is verbatim the scripted task's",
          _src_block(TASK, "def resolve_llm_contract", "# %%\ndef scripted_action")
          == _src_block(SCRIPTED, "def resolve_llm_contract", "# %%\ndef scripted_action"))
    check("the transport-retry wrapper is verbatim the scripted task's",
          _src_block(TASK, "def prompt_with_transport_retries", "def decide")
          == _src_block(SCRIPTED, "def prompt_with_transport_retries", "def decide"))

    # --- presentation is a copy of the self-play task's ---------------------
    check("the prompt template is the context-mapping task's, byte for byte",
          task.PROMPT_TEMPLATE == selfplay.PROMPT_TEMPLATE)
    check("CONTEXTS and MAPPINGS are the context-mapping task's",
          task.CONTEXTS == selfplay.CONTEXTS and task.MAPPINGS == selfplay.MAPPINGS,
          f"contexts {list(task.CONTEXTS)}, mappings {list(task.MAPPINGS)}")

    equal_seat_one = equal_seat_two = True
    samples = 0
    histories = [[], [(safe, unsafe)], [(unsafe, safe), (unsafe, unsafe)],
                 [(safe, safe), (safe, unsafe), (unsafe, safe)]]
    for context in task.CONTEXTS:
        for mapping in task.MAPPINGS:
            cell = load(TASK, f"ctx_cell_{context}_{mapping}",
                        {"AI_RACE_CONTEXT": context, "AI_RACE_MAPPING": mapping})
            for risk in task.RISK_LEVELS_FROZEN:
                for history in histories:
                    rnd = len(history) + 1
                    samples += 1
                    original_p0 = selfplay.build_prompt(context, mapping, risk, rnd, history, 0)
                    equal_seat_one &= cell.build_prompt(0, rnd, history, risk) == original_p0
                    swapped = [(b, a) for a, b in history]
                    equal_seat_two &= (
                        cell.build_prompt(1, rnd, history, risk)
                        == selfplay.build_prompt(context, mapping, risk, rnd, swapped, 0)
                    )
    check("seat-one prompts equal the context-mapping task's own rendering",
          equal_seat_one, f"{samples} cell, risk and history samples")
    check("seat-two prompts are the same rendering with the history told from seat two",
          equal_seat_two)

    # --- parser -------------------------------------------------------------
    cell_a = load(TASK, "ctx_cell_a", {"AI_RACE_MAPPING": "P_SAFE_Q_UNSAFE"})
    cell_b = load(TASK, "ctx_cell_b", {"AI_RACE_MAPPING": "Q_SAFE_P_UNSAFE"})
    accepted = [(" p ", "SAFE"), ("Q", "UNSAFE"), ("q\n", "UNSAFE")]
    check("a bare code decodes through the cell's mapping",
          all(cell_a.parse_action(v) == w for v, w in accepted)
          and cell_a.parse_action(cell_a.ActionDecision(action="P")) == "SAFE"
          and cell_b.parse_action("P") == "UNSAFE" and cell_b.parse_action("Q") == "SAFE")
    rejected = ["I will choose P", "ACTION: P", "SAFE", "UNSAFE", "P Q", "PQ", "P\nQ", "R", "",
                "The answer is Q.", "```P```"]
    check("prose, a code in a sentence, the legacy line and other words are rejected",
          all(cell_a.parse_action(v) is None for v in rejected)
          and cell_a.parse_action(None) is None
          and cell_a.parse_action(cell_a.ActionDecision(action="the answer is P")) is None,
          f"{len(rejected)} rejected strings")

    # --- the canonical prompt ----------------------------------------------
    reproduced = []
    best = (0.0, None)
    reference = scripted.build_prompt(0, 1, [], 0.6)
    for context in task.CONTEXTS:
        for mapping in task.MAPPINGS:
            cell = load(TASK, f"ctx_canon_{context}_{mapping}",
                        {"AI_RACE_CONTEXT": context, "AI_RACE_MAPPING": mapping})
            rendered = cell.build_prompt(0, 1, [], 0.6)
            ratio = difflib.SequenceMatcher(None, reference, rendered).ratio()
            if rendered == reference:
                reproduced.append((context, mapping))
            if ratio > best[0]:
                best = (ratio, (context, mapping))
    check("no context and mapping pair reproduces the ai-race-fairgame-v3 prompt",
          not reproduced,
          "none of the 4 cells is byte-identical; nearest by text similarity is "
          f"{best[1]} at {best[0]:.2f}" if not reproduced else f"reproduced by {reproduced}")
    check("the template carries no scripted-rival disclosure",
          not re.search(r"script|always|conditional|strateg", task.PROMPT_TEMPLATE, re.I))

    # --- full mock races, every cell ---------------------------------------
    horizons = {}
    for rep in range(task.REPETITIONS):
        horizons[rep] = scripted.sample_horizon(
            scripted.stream_seed(scripted.BASE_SEED + rep, scripted.HORIZON_STREAM))[0]
    per_cell_decisions = sum(horizons.values())

    seat_ok = deviation_free = horizon_ok = rows_ok = hidden_ok = codes_ok = True
    setback_ok = True
    for context in task.CONTEXTS:
        for mapping in task.MAPPINGS:
            with tempfile.TemporaryDirectory() as tmp:
                cell = load(TASK, f"ctx_run_{context}_{mapping}",
                            {"AI_RACE_CONTEXT": context, "AI_RACE_MAPPING": mapping,
                             "AI_RACE_BENCHMARK_OUT": tmp})
                code_of = cell.MAPPINGS[mapping]
                for strategy in cell.STRATEGIES_FROZEN:
                    llm = OpenAI(alternating)
                    contract = cell.resolve_llm_contract(llm)
                    turns, players, seats = [], [], []
                    for rep in range(cell.REPETITIONS):
                        race = cell.play_race(llm, 0.6, rep, strategy, turns, players, contract)
                        seats.append(race["route_seat"])
                        horizon_ok &= race["n_rounds"] == horizons[rep]
                        horizon_ok &= race["horizon_seed"] == scripted.stream_seed(
                            scripted.BASE_SEED + rep, scripted.HORIZON_STREAM)
                        draws = scripted.random.Random(scripted.stream_seed(
                            scripted.BASE_SEED + rep, scripted.SETBACK_STREAM))
                        setback_ok &= [race["player_1_setback_draw"], race["player_2_setback_draw"]] == [
                            draws.random(), draws.random()]
                        rows_ok &= race["context"] == context and race["mapping"] == mapping
                    seat_ok &= seats.count(0) == seats.count(1) == cell.REPETITIONS // 2
                    seat_ok &= all(seat == rep % 2 for rep, seat in enumerate(seats))
                    rows_ok &= all(t["context"] == context and t["mapping"] == mapping for t in turns)
                    rows_ok &= all(p["context"] == context and p["mapping"] == mapping for p in players)
                    by_race: dict = {}
                    for t in turns:
                        by_race.setdefault(t["game_id"], []).append(t)
                    for rows in by_race.values():
                        mine = sorted((r for r in rows if r["is_route_decision"]), key=lambda r: r["round"])
                        rival = sorted((r for r in rows if not r["is_route_decision"]), key=lambda r: r["round"])
                        moves = [r["action"] for r in mine]
                        for r in rival:
                            want = {"AS": "safe", "AU": "unsafe"}.get(strategy)
                            if want is None:
                                want = ("safe" if strategy == "CS" else "unsafe") if r["round"] == 1 \
                                    else moves[r["round"] - 2]
                            deviation_free &= r["action"] == want
                        codes_ok &= all(code_of[r["action"].upper()] == r["action_code"] for r in rows)
                        hidden_ok &= all(
                            not re.search(r"script|always|conditional|strateg", r["prompt"], re.I)
                            for r in rows)
                        for a, b in zip(mine, rival):
                            codes_ok &= a["round"] == b["round"]
    check("every strategy, in both cells of every context and mapping, plays as named",
          deviation_free, "rival replayed from the route's own moves, 16 strategy cells")
    check("the route's seat is repetition % 2, five and five, in every cell", seat_ok)
    check("horizons and stop draws equal the scripted task's for the same seeds", horizon_ok)
    check("setback draws equal the scripted task's for the same seeds", setback_ok)
    check("context and mapping are recorded on every turn, player and race row", rows_ok)
    check("action_code on every row decodes to its action through the cell mapping", codes_ok)
    check("no prompt names the rival's strategy or says it is scripted", hidden_ok)

    # --- retry policy and fail-closed behaviour ----------------------------
    cell = load(TASK, "ctx_retry", {"AI_RACE_BENCHMARK_OUT": tempfile.mkdtemp()})
    llm = OpenAI(alternating, prose_first=2)
    contract = cell.resolve_llm_contract(llm)
    prompt = cell.build_prompt(0, 2, [(safe, unsafe)], 0.6)
    decision = cell.decide(llm, prompt, "g", 0, 2, 0, contract)
    check("prose is retried with a fresh seed and then accepted without a fallback",
          decision["attempts"] == 3 and not decision["parse_failed"]
          and [a["parse_failed"] for a in decision["attempt_history"]] == [True, True, False]
          and len({a["sampling_seed"] for a in decision["attempt_history"]}) == 3,
          f"{decision['attempts']} attempts")
    llm = OpenAI(alternating, prose_always=True)
    decision = cell.decide(llm, prompt, "g", 0, 2, 0, contract)
    check("a decision that never parses is marked failed after the retry cap",
          decision["parse_failed"] and decision["attempts"] == cell.MAX_PARSE_RETRIES + 1,
          f"{decision['attempts']} attempts")

    # --- the whole task function, one cell ---------------------------------
    with tempfile.TemporaryDirectory() as tmp:
        cell = load(TASK, "ctx_full", {
            "AI_RACE_BENCHMARK_OUT": tmp, "AI_RACE_CONTEXT": "abstract_game",
            "AI_RACE_MAPPING": "Q_SAFE_P_UNSAFE", "AI_RACE_STRATEGIES": "CS", "AI_RACE_RISKS": "0.6"})
        with contextlib.redirect_stdout(open(os.devnull, "w")):
            summary = cell.ai_race_scripted_context_mapping(OpenAI(alternating))
        import json
        manifest = json.loads((Path(tmp) / "run_manifest.json").read_text(encoding="utf-8"))
        design = manifest["opponent_design"]
        files = all((Path(tmp) / f).exists() for f in
                    ("turns.jsonl", "races.csv", "players.csv", "summary.json", "run_manifest.json"))
        check("the manifest records the cell and the run writes every scripted-task output",
              manifest["status"] == "completed" and files
              and design["context"] == "abstract_game" and design["mapping"] == "Q_SAFE_P_UNSAFE"
              and design["strategies_collected"] == ["CS"] and design["risk_levels_collected"] == [0.6]
              and design["is_partial_collection"] and design["rival_disclosed_in_prompt"] is False
              and manifest["prompt_version"] == "ai-race-context-mapping-v1"
              and manifest["protocol_id"] == "ai-race-scripted-context-mapping-v1"
              and summary["parse_failures"] == 0 and summary["n_races"] == 10,
              f"{summary['n_decisions']} decisions")
        route_decisions = sum(1 for line in (Path(tmp) / "turns.jsonl").read_text().splitlines()
                              if json.loads(line)["is_route_decision"])
        check("one cell holds the expected number of route decisions",
              route_decisions == per_cell_decisions, f"{route_decisions}")

    # --- the bake helper ----------------------------------------------------
    bake = ROOT / "scripts" / "bake_scripted_context_cell.py"
    bake_src = bake.read_text(encoding="utf-8")
    task_src = TASK.read_text(encoding="utf-8")
    lines = re.findall(r'^(?:STRATEGIES|RISK_LEVELS|CONTEXT|MAPPING) = .*$', task_src, re.M)
    expected = [m for m in re.findall(r"^EXPECTED_\w+ = '(.+)'$", bake_src, re.M)
                if not m.startswith("BASE_SEED")]
    check("each selector the bake helper replaces is one line occurring once in the task",
          len(expected) == 4 and all(task_src.count(e) == 1 for e in expected),
          f"{len(lines)} module-level selector assignments")

    if verbose:
        print(f"\n  one cell is {task.REPETITIONS} races and {per_cell_decisions} route decisions "
              f"(horizons {[horizons[r] for r in range(task.REPETITIONS)]})")
        print(f"  the scripted campaign's own cells at this seed hold {per_cell_decisions} as well")
        print(f"  canonical cell by semantics: {CANONICAL_CELL}; byte-identical to the "
              f"fairgame-v3 prompt: {'yes' if reproduced else 'no'}")
        print(f"  grid per route: 4 strategies x 3 risks x 2 contexts x 2 mappings = 48 cells, "
              f"{48 * per_cell_decisions} route decisions")
    return failures


def main() -> None:
    failures = run_checks(verbose=True)
    print()
    if failures:
        raise SystemExit(f"{len(failures)} design check(s) failed: {failures}")
    print("design verified: the task may be pushed")


if __name__ == "__main__":
    main()
