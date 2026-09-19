"""Checks for the frozen F3 probe bank.

Every expected answer is recomputed from reference.py; the copy inlined in the
Kaggle task is compared with bank.py through the AST only (the task file imports
kaggle_benchmarks and runs on import, so it is never executed here); the shared
contract (scorer, prompt, decoding, seeds, thresholds, summary) is required to be
AST-identical to F1.
"""

from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
import re
import sys
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
F1_TASK = REPO / "kaggle" / "benchmarks" / "ai_race_frontier_admission.py"
F3_TASK = REPO / "kaggle" / "benchmarks" / "strategic_state_probe_f3.py"

MC_DRAWS = 200_000
MC_SEED = 20260918
MC_Z = 4.0
MC_MAX_SE = 0.25


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ref = _load("f3_commons_reference", HERE / "reference.py")
bank = _load("f3_commons_bank", HERE / "bank.py")

YES_NO = ("YES", "NO")
EXPECTED_COUNTS = {
    "rule_recall": 4,
    "stage_payoff": 2,
    "state_reconstruction": 5,
    "state_transition": 2,
    "terminal_scoring": 5,
    "expected_payoff": 2,
}


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _tree(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"), filename=str(path))


def _assign(tree: ast.Module, name: str) -> ast.Assign:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node
    raise KeyError(name)


def _toplevel(tree: ast.Module, name: str) -> ast.AST:
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name == name:
            return node
    raise KeyError(name)


def _segment(path: Path, name: str) -> str:
    source = path.read_text(encoding="utf-8")
    return ast.get_source_segment(source, _assign(ast.parse(source), name))


def _task_function(tree: ast.Module) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and any(
            isinstance(d, ast.Call) and isinstance(d.func, ast.Attribute) and d.func.attr == "task" for d in node.decorator_list
        ):
            return node
    raise KeyError("task function")


def _dict_assigned(func: ast.FunctionDef, name: str) -> ast.Dict:
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            assert isinstance(node.value, ast.Dict)
            return node.value
    raise KeyError(name)


def _dict_keys(node: ast.Dict) -> list[str]:
    return [ast.literal_eval(k) for k in node.keys]


def _dict_entry(node: ast.Dict, key: str) -> ast.AST:
    for k, v in zip(node.keys, node.values):
        if ast.literal_eval(k) == key:
            return v
    raise KeyError(key)


def _scorer_namespace(tree: ast.Module) -> dict:
    nodes = [_toplevel(tree, n) for n in ("normalise", "score", "prompt_for")]
    nodes.append(_assign(tree, "RULES_CONTEXT"))
    namespace: dict = {"re": re}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(F3_TASK), "exec"), namespace)
    return namespace


F3_TREE = _tree(F3_TASK)
F1_TREE = _tree(F1_TASK)
SCORER = _scorer_namespace(F3_TREE)
PROBE_BY_ID = {p[0]: p for p in bank.PROBES}


def test_probe_format_counts_and_ids():
    assert len(bank.PROBES) == 20
    ids = [p[0] for p in bank.PROBES]
    assert len(set(ids)) == 20
    counts: dict[str, int] = {}
    for probe in bank.PROBES:
        assert isinstance(probe, tuple) and len(probe) == 5
        probe_id, domain, question, expected, allowed = probe
        assert all(isinstance(x, str) and x for x in (probe_id, domain, question, expected))
        assert isinstance(allowed, tuple)
        counts[domain] = counts.get(domain, 0) + 1
    assert counts == EXPECTED_COUNTS
    assert bank.EXPECTED_DOMAIN_COUNTS == EXPECTED_COUNTS


def test_yes_no_probes_and_numeric_probes():
    for probe_id, domain, _, expected, allowed in bank.PROBES:
        if expected in YES_NO:
            assert allowed == YES_NO, probe_id
        else:
            assert allowed == (), probe_id
            Fraction(expected)
    rule = [p for p in bank.PROBES if p[1] == "rule_recall"]
    assert sum(p[4] == YES_NO for p in rule) == 2
    assert sum(p[4] == () for p in rule) == 2


def test_bank_is_generated_by_reference():
    assert bank.RULES_CONTEXT == ref.render_rules()
    assert bank.PROBES == ref.build_probes()


RECOMPUTE = {
    "rule_simultaneous": lambda: ref.yes_no(ref.CONFIG.simultaneous),
    "rule_hidden_horizon": lambda: ref.yes_no(ref.CONFIG.horizon_revealed),
    "rule_multiplier": lambda: ref.fmt(ref.CONFIG.multiplier),
    "rule_group_target": lambda: ref.fmt(ref.CONFIG.target),
    "payoff_free_ride": lambda: ref.fmt(ref.stage_payoffs((0, 10, 10, 10))[0]),
    "payoff_sole_contributor": lambda: ref.fmt(ref.stage_payoffs((10, 0, 0, 0))[0]),
    "state_own_payoff": lambda: ref.fmt(ref.play(ref.HISTORY).accumulated[0]),
    "state_other_payoff": lambda: ref.fmt(ref.play(ref.HISTORY).accumulated[3]),
    "state_group_average": lambda: ref.fmt(ref.average_round_total(ref.play(ref.HISTORY))),
    "state_rounds_above": lambda: ref.fmt(ref.rounds_with_total_above(ref.play(ref.HISTORY), ref.ABOVE_THRESHOLD)),
    "state_target_gap": lambda: ref.fmt(ref.target_gap(ref.play(ref.HISTORY).group_total)),
    "transition_target_gap": lambda: ref.fmt(ref.target_gap(ref.play(ref.HISTORY + (ref.ROUND5,)).group_total)),
    "transition_own_payoff": lambda: ref.fmt(ref.play(ref.HISTORY + (ref.ROUND5,)).accumulated[0]),
    "terminal_target_missed": lambda: ref.yes_no(ref.target_reached(sum(ref.MISSED_ROUND_TOTALS))),
    "terminal_free_rider_bonus": lambda: ref.yes_no(
        ref.final_payoffs(ref.play([(10, 10, 10, 0)] * 5))[3] - ref.play([(10, 10, 10, 0)] * 5).accumulated[3] == ref.CONFIG.bonus
    ),
    "terminal_payoff_bonus": lambda: ref.fmt(ref.final_payoff(ref.TERMINAL_STAGE, ref.TERMINAL_TOTAL_HIGH)),
    "terminal_payoff_no_bonus": lambda: ref.fmt(ref.final_payoff(ref.TERMINAL_STAGE, ref.TERMINAL_TOTAL_LOW)),
    "terminal_payoff_from_rounds": lambda: ref.fmt(ref.final_payoff(ref.REACHED_OWN_STAGE, sum(ref.REACHED_ROUND_TOTALS))),
    "expected_all_ten": lambda: ref.fmt(ref.expected_final_payoff(ref.PROFILE_ALL_TEN)),
    "expected_all_five": lambda: ref.fmt(ref.expected_final_payoff(ref.PROFILE_ALL_FIVE)),
}


def test_recompute_covers_every_probe():
    assert set(RECOMPUTE) == set(PROBE_BY_ID)


@pytest.mark.parametrize("probe_id", sorted(RECOMPUTE))
def test_expected_answer_recomputed(probe_id):
    assert PROBE_BY_ID[probe_id][3] == RECOMPUTE[probe_id]()


@pytest.mark.parametrize("probe", bank.PROBES, ids=[p[0] for p in bank.PROBES])
def test_answer_unambiguous_under_shared_scorer(probe):
    probe_id, _, _, expected, allowed = probe
    score = SCORER["score"]
    assert score(expected, allowed, expected) == (True, True)
    if allowed:
        other = next(a for a in allowed if a != expected)
        assert score(expected, allowed, other) == (True, False)
        return
    assert (Fraction(expected) * 100).denominator == 1
    assert ref.fmt(Fraction(expected)) == expected
    for delta in ("0.01", "-0.01", "1", "-1"):
        wrong = ref.fmt(Fraction(expected) + Fraction(delta))
        assert score(expected, allowed, wrong) == (True, False), (probe_id, wrong)


def test_state_reconstruction_aggregates_over_players_and_rounds():
    state = ref.play(ref.HISTORY)
    assert 3 <= state.rounds <= 5
    assert state.round_totals == (25, 20, 10, 15)
    assert state.group_total == 70
    assert len({c for round_ in ref.HISTORY for c in round_}) > 1
    for probe_id in ("state_own_payoff", "state_other_payoff", "state_group_average", "state_rounds_above", "state_target_gap"):
        assert ref.history_text(ref.HISTORY) in PROBE_BY_ID[probe_id][2]


def test_stated_scenarios_are_reachable_game_states():
    for totals in (ref.MISSED_ROUND_TOTALS, ref.REACHED_ROUND_TOTALS):
        assert len(totals) >= ref.CONFIG.min_rounds
        assert all(t % 5 == 0 and 0 <= t <= 40 for t in totals)
    assert ref.stage_is_consistent(ref.TERMINAL_STAGE, ref.TERMINAL_TOTAL_HIGH)
    assert ref.stage_is_consistent(ref.TERMINAL_STAGE, ref.TERMINAL_TOTAL_LOW)
    assert ref.stage_is_consistent(ref.REACHED_OWN_STAGE, sum(ref.REACHED_ROUND_TOTALS), len(ref.REACHED_ROUND_TOTALS))
    assert ref.target_reached(ref.TERMINAL_TOTAL_HIGH) and not ref.target_reached(ref.TERMINAL_TOTAL_LOW)
    assert not ref.target_reached(sum(ref.MISSED_ROUND_TOTALS))
    assert ref.target_reached(sum(ref.REACHED_ROUND_TOTALS))


def test_horizon_closed_forms():
    assert ref.expected_length() == 9
    assert sum(ref.prob_length_equals(n) for n in range(1, 400)) == pytest.approx(1.0, abs=1e-12)
    assert sum(n * float(ref.prob_length_equals(n)) for n in range(1, 400)) == pytest.approx(9.0, abs=1e-9)
    assert ref.prob_length_at_least(5) == 1
    assert ref.prob_length_at_least(8) == Fraction(64, 125)
    assert ref.rounds_to_target(40) == 4
    assert ref.rounds_to_target(20) == 8


@pytest.mark.parametrize("profile", [ref.PROFILE_ALL_TEN, ref.PROFILE_ALL_FIVE], ids=["all_ten", "all_five"])
def test_expected_payoff_closed_form_matches_exact_sum(profile):
    assert ref.expected_final_payoff_by_sum(profile) == pytest.approx(float(ref.expected_final_payoff(profile)), abs=1e-9)


@pytest.mark.parametrize("profile", [ref.PROFILE_ALL_TEN, ref.PROFILE_ALL_FIVE], ids=["all_ten", "all_five"])
def test_expected_payoff_monte_carlo(profile):
    exact = float(ref.expected_final_payoff(profile))
    mean, se = ref.simulate_expected_final_payoff(profile, MC_DRAWS, MC_SEED)
    assert se < MC_MAX_SE
    assert abs(mean - exact) <= MC_Z * se, (mean, exact, se)


def test_task_file_parses():
    compile(F3_TREE, str(F3_TASK), "exec")


def test_inlined_constants_are_byte_identical_to_bank():
    bank_path = HERE / "bank.py"
    for name in ("RULES_CONTEXT", "PROBES"):
        assert _segment(F3_TASK, name) == _segment(bank_path, name)
    rules = ast.literal_eval(_assign(F3_TREE, "RULES_CONTEXT").value)
    probes = ast.literal_eval(_assign(F3_TREE, "PROBES").value)
    assert rules == bank.RULES_CONTEXT
    assert probes == bank.PROBES
    assert _sha(rules) == bank.RULES_CONTEXT_SHA256
    assert _sha(json.dumps(probes, sort_keys=True)) == bank.PROBE_BANK_SHA256


def test_task_identity_and_description():
    assert ast.literal_eval(_assign(F3_TREE, "TASK_NAME").value) == bank.TASK_NAME == "strategic-state-probe-f3"
    assert ast.literal_eval(_assign(F3_TREE, "PROTOCOL_ID").value) == bank.PROTOCOL_ID == "strategic-state-probe-f3-v1"
    func = _task_function(F3_TREE)
    decorator = next(d for d in func.decorator_list if isinstance(d, ast.Call))
    kwargs = {k.arg: k.value for k in decorator.keywords}
    assert isinstance(kwargs["name"], ast.Name) and kwargs["name"].id == "TASK_NAME"
    description = ast.literal_eval(kwargs["description"])
    assert len(description) <= 255
    assert not re.search(r"ai[ _-]?race|fairgame|\bdao\b|\bminh\b|chis|technoob", description, re.IGNORECASE)
    source = F3_TASK.read_text(encoding="utf-8")
    env_names = re.findall(r'os\.environ\.get\("([A-Z0-9_]+)"', source)
    assert "LLM_DEFAULT" in env_names
    assert not any("AI_RACE" in n for n in env_names)


def test_manifest_and_summary_structure():
    f1, f3 = _task_function(F1_TREE), _task_function(F3_TREE)
    m1, m3 = _dict_assigned(f1, "manifest"), _dict_assigned(f3, "manifest")
    assert ast.literal_eval(_dict_entry(m3, "family")) == "F3"
    assert ast.literal_eval(_dict_entry(m3, "schema_version")) == "strategic-state-probe-f3-v1"
    assert [k for k in _dict_keys(m3) if k != "family"] == _dict_keys(m1)
    for key in _dict_keys(m1):
        if key != "schema_version":
            assert ast.dump(_dict_entry(m3, key)) == ast.dump(_dict_entry(m1, key)), key
    s1, s3 = _dict_assigned(f1, "summary"), _dict_assigned(f3, "summary")
    assert _dict_keys(s3) == _dict_keys(s1)
    assert ast.literal_eval(_dict_entry(s3, "schema_version")) == "strategic-state-probe-f3-summary-v1"
    for key in _dict_keys(s1):
        if key != "schema_version":
            assert ast.dump(_dict_entry(s3, key)) == ast.dump(_dict_entry(s1, key)), key
    t1, t3 = _dict_assigned(f1, "admission_thresholds"), _dict_assigned(f3, "admission_thresholds")
    assert ast.dump(t3) == ast.dump(t1)


def test_task_body_identical_to_f1_apart_from_named_changes():
    f1, f3 = _task_function(F1_TREE), _task_function(F3_TREE)

    def strip(func: ast.FunctionDef) -> str:
        clone = ast.parse(ast.unparse(func)).body[0]
        clone.name = "TASK"
        clone.decorator_list = []
        for node in ast.walk(clone):
            if isinstance(node, ast.Dict) and node.keys and all(isinstance(k, ast.Constant) for k in node.keys):
                pairs = [
                    (k, v) for k, v in zip(node.keys, node.values)
                    if k.value not in ("schema_version", "family")
                ]
                node.keys = [k for k, _ in pairs]
                node.values = [v for _, v in pairs]
        return ast.dump(clone)

    assert strip(f3) == strip(f1)


SHARED_DEFINITIONS = (
    "AuditAnswer", "utc_now", "sha256_text", "model_tag", "package_versions", "write_json",
    "append_jsonl", "normalise", "score", "prompt_for", "llm_contract", "call_one",
)
SHARED_CONSTANTS = ("PROMPT_VERSION", "BASE_SEED", "MAX_OUTPUT_TOKENS", "TEMPERATURE", "MAX_TRANSPORT_RETRIES")


@pytest.mark.parametrize("name", SHARED_DEFINITIONS)
def test_shared_definition_identical_to_f1(name):
    assert ast.dump(_toplevel(F3_TREE, name)) == ast.dump(_toplevel(F1_TREE, name))


@pytest.mark.parametrize("name", SHARED_CONSTANTS)
def test_shared_constant_identical_to_f1(name):
    assert ast.dump(_assign(F3_TREE, name).value) == ast.dump(_assign(F1_TREE, name).value)


def test_repetition_default_identical_to_f1():
    def default(tree: ast.Module) -> object:
        call = _assign(tree, "REPETITIONS").value
        return ast.literal_eval(call.args[0].args[1])

    assert default(F3_TREE) == default(F1_TREE) == "3"


def test_prompt_header_is_neutral():
    for probe_id, _, question, _, allowed in bank.PROBES:
        prompt = SCORER["prompt_for"](probe_id, question, allowed)
        assert prompt.startswith(bank.RULES_CONTEXT)
        assert not re.search(r"\bAI\b|\brace\b", prompt, re.IGNORECASE), probe_id
