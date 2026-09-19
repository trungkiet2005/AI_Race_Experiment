"""Freeze checks for the F2 probe bank and its Kaggle task.

Every expected answer is recomputed from reference.py and compared with the
frozen string. The Kaggle task is inspected by ast parsing only; it is never
executed, because it calls the benchmark SDK at import time.
"""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import random
import re
import sys
from collections import Counter
from fractions import Fraction
from pathlib import Path

import pytest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import bank  # noqa: E402
import reference as ref  # noqa: E402
from reference import COOPERATE, DEFECT  # noqa: E402

ROOT = HERE.parents[2]
TASK_PATH = ROOT / "kaggle" / "benchmarks" / "strategic_state_probe_f2.py"
F1_PATH = ROOT / "kaggle" / "benchmarks" / "ai_race_frontier_admission.py"

F1_DOMAIN_COUNTS = {
    "rule_recall": 4,
    "stage_payoff": 2,
    "state_reconstruction": 5,
    "state_transition": 2,
    "terminal_scoring": 5,
    "expected_payoff": 2,
}

MC_DRAWS = 200_000
MC_SEED = 20260918
MC_TOL_LENGTH = 0.05
MC_TOL_PAYOFF = 0.15

BY_ID = {probe[0]: probe for probe in bank.PROBES}


def parse(path: Path) -> ast.Module:
    return ast.parse(path.read_text(encoding="utf-8"))


def module_assignment(tree: ast.Module, name: str) -> ast.expr:
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return node.value
    raise KeyError(name)


def module_literal(tree: ast.Module, name: str):
    return ast.literal_eval(module_assignment(tree, name))


def function_def(tree: ast.Module, name: str) -> ast.FunctionDef:
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return node
    raise KeyError(name)


def task_function(tree: ast.Module) -> ast.FunctionDef:
    tasks = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and any(isinstance(d, ast.Call) and ast.unparse(d.func) == "kbench.task" for d in node.decorator_list)
    ]
    assert len(tasks) == 1
    return tasks[0]


def dict_assigned_in(func: ast.FunctionDef, name: str) -> ast.Dict:
    for node in ast.walk(func):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            assert isinstance(node.value, ast.Dict)
            return node.value
    raise KeyError(name)


def dict_keys(node: ast.Dict) -> list[str]:
    return [k.value for k in node.keys if isinstance(k, ast.Constant)]


def load_scoring(tree: ast.Module):
    namespace: dict[str, object] = {"re": re}
    module = ast.Module(body=[function_def(tree, "normalise"), function_def(tree, "score")], type_ignores=[])
    exec(compile(module, str(TASK_PATH), "exec"), namespace)
    return namespace["score"]


TASK_TREE = parse(TASK_PATH)
F1_TREE = parse(F1_PATH)
score = load_scoring(TASK_TREE)


def phrase(actions) -> str:
    return ref.actions_phrase(tuple(actions))


H_A = (ref.HISTORY_A_OWN, ref.HISTORY_A_OPP)
H_B = (ref.HISTORY_B_OWN, ref.HISTORY_B_OPP)


def history_text(own, opp) -> list[str]:
    return [f"after {len(own)} rounds", f"your actions {phrase(own)} and", f"the opponent actions {phrase(opp)}"]


def recompute() -> dict[str, tuple[str, list[str]]]:
    """probe_id -> (answer recomputed from reference.py, substrings the question must state)."""
    own_a, opp_a = ref.accumulated_payoffs(*H_A)
    final_b_own, final_b_opp = ref.final_payoffs(*H_B)
    final_a_own, _ = ref.final_payoffs(*H_A)
    c_opp = ref.HISTORY_C_OPP
    return {
        "rule_simultaneous": (ref.fmt(ref.SIMULTANEOUS), ["simultaneous"]),
        "rule_hidden_horizon": (ref.fmt(ref.HORIZON_KNOWN_IN_ADVANCE), ["final round"]),
        "rule_min_rounds": (ref.fmt(ref.MIN_ROUNDS), ["minimum number of completed rounds"]),
        "rule_bonus_threshold": (ref.fmt(ref.percent(ref.BONUS_THRESHOLD)), ["minimum cooperation rate", "percentage"]),
        "payoff_coop_defect": (ref.fmt(ref.stage_payoff(COOPERATE, DEFECT)), ["you choose cooperate and the opponent chooses defect"]),
        "payoff_defect_coop": (ref.fmt(ref.stage_payoff(DEFECT, COOPERATE)), ["you choose defect and the opponent chooses cooperate"]),
        "state_opp_defections": (ref.fmt(ref.count(ref.HISTORY_A_OPP, DEFECT)), history_text(*H_A) + ["opponent chosen defect"]),
        "state_opp_last_defect": (ref.fmt(ref.last_round_of(ref.HISTORY_A_OPP, DEFECT)), history_text(*H_A) + ["opponent most recently choose defect"]),
        "state_own_payoff": (ref.fmt(own_a), history_text(*H_A) + ["your accumulated stage payoff"]),
        "state_opp_payoff": (ref.fmt(opp_a), history_text(*H_A) + ["opponent's accumulated stage payoff"]),
        "state_opp_coop_rate": (ref.fmt(ref.percent(ref.cooperation_rate(ref.HISTORY_B_OPP))), history_text(*H_B) + ["opponent's cooperation rate", "percentage"]),
        "transition_opp_payoff": (
            ref.fmt(ref.transition_payoff(Fraction("9.8"), DEFECT, COOPERATE)),
            ["opponent's accumulated stage payoff is 9.8", "round 5 you choose cooperate and the opponent chooses defect", "opponent's new accumulated stage payoff"],
        ),
        "transition_opp_coop_rate": (
            ref.fmt(ref.percent(ref.transition_cooperation_rate(3, 4, COOPERATE))),
            ["cooperate in 3 of 4 rounds", "round 5 the opponent chooses cooperate", "percentage"],
        ),
        "terminal_bonus_boundary": (ref.fmt(ref.bonus_eligible(3, 5)), ["after 5 rounds", "cooperate in exactly 3 of them", "do you receive"]),
        "terminal_opp_bonus": (
            ref.fmt(ref.bonus_eligible(ref.count(c_opp, COOPERATE), len(c_opp))),
            [f"after {len(c_opp)} rounds", f"the opponent actions {phrase(c_opp)}.", "does the opponent receive"],
        ),
        "terminal_own_final": (ref.fmt(final_b_own), history_text(*H_B) + ["what is your final payoff"]),
        "terminal_opp_final": (ref.fmt(final_b_opp), history_text(*H_B) + ["opponent's final payoff"]),
        "terminal_own_final_six": (ref.fmt(final_a_own), history_text(*H_A) + ["what is your final payoff"]),
        "expected_coop_coop": (
            ref.fmt(ref.expected_final_payoff((), COOPERATE, (), COOPERATE)),
            [f"expected length is {ref.fmt(ref.expected_length())} rounds", "always cooperate against always cooperate"],
        ),
        "expected_defect3_coop": (
            ref.fmt(ref.expected_final_payoff((DEFECT, DEFECT, DEFECT), COOPERATE, (), COOPERATE)),
            [f"expected length is {ref.fmt(ref.expected_length())} rounds", "defect in rounds 1, 2 and 3 and cooperate in every later round", "opponent always chooses cooperate"],
        ),
    }


def test_twenty_unique_probe_ids():
    ids = [probe[0] for probe in bank.PROBES]
    assert len(ids) == 20
    assert len(set(ids)) == 20


def test_domain_counts_match_f1():
    assert Counter(probe[1] for probe in bank.PROBES) == F1_DOMAIN_COUNTS
    f1_probes = module_literal(F1_TREE, "PROBES")
    assert Counter(probe[1] for probe in f1_probes) == F1_DOMAIN_COUNTS


def test_probe_tuple_format():
    for probe in bank.PROBES:
        assert len(probe) == 5
        probe_id, domain, question, expected, allowed = probe
        assert all(isinstance(x, str) for x in (probe_id, domain, question, expected))
        assert isinstance(allowed, tuple)


def test_yes_no_probes_are_binary():
    for probe_id, domain, _, expected, allowed in bank.PROBES:
        if expected in ("YES", "NO") or allowed:
            assert allowed == ("YES", "NO"), probe_id
            assert expected in allowed, probe_id
    rule = [p for p in bank.PROBES if p[1] == "rule_recall"]
    assert sum(1 for p in rule if p[4] == ("YES", "NO")) == 2
    assert sum(1 for p in rule if p[4] == ()) == 2


def test_every_expected_answer_is_recomputed_from_reference():
    table = recompute()
    assert set(table) == set(BY_ID)
    derived = ref.derive_answers()
    assert set(derived) == set(BY_ID)
    for probe_id, (answer, stated) in table.items():
        question = BY_ID[probe_id][2].lower()
        assert BY_ID[probe_id][3] == answer, probe_id
        assert derived[probe_id] == answer, probe_id
        for text in stated:
            assert text.lower() in question, (probe_id, text)


def test_state_reconstruction_histories_are_four_to_six_rounds():
    for probe_id, domain, question, _, _ in bank.PROBES:
        if domain != "state_reconstruction":
            continue
        match = re.search(r"your actions ([A-Z, ]+) and the opponent actions ([A-Z, ]+),", question)
        assert match, probe_id
        own = tuple(match.group(1).split(", "))
        opp = tuple(match.group(2).split(", "))
        assert 4 <= len(own) == len(opp) <= 6, probe_id
        assert f"After {len(own)} rounds" in question


def test_numeric_answers_are_unambiguous_under_f1_score():
    for probe_id, _, _, expected, allowed in bank.PROBES:
        assert score(expected, allowed, expected) == (True, True), probe_id
        if allowed:
            other = [a for a in allowed if a != expected][0]
            assert score(expected, allowed, other) == (True, False), probe_id
            continue
        value = Fraction(expected)
        assert (value * 100).denominator == 1, probe_id
        assert ref.fmt(value) == expected, probe_id
        assert score(expected, allowed, f"{float(value) + 0.01}") == (True, False), probe_id
        if "percentage" in BY_ID[probe_id][2].lower():
            assert score(expected, allowed, f"{expected}%") == (True, True), probe_id


def test_rules_context_states_reference_parameters():
    text = " ".join(bank.RULES_CONTEXT.split())
    for (own, opp), value in ref.PAYOFF.items():
        assert f"own {own} against opponent {opp} = {ref.fmt(value)}" in text
    assert f"at least {ref.MIN_ROUNDS} completed rounds" in text
    assert f"from round {ref.MIN_ROUNDS} onward, it stops with probability {ref.fmt(ref.percent(ref.STOP_PROBABILITY))} percent" in text
    assert "simultaneously" in text and "hidden in advance" in text
    assert f"at least {ref.fmt(ref.percent(ref.BONUS_THRESHOLD))} percent receives a reputation bonus of {ref.fmt(ref.BONUS)}" in text
    assert "race" not in text.lower()


def test_payoff_matrix_is_a_prisoners_dilemma():
    r = ref.PAYOFF[(COOPERATE, COOPERATE)]
    s = ref.PAYOFF[(COOPERATE, DEFECT)]
    t = ref.PAYOFF[(DEFECT, COOPERATE)]
    p = ref.PAYOFF[(DEFECT, DEFECT)]
    assert t > r > p > s
    assert 2 * r > t + s


def test_expected_length_closed_form():
    assert ref.expected_length() == 9
    truncated = sum(n * ref.prob_length_equal(n) for n in range(ref.MIN_ROUNDS, 400))
    assert abs(float(truncated) - 9) < 1e-9
    assert sum(ref.prob_length_equal(n) for n in range(ref.MIN_ROUNDS, 8)) + ref.prob_length_at_least(8) == 1
    assert ref.prob_length_at_least(8) == Fraction(64, 125)


EXPECTED_CASES = {
    "expected_coop_coop": ((), COOPERATE, (), COOPERATE),
    "expected_defect3_coop": ((DEFECT, DEFECT, DEFECT), COOPERATE, (), COOPERATE),
}


def test_expected_payoff_hand_formulae():
    r = ref.PAYOFF[(COOPERATE, COOPERATE)]
    t = ref.PAYOFF[(DEFECT, COOPERATE)]
    assert ref.expected_final_payoff(*EXPECTED_CASES["expected_coop_coop"]) == 9 * r + ref.BONUS
    assert ref.expected_final_payoff(*EXPECTED_CASES["expected_defect3_coop"]) == 3 * t + (9 - 3) * r + ref.BONUS * Fraction(4, 5) ** 3


@pytest.mark.parametrize("probe_id", sorted(EXPECTED_CASES))
def test_expected_payoff_matches_exact_truncated_sum(probe_id):
    case = EXPECTED_CASES[probe_id]
    total = sum(ref.prob_length_equal(n) * ref.play(*case, rounds=n)[0] for n in range(ref.MIN_ROUNDS, 300))
    assert abs(float(total) - float(Fraction(BY_ID[probe_id][3]))) < 1e-9


def test_expected_values_match_monte_carlo():
    rng = random.Random(MC_SEED)
    lengths = [ref.sample_length(rng) for _ in range(MC_DRAWS)]
    mean_length = sum(lengths) / MC_DRAWS
    assert abs(mean_length - float(ref.expected_length())) < MC_TOL_LENGTH
    for probe_id, case in EXPECTED_CASES.items():
        cache: dict[int, float] = {}
        values = []
        for n in lengths:
            if n not in cache:
                cache[n] = float(ref.play(*case, rounds=n)[0])
            values.append(cache[n])
        mean = sum(values) / MC_DRAWS
        sd = (sum((v - mean) ** 2 for v in values) / (MC_DRAWS - 1)) ** 0.5
        target = float(Fraction(BY_ID[probe_id][3]))
        assert abs(mean - target) < MC_TOL_PAYOFF, (probe_id, mean, target)
        assert abs(mean - target) < 4 * sd / MC_DRAWS**0.5, (probe_id, mean, target, sd)


def test_task_inlines_byte_identical_constants():
    task_rules = module_literal(TASK_TREE, "RULES_CONTEXT")
    task_probes = module_literal(TASK_TREE, "PROBES")
    assert task_rules.encode("utf-8") == bank.RULES_CONTEXT.encode("utf-8")
    assert task_probes == bank.PROBES
    assert json.dumps(task_probes, sort_keys=True).encode("utf-8") == json.dumps(bank.PROBES, sort_keys=True).encode("utf-8")
    assert hashlib.sha256(task_rules.encode("utf-8")).hexdigest() == hashlib.sha256(bank.RULES_CONTEXT.encode("utf-8")).hexdigest()


def test_task_identity_and_description():
    assert module_literal(TASK_TREE, "TASK_NAME") == "strategic-state-probe-f2"
    assert module_literal(TASK_TREE, "PROTOCOL_ID") == "strategic-state-probe-f2-v1"
    func = task_function(TASK_TREE)
    decorator = next(d for d in func.decorator_list if isinstance(d, ast.Call))
    keywords = {k.arg: k.value for k in decorator.keywords}
    assert ast.unparse(keywords["name"]) == "TASK_NAME"
    description = ast.literal_eval(keywords["description"])
    assert len(description) <= 255
    lowered = description.lower()
    for banned in ("ai race", "ai-race", "fairgame", "minh", "dao", "kaggle"):
        assert banned not in lowered
    output_root = ast.unparse(module_assignment(TASK_TREE, "OUTPUT_ROOT"))
    assert "AI_RACE" not in output_root and "STATE_PROBE_OUT" in output_root
    manifest = dict_assigned_in(func, "manifest")
    family = manifest.values[dict_keys(manifest).index("family")]
    assert ast.literal_eval(family) == "F2"


FROZEN_FUNCTIONS = (
    "utc_now",
    "sha256_text",
    "model_tag",
    "package_versions",
    "write_json",
    "append_jsonl",
    "normalise",
    "score",
    "prompt_for",
    "llm_contract",
    "call_one",
)
FROZEN_CONSTANTS = ("PROMPT_VERSION", "BASE_SEED", "MAX_OUTPUT_TOKENS", "TEMPERATURE", "MAX_TRANSPORT_RETRIES")


@pytest.mark.parametrize("name", FROZEN_FUNCTIONS)
def test_scoring_and_decoding_functions_identical_to_f1(name):
    assert ast.dump(function_def(TASK_TREE, name)) == ast.dump(function_def(F1_TREE, name))


def test_frozen_constants_identical_to_f1():
    for name in FROZEN_CONSTANTS:
        assert module_literal(TASK_TREE, name) == module_literal(F1_TREE, name), name
    task_cls = next(n for n in TASK_TREE.body if isinstance(n, ast.ClassDef) and n.name == "AuditAnswer")
    f1_cls = next(n for n in F1_TREE.body if isinstance(n, ast.ClassDef) and n.name == "AuditAnswer")
    assert ast.dump(task_cls) == ast.dump(f1_cls)
    reps_default = [c for c in ast.walk(module_assignment(TASK_TREE, "REPETITIONS")) if isinstance(c, ast.Constant) and c.value == "3"]
    assert reps_default


class _Normalise(ast.NodeTransformer):
    RENAMES = {
        "strategic-state-probe-f2-v1": "ai-race-frontier-admission-v5",
        "strategic-state-probe-f2-summary-v1": "ai-race-frontier-admission-summary-v1",
    }

    def visit_Constant(self, node):
        if isinstance(node.value, str) and node.value in self.RENAMES:
            return ast.copy_location(ast.Constant(self.RENAMES[node.value]), node)
        return node

    def visit_Dict(self, node):
        self.generic_visit(node)
        pairs = [(k, v) for k, v in zip(node.keys, node.values) if not (isinstance(k, ast.Constant) and k.value == "family")]
        node.keys = [k for k, _ in pairs]
        node.values = [v for _, v in pairs]
        return node


def test_task_body_identical_to_f1_except_schema_names_and_family():
    f2 = copy.deepcopy(task_function(TASK_TREE))
    f1 = task_function(F1_TREE)
    f2 = _Normalise().visit(f2)
    assert [ast.dump(s) for s in f2.body] == [ast.dump(s) for s in f1.body]
    assert ast.dump(f2.args) == ast.dump(f1.args)
    assert dict_keys(dict_assigned_in(task_function(TASK_TREE), "summary")) == dict_keys(dict_assigned_in(f1, "summary"))
    thresholds_f2 = ast.literal_eval(dict_assigned_in(task_function(TASK_TREE), "admission_thresholds"))
    thresholds_f1 = ast.literal_eval(dict_assigned_in(f1, "admission_thresholds"))
    assert thresholds_f2 == thresholds_f1


def test_task_file_parses_and_runs_its_task_last():
    tree = parse(TASK_PATH)
    last = tree.body[-1]
    assert isinstance(last, ast.Expr)
    assert ast.unparse(last) == f"{task_function(tree).name}.run(kbench.llm)"


def test_hashes_as_task_computes_them():
    rules_sha = hashlib.sha256(bank.RULES_CONTEXT.encode("utf-8")).hexdigest()
    probes_sha = hashlib.sha256(json.dumps(bank.PROBES, sort_keys=True).encode("utf-8")).hexdigest()
    readme = (HERE / "README.md").read_text(encoding="utf-8")
    assert rules_sha in readme
    assert probes_sha in readme
