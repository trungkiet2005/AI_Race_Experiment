"""Keep the frozen prompt contract and the analyser's copy of it in agreement.

The analyser refuses to pool runs whose prompt text differs from the frozen
template, but it can only do that if its hash table still matches the templates
actually shipped in this repository. Documenting the hashes is not enough: a
whitespace edit would silently make every future run noncanonical, and the
failure would only surface after a GPU run had already been paid for. These
tests turn that into an immediate red test instead.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest

from ai_race.paths import CONFIGS_DIR, PROMPTS_DIR

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYSER_PATH = REPOSITORY_ROOT / "results" / "scripts" / "analyze_ai_race.py"


def _load_analyser():
    """Import the analyser by path; it lives outside the installed package."""

    pytest.importorskip("pandas", reason="the analyser needs the analysis extra")
    spec = importlib.util.spec_from_file_location("_analyze_ai_race", ANALYSER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_shipped_templates_match_the_analyser_hash_table():
    analyser = _load_analyser()
    table = analyser.CANONICAL_PROMPT_SHA256_BY_TEMPLATE
    for template_name, expected_sha256 in table.items():
        path = PROMPTS_DIR / f"{template_name}.txt"
        assert path.is_file(), f"{template_name} is registered but not shipped"
        assert _sha256(path) == expected_sha256, (
            f"{path.name} changed. Bump promptVersion and add a new entry to "
            "CANONICAL_PROMPT_SHA256_BY_TEMPLATE; never relabel modified text."
        )


def test_every_shipped_template_is_registered():
    analyser = _load_analyser()
    shipped = {path.stem for path in PROMPTS_DIR.glob("ai_race_*.txt")}
    registered = set(analyser.CANONICAL_PROMPT_SHA256_BY_TEMPLATE)
    assert shipped == registered, (
        "an unregistered template would be rejected as noncanonical at analysis "
        f"time; shipped={sorted(shipped)} registered={sorted(registered)}"
    )


def test_game_configs_declare_the_canonical_prompt_version():
    analyser = _load_analyser()
    canonical = analyser.CANONICAL_PROMPT_VERSION
    game_configs = sorted((CONFIGS_DIR / "game").glob("*.json"))
    assert game_configs, "no game configurations were discovered"
    for path in game_configs:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert payload.get("promptVersion") == canonical, (
            f"{path.name} declares {payload.get('promptVersion')!r} but the "
            f"analyser treats {canonical!r} as canonical"
        )


BENCHMARK_PATH = REPOSITORY_ROOT / "kaggle" / "benchmarks" / "ai_race_baseline.py"


def _benchmark_string_constants() -> dict[str, str]:
    """Read the task's module-level string constants without executing it.

    The task is a self-contained Kaggle Benchmark script that imports
    ``kaggle_benchmarks``; parsing it keeps this test runnable anywhere.
    """

    import ast

    tree = ast.parse(BENCHMARK_PATH.read_text(encoding="utf-8"))
    constants: dict[str, str] = {}
    for node in tree.body:
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and isinstance(node.value, ast.Constant):
            if isinstance(node.value.value, str):
                constants[target.id] = node.value.value
    return constants


def test_benchmark_task_copies_the_canonical_template_byte_for_byte():
    """The task reimplements the mechanism, so its copy has to be verified.

    It records ``sha256(PROMPT_TEMPLATE)`` in its manifest. If the copy drifts
    from the shipped template by even one character, that hash stops matching the
    canonical set and every race the task produces is dropped from primary
    analysis.
    """

    constants = _benchmark_string_constants()
    shipped = (PROMPTS_DIR / "ai_race_en.txt").read_text(encoding="utf-8")
    assert constants["PROMPT_TEMPLATE"] == shipped
    assert (
        hashlib.sha256(constants["PROMPT_TEMPLATE"].encode("utf-8")).hexdigest()
        == _sha256(PROMPTS_DIR / "ai_race_en.txt")
    )


def test_benchmark_task_declares_the_canonical_version_and_no_persona():
    analyser = _load_analyser()
    constants = _benchmark_string_constants()
    assert constants["PROMPT_VERSION"] == analyser.CANONICAL_PROMPT_VERSION
    assert analyser._is_canonical_prompt(
        constants["PROMPT_VERSION"],
        hashlib.sha256(constants["PROMPT_TEMPLATE"].encode("utf-8")).hexdigest(),
    )
    # The task has no persona seat, but the absence still has to be labelled or
    # its races cannot be told apart from persona races.
    assert constants["PERSONA_CONDITION"] == "none"


def test_benchmark_task_labels_persona_on_every_output_row():
    source = BENCHMARK_PATH.read_text(encoding="utf-8")
    assert source.count('"persona_condition": PERSONA_CONDITION') == 3, (
        "turns, players, and races rows must each carry the persona label"
    )


def test_canonical_prompt_predicate_rejects_relabelled_text():
    analyser = _load_analyser()
    canonical = analyser.CANONICAL_PROMPT_VERSION
    known_sha256 = next(iter(analyser.CANONICAL_PROMPT_SHA256S))

    assert analyser._is_canonical_prompt(canonical, known_sha256)
    assert not analyser._is_canonical_prompt(canonical, "0" * 64)
    assert not analyser._is_canonical_prompt("ai-race-paper-v2", known_sha256)
    assert not analyser._is_canonical_prompt(canonical, None)
