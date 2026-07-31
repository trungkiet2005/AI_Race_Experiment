"""Static contract tests for the private Kaggle prompt-sensitivity workload."""
from __future__ import annotations

import ast
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
NOTEBOOK = ROOT / "kaggle" / "experiments" / "baseline.py"
BOOTSTRAP = ROOT / "kaggle" / "experiments" / "prompt_sensitivity.py"
GREENNODE = ROOT / "kaggle" / "experiments" / "greennode_prompt_sensitivity.py"
CONFIG_DIR = ROOT / "ai_race" / "configs"


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in tree.body:
        if isinstance(node, ast.Assign):
            if any(isinstance(target, ast.Name) and target.id == name for target in node.targets):
                return ast.literal_eval(node.value)
    raise AssertionError(f"{path} does not assign {name}")


def test_prompt_sensitivity_matrix_is_complete_and_configured():
    experiments = _literal_assignment(NOTEBOOK, "PROMPT_SENSITIVITY_EXPERIMENTS")
    assert len(experiments) == len(set(experiments)) == 9

    observed_conditions = []
    for experiment_name in experiments:
        experiment = json.loads(
            (CONFIG_DIR / "experiment" / f"{experiment_name}.json").read_text(
                encoding="utf-8"
            )
        )
        agents = json.loads(
            (CONFIG_DIR / "agents" / f"{experiment['agents']}.json").read_text(
                encoding="utf-8"
            )
        )
        assert experiment["seed"] == 260726
        assert experiment["games"] == [
            "ai_race_risk_10",
            "ai_race_risk_60",
            "ai_race_risk_90",
        ]
        assert experiment["languages"] == ["en"]
        observed_conditions.append(agents.get("personaCondition", "none"))

    assert set(observed_conditions) == {
        "none",
        "R0",
        "R-",
        "R+",
        "S_CC",
        "S_AA",
        "S_AC",
        "S_CA",
    }
    assert observed_conditions.count("none") == 2  # canonical + seat swap


def test_prompt_sensitivity_bootstrap_fails_closed_on_gpu_shape():
    source = BOOTSTRAP.read_text(encoding="utf-8")
    assert 'AI_RACE_RUN_PROFILE", "prompt_sensitivity_smoke"' in source
    assert 'AI_RACE_REQUIRED_GPU", "RTX PRO 6000"' in source
    assert 'AI_RACE_MIN_GPU_VRAM_GIB", "80"' in source


def test_greennode_lanes_partition_the_matrix_without_duplicates():
    experiments = set(_literal_assignment(NOTEBOOK, "PROMPT_SENSITIVITY_EXPERIMENTS"))
    lanes = _literal_assignment(GREENNODE, "LANE_EXPERIMENTS")
    assert set(lanes) == {"a", "b"}
    assert set(lanes["a"]).isdisjoint(lanes["b"])
    assert set(lanes["a"]) | set(lanes["b"]) == experiments
    assert len(lanes["a"]) + len(lanes["b"]) == len(experiments)


def test_greennode_profiles_freeze_smoke_and_pilot_sizes():
    profiles = _literal_assignment(GREENNODE, "PROFILE_REPETITIONS")
    assert profiles == {"smoke": 2, "pilot": 10}
