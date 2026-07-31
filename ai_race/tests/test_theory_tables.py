"""The theory table script must stay data-free and self-consistent.

Its outputs are the ones most likely to be misread — a predicted stationary
distribution looks a lot like an observed strategy classification — so the naming
convention and the metadata caveats are tested, not just the numbers.
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPOSITORY_ROOT / "results" / "scripts" / "build_theory_tables.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("_build_theory_tables", SCRIPT_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


SCRIPT = _load_script()


def _read_csv(path: Path) -> list[dict[str, str]]:
    import csv

    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def test_script_writes_every_theory_table(tmp_path):
    assert SCRIPT.main(["--output", str(tmp_path)]) == 0
    written = {path.name for path in tmp_path.iterdir()}
    assert written == {
        "theory_payoff_matrix.csv",
        "theory_equilibria.csv",
        "theory_stationary_distribution.csv",
        "theory_expected_unsafe.csv",
        "theory_metadata.json",
    }
    # Every output is prefixed so it cannot be confused with a behavioural table.
    assert all(name.startswith("theory_") for name in written)


def test_equilibria_table_reproduces_the_papers_three_treatments(tmp_path):
    SCRIPT.main(["--output", str(tmp_path)])
    rows = {
        float(row["max_private_risk"]): row
        for row in _read_csv(tmp_path / "theory_equilibria.csv")
    }
    assert set(rows) == {0.1, 0.6, 0.9}
    for row in rows.values():
        assert row["stage_game_class"] == "deadlock"
        assert row["as_is_nash"] == "False"
        assert float(row["social_dilemma_threshold"]) == pytest.approx(1 - 59 / 68)
    assert set(rows[0.1]["symmetric_nash_strategies"].split("|")) == {"AU", "CAS"}
    assert set(rows[0.6]["symmetric_nash_strategies"].split("|")) == {"AU", "CAS"}
    assert rows[0.9]["symmetric_nash_strategies"] == "CS"
    assert rows[0.1]["above_social_dilemma_threshold"] == "False"
    assert rows[0.9]["above_social_dilemma_threshold"] == "True"


def test_payoff_matrix_is_complete_and_matches_the_paper(tmp_path):
    SCRIPT.main(["--output", str(tmp_path)])
    rows = _read_csv(tmp_path / "theory_payoff_matrix.csv")
    assert len(rows) == 3 * 16
    low = {
        (row["own_strategy"], row["opponent_strategy"]): float(row["payoff"])
        for row in rows
        if float(row["max_private_risk"]) == 0.1
    }
    assert low[("AS", "AS")] == pytest.approx(59.0, abs=1e-6)
    assert low[("AS", "AU")] == pytest.approx(5.4, abs=1e-6)
    assert low[("AU", "AS")] == pytest.approx(0.9 * 121.6, abs=1e-6)
    assert low[("AU", "AU")] == pytest.approx(0.9 * 68.0, abs=1e-6)


def test_monte_carlo_mode_labels_each_cell_by_its_route(tmp_path):
    SCRIPT.main(
        [
            "--output",
            str(tmp_path),
            "--payoff-method",
            "monte_carlo",
            "--replications",
            "500",
        ]
    )
    rows = _read_csv(tmp_path / "theory_payoff_matrix.csv")
    methods = {row["method"] for row in rows}
    assert methods == {"closed_form", "monte_carlo"}
    # A seed on a closed-form row would imply its value depends on sampling.
    for row in rows:
        if row["method"] == "closed_form":
            assert row["seed"] == "" and row["replications"] == "0"
        else:
            assert row["seed"] == "260726"


def test_stationary_table_is_labelled_as_a_limit_not_a_mutation_rate(tmp_path):
    SCRIPT.main(["--output", str(tmp_path)])
    rows = _read_csv(tmp_path / "theory_stationary_distribution.csv")
    assert {row["mutation_regime"] for row in rows} == {"small_mutation_limit"}
    # nominal_mu records the paper's parameter point; it is not applied.
    assert {row["nominal_mu"] for row in rows} == {"0.02", "0.05"}

    by_cell: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["max_private_risk"], row["parameter_point"])
        by_cell[key] = by_cell.get(key, 0.0) + float(row["frequency"])
    for total in by_cell.values():
        assert total == pytest.approx(1.0)


def test_metadata_warns_that_the_prediction_is_model_independent(tmp_path):
    SCRIPT.main(["--output", str(tmp_path)])
    metadata = json.loads((tmp_path / "theory_metadata.json").read_text())
    assert metadata["reads_experiment_output"] is False
    assert "not evidence about a model" in metadata["independence_warning"]
    assert "strategy_summary_player.csv" in metadata["naming_warning"]
    # The limitation must be stated where a reader of the CSV will find it.
    assert "is NOT applied" in metadata["mutation_regime_caveat"]
    assert "p_r^max = 0.2" in metadata["mutation_regime_caveat"]


def test_the_script_reads_the_checked_in_game_configs(tmp_path):
    configs = SCRIPT.load_game_configs()
    assert [config.max_private_risk for config in configs] == [0.1, 0.6, 0.9]
    assert all(config.engine == "ai_race" for config in configs)
