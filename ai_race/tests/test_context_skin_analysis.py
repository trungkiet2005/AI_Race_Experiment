from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
import pytest


_SCRIPT = Path(__file__).parents[2] / "results" / "scripts" / "analyze_context_skin.py"
_SPEC = importlib.util.spec_from_file_location("context_skin_analysis", _SCRIPT)
assert _SPEC is not None and _SPEC.loader is not None
analysis = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = analysis
_SPEC.loader.exec_module(analysis)


def test_cluster_bootstrap_uses_equal_weight_clusters_and_is_deterministic():
    rows = pd.DataFrame(
        {
            "cluster": ["race-a", "race-a", "race-b"],
            "difference": [0.0, 0.0, 1.0],
        }
    )
    first = analysis.cluster_bootstrap_mean(
        rows,
        "difference",
        "cluster",
        repetitions=500,
        seed=19,
    )
    second = analysis.cluster_bootstrap_mean(
        rows,
        "difference",
        "cluster",
        repetitions=500,
        seed=19,
    )
    assert first == second
    assert first[0] == pytest.approx(0.5)
    assert first[3] == 2


def test_paired_rows_keeps_mapping_in_key_and_drops_failed_side():
    rows = pd.DataFrame(
        [
            {"skin_id": "abstract_contest", "state_id": "s1", "mapping_id": "safe_p", "unsafe": 0, "parse_failed": False},
            {"skin_id": "hospital_deployment", "state_id": "s1", "mapping_id": "safe_p", "unsafe": 1, "parse_failed": False},
            {"skin_id": "abstract_contest", "state_id": "s1", "mapping_id": "safe_q", "unsafe": 0, "parse_failed": False},
            {"skin_id": "hospital_deployment", "state_id": "s1", "mapping_id": "safe_q", "unsafe": 1, "parse_failed": True},
        ]
    )
    paired = analysis._paired_rows(
        rows,
        "hospital_deployment",
        "abstract_contest",
        ["state_id", "mapping_id"],
        "unsafe",
    )
    assert paired[["state_id", "mapping_id"]].to_dict("records") == [
        {"state_id": "s1", "mapping_id": "safe_p"}
    ]
    assert paired.iloc[0]["difference"] == 1


def test_discovery_rejects_silent_pooling_of_incompatible_cohorts(tmp_path: Path):
    for index, temperature in enumerate((0.0, 0.7)):
        run = tmp_path / f"run-{index}"
        run.mkdir()
        (run / "run_manifest.json").write_text(
            __import__("json").dumps(
                {
                    "schema_version": analysis.LIVE_SCHEMA,
                    "status": "completed",
                    "profile": "pilot",
                    "source_sha256": "source",
                    "experiment_config_sha256": "config",
                    "decoding": {"temperature": temperature},
                    "ollama_model": {"name": "model", "digest": "digest"},
                }
            ),
            encoding="utf-8",
        )
    with pytest.raises(ValueError, match="Multiple incompatible cohorts"):
        analysis.discover_runs([tmp_path], analysis.LIVE_SCHEMA)
