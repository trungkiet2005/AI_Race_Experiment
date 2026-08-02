from __future__ import annotations

import json
from pathlib import Path

import pytest

from ai_race.runner.run_experiment import build_games_for_model
from kaggle.experiments.kaggle_crossmodel_scaffold_admission import (
    evaluate_hardware_gate,
    validate_model_runtime_layout,
)
from kaggle.experiments.greennode_crossmodel_scaffold_admission import (
    evaluate_greennode_hardware_gate,
)
from results.scripts.build_capacity_family_contract import build_contract


ROOT = Path(__file__).resolve().parents[2]
CONFIG = (
    ROOT / "ai_race/configs/experiment/capacity_family_targeted_context.json"
)


def test_capacity_family_contract_is_balanced_and_frozen() -> None:
    payload = build_contract(
        json.loads(CONFIG.read_text(encoding="utf-8")), CONFIG
    )
    assert payload["status"] == "passed"
    assert payload["expected_counts"]["admission_probes_all_models"] == 480
    assert payload["expected_counts"]["diagnostic_races_if_all_admitted"] == 2880
    assert payload["expected_counts"]["confirmatory_races_if_all_admitted"] == 8640


def test_generic_runner_cannot_drop_context_or_mapping_dimensions() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="not compatible with the generic race runner"):
        build_games_for_model(config, config["models"][0])


def test_p100_is_rejected_before_weights_load() -> None:
    receipt = evaluate_hardware_gate(
        "qwen25_7b",
        cuda_available=True,
        gpu_name="Tesla P100-PCIE-16GB",
        gpu_count=1,
        total_vram_bytes=16 * 1024**3,
        compute_capability=(6, 0),
        bf16_supported=False,
    )
    assert receipt["passed"] is False
    assert receipt["checks"]["gpu_name"] is False
    assert receipt["checks"]["compute_capability"] is False
    assert receipt["checks"]["bf16_supported"] is False


def test_rtx_pro_6000_96gb_admits_qwen32() -> None:
    receipt = evaluate_hardware_gate(
        "qwen25_32b",
        cuda_available=True,
        gpu_name="NVIDIA RTX PRO 6000 Blackwell Server Edition",
        gpu_count=1,
        total_vram_bytes=96 * 1024**3,
        compute_capability=(12, 0),
        bf16_supported=True,
    )
    assert receipt["passed"] is True


def test_48gb_lane_cannot_silently_run_qwen32() -> None:
    receipt = evaluate_hardware_gate(
        "qwen25_32b",
        cuda_available=True,
        gpu_name="NVIDIA RTX PRO 6000",
        gpu_count=1,
        total_vram_bytes=48 * 1024**3,
        compute_capability=(9, 0),
        bf16_supported=True,
    )
    assert receipt["passed"] is False
    assert receipt["checks"]["vram"] is False


def test_greennode_48gb_lane_routes_qwen14_but_not_qwen32() -> None:
    common = {
        "cuda_available": True,
        "gpu_name": "NVIDIA RTX 6000 Ada Generation",
        "gpu_count": 1,
        "free_vram_bytes": 47 * 1024**3,
        "total_vram_bytes": 48 * 1024**3,
        "compute_capability": (8, 9),
        "bf16_supported": True,
    }
    qwen14 = evaluate_greennode_hardware_gate("qwen25_14b", **common)
    qwen32 = evaluate_greennode_hardware_gate("qwen25_32b", **common)
    assert qwen14["passed"] is True
    assert qwen32["passed"] is False
    assert qwen32["checks"]["free_vram"] is False


def test_primary_panel_rejects_parameter_offload_and_dtype_drift() -> None:
    with pytest.raises(RuntimeError, match="forbids CPU/disk parameter offload"):
        validate_model_runtime_layout(
            ["cuda:0", "cpu"], ["torch.bfloat16"], {"layer.0": 0, "layer.1": "cpu"}
        )
    with pytest.raises(RuntimeError, match="requires every model parameter"):
        validate_model_runtime_layout(["cuda:0"], ["torch.float16"], {"": 0})
    receipt = validate_model_runtime_layout(
        ["cuda:0"], ["torch.bfloat16"], {"": 0}
    )
    assert receipt["parameter_devices"] == ["cuda:0"]
    assert receipt["parameter_dtypes"] == ["torch.bfloat16"]
