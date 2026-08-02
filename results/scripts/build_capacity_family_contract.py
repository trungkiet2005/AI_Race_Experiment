"""Build a deterministic receipt for the capacity/family experiment design."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (
    ROOT / "ai_race" / "configs" / "experiment" / "capacity_family_targeted_context.json"
)
DEFAULT_OUTPUT = ROOT / "results" / "derived" / "capacity_family_protocol_contract"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def build_contract(config: dict[str, Any], config_path: Path) -> dict[str, Any]:
    models = list(config["models"])
    capacity = list(config["capacityAxis"])
    family = list(config["familyAxis"])
    contexts = list(config["contextSkins"])
    mappings = list(config["actionCodeMappings"])
    risks = list(config["games"])
    repetitions = int(config["repetitions"])
    confirmatory = int(config["confirmatoryRepetitions"])

    errors: list[str] = []
    if len(models) != len(set(models)):
        errors.append("model roster contains duplicates")
    if set(capacity) - set(models):
        errors.append("capacity axis contains a model outside the roster")
    if set(family) - set(models):
        errors.append("family axis contains a model outside the roster")
    if len(capacity) != 3 or not all(name.startswith("qwen25-") for name in capacity):
        errors.append("capacity axis must be the three Qwen2.5 checkpoints")
    if len(family) != 3 or len({name.split("-")[0] for name in family}) != 3:
        errors.append("family axis must contain three distinct checkpoint families")
    if contexts != ["abstract_contest", "logistics_contract", "technology_race"]:
        errors.append("targeted context order drifted")
    if mappings != ["safe_p", "safe_q"]:
        errors.append("opaque mapping order drifted")
    if config.get("dtype") != "bfloat16" or config.get("quantization") is not None:
        errors.append("primary panel must use unquantized BF16")
    if config.get("temperature") != 0.0:
        errors.append("primary panel must use greedy decoding")
    if repetitions != 32 or confirmatory != 96:
        errors.append("diagnostic/confirmatory stream counts drifted")
    if config.get("executionAdapter") != "admission_then_targeted_context_v1":
        errors.append("capacity/family execution adapter drifted")
    if config.get("genericRunnerCompatible") is not False:
        errors.append("config must fail closed against the generic race runner")

    per_model_admission = len(risks) * len(mappings) * 16
    diagnostic_races = (
        len(models) * len(contexts) * len(mappings) * len(risks) * repetitions
    )
    confirmatory_races = (
        len(models) * len(contexts) * len(mappings) * len(risks) * confirmatory
    )
    return {
        "schema_version": "ai-race.capacity-family-contract.v1",
        "status": "passed" if not errors else "failed",
        "evidence_class": "protocol",
        "config": {
            "path": config_path.relative_to(ROOT).as_posix(),
            "sha256": sha256(config_path),
        },
        "roster": {
            "all": models,
            "capacity_axis": capacity,
            "family_axis": family,
            "shared_anchor": "qwen25-7b-instruct",
        },
        "frozen_protocol": {
            "contexts": contexts,
            "mappings": mappings,
            "risks": risks,
            "diagnostic_streams": repetitions,
            "confirmatory_streams": confirmatory,
            "dtype": config.get("dtype"),
            "quantization": config.get("quantization"),
            "temperature": config.get("temperature"),
            "backend": config.get("backend"),
            "execution_adapter": config.get("executionAdapter"),
            "generic_runner_compatible": config.get("genericRunnerCompatible"),
        },
        "expected_counts": {
            "admission_probes_per_model": per_model_admission,
            "admission_probes_all_models": per_model_admission * len(models),
            "diagnostic_races_if_all_admitted": diagnostic_races,
            "diagnostic_expected_decisions_at_mean_horizon": diagnostic_races * 18,
            "confirmatory_races_if_all_admitted": confirmatory_races,
        },
        "promotion_gate": {
            "exact_coverage": True,
            "strict_parse_min": 0.95,
            "domain_accuracy_min": 0.80,
            "state_update_min": 0.90,
            "terminal_scoring_min": 0.90,
            "behavior_requires_admission": True,
        },
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    config_path = args.config.resolve()
    output = args.output.resolve()
    payload = build_contract(
        json.loads(config_path.read_text(encoding="utf-8")), config_path
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "contract.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    lines = [
        "# Capacity/family protocol contract",
        "",
        f"- Status: **{payload['status']}**",
        "- Evidence class: **protocol; no model behavior**",
        f"- Admission probes: **{payload['expected_counts']['admission_probes_all_models']:,}**",
        f"- Diagnostic races if all models pass: **{payload['expected_counts']['diagnostic_races_if_all_admitted']:,}**",
        f"- Confirmatory races if all models pass: **{payload['expected_counts']['confirmatory_races_if_all_admitted']:,}**",
        "",
        "This receipt verifies the frozen cell arithmetic and confound contract. It is not an experiment result.",
    ]
    (output / "README.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
    )
    print(json.dumps(payload["expected_counts"], indent=2))
    return 0 if payload["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
