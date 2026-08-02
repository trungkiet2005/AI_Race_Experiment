from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).parents[2]
ANALYSIS_ROOT = (
    ROOT
    / "results"
    / "open_source"
    / "activation_sae"
    / "context_fast_sae_analysis"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def test_context_fast_sae_analysis_preserves_evidence_boundaries_and_hashes():
    summary = json.loads((ANALYSIS_ROOT / "summary.json").read_text(encoding="utf-8"))
    assert summary["schema_version"] == "ai-race.context-fast-sae-analysis.v1"
    assert summary["cross_layer_audit"]["passed"]
    assert set(summary["evidence_classes"]) == {
        "context_shift_descriptive",
        "heldout_probe_association",
        "heldout_causal_intervention",
    }
    for name, artifact in summary["outputs"].items():
        path = ANALYSIS_ROOT / name
        assert path.stat().st_size == artifact["bytes"]
        assert _sha256(path) == artifact["sha256"]


def test_only_layer20_descriptive_lane_is_promoted_and_causal_claims_are_rejected():
    summary = json.loads((ANALYSIS_ROOT / "summary.json").read_text(encoding="utf-8"))
    by_layer = {row["layer"]: row for row in summary["layer_decisions"]}
    assert by_layer[12]["pilot_decision"] == "HOLD"
    assert by_layer[20]["pilot_decision"] == "PROMOTE_CAPTURE_ANALYZE_ONLY"
    for row in by_layer.values():
        assert row["discovery_action_flips"] == 1
        assert row["max_intervention_flip_rate"] == 0.0
        assert row["target_sign_reversal_features"] == 0
        assert row["causal_mediation_admitted"] is False

