"""Manifest provenance written by run_experiment.py for real (non-mock) runs.

Before this, every local run used the lean "ai-race-results-v1" schema, which
omits source/decoding/seed provenance. results/scripts/analyze_ai_race.py then
could not verify two runs shared a protocol, so it fell back to an "unverified"
signature keyed on the run's output path -- persona was perfectly confounded
with the run batch even when every other setting was identical (see
docs/running-proxy-pilots.md). These tests guard the fix: two persona configs
that differ only in `agents` must resolve to the *same* protocol_signature so
the analyser can treat persona as an estimable covariate, not a second batch.
"""
from __future__ import annotations

import copy
import importlib.util
import sys
from pathlib import Path

import pytest

from ai_race.dataio.config_loader import ConfigError
from ai_race.runner.run_experiment import (
    MANIFEST_SCHEMA_LEAN,
    MANIFEST_SCHEMA_PROXY_RUN,
    _agents_provenance,
    _mechanism_provenance,
    _package_versions,
    _prompt_provenance,
    _proxy_decoding_and_seed_provenance,
    _source_tree_sha256,
    _unverified_decoding_and_seed_provenance,
    _write_manifest,
)

pytest.importorskip("pandas", reason="the analyser needs the analysis extra")

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYSER_PATH = REPOSITORY_ROOT / "results" / "scripts" / "analyze_ai_race.py"


def _load_analyser():
    spec = importlib.util.spec_from_file_location("_manifest_analyser", ANALYSER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSER = _load_analyser()

BASE_EXPERIMENT = {
    "name": "test_proxy_experiment",
    "runPhase": "pilot",
    "games": ["ai_race_risk_10", "ai_race_risk_60", "ai_race_risk_90"],
    "agents": "companies_default",
    "languages": ["en"],
    "models": ["google/gemini-3-flash-preview"],
    "useOffline": False,
    "backend": "proxy",
    "samplingSeedApplied": False,
    "proxyOptions": {
        "temperature": 0.7,
        "max_tokens": 256,
        "timeout": 120,
        "max_transport_retries": 6,
        "concurrency": 2,
        "send_seed": False,
    },
    "repetitions": 3,
    "seed": 260726,
    "maxParseRetries": 3,
    "verbose": False,
}


def _persona_variant(agents_name: str) -> dict:
    exp = copy.deepcopy(BASE_EXPERIMENT)
    exp["agents"] = agents_name
    return exp


def _build_real_manifest(exp: dict, *, model: str) -> dict:
    manifest = {
        "created_utc": "2026-08-01T00:00:00+00:00",
        "experiment": exp,
        "model": model,
        "run_phase": exp.get("runPhase", "pilot"),
        **_agents_provenance(exp),
        "status": "completed",
        "n_races": 9,
        "n_turns": 126,
    }
    prompt = _prompt_provenance(exp)
    manifest.update(
        {
            "schema_version": MANIFEST_SCHEMA_PROXY_RUN,
            "source_sha256": _source_tree_sha256(),
            "prompt_version": prompt["version"],
            "prompt_sha256": prompt["sha256"],
            "model_route": model,
            "llm_backend_mro": ["proxy"],
            "mechanism": _mechanism_provenance(exp),
            "package_versions": _package_versions(),
            **_proxy_decoding_and_seed_provenance(exp),
        }
    )
    return manifest


def test_proxy_manifest_passes_verification_without_the_audit_flag():
    manifest = _build_real_manifest(
        _persona_variant("companies_default"), model="google/gemini-3-flash-preview"
    )
    signature, payload, label = ANALYSER._protocol_contract_from_manifest(
        manifest, path=Path("run_manifest.json"), source_run="run", allow_unverified=False
    )
    assert label == "google/gemini-3-flash-preview"
    assert signature
    assert payload["manifest_schema"] == MANIFEST_SCHEMA_PROXY_RUN


def test_two_persona_configs_share_a_protocol_signature():
    """The whole point of the fix: persona must vary *inside* one signature."""
    neutral = _build_real_manifest(
        _persona_variant("persona_neutral"), model="google/gemini-3-flash-preview"
    )
    adversarial = _build_real_manifest(
        _persona_variant("persona_adv_adv"), model="google/gemini-3-flash-preview"
    )
    sig_neutral, _, _ = ANALYSER._protocol_contract_from_manifest(
        neutral, path=Path("a.json"), source_run="run-a", allow_unverified=False
    )
    sig_adversarial, _, _ = ANALYSER._protocol_contract_from_manifest(
        adversarial, path=Path("b.json"), source_run="run-b", allow_unverified=False
    )
    assert sig_neutral == sig_adversarial, (
        "two runs differing only in agents/persona must resolve to the same "
        "protocol_signature, or persona is confounded with the run batch again"
    )


def test_a_real_decoding_difference_still_changes_the_signature():
    exp_a = _persona_variant("companies_default")
    exp_b = copy.deepcopy(exp_a)
    exp_b["proxyOptions"]["temperature"] = 1.0
    manifest_a = _build_real_manifest(exp_a, model="google/gemini-3-flash-preview")
    manifest_b = _build_real_manifest(exp_b, model="google/gemini-3-flash-preview")
    sig_a, _, _ = ANALYSER._protocol_contract_from_manifest(
        manifest_a, path=Path("a.json"), source_run="run-a", allow_unverified=False
    )
    sig_b, _, _ = ANALYSER._protocol_contract_from_manifest(
        manifest_b, path=Path("b.json"), source_run="run-b", allow_unverified=False
    )
    assert sig_a != sig_b


def test_mock_runs_keep_the_lean_schema(tmp_path):
    exp = _persona_variant("companies_default")
    manifest_path = tmp_path / "run_manifest.json"
    _write_manifest(
        manifest_path,
        experiment=exp,
        model="MockModel",
        n_races=1,
        n_turns=2,
        status="completed",
        mock_strategy="random",
    )
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA_LEAN
    assert "source_sha256" not in manifest


def test_real_runs_get_the_provenance_rich_schema(tmp_path):
    exp = _persona_variant("companies_default")
    manifest_path = tmp_path / "run_manifest.json"
    _write_manifest(
        manifest_path,
        experiment=exp,
        model="google/gemini-3-flash-preview",
        n_races=9,
        n_turns=126,
        status="completed",
        mock_strategy=None,
    )
    import json

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == MANIFEST_SCHEMA_PROXY_RUN
    for key in ("source_sha256", "prompt_version", "prompt_sha256", "mechanism", "decoding"):
        assert key in manifest


def test_mechanism_provenance_fails_loudly_on_an_unknown_game():
    exp = _persona_variant("companies_default")
    exp["games"] = ["ai_race_risk_10", "ai_race_risk_60", "baseline_swapped_bad"]
    # An unknown game file must fail loudly, not silently produce a mechanism
    # payload that skips the game the caller actually asked for.
    with pytest.raises(ConfigError):
        _mechanism_provenance(exp)


def test_mechanism_provenance_rejects_a_real_field_mismatch(tmp_path, monkeypatch):
    game_dir = tmp_path / "game"
    game_dir.mkdir()
    base = {
        "name": "g1",
        "engine": "ai_race",
        "nPlayers": 2,
        "safeProgress": 1.0,
        "unsafeProgress": 1.5,
        "stagePayoffs": {
            "safeSafe": 1.0, "safeUnsafe": 0.6, "unsafeSafe": 2.4, "unsafeUnsafe": 2.0,
        },
        "minRounds": 5,
        "stopProbability": 0.2,
        "racePrize": 100,
        "maxPrivateRisk": 0.1,
        "promptTemplate": "ai_race_{language}",
        "promptVersion": "ai-race-fairgame-v3",
        "agents": "companies_default",
    }
    import json as _json

    (game_dir / "g1.json").write_text(_json.dumps(base), encoding="utf-8")
    mismatched = dict(base, name="g2", maxPrivateRisk=0.6, minRounds=10)
    (game_dir / "g2.json").write_text(_json.dumps(mismatched), encoding="utf-8")

    import ai_race.runner.run_experiment as run_experiment_module

    monkeypatch.setattr(run_experiment_module, "CONFIGS_DIR", tmp_path)
    exp = _persona_variant("companies_default")
    exp["games"] = ["g1", "g2"]
    with pytest.raises(ValueError, match="minRounds"):
        _mechanism_provenance(exp)


def test_unverified_fallback_never_claims_a_confirmed_value():
    fallback = _unverified_decoding_and_seed_provenance("openai")
    assert fallback["decoding"]["temperature_effective_confirmed"] is False
    assert fallback["sampling_seed_provenance"]["applied_known"] is False
