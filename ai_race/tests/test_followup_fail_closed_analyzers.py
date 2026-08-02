from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from results.scripts.analyze_payoff_scale_behavior import (
    load_and_validate as load_payoff,
    paired_player_rows,
    summarize as summarize_payoff,
)
from results.scripts.analyze_state_scaffold_factorial import (
    FACTORIAL_CONDITIONS,
    FROZEN_THRESHOLDS,
    PLACEBO_CONDITION,
    analysis_rows,
    factorial_contrasts,
    load_and_validate as load_scaffold,
    placebo_contrasts,
    summarize_contrasts,
)
from results.scripts.followup_analysis_common import sha256


DIGEST = "d" * 64
DECODING = {
    "temperature": 0.0,
    "max_tokens": 16,
    "workers": 16,
    "seed_requested": True,
    "seed_probe_exact_match": True,
}
RISKS = (0.1, 0.6, 0.9)


def common_manifest(schema: str) -> dict:
    return {
        "schema_version": schema,
        "status": "completed",
        "experiment": {"runPhase": "pilot"},
        "experiment_config_sha256": "config-hash",
        "source_sha256": "source-hash",
        "model": {"short_name": "test", "config_sha256": DIGEST},
        "ollama_model": {"name": "test", "digest": DIGEST},
        "decoding": DECODING,
        "base_seed": 260726,
    }


def write_run(
    directory: Path,
    manifest: dict,
    cells: list[tuple[str, str, float]],
    action_by_round: tuple[int, int],
    payoff_multiplier: float = 1.0,
    reps: range = range(32),
) -> None:
    directory.mkdir(parents=True)
    races: list[dict] = []
    players: list[dict] = []
    turns: list[dict] = []
    for condition, mapping, risk in cells:
        for rep in reps:
            game_id = f"{condition}-{mapping}-{risk}-{rep}"
            prompt = (
                f"ai-race-state-scaffold-v1:{condition}:{mapping}"
                if mapping
                else condition
            )
            races.append(
                {
                    "game_id": game_id,
                    "model": "test",
                    "max_private_risk": risk,
                    "prompt_version": prompt,
                    "rep": rep,
                    "game_seed": 260726 + rep,
                    "n_rounds": 2,
                    "stop_draws": "[0.01]",
                    "parse_failures": 0,
                }
            )
            for player in (0, 1):
                unsafe_frequency = sum(action_by_round) / 2
                players.append(
                    {
                        "game_id": game_id,
                        "model": "test",
                        "max_private_risk": risk,
                        "prompt_version": prompt,
                        "rep": rep,
                        "game_seed": 260726 + rep,
                        "n_rounds": 2,
                        "player_index": player,
                        "unsafe_frequency": unsafe_frequency,
                        "final_payoff": payoff_multiplier
                        * (10 + 2 * sum(action_by_round)),
                        "setback_draw": 0.75 + player * 0.01,
                    }
                )
                for round_number, unsafe in enumerate(action_by_round, start=1):
                    turns.append(
                        {
                            "game_id": game_id,
                            "model": "test",
                            "max_private_risk": risk,
                            "prompt_version": prompt,
                            "rep": rep,
                            "game_seed": 260726 + rep,
                            "player_index": player,
                            "round": round_number,
                            "unsafe": unsafe,
                            "parse_failed": False,
                        }
                    )
    pd.DataFrame(races).to_csv(directory / "races.csv", index=False)
    pd.DataFrame(players).to_csv(directory / "players.csv", index=False)
    pd.DataFrame(turns).to_json(directory / "turns.jsonl", orient="records", lines=True)
    manifest = dict(manifest)
    manifest.update(
        expected_races=len(races), n_races=len(races), n_turns=len(turns)
    )
    (directory / "run_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )


def build_payoff_root(root: Path) -> Path:
    schema = "ai-race-payoff-scale-run-v1"
    actions = {
        0.1: (0, 0),
        1.0: (0, 0),
        10.0: (0, 1),
        100.0: (0, 0),
    }
    prompt_ids = {
        0.1: "ai-race-payoff-scale-v1:scale_0p1",
        1.0: "ai-race-payoff-scale-v1:scale_1",
        10.0: "ai-race-payoff-scale-v1:scale_10",
        100.0: "ai-race-payoff-scale-v1:scale_100",
    }
    for lane, reps in (("a", range(0, 32, 2)), ("b", range(1, 32, 2))):
        for scale, action in actions.items():
            manifest = common_manifest(schema)
            manifest["payoff_scale"] = scale
            manifest["lane"] = lane
            write_run(
                root / f"lane-{lane}" / f"scale-{scale}",
                manifest,
                [(prompt_ids[scale], "", risk) for risk in RISKS],
                action,
                payoff_multiplier=scale,
                reps=reps,
            )
    return root


def admission_payload(conditions: tuple[str, ...], raw_path: Path) -> dict:
    cells = {}
    for condition in conditions:
        for mapping in ("safe_p", "safe_q"):
            domains = {
                domain: {
                    "n": 5,
                    "correct": 5,
                    "semantic_accuracy": 1.0,
                    "threshold": threshold,
                    "passed": True,
                }
                for domain, threshold in FROZEN_THRESHOLDS[
                    "domain_semantic_accuracy"
                ].items()
            }
            cells[f"{condition}/{mapping}"] = {
                "condition": condition,
                "mapping_id": mapping,
                "passed": True,
                "coverage_passed": True,
                "n": 20,
                "expected_n": 20,
                "strict_parse_n": 20,
                "strict_parse_correct": 20,
                "strict_parse_rate": 1.0,
                "strict_parse_threshold": 0.95,
                "arithmetic_checks": 1,
                "arithmetic_mismatches": 0,
                "hidden_information_checks": 6,
                "hidden_information_leaks": 0,
                "by_domain": domains,
            }
    return {
        "schema_version": "ai-race-state-scaffold-admission-v1",
        "protocol": "ai-race-state-scaffold-comprehension-v1",
        "passed": True,
        "thresholds": FROZEN_THRESHOLDS,
        "coverage": {"passed": True},
        "model_digest": DIGEST,
        "decoding": DECODING,
        "behavior_source_sha256": "source-hash",
        "behavior_experiment_config_sha256": "config-hash",
        "by_cell": cells,
        "artifacts": {
            "comprehension_raw": {
                "path": raw_path.name,
                "bytes": raw_path.stat().st_size,
                "sha256": sha256(raw_path),
            }
        },
    }


def build_scaffold_root(root: Path) -> tuple[Path, Path]:
    schema = "ai-race-state-scaffold-run-v1"
    actions = {
        "none": (0, 0),
        "transition": (1, 0),
        "terminal": (0, 1),
        "transition_terminal": (1, 1),
        PLACEBO_CONDITION: (0, 0),
    }
    for lane, reps in (("a", range(0, 32, 2)), ("b", range(1, 32, 2))):
        for condition, action in actions.items():
            manifest = common_manifest(schema)
            manifest["condition"] = {"id": condition}
            manifest["lane"] = lane
            cells = [
                (condition, mapping, risk)
                for mapping in ("safe_p", "safe_q")
                for risk in RISKS
            ]
            write_run(
                root / f"lane-{lane}" / condition,
                manifest,
                cells,
                action,
                reps=reps,
            )
    admission = root / "admission.json"
    raw = root / "comprehension_raw.jsonl"
    raw.write_text("{}\n", encoding="utf-8", newline="\n")
    admission.write_text(
        json.dumps(admission_payload(tuple(actions), raw)), encoding="utf-8"
    )
    return root, admission


def test_payoff_analyzer_builds_paired_scale_estimands(tmp_path: Path) -> None:
    players, turns, audit, _ = load_payoff(build_payoff_root(tmp_path / "payoff"))
    paired = paired_player_rows(players, turns)
    summary = summarize_payoff(paired, 1_000).set_index("payoff_scale")
    assert audit["n_races"] == 384
    assert summary.loc[10.0, "trajectory_disagreement_rate"] == 1.0
    assert summary.loc[0.1, "trajectory_disagreement_rate"] == 0.0
    assert summary.loc[100.0, "max_abs_normalized_final_payoff_error"] == 0.0
    assert summary["unsafe_delta_holm_p"].notna().all()


def test_payoff_analyzer_rejects_mixed_model_digest(tmp_path: Path) -> None:
    root = build_payoff_root(tmp_path / "payoff")
    path = root / "lane-b" / "scale-100.0" / "run_manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["model"]["config_sha256"] = "wrong"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="manifest field 'model'"):
        load_payoff(root)


def test_scaffold_analyzer_separates_factorial_and_placebo(tmp_path: Path) -> None:
    root, admission = build_scaffold_root(tmp_path / "scaffold")
    players, turns, audit, cells, _ = load_scaffold(root, admission)
    data = analysis_rows(players, turns)
    factorial = factorial_contrasts(data)
    placebo = placebo_contrasts(data)
    summary = summarize_contrasts(factorial, 1_000)
    transition = summary[
        (summary["endpoint"] == "round1_unsafe")
        & (summary["contrast"] == "transition_main")
    ]
    assert audit["n_races"] == 960
    assert audit["comprehension_admission"]["all_factorial_cells_admitted"]
    assert cells["cell_admitted"].all()
    assert (transition["estimate"] == 1.0).all()
    assert set(placebo["contrast"]) == {"length_placebo_minus_none"}
    assert (placebo["estimate"] == 0.0).all()


def test_scaffold_analyzer_requires_frozen_admission_thresholds(
    tmp_path: Path,
) -> None:
    root, admission = build_scaffold_root(tmp_path / "scaffold")
    payload = json.loads(admission.read_text(encoding="utf-8"))
    payload["thresholds"]["domain_semantic_accuracy"]["state_update"] = 0.5
    admission.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="thresholds differ"):
        load_scaffold(root, admission)


def test_scaffold_analyzer_rejects_crn_horizon_mismatch(tmp_path: Path) -> None:
    root, admission = build_scaffold_root(tmp_path / "scaffold")
    path = root / "lane-a" / "terminal" / "races.csv"
    races = pd.read_csv(path)
    races.loc[0, "n_rounds"] = 3
    races.to_csv(path, index=False)
    with pytest.raises(ValueError, match="hidden horizon"):
        load_scaffold(root, admission)
