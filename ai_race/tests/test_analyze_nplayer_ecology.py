"""Synthetic CPU checks for the N-player ecology analyzer."""
from __future__ import annotations

from dataclasses import asdict
import json

import pandas as pd
import pytest

from results.scripts import analyze_nplayer_ecology as analysis
from kaggle.experiments import greennode_nplayer_ecology as protocol


def _player(module: str, unsafe: float, *, seat: int = 0, model: str = "qwen25_7b") -> dict:
    return {
        "block": "block1", "game_id": f"{module}-{seat}", "module_id": module,
        "n_players": 2, "risk": 0.6, "composition": "all_qwen", "rep": 0,
        "player_index": seat, "seat_model_key": model, "crn_block": "n2:rep0",
        "unsafe_frequency": unsafe, "final_payoff": 10 + unsafe, "setback": False,
    }


def test_progress_rank_distinguishes_leader_middle_last_and_ties() -> None:
    rows = []
    for own, others, expected in [
        (3.0, [2.0, 1.0], "leader"), (2.0, [3.0, 1.0], "middle"),
        (1.0, [3.0, 2.0], "last"), (2.0, [2.0, 1.0], "tied"),
    ]:
        rows.append({"own_progress_before": own, "others_progress_before": others, "n_players": 3})
    ranked = analysis.add_progress_rank(pd.DataFrame(rows))
    assert ranked.progress_rank_category.tolist() == ["leader", "middle", "last", "tied"]
    assert ranked.progress_rank.tolist() == pytest.approx([1.0, 2.0, 3.0, 1.5])


def test_paired_contrasts_use_exact_race_seat_keys() -> None:
    rows = []
    for treatment, control, _ in analysis.PAIR_SPECS:
        rows.extend([_player(treatment, .75), _player(control, .25)])
    players = pd.DataFrame(rows)
    turns = pd.DataFrame([
        {**rows[0], "unsafe": 1, "own_progress_before": 0.0,
         "others_progress_before": [0.0], "progress_rank_percentile": 0.0}
    ])
    tables = analysis.build_tables(players, turns)
    paired = tables["paired_race_seat_contrasts"]
    assert len(paired) == len(analysis.PAIR_SPECS)
    assert paired.unsafe_rate_delta.tolist() == pytest.approx([.5] * len(paired))
    assert set(tables) == {
        "race_seat", "race_crn", "ecology_summary", "progress_rank_summary",
        "paired_race_seat_contrasts", "paired_contrast_summary",
    }


def test_paired_contrast_fails_closed_when_control_is_missing() -> None:
    rows = []
    for treatment, control, _ in analysis.PAIR_SPECS:
        rows.extend([_player(treatment, .75), _player(control, .25)])
    with pytest.raises(ValueError, match="Incomplete paired contrast"):
        analysis.build_tables(pd.DataFrame(rows[:-1]), pd.DataFrame([{
            **rows[0], "unsafe": 1, "own_progress_before": 0.0,
            "others_progress_before": [0.0],
        }]))


def test_full_registry_manifest_and_raw_coverage_validate(tmp_path) -> None:
    races, players, turns = [], [], []
    for module in protocol.MODULES:
        for n_players in protocol.N_PLAYERS:
            for risk in protocol.RISKS:
                for composition in protocol.COMPOSITIONS:
                    game_id = protocol._game_id(module.module_id, n_players, risk, composition, 0)
                    models = protocol.composition_model_keys(composition, n_players)
                    common = {
                        "game_id": game_id, "module_id": module.module_id,
                        "n_players": n_players, "risk": risk,
                        "composition": composition, "rep": 0,
                        "crn_block": f"n{n_players}:rep0",
                    }
                    races.append({**common, "n_rounds": 1, "parse_failures": 0})
                    for seat, model in enumerate(models):
                        players.append({
                            **common, "player_index": seat, "seat_model_key": model,
                            "unsafe_frequency": 0.0, "final_payoff": 1.0,
                            "setback": False,
                        })
                        turns.append({
                            **common, "round": 1, "player_index": seat,
                            "seat_model_key": model, "action": "safe", "unsafe": 0,
                            "parse_failed": False, "own_progress_before": 0.0,
                            "others_progress_before": [0.0] * (n_players - 1),
                        })
    for filename, rows in (("races.jsonl", races), ("players.jsonl", players), ("turns.jsonl", turns)):
        (tmp_path / filename).write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
    manifest = {
        "schema_version": protocol.SCHEMA_VERSION, "protocol": protocol.PROTOCOL,
        "status": "completed", "evidence_class": analysis.EVIDENCE_CLASS,
        "exact_protocol_admission_passed": False,
        "expected_races": protocol.expected_races(), "n_races": len(races),
        "n_turns": len(turns), "outputs": {
            "races": "races.jsonl", "players": "players.jsonl", "turns": "turns.jsonl",
        },
        "design": {
            "modules": [asdict(module) for module in protocol.MODULES],
            "n_players": list(protocol.N_PLAYERS), "risks": list(protocol.RISKS),
            "compositions": list(protocol.COMPOSITIONS),
            "repetitions": protocol.REPETITIONS,
            "alliance_mechanism": False, "alliance_arms": "prompt-framing-only",
        },
    }
    (tmp_path / "run_manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    observed_races, observed_players, observed_turns, _ = analysis.validate_run(tmp_path, "block1")
    assert len(observed_races) == 864
    assert len(observed_players) == 3888
    assert len(observed_turns) == 3888
