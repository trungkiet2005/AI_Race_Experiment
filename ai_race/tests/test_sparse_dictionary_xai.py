from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = ROOT / "results" / "scripts" / "explain_action_sparse_autoencoder.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("_sparse_dictionary_xai", SCRIPT_PATH)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _turns() -> pd.DataFrame:
    rows = []
    for race in range(20):
        for round_number in range(1, 5):
            rows.append(
                {
                    "run_root": "root-a",
                    "run_group": "lane-a",
                    "run_treatment": "baseline",
                    "game_id": f"game-{race}",
                    "round": round_number,
                    "unsafe": float((race + round_number) % 2),
                    "prompt_template_hash": f"prompt-{race}-{round_number}",
                }
            )
    return pd.DataFrame(rows)


def test_race_split_is_deterministic_and_has_no_trajectory_overlap() -> None:
    module = _load_module()
    train_a, test_a, metadata_a = module.split_turns(
        _turns(), split_unit="race", random_state=42
    )
    train_b, test_b, metadata_b = module.split_turns(
        _turns(), split_unit="race", random_state=42
    )

    assert list(train_a["game_id"]) == list(train_b["game_id"])
    assert list(test_a["game_id"]) == list(test_b["game_id"])
    assert set(train_a["game_id"]).isdisjoint(set(test_a["game_id"]))
    assert metadata_a == metadata_b
    assert metadata_a["group_overlap"] == 0


def test_prompt_hash_split_holds_out_complete_prompt_groups() -> None:
    module = _load_module()
    turns = _turns()
    # Duplicate hashes across races to ensure the split groups by prompt content,
    # not by row or by race identifier.
    turns["prompt_template_hash"] = [
        f"prompt-{int(game_id.split('-')[-1]) % 8}"
        for game_id in turns["game_id"]
    ]
    train, test, metadata = module.split_turns(
        turns, split_unit="prompt_hash", random_state=7
    )

    assert set(train["prompt_template_hash"]).isdisjoint(
        set(test["prompt_template_hash"])
    )
    assert metadata["group_overlap"] == 0
