"""Schema-contract tests for durable turn, race, and player outputs."""
from __future__ import annotations

import csv
import json
from collections.abc import Callable

from ai_race.dataio.recorder import (
    player_rows,
    race_row,
    write_players_csv,
    write_races_csv,
    write_turns_jsonl,
)
from ai_race.engine.game import AIRaceGame


TURN_SCHEMA = {
    "game_id",
    "model",
    "max_private_risk",
    "prompt_version",
    "run_phase",
    "rep",
    "game_seed",
    "sampling_seed",
    "sampling_seed_applied",
    "round",
    "player",
    "player_index",
    "opponent",
    "action",
    "unsafe",
    "parse_failed",
    "retry_count",
    "reasoning",
    "raw_response",
    "prompt",
    "own_prev_action",
    "opponent_prev_action",
    "own_progress_before",
    "opponent_progress_before",
    "progress_gap_before",
    "own_stage_payoff_before",
    "opponent_stage_payoff_before",
    "own_private_risk_before",
    "opponent_private_risk_before",
    "round_payoff",
    "step_increment",
    "own_progress_after",
    "opponent_progress_after",
    "progress_gap_after",
    "cumulative_stage_payoff_after",
    "unsafe_count_after",
    "unsafe_fraction_after",
    "current_private_risk_after",
    "stop_draw",
    "stopped",
    "attempt_history",
    "logprobs",
    "latency_ms",
}

RACE_SCHEMA = {
    "game_id",
    "model",
    "max_private_risk",
    "prompt_version",
    "run_phase",
    "rep",
    "game_seed",
    "n_rounds",
    "stop_forced",
    "tie",
    "winner",
    "player_1",
    "player_2",
    "player_1_progress",
    "player_2_progress",
    "player_1_stage_payoff",
    "player_2_stage_payoff",
    "player_1_unsafe_count",
    "player_2_unsafe_count",
    "player_1_unsafe_frequency",
    "player_2_unsafe_frequency",
    "player_1_private_risk",
    "player_2_private_risk",
    "player_1_prize",
    "player_2_prize",
    "player_1_setback",
    "player_2_setback",
    "player_1_final_payoff",
    "player_2_final_payoff",
    "parse_failures",
    "stop_draws",
}

PLAYER_SCHEMA = {
    "game_id",
    "model",
    "max_private_risk",
    "prompt_version",
    "run_phase",
    "rep",
    "game_seed",
    "n_rounds",
    "player",
    "player_index",
    "opponent",
    "outcome",
    "progress",
    "stage_payoff",
    "unsafe_count",
    "unsafe_frequency",
    "private_risk",
    "prize",
    "setback_eligible",
    "setback_draw",
    "setback",
    "final_payoff",
}


def _finished_game(game_factory: Callable[..., AIRaceGame]) -> AIRaceGame:
    game = game_factory(
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=2,
        max_private_risk=0.0,
    )
    responses = [
        {
            "text": "ACTION: UNSAFE",
            "token_logprobs": [-0.2],
            "cumulative_logprob": -0.2,
        },
        "ACTION: SAFE",
    ]
    assert game.apply_round_responses(responses, retry_counts=[1, 0]) is not None
    return game


def test_turn_jsonl_schema_and_structured_response_metadata(
    game_factory: Callable[..., AIRaceGame],
    tmp_path,
) -> None:
    game = _finished_game(game_factory)
    turn = sorted(game.turns, key=lambda item: item.player_index)[0]
    assert set(turn.to_dict()) == TURN_SCHEMA
    assert turn.logprobs == {
        "token_logprobs": [-0.2],
        "cumulative_logprob": -0.2,
    }
    assert turn.attempt_history[0]["attempt"] == 1
    assert turn.attempt_history[0]["raw_response"].endswith("ACTION: UNSAFE")

    path = tmp_path / "nested" / "turns.jsonl"
    write_turns_jsonl(game.turns, path)
    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 2
    payload = json.loads(lines[0])
    assert set(payload) == TURN_SCHEMA
    assert payload["game_id"] == game.game_id
    assert payload["action"] == "unsafe"


def test_race_and_player_rows_have_stable_auditable_schemas(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    game = _finished_game(game_factory)
    result = game.result
    assert result is not None

    race = race_row(result)
    players = player_rows(result)
    assert set(race) == RACE_SCHEMA
    assert len(players) == 2
    assert all(set(row) == PLAYER_SCHEMA for row in players)
    assert json.loads(race["stop_draws"]) == result.stop_draws
    assert race["winner"] == "Company_A"
    assert players[0]["opponent"] == "Company_B"
    assert players[1]["opponent"] == "Company_A"
    assert [row["setback_draw"] for row in players] == result.setback_draws


def test_csv_writers_use_the_declared_race_and_player_headers(
    game_factory: Callable[..., AIRaceGame],
    tmp_path,
) -> None:
    game = _finished_game(game_factory)
    result = game.result
    assert result is not None
    races_path = tmp_path / "races.csv"
    players_path = tmp_path / "players.csv"

    write_races_csv([result], races_path)
    write_players_csv([result], players_path)

    with races_path.open(newline="", encoding="utf-8") as handle:
        race_reader = csv.DictReader(handle)
        race_rows = list(race_reader)
    with players_path.open(newline="", encoding="utf-8") as handle:
        player_reader = csv.DictReader(handle)
        player_output = list(player_reader)

    assert set(race_reader.fieldnames or []) == RACE_SCHEMA
    assert set(player_reader.fieldnames or []) == PLAYER_SCHEMA
    assert len(race_rows) == 1
    assert len(player_output) == 2
