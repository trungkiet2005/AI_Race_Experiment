"""Recorder output-shape tests for the N-player engine."""
from __future__ import annotations

import csv
import json

from ai_race.engine.agent import RaceAgent
from ai_race.engine_nplayer.game import NPlayerAIRaceGame
from ai_race.engine_nplayer.recorder import NPlayerRunJournal
from ai_race.engine_nplayer.state import NPlayerGameConfig


def _play_short_race(n_players: int = 3, seed: int = 1) -> NPlayerAIRaceGame:
    config = NPlayerGameConfig(
        name="test",
        n_players=n_players,
        min_rounds=2,
        stop_probability=1.0,
        max_private_risk=0.5,
    )
    agents = [RaceAgent(name=f"Company_{i + 1}") for i in range(n_players)]
    game = NPlayerAIRaceGame(config, agents, template="", game_id="rec1", seed=seed)
    for _ in range(2):
        responses = ["ACTION: SAFE"] + ["ACTION: UNSAFE"] * (n_players - 1)
        game.apply_round_responses(responses)
    return game


def test_journal_writes_turns_races_and_players_files(tmp_path):
    game = _play_short_race(n_players=3)
    journal = NPlayerRunJournal(tmp_path, reset=True)
    # Replay round-by-round the way run_games_batched's on_round_complete does.
    turns_seen = 0
    for round_index in range(len(game.history)):
        n_players = 3
        round_turns = game.turns[
            round_index * n_players : (round_index + 1) * n_players
        ]
        is_last = round_index == len(game.history) - 1
        journal.record_round(game, game.result if is_last else None, round_turns)
        turns_seen += len(round_turns)

    turns_path = tmp_path / "turns.jsonl"
    races_path = tmp_path / "races.csv"
    players_path = tmp_path / "players.csv"
    assert turns_path.is_file()
    assert races_path.is_file()
    assert players_path.is_file()

    turn_lines = turns_path.read_text(encoding="utf-8").splitlines()
    assert len(turn_lines) == turns_seen == 3 * 2  # n_players * n_rounds
    first_turn = json.loads(turn_lines[0])
    assert "others" in first_turn and len(first_turn["others"]) == 2
    assert "opponent" not in first_turn

    with races_path.open(encoding="utf-8") as handle:
        race_rows = list(csv.DictReader(handle))
    assert len(race_rows) == 1
    assert int(race_rows[0]["n_players"]) == 3
    winners = json.loads(race_rows[0]["winners"])
    assert isinstance(winners, list)

    with players_path.open(encoding="utf-8") as handle:
        player_rows = list(csv.DictReader(handle))
    assert len(player_rows) == 3
    others_field = json.loads(player_rows[0]["others"])
    assert len(others_field) == 2
    assert player_rows[0]["player"] not in others_field


def test_journal_reset_clears_previous_files(tmp_path):
    (tmp_path / "turns.jsonl").write_text("stale\n", encoding="utf-8")
    journal = NPlayerRunJournal(tmp_path, reset=True)
    assert not (tmp_path / "turns.jsonl").exists()
    assert journal.turn_count == 0
