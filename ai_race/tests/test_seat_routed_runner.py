"""CPU-only contract tests for heterogeneous-seat dispatch."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from pathlib import Path

from ai_race.engine.game import AIRaceGame
from ai_race.engine.agent import RaceAgent
from ai_race.engine_nplayer.game import NPlayerAIRaceGame
from ai_race.engine_nplayer.state import NPlayerGameConfig
from ai_race.runner.seat_routed import SeatRequest, run_games_seat_routed
from kaggle.experiments.greennode_heterogeneous_dyad import (
    build_games,
    identity_prompt,
)


def test_seat_route_and_prompt_transform_are_preserved(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    game = game_factory(
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=2,
    )
    routed: list[tuple[int, str, int]] = []

    def dispatch(requests: Sequence[SeatRequest]) -> list[str]:
        for request in requests:
            routed.append(
                (request.player_index, request.prompt, request.sampling_seed)
            )
        return [
            "ACTION: SAFE" if request.player_index == 0 else "ACTION: UNSAFE"
            for request in requests
        ]

    def transform(_: AIRaceGame, player_index: int, prompt: str) -> str:
        return prompt + f"\nROUTE-SEAT: {player_index}\n"

    results = run_games_seat_routed(
        [game], dispatch, prompt_transform=transform, max_parse_retries=0
    )

    assert results[0].per_round_actions == [["safe", "unsafe"]]
    assert [row[0] for row in routed] == [0, 1]
    assert "ROUTE-SEAT: 0" in routed[0][1]
    assert "ROUTE-SEAT: 1" in routed[1][1]
    assert [row[2] for row in routed] == [
        game.sampling_seed(0, 1, 0),
        game.sampling_seed(1, 1, 0),
    ]
    assert [turn.prompt for turn in game.turns] == [routed[0][1], routed[1][1]]


def test_retry_keeps_the_failed_seat_route_and_seed(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    game = game_factory(
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=2,
    )
    calls: list[list[SeatRequest]] = []

    def dispatch(requests: Sequence[SeatRequest]) -> list[str]:
        calls.append(list(requests))
        if len(calls) == 1:
            return ["invalid", "ACTION: SAFE"]
        return ["ACTION: UNSAFE"]

    results = run_games_seat_routed([game], dispatch, max_parse_retries=2)

    assert results[0].per_round_actions == [["unsafe", "safe"]]
    assert [[request.player_index for request in call] for call in calls] == [
        [0, 1],
        [0],
    ]
    assert calls[1][0].prompt == calls[0][0].prompt
    assert calls[1][0].sampling_seed == game.sampling_seed(0, 1, 1)
    assert calls[1][0].attempt == 1


def test_seat_routed_runner_supports_three_player_games() -> None:
    config = NPlayerGameConfig(
        name="seat-route-n3",
        n_players=3,
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=2,
    )
    template = (
        Path(__file__).resolve().parents[1]
        / "engine_nplayer"
        / "prompts"
        / "ai_race_nplayer_en.txt"
    ).read_text(encoding="utf-8")
    game = NPlayerAIRaceGame(
        config,
        [RaceAgent(f"Participant_{index + 1}") for index in range(3)],
        template=template,
        game_id="seat-route-n3",
        seed=17,
    )
    routed: list[int] = []

    def dispatch(requests: Sequence[SeatRequest]) -> list[str]:
        routed.extend(request.player_index for request in requests)
        return [
            "ACTION: UNSAFE" if request.player_index == 2 else "ACTION: SAFE"
            for request in requests
        ]

    results = run_games_seat_routed(
        [game], dispatch, max_parse_retries=0
    )

    assert routed == [0, 1, 2]
    assert results[0].per_round_actions == [["safe", "safe", "unsafe"]]


def test_heterogeneous_design_crosses_identity_and_reverses_cross_family_seats() -> None:
    workers = {"qwen25_7b": "worker-q", "mistral7_01": "worker-m"}
    games = build_games("smoke", workers)

    assert len(games) == 192
    assert len({game.game_id for game in games}) == len(games)
    cross = [
        game
        for game in games
        if game._heterogeneous["dyad_type"] == "cross_family"
    ]
    assert {
        tuple(game._heterogeneous["seat_model_keys"]) for game in cross
    } == {
        ("qwen25_7b", "mistral7_01"),
        ("mistral7_01", "qwen25_7b"),
    }
    assert {
        (
            game._heterogeneous["self_identity_condition"],
            game._heterogeneous["opponent_identity_condition"],
        )
        for game in games
    } == {
        ("not_disclosed", "not_disclosed"),
        ("not_disclosed", "accurate"),
        ("accurate", "not_disclosed"),
        ("accurate", "accurate"),
    }


def test_identity_prompt_separates_self_and_opponent_disclosure() -> None:
    workers = {"qwen25_7b": "worker-q", "mistral7_01": "worker-m"}
    game = next(
        game
        for game in build_games("smoke", workers)
        if game._heterogeneous["seat_model_keys"]
        == ["qwen25_7b", "mistral7_01"]
        and game._heterogeneous["self_identity_condition"] == "accurate"
        and game._heterogeneous["opponent_identity_condition"] == "not_disclosed"
    )
    prompt = identity_prompt(game, 0, "BASE")

    assert "Your endpoint is accurately identified as Qwen2.5-7B-Instruct" in prompt
    assert "opponent endpoint identity is not disclosed" in prompt
    assert "Mistral-7B-Instruct-v0.1" not in prompt
