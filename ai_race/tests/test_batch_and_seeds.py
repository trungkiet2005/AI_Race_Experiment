"""CPU-only tests for common random numbers and per-slot parse retries."""
from __future__ import annotations

from collections.abc import Callable
from typing import Any

from ai_race.engine.game import AIRaceGame
from ai_race.runner.batch import run_games_batched
from ai_race.runner.run_experiment import build_games_for_model


def test_sampling_horizon_and_setback_draws_are_common_across_risk_treatments(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    low = game_factory(
        seed=9001,
        game_id="risk-low",
        name="risk_low",
        max_private_risk=0.1,
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=3,
    )
    high = game_factory(
        seed=9001,
        game_id="risk-high",
        name="risk_high",
        max_private_risk=0.9,
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=3,
    )

    for round_number in (1, 2, 9):
        for retry in (0, 1, 3):
            assert [
                low.sampling_seed(player, round_number, retry) for player in (0, 1)
            ] == [
                high.sampling_seed(player, round_number, retry) for player in (0, 1)
            ]

    low_result = low.apply_round_responses(["ACTION: UNSAFE", "ACTION: SAFE"])
    high_result = high.apply_round_responses(["ACTION: UNSAFE", "ACTION: SAFE"])
    assert low_result is not None
    assert high_result is not None
    assert low_result.stop_draws == high_result.stop_draws
    assert low_result.setback_draws == high_result.setback_draws
    assert low_result.private_risks != high_result.private_risks


def test_experiment_builder_reuses_repetition_seed_across_risk_configs() -> None:
    experiment = {
        "name": "common-seed-test",
        "games": ["ai_race_risk_10", "ai_race_risk_90"],
        "models": ["MockModel"],
        "languages": ["en"],
        "repetitions": 3,
        "seed": 700,
    }
    agents = {
        "name": "companies_default",
        "names": ["Company_A", "Company_B"],
        "personas": {"en": ["", ""]},
    }

    games = build_games_for_model(
        experiment,
        "MockModel",
        agents_cfg=agents,
        agents_name="companies_default",
    )
    assert len(games) == 6
    for rep in range(3):
        treatment_pair = [game for game in games if game.rep == rep]
        assert len(treatment_pair) == 2
        assert {game.seed for game in treatment_pair} == {700 + rep}
        assert len({game.game_id for game in treatment_pair}) == 2
        assert (
            treatment_pair[0].sampling_seed(0, 4, 2)
            == treatment_pair[1].sampling_seed(0, 4, 2)
        )


def test_batch_runner_retries_only_the_failed_slot_with_unchanged_prompt(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    game = game_factory(
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=2,
    )
    calls: list[dict[str, Any]] = []

    def flaky_send(
        prompts: list[str],
        seeds: list[int] | None = None,
    ) -> list[str]:
        calls.append({"prompts": list(prompts), "seeds": list(seeds or [])})
        if len(calls) == 1:
            return ["not a valid action", "ACTION: SAFE"]
        assert len(prompts) == 1
        return ["ACTION: UNSAFE"]

    results = run_games_batched([game], flaky_send, max_parse_retries=3)

    assert len(results) == 1
    assert len(calls) == 2
    assert calls[1]["prompts"] == [calls[0]["prompts"][0]]
    assert calls[0]["seeds"] == [
        game.sampling_seed(0, 1, 0),
        game.sampling_seed(1, 1, 0),
    ]
    assert calls[1]["seeds"] == [game.sampling_seed(0, 1, 1)]
    assert results[0].per_round_actions == [["unsafe", "safe"]]

    turns = sorted(game.turns, key=lambda turn: turn.player_index)
    assert [turn.retry_count for turn in turns] == [1, 0]
    assert [turn.parse_failed for turn in turns] == [False, False]
    assert turns[0].raw_response == "ACTION: UNSAFE"
    assert turns[0].prompt == calls[0]["prompts"][0]
    assert [attempt["parse_failed"] for attempt in turns[0].attempt_history] == [
        True,
        False,
    ]
    assert [attempt["sampling_seed"] for attempt in turns[0].attempt_history] == [
        game.sampling_seed(0, 1, 0),
        game.sampling_seed(0, 1, 1),
    ]


def test_exhausted_retry_records_safe_fallback_and_parse_failure(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    game = game_factory(
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=2,
    )
    calls = 0

    def invalid_send(
        prompts: list[str],
        seeds: list[int] | None = None,
    ) -> list[str]:
        nonlocal calls
        calls += 1
        return ["invalid" for _ in prompts]

    results = run_games_batched([game], invalid_send, max_parse_retries=2)

    assert calls == 3
    assert results[0].per_round_actions == [["safe", "safe"]]
    assert results[0].parse_failures == 2
    assert all(turn.parse_failed for turn in game.turns)
    assert all(turn.retry_count == 2 for turn in game.turns)
    assert all(len(turn.attempt_history) == 3 for turn in game.turns)
