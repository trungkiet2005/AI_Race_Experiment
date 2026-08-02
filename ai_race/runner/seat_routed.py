"""Lockstep race execution with an explicit backend route for every seat.

The canonical batch runner intentionally sends every request to one backend.
Heterogeneous-model dyads need a different contract: the dispatcher receives
the game and seat for every request and must return responses in the same
order.  Keeping this as a separate runner prevents a mixed-model pilot from
silently masquerading as the paper-faithful homogeneous baseline.
"""
from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from ai_race.engine.round import parse_action, raw_response_text


@dataclass(frozen=True)
class SeatRequest:
    """One model request with the routing information kept explicit."""

    game: Any
    player_index: int
    prompt: str
    sampling_seed: int
    attempt: int = 0


SeatDispatcher = Callable[[Sequence[SeatRequest]], Sequence[str | dict[str, Any]]]


def _attempt_record(
    response: str | dict[str, Any], request: SeatRequest, *, seed_applied: bool
) -> dict[str, Any]:
    action, parse_failed = parse_action(response)
    return {
        "attempt": int(request.attempt),
        "sampling_seed": int(request.sampling_seed) if seed_applied else None,
        "sampling_seed_applied": bool(seed_applied),
        "raw_response": raw_response_text(response),
        "parse_failed": bool(parse_failed),
        "parsed_action": None if parse_failed else action.value,
    }


def _dispatch(
    dispatcher: SeatDispatcher, requests: Sequence[SeatRequest]
) -> list[str | dict[str, Any]]:
    responses = list(dispatcher(requests))
    if len(responses) != len(requests):
        raise RuntimeError(
            f"Seat dispatcher returned {len(responses)} responses for "
            f"{len(requests)} requests"
        )
    return responses


def run_games_seat_routed(
    games: Sequence[Any],
    dispatcher: SeatDispatcher,
    *,
    prompt_transform: Callable[[Any, int, str], str] | None = None,
    verbose: bool = False,
    max_parse_retries: int = 3,
    on_round_complete: Callable[[Any, Any | None, list[Any]], None] | None = None,
) -> list[Any]:
    """Run two-player races while preserving a game+seat route per request."""
    if max_parse_retries < 0:
        raise ValueError("max_parse_retries cannot be negative")
    games = list(games)
    if any(game.is_finished for game in games):
        raise ValueError("All input games must be unstarted")
    results: list[Any] = []

    while True:
        active = [game for game in games if not game.is_finished]
        if not active:
            break

        requests: list[SeatRequest] = []
        prompts_by_game: dict[str, list[str]] = {}
        for game in active:
            prompts = list(game.build_round_prompts())
            if len(prompts) != 2:
                raise ValueError("Seat-routed dyad runner requires exactly two seats")
            if prompt_transform is not None:
                prompts = [
                    prompt_transform(game, player_index, prompt)
                    for player_index, prompt in enumerate(prompts)
                ]
            prompts_by_game[game.game_id] = prompts
            for player_index, prompt in enumerate(prompts):
                requests.append(
                    SeatRequest(
                        game=game,
                        player_index=player_index,
                        prompt=prompt,
                        sampling_seed=game.sampling_seed(
                            player_index, game.current_round, 0
                        ),
                    )
                )

        responses = _dispatch(dispatcher, requests)
        retry_counts = [0 for _ in requests]
        attempt_histories = [
            [
                _attempt_record(
                    response,
                    requests[index],
                    seed_applied=bool(requests[index].game.config.sampling_seed_applied),
                )
            ]
            for index, response in enumerate(responses)
        ]

        for attempt in range(1, max_parse_retries + 1):
            failed_indices = [
                index
                for index, response in enumerate(responses)
                if parse_action(response)[1]
            ]
            if not failed_indices:
                break
            retry_requests = [
                SeatRequest(
                    game=requests[index].game,
                    player_index=requests[index].player_index,
                    prompt=requests[index].prompt,
                    sampling_seed=requests[index].game.sampling_seed(
                        requests[index].player_index,
                        requests[index].game.current_round,
                        attempt,
                    ),
                    attempt=attempt,
                )
                for index in failed_indices
            ]
            retry_responses = _dispatch(dispatcher, retry_requests)
            for index, request, response in zip(
                failed_indices, retry_requests, retry_responses
            ):
                responses[index] = response
                retry_counts[index] = attempt
                attempt_histories[index].append(
                    _attempt_record(
                        response,
                        request,
                        seed_applied=bool(request.game.config.sampling_seed_applied),
                    )
                )

        cursor = 0
        for game in active:
            game_responses = responses[cursor : cursor + 2]
            game_retries = retry_counts[cursor : cursor + 2]
            game_attempts = attempt_histories[cursor : cursor + 2]
            cursor += 2
            previous_turn_count = len(game.turns)
            result = game.apply_round_responses(
                game_responses,
                prompts=prompts_by_game[game.game_id],
                retry_counts=game_retries,
                attempt_histories=game_attempts,
            )
            new_turns = list(game.turns[previous_turn_count:])
            if on_round_complete is not None:
                on_round_complete(game, result, new_turns)
            if verbose:
                actions = game.history[-1]["actions"]
                print(
                    f"[{game.game_id}] round={len(game.history)} "
                    f"actions={actions} progress={game.progress}"
                )
            if result is not None:
                results.append(result)
    return results
