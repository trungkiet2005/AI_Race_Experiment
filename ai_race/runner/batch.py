"""Run variable-horizon races in lockstep while batching model calls."""
from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

from ai_race.engine.round import parse_action, raw_response_text


def _attempt_record(
    response: str | dict[str, Any],
    *,
    attempt: int,
    sampling_seed: int,
    sampling_seed_applied: bool,
) -> dict[str, Any]:
    action, parse_failed = parse_action(response)
    return {
        "attempt": int(attempt),
        "sampling_seed": int(sampling_seed) if sampling_seed_applied else None,
        "sampling_seed_applied": bool(sampling_seed_applied),
        "raw_response": raw_response_text(response),
        "parse_failed": bool(parse_failed),
        "parsed_action": None if parse_failed else action.value,
    }


def _send(
    send_batch: Callable[..., Sequence[str | dict[str, Any]]],
    prompts: list[str],
    seeds: list[int],
) -> list[str | dict[str, Any]]:
    # Every built-in connector exposes ``seeds=``. Silently retrying without it
    # would make the recorded sampling provenance false and could also mask a
    # TypeError raised inside the connector itself.
    responses = list(send_batch(prompts, seeds=seeds))
    if len(responses) != len(prompts):
        raise RuntimeError(
            f"Model backend returned {len(responses)} responses for {len(prompts)} prompts"
        )
    return responses


def run_games_batched(
    games: Sequence[Any],
    send_batch: Callable[..., Sequence[str | dict[str, Any]]],
    *,
    verbose: bool = False,
    max_parse_retries: int = 3,
    on_round_complete: (
        Callable[[Any, Any | None, list[Any]], None] | None
    ) = None,
) -> list[Any]:
    """Evaluate all active races one round at a time.

    Races with different stochastic horizons simply drop out of subsequent batches.
    Invalid action lines are retried from the unchanged pre-action prompt with a
    deterministic retry seed.
    """
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

        requests: list[tuple[Any, int, str, int]] = []
        prompts_by_game: dict[str, list[str]] = {}
        for game in active:
            prompts = game.build_round_prompts()
            prompts_by_game[game.game_id] = prompts
            for player_index, prompt in enumerate(prompts):
                requests.append(
                    (
                        game,
                        player_index,
                        prompt,
                        game.sampling_seed(player_index, game.current_round, 0),
                    )
                )

        responses = _send(
            send_batch,
            [request[2] for request in requests],
            [request[3] for request in requests],
        )
        retry_counts = [0 for _ in requests]
        attempt_histories = [
            [
                _attempt_record(
                    response,
                    attempt=0,
                    sampling_seed=requests[index][3],
                    sampling_seed_applied=bool(
                        requests[index][0].config.sampling_seed_applied
                    ),
                )
            ]
            for index, response in enumerate(responses)
        ]

        for retry in range(1, max_parse_retries + 1):
            failed_indices = [
                index
                for index, response in enumerate(responses)
                if parse_action(response)[1]
            ]
            if not failed_indices:
                break
            retry_prompts = [requests[index][2] for index in failed_indices]
            retry_seeds = [
                requests[index][0].sampling_seed(
                    requests[index][1],
                    requests[index][0].current_round,
                    retry,
                )
                for index in failed_indices
            ]
            retry_responses = _send(send_batch, retry_prompts, retry_seeds)
            for index, seed, response in zip(
                failed_indices,
                retry_seeds,
                retry_responses,
            ):
                responses[index] = response
                retry_counts[index] = retry
                attempt_histories[index].append(
                    _attempt_record(
                        response,
                        attempt=retry,
                        sampling_seed=seed,
                        sampling_seed_applied=bool(
                            requests[index][0].config.sampling_seed_applied
                        ),
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
                if verbose:
                    print(
                        f"[done] {game.game_id}: rounds={result.n_rounds} "
                        f"winner={result.winner or 'tie'} setbacks={result.setbacks}"
                    )
    return results
