"""Canonical reference strategies for the N-player AI Race.

Generalises ``ai_race.engine.strategies``' AS/AU/CS to a group of co-players.
Only AS/AU/CS are ported: the reference paper's N-player appendix ("N-player
AI Race Definition") only defines these three for the group game -- CAS is a
two-player-only strategy from the main text and has no stated N-player
reduction, so it is intentionally left out here rather than guessed at.
"""
from __future__ import annotations

from enum import Enum
from typing import Sequence

from .state import Action


class CanonicalStrategy(str, Enum):
    AS = "AS"
    AU = "AU"
    CS = "CS"


def strategy_action(
    strategy: CanonicalStrategy | str,
    round_number: int,
    others_history: Sequence[Sequence[Action | str]],
) -> Action:
    """One player's action for ``round_number``, given every co-player's history so far.

    ``others_history`` holds one sequence per co-player, each containing their
    actions for rounds ``1..round_number - 1`` (i.e. it has ``round_number -
    1`` entries per co-player once ``round_number > 1``).
    """
    strategy = CanonicalStrategy(strategy)
    if round_number < 1:
        raise ValueError("round_number must be positive")
    if strategy is CanonicalStrategy.AS:
        return Action.SAFE
    if strategy is CanonicalStrategy.AU:
        return Action.UNSAFE
    # CS: Safe in round 1; afterwards Safe only if every co-player played Safe
    # in the immediately preceding round, evaluated fresh each round (not
    # "sticky" -- a single Unsafe round from any co-player only costs CS the
    # next round, per the paper's "plays SAFE if everyone in the group plays
    # SAFE in the previous round and plays UNSAFE otherwise").
    if round_number == 1:
        return Action.SAFE
    for history in others_history:
        if len(history) < round_number - 1:
            raise ValueError("others_history does not contain the previous round")
    previous_actions = [
        Action.coerce(history[round_number - 2]) for history in others_history
    ]
    all_safe = all(action is Action.SAFE for action in previous_actions)
    return Action.SAFE if all_safe else Action.UNSAFE


def strategy_trajectory(
    strategy: CanonicalStrategy | str,
    others_actions: Sequence[Sequence[Action | str]],
) -> list[Action]:
    """Generate one action per round for a fixed, realised set of co-player actions.

    ``others_actions`` holds one sequence per co-player, each of the same
    length (the realised horizon); ``others_actions[j][r]`` is co-player
    ``j``'s action in round ``r + 1``.
    """
    if others_actions and len({len(actions) for actions in others_actions}) != 1:
        raise ValueError("Every co-player's action sequence must have the same length")
    horizon = len(others_actions[0]) if others_actions else 0
    generated: list[Action] = []
    histories: list[list[Action]] = [[] for _ in others_actions]
    for round_number in range(1, horizon + 1):
        generated.append(strategy_action(strategy, round_number, histories))
        for co_player_index, actions in enumerate(others_actions):
            histories[co_player_index].append(Action.coerce(actions[round_number - 1]))
    return generated
