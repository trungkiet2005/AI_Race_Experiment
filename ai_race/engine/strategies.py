"""Canonical reduced strategies from the source AI Race paper."""
from __future__ import annotations

from enum import Enum
from typing import Sequence

from .state import Action


class CanonicalStrategy(str, Enum):
    AS = "AS"
    AU = "AU"
    CS = "CS"
    CAS = "CAS"


def strategy_action(
    strategy: CanonicalStrategy | str,
    round_number: int,
    opponent_history: Sequence[Action | str],
) -> Action:
    strategy = CanonicalStrategy(strategy)
    if round_number < 1:
        raise ValueError("round_number must be positive")
    if strategy is CanonicalStrategy.AS:
        return Action.SAFE
    if strategy is CanonicalStrategy.AU:
        return Action.UNSAFE
    if round_number == 1:
        return Action.SAFE if strategy is CanonicalStrategy.CS else Action.UNSAFE
    if len(opponent_history) < round_number - 1:
        raise ValueError("opponent_history does not contain the previous round")
    return Action.coerce(opponent_history[round_number - 2])


def strategy_trajectory(
    strategy: CanonicalStrategy | str,
    opponent_actions: Sequence[Action | str],
) -> list[Action]:
    """Generate one action per opponent action for a fixed realised horizon."""
    generated: list[Action] = []
    history: list[Action] = []
    for round_number, opponent_action in enumerate(opponent_actions, start=1):
        generated.append(strategy_action(strategy, round_number, history))
        history.append(Action.coerce(opponent_action))
    return generated

