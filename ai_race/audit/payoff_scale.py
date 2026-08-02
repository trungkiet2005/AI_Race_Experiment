"""Payoff-scale treatment helpers for a utility-invariance diagnostic.

Multiplying every stage payoff and the terminal prize by the same positive
factor leaves progress, risk, setback events, action rankings, and normalized
final utility unchanged.  The treatment therefore tests numerical presentation
sensitivity without changing the strategic game.
"""
from __future__ import annotations

import copy
from decimal import Decimal
from typing import Iterable

from ai_race.engine.game import AIRaceGame


PAYOFF_SCALE_PROTOCOL = "ai-race-payoff-scale-v1"
PAYOFF_SCALES = (0.1, 1.0, 10.0, 100.0)


def payoff_scale_id(scale: float) -> str:
    if scale <= 0:
        raise ValueError("payoff scale must be positive")
    return "scale_" + format(Decimal(str(scale)).normalize(), "f").replace(".", "p")


def scale_game(game: AIRaceGame, scale: float) -> AIRaceGame:
    """Clone one race with only payoff-valued fields scaled."""
    if scale <= 0:
        raise ValueError("payoff scale must be positive")
    config = copy.deepcopy(game.config)
    config.payoff_safe_safe *= scale
    config.payoff_safe_unsafe *= scale
    config.payoff_unsafe_safe *= scale
    config.payoff_unsafe_unsafe *= scale
    config.race_prize *= scale
    scale_name = payoff_scale_id(scale)
    config.prompt_version = f"{PAYOFF_SCALE_PROTOCOL}:{scale_name}"
    return AIRaceGame(
        config,
        list(game.agents),
        template=game.template,
        game_id=f"{game.game_id}__payoff-{scale_name}",
        seed=game.seed,
        rep=game.rep,
    )


def scale_games(games: Iterable[AIRaceGame], scale: float) -> list[AIRaceGame]:
    return [scale_game(game, scale) for game in games]


def payoff_scale_signature(game: AIRaceGame) -> dict[str, float | int | str]:
    """Return fields that must match after normalizing payoff units."""
    config = game.config
    return {
        "name": config.name,
        "rep": game.rep,
        "game_seed": game.seed,
        "safe_progress": config.safe_progress,
        "unsafe_progress": config.unsafe_progress,
        "min_rounds": config.min_rounds,
        "stop_probability": config.stop_probability,
        "max_rounds_safety_cap": config.max_rounds_safety_cap,
        "max_private_risk": config.max_private_risk,
        "history_mode": config.history_mode,
    }
