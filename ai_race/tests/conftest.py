"""Shared deterministic fixtures that never initialize a model backend."""
from __future__ import annotations

from collections.abc import Callable

import pytest

from ai_race.engine.agent import RaceAgent
from ai_race.engine.game import AIRaceGame
from ai_race.engine.state import GameConfig


TEST_TEMPLATE = """\
player={player_name}
opponent={opponent_name}
round={round_number}
own_progress={own_progress}
opponent_progress={opponent_progress}
gap={progress_gap}
stage={own_stage_payoff}
opponent_stage={opponent_stage_payoff}
risk={own_private_risk_percent}
opponent_risk={opponent_private_risk_percent}
previous:
{previous_round}
"""


def make_config(**overrides: object) -> GameConfig:
    values: dict[str, object] = {
        "name": "test_ai_race",
        "model": "MockModel",
        "min_rounds": 5,
        "stop_probability": 0.2,
        "max_rounds_safety_cap": 20,
        "max_private_risk": 0.6,
    }
    values.update(overrides)
    return GameConfig(**values)


def make_game(
    *,
    seed: int = 123,
    game_id: str = "test-game",
    template: str = TEST_TEMPLATE,
    **config_overrides: object,
) -> AIRaceGame:
    return AIRaceGame(
        make_config(**config_overrides),
        [RaceAgent("Company_A"), RaceAgent("Company_B")],
        template=template,
        game_id=game_id,
        seed=seed,
    )


@pytest.fixture
def game_factory() -> Callable[..., AIRaceGame]:
    return make_game
