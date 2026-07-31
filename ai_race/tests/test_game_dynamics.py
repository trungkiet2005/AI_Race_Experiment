"""Deterministic state-transition and hidden-horizon tests."""
from __future__ import annotations

from collections.abc import Callable, Iterable

import pytest

from ai_race.engine.game import AIRaceGame
from ai_race.paths import PROMPTS_DIR


class ControlledRng:
    """Minimal RNG stub that fails if the engine consumes an unexpected draw."""

    def __init__(self, values: Iterable[float]):
        self._values = iter(values)
        self.calls = 0

    def random(self) -> float:
        self.calls += 1
        try:
            return next(self._values)
        except StopIteration as exc:
            raise AssertionError("The game requested an unexpected RNG draw") from exc


def test_round_responses_are_applied_from_one_simultaneous_snapshot(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    game = game_factory()
    prompts = game.build_round_prompts()

    assert "player=Company_A" in prompts[0]
    assert "own_progress=0" in prompts[0]
    assert "opponent_progress=0" in prompts[0]
    assert "stage=0" in prompts[0]
    assert "opponent_stage=0" in prompts[0]
    assert "risk=0" in prompts[0]
    assert "opponent_risk=0" in prompts[0]
    assert "player=Company_B" in prompts[1]
    assert "own_progress=0" in prompts[1]
    assert "opponent_progress=0" in prompts[1]

    result = game.apply_round_responses(
        ["ACTION: UNSAFE", "ACTION: SAFE"],
        prompts=prompts,
    )

    assert result is None
    assert game.history[0]["actions"] == ["unsafe", "safe"]
    assert game.history[0]["payoffs"] == [2.4, 0.6]
    assert game.history[0]["increments"] == [1.5, 1.0]
    assert game.progress == [1.5, 1.0]
    assert game.stage_payoffs == [2.4, 0.6]

    first, second = game.turns
    assert (first.own_progress_before, first.opponent_progress_before) == (0.0, 0.0)
    assert (second.own_progress_before, second.opponent_progress_before) == (0.0, 0.0)
    assert (first.own_progress_after, first.opponent_progress_after) == (1.5, 1.0)
    assert (second.own_progress_after, second.opponent_progress_after) == (1.0, 1.5)
    assert first.prompt == prompts[0]
    assert second.prompt == prompts[1]

    next_prompts = game.build_round_prompts()
    assert "own_progress=1.5" in next_prompts[0]
    assert "opponent_progress=1" in next_prompts[0]
    assert "stage=2.4" in next_prompts[0]
    assert "opponent_stage=0.6" in next_prompts[0]
    assert "risk=60" in next_prompts[0]
    assert "opponent_risk=0" in next_prompts[0]
    assert "own_progress=1" in next_prompts[1]
    assert "opponent_progress=1.5" in next_prompts[1]
    assert "stage=0.6" in next_prompts[1]
    assert "opponent_stage=2.4" in next_prompts[1]
    assert "risk=0" in next_prompts[1]
    assert "opponent_risk=60" in next_prompts[1]


def test_horizon_is_not_sampled_before_round_five_and_is_hidden_from_players(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    template = (PROMPTS_DIR / "ai_race_en.txt").read_text(encoding="utf-8")
    game = game_factory(
        template=template,
        min_rounds=5,
        stop_probability=1.0,
        max_rounds_safety_cap=10,
    )
    horizon_rng = ControlledRng([0.0])
    game._horizon_rng = horizon_rng

    for _ in range(4):
        assert game.apply_round_responses(["ACTION: SAFE", "ACTION: SAFE"]) is None

    assert horizon_rng.calls == 0
    assert game.stop_draws == []
    round_five_prompts = game.build_round_prompts()
    for prompt in round_five_prompts:
        assert "The current round is number 5." in prompt
        assert "This is the state of the race before your current decision:" in prompt
        assert "You do not know the final round in advance." in prompt
        assert "stop_draw" not in prompt

    result = game.apply_round_responses(
        ["ACTION: SAFE", "ACTION: SAFE"],
        prompts=round_five_prompts,
    )
    assert result is not None
    assert result.n_rounds == 5
    assert horizon_rng.calls == 1
    assert result.stop_draws == [0.0]
    assert result.stop_forced is False


def test_safety_cap_forces_termination_when_stochastic_draws_never_stop(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    game = game_factory(
        min_rounds=5,
        stop_probability=0.2,
        max_rounds_safety_cap=6,
    )
    horizon_rng = ControlledRng([0.9, 0.8])
    game._horizon_rng = horizon_rng

    for _ in range(5):
        assert game.apply_round_responses(["ACTION: SAFE", "ACTION: SAFE"]) is None

    result = game.apply_round_responses(["ACTION: SAFE", "ACTION: SAFE"])
    assert result is not None
    assert result.n_rounds == 6
    assert result.stop_forced is True
    assert result.stop_draws == [0.9, 0.8]
    assert horizon_rng.calls == 2


def test_finished_game_rejects_further_prompts_and_responses(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    game = game_factory(
        min_rounds=1,
        stop_probability=1.0,
        max_rounds_safety_cap=2,
    )
    assert game.apply_round_responses(["ACTION: SAFE", "ACTION: SAFE"]) is not None

    with pytest.raises(RuntimeError, match="finished"):
        game.build_round_prompts()
    with pytest.raises(RuntimeError, match="finished"):
        game.apply_round_responses(["ACTION: SAFE", "ACTION: SAFE"])


def test_fairgame_optional_blocks_track_persona_and_hidden_horizon(
    game_factory: Callable[..., AIRaceGame],
) -> None:
    """FAIRGAME ``{name}: [ ... ]`` blocks are unwrapped or deleted, never leaked."""
    template = (PROMPTS_DIR / "ai_race_en.txt").read_text(encoding="utf-8")

    neutral = game_factory(template=template).build_round_prompts()[0]
    assert "{" not in neutral and "}" not in neutral
    # gameLength stays hidden: the agent must not learn the horizon.
    assert "rounds to decide" not in neutral
    assert "You are ." not in neutral

    persona_game = game_factory(template=template)
    persona_game.agents[0].persona_text = "a safety-first laboratory"
    persona_prompt = persona_game.build_round_prompts()[0]
    assert "You are a safety-first laboratory." in persona_prompt
    assert "{" not in persona_prompt

    # The communicate block is dropped whenever agents do not exchange messages.
    assert "Send one short message" not in neutral
    assert "ACTION: UNSAFE" in neutral
