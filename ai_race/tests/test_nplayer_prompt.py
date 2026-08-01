"""Prompt-rendering tests for the N-player engine."""
from __future__ import annotations

from ai_race.engine.agent import RaceAgent
from ai_race.engine_nplayer.round import NPlayerAIRaceRound
from ai_race.engine_nplayer.runner import NPLAYER_PROMPTS_DIR
from ai_race.engine_nplayer.state import NPlayerGameConfig


class _FakeGame:
    def __init__(self, config, agents):
        self.config = config
        self.agents = agents
        self.template = (
            NPLAYER_PROMPTS_DIR / "ai_race_nplayer_en.txt"
        ).read_text(encoding="utf-8")
        self.history: list[dict] = []
        self.progress = [0.0] * config.n_players
        self.stage_payoffs = [0.0] * config.n_players
        self.unsafe_counts = [0] * config.n_players


def _game(n_players: int = 3) -> _FakeGame:
    config = NPlayerGameConfig(name="test", n_players=n_players)
    agents = [RaceAgent(name=f"Company_{i + 1}") for i in range(n_players)]
    return _FakeGame(config, agents)


def test_prompt_lists_every_co_player_by_name():
    game = _game(3)
    prompts = NPlayerAIRaceRound(game, 1).build_prompts()
    assert len(prompts) == 3
    seat_0 = prompts[0]
    assert "Company_1" in seat_0
    assert "Company_2" in seat_0
    assert "Company_3" in seat_0
    # A seat's own name should not appear in its "others" bullet list section.
    others_section = seat_0.split("This is the state of the race")[1]
    assert "Company_1:" not in others_section


def test_prompt_has_no_leftover_placeholders_or_optional_block_markup():
    game = _game(4)
    prompts = NPlayerAIRaceRound(game, 1).build_prompts()
    for prompt in prompts:
        assert "{" not in prompt and "}" not in prompt
        # apply_optional_blocks strips the surrounding "name: [ ... ]" markup
        # entirely, so a correctly rendered prompt has no brackets left at all.
        assert "[" not in prompt and "]" not in prompt


def test_prompt_contains_exactly_one_action_line_pair():
    game = _game(3)
    prompt = NPlayerAIRaceRound(game, 1).build_prompts()[0]
    assert "ACTION: SAFE" in prompt
    assert "ACTION: UNSAFE" in prompt


def test_payoff_table_has_one_row_per_k_from_0_to_n():
    game = _game(3)
    prompt = NPlayerAIRaceRound(game, 1).build_prompts()[0]
    assert "If 0 of 3 companies choose SAFE" in prompt
    assert "If 1 of 3 companies choose SAFE" in prompt
    assert "If 2 of 3 companies choose SAFE" in prompt
    assert "If all 3 companies choose SAFE" in prompt


def test_persona_reaches_the_rendered_prompt():
    config = NPlayerGameConfig(name="test", n_players=3)
    agents = [
        RaceAgent(name="Company_1", persona_text="a cautious executive"),
        RaceAgent(name="Company_2"),
        RaceAgent(name="Company_3"),
    ]
    game = _FakeGame(config, agents)
    prompts = NPlayerAIRaceRound(game, 1).build_prompts()
    assert "a cautious executive" in prompts[0]
    assert "a cautious executive" not in prompts[1]
