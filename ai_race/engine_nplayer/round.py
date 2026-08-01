"""Simultaneous-round prompt helper for the N-player AI Race.

Response parsing (``parse_action``/``extract_reasoning``/``response_text``)
is reused unchanged from ``ai_race.engine.round`` -- parsing one player's
formatted response never depended on how many other players there were.
"""
from __future__ import annotations

from typing import Any

from .prompt import build_prompt


class NPlayerAIRaceRound:
    """Build every player's prompt from one immutable pre-action snapshot."""

    def __init__(self, game: Any, round_number: int):
        self.game = game
        self.round_number = int(round_number)

    def build_prompts(self) -> list[str]:
        config = self.game.config
        names = [agent.name for agent in self.game.agents]
        return [
            build_prompt(
                self.game.template,
                config,
                player_name=agent.name,
                player_index=player_index,
                player_names=names,
                persona_text=agent.persona_text,
                round_number=self.round_number,
                history=self.game.history,
                progress=self.game.progress,
                stage_payoffs=self.game.stage_payoffs,
                unsafe_counts=self.game.unsafe_counts,
            )
            for player_index, agent in enumerate(self.game.agents)
        ]
