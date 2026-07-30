"""Simultaneous-round prompt helper and strict action parser."""
from __future__ import annotations

import re
from typing import Any

from .prompt import build_prompt
from .state import Action

_ACTION_LINE_RE = re.compile(
    r"^\s*ACTION\s*:\s*(SAFE|UNSAFE)\s*$",
    flags=re.IGNORECASE | re.MULTILINE,
)
def response_text(response: str | dict[str, Any]) -> str:
    if isinstance(response, dict):
        return str(response.get("text", ""))
    return str(response or "")


def parse_action(response: str | dict[str, Any]) -> tuple[Action, bool]:
    """Parse an action-only response containing exactly one formatted line.

    The parser deliberately does not rescue an invalid formatted answer by scanning
    prose. A missing or invalid final line falls back to Safe and is logged as a
    protocol failure; the batch runner may retry before applying the round.
    """
    text = response_text(response)
    if not text:
        return Action.SAFE, True
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if not lines:
        return Action.SAFE, True
    final_match = _ACTION_LINE_RE.fullmatch(lines[0])
    if len(lines) != 1 or final_match is None:
        return Action.SAFE, True
    try:
        return Action.coerce(final_match.group(1)), False
    except ValueError:
        return Action.SAFE, True


def extract_reasoning(response: str | dict[str, Any]) -> str:
    text = response_text(response)
    if not text:
        return ""
    lines = [
        line
        for line in text.splitlines()
        if not re.match(r"^\s*ACTION\s*[:=]", line, flags=re.IGNORECASE)
    ]
    return "\n".join(lines).strip()


class AIRaceRound:
    """Build both player prompts from one immutable pre-action snapshot."""

    def __init__(self, game: Any, round_number: int):
        self.game = game
        self.round_number = int(round_number)

    def build_prompts(self) -> list[str]:
        config = self.game.config
        prompts: list[str] = []
        names = [agent.name for agent in self.game.agents]
        for player_index, agent in enumerate(self.game.agents):
            prompts.append(
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
            )
        return prompts
