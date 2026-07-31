"""Agent metadata independent of any model backend."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RaceAgent:
    """One company seat in a two-player race."""

    name: str
    persona_text: str = ""
    persona_probability: float = 100.0

