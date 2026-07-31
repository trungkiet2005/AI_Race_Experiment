"""Agent metadata independent of any model backend."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RaceAgent:
    """One company seat in a two-player race."""

    name: str
    persona_text: str = ""
    persona_probability: float = 100.0
    # The label for this seat's persona (for example "adversarial"). Recorded on
    # every decision so an asymmetric condition can be split by seat without
    # re-reading the agents configuration.
    persona_role: str = ""

