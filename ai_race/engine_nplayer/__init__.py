"""N-player generalisation of the AI Race engine.

Kept entirely separate from :mod:`ai_race.engine` (the paper-faithful,
hash-pinned two-player mechanism) so that extending the mechanism to N >= 2
players never touches the two-player invariant CLAUDE.md documents. See
``ai_race/engine_nplayer/README.md`` for the mechanism this module implements
and what it deliberately leaves out.
"""
from __future__ import annotations
