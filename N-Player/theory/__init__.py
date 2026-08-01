"""Theoretical (EGT) analysis of the N-team DSAIR, reproducing Appendix B-E
of Han, Pereira, Santos & Lenaerts (JAIR 2020), "To Regulate or Not".

Separate from ``ai_race.engine_nplayer``, which only *plays* one race between
``n_players`` agents (see its README's "deliberately out of scope" section).
This package fills that gap: closed-form DSAI zone conditions
(:mod:`theory.conditions`), finite-population evolutionary dynamics
(:mod:`theory.population`), and average payoffs / social welfare
(:mod:`theory.welfare`).
"""
from __future__ import annotations
