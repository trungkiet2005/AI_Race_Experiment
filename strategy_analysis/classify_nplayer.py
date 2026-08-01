"""Nearest-strategy classification for N-player AI Race trajectories.

N-player counterpart of :mod:`strategy_analysis.classify`. The candidate
strategy set matches ``ai_race.engine_nplayer.strategies.CanonicalStrategy``:
**AS, AU, CS only** -- there is no N-player CAS (the paper's Appendix B only
defines AS/AU/CS for the group game; see that module's docstring) and no
N-player counterpart is invented here.

The only behavioural difference from the two-player classifier is CS's
predicted action from round 2 onward: instead of "copy the single opponent's
previous action", it is "SAFE iff every co-player played SAFE in the
previous round, else UNSAFE" -- exactly
``ai_race.engine_nplayer.strategies.strategy_action``'s rule, reimplemented
here in the vectorised one-shot style ``predict_strategy`` already uses in
the two-player module (shift-by-one), rather than by importing the
round-by-round engine function.

Binary encoding, tie retention, and the exploratory
``BEHIND_UNSAFE_EXPLORATORY`` rule are reused unchanged from
:mod:`strategy_analysis.classify` (progress gap is computed the same way
upstream -- own minus the *max* of co-players -- so "negative gap = behind"
still means the same thing with more than one co-player).
"""
from __future__ import annotations

from typing import Any, Sequence

from strategy_analysis.classify import (
    EXPLORATORY_BEHIND_STRATEGY,
    ClassificationResult,
    StrategyMatch,
    _normalise_gaps,
    normalise_actions,
)

#: No CAS: the N-player engine and its source paper only define these three.
CANONICAL_STRATEGIES_NPLAYER = ("AS", "AU", "CS")


def predict_strategy_nplayer(
    strategy: str,
    others_actions: Sequence[Sequence[Any]],
    *,
    progress_gaps_before: Sequence[Any] | None = None,
) -> tuple[int, ...]:
    """Predict a strategy's actions given every co-player's observed trajectory.

    ``others_actions[j]`` is co-player ``j``'s Safe/Unsafe sequence; every
    co-player must share the same horizon. ``progress_gaps_before`` is
    required only for ``BEHIND_UNSAFE_EXPLORATORY``, and (matching how the
    N-player engine records it) should already be "own minus the leading
    co-player's progress".
    """

    others = tuple(
        normalise_actions(sequence, field=f"others_actions[{index}]")
        for index, sequence in enumerate(others_actions)
    )
    if not others:
        raise ValueError("others_actions must contain at least one co-player")
    lengths = {len(sequence) for sequence in others}
    if len(lengths) != 1:
        raise ValueError("every co-player's action sequence must have the same length")
    horizon = lengths.pop()
    if horizon == 0:
        raise ValueError("a trajectory must contain at least one round")

    if strategy == "AS":
        return (0,) * horizon
    if strategy == "AU":
        return (1,) * horizon
    if strategy == "CS":
        # Round 1: SAFE. Round r>=2: SAFE iff every co-player was SAFE at r-1.
        tail = [
            0 if all(sequence[round_index] == 0 for sequence in others) else 1
            for round_index in range(horizon - 1)
        ]
        return (0, *tail)
    if strategy == EXPLORATORY_BEHIND_STRATEGY:
        if progress_gaps_before is None:
            raise ValueError(
                f"{EXPLORATORY_BEHIND_STRATEGY} requires progress_gaps_before"
            )
        gaps = _normalise_gaps(progress_gaps_before, horizon=horizon)
        return tuple(1 if gap < 0 else 0 for gap in gaps)
    valid = (*CANONICAL_STRATEGIES_NPLAYER, EXPLORATORY_BEHIND_STRATEGY)
    raise ValueError(f"unknown strategy {strategy!r}; expected one of {valid}")


def classify_trajectory_nplayer(
    own_actions: Sequence[Any],
    others_actions: Sequence[Sequence[Any]],
    *,
    include_exploratory_behind: bool = False,
    progress_gaps_before: Sequence[Any] | None = None,
) -> ClassificationResult:
    """Classify one player's variable-horizon N-player trajectory.

    Same minimum-mismatch-rate rule and tie retention as
    :func:`strategy_analysis.classify.classify_trajectory`.
    """

    own = normalise_actions(own_actions, field="own_actions")
    if not own:
        raise ValueError("a trajectory must contain at least one round")
    for index, sequence in enumerate(others_actions):
        other = normalise_actions(sequence, field=f"others_actions[{index}]")
        if len(other) != len(own):
            raise ValueError(
                "own_actions and every others_actions entry must have the same "
                f"length ({len(own)} != {len(other)} at index {index})"
            )

    candidates = list(CANONICAL_STRATEGIES_NPLAYER)
    if include_exploratory_behind:
        if progress_gaps_before is None:
            raise ValueError(
                "progress_gaps_before is required when "
                "include_exploratory_behind=True"
            )
        candidates.append(EXPLORATORY_BEHIND_STRATEGY)

    matches: list[StrategyMatch] = []
    for strategy in candidates:
        predicted = predict_strategy_nplayer(
            strategy,
            others_actions,
            progress_gaps_before=progress_gaps_before,
        )
        mismatches = sum(observed != expected for observed, expected in zip(own, predicted))
        matches.append(
            StrategyMatch(
                strategy=strategy,
                predicted_actions=predicted,
                mismatches=mismatches,
                mismatch_rate=mismatches / len(own),
                canonical=strategy in CANONICAL_STRATEGIES_NPLAYER,
            )
        )

    minimum = min(match.mismatches for match in matches)
    best = tuple(match.strategy for match in matches if match.mismatches == minimum)
    return ClassificationResult(
        horizon=len(own),
        best_strategies=best,
        matches=tuple(matches),
    )
