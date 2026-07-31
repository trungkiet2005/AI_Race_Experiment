"""Expected payoffs of the reduced four-strategy AI Race.

Two routes to the same quantity, kept apart on purpose:

* **Closed form**, valid only when both seats play an unconditional strategy. The
  race payoff is then affine in the realised horizon ``W`` — the stage payoff is
  ``W * pi`` and the Unsafe fraction ``n_U / W`` is a constant 0 or 1 — so the
  expectation over ``W`` reduces to substituting ``E[W]``.
* **Monte Carlo**, for every pair involving a conditional strategy. There the
  Unsafe fraction depends on the realised trajectory, the payoff is not affine in
  ``W``, and substituting ``E[W]`` would be wrong rather than approximate. This is
  the route the source paper uses, at 10^4 replications per ordered pair.

* **Exact enumeration** over the horizon distribution, which supersedes Monte Carlo
  for every pair. All four reduced strategies are deterministic, so once ``W`` is
  fixed the whole race is fixed; the only randomness left is ``W`` itself, and its
  distribution is a truncated geometric with fewer than a hundred atoms. Summing
  over them costs less than sampling and has no sampling error at all.

  That difference matters downstream. ``Pi(CAS, AU)`` and ``Pi(AU, AU)`` are the
  *same number* — CAS facing an always-Unsafe opponent plays Unsafe in every round —
  and the paper's structural claim is that the four AU/CAS profiles are equilibria
  together or not at all. Under Monte Carlo those cells differ by sampling noise of
  order 0.04, which is enough to drop two of the four profiles from an exhaustive
  best-response search. Enumeration makes the identity hold exactly, so the Nash
  and evolutionary results below need no equilibrium-detection tolerance.

All routes take the expectation over the terminal risk lottery **analytically**
(``x (1 - p_r)``) instead of drawing a setback. Drawing it would add variance for
nothing and, worse, would stop the sampled route from converging to the closed
form — which is the cross-check that keeps the routes honest.

The mechanism itself is never reimplemented here: round scoring, race outcomes,
and effective private risk all come from :mod:`ai_race.engine.scoring`.
"""
from __future__ import annotations

import random
from typing import Iterable, Sequence

from ..engine.scoring import effective_private_risk, joint_round_outcome, race_outcomes
from ..engine.state import Action, GameConfig
from ..engine.strategies import CanonicalStrategy, strategy_action

#: The two strategies whose payoff is affine in the horizon.
UNCONDITIONAL_STRATEGIES = (CanonicalStrategy.AS, CanonicalStrategy.AU)

#: Strategy order used for every emitted matrix, so rows stay comparable.
STRATEGY_ORDER: tuple[str, ...] = ("AS", "AU", "CS", "CAS")

# Disjoint from the engine's horizon (17) and setback (29) streams: a theory
# simulation must never consume draws that a recorded race would have used.
_MATCHUP_STREAM = 41


def expected_horizon(config: GameConfig) -> float:
    """``E[W] = min_rounds + (1 - p) / p``.

    The stop lottery only starts once the minimum is reached, so the number of
    extra rounds is geometric with success probability ``stop_probability``.
    """

    stop = float(config.stop_probability)
    return float(config.min_rounds) + (1.0 - stop) / stop


def _as_strategy(strategy: CanonicalStrategy | str) -> CanonicalStrategy:
    return CanonicalStrategy(strategy)


def unconditional_expected_payoff(
    config: GameConfig,
    own: CanonicalStrategy | str,
    opponent: CanonicalStrategy | str,
) -> float:
    """Closed-form expected payoff, for AS/AU pairs only.

    Raises for any conditional strategy. Returning a number there would be the
    easiest mistake in this module to miss: the formula would evaluate happily and
    be quietly wrong, because a conditional strategy's Unsafe fraction depends on
    the realised horizon and the payoff stops being affine in ``W``.
    """

    own_strategy = _as_strategy(own)
    opponent_strategy = _as_strategy(opponent)
    for strategy in (own_strategy, opponent_strategy):
        if strategy not in UNCONDITIONAL_STRATEGIES:
            raise ValueError(
                f"{strategy.value} is conditional, so its payoff is not affine in "
                "the horizon and has no closed form; use simulate_matchup instead"
            )

    own_action = (
        Action.UNSAFE if own_strategy is CanonicalStrategy.AU else Action.SAFE
    )
    opponent_action = (
        Action.UNSAFE if opponent_strategy is CanonicalStrategy.AU else Action.SAFE
    )
    horizon = expected_horizon(config)

    stage_payoff = horizon * config.stage_payoff(own_action, opponent_action)
    progress = [
        horizon * config.progress_for(own_action),
        horizon * config.progress_for(opponent_action),
    ]
    outcome = race_outcomes(progress)[0]
    prize = _prize_for(config, outcome)

    # The Unsafe fraction is 0 or 1 for an unconditional strategy, so the
    # effective risk does not depend on W and the same expression works for any
    # horizon. Passing rounds_played=1 makes that explicit.
    unsafe_rounds = 1 if own_action is Action.UNSAFE else 0
    risk = effective_private_risk(config.max_private_risk, unsafe_rounds, 1)
    applied_risk = risk if outcome in {"winner", "tie"} else 0.0
    return float((1.0 - applied_risk) * (stage_payoff + prize))


def _prize_for(config: GameConfig, outcome: str) -> float:
    if outcome == "tie":
        return float(config.race_prize) / 2.0
    return float(config.race_prize) if outcome == "winner" else 0.0


def sample_horizon(config: GameConfig, rng: random.Random) -> int:
    """Draw one realised horizon the way the engine does.

    Mirrors ``AIRaceGame._should_stop_after_round``: no stop lottery before the
    minimum, one Bernoulli draw per round afterwards, and a hard safety cap.
    """

    round_number = int(config.min_rounds)
    while round_number < int(config.max_rounds_safety_cap):
        if rng.random() < float(config.stop_probability):
            return round_number
        round_number += 1
    return round_number


def _matchup_seed(config: GameConfig, own: str, opponent: str, seed: int | None) -> int:
    """Derive a default seed from the configuration without Python's salted hash."""

    if seed is not None:
        return int(seed)
    label = f"{config.name}|{own}|{opponent}"
    digest = 0
    for character in label:
        digest = (digest * 131 + ord(character)) % 2_147_483_647
    return (digest * 1_000_003 + _MATCHUP_STREAM * 97_409 + 11_729) % 2_147_483_647


def simulate_matchup(
    config: GameConfig,
    own: CanonicalStrategy | str,
    opponent: CanonicalStrategy | str,
    *,
    replications: int = 10_000,
    seed: int | None = None,
) -> float:
    """Mean expected payoff of ``own`` against ``opponent`` over sampled horizons.

    Both seats are advanced round by round through
    :func:`~ai_race.engine.strategies.strategy_action`. ``strategy_trajectory``
    would be wrong here: it takes a *fixed* opponent action sequence, which is only
    correct when the opponent is unconditional. With CS facing CAS each side reacts
    to the other's previous move, so the two histories have to be built together.

    The terminal risk lottery is taken in expectation rather than drawn, so this
    converges to :func:`unconditional_expected_payoff` on the AS/AU pairs.
    """

    own_strategy = _as_strategy(own)
    opponent_strategy = _as_strategy(opponent)
    if replications < 1:
        raise ValueError("replications must be positive")
    rng = random.Random(
        _matchup_seed(config, own_strategy.value, opponent_strategy.value, seed)
    )

    total = 0.0
    for _ in range(replications):
        horizon = sample_horizon(config, rng)
        total += _expected_payoff_for_horizon(
            config,
            own_strategy,
            opponent_strategy,
            horizon,
        )
    return total / float(replications)


def _expected_payoff_for_horizon(
    config: GameConfig,
    own: CanonicalStrategy,
    opponent: CanonicalStrategy,
    horizon: int,
) -> float:
    """Own expected payoff for one realised horizon, risk taken analytically."""

    own_history: list[Action] = []
    opponent_history: list[Action] = []
    stage_payoff = 0.0
    progress = [0.0, 0.0]

    for round_number in range(1, horizon + 1):
        # Both actions are chosen from the pre-round histories, which is what makes
        # the round simultaneous rather than sequential.
        own_action = strategy_action(own, round_number, opponent_history)
        opponent_action = strategy_action(opponent, round_number, own_history)
        payoffs, increments = joint_round_outcome(config, [own_action, opponent_action])
        stage_payoff += payoffs[0]
        progress[0] += increments[0]
        progress[1] += increments[1]
        own_history.append(own_action)
        opponent_history.append(opponent_action)

    outcome = race_outcomes(progress)[0]
    prize = _prize_for(config, outcome)
    unsafe_count = sum(1 for action in own_history if action is Action.UNSAFE)
    risk = effective_private_risk(config.max_private_risk, unsafe_count, horizon)
    applied_risk = risk if outcome in {"winner", "tie"} else 0.0
    return float((1.0 - applied_risk) * (stage_payoff + prize))


def horizon_distribution(config: GameConfig) -> list[tuple[int, float]]:
    """The exact probability mass function of the realised horizon.

    ``P(W = min_rounds + k) = (1 - p)^k * p`` up to the safety cap, which collects
    the whole remaining tail. The cap is part of the mechanism, so it belongs in the
    distribution rather than being ignored for tidiness — although with the paper's
    parameters its mass is around 1e-9 and it changes nothing that is reported.
    """

    stop = float(config.stop_probability)
    minimum = int(config.min_rounds)
    cap = int(config.max_rounds_safety_cap)
    atoms: list[tuple[int, float]] = []
    remaining = 1.0
    for horizon in range(minimum, cap):
        mass = remaining * stop
        atoms.append((horizon, mass))
        remaining -= mass
    atoms.append((cap, remaining))
    return atoms


def exact_expected_payoff(
    config: GameConfig,
    own: CanonicalStrategy | str,
    opponent: CanonicalStrategy | str,
) -> float:
    """Expected payoff by summing over every possible horizon.

    Valid for all sixteen ordered pairs. Both strategies are deterministic given
    ``W``, and the risk lottery is already taken in expectation, so there is nothing
    left to sample.
    """

    own_strategy = _as_strategy(own)
    opponent_strategy = _as_strategy(opponent)
    return float(
        sum(
            mass
            * _expected_payoff_for_horizon(
                config,
                own_strategy,
                opponent_strategy,
                horizon,
            )
            for horizon, mass in horizon_distribution(config)
            if mass > 0.0
        )
    )


def self_play_unsafe_frequency(
    config: GameConfig,
    strategy: CanonicalStrategy | str,
) -> float:
    """``E[n_U / W]`` for a population where everyone plays ``strategy``.

    Measured rather than assumed. CS and CAS have no intrinsic Unsafe rate — it
    depends entirely on the opponent — and only self-play makes them well defined,
    which is exactly the situation the small-mutation limit puts them in.
    """

    own = _as_strategy(strategy)
    total = 0.0
    for horizon, mass in horizon_distribution(config):
        if mass <= 0.0:
            continue
        own_history: list[Action] = []
        opponent_history: list[Action] = []
        for round_number in range(1, horizon + 1):
            own_action = strategy_action(own, round_number, opponent_history)
            opponent_action = strategy_action(own, round_number, own_history)
            own_history.append(own_action)
            opponent_history.append(opponent_action)
        unsafe = sum(1 for action in own_history if action is Action.UNSAFE)
        total += mass * (unsafe / horizon)
    return float(total)


def expected_payoff_matrix(
    config: GameConfig,
    *,
    method: str = "exact",
    replications: int = 10_000,
    seed: int | None = None,
    strategies: Sequence[str] | None = None,
) -> dict[tuple[str, str], float]:
    """Expected payoff to the row strategy against the column strategy.

    ``method="exact"`` enumerates every cell over the horizon distribution and is
    the default, because it is exactly reproducible and preserves the identities
    between cells that the equilibrium analysis depends on.

    ``method="monte_carlo"`` reproduces the paper's construction: closed form on the
    four AS/AU pairs, sampling on the other twelve. Use it to cross-check the exact
    route, not to feed the equilibrium search — two cells that are equal in the game
    will differ by sampling noise, and an exhaustive best-response search reads that
    noise as a strict preference.
    """

    names = tuple(strategies) if strategies is not None else STRATEGY_ORDER
    return {
        (own, opponent): expected_payoff(
            config,
            own,
            opponent,
            method=method,
            replications=replications,
            seed=seed,
        )
        for own in names
        for opponent in names
    }


def expected_payoff(
    config: GameConfig,
    own: CanonicalStrategy | str,
    opponent: CanonicalStrategy | str,
    *,
    method: str = "exact",
    replications: int = 10_000,
    seed: int | None = None,
) -> float:
    """One matrix cell by the requested route."""

    route = payoff_method(own, opponent, method=method)
    if route == "exact_enumeration":
        return exact_expected_payoff(config, own, opponent)
    if route == "closed_form":
        return unconditional_expected_payoff(config, own, opponent)
    return simulate_matchup(
        config,
        own,
        opponent,
        replications=replications,
        seed=seed,
    )


def payoff_method(
    own: CanonicalStrategy | str,
    opponent: CanonicalStrategy | str,
    *,
    method: str = "exact",
) -> str:
    """Which route computes this ordered pair, given the requested method."""

    if method == "exact":
        return "exact_enumeration"
    if method != "monte_carlo":
        raise ValueError(f"unknown payoff method {method!r}")
    unconditional = all(
        _as_strategy(strategy) in UNCONDITIONAL_STRATEGIES
        for strategy in (own, opponent)
    )
    return "closed_form" if unconditional else "monte_carlo"


def matrix_to_rows(
    matrix: dict[tuple[str, str], float],
    *,
    method: str = "exact",
    strategies: Iterable[str] | None = None,
) -> list[dict[str, object]]:
    """Flatten a payoff matrix into records, tagging how each cell was obtained."""

    names = tuple(strategies) if strategies is not None else STRATEGY_ORDER
    return [
        {
            "own_strategy": own,
            "opponent_strategy": opponent,
            "payoff": matrix[(own, opponent)],
            "method": payoff_method(own, opponent, method=method),
        }
        for own in names
        for opponent in names
    ]
