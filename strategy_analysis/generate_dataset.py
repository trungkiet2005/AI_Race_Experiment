"""Synthetic AS/AU/CS/CAS trajectory generator for classifier evaluation.

Generates labelled Safe/Unsafe trajectories for every ordered pairing of the
paper's four canonical strategies (Fernandez Domingos and Han, 2026), reusing
``ai_race.engine.strategies`` so the generated actions can never drift from the
engine's own definition of AS/AU/CS/CAS.

The realised horizon per race follows the same distribution as the engine
(``ai_race/engine/game.py``): a minimum of ``min_rounds``, then an independent
Bernoulli(``stop_probability``) draw after every subsequent round, capped at
``max_rounds_safety_cap``. This is a standalone simulation for dataset
generation only; it does not reuse the engine's seed streams and must not be
mistaken for engine output.

Optional per-round execution noise flips a player's realised action before it
is recorded and before the opponent can copy it next round, mirroring the
noise convention used for the IPD strategy-classifier datasets in the sibling
ClusteringResearch project (D:/AI_RESEARCH/ClusteringResearch).

Output is JSONL with one record per (race, player) — the schema
``strategy_analysis.classify.classify_trajectory`` already expects
(``own_actions``, ``opponent_actions``), plus a ``true_strategy`` label used
only for offline evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterator

from ai_race.engine.state import Action
from ai_race.engine.strategies import CanonicalStrategy, strategy_action

CANONICAL_STRATEGIES = tuple(s.value for s in CanonicalStrategy)


def sample_horizon(
    rng: random.Random,
    *,
    min_rounds: int,
    stop_probability: float,
    max_rounds_safety_cap: int,
) -> int:
    """Draw one race horizon: min_rounds + Geom(stop_probability), capped."""

    round_number = min_rounds
    while round_number < max_rounds_safety_cap:
        if rng.random() < stop_probability:
            break
        round_number += 1
    return round_number


def simulate_race(
    strategy_a: str,
    strategy_b: str,
    horizon: int,
    *,
    noise_p: float,
    rng: random.Random,
) -> tuple[list[int], list[int]]:
    """Return (actions_a, actions_b), each length ``horizon``, as 0/1 ints."""

    history_a: list[Action] = []
    history_b: list[Action] = []
    for round_number in range(1, horizon + 1):
        action_a = strategy_action(strategy_a, round_number, history_b)
        action_b = strategy_action(strategy_b, round_number, history_a)
        if noise_p > 0.0:
            if rng.random() < noise_p:
                action_a = Action.SAFE if action_a is Action.UNSAFE else Action.UNSAFE
            if rng.random() < noise_p:
                action_b = Action.SAFE if action_b is Action.UNSAFE else Action.UNSAFE
        history_a.append(action_a)
        history_b.append(action_b)
    to_bit = lambda a: 1 if a is Action.UNSAFE else 0
    return [to_bit(a) for a in history_a], [to_bit(a) for a in history_b]


def generate_records(
    *,
    strategies: tuple[str, ...],
    games_per_pair: int,
    noise_levels: tuple[float, ...],
    min_rounds: int,
    stop_probability: float,
    max_rounds_safety_cap: int,
    seed: int,
) -> Iterator[dict]:
    rng = random.Random(seed)
    for noise_p in noise_levels:
        for strategy_a in strategies:
            for strategy_b in strategies:
                for game_index in range(games_per_pair):
                    horizon = sample_horizon(
                        rng,
                        min_rounds=min_rounds,
                        stop_probability=stop_probability,
                        max_rounds_safety_cap=max_rounds_safety_cap,
                    )
                    actions_a, actions_b = simulate_race(
                        strategy_a, strategy_b, horizon, noise_p=noise_p, rng=rng
                    )
                    race_id = f"{strategy_a}v{strategy_b}/noise{noise_p:g}/game-{game_index:05d}"
                    yield {
                        "trajectory_id": f"{race_id}/p0",
                        "true_strategy": strategy_a,
                        "opponent_true_strategy": strategy_b,
                        "own_actions": actions_a,
                        "opponent_actions": actions_b,
                        "horizon": horizon,
                        "noise_p": noise_p,
                    }
                    yield {
                        "trajectory_id": f"{race_id}/p1",
                        "true_strategy": strategy_b,
                        "opponent_true_strategy": strategy_a,
                        "own_actions": actions_b,
                        "opponent_actions": actions_a,
                        "horizon": horizon,
                        "noise_p": noise_p,
                    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSONL output path")
    parser.add_argument(
        "--games-per-pair",
        type=int,
        default=200,
        help="races simulated per ordered (strategy_a, strategy_b) pair, per noise level",
    )
    parser.add_argument(
        "--noise-levels",
        type=str,
        default="0.0,0.05,0.10",
        help="comma-separated per-round execution-noise flip probabilities",
    )
    parser.add_argument("--min-rounds", type=int, default=5)
    parser.add_argument("--stop-probability", type=float, default=0.2)
    parser.add_argument("--max-rounds-safety-cap", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260801)
    args = parser.parse_args(argv)

    noise_levels = tuple(float(x) for x in args.noise_levels.split(","))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for record in generate_records(
            strategies=CANONICAL_STRATEGIES,
            games_per_pair=args.games_per_pair,
            noise_levels=noise_levels,
            min_rounds=args.min_rounds,
            stop_probability=args.stop_probability,
            max_rounds_safety_cap=args.max_rounds_safety_cap,
            seed=args.seed,
        ):
            handle.write(json.dumps(record, sort_keys=True) + "\n")
            n_written += 1

    print(
        f"wrote {n_written} trajectories to {args.output} "
        f"| strategies={CANONICAL_STRATEGIES} | noise_levels={noise_levels} "
        f"| games_per_pair={args.games_per_pair}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
