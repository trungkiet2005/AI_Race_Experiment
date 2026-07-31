#!/usr/bin/env python3
"""Emit the game-theoretic and evolutionary predictions of the AI Race mechanism.

This script reads **no run output**. Everything it writes is a property of the game
defined by the checked-in ``configs/game/*.json`` files, so it produces identical
numbers whether the players were humans, LLMs, or nobody. Keeping it out of
``analyze_ai_race.py`` is the point: a prediction that never saw the data cannot be
mistaken for a fit to it.

Every filename carries a ``theory_`` prefix for the same reason. In particular
``theory_stationary_distribution.csv`` is the *predicted* population composition of
an evolutionary model; ``strategy_summary_player.csv`` from the analyser is a
nearest-neighbour classification of *observed* LLM trajectories. They answer
different questions and must never be read as two estimates of one quantity.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from ai_race.engine.state import GameConfig  # noqa: E402
from ai_race.theory.equilibria import equilibrium_summary  # noqa: E402
from ai_race.theory.evolution import (  # noqa: E402
    MUTATION_REGIME,
    expected_unsafe_frequency,
    small_mutation_stationary,
)
from ai_race.theory.payoffs import (  # noqa: E402
    STRATEGY_ORDER,
    expected_payoff_matrix,
    matrix_to_rows,
    self_play_unsafe_frequency,
)

GAME_CONFIG_DIRECTORY = REPOSITORY_ROOT / "ai_race" / "configs" / "game"
DEFAULT_OUTPUT = REPOSITORY_ROOT / "results" / "derived" / "ai_race_theory"

# The paper's reference point (strong selection, mutation at the neutral scale) and
# its reported best fit to the human data (weak selection, high mutation). Both are
# evaluated here in the small-mutation limit; see the metadata caveat.
PARAMETER_POINTS: tuple[dict[str, Any], ...] = (
    {"label": "reference", "beta": 2.0, "nominal_mu": 0.02},
    {"label": "best_fit", "beta": 0.01, "nominal_mu": 0.05},
)
DEFAULT_POPULATION_SIZE = 100


def load_game_configs() -> list[GameConfig]:
    configs = [
        GameConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(GAME_CONFIG_DIRECTORY.glob("*.json"))
    ]
    if not configs:
        raise FileNotFoundError(f"no game configurations under {GAME_CONFIG_DIRECTORY}")
    return sorted(configs, key=lambda config: config.max_private_risk)


def _write_csv(path: Path, rows: Sequence[dict[str, Any]], columns: Sequence[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def build_payoff_matrix_rows(
    configs: Sequence[GameConfig],
    *,
    method: str,
    replications: int,
    seed: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for config in configs:
        matrix = expected_payoff_matrix(
            config,
            method=method,
            replications=replications,
            seed=seed,
        )
        for record in matrix_to_rows(matrix, method=method):
            rows.append(
                {
                    "max_private_risk": config.max_private_risk,
                    "game_config": config.name,
                    **record,
                    # Replications and seed are meaningless for a route that does
                    # not sample; recording them anyway would suggest the exact
                    # numbers depend on a seed.
                    "replications": replications if record["method"] == "monte_carlo" else 0,
                    "seed": seed if record["method"] == "monte_carlo" else "",
                }
            )
    return rows


def build_equilibrium_rows(configs: Sequence[GameConfig]) -> list[dict[str, Any]]:
    return [equilibrium_summary(config) for config in configs]


def build_stationary_rows(
    configs: Sequence[GameConfig],
    *,
    population_size: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    stationary_rows: list[dict[str, Any]] = []
    unsafe_rows: list[dict[str, Any]] = []
    for config in configs:
        matrix = expected_payoff_matrix(config)
        per_strategy = {
            name: self_play_unsafe_frequency(config, name) for name in STRATEGY_ORDER
        }
        for point in PARAMETER_POINTS:
            stationary = small_mutation_stationary(
                matrix,
                Z=population_size,
                beta=point["beta"],
            )
            for strategy in STRATEGY_ORDER:
                stationary_rows.append(
                    {
                        "max_private_risk": config.max_private_risk,
                        "parameter_point": point["label"],
                        "beta": point["beta"],
                        "mutation_regime": MUTATION_REGIME,
                        "nominal_mu": point["nominal_mu"],
                        "population_size": population_size,
                        "strategy": strategy,
                        "frequency": stationary[strategy],
                        "self_play_unsafe_frequency": per_strategy[strategy],
                    }
                )
            unsafe_rows.append(
                {
                    "max_private_risk": config.max_private_risk,
                    "parameter_point": point["label"],
                    "beta": point["beta"],
                    "mutation_regime": MUTATION_REGIME,
                    "nominal_mu": point["nominal_mu"],
                    "population_size": population_size,
                    "expected_unsafe_frequency": expected_unsafe_frequency(
                        stationary,
                        per_strategy,
                    ),
                }
            )
    return stationary_rows, unsafe_rows


def build_metadata(
    configs: Sequence[GameConfig],
    *,
    population_size: int,
    method: str,
    replications: int,
    seed: int,
) -> dict[str, Any]:
    return {
        "schema_version": "ai-race-theory-v1",
        "reads_experiment_output": False,
        "independence_warning": (
            "Every number here is derived from the game mechanism alone. It is "
            "identical for every LLM, every persona condition, and every run. "
            "Agreement with observed behaviour is a property of the game, not "
            "evidence about a model."
        ),
        "naming_warning": (
            "theory_stationary_distribution.csv is a predicted population "
            "composition under evolutionary dynamics. strategy_summary_player.csv "
            "in the behavioural analysis is a nearest-strategy classification of "
            "observed trajectories. They are not two estimates of one quantity."
        ),
        "game_configs": [
            {
                "name": config.name,
                "max_private_risk": config.max_private_risk,
                "min_rounds": config.min_rounds,
                "stop_probability": config.stop_probability,
                "race_prize": config.race_prize,
                "stage_payoffs": {
                    "safe_safe": config.payoff_safe_safe,
                    "safe_unsafe": config.payoff_safe_unsafe,
                    "unsafe_safe": config.payoff_unsafe_safe,
                    "unsafe_unsafe": config.payoff_unsafe_unsafe,
                },
            }
            for config in configs
        ],
        "payoff_method": method,
        "payoff_method_note": (
            "exact enumerates the horizon distribution and has no sampling error; "
            "monte_carlo reproduces the source paper's construction and is offered "
            "for cross-checking. The equilibrium and evolutionary tables always use "
            "the exact matrix, because sampling noise breaks the exact equality "
            "between the AU and CAS cells that the equilibrium structure rests on."
        ),
        "monte_carlo_replications": replications,
        "monte_carlo_seed": seed,
        "population_size": population_size,
        "parameter_points": list(PARAMETER_POINTS),
        "mutation_regime": MUTATION_REGIME,
        "mutation_regime_caveat": (
            "The stationary distributions are the mu -> 0 limit, in which the "
            "population is monomorphic and the chain reduces to fixation "
            "probabilities between the four strategies. nominal_mu records the "
            "mutation rate of the source paper's parameter point that each row "
            "approximates; it is NOT applied. Two consequences: (a) AU and CAS are "
            "exactly payoff-equivalent against each other, so the limit cannot "
            "reproduce the paper's AU-to-CAS transition near p_r^max = 0.2 and "
            "splits their mass evenly instead; (b) at a finite mutation rate the "
            "distribution spreads into mixed populations, which this limit cannot "
            "represent. Both need the full finite-mutation chain."
        ),
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"output directory (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--payoff-method",
        choices=("exact", "monte_carlo"),
        default="exact",
        help=(
            "how theory_payoff_matrix.csv is computed; the equilibrium and "
            "evolutionary tables always use the exact matrix"
        ),
    )
    parser.add_argument(
        "--replications",
        type=int,
        default=10_000,
        help="Monte Carlo replications per ordered pair (source paper uses 10000)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=260726,
        help="Monte Carlo seed; the experiment base seed by default",
    )
    parser.add_argument(
        "--population-size",
        type=int,
        default=DEFAULT_POPULATION_SIZE,
        help="evolutionary population size Z",
    )
    args = parser.parse_args(argv)

    configs = load_game_configs()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    payoff_rows = build_payoff_matrix_rows(
        configs,
        method=args.payoff_method,
        replications=args.replications,
        seed=args.seed,
    )
    _write_csv(
        output / "theory_payoff_matrix.csv",
        payoff_rows,
        [
            "max_private_risk",
            "game_config",
            "own_strategy",
            "opponent_strategy",
            "payoff",
            "method",
            "replications",
            "seed",
        ],
    )

    equilibrium_rows = build_equilibrium_rows(configs)
    _write_csv(
        output / "theory_equilibria.csv",
        equilibrium_rows,
        list(equilibrium_rows[0].keys()),
    )

    stationary_rows, unsafe_rows = build_stationary_rows(
        configs,
        population_size=args.population_size,
    )
    _write_csv(
        output / "theory_stationary_distribution.csv",
        stationary_rows,
        list(stationary_rows[0].keys()),
    )
    _write_csv(
        output / "theory_expected_unsafe.csv",
        unsafe_rows,
        list(unsafe_rows[0].keys()),
    )

    metadata = build_metadata(
        configs,
        population_size=args.population_size,
        method=args.payoff_method,
        replications=args.replications,
        seed=args.seed,
    )
    (output / "theory_metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote 5 theory files to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
