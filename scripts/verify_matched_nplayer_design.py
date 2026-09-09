"""Check the matched group-size sweep offline, before any of it is pushed.

Three properties have to hold for the sweep to mean what the manuscript will say
it means, and none of them is obvious from reading the configs.

First, the mechanism must be the same game at every group size. The stage payoff
follows the group-count rule, so it necessarily changes with N; everything else,
the progress increments, the cost, the benefit, the speed, the minimum horizon,
the stopping probability and the prize, must not.

Second, the two-player arm must be the paper's own two-player game rather than a
near neighbour. It is: the group-count rule evaluated at two players reproduces
the payoff matrix 1.0 / 0.6 / 2.4 / 2.0 exactly. Without that, the sweep would
compare the three- to five-player pilots against a slightly different baseline
and the group-size contrast would be confounded at its anchor.

Third, the contrast must be pairable. The game seed is base_seed + rep and is
independent of the treatment name and of the number of seats, so a repetition
index should reuse one horizon stopping-draw stream across every risk and every
group size. That is what lets a group-size difference be taken within a
repetition, with the horizon differenced out. This script replays the streams
and asserts it rather than trusting the seed derivation to have that property.

Run it before pushing the task. It touches no network and costs nothing.
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ai_race.engine_nplayer.runner import build_games_for_model, load_json  # noqa: E402

GROUP_SIZES = (2, 3, 4, 5)
RISKS = (0.1, 0.6, 0.9)
REPETITIONS = 4
SEED = 260802
DRAWS_REPLAYED = 12

# The headline two-player game, own action by row.
TWO_PLAYER_MATRIX = {
    ("SAFE", "SAFE"): 1.0,
    ("SAFE", "UNSAFE"): 0.6,
    ("UNSAFE", "SAFE"): 2.4,
    ("UNSAFE", "UNSAFE"): 2.0,
}

failures: list[str] = []


def report(name: str, ok: bool, detail: str) -> None:
    print(f"{'PASS' if ok else 'FAIL'}  {name}  {detail}")
    if not ok:
        failures.append(name)


def build(n: int):
    experiment = load_json(ROOT / "ai_race" / "configs" / "experiment" / f"baseline_nplayer_n{n}.json")
    experiment.update(
        {
            "models": ["mock"],
            "repetitions": REPETITIONS,
            "seed": SEED,
            "verbose": False,
            "useOffline": False,
            "runPhase": "pilot",
        }
    )
    return build_games_for_model(experiment, "mock")


def main() -> None:
    games = {n: build(n) for n in GROUP_SIZES}

    for n, group in games.items():
        report(
            f"N={n} builds one game per risk and repetition",
            len(group) == len(RISKS) * REPETITIONS,
            f"{len(group)} games",
        )
        report(
            f"N={n} seats every game with {n} companies",
            all(g.config.n_players == n for g in group),
            f"seat counts {sorted({g.config.n_players for g in group})}",
        )

    # --- the mechanism outside the group-count rule is invariant -------------
    shared = {}
    for n, group in games.items():
        c = group[0].config
        shared[n] = (
            c.safe_progress,
            # unsafe_progress is a method on this config, not a field.
            c.unsafe_progress(),
            c.cost,
            c.benefit,
            c.speed,
            c.min_rounds,
            c.stop_probability,
            c.race_prize,
        )
    report(
        "every group size shares the mechanism outside the group-count rule",
        len(set(shared.values())) == 1,
        f"{shared[2]}",
    )

    # --- the group-count rule is the closed form ----------------------------
    for n, group in games.items():
        c = group[0].config
        ok = True
        for k_safe in range(0, n + 1):
            denominator = k_safe + c.speed * (n - k_safe)
            if 1 <= k_safe <= n:
                ok &= abs(c.stage_payoff_safe(k_safe) - (c.benefit / denominator - c.cost)) < 1e-12
            if 0 <= k_safe < n:
                ok &= abs(c.stage_payoff_unsafe(k_safe) - (c.speed * c.benefit / denominator)) < 1e-12
        report(
            f"N={n} stage payoffs follow the group-count rule",
            ok,
            f"safe(1)={c.stage_payoff_safe(1):+.4f} unsafe(0)={c.stage_payoff_unsafe(0):+.4f}",
        )

    # --- the two-player arm is the paper's own game -------------------------
    c2 = games[2][0].config
    got = {
        ("SAFE", "SAFE"): c2.stage_payoff_safe(2),
        ("SAFE", "UNSAFE"): c2.stage_payoff_safe(1),
        ("UNSAFE", "SAFE"): c2.stage_payoff_unsafe(1),
        ("UNSAFE", "UNSAFE"): c2.stage_payoff_unsafe(0),
    }
    report(
        "the N=2 arm reproduces the headline two-player payoff matrix",
        all(abs(got[k] - v) < 1e-12 for k, v in TWO_PLAYER_MATRIX.items()),
        ", ".join(f"{k[0][0]}v{k[1][0]}={got[k]:.1f}" for k in TWO_PLAYER_MATRIX),
    )

    # --- the horizon stream is shared, so contrasts are pairable ------------
    streams: dict[int, dict[tuple[float, int], tuple[float, ...]]] = {}
    for n, group in games.items():
        streams[n] = {}
        for g in group:
            rng = copy.deepcopy(g._horizon_rng)
            streams[n][(round(g.config.max_private_risk, 3), g.rep)] = tuple(
                round(rng.random(), 12) for _ in range(DRAWS_REPLAYED)
            )

    for n in GROUP_SIZES:
        reps = sorted({key[1] for key in streams[n]})
        risks = sorted({key[0] for key in streams[n]})
        ok = all(len({streams[n][(r, rep)] for r in risks}) == 1 for rep in reps)
        report(
            f"N={n} shares one horizon stream across the three risks",
            ok,
            f"{len(reps)} repetitions checked",
        )

    keys = sorted(streams[GROUP_SIZES[0]])
    ok = all(len({streams[n][key] for n in GROUP_SIZES}) == 1 for key in keys)
    report(
        "the same repetition shares one horizon stream across group sizes",
        ok,
        f"{len(keys)} (risk, repetition) cells checked, {DRAWS_REPLAYED} draws each",
    )

    print()
    if failures:
        raise SystemExit(f"{len(failures)} design check(s) failed: {failures}")
    print("matched group-size design verified offline; safe to push")


if __name__ == "__main__":
    main()
