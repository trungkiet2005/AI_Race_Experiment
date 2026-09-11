"""Trajectory diversity on the nine confirmatory routes, not the pilot export.

The published diversity table is built from seven separate pilot CSVs collected
before the nine-route confirmatory campaign existed. Two consequences follow,
and both bear on what the paper is allowed to claim.

It includes GPT-5 nano, a checkpoint that is no longer reachable and therefore
carries no admission verdict, and it omits GPT-5.4 and GPT-5.5, which are two of
the five routes the gate admits. So the sentence "every checkpoint that passes
the gate is less diverse than the human sample" was being asserted over a set
that did not contain two of the checkpoints that pass the gate.

No new collection is needed to fix that. The trajectory is the focal player's
first five actions paired with the opponent's, and the confirmatory campaign
holds all nine routes at ten races per risk level, which is twenty paired
player-trajectories per cell: exactly the comparison size the pilot analysis
rarefies everything down to. Every one of its 270 races reaches the fifth round,
because the mechanism guarantees a five-round minimum.

This module therefore reuses the published analysis unchanged, statistic for
statistic and seed for seed, and only swaps where the trajectories come from.
Reusing rather than reimplementing is the point: a second implementation would
make any difference between the two tables uninterpretable.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "derived" / "trajectory_diversity_confirmatory"


def _published_analysis():
    """Import the pilot analyser without running it."""
    path = ROOT / "scripts" / "analyze_trajectory_diversity_rarefaction.py"
    spec = importlib.util.spec_from_file_location("_diversity_pilot", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_diversity_pilot"] = module
    spec.loader.exec_module(module)
    return module


def confirmatory_frame(published) -> pd.DataFrame:
    """The same trajectory table, built from the confirmatory campaign."""
    sys.path.insert(0, str(ROOT / "scripts"))
    import figdata

    turns = figdata.baseline_turns()
    wide = turns.pivot_table(
        index=["model_route", "max_private_risk", "game_id", "round"],
        columns="player", values="unsafe",
    ).reset_index()
    opening = wide[wide["round"] <= 5].sort_values("round")

    rows: list[dict[str, object]] = []
    for (route, risk, game), block in opening.groupby(
        ["model_route", "max_private_risk", "game_id"]
    ):
        if len(block) != 5:
            # A race short of five rounds cannot carry this trajectory, and
            # padding one would invent behaviour. The mechanism's five-round
            # minimum means this should never fire; it is here so that a change
            # to the mechanism cannot silently shorten a trajectory.
            raise ValueError(f"{route} {risk} {game}: {len(block)} opening rounds")
        first = tuple(int(v) for v in block["Company_1"])
        second = tuple(int(v) for v in block["Company_2"])
        label = figdata.ROUTE_LABEL_SHORT.get(route, route) if hasattr(
            figdata, "ROUTE_LABEL_SHORT") else route
        for own, other, seat in ((first, second, "p1"), (second, first, "p2")):
            rows.append({
                "population": label,
                "unit": f"{game}::{seat}",
                "cluster": str(game),
                "risk_cap": float(risk),
                "trajectory": published.paired_key(own, other),
            })
    return pd.DataFrame(rows)


def main() -> None:
    published = _published_analysis()
    import figstyle

    model = confirmatory_frame(published)
    model["population"] = model["population"].map(
        lambda r: figstyle.ROUTE_LABEL.get(r, r))
    human = published.load_human()
    frame = pd.concat([human, model], ignore_index=True)

    # build_table walks the population order the pilot analysis declares, which
    # names checkpoints this campaign does not contain. Point it at the routes
    # actually present instead of reimplementing the walk.
    published.MODEL_INPUTS = [
        (figstyle.ROUTE_LABEL[route], None)
        for route in figstyle.ADMITTED + figstyle.REFUSED
    ]

    table = published.build_table(frame)
    OUT.mkdir(parents=True, exist_ok=True)
    table.to_csv(OUT / "trajectory_diversity_confirmatory.csv", index=False)

    admitted = [figstyle.ROUTE_LABEL[r] for r in figstyle.ADMITTED]
    refused = [figstyle.ROUTE_LABEL[r] for r in figstyle.REFUSED]
    verdict = {
        **{name: "admitted" for name in admitted},
        **{name: "refused" for name in refused},
    }

    below = {}
    for name in admitted + refused:
        rows = table[table["population"] == name]
        if rows.empty:
            below[name] = None
            continue
        strictly = all(
            float(rows[rows["risk_cap"] == risk]["q1_ci_high"].iloc[0])
            < float(table[(table["population"] == "Human")
                          & (table["risk_cap"] == risk)]["q1_ci_low"].iloc[0])
            for risk in (0.1, 0.6, 0.9)
        )
        below[name] = bool(strictly)

    payload = {
        "schema_version": "trajectory-diversity-confirmatory-v1",
        "source": "results/frontier/baseline_campaign_v6",
        "statistic": "first-five-round paired action trajectory, rarefied to 20",
        "reuses": "scripts/analyze_trajectory_diversity_rarefaction.py",
        "routes": sorted(set(table["population"]) - {"Human"}),
        "verdict": verdict,
        "entirely_below_human_q1": below,
        "admitted_all_below": all(below[name] for name in admitted),
    }
    (OUT / "trajectory_diversity_confirmatory.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8")

    print(f"{len(payload['routes'])} routes on the confirmatory footing\n")
    for name in admitted + refused:
        mark = "admitted" if name in admitted else "refused "
        q1 = [round(float(table[(table['population'] == name)
                                & (table['risk_cap'] == r)]["q1_mean"].iloc[0]), 1)
              for r in (0.1, 0.6, 0.9)]
        print(f"  {mark}  {name:<24s} q1 {q1}   "
              f"{'entirely below human' if below[name] else 'overlaps human'}")
    human_q1 = [round(float(table[(table['population'] == 'Human')
                                  & (table['risk_cap'] == r)]["q1_mean"].iloc[0]), 1)
                for r in (0.1, 0.6, 0.9)]
    print(f"\n  Human q1 {human_q1}")
    print(f"\n  every admitted route entirely below the human interval: "
          f"{payload['admitted_all_below']}")
    print(f"  -> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
