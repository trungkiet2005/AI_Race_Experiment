#!/usr/bin/env python3
"""Build the cluster-aware human trajectory-diversity null.

The original matched null samples twenty complete human trajectories.  The
source study, however, collected those trajectories in two-person races.  This
analysis preserves the comparison size while sampling ten complete dyads, so
both human seats enter together just as both seats of an LLM race do.  The
participant-level null and its artifact remain unchanged as a sensitivity
analysis; this file is an explicit, versioned correction rather than a silent
replacement.

The unit is a complete human ``group_id`` with exactly two complete
participants.  Four complete trajectories belong to singleton groups and are
excluded from the dyad null; a fifth singleton group has no complete
five-round trajectory and is absent from the diversity frame.  Every draw
contains ten complete dyads, twenty trajectories, and is without replacement
within the source pool.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
HUMAN_CSV = ROOT / "references" / "source_study_dataset" / "airace_deidentified_long.csv"
OUT = ROOT / "results" / "derived" / "trajectory_diversity_dyad_primary"

RISKS = (0.1, 0.6, 0.9)
N_DRAWS = 20_000
SEED = 20260910
DYADS_PER_DRAW = 10
TRAJECTORIES_PER_DYAD = 2
COMPARISON_N = DYADS_PER_DRAW * TRAJECTORIES_PER_DYAD


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_modules():
    import sys

    sys.path.insert(0, str(ROOT / "scripts"))
    import analyze_trajectory_diversity_confirmatory as confirmatory
    import analyze_trajectory_diversity_rarefaction as rarefaction
    import figdata
    import figstyle

    return confirmatory, rarefaction, figdata, figstyle


def load_human_dyads(rarefaction) -> tuple[pd.DataFrame, dict[str, object]]:
    """Return complete dyads as rows and a source-coverage audit."""
    human = rarefaction.load_human()
    raw = pd.read_csv(HUMAN_CSV, usecols=["participant_id", "group_id"])
    group_of = (
        raw.drop_duplicates("participant_id")
        .set_index("participant_id")["group_id"]
        .astype(str)
    )
    human = human.assign(group=human["unit"].map(group_of))
    if human["group"].isna().any():
        raise ValueError("A complete human trajectory has no source group_id")

    records: list[dict[str, object]] = []
    n_singletons = 0
    for (risk, group_id), block in human.groupby(["risk_cap", "group"], sort=True):
        participants = block["unit"].astype(str).unique().tolist()
        if len(participants) == 1:
            n_singletons += 1
            continue
        if len(participants) != 2 or len(block) != 2:
            raise ValueError(
                f"group {group_id} at risk {risk} has {len(participants)} participants "
                f"and {len(block)} complete trajectories, not one complete dyad"
            )
        for seat, row in enumerate(block.sort_values("unit").itertuples(index=False)):
            records.append(
                {
                    "risk_cap": float(risk),
                    "group": str(group_id),
                    "participant": str(row.unit),
                    "seat": seat + 1,
                    "trajectory": str(row.trajectory),
                }
            )

    dyads = pd.DataFrame.from_records(records)
    if dyads.empty:
        raise ValueError("No complete human dyads found")
    group_sizes = dyads.groupby(["risk_cap", "group"], sort=True).size()
    if not (group_sizes == 2).all():
        raise ValueError("The dyad table does not contain exactly two trajectories per group")
    audit = {
        "complete_human_trajectories": int(len(human)),
        "complete_dyads_total": int(dyads["group"].nunique()),
        "singleton_groups_excluded": int(n_singletons),
        "complete_dyads_by_risk": {
            f"{risk:.1f}": int(dyads[dyads["risk_cap"].eq(risk)]["group"].nunique())
            for risk in RISKS
        },
        "complete_trajectories_by_risk": {
            f"{risk:.1f}": int(dyads[dyads["risk_cap"].eq(risk)].shape[0])
            for risk in RISKS
        },
    }
    return dyads, audit


def distinct_per_row(codes: np.ndarray) -> np.ndarray:
    ordered = np.sort(codes, axis=1)
    return 1 + (np.diff(ordered, axis=1) != 0).sum(axis=1)


def encode(values: list[str], vocabulary: dict[str, int]) -> np.ndarray:
    return np.asarray(
        [vocabulary.setdefault(value, len(vocabulary)) for value in values],
        dtype=np.int32,
    )


def dyad_null(dyads: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    draws = []
    for matrix in dyads:
        if len(matrix) < DYADS_PER_DRAW:
            raise ValueError(
                f"Only {len(matrix)} complete dyads available for a {DYADS_PER_DRAW}-dyad draw"
            )
        order = np.argsort(rng.random((N_DRAWS, len(matrix))), axis=1)
        selected = matrix[order[:, :DYADS_PER_DRAW]].reshape(N_DRAWS, COMPARISON_N)
        draws.append(distinct_per_row(selected))
    return np.column_stack(draws)


def main() -> None:
    confirmatory, rarefaction, figdata, figstyle = _load_modules()
    human, coverage = load_human_dyads(rarefaction)
    model = confirmatory.confirmatory_frame(rarefaction)
    model["population"] = model["population"].map(
        lambda value: figstyle.ROUTE_LABEL.get(value, value)
    )

    vocabulary: dict[str, int] = {}
    dyad_matrices = []
    for risk in RISKS:
        block = human[human["risk_cap"].eq(risk)]
        matrix = np.asarray(
            [encode(group["trajectory"].tolist(), vocabulary)
             for _, group in block.groupby("group", sort=True)],
            dtype=np.int32,
        )
        if matrix.shape[1] != TRAJECTORIES_PER_DYAD:
            raise ValueError(f"risk {risk}: dyad matrix shape {matrix.shape}")
        dyad_matrices.append(matrix)

    counts = dyad_null(dyad_matrices, np.random.default_rng([SEED, 2]))
    draws = pd.DataFrame(
        {
            "risk_cap": np.repeat(RISKS, N_DRAWS),
            "draw_id": np.tile(np.arange(N_DRAWS), len(RISKS)),
            "distinct_trajectories": counts.reshape(-1, order="F"),
        }
    )

    routes = [figstyle.ROUTE_LABEL[route] for route in figstyle.ROUTE_ORDER]
    rows: list[dict[str, object]] = []
    for column, risk in enumerate(RISKS):
        distribution = counts[:, column]
        for route in routes:
            cell = model[
                model["population"].eq(route)
                & np.isclose(model["risk_cap"].astype(float), risk)
            ]
            if len(cell) != COMPARISON_N:
                raise ValueError(f"{route} at risk {risk} has {len(cell)} trajectories")
            model_distinct = int(cell["trajectory"].nunique())
            rows.append(
                {
                    "risk_cap": risk,
                    "population": route,
                    "model_distinct": model_distinct,
                    "comparison_n": COMPARISON_N,
                    "n_human_dyads": int(coverage["complete_dyads_by_risk"][f"{risk:.1f}"]),
                    "human_draw_mean": float(distribution.mean()),
                    "human_draw_q025": float(np.quantile(distribution, 0.025)),
                    "human_draw_q975": float(np.quantile(distribution, 0.975)),
                    "human_draw_min": int(distribution.min()),
                    "human_draw_max": int(distribution.max()),
                    "below_every_human_draw": bool(model_distinct < distribution.min()),
                }
            )
    summary = pd.DataFrame(rows)

    admitted = [figstyle.ROUTE_LABEL[route] for route in figstyle.ADMITTED]
    admitted_rows = summary[summary["population"].isin(admitted)]
    admitted_below = admitted_rows["below_every_human_draw"].sum()
    all_below = int(summary["below_every_human_draw"].sum())
    exceptions = summary[~summary["below_every_human_draw"]][
        ["population", "risk_cap", "model_distinct", "human_draw_min"]
    ].to_dict(orient="records")

    source_files = [HUMAN_CSV]
    source_files += sorted(
        path
        for path in figdata.BASELINE_ROOT.glob(
            "*/*/*/results/ai_race_baseline/turns.jsonl"
        )
        if "failed_runs" not in path.parts
    )
    payload = {
        "schema_version": "ai-race-trajectory-diversity-dyad-primary-v1",
        "analysis_role": (
            "cluster-aware primary human null for the trajectory-diversity result; "
            "the original participant-level null is retained as sensitivity"
        ),
        "methodological_status": (
            "explicit clustering correction added after audit; no original artifact overwritten"
        ),
        "estimand": "distinct paired first-five-round trajectories in a sample of twenty trajectories",
        "human_cluster_unit": "complete group_id dyad with two participants",
        "model_cluster_unit": "game_id race with two seats",
        "draw_method": "sample ten complete dyads without replacement within each human risk pool",
        "n_draws": N_DRAWS,
        "seed": [SEED, 2],
        "comparison_n": COMPARISON_N,
        "risk_caps": list(RISKS),
        "coverage": coverage,
        "routes": routes,
        "admitted_routes": admitted,
        "below_every_human_draw": all_below,
        "admitted_below_every_human_draw": int(admitted_below),
        "admitted_cell_count": len(admitted) * len(RISKS),
        "exceptions": exceptions,
        "source_hashes": {
            path.relative_to(ROOT).as_posix(): sha256(path) for path in source_files
        },
        "outputs": {
            "summary": "trajectory_diversity_dyad_primary.csv",
            "draws": "human_dyad_null_draws.csv",
        },
    }

    OUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUT / "trajectory_diversity_dyad_primary.csv", index=False)
    draws.to_csv(OUT / "human_dyad_null_draws.csv", index=False)
    (OUT / "trajectory_diversity_dyad_primary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
        newline="\n",
    )

    print(f"complete human trajectories: {coverage['complete_human_trajectories']}")
    print(
        f"complete dyads: {coverage['complete_dyads_total']} "
        f"({coverage['singleton_groups_excluded']} complete singleton trajectories excluded)"
    )
    print(f"dyad-null floors: {[int(counts[:, i].min()) for i in range(len(RISKS))]}")
    print(f"all routes below every dyad draw: {all_below} of {len(routes) * len(RISKS)}")
    print(f"admitted routes below every dyad draw: {int(admitted_below)} of {len(admitted) * len(RISKS)}")
    print("exceptions:")
    for row in exceptions:
        print(
            f"  {row['population']} risk {row['risk_cap']}: "
            f"model {row['model_distinct']} vs dyad floor {row['human_draw_min']}"
        )
    print(f"-> {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
