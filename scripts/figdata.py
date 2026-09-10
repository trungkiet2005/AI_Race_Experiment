"""Loaders shared by the figure scripts.

Nothing here recomputes a number the manuscript quotes.
``scripts/verify_manuscript_claims.py`` remains the single authority on those,
and a figure that disagrees with it is a bug in the figure.  What these loaders
provide is the raw decision table in a shape a figure can use, with the
provenance checks applied once instead of in every script.

The checks are fail-closed on purpose.  A figure is the most persuasive object
in a paper and the least audited, so the place to refuse a contaminated run is
before it is drawn, not after a reviewer asks.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

BASELINE_ROOT = ROOT / "results" / "frontier" / "baseline_campaign_v6" / "ai-race-baseline"
BASELINE_PROTOCOL = "ai-race-frontier-baseline-v3"
ADMISSION_CSV = (
    ROOT / "results" / "frontier" / "admission_campaign_v6" / "derived"
    / "admission_campaign_v6.csv"
)
AUDIT_CSV = (
    ROOT / "results" / "frontier" / "baseline_campaign_v6" / "derived"
    / "audit_versus_behaviour.csv"
)
MATCHED_ROOT = ROOT / "results" / "frontier" / "nplayer_matched_campaign"
MATCHED_PROTOCOL = "ai-race-nplayer-matched-hosted-confirmatory-v1"
MAPPING_ROOT = ROOT / "results" / "frontier" / "context_mapping_campaign_v3"
REPLICATION = ROOT / "results" / "frontier" / "baseline_replication" / "gemini-3-flash-preview"

EXPECTED_BASELINE_RACES = 30
EXPECTED_BASELINE_DECISIONS = 558


def _read_turns(path: Path) -> pd.DataFrame:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    return pd.DataFrame(rows)


def baseline_turns(*, include_replication: bool = False) -> pd.DataFrame:
    """Every decision of the nine-route neutral baseline, one row per decision.

    Skips ``failed_runs`` entirely: a refused or interrupted run is a failure
    record and never behaviour.  Refuses any run that did not complete, that is
    the wrong protocol, or that carries a parse failure, because a parse failure
    contaminates the race it sits in rather than only its own row.
    """
    frames = []
    for manifest_path in sorted(BASELINE_ROOT.glob("*/*/*/results/ai_race_baseline/run_manifest.json")):
        if "failed_runs" in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        if manifest.get("protocol_id") != BASELINE_PROTOCOL:
            raise ValueError(
                f"{manifest_path} is protocol {manifest.get('protocol_id')!r}, "
                f"not {BASELINE_PROTOCOL}"
            )
        turns = _read_turns(manifest_path.with_name("turns.jsonl"))
        if turns["parse_failed"].any():
            raise ValueError(f"{manifest_path} carries a parse failure")
        if len(turns) != EXPECTED_BASELINE_DECISIONS:
            raise ValueError(
                f"{manifest_path} has {len(turns)} decisions, "
                f"not {EXPECTED_BASELINE_DECISIONS}"
            )
        turns["model_route"] = manifest["model_route"]
        turns["task_version"] = manifest_path.parts[-5]
        turns["run_id"] = manifest_path.parts[-4]
        frames.append(turns)

    if include_replication:
        manifest = json.loads((REPLICATION / "run_manifest.json").read_text(encoding="utf-8"))
        turns = _read_turns(REPLICATION / "turns.jsonl")
        turns["model_route"] = manifest["model_route"]
        turns["task_version"] = "replication"
        turns["run_id"] = "replication"
        frames.append(turns)

    if not frames:
        raise ValueError(f"no completed baseline runs under {BASELINE_ROOT}")
    out = pd.concat(frames, ignore_index=True)

    # One run per route is what every campaign analyser assumes, and a second
    # one silently displacing the first is the exact failure the replication
    # directory exists to avoid.  Say so here rather than averaging them.
    if not include_replication:
        per_route = out.groupby("model_route")["run_id"].nunique()
        if (per_route > 1).any():
            offenders = per_route[per_route > 1].to_dict()
            raise ValueError(f"more than one run for {offenders}; refusing to pool")

    out["unsafe"] = out["unsafe"].astype(int)
    out["max_private_risk"] = out["max_private_risk"].astype(float)
    return out


def race_rates(turns: pd.DataFrame, *, by=("model_route", "max_private_risk")) -> pd.DataFrame:
    """Unsafe rate per race, which is the independent unit for every interval.

    The race is the unit because decisions inside a race are dependent: the same
    two agents, the same horizon draw, a trajectory that each round conditions
    on.  Any interval taken over decisions rather than races is too narrow, and
    the manuscript's own policy is to cluster on the race.
    """
    grouped = turns.groupby(list(by) + ["game_id"], as_index=False).agg(
        unsafe=("unsafe", "sum"), decisions=("unsafe", "size")
    )
    grouped["rate"] = grouped["unsafe"] / grouped["decisions"]
    return grouped


def conditional_response(turns: pd.DataFrame) -> pd.DataFrame:
    """P(unsafe) split by what the opponent did on the previous round.

    Round one has no predecessor and is excluded rather than folded in, because
    an opening move is a different object from a reply and pooling them hides
    both.  The difference between the two columns is the quantity the iterated
    game literature calls reciprocity, and a route whose two columns are equal
    is not responding to its opponent at all.
    """
    replies = turns[turns["opponent_prev_action"].notna()].copy()
    replies["opp_prev"] = replies["opponent_prev_action"].str.upper()
    out = (
        replies.groupby(["model_route", "max_private_risk", "opp_prev"], as_index=False)
        .agg(unsafe=("unsafe", "sum"), decisions=("unsafe", "size"))
    )
    out["rate"] = out["unsafe"] / out["decisions"]
    return out


def first_moves(turns: pd.DataFrame) -> pd.DataFrame:
    """The opening decision of every race, before any information exists.

    Nothing has happened yet, so an opening move is the closest thing the
    protocol has to a prior: it is what the route does with the rules alone.
    """
    opening = turns[turns["round"] == 1]
    out = opening.groupby(["model_route", "max_private_risk"], as_index=False).agg(
        unsafe=("unsafe", "sum"), decisions=("unsafe", "size")
    )
    out["rate"] = out["unsafe"] / out["decisions"]
    return out


def admission() -> pd.DataFrame:
    """Per-route comprehension accuracy, by probe domain."""
    return pd.read_csv(ADMISSION_CSV)


def audit_versus_behaviour() -> pd.DataFrame:
    """The join of comprehension accuracy to risk response, nine routes."""
    return pd.read_csv(AUDIT_CSV)


def matched_cells() -> pd.DataFrame:
    """Every decision of the matched group-size grid, with its cell and receipt.

    Refuses a cell whose seat count disagrees with the directory it sits in, or
    that carries a parse failure, or that is the wrong protocol.
    """
    frames = []
    for receipt_path in sorted(MATCHED_ROOT.glob("*/*/collection_receipt.json")):
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        # The run manifest is the authority on protocol and completion; the
        # receipt only carries what the server could not tell us, which is the
        # collecting identity.  The four risk-0.6 receipts were written before
        # the ingestion script existed and carry no protocol field at all, so
        # asking the receipt would refuse cells that are perfectly sound.
        manifest_path = receipt_path.with_name("run_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != MATCHED_PROTOCOL:
            raise ValueError(
                f"{manifest_path} is protocol {manifest.get('protocol_id')!r}, "
                f"not {MATCHED_PROTOCOL}"
            )
        if manifest.get("status") != "completed":
            raise ValueError(f"{manifest_path} did not complete")
        if receipt.get("parse_failures"):
            raise ValueError(f"{receipt_path} carries a parse failure")
        n_players = int(receipt["cell"]["n_players"])
        risk = float(receipt["cell"]["max_private_risk"])
        if manifest.get("group_sizes") != [n_players] or manifest.get("risk_levels") != [risk]:
            raise ValueError(
                f"{receipt_path} says cell N={n_players} risk={risk} but the "
                f"manifest ran {manifest.get('group_sizes')} x {manifest.get('risk_levels')}"
            )
        matches = list(receipt_path.parent.glob(f"n{n_players}/turns.jsonl"))
        if len(matches) != 1:
            raise ValueError(f"{receipt_path}: expected one turns.jsonl, found {len(matches)}")
        turns = _read_turns(matches[0])
        if turns["parse_failed"].any():
            raise ValueError(f"{matches[0]} carries a parse failure")
        turns["n_players"] = n_players
        turns["cell_risk"] = risk
        turns["executing_identity"] = receipt["executing_identity"]
        frames.append(turns)
    if not frames:
        raise ValueError(f"no matched cells under {MATCHED_ROOT}")
    out = pd.concat(frames, ignore_index=True)
    out["unsafe"] = out["unsafe"].astype(int)
    return out


def mapping_turns() -> pd.DataFrame:
    """Every decision of the crossed representation design, both routes.

    Never descends into ``failed_runs``: the superseded 106-race fragment there
    is an unbalanced piece of a design whose balance is the point.
    """
    frames = []
    for manifest_path in sorted(MAPPING_ROOT.rglob("run_manifest.json")):
        if "failed_runs" in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        turns_path = manifest_path.with_name("turns.jsonl")
        if not turns_path.exists():
            continue
        turns = _read_turns(turns_path)
        if turns["parse_failed"].any():
            raise ValueError(f"{turns_path} carries a parse failure")
        turns["model_route"] = manifest.get("model_route")
        frames.append(turns)
    if not frames:
        raise ValueError(f"no completed mapping runs under {MAPPING_ROOT}")
    return pd.concat(frames, ignore_index=True)


def cluster_bootstrap_ci(rates, *, n_boot=5000, seed=20260910, alpha=0.05):
    """Percentile interval resampling races, which are the independent unit."""
    import numpy as np

    values = np.asarray(list(rates), dtype=float)
    if values.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, values.size, size=(n_boot, values.size))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 100 * alpha / 2)),
        float(np.percentile(means, 100 * (1 - alpha / 2))),
    )


if __name__ == "__main__":
    turns = baseline_turns()
    print(f"baseline: {turns['model_route'].nunique()} routes, "
          f"{turns['game_id'].nunique()} races, {len(turns)} decisions")
    cond = conditional_response(turns)
    print(f"conditional response rows: {len(cond)}")
    matched = matched_cells()
    print(f"matched: {matched.groupby(['n_players', 'cell_risk']).ngroups} cells, "
          f"{len(matched)} decisions")
    mapping = mapping_turns()
    print(f"mapping: {mapping['model_route'].nunique()} routes, {len(mapping)} decisions")
