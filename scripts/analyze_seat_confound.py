#!/usr/bin/env python3
"""Is the physical seat a confound anywhere in this paper?

A reviewer asked for a seat and name counterbalancing run.  The baseline
campaign is self-play: one route holds both Company_1 and Company_2 in every
race, so seat is counterbalanced by construction and the two seats carry exactly
the same number of decisions.  What is not settled by construction is whether
the two seats behave the same, and pooled they do not: Company_2 plays Unsafe
slightly more often than Company_1 at every risk level.

This script measures that gap the way the manuscript measures everything else,
with the race as the clustering unit, and then asks whether any statistic the
paper reports could move because of it.

Estimators
  race-level   mean over races of (seat 2 race rate - seat 1 race rate).  This
               is the published estimator in fig_what_moves_play.seat().
  decision     (sum seat-2 unsafe / sum seat-2 decisions) - same for seat 1.
               This is the estimator that reproduces the pooled percentages a
               reader would compute by hand from turns.jsonl.
Both intervals resample races with replacement inside the reported stratum.
The resampling is fully vectorised over precomputed per-race arrays; nothing is
concatenated inside the bootstrap loop.

Outputs land in results/derived/seat_confound/ and nothing else is touched.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "scripts" / "figures"))

import figdata as D  # noqa: E402

OUT = ROOT / "results" / "derived" / "seat_confound"
N_BOOT = 20000
SEED = 20260911
SEAT1, SEAT2 = "Company_1", "Company_2"
NANO = "openai/gpt-5.4-nano-2026-03-17"
# The five routes the admission audit admitted; every headline result uses these.
ADMITTED = (
    "google/gemini-3-flash-preview",
    "anthropic/claude-opus-5@default",
    "openai/gpt-5.4-2026-03-05",
    "openai/gpt-5.5-2026-04-23",
    "anthropic/claude-sonnet-5@default",
)


# ---------------------------------------------------------------------------
# per-race arrays, computed once


def race_table(turns: pd.DataFrame) -> pd.DataFrame:
    """One row per race: unsafe count and decision count for each seat."""
    grouped = turns.groupby(
        ["model_route", "max_private_risk", "game_id", "player"], as_index=False
    ).agg(unsafe=("unsafe", "sum"), decisions=("unsafe", "size"))
    wide = grouped.pivot_table(
        index=["model_route", "max_private_risk", "game_id"],
        columns="player",
        values=["unsafe", "decisions"],
    )
    out = pd.DataFrame(
        {
            "u1": wide[("unsafe", SEAT1)],
            "u2": wide[("unsafe", SEAT2)],
            "n1": wide[("decisions", SEAT1)],
            "n2": wide[("decisions", SEAT2)],
        }
    ).reset_index()
    if not (out["n1"] == out["n2"]).all():
        raise ValueError("a race gave the two seats different decision counts")
    out["r1"] = out["u1"] / out["n1"]
    out["r2"] = out["u2"] / out["n2"]
    out["d"] = out["r2"] - out["r1"]
    return out


def ratio_gap_ci(block: pd.DataFrame, *, n_boot=N_BOOT, seed=SEED, alpha=0.05):
    """Decision-weighted seat gap, races resampled with replacement.

    ``figdata.cluster_bootstrap_ci`` resamples the mean of a per-race value and
    is used unchanged for the race-level estimator.  A ratio of two sums is not
    a mean of per-race values, so it needs its own resample; the resample is the
    same object (race indices) and the arithmetic stays vectorised.
    """
    u1 = block["u1"].to_numpy(float)
    u2 = block["u2"].to_numpy(float)
    n1 = block["n1"].to_numpy(float)
    n2 = block["n2"].to_numpy(float)
    if u1.size == 0:
        return float("nan"), float("nan"), float("nan")
    point = u2.sum() / n2.sum() - u1.sum() / n1.sum()
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, u1.size, size=(n_boot, u1.size))
    gaps = u2[draws].sum(axis=1) / n2[draws].sum(axis=1) - u1[draws].sum(
        axis=1
    ) / n1[draws].sum(axis=1)
    return (
        float(point),
        float(np.percentile(gaps, 100 * alpha / 2)),
        float(np.percentile(gaps, 100 * (1 - alpha / 2))),
    )


def summarise(block: pd.DataFrame, **keys) -> dict:
    mean_d, lo_d, hi_d = D.cluster_bootstrap_ci(
        block["d"].to_numpy(float), n_boot=N_BOOT, seed=SEED
    )
    point_r, lo_r, hi_r = ratio_gap_ci(block)
    return {
        **keys,
        "races": int(len(block)),
        "decisions": int(block["n1"].sum() + block["n2"].sum()),
        "seat1_rate": float(block["u1"].sum() / block["n1"].sum()),
        "seat2_rate": float(block["u2"].sum() / block["n2"].sum()),
        "gap_decision_pp": 100 * point_r,
        "gap_decision_lo": 100 * lo_r,
        "gap_decision_hi": 100 * hi_r,
        "gap_race_pp": 100 * mean_d,
        "gap_race_lo": 100 * lo_d,
        "gap_race_hi": 100 * hi_d,
        "races_seat2_higher": int((block["d"] > 0).sum()),
        "races_seat1_higher": int((block["d"] < 0).sum()),
        "races_tied": int((block["d"] == 0).sum()),
    }


def prompt_symmetry(turns: pd.DataFrame) -> dict:
    """Are the two seats' prompts the same text with the names exchanged?

    Only round one can be checked as a string, and that is the point rather than
    a limitation: before anything has happened every state field is zero on both
    sides, so any residue after the swap is a genuine asymmetry of the task.
    From round two the state fields legitimately mirror (a progress gap of
    ``+1`` on one side is ``-1`` on the other), so a string comparison there
    would measure the game, not the task.
    """
    opening = turns[turns["round"] == 1]
    wide = opening.pivot_table(
        index=["model_route", "game_id"], columns="player", values="prompt", aggfunc="first"
    )

    def swap(text: str) -> str:
        return (
            text.replace(SEAT1, "\x00").replace(SEAT2, SEAT1).replace("\x00", SEAT2)
        )

    same = [swap(a) == b for a, b in zip(wide[SEAT1], wide[SEAT2])]
    return {
        "races_checked": int(len(same)),
        "identical_under_name_swap": int(sum(same)),
        "all_identical": bool(all(same)),
    }


def gap_by_round(races_turns: pd.DataFrame) -> pd.DataFrame:
    """Where in the race the seat gap lives.

    Self-play reciprocity is the reason to look: if the seats open apart and the
    routes then match each other, the gap has to shrink as the race runs.
    """
    rows = []
    for rnd, block in races_turns.groupby("round"):
        wide = block.pivot_table(
            index=["model_route", "game_id"], columns="player", values="unsafe"
        )
        if SEAT1 not in wide or SEAT2 not in wide:
            continue
        d = (wide[SEAT2] - wide[SEAT1]).to_numpy(float)
        mean, lo, hi = D.cluster_bootstrap_ci(d, n_boot=N_BOOT, seed=SEED)
        rows.append(
            {
                "round": int(rnd),
                "races": int(d.size),
                "seat1_rate": float(wide[SEAT1].mean()),
                "seat2_rate": float(wide[SEAT2].mean()),
                "gap_pp": 100 * mean,
                "lo": 100 * lo,
                "hi": 100 * hi,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# memory-one fingerprint, per seat


def fingerprint(turns: pd.DataFrame) -> pd.DataFrame:
    from fig_policy_shape import prepare  # noqa: E402

    replies = prepare(turns)
    pooled = (
        replies.groupby(["model_route", "ctx"])["unsafe"]
        .agg(["mean", "size"])
        .rename(columns={"mean": "pooled_rate", "size": "pooled_n"})
    )
    per_seat = (
        replies.groupby(["model_route", "ctx", "player"])["unsafe"]
        .agg(["mean", "size"])
        .unstack("player")
    )
    out = pooled.join(
        pd.DataFrame(
            {
                "seat1_rate": per_seat[("mean", SEAT1)],
                "seat2_rate": per_seat[("mean", SEAT2)],
                "seat1_n": per_seat[("size", SEAT1)],
                "seat2_n": per_seat[("size", SEAT2)],
            }
        )
    ).reset_index()
    out["seat_gap_pp"] = 100 * (out["seat2_rate"] - out["seat1_rate"])
    out["seat_share_2"] = out["seat2_n"] / (out["seat1_n"] + out["seat2_n"])
    return out


def lagged_effects_by_seat(turns: pd.DataFrame) -> pd.DataFrame:
    from fig_policy_shape import mh_difference, prepare  # noqa: E402

    replies = prepare(turns)
    rows = []
    for route, block in replies.groupby("model_route"):
        record = {"model_route": route}
        for label, frame in (
            ("pooled", block),
            ("seat1", block[block["player"] == SEAT1]),
            ("seat2", block[block["player"] == SEAT2]),
        ):
            record[f"rival_{label}"] = 100 * mh_difference(
                frame, "opp_prev", ["max_private_risk", "own_prev"]
            )
            record[f"own_{label}"] = 100 * mh_difference(
                frame, "own_prev", ["max_private_risk", "opp_prev"]
            )
        # The decisive check for the published figure: put the seat in the
        # strata.  If the estimate does not move, the seat cannot be carrying
        # any part of it, whatever the two seats look like apart.
        record["rival_seat_stratified"] = 100 * mh_difference(
            block, "opp_prev", ["max_private_risk", "own_prev", "player"]
        )
        record["own_seat_stratified"] = 100 * mh_difference(
            block, "own_prev", ["max_private_risk", "opp_prev", "player"]
        )
        record["rival_shift_pp"] = record["rival_seat_stratified"] - record["rival_pooled"]
        record["own_shift_pp"] = record["own_seat_stratified"] - record["own_pooled"]
        rows.append(record)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# scripted-opponent arm: seat with the rival held fixed


def scripted_seat() -> pd.DataFrame:
    root = ROOT / "results" / "frontier" / "scripted_opponent_campaign"
    rows = []
    for turns_path in sorted(root.glob("*/*/turns.jsonl")):
        manifest = json.loads(
            (turns_path.with_name("run_manifest.json")).read_text(encoding="utf-8")
        )
        if manifest.get("status") != "completed":
            continue
        frame = pd.DataFrame(
            [json.loads(line) for line in turns_path.read_text(encoding="utf-8").splitlines()]
        )
        if frame["parse_failed"].any():
            raise ValueError(f"{turns_path} carries a parse failure")
        mine = frame[frame["is_route_decision"].astype(bool)].copy()
        mine["unsafe"] = mine["unsafe"].astype(int)
        cell = turns_path.parent.parent.name
        per_race = mine.groupby(["game_id", "route_seat"], as_index=False).agg(
            unsafe=("unsafe", "sum"), decisions=("unsafe", "size")
        )
        per_race["rate"] = per_race["unsafe"] / per_race["decisions"]
        seat0 = per_race[per_race["route_seat"] == 0]
        seat1 = per_race[per_race["route_seat"] == 1]
        gap = np.concatenate(
            [seat1["rate"].to_numpy(float), -seat0["rate"].to_numpy(float)]
        )
        rows.append(
            {
                "cell": cell,
                "rival": cell.split("_")[0],
                "risk": float(cell.split("risk")[1].replace("p", ".")),
                "races_seat1": int(len(seat0)),
                "races_seat2": int(len(seat1)),
                "seat1_rate": float(seat0["unsafe"].sum() / seat0["decisions"].sum()),
                "seat2_rate": float(seat1["unsafe"].sum() / seat1["decisions"].sum()),
                "gap_pp": 100
                * (
                    seat1["unsafe"].sum() / seat1["decisions"].sum()
                    - seat0["unsafe"].sum() / seat0["decisions"].sum()
                ),
                "_unused": float(gap.mean()),
            }
        )
    out = pd.DataFrame(rows).drop(columns="_unused")
    # The rival's stance and the risk level set the level of each cell, so the
    # cell is the stratum and the twelve within-cell gaps are the units.
    mean, lo, hi = D.cluster_bootstrap_ci(
        out["gap_pp"].to_numpy(float), n_boot=N_BOOT, seed=SEED
    )
    out.attrs["pooled"] = {
        "cells": int(len(out)),
        "mean_gap_pp": mean,
        "lo": lo,
        "hi": hi,
        "cells_seat2_higher": int((out["gap_pp"] > 0).sum()),
        "cells_seat1_higher": int((out["gap_pp"] < 0).sum()),
        "cells_equal": int((out["gap_pp"] == 0).sum()),
    }
    return out


# ---------------------------------------------------------------------------


def risk_response(turns: pd.DataFrame) -> dict:
    """The headline the seat gap has to be put beside."""
    per_race = turns.groupby(
        ["model_route", "max_private_risk", "game_id"], as_index=False
    ).agg(unsafe=("unsafe", "sum"), decisions=("unsafe", "size"))
    per_race["rate"] = per_race["unsafe"] / per_race["decisions"]
    out = {}
    for risk, block in per_race.groupby("max_private_risk"):
        mean, lo, hi = D.cluster_bootstrap_ci(
            block["rate"].to_numpy(float), n_boot=N_BOOT, seed=SEED
        )
        out[f"rate_at_{risk}"] = {
            "decision_rate_pct": 100 * block["unsafe"].sum() / block["decisions"].sum(),
            "race_mean_pct": 100 * mean,
            "lo": 100 * lo,
            "hi": 100 * hi,
        }
    low = per_race[per_race["max_private_risk"] == 0.1]
    high = per_race[per_race["max_private_risk"] == 0.9]
    out["drop_0.1_to_0.9_decision_pp"] = (
        100 * low["unsafe"].sum() / low["decisions"].sum()
        - 100 * high["unsafe"].sum() / high["decisions"].sum()
    )
    out["drop_0.1_to_0.9_race_pp"] = 100 * (low["rate"].mean() - high["rate"].mean())
    return out


def seat_gap_against_risk_response(turns: pd.DataFrame, gaps: pd.DataFrame) -> pd.DataFrame:
    """Per route, the thing the paper measures beside the thing it does not.

    A seat gap only means something next to the effect the paper is about.  The
    route whose risk response is smallest is the route where a seat gap of the
    same size is most damaging.
    """
    per = turns.groupby(["model_route", "max_private_risk"])["unsafe"].mean().unstack()
    out = pd.DataFrame(
        {
            "risk_response_pp": 100 * (per[0.1] - per[0.9]),
            "rate_0.1_pct": 100 * per[0.1],
            "rate_0.9_pct": 100 * per[0.9],
        }
    )
    route_gaps = gaps[gaps["scope"] == "route"].set_index("model_route")
    out["seat_gap_pp"] = route_gaps["gap_decision_pp"]
    out["seat_gap_lo"] = route_gaps["gap_decision_lo"]
    out["seat_gap_hi"] = route_gaps["gap_decision_hi"]
    out["seat_gap_over_risk_response"] = out["seat_gap_pp"].abs() / out["risk_response_pp"]
    return out.reset_index()


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    turns = D.baseline_turns()
    races = race_table(turns)
    opening = race_table(turns[turns["round"] == 1])

    print(
        f"baseline: {turns['model_route'].nunique()} routes, {len(races)} races, "
        f"{len(turns)} decisions, "
        f"{int(races['n1'].sum())} per seat"
    )

    rows = [summarise(races, scope="overall", model_route="ALL", max_private_risk="ALL")]
    for risk, block in races.groupby("max_private_risk"):
        rows.append(
            summarise(block, scope="risk", model_route="ALL", max_private_risk=risk)
        )
    for route, block in races.groupby("model_route"):
        rows.append(
            summarise(block, scope="route", model_route=route, max_private_risk="ALL")
        )
    for (route, risk), block in races.groupby(["model_route", "max_private_risk"]):
        rows.append(
            summarise(block, scope="route_x_risk", model_route=route, max_private_risk=risk)
        )
    gaps = pd.DataFrame(rows)
    gaps.to_csv(OUT / "seat_gap.csv", index=False)

    # Routes with room: at least one seat off the 0/1 boundary somewhere.
    per_route = gaps[gaps["scope"] == "route"].set_index("model_route")
    pinned = per_route[(per_route["seat1_rate"] >= 1) & (per_route["seat2_rate"] >= 1)].index
    with_room = races[~races["model_route"].isin(pinned)]
    room_row = summarise(
        with_room, scope="routes_with_room", model_route="ALL", max_private_risk="ALL"
    )

    # The pooled gap has to be checked against its own leave-one-route-out
    # profile, because a pooled effect carried by one route is a fact about that
    # route and not about the seat.
    loo = []
    for route in sorted(races["model_route"].unique()):
        loo.append(
            summarise(
                races[races["model_route"] != route],
                scope="leave_one_out",
                model_route=f"without {route}",
                max_private_risk="ALL",
            )
        )
    loo.append(
        summarise(
            races[races["model_route"].isin(ADMITTED)],
            scope="admitted_only",
            model_route="5 admitted routes",
            max_private_risk="ALL",
        )
    )
    # The opening move is the one place the two prompts are the same text, so
    # it gets the same leave-one-out and admitted-only treatment.
    after_one = race_table(turns[turns["round"] >= 2])
    for label, frame in (
        ("opening", opening),
        ("after_round_one", after_one),
        ("whole_race", races),
    ):
        for tag, sub in (
            ("all_nine", frame),
            ("without_nano", frame[frame["model_route"] != NANO]),
            ("admitted_only", frame[frame["model_route"].isin(ADMITTED)]),
        ):
            loo.append(
                summarise(sub, scope=f"{label}:{tag}", model_route=tag,
                          max_private_risk="ALL")
            )
    for route, block in opening.groupby("model_route"):
        loo.append(
            summarise(block, scope="opening:route", model_route=route,
                      max_private_risk="ALL")
        )
    loo_df = pd.DataFrame(loo)
    loo_df.to_csv(OUT / "seat_gap_leave_one_out.csv", index=False)

    opening_rows = [
        summarise(opening, scope="opening_overall", model_route="ALL", max_private_risk="ALL")
    ]
    for risk, block in opening.groupby("max_private_risk"):
        opening_rows.append(
            summarise(block, scope="opening_risk", model_route="ALL", max_private_risk=risk)
        )
    opening_df = pd.DataFrame(opening_rows)
    opening_df.to_csv(OUT / "seat_gap_opening.csv", index=False)

    symmetry = prompt_symmetry(turns)
    by_round = gap_by_round(turns)
    by_round.to_csv(OUT / "seat_gap_by_round.csv", index=False)

    fp = fingerprint(turns)
    fp.to_csv(OUT / "fingerprint_by_seat.csv", index=False)
    eff = lagged_effects_by_seat(turns)
    eff.to_csv(OUT / "lagged_effects_by_seat.csv", index=False)
    scripted = scripted_seat()
    scripted.to_csv(OUT / "scripted_seat.csv", index=False)
    risk = risk_response(turns)
    versus = seat_gap_against_risk_response(turns, gaps)
    versus.to_csv(OUT / "seat_gap_vs_risk_response.csv", index=False)

    summary = {
        "generator": "scripts/analyze_seat_confound.py",
        "source": "results/frontier/baseline_campaign_v6 (protocol ai-race-frontier-baseline-v3)",
        "n_boot": N_BOOT,
        "seed": SEED,
        "overall": rows[0],
        "routes_with_room": room_row,
        "pinned_routes": list(pinned),
        "opening": opening_rows[0],
        "risk_response": risk,
        "round_one_prompt_symmetry": symmetry,
        "scripted_opponent_pooled": scripted.attrs["pooled"],
        "seat_gap_over_risk_response_ratio": rows[0]["gap_decision_pp"]
        / risk["drop_0.1_to_0.9_decision_pp"],
    }
    (OUT / "seat_confound.json").write_text(
        json.dumps(summary, indent=2, default=float), encoding="utf-8"
    )

    pd.set_option("display.width", 200)
    print("\n=== seat gap, seat 2 minus seat 1, percentage points ===")
    show = ["scope", "model_route", "max_private_risk", "races", "seat1_rate",
            "seat2_rate", "gap_decision_pp", "gap_decision_lo", "gap_decision_hi",
            "gap_race_pp", "gap_race_lo", "gap_race_hi",
            "races_seat2_higher", "races_seat1_higher", "races_tied"]
    print(gaps[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\nroutes with room:", room_row)
    print("\n=== leave one route out, and admitted only ===")
    print(loo_df[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\n=== round-one prompt symmetry ===")
    print(json.dumps(symmetry, indent=2))
    print("\n=== seat gap by round ===")
    print(by_round.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\n=== opening move ===")
    print(opening_df[show].to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\n=== risk response, for proportion ===")
    print(json.dumps(risk, indent=2, default=float))
    print("\n=== memory-one fingerprint, per seat ===")
    print(fp.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print("\n=== lagged effects, per seat (pp) ===")
    print(eff.to_string(index=False, float_format=lambda v: f"{v:8.2f}"))
    print("\n=== scripted-opponent arm, rival held fixed ===")
    print(scripted.to_string(index=False, float_format=lambda v: f"{v:8.3f}"))
    print(json.dumps(scripted.attrs["pooled"], indent=2, default=float))
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
