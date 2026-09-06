"""
Build the de-identified public deposit dataset for
"When Competitors Take Risks: Unsafe Behaviour in an Idealised AI Race
Experiment" from the canonical experimental dataset.

Source (not included in this repo, run locally against the analysis repo):
    ~/PycharmProjects/airace-analysis/data/experimental_data/datasets/
        all_apps_with_prolific_airace_long.pkl

Output (written into this folder):
    airace_deidentified_long.csv

This script performs the column selection, renaming, de-identification
(participant/pair ID re-randomisation, nationality collapsing, age binning)
and drops `prolific_id` and every field not used to produce a result in the
paper. See README.md for the full field dictionary and the rationale for
each transformation.

The old-ID -> new-ID mapping is written to a local-only file
(`_id_mapping_DO_NOT_DEPOSIT.csv`) purely so the build is auditable/
re-runnable; that mapping file must never be committed or deposited, since
it re-links the anonymous IDs back to the internal oTree participant codes.
"""

import numpy as np
import pandas as pd

SOURCE_PATH = (
    "/Users/eliasfernandez/PycharmProjects/airace-analysis/"
    "data/experimental_data/datasets/all_apps_with_prolific_airace_long.pkl"
)
OUTPUT_CSV = "airace_deidentified_long.csv"
ID_MAPPING_CSV = "_id_mapping_DO_NOT_DEPOSIT.csv"

RNG_SEED = 20260717  # fixed seed for a reproducible (but non-reversible) build

RENAME_MAP = {
    "round_number": "round_number",
    "max_private_risk": "max_private_risk",
    "player__round_decision": "decision",
    "player__round_decision_other": "decision_opponent",
    "player__previous_round_decision": "decision_lag",
    "player__previous_round_decision_other": "decision_opponent_lag",
    "player__acc_steps_race": "acc_steps",
    "player__acc_steps_race_other": "acc_steps_opponent",
    "player__acc_steps_race_last_round": "acc_steps_lag",
    "player__acc_steps_race_other_last_round": "acc_steps_opponent_lag",
    "player__diff_acc_steps_last_round": "delta_steps_lag",
    "player__won_race": "won_race",
    "group__num_rounds": "num_rounds",
    "Sex": "sex",
    "risk_assessment__gamble_choice": "risk_gamble_choice",
}

AGE_BIN_EDGES = list(range(18, 74, 5))  # 18,23,28,...,68,73
AGE_BIN_LABELS = [f"{lo}-{hi - 1}" for lo, hi in zip(AGE_BIN_EDGES[:-1], AGE_BIN_EDGES[1:])]


def collapse_nationality(nationality: pd.Series) -> pd.Series:
    sentinels = {"DATA_EXPIRED", "CONSENT_REVOKED"}
    return nationality.where(
        nationality.isin({"South Africa", "Poland"} | sentinels),
        "Other",
    )


def bin_age(age_raw: pd.Series) -> pd.Series:
    sentinel_mask = age_raw.isin(["CONSENT_REVOKED"])
    age_numeric = pd.to_numeric(age_raw.where(~sentinel_mask), errors="coerce")
    binned = pd.cut(
        age_numeric,
        bins=AGE_BIN_EDGES,
        labels=AGE_BIN_LABELS,
        right=False,
        include_lowest=True,
    ).astype("object")
    binned[sentinel_mask] = "CONSENT_REVOKED"
    return binned


def anonymise_ids(df: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    participant_ids = df["participant__code"].unique()
    group_ids = df["group_id"].unique()

    participant_order = rng.permutation(len(participant_ids))
    group_order = rng.permutation(len(group_ids))

    participant_map = {
        old: f"P{i + 1:04d}"
        for old, i in zip(participant_ids, participant_order)
    }
    group_map = {
        old: f"G{i + 1:04d}"
        for old, i in zip(group_ids, group_order)
    }

    mapping_df = pd.DataFrame(
        {
            "participant__code": list(participant_map.keys()),
            "participant_id": list(participant_map.values()),
        }
    )
    mapping_df.to_csv(ID_MAPPING_CSV, index=False)

    df = df.copy()
    df["participant_id"] = df["participant__code"].map(participant_map)
    df["group_id"] = df["group_id"].map(group_map)
    return df


def main() -> None:
    rng = np.random.default_rng(RNG_SEED)

    raw = pd.read_pickle(SOURCE_PATH)
    print(f"Loaded source: {raw.shape[0]} rows, {raw['participant__code'].nunique()} participants")

    df = anonymise_ids(raw, rng)

    df["nationality_group"] = collapse_nationality(df["Nationality"])
    df["age"] = bin_age(df["Age"])

    keep_source_cols = list(RENAME_MAP.keys())
    out = df[["participant_id", "group_id"] + keep_source_cols + ["nationality_group", "age"]].rename(
        columns=RENAME_MAP
    )

    column_order = [
        "participant_id", "group_id", "round_number", "max_private_risk",
        "decision", "decision_opponent", "decision_lag", "decision_opponent_lag",
        "acc_steps", "acc_steps_opponent", "acc_steps_lag", "acc_steps_opponent_lag",
        "delta_steps_lag", "won_race", "num_rounds",
        "sex", "age", "nationality_group", "risk_gamble_choice",
    ]
    out = out[column_order].sort_values(["participant_id", "round_number"]).reset_index(drop=True)

    assert "prolific_id" not in out.columns
    assert out.shape[1] == 19

    out.to_csv(OUTPUT_CSV, index=False)
    print(f"Wrote {OUTPUT_CSV}: {out.shape[0]} rows, {out.shape[1]} columns, "
          f"{out['participant_id'].nunique()} participants, {out['group_id'].nunique()} pairs")
    print(f"(Local-only, DO NOT DEPOSIT: {ID_MAPPING_CSV})")


if __name__ == "__main__":
    main()
