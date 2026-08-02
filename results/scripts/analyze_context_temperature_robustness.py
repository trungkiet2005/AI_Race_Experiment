#!/usr/bin/env python3
"""Compare paired context-skin live pilots at temperature 0.0 and 0.7.

The two decoding conditions are never pooled. Temperature-zero repetitions are
treated as common-random-number environment seeds, not independent model draws.
All inferential intervals resample CRN repetition streams. Risk conditions
reuse the same ``base_seed + rep`` stream and are not independent clusters.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


BOOTSTRAP_SEED = 260802
BLUE = "#2563EB"
BLUE_LIGHT = "#AFC8FB"
GOLD = "#D79B00"
ORANGE = "#E8792E"
INK = "#172033"
MUTED = "#687386"
GRID = "#D9E0EA"
PAPER = "#FBFCFE"
WHITE = "#FFFFFF"
CMAP = LinearSegmentedColormap.from_list(
    "temperature_delta", [ORANGE, "#FFF9EB", BLUE]
)

SKIN_ORDER = [
    "abstract_contest",
    "technology_race",
    "logistics_contract",
    "crystal_guild_contract",
    "hospital_deployment",
    "colony_life_support",
    "robotic_expedition",
    "fictional_cartography",
]
SKIN_LABEL = {
    "abstract_contest": "Abstract control",
    "technology_race": "Technology race",
    "logistics_contract": "Logistics contract",
    "crystal_guild_contract": "Crystal guild",
    "hospital_deployment": "Hospital deployment",
    "colony_life_support": "Colony life support",
    "robotic_expedition": "Robotic expedition",
    "fictional_cartography": "Fictional cartography",
}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("results/open_source/context_skin_pilot"),
    )
    parser.add_argument("--t0-root", type=Path, default=None)
    parser.add_argument("--t07-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--bootstrap-repetitions", type=int, default=10_000)
    return parser.parse_args(argv)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def save_csv(frame: pd.DataFrame, path: Path) -> None:
    frame.to_csv(path, index=False, quoting=csv.QUOTE_MINIMAL, lineterminator="\n")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "text.color": INK,
            "figure.facecolor": WHITE,
            "axes.facecolor": PAPER,
            "axes.grid": True,
            "grid.color": GRID,
            "grid.alpha": 0.65,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".png"), dpi=240, bbox_inches="tight", facecolor=WHITE)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)


def percentile_ci(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=float)
    array = array[np.isfinite(array)]
    if not len(array):
        return float("nan"), float("nan")
    low, high = np.quantile(array, [0.025, 0.975])
    return float(low), float(high)


def cluster_bootstrap_mean(
    frame: pd.DataFrame,
    value: str,
    cluster: str,
    repetitions: int,
    seed_offset: int = 0,
) -> tuple[float, float]:
    clusters = list(frame[cluster].drop_duplicates())
    if not clusters:
        return float("nan"), float("nan")
    grouped = frame.groupby(cluster, sort=False)[value].agg(["sum", "count"])
    sums = grouped["sum"].to_numpy(float)
    counts = grouped["count"].to_numpy(float)
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    sampled = rng.integers(0, len(clusters), size=(repetitions, len(clusters)))
    means = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    return percentile_ci(means)


def independent_cluster_bootstrap_difference(
    frame: pd.DataFrame,
    value: str,
    group: str,
    group_a: str,
    group_b: str,
    cluster: str,
    repetitions: int,
    seed_offset: int = 0,
) -> tuple[float, float]:
    rng = np.random.default_rng(BOOTSTRAP_SEED + seed_offset)
    parts: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for name in (group_a, group_b):
        subset = frame.loc[frame[group] == name]
        stats = subset.groupby(cluster, sort=False)[value].agg(["sum", "count"])
        parts[name] = (
            stats["sum"].to_numpy(float),
            stats["count"].to_numpy(float),
        )
    means: dict[str, np.ndarray] = {}
    for name in (group_a, group_b):
        sums, counts = parts[name]
        sampled = rng.integers(0, len(sums), size=(repetitions, len(sums)))
        means[name] = sums[sampled].sum(axis=1) / counts[sampled].sum(axis=1)
    draws = means[group_a] - means[group_b]
    return percentile_ci(draws)


def spearman_rho(x: Sequence[float], y: Sequence[float]) -> float:
    xs = pd.Series(x, dtype=float)
    ys = pd.Series(y, dtype=float)
    if xs.nunique(dropna=True) < 2 or ys.nunique(dropna=True) < 2:
        return float("nan")
    return float(xs.rank(method="average").corr(ys.rank(method="average")))


def sign_code(value: float, tolerance: float = 1e-12) -> int:
    if abs(value) <= tolerance:
        return 0
    return 1 if value > 0 else -1


def mapping_for_rep(rep: int) -> str:
    return "safe_p" if int(rep) % 2 == 0 else "safe_q"


def risk_label(max_private_risk: float) -> str:
    return f"risk_{round(100 * float(max_private_risk)):02d}"


def load_condition(root: Path, expected_temperature: float) -> dict[str, Any]:
    run_manifests: list[dict[str, Any]] = []
    lane_manifests: list[dict[str, Any]] = []
    turns_frames: list[pd.DataFrame] = []
    races_frames: list[pd.DataFrame] = []
    players_frames: list[pd.DataFrame] = []

    for lane_dir in sorted(root.glob("lane_*")):
        lane_manifest = read_json(lane_dir / "lane_manifest.json")
        lane_manifests.append(lane_manifest)
        for skin_dir in sorted(path for path in lane_dir.iterdir() if path.is_dir()):
            manifest = read_json(skin_dir / "run_manifest.json")
            run_manifests.append(manifest)
            skin = manifest["context_skin"]["id"]
            temperature = float(manifest["decoding"]["temperature"])
            if temperature != expected_temperature:
                raise ValueError(
                    f"expected temperature {expected_temperature}, got {temperature} in {skin_dir}"
                )

            turn_records = read_jsonl(skin_dir / "turns.jsonl")
            turns = pd.DataFrame(turn_records)
            races = pd.read_csv(skin_dir / "races.csv")
            players = pd.read_csv(skin_dir / "players.csv")
            for frame in (turns, races, players):
                frame["skin"] = skin
                frame["lane"] = manifest["lane"]
                frame["temperature"] = temperature
                frame["mapping"] = frame["rep"].map(mapping_for_rep)
                frame["risk"] = frame["max_private_risk"].map(risk_label)
                # All risk treatments reuse base_seed + rep.  The independent
                # CRN unit is therefore the repetition stream, not risk x rep.
                frame["crn_cluster"] = frame["rep"].map(
                    lambda rep: f"rep{int(rep):04d}"
                )
            turns_frames.append(turns)
            races_frames.append(races)
            players_frames.append(players)

    turns = pd.concat(turns_frames, ignore_index=True)
    races = pd.concat(races_frames, ignore_index=True)
    players = pd.concat(players_frames, ignore_index=True)
    return {
        "root": root,
        "temperature": expected_temperature,
        "lane_manifests": lane_manifests,
        "run_manifests": run_manifests,
        "turns": turns,
        "races": races,
        "players": players,
    }


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def validate_compatibility(t0: dict[str, Any], t07: dict[str, Any]) -> dict[str, Any]:
    validations: list[dict[str, Any]] = []

    def add(name: str, left: Any, right: Any, required: bool = True) -> None:
        match = canonical_json(left) == canonical_json(right)
        validations.append(
            {
                "field": name,
                "t0": canonical_json(left),
                "t07": canonical_json(right),
                "match": match,
                "required": required,
            }
        )
        if required and not match:
            raise ValueError(f"required compatibility field differs: {name}")

    t0_runs = {m["context_skin"]["id"]: m for m in t0["run_manifests"]}
    t07_runs = {m["context_skin"]["id"]: m for m in t07["run_manifests"]}
    add("skin_set", sorted(t0_runs), sorted(t07_runs))
    if sorted(t0_runs) != sorted(SKIN_ORDER):
        raise ValueError("context skin coverage is incomplete")

    representative0 = t0_runs[SKIN_ORDER[0]]
    representative7 = t07_runs[SKIN_ORDER[0]]
    for field in (
        "model",
        "ollama_model",
        "experiment_config_sha256",
        "effective_experiment_sha256",
        "mechanism_sha256",
        "game_config_sha256",
        "agents_config_sha256",
        "persona_sha256",
        "crn",
        "action_code_factor",
        "base_seed",
        "profile",
    ):
        add(field, representative0[field], representative7[field])
    add(
        "whole_source_sha256",
        representative0["source_sha256"],
        representative7["source_sha256"],
        required=False,
    )
    for skin in SKIN_ORDER:
        add(
            f"template_hashes:{skin}",
            t0_runs[skin]["context_skin"]["template_sha256_by_action_mapping"],
            t07_runs[skin]["context_skin"]["template_sha256_by_action_mapping"],
        )
        add(
            f"lane_assignment:{skin}", t0_runs[skin]["lane"], t07_runs[skin]["lane"]
        )

    lanes0 = {m["lane"]: m for m in t0["lane_manifests"]}
    lanes7 = {m["lane"]: m for m in t07["lane_manifests"]}
    for lane in sorted(lanes0):
        for field in ("hostname", "gpu_name", "ollama_version"):
            add(f"lane_{lane}:{field}", lanes0[lane][field], lanes7[lane][field])

    for label, condition in (("t0", t0), ("t07", t07)):
        if len(condition["races"]) != 768 or len(condition["players"]) != 1536:
            raise ValueError(f"{label} has incomplete race/player coverage")
        if int(condition["races"]["parse_failures"].sum()) != 0:
            raise ValueError(f"{label} contains parse failures")
        if int(condition["turns"]["parse_failed"].sum()) != 0:
            raise ValueError(f"{label} contains turn parse failures")
        if int(condition["turns"]["retry_count"].sum()) != 0:
            raise ValueError(f"{label} contains retried decisions")
        if not condition["turns"]["raw_response"].isin(
            ["P", "Q", "ACTION: P", "ACTION: Q"]
        ).all():
            raise ValueError(f"{label} contains invalid opaque action codes")

    return {
        "status": "pass_with_source_provenance_warning",
        "checks": validations,
        "model_mechanism_configuration_match": all(
            row["match"] for row in validations if row["required"]
        ),
        "whole_source_sha256_match": next(
            row["match"] for row in validations if row["field"] == "whole_source_sha256"
        ),
        "source_warning": (
            "Whole staged-source hashes differ. Mechanism, model, experiment, game, prompt-template, CRN, lane, hardware, and runtime hashes/fields match exactly."
        ),
    }


def paired_frames(t0: dict[str, Any], t07: dict[str, Any]) -> dict[str, pd.DataFrame]:
    turn_keys = ["skin", "risk", "rep", "round", "player_index"]
    turns = t0["turns"].merge(
        t07["turns"],
        on=turn_keys,
        suffixes=("_t0", "_t07"),
        validate="one_to_one",
    )
    if len(turns) != len(t0["turns"]) or len(turns) != len(t07["turns"]):
        raise ValueError("turn pairing is incomplete")
    for field in ("game_seed", "sampling_seed", "mapping", "crn_cluster"):
        if not (turns[f"{field}_t0"] == turns[f"{field}_t07"]).all():
            raise ValueError(f"paired turn field differs: {field}")
    first = turns.loc[turns["round"] == 1]
    if not (first["prompt_t0"] == first["prompt_t07"]).all():
        raise ValueError("first-round prompts differ across temperature conditions")
    turns["mapping"] = turns["mapping_t0"]
    turns["crn_cluster"] = turns["crn_cluster_t0"]
    turns["action_agree"] = (turns["action_t0"] == turns["action_t07"]).astype(int)
    turns["unsafe_delta"] = turns["unsafe_t07"] - turns["unsafe_t0"]

    race_keys = ["skin", "risk", "rep"]
    races = t0["races"].merge(
        t07["races"],
        on=race_keys,
        suffixes=("_t0", "_t07"),
        validate="one_to_one",
    )
    if len(races) != 768:
        raise ValueError("race pairing is incomplete")
    for field in ("game_seed", "n_rounds", "stop_draws", "mapping", "crn_cluster"):
        if not (races[f"{field}_t0"] == races[f"{field}_t07"]).all():
            raise ValueError(f"paired race CRN field differs: {field}")
    races["mapping"] = races["mapping_t0"]
    races["crn_cluster"] = races["crn_cluster_t0"]
    races["total_payoff_t0"] = (
        races["player_1_final_payoff_t0"] + races["player_2_final_payoff_t0"]
    )
    races["total_payoff_t07"] = (
        races["player_1_final_payoff_t07"] + races["player_2_final_payoff_t07"]
    )
    races["total_payoff_delta"] = races["total_payoff_t07"] - races["total_payoff_t0"]
    races["setback_count_t0"] = races["player_1_setback_t0"] + races["player_2_setback_t0"]
    races["setback_count_t07"] = races["player_1_setback_t07"] + races["player_2_setback_t07"]

    player_keys = ["skin", "risk", "rep", "player_index"]
    players = t0["players"].merge(
        t07["players"],
        on=player_keys,
        suffixes=("_t0", "_t07"),
        validate="one_to_one",
    )
    if len(players) != 1536:
        raise ValueError("player pairing is incomplete")
    for field in ("game_seed", "n_rounds", "mapping", "crn_cluster"):
        if not (players[f"{field}_t0"] == players[f"{field}_t07"]).all():
            raise ValueError(f"paired player field differs: {field}")
    players["mapping"] = players["mapping_t0"]
    players["crn_cluster"] = players["crn_cluster_t0"]
    players["unsafe_rate_t0"] = players["unsafe_frequency_t0"]
    players["unsafe_rate_t07"] = players["unsafe_frequency_t07"]
    players["full_unsafe_delta"] = players["unsafe_rate_t07"] - players["unsafe_rate_t0"]
    players["payoff_delta"] = players["final_payoff_t07"] - players["final_payoff_t0"]

    player_turn = (
        turns.groupby(player_keys, as_index=False)
        .agg(
            first_unsafe_t0=("unsafe_t0", "first"),
            first_unsafe_t07=("unsafe_t07", "first"),
            action_agreement=("action_agree", "mean"),
            exact_player_trajectory=("action_agree", "min"),
            n_decisions=("action_agree", "size"),
        )
    )
    player_turn["first_unsafe_delta"] = (
        player_turn["first_unsafe_t07"] - player_turn["first_unsafe_t0"]
    )
    players = players.merge(player_turn, on=player_keys, validate="one_to_one")

    race_agreement = (
        turns.groupby(race_keys, as_index=False)
        .agg(
            action_agreement=("action_agree", "mean"),
            exact_joint_trajectory=("action_agree", "min"),
            n_decisions=("action_agree", "size"),
        )
    )
    races = races.merge(race_agreement, on=race_keys, validate="one_to_one")
    return {"turns": turns, "players": players, "races": races}


def summarize_player_pairs(
    players: pd.DataFrame,
    group_fields: list[str],
    repetitions: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = [((), players)] if not group_fields else players.groupby(group_fields, sort=False)
    for keys, part in grouped:
        if not isinstance(keys, tuple):
            keys = (keys,)
        row = dict(zip(group_fields, keys))
        row.update(
            {
                "n_player_races": len(part),
                "n_crn_clusters": part["crn_cluster"].nunique(),
                "first_unsafe_rate_t0": part["first_unsafe_t0"].mean(),
                "first_unsafe_rate_t07": part["first_unsafe_t07"].mean(),
                "first_unsafe_delta_t07_minus_t0": part["first_unsafe_delta"].mean(),
                "full_unsafe_rate_t0": part["unsafe_rate_t0"].mean(),
                "full_unsafe_rate_t07": part["unsafe_rate_t07"].mean(),
                "full_unsafe_delta_t07_minus_t0": part["full_unsafe_delta"].mean(),
                "mean_action_agreement": part["action_agreement"].mean(),
                "exact_player_trajectory_rate": part["exact_player_trajectory"].mean(),
                "mean_payoff_delta_t07_minus_t0": part["payoff_delta"].mean(),
            }
        )
        for metric in ("first_unsafe_delta", "full_unsafe_delta", "payoff_delta"):
            lo, hi = cluster_bootstrap_mean(
                part, metric, "crn_cluster", repetitions, len(rows) * 10 + len(row)
            )
            prefix = {
                "first_unsafe_delta": "first_delta",
                "full_unsafe_delta": "full_delta",
                "payoff_delta": "payoff_delta",
            }[metric]
            row[f"{prefix}_ci95_low"] = lo
            row[f"{prefix}_ci95_high"] = hi
        rows.append(row)
    return pd.DataFrame(rows)


def summarize_race_agreement(races: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for skin, part in races.groupby("skin"):
        rows.append(
            {
                "skin": skin,
                "n_races": len(part),
                "mean_decision_agreement": part["action_agreement"].mean(),
                "exact_joint_trajectory_rate": part["exact_joint_trajectory"].mean(),
                "mean_total_payoff_delta_t07_minus_t0": part["total_payoff_delta"].mean(),
                "setback_count_t0": int(part["setback_count_t0"].sum()),
                "setback_count_t07": int(part["setback_count_t07"].sum()),
            }
        )
    return pd.DataFrame(rows)


def mapping_interactions(
    players: pd.DataFrame, repetitions: int
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    scopes: list[tuple[str, pd.DataFrame]] = [("all_contexts", players)]
    scopes.extend((skin, part) for skin, part in players.groupby("skin"))
    for scope, part in scopes:
        for metric, label in (
            ("first_unsafe_delta", "first_round"),
            ("full_unsafe_delta", "full_trajectory"),
        ):
            means = part.groupby("mapping")[metric].mean()
            lo, hi = independent_cluster_bootstrap_difference(
                part,
                metric,
                "mapping",
                "safe_p",
                "safe_q",
                "crn_cluster",
                repetitions,
                5000 + len(rows),
            )
            rows.append(
                {
                    "scope": scope,
                    "estimand": label,
                    "n_safe_p_clusters": part.loc[
                        part["mapping"] == "safe_p", "crn_cluster"
                    ].nunique(),
                    "n_safe_q_clusters": part.loc[
                        part["mapping"] == "safe_q", "crn_cluster"
                    ].nunique(),
                    "temperature_delta_safe_p": means["safe_p"],
                    "temperature_delta_safe_q": means["safe_q"],
                    "mapping_interaction_safe_p_minus_safe_q": means["safe_p"]
                    - means["safe_q"],
                    "interaction_ci95_low": lo,
                    "interaction_ci95_high": hi,
                    "claim_boundary": "diagnostic; mapping is assigned by repetition parity",
                }
            )
    return pd.DataFrame(rows)


def context_effect_rows(players: pd.DataFrame, temperature: str) -> pd.DataFrame:
    source = players[
        [
            "skin",
            "risk",
            "rep",
            "player_index",
            "mapping",
            "crn_cluster",
            f"first_unsafe_{temperature}",
            f"unsafe_rate_{temperature}",
        ]
    ].copy()
    abstract = source.loc[source["skin"] == "abstract_contest"].drop(columns="skin")
    abstract = abstract.rename(
        columns={
            f"first_unsafe_{temperature}": "abstract_first",
            f"unsafe_rate_{temperature}": "abstract_full",
        }
    )
    noncontrol = source.loc[source["skin"] != "abstract_contest"]
    paired = noncontrol.merge(
        abstract,
        on=["risk", "rep", "player_index", "mapping", "crn_cluster"],
        validate="many_to_one",
    )
    paired["first_context_effect"] = (
        paired[f"first_unsafe_{temperature}"] - paired["abstract_first"]
    )
    paired["full_context_effect"] = (
        paired[f"unsafe_rate_{temperature}"] - paired["abstract_full"]
    )
    paired["temperature"] = temperature
    return paired


def context_effect_stability(
    players: pd.DataFrame, repetitions: int
) -> tuple[pd.DataFrame, dict[str, Any]]:
    effects0 = context_effect_rows(players, "t0")
    effects7 = context_effect_rows(players, "t07")
    keys = ["skin", "risk", "rep", "player_index", "mapping", "crn_cluster"]
    paired = effects0.merge(
        effects7,
        on=keys,
        suffixes=("_t0", "_t07"),
        validate="one_to_one",
    )
    rows: list[dict[str, Any]] = []
    for skin, part in paired.groupby("skin"):
        row: dict[str, Any] = {
            "skin": skin,
            "n_player_races": len(part),
            "n_crn_clusters": part["crn_cluster"].nunique(),
        }
        for metric in ("first", "full"):
            col0 = f"{metric}_context_effect_t0"
            col7 = f"{metric}_context_effect_t07"
            delta_col = f"{metric}_effect_change"
            part = part.copy()
            part[delta_col] = part[col7] - part[col0]
            lo, hi = cluster_bootstrap_mean(
                part, delta_col, "crn_cluster", repetitions, 7000 + len(rows) * 3
            )
            effect0 = part[col0].mean()
            effect7 = part[col7].mean()
            row.update(
                {
                    f"{metric}_effect_t0": effect0,
                    f"{metric}_effect_t07": effect7,
                    f"{metric}_effect_change_t07_minus_t0": effect7 - effect0,
                    f"{metric}_effect_change_ci95_low": lo,
                    f"{metric}_effect_change_ci95_high": hi,
                    f"{metric}_sign_t0": sign_code(effect0),
                    f"{metric}_sign_t07": sign_code(effect7),
                    f"{metric}_sign_stable": sign_code(effect0) == sign_code(effect7),
                }
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    rank_summary: dict[str, Any] = {}
    for metric in ("first", "full"):
        rank_summary[metric] = {
            "spearman_rho": spearman_rho(
                summary[f"{metric}_effect_t0"], summary[f"{metric}_effect_t07"]
            ),
            "sign_agreement_rate": summary[f"{metric}_sign_stable"].mean(),
            "n_contexts": len(summary),
            "undefined_rank_reason": (
                "At least one temperature has tied/constant effects."
                if not np.isfinite(
                    spearman_rho(
                        summary[f"{metric}_effect_t0"],
                        summary[f"{metric}_effect_t07"],
                    )
                )
                else None
            ),
        }
    return summary, rank_summary


def first_round_replication_audit(t0: dict[str, Any], t07: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, condition in (("t0", t0), ("t07", t07)):
        first = condition["turns"].loc[condition["turns"]["round"] == 1].copy()
        first["prompt_sha256"] = first["prompt"].map(
            lambda text: hashlib.sha256(text.encode("utf-8")).hexdigest()
        )
        for (skin, risk, mapping, player_index), part in first.groupby(
            ["skin", "risk", "mapping", "player_index"]
        ):
            rows.append(
                {
                    "temperature": label,
                    "skin": skin,
                    "risk": risk,
                    "mapping": mapping,
                    "player_index": int(player_index),
                    "n_rows": len(part),
                    "n_unique_prompts": part["prompt_sha256"].nunique(),
                    "n_unique_actions": part["action"].nunique(),
                    "unsafe_rate": part["unsafe"].mean(),
                    "interpretation": (
                        "deterministic duplicate prompt evaluations; not independent model draws"
                        if label == "t0"
                        else "seeded stochastic draws for one unique prompt per cell"
                    ),
                }
            )
    return pd.DataFrame(rows)


def temperature_delta_figure(context: pd.DataFrame, figures: Path) -> None:
    plot = context.set_index("skin").loc[SKIN_ORDER].reset_index()
    y = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(12.4, 5.5), sharey=True)
    specs = [
        (
            "first_unsafe_delta_t07_minus_t0",
            "first_delta_ci95_low",
            "first_delta_ci95_high",
            "First round",
        ),
        (
            "full_unsafe_delta_t07_minus_t0",
            "full_delta_ci95_low",
            "full_delta_ci95_high",
            "Full player trajectory",
        ),
    ]
    for ax, (metric, low, high, title) in zip(axes, specs):
        ax.hlines(y, 100 * plot[low], 100 * plot[high], color=BLUE_LIGHT, linewidth=3)
        ax.scatter(100 * plot[metric], y, color=BLUE, edgecolor=INK, s=64, zorder=3)
        ax.axvline(0, color=INK, linewidth=1.2)
        ax.set_yticks(y, [SKIN_LABEL[s] for s in plot["skin"]])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Unsafe-rate change: temp 0.7 minus temp 0.0 (pp)")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Paired decoding-temperature behavior shifts",
        x=0.08,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.91,
        "Intervals resample CRN race clusters; temperature-zero repeats are environment seeds, not model draws.",
        color=MUTED,
    )
    fig.subplots_adjust(top=0.80, wspace=0.22)
    save_figure(fig, figures / "temperature_unsafe_delta_by_context")


def agreement_figure(context: pd.DataFrame, races: pd.DataFrame, figures: Path) -> None:
    plot = context.merge(races, on="skin", validate="one_to_one")
    plot = plot.set_index("skin").loc[SKIN_ORDER].reset_index()
    y = np.arange(len(plot))
    fig, ax = plt.subplots(figsize=(10.8, 5.6))
    ax.scatter(
        100 * plot["mean_action_agreement"],
        y - 0.16,
        color=BLUE,
        marker="o",
        edgecolor=INK,
        s=62,
        label="Mean player action agreement",
    )
    ax.scatter(
        100 * plot["exact_player_trajectory_rate"],
        y,
        color=GOLD,
        marker="s",
        edgecolor=INK,
        s=58,
        label="Exact player trajectory",
    )
    ax.scatter(
        100 * plot["exact_joint_trajectory_rate"],
        y + 0.16,
        color=ORANGE,
        marker="^",
        edgecolor=INK,
        s=64,
        label="Exact two-player race trajectory",
    )
    ax.set_yticks(y, [SKIN_LABEL[s] for s in plot["skin"]])
    ax.invert_yaxis()
    ax.set_xlim(0, 102)
    ax.set_xlabel("Agreement between temperature 0.0 and 0.7 (%)")
    ax.set_title("Action and trajectory agreement", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right")
    save_figure(fig, figures / "temperature_trajectory_agreement")


def mapping_heatmap_figure(context_mapping: pd.DataFrame, figures: Path) -> None:
    plot = context_mapping.copy()
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 5.6), sharey=True)
    specs = [
        ("first_unsafe_delta_t07_minus_t0", "First-round delta"),
        ("full_unsafe_delta_t07_minus_t0", "Full-trajectory delta"),
    ]
    vmax = max(abs(plot[metric]).max() for metric, _ in specs) * 100
    vmax = max(vmax, 1.0)
    image = None
    for ax, (metric, title) in zip(axes, specs):
        matrix = (
            plot.pivot(index="skin", columns="mapping", values=metric)
            .reindex(SKIN_ORDER)
            .reindex(columns=["safe_p", "safe_q"])
            * 100
        )
        image = ax.imshow(matrix, cmap=CMAP, vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks([0, 1], ["SAFE=P", "SAFE=Q"])
        ax.set_yticks(range(len(SKIN_ORDER)), [SKIN_LABEL[s] for s in SKIN_ORDER])
        ax.set_title(title, fontweight="bold")
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                value = matrix.iloc[row, column]
                ax.text(column, row, f"{value:+.1f}", ha="center", va="center", color=INK)
        ax.grid(False)
    assert image is not None
    colorbar = fig.colorbar(image, ax=axes, fraction=0.035, pad=0.03)
    colorbar.set_label("Temp 0.7 minus 0.0 Unsafe rate (pp)")
    fig.suptitle(
        "Temperature shift by opaque action mapping",
        x=0.08,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.91,
        "Mapping is assigned by repetition parity; cross-mapping interactions remain diagnostic.",
        color=MUTED,
    )
    fig.subplots_adjust(top=0.80, right=0.86, wspace=0.18)
    save_figure(fig, figures / "temperature_mapping_interaction_heatmap")


def effect_stability_figure(effects: pd.DataFrame, figures: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.2))
    for ax, metric, title in (
        (axes[0], "first", "First-round context effects"),
        (axes[1], "full", "Full-trajectory context effects"),
    ):
        x = 100 * effects[f"{metric}_effect_t0"]
        y = 100 * effects[f"{metric}_effect_t07"]
        low = min(float(x.min()), float(y.min()), 0.0)
        high = max(float(x.max()), float(y.max()), 0.0)
        padding = max((high - low) * 0.15, 0.5)
        limits = (low - padding, high + padding)
        ax.plot(limits, limits, color=INK, linestyle="--", linewidth=1)
        ax.scatter(x, y, color=BLUE, edgecolor=INK, s=66)
        for row in effects.itertuples():
            ax.annotate(
                SKIN_LABEL[row.skin],
                (100 * getattr(row, f"{metric}_effect_t0"), 100 * getattr(row, f"{metric}_effect_t07")),
                xytext=(5, 4),
                textcoords="offset points",
                fontsize=8,
                color=MUTED,
            )
        ax.set_xlim(*limits)
        ax.set_ylim(*limits)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Context effect at temp 0.0 (pp)")
        ax.set_ylabel("Context effect at temp 0.7 (pp)")
        ax.set_title(title, loc="left", fontweight="bold")
    fig.suptitle(
        "Context-effect rank and sign stability",
        x=0.08,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(
        0.08,
        0.91,
        "Each effect is context minus abstract control within the same temperature and CRN race key.",
        color=MUTED,
    )
    fig.subplots_adjust(top=0.80, wspace=0.32)
    save_figure(fig, figures / "context_effect_temperature_stability")


def effect_change_figure(effects: pd.DataFrame, figures: Path) -> None:
    plot = effects.set_index("skin").loc[[s for s in SKIN_ORDER if s != "abstract_contest"]].reset_index()
    y = np.arange(len(plot))
    fig, axes = plt.subplots(1, 2, figsize=(12.1, 5.3), sharey=True)
    for ax, metric, title in (
        (axes[0], "first", "First-round effect change"),
        (axes[1], "full", "Full-trajectory effect change"),
    ):
        ax.hlines(
            y,
            100 * plot[f"{metric}_effect_change_ci95_low"],
            100 * plot[f"{metric}_effect_change_ci95_high"],
            color=BLUE_LIGHT,
            linewidth=3,
        )
        ax.scatter(
            100 * plot[f"{metric}_effect_change_t07_minus_t0"],
            y,
            color=BLUE,
            edgecolor=INK,
            s=64,
            zorder=3,
        )
        ax.axvline(0, color=INK, linewidth=1.2)
        ax.set_yticks(y, [SKIN_LABEL[s] for s in plot["skin"]])
        ax.invert_yaxis()
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlabel("Change in context-minus-abstract effect (pp)")
        ax.grid(axis="y", visible=False)
    fig.suptitle(
        "Does temperature change the context effect?",
        x=0.08,
        ha="left",
        fontsize=16,
        fontweight="bold",
    )
    fig.text(0.08, 0.91, "Race-cluster bootstrap 95% intervals.", color=MUTED)
    fig.subplots_adjust(top=0.80, wspace=0.22)
    save_figure(fig, figures / "context_effect_temperature_change")


def make_inventory(roots: list[Path]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for root in roots:
        for path in sorted(root.rglob("*")):
            if path.is_file() and path.suffix.lower() != ".tgz":
                rows.append(
                    {
                        "root": root.name,
                        "path": path.relative_to(root).as_posix(),
                        "bytes": path.stat().st_size,
                        "sha256": sha256_file(path),
                    }
                )
    return pd.DataFrame(rows)


def to_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_jsonable(item) for item in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def build_report(
    validation: dict[str, Any],
    overall: pd.DataFrame,
    context: pd.DataFrame,
    race_agreement: pd.DataFrame,
    interactions: pd.DataFrame,
    effects: pd.DataFrame,
    rank_summary: dict[str, Any],
    comprehension: dict[str, Any],
) -> str:
    row = overall.iloc[0]
    context_join = context.merge(race_agreement, on="skin", validate="one_to_one")
    context_lines = "\n".join(
        f"| {SKIN_LABEL[r.skin]} | {100*r.first_unsafe_delta_t07_minus_t0:+.2f} pp | {100*r.full_unsafe_delta_t07_minus_t0:+.2f} pp | {100*r.mean_action_agreement:.1f}% | {100*r.exact_joint_trajectory_rate:.1f}% | {r.mean_payoff_delta_t07_minus_t0:+.2f} |"
        for r in context_join.set_index("skin").loc[SKIN_ORDER].reset_index().itertuples()
    )
    full_effect = effects.sort_values("full_effect_t07", ascending=False)
    effect_lines = "\n".join(
        f"| {SKIN_LABEL[r.skin]} | {100*r.full_effect_t0:+.2f} pp | {100*r.full_effect_t07:+.2f} pp | {100*r.full_effect_change_t07_minus_t0:+.2f} pp | {'yes' if r.full_sign_stable else 'no'} |"
        for r in full_effect.itertuples()
    )
    all_interaction = interactions.loc[interactions["scope"] == "all_contexts"]
    first_rank = rank_summary["first"]
    full_rank = rank_summary["full"]
    first_rho_text = (
        f"{first_rank['spearman_rho']:.3f}"
        if first_rank["spearman_rho"] is not None
        and np.isfinite(first_rank["spearman_rho"])
        else "undefined"
    )
    full_rho_text = (
        f"{full_rank['spearman_rho']:.3f}"
        if full_rank["spearman_rho"] is not None
        and np.isfinite(full_rank["spearman_rho"])
        else "undefined"
    )
    source_match = validation["whole_source_sha256_match"]
    return f"""# Context-skin decoding-temperature robustness audit

## Bottom line

Temperature 0.7 changed full-trajectory Unsafe behavior by {100*row.full_unsafe_delta_t07_minus_t0:+.2f} percentage points relative to temperature 0.0 (race-cluster 95% interval [{100*row.full_delta_ci95_low:+.2f}, {100*row.full_delta_ci95_high:+.2f}] pp). The overall first-round change was {100*row.first_unsafe_delta_t07_minus_t0:+.2f} pp [{100*row.first_delta_ci95_low:+.2f}, {100*row.first_delta_ci95_high:+.2f}]. Mean action agreement was {100*row.mean_action_agreement:.1f}%, while only {100*row.exact_player_trajectory_rate:.1f}% of complete player trajectories remained identical.

This is a **diagnostic robustness comparison**, not a comprehension-admitted behavioral estimate. The common comprehension audit failed (`admission_passed={str(comprehension['admission_passed']).lower()}`; {comprehension['n_comprehension_rows']} probe rows), so neither temperature condition supports a claim that the model understood the game. Temperature-zero repetitions are common-random-number environment seeds, not independent stochastic model draws.

## Compatibility audit

- Exact model digest, mechanism hash, experiment/effective configuration hashes, three game hashes, prompt-template hashes, CRN contract, lane assignment, hostname, H100 class, Ollama version, base seed, and repetition count matched.
- First-round prompts matched exactly for all paired observations; race horizons and stop-draw streams matched for all 768 paired races.
- Whole staged-source SHA-256 match: `{str(source_match).lower()}`. The staged-source archive hashes differ, so this is not an exact whole-source replication. Mechanism-specific and template hashes do match; results are reported with this provenance warning rather than discarded.
- Both conditions had 768 races, 1,536 player-races, no parse failures, no retries, and only valid opaque `P`/`Q` responses.

## Paired context results

| Context | First-round delta | Full-trajectory delta | Mean action agreement | Exact joint trajectory | Mean player payoff delta |
|---|---:|---:|---:|---:|---:|
{context_lines}

![Temperature unsafe-rate deltas](figures/temperature_unsafe_delta_by_context.png)

![Action and trajectory agreement](figures/temperature_trajectory_agreement.png)

The first-round comparison is especially constrained: within each context/risk/mapping/seat cell, temperature 0.0 repeats the same deterministic response to one unique prompt. Its 32 repetitions must not be interpreted as 32 independent model samples. At later rounds, different CRN horizons and endogenous states provide environment variation, but still not temperature-zero decoding randomness.

## Opaque action mapping interaction

Across contexts, the first-round SAFE=P minus SAFE=Q interaction in the temperature shift was {100*all_interaction.loc[all_interaction.estimand == 'first_round', 'mapping_interaction_safe_p_minus_safe_q'].iloc[0]:+.2f} pp; the full-trajectory interaction was {100*all_interaction.loc[all_interaction.estimand == 'full_trajectory', 'mapping_interaction_safe_p_minus_safe_q'].iloc[0]:+.2f} pp. These interactions compare disjoint even/odd repetition seeds because mapping was assigned by repetition parity. They are diagnostic, not clean randomized mapping effects.

![Temperature mapping interaction](figures/temperature_mapping_interaction_heatmap.png)

## Context-effect rank and sign stability

Context effects are computed against the abstract control within each temperature using matched risk/repetition/seat keys.

| Context | Full effect at temp 0.0 | Full effect at temp 0.7 | Effect change | Sign stable |
|---|---:|---:|---:|---:|
{effect_lines}

The full-trajectory context-effect Spearman rank correlation was {full_rho_text}; sign agreement was {100*full_rank['sign_agreement_rate']:.1f}%. The first-round rank was {first_rho_text} because the context effects were tied/constant; first-round sign agreement was {100*first_rank['sign_agreement_rate']:.1f}%.

![Context-effect stability](figures/context_effect_temperature_stability.png)

![Context-effect changes](figures/context_effect_temperature_change.png)

## Claim boundary

Supported:

- Descriptive paired differences between these two exact decoding protocols on the observed CRN environment seeds.
- Direct first-round agreement/differences before endogenous state feedback.
- Full-trajectory divergence, mapping diagnostics, and payoff differences within this model digest and bounded eight-skin set.

Not supported:

- Independent repeated-sample uncertainty for deterministic temperature-zero outputs.
- Game understanding, strategic rationality, or an internal causal world model.
- A clean mapping main effect, because mapping is fixed by repetition parity.
- Generality across models, quantizations, prompt families, source snapshots, or temperatures other than 0.0 and 0.7.
- Pooling these two decoding conditions into one behavioral rate.

## Reproduce

```bash
python results/scripts/analyze_context_temperature_robustness.py
```

Raw pilot artifacts are untouched. `tables/source_artifact_inventory.csv` records SHA-256 checksums, figures are exported as PNG and vector PDF, and `analysis_summary.json` stores exact machine-readable findings.
"""


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    input_root = args.input_root.resolve()
    t0_root = (args.t0_root or input_root / "live_pilot_t0").resolve()
    t07_root = (args.t07_root or input_root / "live_pilot_t07").resolve()
    output_dir = (
        args.output_dir or input_root / "analysis_temperature_robustness"
    ).resolve()
    tables = output_dir / "tables"
    figures = output_dir / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    t0 = load_condition(t0_root, 0.0)
    t07 = load_condition(t07_root, 0.7)
    validation = validate_compatibility(t0, t07)
    paired = paired_frames(t0, t07)

    comprehension_summaries = []
    for name in ("analysis_live_pilot_t0", "analysis_live_pilot_t07"):
        summary = read_json(input_root / name / "analysis_summary.json")
        comprehension_summaries.append(summary)
    admission_values = {
        summary["validation"]["admission_passed"] for summary in comprehension_summaries
    }
    if admission_values != {False}:
        raise ValueError("expected the documented comprehension admission failure")
    comprehension = {
        "admission_passed": False,
        "n_comprehension_rows": comprehension_summaries[0]["validation"][
            "n_comprehension_rows"
        ],
        "claim_boundary": "diagnostic only; comprehension admission failed",
    }

    overall = summarize_player_pairs(
        paired["players"], [], args.bootstrap_repetitions
    )
    context = summarize_player_pairs(
        paired["players"], ["skin"], args.bootstrap_repetitions
    )
    context_mapping = summarize_player_pairs(
        paired["players"], ["skin", "mapping"], args.bootstrap_repetitions
    )
    context_risk = summarize_player_pairs(
        paired["players"], ["skin", "risk"], args.bootstrap_repetitions
    )
    race_agreement = summarize_race_agreement(paired["races"])
    interactions = mapping_interactions(
        paired["players"], args.bootstrap_repetitions
    )
    effects, rank_summary = context_effect_stability(
        paired["players"], args.bootstrap_repetitions
    )
    first_audit = first_round_replication_audit(t0, t07)

    compatibility_table = pd.DataFrame(validation["checks"])
    inventory = make_inventory([t0_root, t07_root])
    save_csv(compatibility_table, tables / "compatibility_audit.csv")
    save_csv(inventory, tables / "source_artifact_inventory.csv")
    save_csv(overall, tables / "paired_overall_summary.csv")
    save_csv(context, tables / "paired_context_summary.csv")
    save_csv(context_mapping, tables / "paired_context_mapping_summary.csv")
    save_csv(context_risk, tables / "paired_context_risk_summary.csv")
    save_csv(race_agreement, tables / "trajectory_agreement_by_context.csv")
    save_csv(interactions, tables / "mapping_interactions.csv")
    save_csv(effects, tables / "context_effect_stability.csv")
    save_csv(first_audit, tables / "first_round_replication_audit.csv")

    configure_matplotlib()
    temperature_delta_figure(context, figures)
    agreement_figure(context, race_agreement, figures)
    mapping_heatmap_figure(context_mapping, figures)
    effect_stability_figure(effects, figures)
    effect_change_figure(effects, figures)

    report = build_report(
        validation,
        overall,
        context,
        race_agreement,
        interactions,
        effects,
        rank_summary,
        comprehension,
    )
    (output_dir / "temperature_robustness_report.md").write_text(
        report, encoding="utf-8", newline="\n"
    )

    summary = {
        "schema_version": "ai-race.context-temperature-robustness.v1",
        "status": "complete",
        "claim_boundary": comprehension["claim_boundary"],
        "temperature_conditions_pooled": False,
        "temperature_zero_repetitions_interpretation": "CRN environment seeds, not independent model draws",
        "bootstrap": {
            "unit": "CRN repetition stream; risk strata share base_seed + rep",
            "repetitions": args.bootstrap_repetitions,
            "seed": BOOTSTRAP_SEED,
        },
        "validation": validation,
        "comprehension": comprehension,
        "counts": {
            "paired_races": len(paired["races"]),
            "paired_player_races": len(paired["players"]),
            "paired_decisions": len(paired["turns"]),
            "crn_clusters": paired["players"]["crn_cluster"].nunique(),
        },
        "overall": overall.to_dict(orient="records")[0],
        "context": context.to_dict(orient="records"),
        "mapping_interactions": interactions.to_dict(orient="records"),
        "context_effect_stability": effects.to_dict(orient="records"),
        "rank_sign_stability": rank_summary,
        "chart_contracts": [
            {
                "figure": "temperature_unsafe_delta_by_context",
                "question": "How much does Unsafe behavior change across decoding temperatures?",
                "family": "faceted interval plot",
            },
            {
                "figure": "temperature_trajectory_agreement",
                "question": "How often do actions and complete trajectories agree?",
                "family": "grouped dot plot",
            },
            {
                "figure": "temperature_mapping_interaction_heatmap",
                "question": "Does the temperature shift depend on opaque action mapping?",
                "family": "faceted heatmap",
            },
            {
                "figure": "context_effect_temperature_stability",
                "question": "Are context-effect ranks and signs stable?",
                "family": "labeled paired-estimate scatter",
            },
            {
                "figure": "context_effect_temperature_change",
                "question": "Which context effects materially change with temperature?",
                "family": "faceted interval plot",
            },
        ],
    }
    (output_dir / "analysis_summary.json").write_text(
        json.dumps(to_jsonable(summary), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    derived = [path for path in output_dir.rglob("*") if path.is_file()]
    analysis_manifest = {
        "schema_version": "ai-race.context-temperature-analysis-manifest.v1",
        "analysis_script": "results/scripts/analyze_context_temperature_robustness.py",
        "bootstrap_repetitions": args.bootstrap_repetitions,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "derived_artifacts": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in sorted(derived)
            if path.name != "analysis_manifest.json"
        },
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(analysis_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"Wrote temperature robustness analysis to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
