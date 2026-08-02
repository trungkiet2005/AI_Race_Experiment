#!/usr/bin/env python3
"""Fail-closed analysis for the GreenNode N-player ecology diagnostic.

The script treats a whole race/CRN cell as the analysis grain.  Turn-weighted
tables (including the evolving progress-rank view) are emitted only as clearly
labelled descriptive diagnostics.  The experiment is unadmitted by design;
neither the JSON summary nor the figures upgrade that evidence class.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from kaggle.experiments import greennode_nplayer_ecology as protocol


ANALYSIS_SCHEMA = "ai-race-nplayer-ecology-analysis-v1"
EVIDENCE_CLASS = "diagnostic_unadmitted"
MODEL_LABELS = {
    "qwen25_7b": "Qwen2.5-7B",
    "mistral7_01": "Mistral-7B",
}
MODEL_COLORS = {"qwen25_7b": "#2563EB", "mistral7_01": "#E76F51"}
MODULE_LABELS = {item.module_id: item.module_id.replace("_", " ").title() for item in protocol.MODULES}
PAIR_SPECS = (
    ("exact_transition_calculator", "char_length_placebo", "Calculator - placebo"),
    ("pair_alliance_label", "pair_alliance_placebo", "Pair label - placebo"),
    ("majority_alliance_label", "majority_alliance_placebo", "Majority label - placebo"),
    ("competitive_framing", "cooperative_framing", "Competitive - cooperative"),
    ("accurate_checkpoint_disclosure", "opaque_endpoint_ids", "Named - opaque endpoints"),
    ("neutral_per_capita_normalized", "neutral_fixed_total", "Normalized - fixed benefit"),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object: {path}")
    return value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError(f"Expected object at {path}:{number}")
        rows.append(value)
    return rows


def _require(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ValueError(f"{label}: expected {expected!r}, found {actual!r}")


def discover_runs(root: Path) -> list[Path]:
    candidates = sorted(path.parent for path in root.rglob("run_manifest.json"))
    runs = [path for path in candidates if load_json(path / "run_manifest.json").get("protocol") == protocol.PROTOCOL]
    if not runs:
        raise FileNotFoundError(f"No {protocol.PROTOCOL} runs below {root}")
    return runs


def _block_name(run_dir: Path, manifest: dict[str, Any], ordinal: int) -> str:
    return str(manifest.get("lane_block") or manifest.get("block") or run_dir.parent.name or f"block{ordinal}")


def _expected_game_ids() -> set[str]:
    return {
        protocol._game_id(module.module_id, n, risk, composition, rep)
        for module in protocol.MODULES
        for n in protocol.N_PLAYERS
        for risk in protocol.RISKS
        for composition in protocol.COMPOSITIONS
        for rep in range(protocol.REPETITIONS)
    }


def validate_run(run_dir: Path, block: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    manifest_path = run_dir / "run_manifest.json"
    manifest = load_json(manifest_path)
    _require(manifest.get("schema_version"), protocol.SCHEMA_VERSION, f"{block} schema")
    _require(manifest.get("protocol"), protocol.PROTOCOL, f"{block} protocol")
    _require(manifest.get("status"), "completed", f"{block} status")
    _require(manifest.get("evidence_class"), EVIDENCE_CLASS, f"{block} evidence")
    _require(manifest.get("exact_protocol_admission_passed"), False, f"{block} admission")
    _require(int(manifest.get("expected_races", -1)), protocol.expected_races(), f"{block} expected races")
    _require(int(manifest.get("n_races", -1)), protocol.expected_races(), f"{block} race count")

    design = manifest.get("design")
    if not isinstance(design, dict):
        raise ValueError(f"{block}: missing design")
    _require(tuple(design.get("n_players", [])), protocol.N_PLAYERS, f"{block} N grid")
    _require(tuple(design.get("risks", [])), protocol.RISKS, f"{block} risk grid")
    _require(tuple(design.get("compositions", [])), protocol.COMPOSITIONS, f"{block} compositions")
    _require(int(design.get("repetitions", -1)), protocol.REPETITIONS, f"{block} repetitions")
    _require(design.get("alliance_mechanism"), False, f"{block} alliance mechanism")
    _require(design.get("alliance_arms"), "prompt-framing-only", f"{block} alliance arms")
    observed_modules = tuple(item.get("module_id") for item in design.get("modules", []) if isinstance(item, dict))
    _require(observed_modules, tuple(item.module_id for item in protocol.MODULES), f"{block} modules")

    outputs = manifest.get("outputs", {})
    paths = {
        kind: run_dir / str(outputs.get(kind, f"{kind}.jsonl"))
        for kind in ("turns", "races", "players")
    }
    for kind, path in paths.items():
        if not path.is_file():
            raise FileNotFoundError(f"{block}: missing {kind} artifact {path}")
    artifacts = manifest.get("artifacts", {})
    if isinstance(artifacts, dict):
        for path in paths.values():
            if path.name in artifacts:
                _require(sha256_file(path), artifacts[path.name], f"{block} {path.name} hash")

    races = pd.DataFrame(load_jsonl(paths["races"]))
    players = pd.DataFrame(load_jsonl(paths["players"]))
    turns = pd.DataFrame(load_jsonl(paths["turns"]))
    _require(len(races), protocol.expected_races(), f"{block} raw races")
    _require(len(players), sum(protocol.N_PLAYERS) * len(protocol.MODULES) * len(protocol.RISKS) * len(protocol.COMPOSITIONS), f"{block} raw players")
    _require(len(turns), int(manifest.get("n_turns", -1)), f"{block} raw turns")
    _require(set(races["game_id"]), _expected_game_ids(), f"{block} game registry")
    if races["game_id"].duplicated().any():
        raise ValueError(f"{block}: duplicate race game_id")
    if players.duplicated(["game_id", "player_index"]).any():
        raise ValueError(f"{block}: duplicate player record")
    if turns.duplicated(["game_id", "round", "player_index"]).any():
        raise ValueError(f"{block}: duplicate turn record")
    if not set(turns["action"]).issubset({"safe", "unsafe"}):
        raise ValueError(f"{block}: invalid action token")
    if not (turns["unsafe"].astype(int) == turns["action"].eq("unsafe").astype(int)).all():
        raise ValueError(f"{block}: action/unsafe mismatch")
    if int(turns["parse_failed"].astype(bool).sum()) != int(races["parse_failures"].sum()):
        raise ValueError(f"{block}: turn/race parse-failure mismatch")

    race_rounds = races.set_index("game_id")["n_rounds"].astype(int)
    expected_turns = int(sum(int(race_rounds[gid]) * int(n) for gid, n in races.set_index("game_id")["n_players"].items()))
    _require(len(turns), expected_turns, f"{block} decisions implied by completed races")
    expected_player_keys = {(gid, seat) for gid, n in races.set_index("game_id")["n_players"].items() for seat in range(int(n))}
    _require(set(zip(players["game_id"], players["player_index"])), expected_player_keys, f"{block} player coverage")
    expected_turn_keys = {(gid, rnd, seat) for gid, n in races.set_index("game_id")["n_players"].items() for rnd in range(1, int(race_rounds[gid]) + 1) for seat in range(int(n))}
    _require(set(zip(turns["game_id"], turns["round"], turns["player_index"])), expected_turn_keys, f"{block} turn coverage")

    for frame in (races, players, turns):
        frame.insert(0, "block", block)
    return races, players, turns, manifest


def load_and_validate(root: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    all_races: list[pd.DataFrame] = []
    all_players: list[pd.DataFrame] = []
    all_turns: list[pd.DataFrame] = []
    manifests: list[dict[str, Any]] = []
    seen: set[str] = set()
    for ordinal, run_dir in enumerate(discover_runs(root), 1):
        manifest = load_json(run_dir / "run_manifest.json")
        block = _block_name(run_dir, manifest, ordinal)
        if block in seen:
            block = f"{block}_{ordinal}"
        seen.add(block)
        races, players, turns, manifest = validate_run(run_dir, block)
        all_races.append(races); all_players.append(players); all_turns.append(turns); manifests.append(manifest)
    return pd.concat(all_races, ignore_index=True), pd.concat(all_players, ignore_index=True), pd.concat(all_turns, ignore_index=True), manifests


def add_progress_rank(turns: pd.DataFrame) -> pd.DataFrame:
    frame = turns.copy()
    categories: list[str] = []
    percentiles: list[float] = []
    ranks: list[float] = []
    for row in frame.itertuples(index=False):
        own = float(row.own_progress_before)
        others = [float(value) for value in row.others_progress_before]
        greater = sum(value > own + 1e-9 for value in others)
        equal = 1 + sum(abs(value - own) <= 1e-9 for value in others)
        rank = 1.0 + greater + (equal - 1) / 2.0
        ranks.append(rank)
        percentiles.append(0.0 if int(row.n_players) == 1 else (rank - 1.0) / (int(row.n_players) - 1.0))
        if equal > 1:
            categories.append("tied")
        elif greater == 0:
            categories.append("leader")
        elif greater == int(row.n_players) - 1:
            categories.append("last")
        else:
            categories.append("middle")
    frame["progress_rank"] = ranks
    frame["progress_rank_percentile"] = percentiles
    frame["progress_rank_category"] = categories
    return frame


def build_tables(players: pd.DataFrame, turns: pd.DataFrame) -> dict[str, pd.DataFrame]:
    race_seat = players.copy()
    race_seat["unsafe_rate"] = race_seat["unsafe_frequency"].astype(float)
    keys = ["block", "module_id", "n_players", "risk", "composition", "seat_model_key"]
    overview = race_seat.groupby(keys, observed=True).agg(
        n_race_seats=("game_id", "size"),
        n_races=("game_id", "nunique"),
        unsafe_rate_mean=("unsafe_rate", "mean"),
        unsafe_rate_median=("unsafe_rate", "median"),
        final_payoff_mean=("final_payoff", "mean"),
        setback_rate=("setback", "mean"),
    ).reset_index()

    crn = race_seat.groupby(["block", "crn_block", "module_id", "n_players", "risk", "composition"], observed=True).agg(
        n_seats=("player_index", "size"), unsafe_rate=("unsafe_rate", "mean"), final_payoff=("final_payoff", "mean")
    ).reset_index()

    ranked = add_progress_rank(turns)
    rank_summary = ranked.groupby(keys + ["progress_rank_category"], observed=True).agg(
        n_decisions=("unsafe", "size"),
        n_races=("game_id", "nunique"),
        unsafe_rate_decision_weighted=("unsafe", "mean"),
        mean_rank_percentile=("progress_rank_percentile", "mean"),
    ).reset_index()

    index = ["block", "n_players", "risk", "composition", "rep", "player_index", "seat_model_key"]
    paired_rows: list[pd.DataFrame] = []
    for treatment, control, label in PAIR_SPECS:
        left = race_seat[race_seat.module_id.eq(treatment)][index + ["unsafe_rate", "final_payoff"]]
        right = race_seat[race_seat.module_id.eq(control)][index + ["unsafe_rate", "final_payoff"]]
        pair = left.merge(right, on=index, how="inner", suffixes=("_treatment", "_control"), validate="one_to_one")
        expected = len(left)
        if len(pair) != expected or len(right) != expected:
            raise ValueError(f"Incomplete paired contrast {label}: {len(pair)}/{expected}")
        pair["contrast"] = label
        pair["treatment_module"] = treatment
        pair["control_module"] = control
        pair["unsafe_rate_delta"] = pair.unsafe_rate_treatment - pair.unsafe_rate_control
        pair["final_payoff_delta"] = pair.final_payoff_treatment - pair.final_payoff_control
        paired_rows.append(pair)
    paired = pd.concat(paired_rows, ignore_index=True)
    paired_summary = paired.groupby(["block", "contrast", "n_players", "risk", "seat_model_key"], observed=True).agg(
        n_paired_race_seats=("unsafe_rate_delta", "size"),
        mean_unsafe_rate_delta=("unsafe_rate_delta", "mean"),
        median_unsafe_rate_delta=("unsafe_rate_delta", "median"),
        mean_final_payoff_delta=("final_payoff_delta", "mean"),
    ).reset_index()
    return {"race_seat": race_seat, "race_crn": crn, "ecology_summary": overview, "progress_rank_summary": rank_summary, "paired_race_seat_contrasts": paired, "paired_contrast_summary": paired_summary}


def _style() -> None:
    plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "#F8FAFC", "axes.edgecolor": "#CBD5E1", "axes.titleweight": "bold", "axes.labelcolor": "#334155", "font.size": 10, "grid.color": "#E2E8F0", "grid.linewidth": 0.8})


def _save(fig: plt.Figure, output: Path, name: str) -> None:
    fig.savefig(output / f"{name}.png", dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(output / f"{name}.pdf", bbox_inches="tight", facecolor="white")
    plt.close(fig)


def make_figures(tables: dict[str, pd.DataFrame], output: Path, primary_block: str) -> list[str]:
    _style(); output.mkdir(parents=True, exist_ok=True)
    overview = tables["ecology_summary"].query("block == @primary_block")
    figures: list[str] = []

    baseline = overview.query("module_id in ['neutral_fixed_total', 'neutral_per_capita_normalized']")
    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.2), sharey=True)
    for ax, risk in zip(axes, protocol.RISKS):
        part = baseline[np.isclose(baseline.risk, risk)].groupby(["module_id", "n_players", "seat_model_key"], observed=True).unsafe_rate_mean.mean().reset_index()
        for (module_id, model), group in part.groupby(["module_id", "seat_model_key"], observed=True):
            ax.plot(group.n_players, 100 * group.unsafe_rate_mean, marker="o", color=MODEL_COLORS[model], linestyle="-" if module_id == "neutral_fixed_total" else "--", label=f"{MODEL_LABELS[model]} · {'fixed' if module_id.endswith('total') else 'normalized'}")
        ax.set(title=f"Private-risk ceiling {risk:.1f}", xlabel="Number of players (N)"); ax.grid(True, axis="y"); ax.set_ylim(-3, 103)
    axes[0].set_ylabel("Mean race-seat Unsafe rate (%)")
    handles, labels = axes[-1].get_legend_handles_labels(); fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.08))
    fig.suptitle("Group-size scaling: fixed total benefit versus per-capita normalization", y=1.17, fontsize=15, fontweight="bold")
    fig.text(0.01, -0.03, "Diagnostic, unadmitted evidence · points average race-seat rates across registered compositions", color="#64748B")
    _save(fig, output, "nplayer_unsafe_vs_n"); figures.append("nplayer_unsafe_vs_n")

    heat = overview.groupby(["module_id", "n_players"], observed=True).unsafe_rate_mean.mean().unstack()
    heat = heat.reindex([item.module_id for item in protocol.MODULES])
    fig, ax = plt.subplots(figsize=(10.5, 7.0)); image = ax.imshow(100 * heat.to_numpy(), cmap="RdYlBu_r", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(len(heat.columns)), heat.columns); ax.set_yticks(range(len(heat.index)), [MODULE_LABELS[x] for x in heat.index], fontsize=9)
    ax.set_xlabel("Number of players (N)"); ax.set_title("Unsafe behavior across the registered ecology", fontsize=15, pad=14)
    for y in range(len(heat.index)):
        for x in range(len(heat.columns)):
            value = 100 * heat.iloc[y, x]; ax.text(x, y, f"{value:.0f}", ha="center", va="center", fontsize=8, color="white" if value < 20 or value > 80 else "#0F172A")
    bar = fig.colorbar(image, ax=ax, pad=0.02); bar.set_label("Mean race-seat Unsafe rate (%)")
    fig.text(0.01, 0.01, "Primary lane block · averaged descriptively across risks, compositions, and model seats", color="#64748B")
    _save(fig, output, "nplayer_module_heatmap"); figures.append("nplayer_module_heatmap")

    pairs = tables["paired_contrast_summary"].query("block == @primary_block").groupby(["contrast", "seat_model_key"], observed=True).agg(delta=("mean_unsafe_rate_delta", "mean"), n=("n_paired_race_seats", "sum")).reset_index()
    fig, ax = plt.subplots(figsize=(10.5, 5.8)); labels_order = [item[2] for item in PAIR_SPECS]
    ybase = np.arange(len(labels_order)); offsets = {"qwen25_7b": -0.15, "mistral7_01": 0.15}
    ax.axvline(0, color="#475569", linewidth=1)
    for model in MODEL_LABELS:
        part = pairs[pairs.seat_model_key.eq(model)].set_index("contrast").reindex(labels_order)
        ax.scatter(100 * part.delta, ybase + offsets[model], s=64, color=MODEL_COLORS[model], label=MODEL_LABELS[model], zorder=3)
    ax.set_yticks(ybase, labels_order); ax.set_xlabel("Paired change in race-seat Unsafe rate (percentage points)"); ax.set_title("Registered paired module contrasts", fontsize=15, pad=12); ax.grid(True, axis="x"); ax.legend(frameon=False); ax.invert_yaxis()
    fig.text(0.01, 0.01, "Treatment − control, paired on block × N × risk × composition × rep × seat; descriptive only", color="#64748B")
    _save(fig, output, "nplayer_paired_contrasts"); figures.append("nplayer_paired_contrasts")

    rank = tables["progress_rank_summary"].query("block == @primary_block").groupby(["progress_rank_category", "seat_model_key"], observed=True).agg(rate=("unsafe_rate_decision_weighted", "mean"), decisions=("n_decisions", "sum")).reset_index()
    order = [x for x in ("leader", "middle", "last", "tied") if x in set(rank.progress_rank_category)]
    fig, ax = plt.subplots(figsize=(8.5, 4.8)); x = np.arange(len(order)); width = .34
    for j, model in enumerate(MODEL_LABELS):
        part = rank[rank.seat_model_key.eq(model)].set_index("progress_rank_category").reindex(order)
        ax.bar(x + (j - .5) * width, 100 * part.rate, width, color=MODEL_COLORS[model], label=MODEL_LABELS[model])
    ax.set_xticks(x, [label.title() for label in order]); ax.set_ylim(0, 100); ax.set_ylabel("Decision-weighted Unsafe rate (%)"); ax.set_title("Behavior by pre-decision progress rank", fontsize=15); ax.grid(True, axis="y"); ax.legend(frameon=False)
    fig.text(0.01, 0.01, "Trajectory-conditioned diagnostic: decisions are dependent and rank is endogenous; not a causal position effect", color="#64748B")
    _save(fig, output, "nplayer_progress_rank"); figures.append("nplayer_progress_rank")
    return figures


def analyze(root: Path, output: Path) -> dict[str, Any]:
    races, players, turns, manifests = load_and_validate(root)
    block_order = list(dict.fromkeys(races["block"].tolist()))
    primary = block_order[0]
    tables = build_tables(players, turns)
    data_dir = output / "data"; figure_dir = output / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        table.to_csv(data_dir / f"{name}.csv", index=False)
    figures = make_figures(tables, figure_dir, primary)
    summary = {
        "schema_version": ANALYSIS_SCHEMA,
        "protocol": protocol.PROTOCOL,
        "evidence_class": EVIDENCE_CLASS,
        "claim_boundary": "Exploratory diagnostic only: the exact N-player protocol is unadmitted; alliance arms are framing-only; turn-weighted rank patterns are endogenous and non-causal.",
        "analysis_grain": {"primary": "race-seat paired within registered CRN cells", "dependence": "players and decisions within a race are dependent", "turn_tables": "descriptive only"},
        "blocks": block_order,
        "primary_block": primary,
        "n_races": int(len(races)), "n_race_seats": int(len(players)), "n_decisions": int(len(turns)),
        "parse_failures": int(turns.parse_failed.astype(bool).sum()),
        "coverage": {"races_per_block": protocol.expected_races(), "race_seats_per_block": int(len(players) / len(block_order)), "complete": True},
        "figures": figures,
        "source_commits": sorted({str(value.get("source_commit")) for value in manifests}),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "analysis_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("experiment_root", type=Path)
    parser.add_argument("output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args(); summary = analyze(args.experiment_root, args.output)
    print(json.dumps(summary, indent=2)); return 0


if __name__ == "__main__":
    raise SystemExit(main())
