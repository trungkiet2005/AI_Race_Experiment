#!/usr/bin/env python3
"""Build a reproducible, publication-ready EDA bank for the two-player AI Race.

The repository contains several scientifically distinct evidence lanes.  This
script inventories all of them, but never silently pools API frontier pilots,
Kaggle Benchmark retries, local-Qwen runs, prompt-surface perturbations, or
context-skin diagnostics.  The main cross-model estimands use the five complete
frontier baseline pilots.  Uncertainty is computed by resampling the common-
random-number repetition block, not individual turns.

Outputs are written below ``results/derived/two_player_paper_analysis`` by
default.  Each figure has a plotted-data CSV and is exported as PDF, SVG, PNG,
and TIFF.  All evidence remains exploratory: there are no confirmatory N=2 runs
in the current repository snapshot.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import re
import sys
import tarfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd
from scipy import stats


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = REPO_ROOT / "results" / "derived" / "two_player_paper_analysis"
FIGURE_HELPERS = (
    Path.home()
    / ".agents"
    / "skills"
    / "scientific-visualization"
    / "scripts"
)
if FIGURE_HELPERS.is_dir():
    sys.path.insert(0, str(FIGURE_HELPERS))

try:
    from figure_export import save_publication_figure
    from style_presets import apply_publication_style
except ImportError:  # pragma: no cover - fallback for a portable checkout
    save_publication_figure = None
    apply_publication_style = None


ANALYSIS_SCHEMA = "ai-race-two-player-paper-eda-v1"
PRIMARY_MODELS = [
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.5-flash-lite",
    "gpt-5-nano",
    "gpt-5.4-nano",
]
MODEL_LABELS = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "anthropic-claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "qwen2.5:7b-instruct-fp16": "Qwen2.5 7B",
}
MODEL_ORDER = [MODEL_LABELS[x] for x in PRIMARY_MODELS] + [
    "Claude Haiku 4.5",
    "Qwen2.5 7B",
]
MODEL_COLORS = {
    "Gemini 3 Flash": "#0072B2",
    "Gemini 3.1 Flash-Lite": "#56B4E9",
    "Gemini 3.5 Flash-Lite": "#009E73",
    "GPT-5 nano": "#D55E00",
    "GPT-5.4 nano": "#CC79A7",
    "Claude Haiku 4.5": "#E69F00",
    "Qwen2.5 7B": "#4D4D4D",
}
MODEL_MARKERS = {
    "Gemini 3 Flash": "o",
    "Gemini 3.1 Flash-Lite": "s",
    "Gemini 3.5 Flash-Lite": "^",
    "GPT-5 nano": "D",
    "GPT-5.4 nano": "P",
    "Claude Haiku 4.5": "X",
    "Qwen2.5 7B": "v",
}
RISK_ORDER = [0.1, 0.6, 0.9]
RISK_LABELS = {0.1: "10%", 0.6: "60%", 0.9: "90%"}
PERSONA_ORDER = ["R0", "R-", "R+", "S_CC", "S_AC", "S_CA", "S_AA"]
PERSONA_LABELS = {
    "R0": "Neutral persona",
    "R-": "Risk-averse",
    "R+": "Risk-seeking",
    "S_CC": "Coop–coop",
    "S_AC": "Adv–coop",
    "S_CA": "Coop–adv",
    "S_AA": "Adv–adv",
}
FORMATS = ["pdf", "svg", "png", "tiff"]
PANEL_LABELS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"


@dataclass
class RunData:
    source_run: str
    source_kind: str
    manifest: dict[str, Any]
    turns: pd.DataFrame | None = None
    races: pd.DataFrame | None = None
    players: pd.DataFrame | None = None

    @property
    def model(self) -> str:
        value = self.manifest.get("model")
        if isinstance(value, dict):
            return str(value.get("short_name") or value.get("name") or "")
        return str(value or "")

    @property
    def model_label(self) -> str:
        return MODEL_LABELS.get(self.model, self.model)

    @property
    def status(self) -> str:
        return str(self.manifest.get("status") or "unknown").lower()

    @property
    def phase(self) -> str:
        value = self.manifest.get("run_phase")
        if value is None and isinstance(self.manifest.get("experiment"), dict):
            value = self.manifest["experiment"].get("runPhase")
        if value is None:
            value = self.manifest.get("profile")
        return str(value or "unknown").lower()

    @property
    def persona(self) -> str:
        return str(self.manifest.get("persona_condition") or "none")

    @property
    def roles(self) -> list[str]:
        values = self.manifest.get("persona_roles") or ["", ""]
        return [str(x or "") for x in values]

    @property
    def lane(self) -> str:
        return str(self.manifest.get("lane") or "")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    return value if isinstance(value, dict) else {}


def _count_lines(path: Path, *, csv_header: bool = False) -> int:
    if not path.is_file():
        return 0
    with path.open("rb") as handle:
        count = sum(1 for _ in handle)
    return max(0, count - int(csv_header))


def _read_jsonl(path_or_handle: Path | Any) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    if isinstance(path_or_handle, Path):
        handle = path_or_handle.open("r", encoding="utf-8")
        close = True
    else:
        handle = io.TextIOWrapper(path_or_handle, encoding="utf-8")
        close = True
    try:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"JSONL line {line_number} is not an object")
            records.append(value)
    finally:
        if close:
            handle.close()
    return pd.DataFrame.from_records(records)


def _read_directory_run(
    directory: Path,
    *,
    need_turns: bool = True,
    need_races: bool = True,
    need_players: bool = True,
    source_kind: str = "directory",
) -> RunData:
    manifest = _safe_json(directory / "run_manifest.json")
    turns = _read_jsonl(directory / "turns.jsonl") if need_turns else None
    races = pd.read_csv(directory / "races.csv") if need_races else None
    players = pd.read_csv(directory / "players.csv") if need_players else None
    return RunData(
        source_run=directory.resolve().relative_to(REPO_ROOT).as_posix(),
        source_kind=source_kind,
        manifest=manifest,
        turns=turns,
        races=races,
        players=players,
    )


def _read_tar_run(
    archive: Path,
    run_prefix: str,
    *,
    need_turns: bool = True,
    need_races: bool = True,
    need_players: bool = True,
) -> RunData:
    prefix = run_prefix.rstrip("/") + "/"
    with tarfile.open(archive, "r:gz") as tar:
        manifest_member = tar.getmember(prefix + "run_manifest.json")
        manifest_handle = tar.extractfile(manifest_member)
        if manifest_handle is None:
            raise FileNotFoundError(prefix + "run_manifest.json")
        manifest = json.load(manifest_handle)
        turns = None
        races = None
        players = None
        if need_turns:
            handle = tar.extractfile(tar.getmember(prefix + "turns.jsonl"))
            if handle is None:
                raise FileNotFoundError(prefix + "turns.jsonl")
            turns = _read_jsonl(handle)
        if need_races:
            handle = tar.extractfile(tar.getmember(prefix + "races.csv"))
            if handle is None:
                raise FileNotFoundError(prefix + "races.csv")
            races = pd.read_csv(handle)
        if need_players:
            handle = tar.extractfile(tar.getmember(prefix + "players.csv"))
            if handle is None:
                raise FileNotFoundError(prefix + "players.csv")
            players = pd.read_csv(handle)
    rel_archive = archive.resolve().relative_to(REPO_ROOT).as_posix()
    return RunData(
        source_run=f"{rel_archive}::{run_prefix.rstrip('/')}",
        source_kind="tar",
        manifest=manifest,
        turns=turns,
        races=races,
        players=players,
    )


def _prepare_frames(run: RunData) -> RunData:
    for frame_name in ("turns", "races", "players"):
        frame = getattr(run, frame_name)
        if frame is None:
            continue
        frame = frame.copy()
        frame["source_run"] = run.source_run
        frame["source_kind"] = run.source_kind
        frame["model"] = run.model or frame.get("model", "")
        frame["model_label"] = run.model_label
        frame["persona_condition"] = run.persona
        frame["run_phase"] = run.phase
        frame["run_status"] = run.status
        frame["lane"] = run.lane
        if "player_index" in frame:
            idx = pd.to_numeric(frame["player_index"], errors="coerce").astype("Int64")
            frame["player_index"] = idx
            roles = run.roles
            frame["own_role"] = idx.map(
                lambda x: roles[int(x)] if pd.notna(x) and int(x) < len(roles) else ""
            )
            frame["opponent_role"] = idx.map(
                lambda x: roles[1 - int(x)]
                if pd.notna(x) and len(roles) == 2 and int(x) in (0, 1)
                else ""
            )
        if "rep" in frame:
            frame["rep"] = pd.to_numeric(frame["rep"], errors="coerce").astype("Int64")
        if "max_private_risk" in frame:
            frame["max_private_risk"] = pd.to_numeric(
                frame["max_private_risk"], errors="coerce"
            )
        setattr(run, frame_name, frame)
    return run


def _valid_completed_run(run: RunData) -> tuple[bool, str]:
    if run.status != "completed":
        return False, f"status={run.status}"
    if run.turns is None or run.races is None or run.players is None:
        return False, "missing required table"
    if run.races.empty or run.players.empty or run.turns.empty:
        return False, "empty required table"
    if "game_id" not in run.races or run.races["game_id"].duplicated().any():
        return False, "duplicate or missing race key"
    turn_key = [x for x in ("game_id", "round", "player") if x in run.turns]
    if len(turn_key) < 3 or run.turns.duplicated(turn_key).any():
        return False, "duplicate or missing turn key"
    player_key = [x for x in ("game_id", "player") if x in run.players]
    if len(player_key) < 2 or run.players.duplicated(player_key).any():
        return False, "duplicate or missing player key"
    if "parse_failed" in run.turns and run.turns["parse_failed"].astype(bool).any():
        return False, "one or more final parse failures"
    if len(run.players) != 2 * len(run.races):
        return False, "not exactly two player rows per race"
    return True, "admitted exploratory run"


def _frontier_baseline_runs() -> list[RunData]:
    directories = [
        REPO_ROOT / "results" / "frontier" / "baseline" / model.replace("/", "-")
        for model in PRIMARY_MODELS[:3]
    ]
    directories += [
        REPO_ROOT / "results" / "frontier" / "openai" / "baseline" / model
        for model in PRIMARY_MODELS[3:]
    ]
    return [_prepare_frames(_read_directory_run(path)) for path in directories]


def _claude_baseline_run() -> RunData:
    path = (
        REPO_ROOT
        / "results"
        / "kaggle-benchmarks"
        / "ai-race-baseline"
        / "7"
        / "claude-haiku-4-5-20251001"
        / "357935"
        / "results"
        / "ai_race_baseline"
    )
    return _prepare_frames(_read_directory_run(path, source_kind="kbench"))


def _qwen_persona_runs(*, need_turns: bool = False) -> list[RunData]:
    archive = (
        REPO_ROOT
        / "results"
        / "open_source"
        / "gpu_run_archive"
        / "pilot-identified-t1-0-results.tar.gz"
    )
    prefixes = [
        "pilot_identified_t1_0/lane_a/baseline",
        "pilot_identified_t1_0/lane_a/persona_baseline_adv_coop",
        "pilot_identified_t1_0/lane_a/persona_baseline_neutral",
        "pilot_identified_t1_0/lane_a/persona_baseline_risk_averse",
        "pilot_identified_t1_0/lane_b/baseline_swapped",
        "pilot_identified_t1_0/lane_b/persona_baseline_adv_adv",
        "pilot_identified_t1_0/lane_b/persona_baseline_coop_adv",
    ]
    return [
        _prepare_frames(
            _read_tar_run(
                archive,
                prefix,
                need_turns=need_turns,
                need_races=True,
                need_players=True,
            )
        )
        for prefix in prefixes
    ]


def _frontier_persona_runs(*, matrix: bool | None = None) -> list[RunData]:
    roots = [
        REPO_ROOT / "results" / "frontier" / "persona",
        REPO_ROOT / "results" / "frontier" / "openai" / "persona",
    ]
    runs: list[RunData] = []
    for root in roots:
        for manifest_path in root.rglob("run_manifest.json"):
            is_matrix = "risk_matrix" in manifest_path.parts or bool(
                re.fullmatch(r"R\d+_R\d+_risk_matrix", manifest_path.parent.parent.name)
            )
            if matrix is not None and is_matrix != matrix:
                continue
            directory = manifest_path.parent
            required = [directory / x for x in ("turns.jsonl", "races.csv", "players.csv")]
            if not all(x.is_file() for x in required):
                continue
            run = _prepare_frames(
                _read_directory_run(
                    directory,
                    need_turns=True,
                    need_races=True,
                    need_players=True,
                    source_kind="frontier",
                )
            )
            valid, _ = _valid_completed_run(run)
            if valid and is_matrix:
                experiment = run.manifest.get("experiment") or {}
                games = experiment.get("games") or []
                repetitions = experiment.get("repetitions")
                try:
                    expected_races = len(games) * int(repetitions)
                except (TypeError, ValueError):
                    expected_races = 0
                if expected_races and (
                    len(run.races) != expected_races
                    or run.races["game_id"].nunique() != expected_races
                ):
                    valid = False
            if valid:
                runs.append(run)
    return sorted(runs, key=lambda x: x.source_run)


def _concat_run_frame(runs: Sequence[RunData], name: str) -> pd.DataFrame:
    frames = [getattr(run, name) for run in runs if getattr(run, name) is not None]
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True, sort=False)


def _block_id(frame: pd.DataFrame) -> pd.Series:
    return frame["source_run"].astype(str) + "::rep=" + frame["rep"].astype(str)


def _bootstrap_mean(
    values: Sequence[float], *, rng: np.random.Generator, n_boot: int
) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if not len(arr):
        return math.nan, math.nan, math.nan
    estimate = float(arr.mean())
    if len(arr) == 1:
        return estimate, math.nan, math.nan
    draws = rng.choice(arr, size=(n_boot, len(arr)), replace=True).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return estimate, float(low), float(high)


def _cluster_mean_table(
    frame: pd.DataFrame,
    *,
    group_cols: Sequence[str],
    value_col: str,
    block_col: str,
    rng: np.random.Generator,
    n_boot: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for key, group in frame.groupby(list(group_cols), observed=True, dropna=False):
        key_tuple = key if isinstance(key, tuple) else (key,)
        block_values = group.groupby(block_col, observed=True)[value_col].mean()
        estimate, low, high = _bootstrap_mean(
            block_values.to_numpy(), rng=rng, n_boot=n_boot
        )
        rows.append(
            {
                **dict(zip(group_cols, key_tuple)),
                "estimate": estimate,
                "ci95_low": low,
                "ci95_high": high,
                "n_blocks": int(block_values.size),
                "n_observations": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _holm_adjust(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.full(values.shape, np.nan)
    running = 0.0
    m = len(values)
    for rank, index in enumerate(order):
        candidate = min(1.0, (m - rank) * values[index])
        running = max(running, candidate)
        adjusted[index] = running
    return adjusted


def _risk_response_tables(
    players: pd.DataFrame, *, rng: np.random.Generator, n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = players.copy()
    frame["block_id"] = _block_id(frame)
    rates = _cluster_mean_table(
        frame,
        group_cols=["model", "model_label", "max_private_risk"],
        value_col="unsafe_frequency",
        block_col="block_id",
        rng=rng,
        n_boot=n_boot,
    )
    contrasts: list[dict[str, Any]] = []
    for (model, label), group in frame.groupby(["model", "model_label"], observed=True):
        blocks = (
            group.groupby(["block_id", "max_private_risk"], observed=True)[
                "unsafe_frequency"
            ]
            .mean()
            .unstack("max_private_risk")
        )
        for low_risk, high_risk in ((0.1, 0.6), (0.1, 0.9), (0.6, 0.9)):
            if low_risk not in blocks or high_risk not in blocks:
                continue
            diff = (blocks[high_risk] - blocks[low_risk]).dropna().to_numpy(float)
            estimate, low, high = _bootstrap_mean(diff, rng=rng, n_boot=n_boot)
            if len(diff) and np.any(np.abs(diff) > 1e-12):
                try:
                    p_value = float(stats.wilcoxon(diff, method="auto").pvalue)
                except ValueError:
                    p_value = 1.0
            else:
                p_value = 1.0
            contrasts.append(
                {
                    "model": model,
                    "model_label": label,
                    "risk_low": low_risk,
                    "risk_high": high_risk,
                    "contrast": f"{RISK_LABELS[high_risk]} − {RISK_LABELS[low_risk]}",
                    "estimate": estimate,
                    "ci95_low": low,
                    "ci95_high": high,
                    "n_blocks": int(len(diff)),
                    "wilcoxon_p": p_value,
                }
            )
    contrast_frame = pd.DataFrame(contrasts)
    if not contrast_frame.empty:
        primary = contrast_frame["risk_low"].eq(0.1) & contrast_frame["risk_high"].eq(0.9)
        contrast_frame["holm_p_high_vs_low"] = np.nan
        contrast_frame.loc[primary, "holm_p_high_vs_low"] = _holm_adjust(
            contrast_frame.loc[primary, "wilcoxon_p"].to_numpy()
        )
    return rates, contrast_frame


def _save_table(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, lineterminator="\n")


def _style() -> None:
    if apply_publication_style is not None:
        try:
            apply_publication_style()
        except TypeError:
            apply_publication_style("nature")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 7.5,
            "axes.titlesize": 8.5,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6.8,
            "figure.titlesize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.4,
            "lines.markersize": 4.5,
            "savefig.transparent": False,
        }
    )


def _panel_label(
    ax: plt.Axes, label: str, *, x: float = -0.16, y: float = 1.08
) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        va="bottom",
        ha="left",
    )


def _finish_axis(ax: plt.Axes, *, percent_y: bool = False) -> None:
    ax.grid(axis="y", color="#D9D9D9", linewidth=0.55, alpha=0.8, zorder=0)
    ax.set_axisbelow(True)
    if percent_y:
        ax.set_ylim(-0.02, 1.02)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))


def _save_figure(fig: plt.Figure, base: Path, *, dpi: int) -> list[Path]:
    base.parent.mkdir(parents=True, exist_ok=True)
    if save_publication_figure is not None:
        paths = save_publication_figure(
            fig,
            base,
            formats=FORMATS,
            dpi=dpi,
            bbox_inches="tight",
            pad_inches=0.04,
        )
        return [Path(x) for x in paths]
    paths = []
    for fmt in FORMATS:
        path = base.with_suffix("." + fmt)
        fig.savefig(path, dpi=dpi, bbox_inches="tight", pad_inches=0.04)
        paths.append(path)
    return paths


def _figure_baseline_risk_response(
    rates: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    main_rates = rates[rates["model"].isin(PRIMARY_MODELS)].copy()
    main_contrasts = contrasts[
        contrasts["model"].isin(PRIMARY_MODELS)
        & contrasts["risk_low"].eq(0.1)
        & contrasts["risk_high"].eq(0.9)
    ].copy()
    main_rates["section"] = "risk_response"
    main_contrasts["section"] = "high_minus_low"
    source = pd.concat([main_rates, main_contrasts], ignore_index=True, sort=False)
    _save_table(source, table_dir / "fig01_baseline_risk_response_source.csv")

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.25),
        gridspec_kw={"width_ratios": [1.45, 1.0], "wspace": 0.35},
    )
    ax = axes[0]
    for model in PRIMARY_MODELS:
        label = MODEL_LABELS[model]
        group = main_rates[main_rates["model"].eq(model)].sort_values(
            "max_private_risk"
        )
        color = MODEL_COLORS[label]
        marker = MODEL_MARKERS[label]
        ax.plot(
            group["max_private_risk"],
            group["estimate"],
            color=color,
            marker=marker,
            label=label,
            zorder=3,
        )
        ax.fill_between(
            group["max_private_risk"].to_numpy(float),
            group["ci95_low"].to_numpy(float),
            group["ci95_high"].to_numpy(float),
            color=color,
            alpha=0.12,
            linewidth=0,
            zorder=1,
        )
    ax.set_xticks(RISK_ORDER, [RISK_LABELS[x] for x in RISK_ORDER])
    ax.set_xlabel("Maximum private setback risk")
    ax.set_ylabel("Mean Unsafe fraction per player–race")
    ax.set_title("Risk response differs sharply across models")
    _finish_axis(ax, percent_y=True)
    ax.legend(frameon=False, loc="lower left", ncol=1)
    _panel_label(ax, "A")

    ax = axes[1]
    ordered = [MODEL_LABELS[x] for x in PRIMARY_MODELS]
    y_positions = np.arange(len(ordered))[::-1]
    for y, label in zip(y_positions, ordered):
        row = main_contrasts[main_contrasts["model_label"].eq(label)].iloc[0]
        color = MODEL_COLORS[label]
        ax.errorbar(
            row["estimate"],
            y,
            xerr=np.array(
                [
                    [row["estimate"] - row["ci95_low"]],
                    [row["ci95_high"] - row["estimate"]],
                ]
            ),
            fmt=MODEL_MARKERS[label],
            color=color,
            ecolor=color,
            capsize=2.5,
            linewidth=1.2,
            zorder=3,
        )
        ax.text(
            row["ci95_high"] + 0.015,
            y,
            f"n={int(row['n_blocks'])}",
            va="center",
            fontsize=6.5,
            color="#4D4D4D",
        )
    ax.axvline(0, color="#666666", linewidth=0.8, linestyle="--")
    ax.set_yticks(y_positions, ordered)
    ax.set_xlabel("Change in Unsafe fraction\n90% risk − 10% risk")
    ax.set_title("Paired repetition-block contrast")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    ax.set_axisbelow(True)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    _panel_label(ax, "B")
    fig.suptitle("Two-player baseline pilots (exploratory; 95% block-bootstrap CI)")
    return _save_figure(fig, figure_dir / "fig01_baseline_risk_response", dpi=dpi)


def _initialization_and_round_tables(
    turns: pd.DataFrame, *, rng: np.random.Generator, n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = turns.copy()
    frame["unsafe"] = pd.to_numeric(frame["unsafe"], errors="coerce")
    frame["round"] = pd.to_numeric(frame["round"], errors="coerce").astype(int)
    frame["block_id"] = _block_id(frame)
    frame["trajectory_id"] = (
        frame["source_run"].astype(str)
        + "::"
        + frame["game_id"].astype(str)
        + "::"
        + frame["player"].astype(str)
    )
    phase_rows: list[pd.DataFrame] = []
    first = frame[frame["round"].eq(1)].copy()
    first["phase"] = "Round 1"
    first["trajectory_rate"] = first["unsafe"]
    phase_rows.append(first)
    later = (
        frame[frame["round"].ge(2)]
        .groupby(
            [
                "source_run",
                "model",
                "model_label",
                "max_private_risk",
                "rep",
                "block_id",
                "trajectory_id",
            ],
            observed=True,
        )["unsafe"]
        .mean()
        .reset_index(name="trajectory_rate")
    )
    later["phase"] = "Rounds 2+"
    phase_rows.append(later)
    phase = pd.concat(phase_rows, ignore_index=True, sort=False)
    phase_table = _cluster_mean_table(
        phase,
        group_cols=["model", "model_label", "max_private_risk", "phase"],
        value_col="trajectory_rate",
        block_col="block_id",
        rng=rng,
        n_boot=n_boot,
    )

    round_blocks = (
        frame.groupby(
            ["model", "model_label", "max_private_risk", "round", "block_id"],
            observed=True,
        )["unsafe"]
        .mean()
        .reset_index(name="round_rate")
    )
    round_table = _cluster_mean_table(
        round_blocks,
        group_cols=["model", "model_label", "max_private_risk", "round"],
        value_col="round_rate",
        block_col="block_id",
        rng=rng,
        n_boot=n_boot,
    )
    round_table["n_players_at_risk"] = (
        frame.groupby(
            ["model", "model_label", "max_private_risk", "round"], observed=True
        )["trajectory_id"]
        .nunique()
        .reindex(
            pd.MultiIndex.from_frame(
                round_table[["model", "model_label", "max_private_risk", "round"]]
            )
        )
        .to_numpy()
    )
    return phase_table, round_table


def _figure_initialization_and_dynamics(
    phase_table: pd.DataFrame,
    round_table: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    phase = phase_table[phase_table["model"].isin(PRIMARY_MODELS)].copy()
    rounds = round_table[
        round_table["model"].isin(PRIMARY_MODELS) & round_table["round"].le(10)
    ].copy()
    phase["section"] = "phase"
    rounds["section"] = "round"
    _save_table(
        pd.concat([phase, rounds], ignore_index=True, sort=False),
        table_dir / "fig02_initialization_and_dynamics_source.csv",
    )

    fig = plt.figure(figsize=(7.1, 5.5))
    fig.set_constrained_layout(False)
    grid = fig.add_gridspec(
        2,
        2,
        height_ratios=[1, 1.15],
        hspace=0.52,
        wspace=0.3,
        left=0.1,
        right=0.98,
        top=0.86,
        bottom=0.09,
    )
    for panel_index, phase_name in enumerate(["Round 1", "Rounds 2+"]):
        ax = fig.add_subplot(grid[0, panel_index])
        for model in PRIMARY_MODELS:
            label = MODEL_LABELS[model]
            group = phase[
                phase["model"].eq(model) & phase["phase"].eq(phase_name)
            ].sort_values("max_private_risk")
            ax.errorbar(
                group["max_private_risk"],
                group["estimate"],
                yerr=np.vstack(
                    [
                        group["estimate"] - group["ci95_low"],
                        group["ci95_high"] - group["estimate"],
                    ]
                ),
                color=MODEL_COLORS[label],
                marker=MODEL_MARKERS[label],
                capsize=2,
                label=label,
                zorder=3,
            )
        ax.set_xticks(RISK_ORDER, [RISK_LABELS[x] for x in RISK_ORDER])
        ax.set_xlabel("Maximum risk")
        ax.set_ylabel("Unsafe fraction" if panel_index == 0 else "")
        ax.set_title(phase_name)
        _finish_axis(ax, percent_y=True)
        _panel_label(ax, PANEL_LABELS[panel_index])

    ax = fig.add_subplot(grid[1, :])
    risk_averaged = (
        rounds.groupby(["model", "model_label", "round"], observed=True)
        .agg(
            estimate=("estimate", "mean"),
            ci95_low=("ci95_low", "mean"),
            ci95_high=("ci95_high", "mean"),
            n_players_at_risk=("n_players_at_risk", "sum"),
        )
        .reset_index()
    )
    for model in PRIMARY_MODELS:
        label = MODEL_LABELS[model]
        group = risk_averaged[risk_averaged["model"].eq(model)].sort_values("round")
        ax.plot(
            group["round"],
            group["estimate"],
            color=MODEL_COLORS[label],
            marker=MODEL_MARKERS[label],
            label=label,
        )
    ax.axvspan(0.7, 1.3, color="#E6E6E6", alpha=0.75, zorder=0)
    ax.axvline(5, color="#777777", linestyle=":", linewidth=0.9)
    ax.text(5.05, 0.03, "stop lottery begins", color="#666666", fontsize=6.5)
    ax.set_xlim(0.7, 10.3)
    ax.set_xticks(range(1, 11))
    ax.set_xlabel("Round (averaged over risk treatments)")
    ax.set_ylabel("Unsafe fraction among active player–races")
    ax.set_title("Observed round profile; later rounds condition on survival")
    _finish_axis(ax, percent_y=True)
    ax.legend(frameon=False, ncol=3, loc="upper right")
    _panel_label(ax, "C")
    fig.suptitle(
        "Initialization and within-race dynamics (exploratory; 95% block-bootstrap CI)"
    )
    return _save_figure(fig, figure_dir / "fig02_initialization_and_dynamics", dpi=dpi)


def _transition_and_position_tables(
    turns: pd.DataFrame, *, rng: np.random.Generator, n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = turns.copy()
    frame = frame[pd.to_numeric(frame["round"], errors="coerce").ge(2)].copy()
    frame["unsafe"] = pd.to_numeric(frame["unsafe"], errors="coerce")
    frame["block_id"] = _block_id(frame)
    frame["own_prev"] = frame["own_prev_action"].astype(str).str.lower().map(
        {"safe": "S", "unsafe": "U"}
    )
    frame["opp_prev"] = frame["opponent_prev_action"].astype(str).str.lower().map(
        {"safe": "S", "unsafe": "U"}
    )
    frame = frame.dropna(subset=["own_prev", "opp_prev", "unsafe"])
    frame["lag_profile"] = frame["own_prev"] + frame["opp_prev"]
    transitions = _cluster_mean_table(
        frame,
        group_cols=["model", "model_label", "lag_profile"],
        value_col="unsafe",
        block_col="block_id",
        rng=rng,
        n_boot=n_boot,
    )
    gap = pd.to_numeric(frame["progress_gap_before"], errors="coerce")
    frame["position"] = np.select(
        [gap.lt(-1e-9), gap.gt(1e-9)], ["Behind", "Ahead"], default="Tied"
    )
    positions = _cluster_mean_table(
        frame,
        group_cols=["model", "model_label", "position"],
        value_col="unsafe",
        block_col="block_id",
        rng=rng,
        n_boot=n_boot,
    )
    return transitions, positions


def _figure_transition_and_position(
    transitions: pd.DataFrame,
    positions: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    trans = transitions[transitions["model"].isin(PRIMARY_MODELS)].copy()
    pos = positions[positions["model"].isin(PRIMARY_MODELS)].copy()
    trans["section"] = "lag_profile"
    pos["section"] = "position"
    _save_table(
        pd.concat([trans, pos], ignore_index=True, sort=False),
        table_dir / "fig03_conditional_dynamics_source.csv",
    )
    fig = plt.figure(figsize=(7.1, 4.45))
    fig.set_constrained_layout(False)
    grid = fig.add_gridspec(
        2,
        5,
        height_ratios=[1.25, 0.9],
        hspace=0.92,
        wspace=0.5,
        left=0.07,
        right=0.92,
        top=0.84,
        bottom=0.18,
    )
    cmap = matplotlib.colormaps["viridis"]
    image = None
    for index, model in enumerate(PRIMARY_MODELS):
        ax = fig.add_subplot(grid[0, index])
        group = trans[trans["model"].eq(model)].set_index("lag_profile")
        matrix = np.full((2, 2), np.nan)
        count = np.zeros((2, 2), dtype=int)
        for row_index, own in enumerate(["S", "U"]):
            for col_index, opp in enumerate(["S", "U"]):
                profile = own + opp
                if profile in group.index:
                    matrix[row_index, col_index] = group.loc[profile, "estimate"]
                    count[row_index, col_index] = int(group.loc[profile, "n_observations"])
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, aspect="equal")
        for row_index in range(2):
            for col_index in range(2):
                value = matrix[row_index, col_index]
                if np.isfinite(value):
                    color = "white" if value < 0.38 or value > 0.77 else "black"
                    ax.text(
                        col_index,
                        row_index,
                        f"{value:.0%}\n(n={count[row_index, col_index]})",
                        ha="center",
                        va="center",
                        fontsize=5.8,
                        color=color,
                    )
        ax.set_xticks([0, 1], ["S", "U"])
        ax.set_yticks([0, 1], ["S", "U"] if index == 0 else ["", ""])
        ax.set_xlabel("Opponent previous")
        if index == 0:
            ax.set_ylabel("Own previous")
        ax.set_title(MODEL_LABELS[model], fontsize=7.3)
        for spine in ax.spines.values():
            spine.set_visible(False)
        _panel_label(ax, PANEL_LABELS[index], x=-0.27, y=1.18)
    if image is not None:
        cax = fig.add_axes([0.94, 0.61, 0.013, 0.18])
        cbar = fig.colorbar(image, cax=cax, orientation="vertical")
        cbar.set_label("P(Unsafe this round)", fontsize=7)
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))

    ax = fig.add_subplot(grid[1, :])
    position_order = ["Behind", "Tied", "Ahead"]
    offsets = np.linspace(-0.18, 0.18, len(PRIMARY_MODELS))
    for offset, model in zip(offsets, PRIMARY_MODELS):
        label = MODEL_LABELS[model]
        group = pos[pos["model"].eq(model)].set_index("position")
        x = np.arange(3) + offset
        estimates = np.array([group.loc[p, "estimate"] for p in position_order])
        low = np.array([group.loc[p, "ci95_low"] for p in position_order])
        high = np.array([group.loc[p, "ci95_high"] for p in position_order])
        ax.errorbar(
            x,
            estimates,
            yerr=np.vstack([estimates - low, high - estimates]),
            fmt=MODEL_MARKERS[label],
            color=MODEL_COLORS[label],
            capsize=2,
            label=label,
        )
    ax.set_xticks(range(3), position_order)
    ax.set_ylabel("P(Unsafe)")
    ax.set_title("Behavior conditional on relative progress (rounds 2+)")
    _finish_axis(ax, percent_y=True)
    ax.legend(frameon=False, ncol=5, loc="upper center", bbox_to_anchor=(0.5, -0.23))
    _panel_label(ax, "F")
    fig.suptitle("Conditional dynamics are descriptive, not causal")
    return _save_figure(fig, figure_dir / "fig03_conditional_dynamics", dpi=dpi)


def _strategy_table(players: pd.DataFrame) -> pd.DataFrame:
    frame = players.copy()
    value = pd.to_numeric(frame["unsafe_frequency"], errors="coerce")
    frame["strategy"] = np.select(
        [
            value.le(1e-12),
            value.lt(1 / 3),
            value.lt(2 / 3),
            value.lt(1 - 1e-12),
        ],
        ["All Safe", "Mostly Safe", "Mixed", "Mostly Unsafe"],
        default="All Unsafe",
    )
    counts = (
        frame.groupby(
            ["model", "model_label", "max_private_risk", "strategy"], observed=True
        )
        .size()
        .reset_index(name="n_player_races")
    )
    totals = counts.groupby(
        ["model", "model_label", "max_private_risk"], observed=True
    )["n_player_races"].transform("sum")
    counts["proportion"] = counts["n_player_races"] / totals
    return counts


def _figure_strategy_composition(
    strategy: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    source = strategy[strategy["model"].isin(PRIMARY_MODELS)].copy()
    _save_table(source, table_dir / "fig04_strategy_composition_source.csv")
    categories = ["All Safe", "Mostly Safe", "Mixed", "Mostly Unsafe", "All Unsafe"]
    colors = ["#3B4CC0", "#7B9ACC", "#D9D9D9", "#E9A35C", "#A63D2F"]
    fig, axes = plt.subplots(
        1, 5, figsize=(7.1, 3.1), sharey=True, constrained_layout=False
    )
    for index, (ax, model) in enumerate(zip(axes, PRIMARY_MODELS)):
        group = source[source["model"].eq(model)]
        bottom = np.zeros(3)
        for category, color in zip(categories, colors):
            values = []
            for risk in RISK_ORDER:
                cell = group[
                    group["max_private_risk"].eq(risk) & group["strategy"].eq(category)
                ]
                values.append(float(cell["proportion"].iloc[0]) if len(cell) else 0.0)
            ax.bar(
                np.arange(3),
                values,
                bottom=bottom,
                width=0.72,
                color=color,
                edgecolor="white",
                linewidth=0.35,
            )
            bottom += np.asarray(values)
        ax.set_xticks(range(3), [RISK_LABELS[x] for x in RISK_ORDER], rotation=0)
        ax.set_xlabel("Risk")
        ax.set_title(MODEL_LABELS[model], fontsize=7.4)
        ax.set_ylim(0, 1)
        if index == 0:
            ax.set_ylabel("Share of player–races")
            ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.grid(False)
        _panel_label(ax, PANEL_LABELS[index])
    handles = [Patch(facecolor=c, label=k) for k, c in zip(categories, colors)]
    fig.legend(
        handles=handles,
        ncol=5,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.07),
    )
    fig.subplots_adjust(bottom=0.18, top=0.82, left=0.08, right=0.98, wspace=0.25)
    fig.suptitle("Strategy archetypes reveal within-model heterogeneity")
    return _save_figure(fig, figure_dir / "fig04_strategy_composition", dpi=dpi)


def _payoff_table(
    players: pd.DataFrame, *, rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    frame = players.copy()
    frame["block_id"] = _block_id(frame)
    frame["safety_fraction"] = 1 - pd.to_numeric(
        frame["unsafe_frequency"], errors="coerce"
    )
    metrics = {
        "safety": "safety_fraction",
        "final_payoff": "final_payoff",
        "stage_payoff": "stage_payoff",
        "prize": "prize",
        "setback_rate": "setback",
        "tie_rate": None,
    }
    frame["tie_indicator"] = frame["outcome"].astype(str).str.lower().eq("tie").astype(float)
    metrics["tie_rate"] = "tie_indicator"
    tables = []
    for metric, column in metrics.items():
        table = _cluster_mean_table(
            frame,
            group_cols=["model", "model_label", "max_private_risk"],
            value_col=str(column),
            block_col="block_id",
            rng=rng,
            n_boot=n_boot,
        )
        table["metric"] = metric
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    wide = result.pivot_table(
        index=["model", "model_label", "max_private_risk"],
        columns="metric",
        values=["estimate", "ci95_low", "ci95_high", "n_blocks"],
        aggfunc="first",
    )
    wide.columns = [f"{metric}_{stat}" for stat, metric in wide.columns]
    return wide.reset_index()


def _figure_safety_payoff_frontier(
    payoff: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    source = payoff[payoff["model"].isin(PRIMARY_MODELS)].copy()
    _save_table(source, table_dir / "fig05_safety_payoff_frontier_source.csv")
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.35),
        gridspec_kw={"width_ratios": [1.25, 1], "wspace": 0.34},
    )
    ax = axes[0]
    for model in PRIMARY_MODELS:
        label = MODEL_LABELS[model]
        group = source[source["model"].eq(model)].sort_values("max_private_risk")
        ax.plot(
            group["safety_estimate"],
            group["final_payoff_estimate"],
            color=MODEL_COLORS[label],
            linewidth=0.9,
            alpha=0.8,
        )
        for _, row in group.iterrows():
            ax.errorbar(
                row["safety_estimate"],
                row["final_payoff_estimate"],
                xerr=np.array(
                    [
                        [row["safety_estimate"] - row["safety_ci95_low"]],
                        [row["safety_ci95_high"] - row["safety_estimate"]],
                    ]
                ),
                yerr=np.array(
                    [
                        [row["final_payoff_estimate"] - row["final_payoff_ci95_low"]],
                        [row["final_payoff_ci95_high"] - row["final_payoff_estimate"]],
                    ]
                ),
                fmt=MODEL_MARKERS[label],
                color=MODEL_COLORS[label],
                capsize=1.8,
                alpha=0.95,
            )
            ax.annotate(
                RISK_LABELS[float(row["max_private_risk"])],
                (row["safety_estimate"], row["final_payoff_estimate"]),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=5.7,
                color=MODEL_COLORS[label],
            )
    ax.set_xlabel("Safety fraction (1 − Unsafe)")
    ax.set_ylabel("Mean realized final payoff")
    ax.set_xlim(-0.02, 1.02)
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    _finish_axis(ax)
    ax.set_title("Safety–payoff plane")
    _panel_label(ax, "A")

    ax = axes[1]
    summary = (
        source.groupby(["model", "model_label"], observed=True)
        .agg(
            stage_payoff=("stage_payoff_estimate", "mean"),
            prize=("prize_estimate", "mean"),
            final_payoff=("final_payoff_estimate", "mean"),
            setback_rate=("setback_rate_estimate", "mean"),
        )
        .reset_index()
    )
    summary["setback_loss"] = summary["stage_payoff"] + summary["prize"] - summary["final_payoff"]
    y = np.arange(len(PRIMARY_MODELS))[::-1]
    labels = [MODEL_LABELS[x] for x in PRIMARY_MODELS]
    summary = summary.set_index("model_label").loc[labels].reset_index()
    ax.barh(y, summary["stage_payoff"], color="#56B4E9", label="Stage payoff")
    ax.barh(
        y,
        summary["prize"],
        left=summary["stage_payoff"],
        color="#009E73",
        label="Awarded prize",
    )
    ax.barh(
        y,
        -summary["setback_loss"],
        color="#D55E00",
        alpha=0.85,
        label="Realized loss to setbacks",
    )
    ax.axvline(0, color="#555555", linewidth=0.8)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Risk-averaged payoff components")
    ax.set_title("Outcome decomposition")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    ax.legend(frameon=False, loc="lower right", fontsize=6.2)
    _panel_label(ax, "B")
    fig.suptitle("Safety and realized payoff are related through race outcomes and setbacks")
    return _save_figure(fig, figure_dir / "fig05_safety_payoff_frontier", dpi=dpi)


def _persona_players() -> tuple[pd.DataFrame, list[RunData]]:
    simple_runs = _frontier_persona_runs(matrix=False)
    baseline_runs = _frontier_baseline_runs()
    qwen_runs = _qwen_persona_runs(need_turns=False)
    runs = [*baseline_runs, *simple_runs, *qwen_runs]
    players = _concat_run_frame(runs, "players")
    players = players[players["persona_condition"].isin(["none", *PERSONA_ORDER])].copy()
    return players, runs


def _baseline_comparator_key(frame: pd.DataFrame) -> pd.Series:
    is_qwen = frame["model_label"].eq("Qwen2.5 7B")
    lane = frame["lane"].where(is_qwen, "frontier")
    return frame["model_label"].astype(str) + "::" + lane.astype(str)


def _persona_effect_table(
    players: pd.DataFrame, *, rng: np.random.Generator, n_boot: int
) -> pd.DataFrame:
    frame = players.copy()
    frame["comparator_key"] = _baseline_comparator_key(frame)
    frame["seat"] = pd.to_numeric(frame["player_index"], errors="coerce").astype("Int64")
    keys = ["comparator_key", "model_label", "max_private_risk", "rep", "seat"]
    baseline = frame[frame["persona_condition"].eq("none")].copy()
    # Qwen has a lane-specific neutral baseline; frontier has one baseline per model.
    baseline = (
        baseline.groupby(keys, observed=True)["unsafe_frequency"]
        .mean()
        .reset_index(name="baseline_unsafe")
    )
    treated = frame[frame["persona_condition"].ne("none")].copy()
    paired = treated.merge(baseline, on=keys, how="inner", validate="many_to_one")
    paired["difference"] = paired["unsafe_frequency"] - paired["baseline_unsafe"]
    paired["block_id"] = (
        paired["comparator_key"].astype(str) + "::rep=" + paired["rep"].astype(str)
    )
    # Average risks and both seats inside the randomization block.
    block = (
        paired.groupby(
            ["model_label", "persona_condition", "block_id"], observed=True
        )["difference"]
        .mean()
        .reset_index()
    )
    return _cluster_mean_table(
        block,
        group_cols=["model_label", "persona_condition"],
        value_col="difference",
        block_col="block_id",
        rng=rng,
        n_boot=n_boot,
    )


def _figure_persona_effects(
    effects: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    model_order = ["Gemini 3 Flash", "GPT-5 nano", "GPT-5.4 nano", "Qwen2.5 7B"]
    source = effects[effects["model_label"].isin(model_order)].copy()
    _save_table(source, table_dir / "fig06_persona_effects_source.csv")
    fig, axes = plt.subplots(2, 2, figsize=(7.1, 5.55), sharex=True, sharey=False)
    for index, (ax, model) in enumerate(zip(axes.flat, model_order)):
        group = source[source["model_label"].eq(model)].set_index("persona_condition")
        available = [x for x in PERSONA_ORDER if x in group.index]
        y = np.arange(len(available))[::-1]
        for position, condition in zip(y, available):
            row = group.loc[condition]
            color = MODEL_COLORS[model]
            ax.errorbar(
                row["estimate"],
                position,
                xerr=np.array(
                    [
                        [row["estimate"] - row["ci95_low"]],
                        [row["ci95_high"] - row["estimate"]],
                    ]
                ),
                fmt=MODEL_MARKERS[model],
                color=color,
                capsize=2.2,
            )
            ax.text(
                row["ci95_high"] + 0.018,
                position,
                f"n={int(row['n_blocks'])}",
                va="center",
                fontsize=6,
                color="#555555",
            )
        ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
        ax.set_yticks(y, [PERSONA_LABELS[x] for x in available])
        ax.set_title(model)
        ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
        ax.set_axisbelow(True)
        ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        if index >= 2:
            ax.set_xlabel("Δ Unsafe vs no-persona baseline")
        _panel_label(ax, PANEL_LABELS[index])
    fig.suptitle(
        "Persona prompts substantially shift behavior\n"
        "Risk-averaged paired repetition-block differences; exploratory 95% CI"
    )
    return _save_figure(fig, figure_dir / "fig06_persona_effects", dpi=dpi)


def _persona_role_tables(
    players: pd.DataFrame, *, rng: np.random.Generator, n_boot: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    role = players[
        players["persona_condition"].isin(["S_AA", "S_AC", "S_CA", "S_CC"])
    ].copy()
    role["own_role"] = role["own_role"].str.lower()
    role["opponent_role"] = role["opponent_role"].str.lower()
    role = role[
        role["own_role"].isin(["adversarial", "cooperative"])
        & role["opponent_role"].isin(["adversarial", "cooperative"])
    ].copy()
    role["role_pair"] = role["own_role"].str[:3].str.title() + " vs " + role[
        "opponent_role"
    ].str[:3].str.title()
    role["block_id"] = (
        role["source_run"].astype(str) + "::rep=" + role["rep"].astype(str)
    )
    role_rates = _cluster_mean_table(
        role,
        group_cols=["model_label", "role_pair", "own_role", "opponent_role"],
        value_col="unsafe_frequency",
        block_col="block_id",
        rng=rng,
        n_boot=n_boot,
    )
    asym = role[role["persona_condition"].isin(["S_AC", "S_CA"])].copy()
    paired = (
        asym.groupby(
            [
                "model_label",
                "persona_condition",
                "max_private_risk",
                "rep",
                "source_run",
                "own_role",
            ],
            observed=True,
        )["unsafe_frequency"]
        .mean()
        .unstack("own_role")
        .dropna(subset=["adversarial", "cooperative"])
        .reset_index()
    )
    paired["adversarial_minus_cooperative"] = (
        paired["adversarial"] - paired["cooperative"]
    )
    paired["block_id"] = (
        paired["source_run"].astype(str) + "::rep=" + paired["rep"].astype(str)
    )
    role_effect = _cluster_mean_table(
        paired,
        group_cols=["model_label"],
        value_col="adversarial_minus_cooperative",
        block_col="block_id",
        rng=rng,
        n_boot=n_boot,
    )
    return role_rates, role_effect


def _figure_persona_roles(
    role_rates: pd.DataFrame,
    role_effect: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    model_order = ["Gemini 3 Flash", "GPT-5 nano", "GPT-5.4 nano", "Qwen2.5 7B"]
    rates = role_rates[role_rates["model_label"].isin(model_order)].copy()
    effect = role_effect[role_effect["model_label"].isin(model_order)].copy()
    rates["section"] = "role_pair_rate"
    effect["section"] = "within_race_role_difference"
    _save_table(
        pd.concat([rates, effect], ignore_index=True, sort=False),
        table_dir / "fig07_persona_role_asymmetry_source.csv",
    )
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 3.25),
        gridspec_kw={"width_ratios": [1.35, 1.0], "wspace": 0.38},
    )
    ax = axes[0]
    pairs = ["Coo vs Coo", "Adv vs Coo", "Coo vs Adv", "Adv vs Adv"]
    x = np.arange(len(pairs))
    offsets = np.linspace(-0.21, 0.21, len(model_order))
    for offset, model in zip(offsets, model_order):
        group = rates[rates["model_label"].eq(model)].set_index("role_pair")
        available_x = []
        estimates = []
        low = []
        high = []
        for index, pair in enumerate(pairs):
            if pair in group.index:
                row = group.loc[pair]
                available_x.append(index + offset)
                estimates.append(row["estimate"])
                low.append(row["ci95_low"])
                high.append(row["ci95_high"])
        estimates_array = np.asarray(estimates)
        ax.errorbar(
            available_x,
            estimates_array,
            yerr=np.vstack([estimates_array - low, np.asarray(high) - estimates_array]),
            fmt=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            capsize=2,
            label=model,
        )
    ax.set_xticks(x, ["Coop\nvs coop", "Adv\nvs coop", "Coop\nvs adv", "Adv\nvs adv"])
    ax.set_ylabel("Unsafe fraction of focal role")
    ax.set_title("Focal × opponent role")
    _finish_axis(ax, percent_y=True)
    ax.legend(frameon=False, ncol=2, loc="upper left")
    _panel_label(ax, "A")

    ax = axes[1]
    effect = effect.set_index("model_label").reindex(model_order).dropna(subset=["estimate"])
    y = np.arange(len(effect))[::-1]
    for position, (model, row) in zip(y, effect.iterrows()):
        ax.errorbar(
            row["estimate"],
            position,
            xerr=np.array(
                [
                    [row["estimate"] - row["ci95_low"]],
                    [row["ci95_high"] - row["estimate"]],
                ]
            ),
            fmt=MODEL_MARKERS[model],
            color=MODEL_COLORS[model],
            capsize=2.2,
        )
        ax.text(row["ci95_high"] + 0.02, position, f"n={int(row['n_blocks'])}", va="center", fontsize=6)
    ax.axvline(0, color="#666666", linestyle="--", linewidth=0.8)
    ax.set_yticks(y, effect.index.tolist())
    ax.set_xlabel("Adversarial − cooperative Unsafe fraction")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    ax.set_title("Within asymmetric races")
    _panel_label(ax, "B")
    fig.suptitle("Role instructions dominate the asymmetric persona games")
    return _save_figure(fig, figure_dir / "fig07_persona_role_asymmetry", dpi=dpi)


def _matrix_table(runs: Sequence[RunData]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for run in runs:
        if run.players is None:
            continue
        match = re.fullmatch(r"R(\d)-R(\d)", run.persona)
        if not match:
            continue
        frame = run.players.copy()
        frame["persona_1"] = int(match.group(1))
        frame["persona_2"] = int(match.group(2))
        rows.append(frame)
    if not rows:
        return pd.DataFrame()
    frame = pd.concat(rows, ignore_index=True, sort=False)
    table = (
        frame.groupby(
            ["model", "model_label", "max_private_risk", "persona_1", "persona_2"],
            observed=True,
        )
        .agg(
            unsafe_rate=("unsafe_frequency", "mean"),
            n_player_races=("unsafe_frequency", "size"),
            n_races=("game_id", "nunique"),
        )
        .reset_index()
    )
    return table


def _draw_matrix_panels(
    table: pd.DataFrame,
    *,
    models: Sequence[str],
    title: str,
    base: Path,
    table_path: Path,
    dpi: int,
    annotate_missing: bool,
) -> list[Path]:
    source = table[table["model_label"].isin(models)].copy()
    _save_table(source, table_path)
    n_rows = len(models)
    fig, axes = plt.subplots(
        n_rows,
        3,
        figsize=(7.1, 2.55 * n_rows + 0.45),
        squeeze=False,
        sharex=True,
        sharey=True,
    )
    cmap = matplotlib.colormaps["viridis"].copy()
    cmap.set_bad("#E6E6E6")
    image = None
    for row_index, model in enumerate(models):
        for col_index, risk in enumerate(RISK_ORDER):
            ax = axes[row_index, col_index]
            cell = source[
                source["model_label"].eq(model)
                & source["max_private_risk"].eq(risk)
            ]
            matrix = np.full((6, 6), np.nan)
            for _, record in cell.iterrows():
                matrix[int(record["persona_1"]) - 1, int(record["persona_2"]) - 1] = record[
                    "unsafe_rate"
                ]
            image = ax.imshow(matrix, vmin=0, vmax=1, cmap=cmap, origin="upper")
            if annotate_missing:
                for i in range(6):
                    for j in range(6):
                        if not np.isfinite(matrix[i, j]):
                            ax.text(j, i, "×", ha="center", va="center", color="#888888", fontsize=7)
            ax.set_xticks(range(6), [f"R{x}" for x in range(1, 7)])
            ax.set_yticks(range(6), [f"R{x}" for x in range(1, 7)])
            if row_index == n_rows - 1:
                ax.set_xlabel("Player 2 risk persona")
            if col_index == 0:
                ax.set_ylabel(f"{model}\nPlayer 1 persona")
            ax.set_title(f"Maximum risk {RISK_LABELS[risk]}")
            for spine in ax.spines.values():
                spine.set_visible(False)
            _panel_label(ax, PANEL_LABELS[row_index * 3 + col_index])
    if image is not None:
        cbar = fig.colorbar(image, ax=axes.ravel().tolist(), fraction=0.025, pad=0.03)
        cbar.set_label("System-level mean Unsafe fraction")
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(title)
    return _save_figure(fig, base, dpi=dpi)


def _figure_surface_sensitivity(
    *, figure_dir: Path, table_dir: Path, dpi: int
) -> list[Path]:
    path = (
        REPO_ROOT
        / "results"
        / "open_source"
        / "surface_sensitivity_pilot"
        / "variant_summary.csv"
    )
    table = pd.read_csv(path)
    canonical = float(table.loc[table["variant"].eq("canonical"), "unsafe_rate"].iloc[0])
    table["difference_vs_canonical"] = table["unsafe_rate"] - canonical
    table = table.sort_values("unsafe_rate", ascending=True).reset_index(drop=True)
    _save_table(table, table_dir / "fig10_surface_sensitivity_source.csv")

    interpretation_colors = {
        "control": "#4D4D4D",
        "meaning_preserving": "#0072B2",
        "robustness_perturbation": "#999999",
        "behavioral_framing": "#D55E00",
    }
    fig, axes = plt.subplots(
        1,
        2,
        figsize=(7.1, 6.0),
        sharey=True,
        gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.08},
        constrained_layout=False,
    )
    y = np.arange(len(table))
    labels = [x.replace("_", " ").title() for x in table["variant"]]
    ax = axes[0]
    for position, (_, row) in zip(y, table.iterrows()):
        color = interpretation_colors.get(str(row["interpretation"]), "#666666")
        ax.errorbar(
            row["unsafe_rate"],
            position,
            xerr=np.array(
                [
                    [row["unsafe_rate"] - row["unsafe_rate_cluster_bootstrap_ci95_low"]],
                    [row["unsafe_rate_cluster_bootstrap_ci95_high"] - row["unsafe_rate"]],
                ]
            ),
            fmt="o" if row["variant"] != "canonical" else "D",
            color=color,
            capsize=2,
        )
    ax.axvline(canonical, color="#4D4D4D", linestyle="--", linewidth=0.9)
    ax.set_yticks(y, labels)
    ax.set_xlabel("Whole-trajectory Unsafe rate")
    ax.set_title("Prompt variant behavior")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    _panel_label(ax, "A")

    ax = axes[1]
    colors = [
        interpretation_colors.get(str(x), "#666666") for x in table["interpretation"]
    ]
    ax.barh(y, table["first_round_flip_rate_vs_canonical"], color=colors, height=0.62)
    ax.errorbar(
        table["first_round_flip_rate_vs_canonical"],
        y,
        xerr=np.vstack(
            [
                table["first_round_flip_rate_vs_canonical"]
                - table["first_round_flip_cluster_bootstrap_ci95_low"],
                table["first_round_flip_cluster_bootstrap_ci95_high"]
                - table["first_round_flip_rate_vs_canonical"],
            ]
        ),
        fmt="none",
        ecolor="#333333",
        capsize=1.8,
        linewidth=0.8,
    )
    ax.set_xlabel("First-round action flip vs canonical")
    ax.set_title("Matched initial states")
    ax.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    _panel_label(ax, "B")
    interpretation_labels = {
        "control": "Canonical control",
        "meaning_preserving": "Meaning-preserving",
        "robustness_perturbation": "Robustness perturbation",
        "behavioral_framing": "Behavioral framing",
    }
    handles = [
        Patch(facecolor=color, label=interpretation_labels[key])
        for key, color in interpretation_colors.items()
        if key in set(table["interpretation"])
    ]
    fig.legend(
        handles=handles,
        frameon=False,
        ncol=4,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.025),
    )
    fig.subplots_adjust(bottom=0.20, top=0.84, left=0.21, right=0.98)
    fig.suptitle(
        "Meaning-preserving prompt surfaces can dominate behavior\n"
        "Qwen2.5 7B pilot; 30 repetition blocks per variant"
    )
    return _save_figure(fig, figure_dir / "fig10_surface_sensitivity", dpi=dpi)


def _figure_context_diagnostic(
    *, figure_dir: Path, table_dir: Path, dpi: int
) -> list[Path]:
    base = REPO_ROOT / "results" / "open_source" / "context_skin_pilot"
    rows = []
    for temperature, directory in [
        (0.0, "analysis_live_pilot_t0"),
        (0.7, "analysis_live_pilot_t07"),
    ]:
        frame = pd.read_csv(base / directory / "live_context_rates.csv")
        frame["temperature"] = temperature
        rows.append(frame)
    table = pd.concat(rows, ignore_index=True)
    later = table[
        table["phase"].eq("later_rounds") & table["mapping_id"].eq("safe_p")
    ].copy()
    _save_table(later, table_dir / "fig11_context_temperature_diagnostic_source.csv")
    skins = sorted(later["skin_id"].unique().tolist())
    display = [x.replace("_", " ").title() for x in skins]
    fig, axes = plt.subplots(1, 3, figsize=(7.1, 4.05), sharey=True)
    matrices: dict[float, np.ndarray] = {}
    image = None
    for panel_index, temperature in enumerate([0.0, 0.7]):
        matrix = np.full((len(skins), 3), np.nan)
        cell = later[later["temperature"].eq(temperature)]
        for row_index, skin in enumerate(skins):
            for col_index, risk in enumerate(RISK_ORDER):
                value = cell[
                    cell["skin_id"].eq(skin) & cell["max_private_risk"].eq(risk)
                ]["unsafe_rate"]
                if len(value):
                    matrix[row_index, col_index] = float(value.iloc[0])
        matrices[temperature] = matrix
        ax = axes[panel_index]
        image = ax.imshow(matrix, vmin=0, vmax=1, cmap="viridis", aspect="auto")
        ax.set_xticks(range(3), [RISK_LABELS[x] for x in RISK_ORDER])
        ax.set_yticks(range(len(skins)), display)
        ax.set_xlabel("Maximum risk")
        ax.set_title(f"Temperature {temperature:g}")
        for spine in ax.spines.values():
            spine.set_visible(False)
        _panel_label(ax, PANEL_LABELS[panel_index])
    ax = axes[2]
    diff = matrices[0.7] - matrices[0.0]
    limit = max(0.15, float(np.nanmax(np.abs(diff))))
    image_diff = ax.imshow(
        diff,
        cmap="coolwarm",
        aspect="auto",
        norm=mcolors.TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
    )
    ax.set_xticks(range(3), [RISK_LABELS[x] for x in RISK_ORDER])
    ax.set_xlabel("Maximum risk")
    ax.set_title("Temperature 0.7 − 0")
    for spine in ax.spines.values():
        spine.set_visible(False)
    _panel_label(ax, "C")
    if image is not None:
        cbar = fig.colorbar(image, ax=axes[:2], fraction=0.035, pad=0.03)
        cbar.set_label("Unsafe rate")
        cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    cbar = fig.colorbar(image_diff, ax=axes[2], fraction=0.06, pad=0.04)
    cbar.set_label("Difference")
    cbar.ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.suptitle(
        "Context-skin diagnostic — comprehension admission failed\n"
        "Later rounds with P mapped to Safe; Q→Safe cells were uniformly 0% Unsafe"
    )
    return _save_figure(fig, figure_dir / "fig11_context_temperature_diagnostic", dpi=dpi)


def _parse_failure_count(path: Path) -> tuple[int, int]:
    total = 0
    failures = 0
    if not path.is_file():
        return 0, 0
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            total += 1
            try:
                row = json.loads(line)
                failures += int(bool(row.get("parse_failed", False)))
            except (json.JSONDecodeError, AttributeError):
                failures += 1
    return total, failures


def _classify_directory_manifest(path: Path, manifest: dict[str, Any]) -> dict[str, Any]:
    directory = path.parent
    rel = directory.resolve().relative_to(REPO_ROOT).as_posix()
    lower = rel.lower()
    turns_path = directory / "turns.jsonl"
    races_path = directory / "races.csv"
    players_path = directory / "players.csv"
    n_turns, parse_failures = _parse_failure_count(turns_path)
    n_races = _count_lines(races_path, csv_header=True)
    n_players = _count_lines(players_path, csv_header=True)
    unique_races = math.nan
    duplicate_race_rows = math.nan
    duplicate_turn_rows = math.nan
    if races_path.is_file() and n_races:
        race_ids = pd.read_csv(races_path, usecols=["game_id"])["game_id"]
        unique_races = int(race_ids.nunique())
        duplicate_race_rows = int(race_ids.duplicated().sum())
    if turns_path.is_file() and n_turns:
        keys = []
        with turns_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                    keys.append((row.get("game_id"), row.get("round"), row.get("player")))
                except json.JSONDecodeError:
                    keys.append((None, None, None))
        duplicate_turn_rows = len(keys) - len(set(keys))
    model = manifest.get("model")
    if isinstance(model, dict):
        model = model.get("short_name") or model.get("name")
    phase = manifest.get("run_phase")
    if phase is None and isinstance(manifest.get("experiment"), dict):
        phase = manifest["experiment"].get("runPhase")
    phase = phase or manifest.get("profile") or "unknown"
    status = str(manifest.get("status") or "unknown").lower()
    experiment = manifest.get("experiment") or {}
    games = experiment.get("games") or [] if isinstance(experiment, dict) else []
    repetitions = experiment.get("repetitions") if isinstance(experiment, dict) else None
    try:
        expected_races = len(games) * int(repetitions) if games and repetitions is not None else math.nan
    except (TypeError, ValueError):
        expected_races = math.nan
    if "context_skin" in lower:
        family = "context skin"
    elif "risk_matrix" in lower:
        family = "risk-persona matrix"
    elif "/persona/" in lower or "persona" in lower:
        family = "core persona"
    elif "kaggle-benchmarks" in lower:
        family = "Claude KBench baseline"
    elif "/baseline/" in lower or "baseline" in lower:
        family = "frontier baseline"
    else:
        family = "other AI Race artifact"
    reasons = []
    admitted = True
    if status != "completed":
        reasons.append(f"status={status}")
        admitted = False
    if not all(x.is_file() for x in (turns_path, races_path, players_path)):
        reasons.append("missing gameplay table")
        admitted = False
    if parse_failures:
        reasons.append(f"{parse_failures} parse failures")
        admitted = False
    if isinstance(duplicate_race_rows, int) and duplicate_race_rows:
        reasons.append(f"{duplicate_race_rows} duplicate race rows")
        admitted = False
    if isinstance(duplicate_turn_rows, int) and duplicate_turn_rows:
        reasons.append(f"{duplicate_turn_rows} duplicate turn rows")
        admitted = False
    if n_races and n_players != 2 * n_races:
        reasons.append("not two player rows per race")
        admitted = False
    if (
        n_races
        and np.isfinite(expected_races)
        and (n_races != int(expected_races) or unique_races != int(expected_races))
    ):
        if unique_races != int(expected_races):
            reasons.append(
                f"incomplete cell: {int(unique_races)}/{int(expected_races)} unique races"
            )
        else:
            reasons.append(
                f"row/key mismatch: {n_races} rows, {int(unique_races)} unique, "
                f"{int(expected_races)} expected"
            )
        admitted = False
    if "ai_race/results/_api_5games_allrisk" in lower:
        reasons.append("superseded overlapping pilot")
        admitted = False
    if family == "context skin":
        reasons.append("comprehension admission failed; diagnostic only")
        admitted = False
    evidence = "confirmatory" if str(phase).lower() == "confirmatory" else "pilot/diagnostic"
    return {
        "source_run": rel,
        "storage": "directory",
        "family": family,
        "model": str(model or ""),
        "model_label": MODEL_LABELS.get(str(model or ""), str(model or "")),
        "persona_condition": str(manifest.get("persona_condition") or ""),
        "phase": str(phase).lower(),
        "status": status,
        "evidence_class": evidence,
        "n_race_rows": n_races,
        "expected_races": expected_races,
        "n_unique_races": unique_races,
        "n_player_rows": n_players,
        "n_turns": n_turns,
        "parse_failures": parse_failures,
        "duplicate_race_rows": duplicate_race_rows,
        "duplicate_turn_rows": duplicate_turn_rows,
        "exploratory_admitted": admitted,
        "exclusion_or_note": "; ".join(reasons) if reasons else "admitted exploratory run",
    }


def _inventory_table() -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    roots = [REPO_ROOT / "results", REPO_ROOT / "ai_race" / "results"]
    seen: set[Path] = set()
    for root in roots:
        if not root.is_dir():
            continue
        for manifest_path in root.rglob("run_manifest.json"):
            resolved = manifest_path.resolve()
            if resolved in seen:
                continue
            seen.add(resolved)
            lower = resolved.as_posix().lower()
            if "nplayer" in lower or "_n3-" in lower or "ai_race_nplayer" in lower:
                continue
            rows.append(_classify_directory_manifest(manifest_path, _safe_json(manifest_path)))

    archive_dir = REPO_ROOT / "results" / "open_source" / "gpu_run_archive"
    ledger = _safe_json(archive_dir / "archive_ledger.json")
    for record in ledger.get("archives", []):
        evidence = str(record.get("evidence_class") or "")
        rows.append(
            {
                "source_run": f"results/open_source/gpu_run_archive/{record.get('file')}",
                "storage": "tar archive",
                "family": str(record.get("study") or "open-source diagnostic"),
                "model": "qwen2.5:7b-instruct-fp16",
                "model_label": "Qwen2.5 7B",
                "persona_condition": "",
                "phase": "smoke" if "smoke" in evidence else "pilot",
                "status": "verified archive",
                "evidence_class": evidence,
                "n_race_rows": int(record.get("manifest_races") or 0),
                "expected_races": math.nan,
                "n_unique_races": math.nan,
                "n_player_rows": math.nan,
                "n_turns": int(record.get("manifest_turns") or 0),
                "parse_failures": math.nan,
                "duplicate_race_rows": math.nan,
                "duplicate_turn_rows": math.nan,
                "exploratory_admitted": evidence in {"pilot-raw"},
                "exclusion_or_note": (
                    "pilot archive; analyze within study only"
                    if evidence == "pilot-raw"
                    else "smoke, analysis-only, or superseded archive; do not pool"
                ),
            }
        )
    collective_root = REPO_ROOT / ".archive" / "collective_risk" / "results"
    if collective_root.is_dir():
        for turns_path in collective_root.rglob("turns.jsonl"):
            rows.append(
                {
                    "source_run": turns_path.parent.resolve().relative_to(REPO_ROOT).as_posix(),
                    "storage": "archive directory",
                    "family": "collective-risk game (not AI Race)",
                    "model": turns_path.parent.name,
                    "model_label": turns_path.parent.name,
                    "persona_condition": "",
                    "phase": "historical",
                    "status": "archived",
                    "evidence_class": "out of scope",
                    "n_race_rows": _count_lines(turns_path.parent / "games.csv", csv_header=True),
                    "expected_races": math.nan,
                    "n_unique_races": math.nan,
                    "n_player_rows": math.nan,
                    "n_turns": _count_lines(turns_path),
                    "parse_failures": math.nan,
                    "duplicate_race_rows": math.nan,
                    "duplicate_turn_rows": math.nan,
                    "exploratory_admitted": False,
                    "exclusion_or_note": "different game; excluded from AI Race N=2 analysis",
                }
            )
    return pd.DataFrame(rows).sort_values(["family", "source_run"]).reset_index(drop=True)


def _figure_inventory(
    inventory: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    _save_table(inventory, table_dir / "dataset_inventory.csv")
    ai_race = inventory[~inventory["family"].str.contains("collective-risk", na=False)].copy()
    grouped = (
        ai_race.groupby("family", observed=True)
        .agg(
            artifacts=("source_run", "size"),
            race_rows=("n_race_rows", "sum"),
            turns=("n_turns", "sum"),
            admitted_artifacts=("exploratory_admitted", "sum"),
        )
        .reset_index()
        .sort_values("race_rows")
    )
    grouped["excluded_artifacts"] = grouped["artifacts"] - grouped["admitted_artifacts"]
    _save_table(grouped, table_dir / "fig12_evidence_inventory_source.csv")
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.6), gridspec_kw={"wspace": 0.55})
    y = np.arange(len(grouped))
    labels = grouped["family"].tolist()
    ax = axes[0]
    ax.barh(y, grouped["race_rows"], color="#0072B2")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Recorded race rows (log scale)")
    ax.set_xscale("symlog", linthresh=1)
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    ax.set_title("Available N=2 evidence families")
    _panel_label(ax, "A")
    ax = axes[1]
    ax.barh(y, grouped["admitted_artifacts"], color="#009E73", label="Admitted exploratory")
    ax.barh(
        y,
        grouped["excluded_artifacts"],
        left=grouped["admitted_artifacts"],
        color="#BDBDBD",
        hatch="///",
        label="Excluded / diagnostic only",
    )
    ax.set_yticks(y, [""] * len(y))
    ax.set_xlabel("Run/archive artifacts")
    ax.set_title("Admission is run-level and fail-closed")
    ax.grid(axis="x", color="#D9D9D9", linewidth=0.55)
    ax.legend(frameon=False, loc="lower right")
    _panel_label(ax, "B")
    fig.suptitle("Data coverage and evidence boundary — zero confirmatory runs")
    return _save_figure(fig, figure_dir / "fig12_evidence_inventory", dpi=dpi)


def _figure_gemini_repeatability(
    final_run: RunData,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    old_path = (
        REPO_ROOT
        / "ai_race"
        / "results"
        / "_api_5games_allrisk"
        / "google-gemini-3-flash-preview"
    )
    old = _prepare_frames(_read_directory_run(old_path, source_kind="superseded pilot"))
    old_turns = old.turns.copy()
    final_turns = final_run.turns.copy()
    keys = ["game_id", "round", "player"]
    paired = old_turns.merge(
        final_turns[keys + ["action", "unsafe", "max_private_risk"]],
        on=keys,
        suffixes=("_old", "_final"),
        how="inner",
        validate="one_to_one",
    )
    paired["same_action"] = paired["action_old"].eq(paired["action_final"])
    agreement = (
        paired.groupby("max_private_risk_old", observed=True)
        .agg(n_decisions=("same_action", "size"), action_agreement=("same_action", "mean"))
        .reset_index()
        .rename(columns={"max_private_risk_old": "max_private_risk"})
    )
    rates = []
    for label, frame in [("Earlier 5-rep pilot", old.players), ("Final 10-rep pilot", final_run.players)]:
        cell = (
            frame.groupby("max_private_risk", observed=True)
            .agg(unsafe_rate=("unsafe_frequency", "mean"), n_player_races=("unsafe_frequency", "size"))
            .reset_index()
        )
        cell["run"] = label
        rates.append(cell)
    rates_frame = pd.concat(rates, ignore_index=True)
    rates_frame["section"] = "rate"
    agreement["section"] = "agreement"
    _save_table(
        pd.concat([rates_frame, agreement], ignore_index=True, sort=False),
        table_dir / "fig13_repeat_run_stability_source.csv",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 3.0), gridspec_kw={"wspace": 0.34})
    ax = axes[0]
    styles = [
        ("Earlier 5-rep pilot", "#999999", "o", "--"),
        ("Final 10-rep pilot", MODEL_COLORS["Gemini 3 Flash"], "s", "-"),
    ]
    for label, color, marker, linestyle in styles:
        group = rates_frame[rates_frame["run"].eq(label)]
        ax.plot(
            group["max_private_risk"],
            group["unsafe_rate"],
            color=color,
            marker=marker,
            linestyle=linestyle,
            label=label,
        )
    ax.set_xticks(RISK_ORDER, [RISK_LABELS[x] for x in RISK_ORDER])
    ax.set_xlabel("Maximum risk")
    ax.set_ylabel("Unsafe fraction")
    ax.set_title("Run-level risk response")
    _finish_axis(ax, percent_y=True)
    ax.legend(frameon=False)
    _panel_label(ax, "A")
    ax = axes[1]
    ax.bar(
        range(3),
        agreement["action_agreement"],
        color=MODEL_COLORS["Gemini 3 Flash"],
        width=0.65,
    )
    for index, row in agreement.reset_index(drop=True).iterrows():
        ax.text(index, row["action_agreement"] + 0.025, f"n={int(row['n_decisions'])}", ha="center", fontsize=6.5)
    ax.set_xticks(range(3), [RISK_LABELS[x] for x in agreement["max_private_risk"]])
    ax.set_xlabel("Maximum risk")
    ax.set_ylabel("Exact action agreement")
    ax.set_title("Matched game/round/seat decisions")
    _finish_axis(ax, percent_y=True)
    _panel_label(ax, "B")
    fig.suptitle("Repeated unseeded API pilots are not interchangeable replicates")
    return _save_figure(fig, figure_dir / "fig13_repeat_run_stability", dpi=dpi)


def _figure_protocol_robustness_baselines(
    rates: pd.DataFrame,
    contrasts: pd.DataFrame,
    *,
    figure_dir: Path,
    table_dir: Path,
    dpi: int,
) -> list[Path]:
    models = ["Claude Haiku 4.5", "Qwen2.5 7B"]
    source_rates = rates[rates["model_label"].isin(models)].copy()
    source_contrasts = contrasts[
        contrasts["model_label"].isin(models)
        & contrasts["risk_low"].eq(0.1)
        & contrasts["risk_high"].eq(0.9)
    ].copy()
    source_rates["section"] = "rate"
    source_contrasts["section"] = "contrast"
    _save_table(
        pd.concat([source_rates, source_contrasts], ignore_index=True, sort=False),
        table_dir / "fig01b_protocol_robustness_baselines_source.csv",
    )
    fig, axes = plt.subplots(1, 2, figsize=(7.1, 2.85), sharey=True)
    subtitles = {
        "Claude Haiku 4.5": "KBench structured output; 3 blocks/risk",
        "Qwen2.5 7B": "Ollama, seeded T=1; 20 blocks/risk",
    }
    for index, (ax, model) in enumerate(zip(axes, models)):
        group = source_rates[source_rates["model_label"].eq(model)].sort_values(
            "max_private_risk"
        )
        ax.errorbar(
            group["max_private_risk"],
            group["estimate"],
            yerr=np.vstack(
                [
                    group["estimate"] - group["ci95_low"],
                    group["ci95_high"] - group["estimate"],
                ]
            ),
            color=MODEL_COLORS[model],
            marker=MODEL_MARKERS[model],
            capsize=2.5,
        )
        ax.set_xticks(RISK_ORDER, [RISK_LABELS[x] for x in RISK_ORDER])
        ax.set_xlabel("Maximum risk")
        if index == 0:
            ax.set_ylabel("Mean Unsafe fraction per player–race")
        ax.set_title(f"{model}\n{subtitles[model]}", fontsize=8)
        _finish_axis(ax, percent_y=True)
        contrast = source_contrasts[source_contrasts["model_label"].eq(model)]
        if len(contrast):
            row = contrast.iloc[0]
            ax.text(
                0.04,
                0.07,
                f"90%−10%: {row['estimate']:+.1%}\n95% CI [{row['ci95_low']:+.1%}, {row['ci95_high']:+.1%}]",
                transform=ax.transAxes,
                fontsize=6.5,
                bbox={"facecolor": "white", "edgecolor": "#CCCCCC", "pad": 2},
            )
        _panel_label(ax, PANEL_LABELS[index])
    fig.suptitle("Separate-protocol baseline robustness (descriptive pilot evidence)")
    return _save_figure(fig, figure_dir / "fig01b_protocol_robustness_baselines", dpi=dpi)


def _write_report(
    *,
    output_dir: Path,
    inventory: pd.DataFrame,
    main_rates: pd.DataFrame,
    main_contrasts: pd.DataFrame,
    persona_effects: pd.DataFrame,
    surface_table: pd.DataFrame,
    figure_bases: Sequence[str],
) -> Path:
    report_dir = output_dir / "report"
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "two_player_eda_report.md"
    rate_pivot = main_rates.pivot_table(
        index="model_label", columns="max_private_risk", values="estimate", aggfunc="first"
    )
    high_low = main_contrasts[
        main_contrasts["risk_low"].eq(0.1) & main_contrasts["risk_high"].eq(0.9)
    ].set_index("model_label")
    confirmatory = int(inventory["phase"].eq("confirmatory").sum())
    completed_directory = int(
        (
            inventory["storage"].eq("directory")
            & inventory["status"].eq("completed")
            & inventory["exploratory_admitted"].astype(bool)
        ).sum()
    )
    lines = [
        "# Two-player AI Race: exploratory data analysis and paper figure bank",
        "",
        f"Generated: {datetime.now(timezone.utc).isoformat()}",
        "",
        "## Executive boundary",
        "",
        f"- Confirmatory gameplay runs available: **{confirmatory}**.",
        f"- Completed parse-clean directory runs admitted for exploratory analysis: **{completed_directory}**.",
        "- Every behavioral result in this report is pilot or diagnostic evidence. Model comparisons are exploratory because backend, decoding, and seed-forwarding contracts differ across evidence lanes.",
        "- The headline balanced baseline uses five frontier models, 10 repetition blocks × 3 risks × 2 players per model. Claude (3 blocks/risk) and local Qwen are shown only in separate-protocol robustness figures.",
        "",
        "## Primary estimand and uncertainty",
        "",
        "The primary estimand is the mean player-level trajectory Unsafe fraction. Each player–race contributes equally, so long realized horizons do not receive extra weight. Percentile 95% intervals resample whole source-run × repetition blocks, preserving both seats and the three common-random-number risk treatments. The primary within-model contrast is 90% minus 10% maximum private risk.",
        "",
        "## Baseline findings",
        "",
        "| Model | Unsafe @10% | Unsafe @60% | Unsafe @90% | 90%−10% (95% CI) | blocks |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for model in [MODEL_LABELS[x] for x in PRIMARY_MODELS]:
        row = high_low.loc[model]
        lines.append(
            f"| {model} | {rate_pivot.loc[model, 0.1]:.1%} | {rate_pivot.loc[model, 0.6]:.1%} | {rate_pivot.loc[model, 0.9]:.1%} | {row['estimate']:+.1%} [{row['ci95_low']:+.1%}, {row['ci95_high']:+.1%}] | {int(row['n_blocks'])} |"
        )
    lines += [
        "",
        "The robust descriptive pattern is heterogeneity, not one universal LLM response: the Gemini pilots reduce Unsafe play strongly as risk increases; GPT-5 nano remains low and nearly flat; GPT-5.4 nano is non-monotone. Round 1 must be separated from later play because several Gemini cells initialize at 100% Unsafe while GPT-5 nano initializes at 0%.",
        "",
        "## Persona and prompt sensitivity",
        "",
    ]
    for model in ["Gemini 3 Flash", "GPT-5 nano", "GPT-5.4 nano", "Qwen2.5 7B"]:
        group = persona_effects[persona_effects["model_label"].eq(model)]
        if group.empty:
            continue
        largest = group.iloc[group["estimate"].abs().argmax()]
        lines.append(
            f"- {model}: largest available risk-averaged persona shift is {PERSONA_LABELS.get(largest['persona_condition'], largest['persona_condition'])}, {largest['estimate']:+.1%} versus the no-persona baseline (95% CI {largest['ci95_low']:+.1%} to {largest['ci95_high']:+.1%})."
        )
    surface_range = (float(surface_table["unsafe_rate"].min()), float(surface_table["unsafe_rate"].max()))
    lines += [
        f"- The Qwen prompt-surface pilot spans {surface_range[0]:.1%}–{surface_range[1]:.1%} Unsafe across 18 prompt variants despite unchanged game semantics. This is a major robustness result, not a nuisance detail.",
        "- The GPT risk-persona matrix is complete (36/36 cells for both GPT models). Gemini has only 14 clean cells; duplicate/partial cells are masked and must not be imputed.",
        "- Context-skin behavior remains diagnostic only because the frozen comprehension admission gate failed; its figure intentionally retains that warning.",
        "",
        "## Recommended paper shortlist",
        "",
        "1. **Main:** Fig. 01 baseline risk response + paired high-vs-low forest.",
        "2. **Main:** Fig. 02 initialization and round dynamics, because it prevents an all-round average from hiding model-specific first actions.",
        "3. **Main or Results appendix:** Fig. 05 safety–payoff plane and payoff decomposition.",
        "4. **Persona result:** Fig. 06 persona effects or Fig. 07 role asymmetry; use Fig. 08 for the complete factorial surface.",
        "5. **Robustness:** Fig. 10 prompt-surface sensitivity. It is unusually strong and should be discussed even if placed in the supplement.",
        "6. **Supplement/QC:** Figs. 01b, 03, 04, 09, 11–13.",
        "",
        "## Figure catalog",
        "",
    ]
    for name in figure_bases:
        lines.append(f"- `{name}`: PDF, SVG, 600-dpi PNG, 600-dpi TIFF; plotted data in `../tables/{name}_source.csv` when applicable.")
    lines += [
        "",
        "## Interpretation limits",
        "",
        "- Pilot CIs quantify repetition-block variability in this experiment snapshot; they do not represent a population of prompts or model versions.",
        "- Lag, position, winner, and payoff associations are post-action/endogenous summaries and are not causal mechanism estimates.",
        "- Unseeded API outputs mean repeated calls are independent stochastic attempts, not exact replications. Overlapping game identifiers are never pooled without `source_run`.",
        "- Failed, running, protocol-failed, duplicate-key, partial, smoke, and superseded artifacts remain visible in `dataset_inventory.csv` but are excluded from headline behavior.",
        "",
        "## Reproduction",
        "",
        "```bash",
        "python results/scripts/analyze_two_player_paper_figures.py",
        "```",
    ]
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _validate_outputs(output_dir: Path, figure_bases: Sequence[str]) -> dict[str, Any]:
    checks: dict[str, Any] = {"status": "passed", "figures": {}}
    try:
        from PIL import Image
    except ImportError:  # pragma: no cover
        Image = None
    for base in figure_bases:
        record: dict[str, Any] = {}
        for fmt in FORMATS:
            path = output_dir / "figures" / f"{base}.{fmt}"
            if not path.is_file() or path.stat().st_size == 0:
                checks["status"] = "failed"
                record[fmt] = {"exists": False}
                continue
            item: dict[str, Any] = {
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            if Image is not None and fmt in {"png", "tiff"}:
                with Image.open(path) as image:
                    item["pixels"] = [int(image.width), int(image.height)]
                    item["dpi"] = [float(x) for x in image.info.get("dpi", (0, 0))]
            record[fmt] = item
        checks["figures"][base] = record
    return checks


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=260802)
    parser.add_argument("--dpi", type=int, default=600)
    return parser.parse_args()


def main() -> None:
    args = _arguments()
    output_dir = args.output_dir.resolve()
    figure_dir = output_dir / "figures"
    table_dir = output_dir / "tables"
    report_dir = output_dir / "report"
    for directory in (figure_dir, table_dir, report_dir):
        directory.mkdir(parents=True, exist_ok=True)
    _style()
    rng = np.random.default_rng(args.seed)

    inventory = _inventory_table()
    primary_runs = _frontier_baseline_runs()
    for run in primary_runs:
        valid, reason = _valid_completed_run(run)
        if not valid:
            raise ValueError(f"Primary run failed admission: {run.source_run}: {reason}")
    primary_players = _concat_run_frame(primary_runs, "players")
    primary_turns = _concat_run_frame(primary_runs, "turns")

    claude = _claude_baseline_run()
    valid, reason = _valid_completed_run(claude)
    if not valid:
        raise ValueError(f"Claude clean retry failed admission: {reason}")
    qwen_runs = _qwen_persona_runs(need_turns=False)
    qwen_baselines = [run for run in qwen_runs if run.persona == "none"]
    all_baseline_players = pd.concat(
        [primary_players, claude.players, _concat_run_frame(qwen_baselines, "players")],
        ignore_index=True,
        sort=False,
    )
    rates, contrasts = _risk_response_tables(
        all_baseline_players, rng=rng, n_boot=args.bootstrap_reps
    )
    _save_table(rates, table_dir / "baseline_risk_estimates.csv")
    _save_table(contrasts, table_dir / "baseline_risk_contrasts.csv")

    figure_bases: list[str] = []
    _figure_baseline_risk_response(
        rates,
        contrasts,
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=args.dpi,
    )
    figure_bases.append("fig01_baseline_risk_response")
    _figure_protocol_robustness_baselines(
        rates,
        contrasts,
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=args.dpi,
    )
    figure_bases.append("fig01b_protocol_robustness_baselines")

    phase_table, round_table = _initialization_and_round_tables(
        primary_turns, rng=rng, n_boot=args.bootstrap_reps
    )
    _save_table(phase_table, table_dir / "baseline_initialization_estimates.csv")
    _save_table(round_table, table_dir / "baseline_round_estimates.csv")
    _figure_initialization_and_dynamics(
        phase_table,
        round_table,
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=args.dpi,
    )
    figure_bases.append("fig02_initialization_and_dynamics")

    transitions, positions = _transition_and_position_tables(
        primary_turns, rng=rng, n_boot=args.bootstrap_reps
    )
    _save_table(transitions, table_dir / "baseline_lag_profile_estimates.csv")
    _save_table(positions, table_dir / "baseline_position_estimates.csv")
    _figure_transition_and_position(
        transitions,
        positions,
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=args.dpi,
    )
    figure_bases.append("fig03_conditional_dynamics")

    strategy = _strategy_table(primary_players)
    _figure_strategy_composition(
        strategy,
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=args.dpi,
    )
    figure_bases.append("fig04_strategy_composition")
    payoff = _payoff_table(primary_players, rng=rng, n_boot=args.bootstrap_reps)
    _save_table(payoff, table_dir / "baseline_payoff_estimates.csv")
    _figure_safety_payoff_frontier(
        payoff,
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=args.dpi,
    )
    figure_bases.append("fig05_safety_payoff_frontier")

    persona_players, _ = _persona_players()
    persona_effects = _persona_effect_table(
        persona_players, rng=rng, n_boot=args.bootstrap_reps
    )
    _save_table(persona_effects, table_dir / "persona_effect_estimates.csv")
    _figure_persona_effects(
        persona_effects,
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=args.dpi,
    )
    figure_bases.append("fig06_persona_effects")
    role_rates, role_effect = _persona_role_tables(
        persona_players, rng=rng, n_boot=args.bootstrap_reps
    )
    _save_table(role_rates, table_dir / "persona_role_rates.csv")
    _save_table(role_effect, table_dir / "persona_role_effects.csv")
    _figure_persona_roles(
        role_rates,
        role_effect,
        figure_dir=figure_dir,
        table_dir=table_dir,
        dpi=args.dpi,
    )
    figure_bases.append("fig07_persona_role_asymmetry")

    matrix_runs = _frontier_persona_runs(matrix=True)
    matrix_table = _matrix_table(matrix_runs)
    _save_table(matrix_table, table_dir / "risk_persona_matrix_rates.csv")
    _draw_matrix_panels(
        matrix_table,
        models=["GPT-5 nano", "GPT-5.4 nano"],
        title="Complete 6×6 risk-persona surfaces (system-level Unsafe fraction)",
        base=figure_dir / "fig08_gpt_risk_persona_surfaces",
        table_path=table_dir / "fig08_gpt_risk_persona_surfaces_source.csv",
        dpi=args.dpi,
        annotate_missing=False,
    )
    figure_bases.append("fig08_gpt_risk_persona_surfaces")
    _draw_matrix_panels(
        matrix_table,
        models=["Gemini 3 Flash"],
        title="Gemini risk-persona matrix coverage (14/36 clean cells; × = missing)",
        base=figure_dir / "fig09_gemini_risk_persona_partial",
        table_path=table_dir / "fig09_gemini_risk_persona_partial_source.csv",
        dpi=args.dpi,
        annotate_missing=True,
    )
    figure_bases.append("fig09_gemini_risk_persona_partial")

    _figure_surface_sensitivity(figure_dir=figure_dir, table_dir=table_dir, dpi=args.dpi)
    figure_bases.append("fig10_surface_sensitivity")
    _figure_context_diagnostic(figure_dir=figure_dir, table_dir=table_dir, dpi=args.dpi)
    figure_bases.append("fig11_context_temperature_diagnostic")
    _figure_inventory(
        inventory, figure_dir=figure_dir, table_dir=table_dir, dpi=args.dpi
    )
    figure_bases.append("fig12_evidence_inventory")
    gemini_final = next(
        run for run in primary_runs if run.model == "google/gemini-3-flash-preview"
    )
    _figure_gemini_repeatability(
        gemini_final, figure_dir=figure_dir, table_dir=table_dir, dpi=args.dpi
    )
    figure_bases.append("fig13_repeat_run_stability")

    surface_table = pd.read_csv(
        REPO_ROOT
        / "results"
        / "open_source"
        / "surface_sensitivity_pilot"
        / "variant_summary.csv"
    )
    report_path = _write_report(
        output_dir=output_dir,
        inventory=inventory,
        main_rates=rates[rates["model"].isin(PRIMARY_MODELS)],
        main_contrasts=contrasts[contrasts["model"].isin(PRIMARY_MODELS)],
        persona_effects=persona_effects,
        surface_table=surface_table,
        figure_bases=figure_bases,
    )
    checks = _validate_outputs(output_dir, figure_bases)
    (output_dir / "quality_checks.json").write_text(
        json.dumps(checks, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    input_paths = [
        REPO_ROOT / run.source_run / "run_manifest.json"
        for run in primary_runs
        if "::" not in run.source_run
    ]
    manifest = {
        "schema_version": ANALYSIS_SCHEMA,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete" if checks["status"] == "passed" else "quality_check_failed",
        "evidence_status": "exploratory_pilot_only",
        "confirmatory_runs": 0,
        "bootstrap": {
            "method": "percentile cluster bootstrap over source_run × repetition",
            "repetitions": args.bootstrap_reps,
            "seed": args.seed,
        },
        "primary_models": PRIMARY_MODELS,
        "figure_bases": figure_bases,
        "report": report_path.relative_to(output_dir).as_posix(),
        "primary_input_manifest_sha256": {
            path.resolve().relative_to(REPO_ROOT).as_posix(): _sha256(path)
            for path in input_paths
        },
        "code_sha256": _sha256(Path(__file__)),
        "quality_checks": "quality_checks.json",
    }
    (output_dir / "analysis_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": manifest["status"],
                "output_dir": str(output_dir),
                "figures": len(figure_bases),
                "tables": len(list(table_dir.glob("*.csv"))),
                "report": str(report_path),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
