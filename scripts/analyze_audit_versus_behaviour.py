#!/usr/bin/env python3
"""Does measured endpoint task validity predict how strongly play responds to risk?

The admission campaign (``ai-race-frontier-admission-v6``) measures whether an
endpoint can read the game state it is asked to play.  The gameplay baseline
(``ai-race-frontier-baseline-v3``) measures what the same endpoint actually does
when the assigned maximum private risk moves from 0.1 to 0.9.  This script joins
the two campaigns route by route and reports the association between them as a
descriptive statement over nine endpoints, never as a population effect.

Every audited route now has matched gameplay.  Eight routes ran at
``ai-race-baseline`` task version 3 and one, ``google/gemini-3.5-flash-lite``,
at version 4, because that provider rejects the reasoning-budget argument itself
and the task now omits the parameter for such routes.  The manifest records
``reasoning_requested = null`` for that route and ``"none"`` for the other
eight; the derived table carries the value as a column so the difference is
visible rather than inferred.  See the ``2026-09-09 route-resolved reasoning
budget`` amendment in ``docs/frontier-evidence-collection-protocol.md``.  The
correlation block reports both the nine-route sample and the earlier eight-route
sample so the change is auditable.

Uncertainty follows the house convention in ``analyze_nplayer_scope_clustered.py``:
the race (``game_id``) is the resampling cluster, both seats of a race travel
together, 5000 percentile resamples, recorded seed.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

import publication_style as ps

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "frontier" / "baseline_campaign_v6"
RUNS_ROOT = CAMPAIGN / "ai-race-baseline"
FAILED_ROOT = CAMPAIGN / "failed_runs" / "ai-race-baseline"
DERIVED = CAMPAIGN / "derived"
FIGURE_STEMS = (
    CAMPAIGN / "figures" / "audit_versus_behaviour",
    ROOT / "figures" / "paper" / "audit_versus_behaviour",
)
ADMISSION_CSV = (
    ROOT / "results" / "frontier" / "admission_campaign_v6" / "derived" / "admission_campaign_v6.csv"
)

SEED = 20260909
N_BOOT = 5000
RISKS = (0.1, 0.6, 0.9)
EXPECTED_RACES = 30
EXPECTED_DECISIONS = 558
EXPECTED_RACES_PER_CELL = 10
BASELINE_PROTOCOL = "ai-race-frontier-baseline-v3"
ADMISSION_PROTOCOL = "ai-race-frontier-admission-v6"
BOUNDARY_ROUTE = "anthropic/claude-opus-5@default"
EXPECTED_ROUTES = 9
# The route that the reasoning-budget amendment added, and the task version it
# ran at.  The eight routes tabulated before it stay the reference sample.
AMENDED_ROUTE = "google/gemini-3.5-flash-lite"
AMENDED_TASK_VERSION = "4"
PRIOR_TASK_VERSION = "3"
AMENDMENT_REFERENCE = (
    "docs/frontier-evidence-collection-protocol.md :: 2026-09-09 - route-resolved reasoning budget"
)

SHORT_NAMES = {
    "anthropic/claude-opus-5@default": "Claude Opus 5",
    "anthropic/claude-sonnet-5@default": "Claude Sonnet 5",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "openai/gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
    "openai/gpt-5.5-2026-04-23": "GPT-5.5",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def discover_runs() -> list[dict]:
    runs = []
    seen: dict[str, Path] = {}
    for manifest_path in sorted(RUNS_ROOT.glob("*/*/*/results/ai_race_baseline/run_manifest.json")):
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["status"] != "completed":
            raise RuntimeError(f"Non-completed run under the tabulated path: {manifest_path}")
        if manifest["protocol_id"] != BASELINE_PROTOCOL:
            raise RuntimeError(f"Unexpected protocol in {manifest_path}: {manifest['protocol_id']}")
        route = manifest["model_route"]
        if route in seen:
            raise RuntimeError(
                f"Route {route} is tabulated twice: {seen[route]} and {manifest_path}"
            )
        seen[route] = manifest_path
        runs.append(
            {
                "route": route,
                "model_tag": manifest["model"],
                "run_id": manifest_path.parents[2].name,
                "task_version": manifest_path.parents[3].name,
                "manifest": manifest,
                "turns_path": manifest_path.parent / "turns.jsonl",
                "summary_path": manifest_path.parent / "summary.json",
            }
        )
    if not runs:
        raise RuntimeError(f"No completed gameplay runs found under {RUNS_ROOT}")
    return runs


def failure_record() -> dict:
    paths = sorted(FAILED_ROOT.glob("*/*/*/results/ai_race_baseline/run_manifest.json"))
    records = []
    for path in paths:
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["status"] == "completed":
            raise RuntimeError(f"Completed run filed as a failure: {path}")
        run_dir = path.parents[2]
        record = {
            "route": manifest["model_route"],
            "run_id": run_dir.name,
            "task_version": path.parents[3].name,
            "status": manifest["status"],
            "n_races": manifest["n_races"],
            "n_turns": manifest["n_turns"],
            "reasoning_requested": manifest["decoding"].get("reasoning_requested"),
            "error": manifest["error"],
            "path": str(run_dir.relative_to(ROOT).as_posix()),
            "tabulated": False,
        }
        link_path = run_dir / "superseded_by.json"
        if link_path.exists():
            link = json.loads(link_path.read_text(encoding="utf-8"))
            record["superseded_by"] = link["superseded_by"]
            record["cause_of_failure"] = link["cause_of_failure"]
            record["amendment"] = link["amendment"]
            successor = ROOT / link["superseded_by"]["path"]
            if not (successor / "results" / "ai_race_baseline" / "turns.jsonl").exists():
                raise RuntimeError(
                    f"{link_path} names a successor run whose turns.jsonl is missing: {successor}"
                )
        records.append(record)
    return {"failed_routes": records}


def race_bootstrap(cell: pd.DataFrame) -> tuple[float, float, float, int, int]:
    """House convention: resample whole races, equal weight per race."""

    race_rates = cell.groupby("game_id", sort=True)["unsafe"].mean().to_numpy(float)
    rng = np.random.default_rng(SEED)
    samples = rng.choice(race_rates, size=(N_BOOT, len(race_rates)), replace=True).mean(axis=1)
    return (
        float(race_rates.mean()),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        int(len(race_rates)),
        int(len(cell)),
    )


def race_bootstrap_pooled(cell: pd.DataFrame) -> tuple[float, float, float]:
    """Same clusters, but the estimand is the pooled decision-level Unsafe rate.

    This is the number the run's own ``summary.json`` publishes, so it is the one
    an independent reader reproduces from the artefact.
    """

    groups = [group["unsafe"].to_numpy(float) for _, group in cell.groupby("game_id", sort=True)]
    rng = np.random.default_rng(SEED)
    index = rng.integers(0, len(groups), size=(N_BOOT, len(groups)))
    samples = np.array(
        [np.concatenate([groups[j] for j in row]).mean() for row in index], dtype=float
    )
    pooled = float(np.concatenate(groups).mean())
    return pooled, float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def risk_response_bootstrap(
    low: pd.DataFrame, high: pd.DataFrame
) -> tuple[float, float, float, float, float, float]:
    """Unsafe(0.1) minus Unsafe(0.9); the two cells are independent race sets."""

    low_groups = [g["unsafe"].to_numpy(float) for _, g in low.groupby("game_id", sort=True)]
    high_groups = [g["unsafe"].to_numpy(float) for _, g in high.groupby("game_id", sort=True)]
    rng = np.random.default_rng(SEED)
    low_index = rng.integers(0, len(low_groups), size=(N_BOOT, len(low_groups)))
    high_index = rng.integers(0, len(high_groups), size=(N_BOOT, len(high_groups)))
    pooled_draws = np.array(
        [
            np.concatenate([low_groups[j] for j in lo]).mean()
            - np.concatenate([high_groups[j] for j in hi]).mean()
            for lo, hi in zip(low_index, high_index)
        ],
        dtype=float,
    )
    race_low = np.array([g.mean() for g in low_groups], dtype=float)
    race_high = np.array([g.mean() for g in high_groups], dtype=float)
    race_draws = race_low[low_index].mean(axis=1) - race_high[high_index].mean(axis=1)
    pooled = (
        np.concatenate(low_groups).mean() - np.concatenate(high_groups).mean()
    )
    return (
        float(pooled * 100.0),
        float(np.quantile(pooled_draws, 0.025) * 100.0),
        float(np.quantile(pooled_draws, 0.975) * 100.0),
        float((race_low.mean() - race_high.mean()) * 100.0),
        float(np.quantile(race_draws, 0.025) * 100.0),
        float(np.quantile(race_draws, 0.975) * 100.0),
    )


def load_admission() -> pd.DataFrame:
    frame = pd.read_csv(ADMISSION_CSV)
    frame["admitted_for_gameplay"] = frame["admitted_for_gameplay"].astype(bool)
    return frame.set_index("route")


def spearman_block(x: np.ndarray, y: np.ndarray, label: str) -> dict:
    """Spearman rho plus an exact permutation test over the routes.

    With eight (or seven) routes every relabelling can be enumerated, so the
    permutation p-value is exact rather than sampled.
    """

    result = stats.spearmanr(x, y)
    rho = float(result.statistic)
    rank_x = stats.rankdata(x)
    rank_y = stats.rankdata(y)
    cx = rank_x - rank_x.mean()
    cy = rank_y - rank_y.mean()
    denominator = float(np.linalg.norm(cx) * np.linalg.norm(cy))
    permutations = np.array(list(itertools.permutations(cy)), dtype=float)
    permuted = permutations @ cx / denominator
    p_perm = float((np.abs(permuted) >= abs(rho) - 1e-12).mean())
    return {
        "predictor": label,
        "n_routes": int(len(x)),
        "spearman_rho": rho,
        "spearman_p_exact": float(result.pvalue),
        "permutation_p_two_sided": p_perm,
        "permutation_scheme": "exact enumeration of all route relabellings",
        "n_permutations": int(len(permutations)),
    }


def build_rows(runs: list[dict], admission: pd.DataFrame) -> tuple[pd.DataFrame, dict, list[str]]:
    rows = []
    hashes: dict[str, str] = {}
    problems: list[str] = []
    for run in runs:
        turns = pd.read_json(run["turns_path"], lines=True)
        summary = json.loads(run["summary_path"].read_text(encoding="utf-8"))
        route = run["route"]
        hashes[str(run["turns_path"].relative_to(ROOT).as_posix())] = sha256(run["turns_path"])

        n_races = int(turns["game_id"].nunique())
        n_decisions = int(len(turns))
        parse_failures = int(turns["parse_failed"].sum())
        if n_races != EXPECTED_RACES:
            problems.append(f"{route}: {n_races} races, expected {EXPECTED_RACES}")
        if n_decisions != EXPECTED_DECISIONS:
            problems.append(f"{route}: {n_decisions} decisions, expected {EXPECTED_DECISIONS}")
        if parse_failures != 0:
            problems.append(f"{route}: {parse_failures} parse failures, expected 0")
        if int(summary["parse_failures"]) != 0:
            problems.append(f"{route}: summary.json reports {summary['parse_failures']} parse failures")
        if run["manifest"]["run_phase"] != "confirmatory":
            problems.append(f"{route}: run_phase {run['manifest']['run_phase']}, expected confirmatory")

        cells = {}
        record: dict[str, object] = {
            "route": route,
            "short_name": SHORT_NAMES.get(route, route),
            "model_tag": run["model_tag"],
            "baseline_run_id": run["run_id"],
            "n_races_total": n_races,
            "n_decisions_total": n_decisions,
            "parse_failures": parse_failures,
        }
        for risk in RISKS:
            cell = turns[np.isclose(turns["max_private_risk"].astype(float), risk)]
            if cell["game_id"].nunique() != EXPECTED_RACES_PER_CELL:
                problems.append(
                    f"{route}: risk {risk} has {cell['game_id'].nunique()} races, "
                    f"expected {EXPECTED_RACES_PER_CELL}"
                )
            cells[risk] = cell
            race_mean, race_low, race_high, cell_races, cell_decisions = race_bootstrap(cell)
            pooled, pooled_low, pooled_high = race_bootstrap_pooled(cell)
            tag = f"risk_{risk:g}".replace(".", "p")
            record[f"unsafe_rate_{tag}"] = pooled
            record[f"unsafe_ci_low_{tag}"] = pooled_low
            record[f"unsafe_ci_high_{tag}"] = pooled_high
            record[f"unsafe_rate_race_mean_{tag}"] = race_mean
            record[f"unsafe_race_mean_ci_low_{tag}"] = race_low
            record[f"unsafe_race_mean_ci_high_{tag}"] = race_high
            record[f"n_races_{tag}"] = cell_races
            record[f"n_decisions_{tag}"] = cell_decisions
            summary_rate = float(summary["by_risk"][f"{risk:g}"]["unsafe_rate"])
            record[f"unsafe_rate_summary_json_{tag}"] = summary_rate
            if abs(summary_rate - pooled) > 5e-4:
                problems.append(
                    f"{route}: risk {risk} pooled rate {pooled:.4f} disagrees with "
                    f"summary.json {summary_rate:.4f}"
                )

        response = risk_response_bootstrap(cells[0.1], cells[0.9])
        record.update(
            {
                "risk_response_pp": response[0],
                "risk_response_ci_low_pp": response[1],
                "risk_response_ci_high_pp": response[2],
                "risk_response_race_mean_pp": response[3],
                "risk_response_race_mean_ci_low_pp": response[4],
                "risk_response_race_mean_ci_high_pp": response[5],
            }
        )

        within_cell_variance = float(
            np.nanmax(
                [cells[risk].groupby("game_id")["unsafe"].mean().var(ddof=0) for risk in RISKS]
            )
        )
        record["degenerate_step_policy"] = bool(within_cell_variance == 0.0)

        if route not in admission.index:
            raise RuntimeError(f"Gameplay route {route} is absent from the admission campaign table")
        adm = admission.loc[route]
        record.update(
            {
                "admission_run_id": int(adm["run_id"]),
                "admission_task_version": int(adm["task_version"]),
                "admission_overall_accuracy": float(adm["overall_accuracy"]),
                "admission_state_reconstruction_accuracy": float(adm["accuracy_state_reconstruction"]),
                "admission_terminal_scoring_accuracy": float(adm["accuracy_terminal_scoring"]),
                "admission_expected_payoff_accuracy": float(adm["accuracy_expected_payoff"]),
                "admitted_for_gameplay": bool(adm["admitted_for_gameplay"]),
                "admission_evidence_class": str(adm["evidence_class"]),
            }
        )
        rows.append(record)

    frame = pd.DataFrame(rows).sort_values("risk_response_pp", ascending=False).reset_index(drop=True)
    hashes[str(ADMISSION_CSV.relative_to(ROOT).as_posix())] = sha256(ADMISSION_CSV)
    return frame, hashes, problems


def correlations(frame: pd.DataFrame) -> dict:
    out = {}
    subsets = {
        "all_routes": frame,
        "excluding_boundary_case": frame[frame["route"] != BOUNDARY_ROUTE],
    }
    for name, subset in subsets.items():
        y = subset["risk_response_pp"].to_numpy(float)
        out[name] = {
            "routes": subset["route"].tolist(),
            "overall_accuracy": spearman_block(
                subset["admission_overall_accuracy"].to_numpy(float), y, "admission_overall_accuracy"
            ),
            "state_reconstruction_accuracy": spearman_block(
                subset["admission_state_reconstruction_accuracy"].to_numpy(float),
                y,
                "admission_state_reconstruction_accuracy",
            ),
        }
    return out


def admission_contrast(frame: pd.DataFrame) -> dict:
    out = {}
    for label, flag in (("admitted", True), ("not_admitted", False)):
        subset = frame[frame["admitted_for_gameplay"] == flag]
        values = subset["risk_response_pp"].to_numpy(float)
        out[label] = {
            "n_routes": int(len(values)),
            "mean_risk_response_pp": float(values.mean()),
            "min_risk_response_pp": float(values.min()),
            "max_risk_response_pp": float(values.max()),
            "per_route_risk_response_pp": {
                row.route: float(row.risk_response_pp) for row in subset.itertuples()
            },
        }
    out["difference_of_means_pp"] = (
        out["admitted"]["mean_risk_response_pp"] - out["not_admitted"]["mean_risk_response_pp"]
    )
    return out


def make_figure(frame: pd.DataFrame) -> list[Path]:
    ps.configure_publication_style()
    # Panel (a) carries eight directly labelled curves, so the two panels are
    # placed side by side on the full-width float rather than stacked in the
    # 5.17 in single column, where the labels would collide.
    fig, (ax_a, ax_b) = plt.subplots(
        1, 2, figsize=(ps.FULL_WIDTH_IN, 2.60), gridspec_kw={"width_ratios": [1.0, 1.0]}
    )
    colours = ps.EXTENDED_CATEGORICAL
    risks = np.array(RISKS, dtype=float)

    # Order the labels by the value they sit next to (the rate at risk 0.9) so the
    # leader lines run in parallel into the right-hand gutter and never cross.
    ordered = frame.sort_values("unsafe_rate_risk_0p9", ascending=False).reset_index(drop=True)
    label_slots = np.linspace(0.97, 0.05, len(ordered))
    for position, row in ordered.iterrows():
        colour = colours[position % len(colours)]
        admitted = bool(row["admitted_for_gameplay"])
        rates = np.array([row[f"unsafe_rate_risk_{r:g}".replace(".", "p")] for r in risks])
        low = np.array([row[f"unsafe_ci_low_risk_{r:g}".replace(".", "p")] for r in risks])
        high = np.array([row[f"unsafe_ci_high_risk_{r:g}".replace(".", "p")] for r in risks])
        ax_a.plot(
            risks,
            rates,
            color=colour,
            linewidth=1.5 if admitted else 1.1,
            linestyle="-" if admitted else (0, (3.5, 2.0)),
            alpha=1.0 if admitted else 0.62,
            marker=ps.MARKERS[position % len(ps.MARKERS)],
            markersize=3.4,
            zorder=3,
        )
        ax_a.fill_between(
            risks, low, high, color=colour, alpha=0.13 if admitted else 0.08, linewidth=0, zorder=2
        )
        ax_a.annotate(
            row["short_name"],
            xy=(0.9, rates[-1]),
            xytext=(1.00, label_slots[position]),
            textcoords="data",
            color=ps.INK if admitted else ps.MUTED,
            fontsize=8.0,
            va="center",
            ha="left",
            arrowprops={
                "arrowstyle": "-",
                "color": colour,
                "linewidth": 0.6,
                "shrinkA": 1.0,
                "shrinkB": 1.0,
                "alpha": 0.85,
            },
        )
    ax_a.set_xlabel("Assigned maximum private risk")
    ax_a.set_ylabel("Unsafe rate")
    ax_a.set_xticks(risks)
    ax_a.set_xlim(0.06, 1.72)
    ps.set_percent_axis(ax_a, ymax=1.04)
    ps.style_axis(ax_a)
    ps.panel_label(ax_a, "(a)", "Unsafe rate falls with assigned risk")

    label_offsets = {
        "anthropic/claude-opus-5@default": (5.0, 3.5, "left"),
        "anthropic/claude-sonnet-5@default": (5.5, 2.5, "left"),
        "openai/gpt-5.4-2026-03-05": (5.5, 3.0, "left"),
        "google/gemini-3-flash-preview": (2.0, -11.5, "left"),
        "openai/gpt-5.5-2026-04-23": (0.0, -12.0, "center"),
        "google/gemini-3.1-flash-lite-preview": (-4.0, 6.5, "right"),
        "openai/gpt-5.4-nano-2026-03-17": (6.0, 1.5, "left"),
        "openai/gpt-5.4-mini-2026-03-17": (6.0, -1.5, "left"),
        "google/gemini-3.5-flash-lite": (-6.0, -1.0, "right"),
    }
    # A route without a hand-placed label still gets drawn, offset to the right,
    # so adding one never silently drops a point from the panel.
    default_offset = (6.0, 2.0, "left")
    for row in frame.itertuples():
        admitted = bool(row.admitted_for_gameplay)
        colour = ps.BLUE if admitted else ps.RED
        ax_b.errorbar(
            row.admission_state_reconstruction_accuracy,
            row.risk_response_pp,
            yerr=[
                [row.risk_response_pp - row.risk_response_ci_low_pp],
                [row.risk_response_ci_high_pp - row.risk_response_pp],
            ],
            fmt="o",
            markersize=5.0,
            markerfacecolor=colour if admitted else ps.WHITE,
            markeredgecolor=colour,
            markeredgewidth=1.1,
            ecolor=colour,
            elinewidth=0.9,
            capsize=2.2,
            alpha=0.95,
            zorder=3,
        )
        ax_b.annotate(
            row.short_name,
            xy=(row.admission_state_reconstruction_accuracy, row.risk_response_pp),
            xytext=label_offsets.get(row.route, default_offset)[:2],
            textcoords="offset points",
            ha=label_offsets.get(row.route, default_offset)[2],
            fontsize=8.0,
            color=ps.INK if admitted else ps.MUTED,
            zorder=4,
        )
    ax_b.axvline(0.75, color=ps.LINE, linewidth=0.9, linestyle=(0, (4.0, 2.0)), zorder=1)
    ax_b.set_xlabel("State-reconstruction accuracy (admission audit)")
    ax_b.set_ylabel("Risk response (pp)")
    ax_b.set_xticks([0.2, 0.4, 0.6, 0.8, 1.0])
    ax_b.set_xlim(0.10, 1.28)
    ax_b.set_ylim(-8, 118)
    ps.style_axis(ax_b)
    ps.panel_label(ax_b, "(b)", "Audit score against risk response")

    fig.tight_layout(pad=0.4, w_pad=1.4)
    primary, *rest = FIGURE_STEMS
    written = list(ps.save_publication_figure(fig, primary, formats=("pdf", "png")))
    for stem in rest:
        stem.parent.mkdir(parents=True, exist_ok=True)
        for path in written[: len(("pdf", "png"))]:
            target = stem.with_suffix(path.suffix)
            target.write_bytes(path.read_bytes())
            written.append(target)
    return written


def main() -> None:
    DERIVED.mkdir(parents=True, exist_ok=True)
    runs = discover_runs()
    admission = load_admission()
    frame, hashes, problems = build_rows(runs, admission)
    if problems:
        raise RuntimeError("Integrity checks failed:\n  " + "\n  ".join(problems))
    # The count is declared beside the other frozen expectations rather than
    # written here, so re-running after a route is added fails loudly once, at
    # the constant, instead of silently tabulating a different sample.
    if len(frame) != EXPECTED_ROUTES:
        raise RuntimeError(
            f"Expected {EXPECTED_ROUTES} tabulated routes, found {len(frame)}"
        )

    corr = correlations(frame)
    contrast = admission_contrast(frame)
    failures = failure_record()

    frame.to_csv(DERIVED / "audit_versus_behaviour.csv", index=False)
    figures = make_figure(frame)

    payload = {
        "schema_version": "ai-race-audit-versus-behaviour-v1",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S+00:00"),
        "date": "2026-09-09",
        "evidence_class": "exploratory",
        "estimand": (
            "Per-endpoint decision-level Unsafe rate at each assigned maximum private risk, and its "
            "0.1-minus-0.9 contrast in percentage points, described alongside that endpoint's frozen "
            "admission-probe accuracy; the cross-endpoint association is descriptive over eight routes "
            "and is not an estimated population effect."
        ),
        "campaigns": {
            "gameplay": {
                "campaign_id": "baseline_campaign_v6",
                "protocol_id": BASELINE_PROTOCOL,
                "task_name": "ai-race-baseline",
                "task_version": "3",
                "run_phase": "confirmatory",
                "kaggle_identity": "daosyduyminh",
                "n_tabulated_routes": int(len(frame)),
            },
            "admission": {
                "campaign_id": "admission_campaign_v6",
                "protocol_id": ADMISSION_PROTOCOL,
                "task_name": "ai-race-frontier-admission",
                "kaggle_identity": "daosyduyminh",
            },
        },
        "uncertainty": {
            "method": "percentile bootstrap over game_id clusters, both seats of a race resampled together",
            "convention_source": "scripts/analyze_nplayer_scope_clustered.py::race_bootstrap",
            "seed": SEED,
            "n_resamples": N_BOOT,
            "races_per_risk_cell": EXPECTED_RACES_PER_CELL,
            "primary_estimand_weighting": "equal weight per decision (matches the run summary.json)",
            "secondary_estimand_weighting": "equal weight per race (the v1 house estimand), reported alongside",
            "note": (
                "Ten races per risk cell. The risk-response interval is therefore wide and is reported "
                "as measured."
            ),
        },
        "boundary_case": {
            "route": BOUNDARY_ROUTE,
            "reason": (
                "Degenerate step policy: Unsafe on every decision at risk 0.1 and Safe on every decision "
                "at 0.6 and 0.9, so within-cell variance is zero and the cluster bootstrap interval "
                "collapses to a point."
            ),
            "handling": "retained in the table; correlations reported with and without it",
        },
        "correlations": corr,
        "admission_contrast": contrast,
        "failed_runs": failures["failed_routes"],
        "source_sha256": hashes,
        "outputs": {
            "csv": str((DERIVED / "audit_versus_behaviour.csv").relative_to(ROOT).as_posix()),
            "figures": [str(p.relative_to(ROOT).as_posix()) for p in figures],
        },
    }
    (DERIVED / "audit_versus_behaviour.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )

    pd.set_option("display.width", 200)
    show = frame[
        [
            "short_name",
            "admission_overall_accuracy",
            "admission_state_reconstruction_accuracy",
            "unsafe_rate_risk_0p1",
            "unsafe_rate_risk_0p6",
            "unsafe_rate_risk_0p9",
            "risk_response_pp",
            "risk_response_ci_low_pp",
            "risk_response_ci_high_pp",
            "admitted_for_gameplay",
        ]
    ]
    print(show.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print()
    for name, block in corr.items():
        for key in ("overall_accuracy", "state_reconstruction_accuracy"):
            item = block[key]
            print(
                f"{name:26s} {key:30s} rho={item['spearman_rho']:+.4f} "
                f"p={item['spearman_p_exact']:.4f} p_perm={item['permutation_p_two_sided']:.4f} "
                f"n={item['n_routes']}"
            )
    print()
    for label in ("admitted", "not_admitted"):
        block = contrast[label]
        print(
            f"{label}: n={block['n_routes']} mean={block['mean_risk_response_pp']:.1f} pp "
            f"range=[{block['min_risk_response_pp']:.1f}, {block['max_risk_response_pp']:.1f}] pp"
        )
        for route, value in block["per_route_risk_response_pp"].items():
            print(f"    {route:40s} {value:+.1f} pp")
    print()
    for record in failures["failed_routes"]:
        print(f"failure record kept, not tabulated: {record['route']} ({record['status']})")
    for path in figures:
        print(f"figure: {path.relative_to(ROOT).as_posix()}")


if __name__ == "__main__":
    main()
