#!/usr/bin/env python3
"""Cross-model pilot synthesis for the AI Race manuscript's pending RQ sections.

Mines three questions across every audited-clean frontier pilot run (results/frontier):
(1) does the risk-response level/shape and the frozen human-comparison scorecard
(E1-E8, results/scripts/human_reference.json) agree across model checkpoints;
(2) does the numeric risk-persona framing sweep (R1-R6 seat roles) move the Unsafe
rate more than the max_private_risk treatment itself; (3) which human dynamic
effects (E1-E4) are even estimable per checkpoint, and why not when they aren't.

This script never promotes pilot evidence to confirmatory and never hides an
excluded run: every directory under results/frontier is re-audited at run time by
comparing its run_manifest.json counts against the actual turns.jsonl/races.csv
row counts, and every exclusion is recorded in analysis_manifest.json with its
reason. It calls the project's own fail-closed analyzer (analyze_ai_race.py) for
every statistic that tool already computes; it does not recompute cluster-robust
inference itself.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "results" / "cross_model_pilot_synthesis"
FIGURES = OUTPUT / "figures"
DATA = OUTPUT / "data"
PYTHON = ROOT / ".venv-kaggle" / "bin" / "python"
ANALYZER = ROOT / "results" / "scripts" / "analyze_ai_race.py"
FRONTIER_ROOT = ROOT / "results" / "frontier"

ALLOW_FLAGS = [
    "--allow-nonconfirmatory-runs",
    "--allow-nonfinal-runs",
    "--allow-mixed-protocols",
    "--allow-missing-persona-condition",
]

PALETTE = {
    "navy": "#0B132B",
    "blue": "#2563EB",
    "cyan": "#06B6D4",
    "teal": "#0D9488",
    "amber": "#F59E0B",
    "red": "#DC2626",
    "slate": "#64748B",
    "grid": "#DCE3ED",
}
MODEL_COLORS = [PALETTE["blue"], PALETTE["cyan"], PALETTE["teal"], PALETTE["amber"], PALETTE["red"]]
MODEL_ORDER = [
    "gpt-5-nano",
    "gpt-5.4-nano",
    "google/gemini-3-flash-preview",
    "google/gemini-3.1-flash-lite-preview",
    "google/gemini-3.5-flash-lite",
]
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
}


def setup_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titleweight": "bold",
            "axes.titlesize": 13,
            "axes.labelsize": 10,
            "axes.edgecolor": PALETTE["grid"],
            "axes.linewidth": 0.8,
            "xtick.color": PALETTE["slate"],
            "ytick.color": PALETTE["slate"],
            "text.color": PALETTE["navy"],
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )


def save_figure(fig: plt.Figure, stem: Path) -> None:
    fig.savefig(stem.with_suffix(".png"), dpi=220, bbox_inches="tight", facecolor="white")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
    plt.close(fig)


def sha256_of(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- 1. re-audit every frontier run directory -------------------------------------


def count_lines(path: Path) -> int:
    with open(path, "rb") as f:
        return sum(1 for _ in f)


def audit_frontier() -> tuple[list[str], dict[str, str]]:
    clean: list[str] = []
    excluded: dict[str, str] = {}
    for manifest in sorted(FRONTIER_ROOT.rglob("run_manifest.json")):
        d = manifest.parent
        rel = str(d.relative_to(ROOT))
        try:
            m = json.loads(manifest.read_text())
        except Exception as exc:  # noqa: BLE001
            excluded[rel] = f"unreadable manifest: {exc}"
            continue
        status = m.get("status")
        if status != "completed":
            excluded[rel] = f"run_manifest status={status}"
            continue
        n_turns, n_races = m.get("n_turns"), m.get("n_races")
        turns_f, races_f = d / "turns.jsonl", d / "races.csv"
        actual_turns = count_lines(turns_f) if turns_f.exists() else None
        actual_races = (count_lines(races_f) - 1) if races_f.exists() else None
        if n_turns is not None and actual_turns is not None and n_turns != actual_turns:
            excluded[rel] = f"manifest declares {n_turns} turns/{n_races} races; files contain {actual_turns} turns/{actual_races} races"
            continue
        if n_races is not None and actual_races is not None and n_races != actual_races:
            excluded[rel] = f"manifest declares {n_races} races; races.csv has {actual_races} rows"
            continue
        clean.append(rel)
    return clean, excluded


# --- 2. per-checkpoint neutral/baseline lanes via the project analyzer -------------

NEUTRAL_INPUTS = {
    "gpt-5-nano": [
        "results/frontier/openai/baseline/gpt-5-nano",
        "results/frontier/openai/persona/R0_neutral/gpt-5-nano",
    ],
    "gpt-5.4-nano": [
        "results/frontier/openai/baseline/gpt-5.4-nano",
        "results/frontier/openai/persona/R0_neutral/gpt-5.4-nano",
    ],
    "google/gemini-3-flash-preview": [
        "results/frontier/baseline/google-gemini-3-flash-preview",
        "results/frontier/persona/R0_neutral/google-gemini-3-flash-preview",
        # api_5games_allrisk excluded: shares game_id/game_seed with 15 of the 30
        # baseline races (same realised horizon/stopping draws under this project's
        # CRN design), but is an independently-sampled re-run, not a duplicated log --
        # turn-by-turn actions differ in most risk-0.6/0.9 games (risk-0.1 matches
        # exactly only because that cell is ~100% Unsafe either way). Pooling would
        # violate CRN-independence; already flagged "superseded overlapping pilot" in
        # analyze_two_player_paper_figures.py for what is presumably this same reason.
    ],
    "google/gemini-3.1-flash-lite-preview": [
        "results/frontier/baseline/google-gemini-3.1-flash-lite-preview",
    ],
    "google/gemini-3.5-flash-lite": [
        "results/frontier/baseline/google-gemini-3.5-flash-lite",
    ],
}


def run_lane(model_key: str, input_dirs: list[str], clean: set[str], out_dir: Path) -> Path | None:
    dirs = [d for d in input_dirs if d in clean]
    missing = [d for d in input_dirs if d not in clean]
    if missing:
        print(f"  [{model_key}] skipping excluded/absent inputs: {missing}")
    if not dirs:
        return None
    out_dir.mkdir(parents=True, exist_ok=True)
    base_cmd = [str(PYTHON), str(ANALYZER), "--output", str(out_dir), *ALLOW_FLAGS]
    for d in dirs:
        base_cmd += ["--input", d]
    # Try with --fit-logit first (needed for the E1-E4 dynamic coefficients); some
    # checkpoints hit an exact singular design matrix (zero-variance first_round_unsafe,
    # or progress_gap_before collinear with the interaction when both players' previous
    # action is Safe) and the analyzer crashes rather than emit an unreliable estimate.
    # That crash is itself part of the finding (see INSIGHTS.md), so fall back to a
    # logit-free run only to recover the remaining descriptive tables.
    result = subprocess.run(base_cmd + ["--fit-logit"], cwd=ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"  [{model_key}] --fit-logit crashed (rank-deficient design matrix); retrying without it")
        result = subprocess.run(base_cmd, cwd=ROOT, capture_output=True, text=True)
        if result.returncode != 0:
            print(result.stderr[-2000:])
    return out_dir


def read_csv_rows(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path) as f:
        return list(csv.DictReader(f))


def weighted_risk_response(out_dir: Path) -> dict[float, tuple[float, int]]:
    rows = read_csv_rows(out_dir / "unsafe_by_risk_model_player.csv")
    agg: dict[float, list[float]] = defaultdict(lambda: [0.0, 0])
    for row in rows:
        risk = float(row["max_private_risk"])
        n = int(row["n_players"])
        rate = float(row["mean_player_unsafe_rate"])
        agg[risk][0] += rate * n
        agg[risk][1] += n
    return {risk: (s / n, n) for risk, (s, n) in agg.items()}


DYNAMIC_EFFECTS = {"E1", "E2", "E3", "E4"}  # need the Model-6 own*opponent*gap interaction


def spec6_converged(out_dir: Path) -> bool:
    meta_path = out_dir / "clustered_logit_metadata.json"
    if not meta_path.exists():
        return False
    meta = json.loads(meta_path.read_text())
    for spec in meta.get("specifications", []):
        if spec.get("specification") == "6":
            return bool(spec.get("converged"))
    return False


def human_scorecard(out_dir: Path) -> dict[str, str]:
    rows = read_csv_rows(out_dir / "human_comparison.csv")
    verdicts = {row["effect_id"]: row["verdict"] for row in rows}
    # The tool scores E1-E4 off the Model-6 fit even when statsmodels reports
    # non-convergence (it still returns whatever coefficients the optimizer stopped
    # at). A nominal p-value from a non-converged fit is not evidence either way,
    # so treat those four effects as inconclusive here regardless of the raw verdict.
    if not spec6_converged(out_dir):
        for effect in DYNAMIC_EFFECTS:
            if effect in verdicts:
                verdicts[effect] = "inconclusive"
    return verdicts


# --- 3. persona/role gradient, read directly from players.csv ----------------------

ROLE_LABEL = {f"risk-{i}": f"R{i}" for i in range(1, 7)}
GRADIENT_MODELS = ["gpt-5-nano", "gpt-5.4-nano", "google/gemini-3-flash-preview"]


def persona_gradient(clean: set[str]) -> dict[str, dict[str, tuple[float, int]]]:
    agg: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(lambda: [0.0, 0]))
    for rel in clean:
        if "/risk_matrix/" not in rel and not rel.split("/")[-2].endswith("_risk_matrix"):
            continue
        players = ROOT / rel / "players.csv"
        if not players.exists():
            continue
        for row in read_csv_rows(players):
            role = row.get("persona_role", "")
            if role not in ROLE_LABEL:
                continue
            model = row["model"]
            if model not in GRADIENT_MODELS:
                continue
            r = ROLE_LABEL[role]
            rate = float(row["unsafe_frequency"])
            agg[model][r][0] += rate
            agg[model][r][1] += 1
    return {m: {r: (s / n, n) for r, (s, n) in roles.items()} for m, roles in agg.items()}


# --- figures -------------------------------------------------------------------------


def fig_risk_response(risk_response: dict[str, dict[float, tuple[float, int]]]) -> None:
    setup_plot()
    fig, ax = plt.subplots(figsize=(9.6, 5.6))
    fig.subplots_adjust(bottom=0.25, top=0.88)
    ax.axhspan(40, 75, color=PALETTE["grid"], alpha=0.6, zorder=0)
    ax.text(0.03, 76.5, "Human phi_U interval [40, 75]%", fontsize=8.5, color=PALETTE["slate"])
    risks = [0.1, 0.6, 0.9]
    risk_labels = ["10%", "60%", "90%"]
    x = list(range(len(MODEL_ORDER)))
    width = 0.23
    risk_colors = [PALETTE["cyan"], PALETTE["amber"], PALETTE["red"]]
    for j, (risk, label, color) in enumerate(zip(risks, risk_labels, risk_colors)):
        heights = [100 * risk_response.get(model, {}).get(risk, (float("nan"), 0))[0] for model in MODEL_ORDER]
        positions = [xi + (j - 1) * width for xi in x]
        ax.bar(positions, heights, width=width, color=color, label=f"Risk cap {label}", zorder=3)
    ax.set_title("Five checkpoints have clearly different risk responses")
    ax.set_xlabel("Checkpoint")
    ax.set_ylabel("Mean player-level Unsafe rate (%)")
    ax.set_xticks(x, [MODEL_LABELS[m].replace(" ", "\n", 1) for m in MODEL_ORDER], fontsize=8.5)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02))
    fig.text(
        0.135, 0.025,
        "Pilot, neutral/no-persona condition only; player-weighted rates across matched batches per checkpoint.\n"
        "Human reference: Fernandez Domingos & Han (2026), Table S3 (phi_U = 0.584, interval from the analyzer's E7 gate).",
        fontsize=8, color=PALETTE["slate"],
    )
    save_figure(fig, FIGURES / "cross_model_risk_response_neutral")


VERDICT_COLOR = {
    "replicated": PALETTE["teal"],
    "not_replicated": PALETTE["red"],
    "inconclusive": PALETTE["grid"],
}
VERDICT_CODE = {"replicated": 1.0, "not_replicated": 0.0, "inconclusive": 0.5}
EFFECT_ORDER = ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
EFFECT_LABELS = {
    "E1": "E1\nopp. prev.\nunsafe",
    "E2": "E2\nprogress\ngap",
    "E3": "E3\nfirst-round\nunsafe",
    "E4": "E4\nown prev.\nunsafe (null)",
    "E5": "E5\n0.6 v 0.9\n(null)",
    "E6": "E6\n0.1 v 0.6",
    "E7": "E7\nphi_U\nlevel",
    "E8": "E8\nshare\nAS",
}


def fig_scorecard(scorecards: dict[str, dict[str, str]]) -> None:
    setup_plot()
    models = [m for m in MODEL_ORDER if m in scorecards]
    fig, ax = plt.subplots(figsize=(9.6, 0.85 * len(models) + 2.2))
    grid = [[VERDICT_CODE.get(scorecards[m].get(e, ""), 0.5) for e in EFFECT_ORDER] for m in models]
    cmap = plt.matplotlib.colors.ListedColormap([PALETTE["red"], PALETTE["grid"], PALETTE["teal"]])
    ax.imshow(grid, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    for i, row in enumerate(grid):
        for j, val in enumerate(row):
            verdict = scorecards[models[i]].get(EFFECT_ORDER[j], "")
            mark = {"replicated": "✓", "not_replicated": "✗", "inconclusive": "?"}.get(verdict, "")
            text_color = "white" if verdict in ("replicated", "not_replicated") else PALETTE["slate"]
            ax.text(j, i, mark, ha="center", va="center", color=text_color, fontsize=13, weight="bold")
    ax.set_xticks(range(len(EFFECT_ORDER)), [EFFECT_LABELS[e] for e in EFFECT_ORDER], fontsize=8)
    ax.set_yticks(range(len(models)), [MODEL_LABELS[m] for m in models], fontsize=9.5)
    ax.set_title("No checkpoint replicates the human study fully, and none fails the same way")
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xticks([x - 0.5 for x in range(1, len(EFFECT_ORDER))], minor=True)
    ax.set_yticks([y - 0.5 for y in range(1, len(models))], minor=True)
    ax.grid(which="minor", color="white", linewidth=2)
    ax.tick_params(which="minor", bottom=False, left=False)
    handles = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["teal"], markersize=12, label="Replicated"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["red"], markersize=12, label="Not replicated"),
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=PALETTE["grid"], markersize=12, label="Inconclusive / not estimable"),
    ]
    ax.legend(handles=handles, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3, fontsize=8.5)
    fig.text(
        0.01, 0.0,
        "One checkpoint per row analyzed alone (never pooled across checkpoints). E1-E4 need the paper's Model-6 dynamic\n"
        "specification (own/opponent previous action x progress gap); most checkpoints cannot even fit it (see text).",
        fontsize=8, color=PALETTE["slate"],
    )
    save_figure(fig, FIGURES / "human_effect_scorecard")


def fig_persona_gradient(gradient: dict[str, dict[str, tuple[float, int]]]) -> None:
    setup_plot()
    fig, ax = plt.subplots(figsize=(9.6, 5.8))
    roles = [f"R{i}" for i in range(1, 7)]
    x = list(range(len(roles)))
    width = 0.24
    for j, (color, model) in enumerate(zip(MODEL_COLORS, GRADIENT_MODELS)):
        cells = gradient.get(model, {})
        ys = [100 * cells[r][0] if r in cells else None for r in roles]
        positions = [xi + (j - 1) * width for xi in x]
        ax.bar(positions, ys, width=width, color=color, label=MODEL_LABELS[model], zorder=3)
    ax.set_title("Risk-persona framing moves Unsafe play far more than the risk treatment", pad=14)
    ax.set_xlabel("Assigned seat risk-attitude persona (R1 = most risk-averse framing ... R6 = most risk-seeking framing)")
    ax.set_ylabel("Mean player-level Unsafe rate (%),\npooled across 0.1/0.6/0.9 risk and opponent role")
    ax.set_xticks(x, roles)
    ax.set_ylim(0, 108)
    ax.grid(axis="y", color=PALETTE["grid"], linewidth=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.02), fontsize=8.5)
    ax.text(
        0.01, -0.18,
        "Pilot; persona_condition is injected into the same frozen, hash-verified prompt template (prompt_version unchanged).\n"
        "GPT-5/5.4 nano: n=360 players per bar. Persona is perfectly confounded with run\n"
        "batch/protocol signature (see text) -- a large, consistent, diagnostic pattern, not yet an isolated causal effect.",
        transform=ax.transAxes, fontsize=8, color=PALETTE["slate"],
    )
    save_figure(fig, FIGURES / "persona_role_gradient")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    DATA.mkdir(parents=True, exist_ok=True)

    clean_list, excluded = audit_frontier()
    clean = set(clean_list)
    print(f"Audited results/frontier: {len(clean)} clean run directories, {len(excluded)} excluded.")

    risk_response: dict[str, dict[float, tuple[float, int]]] = {}
    scorecards: dict[str, dict[str, str]] = {}
    for model_key, inputs in NEUTRAL_INPUTS.items():
        safe_name = model_key.replace("/", "_")
        out_dir = OUTPUT / "_lane_outputs" / safe_name
        run_lane(model_key, inputs, clean, out_dir)
        risk_response[model_key] = weighted_risk_response(out_dir)
        scorecards[model_key] = human_scorecard(out_dir)

    gradient = persona_gradient(clean)

    fig_risk_response(risk_response)
    fig_scorecard(scorecards)
    fig_persona_gradient(gradient)

    with open(DATA / "risk_response_by_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "max_private_risk", "mean_unsafe_rate", "n_players"])
        for model, cells in risk_response.items():
            for risk, (rate, n) in sorted(cells.items()):
                w.writerow([model, risk, rate, n])

    with open(DATA / "human_effect_scorecard.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", *EFFECT_ORDER])
        for model in MODEL_ORDER:
            if model in scorecards:
                w.writerow([model, *[scorecards[model].get(e, "") for e in EFFECT_ORDER]])

    with open(DATA / "persona_role_gradient.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "role", "mean_unsafe_rate", "n_players"])
        for model, roles in gradient.items():
            for role in sorted(roles, key=lambda r: int(r[1:])):
                rate, n = roles[role]
                w.writerow([model, role, rate, n])

    manifest = {
        "schema": "cross-model-pilot-synthesis-v1",
        "generated_by": "results/scripts/build_cross_model_pilot_synthesis.py",
        "n_clean_run_dirs": len(clean),
        "excluded_run_dirs": excluded,
        "neutral_lane_inputs": NEUTRAL_INPUTS,
        "gradient_models": GRADIENT_MODELS,
        "figures": {},
    }
    for fig_path in sorted(FIGURES.glob("*.png")):
        manifest["figures"][fig_path.name] = sha256_of(fig_path)
    with open(OUTPUT / "analysis_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"Wrote figures to {FIGURES}, data to {DATA}, manifest to {OUTPUT/'analysis_manifest.json'}")


if __name__ == "__main__":
    main()
