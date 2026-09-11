"""What the reduced evolutionary model predicts, and what two routes actually did.

The claim.  The reconstructed evolutionary game, run at the parameter point its
own source text calls the reference, is a step: it holds unsafe play at the
ceiling through 0.6 and drops it to the floor by 0.9.  The two hosted routes
that produced the archived self-play are ramps.  They fall at every step, they
fall by different amounts, and at 0.9 they sit tens of points above a model that
has already reached zero.  Nearest-rule labels are offered as a lens on why, and
the last panel is the reason that lens is never promoted to a finding.

WHICH ROUTES.  Two, and only two, and they are named.  The archived comparison
was produced by ``anthropic/claude-sonnet-5@default`` and
``google/gemini-3-flash-preview``; the ``context`` column of
``llm_strategy_summary_primary_t0.csv`` abbreviates them to "claude" and
"gemini", which is why every earlier version of this figure carried a legend
that could have meant any of the five routes the paper admits or the nine it
audits.  ``ROUTE_OF_CONTEXT`` maps the abbreviation back to the exact route and
``verify_routes`` refuses to draw if the reconstruction manifest disagrees.

This is NOT a figure about admission.  Five routes clear the comprehension gate
(``results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv``:
Gemini 3 Flash, Claude Opus 5, GPT-5.4, GPT-5.5, Claude Sonnet 5).  Two of them
were run through the evolutionary comparison protocol.  A caption that calls
these "the two admitted frontier routes" is wrong about the paper's own gate,
and the replacement wording is in ``docs/handoff-caption-fixes-egt.md``.

UNCERTAINTY.  The evolutionary curves carry a band and the route points do not,
and that asymmetry is the artifacts' and not a drawing choice.
``egt_stationary_summary.csv`` records ``unsafe_frequency_min`` and
``unsafe_frequency_max`` over four independent seeded chains, which
``reconstruction_manifest.json`` labels
``between_independent_chain_range_diagnostic_not_confidence_interval``; it is
drawn, and it is called what it is.  ``llm_strategy_summary_primary_t0.csv``
carries one rate per route and risk and no interval of any kind, so none is
drawn.  A binomial interval over 186 decisions would be available and would be
wrong: the decisions are 10 races of repeated self-play, not 186 independent
draws.  The panel says so on its face rather than in a caption.

Nothing here is fitted.  The two evolutionary parameter points are the ones the
source text declares, the route rates are pooled decision frequencies, and the
nearest-rule assignment is a distance to a fixed rule and not an estimate of a
latent policy.
"""

from __future__ import annotations

import csv
import json
import shutil
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import numpy as np


HERE = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "results" / "frontier" / "egt_frontier_comparison_v2"
COMPARISON = OUTPUT / "theory_llm_comparison.csv"
MODEL_SUMMARY = OUTPUT / "llm_strategy_summary_primary_t0.csv"
STATIONARY = OUTPUT / "egt_stationary_summary.csv"
MANIFEST = OUTPUT / "reconstruction_manifest.json"
FIGDIR = ROOT / "figures" / "paper"

RISKS = (0.1, 0.6, 0.9)
STRATEGIES = ("AS", "AU", "CS", "CAS")

# The ``context`` column is an abbreviation invented by the comparison builder.
# It is not a route, and a figure that prints it is a figure whose series cannot
# be identified.  These are the routes, exactly as the manuscript spells them.
ROUTE_OF_CONTEXT = {
    "claude": "anthropic/claude-sonnet-5@default",
    "gemini": "google/gemini-3-flash-preview",
}
# The manifest spells the same two routes with hyphens.  Kept separate so the
# check below is a real comparison and not a restatement.
MANIFEST_TAG_OF_CONTEXT = {
    "claude": "anthropic-claude-sonnet-5-default",
    "gemini": "google-gemini-3-flash-preview",
}

# Strategy colour.  Hue is the action the strategy starts from and falls back
# to, using the two inks ``figstyle`` already reserves for SAFE and UNSAFE, so a
# reader who has learnt the paper's two actions has already learnt these four
# strategies.  Lightness is the second axis of the rule set: dark is
# unconditional, light is conditional on what the opponent just did.  The same
# four colours are imported by ``build_egt_frontier_invasion.py``, which is the
# whole point of defining them once.
STRATEGY_LIGHT = {"CS": "#a6dba0", "CAS": "#f4a582"}
STRATEGY_NAME = {
    "AS": "Always Safe",
    "AU": "Always Unsafe",
    "CS": "Conditional, starts Safe",
    "CAS": "Conditional, starts Unsafe",
}


def style():
    """``figstyle``, imported late and on demand.

    ``scripts/build_supplementary_figures.py`` imports ``build_derived`` and
    ``read_csv`` from this module after it has configured its own style module.
    ``figstyle`` rewrites ``rcParams`` at import time, so importing it at the top
    of this file would silently restyle every figure that other builder draws
    afterwards.  Importing it inside the drawing functions keeps this module's
    data half free of side effects.
    """
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    import figstyle

    return figstyle


def strategy_colours() -> dict[str, str]:
    """The one strategy palette, shared by both EGT figures."""
    S = style()

    return {
        "AS": S.SAFE_C,
        "AU": S.UNSAFE_C,
        "CS": STRATEGY_LIGHT["CS"],
        "CAS": STRATEGY_LIGHT["CAS"],
    }


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def f(row: dict[str, str], key: str) -> float:
    return float(row[key])


def verify_routes(model_rows: list[dict[str, str]]) -> None:
    """Refuse to draw if the abbreviations do not map to the archived routes.

    The figure names two routes in its legend.  That name is a claim, and the
    manifest is the thing that settles it, so the claim is checked against the
    manifest every time the figure is built rather than trusted because someone
    wrote it down once.
    """
    contexts = sorted({row["context"] for row in model_rows})
    if contexts != sorted(ROUTE_OF_CONTEXT):
        raise SystemExit(
            f"the comparison summary carries contexts {contexts}, which this "
            f"figure cannot name; known abbreviations are {sorted(ROUTE_OF_CONTEXT)}"
        )
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    archived = sorted(manifest["llm_comparison"]["frontier_primary"]["model_routes"])
    expected = sorted(MANIFEST_TAG_OF_CONTEXT[c] for c in contexts)
    if archived != expected:
        raise SystemExit(
            f"{MANIFEST.name} says the comparison ran {archived}, but this figure "
            f"would label the series {expected}"
        )


def theory_bands() -> dict[str, dict[float, tuple[float, float, float]]]:
    """Mean and between-chain range for each declared evolutionary regime.

    ``reconstruction_manifest.json`` calls this range
    ``between_independent_chain_range_diagnostic_not_confidence_interval``.
    It is the only uncertainty either artifact behind this figure carries, and
    it is not a confidence interval; the panel label says so.
    """
    bands: dict[str, dict[float, tuple[float, float, float]]] = {}
    for row in read_csv(STATIONARY):
        bands.setdefault(row["regime"], {})[round(f(row, "max_private_risk"), 1)] = (
            f(row, "unsafe_frequency_mean"),
            f(row, "unsafe_frequency_min"),
            f(row, "unsafe_frequency_max"),
        )
    for regime in ("main_reference", "reported_best_fit"):
        if set(bands.get(regime, {})) != set(RISKS):
            raise SystemExit(f"{STATIONARY.name} is missing risks for {regime}")
    return bands


def build_derived(
    comparison: list[dict[str, str]],
    model_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    """One row per route and risk.

    The signature is unchanged because ``scripts/build_supplementary_figures.py``
    imports this function.  The route name is an added column, not a renamed
    one, so nothing downstream loses a key it was reading.
    """
    comp = {round(f(row, "max_private_risk"), 1): row for row in comparison}
    rows: list[dict[str, object]] = []
    for row in model_rows:
        risk = round(f(row, "max_private_risk"), 1)
        model = row["context"]
        theory = f(comp[risk], "theory_unsafe_main_reference")
        best_fit = f(comp[risk], "theory_unsafe_reported_best_fit")
        frontier = f(row, "unsafe_rate_decision_weighted")
        rows.append(
            {
                "model": model,
                "route": ROUTE_OF_CONTEXT.get(model, model),
                "max_private_risk": risk,
                "frontier_unsafe_rate": frontier,
                "frontier_unsafe_rate_interval": "none_in_artifact",
                "decisions": int(f(row, "decisions")),
                "player_trajectories": int(f(row, "player_trajectories")),
                "theory_main_reference": theory,
                "theory_reported_best_fit": best_fit,
                "frontier_minus_theory_main_pp": 100 * (frontier - theory),
                "frontier_minus_theory_best_fit_pp": 100 * (frontier - best_fit),
                "mean_minimum_mismatch_rate": f(row, "mean_minimum_mismatch_rate"),
                "unique_classification_rate": f(row, "unique_classification_rate"),
                "fractional_nearest_AS": f(row, "fractional_nearest_AS"),
                "fractional_nearest_AU": f(row, "fractional_nearest_AU"),
                "fractional_nearest_CS": f(row, "fractional_nearest_CS"),
                "fractional_nearest_CAS": f(row, "fractional_nearest_CAS"),
            }
        )
    return rows


def report(rows: list[dict[str, object]], bands: dict) -> None:
    """Print every number the figure draws, so the figure can be checked by eye."""
    print(f"  routes: {sorted({str(r['route']) for r in rows})}")
    for regime, label in (("main_reference", "beta=2, mu=0.02"),
                          ("reported_best_fit", "beta=0.01, mu=0.05")):
        for risk in RISKS:
            mean, lo, hi = bands[regime][risk]
            print(f"  EGT {regime:<18} {label:<18} risk {risk}: "
                  f"{100 * mean:6.2f}%  chains {100 * lo:6.2f}-{100 * hi:6.2f}%")
    for context in sorted({str(r["model"]) for r in rows}):
        for row in sorted((r for r in rows if r["model"] == context),
                          key=lambda r: float(r["max_private_risk"])):
            print(f"  {str(row['route']):<36} risk {float(row['max_private_risk'])}: "
                  f"unsafe {100 * float(row['frontier_unsafe_rate']):5.1f}% over "
                  f"{row['decisions']} decisions and {row['player_trajectories']} "
                  f"trajectories, no interval in the artifact; minus reference "
                  f"{float(row['frontier_minus_theory_main_pp']):+6.1f} pp")


def _route_series(rows, context):
    return sorted((r for r in rows if r["model"] == context),
                  key=lambda r: float(r["max_private_risk"]))


def panel_a(ax, rows, bands) -> None:
    """The model is a step and the routes are ramps."""
    S = style()

    grid = np.array(RISKS)
    key_y = 24.0
    for regime, colour, dash, name in (
        ("main_reference", S.INK, "-", r"EGT reference $\beta$=2"),
        ("reported_best_fit", S.INK_2, (0, (4.0, 1.6)), r"EGT best fit $\beta$=0.01"),
    ):
        mean = np.array([100 * bands[regime][r][0] for r in RISKS])
        lo = np.array([100 * bands[regime][r][1] for r in RISKS])
        hi = np.array([100 * bands[regime][r][2] for r in RISKS])
        ax.fill_between(grid, lo, hi, color=colour, alpha=0.20, lw=0, zorder=2)
        ax.plot(grid, mean, lw=1.4, color=colour, ls=dash, marker="o", ms=2.6,
                mec=S.SURFACE, mew=0.5, zorder=3, clip_on=False)
        # The two model curves cross both route lines, so a label at either end
        # of a curve lands on something.  They are named instead in the one
        # region of the panel that no series enters, with a line sample each
        # because solid against dashed is half of what tells them apart.
        ax.plot([0.30, 0.38], [key_y, key_y], lw=1.4, color=colour, ls=dash,
                zorder=6, clip_on=False)
        ax.annotate(name, xy=(0.40, key_y), xytext=(0, 0),
                    textcoords="offset points", ha="left", va="center",
                    fontsize=S.FS_NOTE, color=colour, zorder=6)
        key_y -= 11.0

    for context, route in sorted(ROUTE_OF_CONTEXT.items()):
        series = _route_series(rows, context)
        y = np.array([100 * float(r["frontier_unsafe_rate"]) for r in series])
        ax.plot(grid, y, lw=1.6, color=S.ROUTE_C[route], marker=S.ROUTE_M[route],
                ms=3.4, mec=S.SURFACE, mew=0.6, zorder=5, clip_on=False)
        S.direct_label(ax, 0.9, y[-1], S.ROUTE_LABEL[route], color=S.ROUTE_C[route])

    S.rate_axis(ax, label="Unsafe play (%)")
    S.risk_axis(ax)
    ax.set_xlim(0.0, 1.46)
    ax.set_ylim(-2, 104)
    S.strip(ax, grid_axis="y")
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    S.panel(ax, "a", "The model is a step, the routes are ramps")
    ax.annotate(
        "Band: range over four independent chains, not a confidence interval.\n"
        "Route points carry no interval in the archived summary.",
        xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -33),
        textcoords="offset points", ha="left", va="top", fontsize=S.FS_NOTE,
        color=S.MUTED, linespacing=1.35, annotation_clip=False)


def panel_b(ax, rows) -> None:
    """The same departure, signed, with no dodge on the risk axis."""
    S = style()

    grid = np.array(RISKS)
    # Bounded to the risk range rather than to the axes: the axes run past 1 so
    # the route names have room, and a rule drawn out there offers a reading of
    # a risk the protocol never set.
    ax.plot([0.0, 1.0], [0.0, 0.0], color=S.MUTED, lw=0.8, zorder=1)
    for context, route in sorted(ROUTE_OF_CONTEXT.items()):
        series = _route_series(rows, context)
        y = np.array([float(r["frontier_minus_theory_main_pp"]) for r in series])
        ax.plot(grid, y, lw=1.6, color=S.ROUTE_C[route], marker=S.ROUTE_M[route],
                ms=3.4, mec=S.SURFACE, mew=0.6, zorder=4, clip_on=False)
        S.direct_label(ax, 0.9, y[-1], S.ROUTE_SHORT[route], color=S.ROUTE_C[route])

    S.risk_axis(ax)
    ax.set_xlim(0.0, 1.46)
    ax.set_ylim(-70, 70)
    ax.set_yticks([-60, -30, 0, 30, 60])
    ax.set_ylabel("Observed minus reference (pp)")
    S.strip(ax, grid_axis="y")
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    S.panel(ax, "b", "Below the model, then far above it")
    ax.annotate("Points sit on the risk they were run at; no dodge is applied.",
                xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -33),
                textcoords="offset points", ha="left", va="top",
                fontsize=S.FS_NOTE, color=S.MUTED, annotation_clip=False)


def panel_c(ax, rows) -> None:
    """Nearest-rule composition, with each stack told apart from its neighbour."""
    S = style()

    colours = strategy_colours()
    x = np.arange(len(RISKS), dtype=float)
    width = 0.30
    offset = {"claude": -0.24, "gemini": 0.24}
    for context, route in sorted(ROUTE_OF_CONTEXT.items()):
        series = _route_series(rows, context)
        bottom = np.zeros(len(RISKS))
        for strategy in STRATEGIES:
            values = np.array([float(r[f"fractional_nearest_{strategy}"]) for r in series])
            ax.bar(x + offset[context], values, width=width, bottom=bottom,
                   color=colours[strategy], edgecolor=S.SURFACE, linewidth=0.6,
                   zorder=3)
            for xi, value, base in zip(x, values, bottom):
                # A segment that can hold its own name does not need a legend,
                # and a legend is what a stacked bar cannot afford here.
                if value >= 0.16:
                    ax.text(xi + offset[context], base + value / 2, strategy,
                            ha="center", va="center", fontsize=S.FS_NOTE,
                            color=S.SURFACE if strategy in ("AS", "AU") else S.INK,
                            zorder=4)
            bottom += values
        for xi in x:
            # The stacks are told apart by a name above each one.  The route
            # colours are deliberately not used here: hue in this panel belongs
            # to the four strategies, and spending it twice is what made the
            # earlier version unreadable.
            ax.text(xi + offset[context], 1.02, S.ROUTE_SHORT[route], ha="center",
                    va="bottom", fontsize=S.FS_NOTE, color=S.INK_2, rotation=0)

    ax.set_xticks(list(x))
    ax.set_xticklabels([S.RISK_LABEL[r] for r in RISKS])
    ax.set_xlabel(r"maximum private risk $p_r^{\max}$")
    ax.set_ylim(0, 1.16)
    ax.set_yticks([0.0, 0.25, 0.50, 0.75, 1.0])
    ax.set_yticklabels(["0", "25", "50", "75", "100"])
    ax.set_ylabel("Trajectories matched (%)")
    S.strip(ax, grid_axis="y")
    ax.spines["left"].set_bounds(0.0, 1.0)
    S.panel(ax, "c", "Which rule each route sits nearest")
    keys = "  ".join(f"{s}: {STRATEGY_NAME[s]}" for s in ("AS", "AU"))
    ax.annotate(
        f"{keys}\n"
        f"CS / CAS: conditional, starting Safe / Unsafe",
        xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -33),
        textcoords="offset points", ha="left", va="top", fontsize=S.FS_NOTE,
        color=S.MUTED, linespacing=1.35, annotation_clip=False)


def panel_d(ax, rows) -> None:
    """Why the previous panel is a lens and never a finding."""
    S = style()

    for context, route in sorted(ROUTE_OF_CONTEXT.items()):
        series = _route_series(rows, context)
        xs = [100 * float(r["unique_classification_rate"]) for r in series]
        ys = [100 * float(r["mean_minimum_mismatch_rate"]) for r in series]
        ax.plot(xs, ys, lw=0.7, color=S.ROUTE_C[route], alpha=0.55, zorder=2)
        for xi, yi, row in zip(xs, ys, series):
            S.dot(ax, xi, yi, color=S.ROUTE_C[route], marker=S.ROUTE_M[route],
                  size=22, zorder=4)
            ax.annotate(S.RISK_LABEL[float(row["max_private_risk"])], (xi, yi),
                        xytext=(4, 3), textcoords="offset points",
                        fontsize=S.FS_NOTE, color=S.INK_2)
        # The name goes on the middle point: the end points already carry a
        # risk label, and the 0.1 end of Gemini sits on the axis floor.
        side = "left" if context == "claude" else "right"
        S.direct_label(ax, xs[1], ys[1], S.ROUTE_SHORT[route],
                       color=S.ROUTE_C[route], ha=side,
                       dx=6 if side == "left" else -6, dy=-7)

    ax.set_xlabel("Trajectories with one nearest rule (%)")
    ax.set_ylabel("Mean distance to that rule (%)")
    ax.set_xlim(0, 105)
    ax.set_ylim(-1, 36)
    S.strip(ax, grid_axis="both")
    S.panel(ax, "d", "The lens is worst where it is least unique")
    ax.annotate(
        "A trajectory with no unique nearest rule, or a large distance to it,\n"
        "is a label the data does not support.",
        xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -33),
        textcoords="offset points", ha="left", va="top", fontsize=S.FS_NOTE,
        color=S.MUTED, linespacing=1.35, annotation_clip=False)


def plot(rows: list[dict[str, object]], bands: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    S = style()

    fig, axes = plt.subplots(2, 2, figsize=(S.TEXT, 4.35))
    fig.subplots_adjust(hspace=0.86, wspace=0.30)
    panel_a(axes[0, 0], rows, bands)
    panel_b(axes[0, 1], rows)
    panel_c(axes[1, 0], rows)
    panel_d(axes[1, 1], rows)
    S.caption(
        fig,
        "Evolutionary curves are the reconstructed model at the two parameter "
        "points its source text declares; route points are pooled decision "
        "frequencies from 10 races and 186 decisions per route and risk. The "
        "two objects are not samples of the same process, and nearest-rule "
        "labels are a distance to a fixed rule, not a recovered policy.",
        y=-0.055,
    )
    written = S.save(fig, "egt_frontier_insights", figdir=FIGDIR, width=S.TEXT,
                     formats=("pdf", "png", "svg"))
    for path in written:
        shutil.copy2(path, OUTPUT / path.name)
    return written


REPORT = """# Frontier and EGT: a boundary analysis

## Which routes this is about

Two routes, named in full: `anthropic/claude-sonnet-5@default` and
`google/gemini-3-flash-preview`. They are the two that were run through the
evolutionary comparison protocol, and they are not "the admitted routes".
`results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv`
admits five: Gemini 3 Flash, Claude Opus 5, GPT-5.4, GPT-5.5 and Claude
Sonnet 5. An earlier version of this report and of the figure caption said
"the two admitted frontier routes", which is wrong about the paper's own gate.

## Readout

- At the reference parameter point the reconstructed model is a step in risk:
  99.2% Unsafe at 0.1, 98.0% at 0.6, then 1.9% at 0.9. Neither route does that.
  Claude Sonnet 5 falls 89.2 to 46.2 to 37.6; Gemini 3 Flash falls 98.9 to 74.2
  to 59.1. Both are ramps, and at 0.9 both sit tens of points above a model that
  has already reached the floor.
- The two routes disagree at every checkpoint, by 9.7, 28.0 and 21.5 points.
  A pooled frontier rate would average over a difference larger than most of the
  effects the paper reports, so none is quoted.
- Nearest-rule matching is least unique exactly where the distance to the
  nearest rule is largest. At risk 0.1 Gemini has a unique nearest rule for
  10% of its trajectories; at 0.9 Claude Sonnet 5 has one for 45% of its own,
  at a mean distance of 29.8%. The labels are a lens on the rates, never a
  finding about a latent policy.

## Uncertainty

The evolutionary curves carry a band and the route points do not, and that is
what the artifacts hold rather than a drawing choice.
`egt_stationary_summary.csv` records a minimum and a maximum over four
independent seeded chains, which `reconstruction_manifest.json` calls a
between-chain range diagnostic and not a confidence interval; it is drawn under
that name. `llm_strategy_summary_primary_t0.csv` carries one rate per route and
risk and no interval of any kind, so no interval is drawn on a route point.
"""


def main() -> None:
    comparison = read_csv(COMPARISON)
    model_rows = read_csv(MODEL_SUMMARY)
    verify_routes(model_rows)
    bands = theory_bands()
    rows = build_derived(comparison, model_rows)
    report(rows, bands)

    with (OUTPUT / "egt_frontier_insights.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    summary = {
        "schema_version": "egt-frontier-insights-v2",
        "evidence_class": "descriptive_frontier_vs_faithful_egt_boundary",
        "source_artifacts": [
            str(COMPARISON.relative_to(ROOT)),
            str(MODEL_SUMMARY.relative_to(ROOT)),
            str(STATIONARY.relative_to(ROOT)),
            str(MANIFEST.relative_to(ROOT)),
            str((OUTPUT / "egttools_pinned_source_validation.json").relative_to(ROOT)),
        ],
        "routes": sorted({str(row["route"]) for row in rows}),
        "routes_are_all_admitted_routes": False,
        "admitted_route_count": 5,
        "admitted_route_source": "results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv",
        "uncertainty": {
            "evolutionary_curves": "between_independent_chain_range_diagnostic_not_confidence_interval",
            "route_points": "no interval is carried by llm_strategy_summary_primary_t0.csv and none is drawn",
        },
        "rows": len(rows),
        "interpretation": [
            "The reference evolutionary regime is a step in risk; both routes are ramps.",
            "The two routes disagree at every risk checkpoint, so no pooled frontier rate is justified.",
            "Nearest-rule labels are least unique where the distance to the rule is largest.",
        ],
    }
    (OUTPUT / "egt_frontier_insights.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    (OUTPUT / "egt_frontier_insights.md").write_text(REPORT, encoding="utf-8")

    written = plot(rows, bands)
    print(json.dumps(
        {"status": "complete", "figures": [str(p.relative_to(ROOT)) for p in written],
         "rows": len(rows)}, indent=2))


if __name__ == "__main__":
    main()
