"""The evolutionary benchmark is a cliff. The audited routes are a slope.

The manuscript repeatedly places observed unsafe play beside the reduced
evolutionary model and reads the gap as miscalibration: the model says risk
should suppress unsafe play, the routes suppress it less, so the routes are
badly tuned.  That reading assumes both objects are functions of risk with
comparable gradients.  Solved at the three configured risks the assumption is
invisible.  Solved on a grid of 201 risks it collapses.  At the reference
selection strength the model does not decline with risk at all: it stays pinned
to the ceiling, falls almost its whole height inside one 0.005-wide step, and
stays on the floor.  A step function has no gradient, so there is nothing for a
route to be calibrated against.

The third claim is the consequence, and it is a claim about the model rather
than about the routes.  Because the only place the model bends is the cliff,
asking which risk would make the model emit an observed rate sends nearly every
answer to the same place.  Twenty-four route-by-risk cells, drawn from three
configured levels that span 0.80 of the risk axis, come back inside a band 0.12
wide, and the three configured levels are not separable in the answers.

Panels
  a  The model at both declared selection strengths, on 201 risks, against the
     nine route profiles at the three configured risks.  Eight of the routes are
     drawn in one grey inside their own band, labelled once: they are the
     context, and which of them is which is panel b's question and panel c's,
     not this one's.  Claude Opus 5 keeps its colour because it is the exception
     the panel argues about, and it is an exception in the wrong place: it has a
     cliff of its own, somewhere in the gap between 0.1 and 0.6 where the
     protocol ran no level, while the model's cliff is above 0.6.  Its saturated
     cells sit on marked boundaries and carry their decision counts, because 0%
     and 100% are measurements with no room beside them.
  b  The inversion as a slope chart: configured risk on the left at full scale,
     the risk the model would need in order to emit the observed rate on the
     right at the same full scale.  The fan closes.
  c  That band magnified, one row per configured level.  The rows overlap, which
     is the sharper version of panel b: inverted through this model, a cell run
     at 0.1 and a cell run at 0.9 are not distinguishable.

What this figure does NOT show.  Nothing here is fitted.  The two selection
strengths are the two points ``scripts/build_theory_tables.py`` already
declares, the population size is its default, and no parameter is estimated from
behaviour, so this is not a test of the model and cannot be read as one.  The
small-mutation limit cannot represent the finite mutation rate of the source
parameter points, and its own metadata says so.  The observed rates are pooled
decision-level frequencies with no interval attached: this figure is about the
shape of the two objects and never about which route differs from which.  Panel
b is an inversion and not an estimate of anybody's perceived risk; it says what
the model would have to be handed, which is a statement about the model.  And a
compressed inverse is not an unresponsive route.  All nine move the right way
with risk, Claude Opus 5 most steeply of all; eight of them merely also land
inside the model's own range, which is what lets the inversion answer at all.
The figure is about how little of the model's own axis that movement could
possibly cover.
"""

from __future__ import annotations

import sys
import textwrap
from dataclasses import replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402

# build_theory_tables owns the parameter points, the population size and the
# configuration loader that produced results/derived/ai_race_theory/.  Importing
# it is what stops this figure and that table from drifting apart, and it is
# also what puts the repository root on sys.path for the ``ai_race`` imports.
import build_theory_tables as T  # noqa: E402
from ai_race.theory.evolution import (  # noqa: E402
    expected_unsafe_frequency,
    small_mutation_stationary,
)
from ai_race.theory.payoffs import (  # noqa: E402
    STRATEGY_ORDER,
    expected_payoff_matrix,
    self_play_unsafe_frequency,
)

RISK_GRID = np.linspace(0.0, 1.0, 201)
THEORY_CSV = (
    Path(T.REPOSITORY_ROOT) / "results" / "derived" / "ai_race_theory"
    / "theory_expected_unsafe.csv"
)

# The configuration fields the payoff construction reads.  One configuration is
# swept across the whole grid below, which is a legitimate stand-in for all
# three only if the three differ in the risk and in nothing else.
MECHANISM_FIELDS = (
    "n_players", "safe_progress", "unsafe_progress", "payoff_safe_safe",
    "payoff_safe_unsafe", "payoff_unsafe_safe", "payoff_unsafe_unsafe",
    "min_rounds", "stop_probability", "max_rounds_safety_cap", "race_prize",
)

# Which strength the inversion runs at.  At beta = 2 the model is a step, its
# inverse is defined only inside a 0.005-wide window, and inverting there would
# manufacture the compression the panel claims to find.  The weak-selection
# point is the one that gives the model its best chance of spreading the cells
# apart, so it is the honest one to invert.
INVERSION_BETA = 0.01

OPUS = "anthropic/claude-opus-5@default"
BETA_C = {2.0: S.INK, 0.01: S.INK_2}
BETA_LS = {2.0: "-", 0.01: (0, (4.0, 1.6))}

MAGNIFIED = (0.545, 0.700)


def theory_curves(config) -> dict[float, np.ndarray]:
    """Expected unsafe frequency over the risk grid, at every declared strength.

    This is exactly the chain ``build_theory_tables.build_stationary_rows`` runs
    at three risks, run at 201 instead: the four-by-four expected payoff matrix
    of the reduced game, then the small-mutation stationary distribution over
    the four monomorphic populations, then the population unsafe frequency.  The
    matrix is rebuilt from scratch at every grid point rather than interpolated
    between endpoints, so no property of the payoff construction is assumed
    here.  The per-strategy self-play unsafe frequency is a function of the
    horizon distribution alone and does not move with risk, so it is computed
    once.
    """
    per_strategy = {name: self_play_unsafe_frequency(config, name) for name in STRATEGY_ORDER}
    matrices = [
        expected_payoff_matrix(replace(config, max_private_risk=float(risk)))
        for risk in RISK_GRID
    ]
    return {
        float(point["beta"]): np.array([
            expected_unsafe_frequency(
                small_mutation_stationary(matrix, Z=T.DEFAULT_POPULATION_SIZE,
                                          beta=float(point["beta"])),
                per_strategy,
            )
            for matrix in matrices
        ])
        for point in T.PARAMETER_POINTS
    }


def verify(configs, curves) -> None:
    """Fail closed rather than draw a curve the repository does not hold.

    Two checks.  The three configuration files are compared field by field, so a
    sweep of one of them cannot quietly stand in for the other two.  Then the
    grid is compared against ``theory_expected_unsafe.csv``, which a different
    script wrote from the same functions at the three configured risks; a grid
    that disagrees there is wrong everywhere else too.
    """
    reference = configs[0]
    for other in configs[1:]:
        for field in MECHANISM_FIELDS:
            if getattr(other, field) != getattr(reference, field):
                raise SystemExit(
                    f"{other.name} differs from {reference.name} in {field}: the "
                    "three configurations are not one mechanism at three risks, "
                    "so no one of them can be swept for all"
                )

    table = pd.read_csv(THEORY_CSV)
    for _, row in table.iterrows():
        risk = float(row["max_private_risk"])
        index = int(round(risk * (len(RISK_GRID) - 1)))
        if abs(RISK_GRID[index] - risk) > 1e-12:
            raise SystemExit(f"configured risk {risk} is not on the grid")
        mine = float(curves[float(row["beta"])][index])
        if abs(mine - float(row["expected_unsafe_frequency"])) > 1e-9:
            raise SystemExit(
                f"the grid disagrees with {THEORY_CSV.name} at risk {risk}, beta "
                f"{row['beta']}: {mine:.12f} against "
                f"{row['expected_unsafe_frequency']:.12f}"
            )


def invert(curve: np.ndarray, value: float) -> float:
    """The risk the model would need to emit ``value``, or NaN if there is none.

    The curve is monotone decreasing but bounded: it never reaches zero and does
    not leave the ceiling until the cliff, so an observed rate outside its own
    range has no preimage at all.  Returning NaN there instead of clamping is
    the point.  A clamp would turn a saturated cell into a confident risk.
    """
    if not curve.min() < value < curve.max():
        return float("nan")
    return float(np.interp(value, curve[::-1], RISK_GRID[::-1]))


def compute() -> dict:
    configs = T.load_game_configs()
    curves = theory_curves(configs[0])
    verify(configs, curves)

    turns = D.baseline_turns()
    obs = turns.groupby(["model_route", "max_private_risk"])["unsafe"].mean().unstack()
    counts = turns.groupby(["model_route", "max_private_risk"])["unsafe"].size().unstack()
    risks = [float(level) for level in obs.columns]
    if sorted(risks) != sorted(float(c.max_private_risk) for c in configs):
        raise SystemExit("the observed risk levels are not the configured ones")

    implied = pd.DataFrame(
        [[invert(curves[INVERSION_BETA], float(obs.loc[route, level]))
          for level in obs.columns]
         for route in S.ROUTE_ORDER],
        index=S.ROUTE_ORDER, columns=obs.columns,
    )
    return {"curves": curves, "obs": obs, "counts": counts, "risks": risks,
            "implied": implied}


def report(res: dict) -> dict:
    """Print every number the figure draws, and return the derived ones."""
    curves, obs, counts = res["curves"], res["obs"], res["counts"]
    risks, implied = res["risks"], res["implied"]

    facts = {}
    for beta in sorted(curves, reverse=True):
        y = curves[beta]
        step = np.abs(np.diff(y))
        k = int(step.argmax())
        facts[beta] = {"cliff_lo": float(RISK_GRID[k]), "cliff_hi": float(RISK_GRID[k + 1]),
                       "drop": 100 * float(step[k]),
                       "floor": 100 * float(y.min()), "ceiling": 100 * float(y.max())}
        print(f"  model beta={beta:g}: steepest fall {facts[beta]['drop']:.2f} points "
              f"between risk {facts[beta]['cliff_lo']:.3f} and "
              f"{facts[beta]['cliff_hi']:.3f}, floor {facts[beta]['floor']:.3f}%, "
              f"ceiling {facts[beta]['ceiling']:.4f}%")
    reference = max(curves)
    cliff = 0.5 * (facts[reference]["cliff_lo"] + facts[reference]["cliff_hi"])
    step_width = facts[reference]["cliff_hi"] - facts[reference]["cliff_lo"]
    model_slope = facts[reference]["drop"] / step_width

    print(f"  routes: {obs.shape[0]} x {obs.shape[1]} cells, "
          f"{int(counts.to_numpy().min())}-{int(counts.to_numpy().max())} decisions each")
    slopes = {}
    for route in S.ROUTE_ORDER:
        row = obs.loc[route].to_numpy()
        slopes[route] = 100 * float((np.abs(np.diff(row)) / np.diff(risks)).max())
        print(f"  {S.ROUTE_SHORT[route]:>9}  {100 * row[0]:5.1f} -> {100 * row[1]:5.1f} "
              f"-> {100 * row[2]:5.1f} %   steepest {slopes[route]:6.1f} points per "
              f"unit risk")
    steepest = max(slopes, key=slopes.get)
    others = [route for route in slopes if route != steepest]
    runner_up = max(others, key=slopes.get)
    print(f"  model beta={reference:g} runs at {model_slope:.0f} points per unit risk: "
          f"{model_slope / slopes[steepest]:.0f}x the steepest route "
          f"({S.ROUTE_SHORT[steepest]}, {slopes[steepest]:.0f}), which is itself "
          f"{slopes[steepest] / slopes[runner_up]:.1f}x the next "
          f"({S.ROUTE_SHORT[runner_up]}, {slopes[runner_up]:.0f}); every route other "
          f"than {S.ROUTE_SHORT[steepest]} stays under "
          f"{max(slopes[route] for route in others):.0f}")

    flat = implied.to_numpy().ravel()
    good = flat[~np.isnan(flat)]
    band = (float(good.min()), float(good.max()))
    width = band[1] - band[0]
    span = max(risks) - min(risks)
    print(f"  inversion at beta={INVERSION_BETA:g}: {good.size} of {flat.size} cells "
          f"invertible, band {band[0]:.4f} to {band[1]:.4f}, width {width:.4f} against "
          f"a configured span of {span:.2f} ({span / width:.1f}x compression); the "
          f"beta={reference:g} cliff at {cliff:.4f} lies inside it")
    for level in obs.columns:
        column = implied[level].dropna().to_numpy()
        print(f"    configured {float(level):.1f} implies {column.min():.4f} to "
              f"{column.max():.4f} over {column.size} routes")
    refused = 0
    for route in S.ROUTE_ORDER:
        bad = [level for level in obs.columns if np.isnan(implied.loc[route, level])]
        if bad:
            refused += len(bad)
            print(f"  {S.ROUTE_SHORT[route]:>9} has no inverse at "
                  f"{[float(level) for level in bad]}: observed "
                  f"{[round(100 * float(obs.loc[route, level]), 1) for level in bad]}% "
                  f"over {[int(counts.loc[route, level]) for level in bad]} decisions")

    # A cell at 0% or 100% is a saturated cell, and the count is what stops a
    # reader from taking it for a rate that merely happened to land there.  The
    # count is also the reason the inversion refuses it, so the panel that draws
    # the boundary and the panel that refuses it quote the same number.
    boundary = {}
    for rate in (1.0, 0.0):
        levels = [level for level in obs.columns if float(obs.loc[OPUS, level]) == rate]
        if levels:
            boundary[rate] = {
                "levels": levels,
                "decisions": int(counts.loc[OPUS, levels].sum()),
            }
            print(f"  {S.ROUTE_SHORT[OPUS]:>9} sits on {100 * rate:.0f}% at "
                  f"{[float(level) for level in levels]}, "
                  f"{boundary[rate]['decisions']} decisions with no room beside them")

    return {"facts": facts, "reference": reference, "cliff": cliff,
            "step_width": step_width, "model_slope": model_slope, "slopes": slopes,
            "band": band, "width": width, "span": span, "refused": refused,
            "boundary": boundary}


def spoken(levels) -> str:
    """``[0.6, 0.9]`` as ``0.6 and 0.9``, for a note a reader reads aloud."""
    names = [f"{float(level):g}" for level in levels]
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + " and " + names[-1]


def panel_a(ax, res, d) -> None:
    curves, obs, risks = res["curves"], res["obs"], res["risks"]
    others = [route for route in S.ROUTE_ORDER if route != OPUS]

    lo = 100 * obs.loc[others].min(axis=0).to_numpy()
    hi = 100 * obs.loc[others].max(axis=0).to_numpy()
    ax.fill_between(risks, lo, hi, color=S.BAND, zorder=1, lw=0)
    # Eight routes in one grey.  Their identities are spent in panels b and c,
    # and eight hues crossing each other here would read as eight arguments
    # where the panel is making one: the band is a ramp and the model is not.
    # The markers stay so the three configured levels remain visible as the only
    # places anything was measured.
    for route in others:
        ax.plot(risks, 100 * obs.loc[route].to_numpy(), lw=0.55, color=S.MUTED,
                marker="o", ms=1.7, mfc=S.MUTED, mec=S.MUTED, zorder=2,
                clip_on=False)
    ax.plot(risks, 100 * obs.loc[OPUS].to_numpy(), lw=1.6, color=S.ROUTE_C[OPUS],
            marker=S.ROUTE_M[OPUS], ms=4.0, mec=S.SURFACE, mew=0.6, zorder=5,
            clip_on=False)
    for beta in sorted(curves, reverse=True):
        ax.plot(RISK_GRID, 100 * curves[beta], lw=1.5, color=BETA_C[beta],
                ls=BETA_LS[beta], zorder=6, clip_on=False)

    S.rate_axis(ax, label="Unsafe play (%)")
    S.risk_axis(ax, label=r"maximum private risk $p_r^{\max}$")
    ax.set_xlim(0.0, 1.15)
    # The floor of the axes sits well below zero on purpose: it is where the two
    # notes about Claude Opus 5 go, and putting them there keeps them off the
    # data instead of on top of the curve they are about.
    ax.set_ylim(-23, 108)
    S.strip(ax, grid_axis="y")
    # The floor of the axes is 23 points below zero so the two Opus notes have
    # somewhere to sit, but there is no scale down there: a spine drawn through
    # it would offer the reader a negative unsafe-play reading that does not
    # exist.  Bound the spine to the range the ticks actually cover.
    ax.spines["left"].set_bounds(0.0, 100.0)
    # Same argument on the other axis: the axes run to 1.15 so the notes and the
    # bracket have room, but risk stops at 1, and a spine drawn past it offers a
    # reading of the risk axis that the mechanism does not define.
    ax.spines["bottom"].set_bounds(0.0, 1.0)
    S.ceiling_rule(ax, 100.0, label="ceiling")
    S.ceiling_rule(ax, 0.0, label="floor")

    # The cliff note is a leader into open sky above the weak curve and right of
    # the drop, not a caption over the panel title: the strip above the axes
    # belongs to the claim, and a note that has to live there is a note in the
    # wrong place.
    ax.plot([d["cliff"], d["cliff"]], [0.0, 100.0], color=S.MUTED, lw=0.6,
            ls=(0, (1.5, 1.8)), zorder=2)
    S.direct_label(ax, 0.715, 97.0, f"model, $\\beta$ = {d['reference']:g}",
                   color=BETA_C[d["reference"]], ha="left", va="top", dx=0, dy=0,
                   weight="bold")
    ax.annotate(f"{d['facts'][d['reference']]['drop']:.0f} points of unsafe play\n"
                f"in one {d['step_width']:.3f} step of risk",
                xy=(d["cliff"], 85.0), xytext=(0.715, 90.5), textcoords="data",
                ha="left", va="top", fontsize=S.FS_NOTE, color=S.INK,
                linespacing=1.35,
                arrowprops=dict(arrowstyle="->", lw=0.6, color=S.INK,
                                shrinkA=3.0, shrinkB=0.0))

    # Opus falls between two configured levels and the protocol ran no level in
    # between, so the only honest statement about where its cliff is, is that it
    # is somewhere inside this gap.
    gap = (risks[0], risks[1])
    ax.annotate("", xy=(gap[0], -4.5), xytext=(gap[1], -4.5),
                arrowprops=dict(arrowstyle="<->", lw=0.6, color=S.ROUTE_C[OPUS],
                                shrinkA=0, shrinkB=0))
    ax.annotate("Opus 5's cliff is in here;\nno level was run inside",
                xy=(0.5 * (gap[0] + gap[1]), -7.5), ha="center", va="top",
                fontsize=S.FS_NOTE, color=S.ROUTE_C[OPUS], linespacing=1.35)

    # The saturated cells are marked with their counts, in the strip beside the
    # gap note, because a reader who sees a line lying on 0 has to be told
    # whether that is a small rate or a rate with nowhere left to fall.
    said = []
    for rate, word in ((1.0, "unsafe"), (0.0, "safe")):
        cell = d["boundary"].get(rate)
        if cell:
            # The noun is said once and the second clause inherits it, because
            # three lines of note fit under this axes and four do not.
            noun = " decisions" if not said else ""
            said.append(f"{word} in all {cell['decisions']}{noun} at "
                        f"{spoken(cell['levels'])}")
    if said:
        ax.annotate("\n".join(textwrap.wrap("Opus 5 is " + ", ".join(said), 28)),
                    xy=(1.15, -7.5), ha="right", va="top", fontsize=S.FS_NOTE,
                    color=S.ROUTE_C[OPUS], linespacing=1.35)

    # The weak curve is named on the tail it is still descending, which is the
    # only stretch where the two strengths are far enough apart for a label to
    # belong to one of them without ambiguity.  The reference strength is named
    # above, on the step that is its whole shape.
    weak = min(curves)
    i = int(round(0.705 * (len(RISK_GRID) - 1)))
    S.direct_label(ax, RISK_GRID[i], 100 * curves[weak][i],
                   f"model, $\\beta$ = {weak:g}", color=BETA_C[weak],
                   ha="left", va="bottom", dx=4, dy=3.5, weight="bold")
    S.direct_label(ax, risks[0], 100.0, "Claude Opus 5", color=S.ROUTE_C[OPUS],
                   dx=4, dy=4.5, weight="bold")
    ends = 100 * obs.loc[others, obs.columns[-1]].to_numpy()
    ax.plot([0.945, 0.945], [ends.min(), ends.max()], color=S.MUTED, lw=0.8,
            clip_on=False, zorder=4)
    S.direct_label(ax, 0.965, ends.mean(), "the other\neight routes", color=S.MUTED)
    # Naming the regime on the page, not only in the docstring. Every curve here
    # is the small-mutation limit, which the theory metadata states cannot
    # represent the finite mutation rate of the parameter points it is drawn at,
    # and a reader who sees only "the model" would not know which model.
    S.panel(ax, "a",
            "the small-mutation limit steps off a cliff; eight of the nine "
            "routes walk a ramp")


def panel_b(ax, res, d) -> None:
    obs, risks, implied = res["obs"], res["risks"], res["implied"]
    band = d["band"]

    ax.add_patch(plt.Rectangle((1.0, band[0]), 0.055, d["width"], facecolor=S.BAND,
                               edgecolor="none", zorder=1))
    for route in S.ROUTE_ORDER:
        for level, true in zip(obs.columns, risks):
            value = implied.loc[route, level]
            if np.isnan(value):
                # A refusal to invert, not a missing measurement: the observed
                # rate sits on a boundary the model never reaches.
                ax.plot([0.0, 0.42], [true, true], lw=0.7, color=S.ROUTE_C[route],
                        ls=(0, (1.4, 1.4)), zorder=3)
                ax.plot([0.45], [true], marker="x", ms=3.0, mew=0.9,
                        color=S.ROUTE_C[route], zorder=4)
                continue
            ax.plot([0.0, 1.0], [true, value], lw=0.7, color=S.ROUTE_C[route],
                    zorder=3)
            ax.plot([1.0], [value], marker=S.ROUTE_M[route], ms=2.6,
                    color=S.ROUTE_C[route], mec=S.SURFACE, mew=0.35, zorder=4,
                    clip_on=False)
    for true in risks:
        ax.plot([0.0], [true], marker="_", ms=7, mew=1.4, color=S.INK, zorder=5)
    S.direct_label(ax, 0.47, max(risks), "Claude Opus 5: no inverse",
                   color=S.ROUTE_C[OPUS], dx=3)

    ax.plot([-0.055, -0.055], [min(risks), max(risks)], color=S.INK, lw=0.9,
            solid_capstyle="butt", zorder=4)
    for cap in (min(risks), max(risks)):
        ax.plot([-0.075, -0.035], [cap, cap], color=S.INK, lw=0.9, zorder=4)
    S.direct_label(ax, 1.062, 0.5 * (band[0] + band[1]),
                   f"{band[0]:.2f}\nto {band[1]:.2f}", color=S.INK_2, dx=0)

    ax.set_xlim(-0.10, 1.06)
    ax.set_ylim(0.0, 1.0)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.5", "0.75", "1"])
    ax.set_ylabel(r"risk $p_r^{\max}$", labelpad=1)
    ax.set_xticks([0.0, 1.0])
    ax.set_xticklabels([f"configured\n{min(risks):.1f} to {max(risks):.1f}",
                        "implied by\nthe model"])
    ax.tick_params(axis="x", length=0, pad=2)
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", zorder=0)
    S.panel(ax, "b",
            f"inverted, {d['span']:.2f} of configured risk comes back as "
            f"{d['width']:.2f}")


def dodge(values, *, separation, step=0.22, levels=5):
    """Offsets that stop two nearly equal points from hiding each other.

    Within a row the vertical position carries no information, so a point may be
    nudged off the line.  The alternative is two routes drawn on top of each
    other, which reads as one route and loses a measurement.
    """
    offsets = np.zeros(len(values))
    placed: dict[int, list[float]] = {}
    for index in np.argsort(values):
        for level in range(levels):
            slot = (level + 1) // 2 * (1 if level % 2 else -1)
            if all(abs(values[index] - x) >= separation for x in placed.get(slot, [])):
                offsets[index] = slot * step
                placed.setdefault(slot, []).append(values[index])
                break
    return offsets


def panel_c(ax, res, d) -> None:
    obs, risks, implied = res["obs"], res["risks"], res["implied"]

    ax.axvspan(d["band"][0], d["band"][1], color=S.BAND, zorder=0, lw=0)
    ax.axvline(d["cliff"], color=S.MUTED, lw=0.6, ls=(0, (1.5, 1.8)), zorder=1)
    separation = 0.055 * (MAGNIFIED[1] - MAGNIFIED[0])
    for row, level in enumerate(obs.columns):
        column = implied[level].dropna()
        ax.plot([column.min(), column.max()], [row, row], lw=0.8,
                color=S.HAIRLINE, zorder=2, solid_capstyle="round")
        offsets = dodge(column.to_numpy(), separation=separation)
        for (route, value), offset in zip(column.items(), offsets):
            if offset:
                # The offset carries no information, so it has to cost none: a
                # leader back to the row is what keeps a nudged marker readable
                # as a member of its row rather than of the gap beside it.
                ax.plot([value, value], [row, row + offset], lw=0.4,
                        color=S.ROUTE_C[route], zorder=3, alpha=0.55)
            ax.plot([value], [row + offset], marker=S.ROUTE_M[route], ms=3.0,
                    color=S.ROUTE_C[route], mec=S.SURFACE, mew=0.35, zorder=4)
    ax.annotate(f"the $\\beta$ = {d['reference']:g} cliff", xy=(d["cliff"], -0.80),
                xytext=(2.5, 0), textcoords="offset points", ha="left",
                va="center", fontsize=S.FS_NOTE, color=S.MUTED)

    ax.set_xlim(*MAGNIFIED)
    ax.set_ylim(2.65, -1.05)
    ax.set_yticks(range(len(risks)))
    ax.set_yticklabels([f"{risk:g}" for risk in risks])
    ax.set_ylabel("configured", labelpad=1)
    ax.set_xticks([0.55, 0.60, 0.65, 0.70])
    ax.set_xlabel(r"implied $p_r^{\max}$", labelpad=1)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="x", zorder=0)
    S.panel(ax, "c", "the three levels are not separable in the answers")


def draw(res: dict, d: dict) -> None:
    fig = plt.figure(figsize=(S.TEXT, 3.25))
    gs = fig.add_gridspec(2, 2, width_ratios=[1.58, 1.0],
                          height_ratios=[1.0, 0.72], wspace=0.42, hspace=0.82)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    panel_a(ax_a, res, d)
    panel_b(ax_b, res, d)
    panel_c(ax_c, res, d)

    cells = res["implied"].size
    ax_c.annotate(
        f"{cells - d['refused']} of {cells} cells invert. The other "
        f"{d['refused']} are Claude Opus 5 on 0% or 100%,\nrates no risk in the "
        "model produces, so they are refused rather than placed.",
        xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -25),
        textcoords="offset points", ha="left", va="top", fontsize=S.FS_NOTE,
        color=S.MUTED, linespacing=1.35)

    S.save(fig, "theory_versus_behaviour", width=S.TEXT)


def main() -> None:
    res = compute()
    draw(res, report(res))


if __name__ == "__main__":
    main()
