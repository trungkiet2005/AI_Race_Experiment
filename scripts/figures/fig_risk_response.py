"""How the screened routes trade speed against safety as the danger rises.

The claim.  Every route in the behavioural panel plays Unsafe less as the
stated maximum private risk rises, and four of the five do it by nearly the
same amount: the drop from risk 0.1 to risk 0.9 is 38.2, 38.7, 40.3 and 57.0
points, a spread of nineteen points across four checkpoints from three
vendors.  The fifth does not trade off at all.  Claude Opus 5 plays Unsafe on
every decision at risk 0.1 and on none at all above it, which is a switch
rather than a slope and is reported on its own rather than inside an average.
Averaging the five gives 54.8 points, a number one route dominates, and that
average is not drawn here.

Why the rates are drawn as a table of colour.  Five routes over three risk
levels is five crossing lines and no argument; the numbers themselves are what
a reader wants, and printing them keeps the panel exact while the colour
carries the shape.  The same rates are drawn as lines in the theory comparison,
where the shape is the point and the model curve is what they are compared
against, so the two figures are not two drawings of one picture.

What the right-hand column is.  The drop is the pooled decision-level Unsafe
rate at risk 0.1 minus the same rate at risk 0.9, in percentage points.  It is
a difference of two published cell rates and carries no interval; the paired
contrasts with intervals in this paper are the ones the scripted-rival campaign
supports, where the horizon draw can be differenced out.

The bottom row is not a sixth route.  It is one of the five administered the
same frozen protocol a second time, and it is there because a reader's first
question about ten races in a cell is whether a cell rate is a property of the
route or of a run.  Setting the repeat beside its own first administration
answers that in the same units as everything else on the panel: the two differ
by at most 1.1 points at any risk level, and their drops are 38.7 and 40.3.

What this figure does NOT show.  Five commercial endpoints are not a sample
from a population of models, and nothing here estimates what any other
checkpoint would do.  The rates are pooled over decisions rather than over
races, which is the number the run's own summary publishes and therefore the
number an independent reader reproduces; it is a central value and not an
interval.  Three risk levels were run and nothing was measured between them, so
a route's behaviour between 0.1 and 0.6 is unobserved, which matters most for
Claude Opus 5, whose entire change happens somewhere in that gap.  The four
routes that did not enter the behavioural panel are not drawn here; their rates
are in the supplementary material beside these five.  And every rate on this
page is self-play, both companies in a race being the same endpoint, which the
scripted-rival campaign shows is not a measurement of how a route treats risk.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402

# The route that was administered the frozen protocol a second time.  It is the
# only route with an independent repeat, so it is the only one the bottom row
# can carry.
REPEAT = "google/gemini-3-flash-preview"
# The route whose policy is a step rather than a slope.  Named here rather than
# found by a rule, and then checked against the data below, so a figure can
# never quietly promote a different route into the exception's place.
STEP = "anthropic/claude-opus-5@default"
# The bottom row of the panel.  It is named for what it measures, an independent
# repeat of the frozen protocol on one route, and never for when it happened: a
# row labelled "rerun" tells a reader the order things were done in, which is a
# fact about the project and not about the result.
RERUN_ROW = "repeat"


def build() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """Pooled Unsafe rates and the drop, for the screened routes and the repeat."""
    turns = D.baseline_turns()
    rates = turns.groupby(["model_route", "max_private_risk"])["unsafe"].mean().unstack()
    counts = turns.groupby(["model_route", "max_private_risk"])["unsafe"].size().unstack()
    if list(rates.columns) != list(S.RISKS):
        raise SystemExit(f"risk levels on disk are {list(rates.columns)}, not {list(S.RISKS)}")
    if len(rates) != len(S.ROUTE_ORDER):
        raise SystemExit(f"{len(rates)} routes in the baseline, not {len(S.ROUTE_ORDER)}")

    table = pd.DataFrame(index=S.ADMITTED)
    for risk in S.RISKS:
        table[f"rate_{risk}"] = rates.loc[S.ADMITTED, risk] * 100.0
    table["drop_pp"] = table[f"rate_{S.RISKS[0]}"] - table[f"rate_{S.RISKS[-1]}"]

    # The repeat is loaded separately so that a second administration can never
    # be pooled into a route's own cell rate by accident.
    both = D.baseline_turns(include_replication=True)
    repeat = both[(both["model_route"] == REPEAT) & (both["task_version"] == "replication")]
    if repeat.empty:
        raise SystemExit(f"no second administration of {REPEAT} on disk")
    rerun = repeat.groupby("max_private_risk")["unsafe"].mean() * 100.0
    row = {f"rate_{r}": float(rerun[r]) for r in S.RISKS}
    row["drop_pp"] = row[f"rate_{S.RISKS[0]}"] - row[f"rate_{S.RISKS[-1]}"]
    table.loc[RERUN_ROW] = row
    return table, rerun, counts.loc[S.ADMITTED]


def check(table: pd.DataFrame) -> None:
    """Refuse to draw a claim the numbers no longer carry.

    Three things this panel asserts in words.  That every screened route falls
    with risk; that exactly one of them is a step, meaning saturated at both
    ends; and that the route named as that exception is the one the data picks
    out.  Each is checked rather than trusted, because each is a sentence a
    reader will take from the drawing without reading the caption.
    """
    routes = table.drop(index=RERUN_ROW)
    falling = routes[f"rate_{S.RISKS[0]}"] > routes[f"rate_{S.RISKS[-1]}"]
    if not falling.all():
        raise SystemExit(f"these routes do not fall with risk: {sorted(routes.index[~falling])}")
    saturated = (routes[f"rate_{S.RISKS[0]}"] >= 100.0 - 1e-9) & (
        routes[f"rate_{S.RISKS[-1]}"] <= 1e-9)
    found = sorted(routes.index[saturated])
    if found != [STEP]:
        raise SystemExit(
            f"the step policy in the data is {found}, but this figure names {STEP}; "
            "the row that is set apart is naming the wrong route"
        )

    # The derived table is what the manuscript's prose quotes, so a figure drawn
    # from the decision records has to agree with it or one of the two is wrong.
    published = D.audit_versus_behaviour().set_index("route").reindex(routes.index)
    worst = float((published["risk_response_pp"] - routes["drop_pp"]).abs().max())
    print(f"  cross-check against the derived campaign table, drop: max |diff| {worst:.2e}")
    if worst > 1e-9:
        raise SystemExit(
            "THE RECOMPUTED DROPS DISAGREE WITH results/frontier/baseline_campaign_v6/"
            "derived/audit_versus_behaviour.csv. The figure is not the thing to fix; "
            "one of the two computations is wrong and the paper quotes the other."
        )


def draw_tiles(ax, table: pd.DataFrame, graded: list[str], worst: float) -> None:
    """Four graded routes, the step route below a rule, the repeat below another."""
    order = graded + [STEP, RERUN_ROW]
    arr = table.loc[order, [f"rate_{r}" for r in S.RISKS]].to_numpy(dtype=float)
    names = [S.ROUTE_SHORT[r] for r in graded] + [S.ROUTE_SHORT[STEP], "repeat"]
    colours = [S.ROUTE_C[r] for r in graded] + [S.ROUTE_C[STEP], S.ROUTE_C[REPEAT]]
    S.heat_tiles(ax, arr, names, [S.RISK_LABEL[r] for r in S.RISKS],
                 cmap="viridis", vmin=0.0, vmax=100.0, fmt="{:.0f}",
                 row_colors=colours)
    # Room to the left for the route glyphs and to the right for the drop, both
    # inside the same axes so neither can drift off the panel.
    left, right = -1.30, 3.95
    ax.set_xlim(left, right)

    for i, route in enumerate(order):
        key = REPEAT if route == RERUN_ROW else route
        S.dot(ax, -0.92, i, color=S.ROUTE_C[key], marker=S.ROUTE_M[key], size=14,
              filled=route != RERUN_ROW)
        ax.text(right, i, f"{table.loc[route, 'drop_pp']:.0f}", ha="right",
                va="center", fontsize=S.FS_NOTE, color=S.INK_2)

    # A cell on 0 or on 100 has no room beside it, so it is ringed rather than
    # left to be read as a rate that merely happened to land there.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] in (0.0, 100.0):
                S.highlight(ax, i, j, color=S.tile_ink(arr[i, j]), lw=0.9,
                            inset=0.40)

    # Two rules, two different arguments.  The first separates a policy that is
    # a step from four that are slopes, which is why the average of the five is
    # not reported.  The second separates a route from itself: the bottom row is
    # the same endpoint under the same protocol on a second administration, and
    # a reader who took it for a sixth checkpoint would miscount the panel.
    for split in (len(graded) - 0.5, len(graded) + 0.5):
        ax.plot([left, right], [split, split], color=S.INK_2, lw=0.7, zorder=6,
                clip_on=False)

    # The bracket on the graded drops.  Its width is the finding, so it is drawn
    # as a length rather than left to be worked out from four numerals.
    lo, hi = table.loc[graded, "drop_pp"].min(), table.loc[graded, "drop_pp"].max()
    ax.plot([right + 0.14] * 2, [-0.34, len(graded) - 0.66], color=S.INK_2,
            lw=0.8, clip_on=False, zorder=6)
    ax.annotate(f"all four\nwithin\n{hi - lo:.0f} points",
                xy=(right + 0.20, 0.5 * (len(graded) - 1)), xytext=(2, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=S.FS_NOTE, color=S.INK_2, linespacing=1.25,
                annotation_clip=False)
    ax.annotate("a switch,\nnot a slope", xy=(right + 0.20, len(graded)),
                xytext=(2, 0), textcoords="offset points", ha="left", va="center",
                fontsize=S.FS_NOTE, color=S.ROUTE_C[STEP], linespacing=1.25,
                annotation_clip=False)
    # "never 1.1 points off" says the two administrations are never 1.1 apart,
    # which is the opposite of the finding: 1.1 is the largest gap they reach.
    # Write the bound as a bound.
    ax.annotate(f"the same route on an\nindependent repeat:\n"
                f"at most {worst:.1f} points apart",
                xy=(right + 0.20, len(graded) + 1), xytext=(2, 0),
                textcoords="offset points", ha="left", va="center",
                fontsize=S.FS_NOTE, color=S.ROUTE_C[REPEAT], linespacing=1.25,
                annotation_clip=False)

    # The column header sits above the tiles, where a header belongs, so that
    # the foot of the panel carries only the risk axis and cannot collide.
    # "points" rather than "pp": the panel's own title counts the band in
    # points, and a figure that uses two names for one unit spends the reader's
    # attention on the unit instead of on the number.
    ax.annotate("drop (points)", xy=(right, -0.5), xytext=(0, 2),
                textcoords="offset points", ha="right", va="bottom",
                fontsize=S.FS_NOTE, color=S.INK_2, annotation_clip=False)
    # The outline is a channel the panel uses and nothing else explains.  A cell
    # at 0 or at 100 has no room beside it, so the contrast it enters is
    # truncated, and a reader who does not know that reads a bounded measurement
    # as an ordinary one.
    ax.annotate("outlined: no room above or below", xy=(left, -0.5),
                xytext=(0, 2), textcoords="offset points", ha="left",
                va="bottom", fontsize=S.FS_NOTE, color=S.MUTED,
                annotation_clip=False)
    ax.annotate(r"maximum private risk $p_r^{\max}$",
                xy=(1.0, arr.shape[0] - 0.5), xytext=(0, -12),
                textcoords="offset points", ha="center", va="top",
                fontsize=S.FS_TICK, color=S.INK_2, annotation_clip=False)


def main() -> None:
    table, rerun, counts = build()
    check(table)

    routes = table.drop(index=RERUN_ROW)
    graded = [r for r in routes.index if r != STEP]
    graded = list(routes.loc[graded, "drop_pp"].sort_values(ascending=False).index)
    lo, hi = routes.loc[graded, "drop_pp"].min(), routes.loc[graded, "drop_pp"].max()
    worst = float(max(abs(table.loc[RERUN_ROW, f"rate_{r}"] - routes.loc[REPEAT, f"rate_{r}"])
                      for r in S.RISKS))

    print(f"  five screened routes, {int(counts.to_numpy().sum())} decisions, "
          f"{int(counts.to_numpy().min())} per risk cell")
    for route in graded + [STEP]:
        row = routes.loc[route]
        print(f"    {S.ROUTE_SHORT[route]:>9}  "
              + "  ".join(f"{row[f'rate_{r}']:5.1f}" for r in S.RISKS)
              + f"   drop {row['drop_pp']:5.1f} pp"
              + ("   step policy" if route == STEP else ""))
    print(f"  the four graded routes drop {lo:.1f} to {hi:.1f} pp, "
          f"a band {hi - lo:.1f} points wide; {S.ROUTE_SHORT[STEP]} is a "
          f"{routes.loc[STEP, 'drop_pp']:.1f}-point step and is reported apart")
    print(f"  the five-route mean is {routes['drop_pp'].mean():.1f} pp and is NOT "
          f"drawn: one route contributes the whole scale to it")
    print(f"  at the highest risk the four graded routes still play Unsafe "
          f"{routes.loc[graded, f'rate_{S.RISKS[-1]}'].min():.1f} to "
          f"{routes.loc[graded, f'rate_{S.RISKS[-1]}'].max():.1f}% of the time")
    print(f"    {S.ROUTE_SHORT[REPEAT] + ' repeat':>9}  "
          + "  ".join(f"{table.loc[RERUN_ROW, f'rate_{r}']:5.1f}" for r in S.RISKS)
          + f"   drop {table.loc[RERUN_ROW, 'drop_pp']:5.1f} pp")
    print(f"  the two administrations differ by at most {worst:.2f} points at any "
          f"risk level")

    fig = plt.figure(figsize=(S.COL, 2.05))
    ax = fig.add_subplot(1, 1, 1)
    draw_tiles(ax, table, graded, worst)
    # One panel means no panel letter: the claim is the whole title, and it is
    # set over two lines because at column width one line of it does not fit.
    # The band width is read off the drops rather than typed, because a typed
    # number in a title is the one number in a figure nothing recomputes.
    ax.set_title("every screened route plays Unsafe less as the danger rises;\n"
                 f"four of the five inside one {hi - lo:.0f}-point band, "
                 "the fifth a switch",
                 loc="left", pad=17, x=0.0, fontsize=S.FS_CLAIM, color=S.INK_2,
                 linespacing=1.4)
    S.save(fig, "risk_response", width=S.COL)


if __name__ == "__main__":
    main()
