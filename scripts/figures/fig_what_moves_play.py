"""Which features of the situation actually change how these routes play.

Three manipulations were run against the same game.  Only one of them moves the
play, and the other two are worth drawing precisely because they do not: a
design factor that produces nothing is evidence about the construct, and a
design factor that produces something for the wrong reason is worse than one
that produces nothing at all.

The claim.  Group size moves play and moves it into the ceiling, so the surface
is a saturated corner rather than a plane.  Representation moves it too, and both
main effects are present on both routes; what the design does not resolve is
whether the narrative frame and the opaque code interact.  Both interaction
intervals span zero, but one of them reaches +14.3 points, so the honest reading
is an imprecise estimate consistent with no interaction, never a demonstration
that there is none.  The seat is supposed to be inert, its two prompts being
symmetric word for word, and it is not quite inert at the opening.

Every panel is self-play.  One route holds every seat of a race, so nothing here
is a mixed population, and the three factors are manipulations on a route
playing copies of itself.

Panels
  a  The matched group-size grid, four group sizes by three risk levels, one
     route.  Colour is the unsafe rate, the printed second line is the number of
     safe decisions the cell actually contained.  Five cells contain none, and
     those are hatched, because a cell with no safe decision in it has no room
     left to move and is not a measurement of how much larger the effect could
     have been.
  b  The crossed representation design, two factors and two routes, drawn on a
     common thirty-point vertical span so a slope in the upper panel is the same
     slope as in the lower one.  Near-parallel lines are the finding, and
     near-parallel is not flat: both main effects are there on both routes.
     Mind the sign convention, because the two contrasts are signed in opposite
     physical directions.  ``frame`` is race story minus neutral, so its -3.4
     and -10.8 mean the race story LOWERS unsafe play by 3.4 and 10.8 points.
     ``code`` is Safe = P minus Safe = Q, so its -8.3 and -9.5 mean the mapping
     under which Q means Safe RAISES unsafe play by 8.3 and 9.5 points.  That
     is the direction the caption states, and the figure now states it in words
     rather than leaving a reader to infer a convention from a minus sign.  The
     code effect is also re-expressed as what it algebraically is: twice the
     route's excess tendency to emit one particular letter.  That re-expression
     is an identity of the counterbalancing and holds for any data at all, so it
     explains nothing on its own, and the letter usage behind it is not a fixed
     habit: Gemini 3 Flash emits Q on 77% of its turns when Q means unsafe and
     on 11% when Q means safe.  The code effect is a representation effect; the
     algebra only says which surface carries it.
  c  The two seats, whose prompts are symmetric, at the opening move of every
     race.  Four routes open unsafe from both seats in all thirty races and sit
     on the ceiling, where a seat effect could not appear even if it existed, so
     the pooled gap is carried by the routes with room.  Both boundaries are
     marked: the four routes at the ceiling and the one seat that opened unsafe
     in none of its thirty races.  The pooled number
     divides by all nine routes, because all nine were run under the same design
     with thirty races each.  A route whose measured gap came out at zero is a
     result and not a defect, so none is dropped: the script prints what the
     other denominators would have given, and every one of them is larger.

What this figure does NOT show.  Panel a is one route, so nothing here separates
a property of group size from a property of that route in groups; the twelve
cells are matched to each other and to nothing outside the grid.  Nor is the
column a manipulation of group size alone: the N-player payoffs divide one
market benefit among N, so a company that plays safe while everyone else does
earns 1.0 at N=2 and -0.2 at N=5, and a route that only reads the sign of that
number would draw this same grid.  Panel b tests two factors on two routes at one
prompt version, and an interaction that spans zero is inconclusive rather than
absent, the more so on the route whose interval reaches +14 points; nothing here
is evidence that representation is inert, and the two main effects say the
opposite.  Panel c is a validity check: it establishes that the
seat is not perfectly inert, and it does not licence reading the direction as a
finding about first-mover psychology.  Its pooled gap is an average over these
nine routes and not an estimate of the seat effect a route would show if it had
room: four routes are at the ceiling in both seats and can only pull the average
toward zero, so the number is a floor on the asymmetry rather than a measure of
its size.  None of the three panels is a causal
estimate outside the manipulated factor, and no interval here is corrected for
asking three questions on one figure.
"""

from __future__ import annotations

import re
import sys
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402

N_BOOT = 20000
SEED = 20260910

FRAME = {"technology_race": "race story", "abstract_game": "neutral"}
FRAME_LS = {"technology_race": "-", "abstract_game": (0, (2.6, 1.6))}
CODE = ["P_SAFE_Q_UNSAFE", "Q_SAFE_P_UNSAFE"]
CODE_LABEL = {"P_SAFE_Q_UNSAFE": "Safe = P", "Q_SAFE_P_UNSAFE": "Safe = Q"}
MAP_ROUTES = ["anthropic/claude-sonnet-5@default", "google/gemini-3-flash-preview"]

# One vertical span for both representation panels.  The two routes sit thirty
# points apart, and a panel autoscaled to its own cells would draw the smaller
# route's slopes at three times the gradient of the larger one's, which is the
# one thing an interaction plot must not do.
B_SPAN = 30.0

# Half the horizontal separation between the two seats of one route.
SEAT_DX = 0.115

# A value label that will not fit beside its own marker goes out sideways, and
# the two sides do not need the same room.  Each representation panel is 36 pt
# of plot for a thirty point span, so a 6.2 pt note hung below a marker occupies
# 11.6 pt, which is 9.6 of those units, and anything closer to the floor than
# that lands on the bottom spine and the tick row.  Above the axes there is only
# open gutter, so a note there may overhang it.
BELOW_PAD = 9.6
ABOVE_PAD = 5.0


def wrap(text, inches, *, fs=None):
    """Wrap a note to the width of the panel it hangs under.

    An unwrapped note is drawn as one long line, and ``bbox_inches='tight'``
    then measures the figure as that line's width, so ``save`` rescales the
    whole figure to fit a sentence.
    """
    chars = max(20, int(inches * 72.0 / ((fs or S.FS_NOTE) * 0.50)))
    return "\n".join(textwrap.wrap(" ".join(text.split()), chars))


def ci(values):
    """Percentile interval over independent units, seeded once."""
    return D.cluster_bootstrap_ci(values, n_boot=N_BOOT, seed=SEED)


def seat_gap_ci(paired: pd.DataFrame):
    """Race-clustered interval that resamples races within their own route.

    The design fixes thirty races per route, so an unstratified resample would
    let a draw weight the routes unequally and would carry variance the design
    does not have.
    """
    rng = np.random.default_rng(SEED)
    blocks = [g["d"].to_numpy() for _, g in paired.groupby("model_route")]
    draws = np.concatenate(
        [b[rng.integers(0, b.size, size=(N_BOOT, b.size))] for b in blocks], axis=1
    ).mean(axis=1)
    return (float(paired["d"].mean()),
            float(np.percentile(draws, 2.5)),
            float(np.percentile(draws, 97.5)))


# The N-player payoffs split one market benefit among N, so the reward for safe
# play is not held constant down the row of the grid.  Read it out of the prompts
# the route actually saw rather than quoting it from the protocol, because the
# number is the panel's own caveat and a caveat typed by hand is a caveat that
# can go stale.
ALL_SAFE = re.compile(
    r"If all (\d+) companies choose SAFE: each company earns (-?[0-9.]+)")


def all_safe_payoff(turns):
    found = {}
    for n, block in turns.groupby("n_players"):
        seen = {m.group(2).rstrip(".")
                for m in (ALL_SAFE.search(p) for p in block["prompt"]) if m}
        if len(seen) != 1:
            raise SystemExit(f"N={n} shows {len(seen)} all-safe payoffs: {seen}")
        found[int(n)] = float(seen.pop())
    return found


def group_size_grid():
    turns = D.matched_cells()
    route = turns["model"].unique()
    if len(route) != 1:
        raise SystemExit(f"panel a expects one route, found {list(route)}")
    cells = turns.groupby(["cell_risk", "n_players"]).agg(
        decisions=("unsafe", "size"), unsafe=("unsafe", "sum"),
        races=("game_id", "nunique")).reset_index()
    cells["safe"] = cells["decisions"] - cells["unsafe"]
    cells["rate"] = 100 * cells["unsafe"] / cells["decisions"]
    return str(route[0]), cells, all_safe_payoff(turns)


def representation():
    turns = D.mapping_turns()
    # The horizon draw is shared across the four cells of a repetition, so a
    # repetition is a genuine matched quadruple and the contrast can be taken
    # inside it rather than between two independent samples.
    shared = turns.groupby(["model_route", "max_private_risk", "repetition"])[
        "horizon_draws_sha256"].nunique()
    if (shared != 1).any():
        raise SystemExit("repetitions do not share a horizon draw; pairing is unsound")

    per_pair = turns.groupby(
        ["model_route", "max_private_risk", "repetition", "context", "mapping"]
    )["unsafe"].mean().unstack(["context", "mapping"])
    # The cells the panel draws have to be the same estimand as the contrasts it
    # annotates.  Pooling every decision of a cell weights a long race more than
    # a short one, and the contrasts are means over repetitions, so the two
    # disagree by up to 1.9 pp and a reader who does the arithmetic on the four
    # drawn numbers does not get the number printed above them.
    pooled = per_pair.groupby("model_route").mean()
    letter_q = turns.assign(is_q=(turns["action_code"] == "Q").astype(float)).groupby(
        ["model_route", "max_private_risk", "repetition"])["is_q"].mean()

    rows = []
    for route in MAP_ROUTES:
        block = per_pair.loc[route]
        tp = block[("technology_race", CODE[0])]
        tq = block[("technology_race", CODE[1])]
        ap = block[("abstract_game", CODE[0])]
        aq = block[("abstract_game", CODE[1])]
        contrasts = {
            "interaction": (tp - tq) - (ap - aq),
            "code": 0.5 * ((tp + ap) - (tq + aq)),
            "frame": 0.5 * ((tp + tq) - (ap + aq)),
        }
        record = {"model_route": route, "pairs": len(block)}
        for name, series in contrasts.items():
            m, lo, hi = ci(series.to_numpy())
            record.update({name: m, f"{name}_lo": lo, f"{name}_hi": hi})
        # The counterbalancing makes the code contrast an identity, not an
        # effect: under one mapping Unsafe is the letter Q and under the other it
        # is the letter P, so their difference is 2 P(Q) - 1 whatever the route
        # is doing about the game.  Checked here so the figure can say so.
        record["letter_q"] = float(letter_q.loc[route].mean())
        record["code_identity"] = 2 * (record["letter_q"] - 0.5)
        # And the identity holds whatever the letters are doing, so it is not
        # evidence of a letter habit.  The two mapping-conditional shares are,
        # and on both routes they are far apart: the letter follows the meaning.
        block_q = turns[turns["model_route"] == route]
        for code in CODE:
            side = block_q[block_q["mapping"] == code]
            record[f"q_{code}"] = float((side["action_code"] == "Q").mean())
        record["cells"] = pooled.loc[route]
        rows.append(record)
    return pd.DataFrame(rows).set_index("model_route")


def seat():
    turns = D.baseline_turns()
    opening = turns[turns["round"] == 1]
    wide = opening.pivot_table(index=["model_route", "game_id"], columns="player",
                               values="unsafe").reset_index()
    wide["d"] = wide["Company_2"] - wide["Company_1"]
    per_route = wide.groupby("model_route").agg(
        seat1=("Company_1", "mean"), seat2=("Company_2", "mean"),
        d=("d", "mean"), races=("game_id", "size"))

    every = turns.pivot_table(index=["model_route", "game_id"], columns="player",
                              values="unsafe").reset_index()
    every["d"] = every["Company_2"] - every["Company_1"]
    return wide, per_route, seat_gap_ci(wide), seat_gap_ci(every)


def draw_group_size(ax, fig, cells, route, payoff):
    risks = sorted(cells["cell_risk"].unique())
    sizes = sorted(cells["n_players"].unique())
    rate = cells.pivot(index="cell_risk", columns="n_players", values="rate").loc[risks, sizes]
    safe = cells.pivot(index="cell_risk", columns="n_players", values="safe").loc[risks, sizes]
    arr = rate.to_numpy(dtype=float)
    per_size = {int(n): int(cells[cells["n_players"] == n]["decisions"].iloc[0]) for n in sizes}

    im = S.heat_tiles(
        ax, arr,
        [f"{r:g}" for r in risks],
        [f"{n:g}\n({per_size[int(n)]})" for n in sizes],
        cmap="viridis", vmin=50.0, vmax=100.0, fmt=None,
    )
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            pinned = safe.iat[i, j] == 0
            tone = S.SURFACE if (arr[i, j] - 50.0) / 50.0 <= 0.62 else S.INK
            ax.text(j, i - 0.15, f"{arr[i, j]:.0f}%", ha="center", va="center",
                    fontsize=S.FS_NOTE + 1.0, color=tone, fontweight="bold", zorder=4)
            ax.text(j, i + 0.22, f"{int(safe.iat[i, j])} safe",
                    ha="center", va="center", fontsize=S.FS_NOTE, zorder=4,
                    color=tone, style="italic" if pinned else "normal",
                    fontweight="bold" if pinned else "normal")
            if pinned:
                ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                                       edgecolor=S.INK, hatch="////", lw=0.0,
                                       zorder=2, alpha=0.55))
    ax.set_xlabel("group size $N$   (decisions in the cell)", labelpad=1.5)
    ax.set_ylabel("max private risk $p_r^{\\max}$", labelpad=2)
    cb = fig.colorbar(im, ax=ax, fraction=0.030, pad=0.02)
    cb.set_label("Unsafe (%)", fontsize=S.FS_NOTE, labelpad=2)
    cb.ax.tick_params(labelsize=S.FS_NOTE, length=1.6)
    cb.outline.set_visible(False)
    S.panel(ax, "a", "group size pushes play into the ceiling, not up a plane")
    lo_n, hi_n = min(payoff), max(payoff)
    ax.annotate(wrap(f"{S.ROUTE_LABEL[route]}, {int(cells['races'].iloc[0])} races per cell, "
                     "self-play. Hatched: the cell contained no safe decision at all, so it "
                     "is pinned and its distance from the cells below is a floor, not an "
                     "estimate. The column is not group size alone: all-safe pays "
                     f"{payoff[lo_n]:+.1f} at $N$={lo_n} and {payoff[hi_n]:+.1f} at "
                     f"$N$={hi_n}.",
                     3.7),
                xy=(0.0, -0.31), xycoords="axes fraction", ha="left", va="top",
                fontsize=S.FS_NOTE, color=S.INK_2, annotation_clip=False,
                linespacing=1.45)


def draw_representation(ax, row, route, *, show_x):
    colour = S.ROUTE_C[route]
    cells = row["cells"]
    values = {(c, m): 100 * cells[(c, m)] for c in FRAME for m in CODE}
    upper = max(FRAME, key=lambda c: values[(c, CODE[0])] + values[(c, CODE[1])])

    top = max(values.values())
    if top > 95.0:
        # Snap the window to the boundary rather than centring it, so the reader
        # sees how little room the highest cell has left above it.
        lo, hi = 100.0 - B_SPAN, 100.0
    else:
        centre = float(np.mean(list(values.values())))
        lo = 5.0 * round((centre - B_SPAN / 2) / 5.0)
        hi = lo + B_SPAN

    for frame in FRAME:
        ys = [values[(frame, m)] for m in CODE]
        above = frame == upper
        ax.plot([0, 1], ys, ls=FRAME_LS[frame], color=colour, lw=1.15,
                marker=S.ROUTE_M[route], ms=3.0, mec=S.SURFACE, mew=0.6,
                clip_on=False, zorder=3)
        for x, y in zip((0, 1), ys):
            # A cell too close to an edge has no room for a label there,
            # and one placed anyway would sit on the ceiling rule or under the
            # axis, so it goes out sideways instead.
            if (above and y > hi - ABOVE_PAD) or (not above and y < lo + BELOW_PAD):
                ax.annotate(f"{y:.0f}", xy=(x, y), xytext=(-5.0 if x == 0 else 5.0, 0),
                            textcoords="offset points",
                            ha="right" if x == 0 else "left", va="center",
                            fontsize=S.FS_NOTE, color=colour, zorder=4)
            else:
                ax.annotate(f"{y:.0f}", xy=(x, y), xytext=(0, 4.4 if above else -4.4),
                            textcoords="offset points", ha="center",
                            va="bottom" if above else "top",
                            fontsize=S.FS_NOTE, color=colour, zorder=4)
        # Series names instead of a legend.  Staggered in x and pushed clear of
        # the pair, because on the flatter route the two lines are three points
        # apart and a label centred on one of them would break the other.  A name
        # hung off the right end instead would push the tight bounding box past
        # the column and make ``save`` rescale the whole figure to fit it.
        lx = 0.24 if above else 0.76
        ax.annotate(FRAME[frame], xy=(lx, ys[0] + lx * (ys[1] - ys[0])),
                    xytext=(0, 5.0 if above else -5.0), textcoords="offset points",
                    ha="center", va="bottom" if above else "top",
                    fontsize=S.FS_NOTE, color=colour, zorder=5,
                    bbox=dict(facecolor=S.SURFACE, edgecolor="none", pad=0.9))

    if hi >= 99.99:
        S.ceiling_rule(ax, 100.0, label="")
        ax.annotate("ceiling", xy=(0.01, 100.0), xycoords=("axes fraction", "data"),
                    xytext=(0, -2), textcoords="offset points", ha="left", va="top",
                    fontsize=S.FS_NOTE, color=S.MUTED)
    ax.set_xlim(-0.22, 1.22)
    ax.set_ylim(lo, hi)
    ax.set_yticks([lo, lo + B_SPAN / 2, hi])
    ax.set_xticks([0, 1])
    ax.set_xticklabels([CODE_LABEL[m] for m in CODE] if show_x else ["", ""])
    S.strip(ax, grid_axis="y")
    ax.set_ylabel("Unsafe (%)", labelpad=2)
    ax.annotate(S.ROUTE_LABEL[route], xy=(1.0, 1.0), xycoords="axes fraction",
                xytext=(0, 2.5), textcoords="offset points", ha="right", va="bottom",
                fontsize=S.FS_NOTE, color=colour, fontweight="bold",
                annotation_clip=False)
    # "a null" was the wording here, and an interval reaching +14.3 pp is not a
    # null, it is an interval that spans zero.  Say which one it is.
    ax.annotate(f"interaction {100 * row['interaction']:+.1f} pp "
                f"[{100 * row['interaction_lo']:+.1f}, {100 * row['interaction_hi']:+.1f}] "
                "spans 0",
                xy=(0.0, 1.0), xycoords="axes fraction", xytext=(0, 2.5),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=S.FS_NOTE, color=S.MUTED, annotation_clip=False)


def draw_seat(ax, per_route, pooled, allround, discordant):
    # Routes with room first, largest gap leftmost, and the four that are pinned
    # at the ceiling gathered on the right, so the eye reads the gap where it can
    # exist before it reaches the routes where it could not.
    pinned = (per_route["seat1"] >= 1) & (per_route["seat2"] >= 1)
    order = list(per_route.assign(pinned=pinned).sort_values(
        ["pinned", "d", "seat1"], ascending=[True, False, True]).index)
    xs = np.arange(len(order), dtype=float)
    for x, route in zip(xs, order):
        row = per_route.loc[route]
        y1, y2 = 100 * row["seat1"], 100 * row["seat2"]
        colour = S.ROUTE_C[route]
        # The two seats are drawn side by side rather than on one abscissa: five
        # of the nine routes give the two seats identical rates, and stacked on
        # one x those five would each render as a single marker, which is the one
        # reading this panel must not invite.
        ax.plot([x - SEAT_DX, x + SEAT_DX], [y1, y2], color=colour, lw=1.1,
                zorder=2, solid_capstyle="round")
        S.dot(ax, x - SEAT_DX, y1, color=colour, marker="o", size=17, filled=False)
        S.dot(ax, x + SEAT_DX, y2, color=colour, marker=S.ROUTE_M[route], size=17)
        if pinned[route]:
            ax.annotate("pinned", xy=(x, 100.0), xytext=(0, 6.5),
                        textcoords="offset points", ha="center", va="bottom",
                        fontsize=S.FS_NOTE, color=S.MUTED, style="italic")
        elif min(y1, y2) <= 0.0:
            # The other boundary, and it is a boundary in the same sense: this
            # seat opened unsafe in none of its thirty races, so the gap beside
            # it is measured off a floor rather than off a rate.
            floor_x = x - SEAT_DX if y1 <= y2 else x + SEAT_DX
            ax.annotate("floor", xy=(floor_x, min(y1, y2)), xytext=(5.0, 0),
                        textcoords="offset points", ha="left", va="center",
                        fontsize=S.FS_NOTE, color=S.MUTED, style="italic")
        if abs(y2 - y1) > 1.0:
            ax.annotate(f"{y2 - y1:+.0f}", xy=(x + SEAT_DX, 0.5 * (y1 + y2)),
                        xytext=(5.0, 0), textcoords="offset points",
                        ha="left", va="center",
                        fontsize=S.FS_NOTE, color=colour, fontweight="bold")

    S.ceiling_rule(ax, 100.0, label="")
    ax.annotate("ceiling: a pinned route is already here in both seats", xy=(0.006, 100.0),
                xycoords=("axes fraction", "data"), xytext=(0, 3),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=S.FS_NOTE, color=S.MUTED)
    ax.set_xticks(xs)
    ax.set_xticklabels([S.ROUTE_SHORT[r] for r in order])
    for tick, route in zip(ax.get_xticklabels(), order):
        tick.set_color(S.ROUTE_C[route])
    ax.set_xlim(-0.75, len(order) - 0.25)
    S.rate_axis(ax, label="Opening move: unsafe (%)")
    ax.set_ylim(-9, 116)
    S.strip(ax, grid_axis="y")

    first = order[0]
    S.direct_label(ax, xs[0] - SEAT_DX, 100 * per_route.loc[first, "seat1"], "seat 1",
                   color=S.ROUTE_C[first], ha="right", dx=-5)
    S.direct_label(ax, xs[0] + SEAT_DX, 100 * per_route.loc[first, "seat2"], "seat 2",
                   color=S.ROUTE_C[first], ha="right", dx=-6)
    m, lo, hi = pooled
    am, alo, ahi = allround
    # A stat block rather than a paragraph: the sentence is the panel's claim,
    # and a five-line paragraph anchored to the foot of the axes sat on the
    # bottom spine and let the gridlines run through its type.  The rectangle
    # the pinned routes leave under the ceiling is the only region of this panel
    # where the block crosses no marker, so the three pieces hang off one anchor
    # in that block and cannot drift apart.  Each carries a surface-coloured box
    # so the y grid stops at the type instead of striking through it.
    box = dict(facecolor=S.SURFACE, edgecolor="none", pad=0.9)
    head = (0.555, 0.80)
    ax.annotate("Opening move, seat 2 minus seat 1", xy=head,
                xycoords="axes fraction", ha="left", va="top",
                fontsize=S.FS_NOTE, color=S.MUTED, bbox=box, zorder=6)
    ax.annotate(f"{100 * m:+.1f} pp   [{100 * lo:+.1f}, {100 * hi:+.1f}]", xy=head,
                xycoords="axes fraction", xytext=(0, -10.5),
                textcoords="offset points", ha="left", va="top",
                fontsize=S.FS_NOTE + 1.6, color=S.INK, fontweight="bold",
                bbox=box, zorder=6)
    ax.annotate(f"self-play; races resampled within route, all nine\n"
                f"{discordant[0]} of the {discordant[1]} discordant openings favour seat 2\n"
                f"over every round the gap falls to {100 * am:+.1f} pp "
                f"[{100 * alo:+.1f}, {100 * ahi:+.1f}]",
                xy=head, xycoords="axes fraction", xytext=(0, -25.5),
                textcoords="offset points", ha="left", va="top",
                fontsize=S.FS_NOTE, color=S.INK_2, linespacing=1.5,
                bbox=box, zorder=6)
    S.panel(ax, "c", "the seat is meant to be inert; at the opening it is not quite")


def main() -> None:
    route_a, cells, payoff = group_size_grid()
    pinned_cells = int((cells["safe"] == 0).sum())
    print(f"  a  {S.ROUTE_LABEL[route_a]}, {len(cells)} cells, "
          f"{int(cells['decisions'].sum())} decisions, {pinned_cells} with zero safe decisions")
    print("       all-safe payoff by group size: "
          + ", ".join(f"N={n} {v:+.3f}" for n, v in sorted(payoff.items())))
    for _, c in cells.iterrows():
        print(f"       N={int(c['n_players'])} risk={c['cell_risk']:.1f}  "
              f"unsafe {c['rate']:6.2f}%  safe {int(c['safe']):3d}/{int(c['decisions'])}")
    if pinned_cells != 5:
        print(f"  !! expected 5 ceiling cells, computed {pinned_cells}")

    rep = representation()
    for route, row in rep.iterrows():
        print(f"  b  {S.ROUTE_SHORT[route]}  {row['pairs']} matched repetitions")
        for name in ("interaction", "code", "frame"):
            print(f"       {name:12s} {100 * row[name]:+7.2f} pp "
                  f"[{100 * row[f'{name}_lo']:+7.2f}, {100 * row[f'{name}_hi']:+7.2f}]"
                  + ("   NULL" if row[f"{name}_lo"] < 0 < row[f"{name}_hi"] else ""))
        print("       letter Q share by mapping: "
              + ", ".join(f"{CODE_LABEL[c]} {100 * row['q_' + c]:.1f}%" for c in CODE)
              + "   (a fixed letter habit would print one number twice)")
        gap = abs(row["code"] - row["code_identity"])
        print(f"       P(letter Q) = {100 * row['letter_q']:.2f}%, so 2(P(Q)-1/2) = "
              f"{100 * row['code_identity']:+.2f} pp against the code contrast "
              f"{100 * row['code']:+.2f} pp, |difference| {100 * gap:.6f} pp")
        if gap > 1e-9:
            print("  !! the counterbalancing identity does not hold")

    paired, per_route, pooled, allround = seat()
    disc = paired[paired["d"] != 0]
    discordant = (int((disc["d"] > 0).sum()), int(len(disc)))
    print(f"  c  {len(paired)} races, seat gap at the opening "
          f"{100 * pooled[0]:+.2f} pp [{100 * pooled[1]:+.2f}, {100 * pooled[2]:+.2f}], "
          f"all rounds {100 * allround[0]:+.2f} pp "
          f"[{100 * allround[1]:+.2f}, {100 * allround[2]:+.2f}]")
    print(f"       {discordant[0]} of {discordant[1]} discordant openings favour seat 2; "
          f"{int(((per_route['seat1'] >= 1) & (per_route['seat2'] >= 1)).sum())} routes pinned")
    for route, row in per_route.iterrows():
        print(f"       {S.ROUTE_SHORT[route]:>9}  seat1 {100 * row['seat1']:5.1f}  "
              f"seat2 {100 * row['seat2']:5.1f}  gap {100 * row['d']:+5.1f}")

    # The denominator is the whole disagreement about this number, so print what
    # every other denominator would have given rather than leaving a reader to
    # guess which one the panel drew.  Dropping a route because its measured gap
    # came out at zero is selection on the outcome, and it is not the reason the
    # panel keeps all nine: the panel keeps all nine because all nine were run
    # under the same design, thirty races each.
    for drop, label in ((None, "all nine routes, as drawn"),
                        ("anthropic/claude-opus-5@default",
                         "less Claude Opus 5, whose two seats never diverge")):
        block = paired if drop is None else paired[paired["model_route"] != drop]
        d_m, d_lo, d_hi = seat_gap_ci(block)
        print(f"       {label:52s} {100 * d_m:+6.2f} pp "
              f"[{100 * d_lo:+6.2f}, {100 * d_hi:+6.2f}] "
              f"over {block['model_route'].nunique()} routes")
    with_room = per_route[(per_route["seat1"] < 1) | (per_route["seat2"] < 1)].index
    r_m, r_lo, r_hi = seat_gap_ci(paired[paired["model_route"].isin(with_room)])
    print(f"       {'only the routes with room below the ceiling':52s} {100 * r_m:+6.2f} pp "
          f"[{100 * r_lo:+6.2f}, {100 * r_hi:+6.2f}] over {len(with_room)} routes")

    # Panel c is the row that has to hold a stat block inside its own axes, so
    # it gets height rather than the notes getting squeezed out of the figure.
    fig = plt.figure(figsize=(S.TEXT, 4.62))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.30, 1.22],
                          width_ratios=[1.46, 1.00], hspace=1.00, wspace=0.30)
    ax_a = fig.add_subplot(gs[0, 0])
    inner = gs[0, 1].subgridspec(2, 1, hspace=0.55)
    ax_b1 = fig.add_subplot(inner[0])
    ax_b2 = fig.add_subplot(inner[1])
    ax_c = fig.add_subplot(gs[1, :])

    draw_group_size(ax_a, fig, cells, route_a, payoff)

    draw_representation(ax_b1, rep.loc[MAP_ROUTES[0]], MAP_ROUTES[0], show_x=False)
    draw_representation(ax_b2, rep.loc[MAP_ROUTES[1]], MAP_ROUTES[1], show_x=True)
    # "frame and code do not interact" stood here, over an interval that
    # reaches +14.3 points on the second route.  An interval that wide does not
    # establish an absence; it establishes that the design did not resolve the
    # question.  The headline says which of the two it is.
    S.panel(ax_b1, "b", "frame and code: an imprecise estimate,\n"
            "consistent with no interaction", pad=13)
    anchor = rep.loc[MAP_ROUTES[1]]
    # The note used to print both contrasts as signed numbers and call them
    # "both move this route", which reads as one direction.  They are not one
    # direction: ``frame`` is race story minus neutral and ``code`` is
    # Safe = P minus Safe = Q, so the same minus sign means the race story
    # LOWERS unsafe play while the Q-means-Safe mapping RAISES it.  Say the
    # direction in words and declare the convention, because a reader cannot
    # infer a sign convention from a number.
    ax_b2.annotate(
        wrap("Near-parallel is not flat, and the two signs above point opposite ways: "
             "they are race story minus neutral, and Safe = P minus Safe = Q. So the "
             "race story lowers Unsafe play by "
             f"{abs(100 * anchor['frame']):.1f} pp, while the mapping where Q means Safe "
             f"raises it by {abs(100 * anchor['code']):.1f} pp. The code number is an identity "
             "of the counterbalancing, twice this route's excess use of the letter Q.",
             2.62),
        xy=(0.0, -0.46), xycoords="axes fraction", ha="left", va="top",
        fontsize=S.FS_NOTE, color=S.INK_2, annotation_clip=False, linespacing=1.45)

    draw_seat(ax_c, per_route, pooled, allround, discordant)

    S.save(fig, "what_moves_play", width=S.TEXT)


if __name__ == "__main__":
    main()
