"""The game has a designed cliff at round five, and the routes walk over it.

The prompt every player reads carries one sentence about the horizon, and this
script parses it out of the corpus rather than trusting the protocol document:
"The race lasts at least 5 rounds.  After every completed round from round 5
onward, the race ends with probability 20%."  So the continuation probability is
exactly 1 while the race is short of round five and 0.8 from round five onward.
That is a sharp, designed, stated-in-the-prompt discontinuity in the shadow of
the future, and it is the one quantity in this game that the folk-theorem
tradition says should move cooperative play: a threat that is certain to be
carried out sustains cooperation that a threat with a one-in-five chance of
never arriving does not.  Every player is told where the cliff is before it
arrives.  One of the nine steps up at it, one steps down, and the other seven
have intervals that cover zero, several of them wide enough to hold a
twenty-point move in either direction.

Three confounds sit between this design and a discontinuity estimate, and the
figure handles them in the open rather than in a footnote.

Accumulation.  A later round is also a round with more history behind it and,
because private risk is the fraction of one's own completed actions that were
unsafe, with more accumulated exposure.  Round position and accumulated state
are collinear inside a race, so no contrast between "before round five" and
"round five onward" can separate them.  Panel b therefore reports a contrast
across round positions and says so; panel c is the design that does hold
accumulation nearly fixed, by comparing each race with itself one round later.
Every rung of that ladder is one round of extra history, and exactly one rung
crosses the cliff.

Survivorship.  Races have different sampled lengths, so the races that reach
round twelve are not the races that reached round three.  Every route was run
against the same ten horizon draws, once at each of the three risk levels, so no
route is favoured by this, but the race set still changes along the axis.  Panel
a prints the surviving race count under every round.  Panel b's filled estimate
and the first four rungs of panel c stay inside the common horizon, rounds one
to five, which every race reaches by construction, so there the contributing set
is fixed at thirty races per route and survivorship is not in play.  The last
two rungs of panel c reach past it, onto the twenty-four races per route that
get that far, and the panel says so on its face.

Opening moves.  Round one is not a round like the others: nothing has happened,
so the step out of it is a step out of a state that never recurs.  It is drawn
as the first rung of panel c, where it dwarfs the cliff, and that comparison is
the figure's argument.

Panels
  a  Unsafe play by round, one row per route, with the boundary drawn between
     round four and round five and the surviving race count under each column.
     Rows carry the route colours, so the figure needs no legend.  A ringed tile
     sits on the 100% ceiling, where the contrast above it is truncated rather
     than measured.
  b  The step at the boundary per route, on the common horizon and again on all
     rounds, with a percentile interval that resamples races inside their own
     risk cell.  Signs are mixed, most intervals cover zero, and the two that do
     not point in opposite directions.  The grey estimate is secondary for a
     second reason besides survivorship: over all rounds a long race brings more
     decisions than a short one, so the two windows weight races differently and
     even the route that plays one constant rate acquires a spread there.
  c  The placebo ladder.  Within-race change from one round to the next, pooled
     over routes with the nine route means as grey context.  The rung that
     crosses the cliff is the only rung after the opening two whose interval
     clears zero, so it is not nothing; but it is 6.5 pp against an opening step
     of 39.3 pp, and its sign is negative, which is the direction opposite to the
     one a shortening shadow is supposed to push.  The grey context is not
     decoration: three of the nine route means point the other way, so the
     pooled interval is a statement these nine routes make together and not one
     that each of them makes.  Resampling routes rather than races widens it to
     cover zero.

What this does not show.  Not a regression discontinuity: the running variable
is a round index that also carries history, the design has no observations
arbitrarily close to the threshold, and nothing here is a causal estimate of the
horizon.  Nor is it a test of whether these routes can reason about a horizon at
all; it is a test of whether their play moves when the stated horizon changes,
in this game, at this prompt version.  Nine commercial endpoints are not a
sample from a population of models.  Both seats of every race are the same
route, so what is measured is a route's horizon response against a copy of
itself; nothing here is a horizon response against a different opponent.  The
absence of a step is evidence about the
step, not proof that the prompt sentence was unread: panel a shows one route
whose play is a constant, and a constant cannot move at a boundary.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.transforms import offset_copy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402

N_BOOT = 4000
SEED = 20260910

# The prompt sentence the whole figure rests on.  Parsed, not assumed: if the
# corpus ever ships a different horizon rule the script fails here rather than
# drawing a boundary the players were never told about.
HORIZON_RE = re.compile(
    r"The race lasts at least (\d+) rounds\. After every completed round from "
    r"round (\d+) onward, the race ends with probability (\d+)%\."
)

CH_C = S.INK          # the common-horizon estimate, the one with no survivorship
ALL_C = S.MUTED       # the same contrast over all rounds, kept visibly secondary


def horizon_rule(turns: pd.DataFrame) -> tuple[int, int, float]:
    """Minimum length, first round that can end the race, and the stop chance."""
    hits = [HORIZON_RE.search(p) for p in turns["prompt"]]
    # One distinct rule is not enough on its own: a corpus where only some
    # prompts carry the sentence would also state one rule, and the figure's
    # premise is that every player read it before every decision.
    missing = sum(m is None for m in hits)
    if missing:
        raise SystemExit(
            f"{missing} of {len(hits)} prompts do not state the horizon rule")
    found = {m.groups() for m in hits}
    if len(found) != 1:
        raise SystemExit(f"the corpus states {len(found)} different horizon rules: {found}")
    minimum, first_stop, pct = (int(x) for x in found.pop())
    if minimum != first_stop:
        raise SystemExit(f"minimum length {minimum} but stops begin at {first_stop}")
    return minimum, first_stop, pct / 100.0


def race_table(turns: pd.DataFrame, boundary: int, steps: list[int]) -> pd.DataFrame:
    """One row per race: the pieces every bootstrapped statistic is built from.

    Sums and counts rather than rates, because a resampled route's rate is the
    ratio of the resampled sums and not the mean of the per-race rates.
    """
    per_round = (
        turns.groupby(["model_route", "game_id", "round"], as_index=False)
        .agg(unsafe=("unsafe", "sum"), decisions=("unsafe", "size"),
             risk=("max_private_risk", "first"))
    )
    rows = []
    for (route, game), race in per_round.groupby(["model_route", "game_id"]):
        by_round = race.set_index("round")
        pre = by_round[by_round.index < boundary]
        at = by_round[by_round.index == boundary]
        post = by_round[by_round.index >= boundary]
        record = {
            "model_route": route, "game_id": game,
            "risk": float(race["risk"].iloc[0]),
            "pre_unsafe": pre["unsafe"].sum(), "pre_n": pre["decisions"].sum(),
            "at_unsafe": at["unsafe"].sum(), "at_n": at["decisions"].sum(),
            "post_unsafe": post["unsafe"].sum(), "post_n": post["decisions"].sum(),
            "length": int(by_round.index.max()),
        }
        rate = by_round["unsafe"] / by_round["decisions"]
        for k in steps:
            record[f"step{k}"] = (
                rate.loc[k + 1] - rate.loc[k]
                if k in rate.index and k + 1 in rate.index else np.nan
            )
        rows.append(record)
    return pd.DataFrame(rows)


def cluster_draws(races: pd.DataFrame) -> np.ndarray:
    """Row indices for ``N_BOOT`` resamples, races drawn inside their own cell.

    The design fixes ten races per route per risk level, so a bootstrap that
    ignored the cell could return a route with no races at some risk at all and
    an interval that mixes in that imbalance.  Cells are laid out contiguously
    and in route order, so a route's own columns are a slice of the draw.
    """
    rng = np.random.default_rng(SEED)
    blocks = []
    for _, cell in races.groupby(["model_route", "risk"], sort=False):
        pos = np.asarray(cell.index)
        blocks.append(pos[rng.integers(0, pos.size, size=(N_BOOT, pos.size))])
    return np.concatenate(blocks, axis=1)


def boot_ci(values: np.ndarray) -> tuple[float, float]:
    return float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))


def step_at_boundary(races: pd.DataFrame, draws: np.ndarray, post: str) -> pd.DataFrame:
    """Unsafe rate at and after the boundary minus the rate before it, per route."""
    rows = []
    for route, block in races.groupby("model_route", sort=False):
        pos = np.asarray(block.index)
        columns = np.isin(draws[0], pos)  # cells are contiguous, so this is the slice
        take = draws[:, columns]
        pre = races["pre_unsafe"].to_numpy()[take].sum(1) / races["pre_n"].to_numpy()[take].sum(1)
        aft = (races[f"{post}_unsafe"].to_numpy()[take].sum(1)
               / races[f"{post}_n"].to_numpy()[take].sum(1))
        point = (block[f"{post}_unsafe"].sum() / block[f"{post}_n"].sum()
                 - block["pre_unsafe"].sum() / block["pre_n"].sum())
        lo, hi = boot_ci(aft - pre)
        rows.append({"model_route": route, "point": point, "lo": lo, "hi": hi})
    return pd.DataFrame(rows).set_index("model_route")


def ladder(races: pd.DataFrame, draws: np.ndarray, steps: list[int]) -> pd.DataFrame:
    """Pooled within-race change per rung, with the nine route means beside it."""
    rows = []
    for k in steps:
        v = races[f"step{k}"].to_numpy()
        seen = ~np.isnan(v)
        filled = np.where(seen, v, 0.0)
        counts = seen[draws].sum(1)
        means = filled[draws].sum(1) / counts
        lo, hi = boot_ci(means)
        per_route = races.groupby("model_route", sort=False)[f"step{k}"].mean()
        rows.append({
            "step": k, "point": float(np.nanmean(v)), "lo": lo, "hi": hi,
            "races": int(seen.sum() // races["model_route"].nunique()),
            "routes": per_route,
        })
    return pd.DataFrame(rows).set_index("step")


def _tile_ink(im, value: float) -> str:
    """Type colour for one tile, chosen from that tile's own relative luminance."""
    r, g, b = im.cmap(im.norm(value))[:3]
    chan = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
            for c in (r, g, b)]
    lum = 0.2126 * chan[0] + 0.7152 * chan[1] + 0.0722 * chan[2]
    return S.SURFACE if lum < 0.18 else S.INK


def draw_heat(ax, fig, table, order, counts, boundary):
    """The board itself, with the announced boundary drawn on it."""
    arr = table.reindex(order).to_numpy(dtype=float)
    rounds = list(table.columns)
    im = S.heat_tiles(
        ax, arr, [S.ROUTE_SHORT[r] for r in order], [str(r) for r in rounds],
        cmap="viridis", vmin=0.0, vmax=100.0, fmt=None,
        row_colors=[S.ROUTE_C[r] for r in order], gutter=1.4,
    )
    # The shared helper's single flip point puts white type on the bright end of
    # viridis and dark type on the dark end, which is the wrong way round: white
    # on the 100% yellow measures 1.3:1 and dark on the 17% purple 1.9:1.  So the
    # ink is chosen from the tile's own luminance instead, which never falls
    # below 4:1 anywhere on this map.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            v = arr[i, j]
            if v != v:
                continue
            ax.text(j, i, f"{v:.0f}", ha="center", va="center",
                    fontsize=S.FS_NOTE, color=_tile_ink(im, v))
    # A tile at 100 is not a measurement with room above it, and a heat map that
    # keeps quiet about that invites a truncated contrast to be read as a flat one.
    ceiling = 0
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] >= 100.0:
                S.highlight(ax, i, j, color=S.INK, lw=0.9, inset=0.40)
                ceiling += 1
    ax.axvline(rounds.index(boundary) - 0.5, color=S.INK, lw=1.5, zorder=6,
               clip_on=False)
    ax.annotate(f"a ringed tile sits on the 100% ceiling ({ceiling} of {arr.size})",
                xy=(1.0, 1.0), xycoords="axes fraction",
                xytext=(0, 2.5), textcoords="offset points", ha="right",
                va="bottom", fontsize=S.FS_NOTE, color=S.MUTED,
                annotation_clip=False)

    # Three stacked rows under the axis, each with its own left-hand header, so
    # the round index and the shrinking race set read as two facts rather than as
    # one crowded strip.  Offsets are in points because the panel is wide and low.
    ax.set_xticklabels([])
    ax.tick_params(axis="x", length=0)
    for j, (r, n) in enumerate(zip(rounds, counts)):
        ax.annotate(f"{r}", xy=(j, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -3), textcoords="offset points", ha="center",
                    va="top", fontsize=S.FS_TICK, color=S.INK_2,
                    annotation_clip=False)
        ax.annotate(f"{n}", xy=(j, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(0, -13), textcoords="offset points", ha="center",
                    va="top", fontsize=S.FS_NOTE,
                    color=S.INK_2 if n == max(counts) else S.MUTED,
                    annotation_clip=False)
    for dy, text, colour, size in ((-3, "Round", S.INK_2, S.FS_TICK),
                                   (-13, "races still running", S.MUTED, S.FS_NOTE)):
        ax.annotate(text, xy=(-0.5, 0.0), xycoords=("data", "axes fraction"),
                    xytext=(-5, dy), textcoords="offset points", ha="right",
                    va="top", fontsize=size, color=colour, annotation_clip=False)

    cb = fig.colorbar(im, ax=ax, fraction=0.021, pad=0.012)
    cb.set_label("Unsafe play (%)", fontsize=S.FS_NOTE, labelpad=1)
    cb.ax.tick_params(labelsize=S.FS_NOTE, length=1.6)
    cb.outline.set_visible(False)
    return im


def regime_bar(ax, split, n_cols, minimum, stop):
    """The two regimes, named under the axis, because the cliff is the premise."""
    # The rules sit below the two label rows, so the offset is in points and the
    # x half of the transform stays in data coordinates.
    rule = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=-23.5,
                       units="points")
    for lo, hi, text in (
        (-0.5, split - 0.5, f"rounds 1–{minimum - 1}: the race certainly continues"),
        (split - 0.5, n_cols - 0.5,
         f"round {minimum} onward: each completed round ends the race with "
         f"probability {stop:.0%}"),
    ):
        ax.annotate(text, xy=((lo + hi) / 2, 0.0),
                    xycoords=("data", "axes fraction"),
                    xytext=(0, -27), textcoords="offset points",
                    ha="center", va="top", fontsize=S.FS_NOTE, color=S.INK,
                    annotation_clip=False)
        ax.plot([lo + 0.08, hi - 0.08], [0, 0], transform=rule, color=S.HAIRLINE,
                lw=0.8, clip_on=False, zorder=2, solid_capstyle="butt")


def main() -> None:
    turns = D.baseline_turns()
    minimum, _, stop = horizon_rule(turns)
    boundary = minimum
    steps = [1, 2, 3, 4, 5, 6]

    lengths = turns.groupby("game_id")["round"].max()
    if lengths.min() < boundary:
        raise SystemExit(
            f"a race is {lengths.min()} rounds long, shorter than the stated minimum")
    n_routes = turns["model_route"].nunique()
    print(f"  prompt states: at least {minimum} rounds, then ends with probability "
          f"{stop:.0%}; continuation is 1.00 before round {boundary} and "
          f"{1 - stop:.2f} from it")
    for r in (boundary - 1, boundary, boundary + 1):
        alive = int((lengths >= r).sum())
        print(f"    round {r}: {alive} races alive, "
              f"{(lengths >= r + 1).sum() / alive:.2f} of them continue")

    races = race_table(turns, boundary, steps)
    races = races.sort_values(["model_route", "risk", "game_id"]).reset_index(drop=True)
    draws = cluster_draws(races)

    common = step_at_boundary(races, draws, "at")
    every = step_at_boundary(races, draws, "post")
    rungs = ladder(races, draws, steps)

    by_round = 100 * turns.pivot_table(index="model_route", columns="round",
                                       values="unsafe", aggfunc="mean")
    counts = [int((lengths >= r).sum() // n_routes) for r in by_round.columns]
    # A route whose per-round rate never moves cannot step at a boundary, and its
    # zero-width interval is that fact rather than a missing bar.
    flat = [r for r in by_round.index if by_round.loc[r].nunique() == 1]

    order = list(common["point"].sort_values(ascending=False).index)
    for route in order:
        c, e = common.loc[route], every.loc[route]
        print(f"  {S.ROUTE_SHORT[route]:>9}  round {boundary} vs 1-{boundary - 1} "
              f"{100 * c['point']:6.1f} [{100 * c['lo']:6.1f}, {100 * c['hi']:6.1f}]   "
              f"all rounds {100 * e['point']:6.1f} "
              f"[{100 * e['lo']:6.1f}, {100 * e['hi']:6.1f}]")
    for k, row in rungs.iterrows():
        print(f"  step {k}->{k + 1}: {100 * row['point']:6.1f} "
              f"[{100 * row['lo']:6.1f}, {100 * row['hi']:6.1f}] pp, "
              f"{row['races']} races per route")
    print(f"  constant across every round: {[S.ROUTE_SHORT[r] for r in flat]}")

    fig = plt.figure(figsize=(S.TEXT, 4.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.16, 1.00],
                          width_ratios=[1.00, 1.04], hspace=0.62, wspace=0.28)
    ax_a = fig.add_subplot(gs[0, :])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[1, 1])

    # --- a: the whole board, with the cliff drawn on it -----------------------
    draw_heat(ax_a, fig, by_round, order, counts, boundary)
    regime_bar(ax_a, list(by_round.columns).index(boundary), by_round.shape[1],
               minimum, stop)
    S.panel(ax_a, "a", "the cliff is announced, and the routes disagree about it")

    # --- b: the step each route takes at the boundary -------------------------
    ypos = np.arange(len(order))[::-1]
    for y, route in zip(ypos, order):
        for frame, colour, marker, dy, filled in (
            (common, CH_C, "o", 0.17, True),
            (every, ALL_C, "s", -0.17, False),
        ):
            row = frame.loc[route]
            ax_b.plot([100 * row["lo"], 100 * row["hi"]], [y + dy, y + dy],
                      lw=1.0, color=colour, solid_capstyle="round", zorder=3)
            S.dot(ax_b, 100 * row["point"], y + dy, color=colour, marker=marker,
                  size=13, filled=filled)
        if route in flat:
            # Placed clear of both intervals rather than beside the point, since
            # the grey one is not zero width: over all rounds a long race carries
            # more decisions than a short one, so even a constant policy moves
            # when the two windows weight races differently.
            ax_b.annotate("one rate in every round", xy=(38.0, y),
                          ha="right", va="center", fontsize=S.FS_NOTE,
                          color=S.MUTED, style="italic")
    S.zero_rule(ax_b, 0.0, vertical=True)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([S.ROUTE_SHORT[r] for r in order])
    for tick, route in zip(ax_b.get_yticklabels(), order):
        tick.set_color(S.ROUTE_C[route])
    ax_b.set_ylim(-0.7, len(order) + 1.85)
    ax_b.set_xlim(-33, 39)
    ax_b.set_xticks([-30, -20, -10, 0, 10, 20, 30])
    ax_b.set_xlabel("change in unsafe play at the boundary, percentage points")
    S.strip(ax_b, grid_axis="x")
    key_y = len(order) + 1.55
    S.direct_label(ax_b, 2, key_y,
                   f"round {boundary} against rounds 1–{boundary - 1},\n"
                   "every race contributes", color=CH_C, dx=0, va="top",
                   weight="bold")
    S.direct_label(ax_b, -2, key_y,
                   f"rounds {boundary}+ against 1–{boundary - 1},\n"
                   "survivorship in play", color=ALL_C, ha="right", dx=0,
                   va="top", weight="bold")
    S.panel(ax_b, "b", "a step across positions, not a clean jump")

    # --- c: the ladder that holds one round of history per rung ---------------
    xs = np.arange(len(rungs))
    cross = list(rungs.index).index(boundary - 1)
    ax_c.axvspan(cross - 0.44, cross + 0.44, color=S.BAND, zorder=0)
    for x, (k, row) in zip(xs, rungs.iterrows()):
        colour = S.UNSAFE_C if k == boundary - 1 else S.INK
        jitter = np.linspace(-0.19, 0.19, len(row["routes"]))
        ax_c.scatter(x + jitter, 100 * row["routes"].to_numpy(), s=5,
                     facecolors=S.SURFACE, edgecolors=S.HAIRLINE, linewidths=0.55,
                     zorder=2)
        ax_c.plot([x, x], [100 * row["lo"], 100 * row["hi"]], lw=1.2, color=colour,
                  solid_capstyle="round", zorder=4)
        S.dot(ax_c, x, 100 * row["point"], color=colour, marker="o", size=17,
              zorder=5)
    S.zero_rule(ax_c, 0.0)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([f"{k}→{k + 1}" for k in rungs.index])
    ax_c.set_xlim(-0.62, len(xs) - 0.38)
    ax_c.set_ylim(-88, 88)
    ax_c.set_yticks([-75, -50, -25, 0, 25, 50, 75])
    ax_c.set_xlabel("round pair, inside the same race")
    ax_c.set_ylabel("change in unsafe play (pp)")
    S.strip(ax_c, grid_axis="y")
    ax_c.annotate("crosses the cliff", xy=(cross, 0.015),
                  xycoords=("data", "axes fraction"), ha="center", va="bottom",
                  fontsize=S.FS_NOTE, color=S.UNSAFE_C, fontweight="bold")
    ax_c.annotate("the opening move,\nwhich crosses nothing",
                  xy=(0, 100 * rungs.loc[1, "point"]), xytext=(9, 0),
                  textcoords="offset points", ha="left", va="center",
                  fontsize=S.FS_NOTE, color=S.INK_2)
    short = rungs[rungs["races"] < rungs["races"].max()]
    listed = " and ".join(f"{k}\u2192{k + 1}" for k in short.index)
    ax_c.annotate(f"rungs {listed} rest on the {short['races'].max()} of "
                  f"{rungs['races'].max()} races\nper route that reach those "
                  "rounds",
                  xy=(1.0, 1.0), xycoords="axes fraction", xytext=(0, -1),
                  textcoords="offset points", ha="right", va="top",
                  fontsize=S.FS_NOTE, color=S.MUTED, style="italic")
    ax_c.annotate("each open circle is one route", xy=(1.0, 0.16),
                  xycoords="axes fraction", ha="right", va="center",
                  fontsize=S.FS_NOTE, color=S.MUTED, style="italic")
    S.panel(ax_c, "c", "the cliff rung is small, and points the wrong way")

    S.save(fig, "horizon_discontinuity", width=S.TEXT)


if __name__ == "__main__":
    main()
