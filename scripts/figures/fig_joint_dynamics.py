"""Where a race goes, once it gets somewhere.

The memory-one fingerprint already in the paper is a statement about one seat: it
says what a route plays given what it saw.  A race is not one seat.  The thing
that happens to the world in a round is the *joint* outcome, and with two seats
and two actions that outcome is a state in {both safe, one each, both racing}.
A race is then a walk on three states, which is the object an evolutionary game
theorist reasons about and the object this paper did not have.

The two mixed outcomes are collapsed into one ``split`` state.  These are
self-play races, so the two seats are the same route under the same prompt and
differ only by which chair the protocol sat them in; ``Company_1 unsafe`` and
``Company_2 unsafe`` are therefore the same event under a relabelling, and
keeping them apart would spend a third of the state space on the seat index.
The script tests the exchangeability it assumes rather than asserting it, and
prints the result: the seat that goes unsafe in a split round is Company_2 in a
little over half of them, close enough to a coin that a four-state chain would
buy nothing but noise, and far enough from it that the printed check belongs in
the record.  Transitions are counted only between consecutive rounds of the same
race.  The last round of a race has no successor and is dropped; each race draws
its own horizon once, before the first move, so that censoring cannot depend on
where the walk had got to.

Panels
  a  One transition matrix per route, in reading order from the route that
     spends most of its time in mutual restraint to the route that spends least.
     Rows are this round, columns the next, and a row sums to 100.  Claude Opus 5
     never changed state inside a race and the panel draws it as such: every
     count on the diagonal, a split row that has no value because in 279 rounds
     its two seats never once played differently, and tiles ringed because they
     sit on a boundary rather than near one.  That constancy is within a race
     and not across the design, and the distinction decides what the panel is
     allowed to say.  At risk 0.1 all ten of its races sat in mutual escalation
     for every round; at risk 0.6 and 0.9 all twenty sat in mutual restraint.
     So the route moves further with risk than any of the other eight, 100
     points of unsafe rate against 57 for the next most responsive, and calling
     its pooled matrix frozen or unresponsive would invert the manuscript's own
     reading of it.  It is a step policy: deterministic inside a cell, and the
     one route in this figure whose pooled matrix is an artefact of the pooling
     rather than a summary of it.
  b  Occupancy: how a race divides its rounds between the three states.
  c  The two roads back to mutual restraint, drawn against each other.  Repair
     from a split is the rarer road in every route where a split ever occurred,
     and four arrivals at restraint in five are both seats standing down at the
     same moment out of mutual escalation, 176 against 42.  The ordering is a
     property of the pooled rows and not of every regime inside them: at risk
     0.1, where restraint is reached at all in only 4% of rounds, the two roads
     swap and the split is the likelier one, 7.1% against 3.8%.

What this does not show.  A transition count is not a mechanism: nothing here
says a route noticed the state, and the same matrix would be produced by two
seats reacting only to their own accumulated risk.  Rows are pooled over the
three risk levels, and risk moves these dynamics hard, so a route's matrix is a
summary of three regimes and not a law; the script prints the per-risk matrices
so the pooling can be audited.  Thirty races per route is enough to separate a
route that never moves inside a race from a mobile one and not enough to pin a
single cell to a point.  These are self-play races: every number here is what a
route does against a copy of itself under the same prompt, which is a statement
about that route and not about how it would meet a different one.  Nine
commercial endpoints are not a sample from a population of models.
And a state is only a joint action: two routes can walk the same three states
with entirely different progress, payoff and accumulated private risk behind
them.
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

# The state is the number of seats that played Unsafe, which is what makes the
# collapse of the two mixed outcomes a relabelling rather than a choice.
STATES = (0, 1, 2)
STATE_C = {0: S.SAFE_C, 1: S.INK_2, 2: S.UNSAFE_C}
STATE_WORD = {0: "both safe", 1: "one each", 2: "both racing"}
CMAP = "Blues"          # high values print dark, so white numerals stay legible
N_BOOT = 2000
SEED = 260910
OPUS = "anthropic/claude-opus-5@default"


def joint_walk(turns: pd.DataFrame) -> pd.DataFrame:
    """One row per round of every race: its state and the state that followed."""
    wide = turns.pivot_table(
        index=["model_route", "max_private_risk", "game_id", "round"],
        columns="player", values="unsafe",
    )
    if wide.isna().any().any():
        raise SystemExit("a round is missing a seat; the joint state is undefined")
    walk = wide.reset_index()
    walk["state"] = (walk["Company_1"] + walk["Company_2"]).astype(int)
    walk = walk.sort_values(["model_route", "game_id", "round"])
    walk["next_state"] = walk.groupby(["model_route", "game_id"])["state"].shift(-1)
    return walk


def transitions(walk: pd.DataFrame) -> np.ndarray:
    """Counts from state i to state j, rows this round, columns next."""
    step = walk.dropna(subset=["next_state"])
    counts = np.zeros((3, 3))
    table = pd.crosstab(step["state"], step["next_state"])
    for i in STATES:
        for j in STATES:
            if i in table.index and float(j) in table.columns:
                counts[i, j] = table.loc[i, float(j)]
    return counts


def row_normalise(counts: np.ndarray) -> np.ma.MaskedArray:
    """Percentages, with a row that never occurred masked rather than zeroed."""
    totals = counts.sum(axis=1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        pct = 100.0 * counts / totals
    return np.ma.masked_invalid(pct)


def stationary(counts: np.ndarray) -> np.ndarray | None:
    """Left eigenvector of the row-stochastic matrix, when every row exists."""
    totals = counts.sum(axis=1)
    if (totals == 0).any():
        return None
    matrix = counts / totals[:, None]
    values, vectors = np.linalg.eig(matrix.T)
    vector = np.real(vectors[:, np.argmin(np.abs(values - 1.0))])
    return vector / vector.sum()


def clustered_ci(walk: pd.DataFrame, source: int, target: int, *, rng) -> tuple:
    """Race-clustered interval on P(target | source), resampled within its cell.

    Races are the independent unit, and the design fixes ten of them per route
    and risk level, so a resample that ignored the cell could return a draw with
    no race at some risk at all.
    """
    step = walk.dropna(subset=["next_state"])
    step = step[step["state"] == source]
    per_race = {
        game: (len(part), int((part["next_state"] == target).sum()))
        for game, part in step.groupby("game_id")
    }
    cells: dict[tuple, list] = {}
    for game, part in walk.groupby("game_id"):
        key = (part["model_route"].iloc[0], float(part["max_private_risk"].iloc[0]))
        cells.setdefault(key, []).append(game)
    draws = []
    for _ in range(N_BOOT):
        n = k = 0
        for games in cells.values():
            for game in rng.choice(games, size=len(games), replace=True):
                if game in per_race:
                    dn, dk = per_race[game]
                    n += dn
                    k += dk
        if n:
            draws.append(100.0 * k / n)
    total_n = sum(v[0] for v in per_race.values())
    total_k = sum(v[1] for v in per_race.values())
    point = 100.0 * total_k / total_n if total_n else float("nan")
    return point, float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)), total_k, total_n


def draw_matrix(ax, pct, *, title, colour):
    im = S.heat_tiles(
        ax, pct, [str(i) for i in STATES], [str(j) for j in STATES],
        cmap=CMAP, vmin=0.0, vmax=100.0, fmt="{:.0f}", gutter=1.6,
        # White on Blues at 56 is 2.9:1 and ink on the same tile is 6.5:1.  The
        # two curves cross at 71, so that is where the numeral should change
        # colour; a lower flip sets the mid-range cells in the weakest type in
        # the figure.
        textcolor_flip=0.71,
        row_colors=[STATE_C[i] for i in STATES],
    )
    im.cmap.set_bad(S.GREY_BAD)
    for tick, state in zip(ax.get_xticklabels(), STATES):
        tick.set_color(STATE_C[state])
    ax.tick_params(axis="both", pad=1.2)
    for i in STATES:
        if np.ma.getmaskarray(pct)[i].all():
            # No value because the row never happened, which is the finding.
            ax.text(1, i, "never occurred", ha="center", va="center",
                    fontsize=S.FS_NOTE, color=S.INK_2, style="italic")
            continue
        for j in STATES:
            value = float(pct[i, j])
            if value in (0.0, 100.0):
                S.highlight(ax, i, j, color=S.INK, lw=0.75, inset=0.47)
    ax.annotate(title, xy=(0.5, 1.0), xycoords="axes fraction",
                xytext=(0, 2.4), textcoords="offset points",
                ha="center", va="bottom", fontsize=S.FS_NOTE,
                color=colour, fontweight="bold", annotation_clip=False)
    return im


def draw_key(ax):
    ax.set_axis_off()
    lines = [
        ("row: this round", S.INK, "normal"),
        ("column: the next", S.INK, "normal"),
        ("darker: more likely", S.MUTED, "normal"),
    ]
    y = 1.02
    for text, colour, weight in lines:
        ax.text(0.0, y, text, transform=ax.transAxes, ha="left", va="top",
                fontsize=S.FS_NOTE, color=colour, fontweight=weight)
        y -= 0.155
    y -= 0.05
    for state in STATES:
        ax.text(0.0, y, f"{state}   {STATE_WORD[state]}", transform=ax.transAxes,
                ha="left", va="top", fontsize=S.FS_NOTE, color=STATE_C[state],
                fontweight="bold")
        y -= 0.155
    ax.text(0.0, y - 0.02, "a ringed tile sits on a\nboundary: 0 or 100",
            transform=ax.transAxes, ha="left", va="top", fontsize=S.FS_NOTE,
            color=S.MUTED, linespacing=1.35)


def main() -> None:
    turns = D.baseline_turns()
    walk = joint_walk(turns)
    rng = np.random.default_rng(SEED)

    split = walk[walk["state"] == 1]
    seat_two = float((split["Company_2"] == 1).mean())
    print(f"  {walk['model_route'].nunique()} routes, {walk['game_id'].nunique()} races, "
          f"{len(walk)} rounds, {len(walk.dropna(subset=['next_state']))} transitions")
    print(f"  seat check: Company_2 is the racing seat in {100 * seat_two:.1f}% of "
          f"{len(split)} split rounds")

    order = (
        walk.groupby("model_route")["state"]
        .apply(lambda s: float((s == 0).mean()))
        .sort_values(ascending=False)
    )
    routes = list(order.index)

    counts = {r: transitions(walk[walk["model_route"] == r]) for r in routes}
    pcts = {r: row_normalise(counts[r]) for r in routes}
    occupancy = {
        r: np.array([float((walk.loc[walk["model_route"] == r, "state"] == i).mean())
                     for i in STATES])
        for r in routes
    }

    print("  route            occupancy 0/1/2        stay|0   0|split   0|both racing")
    for r in routes:
        pct = pcts[r]
        stay = float(pct[0, 0])
        repair = "  never" if np.ma.getmaskarray(pct)[1].all() else f"{float(pct[1, 0]):6.1f}"
        down = f"{float(pct[2, 0]):6.1f}"
        occ = " ".join(f"{100 * v:5.1f}" for v in occupancy[r])
        stat = stationary(counts[r])
        tail = "" if stat is None else "  stationary " + " ".join(f"{100 * v:5.1f}" for v in stat)
        print(f"  {S.ROUTE_SHORT[r]:>9}  {occ}   {stay:6.1f}   {repair}   {down}{tail}")

    # The degenerate route has to be held out of the pooled claim: it contributes
    # 166 self-transitions at exactly 100% and would carry the average on its own.
    mobile = walk[walk["model_route"] != OPUS]
    pooled = row_normalise(transitions(mobile))
    print(f"  {mobile['model_route'].nunique()} mobile routes, pooled rows (%):")
    for i in STATES:
        print(f"    from {i}: " + " ".join(f"{float(pooled[i, j]):5.1f}" for j in STATES))
    for source, target, name in ((0, 0, "restraint holds"),
                                 (1, 0, "split repairs"),
                                 (2, 0, "both stand down")):
        point, lo, hi, k, n = clustered_ci(mobile, source, target, rng=rng)
        print(f"  {name:>17}: {point:5.1f}% [{lo:5.1f}, {hi:5.1f}]  ({k} of {n})")

    entries = transitions(mobile)
    from_split, from_race = entries[1, 0], entries[2, 0]
    print(f"  arrivals at mutual restraint: {from_split:.0f} from a split, "
          f"{from_race:.0f} from mutual escalation "
          f"({100 * from_race / (from_split + from_race):.1f}% by simultaneous stand-down)")

    # Both pools, because the step route dominates the restraint row wherever it
    # sits in restraint: at risk 0.6 it moves the pooled 0->0 cell from 25 to 58.
    # A number lifted from the nine-route line into prose would be its artefact.
    for risk, block in walk.groupby("max_private_risk"):
        for tag, part in (("all 9", block),
                          ("mobile 8", block[block["model_route"] != OPUS])):
            pct = row_normalise(transitions(part))
            rows = "  ".join("/".join(f"{float(pct[i, j]):.0f}" for j in STATES)
                             for i in STATES)
            occ = " ".join(f"{100 * float((part['state'] == i).mean()):.0f}" for i in STATES)
            print(f"  risk {risk} {tag:>8}: rows {rows}   occupancy {occ}")

    # ---- figure ----------------------------------------------------------
    fig = plt.figure(figsize=(S.TEXT, 4.48))
    outer = fig.add_gridspec(2, 1, height_ratios=[1.95, 1.14], hspace=0.42)
    grid = outer[0].subgridspec(2, 5, wspace=0.34, hspace=0.62)
    lower = outer[1].subgridspec(1, 2, width_ratios=[1.0, 1.16], wspace=0.40)

    axes = []
    for k, route in enumerate(routes):
        ax = fig.add_subplot(grid[k // 5, k % 5])
        draw_matrix(ax, pcts[route], title=S.ROUTE_SHORT[route],
                    colour=S.ROUTE_C[route])
        axes.append(ax)
    draw_key(fig.add_subplot(grid[1, 4]))
    breaks = sum(1 for r in routes if float(pcts[r][0, 0]) < 50.0)
    # Derived from the condition the claim states rather than from the empty
    # split row, so the sentence cannot outlive the thing it describes.
    still = [r for r in routes if counts[r].sum() == np.trace(counts[r])]
    S.panel(axes[0], "a",
            f"{S.ROUTE_SHORT[still[0]]} never changes state inside a race, and in "
            f"{breaks} of {len(routes)} routes restraint breaks more often than "
            f"it holds", pad=19)

    # --- b: occupancy --------------------------------------------------------
    ax_b = fig.add_subplot(lower[0, 0])
    ypos = np.arange(len(routes))[::-1]
    for y, route in zip(ypos, routes):
        left = 0.0
        for state in STATES:
            share = 100 * occupancy[route][state]
            ax_b.barh(y, share, left=left, height=0.70,
                      color=STATE_C[state], edgecolor=S.SURFACE, linewidth=0.7)
            if share >= 6:
                ax_b.text(left + share / 2, y, f"{share:.0f}", ha="center",
                          va="center", fontsize=S.FS_NOTE, color=S.SURFACE)
            elif share == 0:
                # An empty state is a boundary and a stacked bar draws nothing
                # where nothing happened, so the boundary needs its own mark.
                ax_b.plot([left, left], [y - 0.37, y + 0.37], color=S.INK,
                          lw=1.0, zorder=4, solid_capstyle="butt")
            left += share
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([S.ROUTE_SHORT[r] for r in routes])
    for tick, route in zip(ax_b.get_yticklabels(), routes):
        tick.set_color(S.ROUTE_C[route])
    ax_b.set_xlim(0, 100)
    ax_b.set_xticks([0, 50, 100])
    ax_b.set_xlabel("share of rounds (%)", labelpad=1)
    # The same nine rows as c, so the two panels read as one table and a reader
    # can carry a route across without recounting.
    ax_b.set_ylim(-0.7, len(routes) + 0.85)
    S.strip(ax_b, grid_axis=None)
    ax_b.spines["left"].set_visible(False)
    ax_b.tick_params(axis="y", length=0)
    # The stack order is the key, so it is named in the panel rather than left
    # to the colours and a legend two panels away.
    # Each word sits over the span its own colour actually occupies in eight of
    # the nine rows.  Centring "one each" on 50 would put it over green or red
    # in five of them, which teaches the reader the wrong stack.
    for x, state, ha in ((0, 0, "left"), (33, 1, "center"), (100, 2, "right")):
        S.direct_label(ax_b, x, len(routes) + 0.42, STATE_WORD[state],
                       color=STATE_C[state], ha=ha, dx=0, weight="bold")
    S.panel(ax_b, "b", f"only {S.ROUTE_SHORT[still[0]]} spends most rounds safe, "
                       f"and it never splits")

    # --- c: the two roads back to restraint ----------------------------------
    ax_c = fig.add_subplot(lower[0, 1])
    step = walk.dropna(subset=["next_state"])
    for y, route in zip(ypos, routes):
        block = step[step["model_route"] == route]
        values = {}
        for source in (1, 2):
            sub = block[block["state"] == source]
            values[source] = (
                (float((sub["next_state"] == 0).mean()) * 100, len(sub))
                if len(sub) else (float("nan"), 0)
            )
        if np.isnan(values[1][0]):
            ax_c.annotate("no split ever occurred", xy=(values[2][0], y),
                          xytext=(6, 0), textcoords="offset points", ha="left",
                          va="center", fontsize=S.FS_NOTE, color=S.INK_2,
                          style="italic")
        else:
            ax_c.plot([values[1][0], values[2][0]], [y, y], lw=0.9,
                      color=S.HAIRLINE, zorder=2, solid_capstyle="round")
        for source, marker in ((1, "o"), (2, "s")):
            value, n = values[source]
            if np.isnan(value):
                continue
            S.dot(ax_c, value, y, color=STATE_C[source], marker=marker, size=15)
            if value == 0.0:
                # A boundary, not a small number: it never once happened.
                ax_c.annotate(f"0 of {n}", xy=(value, y), xytext=(-4, 0),
                              textcoords="offset points", ha="right", va="center",
                              fontsize=S.FS_NOTE, color=STATE_C[source])
    S.zero_rule(ax_c, 0.0, vertical=True, lw=0.8, color=S.HAIRLINE)
    ax_c.set_yticks(ypos)
    ax_c.set_yticklabels([S.ROUTE_SHORT[r] for r in routes])
    for tick, route in zip(ax_c.get_yticklabels(), routes):
        tick.set_color(S.ROUTE_C[route])
    ax_c.set_ylim(-0.7, len(routes) + 0.85)
    ax_c.set_xlim(-9, 46)
    ax_c.set_xticks([0, 10, 20, 30, 40])
    ax_c.set_xlabel("chance the next round is mutual restraint (%)", labelpad=1)
    S.strip(ax_c, grid_axis="x")
    ax_c.spines["left"].set_visible(False)
    ax_c.tick_params(axis="y", length=0)
    S.panel(ax_c, "c", "the way back does not run through a split")
    S.direct_label(ax_c, 0, len(routes) + 0.42, "after a split",
                   color=STATE_C[1], ha="left", dx=0, weight="bold")
    S.direct_label(ax_c, 44, len(routes) + 0.42, "after both raced",
                   color=STATE_C[2], ha="right", dx=0, weight="bold")

    S.save(fig, "joint_dynamics", width=S.TEXT)


if __name__ == "__main__":
    main()
