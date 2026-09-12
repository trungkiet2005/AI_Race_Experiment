"""Humans are more diverse, and the gap is a floor rather than a percentile.

"More diverse" is the weakest possible statement of what the trajectory data
says, and it invites the obvious reply: more diverse by how much, and could a
human sample this small have landed down where the models are by luck?  So build
the null that reply deserves.  Twenty thousand times, redraw twenty participants
inside each risk condition, so every human draw is matched to a model cell in
sample size, in risk condition, and in the statistic computed on it.  Then ask
where each model cell falls in that null.  For twenty-one of the twenty-seven
model route-by-risk cells there is no percentile to report, because the null
never once reached down to where the cell sits.  That is a different kind of
claim from a small p-value and the figure is built to make the difference
visible: the region left of the human minimum is tinted, and it is empty of
human draws by construction, not by rounding.

Which nine routes, and why not the seven that were here before.  This figure
used to read ``MODEL_INPUTS`` from
``scripts/analyze_trajectory_diversity_rarefaction.py``, seven separate pilot
CSVs collected before the nine-route confirmatory campaign existed.  That set
was the wrong one to argue over.  It carried GPT-5 nano, a checkpoint no longer
reachable on the audited identity and therefore holding no admission verdict at
all, and it had no cell for GPT-5.4 or GPT-5.5, which are two of the five routes
the gate admits.  A sentence about "every checkpoint that passes the gate" was
being asserted over a set missing two of them.  So the trajectories now come
from ``scripts/analyze_trajectory_diversity_confirmatory.py``, which reads the
nine-route baseline campaign every other figure reads through ``figdata`` and
reuses the pilot analyser's own trajectory encoding, statistic for statistic and
seed for seed.  The encoding is therefore still not invented here: a trajectory
is the focal player's first five actions followed by its opponent's first five,
ten bits, and because the progress increments are fixed that string is the whole
trajectory.  The comparison size of twenty is the pilot analyser's, and the
human file is its de-identified source study.

Panels
  a  The matched null, one facet per risk condition, rows shared with panel b.
     The top row is the human null drawn as a histogram; the dashed rule is its
     smallest value and the tinted strip left of it is territory no human draw
     ever entered.  Each route sits on its own row at its own distinct count,
     with a hairline back to the rule so the distance below the floor is a
     length and not a memory.  Every one of the five admitted routes is left of
     that rule in all three facets.  The only cells that land inside the null
     belong to GPT-5.4 mini and GPT-5.4 nano, and the comprehension gate
     rejected both.
  b  The mechanism, on the same rows.  Every segment is one distinct paired
     five-round sequence and its width is that sequence's share of the sixty
     matched trajectories.  The human row is a comb of thin slivers; Claude
     Opus 5 spends all sixty on two sequences, and inside each risk cell on one.

What this figure does not show.  It is descriptive.  It measures observed
diversity in these samples, not latent policy entropy, and a cell below the
human floor says nothing about any checkpoint outside the nine drawn here,
which are commercial endpoints rather than a sample from a population of models.
The null is a human resampling null, so it answers "could a matched human sample
look this concentrated" and not "would this checkpoint look diverse under
another prompt, temperature or horizon".  Diversity is also not competence: the
two checkpoints that reach the human range are both checkpoints that failed the
gate, and that is marked.

The match is on sample size and not on the unit of independence, and that is
the load-bearing caveat.  A model cell is ten races drawn twice, once per seat,
and the two seats of a race carry the same ten bits in mirror order, so a cell
has ten independent units wearing twenty rows.  A human draw of twenty is
twenty participants, of whom one or two are typically a dyad.  Drawing ten
complete human dyads instead, which is the model's structure exactly, is the
second null this module builds, and both counts are printed, because quoting
only the larger one would be choosing the null after seeing which number it
gives.

The comparison is also human dyads against a model in self-play.  Twenty humans
are twenty people; twenty model rows are one policy replayed, so between-person
variation exists on one side of this figure by construction.  That is a reason
the human side is higher and it is not measured here.

Finally each human minimum is the minimum of a finite resampling, so the count
that rests on it has to be shown to survive the seed.  Across forty independent
participant nulls the minimum ranges 13 to 15 at risk 0.1, 16 to 17 at 0.6 and
15 to 16 at 0.9, and the headline count is twenty-one of twenty-seven under
every one of the forty, with the same two routes inside the null every time.
The dyad null is the softer of the two: its minimum ranges 11 to 13, 14 to 15
and 12 to 14 over forty draws and its count moves between sixteen and twenty,
which is why the dyad number is reported as the weaker companion and never on
its own.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402,F401
import analyze_trajectory_diversity_rarefaction as TD  # noqa: E402
import analyze_trajectory_diversity_confirmatory as CONF  # noqa: E402

N_NULL = 20_000
SEED = 20260910
COMPARISON_N = 20
DYAD_N = 10

# The diversity analyser names checkpoints by display label; the rest of the
# paper names them by model route, which is what carries colour, marker and
# admission verdict.  On the confirmatory campaign every drawn checkpoint is one
# of the nine audited routes, so the map is the registry's own and there is no
# checkpoint here without a verdict.
LABEL_TO_ROUTE = {S.ROUTE_LABEL[route]: route for route in S.ROUTE_ORDER}

# Arial ships no check or ballot glyph, so a tick prints as a hollow box.  A
# bracketed letter survives the font, greyscale and a photocopier, which is the
# entire job of a verdict mark.
VERDICT_MARK = {True: "[A]", False: "[F]"}
VERDICT_KEY = "[A] admitted    [F] failed the comprehension gate"

HUMAN_C = S.ROUTE_C["human"]
NULL_FILL = "#d5dade"
FORBIDDEN = "#f7eef0"

# Geometry shared by both panels, so a row means the same height in each.  The
# pitch between two rows is fixed rather than the top and bottom being fixed,
# because this module now draws two figures with different numbers of rows and a
# fixed frame would print the five-route version at nearly twice the row spacing
# of the nine-route one, which a reader would read as two different objects.
ROW_PITCH = 0.75
ROUTE_BOTTOM = 0.0
HUMAN_HEIGHT = 0.95
BAR_H = 0.55
RULE_LO = -0.30       # rules and tint stop here, leaving the note lane clean
NOTE_Y = -1.05
Y_LO = -1.55
X_LO, X_HI = 0.30, COMPARISON_N + 0.70


class Frame:
    """Where every row sits, for a panel carrying ``n`` model rows."""

    def __init__(self, n: int):
        self.n = n
        self.route_top = ROUTE_BOTTOM + ROW_PITCH * (n - 1)
        self.human_base = self.route_top + 1.05
        self.human_tick = self.route_top + 1.45
        self.top_y = self.route_top + 2.28   # the human minimum, clear of the title
        self.y_hi = self.route_top + 2.95

    def rows(self, order):
        return [(self.route_top - i * ROW_PITCH, label) for i, label in enumerate(order)]

    @property
    def height_in(self) -> float:
        """Canvas height that keeps the row pitch the same on the page."""
        return 1.36 + 0.222 * (self.y_hi - Y_LO)


def identity(label):
    """Colour, marker, short name and verdict for a checkpoint label."""
    route = LABEL_TO_ROUTE[label]
    verdict = route in S.ADMITTED
    return S.ROUTE_C[route], S.ROUTE_M[route], S.ROUTE_SHORT[route], verdict


def distinct_per_row(codes: np.ndarray) -> np.ndarray:
    """Distinct values in each row of an integer matrix.

    Sorting each row and counting where the value changes is the whole
    computation, and it turns twenty thousand draws into one array operation
    instead of twenty thousand Counters.
    """
    ordered = np.sort(codes, axis=1)
    return 1 + (np.diff(ordered, axis=1) != 0).sum(axis=1)


def human_null(cells: list[np.ndarray], rng):
    """Twenty thousand matched draws: twenty participants from each risk cell.

    Without replacement, because the question is what a study of this size drawn
    from this participant pool could have produced.  Drawing with replacement
    would let one participant stand in for several and would hand the null a
    concentration the human sample never actually shows.
    """
    per_cell = []
    for codes in cells:
        order = np.argsort(rng.random((N_NULL, codes.size)), axis=1)
        per_cell.append(codes[order[:, :COMPARISON_N]])
    return np.column_stack([distinct_per_row(c) for c in per_cell]), np.hstack(per_cell)


def dyad_null(dyads: list[np.ndarray], rng) -> np.ndarray:
    """The stricter null: ten complete human dyads, which is twenty rows again.

    A model cell is ten races replayed once per seat, so its twenty rows carry
    ten independent units.  Drawing ten whole dyads gives the human side exactly
    that structure, and it is drawn on its own RNG stream so that adding it here
    cannot move a single number in the participant null above.
    """
    floors = []
    for mat in dyads:
        picks = np.argsort(rng.random((N_NULL, len(mat))), axis=1)[:, :DYAD_N]
        floors.append(distinct_per_row(mat[picks].reshape(N_NULL, 2 * DYAD_N)))
    return np.column_stack(floors)


def composition(keys) -> np.ndarray:
    shares = np.array(sorted(Counter(keys).values(), reverse=True), dtype=float)
    return shares / shares.sum()


def draw_facet(ax, counts, cells, rows, *, risk, show_ylabels, show_xlabel, notes,
               frame, marks):
    floor = int(counts.min())
    hist = np.bincount(counts, minlength=COMPARISON_N + 1)[: COMPARISON_N + 1]
    Y_HI, HUMAN_BASE, HUMAN_TICK, TOP_Y = (
        frame.y_hi, frame.human_base, frame.human_tick, frame.top_y)

    # Every rule and the tint stop above ``NOTE_Y``, which leaves the bottom of
    # each facet as clean paper for the notes.  A dashed rule running through a
    # sentence is read as a strikethrough.
    ax.fill_betweenx([RULE_LO, Y_HI], X_LO, floor - 0.5, color=FORBIDDEN, lw=0,
                     zorder=0)
    ax.plot([floor - 0.5] * 2, [RULE_LO, Y_HI], color=S.UNSAFE_C, lw=0.9,
            ls=(0, (3, 2)), zorder=4)
    # The floor of the statistic itself.  One sequence for all twenty players is
    # not a low reading, it is the smallest number the measure can return, and a
    # reader who cannot see that boundary reads a saturated cell as merely small.
    ax.plot([1.0, 1.0], [RULE_LO, Y_HI], color=S.MUTED, lw=0.5, ls=(0, (1, 2)),
            zorder=1)
    # And the other boundary.  Most human draws sit exactly on 20, so the human
    # reference is a ceiling reading, and a reader who cannot see the ceiling
    # reads the human null as a distribution with room above it.
    ax.plot([float(COMPARISON_N)] * 2, [RULE_LO, Y_HI], color=S.MUTED, lw=0.5,
            ls=(0, (1, 2)), zorder=1)

    ax.fill_between(np.arange(COMPARISON_N + 1),
                    HUMAN_BASE, HUMAN_BASE + HUMAN_HEIGHT * hist / hist.max(),
                    step="mid", color=NULL_FILL, edgecolor=S.MUTED, lw=0.4,
                    zorder=3)
    ax.plot([X_LO, X_HI], [HUMAN_BASE, HUMAN_BASE], color=S.HAIRLINE, lw=0.6,
            zorder=2)

    for y, label in rows:
        colour, marker, _, _ = identity(label)
        value = cells[label]
        # A hairline back to the rule, so "below the human floor" is a length on
        # the page rather than something the reader has to hold in memory.  A
        # cell inside the null gets none: there is no shortfall to measure, and
        # a stem pointing the other way reads as an arrow rather than a gap.
        if value < floor:
            ax.plot([floor - 0.5, value], [y, y], color=colour, lw=0.6,
                    alpha=0.45, zorder=2)
        S.dot(ax, value, y, color=colour, marker=marker, size=14, zorder=5)

    ax.annotate(f"$p_r^{{\\max}}={S.RISK_LABEL[risk]}$", xy=(0.0, 1.0),
                xycoords="axes fraction", xytext=(0, 2.0),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=S.FS_NOTE, color=S.INK, fontweight="bold")
    ax.annotate(f"min {floor}", xy=(floor - 0.5, TOP_Y), xytext=(-3.5, 0),
                textcoords="offset points", ha="right",
                va="center", fontsize=S.FS_NOTE, color=S.UNSAFE_C)
    for text, x, y, colour, ha in notes:
        ax.annotate(text, xy=(x, y), ha=ha, va="center", fontsize=S.FS_NOTE,
                    color=colour, style="italic", linespacing=1.25,
                    annotation_clip=False)

    ax.set_xlim(X_LO, X_HI)
    ax.set_ylim(Y_LO, Y_HI)
    ax.set_xticks([1, 5, 10, 15, 20])
    ax.set_yticks([HUMAN_TICK] + [y for y, _ in rows])
    if show_ylabels:
        labels = ["Humans"]
        colours = [HUMAN_C]
        for _, label in rows:
            colour, _, short, verdict = identity(label)
            # The verdict mark is drawn only where the panel holds routes with
            # two different verdicts.  On the screened-only figure every row is
            # admitted, so a mark on every row would carry no information and
            # would put the screen's vocabulary back into a panel the split
            # exists to keep free of it.
            labels.append(f"{short} {VERDICT_MARK[verdict]}" if marks else short)
            colours.append(colour)
        ax.set_yticklabels(labels)
        for tick, colour in zip(ax.get_yticklabels(), colours):
            tick.set_color(colour)
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0, pad=3)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(S.HAIRLINE)
    if show_xlabel:
        ax.set_xlabel("distinct paired five-round sequences among 20", labelpad=2)


def main() -> None:
    human = TD.load_human()

    # The dyad each participant played in.  It is the only column this figure
    # needs beyond the analyser's own frame, and it is what makes the stricter
    # null possible at all.
    raw = pd.read_csv(TD.HUMAN_CSV, usecols=["participant_id", "group_id"])
    group_of = raw.drop_duplicates("participant_id").set_index("participant_id")["group_id"]
    human = human.assign(group=human["unit"].map(group_of))

    model = CONF.confirmatory_frame(TD)
    model["population"] = model["population"].map(lambda r: S.ROUTE_LABEL.get(r, r))
    missing = set(model["population"]) - set(LABEL_TO_ROUTE)
    if missing:
        raise SystemExit(f"{sorted(missing)} have no identity here; refusing to draw unlabelled")
    frames = {label: block for label, block in model.groupby("population")}
    if len(frames) != len(S.ROUTE_ORDER):
        raise SystemExit(f"{len(frames)} routes on the confirmatory table, not {len(S.ROUTE_ORDER)}")

    vocabulary: dict[str, int] = {}

    def code(key: str) -> int:
        return vocabulary.setdefault(key, len(vocabulary))

    human_cells = [
        np.array([code(k) for k in
                  human[np.isclose(human["risk_cap"], risk)]["trajectory"]], dtype=np.int32)
        for risk in S.RISKS
    ]
    # A dyad counts only when both of its players have a complete five-round
    # trajectory; a half dyad is not the unit this null is about.
    dyad_cells = []
    for risk in S.RISKS:
        block = human[np.isclose(human["risk_cap"], risk)]
        pairs = [g["trajectory"].tolist() for _, g in block.groupby("group") if len(g) == 2]
        dyad_cells.append(np.array([[code(k) for k in pair] for pair in pairs],
                                   dtype=np.int32))

    model_cells = {
        label: [frame[np.isclose(frame["risk_cap"], risk)]["trajectory"].tolist()
                for risk in S.RISKS]
        for label, frame in frames.items()
    }
    for label, cells in model_cells.items():
        sizes = [len(cell) for cell in cells]
        if sizes != [COMPARISON_N] * len(S.RISKS):
            raise SystemExit(f"{label} cells are {sizes}, not the matched size {COMPARISON_N}")

    rng = np.random.default_rng(SEED)
    null_counts, pooled = human_null(human_cells, rng)
    pooled_distinct = distinct_per_row(pooled)
    dyad_counts = dyad_null(dyad_cells, np.random.default_rng([SEED, 2]))

    model_counts = {label: [len(set(cell)) for cell in cells]
                    for label, cells in model_cells.items()}
    model_pooled = {label: len({k for cell in cells for k in cell})
                    for label, cells in model_cells.items()}

    floors = null_counts.min(axis=0)
    dyad_floors = dyad_counts.min(axis=0)
    below = [(risk, label) for i, risk in enumerate(S.RISKS)
             for label in model_counts if model_counts[label][i] < floors[i]]
    below_dyad = [(risk, label) for i, risk in enumerate(S.RISKS)
                  for label in model_counts if model_counts[label][i] < dyad_floors[i]]
    inside = [(risk, label, model_counts[label][i],
               float((null_counts[:, i] <= model_counts[label][i]).mean()))
              for i, risk in enumerate(S.RISKS)
              for label in model_counts if model_counts[label][i] >= floors[i]]
    n_cells = len(model_counts) * len(S.RISKS)
    saturated = [(risk, label) for i, risk in enumerate(S.RISKS)
                 for label in model_counts if model_counts[label][i] == 1]

    admitted = [S.ROUTE_LABEL[route] for route in S.ADMITTED]
    admitted_below = {label for _, label in below} >= set(admitted)
    inside_labels = sorted({label for _, label, _, _ in inside})
    inside_all_refused = all(not identity(label)[3] for label in inside_labels)

    print(f"  human sample {len(human)} complete trajectories, cells "
          f"{[c.size for c in human_cells]} at risk {list(S.RISKS)}")
    print(f"  null: {N_NULL} draws of {COMPARISON_N} per risk cell, seed {SEED}")
    for i, risk in enumerate(S.RISKS):
        col = null_counts[:, i]
        print(f"    risk {risk}: distinct mean {col.mean():.2f}, min {col.min()}, "
              f"max {col.max()}; {len(dyad_cells[i])} complete dyads, "
              f"dyad-null min {dyad_floors[i]}")
    print(f"  human pooled 60: mean {pooled_distinct.mean():.2f}, "
          f"median {int(np.median(pooled_distinct))}, min {pooled_distinct.min()}, "
          f"max {pooled_distinct.max()}")
    for route in S.ROUTE_ORDER:
        label = S.ROUTE_LABEL[route]
        mark = "admitted" if route in S.ADMITTED else "refused "
        print(f"    {mark}  {label:>22}: per cell {model_counts[label]}, "
              f"pooled 60 {model_pooled[label]}")
    print(f"  {len(below)} of {n_cells} model cells below the human minimum; "
          f"{len(below_dyad)} of {n_cells} below the stricter dyad minimum; "
          f"{len(saturated)} cells at the measure's floor of 1")
    print(f"  {len(inside)} cells inside the null: "
          + ", ".join(f"{lab} at risk {r} (distinct {q}, {100 * p:.1f}th pct)"
                      for r, lab, q, p in inside))
    print(f"  every admitted route below the human minimum at every risk: {admitted_below}")
    print(f"  every route reaching the human range is refused: {inside_all_refused} "
          f"({', '.join(inside_labels)})")

    # One draw rather than a summary of draws, so panel b shows a composition
    # that actually occurred.  The median draw is the representative one and its
    # count is an integer, which a mean over draws is not.
    median_distinct = int(np.median(pooled_distinct))
    chosen = int(np.flatnonzero(pooled_distinct == median_distinct)[0])
    inverse = {v: k for k, v in vocabulary.items()}
    human_profile = composition([inverse[c] for c in pooled[chosen]])

    order_all = sorted(model_pooled, key=lambda lab: (-model_pooled[lab], lab))

    shared = dict(
        null_counts=null_counts, model_counts=model_counts, model_cells=model_cells,
        model_pooled=model_pooled, human_profile=human_profile,
        median_distinct=median_distinct, below_dyad=below_dyad,
        inside_labels=inside_labels,
    )

    # The main paper draws the five routes the validity screen admitted, plus
    # the human reference.  Every strategic claim the body makes stands on those
    # five, and a figure that quietly sets four refused routes beside them
    # invites the reader to read the refused rows as results.  The two refused
    # routes that DO reach the human range are not hidden by that choice: they
    # are named on the screened figure as the only ones in the study that get
    # there, which is the honest version of the sentence and the one a reviewer
    # has to see.  The supplement then draws all nine.
    draw_figure("human_versus_model",
                [label for label in order_all
                 if LABEL_TO_ROUTE[label] in S.ADMITTED],
                marks=False, **shared)
    draw_figure("human_versus_model_all_routes", order_all, marks=True, **shared)


def draw_figure(stem, order, *, marks, null_counts, model_counts, model_cells,
                model_pooled, human_profile, median_distinct, below_dyad,
                inside_labels):
    """One figure over one roster.  Everything numeric was computed in ``main``."""
    frame = Frame(len(order))
    rows = frame.rows(order)
    y_of = dict((label, y) for y, label in rows)
    narrowest = order[-1]
    drawn_inside = [label for label in order if label in inside_labels]
    outside_inside = [label for label in inside_labels if label not in order]

    floors = null_counts.min(axis=0)
    below = [(risk, label) for i, risk in enumerate(S.RISKS)
             for label in order if model_counts[label][i] < floors[i]]
    n_cells = len(order) * len(S.RISKS)

    fig = plt.figure(figsize=(S.TEXT, frame.height_in))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.40)
    left = gs[0, 0].subgridspec(1, 3, wspace=0.13)
    axes_a = [fig.add_subplot(left[i]) for i in range(3)]
    ax_b = fig.add_subplot(gs[0, 1])

    # --- a: where each model cell falls in the matched human null -------------
    # One name per line.  Two names on one line is wider than a facet, and an
    # annotation that overruns its facet lands on the neighbouring note.
    def named(label):
        return (f"{identity(label)[2]} {VERDICT_MARK[identity(label)[3]]}"
                if marks else identity(label)[2])

    if drawn_inside:
        inside_note = "only " + "\nand ".join(named(lab) for lab in drawn_inside) \
                      + "\nland inside"
    else:
        inside_note = ("no route drawn here\nreaches the human range")
    for i, (ax, risk) in enumerate(zip(axes_a, S.RISKS)):
        notes = []
        if i == 0:
            notes.append(("1 = all 20 identical,\nthe floor of the measure",
                          0.6, NOTE_Y, S.MUTED, "left"))
        if i == 1:
            notes.append(("no human draw\never landed here", 1.2,
                          frame.route_top + 0.56, S.UNSAFE_C, "left"))
            notes.append(("20 = all 20 different,\n"
                          "the measure's ceiling",
                          0.6, NOTE_Y, S.MUTED, "left"))
        if i == 2:
            notes.append((inside_note, COMPARISON_N + 0.4, NOTE_Y, S.INK_2, "right"))
        draw_facet(ax, null_counts[:, i],
                   {label: model_counts[label][i] for label in order},
                   rows, risk=risk, show_ylabels=(i == 0), show_xlabel=(i == 1),
                   notes=notes, frame=frame, marks=marks)
    S.direct_label(axes_a[0], 1.3, frame.human_base + 0.10,
                   f"{N_NULL:,} draws\nof 20 humans", color=S.INK_2,
                   ha="left", va="bottom", dx=0, dy=0)
    S.panel(axes_a[0], "a",
            f"{len(below)} of {n_cells} model cells sit below every human draw",
            pad=16)

    # --- b: the mechanism, which sequences each population spends its 60 on ---
    for y, name in [(frame.human_tick, "Human")] + rows:
        if name == "Human":
            profile, colour, marker = human_profile, HUMAN_C, S.ROUTE_M["human"]
            count = median_distinct
        else:
            colour, marker, _, _ = identity(name)
            profile = composition([k for cell in model_cells[name] for k in cell])
            count = model_pooled[name]
        offset = 0.0
        for j, share in enumerate(profile):
            ax_b.barh(y, share, left=offset, height=BAR_H, color=colour,
                      alpha=1.0 if j % 2 == 0 else 0.55, edgecolor=S.SURFACE,
                      linewidth=0.3, zorder=3)
            offset += share
        ax_b.plot([-0.035], [y], marker=marker,
                  ms=4.6 if marker == "*" else 3.4, color=colour,
                  markeredgecolor=S.SURFACE, markeredgewidth=0.5,
                  clip_on=False, zorder=5)
        S.direct_label(ax_b, 1.0, y, f"{count}", color=colour, dx=4,
                       weight="bold" if name in ("Human", narrowest) else "normal")

    # The narrowest route is the boundary case of the whole figure: inside a risk
    # cell its twenty trajectories are one string, which is the floor of the
    # measure and not a small reading of it.  Said inside the bar, because the
    # pooled bar is the one place where that boundary stops being visible.
    if all(count == 1 for count in model_counts[narrowest]):
        ax_b.annotate("all 20 identical in every risk cell",
                      xy=(0.02, y_of[narrowest]),
                      ha="left", va="center", fontsize=S.FS_NOTE, color=S.SURFACE,
                      zorder=6)

    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(Y_LO, frame.y_hi)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_xticklabels(["0", "25", "50", "75", "100"])
    ax_b.set_yticks([])
    ax_b.set_xlabel("share of the 60 matched trajectories (%)", labelpad=2)
    S.strip(ax_b, left=False, grid_axis="x")
    ax_b.annotate("sequences\nused", xy=(1.0, frame.top_y), xytext=(4, 0),
                  textcoords="offset points", ha="left", va="center",
                  fontsize=S.FS_NOTE, color=S.MUTED, linespacing=1.15,
                  annotation_clip=False)
    # The headline count is against a null of twenty participants, which matches
    # the model cells in size but not in independence: a model cell is ten races
    # replayed once per seat. Against a null built from ten complete human dyads,
    # which is the model's structure exactly, the count is smaller. Both are
    # printed, because quoting only the larger one would be choosing the null
    # after seeing which number it gives.
    lines = []
    if marks:
        lines.append(VERDICT_KEY)
    # Only say "every" when it is every one.  On the nine-route roster it is
    # not, and the same sentence there would be false.
    if len(below) == n_cells:
        lines.append("every route drawn here sits below every human draw, "
                     "at every risk level")
    if outside_inside:
        lines.append("the only cells in this study that reach the human range "
                     "belong to\n" + " and ".join(identity(lab)[2]
                                                  for lab in outside_inside)
                     + ", which the validity screen refused")
    elif drawn_inside:
        lines.append("the cells that reach the human range belong to "
                     + " and ".join(named(lab) for lab in drawn_inside))
    lines.append(f"the human row is the median of the {N_NULL:,} matched draws")
    lines.append(f"{len(below)} is against a null of 20 participants; against one "
                 f"of 10 complete\ndyads, which matches the model cells' "
                 f"independence, it is "
                 f"{sum(1 for _r, lab in below_dyad if lab in order)} "
                 f"of {n_cells}")
    ax_b.annotate("\n".join(lines),
                  xy=(0.0, 0.0), xycoords="axes fraction", xytext=(-40, -26),
                  textcoords="offset points", ha="left", va="top",
                  fontsize=S.FS_NOTE, color=S.MUTED, linespacing=1.35,
                  annotation_clip=False)
    S.panel(ax_b, "b",
            f"{median_distinct} sequences for 60 humans, "
            f"{model_pooled[narrowest]} for {S.ROUTE_SHORT[LABEL_TO_ROUTE[narrowest]]}",
            pad=4)

    if below and len(below) != n_cells:
        print(f"  {stem}: {n_cells - len(below)} of {n_cells} drawn cells are NOT "
              "below the human minimum")
    S.save(fig, stem, width=S.TEXT)


if __name__ == "__main__":
    main()
