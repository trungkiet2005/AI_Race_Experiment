"""Humans are more diverse, and the gap is a floor rather than a percentile.

"More diverse" is the weakest possible statement of what the trajectory data
says, and it invites the obvious reply: more diverse by how much, and could a
human sample this small have landed down where the models are by luck?  So build
the null that reply deserves.  Twenty thousand times, redraw twenty participants
inside each risk condition, so every human draw is matched to a model cell in
sample size, in risk condition, and in the statistic computed on it.  Then ask
where each model cell falls in that null.  For eighteen of the twenty-one model
route-by-risk cells there is no percentile to report, because the null never
once reached down to where the cell sits.  That is a different kind of claim
from a small p-value and the figure is built to make the difference visible: the
region left of the human minimum is tinted, and it is empty of human draws by
construction, not by rounding.

The trajectory encoding is not invented here.  It is imported from
``scripts/analyze_trajectory_diversity_rarefaction.py``, which owns the
definition the manuscript publishes: a trajectory is the focal player's first
five actions followed by its opponent's first five, ten bits, and because the
progress increments are fixed that string is the whole trajectory.  The
comparison size of twenty is that analyser's, the seven checkpoints are its
``MODEL_INPUTS``, and the human file is its de-identified source study.  A
number here can therefore disagree with the manuscript only if the manuscript is
wrong.

Panels
  a  The matched null, one facet per risk condition, rows shared with panel b.
     The top row is the human null drawn as a histogram; the dashed rule is its
     smallest value and the tinted strip left of it is territory no human draw
     ever entered.  Each route sits on its own row at its own distinct count,
     with a hairline back to the rule so the distance below the floor is a
     length and not a memory.  Only GPT-5.4 nano lands inside the null, and it
     is the checkpoint the comprehension gate rejected.
  b  The mechanism, on the same rows.  Every segment is one distinct paired
     five-round sequence and its width is that sequence's share of the sixty
     matched trajectories.  The human row is a comb of thin slivers; Claude
     Opus 5 spends all sixty on two sequences, and inside each risk cell on one.

What this figure does not show.  It is descriptive.  It measures observed
diversity in these samples, not latent policy entropy, and a cell below the
human floor says nothing about any checkpoint outside the seven drawn here,
which are commercial endpoints rather than a sample from a population of models.
Three routes the nine-route baseline reports, GPT-5.4, GPT-5.4 mini and
GPT-5.5, have no cell in the pilot export this analyser reads, so they are
absent from this figure; absent is not diverse and not concentrated, and the
figure says so rather than leaving seven rows to be read as nine.  They are not
absent from ``results/``: ``baseline_campaign_v6``, the campaign every other
figure reads through ``figdata``, carries their paired actions in
``turns.jsonl`` like every other route.  Read this figure as the pilot export's
seven checkpoints and nothing wider.  The null is a human
resampling null, so it answers "could a matched human sample look this
concentrated" and not "would this checkpoint look diverse under another prompt,
temperature or horizon".  Diversity is also not competence: the one checkpoint
that matches people is the one that failed the gate, and that is marked.
The match is on sample size and not on the unit of independence, and that is
the load-bearing caveat.  A model cell is ten races drawn twice, once per seat,
and the two seats of a race carry the same ten bits in mirror order, so a cell
has ten independent units wearing twenty rows.  A human draw of twenty is
twenty participants, of whom one or two are typically a dyad.  Drawing ten
complete human dyads instead, which is the model's structure exactly, lowers
the minimum to 12, 15 and 12 and leaves fifteen cells below it rather than
eighteen.  The direction survives that, and the collapsed routes stay far below
either null, but eighteen is a count against the looser of the two.

The comparison is also human dyads against a model in self-play.  Twenty humans
are twenty people; twenty model rows are one policy replayed, so between-person
variation exists on one side of this figure by construction.  That is a reason
the human side is higher and it is not measured here.

Finally the human minimum is the minimum of a finite resampling.  Across forty
independent nulls it ranges 13 to 15 at risk 0.1, 16 to 17 at 0.6 and 15 to 16
at 0.9, and GPT-5 nano's thirteen distinct sequences at risk 0.1 sit inside that
range, so on another seed the count reads seventeen.  Every other cell below the
rule is at least three under the lowest floor those forty nulls produced.
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402,F401
import analyze_trajectory_diversity_rarefaction as TD  # noqa: E402

N_NULL = 20_000
SEED = 20260910
COMPARISON_N = 20

# The diversity analyser names checkpoints by display label; the rest of the
# paper names them by model route, which is what carries colour, marker and
# admission verdict.  GPT-5 nano is no longer offered on the audited identity,
# so it has no route entry and no verdict, and that absence is drawn rather than
# quietly filled in.
LABEL_TO_ROUTE = {
    "GPT-5 nano": None,
    "GPT-5.4 nano": "openai/gpt-5.4-nano-2026-03-17",
    "Gemini 3 Flash": "google/gemini-3-flash-preview",
    "Gemini 3.1 Flash Lite": "google/gemini-3.1-flash-lite-preview",
    "Gemini 3.5 Flash Lite": "google/gemini-3.5-flash-lite",
    "Claude Opus 5": "anthropic/claude-opus-5@default",
    "Claude Sonnet 5": "anthropic/claude-sonnet-5@default",
}
ORPHAN_SHORT = "GPT-5 nano"
ORPHAN_MARKER = "h"

# Arial ships no check or ballot glyph, so a tick prints as a hollow box.  A
# bracketed letter survives the font, greyscale and a photocopier, which is the
# entire job of a verdict mark.
VERDICT_MARK = {True: "[A]", False: "[F]", None: "[-]"}
VERDICT_KEY = "[A] admitted    [F] failed the comprehension gate    [-] no verdict"

HUMAN_C = S.ROUTE_C["human"]
NULL_FILL = "#d5dade"
FORBIDDEN = "#f7eef0"

# Geometry shared by both panels, so a row means the same height in each.
ROUTE_TOP = 6.0
HUMAN_BASE = 7.05
HUMAN_HEIGHT = 0.95
HUMAN_TICK = 7.45
Y_LO, Y_HI = -1.55, 8.95
TOP_Y = 8.28          # the human minimum, clear of the facet title above it
RULE_LO = -0.30       # rules and tint stop here, leaving the note lane clean
NOTE_Y = -1.05
X_LO, X_HI = 0.30, COMPARISON_N + 0.70


def identity(label):
    """Colour, marker, short name, verdict, and whether the marker is filled.

    GPT-5 nano is drawn open.  It is the one checkpoint with no route entry and
    no admission verdict, and an unfilled marker carries that in the same
    channel as the shape, so it cannot be confused with GPT-5.4 nano, which owns
    the registry's grey.
    """
    route = LABEL_TO_ROUTE[label]
    if route is None:
        return S.MUTED, ORPHAN_MARKER, ORPHAN_SHORT, None, False
    verdict = True if route in S.ADMITTED else (False if route in S.REFUSED else None)
    return S.ROUTE_C[route], S.ROUTE_M[route], S.ROUTE_SHORT[route], verdict, True


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


def composition(keys) -> np.ndarray:
    shares = np.array(sorted(Counter(keys).values(), reverse=True), dtype=float)
    return shares / shares.sum()


def draw_facet(ax, counts, cells, rows, *, risk, show_ylabels, show_xlabel, notes):
    floor = int(counts.min())
    hist = np.bincount(counts, minlength=COMPARISON_N + 1)[: COMPARISON_N + 1]

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
        colour, marker, _, _, filled = identity(label)
        value = cells[label]
        # A hairline back to the rule, so "below the human floor" is a length on
        # the page rather than something the reader has to hold in memory.  A
        # cell inside the null gets none: there is no shortfall to measure, and
        # a stem pointing the other way reads as an arrow rather than a gap.
        if value < floor:
            ax.plot([floor - 0.5, value], [y, y], color=colour, lw=0.6,
                    alpha=0.45, zorder=2)
        S.dot(ax, value, y, color=colour, marker=marker, size=16, zorder=5,
              filled=filled)

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
            colour, _, short, verdict, _ = identity(label)
            labels.append(f"{short} {VERDICT_MARK[verdict]}")
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
    frames = {label: TD.load_model(label, path) for label, path in TD.MODEL_INPUTS}
    for label in frames:
        if label not in LABEL_TO_ROUTE:
            raise SystemExit(f"{label!r} has no identity here; refusing to draw it unlabelled")

    vocabulary: dict[str, int] = {}

    def code(key: str) -> int:
        return vocabulary.setdefault(key, len(vocabulary))

    human_cells = [
        np.array([code(k) for k in
                  human[np.isclose(human["risk_cap"], risk)]["trajectory"]], dtype=np.int32)
        for risk in S.RISKS
    ]
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

    model_counts = {label: [len(set(cell)) for cell in cells]
                    for label, cells in model_cells.items()}
    model_pooled = {label: len({k for cell in cells for k in cell})
                    for label, cells in model_cells.items()}

    floors = null_counts.min(axis=0)
    below = [(risk, label) for i, risk in enumerate(S.RISKS)
             for label in model_counts if model_counts[label][i] < floors[i]]
    inside = [(risk, label, model_counts[label][i],
               float((null_counts[:, i] <= model_counts[label][i]).mean()))
              for i, risk in enumerate(S.RISKS)
              for label in model_counts if model_counts[label][i] >= floors[i]]
    n_cells = len(model_counts) * len(S.RISKS)
    saturated = [(risk, label) for i, risk in enumerate(S.RISKS)
                 for label in model_counts if model_counts[label][i] == 1]

    print(f"  human sample {len(human)} complete trajectories, cells "
          f"{[c.size for c in human_cells]} at risk {list(S.RISKS)}")
    print(f"  null: {N_NULL} draws of {COMPARISON_N} per risk cell, seed {SEED}")
    for i, risk in enumerate(S.RISKS):
        col = null_counts[:, i]
        print(f"    risk {risk}: distinct mean {col.mean():.2f}, min {col.min()}, "
              f"max {col.max()}")
    print(f"  human pooled 60: mean {pooled_distinct.mean():.2f}, "
          f"median {int(np.median(pooled_distinct))}, min {pooled_distinct.min()}, "
          f"max {pooled_distinct.max()}")
    for label in model_counts:
        print(f"    {label:>22}: per cell {model_counts[label]}, "
              f"pooled 60 {model_pooled[label]}")
    print(f"  {len(below)} of {n_cells} model cells below the human minimum; "
          f"{len(saturated)} cells at the measure's floor of 1; "
          f"{len(inside)} inside the null: "
          + ", ".join(f"{lab} at risk {r} (distinct {q}, {100 * p:.1f}th pct)"
                      for r, lab, q, p in inside))

    # One draw rather than a summary of draws, so panel b shows a composition
    # that actually occurred.  The median draw is the representative one and its
    # count is an integer, which a mean over draws is not.
    median_distinct = int(np.median(pooled_distinct))
    chosen = int(np.flatnonzero(pooled_distinct == median_distinct)[0])
    inverse = {v: k for k, v in vocabulary.items()}
    human_profile = composition([inverse[c] for c in pooled[chosen]])

    order = sorted(model_pooled, key=lambda lab: (-model_pooled[lab], lab))
    narrowest = order[-1]
    rows = [(ROUTE_TOP - i, label) for i, label in enumerate(order)]

    fig = plt.figure(figsize=(S.TEXT, 3.36))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.62, 1.0], wspace=0.40)
    left = gs[0, 0].subgridspec(1, 3, wspace=0.13)
    axes_a = [fig.add_subplot(left[i]) for i in range(3)]
    ax_b = fig.add_subplot(gs[0, 1])

    # --- a: where each model cell falls in the matched human null -------------
    exception = sorted({label for _, label, _, _ in inside})
    _, _, exception_short, exception_verdict, _ = identity(exception[0])
    for i, (ax, risk) in enumerate(zip(axes_a, S.RISKS)):
        notes = []
        if i == 0:
            notes.append(("1 = all 20 identical,\nthe floor of the measure",
                          0.6, NOTE_Y, S.MUTED, "left"))
        if i == 1:
            notes.append(("no human draw\never landed here", 1.2, 6.56,
                          S.UNSAFE_C, "left"))
            notes.append(("20 = all 20 different,\n"
                          "the measure's ceiling",
                          0.6, NOTE_Y, S.MUTED, "left"))
        if i == 2:
            notes.append((f"only {exception_short} {VERDICT_MARK[exception_verdict]}\n"
                          "lands inside", COMPARISON_N + 0.4, NOTE_Y,
                          S.INK_2, "right"))
        draw_facet(ax, null_counts[:, i],
                   {label: model_counts[label][i] for label in order},
                   rows, risk=risk, show_ylabels=(i == 0), show_xlabel=(i == 1),
                   notes=notes)
    S.direct_label(axes_a[0], 1.3, HUMAN_BASE + 0.10,
                   f"{N_NULL:,} draws\nof 20 humans", color=S.INK_2,
                   ha="left", va="bottom", dx=0, dy=0)
    S.panel(axes_a[0], "a",
            f"{len(below)} of {n_cells} model cells sit below every human draw",
            pad=16)

    # --- b: the mechanism, which sequences each population spends its 60 on ---
    for y, name in [(HUMAN_TICK, "Human")] + rows:
        if name == "Human":
            profile, colour, marker = human_profile, HUMAN_C, S.ROUTE_M["human"]
            count = median_distinct
        else:
            colour, marker, _, _, _ = identity(name)
            profile = composition([k for cell in model_cells[name] for k in cell])
            count = model_pooled[name]
        offset = 0.0
        for j, share in enumerate(profile):
            ax_b.barh(y, share, left=offset, height=0.66, color=colour,
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
    narrow_profile = composition([k for cell in model_cells[narrowest] for k in cell])
    ax_b.annotate("all 20 identical in every risk cell",
                  xy=(0.02, ROUTE_TOP - len(order) + 1),
                  ha="left", va="center", fontsize=S.FS_NOTE, color=S.SURFACE,
                  zorder=6)

    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(Y_LO, Y_HI)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_xticklabels(["0", "25", "50", "75", "100"])
    ax_b.set_yticks([])
    ax_b.set_xlabel("share of the 60 matched trajectories (%)", labelpad=2)
    S.strip(ax_b, left=False, grid_axis="x")
    ax_b.annotate("sequences\nused", xy=(1.0, TOP_Y), xytext=(4, 0),
                  textcoords="offset points", ha="left", va="center",
                  fontsize=S.FS_NOTE, color=S.MUTED, linespacing=1.15,
                  annotation_clip=False)
    # The headline count is against a null of twenty participants, which matches
    # the model cells in size but not in independence: a model cell is ten races
    # replayed once per seat. Against a null built from ten complete human dyads,
    # which is the model's structure exactly, the count is fifteen. Both are
    # printed, because quoting only the larger one would be choosing the null
    # after seeing which number it gives.
    ax_b.annotate(VERDICT_KEY
                  + f"\nthe human row is the median of the {N_NULL:,} matched draws"
                    "\nGPT-5.4, GPT-5.4 mini and GPT-5.5 have no cell in "
                    "this pilot export:\nabsent, not concentrated"
                    "\n18 is against a null of 20 participants; against one of 10 "
                    "complete\ndyads, which matches the model cells' independence, "
                    "it is 15 of 21",
                  xy=(0.0, 0.0), xycoords="axes fraction", xytext=(-40, -26),
                  textcoords="offset points", ha="left", va="top",
                  fontsize=S.FS_NOTE, color=S.MUTED, linespacing=1.35,
                  annotation_clip=False)
    S.panel(ax_b, "b",
            f"{median_distinct} sequences for 60 humans, "
            f"{model_pooled[narrowest]} for {S.ROUTE_SHORT[LABEL_TO_ROUTE[narrowest]]}",
            pad=4)

    S.save(fig, "human_versus_model", width=S.TEXT)


if __name__ == "__main__":
    main()
