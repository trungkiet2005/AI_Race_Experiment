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
  b  Hill q=1 effective diversity, with a horizontal interval for each route and
     risk condition.  The human row is the matched reference interval, while
     route colours identify the endpoint and marker shapes identify risk.
  c  Mean pairwise Hamming distance over the ten action bits.  This second
     measure separates how many trajectories are used from how far apart they
     are, so the two diversity panels do not repeat the same statistic.

The all-route supplemental output preserves the former composition barcode and
modal-path fingerprint.  It is generated from the same trajectory keys, but it
is not the main paper's diversity comparison.

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
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402,F401
import analyze_trajectory_diversity_rarefaction as TD  # noqa: E402
import analyze_trajectory_diversity_confirmatory as CONF  # noqa: E402

N_NULL = 20_000
SEED = 20260910
COMPARISON_N = 20
DYAD_N = 10
ROOT = Path(__file__).resolve().parents[2]
DIVERSITY_CSV = (ROOT / "results" / "derived" / "trajectory_diversity_confirmatory"
                  / "trajectory_diversity_confirmatory.csv")
RISK_MARKERS = {0.1: "o", 0.6: "s", 0.9: "^"}
RISK_OFFSETS = {0.1: 0.16, 0.6: 0.0, 0.9: -0.16}

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
# Joint-action colours for the compact game-theoretic path fingerprint.  The
# cells are outcomes of one round, not inferred strategies: the two bits are
# the focal route's action followed by its opponent's action.
PAIR_C = {
    "00": "#dcefe4",  # Safe / Safe
    "01": "#f2dfb2",  # Safe / Unsafe
    "10": "#e9b38f",  # Unsafe / Safe
    "11": S.UNSAFE_C,  # Unsafe / Unsafe
}
PAIR_LABEL = {"00": "S/S", "01": "S/U", "10": "U/S", "11": "U/U"}

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
        return 1.80 + 0.28 * (self.y_hi - Y_LO)


def identity(label):
    """Colour, marker, short name and verdict for a checkpoint label."""
    route = LABEL_TO_ROUTE[label]
    verdict = route in S.ADMITTED
    return S.ROUTE_C[route], S.ROUTE_M[route], S.ROUTE_SHORT[route], verdict


def load_diversity_artifact() -> pd.DataFrame:
    """Load the already-derived q=1 and Hamming values without recomputing them.

    The q=0 panel above is deliberately tied to the trajectory keys used by
    this script because it also builds the matched human null.  Panels b and c
    read the confirmatory analysis export directly, so a presentation edit
    cannot change a diversity estimate or its cluster-bootstrap interval.
    """
    if not DIVERSITY_CSV.is_file():
        raise SystemExit(f"missing verified diversity artifact: {DIVERSITY_CSV}")

    table = pd.read_csv(DIVERSITY_CSV, encoding="utf-8-sig")
    required = {
        "risk_cap", "population", "source_n", "comparison_n", "q1_mean",
        "q1_ci_low", "q1_ci_high", "mean_pairwise_hamming",
        "hamming_ci_low", "hamming_ci_high",
    }
    missing = sorted(required - set(table.columns))
    if missing:
        raise SystemExit(f"diversity artifact lacks required columns: {missing}")

    table = table.copy()
    table["risk_cap"] = pd.to_numeric(table["risk_cap"], errors="raise")
    numeric = [
        "source_n", "comparison_n", "q1_mean", "q1_ci_low", "q1_ci_high",
        "mean_pairwise_hamming", "hamming_ci_low", "hamming_ci_high",
    ]
    for column in numeric:
        table[column] = pd.to_numeric(table[column], errors="raise")
        if not np.isfinite(table[column].to_numpy(float)).all():
            raise SystemExit(f"non-finite values in diversity artifact column {column!r}")

    expected_populations = {"Human"} | {
        S.ROUTE_LABEL[route] for route in S.ROUTE_ORDER
    }
    expected_keys = {
        (population, risk) for population in expected_populations
        for risk in S.RISKS
    }
    observed_keys = set(zip(table["population"], table["risk_cap"]))
    if len(observed_keys) != len(table) or observed_keys != expected_keys:
        raise SystemExit(
            "diversity artifact does not have exactly one row for every audited "
            f"population and risk cell: observed {len(observed_keys)}, "
            f"expected {len(expected_keys)}"
        )

    model_rows = table[table["population"] != "Human"]
    if not (model_rows["comparison_n"] == COMPARISON_N).all():
        raise SystemExit("model diversity cells are not matched to 20 trajectories")
    if not (model_rows["source_n"] == COMPARISON_N).all():
        raise SystemExit("model diversity cells do not have 20 source trajectories")
    return table


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


def modal_path(keys) -> tuple[str, int]:
    """The most frequent five-round joint-action path in one pooled sample."""
    counts = Counter(keys)
    return min(counts, key=lambda key: (-counts[key], key)), max(counts.values())


def paired_states(key: str) -> list[str]:
    """Decode the analyser's focal|opponent ten-bit trajectory encoding."""
    focal, opponent = key.split("|")
    if len(focal) != 5 or len(opponent) != 5:
        raise ValueError(f"not a complete five-round trajectory: {key!r}")
    return [a + b for a, b in zip(focal, opponent)]


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
        ax.set_xlabel("distinct paired first-five-round trajectories among 20", labelpad=2)


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
    human_keys = [inverse[c] for c in pooled[chosen]]
    human_profile = composition(human_keys)
    model_keys = {
        label: [k for cell in cells for k in cell]
        for label, cells in model_cells.items()
    }
    diversity_table = load_diversity_artifact()

    order_all = sorted(model_pooled, key=lambda lab: (-model_pooled[lab], lab))

    shared = dict(
        null_counts=null_counts, model_counts=model_counts, model_cells=model_cells,
        model_pooled=model_pooled, human_profile=human_profile,
        median_distinct=median_distinct, below_dyad=below_dyad,
        inside_labels=inside_labels, human_keys=human_keys, model_keys=model_keys,
        diversity_table=diversity_table,
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
    # Keep the previous composition barcode and modal-path fingerprint in the
    # all-route output for the supplement.  The main output now uses both
    # verified diversity measures instead of repeating the same occupancy view.
    draw_path_figure("human_versus_model_all_routes", order_all, marks=True, **shared)


def diversity_cell(table: pd.DataFrame, population: str, risk: float) -> pd.Series:
    """Return one verified route-by-risk row from the diversity artifact."""
    cell = table[(table["population"] == population)
                 & np.isclose(table["risk_cap"], risk)]
    if len(cell) != 1:
        raise RuntimeError(
            f"expected one diversity row for {population!r} at risk {risk}, "
            f"found {len(cell)}"
        )
    return cell.iloc[0]


def population_label(label: str, *, marks: bool) -> str:
    if label == "Human":
        return "Humans"
    short = identity(label)[2]
    return f"{short} {VERDICT_MARK[identity(label)[3]]}" if marks else short


def draw_metric_panel(ax, table, order, rows, frame, *, value_key, low_key,
                      high_key, xlim, xticks, xlabel, claim, letter, marks):
    """Draw one metric with risk as marker shape and route as colour."""
    entries = [(frame.human_tick, "Human")] + rows
    human_y = frame.human_tick

    for risk in S.RISKS:
        offset = RISK_OFFSETS[risk]
        human = diversity_cell(table, "Human", risk)
        human_mean = float(human[value_key])
        human_low = float(human[low_key])
        human_high = float(human[high_key])
        y = human_y + offset
        # The human reference is a heavier interval with an open marker. It is
        # the comparison band for every coloured route on the same risk row.
        ax.plot([human_low, human_high], [y, y], color=HUMAN_C, lw=2.4,
                solid_capstyle="round", zorder=2)
        ax.plot([human_low, human_low], [y - 0.07, y + 0.07], color=HUMAN_C,
                lw=0.8, zorder=2)
        ax.plot([human_high, human_high], [y - 0.07, y + 0.07], color=HUMAN_C,
                lw=0.8, zorder=2)
        S.dot(ax, human_mean, y, color=HUMAN_C, marker=RISK_MARKERS[risk],
              size=22, filled=False, ring=False, lw=1.0, zorder=4)

        for route_y, population in rows:
            row = diversity_cell(table, population, risk)
            mean = float(row[value_key])
            low = float(row[low_key])
            high = float(row[high_key])
            colour = identity(population)[0]
            if high > low:
                ax.errorbar(
                    mean, route_y + offset,
                    xerr=np.array([[max(0.0, mean - low)],
                                   [max(0.0, high - mean)]]),
                    fmt="none", ecolor=colour, elinewidth=1.0, capsize=2.2,
                    zorder=2,
                )
            S.dot(ax, mean, route_y + offset, color=colour,
                  marker=RISK_MARKERS[risk], size=18, zorder=4)

    y_ticks = [y for y, _ in entries]
    ax.set_yticks(y_ticks, [population_label(name, marks=marks)
                            for _, name in entries])
    for tick, (_, name) in zip(ax.get_yticklabels(), entries):
        tick.set_color(HUMAN_C if name == "Human" else identity(name)[0])
        tick.set_fontweight("bold")
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_xlim(*xlim)
    ax.set_ylim(Y_LO, frame.y_hi)
    ax.set_xticks(xticks)
    ax.set_xlabel(xlabel, labelpad=2)
    S.strip(ax, left=False, grid_axis="x")
    S.panel(ax, letter, claim, pad=5)


def risk_handles():
    return [
        Line2D([], [], marker=RISK_MARKERS[risk], color=HUMAN_C,
               markerfacecolor=S.SURFACE, markeredgecolor=HUMAN_C,
               markeredgewidth=0.8, lw=0, ms=4.0,
               label=fr"$p_r^{{\max}}={S.RISK_LABEL[risk]}$")
        for risk in S.RISKS
    ]


def draw_figure(stem, order, *, marks, null_counts, model_counts, model_cells,
                model_pooled, human_profile, median_distinct, below_dyad,
                inside_labels, human_keys, model_keys, diversity_table):
    """Main diversity figure: q=0 null, q=1, and Hamming distance."""
    del model_cells, model_pooled, human_profile, median_distinct, below_dyad
    del inside_labels, human_keys, model_keys

    frame = Frame(len(order))
    rows = frame.rows(order)
    floors = null_counts.min(axis=0)
    below = [(risk, label) for i, risk in enumerate(S.RISKS)
             for label in order if model_counts[label][i] < floors[i]]
    n_cells = len(order) * len(S.RISKS)

    fig = plt.figure(figsize=(S.TEXT, frame.height_in))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.52, 1.18], wspace=0.42)
    left = gs[0, 0].subgridspec(1, 3, wspace=0.13)
    axes_a = [fig.add_subplot(left[i]) for i in range(3)]
    right = gs[0, 1].subgridspec(2, 1, height_ratios=[1.0, 1.0], hspace=0.90)
    ax_b = fig.add_subplot(right[0, 0])
    ax_c = fig.add_subplot(right[1, 0])

    # --- a: q=0 distinct first-five-round trajectories ----------------------
    for i, (ax, risk) in enumerate(zip(axes_a, S.RISKS)):
        draw_facet(ax, null_counts[:, i],
                   {label: model_counts[label][i] for label in order},
                   rows, risk=risk, show_ylabels=(i == 0), show_xlabel=(i == 1),
                   notes=[], frame=frame, marks=marks)
    S.direct_label(axes_a[0], 1.3, frame.human_base + 0.10,
                   f"{N_NULL:,} draws\nof 20 humans", color=S.INK_2,
                   ha="left", va="bottom", dx=0, dy=0)
    S.panel(axes_a[0], "a",
            f"{len(below)} of {n_cells} model cells sit below every human draw",
            pad=16)

    # --- b/c: verified diversity statistics on the same population rows -----
    draw_metric_panel(
        ax_b, diversity_table, order, rows, frame,
        value_key="q1_mean", low_key="q1_ci_low", high_key="q1_ci_high",
        xlim=(0.0, 21.0), xticks=[0, 5, 10, 15, 20],
        xlabel="effective trajectories (Hill $q{=}1$)",
        claim="effective diversity contracts", letter="b", marks=marks,
    )
    draw_metric_panel(
        ax_c, diversity_table, order, rows, frame,
        value_key="mean_pairwise_hamming", low_key="hamming_ci_low",
        high_key="hamming_ci_high", xlim=(0.0, 0.60),
        xticks=[0.0, 0.2, 0.4, 0.6],
        xlabel="mean pairwise Hamming distance (10 bits)",
        claim="the same compression spans action distance", letter="c", marks=marks,
    )
    # The inter-panel gap is a dedicated legend lane. Keeping the key outside
    # both axes makes the human interval and every route marker unobstructed.
    fig.legend(handles=risk_handles(), loc="center",
               bbox_to_anchor=(0.82, 0.535), ncol=3, frameon=False,
               fontsize=S.FS_NOTE, handletextpad=0.35,
               columnspacing=0.8, borderaxespad=0.0)

    if marks:
        fig.text(0.105, 0.025, VERDICT_KEY, ha="left", va="bottom",
                 fontsize=S.FS_NOTE, color=S.MUTED)

    if below and len(below) != n_cells:
        print(f"  {stem}: {n_cells - len(below)} of {n_cells} drawn cells are NOT "
              "below the human minimum")
    S.save(fig, stem, width=S.TEXT)


def draw_path_figure(stem, order, *, marks, model_cells, model_pooled,
                     human_profile, median_distinct, human_keys, model_keys,
                     **_):
    """Preserve the former composition and modal-path view for the supplement."""
    frame = Frame(len(order))
    rows = frame.rows(order)
    narrowest = order[-1]

    fig = plt.figure(figsize=(S.TEXT, frame.height_in))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.18, 1.0], wspace=0.42)
    ax_b, ax_c = [fig.add_subplot(gs[0, i]) for i in range(2)]

    # --- b: composition barcode of the observed trajectory space -------------
    for y, name in [(frame.human_tick, "Human")] + rows:
        if name == "Human":
            profile, colour = human_profile, HUMAN_C
            count = median_distinct
        else:
            colour = identity(name)[0]
            profile = composition([k for cell in model_cells[name] for k in cell])
            count = model_pooled[name]
        offset = 0.0
        for j, share in enumerate(profile):
            ax_b.barh(y, share, left=offset, height=BAR_H, color=colour,
                      alpha=1.0 if j % 2 == 0 else 0.62, edgecolor=S.SURFACE,
                      linewidth=0.45, zorder=3)
            offset += share
        S.direct_label(ax_b, 1.0, y, f"{count}/60", color=colour, dx=4,
                       weight="bold" if name in ("Human", narrowest) else "normal")

    ax_b.set_xlim(0.0, 1.0)
    ax_b.set_ylim(-0.42, frame.human_tick + 0.42)
    ax_b.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_b.set_xticklabels(["0", "25", "50", "75", "100"])
    labels_b = [population_label("Human", marks=marks)] + [
        population_label(name, marks=marks) for _, name in rows
    ]
    ax_b.set_yticks([frame.human_tick] + [y for y, _ in rows], labels_b)
    for tick, name in zip(ax_b.get_yticklabels(), ["Human"] + [name for _, name in rows]):
        tick.set_color(HUMAN_C if name == "Human" else identity(name)[0])
        tick.set_fontweight("bold")
    ax_b.tick_params(axis="y", length=0, pad=3)
    ax_b.set_xlabel("share of 60 matched trajectories (%)", labelpad=2)
    S.strip(ax_b, grid_axis="x")
    S.panel(ax_b, "b", "routes reuse a small set of paths", pad=5)

    # --- c: compact game-theoretic fingerprint of the modal path -------------
    path_rows = [(frame.human_tick, "Human")] + rows
    population_names = ["Human"] + [name for _, name in rows]
    for y, name in path_rows:
        keys = human_keys if name == "Human" else model_keys[name]
        path, n_mode = modal_path(keys)
        for round_idx, pair in enumerate(paired_states(path)):
            face = PAIR_C[pair]
            ink = S.SURFACE if pair == "11" else S.INK
            ax_c.add_patch(Rectangle((round_idx - 0.46, y - 0.25), 0.92, 0.50,
                                     facecolor=face, edgecolor=S.SURFACE,
                                     linewidth=0.8, zorder=3))
            ax_c.text(round_idx, y, PAIR_LABEL[pair], ha="center", va="center",
                      fontsize=6.1, color=ink, fontweight="bold", zorder=4)
        colour = HUMAN_C if name == "Human" else identity(name)[0]
        ax_c.text(5.03, y, f"{n_mode}/60", ha="left", va="center",
                  fontsize=S.FS_NOTE, color=colour, fontweight="bold")

    ax_c.set_yticks([frame.human_tick] + [y for y, _ in rows],
                    [population_label(name, marks=marks) for name in population_names])
    for tick, name in zip(ax_c.get_yticklabels(), population_names):
        tick.set_color(HUMAN_C if name == "Human" else identity(name)[0])
        tick.set_fontweight("bold")
    ax_c.set_xlim(-0.50, 5.72)
    ax_c.set_ylim(-0.42, frame.human_tick + 0.42)
    ax_c.set_xticks(range(5), ["1", "2", "3", "4", "5"])
    ax_c.set_xlabel("round", labelpad=2)
    ax_c.tick_params(axis="y", length=0, pad=3)
    ax_c.tick_params(axis="x", length=2)
    for side in ("top", "right", "left"):
        ax_c.spines[side].set_visible(False)
    ax_c.spines["bottom"].set_color(S.HAIRLINE)
    S.panel(ax_c, "c", "the modal path is a joint-action sequence", pad=5)
    handles = [Patch(facecolor=PAIR_C[pair], edgecolor=S.HAIRLINE,
                     label=PAIR_LABEL[pair]) for pair in ("00", "01", "10", "11")]
    ax_c.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.0, -0.34),
               ncol=4, frameon=False, fontsize=6.1,
               handlelength=0.8, handleheight=0.8, columnspacing=0.55,
               handletextpad=0.22, borderaxespad=0.0)
    if marks:
        fig.text(0.105, 0.025, VERDICT_KEY, ha="left", va="bottom",
                 fontsize=S.FS_NOTE, color=S.MUTED)
    S.save(fig, stem, width=S.TEXT)


if __name__ == "__main__":
    main()
