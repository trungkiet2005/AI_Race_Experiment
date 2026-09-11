"""What the audited route does against a rival it cannot influence.

Every gameplay result before this one is self-play: both companies in a race are
the same endpoint, so "the route went unsafe after its rival did" and "the route
was in an unsafe phase of its own" are the same sentence, and no amount of
stratification separates them.  This campaign breaks the symmetry.  The audited
route, google/gemini-3-flash-preview, plays against the four reduced strategies
the paper's evolutionary lane is built on, executed by the task file rather than
by a model: Always Safe, Always Unsafe, Conditional Safe and Conditional Unsafe.
The rival's strategy is now exogenous: the two unconditional rivals ignore the
route entirely and the two conditional ones answer only the route's own last
move, so nothing the rival does carries a second model's private state.  The
route is never told the rival is scripted.

The claim.  The rival's strategy moves this route several times further than the
stated private risk does, and it moves it by mirroring: the route keeps most of
its restraint against a rival that stays safe, and abandons almost all of it
against a rival that never does.  What the stated risk was meant to govern, the
rival governs instead.

Panels
  a  The twelve cells, rival down, stated risk across.  Read down a column and
     the route swings across most of the scale; read across a row and it barely
     moves.  Two cells sit at exactly 100 per cent and are ringed, because a
     cell in which all 93 decisions were unsafe is a measurement with no room
     above it rather than a precise one, and every contrast through it is
     truncated rather than complete.  A row name is the strategy the task file
     executed and not a summary of what the rival did, and the two differ:
     because the route opened Unsafe in every race, Conditional Safe plays Safe
     on round one and mirrors from round two, and 55 per cent of its own moves
     in this campaign were unsafe.  An earlier draft tinted the row labels with
     the paper's Safe and Unsafe inks by the rival's opening move, unkeyed,
     which put the Safe ink on that rival; the tint is gone and the fact it hid
     is printed under the panel instead.
  b  The headline contrast, retaliation minus exploitation, differenced inside a
     repetition.  The game seed is a base plus the repetition index and names
     neither the strategy nor the risk, so one repetition is one horizon
     stopping-draw stream in every cell, and differencing inside it removes the
     horizon from the comparison.  The second series is the opening-move
     contrast: Conditional Unsafe and Conditional Safe are the same strategy
     from round two onward, so their difference isolates the rival's very first
     move.  What sits beside that move is sampling noise, which the two arms do
     not share and which the interval is there to bound.
  c  The comparison that reframes the rest of the paper.  Against Always Safe
     the route's rate falls monotonically with stated risk; the same route in
     the neutral self-play baseline sits far above that at every risk level.
     The gap between the two designs is printed at each risk, and it is printed
     with a ``>=`` wherever the self-play arm is against the ceiling: at risk
     0.1 nine of that arm's ten races have every decision unsafe, so 98.9% is a
     rate with no room above it and the 74 pp it yields is a lower bound on the
     distance rather than a measurement of it.  The two lower panels of this
     figure already ring and name that boundary, and a bare number in the third
     would have been the one place the reader was not told.

What this does NOT show.  One route, one game, one prompt version, ten races per
cell.  Nothing here is a claim about models in general, and the four scripted
rivals are reduced strategies rather than a sample of anything.  Panel c is a
contrast between two designs, not a decomposition: the self-play number is what
two copies of this route settle at when each is also the other's provocation,
and the distance to the Always Safe column says how much of that self-play rate
needed a rival that escalated back.  It does not license the claim that a
self-play rate is an artefact, nor that the Always Safe column is the route's
"true" rate; both are behaviour, under different rivals.  Panel b's second
series is about the rival's opening move only, never the route's: the route
opened Unsafe in all 120 races of this campaign, whatever the rival was about to
do, so it has no opening-move contrast of its own to show.
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402

ROUTE = "google/gemini-3-flash-preview"
CAMPAIGN = D.ROOT / "results" / "frontier" / "scripted_opponent_campaign"
PROTOCOL = "ai-race-scripted-opponent-v1"
DERIVED = (
    D.ROOT / "results" / "derived" / "scripted_opponent_campaign"
    / "scripted_opponent_rates.json"
)

# Ascending in what the route does to them, so the surface reads as a gradient
# and the two conditional rivals bracket the two unconditional ones.
ORDER = ["AS", "CS", "CAS", "AU"]
LABEL = {
    "AS": "Always Safe",
    "CS": "Cond. Safe",
    "CAS": "Cond. Unsafe",
    "AU": "Always Unsafe",
}
# The rival's opening move is the one thing the four strategies differ in on
# round one, and panel b's second series is built from it.  It is deliberately
# NOT a colour channel on the row labels: the two conditional rivals mirror the
# route from round two, the route opened Unsafe in all 120 races, and a green
# "Cond. Safe" label would announce a safe rival that played unsafe on most of
# its own moves.  The names already carry the opening move in words.

N_BOOT = 5000
SEED = 20260910

RETAL_C = S.UNSAFE_C
OPEN_C = S.INK_2

# Light to dark, so the tile-text flip in ``heat_tiles`` puts white type on the
# dark tiles and ink on the pale ones.  A perceptually reversed map such as
# viridis would put white type on its brightest yellow cell, which is exactly
# the 100% cell this figure most needs to stay readable.
UNSAFE_MAP = LinearSegmentedColormap.from_list(
    "unsafe_seq",
    ["#f6eeec", "#eec4b8", "#dd8a72", "#c0392b", S.UNSAFE_C, "#6d0f1c"],
)


def cell_rng(*key) -> np.random.Generator:
    """A generator fixed by the cell, so one interval never depends on another."""
    digest = hashlib.sha256("|".join(str(part) for part in key).encode("utf-8")).digest()
    return np.random.default_rng([SEED, int.from_bytes(digest[:8], "big")])


def load_campaign() -> pd.DataFrame:
    """Every recorded turn of the campaign, route decisions still flagged.

    Fail-closed for the same reason ``figdata`` is, with one check the other
    campaigns cannot need: the rival here is code rather than a model, so it
    cannot be audited by reading a response, and code that quietly played the
    wrong strategy would leave a table that looks perfectly healthy while
    answering a different question.
    """
    frames = []
    for receipt_path in sorted(CAMPAIGN.glob("*/*/collection_receipt.json")):
        manifest = json.loads(
            receipt_path.with_name("run_manifest.json").read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != PROTOCOL:
            raise ValueError(f"{receipt_path} is protocol {manifest.get('protocol_id')!r}")
        if manifest.get("status") != "completed":
            raise ValueError(f"{receipt_path} did not complete")
        if manifest.get("model_route") != ROUTE:
            raise ValueError(f"{receipt_path} is route {manifest.get('model_route')!r}")
        paths = list(receipt_path.parent.rglob("turns.jsonl"))
        if len(paths) != 1:
            raise ValueError(f"{receipt_path}: {len(paths)} turns files")
        turns = pd.DataFrame(
            [json.loads(line) for line in paths[0].read_text(encoding="utf-8").splitlines()]
        )
        if turns["parse_failed"].any():
            raise ValueError(f"{paths[0]} carries a parse failure")
        strategy = receipt["cell"]["strategy"]
        if set(turns["opponent_strategy"]) != {strategy}:
            raise ValueError(
                f"{paths[0]} carries strategies {sorted(set(turns['opponent_strategy']))}")
        deviations = replay_rival(turns, strategy)
        if deviations:
            raise ValueError(f"{paths[0]}: the scripted rival deviated in {deviations} rounds")
        seats = turns.groupby("game_id")["route_seat"].first().value_counts().to_dict()
        if sorted(seats.values()) != [5, 5]:
            raise ValueError(f"{paths[0]}: seat counterbalance is {seats}")
        turns["strategy"] = strategy
        turns["cell_risk"] = float(receipt["cell"]["max_private_risk"])
        frames.append(turns)
    if not frames:
        raise ValueError(f"no cells under {CAMPAIGN}")
    return pd.concat(frames, ignore_index=True)


def replay_rival(turns: pd.DataFrame, strategy: str) -> int:
    """Recompute every rival move from the route's own history, count deviations."""
    deviations = 0
    for _, race in turns.groupby("game_id"):
        route_moves = list(
            race[race["is_route_decision"]].sort_values("round")["action"].str.lower())
        for _, row in race[~race["is_route_decision"]].sort_values("round").iterrows():
            rnd = int(row["round"])
            if strategy == "AS":
                expected = "safe"
            elif strategy == "AU":
                expected = "unsafe"
            elif rnd == 1:
                expected = "safe" if strategy == "CS" else "unsafe"
            else:
                expected = route_moves[rnd - 2]
            deviations += str(row["action"]).lower() != expected
    return deviations


def paired_contrast(route: pd.DataFrame, risk: float, left: str, right: str):
    """left minus right within a repetition, bootstrapped over repetitions.

    The repetition, not the decision, is the independent unit here because the
    two cells being differenced share its horizon draw; resampling decisions
    would break the pairing the design was built to exploit and would report an
    interval far too narrow.
    """
    per = (
        route[route["cell_risk"] == risk]
        .groupby(["strategy", "rep"])
        .agg(unsafe=("unsafe", "sum"), decisions=("unsafe", "size"), seed=("game_seed", "first"))
    )
    reps = sorted(set(per.loc[left].index) & set(per.loc[right].index))
    if any(per.loc[(left, rep), "seed"] != per.loc[(right, rep), "seed"] for rep in reps):
        raise ValueError(f"seed pairing broken for {left} against {right} at risk {risk}")
    values = np.array(
        [
            per.loc[(left, rep), "unsafe"] / per.loc[(left, rep), "decisions"]
            - per.loc[(right, rep), "unsafe"] / per.loc[(right, rep), "decisions"]
            for rep in reps
        ],
        dtype=float,
    )
    rng = cell_rng("contrast", left, right, risk)
    draws = rng.integers(0, values.size, size=(N_BOOT, values.size))
    means = values[draws].mean(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        int(values.size),
    )


def clustered_rate(block: pd.DataFrame, *key):
    """Unsafe rate with a race-clustered percentile interval."""
    per_race = block.groupby("game_id")["unsafe"].agg(["sum", "size"])
    unsafe = per_race["sum"].to_numpy(dtype=float)
    total = per_race["size"].to_numpy(dtype=float)
    rng = cell_rng(*key)
    draws = rng.integers(0, unsafe.size, size=(N_BOOT, unsafe.size))
    means = unsafe[draws].sum(axis=1) / total[draws].sum(axis=1)
    return (
        float(unsafe.sum() / total.sum()),
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
        int(unsafe.size),
        int(total.sum()),
    )


def draw_surface(fig, ax, surface, spread_col, spread_row, rival_unsafe):
    arr = 100 * surface.to_numpy(dtype=float)
    im = S.heat_tiles(
        ax, arr,
        [LABEL[s] for s in ORDER],
        [S.RISK_LABEL[r] for r in S.RISKS],
        cmap=UNSAFE_MAP, vmin=0.0, vmax=100.0, fmt="{:.0f}",
    )
    # Both boundary cells are the darkest tiles on the map, so the ring has to
    # be the surface colour or it vanishes into the tile it is marking.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] >= 100.0 - 1e-9:
                S.highlight(ax, i, j, color=S.SURFACE, lw=1.2, inset=0.40)
    ax.set_xlabel(r"maximum private risk $p_r^{\max}$", labelpad=2)
    ax.set_ylabel("the scripted rival", labelpad=3)
    S.panel(ax, "a", "the rival, not the stated risk")
    # The bar is redundant as a scale, since every tile prints its own value.
    # It is here to name the quantity: without it a reader has no way to know
    # that the 100 in the Always Unsafe row is the ROUTE's rate and not the
    # rival's, which in that row happens to be 100 as well.
    bar = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.04, ticks=[0, 50, 100])
    bar.set_label("the route's unsafe play (%)", fontsize=S.FS_NOTE, labelpad=2)
    bar.ax.tick_params(labelsize=S.FS_NOTE, length=1.6)
    bar.outline.set_visible(False)
    # The last two lines are the key the row names do not carry.  A reader who
    # takes "Cond. Safe" for a rival that plays Safe is reading the strategy's
    # name rather than its behaviour, and the number is what settles it.
    for line, offset, colour in (
        (f"down a column {spread_col:.0f} pp, across a row {spread_row:.0f} pp", -27, S.INK_2),
        ("ringed cells sit on the 100% boundary", -36, S.MUTED),
        ("Cond. Safe opens Safe, then mirrors:", -47, S.MUTED),
        (f"{100 * rival_unsafe['CS']:.0f}% of its moves were Unsafe"
         f" (Cond. Unsafe {100 * rival_unsafe['CAS']:.0f}%)", -56, S.MUTED),
    ):
        ax.annotate(line, xy=(0.5, 0.0), xycoords="axes fraction",
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va="top", fontsize=S.FS_NOTE, color=colour,
                    annotation_clip=False)


def draw_contrasts(ax, retal, opening):
    ypos = {risk: y for risk, y in zip(S.RISKS, (2, 1, 0))}
    for series, colour, marker, offset in (
        (retal, RETAL_C, "o", 0.16), (opening, OPEN_C, "s", -0.16)
    ):
        for risk, (point, lo, hi, _n) in series.items():
            y = ypos[risk] + offset
            ax.plot([100 * lo, 100 * hi], [y, y], lw=1.1, color=colour,
                    solid_capstyle="round", zorder=3)
            S.dot(ax, 100 * point, y, color=colour, marker=marker, size=15)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([S.RISK_LABEL[r] for r in ypos])
    # The key sits in a band above the data rather than beside it: the two
    # series are fifty points apart on x, so a label at either series' own x
    # lands on top of the other one.  Each label carries its own marker, which
    # is what ties it to a series without a legend box or a colour swatch.
    ax.set_ylim(-0.75, 3.35)
    ax.set_xlim(-8, 92)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("difference in unsafe play, percentage points")
    ax.set_ylabel(r"$p_r^{\max}$", labelpad=1)
    S.strip(ax, grid_axis=None)
    # The key band is inside the axes, so an axes-wide grid and an axes-wide
    # zero rule both run through the key's own words.  Both stop at the top of
    # the data instead, which leaves the key on clean paper.
    band = (-0.75, 2.40)
    for value in (25, 50, 75):
        ax.vlines(value, *band, color=S.GRID, lw=0.5, zorder=0)
    ax.vlines(0.0, *band, color=S.MUTED, lw=0.8, zorder=1)
    S.panel(ax, "b", "it retaliates, at every risk")
    for y, colour, marker, text in (
        (3.00, RETAL_C, "o", "vs Always Unsafe minus vs Always Safe"),
        (2.48, OPEN_C, "s", "rival opened Unsafe minus Safe"),
    ):
        S.dot(ax, -4.5, y, color=colour, marker=marker, size=15)
        S.direct_label(ax, -4.5, y, text, color=colour, dx=6, weight="bold")
    # Two of the four cells these contrasts are built from are the ones panel a
    # rings, so the reader is told here rather than left to carry it across.
    for line, offset in (("at 0.1 both contrasts run into the", -24),
                         ("100% boundary and are truncated", -33)):
        ax.annotate(line, xy=(0.5, 0.0), xycoords="axes fraction",
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED,
                    annotation_clip=False)


def draw_designs(ax, scripted, selfplay, censored):
    """Two designs, one route, so colour separates the design and not the route.

    Every other figure in the paper spends colour on route identity.  Here there
    is one route and two rivals, so the safe rival takes the paper's Safe ink
    and the self-play line takes plain ink rather than a second green nobody
    could tell apart from the first.
    """
    risks = list(S.RISKS)
    for series, colour, marker, name in (
        (selfplay, S.INK, S.ROUTE_M[ROUTE], "self-play"),
        (scripted, S.SAFE_C, "o", "against\nAlways Safe"),
    ):
        y = [100 * series[r][0] for r in risks]
        ax.plot(risks, y, lw=1.2, color=colour, zorder=3)
        for risk in risks:
            point, lo, hi, _races, _dec = series[risk]
            ax.plot([risk, risk], [100 * lo, 100 * hi], lw=1.0, color=colour,
                    solid_capstyle="round", zorder=3)
            S.dot(ax, risk, 100 * point, color=colour, marker=marker, size=16)
        S.direct_label(ax, risks[-1], y[-1], name, color=colour, dx=5,
                       va="center", weight="bold")

    # A gap through an arm that is against the ceiling is a distance with no
    # room to grow, so it is printed as the lower bound it is.  The panel rings
    # nothing, so the mark has to live on the number itself.
    for risk in risks:
        top, bottom = 100 * selfplay[risk][0], 100 * scripted[risk][0]
        ax.annotate("", xy=(risk, top), xytext=(risk, bottom),
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=S.MUTED,
                                    linestyle=(0, (2, 1.6))), zorder=2)
        bound = "≥" if censored[risk][0] else ""
        ax.annotate(f"{bound}{top - bottom:.0f} pp", xy=(risk, (top + bottom) / 2),
                    xytext=(3, 0), textcoords="offset points",
                    ha="left", va="center", fontsize=S.FS_NOTE, color=S.MUTED)

    # The self-play interval at the lowest risk reaches 100, so the ceiling is
    # part of that estimate and has to be on the page.  Its label goes below the
    # rule and inboard: ``ceiling_rule`` puts one above the line at the right
    # margin, which is where this panel's claim sentence already is.
    S.ceiling_rule(ax, 100.0, label="")
    ax.annotate("every decision unsafe", xy=(0.30, 99.0), xytext=(0, 0),
                textcoords="offset points", ha="left", va="top",
                fontsize=S.FS_NOTE, color=S.MUTED)
    S.rate_axis(ax)
    S.risk_axis(ax)
    ax.set_xlim(0.02, 1.16)
    S.strip(ax)
    S.panel(ax, "c", "a safe rival draws out far less")
    # Which risks the mark applies to, and on what evidence, so the reader is
    # not asked to take the boundary on trust.  Panels a and b carry their own
    # boundary notes in the same place; this is the third.
    pinned = [r for r in risks if censored[r][0]]
    if pinned:
        where = "; ".join(
            f"{S.RISK_LABEL[r]} ({censored[r][1]} of {censored[r][2]} races)"
            for r in pinned
        )
        for line, offset in (
            ("the self-play arm is on the 100% boundary at", -24),
            (f"{where}, so ≥ marks a lower bound", -33),
        ):
            ax.annotate(line, xy=(0.5, 0.0), xycoords="axes fraction",
                        xytext=(0, offset), textcoords="offset points",
                        ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED,
                        annotation_clip=False)


def main() -> None:
    turns = load_campaign()
    route = turns[turns["is_route_decision"]].copy()
    script = turns[~turns["is_route_decision"]]
    print(f"  campaign: {len(turns)} recorded turns, {len(route)} of them the route's own "
          f"({len(script)} are the script), {route['game_id'].nunique()} races")

    surface = (
        route.groupby(["strategy", "cell_risk"])["unsafe"].mean()
        .unstack().reindex(index=ORDER, columns=list(S.RISKS))
    )
    counts = route.groupby(["strategy", "cell_risk"])["unsafe"].size().unstack()
    if counts.to_numpy().min() != counts.to_numpy().max():
        raise ValueError(f"cells are not balanced: {counts.to_dict()}")
    print(f"  every cell is {int(counts.to_numpy().max())} route decisions over 10 races")

    print("  route unsafe rate, per cell:")
    for strategy in ORDER:
        row = "  ".join(f"{100 * surface.loc[strategy, r]:5.1f}" for r in S.RISKS)
        spread = 100 * (surface.loc[strategy].max() - surface.loc[strategy].min())
        print(f"    {LABEL[strategy]:<14} {row}   spread across risk {spread:4.1f} pp")
    for risk in S.RISKS:
        column = surface[risk]
        print(f"    risk {risk}: spread across rivals "
              f"{100 * (column.max() - column.min()):4.1f} pp")
    # The panel's whole argument in one comparison, so it is computed and not
    # asserted, and the claim beside the panel is built from these two numbers.
    spread_col = float(100 * (surface.max(axis=0) - surface.min(axis=0)).median())
    spread_row = float(100 * (surface.max(axis=1) - surface.min(axis=1)).median())
    print(f"  median spread down a column {spread_col:.1f} pp, across a row {spread_row:.1f} pp "
          f"(ratio {spread_col / spread_row:.1f})")

    boundary = [(s, r) for s in ORDER for r in S.RISKS if surface.loc[s, r] >= 1.0 - 1e-12]
    print(f"  cells on the 100% boundary: {boundary}")

    retal = {r: paired_contrast(route, r, "AU", "AS") for r in S.RISKS}
    opening = {r: paired_contrast(route, r, "CAS", "CS") for r in S.RISKS}
    for name, series in (("retaliation - exploitation", retal),
                         ("rival opened Unsafe - Safe", opening)):
        for risk, (point, lo, hi, n) in series.items():
            print(f"  {name:<27} risk {risk}: {100 * point:+5.1f} pp "
                  f"[{100 * lo:+5.1f}, {100 * hi:+5.1f}] over {n} paired repetitions")

    scripted_as = {
        r: clustered_rate(route[(route["strategy"] == "AS") & (route["cell_risk"] == r)],
                          "rate", "AS", r)
        for r in S.RISKS
    }
    baseline = D.baseline_turns()
    own = baseline[baseline["model_route"] == ROUTE]
    selfplay = {r: clustered_rate(own[own["max_private_risk"] == r], "selfplay", r)
                for r in S.RISKS}
    for name, series in (("vs Always Safe", scripted_as), ("self-play", selfplay)):
        for risk, (point, lo, hi, races, decisions) in series.items():
            print(f"  {name:<15} risk {risk}: {100 * point:5.1f}% "
                  f"[{100 * lo:5.1f}, {100 * hi:5.1f}] over {races} races, {decisions} decisions")
    # A race in which every decision was unsafe cannot go higher, so an arm
    # holding such races is censored and every gap measured through it is a
    # lower bound.  Counted rather than asserted, and counted on the arm that
    # is on top: the Always Safe arm has no race at either boundary.
    censored = {}
    for risk in S.RISKS:
        per_race = (
            own[own["max_private_risk"] == risk]
            .groupby("game_id")["unsafe"].agg(["sum", "size"])
        )
        at_ceiling = int((per_race["sum"] == per_race["size"]).sum())
        censored[risk] = (at_ceiling > 0, at_ceiling, int(len(per_race)))
    for risk in S.RISKS:
        pinned, at_ceiling, races = censored[risk]
        print(f"  design gap at risk {risk}: "
              f"{'>=' if pinned else '  '}"
              f"{100 * (selfplay[risk][0] - scripted_as[risk][0]):5.1f} pp "
              f"(self-play races with every decision unsafe: {at_ceiling}/{races})")

    # What the row names in panel a do not say.  The rival is code, so this is
    # a property of the campaign and not an estimate.
    rival_unsafe = script.groupby("strategy")["unsafe"].mean().to_dict()
    for strategy in ORDER:
        first = script[(script["strategy"] == strategy) & (script["round"] == 1)]
        print(f"  the {LABEL[strategy]:<14} rival opened Unsafe in "
              f"{int(first['unsafe'].sum())}/{len(first)} races and played Unsafe on "
              f"{100 * rival_unsafe[strategy]:5.1f}% of its own moves")

    # The route's own opening move is a constant across all twelve cells, which
    # is what licenses reading panel b's second series as the RIVAL's first move.
    openers = route[route["round"] == 1]
    print(f"  the route opened Unsafe in {int(openers['unsafe'].sum())} of "
          f"{len(openers)} races, in every cell")

    published = json.loads(DERIVED.read_text(encoding="utf-8"))
    worst = max(
        abs(row["unsafe_rate"] - surface.loc[row["opponent_strategy"], row["max_private_risk"]])
        for row in published["rates"]
    )
    print(f"  largest disagreement with the derived table: {worst:.2e}")
    if worst > 1e-12:
        raise ValueError("recomputed rates disagree with the published derived table")

    fig = plt.figure(figsize=(S.TEXT, 2.62))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.22, 1.00, 1.02], wspace=0.58)
    draw_surface(fig, fig.add_subplot(gs[0, 0]), surface, spread_col, spread_row,
                 rival_unsafe)
    draw_contrasts(fig.add_subplot(gs[0, 1]), retal, opening)
    draw_designs(fig.add_subplot(gs[0, 2]), scripted_as, selfplay, censored)

    S.save(fig, "scripted_opponent", width=S.TEXT)


if __name__ == "__main__":
    main()
