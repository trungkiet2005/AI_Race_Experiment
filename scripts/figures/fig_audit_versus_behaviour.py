"""What the comprehension audit saw, set beside how much the play moves.

The claim.  Across these nine endpoints the routes that rebuild the game state
in the admission probe are also the routes whose play moves most when the
assigned maximum private risk moves, Spearman rho = 0.87 over nine routes with
an exact permutation p of 0.004, and 0.92 with the degenerate route set aside.
That is a rank agreement between two measurements of the same nine endpoints.
It is not an effect, and nothing here is estimated from a sample.

Why the numbers are drawn the way they are.  The gameplay quantity is the
pooled decision-level Unsafe rate in each risk cell, which is the number the
run's own ``summary.json`` publishes and therefore the number an independent
reader reproduces from the artefact; the risk response is that rate at 0.1
minus the same rate at 0.9, in percentage points.  The audit quantity is
state-reconstruction accuracy, the one probe domain whose threshold refuses
exactly the four routes the gate refuses.  Both are recomputed here from the
turn-level records and the admission table, and checked against the derived
table the previous figure used.

Panels
  a  The whole grid as tiles: nine routes, three risk levels, the Unsafe rate
     printed in every cell, with the 0.1-minus-0.9 drop beside each row.  Nine
     routes over three x-positions is nine crossing lines and no argument, so
     the rates are a table of colour instead.  Rows run from the largest drop
     to the smallest, and the route glyph beside each row is the glyph that
     route carries in panel b and in the rest of the paper.
  b  The association.  State-reconstruction accuracy against the risk response,
     one point per route, the admission threshold drawn where the gate puts it.
     Claude Opus 5 sits on the ceiling of the response scale, which is marked,
     because a 100-point drop is the largest the scale allows and not a
     measurement with room above it.
  c  Why Claude Opus 5 is a boundary case rather than the strongest responder.
     Its policy is a step: every decision Unsafe at risk 0.1, every decision
     Safe at 0.6 and at 0.9, ten identical races in each cell, zero within-cell
     variance, so its interval collapses to a point.  The other eight routes
     are drawn in grey behind it as the shape a graded policy makes.

What this figure does NOT show.  It does not show that comprehension causes
risk sensitivity, or that the gate improves anything.  The admitted and the
refused routes differ in far more than their probe scores: vendor, model size,
price, release date, and whatever else separates a flagship endpoint from a
mini or nano one, so the association in panel b is consistent with any of those
carrying it.  Nine commercial endpoints are not a sample from a population of
models, so rho is a description of these nine and the permutation p refers to
relabellings of these nine routes and to nothing wider.  The panels report
point rates without their clustered intervals, which live in the derived table
and in the manuscript's own text; panel a is a table of central values and
panel c's grey context is context, not a comparison.  Nothing here speaks to
what a route does at a risk level it was never assigned.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402

ADMISSION_RUNS = (
    D.ROOT / "results" / "frontier" / "admission_campaign_v6"
    / "ai-race-frontier-admission"
)
# The domain whose threshold refuses exactly the four routes the gate refuses,
# which is why this is the audit axis rather than the overall score.
PROBE = "state_reconstruction"
BOUNDARY = "anthropic/claude-opus-5@default"


def gate_threshold() -> float:
    """The admission floor, read from the runs rather than typed in.

    A rule drawn as a line in a figure is a number like any other, and the only
    place it is authoritative is the frozen protocol each admission run carries
    with it.  Every run has to agree, because one line is being drawn.
    """
    paths = sorted(ADMISSION_RUNS.glob("*/*/*/results/*/*/admission.json"))
    if not paths:
        raise SystemExit(f"no admission summaries under {ADMISSION_RUNS}")
    seen = {
        json.loads(p.read_text(encoding="utf-8"))["admission_thresholds"][
            f"{PROBE}_accuracy_min"
        ]
        for p in paths
    }
    if len(seen) != 1:
        raise SystemExit(f"the {len(paths)} admission runs disagree on the gate: {seen}")
    return float(seen.pop())


def exact_spearman(x: np.ndarray, y: np.ndarray) -> tuple[float, float, int]:
    """Spearman rho with a p-value from every relabelling of the routes.

    ``spearmanr`` reports an asymptotic p, which on nine points is a statement
    about a distribution that does not apply.  Nine routes is 362,880
    relabellings, which is small enough to count, so the p-value is exact
    rather than sampled.  Ranks are averaged over ties, and there are ties: two
    routes reconstruct the state perfectly and two more share 0.93.
    """
    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = float(np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))
    rho = float((rx * ry).sum() / denom)
    perms = np.array(list(itertools.permutations(range(len(x)))), dtype=np.int8)
    null = (rx[perms] * ry).sum(axis=1) / denom
    return rho, float((np.abs(null) >= abs(rho) - 1e-12).mean()), len(perms)


def tile_ink(value: float, lo: float, hi: float, cmap: str = "viridis") -> str:
    """The ink that survives on the tile this value paints.

    Viridis runs dark at one end and bright at the other, so a fixed cut on the
    value puts white on yellow at one end and near-black on indigo at the
    other.  Ask the colormap for the colour it will draw and take the ink from
    that colour's luminance.
    """
    r, g, b, _ = plt.get_cmap(cmap)((value - lo) / ((hi - lo) or 1.0))
    return S.INK if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.55 else S.SURFACE


def build() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-route rates and the risk response, from the decision records."""
    turns = D.baseline_turns()
    rates = turns.groupby(["model_route", "max_private_risk"])["unsafe"].mean().unstack()
    if list(rates.columns) != list(S.RISKS):
        raise SystemExit(f"risk levels on disk are {list(rates.columns)}, not {list(S.RISKS)}")

    admission = D.admission().set_index("route")
    table = pd.DataFrame(
        {
            "risk_response_pp": (rates[0.1] - rates[0.9]) * 100.0,
            "probe": admission[f"accuracy_{PROBE}"] * 100.0,
            "admitted": admission["admitted_for_gameplay"].astype(bool),
        }
    )
    for risk in S.RISKS:
        table[f"rate_{risk}"] = rates[risk] * 100.0
    if table.isna().any().any() or len(table) != 9:
        raise SystemExit(f"expected nine routes joined on both campaigns, got {len(table)}")

    # The split the rest of the paper uses is declared in figstyle, so a route
    # that changed side in the campaign table has to say so here rather than
    # quietly redrawing the admission line.
    declared = set(S.ADMITTED)
    if set(table.index[table["admitted"]]) != declared:
        raise SystemExit(
            f"the admission table admits {sorted(table.index[table['admitted']])}, "
            f"figstyle declares {sorted(declared)}"
        )

    # Zero variance between the races of every cell: the ten races of that cell
    # are the same race as far as the measurement is concerned, so the route's
    # interval is a point and its position is a boundary, not an estimate.
    spread = (
        turns.groupby(["model_route", "max_private_risk", "game_id"])["unsafe"].mean()
        .groupby(["model_route", "max_private_risk"]).var(ddof=0)
        .groupby("model_route").max()
    )
    table["degenerate"] = spread == 0.0

    counts = turns.groupby(["model_route", "max_private_risk"]).size().unstack()
    return table, counts


def cross_check(table: pd.DataFrame) -> None:
    """Refuse to draw a number that disagrees with the tabulated campaign.

    The derived table is what the manuscript's prose quotes.  A figure drawn
    from the raw records is the right way to make a figure and the wrong thing
    to ship silently if the two ever part company, so the disagreement is the
    error rather than the figure.
    """
    old = D.audit_versus_behaviour().set_index("route").reindex(table.index)
    if old.isna().any().any():
        raise SystemExit("the derived table does not carry all nine drawn routes")
    worst = {}
    for risk in S.RISKS:
        tag = f"risk_{risk:g}".replace(".", "p")
        worst[f"rate at {risk}"] = float(
            (old[f"unsafe_rate_{tag}"] * 100.0 - table[f"rate_{risk}"]).abs().max()
        )
    worst["risk response"] = float(
        (old["risk_response_pp"] - table["risk_response_pp"]).abs().max()
    )
    worst[f"{PROBE} accuracy"] = float(
        (old[f"admission_{PROBE}_accuracy"] * 100.0 - table["probe"]).abs().max()
    )
    worst["degenerate flag"] = float(
        (old["degenerate_step_policy"].astype(bool) != table["degenerate"]).sum()
    )
    for name, delta in worst.items():
        print(f"  cross-check against the derived table, {name}: max |diff| {delta:.2e}")
    bad = {k: v for k, v in worst.items() if v > 1e-9}
    if bad:
        raise SystemExit(
            "THE RECOMPUTED NUMBERS DISAGREE WITH results/frontier/baseline_campaign_v6/"
            f"derived/audit_versus_behaviour.csv: {bad}. The figure is not the thing to "
            "fix; one of the two computations is wrong and the paper quotes the other."
        )


def draw_tiles(ax, table, order, counts) -> None:
    arr = table.loc[order, [f"rate_{r}" for r in S.RISKS]].to_numpy(dtype=float)
    S.heat_tiles(
        ax, arr,
        [S.ROUTE_SHORT[r] for r in order],
        [S.RISK_LABEL[r] for r in S.RISKS],
        cmap="viridis", vmin=0.0, vmax=100.0, fmt="{:.0f}",
        row_colors=[S.ROUTE_C[r] for r in order],
    )
    # Room to the right of the tiles for the drop, and to the left for the
    # glyphs, inside the same axes so neither can drift off the panel.
    ax.set_xlim(-1.45, 3.75)

    for i, route in enumerate(order):
        S.dot(ax, -1.0, i, color=S.ROUTE_C[route], marker=S.ROUTE_M[route],
              size=15, filled=bool(table.loc[route, "admitted"]))
        ax.text(3.70, i, f"{table.loc[route, 'risk_response_pp']:.0f}",
                ha="right", va="center", fontsize=S.FS_NOTE, color=S.INK_2)

    # A cell that sits on 0 or on 100 is a boundary and not a reading with room
    # on both sides of it, so it is ringed and said out loud under the panel.
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if arr[i, j] in (0.0, 100.0):
                S.highlight(ax, i, j, color=tile_ink(arr[i, j], 0.0, 100.0),
                            lw=0.9, inset=0.40)

    bottom = arr.shape[0] - 0.5
    ax.annotate(r"maximum private risk $p_r^{\max}$", xy=(1.0, bottom),
                xytext=(0, -13), textcoords="offset points", ha="center",
                va="top", fontsize=S.FS_TICK, color=S.INK_2, annotation_clip=False)
    ax.annotate("drop\n(pp)", xy=(3.70, bottom), xytext=(0, -2),
                textcoords="offset points", ha="right", va="top",
                fontsize=S.FS_NOTE, color=S.INK_2, linespacing=1.2,
                annotation_clip=False)
    # Wrapped by hand to the panel's own width.  A note set as one long line is
    # 3 in of type under a 2 in panel, and it prints straight through the note
    # belonging to the panel beside it.
    per_cell = int(counts.min().min())


def draw_association(ax, table, order, gate, stats) -> None:
    for route in order:
        row = table.loc[route]
        S.dot(ax, row["probe"], row["risk_response_pp"], color=S.ROUTE_C[route],
              marker=S.ROUTE_M[route], size=20, filled=bool(row["admitted"]))

    # Hand-placed because nine labels on nine points is a packing problem and
    # not a rule; the offsets are in points, so they hold at any canvas size.
    # Four routes sit within three points of each other in the response, so
    # they are staggered above, below and to either side rather than all set
    # beside their own marker, where they would overprint one another.
    # The gate is a rule down the whole panel, so a label is also placed on the
    # side of its own marker that keeps it off that rule.
    offsets = {
        "anthropic/claude-opus-5@default": (6, -5, "left"),
        "anthropic/claude-sonnet-5@default": (6, 0, "left"),
        "openai/gpt-5.4-2026-03-05": (6, 8, "left"),
        "google/gemini-3-flash-preview": (0, -9, "center"),
        "openai/gpt-5.5-2026-04-23": (2, 7, "left"),
        "google/gemini-3.1-flash-lite-preview": (-6, 0, "right"),
        "google/gemini-3.5-flash-lite": (-6, 0, "right"),
        "openai/gpt-5.4-mini-2026-03-17": (-6, 0, "right"),
        "openai/gpt-5.4-nano-2026-03-17": (0, 7, "center"),
    }
    for route in order:
        dx, dy, ha = offsets[route]
        row = table.loc[route]
        S.direct_label(ax, row["probe"], row["risk_response_pp"],
                       S.ROUTE_SHORT[route], color=S.ROUTE_C[route],
                       ha=ha, dx=dx, dy=dy)

    ax.axvline(gate, color=S.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.annotate(f"gate, {gate:.0f}%", xy=(gate, 1.0), xycoords=("data", "axes fraction"),
                xytext=(-3, -1), textcoords="offset points", ha="right", va="top",
                fontsize=S.FS_NOTE, color=S.MUTED)
    # The scale runs to 100 and one route is on it, so the axis keeps a band
    # above the rule: the note goes there, and the route that sits on the rule
    # keeps the room below it for its own name.
    S.ceiling_rule(ax, 100.0, label="the scale's ceiling")

    rho, p, n_perm, rho_out = stats
    # The two numbers sit in the panel, in the empty band left of the gate rule
    # and under the ceiling; how the p-value was obtained is a sentence and
    # goes in the note, because a block wide enough to hold it is a block the
    # gate rule draws a dashed line through.
    ax.annotate(
        f"$\\rho$ = {rho:.2f}, $p$ = {p:.3f}\n"
        f"without {S.ROUTE_SHORT[BOUNDARY]}, {rho_out:.2f}",
        xy=(0.0, 0.80), xycoords="axes fraction", xytext=(4, 0),
        textcoords="offset points", ha="left", va="top",
        fontsize=S.FS_NOTE, color=S.INK_2, linespacing=1.45)

    ax.set_xlim(8, 126)
    ax.set_xticks([20, 40, 60, 80, 100])
    ax.set_ylim(-2, 113)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_xlabel("state reconstruction, % correct")
    ax.set_ylabel("drop in Unsafe play, 0.1 to 0.9 (pp)")
    S.strip(ax)


def draw_step(ax, table, counts) -> None:
    risks = np.array(S.RISKS, dtype=float)
    for route in table.index:
        if route == BOUNDARY:
            continue
        ax.plot(risks, [table.loc[route, f"rate_{r}"] for r in S.RISKS],
                color=S.HAIRLINE, lw=0.8, zorder=2, solid_capstyle="round")

    values = [table.loc[BOUNDARY, f"rate_{r}"] for r in S.RISKS]
    ax.plot(risks, values, color=S.ROUTE_C[BOUNDARY], lw=1.5, zorder=4,
            marker=S.ROUTE_M[BOUNDARY], ms=4.0, clip_on=False,
            markeredgecolor=S.SURFACE, markeredgewidth=0.7)

    S.ceiling_rule(ax, 100.0, label="every decision")
    S.zero_rule(ax, 0.0)
    ax.annotate("no decision", xy=(1.0, 0.0), xycoords=("axes fraction", "data"),
                xytext=(-1, 2), textcoords="offset points", ha="right",
                va="bottom", fontsize=S.FS_NOTE, color=S.MUTED)

    S.direct_label(ax, 0.1, values[0], S.ROUTE_LABEL[BOUNDARY],
                   color=S.ROUTE_C[BOUNDARY], dx=6, dy=-6, weight="bold")
    # Below the lowest grey curve and right of the boundary route's descent,
    # which leaves a triangle rather than a strip, so the label is set on two
    # lines: on one line it is wide enough to reach back across that descent.
    S.direct_label(ax, 0.9, table.drop(index=BOUNDARY)["rate_0.9"].min(),
                   "the other\neight", color=S.MUTED, ha="right", dx=-4, dy=-17)

    S.risk_axis(ax)
    S.rate_axis(ax, label="Unsafe play (%)")
    ax.set_ylim(-4, 108)
    S.strip(ax)

    # Under the panel, not inside it: this panel is the narrowest of the three
    # and four lines of type set in its own middle either overflow the figure
    # or cover the grey curves that make the comparison.
    low = int(counts.loc[BOUNDARY, 0.1])
    high = int(counts.loc[BOUNDARY, [0.6, 0.9]].sum())


def main() -> None:
    table, counts = build()
    cross_check(table)

    gate = gate_threshold()
    # This panel draws one probe domain and one rule, and the reason it is
    # allowed to stand for the whole gate is that this domain's threshold
    # refuses exactly the routes the gate refuses.  If that ever stops being
    # true the panel is drawing a rule that decides nothing, so it is checked
    # rather than asserted.
    by_probe = set(table.index[table["probe"] < 100 * gate - 1e-9])
    if by_probe != set(table.index[~table["admitted"]]):
        raise SystemExit(
            f"the {PROBE} threshold refuses {sorted(by_probe)}, but the gate "
            f"refuses {sorted(table.index[~table['admitted']])}; this panel's "
            "single rule no longer stands for the admission decision"
        )
    order = list(table["risk_response_pp"].sort_values(ascending=False).index)
    rho, p, n_perm = exact_spearman(table["probe"].to_numpy(float),
                                    table["risk_response_pp"].to_numpy(float))
    trimmed = table.drop(index=BOUNDARY)
    rho_out, p_out, _ = exact_spearman(trimmed["probe"].to_numpy(float),
                                       trimmed["risk_response_pp"].to_numpy(float))

    print(f"  nine routes, {int(counts.to_numpy().sum())} decisions, "
          f"gate on {PROBE} at {100 * gate:.0f}%")
    for route in order:
        row = table.loc[route]
        print(f"    {S.ROUTE_SHORT[route]:>9}  "
              + "  ".join(f"{row[f'rate_{r}']:5.1f}" for r in S.RISKS)
              + f"   drop {row['risk_response_pp']:5.1f} pp   probe {row['probe']:5.1f}%"
              + ("  admitted" if row["admitted"] else "  refused")
              + ("  degenerate" if row["degenerate"] else ""))
    print(f"  Spearman rho {rho:.4f}, exact permutation p {p:.5f} over {n_perm} relabellings")
    print(f"  without {S.ROUTE_SHORT[BOUNDARY]}: rho {rho_out:.4f}, p {p_out:.5f}")
    drop = table["risk_response_pp"]
    print(f"  drop ranges {drop.min():.1f} to {drop.max():.1f} pp; "
          f"admitted mean {drop[table['admitted']].mean():.1f}, "
          f"refused mean {drop[~table['admitted']].mean():.1f}")

    # The float this replaces was 2.56 in tall and the paper has no vertical
    # slack, so the panels are drawn to fit the space that exists rather than
    # costing a page.
    fig = plt.figure(figsize=(S.TEXT, 2.48))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.05, 1.30, 0.92], wspace=0.60)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_tiles(ax_a, table, order, counts)
    S.panel(ax_a, "a", f"all nine fall, by {drop.min():.0f} to {drop.max():.0f} points")

    draw_association(ax_b, table, order, 100 * gate,
                     (rho, p, n_perm, rho_out))
    S.panel(ax_b, "b", "the two orders move together")

    draw_step(ax_c, table, counts)
    S.panel(ax_c, "c", "one route is a step, not a slope")

    S.save(fig, "audit_versus_behaviour_v2", width=S.TEXT)


if __name__ == "__main__":
    main()
