"""The comprehension audit reproduces the endpoint's name.

The paper admits a route to gameplay through a twenty-probe comprehension audit,
three calls per probe, sixty calls per endpoint.  The gate exists to separate a
route that plays unsafely because it never understood the rules from a route
that understood them and chose.  That is a real distinction, and it is the
reason the paper is allowed to read behaviour as behaviour at all.

On these nine endpoints the gate does not draw that line independently.  It
splits the nine into five admitted and four refused, and the same split falls
out of reading the model string: every refused route carries a size word,
``mini``, ``nano`` or ``lite``, and no admitted route does.  Nine agreements out
of nine.  A reader who never ran the audit, and only read the names, would have
put the admission line in exactly the same place.

Panels
  a  The dumbbell.  Left anchor, the audit's verdict after sixty calls; right
     anchor, the tier the name announces after none.  One connector per route,
     and not one of them crosses the admission line.  The exact permutation
     p-value is enumerated rather than sampled: with four size-word names among
     nine routes there are C(9,4) = 126 assignments of the tier labels and
     exactly one reproduces the verdict.
  b  Where the ordering comes from.  Accuracy per route per probe domain, with
     the calls behind each column.  Two of the six domains are pinned at the top
     for every route and carry no ordering at all; state reconstruction is the
     domain that fails all four refused routes, and it is the only gate that
     does.
  c  Rank correlation between each domain's accuracy and the risk response, the
     behavioural quantity the rest of the paper reports.  The domain that does
     the refusing is also the domain that tracks the behaviour, on nine points.

What this figure does NOT show.  It does not show that the audit measures
nothing.  Agreeing with a size tier is not proof of emptiness: the audit may be
measuring comprehension perfectly well, and comprehension may simply track model
size across these nine endpoints, which is the reading the gate's designers
would give and which this figure cannot rule out.  What the figure does
establish is narrower and still uncomfortable.  On these nine routes the audit
carries no information the name does not already carry, so no claim resting on
the admitted-versus-refused split is separable here from a claim about model
size, and the gate earns its keep only on a route whose name and comprehension
disagree.  No such route is in this sample.  Nine commercial endpoints are not a
sample from a population of models, and n = 9 is the whole basis of every
p-value drawn here.  The rank correlations in panel c are descriptive: five
domains are tested on the same nine points, and none of the intervals a reader
might want are drawn because nine points do not support them.
"""

from __future__ import annotations

import itertools
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402

# The size words, matched against the route string split on its own separators.
# Substring matching is a trap here and not a hypothetical one: "gemini"
# contains "mini", so a naive search calls every Gemini route a small model and
# silently turns a 9-of-9 agreement into 8 of 9.
SIZE_WORDS = ("mini", "nano", "lite")

DOMAINS = [
    "rule_recall",
    "stage_payoff",
    "state_reconstruction",
    "state_transition",
    "terminal_scoring",
    "expected_payoff",
]
DOMAIN_LABEL = {
    "rule_recall": "rule\nrecall",
    "stage_payoff": "stage\npayoff",
    "state_reconstruction": "state\nrecon.",
    "state_transition": "state\ntrans.",
    "terminal_scoring": "terminal\nscore",
    "expected_payoff": "expected\npayoff",
}

ADMISSION_RUNS = (
    D.ROOT / "results" / "frontier" / "admission_campaign_v6"
    / "ai-race-frontier-admission"
)


def name_tier(route: str) -> str | None:
    """The size word the route string announces, or None if it announces none."""
    tokens = {t for t in re.split(r"[^a-z]+", route.lower()) if t}
    found = [w for w in SIZE_WORDS if w in tokens]
    if len(found) > 1:
        raise SystemExit(f"{route} carries two size words: {found}")
    return found[0] if found else None


def domain_calls() -> dict[str, int]:
    """Calls behind each probe domain, read from the run summaries.

    The derived table carries accuracies and not denominators, and a rate on six
    calls is a different object from the same rate on fifteen.  Every route is
    required to agree on the design, because a column drawn as one column has to
    be one column.
    """
    seen: dict[str, int] | None = None
    thresholds: dict | None = None
    paths = sorted(ADMISSION_RUNS.glob("*/*/*/results/*/*/admission.json"))
    if not paths:
        raise SystemExit(f"no admission summaries under {ADMISSION_RUNS}")
    for path in paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        counts = {k: int(v["rows"]) for k, v in summary["by_domain"].items()}
        if seen is None:
            seen, thresholds = counts, summary["admission_thresholds"]
        elif counts != seen or summary["admission_thresholds"] != thresholds:
            raise SystemExit(f"{path} disagrees with the other runs on the probe design")
    if set(seen) != set(DOMAINS):
        raise SystemExit(f"domains on disk are {sorted(seen)}, not {DOMAINS}")
    return seen, thresholds, len(paths)


def permutation_p(agree: np.ndarray, small: np.ndarray) -> tuple[int, int, dict]:
    """Exact p for the agreement, by enumerating every relabelling.

    The null shuffles the tier labels across the nine routes while keeping how
    many of them there are, so the reference set is the C(9,4) = 126 ways of
    choosing which four names carry a size word.  Enumeration rather than
    sampling, because 126 is small enough to count and a sampled p-value on a
    space this size is a worse number for no reason.
    """
    n, k = len(agree), int(small.sum())
    observed = int((small == agree).sum())
    counts: dict[int, int] = {}
    for combo in itertools.combinations(range(n), k):
        label = np.zeros(n, dtype=bool)
        label[list(combo)] = True
        hits = int((label == agree).sum())
        counts[hits] = counts.get(hits, 0) + 1
    return observed, sum(counts.values()), counts


def exact_spearman(x: np.ndarray, y: np.ndarray, perms: np.ndarray) -> tuple[float, float]:
    """Spearman rho with a p-value from all 9! relabellings of one side.

    ``spearmanr`` reports an asymptotic p, which on nine points is a number
    about a distribution that does not apply.  Ranks are averaged over ties, so
    a domain where eight routes are pinned at the same accuracy contributes the
    tied ranks it actually has rather than an arbitrary order.
    """
    from scipy.stats import rankdata

    rx = rankdata(x)
    ry = rankdata(y)
    rx = rx - rx.mean()
    ry = ry - ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    if denom == 0:
        return float("nan"), float("nan")
    rho = float((rx * ry).sum() / denom)
    null = (rx[perms] * ry).sum(axis=1) / denom
    p = float((np.abs(null) >= abs(rho) - 1e-12).mean())
    return rho, p


def mark_cell(ax, i, j, light: bool, *, style: str):
    """Flag one tile, in whichever ink survives that tile's own colour.

    Two marks, deliberately different in kind rather than in weight.  A gate
    failure is a ring around the whole tile, because it is a fact about that
    measurement.  A saturated value is a corner flag, because it is a fact about
    the probe's headroom, and outlining thirty of those would bury the four
    rings that decide the paper's admission line.
    """
    from matplotlib.patches import Rectangle

    ink = S.INK if light else S.SURFACE
    if style == "gate":
        ax.add_patch(Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, fill=False,
                               edgecolor=ink, lw=1.0, zorder=6))
        return
    # Well inside the tile, and a dot rather than a corner wedge: the white
    # gutter runs along every tile edge, so a light mark that touches an edge
    # reads as a chip taken out of the tile instead of as a flag on it.
    ax.scatter([j + 0.34], [i - 0.30], s=3.6, marker="o", color=ink,
               linewidths=0, zorder=6)


def main() -> None:
    admission = D.admission()
    behaviour = D.audit_versus_behaviour()
    calls, thresholds, n_runs = domain_calls()

    table = admission.merge(
        behaviour[["route", "risk_response_pp", "risk_response_race_mean_pp",
                   "degenerate_step_policy"]],
        on="route", validate="one_to_one",
    )
    if len(table) != 9:
        raise SystemExit(f"expected nine routes, got {len(table)}")
    if n_runs != len(table):
        raise SystemExit(f"{n_runs} run summaries against {len(table)} rows")

    table["size_word"] = table["route"].map(name_tier)
    table["name_small"] = table["size_word"].notna()
    table["refused"] = ~table["admitted_for_gameplay"].astype(bool)
    # The audit's own ordering, worst gate score last, with the route string as
    # a deterministic tiebreak so two routes on 0.90 cannot swap between runs.
    table = table.sort_values(["overall_accuracy", "route"],
                              ascending=[False, True]).reset_index(drop=True)

    agree = table["refused"].to_numpy()
    small = table["name_small"].to_numpy()
    observed, total, null = permutation_p(agree, small)
    p_one = sum(v for h, v in null.items() if h >= observed) / total
    # Two-sided by the minimum-likelihood criterion: every relabelling at least
    # as improbable as the one observed.  The two counts differ, four against
    # five, so no perfect reversal exists and the two values coincide here.
    p_two = sum(v for h, v in null.items() if v <= null[observed]) / total
    n_calls = int(table["n_rows"].iloc[0])

    print(f"  {len(table)} routes, {n_calls} calls each, "
          f"{int(table['repetitions'].iloc[0])} repetitions per probe")
    print(f"  audit refuses {int(agree.sum())}, the name refuses {int(small.sum())}, "
          f"agreement {observed} of {len(table)}")
    print(f"  null over C({len(table)},{int(small.sum())}) = {total} relabellings: "
          f"{dict(sorted(null.items()))}")
    print(f"  exact permutation p = {sum(v for h, v in null.items() if h >= observed)}"
          f"/{total} = {p_one:.5f} one-sided, {p_two:.5f} two-sided")
    for _, row in table.iterrows():
        print(f"    {S.ROUTE_SHORT[row['route']]:>9}  overall {row['overall_accuracy']:.3f}  "
              f"audit {'refused' if row['refused'] else 'admitted'}  "
              f"name {row['size_word'] or 'no size word'}")

    perms = np.array(list(itertools.permutations(range(len(table)))), dtype=np.int16)
    rho, rho_p = {}, {}
    for domain in DOMAINS:
        acc = table[f"accuracy_{domain}"].to_numpy(dtype=float)
        rho[domain], rho_p[domain] = exact_spearman(
            acc, table["risk_response_pp"].to_numpy(dtype=float), perms)
        alt, _ = exact_spearman(
            acc, table["risk_response_race_mean_pp"].to_numpy(dtype=float), perms)
        flat = "flat" if np.nanstd(acc) == 0 else ""
        print(f"    {domain:>21}  n={calls[domain]:>2}  rho {rho[domain]:+.3f} "
              f"p {rho_p[domain]:.4f}   race-mean rho {alt:+.3f}  {flat}")

    # Which single criterion refuses which routes.  The gate is a conjunction,
    # so "one domain does all the refusing" is a claim about coverage and has to
    # be counted rather than asserted.
    criteria = {
        "overall accuracy": (table["overall_accuracy"], thresholds["overall_accuracy_min"]),
        "state reconstruction": (table["accuracy_state_reconstruction"],
                                 thresholds["state_reconstruction_accuracy_min"]),
        "terminal scoring": (table["accuracy_terminal_scoring"],
                             thresholds["terminal_scoring_accuracy_min"]),
    }
    refused_set = set(table.loc[agree, "route"])
    for label, (column, floor) in criteria.items():
        fails = set(table.loc[column < floor - 1e-9, "route"])
        print(f"    {label:>21} below {floor:.2f}: "
              f"{sorted(S.ROUTE_SHORT[r] for r in fails)}"
              f"{'  <- refuses all four' if fails == refused_set else ''}")
    solo = [k for k, (c, f) in criteria.items() if set(table.loc[c < f - 1e-9, 'route']) == refused_set]
    if solo != ["state reconstruction"]:
        raise SystemExit(f"the criterion that refuses exactly the refused set is {solo}")

    acc = table[[f"accuracy_{d}" for d in DOMAINS]].to_numpy(dtype=float) * 100.0
    routes = list(table["route"])
    short = [S.ROUTE_SHORT[r] for r in routes]
    colours = [S.ROUTE_C[r] for r in routes]
    split = int((~agree).sum())

    # Explicit rows, with spacers, because the space under panel a carries two
    # notes and the space under panel b carries a three-deep column header: a
    # single ``hspace`` cannot be right for both and splits the difference by
    # being wrong for each.
    fig = plt.figure(figsize=(S.COL, 4.52))
    gs = fig.add_gridspec(5, 1, height_ratios=[1.30, 0.80, 1.42, 0.44, 0.58],
                          hspace=0.0, left=0.20, right=0.995, top=0.97,
                          bottom=0.03)
    ax_a = fig.add_subplot(gs[0])
    ax_b = fig.add_subplot(gs[2])
    ax_c = fig.add_subplot(gs[4])

    # --- a: two verdicts, nine flat connectors -------------------------------
    X_AUDIT, X_NAME = 0.20, 0.78
    GAP = 2.5
    ys = [len(table) - 1 - i - (GAP if i >= split else 0.0) for i in range(len(table))]

    for y, route, refused, word in zip(ys, routes, agree, table["size_word"]):
        colour = S.ROUTE_C[route]
        ax_a.plot([X_AUDIT, X_NAME], [y, y], lw=0.9, color=colour, zorder=2,
                  solid_capstyle="round")
        for x in (X_AUDIT, X_NAME):
            S.dot(ax_a, x, y, color=colour, marker=S.ROUTE_M[route], size=17,
                  filled=not refused)
        S.direct_label(ax_a, X_NAME, y, word or "none", color=S.INK_2 if word else S.MUTED,
                       dx=6, weight="bold" if word else "normal")

    line_y = (ys[split - 1] + ys[split]) / 2.0
    ax_a.axhline(line_y, color=S.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax_a.annotate("the admission line, drawn twice", xy=(0.0, line_y),
                  xytext=(0, 1.5), textcoords="offset points", ha="left",
                  va="bottom", fontsize=S.FS_NOTE, color=S.MUTED, style="italic")
    for x, head in ((X_AUDIT, f"audit\n{n_calls} calls"), (X_NAME, "name\nno calls")):
        ax_a.annotate(head, xy=(x, ys[0]), xytext=(0, 7), textcoords="offset points",
                      ha="center", va="bottom", fontsize=S.FS_NOTE, color=S.INK,
                      fontweight="bold", linespacing=1.25)
    ax_a.annotate("filled marks admitted, open refused", xy=(1.0, line_y),
                  xytext=(0, 1.5), textcoords="offset points", ha="right",
                  va="bottom", fontsize=S.FS_NOTE, color=S.MUTED)

    ax_a.set_xlim(0.0, 1.0)
    ax_a.set_ylim(min(ys) - 0.5, max(ys) + 0.5)
    ax_a.set_yticks(ys)
    ax_a.set_yticklabels(short)
    for tick, colour in zip(ax_a.get_yticklabels(), colours):
        tick.set_color(colour)
    ax_a.set_xticks([])
    for side in ax_a.spines.values():
        side.set_visible(False)
    ax_a.tick_params(length=0)
    S.panel(ax_a, "a", "the name already drew this line", pad=22)
    ax_a.annotate(
        f"{observed} of {len(table)} agree. exact permutation p = "
        f"{sum(v for h, v in null.items() if h >= observed)}/{total} = {p_one:.3f},\n"
        f"over the C({len(table)},{int(small.sum())}) = {total} ways to relabel the "
        f"tiers. n = {len(table)} endpoints.",
        xy=(0.0, 0.0), xycoords="axes fraction", xytext=(0, -13),
        textcoords="offset points", ha="left", va="top", fontsize=S.FS_NOTE,
        color=S.INK, linespacing=1.5, annotation_clip=False)

    # --- b: accuracy by probe domain -----------------------------------------
    im = S.heat_tiles(
        ax_b, acc, short,
        [DOMAIN_LABEL[d] for d in DOMAINS],
        cmap="viridis", vmin=0.0, vmax=100.0, fmt="{:.0f}",
        row_colors=colours,
    )
    for j, domain in enumerate(DOMAINS):
        ax_b.annotate(str(calls[domain]), xy=(j, -0.5), xytext=(0, 3),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=S.FS_NOTE, color=S.MUTED, annotation_clip=False)
    ax_b.annotate("calls", xy=(-0.5, -0.5), xytext=(-4, 3),
                  textcoords="offset points", ha="right", va="bottom",
                  fontsize=S.FS_NOTE, color=S.MUTED, annotation_clip=False)
    lo, hi = 0.0, 100.0
    for i in range(acc.shape[0]):
        for j, domain in enumerate(DOMAINS):
            value = acc[i, j]
            light = (value - lo) / (hi - lo) > 0.62
            gate = thresholds.get(f"{domain}_accuracy_min")
            enforced = gate is not None and domain != "expected_payoff"
            if enforced and value < 100 * gate - 1e-9:
                mark_cell(ax_b, i, j, light, style="gate")
            elif value in (0.0, 100.0):
                mark_cell(ax_b, i, j, light, style="boundary")
    ax_b.axhline(split - 0.5, color=S.INK, lw=0.9, ls=(0, (3, 2)), zorder=7)
    S.panel(ax_b, "b", "one domain does all the refusing", pad=17)
    cb = fig.colorbar(im, ax=ax_b, fraction=0.030, pad=0.035)
    cb.set_label("correct (%)", fontsize=S.FS_NOTE, labelpad=1)
    cb.ax.tick_params(labelsize=S.FS_NOTE, length=1.6)
    cb.set_ticks([0, 50, 100])
    cb.outline.set_visible(False)

    # --- c: which domain tracks the behaviour --------------------------------
    xs = np.arange(len(DOMAINS))
    for x, domain in zip(xs, DOMAINS):
        value = rho[domain]
        if np.isnan(value):
            # The grey the module reserves for a cell with no value because the
            # context never arose.  A blank column would read as a domain nobody
            # scored, and this is the one all nine scored perfectly.
            ax_c.axvspan(x - 0.5, x + 0.5, color=S.GREY_BAD, lw=0, zorder=0)
            ax_c.annotate("no variation", xy=(x, 0.62), ha="center", va="center",
                          rotation=90, fontsize=S.FS_NOTE, color=S.INK_2,
                          style="italic")
            continue
        strong = domain == solo[0].replace(" ", "_")
        colour = S.INK if strong else S.INK_2
        ax_c.plot([x, x], [0.0, value], lw=1.7 if strong else 1.0, color=colour,
                  solid_capstyle="butt", zorder=3)
        S.dot(ax_c, x, value, color=colour, marker="o", size=17 if strong else 11)
        ax_c.annotate(f"{value:.2f}", xy=(x, value), xytext=(0, 4),
                      textcoords="offset points", ha="center", va="bottom",
                      fontsize=S.FS_NOTE, color=colour,
                      fontweight="bold" if strong else "normal")
    S.zero_rule(ax_c, 0.0)
    ax_c.set_xlim(-0.5, len(DOMAINS) - 0.5)
    ax_c.set_ylim(-0.05, 1.32)
    ax_c.set_xticks(xs)
    ax_c.set_xticklabels([])
    ax_c.set_yticks([0, 0.5, 1.0])
    ax_c.set_ylabel(r"Spearman $\rho$", labelpad=2)
    S.strip(ax_c, grid_axis="y")
    ax_c.tick_params(axis="x", length=0)
    S.panel(ax_c, "c", "and it is the one that tracks behaviour", pad=4)

    # The colourbar takes its width out of panel b, so panel b and panel c stop
    # at different x without this and the column-to-column reading between them,
    # which is the only thing labelling panel c, silently stops being true.
    fig.canvas.draw()
    box_b, box_c = ax_b.get_position(), ax_c.get_position()
    ax_c.set_position([box_b.x0, box_c.y0, box_b.width, box_c.height])
    box_a = ax_a.get_position()
    ax_a.set_position([box_b.x0, box_a.y0, box_b.width, box_a.height])

    enforced_floor = {thresholds["state_reconstruction_accuracy_min"],
                      thresholds["terminal_scoring_accuracy_min"]}
    if len(enforced_floor) != 1:
        raise SystemExit(f"the two enforced gates differ: {enforced_floor}")
    S.caption(
        fig,
        "Panel b. Ring, below a gate: a route must clear "
        f"{100 * enforced_floor.pop():.0f}% on state reconstruction and on terminal "
        f"scoring and {100 * thresholds['overall_accuracy_min']:.0f}% overall, and "
        "state reconstruction is the only one of the three that refuses all four. "
        "Dot: none or all of that column's calls, a value with no room on one side "
        "and so not a distance. Panel c, Spearman correlation "
        "with the risk response; grey marks the domain every route answered "
        "perfectly, an absence of variation and not of data. Gemini 3.5 Flash Lite "
        "was audited on probe set version 8, the other eight on version 7.",
        y=-0.012,
    )

    S.save(fig, "audit_reads_the_name", width=S.COL)


if __name__ == "__main__":
    main()
