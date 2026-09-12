"""Three audited routes against a rival none of them can influence.

Every gameplay result elsewhere in this study is self-play: both companies in a
race are the same endpoint, so "the route went unsafe after its rival did" and
"the route was in an unsafe phase of its own" are the same sentence, and no
amount of stratification separates them.  This campaign breaks the symmetry.
Three gate-admitted routes each play the same game against the four reduced
strategies the paper's evolutionary lane is built on, executed by the task file
rather than by a model: Always Safe, Always Unsafe, Conditional Safe and
Conditional Unsafe.  The rival's strategy is exogenous, the two unconditional
rivals ignore the route entirely, the two conditional ones answer only the
route's own last move, and the route is never told the rival is scripted.

The two claims, and they are deliberately separate.

  1.  The SHAPE replicates.  Every route plays Unsafe least against the rival
      that always plays Safe and most against the rival that never does, with
      the two conditional rivals between them, and the paired rival-stance
      contrast is positive on all nine route-by-risk cells.

  2.  The MAGNITUDE does not.  How far the rival's stance moves a route is a
      property of the checkpoint: about seventy points on Gemini 3 Flash at
      every risk level, about fifty on GPT-5.4, and on Claude Sonnet 5 a
      Gemini-sized answer at the two lower risk levels that falls away at the
      highest.  The second claim is the one that connects to this paper's
      title, so the figure gives it a panel of its own rather than leaving it
      to be read off a table of rates.

Panels
  a  The ordering, one facet per stated risk, four rivals across and the
     route's own Unsafe play up.  A rising line is the ordering; the vertical
     distance between the three lines is the magnitude difference, which is why
     the two findings can be read off one panel.  The note records how many of
     the nine cells order strictly, because one of them does not: at risk 0.1
     Gemini 3 Flash is at 100 per cent against both Always Unsafe and
     Conditional Unsafe, and a pair of cells with no room above them cannot be
     put in an order.  That is a saturated measurement, not a violation.
  b  The rival-stance contrast, differenced inside a repetition.  The game seed
     is a base plus the repetition index and names neither the strategy, the
     risk nor the route, so one repetition is one horizon stopping-draw stream
     in every cell of the campaign, and differencing inside it removes the
     horizon from the comparison.  Point estimates and intervals are read from
     the derived tables the analyser writes rather than recomputed here, so the
     figure and the appendix table cannot drift apart.
  c  What the campaign means for every other number in this paper.  Against a
     rival that stays Safe, each route's rate sits below the rate the same
     route reaches when its rival is a second copy of itself, at every risk
     level on all three routes.  A self-play rate is therefore not a
     measurement of how a route treats risk.

The safe-rival arm is NOT called exploitation, here or anywhere this campaign
is reported.  That name would assert that the route takes the opportunity a
non-punishing rival offers, and the measured rates say the opposite: on every
route that arm carries the lowest rates in the campaign, below what the same
route plays against itself.  What the contrast measures is how far the rival's
stance moves the route, and it is named for that.

What this does NOT show.  Three routes, one game, one prompt version, ten races
per cell.  Three commercial endpoints are not a sample from a population of
models, and the four scripted rivals are reduced strategies rather than a
sample of opponents.  Panel c is a contrast between two designs and not a
decomposition: it does not license the claim that a self-play rate is an
artefact, nor that the Always Safe column is a route's "true" rate.  Both are
behaviour, under different rivals.  Panel b's contrast is about the rival's
stance only; the routes differ in their own opening move, and only Gemini 3
Flash opened Unsafe in all 120 of its races, so no panel here reads as a
statement about what the routes open with.
"""

from __future__ import annotations

import hashlib
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

CAMPAIGN = D.ROOT / "results" / "frontier" / "scripted_opponent_campaign"
DERIVED = D.ROOT / "results" / "derived" / "scripted_opponent_campaign"
PROTOCOL = "ai-race-scripted-opponent-v1"

# The route the campaign began on writes the derived table at the root of
# DERIVED; every later route writes into a folder of its own.  That asymmetry
# is the analyser's, and it exists so that adding an endpoint cannot move an
# interval the manuscript already quotes.
ORIGINAL_ROUTE = "google/gemini-3-flash-preview"

# The order the manuscript always lists admitted routes in, restricted to the
# three this campaign covers with a complete grid.
ROUTES = [r for r in S.ADMITTED if r in {
    "google/gemini-3-flash-preview",
    "openai/gpt-5.4-2026-03-05",
    "anthropic/claude-sonnet-5@default",
}]
# Every route this campaign is allowed to hold a cell for.  A cell belonging to
# anything else is an error rather than a route to skip: it would mean the
# collection wrote into the wrong tree.
FULL_ROSTER = set(S.ROUTE_ORDER)

# Ascending in what the routes do to them, so a rising line IS the ordering.
ORDER = ["AS", "CS", "CAS", "AU"]
LABEL = {
    "AS": "Always\nSafe",
    "CS": "Cond.\nSafe",
    "CAS": "Cond.\nUnsafe",
    "AU": "Always\nUnsafe",
}
# The rival's opening move is the only thing the four strategies differ in on
# round one.  It is deliberately not a colour channel on the tick labels: both
# conditional rivals mirror the route from round two, so a green "Cond. Safe"
# label would announce a safe rival that played Unsafe on roughly half of its
# own moves.  The names carry the opening move in words, and the note under the
# panel carries what the rival actually played.

N_BOOT = 5000
SEED = 20260910

EXPECTED_RACES = 10
EXPECTED_DECISIONS = 93


def cell_rng(*key) -> np.random.Generator:
    """A generator fixed by the cell, so one interval never depends on another."""
    digest = hashlib.sha256("|".join(str(part) for part in key).encode("utf-8")).digest()
    return np.random.default_rng([SEED, int.from_bytes(digest[:8], "big")])


def route_tag(route: str) -> str:
    leaf = str(route).strip().split("/")[-1]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", leaf).strip("-")


def derived_path(route: str) -> Path:
    out = DERIVED if route == ORIGINAL_ROUTE else DERIVED / route_tag(route)
    return out / "scripted_opponent_rates.json"


def load_campaign() -> pd.DataFrame:
    """Every recorded turn of the campaign, on every route, fail-closed.

    One check here is one the other campaigns cannot need: the rival is code
    rather than a model, so it cannot be audited by reading a response, and code
    that quietly played the wrong strategy would leave a table that looks
    perfectly healthy while answering a different question.  The route is
    carried as a column and never averaged over, because two routes read under
    one heading is exactly the failure this campaign's collection plan was
    restructured to prevent.
    """
    frames = []
    partial: dict[str, int] = {}
    for receipt_path in sorted(CAMPAIGN.glob("*/*/collection_receipt.json")):
        manifest = json.loads(
            receipt_path.with_name("run_manifest.json").read_text(encoding="utf-8"))
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
        if manifest.get("protocol_id") != PROTOCOL:
            raise ValueError(f"{receipt_path} is protocol {manifest.get('protocol_id')!r}")
        if manifest.get("status") != "completed":
            raise ValueError(f"{receipt_path} did not complete")
        route = manifest.get("model_route")
        # A route with only part of the grid is set aside rather than drawn or
        # silently dropped.  The campaign's whole argument is a complete four
        # rivals by three risks grid on each endpoint, and a route holding one
        # cell of twelve cannot carry an ordering or a contrast.  It must still
        # be reported, because a reader who finds it on disk and not in the
        # figure is owed the reason.
        if route not in ROUTES:
            if route not in FULL_ROSTER:
                raise ValueError(f"{receipt_path} carries unknown route {route!r}")
            partial[route] = partial.get(route, 0) + 1
            continue
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
        if sorted(seats.values()) != [EXPECTED_RACES // 2] * 2:
            raise ValueError(f"{paths[0]}: seat counterbalance is {seats}")
        if turns["game_id"].nunique() != EXPECTED_RACES:
            raise ValueError(f"{paths[0]}: {turns['game_id'].nunique()} races")
        turns["route"] = route
        turns["strategy"] = strategy
        turns["cell_risk"] = float(receipt["cell"]["max_private_risk"])
        turns["identity"] = receipt["executing_identity"]
        frames.append(turns)
    if not frames:
        raise ValueError(f"no cells under {CAMPAIGN}")
    for route, cells in sorted(partial.items()):
        print(f"  NOT DRAWN: {S.ROUTE_SHORT[route]} has {cells} of "
              f"{len(ORDER) * len(S.RISKS)} cells collected, so it carries neither "
              f"an ordering nor a paired contrast; it is left out of this campaign")
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


def paired_risk_contrast(route_rows: pd.DataFrame, route: str, strategy: str,
                         low: float, high: float) -> dict:
    """How much the stated danger moves play, differenced inside a repetition.

    This is the mirror image of the contrast the campaign analyser already
    publishes.  That one holds the risk fixed and differences two rivals inside
    a repetition; this one holds the rival fixed and differences two risk
    levels inside a repetition.  Both are legitimate for the same reason: the
    game seed is a base plus the repetition index and names neither the rival,
    the risk nor the route, so one repetition index is one horizon
    stopping-draw stream everywhere in the campaign.  The seeds are checked
    rather than assumed, and so is the sampled horizon, because the whole value
    of the pairing is that the horizon cancels.

    Computing it here rather than reading it from the derived tables is a
    departure from how the rival contrast is drawn, and it is flagged as one:
    the analyser does not yet write this contrast, so the appendix table cannot
    yet carry it.  Until it does, the two halves of the comparison come from one
    place each, and this function is the place for the risk half.
    """
    block = route_rows[(route_rows["route"] == route)
                       & (route_rows["strategy"] == strategy)]
    per = block.groupby(["cell_risk", "rep"]).agg(
        unsafe=("unsafe", "sum"), decisions=("unsafe", "size"),
        seed=("game_seed", "first"), horizon=("sampled_total_rounds", "first"))
    per["rate"] = per["unsafe"] / per["decisions"]
    reps = sorted(set(per.loc[low].index) & set(per.loc[high].index))
    if len(reps) != EXPECTED_RACES:
        raise ValueError(f"{route}: {len(reps)} shared repetitions, not {EXPECTED_RACES}")
    paired = all(per.loc[(low, r), "seed"] == per.loc[(high, r), "seed"]
                 and per.loc[(low, r), "horizon"] == per.loc[(high, r), "horizon"]
                 for r in reps)
    if not paired:
        raise ValueError(
            f"{route}: risk {low} and {high} do not share a seed and horizon "
            "inside a repetition, so differencing them does not remove the draw"
        )
    values = np.array([per.loc[(low, r), "rate"] - per.loc[(high, r), "rate"]
                       for r in reps], dtype=float)
    rng = cell_rng(route, "risk_contrast", strategy, low, high)
    draws = rng.integers(0, values.size, size=(N_BOOT, values.size))
    means = values[draws].mean(axis=1)
    return {"mean_difference": float(values.mean()),
            "ci95_low": float(np.percentile(means, 2.5)),
            "ci95_high": float(np.percentile(means, 97.5)),
            "n_blocks": int(values.size),
            "pairing_verified": True}


def key_entry(ax, xfrac, yfrac, route, *, label=None):
    """One route in the figure's key, placed in axes fractions above an axes.

    The key sits over the top row because that is where the reader meets the
    three routes first.  Each entry carries the route's own marker as well as
    its colour, so the identity survives greyscale and the lower panels need no
    key of their own.
    """
    ax.scatter([xfrac], [yfrac], s=15, marker=S.ROUTE_M[route],
               facecolors=S.ROUTE_C[route], edgecolors=S.SURFACE, linewidths=0.7,
               transform=ax.transAxes, clip_on=False, zorder=6)
    ax.annotate(label or S.ROUTE_LABEL[route], xy=(xfrac, yfrac), xycoords="axes fraction",
                xytext=(5, 0), textcoords="offset points", ha="left", va="center",
                fontsize=S.FS_NOTE, color=S.ROUTE_C[route], fontweight="bold",
                annotation_clip=False)


def draw_ordering(ax, surface, risk, *, first, middle, ceiling_cells):
    """One risk level: four rivals across, the route's own Unsafe play up."""
    x = np.arange(len(ORDER))
    for route in ROUTES:
        y = [100 * surface[(route, strategy, risk)] for strategy in ORDER]
        ax.plot(x, y, lw=1.1, color=S.ROUTE_C[route], zorder=3)
        for xi, yi in zip(x, y):
            S.dot(ax, xi, yi, color=S.ROUTE_C[route], marker=S.ROUTE_M[route], size=15)
    if any(r == risk for _route, _s, r in ceiling_cells):
        S.ceiling_rule(ax, 100.0, label="")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=S.FS_NOTE)
    ax.set_xlim(-0.45, len(ORDER) - 0.55)
    S.rate_axis(ax, label="Unsafe play (%)" if first else None)
    if not first:
        ax.set_yticklabels([])
    S.strip(ax)
    ax.set_xlabel("the scripted rival" if middle else None, labelpad=1)
    if first:
        S.panel(ax, "a", "one ordering, three routes")
    else:
        ax.set_title("", loc="left")
    ax.annotate(rf"$p_r^{{\max}} = {S.RISK_LABEL[risk]}$",
                xy=(0.97, 0.03), xycoords="axes fraction",
                ha="right", va="bottom", fontsize=S.FS_NOTE, color=S.INK_2)


def draw_stance(ax, stance, smallest_low):
    """The paired rival-stance contrast, nine cells, three routes."""
    ypos = {risk: y for risk, y in zip(S.RISKS, (2, 1, 0))}
    offsets = {route: off for route, off in zip(ROUTES, (0.24, 0.0, -0.24))}
    for route in ROUTES:
        for risk in S.RISKS:
            point, low, high = stance[(route, risk)]
            y = ypos[risk] + offsets[route]
            ax.plot([100 * low, 100 * high], [y, y], lw=1.1,
                    color=S.ROUTE_C[route], solid_capstyle="round", zorder=3)
            S.dot(ax, 100 * point, y, color=S.ROUTE_C[route],
                  marker=S.ROUTE_M[route], size=15)
    ax.set_yticks(list(ypos.values()))
    ax.set_yticklabels([S.RISK_LABEL[r] for r in ypos])
    ax.set_ylim(-0.62, 2.62)
    ax.set_xlim(-2, 88)
    ax.set_xticks([0, 25, 50, 75])
    ax.set_xlabel("vs Always Unsafe minus vs Always Safe, percentage points")
    ax.set_ylabel(r"$p_r^{\max}$", labelpad=1)
    S.strip(ax, grid_axis="x")
    # Zero is where the claim would fail, so it is drawn as a rule rather than
    # left as one more gridline among the others.
    ax.vlines(0.0, -0.62, 2.62, color=S.MUTED, lw=0.8, zorder=1)
    S.panel(ax, "b", "all three move, by different amounts")
    for line, offset in (
        ("paired inside a repetition, so the horizon draw cancels;", -28),
        (f"all nine are positive, smallest lower bound {smallest_low:+.1f} pp", -37),
    ):
        ax.annotate(line, xy=(0.5, 0.0), xycoords="axes fraction",
                    xytext=(0, offset), textcoords="offset points",
                    ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED,
                    annotation_clip=False)


def draw_designs(ax, safe_arm, selfplay, censored):
    """Two designs on each route: a fixed safe rival, and a copy of itself."""
    offsets = {route: off for route, off in zip(ROUTES, (-0.22, 0.0, 0.22))}
    for i, risk in enumerate(S.RISKS):
        for route in ROUTES:
            x = i + offsets[route]
            low_pt, low_lo, low_hi, _r, _d = safe_arm[(route, risk)]
            top_pt, top_lo, top_hi, _r2, _d2 = selfplay[(route, risk)]
            ax.plot([x, x], [100 * low_pt, 100 * top_pt], lw=0.6,
                    color=S.MUTED, linestyle=(0, (2, 1.6)), zorder=2)
            for lo, hi in ((low_lo, low_hi), (top_lo, top_hi)):
                ax.plot([x, x], [100 * lo, 100 * hi], lw=1.0,
                        color=S.ROUTE_C[route], solid_capstyle="round", zorder=3)
            S.dot(ax, x, 100 * low_pt, color=S.ROUTE_C[route],
                  marker=S.ROUTE_M[route], size=15)
            S.dot(ax, x, 100 * top_pt, color=S.ROUTE_C[route],
                  marker=S.ROUTE_M[route], size=15, filled=False)
    S.ceiling_rule(ax, 100.0, label="")
    ax.set_xticks(range(len(S.RISKS)))
    ax.set_xticklabels([S.RISK_LABEL[r] for r in S.RISKS])
    ax.set_xlim(-0.58, len(S.RISKS) - 0.42)
    ax.set_xlabel(r"maximum private risk $p_r^{\max}$", labelpad=2)
    S.rate_axis(ax)
    ax.set_ylim(0, 130)
    S.strip(ax)
    S.panel(ax, "c", "self-play sits above, on all three")
    # Colour already carries the route here, so the open and filled markers
    # carry the design instead, and the key names that and nothing else.
    for x, filled, text in ((-0.50, False, "self-play"),
                            (0.90, True, "against Always Safe")):
        S.dot(ax, x, 118.0, color=S.INK_2, marker="o", size=15, filled=filled)
        S.direct_label(ax, x, 118.0, text, color=S.INK_2, dx=5, weight="bold")
    pinned = [(route, risk) for (route, risk), flag in censored.items() if flag[0]]
    if pinned:
        where = "; ".join(
            f"{S.ROUTE_SHORT[route]} at {S.RISK_LABEL[risk]} "
            f"({censored[(route, risk)][1]} of {censored[(route, risk)][2]} races)"
            for route, risk in pinned
        )
        for line, offset in (
            ("a self-play arm on the 100% boundary makes its", -28),
            (f"distance a lower bound: {where}", -37),
        ):
            ax.annotate(line, xy=(0.5, 0.0), xycoords="axes fraction",
                        xytext=(0, offset), textcoords="offset points",
                        ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED,
                        annotation_clip=False)


def main() -> None:
    turns = load_campaign()
    route_rows = turns[turns["is_route_decision"]].copy()
    script = turns[~turns["is_route_decision"]]
    found = sorted(set(turns["route"]))
    if found != sorted(ROUTES):
        raise ValueError(f"expected {sorted(ROUTES)}, found {found}")
    cells = route_rows.groupby(["route", "strategy", "cell_risk"])
    if len(cells) != len(ROUTES) * len(ORDER) * len(S.RISKS):
        raise ValueError(f"{len(cells)} cells, not a complete three-route grid")
    sizes = cells.size()
    if set(sizes) != {EXPECTED_DECISIONS}:
        raise ValueError(f"cells are not balanced: {sorted(set(sizes))}")
    races = len(cells) * EXPECTED_RACES
    print(f"  campaign: {len(cells)} cells, {races} races, {len(route_rows)} route "
          f"decisions and {len(script)} scripted rival moves, "
          f"{EXPECTED_DECISIONS} route decisions per cell")
    print(f"  parse failures {int(turns['parse_failed'].sum())}, "
          f"scripted-rival deviations 0 by replay on every cell, "
          f"identities {sorted(set(turns['identity']))}")

    # Rates and contrasts come from the derived tables the analyser writes, so
    # the drawn numbers and the appendix table cannot drift apart.  The raw tree
    # is still read above, and its rates are checked against them below.
    surface, stance, published = {}, {}, {}
    for route in ROUTES:
        payload = json.loads(derived_path(route).read_text(encoding="utf-8"))
        if payload.get("refused"):
            raise ValueError(f"{route}: the analyser refused {payload['refused']}")
        if payload["n_cells"] != len(ORDER) * len(S.RISKS):
            raise ValueError(f"{route}: {payload['n_cells']} derived cells")
        published[route] = payload
        for row in payload["rates"]:
            surface[(route, row["opponent_strategy"], float(row["max_private_risk"]))] = \
                float(row["unsafe_rate"])
        for risk in S.RISKS:
            entry = payload["paired_contrasts"][str(risk)]["rival_unsafe_minus_rival_safe"]
            if not entry["pairing_verified"] or entry["n_blocks"] != EXPECTED_RACES:
                raise ValueError(f"{route} at risk {risk}: pairing not verified")
            stance[(route, risk)] = (entry["mean_difference"],
                                     entry["ci95_low"], entry["ci95_high"])

    recomputed = cells["unsafe"].mean().to_dict()
    worst = max(abs(recomputed[key] - surface[key]) for key in surface)
    print(f"  largest disagreement with the derived tables: {worst:.2e}")
    if worst > 1e-12:
        raise ValueError("recomputed rates disagree with the published derived tables")

    print("  the route's unsafe play, per cell:")
    for route in ROUTES:
        for risk in S.RISKS:
            row = "  ".join(f"{100 * surface[(route, s, risk)]:5.1f}" for s in ORDER)
            print(f"    {S.ROUTE_SHORT[route]:<10} risk {risk}  "
                  f"{'  '.join(s for s in ORDER):<18} -> {row}")

    strict = [(route, risk) for route in ROUTES for risk in S.RISKS
              if all(surface[(route, a, risk)] < surface[(route, b, risk)]
                     for a, b in zip(ORDER, ORDER[1:]))]
    weak = [(route, risk) for route in ROUTES for risk in S.RISKS
            if all(surface[(route, a, risk)] <= surface[(route, b, risk)]
                   for a, b in zip(ORDER, ORDER[1:]))]
    n_cells = len(ROUTES) * len(S.RISKS)
    print(f"  ordering AS < CS < CAS < AU: strict in {len(strict)}/{n_cells}, "
          f"weak in {len(weak)}/{n_cells}")
    if len(weak) != n_cells:
        raise ValueError(f"the ordering is violated in {n_cells - len(weak)} cell(s)")
    ceiling_cells = [(route, s, risk) for route in ROUTES for s in ORDER
                     for risk in S.RISKS if surface[(route, s, risk)] >= 1.0 - 1e-12]
    print(f"  cells on the 100% boundary: "
          f"{[(S.ROUTE_SHORT[r], s, k) for r, s, k in ceiling_cells]}")

    for route in ROUTES:
        for risk in S.RISKS:
            point, low, high = stance[(route, risk)]
            print(f"  rival stance  {S.ROUTE_SHORT[route]:<10} risk {risk}: "
                  f"{100 * point:+5.1f} pp [{100 * low:+5.1f}, {100 * high:+5.1f}]")
    smallest_low = 100 * min(v[1] for v in stance.values())
    print(f"  smallest lower bound anywhere: {smallest_low:+.1f} pp")
    if smallest_low <= 0:
        raise ValueError("a rival-stance interval reaches zero")

    safe_arm, selfplay, censored = {}, {}, {}
    baseline = D.baseline_turns()
    for route in ROUTES:
        own = baseline[baseline["model_route"] == route]
        for risk in S.RISKS:
            block = route_rows[(route_rows["route"] == route)
                               & (route_rows["strategy"] == "AS")
                               & (route_rows["cell_risk"] == risk)]
            safe_arm[(route, risk)] = clustered_rate(block, "rate", route, "AS", risk)
            mirror = own[own["max_private_risk"] == risk]
            selfplay[(route, risk)] = clustered_rate(mirror, "selfplay", route, risk)
            per_race = mirror.groupby("game_id")["unsafe"].agg(["sum", "size"])
            at_ceiling = int((per_race["sum"] == per_race["size"]).sum())
            censored[(route, risk)] = (at_ceiling > 0, at_ceiling, int(len(per_race)))

    for route in ROUTES:
        for risk in S.RISKS:
            gap = 100 * (selfplay[(route, risk)][0] - safe_arm[(route, risk)][0])
            mark = ">=" if censored[(route, risk)][0] else "  "
            print(f"  design gap    {S.ROUTE_SHORT[route]:<10} risk {risk}: {mark}{gap:5.1f} pp "
                  f"(safe arm {100 * safe_arm[(route, risk)][0]:5.1f}%, "
                  f"self-play {100 * selfplay[(route, risk)][0]:5.1f}%, "
                  f"self-play races at the ceiling {censored[(route, risk)][1]}/"
                  f"{censored[(route, risk)][2]})")
    if any(selfplay[key][0] <= safe_arm[key][0] for key in safe_arm):
        raise ValueError("a self-play arm does not sit above its safe-rival arm")

    # What the tick labels in panel a do not say.  The rival is code, so this is
    # a property of the campaign rather than an estimate.
    rival_unsafe = script.groupby(["route", "strategy"])["unsafe"].mean()
    for route in ROUTES:
        print(f"  the Cond. Safe rival against {S.ROUTE_SHORT[route]:<10} played Unsafe on "
              f"{100 * rival_unsafe[(route, 'CS')]:5.1f}% of its own moves "
              f"(Cond. Unsafe {100 * rival_unsafe[(route, 'CAS')]:5.1f}%)")
    openers = route_rows[route_rows["round"] == 1]
    for route in ROUTES:
        own = openers[openers["route"] == route]
        print(f"  {S.ROUTE_SHORT[route]:<10} opened Unsafe in "
              f"{int(own['unsafe'].sum())}/{len(own)} races")

    fig = plt.figure(figsize=(S.TEXT, 4.30))
    gs = fig.add_gridspec(2, 6, height_ratios=[1.0, 1.04],
                          wspace=0.62, hspace=0.92)
    facets = [fig.add_subplot(gs[0, 2 * i:2 * i + 2]) for i in range(len(S.RISKS))]
    for i, (ax, risk) in enumerate(zip(facets, S.RISKS)):
        draw_ordering(ax, surface, risk, first=(i == 0), middle=(i == 1),
                      ceiling_cells=ceiling_cells)
    for ax, xfrac, route in ((facets[1], 0.02, ROUTES[0]),
                             (facets[1], 0.60, ROUTES[1]),
                             (facets[2], 0.10, ROUTES[2])):
        key_entry(ax, xfrac, 1.10, route)

    ceiling_risk = S.RISK_LABEL[ceiling_cells[0][2]] if ceiling_cells else ""
    for line, offset in (
        (f"the ordering is strict in {len(strict)} of the {n_cells} route-by-risk cells: "
         f"at {ceiling_risk}, {S.ROUTE_LABEL[ORIGINAL_ROUTE]} is at 100% against both", -32),
        ("Always Unsafe and Cond. Unsafe, and two cells with no room above them "
         "cannot be put in an order.", -41),
        ("Cond. Safe opens Safe and then mirrors, so it played Unsafe on "
         + ", ".join(f"{100 * rival_unsafe[(route, 'CS')]:.0f}% "
                     f"({S.ROUTE_SHORT[route]})" for route in ROUTES)
         + " of its own moves.", -50),
    ):
        facets[1].annotate(line, xy=(0.5, 0.0), xycoords="axes fraction",
                           xytext=(0, offset), textcoords="offset points",
                           ha="center", va="top", fontsize=S.FS_NOTE, color=S.MUTED,
                           annotation_clip=False)

    draw_stance(fig.add_subplot(gs[1, 0:3]), stance, smallest_low)
    draw_designs(fig.add_subplot(gs[1, 3:6]), safe_arm, selfplay, censored)

    S.save(fig, "scripted_opponent", width=S.TEXT)

    risk_stance = {
        route: paired_risk_contrast(route_rows, route, "AS", S.RISKS[0], S.RISKS[-1])
        for route in ROUTES
    }
    # The analyser now publishes this contrast too, on every arm.  Reading the
    # published value back is what keeps the panel and the appendix table from
    # drifting: the recomputation above is the same construction, so any
    # disagreement means one of the two is wrong rather than that the figure
    # needs adjusting.
    key = f"risk_{S.RISKS[0]:g}_minus_{S.RISKS[-1]:g}"
    for route in ROUTES:
        table = published[route].get("paired_risk_contrasts")
        if not table:
            raise ValueError(
                f"{route}: the derived table carries no paired risk contrast, so "
                "the panel's risk half cannot be checked against it"
            )
        off = abs(table["AS"][key]["mean_difference"]
                  - risk_stance[route]["mean_difference"])
        if off > 1e-12:
            raise ValueError(
                f"{route}: the recomputed risk contrast and the published one "
                f"differ by {off:.3e}; the figure is not the thing to fix"
            )
    print("\n  the danger, differenced the same way, against a rival held at "
          "Always Safe:")
    for route in ROUTES:
        entry = risk_stance[route]
        print(f"  risk stance   {S.ROUTE_SHORT[route]:<10} {S.RISKS[0]} vs {S.RISKS[-1]}: "
              f"{100 * entry['mean_difference']:+5.1f} pp "
              f"[{100 * entry['ci95_low']:+5.1f}, {100 * entry['ci95_high']:+5.1f}] "
              f"over {entry['n_blocks']} repetition blocks")

    # The widest risk contrast anywhere in the grid, which is NOT on the arm the
    # panel draws.  The panel compares the rival against the danger at a rival
    # held Safe, and that comparison is true of the two unconditional arms only:
    # against a rival that copies the route's own last move the danger moves
    # play further than the smallest rival contrast does.  A panel that claims
    # the rival always outweighs the danger would be claiming of the grid what
    # is true of two of its four columns, so the number that refuses that
    # reading is measured here and written on the panel.
    widest = max(
        ({"route": route, "strategy": strategy,
          **published[route]["paired_risk_contrasts"][strategy][key]}
         for route in ROUTES for strategy in ORDER),
        key=lambda entry: entry["mean_difference"],
    )
    print(f"  the widest risk contrast in the grid is on {widest['strategy']} for "
          f"{S.ROUTE_SHORT[widest['route']]}: {100 * widest['mean_difference']:+.1f} pp "
          f"[{100 * widest['ci95_low']:+.1f}, {100 * widest['ci95_high']:+.1f}], "
          f"against a smallest rival contrast of {100 * min(v[1] for v in stance.values()):+.1f} pp "
          "lower bound, so the two families DO meet once the conditional rivals "
          "are included")
    rival_low = 100 * min(v[1] for v in stance.values())
    risk_high = 100 * max(e["ci95_high"] for e in risk_stance.values())
    print(f"  every rival interval starts at or above {rival_low:+.1f} pp and every "
          f"risk interval ends at or below {risk_high:+.1f} pp, so the two families "
          f"do not overlap: {rival_low > risk_high}")
    if rival_low <= risk_high:
        raise ValueError(
            "a rival interval now reaches into the risk intervals; the panel that "
            "draws an empty corridor between the two families would be drawing a "
            "corridor that is not empty"
        )
    ratios = [stance[(route, risk)][0] / risk_stance[route]["mean_difference"]
              for route in ROUTES for risk in S.RISKS]
    print(f"  the rival is worth {min(ratios):.1f} to {max(ratios):.1f} times the "
          f"risk across the nine route-by-risk cells")

def draw_thesis(ax, stance, risk_stance, rival_low, risk_high, ratios, widest):
    """The rival and the danger on one axis, both differenced the same way.

    This is the comparison the paper is named for and it did not previously
    exist as a picture.  Two families of paired contrasts sit on one scale of
    percentage points: what changing the rival does to a route's Unsafe play,
    and what changing the stated danger does to it.  Putting them on one axis is
    the whole argument, because the reader then does not have to hold two
    numbers from two sections in mind to see that one is several times the
    other.  Both families are differenced inside a repetition, so both have the
    horizon draw removed and neither is a comparison of two separately estimated
    rates.
    """
    rival_y, risk_y = 2.55, 0.62
    spread = 0.125
    entries = [(route, risk) for route in ROUTES for risk in S.RISKS]
    for i, (route, risk) in enumerate(entries):
        point, low, high = stance[(route, risk)]
        y = rival_y + (i - (len(entries) - 1) / 2) * spread
        ax.plot([100 * low, 100 * high], [y, y], lw=1.0, color=S.ROUTE_C[route],
                solid_capstyle="round", zorder=3)
        S.dot(ax, 100 * point, y, color=S.ROUTE_C[route],
              marker=S.ROUTE_M[route], size=13)
    for i, route in enumerate(ROUTES):
        entry = risk_stance[route]
        y = risk_y + (i - (len(ROUTES) - 1) / 2) * spread
        ax.plot([100 * entry["ci95_low"], 100 * entry["ci95_high"]], [y, y],
                lw=1.0, color=S.ROUTE_C[route], solid_capstyle="round", zorder=3)
        S.dot(ax, 100 * entry["mean_difference"], y, color=S.ROUTE_C[route],
              marker=S.ROUTE_M[route], size=13)

    # The corridor between the two families.  Nothing is drawn inside it, and
    # saying so is the panel's claim, so it is marked as empty space rather than
    # left for the reader to measure.
    ax.axvspan(risk_high, rival_low, color=S.BAND, zorder=0, lw=0)
    ax.annotate("no interval of\neither kind reaches\nin here",
                xy=(0.5 * (risk_high + rival_low), 1.60), ha="center", va="center",
                fontsize=S.FS_NOTE, color=S.MUTED, linespacing=1.3)

    ax.set_yticks([rival_y, risk_y])
    ax.set_yticklabels(["what the rival\nis doing", "how dangerous\nthe race is"])
    ax.tick_params(axis="y", length=0, pad=3)
    ax.set_ylim(-0.05, 3.32)
    ax.set_xlim(-2, 84)
    ax.set_xticks([0, 25, 50, 75])
    # "points" here and in the risk-response figure, rather than "pp" in one
    # and "points" in the other: two names for one unit spends the reader on
    # the unit instead of on the number.
    ax.set_xlabel("change in the route's Unsafe play (points)", labelpad=2)
    S.strip(ax, grid_axis="x")
    ax.vlines(0.0, -0.05, 3.32, color=S.MUTED, lw=0.8, zorder=1)
    # The claim names the arm it is true of.  Left unqualified it would assert
    # something the campaign's own grid refuses: on the conditional-unsafe arm
    # the danger moves play further than the smallest rival contrast does, and
    # the third line below is that number rather than a hedge.
    S.panel(ax, "b", "against a safe rival, the rival outweighs the danger",
            gap=8.0)
    ax.annotate("above: three routes at three risk levels.\n"
                f"below: risk {S.RISK_LABEL[S.RISKS[0]]} against "
                f"{S.RISK_LABEL[S.RISKS[-1]]}, rival held at Always Safe.\n"
                f"against a rival that copies the route, the danger is worth "
                f"up to {100 * widest['mean_difference']:.0f} points.",
                xy=(0.5, 0.0), xycoords="axes fraction", xytext=(0, -22),
                textcoords="offset points", ha="center", va="top",
                fontsize=S.FS_NOTE, color=S.MUTED, linespacing=1.35,
                annotation_clip=False)


def draw_ribbon(ax, surface, ceiling_cells, n_cells, strict):
    """Four rivals across, every route-by-risk cell as its own rising line."""
    x = np.arange(len(ORDER))
    dashes = {risk: style for risk, style in
              zip(S.RISKS, ["-", (0, (3.2, 1.6)), (0, (1.2, 1.4))])}
    for route in ROUTES:
        for risk in S.RISKS:
            y = [100 * surface[(route, strategy, risk)] for strategy in ORDER]
            ax.plot(x, y, lw=1.0, color=S.ROUTE_C[route], ls=dashes[risk], zorder=3)
            S.dot(ax, x[0], y[0], color=S.ROUTE_C[route], marker=S.ROUTE_M[route],
                  size=12)
            S.dot(ax, x[-1], y[-1], color=S.ROUTE_C[route], marker=S.ROUTE_M[route],
                  size=12)
    if ceiling_cells:
        S.ceiling_rule(ax, 100.0, label="")
    ax.set_xticks(x)
    ax.set_xticklabels([LABEL[s] for s in ORDER], fontsize=S.FS_NOTE)
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.set_xlabel("the scripted rival", labelpad=1)
    S.rate_axis(ax, label="Unsafe play (%)")
    ax.set_ylim(0, 126)
    S.strip(ax)
    S.panel(ax, "a", f"all {n_cells} cells rise, {strict} strictly", gap=8.0)
    # The risk level is a dash pattern here rather than a third colour, because
    # colour is already spent on the route and a reader who has learnt the
    # routes in the figure before this one reads this panel for free.
    for i, risk in enumerate(S.RISKS):
        ax.plot([0.05, 0.45], [121.0 - 7.0 * i] * 2, color=S.INK_2, lw=1.0,
                ls=dashes[risk], zorder=5)
        S.direct_label(ax, 0.52, 121.0 - 7.0 * i,
                       f"risk {S.RISK_LABEL[risk]}", color=S.INK_2, dx=0)


def draw_main_figure(surface, stance, risk_stance, safe_arm, selfplay, censored,
                     strict, n_cells, ceiling_cells, rival_low, risk_high, ratios,
                     widest):
    """The three panels the main paper carries, across both columns.

    The campaign is the only design in this paper whose treatment is code the
    route cannot influence and is never told about, which makes it the only
    place a contrast here is causal by construction.  It therefore carries the
    paper's headline and is given the width to make it: the ordering, the
    comparison the paper is named for, and what that comparison costs every
    self-play rate in the study.
    """
    # A figure spanning both columns is charged twice its height against the
    # page budget and nothing for its width, so height here is the expensive
    # dimension.  The panels read at 1.66 in as well as they did at 1.88, and
    # the difference is about three lines of the manuscript.
    fig = plt.figure(figsize=(S.TEXT, 1.66))
    gs = fig.add_gridspec(1, 3, width_ratios=[0.95, 1.28, 0.80], wspace=0.62)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[0, 2])

    draw_ribbon(ax_a, surface, ceiling_cells, n_cells, len(strict))
    draw_thesis(ax_b, stance, risk_stance, rival_low, risk_high, ratios, widest)

    designs = {route: off for route, off in zip(ROUTES, (-0.24, 0.0, 0.24))}
    for i, risk in enumerate(S.RISKS):
        for route in ROUTES:
            x = i + designs[route]
            low_pt = safe_arm[(route, risk)][0]
            top_pt = selfplay[(route, risk)][0]
            ax_c.plot([x, x], [100 * low_pt, 100 * top_pt], lw=0.6,
                      color=S.MUTED, linestyle=(0, (2, 1.6)), zorder=2)
            S.dot(ax_c, x, 100 * low_pt, color=S.ROUTE_C[route],
                  marker=S.ROUTE_M[route], size=13)
            S.dot(ax_c, x, 100 * top_pt, color=S.ROUTE_C[route],
                  marker=S.ROUTE_M[route], size=13, filled=False)
    S.ceiling_rule(ax_c, 100.0, label="")
    ax_c.set_xticks(range(len(S.RISKS)))
    ax_c.set_xticklabels([S.RISK_LABEL[r] for r in S.RISKS])
    ax_c.set_xlim(-0.60, len(S.RISKS) - 0.40)
    ax_c.set_xlabel(r"maximum private risk $p_r^{\max}$", labelpad=2)
    S.rate_axis(ax_c, label="Unsafe play (%)")
    ax_c.set_ylim(0, 126)
    S.strip(ax_c)
    # The count belongs in the caption rather than here: at this panel width
    # a title carrying it either runs off the figure or has to be shortened to
    # "all 9", which names no unit.  The panel underclaims and the caption
    # says in how many cells.
    S.panel(ax_c, "c", "self-play sits above", gap=8.0)
    # Colour carries the route in every panel, so here the open and filled
    # markers carry the design instead, and the key names only that.  It sits in
    # the band above the ceiling rule, which no measurement can reach.
    for y, filled, text in ((121.0, False, "self-play"),
                            (110.0, True, "vs Always Safe")):
        S.dot(ax_c, -0.42, y, color=S.INK_2, marker="o", size=13, filled=filled)
        S.direct_label(ax_c, -0.42, y, text, color=S.INK_2, dx=4, weight="bold")

    # The routes are named once, over the leftmost panel, and every later panel
    # reuses the hue and the glyph rather than spending space on a second key.
    for i, route in enumerate(ROUTES):
        key_entry(ax_a, 0.02 + 0.40 * i, 1.26, route, label=S.ROUTE_SHORT[route])

    S.save(fig, "scripted_opponent_main", width=S.TEXT)


if __name__ == "__main__":
    main()
