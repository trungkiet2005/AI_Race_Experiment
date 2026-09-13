"""Five audited routes against a rival none of them can influence.

Every gameplay result elsewhere in this study is self-play: both companies in a
race are the same endpoint, so "the route went unsafe after its rival did" and
"the route was in an unsafe phase of its own" are the same sentence, and no
amount of stratification separates them.  This campaign breaks the symmetry.
Five gate-admitted routes each play the same game against the four reduced
strategies the paper's evolutionary lane is built on, executed by the task file
rather than by a model: Always Safe, Always Unsafe, Conditional Safe and
Conditional Unsafe.  The rival's strategy is exogenous, the two unconditional
rivals ignore the route entirely, the two conditional ones answer only the
route's own last move, and the route is never told the rival is scripted.

The two claims, and they are deliberately separate.

  1.  The SHAPE is largely shared.  Every route has a positive paired contrast
      between the Always-Unsafe and Always-Safe rivals, and fourteen of the
      fifteen route-by-risk cells follow the weak ordering.  The one reversal
      occurs at a ceiling, where a conditional rival reaches 100 per cent.

  2.  The MAGNITUDE does not.  How far the rival's stance moves a route is a
      property of the checkpoint.  The figure puts all five routes on one scale
      so a reader can see both the common direction and the different sizes.

Panels
  a  The response surface, one facet per stated risk, four rivals across and
     the route's own Unsafe play in each tile. The rival order is explicit in
     the columns, so a left-to-right darkening shows the shared response while
     the tile values show route-level magnitude differences. A saturated tile
     is a boundary measurement, not a violation of the ordering.
  b  A compact effect matrix.  The explicit zero column is the no-change
     baseline; bordered blocks separate the rival effect at each risk from the
     stated-risk effect with the rival held Always Safe.  Every cell prints its
     paired point estimate, so magnitude is readable without CI strokes.
  c  A rate-and-gap table.  AS is the fixed Always Safe rival, SP is self-play,
     and Delta is the design gap.  The rate tiles retain the response-surface
     palette, while the bordered Delta tiles make the interaction context
     visible without connecting marks between estimates.

The safe-rival arm is NOT called exploitation, here or anywhere this campaign
is reported.  That name would assert that the route takes the opportunity a
non-punishing rival offers.  The two designs can move in either direction, so
the contrast measures how far the rival's stance moves the route, and it is
named for that.

What this does NOT show.  Five routes, one game, one prompt version, ten races
per cell.  Five commercial endpoints are not a sample from a population of
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

import matplotlib.colors as mcolors
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
# five this campaign covers with a complete grid.
ROUTES = [r for r in S.ADMITTED if r in {
    "google/gemini-3-flash-preview",
    "openai/gpt-5.4-2026-03-05",
    "openai/gpt-5.5-2026-04-23",
    "anthropic/claude-opus-5@default",
    "anthropic/claude-sonnet-5@default",
}]
# Every route this campaign is allowed to hold a cell for.  A cell belonging to
# anything else is an error rather than a route to skip: it would mean the
# collection wrote into the wrong tree.
FULL_ROSTER = set(S.ROUTE_ORDER)

# Ascending in what the routes do to them, so a left-to-right tile order IS the
# ordering.
ORDER = ["AS", "CS", "CAS", "AU"]
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

# A game-theory response matrix benefits from an action palette rather than a
# rainbow.  The low end is a Safe tint, the high end is the paper's Unsafe ink,
# and every cell carries its measured value so the scale survives greyscale.
RESPONSE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "safe_to_unsafe",
    ["#EAF5EF", "#F7F3E8", "#FAEEEE", "#B2182B"],
    N=256,
)
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


def _relative_luminance(rgba):
    """WCAG luminance for choosing readable text inside a response tile."""
    def linear(value):
        return value / 12.92 if value <= 0.04045 else ((value + 0.055) / 1.055) ** 2.4

    return (0.2126 * linear(rgba[0])
            + 0.7152 * linear(rgba[1])
            + 0.0722 * linear(rgba[2]))


def draw_response_matrix(ax, surface, risk, *, first=False):
    """Show the route's response as a clean strategy matrix."""
    values = np.asarray([
        [100 * surface[(route, strategy, risk)] for strategy in ORDER]
        for route in ROUTES
    ])
    image = ax.imshow(values, cmap=RESPONSE_CMAP, vmin=0, vmax=100,
                      aspect="auto", interpolation="nearest")
    for row in range(values.shape[0]):
        for col in range(values.shape[1]):
            rgba = image.cmap(image.norm(values[row, col]))
            ink = S.SURFACE if _relative_luminance(rgba) < 0.48 else S.INK
            ax.text(col, row, f"{values[row, col]:.0f}", ha="center", va="center",
                    fontsize=S.FS_NOTE, color=ink, fontweight="bold")
    ax.set_xticks(np.arange(len(ORDER)), [s for s in ORDER],
                  fontsize=S.FS_NOTE)
    ax.set_yticks(np.arange(len(ROUTES)),
                  [S.ROUTE_SHORT[route] if first else "" for route in ROUTES],
                  fontsize=S.FS_NOTE)
    if first:
        for tick, route in zip(ax.get_yticklabels(), ROUTES):
            tick.set_color(S.ROUTE_C[route])
            tick.set_fontweight("bold")
        ax.set_ylabel("route", labelpad=2)
    else:
        ax.tick_params(axis="y", length=0)
    ax.set_xlabel("scripted rival", labelpad=2)
    ax.set_xlim(-0.5, len(ORDER) - 0.5)
    ax.set_ylim(len(ROUTES) - 0.5, -0.5)
    ax.set_xticks(np.arange(len(ORDER) + 1) - 0.5, minor=True)
    ax.set_yticks(np.arange(len(ROUTES) + 1) - 0.5, minor=True)
    ax.grid(which="minor", color=S.SURFACE, linewidth=1.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(axis="both", length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return image


def _blend_with_white(colour: str, amount: float) -> str:
    """Mix a semantic ink with white for a readable categorical tile fill."""
    rgb = np.asarray(mcolors.to_rgb(colour), dtype=float)
    mixed = (1.0 - amount) + amount * rgb
    return mcolors.to_hex(mixed, keep_alpha=False)


def _effect_fill(value: float, *, colour: str, vmax: float,
                 negative_colour: str | None = None) -> str:
    """Map a signed effect to a lightness ramp, refusing hidden sign changes."""
    if not np.isfinite(value):
        raise ValueError(f"non-finite effect value {value!r}")
    if value < -1e-12:
        if negative_colour is None:
            raise ValueError(
                f"negative effect {value:.6f} has no declared negative fill family"
            )
        colour = negative_colour
        value = abs(value)
    amount = 0.16 + 0.66 * min(max(value / (vmax or 1.0), 0.0), 1.0)
    return _blend_with_white(colour, amount)


def _draw_tile(ax, x: float, y: float, value: float, *, face: str, edge: str,
               label: str, text_colour: str = S.INK, linewidth: float = 0.8,
               width: float = 0.92, height: float = 0.82) -> None:
    """Draw one measured value as a bordered tile, never as an interval mark."""
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((x - width / 2, y - height / 2), width, height,
                           facecolor=face, edgecolor=edge, linewidth=linewidth,
                           joinstyle="round", zorder=2))
    ax.text(x, y, label, ha="center", va="center", fontsize=S.FS_NOTE,
            color=text_colour, fontweight="bold", zorder=3)


def _effect_text(value: float) -> str:
    """Signed point estimate, with one decimal retained from the artifact."""
    return f"{100 * value:+.1f}"


def _rate_text(value: float) -> str:
    """Unsafe-play rate in percentage points, rounded only for the display."""
    return f"{100 * value:.0f}"


def _route_rows(ax, *, y_positions) -> None:
    for y, route in zip(y_positions, ROUTES):
        ax.text(-0.65, y, S.ROUTE_SHORT[route], ha="right", va="center",
                fontsize=S.FS_NOTE, color=S.ROUTE_C[route], fontweight="bold",
                clip_on=False)


def draw_effect_matrix(ax, stance, risk_stance):
    """Categorical matrix of rival and risk effects with an explicit zero tile.

    The point estimates are the same paired contrasts used by the former
    interval panel.  Rival effects have red-family fills and borders; the
    stated-risk effect has a neutral ink family.  The separate headers and
    framed blocks make the estimands distinct without CI strokes or connectors.
    """
    from matplotlib.patches import Rectangle

    y_positions = np.arange(len(ROUTES), dtype=float)
    zero_x = 0.0
    rival_x = np.arange(1.0, 4.0)
    risk_x = 4.45
    rival_values = np.asarray([
        [100 * stance[(route, risk)][0] for risk in S.RISKS]
        for route in ROUTES
    ])
    risk_values = np.asarray([
        100 * risk_stance[route]["mean_difference"] for route in ROUTES
    ])
    if rival_values.shape != (len(ROUTES), len(S.RISKS)):
        raise RuntimeError(f"rival effect matrix has shape {rival_values.shape}")
    if risk_values.shape != (len(ROUTES),):
        raise RuntimeError(f"risk effect vector has shape {risk_values.shape}")
    if np.any(~np.isfinite(rival_values)) or np.any(~np.isfinite(risk_values)):
        raise RuntimeError("effect matrix contains a non-finite value")

    _route_rows(ax, y_positions=y_positions)
    for y in y_positions:
        _draw_tile(ax, zero_x, y, 0.0, face=S.BAND, edge=S.HAIRLINE,
                   label="0", linewidth=0.9)
    rival_max = float(np.max(rival_values))
    risk_max = float(np.max(risk_values))
    for row, route in enumerate(ROUTES):
        for col, value in zip(rival_x, rival_values[row]):
            _draw_tile(ax, col, row, value,
                       face=_effect_fill(value, colour=S.UNSAFE_C, vmax=rival_max),
                       edge=S.UNSAFE_C, label=f"+{value:.1f}",
                       text_colour=S.INK, linewidth=0.9)
        value = risk_values[row]
        _draw_tile(ax, risk_x, row, value,
                   face=_effect_fill(value, colour=S.INK_2, vmax=risk_max),
                   edge=S.INK_2, label=f"+{value:.1f}",
                   text_colour=S.INK, linewidth=1.1)

    # Framed blocks are the categorical separator: the red block is a change
    # in rival stance, the graphite block is a change in stated risk.
    ax.add_patch(Rectangle((0.52, -0.48), 2.96, len(ROUTES) - 0.04,
                           fill=False, edgecolor=S.UNSAFE_C, linewidth=1.1,
                           zorder=4, clip_on=False))
    ax.add_patch(Rectangle((3.96, -0.48), 0.98, len(ROUTES) - 0.04,
                           fill=False, edgecolor=S.INK_2, linewidth=1.1,
                           zorder=4, clip_on=False))
    ax.text(2.0, -0.92, "rival effect", ha="center", va="center",
            fontsize=S.FS_NOTE, color=S.UNSAFE_C, fontweight="bold")
    ax.text(risk_x, -0.92, "risk effect", ha="center", va="center",
            fontsize=S.FS_NOTE, color=S.INK_2, fontweight="bold")

    ax.set_xlim(-0.85, 4.95)
    ax.set_ylim(len(ROUTES) - 0.48, -1.18)
    ax.set_xticks([zero_x, *rival_x, risk_x])
    ax.set_xticklabels(["0\nbaseline", "risk 0.1", "risk 0.6", "risk 0.9",
                        "AS: 0.1\nminus 0.9"], fontsize=S.FS_NOTE)
    ax.tick_params(axis="both", length=0)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel("paired change in Unsafe play (percentage points)", labelpad=8)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_context_matrix(ax, safe_arm, selfplay, censored):
    """Aligned rate tiles: baseline design, self-play, and their gap."""
    from matplotlib.patches import Rectangle

    y_positions = np.arange(len(ROUTES), dtype=float)
    x_positions = np.arange(9, dtype=float)
    all_rates = [entry[0] for table in (safe_arm, selfplay) for entry in table.values()]
    if len(all_rates) != 2 * len(ROUTES) * len(S.RISKS):
        raise RuntimeError("context tile input is not a complete rate grid")
    gap_values = {
        (route, risk): selfplay[(route, risk)][0] - safe_arm[(route, risk)][0]
        for route in ROUTES for risk in S.RISKS
    }
    if any(not np.isfinite(value) for value in gap_values.values()):
        raise RuntimeError("context matrix contains a non-finite gap")
    max_rate = max(all_rates)
    max_gap = max(abs(value) for value in gap_values.values())
    _route_rows(ax, y_positions=y_positions)

    for group, risk in enumerate(S.RISKS):
        base = group * 3
        for row, route in enumerate(ROUTES):
            safe_value = safe_arm[(route, risk)][0]
            self_value = selfplay[(route, risk)][0]
            gap_value = gap_values[(route, risk)]
            _draw_tile(ax, base, row, safe_value,
                       face=RESPONSE_CMAP(safe_value), edge=S.HAIRLINE,
                       label=_rate_text(safe_value),
                       text_colour=(S.SURFACE if _relative_luminance(RESPONSE_CMAP(safe_value)) < 0.48
                                    else S.INK), linewidth=0.7, width=0.86)
            _draw_tile(ax, base + 1, row, self_value,
                       face=RESPONSE_CMAP(self_value), edge=S.UNSAFE_C,
                       label=_rate_text(self_value),
                       text_colour=(S.SURFACE if _relative_luminance(RESPONSE_CMAP(self_value)) < 0.48
                                    else S.INK), linewidth=1.0, width=0.86)
            gap_label = _effect_text(gap_value)
            if censored[(route, risk)][0]:
                gap_label = ">=" + gap_label
            _draw_tile(ax, base + 2, row, gap_value,
                       face=_effect_fill(100 * gap_value, colour=S.UNSAFE_C,
                                         negative_colour=S.SAFE_C,
                                         vmax=100 * max_gap),
                       edge=S.UNSAFE_C, label=gap_label, linewidth=0.9, width=0.86)
        ax.add_patch(Rectangle((base - 0.47, -0.48), 2.94, len(ROUTES) - 0.04,
                               fill=False, edgecolor=S.HAIRLINE, linewidth=0.9,
                               zorder=4, clip_on=False))
        ax.text(base + 1, -0.92, f"risk {S.RISK_LABEL[risk]}", ha="center",
                va="center", fontsize=S.FS_NOTE, color=S.INK_2, fontweight="bold")

    ax.set_xlim(-0.82, 8.48)
    ax.set_ylim(len(ROUTES) - 0.48, -1.18)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(["AS", "SP", "Δ"] * len(S.RISKS), fontsize=S.FS_NOTE)
    ax.set_yticks(y_positions)
    ax.set_yticklabels([])
    ax.set_xlabel("Unsafe play (%)", labelpad=8)
    ax.tick_params(axis="both", length=0)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_visible(False)


def main() -> None:
    turns = load_campaign()
    route_rows = turns[turns["is_route_decision"]].copy()
    script = turns[~turns["is_route_decision"]]
    found = sorted(set(turns["route"]))
    if found != sorted(ROUTES):
        raise ValueError(f"expected {sorted(ROUTES)}, found {found}")
    cells = route_rows.groupby(["route", "strategy", "cell_risk"])
    if len(cells) != len(ROUTES) * len(ORDER) * len(S.RISKS):
        raise ValueError(f"{len(cells)} cells, not a complete five-route grid")
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
    violations = [(route, risk) for route in ROUTES for risk in S.RISKS
                  if (route, risk) not in weak]
    n_cells = len(ROUTES) * len(S.RISKS)
    print(f"  ordering AS < CS < CAS < AU: strict in {len(strict)}/{n_cells}, "
          f"weak in {len(weak)}/{n_cells}")
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
    design_reversals = [
        (route, risk) for route in ROUTES for risk in S.RISKS
        if selfplay[(route, risk)][0] <= safe_arm[(route, risk)][0]
    ]
    print(f"  self-play exceeds the safe-rival arm in "
          f"{len(safe_arm) - len(design_reversals)}/{len(safe_arm)} cells; "
          f"reversals {[(S.ROUTE_SHORT[r], k) for r, k in design_reversals]}")

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

    risk_stance = {
        route: paired_risk_contrast(route_rows, route, "AS", S.RISKS[0], S.RISKS[-1])
        for route in ROUTES
    }

    # The figure is rendered only after every source-data and paired-contrast
    # guard below passes.  That makes the redesign fail closed: no plausible
    # PNG/PDF is left behind when the artifact and the figure disagree.

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
    print(f"  rival lower bounds start at {rival_low:+.1f} pp and risk intervals "
          f"end at {risk_high:+.1f} pp; the two families overlap: "
          f"{rival_low <= risk_high}")
    ratios = [stance[(route, risk)][0] / risk_stance[route]["mean_difference"]
              for route in ROUTES for risk in S.RISKS]
    print(f"  the rival is worth {min(ratios):.1f} to {max(ratios):.1f} times the "
          f"risk across the fifteen route-by-risk cells")

    # The response surface makes the game-theory object explicit: columns are
    # ordered rival strategies and each tile is one measured route response.
    # The lower panels are categorical tables.  They intentionally show point
    # effects and rates as tiles, with zero/baseline references, rather than
    # adding interval strokes that make the estimands look like trajectories.
    fig = plt.figure(figsize=(S.TEXT, 4.08))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.28, 1.0],
                          left=0.105, right=0.975, top=0.855, bottom=0.18,
                          width_ratios=[1.0, 1.24], wspace=0.72, hspace=0.98)
    top = gs[0, :].subgridspec(1, len(S.RISKS), wspace=0.34)
    facets = [fig.add_subplot(top[0, i]) for i in range(len(S.RISKS))]
    for i, (ax, risk) in enumerate(zip(facets, S.RISKS)):
        draw_response_matrix(ax, surface, risk, first=(i == 0))
    S.panel(facets[0], "a", "rival stance orders play", pad=5, gap=8.5)
    for i, (ax, risk) in enumerate(zip(facets, S.RISKS)):
        ax.set_title(f"risk {S.RISK_LABEL[risk]}", loc="right" if i == 0 else "center",
                     fontsize=S.FS_CLAIM, color=S.INK, fontweight="bold", pad=26)

    ax_b = fig.add_subplot(gs[1, 0])
    draw_effect_matrix(ax_b, stance, risk_stance)
    S.panel(ax_b, "b", "paired effect magnitudes", pad=5, gap=8.5)

    ax_c = fig.add_subplot(gs[1, 1])
    draw_context_matrix(ax_c, safe_arm, selfplay, censored)
    S.panel(ax_c, "c", "baseline versus self-play", pad=5, gap=8.5)

    fig.text(0.50, 0.045,
             "B: 0 is no effect; red tiles change the rival from AS to AU, "
             "graphite tiles change risk at AS. C: Delta is self-play minus AS; "
             ">= marks a ceiling-censored gap.",
             ha="center", va="bottom", fontsize=S.FS_NOTE, color=S.MUTED)
    S.save(fig, "scripted_opponent", width=S.TEXT)

if __name__ == "__main__":
    main()
