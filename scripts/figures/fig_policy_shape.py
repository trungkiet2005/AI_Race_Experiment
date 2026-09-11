"""What the routes actually do, round by round, rather than how much.

Every headline number in this paper is a rate: the share of decisions that were
unsafe.  A rate is silent about the thing an iterated game is for.  Two routes
can sit at the same rate while one is reading the board and the other is
reciting a constant, and the manuscript has no object that tells them apart.
This figure is that object.

The read-out is the memory-one fingerprint of the evolutionary game theory
literature: the probability of playing Unsafe given the pair of moves that
preceded it, own move first, rival's second.  Four contexts, so four numbers per
route, and every canonical rule of that literature is a corner of the same
space.  The choice is not an analytical convenience.  The prompt shows the
players only the immediately preceding round, so memory-one is the whole state
the game makes available, and a fingerprint over those four contexts is a
complete description of a stationary policy rather than a summary of one.

Panel b reports Mantel-Haenszel differences stratified on the risk level and on
the other lagged move.  Both stratifications are load-bearing.  Risk has to be
held fixed because the three risk cells have very different base rates, and
pooling them lets a route that simply plays more unsafe at low risk look
responsive.  The other lagged move has to be held fixed because these are
self-play races: both companies are the same route, so a raw "what did it do
after the rival went unsafe" would largely measure the route's own phase.

Panels
  a  The fingerprint as a table of colour, one row per route, one column per
     context, the probability printed in every tile.  Row labels carry the
     route colours the rest of the paper uses, so the figure needs no legend.
     Claude Opus 5 has two empty cells, and they are the finding rather than
     missing data: in 30 races its two companies never once played differently,
     so the mixed contexts do not exist to be measured.
  b  Two lagged associations on one scale, stratified differences and not
     causal effects.  The rival term is how far unsafe play differs after the
     rival's unsafe move; the own term is how far it differs after the route's
     own.  Every route that responds to the rival at all also backs off after
     its own unsafe move, so the pattern is not momentum in either direction but
     matching the rival while correcting itself.
  c  One race from each extreme, drawn the way this literature draws example
     gameplay.  A responsive route and a locked one, at the same risk level,
     under the same protocol.

What this does not show.  Responding to the rival is not understanding the
rival, and nothing here licenses a claim about strategy, intent or reasoning.
These are conditional frequencies in one game at one prompt version.  Nine
commercial endpoints are not a sample from a population of models, so the split
between the routes that respond and the two that do not is a fact about these
nine.  The own term in particular is not a preference: after its own unsafe move
a route is also carrying more accumulated private risk, which is a real feature
of the game rather than a confound, but it means the term measures response to
the state and not to the move alone.
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

CTX = ["SS", "SU", "US", "UU"]
CTX_LABEL = {
    "SS": "safe\nsafe",
    "SU": "safe\nunsafe",
    "US": "unsafe\nsafe",
    "UU": "unsafe\nunsafe",
}
N_BOOT = 2000
SEED = 260726

RIVAL_C = S.UNSAFE_C
OWN_C = S.INK_2

# The two races panel c draws, named here so the figure cannot drift onto a
# different pair without the change showing up in the diff.
RESPONSIVE = ("google/gemini-3-flash-preview", 0.6, "rep-007")
LOCKED = ("anthropic/claude-opus-5@default", 0.6, "rep-007")


def prepare(turns: pd.DataFrame) -> pd.DataFrame:
    replies = turns[turns["own_prev_action"].notna()].copy()
    replies["own_prev"] = replies["own_prev_action"].str.upper()
    replies["opp_prev"] = replies["opponent_prev_action"].str.upper()
    replies["ctx"] = replies["own_prev"].str[0] + replies["opp_prev"].str[0]
    return replies


def mh_difference(block: pd.DataFrame, exposure: str, strata: list[str]) -> float:
    """Mantel-Haenszel risk difference, weighting each stratum by n1 n0 / (n1 + n0).

    A stratum that contains only one level of the exposure carries no
    information about it and is dropped rather than imputed.  For Claude Opus 5
    that empties every stratum, which is why its effect is undefined and not
    zero.
    """
    numerator = denominator = 0.0
    for _, cell in block.groupby(strata):
        exposed = cell[cell[exposure] == "UNSAFE"]
        unexposed = cell[cell[exposure] == "SAFE"]
        if exposed.empty or unexposed.empty:
            continue
        weight = len(exposed) * len(unexposed) / (len(exposed) + len(unexposed))
        numerator += weight * (exposed["unsafe"].mean() - unexposed["unsafe"].mean())
        denominator += weight
    return numerator / denominator if denominator else float("nan")


def effects(replies: pd.DataFrame) -> pd.DataFrame:
    """Both lagged effects per route, with a race-clustered interval.

    Races are resampled within their own risk cell, because the design fixes ten
    races per risk and a bootstrap that ignored that could return a sample with
    no races at some risk level at all.
    """
    rows = []
    rng = np.random.default_rng(SEED)
    for route, block in replies.groupby("model_route"):
        rival = mh_difference(block, "opp_prev", ["max_private_risk", "own_prev"])
        own = mh_difference(block, "own_prev", ["max_private_risk", "opp_prev"])
        record = {"model_route": route, "rival": rival, "own": own}
        if np.isnan(rival):
            record.update(rival_lo=np.nan, rival_hi=np.nan, own_lo=np.nan, own_hi=np.nan)
            rows.append(record)
            continue
        by_race = {game: part for game, part in block.groupby("game_id")}
        cells: dict[float, list[str]] = {}
        for game, part in by_race.items():
            cells.setdefault(float(part["max_private_risk"].iloc[0]), []).append(game)
        boots = {"rival": [], "own": []}
        for _ in range(N_BOOT):
            picked = []
            for games in cells.values():
                picked += list(rng.choice(games, size=len(games), replace=True))
            sample = pd.concat([by_race[game] for game in picked])
            for key, exposure, strata in (
                ("rival", "opp_prev", ["max_private_risk", "own_prev"]),
                ("own", "own_prev", ["max_private_risk", "opp_prev"]),
            ):
                value = mh_difference(sample, exposure, strata)
                if not np.isnan(value):
                    boots[key].append(value)
        for key in ("rival", "own"):
            record[f"{key}_lo"] = float(np.percentile(boots[key], 2.5))
            record[f"{key}_hi"] = float(np.percentile(boots[key], 97.5))
        rows.append(record)
    return pd.DataFrame(rows).set_index("model_route")


def race_trace(turns: pd.DataFrame, route: str, risk: float, rep: str) -> pd.DataFrame:
    block = turns[
        (turns["model_route"] == route)
        & (turns["max_private_risk"] == risk)
        & (turns["game_id"].str.endswith(rep))
    ]
    if block.empty:
        raise SystemExit(f"no race for {route} at risk {risk}, {rep}")
    return block.pivot_table(index="round", columns="player", values="unsafe")


def draw_trace(ax, trace, *, name, colour, risk, note, show_x):
    rounds = trace.index.to_numpy()
    for (player, series), off in zip(trace.items(), (0.055, -0.055)):
        ax.plot(rounds, series.to_numpy() + off, marker="o", ms=2.4, lw=0.95,
                color=colour if player.endswith("1") else S.MUTED,
                clip_on=False, zorder=3)
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["Safe", "Unsafe"])
    ax.set_ylim(-0.30, 1.30)
    ax.set_xlim(rounds.min() - 0.4, rounds.max() + 0.4)
    ax.set_xticks([r for r in rounds if r % 3 == 1])
    if show_x:
        ax.set_xlabel("Round")
    else:
        ax.set_xticklabels([])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.set_axisbelow(True)
    ax.grid(True, axis="y", zorder=0)
    ax.annotate(f"{name}, $p_r^{{\\max}}={risk}$", xy=(0, 1.0),
                xycoords="axes fraction", xytext=(0, 2.5),
                textcoords="offset points", ha="left", va="bottom",
                fontsize=S.FS_NOTE, color=colour, fontweight="bold",
                annotation_clip=False)
    if note:
        # Placed inside the axes, in the band this trace leaves empty, because
        # the strip above every panel is already spoken for.
        ax.annotate(note, xy=(0.5, 0.52), xycoords="axes fraction",
                    ha="center", va="center", fontsize=S.FS_NOTE,
                    color=S.MUTED, style="italic")


def main() -> None:
    turns = D.baseline_turns()
    replies = prepare(turns)
    print(f"  {turns['model_route'].nunique()} routes, {turns['game_id'].nunique()} races, "
          f"{len(turns)} decisions, {len(replies)} of them with a predecessor")

    fp = replies.groupby(["model_route", "ctx"])["unsafe"].mean().unstack().reindex(columns=CTX)
    eff = effects(replies)

    for route in eff.index[eff["rival"].isna()]:
        wide = turns[turns["model_route"] == route].pivot_table(
            index=["game_id", "round"], columns="player", values="unsafe")
        diverged = int((wide["Company_1"] != wide["Company_2"]).sum())
        print(f"  {S.ROUTE_LABEL[route]}: {diverged} of {len(wide)} rounds diverged")
    for route, row in eff.iterrows():
        print(f"  {S.ROUTE_SHORT[route]:>9}  rival {100 * row['rival']:6.1f} "
              f"[{100 * row['rival_lo']:6.1f}, {100 * row['rival_hi']:6.1f}]   "
              f"own {100 * row['own']:6.1f} "
              f"[{100 * row['own_lo']:6.1f}, {100 * row['own_hi']:6.1f}]")

    # One ordering for both panels: strongest rival response at the top, the
    # undefined route last.  Two orderings in one figure is one too many.
    order = list(eff["rival"].sort_values(ascending=False, na_position="last").index)

    fig = plt.figure(figsize=(S.TEXT, 2.72))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.30, 1.04, 1.02], wspace=0.60)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    inner = gs[0, 2].subgridspec(2, 1, hspace=0.42)
    ax_c1 = fig.add_subplot(inner[0])
    ax_c2 = fig.add_subplot(inner[1])

    # --- a: the fingerprint --------------------------------------------------
    arr = fp.reindex(order).to_numpy(dtype=float)
    im = S.heat_tiles(
        ax_a, np.ma.masked_invalid(arr),
        [S.ROUTE_SHORT[r] for r in order],
        [CTX_LABEL[c] for c in CTX],
        cmap="viridis", vmin=0.0, vmax=1.0, fmt="{:.2f}",
        row_colors=[S.ROUTE_C[r] for r in order],
    )
    im.cmap.set_bad(S.GREY_BAD)
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if np.isnan(arr[i, j]):
                ax_a.text(j, i, "never", ha="center", va="center",
                          fontsize=S.FS_NOTE, color=S.INK_2, style="italic")
    ax_a.set_xlabel("own last move, then the rival's", labelpad=1)
    S.panel(ax_a, "a", "one rate, several policies")
    cb = fig.colorbar(im, ax=ax_a, fraction=0.040, pad=0.03)
    cb.set_label("P(Unsafe)", fontsize=S.FS_NOTE, labelpad=1)
    cb.ax.tick_params(labelsize=S.FS_NOTE, length=1.6)
    cb.outline.set_visible(False)

    # --- b: the two lagged effects -------------------------------------------
    ypos = np.arange(len(order))[::-1]
    for y, route in zip(ypos, order):
        row = eff.loc[route]
        if np.isnan(row["rival"]):
            ax_b.annotate("neither term is defined", xy=(0.0, y),
                          xytext=(5, 0), textcoords="offset points",
                          ha="left", va="center", fontsize=S.FS_NOTE,
                          color=S.MUTED, style="italic")
            continue
        for key, colour, marker, dy in (("rival", RIVAL_C, "o", 0.17),
                                        ("own", OWN_C, "s", -0.17)):
            ax_b.plot([100 * row[f"{key}_lo"], 100 * row[f"{key}_hi"]],
                      [y + dy, y + dy], lw=1.0, color=colour,
                      solid_capstyle="round", zorder=3)
            S.dot(ax_b, 100 * row[key], y + dy, color=colour, marker=marker, size=13)
    S.zero_rule(ax_b, 0.0, vertical=True)
    ax_b.set_yticks(ypos)
    ax_b.set_yticklabels([S.ROUTE_SHORT[r] for r in order])
    for tick, route in zip(ax_b.get_yticklabels(), order):
        tick.set_color(S.ROUTE_C[route])
    ax_b.set_ylim(-0.7, len(order) + 0.35)
    ax_b.set_xlim(-62, 62)
    ax_b.set_xticks([-50, -25, 0, 25, 50])
    # "effect on P(Unsafe)" stood here.  These are Mantel-Haenszel differences
    # inside a stratum on self-play races, with nothing randomised, so the axis
    # says difference and the caption says association.
    ax_b.set_xlabel("stratified difference in P(Unsafe), percentage points")
    S.strip(ax_b, grid_axis="x")
    # "match the rival, correct yourself" stood here and was not what the panel
    # draws.  Six of the nine rival terms are positive, two are negative, and one
    # route has no term at all; a title that says "match the rival" contradicts
    # the two rows pointing the other way in the same panel.
    S.panel(ax_b, "b", "six routes follow the rival, two lean against it")
    key_y = len(order) - 0.35
    S.direct_label(ax_b, 6, key_y, "the rival's last move", color=RIVAL_C,
                   dx=0, weight="bold")
    S.direct_label(ax_b, -6, key_y, "its own last move", color=OWN_C,
                   ha="right", dx=0, weight="bold")

    # --- c: one race from each extreme ---------------------------------------
    locked_wide = turns[turns["model_route"] == LOCKED[0]].pivot_table(
        index=["game_id", "round"], columns="player", values="unsafe")
    locked_diverged = int((locked_wide["Company_1"] != locked_wide["Company_2"]).sum())

    draw_trace(ax_c1, race_trace(turns, *RESPONSIVE),
               name=S.ROUTE_LABEL[RESPONSIVE[0]], colour=S.ROUTE_C[RESPONSIVE[0]],
               risk=RESPONSIVE[1], note="", show_x=False)
    draw_trace(ax_c2, race_trace(turns, *LOCKED),
               name=S.ROUTE_LABEL[LOCKED[0]], colour=S.ROUTE_C[LOCKED[0]],
               risk=LOCKED[1],
               note=f"the two seats differ in {locked_diverged} of this\n"
                    f"route's {len(locked_wide)} rounds",
               show_x=True)
    # pad was 14, and the superscript of the route label below it reached into
    # the claim between two words.  The claim sits clear of the label's tallest
    # glyph now rather than of its baseline.
    S.panel(ax_c1, "c", "one race from each extreme", pad=21)

    S.save(fig, "policy_shape", width=S.TEXT)


if __name__ == "__main__":
    main()
