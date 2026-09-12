"""Why this race, and what the paper does to it.

The opening figure of a paper answers one question: why should a reader care
about the object on the next seven pages.  The manuscript's previous opening
figure answered a different one.  It drew the mechanism three times, as a
repeated race, as a two-player payoff matrix and as an N-player payoff rule,
which is a specification rather than a motivation, and it spent a third of its
width on the N-player rule, which this paper now reports in the supplementary
material.  It was also drawn in a second visual language: clip-art buildings,
runners and a trophy, at 19 inches wide, scaled to 37 per cent on the page, so
its lettering printed at nearly ten points beside the six-point lettering of
every other figure in the paper.  None of that is fixable by editing the file,
so this is a new figure and the old one is no longer part of the set.

What it draws, left to right.

  a  The scenario the paper is about.  A choice has to be made in a race
     between developers, and it is handed to a language model, which sees the
     rules, the danger, the rival's last move and the running score, and
     returns one of two actions.  The paper is prospective: it asks what a
     delegate would choose, never who delegates today.
  b  What the two actions do.  Safe advances one step and adds no risk; Unsafe
     advances one and a half and raises the chance that a win is wiped out.
     The horizon and the prize are the two other rules a reader needs.
  c  The stage payoffs.  These are worth drawing because they are not the
     prisoner's dilemma a reader of this literature will expect.  Racing pays
     more than restraint against either rival move, 2.4 against 1.0 and 2.0
     against 0.6, and mutual racing also pays more than mutual restraint, so
     Unsafe strictly dominates inside a round and the stage game carries no
     tension at all.  The whole cost of racing sits in panel b: a setback wipes
     out the winner's entire payoff, and its chance rises with the share of
     one's own moves that were Unsafe.  That is the structure the paper's
     results are about, and a figure that drew a dilemma here would be drawing
     a different game.
  d  The three lenses the paper applies to that race, in the order the results
     take them: how far the stated danger moves the play, how far the rival
     moves it, and whether either shape is the one people or the evolutionary
     benchmark show.

Every quantity drawn here is read off an artefact and checked before anything
is drawn.  The mechanism numbers come out of the prompt text the players were
actually sent, parsed from the turn records of all nine routes rather than from
a protocol document, because the prompt is what the model saw and a document is
what someone meant to send.  The risk levels come from the same records, the
scripted-rival grid from the campaign's own derived tables, and the human
sample size from the source study's participant file.  If any of them stops
matching, the script refuses to draw rather than shipping a schematic that has
quietly become fiction.

What this figure is not.  It is a picture of the design, so nothing in it is a
result: no rate, no interval and no comparison appears here.  And the race is
an idealised game with two actions, a fixed prize and a stated risk; it is a
controlled instrument for studying choice under competitive pressure and it
models no real development programme.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import figstyle as S  # noqa: E402
import figdata as D  # noqa: E402

HUMAN_CSV = D.ROOT / "references" / "source_study_dataset" / "airace_deidentified_long.csv"
SCRIPTED_DERIVED = D.ROOT / "results" / "derived" / "scripted_opponent_campaign"

# The rival strategies the scripted campaign ran, in the order the route's own
# Unsafe play rises through them, with the plain-English name the paper uses.
RIVALS = [("AS", "Always Safe"), ("CS", "Cond. Safe"),
          ("CAS", "Cond. Unsafe"), ("AU", "Always Unsafe")]

# Each entry is a phrase that must appear verbatim in the prompt every player
# read, and the name this figure gives the quantity it carries.  Parsing the
# prompt rather than trusting a constant is what stops this figure from drawing
# a game the players were never actually sent.
MECHANISM = {
    "safe_step": r"SAFE advances your company (\d+(?:\.\d+)?) race step",
    "unsafe_step": r"UNSAFE advances your company (\d+(?:\.\d+)?) race steps",
    "both_safe": r"If you both choose SAFE, you both get (\d+(?:\.\d+)?)",
    "sucker": r"you get (\d+(?:\.\d+)?) and Company_2 gets \d+(?:\.\d+)?\.",
    "temptation": r"you get \d+(?:\.\d+)? and Company_2 gets (\d+(?:\.\d+)?)\.",
    "both_unsafe": r"If you both choose UNSAFE, you both get (\d+(?:\.\d+)?)",
    "min_rounds": r"The race lasts at least (\d+) rounds",
    "stop_pct": r"the race ends with probability (\d+)%",
    "prize": r"receives a prize of (\d+)",
    "tie_prize": r"each company receives (\d+)",
}


def mechanism_from_prompts() -> dict[str, float]:
    """The game as the players were sent it, identical on all nine routes.

    One prompt per route is parsed, not one prompt in total, because a figure
    that reads a single record cannot tell a shared mechanism from a lucky one.
    A disagreement between two routes would mean the nine were not playing the
    same game, which would invalidate every comparison in the paper and is
    therefore raised here rather than absorbed.
    """
    seen: dict[str, set[str]] = {key: set() for key in MECHANISM}
    routes = set()
    for manifest_path in sorted(
        D.BASELINE_ROOT.glob("*/*/*/results/ai_race_baseline/run_manifest.json")
    ):
        if "failed_runs" in manifest_path.parts:
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("status") != "completed":
            continue
        routes.add(manifest["model_route"])
        with manifest_path.with_name("turns.jsonl").open(encoding="utf-8") as handle:
            prompt = json.loads(handle.readline())["prompt"]
        for key, pattern in MECHANISM.items():
            match = re.search(pattern, prompt)
            if match is None:
                raise SystemExit(
                    f"{manifest['model_route']} was sent a prompt with no {key}: "
                    "this figure draws the mechanism out of the prompt, so it "
                    "cannot draw one the prompt does not state"
                )
            seen[key].add(match.group(1))
    if routes != set(S.ROUTE_ORDER):
        raise SystemExit(f"the baseline holds {sorted(routes)}, not the nine routes")
    disagreeing = {key: sorted(values) for key, values in seen.items() if len(values) != 1}
    if disagreeing:
        raise SystemExit(
            f"the nine routes were not sent one mechanism: {disagreeing}"
        )
    return {key: float(next(iter(values))) for key, values in seen.items()}


def lens_facts() -> dict[str, object]:
    """The size of each of the three lenses, from the campaign's own tables."""
    turns = D.baseline_turns()
    risks = sorted(float(level) for level in turns["max_private_risk"].unique())
    if risks != list(S.RISKS):
        raise SystemExit(f"the baseline ran risks {risks}, not {list(S.RISKS)}")
    races = turns.groupby("model_route")["game_id"].nunique()
    if races.nunique() != 1:
        raise SystemExit(f"the routes played different race counts: {races.to_dict()}")

    # A route that holds part of the grid carries neither an ordering nor a
    # paired contrast, so the campaign figure leaves it out.  Counting it here
    # would put a number on the opening page that the results never analyse.
    full_grid = len(RIVALS) * len(S.RISKS)
    cells, strategies, scripted_routes, partial = 0, set(), 0, 0
    for path in sorted(SCRIPTED_DERIVED.rglob("scripted_opponent_rates.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("refused"):
            raise SystemExit(f"{path} records a refusal: {payload['refused']}")
        if int(payload["n_cells"]) != full_grid:
            partial += 1
            continue
        cells += int(payload["n_cells"])
        scripted_routes += 1
        strategies |= {row["opponent_strategy"] for row in payload["rates"]}
    if strategies != {code for code, _ in RIVALS}:
        raise SystemExit(f"the campaign ran rivals {sorted(strategies)}, not {RIVALS}")
    if partial:
        print(f"  {partial} route(s) hold part of the scripted grid and are counted "
              f"neither here nor in the campaign figure")

    raw = pd.read_csv(HUMAN_CSV, usecols=["participant_id", "round_number"])
    complete = (
        raw[raw["round_number"].between(1, 5)]
        .groupby("participant_id")["round_number"]
        .nunique()
    )
    humans = int((complete == 5).sum())

    return {
        "risks": risks,
        "routes": len(races),
        "races": int(races.iloc[0]),
        "rival_cells": cells,
        "rival_routes": scripted_routes,
        "humans": humans,
    }


# --- drawing helpers ---------------------------------------------------------

def blank(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    return ax


def chip(ax, x, y, w, h, text, *, ec, fc, fs=None, weight="normal", tc=None,
         radius=0.035, lw=0.7, align="center"):
    """A rounded label box.  Every box in this figure is one of these."""
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle=f"round,pad=0.004,rounding_size={radius}",
        facecolor=fc, edgecolor=ec, linewidth=lw, zorder=2, clip_on=False))
    tx = x + w / 2 if align == "center" else x + 0.035
    ax.text(tx, y + h / 2, text, ha="center" if align == "center" else "left",
            va="center", fontsize=fs or S.FS_NOTE, color=tc or ec,
            fontweight=weight, zorder=3, linespacing=1.30)


def arrow(ax, x0, y0, x1, y1, *, color=None, lw=0.8, head=2.6):
    ax.add_patch(FancyArrowPatch(
        (x0, y0), (x1, y1), arrowstyle=f"-|>,head_width={head / 10:.2f},"
                                       f"head_length={head / 7:.2f}",
        color=color or S.INK_2, lw=lw, shrinkA=0, shrinkB=0, zorder=4,
        mutation_scale=6, clip_on=False))


def tint(colour: str, amount: float = 0.90) -> str:
    """A pale fill of a stroke colour, so a box is never filled with its own ink."""
    import matplotlib.colors as mcolors

    r, g, b = mcolors.to_rgb(colour)
    return mcolors.to_hex(tuple(c + (1.0 - c) * amount for c in (r, g, b)))


SAFE_FILL = tint(S.SAFE_C)
UNSAFE_FILL = tint(S.UNSAFE_C)
INK_FILL = "#f2f4f6"


# --- panels ------------------------------------------------------------------

def panel_delegation(ax):
    """The scenario, top to bottom: what is seen, who decides, what comes back."""
    blank(ax)
    ax.text(0.50, 0.97, "what the model is told", ha="center", va="top",
            fontsize=S.FS_NOTE, color=S.MUTED)
    chip(ax, 0.02, 0.52, 0.96, 0.36,
         "the rules and the payoffs\nhow dangerous the race is\n"
         "the rival's last move\nthe score so far",
         ec=S.HAIRLINE, fc=S.SURFACE, tc=S.INK_2, radius=0.025)
    arrow(ax, 0.50, 0.505, 0.50, 0.415)
    chip(ax, 0.10, 0.22, 0.80, 0.18, "a language model\nchooses for it",
         ec=S.INK, fc=INK_FILL, tc=S.INK, weight="bold")
    arrow(ax, 0.34, 0.205, 0.24, 0.145)
    arrow(ax, 0.66, 0.205, 0.76, 0.145)
    chip(ax, 0.01, 0.00, 0.46, 0.13, "SAFE", ec=S.SAFE_C, fc=SAFE_FILL,
         weight="bold", fs=S.FS_CLAIM)
    chip(ax, 0.53, 0.00, 0.46, 0.13, "UNSAFE", ec=S.UNSAFE_C, fc=UNSAFE_FILL,
         weight="bold", fs=S.FS_CLAIM)
    S.panel(ax, "a", "a race decision, delegated", pad=11, gap=8.0)


def panel_actions(ax, mech):
    """What each action buys and what it costs."""
    blank(ax)
    ax.text(0.02, 0.98, f"SAFE   +{mech['safe_step']:g} step, no added risk",
            fontsize=S.FS_NOTE, color=S.SAFE_C, fontweight="bold", va="top")
    safe_end = 0.02 + (0.96 * mech["safe_step"] / mech["unsafe_step"])
    ax.plot([0.02, 0.98], [0.845, 0.845], color=S.HAIRLINE, lw=0.7, zorder=1)
    arrow(ax, 0.02, 0.845, safe_end, 0.845, color=S.SAFE_C, lw=1.6)

    ax.text(0.02, 0.775, f"UNSAFE   +{mech['unsafe_step']:g} steps, and a setback",
            fontsize=S.FS_NOTE, color=S.UNSAFE_C, fontweight="bold", va="top")
    ax.text(0.02, 0.685, "can wipe out the whole payoff",
            fontsize=S.FS_NOTE, color=S.UNSAFE_C, fontweight="bold", va="top")
    ax.plot([0.02, 0.98], [0.545, 0.545], color=S.HAIRLINE, lw=0.7, zorder=1)
    arrow(ax, 0.02, 0.545, 0.98, 0.545, color=S.UNSAFE_C, lw=1.6)

    ax.text(0.02, 0.44,
            "a setback's chance is the stated\n"
            "maximum times the share of its own\n"
            "moves that were Unsafe",
            fontsize=S.FS_NOTE, color=S.INK_2, va="top", linespacing=1.45)
    ax.text(0.02, 0.17,
            f"at least {mech['min_rounds']:g} rounds, then a "
            f"{mech['stop_pct']:g}% chance of\n"
            f"ending after each; {mech['prize']:g} to the leader\n"
            f"at the end, {mech['tie_prize']:g} each if the two tie",
            fontsize=S.FS_NOTE, color=S.MUTED, va="top", linespacing=1.45)
    S.panel(ax, "b", "speed bought with exposure", pad=11, gap=8.0)


def panel_payoffs(ax, mech):
    """The stage payoff matrix, square, own action down and the rival across."""
    blank(ax)
    values = [[mech["both_safe"], mech["sucker"]],
              [mech["temptation"], mech["both_unsafe"]]]
    x0, y0, side = 0.40, 0.20, 0.30
    for i in range(2):
        for j in range(2):
            edge = S.SAFE_C if i == 0 else S.UNSAFE_C
            ax.add_patch(Rectangle(
                (x0 + j * side, y0 + (1 - i) * side), side, side,
                facecolor=SAFE_FILL if i == 0 else UNSAFE_FILL,
                edgecolor=edge, linewidth=0.7, zorder=2))
            ax.text(x0 + (j + 0.5) * side, y0 + (1.5 - i) * side,
                    f"{values[i][j]:g}", ha="center", va="center",
                    fontsize=S.FS_CLAIM, color=S.INK, fontweight="bold", zorder=3)
    ax.text(x0 + side, y0 + 2 * side + 0.16, "rival", ha="center", va="bottom",
            fontsize=S.FS_NOTE, color=S.MUTED)
    for j, (name, colour) in enumerate((("Safe", S.SAFE_C), ("Unsafe", S.UNSAFE_C))):
        ax.text(x0 + (j + 0.5) * side, y0 + 2 * side + 0.04, name, ha="center",
                va="bottom", fontsize=S.FS_NOTE, color=colour)
    ax.text(x0 - 0.32, y0 + side, "own", rotation=90, ha="center", va="center",
            fontsize=S.FS_NOTE, color=S.MUTED)
    for i, (name, colour) in enumerate((("Safe", S.SAFE_C), ("Unsafe", S.UNSAFE_C))):
        ax.text(x0 - 0.04, y0 + (1.5 - i) * side, name, ha="right", va="center",
                fontsize=S.FS_NOTE, color=colour)
    # The claim lives under the matrix rather than in the panel title.  At this
    # width a title long enough to carry it runs past the panel and lands on the
    # next panel's letter, and a claim that has to be shortened to fit is a claim
    # that loses the words "this round", which are the whole of its meaning.
    ax.text(0.50, 0.14, "what it earns this round", ha="center", va="top",
            fontsize=S.FS_NOTE, color=S.MUTED)
    ax.text(0.50, 0.04, "Unsafe pays more, either way", ha="center", va="top",
            fontsize=S.FS_NOTE, color=S.INK, fontweight="bold")
    S.panel(ax, "c", "the stage payoffs", pad=11, gap=8.0)


def panel_lenses(ax, facts):
    """The three questions the results answer, in the order they answer them."""
    blank(ax)
    levels = ", ".join(S.RISK_LABEL[r] for r in facts["risks"])
    rows = [
        ("how far does the stated danger move it?",
         f"the maximum is set to {levels},\n"
         f"against a copy of the same model, {facts['routes']} of them"),
        ("how far does the rival move it?",
         "the rival becomes fixed code it can neither\n"
         f"influence nor be told about, {len(RIVALS)} of them"),
        ("do people or theory show either shape?",
         f"the same race played by {facts['humans']} people, and the\n"
         "evolutionary benchmark built for it"),
    ]
    top = 0.92
    gap = 0.335
    for i, (question, detail) in enumerate(rows):
        y = top - i * gap
        ax.plot([0.012, 0.012], [y - 0.215, y + 0.015], color=S.HAIRLINE, lw=1.8,
                solid_capstyle="butt", zorder=1)
        ax.text(0.055, y, question, fontsize=S.FS_NOTE, color=S.INK,
                fontweight="bold", va="top")
        ax.text(0.055, y - 0.090, detail, fontsize=S.FS_NOTE, color=S.MUTED,
                va="top", linespacing=1.45)
    S.panel(ax, "d", "three lenses on one race", pad=11, gap=8.0)


def main() -> None:
    mech = mechanism_from_prompts()
    facts = lens_facts()

    print("  the mechanism, parsed from the prompt every player read:")
    for key in MECHANISM:
        print(f"    {key:<12} {mech[key]:g}")
    # Panel c claims that Unsafe strictly dominates inside a round, which is a
    # statement about both columns of the matrix and not about its ordering as
    # a whole.  Check the claim that is drawn.
    if not (mech["temptation"] > mech["both_safe"]
            and mech["both_unsafe"] > mech["sucker"]):
        raise SystemExit(
            "UNSAFE no longer earns more than SAFE against both rival moves, so "
            "panel c's claim that racing pays more this round whatever the rival "
            "does is not what the players were sent"
        )
    if mech["both_unsafe"] <= mech["both_safe"]:
        raise SystemExit(
            "mutual racing no longer pays more than mutual restraint; this game "
            "has become a prisoner's dilemma and the figure's note that it is "
            "not one would be false"
        )
    if mech["unsafe_step"] <= mech["safe_step"]:
        raise SystemExit("UNSAFE no longer advances further than SAFE")
    print(f"  lenses: {facts['routes']} routes x {facts['races']} races at risks "
          f"{facts['risks']}; {facts['rival_cells']} scripted cells on "
          f"{facts['rival_routes']} routes; {facts['humans']} complete human "
          f"five-round trajectories")

    fig = plt.figure(figsize=(S.TEXT, 1.92))
    gs = fig.add_gridspec(1, 4, width_ratios=[0.96, 1.12, 0.92, 1.36], wspace=0.16)
    panel_delegation(fig.add_subplot(gs[0, 0]))
    panel_actions(fig.add_subplot(gs[0, 1]), mech)
    panel_payoffs(fig.add_subplot(gs[0, 2]), mech)
    panel_lenses(fig.add_subplot(gs[0, 3]), facts)

    S.save(fig, "delegation_overview", width=S.TEXT)


if __name__ == "__main__":
    main()
