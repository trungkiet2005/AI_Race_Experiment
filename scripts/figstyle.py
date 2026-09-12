"""Publication style for the AI Race manuscript, second generation.

Why a second module exists rather than an edit to ``publication_style.py``.
That module declares ``TEXT_WIDTH_IN = 5.17`` and enforces an 8.0 pt floor on
every string it draws.  The AAMAS class was asked directly, by compiling a
document with the paper's own ``\\documentclass[sigconf]{aamas}`` and printing
``\\the\\columnwidth`` after ``\\maketitle``, and the answer is 241.147 pt, which
is 3.337 in.  A figure drawn 5.17 in wide and placed at ``\\columnwidth`` is
therefore scaled to 64%, and the 8.0 pt tick labels the floor guarantees print
at 5.11 pt.  The guard is real; it simply guards the drawn size when the thing
that matters is the printed size.  Two of the manuscript's four main figures are
affected, at 63.9% and 64.5%.

So geometry here comes from the class, and ``save`` fits the canvas to the
declared width so that a figure renders 1:1 and its type is the size it says.
The old module keeps working for the figures that already import it; nothing is
broken by adding this one.

Colour.  Okabe-Ito, the palette the evolutionary game theory literature has
settled on, which stays separable under deuteranopia, protanopia and
tritanopia.  Identity is never carried by hue alone: every route also owns a
marker and a short label, so a figure survives greyscale print and a reader who
has learnt the routes in one figure reads every later figure for free.

The relief rule.  Hue is decoration, never the only channel.  Every
multi-series panel carries a direct label, a distinct marker, or a facet title
in the series colour.

Two actions, two inks, used for nothing else in the paper: SAFE and UNSAFE.  A
reader should never have to check a legend to know which is which.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
REPO = HERE.parent
FIGDIR = Path(os.environ.get("AIRACE_FIGDIR", REPO / "figures" / "paper"))

# --- geometry, measured from the class rather than guessed --------------------
# $ pdflatex on \documentclass[sigconf,anonymous]{aamas} + \maketitle:
#     \columnwidth = 241.14749pt      \textwidth = 506.295pt
PT = 1 / 72.27
COL = 241.14749 * PT            # 3.3368 in, one column
TEXT = 506.295 * PT             # 7.0056 in, both columns
HALF_COL = (COL - 0.10) / 2     # two panels inside one column
TWO_THIRD = TEXT * 2 / 3
DPI = 600
PAD = 0.012

# --- ink ----------------------------------------------------------------------
INK = "#101418"
INK_2 = "#454f59"
MUTED = "#8b949d"
HAIRLINE = "#c8ced4"
GRID = "#dfe4e8"
SURFACE = "#ffffff"
BAND = "#eef1f4"
# A tile that has no value because the context never arose, which is a finding
# and not missing data.  Kept distinct from every colour in the sequential map
# so it cannot be misread as a low value.
GREY_BAD = "#e8ebee"

# --- the two actions ----------------------------------------------------------
# One ink each, used nowhere else, so the reader learns them once.
SAFE_C = "#1b7837"
UNSAFE_C = "#b2182b"
ACTION_C = {"SAFE": SAFE_C, "UNSAFE": UNSAFE_C}
ACTION_LABEL = {"SAFE": "Safe", "UNSAFE": "Unsafe"}

# --- route identity (Okabe-Ito) ----------------------------------------------
# Assigned once, in the order the manuscript always lists them, so a route keeps
# its hue and its glyph in every figure of the paper and of the supplement.
ROUTE_C = {
    "anthropic/claude-opus-5@default": "#d55e00",              # vermillion
    "anthropic/claude-sonnet-5@default": "#e69f00",            # orange
    "google/gemini-3-flash-preview": "#009e73",                # bluish green
    "google/gemini-3.1-flash-lite-preview": "#56b4e9",         # sky blue
    "google/gemini-3.5-flash-lite": "#0072b2",                 # blue
    "openai/gpt-5.5-2026-04-23": "#cc79a7",                    # reddish purple
    "openai/gpt-5.4-2026-03-05": "#5d3a9b",                    # violet
    "openai/gpt-5.4-mini-2026-03-17": "#8b6f47",               # brown
    "openai/gpt-5.4-nano-2026-03-17": "#7f7f7f",               # grey
    "human": INK,
}
ROUTE_M = {
    "anthropic/claude-opus-5@default": "o",
    "anthropic/claude-sonnet-5@default": "s",
    "google/gemini-3-flash-preview": "D",
    "google/gemini-3.1-flash-lite-preview": "^",
    "google/gemini-3.5-flash-lite": "v",
    "openai/gpt-5.5-2026-04-23": "P",
    "openai/gpt-5.4-2026-03-05": "X",
    "openai/gpt-5.4-mini-2026-03-17": "<",
    "openai/gpt-5.4-nano-2026-03-17": ">",
    "human": "*",
}
ROUTE_LABEL = {
    "anthropic/claude-opus-5@default": "Claude Opus 5",
    "anthropic/claude-sonnet-5@default": "Claude Sonnet 5",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "openai/gpt-5.5-2026-04-23": "GPT-5.5",
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "openai/gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
    "human": "Humans",
}
ROUTE_SHORT = {
    "anthropic/claude-opus-5@default": "Opus 5",
    "anthropic/claude-sonnet-5@default": "Sonnet 5",
    "google/gemini-3-flash-preview": "G3 Flash",
    "google/gemini-3.1-flash-lite-preview": "G3.1 FL",
    "google/gemini-3.5-flash-lite": "G3.5 FL",
    "openai/gpt-5.5-2026-04-23": "GPT-5.5",
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.4-mini-2026-03-17": "5.4 mini",
    "openai/gpt-5.4-nano-2026-03-17": "5.4 nano",
    "human": "Humans",
}

# Routes that appear only in the supplement.  They were never put through the
# admission audit, so they are deliberately NOT added to ``ROUTE_ORDER`` below:
# roster membership is what tells a figure whether a route can carry a verdict.
# They still need a stable hue, glyph and name, because a supplement that
# invents a second visual language costs the reader the identities they learnt
# in the main paper.  The three hues are the unused Tol-muted slots, chosen to
# sit away from the nine already spent; as everywhere here, hue is the second
# channel and the glyph plus the written name carry the identity.
ROUTE_C.update({
    "openai/gpt-5-nano": "#44aa99",                            # teal
    "openai/gpt-5.6-luna": "#882255",                          # wine
    "openai/gpt-5.6-terra": "#999933",                         # olive
})
ROUTE_M.update({
    "openai/gpt-5-nano": "p",
    "openai/gpt-5.6-luna": "h",
    "openai/gpt-5.6-terra": "d",
})
ROUTE_LABEL.update({
    "openai/gpt-5-nano": "GPT-5 nano",
    "openai/gpt-5.6-luna": "GPT-5.6 Luna",
    "openai/gpt-5.6-terra": "GPT-5.6 Terra",
})
ROUTE_SHORT.update({
    "openai/gpt-5-nano": "5 nano",
    "openai/gpt-5.6-luna": "5.6 Luna",
    "openai/gpt-5.6-terra": "5.6 Terra",
})

# The result tables under ``results/cross_model_pilot_synthesis`` key a route by
# a short local name rather than by its provider-qualified id.  One alias table,
# here, is what stops a figure from inventing its own colour and its own
# spelling for a route the main paper has already named.
ROUTE_ALIAS = {
    "gpt-5-nano": "openai/gpt-5-nano",
    "gpt-5.4-nano": "openai/gpt-5.4-nano-2026-03-17",
    "gpt-5.6-luna": "openai/gpt-5.6-luna",
    "gpt-5.6-terra": "openai/gpt-5.6-terra",
    "claude-opus-5": "anthropic/claude-opus-5@default",
    "claude-sonnet-5": "anthropic/claude-sonnet-5@default",
}


def route_id(key: str) -> str:
    """Canonical route id for a local table key, or the key if it is already one."""
    return ROUTE_ALIAS.get(key, key)


# The five admitted routes, in the order they are always listed, then the four
# refused.  Admission is a property of the route, so the split lives here and
# not in each figure script.
ADMITTED = [
    "google/gemini-3-flash-preview",
    "anthropic/claude-opus-5@default",
    "openai/gpt-5.4-2026-03-05",
    "openai/gpt-5.5-2026-04-23",
    "anthropic/claude-sonnet-5@default",
]
REFUSED = [
    "google/gemini-3.1-flash-lite-preview",
    "openai/gpt-5.4-mini-2026-03-17",
    "google/gemini-3.5-flash-lite",
    "openai/gpt-5.4-nano-2026-03-17",
]
ROUTE_ORDER = ADMITTED + REFUSED


def in_roster(key: str) -> bool:
    """Is this route one of the nine the admission audit ranged over?

    A figure that hard-codes the answer as a list of names drifts the moment a
    route is added, which is how ``gpt-5-nano`` came to be drawn as a roster
    member in the supplement while two other non-roster routes were dashed.
    """
    return route_id(key) in ROUTE_ORDER

RISKS = (0.1, 0.6, 0.9)
RISK_LABEL = {0.1: "0.1", 0.6: "0.6", 0.9: "0.9"}

# --- rcParams -----------------------------------------------------------------
# These are the PRINTED sizes, because ``save`` renders 1:1.  The predecessor
# module's 8.0 pt floor printed at 5.11 pt on a column figure, so a smaller
# declared number here is a larger number on the page.
RC = {
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    # Without this matplotlib sets body text in the sans family above but every
    # $p_r$ in DejaVu, so a figure ships two typefaces and the maths does not
    # match the prose beside it.
    "mathtext.fontset": "custom",
    "mathtext.rm": "sans",
    "mathtext.it": "sans:italic",
    "mathtext.bf": "sans:bold",
    "mathtext.default": "it",
    "font.size": 7.2,
    "axes.titlesize": 7.6,
    "axes.labelsize": 7.2,
    "xtick.labelsize": 6.8,
    "ytick.labelsize": 6.8,
    "legend.fontsize": 6.9,
    "axes.edgecolor": HAIRLINE,
    "axes.linewidth": 0.6,
    "axes.labelcolor": INK_2,
    "text.color": INK,
    "xtick.color": INK_2,
    "ytick.color": INK_2,
    "xtick.major.size": 2.4,
    "ytick.major.size": 2.4,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.minor.size": 1.3,
    "ytick.minor.size": 1.3,
    "grid.color": GRID,
    "grid.linewidth": 0.5,
    "grid.linestyle": "-",
    "legend.frameon": False,
    "legend.handletextpad": 0.5,
    "legend.labelspacing": 0.30,
    "legend.columnspacing": 1.3,
    "lines.solid_capstyle": "butt",
    "lines.linewidth": 1.2,
    "lines.markersize": 3.2,
    "patch.linewidth": 0.6,
    # Type 3 fonts are matplotlib's default and are rejected by ACM's checker.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "svg.fonttype": "none",
}
mpl.rcParams.update(RC)

FS_PANEL = 8.2      # panel letter
FS_CLAIM = 7.3      # the claim sentence beside the panel letter
FS_NOTE = 6.2       # in-plot notes, direct labels, counts
FS_TICK = 6.8

# The floor is on the PRINTED size, which is the drawn size because save fits
# the canvas.  6.0 pt is the smallest type that survives a 600 dpi raster and a
# printed page; anything below it is a mistake rather than a choice.
MIN_PRINTED_POINTS = 6.0


def strip(ax, *, left=True, bottom=True, grid_axis="y"):
    """Recessive frame: keep only the spines that carry a scale."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not left:
        ax.spines["left"].set_visible(False)
    if not bottom:
        ax.spines["bottom"].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)


def panel(ax, letter, claim=None, *, pad=5, x=0.0, gap=9.5):
    """Panel letter in bold, then the claim the panel makes, above the axes.

    The claim is the figure's argument and belongs where the eye lands first,
    not at the end of a caption three inches below.  The gap is in points and
    not in axes fractions, because an axes fraction is a different distance on
    a narrow panel than on a wide one.
    """
    ax.set_title(letter, loc="left", pad=pad, x=x,
                 fontsize=FS_PANEL, color=INK, fontweight="bold")
    if claim:
        ax.annotate(claim, xy=(x, 1.0), xycoords="axes fraction",
                    xytext=(gap, pad - 3), textcoords="offset points",
                    ha="left", va="bottom", fontsize=FS_CLAIM, color=INK_2,
                    annotation_clip=False)


def risk_axis(ax, *, label=r"maximum private risk $p_r^{\max}$", ticks=True):
    """The risk axis shared by most panels."""
    if ticks:
        ax.set_xticks(list(RISKS))
        ax.set_xticklabels([RISK_LABEL[r] for r in RISKS])
    ax.set_xlim(0.0, 1.0)
    if label:
        ax.set_xlabel(label)


def rate_axis(ax, *, label="Unsafe play (%)", lo=0, hi=100):
    ax.set_ylim(lo, hi)
    ax.set_yticks([0, 25, 50, 75, 100])
    if label:
        ax.set_ylabel(label)


def ceiling_rule(ax, value=100.0, *, label="ceiling"):
    """Mark the boundary that truncates a contrast, so it is never mistaken.

    A cell at 100% is not a precise measurement, it is a measurement with no
    room above it, and a figure that does not say so invites the reader to read
    a truncated contrast as a small one.
    """
    ax.axhline(value, color=MUTED, lw=0.7, ls=(0, (3, 2)), zorder=1)
    if label:
        ax.annotate(label, xy=(1.0, value), xycoords=("axes fraction", "data"),
                    xytext=(-1, 2), textcoords="offset points",
                    ha="right", va="bottom", fontsize=FS_NOTE, color=MUTED)


def zero_rule(ax, value=0.0, *, vertical=False, lw=0.8, color=None):
    fn = ax.axvline if vertical else ax.axhline
    fn(value, color=color or MUTED, lw=lw, zorder=1)


def dot(ax, x, y, *, color, marker="o", size=16, filled=True, zorder=4,
        ring=True, lw=None):
    """A marker with a surface-coloured ring, so coincident points stay two."""
    if filled:
        face, edge, elw = color, (SURFACE if ring else color), (0.7 if ring else 0.0)
    else:
        face, edge, elw = SURFACE, color, 1.0
    ax.scatter([x], [y], s=size, marker=marker, zorder=zorder,
               facecolors=face, edgecolors=edge,
               linewidths=elw if lw is None else lw)


def tile_ink(value, *, lo=0.0, hi=100.0, cmap="viridis"):
    """The ink that survives on the tile a value paints.

    Ask the colormap for the colour it will actually draw and take the ink from
    that colour's relative luminance.  A fixed cut in the VALUE cannot do this
    job, because it assumes the map darkens as the value rises: viridis
    brightens, so a fixed cut puts white type on the yellow end and dark type on
    the purple end, which is the weakest type available at both ends of the
    scale.  That defect reached three shipped figures and was patched three
    separate times in three scripts before it was fixed here.
    """
    t = (float(value) - lo) / ((hi - lo) or 1.0)
    r, g, b, _ = plt.get_cmap(cmap)(min(max(t, 0.0), 1.0))
    return INK if (0.2126 * r + 0.7152 * g + 0.0722 * b) > 0.55 else SURFACE


def heat_tiles(ax, arr, row_labels, col_labels, *, cmap="viridis", vmin=None,
               vmax=None, fmt="{:.0f}", fs=None, gutter=2.0,
               textcolor_flip=None, row_colors=None):
    """imshow with a white gutter grid, so a heatmap reads as tiles.

    ``row_colors`` tints the row labels with the route colours, which is what
    lets a figure carry route identity without spending space on a legend.
    Returns the image so the caller can hang a colourbar on it.

    The numeral on a tile takes its ink from the tile's own luminance.
    ``textcolor_flip`` overrides that with a fixed cut in the normalised value,
    for a caller that has measured its own crossing point on its own colormap;
    leave it unset unless you have.
    """
    import numpy as np

    im = ax.imshow(arr, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
    if fmt:
        lo = vmin if vmin is not None else float(np.nanmin(arr))
        hi = vmax if vmax is not None else float(np.nanmax(arr))
        rng = (hi - lo) or 1.0
        masked = np.ma.getmaskarray(arr) if np.ma.isMaskedArray(arr) else None
        for i in range(arr.shape[0]):
            for j in range(arr.shape[1]):
                # A masked cell has no value to print, and printing one anyway
                # draws the mask's fill value straight through whatever note the
                # caller put there.
                if masked is not None and masked[i, j]:
                    continue
                v = arr[i, j]
                if v != v:
                    continue
                if textcolor_flip is None:
                    ink = tile_ink(v, lo=lo, hi=hi, cmap=cmap)
                else:
                    ink = SURFACE if (v - lo) / rng > textcolor_flip else INK
                ax.text(j, i, fmt.format(v), ha="center", va="center",
                        fontsize=fs or FS_NOTE, color=ink)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels)
    ax.set_yticks(range(len(row_labels)))
    ax.set_yticklabels(row_labels)
    if row_colors:
        for tick, colour in zip(ax.get_yticklabels(), row_colors):
            tick.set_color(colour)
    ax.set_xticks([x - 0.5 for x in range(len(col_labels) + 1)], minor=True)
    ax.set_yticks([y - 0.5 for y in range(len(row_labels) + 1)], minor=True)
    ax.grid(which="minor", color=SURFACE, linewidth=gutter)
    ax.grid(which="major", visible=False)
    ax.tick_params(which="minor", length=0)
    ax.tick_params(which="major", length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    return im


def highlight(ax, i, j, *, color=None, lw=1.1, inset=0.44):
    """Ring one tile, for the cells an adjacent panel expands.

    The reference figures in this literature call out the exact matchups they
    then draw in full, and the ring is what ties the two panels together
    without a sentence of caption doing it.
    """
    from matplotlib.patches import Rectangle

    ax.add_patch(Rectangle((j - inset, i - inset), 2 * inset, 2 * inset,
                           fill=False, edgecolor=color or UNSAFE_C, lw=lw,
                           zorder=5, clip_on=False))


def direct_label(ax, x, y, text, *, color, ha="left", va="center", dx=3, dy=0,
                 weight="normal"):
    """A series label at the end of its own line, instead of a legend."""
    ax.annotate(text, xy=(x, y), xytext=(dx, dy), textcoords="offset points",
                ha=ha, va=va, fontsize=FS_NOTE, color=color,
                fontweight=weight, annotation_clip=False)


def caption(fig, text, *, y=-0.02, width_frac=0.98):
    """A bottom note wrapped to the figure's own width.

    A single long ``fig.text`` does not wrap, and with ``bbox_inches='tight'``
    it pushes the saved bounding box out past the figure, so the file on disk
    is wider than the column even though ``figsize`` still says otherwise.
    Wrapping here is what keeps the saved width honest.
    """
    chars = int(fig.get_size_inches()[0] * width_frac * 72.0 / (FS_NOTE * 0.50))
    wrapped = "\n".join(textwrap.wrap(" ".join(text.split()), chars))
    return fig.text(0.5, y, wrapped, ha="center", va="top",
                    fontsize=FS_NOTE, color=MUTED, linespacing=1.4)


def _tight_size(fig):
    fig.canvas.draw()
    bb = fig.get_tightbbox(fig.canvas.get_renderer())
    return bb.width + 2 * PAD, bb.height + 2 * PAD


def check_printed_type(fig, *, minimum=MIN_PRINTED_POINTS):
    """Refuse a figure whose smallest printed string is below the floor.

    This checks the drawn size, which ``save`` has made equal to the printed
    size.  That equality is the whole point: the predecessor module enforced
    8.0 pt on a canvas that LaTeX then shrank to 64%.
    """
    small = []
    for artist in fig.findobj(mpl.text.Text):
        if artist.get_visible() and artist.get_text().strip():
            size = artist.get_fontsize()
            if size < minimum - 1e-6:
                small.append((artist.get_text()[:40], size))
    if small:
        worst = ", ".join(f"{t!r} at {s:.2f} pt" for t, s in small[:4])
        raise ValueError(
            f"{len(small)} string(s) below the {minimum} pt printed floor: {worst}"
        )


def save(fig, name, *, figdir=None, width=None, formats=("pdf", "png"),
         strict=True, check_type=True):
    """Save at exactly the declared width, so the type is the size it says.

    ``bbox_inches='tight'`` trims to the ink, so the file on disk is almost
    never the size ``figsize`` declared: an outside legend makes it wider and
    ordinary margins make it narrower.  LaTeX then scales whatever it gets to
    ``\\columnwidth``, which rescales every label in the figure.  So measure the
    tight box, grow or shrink the canvas by the difference, and measure again.
    Font sizes are in points and do not move when the canvas does, so this
    changes only how much room the axes get.

    ``width`` is the intended placement width in inches, normally ``COL`` or
    ``TEXT``.  It defaults to whatever ``figsize`` declared.
    """
    d = Path(figdir) if figdir else FIGDIR
    d.mkdir(parents=True, exist_ok=True)
    target = float(width) if width else float(fig.get_size_inches()[0])

    for _ in range(4):
        w, h = _tight_size(fig)
        if abs(w - target) < 1 / 64:
            break
        cur_w, cur_h = fig.get_size_inches()
        fig.set_size_inches(cur_w + (target - w), cur_h, forward=True)

    if check_type:
        check_printed_type(fig)

    w, h = _tight_size(fig)
    off = abs(w - target)
    if strict and off > 1 / 16:
        raise ValueError(
            f"{name}: saved width {w:.3f} in against a target of {target:.3f} in, "
            f"which LaTeX would rescale by {100 * target / w:.0f}%"
        )

    written = []
    for ext in formats:
        path = d / f"{name}.{ext}"
        fig.savefig(path, bbox_inches="tight", pad_inches=PAD,
                    dpi=DPI if ext == "png" else None)
        written.append(path)
    plt.close(fig)

    print(f"  {name}: {w:.3f} x {h:.3f} in at target {target:.3f} in "
          f"({100 * target / w:.1f}% rescale)")
    for path in written:
        try:
            shown = path.relative_to(REPO)
        except ValueError:
            shown = path
        print(f"    -> {shown}")
    return written
