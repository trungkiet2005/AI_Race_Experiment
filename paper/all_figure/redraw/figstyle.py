"""Shared drawing vocabulary for the redrawn paper figures.

The palette below is the design-system categorical order, validated with the
dataviz skill's checker rather than chosen by eye:

    slots 1-5, adjacent pairlist, light surface
        CVD separation      PASS  worst adjacent dE 9.1 (protan)
        Normal-vision floor PASS  worst adjacent dE 19.6
    slots 1-3, all-pairs (used wherever series can touch any other series)
        CVD separation      PASS  worst pair dE 9.2
        Normal-vision floor PASS  worst pair dE 24.0

Three slots sit below 3:1 contrast on a white surface, which obliges the relief
rule: every figure ships a source table under ``tables/`` and carries direct
labels, so identity is never colour-alone. Every multi-series figure also varies
marker shape, which keeps it readable in greyscale print and under CVD.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt

FIG_DIR = Path(__file__).resolve().parent / "figures"
TABLE_DIR = Path(__file__).resolve().parent / "tables"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ACM sigconf geometry.
COL_W = 3.33
FULL_W = 7.00

# Categorical slots, fixed order. Colour follows the entity: a model keeps its hue
# in every figure it appears in, so a reader can carry identity across the paper.
C_BLUE = "#2a78d6"
C_ORANGE = "#eb6834"
C_AQUA = "#1baf7a"
C_YELLOW = "#eda100"
C_MAGENTA = "#e87ba4"

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#8a8a86"
GRID = "#d9d9d6"
SURFACE = "#ffffff"

# One hue per checkpoint, reused across figures D, E and F.
MODEL_STYLE = {
    "Gemini 3 Flash": (C_BLUE, "o"),
    "Gemini 3.1 Flash-Lite": (C_ORANGE, "s"),
    "Gemini 3.5 Flash-Lite": (C_AQUA, "^"),
    "GPT-5 nano": (C_YELLOW, "D"),
    "GPT-5.4 nano": (C_MAGENTA, "P"),
}
# The cross-provider matchups use a different checkpoint set; Gemini 3.5 Flash-Lite
# appears in both and deliberately keeps its aqua/triangle identity.
PROVIDER_STYLE = {
    "GPT Luna": (C_BLUE, "o"),
    "Claude Haiku 4.5": (C_ORANGE, "s"),
    "Gemini 3.5 Flash-Lite": (C_AQUA, "^"),
}

RISK_LABEL = {0.1: "10%", 0.6: "60%", 0.9: "90%"}


def use_paper_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE,
        "axes.facecolor": SURFACE,
        "savefig.facecolor": SURFACE,
        "font.size": 7.6,
        "axes.titlesize": 8.2,
        "axes.labelsize": 7.6,
        "xtick.labelsize": 7.2,
        "ytick.labelsize": 7.2,
        "legend.fontsize": 7.0,
        "axes.edgecolor": GRID,
        "axes.linewidth": 0.6,
        "axes.labelcolor": INK_2,
        "text.color": INK,
        "xtick.color": INK_2,
        "ytick.color": INK_2,
        "xtick.major.size": 2.5,
        "ytick.major.size": 2.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "grid.color": GRID,
        "grid.linewidth": 0.55,
        # Solid, never dashed: a dashed rule reads as a threshold, and none of these
        # gridlines are one.
        "grid.linestyle": "-",
        "legend.frameon": False,
        "lines.solid_capstyle": "butt",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    })


def strip(ax, *, x: bool = True, y: bool = False, grid_axis: str = "x") -> None:
    """Recessive frame: keep only the spines that carry a scale."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if not x:
        ax.spines["bottom"].set_visible(False)
    if not y:
        ax.spines["left"].set_visible(False)
    ax.set_axisbelow(True)
    if grid_axis:
        ax.grid(True, axis=grid_axis, zorder=0)


def zero_rule(ax, value: float = 0.0, *, vertical: bool = True, label: str | None = None):
    """A reference line for a meaningful zero or a named baseline. Solid by design."""
    fn = ax.axvline if vertical else ax.axhline
    fn(value, color=MUTED, lw=0.9, zorder=1)
    if label:
        if vertical:
            ax.annotate(label, xy=(value, 1.0), xycoords=("data", "axes fraction"),
                        xytext=(3, -8), textcoords="offset points",
                        fontsize=6.6, color=MUTED, ha="left", va="top")
        else:
            ax.annotate(label, xy=(0.0, value), xycoords=("axes fraction", "data"),
                        xytext=(3, 3), textcoords="offset points",
                        fontsize=6.6, color=MUTED, ha="left", va="bottom")


def interval(ax, y, lo, hi, *, color, lw: float = 1.8, zorder: int = 3, alpha: float = 1.0):
    ax.plot([lo, hi], [y, y], color=color, lw=lw, solid_capstyle="butt",
            zorder=zorder, alpha=alpha)


def dot(ax, x, y, *, color, marker="o", size: float = 34, filled: bool = True,
        zorder: int = 4, ring: bool = True):
    """A data point. ``filled=False`` is reserved for values that are degenerate or
    saturated -- an open marker says 'this is not a precise estimate'.

    Filled marks carry a surface-coloured ring so two coincident points stay two
    points instead of merging into one blob."""
    if filled:
        face, edge, lw = color, (SURFACE if ring else color), (1.2 if ring else 0.0)
    else:
        face, edge, lw = SURFACE, color, 1.5
    ax.scatter([x], [y], s=size, marker=marker, zorder=zorder,
               facecolors=face, edgecolors=edge, linewidths=lw)


def save(fig, name: str) -> None:
    for ext in ("pdf", "png"):
        fig.savefig(FIG_DIR / f"{name}.{ext}", dpi=400, bbox_inches="tight",
                    pad_inches=0.02)
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf + .png")
