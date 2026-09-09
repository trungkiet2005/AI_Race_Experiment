"""Shared visual language for the AI Race paper.

The reference figures use a restrained white canvas, compact panel labels, and
one meaning per colour. Keeping those choices in one module prevents the paper
figures from drifting back to dashboard-style defaults when a generator is
rerun.
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt


WHITE = "#FFFFFF"
INK = "#17212B"
MUTED = "#4B5865"
LINE = "#778492"
GRID = "#D9DEE8"
GREY_LIGHT = "#F4F6F8"

# Semantic pairs. Use the saturated colour for marks and the light tint for a
# panel/card fill. GOLD is a fill/mark colour, never a text colour.
BLUE, BLUE_LIGHT = "#2F6283", "#EAF2F7"
GOLD, GOLD_LIGHT = "#B9842C", "#FBF4E6"
GREEN, GREEN_LIGHT = "#39745A", "#EAF5EF"
RED, RED_LIGHT = "#A14D4B", "#FAEEEE"

# Categorical series use a second channel as well as colour. These are muted
# enough to sit beside the semantic pairs and remain legible on a white page.
CATEGORICAL = ["#2F6283", "#C27A35", "#3A8064", "#8A5E91", "#A14D55", "#647889"]
EXTENDED_CATEGORICAL = CATEGORICAL + ["#4477AA", "#AA3377", "#CC6677"]
MARKERS = ["o", "s", "^", "D", "P", "X", "v", "<", ">"]
HATCHES = ["", "//", "xx", "\\\\", "..", "++"]

# Fixed meanings across the manuscript. The labels stay black in the figures;
# the colours belong to the data marks, so they do not become a text encoding.
MODEL_COLOURS = {
    "gpt-5-nano": CATEGORICAL[0],
    "gpt-5.4-nano": CATEGORICAL[1],
    "google/gemini-3-flash-preview": CATEGORICAL[2],
    "google/gemini-3.1-flash-lite-preview": CATEGORICAL[3],
    "google/gemini-3.5-flash-lite": CATEGORICAL[4],
    "gpt-5.6-luna": EXTENDED_CATEGORICAL[6],
    "gpt-5.6-terra": EXTENDED_CATEGORICAL[7],
    "claude-opus-5": EXTENDED_CATEGORICAL[8],
    "claude-sonnet-5": CATEGORICAL[5],
    "human": INK,
}
MODEL_MARKERS = {name: MARKERS[index % len(MARKERS)] for index, name in enumerate(MODEL_COLOURS)}

TEXT_WIDTH_IN = 5.17  # 372 pt, the measured AAMAS text width.
FULL_WIDTH_IN = 7.10
MIN_TEXT_POINTS = 8.0
PNG_DPI = 600

# Figures 1 and 2 are author-supplied artwork. Keep them under their canonical
# paper paths, but make every publication writer fail closed before it can
# replace them with a generated variant. Figure 4 is generated from the
# checked-in trajectory table. The former Figure 8 remains archived, but is
# not part of the current protected manuscript set.
PROTECTED_MANUAL_FIGURE_STEMS = frozenset(
    {
        "figures/paper/AIRaceOverview",
        "figures/paper/ExpOverview",
    }
)


def _protected_manual_stem(stem: Path) -> str | None:
    try:
        relative = stem.resolve().relative_to(Path(__file__).resolve().parents[1]).as_posix()
    except ValueError:
        return None
    return relative if relative in PROTECTED_MANUAL_FIGURE_STEMS else None


def configure_publication_style() -> None:
    """Set the single rcParams policy used by every paper figure."""

    mpl.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.4,
            "text.color": INK,
            "axes.labelcolor": INK,
            "axes.titlecolor": INK,
            "axes.titlesize": 9.2,
            "axes.titleweight": "bold",
            "axes.labelsize": 8.4,
            "xtick.labelsize": 8.0,
            "ytick.labelsize": 8.0,
            "legend.fontsize": 8.0,
            "axes.edgecolor": INK,
            "axes.linewidth": 0.8,
            "axes.facecolor": WHITE,
            "figure.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.edgecolor": WHITE,
            "savefig.transparent": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
            "axes.axisbelow": True,
            "savefig.pad_inches": 0.0,
        }
    )


def style_axis(ax: plt.Axes, *, grid_axis: str | None = "y") -> None:
    """Apply the shared axes chrome without changing the data encoding."""

    ax.set_facecolor(WHITE)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(INK)
    ax.spines["bottom"].set_color(INK)
    ax.tick_params(colors=MUTED, width=0.7, length=3)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.6, alpha=0.9)
    ax.set_axisbelow(True)


def panel_label(ax: plt.Axes, label: str, title: str | None = None) -> None:
    """Place the same compact panel label and optional title everywhere."""

    text = label if title is None else f"{label}  {title}"
    ax.set_title(text, loc="left", pad=6, weight="bold", color=INK)


def ensure_minimum_text(fig: plt.Figure, *, minimum: float = MIN_TEXT_POINTS) -> None:
    """Fail before export if a text artist is too small for print."""

    for artist in fig.findobj(mpl.text.Text):
        if artist.get_visible() and artist.get_fontsize() < minimum - 1e-6:
            content = artist.get_text().replace("\n", " ")[:36]
            raise RuntimeError(
                f"Text below publication floor: {content!r} at {artist.get_fontsize():.2f} pt"
            )


def _strip_svg_whitespace(path: Path) -> None:
    if path.suffix.lower() != ".svg" or not path.is_file():
        return
    path.write_text(
        "\n".join(line.rstrip() for line in path.read_text(encoding="utf-8").splitlines())
        + "\n",
        encoding="utf-8",
    )


def save_publication_figure(
    fig: plt.Figure,
    stem: Path,
    *,
    formats: Iterable[str] = ("pdf", "png", "svg"),
    dpi: int = PNG_DPI,
    bbox_inches: str | None = "tight",
    pad_inches: float = 0.04,
) -> list[Path]:
    """Export the same figure as vector PDF, PNG preview, and editable SVG."""

    protected = _protected_manual_stem(stem)
    if protected is not None:
        raise RuntimeError(
            f"Refusing to overwrite author-supplied artwork: {protected}. "
            "Use a new generated stem for an analysis variant."
        )
    stem.parent.mkdir(parents=True, exist_ok=True)
    configure_publication_style()
    fig.set_facecolor(WHITE)
    for ax in fig.axes:
        ax.set_facecolor(WHITE)
    ensure_minimum_text(fig)
    paths: list[Path] = []
    for fmt in formats:
        path = stem.with_suffix(f".{fmt}")
        kwargs = {
            "bbox_inches": bbox_inches,
            "pad_inches": pad_inches,
            "facecolor": WHITE,
            "edgecolor": WHITE,
            "transparent": False,
        }
        if fmt == "png":
            kwargs["dpi"] = dpi
        if fmt == "pdf":
            kwargs["metadata"] = {
                "Title": stem.stem,
                "Creator": "AI Race publication figure generator",
                "Subject": "AI Race empirical research figure",
            }
        fig.savefig(path, **kwargs)
        _strip_svg_whitespace(path)
        paths.append(path)
    plt.close(fig)
    return paths


def set_percent_axis(ax: plt.Axes, *, ymax: float = 1.0) -> None:
    ax.set_ylim(0, ymax)
    ax.yaxis.set_major_formatter(mpl.ticker.PercentFormatter(1.0, decimals=0))
