"""Refuse a figure that LaTeX would silently rescale on the page.

The failure this exists to catch is invisible in every other check.  A figure
is drawn at some width, matplotlib saves it at some other width because
``bbox_inches='tight'`` trims to the ink, and LaTeX then scales whatever it
gets to fit the ``width=`` in the ``\\includegraphics``.  Nothing errors.  The
figure looks fine on its own.  It is only on the page that its 8 pt tick labels
print at 5 pt, and by then the difference between two figures at nominally the
same font size is a difference the reader can see and the author cannot explain.

It was caught here by asking the class for its own geometry rather than reading
a guide: compiling ``\\documentclass[sigconf,anonymous]{aamas}`` with a
``\\maketitle`` and printing ``\\the\\columnwidth`` gives 241.147 pt, or 3.337 in,
against the 5.17 in that ``scripts/publication_style.py`` had been declaring.
Two of the four main figures were printing at about 64%.

The check is deliberately about the ratio and not about a font size, because
the ratio is the thing that is invisible.  A figure drawn 1:1 with 6.8 pt ticks
prints 6.8 pt ticks, and that is verifiable from the file alone.

Run it as a gate:

    python scripts/check_figure_geometry.py
    python scripts/check_figure_geometry.py --tolerance 0.03
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
SOURCES = ("main.tex", "supplementary.tex")

# Measured from the class itself, not from a guide.  See the module docstring.
PT = 1 / 72.27
COLUMN_IN = 241.14749 * PT
TEXT_IN = 506.295 * PT

# A figure the manuscript ships as author artwork rather than as a generated
# plot.  A schematic has no tick labels to shrink, so the ratio rule does not
# apply to it in the same way; it is reported and not enforced.
ARTWORK = {"AIRaceOverview.pdf", "ExpOverview.pdf"}

INCLUDE = re.compile(
    r"\\includegraphics\s*(?:\[(?P<opts>[^\]]*)\])?\s*\{(?P<path>[^}]+)\}"
)
WIDTH = re.compile(
    r"width\s*=\s*(?P<factor>[0-9.]*)\s*\\(?P<unit>columnwidth|textwidth|linewidth)"
)


def declared_width_in(opts: str, in_figure_star: bool) -> float | None:
    """The width in inches that the include asks LaTeX for."""
    if not opts:
        return None
    m = WIDTH.search(opts)
    if not m:
        return None
    factor = float(m.group("factor")) if m.group("factor") else 1.0
    unit = m.group("unit")
    if unit == "textwidth":
        base = TEXT_IN
    elif unit == "columnwidth":
        base = COLUMN_IN
    else:
        # \linewidth is the column inside a one-column float and the full text
        # width inside a starred one, which is exactly why it is ambiguous and
        # why the manuscript should prefer the explicit names.
        base = TEXT_IN if in_figure_star else COLUMN_IN
    return factor * base


def file_width_in(path: Path) -> float | None:
    """The natural width of the file on disk, in inches."""
    if path.suffix.lower() == ".pdf":
        out = subprocess.run(
            ["pdfinfo", str(path)], capture_output=True, text=True
        )
        if out.returncode != 0:
            return None
        for line in out.stdout.splitlines():
            if line.startswith("Page size"):
                return float(line.split(":")[1].split("x")[0]) * PT
        return None
    if path.suffix.lower() in {".png", ".jpg", ".jpeg"}:
        try:
            from PIL import Image
        except ImportError:
            return None
        with Image.open(path) as im:
            width_px = im.size[0]
            dpi = im.info.get("dpi", (72.0, 72.0))[0] or 72.0
        # A raster carries its own resolution, and LaTeX honours it, so the
        # natural width is pixels over dpi.  A PNG saved without a dpi tag
        # defaults to 72, which is why a 3,000 px figure can arrive four feet
        # wide and be scaled to nothing.
        return width_px / float(dpi)
    return None


def scan() -> list[dict]:
    rows: list[dict] = []
    for name in SOURCES:
        source = PAPER / name
        if not source.exists():
            continue
        text = source.read_text(encoding="utf-8")
        for match in INCLUDE.finditer(text):
            before = text[: match.start()]
            # The nearest preceding float opening decides what \linewidth means.
            last_star = before.rfind(r"\begin{figure*}")
            last_plain = before.rfind(r"\begin{figure}")
            in_star = last_star > last_plain
            target = declared_width_in(match.group("opts") or "", in_star)
            rel = match.group("path")
            path = (PAPER / rel).resolve()
            rows.append(
                {
                    "source": name,
                    "path": path,
                    "name": path.name,
                    "declared_in": target,
                    "file_in": file_width_in(path) if path.exists() else None,
                    "exists": path.exists(),
                    "artwork": path.name in ARTWORK,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tolerance",
        type=float,
        default=0.05,
        help="allowed |1 - printed/drawn| before a generated figure is refused",
    )
    args = parser.parse_args()

    rows = scan()
    if not rows:
        raise SystemExit("no \\includegraphics found; is the manuscript in paper/?")

    failures = 0
    print(f"column {COLUMN_IN:.4f} in, text {TEXT_IN:.4f} in, "
          f"measured from the aamas class\n")
    width = max(len(r["name"]) for r in rows)
    for row in sorted(rows, key=lambda r: (r["source"], r["name"])):
        if not row["exists"]:
            print(f"MISSING  {row['name']:<{width}}  {row['path']}")
            failures += 1
            continue
        if row["declared_in"] is None or row["file_in"] is None:
            print(f"SKIP     {row['name']:<{width}}  no width= or unreadable size")
            continue
        ratio = row["declared_in"] / row["file_in"]
        off = abs(1.0 - ratio)
        if row["artwork"]:
            verdict = "ART "
        elif off > args.tolerance:
            verdict = "FAIL"
            failures += 1
        else:
            verdict = "ok  "
        print(f"{verdict}     {row['name']:<{width}}  drawn {row['file_in']:.3f} in, "
              f"placed {row['declared_in']:.3f} in, printed at {100 * ratio:.1f}%")
        if verdict == "FAIL":
            print(f"{'':9}{'':<{width}}  a {7.0:.1f} pt label in this figure "
                  f"prints at {7.0 * ratio:.2f} pt")

    print()
    if failures:
        raise SystemExit(
            f"{failures} figure(s) would be rescaled on the page. Redraw at the "
            f"placement width, or place at the drawn width; do not adjust the "
            f"font size to compensate, because that hides the mismatch instead "
            f"of removing it."
        )
    print(f"{len(rows)} figure(s) render at their drawn size")


if __name__ == "__main__":
    main()
