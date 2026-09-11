"""Measure the smallest type each figure actually puts on the page.

``check_figure_geometry.py`` proves a figure is not rescaled. That is necessary
and not sufficient: a figure drawn at the right width can still carry 4 pt
labels, and a figure that IS rescaled can still be legible if it was drawn with
very large type, which is exactly the case for the hand-drawn overview schematic
here. Neither situation is visible from the placement width alone.

So this reads the type sizes out of each figure's content stream, multiplies by
the scale LaTeX will apply, and reports the smallest string that reaches the
page. A vector figure carries its font sizes explicitly; a raster does not, so a
raster is reported as unmeasurable rather than silently passed.

    python scripts/check_printed_type.py
    python scripts/check_printed_type.py --floor 6.0
"""

from __future__ import annotations

import argparse
import re
import subprocess
import zlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"
PT = 1 / 72.27
COLUMN_IN = 241.14749 * PT
TEXT_IN = 506.295 * PT

INCLUDE = re.compile(
    r"\\includegraphics\s*(?:\[(?P<opts>[^\]]*)\])?\s*\{(?P<path>[^}]+)\}"
)
WIDTH = re.compile(r"width\s*=\s*(?P<factor>[0-9.]*)\s*\\(?P<unit>columnwidth|textwidth|linewidth)")
# "/F3 8.4 Tf" sets font F3 at 8.4 units; the text matrix can then scale it, but
# matplotlib and most vector exporters emit an identity text matrix, so the Tf
# operand is the drawn point size.
TF = re.compile(rb"/[A-Za-z0-9]+\s+([0-9.]+)\s+Tf")


def declared_width_in(opts: str, in_star: bool) -> float | None:
    if not opts:
        return None
    m = WIDTH.search(opts)
    if not m:
        return None
    factor = float(m.group("factor")) if m.group("factor") else 1.0
    unit = m.group("unit")
    base = TEXT_IN if unit == "textwidth" else (
        TEXT_IN if (unit == "linewidth" and in_star) else COLUMN_IN)
    return factor * base


def natural_width_in(path: Path) -> float | None:
    out = subprocess.run(["pdfinfo", str(path)], capture_output=True, text=True)
    if out.returncode != 0:
        return None
    for line in out.stdout.splitlines():
        if line.startswith("Page size"):
            return float(line.split(":")[1].split("x")[0]) * PT
    return None


def drawn_type_sizes(path: Path) -> list[float]:
    """Every font size set in the file's content streams, in drawn points."""
    raw = path.read_bytes()
    sizes: list[float] = []
    for match in re.finditer(rb"stream\r?\n(.*?)endstream", raw, re.S):
        chunk = match.group(1)
        try:
            chunk = zlib.decompress(chunk)
        except zlib.error:
            pass
        sizes += [float(v) for v in TF.findall(chunk)]
    return [s for s in sizes if s > 0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floor", type=float, default=6.0,
                        help="floor on the type a reader has to read")
    parser.add_argument("--script-floor", type=float, default=4.0,
                        help="floor on nested sub- and superscript levels")
    parser.add_argument("--body-share", type=float, default=0.05,
                        help="a size used for at least this share of the "
                             "settings counts as body type")
    args = parser.parse_args()

    rows = []
    for name in ("main.tex", "supplementary.tex"):
        source = PAPER / name
        if not source.is_file():
            continue
        text = source.read_text(encoding="utf-8")
        for m in INCLUDE.finditer(text):
            before = text[: m.start()]
            in_star = before.rfind(r"\begin{figure*}") > before.rfind(r"\begin{figure}")
            path = (PAPER / m.group("path")).resolve()
            rows.append((name, path, declared_width_in(m.group("opts") or "", in_star)))

    failures = 0
    print(f"floor {args.floor} pt of printed type\n")
    width = max(len(p.name) for _, p, _ in rows)
    for source, path, placed in sorted({(s, p, w) for s, p, w in rows}):
        if not path.is_file():
            print(f"MISSING  {path.name}")
            failures += 1
            continue
        if path.suffix.lower() != ".pdf":
            print(f"raster   {path.name:<{width}}  no type sizes in a raster; "
                  f"legibility must be judged by eye")
            continue
        natural = natural_width_in(path)
        sizes = drawn_type_sizes(path)
        if natural is None or placed is None:
            print(f"SKIP     {path.name:<{width}}  cannot resolve its width")
            continue
        if not sizes:
            print(f"no type  {path.name:<{width}}  draws no text of its own")
            continue
        scale = placed / natural
        # Mathtext renders a nested script level smaller than its parent, so a
        # 7.2 pt label reading "$p_r^{\max}$" emits the \max at about 4.3 pt.
        # That is expected typography, not a defect, and the reader does not
        # have to read \max to use an axis whose words already say "maximum".
        # So the floor that matters applies to the type a figure actually
        # speaks in, identified as any size used for a non-trivial share of its
        # settings, and a separate lower floor catches a script level that has
        # shrunk to nothing.
        # A script level is identified by arithmetic rather than by how often it
        # appears: mathtext scales each nesting level by 0.7, so a size that is
        # 0.7 or 0.49 times another size present in the same file is a script
        # of it. A figure whose type has simply been shrunk has no such parent
        # and is caught.
        present = sorted({round(size, 2) for size in sizes})
        body = [
            s for s in present
            if not any(abs(s - ratio * parent) < 0.03 * parent
                       for parent in present if parent > s
                       for ratio in (0.7, 0.49))
        ]
        smallest_body = min(body) * scale if body else min(sizes) * scale
        smallest_any = min(sizes) * scale

        if smallest_body < args.floor:
            verdict, failures = "FAIL", failures + 1
        elif smallest_any < args.script_floor:
            verdict, failures = "FAIL", failures + 1
        else:
            verdict = "ok  "
        note = ""
        if smallest_any < smallest_body - 0.01:
            note = f", scripts down to {smallest_any:.2f} pt"
        print(f"{verdict}     {path.name:<{width}}  body {smallest_body:.2f} pt "
              f"printed at {100 * scale:.0f}%{note}")

    print()
    if failures:
        raise SystemExit(
            f"{failures} figure(s) put type below {args.floor} pt on the page. "
            "Redraw at the placement width or enlarge the type; do not rely on a "
            "reader zooming in."
        )
    print("every vector figure clears the printed-type floor")


if __name__ == "__main__":
    main()
