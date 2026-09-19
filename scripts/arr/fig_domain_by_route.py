#!/usr/bin/env python3
"""Accuracy per route per probe domain, admitted routes above refused ones.

Drawn from the raw probe rows through the same fail-closed loader as
analyze_screen_baselines.py, so the figure refuses exactly the campaigns the
analyser refuses and cannot drift from its tables.  Rows are grouped by the
recorded verdict first and ordered by overall accuracy inside each group, so the
separation stays readable when a later roster has a refused route that outscores
an admitted one.  The overall accuracy is printed at the right of each row; it is
not a seventh colour column, because it is a different quantity (the gate) and
would draw the eye to the aggregate this figure exists to take apart.

No title and no caption inside the figure: the caption lives in LaTeX.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_screen_baselines as A  # noqa: E402

ROOT = A.ROOT
DEFAULT_FIGDIR = ROOT / "figures" / "arr"
WIDTH_IN = 3.3
FS = 7.0
CMAP = "cividis"

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS,
    "axes.linewidth": 0.6,
})

DOMAIN_LABEL = {
    "rule_recall": "Rule recall",
    "stage_payoff": "Stage payoff",
    "state_reconstruction": "State recon.",
    "state_transition": "Transition",
    "terminal_scoring": "Terminal",
    "expected_payoff": "Exp. payoff",
}
ROUTE_LABEL = {
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "anthropic/claude-opus-5@default": "Claude Opus 5",
    "openai/gpt-5.4-2026-03-05": "GPT-5.4",
    "openai/gpt-5.5-2026-04-23": "GPT-5.5",
    "anthropic/claude-sonnet-5@default": "Claude Sonnet 5",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "openai/gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "openai/gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
}


def route_label(route: str) -> str:
    if route in ROUTE_LABEL:
        return ROUTE_LABEL[route]
    name = route.split("/", 1)[-1]
    name = re.sub(r"@default$", "", name)
    return re.sub(r"-\d{4}-\d{2}-\d{2}$", "", name)


def domain_label(domain: str) -> str:
    return DOMAIN_LABEL.get(domain, domain.replace("_", " "))


def cell_text(v: float) -> str:
    x = 100 * v
    return f"{x:.0f}" if abs(x - round(x)) < 0.05 else f"{x:.1f}"


def draw(c: A.Campaign, out_pdf: Path) -> tuple[Path, Path]:
    admitted = sorted([r for r in c.routes if r.admitted], key=lambda r: (-r.overall, r.route))
    refused = sorted([r for r in c.routes if not r.admitted], key=lambda r: (-r.overall, r.route))
    ordered = admitted + refused
    gap = 0.35 if admitted and refused else 0.0
    ys = [i + (gap if i >= len(admitted) else 0.0) for i in range(len(ordered))]
    doms = c.domains
    acc = np.array([[r.accuracy(d) for d in doms] for r in ordered])
    rows_per = [ordered[0].domains[d]["rows"] for d in doms]

    height = 0.86 + 0.205 * (len(ordered) + gap)
    fig = plt.figure(figsize=(WIDTH_IN, height))
    left, right = 1.14 / WIDTH_IN, 1 - 0.50 / WIDTH_IN
    top, bottom = 1 - 0.80 / height, 0.04 / height
    ax = fig.add_axes([left, bottom, right - left, top - bottom])
    cmap = plt.get_cmap(CMAP)
    for yi, row in zip(ys, acc):
        for xi, v in enumerate(row):
            ax.add_patch(plt.Rectangle((xi, yi), 1, 1, facecolor=cmap(v), edgecolor="white", lw=0.8))
            r, g, b, _ = cmap(v)
            lum = 0.2126 * r + 0.7152 * g + 0.0722 * b
            ax.text(xi + 0.5, yi + 0.5, cell_text(v), ha="center", va="center", fontsize=FS,
                    color="black" if lum > 0.5 else "white")
    ax.set_xlim(0, len(doms))
    ax.set_ylim(ys[-1] + 1, 0)
    ax.set_xticks([i + 0.5 for i in range(len(doms))])
    ax.set_xticklabels([f"{domain_label(d)}\n(n={n})" for d, n in zip(doms, rows_per)], fontsize=FS,
                       rotation=90, ha="center", va="bottom", linespacing=1.0)
    ax.xaxis.set_ticks_position("top")
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([y + 0.5 for y in ys])
    ax.set_yticklabels([route_label(r.route) for r in ordered], fontsize=FS)
    ax.tick_params(axis="y", length=0, pad=2)
    for s in ax.spines.values():
        s.set_visible(False)

    ax.text(len(doms) + 0.42, -0.08, "Overall", rotation=90, ha="center", va="bottom", fontsize=FS)
    for yi, r in zip(ys, ordered):
        ax.text(len(doms) + 0.12, yi + 0.5, cell_text(r.overall), ha="left", va="center", fontsize=FS)
    bx = len(doms) + 1.02
    for group, label in ((admitted, "admitted"), (refused, "refused")):
        if not group:
            continue
        idx = [ordered.index(r) for r in group]
        y0, y1 = ys[idx[0]] + 0.08, ys[idx[-1]] + 0.92
        ax.plot([bx, bx], [y0, y1], color="black", lw=0.7, clip_on=False)
        ax.text(bx + 0.12, (y0 + y1) / 2, label, rotation=270, ha="left", va="center", fontsize=FS)
    if admitted and refused:
        ysep = ys[len(admitted)] - gap / 2
        ax.plot([0, len(doms)], [ysep, ysep], color="black", lw=0.6, ls=(0, (3, 2)), clip_on=False)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)
    out_png = out_pdf.with_suffix(".png")
    fig.savefig(out_pdf)
    fig.savefig(out_png, dpi=300)
    plt.close(fig)
    return out_pdf, out_png


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign-root", action="append", type=Path,
                    help="admission campaign directory; repeat to add routes (default: admission_campaign_v6)")
    ap.add_argument("--family-label", default=A.LEGACY_FAMILY)
    ap.add_argument("--expected-rows", type=int, default=60)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)
    roots = args.campaign_root or [A.DEFAULT_CAMPAIGN]
    c = A.load_campaign(roots, args.family_label, args.expected_rows)
    name = "domain_by_route.pdf" if args.family_label == A.LEGACY_FAMILY else f"domain_by_route_{args.family_label}.pdf"
    pdf, png = draw(c, args.output or DEFAULT_FIGDIR / name)
    print(f"wrote {A._rel(pdf)} and {A._rel(png)}: {len(c.routes)} routes x {len(c.domains)} domains")


if __name__ == "__main__":
    main()
