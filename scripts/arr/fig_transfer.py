#!/usr/bin/env python3
"""Figures for the ARR transfer campaign, drawn from the outputs of analyze_transfer.py.

transfer_heatmap: routes complete in all three families, ordered by F1 overall accuracy;
    column blocks F1 | F2 | F3 with the six domains each (cell = accuracy in percent) and a
    marker column per block for that family's admitted routes.
transfer_scatter: state_reconstruction accuracy family vs family over the same routes;
    coincident routes are drawn as one marker whose area and label give the count.
item_load: exploratory item accuracy against arithmetic term count, coloured by family.

The figures read results/derived/arr_transfer/ only, so they cannot disagree with the
tables.  No titles inside the figures: captions live in LaTeX.
"""
from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DERIVED = ROOT / "results" / "derived" / "arr_transfer"
FIGDIR = ROOT / "figures" / "arr"
FS = 7.0
CMAP = "cividis"
FAMILIES = ("F1", "F2", "F3")
DOMAINS = ("rule_recall", "stage_payoff", "state_reconstruction", "state_transition",
           "terminal_scoring", "expected_payoff")
SR = "state_reconstruction"
FAMILY_COLOR = {"F1": "#0072B2", "F2": "#E69F00", "F3": "#009E73"}
FAMILY_MARKER = {"F1": "o", "F2": "s", "F3": "^"}
FAMILY_NAME = {"F1": "F1 race", "F2": "F2 dilemma", "F3": "F3 commons"}

plt.rcParams.update({
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": FS,
    "axes.labelsize": FS,
    "xtick.labelsize": FS,
    "ytick.labelsize": FS,
    "legend.fontsize": FS,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
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
    "claude-opus-5-default": "Claude Opus 5",
    "claude-sonnet-5-default": "Claude Sonnet 5",
    "claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "gemini-2.5-flash": "Gemini 2.5 Flash",
    "gemini-3-flash-preview": "Gemini 3 Flash",
    "gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash Lite",
    "gemini-3.5-flash": "Gemini 3.5 Flash",
    "gemini-3.5-flash-lite": "Gemini 3.5 Flash Lite",
    "gpt-5.4-2026-03-05": "GPT-5.4",
    "gpt-5.4-mini-2026-03-17": "GPT-5.4 mini",
    "gpt-5.4-nano-2026-03-17": "GPT-5.4 nano",
    "gpt-5.5-2026-04-23": "GPT-5.5",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "gpt-oss-20b": "gpt-oss-20b",
    "grok-4.20-0309-non-reasoning": "Grok 4.20 (non-reas.)",
    "qwen3-235b-a22b-instruct-2507": "Qwen3 235B A22B",
}


def load():
    res = json.loads((DERIVED / "transfer.json").read_text(encoding="utf-8"))
    acc = {}
    with (DERIVED / "route_family_domain.csv").open(encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            acc[(r["route"], r["family"], r["domain"])] = (float(r["accuracy"]), r["admitted"] == "True")
    with (DERIVED / "item_analysis.csv").open(encoding="utf-8") as fh:
        items = list(csv.DictReader(fh))
    return res, acc, items


def save(fig, name: str) -> None:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{name}.pdf")
    fig.savefig(FIGDIR / f"{name}.png", dpi=300)
    plt.close(fig)
    print(f"wrote figures/arr/{name}.pdf and .png")


def heatmap(res, acc) -> None:
    routes = sorted(res["transfer"]["routes_all_three"], key=lambda r: (-acc[(r, "F1", "overall")][0], r))
    nd = len(DOMAINS)
    mark_w, gap = 0.8, 0.45
    block_w = nd + mark_w
    x0 = {f: i * (block_w + gap) for i, f in enumerate(FAMILIES)}
    total_w = len(FAMILIES) * block_w + (len(FAMILIES) - 1) * gap

    width = 6.3
    row_h = 0.19
    head = 0.98
    height = head + row_h * len(routes) + 0.08
    fig = plt.figure(figsize=(width, height))
    left_in, right_in = 1.22, 0.05
    ax = fig.add_axes([left_in / width, 0.08 / height, 1 - (left_in + right_in) / width, (row_h * len(routes)) / height])
    cmap = plt.get_cmap(CMAP)
    for yi, r in enumerate(routes):
        for f in FAMILIES:
            for xi, d in enumerate(DOMAINS):
                v, _ = acc[(r, f, d)]
                ax.add_patch(plt.Rectangle((x0[f] + xi, yi), 1, 1, facecolor=cmap(v), edgecolor="white", lw=0.6))
                cr, cg, cb, _ = cmap(v)
                lum = 0.2126 * cr + 0.7152 * cg + 0.0722 * cb
                ax.text(x0[f] + xi + 0.5, yi + 0.52, f"{100 * v:.0f}", ha="center", va="center", fontsize=FS,
                        color="black" if lum > 0.5 else "white")
            if acc[(r, f, "overall")][1]:
                ax.plot(x0[f] + nd + mark_w / 2, yi + 0.5, marker="o", ms=4.2, color="black", mec="black")
    ax.set_xlim(0, total_w)
    ax.set_ylim(len(routes), 0)
    ax.set_yticks([i + 0.5 for i in range(len(routes))])
    ax.set_yticklabels([ROUTE_LABEL.get(r, r) for r in routes])
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xticks([])
    for s in ax.spines.values():
        s.set_visible(False)
    for f in FAMILIES:
        for xi, d in enumerate(DOMAINS):
            ax.text(x0[f] + xi + 0.5, -0.15, DOMAIN_LABEL[d], rotation=90, ha="center", va="bottom", fontsize=FS)
        ax.text(x0[f] + nd + mark_w / 2, -0.15, "Admitted", rotation=90, ha="center", va="bottom", fontsize=FS)
        n_adm = sum(acc[(r, f, "overall")][1] for r in routes)
        ytop = -4.55
        ax.plot([x0[f] + 0.05, x0[f] + block_w - 0.05], [ytop + 0.35, ytop + 0.35], color="black", lw=0.7, clip_on=False)
        ax.text(x0[f] + block_w / 2, ytop, f"{FAMILY_NAME[f]} ({n_adm} of {len(routes)} admitted)",
                ha="center", va="bottom", fontsize=FS)
    save(fig, "transfer_heatmap")


def scatter(res, acc) -> None:
    routes = res["transfer"]["routes_all_three"]
    corr = {(r["pair"], r["scope"], r["metric"]): r for r in res["transfer"]["correlations"]}
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.1))
    fig.subplots_adjust(left=0.075, right=0.99, bottom=0.19, top=0.98, wspace=0.32)
    for ax, (a, b) in zip(axes, (("F1", "F2"), ("F1", "F3"), ("F2", "F3"))):
        pts = Counter((round(acc[(r, a, SR)][0], 4), round(acc[(r, b, SR)][0], 4)) for r in routes)
        ax.plot([0, 1], [0, 1], color="0.75", lw=0.6, ls=(0, (3, 2)), zorder=0)
        ax.axvline(0.75, color="0.55", lw=0.5, ls=":", zorder=0)
        ax.axhline(0.75, color="0.55", lw=0.5, ls=":", zorder=0)
        for (x, y), k in pts.items():
            ax.scatter([x], [y], s=12 if k == 1 else 48, color="#0072B2", edgecolor="white", lw=0.4, zorder=2)
            if k > 1:
                ax.text(x, y, str(k), fontsize=FS, color="white", ha="center", va="center_baseline",
                        fontweight="bold", zorder=3)
        c = corr[(f"{a}-{b}", "all_three", SR)]
        ax.text(0.03, 0.97, f"rho = {c['rho']:.2f}\n[{c['ci_low']:.2f}, {c['ci_high']:.2f}]\nn = {c['n']}",
                transform=ax.transAxes, ha="left", va="top", fontsize=FS, linespacing=1.15)
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_yticks([0, 0.25, 0.5, 0.75, 1])
        ax.set_xticklabels(["0", ".25", ".5", ".75", "1"])
        ax.set_yticklabels(["0", ".25", ".5", ".75", "1"])
        ax.set_xlabel(f"{a} state recon.")
        ax.set_ylabel(f"{b} state recon.", labelpad=1)
        ax.set_aspect("equal")
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    save(fig, "transfer_scatter")


def item_load(res, items) -> None:
    fig, ax = plt.subplots(figsize=(3.2, 2.2))
    fig.subplots_adjust(left=0.15, right=0.98, bottom=0.17, top=0.88)
    offset = {"F1": -0.22, "F2": 0.0, "F3": 0.22}
    for f in FAMILIES:
        sel = [x for x in items if x["family"] == f]
        ax.scatter([int(x["term_count"]) + offset[f] for x in sel], [float(x["accuracy_common_routes"]) for x in sel],
                   s=16, marker=FAMILY_MARKER[f], color=FAMILY_COLOR[f], alpha=0.85, edgecolor="white", lw=0.3,
                   label=FAMILY_NAME[f], zorder=2)
    t = res["item_tests"]["all_60"]
    ax.text(0.98, 0.97, f"rho = {t['rho']:.2f}, n = {t['n']}", transform=ax.transAxes, ha="right", va="top", fontsize=FS)
    ax.set_xlabel("Arithmetic terms a correct answer needs")
    ax.set_ylabel("Item accuracy")
    ax.set_ylim(-0.04, 1.04)
    ax.set_xlim(0, 20)
    ax.set_xticks([1, 5, 10, 15, 19])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles = [Line2D([], [], ls="", marker=FAMILY_MARKER[f], color=FAMILY_COLOR[f], ms=4.5, label=FAMILY_NAME[f])
               for f in FAMILIES]
    ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.0, 1.01), ncol=3, frameon=False,
              handletextpad=0.2, columnspacing=0.9, borderaxespad=0.0)
    save(fig, "item_load")


def main() -> None:
    res, acc, items = load()
    heatmap(res, acc)
    scatter(res, acc)
    item_load(res, items)


if __name__ == "__main__":
    main()
