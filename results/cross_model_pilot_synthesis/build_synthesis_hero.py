"""Build an approval-only cross-protocol synthesis figure.

The panels share an Unsafe-rate scale but do not treat EGT, human, dyadic LLM,
and N-player pilots as exchangeable samples. Protocol boundaries are explicit.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Type 3 fonts are rejected by publishers and are matplotlib's default.
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

from matplotlib.colors import LinearSegmentedColormap

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "figure_candidates_20260802"
OUT.mkdir(parents=True, exist_ok=True)

INK, MUTED, GRID = "#152036", "#5C6B82", "#DFE6EF"
CMAP = LinearSegmentedColormap.from_list("unsafe", ["#EEF8F6", "#BEE7DF", "#F4C66C", "#E95A55"])
NAMES = {
    "gpt-5-nano": "GPT-5 nano", "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna", "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5", "claude-sonnet-5": "Claude Sonnet 5",
}


def setup():
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9,
                         "axes.edgecolor": GRID, "text.color": INK,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED, "figure.facecolor": "white",
                         "axes.facecolor": "white", "savefig.facecolor": "white",
                         "savefig.bbox": "tight"})


def tile(ax, array, row_labels, col_labels, title, values=True):
    a = np.asarray(array, dtype=float)
    ax.imshow(a, aspect="auto", cmap=CMAP, vmin=0, vmax=1)
    if values:
        for i in range(a.shape[0]):
            for j in range(a.shape[1]):
                v = a[i, j]
                ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=8.2,
                        weight="bold", color="white" if v > .74 else INK)
    ax.set_yticks(range(len(row_labels)), row_labels)
    ax.set_xticks(range(len(col_labels)), col_labels)
    ax.tick_params(length=0, pad=5)
    ax.set_xticks(np.arange(-.5, len(col_labels), 1), minor=True)
    ax.set_yticks(np.arange(-.5, len(row_labels), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=3)
    ax.tick_params(which="minor", length=0)
    ax.spines[:].set_visible(False)
    ax.set_title(title, loc="left", fontsize=11.5, weight="bold", pad=9)


def build():
    risk = pd.read_csv(HERE / "data" / "risk_response_by_model.csv")
    persona = pd.read_csv(HERE / "data" / "persona_role_gradient_extended.csv")
    nsrc = pd.read_csv(OUT / "candidate_fig6_source.csv")
    egt = pd.read_csv(ROOT / "results/open_source/egt_reproduction/egt_stationary_summary.csv")
    egt = egt[egt.regime == "main_reference"].sort_values("max_private_risk")
    with open(HERE / "data/human_vs_llm_validation.json", encoding="utf-8") as f:
        human = json.load(f)["fig2a_mean_by_risk"]

    risk_p = risk.pivot(index="model", columns="max_private_risk", values="mean_unsafe_rate")
    model_order = risk_p.assign(span=lambda x: x.max(axis=1)-x.min(axis=1)).sort_values("span", ascending=False).index
    benchmark = np.vstack([
        egt.unsafe_frequency_mean.to_numpy(),
        np.array([human[str(x)] for x in (.1, .6, .9)]),
        risk_p.loc[model_order, [.1, .6, .9]].to_numpy(),
    ])
    benchmark_labels = ["EGT stationary†", "Human reference‡"] + [NAMES[m] for m in model_order]

    pp = persona[persona.role != "none"].pivot(index="model", columns="role", values="mean_unsafe_rate")
    pp = pp.loc[sorted(pp.index, key=lambda m: pp.loc[m, "R6"]-pp.loc[m, "R1"], reverse=True),
                [f"R{i}" for i in range(1, 7)]]

    # N-player rows are model × risk; columns are N=3,4,5.
    nrows, nlabels = [], []
    for model in ("gpt-5-nano", "gpt-5.4-nano"):
        for rv in (.1, .6, .9):
            s = nsrc[(nsrc.model == model) & (nsrc.risk == rv)].sort_values("n_players")
            nrows.append(s["mean"].to_numpy())
            short = "GPT-5" if model == "gpt-5-nano" else "GPT-5.4"
            nlabels.append(f"{short} · r={rv}")

    # Descriptive spans; each dot is one eligible model or model×risk cell.
    risk_spans = risk.groupby("model").mean_unsafe_rate.agg(lambda x: x.max()-x.min()).to_numpy()
    persona_spans = persona[persona.role != "none"].groupby("model").mean_unsafe_rate.agg(lambda x: x.max()-x.min()).to_numpy()
    n_spans = nsrc.groupby(["model", "risk"])["mean"].agg(lambda x: x.max()-x.min()).to_numpy()
    cross_model = risk.groupby("max_private_risk").mean_unsafe_rate.agg(lambda x: x.max()-x.min()).to_numpy()
    span_sets = [cross_model, persona_spans, risk_spans, n_spans]
    span_labels = ["Checkpoint spread\nat fixed risk", "Persona R1–R6\nwithin checkpoint",
                   "Risk 0.1–0.9\nwithin checkpoint", "Group size N=3–5\nwithin checkpoint+risk"]

    fig = plt.figure(figsize=(14.8, 9.2))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.34, 1.08, 1.18], height_ratios=[1.25, 1],
                          left=.105, right=.975, top=.865, bottom=.10, wspace=.48, hspace=.42)
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1:])
    ax_c = fig.add_subplot(gs[1, 1])
    ax_d = fig.add_subplot(gs[1, 2])

    tile(ax_a, benchmark, benchmark_labels, ["risk .1", "risk .6", "risk .9"],
         "A  Dyadic (N=2) risk fingerprints across benchmark layers")
    ax_a.axhline(1.5, color=INK, lw=1.2)

    tile(ax_b, pp.to_numpy(), [NAMES[m] for m in pp.index], list(pp.columns),
         "B  Persona is a policy intervention, not cosmetic wording")
    ax_b.set_xlabel("Risk-seeking role strength  →", weight="bold", labelpad=7)

    tile(ax_c, np.vstack(nrows), nlabels, ["N=3", "N=4", "N=5"],
         "C  N-player robustness")

    colors = ["#7C3AED", "#D34A91", "#0F9D8B", "#F09A22"]
    rng = np.random.default_rng(20260802)
    for i, (vals, lab, color) in enumerate(zip(span_sets, span_labels, colors)):
        y = i + rng.uniform(-.10, .10, len(vals))
        ax_d.scatter(vals, y, s=42, color=color, alpha=.72, edgecolor="white", linewidth=.7)
        med = np.median(vals)
        ax_d.plot([med, med], [i-.22, i+.22], color=INK, lw=2.4)
        ax_d.text(min(1.02, med+.035), i-.25, f"median {med:.0%}", fontsize=7.7, weight="bold")
    ax_d.set_yticks(range(4), span_labels, fontsize=7.6); ax_d.set_xlim(0, 1.05); ax_d.set_ylim(3.55, -.5)
    ax_d.set_xticks([0, .25, .5, .75, 1], ["0", "25", "50", "75", "100 pp"])
    ax_d.grid(axis="x", color=GRID, lw=.8); ax_d.set_axisbelow(True)
    ax_d.spines[["top", "right", "left"]].set_visible(False); ax_d.tick_params(axis="y", length=0)
    ax_d.set_title("D  Descriptive policy span", loc="left", fontsize=11.5, weight="bold", pad=9)

    fig.suptitle("AI-race behavior has no universal safety ordering", x=.035, ha="left",
                 fontsize=21, weight="bold", y=.972)
    fig.text(.035, .917,
             "Checkpoint and persona can move Unsafe play by most of the scale; group-size associations are smaller in the current N=3–5 pilot",
             fontsize=11.2, color=MUTED)
    fig.text(.035, .035,
             "† EGT is a stationary population benchmark, not an LLM prediction.  ‡ Human data are an external reference. "
             "All spans are descriptive; protocol-separated panels must not be read as one pooled causal model.",
             fontsize=8.6, color=MUTED)

    png, pdf = OUT / "candidate_synthesis_hero.png", OUT / "candidate_synthesis_hero.pdf"
    fig.savefig(png, dpi=240, pad_inches=.12); fig.savefig(pdf, pad_inches=.12); plt.close(fig)

    pd.DataFrame({"span_family": np.repeat(span_labels, [len(x) for x in span_sets]),
                  "span": np.concatenate(span_sets)}).to_csv(OUT / "candidate_synthesis_span_source.csv", index=False)
    print(png)


if __name__ == "__main__":
    setup(); build()
