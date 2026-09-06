"""Build approval-only paper figure candidates without touching the manuscript."""
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
# Type 3 fonts are rejected by publishers and are matplotlib's default.
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

ROOT = Path(__file__).resolve().parents[2]
HERE = Path(__file__).resolve().parent
OUT = HERE / "figure_candidates_20260802"
OUT.mkdir(parents=True, exist_ok=True)

INK = "#172033"
GRID = "#DCE3EC"
RISK_COLORS = {0.1: "#12A594", 0.6: "#F59E0B", 0.9: "#E45756"}
MODEL_LABELS = {
    "gpt-5-nano": "GPT-5 nano",
    "gpt-5.4-nano": "GPT-5.4 nano",
    "google/gemini-3-flash-preview": "Gemini 3 Flash",
    "google/gemini-3.1-flash-lite-preview": "Gemini 3.1 Flash-Lite",
    "google/gemini-3.5-flash-lite": "Gemini 3.5 Flash-Lite",
    "gpt-5.6-luna": "GPT-5.6 Luna",
    "gpt-5.6-terra": "GPT-5.6 Terra",
    "claude-opus-5": "Claude Opus 5",
    "claude-sonnet-5": "Claude Sonnet 5",
}


def style():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9.5,
        "axes.titlesize": 11, "axes.labelsize": 10,
        "axes.edgecolor": GRID, "axes.labelcolor": INK,
        "xtick.color": "#536176", "ytick.color": "#536176",
        "text.color": INK, "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white", "savefig.bbox": "tight",
    })


def save(fig, stem):
    fig.savefig(OUT / f"{stem}.png", dpi=240, pad_inches=.12)
    fig.savefig(OUT / f"{stem}.pdf", pad_inches=.12)
    plt.close(fig)


def figure3():
    df = pd.read_csv(HERE / "data" / "risk_response_by_model.csv")
    order = (df.pivot(index="model", columns="max_private_risk", values="mean_unsafe_rate")
               .assign(delta=lambda x: x[0.9] - x[0.1]).sort_values("delta").index.tolist())
    fig, axes = plt.subplots(3, 3, figsize=(10.6, 7.15), sharex=True, sharey=True)
    for ax, model in zip(axes.flat, order):
        d = df[df.model == model].sort_values("max_private_risk")
        ax.plot(d.max_private_risk, d.mean_unsafe_rate, color="#50627A", lw=2.2, zorder=1)
        ax.scatter(d.max_private_risk, d.mean_unsafe_rate,
                   c=[RISK_COLORS[x] for x in d.max_private_risk], s=58,
                   edgecolor="white", linewidth=1.1, zorder=2)
        delta = d.iloc[-1].mean_unsafe_rate - d.iloc[0].mean_unsafe_rate
        ax.set_title(MODEL_LABELS.get(model, model), loc="left", weight="bold", pad=7)
        ax.text(.98, .92, f"Δ = {delta:+.0%}", transform=ax.transAxes, ha="right", va="top",
                color="#087F73" if delta < 0 else "#B45309", weight="bold", fontsize=9)
        ax.grid(axis="y", color=GRID, lw=.7)
        ax.set_xlim(.04, .96); ax.set_ylim(-.04, 1.05)
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0)
    for ax in axes[-1]: ax.set_xlabel("Maximum private setback risk")
    for ax in axes[:, 0]: ax.set_ylabel("Unsafe decisions")
    for ax in axes.flat:
        ax.set_xticks([.1, .6, .9]); ax.yaxis.set_major_formatter(PercentFormatter(1))
    fig.suptitle("The same risk signal produces sharply different strategic responses", x=.06,
                 ha="left", fontsize=17, weight="bold", y=1.015)
    fig.text(.06, .974, "Dyadic neutral-prompt pilots · lines connect observed condition means; Δ compares risk 0.9 with 0.1",
             color="#536176", fontsize=10)
    fig.subplots_adjust(top=.90, hspace=.42, wspace=.16)
    save(fig, "candidate_fig3_risk_response_small_multiples")


def bootstrap_game_ci(players, seed=27, n_boot=4000):
    rng = np.random.default_rng(seed)
    games = players.groupby("game_id", as_index=False).unsafe_frequency.mean()
    values = games.unsafe_frequency.to_numpy()
    means = np.mean(rng.choice(values, size=(n_boot, len(values)), replace=True), axis=1)
    return values.mean(), *np.quantile(means, [.025, .975])


def figure6():
    rows = []
    for n in (3, 4, 5):
        for model in ("gpt-5-nano", "gpt-5.4-nano"):
            p = pd.read_csv(ROOT / "results" / "nplayer" / "openai" / f"baseline_n{n}" / model / "players.csv")
            for risk, d in p.groupby("max_private_risk"):
                mean, lo, hi = bootstrap_game_ci(d, seed=2700 + n + int(risk * 10))
                rows.append((model, n, risk, mean, lo, hi, d.game_id.nunique()))
    out = pd.DataFrame(rows, columns=["model", "n_players", "risk", "mean", "lo", "hi", "n_races"])
    out.to_csv(OUT / "candidate_fig6_source.csv", index=False)
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.25), sharey=True)
    for ax, model in zip(axes, ("gpt-5-nano", "gpt-5.4-nano")):
        d = out[out.model == model]
        for risk in (.1, .6, .9):
            s = d[d.risk == risk].sort_values("n_players")
            c = RISK_COLORS[risk]
            ax.plot(s.n_players, s["mean"], marker="o", ms=6, lw=2.2, color=c, label=f"risk = {risk}")
            ax.fill_between(s.n_players, s.lo, s.hi, color=c, alpha=.13, linewidth=0)
        ax.set_title(MODEL_LABELS[model], loc="left", weight="bold", pad=9)
        ax.set_xticks([3, 4, 5]); ax.set_xlabel("Number of players")
        ax.grid(axis="y", color=GRID, lw=.7); ax.spines[["top", "right", "left"]].set_visible(False)
        ax.tick_params(axis="y", length=0); ax.set_ylim(0, 1)
    axes[0].set_ylabel("Unsafe decisions"); axes[0].yaxis.set_major_formatter(PercentFormatter(1))
    axes[1].legend(frameon=False, loc="lower right", title="Private setback")
    fig.suptitle("Group size does not move both checkpoints in the same direction", x=.07,
                 ha="left", fontsize=17, weight="bold", y=1.03)
    fig.text(.07, .95, "Neutral N-player protocol · 10 race clusters per model × risk × N cell · bands are race-bootstrap 95% intervals",
             color="#536176", fontsize=10)
    fig.text(.07, -.01, "N=2 is intentionally not connected: its dyadic engine, prompt, and payoff protocol differ from this N-player experiment.",
             color="#7C3AED", fontsize=9.3, weight="bold")
    fig.subplots_adjust(top=.82, bottom=.18, wspace=.13)
    save(fig, "candidate_fig6_group_size_response")


def figure7():
    d = pd.read_csv(ROOT / "results" / "open_source" / "egt_reproduction" / "theory_llm_comparison.csv")
    rows = []
    for _, r in d.iterrows():
        for protocol, obs_col in (("Primary, T=0", "llm_primary_t0_unsafe_technology_race"),
                                  ("Sensitivity, T=.7", "llm_sensitivity_t07_unsafe_technology_race")):
            rows.append((r.max_private_risk, protocol,
                         r[obs_col] - r.theory_unsafe_main_reference))
    x = pd.DataFrame(rows, columns=["risk", "protocol", "gap"])
    x.to_csv(OUT / "candidate_fig7_source.csv", index=False)
    fig, ax = plt.subplots(figsize=(8.8, 4.35))
    offsets = {"Primary, T=0": -.035, "Sensitivity, T=.7": .035}
    colors = {"Primary, T=0": "#335CFF", "Sensitivity, T=.7": "#D04A9B"}
    for protocol in offsets:
        s = x[x.protocol == protocol]
        ax.scatter(s.risk + offsets[protocol], s.gap, s=75, color=colors[protocol],
                   edgecolor="white", linewidth=1, label=protocol, zorder=3)
        for _, r in s.iterrows():
            ax.vlines(r.risk + offsets[protocol], 0, r.gap, color=colors[protocol], alpha=.35, lw=2)
    ax.axhline(0, color=INK, lw=1.1)
    ax.axhspan(-1, 0, color="#EEF4FF", alpha=.6, zorder=0)
    ax.set_xticks([.1, .6, .9]); ax.set_xlim(.02, .98); ax.set_ylim(-1.05, .2)
    ax.set_xlabel("Maximum private setback risk"); ax.set_ylabel("Observed − EGT-predicted Unsafe rate")
    ax.yaxis.set_major_formatter(PercentFormatter(1)); ax.grid(axis="y", color=GRID, lw=.7)
    ax.spines[["top", "right", "left"]].set_visible(False); ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, ncol=2, loc="lower right")
    ax.set_title("LLM play departs from the EGT reference most at low and medium risk", loc="left",
                 fontsize=15.5, weight="bold", pad=22)
    ax.text(.0, 1.02, "Zero means agreement; negative values mean less Unsafe play than the stationary reference",
            transform=ax.transAxes, color="#536176", fontsize=10)
    save(fig, "candidate_fig7_egt_discrepancy")


def figure3_composite():
    """Heatmap + response magnitude strip: compact, comparative, no line-chart clutter."""
    df = pd.read_csv(HERE / "data" / "risk_response_by_model.csv")
    p = df.pivot(index="model", columns="max_private_risk", values="mean_unsafe_rate")
    p["delta"] = p[0.9] - p[0.1]
    p = p.sort_values("delta")
    rates = p[[0.1, 0.6, 0.9]].to_numpy()
    deltas = p[["delta"]].to_numpy()
    cmap = LinearSegmentedColormap.from_list("unsafe", ["#F4FBFA", "#BCE7DE", "#F5C66D", "#E85B57"])
    dcmap = LinearSegmentedColormap.from_list("delta", ["#087F73", "#E8EDF3", "#B45309"])
    fig = plt.figure(figsize=(10.7, 6.6))
    gs = fig.add_gridspec(1, 2, width_ratios=[3.25, 1], wspace=.10,
                          left=.28, right=.94, top=.82, bottom=.13)
    ax = fig.add_subplot(gs[0])
    ad = fig.add_subplot(gs[1], sharey=ax)
    ax.imshow(rates, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ad.imshow(deltas, aspect="auto", cmap=dcmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=.15))
    for i in range(len(p)):
        for j in range(3):
            v = rates[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", weight="bold",
                    color="white" if v > .72 else INK, fontsize=10)
        v = deltas[i, 0]
        ad.text(0, i, f"{v:+.0%}", ha="center", va="center", weight="bold",
                color="white" if abs(v) > .55 else INK, fontsize=10)
    labels = [MODEL_LABELS.get(m, m) for m in p.index]
    ax.set_yticks(range(len(p)), labels=labels); ad.tick_params(labelleft=False)
    ax.set_xticks(range(3), ["Low\n0.1", "Medium\n0.6", "High\n0.9"])
    ad.set_xticks([0], ["Response\nΔ high − low"])
    for a in (ax, ad):
        a.tick_params(length=0, pad=8); a.spines[:].set_visible(False)
        a.set_yticks(np.arange(-.5, len(p), 1), minor=True)
        a.grid(which="minor", axis="y", color="white", linewidth=4)
    ax.set_title("Observed Unsafe rate", loc="left", weight="bold", pad=14)
    ad.set_title("Risk response", loc="center", weight="bold", pad=14)
    fig.suptitle("Risk response is a model fingerprint—not a shared safety curve", x=.055,
                 ha="left", fontsize=17, weight="bold", y=.965)
    fig.text(.055, .895, "Dyadic neutral-prompt pilots · cells are condition means · models ordered by high-minus-low response",
             fontsize=10.2, color="#536176")
    fig.text(.055, .045, "Reading guide: teal Δ = Unsafe play falls as disclosed setback risk rises; orange Δ = it rises.",
             fontsize=9.5, color="#536176")
    save(fig, "candidate_fig3_composite_fingerprint")


def figure6_composite():
    """Bubble matrix makes the model-level contrast primary and avoids implying smooth N trends."""
    src = OUT / "candidate_fig6_source.csv"
    if not src.exists():
        figure6()
    d = pd.read_csv(src)
    cmap = LinearSegmentedColormap.from_list("unsafe", ["#E7F7F4", "#F6C86F", "#E55754"])
    fig, axes = plt.subplots(1, 2, figsize=(10.7, 4.85), sharey=True)
    for ax, model in zip(axes, ("gpt-5-nano", "gpt-5.4-nano")):
        s = d[d.model == model]
        size = 900 + 2800 * (s.hi - s.lo)
        ax.scatter(s.n_players, s.risk, s=size, c=s["mean"], cmap=cmap, vmin=0, vmax=1,
                   edgecolor="white", linewidth=2.3)
        for r in s.itertuples():
            ax.text(r.n_players, r.risk, f"{r.mean:.0%}", ha="center", va="center",
                    weight="bold", color="white" if r.mean > .68 else INK, fontsize=10)
        ax.set_title(MODEL_LABELS[model], loc="left", weight="bold", pad=15)
        ax.set_xticks([3, 4, 5]); ax.set_xlabel("Players in race")
        ax.set_yticks([.1, .6, .9], ["Low · 0.1", "Medium · 0.6", "High · 0.9"])
        ax.set_xlim(2.55, 5.45); ax.set_ylim(.0, 1.0)
        ax.grid(color=GRID, linewidth=.8); ax.set_axisbelow(True)
        ax.spines[:].set_visible(False); ax.tick_params(length=0, pad=7)
    axes[0].set_ylabel("Maximum private setback risk")
    fig.suptitle("Checkpoint identity dominates the N=3–5 group-size perturbation", x=.055,
                 ha="left", fontsize=17, weight="bold", y=1.02)
    fig.text(.055, .935, "Bubble colour = Unsafe rate · bubble area = race-bootstrap interval width · 10 race clusters per cell",
             color="#536176", fontsize=10.2)
    fig.text(.055, .02, "N=2 remains a separate dyadic reference because its engine, prompt, and payoff protocol are not harmonized.",
             color="#7C3AED", fontsize=9.4, weight="bold")
    fig.subplots_adjust(top=.82, bottom=.20, left=.12, right=.97, wspace=.22)
    save(fig, "candidate_fig6_composite_bubble_matrix")


def figure7_composite():
    """Three-layer tile: theory, observed protocols, and signed mismatch in one reading path."""
    d = pd.read_csv(ROOT / "results" / "open_source" / "egt_reproduction" / "theory_llm_comparison.csv")
    risks = d.max_private_risk.to_numpy()
    theory = d.theory_unsafe_main_reference.to_numpy()
    primary = d.llm_primary_t0_unsafe_technology_race.to_numpy()
    sensitivity = d.llm_sensitivity_t07_unsafe_technology_race.to_numpy()
    values = np.vstack([theory, primary, sensitivity])
    gaps = np.vstack([primary - theory, sensitivity - theory])
    cmap = LinearSegmentedColormap.from_list("unsafe", ["#EDF7F5", "#F7CD78", "#E55754"])
    dcmap = LinearSegmentedColormap.from_list("gap", ["#315EF4", "#F5F7FA", "#D44796"])
    fig = plt.figure(figsize=(10.5, 5.25))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.35, 1], left=.21, right=.96,
                          top=.78, bottom=.17, wspace=.16)
    ax, ag = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    ax.imshow(values, aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ag.imshow(gaps, aspect="auto", cmap=dcmap, norm=TwoSlopeNorm(vmin=-1, vcenter=0, vmax=.15))
    for i in range(3):
        for j in range(3):
            v = values[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", weight="bold",
                    color="white" if v > .74 else INK, fontsize=11)
    for i in range(2):
        for j in range(3):
            v = gaps[i, j]
            ag.text(j, i, f"{v:+.0%}", ha="center", va="center", weight="bold",
                    color="white" if abs(v) > .55 else INK, fontsize=11)
    ax.set_yticks(range(3), ["EGT stationary reference", "LLM primary · T=0", "LLM sensitivity · T=.7"])
    ag.set_yticks(range(2), ["Primary − EGT", "Sensitivity − EGT"])
    for a in (ax, ag):
        a.set_xticks(range(3), [f"risk {r:g}" for r in risks]); a.tick_params(length=0, pad=9)
        a.spines[:].set_visible(False)
        a.set_xticks(np.arange(-.5, 3, 1), minor=True); a.set_yticks(np.arange(-.5, a.images[0].get_array().shape[0], 1), minor=True)
        a.grid(which="minor", color="white", linewidth=5); a.tick_params(which="minor", length=0)
    ax.set_title("Rates", loc="left", weight="bold", pad=14)
    ag.set_title("Signed mismatch", loc="left", weight="bold", pad=14)
    fig.suptitle("The EGT reference and LLM policy agree only near the high-risk boundary", x=.055,
                 ha="left", fontsize=16.8, weight="bold", y=.96)
    fig.text(.055, .875, "One reading path: stationary benchmark → observed protocol → observed-minus-theory diagnostic",
             color="#536176", fontsize=10.2)
    fig.text(.055, .055, "Blue mismatch cells mean the LLM chose Unsafe less often than the EGT stationary reference.",
             color="#315EF4", fontsize=9.5, weight="bold")
    save(fig, "candidate_fig7_composite_theory_observation_gap")


if __name__ == "__main__":
    style(); figure3(); figure6(); figure7()
    figure3_composite(); figure6_composite(); figure7_composite()
    print(OUT)
