#!/usr/bin/env python3
"""Best-response and quantal-response baseline for the scripted-opponent campaign.

A positive Always-Unsafe minus Always-Safe contrast, and a rival-by-risk
interaction, are what a payoff-maximising agent would also produce, because the
mechanism itself makes the best move depend on the rival and on the stated risk.
This script solves that single-agent decision problem exactly, so the observed
endpoint rates can be read against a rational benchmark instead of against zero.

Decision problem. One company faces a fixed scripted rival (AS, AU, CS, CAS).
Round payoffs are linear: S = r + 1.4 u - 0.4 v for r rounds, u own Unsafe moves
and v rival Unsafe moves. Progress is r + 0.5 u against r + 0.5 v, so the race
winner is decided by u against v. The horizon is hidden: no stop before round
min_rounds, then a stop with probability stop_probability after each round, with
the engine's forced stop at the safety cap. The terminal payoff is
(S + prize) (1 - risk) for a winner, (S + prize / 2) (1 - risk) for a tie and S for
a loser, with risk = max_private_risk * u / r. The state (r, u, v, own last move)
is therefore sufficient, and backward induction is exact.

Outputs: results/derived/best_response/best_response.json, best_response.csv,
quantal_response.csv, endpoint_vs_best_response.csv, best_response.pdf/png.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

OUT = ROOT / "results" / "derived" / "best_response"
STRATEGIES = ("AS", "AU", "CS", "CAS")
RISKS = (0.1, 0.6, 0.9)
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0)
TIE_TOL = 1e-9
TAIL_TARGET = 1e-9

SAFE_STAGE = {(0, 0): 1.0, (0, 1): 0.6, (1, 0): 2.4, (1, 1): 2.0}
PROGRESS = {0: 1.0, 1: 1.5}


def rival_move(strategy: str, r: int, last: int) -> int:
    """Rival move in round r + 1 given the solver's own last move (round r)."""
    if strategy == "AS":
        return 0
    if strategy == "AU":
        return 1
    if r == 0:
        return 0 if strategy == "CS" else 1
    return last


def terminal_value(r, u, v, risk_max, prize):
    """Payoff when the race stops after r rounds, over a grid of (u, v)."""
    stage = r + 1.4 * u - 0.4 * v
    risk = np.clip(risk_max * u / r, 0.0, 1.0)
    win = (r + 0.5 * u) > (r + 0.5 * v) + 1e-9
    tie = np.abs(u - v) * 0.5 <= 1e-9
    prize_part = np.where(win, prize, np.where(tie, prize / 2.0, 0.0))
    eligible = win | tie
    return np.where(eligible, (stage + prize_part) * (1.0 - risk), stage + prize_part)


def _shift(m, du, dv):
    out = np.zeros_like(m)
    n = m.shape[0]
    out[: n - du, : n - dv] = m[du:, dv:]
    return out


def solve(strategy, risk_max, *, lam=None, tie="mix", min_rounds=5, stop_prob=0.2,
          cap=100, prize=100.0):
    """Backward induction. lam=None is the exact best response; else softmax(lam*Q).

    Returns a dict with the value, the policy table pi[r][last] (probability of
    Unsafe over a (u, v) grid), and the root Q values.
    """
    n = cap + 3
    uu, vv = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float),
                         indexing="ij")
    W_next = None
    X_next = None
    pol = {}
    q_root = None
    for r in range(cap, -1, -1):
        if r == cap:
            term = terminal_value(r, uu, vv, risk_max, prize)
            W = np.stack([term, term])
            W_next = W
            X_next = np.zeros_like(W)
            continue
        if r >= 1:
            term = terminal_value(r, uu, vv, risk_max, prize)
        Wr = np.zeros((2, n, n))
        Xr = np.zeros((2, n, n))
        pol_r = np.zeros((2, n, n))
        for last in (0, 1):
            b = rival_move(strategy, r, last)
            q = [_shift(W_next[a], a, b) for a in (0, 1)]
            x = [_shift(X_next[a], a, b) + a for a in (0, 1)]
            if lam is None:
                diff = q[1] - q[0]
                tied = np.where(tie == "min", np.where(x[1] < x[0] - TIE_TOL, 1.0, 0.0),
                                np.where(x[1] > x[0] + TIE_TOL, 1.0, 0.0)) if tie != "mix" else 0.5
                p_unsafe = np.where(diff > TIE_TOL, 1.0, np.where(diff < -TIE_TOL, 0.0, tied))
            else:
                z = np.clip(lam * (q[1] - q[0]), -700, 700)
                p_unsafe = 1.0 / (1.0 + np.exp(-z))
            cont = (1.0 - p_unsafe) * q[0] + p_unsafe * q[1]
            xcont = (1.0 - p_unsafe) * x[0] + p_unsafe * x[1]
            if r == 0:
                q_root = (float(q[0][0, 0]), float(q[1][0, 0]), float(p_unsafe[0, 0]))
                Wr[last] = cont
            elif r >= min_rounds:
                Wr[last] = stop_prob * term + (1.0 - stop_prob) * cont
                Xr[last] = (1.0 - stop_prob) * xcont
            else:
                Wr[last] = cont
                Xr[last] = xcont
            pol_r[last] = p_unsafe
        pol[r] = pol_r
        W_next = Wr
        X_next = Xr
    value = float(W_next[0][0, 0])
    return {"value": value, "policy": pol, "q_root": q_root, "n": n, "cap": cap,
            "min_rounds": min_rounds, "stop_prob": stop_prob, "strategy": strategy,
            "risk": risk_max, "prize": prize}


def forward(sol):
    """Race statistics under the solved policy: exact forward distribution."""
    strategy, cap, n = sol["strategy"], sol["cap"], sol["n"]
    min_rounds, stop_prob = sol["min_rounds"], sol["stop_prob"]
    # mass[last] over (u, v): probability of the start of round r + 1 being reached
    mass = np.zeros((2, n, n))
    mass[0, 0, 0] = 1.0
    exp_unsafe = 0.0
    exp_rounds = 0.0
    race_mean_rate = 0.0
    p_unsafe_round1 = None
    p_by_round = []
    for r in range(0, cap):
        pol_r = sol["policy"][r]
        reach = float(mass.sum())
        exp_rounds += reach
        un = float((mass * pol_r).sum())
        exp_unsafe += un
        p_by_round.append(un / reach if reach > 0 else 0.0)
        if r == 0:
            p_unsafe_round1 = float((mass * pol_r).sum())
        new = np.zeros_like(mass)
        for last in (0, 1):
            b = rival_move(strategy, r, last)
            for a in (0, 1):
                pa = pol_r[last] if a == 1 else 1.0 - pol_r[last]
                new[a] += _shift_back(mass[last] * pa, a, b)
        r1 = r + 1
        if r1 >= min_rounds:
            stop_mass = new * (1.0 if r1 >= cap else stop_prob)
            if r1 >= 1:
                uidx = np.arange(n, dtype=float)[:, None]
                race_mean_rate += float((stop_mass.sum(axis=0) * (uidx / r1)).sum())
            new = new - stop_mass
        mass = new
    exp_rounds += 0.0
    return {
        "expected_rounds": exp_rounds,
        "expected_unsafe_moves": exp_unsafe,
        "pooled_unsafe_rate": exp_unsafe / exp_rounds,
        "race_mean_unsafe_rate": race_mean_rate,
        "first_move_unsafe_prob": p_unsafe_round1,
        "unsafe_prob_by_round": p_by_round[:12],
        "residual_mass": float(mass.sum()),
    }


def _shift_back(m, du, dv):
    out = np.zeros_like(m)
    n = m.shape[0]
    out[du:, dv:] = m[: n - du, : n - dv]
    return out


def grid(lam=None, tie="mix", **kw):
    rows = []
    for strategy in STRATEGIES:
        for risk in RISKS:
            sol = solve(strategy, risk, lam=lam, tie=tie, **kw)
            fwd = forward(sol)
            q_s, q_u, p_first = sol["q_root"]
            rows.append({
                "strategy": strategy, "risk": risk,
                "lambda": "BR" if lam is None else lam,
                "tie_rule": tie if lam is None else "none",
                "value": sol["value"], "q_first_safe": q_s, "q_first_unsafe": q_u,
                "first_move": ("UNSAFE" if abs(q_u - q_s) > 1e-7 and q_u > q_s else
                               "SAFE" if abs(q_u - q_s) > 1e-7 else "INDIFFERENT")
                if lam is None else f"P(U)={p_first:.3f}",
                "p_first_unsafe": p_first, **{k: v for k, v in fwd.items()
                                              if k != "unsafe_prob_by_round"},
            })
    return rows


def contrasts(rows, key="pooled_unsafe_rate"):
    def rate(s, r):
        return next(x[key] for x in rows if x["strategy"] == s and x["risk"] == r)
    au_as = {r: rate("AU", r) - rate("AS", r) for r in RISKS}
    return {"au_minus_as": au_as,
            "interaction_0p9_minus_0p1": au_as[0.9] - au_as[0.1]}


def simulate_engine(sol, n_races, seed0=910_000):
    """Play the solved policy through the real engine and score the races."""
    from ai_race.engine.agent import RaceAgent
    from ai_race.engine.game import AIRaceGame
    from ai_race.engine.state import Action, GameConfig
    from ai_race.engine.strategies import strategy_action

    cfg = GameConfig(name="br", max_private_risk=sol["risk"],
                     min_rounds=sol["min_rounds"], stop_probability=sol["stop_prob"],
                     max_rounds_safety_cap=sol["cap"], race_prize=sol["prize"],
                     run_phase="pilot")
    agents = [RaceAgent("Company_1"), RaceAgent("Company_2")]
    rng = np.random.default_rng(seed0 + int(sol["risk"] * 100) + 7 * STRATEGIES.index(sol["strategy"]))
    payoffs, unsafe, rounds = [], [], []
    for i in range(n_races):
        seed = int(rng.integers(1, 2_000_000_000))
        game = AIRaceGame(cfg, agents, template="", game_id=f"br{i}", seed=seed, rep=i)
        own, u, v, last = [], 0, 0, 0
        while not game.is_finished:
            r = len(game.history)
            p = sol["policy"][r][last][u, v]
            a = 1 if (p >= 1.0 or (0 < p < 1 and rng.random() < p)) else 0
            rival = strategy_action(sol["strategy"], r + 1, own)
            resp = ["ACTION: UNSAFE" if a else "ACTION: SAFE",
                    "ACTION: UNSAFE" if rival is Action.UNSAFE else "ACTION: SAFE"]
            game.apply_round_responses(resp, prompts=["", ""])
            own.append(Action.UNSAFE if a else Action.SAFE)
            u += a
            v += int(rival is Action.UNSAFE)
            last = a
        res = game.result
        payoffs.append(res.final_payoffs[0])
        unsafe.append(res.unsafe_counts[0])
        rounds.append(res.n_rounds)
    payoffs, unsafe, rounds = map(np.asarray, (payoffs, unsafe, rounds))
    n = len(payoffs)
    ratio = unsafe.sum() / rounds.sum()
    lin = (unsafe - ratio * rounds) / rounds.mean()
    return {
        "n_races": n_races,
        "mc_payoff": float(payoffs.mean()),
        "mc_payoff_se": float(payoffs.std(ddof=1) / np.sqrt(n)),
        "mc_pooled_unsafe_rate": float(ratio),
        "mc_pooled_unsafe_rate_se": float(lin.std(ddof=1) / np.sqrt(n)),
        "mc_mean_rounds": float(rounds.mean()),
    }


def observed_endpoints():
    """Pooled route-decision Unsafe rate per endpoint, rival, risk, plus the paired interaction."""
    from scripts import analyze_scripted_opponent as aso
    out = {}
    for route in aso.routes_present():
        turns, _, problems = aso.load_cells(route)
        if problems or turns.empty:
            raise SystemExit(f"{route}: refused cells {problems}")
        dec = turns[turns["is_route_decision"]]
        rates = {}
        for (s, rk), blk in dec.groupby(["strategy", "cell_risk"]):
            rates[(s, float(rk))] = float(blk["unsafe"].sum() / len(blk))
        blocks = aso.repetition_blocks(turns)
        vals = []
        for rep in sorted(set.intersection(*(set(blocks[k]) for k in
                                             (("AU", .1), ("AS", .1), ("AU", .9), ("AS", .9))))):
            g = lambda s, r: blocks[(s, r)][rep]["rate"]
            vals.append((g("AU", .9) - g("AS", .9)) - (g("AU", .1) - g("AS", .1)))
        out[route] = {"rates": rates, "interaction_block_mean": float(np.mean(vals))}
    return out


def make_figure(br, qr_by_lam, obs, path_stem):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    INK, MUTED, GRIDC = "#17212B", "#4B5865", "#D9DEE8"
    cols = {"google/gemini-3-flash-preview": "#2A6F97", "openai/gpt-5.4-2026-03-05": "#8D6A9F",
            "openai/gpt-5.5-2026-04-23": "#5B8E4A", "anthropic/claude-sonnet-5@default": "#C9783C",
            "anthropic/claude-opus-5@default": "#A33B3B"}
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                         "text.color": INK, "pdf.fonttype": 42, "ps.fonttype": 42})
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5), sharey=True)
    xs = np.arange(len(STRATEGIES))
    for ax, risk in zip(axes, RISKS):
        lo = [next(x["pooled_unsafe_rate"] for x in br["min"] if x["strategy"] == s and x["risk"] == risk) for s in STRATEGIES]
        hi = [next(x["pooled_unsafe_rate"] for x in br["max"] if x["strategy"] == s and x["risk"] == risk) for s in STRATEGIES]
        ax.bar(xs, np.maximum(np.array(hi) - np.array(lo), 0.012), bottom=lo, width=0.7, color=GRIDC,
               edgecolor=INK, lw=0.8, label="best-response range", zorder=1)
        mid = qr_by_lam.get(0.1)
        if mid:
            q = [next(x["pooled_unsafe_rate"] for x in mid if x["strategy"] == s and x["risk"] == risk) for s in STRATEGIES]
            ax.plot(xs, q, ls="--", color=MUTED, lw=1, marker="_", label=r"quantal, $\lambda$=0.1")
        for k, (route, d) in enumerate(obs.items()):
            y = [d["rates"][(s, risk)] for s in STRATEGIES]
            ax.scatter(xs + (k - 2) * 0.07, y, s=12, color=cols.get(route, MUTED),
                       label=route.split("/")[-1].split("@")[0] if risk == RISKS[0] else None, zorder=3)
        ax.set_xticks(xs, STRATEGIES)
        ax.set_title(f"stated risk {risk}", fontsize=8)
        ax.grid(axis="y", color=GRIDC, lw=.6)
        ax.set_ylim(-.03, 1.03)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel("Unsafe rate")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=4, frameon=False, fontsize=6.5)
    fig.tight_layout(rect=(0, 0.14, 1, 1))
    fig.savefig(f"{path_stem}.pdf")
    fig.savefig(f"{path_stem}.png", dpi=200)
    plt.close(fig)


def bracket(rows_min, rows_max, key="pooled_unsafe_rate"):
    """Exact range of the AU-AS contrast and interaction over optimal policies.

    Ties between equally good moves leave the Unsafe rate undetermined, so the
    prediction is a bracket: the lexicographic DP gives the least and the most
    Unsafe play any payoff-optimal policy can show in each cell, and the
    contrast ranges follow from interval arithmetic over independent cells.
    """
    def get(rows, s, r):
        return next(x[key] for x in rows if x["strategy"] == s and x["risk"] == r)
    lo = {r: get(rows_min, "AU", r) - get(rows_max, "AS", r) for r in RISKS}
    hi = {r: get(rows_max, "AU", r) - get(rows_min, "AS", r) for r in RISKS}
    return {"au_minus_as_low": lo, "au_minus_as_high": hi,
            "interaction_low": lo[0.9] - hi[0.1], "interaction_high": hi[0.9] - lo[0.1]}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mc-races", type=int, default=6000)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    br = {t: grid(None, tie=t) for t in ("min", "max", "mix")}
    br_rows = br["min"] + br["max"] + br["mix"]
    br_c = {t: contrasts(br[t]) for t in br}
    br_bracket = bracket(br["min"], br["max"])
    qr_rows, qr_summary, qr_by_lam = [], {}, {}
    for lam in LAMBDAS:
        rows = grid(lam)
        qr_rows += rows
        qr_by_lam[lam] = rows
        qr_summary[str(lam)] = contrasts(rows)

    mc = []
    for tie in ("min", "max"):
        for strategy in STRATEGIES:
            for risk in RISKS:
                sol = solve(strategy, risk, tie=tie)
                sim = simulate_engine(sol, args.mc_races)
                fwd = next(x for x in br[tie] if x["strategy"] == strategy and x["risk"] == risk)
                sim.update({"tie_rule": tie, "strategy": strategy, "risk": risk, "dp_value": fwd["value"],
                            "dp_pooled_unsafe_rate": fwd["pooled_unsafe_rate"]})
                z1 = (sim["mc_payoff"] - sim["dp_value"]) / sim["mc_payoff_se"]
                z2 = (sim["mc_pooled_unsafe_rate"] - sim["dp_pooled_unsafe_rate"]) / max(sim["mc_pooled_unsafe_rate_se"], 1e-9)
                sim["z_payoff"], sim["z_unsafe_rate"] = float(z1), float(z2)
                sim["abs_rate_error"] = abs(sim["mc_pooled_unsafe_rate"] - sim["dp_pooled_unsafe_rate"])
                mc.append(sim)
                print(tie, strategy, risk, f"DP {fwd['value']:.3f} MC {sim['mc_payoff']:.3f} (z {z1:+.2f}); "
                      f"rate DP {fwd['pooled_unsafe_rate']:.4f} MC {sim['mc_pooled_unsafe_rate']:.4f} (z {z2:+.2f})")

    obs = observed_endpoints()
    comp = []
    for route, d in obs.items():
        obs_au_as = {r: d["rates"][("AU", r)] - d["rates"][("AS", r)] for r in RISKS}
        obs_inter = obs_au_as[0.9] - obs_au_as[0.1]
        lo, hi = br_bracket["interaction_low"], br_bracket["interaction_high"]
        comp.append({
            "endpoint": route,
            "observed_interaction_pooled": obs_inter,
            "observed_interaction_block_mean": d["interaction_block_mean"],
            "best_response_interaction_low": lo,
            "best_response_interaction_high": hi,
            "best_response_interaction_mix": br_c["mix"]["interaction_0p9_minus_0p1"],
            "best_response_interaction_race_mean_rate": contrasts(br["mix"], "race_mean_unsafe_rate")["interaction_0p9_minus_0p1"],
            "best_response_cell_rates": {f"{x['strategy']}@{x['risk']}": x["pooled_unsafe_rate"] for x in br["mix"]},
            "best_response_au_minus_as": {str(k): v for k, v in br_c["mix"]["au_minus_as"].items()},
            "best_response_sign_determined": bool(lo > 0 or hi < 0),
            "sign_observed": int(np.sign(round(d["interaction_block_mean"], 6))),
            "sign_best_response_mix": int(np.sign(round(br_c["mix"]["interaction_0p9_minus_0p1"], 9))),
            "sign_matches_mix": bool(np.sign(d["interaction_block_mean"]) == np.sign(br_c["mix"]["interaction_0p9_minus_0p1"])),
            "observed_inside_bracket": bool(lo <= d["interaction_block_mean"] <= hi),
            "observed_au_minus_as": {str(k): v for k, v in obs_au_as.items()},
            "cell_rates_observed": {f"{s}@{r}": d["rates"][(s, r)] for s in STRATEGIES for r in RISKS},
            "cell_inside_br_range": {
                f"{s}@{r}": bool(next(x["pooled_unsafe_rate"] for x in br["min"] if x["strategy"] == s and x["risk"] == r) - 1e-9
                                 <= d["rates"][(s, r)] <=
                                 next(x["pooled_unsafe_rate"] for x in br["max"] if x["strategy"] == s and x["risk"] == r) + 1e-9)
                for s in STRATEGIES for r in RISKS},
        })

    max_z = max(max(abs(m["z_payoff"]), abs(m["z_unsafe_rate"])) for m in mc)
    max_err = max(m["abs_rate_error"] for m in mc)
    payload = {
        "protocol": "best-response baseline for ai-race-scripted-opponent-v1",
        "assumptions": {"objective": "risk-neutral expected total payoff", "horizon": "min 5 then stop 0.2, engine cap 100",
                        "forward_residual_mass": max(x["residual_mass"] for x in br_rows),
                        "tie_rule": "exact ties are common; min/max are the least/most Unsafe play of any optimal policy (lexicographic DP), mix splits ties 50/50"},
        "best_response": br_rows,
        "best_response_contrasts_pooled": br_c,
        "best_response_bracket": br_bracket,
        "quantal_response_contrasts": qr_summary,
        "engine_monte_carlo": {"rows": mc, "n_races_per_cell": args.mc_races,
                               "tolerance": "every |z| < 4 for payoff and Unsafe rate",
                               "max_abs_z": max_z, "max_abs_rate_error": max_err,
                               "passed": bool(max_z < 4.0)},
        "endpoints_vs_best_response": comp,
    }
    (OUT / "best_response.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    import csv
    def dump(path, rows):
        keys = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for row in rows:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})
    dump(OUT / "best_response.csv", br_rows)
    dump(OUT / "quantal_response.csv", qr_rows)
    dump(OUT / "endpoint_vs_best_response.csv", comp)
    dump(OUT / "engine_monte_carlo.csv", mc)
    make_figure(br, qr_by_lam, obs, str(OUT / "best_response"))

    print("\nBEST RESPONSE (pooled Unsafe rate: least / most over optimal policies)")
    for lo_row, hi_row in zip(br["min"], br["max"]):
        print(f"  {lo_row['strategy']:>3} @ {lo_row['risk']}: first={lo_row['first_move']:<12} "
              f"rate {lo_row['pooled_unsafe_rate']:.4f} to {hi_row['pooled_unsafe_rate']:.4f}  "
              f"Q(S)={lo_row['q_first_safe']:.2f} Q(U)={lo_row['q_first_unsafe']:.2f} V={lo_row['value']:.2f}")
    print("bracket", json.dumps(br_bracket))
    print("mix interaction", br_c["mix"]["interaction_0p9_minus_0p1"])
    for lam, c in qr_summary.items():
        print(f"  lambda {lam}: interaction {c['interaction_0p9_minus_0p1']:+.4f}  AU-AS {c['au_minus_as']}")
    for c in comp:
        print(c["endpoint"], f"obs {c['observed_interaction_block_mean']:+.4f}", "sign determined" if c["best_response_sign_determined"] else "BR sign undetermined", "inside" if c["observed_inside_bracket"] else "outside")
    print(f"MC max |z| {max_z:.2f}, max abs rate error {max_err:.4f}, passed={payload['engine_monte_carlo']['passed']}")


if __name__ == "__main__":
    main()
