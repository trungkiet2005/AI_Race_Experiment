#!/usr/bin/env python3
"""Is the departure from best response just noise? A logit-QRE fit to the scripted-rival data.

A blind reviewer's objection to the Claude Sonnet 5 reversal (the Always-Unsafe
minus Always-Safe contrast is smaller at high stated risk than at low) is that a
noisy best responder produces a negative rival-by-risk interaction by itself.
This script answers it with the endpoints' own decisions.

Model. Against a fixed scripted rival the endpoint faces the single-agent
dynamic programme of analyze_best_response.py, state (rounds played, own Unsafe
count u, rival Unsafe count v, own last move). The logit quantal response
equilibrium (QRE) plays Unsafe with probability sigmoid(lambda (Q1 - Q0)), where
Q is the action value when the continuation is played under the same quantal
policy (a logit-QRE dynamic programme, solved by backward induction). lambda = 0
is a coin flip and lambda -> infinity is the exact best response.

1. lambda is fitted by maximum likelihood on the endpoint's actual decisions at
   their actual states, with a profile-likelihood interval (which treats the
   decisions as independent) and a repetition-block cluster bootstrap interval
   (resampling repetitions inside each collection block, which is the pairing
   unit of the campaign).
2. With lambda fixed, the rival-by-risk interaction
   [(AU-AS)_0.9 - (AU-AS)_0.1] is predicted by simulating the fitted policy on
   the matched seeds (each cell and repetition keeps its recorded horizon, which
   the engine reproduces from the game seed), many times, giving a parametric
   bootstrap band. The same estimator is applied to the observed data.
3. Goodness of fit: log-likelihood per decision of the fitted QRE against a
   risk-blind QRE (stated setback risk set to 0 in the solver), a rival-blind
   QRE (the endpoint believes the rival plays Unsafe with a fixed probability q,
   fitted over a grid, whatever it has done) and a coin flip.
4. Cell consistency: for each endpoint x rival x risk cell, is the observed
   Unsafe rate inside the 95 percent interval of the exact best response played
   on the cell's own seeds (range over the tie rules)? A second, wider interval
   redraws the horizons, as a sensitivity.

Outputs under results/derived/quantal_fit/: quantal_fit.json plus
lambda_fit.csv, interaction_prediction.csv, fit_comparison.csv,
cell_consistency.csv.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_best_response as br  # noqa: E402
from scripts import analyze_scripted_opponent as aso  # noqa: E402

OUT = ROOT / "results" / "derived" / "quantal_fit"
BLOCK_DIRS = {
    "A": ROOT / "results" / "frontier" / "scripted_opponent_campaign",
    "B": ROOT / "results" / "frontier" / "scripted_opponent_replication" / "block_B",
    "low": ROOT / "results" / "frontier" / "scripted_opponent_contract_swap" / "sonnet_low",
}
SONNET = "anthropic/claude-sonnet-5@default"
LAMS = np.concatenate([[0.0], np.logspace(-3, 1.5, 91)])
Q_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
SEED = 20261123
N_BOOT = 4000
N_SIM = 4000
N_CELL_SIM = 2000
CAP = 100
STRATS = ("AS", "AU", "CS", "CAS")
RISKS = (0.1, 0.6, 0.9)


def sigmoid(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))


def solve_qre(rival_p, risk, lam, min_rounds=5, stop_prob=0.2, cap=CAP, prize=100.0):
    """Logit-QRE dynamic programme. Returns the logit table Z[r, last, u, v].

    rival_p(r, last) is the probability that the rival plays Unsafe in round r + 1
    (0 or 1 for the scripted strategies). The continuation value uses the same
    quantal policy, so this is not the Q of the optimal policy.
    """
    n = cap + 3
    uu, vv = np.meshgrid(np.arange(n, dtype=float), np.arange(n, dtype=float), indexing="ij")
    Z = np.zeros((cap, 2, n, n))
    W_next = None
    for r in range(cap, -1, -1):
        if r == cap:
            term = br.terminal_value(r, uu, vv, risk, prize)
            W_next = np.stack([term, term])
            continue
        term = br.terminal_value(r, uu, vv, risk, prize) if r >= 1 else None
        Wr = np.zeros((2, n, n))
        for last in (0, 1):
            pb = float(rival_p(r, last))
            q = []
            for a in (0, 1):
                val = 0.0
                if pb < 1.0:
                    val = (1.0 - pb) * br._shift(W_next[a], a, 0)
                if pb > 0.0:
                    val = val + pb * br._shift(W_next[a], a, 1)
                q.append(val)
            z = np.clip(lam * (q[1] - q[0]), -700.0, 700.0)
            p = sigmoid(z)
            cont = (1.0 - p) * q[0] + p * q[1]
            Z[r, last] = z
            if r == 0:
                Wr[last] = cont
            elif r >= min_rounds:
                Wr[last] = stop_prob * term + (1.0 - stop_prob) * cont
            else:
                Wr[last] = cont
        W_next = Wr
    return Z


def scripted(strategy):
    return lambda r, last: float(br.rival_move(strategy, r, last))


def fixed_belief(q):
    return lambda r, last: float(q)


def br_table(strategy, risk, tie):
    sol = br.solve(strategy, risk, lam=None, tie=tie)
    return np.stack([sol["policy"][r] for r in range(CAP)])


def log_lik(z, y):
    return -np.logaddexp(0.0, np.where(y == 1, -z, z))


def load_data():
    frames, problems_all = [], {}
    for block, path in BLOCK_DIRS.items():
        aso.CAMPAIGN = path
        for route in aso.routes_present():
            turns, _, problems = aso.load_cells(route)
            if problems or turns.empty:
                problems_all[f"{block}/{route}"] = problems
                continue
            dec = turns[turns["is_route_decision"]].sort_values(
                ["strategy", "cell_risk", "game_id", "round"]).copy()
            dec["opp_unsafe"] = (dec["opponent_action"].str.lower() == "unsafe").astype(int)
            g = dec.groupby("game_id")
            dec["u"] = g["unsafe"].cumsum() - dec["unsafe"]
            dec["v"] = g["opp_unsafe"].cumsum() - dec["opp_unsafe"]
            dec["last"] = g["unsafe"].shift(1).fillna(0).astype(int)
            dec["r"] = dec["round"] - 1
            if not np.allclose(dec["own_progress_before"], dec["r"] + 0.5 * dec["u"]):
                raise SystemExit(f"{block}/{route}: own progress disagrees with reconstructed u")
            if not np.allclose(dec["opponent_progress_before"], dec["r"] + 0.5 * dec["v"]):
                raise SystemExit(f"{block}/{route}: rival progress disagrees with reconstructed v")
            endpoint = f"{route} [low]" if block == "low" else route
            dec["endpoint"] = endpoint
            dec["block"] = block
            frames.append(dec[["endpoint", "block", "strategy", "cell_risk", "rep", "game_id",
                               "game_seed", "sampled_total_rounds", "round", "r", "u", "v",
                               "last", "unsafe"]].rename(columns={"cell_risk": "risk", "unsafe": "y"}))
    D = pd.concat(frames, ignore_index=True)
    D["cl"] = pd.factorize(pd.Series(list(zip(D["endpoint"], D["block"], D["rep"]))))[0]
    return D, problems_all


def accumulate(D, log):
    ncl = int(D["cl"].max()) + 1
    nl = len(LAMS)
    r, u, v, last, y, cl = (D[c].to_numpy() for c in ("r", "u", "v", "last", "y", "cl"))
    strat, risk = D["strategy"].to_numpy(), D["risk"].to_numpy()
    LL = {"full": np.zeros((ncl, nl)), "risk_blind": np.zeros((ncl, nl)),
          "rival_blind": np.zeros((len(Q_GRID), ncl, nl))}
    for s in STRATS:
        for k in RISKS:
            m = (strat == s) & (risk == k)
            if not m.any():
                continue
            for j, lam in enumerate(LAMS):
                Z = solve_qre(scripted(s), k, lam)
                ll = log_lik(Z[r[m], last[m], u[m], v[m]], y[m])
                LL["full"][:, j] += np.bincount(cl[m], weights=ll, minlength=ncl)
        log(f"full {s}")
    for s in STRATS:
        m = strat == s
        if not m.any():
            continue
        for j, lam in enumerate(LAMS):
            Z = solve_qre(scripted(s), 0.0, lam)
            ll = log_lik(Z[r[m], last[m], u[m], v[m]], y[m])
            LL["risk_blind"][:, j] += np.bincount(cl[m], weights=ll, minlength=ncl)
        log(f"risk-blind {s}")
    for iq, q in enumerate(Q_GRID):
        for k in RISKS:
            m = risk == k
            for j, lam in enumerate(LAMS):
                Z = solve_qre(fixed_belief(q), k, lam)
                ll = log_lik(Z[r[m], last[m], u[m], v[m]], y[m])
                LL["rival_blind"][iq][:, j] += np.bincount(cl[m], weights=ll, minlength=ncl)
        log(f"rival-blind q={q}")
    nd = np.bincount(cl, minlength=ncl)
    return LL, nd


def lam_interval(ll_curve, drop=1.92):
    best = ll_curve.max()
    ok = np.where(ll_curve >= best - drop)[0]
    return float(LAMS[ok.min()]), float(LAMS[ok.max()])


def boot_lam(curve, strata, rng, n_boot=N_BOOT):
    out = np.empty(n_boot)
    for b in range(n_boot):
        idx = np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
        out[b] = LAMS[np.argmax(curve[idx].sum(axis=0))]
    return out


def observed_interaction(D, endpoint, blocks):
    vals, per_block = [], {}
    for b in blocks:
        sub = D[(D.endpoint == endpoint) & (D.block == b)]
        if sub.empty:
            continue
        agg = sub.groupby(["strategy", "risk", "rep"]).agg(u=("y", "sum"), n=("y", "size"))
        agg["rate"] = agg["u"] / agg["n"]
        keys = (("AU", 0.9), ("AS", 0.9), ("AU", 0.1), ("AS", 0.1))
        if any(k not in agg.index.droplevel("rep") for k in keys):
            continue
        reps = sorted(set.intersection(*(set(agg.loc[k].index) for k in keys)))
        bv = [(agg.loc[("AU", 0.9, rp), "rate"] - agg.loc[("AS", 0.9, rp), "rate"])
              - (agg.loc[("AU", 0.1, rp), "rate"] - agg.loc[("AS", 0.1, rp), "rate"]) for rp in reps]
        per_block[b] = float(np.mean(bv))
        vals += bv
    if not vals:
        return None, per_block
    return float(np.mean(vals)), per_block


def sim_counts(P, strategy, H, rng):
    """Unsafe counts per race when the probability table P is played against the rival."""
    H = np.asarray(H)
    u = np.zeros(H.shape, dtype=int)
    v = np.zeros(H.shape, dtype=int)
    last = np.zeros(H.shape, dtype=int)
    tot = np.zeros(H.shape, dtype=int)
    for t in range(int(H.max())):
        active = t < H
        p = P[t, last, u, v]
        a = ((rng.random(H.shape) < p) & active).astype(int)
        if strategy == "AS":
            b = np.zeros_like(a)
        elif strategy == "AU":
            b = np.ones_like(a)
        elif strategy == "CS":
            b = last if t > 0 else np.zeros_like(a)
        else:
            b = last if t > 0 else np.ones_like(a)
        v = v + b * active
        u = u + a
        tot += a
        last = np.where(active, a, last)
    return tot


def cell_horizons(D, endpoint, block):
    sub = D[(D.endpoint == endpoint) & (D.block == block)]
    out = {}
    for (s, k), g in sub.groupby(["strategy", "risk"]):
        per = g.groupby("rep").agg(n=("y", "size"), u=("y", "sum"), seed=("game_seed", "first"))
        out[(s, k)] = per
    return out


def predict_interaction(D, endpoint, blocks, lam, rng, cache):
    """Parametric-bootstrap distribution of the pooled interaction under QRE(lam)."""
    per_rep = []
    for b in blocks:
        cells = cell_horizons(D, endpoint, b)
        keys = (("AU", 0.9), ("AS", 0.9), ("AU", 0.1), ("AS", 0.1))
        if any(k not in cells for k in keys):
            continue
        reps = sorted(set.intersection(*(set(cells[k].index) for k in keys)))
        rates = {}
        for (s, k) in keys:
            ck = (s, k, float(lam))
            if ck not in cache:
                cache[ck] = sigmoid(solve_qre(scripted(s), k, lam))
            H = cells[(s, k)].loc[reps, "n"].to_numpy()
            H = np.broadcast_to(H, (N_SIM, len(reps)))
            rates[(s, k)] = sim_counts(cache[ck], s, H, rng) / H
        per_rep.append((rates[("AU", 0.9)] - rates[("AS", 0.9)])
                       - (rates[("AU", 0.1)] - rates[("AS", 0.1)]))
    return np.concatenate(per_rep, axis=1).mean(axis=1)


def engine_check(D, lam, rng):
    """Play the fitted policy through AIRaceGame and compare with the table simulation."""
    from ai_race.engine.agent import RaceAgent
    from ai_race.engine.game import AIRaceGame
    from ai_race.engine.state import Action, GameConfig
    from ai_race.engine.strategies import strategy_action

    def engine_run(P, strategy, risk, seed):
        cfg = GameConfig(name="qfit", max_private_risk=risk, min_rounds=5, stop_probability=0.2,
                         max_rounds_safety_cap=CAP, race_prize=100.0, run_phase="pilot")
        game = AIRaceGame(cfg, [RaceAgent("Company_1"), RaceAgent("Company_2")], template="",
                          game_id="q", seed=int(seed), rep=0)
        own, u, v, last = [], 0, 0, 0
        while not game.is_finished:
            r = len(game.history)
            p = P[r, last, u, v]
            a = 1 if rng.random() < p else 0
            rival = strategy_action(strategy, r + 1, own)
            game.apply_round_responses(["ACTION: UNSAFE" if a else "ACTION: SAFE",
                                        "ACTION: UNSAFE" if rival is Action.UNSAFE else "ACTION: SAFE"],
                                       prompts=["", ""])
            own.append(Action.UNSAFE if a else Action.SAFE)
            u += a
            v += int(rival is Action.UNSAFE)
            last = a
        return game.result.unsafe_counts[0], game.result.n_rounds

    seeds = D.groupby(["block", "rep"]).agg(seed=("game_seed", "first"), H=("sampled_total_rounds", "first")).reset_index()
    horizon_ok = True
    for _, row in seeds.iterrows():
        _, n_rounds = engine_run(np.zeros((CAP, 2, CAP + 3, CAP + 3)), "AS", 0.1, row["seed"])
        horizon_ok &= bool(n_rounds == row["H"])
    P = sigmoid(solve_qre(scripted("AU"), 0.9, lam))
    cells = cell_horizons(D, SONNET, "A")[("AU", 0.9)]
    reps_ = 200
    eng = []
    for _ in range(reps_):
        c = [engine_run(P, "AU", 0.9, s) for s in cells["seed"]]
        eng.append(sum(x[0] for x in c) / sum(x[1] for x in c))
    H = np.broadcast_to(cells["n"].to_numpy(), (4000, len(cells)))
    vec = sim_counts(P, "AU", H, rng).sum(axis=1) / H.sum(axis=1)
    se = np.sqrt(np.var(eng, ddof=1) / reps_ + np.var(vec, ddof=1) / len(vec))
    return {"horizon_matches_engine_for_all_seeds": bool(horizon_ok),
            "n_seeds_checked": int(len(seeds)),
            "engine_mean_rate": float(np.mean(eng)), "table_mean_rate": float(vec.mean()),
            "z": float((np.mean(eng) - vec.mean()) / se), "lambda": float(lam)}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()
    log = (lambda *a: None) if args.quiet else (lambda *a: print(*a, flush=True))
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    D, problems = load_data()
    if problems:
        log("refused cells:", problems)
    LL, nd = accumulate(D, log)
    cl_info = D.groupby("cl").agg(endpoint=("endpoint", "first"), block=("block", "first"),
                                  rep=("rep", "first")).reset_index()

    endpoints = [e for e in D["endpoint"].unique()]
    scopes = []
    for e in endpoints:
        blocks_here = sorted(D[D.endpoint == e]["block"].unique())
        if blocks_here == ["low"]:
            scopes.append((e, "low", ["low"]))
            continue
        for lab, bl in (("A", ["A"]), ("B", ["B"]), ("A+B", ["A", "B"])):
            if all(b in blocks_here for b in bl):
                scopes.append((e, lab, bl))

    fit_rows, pred_rows, cmp_rows, payload = [], [], [], {"scopes": []}
    sim_cache = {}
    for e, lab, bl in scopes:
        ids = cl_info[(cl_info.endpoint == e) & (cl_info.block.isin(bl))]
        cid = ids["cl"].to_numpy()
        strata = [ids[ids.block == b]["cl"].to_numpy() for b in bl]
        n_dec = int(nd[cid].sum())
        curve = LL["full"][cid].sum(axis=0)
        j = int(np.argmax(curve))
        lam_hat = float(LAMS[j])
        pl_lo, pl_hi = lam_interval(curve)
        draws = boot_lam(LL["full"], strata, rng)
        b_lo, b_hi = (float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5)))
        boundary = "lower" if j == 0 else ("upper" if j == len(LAMS) - 1 else "interior")
        fit_rows.append({"endpoint": e, "scope": lab, "n_decisions": n_dec, "lambda_hat": lam_hat,
                         "profile_lo": pl_lo, "profile_hi": pl_hi, "boot_lo": b_lo, "boot_hi": b_hi,
                         "boundary": boundary, "loglik": float(curve[j]),
                         "loglik_per_decision": float(curve[j] / n_dec)})

        rb = LL["risk_blind"][cid].sum(axis=0)
        jr = int(np.argmax(rb))
        rv = LL["rival_blind"][:, cid, :].sum(axis=1)
        iq, jv = np.unravel_index(np.argmax(rv), rv.shape)
        coin = n_dec * np.log(0.5)
        boot_idx = [np.concatenate([rng.choice(s, size=len(s), replace=True) for s in strata])
                    for _ in range(2000)]
        def gap(a_curve_all, ja, b_curve_all, jb):
            vals = [(a_curve_all[ix, ja].sum() - b_curve_all[ix, jb].sum()) / nd[ix].sum() for ix in boot_idx]
            return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
        rv_all = LL["rival_blind"][iq]
        cmp_rows.append({
            "endpoint": e, "scope": lab, "n_decisions": n_dec,
            "ll_per_decision_full": float(curve[j] / n_dec), "lambda_full": lam_hat,
            "ll_per_decision_risk_blind": float(rb[jr] / n_dec), "lambda_risk_blind": float(LAMS[jr]),
            "ll_per_decision_rival_blind": float(rv[iq, jv] / n_dec),
            "lambda_rival_blind": float(LAMS[jv]), "q_rival_blind": float(Q_GRID[iq]),
            "ll_per_decision_coin": float(coin / n_dec),
            "gain_vs_risk_blind": float((curve[j] - rb[jr]) / n_dec),
            "gain_vs_risk_blind_ci": gap(LL["full"], j, LL["risk_blind"], jr),
            "gain_vs_rival_blind": float((curve[j] - rv[iq, jv]) / n_dec),
            "gain_vs_rival_blind_ci": gap(LL["full"], j, rv_all, jv),
        })

        obs, obs_blocks = observed_interaction(D, e, bl)
        if obs is None:
            log(f"{e} {lab}: no block holds all four interaction cells")
            continue
        row = {"endpoint": e, "scope": lab, "observed_interaction": obs,
               "observed_by_block": obs_blocks, "interaction_blocks_used": sorted(obs_blocks),
               "lambda_hat": lam_hat}
        for tag, lam in (("hat", lam_hat), ("lo", b_lo), ("hi", b_hi)):
            sims = predict_interaction(D, e, bl, lam, rng, sim_cache)
            lo, hi = float(np.percentile(sims, 2.5)), float(np.percentile(sims, 97.5))
            row[f"lambda_{tag}"] = float(lam)
            row[f"pred_mean_{tag}"] = float(sims.mean())
            row[f"band_lo_{tag}"] = lo
            row[f"band_hi_{tag}"] = hi
            row[f"observed_outside_band_{tag}"] = bool(obs < lo or obs > hi)
            row[f"p_two_sided_{tag}"] = float(min(1.0, 2 * min((sims <= obs).mean(), (sims >= obs).mean())))
        pred_rows.append(row)
        log(f"{e} {lab}: lambda {lam_hat:.4g} boot [{b_lo:.3g},{b_hi:.3g}] obs {obs:+.4f} "
            f"pred {row['pred_mean_hat']:+.4f} band [{row['band_lo_hat']:+.4f},{row['band_hi_hat']:+.4f}]")
        payload["scopes"].append({"endpoint": e, "scope": lab})

    sonnet_ab = next(r for r in fit_rows if r["endpoint"] == SONNET and r["scope"] == "A+B")
    engine = engine_check(D, sonnet_ab["lambda_hat"], rng)
    log("engine check", engine)

    cons_rows = consistency(D, rng)

    payload.update({
        "protocol": "logit-QRE fit to the scripted-opponent campaign",
        "lambda_grid": {"min_positive": float(LAMS[1]), "max": float(LAMS[-1]), "n": int(len(LAMS))},
        "rival_blind_q_grid": list(Q_GRID),
        "n_boot_lambda": N_BOOT, "n_sim_interaction": N_SIM, "n_cell_sim": N_CELL_SIM, "seed": SEED,
        "refused_cells": problems, "lambda_fit": fit_rows, "interaction_prediction": pred_rows,
        "fit_comparison": cmp_rows, "cell_consistency": cons_rows,
        "cell_consistency_counts": consistency_counts(cons_rows), "engine_validation": engine,
    })
    (OUT / "quantal_fit.json").write_text(json.dumps(payload, indent=2, default=float), encoding="utf-8")
    dump(OUT / "lambda_fit.csv", fit_rows)
    dump(OUT / "interaction_prediction.csv", pred_rows)
    dump(OUT / "fit_comparison.csv", cmp_rows)
    dump(OUT / "cell_consistency.csv", cons_rows)
    log(json.dumps(payload["cell_consistency_counts"], indent=1))


def consistency(D, rng):
    tables = {}
    for s in STRATS:
        for k in RISKS:
            tables[(s, k)] = {t: br_table(s, k, t) for t in ("min", "max", "mix")}
    dp_range = {}
    for t in ("min", "max"):
        for row in br.grid(None, tie=t):
            dp_range.setdefault((row["strategy"], row["risk"]), {})[t] = row["pooled_unsafe_rate"]
    rows = []
    for (e, b), _ in D.groupby(["endpoint", "block"]):
        cells = cell_horizons(D, e, b)
        for (s, k), per in sorted(cells.items()):
            Hs = per["n"].to_numpy()
            obs = float(per["u"].sum() / per["n"].sum())
            fresh_H = np.minimum(4 + rng.geometric(0.2, size=(N_CELL_SIM, len(Hs))), CAP)
            same_H = np.broadcast_to(Hs, (N_CELL_SIM, len(Hs)))
            res = {}
            for mode, Hm in (("same", same_H), ("fresh", fresh_H)):
                los, his = [], []
                for t, P in tables[(s, k)].items():
                    rate = sim_counts(P, s, Hm, rng).sum(axis=1) / Hm.sum(axis=1)
                    los.append(float(np.percentile(rate, 2.5)))
                    his.append(float(np.percentile(rate, 97.5)))
                res[mode] = (min(los), max(his))
            dlo, dhi = min(dp_range[(s, k)].values()), max(dp_range[(s, k)].values())
            rows.append({
                "endpoint": e, "block": b, "rival": s, "risk": k, "n_races": int(len(Hs)),
                "observed_rate": obs,
                "br_same_seeds_lo": res["same"][0], "br_same_seeds_hi": res["same"][1],
                "consistent_same_seeds": bool(res["same"][0] - 1e-9 <= obs <= res["same"][1] + 1e-9),
                "br_fresh_horizon_lo": res["fresh"][0], "br_fresh_horizon_hi": res["fresh"][1],
                "consistent_fresh_horizon": bool(res["fresh"][0] - 1e-9 <= obs <= res["fresh"][1] + 1e-9),
                "dp_range_lo": dlo, "dp_range_hi": dhi,
                "consistent_within_10_points_rule": bool(dlo - 0.10 <= obs <= dhi + 0.10),
            })
    return rows


def consistency_counts(rows):
    out = {}
    df = pd.DataFrame(rows)
    for (e, b), g in df.groupby(["endpoint", "block"]):
        out[f"{e}|{b}"] = {"cells": int(len(g)),
                           "same_seeds": int(g["consistent_same_seeds"].sum()),
                           "fresh_horizon": int(g["consistent_fresh_horizon"].sum()),
                           "within_10_points_rule": int(g["consistent_within_10_points_rule"].sum())}
    return out


def dump(path, rows):
    keys = list(rows[0].keys())
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list, tuple)) else v)
                        for k, v in row.items()})


if __name__ == "__main__":
    main()
