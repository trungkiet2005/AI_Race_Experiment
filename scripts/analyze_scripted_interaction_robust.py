#!/usr/bin/env python3
"""Robustness of the rival-by-risk interaction in the scripted-opponent campaign.

Table 2 of the behaviour paper reports, per endpoint, the interaction
[(AU - AS) at risk 0.9] - [(AU - AS) at risk 0.1] on the rate scale with a
percentile bootstrap over ten matched repetitions. Three weaknesses are
addressed here, without changing any reported point estimate:

1. the rate scale is bounded, and several cells sit at the ceiling, so the
   interaction is also estimated on the log-odds scale with a Firth-penalised
   logit (finite under complete separation) and a repetition-clustered
   sandwich variance (CR1, t with G - 1 degrees of freedom);
2. ten blocks make a percentile interval unreliable, so a BCa bootstrap and an
   exact sign-flip randomisation interval over the matched blocks are added;
3. no multiplicity correction was applied, so Holm adjustment is applied across
   the five interaction tests and across the fifteen AU - AS contrasts.

Before anything is added the script recomputes the Table 2 point estimates and
percentile intervals from the raw turn logs and stops if they do not match the
numbers printed in the manuscript.

Outputs: results/derived/scripted_interaction_robust/*.json and *.csv.
"""

from __future__ import annotations

import csv
import itertools
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import analyze_scripted_opponent as aso  # noqa: E402

OUT = ROOT / "results" / "derived" / "scripted_interaction_robust"
LOW, HIGH = 0.1, 0.9
RISKS = aso.RISKS
SEED = 20261122
N_BOOT = 200_000
CEILING, FLOOR = 0.95, 0.05

TABLE2 = {  # percentage points, as printed in paper/ijcai/main.tex, tab:rival-risk-interaction
    "google/gemini-3-flash-preview": (-1.8, -7.9, 3.7),
    "openai/gpt-5.4-2026-03-05": (2.3, -4.6, 9.4),
    "openai/gpt-5.5-2026-04-23": (36.4, 29.0, 44.0),
    "anthropic/claude-sonnet-5@default": (-18.9, -25.1, -11.8),
    "anthropic/claude-opus-5@default": (29.2, 20.9, 36.4),
}


def firth_logit(X, y, max_iter=200, tol=1e-10):
    """Jeffreys-penalised logistic regression, finite under separation."""
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        eta = X @ beta
        p = 1.0 / (1.0 + np.exp(-eta))
        w = p * (1 - p)
        XtWX = X.T @ (X * w[:, None])
        inv = np.linalg.inv(XtWX)
        h = np.einsum("ij,jk,ik->i", X, inv, X) * w
        score = X.T @ (y - p + h * (0.5 - p))
        step = inv @ score
        step = np.clip(step, -5, 5)
        beta = beta + step
        if np.max(np.abs(step)) < tol:
            break
    p = 1.0 / (1.0 + np.exp(-(X @ beta)))
    w = p * (1 - p)
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    return beta, p, bread


def cluster_logit(rows, interaction_risks):
    """Interaction coefficient of rival (AU=1) x risk, clustered on repetition.

    rows: list of (rival_is_au, risk, rep, unsafe, n_decisions) per race block.
    The regression is fitted on the decision-level counts expanded by block.
    """
    X, y, g = [], [], []
    for au, risk, rep, k, n in rows:
        if risk not in interaction_risks:
            continue
        if len(interaction_risks) == 2:
            z = 1.0 if risk == max(interaction_risks) else 0.0
        else:
            z = risk
        for yi in [1.0] * int(k) + [0.0] * int(n - k):
            X.append([1.0, au, z, au * z])
            y.append(yi)
            g.append(rep)
    X, y, g = np.array(X), np.array(y), np.array(g)
    beta, p, bread = firth_logit(X, y)
    clusters = np.unique(g)
    meat = np.zeros((4, 4))
    for c in clusters:
        m = g == c
        s = X[m].T @ (y[m] - p[m])
        meat += np.outer(s, s)
    G, N, K = len(clusters), len(y), X.shape[1]
    cr1 = G / (G - 1) * (N - 1) / (N - K)
    cov = cr1 * bread @ meat @ bread
    se = float(np.sqrt(cov[3, 3]))
    est = float(beta[3])
    df = G - 1
    t = est / se if se > 0 else float("inf")
    pval = float(2 * stats.t.sf(abs(t), df))
    crit = stats.t.ppf(0.975, df)
    return {"estimate": est, "se_cluster": se, "t": float(t), "df": df, "p_value": pval,
            "ci95_low": est - crit * se, "ci95_high": est + crit * se,
            "n_decisions": int(N), "n_clusters": int(G)}


def cluster_bootstrap_logit(rows, rng, B=1000):
    reps = sorted({r[2] for r in rows})
    by_rep = {rep: [r for r in rows if r[2] == rep] for rep in reps}
    est = []
    for _ in range(B):
        pick = rng.integers(0, len(reps), len(reps))
        resampled = []
        for new, idx in enumerate(pick):
            resampled += [(a, b, new, c, d) for (a, b, _, c, d) in by_rep[reps[idx]]]
        try:
            est.append(cluster_logit_point(resampled))
        except np.linalg.LinAlgError:
            continue
    est = np.array(est)
    return float(np.percentile(est, 2.5)), float(np.percentile(est, 97.5)), len(est)


def cluster_logit_point(rows):
    X, y = [], []
    for au, risk, rep, k, n in rows:
        if risk not in (LOW, HIGH):
            continue
        z = 1.0 if risk == HIGH else 0.0
        for yi in [1.0] * int(k) + [0.0] * int(n - k):
            X.append([1.0, au, z, au * z])
            y.append(yi)
    beta, _, _ = firth_logit(np.array(X), np.array(y))
    return float(beta[3])


def bca_interval(values, rng, B=N_BOOT, alpha=0.05):
    values = np.asarray(values, float)
    n = values.size
    theta = values.mean()
    draws = rng.integers(0, n, size=(B, n))
    boot = values[draws].mean(axis=1)
    if np.ptp(values) == 0:
        return float(theta), float(theta), None, None
    frac = (boot < theta).mean() + 0.5 * (boot == theta).mean()
    frac = min(max(frac, 1.0 / (B + 1)), 1 - 1.0 / (B + 1))
    z0 = stats.norm.ppf(frac)
    jack = np.array([np.delete(values, i).mean() for i in range(n)])
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = np.sum(d ** 3) / denom if denom > 0 else 0.0
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        zq = stats.norm.ppf(q)
        adj = stats.norm.cdf(z0 + (z0 + zq) / (1 - a * (z0 + zq)))
        out.append(float(np.percentile(boot, 100 * adj)))
    return out[0], out[1], float(z0), float(a)


def signflip_p(values, theta=0.0):
    """Exact two-sided sign-flip randomisation p for mean = theta over blocks."""
    x = np.asarray(values, float) - theta
    n = x.size
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    means = (signs * x).mean(axis=1)
    obs = abs(x.mean())
    return float((np.abs(means) >= obs - 1e-12).mean())


def signflip_interval(values, alpha=0.05):
    """Exact interval by inverting the sign-flip test over a fine grid of theta."""
    x = np.asarray(values, float)
    n = x.size
    signs = np.array(list(itertools.product((-1.0, 1.0), repeat=n)))
    lo, hi = x.min() - 0.02, x.max() + 0.02
    grid = np.linspace(lo, hi, 4001)
    keep = []
    for th in grid:
        y = x - th
        p = (np.abs((signs * y).mean(axis=1)) >= abs(y.mean()) - 1e-12).mean()
        if p > alpha:
            keep.append(th)
    if not keep:
        return float("nan"), float("nan")
    return float(min(keep)), float(max(keep))


def holm(pvals):
    order = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvals[idx])
        adj[idx] = min(1.0, running)
    return adj


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    endpoints = {}
    for route in aso.routes_present():
        turns, _, problems = aso.load_cells(route)
        if problems or turns.empty:
            raise SystemExit(f"{route}: refused cells {problems}")
        endpoints[route] = (turns, aso.repetition_blocks(turns))

    # --- reproduce Table 2 before adding anything -----------------------------------
    repro = {}
    for route, (turns, blocks) in endpoints.items():
        res = aso.paired_rival_risk_interaction(
            blocks, LOW, HIGH,
            aso.cell_rng(*aso.rng_key(route, "rival_risk_interaction", *aso.RISK_CONTRAST_PAIRS[0])))
        point, lo, hi = (round(100 * res[k], 1) for k in ("mean_difference", "ci95_low", "ci95_high"))
        target = TABLE2[route]
        ok = (point, lo, hi) == target
        repro[route] = {"recomputed_pp": [point, lo, hi], "manuscript_pp": list(target), "match": ok,
                       "percentile_ci_pp_full": [100 * res["ci95_low"], 100 * res["ci95_high"]]}
    (OUT / "table2_reproduction.json").write_text(json.dumps(repro, indent=2), encoding="utf-8")
    bad = {r: v for r, v in repro.items() if not v["match"]}
    if bad:
        print(json.dumps(bad, indent=2))
        raise SystemExit("Table 2 does not reproduce; stopping before any new analysis")
    print("Table 2 reproduces exactly (point estimates and percentile intervals) for all five endpoints")

    # --- robust analysis --------------------------------------------------------------
    inter_rows, contrast_rows, cell_rows = [], [], []
    for route, (turns, blocks) in endpoints.items():
        reps = sorted(set.intersection(*(set(blocks[(s, r)]) for s in ("AU", "AS") for r in (LOW, HIGH))))
        vals = np.array([(blocks[("AU", HIGH)][k]["rate"] - blocks[("AS", HIGH)][k]["rate"])
                         - (blocks[("AU", LOW)][k]["rate"] - blocks[("AS", LOW)][k]["rate"]) for k in reps])
        est = float(vals.mean())
        bca_lo, bca_hi, z0, acc = bca_interval(vals, rng)
        sf_lo, sf_hi = signflip_interval(vals)
        sf_p = signflip_p(vals)

        rows = []
        for (strategy, risk), blk in blocks.items():
            if strategy in ("AU", "AS") and risk in RISKS:
                for rep, b in blk.items():
                    rows.append((1.0 if strategy == "AU" else 0.0, risk, rep, b["unsafe"], b["decisions"]))
        lo2 = cluster_logit(rows, (LOW, HIGH))
        lo3 = cluster_logit(rows, RISKS)
        b_lo, b_hi, b_n = cluster_bootstrap_logit(rows, rng)

        cells_involved = [(s, r) for s in ("AU", "AS") for r in (LOW, HIGH)]
        flags = {}
        for strategy in aso.STRATEGIES:
            for risk in RISKS:
                blk = blocks[(strategy, risk)]
                unsafe = sum(b["unsafe"] for b in blk.values())
                n = sum(b["decisions"] for b in blk.values())
                rate = unsafe / n
                at1 = sum(b["rate"] >= 1.0 for b in blk.values())
                at0 = sum(b["rate"] <= 0.0 for b in blk.values())
                flag = ("ceiling" if rate >= CEILING else "floor" if rate <= FLOOR else "")
                flags[(strategy, risk)] = flag
                cell_rows.append({"endpoint": route, "rival": strategy, "risk": risk, "rate": rate,
                                  "decisions": int(n), "blocks_at_1": int(at1), "blocks_at_0": int(at0),
                                  "n_blocks": len(blk), "flag": flag or "interior",
                                  "all_blocks_saturated": bool(at1 == len(blk) or at0 == len(blk))})
        involved_flags = {f"{s}@{r}": flags[(s, r)] for s, r in cells_involved if flags[(s, r)]}

        inter_rows.append({
            "endpoint": route, "n_blocks": len(reps),
            "rate_interaction_pp": 100 * est,
            "percentile_ci_pp": repro[route]["percentile_ci_pp_full"],
            "bca_ci_pp": [100 * bca_lo, 100 * bca_hi], "bca_z0": z0, "bca_accel": acc,
            "signflip_ci_pp": [100 * sf_lo, 100 * sf_hi], "signflip_p": sf_p,
            "logodds_corner": lo2, "logodds_corner_cluster_bootstrap_ci": [b_lo, b_hi],
            "logodds_corner_bootstrap_draws_ok": b_n,
            "logodds_linear_risk": lo3,
            "interaction_cells_flagged": involved_flags,
        })

        for risk in RISKS:
            d = np.array([blocks[("AU", risk)][k]["rate"] - blocks[("AS", risk)][k]["rate"]
                          for k in sorted(set(blocks[("AU", risk)]) & set(blocks[("AS", risk)]))])
            lo_, hi_ = signflip_interval(d)
            contrast_rows.append({"endpoint": route, "risk": risk, "au_minus_as_pp": 100 * float(d.mean()),
                                  "signflip_ci_pp": [100 * lo_, 100 * hi_],
                                  "signflip_p": signflip_p(d), "min_attainable_p": 2.0 / 2 ** len(d),
                                  "flags": {f"{s}": flags[(s, risk)] for s in ("AU", "AS") if flags[(s, risk)]}})

    for fam, rows_, key in (("interaction_signflip", inter_rows, "signflip_p"),):
        adj = holm(np.array([r[key] for r in rows_]))
        for r, a in zip(rows_, adj):
            r["holm_p_signflip_5"] = float(a)
    adj = holm(np.array([r["logodds_corner"]["p_value"] for r in inter_rows]))
    for r, a in zip(inter_rows, adj):
        r["holm_p_logodds_corner_5"] = float(a)
    adj = holm(np.array([r["logodds_linear_risk"]["p_value"] for r in inter_rows]))
    for r, a in zip(inter_rows, adj):
        r["holm_p_logodds_linear_5"] = float(a)
    adj = holm(np.array([c["signflip_p"] for c in contrast_rows]))
    for c, a in zip(contrast_rows, adj):
        c["holm_p_signflip_15"] = float(a)

    sign_obs = {r["endpoint"]: int(np.sign(r["rate_interaction_pp"])) for r in inter_rows}
    summary = {
        "table2_reproduced": True,
        "interactions": inter_rows,
        "au_minus_as_contrasts": contrast_rows,
        "min_attainable_signflip_p": 2.0 / 2 ** 10,
        "notes": {
            "logit": "Firth-penalised logit on decision rows; CR1 sandwich clustered on repetition; t with G-1=9 df; corner = AU/AS x risk 0.1/0.9 (matches Table 2); linear = risk as numeric over 0.1/0.6/0.9",
            "signflip": "exact, 2^10 sign patterns over matched repetition blocks, two-sided; interval by test inversion on a 4001-point grid",
            "bca": f"{N_BOOT} resamples, jackknife acceleration",
            "ceiling_floor": f"flag when pooled rate >= {CEILING} or <= {FLOOR}",
        },
    }
    (OUT / "interaction_robust.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def flat(path, rows_):
        keys = list(rows_[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=keys)
            w.writeheader()
            for row in rows_:
                w.writerow({k: (json.dumps(v) if isinstance(v, (dict, list)) else v) for k, v in row.items()})
    flat(OUT / "interaction_robust.csv", inter_rows)
    flat(OUT / "au_minus_as_contrasts.csv", contrast_rows)
    flat(OUT / "cell_ceiling_floor_flags.csv", cell_rows)

    for r in inter_rows:
        lc, ll = r["logodds_corner"], r["logodds_linear_risk"]
        print(f"{r['endpoint']}\n  rate {r['rate_interaction_pp']:+.1f}pp  pct {r['percentile_ci_pp'][0]:+.1f},{r['percentile_ci_pp'][1]:+.1f}"
              f"  BCa {r['bca_ci_pp'][0]:+.1f},{r['bca_ci_pp'][1]:+.1f}  signflip {r['signflip_ci_pp'][0]:+.1f},{r['signflip_ci_pp'][1]:+.1f}"
              f"  p_sf {r['signflip_p']:.4f} holm5 {r['holm_p_signflip_5']:.4f}")
        print(f"  logit corner {lc['estimate']:+.2f} [{lc['ci95_low']:+.2f},{lc['ci95_high']:+.2f}] p {lc['p_value']:.4f} holm {r['holm_p_logodds_corner_5']:.4f}"
              f" | boot [{r['logodds_corner_cluster_bootstrap_ci'][0]:+.2f},{r['logodds_corner_cluster_bootstrap_ci'][1]:+.2f}]"
              f" | linear {ll['estimate']:+.2f} [{ll['ci95_low']:+.2f},{ll['ci95_high']:+.2f}] p {ll['p_value']:.4f} holm {r['holm_p_logodds_linear_5']:.4f}")
        print(f"  flagged cells: {r['interaction_cells_flagged']}")
    print("AU-AS contrasts")
    for c in contrast_rows:
        print(f"  {c['endpoint']:<36} {c['risk']}  {c['au_minus_as_pp']:+6.1f}pp [{c['signflip_ci_pp'][0]:+.1f},{c['signflip_ci_pp'][1]:+.1f}] p {c['signflip_p']:.4f} holm15 {c['holm_p_signflip_15']:.4f}")


if __name__ == "__main__":
    main()
