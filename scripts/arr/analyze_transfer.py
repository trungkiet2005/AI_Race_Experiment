#!/usr/bin/env python3
"""ARR transfer campaign: the preregistered primary analyses and the exploratory item analysis.

Authority: docs/arr-transfer-campaign-prereg-2026-09-19.md, sections 3, 4 and 6, its
amendments and its collection outcome.  Primary analyses (section 4):

  1. name baselines N1 and N2 (section 3) on F1;
  2. transfer of state_reconstruction and overall accuracy between families
     (Spearman, permutation p, bootstrap interval) and verdict agreement (count, kappa);
  3. cheap baselines per family;
  4. repetition reliability per family.

Everything else is labelled exploratory: the item analysis with arithmetic term
counts and the long / wide accuracy tables.

Fail-closed.  Cells are loaded through analyze_screen_baselines.load_campaign
(60 rows, one bank and one rules context per family, recomputed verdicts, no
duplicate route, failed_runs/ never opened).  On top of that this script refuses
when a family's probe_bank_sha256 is not the frozen hash recorded for it, when the
bank text in the task source does not hash to that value, when a route slug maps
to two cells in one family, when a loaded cell of the transfer campaign is not a
completed, admitted ledger row (or the reverse), or when a directory name does not
match the normalised route slug.
"""
from __future__ import annotations

import argparse
import ast
import csv
import hashlib
import itertools
import json
import math
import re
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

sys.path.insert(0, str(Path(__file__).resolve().parent))
import analyze_screen_baselines as A  # noqa: E402

ROOT = A.ROOT
V6_ROOT = ROOT / "results" / "frontier" / "admission_campaign_v6"
ARR_ROOT = ROOT / "results" / "frontier" / "arr_transfer_campaign"
LEDGER = ARR_ROOT / "ledger.csv"
OUT = ROOT / "results" / "derived" / "arr_transfer"
PREREG = "docs/arr-transfer-campaign-prereg-2026-09-19.md"

FAMILIES = ("F1", "F2", "F3")
PAIRS = (("F1", "F2"), ("F1", "F3"), ("F2", "F3"))
FROZEN_BANK = {
    "F1": "2953fb472dcba47511de108a47d4126b0f661e79bf60c96a3ab46f581abca8bc",
    "F2": "1cf954f988b379fbf97e72861eea6e2de16d1bf62e8771b4c6d4c2f3d7a2c1e0",
    "F3": "31ff7c47875bd700f126e42392d6769c1ab4a75d9d4c09fccb71c83033410b5b",
}
FAMILY_README = {"F2": ROOT / "arr/families/f2_dilemma/README.md",
                 "F3": ROOT / "arr/families/f3_commons/README.md"}
BANK_SOURCE = {"F1": ROOT / "kaggle/benchmarks/ai_race_frontier_admission.py",
               "F2": ROOT / "arr/families/f2_dilemma/bank.py",
               "F3": ROOT / "arr/families/f3_commons/bank.py"}
DOMAINS = A.DOMAIN_ORDER
SR, TS = "state_reconstruction", "terminal_scoring"
SEED = 20260919
N_PERM = 100_000
N_BOOT = 10_000
EXACT_MAX_N = 10
N1_WORDS = {"mini", "nano", "lite"}
N2_WORDS = {"mini", "nano", "lite", "haiku", "gemma"}

TERM_RULE = (
    "term_count = number of numeric operands in the minimal computation of the correct answer from "
    "the values stated in the probe and the rules context: a recalled rule or a single table lookup "
    "counts 1; counting occurrences in a k-element action list counts k (one indicator per element); "
    "a rate's denominator counts 1; a threshold that a computed quantity is compared with counts 1; "
    "every constant that enters the formula (prize, bonus, multiplier, maximum risk, expected length, "
    "stopping probability, minimum rounds) counts 1; a value used twice counts once; scaling by 100 "
    "for a percentage is not counted.")

TERMS = {
    ("F1", "rule_simultaneous"): (1, "rule recall"),
    ("F1", "rule_hidden_horizon"): (1, "rule recall"),
    ("F1", "rule_min_rounds"): (1, "rule recall"),
    ("F1", "rule_stop_probability"): (1, "rule recall"),
    ("F1", "payoff_safe_unsafe"): (1, "one payoff-matrix lookup (0.6)"),
    ("F1", "payoff_unsafe_safe"): (1, "one payoff-matrix lookup (2.4)"),
    ("F1", "state_own_progress"): (3, "1.5+1.5+1.0: three per-round progress increments"),
    ("F1", "state_opp_progress"): (3, "1.5+1.0+1.0: three per-round progress increments"),
    ("F1", "state_own_payoff"): (3, "2.0+2.4+1.0: three joint payoff lookups summed"),
    ("F1", "state_own_risk"): (5, "60 x (1+1+0)/3: max risk, three UNSAFE indicators, denominator"),
    ("F1", "state_gap"): (2, "4-3.5"),
    ("F1", "transition_progress"): (2, "4+1.5"),
    ("F1", "transition_risk"): (5, "60 x (2+1)/(3+1): max risk, prior count, new action, prior rounds, new round"),
    ("F1", "terminal_winner"): (2, "compare 6 with 5"),
    ("F1", "terminal_loser_setback"): (1, "rule recall (setback applies only to winners)"),
    ("F1", "terminal_winner_risk"): (3, "60 x 2/5"),
    ("F1", "terminal_winner_payoff"): (4, "draw 0.50 vs risk 24% (no setback), then 10+100"),
    ("F1", "terminal_setback_payoff"): (2, "draw 0.10 vs risk 24% -> setback -> 0; stage and prize not needed"),
    ("F1", "expected_safe_safe"): (3, "1.0 x 9 + 50 (tie prize, zero risk under SAFE)"),
    ("F1", "expected_unsafe_safe"): (4, "(2.4 x 9 + 100) x (1 - 0.10): stage payoff, expected length, prize, risk"),
    ("F2", "rule_simultaneous"): (1, "rule recall"),
    ("F2", "rule_hidden_horizon"): (1, "rule recall"),
    ("F2", "rule_min_rounds"): (1, "rule recall"),
    ("F2", "rule_bonus_threshold"): (1, "rule recall"),
    ("F2", "payoff_coop_defect"): (1, "one payoff-matrix lookup (0.4)"),
    ("F2", "payoff_defect_coop"): (1, "one payoff-matrix lookup (4.7)"),
    ("F2", "state_opp_defections"): (6, "count DEFECT in 6 opponent actions: 6 indicators"),
    ("F2", "state_opp_last_defect"): (1, "positional lookup of the last DEFECT; nothing is summed"),
    ("F2", "state_own_payoff"): (6, "sum of 6 joint payoff lookups (3.2+3.2+... = 13.2)"),
    ("F2", "state_opp_payoff"): (6, "sum of 6 joint payoff lookups (= 17.5)"),
    ("F2", "state_opp_coop_rate"): (6, "5 COOPERATE indicators / denominator 5"),
    ("F2", "transition_opp_payoff"): (2, "9.8 + 4.7"),
    ("F2", "transition_opp_coop_rate"): (4, "(3+1)/(4+1)"),
    ("F2", "terminal_bonus_boundary"): (3, "3/5 compared with 60%"),
    ("F2", "terminal_opp_bonus"): (9, "7 COOPERATE indicators / 7 compared with 60%"),
    ("F2", "terminal_own_final"): (13, "5 stage payoffs + (5 indicators / 5 vs 60%) + bonus 7.5"),
    ("F2", "terminal_opp_final"): (12, "5 stage payoffs + (5 indicators / 5 vs 60%, fails, no bonus)"),
    ("F2", "terminal_own_final_six"): (15, "6 stage payoffs + (6 indicators / 6 vs 60%) + bonus 7.5"),
    ("F2", "expected_coop_coop"): (4, "3.2 x 9 + 7.5, with the 60% threshold check"),
    ("F2", "expected_defect3_coop"): (8, "4.7 x 3 + 3.2 x (9-3) + 7.5 x P(N>=8): 4.7, 3 (used twice, counted once), 3.2, 9, 60%, 0.2, 5, 7.5"),
    ("F3", "rule_simultaneous"): (1, "rule recall"),
    ("F3", "rule_hidden_horizon"): (1, "rule recall"),
    ("F3", "rule_multiplier"): (1, "rule recall"),
    ("F3", "rule_group_target"): (1, "rule recall"),
    ("F3", "payoff_free_ride"): (6, "10 - 0 + 0.4 x (0+10+10+10): endowment, factor, 4 contributions"),
    ("F3", "payoff_sole_contributor"): (6, "10 - 10 + 0.4 x (10+0+0+0): endowment, factor, 4 contributions"),
    ("F3", "state_own_payoff"): (19, "4 x 10 - own total + 0.4 x group total over 16 contributions: 16 contributions, endowment, rounds, factor"),
    ("F3", "state_other_payoff"): (19, "same structure as state_own_payoff for P4: 16 contributions, endowment, rounds, factor"),
    ("F3", "state_group_average"): (17, "16 contributions summed / 4 rounds"),
    ("F3", "state_rounds_above"): (17, "4 round totals from 16 contributions, each compared with 12"),
    ("F3", "state_target_gap"): (17, "150 - sum of 16 contributions"),
    ("F3", "transition_target_gap"): (5, "80 - (10+10+5+0)"),
    ("F3", "transition_own_payoff"): (7, "48 + 10 - 10 + 0.4 x (10+10+5+0): prior, endowment, factor, 4 contributions"),
    ("F3", "terminal_target_missed"): (7, "25+30+20+30+20+20 = 145 compared with 150"),
    ("F3", "terminal_free_rider_bonus"): (1, "rule application: bonus does not depend on own contribution"),
    ("F3", "terminal_payoff_bonus"): (4, "160 >= 150, so 124 + 40"),
    ("F3", "terminal_payoff_no_bonus"): (3, "135 < 150, so 124"),
    ("F3", "terminal_payoff_from_rounds"): (8, "30+35+25+30+35 = 155 >= 150, so 72 + 40"),
    ("F3", "expected_all_ten"): (9, "(10 - 10 + 0.4 x 40) x 9 + 40 (target met for every length): endowment, factor, 4 contributions, 9, 150, 40"),
    ("F3", "expected_all_five"): (12, "(10 - 5 + 0.4 x 20) x 9 + 40 x P(N>=8): endowment, factor, 4 contributions, 9, 150, 0.2, 5, 40"),
}


def fail(msg: str) -> None:
    A.fail(msg)


def slug_of(model_route: str) -> str:
    return model_route.split("/", 1)[-1].replace("@", "-").lower()


def tokens(slug: str) -> list[str]:
    return [t for t in re.split(r"[-_./@]", slug.lower()) if t]


def n1_refused(slug: str) -> bool:
    return bool(N1_WORDS & set(tokens(slug)))


def n2_refused(slug: str) -> bool:
    t = set(tokens(slug))
    return bool(N2_WORDS & t) or {"oss", "20b"} <= t


# ---------------------------------------------------------------------------
# loading


def literal_assign(path: Path, name: str):
    tree = ast.parse(A.read_text(path))
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            return ast.literal_eval(node.value)
    fail(f"{name} not found in {A._rel(path)}")


def bank_of(family: str) -> dict:
    probes = literal_assign(BANK_SOURCE[family], "PROBES")
    rules = literal_assign(BANK_SOURCE[family], "RULES_CONTEXT")
    bank_hash = hashlib.sha256(json.dumps(probes, sort_keys=True).encode("utf-8")).hexdigest()
    rules_hash = hashlib.sha256(rules.encode("utf-8")).hexdigest()
    if bank_hash != FROZEN_BANK[family]:
        fail(f"{family}: bank text in {A._rel(BANK_SOURCE[family])} hashes to {bank_hash}, frozen {FROZEN_BANK[family]}")
    readme_hash = None
    if family in FAMILY_README:
        m = re.search(r"probe_bank_sha256`?\s*\|\s*`([0-9a-f]{64})`", A.read_text(FAMILY_README[family]))
        if not m:
            fail(f"{family}: no probe_bank_sha256 in {A._rel(FAMILY_README[family])}")
        readme_hash = m.group(1)
        if readme_hash != FROZEN_BANK[family]:
            fail(f"{family}: README hash {readme_hash} != frozen {FROZEN_BANK[family]}")
    return {"probes": {p[0]: {"domain": p[1], "question": p[2], "answer": p[3]} for p in probes},
            "order": [p[0] for p in probes], "probe_bank_sha256": bank_hash,
            "rules_context_sha256": rules_hash, "readme_hash": readme_hash,
            "source": A._rel(BANK_SOURCE[family])}


def read_ledger() -> list[dict]:
    return list(csv.DictReader(A.read_text(LEDGER).splitlines()))


def load_family(family: str, ledger: list[dict]) -> dict:
    roots = [V6_ROOT, ARR_ROOT / family] if family == "F1" else [ARR_ROOT / family]
    c = A.load_campaign(roots, A.LEGACY_FAMILY, 60)
    c.family = family
    bank = bank_of(family)
    if c.probe_bank_sha256 != FROZEN_BANK[family]:
        fail(f"{family}: cells carry probe_bank_sha256 {c.probe_bank_sha256}, frozen {FROZEN_BANK[family]}")
    if c.rules_context_sha256 != bank["rules_context_sha256"]:
        fail(f"{family}: rules_context_sha256 {c.rules_context_sha256} != source {bank['rules_context_sha256']}")
    if set(c.probes) != set(bank["probes"]) or any(c.probe_domain[p] != bank["probes"][p]["domain"] for p in c.probes):
        fail(f"{family}: probe ids or domains in the rows differ from the bank source")
    cells, mapping = {}, []
    for r in c.routes:
        mf = r.manifest.get("family")
        if mf not in (None, family):
            fail(f"{r.route}: manifest family {mf!r} loaded as {family}")
        if family != "F1" and mf != family:
            fail(f"{r.route}: {family} manifest lacks family={family}")
        slug = slug_of(r.route)
        parts = r.raw_path.relative_to(ROOT).parts
        run_id, model_dir, above = r.raw_path.parents[3].name, r.raw_path.parents[4].name, r.raw_path.parents[5].name
        if model_dir != slug:
            fail(f"{r.route}: directory {model_dir!r} does not match normalised slug {slug!r}")
        if slug in cells:
            fail(f"{family}: route {slug} is complete twice")
        source = "admission_campaign_v6" if "admission_campaign_v6" in parts else "arr_transfer_campaign"
        cells[slug] = r
        mapping.append({"family": family, "source": source,
                        "task_version_or_family_dir": above, "directory_name": model_dir,
                        "model_route": r.route, "slug": slug, "run_id": run_id,
                        "raw_path": A._rel(r.raw_path)})
    arr_loaded = {(m["slug"], m["run_id"]) for m in mapping if m["source"] == "arr_transfer_campaign"}
    arr_ledger = {(x["model"], x["run_id"]) for x in ledger
                  if x["family"] == family and x["state"] == "completed" and x["admitted_to_analysis"] == "True"}
    if arr_loaded != arr_ledger:
        fail(f"{family}: loaded cells {sorted(arr_loaded ^ arr_ledger)} disagree with the ledger")
    for x in ledger:
        if x["family"] == family and (x["model"], x["run_id"]) in arr_ledger and x["raw_rows"] != "60":
            fail(f"{family}: ledger raw_rows for {x['model']} is {x['raw_rows']}")
    return {"campaign": c, "cells": cells, "mapping": mapping, "bank": bank}


# ---------------------------------------------------------------------------
# statistics


def pearson_rows(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    xc = x - x.mean(axis=-1, keepdims=True)
    yc = y - y.mean(axis=-1, keepdims=True)
    num = (xc * yc).sum(axis=-1)
    den = np.sqrt((xc ** 2).sum(axis=-1) * (yc ** 2).sum(axis=-1))
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / np.where(den > 0, den, 1), np.nan)


def spearman(x, y) -> float:
    rx, ry = rankdata(x), rankdata(y)
    return float(pearson_rows(rx[None, :], ry[None, :])[0])


def spearman_test(x, y, label: str) -> dict:
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = len(x)
    rho = spearman(x, y)
    out = {"n": n, "rho": rho}
    if math.isnan(rho):
        out.update({"p_two_sided": None, "p_one_sided_greater": None, "p_method": "undefined (a constant vector)",
                    "ci_low": None, "ci_high": None, "boot_undefined": None})
        return out
    rx, ry = rankdata(x), rankdata(y)
    if n <= EXACT_MAX_N:
        perms = np.array(list(itertools.permutations(ry)))
        null = pearson_rows(np.broadcast_to(rx, perms.shape), perms)
        ge2 = np.sum(np.abs(null) >= abs(rho) - A.EPS)
        ge1 = np.sum(null >= rho - A.EPS)
        out.update({"p_two_sided": float(ge2 / len(null)), "p_one_sided_greater": float(ge1 / len(null)),
                    "p_method": f"exact, all {len(null)} permutations"})
    else:
        rng = np.random.default_rng(SEED)
        perms = rng.permuted(np.tile(ry, (N_PERM, 1)), axis=1)
        null = pearson_rows(np.broadcast_to(rx, perms.shape), perms)
        ge2 = int(np.sum(np.abs(null) >= abs(rho) - A.EPS))
        ge1 = int(np.sum(null >= rho - A.EPS))
        out.update({"p_two_sided": (ge2 + 1) / (N_PERM + 1), "p_one_sided_greater": (ge1 + 1) / (N_PERM + 1),
                    "p_method": f"{N_PERM} random permutations of one variable, numpy default_rng({SEED}), (b+1)/(N+1)"})
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    bx = rankdata(x[idx], axis=1)
    by = rankdata(y[idx], axis=1)
    boot = pearson_rows(bx, by)
    ok = boot[~np.isnan(boot)]
    out.update({"ci_low": float(np.percentile(ok, 2.5)), "ci_high": float(np.percentile(ok, 97.5)),
                "boot_undefined": int(np.isnan(boot).sum()),
                "ci_method": (f"percentile bootstrap over routes, {N_BOOT} resamples, numpy default_rng({SEED}); "
                              "resamples with a constant vector (rho undefined) are dropped and counted")})
    return out


def cohen_kappa(a: list[bool], b: list[bool]) -> dict:
    n = len(a)
    agree = sum(x == y for x, y in zip(a, b))
    pa, pb = sum(a) / n, sum(b) / n
    constant = [name for name, p in (("first", pa), ("second", pb)) if p in (0.0, 1.0)]
    res = {"n": n, "agreement": agree, "n_admitted_first": sum(a), "n_admitted_second": sum(b)}
    if constant:
        pe = pa * pb + (1 - pa) * (1 - pb)
        formula = None if abs(1 - pe) < A.EPS else (agree / n - pe) / (1 - pe)
        res.update({"kappa": None, "kappa_undefined_reason": (
            "both verdicts are constant, so chance agreement is 1 and kappa is 0/0" if formula is None else
            "one verdict is constant (admits nobody or everybody), so chance agreement equals the observed "
            f"agreement by construction and kappa is {formula:.0f} whatever the other family does; it "
            "measures nothing"),
            "kappa_formula_value": formula, "constant_family": constant})
    else:
        pe = pa * pb + (1 - pa) * (1 - pb)
        res.update({"kappa": (agree / n - pe) / (1 - pe), "kappa_undefined_reason": None, "constant_family": []})
    return res


def perm_name(pred_admit: dict[str, bool], verdict: dict[str, bool]) -> dict:
    return A.exact_permutation(pred_admit, verdict)


# ---------------------------------------------------------------------------
# analyses


def acc(route: A.Route, gate: str) -> float:
    return route.accuracy(gate)


def name_baselines(fam: dict) -> tuple[dict, list[dict]]:
    cells = fam["cells"]
    verdict = {s: r.admitted for s, r in cells.items()}
    res, rows = {}, []
    for name, rule in (("N1", n1_refused), ("N2", n2_refused)):
        pred = {s: not rule(s) for s in verdict}
        perm = perm_name(pred, verdict)
        breaks = sorted(s for s in verdict if pred[s] != verdict[s])
        res[name] = {**perm, "n_routes": len(verdict), "n_admitted": sum(verdict.values()),
                     "n_predicted_refused": sum(not v for v in pred.values()), "broken": bool(breaks),
                     "routes_breaking": breaks,
                     "false_admits": sorted(s for s in breaks if pred[s]),
                     "false_refusals": sorted(s for s in breaks if not pred[s])}
        for s in sorted(verdict):
            rows.append({"family": fam["campaign"].family, "baseline": name, "route": s,
                         "tokens": " ".join(tokens(s)), "predicted_refused": not pred[s],
                         "recorded_admitted": verdict[s], "agrees": pred[s] == verdict[s]})
    return res, rows


def transfer(fams: dict) -> tuple[dict, list[dict]]:
    complete = {f: set(fams[f]["cells"]) for f in FAMILIES}
    all3 = sorted(set.intersection(*complete.values()))
    rows, res = [], {"routes_all_three": all3, "n_all_three": len(all3)}
    for a, b in PAIRS:
        both = sorted(complete[a] & complete[b])
        for scope, routes in (("pairwise", both), ("all_three", all3)):
            for metric in (SR, A.OVERALL):
                x = [acc(fams[a]["cells"][r], metric) for r in routes]
                y = [acc(fams[b]["cells"][r], metric) for r in routes]
                t = spearman_test(x, y, f"{a}-{b}-{scope}-{metric}")
                rows.append({"pair": f"{a}-{b}", "scope": scope, "metric": metric, **t,
                             "ci_includes_zero": None if t["ci_low"] is None else (t["ci_low"] <= 0 <= t["ci_high"]),
                             "routes": ";".join(routes)})
            va = [fams[a]["cells"][r].admitted for r in routes]
            vb = [fams[b]["cells"][r].admitted for r in routes]
            k = cohen_kappa(va, vb)
            res[f"verdict:{a}-{b}:{scope}"] = {**k, "routes": routes,
                                               "disagreeing_routes": [r for r, x, y in zip(routes, va, vb) if x != y]}
    res["correlations"] = rows
    return res, rows


def cheap_baselines(fam: dict) -> tuple[list[dict], dict]:
    c = fam["campaign"]
    slug = {r.route: slug_of(r.route) for r in c.routes}
    verdict = {slug[r.route]: r.admitted for r in c.routes}
    n = len(verdict)
    rows = []

    def add(baseline, definition, pred, extra=None):
        pred = {slug.get(k, k): v for k, v in pred.items()}
        ag = A.agreement(pred, verdict)
        rows.append({"family": c.family, "baseline": baseline, "definition": definition,
                     "agreement": ag, "n_routes": n, "reproduces_verdict": ag == n,
                     "n_predicted_admitted": sum(pred.values()), "n_admitted": sum(verdict.values()),
                     "false_admits": ";".join(sorted(r for r in verdict if pred[r] and not verdict[r])),
                     "false_refusals": ";".join(sorted(r for r in verdict if not pred[r] and verdict[r])),
                     **(extra or {})})

    tab = A.correctness_table(c)
    per_probe = []
    for p in c.probes:
        pred = {r.route: sum(tab[r.route][p].values()) > len(c.repetitions) / 2 for r in c.routes}
        add(f"single_probe_majority:{p}", f"admit iff majority (>=2 of 3) of repetitions of {p} correct", pred,
            {"domain": c.probe_domain[p]})
        per_probe.append(rows[-1]["agreement"])
    summ = {"min": min(per_probe), "median": statistics.median(per_probe), "max": max(per_probe),
            "mean": statistics.fmean(per_probe), "n_probes": len(per_probe),
            "n_reproduce": sum(1 for v in per_probe if v == n)}
    for d in c.domains:
        pred = {r.route: r.gate_counts(d)[0] >= 0.75 * r.gate_counts(d)[1] - A.EPS for r in c.routes}
        add(f"domain_alone:{d}", f"admit iff {d} accuracy >= 0.75", pred, {"domain": d})
    pred = {r.route: r.correct >= 0.80 * r.n - A.EPS for r in c.routes}
    add("overall_alone", "admit iff overall accuracy >= 0.80", pred)
    valid = {r.route: sum(1 for x in r.rows if x["semantic_valid"] is True) / r.n for r in c.routes}
    add("parse_health", "admit iff every one of the 60 rows is semantic_valid", {k: v >= 1 - A.EPS for k, v in valid.items()})
    search = A.threshold_search({slug[k]: v for k, v in valid.items()}, verdict)
    rows[-1]["best_agreement_any_threshold"] = max(search["higher_admits"]["best_agreement"],
                                                   search["lower_admits"]["best_agreement"])
    for name, rule in (("N1", n1_refused), ("N2", n2_refused)):
        pred = {s: not rule(s) for s in verdict}
        perm = perm_name(pred, verdict)
        add(name, f"{name} name rule (prereg section 3): admit iff not flagged", pred,
            {"p_exact_one_sided": perm["p_one_sided"]})
    add("constant_refuse_all", "refuse every route (floor)", {s: False for s in verdict})
    add("constant_admit_all", "admit every route (floor)", {s: True for s in verdict})
    return rows, summ


def reliability(fam: dict) -> tuple[list[dict], dict]:
    c = fam["campaign"]
    rel = A.reliability(c)
    rows = []
    for x in rel["rows"]:
        rows.append({"family": c.family, "route": x["route"] if x["route"] == "ALL" else slug_of(x["route"]),
                     **{k: v for k, v in x.items() if k != "route"}})
    pooled = rel["pooled_all"]
    return rows, pooled


def item_analysis(fams: dict, all3: list[str]) -> tuple[list[dict], list[dict], dict]:
    items, terms = [], []
    for f in FAMILIES:
        fam, c = fams[f], fams[f]["campaign"]
        tab = {slug_of(r.route): A.correctness_table(c)[r.route] for r in c.routes}
        for p in fam["bank"]["order"]:
            if (f, p) not in TERMS:
                fail(f"no term count for {f}/{p}")
            tc, why = TERMS[(f, p)]
            common = [v for r in all3 for v in tab[r][p].values()]
            allr = [v for r in tab for v in tab[r][p].values()]
            items.append({"family": f, "probe_id": p, "domain": c.probe_domain[p], "term_count": tc,
                          "accuracy_common_routes": sum(common) / len(common), "n_responses_common": len(common),
                          "n_routes_common": len(all3),
                          "accuracy_all_complete_routes": sum(allr) / len(allr), "n_responses_all": len(allr),
                          "n_routes_all": len(tab)})
            terms.append({"family": f, "probe_id": p, "domain": c.probe_domain[p], "term_count": tc,
                          "justification": why, "expected_answer": fam["bank"]["probes"][p]["answer"],
                          "question": fam["bank"]["probes"][p]["question"]})
    if len(items) != 60:
        fail(f"item analysis has {len(items)} probes, not 60")
    tests = {}
    for scope, keep in (("all_60", lambda d: True), ("state_reconstruction+terminal_scoring", lambda d: d in (SR, TS))):
        sel = [x for x in items if keep(x["domain"])]
        tests[scope] = spearman_test([x["term_count"] for x in sel], [x["accuracy_common_routes"] for x in sel], scope)
    for f in FAMILIES:
        sel = [x for x in items if x["family"] == f]
        tests[f"within_{f}_20"] = spearman_test([x["term_count"] for x in sel],
                                                [x["accuracy_common_routes"] for x in sel], f)
    return items, terms, tests


def long_table(fams: dict) -> list[dict]:
    rows = []
    for f in FAMILIES:
        for s, r in sorted(fams[f]["cells"].items()):
            for d in list(DOMAINS) + [A.OVERALL]:
                cor, n = r.gate_counts(d)
                rows.append({"route": s, "family": f, "domain": d, "correct": cor, "rows": n,
                             "accuracy": cor / n, "admitted": r.admitted})
    return rows


def wide_table(fams: dict) -> list[dict]:
    slugs = sorted(set().union(*(fams[f]["cells"] for f in FAMILIES)))
    rows = []
    for s in slugs:
        row = {"route": s}
        for f in FAMILIES:
            r = fams[f]["cells"].get(s)
            row[f"{f}_complete"] = r is not None
            row[f"{f}_admitted"] = None if r is None else r.admitted
            row[f"{f}_overall"] = None if r is None else r.overall
            row[f"{f}_{SR}"] = None if r is None else r.accuracy(SR)
            row[f"{f}_{TS}"] = None if r is None else r.accuracy(TS)
        rows.append(row)
    return rows


def sr_resolution(fams: dict, routes: list[str]) -> dict:
    out = {}
    for f in FAMILIES:
        vals = [round(fams[f]["cells"][r].accuracy(SR), 6) for r in routes]
        mode, k = Counter(vals).most_common(1)[0]
        out[f] = {"n": len(vals), "n_distinct": len(set(vals)), "mode": mode, "mode_count": k}
    return out


def failures(ledger: list[dict]) -> list[dict]:
    return [{"family": x["family"], "route": x["model"], "run_id": x["run_id"], "state": x["state"],
             "raw_rows": x["raw_rows"], "failure_class": x["failure_class"]}
            for x in ledger if x["admitted_to_analysis"] != "True"]


# ---------------------------------------------------------------------------
# report


def f3(x, nd=3):
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}"


def fp(x):
    if x is None:
        return "n/a"
    if abs(x - 1 / (N_PERM + 1)) < 1e-12:
        return f"<= 1/{N_PERM + 1} (floor)"
    return f"{x:.2g}" if x < 0.001 else f"{x:.4f}"


def report(res: dict) -> str:
    L = []
    w = L.append
    w("# ARR transfer campaign: preregistered analyses")
    w("")
    w(f"Generated {res['generated_utc']} by `scripts/arr/analyze_transfer.py` from the raw probe rows. "
      f"Authority: `{PREREG}` (sections 3, 4, 6, amendments, collection outcome). Every number is written by "
      "the script; regenerate, never edit.")
    w("")
    w("## Inputs and checks that passed")
    w("")
    for f in FAMILIES:
        x = res["families"][f]
        w(f"- {f}: {x['n_complete']} complete cells, {x['n_admitted']} admitted; protocol `{x['protocol_id']}`; "
          f"probe_bank_sha256 `{x['probe_bank_sha256']}` (equal to the frozen hash, to the hash of the bank text in "
          f"`{x['bank_source']}`" + (", and to the family README" if x["readme_hash"] else "") + "); rules context "
          f"`{x['rules_context_sha256']}`.")
    w("- Each cell has exactly 60 rows, per-domain counts and the verdict recompute to its `admission.json`, no route "
      "is complete twice in a family, every transfer-campaign cell matches a `completed` ledger row with "
      "`admitted_to_analysis=True` and the reverse. `failed_runs/` is never opened; failure records come from "
      "`ledger.csv` only.")
    w(f"- Routes complete in all three families: {res['transfer']['n_all_three']} (collection outcome expects 16).")
    w("")
    w("Route id normalisation: `model_route` minus its provider prefix, `@` replaced by `-`, lower case. "
      "v6 directory names and the new directory names both equal this slug (checked).")
    w("")
    w("| family | source | dir | model_route | slug | run |")
    w("|---|---|---|---|---|---|")
    for m in res["route_mapping"]:
        w(f"| {m['family']} | {m['source']}/{m['task_version_or_family_dir']} | {m['directory_name']} | "
          f"`{m['model_route']}` | {m['slug']} | {m['run_id']} |")
    w("")

    w("## Primary 1. Name baselines on F1 (prereg sections 3 and 4.1)")
    w("")
    nb = res["name_baselines_F1"]
    w("Tokens: the lower-cased slug split on `-`, `_`, `.`, `/`, `@`. N1 predicts refused iff a token is `mini`, "
      "`nano` or `lite`. N2 predicts refused iff a token is `mini`, `nano`, `lite`, `haiku` or `gemma`, or the "
      "tokens include both `oss` and `20b`. Agreement = routes whose predicted verdict equals the F1 verdict. "
      "p = share of all C(n, k) labellings with the observed number admitted k whose agreement is at least the "
      "observed one (enumerated).")
    w("")
    w("| rule | n routes | admitted | predicted refused | agreement | routes breaking the rule | labellings >= observed | exact p |")
    w("|---|---|---|---|---|---|---|---|")
    for k in ("N1", "N2"):
        x = nb[k]
        br = ", ".join(f"{r} ({'predicted admit, refused' if r in x['false_admits'] else 'predicted refuse, admitted'})"
                       for r in x["routes_breaking"]) or "none"
        w(f"| {k} | {x['n_routes']} | {x['n_admitted']} | {x['n_predicted_refused']} | {x['observed_agreement']}/{x['n_routes']} | "
          f"{br} | {x['labellings_at_least_as_good']}/{x['labellings']} | {x['p_one_sided']:.4f} |")
    w("")
    w(f"The prereg asked for 24 routes; {nb['N1']['n_routes']} have a complete F1 cell (the rest are unreachable, "
      "excluded, or failed for a non-refusal reason; see Failures). N1 is "
      + ("**broken**" if nb["N1"]["broken"] else "**not broken**") + " on F1; N2 is "
      + ("**broken**" if nb["N2"]["broken"] else "**not broken**") + ".")
    w("")

    w("## Primary 2. Transfer across families (prereg section 4.2, RQ4)")
    w("")
    tr = res["transfer"]
    w("Spearman rho (average ranks for ties) of per-route accuracy between two families. p two-sided = share of "
      "permutations with |rho| >= observed; " + tr["correlations"][0].get("p_method", "") + ". 95% interval: "
      + next(r["ci_method"] for r in tr["correlations"] if r.get("ci_method")) + ". The prereg scope is routes "
      "complete in all three families; the pairwise scope (routes complete in both) is reported beside it.")
    w("")
    w("| pair | scope | metric | n | rho | p two-sided | p one-sided (rho>0) | 95% CI | CI includes 0 | undefined resamples |")
    w("|---|---|---|---|---|---|---|---|---|---|")
    for r in sorted(tr["correlations"], key=lambda r: (r["scope"] != "all_three", r["metric"] != SR, r["pair"])):
        w(f"| {r['pair']} | {r['scope']} | {r['metric']} | {r['n']} | {f3(r['rho'])} | {fp(r['p_two_sided'])} | "
          f"{fp(r['p_one_sided_greater'])} | [{f3(r['ci_low'])}, {f3(r['ci_high'])}] | "
          f"{'yes' if r['ci_includes_zero'] else 'no'} | {r['boot_undefined']} |")
    w("")
    w("Verdict agreement (admit/refuse) between families:")
    w("")
    w("| pair | scope | n | admitted (first, second) | same verdict | Cohen's kappa | routes that differ |")
    w("|---|---|---|---|---|---|---|")
    for key in sorted(k for k in tr if k.startswith("verdict:")):
        x = tr[key]
        _, pair, scope = key.split(":")
        kap = f3(x["kappa"]) if x["kappa"] is not None else f"undefined: {x['kappa_undefined_reason']}"
        w(f"| {pair} | {scope} | {x['n']} | {x['n_admitted_first']}, {x['n_admitted_second']} | {x['agreement']}/{x['n']} | "
          f"{kap} | {', '.join(x['disagreeing_routes']) or '-'} |")
    w("")
    w("Admission counts: " + "; ".join(f"{f} admits {res['families'][f]['n_admitted']} of {res['families'][f]['n_complete']} "
                                       f"({', '.join(res['families'][f]['admitted']) or 'none'})" for f in FAMILIES)
      + ". Over the routes complete in all three families F2 and F3 both admit nobody (F2's only admitted route, "
      "gpt-oss-20b, has no complete F3 cell), so kappa is undefined in every all-three scope.")
    w("")
    w("Resolution of state_reconstruction (descriptive, needed to read the undefined bootstrap resamples): "
      + "; ".join(f"{f}: {v['n_distinct']} distinct values over {v['n']} routes, the most common "
                  f"({v['mode']:.3f}) on {v['mode_count']}" for f, v in res["sr_resolution"].items()) + ".")
    w("")

    w("## Primary 3. Cheap baselines per family (prereg section 4.3, RQ3)")
    w("")
    w("Agreement with that family's own verdict over its complete routes. Single probe: admit iff at least 2 of the "
      "3 repetitions of that probe are correct; summarised over the 20 probes. Domain alone: that domain's accuracy "
      ">= 0.75. Overall alone: overall >= 0.80. Parse health: all 60 rows semantic_valid (best agreement of any "
      "one-sided threshold on the valid rate in brackets). N1, N2 as in section 3 with exact p.")
    w("")
    w("| family | n | admitted | single probe (min / median / max; probes exact) | overall alone | parse health | N1 | N2 | refuse all |")
    w("|---|---|---|---|---|---|---|---|---|")
    for f in FAMILIES:
        cb = {x["baseline"]: x for x in res["cheap_baselines"] if x["family"] == f}
        s = res["single_probe_summary"][f]
        n = cb["N1"]["n_routes"]
        w(f"| {f} | {n} | {cb['N1']['n_admitted']} | {s['min']} / {s['median']} / {s['max']}; {s['n_reproduce']} of 20 | "
          f"{cb['overall_alone']['agreement']}/{n} | {cb['parse_health']['agreement']}/{n} "
          f"({cb['parse_health']['best_agreement_any_threshold']}/{n}) | {cb['N1']['agreement']}/{n} (p={cb['N1']['p_exact_one_sided']:.4f}) | "
          f"{cb['N2']['agreement']}/{n} (p={cb['N2']['p_exact_one_sided']:.4f}) | {cb['constant_refuse_all']['agreement']}/{n} |")
    w("")
    w("| family | " + " | ".join(DOMAINS) + " |")
    w("|---|" + "---|" * len(DOMAINS))
    for f in FAMILIES:
        cb = {x["baseline"]: x for x in res["cheap_baselines"] if x["family"] == f}
        n = cb["N1"]["n_routes"]
        w(f"| {f} | " + " | ".join(f"{cb['domain_alone:' + d]['agreement']}/{n}" for d in DOMAINS) + " |")
    w("")
    if any(res["families"][f]["n_admitted"] == 0 for f in FAMILIES):
        z = [f for f in FAMILIES if res["families"][f]["n_admitted"] == 0]
        w(f"{', '.join(z)} admits no route, so refusing everyone reproduces its verdict exactly and every baseline's "
          "agreement there only counts how many routes it refuses; its p-values are 1 by construction (one labelling).")
        w("")

    w("## Primary 4. Reliability (prereg section 4.4, RQ2)")
    w("")
    w("Share of (route, probe) pairs whose three repetitions agree on correctness (all correct or all wrong); "
      "Fleiss kappa on the binary rating. Per route and domain in `reliability.csv`.")
    w("")
    w("| family | route-probe pairs | all 3 agree | proportion | Fleiss kappa |")
    w("|---|---|---|---|---|")
    for f in FAMILIES:
        p = res["reliability_pooled"][f]
        w(f"| {f} | {p['n_route_probe_items']} | {p['n_all_reps_agree']} | {p['prop_all_reps_agree']:.4f} | {f3(p['fleiss_kappa'])} |")
    w("")

    w("## Prereg section 6: decision rule triggered")
    w("")
    for line in res["decision_rules"]["lines"]:
        w(f"- {line}")
    w("")

    w("## Exploratory 5. Item analysis and arithmetic load (not preregistered)")
    w("")
    w("Item accuracy = share of correct responses to that probe over the routes complete in all three families "
      f"({tr['n_all_three']} routes x 3 repetitions = {3 * tr['n_all_three']} responses per probe). {TERM_RULE} "
      "Counts and one-line justifications are in `item_term_counts.csv` for human audit. Spearman as in section 2.")
    w("")
    w("| scope | n probes | rho | p two-sided | 95% CI |")
    w("|---|---|---|---|---|")
    for k, t in res["item_tests"].items():
        w(f"| {k} | {t['n']} | {f3(t['rho'])} | {fp(t['p_two_sided'])} | [{f3(t['ci_low'])}, {f3(t['ci_high'])}] |")
    w("")
    w("| family | probe | domain | terms | accuracy (common routes) |")
    w("|---|---|---|---|---|")
    for x in res["items"]:
        w(f"| {x['family']} | {x['probe_id']} | {x['domain']} | {x['term_count']} | {x['accuracy_common_routes']:.3f} |")
    w("")

    w("## Exploratory 6. Accuracy by route, family and domain")
    w("")
    w("Long form in `route_family_domain.csv`; wide summary in `route_family_wide.csv`. Wide summary "
      "(overall / state_reconstruction; * admitted, - not complete):")
    w("")
    w("| route | F1 | F2 | F3 |")
    w("|---|---|---|---|")
    for r in res["wide"]:
        cells = []
        for f in FAMILIES:
            if not r[f"{f}_complete"]:
                cells.append("-")
            else:
                cells.append(f"{r[f'{f}_overall']:.3f} / {r[f'{f}_{SR}']:.3f}{' *' if r[f'{f}_admitted'] else ''}")
        w(f"| {r['route']} | " + " | ".join(cells) + " |")
    w("")

    w("## Failures (from ledger.csv, never evidence, never counted as refused)")
    w("")
    w("| family | route | run | state | raw rows | class |")
    w("|---|---|---|---|---|---|")
    for x in res["failures"]:
        w(f"| {x['family']} | {x['route']} | {x['run_id']} | {x['state']} | {x['raw_rows']} | {x['failure_class']} |")
    w("")
    cnt = Counter((x["failure_class"]) for x in res["failures"])
    w("Counts by class: " + ", ".join(f"{k} {v}" for k, v in sorted(cnt.items())) + ". Excluded before collection "
      "(prereg amendment, mandatory thinking): gemini-2.5-pro.")
    w("")
    w("## Inputs")
    w("")
    for f in res["inputs"]:
        w(f"- `{f['path']}` sha256 `{f['sha256']}`")
    w("")
    return "\n".join(L)


def write_csv(path: Path, rows: list[dict]) -> None:
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({k: ("" if r.get(k) is None else r.get(k)) for k in fields})


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--output", type=Path, default=OUT)
    args = ap.parse_args(argv)

    ledger = read_ledger()
    fams = {f: load_family(f, ledger) for f in FAMILIES}
    nb, nb_rows = name_baselines(fams["F1"])
    tr, tr_rows = transfer(fams)
    cheap, single = [], {}
    for f in FAMILIES:
        rows, summ = cheap_baselines(fams[f])
        cheap += rows
        single[f] = summ
    rel_rows, rel_pooled = [], {}
    for f in FAMILIES:
        rows, pooled = reliability(fams[f])
        rel_rows += rows
        rel_pooled[f] = pooled
    items, terms, item_tests = item_analysis(fams, tr["routes_all_three"])
    long_rows = long_table(fams)
    wide = wide_table(fams)
    fail_rows = failures(ledger)

    sr3 = [r for r in tr_rows if r["scope"] == "all_three" and r["metric"] == SR]
    lines = []
    if nb["N1"]["broken"]:
        lines.append(f"Rule 1 TRIGGERED: N1 is broken on F1 ({nb['N1']['observed_agreement']}/{nb['N1']['n_routes']} agree; "
                     f"breaking routes: {', '.join(nb['N1']['routes_breaking'])}). The paper argues the screen carries "
                     "information beyond the name, reports how much, and asks whether it transfers.")
    else:
        lines.append("Rule 2 TRIGGERED: N1 is not broken on the complete F1 routes.")
    lines.append(f"Rule 2 NOT triggered as written: it is conditioned on N1 not being broken on 24 routes; "
                 f"{nb['N1']['n_routes']} routes have a complete F1 cell.")
    inc = [r for r in sr3 if r["ci_includes_zero"]]
    for r in sr3:
        lines.append(f"state_reconstruction {r['pair']} over {r['n']} routes: rho {f3(r['rho'])}, 95% CI "
                     f"[{f3(r['ci_low'])}, {f3(r['ci_high'])}] ({'includes' if r['ci_includes_zero'] else 'excludes'} zero).")
    if inc:
        lines.append(f"Rule 3 TRIGGERED for {', '.join(r['pair'] for r in inc)}: state reconstruction does not transfer "
                     "(interval includes zero); the paper reports that a screen must be rebuilt per environment"
                     + ("" if len(inc) == len(sr3) else " (for the pairs listed; the other pairs' intervals exclude zero)") + ".")
    else:
        lines.append("Rule 3 NOT triggered: every pair's state_reconstruction interval excludes zero.")
    for r in tr_rows:
        if r["scope"] == "pairwise" and r["metric"] == SR and r["n"] != tr["n_all_three"]:
            lines.append(f"Sensitivity, not the prereg scope: {r['pair']} over the {r['n']} routes complete in both "
                         f"gives rho {f3(r['rho'])}, 95% CI [{f3(r['ci_low'])}, {f3(r['ci_high'])}] "
                         f"({'includes' if r['ci_includes_zero'] else 'excludes'} zero).")
    ov = [r for r in tr_rows if r["scope"] == "all_three" and r["metric"] == A.OVERALL]
    lines.append("Overall accuracy over the same routes: " + "; ".join(
        f"{r['pair']} rho {f3(r['rho'])} [{f3(r['ci_low'])}, {f3(r['ci_high'])}]" for r in ov) + ".")

    res = {
        "generator": "scripts/arr/analyze_transfer.py",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "prereg": PREREG,
        "seed": SEED, "n_perm": N_PERM, "n_boot": N_BOOT,
        "families": {f: {"n_complete": len(fams[f]["cells"]),
                         "n_admitted": sum(r.admitted for r in fams[f]["cells"].values()),
                         "admitted": sorted(s for s, r in fams[f]["cells"].items() if r.admitted),
                         "routes": sorted(fams[f]["cells"]),
                         "protocol_id": fams[f]["campaign"].protocol_id,
                         "probe_bank_sha256": fams[f]["campaign"].probe_bank_sha256,
                         "rules_context_sha256": fams[f]["campaign"].rules_context_sha256,
                         "bank_source": fams[f]["bank"]["source"], "readme_hash": fams[f]["bank"]["readme_hash"],
                         "pruned_failed_dirs": fams[f]["campaign"].pruned_failed_dirs}
                     for f in FAMILIES},
        "route_mapping": [m for f in FAMILIES for m in fams[f]["mapping"]],
        "name_baselines_F1": nb,
        "transfer": tr,
        "cheap_baselines": cheap,
        "single_probe_summary": single,
        "reliability_pooled": rel_pooled,
        "sr_resolution": sr_resolution(fams, tr["routes_all_three"]),
        "decision_rules": {"lines": lines, "N1_broken_on_F1": nb["N1"]["broken"],
                           "state_reconstruction_ci_includes_zero": {r["pair"]: r["ci_includes_zero"] for r in sr3}},
        "term_rule": TERM_RULE,
        "item_tests": item_tests,
        "items": items,
        "wide": wide,
        "failures": fail_rows,
    }
    if any(A.FAILED in Path(f["path"]).parts for f in A.READ_LOG):
        fail("a failed_runs file was read")
    res["inputs"] = sorted({f["path"]: f for f in A.READ_LOG}.values(), key=lambda f: f["path"])

    out = args.output
    out.mkdir(parents=True, exist_ok=True)
    write_csv(out / "route_family_domain.csv", long_rows)
    write_csv(out / "route_family_wide.csv", wide)
    write_csv(out / "name_baselines.csv", nb_rows)
    write_csv(out / "transfer_correlations.csv", tr_rows)
    write_csv(out / "cheap_baselines.csv", cheap)
    write_csv(out / "reliability.csv", rel_rows)
    write_csv(out / "item_analysis.csv", items)
    write_csv(out / "item_term_counts.csv", terms)
    write_csv(out / "failures.csv", fail_rows)
    (out / "transfer.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    (out / "report.md").write_text(report(res), encoding="utf-8")
    print(f"wrote {A._rel(out)}")
    for line in lines:
        print(line)


if __name__ == "__main__":
    main()
