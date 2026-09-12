"""Cheaper-screen baselines for the comprehension-screen evaluation.

The screen costs sixty scored answers per endpoint. This script asks what
cheaper signals reproduce its admit-or-refuse verdict on the same nine
endpoints, which is the discriminant-validity question:

  1. the endpoint name, read for a size word;
  2. each probe domain on its own, at its best achievable threshold;
  3. one single probe, every probe tried in turn;
  4. reported parse health.

Everything is read from the frozen probe-level log, so no number here depends
on a summary that was written by hand.

Inputs
  results/frontier/admission_campaign_v6/**/admission.json
  results/frontier/admission_campaign_v6/**/raw_responses.jsonl

Output
  results/derived/screen_baselines/screen_baselines.json
"""
from __future__ import annotations

import glob
import json
import re
from collections import defaultdict
from itertools import combinations
from math import comb
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CAMPAIGN = ROOT / "results" / "frontier" / "admission_campaign_v6"
OUT_DIR = ROOT / "results" / "derived" / "screen_baselines"

SIZE_WORD = re.compile(r"(?:^|[^a-z])(mini|nano|lite)(?:[^a-z]|$)")


def load() -> tuple[dict[str, dict], dict[str, list[dict]]]:
    verdicts: dict[str, dict] = {}
    for path in glob.glob(str(CAMPAIGN / "**" / "admission.json"), recursive=True):
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        verdicts[d["model_route"]] = d
    rows: dict[str, list[dict]] = defaultdict(list)
    for path in glob.glob(str(CAMPAIGN / "**" / "raw_responses.jsonl"), recursive=True):
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            rows[d["model_route"]].append(d)
    return verdicts, rows


def agreement(predicted: set[str], truth: set[str], routes: list[str]) -> int:
    return sum(1 for r in routes if (r in predicted) == (r in truth))


def exact_p(n_routes: int, n_admitted: int) -> float:
    """One-sided probability that a random split of this size agrees perfectly."""
    return 1.0 / comb(n_routes, n_admitted)


def main() -> None:
    verdicts, rows = load()
    routes = sorted(verdicts)
    admitted = {r for r in routes if verdicts[r]["admitted_for_gameplay"]}
    n = len(routes)
    report: dict[str, object] = {
        "generator": "scripts/analyze_screen_baselines.py",
        "source": "results/frontier/admission_campaign_v6",
        "n_routes": n,
        "n_admitted": len(admitted),
        "admitted": sorted(admitted),
        "refused": sorted(set(routes) - admitted),
        "calls_per_route": {r: len(rows[r]) for r in routes},
    }

    # ---- 1. the name -------------------------------------------------
    name_admits = {r for r in routes if not SIZE_WORD.search(r.lower())}
    report["name_baseline"] = {
        "rule": "refuse when mini, nano or lite appears as a whole word",
        "calls_per_route": 0,
        "predicted_admitted": sorted(name_admits),
        "agreement": agreement(name_admits, admitted, routes),
        "exact_one_sided_p": exact_p(n, len(admitted)),
        "n_labellings": comb(n, len(admitted)),
    }

    # ---- 2. one domain at a time ------------------------------------
    domains = sorted({d["domain"] for r in routes for d in rows[r]})
    per_domain: dict[str, dict] = {}
    for dom in domains:
        correct = {r: sum(1 for d in rows[r] if d["domain"] == dom and d["semantic_correct"])
                   for r in routes}
        total = {r: sum(1 for d in rows[r] if d["domain"] == dom) for r in routes}
        size = sorted(set(total.values()))
        best = None
        for k in range(0, max(total.values()) + 1):
            pred = {r for r in routes if correct[r] >= k}
            a = agreement(pred, admitted, routes)
            if best is None or a > best[1]:
                best = (k, a, sorted(pred))
        per_domain[dom] = {
            "calls_per_route": size[0] if len(size) == 1 else size,
            "correct_per_route": correct,
            "best_threshold_in_correct_answers": best[0],
            "best_agreement": best[1],
            "predicted_admitted_at_best": best[2],
            "reproduces_verdict": best[1] == n,
        }
    report["domain_baselines"] = per_domain

    # ---- 3. one probe at a time -------------------------------------
    probes = sorted({(d["domain"], d["probe_id"]) for r in routes for d in rows[r]})
    per_probe = []
    for dom, pid in probes:
        correct = {r: sum(1 for d in rows[r]
                          if d["probe_id"] == pid and d["semantic_correct"])
                   for r in routes}
        reps = {sum(1 for d in rows[r] if d["probe_id"] == pid) for r in routes}
        best = None
        for k in range(0, max(reps) + 1):
            pred = {r for r in routes if correct[r] >= k}
            a = agreement(pred, admitted, routes)
            if best is None or a > best[1]:
                best = (k, a, sorted(pred))
        per_probe.append({
            "domain": dom,
            "probe_id": pid,
            "calls_per_route": sorted(reps),
            "correct_per_route": correct,
            "best_threshold_in_correct_answers": best[0],
            "best_agreement": best[1],
            "reproduces_verdict": best[1] == n,
        })
    report["single_probe_baselines"] = per_probe
    report["single_probe_summary"] = {
        "n_probes": len(per_probe),
        "n_reproducing_verdict": sum(1 for p in per_probe if p["reproduces_verdict"]),
        "reproducing": [f"{p['domain']}/{p['probe_id']}" for p in per_probe
                        if p["reproducing" if False else "reproduces_verdict"]],
        "best_agreement_distribution": {
            str(k): sum(1 for p in per_probe if p["best_agreement"] == k)
            for k in sorted({p["best_agreement"] for p in per_probe})
        },
    }

    # ---- 3b. multiplicity: twenty probes were searched, not one ------
    # Exact test. Enumerate every way of relabelling which five of the nine
    # endpoints are admitted, and ask how often SOME probe in the bank would
    # have reproduced that labelling as well as the best probe reproduces the
    # real one. This charges the search over twenty probes honestly.
    probe_correct = {
        (dom, pid): {r: sum(1 for d in rows[r]
                            if d["probe_id"] == pid and d["semantic_correct"])
                     for r in routes}
        for dom, pid in probes
    }

    def best_agreement_against(label: set[str]) -> int:
        best_a = 0
        for key, correct in probe_correct.items():
            reps = max(sum(1 for d in rows[r] if d["probe_id"] == key[1]) for r in routes)
            for k in range(0, reps + 1):
                pred = {r for r in routes if correct[r] >= k}
                best_a = max(best_a, agreement(pred, label, routes))
        return best_a

    observed_best = max(p["best_agreement"] for p in per_probe)
    hits = 0
    labellings = list(combinations(routes, len(admitted)))
    for lab in labellings:
        if best_agreement_against(set(lab)) >= observed_best:
            hits += 1
    report["single_probe_multiplicity_test"] = {
        "question": "searching twenty probes, how often does some probe reproduce an "
                    "arbitrary five-of-nine labelling as well as the best probe "
                    "reproduces the real verdict?",
        "observed_best_agreement": observed_best,
        "n_labellings": len(labellings),
        "n_labellings_matched": hits,
        "exact_p": hits / len(labellings),
    }

    # ---- 4. parse health --------------------------------------------
    valid = {r: sum(1 for d in rows[r] if d["semantic_valid"]) for r in routes}
    total_rows = {r: len(rows[r]) for r in routes}
    best = None
    for k in range(0, max(total_rows.values()) + 1):
        pred = {r for r in routes if valid[r] >= k}
        a = agreement(pred, admitted, routes)
        if best is None or a > best[1]:
            best = (k, a, sorted(pred))
    report["parse_health_baseline"] = {
        "valid_per_route": valid,
        "rows_per_route": total_rows,
        "n_invalid_total": sum(total_rows.values()) - sum(valid.values()),
        "invalid_rows": [
            {"route": r, "domain": d["domain"], "probe_id": d["probe_id"]}
            for r in routes for d in rows[r] if not d["semantic_valid"]
        ],
        "best_threshold_in_valid_answers": best[0],
        "best_agreement": best[1],
        "reproduces_verdict": best[1] == n,
    }

    # ---- 5. how small a battery still reproduces the verdict ---------
    # Score state reconstruction only, using k of its probes, all repetitions.
    sr_probes = [p for d, p in probes if d == "state_reconstruction"]
    smallest = None
    for size in range(1, len(sr_probes) + 1):
        found = []
        for subset in combinations(sr_probes, size):
            correct = {r: sum(1 for d in rows[r]
                              if d["probe_id"] in subset and d["semantic_correct"])
                       for r in routes}
            calls = max(sum(1 for d in rows[r] if d["probe_id"] in subset) for r in routes)
            for k in range(0, calls + 1):
                pred = {r for r in routes if correct[r] >= k}
                if pred == admitted:
                    found.append({"probes": list(subset),
                                  "threshold_in_correct_answers": k,
                                  "calls_per_route": calls})
                    break
        if found:
            smallest = {"n_probes": size, "calls_per_route": found[0]["calls_per_route"],
                        "n_subsets_that_work": len(found),
                        "n_subsets_tried": comb(len(sr_probes), size),
                        "example": found[0]}
            break
    report["smallest_state_reconstruction_battery"] = smallest

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUT_DIR / "screen_baselines.json").write_text(
        json.dumps(report, indent=2, sort_keys=False), encoding="utf-8")

    # ---- console summary --------------------------------------------
    print(f"{n} routes, {len(admitted)} admitted, {sum(total_rows.values())} scored answers")
    print()
    print(f"name baseline: agreement {report['name_baseline']['agreement']}/{n}, "
          f"exact one-sided p = 1/{report['name_baseline']['n_labellings']} = "
          f"{report['name_baseline']['exact_one_sided_p']:.4f}")
    print()
    print("domain baselines (best achievable threshold on that domain alone):")
    for dom, v in per_domain.items():
        print(f"  {dom:22s} calls/route {v['calls_per_route']!s:>3s}  "
              f"agreement {v['best_agreement']}/{n}  "
              f"reproduces verdict: {v['reproduces_verdict']}")
    print()
    s = report["single_probe_summary"]
    print(f"single probes: {s['n_reproducing_verdict']} of {s['n_probes']} reproduce the verdict")
    print("  agreement distribution:", s["best_agreement_distribution"])
    for p in per_probe:
        if p["reproduces_verdict"]:
            print(f"    {p['domain']}/{p['probe_id']} at {p['best_threshold_in_correct_answers']} "
                  f"of {p['calls_per_route']}")
    m = report["single_probe_multiplicity_test"]
    print(f"  charging the search over {s['n_probes']} probes: some probe reaches "
          f"{m['observed_best_agreement']}/{n} on {m['n_labellings_matched']} of "
          f"{m['n_labellings']} labellings, exact p = {m['exact_p']:.4f}")
    print()
    ph = report["parse_health_baseline"]
    print(f"parse health: {ph['n_invalid_total']} unparsed answers, "
          f"best agreement {ph['best_agreement']}/{n}, reproduces verdict: {ph['reproduces_verdict']}")
    print("  unparsed rows:", [(r["route"], r["domain"]) for r in ph["invalid_rows"]])
    print()
    print("smallest state-reconstruction battery that reproduces the verdict:")
    print(" ", smallest)
    print()
    print(f"written: {OUT_DIR / 'screen_baselines.json'}")


if __name__ == "__main__":
    main()
