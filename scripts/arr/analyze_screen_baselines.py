#!/usr/bin/env python3
"""What does a sixty-call comprehension battery buy over a cheap screen?

The ARR evaluation paper (docs/acl-audit-paper-plan.md, sections 3, 5.3, 5.4)
must say, against the recorded admit/refuse verdicts, how far each cheap
alternative gets: one probe, one domain, the overall score alone, parse health,
and the route's name.  It must also re-derive the threshold band (E12), report
how repeatable the battery is across its three repetitions, and settle whether a
per-route opening-move seat gap (E21) can be computed at all.

Why the design is fail-closed.  Every number here is compared with a verdict, so
a route whose row count, probe bank, protocol, per-domain sums or recomputed
verdict does not check out would silently move an agreement count.  The script
refuses instead: any route with a row count other than --expected-rows, any
probe-bank or rules-context hash that differs across routes, any recomputed
verdict that differs from admission.json, any duplicate route across campaign
roots, and any attempt to open a file under failed_runs/ all stop the run.

Why routes and families are CLI arguments.  The paper needs new routes (a
widened roster) and new task families (F2, F3) administered with the same
raw_responses.jsonl schema plus a "family" field.  Add a campaign directory with
--campaign-root (repeatable) and select a family with --family-label.  Rows
without a "family" field are the original race bank and count as F1.

Nothing outside --output is written.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CAMPAIGN = ROOT / "results" / "frontier" / "admission_campaign_v6"
DEFAULT_BASELINE = ROOT / "results" / "frontier" / "baseline_campaign_v6"
DEFAULT_AUDIT_CSV = DEFAULT_BASELINE / "derived" / "audit_versus_behaviour.csv"
DEFAULT_SEAT_LOO = ROOT / "results" / "derived" / "seat_confound" / "seat_gap_leave_one_out.csv"
DEFAULT_OUT = ROOT / "results" / "derived" / "arr_screen_baselines"
DEFAULT_E21_ROUTE = "openai/gpt-5.4-nano-2026-03-17"

LEGACY_FAMILY = "F1"
FAILED = "failed_runs"
EPS = 1e-9
OVERALL = "overall"
DOMAIN_ORDER = (
    "rule_recall",
    "stage_payoff",
    "state_reconstruction",
    "state_transition",
    "terminal_scoring",
    "expected_payoff",
)
BINDING_DOMAIN = "state_reconstruction"
SIZE_WORDS = ("mini", "nano", "lite")
BAND_HALF_WIDTH = 0.05
BAND_STEP = 0.01

READ_LOG: list[dict] = []


def fail(message: str) -> None:
    raise SystemExit(f"REFUSED: {message}")


def _guard(path: Path) -> Path:
    path = Path(path).resolve()
    if FAILED in path.parts:
        fail(f"attempted to read a failure record: {path}")
    return path


def read_text(path: Path) -> str:
    path = _guard(path)
    data = path.read_bytes()
    READ_LOG.append({"path": _rel(path), "sha256": hashlib.sha256(data).hexdigest()})
    return data.decode("utf-8")


def _rel(path: Path) -> str:
    try:
        return Path(path).resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return Path(path).resolve().as_posix()


def walk_live(root: Path, filename: str) -> tuple[list[Path], int]:
    """Every `filename` under root, never entering a failed_runs directory."""
    hits, pruned = [], 0
    for here, dirs, files in os.walk(root):
        keep = [d for d in dirs if d != FAILED]
        pruned += len(dirs) - len(keep)
        dirs[:] = sorted(keep)
        if filename in files:
            hits.append(Path(here) / filename)
    return sorted(hits), pruned


def domain_sort_key(domain: str) -> tuple[int, str]:
    return (DOMAIN_ORDER.index(domain) if domain in DOMAIN_ORDER else len(DOMAIN_ORDER), domain)


# ---------------------------------------------------------------------------
# loading


@dataclass
class Route:
    route: str
    family: str
    raw_path: Path
    rows: list[dict]
    admission: dict
    manifest: dict
    admitted: bool
    domains: dict[str, dict] = field(default_factory=dict)

    @property
    def n(self) -> int:
        return len(self.rows)

    @property
    def correct(self) -> int:
        return sum(1 for r in self.rows if r["semantic_correct"] is True)

    @property
    def overall(self) -> float:
        return self.correct / self.n

    def gate_counts(self, gate: str) -> tuple[int, int]:
        if gate == OVERALL:
            return self.correct, self.n
        d = self.domains[gate]
        return d["correct"], d["rows"]

    def accuracy(self, gate: str) -> float:
        c, n = self.gate_counts(gate)
        return c / n


@dataclass
class Campaign:
    family: str
    roots: list[str]
    routes: list[Route]
    probes: list[str]
    probe_domain: dict[str, str]
    repetitions: list[int]
    domains: list[str]
    thresholds: dict[str, float]
    gates: dict[str, float]
    diagnostic_domains: list[str]
    probe_bank_sha256: str
    rules_context_sha256: str | None
    protocol_id: str
    pruned_failed_dirs: int

    @property
    def verdict(self) -> dict[str, bool]:
        return {r.route: r.admitted for r in self.routes}

    def by_route(self) -> dict[str, Route]:
        return {r.route: r for r in self.routes}


def _family_of(obj: dict) -> str:
    return obj.get("family", LEGACY_FAMILY)


def _gates_from(adm: dict, domains: list[str]) -> tuple[dict[str, float], list[str]]:
    thresholds = adm["admission_thresholds"]
    gates: dict[str, float] = {}
    diagnostic = []
    for key, value in thresholds.items():
        if not key.endswith("_accuracy_min"):
            fail(f"unrecognised threshold key {key!r} in {adm['model_route']}")
        name = key[: -len("_accuracy_min")]
        if name == "overall":
            gates[OVERALL] = float(value)
        elif name in domains:
            if adm.get(f"{name}_is_diagnostic_only") is True:
                diagnostic.append(name)
            else:
                gates[name] = float(value)
        else:
            fail(f"threshold {key!r} names a domain absent from the probe bank")
    return gates, sorted(diagnostic, key=domain_sort_key)


def _recompute_verdict(route: Route, gates: dict[str, float]) -> bool:
    for gate, thr in gates.items():
        c, n = route.gate_counts(gate)
        if c < thr * n - EPS:
            return False
    return True


def load_campaign(roots: list[Path], family: str, expected_rows: int) -> Campaign:
    routes: dict[str, Route] = {}
    pruned_total = 0
    for root in roots:
        root = Path(root).resolve()
        if FAILED in root.parts:
            fail(f"campaign root lies inside a failure record: {root}")
        if not root.is_dir():
            fail(f"campaign root does not exist: {root}")
        raws, pruned = walk_live(root, "raw_responses.jsonl")
        pruned_total += pruned
        for raw in raws:
            all_rows = [json.loads(line) for line in read_text(raw).splitlines() if line.strip()]
            rows = [r for r in all_rows if _family_of(r) == family]
            if not rows:
                continue
            adm_path, man_path = raw.with_name("admission.json"), raw.with_name("run_manifest.json")
            if not adm_path.is_file() or not man_path.is_file():
                fail(f"{_rel(raw)} lacks a sibling admission.json or run_manifest.json")
            adm = json.loads(read_text(adm_path))
            man = json.loads(read_text(man_path))
            if _family_of(adm) != family:
                fail(f"{_rel(adm_path)} is family {_family_of(adm)!r}, rows are {family!r}")
            route = adm["model_route"]
            if len(rows) != expected_rows:
                fail(f"{route} has {len(rows)} {family} rows, not {expected_rows}")
            if adm.get("n_rows") != len(rows):
                fail(f"{route}: admission.json n_rows {adm.get('n_rows')} != {len(rows)} raw rows")
            if man.get("status") != "completed":
                fail(f"{route}: run_manifest status is {man.get('status')!r}")
            if man.get("expected_rows") not in (None, expected_rows):
                fail(f"{route}: manifest expects {man.get('expected_rows')} rows")
            if man.get("model_route") != route or any(r["model_route"] != route for r in rows):
                fail(f"{route}: model_route disagrees between rows, manifest and admission.json")
            protocols = {r["protocol_id"] for r in rows} | {adm["protocol_id"], man["protocol_id"]}
            if len(protocols) != 1:
                fail(f"{route}: protocol_id disagrees within the route: {sorted(protocols)}")
            for r in rows:
                if "probe_bank_sha256" in r and r["probe_bank_sha256"] != man.get("probe_bank_sha256"):
                    fail(f"{route}: a row's probe_bank_sha256 differs from its manifest")
            keys = Counter((r["probe_id"], r["repetition"]) for r in rows)
            if max(keys.values()) != 1:
                fail(f"{route}: a (probe, repetition) pair appears more than once")
            if route in routes:
                fail(f"{route} appears twice across campaign roots "
                     f"({_rel(routes[route].raw_path)} and {_rel(raw)}); refusing to let one displace the other")
            obj = Route(route=route, family=family, raw_path=raw, rows=rows, admission=adm,
                        manifest=man, admitted=bool(adm["admitted_for_gameplay"]))
            per = defaultdict(lambda: {"rows": 0, "valid": 0, "correct": 0})
            for r in rows:
                d = per[r["domain"]]
                d["rows"] += 1
                d["valid"] += int(r["semantic_valid"] is True)
                d["correct"] += int(r["semantic_correct"] is True)
            obj.domains = dict(per)
            recorded = adm["by_domain"]
            if set(recorded) != set(per):
                fail(f"{route}: domains in admission.json differ from the raw rows")
            for dom, d in per.items():
                for k in ("rows", "valid", "correct"):
                    if recorded[dom][k] != d[k]:
                        fail(f"{route}: {dom} {k} recorded {recorded[dom][k]}, recomputed {d[k]}")
            if abs(adm["overall_accuracy"] - obj.overall) > EPS:
                fail(f"{route}: recorded overall accuracy {adm['overall_accuracy']} != {obj.overall}")
            routes[route] = obj

    if not routes:
        fail(f"no route carries {family} rows under {', '.join(str(r) for r in roots)}")
    ordered = sorted(routes.values(), key=lambda r: r.route)
    first = ordered[0]

    def same(label: str, getter) -> object:
        values = {r.route: getter(r) for r in ordered}
        if len(set(map(json.dumps, values.values()))) != 1:
            fail(f"{label} differs across routes: {values}")
        return getter(first)

    bank = same("probe_bank_sha256", lambda r: r.manifest.get("probe_bank_sha256"))
    if not bank:
        fail("a route's run_manifest carries no probe_bank_sha256")
    rules = same("rules_context_sha256", lambda r: r.manifest.get("rules_context_sha256"))
    protocol = same("protocol_id", lambda r: r.admission["protocol_id"])
    probe_domain = same("probe-to-domain map",
                        lambda r: dict(sorted({x["probe_id"]: x["domain"] for x in r.rows}.items())))
    for r in ordered:
        per_probe = defaultdict(set)
        for x in r.rows:
            per_probe[x["probe_id"]].add(x["domain"])
        if any(len(v) != 1 for v in per_probe.values()):
            fail(f"{r.route}: a probe maps to more than one domain")
    reps = same("repetition layout",
                lambda r: sorted({json.dumps(sorted(x["repetition"] for x in r.rows if x["probe_id"] == p))
                                  for p in {y["probe_id"] for y in r.rows}}))
    if len(reps) != 1:
        fail("probes do not share one repetition layout")
    repetitions = json.loads(reps[0])
    thresholds = same("admission_thresholds", lambda r: r.admission["admission_thresholds"])
    domains = sorted({d for d in probe_domain.values()}, key=domain_sort_key)
    gates, diagnostic = _gates_from(first.admission, domains)
    same("diagnostic-only domains", lambda r: _gates_from(r.admission, domains)[1])
    for r in ordered:
        if _recompute_verdict(r, gates) != r.admitted:
            fail(f"{r.route}: recomputed verdict differs from admitted_for_gameplay={r.admitted}")
    for gate in gates:
        if gate != OVERALL:
            same(f"row count of gate {gate}", lambda r, g=gate: r.domains[g]["rows"])
    probes = sorted(probe_domain, key=lambda p: (domain_sort_key(probe_domain[p]), p))
    return Campaign(
        family=family,
        roots=[_rel(Path(r)) for r in roots],
        routes=ordered,
        probes=probes,
        probe_domain=probe_domain,
        repetitions=repetitions,
        domains=domains,
        thresholds={k: float(v) for k, v in thresholds.items()},
        gates=gates,
        diagnostic_domains=diagnostic,
        probe_bank_sha256=bank,
        rules_context_sha256=rules,
        protocol_id=protocol,
        pruned_failed_dirs=pruned_total,
    )


# ---------------------------------------------------------------------------
# shared arithmetic


def agreement(predicted: dict[str, bool], verdict: dict[str, bool]) -> int:
    return sum(1 for r, v in verdict.items() if predicted[r] == v)


def threshold_search(values: dict[str, float], verdict: dict[str, bool]) -> dict:
    """Best agreement of any one-sided cut on `values`, in either direction.

    'higher_admits': admit iff value >= t.  'lower_admits': admit iff value <= t.
    Every partition a threshold can induce is enumerated, including admit-none.
    """
    finite = {r: v for r, v in values.items() if v is not None and not math.isnan(v)}
    if len(finite) != len(verdict):
        return {"available": False, "missing_routes": sorted(set(verdict) - set(finite))}
    cuts = sorted(set(finite.values()))
    out = {"available": True, "n_distinct_values": len(cuts)}
    for direction in ("higher_admits", "lower_admits"):
        best, exact = -1, []
        candidates = cuts + [math.inf] if direction == "higher_admits" else [-math.inf] + cuts
        for t in candidates:
            pred = {r: (v >= t - EPS) if direction == "higher_admits" else (v <= t + EPS)
                    for r, v in finite.items()}
            a = agreement(pred, verdict)
            if a > best:
                best = a
            if a == len(verdict):
                exact.append(t)
        out[direction] = {"best_agreement": best,
                          "exact_cuts": [c for c in exact if math.isfinite(c)],
                          "reproduces_verdict": bool(exact)}
    adm = [v for r, v in finite.items() if verdict[r]]
    ref = [v for r, v in finite.items() if not verdict[r]]
    out["admitted_range"] = [min(adm), max(adm)] if adm else None
    out["refused_range"] = [min(ref), max(ref)] if ref else None
    out["any_threshold_reproduces_verdict"] = (out["higher_admits"]["reproduces_verdict"]
                                               or out["lower_admits"]["reproduces_verdict"])
    return out


def summarise_agreements(values: list[int], n_routes: int) -> dict:
    return {
        "n": len(values),
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
        "mean": statistics.fmean(values),
        "n_exact": sum(1 for v in values if v == n_routes),
        "histogram": {str(k): v for k, v in sorted(Counter(values).items())},
    }


def fleiss_binary(correct_counts: list[int], raters: int) -> float | None:
    """Fleiss' kappa for items rated correct/incorrect by `raters` repetitions."""
    n_items = len(correct_counts)
    if n_items == 0 or raters < 2:
        return None
    p_i = [((c * c + (raters - c) ** 2) - raters) / (raters * (raters - 1)) for c in correct_counts]
    p_bar = sum(p_i) / n_items
    p1 = sum(correct_counts) / (n_items * raters)
    p_e = p1 ** 2 + (1 - p1) ** 2
    if abs(1 - p_e) < EPS:
        return None
    return (p_bar - p_e) / (1 - p_e)


def fmt(x, nd=3) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def pct(x: float, nd=1) -> str:
    return "n/a" if x is None else f"{100 * x:.{nd}f}"


# ---------------------------------------------------------------------------
# E14a single probe / single response


def correctness_table(c: Campaign) -> dict[str, dict[str, dict[int, bool]]]:
    table: dict[str, dict[str, dict[int, bool]]] = {}
    for r in c.routes:
        t = defaultdict(dict)
        for x in r.rows:
            t[x["probe_id"]][x["repetition"]] = x["semantic_correct"] is True
        table[r.route] = dict(t)
    return table


def e14a(c: Campaign) -> dict:
    verdict = c.verdict
    tab = correctness_table(c)
    rep0 = c.repetitions[0]
    n_reps = len(c.repetitions)
    probe_rows = []
    for p in c.probes:
        rules = {
            "rep0": {r: tab[r][p][rep0] for r in verdict},
            "majority": {r: sum(tab[r][p].values()) > n_reps / 2 for r in verdict},
        }
        for rule, pred in rules.items():
            probe_rows.append({
                "probe_id": p,
                "domain": c.probe_domain[p],
                "rule": rule,
                "agreement": agreement(pred, verdict),
                "n_routes": len(verdict),
                "reproduces_verdict": agreement(pred, verdict) == len(verdict),
                "n_predicted_admitted": sum(pred.values()),
                "false_admits": ";".join(sorted(r for r in verdict if pred[r] and not verdict[r])),
                "false_refusals": ";".join(sorted(r for r in verdict if not pred[r] and verdict[r])),
            })
    response_rows = []
    for p in c.probes:
        for rep in c.repetitions:
            pred = {r: tab[r][p][rep] for r in verdict}
            response_rows.append({
                "probe_id": p, "domain": c.probe_domain[p], "repetition": rep,
                "agreement": agreement(pred, verdict), "n_routes": len(verdict),
                "reproduces_verdict": agreement(pred, verdict) == len(verdict),
            })
    summaries = {}
    for scope, keep in (("all_probes", lambda d: True), (BINDING_DOMAIN, lambda d: d == BINDING_DOMAIN)):
        for rule in ("rep0", "majority"):
            vals = [x["agreement"] for x in probe_rows if x["rule"] == rule and keep(x["domain"])]
            if vals:
                summaries[f"{scope}:{rule}"] = summarise_agreements(vals, len(verdict))
        vals = [x["agreement"] for x in response_rows if keep(x["domain"])]
        if vals:
            summaries[f"{scope}:single_response_uniform_draw"] = summarise_agreements(vals, len(verdict))
    n_adm = sum(verdict.values())
    return {
        "definition": ("A route is admitted iff it answers probe p correctly (rep0: in repetition "
                       f"{rep0} only; majority: in more than {n_reps}/2 of its {n_reps} repetitions). "
                       "single_response_uniform_draw enumerates every (probe, repetition) cell, which is "
                       "the exact distribution of a single call drawn uniformly at random. Agreement = "
                       "number of routes whose predicted verdict equals the recorded one."),
        "constant_predictor_floor": {"admit_all": n_adm, "refuse_all": len(verdict) - n_adm},
        "summaries": summaries,
        "probe_rows": probe_rows,
        "response_rows": response_rows,
    }


# ---------------------------------------------------------------------------
# E14b domain alone / overall alone


def e14b(c: Campaign) -> dict:
    verdict = c.verdict
    domain_thr = c.thresholds.get(f"{BINDING_DOMAIN}_accuracy_min")
    overall_thr = c.gates.get(OVERALL)
    rows = []
    for gate in [OVERALL] + c.domains:
        thr = overall_thr if gate == OVERALL else domain_thr
        acc = {r.route: r.accuracy(gate) for r in c.routes}
        cnt = {r.route: r.gate_counts(gate) for r in c.routes}
        pred = {r: cnt[r][0] >= thr * cnt[r][1] - EPS for r in verdict}
        search = threshold_search(acc, verdict)
        rows.append({
            "gate": gate,
            "threshold": thr,
            "agreement": agreement(pred, verdict),
            "n_routes": len(verdict),
            "reproduces_verdict": agreement(pred, verdict) == len(verdict),
            "n_predicted_admitted": sum(pred.values()),
            "false_admits": ";".join(sorted(r for r in verdict if pred[r] and not verdict[r])),
            "false_refusals": ";".join(sorted(r for r in verdict if not pred[r] and verdict[r])),
            "diagnostic_only_in_protocol": gate in c.diagnostic_domains,
            "variance_across_routes": float(np.var(list(acc.values()))),
            "admitted_min": search["admitted_range"][0],
            "refused_max": search["refused_range"][1],
            "any_threshold_reproduces_verdict": search["any_threshold_reproduces_verdict"],
            "best_agreement_any_threshold": max(search["higher_admits"]["best_agreement"],
                                                search["lower_admits"]["best_agreement"]),
        })
    return {
        "definition": (f"Each domain alone: admit iff that domain's accuracy >= {domain_thr} (the recorded "
                       f"{BINDING_DOMAIN} threshold, applied to every domain). Overall alone: admit iff "
                       f"overall accuracy >= {overall_thr}. any_threshold_* searches every one-sided cut."),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# E14c parse health


def load_gameplay_parse(audit_csv: Path | None) -> dict[str, dict]:
    if audit_csv is None or not Path(audit_csv).is_file():
        return {}
    out = {}
    for row in csv.DictReader(read_text(audit_csv).splitlines()):
        out[row["route"]] = {"parse_failures": int(float(row["parse_failures"])),
                             "n_decisions_total": int(float(row["n_decisions_total"])),
                             "n_races_total": int(float(row["n_races_total"])),
                             "baseline_run_id": row.get("baseline_run_id")}
    return out


def e14c(c: Campaign, gameplay: dict[str, dict]) -> dict:
    verdict = c.verdict
    rows = []
    for r in c.routes:
        invalid = [x for x in r.rows if x["semantic_valid"] is not True]
        transport = [x for x in r.rows if x.get("transport_errors")]
        g = gameplay.get(r.route)
        rows.append({
            "route": r.route,
            "admitted": r.admitted,
            "probe_rows": r.n,
            "semantic_valid": r.n - len(invalid),
            "semantic_invalid": len(invalid),
            "semantic_valid_rate": (r.n - len(invalid)) / r.n,
            "invalid_by_domain": ";".join(f"{d}:{k}" for d, k in sorted(Counter(x["domain"] for x in invalid).items())),
            "transport_error_rows": len(transport),
            "gameplay_parse_failures": None if g is None else g["parse_failures"],
            "gameplay_decisions": None if g is None else g["n_decisions_total"],
            "gameplay_parse_failure_rate": None if g is None else g["parse_failures"] / g["n_decisions_total"],
        })
    probe_valid = threshold_search({x["route"]: x["semantic_valid_rate"] for x in rows}, verdict)
    transport = threshold_search({x["route"]: float(x["transport_error_rows"]) for x in rows}, verdict)
    game = threshold_search({x["route"]: (math.nan if x["gameplay_parse_failure_rate"] is None
                                          else x["gameplay_parse_failure_rate"]) for x in rows}, verdict)
    totals = {
        "probe_rows": sum(x["probe_rows"] for x in rows),
        "semantic_invalid": sum(x["semantic_invalid"] for x in rows),
        "transport_error_rows": sum(x["transport_error_rows"] for x in rows),
        "gameplay_parse_failures": (None if any(x["gameplay_parse_failures"] is None for x in rows)
                                    else sum(x["gameplay_parse_failures"] for x in rows)),
        "gameplay_decisions": (None if any(x["gameplay_decisions"] is None for x in rows)
                               else sum(x["gameplay_decisions"] for x in rows)),
    }
    return {
        "definition": ("semantic_valid_rate = share of a route's probe rows with semantic_valid true. "
                       "gameplay_parse_failure_rate = parse_failures / n_decisions_total from the "
                       "baseline audit CSV. For each, every one-sided threshold in both directions is "
                       "tried against the recorded verdict."),
        "rows": rows,
        "totals": totals,
        "threshold_search": {"semantic_valid_rate": probe_valid,
                             "transport_error_rows": transport,
                             "gameplay_parse_failure_rate": game},
    }


# ---------------------------------------------------------------------------
# E13 name baseline


def name_words(route: str) -> set[str]:
    return {t for t in re.split(r"[^a-z]+", route.lower()) if t}


def exact_permutation(pred_admit: dict[str, bool], verdict: dict[str, bool]) -> dict:
    routes = sorted(verdict)
    k = sum(verdict.values())
    observed = agreement(pred_admit, verdict)
    total = at_least = 0
    for admitted in itertools.combinations(routes, k):
        s = set(admitted)
        total += 1
        if agreement(pred_admit, {r: r in s for r in routes}) >= observed:
            at_least += 1
    return {"observed_agreement": observed, "labellings": total,
            "labellings_at_least_as_good": at_least, "p_one_sided": at_least / total}


def e13(c: Campaign) -> dict:
    verdict = c.verdict
    rules = {
        "substring_regex": lambda r: re.search(r"(mini|nano|lite)", r.lower()) is not None,
        "whole_word": lambda r: any(w in name_words(r) for w in SIZE_WORDS),
    }
    out = {"definition": ("Predict refused iff the route string matches the size-word rule. "
                          "substring_regex: re.search('(mini|nano|lite)') on the lower-cased route. "
                          "whole_word: the same words as whole tokens after splitting the route on "
                          "non-letters (so 'gemini' is not 'mini'). p_one_sided = share of all "
                          f"C({len(verdict)},{sum(verdict.values())}) labellings with the recorded "
                          "number admitted whose agreement is at least the observed one."),
           "rules": {}}
    rows = []
    for name, rule in rules.items():
        flagged = {r: rule(r) for r in verdict}
        pred = {r: not f for r, f in flagged.items()}
        perm = exact_permutation(pred, verdict)
        out["rules"][name] = {**perm, "n_flagged": sum(flagged.values()),
                              "disagreements": sorted(r for r in verdict if pred[r] != verdict[r])}
        for r in sorted(verdict):
            rows.append({"rule": name, "route": r, "flagged_small": flagged[r],
                         "predicted_admitted": pred[r], "recorded_admitted": verdict[r],
                         "agrees": pred[r] == verdict[r]})
    out["rows"] = rows
    out["n_labellings"] = math.comb(len(verdict), sum(verdict.values()))
    return out


# ---------------------------------------------------------------------------
# E12 threshold band


def admitted_under(c: Campaign, thresholds: dict[str, float]) -> frozenset[str]:
    return frozenset(r.route for r in c.routes
                     if all(r.gate_counts(g)[0] >= t * r.gate_counts(g)[1] - EPS
                            for g, t in thresholds.items()))


def e12(c: Campaign) -> dict:
    declared = admitted_under(c, c.gates)
    if declared != frozenset(r for r, v in c.verdict.items() if v):
        fail("the declared gate does not reproduce the recorded admitted set")
    gates = list(c.gates)
    steps = int(round(BAND_HALF_WIDTH / BAND_STEP))
    axes = {g: [round(c.gates[g] + k * BAND_STEP, 2) for k in range(-steps, steps + 1)] for g in gates}
    grid_rows, sets = [], Counter()
    for combo in itertools.product(*(axes[g] for g in gates)):
        thr = dict(zip(gates, combo))
        s = admitted_under(c, thr)
        sets[s] += 1
        grid_rows.append({**{f"thr_{g}": thr[g] for g in gates},
                          "n_admitted": len(s), "identical_to_verdict": s == declared,
                          "admitted_set": ";".join(sorted(s))})
    per_gate_unchanged_pct = {}
    for g in gates:
        keep = [v for v in axes[g] if admitted_under(c, {**c.gates, g: v}) == declared]
        per_gate_unchanged_pct[g] = keep

    rows_per_gate = {g: c.routes[0].gate_counts(g)[1] for g in gates}
    levels_unchanged = {}
    for g in gates:
        n = rows_per_gate[g]
        levels_unchanged[g] = [k for k in range(n + 1)
                               if admitted_under(c, {**c.gates, g: k / n}) == declared]
    box = list(itertools.product(*(levels_unchanged[g] for g in gates)))
    box_ok = sum(1 for t in box
                 if admitted_under(c, {g: k / rows_per_gate[g] for g, k in zip(gates, t)}) == declared)
    near = {g: [k for k in range(rows_per_gate[g] + 1)
                if abs(k / rows_per_gate[g] - c.gates[g]) <= BAND_HALF_WIDTH + EPS] for g in gates}
    near_box = list(itertools.product(*(near[g] for g in gates)))
    near_ok = sum(1 for t in near_box
                  if admitted_under(c, {g: k / rows_per_gate[g] for g, k in zip(gates, t)}) == declared)
    return {
        "definition": (f"Admit iff every gate clears its threshold (correct >= threshold x rows). "
                       f"Grid A: each gate's recorded threshold +/- {BAND_HALF_WIDTH} in steps of "
                       f"{BAND_STEP}, all combinations. Grid B reproduces scripts/verify_manuscript_claims.py: "
                       "a threshold is carried as a whole number of correct answers; for each gate separately, "
                       "with the other gates at their declared values, keep every achievable level whose "
                       "admitted set equals the verdict; the box is the product of those per-gate level sets. "
                       "Grid C: every combination of achievable levels within +/- 5 points of each declared "
                       "threshold."),
        "gates": c.gates,
        "grid_a_decimal": {
            "axes": axes,
            "n_combinations": len(grid_rows),
            "n_identical": sum(1 for x in grid_rows if x["identical_to_verdict"]),
            "per_gate_values_keeping_verdict_others_declared": per_gate_unchanged_pct,
            "distinct_admitted_sets": [{"admitted": sorted(s), "n_combinations": k}
                                       for s, k in sets.most_common()],
        },
        "grid_b_integer_box": {
            "rows_per_gate": rows_per_gate,
            "levels_keeping_verdict": {g: v for g, v in levels_unchanged.items()},
            "levels_count": {g: len(v) for g, v in levels_unchanged.items()},
            "levels_pct_range": {g: [100 * min(v) / rows_per_gate[g], 100 * max(v) / rows_per_gate[g]]
                                 for g, v in levels_unchanged.items()},
            "n_combinations": len(box),
            "n_identical": box_ok,
            "reproduces_676": len(box) == 676 and box_ok == 676,
        },
        "grid_c_integer_within_5_points": {
            "levels": near,
            "levels_pct": {g: [round(100 * k / rows_per_gate[g], 2) for k in v] for g, v in near.items()},
            "n_combinations": len(near_box),
            "n_identical": near_ok,
        },
        "grid_rows": grid_rows,
    }


# ---------------------------------------------------------------------------
# repetition reliability


def reliability(c: Campaign) -> dict:
    n_reps = len(c.repetitions)
    rows = []
    pooled = defaultdict(list)
    for r in c.routes:
        by_probe = defaultdict(list)
        for x in r.rows:
            by_probe[x["probe_id"]].append(x)
        for scope in c.domains + ["ALL"]:
            probes = [p for p in c.probes if scope == "ALL" or c.probe_domain[p] == scope]
            counts = [sum(1 for x in by_probe[p] if x["semantic_correct"] is True) for p in probes]
            same_answer = [len({str(x["answer"]).strip() for x in by_probe[p]}) == 1 for p in probes]
            all_agree = [k in (0, n_reps) for k in counts]
            if scope != "ALL":
                pooled[scope].extend(counts)
            rows.append({
                "route": r.route, "domain": scope, "n_probes": len(probes), "repetitions": n_reps,
                "accuracy": sum(counts) / (len(probes) * n_reps),
                "n_all_reps_agree_correctness": sum(all_agree),
                "prop_all_reps_agree_correctness": sum(all_agree) / len(probes),
                "n_identical_answer_string": sum(same_answer),
                "prop_identical_answer_string": sum(same_answer) / len(probes),
                "fleiss_kappa": fleiss_binary(counts, n_reps),
            })
    for scope, counts in pooled.items():
        rows.append({
            "route": "ALL", "domain": scope, "n_probes": len(counts), "repetitions": n_reps,
            "accuracy": sum(counts) / (len(counts) * n_reps),
            "n_all_reps_agree_correctness": sum(1 for k in counts if k in (0, n_reps)),
            "prop_all_reps_agree_correctness": sum(1 for k in counts if k in (0, n_reps)) / len(counts),
            "n_identical_answer_string": None, "prop_identical_answer_string": None,
            "fleiss_kappa": fleiss_binary(counts, n_reps),
        })
    all_counts = [k for v in pooled.values() for k in v]
    temps = {r.route: r.manifest.get("decoding", {}).get("temperature_requested") for r in c.routes}
    reasoning = {r.route: r.manifest.get("decoding", {}).get("reasoning_requested") for r in c.routes}
    return {
        "definition": ("Per route and domain: items are probes, raters are the repetitions, the rating is "
                       "semantic_correct. prop_all_reps_agree_correctness = share of probes whose "
                       "repetitions are all correct or all wrong. Fleiss' kappa on the binary rating; "
                       "undefined (n/a) when every rating in the cell is the same category, which is the "
                       "common case at ceiling or floor. prop_identical_answer_string = share of probes whose "
                       "answer string is identical in every repetition."),
        "temperature_requested": temps,
        "reasoning_requested": reasoning,
        "pooled_all": {
            "n_route_probe_items": len(all_counts),
            "n_all_reps_agree": sum(1 for k in all_counts if k in (0, n_reps)),
            "prop_all_reps_agree": sum(1 for k in all_counts if k in (0, n_reps)) / len(all_counts),
            "fleiss_kappa": fleiss_binary(all_counts, n_reps),
        },
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# E21 opening-move seat gap from raw baseline turns


def e21(baseline_root: Path | None, target: str, n_boot: int, seed: int,
        existing_csv: Path | None) -> dict:
    if baseline_root is None or not Path(baseline_root).is_dir():
        return {"computable": False, "reason": f"baseline root {baseline_root} not found"}
    root = Path(baseline_root).resolve()
    if FAILED in root.parts:
        fail("baseline root lies inside a failure record")
    turns_files, pruned = walk_live(root, "turns.jsonl")
    per_route = {}
    for tf in turns_files:
        man_path = tf.with_name("run_manifest.json")
        if not man_path.is_file():
            fail(f"{_rel(tf)} has no run_manifest.json")
        man = json.loads(read_text(man_path))
        route = man.get("model_route")
        if man.get("status") != "completed" or man.get("run_phase") != "confirmatory":
            fail(f"{route}: baseline manifest is {man.get('status')}/{man.get('run_phase')}")
        if route in per_route:
            fail(f"{route}: two baseline runs under {root}; refusing to pick one")
        turns = [json.loads(line) for line in read_text(tf).splitlines() if line.strip()]
        if len(turns) != man.get("n_turns"):
            fail(f"{route}: {len(turns)} turns, manifest says {man.get('n_turns')}")
        contaminated = {t["game_id"] for t in turns if t.get("parse_failed")}
        opening = defaultdict(dict)
        for t in turns:
            if t["round"] == 1:
                if t["player_index"] in opening[t["game_id"]]:
                    fail(f"{route}: duplicate opening decision in {t['game_id']}")
                opening[t["game_id"]][t["player_index"]] = int(t["unsafe"])
        if len(opening) != man.get("n_races"):
            fail(f"{route}: {len(opening)} races with an opening move, manifest says {man.get('n_races')}")
        if any(set(v) != {0, 1} for v in opening.values()):
            fail(f"{route}: a race lacks exactly two seats at round 1")
        games = sorted(g for g in opening if g not in contaminated)
        u1 = np.array([opening[g][0] for g in games], float)
        u2 = np.array([opening[g][1] for g in games], float)
        d = u2 - u1
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, len(d), size=(n_boot, len(d)))
        boot = d[draws].mean(axis=1)
        per_route[route] = {
            "route": route,
            "baseline_run_dir": _rel(tf.parents[2]),
            "races": len(games),
            "races_excluded_parse_failure": len(contaminated),
            "seat1_opening_unsafe_rate": float(u1.mean()),
            "seat2_opening_unsafe_rate": float(u2.mean()),
            "gap_pp": float(100 * d.mean()),
            "ci_low_pp": float(100 * np.percentile(boot, 2.5)),
            "ci_high_pp": float(100 * np.percentile(boot, 97.5)),
            "races_seat2_higher": int((d > 0).sum()),
            "races_seat1_higher": int((d < 0).sum()),
            "races_tied": int((d == 0).sum()),
        }
    existing = None
    if existing_csv is not None and Path(existing_csv).is_file():
        for row in csv.DictReader(read_text(existing_csv).splitlines()):
            if row["scope"] == "opening:route" and row["model_route"] == target:
                existing = {"file": _rel(existing_csv), "gap_race_pp": float(row["gap_race_pp"]),
                            "gap_race_lo": float(row["gap_race_lo"]), "gap_race_hi": float(row["gap_race_hi"]),
                            "races": int(row["races"]),
                            "note": "generated by scripts/analyze_seat_confound.py with its own "
                                    "N_BOOT=20000 and SEED=20260911"}
    if target not in per_route:
        return {"computable": False, "reason": f"{target} has no baseline turns under {_rel(root)}",
                "pruned_failed_dirs": pruned}
    return {
        "computable": True,
        "definition": ("Opening move = round 1. Per race d = seat-2 unsafe (0/1) - seat-1 unsafe (0/1); "
                       "gap = mean d in points. Races resampled with replacement, "
                       f"{n_boot} draws, numpy default_rng({seed}) re-seeded per route, percentile 95% "
                       "interval. Races containing any parse_failed decision are excluded (none here)."),
        "target_route": target,
        "target": per_route[target],
        "all_routes": [per_route[r] for r in sorted(per_route)],
        "existing_seat_confound_row": existing,
        "n_boot": n_boot,
        "seed": seed,
        "pruned_failed_dirs": pruned,
    }


# ---------------------------------------------------------------------------
# outputs


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        for r in rows:
            w.writerow({k: ("" if v is None else v) for k, v in r.items()})


def route_table(c: Campaign, parse: dict) -> list[dict]:
    health = {x["route"]: x for x in parse["rows"]}
    order = sorted(c.routes, key=lambda r: (not r.admitted, -r.overall, r.route))
    rows = []
    for r in order:
        row = {"route": r.route, "admitted": r.admitted, "overall_accuracy": r.overall,
               "overall_correct": r.correct, "overall_rows": r.n}
        for d in c.domains:
            row[f"{d}_accuracy"] = r.accuracy(d)
            row[f"{d}_correct"] = r.domains[d]["correct"]
            row[f"{d}_rows"] = r.domains[d]["rows"]
        row["semantic_valid_rate"] = health[r.route]["semantic_valid_rate"]
        row["gameplay_parse_failures"] = health[r.route]["gameplay_parse_failures"]
        row["whole_word_size_flag"] = any(w in name_words(r.route) for w in SIZE_WORDS)
        row["task_dir"] = _rel(r.raw_path.parent)
        rows.append(row)
    return rows


def _agree_line(s: dict, n: int) -> str:
    return (f"min {s['min']}/{n}, median {s['median']}/{n}, max {s['max']}/{n}, mean {s['mean']:.2f}; "
            f"{s['n_exact']} of {s['n']} reproduce the verdict exactly")


def report_md(out: dict, c: Campaign) -> str:
    n = len(c.routes)
    n_adm = sum(c.verdict.values())
    a, b, pc, nm, band, rel, s21 = (out["e14a"], out["e14b"], out["e14c"], out["e13"], out["e12"],
                                    out["reliability"], out["e21"])
    L = []
    L.append("# Cheap screen baselines against the admission verdict")
    L.append("")
    L.append(f"Generated {out['generated_utc']} by `scripts/arr/analyze_screen_baselines.py`. "
             "Every number below is written by the script from the files listed under Inputs; "
             "none is typed by hand. Regenerate, never edit.")
    L.append("")
    L.append("## Scope and checks that passed")
    L.append("")
    L.append(f"- Family `{c.family}`, protocol `{c.protocol_id}`, campaign roots: "
             + ", ".join(f"`{r}`" for r in c.roots) + ".")
    L.append(f"- {n} routes, {len(c.probes)} probes x {len(c.repetitions)} repetitions = "
             f"{out['campaign']['expected_rows']} rows per route, {sum(r.n for r in c.routes)} rows in total. "
             f"Recorded verdict: {n_adm} admitted, {n - n_adm} refused.")
    L.append(f"- One probe bank on every route: `probe_bank_sha256 = {c.probe_bank_sha256}`; "
             f"one rules context: `{c.rules_context_sha256}`.")
    L.append(f"- Per-domain counts and the verdict were recomputed from the raw rows and match every "
             f"`admission.json`. Gates: " + ", ".join(f"{g} >= {t}" for g, t in c.gates.items())
             + f"; diagnostic only: {', '.join(c.diagnostic_domains) or 'none'}.")
    L.append(f"- `failed_runs/` directories pruned without being opened: admission "
             f"{c.pruned_failed_dirs}, baseline {s21.get('pruned_failed_dirs', 'n/a')}. "
             f"No file under `failed_runs/` was read (the reader refuses such a path).")
    L.append("- Decoding: admission manifests request temperature "
             + ", ".join(sorted({str(v) for v in rel['temperature_requested'].values()}))
             + "; reasoning requested "
             + ", ".join(f"`{k}` on {v} route(s)" for k, v in sorted(Counter(map(str, rel['reasoning_requested'].values())).items()))
             + ". Routes whose manifest records no reasoning value (argument omitted): "
             + (", ".join(r for r, v in sorted(rel['reasoning_requested'].items()) if v is None) or "none") + ".")
    L.append("")
    L.append("Agreement always means: the number of the "
             f"{n} routes whose predicted verdict equals the recorded one. A constant predictor already gets "
             f"{a['constant_predictor_floor']['admit_all']}/{n} (admit all) or "
             f"{a['constant_predictor_floor']['refuse_all']}/{n} (refuse all).")
    L.append("")

    L.append("## E14a Single-probe baseline")
    L.append("")
    L.append(a["definition"])
    L.append("")
    L.append("| scope | rule | agreement distribution |")
    L.append("|---|---|---|")
    for key, s in a["summaries"].items():
        scope, rule = key.split(":")
        L.append(f"| {scope} | {rule} | {_agree_line(s, n)} |")
    L.append("")
    L.append("Per probe (rep0 / majority):")
    L.append("")
    L.append("| probe | domain | rep0 | majority |")
    L.append("|---|---|---|---|")
    by = defaultdict(dict)
    for x in a["probe_rows"]:
        by[x["probe_id"]][x["rule"]] = x
    for p in c.probes:
        L.append(f"| `{p}` | {c.probe_domain[p]} | {by[p]['rep0']['agreement']}/{n} | "
                 f"{by[p]['majority']['agreement']}/{n} |")
    exact = sorted({x["probe_id"] for x in a["probe_rows"] if x["reproduces_verdict"]})
    L.append("")
    L.append("Probes that reproduce the verdict exactly under at least one rule: "
             + (", ".join(f"`{p}`" for p in exact) if exact else "none") + ".")
    L.append("")

    L.append("## E14b Each domain alone, overall alone")
    L.append("")
    L.append(b["definition"])
    L.append("")
    L.append("| gate | threshold | agreement | predicted admitted | false admits | false refusals | "
             "admitted min | refused max | any threshold exact? | best any threshold |")
    L.append("|---|---|---|---|---|---|---|---|---|---|")
    for x in b["rows"]:
        L.append(f"| {x['gate']}{' (diagnostic)' if x['diagnostic_only_in_protocol'] else ''} | {x['threshold']} | "
                 f"{x['agreement']}/{n} | {x['n_predicted_admitted']} | {x['false_admits'] or '-'} | "
                 f"{x['false_refusals'] or '-'} | {fmt(x['admitted_min'])} | {fmt(x['refused_max'])} | "
                 f"{'yes' if x['any_threshold_reproduces_verdict'] else 'no'} | {x['best_agreement_any_threshold']}/{n} |")
    L.append("")

    L.append("## E14c Parse-health baseline")
    L.append("")
    L.append(pc["definition"])
    L.append("")
    L.append("| route | admitted | valid / rows | invalid by domain | transport-error rows | gameplay parse failures / decisions |")
    L.append("|---|---|---|---|---|---|")
    for x in pc["rows"]:
        g = ("n/a" if x["gameplay_parse_failures"] is None
             else f"{x['gameplay_parse_failures']}/{x['gameplay_decisions']}")
        L.append(f"| {x['route']} | {x['admitted']} | {x['semantic_valid']}/{x['probe_rows']} | "
                 f"{x['invalid_by_domain'] or '-'} | {x['transport_error_rows']} | {g} |")
    t = pc["totals"]
    L.append("")
    L.append(f"Totals: {t['probe_rows'] - t['semantic_invalid']} of {t['probe_rows']} probe rows valid, "
             f"{t['semantic_invalid']} invalid; {t['transport_error_rows']} row(s) with a transport error; "
             f"gameplay {t['gameplay_parse_failures']} parse failures in {t['gameplay_decisions']} decisions.")
    L.append("")
    for key, s in pc["threshold_search"].items():
        if not s["available"]:
            L.append(f"- {key}: not available for {', '.join(s['missing_routes'])}.")
            continue
        L.append(f"- {key}: {s['n_distinct_values']} distinct value(s); best agreement "
                 f"{s['higher_admits']['best_agreement']}/{n} if higher admits, "
                 f"{s['lower_admits']['best_agreement']}/{n} if lower admits; any threshold reproduces "
                 f"the verdict: {'yes' if s['any_threshold_reproduces_verdict'] else 'no'}. "
                 f"Admitted range {s['admitted_range']}, refused range {s['refused_range']}.")
    L.append("")

    L.append("## E13 Name baseline")
    L.append("")
    L.append(nm["definition"])
    L.append("")
    L.append("| rule | routes flagged | agreement | disagreements | labellings at least as good | exact one-sided p |")
    L.append("|---|---|---|---|---|---|")
    for name, s in nm["rules"].items():
        L.append(f"| {name} | {s['n_flagged']} | {s['observed_agreement']}/{n} | "
                 f"{', '.join(s['disagreements']) or '-'} | {s['labellings_at_least_as_good']}/{s['labellings']} | "
                 f"{s['p_one_sided']:.4f} |")
    L.append("")
    only_sub = sorted({x["route"] for x in nm["rows"] if x["rule"] == "substring_regex" and x["flagged_small"]}
                      - {x["route"] for x in nm["rows"] if x["rule"] == "whole_word" and x["flagged_small"]})
    if only_sub:
        L.append("The literal regex `(mini|nano|lite)` also flags " + ", ".join(only_sub)
                 + " (it matches inside a word such as `gemini`), so the perfect-agreement figure belongs to "
                 "the whole-word rule, not to the literal regex.")
    L.append("")

    L.append("## E12 Threshold band")
    L.append("")
    L.append(band["definition"])
    L.append("")
    ga, gb, gc = band["grid_a_decimal"], band["grid_b_integer_box"], band["grid_c_integer_within_5_points"]
    L.append(f"- Grid A (decimal, " + " x ".join(f"{len(v)}" for v in ga["axes"].values())
             + f"): {ga['n_identical']} of {ga['n_combinations']} combinations give the recorded admitted set.")
    for g, vals in ga["per_gate_values_keeping_verdict_others_declared"].items():
        L.append(f"  - {g}: verdict unchanged at {len(vals)} of {len(ga['axes'][g])} values "
                 f"({min(vals) if vals else 'none'} to {max(vals) if vals else 'none'}), others at declared.")
    for s in ga["distinct_admitted_sets"]:
        L.append(f"  - {s['n_combinations']} combinations admit {len(s['admitted'])}: {', '.join(s['admitted'])}")
    L.append(f"- Grid B (integer box, the verify_manuscript_claims.py construction): per-gate level counts "
             + " x ".join(f"{g} {k}" for g, k in gb["levels_count"].items())
             + f" = {gb['n_combinations']} combinations, {gb['n_identical']} identical. "
             + "Percent ranges kept: "
             + "; ".join(f"{g} {lo:.1f} to {hi:.1f}" for g, (lo, hi) in gb["levels_pct_range"].items())
             + f". Reproduces 676: {'yes' if gb['reproduces_676'] else 'no'}.")
    L.append(f"- Grid C (achievable levels within 5 points of each declared threshold, levels in percent "
             + "; ".join(f"{g} {v}" for g, v in gc["levels_pct"].items())
             + f"): {gc['n_identical']} of {gc['n_combinations']} identical.")
    L.append("")
    if gb["reproduces_676"]:
        L.append("676 is not 26 x 26. It is the product of the per-gate achievable levels at which the verdict is "
                 "unchanged with the other two gates held at their declared values, so the phrase \"within five "
                 "points on each condition\" in the plan misdescribes it: the overall axis of that box runs from "
                 "0/60 upward, not from 75%, and the state-reconstruction axis holds a single level.")
    else:
        L.append(f"The integer box has {gb['n_combinations']} combinations here, not the 676 the plan cites.")
    L.append("")

    L.append("## Repetition reliability")
    L.append("")
    L.append(rel["definition"])
    L.append("")
    p = rel["pooled_all"]
    L.append(f"Pooled over every route and probe: {p['n_all_reps_agree']} of {p['n_route_probe_items']} "
             f"route-probe items have all repetitions agree on correctness ({pct(p['prop_all_reps_agree'])}%), "
             f"Fleiss kappa {fmt(p['fleiss_kappa'])}.")
    L.append("")
    L.append("| route | domain | probes | accuracy | all reps agree | identical answer string | Fleiss kappa |")
    L.append("|---|---|---|---|---|---|---|")
    for x in rel["rows"]:
        ident = ("-" if x["n_identical_answer_string"] is None
                 else f"{x['n_identical_answer_string']}/{x['n_probes']}")
        L.append(f"| {x['route']} | {x['domain']} | {x['n_probes']} | {pct(x['accuracy'])} | "
                 f"{x['n_all_reps_agree_correctness']}/{x['n_probes']} | {ident} | {fmt(x['fleiss_kappa'])} |")
    L.append("")

    L.append("## E21 Per-route opening-move seat gap")
    L.append("")
    if not s21["computable"]:
        L.append(f"Not computable: {s21['reason']}.")
    else:
        t = s21["target"]
        L.append(s21["definition"])
        L.append("")
        L.append(f"**{t['route']}**: seat 1 opening Unsafe rate {t['seat1_opening_unsafe_rate']:.3f}, "
                 f"seat 2 {t['seat2_opening_unsafe_rate']:.3f}, gap {t['gap_pp']:.2f} points "
                 f"[{t['ci_low_pp']:.2f}, {t['ci_high_pp']:.2f}] over {t['races']} races "
                 f"({t['races_seat2_higher']} races seat 2 higher, {t['races_seat1_higher']} seat 1 higher, "
                 f"{t['races_tied']} tied), from `{t['baseline_run_dir']}`.")
        e = s21["existing_seat_confound_row"]
        if e:
            L.append("")
            L.append(f"The per-route opening row does exist on disk: `{e['file']}`, scope `opening:route`, "
                     f"gap {e['gap_race_pp']:.2f} [{e['gap_race_lo']:.2f}, {e['gap_race_hi']:.2f}] over "
                     f"{e['races']} races ({e['note']}). Section 3.6 and 4.2 of the plan, which say no per-route "
                     "opening row exists, are stale on this point.")
        L.append("")
        L.append("| route | races | seat 1 | seat 2 | gap (pp) | 95% interval |")
        L.append("|---|---|---|---|---|---|")
        for x in s21["all_routes"]:
            L.append(f"| {x['route']} | {x['races']} | {x['seat1_opening_unsafe_rate']:.3f} | "
                     f"{x['seat2_opening_unsafe_rate']:.3f} | {x['gap_pp']:.2f} | "
                     f"[{x['ci_low_pp']:.2f}, {x['ci_high_pp']:.2f}] |")
    L.append("")
    L.append("## Inputs")
    L.append("")
    for f in out["inputs"]:
        L.append(f"- `{f['path']}` sha256 `{f['sha256']}`")
    L.append("")
    return "\n".join(L)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--campaign-root", action="append", type=Path,
                    help="admission campaign directory; repeat to add routes (default: admission_campaign_v6)")
    ap.add_argument("--family-label", default=LEGACY_FAMILY,
                    help="task family to analyse; rows without a 'family' field are F1")
    ap.add_argument("--expected-rows", type=int, default=60)
    ap.add_argument("--baseline-root", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--audit-csv", type=Path, default=DEFAULT_AUDIT_CSV)
    ap.add_argument("--seat-confound-csv", type=Path, default=DEFAULT_SEAT_LOO)
    ap.add_argument("--e21-route", default=DEFAULT_E21_ROUTE)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--output", type=Path, default=None)
    args = ap.parse_args(argv)

    roots = args.campaign_root or [DEFAULT_CAMPAIGN]
    out_dir = args.output or (DEFAULT_OUT if args.family_label == LEGACY_FAMILY
                              else DEFAULT_OUT / args.family_label)
    c = load_campaign(roots, args.family_label, args.expected_rows)
    gameplay = load_gameplay_parse(args.audit_csv) if args.family_label == LEGACY_FAMILY else {}

    res = {
        "generator": "scripts/arr/analyze_screen_baselines.py",
        "generated_utc": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "campaign": {
            "family": c.family, "roots": c.roots, "protocol_id": c.protocol_id,
            "probe_bank_sha256": c.probe_bank_sha256, "rules_context_sha256": c.rules_context_sha256,
            "expected_rows": args.expected_rows, "probes": c.probes, "probe_domain": c.probe_domain,
            "repetitions": c.repetitions, "domains": c.domains, "thresholds": c.thresholds,
            "gates": c.gates, "diagnostic_domains": c.diagnostic_domains,
            "verdict": c.verdict, "pruned_failed_dirs": c.pruned_failed_dirs,
        },
    }
    res["e14a"] = e14a(c)
    res["e14b"] = e14b(c)
    res["e14c"] = e14c(c, gameplay)
    res["e13"] = e13(c)
    res["e12"] = e12(c)
    res["reliability"] = reliability(c)
    res["e21"] = (e21(args.baseline_root, args.e21_route, args.n_boot, args.seed, args.seat_confound_csv)
                  if args.family_label == LEGACY_FAMILY
                  else {"computable": False, "reason": "gameplay turns exist only for the race family F1"})
    if any(FAILED in Path(f["path"]).parts for f in READ_LOG):
        fail("a failed_runs file was read")
    res["inputs"] = sorted({f["path"]: f for f in READ_LOG}.values(), key=lambda f: f["path"])

    out_dir.mkdir(parents=True, exist_ok=True)
    write_csv(out_dir / "route_table.csv", route_table(c, res["e14c"]))
    write_csv(out_dir / "e14a_single_probe.csv", res["e14a"]["probe_rows"])
    write_csv(out_dir / "e14a_single_response.csv", res["e14a"]["response_rows"])
    write_csv(out_dir / "e14b_domain_alone.csv", res["e14b"]["rows"])
    write_csv(out_dir / "e14c_parse_health.csv", res["e14c"]["rows"])
    write_csv(out_dir / "e13_name_baseline.csv", res["e13"]["rows"])
    write_csv(out_dir / "e12_threshold_grid.csv", res["e12"]["grid_rows"])
    write_csv(out_dir / "reliability.csv", res["reliability"]["rows"])
    if res["e21"].get("computable"):
        write_csv(out_dir / "e21_opening_seat_gap.csv", res["e21"]["all_routes"])
    slim = json.loads(json.dumps(res, default=str))
    slim["e12"].pop("grid_rows")
    slim["e14a"].pop("response_rows")
    (out_dir / "baselines.json").write_text(json.dumps(slim, indent=2), encoding="utf-8")
    (out_dir / "report.md").write_text(report_md(res, c), encoding="utf-8")
    print(f"wrote {_rel(out_dir)}: {len(c.routes)} routes, family {c.family}")


if __name__ == "__main__":
    main()
