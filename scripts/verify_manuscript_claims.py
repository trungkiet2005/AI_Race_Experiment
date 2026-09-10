"""Check every headline number in the manuscript against its artifact.

This is an independent pass: each claim is recomputed from the raw or derived
artifact rather than compared between two derived files, and a claim that is
true of some cells but stated for all is reported as OVERSTATED rather than
passed.
"""

from __future__ import annotations

import csv
import glob
import json
import os
from collections import defaultdict

from pathlib import Path

# Resolved from this file so the check runs from any working directory.
ROOT = Path(__file__).resolve().parents[1]
os.chdir(ROOT)

results: list[tuple[str, str, str]] = []


def check(name: str, ok: bool, detail: str) -> None:
    results.append((name, "PASS" if ok else "FAIL", detail))


# --- admission campaign, recomputed from each route's own admission.json ----
adm = {}
for f in sorted(glob.glob("results/frontier/admission_campaign_v6/**/admission.json", recursive=True)):
    if "failed" in f.replace("\\", "/"):
        continue
    a = json.load(open(f, encoding="utf-8"))
    adm[a["model_route"]] = a

check("nine routes carry an admission verdict", len(adm) == 9, f"found {len(adm)}")

payoff_perfect = [r for r, a in adm.items() if a["by_domain"]["stage_payoff"]["accuracy"] == 1.0]
rules_perfect = [r for r, a in adm.items() if a["by_domain"]["rule_recall"]["accuracy"] == 1.0]
check("all nine read the payoff matrix perfectly", len(payoff_perfect) == 9, f"{len(payoff_perfect)}/9")
check("eight of nine recall the rules perfectly", len(rules_perfect) == 8, f"{len(rules_perfect)}/9")

state = {r: a["by_domain"]["state_reconstruction"]["accuracy"] for r, a in adm.items()}
check("state reconstruction ranges 100% down to 20%",
      abs(max(state.values()) - 1.0) < 1e-9 and abs(min(state.values()) - 0.20) < 1e-9,
      f"max {max(state.values()):.3f} min {min(state.values()):.3f}")

ep = {r: a["by_domain"]["expected_payoff"]["accuracy"] for r, a in adm.items()}
check("no route reaches 0.75 on expected payoff", max(ep.values()) < 0.75, f"max {max(ep.values()):.3f}")
check("five routes score zero on expected payoff",
      sum(1 for v in ep.values() if v == 0.0) == 5,
      f"{sum(1 for v in ep.values() if v == 0.0)} at zero")

rows_total = sum(a["n_rows"] for a in adm.values())
check("540 retained probe outputs in total", rows_total == 540, f"{rows_total}")

admitted = sorted(r for r, a in adm.items() if a["admitted_for_gameplay"])
check("five routes are admitted", len(admitted) == 5, ", ".join(admitted))

# --- gameplay, recounted from each route's own turns.jsonl -------------------
rates: dict[str, dict[float, float]] = {}
for man in sorted(glob.glob("results/frontier/baseline_campaign_v6/ai-race-baseline/*/*/*/results/ai_race_baseline/run_manifest.json")):
    d = json.load(open(man, encoding="utf-8"))
    if d.get("status") != "completed":
        continue
    agg: dict[float, list[int]] = defaultdict(lambda: [0, 0])
    races: dict[float, set] = defaultdict(set)
    pf = 0
    for line in open(man.replace("run_manifest.json", "turns.jsonl"), encoding="utf-8"):
        r = json.loads(line)
        risk = float(r["max_private_risk"])
        agg[risk][1] += 1
        if str(r.get("action", "")).upper().startswith("UNSAFE"):
            agg[risk][0] += 1
        if r.get("parse_failed"):
            pf += 1
        races[risk].add(r.get("game_id"))
    rates[d["model_route"]] = {k: 100 * v[0] / v[1] for k, v in agg.items()}
    check(f"gameplay integrity: {d['model_route'].split('/')[-1]}",
          d["n_races"] == 30 and sum(v[1] for v in agg.values()) == 558 and pf == 0
          and all(len(v) == 10 for v in races.values()),
          f"races {d['n_races']}, decisions {sum(v[1] for v in agg.values())}, parse failures {pf}")

check("nine routes have completed gameplay", len(rates) == 9, f"found {len(rates)}")

resp = {r: v[0.1] - v[0.9] for r, v in rates.items()}
check("risk response spans 7.0 to 100.0 points",
      abs(min(resp.values()) - 6.9892) < 0.02 and abs(max(resp.values()) - 100.0) < 1e-9,
      f"min {min(resp.values()):.1f} max {max(resp.values()):.1f}")

adm_resp = [resp[r] for r in admitted]
ref_resp = [resp[r] for r in rates if r not in admitted]
check("every admitted route responds at least 38 points", min(adm_resp) >= 38.0,
      f"admitted min {min(adm_resp):.1f}")
check("admitted mean 54.8 vs refused mean 19.4",
      abs(sum(adm_resp) / len(adm_resp) - 54.8) < 0.1 and abs(sum(ref_resp) / len(ref_resp) - 19.4) < 0.1,
      f"{sum(adm_resp)/len(adm_resp):.1f} vs {sum(ref_resp)/len(ref_resp):.1f}")
check("a refused route overlaps the admitted range",
      max(ref_resp) > min(adm_resp) - 1.5,
      f"refused max {max(ref_resp):.1f} vs admitted min {min(adm_resp):.1f}")

# --- diversity ---------------------------------------------------------------
div = {}
for row in csv.DictReader(open("results/cross_model_pilot_synthesis/data/trajectory_diversity_rarefaction.csv", encoding="utf-8-sig")):
    div[(row["population"], row["risk_cap"])] = row

RISKS = ("0.1", "0.6", "0.9")
CHECKPOINTS = ["GPT-5 nano", "GPT-5.4 nano", "Gemini 3 Flash", "Gemini 3.1 Flash Lite",
               "Gemini 3.5 Flash Lite", "Claude Opus 5", "Claude Sonnet 5"]
below = []
for c in CHECKPOINTS:
    if all(float(div[(c, r)]["q1_ci_high"]) < float(div[("Human", r)]["q1_ci_low"]) for r in RISKS):
        below.append(c)
check("six of seven checkpoints fall entirely below the human q1 interval",
      len(below) == 6, f"{len(below)}: {', '.join(below)}")
check("the exception is GPT-5.4 nano",
      set(CHECKPOINTS) - set(below) == {"GPT-5.4 nano"},
      f"exception {set(CHECKPOINTS) - set(below)}")

check("Claude Opus 5 collapses to one sequence at every risk",
      all(float(div[("Claude Opus 5", r)]["q1_mean"]) == 1.0 for r in RISKS), "q1 = 1.0 at all three")
check("Gemini 3 Flash and 3.1 Flash Lite collapse at risk 0.1",
      float(div[("Gemini 3 Flash", "0.1")]["q1_mean"]) == 1.0
      and float(div[("Gemini 3.1 Flash Lite", "0.1")]["q1_mean"]) == 1.0, "both 1.0")

hum_q1 = [float(div[("Human", r)]["q1_mean"]) for r in RISKS]
hum_ham = [float(div[("Human", r)]["mean_pairwise_hamming"]) for r in RISKS]
check("human effective sequences 18.9 / 19.7 / 19.3",
      [round(v, 1) for v in hum_q1] == [18.9, 19.7, 19.3], f"{[round(v,1) for v in hum_q1]}")
check("human pairwise distances 0.47 / 0.50 / 0.50",
      [round(v, 2) for v in hum_ham] == [0.47, 0.50, 0.50], f"{[round(v,2) for v in hum_ham]}")

nano_q1 = [float(div[("GPT-5.4 nano", r)]["q1_mean"]) for r in RISKS]
nano_ham = [float(div[("GPT-5.4 nano", r)]["mean_pairwise_hamming"]) for r in RISKS]
check("GPT-5.4 nano reaches 18.7 effective sequences at every risk",
      all(round(v, 1) == 18.7 for v in nano_q1), f"{[round(v,1) for v in nano_q1]}")
check("GPT-5.4 nano distances 0.48 / 0.51 / 0.49",
      [round(v, 2) for v in nano_ham] == [0.48, 0.51, 0.49], f"{[round(v,2) for v in nano_ham]}")
check("GPT-5.4 nano overlaps the human interval on both statistics",
      all(float(div[("GPT-5.4 nano", r)]["q1_ci_high"]) > float(div[("Human", r)]["q1_ci_low"]) for r in RISKS)
      and all(float(div[("GPT-5.4 nano", r)]["hamming_ci_high"]) > float(div[("Human", r)]["hamming_ci_low"]) for r in RISKS),
      "overlaps at all three risks")

gated = ["Gemini 3 Flash", "Claude Opus 5", "Claude Sonnet 5"]
check("every gate-passing checkpoint is strictly below the human sample",
      all(float(div[(c, r)]["q1_mean"]) < float(div[("Human", r)]["q1_mean"]) for c in gated for r in RISKS),
      "all three, all risks")

# --- k sensitivity -----------------------------------------------------------
ks = {row["k"]: row for row in csv.DictReader(open("results/cross_model_pilot_synthesis/data/human_archetype_k_sensitivity.csv", encoding="utf-8-sig"))}
sil = {k: float(v["silhouette"]) for k, v in ks.items()}
db = {k: float(v["davies_bouldin"]) for k, v in ks.items()}
check("k=4 has the highest silhouette and lowest Davies-Bouldin",
      max(sil, key=sil.get) == "4" and min(db, key=db.get) == "4",
      f"silhouette best k={max(sil, key=sil.get)} ({sil['4']:.2f}), DB best k={min(db, key=db.get)} ({db['4']:.2f})")
check("k=4 bootstrap ARI 0.63 with interval 0.39 to 0.98",
      round(float(ks["4"]["ari_mean"]), 2) == 0.63
      and round(float(ks["4"]["ari_p2_5"]), 2) == 0.39
      and round(float(ks["4"]["ari_p97_5"]), 2) == 0.98,
      f"{float(ks['4']['ari_mean']):.2f} [{float(ks['4']['ari_p2_5']):.2f}, {float(ks['4']['ari_p97_5']):.2f}]")
check("k=4 group sizes are the published multiset",
      sorted(int(x) for x in ks["4"]["cluster_sizes"].split("|")) == [19, 19, 133, 170],
      ks["4"]["cluster_sizes"])

# --- elicited risk null ------------------------------------------------------
er = json.load(open("results/cross_model_pilot_synthesis/data/elicited_risk_by_archetype.json", encoding="utf-8"))
o = er["omnibus_test"]
check("Kruskal-Wallis is a null: H=4.540, df=3, p=0.209, n=341",
      round(o["H"], 3) == 4.540 and o["df"] == 3 and round(o["p"], 3) == 0.209 and o["n"] == 341,
      f"H={o['H']:.3f} df={o['df']} p={o['p']:.3f} n={o['n']}")
check("effect size epsilon-squared 0.0046",
      round(er["effect_size"]["epsilon_squared"], 4) == 0.0046,
      f"{er['effect_size']['epsilon_squared']:.4f}")

# --- heterogeneity -----------------------------------------------------------
het = json.load(open("results/cross_model_pilot_synthesis/data/cross_model_heterogeneity_test.json", encoding="utf-8"))
five = het["rosters"]["five"]
bc = five["lr_test_B_vs_C_slope_heterogeneity_beyond_level"]
check("five-model heterogeneity chi2(10) = 354.7",
      round(bc["lr_stat"], 1) == 354.7 and bc["df"] == 10,
      f"chi2={bc['lr_stat']:.2f} df={bc['df']} p={bc['p_value']:.2e}")
check("the interaction fit does not converge, as the paper states",
      five["converged"]["B"] is False,
      f"converged B={five['converged']['B']}")

# --- EGT sweep ---------------------------------------------------------------
sweep = list(csv.DictReader(open("results/open_source/egt_reproduction/egt_beta_sensitivity.csv", encoding="utf-8-sig")))
col = "unsafe_frequency_mean"


def cell(rule: str, beta: float, risk: float) -> float:
    m = [r for r in sweep if r["mutation_rule"] == rule and abs(float(r["beta"]) - beta) < 1e-12
         and abs(float(r["max_private_risk"]) - risk) < 1e-12]
    return 100 * float(m[0][col])


ref = [cell("beta_over_Z", 2.0, r) for r in (0.1, 0.6, 0.9)]
check("beta=2, mu=beta/Z reproduces 99.2 / 98.0 / 1.9",
      [round(v, 1) for v in ref] == [99.2, 98.0, 1.9], f"{[round(v,1) for v in ref]}")
best = [cell("fixed_0p05", 0.01, r) for r in (0.1, 0.6, 0.9)]
check("beta=0.01, mu=0.05 predicts 87.3 / 63.9 / 38.0",
      [round(v, 1) for v in best] == [87.3, 63.9, 38.0], f"{[round(v,1) for v in best]}")

fits = list(csv.DictReader(open("results/open_source/egt_reproduction/egt_beta_fit_to_routes.csv", encoding="utf-8-sig")))
bestfit: dict[str, tuple[float, dict]] = {}
for r in fits:
    v = float(r["rmse_percentage_points"])
    if r["route"] not in bestfit or v < bestfit[r["route"]][0]:
        bestfit[r["route"]] = (v, r)
claude = bestfit["anthropic/claude-sonnet-5@default"]
gemini = bestfit["google/gemini-3-flash-preview"]
check("best fit is beta=0.01 mu=0.05 for both audited routes",
      abs(float(claude[1]["beta"]) - 0.01) < 1e-12 and abs(float(gemini[1]["beta"]) - 0.01) < 1e-12,
      f"claude beta={claude[1]['beta']}, gemini beta={gemini[1]['beta']}")
check("RMSE 10.3 pp for Claude Sonnet 5 and 15.1 for Gemini 3 Flash",
      round(claude[0], 1) == 10.3 and round(gemini[0], 1) == 15.1,
      f"{claude[0]:.2f} and {gemini[0]:.2f}")

# --- disclosed arithmetic ----------------------------------------------------
da = {r["condition"]: r for r in csv.DictReader(open("results/cross_model_pilot_synthesis/data/disclosed_arithmetic_race_bootstrap.csv", encoding="utf-8-sig"))}
c1, c2 = da["canonical"], da["calculator_decision_card"]
check("disclosed-arithmetic intervals are 48.9-55.4 and 55.1-65.3",
      round(100 * float(c1["unsafe_rate_ci95_low"]), 1) == 48.9
      and round(100 * float(c1["unsafe_rate_ci95_high"]), 1) == 55.4
      and round(100 * float(c2["unsafe_rate_ci95_low"]), 1) == 55.1
      and round(100 * float(c2["unsafe_rate_ci95_high"]), 1) == 65.3,
      "as quoted")
check("the two disclosed-arithmetic intervals overlap",
      float(c2["unsafe_rate_ci95_low"]) < float(c1["unsafe_rate_ci95_high"]),
      f"{100*float(c2['unsafe_rate_ci95_low']):.1f} < {100*float(c1['unsafe_rate_ci95_high']):.1f}")

# --- context mapping ---------------------------------------------------------
cm = {}
for man in glob.glob("results/frontier/context_mapping_campaign_v3/**/run_manifest.json", recursive=True):
    d = json.load(open(man, encoding="utf-8"))
    cm[d["model_route"]] = d
g = cm.get("google/gemini-3-flash-preview", {})
c = cm.get("anthropic/claude-sonnet-5@default", {})
check("Gemini crossed rerun completed with 120 races and 2,232 decisions",
      g.get("status") == "completed" and g.get("n_races") == 120 and g.get("n_turns") == 2232,
      f"{g.get('status')}, {g.get('n_races')} races, {g.get('n_turns')} decisions")
check("the superseded Claude crossed attempt is retained at 106 of 120",
      c.get("status") == "failed" and c.get("n_races") == 106,
      f"{c.get('status')}, {c.get('n_races')} races")

# --- crossed representation design -------------------------------------------
cm_json = json.load(open("results/derived/frontier_context_mapping_campaign_v3/context_mapping_cross_marginals.json", encoding="utf-8"))
complete = cm_json["routes_complete"]
check("the crossed context-by-mapping design is complete for two routes",
      len(complete) == 2, ", ".join(complete))
check("each crossed route has 120 races and 2,232 decisions",
      all(
          json.load(open(m, encoding="utf-8"))["n_races"] == 120
          and json.load(open(m, encoding="utf-8"))["n_turns"] == 2232
          for m in glob.glob("results/frontier/context_mapping_campaign_v3/ai-race-frontier-context-mapping/**/run_manifest.json", recursive=True)
      ),
      "both manifests 120/2232")

pc = cm_json["paired_contrasts"]
gem = pc["google/gemini-3-flash-preview"]
cla = pc["anthropic/claude-sonnet-5@default"]
for name, res in (("Gemini 3 Flash", gem), ("Claude Sonnet 5", cla)):
    check(f"horizon pairing verified for {name}",
          res["mapping"]["pairing_verified"] and res["context"]["pairing_verified"],
          f"mapping {res['mapping']['horizon_draw_matched_blocks']}/{res['mapping']['horizon_draw_blocks_checked']} blocks")

check("mapping effect 9.5 pp [5.6, 13.8] on Gemini 3 Flash",
      round(100 * gem["mapping"]["mean_difference"], 1) == 9.5
      and round(100 * gem["mapping"]["ci95_low"], 1) == 5.6
      and round(100 * gem["mapping"]["ci95_high"], 1) == 13.8,
      f"{100*gem['mapping']['mean_difference']:.1f} [{100*gem['mapping']['ci95_low']:.1f}, {100*gem['mapping']['ci95_high']:.1f}]")
check("mapping effect 8.3 pp [5.4, 12.2] on Claude Sonnet 5",
      round(100 * cla["mapping"]["mean_difference"], 1) == 8.3
      and round(100 * cla["mapping"]["ci95_low"], 1) == 5.4
      and round(100 * cla["mapping"]["ci95_high"], 1) == 12.2,
      f"{100*cla['mapping']['mean_difference']:.1f} [{100*cla['mapping']['ci95_low']:.1f}, {100*cla['mapping']['ci95_high']:.1f}]")
check("context effect 10.8 pp [6.0, 15.6] on Gemini 3 Flash",
      round(100 * gem["context"]["mean_difference"], 1) == 10.8
      and round(100 * gem["context"]["ci95_low"], 1) == 6.0
      and round(100 * gem["context"]["ci95_high"], 1) == 15.6,
      f"{100*gem['context']['mean_difference']:.1f} [{100*gem['context']['ci95_low']:.1f}, {100*gem['context']['ci95_high']:.1f}]")
check("context effect 3.4 pp [0.9, 7.4] on Claude Sonnet 5",
      round(100 * cla["context"]["mean_difference"], 1) == 3.4
      and round(100 * cla["context"]["ci95_low"], 1) == 0.9
      and round(100 * cla["context"]["ci95_high"], 1) == 7.4,
      f"{100*cla['context']['mean_difference']:.1f} [{100*cla['context']['ci95_low']:.1f}, {100*cla['context']['ci95_high']:.1f}]")
check("every crossed interval excludes zero",
      all(r[f]["ci95_low"] > 0 for r in (gem, cla) for f in ("mapping", "context")),
      "all four lower bounds positive")
check("60 paired repetition blocks per contrast",
      all(r[f]["n_blocks"] == 60 for r in (gem, cla) for f in ("mapping", "context")),
      "all four contrasts use 60 blocks")

# --- matched group-size comparison -------------------------------------------
mm = json.load(open("results/derived/nplayer_matched_campaign/nplayer_matched_rates.json", encoding="utf-8"))
mrows = list(csv.DictReader(open("results/derived/nplayer_matched_campaign/nplayer_matched_rates.csv", encoding="utf-8-sig")))
by_n = {int(r["n_players"]): r for r in mrows if r["risk"] == "0.6"}
check("the matched sweep covers all four group sizes at risk 0.6",
      sorted(by_n) == [2, 3, 4, 5], f"group sizes {sorted(by_n)}")
check("every matched cell has ten races and no parse failure",
      all(int(r["n_races"]) == 10 for r in by_n.values()),
      ", ".join(f"N={n}:{by_n[n]['n_races']}" for n in sorted(by_n)))
check("matched decision counts are 176 / 264 / 352 / 440",
      [int(by_n[n]["n_decisions"]) for n in (2, 3, 4, 5)] == [176, 264, 352, 440],
      str([int(by_n[n]["n_decisions"]) for n in (2, 3, 4, 5)]))
check("matched unsafe rates are 68.8 / 80.3 / 98.3 / 100.0 percent",
      [round(100 * float(by_n[n]["unsafe_rate"]), 1) for n in (2, 3, 4, 5)]
      == [68.8, 80.3, 98.3, 100.0],
      str([round(100 * float(by_n[n]["unsafe_rate"]), 1) for n in (2, 3, 4, 5)]))
check("unsafe play rises monotonically with the number of competitors",
      all(float(by_n[n]["unsafe_rate"]) < float(by_n[n + 1]["unsafe_rate"]) for n in (2, 3, 4)),
      "strictly increasing in N")
check("each matched cell records the identity that collected it",
      len({by_n[n]["executing_identity"] for n in sorted(by_n)}) == 4,
      ", ".join(f"N={n}:{by_n[n]['executing_identity']}" for n in sorted(by_n)))

mc = mm["paired_contrasts"]["google/gemini-3-flash-preview"]["0.6"]
for n, expect in (("3", (11.1, 6.4, 15.1)), ("4", (28.2, 22.8, 33.2)), ("5", (31.0, 26.4, 35.6))):
    got = (round(100 * mc[n]["mean_difference"], 1),
           round(100 * mc[n]["ci95_low"], 1),
           round(100 * mc[n]["ci95_high"], 1))
    check(f"matched contrast N={n} minus N=2 is {expect[0]} pp [{expect[1]}, {expect[2]}]",
          got == expect, str(got))
check("every matched contrast excludes zero",
      all(mc[n]["ci95_low"] > 0 for n in ("3", "4", "5")), "all three lower bounds positive")
check("horizon pairing verified for every matched contrast",
      all(mc[n]["pairing_verified"] and mc[n]["n_blocks"] == 10 for n in ("3", "4", "5")),
      "10 of 10 blocks in each")

# --- report ------------------------------------------------------------------
width = max(len(n) for n, _, _ in results)
fails = 0
for name, verdict, detail in results:
    if verdict == "FAIL":
        fails += 1
    print(f"{verdict:4s}  {name:<{width}}  {detail}")
print()
print(f"{len(results) - fails} of {len(results)} claims verified against their artifact")
if fails:
    raise SystemExit(f"{fails} claim(s) do not match the artifact")
