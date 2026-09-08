"""Re-derive every plotted value straight from the upstream run artifacts and assert
it matches the figure source tables.

This is deliberately a second, independent path: prepare_data.py builds the tables,
this file checks them against the original CSVs and JSONL without reusing any of
that code. The audit that motivated the redraw found figures whose error bars were
+/-1 SE while the caption implied 95% CI, and a headline ranking that silently mixed
two game mechanisms -- both would have been caught by a check like this.

Run: python verify.py
"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
TABLES = Path(__file__).resolve().parent / "tables"
FIGURES = Path(__file__).resolve().parent / "figures"

checks: list[tuple[bool, str]] = []


def check(ok: bool, what: str) -> None:
    checks.append((bool(ok), what))


def close(a, b, tol: float = 1e-9) -> bool:
    return bool(np.allclose(np.asarray(a, dtype=float), np.asarray(b, dtype=float),
                            atol=tol, rtol=0))


# --- A: coefficients come from specification 6, with 95% CIs (not +/-1 SE) --------
coef = pd.read_csv(REPO / "results/open_source/prompt_sensitivity_pilot"
                   / "clustered_logit_coefficients.csv")
spec6 = coef[coef.specification == 6].set_index("term")
figA = pd.read_csv(TABLES / "figA_human_reference.csv").set_index("term")
for term in figA.index:
    r, u = figA.loc[term], spec6.loc[term]
    check(close(r.llm_beta, u.coefficient), f"A/{term}: point estimate")
    check(close([r.llm_ci_low, r.llm_ci_high], [u.ci_95_low, u.ci_95_high]),
          f"A/{term}: 95% CI (not +/-1 SE)")
    # The superseded scorecard drew beta +/- 1*SE; assert we did not.
    check(not close(r.llm_ci_high - r.llm_beta, u.cluster_robust_se, tol=1e-6),
          f"A/{term}: interval is wider than one SE")
jack = pd.read_csv(REPO / "results/open_source/prompt_sensitivity_pilot"
                   / "logit_robustness_jackknife.csv")
jack = jack[jack.variant == "full"].set_index("term")
for term in figA.index:
    check(bool(figA.loc[term, "sign_stable"]) == bool(jack.loc[term, "sign_stable"]),
          f"A/{term}: jackknife sign stability")
check(int((~figA.sign_stable).sum()) == 1, "A: exactly one unstable coefficient")

side = pd.read_csv(TABLES / "figA_side_table.csv")
hc = pd.read_csv(REPO / "results/open_source/prompt_sensitivity_pilot"
                 / "human_comparison.csv").set_index("effect_id")
check(pd.isna(hc.loc["E8", "human_value"]), "A: E8 has no published human value")
for eid in ("E5", "E6", "E7", "E8"):
    check(close(side.set_index("effect_id").loc[eid, "llm_value"],
                float(hc.loc[eid, "llm_value"])), f"A/{eid}: LLM value")

# --- B: variant rates and the pooling justification -------------------------------
vs = pd.read_csv(REPO / "results/open_source/surface_sensitivity_pilot"
                 / "variant_summary.csv").set_index("variant")
figB = pd.read_csv(TABLES / "figB_surface_variants.csv").set_index("variant")
check(len(figB) == 18, "B: 18 variants")
check(close(figB.unsafe_rate, vs.loc[figB.index, "unsafe_rate"]), "B: unsafe rates")
check(close(figB.first_round_flip_rate_vs_canonical,
            vs.loc[figB.index, "first_round_flip_rate_vs_canonical"]),
      "B: round-1 flip rates")
check(int(vs.parse_failures.sum()) == 0, "B: zero parse failures across all variants")
mp = figB[figB.interpretation == "meaning_preserving"]
check(len(mp) == 15 and mp.first_round_flip_rate_vs_canonical.max() <= 0.15,
      "B: all 15 meaning-preserving variants flip <=15% of round-1 decisions")
check(close(figB.loc["order_actions_reversed", "first_round_flip_rate_vs_canonical"],
            2 / 60), "B: action-order reversal flips exactly 2 of 60 at round 1")
check(int(vs.loc["canonical"].n_decisions) == 558, "B: 558 decisions per variant")
rank = figB.unsafe_rate.rank()
check(int(rank["canonical"]) == 4,
      "B: canonical is 4th lowest of 18 (not mid-distribution, contra the old note)")

# --- C: two-player only, and the paired delta -------------------------------------
XP = REPO / "results/cross_provider"
MATCHUPS = {
    "Luna vs Haiku": "openai_claude/openai-gpt-5.6-luna-bedrock__vs__anthropic-claude-haiku-4.5-bedrock",
    "Gemini vs Haiku": "gemini_claude/google-gemini-3.5-flash-lite__vs__anthropic-claude-haiku-4.5-bedrock",
    "Gemini vs Luna": "gemini_openai/google-gemini-3.5-flash-lite__vs__openai-gpt-5.6-luna-bedrock",
}
raw = []
for name, rel in MATCHUPS.items():
    with open(XP / rel / "turns.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                d["matchup"] = name
                raw.append(d)
raw = pd.DataFrame(raw)
check((raw.prompt_version == "ai-race-fairgame-v3").all(),
      "C: every plotted decision uses the two-player mechanism")
check(raw.game_seed.nunique() == 10, "C: 10 shared CRN game seeds")
seeds_per_matchup = raw.groupby("matchup").game_seed.unique().apply(sorted)
check(all(list(s) == list(seeds_per_matchup.iloc[0]) for s in seeds_per_matchup),
      "C: all three matchups share the same seed set (pairing is real)")
per_rep = (raw.groupby(["player", "opponent", "max_private_risk", "rep"])
           .unsafe.mean().reset_index(name="rate"))
figC = pd.read_csv(TABLES / "figC_paired_delta.csv")
for _, r in figC[figC.max_private_risk == "pooled"].iterrows():
    s = per_rep[per_rep.player == r.player]
    a = s[s.opponent == r.opponent_a].groupby("rep").rate.mean()
    b = s[s.opponent == r.opponent_b].groupby("rep").rate.mean()
    check(close(r.paired_delta, (b - a).mean(), tol=1e-9),
          f"C/{r.player}: paired delta")
    check(r.n_matched_reps == 10, f"C/{r.player}: 10 matched reps")
    check(r.ci_low > 0, f"C/{r.player}: interval excludes zero")

# --- D: position estimates, degenerate cells, and the exact pairing ----------------
pos = pd.read_csv(REPO / "results/derived/two_player_paper_analysis/tables"
                  / "baseline_position_estimates.csv")
figD = pd.read_csv(TABLES / "figD_position.csv")
check(close(figD.estimate, pos.estimate), "D: estimates match upstream")
for model, g in figD.groupby("model_label"):
    g = g.set_index("position")
    check(g.loc["Ahead", "n_observations"] == g.loc["Behind", "n_observations"],
          f"D/{model}: ahead and behind are exactly paired")
check(int(figD.degenerate.sum()) == 5, "D: five degenerate cells flagged, not hidden")
check((figD.n_blocks == 10).all(), "D: 10 CRN blocks in every cell")
# The five checkpoints split across two provider roots. Unlike results/cross_provider/,
# every one of them carries a completed run manifest -- which is why figure D is drawn
# from this source rather than from the cross-provider corpus.
BASELINE_RUNS = [
    "results/frontier/baseline/google-gemini-3-flash-preview",
    "results/frontier/baseline/google-gemini-3.1-flash-lite-preview",
    "results/frontier/baseline/google-gemini-3.5-flash-lite",
    "results/frontier/openai/baseline/gpt-5-nano",
    "results/frontier/openai/baseline/gpt-5.4-nano",
]
for rel in BASELINE_RUNS:
    path = REPO / rel / "run_manifest.json"
    ok = path.exists() and json.loads(path.read_text()).get("status") == "completed"
    check(ok, f"D: {Path(rel).name} has a completed run_manifest.json")
check(not list((REPO / "results/cross_provider").rglob("run_manifest.json")),
      "D: confirms cross_provider has no manifests (why figure D avoids it)")

# --- E: saturation, and p-values kept out of the figure ---------------------------
src = pd.read_csv(REPO / "results/derived/two_player_paper_analysis/tables"
                  / "fig01_baseline_risk_response_source.csv")
figE = pd.read_csv(TABLES / "figE_risk_response.csv")
check(close(figE.estimate, src[src.section == "risk_response"].estimate.to_numpy()),
      "E: risk-response estimates match upstream")
check(int(figE.saturated.sum()) == 2, "E: two saturated cells flagged")
check(figE[figE.saturated].estimate.eq(1.0).all(), "E: saturated cells are exactly 1.0")
contrast = pd.read_csv(TABLES / "figE_contrast.csv")
# Holm upstream was adjusted over a seven-model family; confirm, since that is the
# reason the figure carries no p-values.
row = contrast[contrast.model_label == "Gemini 3 Flash"].iloc[0]
check(close(row.holm_p_high_vs_low, row.wilcoxon_p * 7),
      "E: Holm family size is 7, not the 5 models plotted")
check(close(contrast[contrast.model_label == "GPT-5 nano"].iloc[0].estimate, 0.022032,
            tol=1e-6), "E: GPT-5 nano contrast is the near-zero one")

# --- F: recovered run, and the forced 100% ----------------------------------------
OLD = "ai_race/results/_api_5games_allrisk/google-gemini-3-flash-preview/turns.jsonl"
blob = subprocess.run(["git", "show", f"a81a8c8:{OLD}"], cwd=REPO,
                      capture_output=True, check=True).stdout.decode("utf-8")
old = pd.DataFrame(json.loads(l) for l in blob.splitlines() if l.strip())
new = pd.DataFrame(
    json.loads(l) for l in
    open(REPO / "results/frontier/baseline/google-gemini-3-flash-preview/turns.jsonl",
         encoding="utf-8") if l.strip())
paired = old.merge(new[["game_id", "round", "player", "action"]],
                   on=["game_id", "round", "player"], suffixes=("_old", "_new"),
                   validate="one_to_one")
paired["same"] = paired.action_old.eq(paired.action_new)
agree = paired.groupby("max_private_risk").same.mean()
figF = pd.read_csv(TABLES / "figF_agreement_pooled.csv").set_index("max_private_risk")
check(close(figF.agreement, agree.loc[figF.index]), "F: agreement rates")
check((paired.groupby("max_private_risk").size() == 74).all(),
      "F: 74 matched decisions per risk level")
check(paired.game_seed.nunique() == 5, "F: only five shared race seeds behind each bar")
r01 = paired[paired.max_private_risk == 0.1]
check(r01.action_old.eq("unsafe").all() and r01.action_new.eq("unsafe").all(),
      "F: the 100% agreement at risk 0.1 is forced by both runs being all-Unsafe")
check(bool(figF.loc[0.1, "forced"]) and not bool(figF.loc[0.6, "forced"]),
      "F: forced cell flagged")
check(figF.loc[0.6, "kappa"] < 0.2,
      "F: 65% agreement is only marginally above chance (kappa < 0.2)")

# --- G: comprehension, true zero, and the gate scope ------------------------------
dom = pd.read_csv(REPO / "results/open_source/context_skin_pilot/analysis_live_pilot_t0"
                  / "comprehension_by_domain.csv").set_index("domain")
figG = pd.read_csv(TABLES / "figG_comprehension.csv")
sem = figG[(figG.metric == "semantic accuracy") & (figG.domain != "OVERALL")]
check(close(sem.set_index("domain").value, dom.loc[sem.domain, "semantic_accuracy"]),
      "G: semantic accuracies")
check(close(figG[(figG.metric == "strict format validity") &
                 (figG.domain == "rule_recall")].value.iloc[0], 0.0),
      "G: rule recall strict validity is a true zero")
overall = figG[(figG.domain == "OVERALL") & (figG.metric == "semantic accuracy")].iloc[0]
check(close(overall.value, dom.semantic_accuracy.mean(), tol=1e-9),
      "G: overall semantic accuracy = 57.0%")
check(overall.value < 0.90, "G: overall fails the 90% gate")
check((figG[figG.metric == "strict format validity"].gated == False).all(),
      "G: strict format validity is marked as not gated")
check(close(figG.ci_low.notna().mean(), 1.0), "G: every point carries a Wilson interval")

# --- outputs exist ----------------------------------------------------------------
for name in ("figA_human_reference", "figB_surface_wording", "figB2_surface_flip_direction",
             "figC_opponent_contingency", "figD_race_position",
             "figE_baseline_risk_response", "figF_repeat_run_stability",
             "figG_comprehension_audit"):
    check((FIGURES / f"{name}.pdf").exists() and (FIGURES / f"{name}.png").exists(),
          f"output: {name}.pdf + .png")

failed = [w for ok, w in checks if not ok]
for ok, what in checks:
    print(f"  {'ok  ' if ok else 'FAIL'}  {what}")
print(f"\n{len(checks) - len(failed)}/{len(checks)} checks passed")
sys.exit(1 if failed else 0)
