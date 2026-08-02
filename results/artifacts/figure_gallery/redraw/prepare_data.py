"""Derive every number the redrawn paper figures plot, and write one source table
per figure.

Why this exists as a separate step: the audit in ``FIGURE_REDRAW_PLAN.md`` found
several figures whose plotted values could not be reproduced from any checked-in
table (fig 01's error bars were +/-1 SE while its caption implied 95% CI; fig 04's
paired contrast was never computed at all). Splitting derivation from drawing means
every mark in the final PDFs traces to a CSV a reviewer can open, which is also the
"table view" relief the palette contrast check requires.

Outputs land in ``results/artifacts/figure_gallery/redraw/tables/``. Drawing code reads only those
tables and never touches a raw run directory.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent / "tables"
OUT.mkdir(parents=True, exist_ok=True)

PERSONA = REPO / "results/open_source/prompt_sensitivity_pilot"
SURFACE = REPO / "results/open_source/surface_sensitivity_pilot"
XPROV = REPO / "results/cross_provider"
TWOP = REPO / "results/derived/two_player_paper_analysis/tables"
CONTEXT = REPO / "results/open_source/context_skin_pilot/analysis_live_pilot_t0"

provenance: dict[str, dict] = {}


def emit(name: str, frame: pd.DataFrame, sources: list[str], notes: str) -> None:
    path = OUT / f"{name}.csv"
    frame.to_csv(path, index=False)
    provenance[name] = {"rows": len(frame), "sources": sources, "notes": notes}
    print(f"  {name}.csv  ({len(frame)} rows)")


# ---------------------------------------------------------------------------
# Figure A -- human-reference comparison
# ---------------------------------------------------------------------------
# Human values are Fernandez Domingos & Han (2026) Table 1 model 6. The SEs live
# only inside the free-text `description` column of human_comparison.csv, so they
# are restated here as machine-readable numbers; see the provenance note.
HUMAN_MODEL6 = {
    "first_round_unsafe": (0.217, 0.116),
    "progress_gap_before": (-0.296, 0.149),
    "own_prev_unsafe": (-0.193, 0.192),
    "opponent_prev_unsafe": (0.607, 0.192),
}
ROW_LABEL = {
    "first_round_unsafe": "Chose Unsafe in round 1",
    "progress_gap_before": "Own progress lead over opponent",
    "own_prev_unsafe": "Own previous action was Unsafe",
    "opponent_prev_unsafe": "Opponent's previous action was Unsafe",
    "C(max_private_risk)[T.0.6]": "Max private risk 0.6 (vs 0.1)",
    "C(max_private_risk)[T.0.9]": "Max private risk 0.9 (vs 0.1)",
}


def figure_a() -> None:
    print("figure A -- human-reference comparison")
    coef = pd.read_csv(PERSONA / "clustered_logit_coefficients.csv")
    spec6 = coef[coef.specification == 6].set_index("term")
    jack = pd.read_csv(PERSONA / "logit_robustness_jackknife.csv")
    jack = jack[jack.variant == "full"].set_index("term")

    rows = []
    for term in HUMAN_MODEL6:
        beta, se = HUMAN_MODEL6[term]
        llm = spec6.loc[term]
        rows.append(
            {
                "block": "A",
                "term": term,
                "label": ROW_LABEL[term],
                "human_beta": beta,
                "human_se": se,
                "human_ci_low": beta - 1.96 * se,
                "human_ci_high": beta + 1.96 * se,
                "llm_beta": llm.coefficient,
                "llm_se": llm.cluster_robust_se,
                "llm_ci_low": llm.ci_95_low,
                "llm_ci_high": llm.ci_95_high,
                # Sort key: how far the pilot sits from the human estimate in units
                # of the human study's own precision. Scale-free, and it produces the
                # same order as the raw absolute difference, so the ordering is not
                # an artefact of the normaliser.
                "divergence": abs(llm.coefficient - beta) / se,
                "sign_stable": bool(jack.loc[term, "sign_stable"]),
                "jack_min": jack.loc[term, "coefficient_min"],
                "jack_max": jack.loc[term, "coefficient_max"],
            }
        )
    for term in ("C(max_private_risk)[T.0.6]", "C(max_private_risk)[T.0.9]"):
        llm = spec6.loc[term]
        rows.append(
            {
                "block": "B",
                "term": term,
                "label": ROW_LABEL[term],
                "human_beta": np.nan,
                "human_se": np.nan,
                "human_ci_low": np.nan,
                "human_ci_high": np.nan,
                "llm_beta": llm.coefficient,
                "llm_se": llm.cluster_robust_se,
                "llm_ci_low": llm.ci_95_low,
                "llm_ci_high": llm.ci_95_high,
                "divergence": np.nan,
                "sign_stable": bool(jack.loc[term, "sign_stable"]),
                "jack_min": jack.loc[term, "coefficient_min"],
                "jack_max": jack.loc[term, "coefficient_max"],
            }
        )
    frame = pd.DataFrame(rows)
    block_a = frame[frame.block == "A"].sort_values("divergence", ascending=False)
    frame = pd.concat([block_a, frame[frame.block == "B"]], ignore_index=True)
    emit(
        "figA_human_reference",
        frame,
        [
            "results/open_source/prompt_sensitivity_pilot/clustered_logit_coefficients.csv (specification==6)",
            "results/open_source/prompt_sensitivity_pilot/logit_robustness_jackknife.csv (variant=='full')",
            "Fernandez Domingos & Han (2026) Table 1 model 6 (human beta and SE)",
        ],
        "LLM intervals are the 95% cluster-robust CIs already in the CSV, NOT the "
        "+/-1 SE bars the superseded scorecard drew. Human CIs are beta +/- 1.96*SE; "
        "the SEs are transcribed from the free-text description column of "
        "human_comparison.csv because no machine-readable human_se field exists.",
    )

    # --- companion table: the four effects that share no scale with the forest ---
    hc = pd.read_csv(PERSONA / "human_comparison.csv")
    tail = hc[hc.effect_id.isin(["E5", "E6", "E7", "E8"])].copy()

    contrasts = pd.read_csv(PERSONA / "treatment_contrasts_round2plus.csv")
    strat = pd.read_csv(PERSONA / "strategy_summary_player.csv")

    def strata_range(left: float, right: float) -> str:
        """The scorecard drew E5/E6 as one bar each; they are means over six persona
        strata whose Cohen's d values disagree in sign, which the bar concealed."""
        sub = contrasts[(contrasts.level_left == left) & (contrasts.level_right == right)]
        assert not sub.empty, f"no contrast rows for {left} vs {right}"
        return (f"{sub.cohens_d.min():+.3f} to {sub.cohens_d.max():+.3f} "
                f"across {len(sub)} persona strata")

    tail["llm_value"] = pd.to_numeric(tail.llm_value, errors="coerce")
    tail["human_value"] = pd.to_numeric(tail.human_value, errors="coerce")
    tail["strata_spread"] = [
        strata_range(0.6, 0.9),
        strata_range(0.1, 0.6),
        "",
        "",
    ]
    emit(
        "figA_side_table",
        tail[["effect_id", "name", "kind", "test", "human_value", "llm_value",
              "criterion", "verdict", "strata_spread"]],
        ["results/open_source/prompt_sensitivity_pilot/human_comparison.csv",
         "results/open_source/prompt_sensitivity_pilot/treatment_contrasts_round2plus.csv",
         "results/open_source/prompt_sensitivity_pilot/strategy_summary_player.csv"],
        "E8 has no published human value (human_value is empty upstream). E8's LLM "
        f"denominator is {int(strat.shape[0])} players with a unique nearest-strategy "
        "label, not the full pilot roster -- ties are kept unlabelled by "
        "strategy_analysis/classify.py.",
    )


# ---------------------------------------------------------------------------
# Figure B -- surface-wording sensitivity
# ---------------------------------------------------------------------------
READABLE = {
    "canonical": "canonical wording (reference)",
    "emotional_importance": "emotional framing added",
    "position_goal_first": "goal stated first",
    "emphasis_uppercase": "key terms in uppercase",
    "order_state_reversed": "state block order reversed",
    "format_markdown": "markdown formatting",
    "format_numbered_state": "state block numbered",
    "order_payoffs_reversed": "payoff list order reversed",
    "boundary_compact": "section boundaries compacted",
    "format_dense": "whitespace removed",
    "format_extra_spacing": "extra whitespace added",
    "format_xml": "XML tags instead of headings",
    "lexical_synonyms": "synonyms substituted",
    "noise_minor_typo": "minor typos introduced",
    "order_actions_reversed": "action options listed in reverse",
    "paraphrase_instruction": "instruction paraphrased",
    "position_risk_near_response": "risk rules moved next to the answer",
    "voice_impersonal": "impersonal voice",
}


def figure_b() -> None:
    print("figure B -- surface wording")
    v = pd.read_csv(SURFACE / "variant_summary.csv")
    v["label"] = v.variant.map(READABLE)
    assert v.label.notna().all(), "unmapped variant label"
    canonical = float(v.loc[v.variant == "canonical", "unsafe_rate"].iloc[0])
    v["delta_vs_canonical"] = v.unsafe_rate - canonical
    v["signed_flip_to_unsafe"] = v.first_round_safe_to_unsafe
    v["signed_flip_to_safe"] = -v.first_round_unsafe_to_safe
    # Sorting both panels by the round-1 flip rate is what makes the dissociation
    # legible: a variant can be flat at round 1 and extreme over the trajectory.
    v = v.sort_values(
        ["first_round_flip_rate_vs_canonical", "unsafe_rate"], ascending=[False, False]
    )
    emit(
        "figB_surface_variants",
        v,
        ["results/open_source/surface_sensitivity_pilot/variant_summary.csv"],
        "unsafe_rate intervals are cluster-bootstrap over 10 repetition blocks and are "
        "marginal: each variant was resampled on its own seed stream, so differences "
        "between them have no valid interval here and delta_vs_canonical is reported "
        "as a point value only.",
    )

    r = pd.read_csv(SURFACE / "risk_variant_summary.csv")
    within = r.groupby("variant").unsafe_rate.agg(lambda s: s.max() - s.min())
    between = r.groupby("max_private_risk").unsafe_rate.agg(lambda s: s.max() - s.min())
    emit(
        "figB_pooling_check",
        pd.DataFrame(
            {
                "quantity": ["within-variant spread across risk (median)",
                             "within-variant spread across risk (max)",
                             "between-variant spread at risk 0.1",
                             "between-variant spread at risk 0.6",
                             "between-variant spread at risk 0.9"],
                "value": [within.median(), within.max(),
                          between.loc[0.1], between.loc[0.6], between.loc[0.9]],
            }
        ),
        ["results/open_source/surface_sensitivity_pilot/risk_variant_summary.csv"],
        "Justifies pooling risk in figure B: between-variant spread exceeds "
        "within-variant spread across risk by roughly 15x and the rank order is "
        "preserved at all three risk levels.",
    )


# ---------------------------------------------------------------------------
# Figure C -- cross-provider opponent contingency
# ---------------------------------------------------------------------------
MATCHUPS = {
    "Luna vs Haiku": "openai_claude/openai-gpt-5.6-luna-bedrock__vs__anthropic-claude-haiku-4.5-bedrock",
    "Gemini vs Haiku": "gemini_claude/google-gemini-3.5-flash-lite__vs__anthropic-claude-haiku-4.5-bedrock",
    "Gemini vs Luna": "gemini_openai/google-gemini-3.5-flash-lite__vs__openai-gpt-5.6-luna-bedrock",
}


def _load_turns(rel: str) -> pd.DataFrame:
    rows = []
    with open(XPROV / rel / "turns.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                d = json.loads(line)
                rows.append(
                    {k: d[k] for k in
                     ("game_seed", "rep", "max_private_risk", "player", "opponent",
                      "unsafe", "parse_failed", "prompt_version",
                      "own_progress_before", "opponent_progress_before")}
                )
    return pd.DataFrame(rows)


def figure_c() -> None:
    print("figure C -- cross-provider opponent contingency")
    frames = []
    for name, rel in MATCHUPS.items():
        t = _load_turns(rel)
        assert (t.prompt_version == "ai-race-fairgame-v3").all(), f"{name}: mechanism mix"
        assert not t.parse_failed.any(), f"{name}: parse failures present"
        t["matchup"] = name
        frames.append(t)
    turns = pd.concat(frames, ignore_index=True)

    # Repetition is the common-random-number block: matched game_seed and horizon
    # across matchups, so a model's two opponent contexts are paired rep by rep.
    per_rep = (
        turns.groupby(["matchup", "player", "opponent", "max_private_risk", "rep"])
        .unsafe.mean().reset_index(name="unsafe_rate")
    )
    cell = (
        per_rep.groupby(["matchup", "player", "opponent", "max_private_risk"])
        .unsafe_rate.agg(mean="mean", sd="std", n_reps="size").reset_index()
    )
    emit(
        "figC_levels",
        cell,
        [f"results/cross_provider/{r}/turns.jsonl" for r in MATCHUPS.values()],
        "Cell mean is the mean of per-repetition rates (the CRN block is the sampling "
        "unit, not the decision). Two-player mechanism only; the N=3 run is excluded.",
    )

    # Paired contrast: same model, same rep, two different opponents.
    from scipy import stats

    pairs = []
    for model, opp_a, opp_b in [
        ("GPT Luna", "Claude Haiku 4.5", "Gemini 3.5 Flash-Lite"),
        ("Gemini 3.5 Flash-Lite", "Claude Haiku 4.5", "GPT Luna"),
        ("Claude Haiku 4.5", "GPT Luna", "Gemini 3.5 Flash-Lite"),
    ]:
        sub = per_rep[per_rep.player == model]
        for risk in (None, 0.1, 0.6, 0.9):
            s = sub if risk is None else sub[sub.max_private_risk == risk]
            a = s[s.opponent == opp_a].groupby("rep").unsafe_rate.mean()
            b = s[s.opponent == opp_b].groupby("rep").unsafe_rate.mean()
            common = a.index.intersection(b.index)
            d = (b.loc[common] - a.loc[common]).to_numpy()
            n = len(d)
            half = stats.t.ppf(0.975, n - 1) * d.std(ddof=1) / np.sqrt(n)
            pairs.append(
                {
                    "player": model, "opponent_a": opp_a, "opponent_b": opp_b,
                    "max_private_risk": "pooled" if risk is None else risk,
                    "rate_vs_a": a.loc[common].mean(), "rate_vs_b": b.loc[common].mean(),
                    "paired_delta": d.mean(), "ci_low": d.mean() - half,
                    "ci_high": d.mean() + half, "n_matched_reps": n,
                }
            )
    emit(
        "figC_paired_delta",
        pd.DataFrame(pairs),
        [f"results/cross_provider/{r}/turns.jsonl" for r in MATCHUPS.values()],
        "Paired within-model, between-opponent contrast on matched repetition blocks; "
        "95% t-interval on the n=10 per-rep differences. This is the estimand the "
        "superseded sorted forest discarded by computing unpaired marginal intervals.",
    )


# ---------------------------------------------------------------------------
# Figure D -- race position, five-checkpoint baseline
# ---------------------------------------------------------------------------
def figure_d() -> None:
    print("figure D -- race position")
    pos = pd.read_csv(TWOP / "baseline_position_estimates.csv")
    # A degenerate interval here means every block agreed, not that the estimate is
    # precise; the drawing code marks these rather than rendering a zero-width bar.
    pos["degenerate"] = np.isclose(pos.ci95_low, pos.ci95_high)
    wide = pos.pivot(index="model_label", columns="position", values="estimate")
    contrast = (wide["Behind"] - wide["Ahead"]).rename("behind_minus_ahead").reset_index()
    emit(
        "figD_position",
        pos,
        ["results/derived/two_player_paper_analysis/tables/baseline_position_estimates.csv"],
        "Ahead and Behind carry identical n by construction: every asymmetric round "
        "contributes exactly one of each. Five checkpoints, 10 CRN blocks each, all "
        "with SHA-256 run manifests.",
    )
    emit(
        "figD_contrast",
        contrast,
        ["results/derived/two_player_paper_analysis/tables/baseline_position_estimates.csv"],
        "Point contrast only: the upstream table publishes marginal intervals per "
        "position, not a paired interval on the difference.",
    )


# ---------------------------------------------------------------------------
# Figure E -- baseline risk response
# ---------------------------------------------------------------------------
def figure_e() -> None:
    print("figure E -- baseline risk response")
    src = pd.read_csv(TWOP / "fig01_baseline_risk_response_source.csv")
    rate = src[src.section == "risk_response"].copy()
    rate["saturated"] = np.isclose(rate.ci95_low, rate.ci95_high)
    contrast = src[src.section == "high_minus_low"].copy()
    emit(
        "figE_risk_response",
        rate,
        ["results/derived/two_player_paper_analysis/tables/fig01_baseline_risk_response_source.csv"],
        "Two Gemini cells at risk 0.1 are saturated (20/20 player-races Unsafe), so "
        "every bootstrap resample equals 1.0 and the interval is degenerate by "
        "construction rather than precise.",
    )
    emit(
        "figE_contrast",
        contrast,
        ["results/derived/two_player_paper_analysis/tables/fig01_baseline_risk_response_source.csv"],
        "holm_p_high_vs_low upstream was adjusted over a SEVEN-model family, not the "
        "five plotted; p-values are therefore kept out of the figure. The two smallest "
        "Wilcoxon p-values sit at the n=10 signed-rank floor of 2/2^10.",
    )


# ---------------------------------------------------------------------------
# Figure F -- repeat-run stability
# ---------------------------------------------------------------------------
# The earlier pilot's run directory was deleted from the working tree in b07ae73 and
# analyze_two_player_paper_figures.py still hard-codes the dead path, so this figure
# cannot be regenerated from a checkout. The blob is still reachable in history, so we
# read it from there rather than dropping the panel.
OLD_RUN = "ai_race/results/_api_5games_allrisk/google-gemini-3-flash-preview/turns.jsonl"
OLD_COMMIT = "a81a8c8"
NEW_RUN = REPO / "results/frontier/baseline/google-gemini-3-flash-preview/turns.jsonl"


def _git_turns(commit: str, path: str) -> pd.DataFrame:
    blob = subprocess.run(
        ["git", "show", f"{commit}:{path}"], cwd=REPO,
        capture_output=True, check=True,
    ).stdout.decode("utf-8")
    return pd.DataFrame(
        json.loads(line) for line in blob.splitlines() if line.strip()
    )


def figure_f() -> None:
    print("figure F -- repeat-run stability")
    src = pd.read_csv(TWOP / "fig13_repeat_run_stability_source.csv")
    emit(
        "figF_rates",
        src[src.section == "rate"],
        ["results/derived/two_player_paper_analysis/tables/fig13_repeat_run_stability_source.csv"],
        "Panel A: aggregate risk response of the two independent unseeded pilots.",
    )
    old = _git_turns(OLD_COMMIT, OLD_RUN)
    print(f"    recovered {len(old)} decisions from {OLD_COMMIT}:{OLD_RUN}")
    new = pd.DataFrame(
        json.loads(line) for line in open(NEW_RUN, encoding="utf-8") if line.strip()
    )
    keys = ["game_id", "round", "player"]
    paired = old.merge(
        new[keys + ["action"]], on=keys, suffixes=("_old", "_new"),
        how="inner", validate="one_to_one",
    )
    paired["same_action"] = paired.action_old.eq(paired.action_new)

    # The sampling unit is the race seed, not the decision: five seeds recur at every
    # risk level, so the 74 decisions behind each pooled rate are ten trajectories, not
    # 74 independent draws. Reporting per-seed rates shows that instead of hiding it
    # behind an interval a five-block bootstrap cannot stabilise.
    per_seed = (
        paired.groupby(["max_private_risk", "game_seed"])
        .agg(agreement=("same_action", "mean"), n_decisions=("same_action", "size"))
        .reset_index()
    )
    pooled = (
        paired.groupby("max_private_risk")
        .agg(agreement=("same_action", "mean"), n_decisions=("same_action", "size"),
             n_seeds=("game_seed", "nunique"))
        .reset_index()
    )
    # Agreement is only informative against the rate two independent runs would hit by
    # chance given their own marginals; at risk 0.1 both runs are uniformly Unsafe, so
    # agreement is forced to 1.0 and carries no information about reproducibility.
    marg = paired.groupby("max_private_risk").agg(
        p_old=("action_old", lambda s: (s == "unsafe").mean()),
        p_new=("action_new", lambda s: (s == "unsafe").mean()),
    ).reset_index()
    pooled = pooled.merge(marg, on="max_private_risk")
    pooled["chance_agreement"] = (
        pooled.p_old * pooled.p_new + (1 - pooled.p_old) * (1 - pooled.p_new)
    )
    pooled["kappa"] = (
        (pooled.agreement - pooled.chance_agreement) / (1 - pooled.chance_agreement)
    )
    pooled["forced"] = np.isclose(pooled.chance_agreement, 1.0)

    emit("figF_agreement_pooled", pooled,
         [f"git {OLD_COMMIT}:{OLD_RUN}", str(NEW_RUN.relative_to(REPO))],
         "Chance agreement and Cohen's kappa are computed here, not upstream. At risk "
         "0.1 both runs are uniformly Unsafe so agreement is forced to 1.0 and kappa is "
         "undefined; the 65% figures at risk 0.6/0.9 sit only slightly above chance.")
    emit("figF_agreement_per_seed", per_seed,
         [f"git {OLD_COMMIT}:{OLD_RUN}", str(NEW_RUN.relative_to(REPO))],
         "Five shared race seeds per risk level; these are the clusters the pooled rate "
         "averages over.")
    emit("figF_agreement_per_decision",
         paired[["max_private_risk", "game_seed", "round", "player", "action_old",
                 "action_new", "same_action"]].sort_values(
             ["max_private_risk", "game_seed", "round", "player"]),
         [f"git {OLD_COMMIT}:{OLD_RUN}", str(NEW_RUN.relative_to(REPO))],
         "Every matched decision, so the figure can show the 222 units themselves "
         "rather than three summary heights.")


def figure_g() -> None:
    print("figure G -- comprehension audit")
    dom = pd.read_csv(CONTEXT / "comprehension_by_domain.csv")
    cell = pd.read_csv(CONTEXT / "comprehension_by_cell.csv")

    def wilson(k: float, n: int, z: float = 1.96) -> tuple[float, float]:
        p = k / n
        d = 1 + z * z / n
        c = p + z * z / (2 * n)
        h = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
        return (c - h) / d, (c + h) / d

    rows = []
    for _, r in dom.iterrows():
        for metric, value in (("semantic accuracy", r.semantic_accuracy),
                              ("strict format validity", r.strict_valid_rate)):
            lo, hi = wilson(value * r.n, int(r.n))
            rows.append({"domain": r.domain, "metric": metric, "value": value,
                         "ci_low": lo, "ci_high": hi, "n": int(r.n),
                         "gated": metric == "semantic accuracy"})
    n_all = int(dom.n.sum())
    for metric, col in (("semantic accuracy", "semantic_accuracy"),
                        ("strict format validity", "strict_valid_rate")):
        value = float((dom[col] * dom.n).sum() / n_all)
        lo, hi = wilson(value * n_all, n_all)
        rows.append({"domain": "OVERALL", "metric": metric, "value": value,
                     "ci_low": lo, "ci_high": hi, "n": n_all,
                     "gated": metric == "semantic accuracy"})
    emit(
        "figG_comprehension",
        pd.DataFrame(rows),
        [str(CONTEXT.relative_to(REPO) / "comprehension_by_domain.csv")],
        "Wilson 95% intervals, absent upstream. The frozen gate applies to semantic "
        "accuracy only (>=90% overall, >=75% per domain); strict format validity is "
        "not gated, so the two metrics are drawn against different references. Rule "
        "recall's 0% strict rate is a true zero, not missing data.",
    )
    passed = 0
    if "semantic_accuracy" in cell:
        passed = int((cell.semantic_accuracy >= 0.90).sum())
    print(f"    cells passing the 90% overall gate: {passed} / {len(cell)}")


if __name__ == "__main__":
    figure_a()
    figure_b()
    figure_c()
    figure_d()
    figure_e()
    figure_f()
    figure_g()
    (OUT / "provenance.json").write_text(json.dumps(provenance, indent=2), encoding="utf-8")
    print(f"\nwrote {len(provenance)} source tables to {OUT}")
