# 07. Independent recomputation of the headline results from raw logs

Date: 2026-09-18. Scope: `paper/main.tex` and `paper/supplementary.tex` (AAMAS), plus the one ARR number (e).

## How this was done

The numbers were derived from raw logs only: `turns.jsonl`, `run_manifest.json` and the admission `raw_responses.jsonl` under the live campaign trees. No project analyser, `verify_manuscript_claims.py` or `ai_race/analysis` code was imported or read before the first set of numbers existed. The walk never enters `failed_runs/`. If a race has any `parse_failed` decision, the whole race is dropped (none were).

- Script: `tmp/independent_recompute_2026-09-18/recompute.py`. Output: `recompute_results.json` and `summary.txt`.
- Unsafe rate: route Unsafe decisions divided by route decisions, pooled over the cell's races (decision weighted). In the scripted campaign only `is_route_decision == true` rows count.
- Intervals: percentile bootstrap, 5,000 resamples, fixed seed 20260918 with a per-cell generator. Rates resample races. Paired contrasts resample the repetition index (the CRN block) jointly across both arms, and the contrast is the difference of pooled rates inside each resample.
- Spearman: average ranks, exact two-sided permutation over all 9! = 362,880 relabellings.

After the first numbers were in, I read `scripts/analyze_scripted_opponent.py`, `analyze_nplayer_matched.py`, `analyze_frontier_context_mapping_cross.py` and `analyze_audit_versus_behaviour.py` to explain the gaps. `diagnose_estimators.py` then reimplements the analysers' contrast estimator from scratch. It does not change the primary numbers.

## Structural facts (all agree)

| check | result |
|---|---|
| Scripted: cells, races, route decisions, rival moves | 60, 600, 5,580, 5,580 |
| Scripted: parse failures, contaminated races, rival replay deviations (recomputed from AS/AU/CS/CAS rules against the route's recorded history) | 0, 0, 0 |
| Scripted: races per cell / seat split | 10 in all 60 / 5 and 5 in all 60 |
| Scripted: realised horizon identical across the 12 cells of each (route, rep) | yes, 0 mismatches |
| Baseline v6: nine routes, each 30 races, 558 decisions, 10 per risk, 0 parse failures, 0 retries, `completed`, `confirmatory` | yes, all nine |
| Group-size grid: 12 cells, 120 races, 3,696 decisions (176/264/352/440 per cell), 0 parse failures, horizon shared across N in every rep | yes |
| Context/mapping: 120 races and 2,232 decisions per route, 12 cells of 10, 0 parse failures, one `horizon_draws_sha256` per repetition | yes, both routes |

## Result table

Estimator key: **pooled** is my primary (the difference of decision-pooled rates, reps resampled). **rep-mean** is the analysers' estimator (the unweighted mean over blocks of the per-race rate difference).

| quantity | paper value (file:line) | CLAUDE.md value | my value | my CI | agree? | explanation of any gap |
|---|---|---|---|---|---|---|
| Scripted ordering AS<=CS<=CAS<=AU | 14/15 weak, 12 strict; reversal GPT-5.5 r=0.1 CAS 100.0 vs AU 97.8 (main.tex:791-794) | 14/15, strict 12 | 14/15 weak, 12/15 strict; reversal GPT-5.5 r=0.1 (CAS 100.00, AU 97.85). Non-strict ties: Gemini r=0.1 (CAS=AU=100), Opus r=0.1 (CS=CAS=AU=100) | n/a | yes | none |
| AS to AU paired lower bounds all positive | "All fifteen ... positive, smallest +16.5" (main.tex:739, 797-798; supp:1594-1595) | all 15 positive | 15/15 positive; smallest lower bound +20.5 (GPT-5.5 r=0.1, [20.5, 31.5]); Opus r=0.1 [22.2, 47.0] | see file | claim yes; "+16.5" no | +16.5 is the rep-mean estimator's Opus r=0.1 bound (I get [16.5, 43.1] with it). Under pooling the Opus contrast is +36.6, not +30.1 |
| AU minus AS range over 15 cells | 25.3 to 87.0 (main.tex:779, 805-806; supp:1556-1574) | n/a | pooled 25.8 (GPT-5.5 0.1) to 88.2 (Opus 0.6) | e.g. Opus 0.6 [85.4, 90.5] | within ~1.2 pp | Estimator. Rep-mean gives 25.27 and 87.03, which matches. The supp table prints AS 24.7 and AU 100.0 for Gemini r=0.1 beside a contrast of +73.5, so a reader subtracting the printed rates gets 75.3 |
| Opus AU minus AS, r=0.1 | +30.1 [16.5, 42.8] (supp:1560) | n/a | pooled +36.6 | [22.2, 47.0] | **no, 6.5 pp** | Estimator. Rep-mean +30.14 [16.5, 43.1] reproduces it. Opus AS r=0.1 plays Unsafe in short races and Safe in long ones, so decision weighting and race weighting diverge |
| Risk contrast at AS (0.1 minus 0.9), range | 10.4 to 69.9, Opus +69.9 (main.tex:781, 785; supp:1556-1572, 1606) | n/a | pooled Gemini 10.8, Opus 63.4, GPT-5.4 10.8, GPT-5.5 50.5, Sonnet 11.8 | Opus rep-mean [57.1, 83.4] | **no for Opus (6.5 pp)**; others within 1.6 pp | Estimator. Rep-mean gives 10.45, 69.86, 12.35, 48.30, 12.66, which matches. The printed AS rates for Opus are 63.4 and 0.0, yet the stated contrast is +69.9 |
| GPT-5.4 AS rates | 36.6 / 32.3 / 25.8 (main.tex:809) | n/a | 36.56 / 32.26 / 25.81 | n/a | yes | none |
| Self-play vs AS, which route | Gemini 3 Flash: 98.9/73.1/60.2 (main.tex:836) and AS 24.7/17.2/14.0 (supp:1482, 1556-1558) | "24.7/17.2/14.0 vs 98.9/73.1/60.2 for one route" | Gemini 3 Flash: self-play 98.92/73.12/60.22; AS 24.73/17.20/13.98; paired diff +74.2/+55.9/+46.2 | [68.9, 78.3], [44.8, 65.0], [40.0, 50.9] | yes | none. Run 1395291 |
| Self-play vs AS differ in 13 of 15 cells | main.tex:815-816 | n/a | 13/15 paired intervals exclude 0; Opus r=0.6 (0.0 vs 1.1, [-3.2, 0.0]) and r=0.9 (0.0 vs 0.0) do not | n/a | yes | Wording: at r=0.9 Opus is a tie (0.0 vs 0.0), not a "reversal" |
| Gemini self-play r=0.1: 9 of 10 races all Unsafe | main.tex:818-819 | n/a | 9 | n/a | yes | none |
| Sonnet self-play | 89.2 / 48.4 / 32.3 (main.tex:837) | 0.3226 at 0.9 (v6) | 89.25 / 48.39 / 32.26 | [86.5, 91.2], [43.2, 53.5], [29.0, 35.7] | yes | 166/186 = 89.247, so it rounds to 89.2 |
| Graded-route risk response band; five-route mean; nine-route spread | 38.2 to 57.0; 54.8; 93.0 (main.tex:698, 700, 721) | n/a | 38.17 (GPT-5.5) to 56.99 (Sonnet); 54.84; 100.00 minus 6.99 = 93.01 | e.g. Sonnet [52.0, 60.8] | yes | none |
| Gemini risk response | 38.7 (main.tex:722) | 38.7 | 38.71 | [36.4, 41.5] | yes | The replication value 40.3 was not recomputed (out of scope) |
| Group size r=0.6 rates | 68.8/80.3/98.3/100.0 (supp:2980-2983) | same | 68.75/80.30/98.30/100.00 | [65.4, 72.6], [75.8, 83.8], [95.7, 100], [100, 100] | yes | Rate CIs within 0.3 pp of the paper's |
| Group size r=0.9 rates | 58.0/68.9/83.8/96.1 (supp:2986-2989) | same | 57.95/68.94/83.81/96.14 | [50.8, 63.6], [62.3, 74.9], [80.8, 86.1], [92.2, 98.5] | yes | Within 0.2 pp |
| Group size contrasts vs N=2, r=0.6 | +11.1, +28.2, +31.0 (supp:2981-2983, 3081-3084) | same | pooled +11.55, +29.55, +31.25 | [7.4, 14.7], [24.7, 33.0], [27.7, 34.7] | **N=4 no (1.4 pp)**; others within 0.5 | Estimator. Rep-mean gives 11.12 [6.3, 15.1], 28.18 [22.6, 33.2], 31.01 [26.4, 35.3], which matches |
| Group size contrasts vs N=2, r=0.9 | +11.0, +28.1, +40.0 (supp:2987-2989, 3081-3084) | same | pooled +10.98, +25.85, +38.18 | [7.1, 14.8], [22.2, 30.8], [33.1, 45.2] | **N=4 no (2.3 pp), N=5 no (1.8 pp)** | Estimator. Rep-mean gives 10.98, 28.07 [24.4, 32.1], 39.95 [33.1, 47.7], which matches. Rep 3 lasts 23 rounds against 5 to 13 for the others, so decision weighting leans on it |
| "N=3 and N=4 steps agree to a tenth of a point across risks" | supp:3081-3083; CLAUDE.md calls it "the strongest internal evidence" | same | pooled: N=3 +11.55 vs +10.98 (agree); N=4 +29.55 vs +25.85 (**3.7 pp apart**) | n/a | **only under rep-mean** | The N=4 agreement depends on the estimator. The same table prints rates whose differences are 29.5 and 25.8 |
| Code-swap effect (Q means Safe) | Gemini +9.5 [5.6, 13.8]; Sonnet +8.3 [5.4, 12.2] (main.tex:728-729; supp:1190-1191) | same | pooled Gemini +11.47, Sonnet +7.08 | [7.3, 15.5], [4.5, 11.5] | **no: 2.0 and 1.2 pp** | Estimator. Rep-mean over the 60 (rep x other factor x risk) blocks gives 9.55 [5.5, 13.9] and 8.35 [5.4, 12.3], which matches |
| Narrative-skin effect | "abstract contest" +10.8 [6.0, 15.6] Gemini; +3.4 [0.9, 7.4] Sonnet (supp:1193-1194); "race story lowers Unsafe by 3.4 and 10.8" (supp:3016-3017) | "the narrative skin **raises** it by 10.8 ... and 3.4" | abstract minus technology, pooled: Gemini +11.29, Sonnet +3.14 | [9.0, 14.0], [1.2, 6.6] | paper yes (within 0.5); **CLAUDE.md sign wrong** | The race story lowers Unsafe, as the paper says. CLAUDE.md line 55 has the direction reversed. The paper's CI treats 60 blocks as independent; resampling the 10 repetitions instead gives Gemini [8.8, 13.2] and Sonnet [1.1, 7.1] |
| Spearman rho, state reconstruction vs risk response, 9 routes | not in AAMAS paper; paper/acl/main.tex:709: 0.874, p=0.004; 0.922/0.003 without Opus (acl:710) | 0.87, p=0.004; 0.92 | 0.874, exact p=0.00377; without Opus 0.922, p=0.00268; aggregate score 0.770 | n/a | yes | none. Accuracy is recomputed from `semantic_correct`: state reconstruction 0.933, 1.0, 0.933, 0.733, 0.6, 1.0, 0.667, 0.2, 0.8 |

## Findings to act on

1. **Rates and contrasts use different estimators, and the tables print them side by side.** Every rate in the paper is decision-pooled. Every paired contrast (scripted rival stance, risk at Safe, group size, context/mapping) is the unweighted mean of per-race differences. Both are defensible. But a reader who subtracts two printed rates will not get the printed contrast: Opus 63.4 minus 0.0 is shown as +69.9, Gemini 100.0 minus 24.7 as +73.5, and 83.8 minus 58.0 as +28.1. Either name the contrast as a race-weighted estimand in each caption, or report contrasts on the same weighting as the rates. The ordering count, the "all fifteen lower bounds positive" claim, the Spearman result and all the structural facts hold under both estimators.
2. **The "agree to a tenth of a point" argument (supp:3081-3083) depends on the estimator.** Decision-pooled contrasts give N=4 at +29.5 and +25.8. Do not present the coincidence as the strongest internal evidence without saying which weighting produced it.
3. **The Opus r=0.1 figures are the most estimator-sensitive:** rival stance +30.1 vs +36.6, lower bound 16.5 vs 22.2, risk-at-Safe +69.9 vs +63.4. The "smallest lower bound +16.5" sentence (main.tex:785, 798) is an artifact of race weighting.
4. **CLAUDE.md line 55 reverses the direction of the skin effect.** The data and the paper both say the race story lowers Unsafe play.
5. **The context/mapping CIs resample 60 blocks as independent units**, although 6 blocks share each repetition. For the Gemini skin effect, a repetition-clustered interval is narrower ([8.8, 13.2] vs [6.0, 15.6]). The code-swap intervals barely change. This is conservative, but the text calls the unit a "repetition block".
6. Minor wording: main.tex:816 says Opus "reverses the direction" at r=0.6 and 0.9. At 0.9 both rates are exactly 0.0.
