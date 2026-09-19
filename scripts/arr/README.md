# scripts/arr

Analysis and figures for the ARR evaluation paper (`paper/acl/`). The plan they
serve is `docs/acl-audit-paper-plan.md`, sections 3, 5.3 and 5.4. Nothing here
builds or touches the AAMAS paper.

| script | writes | what it answers |
|---|---|---|
| `analyze_screen_baselines.py` | `results/derived/arr_screen_baselines/` | E14a single probe, E14b one domain or overall alone, E14c parse health, E13 route name, E12 threshold band, repetition reliability, E21 per route opening seat gap |
| `fig_domain_by_route.py` | `figures/arr/domain_by_route.pdf` and `.png` | accuracy per route per domain, admitted routes above refused ones |

```bash
python scripts/arr/analyze_screen_baselines.py
python scripts/arr/fig_domain_by_route.py
```

Both read the raw probe rows through one loader, so the figure and the tables
refuse the same inputs. The loader stops the run when:

- a route has a row count other than `--expected-rows` (default 60);
- `probe_bank_sha256` or `rules_context_sha256` differs across routes;
- the per domain counts or the admit flag recomputed from the raw rows differ
  from the route's `admission.json`;
- the same route appears under two campaign roots;
- any path under `failed_runs/` would be opened. The directory walk never
  enters `failed_runs/`, and the file reader refuses such a path as a second
  line of defence.

## Adding routes or task families

New routes: pass another campaign directory. Every `raw_responses.jsonl` below
it is read, and each needs a sibling `admission.json` and `run_manifest.json`.

```bash
python scripts/arr/analyze_screen_baselines.py \
  --campaign-root results/frontier/admission_campaign_v6 \
  --campaign-root results/frontier/<new_campaign>
```

New families (F2, F3): use the same row schema plus a `"family"` field, put a
matching `"family"` in `admission.json`, and select the family by label. Rows
without the field are the original race bank and count as F1. Outputs for a
family other than F1 go to `results/derived/arr_screen_baselines/<label>/` and
`figures/arr/domain_by_route_<label>.pdf`.

```bash
python scripts/arr/analyze_screen_baselines.py --campaign-root <root> --family-label F2
python scripts/arr/fig_domain_by_route.py --campaign-root <root> --family-label F2
```

The gates are read from `admission_thresholds` in `admission.json`: any
`<domain>_accuracy_min` is a gate unless `<domain>_is_diagnostic_only` is true.
A new family therefore needs no code change as long as its thresholds follow
that naming. E14c's gameplay parse count and E21 exist only for F1, because
only the race has gameplay turns.

## Fixed choices

- Bootstrap for E21: races resampled with replacement, 2000 draws, seed
  20260918, reseeded per route so one route's interval cannot move when
  another route is added. `--n-boot` and `--seed` change them.
- The route name rule is reported two ways: the literal regex
  `(mini|nano|lite)`, which also matches inside `gemini`, and the whole word
  rule used by `scripts/figures/fig_audit_reads_the_name.py`.
- E12 is reported on three grids: the decimal grid of the recorded thresholds
  plus or minus 0.05 in steps of 0.01, the integer box built by
  `scripts/verify_manuscript_claims.py` (the source of the "676"), and the
  achievable levels within 5 points of each threshold.

## Transfer campaign (F1, F2, F3)

The preregistered analyses of `docs/arr-transfer-campaign-prereg-2026-09-19.md`
(sections 3, 4, 6, its amendments and its collection outcome).

| script | writes | what it answers |
|---|---|---|
| `analyze_transfer.py` | `results/derived/arr_transfer/` | primary: N1/N2 name baselines on F1 with exact permutation p; Spearman transfer of state_reconstruction and overall accuracy between families (permutation p, bootstrap CI) and verdict agreement with Cohen's kappa; cheap baselines per family; repetition reliability. Exploratory: item accuracy against arithmetic term count, long and wide accuracy tables |
| `fig_transfer.py` | `figures/arr/transfer_heatmap`, `transfer_scatter`, `item_load` (`.pdf` and `.png`) | the heatmap and the family-vs-family scatter over the routes complete in all three families; item accuracy against term count |

```bash
python scripts/arr/analyze_transfer.py
python scripts/arr/fig_transfer.py
```

The figure script reads only the analyser's outputs, so run the analyser first.

Inputs. F1 is `results/frontier/admission_campaign_v6/` (the nine v6 routes)
plus `results/frontier/arr_transfer_campaign/F1/`; F2 and F3 are
`arr_transfer_campaign/F2/` and `F3/`. Cells load through
`analyze_screen_baselines.load_campaign`, so every check listed above applies.
The raw rows carry no `"family"` field, so each family is loaded from its own
roots and checked against its manifest's `family` instead. On top of that the
analyser refuses when:

- a family's `probe_bank_sha256` is not the frozen value (F1 `2953fb47...`,
  F2 `1cf954f9...`, F3 `31ff7c47...`), or the bank text in the task source
  (`kaggle/benchmarks/ai_race_frontier_admission.py`, `arr/families/*/bank.py`)
  or the family README gives another hash;
- a route is complete twice in one family, or a directory name differs from the
  normalised route slug (`model_route` without provider, `@` replaced by `-`);
- a transfer-campaign cell is not a `completed`, `admitted_to_analysis=True`
  row of `ledger.csv`, or such a ledger row has no cell.

Failure records are tabulated from `ledger.csv` only; `failed_runs/` is never
opened.

Fixed choices. Seed 20260919 for every permutation and bootstrap, reseeded per
test so no number moves when another test is added. Spearman uses average ranks.
p is exact over all permutations when n <= 10, else 100,000 random permutations
with (b+1)/(N+1). The 95% interval is a percentile bootstrap over routes (10,000
resamples); resamples where one variable is constant have no rho and are dropped
and counted in `transfer_correlations.csv`. Kappa is reported as undefined, with
the reason, when a family's verdict is constant over the routes compared. The
arithmetic term count rule and a one-line justification per probe are in
`item_term_counts.csv` for human audit; the item analysis is exploratory.
