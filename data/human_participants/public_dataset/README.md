# AI Race experiment — public dataset

Data underlying: **"When Competitors Take Risks: Unsafe Behaviour in an
Idealised AI Race Experiment"** (Fernández Domingos & Han).

This folder contains a de-identified version of the experimental dataset
used to produce the paper's quantitative results (Figures 2, 3, S1; Table 1
and all SI regression/summary tables). See [Scope and limitations](#scope-and-limitations)
below for what this file does *not* cover.

## Files

- `airace_deidentified_long.csv` — the dataset. One row per
  participant-round (long format), 3,245 rows, 341 participants, 19 columns.
- `build_dataset.py` — the script that produced this file from the internal
  analysis repository's canonical dataset, included for transparency. It is
  not runnable outside that repository (the source path is local), but
  documents every transformation exactly.
- This `README.md`.

## Study design (brief)

Participants were matched in pairs and repeatedly chose between Safe and
Unsafe technological development in a framed two-player "AI race" with a
stochastic stopping rule (minimum 5 rounds, 20% chance of ending each round
thereafter). Unsafe development advanced the race faster and paid more per
round, but accumulated a private risk of a "setback" (losing all task
earnings) capped at a treatment-specific maximum, $p_r^{\max} \in
\{0.1, 0.6, 0.9\}$. Full task description: paper Methods and SI
§S1.3 (experimental instructions).

## Field dictionary

| Column | Type | Description |
|---|---|---|
| `participant_id` | string | Anonymous participant identifier (`P0001`–`P0341`). Freshly assigned for this deposit; does not correspond to any internal study ID. |
| `group_id` | string | Anonymous pair/game identifier (`G0001`–`G0173`). Two participants share a `group_id` if they were matched together. |
| `round_number` | int | Round index within the game, starting at 1. |
| `max_private_risk` | float | Treatment: maximum private setback risk, $p_r^{\max} \in \{0.1, 0.6, 0.9\}$. |
| `decision` | 0/1 | The participant's own choice this round (0 = Safe, 1 = Unsafe). |
| `decision_opponent` | 0/1 | The opponent's choice this round. |
| `decision_lag` | -1/0/1 | The participant's own choice in the *previous* round ($a_i^{t-1}$). `-1` in round 1, where no lag exists. |
| `decision_opponent_lag` | -1/0/1 | The opponent's choice in the previous round ($a_{-i}^{t-1}$). `-1` in round 1. |
| `acc_steps` | float | Own cumulative race progress through this round. |
| `acc_steps_opponent` | float | Opponent's cumulative race progress through this round. |
| `acc_steps_lag` | float | Own cumulative race progress through the *previous* round. |
| `acc_steps_opponent_lag` | float | Opponent's cumulative race progress through the previous round. |
| `delta_steps_lag` | float | $\Delta S_{t-1}$ = `acc_steps_lag` − `acc_steps_opponent_lag`, the race-position gap entering this round (positive = participant was ahead). Reported *uncentred* here; the paper centres this on the analysed sample's mean before regression — see [Reproducing the analysed sample](#reproducing-the-analysed-sample-table-1). |
| `won_race` | 0/1 | Whether the participant won (or tied for) the race overall. Constant across all rows for a given `participant_id`. |
| `num_rounds` | float | Total number of rounds $W$ played in this game. Constant across all rows sharing a `group_id`. |
| `sex` | string | `Female` / `Male` / `Prefer not to say` / `CONSENT_REVOKED`. Self-reported via Prolific. |
| `age` | string | Age in 5-year bins (`18-22`, `23-27`, ..., `68-72`) or `CONSENT_REVOKED`. See [De-identification](#de-identification) for why this is binned rather than exact. |
| `nationality_group` | string | `South Africa` / `Poland` / `Other` / `DATA_EXPIRED` / `CONSENT_REVOKED`. Collapsed from the participant's raw nationality — see [De-identification](#de-identification). |
| `risk_gamble_choice` | float | Risk-preference score (0–5) from the pre-task Eckel–Grossman gamble-choice elicitation (SI §S1.3.2); higher = the participant chose a higher-payoff-risk gamble, i.e. is less risk-averse by this measure. Constant across all rows for a given `participant_id`. |

### Sentinel values: `DATA_EXPIRED` and `CONSENT_REVOKED`

Two participants (both in the $p_r^{\max}=0.9$ treatment) have
`nationality_group = DATA_EXPIRED`: their Prolific demographic-data snapshot
had expired at the time of export. **This is unrelated to task
completion** — both participants finished the AI-race game in full (all
rounds, matching their partner's game length exactly), so their behavioural
data (`decision`, `acc_steps`, etc.) is complete and valid. They are
included in every result that doesn't require nationality as a covariate,
and excluded only from the specific regressions that do (see the mapping
table below) — this mirrors exactly what the paper's own analysis code
does, and is deliberate: dropping them from the dataset entirely would be
an unjustified, ad hoc exclusion unrelated to data quality.

One participant has `sex = CONSENT_REVOKED` and `age = CONSENT_REVOKED`
(a Prolific participant who withdrew consent for their personal data after
the study; their behavioural game data is retained per standard practice,
but their withdrawn demographic fields are not populated). Same handling:
excluded only from analyses requiring that specific field.

## De-identification

This file excludes `prolific_id` and every other raw Prolific
identifier/administrative field (timestamps, completion codes, IP-adjacent
approval counts, etc.) — none of these were used in any paper result. It
also excludes the free-text and Likert post-experiment survey responses,
which are not analysed in the paper and which free text in particular could
itself be identifying.

Two further transformations were applied to reduce re-identification risk
in the fields that *are* used in the paper's analyses, given the sample is
only 338 analysed participants concentrated in a small number of countries
(South Africa and Poland alone account for 44% and 11% of the sample):

- **Nationality** is collapsed to `South Africa` / `Poland` / `Other`
  (plus the two sentinels above) instead of the raw ~46-country field. This
  matches exactly the categorisation the paper's own regressions use
  (`nationality_cat` in the analysis code) — no analytical content is lost,
  since the paper never uses finer-grained nationality than this.
- **Age** is binned into 5-year ranges instead of reported exactly. The
  paper uses age as a covariate (`age_c`, centred) in every regression, and
  reports it is never a significant predictor; binning preserves the
  ability to closely reproduce this (null) finding while removing exact
  age as a quasi-identifier.
- `participant_id` and `group_id` are freshly assigned, randomly permuted
  identifiers — they do not correspond to any internal oTree/Prolific code
  and cannot be used to look up a participant in any other export.

See the paper's Data Availability statement and SI §S1.2.1 for the
manuscript-facing version of this rationale.

## Reproducing the analysed sample (Table 1)

The paper's main regression table (Table 1 / `tab:cluster_logit_unsafe`)
uses round $t\geq2$ (round 1 is used only to define the first-round-action
covariate) and requires every covariate to be non-missing. To reproduce
that sample ($N=2{,}888$ observations, 172 pair clusters, 338
participants) from this file:

1. Keep only `round_number > 1`.
2. Drop rows with `sex == "CONSENT_REVOKED"`.
3. Treat `nationality_group` values `"DATA_EXPIRED"` and
   `"CONSENT_REVOKED"` as missing, and drop rows where it's missing.
4. Drop any remaining rows with a missing value in `decision`,
   `decision_lag`, `decision_opponent_lag`, `delta_steps_lag`,
   `risk_gamble_choice`, or `age`.
5. Centre `delta_steps_lag` and (numeric) `age` on their sample means
   within this filtered set before regression, matching the paper's
   `age_c` / $\Delta S_{t-1}$ (centred) covariates.

We verified this procedure, applied to this file alone, reproduces the
paper's reported sample sizes and pairwise treatment-comparison statistics
exactly (participant counts 97/105/136 by treatment; Cohen's $d=
0.341/0.322/-0.027$ for the three pairwise comparisons).

## How this file produces each paper result

| Paper result | Columns / procedure |
|---|---|
| **Fig. 2A** (Unsafe frequency by treatment, pairwise $t$-tests) | Per-participant mean of `decision` across **all** rounds (no round filter), grouped by `max_private_risk`; independent-samples $t$-tests, Bonferroni-corrected across the 3 pairwise comparisons. Uses the raw, unfiltered participant set (341 participants; group sizes 98/105/138), *not* the round-$\geq2$ analysed sample. |
| **Fig. 2B** (Unsafe choice vs. $\Delta S_{t-1}$, own/opponent lag) | `decision` as outcome; `delta_steps_lag`, `decision_lag`, `decision_opponent_lag` as predictors, on the round-$\geq2$ analysed sample (see above). |
| **Fig. 2C** (winners vs. losers average Unsafe frequency) | Per-participant mean `decision` across all rounds, split by `won_race`, aggregated to the pair level via `group_id`. |
| **Fig. 3B** (experimental vs. model median $\phi_U$) | Median of per-participant mean `decision` (all rounds) by `max_private_risk`, compared against the reduced evolutionary model's predictions (model output not in this file — see main analysis repository). |
| **Fig. S1** (rounds-per-game distribution) | One row per distinct `group_id`, using `num_rounds`. $N=173$ games (the raw pair count, before any participant-level exclusion). |
| **Table 1 / `tab:cluster_logit_unsafe`** | Cluster-robust logistic regression of `decision` on `max_private_risk`, `decision_lag`, `decision_opponent_lag`, `delta_steps_lag` (centred) and their interactions, plus `sex`, `age` (centred), `nationality_group`, `risk_gamble_choice`; clustered by `group_id`. Round-$\geq2$ analysed sample (see above). |
| **Table S1** (pre-registered mixed-effects model) | A *different* specification from Table 1, not "the same predictors with a random intercept": `max_private_risk` enters as a **centred continuous** score (not the 3-level categorical dummy Table 1 uses), **no sex/age/nationality covariates**, and an added `risk_gamble_choice × max_private_risk` interaction that Table 1 never fits. Random intercept by `group_id` in place of cluster-robust SEs. See `code/si_mixed_model.R` in this repository for the exact formula. |
| **Table S1 (no-dropout robustness)** | Same as Table 1, with the single pair affected by a mid-race decision timeout removed. Incomplete games (a decision timeout cut the game short) show up as a `group_id` whose observed maximum `round_number` is *less than* `num_rounds` (the game's assigned length) — but note this file contains more than one such incomplete game, and only one of them has a member who otherwise passes the round-$\geq2$ covariate-complete filter (the rest are already excluded from that panel for other reasons, e.g. a missing covariate). The pair to drop for this robustness check is specifically that one — the incomplete game whose member *is* present in the Table 1 analysed sample — not just any incomplete game. |
| **Summary statistics table** (`tab:si:summary-stats`) | Participant/pair counts, `sex`, `age`, `nationality_group`, mean `decision`, and `risk_gamble_choice` distributions, by `max_private_risk`, on the round-$\geq2$ analysed sample. |
| **Demographic/risk-preference covariate tables** (`tab:si:panel-covariates`) | The `sex`, `age`, `nationality_group`, `risk_gamble_choice` coefficients from the same Table 1 regressions. |

### Scope and limitations

This file does **not** reproduce Table S2 (pre-race exclusion counts:
comprehension-test failures, participants who were never matched with a
partner, etc.) — those individuals never entered the AI-race task and so
generated no rows in this dataset. Table S2 is reported as an aggregate
count table in the paper's SI and does not require row-level data to be
verified in the way the figures/tables above do.

Figure 3's evolutionary-model predictions (the non-experimental half of
that figure) are simulation output, not participant data, and are not
included here.
