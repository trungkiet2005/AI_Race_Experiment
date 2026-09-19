# ARR transfer campaign: preregistration (2026-09-19)

Written before any F2 or F3 probe has been sent to any endpoint and before any
new route has been administered F1. It fixes the roster, the name baselines, the
analysis unit, the identity assignment and the decision rules that the ARR
paper (`paper/acl/main.tex`) will report. Plan context:
`docs/acl-audit-paper-plan.md` sections 5.1 and 5.2. Target: ARR October 2026
cycle, submission deadline 12 October 2026.

Any change after the first evidence run is an amendment appended at the bottom
with its date and reason, never an edit above this line.

## 1. Families

| family | task | protocol id | state the model must track |
|---|---|---|---|
| F1 | `ai-race-frontier-admission` (frozen, unchanged bank) | `ai-race-frontier-admission-v6` | own and rival progress, private risk in a two-player race |
| F2 | `strategic-state-probe-f2` | `strategic-state-probe-f2-v1` | the opponent's history in a repeated social dilemma |
| F3 | `strategic-state-probe-f3` | `strategic-state-probe-f3-v1` | an aggregate over four players in a repeated public goods game |

All three share: 20 probes with identical domain counts (rule_recall 4,
stage_payoff 2, state_reconstruction 5, state_transition 2, terminal_scoring 5,
expected_payoff 2), three repetitions, temperature 0, 256 output tokens, the
same scorer, the same seed formula, the same admission thresholds (overall
>= 0.80, state_reconstruction >= 0.75, terminal_scoring >= 0.75;
expected_payoff diagnostic only). F2 and F3 answers are computed by a reference
engine and checked by `arr/families/*/test_bank.py`; the bank and rules hashes
are recorded in each family README before the first run.

## 2. Roster

The nine F1 v6 routes (already administered F1) plus fifteen new routes chosen
to stress the name/verdict tie rather than to extend it: small models whose
names carry no `mini|nano|lite` token, older generations of admitted families,
and open-weight routes.

| route slug | why it is here |
|---|---|
| `claude-haiku-4-5-20251001` | small tier, no tier word in the N1 sense |
| `claude-sonnet-4-20250514` | older generation of an admitted family |
| `claude-sonnet-4-5-20250929` | intermediate generation |
| `claude-opus-4-1-20250805` | older large tier |
| `gemini-2.5-flash` | older generation of an admitted family |
| `gemini-2.5-pro` | older large tier |
| `gemini-3.5-flash` | same family as a refused `-lite` route, without the word |
| `gemma-4-26b-a4b-it` | open weights, 4B active parameters, no tier word |
| `gemma-4-31b-it` | open weights, dense |
| `gpt-oss-20b` | open weights, small, no tier word |
| `gpt-oss-120b` | open weights, larger sibling |
| `gpt-5.6-terra` | newer generation of an admitted family |
| `qwen3-235b-a22b-instruct-2507` | non-US open-weight family |
| `glm-5` | non-US family |
| `grok-4.20-0309-non-reasoning` | fourth provider, non-reasoning variant |

Routes that only run in a mandatory thinking mode (`gemini-3.1-pro-preview`,
`*-thinking`, `grok-*-reasoning`, `deepseek-r1-0528`) are excluded, because under
the shared 256-token cap their reasoning can consume the whole allowance
(failure class 3 in the project's failure taxonomy), which would score the cap
rather than the model.

**Reachability smoke.** Before any evidence run, each new route receives one
connectivity request (`kaggle/benchmarks/connectivity_ping.py`) to resolve its
reasoning-parameter contract, as done for v6 on 2026-09-09. A smoke produces no
probe answer and is never evidence. If a route rejects the recorded contract, the
route-resolved reasoning rule of the 2026-09-09 amendment applies (omit or set
`low`), identically in all three family tasks, and the route's contract is
flagged in every table. A route that cannot be reached under any contract is
listed as unreachable, not as refused.

## 3. Name baselines, fixed now

Matching is on whole tokens: the lower-cased slug is split on `-`, `_`, `.`,
`/` and `@`, and a word matches only a complete token. A substring rule would
match the `mini` inside `gemini` and misclassify every Gemini route.

- **N1 (primary):** a route is predicted refused iff one of its tokens is
  `mini`, `nano` or `lite`. This is the rule used by
  `scripts/figures/fig_audit_reads_the_name.py` and it reproduces all nine F1
  v6 verdicts (exact p = 1/126).
- **N2 (secondary, broader tier words):** predicted refused iff a token is
  `mini`, `nano`, `lite`, `haiku` or `gemma`, or the slug contains the token
  pair `oss`, `20b`.

The tie is **broken** for a baseline if at least one route's F1 verdict differs
from that baseline's prediction. Both baselines are reported on all 24 routes
with their agreement count and the exact permutation probability of the observed
agreement given the observed number admitted.

## 4. Primary analyses (fixed now)

1. **Name baselines** (section 3) on F1 over 24 routes.
2. **Transfer (RQ4):** Spearman rank correlation of `state_reconstruction`
   accuracy between each pair of families over the routes that completed all
   three, with an exact permutation p-value if n <= 10 and 100,000 random
   permutations otherwise; the same for overall accuracy. Also the verdict
   agreement between families (count of routes with the same admit/refuse
   verdict) and Cohen's kappa.
3. **Cheap baselines (RQ3)** on each family: single probe, each domain alone,
   parse health, N1, N2; agreement with that family's verdict.
4. **Reliability (RQ2):** repetition agreement per route per domain.

No analysis will pool families into a single score. No route is dropped after
seeing its answers.

## 5. Analysis unit and identity assignment

The unit is one (family, route) cell: 60 probe calls, complete or not collected.
Each identity pushes one family task once (the push starts one validation run of
60 calls on the default route; that run is never downloaded into the tree) and
then runs its assigned routes. Assignment, fixed before the first run:

| identity | family | routes |
|---|---|---|
| `account-F1a` | F1 | first 8 new routes in section 2 order |
| `account-F1b` | F1 | last 7 new routes |
| `account-F2a` | F2 | the 9 v6 routes + first 3 new routes |
| `account-F2b` | F2 | remaining 12 new routes |
| `account-F3a` | F3 | the 9 v6 routes + first 3 new routes |
| `account-F3b` | F3 | remaining 12 new routes |

**Backup order**, used only when an identity returns an HTTP 403 quota refusal:
`account-R1`, `account-R2`, `account-R3`, `account-R4`. The refused cell is
collected complete on the next backup identity; its partial rows are kept under
`failed_runs/` with the refusal text and never tabulated. A transport or auth
error is retried on the same identity. A reasoning-cap exhaustion is recorded as
such and not retried. The collecting identity of every cell is written to a
receipt outside the paper; the paper reports only that cells were collected on
several identities.

## 6. Decision rules for the paper's framing (fixed now)

- If N1 is broken on F1: the paper argues the screen carries information beyond
  the name, reports how much, and asks whether it transfers.
- If N1 is not broken on 24 routes: the paper's claim becomes "in current
  commercial families this screen is not separable from model tier", and it
  reports that as the finding, together with the transfer result.
- If state reconstruction does not transfer across families (rank correlation
  interval includes zero): the paper reports that a screen must be rebuilt per
  environment. That is a result, not a failure.

## Amendments

### 2026-09-19, before any evidence run: route contracts from the smoke

`kaggle/benchmarks/route_contract_smoke.py` ran once on each of the fifteen new
routes (identity `account-F1a`, one arithmetic question unrelated to any bank,
contracts tried in the order `none`, `low`, omitted, `high`). Outcome:

| contract | routes |
|---|---|
| `none` (same as v6) | claude-haiku-4-5, claude-sonnet-4-5, gemini-2.5-flash, gemini-3.5-flash, gpt-5.6-terra |
| `low` | glm-5, qwen3-235b-a22b-instruct-2507 (both already `low` in the v6 rule), gpt-oss-20b, gpt-oss-120b |
| argument omitted | gemma-4-26b-a4b-it, gemma-4-31b-it (already omitted in v6), grok-4.20-0309-non-reasoning |
| unresolved: HTTP 503 on every contract | claude-opus-4-1-20250805, claude-sonnet-4-20250514 |
| excluded | gemini-2.5-pro: `none` is refused (HTTP 400) and every other contract exhausts the 256-token cap before an answer (failure class 3), i.e. a mandatory-thinking route under section 2 |

Consequences, applied identically to the F1, F2 and F3 task sources:
`gpt-oss` routes request `low`, `grok` routes omit the argument. No rule that
applies to a v6 route changed, so every v6 route still sends a byte-identical
request, and the F1 probe bank and rules hashes are unchanged
(`2953fb47...`, `b80981a5...`).

The two 503 routes stay in the roster. A 503 is a transport failure, so each
gets one full retry of its cell on the same identity; if it fails again it is
reported as unreachable, never as refused. The roster is therefore 14 new routes
plus 9 v6 routes, at most 23, and the section 5 assignment is unchanged apart
from `gemini-2.5-pro` being dropped from the `account-F1a` list.

### 2026-09-19, before any evidence run: push validation guard

The first launch of all six blocks failed at push time, before any roster route
was called: `kaggle b t push` now validates a new task version on the platform
default route `google/gemini-3.7-flash`, whose reasoning consumed 242 of the 256
output tokens (failure class 3), so task creation errored and no run could be
scheduled. The three task sources now carry an identical `ROSTER_SLUGS` list of
the 23 preregistered routes and return `status: skipped` without sending any
probe when the executing route is not on it. Roster routes are unaffected: the
probe loop, contract, scorer and hashes are byte-identical, and the validation
run no longer spends 60 requests per push.

## Collection outcome (2026-09-19, recorded before the analysis was run)

41 complete cells: F1 8 new routes, F2 17 routes, F3 16 routes. 16 routes are
complete in all three families; `gpt-oss-20b` is complete in F1 and F2 only.
Failure records, none tabulated as evidence and none counted as refused
(`results/frontier/arr_transfer_campaign/ledger.csv`):

- transport, after the one preregistered retry: `claude-opus-4-1-20250805` and
  `claude-sonnet-4-20250514` (HTTP 503 "model is currently not reachable" on
  every attempt, all three families): unreachable.
- reasoning cap exhausted at 256 tokens: `glm-5` (all families),
  `gpt-oss-120b` (all families), `gpt-oss-20b` (F3 only).
- structured output not honoured: `gemma-4-26b-a4b-it`, `gemma-4-31b-it`
  (all families).
- `grok-4.20-0309-non-reasoning` F1 first failed on HTTP 429 and completed on
  the retry.

## Exploratory arm, declared 2026-09-19 before it was run: room to work

Not part of the preregistered primary analyses; it is reported as exploratory.
After seeing that item accuracy falls with the number of values an answer must
combine, we ask whether that is a limit of the models or of the one-shot
answer contract. The three banks are re-administered unchanged (same questions,
same hashes, same scorer, same thresholds, same reasoning-parameter rule) with
two changes only: the structured answer gains a `working` field that comes
before `answer`, and the output cap is 768 tokens instead of 256. Tasks:
`strategic-state-probe-f{1,2,3}-scratch`, protocol ids
`strategic-state-probe-f{1,2,3}-scratch-v1`.

Eight routes, chosen to span the F1 range and all complete in all three
families: gemini-3-flash-preview, gpt-5.4-2026-03-05, claude-sonnet-5-default,
claude-haiku-4-5-20251001, gemini-3.5-flash-lite, gpt-5.4-nano-2026-03-17,
qwen3-235b-a22b-instruct-2507, grok-4.20-0309-non-reasoning.

Assignment (first four routes / last four routes): F1 account-R1 / account-R2;
F2 account-R3 / account-R4; F3 account-R5 / account-R6. Same retry and failure
rules as section 5. Analyses fixed now: per family, the number admitted and the
per-domain accuracy under both contracts for these eight routes; the paired
change in state_reconstruction accuracy per route; and whether item accuracy
still falls with the item's term count.

Status 2026-09-19: declared, not run. No scratch-arm request has been sent.

## Redaction note (2026-09-19)

Collection identities are written as neutral labels (`account-F1a` ... `account-R6`) because this repository is public. The label-to-account map is kept outside the repository with the credential inventory. No assignment, route or rule changed.
