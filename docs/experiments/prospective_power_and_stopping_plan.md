# Prospective precision, power, and stopping plan

## Decision

The frozen 32-stream context × mapping run remains a **diagnostic replication**.
It is not relabelled confirmatory after seeing its result. A separate future
confirmatory run uses 96 independent CRN repetition streams, targeting a
15-percentage-point context × mapping interaction with at least 80% design
power under the conservative pilot-based sensitivity analysis.

## Why risk is not a replication unit

Every risk treatment uses the same `base_seed + repetition` environment stream.
The independent resampling unit is therefore `repetition`; the three risk
strata and both player roles remain inside that block. Treating `(risk,
repetition)` as independent would turn 32 streams into 96 artificial clusters
and make intervals too narrow.

## Prospective inputs

- Primary family: seven context-versus-abstract mapping interactions.
- Familywise error: 0.05, Holm corrected.
- Smallest effect of scientific interest: 15 percentage points in semantic
  Unsafe rate.
- Target power: 0.80.
- Invalid-output allowance: entire paired cells are excluded; coverage loss is
  reported and never replaced with extra post-hoc repetitions.
- Stopping: fixed N. No optional continuation, peeking, model substitution, or
  sample-size re-estimation from the diagnostic mapping result.

The completed parity-confounded context pilot cannot identify the variance of a
fully crossed mapping interaction. `simulate_context_mapping_power.py` therefore
uses the largest observed repetition-level context-delta standard deviation as
an explicit conservative proxy, bootstraps its centered residuals, and shows
power across 16–192 streams and 5–20 pp effects. This is a design sensitivity,
not a new behavioral finding.

## Execution labels

- 32 streams: preregistered diagnostic; useful for checking direction,
  pipeline integrity, and planning the independent replication.
- 96 streams: frozen confirmatory target for a separate run and exact model
  digest, conditional on comprehension admission.
- Temperature 0 first-round repetitions are environment trajectories, not
  independent stochastic model draws. A temperature-0.7 robustness stratum
  requires separately frozen call IDs and is never pooled with temperature 0.
