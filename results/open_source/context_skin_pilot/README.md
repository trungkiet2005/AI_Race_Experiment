# Context-skin pilot results

The primary reader-facing artifact is the temperature-zero report
[`analysis_live_pilot_t0/context_skin_analysis.md`](analysis_live_pilot_t0/context_skin_analysis.md).
The separate decoding audit is
[`analysis_temperature_robustness/temperature_robustness_report.md`](analysis_temperature_robustness/temperature_robustness_report.md).
The latter pairs T=0 and T=.7 on common environment seeds but never pools them.

## Accepted result sets

- `live_pilot_t0/`: primary 8-context live simulation, 32 repetitions x 3 risk
  cells per context, 768 races and 13,680 decisions at temperature 0.
- `live_pilot_t07/`: separate decoding-sensitivity simulation with the same
  coverage at temperature 0.7.
- `fixed_state_pilot_t0/`: 96 shared engine-reachable states x 8 contexts x 2
  opaque action mappings, 1,536 paired replay rows at temperature 0.
- `analysis_live_pilot_t0/` and `analysis_live_pilot_t07/`: protocol-specific
  validated summaries and publication figures.
- `analysis_temperature_robustness/`: paired T=0 versus T=.7 summaries, 10,000
  race-cluster bootstrap replicates, and five figures in PNG/PDF.
- `context_recognition_v2_t0_confirm/`: frozen corrected recognition audit;
  16/16 strict-valid generic-resemblance reports and no specific named match.
- `context_recognition_t0_pilot/`: rejected v1 retained unchanged because its
  prompt/parser contract was contradictory.

The `smoke_t0/`, `fixed_state_smoke_t0/`, and `analysis_smoke_t0/` directories
are retained as protocol-development evidence, not as primary estimates.

## Evidence boundary

The admitted manifests passed model-digest, mechanism/configuration,
artifact-hash, rectangular-coverage, CRN, and parse checks. For the temperature
comparison, mechanism-specific hashes match but the whole staged-source archive
hash differs, so the provenance warning remains in the report. The comprehension
admission gate failed, especially for state update and terminal scoring.
Therefore all behavior remains exploratory diagnostic evidence; it does not
show verified utility understanding or a causal neural mechanism.

Regenerate the accepted analysis with:

```bash
python results/scripts/analyze_context_skin.py \
  --live-root results/open_source/context_skin_pilot/live_pilot_t0 \
  --fixed-root results/open_source/context_skin_pilot/fixed_state_pilot_t0 \
  --output-dir results/open_source/context_skin_pilot/analysis_live_pilot_t0

python results/scripts/analyze_context_temperature_robustness.py
```
