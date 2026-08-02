# AI Race impact-upgrade package

This directory is the current cross-study reporting surface. It keeps provider
pilots, diagnostic context runs, XAI associations/interventions, and validated
method reconstructions in explicit evidence classes rather than pooling them.

## Open first

- `impact_report.html` — validated self-contained technical report
- `impact_report.md` — plain-text report source
- `figures/` — paper-ready PNG and vector PDF figures
- `data/` — source-backed tables and the trajectory-demo payload
- `analysis_manifest.json` — hashes and quality-gate receipt
- `release_manifest.json` — hashes for the complete report/demo/paper/deck surface
- `data_quality_audit.json` — compact coverage audit
- `experiment_gap_audit.md` — ranked next-run roadmap and promotion rules
- `xai_claim_audit.md` — association-versus-causation boundary for FAST-SAE
- `data/experiment_priority.csv` — machine-readable experiment queue

Interactive demo:
[`docs/demos/trajectory_lab/index.html`](../../docs/demos/trajectory_lab/index.html).

The fully crossed mapping follow-up is frozen in
[`docs/experiments/context_mapping_fully_crossed_protocol.md`](../../docs/experiments/context_mapping_fully_crossed_protocol.md)
with its launch runner at
[`kaggle/experiments/greennode_context_mapping_cross.py`](../../kaggle/experiments/greennode_context_mapping_cross.py).

## Rebuild

```bash
python results/scripts/analyze_impact_upgrade.py
python results/scripts/build_impact_report.py
python results/scripts/build_trajectory_demo.py
node <data-analytics-plugin>/skills/build-report/scripts/build_portable_artifact.mjs \
  --input results/impact_upgrade/artifact.json \
  --output results/impact_upgrade/impact_report.html
python results/scripts/fix_portable_report_overflow.py \
  results/impact_upgrade/impact_report.html
python results/scripts/build_release_manifest.py
python results/scripts/build_release_manifest.py --check
```

The final HTML is verified at 1440 px and 390 px with the portable artifact
verifier. The small overflow post-process compensates for Windows scrollbar
width interacting with the canonical reader's `100vw` sticky header.
