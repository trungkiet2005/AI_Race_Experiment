# Pilot insight report package

This folder stores the generated outputs for the pilot-analytics visualization pipeline.

## Files

- `artifact.json`: machine-readable source for portable report rendering
- `visualization_audit.md`: pointer to the canonical consolidated audit file

Core consolidated visualization insight log:
- [results/visualization_insight_full.md](/AI_Race_Experiment/results/visualization_insight_full.md)

## Rebuild command

```bash
python results/scripts/build_pilot_insight_report.py
```

## Notes

- Keep only artifacts that are reproducible from script output.
- Temporary render debug files are ignored by `.gitignore`:
  - `candidate.html`
  - `fail.png`
  - `*.verification-failure.png` generated under `docs/reports/...`
