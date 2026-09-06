# Two-agent report exports

Kaggle GPU run of two open-source checkpoints (`qwen2.5-14b-instruct`,
`gemma-3-12b-it`) plus the cross-model tables built from it. Provenance is in
`run_manifest.json`: source dataset, `source_sha256`, and the package versions
the run executed against.

| file | contents |
|---|---|
| `run_manifest.json` | run provenance: input dataset, source hash, package versions, model paths |
| `ai_race_all_models.csv` | per-race rows pooled across the checkpoints in this run |
| `ai_race_players_all_models.csv` | per-player rows pooled across the same checkpoints |
| `qwen2.5-14b-instruct/baseline/` | the self-contained run directory: `turns.jsonl`, `races.csv`, `players.csv`, `all_results.csv`, `run_manifest.json` |

These arrived on the `results/visualize` lane and originally sat in a top-level
`output/` directory. That is a retired root — `results/migration_manifest.json`
asserts it is absent and `results/README.md` names `results/` the single
canonical home — so they were moved here rather than left outside the tree the
catalog indexes.

Regenerate the report tables with
[`scripts/report_two_agent_race.py`](../../../scripts/report_two_agent_race.py);
its `--output` is a required argument, so point it at a directory under
`results/` rather than back at the repository root.
