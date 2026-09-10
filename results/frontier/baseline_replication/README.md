# Independent repeat of one baseline cell

`google/gemini-3-flash-preview` was collected twice under the frozen protocol
`ai-race-frontier-baseline-v3`: once at task version 3 on 2026-09-08, which is
the run the manuscript reports, and once at task version 4 on 2026-09-09, which
is the run stored here. The repeat was not planned as evidence; it rode along
with the push that added the reasoning-budget amendment for
`google/gemini-3.5-flash-lite`, and it was recovered from the Kaggle archive on
2026-09-10.

It is kept **outside** `baseline_campaign_v6/ai-race-baseline/` on purpose. Both
campaign analysers key their results by `model_route`, so a second run of an
already-represented route placed inside that tree would not raise an error. It
would silently displace the run the manuscript reports, and the displacement
would be invisible in every table.

Read it with `scripts/analyze_baseline_replication.py`, which refuses the
comparison unless both runs carry the same protocol, prompt hash, mechanism,
seed and set of (game seed, risk) blocks. Result:
`results/derived/baseline_replication.json`.
