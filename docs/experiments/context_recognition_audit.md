# Context recognition and contamination audit

Protocol: `ai-race-context-recognition-audit-v2`

Version 1 was rejected after its pilot exposed a contradictory response
contract: the generic-resemblance class was described as lacking a specific
match, while the parser still required a candidate string. The untouched v1
outputs and failed parser decisions remain archived. Version 2 permits
`candidate: null` for generic resemblance, requires a compact candidate only
for a specific named match, and does not retroactively rescore v1.

## Scope and evidence boundary

This is a separate, post-hoc model self-report audit. It never enters gameplay
prompts, the fixed-state replay, or the comprehension admission gate. A model
saying that a scenario resembles a named game is not proof that the checkpoint
memorised a benchmark or that training-data contamination occurred. Conversely,
failure to name a game does not establish absence of contamination.

Each of the eight context skins is crossed with both balanced P/Q mappings. The
audit uses the same medium-risk mechanism (`maxPrivateRisk = 0.6`) in every cell.
It renders the real static rules, then removes decision state, action history,
the optimization objective, and the action-output contract. The recognition
question supplies no candidate benchmark or game names.

The model must return one strict JSON object with a recognition class, a compact
candidate name when applicable, and low/medium/high confidence. Invalid output
is retried by sending the exact same prompt with a new recorded seed. No retry
adds a correction, example, candidate, or hint.

## Profiles

- `confirm`: 1 deterministic request per skin/mapping cell, 16 rows over both lanes.
- `smoke`: 2 repetitions per skin/mapping cell, 32 rows over both lanes.
- `pilot`: 20 repetitions per cell, 320 rows over both lanes.

All rows retain the complete prompt, prompt and scenario SHA-256 hashes, base
sampling seed, every retry seed and raw response, strict parser result, skin,
mapping, and repetition. The manifest records the exact Ollama model digest,
source-tree and per-file hashes, config hashes, hardware, decoding, and artifact
hashes.

## GreenNode launch

Stage the same committed checkout under `/home/jovyan` on both pods. Do not use
the currently unreliable shared `/network-volume` for active output.

Pod A:

```bash
cd /home/jovyan/AI_Race_Experiment
nohup python -m kaggle.experiments.greennode_context_recognition \
  --lane a --profile pilot --backend ollama \
  --repo-root /home/jovyan/AI_Race_Experiment \
  --output-root /home/jovyan/ai_race_runs/context_recognition_t0/lane_a \
  --temperature 0 --required-gpu H100 \
  > /home/jovyan/context_recognition_lane_a.log 2>&1 &
```

Pod B:

```bash
cd /home/jovyan/AI_Race_Experiment
nohup python -m kaggle.experiments.greennode_context_recognition \
  --lane b --profile pilot --backend ollama \
  --repo-root /home/jovyan/AI_Race_Experiment \
  --output-root /home/jovyan/ai_race_runs/context_recognition_t0/lane_b \
  --temperature 0 --required-gpu H100 \
  > /home/jovyan/context_recognition_lane_b.log 2>&1 &
```

The lane summaries report strict validity, retry rate, self-reported specific
match rate, broader resemblance rate, P/Q mapping stability, and normalized
candidate counts. Report these beside, and never pool them into, the behavioral and
comprehension estimands.
