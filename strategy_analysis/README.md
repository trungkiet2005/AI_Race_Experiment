# AI Race strategy classification

This directory implements a transparent nearest-strategy classifier for repeated
Safe/Unsafe trajectories. It is a descriptive lens on LLM behaviour, not evidence
that a model internally represents or deliberately follows one of these strategies.

## Canonical strategy set

Actions use `Safe = 0` and `Unsafe = 1`. For focal action \(a_t\) and the
opponent's action \(b_t\):

| label | predicted trajectory |
|---|---|
| **AS** | Safe in every observed round |
| **AU** | Unsafe in every observed round |
| **CS** | Safe in round 1; for \(t \ge 2\), copy \(b_{t-1}\) |
| **CAS** | Unsafe in round 1; for \(t \ge 2\), copy \(b_{t-1}\) |

These are the four reduced strategies defined in *Falling Behind Drives Unsafe
Development in an Idealised AI Race Experiment*. In particular, CS and CAS differ
only in their first-round action.

`classify_trajectory` predicts all four trajectories over the realised horizon and
computes Hamming mismatches against the focal player's observed actions. The nearest
label has the fewest mismatches. The returned mismatch rate is
`mismatches / realised_horizon`, which permits descriptive comparisons across races
with different stopping times. All minimum-distance ties are retained; a short or
noisy trajectory must not be forced into an arbitrary unique class.

```python
from strategy_analysis.classify import classify_trajectory

result = classify_trajectory(
    own_actions=[0, 1, 0, 1],
    opponent_actions=[1, 0, 1, 0],
)

print(result.best_strategies)
print(result.unique_best_strategy)  # None when the nearest class is tied
```

String actions (`"Safe"`, `"Unsafe"`, `"S"`, `"U"`) are accepted at the API
boundary and normalised to binary values. Missing actions are not imputed: a
trajectory must be complete, non-empty, and have the same focal and opponent
horizon.

## JSON/JSONL command line

Each input record contains arrays rather than player-round rows:

```json
{"trajectory_id":"run-01/game-004/player-0","own_actions":[0,1,0],"opponent_actions":[1,0,1]}
```

Classify records and write one JSON object per trajectory:

```bash
python strategy_analysis/classify.py trajectories.jsonl --output strategy_matches.jsonl
```

The project-level result analyser reconstructs these arrays from `turns.jsonl`; the
standalone interface is useful for audits and small external datasets.

## Exploratory behind-responsive rule

The optional `BEHIND_UNSAFE_EXPLORATORY` rule plays Unsafe exactly when the focal
player's pre-decision progress gap is negative and otherwise plays Safe. It is
scientifically motivated by the LLM study question, but it is **not** one of the
paper's canonical AS/AU/CS/CAS strategies and must remain labelled exploratory.

Enable it only with both the explicit flag and one pre-decision gap per round:

```bash
python strategy_analysis/classify.py trajectories.jsonl \
  --include-exploratory-behind \
  --output strategy_matches_with_exploratory.jsonl
```

Do not silently pool that rule with the canonical four in confirmatory summaries.

## Synthetic dataset generation and classifier evaluation

`generate_dataset.py` simulates every ordered AS/AU/CS/CAS matchup, reusing
`ai_race.engine.strategies` so the generated actions can never drift from the
engine's own definition of the four strategies. Horizons follow the paper's
distribution (`min_rounds + Geom(stop_probability)`, capped at
`max_rounds_safety_cap`) via a standalone RNG — this is a dataset generator
for offline evaluation, not engine output, and does not reuse the engine's
seed streams. An optional per-round execution-noise flip (applied to the
realised action before the opponent can copy it) lets you probe robustness
under imperfect rule-following, the same noise convention used for the
sibling IPD project's strategy-classifier datasets in
`D:/AI_RESEARCH/ClusteringResearch/scripts/generate_noise_dataset_30round.py`.

Run as a module from the repo root (`generate_dataset.py` and
`evaluate_baseline.py` import `ai_race` and `strategy_analysis`, so they need
the repo root on `sys.path` the way `python -m` provides it — invoking them
as bare scripts, e.g. `python strategy_analysis/generate_dataset.py`, fails
with `ModuleNotFoundError: No module named 'ai_race'`):

```bash
python -m strategy_analysis.generate_dataset \
  --output strategy_analysis/datasets/ascsaucas.jsonl \
  --games-per-pair 200 --noise-levels 0.0,0.05,0.10
```

`evaluate_baseline.py` scores the existing nearest-strategy (Hamming
distance) classifier against a generated dataset and reports tied/exact
accuracy, mean tie width, and a confusion table per noise level:

```bash
python -m strategy_analysis.evaluate_baseline \
  strategy_analysis/datasets/ascsaucas.jsonl \
  --output strategy_analysis/datasets/ascsaucas_baseline_summary.json
```

At `noise_p = 0`, tied accuracy is 1.0 by construction, but *exact* accuracy
is only ~0.5: AS is behaviourally indistinguishable from CS (and AU from
CAS) whenever the opponent never plays the action that would make CS/CAS
diverge from AS/AU — e.g. AS vs. a CS opponent that starts and stays Safe in
response produces the exact same all-Safe trajectory AS would produce. This
is a real identifiability limit of the reduced four-strategy space, not a
classifier bug, which is exactly why `classify_trajectory` retains ties
instead of forcing a unique label.

`train_lstm_classifier.py` is an **optional** learned classifier mirroring
the LSTM architecture used for the sibling project's 30-round IPD strategy
classifiers (Masking -> LSTM(32) -> Dropout -> LSTM(16) -> Dropout ->
softmax over the four labels), trained on the same JSONL dataset with
class-balanced oversampling and early stopping. It requires the optional
`strategy-ml` extra (`pip install -e ".[strategy-ml]"`: tensorflow,
scikit-learn, imbalanced-learn) and is meant to answer "does a learned model
tolerate noise better than exact rule matching," not to replace the
paper-faithful nearest-strategy baseline in any confirmatory analysis:

```bash
python -m strategy_analysis.train_lstm_classifier \
  strategy_analysis/datasets/ascsaucas.jsonl
```
