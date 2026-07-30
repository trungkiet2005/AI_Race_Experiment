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
