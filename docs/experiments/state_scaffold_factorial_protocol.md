# State-computation scaffold factorial

## Why this experiment

The current checkpoint recalled rules and one-stage payoffs but failed the
state-update and terminal-scoring admission domains. A single calculator card
improved atomic probes but bundled transition and action information. This
follow-up separates two public-information computations in a `2 × 2` design:

| Condition | Exact next-state rows | Conditional terminal rows |
|---|---:|---:|
| `none` | no | no |
| `transition` | yes | no |
| `terminal` | no | yes |
| `transition_terminal` | yes | yes |
| `length_placebo` | no | no (character-length control) |

The cards reveal no hidden horizon, setback draw, or opponent action. They only
enumerate quantities deterministically derivable from the displayed rules and
pre-decision state.

## Frozen design

- abstract payoff-preserving context only;
- both Safe=P and Safe=Q mappings inside every seed block;
- three risk levels and 32 repetitions;
- `4 × 2 × 3 × 32 = 768` factorial live races plus 192 matched-length
  placebo races (`960` total);
- temperature 0, same checkpoint digest, strict opaque-code parser, identical
  game/sampling seeds across all ten prompt cells;
- the comprehension battery is repeated under each scaffold before behavioral
  interpretation.
- both GPU lanes run every condition on disjoint even/odd repetition shards, so
  treatment is not assigned by hardware lane;
- every prompt ends with the identical response-contract suffix. The placebo
  block matches the joint-tool block by characters; token counts are audited
  by model digest and reported as a residual limitation.

## Estimands and admission

Primary comprehension endpoints are state-update and terminal-scoring accuracy.
Primary behavioral endpoints are paired first-round disagreement, full-trajectory
Unsafe-rate shift, and final payoff. Round-1 effects are the direct prompt
contrast; live trajectory effects are labelled symmetric game-level total
intervention effects because both players receive the aid. Factorial main effects and their interaction
are reported separately by mapping, with race-block bootstrap intervals.

Gameplay remains diagnostic unless every required comprehension domain clears
its frozen threshold. Better performance with a card demonstrates aid uptake,
not an unaided internal world model. A behavioral change without a comprehension
gain is reported as prompt intervention, not improved rationality.
The matched-length placebo-versus-none contrast is reported separately and is
not treated as a fifth factorial cell.

### Frozen comprehension gate

`results/scripts/analyze_state_scaffold_factorial.py` requires a completed
`ai-race-state-scaffold-admission-v1` JSON bound to the behavioral source hash,
experiment-config hash, exact model digest, and decoding object. It recomputes
accuracy from raw integer counts for every condition-by-mapping cell. The
frozen minimums are:

- every semantic domain at least `0.80` per cell;
- state-update and terminal-scoring semantic accuracy at least `0.90` per cell;
- strict parse rate at least `0.95` per cell;
- at least four frozen items for each comprehension domain per cell;
- zero scaffold-arithmetic mismatches and zero hidden-information leaks.

Missing or malformed admission evidence stops analysis. A valid completed
admission that misses a performance threshold does not erase the experiment:
outputs are retained but labelled `diagnostic_pilot_comprehension_not_admitted`.
The JSON stores raw `*_n` and `*_correct` fields, never only rounded rates.

### Behavioral analyzer

The analyzer requires complete factorial coverage, and requires and reports the
placebo whenever it is present. It checks unique race/player/turn cells, exact
model and decoding provenance, all CRN seeds, horizons, stop draws and setback
draws. A parse failure excludes the whole ten-cell `(risk, repetition)` block.

For each mapping and endpoint it freezes the contrasts as
`transition_main = ((T-N) + (TT-Terminal))/2`,
`terminal_main = ((Terminal-N) + (TT-T))/2`, and
`interaction = TT-T-Terminal+N`. Holm correction is applied to these three
tests within each mapping-by-endpoint family. Round-1 action is labelled a
direct pre-feedback effect. Full-trajectory Unsafe rate and final payoff are
labelled symmetric live total effects. Placebo-minus-none is a separate audit.

```bash
python -m results.scripts.analyze_state_scaffold_factorial \
  --input-root <downloaded-state-scaffold-root> \
  --admission-json <completed-admission.json> \
  --output-dir results/derived/state_scaffold_factorial \
  --bootstrap-repetitions 5000
```

## GPU launch

Run the frozen comprehension admission once before gameplay. It writes
`admission.json` and `comprehension_raw.jsonl`, including exact prompt-bank,
source, config, model-digest, and raw-artifact hashes. A scientifically valid
failed gate retains both artifacts but exits with status 2, preventing a shell
pipeline from launching gameplay.

```bash
python -m kaggle.experiments.greennode_scaffold_comprehension \
  --profile pilot --repo-root /network-volume/icse27/AI_Race_Experiment \
  --output-root /network-volume/icse27/AI_Race_Experiment/results/open_source/state_scaffold/admission \
  --required-model-digest <exact-ollama-digest>
```

Only launch the two disjoint gameplay shards after the admission command
returns zero:

```bash
python -m kaggle.experiments.greennode_state_scaffold \
  --lane a --profile pilot --repo-root /network-volume/icse27/AI_Race_Experiment \
  --output-root /network-volume/icse27/AI_Race_Experiment/results/open_source/state_scaffold/lane_a \
  --required-model-digest <exact-ollama-digest>

python -m kaggle.experiments.greennode_state_scaffold \
  --lane b --profile pilot --repo-root /network-volume/icse27/AI_Race_Experiment \
  --output-root /network-volume/icse27/AI_Race_Experiment/results/open_source/state_scaffold/lane_b \
  --required-model-digest <exact-ollama-digest>
```
