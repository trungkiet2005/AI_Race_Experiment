# Literature-grounded experiment program for a high-impact AI Race paper

Survey frozen: 2026-08-02. This document turns the paper's remaining validity
threats into experiments with explicit estimands and promotion rules. It is a
targeted primary-source survey, not a systematic review.

## What the literature changes in this project

| Evidence from primary work | Design consequence here |
|---|---|
| Sclar et al. report large variation under meaning-preserving formatting and recommend ranges over plausible formats ([arXiv:2310.11324](https://arxiv.org/abs/2310.11324)). | Keep the completed multi-prompt surface suite, but do not spend the next GPU budget on more bundled synonyms. |
| Option and position biases can materially change strategic-game performance ([Herr et al., arXiv:2407.04467](https://arxiv.org/abs/2407.04467)); multiple-choice order is independently documented by [Pezeshkpour and Hruschka](https://aclanthology.org/2024.findings-naacl.130/). | Fully cross opaque mappings within seed and add a per-turn remapping follow-up only after recall gates pass. |
| Counterfactual variants separate transferable reasoning from familiar-task recitation ([Wu et al., arXiv:2307.02477](https://arxiv.org/abs/2307.02477)). | Add a positive payoff-scale transformation that is mechanically invariant but numerically unfamiliar. |
| Game-theoretic rationality requires separating desire, belief, and action ([Fan et al., arXiv:2312.05488](https://arxiv.org/abs/2312.05488)). | Add the opponent-belief/action-coherence protocol rather than inferring strategy from action frequency alone. |
| Repeated-game behavior can change when agents predict opponents before acting ([Akata et al., arXiv:2305.16867](https://arxiv.org/abs/2305.16867)); recent opponent-simulation work explicitly separates belief formation and best response ([Liu et al., arXiv:2602.19309](https://arxiv.org/abs/2602.19309)). | Compare action-only and forecast-then-action conditions against deterministic reference strategies. |
| SAE studies emphasize feature identification, intervention position, and multi-layer representation ([SAIF](https://arxiv.org/abs/2502.11356), [RouteSAE](https://arxiv.org/abs/2503.08200)). | Preserve the current negative causal controls; do not call high AUC an explanation. Future XAI must use neutral reconstruction, held-out races, random-direction nulls, sign/dose tests, and pre-divergence live replication. |

## Ranked program and exact status

| Priority | Experiment | Main limitation closed | Size | Status |
|---:|---|---|---:|---|
| P0 | Fully crossed context × opaque mapping | Mapping was tied to repetition parity | 1,536 races | Runner, analyzer, tests, protocol ready; GPU blocked by SSH authentication |
| P0 | Positive payoff-scale invariance | Numeric/payoff bias under an unchanged strategic game | 384 races | Mechanical contract passed 1,048,512 terminal comparisons; GPU runner ready and blocked by SSH authentication |
| P0 | Transition × terminal computation scaffold | State-update versus terminal-risk failure is currently bundled | 768 races | 2×2 runner, prompt tests, and protocol ready; GPU blocked by SSH authentication |
| P1 | Comprehension-admitted cross-family replication | One open checkpoint and one digest | 3+ model families | Prepared design; execute only after P0 prompt cells are frozen |
| P1 | Replay-to-fork feedback experiment | Live-minus-fixed is not a mediation estimate | 96+ divergence states | Prepared; requires fresh model continuations from identical fork states |
| P1 | Opponent belief/action coherence | A plausible action does not establish belief or best response | 768 races | Protocol prepared; implementation follows P0 admission |
| P2 | Per-turn opaque remapping | Stable code policy may mimic a strategy | 768 races | Conditional on ≥95% per-turn mapping recall |
| P2 | SAE causal promotion | Decodability is not feature-specific control | ≥30 held-out race clusters | Conditional on behaviorally neutral reconstruction and ≥10 discovery flips |

## The strongest paper story after P0

The highest-impact contribution is not “another LLM plays another game.” It is
a validity decomposition with three deliberately orthogonal invariance tests:

1. **Narrative invariance:** same numbers and mechanism, different context.
2. **Code invariance:** same context and mechanism, different opaque action map.
3. **Utility-unit invariance:** same strategic game, different positive payoff scale.

The scaffold factorial then asks why invariance fails: can verified transition or
terminal arithmetic repair comprehension and does that repair behavior? This
creates a reviewer-auditable chain from code correctness, to task comprehension,
to behavior, to trajectory feedback, to activation-level association and causal
failure.

## “Wow” demo sequence

1. Start two mechanically identical trajectories side by side and show their
   first divergence.
2. Toggle Safe=P/Safe=Q and display the fully crossed effect with block-level
   uncertainty.
3. Change payoff units from `1` to `100` without changing normalized utility;
   any action flip becomes a visually immediate failure of strategic invariance.
4. Toggle transition and terminal tools to show whether correct public arithmetic
   repairs comprehension, behavior, both, or neither.
5. Finish with the XAI control panel: high held-out decodability beside null
   target-specific steering, making the association/causation boundary tangible.

## Promotion rules

- A completed run remains a pilot until coverage, raw-output, parser, CRN,
  provenance, and admission checks pass.
- A model family is a replication unit; provider routes and precisions are never
  silently pooled.
- A comprehension-aided behavior is tool-assisted performance, not an unaided
  world model.
- A context or scale effect is checkpoint- and protocol-scoped; it is not a
  psychological preference.
- XAI is causal only when the selected direction beats frozen empirical controls
  on held-out race clusters and the reconstruction control is behaviorally neutral.
