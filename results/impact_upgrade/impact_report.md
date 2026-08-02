# AI Race evidence synthesis and impact upgrade

## Technical summary

The upgraded evidence changes the paper's strongest defensible story. The result is not that one language model has a stable level of risk appetite. It is that **strategic behavior is jointly determined by model checkpoint, prompt surface, semantic framing, and endogenous trajectory feedback—even when payoff mechanics are unchanged**.

Five baseline checkpoints cover sharply different regimes: GPT-5 nano remains near 12–15% Unsafe, GPT-5.4 nano is U-shaped near 50–58%, while the three Gemini checkpoints begin at 84–100% Unsafe under 10% risk and decline as risk rises. These are cross-provider pilots and are never pooled inferentially, but they falsify any simple one-model generalization.

The Qwen context experiment provides a more controlled mechanism result. All paired first-round decisions agree, yet fixed-state replay detects direct context effects up to 16.7 percentage points and live trajectories reach a 34.0-point aggregate gap. The crucial new interaction is stronger: **the context effect is entirely gated by opaque action-code mapping in this pilot**. When Safe is code Q, the seven contexts produce no divergence from the abstract reference; when Safe is code P, six contexts diverge in every paired player trajectory, with the largest Unsafe-rate difference reaching 68.0 points. Mapping is balanced but assigned by repetition parity, so this is a high-priority replication target rather than a mapping-causal estimate.

The validity boundary is equally important. Qwen passes rule recall (100%) and stage payoff (98.4%), but state updating is 12.5% and terminal scoring 17.2%; the preregistered comprehension admission gate therefore fails. SAE probes can predict action-associated representations (held-out AUC up to 0.985), but context steering causes no action flips in the context run, and the self-play target-minus-control intervention contrasts do not establish a reliable causal controller. Those negative results are retained, not hidden.

## Key findings and visual evidence

### 1. Cross-model heterogeneity is qualitative, not a scale shift

The same three risk caps produce monotone-decreasing Gemini curves, a low flat GPT-5 nano curve, and a U-shaped GPT-5.4 nano curve. Reporting only an aggregate model mean would erase the phenomenon the study is trying to measure. The paper should report checkpoint-level curves and frame cross-model evidence as replication of instability, not as one pooled treatment effect.

### 2. Action-code position gates semantic framing

The mapping interaction is the most demo-worthy result because it is visible round by round and has a concrete experimental remedy. Opaque IDs were intended to neutralize Safe/Unsafe wording, yet which ID denotes Safe determines whether context can move behavior. A fully crossed diagnostic follow-up is now frozen: both mappings run inside every seed block instead of being assigned by repetition parity.

### 3. Direct prompt response grows along live trajectories

Fixed-state replay measures the action change caused by context while holding the state constant. Live play repeats the context and allows earlier decisions to alter later states. Logistics has a 15.6-point fixed-state effect and a 34.0-point live effect; the 18.4-point difference is descriptive evidence consistent with amplification, not a causal mediation estimate because the analysis units differ.

### 4. No round-1 flip does not imply robustness

All 1,344 paired player-trajectory comparisons agree in round 1. Divergence begins later, then changes progress, private risk, setbacks, and terminal payoff. Entry-only audits therefore miss state-conditional sensitivity in repeated games.

### 5. Mechanistic interpretability supplied a useful negative result

The FAST-SAE pipeline has pinned model/SAE revisions, held-out splits, matched random and unrelated-feature controls, and live self-play. It detects predictive representations, but the intervention evidence does not clear the causal bar. This distinguishes *decodable information* from *behavioral control* and makes the XAI section more credible.

## Scope, data, and metric definitions

The synthesis covers 768 Qwen T=0 context races (13,680 decisions), a separate matched T=0.7 stratum of the same size, 2,640 OpenAI pilot races (49,104 decisions), 177 Gemini pilot races (3,168 decisions), an N=3 Qwen pilot, FAST-SAE representation and intervention audits, and an independent EGTTools transition validation. Temperature strata and provider protocols are not pooled.

The player-level Unsafe rate averages each player's trajectory before aggregation. Mapping-stratified live contrasts compare the same risk, repetition, player seat, and mapping against the abstract context. Kaplan–Meier curves treat race termination as censoring. The fixed-state estimand compares prompts on identical replayed states. Terminal payoff differences are realized, so they include setback draws fixed by common-random-number block.

## Methodology and quality gates

Every source is admitted through explicit checks: manifest status and phase, model and prompt identity, expected cell coverage, zero parse failures where claimed, player/turn count reconciliation, canonical mechanics, and common-random-number alignment. The impact script fails closed on missing contexts, risks, mappings, rows, or inconsistent horizons. Output tables and figures are hashed in `analysis_manifest.json`.

The EGT reconstruction is labelled faithful rather than bitwise because the original paper has not released its private code and seeds. The independent transition matrix matches the pinned EGTTools implementation to 1.11e-16 and its stationary distribution to 7.63e-15 in the validation case.

## Limitations, uncertainty, and robustness boundary

- Context findings remain **diagnostic** because the comprehension admission gate fails.
- OpenAI and Gemini results are **pilots**; their local-run manifests do not identify persona effects cleanly across protocol signatures.
- Mapping is balanced but assigned by repetition parity in the completed live context runs. The mapping interaction is therefore a replication target, not a clean mapping main effect.
- Decisions within a race are dependent. Turn-weighted rates are descriptive, not independent Bernoulli trials.
- The live-minus-fixed gap combines repeated prompt exposure and endogenous state feedback; it is not a causal mediation effect.
- SAE feature association is not feature causation. The strongest controlled intervention contrasts do not support a reliable action controller.
- N=3 persona cells have two races each and belong in the demo/appendix, not the headline claim.

## Recommended next experiments

1. **Launch the frozen fully crossed mapping × context pilot.** For every seed, execute both Safe=P and Safe=Q across all eight contexts. The checked-in protocol fixes 1,536 races, paired estimands, Holm correction, failure gates, and promotion rules. This closes the largest identified confound at the lowest compute cost while retaining the failed-comprehension diagnostic boundary.
2. **Admission-gated cross-family replication.** Run at least three model families at one fixed decoding setting; analyze gameplay only for model/configuration cells that pass rule recall, state transition, terminal scoring, and expected-payoff thresholds.
3. **Direct-versus-feedback replay.** Freeze a logged state sequence, replay all contexts at each state, then separately launch endogenous trajectories from the same first divergence. This creates commensurable direct and feedback estimands.
4. **Opaque-ID randomization per decision.** Randomize labels on every turn and decode after response to test whether position bias persists when a stable code policy cannot form.
5. **Causal SAE promotion rule.** Increase discovery action flips, pre-register feature selection on discovery races, and require target steering to exceed matched-random and unrelated-feature controls on held-out states and live races.

## Further questions

- Why does context have leverage only under one opaque mapping: token prior, recency, instruction position, or a learned code policy?
- Which comprehension failure best predicts later divergence after controlling for model and first-round choice?
- Are the Gemini risk slopes stable under the same action mapping and prompt template used for Qwen?
- Does trajectory feedback amplify context because of self-consistency, opponent imitation, progress-gap response, or terminal-risk calculation?
- Can a neutral verified state tool improve payoff without itself becoming a new framing treatment?

## Literature alignment

The result aligns with recent work showing that equivalent strategic games can change under narrative context ([Same Game, Different Story](https://arxiv.org/abs/2607.19670)) and that framing shapes LLM decisions ([Framing the Game](https://arxiv.org/abs/2503.04840)). It also addresses the warning that some apparent prompt sensitivity can be an evaluation artifact ([Hua et al., EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1006/)): here actions are parsed as exact opaque codes, mechanics are recomputed from raw logs, and divergence changes enacted trajectories, so the main effect is not a fuzzy answer-matching artifact. The XAI boundary follows SAE work that separates representation discovery from controlled intervention ([SAIF](https://arxiv.org/abs/2502.11356), [FAST](https://arxiv.org/abs/2506.07691), [RouteSAE](https://arxiv.org/abs/2503.08200)).
