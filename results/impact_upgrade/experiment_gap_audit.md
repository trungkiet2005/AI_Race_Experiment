# Experiment gap audit and impact roadmap

## Decision

The project does **not** need another broad prompt-sensitivity sweep before its
largest validity gap is closed. The highest-value next run is the frozen,
fully-crossed context × opaque-action-mapping pilot. It turns the strongest
post-hoc pattern into an identified within-seed contrast at roughly the cost of
one additional context pilot.

## What is already covered

| Question | Evidence available | Boundary |
|---|---|---|
| Does prompt surface change behavior? | Paired canonical/card and broad surface pilots | Pilot, checkpoint scoped |
| Does semantic context matter with mechanics fixed? | Eight contexts at T=0 and T=0.7; fixed-state and live protocols | Diagnostic because comprehension fails |
| Does the model know the rules and arithmetic? | 41 atomic probes plus context comprehension gates | Rule recall strong; state transition and terminal scoring fail |
| Are mechanics/payoffs correct? | Unit tests, raw-log recomputation, CRN checks, EGT transition parity | Code/mechanism validation |
| Is the effect cross-model? | Two OpenAI and three Gemini pilot baselines | Descriptive only; provider protocols differ |
| Do SAE features explain the action? | Held-out probes, controlled steering, live self-play | Association retained; causal claim rejected |
| Can the result be demonstrated clearly? | Paired trajectory lab and portable evidence report | Source-backed diagnostic demo |

## Ranked next experiments

### P0 — fully crossed mapping × context

- **Gap closed:** mapping was previously tied to repetition parity.
- **Design:** 8 contexts × 2 mappings × 3 risks × 32 seeds = 1,536 races.
- **Primary estimand:** context-versus-abstract difference-in-differences across
  Safe=P and Safe=Q, clustered by risk/repetition and stratified by role.
- **Why first:** strongest current effect, smallest direct repair, excellent live
  demo, and no new model dependency.
- **Status:** config, runner, protocol, CRN validator, and tests are checked in;
  GPU launch awaits current GreenNode NodePorts.

### P1 — comprehension-admitted cross-family replication

- **Gap closed:** current context result is one checkpoint that fails admission.
- **Design:** same frozen fixed-state bank and mapping grid on at least three
  model families at temperature 0; gameplay only after all comprehension domains
  meet thresholds.
- **Why second:** converts a checkpoint diagnostic into a model-family result.
- **Cost/risk:** high inference cost; some cells may fail admission and yield no
  gameplay evidence, which is scientifically informative but less demo-friendly.

### P1 — replay-to-fork feedback experiment

- **Gap closed:** live-minus-fixed is not currently a mediation estimand.
- **Design:** replay identical logged states, identify the first prompt-induced
  action divergence, then fork matched endogenous continuations from that state.
- **Why second:** directly separates prompt response from feedback amplification.
- **Demo value:** very high—a single state branches into two auditable futures.

### P2 — per-turn opaque-ID randomization

- **Gap closed:** stable P/Q code policies and positional priors remain possible.
- **Design:** independently randomize mapping each round, disclose it in the same
  template position, decode after response, and pair the randomization schedule.
- **Risk:** the remapping burden may itself reduce comprehension; it therefore
  needs a mapping-recall gate on every turn.

### P2 — SAE causal promotion run

- **Gap closed:** current feature discovery has one action flip and insufficient
  support for a discrete steering target.
- **Design:** collect at least 10 discovery flips or freeze a continuous
  log-odds target; use multiple matched-random directions, held-out states,
  dose/sign checks, and behaviorally calibrated reconstruction controls.
- **Promotion:** target steering must exceed the empirical control distribution
  and reproduce in live pre-divergence decisions.

## Experiments not recommended now

- More synonyms, whitespace variants, or personas without a factorial design:
  coverage grows while attribution gets weaker.
- Larger SAE steering doses on the same selected features: reconstruction already
  changes 12.5% of fixed prompts, so dose escalation can amplify artifacts.
- Pooling OpenAI, Gemini, and Qwen rows into one regression: model, provider,
  template, and execution protocol are not exchangeable.
- Calling the fully crossed run “confirmatory”: the model's comprehension gate
  remains failed, so the strongest honest label is a preregistered diagnostic
  replication.

## Paper-impact sequence

1. Lead with the validity result: identical mechanics do not guarantee stable
   strategic behavior.
2. Show the paired trajectory demo, then reveal the mapping gate.
3. Use the fully crossed run to identify whether the gate replicates.
4. Keep comprehension and negative SAE intervention results adjacent to the
   behavioral claim.
5. Reserve broad model-family claims until an admitted cross-family grid exists.
