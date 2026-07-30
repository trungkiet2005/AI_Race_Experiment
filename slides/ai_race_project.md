# Deck outline: LLM agents in an idealised AI race

**Format:** 15-minute research-plan talk plus questions  
**Audience:** game theory, multi-agent LLM evaluation, and AI governance researchers  
**Status:** protocol only; project results pending  
**Core source:** Fernández Domingos & Han (2026), arXiv:2607.26034

## Slide 1 — LLM agents in an idealised AI race

**On-screen**

- Speed versus safety
- Repeated two-agent game
- Project protocol — results pending

**Visual**

A clean split path from two model agents to Safe and Unsafe actions, converging on a race track with an uncertain finish. Add a prominent grey `RESULTS PENDING` chip.

**Speaker note**

Introduce the project as a controlled adaptation of a human behavioural AI-race experiment. State immediately that no LLM result is being presented.

## Slide 2 — Competition changes the decision

**On-screen**

- Unsafe moves faster
- Unsafe pays more now
- Unsafe accumulates risk

**Visual**

One decision fork: Safe advances one step; Unsafe advances 1.5 steps and fills a risk gauge. Show the rival beside the focal agent so the trade-off is visibly strategic.

**Speaker note**

The question is not simply whether a model avoids risk. It is whether its action changes when a rival is moving faster or when it falls behind.

## Slide 3 — The source is a human study

**On-screen**

- Paired human participants
- Online repeated race
- Evolutionary interpretation

**Visual**

A two-column boundary. Left: `SOURCE STUDY — HUMAN PARTICIPANTS`, with Prolific and oTree represented generically. Right: `THIS PROJECT — LLM AGENTS`, with Kaggle and model endpoints. Place a bold arrow labelled `adapt mechanics, test anew`.

**Speaker note**

The source paper is Fernández Domingos and Han (2026). Its findings motivate hypotheses, but they are not LLM evidence and will never appear as this project’s results.

**Citation**

Fernández Domingos & Han (2026), arXiv:2607.26034.

## Slide 4 — Canonical game mechanics

**On-screen**

- Simultaneous \(S/U\)
- Progress: \(1\) vs \(1.5\)
- Minimum 5 rounds
- Then 20% stop chance

**Visual**

A round timeline with two sealed action cards, state update, and a stopping coin after round 5. Include \(E[T]=9\), not 10.

**Speaker note**

Both same-round actions are committed before either is revealed. After the fifth round, a fresh stop draw occurs after every additional completed round.

**Citation**

Fernández Domingos & Han (2026), Methods §6.2.

## Slide 5 — Payoff now, private risk later

**On-screen**

\[
\pi=\begin{pmatrix}1&0.6\\2.4&2\end{pmatrix}
\]

\[
q_i=p_r^{\max}\frac{n_i^U}{T}
\]

**Visual**

Put the payoff matrix on the left with rows labelled focal Safe/Unsafe and columns labelled opponent Safe/Unsafe. On the right, show a 100-ECU winner prize feeding into a terminal setback draw. Label risk caps \(10\%,60\%,90\%\).

**Speaker note**

Unsafe strictly dominates in immediate payoff and also advances faster. Risk is accumulated from the realised Unsafe fraction and applies to a winner or tied winner at termination. This is private terminal risk, not a group-wide loss.

**Citation**

Fernández Domingos & Han (2026), Methods §6.2.

## Slide 6 — What the human study found

**On-screen**

- 60% vs 90%: null
- Risk preference: null
- Opponent action mattered
- Relative position mattered

**Visual**

Use a source-labelled evidence card. Separate two preregistered null findings from three exploratory dynamic associations. Do not display these as project bars or reuse them as LLM estimates.

**Speaker note**

The preregistered 0.60-versus-0.90 difference was negligible, and elicited risk preference did not predict Unsafe choice. Exploratory panel analyses associated Unsafe choice with the opponent’s preceding action, race position, and first-round behaviour. The lagged associations were not identified causal effects.

**Citation**

Fernández Domingos & Han (2026), Results §§2.1–2.3 and Limitations.

## Slide 7 — The LLM research gap

**On-screen**

- Same game, different agents
- Prompts create protocol choices
- Sampling is not preference
- Behaviour is not experience

**Visual**

A bridge with four labelled gaps: checkpoint, prompt, memory, decoding. Beneath it, place the warning `observable conditioning ≠ subjective fear`.

**Speaker note**

LLMs have no directly comparable Eckel–Grossman measure. Temperature is not psychological risk tolerance. The adaptation must study observable actions while avoiding anthropomorphic conclusions.

## Slide 8 — Prespecified research questions

**On-screen**

1. Does risk cap change Unsafe choice?
2. Does opponent \(U_{t-1}\) predict \(U_t\)?
3. Does falling behind predict \(U_t\)?
4. Does round 1 carry forward?

**Visual**

A four-node causal-question map around a central Unsafe-action outcome. Use question marks rather than directional arrows for effects that remain unestimated.

**Speaker note**

Freeze exact directional hypotheses, primary contrast, and multiplicity handling before confirmatory runs. Human coefficient signs are external reference points, not target values.

## Slide 9 — One race, end to end

**On-screen**

- Seed environment
- Query both agents
- Commit both actions
- Update exact state
- Log terminal risk

**Visual**

A horizontal pipeline from configuration manifest to append-only event log. Show same-round response isolation before the reveal gate.

**Speaker note**

The environment, not the LLM, performs arithmetic. Every raw response, parse decision, retry, state transition, stop draw, and setback draw must remain auditable.

## Slide 10 — What must be frozen

**On-screen**

- Model revision
- Prompt hash
- Decoding settings
- Pairing and seeds
- Retry policy

**Visual**

A versioned manifest card with immutable hashes. Add a small clock to represent endpoint drift and a lock to represent frozen confirmatory configuration.

**Speaker note**

Hosted model behaviour can change. Balance treatment allocation within model-pair strata, randomise or block run order, and keep pilots out of confirmatory inference.

## Slide 11 — Kaggle-only execution

**On-screen**

- Kaggle compute
- Checkpoint safely
- Resume deterministically
- Export logs and manifest

**Visual**

A Kaggle notebook box connected to durable checkpoints, event logs, and an analysis snapshot. Avoid depicting local compute as part of the execution path.

**Speaker note**

Notebook console text is not the dataset. A resumed session must reconstruct state from explicit checkpoints, while the final analysis must be reproducible from exported logs.

## Slide 12 — Analysis respects repetition

**On-screen**

- Outcome: \(P(U_{i,t}=1)\)
- Rounds nested in races
- Cluster by race/dyad
- Report uncertainty

**Visual**

A hierarchy diagram: decisions inside agents, agents inside seeded races, races inside model-pair and treatment strata. Next to it, sketch a coefficient plot with blank points labelled `pending`.

**Speaker note**

Do not count each round as an independent replicate. The planned dynamic model includes risk treatment, both preceding actions, relative progress, first-round choice, and prespecified interactions.

## Slide 13 — Validation before behaviour

**On-screen**

- Payoff fixtures
- Winner and tie cases
- Horizon simulation
- Risk draw audit
- Parser failures visible

**Visual**

A checklist over six miniature state-transition examples. Keep all boxes grey until validation artifacts exist.

**Speaker note**

The behavioural study should not begin until exact deterministic fixtures and seeded stochastic checks verify game semantics. Validation failures are engineering outcomes, not behavioural data.

## Slide 14 — Results template

**On-screen**

**LLM EXPERIMENT — THIS PROJECT**

- Run accounting: pending
- Treatment effects: pending
- Dynamic effects: pending
- Heterogeneity: pending

**Visual**

Four empty chart frames with diagonal `PENDING` watermarks. Reserve fixed areas for sample units, confidence intervals, exclusions, and snapshot identifiers.

**Speaker note**

This slide is deliberately empty. Replace it only with figures generated from a frozen analysis snapshot; never fill it with human-study values or pilot anecdotes.

## Slide 15 — How conclusions will be bounded

**On-screen**

- Match: task-level alignment
- Mismatch: population difference
- Neither implies psychology
- External validity stays limited

**Visual**

A \(2\times2\) interpretation matrix contrasting human-source direction and LLM estimate, with every cell ending at `behaviour in this task`.

**Speaker note**

Agreement would show similar observable conditioning under a shared game. It would not establish human-like fear or risk preference. Disagreement would not invalidate the human result because the populations and mechanisms are different.

## Slide 16 — The contribution, for now

**On-screen**

- Clean AI Race environment
- Auditable LLM protocol
- Preregistered dynamic tests
- Evidence comes next

**Visual**

Return to the opening race graphic, now overlaid with a transparent chain from specification to logs to estimates. Keep the final node grey and labelled `Kaggle runs pending`.

**Speaker note**

Close on the present contribution: a faithful, inspectable adaptation and a disciplined analysis boundary. The next milestone is validated Kaggle execution, not a prewritten conclusion.

## Backup A — Exact source-study accounting

**On-screen**

- 471 recruited
- 340 full completers
- 338 in main covariate-complete panel
- 2,888 post-round-1 decisions
- 172 pair clusters

**Visual**

A transparent participant-flow diagram labelled `SOURCE STUDY — HUMAN PARTICIPANTS`.

**Speaker note**

Use only if asked about the human evidence base. Explain that one partially completed pair contributed three valid post-first-round decisions and two completers lacked the nationality covariate.

**Citation**

Fernández Domingos & Han (2026), Supplement §S1.2.

## Backup B — Reduced human-study model

**On-screen**

- AS: always Safe
- AU: always Unsafe
- CS: start Safe, then copy
- CAS: start Unsafe, then copy

**Visual**

A four-strategy state machine with the two conditional strategies sharing the same copy rule but different first actions.

**Speaker note**

The reduced evolutionary model interprets the human pattern; it is not the policy class imposed on LLM agents. Conditional matchups used \(10^4\) simulated races per ordered matchup in the source.

**Citation**

Fernández Domingos & Han (2026), Methods §§6.4–6.5.

## Reference slide

Fernández Domingos, E., & Han, T. A. (2026). *Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment*. arXiv:2607.26034. <https://arxiv.org/abs/2607.26034>
