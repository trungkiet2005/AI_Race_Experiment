# Research note: *Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment*

## Source identity

**Full citation:** Elias Fernández Domingos and The Anh Han (2026), “Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment,” arXiv:2607.26034v1, submitted 28 July 2026.  
**Canonical record:** <https://arxiv.org/abs/2607.26034>  
**Full-text HTML used to verify this note:** <https://arxiv.org/html/2607.26034>  
**Local source:** [`../pdf/Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment.pdf`](../pdf/Falling%20Behind%20Drives%20Unsafe%20Development%20in%20an%20Idealised%20AI%20Race%20Experiment.pdf)

This note is an original technical summary rather than a transcription. The source is a **human behavioural study**, augmented by an evolutionary game-theoretic model. It is not an evaluation of LLM agents.

## Research question and study status

The study asks whether unsafe technological development in a stylised two-company AI race is explained by the imposed level of private risk, participants’ separately elicited risk preferences, or the evolving strategic state of repeated competition. The comparison between maximum private-risk levels of 0.60 and 0.90 and the risk-preference tests were preregistered. A 0.10 treatment was added later, and the analyses of opponent behaviour, relative position, and first-round behaviour were exploratory.

## Exact experimental mechanics

The experiment ran online in oTree with participants recruited through Prolific. Participants were paired and framed as companies competing to develop a technology. In every round, both members of a pair chose simultaneously between **Safe** (\(S\)) and **Unsafe** (\(U\)). Before a new choice, they could observe both players’ preceding actions, accumulated earnings, accumulated private risks, and race positions.

Action \(S\) advanced the focal player by \(1\) race step, while \(U\) advanced them by \(1.5\) steps. With rows denoting the focal player’s action and columns the opponent’s action, the per-round payoff matrix was

\[
\pi =
\begin{pmatrix}
1 & 0.6 \\
2.4 & 2
\end{pmatrix}.
\]

Consequently, \(U\) strictly dominated \(S\) in the one-shot payoff matrix and also conferred faster progress. The safety incentive entered through terminal private setback risk rather than a direct cost of mutual Unsafe play in the stage game.

Every race lasted at least five rounds. Immediately after round 5, and after every subsequently completed round, the race stopped with probability \(0.2\). Equivalently,

\[
T = 5 + G,\qquad G\sim \operatorname{Geom}(0.2)-1,
\]

so the theoretical mean is \(E[T]=9\). The instructions and preregistration mistakenly described the mean as 10 rounds. The authors report that this arithmetic error was discovered after collection; the realised mean was 9.56 rounds across 173 games.

At termination, the participant with greater cumulative progress received a race prize of 100 experimental currency units (ECUs); a tie split that prize. A winner’s or tied winner’s private setback probability was

\[
q_i(T)=p_r^{\max}\frac{n_i^U(T)}{T},
\]

where \(n_i^U(T)\) is the number of that participant’s Unsafe choices. If the setback occurred, the affected participant lost the task payoff. The between-participant treatment was the cap \(p_r^{\max}\in\{0.10,0.60,0.90\}\). The competitive rules otherwise remained fixed. Risk preferences were elicited before the race with an Eckel–Grossman gamble-choice task.

The study recruited 471 people. After procedural exclusions, 340 completed the full experiment. The main round-\(t\geq2\) panel analysis used 2,888 decisions from 338 participants and 172 pair clusters after requiring complete covariates; one partially completed pair contributed three valid post-first-round observations. Standard errors were clustered at the pair level.

## Empirical findings

The preregistered 0.60-versus-0.90 comparison was null. In the raw all-round participant-level comparison reported with the main figure, the authors give \(t=-0.0101\) and Bonferroni-corrected \(p=1\). In the covariate-complete round-\(t\geq2\) sample, the corresponding effect was negligible (\(d=-0.027\), \(t=-0.206\), corrected \(p=1\)). Elicited risk preference also did not significantly predict Unsafe decisions or interact significantly with maximum private risk.

The subsequently added 0.10 treatment differed from both higher-risk treatments: Unsafe play was higher at 0.10 than at 0.60 and 0.90. On the raw all-round participant means, the corrected comparisons were \(p=0.0272\) and \(p=0.0161\), respectively. This treatment-level pattern should be interpreted cautiously because 0.10 was not part of the original preregistered comparison and was collected in a later wave. A sentence in the source paper's Figure 2 caption states the direction oppositely; the reported means, effect signs, Figure 3, and supplementary comparisons support the direction stated here.

The main dynamic findings were exploratory. In the full cluster-robust logistic specification that included first-round choice, the opponent’s preceding Unsafe action positively predicted a focal Unsafe action (\(\hat\beta=0.607,\ p=0.002\)). When both players had previously chosen Safe, being farther ahead reduced Unsafe play (\(\hat\beta=-0.296,\ p=0.048\)); the interaction between the focal player’s preceding action and position was also positive (\(\hat\beta=0.466,\ p=0.011\)). A first-round Unsafe choice predicted a greater later tendency toward Unsafe play, although its strength became marginal in the fullest specification. The focal player’s own immediately preceding action was not a robust standalone predictor after the other state variables were included.

These coefficients are conditional associations, not identified causal effects. Current choices inherit the entire prior interaction history, so the lagged actions are endogenous to the evolving game.

## Reduced evolutionary model

The paper interprets the behavioural pattern with four deterministic strategies:

- **Always Safe (AS):** choose \(S\) in every round.
- **Always Unsafe (AU):** choose \(U\) in every round.
- **Conditionally Safe (CS):** begin with \(S\), then copy the opponent’s preceding action.
- **Conditionally Antisocial Safe (CAS):** begin with \(U\), then copy the opponent’s preceding action.

Expected payoffs for matches involving CS or CAS were estimated with \(10^4\) Monte Carlo races per ordered matchup; unconditional matchups used closed-form expectations. A finite-population pairwise-comparison process with Fermi imitation, selection strength \(\beta\), and mutation probability \(\mu\) produced a stationary distribution over strategies.

The model reproduced the qualitative treatment ordering rather than establishing a unique psychological explanation. At the reference parametrisation, AU was favoured at low maximum risk, CAS at intermediate risk, and CS at the highest risk. A noisier best-fitting region used \(\mu=0.05\) and \(\beta=0.01\). The fit should be read as a compact mechanistic interpretation of the human data, not as a validated model of actual AI laboratories or LLM policy formation.

## Limitations stated or implied by the source

The race is deliberately short and stylised, has only two actors, and omits public signals, uncertain capabilities, regulation, reputation, and multi-actor institutional structure. Risk is private: the design does not model a collective or systemic catastrophe imposed on outsiders. Only the maximum private-risk cap changes between treatments; competitive pressure itself is not experimentally varied. The four-strategy model excludes graded distance-sensitive responses and heterogeneous learning. The central dynamic results are exploratory, and the lagged predictors cannot support a simple causal reading. The added 0.10 condition was collected after the preregistered 0.60/0.90 waves.

At arXiv v1, the authors state that the analysis code will be deposited on Zenodo and the de-identified data in the preregistration's OSF repository upon publication; the public release is therefore described prospectively in the paper. An implementation should not assume that this project framework is the source experiment's reference code unless a later deposited release is explicitly matched and verified.

## Implementation requirements for an LLM adaptation

An LLM experiment should preserve the distinction between **source-study mechanics** and **new agent protocol**. A faithful canonical environment needs to implement simultaneous hidden choices, the exact payoff orientation, progress increments of 1 and 1.5, the minimum-five-round horizon, a fresh 0.2 stopping draw after every later round, the 100-ECU winner prize and tie split, and the terminal risk formula based on the realised Unsafe fraction. Setback risk applies only to winners and tied winners in the source design. Every random draw, state transition, action parse, and terminal calculation should be logged.

The state supplied to an agent should be explicit and versioned. At minimum, logs should preserve round index, both preceding actions once revealed, both positions, both accumulated stage payoffs, both current private risks, treatment, prompt version, model identifier, decoding parameters, seed, raw response, parsed action, and any retry or fallback. Decisions must be obtained before either agent sees the opponent’s same-round response. Conversation memory, natural-language rationales, personas, and extra feedback are experimental factors because they can add strategic information absent from the source task.

LLMs do not possess a directly comparable Eckel–Grossman human risk-preference measure. Any analogue must be justified and preregistered rather than treated as equivalent. Likewise, an API sampling temperature is not a psychological risk attitude.

The dyad or independently seeded race—not the individual round—should define the principal clustering unit. Round-level rows are repeated observations and must not be analysed as independent samples. First-round decisions should be retained as both an outcome and a prespecified predictor, because the source study found that early behaviour carried information about later play.

Model checkpoints, hosted endpoints, and inference defaults can change. Freeze exact model revisions when possible, save the complete environment and prompt manifests, separate pilot data from confirmatory runs, balance seed allocation across risk treatments and model pairings, and record failed or truncated generations. For Kaggle-only execution, the notebook or script should produce a portable run manifest and append-only event log so an interrupted session can be audited without reconstructing state from notebook output.

Most importantly, results from the human paper are **external benchmarks**, not expected LLM results. An LLM study must report its own estimates and uncertainty and should not use phrases such as “replicates fear of falling behind” unless a preregistered analysis supports that claim. Observable action conditioning is not evidence that a model experiences fear, understands risk, or represents an AI company.

## Design implications for this project

The cleanest first experiment is a canonical two-agent condition that changes only \(p_r^{\max}\), followed by explicitly labelled extensions for model pairing, prompt framing, memory, or governance interventions. The primary outcome can be the probability of choosing Unsafe. Prespecified secondary analyses can test opponent-action responsiveness, relative-position effects, first-round momentum, terminal win/setback outcomes, and heterogeneity across model families. Human-study coefficients may be shown as contextual reference points, but statistical claims should be based entirely on the project’s independently generated LLM races.
