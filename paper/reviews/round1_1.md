Humans Are More Diverse: Frontier LLMs Show Extreme Policies in Idealised AI Development Races
ICLR
Submitted: September 7, 2026
Contents
Summary
Strengths
Weaknesses
Detailed Comments
Questions
Overall Assessment
Summary
This paper investigates how large language model (LLM) agents behave in an “AI development race” repeated game where agents choose between Safe (slower, lower private risk) and Unsafe (faster, higher private risk) development. The core contribution is an audit-first methodology that separates mechanical validity, task validity, and representation robustness before any strategic interpretation, followed by trajectory-level analyses that compare LLM self-play with evolutionary game-theory benchmarks and human data across two- to five-player races and persona framings. The key findings are that (i) strong rule recall can coexist with poor state tracking and expected-payoff calculation; (ii) semantically equivalent prompt manipulations (e.g., disclosing arithmetic or remapping SAFE/UNSAFE to opaque labels) can materially change action sequences; and (iii) LLMs with similar aggregate Unsafe rates can have starkly different trajectory-level response patterns to risk, opponent history, relative position, and group size, while human behavior exhibits broader within-population diversity.

Strengths
Technical novelty and innovation
Introduces a rigorous “audit-first” pipeline that gates strategic interpretation on mechanical and task validity (rule recall, state tracking, payoff calculation), plus representation robustness checks.
Extends a two-player AI race game to N-player settings (N ∈ {3, 4, 5}) with a clear mapping to the Han et al. multi-agent payoff rule, preserving other mechanics to isolate stage-payoff differences.
Employs trajectory-level analyses (HDBSCAN archetypes, t-SNE embeddings, decision trees) rather than only aggregate Unsafe rates, revealing strategic diversity and path dependence.
Uses evolutionary game-theory strategies as a qualitative benchmark and compares to publicly available human trajectories, highlighting differences in both level and distribution.
Experimental rigor and validation
Separates diagnostic pilots from confirmatory evidence and documents configuration, seeds, and parsing status; emphasizes that races, not decisions, are the appropriate experimental unit.
Validity audit identifies concrete failure modes (state update, expected-payoff calculation), and representation diagnostics demonstrate non-trivial sensitivity to meaning-preserving changes.
Careful caveats around confounding (e.g., rank not randomized; selection effects when Unsafe increases progress), with constructive proposals for future causal identification.
Clarity of presentation
Clear game specification with equations, horizon details, and private-risk mechanism; explicit distinction between stage payoffs and terminal prize.
Transparent articulation of what is and is not claimed; consistent warnings against over-interpreting aggregate rates or persona prompts as measuring intrinsic “risk preference.”
Significance of contributions
Addresses an increasingly important use case—LLM social/strategic simulation—and demonstrates why naïve interpretations are fragile without validity and robustness gates.
The contrast between human diversity and model-specific narrow bands has immediate implications for using LLMs as human proxies or for policy-related AI-race modeling.
Weaknesses
Technical limitations or concerns
The main task-validity audit is conducted on a single checkpoint (Qwen2.5-7B-Instruct), yet many behavioral conclusions are drawn from other endpoints; cross-model generalization of the audit is not established.
Several representation-robustness experiments fail the associated comprehension gate; although flagged appropriately, the resulting behavioral differences remain difficult to interpret beyond prompt sensitivity.
Rank effects in N-player races are observational and likely confounded by prior Unsafe choices and progress accumulation; no design-based isolation of causal rank effects is provided.
Experimental gaps or methodological issues
Some reporting is incomplete or inconsistent (e.g., Table 2 has missing cells; model naming varies across figures, and a few labels appear inconsistent across sections), which impedes reproducibility and clarity.
The opaque-code mapping (P/Q) manipulation is not fully counterbalanced within seed blocks; mapping and repetition parity are entangled, weakening identification.
Persona effects are strong but are not cleanly disentangled from token/format biases; a placebo control is mentioned but quantitative placebo-versus-persona contrasts are not deeply discussed.
The comparison to the evolutionary benchmark is qualitative; more systematic alignment (e.g., controlled simulations under the same horizon distributions) could help interpret discrepancies.
Clarity or presentation issues
Occasional terminology/model-ID inconsistencies across figures and text (e.g., “GPT-5-nano” vs “OPT-5,” Claude Sonnet versions) can confuse readers about which endpoints were actually tested.
Some critical numbers are only in the supplement (e.g., per-cell counts for rank analyses, SHAP breakdowns), limiting the standalone interpretability of the main text.
Missing related work or comparisons
While related work coverage is strong, the paper could acknowledge additional agent frameworks and prompting protocols (e.g., AutoGen/CAMEL-style multi-agent scaffolds, safety-card/decision-sheet paradigms) and clarify how its “audit-first” procedure complements or improves upon them.
The paper could more explicitly connect to recent mechanistic accounts of LLM deviations from Nash behavior and discuss whether observed validity failures align with known late-layer override phenomena.
Detailed Comments
Technical soundness evaluation
The game mechanics and N-player generalization are well-specified and grounded in prior work. The audit design is a meaningful methodological contribution that many LLM-agent papers lack. However, the core validity audit is not replicated across the set of endpoints used for behavioral claims; this makes it hard to know if the same state-tracking and expected-payoff failures apply to the other models.
The representation robustness tests are well-motivated and consistent with broader brittleness literature. Still, the strongest divergences occur exactly where the comprehension gate fails, limiting strategic interpretation and underscoring the need for fully counterbalanced designs and per-model audits.
Experimental evaluation assessment
The decision to treat races as the unit of analysis is appropriate, and seeds/configs are tracked. The use of paired first-round comparisons and fixed-state replay to isolate prompt effects is a good design pattern.
Human–LLM comparisons via full trajectories (HDBSCAN, embeddings, and simple classification) are informative and avoid over-fitting to hand-crafted features. The nested logistic models showing LLM-specific risk–response shapes are compelling, though model convergence warnings and floor/ceiling issues suggest caution in inference.
The N-player analyses offer useful pilot observations but should be reframed as exploratory; position effects are confounded, and sample sizes per persona/rank cell appear small (authors note this, but the main text could more prominently report per-cell n and CIs for transparency).
Comparison with related work (using the summaries provided)
The “audit-first” stance and robustness checks align with the emerging consensus from Brittlebench and robustness benchmarks that semantics-preserving perturbations can materially shift outcomes; this paper extends that insight to multi-round strategic settings and demonstrates why validity gating is essential before behavioral interpretation.
Relative to FAIRGAME (and its extensions), this paper emphasizes task comprehension and state tracking as prerequisites and makes a distinct contribution by connecting to human data and by probing representation sensitivity (e.g., SAFE/UNSAFE mapping), complementing FAIRGAME’s cross-language/personality/game breadth.
Compared to prior repeated-game LLM studies showing departures from Nash or language/persona effects, this paper’s distributional view (trajectory archetypes and human diversity) adds depth and nuance, highlighting that similarity in aggregate rates can mask very different dynamics.
Discussion of broader impact and significance
The core message—that aggregate similarity (e.g., Unsafe rate) is insufficient evidence of understanding or human-likeness—has high stakes for any work using LLMs as human proxies or as agents in safety-policy simulations. The paper sets stronger standards for validity, logging, and trajectory-level analysis that could improve the field’s methodological rigor.
The finding that persona prompts act as strong behavioral interventions, while human measured risk preferences barely correlate with Unsafe play, is important to avoid misattributing persona effects to latent “psychology.”
Limitations are adequately and candidly discussed, particularly around confounds, counterbalancing, and the scope of claims; future work suggestions (e.g., randomizing rank or conditioning on own-history within rank) are concrete and valuable.
Questions for Authors
Can you extend the task-validity audit beyond Qwen2.5-7B-Instruct to at least a subset of the other endpoints you analyze behaviorally? Even a reduced audit (state tracking, terminal scoring, expected-payoff) would greatly strengthen cross-model conclusions.
In the opaque P/Q mapping experiment, can you provide a fully counterbalanced design within seed blocks (mapping × narrative × repetition) and rerun the fixed-state replay to isolate direct prompt effects? How much of the divergence remains once counterbalancing and the comprehension gate are enforced?
For the N-player rank analysis, would you consider conditioning on prior own Unsafe history (or randomizing early actions) to separate selection effects from a causal “being ahead/behind” effect on the next decision?
Could you report parse-failure rates, fallback rules, and “contaminated” race counts per model and condition in the main text? Since a single parse failure contaminates a whole trajectory, this seems crucial for interpreting behavioral distributions.
The decision-card manipulation increased Unsafe decisions while leaving round-1 choices nearly unchanged. Do you have ablations that separate “disclosed arithmetic” from “response formatting” effects in the card, e.g., by providing only arithmetic without the tabular layout or vice versa?
Several tables/figures show naming inconsistencies (e.g., model IDs across figures) and missing cells (Table 2). Can you reconcile these and release the full artifact (code, prompts, seeds, logs) to ensure reproducibility?
Did you explore language effects (English vs. other languages) on the audit and on behavior, akin to FAIRGAME results, and if so, are validity failures language-dependent?
Overall Assessment
Estimated Score:
5.8/10
(Calibrated to ICLR scale)
This is a careful, timely, and consequential study that advances standards for evaluating LLM agents in strategic multi-round settings. The audit-first methodology and representation-robustness diagnostics are genuine contributions that many current agent papers lack, and the trajectory-level human comparison is nuanced and revealing: humans occupy a broad, diverse region of behavior space, whereas individual LLM checkpoints often compress into narrow, model-specific bands despite sometimes matching human means. The paper is also appropriately cautious, repeatedly refraining from over-claiming when validity gates fail or confounds persist.

The main weaknesses are empirical rather than conceptual. Several of the strongest representation-sensitivity findings occur exactly where the comprehension gate fails; the audit is not replicated across the broader model slate; and some reporting inconsistencies (missing table cells, model ID drift) reduce clarity. The N-player results are exploratory and confounded by construction, which the paper acknowledges, but stronger design-based identification would elevate this part considerably.

On balance, I view this as a valuable methodological and empirical contribution. It raises the bar for validity and analysis in LLM-agent studies, offers actionable diagnostics, and provides a more faithful picture of where current models stand relative to human diversity. With improved cross-model auditing, fully counterbalanced robustness tests, and cleaned-up reporting, this work would be a strong fit for ICLR. I recommend a weak accept.