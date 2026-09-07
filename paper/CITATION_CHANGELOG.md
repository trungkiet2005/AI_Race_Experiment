# Citation revision changelog

This note records the citation-focused revision of `paper/main.tex`. Line numbers refer to the current files after the revision.

## Copy map

| File and lines | Change |
|---|---|
| `paper/main.tex:40` | Adds the human-simulation validity boundary using Aher et al. and del Rio-Chanona et al. |
| `paper/main.tex:50` | Adds prompt-format, option-order, response-token, whitespace/output-format, and persona evidence. |
| `paper/main.tex:54` | Replaces the broad game-bias sentence with claim-matched Herr and Akata wording. |
| `paper/main.tex:110` | Separates the executed single-layer FAST artifact from RouteSAE's cross-layer method. |
| `paper/main.tex:120` | Separates verbal reports from behavioural evidence. |
| `paper/main.tex:201--205` | Rephrases the SAE result as failure to establish specificity, not evidence of equivalence. |
| `paper/main.tex:249` | Adds principled SAE-evaluation and feature-selection boundaries. |
| `paper/main.tex:255` | Adds the distributional human-simulation boundary. |
| `paper/main.tex:281` | Replaces “survive causal controls” with the narrower feature-specific conclusion. |
| `paper/refs.bib:128--231` | Adds ten checked bibliography records. |

## Copy-ready LaTeX

### Human-simulation validity

```latex
LLM-based human simulation has produced both qualitative replications and systematic distortions. Aher et al.\ reproduced findings in three of four behavioural experiments but identified a hyper-accuracy distortion in the fourth \citep{aherEtAl2023}. Dynamic market experiments have likewise recovered broad human treatment differences while retaining less behavioural heterogeneity at finer resolution \citep{delRioChanonaEtAl2025}. We therefore compare distributions and prespecified estimands, not surface-level human likeness.
```

### Prompt sensitivity

```latex
Prompt-based evaluation is not invariant to seemingly irrelevant presentation choices. Meaning-preserving changes in few-shot formatting can produce large performance spreads, and those effects need not transfer across models \citep{sclarEtAl2024}. Reordering multiple-choice options can change predictions \citep{pezeshkpourHruschka2024}, while order and response-token identity can exert distinct selection biases \citep{weiEtAl2024}. Controlled variants have also found changes under output-format requests and atomic whitespace perturbations \citep{salinasMorstatter2024}. Persona prompts should therefore be treated as interventions rather than generic performance improvements: across objective factual questions, their average benefit was absent or slightly negative and the best persona was difficult to select reliably \citep{zhengEtAl2024Persona}. These results motivate separate, crossed manipulations of wording, format, response mapping, and persona rather than one bundled robustness condition.
```

### Game-behaviour evidence

```latex
In two-player Stag Hunt and Prisoner's Dilemma presentations, reversing action-label order and reassigning payoffs changed choice frequencies, with model-specific positional and payoff biases \citep{herrEtAl2024}. In finitely repeated games, Akata et al.\ found that GPT-4 could predict an opponent's alternating pattern yet did not initially act in accordance with it; requiring an opponent prediction before action improved coordination and scores \citep{akataEtAl2025}.
```

### SAE/XAI scope

```latex
RouteSAE motivates a complementary cross-layer view by routing activations from multiple layers through a shared dictionary \citep{shiEtAl2025Route}. The present artifact is a single-layer FAST JumpReLU SAE: RouteSAE was not run, and this audit does not identify a cross-layer circuit.
```

```latex
With only six held-out race clusters, the fact that none of the 12 target-minus-control intervals excluded zero is a failure to establish feature specificity, not evidence that target and control interventions are equivalent.
```

```latex
Reconstruction fidelity, sparsity, semantic interpretability, and task-level controllability are distinct evaluation axes \citep{makelovEtAl2025Principled}; activation association alone need not predict an intervention's output effect \citep{aradEtAl2025Steering}.
```

## Source-evidence ledger

The arXiv source archives were extracted under the system temporary directory, not the repository. Published venue metadata was preferred where available.

| Citation | Primary source inspected | Exact support used | Deliberately not claimed |
|---|---|---|---|
| Sclar et al. 2024 | arXiv 2310.11324 source: `abstract_v2.tex:1`; `iclr2024_conference.tex:158--175,211--235` | Meaning-preserving few-shot formatting changes can yield large, model-dependent spreads. | Not evidence for semantic paraphrase effects. |
| Pezeshkpour and Hruschka 2024 | arXiv 2308.11483 source: `main.tex:211--229,247--267` | Reordering answer options changes predictions. | The proposed uncertainty/placement mechanism is a conjecture, not established causality. |
| Wei et al. 2024 | arXiv 2406.03009 source, order- and token-bias experiments | Order and response-token identity are distinct selection-bias targets. | Not a universal effect size for every task or model. |
| Salinas and Morstatter 2024 | arXiv 2401.03729 source: `acl_latex.tex:52,61--66` | Controlled output-format and atomic whitespace variants can change answers. | Not evidence that every added space changes behaviour. |
| Zheng et al. 2024 | arXiv 2311.10054 source: lines 100--117,335--340 | Across their factual-QA setup, average persona benefit was absent/slightly negative and automatic selection was unreliable. | Not a universal claim about all persona tasks. |
| Aher et al. 2023 | full ICML source, results for four replicated behavioural studies | Three qualitative replications and one hyper-accuracy distortion; actions can differ from survey-style answers. | Not proof that LLM samples are interchangeable with human populations. |
| del Rio-Chanona et al. 2025 | arXiv 2505.07457 full source | Broad market-treatment patterns with reduced within-condition heterogeneity. | Not evidence of psychological equivalence. |
| Akata et al. 2025 | arXiv 2305.16867 source: `main.tex:210--214` and opponent-prediction experiment | GPT-4 predicted an alternating strategy before initially acting inconsistently; prediction-before-action improved coordination. | Not a general theory-of-mind claim. |
| RouteSAE | arXiv 2503.08200 source: `chapter/0_abstract.tex`, `chapter/3_methods.tex`, `chapter/4_experiments.tex:25--34,62--76,204--219`, `chapter/5_limitations.tex:3--10` | Multiple-layer routing into a shared SAE; cross-layer extraction remains early-stage. | RouteSAE was not run here and does not validate the FAST layer-12 feature. |
| Makelov et al. 2025 | official ICLR paper; source `arxiv-version.tex:128--150,159--169` | Reconstruction/approximation, control, and interpretability are separable evaluation axes. | No single metric proves causal feature identity. |
| Arad et al. 2025 | ACL Anthology paper, abstract and feature-selection experiments | Activation association need not predict output effect; output-aware selection improves steering. | Does not identify this project's chosen feature as causal. |
| Buscemi et al. 2025 | arXiv 2504.14325v5 source: `m1210.tex:98,109--115,125--128,181--241` | FAIRGAME standardises configurable repeated-game experiments across models, languages, and persona conditions and enables comparison with game-theoretic predictions. | Not evidence that an LLM understands a game or that FAIRGAME results generalise to the AI-race mechanism. |
| Huynh et al. 2025 | arXiv 2512.07462v2 source: `main.tex:114--139,148--157,203--286` | Extends FAIRGAME with a payoff-scaled Prisoner's Dilemma, a three-player Public Goods Game, and supervised recognition of canonical strategies from trajectories. | Canonical-strategy classification is not direct access to latent intention, and the study does not use this paper's AI-race mechanism. |
| Huynh et al. 2026 | arXiv 2601.19082v2 source: `main.tex:237--267,277--318` | Reports model-, language-, and stake-dependent cooperation and inferred strategy distributions, including disagreement in direction with its EGT benchmark. | Does not establish the same direction or effect size for AI-development races. |

## Build and preservation notes

- Built with `pdflatex`, `bibtex`, and two final `pdflatex` passes.
- Final PDFs: `paper/ai_race_paper.pdf` and `paper/ai_race_supplementary.pdf`.
- The current build completes without fatal LaTeX errors, but the log still contains layout warnings and an undefined supplementary reference. Do not treat the PDFs as submission-ready until those warnings are resolved.
- Rendered all 19 pages to PNG and visually checked the edited prose and reference pages.
- Existing user edits in `paper/AAMAS_2026_sample.tex` and `paper/references.bib` were not modified.
