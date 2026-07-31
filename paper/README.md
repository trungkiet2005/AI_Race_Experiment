# LLM AI Race manuscript

This directory contains the manuscript for the project’s **LLM-agent AI Race experiment**.

## Current status

The manuscript reports a bounded, diagnostic game-understanding and calculator-aided behavioral pilot. Confirmatory risk-treatment and dynamic-state analyses remain pending. Pilot evidence is explicitly excluded from later confirmatory pooling.

The focal prior work—Fernández Domingos and Han (2026), arXiv:2607.26034—is a study of **human participants** plus an evolutionary model. It motivates the environment and planned analyses but is not evidence about LLM behaviour. See the project’s [AI-readable source note](../references/papers/markdown/falling-behind-ai-race.md).

## Files

- [`main.tex`](main.tex): article with the canonical game, validation protocol, admitted pilot audit, analysis plan, and limitations.
- [`refs.bib`](refs.bib): focal human-study citation and the supplied validity-survey manuscript citation.
- [`figures/`](figures/): reproducibly generated pilot figures.

## Evidence rules

Do not insert a numerical result unless it can be traced to a frozen run manifest and an analysis artifact. Do not reuse legacy results, figures, captions, or claims. Human-study findings belong in prior work and must be labelled as such.

The words *fear*, *preference*, *belief*, and *intent* require special care for LLM agents. Behavioural conditioning on race position may be reported as an observable association; it is not, by itself, evidence of a subjective mental state.

## Before results are written

Record the exact model and endpoint revision, prompt hash, game configuration, decoding settings, seed schedule, model-pair allocation, exclusion rules, retry policy, and planned sample size. Separate pilots from confirmatory runs. Define the dyad/race as the clustering unit and preserve round-level event logs.

The analysis should first reproduce a deterministic validation suite for payoff, progress, stopping, prize, tie, and setback calculations. It should then generate a dataset-accounting table before any behavioural estimate is interpreted.

## Visual backbone

The pilot includes rule/arithmetic accuracy and calculator-ablation figures. Later confirmatory work should add:

1. a graphical abstract showing `LLM dyad → repeated AI race → logged state/action dynamics → preregistered estimates`;
2. a canonical game schematic with simultaneous choices, progress, payoff, stopping, and terminal risk;
3. a treatment/model overview with the number of independent races;
4. an Unsafe-choice dynamics figure with uncertainty intervals;
5. a race-position and opponent-response effect plot; and
6. a reproducibility/data-flow diagram.

All figures must distinguish source-study values from this project’s estimates and include accessible colours, direct labels, sample units, and uncertainty.

## Build policy

The stable manuscript preview is written to `output/pdf/ai_race_paper.pdf`. From the repository root, build with:

```bash
pdflatex -output-directory=output/pdf -jobname=ai_race_paper paper/main.tex
bibtex output/pdf/ai_race_paper
pdflatex -output-directory=output/pdf -jobname=ai_race_paper paper/main.tex
pdflatex -output-directory=output/pdf -jobname=ai_race_paper paper/main.tex
```

Generated auxiliary files remain untracked; the named PDF preview is tracked.
