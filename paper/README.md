# LLM AI Race manuscript

This directory contains the manuscript scaffold for the project’s **LLM-agent AI Race experiment**.

## Current status

The manuscript is a protocol-oriented draft. No project experiment has been run for this clean-up, and no LLM results are reported. Every result-bearing section in [`main.tex`](main.tex) is explicitly marked **pending**.

The focal prior work—Fernández Domingos and Han (2026), arXiv:2607.26034—is a study of **human participants** plus an evolutionary model. It motivates the environment and planned analyses but is not evidence about LLM behaviour. See the project’s [AI-readable source note](../references/papers/markdown/falling-behind-ai-race.md).

## Files

- [`main.tex`](main.tex): article scaffold with introduction, canonical game specification, LLM protocol, analysis plan, pending Results, and limitations.
- [`refs.bib`](refs.bib): verified citation for the focal human-study preprint.

## Evidence rules

Do not insert a numerical result unless it can be traced to a frozen run manifest and an analysis artifact. Do not reuse legacy results, figures, captions, or claims. Human-study findings belong in prior work and must be labelled as such.

The words *fear*, *preference*, *belief*, and *intent* require special care for LLM agents. Behavioural conditioning on race position may be reported as an observable association; it is not, by itself, evidence of a subjective mental state.

## Before results are written

Record the exact model and endpoint revision, prompt hash, game configuration, decoding settings, seed schedule, model-pair allocation, exclusion rules, retry policy, and planned sample size. Separate pilots from confirmatory runs. Define the dyad/race as the clustering unit and preserve round-level event logs.

The analysis should first reproduce a deterministic validation suite for payoff, progress, stopping, prize, tie, and setback calculations. It should then generate a dataset-accounting table before any behavioural estimate is interpreted.

## Planned visual backbone

No figure or build artifact is created during this clean-up. Once validated data exist, the manuscript should add:

1. a graphical abstract showing `LLM dyad → repeated AI race → logged state/action dynamics → preregistered estimates`;
2. a canonical game schematic with simultaneous choices, progress, payoff, stopping, and terminal risk;
3. a treatment/model overview with the number of independent races;
4. an Unsafe-choice dynamics figure with uncertainty intervals;
5. a race-position and opponent-response effect plot; and
6. a reproducibility/data-flow diagram.

All figures must distinguish source-study values from this project’s estimates and include accessible colours, direct labels, sample units, and uncertainty.

## Build policy

No PDF is stored here. Compilation was intentionally not run on the local machine. If a manuscript preview is needed, build it in the Kaggle environment after the project dependencies are prepared, for example with:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

Generated `.aux`, `.bbl`, `.blg`, `.log`, `.out`, and `.pdf` files should remain untracked.
