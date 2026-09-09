# LLM AI Race manuscript

This directory contains the manuscript for the project’s **LLM-agent AI Race experiment**.

## Current status

The manuscript reports a bounded frontier-model baseline: a per-endpoint task-validity audit, confirmatory admissions for Gemini 3 Flash and Claude Sonnet 5, 30-race behavioural runs for both admitted routes, published-human reference values, and a faithful EGT reconstruction. Gemini 3 Flash also has a completed and independently validated 120-race context and mapping run. The frontier evidence remains descriptive and endpoint-specific. The matched Claude context run, fully crossed persona and language reruns, and frontier N-player reruns are not admitted; legacy local-model pilots are excluded from headline evidence.

The focal prior work—Fernández Domingos and Han (2026), arXiv:2607.26034—is a study of **human participants** plus an evolutionary model. It motivates the environment and planned analyses but is not evidence about LLM behaviour. See the project’s [AI-readable source note](../references/papers/markdown/falling-behind-ai-race.md).

## Files

- [`ai_race_paper.pdf`](ai_race_paper.pdf): the current compiled manuscript.
- [`ai_race_supplementary.pdf`](ai_race_supplementary.pdf): the current compiled supplementary material.
- [`main.tex`](main.tex): article source with the canonical game, validation protocol, admitted audit, analysis plan, and limitations.
- [`supplementary.tex`](supplementary.tex): supplementary source.
- [`references.bib`](references.bib): bibliography for both documents.
- [`../figures/paper/`](../figures/paper/): paper and presentation figures; Figures 1, 2, 4, and 8 are protected author artwork and the remaining slots are reproducibly generated.

## Evidence rules

Do not insert a numerical result unless it can be traced to a frozen run manifest and an analysis artifact. Do not reuse legacy results, figures, captions, or claims. Human-study findings belong in prior work and must be labelled as such.

The words *fear*, *preference*, *belief*, and *intent* require special care for LLM agents. Behavioural conditioning on race position may be reported as an observable association; it is not, by itself, evidence of a subjective mental state.

## Before results are written

Record the exact model and endpoint revision, prompt hash, game configuration, decoding settings, seed schedule, model-pair allocation, exclusion rules, retry policy, and planned sample size. Separate pilots from confirmatory runs. Define the dyad/race as the clustering unit and preserve round-level event logs.

The analysis should first reproduce a deterministic validation suite for payoff, progress, stopping, prize, tie, and setback calculations. It should then generate a dataset-accounting table before any behavioural estimate is interpreted.

## Current visual workflow

The canonical fourteen-slot set (eight main-paper and six supplementary)
combines four protected author-supplied artworks with generated figures. The
generated subset is built by
[`scripts/build_publication_figures.py`](../scripts/build_publication_figures.py)
which also checks that the protected artwork is present and does not overwrite
it. The central browsing
and selection notes are in
[`figures/SELECTION_GUIDE.md`](../figures/SELECTION_GUIDE.md). Candidate and
diagnostic figures remain in `figures/gallery/` and `figures/diagnostics/`;
they are not paper evidence until their source, evidence class, estimand, and
caption are checked.

## Historical visual wishlist (superseded)

The pilot includes rule/arithmetic accuracy and calculator-ablation figures. Later confirmatory work should add:

1. a graphical abstract showing `LLM dyad → repeated AI race → logged state/action dynamics → preregistered estimates`;
2. a canonical game schematic with simultaneous choices, progress, payoff, stopping, and terminal risk;
3. a treatment/model overview with the number of independent races;
4. an Unsafe-choice dynamics figure with uncertainty intervals;
5. a race-position and opponent-response effect plot; and
6. a reproducibility/data-flow diagram; and
7. an EGT-versus-frontier Unsafe-rate comparison with independent-chain diagnostics.

All figures must distinguish source-study values from this project’s estimates and include accessible colours, direct labels, sample units, and uncertainty.

## Build policy

The final manuscript PDFs are written directly to `paper/`:
`paper/ai_race_paper.pdf` and `paper/ai_race_supplementary.pdf`. From the
repository root, build both the paper and deck with:

```bash
python scripts/build_publication.py
python scripts/check_publication.py --allow-placeholder-id
```

Generated auxiliary files stay under the ignored `results/_build/latex/` tree.
The build also synchronizes the two paper PDFs to
`results/artifacts/publication/` for submission and release scripts.
The QA command permits the local `TBD` submission ID during drafting; remove
that option for the final anonymous submission check. Set the ID once in
`submission_id.tex` when AAMAS assigns it.
