# AI Race presentation assets

## Built artifacts

- `ai_race_research_deck.tex` is the canonical 21-slide, 16:9 LaTeX Beamer source.
- `../output/pdf/ai_race_research_deck.pdf` is the compiled presentation.
- `index.html` is an optional browser-based companion deck.

Compile from the repository root (run twice so Beamer resolves page metadata):

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf slides/ai_race_research_deck.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=output/pdf slides/ai_race_research_deck.tex
```

This directory contains a research-talk outline for the **LLM AI Race experiment**.

- [`ai_race_project.md`](ai_race_project.md) is a visual-first, 15-minute project deck plan.
- No `.pptx`, Beamer source, exported PDF, generated image, or result figure is included.

## Evidence status

The project deck is a protocol and research-plan presentation. LLM results are **pending**. Findings attributed to Fernández Domingos and Han (2026) come from a human behavioural experiment and must remain visually and verbally separated from this project’s future LLM results.

Do not bring forward legacy labels, figures, effect sizes, screenshots, or conclusions. The canonical baseline here is the two-player repeated AI race with private terminal setback risk.

## Design direction

Use a restrained, high-contrast palette: deep navy for the race environment, amber for Unsafe, cyan for Safe, and neutral grey for pending or unobserved results. Avoid red–green encoding. Each slide should communicate one idea, reserve substantial white space, and use direct labels rather than dense legends.

The visual backbone should be the race-state diagram, exact game mechanics, source-versus-project boundary, Kaggle execution pipeline, and prespecified effect plots. Human-source panels must carry a persistent label such as **SOURCE STUDY — HUMAN PARTICIPANTS**. Future result slides must carry **LLM EXPERIMENT — THIS PROJECT** and remain visibly marked **PENDING** until a frozen analysis exists.

## Production checklist

Before creating a presentation artifact, replace all author and venue placeholders, confirm the allotted duration, attach only validated figures, add figure-level sample units and uncertainty, and cross-check every numeric claim against the source note or a frozen project output. Keep citations readable at the bottom of the relevant slide.

The deck has not been built or rendered on the local machine.
