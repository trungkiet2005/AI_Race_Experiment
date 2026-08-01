# AI Race presentation assets

## Canonical research deck

- `ai_race_research_deck.tex` is the 24-frame, 16:9 Beamer evidence deck.
- `../references/output/pdf/ai_race_research_deck.pdf` is the compiled presentation.
- `ai_race_project.md` is the slide-by-slide evidence map and speaker outline.
- `index.html` remains an optional browser companion; it is not the canonical evidence artifact.

Compile twice from the repository root:

```bash
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=references/output/pdf slides/ai_race_research_deck.tex
pdflatex -interaction=nonstopmode -halt-on-error \
  -output-directory=references/output/pdf slides/ai_race_research_deck.tex
```

## Evidence status

The deck now reports validated **exploratory pilot artifacts**, not a results-pending protocol. It covers the engine and game-understanding audit, calculator ablation, 18-variant surface sensitivity grid, the primary temperature-zero eight-context study, a separate temperature-0.7 robustness comparison, recognition audit, actual-self-play FAST-SAE analyses, and a reduced evolutionary-game reconstruction.

The central admission boundary is explicit: the context comprehension gate failed, so context and mapping effects diagnose prompt-conditioned behavior rather than verified informed utility optimization. SAE AUC/correlation is presented as association; causal attribution is withheld because target steering did not outperform controls.

The EGT frame reports a faithful reconstruction, not bitwise reproduction: the source paper has not released its author code, generated payoff matrices, EGTTools lockfile, or Monte Carlo seeds. The independent transition implementation matches the pinned official EGTTools source to numerical precision, while the LLM comparison remains descriptive and non-equivalent to population evolution.

## Figure policy

The Beamer source embeds publication PDFs directly from `paper/figures/` and `results/open_source/`. Do not replace them with screenshots. Every result frame states its evidence class, denominator, and the boundary needed to interpret it.

Design uses deep navy, cyan, amber, lime, and neutral ink; it avoids red-green semantics. Figures must remain readable at 16:9 projector scale and must not be cropped to remove uncertainty intervals, sample units, or captions.

## Release checklist

1. Compile twice with `-halt-on-error`.
2. Render every page to PNG and inspect for clipping, overlap, broken glyphs, and illegible labels.
3. Confirm all numeric claims against frozen CSV/JSON artifacts.
4. Keep human-source findings visually separate from project pilots.
5. Do not promote diagnostic results to confirmatory claims without passing the listed gates.
