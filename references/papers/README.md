# AI Race paper library

This directory is the project’s curated literature source for the AI Race experiment.

## Library

### Fernández Domingos and Han (2026)

- [`pdf/Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment.pdf`](pdf/Falling%20Behind%20Drives%20Unsafe%20Development%20in%20an%20Idealised%20AI%20Race%20Experiment.pdf) — retained source PDF.
- [`markdown/falling-behind-ai-race.md`](markdown/falling-behind-ai-race.md) — AI-readable research note covering the study’s mechanics, findings, and implementation caveats.
- [`sources/arXiv-2607.26034v1/`](sources/arXiv-2607.26034v1/) — complete arXiv v1 LaTeX source bundle and figures.

These are the canonical copies of the paper output and figures; they are not
duplicated under the project-level `output/` directory.

Elias Fernández Domingos and The Anh Han. “Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment.” arXiv:2607.26034, 2026. <https://arxiv.org/abs/2607.26034>

### Han et al. (2020)

- [`sources/jair-12225/JAIR-12225-ArticlePDF-25030-1-10-20201122.pdf`](sources/jair-12225/JAIR-12225-ArticlePDF-25030-1-10-20201122.pdf) — retained JAIR article PDF.
- [`sources/jair-12225/JAIR-12225-ArticlePDF-25030-1-10-20201122.md`](sources/jair-12225/JAIR-12225-ArticlePDF-25030-1-10-20201122.md) — text extracted from the PDF with PyMuPDF; equations should be checked against the PDF.

The Anh Han, Luís Moniz Pereira, Francisco C. Santos, and Tom Lenaerts. “To Regulate or Not: A Social Dynamics Analysis of an Idealised AI Race.” *Journal of Artificial Intelligence Research* 69 (2020): 881–921. <https://doi.org/10.1613/jair.1.12225>

## Scope and use

The 2026 paper reports an online behavioural experiment with human participants and a reduced evolutionary model. It does **not** report an LLM-agent experiment. The markdown note may inform environment design, preregistration, and analysis, but it must not be cited as evidence that language models reproduce the human findings. The 2020 JAIR paper provides the earlier theoretical social-dynamics model of regulation in an idealised AI race.

When implementing the project, treat the action semantics, stochastic stopping rule, payoff matrix, progress increments, terminal prize, and private-risk calculation as a linked specification. Any intentional deviation should be named as a new experimental condition and documented in the run manifest.

The `sources/` directory preserves imported source bundles and publisher artifacts. Generated project PDFs, slide exports, and unrelated build artifacts do not belong in this library.
