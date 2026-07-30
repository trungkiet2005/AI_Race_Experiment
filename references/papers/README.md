# AI Race paper library

This directory is the project’s curated literature source for the AI Race experiment. It intentionally contains one focal paper:

- [`pdf/Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment.pdf`](pdf/Falling%20Behind%20Drives%20Unsafe%20Development%20in%20an%20Idealised%20AI%20Race%20Experiment.pdf) — the retained source PDF.
- [`markdown/falling-behind-ai-race.md`](markdown/falling-behind-ai-race.md) — an original, AI-readable research note covering the source study’s mechanics, findings, and implementation caveats.

## Citation

Elias Fernández Domingos and The Anh Han. “Falling Behind Drives Unsafe Development in an Idealised AI Race Experiment.” arXiv:2607.26034, 2026. <https://arxiv.org/abs/2607.26034>

## Scope and use

The source paper reports an online behavioural experiment with human participants and a reduced evolutionary model. It does **not** report an LLM-agent experiment. The markdown note may inform environment design, preregistration, and analysis, but it must not be cited as evidence that language models reproduce the human findings.

When implementing the project, treat the action semantics, stochastic stopping rule, payoff matrix, progress increments, terminal prize, and private-risk calculation as a linked specification. Any intentional deviation should be named as a new experimental condition and documented in the run manifest.

No generated PDFs, slide exports, or derived paper copies belong in this directory.
