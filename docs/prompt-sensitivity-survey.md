# AI Race prompt-sensitivity survey and protocol

## What the literature supports

Prompt sensitivity is not one treatment. The primary literature distinguishes at
least the following families:

| Family | Evidence | AI Race treatment |
|---|---|---|
| Demonstration / option order | Lu et al. report large few-shot variation across demonstration permutations; Pezeshkpour & Hruschka find large multiple-choice gaps after option reordering. | Reverse SAFE/UNSAFE mention/output order; reverse payoff cases and state rows. |
| Information position | `Lost in the Middle` finds that performance depends on where relevant context occurs, often favoring the beginning or end. | Move the objective early; move terminal-risk rules next to the response instruction. |
| Paraphrase / lexical form | Mizrahi et al. evaluate 6.5M instances and show that instruction paraphrases change absolute and relative model rankings. | Close synonyms, instruction paraphrase, impersonal/passive syntax. |
| Format / delimiters | Sclar et al. find up to 76 accuracy points of variation under subtle meaning-preserving formats and recommend reporting a range over formats. | Markdown, XML, dense layout, extra spacing, numbered state, uppercase emphasis. |
| Spelling / wording / template perturbations | POSIX measures sensitivity across spelling, wording, and template-level changes; PromptBench supplies adversarial prompt attacks. | One recoverable typo plus the structured variants above. |
| Emotional framing | EmotionPrompt reports behavior/performance changes after adding emotional stimuli. | One explicitly non-invariance treatment, `emotional_importance`. |

Primary sources:

- Lu et al. (ACL 2022), [Fantastically Ordered Prompts and Where to Find Them](https://aclanthology.org/2022.acl-long.556/)
- Pezeshkpour & Hruschka (Findings of NAACL 2024), [Large Language Models Sensitivity to the Order of Options in Multiple-Choice Questions](https://aclanthology.org/2024.findings-naacl.130/)
- Liu et al. (TACL 2024), [Lost in the Middle](https://arxiv.org/abs/2307.03172)
- Sclar et al. (ICLR 2024), [Quantifying Language Models' Sensitivity to Spurious Features in Prompt Design](https://arxiv.org/abs/2310.11324)
- Mizrahi et al. (TACL 2024), [State of What Art? A Call for Multi-Prompt LLM Evaluation](https://aclanthology.org/2024.tacl-1.52/)
- Weber et al. (Findings of EMNLP 2024), [POSIX: A Prompt Sensitivity Index for Large Language Models](https://aclanthology.org/2024.findings-emnlp.852/)
- Zhu et al. (2023), [PromptBench: Towards Evaluating the Robustness of Large Language Models on Adversarial Prompts](https://arxiv.org/abs/2306.04528)
- Li et al. (2023), [EmotionPrompt](https://arxiv.org/abs/2307.11760)
- Hua et al. (EMNLP 2025), [Flaw or Artifact? Rethinking Prompt Sensitivity in Evaluating LLMs](https://aclanthology.org/2025.emnlp-main.1006/)

The last paper is important for this project: rigid answer matching can manufacture
apparent sensitivity. AI Race therefore keeps the parser and the two legal labels
fixed, records raw responses and parse failures separately, and never treats a
formatting failure as an UNSAFE behavioral choice without disclosing it.

## What is deliberately not called a meaning-preserving perturbation

- `Analyze`, `explain`, `list`, and `evaluate` specify different tasks; swapping
  them is not a clean robustness test for a binary game decision.
- `Risk-free` versus `dangerous`, or `cautious` versus `aggressive`, changes the
  framing of SAFE/UNSAFE and can change preferences for substantive reasons.
- Vague versus exact length constraints change the response contract. The suite
  instead compares two semantically equivalent exact-one-line formulations.
- Expert personas are already a separate factorial manipulation in this repo.
  Mixing persona and surface perturbation in the smoke matrix would obscure the
  estimand.
- Capitalization is tested as a surface feature; the protocol does **not** claim
  that uppercase text directly “increases attention weights.”

## Frozen matrix

The registry in `ai_race/prompts/sensitivity.py` contains 18 versioned cells:

- one frozen canonical control;
- fifteen meaning-preserving order, position, lexical, paraphrase, syntax,
  formatting, whitespace, emphasis, and boundary variants;
- one behavioral-framing cell (`emotional_importance`);
- one robustness/noise cell (`noise_minor_typo`).

Every transform fails closed if its expected v3 source fragment changes. Every
non-control run records a unique prompt version and SHA-256. The mechanism values,
three risk treatments, hidden horizon, neutral agents, model, decoding parameters,
repetition seeds, and parser remain fixed.

## Estimands

The paired design reports two different quantities:

1. **First-round flip rate versus canonical.** State, history, horizon seed, seat,
   risk treatment, and model sampling seed match exactly. This is the cleanest
   measure of surface sensitivity.
2. **Whole-trajectory UNSAFE-rate shift.** This is the total behavioral effect.
   It includes feedback: an early changed action changes later progress, payoff,
   risk, and history, so later decisions no longer share identical states.

Parse failures, SAFE→UNSAFE flips, UNSAFE→SAFE flips, risk-stratified rates, and
paired horizon equality are audited separately. Smoke results select cells worth
scaling; confirmatory claims require more repetitions and a frozen analysis plan.

## Reproduce on GreenNode

Run one disjoint lane per pod:

```bash
python3 -m kaggle.experiments.greennode_surface_sensitivity \
  --lane a --profile smoke --repo-root . --output-root /persistent/lane-a

python3 -m kaggle.experiments.greennode_surface_sensitivity \
  --lane b --profile smoke --repo-root . --output-root /persistent/lane-b
```

Then validate and summarize both lanes:

```bash
python3 results/scripts/analyze_surface_sensitivity.py \
  --lane-root /persistent/lane-a --lane-root /persistent/lane-b \
  --output-dir /persistent/analysis
```
