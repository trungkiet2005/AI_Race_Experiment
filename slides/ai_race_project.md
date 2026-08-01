# Deck map: same game, different words, different behavior

**Format:** 18-25 minute research talk plus questions

**Audience:** multi-agent LLM evaluation, game theory, interpretability, and AI governance

**Status:** validated exploratory pilots; confirmatory promotion blocked by comprehension admission

**Exact open model:** Qwen2.5-7B-Instruct F16, digest `59805ce4...ff16c`

## Narrative

The talk begins with the validity warning, then earns each behavioral claim through four layers: engine correctness, comprehension admission, behavior under controlled perturbations, and causal interpretability controls. The core empirical pattern is strong output sensitivity to surface form, context, opaque action mapping, and decoding temperature. The core scientific boundary is equally strong: the model failed state-update and terminal-scoring admission, and SAE target directions failed causal specificity controls. A separate reduced evolutionary-game reconstruction supplies a theory lens without treating LLM self-play as an evolutionary sample.

## Frame-by-frame evidence map

1. **Title** - current pilot status, exact checkpoint family, two-H100 execution.
2. **Evidence in one slide** - surface span 8.4-89.2%, largest primary T=0 live context contrast +34.0 pp, comprehension failed.
3. **Canonical game** - exact payoff, progress, horizon, and setback mechanics.
4. **Four evidence layers** - engine, admission, behavior, mechanism; each has a separate gate.
5. **Game understanding** - 685 raw probes; overall semantic accuracy 59.1%; unaided 52.1%.
6. **Calculator ablation** - Unsafe rose from 52.0% to 60.8%; disclosed arithmetic is not an internal world model.
7. **Surface sensitivity** - 18 variants, 540 races, 10,044 decisions, zero parse failures; matched first-round emotional-framing flips reached 83.3%.
8. **Context design** - eight payoff-preserving skins and crossed opaque P/Q meanings.
9. **Primary T=0 live/fixed context effects** - first round 0 pp; largest live effect +34.0 pp; live and fixed protocols are not pooled.
10. **Primary T=0 mapping diagnostic** - Safe=P produced 37.2% semantic Unsafe, Safe=Q produced 0%; fixed replay identifies the asymmetry.
11. **Temperature robustness** - T=0.7 raises overall Unsafe by +3.04 pp [2.19, 3.97]; action agreement 88.2%, exact trajectories 62.6%, context-effect rank rho .857; no comprehension claim.
12. **Comprehension admission** - rule recall 100%, state update 12.5%, terminal scoring 17.2%; behavior stays diagnostic.
13. **Recognition audit** - v1 rejected due contradictory schema; v2 accepted 16/16 but remains self-report only.
14. **FAST-SAE design** - actual self-play, whole-race discovery/evaluation split, exact replay, causal controls.
15. **SAE association** - held-out feature associations are screening evidence, not explanations.
16. **Fixed-state steering** - 0/12 target-control intervals excluded zero; reconstruction itself flipped 12.5%.
17. **Live steering** - common-random-number trajectories expose direct flips versus endogenous feedback.
18. **Context SAE** - L12/L20 double-held-out probes and context-shift descriptives.
19. **Context SAE promotion** - promote L20 capture/analysis only; stop causal steering at smoke scale.
20. **Reduced EGT reconstruction** - main reference predicts AU/CAS/CS dominance and 99.2/98.0/1.9% Unsafe; Qwen remains context-dependent. Faithful, not bitwise.
21. **Evidence ledger** - supported versus unsupported claims by layer.
22. **Promotion gates** - comprehension, crossed mapping, model families, matched states, and stronger steering controls.
23. **Bottom line** - same mathematical game can produce different output without establishing informed optimization.
24. **References and provenance** - selected primary sources and exact artifact counts.

## Speaking boundaries

- Say **prompt-conditioned behavior**, not preference, fear, intent, or strategic understanding.
- Say **held-out association** for SAE correlation/AUC; say **causal audit failed** when controls match the target direction.
- Treat full live trajectories as total effects after feedback. Use fixed-state replay for direct prompt effects at sampled states.
- Treat temperature 0 as the primary context result. Report temperature 0.7 only as a separate paired robustness protocol; never pool the rates.
- Do not infer training-data contamination from recognition self-report.
- Do not infer broad realism/fiction effects from three hand-authored pairs.

## EGT reconstruction boundary

The reconstructed reduced strategy system reproduces the disclosed qualitative AU-to-CAS-to-CS shift and validates its transition construction against pinned official EGTTools source. It is not bitwise author-code reproduction because the paper has not released the original code, payoff matrices, EGTTools revision, or Monte Carlo seeds. Nearest-strategy labels for LLM trajectories are a descriptive behavioral lens, not latent-strategy identification.
