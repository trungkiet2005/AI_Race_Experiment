# AAMAS hard-review audit and experiment decision

Survey and repository audit frozen: 2026-08-02.

## Decision

The repository has extensive pilot evidence, but it has **not run every planned
experiment and has no comprehension-admitted confirmatory gameplay result**.
The strongest defensible contribution is a validity-gated strategic invariance
study: when behavior survives mechanism-preserving prompt changes, when it is
prompt-conditioned, and which apparent dynamic responses survive controlled
counterfactual tests.

Current submission readiness is strong reject. `paper/main.tex` is a 17-page
generic article, while the checked-in AAMAS sample still contains template
authors and a placeholder submission ID. A future AAMAS submission must use the
current call/template, double-blind metadata, and page limit; the 2026 deadline
has passed.

## What has actually run

| Evidence | Actual coverage | Honest label |
|---|---:|---|
| Qwen2.5-7B surface sensitivity | 540 races / 10,044 turns | pilot |
| Qwen2.5-7B persona | 210 races / 3,906 turns | pilot |
| Qwen2.5-7B context, T=0 | 768 races / 13,680 turns | diagnostic; comprehension failed |
| Qwen2.5-7B context, T=0.7 | 768 races / 13,680 turns | separate robustness diagnostic |
| Fixed-state context replay | 1,536 rows | direct-response diagnostic |
| Context comprehension | 256 rows | failed admission: state 12.5%, terminal 17.2% |
| Frontier baseline | five checkpoints, 150 races | descriptive; provider protocols differ |
| OpenAI persona grid | 2,640 races | pilot; persona/protocol caveats |
| Gemini persona cells | incomplete / asymmetric | partial pilot only |
| Claude Haiku | one usable 9-race smoke | smoke only |
| Qwen N=3 | 96 pilot races across neutral/persona cells | appendix/demo |
| SAE association and interventions | held-out probes and 114 self-play races | association; causal promotion failed |
| Payoff-scale contract | 1,048,512 engine comparisons | mechanical validation only |

Every discovered gameplay aggregate is English. The Vietnamese prompt exists,
but no multilingual gameplay result is present.

## Blockers found before new compute

1. The context-mapping runner produces 16 lane-by-skin manifests, while its
   analyzer accepted only eight. The analyzer has been repaired to require one
   cell for each of two lanes × eight skins and the exact even/odd shard union.
2. Follow-up inference treated `(risk, repetition)` as 96 clusters even though
   all risks reuse `base_seed + repetition`. The corrected unit is 32
   repetition streams; historical intervals and figures were recomputed.
3. The completed mapping pilot still confounds mapping with repetition parity;
   the fully crossed run is mandatory.
4. Mapping identity is not yet separated from display position `[P,Q]`. A small
   two-context mapping × display-order screen is higher value than more random
   synonyms or personas.
5. Temperature-zero repeated first-round prompts are environment replications,
   not stochastic model samples.

## Ranked execution plan

### P0

1. **Admission-first cross-checkpoint smoke.** Run the frozen state/terminal
   comprehension battery before any new gameplay. Failed cells remain complete
   diagnostic evidence and do not silently proceed.
2. **Context × mapping diagnostic.** Eight contexts × two mappings × three risks
   × 32 streams = 1,536 races. Report each risk stratum before the pooled
   repetition-clustered interaction.
3. **Payoff-scale behavior.** Four positive scales × three risks × 32 streams =
   384 races. This tests strategic invariance, not arithmetic access alone.
4. **State × terminal scaffold.** Run 800 admission probes first; only admitted
   cells proceed to 960 gameplay races including placebo.
5. **Mapping × display-order screen.** Abstract plus one frozen high-effect
   context × two mappings × two display orders × three risks × 32 streams = 384
   races.

### P1

- Protocol-matched open checkpoints under one backend and precision. Qwen-7B,
  Qwen-14B, and Gemma-9B are accessible on the authenticated Kaggle account but
  span only two families. Official Llama-3.1-8B is gated; do not substitute an
  unpinned mirror and claim three-family replication.
- Engine-reachable matched histories that change only the opponent's most
  recent action, followed by scripted AS/AU/CS/CAS opponents.
- Replay-to-first-divergence and matched fork continuations to separate direct
  prompt response from endogenous feedback.
- EN/VI/zh parallel comprehension and fixed-state gates with independent human
  equivalence review; only admitted language/checkpoint cells enter live play.

## Persona decision

Do not add prompts such as “I am OpenAI” or “I am Claude.” Those test brand
stereotypes, not provider behavior. If role prompting remains a research
question, the smallest identifiable grid is no persona, length-matched neutral,
generic CEO, safety-mandate CEO, and growth-mandate CEO, with fictional company
names, seat balance, and multiplicity correction. It remains a role-prompt
effect, never evidence about real CEOs.

## Prospective stopping rule

The 32-stream mapping run is diagnostic. The power sensitivity uses the largest
observed repetition-level context-delta spread as a conservative proxy because
the parity-confounded pilot cannot identify mapping-interaction variance. A
separate 96-stream replication is frozen for a 15-percentage-point smallest
effect of scientific interest, 80% target power, Holm family size seven, fixed
N, and no optional continuation. The simulation estimates 93.7% design power at
96 streams; it is not a behavioral finding.

## Primary literature alignment

- [AAMAS 2026 proceedings](https://www.ifaamas.org/Proceedings/aamas2026/forms/contents.htm)
  already include repeated-game LLM work, so novelty cannot be “an LLM plays a
  repeated game.”
- [Akata et al., Nature Human Behaviour 2025](https://www.nature.com/articles/s41562-025-02172-y)
  use multiple models, scripted opponents, relabel/order, utility units, cover
  stories, and opponent prediction; the present paper should lead with validity
  decomposition and admission gates.
- [Pezeshkpour and Hruschka, NAACL Findings 2024](https://aclanthology.org/2024.findings-naacl.130/)
  establish option-order sensitivity, motivating mapping × position controls.
- [Wei et al., ACL Findings 2024](https://aclanthology.org/2024.findings-acl.333/)
  separate order and token selection biases.
- [POSIX, EMNLP Findings 2024](https://aclanthology.org/2024.findings-emnlp.852/)
  motivates sensitivity ranges rather than one preferred prompt.
- [Principled Personas, EMNLP 2025](https://aclanthology.org/2025.emnlp-main.1364/)
  shows persona effects are inconsistent and require construct-specific design.
- [RouteSAE](https://arxiv.org/abs/2503.08200) and
  [SAIF](https://arxiv.org/abs/2502.11356) motivate held-out controls and causal
  intervention tests; high decoder AUC alone is not an explanation.

## Infrastructure receipt

- GreenNode: both TCP ports are reachable, but explicit `~/.ssh/id_rsa` with
  `IdentitiesOnly=yes` is rejected by Pod A and Pod B with exit 255
  (`Permission denied (publickey,password)`); no process inspection or job
  launch was authorized by the remote hosts.
- Kaggle: authenticated as `daosyduyminh`, 30/30 GPU hours available. The old
  metadata points to inaccessible private `foundnotkiet/*` inputs.
- A new private admission kernel and pinned source Dataset were launched under
  the authenticated account using accessible pinned model sources. Versions
  1--4 terminated in error before producing an admissible output; v2 and v3
  logs identify source-mount discovery failures, while the v4 log endpoint was
  rate-limited during this release audit. These are failed-run receipts, not
  cross-model evidence. No pilot scale-up is permitted until a smoke version
  produces and passes the frozen 160-row validation contract.
