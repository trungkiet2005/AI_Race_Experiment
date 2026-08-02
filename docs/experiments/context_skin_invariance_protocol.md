# Context-skin invariance pilot

## Research question

Does the same Qwen2.5-7B-Instruct checkpoint choose differently when the exact
same sequential game is described using a different cover story?

This is a bounded eight-skin pilot. It does not estimate invariance across all
possible narratives, languages, models, or tasks.

## Frozen design

The controls are `technology_race` and domain-neutral `abstract_contest`. Three
matched realistic/fictional pairs add bounded coverage: commercial logistics
(`logistics_contract`, `crystal_guild_contract`), public-safety deployment
(`hospital_deployment`, `colony_life_support`), and neutral exploration
(`robotic_expedition`, `fictional_cartography`). All eight are rendered from one
template skeleton. Only the introduction and domain nouns for actor, progress
unit, terminal reward, and setback differ. These pairs improve internal
comparability; they are not a representative sample of narrative domains.

Every condition keeps the following fixed:

- the 1.0 versus 1.5 progress increments;
- the four stage payoffs 1.0, 0.6, 2.4, and 2.0;
- risk treatments 0.1, 0.6, and 0.9;
- the hidden horizon, minimum five rounds, and 0.2 stopping probability;
- terminal prize, tie rule, winner-only private setback, and state disclosure;
- neutral participant names, response boundary, model digest, decoding, and
  parser;
- game seed and sampling seed for a matched `(risk, repetition, seat)` cell.

The model never sees the strings `SAFE` or `UNSAFE`, including in round-2-plus
history. It responds with opaque
code `P` or `Q`. Presentation order is always P then Q. For even repetitions,
P maps to Safe; for odd repetitions, Q maps to Safe. This balanced mapping is a
separate action-code factor, not part of the narrative-context estimand. The
adapter strictly accepts one `ACTION: P|Q` line, decodes it to the engine action,
and preserves the original P/Q output in `raw_response`.

The test suite proves that all skins have the same mechanism signature and
payoff functions. It also replays one fixed action schedule and requires equal
horizon draws, setback draws, trajectories, stage payoffs, risks, and final
payoffs across all eight skins and both mapping assignments.

## Estimands

Primary direct-effect estimand:

- paired difference in first-round Unsafe choice probability between each
  cover story and `abstract_contest`, macro-averaged over risk and seat;
- pairs are `(model_digest, temperature, risk, repetition, seat,
  action_code_mapping)`.

Secondary total-effect estimands:

- race-level Unsafe fraction and final payoff differences;
- round-2-plus Unsafe differences conditional on the pre-action progress gap;
- parse-failure and retry rates by context and action-code mapping.

First-round decisions occur before endogenous state feedback. Later-round
effects combine continued prompt exposure with context-induced trajectory
changes and must not be described as direct prompt effects. Report action-code
mapping strata and their interaction with context before averaging them.

## Two-pod execution

Run the smoke profile first. Each pod owns a disjoint skin lane and writes only
to its local persistent `/home/jovyan` directory because the shared NFS mount is
not required.

```bash
# Pod A
python -m kaggle.experiments.greennode_context_skin \
  --lane a --profile smoke \
  --repo-root /home/jovyan/AI_Race_Experiment \
  --output-root /home/jovyan/ai_race_runs/context_skin_t0/lane_a \
  --temperature 0.0 --required-gpu H100

# Pod B
python -m kaggle.experiments.greennode_context_skin \
  --lane b --profile smoke \
  --repo-root /home/jovyan/AI_Race_Experiment \
  --output-root /home/jovyan/ai_race_runs/context_skin_t0/lane_b \
  --temperature 0.0 --required-gpu H100
```

Promote both lanes to `--profile pilot` only if the manifests report the exact
model digest, H100, fixed-seed probe success, complete expected races, zero
mixed condition IDs, and inspectable raw P/Q outputs. Temperature 0 is the
primary robustness-to-context run. A separate temperature-0.7 run may be added
under a different output root; never pool the two decoding conditions.

The same experiment JSON can be staged on Kaggle with the repository's native
Transformers runner, but that is a different backend condition and requires its
own smoke, model-revision record, and result directory.

## Connection to activation-level XAI

This workload is the behavioral simulation. A sparse-autoencoder analysis is a
second stage over the prompts logged immediately before these choices. The
primary activation target is the final prompt token before any answer is
generated. Compare matched prompts across contexts at the same risk, state,
seat, repetition, and P/Q mapping. A probe AUC measures predictive separation;
it is not itself gameplay and is not evidence that a latent feature caused an
action. Causal steering requires held-out features, decoder-direction
interventions, matched-norm controls, and a rerun of the choice distribution.
