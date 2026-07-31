# Game-understanding and payoff audit protocol

Protocol: `ai-race-game-understanding-v1`  
Frozen before model inference: 2026-07-31

## Claim boundary

This audit tests whether a sampled model response can correctly report and apply
the public AI Race rules under specified prompts. It does **not** identify an
internal world model, subjective understanding, intent, fear, or human-like risk
preference. Behavioral action choice is never scored as proof of comprehension.

The calculator condition reveals verified arithmetic. It is a tool-uptake ceiling,
not an unaided-comprehension condition. Reliability (repeatability and prompt
stability), task validity (rule and arithmetic accuracy), and generalizability
(other models, languages, mechanisms, or deployment contexts) are reported as
separate questions.

## Evidence ledger

| Evidence | Status before GPU run | Admission rule |
|---|---|---|
| Python stage, transition, risk, and terminal oracle | Complete | Unit tests plus exhaustive joint histories through four rounds pass |
| Browser simulator parity | Complete | Same payoff/transition/terminal fixtures pass in Node |
| Non-finite and impossible-state rejection | Complete | Python and browser tests reject invalid mechanism and terminal inputs |
| Model rule/application probes | Prepared | Admit only a completed, uniquely covered, exactly reconstructable raw run |
| Canonical versus calculator-aided behavior | Prepared | Admit only paired seeds/horizons with calculator arithmetic reverified from raw turns |
| Other checkpoints, languages, APIs, or real organizations | Blocked/out of scope | Requires a separately frozen protocol and new evidence |

Pilot results are diagnostic evidence. They cannot silently become confirmatory
evidence or support claims beyond the checkpoint, English prompt, mechanism,
decoding settings, and hardware recorded in the manifests.

## Probe bank and estimands

The bank contains 41 atomic, prespecified items in six domains:

1. rule recall;
2. one-stage payoff lookup;
3. multi-round state reconstruction;
4. one-round state transition;
5. terminal prize/risk/setback scoring; and
6. analytic expected-payoff calculation.

Every item is grounded in the actual rendered behavioral prompt, not a second
hand-written rule summary. Numeric items have direct, paraphrase, and calculator
conditions. Categorical items additionally reverse the displayed answer order in
the direct and paraphrased forms. The repeated sampling seed is paired within each
item/repetition so wording comparisons do not confound the requested RNG stream.

Primary descriptive estimands are semantic accuracy by domain and condition,
strict one-line-format accuracy, semantic-minus-strict recovery, direct-versus-
paraphrase agreement, and forward-versus-reverse answer-order agreement. Strict
and semantic parsing are retained separately so formatting errors cannot be
relabelled as rule errors.

## Behavioral ablation

For each of the three maximum-private-risk treatments, the same baseline games are
run under:

- the canonical prompt; and
- the canonical prompt plus a deterministic four-row decision card enumerating
  own SAFE/UNSAFE by opponent SAFE/UNSAFE.

The card reports only current-round stage payoff, resulting own progress, and
resulting own private risk. It neither predicts the opponent nor reveals the hidden
terminal round. Primary diagnostics are UNSAFE rate, parse health, mean terminal
payoff, and paired first-round action flips. Repetitions are the resampling cluster.

## Frozen execution contract

- Model: `qwen2.5:7b-instruct-fp16` through local Ollama, with exact digest logged.
- Probe decoding: temperature 0.0, maximum 32 generated tokens, fixed seed.
- Behavioral decoding: temperature 0.7, maximum 32 generated tokens, fixed seed.
- Mechanism: checked-in `baseline.json`; risk treatments 0.1, 0.6, and 0.9.
- Smoke: one probe repetition and two behavioral repetitions.
- Pilot: five probe repetitions and ten behavioral repetitions.
- Hardware requirement: NVIDIA H100; hostname, GPU name, Ollama version, Python
  version, source-tree hash, config hash, prompt/context hash, and model digest are
  mandatory manifest fields.
- Raw retention: exact prompt, unmodified response, parse fields, seed, every game
  turn, terminal outcome, and run manifest are persistent artifacts.
- No retry is used for comprehension probes. Behavioral parsing uses the frozen
  baseline retry policy and logs every attempt.

## Promotion and admission gates

Smoke may be promoted to the pilot only when all of the following hold:

1. manifests say `completed`, expected and observed counts agree, and every
   item/condition/repetition key is unique;
2. the analyzer exactly re-renders every logged probe prompt and independently
   re-scores every raw response;
3. source hash, model digest, hardware family, mechanism, and decoding contract
   match the frozen lane specification;
4. canonical/calculator games have identical paired horizons and complete turn and
   terminal records;
5. every calculator fragment is independently recomputed from the logged pre-turn
   state, with zero arithmetic mismatches and no final-round leakage;
6. at least two correct and two wrong/invalid raw examples per available
   domain/condition are manually inspected (all errors when fewer than two); and
7. no failure is explained away by changing the parser or target after seeing the
   responses. Any correction creates a new protocol version and rerun.

The analyzer writes an `admission.json` artifact. Tables or figures enter the paper
only when that artifact admits the relevant evidence and the text preserves the
claim boundary above.

## Reproduction

```bash
python kaggle/experiments/greennode_game_understanding.py \
  --lane probes --profile smoke --repo-root . --output-root <persistent-run-root> \
  --temperature 0.0 --required-gpu H100

python kaggle/experiments/greennode_game_understanding.py \
  --lane behavior --profile smoke --repo-root . --output-root <persistent-run-root> \
  --temperature 0.7 --required-gpu H100

python results/scripts/analyze_game_understanding.py \
  --probe-root <probe-run-root> --behavior-root <behavior-run-root> \
  --output-dir <analysis-root>
```
