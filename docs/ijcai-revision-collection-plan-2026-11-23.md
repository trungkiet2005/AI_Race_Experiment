# IJCAI revision: declared collection plan, 2026-11-23

Written and committed **before** any cell below is started. The assignment is
fixed here and is not adjusted once results arrive. Same rules as
`scripted-opponent-collection-plan-2026-09-10.md` unless stated.

## Why

A blind IJCAI-style review (`paper/reviews/ijcai_blind_round1.md`) asked for
(E2) more matched repetitions on the two unconditional rivals, so the rival by
risk interaction is not resting on ten blocks, and (E8) the scripted-rival
design on the four endpoints the task-validity screen refused, so the exclusion
can be checked rather than asserted.

## Design, frozen

- Task: `kaggle/benchmarks/ai_race_scripted_opponent.py` unchanged
  (protocol `ai-race-scripted-opponent-v1`, prompt `ai-race-fairgame-v3`,
  per-route decoding contract exactly as in the original campaign). Each cell
  is baked with `scripts/bake_scripted_opponent_cell.py`, which changes only the
  strategy, the risk and, for E2, the seed block.
- Ten races per cell, seat counterbalanced five and five, rival replay audited
  at ingestion by `scripts/ingest_scripted_cell.py`.
- **E2, replication blocks.** Rivals AS and AU, risks 0.1, 0.6, 0.9, the five
  admitted routes (`gemini-3-flash-preview`, `gpt-5.4-2026-03-05`,
  `gpt-5.5-2026-04-23`, `claude-sonnet-5-default`, `claude-opus-5-default`).
  Two new seed blocks: B with base seed 270726 and C with base seed 280726
  (game seeds base..base+9; disjoint from the canonical 260726..260735).
  Filed under `results/frontier/scripted_opponent_replication/block_B/` and
  `.../block_C/`, never inside `scripted_opponent_campaign/`.
  Analysis: the canonical block A is reported as before; blocks B and C are a
  pre-declared replication. The pooled A+B+C interaction (30 blocks) is
  reported beside the block-A value, with block as a factor, and the paper says
  which it quotes.
- **E8, refused routes.** Rivals AS and AU, risks 0.1, 0.6, 0.9, canonical
  seed block A (260726), the four refused routes
  (`gemini-3.1-flash-lite-preview`, `gpt-5.4-mini-2026-03-17`,
  `gemini-3.5-flash-lite`, `gpt-5.4-nano-2026-03-17`). Filed under
  `results/frontier/scripted_opponent_refused/`. Reported in the supplement as
  descriptive; never pooled with admitted routes.

## Assignment

One baked slice per identity. The identity pushes that slice once (the push
also runs one validation pass on `gemini-3-flash-preview`, which is never
downloaded into the tree) and then runs it on every route listed for it.

| Slice | Identity | Routes |
|---|---|---|
| E2 B AS 0.1 | boymagic | five admitted |
| E2 B AS 0.6 | chisboiz | five admitted |
| E2 B AS 0.9 | chunaiu | five admitted |
| E2 B AU 0.1 | daosyduyminh | five admitted |
| E2 B AU 0.6 | hnhhctiliu | five admitted |
| E2 B AU 0.9 | kakagotto | five admitted |
| E2 C AS 0.1 | minh2duy | five admitted |
| E2 C AS 0.6 | osduyminh | five admitted |
| E2 C AS 0.9 | tonngohan | five admitted |
| E2 C AU 0.1 | trngbotrn | five admitted |
| E2 C AU 0.6 | vinhdinhthien | five admitted |
| E2 C AU 0.9 | trunkdabest | five admitted |
| E8 A AS 0.1 | kit567 | four refused |
| E8 A AS 0.6 | tnkiet | four refused |
| E8 A AS 0.9 | hunhtrungkit | four refused |
| E8 A AU 0.1 | foundnotkiet | four refused |
| E8 A AU 0.6 | trungkiet | four refused |
| E8 A AU 0.9 | trngthtnhi | four refused |

Each identity sees exactly one rival, one risk and one block, so an identity
effect would show as one inconsistent slice across all routes rather than hide
inside a contrast. Expected load: about 93 route requests per route cell, so
about 560 requests for an E2 identity and 470 for an E8 identity, inside the
observed 800 to 1,400 band.

## Stopping rule

Unchanged. A cell refused on quota (HTTP 403) is kept as a failure record and is
not moved to another identity. A transport failure may be retried on its own
identity. A route missing any cell of a block is not reported as that block's
grid; the supplement lists the cells that exist.

Results worth stating now: if the added blocks move the Gemini 3 Flash and
GPT-5.4 interactions away from zero, the "flat" description changes; if Claude
Sonnet 5 stays negative across blocks, its reversal of the best-response sign
replicates. If refused routes show the same rival ordering, the screen does not
drive the rival result. All outcomes are reported.
