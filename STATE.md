# STATE

## Now
IJCAI 2027 port of the behaviour paper (`paper/ijcai/`), body 7 pages, refs p8-9.
Story (2026-11-23, after the quantal fit): title "Self-Play Rates Are Not Risk Profiles: Five LLM Endpoints Against Fixed Rivals in an AI Race". Two benchmarks bracket the rival x risk interaction sign: exact best response (+33.3) and a noisy (logit quantal) best response fitted per endpoint (negative for all). Sonnet 5 and Gemini 3 Flash match the noisy benchmark (Sonnet -25.9 vs predicted -24.1); Opus 5 and GPT-5.5 show the exact benchmark's sign, which noise cannot produce; Sonnet 5 with reasoning on departs from both (-35.2 vs +0.1). A rival-blind model (race position + fixed belief) fits 4/5 endpoints better than the noisy best response.
Reviews: round 1 (pre-reframe) 4.3/10 reject; round 2 6/5/4, AC borderline reject ~25% (`paper/reviews/ijcai_blind_round2.md`); round 3 running.

## Lanes and owners
- Mac Claude: Kaggle Benchmarks campaign E2/E8, paper integration, review loop.
- Subagent: E5 code port (`kaggle/benchmarks/ai_race_scripted_context_mapping.py`), code only.

## Running jobs
- E4 contract-swap cells (amendment 5, `docs/ijcai-revision-e4-assignment.json`), Mac pid 27068, log `results/_build/ijcai_revision/e4_stdout.log`. Outputs `results/frontier/scripted_opponent_contract_swap/{sonnet_low,opus_active}/`. Do NOT edit `scripts/run_ijcai_revision_slice.sh` while it runs (bash reads scripts incrementally; an edit broke slices on 2026-11-23).

## Collected (2026-11-23)
- E2 block B: complete for Gemini 3 Flash, GPT-5.4, Sonnet 5, Opus 5; GPT-5.5 missing 2 cells (quota). Block C: AU@0.9 missing for all routes (quota), not tabulated.
- E8 refused routes: AS@0.1, AU@0.1, AU@0.6 only.
- Analysis `scripts/analyze_scripted_replication.py` -> `results/derived/scripted_replication/`. Pooled A+B rate interaction: Sonnet -25.9 [-30.9,-20.7], Opus +22.3, Gemini -4.7 (Holm .049), GPT-5.4 +1.1. Integrated into paper (Table 2, supplement S5.3).
- Kaggle quota: most accounts now exhausted; reconcile state with `scripts/reconcile_ijcai_revision.py --dry-run`.

## Next steps
Review rounds 2 and 3 both scored 6/5/4 (AC borderline reject ~25%): text fixes have plateaued; remaining must-items are experiments that need Kaggle Model Proxy quota (exhausted on nearly all accounts 2026-11-23).
1. When quota returns: E5 = scripted AS/AU x risk 0.1/0.9 under the other three prompt renderings (task `kaggle/benchmarks/ai_race_scripted_context_mapping.py`, ready and verified), Sonnet 5 first, then Opus 5 and GPT-5.5. Declare a plan before the first request.
2. Common decoding contract for all five endpoints (reasoning low, 1024 tokens) on AS/AU x 0.1/0.9: Opus and GPT-5.5 already; Sonnet done (E4); add Gemini 3 Flash and GPT-5.4.
3. Complete GPT-5.5 block B (AS@0.6, AU@0.9) and a third block on new seeds for the interaction cells.
4. Remaining round-3 text items: T9/T10 compress Sec 4.3-4.4, T11 novelty vs Akata/Payne, T13 confirmatory definition, T14 figure footnote, T15 block-B column, T19 archive contents, T20 vocabulary; supplement Table 6 code strings; Fig 1c label overlap.
5. Rerun a blind panel after each experiment lands.
6. Submission: IJCAI abstract 2027-01-04 AoE, full paper 2027-01-11 AoE.

## Decisions ledger
- 2026-11-23: claim that Sonnet 'reverses the best response' withdrawn; its interaction matches a fitted noisy best response (Claude, forced by quantal-fit result). Paper reports both benchmarks.
- 2026-11-22: target venue for the behaviour paper is IJCAI 2027 (user).
- 2026-11-22: project lives in `~/PhD_Farming/active/AI_Race_Experiment` on the Mac.
- 2026-11-22: paper was never submitted to AAMAS (user), so no dual-submission issue.
- 2026-11-22: user: "do anything for ijcai submission" (full autonomy on port, review loop, experiments, Kaggle quota).
- 2026-11-23: reframe around departure from exact best response (Claude, forced by E1 result).

## Manual steps for the user
- Confirm the manuscript text was written by the authors, with AI tools used only for editing and code (IJCAI AI-use policy; statement is before the references in `paper/ijcai/main.tex`).
- Note: under the standing "method papers only" rule this is a findings paper; the user chose it for IJCAI.
