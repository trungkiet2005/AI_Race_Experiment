# STATE

## Now
Behaviour paper ported to IJCAI 2027 (`paper/ijcai/main.tex`, `paper/ijcai/supplementary.tex`); AAMAS sources in `paper/` kept as legacy. Body fits 7 pages, references start page 8.
Story reframed after the best-response baseline (2026-11-23): a positive rival contrast and rival x risk interaction are what an exact best responder does; the finding is each endpoint's departure from it (Sonnet 5 reverses the sign on rate and log-odds scales; Gemini 3 Flash and GPT-5.4 flat; GPT-5.5 sign scale-dependent; Opus matches). Title: "Self-Play Rates Are Not Risk Profiles: Frontier LLMs Depart from the Best Response to Fixed Rivals in an AI Race".
Simulated blind IJCAI review round 1 (before reframe): reject, mean 4.3/10 (`paper/reviews/ijcai_blind_round1.md`).

## Lanes and owners
- Mac Claude: Kaggle Benchmarks campaign E2/E8, paper integration, review loop.
- Subagent: E5 code port (`kaggle/benchmarks/ai_race_scripted_context_mapping.py`), code only.

## Running jobs
- Campaign runner `scripts/run_ijcai_revision_campaign.py` (Mac pid 13726, 6 slices at a time), plan `docs/ijcai-revision-collection-plan-2026-11-23.md` + amendment 1. Ledger `results/_build/ijcai_revision/campaign_ledger.jsonl`; per-slice logs `results/_build/ijcai_revision/<slice>/log.txt`. Outputs: `results/frontier/scripted_opponent_replication/block_{B,C}/`, `results/frontier/scripted_opponent_refused/`.

## Next steps
1. When E2/E8 land: analyse blocks B+C (pooled interaction with block factor; best-response departure), refused routes (descriptive, supplement).
2. E4: declare amendment and run Sonnet 5 under the GPT-5.5 contract (reasoning low, 1024 tokens) at AS/AU x 0.1/0.9, block A; rerun Opus AS@0.6 under the active contract. Reserve identities left after E2/E8.
3. E5: after the port is verified offline, declare plan and run AS/AU x 0.1/0.9 x 4 renderings on Gemini 3 Flash and Sonnet 5.
4. Remaining TODO-E3 in main.tex: diversity effect size with interval.
5. Integrate new results into paper, rerun blind review panel (fresh reviewers, no prior rounds), loop until accept.
6. Submission: abstract on the IJCAI site by 2027-01-04 AoE, full paper 2027-01-11 AoE; supplement as separate anonymised PDF/ZIP (<= 50 MB).

## Decisions ledger
- 2026-11-22: target venue for the behaviour paper is IJCAI 2027 (user).
- 2026-11-22: project lives in `~/PhD_Farming/active/AI_Race_Experiment` on the Mac.
- 2026-11-22: paper was never submitted to AAMAS (user), so no dual-submission issue.
- 2026-11-22: user: "do anything for ijcai submission" (full autonomy on port, review loop, experiments, Kaggle quota).
- 2026-11-23: reframe around departure from exact best response (Claude, forced by E1 result).

## Manual steps for the user
- Confirm the manuscript text was written by the authors, with AI tools used only for editing and code (IJCAI AI-use policy; statement is before the references in `paper/ijcai/main.tex`).
- Note: under the standing "method papers only" rule this is a findings paper; the user chose it for IJCAI.
