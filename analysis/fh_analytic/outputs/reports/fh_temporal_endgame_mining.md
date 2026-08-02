# FH Temporal / Shadow-Of-The-Future Mining

## Scope And Question

61,824 completed, non-duplicate turn decisions across 3,345 completed races. The horizon is hidden by design -- a player never sees the final round in advance -- so any relationship between unsafe behavior and hindsight-known `rounds_remaining` cannot be literal horizon-aware reasoning. It can only be an emergent side effect of something correlated with lateness inside a specific race (accumulated private risk, opponent's cumulative unsafe count, race-specific momentum). This distinguishes 'closer to the actual end' from the `round_phase` (time since start) buckets used elsewhere, which cannot separate 'round 9 of an 11-round race' from 'round 9 of a 10-round race that is about to stop.'

## Executive Summary

- **Pooled unsafe rate moves from 0.372 at 4+ rounds remaining to 0.368 on the literal final decision of a race** (-0.4% shift). Since the horizon is hidden, this cannot be planned end-game defection; the mechanism most likely runs through accumulated private risk and cumulative unsafe-count, which rise across any race and happen to also rise near a race's actual end.
- **Per-model curves diverge in shape**, not just level -- see the model-split figure and `temporal_endgame_by_rounds_remaining.csv` for whether a given model's curve is flat, rising, or falling toward the end.
- **The single literal final turn of a race is not the same population as 'rounds_remaining=1'** (`temporal_final_turn_vs_rest.csv`) -- final turns pool short and long races together, so a flat aggregate curve can still hide a real within-race trend; use the banded table as the primary evidence.
- **Unsafe rate also varies with how long a race happened to run** (`temporal_horizon_length.csv`): races that ran longer are not random draws from the same behavioral population as races that stopped early, since `stop_forced`/continuation itself partly depends on prior unsafe play through progress and setback mechanics.

## Pooled Endgame Curve (All Completed)

| rounds_remaining_band | n | unsafe_rate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- |
| 0 | 6390 | 0.3678 | 0.3559 | 0.3796 |
| 1 | 6384 | 0.3758 | 0.3639 | 0.3877 |
| 2 | 6378 | 0.3674 | 0.3555 | 0.3792 |
| 3 | 6378 | 0.3785 | 0.3666 | 0.3904 |
| 4plus | 33504 | 0.3722 | 0.367 | 0.3774 |

Visual: `figures/temporal_endgame/01_endgame_curve.png` (gold band marks the human phi_U 95% interval, 40-75%, from `human_reference.json` E7).

## By Model And Scope

| analysis_scope | model_slug | rounds_remaining_band | n | unsafe_rate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| all_completed | google-gemini-3-flash-preview | 0 | 1230 | 0.5236 | 0.4957 | 0.5515 |
| all_completed | google-gemini-3-flash-preview | 1 | 1224 | 0.5196 | 0.4916 | 0.5476 |
| all_completed | google-gemini-3-flash-preview | 2 | 1218 | 0.4778 | 0.4498 | 0.5059 |
| all_completed | google-gemini-3-flash-preview | 3 | 1218 | 0.4639 | 0.4359 | 0.4919 |
| all_completed | google-gemini-3-flash-preview | 4plus | 6156 | 0.4409 | 0.4285 | 0.4533 |
| all_completed | gpt-5-nano | 0 | 2580 | 0.2516 | 0.2348 | 0.2683 |
| all_completed | gpt-5-nano | 1 | 2580 | 0.2667 | 0.2496 | 0.2837 |
| all_completed | gpt-5-nano | 2 | 2580 | 0.2516 | 0.2348 | 0.2683 |
| all_completed | gpt-5-nano | 3 | 2580 | 0.3066 | 0.2888 | 0.3244 |
| all_completed | gpt-5-nano | 4plus | 13674 | 0.263 | 0.2556 | 0.2704 |
| all_completed | gpt-5.4-nano | 0 | 2580 | 0.4097 | 0.3907 | 0.4287 |
| all_completed | gpt-5.4-nano | 1 | 2580 | 0.4167 | 0.3976 | 0.4357 |
| all_completed | gpt-5.4-nano | 2 | 2580 | 0.431 | 0.4119 | 0.4501 |
| all_completed | gpt-5.4-nano | 3 | 2580 | 0.4101 | 0.3911 | 0.4291 |
| all_completed | gpt-5.4-nano | 4plus | 13674 | 0.4505 | 0.4422 | 0.4588 |

Baseline-only scope, for comparison with the rest of the pipeline:

| analysis_scope | model_slug | rounds_remaining_band | n | unsafe_rate | ci95_low | ci95_high |
| --- | --- | --- | --- | --- | --- | --- |
| baseline_completed | google-gemini-3-flash-preview | 0 | 60 | 0.8333 | 0.739 | 0.9276 |
| baseline_completed | google-gemini-3-flash-preview | 1 | 60 | 0.8 | 0.6988 | 0.9012 |
| baseline_completed | google-gemini-3-flash-preview | 2 | 60 | 0.7 | 0.584 | 0.816 |
| baseline_completed | google-gemini-3-flash-preview | 3 | 60 | 0.7333 | 0.6214 | 0.8452 |
| baseline_completed | google-gemini-3-flash-preview | 4plus | 318 | 0.761 | 0.7141 | 0.8079 |
| baseline_completed | google-gemini-3.1-flash-lite-preview | 0 | 60 | 0.8167 | 0.7188 | 0.9146 |
| baseline_completed | google-gemini-3.1-flash-lite-preview | 1 | 60 | 0.8667 | 0.7807 | 0.9527 |
| baseline_completed | google-gemini-3.1-flash-lite-preview | 2 | 60 | 0.7833 | 0.6791 | 0.8876 |
| baseline_completed | google-gemini-3.1-flash-lite-preview | 3 | 60 | 0.8333 | 0.739 | 0.9276 |
| baseline_completed | google-gemini-3.1-flash-lite-preview | 4plus | 318 | 0.8428 | 0.8028 | 0.8828 |
| baseline_completed | google-gemini-3.5-flash-lite | 0 | 60 | 0.7833 | 0.6791 | 0.8876 |
| baseline_completed | google-gemini-3.5-flash-lite | 1 | 60 | 0.7167 | 0.6026 | 0.8307 |
| baseline_completed | google-gemini-3.5-flash-lite | 2 | 60 | 0.7167 | 0.6026 | 0.8307 |
| baseline_completed | google-gemini-3.5-flash-lite | 3 | 60 | 0.65 | 0.5293 | 0.7707 |
| baseline_completed | google-gemini-3.5-flash-lite | 4plus | 318 | 0.7075 | 0.6575 | 0.7575 |
| baseline_completed | gpt-5-nano | 0 | 60 | 0.1 | 0.02409 | 0.1759 |
| baseline_completed | gpt-5-nano | 1 | 60 | 0.08333 | 0.0134 | 0.1533 |
| baseline_completed | gpt-5-nano | 2 | 60 | 0.1167 | 0.03544 | 0.1979 |
| baseline_completed | gpt-5-nano | 3 | 60 | 0.1833 | 0.08542 | 0.2812 |
| baseline_completed | gpt-5-nano | 4plus | 318 | 0.1258 | 0.08934 | 0.1622 |
| baseline_completed | gpt-5.4-nano | 0 | 60 | 0.65 | 0.5293 | 0.7707 |
| baseline_completed | gpt-5.4-nano | 1 | 60 | 0.4833 | 0.3569 | 0.6098 |
| baseline_completed | gpt-5.4-nano | 2 | 60 | 0.5167 | 0.3902 | 0.6431 |
| baseline_completed | gpt-5.4-nano | 3 | 60 | 0.5833 | 0.4586 | 0.7081 |
| baseline_completed | gpt-5.4-nano | 4plus | 318 | 0.5377 | 0.4829 | 0.5925 |

## Final Turn Vs Rest Of Race

| model_slug | is_final_turn | n | unsafe_rate |
| --- | --- | --- | --- |
| google-gemini-3-flash-preview | False | 9816 | 0.4581 |
| google-gemini-3-flash-preview | True | 1230 | 0.5236 |
| gpt-5-nano | False | 21414 | 0.2673 |
| gpt-5-nano | True | 2580 | 0.2516 |
| gpt-5.4-nano | False | 21414 | 0.4392 |
| gpt-5.4-nano | True | 2580 | 0.4097 |

## Unsafe Rate By How Long The Race Actually Ran

| model_slug | horizon_band | n | unsafe_rate |
| --- | --- | --- | --- |
| google-gemini-3-flash-preview | long_9_10 | 3414 | 0.4672 |
| google-gemini-3-flash-preview | mid_7_8 | 1926 | 0.4533 |
| google-gemini-3-flash-preview | short_5_6 | 1290 | 0.4279 |
| google-gemini-3-flash-preview | very_long_11plus | 4416 | 0.4803 |
| gpt-5-nano | long_9_10 | 7224 | 0.2672 |
| gpt-5-nano | mid_7_8 | 3870 | 0.2762 |
| gpt-5-nano | short_5_6 | 2580 | 0.2698 |
| gpt-5-nano | very_long_11plus | 10320 | 0.2595 |
| gpt-5.4-nano | long_9_10 | 7224 | 0.4402 |
| gpt-5.4-nano | mid_7_8 | 3870 | 0.4473 |
| gpt-5.4-nano | short_5_6 | 2580 | 0.4581 |
| gpt-5.4-nano | very_long_11plus | 10320 | 0.4234 |

## Caveats

- `rounds_remaining` is computed in hindsight from the completed race's `n_rounds`; it is never available to the model at decision time. Any effect found here is descriptive/emergent, not evidence of horizon inference.
- Race length (`n_rounds`) is itself endogenous to play (stopping is probabilistic each round from round 5 onward, per the paper-faithful mechanism), so comparing across `rounds_remaining` mixes races of different lengths; the horizon-length table above is provided to make that visible, not to control for it.
- Uses `turns_canonical.csv` joined to `races_canonical.csv` on `(source_run, game_id)`; both filtered to completed, non-duplicate-grain rows.