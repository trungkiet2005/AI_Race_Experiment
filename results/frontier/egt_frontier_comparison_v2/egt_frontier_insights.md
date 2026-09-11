# Frontier and EGT: a boundary analysis

## Which routes this is about

Two routes, named in full: `anthropic/claude-sonnet-5@default` and
`google/gemini-3-flash-preview`. They are the two that were run through the
evolutionary comparison protocol, and they are not "the admitted routes".
`results/frontier/admission_campaign_v6/derived/admission_campaign_v6.csv`
admits five: Gemini 3 Flash, Claude Opus 5, GPT-5.4, GPT-5.5 and Claude
Sonnet 5. An earlier version of this report and of the figure caption said
"the two admitted frontier routes", which is wrong about the paper's own gate.

## Readout

- At the reference parameter point the reconstructed model is a step in risk:
  99.2% Unsafe at 0.1, 98.0% at 0.6, then 1.9% at 0.9. Neither route does that.
  Claude Sonnet 5 falls 89.2 to 46.2 to 37.6; Gemini 3 Flash falls 98.9 to 74.2
  to 59.1. Both are ramps, and at 0.9 both sit tens of points above a model that
  has already reached the floor.
- The two routes disagree at every checkpoint, by 9.7, 28.0 and 21.5 points.
  A pooled frontier rate would average over a difference larger than most of the
  effects the paper reports, so none is quoted.
- Nearest-rule matching is least unique exactly where the distance to the
  nearest rule is largest. At risk 0.1 Gemini has a unique nearest rule for
  10% of its trajectories; at 0.9 Claude Sonnet 5 has one for 45% of its own,
  at a mean distance of 29.8%. The labels are a lens on the rates, never a
  finding about a latent policy.

## Uncertainty

The evolutionary curves carry a band and the route points do not, and that is
what the artifacts hold rather than a drawing choice.
`egt_stationary_summary.csv` records a minimum and a maximum over four
independent seeded chains, which `reconstruction_manifest.json` calls a
between-chain range diagnostic and not a confidence interval; it is drawn under
that name. `llm_strategy_summary_primary_t0.csv` carries one rate per route and
risk and no interval of any kind, so no interval is drawn on a route point.
