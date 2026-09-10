# Akata et al. 2025, read rather than cited

*Playing repeated games with large language models*, Nature Human Behaviour
9(7), 1380-1390. Full text pulled from arXiv:2305.16867 and read on 2026-09-10,
because this is the closest published work to our scripted-opponent campaign and
a citation to its abstract would not have settled the question.

## What they actually did

Games: the 2x2 family, with the Prisoner's Dilemma and Battle of the Sexes
analysed in detail. Models: GPT-4, text-davinci-003, text-davinci-002, Claude 2,
Llama 2.

Their scripted opponents, quoted:

> As before, we let GPT-4, text-davinci-003, text-davinci-002, Claude 2 and Llama 2 play against each other. Additionally, we introduce three simplistic strategies. Two of these strategies are simple singleton players, who either always cooperate or defect. Finally, we also introduce an agent who defects in the first round but cooperates in all of the following rounds. We introduced this agent to assess if the different LLMs would start cooperating with this agent again, signaling the potential of building trust. davinci-002 davinci-003 GPT-4 davinci-002 davinci-003 GPT-4 3/15 a Player 1 Defection Rate 100 GPT-4 90 0 90 0 70 0 80 0 text-davinci-003 40 90 10 90 20 90 10 20 80 text-dav

So: Always Cooperate, Always Defect, and Defect Once. Their headline behavioural
finding is that GPT-4 is unforgiving toward the Defect Once agent, and that
telling it explicitly what that agent would do made it defect throughout.

Their stated purpose for the scripted arm, quoted:

> For the two additional games, we also let LLMs play against simple, hand-coded strategies to further understand their behaviour. These simple strategies are designed to assess how LLMs behave when playing with more human-like players. Statistical tests. All reported tests are two-sided. We also report Bayes Factors quantifying the likelihood of the data under HA relative to the likelihood of the data under H0. We calculate the default two-sided Bayesian t-test using a Jeffreys-ZellnerSiow prior with its scale set to 2/2, following75. For parametric tests, the data distribution was assumed to be normal but this was not formally tested. We report effect sizes as either Cohen's d or standardize

Their prompt-robustness battery, quoted in part:

> To make sure that the observed unforgivingness was not due to the particular prompt used, we run several versions of the game as robustness checks, randomising the order of the presented options, relabeling the choice options, and changing the presented utilities to be represented by either points, dollars, or coins (see Figure 4). We also repeated our analysis with two different cover stories, added explicit end goals to our prompt, ran games with longer playing horizons and described numerical outcomes with text, also see Supplementary Figure 3. The results of these simulations showed that the reluctance to forgive was not due to any particular characteristics of the prompts. A crucial que

## What they did not do, and this is the part that matters to us

The string "self-play" does not appear anywhere in the paper, nor does "against
itself" or "copy of itself". Their Figure 3a heatmap has a diagonal, so an
LLM-versus-the-same-LLM cell exists in their data, but they never treat it as a
methodological problem, never contrast it against the fixed-opponent cells as a
validity check, and draw no conclusion about a self-play rate misrepresenting a
model's behaviour.

## Consequence for our claims

- **The scripted-opponent design is not ours to claim.** Always Cooperate and
  Always Defect are their Always Safe and Always Unsafe. Any sentence implying
  we introduce the idea of playing an LLM against fixed strategies is false, and
  a reviewer in this area will know it immediately.
- **Our conditional pair is a different instrument.** Defect Once isolates
  forgiveness after a single defection. Our Conditional Safe and Conditional
  Unsafe are identical from round two, both mirroring the route, and differ only
  in the opening move, so the pair isolates the opening move rather than
  forgiveness.
- **The representation-robustness design is adjacent to theirs too.** They
  relabelled the choice options and swapped the utility units between points,
  dollars and coins. Ours is fully crossed, paired within repetition, and
  identifies the letter anchor algebraically, but "we test whether relabelling
  the options changes play" is not new.
- **What is unclaimed is the self-play finding, not the apparatus that produces
  it.** State it that way: the closest work plays LLMs against fixed strategies
  and against each other and reports neither comparison as a validity result; we
  use the same apparatus to show that a self-play rate is an escalation
  equilibrium rather than a measurement of the policy, and we quantify it.
