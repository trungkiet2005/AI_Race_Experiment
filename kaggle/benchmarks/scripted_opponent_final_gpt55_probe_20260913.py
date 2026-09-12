# %%
"""One structured-output, 1024-token probe for the final GPT-5.5 cells.

This is an infrastructure diagnostic, not an AI Race measurement. It has one
request and must never be ingested into the campaign evidence tree.
"""

import kaggle_benchmarks as kbench
from pydantic import BaseModel


class ProbeDecision(BaseModel):
    action: str


@kbench.task(name="scripted-opponent-final-gpt55-probe-20260913")
def scripted_opponent_final_gpt55_probe(llm) -> dict:
    route = str(getattr(llm, "model", "")).lower()
    reasoning = "low" if ("claude-opus-5" in route or "gpt-5.5" in route) else None
    kwargs = {} if reasoning is None else {"reasoning": reasoning}
    response = llm.prompt(
        "Return a structured decision with action exactly SAFE.",
        schema=ProbeDecision,
        temperature=0.7,
        seed=260726,
        extra_api_params={"max_tokens": 1024},
        **kwargs,
    )
    action = response.action if isinstance(response, ProbeDecision) else str(response)
    kbench.assertions.assert_equal(
        "SAFE",
        action.upper().strip(),
        expectation="The final GPT-5.5 probe must return SAFE.",
    )
    return {"route": route, "reasoning": reasoning, "action": action}


scripted_opponent_final_gpt55_probe.run(kbench.llm)
