# %%
"""One structured-output, 1024-token reservation probe for resumption planning.

This is an infrastructure diagnostic, not an AI Race measurement. It has one
request and must never be ingested into the campaign evidence tree.
"""

import kaggle_benchmarks as kbench
from pydantic import BaseModel


class ProbeDecision(BaseModel):
    action: str


@kbench.task(name="scripted-opponent-resumption-probe")
def scripted_opponent_resumption_probe(llm) -> dict:
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
        expectation="The workload-compatible probe must return SAFE.",
    )
    return {"route": route, "reasoning": reasoning, "action": action}


scripted_opponent_resumption_probe.run(kbench.llm)
