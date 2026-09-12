# %%
"""One structured-output, 1024-token diagnostic probe for GPT-5.5.

The probe is infrastructure validation only and is never ingested as campaign
evidence.
"""

import kaggle_benchmarks as kbench
from pydantic import BaseModel


class ProbeDecision(BaseModel):
    action: str


@kbench.task(name="scripted-opponent-final-gpt55-probe-20260913-v2")
def scripted_opponent_final_gpt55_probe_v2(llm) -> dict:
    route = str(getattr(llm, "model", "")).lower()
    response = llm.prompt(
        "Return a structured decision with action exactly SAFE.",
        schema=ProbeDecision,
        temperature=0.7,
        seed=260726,
        extra_api_params={"max_tokens": 1024},
        reasoning="low",
    )
    action = response.action if isinstance(response, ProbeDecision) else str(response)
    kbench.assertions.assert_equal(
        "SAFE",
        action.upper().strip(),
        expectation="The final GPT-5.5 probe must return SAFE.",
    )
    return {"route": route, "reasoning": "low", "action": action}


scripted_opponent_final_gpt55_probe_v2.run(kbench.llm)
