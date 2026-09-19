# %%
"""Resolve which reasoning-parameter contract a route accepts.

Not evidence. One arithmetic question, unrelated to any probe bank, is sent
under each candidate contract in a fixed order until one returns a structured
answer. The result decides the route-resolved reasoning rule used identically
by the three state-probe family tasks (see
docs/arr-transfer-campaign-prereg-2026-09-19.md, section 2).
"""

# %%
import kaggle_benchmarks as kbench
from pydantic import BaseModel, Field

MAX_OUTPUT_TOKENS = 256
CANDIDATES = ("none", "low", None, "high")


class SmokeAnswer(BaseModel):
    answer: str = Field(description="The answer.")


def token_parameter(llm) -> str:
    names = {cls.__name__ for cls in type(llm).__mro__}
    if "GoogleGenAI" in names:
        return "max_output_tokens"
    if "OpenAI" in names:
        return "max_tokens"
    raise RuntimeError(f"Unknown Kaggle Benchmark backend: {sorted(names)}")


@kbench.task(
    name="route-contract-smoke",
    description="Reachability smoke that resolves which reasoning parameter a hosted route accepts. Not an evaluation.",
)
def route_contract_smoke(llm) -> dict:
    route = str(getattr(llm, "model", None) or "unknown")
    extra = {token_parameter(llm): MAX_OUTPUT_TOKENS}
    attempts = []
    accepted = "unresolved"
    for index, candidate in enumerate(CANDIDATES):
        kwargs = {} if candidate is None else {"reasoning": candidate}
        try:
            with kbench.chats.new(f"smoke-{index}", orphan=True):
                response = llm.prompt(
                    "What is 17 plus 25? Give only the number.",
                    schema=SmokeAnswer,
                    temperature=0.0,
                    seed=1,
                    extra_api_params=extra,
                    **kwargs,
                )
            answer = response.answer if isinstance(response, SmokeAnswer) else str(response)
            attempts.append({"reasoning": candidate, "ok": True, "answer": answer})
            accepted = "omit" if candidate is None else candidate
            break
        except Exception as error:
            attempts.append({"reasoning": candidate, "ok": False, "error": f"{type(error).__name__}: {str(error)[:300]}"})
    return {"route": route, "accepted_contract": accepted, "attempts": attempts}


route_contract_smoke.run(kbench.llm)
