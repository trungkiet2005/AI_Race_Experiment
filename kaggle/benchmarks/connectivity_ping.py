# %%
"""Minimal reachability probe for Kaggle Benchmark Model Proxy routes.

Not part of the AI Race protocol: this issues a single trivial prompt per
model to check the route responds, nothing more. Disposable — delete after
use, do not fold into confirmatory tooling.
"""

# %%
import kaggle_benchmarks as kbench


# %%
@kbench.task(name="connectivity-ping")
def connectivity_ping(llm) -> dict:
    response = llm.prompt("Reply with exactly one word: PONG")
    kbench.assertions.assert_in(
        "PONG", response.upper(), expectation="Model must echo PONG"
    )
    return {"response": response}


connectivity_ping.run(kbench.llm)
