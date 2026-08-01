"""Direct NVIDIA API Catalog backend for the AI Race engine.

Talks to NVIDIA's OpenAI-compatible endpoint (``integrate.api.nvidia.com``),
not the Kaggle model proxy and not OpenAI. Mirrors
:mod:`ai_race.models.openai_direct` (bounded ``max_tokens``, explicit
temperature, transport retries, bounded concurrency), reusing the ``openai``
Python SDK with a custom ``base_url`` since NVIDIA's endpoint speaks the same
Chat Completions protocol.

Transport failures raise. They are never converted into a game action: a
fabricated Safe choice would enter the panel data as a real decision.

DeepSeek's V4 models accept a ``chat_template_kwargs.thinking`` flag. Thinking
defaults to off here: a thinking model can spend the whole ``max_tokens``
budget on hidden reasoning and return an empty visible message (the same
failure mode ``openai_direct.py`` documents for gpt-5-series reasoning
models), which is a parse failure under this project's strict one-line
``ACTION: SAFE|UNSAFE`` contract. Turn thinking on only after confirming the
model does not exhaust its budget before writing the action line.

Even with ``thinking`` off, ``deepseek-ai/deepseek-v4-flash`` was observed
prefacing its answer with visible chain-of-thought prose once the game state
got non-trivial (round 3+ of a smoke test), landing the whole message on the
strict-parser's "not exactly one line" failure path 40% of the time. A system
message that demands output-only compliance (not a change to the user-role
game prompt, so ``prompt_version``/its hash are untouched) fixed every
observed case; see ``docs/running-nvidia-frontier-pilots.md``.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
STRICT_OUTPUT_SYSTEM_MESSAGE = (
    "You must output ONLY the exact single line requested. No explanation, "
    "no reasoning, no extra words before or after."
)
DEFAULT_TEMPERATURE = 0.7
DEFAULT_TOP_P = 0.95
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TRANSPORT_RETRIES = 3
DEFAULT_CONCURRENCY = 4


class NvidiaConfigError(RuntimeError):
    pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from ai_race.paths import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")


def nvidia_api_key() -> str:
    _load_env()
    api_key = (os.environ.get("NVIDIA_API_KEY") or "").strip()
    if not api_key:
        raise NvidiaConfigError(
            "Missing NVIDIA_API_KEY in the environment. Add it to .env at the "
            "repo root (see docs/running-nvidia-frontier-pilots.md)."
        )
    return api_key


def make_nvidia_send_batch(
    model_id: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    concurrency: int = DEFAULT_CONCURRENCY,
    thinking: bool = False,
) -> Callable[..., list[str]]:
    """Build ``send_batch(prompts, seeds=None)`` against NVIDIA's API Catalog.

    ``model_id`` is used verbatim (e.g. ``"deepseek-ai/deepseek-v4-flash"``)
    against NVIDIA's OpenAI-compatible Chat Completions endpoint.
    """
    from openai import OpenAI

    client = OpenAI(api_key=nvidia_api_key(), base_url=NVIDIA_BASE_URL, timeout=timeout)
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    def one(prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(int(max_transport_retries) + 1):
            try:
                completion = client.chat.completions.create(
                    model=model_id,
                    messages=[
                        {"role": "system", "content": STRICT_OUTPUT_SYSTEM_MESSAGE},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=float(temperature),
                    top_p=float(top_p),
                    max_tokens=int(max_tokens),
                    extra_body={"chat_template_kwargs": {"thinking": bool(thinking)}},
                )
                return completion.choices[0].message.content or ""
            except Exception as error:  # transport, rate limit, or auth
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"NVIDIA API call failed after {max_transport_retries} retries for "
            f"model {model_id!r}. No fallback action was applied. Last error: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def send_batch(
        prompts: list[str],
        seeds: Optional[list[int]] = None,
    ) -> list[str]:
        del seeds  # NVIDIA does not confirm seed application; never claim CRN here.
        workers = min(int(concurrency), max(len(prompts), 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    return send_batch
