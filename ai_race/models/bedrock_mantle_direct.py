"""Direct-Bedrock-Mantle hosted-model backend for the AI Race engine.

Talks to Amazon Bedrock's ``bedrock-mantle`` endpoint, the separate service
(distinct from ``bedrock-runtime``/Converse used by
:mod:`ai_race.models.bedrock_direct`) that serves OpenAI's GPT-5.6 family
(Luna, Terra, Sol) on Bedrock. This endpoint speaks OpenAI's Responses API,
not Chat Completions or Converse, so it goes through the ``openai`` SDK's
``responses.create()`` with a custom ``base_url`` rather than through either
of the other two backends.

Auth is the same Bedrock API key used by :mod:`ai_race.models.bedrock_direct`
(``AWS_BEARER_TOKEN_BEDROCK``), passed as the OpenAI SDK's ``api_key`` — this
is exactly what AWS's own model-card sample code does. ``AWS_REGION``
selects the in-region endpoint (``https://bedrock-mantle.{region}.api.aws/openai/v1``);
only ``us-east-1``, ``us-east-2``, and ``us-west-2`` serve GPT-5.6 as of the
model cards checked (no Geo/Global cross-region routing for this model).

GPT-5.6 is a reasoning model. Observed directly: with ``reasoning.effort``
left at its default, a single call to the real (long) AI Race game prompt
spent 516 of 526 output tokens on hidden reasoning, and at a bounded
``max_output_tokens=256`` that reasoning budget alone consumes the whole cap
before any visible text is emitted — the same empty-response failure class
documented in :mod:`ai_race.models.openai_direct` (gpt-5-series) and
:mod:`ai_race.models.bedrock_direct` (Claude's ``thinking``). ``none`` is
the only ``reasoning.effort`` value that produced a clean, fast, ~0-token
answer in that same test (``'minimal'`` is rejected outright: this model's
effort levels are ``none``/``low``/``medium``/``high``/``xhigh``, not
OpenAI's older ``minimal``). So ``reasoning_effort`` defaults to ``"none"``
here, matching the "disable reasoning" default already used by
:mod:`ai_race.models.nvidia_direct` and :mod:`ai_race.models.bedrock_direct`.

Transport failures raise. They are never converted into a game action: a
fabricated Safe choice would enter the panel data as a real decision.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_REASONING_EFFORT = "none"
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TRANSPORT_RETRIES = 6
DEFAULT_CONCURRENCY = 4


class BedrockMantleConfigError(RuntimeError):
    pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from ai_race.paths import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")


def bedrock_mantle_api_key() -> str:
    _load_env()
    api_key = (os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip()
    if not api_key:
        raise BedrockMantleConfigError(
            "Missing AWS_BEARER_TOKEN_BEDROCK in the environment. Add it to "
            ".env at the repo root (a Bedrock API key; bedrock-mantle does "
            "not accept plain AWS_PROFILE/access-key SigV4 auth)."
        )
    return api_key


def bedrock_mantle_base_url() -> str:
    _load_env()
    region = (os.environ.get("AWS_REGION") or "").strip()
    if not region:
        raise BedrockMantleConfigError(
            "Missing AWS_REGION in the environment. Add it to .env at the "
            "repo root."
        )
    return f"https://bedrock-mantle.{region}.api.aws/openai/v1"


def make_bedrock_mantle_send_batch(
    model_id: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reasoning_effort: str = DEFAULT_REASONING_EFFORT,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Callable[..., list[str]]:
    """Build ``send_batch(prompts, seeds=None)`` against Bedrock Mantle.

    ``model_id`` is used verbatim as the Bedrock Mantle model id (e.g.
    ``"openai.gpt-5.6-luna"``) — no abstract-name indirection, unlike
    FAIRGAME's ``ChatModelFactory``.
    """
    from openai import BadRequestError, OpenAI, RateLimitError

    client = OpenAI(
        api_key=bedrock_mantle_api_key(),
        base_url=bedrock_mantle_base_url(),
        timeout=timeout,
    )
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    def build_request(prompt: str) -> dict[str, Any]:
        return {
            "model": model_id,
            "input": prompt,
            "max_output_tokens": int(max_tokens),
            "temperature": float(temperature),
            "reasoning": {"effort": reasoning_effort},
        }

    def one(prompt: str) -> str:
        last_error: Optional[Exception] = None
        for attempt in range(int(max_transport_retries) + 1):
            try:
                response = client.responses.create(**build_request(prompt))
                return response.output_text or ""
            except BadRequestError as error:
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                time.sleep(min(2**attempt, 8))
                continue
            except RateLimitError as error:
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                wait_s = min(10 * (attempt + 1), 60)
                time.sleep(wait_s)
                continue
            except Exception as error:  # transport or auth
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                time.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"Bedrock Mantle call failed after {max_transport_retries} "
            f"retries for model {model_id!r}. No fallback action was "
            f"applied. Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def send_batch(
        prompts: list[str],
        seeds: Optional[list[int]] = None,
    ) -> list[str]:
        del seeds  # Bedrock Mantle does not confirm seed application.
        workers = min(int(concurrency), max(len(prompts), 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    return send_batch
