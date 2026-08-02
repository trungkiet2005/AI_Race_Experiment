"""Direct-OpenAI hosted-model backend for the AI Race engine.

Talks to the real OpenAI API (``api.openai.com``), not the Kaggle model
proxy. Mirrors :mod:`ai_race.models.proxy` (bounded ``max_tokens``, explicit
temperature, transport retries, bounded concurrency) rather than routing
through FAIRGAME's vendored ``OpenAIConnector``, because that connector calls
``chat.completions.create()`` with no ``max_tokens`` at all — output length
(and cost, billed several times higher than input) would be unbounded.

Transport failures raise. They are never converted into a game action: a
fabricated Safe choice would enter the panel data as a real decision.
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TRANSPORT_RETRIES = 6
DEFAULT_CONCURRENCY = 4


class OpenAIConfigError(RuntimeError):
    pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from ai_race.paths import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")


def openai_api_key() -> str:
    _load_env()
    api_key = (os.environ.get("API_KEY_OPENAI") or "").strip()
    if not api_key:
        raise OpenAIConfigError(
            "Missing API_KEY_OPENAI in the environment. Add it to .env at the "
            "repo root (see docs/running-openai-frontier-pilots.md)."
        )
    return api_key


def make_openai_send_batch(
    model_id: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Callable[..., list[str]]:
    """Build ``send_batch(prompts, seeds=None)`` against the real OpenAI API.

    ``model_id`` is used verbatim as the OpenAI model id — no abstract-name
    indirection, unlike FAIRGAME's ``ChatModelFactory``/``MODEL_PROVIDER_MAP``.
    """
    from openai import BadRequestError, OpenAI, RateLimitError

    client = OpenAI(api_key=openai_api_key(), timeout=timeout)
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    # Some gpt-5-series models (observed: gpt-5-nano) reject any explicit
    # temperature other than their default (1) with a 400 unsupported_value.
    # Discovered once per model, not re-probed per prompt: retrying every
    # single prompt through the same failing request would double the
    # request count (and cost) for that model's whole run.
    temperature_supported = True

    # Reasoning-capable gpt-5-series models (observed: gpt-5-nano, gpt-5-mini)
    # can spend the entire max_completion_tokens budget on hidden reasoning
    # tokens for this game's prompt and return an EMPTY visible message with
    # finish_reason="length" — no exception is raised, so this cannot be
    # caught as an error. That silently turns into a 100% parse-failure race
    # (CLAUDE.md: one parse_failed decision voids the whole race), which is
    # far worse than a loud crash. A low reasoning_effort fixes it, but the
    # accepted value string is not consistent across model generations
    # (gpt-5-nano/gpt-5-mini: "minimal"; gpt-5.6: 400 "Unsupported value:
    # 'minimal' ... Supported values are: 'none', 'low', 'medium', 'high',
    # 'xhigh'" — confirmed directly against gpt-5.6-luna/terra/sol on Bedrock
    # Mantle, see ai_race/models/bedrock_mantle_direct.py). Candidates are
    # tried in order and the first one that returns non-empty content wins.
    # Non-reasoning models (gpt-4o-mini, gpt-4.1-mini) 400 on the whole
    # parameter ("Unrecognized request argument"), not on its value, so it is
    # only added after the empty-response signature is actually observed for
    # this model.
    #   None      -> not yet probed
    #   "<value>" -> confirmed needed, send this value on every request
    #   False     -> confirmed unsupported by this model, never send
    _REASONING_EFFORT_CANDIDATES = ("minimal", "none")
    reasoning_effort_state: Optional[str] = None

    def build_request(
        prompt: str, *, force_reasoning_effort: Optional[str] = None
    ) -> dict[str, Any]:
        request: dict[str, Any] = {
            "model": model_id,
            "messages": [{"role": "user", "content": prompt}],
            # gpt-5-series models reject the legacy "max_tokens" param
            # (400 unsupported_parameter) and require "max_completion_tokens";
            # OpenAI's docs say the latter works across current chat models.
            "max_completion_tokens": int(max_tokens),
        }
        if temperature_supported:
            request["temperature"] = float(temperature)
        effort = force_reasoning_effort if force_reasoning_effort is not None else reasoning_effort_state
        if effort:
            request["reasoning_effort"] = effort
        return request

    def reasoning_tokens_spent(completion: Any) -> int:
        details = getattr(completion.usage, "completion_tokens_details", None)
        return int(getattr(details, "reasoning_tokens", 0) or 0)

    def probe_reasoning_effort(prompt: str) -> str:
        """Called once, only when a model's first empty response looks like
        a reasoning-budget exhaustion (empty content, reasoning_tokens > 0).
        Tries the same prompt again against each candidate effort value in
        turn, and remembers the outcome (which value worked / nothing
        helps) for every later prompt sent to this model in this run.

        A BadRequestError on a candidate just moves on to the next one,
        rather than trying to distinguish "this value is rejected" from
        "the whole parameter is rejected" by matching on error text: real
        error strings are not consistent enough across model generations to
        match reliably (gpt-5.6's rejection of 'minimal' does not contain
        the substring "reasoning_effort" at all), and the only cost of not
        distinguishing them is one extra wasted call for a model that
        rejects the parameter outright, versus every prompt afterward
        silently losing this exact race to an unrecovered empty response."""
        nonlocal reasoning_effort_state
        for candidate in _REASONING_EFFORT_CANDIDATES:
            request = build_request(prompt, force_reasoning_effort=candidate)
            try:
                completion = client.chat.completions.create(**request)
            except BadRequestError:
                continue  # this candidate value (or the whole param) is rejected; try the next one
            content = completion.choices[0].message.content or ""
            if content:
                reasoning_effort_state = candidate
                print(
                    f"[openai_direct] {model_id!r} returned an empty message "
                    "after spending its whole token budget on hidden "
                    f"reasoning; sending reasoning_effort={candidate!r} for "
                    "the rest of this run.",
                    file=sys.stderr,
                )
                return content

        reasoning_effort_state = False
        print(
            f"[openai_direct] {model_id!r} returned an empty message after "
            "spending its whole token budget on hidden reasoning; none of "
            f"{_REASONING_EFFORT_CANDIDATES} helped; leaving the empty "
            "response as a genuine parse failure.",
            file=sys.stderr,
        )
        return ""

    def one(prompt: str) -> str:
        nonlocal temperature_supported
        last_error: Optional[Exception] = None
        for attempt in range(int(max_transport_retries) + 1):
            request = build_request(prompt)
            try:
                completion = client.chat.completions.create(**request)
            except BadRequestError as error:
                if temperature_supported and "temperature" in str(error).lower():
                    temperature_supported = False
                    print(
                        f"[openai_direct] {model_id!r} rejected temperature="
                        f"{temperature}; falling back to the model's default "
                        "for the rest of this run.",
                        file=sys.stderr,
                    )
                    continue  # retry immediately with temperature dropped
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                time.sleep(min(2**attempt, 8))
                continue
            except RateLimitError as error:
                # This org's requests-per-minute cap is shared across every
                # concurrent job/model, not per-process — a burst here means
                # the whole account is saturated, not just this one call.
                # Transient-error backoff (a few seconds) is not long enough
                # to ride that out, so this gets its own, much longer wait.
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                wait_s = min(10 * (attempt + 1), 60)
                print(
                    f"[openai_direct] {model_id!r} rate-limited (429); "
                    f"waiting {wait_s}s before retry {attempt + 1}/"
                    f"{max_transport_retries}.",
                    file=sys.stderr,
                )
                time.sleep(wait_s)
                continue
            except Exception as error:  # transport or auth
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                time.sleep(min(2**attempt, 8))
                continue

            content = completion.choices[0].message.content or ""
            if content or reasoning_effort_state is not None:
                return content
            if reasoning_tokens_spent(completion) > 0:
                return probe_reasoning_effort(prompt)
            return content  # empty for some other reason; a genuine parse failure

        raise RuntimeError(
            f"OpenAI API call failed after {max_transport_retries} retries for "
            f"model {model_id!r}. No fallback action was applied. Last error: "
            f"{type(last_error).__name__}: {last_error}"
        ) from last_error

    def send_batch(
        prompts: list[str],
        seeds: Optional[list[int]] = None,
    ) -> list[str]:
        del seeds  # chat.completions accepts a seed but OpenAI does not confirm
        # it was applied; never claim CRN sampling here (samplingSeedApplied
        # stays false in every openai_*.json config).
        workers = min(int(concurrency), max(len(prompts), 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    return send_batch
