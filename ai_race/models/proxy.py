"""Hosted-model backend for the AI Race engine.

Talks to the Kaggle model proxy over its OpenAI-compatible ``/openapi`` route, the
same endpoint ``kaggle_benchmarks`` uses. Credentials come from ``MODEL_PROXY_URL``
and ``MODEL_PROXY_API_KEY`` (refresh with ``kaggle benchmarks auth``); the token is
short-lived, so an expired key surfaces here as an authentication error rather than
as a silent fallback action.

Transport failures raise. They are never converted into a game action: a fabricated
Safe choice would enter the panel data as a real decision.
"""
from __future__ import annotations

import os
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_TIMEOUT_SECONDS = 120.0
DEFAULT_MAX_TRANSPORT_RETRIES = 3
DEFAULT_CONCURRENCY = 4


class ProxyConfigError(RuntimeError):
    pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from ai_race.paths import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")


def proxy_credentials() -> tuple[str, str]:
    """Return ``(base_url, api_key)`` for the OpenAI-compatible proxy route."""
    _load_env()
    base_url = (os.environ.get("MODEL_PROXY_URL") or "").strip()
    api_key = (os.environ.get("MODEL_PROXY_API_KEY") or "").strip()
    missing = [
        name
        for name, value in (("MODEL_PROXY_URL", base_url), ("MODEL_PROXY_API_KEY", api_key))
        if not value
    ]
    if missing:
        raise ProxyConfigError(
            f"Missing model-proxy environment variables: {', '.join(missing)}. "
            "Run 'kaggle benchmarks auth' and re-export the .env file."
        )
    base_url = base_url.rstrip("/")
    for suffix in ("/openapi", "/genai"):
        if base_url.endswith(suffix):
            base_url = base_url[: -len(suffix)]
            break
    return base_url + "/openapi", api_key


def proxy_token_expiry() -> Optional[str]:
    _load_env()
    return (os.environ.get("MODEL_PROXY_EXPIRY_TIME") or "").strip() or None


def make_proxy_send_batch(
    model_route: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    timeout: float = DEFAULT_TIMEOUT_SECONDS,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    concurrency: int = DEFAULT_CONCURRENCY,
    send_seed: bool = False,
) -> Callable[..., list[str]]:
    """Build ``send_batch(prompts, seeds=None)`` against one hosted model route.

    ``send_seed`` stays False by default: the proxy does not confirm that a
    forwarded ``seed`` was applied, and PROJECT.md forbids claiming common random
    numbers the provider has not acknowledged.
    """
    from openai import OpenAI

    base_url, api_key = proxy_credentials()
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    def one(prompt: str, seed: Optional[int]) -> str:
        request: dict[str, Any] = {
            "model": model_route,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": float(temperature),
        }
        if send_seed and seed is not None:
            request["seed"] = int(seed)
        last_error: Optional[Exception] = None
        for attempt in range(int(max_transport_retries) + 1):
            try:
                completion = client.chat.completions.create(**request)
                return completion.choices[0].message.content or ""
            except Exception as error:  # transport, rate limit, or auth
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                time.sleep(min(2**attempt, 8))
        raise RuntimeError(
            f"Model proxy call failed after {max_transport_retries} retries for route "
            f"{model_route!r}. Refresh Kaggle Benchmark authentication and resume; no "
            f"fallback action was applied. Last error: {type(last_error).__name__}: "
            f"{last_error}"
        ) from last_error

    def send_batch(
        prompts: list[str],
        seeds: Optional[list[int]] = None,
    ) -> list[str]:
        seed_list: list[Optional[int]] = (
            list(seeds) if seeds is not None else [None] * len(prompts)
        )
        if len(seed_list) != len(prompts):
            raise ValueError("seeds must align with prompts")
        workers = min(int(concurrency), max(len(prompts), 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts, seed_list))

    return send_batch
