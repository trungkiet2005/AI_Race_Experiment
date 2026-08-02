"""Direct-Gemini hosted-model backend for the AI Race engine.

Talks to the real Google Gemini API (``generativelanguage.googleapis.com``)
via the ``google-genai`` SDK, not the Kaggle model proxy. Mirrors
:mod:`ai_race.models.openai_direct` (bounded ``max_tokens``, explicit
temperature, transport retries, bounded concurrency).

Transport failures raise. They are never converted into a game action: a
fabricated Safe choice would enter the panel data as a real decision.

Each Gemini API key has its own daily per-model quota (observed: 250
requests/day for gemini-3.1-pro on a free-tier key), and that quota does not
recover within a run's retry/backoff window (Google's own RetryInfo asked
for ~21h). ``GEMINI_API_KEY_2``, ``GEMINI_API_KEY_3``, ... in the
environment are an optional pool this backend rotates into once the
in-use key reports RESOURCE_EXHAUSTED, instead of burning retries against a
wall that will not move.
"""
from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MAX_TRANSPORT_RETRIES = 6
DEFAULT_CONCURRENCY = 4

_RETRYABLE_HTTP_STATUS = {429, 500, 502, 503, 504}


class GeminiConfigError(RuntimeError):
    pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from ai_race.paths import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")


def gemini_api_keys() -> list[str]:
    """Return the configured Gemini API key pool, primary key first.

    ``GEMINI_API_KEY`` is the primary key; ``GEMINI_API_KEY_2``,
    ``GEMINI_API_KEY_3``, ... extend the pool for quota rotation. Numbering
    stops at the first gap.
    """
    _load_env()
    keys: list[str] = []
    primary = (os.environ.get("GEMINI_API_KEY") or "").strip()
    if primary:
        keys.append(primary)
    index = 2
    while True:
        extra = (os.environ.get(f"GEMINI_API_KEY_{index}") or "").strip()
        if not extra:
            break
        keys.append(extra)
        index += 1
    if not keys:
        raise GeminiConfigError(
            "Missing GEMINI_API_KEY in the environment. Add it to .env at "
            "the repo root."
        )
    return keys


# Both mean "this key is unusable, try the next one" rather than "the
# request was bad": RESOURCE_EXHAUSTED is a key's daily quota (observed: 250
# req/day/model on a free-tier key); PERMISSION_DENIED is observed when a
# key itself has been suspended (CONSUMER_SUSPENDED) rather than anything
# about this particular request.
_KEY_ROTATION_STATUSES = {"RESOURCE_EXHAUSTED", "PERMISSION_DENIED"}


def _is_key_unusable(error: Exception) -> bool:
    return getattr(error, "status", None) in _KEY_ROTATION_STATUSES


def make_gemini_send_batch(
    model_id: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    concurrency: int = DEFAULT_CONCURRENCY,
) -> Callable[..., list[str]]:
    """Build ``send_batch(prompts, seeds=None)`` against the Gemini API.

    ``model_id`` is used verbatim as the Gemini model id (e.g.
    ``"gemini-3-pro"``) — no abstract-name indirection, unlike FAIRGAME's
    ``ChatModelFactory``.
    """
    from google import genai
    from google.genai import types
    from google.genai.errors import APIError

    clients = [genai.Client(api_key=key) for key in gemini_api_keys()]
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    key_lock = threading.Lock()
    active_index = 0  # index into `clients`; advanced on RESOURCE_EXHAUSTED

    def one(prompt: str) -> str:
        nonlocal active_index
        last_error: Optional[Exception] = None
        transport_attempt = 0  # counts only genuine transient retries, not key rotations
        while True:
            with key_lock:
                idx = active_index
            if idx >= len(clients):
                break  # every key in the pool is unusable
            try:
                response = clients[idx].models.generate_content(
                    model=model_id,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        temperature=float(temperature),
                        max_output_tokens=int(max_tokens),
                    ),
                )
                return response.text or ""
            except APIError as error:
                last_error = error
                if _is_key_unusable(error):
                    # Rotating to the next key is bounded by len(clients),
                    # not by max_transport_retries: those are two different
                    # budgets, and sharing one counter for both meant a pool
                    # with more than max_transport_retries dead keys in a
                    # row gave up before ever trying the working ones.
                    #
                    # Checked by the error itself, not by re-reading
                    # active_index first: under concurrency > 1, several
                    # threads can all be mid-flight on the same dead key and
                    # land here together. Each one tries to advance past its
                    # own idx; only whichever gets there first actually moves
                    # the pointer (the rest are no-ops), and every thread
                    # retries against whatever key ends up active — so no
                    # thread is lost to the race, mirroring the fix in
                    # bedrock_direct.py's temperature fallback.
                    with key_lock:
                        if active_index == idx:
                            active_index += 1
                            if active_index < len(clients):
                                print(
                                    f"[gemini_direct] Gemini key #{idx + 1} "
                                    f"is unusable ({error.status}); switching "
                                    f"to key #{active_index + 1} for the "
                                    "rest of this run.",
                                    file=sys.stderr,
                                )
                    continue  # retry now, against whichever key is active
                status = getattr(error, "code", None)
                if status in _RETRYABLE_HTTP_STATUS and transport_attempt < int(max_transport_retries):
                    time.sleep(min(2**transport_attempt, 30))
                    transport_attempt += 1
                    continue
                if status in _RETRYABLE_HTTP_STATUS:
                    break
                raise  # bad model id, auth, validation -> don't retry-mask it
            except Exception as error:  # transport
                last_error = error
                if transport_attempt >= int(max_transport_retries):
                    break
                time.sleep(min(2**transport_attempt, 8))
                transport_attempt += 1

        if isinstance(last_error, APIError) and _is_key_unusable(last_error):
            raise RuntimeError(
                f"All {len(clients)} Gemini API key(s) are unusable (quota "
                f"exhausted or suspended) for model {model_id!r}. Add "
                "another GEMINI_API_KEY_N to .env, or wait for quota to "
                "reset. No fallback action was applied."
            ) from last_error
        raise RuntimeError(
            f"Gemini API call failed after {max_transport_retries} retries "
            f"for model {model_id!r}. No fallback action was applied. Last "
            f"error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def send_batch(
        prompts: list[str],
        seeds: Optional[list[int]] = None,
    ) -> list[str]:
        del seeds  # Gemini's seed config field is not confirmed applied.
        workers = min(int(concurrency), max(len(prompts), 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    return send_batch
