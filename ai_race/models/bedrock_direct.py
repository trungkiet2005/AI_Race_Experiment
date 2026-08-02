"""Direct-Bedrock hosted-model backend for the AI Race engine.

Talks to Amazon Bedrock's Converse API (``bedrock-runtime``), not the Kaggle
model proxy and not FAIRGAME's vendored ``anthropic_connector`` (that one
calls the direct Anthropic API with ``API_KEY_ANTHROPIC``; this one calls
Claude, or any other Converse-compatible Bedrock model, through AWS). Mirrors
:mod:`ai_race.models.openai_direct` (bounded ``max_tokens``, explicit
temperature, transport retries, bounded concurrency).

Auth is a Bedrock API key: set ``AWS_BEARER_TOKEN_BEDROCK`` (bearer-token
auth, no SigV4 credential chain needed) and ``AWS_REGION`` in the
environment; boto3 picks the bearer token up automatically from the env var.
This needs a boto3/botocore release new enough to know about Bedrock API
keys (2025+) — an older SDK just ignores the token and Bedrock replies with
a normal missing-credentials error, not a silent fallback.

Transport failures raise. They are never converted into a game action: a
fabricated Safe choice would enter the panel data as a real decision.

Some newer Claude models (observed: ``claude-opus-5`` on Bedrock) reject any
explicit ``temperature`` with ``ValidationException: `temperature` is
deprecated for this model``. Discovered once per model, not re-probed per
prompt, mirroring the equivalent fallback in
:mod:`ai_race.models.openai_direct`.

Claude 5-family models default to extended thinking on Bedrock. Observed
directly on ``claude-opus-5``: with ``thinking`` left at its default, the
model spent the entire ``maxTokens`` budget on a ``reasoningContent`` block
and returned ``stopReason: max_tokens`` with no ``text`` block at all — a
silent, 100%-empty response, not an exception, so it cannot be caught as an
error. That is the same failure class documented in
:mod:`ai_race.models.nvidia_direct` for DeepSeek's ``thinking`` flag and in
:mod:`ai_race.models.openai_direct` for gpt-5-series reasoning budgets, and
under this project's strict one-line ``ACTION: SAFE|UNSAFE`` contract it is
worse than a loud crash (CLAUDE.md: one parse_failed decision voids the
whole race). ``thinking`` therefore defaults to disabled here via
``additionalModelRequestFields``, matching the NVIDIA backend's default.

Disabling ``thinking`` does not stop the model from reasoning in visible
prose instead: observed directly on ``claude-opus-5`` with ``thinking``
disabled, several round-2+ responses opened with `<thinking>`/`<think>`/
`<br>` or a `## Reasoning` header and got truncated by ``maxTokens`` before
ever reaching the required ``ACTION:`` line — same failure class, same
underlying cause (a strict single-line contract vs. a model that wants to
show its work), as the visible chain-of-thought prefacing documented in
:mod:`ai_race.models.nvidia_direct` for DeepSeek. The same fix applies: a
system message demanding output-only compliance, confirmed to clear every
one of 5 previously-failing prompts sampled directly from a contaminated
run.
"""
from __future__ import annotations

import os
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Optional

STRICT_OUTPUT_SYSTEM_MESSAGE = (
    "You must output ONLY the exact single line requested. No explanation, "
    "no reasoning, no extra words before or after."
)

DEFAULT_TEMPERATURE = 0.7
DEFAULT_MAX_TOKENS = 256
DEFAULT_MAX_TRANSPORT_RETRIES = 6
DEFAULT_CONCURRENCY = 4

_RETRYABLE_ERROR_CODES = {
    "ThrottlingException",
    "ServiceUnavailableException",
    "ModelTimeoutException",
    "ModelNotReadyException",
}


class BedrockConfigError(RuntimeError):
    pass


def _load_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    from ai_race.paths import REPO_ROOT

    load_dotenv(REPO_ROOT / ".env")


def bedrock_region() -> str:
    _load_env()
    region = (os.environ.get("AWS_REGION") or "").strip()
    if not region:
        raise BedrockConfigError(
            "Missing AWS_REGION in the environment. Add it to .env at the "
            "repo root."
        )
    return region


def _require_bedrock_credentials() -> None:
    _load_env()
    has_bearer = bool((os.environ.get("AWS_BEARER_TOKEN_BEDROCK") or "").strip())
    has_profile = bool((os.environ.get("AWS_PROFILE") or "").strip())
    has_keys = bool((os.environ.get("AWS_ACCESS_KEY_ID") or "").strip())
    if not (has_bearer or has_profile or has_keys):
        raise BedrockConfigError(
            "No Bedrock credentials found in the environment. Set either "
            "AWS_BEARER_TOKEN_BEDROCK (a Bedrock API key) or a standard "
            "AWS_PROFILE / access-key pair."
        )


def make_bedrock_send_batch(
    model_id: str,
    *,
    temperature: float = DEFAULT_TEMPERATURE,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    max_transport_retries: int = DEFAULT_MAX_TRANSPORT_RETRIES,
    concurrency: int = DEFAULT_CONCURRENCY,
    thinking: bool = False,
) -> Callable[..., list[str]]:
    """Build ``send_batch(prompts, seeds=None)`` against Bedrock Converse.

    ``model_id`` is used verbatim as the Bedrock model id or inference-profile
    ARN/id (e.g. ``"us.anthropic.claude-sonnet-5-..."``) — no abstract-name
    indirection, unlike FAIRGAME's ``ChatModelFactory``.
    """
    import boto3
    from botocore.exceptions import ClientError

    _require_bedrock_credentials()
    client = boto3.client("bedrock-runtime", region_name=bedrock_region())
    if concurrency < 1:
        raise ValueError("concurrency must be positive")

    temperature_supported = True
    additional_fields = {"thinking": {"type": "enabled" if thinking else "disabled"}}

    def build_inference_config() -> dict[str, Any]:
        config: dict[str, Any] = {"maxTokens": int(max_tokens)}
        if temperature_supported:
            config["temperature"] = float(temperature)
        return config

    def one(prompt: str) -> str:
        nonlocal temperature_supported
        last_error: Optional[Exception] = None
        for attempt in range(int(max_transport_retries) + 1):
            try:
                response = client.converse(
                    modelId=model_id,
                    messages=[{"role": "user", "content": [{"text": prompt}]}],
                    system=[{"text": STRICT_OUTPUT_SYSTEM_MESSAGE}],
                    inferenceConfig=build_inference_config(),
                    additionalModelRequestFields=additional_fields,
                )
                blocks = response["output"]["message"]["content"]
                return "".join(block.get("text", "") for block in blocks)
            except ClientError as error:
                code = error.response.get("Error", {}).get("Code", "")
                message = str(error)
                if code == "ValidationException" and "temperature" in message.lower():
                    # Checked by message/code, not by the current value of
                    # temperature_supported: under concurrency > 1, several
                    # threads can all send the first request before any of
                    # them flips the flag, so each one independently lands
                    # here. Whichever thread arrives first prints the notice;
                    # every thread retries with temperature dropped either
                    # way, so no request is lost to the race.
                    if temperature_supported:
                        temperature_supported = False
                        print(
                            f"[bedrock_direct] {model_id!r} rejected an "
                            "explicit temperature; falling back to the "
                            "model's default for the rest of this run.",
                            file=sys.stderr,
                        )
                    continue  # retry immediately with temperature dropped
                last_error = error
                if code in _RETRYABLE_ERROR_CODES and attempt < int(max_transport_retries):
                    time.sleep(min(2**attempt, 30))
                    continue
                if code in _RETRYABLE_ERROR_CODES:
                    break
                raise  # bad model id, auth, validation -> don't retry-mask it
            except Exception as error:  # transport
                last_error = error
                if attempt >= int(max_transport_retries):
                    break
                time.sleep(min(2**attempt, 8))

        raise RuntimeError(
            f"Bedrock Converse call failed after {max_transport_retries} "
            f"retries for model {model_id!r}. No fallback action was "
            f"applied. Last error: {type(last_error).__name__}: {last_error}"
        ) from last_error

    def send_batch(
        prompts: list[str],
        seeds: Optional[list[int]] = None,
    ) -> list[str]:
        del seeds  # Bedrock Converse has no seed/CRN parameter to forward.
        workers = min(int(concurrency), max(len(prompts), 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            return list(pool.map(one, prompts))

    return send_batch
