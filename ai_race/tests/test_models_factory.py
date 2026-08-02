"""Tests for the backend dispatcher: direct OpenAI, Bedrock, and Gemini branches.

These direct backends exist so a frontier model can be named directly in an
experiment config's ``models`` list, the same way the ``proxy`` backend takes
a raw provider slug — without registering an abstract name in FAIRGAME's
vendored ``MODEL_PROVIDER_MAP`` for every model tried, and with a bounded
``max_tokens`` (FAIRGAME's own connectors don't all bound it).
"""
from __future__ import annotations

import threading
from types import SimpleNamespace

import pytest

from ai_race.models.factory import get_send_batch
from ai_race.models.openai_direct import OpenAIConfigError

NOT_IN_MODEL_PROVIDER_MAP = "gpt-5.6-luna"


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported model backend"):
        get_send_batch("some-model", offline=False, backend="not-a-backend")


def test_openai_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY_OPENAI", raising=False)
    # Isolate from a real .env (this repo's .env may legitimately have a key
    # once someone is running pilots) so the test exercises the missing-key
    # path regardless of local machine state.
    import ai_race.models.openai_direct as openai_direct_module

    monkeypatch.setattr(openai_direct_module, "_load_env", lambda: None)
    with pytest.raises(OpenAIConfigError, match="API_KEY_OPENAI"):
        get_send_batch(NOT_IN_MODEL_PROVIDER_MAP, offline=False, backend="openai")


class _FakeCompletions:
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def create(self, **request):
        self._calls.append(request)

        class _Message:
            content = f"ACTION: SAFE  # {request['model']}"

        class _Choice:
            message = _Message()

        class _Completion:
            choices = [_Choice()]

        return _Completion()


class _FakeChat:
    def __init__(self, calls: list[dict]) -> None:
        self.completions = _FakeCompletions(calls)


class _FakeOpenAIClient:
    def __init__(self, calls: list[dict], **_kwargs) -> None:
        self.chat = _FakeChat(calls)


def test_openai_backend_accepts_a_raw_model_id_and_caps_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("API_KEY_OPENAI", "sk-test-not-real")
    calls: list[dict] = []

    import openai as openai_sdk

    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: _FakeOpenAIClient(calls, **kw))

    send_batch = get_send_batch(
        NOT_IN_MODEL_PROVIDER_MAP,
        offline=False,
        backend="openai",
        proxy_options={"temperature": 0.7, "max_tokens": 256, "concurrency": 1},
    )
    results = send_batch(["p1", "p2"], seeds=[1, 2])

    assert results == [
        "ACTION: SAFE  # gpt-5.6-luna",
        "ACTION: SAFE  # gpt-5.6-luna",
    ]
    assert len(calls) == 2
    for call in calls:
        assert call["model"] == NOT_IN_MODEL_PROVIDER_MAP
        assert call["max_completion_tokens"] == 256
        assert call["temperature"] == 0.7


def test_openai_backend_defaults_are_bounded(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("API_KEY_OPENAI", "sk-test-not-real")
    calls: list[dict] = []

    import openai as openai_sdk

    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: _FakeOpenAIClient(calls, **kw))

    send_batch = get_send_batch(NOT_IN_MODEL_PROVIDER_MAP, offline=False, backend="openai")
    send_batch(["p1"])

    assert calls[0]["max_completion_tokens"] == 256
    assert calls[0]["temperature"] == 0.7


def test_openai_backend_drops_temperature_when_model_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some gpt-5-series models (observed: gpt-5-nano) 400 on any explicit
    temperature. The backend must fall back to the model default instead of
    surfacing the error, and must remember the fallback so it does not repeat
    the failing request for every subsequent prompt in the same run."""
    monkeypatch.setenv("API_KEY_OPENAI", "sk-test-not-real")
    calls: list[dict] = []

    import httpx
    from openai import BadRequestError

    class _RejectsTemperatureCompletions:
        def create(self, **request):
            calls.append(request)
            if "temperature" in request:
                response = httpx.Response(
                    400,
                    request=httpx.Request(
                        "POST", "https://api.openai.com/v1/chat/completions"
                    ),
                )
                raise BadRequestError(
                    "Unsupported value: 'temperature' does not support 0.7 "
                    "with this model. Only the default (1) value is supported.",
                    response=response,
                    body=None,
                )

            class _Message:
                content = "ACTION: SAFE"

            class _Choice:
                message = _Message()

            class _Completion:
                choices = [_Choice()]

            return _Completion()

    class _FakeChatRejects:
        def __init__(self) -> None:
            self.completions = _RejectsTemperatureCompletions()

    class _FakeClientRejects:
        def __init__(self, **_kwargs) -> None:
            self.chat = _FakeChatRejects()

    import openai as openai_sdk

    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: _FakeClientRejects(**kw))

    send_batch = get_send_batch(
        "gpt-5-nano",
        offline=False,
        backend="openai",
        proxy_options={"concurrency": 1},
    )
    results = send_batch(["p1", "p2"])

    assert results == ["ACTION: SAFE", "ACTION: SAFE"]
    assert "temperature" in calls[0]
    assert all("temperature" not in call for call in calls[1:])


def _make_completion(content: str, *, reasoning_tokens: int = 0) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content))],
        usage=SimpleNamespace(
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens)
        ),
    )


def test_openai_backend_recovers_from_reasoning_budget_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Reasoning-capable models (observed: gpt-5-nano) can spend the whole
    max_completion_tokens budget on hidden reasoning and return an empty
    message with no error raised at all — this would otherwise show up as a
    100% parse-failure race. The backend must detect that signature (empty
    content + reasoning_tokens > 0), retry with reasoning_effort='minimal',
    and remember it for the rest of the model's run."""
    monkeypatch.setenv("API_KEY_OPENAI", "sk-test-not-real")
    calls: list[dict] = []

    class _BudgetExhaustedCompletions:
        def create(self, **request):
            calls.append(request)
            if request.get("reasoning_effort") == "minimal":
                return _make_completion("ACTION: SAFE")
            # Simulates the model burning the whole budget on hidden reasoning.
            return _make_completion("", reasoning_tokens=256)

    class _FakeChatBudget:
        def __init__(self) -> None:
            self.completions = _BudgetExhaustedCompletions()

    class _FakeClientBudget:
        def __init__(self, **_kwargs) -> None:
            self.chat = _FakeChatBudget()

    import openai as openai_sdk

    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: _FakeClientBudget(**kw))

    send_batch = get_send_batch(
        "gpt-5-nano",
        offline=False,
        backend="openai",
        proxy_options={"concurrency": 1},
    )
    results = send_batch(["p1", "p2"])

    assert results == ["ACTION: SAFE", "ACTION: SAFE"]
    # p1: first call has no reasoning_effort and comes back empty; the probe
    # call (2nd) forces it and succeeds. p2 (3rd call) should send
    # reasoning_effort from the start — no wasted empty call this time.
    assert len(calls) == 3
    assert calls[0].get("reasoning_effort") is None
    assert calls[1]["reasoning_effort"] == "minimal"
    assert calls[2]["reasoning_effort"] == "minimal"


def test_openai_backend_falls_back_to_none_when_model_rejects_minimal_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: gpt-5.6 (observed: gpt-5.6-luna/terra, confirmed
    directly against Bedrock Mantle) 400s on reasoning_effort='minimal' with
    'Unsupported value... Supported values are: none, low, medium, high,
    xhigh' -- a *value* rejection, not a *parameter* rejection. The old code
    treated any BadRequestError mentioning reasoning_effort as "model
    doesn't support this at all" and gave up permanently, which is exactly
    what turned into the 14-42% real parse-failure rate observed in
    openai_terra pilot data. The probe must fall through to the next
    candidate value instead of giving up on the first rejected one."""
    monkeypatch.setenv("API_KEY_OPENAI", "sk-test-not-real")
    calls: list[dict] = []

    import httpx
    from openai import BadRequestError

    class _RejectsMinimalValueCompletions:
        def create(self, **request):
            calls.append(request)
            if request.get("reasoning_effort") == "minimal":
                response = httpx.Response(
                    400,
                    request=httpx.Request("POST", "https://api.openai.com/v1/chat/completions"),
                )
                raise BadRequestError(
                    "Unsupported value: 'minimal' is not supported with the "
                    "'gpt-5.6-luna' model. Supported values are: 'none', "
                    "'low', 'medium', 'high', and 'xhigh'.",
                    response=response,
                    body=None,
                )
            if request.get("reasoning_effort") == "none":
                return _make_completion("ACTION: SAFE")
            # No reasoning_effort sent yet: simulate burning the whole budget.
            return _make_completion("", reasoning_tokens=256)

    class _FakeChatRejectsMinimal:
        def __init__(self) -> None:
            self.completions = _RejectsMinimalValueCompletions()

    class _FakeClientRejectsMinimal:
        def __init__(self, **_kwargs) -> None:
            self.chat = _FakeChatRejectsMinimal()

    import openai as openai_sdk

    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: _FakeClientRejectsMinimal(**kw))

    send_batch = get_send_batch(
        "gpt-5.6-luna",
        offline=False,
        backend="openai",
        proxy_options={"concurrency": 1},
    )
    results = send_batch(["p1", "p2"])

    assert results == ["ACTION: SAFE", "ACTION: SAFE"]
    # p1: empty (no effort) -> probe tries "minimal" (400) -> probe tries
    # "none" (succeeds) = 3 calls. p2 sends "none" straight away = 1 call.
    assert len(calls) == 4
    assert calls[0].get("reasoning_effort") is None
    assert calls[1]["reasoning_effort"] == "minimal"
    assert calls[2]["reasoning_effort"] == "none"
    assert calls[3]["reasoning_effort"] == "none"


def test_bedrock_backend_requires_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    for var in ("AWS_BEARER_TOKEN_BEDROCK", "AWS_PROFILE", "AWS_ACCESS_KEY_ID", "AWS_REGION"):
        monkeypatch.delenv(var, raising=False)
    import ai_race.models.bedrock_direct as bedrock_direct_module

    monkeypatch.setattr(bedrock_direct_module, "_load_env", lambda: None)
    from ai_race.models.bedrock_direct import BedrockConfigError

    with pytest.raises(BedrockConfigError, match="No Bedrock credentials"):
        get_send_batch(NOT_IN_MODEL_PROVIDER_MAP, offline=False, backend="bedrock")


class _FakeBedrockClient:
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def converse(self, **request):
        self._calls.append(request)
        return {
            "output": {
                "message": {
                    "content": [{"text": f"ACTION: SAFE  # {request['modelId']}"}]
                }
            }
        }


def test_bedrock_backend_accepts_a_raw_model_id_and_caps_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    calls: list[dict] = []

    import boto3

    import ai_race.models.bedrock_direct as bedrock_direct

    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _FakeBedrockClient(calls))

    send_batch = get_send_batch(
        NOT_IN_MODEL_PROVIDER_MAP,
        offline=False,
        backend="bedrock",
        proxy_options={"temperature": 0.7, "max_tokens": 256, "concurrency": 1},
    )
    results = send_batch(["p1", "p2"], seeds=[1, 2])

    assert results == [
        "ACTION: SAFE  # gpt-5.6-luna",
        "ACTION: SAFE  # gpt-5.6-luna",
    ]
    assert len(calls) == 2
    for call in calls:
        assert call["modelId"] == NOT_IN_MODEL_PROVIDER_MAP
        assert call["inferenceConfig"]["maxTokens"] == 256
        assert call["inferenceConfig"]["temperature"] == 0.7
        assert call["additionalModelRequestFields"] == {"thinking": {"type": "disabled"}}
        assert call["system"] == [{"text": bedrock_direct.STRICT_OUTPUT_SYSTEM_MESSAGE}]


def test_bedrock_backend_can_enable_thinking(monkeypatch: pytest.MonkeyPatch) -> None:
    """thinking defaults to disabled (observed: claude-opus-5 silently burns
    the whole maxTokens budget on a reasoningContent block and returns no
    text at all otherwise), but callers that want it can still opt in."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    calls: list[dict] = []

    import boto3

    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _FakeBedrockClient(calls))

    send_batch = get_send_batch(
        NOT_IN_MODEL_PROVIDER_MAP,
        offline=False,
        backend="bedrock",
        proxy_options={"concurrency": 1, "thinking": True},
    )
    send_batch(["p1"])

    assert calls[0]["additionalModelRequestFields"] == {"thinking": {"type": "enabled"}}


def test_bedrock_backend_drops_temperature_when_model_rejects_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Some Claude models on Bedrock (observed: claude-opus-5) reject any
    explicit temperature with ValidationException('`temperature` is
    deprecated for this model'). The backend must fall back to the model
    default instead of surfacing the error, and remember the fallback so it
    does not repeat the failing request for every subsequent prompt."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    calls: list[dict] = []

    import boto3
    from botocore.exceptions import ClientError

    class _RejectsTemperatureBedrockClient:
        def converse(self, **request):
            calls.append(request)
            if "temperature" in request["inferenceConfig"]:
                raise ClientError(
                    {
                        "Error": {
                            "Code": "ValidationException",
                            "Message": "`temperature` is deprecated for this model.",
                        }
                    },
                    "Converse",
                )
            return {"output": {"message": {"content": [{"text": "ACTION: SAFE"}]}}}

    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _RejectsTemperatureBedrockClient())

    send_batch = get_send_batch(
        "us.anthropic.claude-opus-5",
        offline=False,
        backend="bedrock",
        proxy_options={"concurrency": 1},
    )
    results = send_batch(["p1", "p2"])

    assert results == ["ACTION: SAFE", "ACTION: SAFE"]
    assert "temperature" in calls[0]["inferenceConfig"]
    assert all("temperature" not in call["inferenceConfig"] for call in calls[1:])


def test_bedrock_backend_survives_concurrent_temperature_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: with concurrency > 1, several worker threads can all
    send their first request with temperature set before any of them
    observes the fallback flag flip. A barrier forces exactly that pileup
    here. Every thread must still retry and succeed -- earlier code raised
    for whichever thread lost the race to flip the flag first, because it
    re-checked the (already-flipped) flag instead of the error itself."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    calls: list[dict] = []
    calls_lock = threading.Lock()
    first_wave = threading.Barrier(4, timeout=5)

    import boto3
    from botocore.exceptions import ClientError

    class _RejectsTemperatureConcurrentClient:
        def converse(self, **request):
            with calls_lock:
                calls.append(request)
            if "temperature" in request["inferenceConfig"]:
                first_wave.wait()  # force all 4 first attempts to land together
                raise ClientError(
                    {
                        "Error": {
                            "Code": "ValidationException",
                            "Message": "`temperature` is deprecated for this model.",
                        }
                    },
                    "Converse",
                )
            return {"output": {"message": {"content": [{"text": "ACTION: SAFE"}]}}}

    monkeypatch.setattr(boto3, "client", lambda *a, **kw: _RejectsTemperatureConcurrentClient())

    send_batch = get_send_batch(
        "us.anthropic.claude-opus-5",
        offline=False,
        backend="bedrock",
        proxy_options={"concurrency": 4},
    )
    results = send_batch(["p1", "p2", "p3", "p4"])

    assert results == ["ACTION: SAFE"] * 4


def test_bedrock_mantle_backend_requires_bearer_token(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AWS_BEARER_TOKEN_BEDROCK", raising=False)
    import ai_race.models.bedrock_mantle_direct as bedrock_mantle_module

    monkeypatch.setattr(bedrock_mantle_module, "_load_env", lambda: None)
    from ai_race.models.bedrock_mantle_direct import BedrockMantleConfigError

    with pytest.raises(BedrockMantleConfigError, match="AWS_BEARER_TOKEN_BEDROCK"):
        get_send_batch(NOT_IN_MODEL_PROVIDER_MAP, offline=False, backend="bedrock-mantle")


def test_bedrock_mantle_backend_requires_region(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    monkeypatch.delenv("AWS_REGION", raising=False)
    import ai_race.models.bedrock_mantle_direct as bedrock_mantle_module

    monkeypatch.setattr(bedrock_mantle_module, "_load_env", lambda: None)
    from ai_race.models.bedrock_mantle_direct import BedrockMantleConfigError

    with pytest.raises(BedrockMantleConfigError, match="AWS_REGION"):
        get_send_batch(NOT_IN_MODEL_PROVIDER_MAP, offline=False, backend="bedrock-mantle")


class _FakeResponsesEndpoint:
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def create(self, **request):
        self._calls.append(request)
        return SimpleNamespace(output_text=f"ACTION: SAFE  # {request['model']}")


class _FakeMantleClient:
    def __init__(self, calls: list[dict], **_kwargs) -> None:
        self.responses = _FakeResponsesEndpoint(calls)


def test_bedrock_mantle_backend_accepts_a_raw_model_id_and_disables_reasoning_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GPT-5.6 on Bedrock Mantle is a reasoning model: left at its default
    reasoning effort, a single call to the real AI Race game prompt spent
    516 of 526 output tokens on hidden reasoning and would blow a bounded
    max_output_tokens budget before emitting any visible text (the same
    failure class as gpt-5-series in openai_direct.py and Claude's
    ``thinking`` in bedrock_direct.py). reasoning.effort must default to
    'none'."""
    monkeypatch.setenv("AWS_BEARER_TOKEN_BEDROCK", "not-a-real-token")
    monkeypatch.setenv("AWS_REGION", "us-east-1")
    calls: list[dict] = []

    import openai as openai_sdk

    monkeypatch.setattr(openai_sdk, "OpenAI", lambda **kw: _FakeMantleClient(calls, **kw))

    send_batch = get_send_batch(
        "openai.gpt-5.6-luna",
        offline=False,
        backend="bedrock-mantle",
        proxy_options={"temperature": 0.7, "max_tokens": 256, "concurrency": 1},
    )
    results = send_batch(["p1", "p2"], seeds=[1, 2])

    assert results == [
        "ACTION: SAFE  # openai.gpt-5.6-luna",
        "ACTION: SAFE  # openai.gpt-5.6-luna",
    ]
    assert len(calls) == 2
    for call in calls:
        assert call["model"] == "openai.gpt-5.6-luna"
        assert call["input"] in ("p1", "p2")
        assert call["max_output_tokens"] == 256
        assert call["temperature"] == 0.7
        assert call["reasoning"] == {"effort": "none"}


def _delenv_gemini_key_pool(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    for i in range(2, 10):
        monkeypatch.delenv(f"GEMINI_API_KEY_{i}", raising=False)


def test_gemini_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    _delenv_gemini_key_pool(monkeypatch)
    import ai_race.models.gemini_direct as gemini_direct_module

    monkeypatch.setattr(gemini_direct_module, "_load_env", lambda: None)
    from ai_race.models.gemini_direct import GeminiConfigError

    with pytest.raises(GeminiConfigError, match="GEMINI_API_KEY"):
        get_send_batch(NOT_IN_MODEL_PROVIDER_MAP, offline=False, backend="gemini")


class _FakeGeminiModels:
    def __init__(self, calls: list[dict]) -> None:
        self._calls = calls

    def generate_content(self, **request):
        self._calls.append(request)
        return SimpleNamespace(text=f"ACTION: SAFE  # {request['model']}")


class _FakeGeminiClient:
    def __init__(self, calls: list[dict], **_kwargs) -> None:
        self.models = _FakeGeminiModels(calls)


def test_gemini_backend_accepts_a_raw_model_id_and_caps_max_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _delenv_gemini_key_pool(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "not-a-real-key")
    calls: list[dict] = []

    import google.genai as genai_sdk

    monkeypatch.setattr(genai_sdk, "Client", lambda **kw: _FakeGeminiClient(calls, **kw))

    send_batch = get_send_batch(
        "gemini-3-pro",
        offline=False,
        backend="gemini",
        proxy_options={"temperature": 0.7, "max_tokens": 256, "concurrency": 1},
    )
    results = send_batch(["p1", "p2"], seeds=[1, 2])

    assert results == ["ACTION: SAFE  # gemini-3-pro", "ACTION: SAFE  # gemini-3-pro"]
    assert len(calls) == 2
    for call in calls:
        assert call["model"] == "gemini-3-pro"
        assert call["config"].max_output_tokens == 256
        assert call["config"].temperature == 0.7


def _quota_exhausted_error():
    from google.genai.errors import APIError

    return APIError(
        429,
        {
            "error": {
                "code": 429,
                "message": "You exceeded your current quota",
                "status": "RESOURCE_EXHAUSTED",
            }
        },
    )


def test_gemini_backend_rotates_to_next_key_on_quota_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backoff/retry cannot fix a daily quota wall (Google's own RetryInfo
    asks for ~21h). Once GEMINI_API_KEY reports RESOURCE_EXHAUSTED, the
    backend must switch to GEMINI_API_KEY_2 instead of exhausting its retry
    budget against the same dead key."""
    _delenv_gemini_key_pool(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-1-exhausted")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-2-fresh")
    calls: list[dict] = []

    import google.genai as genai_sdk

    class _FailingModels:
        def generate_content(self, **request):
            raise _quota_exhausted_error()

    def fake_client(**kw):
        if kw.get("api_key") == "key-1-exhausted":
            return SimpleNamespace(models=_FailingModels())
        return SimpleNamespace(models=_FakeGeminiModels(calls))

    monkeypatch.setattr(genai_sdk, "Client", fake_client)

    send_batch = get_send_batch(
        "gemini-3-pro",
        offline=False,
        backend="gemini",
        proxy_options={"concurrency": 1},
    )
    results = send_batch(["p1", "p2"])

    assert results == ["ACTION: SAFE  # gemini-3-pro", "ACTION: SAFE  # gemini-3-pro"]
    assert len(calls) == 2


def _suspended_key_error():
    from google.genai.errors import APIError

    return APIError(
        403,
        {
            "error": {
                "code": 403,
                "message": "Permission denied: Consumer 'api_key:...' has been suspended.",
                "status": "PERMISSION_DENIED",
            }
        },
    )


def test_gemini_backend_key_rotation_is_not_bounded_by_transport_retries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test: a pool with more dead keys in a row than
    max_transport_retries must still reach the working key at the end,
    instead of giving up early and reporting every key as unusable when most
    were never even tried. Five dead keys, max_transport_retries=1 (a budget
    that alone could not cover cycling past even two dead keys), one working
    key at the end."""
    _delenv_gemini_key_pool(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-1-dead")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-2-dead")
    monkeypatch.setenv("GEMINI_API_KEY_3", "key-3-dead")
    monkeypatch.setenv("GEMINI_API_KEY_4", "key-4-dead")
    monkeypatch.setenv("GEMINI_API_KEY_5", "key-5-dead")
    monkeypatch.setenv("GEMINI_API_KEY_6", "key-6-fresh")
    calls: list[dict] = []

    import google.genai as genai_sdk

    class _DeadModels:
        def generate_content(self, **request):
            raise _quota_exhausted_error()

    def fake_client(**kw):
        if kw.get("api_key") == "key-6-fresh":
            return SimpleNamespace(models=_FakeGeminiModels(calls))
        return SimpleNamespace(models=_DeadModels())

    monkeypatch.setattr(genai_sdk, "Client", fake_client)

    send_batch = get_send_batch(
        "gemini-3-pro",
        offline=False,
        backend="gemini",
        proxy_options={"concurrency": 1, "max_transport_retries": 1},
    )
    results = send_batch(["p1"])

    assert results == ["ACTION: SAFE  # gemini-3-pro"]
    assert len(calls) == 1


def test_gemini_backend_rotates_past_a_suspended_key(monkeypatch: pytest.MonkeyPatch) -> None:
    """A key can also be flat-out suspended (403 PERMISSION_DENIED /
    CONSUMER_SUSPENDED), not just quota-exhausted. That must rotate to the
    next key too, not abort the whole run -- this exact case broke the first
    version of the rotation logic, which only recognized 429
    RESOURCE_EXHAUSTED."""
    _delenv_gemini_key_pool(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-1-suspended")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-2-fresh")
    calls: list[dict] = []

    import google.genai as genai_sdk

    class _SuspendedModels:
        def generate_content(self, **request):
            raise _suspended_key_error()

    def fake_client(**kw):
        if kw.get("api_key") == "key-1-suspended":
            return SimpleNamespace(models=_SuspendedModels())
        return SimpleNamespace(models=_FakeGeminiModels(calls))

    monkeypatch.setattr(genai_sdk, "Client", fake_client)

    send_batch = get_send_batch(
        "gemini-3-pro",
        offline=False,
        backend="gemini",
        proxy_options={"concurrency": 1},
    )
    results = send_batch(["p1", "p2"])

    assert results == ["ACTION: SAFE  # gemini-3-pro", "ACTION: SAFE  # gemini-3-pro"]
    assert len(calls) == 2


def test_gemini_backend_survives_concurrent_quota_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression test for the same class of race fixed in bedrock_direct.py:
    with concurrency > 1, several worker threads can all hit the exhausted
    key together before any of them observes the rotation. A barrier forces
    exactly that pileup. Every thread must still retry against the next key
    and succeed."""
    _delenv_gemini_key_pool(monkeypatch)
    monkeypatch.setenv("GEMINI_API_KEY", "key-1-exhausted")
    monkeypatch.setenv("GEMINI_API_KEY_2", "key-2-fresh")
    calls: list[dict] = []
    first_wave = threading.Barrier(4, timeout=5)

    import google.genai as genai_sdk

    class _FailingModels:
        def generate_content(self, **request):
            first_wave.wait()  # force all 4 first attempts to land together
            raise _quota_exhausted_error()

    def fake_client(**kw):
        if kw.get("api_key") == "key-1-exhausted":
            return SimpleNamespace(models=_FailingModels())
        return SimpleNamespace(models=_FakeGeminiModels(calls))

    monkeypatch.setattr(genai_sdk, "Client", fake_client)

    send_batch = get_send_batch(
        "gemini-3-pro",
        offline=False,
        backend="gemini",
        proxy_options={"concurrency": 4},
    )
    results = send_batch(["p1", "p2", "p3", "p4"])

    assert results == ["ACTION: SAFE  # gemini-3-pro"] * 4
