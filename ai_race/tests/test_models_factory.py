"""Tests for the backend dispatcher, focused on the direct-OpenAI branch.

The ``openai`` backend exists so a frontier GPT model can be named directly in
an experiment config's ``models`` list, the same way the ``proxy`` backend
takes a raw provider slug — without registering an abstract name in
FAIRGAME's vendored ``MODEL_PROVIDER_MAP`` for every model tried, and with a
bounded ``max_tokens`` (FAIRGAME's ``OpenAIConnector`` has none).
"""
from __future__ import annotations

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
