"""Tests for the backend dispatcher, focused on the direct-OpenAI branch.

The ``openai`` backend exists so a frontier GPT model can be named directly in
an experiment config's ``models`` list, the same way the ``proxy`` backend
takes a raw provider slug — without registering an abstract name in
FAIRGAME's vendored ``MODEL_PROVIDER_MAP`` for every model tried.
"""
from __future__ import annotations

import pytest

from ai_race.models.factory import get_send_batch

NOT_IN_MODEL_PROVIDER_MAP = "gpt-5-frontier-2027-01-01"


def test_unsupported_backend_raises() -> None:
    with pytest.raises(ValueError, match="Unsupported model backend"):
        get_send_batch("some-model", offline=False, backend="not-a-backend")


def test_openai_backend_requires_api_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("API_KEY_OPENAI", raising=False)
    with pytest.raises(EnvironmentError, match="API_KEY_OPENAI"):
        get_send_batch(NOT_IN_MODEL_PROVIDER_MAP, offline=False, backend="openai")


def test_openai_backend_accepts_a_raw_model_id_and_calls_send_prompt_per_prompt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, str, float]] = []

    class FakeConnector:
        def __init__(self, provider_model: str, temperature: float = 1.0) -> None:
            self.provider_model = provider_model
            self.temperature = temperature

        def send_prompt(self, prompt: str) -> str:
            calls.append((self.provider_model, prompt, self.temperature))
            return f"echo:{prompt}"

    import FAIRGAME.src.llm_connectors.openai_connector as openai_connector_module

    monkeypatch.setattr(openai_connector_module, "OpenAIConnector", FakeConnector)

    send_batch = get_send_batch(
        NOT_IN_MODEL_PROVIDER_MAP,
        offline=False,
        backend="openai",
        proxy_options={"temperature": 0.7},
    )
    # seeds are accepted (the runner passes them unconditionally to every
    # backend) but must be silently dropped, never faked into a request.
    results = send_batch(["p1", "p2"], seeds=[1, 2])

    assert results == ["echo:p1", "echo:p2"]
    assert calls == [
        (NOT_IN_MODEL_PROVIDER_MAP, "p1", 0.7),
        (NOT_IN_MODEL_PROVIDER_MAP, "p2", 0.7),
    ]


def test_openai_backend_defaults_temperature_to_0_7_not_the_connector_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, float] = {}

    class FakeConnector:
        def __init__(self, provider_model: str, temperature: float = 1.0) -> None:
            captured["temperature"] = temperature

        def send_prompt(self, prompt: str) -> str:
            return "ACTION: SAFE"

    import FAIRGAME.src.llm_connectors.openai_connector as openai_connector_module

    monkeypatch.setattr(openai_connector_module, "OpenAIConnector", FakeConnector)

    get_send_batch(NOT_IN_MODEL_PROVIDER_MAP, offline=False, backend="openai")

    assert captured["temperature"] == 0.7
