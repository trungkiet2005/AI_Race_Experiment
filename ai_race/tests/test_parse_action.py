"""Strict parsing tests for the model's final action protocol."""
from __future__ import annotations

import pytest

from ai_race.engine.round import extract_reasoning, parse_action, response_text
from ai_race.engine.state import Action


@pytest.mark.parametrize(
    ("response", "expected"),
    [
        ("ACTION: SAFE", Action.SAFE),
        ({"text": "ACTION: UNSAFE"}, Action.UNSAFE),
    ],
)
def test_parse_action_accepts_only_a_standalone_formatted_line(
    response: str | dict[str, str],
    expected: Action,
) -> None:
    action, failed = parse_action(response)
    assert action is expected
    assert failed is False


@pytest.mark.parametrize(
    "response",
    [
        "",
        "SAFE",
        "I choose UNSAFE.",
        "prefix ACTION: SAFE",
        "ACTION = SAFE",
        "ACTION: S",
        "ACTION: UNSAFE!",
        "reasoning first\nACTION: UNSAFE",
        "ACTION: MAYBE",
        "ACTION: SAFE because it is prudent",
        {"text": ""},
        {},
    ],
)
def test_parse_action_falls_back_to_safe_on_protocol_failure(
    response: str | dict[str, str],
) -> None:
    action, failed = parse_action(response)
    assert action is Action.SAFE
    assert failed is True


def test_parse_action_rejects_more_than_one_action_line() -> None:
    action, failed = parse_action(
        "ACTION: SAFE\nI reconsidered after checking the state.\nACTION: UNSAFE"
    )
    assert action is Action.SAFE
    assert failed is True


def test_invalid_final_action_cannot_be_rescued_by_an_earlier_valid_line() -> None:
    """An invalid final protocol line must not silently reuse a draft decision."""
    action, failed = parse_action("ACTION: UNSAFE\nACTION: MAYBE")
    assert action is Action.SAFE
    assert failed is True


def test_action_must_be_the_final_nonempty_line() -> None:
    """The prompt promises a final action line, so trailing prose is a failure."""
    action, failed = parse_action("ACTION: UNSAFE\nThis is my final explanation.")
    assert action is Action.SAFE
    assert failed is True


def test_response_helpers_support_structured_backend_outputs() -> None:
    response = {
        "text": "Line one.\nACTION: SAFE\n",
        "token_logprobs": [-0.1, -0.2],
    }
    assert response_text(response) == response["text"]
    assert extract_reasoning(response) == "Line one."
