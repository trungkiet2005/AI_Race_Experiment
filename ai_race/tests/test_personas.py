"""Enforce the rules that make a persona a manipulation rather than an instruction.

A persona that names an action ("choose UNSAFE more often") measures compliance,
not strategic behaviour, and a persona that restates the payoff matrix adds
strategic information the neutral baseline never had — either one silently
changes what the comparison means. A persona that is simply *longer* than the
others confounds content with prompt length. These tests keep all three failure
modes out of the shipped configurations, and check that the condition labels
downstream analysis relies on are internally consistent.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from ai_race.dataio.config_loader import (
    ConfigError,
    personas_sha256,
    validate_agents,
)
from ai_race.engine.agent import RaceAgent
from ai_race.engine.prompt import build_prompt
from ai_race.engine.state import GameConfig
from ai_race.paths import CONFIGS_DIR, PROMPTS_DIR

AGENTS_DIR = CONFIGS_DIR / "agents"
EXPERIMENT_DIR = CONFIGS_DIR / "experiment"

# Whole words only: "secured" must not trip the SAFE check.
ACTION_WORDS = re.compile(r"\b(safe|unsafe)\b", re.IGNORECASE)
# Any digit is treated as a restated payoff, risk, step, or horizon number.
NUMERIC = re.compile(r"\d")
LENGTH_TOLERANCE = 0.10


def _agents_configs() -> list[tuple[str, dict]]:
    return [
        (path.stem, json.loads(path.read_text(encoding="utf-8")))
        for path in sorted(AGENTS_DIR.glob("*.json"))
    ]


def _persona_texts() -> list[tuple[str, str, str]]:
    """Yield (config, language, text) for every non-empty persona."""

    found: list[tuple[str, str, str]] = []
    for name, payload in _agents_configs():
        for language, personas in (payload.get("personas") or {}).items():
            for text in personas:
                if str(text).strip():
                    found.append((name, language, str(text)))
    return found


def test_repository_ships_persona_conditions_to_compare():
    conditions = {
        str(payload.get("personaCondition", "none"))
        for _, payload in _agents_configs()
    }
    assert "none" in conditions, "the neutral baseline seat must remain available"
    assert len(conditions) > 1, "no persona condition is configured to compare"


def test_every_shipped_agents_config_validates():
    for name, payload in _agents_configs():
        validate_agents(payload)
        assert payload.get("name") == name, (
            f"{name}.json declares name={payload.get('name')!r}; the runner resolves "
            "agents by filename, so a mismatch mislabels the run"
        )


def test_persona_text_never_names_an_action():
    offenders = [
        (name, language, ACTION_WORDS.findall(text))
        for name, language, text in _persona_texts()
        if ACTION_WORDS.search(text)
    ]
    assert not offenders, (
        "persona text must not name Safe or Unsafe; naming an action turns the "
        f"manipulation into an instruction: {offenders}"
    )


def test_persona_text_never_restates_the_mechanism():
    offenders = [
        (name, language, text)
        for name, language, text in _persona_texts()
        if NUMERIC.search(text)
    ]
    assert not offenders, (
        "persona text must not contain numbers: repeating a payoff, risk, step, or "
        f"horizon adds strategic information the baseline lacks: {offenders}"
    )


@pytest.mark.parametrize("language", ["en", "vi"])
def test_persona_lengths_are_balanced_within_a_language(language: str):
    lengths = {
        f"{name}:{index}": len(str(text).split())
        for name, payload in _agents_configs()
        for index, text in enumerate((payload.get("personas") or {}).get(language, []))
        if str(text).strip()
    }
    if not lengths:
        pytest.skip(f"no {language} personas are configured")
    mean_length = sum(lengths.values()) / len(lengths)
    outliers = {
        key: length
        for key, length in lengths.items()
        if abs(length - mean_length) > LENGTH_TOLERANCE * mean_length
    }
    assert not outliers, (
        f"{language} persona lengths must stay within "
        f"{LENGTH_TOLERANCE:.0%} of the mean ({mean_length:.1f} words), otherwise "
        f"content is confounded with prompt length: {outliers}"
    )


def test_persona_conditions_cover_both_seat_orders():
    """Every asymmetric cell needs its mirror, or persona confounds with seat."""

    by_condition = {
        str(payload.get("personaCondition")): tuple(payload.get("personaRoles", []))
        for _, payload in _agents_configs()
    }
    asymmetric = {
        condition: roles
        for condition, roles in by_condition.items()
        if len(roles) == 2 and roles[0] != roles[1]
    }
    for condition, roles in asymmetric.items():
        mirrored = tuple(reversed(roles))
        assert mirrored in by_condition.values(), (
            f"condition {condition!r} assigns {roles} but no configuration assigns "
            f"{mirrored}; without the mirror the persona effect cannot be "
            "separated from a seat-position effect"
        )


def test_persona_hash_changes_with_persona_text():
    baseline = {"en": ["", ""]}
    edited = {"en": ["", "a single new word"]}
    assert personas_sha256(baseline) != personas_sha256(edited)
    assert personas_sha256(baseline) == personas_sha256({"en": ["", ""]})


def test_condition_label_must_agree_with_persona_text():
    labelled_but_empty = {
        "name": "broken",
        "names": ["A", "B"],
        "personaCondition": "S_AA",
        "personaRoles": ["adversarial", "adversarial"],
        "personas": {"en": ["", ""]},
    }
    with pytest.raises(ConfigError, match="contradicts the persona texts"):
        validate_agents(labelled_but_empty)

    text_but_unlabelled = {
        "name": "broken",
        "names": ["A", "B"],
        "personaCondition": "none",
        "personaRoles": ["", ""],
        "personas": {"en": ["you are cautious", ""]},
    }
    with pytest.raises(ConfigError, match="contradicts the persona texts"):
        validate_agents(text_but_unlabelled)


def test_asymmetric_condition_must_name_both_seat_roles():
    missing_role = {
        "name": "broken",
        "names": ["A", "B"],
        "personaCondition": "S_AC",
        "personaRoles": ["adversarial", ""],
        "personas": {"en": ["you are one way", "you are another way"]},
    }
    with pytest.raises(ConfigError, match="role of both seats"):
        validate_agents(missing_role)


def _render(persona_text: str) -> str:
    config = GameConfig(name="ai_race_risk_60", max_private_risk=0.6)
    template = (PROMPTS_DIR / "ai_race_en.txt").read_text(encoding="utf-8")
    return build_prompt(
        template,
        config,
        player_name="Company_1",
        player_index=0,
        player_names=["Company_1", "Company_2"],
        persona_text=persona_text,
        round_number=1,
        history=[],
        progress=[0.0, 0.0],
        stage_payoffs=[0.0, 0.0],
        unsafe_counts=[0, 0],
    )


def test_persona_reaches_the_rendered_prompt_and_neutral_stays_neutral():
    persona = "an executive who is uneasy about anything left to chance"
    rendered = _render(persona)
    assert persona in rendered
    assert "You are an executive who is uneasy" in rendered

    neutral = _render("")
    assert "You are Company_1" in neutral
    assert "[" not in neutral, (
        "an unfilled optional block leaked its bracket markup into the prompt"
    )


def test_opponent_persona_is_never_disclosed():
    """Each seat sees only its own persona.

    If the opponent's persona leaked, an asymmetric cell would become a
    perfect-information game and could no longer be compared with the human task.
    """

    rendered = _render("an executive who is uneasy about anything left to chance")
    assert "probability of being" not in rendered
    assert rendered.count("uneasy") == 1


def test_persona_experiments_share_the_baseline_seed_and_treatments():
    """Matched repetitions must reuse one horizon draw across persona cells."""

    baseline = json.loads(
        (EXPERIMENT_DIR / "baseline.json").read_text(encoding="utf-8")
    )
    persona_experiments = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(EXPERIMENT_DIR.glob("persona_baseline_*.json"))
    ]
    assert persona_experiments, "no persona experiment configuration was found"
    for experiment in persona_experiments:
        assert experiment["seed"] == baseline["seed"], (
            f"{experiment['name']} uses a different base seed, which breaks the "
            "common-random-number match with the neutral baseline"
        )
        assert experiment["games"] == baseline["games"]
        assert experiment["repetitions"] == baseline["repetitions"]
