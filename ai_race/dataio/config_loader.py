"""Load and lightly validate AI Race JSON configuration."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Optional

from ai_race.engine.state import GameConfig


class ConfigError(ValueError):
    pass


def load_json(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    try:
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except FileNotFoundError as exc:
        raise ConfigError(f"Configuration file not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ConfigError(f"Top-level JSON value must be an object: {path}")
    return data


def validate_game(data: dict[str, Any]) -> dict[str, Any]:
    required = {
        "name",
        "nPlayers",
        "safeProgress",
        "unsafeProgress",
        "stagePayoffs",
        "minRounds",
        "stopProbability",
        "racePrize",
        "maxPrivateRisk",
    }
    missing = sorted(required.difference(data))
    if missing:
        raise ConfigError(f"Game configuration is missing keys: {missing}")
    payoffs = data.get("stagePayoffs")
    if not isinstance(payoffs, dict):
        raise ConfigError("stagePayoffs must be an object")
    payoff_keys = {"safeSafe", "safeUnsafe", "unsafeSafe", "unsafeUnsafe"}
    missing_payoffs = sorted(payoff_keys.difference(payoffs))
    if missing_payoffs:
        raise ConfigError(f"stagePayoffs is missing keys: {missing_payoffs}")
    try:
        GameConfig.from_dict(data)
    except (TypeError, ValueError, KeyError) as exc:
        raise ConfigError(str(exc)) from exc
    return data


def load_game_config(
    path: str | Path,
    *,
    language: Optional[str] = None,
    model: Optional[str] = None,
) -> GameConfig:
    data = validate_game(load_json(path))
    return GameConfig.from_dict(data, language=language, model=model)


def personas_sha256(personas_by_language: dict[str, Any]) -> str:
    """Hash every persona text so a silent edit cannot masquerade as the same run."""

    canonical = json.dumps(
        personas_by_language,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def validate_agents(data: dict[str, Any]) -> dict[str, Any]:
    """Validate a seat/persona configuration and its condition label.

    The label is what keeps persona and non-persona races apart downstream, so an
    unlabelled or mislabelled persona is rejected here rather than discovered
    after a run has already been paid for.
    """

    names = data.get("names")
    if not isinstance(names, list) or len(names) != 2:
        raise ConfigError("Agent configuration must define exactly two names")
    if len(set(map(str, names))) != 2:
        raise ConfigError("Agent names must be distinct so seats stay identifiable")

    personas_by_language = data.get("personas", {}) or {}
    if not isinstance(personas_by_language, dict):
        raise ConfigError("Agent personas must be an object keyed by language")

    any_persona_text = False
    for language, personas in personas_by_language.items():
        if not isinstance(personas, list) or len(personas) != 2:
            raise ConfigError(
                f"Agent configuration must define two {language!r} personas"
            )
        any_persona_text |= any(str(persona).strip() for persona in personas)

    condition = str(data.get("personaCondition", "none")).strip()
    if not condition:
        raise ConfigError(
            "personaCondition must be a non-empty label; use 'none' for the "
            "neutral baseline"
        )
    if (condition == "none") is not (not any_persona_text):
        raise ConfigError(
            f"personaCondition={condition!r} contradicts the persona texts: "
            "'none' requires every persona to be empty, and any non-empty persona "
            "requires a condition label other than 'none'"
        )

    roles = data.get("personaRoles", ["", ""])
    if not isinstance(roles, list) or len(roles) != 2:
        raise ConfigError("personaRoles must be a two-element list, one per seat")
    if condition == "none" and any(str(role).strip() for role in roles):
        raise ConfigError("personaCondition='none' cannot declare persona roles")
    if condition != "none" and not all(str(role).strip() for role in roles):
        raise ConfigError(
            "a persona condition must name the role of both seats so asymmetric "
            "cells can be split by seat"
        )
    return data


def validate_experiment(data: dict[str, Any]) -> dict[str, Any]:
    required = {"name", "games", "models", "repetitions", "seed"}
    missing = sorted(required.difference(data))
    if missing:
        raise ConfigError(f"Experiment configuration is missing keys: {missing}")
    if not isinstance(data["games"], list) or not data["games"]:
        raise ConfigError("Experiment games must be a non-empty list")
    if not isinstance(data["models"], list) or not data["models"]:
        raise ConfigError("Experiment models must be a non-empty list")
    if int(data["repetitions"]) < 1:
        raise ConfigError("Experiment repetitions must be positive")
    if str(data.get("runPhase", "pilot")) not in {"pilot", "confirmatory"}:
        raise ConfigError("Experiment runPhase must be 'pilot' or 'confirmatory'")
    return data
