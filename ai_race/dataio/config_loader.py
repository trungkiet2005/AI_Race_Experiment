"""Load and lightly validate AI Race JSON configuration."""
from __future__ import annotations

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
