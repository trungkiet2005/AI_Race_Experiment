"""Paper-faithful prompt construction for one AI Race decision."""
from __future__ import annotations

from typing import Any, Sequence

from .state import Action, GameConfig


def _fmt(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _previous_round_text(
    history: Sequence[dict[str, Any]],
    player_index: int,
    player_names: Sequence[str],
    history_mode: str,
) -> str:
    if not history:
        return "No previous round has been played."

    selected = history if history_mode == "full" else history[-1:]
    lines: list[str] = []
    opponent_index = 1 - player_index
    for record in selected:
        own_action = Action.coerce(record["actions"][player_index]).label
        opponent_action = Action.coerce(record["actions"][opponent_index]).label
        own_payoff = record["payoffs"][player_index]
        opponent_payoff = record["payoffs"][opponent_index]
        own_step = record["increments"][player_index]
        opponent_step = record["increments"][opponent_index]
        lines.append(
            f"Round {record['round']}: you chose {own_action}; "
            f"{player_names[opponent_index]} chose {opponent_action}. "
            f"Round payoffs were {_fmt(own_payoff)} for you and "
            f"{_fmt(opponent_payoff)} for the opponent; progress increments were "
            f"{_fmt(own_step)} and {_fmt(opponent_step)}."
        )
    return "\n".join(lines)


def build_prompt(
    template: str,
    config: GameConfig,
    *,
    player_name: str,
    player_index: int,
    player_names: Sequence[str],
    persona_text: str,
    round_number: int,
    history: Sequence[dict[str, Any]],
    progress: Sequence[float],
    stage_payoffs: Sequence[float],
    unsafe_counts: Sequence[int],
) -> str:
    """Render the information available before a simultaneous decision."""
    if len(player_names) != 2:
        raise ValueError("AI Race prompts require exactly two player names")
    opponent_index = 1 - player_index
    completed_rounds = len(history)
    own_current_risk = (
        config.max_private_risk * unsafe_counts[player_index] / completed_rounds
        if completed_rounds
        else 0.0
    )
    opponent_current_risk = (
        config.max_private_risk * unsafe_counts[opponent_index] / completed_rounds
        if completed_rounds
        else 0.0
    )
    persona_block = (
        f"\nAdditional role instruction:\n{persona_text.strip()}\n"
        if persona_text.strip()
        else ""
    )
    values = {
        "player_name": player_name,
        "opponent_name": player_names[opponent_index],
        "persona_block": persona_block,
        "round_number": round_number,
        "max_private_risk_percent": _fmt(config.max_private_risk * 100),
        "safe_progress": _fmt(config.safe_progress),
        "unsafe_progress": _fmt(config.unsafe_progress),
        "payoff_safe_safe": _fmt(config.payoff_safe_safe),
        "payoff_safe_unsafe": _fmt(config.payoff_safe_unsafe),
        "payoff_unsafe_safe": _fmt(config.payoff_unsafe_safe),
        "payoff_unsafe_unsafe": _fmt(config.payoff_unsafe_unsafe),
        "min_rounds": config.min_rounds,
        "stop_probability_percent": _fmt(config.stop_probability * 100),
        "race_prize": _fmt(config.race_prize),
        "tie_prize": _fmt(config.race_prize / 2.0),
        "own_stage_payoff": _fmt(stage_payoffs[player_index]),
        "opponent_stage_payoff": _fmt(stage_payoffs[opponent_index]),
        "own_private_risk_percent": _fmt(own_current_risk * 100),
        "opponent_private_risk_percent": _fmt(opponent_current_risk * 100),
        "own_progress": _fmt(progress[player_index]),
        "opponent_progress": _fmt(progress[opponent_index]),
        "progress_gap": _fmt(progress[player_index] - progress[opponent_index]),
        "previous_round": _previous_round_text(
            history,
            player_index,
            player_names,
            config.history_mode,
        ),
    }
    try:
        rendered = template.format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown prompt placeholder: {exc.args[0]}") from exc
    return rendered.strip() + "\n"
