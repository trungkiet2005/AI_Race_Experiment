"""Paper-faithful prompt construction for one AI Race decision.

Templates follow the FAIRGAME ``resources/game_templates`` convention: camelCase
placeholders plus optional blocks written as ``{blockName}: [ ... ]`` that are
either unwrapped or deleted before ``str.format`` runs. Legacy snake_case
placeholders remain available as aliases so older templates keep rendering.
"""
from __future__ import annotations

import re
from typing import Any, Sequence

from .state import Action, GameConfig

_OPTIONAL_BLOCK_RE = r"\{{{name}\}}:\s*\[(.*?)\]"


def _fmt(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def apply_optional_blocks(template: str, blocks: dict[str, bool]) -> str:
    """Unwrap or delete FAIRGAME ``{name}: [ ... ]`` blocks.

    ``blocks[name] is True`` keeps the bracketed text, ``False`` removes the whole
    construct. Names absent from ``blocks`` are left untouched so an unexpected
    block surfaces as a formatting error rather than being silently dropped.
    """
    rendered = template
    for name, keep in blocks.items():
        pattern = re.compile(_OPTIONAL_BLOCK_RE.format(name=re.escape(name)), re.DOTALL)
        replacement = (lambda match: match.group(1)) if keep else (lambda match: "")
        rendered = pattern.sub(replacement, rendered)
    # Deleting a block leaves an empty line behind; collapse those, not real blanks.
    rendered = re.sub(r"[ \t]+\n", "\n", rendered)
    return re.sub(r"\n{3,}", "\n\n", rendered)


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
    # FAIRGAME game_template placeholder names. ``weight`` ordering follows the
    # FAIRGAME convention: 1 = both strategy1, 2 = the strategy2 chooser against
    # strategy1, 3 = the strategy1 chooser against strategy2, 4 = both strategy2.
    values.update(
        {
            "currentPlayerName": player_name,
            "opponent1": player_names[opponent_index],
            "personality": persona_text.strip(),
            "currentRound": round_number,
            "strategy1": Action.SAFE.label,
            "strategy2": Action.UNSAFE.label,
            "weight1": values["payoff_safe_safe"],
            "weight2": values["payoff_unsafe_safe"],
            "weight3": values["payoff_safe_unsafe"],
            "weight4": values["payoff_unsafe_unsafe"],
            "step1": values["safe_progress"],
            "step2": values["unsafe_progress"],
            "nRounds": config.min_rounds,
            "minRounds": config.min_rounds,
            "stopProbabilityPercent": values["stop_probability_percent"],
            "racePrize": values["race_prize"],
            "tiePrize": values["tie_prize"],
            "maxPrivateRiskPercent": values["max_private_risk_percent"],
            "ownPrivateRiskPercent": values["own_private_risk_percent"],
            "opponentPrivateRiskPercent": values["opponent_private_risk_percent"],
            "ownStagePayoff": values["own_stage_payoff"],
            "opponentStagePayoff": values["opponent_stage_payoff"],
            "ownProgress": values["own_progress"],
            "opponentProgress": values["opponent_progress"],
            "progressGap": values["progress_gap"],
            "history": values["previous_round"],
        }
    )
    prepared = apply_optional_blocks(
        template,
        {
            # The persona seat is empty in the paper-faithful baseline.
            "intro": bool(persona_text.strip()),
            # No opponent-personality prior is disclosed in this design.
            "opponentIntro": False,
            # The horizon is deliberately hidden from the agents.
            "gameLength": False,
            # Agents never exchange messages, so only the decision phase exists.
            "communicate": config.agents_communicate,
            "choose": True,
        },
    )
    try:
        rendered = prepared.format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown prompt placeholder: {exc.args[0]}") from exc
    return rendered.strip() + "\n"
