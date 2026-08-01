"""Prompt construction for one N-player AI Race decision.

Generalises ``ai_race.engine.prompt.build_prompt`` from a single named
opponent to a list of co-players. ``str.format`` has no loop construct, so
the "other companies" state, the per-round payoff reference table, and the
round history are all pre-joined into multi-line blocks in Python and handed
to the template as single placeholders -- the same strategy
``FAIRGAME/src/prompt_creator.py`` uses for its own variable-length
``opponents`` list, just collapsed to one block per section instead of one
placeholder per opponent (arbitrary N).

Reuses :func:`ai_race.engine.prompt.apply_optional_blocks` unchanged -- it
operates on template text and a blocks dict, with no player-count assumption.
"""
from __future__ import annotations

from typing import Any, Sequence

from ai_race.engine.prompt import apply_optional_blocks

from .state import Action, NPlayerGameConfig


def _fmt(value: float) -> str:
    return f"{float(value):.3f}".rstrip("0").rstrip(".")


def _join_names(names: Sequence[str]) -> str:
    names = list(names)
    if len(names) == 1:
        return names[0]
    return ", ".join(names[:-1]) + f", and {names[-1]}"


def _payoff_table(config: NPlayerGameConfig) -> str:
    """One line per possible SAFE-count ``k`` this round, fully spelled out."""
    n = config.n_players
    lines: list[str] = []
    for k in range(0, n + 1):
        if k == 0:
            lines.append(
                f"- If 0 of {n} companies choose SAFE (everyone UNSAFE): each "
                f"UNSAFE company earns {_fmt(config.stage_payoff_unsafe(0))}."
            )
        elif k == n:
            lines.append(
                f"- If all {n} companies choose SAFE: each company earns "
                f"{_fmt(config.stage_payoff_safe(n))}."
            )
        else:
            lines.append(
                f"- If {k} of {n} companies choose SAFE: each SAFE company earns "
                f"{_fmt(config.stage_payoff_safe(k))}; each UNSAFE company earns "
                f"{_fmt(config.stage_payoff_unsafe(k))}."
            )
    return "\n".join(lines)


def _others_state(
    others_names: Sequence[str],
    others_progress: Sequence[float],
    others_stage_payoff: Sequence[float],
    others_risk_percent: Sequence[float],
) -> str:
    lines = [
        f"- {name}: accumulated round payoff {_fmt(payoff)}, current private "
        f"setback risk {_fmt(risk)}%, cumulative progress {_fmt(progress)}"
        for name, payoff, risk, progress in zip(
            others_names, others_stage_payoff, others_risk_percent, others_progress
        )
    ]
    return "\n".join(lines)


def _previous_round_text(
    history: Sequence[dict[str, Any]],
    player_index: int,
    player_names: Sequence[str],
    history_mode: str,
) -> str:
    if not history:
        return "No previous round has been played."

    if history_mode == "full":
        selected = list(history)
    elif history_mode == "first_and_previous":
        selected = [history[0]] if len(history) == 1 else [history[0], history[-1]]
    else:
        selected = history[-1:]

    other_indices = [i for i in range(len(player_names)) if i != player_index]
    lines: list[str] = []
    for record in selected:
        own_action = Action.coerce(record["actions"][player_index]).label
        own_payoff = record["payoffs"][player_index]
        own_step = record["increments"][player_index]
        others_text = "; ".join(
            f"{player_names[i]} chose {Action.coerce(record['actions'][i]).label} "
            f"(payoff {_fmt(record['payoffs'][i])}, step {_fmt(record['increments'][i])})"
            for i in other_indices
        )
        lines.append(
            f"Round {record['round']}: you chose {own_action} (payoff "
            f"{_fmt(own_payoff)}, step {_fmt(own_step)}). {others_text}."
        )
    return "\n".join(lines)


def build_prompt(
    template: str,
    config: NPlayerGameConfig,
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
    """Render the information available before one simultaneous N-player decision."""
    n = config.n_players
    if len(player_names) != n:
        raise ValueError(f"AI Race prompts require exactly {n} player names")
    other_indices = [i for i in range(n) if i != player_index]
    completed_rounds = len(history)

    def current_risk(index: int) -> float:
        if not completed_rounds:
            return 0.0
        return config.max_private_risk * unsafe_counts[index] / completed_rounds

    others_names = [player_names[i] for i in other_indices]
    others_progress = [progress[i] for i in other_indices]
    others_stage_payoff = [stage_payoffs[i] for i in other_indices]
    others_risk_percent = [current_risk(i) * 100 for i in other_indices]

    values = {
        "currentPlayerName": player_name,
        "nPlayers": n,
        "othersNamesList": _join_names(others_names),
        "personality": persona_text.strip(),
        "strategy1": Action.SAFE.label,
        "strategy2": Action.UNSAFE.label,
        "step1": _fmt(config.safe_progress),
        "step2": _fmt(config.unsafe_progress()),
        "minRounds": config.min_rounds,
        "stopProbabilityPercent": _fmt(config.stop_probability * 100),
        "currentRound": round_number,
        "payoffTable": _payoff_table(config),
        "racePrize": _fmt(config.race_prize),
        "maxPrivateRiskPercent": _fmt(config.max_private_risk * 100),
        "ownStagePayoff": _fmt(stage_payoffs[player_index]),
        "ownPrivateRiskPercent": _fmt(current_risk(player_index) * 100),
        "ownProgress": _fmt(progress[player_index]),
        "progressGapToLeader": _fmt(progress[player_index] - max(others_progress)),
        "othersState": _others_state(
            others_names, others_progress, others_stage_payoff, others_risk_percent
        ),
        "history": _previous_round_text(
            history, player_index, player_names, config.history_mode
        ),
    }
    prepared = apply_optional_blocks(
        template,
        {
            "intro": bool(persona_text.strip()),
            "choose": True,
        },
    )
    try:
        rendered = prepared.format(**values)
    except KeyError as exc:
        raise ValueError(f"Unknown prompt placeholder: {exc.args[0]}") from exc
    return rendered.strip() + "\n"
