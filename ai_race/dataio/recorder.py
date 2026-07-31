"""Write self-describing per-turn, per-race, and per-player outputs."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable


def _as_dict(record: Any) -> dict[str, Any]:
    if hasattr(record, "to_dict"):
        return record.to_dict()
    if isinstance(record, dict):
        return dict(record)
    raise TypeError(f"Unsupported record type: {type(record).__name__}")


def write_turns_jsonl(turns: Iterable[Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(json.dumps(_as_dict(turn), ensure_ascii=False) + "\n")


def append_turns_jsonl(turns: Iterable[Any], path: str | Path) -> int:
    """Append decision records and return the number written."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("a", encoding="utf-8") as handle:
        for turn in turns:
            handle.write(json.dumps(_as_dict(turn), ensure_ascii=False) + "\n")
            count += 1
        handle.flush()
    return count


def race_row(result: Any) -> dict[str, Any]:
    names = list(result.player_names)
    if len(names) != 2:
        raise ValueError("Race output requires exactly two player names")
    return {
        "game_id": result.game_id,
        "model": result.model,
        "max_private_risk": result.max_private_risk,
        "prompt_version": (result.config or {}).get("prompt_version", ""),
        "run_phase": result.run_phase,
        "persona_condition": result.persona_condition,
        "player_1_persona_role": result.persona_roles[0],
        "player_2_persona_role": result.persona_roles[1],
        "rep": result.rep,
        "game_seed": result.game_seed,
        "n_rounds": result.n_rounds,
        "stop_forced": int(bool(result.stop_forced)),
        "tie": int(bool(result.tie)),
        "winner": result.winner or "",
        "player_1": names[0],
        "player_2": names[1],
        "player_1_progress": result.progress[0],
        "player_2_progress": result.progress[1],
        "player_1_stage_payoff": result.stage_payoffs[0],
        "player_2_stage_payoff": result.stage_payoffs[1],
        "player_1_unsafe_count": result.unsafe_counts[0],
        "player_2_unsafe_count": result.unsafe_counts[1],
        "player_1_unsafe_frequency": result.unsafe_frequencies[0],
        "player_2_unsafe_frequency": result.unsafe_frequencies[1],
        "player_1_private_risk": result.private_risks[0],
        "player_2_private_risk": result.private_risks[1],
        "player_1_prize": result.prizes[0],
        "player_2_prize": result.prizes[1],
        "player_1_setback": int(bool(result.setbacks[0])),
        "player_2_setback": int(bool(result.setbacks[1])),
        "player_1_final_payoff": result.final_payoffs[0],
        "player_2_final_payoff": result.final_payoffs[1],
        "parse_failures": result.parse_failures,
        "stop_draws": json.dumps(result.stop_draws, separators=(",", ":")),
    }


def player_rows(result: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for player_index, player in enumerate(result.player_names):
        rows.append(
            {
                "game_id": result.game_id,
                "model": result.model,
                "max_private_risk": result.max_private_risk,
                "prompt_version": (result.config or {}).get("prompt_version", ""),
                "run_phase": result.run_phase,
                "persona_condition": result.persona_condition,
                "persona_role": result.persona_roles[player_index],
                "rep": result.rep,
                "game_seed": result.game_seed,
                "n_rounds": result.n_rounds,
                "player": player,
                "player_index": player_index,
                "opponent": result.player_names[1 - player_index],
                "outcome": result.outcomes[player_index],
                "progress": result.progress[player_index],
                "stage_payoff": result.stage_payoffs[player_index],
                "unsafe_count": result.unsafe_counts[player_index],
                "unsafe_frequency": result.unsafe_frequencies[player_index],
                "private_risk": result.private_risks[player_index],
                "prize": result.prizes[player_index],
                "setback_eligible": int(bool(result.setback_eligible[player_index])),
                "setback_draw": result.setback_draws[player_index],
                "setback": int(bool(result.setbacks[player_index])),
                "final_payoff": result.final_payoffs[player_index],
            }
        )
    return rows


def all_results_row(result: Any) -> dict[str, Any]:
    """One row per race: per-round action/payoff sequences plus terminal outcomes.

    Modeled on FAIRGAME's ``all_results.csv`` (one row per game, per-round
    strategies/scores as list columns) but carries this project's own fields
    (progress, private risk, prize, setback) instead of FAIRGAME's payoff-matrix
    columns. Excludes prompts/raw responses on purpose — this file is for
    skimming outcomes, not for prompt auditing (see turns.jsonl for that).
    """
    names = list(result.player_names)
    if len(names) != 2:
        raise ValueError("Race output requires exactly two player names")
    cfg = result.config or {}
    row: dict[str, Any] = {
        "game_id": result.game_id,
        "model": result.model,
        "language": cfg.get("language", ""),
        "run_phase": result.run_phase,
        # Carried here too: a reader skimming outcomes must be able to tell a
        # persona cell from the neutral baseline, and persona leaves no trace in
        # the prompt hash.
        "persona_condition": result.persona_condition,
        "rep": result.rep,
        "game_seed": result.game_seed,
        "n_rounds": result.n_rounds,
        "max_private_risk": result.max_private_risk,
        "agents_communicate": int(bool(cfg.get("agents_communicate", False))),
        "stop_forced": int(bool(result.stop_forced)),
        "tie": int(bool(result.tie)),
        "winner": result.winner or "",
    }
    for player_index, name in enumerate(names):
        prefix = f"player{player_index + 1}"
        actions = [
            str(round_actions[player_index]).upper()
            for round_actions in result.per_round_actions
        ]
        payoffs = [
            round_record[player_index] for round_record in result.per_round_payoffs
        ]
        row[f"{prefix}_name"] = name
        row[f"{prefix}_persona_role"] = result.persona_roles[player_index]
        row[f"{prefix}_actions"] = json.dumps(actions)
        row[f"{prefix}_round_payoffs"] = json.dumps(payoffs)
        row[f"{prefix}_progress"] = result.progress[player_index]
        row[f"{prefix}_stage_payoff"] = result.stage_payoffs[player_index]
        row[f"{prefix}_unsafe_count"] = result.unsafe_counts[player_index]
        row[f"{prefix}_unsafe_frequency"] = result.unsafe_frequencies[player_index]
        row[f"{prefix}_private_risk"] = result.private_risks[player_index]
        row[f"{prefix}_prize"] = result.prizes[player_index]
        row[f"{prefix}_setback"] = int(bool(result.setbacks[player_index]))
        row[f"{prefix}_final_payoff"] = result.final_payoffs[player_index]
    return row


def write_all_results_csv(results: Iterable[Any], path: str | Path) -> None:
    _write_csv([all_results_row(result) for result in results], path)


def append_all_results_csv(results: Iterable[Any], path: str | Path) -> int:
    return _append_csv([all_results_row(result) for result in results], path)


def _write_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_csv(rows: list[dict[str, Any]], path: str | Path) -> int:
    if not rows:
        return 0
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        if write_header:
            writer.writeheader()
        writer.writerows(rows)
        handle.flush()
    return len(rows)


def write_races_csv(results: Iterable[Any], path: str | Path) -> None:
    _write_csv([race_row(result) for result in results], path)


def write_players_csv(results: Iterable[Any], path: str | Path) -> None:
    rows: list[dict[str, Any]] = []
    for result in results:
        rows.extend(player_rows(result))
    _write_csv(rows, path)


class RunJournal:
    """Durably append one completed round/race at a time.

    A process interruption can still lose an in-flight model request, but every
    round already applied to game state is flushed to disk immediately. Incomplete
    races remain visible in ``turns.jsonl`` and have no matching terminal CSV row.
    """

    _FILES = ("turns.jsonl", "races.csv", "players.csv", "all_results.csv")

    def __init__(self, output_dir: str | Path, *, reset: bool = False):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        if reset:
            for filename in self._FILES:
                path = self.output_dir / filename
                if path.exists():
                    path.unlink()
        self.turn_count = 0
        self.race_count = 0
        self.player_count = 0

    def record_round(
        self,
        game: Any,
        result: Any | None,
        new_turns: Iterable[Any],
    ) -> None:
        del game
        self.turn_count += append_turns_jsonl(
            new_turns,
            self.output_dir / "turns.jsonl",
        )
        if result is None:
            return
        self.race_count += _append_csv(
            [race_row(result)],
            self.output_dir / "races.csv",
        )
        _append_csv(
            [all_results_row(result)],
            self.output_dir / "all_results.csv",
        )
        self.player_count += _append_csv(
            player_rows(result),
            self.output_dir / "players.csv",
        )
