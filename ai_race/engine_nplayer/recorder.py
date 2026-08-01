"""Write self-describing per-turn, per-race, and per-player outputs for N-player races.

Long format only: every table has one row per player (or one row per race for
race-level summaries), with co-player information carried as a JSON-encoded
list column rather than fixed ``player_1_*``/``player_2_*`` columns. That is
what lets the schema generalise to any ``n_players`` without inventing a new
column per seat -- ``ai_race.dataio.recorder``'s wide ``races.csv``/
``all_results.csv`` cannot do that, which is why they are not reused here.

``append_turns_jsonl``/``write_turns_jsonl`` from ``ai_race.dataio.recorder``
*are* reused unchanged below: they only ever call ``record.to_dict()`` and
serialise the result, with no player-count assumption.
"""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

from ai_race.dataio.recorder import append_turns_jsonl, write_turns_jsonl

__all__ = [
    "append_turns_jsonl",
    "write_turns_jsonl",
    "race_row",
    "player_rows",
    "write_races_csv",
    "write_players_csv",
    "NPlayerRunJournal",
]


def race_row(result: Any) -> dict[str, Any]:
    return {
        "game_id": result.game_id,
        "model": result.model,
        "n_players": result.n_players,
        "max_private_risk": result.max_private_risk,
        "prompt_version": (result.config or {}).get("prompt_version", ""),
        "run_phase": result.run_phase,
        "persona_condition": result.persona_condition,
        "persona_roles": json.dumps(list(result.persona_roles), separators=(",", ":")),
        "rep": result.rep,
        "game_seed": result.game_seed,
        "n_rounds": result.n_rounds,
        "stop_forced": int(bool(result.stop_forced)),
        "is_full_tie": int(bool(result.is_full_tie)),
        "winners": json.dumps(list(result.winners), separators=(",", ":")),
        "parse_failures": result.parse_failures,
        "stop_draws": json.dumps(result.stop_draws, separators=(",", ":")),
    }


def player_rows(result: Any) -> list[dict[str, Any]]:
    names = list(result.player_names)
    rows: list[dict[str, Any]] = []
    for player_index, player in enumerate(names):
        others = [name for i, name in enumerate(names) if i != player_index]
        rows.append(
            {
                "game_id": result.game_id,
                "model": result.model,
                "n_players": result.n_players,
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
                "others": json.dumps(others, separators=(",", ":")),
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


class NPlayerRunJournal:
    """Durably append one completed round/race at a time.

    Same append-per-round contract as ``ai_race.dataio.recorder.RunJournal``:
    a process interruption can still lose an in-flight model request, but
    every round already applied to game state is flushed to disk immediately.
    Writes three files instead of four -- no wide ``all_results.csv``, see
    the module docstring.
    """

    _FILES = ("turns.jsonl", "races.csv", "players.csv")

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
        self.player_count += _append_csv(
            player_rows(result),
            self.output_dir / "players.csv",
        )
