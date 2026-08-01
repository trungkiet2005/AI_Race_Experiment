"""Print every round of one recorded N-player race from turns.jsonl.

Reads back what was actually sent/received (no model call), same purpose as
the ``dump_race_prompts``/``list_race_ids`` debug cell in
``kaggle/experiments/baseline.py``, adapted for this module's ``turns.jsonl``
schema: ``others``/``others_prev_actions`` (a list of co-players) instead of
a singular ``opponent``/``opponent_prev_action``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional


def list_race_ids(
    output_dir: str | Path, limit: Optional[int] = None
) -> list[tuple[str, tuple[Any, ...]]]:
    """List game_id present under output_dir, with model and risk level."""
    output_dir = Path(output_dir)
    seen: dict[str, tuple[Any, ...]] = {}
    for path in sorted(output_dir.rglob("turns.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            seen.setdefault(
                row["game_id"],
                (row.get("model"), row.get("max_private_risk"), row.get("rep"), path),
            )
    items = list(seen.items())
    return items[:limit] if limit else items


def _load_race_turns(
    game_id: Optional[str], output_dir: str | Path
) -> tuple[Optional[str], list[dict[str, Any]]]:
    output_dir = Path(output_dir)
    rows: list[dict[str, Any]] = []
    for path in sorted(output_dir.rglob("turns.jsonl")):
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            if game_id is None or row["game_id"] == game_id:
                rows.append(row)
        if rows and game_id is None:
            game_id = rows[0]["game_id"]
            rows = [r for r in rows if r["game_id"] == game_id]
            break
        if rows:
            break
    rows.sort(key=lambda r: (int(r["round"]), int(r.get("player_index", 0))))
    return game_id, rows


def dump_race_prompts(
    game_id: Optional[str] = None,
    output_dir: str | Path = ".",
    *,
    show_response: bool = True,
    max_prompt_chars: Optional[int] = None,
) -> None:
    """Print every round of one race: prompt sent, raw response, parsed action."""
    game_id, rows = _load_race_turns(game_id, output_dir)
    if not rows:
        print(f"[debug] no turns found for game_id={game_id!r}")
        return

    head = rows[0]
    n_players = int(head.get("n_players", 0))
    print("=" * 78)
    print(f"RACE   {game_id}")
    print(
        f"model  {head.get('model')}   n_players={n_players}   "
        f"p_max={head.get('max_private_risk')}   rep={head.get('rep')}   "
        f"seed={head.get('game_seed')}"
    )
    print(
        f"prompt_version={head.get('prompt_version')}  "
        f"persona={head.get('persona_condition')}  "
        f"rounds={max(int(r['round']) for r in rows)}"
    )
    print("=" * 78)

    for round_number in sorted({int(r["round"]) for r in rows}):
        seats = [r for r in rows if int(r["round"]) == round_number]
        print(f"\n{'-' * 78}\nROUND {round_number}\n{'-' * 78}")
        for seat in seats:
            label = f"[{seat.get('player')} idx={seat.get('player_index')}]"
            print(f"\n{label}  others={seat.get('others')}")
            prompt = seat.get("prompt") or ""
            shown = prompt if max_prompt_chars is None else prompt[:max_prompt_chars]
            print(shown)
            if max_prompt_chars is not None and len(prompt) > max_prompt_chars:
                print(f"... (truncated {len(prompt) - max_prompt_chars} chars)")
            if show_response:
                print(f"  -> raw     : {seat.get('raw_response')!r}")
                print(
                    f"  -> action  : {seat.get('action')}   "
                    f"parse_failed={seat.get('parse_failed')}   "
                    f"retry={seat.get('retry_count')}"
                )

    print(f"\n{'=' * 78}\nTRAJECTORIES")
    players = list(dict.fromkeys(r.get("player") for r in rows))
    for player in players:
        trail = [r["action"][0].upper() for r in rows if r.get("player") == player]
        print(f"  {player:12s} {' '.join(trail)}")
    print("=" * 78)
