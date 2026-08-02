"""Build and execute config-driven N-player AI Race experiments.

Mirrors ``ai_race.runner.run_experiment`` + ``ai_race.runner.batch`` for
``NPlayerAIRaceGame``. The one real bug those modules would have if reused
directly is ``batch.py``'s ``cursor : cursor + 2`` response slicing, which
hardcodes "every game contributes exactly two requests per round" --
``run_games_batched`` below slices with ``len(game.agents)`` per active game
instead. Reuses ``ai_race.runner.run_experiment.{model_slug,
make_mock_send_batch}`` and ``ai_race.dataio.config_loader.{load_json,
validate_experiment, personas_sha256}`` unchanged: none of those depend on
the number of players.
"""
from __future__ import annotations

import json
from collections.abc import Callable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from ai_race.dataio.config_loader import load_json, personas_sha256, validate_experiment
from ai_race.engine.agent import RaceAgent
from ai_race.engine.round import parse_action, response_text
from ai_race.models.factory import get_send_batch, init_offline_backend
from ai_race.paths import CONFIGS_DIR, RESULTS_DIR
from ai_race.runner.run_experiment import make_mock_send_batch, model_slug

from .game import NPlayerAIRaceGame
from .recorder import NPlayerRunJournal, write_players_csv, write_races_csv
from .state import NPlayerGameConfig

# Templates live inside this package, not ai_race/prompts/: that directory's
# contents are hash-pinned against results/scripts/analyze_ai_race.py's
# CANONICAL_PROMPT_SHA256_BY_TEMPLATE registry for the two-player paper-faithful
# contract (see ai_race/tests/test_prompt_contract.py), which this module is
# deliberately outside of.
NPLAYER_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


def _load_agents(exp: dict[str, Any], agents_name: Optional[str]) -> tuple[str, dict]:
    name = agents_name or str(exp.get("agents", "companies_nplayer_default"))
    path = CONFIGS_DIR / "agents_nplayer" / f"{name}.json"
    return name, load_json(path)


def _validate_agents_nplayer(agents_cfg: dict, n_players: int) -> None:
    names = agents_cfg.get("names")
    if not isinstance(names, list) or len(names) != n_players:
        raise ValueError(f"Agent configuration must define exactly {n_players} names")
    if len(set(map(str, names))) != n_players:
        raise ValueError("Agent names must be distinct so seats stay identifiable")
    declared_n = agents_cfg.get("nPlayers")
    if declared_n is not None and int(declared_n) != n_players:
        raise ValueError(
            f"Agent configuration declares nPlayers={declared_n} but the game "
            f"config requires {n_players}"
        )


def _agents_for_language(
    agents_cfg: dict, language: str, n_players: int
) -> list[RaceAgent]:
    _validate_agents_nplayer(agents_cfg, n_players)
    names = list(agents_cfg.get("names", []))
    personas_by_language = agents_cfg.get("personas", {}) or {}
    personas = list(personas_by_language.get(language, [""] * n_players))
    if len(personas) != n_players:
        raise ValueError(
            f"Agent configuration must define {n_players} {language!r} personas"
        )
    probabilities_by_language = agents_cfg.get("personaProbabilities", {}) or {}
    probabilities = list(
        probabilities_by_language.get(language, [100.0] * n_players)
    )
    if len(probabilities) != n_players:
        raise ValueError(
            f"Agent configuration must define {n_players} {language!r} "
            "personaProbabilities"
        )
    condition = str(agents_cfg.get("personaCondition", "none")).strip()
    if condition != "none" and not all(str(text).strip() for text in personas):
        raise ValueError(
            f"personaCondition={condition!r} has no {language!r} persona text; "
            "translate the personas before running that language"
        )
    roles = list(agents_cfg.get("personaRoles", [""] * n_players))
    return [
        RaceAgent(
            name=str(name),
            persona_text=str(persona),
            persona_probability=float(probability),
            persona_role=str(role),
        )
        for name, persona, probability, role in zip(names, personas, probabilities, roles)
    ]


def build_games_for_model(
    exp: dict[str, Any],
    model: str,
    agents_cfg: Optional[dict] = None,
    agents_name: Optional[str] = None,
) -> list[NPlayerAIRaceGame]:
    """Construct all treatment x language x repetition races for one model."""
    validate_experiment(exp)
    if agents_cfg is None:
        resolved_agents_name, agents_cfg = _load_agents(exp, agents_name)
    else:
        resolved_agents_name = agents_name or str(
            agents_cfg.get("name", exp.get("agents", "companies_nplayer_default"))
        )

    persona_condition = str(agents_cfg.get("personaCondition", "none")).strip()
    persona_hash = personas_sha256(agents_cfg.get("personas", {}) or {})

    repetitions = int(exp["repetitions"])
    base_seed = int(exp["seed"])
    languages = list(exp.get("languages", ["en"]))
    games: list[NPlayerAIRaceGame] = []

    for game_name in exp["games"]:
        game_path = CONFIGS_DIR / "game_nplayer" / f"{game_name}.json"
        game_data = load_json(game_path)
        for language in languages:
            config = NPlayerGameConfig.from_dict(
                game_data,
                language=str(language),
                model=str(model),
                sampling_seed_applied=bool(
                    exp.get("samplingSeedApplied", exp.get("useOffline", True))
                ),
                run_phase=str(exp.get("runPhase", "pilot")),
                persona_condition=persona_condition,
                persona_sha256=persona_hash,
            )
            template_path = NPLAYER_PROMPTS_DIR / f"{config.prompt_template}.txt"
            template = template_path.read_text(encoding="utf-8")
            for rep in range(repetitions):
                # Deliberately independent of the treatment name, same as the
                # two-player engine: matched repetitions share the horizon and
                # fixed-seat setback draws across risk conditions.
                game_seed = base_seed + rep
                game_id = (
                    f"{config.name}__{model_slug(model)}__{language}__"
                    f"{resolved_agents_name}__n{config.n_players}__rep{rep:04d}"
                )
                games.append(
                    NPlayerAIRaceGame(
                        config,
                        _agents_for_language(agents_cfg, str(language), config.n_players),
                        template=template,
                        game_id=game_id,
                        seed=game_seed,
                        rep=rep,
                    )
                )
    return games


def _send(
    send_batch: Callable[..., Sequence[str | dict[str, Any]]],
    prompts: list[str],
    seeds: list[int],
) -> list[str | dict[str, Any]]:
    responses = list(send_batch(prompts, seeds=seeds))
    if len(responses) != len(prompts):
        raise RuntimeError(
            f"Model backend returned {len(responses)} responses for {len(prompts)} prompts"
        )
    return responses


def _attempt_record(
    response: str | dict[str, Any],
    *,
    attempt: int,
    sampling_seed: int,
    sampling_seed_applied: bool,
) -> dict[str, Any]:
    action, parse_failed = parse_action(response)
    return {
        "attempt": int(attempt),
        "sampling_seed": int(sampling_seed) if sampling_seed_applied else None,
        "sampling_seed_applied": bool(sampling_seed_applied),
        "raw_response": response_text(response),
        "parse_failed": bool(parse_failed),
        "parsed_action": None if parse_failed else action.value,
    }


def run_games_batched(
    games: Sequence[Any],
    send_batch: Callable[..., Sequence[str | dict[str, Any]]],
    *,
    verbose: bool = False,
    max_parse_retries: int = 3,
    on_round_complete: (
        Callable[[Any, Any | None, list[Any]], None] | None
    ) = None,
) -> list[Any]:
    """Evaluate all active races one round at a time.

    Same lockstep-batching shape as ``ai_race.runner.batch.run_games_batched``,
    except the per-game response slice uses ``len(game.agents)`` instead of a
    hardcoded 2 -- that literal is what makes the two-player version unsafe to
    reuse for N != 2.
    """
    if max_parse_retries < 0:
        raise ValueError("max_parse_retries cannot be negative")
    games = list(games)
    if any(game.is_finished for game in games):
        raise ValueError("All input games must be unstarted")
    results: list[Any] = []

    while True:
        active = [game for game in games if not game.is_finished]
        if not active:
            break

        requests: list[tuple[Any, int, str, int]] = []
        prompts_by_game: dict[str, list[str]] = {}
        for game in active:
            prompts = game.build_round_prompts()
            prompts_by_game[game.game_id] = prompts
            for player_index, prompt in enumerate(prompts):
                requests.append(
                    (
                        game,
                        player_index,
                        prompt,
                        game.sampling_seed(player_index, game.current_round, 0),
                    )
                )

        responses = _send(
            send_batch,
            [request[2] for request in requests],
            [request[3] for request in requests],
        )
        retry_counts = [0 for _ in requests]
        attempt_histories = [
            [
                _attempt_record(
                    response,
                    attempt=0,
                    sampling_seed=requests[index][3],
                    sampling_seed_applied=bool(
                        requests[index][0].config.sampling_seed_applied
                    ),
                )
            ]
            for index, response in enumerate(responses)
        ]

        for retry in range(1, max_parse_retries + 1):
            failed_indices = [
                index
                for index, response in enumerate(responses)
                if parse_action(response)[1]
            ]
            if not failed_indices:
                break
            retry_prompts = [requests[index][2] for index in failed_indices]
            retry_seeds = [
                requests[index][0].sampling_seed(
                    requests[index][1],
                    requests[index][0].current_round,
                    retry,
                )
                for index in failed_indices
            ]
            retry_responses = _send(send_batch, retry_prompts, retry_seeds)
            for index, seed, response in zip(failed_indices, retry_seeds, retry_responses):
                responses[index] = response
                retry_counts[index] = retry
                attempt_histories[index].append(
                    _attempt_record(
                        response,
                        attempt=retry,
                        sampling_seed=seed,
                        sampling_seed_applied=bool(
                            requests[index][0].config.sampling_seed_applied
                        ),
                    )
                )

        cursor = 0
        for game in active:
            n = len(game.agents)
            game_responses = responses[cursor : cursor + n]
            game_retries = retry_counts[cursor : cursor + n]
            game_attempts = attempt_histories[cursor : cursor + n]
            cursor += n
            previous_turn_count = len(game.turns)
            result = game.apply_round_responses(
                game_responses,
                prompts=prompts_by_game[game.game_id],
                retry_counts=game_retries,
                attempt_histories=game_attempts,
            )
            new_turns = list(game.turns[previous_turn_count:])
            if on_round_complete is not None:
                on_round_complete(game, result, new_turns)
            if verbose:
                actions = game.history[-1]["actions"]
                print(
                    f"[{game.game_id}] round={len(game.history)} "
                    f"actions={actions} progress={game.progress}"
                )
            if result is not None:
                results.append(result)
                if verbose:
                    print(
                        f"[done] {game.game_id}: rounds={result.n_rounds} "
                        f"winners={result.winners} setbacks={result.setbacks}"
                    )
    return results


def _agents_provenance(exp: dict[str, Any]) -> dict[str, Any]:
    name = str(exp.get("agents", "companies_nplayer_default"))
    path = CONFIGS_DIR / "agents_nplayer" / f"{name}.json"
    agents_cfg = load_json(path)
    return {
        "agents_name": name,
        "persona_condition": str(agents_cfg.get("personaCondition", "none")).strip(),
        "persona_roles": [str(role) for role in agents_cfg.get("personaRoles", [])],
        "persona_sha256": personas_sha256(agents_cfg.get("personas", {}) or {}),
    }


def _write_manifest(
    path: Path,
    *,
    experiment: dict[str, Any],
    model: str,
    n_races: int,
    n_turns: int,
    status: str,
    error: Optional[str] = None,
) -> None:
    manifest = {
        "schema_version": "ai-race-nplayer-results-v1",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "experiment": experiment,
        "model": model,
        "run_phase": str(experiment.get("runPhase", "pilot")),
        **_agents_provenance(experiment),
        "status": status,
        "n_races": n_races,
        "n_turns": n_turns,
    }
    if error:
        manifest["error"] = str(error)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")


def run_experiment(
    exp: dict[str, Any],
    *,
    output_root: Path,
    mock_strategy: Optional[str] = None,
) -> list[Any]:
    """Run all configured models and persist a separate result directory per model."""
    validate_experiment(exp)
    all_results: list[Any] = []
    for model_index, model in enumerate(exp["models"]):
        model_dir = output_root / model_slug(str(model))
        journal = NPlayerRunJournal(model_dir, reset=True)
        _write_manifest(
            model_dir / "run_manifest.json",
            experiment=exp,
            model=str(model),
            n_races=0,
            n_turns=0,
            status="running",
        )
        try:
            games = build_games_for_model(exp, str(model))
            if mock_strategy:
                send_batch = make_mock_send_batch(mock_strategy)
                max_retries = 0
            elif bool(exp.get("useOffline", True)):
                settings_name = str(exp.get("offlineSettings", "offline_settings"))
                settings = load_json(CONFIGS_DIR / "offline" / f"{settings_name}.json")
                preset_name = str(
                    (exp.get("modelPresets", {}) or {}).get(model, settings["modelPreset"])
                )
                preset = load_json(CONFIGS_DIR / "offline" / f"{preset_name}.json")
                init_offline_backend(settings, preset, force=model_index > 0)
                send_batch = get_send_batch(
                    str(model),
                    offline=True,
                    batch_size=int(settings.get("batchSize", 0)),
                )
                max_retries = int(
                    exp.get("maxParseRetries", settings.get("maxParseRetries", 3))
                )
            else:
                send_batch = get_send_batch(
                    str(model),
                    offline=False,
                    backend=str(exp.get("backend", "api")),
                    proxy_options=dict(exp.get("proxyOptions", {}) or {}),
                )
                max_retries = int(exp.get("maxParseRetries", 3))

            results = run_games_batched(
                games,
                send_batch,
                verbose=bool(exp.get("verbose", True)),
                max_parse_retries=max_retries,
                on_round_complete=journal.record_round,
            )
        except Exception as exc:
            _write_manifest(
                model_dir / "run_manifest.json",
                experiment=exp,
                model=str(model),
                n_races=journal.race_count,
                n_turns=journal.turn_count,
                status="failed",
                error=f"{type(exc).__name__}: {exc}",
            )
            raise
        _write_manifest(
            model_dir / "run_manifest.json",
            experiment=exp,
            model=str(model),
            n_races=journal.race_count,
            n_turns=journal.turn_count,
            status="completed",
        )
        all_results.extend(results)
    write_races_csv(all_results, output_root / "races.csv")
    write_players_csv(all_results, output_root / "players.csv")
    return all_results
