from __future__ import annotations

from collections import Counter
import json
import re

import pytest

from ai_race.dataio.config_loader import load_json, validate_experiment
from ai_race.engine.state import Action
from ai_race.paths import CONFIGS_DIR
from ai_race.prompts.context_skins import (
    ACTION_CODE_PROTOCOL,
    CONTEXT_SKIN_PROTOCOL,
    MECHANISM_PLACEHOLDERS,
    SKINS,
    context_skin_sha256,
    render_context_skin,
)
from kaggle.experiments.greennode_context_skin import (
    LANE_SKINS,
    OpaqueActionBackend,
    PROFILE_REPETITIONS,
    build_context_games,
    crn_contract,
    mechanism_sha256,
    mechanism_signature,
)
from ai_race.runner.batch import run_games_batched


def _experiment(*, repetitions: int = 1) -> dict:
    experiment = load_json(
        CONFIGS_DIR / "experiment" / "context_skin_invariance.json"
    )
    experiment["repetitions"] = repetitions
    experiment["models"] = ["Mock"]
    return validate_experiment(experiment)


def _placeholder_counts(template: str) -> Counter[str]:
    return Counter(re.findall(r"\{([A-Za-z0-9_]+)\}", template))


def _finish_with_fixed_policy(game) -> None:
    schedule = (
        ("ACTION: SAFE", "ACTION: SAFE"),
        ("ACTION: SAFE", "ACTION: UNSAFE"),
        ("ACTION: UNSAFE", "ACTION: SAFE"),
        ("ACTION: UNSAFE", "ACTION: UNSAFE"),
    )
    while not game.is_finished:
        prompts = game.build_round_prompts()
        responses = schedule[(game.current_round - 1) % len(schedule)]
        game.apply_round_responses(responses, prompts=prompts)


def test_context_config_and_lane_partition_cover_each_skin_once():
    experiment = _experiment()
    assert set(experiment["contextSkins"]) == set(SKINS)
    assert set(LANE_SKINS) == {"a", "b"}
    assert set(LANE_SKINS["a"]).isdisjoint(LANE_SKINS["b"])
    assert set(LANE_SKINS["a"] + LANE_SKINS["b"]) == set(SKINS)
    assert PROFILE_REPETITIONS == {"smoke": 2, "pilot": 32}


def test_skins_share_exact_mechanism_placeholder_contract_and_are_distinct():
    templates = {skin_id: render_context_skin(skin_id) for skin_id in SKINS}
    counts = [_placeholder_counts(template) for template in templates.values()]
    assert all(count == counts[0] for count in counts[1:])
    assert MECHANISM_PLACEHOLDERS.issubset(counts[0])
    assert len({context_skin_sha256(skin_id) for skin_id in SKINS}) == len(SKINS)
    assert all("[[" not in template for template in templates.values())
    assert all("SAFE" not in template and "UNSAFE" not in template for template in templates.values())


@pytest.mark.parametrize("skin_id", sorted(SKINS))
def test_every_skin_renders_and_changes_only_prompt_metadata(skin_id: str):
    games = build_context_games(_experiment(), "Mock", skin_id)
    assert len(games) == 3
    prompt = games[0].build_round_prompts()[0]
    assert "ACTION: P" in prompt
    assert "ACTION: Q" in prompt
    assert "SAFE" not in prompt
    assert "UNSAFE" not in prompt
    assert "{weight" not in prompt
    assert games[0].config.prompt_version == (
        f"{CONTEXT_SKIN_PROTOCOL}:{skin_id}:{ACTION_CODE_PROTOCOL}:safe_p"
    )
    assert games[0].game_id.endswith(
        f"__context-{skin_id}__action-map-safe_p"
    )


def test_opaque_mapping_is_balanced_by_rep_and_orthogonal_to_context():
    experiment = _experiment(repetitions=4)
    for skin_id in SKINS:
        games = build_context_games(experiment, "Mock", skin_id)
        mapping_ids = [game.config.prompt_version.rsplit(":", 1)[-1] for game in games]
        # Three risk treatments each receive the same 2/2 mapping allocation.
        assert mapping_ids.count("safe_p") == 6
        assert mapping_ids.count("safe_q") == 6
        assert [game.rep % 2 for game in games] == [0, 1, 0, 1] * 3


class _FixedOpaqueBackend:
    def __init__(self, response: str):
        self.response = response

    def __call__(self, prompts, seeds=None):
        return [self.response for _ in prompts]


@pytest.mark.parametrize(
    ("mapping_id", "code", "expected"),
    [
        ("safe_p", "P", "ACTION: SAFE"),
        ("safe_p", "Q", "ACTION: UNSAFE"),
        ("safe_q", "P", "ACTION: UNSAFE"),
        ("safe_q", "Q", "ACTION: SAFE"),
    ],
)
def test_opaque_backend_decodes_both_balanced_mappings_and_preserves_raw(
    mapping_id: str, code: str, expected: str
):
    prompt = render_context_skin("abstract_contest", mapping_id)
    backend = OpaqueActionBackend(_FixedOpaqueBackend(f"ACTION: {code}"))
    response = backend([prompt], seeds=[123])[0]
    assert response["text"] == expected
    assert response["raw_response"] == f"ACTION: {code}"
    assert response["action_code_mapping"] == mapping_id


def test_opaque_backend_fails_closed_on_prose_or_unknown_code():
    prompt = render_context_skin("abstract_contest", "safe_p")
    for raw in ("I choose P", "ACTION: X", "ACTION: P\nBecause it is best"):
        response = OpaqueActionBackend(_FixedOpaqueBackend(raw))(
            [prompt], seeds=[123]
        )[0]
        assert response == {"text": raw, "raw_response": raw}


def test_opaque_backend_runs_engine_and_logs_original_pq_output():
    games = build_context_games(
        _experiment(repetitions=2), "Mock", "abstract_contest"
    )[:2]
    backend = OpaqueActionBackend(_FixedOpaqueBackend("ACTION: P"))
    results = run_games_batched(games, backend, max_parse_retries=0)
    assert len(results) == 2
    assert {turn.action for turn in games[0].turns} == {"safe"}
    assert {turn.action for turn in games[1].turns} == {"unsafe"}
    for game in games:
        assert all(turn.raw_response == "ACTION: P" for turn in game.turns)
        assert all(
            turn.attempt_history[0]["raw_response"] == "ACTION: P"
            for turn in game.turns
        )


def test_round_two_history_never_leaks_engine_action_names():
    games = build_context_games(
        _experiment(repetitions=2), "Mock", "abstract_contest"
    )[:2]
    expected_history = [
        "you chose P; Participant_2 chose Q",
        "you chose Q; Participant_2 chose P",
    ]
    for game, expected in zip(games, expected_history):
        round_one = game.build_round_prompts()
        game.apply_round_responses(
            ["ACTION: SAFE", "ACTION: UNSAFE"], prompts=round_one
        )
        round_two = game.build_round_prompts()[0]
        assert expected in round_two
        assert re.search(r"\b(?:SAFE|UNSAFE)\b", round_two, re.IGNORECASE) is None


def test_all_skins_have_identical_game_config_and_payoff_functions():
    experiment = _experiment()
    by_skin = {
        skin_id: build_context_games(experiment, "Mock", skin_id)
        for skin_id in SKINS
    }
    reference = by_skin["technology_race"]
    for skin_id, games in by_skin.items():
        for reference_game, game in zip(reference, games):
            assert mechanism_signature(game.config) == mechanism_signature(
                reference_game.config
            ), skin_id
            assert mechanism_sha256(game.config) == mechanism_sha256(
                reference_game.config
            )
            for own in Action:
                for opponent in Action:
                    assert game.config.stage_payoff(
                        own, opponent
                    ) == reference_game.config.stage_payoff(own, opponent)
            assert game.config.progress_for(Action.SAFE) == 1.0
            assert game.config.progress_for(Action.UNSAFE) == 1.5


def test_round_one_numeric_disclosure_and_objective_are_identical_across_skins():
    experiment = _experiment()
    prompts = {
        skin_id: build_context_games(experiment, "Mock", skin_id)[0]
        .build_round_prompts()[0]
        for skin_id in SKINS
    }
    numeric_tokens = {
        skin_id: re.findall(r"(?<![A-Za-z_])\d+(?:\.\d+)?%?", prompt)
        for skin_id, prompt in prompts.items()
    }
    assert len({tuple(tokens) for tokens in numeric_tokens.values()}) == 1
    objective = "Your goal is to maximize your own expected payoff using the information above."
    assert all(prompt.count(objective) == 1 for prompt in prompts.values())


def test_paired_crn_and_fixed_actions_produce_identical_trajectories():
    experiment = _experiment(repetitions=2)
    comparisons = {}
    contracts = {}
    for skin_id in SKINS:
        games = build_context_games(experiment, "Mock", skin_id)
        contracts[skin_id] = crn_contract(games)
        for game in games:
            _finish_with_fixed_policy(game)
        comparisons[skin_id] = [
            {
                "game_seed": game.result.game_seed,
                "n_rounds": game.result.n_rounds,
                "stop_draws": game.result.stop_draws,
                "setback_draws": game.result.setback_draws,
                "actions": game.result.per_round_actions,
                "round_payoffs": game.result.per_round_payoffs,
                "progress": game.result.progress,
                "stage_payoffs": game.result.stage_payoffs,
                "private_risks": game.result.private_risks,
                "setbacks": game.result.setbacks,
                "final_payoffs": game.result.final_payoffs,
            }
            for game in games
        ]

    reference_contract = contracts["technology_race"]
    reference_results = comparisons["technology_race"]
    for skin_id in SKINS:
        assert contracts[skin_id] == reference_contract
        assert comparisons[skin_id] == reference_results


def test_context_experiment_json_is_stable_json():
    path = CONFIGS_DIR / "experiment" / "context_skin_invariance.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["seed"] == 260726
    assert payload["games"] == [
        "ai_race_risk_10",
        "ai_race_risk_60",
        "ai_race_risk_90",
    ]
    assert payload["agents"] == "context_skin_neutral"
