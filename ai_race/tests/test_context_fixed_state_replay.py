from __future__ import annotations

from collections import Counter
import math

from ai_race.audit.context_replay import (
    build_context_probe_bank,
    comprehension_summary,
    generate_reachable_states,
    paired_coverage,
    parse_opaque_action,
    render_replay_prompt,
    run_replay_matrix,
    score_context_probe,
)
from ai_race.engine.state import Action, GameConfig
from ai_race.prompts.context_skins import ACTION_CODE_MAPPINGS, SKINS
from kaggle.experiments.greennode_context_replay import (
    LANE_SKINS,
    PROFILE_STATES_PER_RISK,
)


def _config(risk: float = 0.6) -> GameConfig:
    return GameConfig(
        name=f"test-risk-{risk:g}",
        max_private_risk=risk,
        history_mode="first_and_previous",
    )


def _recompute(state):
    config = GameConfig(**state.config)
    progress = [0.0, 0.0]
    payoffs = [0.0, 0.0]
    unsafe = [0, 0]
    for record in state.history:
        actions = [Action.coerce(value) for value in record["actions"]]
        for player_index in (0, 1):
            opponent_index = 1 - player_index
            progress[player_index] += config.progress_for(actions[player_index])
            payoffs[player_index] += config.stage_payoff(
                actions[player_index], actions[opponent_index]
            )
            unsafe[player_index] += actions[player_index].unsafe
    return progress, payoffs, unsafe


def test_reachable_state_bank_is_deterministic_diverse_and_engine_consistent():
    configs = [_config(0.1), _config(0.6), _config(0.9)]
    first = generate_reachable_states(configs, states_per_config=8, base_seed=260726)
    second = generate_reachable_states(configs, states_per_config=8, base_seed=260726)
    assert [state.to_dict() for state in first] == [state.to_dict() for state in second]
    assert Counter(state.max_private_risk for state in first) == {
        0.1: 8,
        0.6: 8,
        0.9: 8,
    }
    assert len({state.state_id for state in first}) == len(first)
    assert {1, 4, 5}.issubset({state.round_number for state in first})
    assert max(state.round_number for state in first) >= 9
    for state in first:
        progress, payoffs, unsafe = _recompute(state)
        assert progress == list(state.progress)
        assert payoffs == list(state.stage_payoffs)
        assert unsafe == list(state.unsafe_counts)
        assert len(state.history) == state.round_number - 1


def test_every_replay_cell_uses_same_state_and_hides_canonical_labels():
    state = generate_reachable_states(
        [_config()], states_per_config=8, base_seed=260726
    )[-1]
    assert state.round_number > 1
    prompts = {}
    for skin_id in SKINS:
        for mapping_id in ACTION_CODE_MAPPINGS:
            prompt = render_replay_prompt(
                state, skin_id=skin_id, mapping_id=mapping_id
            )
            prompts[(skin_id, mapping_id)] = prompt
            assert "ACTION: P" in prompt and "ACTION: Q" in prompt
            assert "SAFE" not in prompt and "UNSAFE" not in prompt
            assert str(state.round_number) in prompt
    assert len(set(prompts.values())) == len(SKINS) * len(ACTION_CODE_MAPPINGS)


def test_opaque_action_decoding_is_balanced_and_strict():
    assert parse_opaque_action("ACTION: P", "safe_p") == (
        "safe",
        0,
        "P",
        False,
    )
    assert parse_opaque_action("ACTION: P", "safe_q") == (
        "unsafe",
        1,
        "P",
        False,
    )
    assert parse_opaque_action("I choose P", "safe_p") == (None, None, None, True)


def test_comprehension_bank_covers_all_admission_domains_and_scores_answers():
    probes = build_context_probe_bank(_config(), "safe_p")
    assert Counter(probe.domain for probe in probes) == {
        "rule_recall": 4,
        "stage_payoff": 4,
        "state_update": 4,
        "terminal_scoring": 4,
    }
    rows = []
    for probe in probes:
        response = f"ANSWER: {probe.expected}"
        score = score_context_probe(probe, response)
        assert score.strict_valid
        assert score.semantic_correct
        rows.append({"domain": probe.domain, **score.__dict__})
    summary = comprehension_summary(rows)
    assert summary["passed"]
    assert math.isclose(summary["semantic_accuracy"], 1.0)


class _AlwaysPBackend:
    def __call__(self, prompts, seeds=None):
        assert seeds is not None and len(seeds) == len(prompts)
        return ["ACTION: P"] * len(prompts)


def test_replay_matrix_is_rectangular_and_emits_direct_estimand_rows():
    states = generate_reachable_states(
        [_config()], states_per_config=3, base_seed=260726
    )
    skins = ["abstract_contest", "hospital_deployment"]
    rows = run_replay_matrix(
        states,
        skins,
        _AlwaysPBackend(),
        batch_size=3,
        max_parse_retries=0,
    )
    coverage = paired_coverage(rows, skins)
    assert coverage["passed"]
    assert len(rows) == 3 * 2 * 2
    for state in states:
        subset = [row for row in rows if row["state_id"] == state.state_id]
        assert {row["skin_id"] for row in subset} == set(skins)
        assert {row["mapping_id"] for row in subset} == set(ACTION_CODE_MAPPINGS)
        assert {row["action"] for row in subset} == {"safe", "unsafe"}
        assert {row["pair_id"] for row in subset} == {state.state_id}
        assert all(row["prompt"] and row["raw_response"] for row in subset)


def test_replay_profiles_and_two_gpu_lanes_cover_all_skins_once():
    assert PROFILE_STATES_PER_RISK == {"smoke": 2, "pilot": 32}
    assert set(LANE_SKINS) == {"a", "b"}
    assert set(LANE_SKINS["a"]).isdisjoint(LANE_SKINS["b"])
    assert set(LANE_SKINS["a"] + LANE_SKINS["b"]) == set(SKINS)
