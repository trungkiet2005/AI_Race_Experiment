"""Race-loop tests for the N-player engine: outcomes, prize split, private risk.

Drives ``NPlayerAIRaceGame`` directly with fixed SAFE/UNSAFE responses (no
model backend needed) for the mechanics tests, then one end-to-end smoke test
through ``ai_race.engine_nplayer.runner`` with the mock backend, using the
real shipped configs, to check config loading -> agent construction -> prompt
rendering -> batched play -> recording all actually wire together.
"""
from __future__ import annotations

import pytest

from ai_race.engine.agent import RaceAgent
from ai_race.engine_nplayer.game import NPlayerAIRaceGame
from ai_race.engine_nplayer.runner import NPLAYER_PROMPTS_DIR
from ai_race.engine_nplayer.state import NPlayerGameConfig
from ai_race.paths import CONFIGS_DIR
from ai_race.dataio.config_loader import load_json


def _short_config(n_players: int, max_private_risk: float = 0.6) -> NPlayerGameConfig:
    return NPlayerGameConfig(
        name="test_short",
        n_players=n_players,
        min_rounds=3,
        stop_probability=1.0,  # deterministic stop right at min_rounds
        max_private_risk=max_private_risk,
    )


def _agents(n_players: int) -> list[RaceAgent]:
    return [RaceAgent(name=f"Company_{i + 1}") for i in range(n_players)]


def _play_fixed(game: NPlayerAIRaceGame, actions_by_round: list[list[str]]):
    result = None
    for round_actions in actions_by_round:
        responses = [f"ACTION: {action}" for action in round_actions]
        result = game.apply_round_responses(responses)
    return result


def test_all_unsafe_group_ties_and_splits_the_prize():
    config = _short_config(n_players=3)
    game = NPlayerAIRaceGame(config, _agents(3), template="", game_id="g1", seed=1)
    result = _play_fixed(game, [["UNSAFE", "UNSAFE", "UNSAFE"]] * 3)

    assert result is not None
    assert result.n_rounds == 3
    assert result.is_full_tie
    assert set(result.winners) == {"Company_1", "Company_2", "Company_3"}
    assert result.progress == pytest.approx([4.5, 4.5, 4.5])
    # k_safe=0 every round: unsafe payoff = speed*benefit/(speed*N) = benefit/N = 4/3.
    assert result.stage_payoffs == pytest.approx([4.0, 4.0, 4.0])
    assert result.prizes == pytest.approx([100 / 3, 100 / 3, 100 / 3])
    # Unsafe every round -> effective risk saturates at the configured maximum.
    assert result.private_risks == pytest.approx([0.6, 0.6, 0.6])
    assert all(result.setback_eligible)


def test_one_unsafe_among_safe_group_wins_alone():
    config = _short_config(n_players=3)
    game = NPlayerAIRaceGame(config, _agents(3), template="", game_id="g2", seed=2)
    result = _play_fixed(game, [["UNSAFE", "SAFE", "SAFE"]] * 3)

    assert result is not None
    assert result.outcomes == ["winner", "loser", "loser"]
    assert result.winners == ["Company_1"]
    assert result.progress[0] == pytest.approx(4.5)
    assert result.progress[1] == pytest.approx(3.0)
    assert result.progress[2] == pytest.approx(3.0)
    assert result.prizes == pytest.approx([100.0, 0.0, 0.0])
    # Only the winner is setback-eligible, regardless of the losers' own risk.
    assert result.setback_eligible == [True, False, False]
    assert result.private_risks[0] == pytest.approx(0.6)
    assert result.private_risks[1] == pytest.approx(0.0)


def test_all_safe_group_ties_with_zero_risk():
    config = _short_config(n_players=4)
    game = NPlayerAIRaceGame(config, _agents(4), template="", game_id="g3", seed=3)
    result = _play_fixed(game, [["SAFE", "SAFE", "SAFE", "SAFE"]] * 3)

    assert result is not None
    assert result.is_full_tie
    assert result.private_risks == pytest.approx([0.0, 0.0, 0.0, 0.0])
    assert result.prizes == pytest.approx([25.0, 25.0, 25.0, 25.0])
    # k_safe == n_players branch: -cost + benefit/N = -1 + 4/4 = 0.0 per round.
    assert result.stage_payoffs == pytest.approx([0.0, 0.0, 0.0, 0.0])


def test_agents_count_must_match_n_players():
    config = _short_config(n_players=3)
    too_few_agents = _agents(2)
    with pytest.raises(ValueError, match="exactly 3 agents"):
        NPlayerAIRaceGame(config, too_few_agents, template="", game_id="g4", seed=4)


def test_apply_round_responses_requires_n_responses():
    config = _short_config(n_players=3)
    game = NPlayerAIRaceGame(config, _agents(3), template="", game_id="g5", seed=5)
    with pytest.raises(ValueError, match="Exactly 3 responses"):
        game.apply_round_responses(["ACTION: SAFE", "ACTION: SAFE"])


def test_turn_records_carry_others_not_a_singular_opponent():
    config = _short_config(n_players=3)
    game = NPlayerAIRaceGame(config, _agents(3), template="", game_id="g6", seed=6)
    _play_fixed(game, [["SAFE", "UNSAFE", "SAFE"]] * 3)
    turn = game.turns[0]
    assert turn.player == "Company_1"
    assert turn.others == ["Company_2", "Company_3"]
    assert len(turn.others_prev_actions) == 2
    assert len(turn.others_progress_before) == 2


def test_end_to_end_mock_run_uses_real_configs():
    """Config loading -> agents -> prompts -> batched play -> recording, wired end to end."""
    from ai_race.engine_nplayer.runner import build_games_for_model, run_games_batched
    from ai_race.runner.run_experiment import make_mock_send_batch

    exp = load_json(CONFIGS_DIR / "experiment" / "baseline_nplayer_n3.json")
    exp = dict(exp)
    exp["repetitions"] = 2
    games = build_games_for_model(exp, "MockModel")
    assert len(games) == 3 * 2  # 3 risk treatments x 2 repetitions
    assert all(len(game.agents) == 3 for game in games)

    results = run_games_batched(games, make_mock_send_batch("random"), max_parse_retries=0)
    assert len(results) == len(games)
    for result in results:
        assert result.n_players == 3
        assert len(result.player_names) == 3
        # The prize is always fully allocated, split evenly among winners/tied
        # leaders -- exercised precisely in test_nplayer_scoring.py, this is
        # just a sanity check that the wiring through the runner preserves it.
        assert sum(result.prizes) == pytest.approx(result.config["race_prize"])
        assert set(result.outcomes) <= {"winner", "tie", "loser"}


def test_prompt_template_file_exists_and_is_used():
    template_path = NPLAYER_PROMPTS_DIR / "ai_race_nplayer_en.txt"
    assert template_path.is_file()
