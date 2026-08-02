from __future__ import annotations

from collections import Counter, defaultdict

import pandas as pd
import pytest

from ai_race.dataio.config_loader import load_json, validate_experiment
from ai_race.paths import CONFIGS_DIR
from ai_race.prompts.context_skins import ACTION_CODE_MAPPINGS, SKINS
from kaggle.experiments.greennode_context_mapping_cross import (
    CROSS_LANE_SKINS,
    LANE_REP_PARITY,
    crossed_crn_contract,
)
from kaggle.experiments.greennode_context_skin import (
    build_fully_crossed_context_games,
    mechanism_sha256,
)
from results.scripts.analyze_context_mapping_cross import discover, holm, paired_rows


def _experiment(*, repetitions: int = 2) -> dict:
    payload = load_json(
        CONFIGS_DIR / "experiment" / "context_mapping_fully_crossed.json"
    )
    payload["repetitions"] = repetitions
    payload["models"] = ["Mock"]
    return validate_experiment(payload)


def test_fully_crossed_config_freezes_both_mappings_and_all_skins() -> None:
    experiment = _experiment()
    assert experiment["runPhase"] == "pilot"
    assert experiment["actionCodeAssignment"] == "fully_crossed_within_seed"
    assert set(experiment["actionCodeMappings"]) == set(ACTION_CODE_MAPPINGS)
    assert set(experiment["contextSkins"]) == set(SKINS)
    assert all(set(skins) == set(SKINS) for skins in CROSS_LANE_SKINS.values())
    assert set(LANE_REP_PARITY.values()) == {0, 1}


def test_every_seed_risk_and_context_has_both_opaque_mappings() -> None:
    experiment = _experiment(repetitions=3)
    for skin_id in SKINS:
        games = build_fully_crossed_context_games(experiment, "Mock", skin_id)
        assert len(games) == 3 * 3 * 2
        counts = Counter(
            (game.config.name, game.rep, game.action_code_mapping.id)
            for game in games
        )
        assert set(counts.values()) == {1}
        for game_name in experiment["games"]:
            for rep in range(3):
                observed = {
                    mapping_id
                    for name, observed_rep, mapping_id in counts
                    if name == game_name and observed_rep == rep
                }
                assert observed == set(ACTION_CODE_MAPPINGS)


def test_mapping_pairs_share_seed_sampling_streams_and_mechanism() -> None:
    games = build_fully_crossed_context_games(
        _experiment(repetitions=2), "Mock", "abstract_contest"
    )
    rows = crossed_crn_contract(games)
    blocks = defaultdict(list)
    for row in rows:
        blocks[(row["game_name"], row["rep"])].append(row)

    assert len(blocks) == 3 * 2
    for block_rows in blocks.values():
        assert {row["mapping_id"] for row in block_rows} == set(
            ACTION_CODE_MAPPINGS
        )
        assert len({row["game_seed"] for row in block_rows}) == 1
        assert len({row["seat0_round1_sampling_seed"] for row in block_rows}) == 1
        assert len({row["seat1_round1_sampling_seed"] for row in block_rows}) == 1
        assert len({row["mechanism_sha256"] for row in block_rows}) == 1


def test_mapping_changes_only_prompt_contract_not_mechanism() -> None:
    games = build_fully_crossed_context_games(
        _experiment(repetitions=1), "Mock", "logistics_contract"
    )
    grouped = defaultdict(list)
    for game in games:
        grouped[(game.config.name, game.rep)].append(game)

    for pair in grouped.values():
        assert len(pair) == 2
        assert pair[0].seed == pair[1].seed
        assert mechanism_sha256(pair[0].config) == mechanism_sha256(pair[1].config)
        assert pair[0].config.prompt_version != pair[1].config.prompt_version
        prompts = [game.build_round_prompts()[0] for game in pair]
        assert prompts[0] != prompts[1]
        assert all("ACTION: P" in prompt and "ACTION: Q" in prompt for prompt in prompts)
        assert all("SAFE" not in prompt and "UNSAFE" not in prompt for prompt in prompts)


def test_analysis_recovers_a_known_mapping_interaction() -> None:
    rows = []
    for skin_id in ("abstract_contest", "technology_race"):
        for mapping_id in ACTION_CODE_MAPPINGS:
            rows.append(
                {
                    "skin_id": skin_id,
                    "max_private_risk": 0.1,
                    "rep": 0,
                    "player_index": 0,
                    "mapping_id": mapping_id,
                    "unsafe_frequency": (
                        0.7
                        if skin_id == "technology_race" and mapping_id == "safe_p"
                        else 0.2
                    ),
                }
            )
    result = paired_rows(pd.DataFrame(rows))
    technology = result[result["context"] == "technology_race"].iloc[0]
    assert technology["safe_p"] == pytest.approx(0.5)
    assert technology["safe_q"] == pytest.approx(0.0)
    assert technology["interaction_did"] == pytest.approx(0.5)


def test_holm_adjustment_is_monotone_in_sorted_p_values() -> None:
    raw = [0.04, 0.001, 0.02]
    adjusted = holm(raw)
    by_raw = sorted(zip(raw, adjusted))
    assert all(by_raw[index][1] <= by_raw[index + 1][1] for index in range(2))
    assert all(raw_value <= adjusted_value <= 1 for raw_value, adjusted_value in zip(raw, adjusted))


def test_analysis_discovery_requires_both_lanes_for_every_skin(tmp_path) -> None:
    for lane in ("a", "b"):
        for skin in SKINS:
            directory = tmp_path / lane / skin
            directory.mkdir(parents=True)
            (directory / "run_manifest.json").write_text(
                __import__("json").dumps(
                    {
                        "schema_version": "ai-race-context-mapping-cross-run-v1",
                        "lane": lane,
                        "context_skin": {"id": skin},
                    }
                ),
                encoding="utf-8",
            )
    assert len(discover(tmp_path)) == 2 * len(SKINS)
    (tmp_path / "b" / next(iter(SKINS)) / "run_manifest.json").unlink()
    with pytest.raises(ValueError, match="two lanes"):
        discover(tmp_path)
