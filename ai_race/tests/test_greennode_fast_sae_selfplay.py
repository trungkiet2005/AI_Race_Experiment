from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import torch

from kaggle.interpretability import greennode_fast_sae_selfplay as runner
from kaggle.interpretability.greennode_fast_sae_selfplay import (
    choose_from_log_odds,
    feature_ablation_delta,
    grouped_race_split,
    rank_features,
)


def _records() -> list[dict]:
    rows = []
    for treatment in ("risk10", "risk60"):
        for race in range(4):
            for decision in range(2):
                rows.append(
                    {
                        "treatment": treatment,
                        "game_id": f"{treatment}-race-{race}",
                        "unsafe_log_odds": float((race + decision) % 2),
                        "action": "unsafe" if (race + decision) % 2 else "safe",
                    }
                )
    return rows


def test_grouped_split_is_deterministic_disjoint_and_stratified() -> None:
    records = _records()
    first = grouped_race_split(records, eval_fraction=0.25, seed=17)
    second = grouped_race_split(records, eval_fraction=0.25, seed=17)
    assert first == second
    assert set(first.values()) == {"discovery", "eval"}
    for treatment in ("risk10", "risk60"):
        assignments = {
            first[row["game_id"]] for row in records if row["treatment"] == treatment
        }
        assert assignments == {"discovery", "eval"}


def test_feature_ranking_uses_discovery_but_reports_eval() -> None:
    records = _records()
    split = grouped_race_split(records, eval_fraction=0.25, seed=17)
    target = np.asarray([row["unsafe_log_odds"] for row in records], dtype=np.float32)
    codes = np.column_stack(
        [
            target + 0.01 * np.arange(len(records)),
            np.ones(len(records), dtype=np.float32),
            np.arange(len(records), dtype=np.float32) % 3,
        ]
    )
    ranking = rank_features(codes, records, split, min_prevalence=0.01)
    assert ranking[0]["feature_id"] == 0
    assert "discovery_corr_log_odds" in ranking[0]
    assert "eval_corr_log_odds" in ranking[0]


def test_constrained_choice_temperature_zero_and_seeded_sampling() -> None:
    assert choose_from_log_odds(2.0, temperature=0.0, seed=1) == ("unsafe", 1.0)
    assert choose_from_log_odds(-2.0, temperature=0.0, seed=1) == ("safe", 0.0)
    assert choose_from_log_odds(0.7, temperature=1.0, seed=9) == choose_from_log_odds(
        0.7, temperature=1.0, seed=9
    )


def test_feature_ablation_removes_original_decoder_contribution() -> None:
    decoder_row = torch.tensor([3.0, 4.0])
    assert torch.equal(
        feature_ablation_delta(2.0, decoder_row), torch.tensor([-6.0, -8.0])
    )


def test_selfplay_writes_real_engine_turns_and_terminal_payoffs(tmp_path, monkeypatch) -> None:
    class FakeTokenizer:
        def apply_chat_template(self, messages, **kwargs):  # noqa: ANN001, ANN003
            return "<assistant>\n" + messages[0]["content"]

    class FakeModel:
        tokenizer = FakeTokenizer()

    class FakeSAE:
        @staticmethod
        def encode(activation):  # noqa: ANN001
            return torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32)

        @staticmethod
        def decode(code):  # noqa: ANN001
            return torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    def fake_score(model, prompt, **kwargs):  # noqa: ANN001, ANN003
        del model, prompt, kwargs
        return {"safe": -2.0, "unsafe": -1.0}, torch.tensor([1.0, 0.0])

    monkeypatch.setattr(runner, "score_action_pair", fake_score)
    args = SimpleNamespace(
        treatments=["ai_race_risk_60"],
        repetitions=2,
        base_seed=20260801,
        prompt_variant="canonical",
        layer=12,
        decision_temperature=0.0,
    )
    manifest_path = tmp_path / "manifest.json"
    manifest = {"config_fingerprint": "test-fingerprint", "stages": {}}
    runner._atomic_json(manifest_path, manifest)
    runner._selfplay(args, FakeModel(), FakeSAE(), manifest_path, manifest)

    shards = sorted((tmp_path / "race_shards").glob("*.json"))
    assert len(shards) == 2
    payload = json.loads(shards[0].read_text(encoding="utf-8"))
    assert payload["engine_result"]["parse_failures"] == 0
    assert len(payload["engine_result"]["final_payoffs"]) == 2
    assert payload["records"][0]["engine_turn"]["action"] == "unsafe"
    assert payload["records"][0]["engine_turn"]["round_payoff"] == 2.0


def test_live_steering_runs_real_paired_trajectories_with_crn(tmp_path, monkeypatch) -> None:
    class FakeModel:
        pass

    class FakeSAE:
        W_dec = torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=torch.float32
        )

    # Zero chooses Unsafe; every edited condition chooses Safe. This forces a
    # first-round flip and then endogenous state divergence on later rounds.
    def fake_score(model, prompt, **kwargs):  # noqa: ANN001, ANN003
        del model, prompt
        if kwargs.get("edit_factory") is None:
            return {"safe": -2.0, "unsafe": -1.0}, None
        return {"safe": -1.0, "unsafe": -2.0}, None

    monkeypatch.setattr(runner, "score_action_pair", fake_score)
    args = SimpleNamespace(
        treatments=["ai_race_risk_60"],
        repetitions=2,
        base_seed=20260801,
        prompt_variant="canonical",
        layer=12,
        decision_temperature=0.0,
        live_alpha=1.0,
        steered_play_max_races_per_treatment=1,
    )
    model_label = f"{runner.MODEL_REPO}@{runner.MODEL_REVISION[:12]}+constrained-sequence"
    games = runner.build_games_for_model(runner._live_experiment(args, model_label), model=model_label)
    feature_summary = {
        "race_split": {games[0].game_id: "discovery", games[1].game_id: "eval"},
        "selected_features": [{"feature_id": 0, "discovery_scale": 1.0}],
        "unrelated_control_feature": {"feature_id": 1},
    }
    manifest_path = tmp_path / "manifest.json"
    manifest = {"config_fingerprint": "live-test", "stages": {}}
    runner._atomic_json(manifest_path, manifest)
    runner._steered_play(
        args, FakeModel(), FakeSAE(), manifest_path, manifest, feature_summary
    )

    summary = json.loads((tmp_path / "steered_play_summary.json").read_text(encoding="utf-8"))
    assert summary["n_unique_race_seeds"] == 1
    assert summary["common_random_numbers"]["hidden_horizon_stream"] is True
    rows = [
        json.loads(line)
        for line in (tmp_path / "steered_play_races.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    zero = next(row for row in rows if row["condition"]["condition_id"] == "zero")
    target = next(
        row
        for row in rows
        if row["condition"]["condition_id"] == "feature-0__target_feature__positive"
    )
    assert target["paired_zero"]["stop_draws_identical"] is True
    assert target["paired_zero"]["setback_draws_identical"] is True
    assert target["decisions"][0]["state_comparable_to_zero"] is True
    assert target["decisions"][0]["action_flipped_if_comparable"] == 1
    assert any(not row["state_comparable_to_zero"] for row in target["decisions"][2:])
    assert len(zero["engine_result"]["final_payoffs"]) == 2
