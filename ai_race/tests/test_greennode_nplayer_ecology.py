"""CPU-only contract tests for the frozen GreenNode N-player ecology registry."""
from __future__ import annotations

from collections import Counter
from types import SimpleNamespace

import pytest

from kaggle.experiments import greennode_nplayer_ecology as ecology


WORKERS = {
    "qwen25_7b": "worker-qwen",
    "mistral7_01": "worker-mistral",
}


@pytest.fixture(scope="module")
def games() -> list:
    return ecology.build_games(WORKERS)


def _pick(games: list, module_id: str, *, n_players: int = 3):
    return next(
        game
        for game in games
        if game._ecology["module_id"] == module_id
        and game.config.n_players == n_players
        and game.config.max_private_risk == 0.6
        and game._ecology["composition"] == "alternating_qwen_first"
    )


def test_registry_has_exactly_twelve_frozen_modules() -> None:
    assert [module.module_id for module in ecology.module_registry()] == [
        "neutral_fixed_total",
        "neutral_per_capita_normalized",
        "exact_transition_calculator",
        "char_length_placebo",
        "competitive_framing",
        "cooperative_framing",
        "pair_alliance_label",
        "pair_alliance_placebo",
        "majority_alliance_label",
        "majority_alliance_placebo",
        "accurate_checkpoint_disclosure",
        "opaque_endpoint_ids",
    ]
    assert len(ecology.MODULE_BY_ID) == 12
    assert ecology.expected_races() == 864
    assert ecology.expected_decisions() == 34_992


def test_registry_crosses_n2_to_n7_risks_and_compositions_once(games: list) -> None:
    assert len(games) == 864
    assert len({game.game_id for game in games}) == 864
    assert {game.config.n_players for game in games} == set(range(2, 8))
    assert {game.config.max_private_risk for game in games} == {0.1, 0.6, 0.9}
    assert {game._ecology["composition"] for game in games} == set(
        ecology.COMPOSITIONS
    )
    cells = Counter(
        (
            game._ecology["module_id"],
            game.config.n_players,
            game.config.max_private_risk,
            game._ecology["composition"],
            game.rep,
        )
        for game in games
    )
    assert set(cells.values()) == {1}
    assert {game.rep for game in games} == {0}


def test_per_capita_benefit_normalization_is_exact(games: list) -> None:
    for game in games:
        module_id = game._ecology["module_id"]
        n_players = game.config.n_players
        if module_id == "neutral_per_capita_normalized":
            assert game.config.benefit == 2.0 * n_players
            assert game.config.stage_payoff_safe(n_players) == pytest.approx(1.0)
            assert game._ecology["game_mechanism_change"] is True
        else:
            assert game.config.benefit == 4.0
            assert game._ecology["game_mechanism_change"] is False


@pytest.mark.parametrize(
    ("composition", "expected"),
    [
        ("all_qwen", ["qwen25_7b"] * 7),
        ("all_mistral", ["mistral7_01"] * 7),
        (
            "alternating_qwen_first",
            [
                "qwen25_7b",
                "mistral7_01",
                "qwen25_7b",
                "mistral7_01",
                "qwen25_7b",
                "mistral7_01",
                "qwen25_7b",
            ],
        ),
        (
            "alternating_mistral_first",
            [
                "mistral7_01",
                "qwen25_7b",
                "mistral7_01",
                "qwen25_7b",
                "mistral7_01",
                "qwen25_7b",
                "mistral7_01",
            ],
        ),
    ],
)
def test_model_and_worker_routes_are_exact(
    games: list, composition: str, expected: list[str]
) -> None:
    game = next(
        game
        for game in games
        if game._ecology["module_id"] == "neutral_fixed_total"
        and game.config.n_players == 7
        and game.config.max_private_risk == 0.1
        and game._ecology["composition"] == composition
    )
    assert game._ecology["seat_model_keys"] == expected
    assert game._ecology["seat_worker_ids"] == [WORKERS[key] for key in expected]


def test_transition_calculator_uses_engine_arithmetic(games: list) -> None:
    game = _pick(games, "exact_transition_calculator", n_players=3)
    rows = ecology.transition_rows(game, 0)
    assert len(rows) == 6
    safe_zero = next(
        row
        for row in rows
        if row["other_safe"] == 0 and row["own_action"] == "safe"
    )
    unsafe_zero = next(
        row
        for row in rows
        if row["other_safe"] == 0 and row["own_action"] == "unsafe"
    )
    safe_all = next(
        row
        for row in rows
        if row["other_safe"] == 2 and row["own_action"] == "safe"
    )

    assert safe_zero == {
        "other_safe": 0,
        "own_action": "safe",
        "total_safe": 1,
        "round_payoff": pytest.approx(0.0),
        "step_increment": pytest.approx(1.0),
        "progress_after": pytest.approx(1.0),
        "stage_payoff_after": pytest.approx(0.0),
        "unsafe_count_after": 0,
        "private_risk_after": pytest.approx(0.0),
    }
    assert unsafe_zero["round_payoff"] == pytest.approx(4.0 / 3.0)
    assert unsafe_zero["step_increment"] == pytest.approx(1.5)
    assert unsafe_zero["progress_after"] == pytest.approx(1.5)
    assert unsafe_zero["unsafe_count_after"] == 1
    assert unsafe_zero["private_risk_after"] == pytest.approx(0.6)
    assert safe_all["round_payoff"] == pytest.approx(1.0 / 3.0)

    block = ecology.transition_calculator_block(game, 0)
    assert "other_SAFE=0; own=safe; total_SAFE=1; round_payoff=0.000000" in block
    assert "other_SAFE=0; own=unsafe; total_SAFE=0" in block
    assert "does not predict simultaneous choices" in block


def test_calculator_and_alliance_placebos_are_character_matched(games: list) -> None:
    pairs = [
        ("exact_transition_calculator", "char_length_placebo"),
        ("pair_alliance_label", "pair_alliance_placebo"),
        ("majority_alliance_label", "majority_alliance_placebo"),
    ]
    for n_players in range(2, 8):
        for treatment, control in pairs:
            treatment_game = _pick(games, treatment, n_players=n_players)
            control_game = _pick(games, control, n_players=n_players)
            for seat in range(n_players):
                assert len(ecology.module_block(treatment_game, seat)) == len(
                    ecology.module_block(control_game, seat)
                )


def test_alliance_modules_never_claim_a_coalition_mechanism(games: list) -> None:
    for module_id in ("pair_alliance_label", "majority_alliance_label"):
        module = ecology.MODULE_BY_ID[module_id]
        assert module.alliance_framing_only is True
        assert module.game_mechanism_change is False
        for n_players in range(2, 8):
            game = _pick(games, module_id, n_players=n_players)
            assert game._ecology["alliance_framing_only"] is True
            assert game._ecology["alliance_mechanism"] is False
            assert game.config.benefit == 4.0
            for seat in range(n_players):
                block = ecology.module_block(game, seat)
                assert "framing only" in block.lower()
                assert "no coalition utility" in block.lower()
                assert "or change to any action, payoff, risk" in block.lower()
                assert "maximizing your own final payoff" in block

    for module_id in ("pair_alliance_placebo", "majority_alliance_placebo"):
        game = _pick(games, module_id)
        assert game._ecology["alliance_mechanism"] is False
        assert "LENGTH-MATCHED NEUTRAL CONTROL" in ecology.module_block(game, 0)


def test_prompt_audit_hashes_the_exact_transformed_prompt(games: list) -> None:
    game = _pick(games, "accurate_checkpoint_disclosure")
    prompt = ecology.ecology_prompt(game, 1, "BASE PROMPT\n")
    audit = game._ecology_prompt_audit[(1, 1)]
    assert audit["prompt_chars"] == len(prompt)
    assert audit["module_block_chars"] == len(ecology.module_block(game, 1))
    assert audit["prompt_sha256"] == ecology.hashlib.sha256(
        prompt.encode("utf-8")
    ).hexdigest()
    assert "Mistral-7B-Instruct-v0.1" in prompt
    assert "Qwen2.5-7B-Instruct" in prompt


def test_response_envelope_validation_is_fail_closed() -> None:
    valid = {
        "protocol": ecology.WORKER_PROTOCOL,
        "request_id": "request-1",
        "responses": ["ACTION: SAFE"],
        "error": None,
    }
    assert ecology._validate_response_envelope(
        valid,
        worker_id="worker-qwen",
        request_id="request-1",
        expected_count=1,
    ) == ["ACTION: SAFE"]

    for mutation in (
        {**valid, "protocol": "wrong"},
        {**valid, "request_id": "wrong"},
        {**valid, "responses": []},
        {**valid, "responses": [1]},
        {**valid, "error": "backend failed"},
    ):
        with pytest.raises(RuntimeError):
            ecology._validate_response_envelope(
                mutation,
                worker_id="worker-qwen",
                request_id="request-1",
                expected_count=1,
            )


def test_exact_protocol_admission_gate_requires_explicit_diagnostic_override() -> None:
    with pytest.raises(RuntimeError, match="allow-unadmitted-diagnostic"):
        ecology.run_coordinator(
            SimpleNamespace(allow_unadmitted_diagnostic=False)
        )
