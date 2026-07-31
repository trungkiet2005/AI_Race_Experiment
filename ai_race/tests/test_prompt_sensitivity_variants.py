from __future__ import annotations

from collections import Counter
import hashlib
from pathlib import Path
import re

import pytest

from ai_race.dataio.config_loader import load_json
from ai_race.paths import CONFIGS_DIR, PROMPTS_DIR
from ai_race.prompts.sensitivity import (
    SENSITIVITY_PROTOCOL,
    VARIANTS,
    apply_prompt_variant,
)
from ai_race.runner.run_experiment import build_games_for_model


CANONICAL = (PROMPTS_DIR / "ai_race_en.txt").read_text(encoding="utf-8")
MECHANIC_PLACEHOLDERS = {
    "step1", "step2", "minRounds", "stopProbabilityPercent", "currentRound",
    "weight1", "weight2", "weight3", "weight4", "racePrize", "tiePrize",
    "maxPrivateRiskPercent", "ownStagePayoff", "opponentStagePayoff",
    "ownPrivateRiskPercent", "opponentPrivateRiskPercent", "ownProgress",
    "opponentProgress", "progressGap", "history",
}


def _placeholders(template: str) -> Counter[str]:
    return Counter(re.findall(r"\{([A-Za-z0-9_]+)\}", template))


@pytest.mark.parametrize("variant_id", sorted(VARIANTS))
def test_variant_preserves_mechanic_information_and_renders(variant_id: str):
    transformed = apply_prompt_variant(CANONICAL, variant_id)
    placeholders = _placeholders(transformed)
    assert MECHANIC_PLACEHOLDERS.issubset(placeholders)
    assert placeholders["strategy1"] >= 1
    assert placeholders["strategy2"] >= 1
    assert "ACTION:" in transformed

    experiment = load_json(CONFIGS_DIR / "experiment" / "baseline.json")
    experiment.update(repetitions=1, models=["Mock"], promptVariant=variant_id)
    game = build_games_for_model(experiment, "Mock")[0]
    prompt = game.build_round_prompts()[0]
    assert "{strategy" not in prompt
    assert "ACTION:" in prompt
    assert game.config.safe_progress == 1.0
    assert game.config.unsafe_progress == 1.5
    if variant_id == "canonical":
        assert game.config.prompt_version == "ai-race-fairgame-v3"
        assert "__prompt-" not in game.game_id
    else:
        assert game.config.prompt_version == f"{SENSITIVITY_PROTOCOL}:{variant_id}"
        assert f"__prompt-{variant_id}__" in game.game_id


def test_all_surface_variants_are_distinct_and_canonical_is_frozen():
    hashes = {
        variant_id: hashlib.sha256(
            apply_prompt_variant(CANONICAL, variant_id).encode("utf-8")
        ).hexdigest()
        for variant_id in VARIANTS
    }
    assert len(set(hashes.values())) == len(hashes)
    assert apply_prompt_variant(CANONICAL, "canonical") == CANONICAL


def test_unknown_variant_fails_closed():
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        apply_prompt_variant(CANONICAL, "not-a-real-treatment")


def test_transform_fails_closed_if_frozen_source_changes():
    altered = CANONICAL.replace("following choices", "available moves")
    with pytest.raises(ValueError, match="expected one source fragment"):
        apply_prompt_variant(altered, "order_actions_reversed")
