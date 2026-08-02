from pathlib import Path

import pandas as pd

from results.scripts.analyze_impact_upgrade import (
    ROOT,
    context_from_prompt,
    divergence_curve,
    mapping_from_prompt,
)


def test_context_prompt_parser_preserves_skin_and_mapping() -> None:
    prompt = "ai-race-context-skin-v1:logistics_contract:opaque-pq-balanced-v1:safe_q"
    assert context_from_prompt(prompt) == "logistics_contract"
    assert mapping_from_prompt(prompt) == "safe_q"


def test_kaplan_meier_divergence_is_monotone_under_censoring() -> None:
    rows = pd.DataFrame(
        [
            {"context": "x", "n_rounds": 5, "first_divergence_round": 2},
            {"context": "x", "n_rounds": 3, "first_divergence_round": None},
            {"context": "x", "n_rounds": 8, "first_divergence_round": 6},
        ]
    )
    curve = divergence_curve(rows)
    values = curve["kaplan_meier_cumulative_divergence"].to_numpy()
    assert (values[1:] >= values[:-1]).all()
    assert values[0] == 0.0


def test_impact_outputs_pass_expected_coverage() -> None:
    output = Path(ROOT) / "results" / "impact_upgrade"
    quality = pd.read_json(output / "data_quality_audit.json", typ="series")
    assert quality["status"] == "passed"
    assert quality["n_turns"] == 13_680
    assert quality["trajectory_pairs"] == 1_344
    assert bool(quality["all_first_round_actions_agree"])

    mapping = pd.read_csv(output / "data" / "context_mapping_interaction.csv")
    safe_q = mapping[mapping["mapping"] == "safe_q"]
    assert (safe_q["ever_diverged_rate"] == 0).all()
    assert (safe_q["mean_unsafe_delta"] == 0).all()
