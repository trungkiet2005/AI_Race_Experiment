"""Test the analysis layer added for the LLM behaviour study.

These cover the pieces that are easy to get quietly wrong: the derived race-state
columns, the equivalence test that decides whether a *null* human result was
reproduced, and the mechanical scoring of the frozen human criteria. A wrong sign
or an off-by-one lag here would not crash anything — it would just produce a
plausible table.
"""
from __future__ import annotations

import importlib.util
import random
import json
import sys
from pathlib import Path

import pytest

pytest.importorskip("pandas", reason="the analyser needs the analysis extra")

import pandas as pd  # noqa: E402

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
ANALYSER_PATH = REPOSITORY_ROOT / "results" / "scripts" / "analyze_ai_race.py"


def _load_analyser():
    spec = importlib.util.spec_from_file_location("_analyze_estimands", ANALYSER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


ANALYSER = _load_analyser()


def _turns_frame(rows: list[dict]) -> pd.DataFrame:
    """Build the minimal turns frame that _add_dynamic_columns consumes."""

    frame = pd.DataFrame.from_records(rows)
    frame["source_run"] = "run"
    frame["parse_failed"] = frame.get("parse_failed", False)
    return frame


def _two_player_race(own_actions: list[int], opponent_actions: list[int]):
    rows = []
    for index, (own, opponent) in enumerate(zip(own_actions, opponent_actions), 1):
        # progress = round + 0.5 * unsafe_count, evaluated before this decision
        own_before = sum(own_actions[: index - 1])
        opponent_before = sum(opponent_actions[: index - 1])
        gap = 0.5 * (own_before - opponent_before)
        rows.append(
            {
                "game_id": "g1",
                "player_id": "P0",
                "opponent_id": "P1",
                "player_index": 0,
                "round": index,
                "unsafe": own,
                "progress_gap_before": gap,
            }
        )
        rows.append(
            {
                "game_id": "g1",
                "player_id": "P1",
                "opponent_id": "P0",
                "player_index": 1,
                "round": index,
                "unsafe": opponent,
                "progress_gap_before": -gap,
            }
        )
    return _turns_frame(rows)


def test_lags_and_race_state_track_the_actual_trajectory():
    #      round:      1  2  3  4
    own = [0, 1, 1, 0]
    opponent = [1, 1, 0, 0]
    result = ANALYSER._add_dynamic_columns(_two_player_race(own, opponent))
    focal = result.loc[result["player_id"].eq("P0")].sort_values("round")

    assert focal["own_prev_unsafe"].tolist()[1:] == [0.0, 1.0, 1.0]
    assert focal["opponent_prev_unsafe"].tolist()[1:] == [1.0, 1.0, 0.0]
    assert focal["first_round_unsafe"].unique().tolist() == [0.0]
    # Focal trails after round 1 (0 vs 1 Unsafe), draws level after round 3.
    assert focal["race_state"].tolist() == ["tied", "behind", "behind", "tied"]


def test_progress_gap_is_exactly_half_the_unsafe_count_difference():
    """The two predictors are one variable; the analyser must expose that."""

    result = ANALYSER._add_dynamic_columns(
        _two_player_race([1, 1, 0, 1], [0, 0, 1, 0])
    )
    implied = 0.5 * result["unsafe_count_diff_before"].astype(float)
    assert (result["progress_gap_before"].astype(float) - implied).abs().max() < 1e-9


def test_gap_bins_split_one_round_behind_from_well_behind():
    result = ANALYSER._add_dynamic_columns(
        _two_player_race([1, 1, 1, 0], [0, 0, 0, 0])
    )
    focal = result.loc[result["player_id"].eq("P0")].sort_values("round")
    assert focal["gap_bin"].tolist() == ["0.0", "+0.5", ">=+1.0", ">=+1.0"]
    trailing = result.loc[result["player_id"].eq("P1")].sort_values("round")
    assert trailing["gap_bin"].tolist() == ["0.0", "-0.5", "<=-1.0", "<=-1.0"]


def test_seat_index_is_recorded_for_the_seat_artefact_check():
    result = ANALYSER._add_dynamic_columns(_two_player_race([0, 1], [1, 0]))
    assert sorted(result["seat_index"].dropna().unique().tolist()) == [0.0, 1.0]


def test_welch_contrast_matches_a_known_separation():
    left = pd.Series([1.0, 1.1, 0.9, 1.0, 1.05])
    right = pd.Series([2.0, 2.1, 1.9, 2.0, 2.05])
    contrast = ANALYSER._welch_contrast(left, right)
    assert contrast["t"] < 0
    assert contrast["cohens_d"] < -2
    assert contrast["p_value"] < 0.01

    identical = ANALYSER._welch_contrast(left, left.copy())
    assert identical["t"] == pytest.approx(0.0)
    assert identical["cohens_d"] == pytest.approx(0.0)


def test_welch_contrast_refuses_a_single_observation():
    contrast = ANALYSER._welch_contrast(pd.Series([1.0]), pd.Series([2.0, 3.0]))
    assert contrast["n_left"] == 1
    assert pd.isna(contrast["t"])


def test_equivalence_needs_precision_not_just_a_small_estimate():
    """A small but imprecise estimate must not count as a reproduced null."""

    precise, precise_p = ANALYSER._tost_equivalent(0.02, 0.05, 0.3, 0.05)
    assert precise and precise_p < 0.05

    imprecise, imprecise_p = ANALYSER._tost_equivalent(0.02, 0.60, 0.3, 0.05)
    assert not imprecise and imprecise_p > 0.05

    large, _ = ANALYSER._tost_equivalent(0.9, 0.05, 0.3, 0.05)
    assert not large


def test_pairwise_contrasts_apply_within_stratum_bonferroni():
    frame = pd.DataFrame(
        {
            "model": ["m"] * 12,
            "max_private_risk": [0.1] * 4 + [0.6] * 4 + [0.9] * 4,
            "unsafe_rate": [0.9, 0.8, 0.85, 0.95, 0.2, 0.3, 0.25, 0.15, 0.2, 0.3, 0.25, 0.15],
        }
    )
    contrasts = ANALYSER._pairwise_contrasts(
        frame,
        strata=["model"],
        factor="max_private_risk",
        value="unsafe_rate",
    )
    assert len(contrasts) == 3
    assert set(contrasts["n_comparisons_in_stratum"]) == {3}
    assert (
        contrasts["p_value_bonferroni"] >= contrasts["p_value"]
    ).all()
    assert (contrasts["p_value_bonferroni"] <= 1.0).all()

    low_versus_high = contrasts.loc[
        contrasts["level_left"].eq(0.1) & contrasts["level_right"].eq(0.9)
    ].iloc[0]
    assert low_versus_high["cohens_d"] > 0


def test_human_reference_criteria_are_frozen_and_complete():
    reference = json.loads(
        ANALYSER.HUMAN_REFERENCE_PATH.read_text(encoding="utf-8")
    )
    identifiers = [effect["id"] for effect in reference["effects"]]
    assert identifiers == ["E1", "E2", "E3", "E4", "E5", "E6", "E7", "E8"]
    assert len(set(identifiers)) == len(identifiers)
    for effect in reference["effects"]:
        assert effect["test"] in {
            "directional",
            "equivalence",
            "directional_effect_size",
            "interval",
            "upper_bound",
        }
        if effect["test"] == "equivalence":
            assert effect["equivalence_bound"] > 0
    # The two human null results must be scored by equivalence, never by a
    # failure to reject.
    by_id = {effect["id"]: effect for effect in reference["effects"]}
    assert by_id["E4"]["test"] == "equivalence"
    assert by_id["E5"]["test"] == "equivalence"
    # E6's direction comes from Table S3, which contradicts the Figure 2A caption.
    assert by_id["E6"]["human_value"] > 0
    assert by_id["E6"]["expected_sign"] == "positive"


def _comparison_inputs(coefficient: float, standard_error: float):
    coefficients = pd.DataFrame(
        {
            "specification": ["6", "6", "6"],
            "term": [
                "opponent_prev_unsafe",
                "progress_gap_before",
                "first_round_unsafe",
            ],
            "coefficient": [coefficient, -0.4, 0.3],
            "cluster_robust_se": [standard_error, 0.1, 0.1],
            "p_value": [0.001, 0.01, 0.02],
        }
    )
    contrasts = pd.DataFrame(
        {
            "factor": ["max_private_risk", "max_private_risk"],
            "level_left": [0.6, 0.1],
            "level_right": [0.9, 0.6],
            "cohens_d": [0.01, 0.5],
        }
    )
    player_metrics = pd.DataFrame(
        {
            "unsafe_rate": [0.6, 0.55, 0.62],
            "strategy_best": ["CAS", "CS", "AU"],
        }
    )
    return coefficients, contrasts, player_metrics


def test_human_comparison_scores_a_matching_llm_result_as_replicated():
    coefficients, contrasts, player_metrics = _comparison_inputs(0.61, 0.1)
    comparison, metadata = ANALYSER._build_human_comparison(
        coefficients=coefficients,
        treatment_contrasts=contrasts,
        player_metrics=player_metrics,
        reference_path=ANALYSER.HUMAN_REFERENCE_PATH,
    )
    verdicts = dict(zip(comparison["effect_id"], comparison["verdict"]))
    assert verdicts["E1"] == "replicated"
    assert verdicts["E2"] == "replicated"
    assert verdicts["E3"] == "replicated"
    assert verdicts["E5"] == "replicated"
    assert verdicts["E6"] == "replicated"
    assert verdicts["E7"] == "replicated"
    assert verdicts["E8"] == "replicated"
    assert "pooling_warning" in metadata


def test_human_comparison_flags_a_reversed_sign():
    coefficients, contrasts, player_metrics = _comparison_inputs(-0.61, 0.1)
    comparison, _ = ANALYSER._build_human_comparison(
        coefficients=coefficients,
        treatment_contrasts=contrasts,
        player_metrics=player_metrics,
        reference_path=ANALYSER.HUMAN_REFERENCE_PATH,
    )
    verdicts = dict(zip(comparison["effect_id"], comparison["verdict"]))
    assert verdicts["E1"] == "not_replicated"


def test_human_comparison_is_inconclusive_without_a_fitted_logit():
    _, contrasts, player_metrics = _comparison_inputs(0.61, 0.1)
    comparison, _ = ANALYSER._build_human_comparison(
        coefficients=None,
        treatment_contrasts=contrasts,
        player_metrics=player_metrics,
        reference_path=ANALYSER.HUMAN_REFERENCE_PATH,
    )
    verdicts = dict(zip(comparison["effect_id"], comparison["verdict"]))
    for effect_id in ("E1", "E2", "E3", "E4"):
        assert verdicts[effect_id] == "inconclusive", (
            "a missing estimate must not be reported as a failed replication"
        )
    # Descriptive effects do not depend on the logit and stay scored.
    assert verdicts["E7"] == "replicated"


def test_logit_specifications_are_the_six_nested_paper_models():
    identifiers = [name for name, _ in ANALYSER.LOGIT_SPECIFICATIONS]
    assert identifiers == ["1", "2", "3", "4", "5", "6"]
    formulas = dict(ANALYSER.LOGIT_SPECIFICATIONS)
    assert formulas["6"] == ANALYSER.LOGIT_FORMULA
    assert "first_round_unsafe" not in formulas["1"]
    assert "first_round_unsafe" not in formulas["3"]
    assert "first_round_unsafe" in formulas["4"]
    assert "*" not in formulas["2"] and "*" not in formulas["5"]
    assert "*" in formulas["3"] and "*" in formulas["6"]


def test_persona_condition_stratifies_every_descriptive_table():
    assert "persona_condition" in ANALYSER.CONTEXT, (
        "without this, a persona run and the neutral baseline are averaged "
        "together in every table"
    )


def _formula_frame(signatures: list[str], personas: list[str]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "prompt_version": ["v3"] * len(personas),
            "model": ["m"] * len(personas),
            "run_phase": ["pilot"] * len(personas),
            "run_status": ["completed"] * len(personas),
            "protocol_signature": signatures,
            "persona_condition": personas,
        }
    )


def test_persona_control_is_dropped_when_the_signature_already_absorbs_it():
    """One run directory per persona makes the two sets of dummies identical."""

    frame = _formula_frame(["sigA", "sigA", "sigB", "sigB"], ["none", "none", "S_AA", "S_AA"])
    formula = ANALYSER._logit_formula_for(frame)
    assert "C(protocol_signature)" in formula
    assert "C(persona_condition)" not in formula, (
        "adding both would produce duplicate columns and a singular fit"
    )


def test_persona_confound_with_the_run_batch_is_detected():
    """Persona split across batches is unidentifiable, not merely uncontrolled."""

    confounded = ANALYSER._persona_identification(
        _formula_frame(["sigA", "sigA", "sigB", "sigB"], ["none", "none", "S_AA", "S_AA"])
    )
    assert confounded["confounded_with_protocol_signature"]
    assert not confounded["identified"]
    assert confounded["persona_conditions"] == ["S_AA", "none"]
    assert "one batch" in confounded["remedy"]

    identified = ANALYSER._persona_identification(
        _formula_frame(["sig"] * 4, ["none", "none", "S_AA", "S_AA"])
    )
    assert identified["identified"]
    assert not identified["confounded_with_protocol_signature"]

    single = ANALYSER._persona_identification(_formula_frame(["sig"] * 2, ["none"] * 2))
    assert not single["identified"]
    assert not single["confounded_with_protocol_signature"], (
        "one condition is nothing to identify, not a confound"
    )


def test_persona_control_is_kept_when_it_varies_inside_one_signature():
    frame = _formula_frame(["sig", "sig", "sig", "sig"], ["none", "none", "S_AA", "S_AA"])
    formula = ANALYSER._logit_formula_for(frame)
    assert "C(persona_condition)" in formula, (
        "pooling persona cells without a control loads the persona effect onto "
        "the treatment and lagged-action coefficients"
    )


def test_no_persona_control_when_only_one_condition_is_present():
    frame = _formula_frame(["sig"] * 3, ["none"] * 3)
    assert "C(persona_condition)" not in ANALYSER._logit_formula_for(frame)


def test_unlabelled_persona_runs_are_refused_by_default():
    races = pd.DataFrame({"source_run": ["run"], "game_id": ["g1"]})
    turns = races.copy()
    players = races.copy()
    with pytest.raises(ValueError, match="persona_condition is missing"):
        ANALYSER._resolve_persona_conditions(
            turns,
            races,
            players,
            allow_missing_persona_condition=False,
        )

    *_, conditions = ANALYSER._resolve_persona_conditions(
        turns,
        races,
        players,
        allow_missing_persona_condition=True,
    )
    assert conditions == [ANALYSER.MISSING_PERSONA_CONDITION], (
        "an audit override must keep unlabelled races in their own stratum "
        "rather than folding them into the neutral baseline"
    )


def test_conflicting_persona_labels_within_one_race_are_fatal():
    races = pd.DataFrame({"source_run": ["run"], "game_id": ["g1"]})
    turns = pd.DataFrame(
        {
            "source_run": ["run", "run"],
            "game_id": ["g1", "g1"],
            "persona_condition": ["none", "S_AA"],
        }
    )
    with pytest.raises(ValueError, match="persona_condition conflicts"):
        ANALYSER._resolve_persona_conditions(
            turns,
            races,
            races.copy(),
            allow_missing_persona_condition=True,
        )


def _logit_ready_turns(signatures: list[str], personas: list[str]) -> pd.DataFrame:
    """A minimal frame that satisfies every precondition of the logit fitter."""

    # Outcomes are drawn rather than derived from the predictors: a deterministic
    # outcome would separate perfectly and the fit would fail for that reason
    # instead of the one under test.
    rng = random.Random(11)
    rows = []
    for index, (signature, persona) in enumerate(zip(signatures, personas)):
        for rep in range(8):
            for risk in (0.1, 0.6, 0.9):
                for round_number in (2, 3, 4):
                    rows.append(
                        {
                            "source_run": signature,
                            "game_id": f"{signature}-{rep}-{risk}",
                            "player_id": f"P{index}",
                            "round": round_number,
                            "valid_unsafe": float(rng.random() < 0.5),
                            "max_private_risk": risk,
                            "first_round_unsafe": float(rng.random() < 0.5),
                            "own_prev_unsafe": float(rng.random() < 0.5),
                            "opponent_prev_unsafe": float(rng.random() < 0.5),
                            "progress_gap_before": rng.choice([-1.0, -0.5, 0.0, 0.5]),
                            "randomization_block_id": f"m::rep{rep}",
                            "prompt_version": "v3",
                            "model": "m",
                            "run_phase": "pilot",
                            "run_status": "completed",
                            "protocol_signature": signature,
                            "persona_condition": persona,
                        }
                    )
    return pd.DataFrame.from_records(rows)


def test_logit_refuses_a_persona_confound_in_primary_mode(tmp_path):
    turns = _logit_ready_turns(["sigA", "sigB"], ["none", "S_AA"])
    with pytest.raises(ValueError, match="perfectly confounded with the run batch"):
        ANALYSER._fit_clustered_logit(
            turns,
            output_directory=tmp_path,
            allow_mixed_protocols=False,
        )


def test_logit_warns_but_proceeds_for_a_labelled_audit(tmp_path, capsys):
    turns = _logit_ready_turns(["sigA", "sigB"], ["none", "S_AA"])
    written = ANALYSER._fit_clustered_logit(
        turns,
        output_directory=tmp_path,
        allow_mixed_protocols=True,
    )
    assert "clustered_logit_coefficients.csv" in written
    assert "perfectly confounded" in capsys.readouterr().err

    metadata = json.loads((tmp_path / "clustered_logit_metadata.json").read_text())
    assert metadata["persona_identification"]["identified"] is False, (
        "the audit output must state that persona is not estimable here"
    )


def test_shared_base_seed_widens_the_crn_block_only_when_verifiable():
    shared = pd.DataFrame({"manifest_base_seed": [260726, 260726, 260726]})
    assert ANALYSER._shared_base_seed(shared)

    mixed = pd.DataFrame({"manifest_base_seed": [260726, 99]})
    assert not ANALYSER._shared_base_seed(mixed)

    partial = pd.DataFrame({"manifest_base_seed": [260726, None]})
    assert not ANALYSER._shared_base_seed(partial)

    assert not ANALYSER._shared_base_seed(pd.DataFrame({"other": [1]}))
