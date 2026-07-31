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
    # E5/E6 read the round-2+ table; the all-rounds table carries deliberately
    # different numbers so a test that passes cannot be reading the wrong one.
    contrast_tables = {
        "treatment_contrasts.csv": contrasts.assign(cohens_d=[9.0, -9.0]),
        "treatment_contrasts_round2plus.csv": contrasts,
    }
    player_metrics = pd.DataFrame(
        {
            "unsafe_rate": [0.6, 0.55, 0.62],
            "strategy_best": ["CAS", "CS", "AU"],
        }
    )
    return coefficients, contrast_tables, player_metrics


def test_human_comparison_scores_a_matching_llm_result_as_replicated():
    coefficients, contrast_tables, player_metrics = _comparison_inputs(0.61, 0.1)
    comparison, metadata = ANALYSER._build_human_comparison(
        coefficients=coefficients,
        contrast_tables=contrast_tables,
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
    coefficients, contrast_tables, player_metrics = _comparison_inputs(-0.61, 0.1)
    comparison, _ = ANALYSER._build_human_comparison(
        coefficients=coefficients,
        contrast_tables=contrast_tables,
        player_metrics=player_metrics,
        reference_path=ANALYSER.HUMAN_REFERENCE_PATH,
    )
    verdicts = dict(zip(comparison["effect_id"], comparison["verdict"]))
    assert verdicts["E1"] == "not_replicated"


def test_human_comparison_is_inconclusive_without_a_fitted_logit():
    _, contrast_tables, player_metrics = _comparison_inputs(0.61, 0.1)
    comparison, _ = ANALYSER._build_human_comparison(
        coefficients=None,
        contrast_tables=contrast_tables,
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


def _player_metrics_for_two_windows() -> pd.DataFrame:
    """Player metrics whose round-1 behaviour contradicts their later behaviour.

    The whole reason the paper reports two treatment tests is that they can
    disagree; a fixture where they agree would let a table mix-up pass.
    """

    rows = []
    for risk, all_rounds, later in ((0.1, 0.9, 0.1), (0.6, 0.2, 0.8)):
        for offset in (-0.05, 0.0, 0.05):
            rows.append(
                {
                    "model": "m",
                    "max_private_risk": risk,
                    "unsafe_rate": all_rounds + offset,
                    "later_unsafe_rate": later + offset,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_round2plus_contrasts_differ_from_the_all_round_contrasts():
    metrics = _player_metrics_for_two_windows()
    all_rounds = ANALYSER._pairwise_contrasts(
        metrics,
        strata=["model"],
        factor="max_private_risk",
        value="unsafe_rate",
    ).iloc[0]
    later = ANALYSER._pairwise_contrasts(
        metrics,
        strata=["model"],
        factor="max_private_risk",
        value="later_unsafe_rate",
    ).iloc[0]
    assert all_rounds["cohens_d"] > 0 and later["cohens_d"] < 0, (
        "the fixture is built so the two analysis windows disagree in sign"
    )
    assert all_rounds["cohens_d"] != pytest.approx(later["cohens_d"])


def test_human_comparison_scores_contrasts_on_the_round2plus_table():
    reference = json.loads(
        ANALYSER.HUMAN_REFERENCE_PATH.read_text(encoding="utf-8")
    )
    by_id = {effect["id"]: effect for effect in reference["effects"]}
    for effect_id in ("E5", "E6"):
        assert (
            by_id[effect_id]["contrast_table"]
            == "treatment_contrasts_round2plus.csv"
        ), "the human Cohen's d comes from the round-2+ pairwise table"

    coefficients, contrast_tables, player_metrics = _comparison_inputs(0.61, 0.1)
    comparison, _ = ANALYSER._build_human_comparison(
        coefficients=coefficients,
        contrast_tables=contrast_tables,
        player_metrics=player_metrics,
        reference_path=ANALYSER.HUMAN_REFERENCE_PATH,
    )
    scored = comparison.set_index("effect_id")
    # The all-rounds table carries d = 9.0 / -9.0; reading it would flip both.
    assert scored.loc["E5", "llm_value"] == pytest.approx(0.01)
    assert scored.loc["E6", "llm_value"] == pytest.approx(0.5)
    assert scored.loc["E5", "contrast_table"] == "treatment_contrasts_round2plus.csv"


def test_human_comparison_refuses_an_unknown_contrast_table(tmp_path):
    reference = json.loads(
        ANALYSER.HUMAN_REFERENCE_PATH.read_text(encoding="utf-8")
    )
    for effect in reference["effects"]:
        if effect["id"] == "E5":
            effect["contrast_table"] = "does_not_exist.csv"
    path = tmp_path / "human_reference.json"
    path.write_text(json.dumps(reference), encoding="utf-8")

    _, contrast_tables, player_metrics = _comparison_inputs(0.61, 0.1)
    with pytest.raises(ValueError, match="does not emit"):
        ANALYSER._build_human_comparison(
            coefficients=None,
            contrast_tables=contrast_tables,
            player_metrics=player_metrics,
            reference_path=path,
        )


def _sample_summary_frames():
    """Two cells, one of which loses a race to a parse failure."""

    context = {
        "model": "m",
        "persona_condition": "none",
        "prompt_version": "v3",
        "protocol_signature": "sig",
        "run_phase": "pilot",
        "run_status": "completed",
    }
    turn_rows, all_turn_rows, race_rows, player_rows = [], [], [], []
    for risk, n_races in ((0.1, 2), (0.6, 1)):
        for race in range(n_races):
            game_id = f"g{risk}_{race}"
            for seat in (0, 1):
                for round_number in (1, 2, 3):
                    row = {
                        **context,
                        "source_run": "run",
                        "max_private_risk": risk,
                        "game_id": game_id,
                        "player_id": f"{game_id}_p{seat}",
                        "round": round_number,
                        "parse_failed": False,
                        "retry_count": 0,
                    }
                    turn_rows.append(row)
                    all_turn_rows.append(row)
                player_rows.append(
                    {
                        **context,
                        "source_run": "run",
                        "max_private_risk": risk,
                        "game_id": game_id,
                        "player_id": f"{game_id}_p{seat}",
                        "unsafe_rate": 0.2 + 0.2 * seat,
                        "later_unsafe_rate": 0.1 + 0.2 * seat,
                    }
                )
            race_rows.append(
                {
                    **context,
                    "source_run": "run",
                    "max_private_risk": risk,
                    "game_id": game_id,
                    "n_rounds": 3,
                    "included_in_behavioral_estimands": True,
                }
            )
    # A contaminated race: recorded and counted for protocol health, excluded from
    # every behavioural number.
    for seat in (0, 1):
        all_turn_rows.append(
            {
                **context,
                "source_run": "run",
                "max_private_risk": 0.6,
                "game_id": "g_bad",
                "player_id": f"g_bad_p{seat}",
                "round": 1,
                "parse_failed": True,
                "retry_count": 1,
            }
        )
    race_rows.append(
        {
            **context,
            "source_run": "run",
            "max_private_risk": 0.6,
            "game_id": "g_bad",
            "n_rounds": 1,
            "included_in_behavioral_estimands": False,
        }
    )
    return (
        pd.DataFrame.from_records(turn_rows),
        pd.DataFrame.from_records(player_rows),
        pd.DataFrame.from_records(all_turn_rows),
        pd.DataFrame.from_records(race_rows),
    )


def test_sample_summary_has_one_row_per_analysis_cell():
    turns, player_metrics, all_turns, race_quality = _sample_summary_frames()
    summary = ANALYSER._build_sample_summary(
        turns,
        player_metrics,
        all_turns=all_turns,
        race_quality=race_quality,
    )
    assert len(summary) == 2
    assert summary[ANALYSER.CONTEXT].drop_duplicates().shape[0] == len(summary)
    assert summary["n_races"].sum() == 3, "behavioural races only"
    assert (
        summary["n_races_recorded"].sum() == len(race_quality)
    ), "recorded races must reconcile with race_quality.csv"


def test_sample_summary_separates_behavioural_and_protocol_denominators():
    turns, player_metrics, all_turns, race_quality = _sample_summary_frames()
    summary = ANALYSER._build_sample_summary(
        turns,
        player_metrics,
        all_turns=all_turns,
        race_quality=race_quality,
    ).set_index("max_private_risk")

    contaminated = summary.loc[0.6]
    assert contaminated["n_races"] == 1 and contaminated["n_races_excluded"] == 1
    # 6 clean decisions + 2 failed ones; the failure rate uses all eight.
    assert contaminated["n_decisions"] == 6
    assert contaminated["n_decisions_all_races"] == 8
    assert contaminated["parse_failure_rate"] == pytest.approx(0.25)
    assert summary.loc[0.1, "parse_failure_rate"] == pytest.approx(0.0)


def test_sample_summary_reports_the_median_the_theory_bridge_needs():
    turns, player_metrics, all_turns, race_quality = _sample_summary_frames()
    summary = ANALYSER._build_sample_summary(
        turns,
        player_metrics,
        all_turns=all_turns,
        race_quality=race_quality,
    )
    assert "median_phi_U" in summary, (
        "Figure 3B of the source paper matches the median, not the mean"
    )
    assert summary["median_phi_U"].iloc[0] == pytest.approx(0.3)
    assert summary["mean_n_rounds"].iloc[0] == pytest.approx(3.0)


def _jackknife_turns(*, contaminated_block: str | None) -> pd.DataFrame:
    """Blocks that agree on the opponent effect, except one that reverses it.

    The jackknife is only useful if removing the offending block visibly moves the
    coefficient, so the fixture builds exactly that situation and the test asserts
    the block is named.
    """

    rng = random.Random(7)
    rows = []
    for rep in range(12):
        block = f"m::rep{rep}"
        reversed_block = contaminated_block is not None and block == contaminated_block
        for risk in (0.1, 0.6, 0.9):
            for seat in (0, 1):
                for round_number in range(2, 8):
                    opponent_prev = float(rng.random() < 0.5)
                    # Elsewhere: opponent Unsafe pushes Unsafe up. In the
                    # contaminated block the association is inverted and much
                    # stronger, so it drags the pooled coefficient.
                    if reversed_block:
                        probability = 0.05 if opponent_prev else 0.95
                    else:
                        probability = 0.75 if opponent_prev else 0.35
                    rows.append(
                        {
                            "source_run": "run",
                            "game_id": f"g-{rep}-{risk}",
                            "player_id": f"g-{rep}-{risk}-p{seat}",
                            "round": round_number,
                            "valid_unsafe": float(rng.random() < probability),
                            "max_private_risk": risk,
                            "first_round_unsafe": float(rng.random() < 0.5),
                            "own_prev_unsafe": float(rng.random() < 0.5),
                            "opponent_prev_unsafe": opponent_prev,
                            "progress_gap_before": rng.choice([-1.0, -0.5, 0.0, 0.5]),
                            "retry_count": 0,
                            "randomization_block_id": block,
                            "prompt_version": "v3",
                            "model": "m",
                            "run_phase": "pilot",
                            "run_status": "completed",
                            "protocol_signature": "sig",
                            "persona_condition": "none",
                        }
                    )
    return pd.DataFrame.from_records(rows)


def test_jackknife_names_the_block_that_carries_a_coefficient(tmp_path):
    pytest.importorskip("statsmodels")
    turns = _jackknife_turns(contaminated_block="m::rep3")
    ANALYSER._fit_logit_robustness(turns, output_directory=tmp_path)
    table = pd.read_csv(tmp_path / "logit_robustness_jackknife.csv")
    row = table.loc[
        table["variant"].eq("full") & table["term"].eq("opponent_prev_unsafe")
    ].iloc[0]
    assert row["block_of_max_shift"] == "m::rep3"
    assert row["n_blocks_refitted"] == row["n_blocks"] == 12


def test_jackknife_reports_a_homogeneous_effect_as_stable(tmp_path):
    pytest.importorskip("statsmodels")
    turns = _jackknife_turns(contaminated_block=None)
    ANALYSER._fit_logit_robustness(turns, output_directory=tmp_path)
    table = pd.read_csv(tmp_path / "logit_robustness_jackknife.csv")
    row = table.loc[
        table["variant"].eq("full") & table["term"].eq("opponent_prev_unsafe")
    ].iloc[0]
    assert row["coefficient_full"] > 0 and bool(row["sign_stable"])
    # Removing one of twelve blocks cannot move a real effect by its own size.
    assert row["max_abs_shift"] < abs(row["coefficient_full"])


def test_jackknife_emits_every_exclusion_variant(tmp_path):
    pytest.importorskip("statsmodels")
    turns = _jackknife_turns(contaminated_block=None)
    ANALYSER._fit_logit_robustness(turns, output_directory=tmp_path)
    table = pd.read_csv(tmp_path / "logit_robustness_jackknife.csv")
    assert set(table["variant"]) == {
        "full",
        "exclude_retried_races",
        "exclude_min_horizon",
    }
    metadata = json.loads((tmp_path / "logit_robustness_metadata.json").read_text())
    assert "not an inferential test" in metadata["interpretation"]


def test_a_negligible_coefficient_is_not_reported_as_a_sign_flip():
    assert ANALYSER._tolerant_sign(1e-16) == 0
    assert ANALYSER._tolerant_sign(-1e-16) == 0
    assert ANALYSER._tolerant_sign(0.2) == 1
    assert ANALYSER._tolerant_sign(-0.2) == -1
    assert ANALYSER._tolerant_sign(float("nan")) == 0


def test_robustness_variants_drop_whole_races_not_single_rows():
    frame = pd.DataFrame(
        {
            "source_run": ["run"] * 10,
            "game_id": ["short"] * 4 + ["long"] * 4 + ["retried"] * 2,
            "round": [2, 3, 4, 5] + [2, 3, 4, 9] + [2, 3],
            "retry_count": [0] * 8 + [0, 2],
        }
    )
    variants = ANALYSER._robustness_variants(frame)
    assert set(variants["exclude_min_horizon"]["game_id"]) == {"long"}, (
        "the retried race also stops at the minimum, so both are dropped"
    )
    # One retried decision must remove the whole race, or the surviving lags would
    # point at rounds no longer in the sample.
    assert "retried" not in set(variants["exclude_retried_races"]["game_id"])
    assert variants["full"] is frame


def _theory_sample_summary() -> pd.DataFrame:
    """One row per analysis cell, two models, so model-independence is testable."""

    rows = []
    for model in ("m1", "m2"):
        for risk, median in ((0.1, 0.8), (0.6, 0.5), (0.9, 0.2)):
            rows.append(
                {
                    "model": model,
                    "max_private_risk": risk,
                    "persona_condition": "none",
                    "prompt_version": "v3",
                    "protocol_signature": "sig",
                    "run_phase": "pilot",
                    "run_status": "completed",
                    "n_players": 20,
                    "mean_phi_U": median + 0.02,
                    "median_phi_U": median,
                }
            )
    return pd.DataFrame.from_records(rows)


def test_theory_prediction_does_not_depend_on_the_model():
    """The column most likely to be misread as a fit. It has no model input."""

    table, _ = ANALYSER._build_theory_comparison(_theory_sample_summary())
    for (risk, point), group in table.groupby(
        ["max_private_risk", "parameter_point"], observed=True
    ):
        assert group["predicted_phi_U"].nunique() == 1, (
            f"prediction differs across models at {risk}/{point}"
        )
    assert set(table["model"]) == {"m1", "m2"}


def test_theory_comparison_uses_the_median_not_the_mean():
    summary = _theory_sample_summary()
    table, metadata = ANALYSER._build_theory_comparison(summary)
    row = table.loc[
        table["model"].eq("m1")
        & table["max_private_risk"].eq(0.1)
        & table["parameter_point"].eq("reference")
    ].iloc[0]
    assert row["observed_median_phi_U"] == pytest.approx(0.8)
    assert row["difference"] == pytest.approx(
        row["observed_median_phi_U"] - row["predicted_phi_U"]
    )
    assert "Figure 3B" in metadata["statistic"]


def test_theory_prediction_falls_with_the_risk_treatment():
    """The direction the evolutionary model gives: less Unsafe at higher risk."""

    table, _ = ANALYSER._build_theory_comparison(_theory_sample_summary())
    reference = (
        table.loc[table["parameter_point"].eq("reference")]
        .drop_duplicates("max_private_risk")
        .set_index("max_private_risk")["predicted_phi_U"]
    )
    assert reference[0.1] >= reference[0.6] > reference[0.9]


def test_theory_comparison_metadata_carries_both_warnings():
    _, metadata = ANALYSER._build_theory_comparison(_theory_sample_summary())
    assert "not a fit" in metadata["model_independence_warning"]
    assert "identical for every LLM" in metadata["model_independence_warning"]
    # The mutation rate is nominal, and the reader has to be told so.
    assert metadata["mutation_regime"] == "small_mutation_limit"
    assert "is not applied" in metadata["mutation_regime_caveat"]
    assert metadata["unmatched_risk_treatments"] == []


def test_theory_comparison_flags_a_treatment_with_no_configured_mechanism():
    summary = _theory_sample_summary()
    summary.loc[summary["max_private_risk"].eq(0.6), "max_private_risk"] = 0.42
    table, metadata = ANALYSER._build_theory_comparison(summary)
    assert metadata["unmatched_risk_treatments"] == [0.42]
    unmatched = table.loc[table["max_private_risk"].eq(0.42)]
    assert unmatched["predicted_phi_U"].isna().all(), (
        "an unconfigured treatment must produce no prediction rather than a "
        "prediction borrowed from another treatment"
    )
