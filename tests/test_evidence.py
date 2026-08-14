import numpy as np
import pandas as pd
import pytest

from analysis.evidence import (
    build_categorical_level_effects,
    build_operating_window_hints,
    classify_correlation_direction,
    dataframe_to_records,
    infer_outcome_objective,
    summarize_refined_response_window,
)
from analysis.profiling import profile_merged_data


def build_single_driver_ranking(outcome: str, process_variable: str) -> pd.DataFrame:
    """Build a minimal ranked-driver table for one outcome/variable pair."""
    return pd.DataFrame(
        [
            {
                "outcome": outcome,
                "rank": 1,
                "process_variable": process_variable,
                "combined_score": 0.9,
                "confidence": "high",
                "evidence_methods": "PLS, Random Forest",
            }
        ]
    )


def test_operating_window_hints_preserve_middle_sweet_spots():
    sodium_hydroxide = np.linspace(76.0, 124.0, 96)
    yield_percent = 82.0 - 0.035 * (sodium_hydroxide - 101.0) ** 2
    dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{i:03d}" for i in range(96)],
            "sodium_hydroxide_g": sodium_hydroxide,
            "yield_percent": yield_percent,
        }
    )
    profile_result = profile_merged_data(
        dataframe,
        outcome_columns=["yield_percent"],
        batch_id_column="batch_id",
    )
    ranked_drivers = pd.DataFrame(
        [
            {
                "outcome": "yield_percent",
                "rank": 1,
                "process_variable": "sodium_hydroxide_g",
                "combined_score": 0.9,
                "confidence": "high",
                "evidence_methods": "PLS, Random Forest",
            }
        ]
    )

    hints = build_operating_window_hints(
        ranked_drivers=ranked_drivers,
        merged_dataframe=dataframe,
        profile_result=profile_result,
        outcomes=["yield_percent"],
    )

    assert len(hints) == 1
    hint = hints.iloc[0]
    assert hint["pattern_type"] == "middle_sweet_spot"
    assert 88.0 <= hint["range_min"] <= 101.0
    assert 101.0 <= hint["range_max"] <= 113.0
    assert pd.notna(hint["refined_range_label"])
    assert hint["refined_pattern_type"] == "middle_best_bin"


def test_outcome_objective_matches_keywords_as_whole_tokens():
    """"gloss_units" contains the substring "loss" but is not a loss measurement."""
    assert infer_outcome_objective("yield_g_L") == "maximize"
    assert infer_outcome_objective("hcp_ppm") == "minimize"
    assert infer_outcome_objective("color_index") == "minimize"
    assert infer_outcome_objective("gloss_units") == "unknown"


def test_outcome_objective_is_unknown_when_both_keyword_lists_match():
    """"residual_activity" used to match "residual" and be optimized in the wrong direction."""
    assert infer_outcome_objective("residual_activity") == "unknown"
    assert infer_outcome_objective("residualActivity") == "unknown"


def test_outcome_objective_accepts_an_explicit_override():
    assert (
        infer_outcome_objective(
            "residual_activity",
            outcome_objectives={"residual_activity": "maximize"},
        )
        == "maximize"
    )


def build_monotonic_activity_frame() -> tuple[pd.DataFrame, object]:
    """Build data where residual activity rises with a process variable."""
    incubation_time_hours = np.linspace(2.0, 26.0, 120)
    dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{index:03d}" for index in range(120)],
            "incubation_time_hours": incubation_time_hours,
            "residual_activity": 40.0 + 1.5 * incubation_time_hours,
        }
    )
    profile_result = profile_merged_data(
        dataframe,
        outcome_columns=["residual_activity"],
        batch_id_column="batch_id",
    )
    return dataframe, profile_result


def test_ambiguous_outcome_name_suppresses_the_response_band():
    """A wrongly inferred direction would publish a band aimed at minimizing an outcome to be maximized."""
    dataframe, profile_result = build_monotonic_activity_frame()

    hints = build_operating_window_hints(
        ranked_drivers=build_single_driver_ranking("residual_activity", "incubation_time_hours"),
        merged_dataframe=dataframe,
        profile_result=profile_result,
        outcomes=["residual_activity"],
    )

    assert hints.empty


def test_outcome_objective_override_restores_the_response_band():
    dataframe, profile_result = build_monotonic_activity_frame()

    hints = build_operating_window_hints(
        ranked_drivers=build_single_driver_ranking("residual_activity", "incubation_time_hours"),
        merged_dataframe=dataframe,
        profile_result=profile_result,
        outcomes=["residual_activity"],
        outcome_objectives={"residual_activity": "maximize"},
    )

    assert len(hints) == 1
    hint = hints.iloc[0]
    assert hint["objective"] == "maximize outcome"
    assert hint["range_min"] > 14.0


def build_thin_winning_bin_data(best_bin_outcome: float) -> pd.DataFrame:
    """Build data whose quantile edges collapse into one bin holding only two rows."""
    variable_values = list(range(1, 11)) + [20.0] * 18 + list(range(21, 33))
    # The (20.0, 22.25] bin holds only the values 21 and 22 after edge collapse.
    outcome_values = [
        best_bin_outcome if 20.0 < value <= 22.25 else 1.0 for value in variable_values
    ]
    return pd.DataFrame(
        {
            "variable_value": pd.Series(variable_values, dtype=float),
            "outcome_value": pd.Series(outcome_values, dtype=float),
        }
    )


def test_refined_window_is_suppressed_when_the_winning_bin_holds_too_few_batches():
    """Collapsed quantile edges can leave a two-batch bin whose 'best' mean is pure noise."""
    usable_data = build_thin_winning_bin_data(best_bin_outcome=99.0)

    refined_window = summarize_refined_response_window(
        variable="reaction_time_min",
        objective="maximize",
        usable_data=usable_data,
        lower_bound=1.0,
        upper_bound=32.0,
    )

    assert refined_window is None


def test_refined_window_is_still_reported_when_a_well_filled_bin_wins():
    """Guards against the thin-bin test passing for the wrong reason."""
    usable_data = build_thin_winning_bin_data(best_bin_outcome=0.0)
    usable_data.loc[usable_data["variable_value"] == 20.0, "outcome_value"] = 99.0

    refined_window = summarize_refined_response_window(
        variable="reaction_time_min",
        objective="maximize",
        usable_data=usable_data,
        lower_bound=1.0,
        upper_bound=32.0,
    )

    assert refined_window is not None
    assert refined_window["refined_count"] == 18


def test_refined_window_reports_a_standard_error_beside_the_lift():
    """The lift is a max over noisy bin means, so the caveat has to be quantitative."""
    sodium_hydroxide = np.linspace(76.0, 124.0, 96)
    yield_percent = 82.0 - 0.035 * (sodium_hydroxide - 101.0) ** 2
    dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{index:03d}" for index in range(96)],
            "sodium_hydroxide_g": sodium_hydroxide,
            "yield_percent": yield_percent,
        }
    )
    profile_result = profile_merged_data(
        dataframe,
        outcome_columns=["yield_percent"],
        batch_id_column="batch_id",
    )

    hints = build_operating_window_hints(
        ranked_drivers=build_single_driver_ranking("yield_percent", "sodium_hydroxide_g"),
        merged_dataframe=dataframe,
        profile_result=profile_result,
        outcomes=["yield_percent"],
    )
    hint = hints.iloc[0]

    assert hint["refined_count"] >= 8
    assert hint["refined_directional_lift_standard_error"] > 0
    assert hint["refined_directional_lift_z_score"] == pytest.approx(
        hint["refined_directional_lift_vs_other_bins"]
        / hint["refined_directional_lift_standard_error"],
        rel=1e-2,
    )


def test_categorical_level_deltas_decompose_against_the_reported_baseline():
    """Rows missing the variable are excluded from level means, so they must leave the baseline too."""
    dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{index:03d}" for index in range(60)],
            "supplier": ["Supplier_A"] * 20 + ["Supplier_B"] * 20 + [None] * 20,
            "purity_percent": [100.0] * 20 + [80.0] * 20 + [10.0] * 20,
        }
    )
    profile_result = profile_merged_data(
        dataframe,
        outcome_columns=["purity_percent"],
        batch_id_column="batch_id",
    )

    effects = build_categorical_level_effects(
        merged_dataframe=dataframe,
        profile_result=profile_result,
        outcome="purity_percent",
        candidate_variables=["supplier"],
    )

    assert len(effects) == 1
    effect = effects[0]
    assert effect["overall_outcome_mean"] == pytest.approx(90.0)
    assert effect["overall_outcome_mean_all_rows"] == pytest.approx(63.333, abs=1e-3)
    assert effect["rows_used"] == 40
    assert effect["rows_excluded_missing_variable_or_outcome"] == 20

    weighted_delta_total = sum(
        level["count"] * level["delta_from_overall_mean"]
        for level in effect["largest_level_effects"]
    )
    assert weighted_delta_total == pytest.approx(0.0, abs=1e-6)


def test_weak_correlation_direction_label_is_not_doubled():
    assert classify_correlation_direction(0.0) == "weak_or_non_linear_signal"


def test_dataframe_to_records_treats_zero_max_rows_as_zero_rows():
    """`if max_rows` made an explicit 0 fall through to returning every row."""
    dataframe = pd.DataFrame({"value": [1, 2, 3]})

    assert dataframe_to_records(dataframe, max_rows=0) == []
    assert len(dataframe_to_records(dataframe, max_rows=2)) == 2
    assert len(dataframe_to_records(dataframe)) == 3
