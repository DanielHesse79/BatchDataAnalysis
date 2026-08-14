"""Guards for the statistical claims the app makes about its own results."""

import numpy as np
import pandas as pd
import pytest

from analysis.methods import (
    SMALL_SAMPLE_ROW_COUNT,
    USABLE_VALIDATION_R2,
    aggregate_pls_variable_importance,
    assign_confidence,
    choose_pls_component_count,
    fit_feature_preprocessor,
    get_best_validation_r2,
    is_modelable_numeric_column,
    make_time_ordered_split,
    run_all_analyses,
    select_components_within_one_standard_error,
)
from analysis.profiling import profile_merged_data


def build_pure_noise_dataframe(row_count: int = 120, seed: int = 7) -> pd.DataFrame:
    """Random data with no relationship between any variable and the outcome."""
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "batch_id": [f"B{index:03d}" for index in range(row_count)],
            "temperature_C": generator.normal(37.0, 1.0, row_count),
            "duration_hours": generator.normal(168.0, 8.0, row_count),
            "feed_rate_mL_h": generator.normal(40.0, 5.0, row_count),
            "reactor_id": generator.choice(["RX-1", "RX-2", "RX-3"], row_count),
            "yield_g_L": generator.normal(100.0, 10.0, row_count),
        }
    )


def test_pure_noise_never_produces_a_confident_driver():
    """Scores are normalized by their own maximum, so something always ranks 1.0.

    Without a validation gate, a dataset with no signal at all still produced
    "high" confidence drivers backed by "PLS, Random Forest".
    """
    dataframe = build_pure_noise_dataframe()
    process_columns = [
        "temperature_C",
        "duration_hours",
        "feed_rate_mL_h",
        "reactor_id",
    ]

    analysis_results = run_all_analyses(
        dataframe,
        process_columns=process_columns,
        outcome_columns=["yield_g_L"],
    )

    confidence_labels = set(analysis_results["ranked_drivers"]["confidence"])
    assert confidence_labels == {"exploratory"}


def test_pure_noise_validation_scores_stay_below_the_usable_bar():
    dataframe = build_pure_noise_dataframe()
    analysis_results = run_all_analyses(
        dataframe,
        process_columns=["temperature_C", "duration_hours", "feed_rate_mL_h"],
        outcome_columns=["yield_g_L"],
    )

    best_validation_r2 = get_best_validation_r2(
        analysis_results["pls"]["yield_g_L"],
        analysis_results["random_forest"]["yield_g_L"],
    )

    assert not np.isfinite(best_validation_r2) or best_validation_r2 < USABLE_VALIDATION_R2


def test_planted_signal_still_earns_a_confident_label(synthetic_pipeline):
    """The gate must not suppress real, validated drivers."""
    ranked_drivers = synthetic_pipeline["analysis"]["ranked_drivers"]
    yield_drivers = ranked_drivers[ranked_drivers["outcome"] == "yield_g_L"]

    assert yield_drivers.iloc[0]["confidence"] == "high"


@pytest.mark.parametrize(
    ("best_validation_r2", "expected_confidence"),
    [
        (0.80, "high"),
        (USABLE_VALIDATION_R2 - 0.01, "exploratory"),
        (float("nan"), "exploratory"),
        (None, "exploratory"),
    ],
)
def test_confidence_requires_a_model_that_predicts_held_out_data(
    best_validation_r2,
    expected_confidence,
):
    assert (
        assign_confidence(
            combined_score=0.9,
            evidence_methods=["PLS", "Random Forest"],
            best_validation_r2=best_validation_r2,
            rows_used=200,
        )
        == expected_confidence
    )


def test_small_samples_are_capped_below_high_confidence():
    assert (
        assign_confidence(
            combined_score=0.9,
            evidence_methods=["PLS", "Random Forest"],
            best_validation_r2=0.8,
            rows_used=SMALL_SAMPLE_ROW_COUNT - 1,
        )
        == "medium"
    )


def test_cross_validation_fits_preprocessing_inside_each_fold():
    """Fitting the scaler on all rows before splitting leaks the test folds."""
    dataframe = build_pure_noise_dataframe(row_count=60)
    process_columns = ["temperature_C", "duration_hours", "feed_rate_mL_h"]

    _, cv_results = choose_pls_component_count(
        modeling_dataframe=dataframe,
        process_columns=process_columns,
        y=dataframe["yield_g_L"].to_numpy(dtype=float),
        max_components=3,
        cv_folds=5,
    )

    # Per-fold scores, not one pooled number, plus the spread needed to judge them.
    assert {"cv_r2", "cv_r2_std_error", "cv_fold_count"}.issubset(cv_results.columns)
    assert (cv_results["cv_fold_count"] > 1).all()
    # Honest CV on noise cannot look predictive.
    assert cv_results["cv_r2"].max() < USABLE_VALIDATION_R2


def test_component_count_prefers_the_simplest_model_within_one_standard_error():
    cv_results = pd.DataFrame(
        {
            "n_components": [1, 2, 3],
            "cv_r2": [0.60, 0.62, 0.63],
            "cv_r2_std_error": [0.05, 0.05, 0.05],
            "cv_fold_count": [5, 5, 5],
        }
    )

    assert select_components_within_one_standard_error(cv_results) == 1


def test_component_count_still_grows_for_a_clearly_better_model():
    cv_results = pd.DataFrame(
        {
            "n_components": [1, 2],
            "cv_r2": [0.10, 0.80],
            "cv_r2_std_error": [0.01, 0.01],
            "cv_fold_count": [5, 5],
        }
    )

    assert select_components_within_one_standard_error(cv_results) == 2


def test_pls_importance_does_not_reward_extra_category_levels():
    """Summing |coefficient| over one-hot levels ranked by cardinality, not signal."""
    feature_coefficients = pd.DataFrame(
        {
            "feature": ["temperature_C", "lot=A", "lot=B", "lot=C", "lot=D"],
            "process_variable": ["temperature_C", "lot", "lot", "lot", "lot"],
            "abs_coefficient": [0.50, 0.20, 0.20, 0.20, 0.20],
            "vip": [1.4, 0.5, 0.5, 0.5, 0.5],
        }
    )
    feature_groups = {"temperature_C": [0], "lot": [1, 2, 3, 4]}

    variable_importance = aggregate_pls_variable_importance(
        feature_coefficients=feature_coefficients,
        feature_groups=feature_groups,
    )

    top_variable = variable_importance.iloc[0]["process_variable"]
    assert top_variable == "temperature_C"


def test_numeric_columns_stored_as_text_are_not_one_hot_encoded():
    """A numeric column left as text became one category per distinct value."""
    dataframe = pd.DataFrame(
        {
            "temperature_C": ["36.8", "37.1", "37.4", "36.9"],
            "reactor_id": ["RX-1", "RX-2", "RX-1", "RX-2"],
        }
    )

    preprocessor = fit_feature_preprocessor(
        dataframe=dataframe,
        process_columns=["temperature_C", "reactor_id"],
        scale_all_features=False,
    )

    assert preprocessor.numeric_columns == ["temperature_C"]
    assert preprocessor.categorical_columns == ["reactor_id"]


@pytest.mark.parametrize(
    ("series", "expected"),
    [
        (pd.Series([1.0, 2.0]), True),
        (pd.Series(["1.0", "2.0"]), True),
        (pd.Series(["RX-1", "RX-2"]), False),
        (pd.Series([None, None], dtype="object"), False),
        (pd.Series(pd.to_datetime(["2024-01-01", "2024-01-02"])), False),
    ],
)
def test_numeric_column_detection(series, expected):
    assert is_modelable_numeric_column(series) is expected


def test_random_forest_permutation_is_scored_on_held_out_rows(synthetic_pipeline):
    """Permuting the training rows of an unconstrained forest measures memorization."""
    random_forest_result = synthetic_pipeline["analysis"]["random_forest"]["yield_g_L"]

    assert random_forest_result["permutation_scoring_basis"] == "held-out rows"


def test_date_validation_boundaries_are_readable_not_epoch_integers():
    """A date boundary was reported to the user as 1704067200000000000."""
    row_count = 40
    dataframe = pd.DataFrame(
        {
            "run_date": pd.date_range("2024-01-01", periods=row_count, freq="D"),
            "yield_g_L": np.linspace(10.0, 20.0, row_count),
        }
    )

    split = make_time_ordered_split(
        dataframe,
        outcome_column="yield_g_L",
        validation_order_column="run_date",
    )

    assert split["test_end_order_value"] == "2024-02-09"
    assert split["test_start_order_value"].startswith("2024-")


def test_profiling_and_modeling_agree_on_the_synthetic_columns(synthetic_pipeline):
    """Guards against the profile and the models disagreeing about column types."""
    profile_result = profile_merged_data(
        synthetic_pipeline["dataframe"],
        outcome_columns=synthetic_pipeline["outcomes"],
        matched_batch_count=synthetic_pipeline["merge"].matched_batch_count,
    )

    assert profile_result.process_columns
    assert not set(profile_result.process_columns).intersection(synthetic_pipeline["outcomes"])
