"""Regressions for defects found reviewing the review-hardening work.

Each test here covers a path the previous suite exercised only in pieces: it
tested that ordered splits could be built but never ran an ordered PLS, tested
fully qualitative pivots but never mixed values inside one test, and tested
mapping-profile parsing but never that the parsed profile reached the widgets.
"""

import numpy as np
import pandas as pd
import pytest

from analysis.advanced_methods import (
    get_xgboost_dependency_status,
    run_catboost_shap,
    run_xgboost_shap,
)
from analysis.aggregation import (
    aggregate_duplicate_batch_rows,
    pivot_long_to_wide,
    select_numeric_test_names,
)
from analysis.data_prep import get_intake_warnings, merge_process_and_qc_data
from analysis.mapping import load_mapping_profile, mapping_profile_to_json
from analysis.methods import (
    calculate_group_permutation_importance,
    run_all_analyses,
    run_pls,
)
from analysis.normalization import make_internal_column_name
from ui.intake_panel import build_mapping_profile_widget_values


def build_dated_frame(row_count: int = 60, seed: int = 0) -> pd.DataFrame:
    generator = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "batch_id": [f"B{index:03d}" for index in range(row_count)],
            "run_date": pd.date_range("2024-01-01", periods=row_count, freq="D"),
            "temperature_C": generator.normal(37.0, 1.0, row_count),
            "duration_hours": generator.normal(168.0, 5.0, row_count),
            "grade": generator.choice(["A", "B", "C"], row_count),
            "yield_g_L": generator.normal(100.0, 8.0, row_count),
        }
    )


# --- F1: ordered PLS validation actually runs --------------------------------

def test_ordered_pls_validation_runs_end_to_end():
    """choose_pls_component_count's signature changed; this call site was missed."""
    frame = build_dated_frame()

    result = run_pls(
        frame,
        process_columns=["temperature_C", "duration_hours", "grade"],
        outcome_column="yield_g_L",
        validation_order_column="run_date",
    )

    validation = result["time_ordered_validation"]
    assert validation["available"] is True
    assert validation["n_components"] >= 1
    assert np.isfinite(validation["test_r2"])


def test_full_analysis_run_survives_an_order_column():
    """The UI enables ordered validation automatically for date columns."""
    frame = build_dated_frame()

    analysis_results = run_all_analyses(
        frame,
        process_columns=["temperature_C", "duration_hours", "grade"],
        outcome_columns=["yield_g_L"],
        validation_order_column="run_date",
    )

    assert not analysis_results["ranked_drivers"].empty
    assert analysis_results["pls"]["yield_g_L"]["time_ordered_validation"]["available"]
    assert analysis_results["random_forest"]["yield_g_L"]["time_ordered_validation"]["available"]


# --- F2: mixed long-format results ------------------------------------------

def test_one_numeric_value_among_three_does_not_convert_a_test():
    """int(0.60 * 3) == 1, so a floored threshold erased Pass/Fail results."""
    long_frame = pd.DataFrame(
        {
            "batch_id": ["B1", "B2", "B3"],
            "test_name": ["appearance"] * 3,
            "result": ["10", "Pass", "Fail"],
        }
    )

    numeric_names, _ = select_numeric_test_names(long_frame, "test_name", "result")
    wide_frame = pivot_long_to_wide(long_frame, "batch_id", "test_name", "result", "first")

    assert numeric_names == set()
    assert wide_frame["appearance"].tolist() == ["10", "Pass", "Fail"]


def test_a_mostly_numeric_test_still_converts_but_reports_the_loss():
    long_frame = pd.DataFrame(
        {
            "batch_id": ["B1", "B2", "B3", "B4"],
            "test_name": ["assay"] * 4,
            "result": ["10", "11", "12", "Pass"],
        }
    )

    wide_frame = pivot_long_to_wide(long_frame, "batch_id", "test_name", "result", "first")

    assert wide_frame["assay"].tolist()[:3] == pytest.approx([10.0, 11.0, 12.0])
    assert pd.isna(wide_frame["assay"].tolist()[3])
    assert any("non-numeric" in warning for warning in get_intake_warnings(wide_frame))


@pytest.mark.parametrize(
    ("values", "expect_numeric"),
    [
        (["1", "2", "3"], True),
        (["1", "2", "x"], True),
        (["1", "x", "y"], False),
        (["x", "y", "z"], False),
    ],
)
def test_numeric_classification_uses_a_real_ratio(values, expect_numeric):
    long_frame = pd.DataFrame(
        {
            "batch_id": [f"B{index}" for index in range(len(values))],
            "test_name": ["t"] * len(values),
            "result": values,
        }
    )

    numeric_names, _ = select_numeric_test_names(long_frame, "test_name", "result")

    assert ("t" in numeric_names) is expect_numeric


# --- F3: model-hostile column and category names ----------------------------

HOSTILE_COLUMNS = ["temperature<limit", "grade"]


def build_hostile_frame(row_count: int = 60) -> pd.DataFrame:
    generator = np.random.default_rng(5)
    return pd.DataFrame(
        {
            "temperature<limit": generator.normal(37.0, 1.0, row_count),
            "grade": generator.choice(["A[1]", "B]2[", "C<3"], row_count),
            "yield_g_L": generator.normal(100.0, 8.0, row_count),
        }
    )


@pytest.mark.skipif(
    not get_xgboost_dependency_status()["available"],
    reason="xgboost is an optional extra",
)
def test_xgboost_accepts_columns_and_categories_it_cannot_name():
    """XGBoost rejects '[', ']' and '<' in feature names; users' columns have them."""
    result = run_xgboost_shap(build_hostile_frame(), HOSTILE_COLUMNS, "yield_g_L")

    assert result["available"] is True, result["reason"]
    assert set(result["variable_importance"]["process_variable"]) == set(HOSTILE_COLUMNS)


def test_catboost_accepts_the_same_hostile_names():
    result = run_catboost_shap(build_hostile_frame(), HOSTILE_COLUMNS, "yield_g_L")

    if not result["available"]:
        pytest.skip(result["reason"])
    assert set(result["variable_importance"]["process_variable"]) == set(HOSTILE_COLUMNS)


# --- F5: permutation importance must not see held-out preprocessing ---------

def test_group_permutation_fits_preprocessing_on_training_rows_only():
    """Imputation medians and the one-hot vocabulary must come from train rows."""
    row_count = 80
    generator = np.random.default_rng(2)
    frame = pd.DataFrame(
        {
            "temperature_C": generator.normal(37.0, 1.0, row_count),
            "grade": generator.choice(["A", "B"], row_count),
        }
    )
    frame.loc[frame.index[:15], "temperature_C"] = np.nan
    y = generator.normal(100.0, 8.0, row_count)

    importances, scoring_basis = calculate_group_permutation_importance(
        modeling_dataframe=frame,
        process_columns=["temperature_C", "grade"],
        y=y,
        repeats=2,
        n_estimators=40,
    )

    assert scoring_basis == "held-out rows"
    assert set(importances["process_variable"]) == {"temperature_C", "grade"}


def test_group_permutation_falls_back_for_small_datasets():
    generator = np.random.default_rng(4)
    frame = pd.DataFrame({"temperature_C": generator.normal(37.0, 1.0, 12)})

    _, scoring_basis = calculate_group_permutation_importance(
        modeling_dataframe=frame,
        process_columns=["temperature_C"],
        y=generator.normal(100.0, 8.0, 12),
        repeats=2,
        n_estimators=20,
    )

    assert "too few batches" in scoring_basis


# --- F6: a loaded mapping profile must reach the widgets --------------------

def build_round_tripped_profile():
    from analysis.mapping import build_mapping_profile

    profile = build_mapping_profile(
        process_file_name="process.xlsx",
        qc_file_name="qc.xlsx",
        process_intake_options={
            "sheet_name": "process_data",
            "header_row": 3,
            "parse_numeric_like_columns": False,
        },
        qc_intake_options={
            "sheet_name": "qc_results",
            "header_row": 1,
            "parse_numeric_like_columns": True,
        },
        process_batch_id_column="batch_id",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_percent"],
        process_duplicate_strategy="mean",
        qc_duplicate_strategy="keep_last",
    )
    return load_mapping_profile(
        mapping_profile_to_json(profile),
        process_columns=["batch_id", "temperature_C"],
        qc_columns=["batch_id", "yield_percent"],
    )


def test_loaded_profile_populates_every_intake_widget():
    """Only the batch-ID and outcome selections were being applied."""
    widget_values = build_mapping_profile_widget_values(build_round_tripped_profile())

    assert widget_values["process_duplicate_strategy"] == "mean"
    assert widget_values["qc_duplicate_strategy"] == "keep_last"
    assert widget_values["process_intake_sheet"] == "process_data"
    assert widget_values["process_intake_header_row"] == 3
    assert widget_values["process_intake_parse_numeric"] is False
    assert widget_values["qc_intake_header_row"] == 1
    assert widget_values["qc_intake_parse_numeric"] is True
    assert widget_values["selected_outcome_columns_widget"] == ["yield_percent"]


def test_profile_widget_keys_match_the_keys_the_controls_use():
    """A renamed widget key would silently stop the profile from applying."""
    from pathlib import Path

    project_root = Path(__file__).resolve().parents[1]
    app_source = (project_root / "app.py").read_text(encoding="utf-8")
    intake_source = (project_root / "ui" / "intake_panel.py").read_text(encoding="utf-8")

    widget_values = build_mapping_profile_widget_values(build_round_tripped_profile())
    for widget_key in widget_values:
        if widget_key.endswith(("_sheet", "_header_row", "_parse_numeric")):
            continue  # built from key_prefix at runtime
        assert (
            f'"{widget_key}"' in app_source or f'"{widget_key}"' in intake_source
        ), f"no control uses widget key {widget_key}"


# --- F7: helper columns must not overwrite user columns ---------------------

def test_helper_column_names_avoid_existing_columns():
    assert make_internal_column_name("__k", ["a", "b"]) == "__k"
    assert make_internal_column_name("__k", ["__k"]) == "__k_2"
    assert make_internal_column_name("__k", ["__k", "__k_2"]) == "__k_3"
    assert make_internal_column_name("__k", ["__k"], ["__k_2"]) == "__k_3"


@pytest.mark.parametrize("strategy", ["keep_first", "keep_last", "mean", "median"])
def test_duplicate_handling_keeps_a_user_column_named_like_its_helper(strategy):
    dataframe = pd.DataFrame(
        {
            "batch_id": ["B1", "B1", "B2"],
            "__normalized_batch_key": ["mine-1", "mine-2", "mine-3"],
            "__row_position": [9, 8, 7],
            "value": [1.0, 2.0, 3.0],
        }
    )

    result = aggregate_duplicate_batch_rows(dataframe, "batch_id", strategy, "Process")

    assert "__normalized_batch_key" in result.dataframe.columns
    assert "__row_position" in result.dataframe.columns
    assert "mine-3" in result.dataframe["__normalized_batch_key"].tolist()


def test_merge_keeps_a_user_column_named_like_the_merge_key():
    process_dataframe = pd.DataFrame(
        {
            "batch_id": ["B1", "B2"],
            "__batch_id_key": ["mine-1", "mine-2"],
            "temperature_C": [36.5, 37.1],
        }
    )
    qc_dataframe = pd.DataFrame({"batch_id": ["B1", "B2"], "yield_g_L": [10.0, 11.0]})

    merge_result = merge_process_and_qc_data(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="batch_id",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_g_L"],
    )

    assert merge_result.dataframe["__batch_id_key"].tolist() == ["mine-1", "mine-2"]
