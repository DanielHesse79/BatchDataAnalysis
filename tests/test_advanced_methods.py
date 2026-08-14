import numpy as np
import pandas as pd

import analysis.advanced_methods as advanced_methods
from analysis.advanced_methods import (
    EMPTY_XGBOOST_RESULT_TABLES,
    build_catboost_variable_importance,
    build_long_shap_values,
    build_xgboost_variable_importance,
    prepare_catboost_dataframe,
    run_catboost_shap,
    run_xgboost_shap,
)


def test_test_rows_are_imputed_with_training_medians():
    """Imputing a test set from its own median leaks the held-out rows."""
    train_dataframe = pd.DataFrame({"temperature_C": [36.0, 36.0, 36.0, 36.0]})
    test_dataframe = pd.DataFrame({"temperature_C": [50.0, None]})

    _, _, train_medians = prepare_catboost_dataframe(
        train_dataframe,
        process_columns=["temperature_C"],
    )
    prepared_test, _, _ = prepare_catboost_dataframe(
        test_dataframe,
        process_columns=["temperature_C"],
        fitted_medians=train_medians,
    )

    assert train_medians["temperature_C"] == 36.0
    # Without the fitted median this would be 50.0, the test set's own median.
    assert prepared_test["temperature_C"].tolist() == [50.0, 36.0]


def test_long_shap_table_never_truncates_inside_a_batch():
    """A batch cut mid-way reads as a real zero contribution for its other variables."""
    process_columns = ["temperature_C", "duration_hours", "feed_rate_mL_h"]
    batch_count = 10
    shap_values = np.arange(batch_count * len(process_columns), dtype=float).reshape(
        batch_count,
        len(process_columns),
    )

    long_values = build_long_shap_values(
        source_index=pd.Index([f"B{index}" for index in range(batch_count)]),
        process_columns=process_columns,
        shap_values=shap_values,
        max_rows=7,
    )

    variables_per_batch = long_values.groupby("source_index")["process_variable"].count()
    assert set(variables_per_batch) == {len(process_columns)}
    assert len(long_values) <= 7 + len(process_columns)


def test_long_shap_table_handles_no_process_columns():
    empty_values = build_long_shap_values(
        source_index=pd.Index([]),
        process_columns=[],
        shap_values=np.empty((0, 0)),
    )

    assert empty_values.empty
    assert list(empty_values.columns) == ["source_index", "process_variable", "shap_value"]


def test_catboost_dependency_status_reports_missing_package(monkeypatch):
    monkeypatch.setattr(advanced_methods, "find_spec", lambda package_name: None)

    status = advanced_methods.get_catboost_shap_dependency_status()

    assert status["available"] is False
    assert status["missing_packages"] == ["catboost"]


def test_run_catboost_shap_returns_stable_unavailable_shape_when_missing(monkeypatch):
    monkeypatch.setattr(advanced_methods, "find_spec", lambda package_name: None)
    dataframe = pd.DataFrame(
        {
            "temperature_C": [36.8, 37.0, 37.2],
            "reactor_id": ["R1", "R2", "R1"],
            "yield_g_L": [4.1, 4.2, 4.5],
        }
    )

    result = run_catboost_shap(
        dataframe=dataframe,
        process_columns=["temperature_C", "reactor_id"],
        outcome_column="yield_g_L",
    )

    assert result["available"] is False
    assert result["variable_importance"].empty
    assert result["shap_values"].empty


def test_prepare_catboost_dataframe_imputes_numeric_and_categorical_values():
    dataframe = pd.DataFrame(
        {
            "temperature_C": [36.8, None, 37.2],
            "reactor_id": ["R1", None, "R2"],
        }
    )

    prepared_frame, categorical_columns, _ = prepare_catboost_dataframe(
        dataframe,
        process_columns=["temperature_C", "reactor_id"],
    )

    assert categorical_columns == ["reactor_id"]
    assert prepared_frame["temperature_C"].isna().sum() == 0
    assert prepared_frame["reactor_id"].tolist() == ["R1", "Missing", "R2"]


def test_catboost_variable_importance_combines_prediction_change_and_shap_scores():
    importance = build_catboost_variable_importance(
        process_columns=["temperature_C", "reactor_id"],
        prediction_values_change=np.array([20.0, 80.0]),
        shap_values=np.array([[0.1, 1.0], [0.2, -1.5], [0.3, 0.5]]),
    )

    assert importance.iloc[0]["process_variable"] == "reactor_id"
    assert importance.iloc[0]["catboost_score"] > importance.iloc[1]["catboost_score"]


def test_xgboost_dependency_status_reports_missing_package(monkeypatch):
    monkeypatch.setattr(advanced_methods, "find_spec", lambda package_name: None)

    status = advanced_methods.get_xgboost_dependency_status()

    assert status["available"] is False
    assert status["missing_packages"] == ["xgboost"]


def test_run_xgboost_shap_returns_stable_unavailable_shape_when_missing(monkeypatch):
    monkeypatch.setattr(advanced_methods, "find_spec", lambda package_name: None)
    dataframe = pd.DataFrame(
        {
            "temperature_C": [36.8, 37.0, 37.2],
            "reactor_id": ["R1", "R2", "R1"],
            "yield_g_L": [4.1, 4.2, 4.5],
        }
    )

    result = run_xgboost_shap(
        dataframe=dataframe,
        process_columns=["temperature_C", "reactor_id"],
        outcome_column="yield_g_L",
    )

    assert result["available"] is False
    assert result["variable_importance"].empty
    assert result["shap_values"].empty
    assert list(result["variable_importance"].columns) == (
        EMPTY_XGBOOST_RESULT_TABLES["variable_importance"]
    )


def test_xgboost_variable_importance_collapses_encoded_columns_to_variables():
    # temperature_C is one numeric column (index 0); reactor_id is one-hot encoded
    # across two columns (indices 1 and 2). The reactor signal is stronger.
    importance = build_xgboost_variable_importance(
        feature_groups={"temperature_C": [0], "reactor_id": [1, 2]},
        model_importances=np.array([0.1, 0.4, 0.4]),
        shap_values=np.array([[0.1, 1.0, 0.8], [0.2, -1.5, -0.9], [0.3, 0.5, 0.6]]),
    )

    assert list(importance.columns) == (
        EMPTY_XGBOOST_RESULT_TABLES["variable_importance"][:-1]
        + ["gain_score", "mean_abs_shap_score", "xgboost_score"]
    )
    assert importance.iloc[0]["process_variable"] == "reactor_id"
    assert importance.iloc[0]["xgboost_score"] > importance.iloc[1]["xgboost_score"]
