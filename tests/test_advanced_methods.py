import numpy as np
import pandas as pd

import analysis.advanced_methods as advanced_methods
from analysis.advanced_methods import (
    build_catboost_variable_importance,
    prepare_catboost_dataframe,
    run_catboost_shap,
)


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

    prepared_frame, categorical_columns = prepare_catboost_dataframe(
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
