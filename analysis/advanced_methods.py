"""Optional advanced model helpers.

CatBoost is useful for SME-style tabular data because it handles categorical
variables directly. This module is deliberately optional: the rest of the app
must keep working when CatBoost is not installed.
"""

from __future__ import annotations

from importlib.util import find_spec
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

from analysis.methods import (
    RANDOM_STATE,
    make_time_ordered_split,
    normalize_scores,
    safe_r2_score,
    split_without_indices,
)


EMPTY_CATBOOST_RESULT_TABLES = {
    "variable_importance": [
        "process_variable",
        "prediction_values_change",
        "mean_abs_shap",
        "mean_shap",
        "catboost_score",
    ],
    "shap_values": ["source_index", "process_variable", "shap_value"],
}


def get_catboost_shap_dependency_status() -> dict[str, Any]:
    """Return whether optional CatBoost explainability can run."""
    catboost_available = find_spec("catboost") is not None
    if catboost_available:
        return {
            "available": True,
            "missing_packages": [],
            "message": "CatBoost is installed. Native CatBoost SHAP values can be computed.",
        }
    return {
        "available": False,
        "missing_packages": ["catboost"],
        "message": (
            "CatBoost is not installed. Install it to enable the optional CatBoost + SHAP tab."
        ),
    }


def run_catboost_shap(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
    validation_order_column: str | None = None,
    iterations: int = 350,
    learning_rate: float = 0.04,
    depth: int = 5,
) -> dict[str, Any]:
    """Fit CatBoost and return native SHAP-style global explanations.

    The function returns an unavailable result instead of raising when CatBoost
    is missing. This keeps the feature safe for machines where the optional
    dependency has not been installed yet.
    """
    dependency_status = get_catboost_shap_dependency_status()
    if not dependency_status["available"]:
        return unavailable_catboost_result(outcome_column, dependency_status["message"])

    from catboost import CatBoostRegressor, Pool

    modeling_dataframe, y = prepare_modeling_frame(dataframe, process_columns, outcome_column)
    if len(modeling_dataframe) < 20:
        return unavailable_catboost_result(
            outcome_column,
            "CatBoost + SHAP needs at least 20 usable rows for this app workflow.",
        )

    prepared_frame, categorical_columns = prepare_catboost_dataframe(
        modeling_dataframe,
        process_columns,
    )
    cat_feature_indices = [
        prepared_frame.columns.get_loc(column_name) for column_name in categorical_columns
    ]
    train_pool = Pool(prepared_frame, y, cat_features=cat_feature_indices)

    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool)
    predictions = model.predict(train_pool)

    prediction_values_change = model.get_feature_importance(
        train_pool,
        type="PredictionValuesChange",
    )
    shap_matrix = model.get_feature_importance(train_pool, type="ShapValues")
    shap_values = np.asarray(shap_matrix[:, :-1], dtype=float)

    variable_importance = build_catboost_variable_importance(
        process_columns=process_columns,
        prediction_values_change=prediction_values_change,
        shap_values=shap_values,
    )
    time_ordered_validation = run_catboost_time_ordered_validation(
        dataframe=dataframe,
        process_columns=process_columns,
        outcome_column=outcome_column,
        validation_order_column=validation_order_column,
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
    )

    return {
        "available": True,
        "reason": "",
        "model": model,
        "outcome": outcome_column,
        "n_rows_used": int(len(y)),
        "numeric_feature_count": int(len(process_columns) - len(categorical_columns)),
        "categorical_feature_count": int(len(categorical_columns)),
        "training_r2": float(r2_score(y, predictions)),
        "time_ordered_validation": time_ordered_validation,
        "variable_importance": variable_importance,
        "shap_values": build_long_shap_values(
            source_index=modeling_dataframe.index,
            process_columns=process_columns,
            shap_values=shap_values,
        ),
    }


def unavailable_catboost_result(outcome_column: str, reason: str) -> dict[str, Any]:
    """Return a stable unavailable result shape."""
    return {
        "available": False,
        "reason": reason,
        "outcome": outcome_column,
        "n_rows_used": 0,
        "numeric_feature_count": 0,
        "categorical_feature_count": 0,
        "training_r2": np.nan,
        "time_ordered_validation": {"available": False, "reason": reason},
        "variable_importance": pd.DataFrame(
            columns=EMPTY_CATBOOST_RESULT_TABLES["variable_importance"]
        ),
        "shap_values": pd.DataFrame(columns=EMPTY_CATBOOST_RESULT_TABLES["shap_values"]),
    }


def prepare_modeling_frame(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Keep rows with a numeric outcome and selected process columns."""
    numeric_outcome = pd.to_numeric(dataframe[outcome_column], errors="coerce")
    usable_mask = numeric_outcome.notna()
    modeling_dataframe = dataframe.loc[usable_mask, process_columns].copy()
    y = numeric_outcome.loc[usable_mask].to_numpy(dtype=float)
    return modeling_dataframe, y


def prepare_catboost_dataframe(
    dataframe: pd.DataFrame,
    process_columns: list[str],
) -> tuple[pd.DataFrame, list[str]]:
    """Impute process columns in a CatBoost-friendly way."""
    prepared_frame = pd.DataFrame(index=dataframe.index)
    categorical_columns: list[str] = []

    for column_name in process_columns:
        series = dataframe[column_name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_values = pd.to_numeric(series, errors="coerce")
            median_value = numeric_values.median()
            if pd.isna(median_value):
                median_value = 0.0
            prepared_frame[column_name] = numeric_values.fillna(float(median_value))
        else:
            categorical_columns.append(column_name)
            prepared_frame[column_name] = (
                series.astype("object")
                .where(series.notna(), "Missing")
                .astype(str)
            )

    return prepared_frame[process_columns], categorical_columns


def build_catboost_variable_importance(
    process_columns: list[str],
    prediction_values_change: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Combine CatBoost importance and native SHAP values per process variable."""
    variable_importance = pd.DataFrame(
        {
            "process_variable": process_columns,
            "prediction_values_change": np.asarray(prediction_values_change, dtype=float),
            "mean_abs_shap": np.mean(np.abs(shap_values), axis=0),
            "mean_shap": np.mean(shap_values, axis=0),
        }
    )
    variable_importance["prediction_values_change_score"] = normalize_scores(
        variable_importance["prediction_values_change"]
    )
    variable_importance["mean_abs_shap_score"] = normalize_scores(
        variable_importance["mean_abs_shap"]
    )
    variable_importance["catboost_score"] = (
        variable_importance["prediction_values_change_score"]
        + variable_importance["mean_abs_shap_score"]
    ) / 2.0
    return variable_importance.sort_values(
        "catboost_score",
        ascending=False,
    ).reset_index(drop=True)


def build_long_shap_values(
    source_index: pd.Index,
    process_columns: list[str],
    shap_values: np.ndarray,
    max_rows: int = 2000,
) -> pd.DataFrame:
    """Return a compact long SHAP table for optional inspection/export."""
    rows = []
    for row_position, source_value in enumerate(source_index):
        for column_position, process_variable in enumerate(process_columns):
            rows.append(
                {
                    "source_index": source_value,
                    "process_variable": process_variable,
                    "shap_value": float(shap_values[row_position, column_position]),
                }
            )
            if len(rows) >= max_rows:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def run_catboost_time_ordered_validation(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
    validation_order_column: str | None,
    iterations: int,
    learning_rate: float,
    depth: int,
) -> dict[str, Any]:
    """Evaluate CatBoost on the most recent rows when an order column exists."""
    split = make_time_ordered_split(dataframe, outcome_column, validation_order_column)
    if not split["available"]:
        return split

    from catboost import CatBoostRegressor, Pool

    train_dataframe = dataframe.loc[split["train_index"], process_columns].copy()
    test_dataframe = dataframe.loc[split["test_index"], process_columns].copy()
    y_train = pd.to_numeric(
        dataframe.loc[split["train_index"], outcome_column],
        errors="coerce",
    ).to_numpy(dtype=float)
    y_test = pd.to_numeric(
        dataframe.loc[split["test_index"], outcome_column],
        errors="coerce",
    ).to_numpy(dtype=float)

    prepared_train, categorical_columns = prepare_catboost_dataframe(
        train_dataframe,
        process_columns,
    )
    prepared_test, _ = prepare_catboost_dataframe(test_dataframe, process_columns)
    cat_feature_indices = [
        prepared_train.columns.get_loc(column_name) for column_name in categorical_columns
    ]

    train_pool = Pool(prepared_train, y_train, cat_features=cat_feature_indices)
    test_pool = Pool(prepared_test, y_test, cat_features=cat_feature_indices)
    model = CatBoostRegressor(
        iterations=iterations,
        learning_rate=learning_rate,
        depth=depth,
        loss_function="RMSE",
        random_seed=RANDOM_STATE,
        verbose=False,
        allow_writing_files=False,
    )
    model.fit(train_pool)
    train_predictions = model.predict(train_pool)
    test_predictions = model.predict(test_pool)
    return {
        **split_without_indices(split),
        "train_r2": safe_r2_score(y_train, train_predictions),
        "test_r2": safe_r2_score(y_test, test_predictions),
    }
