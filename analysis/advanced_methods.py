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
    fit_feature_preprocessor,
    make_time_ordered_split,
    normalize_scores,
    preprocess_features,
    safe_r2_score,
    split_without_indices,
    transform_features,
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

EMPTY_XGBOOST_RESULT_TABLES = {
    "variable_importance": [
        "process_variable",
        "gain_importance",
        "mean_abs_shap",
        "mean_shap",
        "xgboost_score",
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

    prepared_frame, categorical_columns, _ = prepare_catboost_dataframe(
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
    fitted_medians: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, list[str], dict[str, float]]:
    """Impute process columns in a CatBoost-friendly way.

    Pass ``fitted_medians`` from the training rows when preparing a test set.
    Computing medians from the test rows themselves leaks held-out information
    and makes the reported test score incomparable with the other methods.
    """
    prepared_frame = pd.DataFrame(index=dataframe.index)
    categorical_columns: list[str] = []
    medians: dict[str, float] = {}

    for column_name in process_columns:
        series = dataframe[column_name]
        if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
            numeric_values = pd.to_numeric(series, errors="coerce")
            if fitted_medians is not None and column_name in fitted_medians:
                median_value = fitted_medians[column_name]
            else:
                median_value = numeric_values.median()
            if pd.isna(median_value):
                median_value = 0.0
            medians[column_name] = float(median_value)
            prepared_frame[column_name] = numeric_values.fillna(float(median_value))
        else:
            categorical_columns.append(column_name)
            prepared_frame[column_name] = (
                series.astype("object")
                .where(series.notna(), "Missing")
                .astype(str)
            )

    return prepared_frame[process_columns], categorical_columns, medians


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
    """Return a compact long SHAP table for optional inspection/export.

    The row cap is applied per batch, not per row. Breaking inside a batch left
    the last batch holding only some of its variables, which reads as a real
    zero contribution for the rest.
    """
    if not process_columns or len(source_index) == 0:
        return pd.DataFrame(columns=["source_index", "process_variable", "shap_value"])

    max_batches = max(1, max_rows // len(process_columns))
    included_positions = range(min(len(source_index), max_batches))

    rows = [
        {
            "source_index": source_index[row_position],
            "process_variable": process_variable,
            "shap_value": float(shap_values[row_position, column_position]),
        }
        for row_position in included_positions
        for column_position, process_variable in enumerate(process_columns)
    ]
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

    prepared_train, categorical_columns, train_medians = prepare_catboost_dataframe(
        train_dataframe,
        process_columns,
    )
    prepared_test, _, _ = prepare_catboost_dataframe(
        test_dataframe,
        process_columns,
        fitted_medians=train_medians,
    )
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


# ---------------------------------------------------------------------------
# XGBoost + native TreeSHAP (optional)
#
# XGBoost is a third non-linear driver vote. Unlike CatBoost it has no clean
# native categorical handling, so we reuse the same one-hot preprocessing as the
# base PLS/Random Forest path (`preprocess_features`) and collapse encoded SHAP
# contributions back to process variables via `feature_groups`.
#
# SHAP comes from XGBoost's own TreeSHAP (`pred_contribs=True`), so the heavy
# `shap` package is intentionally not a dependency.
# ---------------------------------------------------------------------------

XGBOOST_DEFAULTS = {
    "n_estimators": 400,
    "learning_rate": 0.05,
    "max_depth": 4,
}


def get_xgboost_dependency_status() -> dict[str, Any]:
    """Return whether optional XGBoost explainability can run."""
    xgboost_available = find_spec("xgboost") is not None
    if xgboost_available:
        return {
            "available": True,
            "missing_packages": [],
            "message": "XGBoost is installed. Native TreeSHAP contributions can be computed.",
        }
    return {
        "available": False,
        "missing_packages": ["xgboost"],
        "message": (
            "XGBoost is not installed. Install it to enable the optional XGBoost + SHAP tab."
        ),
    }


def run_xgboost_shap(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
    validation_order_column: str | None = None,
    n_estimators: int = XGBOOST_DEFAULTS["n_estimators"],
    learning_rate: float = XGBOOST_DEFAULTS["learning_rate"],
    max_depth: int = XGBOOST_DEFAULTS["max_depth"],
) -> dict[str, Any]:
    """Fit XGBoost and return native TreeSHAP global explanations.

    Returns an unavailable result instead of raising when XGBoost is missing,
    keeping the feature safe on machines without the optional dependency.
    """
    dependency_status = get_xgboost_dependency_status()
    if not dependency_status["available"]:
        return unavailable_xgboost_result(outcome_column, dependency_status["message"])

    import xgboost as xgb

    modeling_dataframe, y = prepare_modeling_frame(dataframe, process_columns, outcome_column)
    if len(modeling_dataframe) < 20:
        return unavailable_xgboost_result(
            outcome_column,
            "XGBoost + SHAP needs at least 20 usable rows for this app workflow.",
        )

    features = preprocess_features(
        dataframe=modeling_dataframe,
        process_columns=process_columns,
        scale_all_features=False,
    )

    model = build_xgboost_regressor(xgb, n_estimators, learning_rate, max_depth)
    model.fit(features.matrix, y)
    predictions = model.predict(features.matrix)

    booster = model.get_booster()
    contribution_matrix = booster.predict(
        xgb.DMatrix(features.matrix, feature_names=features.feature_names),
        pred_contribs=True,
    )
    # The final column is the bias/expected-value term; drop it.
    shap_values = np.asarray(contribution_matrix[:, :-1], dtype=float)

    variable_importance = build_xgboost_variable_importance(
        feature_groups=features.feature_groups,
        model_importances=np.asarray(model.feature_importances_, dtype=float),
        shap_values=shap_values,
    )
    time_ordered_validation = run_xgboost_time_ordered_validation(
        dataframe=dataframe,
        process_columns=process_columns,
        outcome_column=outcome_column,
        validation_order_column=validation_order_column,
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
    )

    return {
        "available": True,
        "reason": "",
        "model": model,
        "outcome": outcome_column,
        "n_rows_used": int(len(y)),
        "numeric_feature_count": int(len(features.numeric_columns)),
        "categorical_feature_count": int(len(features.categorical_columns)),
        "training_r2": float(r2_score(y, predictions)),
        "time_ordered_validation": time_ordered_validation,
        "variable_importance": variable_importance,
        "shap_values": build_long_grouped_shap_values(
            source_index=modeling_dataframe.index,
            feature_groups=features.feature_groups,
            shap_values=shap_values,
        ),
    }


def unavailable_xgboost_result(outcome_column: str, reason: str) -> dict[str, Any]:
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
            columns=EMPTY_XGBOOST_RESULT_TABLES["variable_importance"]
        ),
        "shap_values": pd.DataFrame(columns=EMPTY_XGBOOST_RESULT_TABLES["shap_values"]),
    }


def build_xgboost_regressor(
    xgb_module: Any,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> Any:
    """Build a conservatively-regularized XGBoost regressor for small SME data."""
    return xgb_module.XGBRegressor(
        n_estimators=n_estimators,
        learning_rate=learning_rate,
        max_depth=max_depth,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=1.0,
        min_child_weight=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        tree_method="hist",
    )


def build_xgboost_variable_importance(
    feature_groups: dict[str, list[int]],
    model_importances: np.ndarray,
    shap_values: np.ndarray,
) -> pd.DataFrame:
    """Collapse encoded XGBoost gain and TreeSHAP values to process-variable level."""
    mean_abs_shap_encoded = np.mean(np.abs(shap_values), axis=0)
    mean_shap_encoded = np.mean(shap_values, axis=0)

    rows: list[dict[str, Any]] = []
    for process_variable, feature_indices in feature_groups.items():
        rows.append(
            {
                "process_variable": process_variable,
                "gain_importance": float(np.sum(model_importances[feature_indices])),
                "mean_abs_shap": float(np.sum(mean_abs_shap_encoded[feature_indices])),
                "mean_shap": float(np.sum(mean_shap_encoded[feature_indices])),
            }
        )

    variable_importance = pd.DataFrame(rows)
    variable_importance["gain_score"] = normalize_scores(variable_importance["gain_importance"])
    variable_importance["mean_abs_shap_score"] = normalize_scores(
        variable_importance["mean_abs_shap"]
    )
    variable_importance["xgboost_score"] = (
        variable_importance["gain_score"] + variable_importance["mean_abs_shap_score"]
    ) / 2.0
    return variable_importance.sort_values(
        "xgboost_score",
        ascending=False,
    ).reset_index(drop=True)


def build_long_grouped_shap_values(
    source_index: pd.Index,
    feature_groups: dict[str, list[int]],
    shap_values: np.ndarray,
    max_rows: int = 2000,
) -> pd.DataFrame:
    """Return a compact long SHAP table, summed per process variable per batch."""
    rows = []
    for row_position, source_value in enumerate(source_index):
        for process_variable, feature_indices in feature_groups.items():
            rows.append(
                {
                    "source_index": source_value,
                    "process_variable": process_variable,
                    "shap_value": float(np.sum(shap_values[row_position, feature_indices])),
                }
            )
            if len(rows) >= max_rows:
                return pd.DataFrame(rows)
    return pd.DataFrame(rows)


def run_xgboost_time_ordered_validation(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
    validation_order_column: str | None,
    n_estimators: int,
    learning_rate: float,
    max_depth: int,
) -> dict[str, Any]:
    """Evaluate XGBoost on the most recent rows when an order column exists."""
    split = make_time_ordered_split(dataframe, outcome_column, validation_order_column)
    if not split["available"]:
        return split

    import xgboost as xgb

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

    preprocessor = fit_feature_preprocessor(
        train_dataframe,
        process_columns,
        scale_all_features=False,
    )
    train_features = transform_features(train_dataframe, preprocessor)
    test_features = transform_features(test_dataframe, preprocessor)

    model = build_xgboost_regressor(xgb, n_estimators, learning_rate, max_depth)
    model.fit(train_features.matrix, y_train)
    train_predictions = model.predict(train_features.matrix)
    test_predictions = model.predict(test_features.matrix)
    return {
        **split_without_indices(split),
        "train_r2": safe_r2_score(y_train, train_predictions),
        "test_r2": safe_r2_score(y_test, test_predictions),
    }
