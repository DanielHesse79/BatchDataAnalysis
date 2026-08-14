"""Analysis methods for Batch Insight Analyzer.

The functions in this module intentionally return plain dictionaries and
DataFrames. That keeps the Streamlit layer simple and makes each method easy to
test from a short script.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler


RANDOM_STATE = 42
IMPORTANT_SIGNAL_THRESHOLD = 0.55
CV_RESULT_COLUMNS = ["n_components", "cv_r2", "cv_r2_std_error", "cv_fold_count"]
MINIMUM_ROWS_FOR_HELD_OUT_PERMUTATION = 40
# Below this, cross-validated R2 is not distinguishable from noise, so no driver
# should carry a confident label no matter how the methods rank it.
USABLE_VALIDATION_R2 = 0.15
SMALL_SAMPLE_ROW_COUNT = 30


@dataclass(frozen=True)
class PreprocessedFeatures:
    """Feature matrix plus enough metadata to map encoded columns back to variables."""

    matrix: np.ndarray
    feature_names: list[str]
    feature_to_process_variable: dict[str, str]
    feature_groups: dict[str, list[int]]
    numeric_columns: list[str]
    categorical_columns: list[str]


@dataclass(frozen=True)
class FeaturePreprocessor:
    """Fitted preprocessing state for train/test-safe transforms."""

    numeric_columns: list[str]
    categorical_columns: list[str]
    numeric_medians: pd.Series
    categorical_modes: dict[str, Any]
    encoder: OneHotEncoder | None
    scaler: StandardScaler | None
    feature_names: list[str]
    feature_to_process_variable: dict[str, str]
    feature_groups: dict[str, list[int]]


def choose_validation_order_column(audit_result, merged_dataframe: pd.DataFrame) -> str | None:
    """Choose the first audit-detected date/sequence column for ordered validation."""
    if audit_result is None or audit_result.date_or_drift_columns.empty:
        return None

    for column_name in audit_result.date_or_drift_columns["column"].tolist():
        if column_name in merged_dataframe.columns:
            return column_name

    return None


def get_modeling_process_columns(
    process_columns: list[str],
    validation_order_column: str | None,
) -> list[str]:
    """Drop the ordering column from the modeling features.

    The column that defines the time order must not also be a predictor, or the
    model can simply learn the batch sequence. This pairs with
    ``choose_validation_order_column`` and belongs beside it: the two were
    duplicated across four UI call sites, where one copy drifting would silently
    change what the models were fitted on.
    """
    return [
        column_name
        for column_name in process_columns
        if column_name != validation_order_column
    ]


def run_all_analyses(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_columns: list[str],
    validation_order_column: str | None = None,
) -> dict[str, Any]:
    """Run Phase 4 analysis methods for all selected outcomes."""
    pca_result = run_pca(dataframe, process_columns)

    pls_results = {}
    random_forest_results = {}
    for outcome_column in outcome_columns:
        pls_results[outcome_column] = run_pls(
            dataframe,
            process_columns,
            outcome_column,
            validation_order_column=validation_order_column,
        )
        random_forest_results[outcome_column] = run_random_forest(
            dataframe,
            process_columns,
            outcome_column,
            validation_order_column=validation_order_column,
        )

    ranked_drivers = combine_ranked_drivers(
        pca_result=pca_result,
        pls_results=pls_results,
        random_forest_results=random_forest_results,
        outcome_columns=outcome_columns,
    )

    return {
        "pca": pca_result,
        "pls": pls_results,
        "random_forest": random_forest_results,
        "ranked_drivers": ranked_drivers,
        "validation_order_column": validation_order_column,
    }


def run_pca(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    max_components: int = 8,
) -> dict[str, Any]:
    """Standardize process features and run PCA."""
    features = preprocess_features(
        dataframe=dataframe,
        process_columns=process_columns,
        scale_all_features=True,
    )
    n_components = min(max_components, features.matrix.shape[0], features.matrix.shape[1])
    if n_components < 1:
        raise ValueError("PCA needs at least one usable process feature.")

    model = PCA(n_components=n_components, random_state=RANDOM_STATE)
    transformed_matrix = model.fit_transform(features.matrix)
    component_names = [f"PC{component_number}" for component_number in range(1, n_components + 1)]

    transformed_data = pd.DataFrame(transformed_matrix, columns=component_names, index=dataframe.index)
    transformed_data.insert(0, "source_index", dataframe.index)

    explained_variance = pd.DataFrame(
        {
            "component": component_names,
            "explained_variance_ratio": model.explained_variance_ratio_,
            "cumulative_variance_ratio": np.cumsum(model.explained_variance_ratio_),
        }
    )

    components = pd.DataFrame(model.components_, columns=features.feature_names)
    components.insert(0, "component", component_names)

    loading_values = model.components_.T * np.sqrt(model.explained_variance_)
    loadings = pd.DataFrame(
        loading_values,
        columns=component_names,
    )
    loadings.insert(0, "feature", features.feature_names)
    loadings.insert(
        1,
        "process_variable",
        [features.feature_to_process_variable[feature_name] for feature_name in features.feature_names],
    )

    process_loadings = aggregate_pca_loadings(
        loading_values=loading_values,
        component_names=component_names,
        explained_variance_ratio=model.explained_variance_ratio_,
        feature_groups=features.feature_groups,
    )

    return {
        "model": model,
        "components": components,
        "explained_variance": explained_variance,
        "loadings": loadings,
        "process_loadings": process_loadings,
        "transformed_data": transformed_data,
        "feature_names": features.feature_names,
        "feature_groups": features.feature_groups,
    }


def run_pls(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
    max_components: int = 4,
    cv_folds: int = 5,
    validation_order_column: str | None = None,
) -> dict[str, Any]:
    """Run PLS regression with cross-validation for one continuous outcome."""
    modeling_dataframe, y = get_modeling_frame_and_outcome(dataframe, process_columns, outcome_column)
    features = preprocess_features(
        dataframe=modeling_dataframe,
        process_columns=process_columns,
        scale_all_features=True,
    )

    best_component_count, cv_results = choose_pls_component_count(
        modeling_dataframe=modeling_dataframe,
        process_columns=process_columns,
        y=y,
        max_components=max_components,
        cv_folds=cv_folds,
    )

    model = PLSRegression(n_components=best_component_count, scale=False, max_iter=1000)
    model.fit(features.matrix, y)
    predictions = model.predict(features.matrix).reshape(-1)
    time_ordered_validation = run_pls_time_ordered_validation(
        dataframe=dataframe,
        process_columns=process_columns,
        outcome_column=outcome_column,
        validation_order_column=validation_order_column,
        max_components=max_components,
        cv_folds=cv_folds,
    )

    coefficients = np.asarray(model.coef_).reshape(-1)
    vip_scores = calculate_vip_scores(model, features.matrix)

    feature_coefficients = pd.DataFrame(
        {
            "feature": features.feature_names,
            "process_variable": [
                features.feature_to_process_variable[feature_name]
                for feature_name in features.feature_names
            ],
            "coefficient": coefficients,
            "abs_coefficient": np.abs(coefficients),
            "vip": vip_scores,
        }
    ).sort_values("abs_coefficient", ascending=False)

    variable_importance = aggregate_pls_variable_importance(
        feature_coefficients=feature_coefficients,
        feature_groups=features.feature_groups,
    )

    return {
        "model": model,
        "outcome": outcome_column,
        "n_rows_used": int(len(y)),
        "n_components": best_component_count,
        "cv_results": cv_results,
        "cv_r2": get_best_cv_r2(cv_results, best_component_count),
        "q2": get_best_cv_r2(cv_results, best_component_count),
        "q2_std_error": get_cv_std_error(cv_results, best_component_count),
        "training_r2": float(r2_score(y, predictions)),
        "time_ordered_validation": time_ordered_validation,
        "feature_coefficients": feature_coefficients,
        "variable_importance": variable_importance,
    }


def run_random_forest(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
    n_estimators: int = 150,
    permutation_repeats: int = 3,
    validation_order_column: str | None = None,
) -> dict[str, Any]:
    """Fit a Random Forest and return feature and process-variable importances."""
    modeling_dataframe, y = get_modeling_frame_and_outcome(dataframe, process_columns, outcome_column)
    features = preprocess_features(
        dataframe=modeling_dataframe,
        process_columns=process_columns,
        scale_all_features=False,
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    model.fit(features.matrix, y)
    predictions = model.predict(features.matrix)
    time_ordered_validation = run_random_forest_time_ordered_validation(
        dataframe=dataframe,
        process_columns=process_columns,
        outcome_column=outcome_column,
        validation_order_column=validation_order_column,
        n_estimators=n_estimators,
    )

    encoded_permutation = permutation_importance(
        model,
        features.matrix,
        y,
        n_repeats=permutation_repeats,
        random_state=RANDOM_STATE,
        scoring="r2",
        n_jobs=1,
    )

    group_permutation, permutation_scoring_basis = calculate_group_permutation_importance(
        modeling_dataframe=modeling_dataframe,
        process_columns=process_columns,
        y=y,
        repeats=permutation_repeats,
        n_estimators=n_estimators,
    )

    feature_importances = pd.DataFrame(
        {
            "feature": features.feature_names,
            "process_variable": [
                features.feature_to_process_variable[feature_name]
                for feature_name in features.feature_names
            ],
            "model_importance": model.feature_importances_,
            "encoded_permutation_importance": encoded_permutation.importances_mean,
            "encoded_permutation_std": encoded_permutation.importances_std,
        }
    ).sort_values("model_importance", ascending=False)

    variable_importance = aggregate_random_forest_variable_importance(
        feature_importances=feature_importances,
        group_permutation=group_permutation,
        feature_groups=features.feature_groups,
    )

    return {
        "model": model,
        "outcome": outcome_column,
        "n_rows_used": int(len(y)),
        "training_r2": float(r2_score(y, predictions)),
        "oob_r2": float(model.oob_score_),
        "time_ordered_validation": time_ordered_validation,
        "feature_importances": feature_importances,
        "group_permutation_importance": group_permutation,
        "permutation_scoring_basis": permutation_scoring_basis,
        "variable_importance": variable_importance,
    }


def preprocess_features(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    scale_all_features: bool,
) -> PreprocessedFeatures:
    """Impute, encode, and optionally standardize process features."""
    preprocessor = fit_feature_preprocessor(
        dataframe=dataframe,
        process_columns=process_columns,
        scale_all_features=scale_all_features,
    )
    return transform_features(dataframe, preprocessor)


def fit_feature_preprocessor(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    scale_all_features: bool,
) -> FeaturePreprocessor:
    """Fit imputation, encoding, and scaling state on one dataset."""
    if not process_columns:
        raise ValueError("At least one process column is required.")

    numeric_columns = [
        column_name
        for column_name in process_columns
        if is_modelable_numeric_column(dataframe[column_name])
    ]
    categorical_columns = [
        column_name for column_name in process_columns if column_name not in numeric_columns
    ]

    feature_names: list[str] = []
    feature_to_process_variable: dict[str, str] = {}
    feature_groups: dict[str, list[int]] = {column_name: [] for column_name in process_columns}
    numeric_medians = pd.Series(dtype=float)
    categorical_modes: dict[str, Any] = {}
    encoder = None

    if numeric_columns:
        numeric_frame = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce")
        numeric_medians = numeric_frame.median().fillna(0.0)
        for column_name in numeric_columns:
            feature_index = len(feature_names)
            feature_names.append(column_name)
            feature_to_process_variable[column_name] = column_name
            feature_groups[column_name].append(feature_index)

    if categorical_columns:
        categorical_frame, categorical_modes = impute_categorical_columns_with_modes(
            dataframe,
            categorical_columns,
        )
        encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
        encoder.fit(categorical_frame)

        for column_name, categories in zip(categorical_columns, encoder.categories_):
            for category in categories:
                feature_index = len(feature_names)
                feature_name = f"{column_name}={category}"
                feature_names.append(feature_name)
                feature_to_process_variable[feature_name] = column_name
                feature_groups[column_name].append(feature_index)

    if not feature_names:
        raise ValueError("No usable process features were found.")

    # Numeric columns and one-hot dummies are standardized together so a
    # categorical variable stays comparable with a numeric one. Standardizing a
    # dummy does amplify a rare level, but leaving dummies raw caps their
    # variance at 0.25 against 1.0 for numerics and makes categoricals nearly
    # invisible to PCA. The cardinality bias is handled where it actually
    # distorts the ranking - in the per-variable aggregation below.
    scaler = None
    if scale_all_features:
        unscaled_features = transform_features(
            dataframe,
            FeaturePreprocessor(
                numeric_columns=numeric_columns,
                categorical_columns=categorical_columns,
                numeric_medians=numeric_medians,
                categorical_modes=categorical_modes,
                encoder=encoder,
                scaler=None,
                feature_names=feature_names,
                feature_to_process_variable=feature_to_process_variable,
                feature_groups=feature_groups,
            ),
        )
        scaler = StandardScaler().fit(unscaled_features.matrix)

    return FeaturePreprocessor(
        numeric_columns=numeric_columns,
        categorical_columns=categorical_columns,
        numeric_medians=numeric_medians,
        categorical_modes=categorical_modes,
        encoder=encoder,
        scaler=scaler,
        feature_names=feature_names,
        feature_to_process_variable=feature_to_process_variable,
        feature_groups=feature_groups,
    )


def is_modelable_numeric_column(series: pd.Series, min_parse_fraction: float = 0.80) -> bool:
    """Return whether a column should be modeled as a number.

    Dtype alone is not enough: a numeric column that survived intake as text
    ("36.8") would otherwise be one-hot encoded into one category per distinct
    value, which silently wrecks every model that uses it.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return True

    if pd.api.types.is_datetime64_any_dtype(series):
        return False

    non_missing = series.dropna()
    if non_missing.empty:
        return False

    parsed_count = int(pd.to_numeric(non_missing, errors="coerce").notna().sum())
    return parsed_count / len(non_missing) >= min_parse_fraction


def transform_features(
    dataframe: pd.DataFrame,
    preprocessor: FeaturePreprocessor,
) -> PreprocessedFeatures:
    """Transform a dataset using fitted preprocessing state."""
    matrix_parts: list[np.ndarray] = []

    if preprocessor.numeric_columns:
        numeric_matrix = impute_numeric_columns_with_medians(
            dataframe,
            preprocessor.numeric_columns,
            preprocessor.numeric_medians,
        )
        matrix_parts.append(numeric_matrix)

    if preprocessor.categorical_columns:
        if preprocessor.encoder is None:
            raise ValueError("Categorical columns are present but no encoder was fitted.")
        categorical_frame = impute_categorical_columns_from_modes(
            dataframe,
            preprocessor.categorical_columns,
            preprocessor.categorical_modes,
        )
        matrix_parts.append(preprocessor.encoder.transform(categorical_frame))

    if not matrix_parts:
        raise ValueError("No usable process features were found.")

    feature_matrix = np.concatenate(matrix_parts, axis=1).astype(float)
    if preprocessor.scaler is not None:
        feature_matrix = preprocessor.scaler.transform(feature_matrix)

    return PreprocessedFeatures(
        matrix=feature_matrix,
        feature_names=preprocessor.feature_names,
        feature_to_process_variable=preprocessor.feature_to_process_variable,
        feature_groups=preprocessor.feature_groups,
        numeric_columns=preprocessor.numeric_columns,
        categorical_columns=preprocessor.categorical_columns,
    )


def impute_numeric_columns_with_medians(
    dataframe: pd.DataFrame,
    numeric_columns: list[str],
    medians: pd.Series,
) -> np.ndarray:
    """Median-impute numeric columns with fitted medians."""
    numeric_frame = dataframe[numeric_columns].apply(pd.to_numeric, errors="coerce").copy()

    for column_name in numeric_columns:
        fill_value = medians.get(column_name, 0.0)
        if pd.isna(fill_value):
            fill_value = 0.0
        numeric_frame[column_name] = numeric_frame[column_name].fillna(float(fill_value))

    return numeric_frame.to_numpy(dtype=float)


def impute_categorical_columns_with_modes(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Mode-impute categorical columns and return fitted modes."""
    categorical_frame = pd.DataFrame(index=dataframe.index)
    modes: dict[str, Any] = {}

    for column_name in categorical_columns:
        series = dataframe[column_name].astype("object")
        mode_values = series.dropna().mode()
        fill_value = mode_values.iloc[0] if not mode_values.empty else "Missing"
        modes[column_name] = fill_value
        categorical_frame[column_name] = series.fillna(fill_value).astype(str)

    return categorical_frame, modes


def impute_categorical_columns_from_modes(
    dataframe: pd.DataFrame,
    categorical_columns: list[str],
    modes: dict[str, Any],
) -> pd.DataFrame:
    """Mode-impute categorical columns using fitted modes."""
    categorical_frame = pd.DataFrame(index=dataframe.index)

    for column_name in categorical_columns:
        fill_value = modes.get(column_name, "Missing")
        categorical_frame[column_name] = (
            dataframe[column_name].astype("object").fillna(fill_value).astype(str)
        )

    return categorical_frame


def get_modeling_frame_and_outcome(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Drop rows where the selected continuous outcome is missing or non-numeric."""
    numeric_outcome = pd.to_numeric(dataframe[outcome_column], errors="coerce")
    usable_mask = numeric_outcome.notna()
    if int(usable_mask.sum()) < 3:
        raise ValueError(f"{outcome_column} needs at least 3 numeric rows for modeling.")

    modeling_dataframe = dataframe.loc[usable_mask, process_columns].copy()
    y = numeric_outcome.loc[usable_mask].to_numpy(dtype=float)
    return modeling_dataframe, y


def run_pls_time_ordered_validation(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
    validation_order_column: str | None,
    max_components: int,
    cv_folds: int,
) -> dict[str, Any]:
    """Evaluate PLS on the most recent rows when an order column exists."""
    split = make_time_ordered_split(dataframe, outcome_column, validation_order_column)
    if not split["available"]:
        return split

    train_dataframe = dataframe.loc[split["train_index"], process_columns].copy()
    test_dataframe = dataframe.loc[split["test_index"], process_columns].copy()
    y_train = pd.to_numeric(dataframe.loc[split["train_index"], outcome_column], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(dataframe.loc[split["test_index"], outcome_column], errors="coerce").to_numpy(dtype=float)

    preprocessor = fit_feature_preprocessor(
        train_dataframe,
        process_columns,
        scale_all_features=True,
    )
    train_features = transform_features(train_dataframe, preprocessor)
    test_features = transform_features(test_dataframe, preprocessor)

    # Component selection cross-validates within the training rows only, and
    # fits its own preprocessing per fold, so it takes the raw frame.
    component_count, cv_results = choose_pls_component_count(
        modeling_dataframe=train_dataframe,
        process_columns=process_columns,
        y=y_train,
        max_components=max_components,
        cv_folds=cv_folds,
    )
    model = PLSRegression(n_components=component_count, scale=False, max_iter=1000)
    model.fit(train_features.matrix, y_train)
    test_predictions = model.predict(test_features.matrix).reshape(-1)
    train_predictions = model.predict(train_features.matrix).reshape(-1)

    return {
        **split_without_indices(split),
        "n_components": component_count,
        "train_r2": safe_r2_score(y_train, train_predictions),
        "test_r2": safe_r2_score(y_test, test_predictions),
        "train_cv_q2": get_best_cv_r2(cv_results, component_count),
    }


def run_random_forest_time_ordered_validation(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_column: str,
    validation_order_column: str | None,
    n_estimators: int,
) -> dict[str, Any]:
    """Evaluate Random Forest on the most recent rows when an order column exists."""
    split = make_time_ordered_split(dataframe, outcome_column, validation_order_column)
    if not split["available"]:
        return split

    train_dataframe = dataframe.loc[split["train_index"], process_columns].copy()
    test_dataframe = dataframe.loc[split["test_index"], process_columns].copy()
    y_train = pd.to_numeric(dataframe.loc[split["train_index"], outcome_column], errors="coerce").to_numpy(dtype=float)
    y_test = pd.to_numeric(dataframe.loc[split["test_index"], outcome_column], errors="coerce").to_numpy(dtype=float)

    preprocessor = fit_feature_preprocessor(
        train_dataframe,
        process_columns,
        scale_all_features=False,
    )
    train_features = transform_features(train_dataframe, preprocessor)
    test_features = transform_features(test_dataframe, preprocessor)

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        oob_score=True,
        bootstrap=True,
    )
    model.fit(train_features.matrix, y_train)
    train_predictions = model.predict(train_features.matrix)
    test_predictions = model.predict(test_features.matrix)

    return {
        **split_without_indices(split),
        "train_r2": safe_r2_score(y_train, train_predictions),
        "test_r2": safe_r2_score(y_test, test_predictions),
        "oob_r2": float(model.oob_score_),
    }


def make_time_ordered_split(
    dataframe: pd.DataFrame,
    outcome_column: str,
    validation_order_column: str | None,
    test_fraction: float = 0.20,
    minimum_test_rows: int = 8,
) -> dict[str, Any]:
    """Create an older-train/newer-test split from a date or sequence column."""
    if not validation_order_column:
        return {
            "available": False,
            "reason": "No date, sequence, or campaign-like column was selected for time-ordered validation.",
            "validation_order_column": None,
        }

    if validation_order_column not in dataframe.columns:
        return {
            "available": False,
            "reason": f"{validation_order_column} is not present in the merged data.",
            "validation_order_column": validation_order_column,
        }

    outcome_values = pd.to_numeric(dataframe[outcome_column], errors="coerce")
    order_values = convert_order_values(dataframe[validation_order_column])
    usable_mask = outcome_values.notna() & order_values.notna()
    usable_indices = dataframe.index[usable_mask]

    if len(usable_indices) < 20:
        return {
            "available": False,
            "reason": "At least 20 rows with outcome and order values are needed for time-ordered validation.",
            "validation_order_column": validation_order_column,
        }

    ordered_indices = order_values.loc[usable_indices].sort_values().index
    test_row_count = max(minimum_test_rows, int(round(len(ordered_indices) * test_fraction)))
    test_row_count = min(test_row_count, len(ordered_indices) - 10)

    if test_row_count < 3:
        return {
            "available": False,
            "reason": "Not enough rows remain for a stable time-ordered test set.",
            "validation_order_column": validation_order_column,
        }

    train_index = list(ordered_indices[:-test_row_count])
    test_index = list(ordered_indices[-test_row_count:])
    # Display the original values, not the int64 sort keys, or a date boundary
    # is reported to the user as 1704067200000000000.
    sorted_order_values = dataframe[validation_order_column].loc[ordered_indices]

    return {
        "available": True,
        "validation_order_column": validation_order_column,
        "train_rows": len(train_index),
        "test_rows": len(test_index),
        "test_fraction": round(test_row_count / len(ordered_indices), 3),
        "test_start_order_value": order_value_for_display(sorted_order_values.iloc[-test_row_count]),
        "test_end_order_value": order_value_for_display(sorted_order_values.iloc[-1]),
        "train_index": train_index,
        "test_index": test_index,
    }


def convert_order_values(series: pd.Series) -> pd.Series:
    """Convert date-like or numeric order values into sortable numeric values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        # astype("int64") turns NaT into INT64_MIN, which would sort missing
        # dates in front of every real batch and pull them into the train split.
        return datetime_to_sortable_values(series)

    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().mean() >= 0.80:
        return numeric_values

    parsed_dates = pd.to_datetime(series, errors="coerce")
    if parsed_dates.notna().mean() >= 0.80:
        return datetime_to_sortable_values(parsed_dates)

    return pd.Series(np.nan, index=series.index)


def datetime_to_sortable_values(series: pd.Series) -> pd.Series:
    """Convert a datetime Series to float sort keys, keeping NaT missing."""
    sortable_values = pd.Series(np.nan, index=series.index, dtype=float)
    present_mask = series.notna()
    sortable_values.loc[present_mask] = series.loc[present_mask].astype("int64")
    return sortable_values


def split_without_indices(split: dict[str, Any]) -> dict[str, Any]:
    """Remove large index lists before storing validation metadata."""
    return {
        key: value
        for key, value in split.items()
        if key not in {"train_index", "test_index"}
    }


def safe_r2_score(y_true: np.ndarray, y_predicted: np.ndarray) -> float:
    """Return R2, or NaN when it is not defined."""
    if len(y_true) < 2 or np.nanstd(y_true) == 0:
        return np.nan
    return float(r2_score(y_true, y_predicted))


def order_value_for_display(value: Any) -> Any:
    """Make validation order values compact for display."""
    if pd.isna(value):
        return None
    if isinstance(value, pd.Timestamp):
        return value.date().isoformat()
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 3)
    return str(value)


def choose_pls_component_count(
    modeling_dataframe: pd.DataFrame,
    process_columns: list[str],
    y: np.ndarray,
    max_components: int,
    cv_folds: int,
) -> tuple[int, pd.DataFrame]:
    """Choose a PLS component count by honest cross-validation.

    Preprocessing is fitted inside each fold, so imputation medians and scaling
    never see the held-out rows. Scores are averaged across folds rather than
    pooled, and the smallest component count within one standard error of the
    best is chosen so the reported Q2 is not simply the maximum of several
    noisy estimates.
    """
    row_count = len(modeling_dataframe)
    maximum_allowed_components = max(1, min(max_components, len(process_columns), row_count - 2))

    n_splits = min(cv_folds, row_count)
    if n_splits < 3:
        return 1, pd.DataFrame(columns=CV_RESULT_COLUMNS)

    splitter = KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    folds = list(splitter.split(np.arange(row_count)))
    rows = []

    for component_count in range(1, maximum_allowed_components + 1):
        fold_scores = [
            score_pls_fold(
                modeling_dataframe=modeling_dataframe,
                process_columns=process_columns,
                y=y,
                train_index=train_index,
                test_index=test_index,
                component_count=component_count,
            )
            for train_index, test_index in folds
        ]
        usable_scores = [score for score in fold_scores if np.isfinite(score)]
        if not usable_scores:
            continue

        mean_score = float(np.mean(usable_scores))
        standard_error = (
            float(np.std(usable_scores, ddof=1) / np.sqrt(len(usable_scores)))
            if len(usable_scores) > 1
            else 0.0
        )
        rows.append(
            {
                "n_components": component_count,
                "cv_r2": mean_score,
                "cv_r2_std_error": standard_error,
                "cv_fold_count": len(usable_scores),
            }
        )

    if not rows:
        return 1, pd.DataFrame(columns=CV_RESULT_COLUMNS)

    cv_results = pd.DataFrame(rows)
    return select_components_within_one_standard_error(cv_results), cv_results


def score_pls_fold(
    modeling_dataframe: pd.DataFrame,
    process_columns: list[str],
    y: np.ndarray,
    train_index: np.ndarray,
    test_index: np.ndarray,
    component_count: int,
) -> float:
    """Fit preprocessing and PLS on one training fold and score the held-out fold."""
    train_frame = modeling_dataframe.iloc[train_index]
    test_frame = modeling_dataframe.iloc[test_index]

    try:
        preprocessor = fit_feature_preprocessor(
            dataframe=train_frame,
            process_columns=process_columns,
            scale_all_features=True,
        )
        train_features = transform_features(train_frame, preprocessor)
        test_features = transform_features(test_frame, preprocessor)
    except ValueError:
        return np.nan

    usable_components = min(
        component_count,
        train_features.matrix.shape[1],
        len(train_index) - 1,
    )
    if usable_components < 1:
        return np.nan

    model = PLSRegression(n_components=usable_components, scale=False, max_iter=1000)
    model.fit(train_features.matrix, y[train_index])
    predictions = model.predict(test_features.matrix).reshape(-1)
    return safe_r2_score(y[test_index], predictions)


def select_components_within_one_standard_error(cv_results: pd.DataFrame) -> int:
    """Return the simplest component count that is statistically as good as the best."""
    best_row = cv_results.sort_values(["cv_r2", "n_components"], ascending=[False, True]).iloc[0]
    threshold = float(best_row["cv_r2"]) - float(best_row["cv_r2_std_error"])

    qualifying = cv_results[cv_results["cv_r2"] >= threshold]
    if qualifying.empty:
        return int(best_row["n_components"])

    return int(qualifying["n_components"].min())


def calculate_vip_scores(model: PLSRegression, feature_matrix: np.ndarray) -> np.ndarray:
    """Calculate Variable Importance in Projection scores for a fitted PLS model."""
    scores = model.x_scores_
    weights = model.x_weights_
    y_loadings = model.y_loadings_.reshape(-1)

    feature_count = feature_matrix.shape[1]
    explained_y_by_component = np.sum(scores**2, axis=0) * y_loadings**2
    total_explained_y = np.sum(explained_y_by_component)

    if total_explained_y <= 0:
        return np.zeros(feature_count)

    normalized_weights = weights / np.linalg.norm(weights, axis=0, keepdims=True)
    return np.sqrt(
        feature_count
        * np.dot(normalized_weights**2, explained_y_by_component)
        / total_explained_y
    )


def aggregate_pca_loadings(
    loading_values: np.ndarray,
    component_names: list[str],
    explained_variance_ratio: np.ndarray,
    feature_groups: dict[str, list[int]],
) -> pd.DataFrame:
    """Collapse encoded PCA loadings back to original process variables."""
    rows: list[dict[str, Any]] = []

    for process_variable, feature_indices in feature_groups.items():
        row: dict[str, Any] = {"process_variable": process_variable}
        weighted_score = 0.0

        for component_index, component_name in enumerate(component_names):
            # Root-MEAN-square, not root-sum-square: a categorical with L levels
            # contributes L columns of loading mass against one column for a
            # numeric variable, so summing ranks by cardinality, not by signal.
            component_loading = float(
                np.sqrt(np.mean(loading_values[feature_indices, component_index] ** 2))
            ) if feature_indices else 0.0
            row[component_name] = component_loading
            weighted_score += component_loading * float(explained_variance_ratio[component_index])

        row["pca_score_raw"] = weighted_score
        rows.append(row)

    process_loadings = pd.DataFrame(rows)
    process_loadings["pca_score"] = normalize_scores(process_loadings["pca_score_raw"])
    return process_loadings.sort_values("pca_score", ascending=False).reset_index(drop=True)


def aggregate_pls_variable_importance(
    feature_coefficients: pd.DataFrame,
    feature_groups: dict[str, list[int]],
) -> pd.DataFrame:
    """Collapse encoded PLS coefficients and VIP scores to process-variable level."""
    rows: list[dict[str, Any]] = []

    for process_variable in feature_groups:
        variable_features = feature_coefficients[
            feature_coefficients["process_variable"] == process_variable
        ]
        if variable_features.empty:
            continue

        rows.append(
            {
                "process_variable": process_variable,
                # Mean, not sum: summing |coefficient| over one-hot levels gives a
                # six-level categorical six times the mass of a numeric variable.
                "mean_abs_coefficient": float(variable_features["abs_coefficient"].mean()),
                "max_abs_coefficient": float(variable_features["abs_coefficient"].max()),
                "max_vip": float(variable_features["vip"].max()),
            }
        )

    variable_importance = pd.DataFrame(rows)
    variable_importance["coefficient_score"] = normalize_scores(
        variable_importance["max_abs_coefficient"]
    )
    variable_importance["vip_score"] = normalize_scores(variable_importance["max_vip"])
    variable_importance["pls_score"] = (
        variable_importance["coefficient_score"] + variable_importance["vip_score"]
    ) / 2.0
    return variable_importance.sort_values("pls_score", ascending=False).reset_index(drop=True)


def aggregate_random_forest_variable_importance(
    feature_importances: pd.DataFrame,
    group_permutation: pd.DataFrame,
    feature_groups: dict[str, list[int]],
) -> pd.DataFrame:
    """Collapse encoded Random Forest importances to process-variable level."""
    rows: list[dict[str, Any]] = []

    for process_variable in feature_groups:
        variable_features = feature_importances[
            feature_importances["process_variable"] == process_variable
        ]
        rows.append(
            {
                "process_variable": process_variable,
                "model_importance": float(variable_features["model_importance"].sum()),
                "encoded_permutation_importance": float(
                    variable_features["encoded_permutation_importance"].clip(lower=0).sum()
                ),
            }
        )

    variable_importance = pd.DataFrame(rows)
    variable_importance = variable_importance.merge(
        group_permutation,
        on="process_variable",
        how="left",
    )
    variable_importance["model_importance_score"] = normalize_scores(
        variable_importance["model_importance"]
    )
    variable_importance["permutation_score"] = normalize_scores(
        variable_importance["group_permutation_importance"].clip(lower=0)
    )
    # Impurity importance (MDI) is kept for reference but not scored: it is
    # biased toward high-cardinality and continuous features regardless of
    # signal. Held-out permutation importance carries the ranking.
    variable_importance["rf_score"] = variable_importance["permutation_score"]

    return variable_importance.sort_values("rf_score", ascending=False).reset_index(drop=True)


def calculate_group_permutation_importance(
    modeling_dataframe: pd.DataFrame,
    process_columns: list[str],
    y: np.ndarray,
    repeats: int,
    n_estimators: int,
) -> tuple[pd.DataFrame, str]:
    """Permute all encoded columns of one process variable together.

    Scored on held-out rows, with preprocessing fitted on the training rows
    only. Two separate leaks matter here: an unconstrained forest fits its
    training rows almost perfectly, so permuting there measures memorization
    rather than prediction; and imputation medians, modes, and the one-hot
    vocabulary must not be learned from the rows being scored, because this
    score becomes ``rf_score`` and therefore drives the driver ranking.
    """
    row_count = len(modeling_dataframe)

    if row_count >= MINIMUM_ROWS_FOR_HELD_OUT_PERMUTATION:
        train_index, test_index = train_test_split(
            np.arange(row_count),
            test_size=0.25,
            random_state=RANDOM_STATE,
        )
        scoring_basis = "held-out rows"
    else:
        train_index = np.arange(row_count)
        test_index = np.arange(row_count)
        scoring_basis = "training rows (too few batches to hold data out)"

    train_frame = modeling_dataframe.iloc[train_index]
    test_frame = modeling_dataframe.iloc[test_index]
    preprocessor = fit_feature_preprocessor(
        dataframe=train_frame,
        process_columns=process_columns,
        scale_all_features=False,
    )
    train_features = transform_features(train_frame, preprocessor)
    feature_groups = preprocessor.feature_groups

    scoring_model = RandomForestRegressor(
        n_estimators=n_estimators,
        min_samples_leaf=2,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        bootstrap=True,
    )
    scoring_model.fit(train_features.matrix, y[train_index])

    scoring_matrix = transform_features(test_frame, preprocessor).matrix
    scoring_y = y[test_index]
    baseline_score = safe_r2_score(scoring_y, scoring_model.predict(scoring_matrix))

    rng = np.random.default_rng(RANDOM_STATE)
    rows = []

    for process_variable, feature_indices in feature_groups.items():
        if not feature_indices:
            rows.append(
                {
                    "process_variable": process_variable,
                    "group_permutation_importance": 0.0,
                    "group_permutation_std": 0.0,
                }
            )
            continue

        score_drops = []
        for _ in range(repeats):
            permuted_matrix = scoring_matrix.copy()
            shuffled_row_indices = rng.permutation(scoring_matrix.shape[0])
            permuted_matrix[:, feature_indices] = scoring_matrix[shuffled_row_indices][
                :, feature_indices
            ]
            permuted_score = safe_r2_score(scoring_y, scoring_model.predict(permuted_matrix))
            score_drops.append(baseline_score - permuted_score)

        rows.append(
            {
                "process_variable": process_variable,
                "group_permutation_importance": float(np.nanmean(score_drops)),
                "group_permutation_std": float(np.nanstd(score_drops)),
            }
        )

    return pd.DataFrame(rows), scoring_basis


def combine_ranked_drivers(
    pca_result: dict[str, Any],
    pls_results: dict[str, dict[str, Any]],
    random_forest_results: dict[str, dict[str, Any]],
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Create one ranked driver table across all methods and outcomes."""
    pca_scores = pca_result["process_loadings"][["process_variable", "pca_score"]]
    rows = []

    for outcome_column in outcome_columns:
        pls_result = pls_results[outcome_column]
        random_forest_result = random_forest_results[outcome_column]
        pls_scores = pls_result["variable_importance"][["process_variable", "pls_score"]]
        rf_scores = random_forest_result["variable_importance"][["process_variable", "rf_score"]]
        best_validation_r2 = get_best_validation_r2(pls_result, random_forest_result)
        rows_used = int(random_forest_result.get("n_rows_used") or 0) or None

        merged_scores = pca_scores.merge(pls_scores, on="process_variable", how="outer")
        merged_scores = merged_scores.merge(rf_scores, on="process_variable", how="outer")
        merged_scores[["pca_score", "pls_score", "rf_score"]] = merged_scores[
            ["pca_score", "pls_score", "rf_score"]
        ].fillna(0.0)

        for _, score_row in merged_scores.iterrows():
            pca_score = float(score_row["pca_score"])
            pls_score = float(score_row["pls_score"])
            rf_score = float(score_row["rf_score"])
            evidence_methods = get_evidence_methods(pca_score, pls_score, rf_score)
            combined_score = (0.05 * pca_score) + (0.35 * pls_score) + (0.60 * rf_score)

            rows.append(
                {
                    "outcome": outcome_column,
                    "process_variable": score_row["process_variable"],
                    "combined_score": combined_score,
                    "confidence": assign_confidence(
                        combined_score,
                        evidence_methods,
                        best_validation_r2=best_validation_r2,
                        rows_used=rows_used,
                    ),
                    "evidence_methods": ", ".join(evidence_methods) if evidence_methods else "none",
                    "pca_score": pca_score,
                    "pls_score": pls_score,
                    "rf_score": rf_score,
                }
            )

    ranked_drivers = pd.DataFrame(rows)
    ranked_drivers["rank"] = (
        ranked_drivers.groupby("outcome")["combined_score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    display_columns = [
        "outcome",
        "rank",
        "process_variable",
        "combined_score",
        "confidence",
        "evidence_methods",
        "pca_score",
        "pls_score",
        "rf_score",
    ]
    return (
        ranked_drivers[display_columns]
        .sort_values(["outcome", "rank"])
        .reset_index(drop=True)
    )


def get_evidence_methods(pca_score: float, pls_score: float, rf_score: float) -> list[str]:
    """Return method names that consider a variable important."""
    evidence_methods = []
    if pca_score >= IMPORTANT_SIGNAL_THRESHOLD:
        evidence_methods.append("PCA")
    if pls_score >= IMPORTANT_SIGNAL_THRESHOLD:
        evidence_methods.append("PLS")
    if rf_score >= IMPORTANT_SIGNAL_THRESHOLD:
        evidence_methods.append("Random Forest")
    return evidence_methods


def assign_confidence(
    combined_score: float,
    evidence_methods: list[str],
    best_validation_r2: float | None = None,
    rows_used: int | None = None,
) -> str:
    """Assign a readable confidence label from agreement, validation, and sample size.

    Method scores are normalized by their own maximum, so some variable always
    scores 1.0 for each method - even on pure noise. Agreement alone would then
    label a random variable "high", so a label above "exploratory" also requires
    a model that actually predicts held-out data.
    """
    evidence_count = len(evidence_methods)

    if combined_score >= 0.55 and evidence_count >= 2:
        label = "high"
    elif combined_score >= 0.30 and evidence_count >= 1:
        label = "medium"
    else:
        label = "exploratory"

    if label == "exploratory":
        return label

    if best_validation_r2 is None or not np.isfinite(best_validation_r2):
        return "exploratory"

    if best_validation_r2 < USABLE_VALIDATION_R2:
        return "exploratory"

    if label == "high" and rows_used is not None and rows_used < SMALL_SAMPLE_ROW_COUNT:
        return "medium"

    return label


def get_best_validation_r2(
    pls_result: dict[str, Any],
    random_forest_result: dict[str, Any],
) -> float:
    """Return the strongest honest validation score available for an outcome.

    Training R2 is deliberately excluded: it is not evidence that the model
    generalizes.
    """
    candidate_scores = [
        pls_result.get("q2"),
        random_forest_result.get("oob_r2"),
        get_time_ordered_test_r2(pls_result),
        get_time_ordered_test_r2(random_forest_result),
    ]
    usable_scores = [
        float(score)
        for score in candidate_scores
        if score is not None and np.isfinite(float(score))
    ]
    return max(usable_scores) if usable_scores else np.nan


def get_time_ordered_test_r2(method_result: dict[str, Any]) -> float | None:
    """Read the time-ordered test R2 from a method result when available."""
    validation = method_result.get("time_ordered_validation") or {}
    if not validation.get("available"):
        return None
    return validation.get("test_r2")


def normalize_scores(values: pd.Series | np.ndarray) -> pd.Series:
    """Normalize non-negative scores to 0-1 while handling all-zero inputs."""
    score_series = pd.Series(values, dtype=float).fillna(0.0).clip(lower=0.0)
    maximum_value = float(score_series.max())

    if maximum_value <= 0.0:
        return pd.Series(np.zeros(len(score_series)), index=score_series.index)

    return score_series / maximum_value


def get_best_cv_r2(cv_results: pd.DataFrame, component_count: int) -> float:
    """Return the cross-validated R2 for the selected PLS component count."""
    return get_cv_result_value(cv_results, component_count, "cv_r2")


def get_cv_std_error(cv_results: pd.DataFrame, component_count: int) -> float:
    """Return the across-fold standard error for the selected component count."""
    return get_cv_result_value(cv_results, component_count, "cv_r2_std_error")


def get_cv_result_value(
    cv_results: pd.DataFrame,
    component_count: int,
    column_name: str,
) -> float:
    """Read one cross-validation statistic for a component count."""
    if cv_results.empty or column_name not in cv_results.columns:
        return np.nan

    matching_rows = cv_results[cv_results["n_components"] == component_count]
    if matching_rows.empty:
        return np.nan

    return float(matching_rows.iloc[0][column_name])
