"""Data profiling helpers for Batch Insight Analyzer."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


HIGH_MISSING_PERCENT_THRESHOLD = 20.0
NEAR_CONSTANT_TOP_VALUE_PERCENT_THRESHOLD = 95.0


@dataclass(frozen=True)
class ProfileResult:
    """A compact summary of the merged dataset before modeling."""

    matched_batch_count: int
    process_columns: list[str]
    outcome_columns: list[str]
    variable_type_counts: dict[str, int]
    variable_types: pd.DataFrame
    missingness: pd.DataFrame
    near_constant_columns: pd.DataFrame
    outcome_statistics: pd.DataFrame
    warnings: list[str]


def profile_merged_data(
    merged_dataframe: pd.DataFrame,
    outcome_columns: list[str],
    batch_id_column: str = "batch_id",
    matched_batch_count: int | None = None,
) -> ProfileResult:
    """Profile a merged process/QC dataset before running analysis methods."""
    if merged_dataframe.empty:
        raise ValueError("The merged dataset is empty.")

    missing_outcomes = [
        column_name for column_name in outcome_columns if column_name not in merged_dataframe.columns
    ]
    if missing_outcomes:
        missing_text = ", ".join(missing_outcomes)
        raise ValueError(f"Selected outcome column(s) are missing from the merged data: {missing_text}")

    process_columns = [
        column_name
        for column_name in merged_dataframe.columns
        if column_name not in outcome_columns and column_name != batch_id_column
    ]

    variable_types = classify_process_variables(merged_dataframe, process_columns)
    variable_type_counts = {
        variable_type: int((variable_types["type"] == variable_type).sum())
        for variable_type in ["continuous", "categorical", "binary"]
    }

    missingness = calculate_missingness(merged_dataframe)
    near_constant_columns = find_near_constant_columns(merged_dataframe, process_columns)
    outcome_statistics = summarize_outcomes(merged_dataframe, outcome_columns)

    warnings = build_profile_warnings(
        missingness=missingness,
        near_constant_columns=near_constant_columns,
        outcome_statistics=outcome_statistics,
    )

    return ProfileResult(
        matched_batch_count=matched_batch_count or len(merged_dataframe),
        process_columns=process_columns,
        outcome_columns=outcome_columns,
        variable_type_counts=variable_type_counts,
        variable_types=variable_types,
        missingness=missingness,
        near_constant_columns=near_constant_columns,
        outcome_statistics=outcome_statistics,
        warnings=warnings,
    )


def classify_process_variables(
    dataframe: pd.DataFrame,
    process_columns: list[str],
) -> pd.DataFrame:
    """Classify process columns as continuous, categorical, or binary."""
    rows: list[dict[str, Any]] = []

    for column_name in process_columns:
        series = dataframe[column_name]
        non_missing_series = series.dropna()
        unique_count = int(non_missing_series.nunique())
        variable_type = classify_series(series)

        rows.append(
            {
                "column": column_name,
                "type": variable_type,
                "non_missing_count": int(non_missing_series.shape[0]),
                "unique_count": unique_count,
            }
        )

    return pd.DataFrame(rows).sort_values(["type", "column"]).reset_index(drop=True)


def classify_series(series: pd.Series) -> str:
    """Return the variable type that the analysis pipeline should assume."""
    non_missing_series = series.dropna()
    unique_count = int(non_missing_series.nunique())

    if unique_count <= 2:
        return "binary"

    if pd.api.types.is_numeric_dtype(series):
        return "continuous"

    return "categorical"


def calculate_missingness(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Calculate missing values and flag columns above the configured threshold."""
    row_count = max(len(dataframe), 1)
    rows = []

    for column_name in dataframe.columns:
        missing_count = int(dataframe[column_name].isna().sum())
        missing_percent = missing_count / row_count * 100.0
        rows.append(
            {
                "column": column_name,
                "missing_count": missing_count,
                "missing_percent": round(missing_percent, 2),
                "flag": missing_percent > HIGH_MISSING_PERCENT_THRESHOLD,
            }
        )

    return pd.DataFrame(rows).sort_values("missing_percent", ascending=False).reset_index(drop=True)


def find_near_constant_columns(
    dataframe: pd.DataFrame,
    process_columns: list[str],
) -> pd.DataFrame:
    """Find process variables that are constant or dominated by one value."""
    rows: list[dict[str, Any]] = []

    for column_name in process_columns:
        series = dataframe[column_name]
        non_missing_series = series.dropna()
        non_missing_count = int(non_missing_series.shape[0])

        if non_missing_count == 0:
            rows.append(
                {
                    "column": column_name,
                    "reason": "all values missing",
                    "unique_count": 0,
                    "top_value": None,
                    "top_value_percent": 100.0,
                }
            )
            continue

        value_counts = non_missing_series.value_counts(dropna=True)
        unique_count = int(value_counts.shape[0])
        top_value = value_counts.index[0]
        top_value_percent = float(value_counts.iloc[0] / non_missing_count * 100.0)

        if unique_count <= 1:
            reason = "constant"
        elif top_value_percent >= NEAR_CONSTANT_TOP_VALUE_PERCENT_THRESHOLD:
            reason = "near-constant"
        elif pd.api.types.is_numeric_dtype(series) and float(non_missing_series.var()) <= 1e-12:
            reason = "near-zero variance"
        else:
            continue

        rows.append(
            {
                "column": column_name,
                "reason": reason,
                "unique_count": unique_count,
                "top_value": format_profile_value(top_value),
                "top_value_percent": round(top_value_percent, 2),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["column", "reason", "unique_count", "top_value", "top_value_percent"]
        )

    return pd.DataFrame(rows).sort_values("top_value_percent", ascending=False).reset_index(drop=True)


def summarize_outcomes(dataframe: pd.DataFrame, outcome_columns: list[str]) -> pd.DataFrame:
    """Summarize selected QC outcomes as continuous variables."""
    rows: list[dict[str, Any]] = []

    for column_name in outcome_columns:
        numeric_series = pd.to_numeric(dataframe[column_name], errors="coerce")
        missing_count = int(numeric_series.isna().sum())
        usable_series = numeric_series.dropna()

        rows.append(
            {
                "outcome": column_name,
                "count": int(usable_series.shape[0]),
                "missing_count": missing_count,
                "missing_percent": round(missing_count / max(len(dataframe), 1) * 100.0, 2),
                "mean": round_or_nan(usable_series.mean()),
                "std": round_or_nan(usable_series.std()),
                "min": round_or_nan(usable_series.min()),
                "median": round_or_nan(usable_series.median()),
                "max": round_or_nan(usable_series.max()),
            }
        )

    return pd.DataFrame(rows)


def build_profile_warnings(
    missingness: pd.DataFrame,
    near_constant_columns: pd.DataFrame,
    outcome_statistics: pd.DataFrame,
) -> list[str]:
    """Create plain-language warnings for the profile summary."""
    warnings: list[str] = []

    high_missing_columns = missingness[missingness["flag"]]
    if not high_missing_columns.empty:
        warnings.append(
            f"{len(high_missing_columns):,} column(s) have more than "
            f"{HIGH_MISSING_PERCENT_THRESHOLD:.0f}% missing values."
        )

    if not near_constant_columns.empty:
        warnings.append(
            f"{len(near_constant_columns):,} process variable(s) are constant or near-constant."
        )

    empty_outcomes = outcome_statistics[outcome_statistics["count"] == 0]
    if not empty_outcomes.empty:
        outcomes_text = ", ".join(empty_outcomes["outcome"].tolist())
        warnings.append(
            "These selected outcomes do not contain numeric values and cannot be modeled as "
            f"continuous outcomes: {outcomes_text}."
        )

    return warnings


def create_histogram_dataframe(
    dataframe: pd.DataFrame,
    outcome_column: str,
    bin_count: int = 20,
) -> pd.DataFrame:
    """Return histogram bins for a selected outcome."""
    numeric_series = pd.to_numeric(dataframe[outcome_column], errors="coerce").dropna()

    if numeric_series.empty:
        return pd.DataFrame(columns=["bin_center", "count"])

    counts, bin_edges = np.histogram(numeric_series, bins=bin_count)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    return pd.DataFrame(
        {
            "bin_center": bin_centers,
            "count": counts,
        }
    )


def round_or_nan(value: float, digits: int = 3) -> float:
    """Round numeric values while preserving NaN for display."""
    if pd.isna(value):
        return np.nan
    return round(float(value), digits)


def format_profile_value(value: Any) -> str:
    """Format values for small profile tables."""
    if pd.isna(value):
        return ""
    if isinstance(value, float):
        return f"{value:.3g}"
    return str(value)
