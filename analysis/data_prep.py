"""Data loading and batch matching helpers.

The Streamlit app uses these functions for Phase 2. Keeping them here makes the
merge behavior easier to test without clicking through the UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis.normalization import normalize_batch_id_series


INTERNAL_BATCH_KEY = "__batch_id_key"


class DataPrepError(ValueError):
    """User-facing error raised when uploaded files cannot be prepared."""


@dataclass(frozen=True)
class MergeResult:
    """Result object returned after a successful batch merge."""

    dataframe: pd.DataFrame
    matched_batch_count: int
    process_batch_count: int
    qc_batch_count: int
    unmatched_process_batch_ids: list[str]
    unmatched_qc_batch_ids: list[str]


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    `uploaded_file` can be a Streamlit UploadedFile or a local path-like object,
    which keeps this helper usable in simple tests.
    """
    if uploaded_file is None:
        raise DataPrepError("No file was provided.")

    file_name = getattr(uploaded_file, "name", str(uploaded_file))
    file_extension = Path(file_name).suffix.lower()

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        if file_extension == ".csv":
            dataframe = pd.read_csv(uploaded_file)
        elif file_extension in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(uploaded_file)
        else:
            raise DataPrepError("Please upload a .csv, .xlsx, or .xls file.")
    except pd.errors.EmptyDataError as error:
        raise DataPrepError(f"{file_name} appears to be empty.") from error
    except ImportError as error:
        raise DataPrepError(
            f"Reading {file_extension} files requires an additional Excel dependency."
        ) from error
    except Exception as error:
        raise DataPrepError(f"Could not read {file_name}: {error}") from error

    dataframe = clean_column_names(dataframe)
    validate_non_empty_dataframe(dataframe, file_name)
    return dataframe


def clean_column_names(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with string column names stripped of surrounding spaces."""
    cleaned_dataframe = dataframe.copy()
    cleaned_dataframe.columns = [str(column_name).strip() for column_name in cleaned_dataframe.columns]
    return cleaned_dataframe


def validate_non_empty_dataframe(dataframe: pd.DataFrame, label: str) -> None:
    """Raise a clear error if a loaded file has no usable rows or columns."""
    if dataframe.empty or len(dataframe.columns) == 0:
        raise DataPrepError(f"{label} does not contain any rows or columns.")

    if len(dataframe.index) == 0:
        raise DataPrepError(f"{label} does not contain any data rows.")


def get_batch_column_default_index(columns: Iterable[str]) -> int:
    """Choose a helpful default index for a batch ID selectbox."""
    column_list = list(columns)
    normalized_columns = [column_name.strip().lower() for column_name in column_list]

    for candidate_name in ("batch_id", "batch id", "batchid", "batch"):
        if candidate_name in normalized_columns:
            return normalized_columns.index(candidate_name)

    return 0


def get_default_outcome_columns(qc_dataframe: pd.DataFrame, batch_id_column: str) -> list[str]:
    """Suggest numeric QC columns as default quality outcomes."""
    candidate_columns = [
        column_name for column_name in qc_dataframe.columns if column_name != batch_id_column
    ]
    numeric_columns = [
        column_name
        for column_name in candidate_columns
        if pd.api.types.is_numeric_dtype(qc_dataframe[column_name])
    ]
    return numeric_columns or candidate_columns


def merge_process_and_qc_data(
    process_dataframe: pd.DataFrame,
    qc_dataframe: pd.DataFrame,
    process_batch_id_column: str,
    qc_batch_id_column: str,
    outcome_columns: list[str],
) -> MergeResult:
    """Validate and inner-join process data with selected QC outcomes."""
    validate_non_empty_dataframe(process_dataframe, "The process data file")
    validate_non_empty_dataframe(qc_dataframe, "The QC results file")
    validate_required_columns(process_dataframe, [process_batch_id_column], "process data")
    validate_required_columns(qc_dataframe, [qc_batch_id_column], "QC results")

    if not outcome_columns:
        raise DataPrepError("Select at least one quality outcome column.")

    validate_required_columns(qc_dataframe, outcome_columns, "QC results")
    validate_no_outcome_name_collisions(
        process_dataframe,
        process_batch_id_column,
        outcome_columns,
    )

    process_keys = normalize_batch_ids(process_dataframe[process_batch_id_column])
    qc_keys = normalize_batch_ids(qc_dataframe[qc_batch_id_column])

    validate_batch_ids(process_keys, "process data")
    validate_batch_ids(qc_keys, "QC results")

    process_for_merge = process_dataframe.copy()
    qc_for_merge = qc_dataframe[[qc_batch_id_column, *outcome_columns]].copy()

    process_for_merge[INTERNAL_BATCH_KEY] = process_keys
    qc_for_merge[INTERNAL_BATCH_KEY] = qc_keys

    merged_dataframe = process_for_merge.merge(
        qc_for_merge[[INTERNAL_BATCH_KEY, *outcome_columns]],
        on=INTERNAL_BATCH_KEY,
        how="inner",
    )

    if merged_dataframe.empty:
        raise DataPrepError(
            "No matching batch IDs were found. Check that the selected batch ID columns use the same IDs."
        )

    matched_keys = set(merged_dataframe[INTERNAL_BATCH_KEY])
    process_key_set = set(process_keys)
    qc_key_set = set(qc_keys)

    unmatched_process_batch_ids = sorted(process_key_set - matched_keys)
    unmatched_qc_batch_ids = sorted(qc_key_set - matched_keys)

    merged_dataframe = merged_dataframe.drop(columns=[INTERNAL_BATCH_KEY])
    if process_batch_id_column != "batch_id" and "batch_id" not in merged_dataframe.columns:
        merged_dataframe = merged_dataframe.rename(columns={process_batch_id_column: "batch_id"})

    return MergeResult(
        dataframe=merged_dataframe,
        matched_batch_count=len(merged_dataframe),
        process_batch_count=len(process_key_set),
        qc_batch_count=len(qc_key_set),
        unmatched_process_batch_ids=unmatched_process_batch_ids,
        unmatched_qc_batch_ids=unmatched_qc_batch_ids,
    )


def validate_required_columns(
    dataframe: pd.DataFrame,
    required_columns: list[str],
    label: str,
) -> None:
    """Raise if any selected columns are missing from a DataFrame."""
    missing_columns = [column_name for column_name in required_columns if column_name not in dataframe.columns]
    if missing_columns:
        missing_text = ", ".join(missing_columns)
        raise DataPrepError(f"Missing column(s) in {label}: {missing_text}")


def validate_no_outcome_name_collisions(
    process_dataframe: pd.DataFrame,
    process_batch_id_column: str,
    outcome_columns: list[str],
) -> None:
    """Avoid ambiguous merged columns when QC outcome names already exist in process data."""
    process_columns = set(process_dataframe.columns) - {process_batch_id_column}
    overlapping_columns = sorted(process_columns.intersection(outcome_columns))

    if overlapping_columns:
        overlapping_text = ", ".join(overlapping_columns)
        raise DataPrepError(
            "These selected QC outcome column names also exist in the process file: "
            f"{overlapping_text}. Rename one side before merging so the results are clear."
        )


def normalize_batch_ids(batch_id_series: pd.Series) -> pd.Series:
    """Normalize batch IDs for matching while preserving the displayed columns."""
    return normalize_batch_id_series(batch_id_series)


def validate_batch_ids(batch_id_series: pd.Series, label: str) -> None:
    """Validate missing and duplicate batch IDs before merging."""
    missing_mask = batch_id_series.isna() | (batch_id_series == "")
    if missing_mask.any():
        missing_count = int(missing_mask.sum())
        raise DataPrepError(f"The selected batch ID column in {label} has {missing_count} missing value(s).")

    duplicate_batch_ids = batch_id_series[batch_id_series.duplicated()].drop_duplicates().tolist()
    if duplicate_batch_ids:
        example_ids = ", ".join(duplicate_batch_ids[:5])
        extra_count = max(len(duplicate_batch_ids) - 5, 0)
        extra_text = f" and {extra_count} more" if extra_count else ""
        raise DataPrepError(
            f"The selected batch ID column in {label} contains duplicate IDs: {example_ids}{extra_text}."
        )
