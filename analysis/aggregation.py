"""Duplicate and long-format handling for field data intake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analysis.data_prep import DataPrepError
from analysis.normalization import normalize_batch_id_series, parse_numeric_series


DUPLICATE_STRATEGIES = {
    "error": "Stop and let me fix duplicates",
    "keep_first": "Keep first row per batch",
    "keep_last": "Keep last row per batch",
    "mean": "Average numeric replicate rows",
    "median": "Median numeric replicate rows",
}

LONG_FORMAT_AGGREGATIONS = {
    "first": "Keep first value",
    "mean": "Average numeric values",
    "median": "Median numeric values",
}


@dataclass(frozen=True)
class DuplicateResolutionResult:
    """A DataFrame after duplicate handling plus a resolution log."""

    dataframe: pd.DataFrame
    duplicate_batch_count: int
    rows_before: int
    rows_after: int
    warnings: list[str]


@dataclass(frozen=True)
class LongFormatSuggestion:
    """Likely columns for a long-format table."""

    batch_id_candidates: list[str]
    name_candidates: list[str]
    value_candidates: list[str]
    likely_long_format: bool


def count_duplicate_batch_ids(dataframe: pd.DataFrame, batch_id_column: str) -> int:
    """Count duplicate normalized batch IDs."""
    if batch_id_column not in dataframe.columns:
        return 0

    batch_keys = normalize_batch_id_series(dataframe[batch_id_column])
    duplicate_keys = batch_keys[batch_keys.notna() & batch_keys.duplicated()].drop_duplicates()
    return int(len(duplicate_keys))


def aggregate_duplicate_batch_rows(
    dataframe: pd.DataFrame,
    batch_id_column: str,
    strategy: str,
    label: str,
) -> DuplicateResolutionResult:
    """Resolve duplicate batch rows according to a user-selected strategy."""
    if batch_id_column not in dataframe.columns:
        raise DataPrepError(f"{label}: selected batch ID column is missing.")

    if strategy not in DUPLICATE_STRATEGIES:
        raise DataPrepError(f"{label}: unknown duplicate handling strategy: {strategy}")

    output_dataframe = dataframe.copy()
    batch_keys = normalize_batch_id_series(output_dataframe[batch_id_column])
    duplicate_count = count_duplicate_batch_ids(output_dataframe, batch_id_column)
    warnings: list[str] = []

    if duplicate_count == 0:
        return DuplicateResolutionResult(
            dataframe=output_dataframe,
            duplicate_batch_count=0,
            rows_before=len(output_dataframe),
            rows_after=len(output_dataframe),
            warnings=[],
        )

    if strategy == "error":
        examples = ", ".join(
            batch_keys[batch_keys.notna() & batch_keys.duplicated()].drop_duplicates().head(5)
        )
        raise DataPrepError(
            f"{label}: duplicate batch IDs are still present ({examples}). "
            "Choose a duplicate handling rule or fix the source file."
        )

    output_dataframe["__normalized_batch_key"] = batch_keys

    if strategy == "keep_first":
        output_dataframe = output_dataframe.drop_duplicates("__normalized_batch_key", keep="first")
    elif strategy == "keep_last":
        output_dataframe = output_dataframe.drop_duplicates("__normalized_batch_key", keep="last")
    else:
        output_dataframe = aggregate_numeric_duplicates(
            output_dataframe,
            key_column="__normalized_batch_key",
            strategy=strategy,
        )

    output_dataframe = output_dataframe.drop(columns=["__normalized_batch_key"], errors="ignore")
    warnings.append(
        f"{label}: resolved {duplicate_count} duplicate batch ID(s) using '{DUPLICATE_STRATEGIES[strategy]}'."
    )
    return DuplicateResolutionResult(
        dataframe=output_dataframe.reset_index(drop=True),
        duplicate_batch_count=duplicate_count,
        rows_before=len(dataframe),
        rows_after=len(output_dataframe),
        warnings=warnings,
    )


def aggregate_numeric_duplicates(
    dataframe: pd.DataFrame,
    key_column: str,
    strategy: str,
) -> pd.DataFrame:
    """Aggregate duplicate groups, summarizing numeric columns and keeping first text."""
    rows: list[dict[str, Any]] = []
    grouped = dataframe.groupby(key_column, dropna=False, sort=False)

    for _, group in grouped:
        row: dict[str, Any] = {}
        for column_name in dataframe.columns:
            if column_name == key_column:
                row[column_name] = group[column_name].iloc[0]
                continue

            series = group[column_name]
            numeric_series = pd.to_numeric(series, errors="coerce")
            if numeric_series.notna().sum() > 0 and numeric_series.notna().sum() >= max(1, len(series) // 2):
                row[column_name] = (
                    numeric_series.mean() if strategy == "mean" else numeric_series.median()
                )
            else:
                non_missing = series.dropna()
                row[column_name] = non_missing.iloc[0] if not non_missing.empty else pd.NA
        rows.append(row)

    return pd.DataFrame(rows)


def detect_long_format_candidates(dataframe: pd.DataFrame) -> LongFormatSuggestion:
    """Suggest long-format batch/test/value columns."""
    columns = list(dataframe.columns)
    lowered = {column: column.lower().replace(" ", "_") for column in columns}

    batch_candidates = [
        column
        for column in columns
        if any(token in lowered[column] for token in ["batch", "lot_id", "run_id"])
    ]
    name_candidates = [
        column
        for column in columns
        if any(token in lowered[column] for token in ["test", "parameter", "attribute", "analyte", "tag", "name"])
        and column not in batch_candidates
    ]
    value_candidates = [
        column
        for column in columns
        if any(token in lowered[column] for token in ["result", "value", "reading", "measurement"])
        and "date" not in lowered[column]
        and column not in batch_candidates
        and column not in name_candidates
    ]

    for column in columns:
        if column in value_candidates or column in batch_candidates or column in name_candidates:
            continue
        if "date" in lowered[column]:
            continue
        if pd.api.types.is_numeric_dtype(dataframe[column]):
            value_candidates.append(column)
            continue
        parsed_values, _ = parse_numeric_series(dataframe[column])
        non_missing = dataframe[column].notna().sum()
        if non_missing and parsed_values.notna().sum() / non_missing >= 0.80:
            value_candidates.append(column)

    likely_long = False
    if batch_candidates and name_candidates and value_candidates:
        first_batch = batch_candidates[0]
        first_name = name_candidates[0]
        likely_long = (
            dataframe[first_batch].nunique(dropna=True) < len(dataframe)
            and dataframe[first_name].nunique(dropna=True) >= 2
        )

    return LongFormatSuggestion(
        batch_id_candidates=batch_candidates,
        name_candidates=name_candidates,
        value_candidates=value_candidates,
        likely_long_format=likely_long,
    )


def pivot_long_to_wide(
    dataframe: pd.DataFrame,
    batch_id_column: str,
    name_column: str,
    value_column: str,
    aggregation: str,
) -> pd.DataFrame:
    """Pivot a long batch/test/value table into one row per batch."""
    for column_name in [batch_id_column, name_column, value_column]:
        if column_name not in dataframe.columns:
            raise DataPrepError(f"Long-format pivot column is missing: {column_name}")

    if aggregation not in LONG_FORMAT_AGGREGATIONS:
        raise DataPrepError(f"Unknown long-format aggregation: {aggregation}")

    pivot_frame = dataframe[[batch_id_column, name_column, value_column]].copy()
    pivot_frame[name_column] = pivot_frame[name_column].astype(str).str.strip()
    parsed_values, _ = parse_numeric_series(pivot_frame[value_column])
    if parsed_values.notna().sum() >= max(1, int(0.60 * pivot_frame[value_column].notna().sum())):
        pivot_frame[value_column] = parsed_values

    if aggregation in {"mean", "median"}:
        pivot_table = pivot_frame.pivot_table(
            index=batch_id_column,
            columns=name_column,
            values=value_column,
            aggfunc=aggregation,
        )
    else:
        pivot_table = pivot_frame.pivot_table(
            index=batch_id_column,
            columns=name_column,
            values=value_column,
            aggfunc=lambda values: values.dropna().iloc[0] if values.dropna().size else pd.NA,
        )

    pivot_table = pivot_table.reset_index()
    pivot_table.columns = [str(column_name).strip() for column_name in pivot_table.columns]
    return pivot_table
