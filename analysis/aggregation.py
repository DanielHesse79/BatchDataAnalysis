"""Duplicate and long-format handling for field data intake."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analysis.data_prep import DataPrepError, attach_intake_warnings
from analysis.normalization import (
    build_missing_like_mask,
    make_internal_column_name,
    is_missing_like,
    normalize_batch_id_series,
    parse_numeric_series,
    should_attempt_numeric_parse,
)


NORMALIZED_KEY_COLUMN = "__normalized_batch_key"
ROW_POSITION_COLUMN = "__row_position"
LONG_FORMAT_SAMPLE_ROWS = 200

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

    return count_duplicate_keys(normalize_batch_id_series(dataframe[batch_id_column]))


def count_duplicate_keys(batch_keys: pd.Series) -> int:
    """Count duplicate batch IDs from keys that are already normalized.

    Callers that already hold normalized keys use this to avoid normalizing the
    same column several times per screen refresh.
    """
    duplicate_keys = batch_keys[batch_keys.notna() & batch_keys.duplicated()].drop_duplicates()
    return int(len(duplicate_keys))


def get_duplicate_batch_id_examples(
    dataframe: pd.DataFrame,
    batch_id_column: str,
    max_rows: int = 10,
) -> pd.DataFrame:
    """Return duplicate normalized batch IDs with their row counts.

    Uses the same key filter as ``count_duplicate_batch_ids`` so the warning and
    the example table can never disagree about what counts as a duplicate.
    """
    empty_examples = pd.DataFrame(columns=["normalized_batch_id", "row_count"])
    if batch_id_column not in dataframe.columns:
        return empty_examples

    batch_keys = normalize_batch_id_series(dataframe[batch_id_column])
    counts = batch_keys[batch_keys.notna()].value_counts()
    duplicate_counts = counts[counts > 1].head(max_rows)
    if duplicate_counts.empty:
        return empty_examples

    return pd.DataFrame(
        {
            "normalized_batch_id": duplicate_counts.index.astype(str),
            "row_count": duplicate_counts.to_numpy(dtype=int),
        }
    )


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
    duplicate_count = count_duplicate_keys(batch_keys)
    warnings: list[str] = []

    # Rows without a usable batch ID are not duplicates of each other. Grouping
    # them would merge unrelated batches into one fabricated row.
    missing_key_mask = batch_keys.isna()
    missing_key_count = int(missing_key_mask.sum())
    if missing_key_count:
        warnings.append(
            f"{label}: {missing_key_count:,} row(s) have no usable batch ID. "
            "They were left unchanged and cannot be matched to the other file."
        )

    if duplicate_count == 0:
        return DuplicateResolutionResult(
            dataframe=output_dataframe,
            duplicate_batch_count=0,
            rows_before=len(output_dataframe),
            rows_after=len(output_dataframe),
            warnings=warnings,
        )

    if strategy == "error":
        examples = ", ".join(
            batch_keys[batch_keys.notna() & batch_keys.duplicated()].drop_duplicates().head(5)
        )
        raise DataPrepError(
            f"{label}: duplicate batch IDs are still present ({examples}). "
            "Choose a duplicate handling rule or fix the source file."
        )

    # A real export can contain a column called `__normalized_batch_key`, which
    # would be overwritten here and dropped below, losing user data.
    key_column = make_internal_column_name(NORMALIZED_KEY_COLUMN, dataframe.columns)
    output_dataframe[key_column] = batch_keys
    identified_rows = output_dataframe[~missing_key_mask]
    unidentified_rows = output_dataframe[missing_key_mask]

    if strategy == "keep_first":
        resolved_dataframe = identified_rows.drop_duplicates(key_column, keep="first")
    elif strategy == "keep_last":
        resolved_dataframe = identified_rows.drop_duplicates(key_column, keep="last")
    else:
        resolved_dataframe = aggregate_numeric_duplicates(
            identified_rows,
            key_column=key_column,
            strategy=strategy,
        )

    if not unidentified_rows.empty:
        resolved_dataframe = pd.concat(
            [resolved_dataframe, unidentified_rows],
            ignore_index=True,
        )

    output_dataframe = resolved_dataframe.drop(columns=[key_column], errors="ignore")
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
    """Aggregate duplicate groups, summarizing numeric columns and keeping first text.

    Numeric-versus-text is decided once per column, not once per group, and only
    the duplicated keys are rebuilt. Rows that already have a unique key are
    carried through untouched, which keeps large exports out of a Python loop.
    """
    if strategy not in {"mean", "median"}:
        raise DataPrepError(f"Unknown numeric duplicate strategy: {strategy}")

    working_dataframe = dataframe.reset_index(drop=True)
    duplicated_mask = working_dataframe[key_column].duplicated(keep=False)
    if not duplicated_mask.any():
        return working_dataframe

    value_columns = [
        column_name for column_name in working_dataframe.columns if column_name != key_column
    ]
    working_dataframe, numeric_columns = prepare_numeric_columns_for_aggregation(
        working_dataframe,
        value_columns,
    )
    row_position_column = make_internal_column_name(
        ROW_POSITION_COLUMN,
        working_dataframe.columns,
    )
    working_dataframe[row_position_column] = range(len(working_dataframe))
    aggregation_map: dict[str, Any] = {
        column_name: (strategy if column_name in numeric_columns else "first")
        for column_name in value_columns
    }
    aggregation_map[row_position_column] = "min"

    aggregated_rows = (
        working_dataframe[duplicated_mask]
        .groupby(key_column, dropna=False, sort=False)
        .agg(aggregation_map)
        .reset_index()
    )
    combined_dataframe = pd.concat(
        [working_dataframe[~duplicated_mask], aggregated_rows],
        ignore_index=True,
    )
    combined_dataframe = combined_dataframe.sort_values(row_position_column).reset_index(drop=True)
    return combined_dataframe[list(dataframe.columns)]


def prepare_numeric_columns_for_aggregation(
    dataframe: pd.DataFrame,
    value_columns: list[str],
    min_parse_fraction: float = 0.75,
) -> tuple[pd.DataFrame, list[str]]:
    """Decide per column whether replicate values may be averaged.

    Deciding per group turns numeric-looking identifier codes into numbers:
    operator "07" and "08" average to 7.5, which then reaches confounding checks
    and models as a continuous variable.
    """
    prepared_dataframe = dataframe.copy()
    numeric_columns: list[str] = []

    for column_name in value_columns:
        series = prepared_dataframe[column_name]
        if pd.api.types.is_bool_dtype(series):
            continue

        if pd.api.types.is_numeric_dtype(series):
            numeric_columns.append(column_name)
            continue

        if not should_attempt_numeric_parse(column_name, series):
            continue

        non_missing_count = int((~build_missing_like_mask(series)).sum())
        if non_missing_count == 0:
            continue

        parsed_values, _ = parse_numeric_series(series)
        if int(parsed_values.notna().sum()) / non_missing_count >= min_parse_fraction:
            prepared_dataframe[column_name] = parsed_values
            numeric_columns.append(column_name)

    return prepared_dataframe, numeric_columns


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
        if column_looks_numeric_in_sample(dataframe[column]):
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


def column_looks_numeric_in_sample(
    series: pd.Series,
    sample_rows: int = LONG_FORMAT_SAMPLE_ROWS,
    min_parse_fraction: float = 0.80,
) -> bool:
    """Return whether a sample of a text column parses as numbers.

    This only *suggests* columns in the UI, so it samples. Running the per-cell
    parser over every column of a 100k-row export costs millions of regex calls
    before the user has chosen anything.
    """
    non_missing_values = series.dropna().head(sample_rows)
    if non_missing_values.empty:
        return False

    parsed_values, _ = parse_numeric_series(non_missing_values)
    return int(parsed_values.notna().sum()) / len(non_missing_values) >= min_parse_fraction


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

    if pivot_frame.empty:
        return pd.DataFrame(columns=[batch_id_column])

    # Pivot on normalized keys. Pivoting on raw IDs turns "B-001" and "B001"
    # into two half-filled rows, and duplicate handling later discards one
    # spelling's results entirely.
    batch_keys = normalize_batch_id_series(pivot_frame[batch_id_column])
    display_batch_ids = build_display_batch_id_lookup(pivot_frame[batch_id_column], batch_keys)
    pivot_frame[batch_id_column] = batch_keys

    # Rows without a usable batch ID share no key, so pivoting them together
    # would fabricate one batch out of unrelated results.
    missing_key_mask = batch_keys.isna()
    unidentified_row_count = int(missing_key_mask.sum())
    pivot_warnings: list[str] = []
    if unidentified_row_count:
        pivot_frame = pivot_frame[~missing_key_mask]
        pivot_warnings.append(
            f"{unidentified_row_count:,} long-format row(s) have no usable batch ID and "
            "could not be assigned to a batch."
        )

    if pivot_frame.empty:
        empty_pivot_table = pd.DataFrame(columns=[batch_id_column])
        attach_intake_warnings(empty_pivot_table, pivot_warnings)
        return empty_pivot_table

    # Decide numeric conversion per test, not for the whole value column. A QC
    # export mixes numeric assays with qualitative results ("Conforms"); a
    # column-wide decision would blank out one kind or the other.
    numeric_test_names, conversion_warnings = select_numeric_test_names(
        pivot_frame,
        name_column,
        value_column,
    )
    pivot_warnings.extend(conversion_warnings)
    numeric_mask = pivot_frame[name_column].isin(numeric_test_names)

    pivot_tables: list[pd.DataFrame] = []

    if numeric_mask.any():
        numeric_frame = pivot_frame[numeric_mask].copy()
        parsed_values, _ = parse_numeric_series(numeric_frame[value_column])
        numeric_frame[value_column] = parsed_values
        pivot_tables.append(
            pivot_value_frame(
                pivot_frame=numeric_frame,
                batch_id_column=batch_id_column,
                name_column=name_column,
                value_column=value_column,
                aggregation=aggregation,
            )
        )

    if (~numeric_mask).any():
        # Text results cannot be averaged, so they always keep the first value.
        pivot_tables.append(
            pivot_value_frame(
                pivot_frame=pivot_frame[~numeric_mask].copy(),
                batch_id_column=batch_id_column,
                name_column=name_column,
                value_column=value_column,
                aggregation="first",
            )
        )

    pivot_table = pivot_tables[0]
    for extra_table in pivot_tables[1:]:
        pivot_table = pivot_table.join(extra_table, how="outer")

    pivot_table = pivot_table.reset_index()
    pivot_table[batch_id_column] = (
        pivot_table[batch_id_column].map(display_batch_ids).fillna(pivot_table[batch_id_column])
    )
    pivot_table.columns = [str(column_name).strip() for column_name in pivot_table.columns]
    attach_intake_warnings(pivot_table, pivot_warnings)
    return pivot_table


def build_display_batch_id_lookup(
    raw_batch_ids: pd.Series,
    batch_keys: pd.Series,
) -> dict[str, Any]:
    """Map each normalized batch key back to the first raw spelling seen."""
    lookup: dict[str, Any] = {}
    for raw_value, key_value in zip(raw_batch_ids, batch_keys):
        if pd.isna(key_value) or key_value in lookup:
            continue
        lookup[key_value] = raw_value
    return lookup


def select_numeric_test_names(
    pivot_frame: pd.DataFrame,
    name_column: str,
    value_column: str,
    min_parse_fraction: float = 0.60,
) -> tuple[set[str], list[str]]:
    """Return the test names whose results are mostly numeric, plus loss warnings.

    The fraction is compared directly rather than through a rounded row count.
    Flooring the requirement let a single numeric value among three
    (``int(0.60 * 3) == 1``) convert an entire qualitative test, turning
    "Pass"/"Fail" into missing values.
    """
    numeric_test_names: set[str] = set()
    warnings: list[str] = []

    for test_name, group in pivot_frame.groupby(name_column, sort=False):
        values = group[value_column]
        present_values = values[~values.map(is_missing_like)]
        if present_values.empty:
            continue

        parsed_values, _ = parse_numeric_series(present_values)
        parsed_count = int(parsed_values.notna().sum())
        if parsed_count / len(present_values) < min_parse_fraction:
            continue

        numeric_test_names.add(str(test_name))

        # A mostly-numeric test can still carry text results. Converting drops
        # them, so say which ones rather than losing them silently.
        unparsed_values = present_values[parsed_values.isna()]
        if not unparsed_values.empty:
            examples = ", ".join(unparsed_values.astype(str).drop_duplicates().head(3))
            warnings.append(
                f"Test '{test_name}' is mostly numeric, so {len(unparsed_values):,} "
                f"non-numeric result(s) became missing (examples: {examples})."
            )

    return numeric_test_names, warnings


def pivot_value_frame(
    pivot_frame: pd.DataFrame,
    batch_id_column: str,
    name_column: str,
    value_column: str,
    aggregation: str,
) -> pd.DataFrame:
    """Pivot one long frame into wide form, keeping all-missing tests visible."""
    if aggregation in {"mean", "median"}:
        aggregation_function = aggregation
    else:
        aggregation_function = (
            lambda values: values.dropna().iloc[0] if values.dropna().size else pd.NA
        )

    return pivot_frame.pivot_table(
        index=batch_id_column,
        columns=name_column,
        values=value_column,
        aggfunc=aggregation_function,
        dropna=False,
    )
