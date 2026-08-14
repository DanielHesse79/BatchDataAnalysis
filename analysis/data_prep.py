"""Data loading and batch matching helpers.

The Streamlit app uses these functions for Phase 2. Keeping them here makes the
merge behavior easier to test without clicking through the UI.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
import io
from pathlib import Path
from typing import Iterable

import pandas as pd

from analysis.normalization import normalize_batch_id_series


INTERNAL_BATCH_KEY = "__batch_id_key"
INTAKE_WARNINGS_ATTRIBUTE = "intake_warnings"

# utf-8-sig also covers plain UTF-8 and strips the BOM Excel writes. cp1252 and
# latin-1 cover European exports containing µ, °C, and accented headers; latin-1
# never fails, so it is the final safety net.
CSV_ENCODING_CANDIDATES: tuple[str, ...] = ("utf-8-sig", "cp1252", "latin-1")
CSV_DELIMITER_CANDIDATES: tuple[str, ...] = (",", ";", "\t", "|")
CSV_SNIFF_LINE_COUNT = 40


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
    batch_id_column: str = "batch_id"


@dataclass(frozen=True)
class CsvReadResult:
    """A CSV read together with what had to be guessed to read it."""

    dataframe: pd.DataFrame
    encoding: str
    delimiter: str
    warnings: list[str] = field(default_factory=list)


def load_tabular_file(uploaded_file) -> pd.DataFrame:
    """Load a CSV or Excel file into a DataFrame.

    `uploaded_file` can be a Streamlit UploadedFile or a local path-like object,
    which keeps this helper usable in simple tests.
    """
    if uploaded_file is None:
        raise DataPrepError("No file was provided.")

    file_name = getattr(uploaded_file, "name", str(uploaded_file))
    file_extension = Path(file_name).suffix.lower()
    read_warnings: list[str] = []

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        if file_extension == ".csv":
            csv_result = read_csv_flexibly(uploaded_file)
            dataframe = csv_result.dataframe
            read_warnings = csv_result.warnings
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
    except DataPrepError:
        raise
    except Exception as error:
        raise DataPrepError(f"Could not read {file_name}: {error}") from error

    dataframe = clean_column_names(dataframe)
    validate_non_empty_dataframe(dataframe, file_name)
    attach_intake_warnings(dataframe, read_warnings)
    return dataframe


def read_csv_flexibly(
    uploaded_file,
    header: int | None = 0,
    nrows: int | None = None,
) -> CsvReadResult:
    """Read a CSV without assuming a comma delimiter or a UTF-8 encoding.

    Semicolon exports (the default in decimal-comma locales) otherwise load as a
    single column with no error at all, and cp1252 exports raise an opaque
    UnicodeDecodeError. Rows with a stray extra field - a title line above the
    table, a trailing total - are tolerated instead of aborting the read.
    """
    raw_bytes = read_file_bytes(uploaded_file)
    text, encoding = decode_file_bytes(raw_bytes)
    delimiter = sniff_csv_delimiter(text)
    read_warnings: list[str] = []

    if encoding != CSV_ENCODING_CANDIDATES[0]:
        read_warnings.append(
            f"The file is not valid UTF-8, so it was read as {encoding}. "
            "Check characters such as µ, °C, and accented column names."
        )
    if delimiter != ",":
        read_warnings.append(
            f"The file uses '{describe_delimiter(delimiter)}' as its column separator, not a comma."
        )

    try:
        dataframe = pd.read_csv(
            io.StringIO(text),
            sep=delimiter,
            header=header,
            nrows=nrows,
        )
    except pd.errors.ParserError:
        dataframe = read_ragged_csv(text, delimiter=delimiter, header=header, nrows=nrows)
        read_warnings.append(
            "Some rows have more columns than the header row. Extra fields were kept "
            "as additional unnamed columns instead of failing the import."
        )

    return CsvReadResult(
        dataframe=dataframe,
        encoding=encoding,
        delimiter=delimiter,
        warnings=read_warnings,
    )


def read_ragged_csv(
    text: str,
    delimiter: str,
    header: int | None,
    nrows: int | None,
) -> pd.DataFrame:
    """Read a CSV whose rows do not all have the same field count."""
    maximum_field_count = count_maximum_csv_fields(text, delimiter)
    dataframe = pd.read_csv(
        io.StringIO(text),
        sep=delimiter,
        header=None,
        names=list(range(maximum_field_count)),
        engine="python",
        nrows=None if header is not None else nrows,
    )

    if header is None:
        return dataframe

    dataframe = promote_header_row(dataframe, header)
    if nrows is not None:
        dataframe = dataframe.head(nrows)
    return dataframe


def promote_header_row(dataframe: pd.DataFrame, header_row_index: int) -> pd.DataFrame:
    """Use one row of a headerless read as the column names."""
    if header_row_index >= len(dataframe.index):
        return dataframe

    header_values = dataframe.iloc[header_row_index].tolist()
    promoted_dataframe = dataframe.iloc[header_row_index + 1 :].reset_index(drop=True)
    promoted_dataframe.columns = [
        f"unnamed_column_{position + 1}" if pd.isna(value) else str(value).strip()
        for position, value in enumerate(header_values)
    ]
    return convert_text_columns_to_numeric(promoted_dataframe)


def convert_text_columns_to_numeric(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Restore numeric dtypes lost by reading a file without a header row."""
    converted_columns: list[pd.Series] = []

    for _, series in dataframe.items():
        numeric_series = pd.to_numeric(series, errors="coerce")
        non_missing_mask = series.notna()
        if non_missing_mask.any() and bool(numeric_series[non_missing_mask].notna().all()):
            converted_columns.append(numeric_series)
        else:
            converted_columns.append(series)

    if not converted_columns:
        return dataframe

    converted_dataframe = pd.concat(converted_columns, axis=1)
    converted_dataframe.columns = list(dataframe.columns)
    return converted_dataframe


def read_file_bytes(uploaded_file) -> bytes:
    """Read raw bytes from an upload stream or a local path."""
    if hasattr(uploaded_file, "read"):
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        raw_content = uploaded_file.read()
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)
        if isinstance(raw_content, str):
            return raw_content.encode("utf-8")
        return bytes(raw_content)

    return Path(str(uploaded_file)).read_bytes()


def decode_file_bytes(raw_bytes: bytes) -> tuple[str, str]:
    """Decode file bytes, falling back from UTF-8 to legacy Windows encodings."""
    for encoding in CSV_ENCODING_CANDIDATES:
        try:
            return raw_bytes.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    return raw_bytes.decode(CSV_ENCODING_CANDIDATES[-1], errors="replace"), CSV_ENCODING_CANDIDATES[-1]


def sniff_csv_delimiter(text: str) -> str:
    """Detect the column separator of a CSV sample, defaulting to a comma."""
    sample_lines = [line for line in text.splitlines() if line.strip()][:CSV_SNIFF_LINE_COUNT]
    if not sample_lines:
        return ","

    try:
        dialect = csv.Sniffer().sniff(
            "\n".join(sample_lines),
            delimiters="".join(CSV_DELIMITER_CANDIDATES),
        )
        if dialect.delimiter in CSV_DELIMITER_CANDIDATES:
            return dialect.delimiter
    except csv.Error:
        pass

    return pick_delimiter_by_field_counts(sample_lines)


def pick_delimiter_by_field_counts(sample_lines: list[str]) -> str:
    """Pick the delimiter that splits the most lines into the same field count."""
    best_delimiter = ","
    best_score = (0, 0)

    for delimiter in CSV_DELIMITER_CANDIDATES:
        field_counts = [
            len(fields) for fields in csv.reader(sample_lines, delimiter=delimiter)
        ]
        repeated_counts = [count for count in field_counts if count >= 2]
        if not repeated_counts:
            continue

        most_common_count = max(set(repeated_counts), key=repeated_counts.count)
        score = (repeated_counts.count(most_common_count), most_common_count)
        if score > best_score:
            best_score = score
            best_delimiter = delimiter

    return best_delimiter


def count_maximum_csv_fields(text: str, delimiter: str) -> int:
    """Count the widest row in a CSV so ragged rows can still be read."""
    maximum_field_count = 1
    for fields in csv.reader(io.StringIO(text), delimiter=delimiter):
        maximum_field_count = max(maximum_field_count, len(fields))
    return maximum_field_count


def describe_delimiter(delimiter: str) -> str:
    """Return a printable name for a delimiter."""
    return {"\t": "tab"}.get(delimiter, delimiter)


def attach_intake_warnings(dataframe: pd.DataFrame, read_warnings: list[str]) -> None:
    """Record how a file had to be read so the UI can surface it later."""
    if read_warnings:
        dataframe.attrs[INTAKE_WARNINGS_ATTRIBUTE] = list(read_warnings)


def get_intake_warnings(dataframe: pd.DataFrame) -> list[str]:
    """Return the intake warnings recorded when a file was read."""
    return list(dataframe.attrs.get(INTAKE_WARNINGS_ATTRIBUTE, []))


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

    # The merged ID column keeps its own name when the process file already has
    # an unrelated column called `batch_id`. Downstream stages must use this
    # name instead of assuming the literal "batch_id", or the real ID is treated
    # as a per-row-unique categorical feature.
    merged_batch_id_column = process_batch_id_column
    if process_batch_id_column != "batch_id" and "batch_id" not in merged_dataframe.columns:
        merged_dataframe = merged_dataframe.rename(columns={process_batch_id_column: "batch_id"})
        merged_batch_id_column = "batch_id"

    return MergeResult(
        dataframe=merged_dataframe,
        matched_batch_count=len(merged_dataframe),
        process_batch_count=len(process_key_set),
        qc_batch_count=len(qc_key_set),
        unmatched_process_batch_ids=unmatched_process_batch_ids,
        unmatched_qc_batch_ids=unmatched_qc_batch_ids,
        batch_id_column=merged_batch_id_column,
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
