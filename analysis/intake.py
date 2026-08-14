"""File inspection and loading for field-style tabular exports."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from analysis.data_prep import (
    CsvReadResult,
    DataPrepError,
    attach_intake_warnings,
    read_csv_flexibly,
    validate_non_empty_dataframe,
)


PREVIEW_ROWS = 20


@dataclass(frozen=True)
class FileInspection:
    """What the app could detect before loading a file canonically."""

    file_name: str
    file_extension: str
    sheets: list[str]
    suggested_sheet: str | None
    suggested_header_row: int
    raw_preview: pd.DataFrame
    warnings: list[str]


@dataclass(frozen=True)
class IntakeLoadOptions:
    """User-confirmed file loading choices."""

    sheet_name: str | None = None
    header_row: int = 0


def inspect_tabular_file(uploaded_file) -> FileInspection:
    """Inspect a CSV/XLS/XLSX file before normal loading."""
    if uploaded_file is None:
        raise DataPrepError("No file was provided.")

    file_name = getattr(uploaded_file, "name", str(uploaded_file))
    file_extension = Path(file_name).suffix.lower()
    warnings: list[str] = []

    try:
        if file_extension == ".csv":
            preview_result = read_csv_preview(uploaded_file)
            raw_preview = preview_result.dataframe
            warnings.extend(preview_result.warnings)
            sheets: list[str] = []
            suggested_sheet = None
        elif file_extension in {".xlsx", ".xls"}:
            excel_file = build_excel_file(uploaded_file)
            sheets = excel_file.sheet_names
            if not sheets:
                raise DataPrepError(f"{file_name} does not contain any sheets.")
            suggested_sheet = sheets[0]
            raw_preview = pd.read_excel(
                excel_file,
                sheet_name=suggested_sheet,
                header=None,
                nrows=PREVIEW_ROWS,
            )
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
        raise DataPrepError(f"Could not inspect {file_name}: {error}") from error

    if raw_preview.empty:
        warnings.append("The preview is empty. Check the selected sheet or file export.")

    suggested_header_row = suggest_header_row(raw_preview)
    return FileInspection(
        file_name=file_name,
        file_extension=file_extension,
        sheets=sheets,
        suggested_sheet=suggested_sheet,
        suggested_header_row=suggested_header_row,
        raw_preview=raw_preview,
        warnings=warnings,
    )


def load_intake_dataframe(
    uploaded_file,
    options: IntakeLoadOptions,
) -> pd.DataFrame:
    """Load a file using user-confirmed intake options."""
    if uploaded_file is None:
        raise DataPrepError("No file was provided.")

    file_name = getattr(uploaded_file, "name", str(uploaded_file))
    file_extension = Path(file_name).suffix.lower()
    read_warnings: list[str] = []

    try:
        if hasattr(uploaded_file, "seek"):
            uploaded_file.seek(0)

        if file_extension == ".csv":
            csv_result = read_csv_flexibly(uploaded_file, header=options.header_row)
            dataframe = csv_result.dataframe
            read_warnings = csv_result.warnings
        elif file_extension in {".xlsx", ".xls"}:
            dataframe = pd.read_excel(
                uploaded_file,
                sheet_name=options.sheet_name or 0,
                header=options.header_row,
            )
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

    validate_non_empty_dataframe(dataframe, file_name)
    attach_intake_warnings(dataframe, read_warnings)
    return dataframe


def read_csv_preview(uploaded_file) -> CsvReadResult:
    """Read a raw CSV preview without assuming which row is the header.

    Field exports often start with a title line above the table, which is the
    exact case the header-row suggestion exists for. Reading it must not raise.
    """
    return read_csv_flexibly(uploaded_file, header=None, nrows=PREVIEW_ROWS)


def build_excel_file(uploaded_file) -> pd.ExcelFile:
    """Build an ExcelFile object while handling Streamlit upload streams."""
    if hasattr(uploaded_file, "seek"):
        uploaded_file.seek(0)
    return pd.ExcelFile(uploaded_file)


def suggest_header_row(raw_preview: pd.DataFrame) -> int:
    """Pick the row most likely to contain column names."""
    if raw_preview.empty:
        return 0

    best_row_index = 0
    best_score = -1.0
    for row_index, row in raw_preview.iterrows():
        score = score_header_candidate(row)
        if score > best_score:
            best_score = score
            best_row_index = int(row_index)

    return best_row_index


def score_header_candidate(row: pd.Series) -> float:
    """Score a potential header row."""
    values = [value for value in row.tolist() if not pd.isna(value)]
    if not values:
        return -1.0

    text_values = [str(value).strip() for value in values if str(value).strip()]
    if not text_values:
        return -1.0

    unique_fraction = len(set(text_values)) / len(text_values)
    text_fraction = sum(not looks_numeric(value) for value in text_values) / len(text_values)
    useful_name_bonus = 0.0
    lowered = " ".join(value.lower() for value in text_values)
    for token in ["batch", "lot", "id", "result", "value", "test", "temp", "ph"]:
        if token in lowered:
            useful_name_bonus += 0.25

    return len(text_values) + (2.0 * unique_fraction) + text_fraction + useful_name_bonus


def looks_numeric(value: Any) -> bool:
    """Return whether a value looks primarily numeric."""
    try:
        float(str(value).replace(",", "."))
        return True
    except ValueError:
        return False
