from pathlib import Path

import pandas as pd
import pytest

from analysis.intake import (
    IntakeLoadOptions,
    inspect_tabular_file,
    load_intake_dataframe,
    read_csv_preview,
)


def write_csv_file(directory: Path, file_name: str, content: str, encoding: str = "utf-8") -> Path:
    """Write a CSV fixture exactly as a customer export would arrive."""
    file_path = directory / file_name
    file_path.write_bytes(content.encode(encoding))
    return file_path


def test_csv_preview_survives_a_title_line_above_the_table(tmp_path):
    """The header-row suggestion exists for these files, so previewing must not raise."""
    file_path = write_csv_file(
        tmp_path,
        "qc.csv",
        "QC Report Export\nBatch,Result,Unit\nB-001,81.2,pct\nB-002,79.0,pct\n",
    )

    preview_result = read_csv_preview(file_path)

    assert preview_result.dataframe.shape == (4, 3)
    assert preview_result.dataframe.iloc[1].tolist() == ["Batch", "Result", "Unit"]


def test_header_row_is_suggested_below_a_title_line(tmp_path):
    file_path = write_csv_file(
        tmp_path,
        "qc.csv",
        "QC Report Export\nBatch,Result,Unit\nB-001,81.2,pct\nB-002,79.0,pct\n",
    )

    inspection = inspect_tabular_file(file_path)

    assert inspection.suggested_header_row == 1
    assert any("more columns than the header row" in warning for warning in inspection.warnings)


def test_loading_with_the_suggested_header_row_returns_the_real_table(tmp_path):
    file_path = write_csv_file(
        tmp_path,
        "qc.csv",
        "QC Report Export\nBatch,Result,Unit\nB-001,81.2,pct\nB-002,79.0,pct\n",
    )

    dataframe = load_intake_dataframe(file_path, IntakeLoadOptions(header_row=1))

    assert list(dataframe.columns) == ["Batch", "Result", "Unit"]
    assert dataframe["Result"].tolist() == pytest.approx([81.2, 79.0])


def test_semicolon_export_previews_as_separate_columns(tmp_path):
    """A single-column preview hides the real header from the user."""
    file_path = write_csv_file(
        tmp_path,
        "process.csv",
        "batch_id;temperature_C\nB-001;36,8\nB-002;37,1\n",
    )

    inspection = inspect_tabular_file(file_path)

    assert inspection.raw_preview.shape[1] == 2
    assert inspection.suggested_header_row == 0


def test_semicolon_export_loads_with_its_columns_separated(tmp_path):
    file_path = write_csv_file(
        tmp_path,
        "process.csv",
        "batch_id;temperature_C\nB-001;36,8\nB-002;37,1\n",
    )

    dataframe = load_intake_dataframe(file_path, IntakeLoadOptions(header_row=0))

    assert list(dataframe.columns) == ["batch_id", "temperature_C"]
    assert dataframe["temperature_C"].tolist() == ["36,8", "37,1"]


def test_legacy_windows_encoded_export_loads_through_intake(tmp_path):
    file_path = write_csv_file(
        tmp_path,
        "qc.csv",
        "batch_id,temperature_°C\nB-001,36.8\n",
        encoding="cp1252",
    )

    dataframe = load_intake_dataframe(file_path, IntakeLoadOptions(header_row=0))

    assert list(dataframe.columns) == ["batch_id", "temperature_°C"]


def test_excel_files_are_still_inspected_sheet_by_sheet(tmp_path):
    file_path = tmp_path / "process.xlsx"
    with pd.ExcelWriter(file_path) as writer:
        pd.DataFrame({"batch_id": ["B-001"], "temperature_C": [36.8]}).to_excel(
            writer,
            sheet_name="process_data",
            index=False,
        )

    inspection = inspect_tabular_file(file_path)

    assert inspection.sheets == ["process_data"]
    assert inspection.suggested_sheet == "process_data"
