from pathlib import Path

import pandas as pd
import pytest

from analysis.data_prep import (
    DataPrepError,
    get_intake_warnings,
    load_tabular_file,
    merge_process_and_qc_data,
)
from analysis.profiling import profile_merged_data


def write_csv_file(directory: Path, file_name: str, content: str, encoding: str = "utf-8") -> Path:
    """Write a CSV fixture exactly as a customer export would arrive."""
    file_path = directory / file_name
    file_path.write_bytes(content.encode(encoding))
    return file_path


def test_semicolon_delimited_export_loads_as_separate_columns(tmp_path):
    """Decimal-comma locales export with ';'; sniffing nothing gives one column."""
    file_path = write_csv_file(
        tmp_path,
        "process.csv",
        "batch_id;temperature_C;operator_id\nB-001;36,8;07\nB-002;37,1;08\n",
    )

    dataframe = load_tabular_file(file_path)

    assert list(dataframe.columns) == ["batch_id", "temperature_C", "operator_id"]
    assert len(dataframe) == 2
    assert any("separator" in warning for warning in get_intake_warnings(dataframe))


def test_tab_delimited_export_loads_as_separate_columns(tmp_path):
    file_path = write_csv_file(
        tmp_path,
        "process.csv",
        "batch_id\ttemperature_C\nB-001\t36.8\nB-002\t37.1\n",
    )

    dataframe = load_tabular_file(file_path)

    assert list(dataframe.columns) == ["batch_id", "temperature_C"]


def test_comma_delimited_export_is_unaffected_by_delimiter_detection(tmp_path):
    file_path = write_csv_file(
        tmp_path,
        "process.csv",
        "batch_id,temperature_C\nB-001,36.8\nB-002,37.1\n",
    )

    dataframe = load_tabular_file(file_path)

    assert list(dataframe.columns) == ["batch_id", "temperature_C"]
    assert dataframe["temperature_C"].tolist() == pytest.approx([36.8, 37.1])
    assert get_intake_warnings(dataframe) == []


def test_excel_utf8_byte_order_mark_is_stripped_from_the_first_header(tmp_path):
    """Excel's UTF-8 CSV export prefixes the first column name with a BOM."""
    file_path = write_csv_file(
        tmp_path,
        "process.csv",
        "batch_id,temperature_C\nB-001,36.8\n",
        encoding="utf-8-sig",
    )

    dataframe = load_tabular_file(file_path)

    assert list(dataframe.columns) == ["batch_id", "temperature_C"]


def test_legacy_windows_encoded_export_loads_with_readable_headers(tmp_path):
    """cp1252 exports with µ or °C otherwise fail with a generic read error."""
    file_path = write_csv_file(
        tmp_path,
        "qc.csv",
        "batch_id,temperature_°C,conc_µg_mL\nB-001,36.8,4.1\n",
        encoding="cp1252",
    )

    dataframe = load_tabular_file(file_path)

    assert list(dataframe.columns) == ["batch_id", "temperature_°C", "conc_µg_mL"]
    assert any("UTF-8" in warning for warning in get_intake_warnings(dataframe))


def test_merge_reports_the_id_column_it_actually_used(tmp_path):
    """A stray pre-existing `batch_id` column must not become the batch key."""
    process_dataframe = pd.DataFrame(
        {
            "batch_id": ["internal-1", "internal-2"],
            "Lot Number": ["B-001", "B-002"],
            "temperature_C": [36.8, 37.1],
        }
    )
    qc_dataframe = pd.DataFrame({"batch_id": ["B001", "B002"], "yield_g_L": [4.1, 4.4]})

    result = merge_process_and_qc_data(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="Lot Number",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_g_L"],
    )

    assert result.batch_id_column == "Lot Number"

    profile_result = profile_merged_data(
        result.dataframe,
        outcome_columns=["yield_g_L"],
        batch_id_column=result.batch_id_column,
    )
    assert "Lot Number" not in profile_result.process_columns


def test_merge_renames_the_selected_id_column_when_no_collision_exists():
    process_dataframe = pd.DataFrame({"Lot Number": ["B-001"], "temperature_C": [36.8]})
    qc_dataframe = pd.DataFrame({"batch_id": ["B001"], "yield_g_L": [4.1]})

    result = merge_process_and_qc_data(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="Lot Number",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_g_L"],
    )

    assert result.batch_id_column == "batch_id"
    assert result.dataframe["batch_id"].tolist() == ["B-001"]


def test_merge_matches_normalized_batch_ids_without_changing_display_id():
    process_dataframe = pd.DataFrame(
        {
            "Batch ID": ["B-001", "B-002"],
            "temperature_C": [36.8, 37.1],
        }
    )
    qc_dataframe = pd.DataFrame(
        {
            "batch_id": ["B001", "B002"],
            "yield_g_L": [4.1, 4.4],
        }
    )

    result = merge_process_and_qc_data(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="Batch ID",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_g_L"],
    )

    assert result.matched_batch_count == 2
    assert result.dataframe["batch_id"].tolist() == ["B-001", "B-002"]


def test_merge_rejects_duplicate_normalized_batch_ids():
    process_dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B001"],
            "temperature_C": [36.8, 37.1],
        }
    )
    qc_dataframe = pd.DataFrame({"batch_id": ["B001"], "yield_g_L": [4.1]})

    with pytest.raises(DataPrepError, match="duplicate IDs"):
        merge_process_and_qc_data(
            process_dataframe=process_dataframe,
            qc_dataframe=qc_dataframe,
            process_batch_id_column="batch_id",
            qc_batch_id_column="batch_id",
            outcome_columns=["yield_g_L"],
        )


def test_merge_rejects_outcome_name_collision_with_process_columns():
    process_dataframe = pd.DataFrame(
        {
            "batch_id": ["B001"],
            "yield_g_L": [999.0],
        }
    )
    qc_dataframe = pd.DataFrame({"batch_id": ["B001"], "yield_g_L": [4.1]})

    with pytest.raises(DataPrepError, match="also exist in the process file"):
        merge_process_and_qc_data(
            process_dataframe=process_dataframe,
            qc_dataframe=qc_dataframe,
            process_batch_id_column="batch_id",
            qc_batch_id_column="batch_id",
            outcome_columns=["yield_g_L"],
        )
