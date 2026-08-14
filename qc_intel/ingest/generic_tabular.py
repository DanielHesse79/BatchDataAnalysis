"""Generic tabular adapter: CSV/TSV/XLSX to the canonical schema.

Adapters exist so analytical logic never learns a file format. A MassLynx or
UNIFI adapter added later must produce the same canonical rows and nothing
downstream changes.

Every imported row carries its ingest_id and a source_row string, so any number
on a chart can be traced back to the file and record that produced it.
"""

from __future__ import annotations

from pathlib import Path
import sqlite3

import pandas as pd

from qc_intel.db import DuplicateSourceError, insert_frame, register_ingest


ADAPTER_NAME = "generic_tabular"

# Canonical column sets. Extra columns in a source file are ignored rather than
# guessed at, so a vendor adding a column cannot silently change a statistic.
TABLE_COLUMNS = {
    "method": [
        "method_id", "method_name", "method_version", "analyte", "matrix",
        "platform", "units", "lloq", "uloq", "effective_date", "retired_date",
    ],
    "instrument": [
        "instrument_id", "instrument_model", "asset_tag", "installation_date",
        "software_version",
    ],
    "material_lot": [
        "material_id", "material_type", "manufacturer", "lot_number",
        "date_opened", "expiry_date", "concentration",
    ],
    "analytical_run": [
        "run_id", "method_id", "instrument_id", "study_id", "acquisition_timestamp",
        "run_type", "acceptance_status", "acceptance_reason", "column_id",
        "column_injection_count", "processing_method_version", "analyst_ref",
        "ingest_id", "source_row",
    ],
    "run_material": ["run_id", "material_id", "role"],
    "qc_result": [
        "run_id", "qc_level", "evaluation_type", "nominal_value", "measured_value",
        "units", "percent_bias", "replicate_number", "injection_index",
        "analyte_area", "is_area", "area_ratio", "retention_time",
        "manual_integration", "pass_fail", "acceptance_limit_lower",
        "acceptance_limit_upper", "ingest_id", "source_row",
    ],
    "calibration": [
        "calibration_id", "run_id", "calibration_model", "weighting", "slope",
        "intercept", "r_squared", "calibration_status", "number_of_calibrators",
        "ingest_id",
    ],
    "calibrator_point": [
        "calibration_id", "level_name", "nominal_value", "back_calculated",
        "percent_deviation", "included",
    ],
    "lab_event": [
        "event_id", "entity_type", "entity_id", "event_type", "event_timestamp",
        "description", "ingest_id",
    ],
}

REQUIRED_COLUMNS = {
    "analytical_run": ["run_id", "method_id", "instrument_id", "acquisition_timestamp"],
    "qc_result": ["run_id", "qc_level", "measured_value"],
    "method": ["method_id", "method_name", "method_version"],
    "instrument": ["instrument_id"],
}


class IngestError(ValueError):
    """Raised when a source file cannot be imported."""


def read_tabular(path: Path | str) -> pd.DataFrame:
    """Read a CSV, TSV or Excel file into a DataFrame."""
    path = Path(path)
    suffix = path.suffix.lower()

    try:
        if suffix in {".xlsx", ".xls"}:
            return pd.read_excel(path)
        if suffix == ".tsv":
            return pd.read_csv(path, sep="\t")
        return pd.read_csv(path)
    except Exception as error:
        raise IngestError(f"Could not read {path.name}: {error}") from error


def validate_frame(frame: pd.DataFrame, table: str, source_name: str) -> None:
    """Reject a file that cannot populate the table it claims to."""
    if table not in TABLE_COLUMNS:
        raise IngestError(f"Unknown target table '{table}'.")

    if frame.empty:
        raise IngestError(f"{source_name} contains no rows.")

    missing = [
        column for column in REQUIRED_COLUMNS.get(table, []) if column not in frame.columns
    ]
    if missing:
        raise IngestError(
            f"{source_name} is missing required column(s) for {table}: {', '.join(missing)}."
        )


def align_to_schema(frame: pd.DataFrame, table: str) -> pd.DataFrame:
    """Keep the canonical columns, in schema order, adding absent ones as null."""
    aligned = pd.DataFrame(index=frame.index)
    for column in TABLE_COLUMNS[table]:
        aligned[column] = frame[column] if column in frame.columns else None
    return aligned


def import_table(
    connection: sqlite3.Connection,
    path: Path | str,
    table: str,
    skip_if_imported: bool = False,
) -> int:
    """Import one source file into one canonical table.

    Returns the number of rows written. Raises DuplicateSourceError if the same
    file has already been imported by this adapter, unless skipping is allowed.
    """
    path = Path(path)
    frame = read_tabular(path)
    validate_frame(frame, table, path.name)

    try:
        ingest_record = register_ingest(
            connection,
            source_file=path,
            source_format=path.suffix.lstrip(".").lower(),
            adapter=f"{ADAPTER_NAME}:{table}",
            row_count=len(frame),
        )
    except DuplicateSourceError:
        if skip_if_imported:
            return 0
        raise

    aligned = align_to_schema(frame, table)
    if "ingest_id" in aligned.columns:
        aligned["ingest_id"] = ingest_record.ingest_id
    if "source_row" in aligned.columns:
        aligned["source_row"] = [
            existing if pd.notna(existing) else f"{path.name}:{index + 2}"
            for index, existing in enumerate(aligned["source_row"])
        ]

    return insert_frame(connection, table, aligned, replace_existing=True)
