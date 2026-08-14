"""Canonical import templates offered to users.

These shapes define what the app asks a production or QC team to export, so
they belong beside the analysis code that consumes them rather than in the UI.
"""

from __future__ import annotations

from io import BytesIO

import pandas as pd

from analysis.specs import build_default_spec_template


PROCESS_SHEET_NAME = "process_data"
QC_WIDE_SHEET_NAME = "qc_results_wide"
QC_LONG_SHEET_NAME = "qc_results_long"
SPEC_SHEET_NAME = "specs_windows"


def build_process_template_dataframe() -> pd.DataFrame:
    """Return a simple process-data template."""
    return pd.DataFrame(
        [
            {
                "batch_id": "B001",
                "batch_sequence_number": 1,
                "production_date": "2026-01-02",
                "sodium_hydroxide_g": 100.0,
                "temperature_C": 37.0,
                "duration_hours": 168,
                "reactor_id": "RX-1",
                "operator_shift": "Day",
                "raw_material_lot": "RM-001",
            },
            {
                "batch_id": "B002",
                "batch_sequence_number": 2,
                "production_date": "2026-01-09",
                "sodium_hydroxide_g": 101.5,
                "temperature_C": 37.1,
                "duration_hours": 166,
                "reactor_id": "RX-2",
                "operator_shift": "Night",
                "raw_material_lot": "RM-001",
            },
        ]
    )


def build_qc_wide_template_dataframe() -> pd.DataFrame:
    """Return a one-row-per-batch QC template."""
    return pd.DataFrame(
        [
            {
                "batch_id": "B001",
                "yield_percent": 81.2,
                "purity_percent": 96.4,
                "moisture_percent": 12.1,
                "hcp_ppm": 145,
            },
            {
                "batch_id": "B002",
                "yield_percent": 80.6,
                "purity_percent": 95.9,
                "moisture_percent": 11.8,
                "hcp_ppm": 151,
            },
        ]
    )


def build_qc_long_template_dataframe() -> pd.DataFrame:
    """Return a long-format QC template."""
    return pd.DataFrame(
        [
            {"batch_id": "B001", "test_name": "yield_percent", "result": 81.2, "unit": "%"},
            {"batch_id": "B001", "test_name": "moisture_percent", "result": 12.1, "unit": "%"},
            {"batch_id": "B002", "test_name": "yield_percent", "result": 80.6, "unit": "%"},
            {"batch_id": "B002", "test_name": "moisture_percent", "result": 11.8, "unit": "%"},
        ]
    )


def build_combined_workbook_template_bytes(
    process_template: pd.DataFrame,
    qc_wide_template: pd.DataFrame,
    qc_long_template: pd.DataFrame,
    spec_template: pd.DataFrame | None = None,
) -> bytes:
    """Build an Excel workbook with separate process/QC/spec sheets."""
    if spec_template is None:
        spec_template = build_default_spec_template()

    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        process_template.to_excel(writer, sheet_name=PROCESS_SHEET_NAME, index=False)
        qc_wide_template.to_excel(writer, sheet_name=QC_WIDE_SHEET_NAME, index=False)
        qc_long_template.to_excel(writer, sheet_name=QC_LONG_SHEET_NAME, index=False)
        spec_template.to_excel(writer, sheet_name=SPEC_SHEET_NAME, index=False)
    return buffer.getvalue()


def build_template_bundle() -> dict[str, object]:
    """Build every download template plus its serialized form."""
    process_template = build_process_template_dataframe()
    qc_wide_template = build_qc_wide_template_dataframe()
    qc_long_template = build_qc_long_template_dataframe()

    return {
        "process": process_template,
        "qc_wide": qc_wide_template,
        "qc_long": qc_long_template,
        "process_csv": process_template.to_csv(index=False),
        "qc_wide_csv": qc_wide_template.to_csv(index=False),
        "qc_long_csv": qc_long_template.to_csv(index=False),
        "workbook_bytes": build_combined_workbook_template_bytes(
            process_template,
            qc_wide_template,
            qc_long_template,
        ),
    }
