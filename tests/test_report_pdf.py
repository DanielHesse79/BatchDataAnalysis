import pandas as pd
import pytest

from utils.report import (
    MAX_TABLE_CELL_CHARACTERS,
    ReportGenerationError,
    build_dataframe_table,
    generate_analysis_report_pdf,
)


def test_pdf_report_builds_for_the_reference_pipeline(synthetic_pipeline):
    pdf_bytes = generate_analysis_report_pdf(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=synthetic_pipeline["analysis"],
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
        interpretation_markdown="## Executive Summary\n\nSmoke test narrative.",
    )

    assert pdf_bytes[:5] == b"%PDF-"
    assert len(pdf_bytes) > 10_000


def test_pdf_report_builds_without_an_interpretation(synthetic_pipeline):
    pdf_bytes = generate_analysis_report_pdf(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=synthetic_pipeline["analysis"],
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
    )

    assert pdf_bytes[:5] == b"%PDF-"


def test_very_long_cell_text_is_truncated_instead_of_breaking_the_layout():
    """A cell taller than one page aborts the whole ReportLab build."""
    dataframe = pd.DataFrame({"variable": ["x" * 5_000], "note": ["y" * 12_000]})

    table = build_dataframe_table(dataframe, max_rows=5)

    rendered_cells = [
        cell.text for row in table._cellvalues[1:] for cell in row
    ]
    assert rendered_cells
    assert all(len(text) <= MAX_TABLE_CELL_CHARACTERS + 32 for text in rendered_cells)


def test_report_generation_errors_are_wrapped_for_the_ui(monkeypatch, synthetic_pipeline):
    """ReportLab layout errors must surface as ReportGenerationError, not a traceback."""
    import utils.report as report_module

    class ExplodingDocument:
        def __init__(self, *args, **kwargs):
            pass

        def build(self, *args, **kwargs):
            raise RuntimeError("cell is too tall for the frame")

    monkeypatch.setattr(report_module, "SimpleDocTemplate", ExplodingDocument)

    with pytest.raises(ReportGenerationError, match="layout engine"):
        generate_analysis_report_pdf(
            profile_result=synthetic_pipeline["profile"],
            audit_result=synthetic_pipeline["audit"],
            analysis_results=synthetic_pipeline["analysis"],
            merged_dataframe=synthetic_pipeline["dataframe"],
            outcomes=synthetic_pipeline["outcomes"],
        )
