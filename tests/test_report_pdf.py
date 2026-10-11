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


def test_pdf_honours_the_outcome_direction_override(monkeypatch, synthetic_pipeline):
    """A PDF built without the override contradicts the screen and the LLM pack."""
    import utils.report as report_module

    captured_objectives = {}
    original_builder = report_module.build_operating_window_hints

    def recording_builder(*args, **kwargs):
        captured_objectives["value"] = kwargs.get("outcome_objectives")
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(report_module, "build_operating_window_hints", recording_builder)

    overrides = {"yield_g_L": "minimize"}
    generate_analysis_report_pdf(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=synthetic_pipeline["analysis"],
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
        outcome_objectives=overrides,
    )

    assert captured_objectives["value"] == overrides


def test_a_full_length_narrative_keeps_its_caveats_section():
    """A 45-line cap dropped the last section, which is the one with the caveats."""
    from analysis.report_validator import REQUIRED_HEADINGS
    from utils.report import markdown_to_report_blocks

    narrative = "\n".join(
        f"{heading}\n" + "\n".join(f"- Point {index} under {heading}." for index in range(12))
        for heading in REQUIRED_HEADINGS
    )

    blocks = markdown_to_report_blocks(narrative)

    assert blocks[-1][0] == "- Point 11 under ## Data Quality and Confidence Notes."
    assert not any("truncated" in text for text, _ in blocks)


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
