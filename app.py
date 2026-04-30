"""Streamlit entry point for Batch Insight Analyzer."""

from __future__ import annotations

from datetime import datetime
from html import escape
from io import BytesIO
from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.advanced_methods import (
    get_catboost_shap_dependency_status,
    run_catboost_shap,
)
from analysis.audit import run_preflight_audit
from analysis.confidence import build_driver_confidence_breakdown
from analysis.data_prep import (
    DataPrepError,
    get_batch_column_default_index,
    get_default_outcome_columns,
    load_tabular_file,
    merge_process_and_qc_data,
)
from analysis.aggregation import (
    DUPLICATE_STRATEGIES,
    LONG_FORMAT_AGGREGATIONS,
    aggregate_duplicate_batch_rows,
    detect_long_format_candidates,
    pivot_long_to_wide,
)
from analysis.evidence import build_operating_window_hints, build_report_pack
from analysis.intake import IntakeLoadOptions, inspect_tabular_file, load_intake_dataframe
from analysis.interpreter import (
    OllamaInterpreterError,
    build_model_options,
    choose_default_model,
    get_available_ollama_models,
    get_ollama_base_url,
    is_cloud_model,
    sanitize_interpretation_text,
    stream_interpretation,
)
from analysis.key_findings import build_deterministic_key_findings
from analysis.mapping import build_mapping_profile, mapping_profile_to_json
from analysis.method_registry import list_method_explainers
from analysis.methods import run_all_analyses
from analysis.normalization import normalize_dataframe_values
from analysis.profiling import profile_merged_data
from analysis.readiness import calculate_data_readiness
from analysis.report_validator import validate_interpretation_text
from analysis.specs import (
    SpecError,
    assess_specs,
    build_default_spec_template,
)
from utils.plots import (
    loading_plot,
    outcome_distribution,
    outcome_vs_spec_variable_plot,
    pca_scatter,
    qc_control_chart,
    qc_control_summary,
    qc_trend_plot,
    scree_plot,
    spec_distribution_plot,
    spec_margin_bar,
    variable_importance_bar,
)
from utils.report import ReportGenerationError, generate_analysis_report_pdf


APP_ROOT = Path(__file__).resolve().parent
DOCUMENTATION_FILES = {
    "User guide": APP_ROOT / "docs" / "USER_GUIDE.md",
    "Purpose and scope": APP_ROOT / "docs" / "PURPOSE_AND_SCOPE.md",
    "Theory": APP_ROOT / "docs" / "THEORY.md",
    "Field data intake": APP_ROOT / "docs" / "FIELD_DATA_INTAKE.md",
    "Specs and operating windows": APP_ROOT / "docs" / "SPECS_AND_WINDOWS.md",
    "Validation notes": APP_ROOT / "docs" / "VALIDATION.md",
    "Development notes": APP_ROOT / "docs" / "DEVELOPMENT.md",
    "README / install": APP_ROOT / "README.md",
}


st.set_page_config(
    page_title="Batch Insight Analyzer",
    page_icon="BIA",
    layout="wide",
)


def apply_custom_theme() -> None:
    """Apply app-level styling for a quieter analytics-workbench feel."""
    st.markdown(
        """
        <style>
        :root {
            --bia-ink: #18201b;
            --bia-muted: #66736d;
            --bia-border: #d9ded8;
            --bia-surface: #ffffff;
            --bia-surface-soft: #f9f7f2;
            --bia-accent: #0f766e;
            --bia-accent-strong: #0b5e58;
            --bia-warn: #b45309;
            --bia-danger: #be3a2f;
        }

        .stApp {
            background: #f6f4ef;
            color: var(--bia-ink);
        }

        [data-testid="stAppViewContainer"] > .main {
            background: #f6f4ef;
        }

        [data-testid="stHeader"] {
            background: rgba(246, 244, 239, 0.92);
            border-bottom: 1px solid rgba(24, 32, 27, 0.08);
            backdrop-filter: blur(10px);
        }

        [data-testid="stSidebar"] {
            background: #eff4f1;
            border-right: 1px solid var(--bia-border);
        }

        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
        [data-testid="stSidebar"] [data-testid="stMarkdownContainer"] li {
            color: #405149;
            font-size: 0.92rem;
            line-height: 1.45;
        }

        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3 {
            color: var(--bia-ink);
            letter-spacing: 0;
        }

        .block-container {
            padding-top: 2rem;
            padding-bottom: 4rem;
            max-width: 1320px;
        }

        .bia-hero {
            background: var(--bia-surface);
            border: 1px solid var(--bia-border);
            border-left: 5px solid var(--bia-accent);
            border-radius: 8px;
            padding: 1.35rem 1.5rem 1.2rem;
            margin: 0 0 1.3rem;
            box-shadow: 0 12px 32px rgba(35, 46, 40, 0.08);
        }

        .bia-eyebrow {
            color: var(--bia-accent-strong);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin-bottom: 0.25rem;
        }

        .bia-hero h1 {
            color: var(--bia-ink);
            font-size: 2rem;
            line-height: 1.18;
            letter-spacing: 0;
            margin: 0;
            padding: 0;
        }

        .bia-hero p {
            color: var(--bia-muted);
            max-width: 880px;
            margin: 0.55rem 0 0;
            font-size: 1rem;
            line-height: 1.55;
        }

        .bia-status-row {
            display: flex;
            flex-wrap: wrap;
            gap: 0.55rem;
            margin-top: 1rem;
        }

        .bia-status-pill {
            border: 1px solid var(--bia-border);
            border-radius: 999px;
            background: var(--bia-surface-soft);
            color: #3e4d46;
            display: inline-flex;
            align-items: center;
            min-height: 2rem;
            padding: 0.32rem 0.72rem;
            font-size: 0.86rem;
            font-weight: 650;
            white-space: nowrap;
        }

        .bia-status-pill.active {
            border-color: rgba(15, 118, 110, 0.4);
            background: #e8f5f2;
            color: var(--bia-accent-strong);
        }

        .bia-help-grid {
            display: grid;
            grid-template-columns: repeat(3, minmax(0, 1fr));
            gap: 0.85rem;
            margin: 0.6rem 0 1.1rem;
        }

        .bia-help-card,
        .bia-issue-card {
            background: var(--bia-surface);
            border: 1px solid var(--bia-border);
            border-radius: 8px;
            padding: 0.95rem 1rem;
            box-shadow: 0 8px 22px rgba(35, 46, 40, 0.05);
        }

        .bia-help-card strong,
        .bia-issue-card strong {
            color: var(--bia-ink);
            display: block;
            font-size: 0.95rem;
            margin-bottom: 0.25rem;
        }

        .bia-help-card p,
        .bia-issue-card p {
            color: var(--bia-muted);
            font-size: 0.88rem;
            line-height: 1.45;
            margin: 0;
        }

        .bia-method-status {
            border-radius: 999px;
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 750;
            margin-bottom: 0.45rem;
            padding: 0.18rem 0.52rem;
            text-transform: uppercase;
        }

        .bia-method-status.available {
            background: #e8f5f2;
            color: var(--bia-accent-strong);
        }

        .bia-method-status.planned {
            background: #f4eee2;
            color: var(--bia-warn);
        }

        .bia-method-status.optional {
            background: #edf2f7;
            color: #475569;
        }

        .bia-issue-card.blocker {
            border-left: 5px solid var(--bia-danger);
        }

        .bia-issue-card.warning {
            border-left: 5px solid var(--bia-warn);
        }

        .bia-issue-card.info {
            border-left: 5px solid var(--bia-accent);
        }

        .bia-template-note {
            color: var(--bia-muted);
            font-size: 0.9rem;
            line-height: 1.45;
            margin: 0.1rem 0 0.65rem;
        }

        .bia-section-kicker {
            color: var(--bia-accent-strong);
            font-size: 0.78rem;
            font-weight: 750;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            margin: 1.2rem 0 -0.2rem;
        }

        .bia-sidebar-brand {
            border-bottom: 1px solid rgba(24, 32, 27, 0.12);
            margin-bottom: 1rem;
            padding-bottom: 0.95rem;
        }

        .bia-sidebar-brand h2 {
            font-size: 1.15rem;
            line-height: 1.25;
            margin: 0;
            padding: 0;
        }

        .bia-sidebar-brand p {
            color: var(--bia-muted);
            margin: 0.35rem 0 0;
        }

        .bia-sidebar-step {
            border-left: 3px solid #b7c8c2;
            margin: 0.35rem 0;
            padding: 0.12rem 0 0.12rem 0.65rem;
        }

        .bia-sidebar-step strong {
            color: var(--bia-ink);
            display: block;
            font-size: 0.92rem;
        }

        .bia-sidebar-step span {
            color: var(--bia-muted);
            font-size: 0.82rem;
        }

        h2, h3 {
            color: var(--bia-ink);
            letter-spacing: 0;
        }

        h2 {
            margin-top: 1.8rem;
            padding-bottom: 0.25rem;
        }

        h3 {
            margin-top: 1.25rem;
        }

        [data-testid="stMetric"] {
            background: var(--bia-surface);
            border: 1px solid var(--bia-border);
            border-radius: 8px;
            padding: 0.85rem 0.95rem;
            box-shadow: 0 8px 22px rgba(35, 46, 40, 0.05);
        }

        [data-testid="stMetricLabel"] {
            color: var(--bia-muted);
            font-weight: 650;
        }

        [data-testid="stMetricValue"] {
            color: var(--bia-ink);
            font-weight: 760;
        }

        [data-testid="stFileUploader"] {
            background: var(--bia-surface);
            border: 1px solid var(--bia-border);
            border-radius: 8px;
            padding: 0.9rem 1rem;
            box-shadow: 0 8px 22px rgba(35, 46, 40, 0.04);
        }

        [data-testid="stFileUploaderDropzone"] {
            border-color: #a9bab4;
            background: #f8fbfa;
            border-radius: 8px;
        }

        [data-testid="stTabs"] [role="tablist"] {
            border-bottom: 1px solid var(--bia-border);
            gap: 0.15rem;
        }

        [data-testid="stTabs"] [role="tab"] {
            color: #53645c;
            border-radius: 6px 6px 0 0;
            padding: 0.65rem 0.95rem;
            font-weight: 650;
        }

        [data-testid="stTabs"] [aria-selected="true"] {
            color: var(--bia-accent-strong);
            background: #e8f5f2;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius: 7px;
            border: 1px solid var(--bia-accent-strong);
            background: var(--bia-accent);
            color: #ffffff;
            font-weight: 700;
            min-height: 2.6rem;
            box-shadow: 0 8px 18px rgba(15, 118, 110, 0.18);
        }

        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color: var(--bia-accent-strong);
            background: var(--bia-accent-strong);
            color: #ffffff;
        }

        .stButton > button:disabled {
            border-color: #c9d1cc;
            background: #d9ded8;
            color: #68766f;
            box-shadow: none;
        }

        [data-testid="stAlert"] {
            border-radius: 8px;
            border: 1px solid rgba(24, 32, 27, 0.09);
        }

        [data-testid="stExpander"] {
            background: var(--bia-surface);
            border: 1px solid var(--bia-border);
            border-radius: 8px;
        }

        [data-testid="stDataFrame"] {
            border-radius: 8px;
            overflow: hidden;
            border: 1px solid var(--bia-border);
        }

        .stSelectbox,
        .stMultiSelect,
        .stTextInput {
            background: transparent;
        }

        div[data-baseweb="select"] > div,
        div[data-baseweb="input"] > div {
            border-radius: 7px;
            border-color: #bcc8c2;
            background: #ffffff;
        }

        hr {
            border-color: var(--bia-border);
        }

        @media (max-width: 760px) {
            .block-container {
                padding-left: 1rem;
                padding-right: 1rem;
            }

            .bia-hero {
                padding: 1.05rem 1rem;
            }

            .bia-hero h1 {
                font-size: 1.55rem;
            }

            .bia-help-grid {
                grid-template-columns: 1fr;
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_header() -> None:
    """Show a clear location marker at the top of the app."""
    st.markdown(
        """
        <div class="bia-hero">
            <div class="bia-eyebrow">Process analytics workbench</div>
            <h1>Batch Insight Analyzer</h1>
            <p>
                Merge batch records with QC outcomes, audit the data, then compare PCA,
                PLS, Random Forest, and local or cloud Ollama interpretation in one guided flow.
            </p>
            <div class="bia-status-row">
                <span class="bia-status-pill active">Local-first analysis</span>
                <span class="bia-status-pill">Synthetic validation data included</span>
                <span class="bia-status-pill">Decision support, not batch release</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar() -> None:
    """Render compact workflow guidance in the sidebar."""
    with st.sidebar:
        st.markdown(
            """
            <div class="bia-sidebar-brand">
                <h2>Batch Insight Analyzer</h2>
                <p>Guided process-to-QC driver analysis for small batch datasets.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.markdown("### Workflow")
        workflow_steps = [
            ("1. Ingest", "Upload process and QC files."),
            ("2. Match", "Select batch ID columns and outcomes."),
            ("3. Specs", "Optionally add process windows and QC limits."),
            ("4. Audit", "Check missingness, leakage, drift, and outliers."),
            ("5. Analyze", "Run PCA, PLS, and Random Forest."),
            ("6. Explain", "Generate a plain-language interpretation."),
        ]
        for step_title, step_description in workflow_steps:
            st.markdown(
                f"""
                <div class="bia-sidebar-step">
                    <strong>{step_title}</strong>
                    <span>{step_description}</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("### Guardrails")
        st.markdown(
            """
            - Treat driver rankings as associations.
            - Review audit warnings before trusting models.
            - Use DoE or confirmatory runs before changing setpoints.
            """
        )


def render_documentation_section() -> None:
    """Render local Markdown documentation inside the Streamlit UI."""
    available_documents = {
        label: path for label, path in DOCUMENTATION_FILES.items() if path.exists()
    }
    if not available_documents:
        return

    with st.expander("Documentation and theory", expanded=False):
        st.caption(
            "Read the local project documentation without leaving the app. "
            "These files live in the repository and can be edited later as the product evolves."
        )
        selected_document_label = st.selectbox(
            "Document",
            options=list(available_documents.keys()),
            key="documentation_document_select",
        )
        selected_document_path = available_documents[selected_document_label]

        try:
            document_text = selected_document_path.read_text(encoding="utf-8")
        except OSError as error:
            st.error(f"Could not read {selected_document_path.name}: {error}")
            return

        st.download_button(
            "Download selected document",
            data=document_text,
            file_name=selected_document_path.name,
            mime="text/markdown",
            key="download_selected_documentation",
        )
        st.markdown("---")
        st.markdown(document_text)


def render_intake_workflow_explainer() -> None:
    """Explain the expected data shape and the messy-data intake flow."""
    st.markdown(
        """
        <div class="bia-help-grid">
            <div class="bia-help-card">
                <strong>Recommended shape</strong>
                <p>One row per batch, one batch ID column, process variables in the process table, and QC outcomes in the QC table.</p>
            </div>
            <div class="bia-help-card">
                <strong>Excel workbooks are fine</strong>
                <p>Process and QC data may be two files or two sheets in the same workbook. Select the correct sheet and header row before matching.</p>
            </div>
            <div class="bia-help-card">
                <strong>Messy values are flagged</strong>
                <p>The app parses units, decimal commas, long-format QC results, and duplicate batch IDs, then shows blockers and warnings before analysis.</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_data_template_downloads() -> None:
    """Offer simple import templates to guide users toward stable data shapes."""
    with st.expander("Download data templates", expanded=False):
        st.markdown(
            """
            <p class="bia-template-note">
            Use these templates when asking a production or QC team for exports.
            The app can handle messier files, but these shapes make analysis faster and safer.
            </p>
            """,
            unsafe_allow_html=True,
        )

        process_template = build_process_template_dataframe()
        qc_wide_template = build_qc_wide_template_dataframe()
        qc_long_template = build_qc_long_template_dataframe()
        combined_workbook_bytes = build_combined_workbook_template_bytes(
            process_template,
            qc_wide_template,
            qc_long_template,
            build_default_spec_template(),
        )

        template_columns = st.columns(4)
        template_columns[0].download_button(
            "Process CSV template",
            data=process_template.to_csv(index=False),
            file_name="batch_insight_process_template.csv",
            mime="text/csv",
            key="download_process_template",
        )
        template_columns[1].download_button(
            "QC wide CSV template",
            data=qc_wide_template.to_csv(index=False),
            file_name="batch_insight_qc_wide_template.csv",
            mime="text/csv",
            key="download_qc_wide_template",
        )
        template_columns[2].download_button(
            "QC long CSV template",
            data=qc_long_template.to_csv(index=False),
            file_name="batch_insight_qc_long_template.csv",
            mime="text/csv",
            key="download_qc_long_template",
        )
        template_columns[3].download_button(
            "Combined Excel template",
            data=combined_workbook_bytes,
            file_name="batch_insight_combined_workbook_template.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            key="download_combined_workbook_template",
        )

        preview_tab, qc_tab, long_tab = st.tabs(["Process", "QC wide", "QC long"])
        with preview_tab:
            st.dataframe(process_template, width="stretch", hide_index=True)
        with qc_tab:
            st.dataframe(qc_wide_template, width="stretch", hide_index=True)
        with long_tab:
            st.dataframe(qc_long_template, width="stretch", hide_index=True)


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
    spec_template: pd.DataFrame,
) -> bytes:
    """Build an Excel workbook with separate process/QC/spec sheets."""
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        process_template.to_excel(writer, sheet_name="process_data", index=False)
        qc_wide_template.to_excel(writer, sheet_name="qc_results_wide", index=False)
        qc_long_template.to_excel(writer, sheet_name="qc_results_long", index=False)
        spec_template.to_excel(writer, sheet_name="specs_windows", index=False)
    return buffer.getvalue()


def render_file_intake(uploaded_file, label: str, key_prefix: str):
    """Inspect, load, normalize, and optionally pivot one uploaded file."""
    if uploaded_file is None:
        return None, {}

    try:
        inspection = inspect_tabular_file(uploaded_file)
    except DataPrepError as error:
        st.error(f"{label}: {error}")
        return None, {}

    metadata = {
        "file_name": inspection.file_name,
        "file_extension": inspection.file_extension,
    }

    with st.expander(f"{label} intake settings", expanded=True):
        st.caption(
            "Confirm the sheet/header row and let the app normalize obvious field-data issues before matching batches."
        )

        sheet_name = inspection.suggested_sheet
        if inspection.sheets:
            sheet_name = st.selectbox(
                "Sheet",
                options=inspection.sheets,
                index=option_default_index(inspection.sheets, inspection.suggested_sheet),
                key=f"{key_prefix}_sheet",
            )

        header_row = st.number_input(
            "Header row number",
            min_value=0,
            max_value=50,
            value=int(inspection.suggested_header_row),
            step=1,
            key=f"{key_prefix}_header_row",
            help="0 means the first row. If the file has title rows above the headers, increase this.",
        )
        parse_numeric_like_columns = st.checkbox(
            "Parse numeric-looking text values",
            value=True,
            key=f"{key_prefix}_parse_numeric",
            help="Converts values such as '12,5 %', '<20 ppm', or '100 g' into numeric values where safe.",
        )

        if inspection.warnings:
            for warning in inspection.warnings:
                st.warning(warning)

        with st.expander("Raw preview before parsing", expanded=False):
            st.dataframe(inspection.raw_preview, width="stretch")

        try:
            dataframe = load_intake_dataframe(
                uploaded_file,
                IntakeLoadOptions(sheet_name=sheet_name, header_row=int(header_row)),
            )
        except DataPrepError as error:
            st.error(f"{label}: {error}")
            return None, metadata

        normalization_result = normalize_dataframe_values(
            dataframe,
            parse_numeric_like_columns=parse_numeric_like_columns,
        )
        dataframe = normalization_result.dataframe
        metadata.update(
            {
                "sheet_name": sheet_name,
                "header_row": int(header_row),
                "parse_numeric_like_columns": parse_numeric_like_columns,
                "numeric_parse_report": normalization_result.numeric_parse_report,
            }
        )

        for warning in normalization_result.warnings:
            st.warning(warning)

        if not normalization_result.numeric_parse_report.empty:
            with st.expander("Numeric parsing log", expanded=False):
                st.dataframe(
                    normalization_result.numeric_parse_report,
                    width="stretch",
                    hide_index=True,
                )

        dataframe, pivot_metadata = render_long_format_controls(
            dataframe=dataframe,
            label=label,
            key_prefix=key_prefix,
        )
        metadata["long_format_pivot"] = pivot_metadata

        st.success(f"{label}: prepared {len(dataframe):,} rows and {len(dataframe.columns):,} columns.")
        st.dataframe(dataframe.head(10), width="stretch")

    return dataframe, metadata


def render_long_format_controls(dataframe, label: str, key_prefix: str):
    """Offer a long-to-wide pivot when the file looks like batch/test/value rows."""
    suggestion = detect_long_format_candidates(dataframe)
    metadata = {"enabled": False}
    if not suggestion.likely_long_format:
        with st.expander("Long-format pivot", expanded=False):
            st.caption("Use this only if the file has rows like batch_id | test_name | result.")
            enable_pivot = st.checkbox(
                "Pivot this long table to wide format",
                value=False,
                key=f"{key_prefix}_enable_pivot",
            )
            if not enable_pivot:
                return dataframe, metadata
    else:
        st.warning(
            f"{label} looks like long format. Pivot it to one row per batch before analysis."
        )
        enable_pivot = st.checkbox(
            "Pivot this long table to wide format",
            value=True,
            key=f"{key_prefix}_enable_pivot",
        )
        if not enable_pivot:
            return dataframe, metadata

    columns = list(dataframe.columns)
    batch_id_column = st.selectbox(
        "Long-format batch ID column",
        options=columns,
        index=option_default_index(columns, first_or_none(suggestion.batch_id_candidates)),
        key=f"{key_prefix}_pivot_batch",
    )
    name_column = st.selectbox(
        "Long-format variable/test name column",
        options=columns,
        index=option_default_index(columns, first_or_none(suggestion.name_candidates)),
        key=f"{key_prefix}_pivot_name",
    )
    value_column = st.selectbox(
        "Long-format result/value column",
        options=columns,
        index=option_default_index(columns, first_or_none(suggestion.value_candidates)),
        key=f"{key_prefix}_pivot_value",
    )
    aggregation = st.selectbox(
        "Duplicate batch/test aggregation",
        options=list(LONG_FORMAT_AGGREGATIONS.keys()),
        format_func=lambda key: LONG_FORMAT_AGGREGATIONS[key],
        index=0,
        key=f"{key_prefix}_pivot_aggregation",
    )

    try:
        wide_dataframe = pivot_long_to_wide(
            dataframe=dataframe,
            batch_id_column=batch_id_column,
            name_column=name_column,
            value_column=value_column,
            aggregation=aggregation,
        )
    except DataPrepError as error:
        st.error(str(error))
        return dataframe, metadata

    st.success(
        f"Pivoted to {len(wide_dataframe):,} batch rows and {len(wide_dataframe.columns):,} columns."
    )
    metadata = {
        "enabled": True,
        "batch_id_column": batch_id_column,
        "name_column": name_column,
        "value_column": value_column,
        "aggregation": aggregation,
    }
    return wide_dataframe, metadata


def option_default_index(options, preferred_value) -> int:
    """Return a safe selectbox default index."""
    if preferred_value in options:
        return list(options).index(preferred_value)
    return 0


def first_or_none(values):
    """Return the first item or None."""
    return values[0] if values else None


def show_dataframe_preview(label: str, dataframe) -> None:
    """Display a compact preview without overwhelming the user."""
    with st.expander(f"Preview {label}", expanded=False):
        st.dataframe(dataframe.head(10), width="stretch")


def render_duplicate_strategy(label: str, dataframe, batch_id_column: str, key: str) -> str:
    """Let the user choose how to handle duplicate batch IDs."""
    duplicate_count = count_duplicate_batch_ids_for_ui(dataframe, batch_id_column)
    if duplicate_count == 0:
        return "error"

    st.warning(
        f"{label} has {duplicate_count:,} duplicate normalized batch ID(s). "
        "Choose how to resolve them before analysis."
    )
    return st.selectbox(
        f"{label} duplicate handling",
        options=list(DUPLICATE_STRATEGIES.keys()),
        format_func=lambda strategy: DUPLICATE_STRATEGIES[strategy],
        index=0,
        key=key,
    )


def count_duplicate_batch_ids_for_ui(dataframe, batch_id_column: str) -> int:
    """Count duplicate batch IDs without breaking the UI if selection is invalid."""
    try:
        from analysis.aggregation import count_duplicate_batch_ids

        return count_duplicate_batch_ids(dataframe, batch_id_column)
    except (DataPrepError, KeyError, ValueError):
        return 0


def render_readiness_panel(readiness_result) -> None:
    """Render a field-data readiness score before merge/analyze."""
    st.markdown(
        '<div class="bia-section-kicker">Step 2b - Data readiness</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Data readiness")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Readiness score", f"{readiness_result.score}/100")
    metric_columns[1].metric("Blockers", f"{len(readiness_result.blockers):,}")
    metric_columns[2].metric("Warnings", f"{len(readiness_result.warnings):,}")
    metric_columns[3].metric(
        "Candidate matches",
        f"{readiness_result.details.get('candidate_matched_batches', 0):,}",
    )

    render_readiness_issue_cards(readiness_result)

    if readiness_result.blockers:
        st.error("Resolve blockers before merging and analyzing.")
        for blocker in readiness_result.blockers:
            st.write(f"- {blocker}")
    elif readiness_result.warnings:
        st.warning("The data can be analyzed, but review these warnings first.")
        for warning in readiness_result.warnings:
            st.write(f"- {warning}")
    else:
        st.success("No readiness blockers were found.")

    if readiness_result.info:
        with st.expander("Readiness notes", expanded=False):
            for info_message in readiness_result.info:
                st.write(f"- {info_message}")

    issue_table = build_readiness_issue_table(readiness_result)
    if not issue_table.empty:
        with st.expander("Issue details", expanded=bool(readiness_result.blockers)):
            st.dataframe(issue_table, width="stretch", hide_index=True)


def render_readiness_issue_cards(readiness_result) -> None:
    """Show readiness blockers/warnings/info as visual cards."""
    cards = []
    for severity, messages in [
        ("blocker", readiness_result.blockers[:3]),
        ("warning", readiness_result.warnings[:3]),
        ("info", readiness_result.info[:2]),
    ]:
        for message in messages:
            cards.append(
                f"""
                <div class="bia-issue-card {severity}">
                    <strong>{severity.title()}</strong>
                    <p>{escape(str(message))}</p>
                </div>
                """
            )

    if not cards:
        return

    st.markdown(
        '<div class="bia-help-grid">' + "\n".join(cards[:6]) + "</div>",
        unsafe_allow_html=True,
    )


def build_readiness_issue_table(readiness_result) -> pd.DataFrame:
    """Build a tabular issue list for users who want details."""
    rows = []
    for severity, messages in [
        ("Blocker", readiness_result.blockers),
        ("Warning", readiness_result.warnings),
        ("Info", readiness_result.info),
    ]:
        for message in messages:
            rows.append({"severity": severity, "message": message})
    return pd.DataFrame(rows)


def apply_duplicate_handling_for_merge(
    process_dataframe,
    qc_dataframe,
    process_batch_id_column: str,
    qc_batch_id_column: str,
    process_duplicate_strategy: str,
    qc_duplicate_strategy: str,
):
    """Resolve duplicates immediately before merge."""
    process_result = aggregate_duplicate_batch_rows(
        dataframe=process_dataframe,
        batch_id_column=process_batch_id_column,
        strategy=process_duplicate_strategy,
        label="Process data",
    )
    qc_result = aggregate_duplicate_batch_rows(
        dataframe=qc_dataframe,
        batch_id_column=qc_batch_id_column,
        strategy=qc_duplicate_strategy,
        label="QC data",
    )
    return process_result, qc_result


def render_mapping_profile_download(
    process_file,
    qc_file,
    process_metadata: dict,
    qc_metadata: dict,
    process_batch_id_column: str,
    qc_batch_id_column: str,
    selected_outcome_columns: list[str],
    process_duplicate_strategy: str,
    qc_duplicate_strategy: str,
) -> None:
    """Offer a downloadable mapping profile for repeat imports."""
    profile = build_mapping_profile(
        process_file_name=getattr(process_file, "name", "process_file"),
        qc_file_name=getattr(qc_file, "name", "qc_file"),
        process_intake_options=strip_non_json_metadata(process_metadata),
        qc_intake_options=strip_non_json_metadata(qc_metadata),
        process_batch_id_column=process_batch_id_column,
        qc_batch_id_column=qc_batch_id_column,
        outcome_columns=selected_outcome_columns,
        process_duplicate_strategy=process_duplicate_strategy,
        qc_duplicate_strategy=qc_duplicate_strategy,
    )
    st.download_button(
        "Download mapping profile",
        data=mapping_profile_to_json(profile),
        file_name="batch_insight_mapping_profile.json",
        mime="application/json",
        key="download_mapping_profile",
        help="Useful when a customer sends the same export format repeatedly.",
    )


def strip_non_json_metadata(metadata: dict) -> dict:
    """Remove DataFrames from intake metadata before writing JSON."""
    output = {}
    for key, value in metadata.items():
        if key == "numeric_parse_report":
            if value is not None and not value.empty:
                output[key] = value.to_dict(orient="records")
            else:
                output[key] = []
        else:
            output[key] = value
    return output


def render_spec_input():
    """Render optional spec/window file controls."""
    st.markdown(
        '<div class="bia-section-kicker">Optional - Specs & operating windows</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Specs & operating windows")
    st.caption(
        "Upload process windows and QC specs to compare official limits against historical behavior. "
        "You can also continue without specs."
    )

    template_dataframe = build_default_spec_template()
    template_csv = template_dataframe.to_csv(index=False)
    control_columns = st.columns([1, 1])
    with control_columns[0]:
        spec_file = st.file_uploader(
            "Upload spec/window file",
            type=["csv", "xlsx", "xls"],
            key="spec_file",
            help="Expected columns: variable, role, target, lower_limit, upper_limit, unit, criticality, notes.",
        )
    with control_columns[1]:
        st.download_button(
            "Download spec template",
            data=template_csv,
            file_name="batch_insight_spec_template.csv",
            mime="text/csv",
            key="download_spec_template",
        )
        st.caption("Tip: the repository also includes `data/synthetic_specs.csv`.")

    return spec_file


def render_merge_summary(merge_result) -> None:
    """Show the merge result in plain language."""
    st.success(
        f"Matched {merge_result.matched_batch_count:,} batches "
        f"from {merge_result.process_batch_count:,} process IDs "
        f"and {merge_result.qc_batch_count:,} QC IDs."
    )

    warning_messages = []
    if merge_result.unmatched_process_batch_ids:
        examples = ", ".join(merge_result.unmatched_process_batch_ids[:5])
        warning_messages.append(
            f"{len(merge_result.unmatched_process_batch_ids):,} process batch ID(s) had no QC match "
            f"(examples: {examples})."
        )

    if merge_result.unmatched_qc_batch_ids:
        examples = ", ".join(merge_result.unmatched_qc_batch_ids[:5])
        warning_messages.append(
            f"{len(merge_result.unmatched_qc_batch_ids):,} QC batch ID(s) had no process match "
            f"(examples: {examples})."
        )

    for warning_message in warning_messages:
        st.warning(warning_message)


def render_profile(profile_result, merged_dataframe) -> None:
    """Render Phase 3 profiling results."""
    st.markdown(
        '<div class="bia-section-kicker">Step 3 - Data audit</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Data profile")

    first_row = st.columns(4)
    first_row[0].metric("Matched batches", f"{profile_result.matched_batch_count:,}")
    first_row[1].metric("Process variables", f"{len(profile_result.process_columns):,}")
    first_row[2].metric("Quality outcomes", f"{len(profile_result.outcome_columns):,}")
    first_row[3].metric(
        "High-missing columns",
        f"{int(profile_result.missingness['flag'].sum()):,}",
    )

    second_row = st.columns(3)
    second_row[0].metric(
        "Continuous",
        f"{profile_result.variable_type_counts.get('continuous', 0):,}",
    )
    second_row[1].metric(
        "Categorical",
        f"{profile_result.variable_type_counts.get('categorical', 0):,}",
    )
    second_row[2].metric(
        "Binary",
        f"{profile_result.variable_type_counts.get('binary', 0):,}",
    )

    if profile_result.warnings:
        for warning in profile_result.warnings:
            st.warning(warning)
    else:
        st.success("No major data quality warnings were found in this profile.")

    with st.expander("Process variable type detection", expanded=False):
        st.dataframe(profile_result.variable_types, width="stretch")

    st.markdown("#### Missing data")
    high_missing_columns = profile_result.missingness[profile_result.missingness["flag"]]
    if high_missing_columns.empty:
        st.success("No columns have more than 20% missing data.")
    else:
        st.warning("Columns above 20% missing data should be reviewed before modeling.")
        st.dataframe(high_missing_columns, width="stretch", hide_index=True)

    with st.expander("Show missing data for all columns", expanded=False):
        st.dataframe(profile_result.missingness, width="stretch", hide_index=True)

    st.markdown("#### Constant or near-constant process variables")
    if profile_result.near_constant_columns.empty:
        st.success("No constant or near-constant process variables were detected.")
    else:
        st.warning("These variables carry little information and may be removed during modeling.")
        st.dataframe(
            profile_result.near_constant_columns,
            width="stretch",
            hide_index=True,
        )

    st.markdown("#### Outcome statistics")
    st.dataframe(profile_result.outcome_statistics, width="stretch", hide_index=True)

    numeric_outcomes = profile_result.outcome_statistics[
        profile_result.outcome_statistics["count"] > 0
    ]["outcome"].tolist()
    if numeric_outcomes:
        histogram_tabs = st.tabs(numeric_outcomes)
        for histogram_tab, outcome_column in zip(histogram_tabs, numeric_outcomes):
            with histogram_tab:
                st.plotly_chart(
                    outcome_distribution(merged_dataframe, outcome_column),
                    width="stretch",
                    key=f"profile_outcome_distribution_{outcome_column}",
                )


def render_preflight_audit(audit_result) -> None:
    """Render conservative audit checks before modeling."""
    st.markdown(
        '<div class="bia-section-kicker">Guardrail checks</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Pre-flight audit")
    st.caption("These checks flag ways historical batch data can mislead an analysis.")

    metric_columns = st.columns(4)
    metric_columns[0].metric("Audit warnings", f"{audit_result.warning_count:,}")
    metric_columns[1].metric(
        "Leakage suspects",
        f"{audit_result.summary['leakage_name_warning_count'] + audit_result.summary['high_feature_target_correlation_count']:,}",
    )
    metric_columns[2].metric(
        "Drift/time columns",
        f"{audit_result.summary['date_or_drift_column_count']:,}",
    )
    metric_columns[3].metric(
        "Outlier columns",
        f"{audit_result.summary['columns_with_outliers']:,}",
    )

    st.info(audit_result.sample_size_guidance)
    if audit_result.warnings:
        for warning in audit_result.warnings:
            st.warning(warning)
    else:
        st.success("No major pre-flight audit warnings were found.")

    audit_tabs = st.tabs(
        [
            "Leakage",
            "Drift & confounding",
            "Missingness & outliers",
            "Multicollinearity",
        ]
    )

    with audit_tabs[0]:
        show_audit_table(
            "Name-based leakage suspects",
            audit_result.leakage_name_warnings,
            "No process columns look like obvious post-hoc QC or release fields.",
        )
        show_audit_table(
            "Near-perfect feature/outcome correlations",
            audit_result.feature_target_correlations,
            "No feature/outcome correlations exceeded the leakage threshold.",
        )

    with audit_tabs[1]:
        show_audit_table(
            "Date, sequence, or campaign-like columns",
            audit_result.date_or_drift_columns,
            "No date, sequence, or campaign-like columns were detected.",
        )
        show_audit_table(
            "Potentially confounded categorical variables",
            audit_result.confounding_categoricals,
            "No lot/operator/equipment-style categorical variables were detected.",
        )
        show_audit_table(
            "High-cardinality categorical variables",
            audit_result.high_cardinality_categoricals,
            "No high-cardinality categorical variables were detected.",
        )

    with audit_tabs[2]:
        show_audit_table(
            "Row-level missingness",
            audit_result.row_missingness,
            "No row-level missingness summary is available.",
            success_when_empty=False,
        )
        show_audit_table(
            "IQR outlier flags",
            audit_result.outlier_flags,
            "No numeric IQR outliers were detected.",
        )

    with audit_tabs[3]:
        show_audit_table(
            "Highly correlated numeric process variable pairs",
            audit_result.multicollinearity_pairs,
            "No highly correlated numeric process variable pairs were detected.",
        )


def render_qc_trend_overview(merged_dataframe, profile_result, audit_result) -> None:
    """Render QC trend plots and simple control charts for the audit workflow."""
    st.markdown(
        '<div class="bia-section-kicker">QC behavior over batches</div>',
        unsafe_allow_html=True,
    )
    st.subheader("QC trend overview")

    outcome_options = profile_result.outcome_statistics[
        profile_result.outcome_statistics["count"] > 0
    ]["outcome"].tolist()
    if not outcome_options:
        st.info("No numeric quality outcomes are available for trend charts.")
        return

    order_column = choose_validation_order_column(audit_result, merged_dataframe)
    if order_column:
        st.caption(f"Charts are ordered by `{order_column}`.")
    else:
        st.caption("Charts are ordered by upload row order.")

    color_options = ["None"] + get_qc_trend_color_options(profile_result, merged_dataframe)
    color_column = st.selectbox(
        "Color trend points by",
        options=color_options,
        key="qc_trend_color_column",
    )
    color_column = None if color_column == "None" else color_column

    outcome_tabs = st.tabs(outcome_options)
    for outcome_tab, outcome_column in zip(outcome_tabs, outcome_options):
        with outcome_tab:
            chart_columns = st.columns(2)
            with chart_columns[0]:
                st.markdown("#### Trend")
                st.plotly_chart(
                    qc_trend_plot(
                        dataframe=merged_dataframe,
                        outcome_col=outcome_column,
                        order_column=order_column,
                        color_column=color_column,
                    ),
                    width="stretch",
                    key=f"qc_trend_{outcome_column}_{color_column or 'none'}",
                )

            with chart_columns[1]:
                st.markdown("#### Control chart")
                st.plotly_chart(
                    qc_control_chart(
                        dataframe=merged_dataframe,
                        outcome_col=outcome_column,
                        order_column=order_column,
                    ),
                    width="stretch",
                    key=f"qc_control_{outcome_column}",
                )

            flagged_rows = qc_control_summary(
                dataframe=merged_dataframe,
                outcome_col=outcome_column,
                order_column=order_column,
            )
            if flagged_rows.empty:
                st.success("No points outside mean +/- 3 sigma were detected.")
            else:
                st.warning(
                    f"{len(flagged_rows):,} point(s) are outside mean +/- 3 sigma. Review before modeling."
                )
                st.dataframe(
                    format_numeric_columns(flagged_rows),
                    width="stretch",
                    hide_index=True,
                )


def render_spec_assessment(spec_assessment) -> None:
    """Render spec/window assessment for the current analysis state."""
    st.markdown(
        '<div class="bia-section-kicker">Historical operating window assessment</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Specs & margins")

    if spec_assessment is None or not spec_assessment.has_specs:
        st.info("No spec/window file was supplied. You can still run the analysis without specs.")
        return

    if spec_assessment.warnings:
        for warning in spec_assessment.warnings:
            st.warning(warning)

    variable_summary = spec_assessment.variable_summary
    metric_columns = st.columns(4)
    metric_columns[0].metric("Spec rows", f"{len(spec_assessment.specs):,}")
    metric_columns[1].metric("Matched variables", f"{len(spec_assessment.matched_specs):,}")
    metric_columns[2].metric("Unmatched variables", f"{len(spec_assessment.unmatched_specs):,}")
    metric_columns[3].metric(
        "Out-of-spec batches",
        f"{len(spec_assessment.out_of_spec_batches):,}",
    )

    if variable_summary.empty:
        st.info("No numeric spec-controlled variables were available for assessment.")
        return

    st.caption(
        "These labels are conservative historical checks. They do not prove a design space or justify spec changes by themselves."
    )
    st.dataframe(
        format_numeric_columns(
            variable_summary[
                [
                    "variable",
                    "role",
                    "target",
                    "lower_limit",
                    "upper_limit",
                    "percent_inside",
                    "percent_outside",
                    "percent_close_to_limit",
                    "used_range_ratio",
                    "classification",
                    "reason",
                ]
            ]
        ),
        width="stretch",
        hide_index=True,
    )

    if not spec_assessment.unmatched_specs.empty:
        with st.expander("Unmatched spec rows", expanded=False):
            st.dataframe(spec_assessment.unmatched_specs, width="stretch", hide_index=True)

    if not spec_assessment.out_of_spec_batches.empty:
        with st.expander("Out-of-spec batch details", expanded=False):
            st.dataframe(
                format_numeric_columns(spec_assessment.out_of_spec_batches.head(100)),
                width="stretch",
                hide_index=True,
            )


def render_method_explainers(
    method_keys: list[str],
    title: str = "How to read this section",
    expanded: bool = False,
) -> None:
    """Render concise method limitations and next-step guidance."""
    explainers = list_method_explainers(method_keys)
    if not explainers:
        return

    with st.expander(title, expanded=expanded):
        for explainer in explainers:
            status = explainer.status
            status_label = status if status in {"available", "optional", "planned"} else "available"
            st.markdown(
                f"""
                <div class="bia-help-card">
                    <span class="bia-method-status {escape(status_label)}">{escape(status_label)}</span>
                    <strong>{escape(explainer.name)} - {escape(explainer.short_label)}</strong>
                    <p><b>What it is:</b> {escape(explainer.what_it_is)}</p>
                    <p><b>Good for:</b> {escape(explainer.what_it_can_tell_you)}</p>
                    <p><b>Cannot prove:</b> {escape(explainer.what_it_cannot_prove)}</p>
                    <p><b>Confidence:</b> {escape(explainer.confidence_guidance)}</p>
                    <p><b>Next:</b> {escape(explainer.next_step)}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_deterministic_key_findings(
    profile_result,
    audit_result,
    analysis_results,
    merged_dataframe,
    outcomes: list[str],
    spec_assessment,
    selected_outcome: str | None = None,
) -> None:
    """Render key facts computed by Python before any LLM interpretation."""
    key_findings = build_deterministic_key_findings(
        profile_result=profile_result,
        audit_result=audit_result,
        analysis_results=analysis_results,
        merged_dataframe=merged_dataframe,
        outcomes=outcomes,
        spec_assessment=spec_assessment,
    )
    if key_findings.empty:
        return

    if selected_outcome is not None:
        display_findings = key_findings[
            key_findings["outcome"].isin([selected_outcome, "all outcomes", "process", "qc"])
        ].copy()
    else:
        display_findings = key_findings.copy()

    if display_findings.empty:
        return

    st.markdown("#### Python-verified key findings")
    st.caption(
        "These rows are computed deterministically before the LLM writes a narrative. "
        "They are the safer source of truth when a model summary sounds too confident."
    )
    st.dataframe(
        format_numeric_columns(display_findings.head(14)),
        width="stretch",
        hide_index=True,
    )


def render_driver_confidence_breakdown(
    outcome_drivers: pd.DataFrame,
    analysis_results,
    profile_result,
    audit_result,
    selected_outcome: str,
) -> None:
    """Let the user inspect why a driver received its practical confidence label."""
    if outcome_drivers.empty:
        return

    st.markdown("#### Confidence breakdown")
    selected_driver = st.selectbox(
        "Driver to explain",
        options=outcome_drivers["process_variable"].tolist(),
        key=f"driver_confidence_variable_{selected_outcome}",
    )
    ranked_driver_row = outcome_drivers[
        outcome_drivers["process_variable"] == selected_driver
    ].iloc[0]
    confidence_breakdown = build_driver_confidence_breakdown(
        ranked_driver_row=ranked_driver_row,
        analysis_results=analysis_results,
        profile_result=profile_result,
        audit_result=audit_result,
    )
    st.dataframe(confidence_breakdown, width="stretch", hide_index=True)
    st.caption(
        "This is a practical confidence explanation. It is not a formal validation, "
        "p-value, causal claim, or change-control recommendation."
    )


def render_spec_results(spec_assessment, merged_dataframe, outcome_options: list[str]) -> None:
    """Render detailed spec/window plots and zone summaries."""
    if spec_assessment is None or not spec_assessment.has_specs:
        st.info("Upload a spec/window file before clicking Analyze to enable this tab.")
        return

    variable_summary = spec_assessment.variable_summary
    if variable_summary.empty:
        st.info("No numeric spec-controlled variables were available for plotting.")
        return

    st.markdown("#### Spec challenge overview")
    st.plotly_chart(
        spec_margin_bar(variable_summary),
        width="stretch",
        key="spec_margin_bar",
    )

    st.markdown("#### Assessment table")
    st.dataframe(format_numeric_columns(variable_summary), width="stretch", hide_index=True)

    selectable_variables = variable_summary["variable"].tolist()
    selected_spec_variable = st.selectbox(
        "Spec-controlled variable",
        options=selectable_variables,
        key="spec_results_variable",
    )
    selected_spec_row = variable_summary[
        variable_summary["variable"] == selected_spec_variable
    ].iloc[0]

    chart_columns = st.columns(2)
    with chart_columns[0]:
        st.markdown("#### Distribution with limits")
        st.plotly_chart(
            spec_distribution_plot(merged_dataframe, selected_spec_row),
            width="stretch",
            key=f"spec_distribution_{selected_spec_variable}",
        )

    with chart_columns[1]:
        process_spec_variables = variable_summary[
            variable_summary["role"] == "process"
        ]["variable"].tolist()
        if selected_spec_variable in process_spec_variables and outcome_options:
            selected_spec_outcome = st.selectbox(
                "Outcome for spec-response plot",
                options=outcome_options,
                key="spec_results_outcome",
            )
            st.markdown("#### Outcome response near limits")
            st.plotly_chart(
                outcome_vs_spec_variable_plot(
                    merged_dataframe,
                    selected_spec_row,
                    selected_spec_outcome,
                ),
                width="stretch",
                key=f"spec_outcome_response_{selected_spec_variable}_{selected_spec_outcome}",
            )
        else:
            st.info("Outcome response plots are shown for process variables with selected QC outcomes.")

    if not spec_assessment.outcome_zone_summary.empty:
        st.markdown("#### Outcome means by spec zone")
        st.dataframe(
            format_numeric_columns(spec_assessment.outcome_zone_summary),
            width="stretch",
            hide_index=True,
        )

    if not spec_assessment.out_of_spec_batches.empty:
        st.markdown("#### Out-of-spec batches")
        st.dataframe(
            format_numeric_columns(spec_assessment.out_of_spec_batches),
            width="stretch",
            hide_index=True,
        )


def get_qc_trend_color_options(profile_result, merged_dataframe) -> list[str]:
    """Return categorical columns that are useful for coloring QC trend plots."""
    candidate_columns = profile_result.variable_types[
        profile_result.variable_types["type"].isin(["categorical", "binary"])
    ]["column"].tolist()

    return [
        column_name
        for column_name in candidate_columns
        if column_name in merged_dataframe.columns
        and int(merged_dataframe[column_name].nunique(dropna=True)) <= 20
    ]


def show_audit_table(
    title: str,
    dataframe,
    empty_message: str,
    success_when_empty: bool = True,
) -> None:
    """Render one audit table with a clear empty state."""
    st.markdown(f"#### {title}")
    if dataframe.empty:
        if success_when_empty:
            st.success(empty_message)
        else:
            st.info(empty_message)
        return

    st.dataframe(format_numeric_columns(dataframe), width="stretch", hide_index=True)


def render_time_ordered_validation(
    title: str,
    validation_result: dict,
    score_label: str,
) -> None:
    """Render train-on-old/test-on-recent validation metrics."""
    st.markdown(f"#### {title}")

    if not validation_result.get("available"):
        st.info(validation_result.get("reason", "Time-ordered validation was not available."))
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Train rows", f"{validation_result['train_rows']:,}")
    metric_columns[1].metric("Recent test rows", f"{validation_result['test_rows']:,}")
    metric_columns[2].metric(score_label, format_metric(validation_result["test_r2"]))
    metric_columns[3].metric("Train R2", format_metric(validation_result["train_r2"]))

    st.caption(
        f"Recent test window starts at order value {validation_result['test_start_order_value']} "
        f"and ends at {validation_result['test_end_order_value']}."
    )


def render_catboost_shap_tab(
    merged_dataframe,
    profile_result,
    analysis_results,
    outcome_options: list[str],
) -> None:
    """Render optional CatBoost + native SHAP explanations."""
    render_method_explainers(["catboost_shap"], expanded=False)
    st.caption(
        "This optional model is for explanation and comparison. It does not currently change the main ranked-driver score."
    )

    dependency_status = get_catboost_shap_dependency_status()
    if not dependency_status["available"]:
        st.info(dependency_status["message"])
        st.code(".\\.venv\\Scripts\\python.exe -m pip install catboost", language="powershell")
        return

    selected_outcome = st.selectbox(
        "Outcome",
        options=outcome_options,
        key="catboost_shap_outcome",
    )
    validation_order_column = analysis_results.get("validation_order_column")
    process_columns = [
        column_name
        for column_name in profile_result.process_columns
        if column_name != validation_order_column
    ]

    result_key = f"{selected_outcome}::{validation_order_column or 'no_order'}"
    catboost_results = st.session_state.setdefault("catboost_shap_results", {})
    run_clicked = st.button(
        "Run CatBoost + SHAP for selected outcome",
        key=f"run_catboost_shap_{selected_outcome}",
    )

    if run_clicked:
        with st.spinner("Fitting CatBoost and computing native SHAP values..."):
            catboost_results[result_key] = run_catboost_shap(
                dataframe=merged_dataframe,
                process_columns=process_columns,
                outcome_column=selected_outcome,
                validation_order_column=validation_order_column,
            )

    catboost_result = catboost_results.get(result_key)
    if catboost_result is None:
        st.info("Run the optional model to compare CatBoost/SHAP against PLS and Random Forest.")
        return

    if not catboost_result.get("available"):
        st.warning(catboost_result.get("reason", "CatBoost + SHAP is not available."))
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows used", f"{catboost_result['n_rows_used']:,}")
    metric_columns[1].metric(
        "Training R2",
        format_metric(catboost_result["training_r2"]),
    )
    metric_columns[2].metric(
        "Numeric features",
        f"{catboost_result['numeric_feature_count']:,}",
    )
    metric_columns[3].metric(
        "Categorical features",
        f"{catboost_result['categorical_feature_count']:,}",
    )

    render_time_ordered_validation(
        "Time-ordered CatBoost validation",
        catboost_result["time_ordered_validation"],
        score_label="Test R2",
    )

    st.markdown("#### CatBoost + SHAP variable importance")
    st.dataframe(
        format_numeric_columns(catboost_result["variable_importance"].head(20)),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Long SHAP contribution sample", expanded=False):
        st.caption(
            "Positive values push the model prediction upward for that batch; negative values push it downward. "
            "These are model-behavior explanations, not causal effects."
        )
        st.dataframe(
            format_numeric_columns(catboost_result["shap_values"].head(200)),
            width="stretch",
            hide_index=True,
        )


def choose_validation_order_column(audit_result, merged_dataframe) -> str | None:
    """Choose the first audit-detected date/sequence column for ordered validation."""
    if audit_result.date_or_drift_columns.empty:
        return None

    for column_name in audit_result.date_or_drift_columns["column"].tolist():
        if column_name in merged_dataframe.columns:
            return column_name

    return None


def render_analysis_results(
    analysis_results,
    merged_dataframe,
    profile_result,
    audit_result,
    spec_assessment,
) -> None:
    """Render Phase 5 analysis output with charts and tables."""
    st.markdown(
        '<div class="bia-section-kicker">Step 4 - Driver analysis</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Analysis results")

    ranked_drivers = analysis_results["ranked_drivers"]
    outcome_options = ranked_drivers["outcome"].drop_duplicates().tolist()

    overview_tab, top_drivers_tab, methods_tab, specs_tab, explanation_tab = st.tabs(
        ["Overview", "Top drivers", "Individual methods", "Specs & margins", "Explanation"]
    )

    with overview_tab:
        render_method_explainers(["ranked_drivers"], expanded=False)
        selected_overview_outcome = st.selectbox(
            "Outcome",
            options=outcome_options,
            key="overview_outcome",
        )
        top_overview_drivers = ranked_drivers[
            ranked_drivers["outcome"] == selected_overview_outcome
        ].head(5)

        metric_columns = st.columns(4)
        metric_columns[0].metric("Outcomes analyzed", f"{len(outcome_options):,}")
        metric_columns[1].metric(
            "High-confidence drivers",
            f"{int((ranked_drivers['confidence'] == 'high').sum()):,}",
        )
        metric_columns[2].metric(
            "Medium-confidence drivers",
            f"{int((ranked_drivers['confidence'] == 'medium').sum()):,}",
        )
        metric_columns[3].metric(
            "PCA cumulative variance",
            format_percent(
                analysis_results["pca"]["explained_variance"][
                    "cumulative_variance_ratio"
                ].iloc[-1]
            ),
        )

        validation_order_column = analysis_results.get("validation_order_column")
        if validation_order_column:
            st.info(
                f"Time-ordered validation used `{validation_order_column}`: models train on earlier rows and test on the most recent rows. "
                "That column is excluded from driver modeling to reduce date/sequence leakage."
            )
        else:
            st.info(
                "No date or sequence column was available for time-ordered validation. Use cross-validation and OOB scores cautiously."
            )

        render_deterministic_key_findings(
            profile_result=profile_result,
            audit_result=audit_result,
            analysis_results=analysis_results,
            merged_dataframe=merged_dataframe,
            outcomes=outcome_options,
            spec_assessment=spec_assessment,
            selected_outcome=selected_overview_outcome,
        )

        left_column, right_column = st.columns([1, 1])
        with left_column:
            st.markdown("#### Top drivers for selected outcome")
            st.dataframe(
                format_numeric_columns(
                    top_overview_drivers[
                        [
                            "rank",
                            "process_variable",
                            "combined_score",
                            "confidence",
                            "evidence_methods",
                        ]
                    ]
                ),
                width="stretch",
                hide_index=True,
            )

        with right_column:
            st.markdown("#### Outcome distribution")
            st.plotly_chart(
                outcome_distribution(merged_dataframe, selected_overview_outcome),
                width="stretch",
                key=f"overview_outcome_distribution_{selected_overview_outcome}",
            )

        st.markdown("#### PCA map")
        st.plotly_chart(
            pca_scatter(
                analysis_results["pca"],
                merged_dataframe[selected_overview_outcome],
            ),
            width="stretch",
            key=f"overview_pca_scatter_{selected_overview_outcome}",
        )

    with top_drivers_tab:
        render_method_explainers(
            ["ranked_drivers", "response_bands", "interaction_screen", "catboost_shap", "opls"],
            expanded=False,
        )
        selected_outcome = st.selectbox(
            "Outcome",
            options=outcome_options,
            key="ranked_drivers_outcome",
        )
        outcome_drivers = ranked_drivers[ranked_drivers["outcome"] == selected_outcome].head(15)
        operating_window_hints = build_operating_window_hints(
            ranked_drivers=ranked_drivers,
            merged_dataframe=merged_dataframe,
            profile_result=profile_result,
            outcomes=[selected_outcome],
        )

        st.markdown("#### Ranked driver signals")
        st.plotly_chart(
            variable_importance_bar(
                ranked_drivers=ranked_drivers,
                outcome=selected_outcome,
                top_n=15,
            ),
            width="stretch",
            key=f"top_drivers_importance_{selected_outcome}",
        )

        st.markdown("#### Ranked driver table")
        st.dataframe(format_numeric_columns(outcome_drivers), width="stretch", hide_index=True)
        render_driver_confidence_breakdown(
            outcome_drivers=outcome_drivers,
            analysis_results=analysis_results,
            profile_result=profile_result,
            audit_result=audit_result,
            selected_outcome=selected_outcome,
        )

        if not operating_window_hints.empty:
            st.markdown("#### Suggested historical response bands")
            st.caption(
                "Broad bands are more robust quartile summaries. Refined bins are narrower best-observed regions, "
                "but they are more sensitive to noise. Use both as investigation targets, not validated setpoints or spec changes."
            )
            display_columns = [
                "process_variable",
                "objective",
                "range_label",
                "refined_range_label",
                "mean_in_range",
                "refined_mean",
                "mean_outside_range",
                "directional_lift_vs_outside",
                "pattern_type",
                "evidence_note",
            ]
            available_display_columns = [
                column_name
                for column_name in display_columns
                if column_name in operating_window_hints.columns
            ]
            display_hints = operating_window_hints[available_display_columns].rename(
                columns={
                    "range_label": "broad_quartile_band",
                    "refined_range_label": "best_narrow_bin",
                    "mean_in_range": "broad_band_mean",
                    "refined_mean": "narrow_bin_mean",
                }
            )
            st.dataframe(
                format_numeric_columns(display_hints),
                width="stretch",
                hide_index=True,
            )

    with methods_tab:
        render_method_explainers(["pca", "pls", "random_forest", "catboost_shap"], expanded=False)
        pca_method_tab, pls_method_tab, random_forest_method_tab, catboost_shap_method_tab = st.tabs(
            ["PCA", "PLS regression", "Random Forest", "CatBoost + SHAP"]
        )

        with pca_method_tab:
            st.markdown("#### Scree plot")
            st.plotly_chart(
                scree_plot(analysis_results["pca"]["explained_variance"]),
                width="stretch",
                key="methods_pca_scree_plot",
            )

            st.markdown("#### Loading plot")
            st.plotly_chart(
                loading_plot(analysis_results["pca"]["loadings"]),
                width="stretch",
                key="methods_pca_loading_plot",
            )

            st.markdown("#### Top process variables by PCA loading")
            st.dataframe(
                format_numeric_columns(analysis_results["pca"]["process_loadings"].head(15)),
                width="stretch",
                hide_index=True,
            )

        with pls_method_tab:
            selected_pls_outcome = st.selectbox(
                "Outcome",
                options=outcome_options,
                key="pls_outcome",
            )
            pls_result = analysis_results["pls"][selected_pls_outcome]

            metric_columns = st.columns(3)
            metric_columns[0].metric("Rows used", f"{pls_result['n_rows_used']:,}")
            metric_columns[1].metric("Components", f"{pls_result['n_components']:,}")
            metric_columns[2].metric("CV Q2", format_metric(pls_result["q2"]))

            render_time_ordered_validation(
                "Time-ordered PLS validation",
                pls_result["time_ordered_validation"],
                score_label="Test Q2",
            )

            st.markdown("#### Variable importance")
            st.dataframe(
                format_numeric_columns(pls_result["variable_importance"].head(15)),
                width="stretch",
                hide_index=True,
            )

            with st.expander("Feature-level PLS coefficients", expanded=False):
                st.dataframe(
                    format_numeric_columns(pls_result["feature_coefficients"].head(30)),
                    width="stretch",
                    hide_index=True,
                )

            with st.expander("PLS cross-validation by component count", expanded=False):
                st.dataframe(
                    format_numeric_columns(pls_result["cv_results"]),
                    width="stretch",
                    hide_index=True,
                )

        with random_forest_method_tab:
            selected_rf_outcome = st.selectbox(
                "Outcome",
                options=outcome_options,
                key="rf_outcome",
            )
            random_forest_result = analysis_results["random_forest"][selected_rf_outcome]

            metric_columns = st.columns(3)
            metric_columns[0].metric("Rows used", f"{random_forest_result['n_rows_used']:,}")
            metric_columns[1].metric(
                "Training R2",
                format_metric(random_forest_result["training_r2"]),
            )
            metric_columns[2].metric(
                "OOB R2",
                format_metric(random_forest_result["oob_r2"]),
            )
            st.caption(
                "Training R2 is shown for fit diagnostics only. OOB R2 and time-ordered test R2 are better indicators of predictive usefulness."
            )

            render_time_ordered_validation(
                "Time-ordered Random Forest validation",
                random_forest_result["time_ordered_validation"],
                score_label="Test R2",
            )

            st.markdown("#### Variable importance")
            st.dataframe(
                format_numeric_columns(random_forest_result["variable_importance"].head(15)),
                width="stretch",
                hide_index=True,
            )

            with st.expander("Feature-level Random Forest importances", expanded=False):
                st.dataframe(
                    format_numeric_columns(random_forest_result["feature_importances"].head(30)),
                    width="stretch",
                    hide_index=True,
                )

        with catboost_shap_method_tab:
            render_catboost_shap_tab(
                merged_dataframe=merged_dataframe,
                profile_result=profile_result,
                analysis_results=analysis_results,
                outcome_options=outcome_options,
            )

    with specs_tab:
        render_method_explainers(["specs"], expanded=False)
        render_spec_results(spec_assessment, merged_dataframe, outcome_options)

    with explanation_tab:
        render_method_explainers(["llm_interpretation"], expanded=False)
        render_explanation_controls(
            profile_result=profile_result,
            audit_result=audit_result,
            analysis_results=analysis_results,
            merged_dataframe=merged_dataframe,
            outcomes=outcome_options,
            spec_assessment=spec_assessment,
        )
        render_deterministic_key_findings(
            profile_result=profile_result,
            audit_result=audit_result,
            analysis_results=analysis_results,
            merged_dataframe=merged_dataframe,
            outcomes=outcome_options,
            spec_assessment=spec_assessment,
        )

        st.markdown("#### Current top findings")
        top_findings = (
            ranked_drivers[ranked_drivers["rank"] <= 3]
            .sort_values(["outcome", "rank"])
            .reset_index(drop=True)
        )
        st.dataframe(
            format_numeric_columns(
                top_findings[
                    [
                        "outcome",
                        "rank",
                        "process_variable",
                        "confidence",
                        "evidence_methods",
                        "combined_score",
                    ]
                ]
            ),
            width="stretch",
            hide_index=True,
        )
        render_pdf_report_download(
            profile_result=profile_result,
            audit_result=audit_result,
            analysis_results=analysis_results,
            merged_dataframe=merged_dataframe,
            outcomes=outcome_options,
            spec_assessment=spec_assessment,
        )


def render_pdf_report_download(
    profile_result,
    audit_result,
    analysis_results,
    merged_dataframe,
    outcomes: list[str],
    spec_assessment=None,
) -> None:
    """Render PDF report generation controls."""
    st.markdown("#### PDF report")
    st.caption(
        "Creates a local PDF with data summary, key findings, top-driver charts, PCA overview, specs, and ranked-driver appendix."
    )

    prepare_clicked = st.button(
        "Prepare PDF report",
        key="prepare_pdf_report",
    )
    if prepare_clicked:
        try:
            st.session_state["pdf_report_bytes"] = generate_analysis_report_pdf(
                profile_result=profile_result,
                audit_result=audit_result,
                analysis_results=analysis_results,
                merged_dataframe=merged_dataframe,
                outcomes=outcomes,
                interpretation_markdown=st.session_state.get("ollama_interpretation", ""),
                interpretation_validation_warnings=st.session_state.get(
                    "interpretation_validation_warnings",
                    [],
                ),
                spec_assessment=spec_assessment,
            )
        except (ReportGenerationError, OSError, ValueError, RuntimeError) as error:
            st.error(f"Could not build PDF report: {error}")
            st.session_state["pdf_report_bytes"] = None

    if st.session_state.get("pdf_report_bytes"):
        st.download_button(
            "Download PDF report",
            data=st.session_state["pdf_report_bytes"],
            file_name=f"batch_insight_report_{datetime_stamp_for_filename()}.pdf",
            mime="application/pdf",
            key="download_pdf_report",
        )


def render_explanation_controls(
    profile_result,
    audit_result,
    analysis_results,
    merged_dataframe,
    outcomes: list[str],
    spec_assessment=None,
) -> None:
    """Render local Ollama interpretation controls."""
    st.markdown("#### Local interpretation")
    st.caption(
        "Uses your Ollama server. Local models stay on this machine; cloud-tagged models are routed through Ollama Cloud."
    )

    base_url = st.text_input(
        "Ollama URL",
        value=get_ollama_base_url(),
        key="ollama_base_url",
    )

    available_models = []
    model_list_error = None
    try:
        available_models = get_available_ollama_models(base_url)
    except OllamaInterpreterError as error:
        model_list_error = str(error)

    model_options = build_model_options(available_models)

    if model_list_error:
        st.warning(model_list_error)
        selected_model = st.text_input(
            "Ollama model",
            value="nemotron-3-super:cloud",
            key="ollama_model_text",
        )
    elif model_options:
        default_model = choose_default_model(model_options)
        default_index = model_options.index(default_model) if default_model in model_options else 0
        selected_model = st.selectbox(
            "Ollama model",
            options=model_options,
            index=default_index,
            key="ollama_model_select",
        )
    else:
        st.warning("Ollama is reachable, but no local models were found.")
        selected_model = st.text_input(
            "Ollama model",
            value="llama3.1",
            key="ollama_model_text_empty",
        )

    if is_cloud_model(selected_model):
        st.warning(
            "Cloud model selected. The app still calls your local Ollama server, but Ollama may send the analysis summary to Ollama Cloud. "
            "Use this only with data you are comfortable sending outside this machine."
        )

    generated_this_run = False
    generate_clicked = st.button(
        "Generate interpretation",
        type="primary",
        disabled=not selected_model,
        key="generate_ollama_interpretation",
    )

    if generate_clicked:
        generated_this_run = True
        st.session_state["ollama_interpretation"] = ""
        try:
            with st.spinner(f"Generating interpretation with {selected_model}..."):
                interpretation_text = st.write_stream(
                    stream_interpretation(
                        profile_result=profile_result,
                        audit_result=audit_result,
                        analysis_results=analysis_results,
                        merged_dataframe=merged_dataframe,
                        outcomes=outcomes,
                        spec_assessment=spec_assessment,
                        model=selected_model,
                        base_url=base_url,
                    )
                )
            st.session_state["ollama_interpretation"] = sanitize_interpretation_text(
                interpretation_text
            )
            report_pack = build_report_pack(
                profile_result=profile_result,
                audit_result=audit_result,
                analysis_results=analysis_results,
                merged_dataframe=merged_dataframe,
                outcomes=outcomes,
                spec_assessment=spec_assessment,
            )
            validation_result = validate_interpretation_text(
                st.session_state["ollama_interpretation"],
                report_pack,
            )
            st.session_state["interpretation_validation_warnings"] = validation_result.warnings
            st.session_state["pdf_report_bytes"] = None
        except OllamaInterpreterError as error:
            st.error(str(error))

    if st.session_state.get("ollama_interpretation"):
        if not generated_this_run:
            st.markdown(st.session_state["ollama_interpretation"])

        render_interpretation_validation_warnings()

        st.download_button(
            "Download interpretation as Markdown",
            data=st.session_state["ollama_interpretation"],
            file_name="batch_insight_interpretation.md",
            mime="text/markdown",
            key="download_ollama_interpretation",
        )


def render_interpretation_validation_warnings() -> None:
    """Show post-generation validation warnings for the LLM narrative."""
    validation_warnings = st.session_state.get("interpretation_validation_warnings", [])
    if not validation_warnings:
        st.success("Report validation found no obvious hallucination or overclaim warnings.")
        return

    with st.expander("Report validation warnings", expanded=True):
        st.warning(
            "Review the generated interpretation before using it externally. "
            "These warnings are heuristic checks, not a final approval system."
        )
        for warning in validation_warnings:
            st.write(f"- {warning}")


def format_percent(value) -> str:
    """Format a decimal ratio as a percentage."""
    try:
        if value != value:
            return "n/a"
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "n/a"


def format_numeric_columns(dataframe):
    """Round numeric columns for compact display."""
    display_dataframe = dataframe.copy()
    numeric_columns = display_dataframe.select_dtypes(include="number").columns
    display_dataframe[numeric_columns] = display_dataframe[numeric_columns].round(3)
    return display_dataframe


def format_metric(value) -> str:
    """Format a metric value that may be NaN."""
    try:
        if value != value:
            return "n/a"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"


def datetime_stamp_for_filename() -> str:
    """Return a compact timestamp for generated downloads."""
    return datetime.now().strftime("%Y%m%d_%H%M")


def main() -> None:
    apply_custom_theme()

    st.session_state.setdefault("merged_dataframe", None)
    st.session_state.setdefault("merge_result", None)
    st.session_state.setdefault("selected_outcome_columns", [])
    st.session_state.setdefault("profile_result", None)
    st.session_state.setdefault("audit_result", None)
    st.session_state.setdefault("spec_dataframe", None)
    st.session_state.setdefault("spec_assessment", None)
    st.session_state.setdefault("analysis_requested", False)
    st.session_state.setdefault("analysis_results", None)
    st.session_state.setdefault("ollama_interpretation", "")
    st.session_state.setdefault("interpretation_validation_warnings", [])
    st.session_state.setdefault("pdf_report_bytes", None)
    st.session_state.setdefault("catboost_shap_results", {})

    render_sidebar()
    render_header()
    render_documentation_section()

    st.markdown(
        '<div class="bia-section-kicker">Step 1 - Upload files</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Upload and prepare source data")
    st.caption(
        "Start with either two separate files or one Excel workbook with process and QC data on different sheets."
    )
    render_intake_workflow_explainer()
    render_data_template_downloads()

    same_workbook_mode = st.checkbox(
        "Process and QC data are in the same Excel workbook",
        value=False,
        key="same_workbook_mode",
        help="Use this when one workbook has separate process and analytical/QC sheets.",
    )

    if same_workbook_mode:
        combined_file = st.file_uploader(
            "Upload combined process/QC workbook",
            type=["csv", "xlsx", "xls"],
            key="combined_workbook_file",
            help="Excel workbooks are preferred for this mode because process and QC can be selected from different sheets.",
        )
        process_file = combined_file
        qc_file = combined_file
        if combined_file is not None and Path(getattr(combined_file, "name", "")).suffix.lower() == ".csv":
            st.warning(
                "A CSV has only one table. Same-workbook mode is most useful for XLSX/XLS files with multiple sheets."
            )
    else:
        left_column, right_column = st.columns(2)
        with left_column:
            process_file = st.file_uploader(
                "Upload batch process data",
                type=["csv", "xlsx", "xls"],
                key="process_file",
            )

        with right_column:
            qc_file = st.file_uploader(
                "Upload QC results",
                type=["csv", "xlsx", "xls"],
                key="qc_file",
            )

    process_dataframe, process_intake_metadata = render_file_intake(
        process_file,
        "Process data",
        "process_intake",
    )
    qc_dataframe, qc_intake_metadata = render_file_intake(
        qc_file,
        "QC results",
        "qc_intake",
    )

    if process_dataframe is None or qc_dataframe is None:
        st.info("Upload both files to choose batch ID columns and quality outcomes.")
        return

    show_dataframe_preview("process data", process_dataframe)
    show_dataframe_preview("QC results", qc_dataframe)

    st.markdown(
        '<div class="bia-section-kicker">Step 2 - Match batches</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Match batches")

    process_batch_default_index = get_batch_column_default_index(process_dataframe.columns)
    qc_batch_default_index = get_batch_column_default_index(qc_dataframe.columns)

    process_batch_id_column = st.selectbox(
        "Batch ID column in process data",
        options=list(process_dataframe.columns),
        index=process_batch_default_index,
    )
    qc_batch_id_column = st.selectbox(
        "Batch ID column in QC results",
        options=list(qc_dataframe.columns),
        index=qc_batch_default_index,
    )

    outcome_options = [
        column_name for column_name in qc_dataframe.columns if column_name != qc_batch_id_column
    ]
    default_outcomes = get_default_outcome_columns(qc_dataframe, qc_batch_id_column)

    selected_outcome_columns = st.multiselect(
        "Quality outcome columns",
        options=outcome_options,
        default=[column_name for column_name in default_outcomes if column_name in outcome_options],
    )

    process_duplicate_strategy = render_duplicate_strategy(
        "Process data",
        process_dataframe,
        process_batch_id_column,
        "process_duplicate_strategy",
    )
    qc_duplicate_strategy = render_duplicate_strategy(
        "QC data",
        qc_dataframe,
        qc_batch_id_column,
        "qc_duplicate_strategy",
    )

    readiness_result = calculate_data_readiness(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column=process_batch_id_column,
        qc_batch_id_column=qc_batch_id_column,
        outcome_columns=selected_outcome_columns,
        process_duplicate_strategy=process_duplicate_strategy,
        qc_duplicate_strategy=qc_duplicate_strategy,
        process_numeric_parse_report=process_intake_metadata.get("numeric_parse_report"),
        qc_numeric_parse_report=qc_intake_metadata.get("numeric_parse_report"),
    )
    render_readiness_panel(readiness_result)

    render_mapping_profile_download(
        process_file=process_file,
        qc_file=qc_file,
        process_metadata=process_intake_metadata,
        qc_metadata=qc_intake_metadata,
        process_batch_id_column=process_batch_id_column,
        qc_batch_id_column=qc_batch_id_column,
        selected_outcome_columns=selected_outcome_columns,
        process_duplicate_strategy=process_duplicate_strategy,
        qc_duplicate_strategy=qc_duplicate_strategy,
    )

    spec_file = render_spec_input()

    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        disabled=not selected_outcome_columns or bool(readiness_result.blockers),
    )

    if analyze_clicked:
        spec_dataframe = None
        if spec_file is not None:
            try:
                spec_dataframe = load_tabular_file(spec_file)
            except DataPrepError as error:
                st.error(f"Spec/window file: {error}")
                return

        try:
            process_duplicate_result, qc_duplicate_result = apply_duplicate_handling_for_merge(
                process_dataframe=process_dataframe,
                qc_dataframe=qc_dataframe,
                process_batch_id_column=process_batch_id_column,
                qc_batch_id_column=qc_batch_id_column,
                process_duplicate_strategy=process_duplicate_strategy,
                qc_duplicate_strategy=qc_duplicate_strategy,
            )
            for warning in [*process_duplicate_result.warnings, *qc_duplicate_result.warnings]:
                st.warning(warning)

            merge_result = merge_process_and_qc_data(
                process_dataframe=process_duplicate_result.dataframe,
                qc_dataframe=qc_duplicate_result.dataframe,
                process_batch_id_column=process_batch_id_column,
                qc_batch_id_column=qc_batch_id_column,
                outcome_columns=selected_outcome_columns,
            )
        except DataPrepError as error:
            st.error(str(error))
            return

        st.session_state["merged_dataframe"] = merge_result.dataframe
        st.session_state["merge_result"] = merge_result
        st.session_state["selected_outcome_columns"] = selected_outcome_columns
        st.session_state["spec_dataframe"] = spec_dataframe
        st.session_state["spec_assessment"] = None
        st.session_state["analysis_requested"] = False
        st.session_state["analysis_results"] = None
        st.session_state["ollama_interpretation"] = ""
        st.session_state["interpretation_validation_warnings"] = []
        st.session_state["pdf_report_bytes"] = None
        st.session_state["catboost_shap_results"] = {}
        st.session_state["audit_result"] = None

    if st.session_state["merge_result"] is not None:
        render_merge_summary(st.session_state["merge_result"])

        merged_dataframe = st.session_state["merged_dataframe"]
        selected_outcomes = st.session_state["selected_outcome_columns"]
        profile_result = profile_merged_data(
            merged_dataframe=merged_dataframe,
            outcome_columns=selected_outcomes,
            matched_batch_count=st.session_state["merge_result"].matched_batch_count,
        )
        st.session_state["profile_result"] = profile_result
        audit_result = run_preflight_audit(
            dataframe=merged_dataframe,
            process_columns=profile_result.process_columns,
            outcome_columns=profile_result.outcome_columns,
            variable_types=profile_result.variable_types,
        )
        st.session_state["audit_result"] = audit_result
        ranked_drivers_for_specs = None
        if st.session_state["analysis_results"] is not None:
            ranked_drivers_for_specs = st.session_state["analysis_results"]["ranked_drivers"]

        try:
            spec_assessment = assess_specs(
                dataframe=merged_dataframe,
                spec_dataframe=st.session_state["spec_dataframe"],
                process_columns=profile_result.process_columns,
                outcome_columns=profile_result.outcome_columns,
                ranked_drivers=ranked_drivers_for_specs,
                audit_result=audit_result,
            )
        except SpecError as error:
            st.error(f"Spec/window file: {error}")
            spec_assessment = None
        st.session_state["spec_assessment"] = spec_assessment

        with st.expander("Preview merged data", expanded=False):
            st.dataframe(
                merged_dataframe.head(20),
                width="stretch",
            )

        render_profile(profile_result, merged_dataframe)
        render_preflight_audit(audit_result)
        render_spec_assessment(spec_assessment)
        render_qc_trend_overview(merged_dataframe, profile_result, audit_result)

        run_analysis_clicked = st.button(
            "Run analysis",
            type="primary",
            disabled=profile_result.outcome_statistics["count"].eq(0).any(),
        )
        if run_analysis_clicked:
            st.session_state["analysis_requested"] = True
            with st.spinner(
                "Running PCA, PLS regression, and Random Forest. This may take a little while."
            ):
                try:
                    validation_order_column = choose_validation_order_column(
                        audit_result,
                        merged_dataframe,
                    )
                    modeling_process_columns = [
                        column_name
                        for column_name in profile_result.process_columns
                        if column_name != validation_order_column
                    ]
                    st.session_state["analysis_results"] = run_all_analyses(
                        dataframe=merged_dataframe,
                        process_columns=modeling_process_columns,
                        outcome_columns=profile_result.outcome_columns,
                        validation_order_column=validation_order_column,
                    )
                    if st.session_state["spec_dataframe"] is not None:
                        st.session_state["spec_assessment"] = assess_specs(
                            dataframe=merged_dataframe,
                            spec_dataframe=st.session_state["spec_dataframe"],
                            process_columns=profile_result.process_columns,
                            outcome_columns=profile_result.outcome_columns,
                            ranked_drivers=st.session_state["analysis_results"]["ranked_drivers"],
                            audit_result=audit_result,
                        )
                    st.session_state["ollama_interpretation"] = ""
                    st.session_state["interpretation_validation_warnings"] = []
                    st.session_state["pdf_report_bytes"] = None
                    st.session_state["catboost_shap_results"] = {}
                except (ValueError, SpecError) as error:
                    st.error(str(error))
                    st.session_state["analysis_results"] = None

        if st.session_state["analysis_results"] is not None:
            render_analysis_results(
                st.session_state["analysis_results"],
                merged_dataframe,
                profile_result,
                audit_result,
                st.session_state["spec_assessment"],
            )


if __name__ == "__main__":
    main()
