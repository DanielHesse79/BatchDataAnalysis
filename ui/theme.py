"""Page styling, header, sidebar, and the in-app documentation reader."""

from __future__ import annotations
from pathlib import Path

import streamlit as st


# This module lives in ui/, so the project root is one level up.
APP_ROOT = Path(__file__).resolve().parent.parent

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
