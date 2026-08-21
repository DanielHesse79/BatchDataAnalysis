"""Shared entry point for the two Batch Insight analysis workspaces."""

from __future__ import annotations

import streamlit as st

from app import main as render_batch_analysis
from qc_intel.app import main as render_qc_monitoring


def render_workspace_chooser() -> None:
    """Explain the product split and let the user enter either workspace."""
    st.title("Batch Insight")
    st.caption("Choose the analysis workspace that matches the question you need to answer.")

    batch_column, qc_column = st.columns(2, gap="large")

    with batch_column:
        with st.container(border=True):
            st.subheader("Batch Insight Analyzer")
            st.markdown("**Understand which process conditions are associated with QC outcomes.**")
            st.write(
                "Combine batch process data with QC results, compare PCA, PLS and tree models, "
                "inspect response bands, and prepare a traceable interpretation or PDF report."
            )
            st.caption("Best for: cross-sectional analysis across a set of manufacturing batches.")
            st.page_link(
                BATCH_ANALYSIS_PAGE,
                label="Open batch driver analysis",
                icon=":material/account_tree:",
                use_container_width=True,
            )

    with qc_column:
        with st.container(border=True):
            st.subheader("QC Intelligence Layer")
            st.markdown("**Monitor whether an analytical method is changing over time.**")
            st.write(
                "Trend QC bias and precision by method, instrument and QC level, rank drift "
                "signals, compare instruments, and trace every point back to its source file."
            )
            st.caption("Best for: longitudinal QC monitoring across runs, instruments and months.")
            st.page_link(
                QC_MONITORING_PAGE,
                label="Open QC monitoring",
                icon=":material/monitoring:",
                use_container_width=True,
            )

    st.info(
        "The workspaces are deliberately separate. Batch driver analysis asks what varied "
        "between manufacturing batches; QC monitoring asks whether an analytical method is "
        "quietly moving over time. Use the navigation above to switch workspace at any time."
    )


START_PAGE = st.Page(
    render_workspace_chooser,
    title="Choose workspace",
    icon=":material/home:",
    url_path="home",
    default=True,
)
BATCH_ANALYSIS_PAGE = st.Page(
    render_batch_analysis,
    title="Batch drivers",
    icon=":material/account_tree:",
    url_path="batch-drivers",
)
QC_MONITORING_PAGE = st.Page(
    render_qc_monitoring,
    title="QC monitoring",
    icon=":material/monitoring:",
    url_path="qc-monitoring",
)


def main() -> None:
    """Configure and run the shared top-level navigation."""
    st.set_page_config(page_title="Batch Insight", page_icon="BI", layout="wide")
    navigation = st.navigation(
        [START_PAGE, BATCH_ANALYSIS_PAGE, QC_MONITORING_PAGE],
        position="top",
    )
    navigation.run()


if __name__ == "__main__":
    main()
