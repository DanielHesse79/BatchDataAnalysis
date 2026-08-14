"""Streamlit entry point for Batch Insight Analyzer.

This module owns page setup, session state, and the stage-by-stage flow.
Rendering lives in `ui/`; analysis lives in `analysis/` and `utils/`.
"""

from __future__ import annotations
from pathlib import Path
from analysis.audit import run_preflight_audit
from analysis.data_prep import (
    DataPrepError,
    get_batch_column_default_index,
    get_default_outcome_columns,
    merge_process_and_qc_data,
)
from analysis.methods import (
    choose_validation_order_column,
    get_modeling_process_columns,
    run_all_analyses,
)
from analysis.profiling import profile_merged_data
from analysis.readiness import calculate_data_readiness
from analysis.specs import (
    SpecError,
    assess_specs,
)

import streamlit as st


from ui.state import (
    build_input_fingerprint,
    bump_analysis_run_token,
    intake_fingerprint_fields,
    load_spec_input,
    resolve_default_index,
    spec_input_fingerprint,
    uploaded_file_identity,
)
from ui.theme import (
    apply_custom_theme,
    render_documentation_section,
    render_header,
    render_intake_workflow_explainer,
    render_sidebar,
)
from ui.intake_panel import (
    apply_duplicate_handling_for_merge,
    render_data_template_downloads,
    render_duplicate_strategy,
    render_file_intake,
    render_mapping_profile_download,
    render_mapping_profile_import,
    render_readiness_panel,
    render_spec_input,
    show_dataframe_preview,
)
from ui.diagnostics_panel import (
    render_merge_summary,
    render_preflight_audit,
    render_profile,
    render_qc_trend_overview,
    render_spec_assessment,
)
from ui.results_panel import (
    render_analysis_results,
)


st.set_page_config(
    page_title="Batch Insight Analyzer",
    page_icon="BIA",
    layout="wide",
)


def main() -> None:
    apply_custom_theme()

    st.session_state.setdefault("merged_dataframe", None)
    st.session_state.setdefault("merge_result", None)
    st.session_state.setdefault("selected_outcome_columns", [])
    st.session_state.setdefault("profile_result", None)
    st.session_state.setdefault("audit_result", None)
    st.session_state.setdefault("spec_dataframe", None)
    st.session_state.setdefault("spec_assessment", None)
    st.session_state.setdefault("analysis_results", None)
    st.session_state.setdefault("analysis_input_fingerprint", None)
    st.session_state.setdefault("analysis_run_token", "")
    st.session_state.setdefault("ollama_interpretation", "")
    st.session_state.setdefault("interpretation_validation_warnings", [])
    st.session_state.setdefault("pdf_report_bytes", None)
    st.session_state.setdefault("catboost_shap_results", {})
    st.session_state.setdefault("xgboost_shap_results", {})

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

    render_mapping_profile_import(
        process_columns=list(process_dataframe.columns),
        qc_columns=list(qc_dataframe.columns),
    )
    loaded_profile = st.session_state.get("loaded_mapping_profile")

    process_batch_id_column = st.selectbox(
        "Batch ID column in process data",
        options=list(process_dataframe.columns),
        index=resolve_default_index(
            list(process_dataframe.columns),
            getattr(loaded_profile, "process_batch_id_column", None),
            get_batch_column_default_index(process_dataframe.columns),
        ),
        key="process_batch_id_column",
    )
    qc_batch_id_column = st.selectbox(
        "Batch ID column in QC results",
        options=list(qc_dataframe.columns),
        index=resolve_default_index(
            list(qc_dataframe.columns),
            getattr(loaded_profile, "qc_batch_id_column", None),
            get_batch_column_default_index(qc_dataframe.columns),
        ),
        key="qc_batch_id_column",
    )

    outcome_options = [
        column_name for column_name in qc_dataframe.columns if column_name != qc_batch_id_column
    ]
    default_outcomes = (
        getattr(loaded_profile, "outcome_columns", None)
        or get_default_outcome_columns(qc_dataframe, qc_batch_id_column)
    )

    selected_outcome_columns = st.multiselect(
        "Quality outcome columns",
        options=outcome_options,
        default=[column_name for column_name in default_outcomes if column_name in outcome_options],
        key="selected_outcome_columns_widget",
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

    spec_input = render_spec_input()

    current_input_fingerprint = build_input_fingerprint(
        process_file=uploaded_file_identity(process_file),
        qc_file=uploaded_file_identity(qc_file),
        spec_input=spec_input_fingerprint(spec_input),
        process_intake=intake_fingerprint_fields(process_intake_metadata),
        qc_intake=intake_fingerprint_fields(qc_intake_metadata),
        process_batch_id_column=process_batch_id_column,
        qc_batch_id_column=qc_batch_id_column,
        selected_outcome_columns=sorted(selected_outcome_columns),
        process_duplicate_strategy=process_duplicate_strategy,
        qc_duplicate_strategy=qc_duplicate_strategy,
    )

    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        disabled=not selected_outcome_columns or bool(readiness_result.blockers),
    )

    if analyze_clicked:
        try:
            spec_dataframe = load_spec_input(spec_input)
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

        profile_result = profile_merged_data(
            merged_dataframe=merge_result.dataframe,
            outcome_columns=selected_outcome_columns,
            batch_id_column=merge_result.batch_id_column,
            matched_batch_count=merge_result.matched_batch_count,
        )
        audit_result = run_preflight_audit(
            dataframe=merge_result.dataframe,
            process_columns=profile_result.process_columns,
            outcome_columns=profile_result.outcome_columns,
            variable_types=profile_result.variable_types,
        )
        try:
            spec_assessment = assess_specs(
                dataframe=merge_result.dataframe,
                spec_dataframe=spec_dataframe,
                process_columns=profile_result.process_columns,
                outcome_columns=profile_result.outcome_columns,
                audit_result=audit_result,
            )
        except SpecError as error:
            st.error(f"Spec/window file: {error}")
            spec_assessment = None

        st.session_state["merged_dataframe"] = merge_result.dataframe
        st.session_state["merge_result"] = merge_result
        st.session_state["selected_outcome_columns"] = selected_outcome_columns
        st.session_state["spec_dataframe"] = spec_dataframe
        st.session_state["profile_result"] = profile_result
        st.session_state["audit_result"] = audit_result
        st.session_state["spec_assessment"] = spec_assessment
        st.session_state["analysis_input_fingerprint"] = current_input_fingerprint
        st.session_state["analysis_results"] = None
        st.session_state["ollama_interpretation"] = ""
        st.session_state["interpretation_validation_warnings"] = []
        st.session_state["pdf_report_bytes"] = None
        st.session_state["catboost_shap_results"] = {}
        st.session_state["xgboost_shap_results"] = {}
        bump_analysis_run_token()

    if st.session_state["merge_result"] is not None:
        # Profiling, the audit, and the spec assessment are deterministic
        # functions of the merged data, so they are computed when Analyze runs
        # and read back here instead of on every widget interaction.
        merged_dataframe = st.session_state["merged_dataframe"]
        selected_outcomes = st.session_state["selected_outcome_columns"]
        profile_result = st.session_state["profile_result"]
        audit_result = st.session_state["audit_result"]
        spec_assessment = st.session_state["spec_assessment"]

        inputs_changed = (
            st.session_state["analysis_input_fingerprint"] != current_input_fingerprint
        )
        if inputs_changed:
            st.warning(
                "**These results are out of date.** The uploaded files or the matching "
                "settings changed after this analysis ran, so everything below still "
                "describes the previous data. Press **Analyze** again to refresh.",
                icon=":material/warning:",
            )

        render_merge_summary(st.session_state["merge_result"])

        with st.expander("Preview merged data", expanded=False):
            st.dataframe(
                merged_dataframe.head(20),
                width="stretch",
            )

        render_profile(profile_result, merged_dataframe)
        render_preflight_audit(audit_result)
        render_spec_assessment(spec_assessment)
        render_qc_trend_overview(merged_dataframe, profile_result, audit_result)

        unusable_outcomes = profile_result.outcome_statistics.loc[
            profile_result.outcome_statistics["count"].eq(0),
            "outcome",
        ].tolist()
        if unusable_outcomes:
            st.error(
                "Run analysis is blocked because these outcome column(s) contain no numeric "
                f"values: {', '.join(unusable_outcomes)}. Deselect them under "
                "'Quality outcome columns', then press Analyze again."
            )

        run_analysis_clicked = st.button(
            "Run analysis",
            type="primary",
            disabled=bool(unusable_outcomes) or inputs_changed,
            help=(
                "Press Analyze again first - the inputs changed since this merge was built."
                if inputs_changed
                else None
            ),
        )
        if run_analysis_clicked:
            with st.spinner(
                "Running PCA, PLS regression, and Random Forest. This may take a little while."
            ):
                try:
                    validation_order_column = choose_validation_order_column(
                        audit_result,
                        merged_dataframe,
                    )
                    modeling_process_columns = get_modeling_process_columns(
                        profile_result.process_columns,
                        validation_order_column,
                    )
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
                    st.session_state["xgboost_shap_results"] = {}
                except (ValueError, SpecError) as error:
                    st.error(str(error))
                    st.session_state["analysis_results"] = None
                finally:
                    bump_analysis_run_token()

            spec_assessment = st.session_state["spec_assessment"]

        if st.session_state["analysis_results"] is not None:
            render_analysis_results(
                st.session_state["analysis_results"],
                merged_dataframe,
                profile_result,
                audit_result,
                spec_assessment,
            )


if __name__ == "__main__":
    main()
