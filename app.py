"""Streamlit entry point for Batch Insight Analyzer."""

from __future__ import annotations

import streamlit as st

from analysis.audit import run_preflight_audit
from analysis.data_prep import (
    DataPrepError,
    get_batch_column_default_index,
    get_default_outcome_columns,
    load_tabular_file,
    merge_process_and_qc_data,
)
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
from analysis.methods import run_all_analyses
from analysis.profiling import profile_merged_data
from utils.plots import (
    loading_plot,
    outcome_distribution,
    pca_scatter,
    qc_control_chart,
    qc_control_summary,
    qc_trend_plot,
    scree_plot,
    variable_importance_bar,
)


st.set_page_config(
    page_title="Batch Insight Analyzer",
    page_icon="BIA",
    layout="wide",
)


def render_header() -> None:
    """Show a clear location marker at the top of the app."""
    st.title("Batch Insight Analyzer")
    st.caption("Phase 6: local upload, analysis, visualization, and Ollama interpretation.")


def load_uploaded_dataframe(uploaded_file, label: str):
    """Load one uploaded file and display a Streamlit-friendly error if needed."""
    if uploaded_file is None:
        return None

    try:
        dataframe = load_tabular_file(uploaded_file)
    except DataPrepError as error:
        st.error(f"{label}: {error}")
        return None

    st.success(f"{label}: loaded {len(dataframe):,} rows and {len(dataframe.columns):,} columns.")
    return dataframe


def show_dataframe_preview(label: str, dataframe) -> None:
    """Display a compact preview without overwhelming the user."""
    with st.expander(f"Preview {label}", expanded=False):
        st.dataframe(dataframe.head(10), width="stretch")


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


def choose_validation_order_column(audit_result, merged_dataframe) -> str | None:
    """Choose the first audit-detected date/sequence column for ordered validation."""
    if audit_result.date_or_drift_columns.empty:
        return None

    for column_name in audit_result.date_or_drift_columns["column"].tolist():
        if column_name in merged_dataframe.columns:
            return column_name

    return None


def render_analysis_results(analysis_results, merged_dataframe, profile_result, audit_result) -> None:
    """Render Phase 5 analysis output with charts and tables."""
    st.subheader("Analysis results")

    ranked_drivers = analysis_results["ranked_drivers"]
    outcome_options = ranked_drivers["outcome"].drop_duplicates().tolist()

    overview_tab, top_drivers_tab, methods_tab, explanation_tab = st.tabs(
        ["Overview", "Top drivers", "Individual methods", "Explanation"]
    )

    with overview_tab:
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
        selected_outcome = st.selectbox(
            "Outcome",
            options=outcome_options,
            key="ranked_drivers_outcome",
        )
        outcome_drivers = ranked_drivers[ranked_drivers["outcome"] == selected_outcome].head(15)

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

    with methods_tab:
        pca_method_tab, pls_method_tab, random_forest_method_tab = st.tabs(
            ["PCA", "PLS regression", "Random Forest"]
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

    with explanation_tab:
        render_explanation_controls(
            profile_result=profile_result,
            audit_result=audit_result,
            analysis_results=analysis_results,
            merged_dataframe=merged_dataframe,
            outcomes=outcome_options,
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


def render_explanation_controls(
    profile_result,
    audit_result,
    analysis_results,
    merged_dataframe,
    outcomes: list[str],
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
                        model=selected_model,
                        base_url=base_url,
                    )
                )
            st.session_state["ollama_interpretation"] = sanitize_interpretation_text(
                interpretation_text
            )
        except OllamaInterpreterError as error:
            st.error(str(error))

    if st.session_state.get("ollama_interpretation"):
        if not generated_this_run:
            st.markdown(st.session_state["ollama_interpretation"])

        st.download_button(
            "Download interpretation as Markdown",
            data=st.session_state["ollama_interpretation"],
            file_name="batch_insight_interpretation.md",
            mime="text/markdown",
            key="download_ollama_interpretation",
        )


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


def main() -> None:
    render_header()

    st.session_state.setdefault("merged_dataframe", None)
    st.session_state.setdefault("merge_result", None)
    st.session_state.setdefault("selected_outcome_columns", [])
    st.session_state.setdefault("profile_result", None)
    st.session_state.setdefault("audit_result", None)
    st.session_state.setdefault("analysis_requested", False)
    st.session_state.setdefault("analysis_results", None)
    st.session_state.setdefault("ollama_interpretation", "")

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

    process_dataframe = load_uploaded_dataframe(process_file, "Process data")
    qc_dataframe = load_uploaded_dataframe(qc_file, "QC results")

    if process_dataframe is None or qc_dataframe is None:
        st.info("Upload both files to choose batch ID columns and quality outcomes.")
        return

    show_dataframe_preview("process data", process_dataframe)
    show_dataframe_preview("QC results", qc_dataframe)

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

    analyze_clicked = st.button(
        "Analyze",
        type="primary",
        disabled=not selected_outcome_columns,
    )

    if analyze_clicked:
        try:
            merge_result = merge_process_and_qc_data(
                process_dataframe=process_dataframe,
                qc_dataframe=qc_dataframe,
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
        st.session_state["analysis_requested"] = False
        st.session_state["analysis_results"] = None
        st.session_state["ollama_interpretation"] = ""
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

        with st.expander("Preview merged data", expanded=False):
            st.dataframe(
                merged_dataframe.head(20),
                width="stretch",
            )

        render_profile(profile_result, merged_dataframe)
        render_preflight_audit(audit_result)
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
                    st.session_state["ollama_interpretation"] = ""
                except ValueError as error:
                    st.error(str(error))
                    st.session_state["analysis_results"] = None

        if st.session_state["analysis_results"] is not None:
            render_analysis_results(
                st.session_state["analysis_results"],
                merged_dataframe,
                profile_result,
                audit_result,
            )


if __name__ == "__main__":
    main()
