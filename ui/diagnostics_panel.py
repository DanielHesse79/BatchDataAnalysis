"""Step 3: what the merged data looks like before any model is fitted."""

from __future__ import annotations
from analysis.methods import choose_validation_order_column
from analysis.profiling import get_categorical_color_options
from utils.format import (
    format_metric,
    format_numeric_columns,
)
from utils.plots import (
    outcome_distribution,
    qc_control_chart,
    qc_control_summary,
    qc_trend_plot,
)

import streamlit as st



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

@st.fragment
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

    color_options = ["None"] + get_categorical_color_options(profile_result, merged_dataframe)
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
