"""Step 4: ranked drivers, per-method views, and optional boosted models."""

from __future__ import annotations
from html import escape
from analysis.advanced_methods import (
    get_catboost_shap_dependency_status,
    get_xgboost_dependency_status,
    run_catboost_shap,
    run_xgboost_shap,
)
from analysis.confidence import build_driver_confidence_breakdown
from analysis.evidence import (
    build_operating_window_hints,
    get_numeric_driver_candidates,
    get_response_band_for_variable,
)
from analysis.key_findings import build_deterministic_key_findings
from analysis.method_registry import list_method_explainers
from analysis.methods import get_modeling_process_columns
from utils.format import (
    format_metric,
    format_numeric_columns,
    format_percent,
)
from utils.plots import (
    loading_plot,
    outcome_distribution,
    outcome_vs_spec_variable_plot,
    pca_scatter,
    response_shape_plot,
    scree_plot,
    spec_distribution_plot,
    spec_margin_bar,
    variable_importance_bar,
)

import pandas as pd
import streamlit as st


from ui.state import (
    build_input_fingerprint,
    discard_generated_interpretation,
    get_analysis_artifact,
    get_outcome_objective_overrides,
    get_shared_report_pack,
)
from ui.diagnostics_panel import (
    render_time_ordered_validation,
)
from ui.explanation_panel import (
    render_explanation_controls,
    render_pdf_report_download,
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
    key_findings = get_analysis_artifact(
        "key_findings",
        lambda: build_deterministic_key_findings(
            profile_result=profile_result,
            audit_result=audit_result,
            analysis_results=analysis_results,
            merged_dataframe=merged_dataframe,
            outcomes=outcomes,
            spec_assessment=spec_assessment,
            report_pack=get_shared_report_pack(
                profile_result=profile_result,
                audit_result=audit_result,
                analysis_results=analysis_results,
                merged_dataframe=merged_dataframe,
                outcomes=outcomes,
                spec_assessment=spec_assessment,
            ),
        ),
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

@st.fragment
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

@st.fragment
def render_response_shape_section(
    merged_dataframe: pd.DataFrame,
    profile_result,
    selected_outcome: str,
    outcome_drivers: pd.DataFrame,
    operating_window_hints: pd.DataFrame,
) -> None:
    """Render response-shape plot for top numeric drivers."""
    numeric_driver_candidates = get_numeric_driver_candidates(
        outcome_drivers=outcome_drivers,
        profile_result=profile_result,
        merged_dataframe=merged_dataframe,
    )
    if not numeric_driver_candidates:
        return

    st.markdown("#### Response-shape plot")
    st.caption(
        "Raw batches show scatter; the line shows binned historical means. "
        "Highlighted bands are exploratory historical hypotheses, not optimized setpoints."
    )
    selected_variable = st.selectbox(
        "Numeric driver",
        options=numeric_driver_candidates,
        key=f"response_shape_variable_{selected_outcome}",
    )
    response_band = get_response_band_for_variable(
        operating_window_hints=operating_window_hints,
        variable_name=selected_variable,
    )
    st.plotly_chart(
        response_shape_plot(
            dataframe=merged_dataframe,
            variable_col=selected_variable,
            outcome_col=selected_outcome,
            response_band=response_band,
        ),
        width="stretch",
        key=f"response_shape_{selected_outcome}_{selected_variable}",
    )

    if response_band is not None:
        st.caption(
            f"Broad band: {response_band.get('range_label', 'n/a')}; "
            f"best narrow bin: {response_band.get('refined_range_label', 'not available')}."
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
    st.caption(
        "Capability is reported as Pp and Ppk, which use the overall (long-term) standard "
        "deviation of the batches supplied. They are not Cp and Cpk, which need rational "
        "subgroups and a stable process. One-sided specs get a one-sided Ppk with Pp left "
        "empty; the capability_basis column names which index was used."
    )
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
    process_columns = get_modeling_process_columns(
        profile_result.process_columns,
        validation_order_column,
    )

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

def render_xgboost_shap_tab(
    merged_dataframe,
    profile_result,
    analysis_results,
    outcome_options: list[str],
) -> None:
    """Render optional XGBoost + native TreeSHAP explanations."""
    render_method_explainers(["xgboost_shap"], expanded=False)
    st.caption(
        "This optional model is for explanation and comparison. It does not currently change the main ranked-driver score."
    )

    dependency_status = get_xgboost_dependency_status()
    if not dependency_status["available"]:
        st.info(dependency_status["message"])
        st.code(".\\.venv\\Scripts\\python.exe -m pip install xgboost", language="powershell")
        return

    selected_outcome = st.selectbox(
        "Outcome",
        options=outcome_options,
        key="xgboost_shap_outcome",
    )
    validation_order_column = analysis_results.get("validation_order_column")
    process_columns = get_modeling_process_columns(
        profile_result.process_columns,
        validation_order_column,
    )

    result_key = f"{selected_outcome}::{validation_order_column or 'no_order'}"
    xgboost_results = st.session_state.setdefault("xgboost_shap_results", {})
    run_clicked = st.button(
        "Run XGBoost + SHAP for selected outcome",
        key=f"run_xgboost_shap_{selected_outcome}",
    )

    if run_clicked:
        with st.spinner("Fitting XGBoost and computing native TreeSHAP values..."):
            xgboost_results[result_key] = run_xgboost_shap(
                dataframe=merged_dataframe,
                process_columns=process_columns,
                outcome_column=selected_outcome,
                validation_order_column=validation_order_column,
            )

    xgboost_result = xgboost_results.get(result_key)
    if xgboost_result is None:
        st.info("Run the optional model to compare XGBoost/SHAP against PLS, Random Forest, and CatBoost.")
        return

    if not xgboost_result.get("available"):
        st.warning(xgboost_result.get("reason", "XGBoost + SHAP is not available."))
        return

    metric_columns = st.columns(4)
    metric_columns[0].metric("Rows used", f"{xgboost_result['n_rows_used']:,}")
    metric_columns[1].metric(
        "Training R2",
        format_metric(xgboost_result["training_r2"]),
    )
    metric_columns[2].metric(
        "Numeric features",
        f"{xgboost_result['numeric_feature_count']:,}",
    )
    metric_columns[3].metric(
        "Categorical features",
        f"{xgboost_result['categorical_feature_count']:,}",
    )

    render_time_ordered_validation(
        "Time-ordered XGBoost validation",
        xgboost_result["time_ordered_validation"],
        score_label="Test R2",
    )

    st.markdown("#### XGBoost + SHAP variable importance")
    st.dataframe(
        format_numeric_columns(xgboost_result["variable_importance"].head(20)),
        width="stretch",
        hide_index=True,
    )

    with st.expander("Long SHAP contribution sample", expanded=False):
        st.caption(
            "Encoded SHAP contributions are summed per process variable for each batch. "
            "Positive values push the prediction upward; negative values push it downward. "
            "These are model-behavior explanations, not causal effects."
        )
        st.dataframe(
            format_numeric_columns(xgboost_result["shap_values"].head(200)),
            width="stretch",
            hide_index=True,
        )

OBJECTIVE_CHOICES = {
    "auto": "Detect from the outcome name",
    "maximize": "Higher is better",
    "minimize": "Lower is better",
}

def render_outcome_objective_controls(outcome_options: list[str]) -> None:
    """Let the user state whether each outcome should go up or down.

    The name-based guess cannot always be right - "residual_activity" reads as a
    residual to minimize but is an activity to maximize - and the guess decides
    which end of a response band is called the sweet spot.
    """
    with st.expander("Outcome direction (advanced)", expanded=False):
        st.caption(
            "Response bands and directional lifts are computed toward the better end of each "
            "outcome. The app infers that from the outcome name; override it here when the "
            "name is misleading. Outcomes left ambiguous get no band, which is the safe default."
        )
        objectives = dict(st.session_state.get("outcome_objective_overrides", {}))

        for outcome in outcome_options:
            selected_choice = st.selectbox(
                outcome,
                options=list(OBJECTIVE_CHOICES),
                format_func=lambda choice: OBJECTIVE_CHOICES[choice],
                index=list(OBJECTIVE_CHOICES).index(objectives.get(outcome, "auto")),
                key=f"outcome_objective::{outcome}",
            )
            if selected_choice == "auto":
                objectives.pop(outcome, None)
            else:
                objectives[outcome] = selected_choice

        if objectives != st.session_state.get("outcome_objective_overrides", {}):
            st.session_state["outcome_objective_overrides"] = objectives
            # An already-generated narrative and PDF were built for the previous
            # directions, so they would now contradict the screen.
            discard_generated_interpretation(
                "Outcome direction changed, so the previous interpretation and PDF "
                "were cleared. Generate them again to match the new direction."
            )

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

    render_outcome_objective_controls(outcome_options)

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
            [
                "ranked_drivers",
                "response_bands",
                "interaction_screen",
                "catboost_shap",
                "xgboost_shap",
                "opls",
            ],
            expanded=False,
        )
        selected_outcome = st.selectbox(
            "Outcome",
            options=outcome_options,
            key="ranked_drivers_outcome",
        )
        outcome_drivers = ranked_drivers[ranked_drivers["outcome"] == selected_outcome].head(15)
        outcome_objectives = get_outcome_objective_overrides()
        operating_window_hints = get_analysis_artifact(
            f"operating_window_hints::{selected_outcome}::"
            f"{build_input_fingerprint(objectives=outcome_objectives)}",
            lambda: build_operating_window_hints(
                ranked_drivers=ranked_drivers,
                merged_dataframe=merged_dataframe,
                profile_result=profile_result,
                outcomes=[selected_outcome],
                outcome_objectives=outcome_objectives,
            ),
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
        render_response_shape_section(
            merged_dataframe=merged_dataframe,
            profile_result=profile_result,
            selected_outcome=selected_outcome,
            outcome_drivers=outcome_drivers,
            operating_window_hints=operating_window_hints,
        )

        if not operating_window_hints.empty:
            st.markdown("#### Suggested historical response bands")
            st.caption(
                "Broad bands are more robust quartile summaries. Refined bins are narrower best-observed regions, "
                "but they are more sensitive to noise. The narrow bin is the best of several noisy bin means, so its "
                "lift is optimistically biased: treat a z-score below about 2 as indistinguishable from chance. "
                "Use both as investigation targets, not validated setpoints or spec changes."
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
                "refined_directional_lift_vs_other_bins",
                "refined_directional_lift_z_score",
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
                    "refined_directional_lift_vs_other_bins": "narrow_bin_lift",
                    "refined_directional_lift_z_score": "narrow_bin_lift_z",
                }
            )
            st.dataframe(
                format_numeric_columns(display_hints),
                width="stretch",
                hide_index=True,
            )

    with methods_tab:
        render_method_explainers(
            ["pca", "pls", "random_forest", "catboost_shap", "xgboost_shap"],
            expanded=False,
        )
        (
            pca_method_tab,
            pls_method_tab,
            random_forest_method_tab,
            catboost_shap_method_tab,
            xgboost_shap_method_tab,
        ) = st.tabs(
            ["PCA", "PLS regression", "Random Forest", "CatBoost + SHAP", "XGBoost + SHAP"]
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

        with xgboost_shap_method_tab:
            render_xgboost_shap_tab(
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
