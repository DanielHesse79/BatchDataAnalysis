"""Step 5: PDF export and the local Ollama narrative."""

from __future__ import annotations
from analysis.interpreter import (
    OllamaInterpreterError,
    OllamaInterpreterIncompleteError,
    build_model_options,
    choose_default_model,
    get_configured_ollama_model,
    get_ollama_base_url,
    is_cloud_model,
    sanitize_interpretation_text,
    stream_interpretation,
)
from analysis.narrative_loop import generate_validated_report, replaying_generator
from analysis.report_validator import validate_interpretation_text
from utils.format import datetime_stamp_for_filename
from utils.report import (
    ReportGenerationError,
    generate_analysis_report_pdf,
)

import streamlit as st


from ui.state import (
    collect_stream_chunks,
    get_outcome_objective_overrides,
    get_shared_report_pack,
    list_ollama_models_cached,
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
                outcome_objectives=get_outcome_objective_overrides(),
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

@st.fragment
def render_explanation_controls(
    profile_result,
    audit_result,
    analysis_results,
    merged_dataframe,
    outcomes: list[str],
    spec_assessment=None,
) -> None:
    """Render local Ollama interpretation controls.

    A fragment so that typing an Ollama URL or switching models does not rerun
    the whole app. Generating a report does affect the rest of the page, so that
    path ends with an app-wide rerun.
    """
    st.markdown("#### Local interpretation")
    st.caption(
        "Uses your Ollama server. Local models stay on this machine; cloud-tagged models are routed through Ollama Cloud."
    )

    base_url = st.text_input(
        "Ollama URL",
        value=get_ollama_base_url(),
        key="ollama_base_url",
    )

    allow_cloud_models = st.checkbox(
        "Allow Ollama Cloud models",
        value=False,
        key="allow_cloud_models",
        help=(
            "Cloud-tagged models send the analysis summary, including batch IDs and "
            "categorical values, outside this machine. Leave this off for confidential data."
        ),
    )

    available_models = []
    model_list_error = None
    try:
        available_models = list_ollama_models_cached(base_url)
    except OllamaInterpreterError as error:
        model_list_error = str(error)

    model_options = build_model_options(
        available_models,
        include_cloud_models=allow_cloud_models,
    )

    if model_list_error:
        st.warning(model_list_error)
        selected_model = st.text_input(
            "Ollama model",
            value=get_configured_ollama_model(),
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
        st.warning(
            "Ollama is reachable, but no local models were found. "
            "Pull a model with 'ollama pull llama3.1', or tick 'Allow Ollama Cloud models' above."
        )
        selected_model = st.text_input(
            "Ollama model",
            value=get_configured_ollama_model(),
            key="ollama_model_text_empty",
        )

    if is_cloud_model(selected_model):
        st.warning(
            "Cloud model selected. The app still calls your local Ollama server, but Ollama may send the analysis summary to Ollama Cloud. "
            "Use this only with data you are comfortable sending outside this machine."
        )

    repair_enabled = st.checkbox(
        "Repair the report if the checks fail",
        value=True,
        key="repair_interpretation",
        help=(
            "Re-asks the model to correct what the checks caught, up to twice. "
            "A report that still fails is shown with its warnings, never as clean."
        ),
    )

    generate_clicked = st.button(
        "Generate interpretation",
        type="primary",
        disabled=not selected_model,
        key="generate_ollama_interpretation",
    )

    if generate_clicked:
        report_pack = get_shared_report_pack(
            profile_result=profile_result,
            audit_result=audit_result,
            analysis_results=analysis_results,
            merged_dataframe=merged_dataframe,
            outcomes=outcomes,
            spec_assessment=spec_assessment,
        )
        st.session_state["ollama_interpretation"] = ""
        # Stream into a placeholder so the sanitized text replaces the raw stream.
        # Otherwise the narrative on screen differs from the downloaded and PDF copies.
        stream_placeholder = st.empty()
        streamed_chunks: list[str] = []
        try:
            with st.spinner(f"Generating interpretation with {selected_model}..."):
                with stream_placeholder.container():
                    st.write_stream(
                        collect_stream_chunks(
                            stream_interpretation(
                                profile_result=profile_result,
                                audit_result=audit_result,
                                analysis_results=analysis_results,
                                merged_dataframe=merged_dataframe,
                                outcomes=outcomes,
                                spec_assessment=spec_assessment,
                                model=selected_model,
                                base_url=base_url,
                                report_pack=report_pack,
                            ),
                            streamed_chunks,
                        )
                    )
        except OllamaInterpreterIncompleteError as error:
            stream_placeholder.empty()
            st.warning(
                f"{error} The partial narrative was kept, but it is incomplete - "
                "generate again for a full report."
            )
        except OllamaInterpreterError as error:
            stream_placeholder.empty()
            st.error(str(error))

        interpretation_text = "".join(streamed_chunks)

        if interpretation_text:
            narrative = sanitize_interpretation_text(interpretation_text)
            warnings = validate_interpretation_text(narrative, report_pack).warnings

            # The streamed report is handed to the loop as its first attempt, so
            # a repair costs only the attempts it actually needs.
            if warnings and repair_enabled:
                with st.spinner("Checks failed. Asking the model to correct it..."):
                    loop_result = generate_validated_report(
                        report_pack,
                        model=selected_model,
                        base_url=base_url,
                        generate=replaying_generator(narrative),
                        # Offer the other local models as a last resort. Their
                        # failures are complementary, not merely different.
                        fallback_models=[
                            option for option in model_options
                            if option != selected_model and not is_cloud_model(option)
                        ][:2],
                    )
                narrative = loop_result.text or narrative
                warnings = loop_result.warnings
                st.session_state["interpretation_repair_note"] = loop_result.describe()
            else:
                st.session_state["interpretation_repair_note"] = ""

            st.session_state["ollama_interpretation"] = narrative
            st.session_state["interpretation_validation_warnings"] = warnings
            st.session_state["pdf_report_bytes"] = None
            stream_placeholder.empty()
            # The PDF section outside this fragment reads the stored narrative.
            st.rerun(scope="app")

        stream_placeholder.empty()

    if st.session_state.get("ollama_interpretation"):
        repair_note = st.session_state.get("interpretation_repair_note")
        if repair_note:
            st.caption(repair_note)
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
