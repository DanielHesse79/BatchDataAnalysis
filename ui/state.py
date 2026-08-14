"""Session-scoped caching, fingerprints, and derived-artifact memos.

Streamlit reruns the whole script on every interaction, so anything expensive
has to be keyed and stored rather than recomputed."""

from __future__ import annotations
from datetime import datetime
from typing import (
    Any,
    Callable,
)
from analysis.data_prep import load_tabular_file
from analysis.evidence import build_report_pack
from analysis.intake import (
    IntakeLoadOptions,
    inspect_tabular_file,
    load_intake_dataframe,
)
from analysis.interpreter import get_available_ollama_models
from analysis.normalization import normalize_dataframe_values
from analysis.templates import build_template_bundle

import hashlib
import json
import pandas as pd
import streamlit as st



def uploaded_file_identity(uploaded_file) -> str:
    """Return a stable identity for an uploaded file, for use in cache keys."""
    if uploaded_file is None:
        return "none"

    file_id = getattr(uploaded_file, "file_id", None)
    if file_id:
        return str(file_id)

    return hashlib.sha256(uploaded_file.getvalue()).hexdigest()

@st.cache_data(show_spinner=False, max_entries=8)
def inspect_uploaded_file_cached(_uploaded_file, cache_key: str):
    """Inspect an uploaded file once per upload instead of once per rerun."""
    return inspect_tabular_file(_uploaded_file)

@st.cache_data(show_spinner=False, max_entries=16)
def load_and_normalize_uploaded_file_cached(
    _uploaded_file,
    cache_key: str,
    sheet_name: str | None,
    header_row: int,
    parse_numeric_like_columns: bool,
) -> tuple[pd.DataFrame, pd.DataFrame, list[str]]:
    """Load and normalize an uploaded file, keyed on the file and its options.

    Streamlit reruns the whole script on every widget interaction, so without
    this the app re-parsed and re-normalized both workbooks on every click.
    """
    dataframe = load_intake_dataframe(
        _uploaded_file,
        IntakeLoadOptions(sheet_name=sheet_name, header_row=int(header_row)),
    )
    normalization_result = normalize_dataframe_values(
        dataframe,
        parse_numeric_like_columns=parse_numeric_like_columns,
    )
    return (
        normalization_result.dataframe,
        normalization_result.numeric_parse_report,
        normalization_result.warnings,
    )

@st.cache_data(show_spinner=False, ttl=60)
def list_ollama_models_cached(base_url: str) -> list[str]:
    """List Ollama models at most once a minute per URL.

    Without the cache this blocking HTTP call fired on every rerun of the
    results view, stalling the whole app whenever Ollama was slow or absent.
    """
    return get_available_ollama_models(base_url)

@st.cache_data(show_spinner=False)
def build_template_bundle_cached() -> dict[str, Any]:
    """Build the download templates once per session."""
    return build_template_bundle()

def get_analysis_artifact(name: str, builder: Callable[[], Any]) -> Any:
    """Memoize a derived artifact for the current analysis run.

    The evidence pack and operating-window hints are deterministic functions of
    the stored analysis results, but they are expensive (quartile binning and an
    interaction screen). Keying on the run token avoids hashing large DataFrames.
    """
    run_token = st.session_state.get("analysis_run_token", "")
    artifact_cache = st.session_state.setdefault("analysis_artifact_cache", {})

    if artifact_cache.get("__run_token__") != run_token:
        artifact_cache.clear()
        artifact_cache["__run_token__"] = run_token

    if name not in artifact_cache:
        artifact_cache[name] = builder()

    return artifact_cache[name]

def collect_stream_chunks(chunks, collected: list[str]):
    """Tee a stream into a list so partial output survives an interruption."""
    for chunk in chunks:
        collected.append(chunk)
        yield chunk

def get_shared_report_pack(
    profile_result,
    audit_result,
    analysis_results,
    merged_dataframe,
    outcomes: list[str],
    spec_assessment,
) -> dict[str, Any]:
    """Build the deterministic evidence pack once per analysis run.

    Key findings, the operating-window hints, and the Ollama prompt all consume
    the same pack; building it three times per rerun was pure waste.
    """
    outcome_objectives = get_outcome_objective_overrides()
    return get_analysis_artifact(
        f"report_pack::{build_input_fingerprint(objectives=outcome_objectives)}",
        lambda: build_report_pack(
            profile_result=profile_result,
            audit_result=audit_result,
            analysis_results=analysis_results,
            merged_dataframe=merged_dataframe,
            outcomes=outcomes,
            spec_assessment=spec_assessment,
            outcome_objectives=outcome_objectives,
        ),
    )

def bump_analysis_run_token() -> None:
    """Invalidate derived artifacts after results or specs change."""
    st.session_state["analysis_run_token"] = datetime.now().isoformat(timespec="microseconds")
    st.session_state.pop("analysis_artifact_cache", None)

def build_input_fingerprint(**inputs: Any) -> str:
    """Hash every input that feeds the merge, so staleness is detectable."""
    serialized = json.dumps(inputs, sort_keys=True, default=str)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

def load_spec_input(spec_input) -> pd.DataFrame | None:
    """Turn the spec control's output into a DataFrame.

    The control returns an uploaded file, a table the user typed in, or nothing.
    """
    if spec_input is None:
        return None
    if isinstance(spec_input, pd.DataFrame):
        return spec_input

    return load_tabular_file(spec_input)

def spec_input_fingerprint(spec_input) -> str:
    """Identify the current spec input for staleness checks."""
    if spec_input is None:
        return "none"
    if isinstance(spec_input, pd.DataFrame):
        return hashlib.sha256(
            pd.util.hash_pandas_object(spec_input, index=True).to_numpy().tobytes()
        ).hexdigest()

    return uploaded_file_identity(spec_input)

def intake_fingerprint_fields(intake_metadata: dict[str, Any]) -> dict[str, Any]:
    """Pick the intake settings that change the prepared table."""
    return {
        "file_identity": intake_metadata.get("file_identity"),
        "sheet_name": intake_metadata.get("sheet_name"),
        "header_row": intake_metadata.get("header_row"),
        "parse_numeric_like_columns": intake_metadata.get("parse_numeric_like_columns"),
        "long_format_pivot": intake_metadata.get("long_format_pivot"),
    }

def resolve_default_index(options: list, preferred_value, fallback_index: int) -> int:
    """Return the index of a preferred option, falling back when it is absent."""
    if preferred_value in options:
        return options.index(preferred_value)
    return fallback_index

def option_default_index(options, preferred_value) -> int:
    """Return a safe selectbox default index."""
    if preferred_value in options:
        return list(options).index(preferred_value)
    return 0

def first_or_none(values):
    """Return the first item or None."""
    return values[0] if values else None

def discard_generated_interpretation(reason: str) -> None:
    """Clear the LLM narrative and PDF bytes after their inputs changed.

    Both are snapshots. Leaving them downloadable after a setting change hands
    the user a report that disagrees with what the screen now says.
    """
    had_output = bool(
        st.session_state.get("ollama_interpretation")
        or st.session_state.get("pdf_report_bytes")
    )
    st.session_state["ollama_interpretation"] = ""
    st.session_state["interpretation_validation_warnings"] = []
    st.session_state["pdf_report_bytes"] = None

    if had_output:
        st.warning(reason)


def get_outcome_objective_overrides() -> dict[str, str] | None:
    """Return explicit outcome directions, or None when all are automatic."""
    return st.session_state.get("outcome_objective_overrides") or None
