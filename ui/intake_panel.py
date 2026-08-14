"""Step 1-2: templates, file intake, readiness, mapping profiles, and specs."""

from __future__ import annotations
from html import escape
from typing import Any
from analysis.aggregation import (
    DUPLICATE_STRATEGIES,
    LONG_FORMAT_AGGREGATIONS,
    aggregate_duplicate_batch_rows,
    count_duplicate_batch_ids,
    detect_long_format_candidates,
    get_duplicate_batch_id_examples,
    pivot_long_to_wide,
)
from analysis.data_prep import (
    DataPrepError,
    get_intake_warnings,
)
from analysis.mapping import (
    MappingProfileError,
    build_mapping_profile,
    load_mapping_profile,
    mapping_profile_to_json,
    strip_non_json_metadata,
)
from analysis.specs import build_default_spec_template

import pandas as pd
import streamlit as st


from ui.state import (
    build_template_bundle_cached,
    first_or_none,
    inspect_uploaded_file_cached,
    load_and_normalize_uploaded_file_cached,
    option_default_index,
    uploaded_file_identity,
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

        template_bundle = build_template_bundle_cached()
        process_template = template_bundle["process"]
        qc_wide_template = template_bundle["qc_wide"]
        qc_long_template = template_bundle["qc_long"]
        combined_workbook_bytes = template_bundle["workbook_bytes"]

        template_columns = st.columns(4)
        template_columns[0].download_button(
            "Process CSV template",
            data=template_bundle["process_csv"],
            file_name="batch_insight_process_template.csv",
            mime="text/csv",
            key="download_process_template",
        )
        template_columns[1].download_button(
            "QC wide CSV template",
            data=template_bundle["qc_wide_csv"],
            file_name="batch_insight_qc_wide_template.csv",
            mime="text/csv",
            key="download_qc_wide_template",
        )
        template_columns[2].download_button(
            "QC long CSV template",
            data=template_bundle["qc_long_csv"],
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

def render_file_intake(uploaded_file, label: str, key_prefix: str):
    """Inspect, load, normalize, and optionally pivot one uploaded file."""
    if uploaded_file is None:
        return None, {}

    file_identity = uploaded_file_identity(uploaded_file)

    try:
        inspection = inspect_uploaded_file_cached(uploaded_file, file_identity)
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
            dataframe, numeric_parse_report, normalization_warnings = (
                load_and_normalize_uploaded_file_cached(
                    uploaded_file,
                    file_identity,
                    sheet_name,
                    int(header_row),
                    parse_numeric_like_columns,
                )
            )
        except DataPrepError as error:
            st.error(f"{label}: {error}")
            return None, metadata

        metadata.update(
            {
                "sheet_name": sheet_name,
                "header_row": int(header_row),
                "parse_numeric_like_columns": parse_numeric_like_columns,
                "numeric_parse_report": numeric_parse_report,
                "file_identity": file_identity,
            }
        )

        for warning in [*get_intake_warnings(dataframe), *normalization_warnings]:
            st.warning(warning)

        if not numeric_parse_report.empty:
            with st.expander("Numeric parsing log", expanded=False):
                st.dataframe(
                    numeric_parse_report,
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

def show_dataframe_preview(label: str, dataframe) -> None:
    """Display a compact preview without overwhelming the user."""
    with st.expander(f"Preview {label}", expanded=False):
        st.dataframe(dataframe.head(10), width="stretch")

def render_duplicate_strategy(label: str, dataframe, batch_id_column: str, key: str) -> str:
    """Let the user choose how to handle duplicate batch IDs."""
    try:
        duplicate_count = count_duplicate_batch_ids(dataframe, batch_id_column)
    except (DataPrepError, KeyError, ValueError):
        return "error"

    if duplicate_count == 0:
        return "error"

    st.warning(
        f"{label} has {duplicate_count:,} duplicate normalized batch ID(s). "
        "Choose how to resolve them before analysis."
    )
    selected_strategy = st.selectbox(
        f"Resolve duplicate {label.lower()} batch IDs",
        options=list(DUPLICATE_STRATEGIES.keys()),
        format_func=lambda strategy: DUPLICATE_STRATEGIES[strategy],
        index=0,
        key=key,
        help=(
            "Use Stop if the source file should be corrected. Use average/median when duplicate "
            "rows are technical replicate measurements. Use first/last only when row order has "
            "business meaning, such as revised exports."
        ),
    )

    duplicate_examples = get_duplicate_batch_id_examples(dataframe, batch_id_column)
    if not duplicate_examples.empty:
        with st.expander(f"View duplicate {label.lower()} batch IDs", expanded=False):
            st.dataframe(duplicate_examples, width="stretch", hide_index=True)

    return selected_strategy

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
        render_readiness_resolution_help(readiness_result)
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
            cards.append(build_readiness_issue_card_html(severity, str(message)))

    if not cards:
        return

    st.markdown(
        '<div class="bia-help-grid">' + "\n".join(cards[:6]) + "</div>",
        unsafe_allow_html=True,
    )

def build_readiness_issue_card_html(severity: str, message: str) -> str:
    """Build compact HTML for one readiness issue card without Markdown code indentation."""
    safe_severity = escape(severity)
    safe_title = escape(severity.title())
    safe_message = escape(message)
    return (
        f'<div class="bia-issue-card {safe_severity}">'
        f"<strong>{safe_title}</strong>"
        f"<p>{safe_message}</p>"
        "</div>"
    )

def render_readiness_resolution_help(readiness_result) -> None:
    """Show concrete next actions for common readiness blockers."""
    process_duplicates = readiness_result.details.get("process_duplicate_batch_ids", 0)
    qc_duplicates = readiness_result.details.get("qc_duplicate_batch_ids", 0)
    duplicate_messages = [
        message for message in readiness_result.blockers if "duplicate batch ID" in message
    ]
    if not duplicate_messages:
        return

    duplicate_sources = []
    if process_duplicates:
        duplicate_sources.append("Process data")
    if qc_duplicates:
        duplicate_sources.append("QC data")
    source_label = " and ".join(duplicate_sources) if duplicate_sources else "The uploaded data"

    st.info(
        f"{source_label} contains repeated normalized batch IDs. Use the duplicate handling "
        "dropdown above this Data readiness section, then the blocker will change into a warning."
    )
    st.markdown(
        """
        **Which rule should I choose?**

        - **Average numeric replicate rows**: best when duplicate rows are replicate measurements for the same batch.
        - **Median numeric replicate rows**: good when one replicate may be noisy or extreme.
        - **Keep first/last row per batch**: only use when the export order means original vs revised record.
        - **Stop and let me fix duplicates**: safest when duplicates are accidental or need manual review in the source file.
        """
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

def render_spec_input():
    """Render optional spec/window controls, by upload or by typing limits."""
    st.markdown(
        '<div class="bia-section-kicker">Optional - Specs & operating windows</div>',
        unsafe_allow_html=True,
    )
    st.subheader("Specs & operating windows")
    st.caption(
        "Add process windows and QC specs to compare official limits against historical behavior. "
        "You can also continue without specs."
    )

    upload_tab, type_tab = st.tabs(["Upload a spec file", "Type limits here"])

    with upload_tab:
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
                data=build_default_spec_template().to_csv(index=False),
                file_name="batch_insight_spec_template.csv",
                mime="text/csv",
                key="download_spec_template",
            )
            st.caption("Tip: the repository also includes `data/synthetic_specs.csv`.")

    with type_tab:
        typed_specs = render_spec_editor()

    if spec_file is not None:
        return spec_file

    return typed_specs

def render_spec_editor():
    """Let the user type spec limits instead of round-tripping through Excel."""
    st.caption(
        "Edit the rows below, or add your own. Leave a limit empty for a one-sided spec. "
        "An uploaded file always takes priority over this table."
    )

    edited_specs = st.data_editor(
        build_spec_editor_seed(),
        num_rows="dynamic",
        width="stretch",
        hide_index=True,
        key="spec_editor",
        column_config={
            "variable": st.column_config.TextColumn(
                "variable",
                help="Must match a column name in the merged data.",
            ),
            "role": st.column_config.SelectboxColumn(
                "role",
                options=["process", "qc"],
                help="process = operating window, qc = release spec.",
            ),
            "target": st.column_config.NumberColumn("target"),
            "lower_limit": st.column_config.NumberColumn("lower_limit"),
            "upper_limit": st.column_config.NumberColumn("upper_limit"),
        },
    )

    usable_specs = edited_specs[
        edited_specs["variable"].astype("string").str.strip().fillna("") != ""
    ]
    if usable_specs.empty:
        return None

    st.success(f"{len(usable_specs):,} typed spec row(s) will be used.")
    return usable_specs.reset_index(drop=True)

@st.cache_data(show_spinner=False)
def build_spec_editor_seed() -> pd.DataFrame:
    """Return an empty spec table with the right columns and dtypes."""
    return build_default_spec_template().iloc[0:0].copy()

def render_mapping_profile_import(process_columns: list[str], qc_columns: list[str]) -> None:
    """Load a saved mapping profile so a repeat export can be set up in one step."""
    profile_file = st.file_uploader(
        "Load a saved mapping profile",
        type=["json"],
        key="mapping_profile_file",
        help="Applies the batch ID columns, outcomes, duplicate rules, and intake settings saved from a previous run.",
    )
    if profile_file is None:
        return

    try:
        loaded_profile = load_mapping_profile(
            profile_file.getvalue().decode("utf-8"),
            process_columns=process_columns,
            qc_columns=qc_columns,
        )
    except (MappingProfileError, UnicodeDecodeError) as error:
        st.error(f"Mapping profile: {error}")
        return

    for warning in loaded_profile.warnings:
        st.warning(warning)

    applied_profile_id = uploaded_file_identity(profile_file)
    if st.session_state.get("applied_mapping_profile_id") != applied_profile_id:
        st.session_state["applied_mapping_profile_id"] = applied_profile_id
        st.session_state["loaded_mapping_profile"] = loaded_profile
        apply_mapping_profile_to_widgets(loaded_profile)
        # The intake controls above have already rendered this run, so a rerun is
        # what actually makes the restored sheet and header row take effect.
        st.rerun()

    st.session_state["loaded_mapping_profile"] = loaded_profile
    st.success(
        f"Profile applied from {loaded_profile.process_file_name} / {loaded_profile.qc_file_name}."
    )


def build_mapping_profile_widget_values(loaded_profile) -> dict[str, Any]:
    """Map a loaded profile onto the widget keys the intake controls read.

    Storing the parsed profile alone was not enough: the duplicate-strategy
    selectors and the file-intake controls are keyed widgets, so they keep
    whatever is in session state and ignore a value passed as a default.
    """
    widget_values: dict[str, Any] = {
        "process_batch_id_column": loaded_profile.process_batch_id_column,
        "qc_batch_id_column": loaded_profile.qc_batch_id_column,
        "selected_outcome_columns_widget": list(loaded_profile.outcome_columns),
        "process_duplicate_strategy": loaded_profile.process_duplicate_strategy,
        "qc_duplicate_strategy": loaded_profile.qc_duplicate_strategy,
    }

    for key_prefix, intake_options in [
        ("process_intake", loaded_profile.process_intake or {}),
        ("qc_intake", loaded_profile.qc_intake or {}),
    ]:
        sheet_name = intake_options.get("sheet_name")
        if sheet_name:
            widget_values[f"{key_prefix}_sheet"] = sheet_name

        header_row = intake_options.get("header_row")
        if header_row is not None:
            widget_values[f"{key_prefix}_header_row"] = int(header_row)

        parse_numeric = intake_options.get("parse_numeric_like_columns")
        if parse_numeric is not None:
            widget_values[f"{key_prefix}_parse_numeric"] = bool(parse_numeric)

    return {key: value for key, value in widget_values.items() if value is not None}


def apply_mapping_profile_to_widgets(loaded_profile) -> None:
    """Write a loaded profile into the widget state the controls read."""
    for widget_key, value in build_mapping_profile_widget_values(loaded_profile).items():
        st.session_state[widget_key] = value
