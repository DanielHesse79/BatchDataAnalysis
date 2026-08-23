"""Load data tab for the QC Intelligence Layer.

View layer only. Every decision about what a column means, what type it is and
whether it looks like personal data is made in qc_intel.ingest.interactive; this
module asks the questions and shows the answers.

The order is deliberate. A file is inspected, screened for personal data, then
mapped, then written - so an analyst sees what would enter the database before
anything does.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from analysis.intake import IntakeLoadOptions, inspect_tabular_file, load_intake_dataframe
from qc_intel import db
from qc_intel.ingest.generic_tabular import IngestError, REQUIRED_COLUMNS, import_frame
from qc_intel.ingest.interactive import (
    TARGET_TABLES,
    apply_mapping,
    missing_required_columns,
    screen_for_personal_data,
    suggest_column_mapping,
    suggest_target_table,
)


PREVIEW_ROWS = 8
LOAD_ORDER_NOTE = (
    "Load reference data before the rows that point at it: methods and "
    "instruments, then runs, then QC results."
)


def render_intake(database_path: str, example_marker: Path | None) -> None:
    """Render the whole upload flow. Reruns the app after a successful load."""
    st.subheader("Load laboratory data")
    st.caption(
        "Files are read, screened and mapped here. Only the columns you map are "
        "written; everything else is discarded rather than stored."
    )

    uploaded_file = st.file_uploader(
        "QC export (.csv, .xlsx, .xls)",
        type=["csv", "xlsx", "xls"],
        key="qc_intake_file",
    )
    if uploaded_file is None:
        st.info(LOAD_ORDER_NOTE)
        _render_ingest_history(database_path)
        return

    try:
        inspection = inspect_tabular_file(uploaded_file)
    except Exception as error:  # noqa: BLE001 - surfaced to the analyst
        st.error(f"Could not read that file: {error}")
        return

    for warning in inspection.warnings:
        st.warning(warning)

    sheet_name, header_row = _render_read_options(inspection)

    try:
        frame = load_intake_dataframe(
            uploaded_file, IntakeLoadOptions(sheet_name=sheet_name, header_row=header_row)
        )
    except Exception as error:  # noqa: BLE001 - surfaced to the analyst
        st.error(f"Could not load that file: {error}")
        return

    st.write(f"**{len(frame):,} rows, {len(frame.columns)} columns**")
    st.dataframe(frame.head(PREVIEW_ROWS), width="stretch")

    blocked_columns = _render_personal_data_screen(frame)
    usable = frame.drop(columns=list(blocked_columns), errors="ignore")
    if usable.empty or not len(usable.columns):
        st.error("Every column was withheld. There is nothing left to load.")
        return

    table = _render_table_choice(usable.columns)
    mapping = _render_mapping_editor(usable.columns, table)

    missing = missing_required_columns(mapping, table)
    if missing:
        st.error(
            f"Map the required column(s) before loading: {', '.join(missing)}."
        )
        return

    canonical, report = apply_mapping(usable, mapping, table)
    _render_mapping_report(canonical, report)

    if st.button("Load into the database", type="primary"):
        _perform_load(
            database_path=database_path,
            frame=canonical,
            table=table,
            uploaded_file=uploaded_file,
            example_marker=example_marker,
        )

    _render_ingest_history(database_path)


def _render_read_options(inspection) -> tuple[str | None, int]:
    """Sheet and header row, pre-filled with what inspection suggested."""
    columns = st.columns(2)

    sheet_name = None
    if inspection.sheets:
        with columns[0]:
            sheet_name = st.selectbox(
                "Sheet", inspection.sheets,
                index=inspection.sheets.index(inspection.suggested_sheet)
                if inspection.suggested_sheet in inspection.sheets else 0,
            )

    with columns[1 if inspection.sheets else 0]:
        header_row = st.number_input(
            "Header row (0 is the first row)",
            min_value=0, max_value=50,
            value=int(inspection.suggested_header_row),
            help="The row holding column names. Exports often start with a title block.",
        )

    return sheet_name, int(header_row)


def _render_personal_data_screen(frame: pd.DataFrame) -> set[str]:
    """Show what looks identifiable and return the columns being withheld."""
    findings = screen_for_personal_data(frame)
    if not findings:
        st.success("No columns look like personal data.")
        return set()

    st.error(
        f"**{len(findings)} column(s) look like personal data.** They are "
        "withheld from the database. This tool trends instrument and method "
        "behaviour; it does not need anything that identifies a person."
    )
    st.dataframe(
        pd.DataFrame(
            [{"Column": f.column, "Why": f.reason} for f in findings]
        ),
        width="stretch", hide_index=True,
    )

    blocked = {finding.column for finding in findings}
    released = st.multiselect(
        "Override: load one of these anyway",
        sorted(blocked),
        default=[],
        help=(
            "Only for a column the screen misread - a pseudonymous analyst code "
            "read as a name, for example. Never for identifiable data."
        ),
    )
    if released:
        st.warning(
            f"Overriding the screen for: {', '.join(released)}. Confirm these "
            "carry no identifiable data before loading."
        )

    return blocked - set(released)


def _render_table_choice(source_columns) -> str:
    """Which canonical table this file populates."""
    suggested = suggest_target_table(source_columns)
    labels = {table: label for table, label, _ in TARGET_TABLES}
    options = [table for table, _, _ in TARGET_TABLES]

    index = options.index(suggested) if suggested in options else 0
    table = st.selectbox(
        "What does this file contain?",
        options, index=index,
        format_func=lambda value: labels[value],
    )

    description = next(text for name, _, text in TARGET_TABLES if name == table)
    if suggested == table:
        st.caption(f"{description} Detected from the column names.")
    else:
        st.caption(description)

    return table


def _render_mapping_editor(source_columns, table: str) -> dict[str, str | None]:
    """Let the analyst correct the proposed column mapping."""
    suggestion = suggest_column_mapping(source_columns, table)
    required = set(REQUIRED_COLUMNS.get(table, []))
    unmapped = "(not in this file)"
    choices = [unmapped] + [str(column) for column in source_columns]

    editable = pd.DataFrame([
        {
            "Canonical column": canonical,
            "Required": canonical in required,
            "Source column": source if source else unmapped,
        }
        for canonical, source in suggestion.items()
    ])

    st.caption(
        "Check the mapping. Unmapped canonical columns are stored empty; "
        "unmapped source columns are discarded."
    )
    edited = st.data_editor(
        editable,
        width="stretch", hide_index=True, key=f"qc_mapping_{table}",
        column_config={
            "Canonical column": st.column_config.TextColumn(disabled=True),
            "Required": st.column_config.CheckboxColumn(disabled=True),
            "Source column": st.column_config.SelectboxColumn(options=choices, required=True),
        },
    )

    return {
        row["Canonical column"]: (
            None if row["Source column"] == unmapped else row["Source column"]
        )
        for _, row in edited.iterrows()
    }


def _render_mapping_report(canonical: pd.DataFrame, report) -> None:
    """Show exactly what would be written."""
    st.markdown("**What will be written**")
    st.dataframe(canonical.head(PREVIEW_ROWS), width="stretch")

    if report.dropped_columns:
        st.caption(
            f"Discarded, not stored: {', '.join(report.dropped_columns[:12])}"
            + (" ..." if len(report.dropped_columns) > 12 else "")
        )

    for column, count in report.unparsed_numbers.items():
        st.warning(
            f"{count} value(s) in '{column}' could not be read as a number and "
            "will be stored empty."
        )
    for column, count in report.unparsed_timestamps.items():
        st.warning(
            f"{count} value(s) in '{column}' could not be read as a date and "
            "will be stored empty."
        )


def _perform_load(
    database_path: str,
    frame: pd.DataFrame,
    table: str,
    uploaded_file,
    example_marker: Path | None,
) -> None:
    """Write the rows, then drop the example data and rebuild."""
    uploaded_file.seek(0)
    payload = uploaded_file.read()

    try:
        with db.connection_scope(database_path) as connection:
            written = import_frame(
                connection,
                frame,
                table,
                source_name=uploaded_file.name,
                source_format=Path(uploaded_file.name).suffix.lstrip(".").lower(),
                checksum=db.bytes_checksum(payload),
            )
    except db.DuplicateSourceError as error:
        st.error(str(error))
        return
    except IngestError as error:
        st.error(str(error))
        return
    except Exception as error:  # noqa: BLE001 - surfaced to the analyst
        st.error(f"The load failed and nothing was written: {error}")
        return

    # Real data has arrived, so the dashboard should stop calling itself a demo.
    if example_marker is not None and example_marker.exists():
        example_marker.unlink()

    st.cache_data.clear()
    st.success(f"Loaded {written:,} rows into {table}.")
    st.rerun()


def _render_ingest_history(database_path: str) -> None:
    """Every file this database has accepted, newest first."""
    with st.expander("Files already loaded"):
        try:
            with db.connection_scope(database_path) as connection:
                history = db.read_sql(
                    connection,
                    "SELECT ingested_at, source_file, adapter, row_count, source_checksum "
                    "FROM ingest_log ORDER BY ingest_id DESC",
                )
        except Exception as error:  # noqa: BLE001 - surfaced to the analyst
            st.warning(f"Could not read the ingest log: {error}")
            return

        if history.empty:
            st.caption("Nothing has been loaded yet.")
            return

        history["source_file"] = history["source_file"].map(lambda value: Path(value).name)
        history["source_checksum"] = history["source_checksum"].str.slice(0, 12)
        st.dataframe(history, width="stretch", hide_index=True)
