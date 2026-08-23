"""SQLite access for the QC Intelligence Layer.

Deliberately thin. The schema lives in schema.sql as plain DDL so the move to
PostgreSQL stays a visible, reviewable change rather than an ORM dialect switch.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import os
import sqlite3
import sys

import pandas as pd


SCRIPT_VERSION = "qc_intel-0.1.0"
PACKAGE_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = PACKAGE_ROOT / "schema.sql"
EXAMPLE_DATABASE_PATH = PACKAGE_ROOT / "data" / "qc_intel.sqlite"


def user_data_directory() -> Path:
    """Return a directory this user can always write to."""
    if sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA") or Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME") or Path.home() / ".local" / "share")
    return base / "BatchInsight" / "qc_intel"


def resolve_database_path() -> Path:
    """Return where this installation keeps its database.

    A bundled application may sit somewhere the user cannot write - Program
    Files, a read-only network share - so the database cannot live beside the
    code. From source it stays in the package, where the build scripts and the
    tests expect it.
    """
    if getattr(sys, "frozen", False):
        return user_data_directory() / "qc_intel.sqlite"
    return EXAMPLE_DATABASE_PATH


DEFAULT_DATABASE_PATH = resolve_database_path()


@dataclass(frozen=True)
class IngestRecord:
    """Identifies the file a set of rows came from."""

    ingest_id: int
    source_file: str
    source_checksum: str


def connect(database_path: Path | str = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced."""
    database_path = Path(database_path)
    database_path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(database_path)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


@contextmanager
def connection_scope(database_path: Path | str = DEFAULT_DATABASE_PATH):
    """Open a connection for one unit of work, then close it.

    SQLite connections belong to the thread that created them. Streamlit runs
    each script run on a different script-runner thread, so a connection held
    across runs raises "SQLite objects created in a thread can only be used in
    that same thread". Opening an existing database file is cheap; holding the
    handle is what causes trouble.
    """
    connection = connect(database_path)
    try:
        yield connection
    finally:
        connection.close()


def create_schema(connection: sqlite3.Connection) -> None:
    """Apply schema.sql. Safe to call repeatedly."""
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    connection.commit()


def reset_database(database_path: Path | str = DEFAULT_DATABASE_PATH) -> sqlite3.Connection:
    """Drop and rebuild the database.

    Only for the synthetic prototype. Real deployments never delete RAW data;
    corrections arrive as new rows with a new ingest_id.
    """
    database_path = Path(database_path)
    if database_path.exists():
        database_path.unlink()
    connection = connect(database_path)
    create_schema(connection)
    return connection


def file_checksum(path: Path | str) -> str:
    """Return a SHA-256 checksum of a source file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(65536), b""):
            digest.update(block)
    return digest.hexdigest()


def bytes_checksum(data: bytes) -> str:
    """Return a SHA-256 checksum of an in-memory source, such as an upload."""
    return hashlib.sha256(data).hexdigest()


@contextmanager
def transaction(connection: sqlite3.Connection):
    """Group writes so a failure leaves no trace, provenance included.

    Registering an ingest and inserting its rows have to succeed or fail
    together. Committed separately, a rejected file still leaves a row in
    ingest_log - which claims the file was loaded and then blocks the corrected
    version as a duplicate.
    """
    try:
        yield connection
    except Exception:
        connection.rollback()
        raise
    else:
        connection.commit()


def register_ingest(
    connection: sqlite3.Connection,
    source_file: Path | str,
    source_format: str,
    adapter: str,
    row_count: int,
    checksum: str | None = None,
    commit: bool = True,
) -> IngestRecord:
    """Record where a batch of rows came from, and refuse silent re-imports.

    An uploaded file has no path to read twice, so its checksum is computed from
    the bytes and passed in. Provenance is identical either way.
    """
    checksum = checksum or file_checksum(source_file)
    existing = connection.execute(
        "SELECT ingest_id, source_file, source_checksum FROM ingest_log "
        "WHERE source_checksum = ? AND adapter = ?",
        (checksum, adapter),
    ).fetchone()
    if existing is not None:
        raise DuplicateSourceError(
            f"{Path(source_file).name} was already imported by {adapter} "
            f"(ingest_id={existing['ingest_id']}). Import it again only under a "
            "new adapter or after an explicit correction entry."
        )

    cursor = connection.execute(
        "INSERT INTO ingest_log (source_file, source_checksum, source_format, adapter, "
        "script_version, ingested_at, row_count) VALUES (?, ?, ?, ?, ?, ?, ?)",
        (
            str(source_file),
            checksum,
            source_format,
            adapter,
            SCRIPT_VERSION,
            datetime.now(timezone.utc).isoformat(timespec="seconds"),
            row_count,
        ),
    )
    if commit:
        connection.commit()
    return IngestRecord(int(cursor.lastrowid), str(source_file), checksum)


class DuplicateSourceError(ValueError):
    """Raised when the same source file is imported twice by the same adapter."""


def insert_frame(
    connection: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    replace_existing: bool = False,
    commit: bool = True,
) -> int:
    """Insert a DataFrame whose columns match the table's columns."""
    if frame.empty:
        return 0

    columns = list(frame.columns)
    placeholders = ", ".join("?" for _ in columns)
    verb = "INSERT OR REPLACE" if replace_existing else "INSERT"
    statement = f"{verb} INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    connection.executemany(statement, frame.astype(object).where(frame.notna(), None).to_numpy().tolist())
    if commit:
        connection.commit()
    return len(frame)


def read_sql(connection: sqlite3.Connection, query: str, parameters: tuple = ()) -> pd.DataFrame:
    """Run a query and return a DataFrame."""
    return pd.read_sql_query(query, connection, params=parameters)


QC_OBSERVATION_QUERY = """
SELECT
    r.run_id,
    r.method_id,
    m.method_name,
    m.method_version,
    m.analyte,
    m.units,
    r.instrument_id,
    r.study_id,
    r.acquisition_timestamp,
    r.acceptance_status,
    r.column_id,
    q.qc_result_id,
    q.qc_level,
    q.evaluation_type,
    q.nominal_value,
    q.measured_value,
    q.percent_bias,
    q.replicate_number,
    q.injection_index,
    q.is_area,
    q.area_ratio,
    q.retention_time,
    q.pass_fail,
    q.acceptance_limit_lower,
    q.acceptance_limit_upper,
    q.ingest_id,
    q.source_row
FROM qc_result q
JOIN analytical_run r ON r.run_id = q.run_id
JOIN method m ON m.method_id = r.method_id
WHERE q.evaluation_type = 'quantitative'
ORDER BY r.acquisition_timestamp, q.injection_index
"""


def parse_timestamps(values: pd.Series) -> pd.Series:
    """Read ISO-8601 timestamps that may or may not carry an offset.

    Exports disagree: a chromatography system usually writes an offset, a LIMS
    export often writes none. Left to infer, pandas takes the format of the
    first row and then fails on the second source - which is what happens as
    soon as one database holds files from both.

    Everything is read as ISO-8601 and expressed in UTC. A value with no offset
    is taken to be UTC rather than shifted, so a laboratory's own runs stay in
    the order it recorded them.
    """
    return pd.to_datetime(values, format="ISO8601", utc=True)


def load_qc_observations(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load the flat QC observation table every statistic is built from."""
    frame = read_sql(connection, QC_OBSERVATION_QUERY)
    if not frame.empty:
        frame["acquisition_timestamp"] = parse_timestamps(frame["acquisition_timestamp"])
    return frame


def load_events(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load laboratory events."""
    frame = read_sql(connection, "SELECT * FROM lab_event ORDER BY event_timestamp")
    if not frame.empty:
        frame["event_timestamp"] = parse_timestamps(frame["event_timestamp"])
    return frame


def load_calibrations(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load calibration summaries joined to their run."""
    frame = read_sql(
        connection,
        """
        SELECT c.*, r.method_id, r.instrument_id, r.acquisition_timestamp
        FROM calibration c
        JOIN analytical_run r ON r.run_id = c.run_id
        ORDER BY r.acquisition_timestamp
        """,
    )
    if not frame.empty:
        frame["acquisition_timestamp"] = parse_timestamps(frame["acquisition_timestamp"])
    return frame
