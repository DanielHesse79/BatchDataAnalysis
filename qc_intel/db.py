"""SQLite access for the QC Intelligence Layer.

Deliberately thin. The schema lives in schema.sql as plain DDL so the move to
PostgreSQL stays a visible, reviewable change rather than an ORM dialect switch.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
import hashlib
import sqlite3

import pandas as pd


SCRIPT_VERSION = "qc_intel-0.1.0"
PACKAGE_ROOT = Path(__file__).resolve().parent
SCHEMA_PATH = PACKAGE_ROOT / "schema.sql"
DEFAULT_DATABASE_PATH = PACKAGE_ROOT / "data" / "qc_intel.sqlite"


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


def register_ingest(
    connection: sqlite3.Connection,
    source_file: Path | str,
    source_format: str,
    adapter: str,
    row_count: int,
) -> IngestRecord:
    """Record where a batch of rows came from, and refuse silent re-imports."""
    checksum = file_checksum(source_file)
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
    connection.commit()
    return IngestRecord(int(cursor.lastrowid), str(source_file), checksum)


class DuplicateSourceError(ValueError):
    """Raised when the same source file is imported twice by the same adapter."""


def insert_frame(
    connection: sqlite3.Connection,
    table: str,
    frame: pd.DataFrame,
    replace_existing: bool = False,
) -> int:
    """Insert a DataFrame whose columns match the table's columns."""
    if frame.empty:
        return 0

    columns = list(frame.columns)
    placeholders = ", ".join("?" for _ in columns)
    verb = "INSERT OR REPLACE" if replace_existing else "INSERT"
    statement = f"{verb} INTO {table} ({', '.join(columns)}) VALUES ({placeholders})"
    connection.executemany(statement, frame.astype(object).where(frame.notna(), None).to_numpy().tolist())
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


def load_qc_observations(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load the flat QC observation table every statistic is built from."""
    frame = read_sql(connection, QC_OBSERVATION_QUERY)
    if not frame.empty:
        frame["acquisition_timestamp"] = pd.to_datetime(frame["acquisition_timestamp"])
    return frame


def load_events(connection: sqlite3.Connection) -> pd.DataFrame:
    """Load laboratory events."""
    frame = read_sql(connection, "SELECT * FROM lab_event ORDER BY event_timestamp")
    if not frame.empty:
        frame["event_timestamp"] = pd.to_datetime(frame["event_timestamp"])
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
        frame["acquisition_timestamp"] = pd.to_datetime(frame["acquisition_timestamp"])
    return frame
