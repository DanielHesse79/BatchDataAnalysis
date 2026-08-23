-- QC Intelligence Layer -- canonical schema.
--
-- Written as plain DDL rather than through an ORM so the migration path to
-- PostgreSQL is visible: only the AUTOINCREMENT/TEXT-timestamp choices below
-- need changing. Timestamps are ISO-8601 strings in UTC.
--
-- Layering: raw source rows are never edited. ingest_log records the file a row
-- came from; every derived table is rebuilt from these tables and can be dropped
-- and recomputed without data loss.

PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------- provenance
CREATE TABLE IF NOT EXISTS ingest_log (
    ingest_id         INTEGER PRIMARY KEY AUTOINCREMENT,
    source_file       TEXT    NOT NULL,
    source_checksum   TEXT    NOT NULL,
    source_format     TEXT    NOT NULL,
    adapter           TEXT    NOT NULL,
    script_version    TEXT    NOT NULL,
    ingested_at       TEXT    NOT NULL,
    row_count         INTEGER NOT NULL,
    UNIQUE (source_checksum, adapter)
);

-- ---------------------------------------------------------------- reference
CREATE TABLE IF NOT EXISTS method (
    method_id         TEXT PRIMARY KEY,
    method_name       TEXT NOT NULL,
    method_version    TEXT NOT NULL,
    analyte           TEXT NOT NULL,
    matrix            TEXT,
    platform          TEXT,
    units             TEXT,
    lloq              REAL,
    uloq              REAL,
    effective_date    TEXT,
    retired_date      TEXT
);

CREATE TABLE IF NOT EXISTS instrument (
    instrument_id     TEXT PRIMARY KEY,
    instrument_model  TEXT,
    asset_tag         TEXT,
    installation_date TEXT,
    software_version  TEXT
);

CREATE TABLE IF NOT EXISTS material_lot (
    material_id       TEXT PRIMARY KEY,
    material_type     TEXT NOT NULL,   -- reference_standard | reagent | column | matrix
    manufacturer      TEXT,
    lot_number        TEXT,
    date_opened       TEXT,
    expiry_date       TEXT,
    concentration     REAL
);

-- ---------------------------------------------------------------- runs
CREATE TABLE IF NOT EXISTS analytical_run (
    run_id                    TEXT PRIMARY KEY,
    method_id                 TEXT NOT NULL REFERENCES method(method_id),
    instrument_id             TEXT NOT NULL REFERENCES instrument(instrument_id),
    study_id                  TEXT,
    -- Acquisition time, not processing or approval time. Mixing these is a
    -- silent source of wrong trend lines.
    acquisition_timestamp     TEXT NOT NULL,
    run_type                  TEXT,
    -- Ingested from the CDS/LIMS verdict. Never recomputed here.
    acceptance_status         TEXT,
    acceptance_reason         TEXT,
    column_id                 TEXT REFERENCES material_lot(material_id),
    column_injection_count    INTEGER,
    processing_method_version TEXT,
    analyst_ref               TEXT,    -- pseudonymous; disabled by default in the UI
    ingest_id                 INTEGER REFERENCES ingest_log(ingest_id),
    source_row                TEXT
);

CREATE TABLE IF NOT EXISTS run_material (
    run_id            TEXT NOT NULL REFERENCES analytical_run(run_id),
    material_id       TEXT NOT NULL REFERENCES material_lot(material_id),
    role              TEXT,
    PRIMARY KEY (run_id, material_id, role)
);

-- ---------------------------------------------------------------- results
CREATE TABLE IF NOT EXISTS qc_result (
    qc_result_id      INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id            TEXT NOT NULL REFERENCES analytical_run(run_id),
    qc_level          TEXT NOT NULL,
    evaluation_type   TEXT NOT NULL DEFAULT 'quantitative', -- or 'interference'
    nominal_value     REAL,
    measured_value    REAL,
    units             TEXT,
    percent_bias      REAL,
    replicate_number  INTEGER,
    injection_index   INTEGER,
    analyte_area      REAL,
    is_area           REAL,
    area_ratio        REAL,
    retention_time    REAL,
    manual_integration INTEGER DEFAULT 0,
    pass_fail         TEXT,
    acceptance_limit_lower REAL,
    acceptance_limit_upper REAL,
    ingest_id         INTEGER REFERENCES ingest_log(ingest_id),
    source_row        TEXT
);

CREATE TABLE IF NOT EXISTS calibration (
    calibration_id     TEXT PRIMARY KEY,
    run_id             TEXT NOT NULL REFERENCES analytical_run(run_id),
    calibration_model  TEXT,
    weighting          TEXT,
    slope              REAL,
    intercept          REAL,
    r_squared          REAL,
    calibration_status TEXT,
    number_of_calibrators INTEGER,
    ingest_id          INTEGER REFERENCES ingest_log(ingest_id)
);

CREATE TABLE IF NOT EXISTS calibrator_point (
    calibrator_point_id INTEGER PRIMARY KEY AUTOINCREMENT,
    calibration_id      TEXT NOT NULL REFERENCES calibration(calibration_id),
    level_name          TEXT NOT NULL,
    nominal_value       REAL,
    back_calculated     REAL,
    percent_deviation   REAL,
    -- Which calibrators were dropped is itself an early drift signal.
    included            INTEGER NOT NULL DEFAULT 1
);

-- ---------------------------------------------------------------- events
CREATE TABLE IF NOT EXISTS lab_event (
    event_id          TEXT PRIMARY KEY,
    entity_type       TEXT NOT NULL,   -- instrument | method | run | material
    entity_id         TEXT NOT NULL,
    event_type        TEXT NOT NULL,
    event_timestamp   TEXT NOT NULL,
    description       TEXT,
    ingest_id         INTEGER REFERENCES ingest_log(ingest_id)
);

-- ---------------------------------------------------------------- indexes
CREATE INDEX IF NOT EXISTS ix_run_method_instrument
    ON analytical_run (method_id, instrument_id, acquisition_timestamp);
CREATE INDEX IF NOT EXISTS ix_qc_result_run
    ON qc_result (run_id, qc_level);
CREATE INDEX IF NOT EXISTS ix_event_entity
    ON lab_event (entity_type, entity_id, event_timestamp);
