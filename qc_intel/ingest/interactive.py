"""Interactive intake: a messy laboratory export turned into canonical QC rows.

The batch application already copes with real exported files - sheets, header
rows three lines down, decimal commas, units written into the cells. That code
is reused here rather than reimplemented. What QC data adds is a target table, a
column mapping the analyst can correct before anything is written, and a screen
for personal data that runs *before* the first insert.

The screen is the point. A trending tool for a clinical laboratory should not be
the place where identifiable data first enters a database, so unmapped columns
are dropped rather than stored, and a column that looks like a person is refused
rather than warned about. That is privacy by architecture instead of a policy
document.

No Streamlit in this module. It returns data; the panel renders it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable
import re

import pandas as pd

from analysis.normalization import parse_numeric_series
from qc_intel.ingest.generic_tabular import REQUIRED_COLUMNS, TABLE_COLUMNS


# Tables an analyst can load, in the order they have to be loaded: a run needs
# its method and instrument to exist, and a result needs its run.
TARGET_TABLES = [
    ("method", "Methods", "One row per assay method and version."),
    ("instrument", "Instruments", "One row per instrument."),
    ("material_lot", "Materials and lots", "Reference standards, reagents, columns."),
    ("analytical_run", "Runs", "One row per analytical run, with its acquisition time."),
    ("qc_result", "QC results", "One row per QC measurement."),
    ("calibration", "Calibrations", "One row per calibration curve."),
    ("lab_event", "Laboratory events", "Maintenance, calibration, column changes."),
]

# Columns the schema stores as numbers or timestamps. Everything else is text.
# Listed explicitly rather than sniffed, because guessing the type of a QC
# column from its contents is exactly how a level named "1.0" becomes a number.
NUMERIC_COLUMNS = {
    "lloq", "uloq", "concentration", "nominal_value", "measured_value",
    "percent_bias", "analyte_area", "is_area", "area_ratio", "retention_time",
    "acceptance_limit_lower", "acceptance_limit_upper", "slope", "intercept",
    "r_squared", "back_calculated", "percent_deviation",
}
INTEGER_COLUMNS = {
    "replicate_number", "injection_index", "manual_integration",
    "column_injection_count", "number_of_calibrators", "included",
}
TIMESTAMP_COLUMNS = {
    "acquisition_timestamp", "event_timestamp", "effective_date", "retired_date",
    "installation_date", "date_opened", "expiry_date",
}

# Alternate headings seen in LIMS and chromatography exports. Matching is on
# normalised tokens, so "Acq. Date/Time" and "acq_date_time" behave the same.
SYNONYMS: dict[str, tuple[str, ...]] = {
    "run_id": (
        "run", "runname", "batch", "batchid", "batchname", "sequence",
        "sequencename", "sampleset", "samplesetname",
    ),
    "method_id": ("method", "methodname", "assay", "assayid", "test", "testcode"),
    "instrument_id": ("instrument", "system", "systemname", "equipment", "analyser", "analyzer"),
    "acquisition_timestamp": (
        "acquired", "acquisitiondate", "acqdate", "acqdatetime", "injectiontime",
        "datetime", "date", "timestamp", "rundate", "analysisdate",
    ),
    "qc_level": ("level", "qclevel", "controllevel", "qc", "control", "qcname"),
    "measured_value": (
        "result", "value", "concentration", "conc", "measured", "found",
        "calculatedconcentration", "backcalculated",
    ),
    "nominal_value": ("nominal", "theoretical", "target", "expected", "assigned"),
    "percent_bias": ("bias", "percentbias", "deviation", "percentdeviation", "accuracy"),
    "units": ("unit", "uom", "measurementunit"),
    "pass_fail": ("passfail", "status", "verdict", "result_status", "flag"),
    "acceptance_status": ("runstatus", "accepted", "acceptance", "disposition"),
    "analyte": ("compound", "analytename", "component"),
    "matrix": ("sampletype", "matrixtype"),
    "replicate_number": ("replicate", "rep", "repnumber"),
    "injection_index": ("injection", "injectionnumber", "injnumber"),
    "retention_time": ("rt", "retentiontime"),
    "analyte_area": ("area", "peakarea", "analytearea"),
    "is_area": ("isarea", "istdarea", "internalstandardarea"),
    "area_ratio": ("arearatio", "responseratio", "response"),
    "analyst_ref": ("analyst", "operator", "performedby", "user"),
    "study_id": ("study", "project", "protocol"),
    "column_id": ("column", "columnserial", "columnid"),
    "event_timestamp": ("eventdate", "eventtime", "date"),
    "event_type": ("event", "eventname", "activity", "maintenancetype"),
    "description": ("comment", "notes", "remark", "detail"),
}

# Column names that suggest identifiable data. Deliberately broad, in English and
# Swedish, because a false alarm costs a click and a miss costs a data breach.
PERSONAL_NAME_TOKENS = (
    "patient", "subject", "donor", "volunteer", "participant", "person",
    "personnummer", "personalid", "nationalid", "ssn", "socialsecurity", "nhs",
    "mrn", "medicalrecord", "firstname", "lastname", "surname", "fullname",
    "forename", "givenname", "namn", "fornamn", "efternamn", "birth", "dob",
    "fodelsedatum", "fodd", "diagnosis", "diagnos", "email", "mail", "phone",
    "telefon", "mobile", "address", "adress", "postcode", "postnummer",
)
# A bare "name" or "id" is too common to flag on its own; these qualify it.
PERSONAL_CONTEXT_TOKENS = ("patient", "subject", "donor", "person", "participant")

SWEDISH_IDENTITY_NUMBER = re.compile(r"\b(?:19|20)?\d{6}[-+]?\d{4}\b")
EMAIL_ADDRESS = re.compile(r"[^@\s]+@[^@\s]+\.[A-Za-z]{2,}")

CONTENT_SAMPLE_ROWS = 200


@dataclass(frozen=True)
class PersonalDataFinding:
    """A column that should not enter the database."""

    column: str
    reason: str
    blocking: bool


@dataclass
class MappingReport:
    """What happened when a mapping was applied."""

    row_count: int = 0
    mapped: dict[str, str] = field(default_factory=dict)
    dropped_columns: list[str] = field(default_factory=list)
    unparsed_numbers: dict[str, int] = field(default_factory=dict)
    unparsed_timestamps: dict[str, int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)


def normalise_name(name: object) -> str:
    """Reduce a heading to comparable tokens: 'Acq. Date/Time' -> 'acqdatetime'."""
    return re.sub(r"[^a-z0-9]+", "", str(name).lower())


def suggest_target_table(source_columns: Iterable[str]) -> str | None:
    """Guess which canonical table a file is trying to populate.

    Scored on how many of the table's *required* columns can be matched, because
    a file that cannot fill those is not that table however many optional
    columns it happens to share.
    """
    normalised = {normalise_name(column) for column in source_columns}

    best_table, best_score = None, 0.0
    for table, required in REQUIRED_COLUMNS.items():
        matched = sum(
            1 for column in required
            if _match_source_column(column, normalised) is not None
        )
        score = matched / len(required)
        # qc_result and analytical_run share run_id; prefer the more specific one
        # by weighting on how many optional columns also match.
        optional = [c for c in TABLE_COLUMNS[table] if c not in required]
        optional_matched = sum(
            1 for column in optional
            if _match_source_column(column, normalised) is not None
        )
        score += 0.01 * optional_matched

        if score > best_score:
            best_table, best_score = table, score

    # Below a full set of required columns the guess is not worth offering.
    return best_table if best_score >= 1.0 else None


def suggest_column_mapping(source_columns: Iterable[str], table: str) -> dict[str, str | None]:
    """Propose a source column for each canonical column of a table."""
    source_columns = list(source_columns)
    by_normalised = {normalise_name(column): column for column in source_columns}

    mapping: dict[str, str | None] = {}
    taken: set[str] = set()
    # Required columns choose first, so an optional column cannot steal a match.
    ordered = REQUIRED_COLUMNS.get(table, []) + [
        column for column in TABLE_COLUMNS[table]
        if column not in REQUIRED_COLUMNS.get(table, [])
    ]

    for canonical in ordered:
        if canonical in {"ingest_id", "source_row"}:
            continue  # written by the importer, never mapped from a file
        match = _match_source_column(canonical, set(by_normalised) - {
            normalise_name(name) for name in taken
        })
        chosen = by_normalised.get(match) if match else None
        mapping[canonical] = chosen
        if chosen:
            taken.add(chosen)

    return mapping


def _match_source_column(canonical: str, normalised_sources: set[str]) -> str | None:
    """Return the normalised source name that best fits a canonical column."""
    target = normalise_name(canonical)
    if target in normalised_sources:
        return target

    for synonym in SYNONYMS.get(canonical, ()):
        if synonym in normalised_sources:
            return synonym

    # Fall back to containment, which catches 'qc_level_name' for 'qc_level'.
    candidates = [name for name in normalised_sources if target and target in name]
    if len(candidates) == 1:
        return candidates[0]

    return None


def screen_for_personal_data(frame: pd.DataFrame) -> list[PersonalDataFinding]:
    """Flag columns that look like they identify a person.

    Both the heading and the values are examined: an export can call a column
    'Ref 2' and fill it with personal identity numbers.
    """
    findings: list[PersonalDataFinding] = []

    for column in frame.columns:
        normalised = normalise_name(column)

        matched_token = next(
            (token for token in PERSONAL_NAME_TOKENS if token in normalised), None
        )
        if matched_token is not None:
            findings.append(PersonalDataFinding(
                column=str(column),
                reason=f"the heading contains '{matched_token}'",
                blocking=True,
            ))
            continue

        if normalised in {"name", "id", "ref"} or normalised.endswith("name"):
            if any(token in normalised for token in PERSONAL_CONTEXT_TOKENS):
                findings.append(PersonalDataFinding(
                    column=str(column),
                    reason="the heading names a person",
                    blocking=True,
                ))
                continue

        content_reason = _screen_column_contents(frame[column])
        if content_reason is not None:
            findings.append(PersonalDataFinding(
                column=str(column), reason=content_reason, blocking=True,
            ))

    return findings


def _screen_column_contents(series: pd.Series) -> str | None:
    """Return why a column's values look identifiable, or None."""
    if pd.api.types.is_numeric_dtype(series):
        return None

    sample = series.dropna().astype(str).head(CONTENT_SAMPLE_ROWS)
    if sample.empty:
        return None

    if sample.map(lambda value: EMAIL_ADDRESS.search(value) is not None).any():
        return "the values contain e-mail addresses"

    identity_hits = sample.map(
        lambda value: SWEDISH_IDENTITY_NUMBER.fullmatch(value.strip()) is not None
    ).sum()
    if identity_hits >= max(2, 0.2 * len(sample)):
        return "the values look like personal identity numbers"

    return None


def missing_required_columns(mapping: dict[str, str | None], table: str) -> list[str]:
    """Return the required canonical columns the mapping does not fill."""
    return [
        column for column in REQUIRED_COLUMNS.get(table, [])
        if not mapping.get(column)
    ]


def apply_mapping(
    frame: pd.DataFrame,
    mapping: dict[str, str | None],
    table: str,
) -> tuple[pd.DataFrame, MappingReport]:
    """Rename, convert and trim a source frame to canonical columns.

    Anything the mapping does not name is dropped rather than carried along, so
    a column nobody chose cannot end up in the database.
    """
    report = MappingReport(row_count=len(frame))
    canonical = pd.DataFrame(index=frame.index)

    chosen = {source for source in mapping.values() if source}
    report.dropped_columns = [
        str(column) for column in frame.columns if column not in chosen
    ]

    for canonical_name, source_name in mapping.items():
        if not source_name or source_name not in frame.columns:
            continue

        report.mapped[canonical_name] = str(source_name)
        values = frame[source_name]

        if canonical_name in TIMESTAMP_COLUMNS:
            converted, failures = _to_iso_timestamps(values)
            if failures:
                report.unparsed_timestamps[canonical_name] = failures
        elif canonical_name in NUMERIC_COLUMNS or canonical_name in INTEGER_COLUMNS:
            converted, failures = _to_numbers(values)
            if canonical_name in INTEGER_COLUMNS:
                converted = converted.round().astype("Int64")
            if failures:
                report.unparsed_numbers[canonical_name] = failures
        else:
            converted = values.map(
                lambda value: None if pd.isna(value) else str(value).strip()
            )

        canonical[canonical_name] = converted

    # Schema order, with the columns the importer fills left absent.
    ordered = [
        column for column in TABLE_COLUMNS[table] if column in canonical.columns
    ]
    return canonical[ordered], report


def _to_numbers(series: pd.Series) -> tuple[pd.Series, int]:
    """Parse numbers written with decimal commas, units or qualifiers."""
    if pd.api.types.is_numeric_dtype(series):
        return series.astype("float64"), 0

    parsed, _qualifiers = parse_numeric_series(series)
    had_value = ~series.isna() & (series.astype(str).str.strip() != "")
    failures = int((had_value & parsed.isna()).sum())
    return parsed, failures


def _to_iso_timestamps(series: pd.Series) -> tuple[pd.Series, int]:
    """Convert to ISO-8601 text, the form the schema stores.

    Timezone-aware values are converted to UTC. Naive values are kept as written
    rather than assumed to be UTC: silently shifting a laboratory's local
    timestamps would move every event relative to every run.
    """
    parsed = pd.to_datetime(series, errors="coerce", format="mixed")

    if isinstance(parsed.dtype, pd.DatetimeTZDtype):
        parsed = parsed.dt.tz_convert("UTC").dt.tz_localize(None)

    had_value = ~series.isna() & (series.astype(str).str.strip() != "")
    failures = int((had_value & parsed.isna()).sum())
    return parsed.dt.strftime("%Y-%m-%dT%H:%M:%S"), failures
