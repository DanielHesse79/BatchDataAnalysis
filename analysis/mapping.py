"""Mapping profile helpers for repeat customer file formats."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
import json
from typing import Any

from analysis.aggregation import DUPLICATE_STRATEGIES
from analysis.data_prep import DataPrepError


SCHEMA_VERSION = 1
MAPPING_PROFILE_TYPE = "batch_insight_mapping_profile"
DEFAULT_DUPLICATE_STRATEGY = "error"


class MappingProfileError(DataPrepError):
    """User-facing error raised when a saved mapping profile cannot be applied."""


@dataclass(frozen=True)
class MappingProfileLoadResult:
    """A validated mapping profile, ready to prefill the intake controls."""

    process_batch_id_column: str
    qc_batch_id_column: str
    outcome_columns: list[str]
    process_intake: dict[str, Any]
    qc_intake: dict[str, Any]
    process_duplicate_strategy: str
    qc_duplicate_strategy: str
    process_file_name: str
    qc_file_name: str
    version: int
    warnings: list[str] = field(default_factory=list)


def strip_non_json_metadata(metadata: dict[str, Any]) -> dict[str, Any]:
    """Make intake metadata JSON-serializable.

    Callers should not have to know which metadata values are DataFrames, so the
    conversion lives with the profile writer that needs it.
    """
    serializable_metadata: dict[str, Any] = {}

    for key, value in metadata.items():
        if key == "numeric_parse_report":
            serializable_metadata[key] = (
                value.to_dict(orient="records")
                if value is not None and not value.empty
                else []
            )
        else:
            serializable_metadata[key] = value

    return serializable_metadata


def build_mapping_profile(
    process_file_name: str,
    qc_file_name: str,
    process_intake_options: dict[str, Any],
    qc_intake_options: dict[str, Any],
    process_batch_id_column: str,
    qc_batch_id_column: str,
    outcome_columns: list[str],
    process_duplicate_strategy: str,
    qc_duplicate_strategy: str,
) -> dict[str, Any]:
    """Build a portable mapping profile for repeated imports."""
    return {
        "profile_type": MAPPING_PROFILE_TYPE,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": SCHEMA_VERSION,
        "process_file_name": process_file_name,
        "qc_file_name": qc_file_name,
        "process_intake": process_intake_options,
        "qc_intake": qc_intake_options,
        "mapping": {
            "process_batch_id_column": process_batch_id_column,
            "qc_batch_id_column": qc_batch_id_column,
            "outcome_columns": outcome_columns,
        },
        "duplicate_handling": {
            "process": process_duplicate_strategy,
            "qc": qc_duplicate_strategy,
        },
    }


def mapping_profile_to_json(profile: dict[str, Any]) -> str:
    """Serialize a mapping profile for download."""
    return json.dumps(profile, indent=2, ensure_ascii=True)


def load_mapping_profile(
    json_text: str,
    process_columns: list[str] | None = None,
    qc_columns: list[str] | None = None,
) -> MappingProfileLoadResult:
    """Validate a saved mapping profile against the files that were uploaded now.

    A profile written for last month's export is only reusable if its columns
    still exist. Missing batch ID columns are rejected outright; missing outcome
    columns are dropped with a warning so the rest of the profile still applies.
    """
    profile = parse_mapping_profile_json(json_text)
    validate_profile_identity(profile)

    mapping_section = require_dictionary(profile, "mapping")
    duplicate_section = optional_dictionary(profile, "duplicate_handling")
    profile_warnings: list[str] = []

    process_batch_id_column = require_column_name(mapping_section, "process_batch_id_column")
    qc_batch_id_column = require_column_name(mapping_section, "qc_batch_id_column")
    outcome_columns = require_outcome_columns(mapping_section)

    validate_column_exists(
        process_batch_id_column,
        available_columns=process_columns,
        label="process file",
        description="batch ID column",
    )
    validate_column_exists(
        qc_batch_id_column,
        available_columns=qc_columns,
        label="QC file",
        description="batch ID column",
    )
    outcome_columns = filter_available_outcome_columns(
        outcome_columns,
        available_columns=qc_columns,
        profile_warnings=profile_warnings,
    )

    return MappingProfileLoadResult(
        process_batch_id_column=process_batch_id_column,
        qc_batch_id_column=qc_batch_id_column,
        outcome_columns=outcome_columns,
        process_intake=normalize_intake_options(profile.get("process_intake")),
        qc_intake=normalize_intake_options(profile.get("qc_intake")),
        process_duplicate_strategy=normalize_duplicate_strategy(
            duplicate_section.get("process"),
            label="process",
            profile_warnings=profile_warnings,
        ),
        qc_duplicate_strategy=normalize_duplicate_strategy(
            duplicate_section.get("qc"),
            label="QC",
            profile_warnings=profile_warnings,
        ),
        process_file_name=str(profile.get("process_file_name") or ""),
        qc_file_name=str(profile.get("qc_file_name") or ""),
        version=int(profile["version"]),
        warnings=profile_warnings,
    )


def parse_mapping_profile_json(json_text: str) -> dict[str, Any]:
    """Parse mapping profile JSON into a dictionary."""
    try:
        profile = json.loads(json_text)
    except (TypeError, ValueError) as error:
        raise MappingProfileError(
            "This mapping profile is not valid JSON. Upload the file downloaded from this app."
        ) from error

    if not isinstance(profile, dict):
        raise MappingProfileError("A mapping profile must be a JSON object.")

    return profile


def validate_profile_identity(profile: dict[str, Any]) -> None:
    """Check that a file really is a mapping profile this version understands."""
    if profile.get("profile_type") != MAPPING_PROFILE_TYPE:
        raise MappingProfileError(
            "This file is not a Batch Insight mapping profile "
            f"(expected profile_type '{MAPPING_PROFILE_TYPE}')."
        )

    raw_version = profile.get("version")
    if not isinstance(raw_version, int) or isinstance(raw_version, bool) or raw_version < 1:
        raise MappingProfileError("This mapping profile has no usable version number.")

    if raw_version > SCHEMA_VERSION:
        raise MappingProfileError(
            f"This mapping profile was saved by a newer version (version {raw_version}; "
            f"this app understands up to version {SCHEMA_VERSION})."
        )


def require_dictionary(profile: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return a required profile section."""
    section = profile.get(section_name)
    if not isinstance(section, dict):
        raise MappingProfileError(f"This mapping profile has no '{section_name}' section.")
    return section


def optional_dictionary(profile: dict[str, Any], section_name: str) -> dict[str, Any]:
    """Return an optional profile section as a dictionary."""
    section = profile.get(section_name)
    return section if isinstance(section, dict) else {}


def require_column_name(mapping_section: dict[str, Any], key: str) -> str:
    """Return a required column name from the mapping section."""
    value = mapping_section.get(key)
    if not isinstance(value, str) or not value.strip():
        raise MappingProfileError(f"This mapping profile is missing '{key}'.")
    return value


def require_outcome_columns(mapping_section: dict[str, Any]) -> list[str]:
    """Return the saved outcome columns from the mapping section."""
    values = mapping_section.get("outcome_columns")
    if not isinstance(values, list) or not values:
        raise MappingProfileError("This mapping profile does not list any outcome columns.")

    outcome_columns = [str(value) for value in values if isinstance(value, str) and value.strip()]
    if not outcome_columns:
        raise MappingProfileError("This mapping profile does not list any outcome columns.")
    return outcome_columns


def validate_column_exists(
    column_name: str,
    available_columns: list[str] | None,
    label: str,
    description: str,
) -> None:
    """Raise when a saved column is not present in the uploaded file."""
    if available_columns is None:
        return

    if column_name not in set(available_columns):
        raise MappingProfileError(
            f"The saved {description} '{column_name}' is not in the uploaded {label}. "
            "Select the columns manually, then save a new mapping profile."
        )


def filter_available_outcome_columns(
    outcome_columns: list[str],
    available_columns: list[str] | None,
    profile_warnings: list[str],
) -> list[str]:
    """Drop saved outcome columns that the uploaded QC file no longer contains."""
    if available_columns is None:
        return outcome_columns

    available_set = set(available_columns)
    kept_columns = [column for column in outcome_columns if column in available_set]
    missing_columns = [column for column in outcome_columns if column not in available_set]

    if not kept_columns:
        raise MappingProfileError(
            "None of the saved outcome columns are in the uploaded QC file: "
            + ", ".join(outcome_columns)
        )

    if missing_columns:
        profile_warnings.append(
            "These saved outcome columns are not in the uploaded QC file and were skipped: "
            + ", ".join(missing_columns)
        )

    return kept_columns


def normalize_intake_options(intake_options: Any) -> dict[str, Any]:
    """Return the sheet/header intake options a profile can safely restore."""
    if not isinstance(intake_options, dict):
        return {"sheet_name": None, "header_row": 0}

    sheet_name = intake_options.get("sheet_name")
    header_row = intake_options.get("header_row")
    is_usable_header_row = isinstance(header_row, int) and not isinstance(header_row, bool)

    return {
        "sheet_name": sheet_name if isinstance(sheet_name, str) and sheet_name else None,
        "header_row": header_row if is_usable_header_row and header_row >= 0 else 0,
    }


def normalize_duplicate_strategy(
    strategy: Any,
    label: str,
    profile_warnings: list[str],
) -> str:
    """Return a known duplicate strategy, falling back to the conservative one."""
    if isinstance(strategy, str) and strategy in DUPLICATE_STRATEGIES:
        return strategy

    profile_warnings.append(
        f"The saved {label} duplicate handling rule is not recognized; "
        f"'{DUPLICATE_STRATEGIES[DEFAULT_DUPLICATE_STRATEGY]}' was used instead."
    )
    return DEFAULT_DUPLICATE_STRATEGY
