import json

import pytest

from analysis.mapping import (
    MappingProfileError,
    SCHEMA_VERSION,
    build_mapping_profile,
    load_mapping_profile,
    mapping_profile_to_json,
)


PROCESS_COLUMNS = ["batch_id", "temperature_C", "ph_setpoint"]
QC_COLUMNS = ["batch_id", "yield_g_L", "purity_percent"]


def build_reference_profile_json(**overrides) -> str:
    """Serialize the profile the app writes today, with optional overrides."""
    profile = build_mapping_profile(
        process_file_name="process.csv",
        qc_file_name="qc.csv",
        process_intake_options={"sheet_name": None, "header_row": 0},
        qc_intake_options={"sheet_name": "qc_results", "header_row": 2},
        process_batch_id_column="batch_id",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_g_L", "purity_percent"],
        process_duplicate_strategy="keep_first",
        qc_duplicate_strategy="mean",
    )
    profile.update(overrides)
    return mapping_profile_to_json(profile)


def test_a_saved_mapping_profile_can_be_loaded_back():
    """Profiles were exportable but never importable, so repeat imports were manual."""
    result = load_mapping_profile(
        build_reference_profile_json(),
        process_columns=PROCESS_COLUMNS,
        qc_columns=QC_COLUMNS,
    )

    assert result.process_batch_id_column == "batch_id"
    assert result.qc_batch_id_column == "batch_id"
    assert result.outcome_columns == ["yield_g_L", "purity_percent"]
    # Numeric parsing is restored too; a profile saved with it off must not come
    # back with it on, or the prepared table differs from the saved one.
    assert result.qc_intake == {
        "sheet_name": "qc_results",
        "header_row": 2,
        "parse_numeric_like_columns": True,
    }
    assert result.process_duplicate_strategy == "keep_first"
    assert result.qc_duplicate_strategy == "mean"
    assert result.version == SCHEMA_VERSION
    assert result.warnings == []


def test_a_profile_can_be_loaded_before_the_files_are_known():
    result = load_mapping_profile(build_reference_profile_json())

    assert result.outcome_columns == ["yield_g_L", "purity_percent"]


def test_files_that_are_not_mapping_profiles_are_rejected():
    profile_json = json.dumps({"profile_type": "something_else", "version": 1})

    with pytest.raises(MappingProfileError, match="not a Batch Insight mapping profile"):
        load_mapping_profile(profile_json)


def test_invalid_json_is_rejected_with_a_readable_message():
    with pytest.raises(MappingProfileError, match="not valid JSON"):
        load_mapping_profile("{not json")


def test_profiles_from_a_newer_schema_version_are_rejected():
    """Silently applying an unknown schema would map columns by guesswork."""
    profile_json = build_reference_profile_json(version=SCHEMA_VERSION + 1)

    with pytest.raises(MappingProfileError, match="newer version"):
        load_mapping_profile(profile_json)


def test_profiles_without_a_version_are_rejected():
    profile_json = build_reference_profile_json(version="one")

    with pytest.raises(MappingProfileError, match="version number"):
        load_mapping_profile(profile_json)


def test_a_missing_batch_id_column_stops_the_profile_from_being_applied():
    with pytest.raises(MappingProfileError, match="batch ID column"):
        load_mapping_profile(
            build_reference_profile_json(),
            process_columns=["lot_number", "temperature_C"],
            qc_columns=QC_COLUMNS,
        )


def test_outcome_columns_missing_from_the_new_file_are_skipped_with_a_warning():
    result = load_mapping_profile(
        build_reference_profile_json(),
        process_columns=PROCESS_COLUMNS,
        qc_columns=["batch_id", "yield_g_L"],
    )

    assert result.outcome_columns == ["yield_g_L"]
    assert any("purity_percent" in warning for warning in result.warnings)


def test_a_profile_whose_outcomes_are_all_gone_is_rejected():
    with pytest.raises(MappingProfileError, match="None of the saved outcome columns"):
        load_mapping_profile(
            build_reference_profile_json(),
            process_columns=PROCESS_COLUMNS,
            qc_columns=["batch_id", "color_index"],
        )


def test_an_unknown_duplicate_rule_falls_back_to_the_conservative_one():
    profile_json = build_reference_profile_json(
        duplicate_handling={"process": "average_everything", "qc": "keep_first"}
    )

    result = load_mapping_profile(profile_json)

    assert result.process_duplicate_strategy == "error"
    assert result.qc_duplicate_strategy == "keep_first"
    assert any("duplicate handling rule" in warning for warning in result.warnings)


def test_a_profile_without_a_mapping_section_is_rejected():
    profile_json = json.dumps(
        {
            "profile_type": "batch_insight_mapping_profile",
            "version": SCHEMA_VERSION,
        }
    )

    with pytest.raises(MappingProfileError, match="'mapping' section"):
        load_mapping_profile(profile_json)


def test_a_profile_without_outcome_columns_is_rejected():
    profile_json = build_reference_profile_json(
        mapping={
            "process_batch_id_column": "batch_id",
            "qc_batch_id_column": "batch_id",
            "outcome_columns": [],
        }
    )

    with pytest.raises(MappingProfileError, match="outcome columns"):
        load_mapping_profile(profile_json)
