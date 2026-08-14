import pandas as pd

from analysis.readiness import calculate_data_readiness


def build_readiness_frames(qc_outcome_values: list) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build a minimal matched process/QC pair with one outcome column."""
    batch_ids = [f"B-{index:03d}" for index in range(1, len(qc_outcome_values) + 1)]
    process_dataframe = pd.DataFrame(
        {
            "batch_id": batch_ids,
            "temperature_C": [36.0 + index * 0.1 for index in range(len(batch_ids))],
        }
    )
    qc_dataframe = pd.DataFrame(
        {
            "batch_id": [batch_id.replace("-", "") for batch_id in batch_ids],
            "yield_percent": qc_outcome_values,
        }
    )
    return process_dataframe, qc_dataframe


def calculate_readiness_for(qc_outcome_values: list):
    """Run readiness with the app's usual duplicate handling defaults."""
    process_dataframe, qc_dataframe = build_readiness_frames(qc_outcome_values)
    return calculate_data_readiness(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="batch_id",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_percent"],
        process_duplicate_strategy="keep_first",
        qc_duplicate_strategy="keep_first",
    )


def test_decimal_comma_outcomes_are_not_a_readiness_blocker():
    """`pd.to_numeric` rejects "81,2 %", which the app's own parser reads fine."""
    result = calculate_readiness_for(["81,2 %", "79,4 %", "80,0 %"])

    assert result.blockers == []
    assert not any("no numeric values" in warning for warning in result.warnings)


def test_outcomes_with_no_readable_numbers_are_still_blocked():
    result = calculate_readiness_for(["Conforms", "Conforms", "Not tested"])

    assert any("no numeric values after parsing" in blocker for blocker in result.blockers)


def test_partially_unreadable_outcomes_are_warned_about_not_blocked():
    result = calculate_readiness_for(["81,2 %", "Conforms", "Not tested", "80,0 %"])

    assert result.blockers == []
    assert any("missing or non-numeric" in warning for warning in result.warnings)


def test_duplicate_batch_ids_are_still_counted_from_normalized_keys():
    """Readiness reuses one normalized key series; the counts must not change."""
    process_dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B001", "B-002"],
            "temperature_C": [36.8, 37.1, 35.0],
        }
    )
    qc_dataframe = pd.DataFrame(
        {
            "batch_id": ["B001", "B002"],
            "yield_percent": [81.2, 79.4],
        }
    )

    result = calculate_data_readiness(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="batch_id",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_percent"],
        process_duplicate_strategy="error",
        qc_duplicate_strategy="keep_first",
    )

    assert result.details["process_duplicate_batch_ids"] == 1
    assert result.details["qc_duplicate_batch_ids"] == 0
    assert any("duplicate batch ID" in blocker for blocker in result.blockers)


def test_excel_float_batch_ids_still_match_plain_integer_ids():
    """"12.00" versus "12" used to leave readiness with no candidate matches."""
    process_dataframe = pd.DataFrame({"batch_id": ["12.00", "13.00"], "temperature_C": [36.8, 37.1]})
    qc_dataframe = pd.DataFrame({"batch_id": ["12", "13"], "yield_percent": [81.2, 79.4]})

    result = calculate_data_readiness(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="batch_id",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_percent"],
        process_duplicate_strategy="keep_first",
        qc_duplicate_strategy="keep_first",
    )

    assert result.details["candidate_matched_batches"] == 2
    assert result.blockers == []
