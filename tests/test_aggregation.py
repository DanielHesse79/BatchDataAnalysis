import pandas as pd
import pytest

from analysis.aggregation import (
    aggregate_duplicate_batch_rows,
    detect_long_format_candidates,
    pivot_long_to_wide,
)
from analysis.data_prep import get_intake_warnings


def test_long_format_qc_results_can_be_pivoted_to_wide_numeric_table():
    long_dataframe = pd.DataFrame(
        {
            "Batch Identifier": ["B-001", "B-001", "B-002", "B-002"],
            "Test Name": ["yield_percent", "moisture_percent", "yield_percent", "moisture_percent"],
            "Result": ["81,2 %", "18.5%", "79.0%", "<20 %"],
        }
    )

    wide_dataframe = pivot_long_to_wide(
        long_dataframe,
        batch_id_column="Batch Identifier",
        name_column="Test Name",
        value_column="Result",
        aggregation="first",
    )

    assert wide_dataframe["yield_percent"].tolist() == pytest.approx([81.2, 79.0])
    assert wide_dataframe["moisture_percent"].tolist() == pytest.approx([18.5, 20.0])


@pytest.mark.parametrize("aggregation", ["first", "mean"])
def test_pivot_keeps_qualitative_tests_alongside_numeric_tests(aggregation):
    """A mostly-numeric value column must not blank out qualitative CQAs."""
    long_dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B-001", "B-002", "B-002", "B-003", "B-003"],
            "test_name": ["assay", "appearance"] * 3,
            "result": ["10.1", "Conforms", "10.5", "Slightly hazy", "9.8", "Conforms"],
        }
    )

    wide_dataframe = pivot_long_to_wide(
        long_dataframe,
        batch_id_column="batch_id",
        name_column="test_name",
        value_column="result",
        aggregation=aggregation,
    )

    assert "appearance" in wide_dataframe.columns
    assert wide_dataframe["assay"].tolist() == pytest.approx([10.1, 10.5, 9.8])
    assert wide_dataframe["appearance"].tolist() == [
        "Conforms",
        "Slightly hazy",
        "Conforms",
    ]


def test_pivot_of_text_results_does_not_raise_on_numeric_aggregation():
    """Averaging text is impossible, so text tests fall back to the first value."""
    long_dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B-002"],
            "test_name": ["appearance", "appearance"],
            "result": ["Conforms", "Hazy"],
        }
    )

    wide_dataframe = pivot_long_to_wide(
        long_dataframe,
        batch_id_column="batch_id",
        name_column="test_name",
        value_column="result",
        aggregation="mean",
    )

    assert wide_dataframe["appearance"].tolist() == ["Conforms", "Hazy"]


@pytest.mark.parametrize("strategy", ["mean", "median", "keep_first", "keep_last"])
def test_rows_without_batch_ids_are_never_merged_into_one_batch(strategy):
    """Missing IDs are not a shared key; merging them fabricates a batch."""
    dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B001", None, None, None],
            "sodium_hydroxide_g": [99.0, 101.0, 3.0, 4.0, 5.0],
        }
    )

    result = aggregate_duplicate_batch_rows(
        dataframe,
        batch_id_column="batch_id",
        strategy=strategy,
        label="Process",
    )

    unidentified = result.dataframe[result.dataframe["batch_id"].isna()]
    assert len(unidentified) == 3
    assert sorted(unidentified["sodium_hydroxide_g"].tolist()) == [3.0, 4.0, 5.0]
    assert result.duplicate_batch_count == 1
    assert any("no usable batch ID" in warning for warning in result.warnings)


def test_missing_batch_ids_are_reported_even_without_duplicates():
    dataframe = pd.DataFrame({"batch_id": ["B-001", None], "value": [1.0, 2.0]})

    result = aggregate_duplicate_batch_rows(
        dataframe,
        batch_id_column="batch_id",
        strategy="keep_first",
        label="QC",
    )

    assert result.rows_after == 2
    assert any("no usable batch ID" in warning for warning in result.warnings)


@pytest.mark.parametrize("strategy", ["mean", "median"])
def test_numeric_looking_identifier_codes_are_never_averaged(strategy):
    """Averaging operator "07" and "08" invents operator 7.5 as a continuous driver."""
    dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B001", "B-002"],
            "operator_id": ["07", "08", "07"],
            "temperature_C": [36.8, 37.2, 35.0],
        }
    )

    result = aggregate_duplicate_batch_rows(
        dataframe,
        batch_id_column="batch_id",
        strategy=strategy,
        label="Process",
    )

    assert result.dataframe["operator_id"].tolist() == ["07", "07"]
    assert result.dataframe["temperature_C"].tolist() == pytest.approx([37.0, 35.0])


def test_numeric_text_with_units_is_still_averaged_across_replicates():
    dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B001"],
            "sodium_hydroxide_g": ["100 g", "105 g"],
        }
    )

    result = aggregate_duplicate_batch_rows(
        dataframe,
        batch_id_column="batch_id",
        strategy="mean",
        label="Process",
    )

    assert result.dataframe["sodium_hydroxide_g"].tolist() == pytest.approx([102.5])


def test_aggregation_keeps_unique_rows_untouched_and_in_original_order():
    dataframe = pd.DataFrame(
        {
            "batch_id": ["B-002", "B-001", "B001"],
            "appearance": ["Hazy", "Conforms", "Conforms"],
            "temperature_C": [35.0, 36.8, 37.2],
        }
    )

    result = aggregate_duplicate_batch_rows(
        dataframe,
        batch_id_column="batch_id",
        strategy="mean",
        label="Process",
    )

    assert result.dataframe["batch_id"].tolist() == ["B-002", "B-001"]
    assert result.dataframe["temperature_C"].tolist() == pytest.approx([35.0, 37.0])
    assert result.dataframe["appearance"].tolist() == ["Hazy", "Conforms"]


def test_long_format_pivot_merges_batch_id_spellings_into_one_row():
    """"B-001" and "B001" are one batch; pivoting raw IDs half-fills two rows."""
    long_dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B001", "B-002", "B 002"],
            "test_name": ["assay", "moisture", "assay", "moisture"],
            "result": ["10.1", "2.0", "9.9", "2.4"],
        }
    )

    wide_dataframe = pivot_long_to_wide(
        long_dataframe,
        batch_id_column="batch_id",
        name_column="test_name",
        value_column="result",
        aggregation="first",
    )

    assert len(wide_dataframe) == 2
    assert wide_dataframe["batch_id"].tolist() == ["B-001", "B-002"]
    assert wide_dataframe["assay"].tolist() == pytest.approx([10.1, 9.9])
    assert wide_dataframe["moisture"].tolist() == pytest.approx([2.0, 2.4])


def test_long_format_rows_without_a_batch_id_are_never_pivoted_into_one_batch():
    """Missing IDs are not a shared key; pivoting them together fabricates a batch."""
    long_dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "n/a", None],
            "test_name": ["assay", "assay", "assay"],
            "result": ["10.1", "2.0", "3.0"],
        }
    )

    wide_dataframe = pivot_long_to_wide(
        long_dataframe,
        batch_id_column="batch_id",
        name_column="test_name",
        value_column="result",
        aggregation="first",
    )

    assert wide_dataframe["batch_id"].tolist() == ["B-001"]
    assert any(
        "no usable batch ID" in warning
        for warning in get_intake_warnings(wide_dataframe)
    )


def test_long_format_value_candidates_are_detected_from_a_sample():
    """Suggestions sample rows; parsing every cell of a 100k-row export is wasted work."""
    numeric_rows = 200
    text_rows = 100
    long_dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{index:04d}" for index in range(numeric_rows + text_rows)],
            "test_name": ["assay"] * (numeric_rows + text_rows),
            "assay_output": [f"{10 + index / 100:.2f}" for index in range(numeric_rows)]
            + ["Not tested"] * text_rows,
        }
    )

    suggestion = detect_long_format_candidates(long_dataframe)

    assert "assay_output" in suggestion.value_candidates


def test_duplicate_batch_rows_can_be_aggregated_by_normalized_batch_id():
    dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B001", "B-002"],
            "sodium_hydroxide_g": [99.0, 101.0, 95.0],
            "operator_shift": ["Day", "Night", "Day"],
        }
    )

    result = aggregate_duplicate_batch_rows(
        dataframe,
        batch_id_column="batch_id",
        strategy="mean",
        label="Process",
    )

    assert result.duplicate_batch_count == 1
    assert result.rows_after == 2
    assert result.dataframe.loc[0, "sodium_hydroxide_g"] == pytest.approx(100.0)
