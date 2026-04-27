import pandas as pd
import pytest

from analysis.aggregation import aggregate_duplicate_batch_rows, pivot_long_to_wide


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
