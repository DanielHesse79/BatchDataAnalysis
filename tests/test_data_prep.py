import pandas as pd
import pytest

from analysis.data_prep import DataPrepError, merge_process_and_qc_data


def test_merge_matches_normalized_batch_ids_without_changing_display_id():
    process_dataframe = pd.DataFrame(
        {
            "Batch ID": ["B-001", "B-002"],
            "temperature_C": [36.8, 37.1],
        }
    )
    qc_dataframe = pd.DataFrame(
        {
            "batch_id": ["B001", "B002"],
            "yield_g_L": [4.1, 4.4],
        }
    )

    result = merge_process_and_qc_data(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="Batch ID",
        qc_batch_id_column="batch_id",
        outcome_columns=["yield_g_L"],
    )

    assert result.matched_batch_count == 2
    assert result.dataframe["batch_id"].tolist() == ["B-001", "B-002"]


def test_merge_rejects_duplicate_normalized_batch_ids():
    process_dataframe = pd.DataFrame(
        {
            "batch_id": ["B-001", "B001"],
            "temperature_C": [36.8, 37.1],
        }
    )
    qc_dataframe = pd.DataFrame({"batch_id": ["B001"], "yield_g_L": [4.1]})

    with pytest.raises(DataPrepError, match="duplicate IDs"):
        merge_process_and_qc_data(
            process_dataframe=process_dataframe,
            qc_dataframe=qc_dataframe,
            process_batch_id_column="batch_id",
            qc_batch_id_column="batch_id",
            outcome_columns=["yield_g_L"],
        )


def test_merge_rejects_outcome_name_collision_with_process_columns():
    process_dataframe = pd.DataFrame(
        {
            "batch_id": ["B001"],
            "yield_g_L": [999.0],
        }
    )
    qc_dataframe = pd.DataFrame({"batch_id": ["B001"], "yield_g_L": [4.1]})

    with pytest.raises(DataPrepError, match="also exist in the process file"):
        merge_process_and_qc_data(
            process_dataframe=process_dataframe,
            qc_dataframe=qc_dataframe,
            process_batch_id_column="batch_id",
            qc_batch_id_column="batch_id",
            outcome_columns=["yield_g_L"],
        )
