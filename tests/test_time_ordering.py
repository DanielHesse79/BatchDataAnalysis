import numpy as np
import pandas as pd
import pytest

from analysis.methods import convert_order_values, make_time_ordered_split
from utils.plots import convert_sort_values


CONVERTERS = [convert_order_values, convert_sort_values]


@pytest.mark.parametrize("converter", CONVERTERS)
def test_missing_dates_stay_missing_instead_of_becoming_int64_min(converter):
    """astype("int64") maps NaT to INT64_MIN, which sorts before every batch."""
    dates = pd.Series(pd.to_datetime(["2024-01-03", None, "2024-01-01", "2024-01-02"]))

    order_values = converter(dates)

    assert order_values.isna().sum() == 1
    assert order_values.dropna().min() > 0
    assert order_values.dropna().idxmin() == 2


@pytest.mark.parametrize("converter", CONVERTERS)
def test_missing_dates_in_text_columns_stay_missing(converter):
    # convert_order_values only treats a column as dates when at least 80% parse,
    # so keep the unparseable share below that gate.
    dates = pd.Series(
        [f"2024-01-{day:02d}" for day in range(1, 10)] + [""],
    )

    order_values = converter(dates)

    assert order_values.isna().sum() == 1
    assert order_values.dropna().min() > 0
    assert order_values.dropna().idxmin() == 0


@pytest.mark.parametrize("converter", CONVERTERS)
def test_numeric_sequence_columns_are_unchanged(converter):
    sequence = pd.Series([3.0, 1.0, 2.0, 4.0])

    assert converter(sequence).tolist() == pytest.approx([3.0, 1.0, 2.0, 4.0])


def test_batches_with_missing_dates_are_excluded_from_the_time_ordered_split():
    """Undated batches must not be treated as the oldest rows in the train set."""
    row_count = 40
    dates = pd.Series(pd.date_range("2024-01-01", periods=row_count, freq="D"))
    dates.iloc[:5] = pd.NaT
    dataframe = pd.DataFrame(
        {
            "run_date": dates,
            "yield_g_L": np.linspace(10.0, 20.0, row_count),
        }
    )

    split = make_time_ordered_split(
        dataframe,
        outcome_column="yield_g_L",
        validation_order_column="run_date",
    )

    assert split["available"] is True
    assert split["train_rows"] + split["test_rows"] == row_count - 5
    undated_indices = set(range(5))
    assert not undated_indices.intersection(split["train_index"])
    assert not undated_indices.intersection(split["test_index"])
