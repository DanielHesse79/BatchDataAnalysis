from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

from analysis.methods import (
    choose_validation_order_column,
    convert_order_values,
    is_date_ordered_column,
    is_sequence_ordered_column,
    make_time_ordered_split,
)
from utils.plots import convert_sort_values


@dataclass
class FakeAuditResult:
    """Stands in for the audit's date/drift table."""

    date_or_drift_columns: pd.DataFrame


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


def build_audit_with_flagged_columns(*column_names: str) -> FakeAuditResult:
    return FakeAuditResult(pd.DataFrame({"column": list(column_names)}))


def test_a_duration_column_is_never_used_as_the_time_order():
    """The audit flags any name containing "time", including "Drying Time".

    Ordering by a duration is not a chronology, and the chosen column is also
    excluded from driver modeling, so a bad pick loses a real process variable.
    """
    dataframe = pd.DataFrame(
        {
            "Drying Time": [6.71, 8.7, 8.2, 7.4, 9.1, 6.9, 8.8, 7.7],
            "yield_percent": np.linspace(70, 80, 8),
        }
    )

    chosen = choose_validation_order_column(
        build_audit_with_flagged_columns("Drying Time"),
        dataframe,
    )

    assert chosen is None


def test_a_batch_counter_is_used_as_the_time_order():
    dataframe = pd.DataFrame(
        {
            "batch_sequence_number": range(1, 21),
            "yield_percent": np.linspace(70, 80, 20),
        }
    )

    chosen = choose_validation_order_column(
        build_audit_with_flagged_columns("batch_sequence_number"),
        dataframe,
    )

    assert chosen == "batch_sequence_number"


def test_a_real_date_column_is_preferred_over_a_counter():
    dataframe = pd.DataFrame(
        {
            "mixing_time_min": [30.5, 31.2, 29.8, 30.1, 32.0, 31.7],
            "production_date": pd.date_range("2024-01-01", periods=6, freq="D"),
            "batch_sequence_number": range(1, 7),
        }
    )

    chosen = choose_validation_order_column(
        build_audit_with_flagged_columns(
            "mixing_time_min", "production_date", "batch_sequence_number"
        ),
        dataframe,
    )

    assert chosen == "production_date"


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        (range(1, 30), True),
        ([6.71, 8.7, 8.2, 7.4, 9.1], False),  # durations are not whole numbers
        ([1, 1, 1, 2, 2, 2], False),  # repeated codes are not a counter
        ([10, 900, 4000, 55000], False),  # sparse ids are not a batch order
    ],
)
def test_sequence_detection(values, expected):
    assert is_sequence_ordered_column(pd.Series(list(values))) is expected


def test_date_detection_ignores_numeric_columns():
    """pandas will happily read plain numbers as dates; a counter is not a date."""
    assert is_date_ordered_column(pd.Series([1, 2, 3, 4, 5])) is False
    assert is_date_ordered_column(pd.Series(pd.date_range("2024-01-01", periods=5))) is True


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
