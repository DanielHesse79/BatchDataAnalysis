import math

import numpy as np
import pandas as pd
import pytest

from analysis.specs import (
    SpecError,
    assess_specs,
    build_out_of_spec_rows,
    calculate_capability,
)


# Mean 10.0 with a sample standard deviation of exactly 2.0 over ten values,
# so every expected capability index below is a clean hand-computed number.
HAND_COMPUTED_VALUES = pd.Series([7.0, 8.0, 8.0, 9.0, 10.0, 10.0, 11.0, 12.0, 12.0, 13.0])


def get_spec_row(spec_assessment, variable_name: str) -> pd.Series:
    rows = spec_assessment.variable_summary[
        spec_assessment.variable_summary["variable"] == variable_name
    ]
    assert not rows.empty
    return rows.iloc[0]


def test_mock_specs_flag_planted_window_and_qc_problems(mock_spec_pipeline):
    spec_assessment = mock_spec_pipeline["specs"]

    sodium_hydroxide = get_spec_row(spec_assessment, "sodium_hydroxide_g")
    moisture = get_spec_row(spec_assessment, "moisture_percent")
    drying_temperature = get_spec_row(spec_assessment, "drying_temperature_C")
    neutralization_ph = get_spec_row(spec_assessment, "pH_after_neutralization")

    assert sodium_hydroxide["classification"] == "Possibly too wide"
    assert moisture["classification"] == "High failure risk"
    assert drying_temperature["classification"] == "Possibly too narrow"
    assert neutralization_ph["classification"] == "Target may be off-center"
    assert not spec_assessment.out_of_spec_batches.empty
    assert "moisture_percent" in spec_assessment.out_of_spec_batches["variable"].tolist()


def test_spec_assessment_reports_unmatched_rows_without_crashing(mock_spec_pipeline):
    spec_dataframe = pd.DataFrame(
        [
            {
                "variable": "not_in_data",
                "role": "process",
                "target": 1,
                "lower_limit": 0,
                "upper_limit": 2,
            }
        ]
    )

    result = assess_specs(
        dataframe=mock_spec_pipeline["dataframe"],
        spec_dataframe=spec_dataframe,
        process_columns=mock_spec_pipeline["profile"].process_columns,
        outcome_columns=mock_spec_pipeline["profile"].outcome_columns,
    )

    assert result.variable_summary.empty
    assert result.unmatched_specs["variable"].tolist() == ["not_in_data"]
    assert result.warnings


def test_spec_assessment_rejects_crossed_limits(mock_spec_pipeline):
    """Transposed limits count every batch as both below and above the spec."""
    spec_dataframe = pd.DataFrame(
        [{"variable": "moisture_percent", "role": "qc", "lower_limit": 50, "upper_limit": 10}]
    )

    with pytest.raises(SpecError, match="lower_limit is greater than upper_limit"):
        assess_specs(
            dataframe=mock_spec_pipeline["dataframe"],
            spec_dataframe=spec_dataframe,
            process_columns=mock_spec_pipeline["profile"].process_columns,
            outcome_columns=mock_spec_pipeline["profile"].outcome_columns,
        )


def test_one_sided_and_two_sided_specs_are_still_accepted(mock_spec_pipeline):
    spec_dataframe = pd.DataFrame(
        [
            {"variable": "moisture_percent", "role": "qc", "upper_limit": 20},
            {"variable": "yield_percent", "role": "qc", "lower_limit": 70, "upper_limit": 95},
        ]
    )

    result = assess_specs(
        dataframe=mock_spec_pipeline["dataframe"],
        spec_dataframe=spec_dataframe,
        process_columns=mock_spec_pipeline["profile"].process_columns,
        outcome_columns=mock_spec_pipeline["profile"].outcome_columns,
    )

    assert sorted(result.variable_summary["variable"].tolist()) == [
        "moisture_percent",
        "yield_percent",
    ]


def test_spec_assessment_rejects_duplicate_spec_rows(mock_spec_pipeline):
    spec_dataframe = pd.DataFrame(
        [
            {"variable": "moisture_percent", "role": "qc", "upper_limit": 20},
            {"variable": "moisture_percent", "role": "qc", "upper_limit": 21},
        ]
    )

    with pytest.raises(SpecError, match="duplicate variable rows"):
        assess_specs(
            dataframe=mock_spec_pipeline["dataframe"],
            spec_dataframe=spec_dataframe,
            process_columns=mock_spec_pipeline["profile"].process_columns,
            outcome_columns=mock_spec_pipeline["profile"].outcome_columns,
        )


def test_capability_matches_hand_computed_two_sided_formula():
    pp, ppk = calculate_capability(HAND_COMPUTED_VALUES, lower_limit=4.0, upper_limit=22.0)

    assert pp == pytest.approx(1.5)
    assert ppk == pytest.approx(1.0)


def test_capability_reports_a_one_sided_index_for_an_upper_only_spec():
    """The default spec template makes HCP and aggregate upper-only, so this is the common case."""
    pp, ppk = calculate_capability(
        HAND_COMPUTED_VALUES,
        lower_limit=float("nan"),
        upper_limit=16.0,
    )

    assert pp is None
    assert ppk == pytest.approx(1.0)


def test_capability_reports_a_one_sided_index_for_a_lower_only_spec():
    """The default spec template makes purity lower-only, so this is the common case."""
    pp, ppk = calculate_capability(
        HAND_COMPUTED_VALUES,
        lower_limit=1.0,
        upper_limit=float("nan"),
    )

    assert pp is None
    assert ppk == pytest.approx(1.5)


def test_capability_is_withheld_without_any_limit():
    assert calculate_capability(
        HAND_COMPUTED_VALUES,
        lower_limit=float("nan"),
        upper_limit=float("nan"),
    ) == (None, None)


def test_capability_is_withheld_below_ten_usable_values():
    """Missing values must not be counted toward the minimum sample size."""
    sparse_values = pd.concat([HAND_COMPUTED_VALUES.head(8), pd.Series([np.nan, np.nan, np.nan])])

    assert calculate_capability(sparse_values, lower_limit=4.0, upper_limit=22.0) == (None, None)


def test_capability_is_withheld_when_sigma_is_zero():
    constant_values = pd.Series([5.0] * 12)

    assert calculate_capability(constant_values, lower_limit=0.0, upper_limit=10.0) == (None, None)


def test_capability_stays_finite_for_near_zero_sigma():
    """A nearly constant variable should give a huge but finite index, never inf or a crash."""
    near_constant_values = pd.Series([5.0] * 11 + [5.000001])

    pp, ppk = calculate_capability(near_constant_values, lower_limit=0.0, upper_limit=10.0)

    assert math.isfinite(pp) and math.isfinite(ppk)
    assert ppk > 1000.0


def test_one_sided_qc_spec_gets_a_capability_index_in_the_summary(mock_spec_pipeline):
    spec_dataframe = pd.DataFrame(
        [{"variable": "moisture_percent", "role": "qc", "upper_limit": 20}]
    )

    result = assess_specs(
        dataframe=mock_spec_pipeline["dataframe"],
        spec_dataframe=spec_dataframe,
        process_columns=mock_spec_pipeline["profile"].process_columns,
        outcome_columns=mock_spec_pipeline["profile"].outcome_columns,
    )
    summary_row = get_spec_row(result, "moisture_percent")

    assert summary_row["pp"] is None
    assert summary_row["ppk"] is not None
    assert summary_row["capability_basis"] == "upper_one_sided_ppu_overall_sigma"


def test_two_sided_spec_summary_names_the_overall_sigma_basis(mock_spec_pipeline):
    spec_dataframe = pd.DataFrame(
        [{"variable": "yield_percent", "role": "qc", "lower_limit": 70, "upper_limit": 95}]
    )

    result = assess_specs(
        dataframe=mock_spec_pipeline["dataframe"],
        spec_dataframe=spec_dataframe,
        process_columns=mock_spec_pipeline["profile"].process_columns,
        outcome_columns=mock_spec_pipeline["profile"].outcome_columns,
    )
    summary_row = get_spec_row(result, "yield_percent")

    assert summary_row["pp"] is not None
    assert summary_row["ppk"] is not None
    assert summary_row["capability_basis"] == "two_sided_ppk_overall_sigma"
    assert "cp" not in summary_row.index and "cpk" not in summary_row.index


def test_out_of_spec_rows_fall_back_to_the_row_index_without_a_batch_id_column():
    """Direct library use skips the merge that guarantees batch_id and used to crash on the first flagged row."""
    dataframe = pd.DataFrame({"moisture_percent": [5.0, 25.0, 30.0]}, index=["a", "b", "c"])
    spec_row = pd.Series(
        {
            "variable": "moisture_percent",
            "lower_limit": float("nan"),
            "upper_limit": 20.0,
            "unit": "%",
        }
    )

    rows = build_out_of_spec_rows(
        dataframe=dataframe,
        variable_name="moisture_percent",
        values=dataframe["moisture_percent"],
        spec_row=spec_row,
        role="qc",
    )

    assert [row["batch_id"] for row in rows] == ["b", "c"]
    assert [row["status"] for row in rows] == ["above_upper_limit", "above_upper_limit"]
