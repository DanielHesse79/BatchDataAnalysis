import pandas as pd
import pytest

from analysis.specs import SpecError, assess_specs


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
