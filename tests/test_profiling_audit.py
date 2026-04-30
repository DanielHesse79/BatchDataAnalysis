import numpy as np
import pandas as pd

from analysis.audit import run_preflight_audit
from analysis.profiling import profile_merged_data


def build_audit_fixture_dataframe() -> pd.DataFrame:
    """Create a compact dataset with known data-quality problems."""
    row_count = 40
    temperature = np.linspace(35.0, 39.0, row_count)
    yield_values = 2.5 + 0.2 * temperature
    dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{i:03d}" for i in range(row_count)],
            "temperature_C": temperature,
            "temperature_copy_C": temperature * 1.01,
            "deviation_reported": [0, 1] * 20,
            "supplier_lot": ["Lot-A"] * 14 + ["Lot-B"] * 13 + ["Lot-C"] * 13,
            "operator_shift": ["Day", "Night"] * 20,
            "batch_sequence_number": range(1, row_count + 1),
            "yield_estimate_after_release": yield_values,
            "near_constant_flag": [1] * 39 + [0],
            "mostly_missing_sensor": [np.nan] * 12 + list(range(28)),
            "yield_g_L": yield_values,
        }
    )
    dataframe.loc[0, "temperature_C"] = 100.0
    dataframe.loc[0, "temperature_copy_C"] = 101.0
    return dataframe


def test_profile_classifies_variables_and_flags_missing_or_near_constant_columns():
    dataframe = build_audit_fixture_dataframe()

    profile_result = profile_merged_data(
        dataframe,
        outcome_columns=["yield_g_L"],
        batch_id_column="batch_id",
    )

    variable_types = dict(
        zip(profile_result.variable_types["column"], profile_result.variable_types["type"])
    )
    assert variable_types["temperature_C"] == "continuous"
    assert variable_types["supplier_lot"] == "categorical"
    assert variable_types["deviation_reported"] == "binary"
    assert "mostly_missing_sensor" in profile_result.missingness[
        profile_result.missingness["flag"]
    ]["column"].tolist()
    assert "near_constant_flag" in profile_result.near_constant_columns["column"].tolist()


def test_preflight_audit_flags_common_field_data_risks():
    dataframe = build_audit_fixture_dataframe()
    profile_result = profile_merged_data(
        dataframe,
        outcome_columns=["yield_g_L"],
        batch_id_column="batch_id",
    )

    audit_result = run_preflight_audit(
        dataframe,
        process_columns=profile_result.process_columns,
        outcome_columns=profile_result.outcome_columns,
        variable_types=profile_result.variable_types,
    )

    assert "yield_estimate_after_release" in audit_result.leakage_name_warnings[
        "column"
    ].tolist()
    assert "batch_sequence_number" in audit_result.date_or_drift_columns["column"].tolist()
    assert "supplier_lot" in audit_result.confounding_categoricals["column"].tolist()
    assert "temperature_C" in audit_result.outlier_flags["column"].tolist()
    multicollinearity_pairs = audit_result.multicollinearity_pairs[
        ["first_column", "second_column"]
    ].apply(tuple, axis=1).tolist()
    assert ("temperature_C", "temperature_copy_C") in multicollinearity_pairs
    assert audit_result.warning_count >= 5
