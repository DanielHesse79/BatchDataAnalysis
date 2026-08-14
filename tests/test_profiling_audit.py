import numpy as np
import pandas as pd

from analysis.audit import detect_multicollinearity_pairs, run_preflight_audit, summarize_row_missingness
from analysis.profiling import calculate_missingness, classify_series, profile_merged_data


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


def test_all_missing_columns_are_typed_empty_instead_of_binary():
    """An all-NaN sensor column has zero levels, so `<= 2` called it binary."""
    dataframe = build_audit_fixture_dataframe()
    dataframe["dead_sensor"] = np.nan

    profile_result = profile_merged_data(
        dataframe,
        outcome_columns=["yield_g_L"],
        batch_id_column="batch_id",
    )

    variable_types = dict(
        zip(profile_result.variable_types["column"], profile_result.variable_types["type"])
    )
    assert variable_types["dead_sensor"] == "empty"
    assert profile_result.variable_type_counts["empty"] == 1
    assert classify_series(pd.Series([np.nan, np.nan])) == "empty"


def test_empty_columns_are_kept_out_of_categorical_audit_checks():
    dataframe = build_audit_fixture_dataframe()
    dataframe["dead_sensor"] = np.nan

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

    assert "dead_sensor" not in audit_result.high_cardinality_categoricals["column"].tolist()
    assert "dead_sensor" not in audit_result.confounding_categoricals["column"].tolist()


def test_missing_like_text_tokens_are_counted_as_missing_values():
    """"n/a" and "not tested" in a text column reported as 0% missing before."""
    dataframe = pd.DataFrame(
        {
            "batch_id": ["B001", "B002", "B003", "B004"],
            "appearance": ["Conforms", "n/a", "-", "not tested"],
        }
    )

    missingness = calculate_missingness(dataframe)
    appearance_row = missingness[missingness["column"] == "appearance"].iloc[0]

    assert appearance_row["missing_count"] == 3
    assert appearance_row["missing_percent"] == 75.0
    assert bool(appearance_row["flag"]) is True


def test_row_missingness_counts_missing_like_text_tokens():
    dataframe = pd.DataFrame(
        {
            "appearance": ["Conforms", "n/a", "Conforms"],
            "temperature_C": [36.8, 37.1, np.nan],
            "yield_g_L": [4.1, 4.2, 4.3],
        }
    )

    row_missingness = summarize_row_missingness(
        dataframe,
        process_columns=["appearance", "temperature_C"],
        outcome_columns=["yield_g_L"],
    )
    metric_counts = dict(zip(row_missingness["metric"], row_missingness["count"]))

    assert metric_counts["rows_with_missing_process_field"] == 2
    assert metric_counts["rows_with_any_missing_selected_field"] == 2
    assert metric_counts["rows_with_missing_outcome_field"] == 0


def test_correlated_pairs_need_enough_overlapping_rows():
    """Correlations from a handful of rows are noise, not multicollinearity."""
    dataframe = pd.DataFrame(
        {
            "temperature_C": [35.0, 36.0, 37.0, 38.0, 39.0, 40.0, np.nan, np.nan],
            "temperature_copy_C": [35.1, 36.1, 37.1, 38.1, 39.1, 40.1, 41.0, 42.0],
        }
    )

    pairs = detect_multicollinearity_pairs(
        dataframe,
        process_columns=["temperature_C", "temperature_copy_C"],
    )

    assert pairs.empty


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
