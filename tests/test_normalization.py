import pandas as pd
import pytest

from analysis.normalization import (
    build_missing_like_mask,
    normalize_batch_id_series,
    normalize_column_names,
    normalize_dataframe_values,
    parse_numeric_value,
)


def test_numeric_parser_preserves_identifier_columns_and_parses_units():
    dataframe = pd.DataFrame(
        {
            "Batch ID": ["B-001", "B 002"],
            "operator_id": ["07", "08"],
            "sodium_hydroxide_g": ["100 g", "<95,5 g"],
        }
    )

    result = normalize_dataframe_values(dataframe, exclude_columns=["Batch ID"])

    assert result.dataframe["Batch ID"].tolist() == ["B-001", "B 002"]
    assert result.dataframe["operator_id"].tolist() == ["07", "08"]
    assert result.dataframe["sodium_hydroxide_g"].tolist() == [100.0, 95.5]
    assert int(result.numeric_parse_report["qualifier_count"].sum()) == 1


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("1.0E+03", 1000.0),
        ("1.2e5", 120000.0),
        ("3.5E+03 CFU/mL", 3500.0),
        ("-2.4E-3", -0.0024),
        ("1,5E+03", 1500.0),
        (".5", 0.5),
    ],
)
def test_scientific_notation_keeps_its_exponent(raw_value, expected_value):
    """QC exports use E-notation; truncating to the mantissa is a 10^n error."""
    parsed_value, _ = parse_numeric_value(raw_value)

    assert parsed_value == pytest.approx(expected_value)


@pytest.mark.parametrize("raw_value", ["inf", "-inf", "infinity"])
def test_non_finite_text_is_treated_as_unparseable(raw_value):
    assert parse_numeric_value(raw_value)[0] is None


def test_messy_number_formats_still_parse_after_exponent_support():
    assert parse_numeric_value("4,1")[0] == pytest.approx(4.1)
    assert parse_numeric_value("1 200,5")[0] == pytest.approx(1200.5)
    assert parse_numeric_value("1,234.5")[0] == pytest.approx(1234.5)
    assert parse_numeric_value("<0.05") == (pytest.approx(0.05), "<")


def test_batch_id_normalization_matches_common_field_formats():
    values = pd.Series([" B-001 ", "B 001", "b_001", "12.0", None])

    normalized = normalize_batch_id_series(values)

    assert normalized.iloc[0] == "B001"
    assert normalized.iloc[1] == "B001"
    assert normalized.iloc[2] == "B001"
    assert normalized.iloc[3] == "12"
    assert pd.isna(normalized.iloc[4])


@pytest.mark.parametrize("raw_value", ["12.0", "12.00", "12.000"])
def test_excel_float_batch_ids_match_the_plain_integer_id(raw_value):
    """Excel writes 12 as "12.00"; leaving the zeros blocks every batch match."""
    normalized = normalize_batch_id_series(pd.Series([raw_value]))

    assert normalized.iloc[0] == "12"


def test_deduplicated_column_names_never_collide_with_an_existing_name():
    """["temp", "temp", "temp__2"] must not produce two temp__2 columns."""
    dataframe = pd.DataFrame(
        [[1, 2, 3]],
        columns=["temp", "temp", "temp__2"],
    )

    renamed_dataframe, renamed_columns = normalize_column_names(dataframe)

    assert len(set(renamed_dataframe.columns)) == len(renamed_dataframe.columns)
    assert renamed_dataframe.columns[0] == "temp"
    assert renamed_dataframe.columns[2] == "temp__2"
    assert len(renamed_columns) == 1


def test_normalizing_values_survives_repeated_and_suffixed_column_names():
    """A colliding rename made `dataframe[column]` return a DataFrame and raise."""
    dataframe = pd.DataFrame(
        [["36.8", "37.1", "38.0"], ["36.9", "37.2", "38.1"]],
        columns=["temp", "temp", "temp__2"],
    )

    result = normalize_dataframe_values(dataframe)

    assert len(set(result.dataframe.columns)) == 3
    assert result.dataframe.iloc[0].tolist() == [36.8, 37.1, 38.0]


def test_rename_notes_keep_one_entry_per_repeated_column():
    dataframe = pd.DataFrame([[1, 2, 3]], columns=["temp", "temp", "temp"])

    _, renamed_columns = normalize_column_names(dataframe)

    assert len(renamed_columns) == 2
    assert sorted(renamed_columns.values()) == ["temp__2", "temp__3"]


def test_missing_like_text_tokens_are_counted_as_missing():
    """"n/a" and "not tested" are missing values, not distinct text levels."""
    series = pd.Series(["Conforms", "n/a", "-", "not tested", None, "Hazy"])

    missing_mask = build_missing_like_mask(series)

    assert missing_mask.tolist() == [False, True, True, True, True, False]


def test_numeric_columns_only_count_real_missing_values():
    series = pd.Series([1.0, None, 3.0])

    assert build_missing_like_mask(series).tolist() == [False, True, False]
