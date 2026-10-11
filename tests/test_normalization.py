import pandas as pd
import pytest

from analysis.normalization import (
    build_missing_like_mask,
    normalize_batch_id_series,
    normalize_column_names,
    normalize_dataframe_values,
    parse_numeric_series,
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


@pytest.mark.parametrize(
    "column_name",
    ["inoculation_timestamp", "harvest", "sampled_at"],
)
@pytest.mark.parametrize(
    "values",
    [
        ["2024-01-15", "2024-01-16", "2024-02-01"],
        ["01/15/2024", "01/16/2024", "02/01/2024"],
        ["15.01.2024", "16.01.2024", "01.02.2024"],
        ["2024-01-15 08:30", "2024-01-16T09:45:00", "2024-02-01 10:00:00"],
    ],
)
def test_dates_are_never_turned_into_numbers(column_name, values):
    """Only names containing "date" were protected; 2024-01-15 became 2024."""
    dataframe = pd.DataFrame({column_name: values})

    result = normalize_dataframe_values(dataframe)

    assert result.dataframe[column_name].tolist() == values
    assert result.numeric_parse_report.empty


def test_a_datetime_column_is_left_alone():
    dataframe = pd.DataFrame({"inoculated": pd.to_datetime(["2024-01-15", "2024-01-16"])})

    result = normalize_dataframe_values(dataframe)

    assert pd.api.types.is_datetime64_any_dtype(result.dataframe["inoculated"])


def test_clock_values_are_read_as_hours_not_as_their_hour():
    """08:30 was read as 8."""
    dataframe = pd.DataFrame({"start_clock": ["08:30", "14:15", "8:30 PM", "1:30:00"]})

    result = normalize_dataframe_values(dataframe)

    assert result.dataframe["start_clock"].tolist() == pytest.approx([8.5, 14.25, 20.5, 1.5])
    assert "hours" in result.numeric_parse_report.iloc[0]["notes"]


def test_a_lone_ratio_or_clock_value_has_no_single_number():
    assert parse_numeric_value("1:100")[0] is None
    assert parse_numeric_value("08:30")[0] is None


@pytest.mark.parametrize(
    ("values", "expected"),
    [
        # Hundreds without a comma, so the commas group thousands.
        (["1,250", "980", "2,100"], [1250.0, 980.0, 2100.0]),
        # The messy-field color_index: three decimals with a decimal comma.
        (["1,523", "1.487", "1.612", "1,498"], [1.523, 1.487, 1.612, 1.498]),
        # 95,5 settles it: decimal commas.
        (["95,5", "1,250"], [95.5, 1.25]),
        # 1,234.5 settles it: thousands separators.
        (["1,234.5", "1,250"], [1234.5, 1250.0]),
        (["1,234,567"], [1234567.0]),
    ],
)
def test_the_comma_is_read_from_the_whole_column(values, expected):
    """A single comma was always a decimal comma, so 1,250 batches read as 1.25."""
    parsed_values, _ = parse_numeric_series(pd.Series(values))

    assert parsed_values.tolist() == pytest.approx(expected)


def test_an_unsettled_comma_is_reported_not_guessed_silently():
    dataframe = pd.DataFrame({"feed_total": ["1,250", "2,100", "3,400"]})

    result = normalize_dataframe_values(dataframe)

    assert any("feed_total" in warning and "1,250" in warning for warning in result.warnings)
    assert "ambiguous" in result.numeric_parse_report.iloc[0]["notes"]


@pytest.mark.parametrize("raw_value", ["6.8-7.2", "6,8 - 7,2", "12 to 48 h", "36.5–37.5 C"])
def test_a_range_is_not_reduced_to_its_first_number(raw_value):
    assert parse_numeric_value(raw_value)[0] is None


def test_range_values_are_reported_when_a_column_is_converted():
    dataframe = pd.DataFrame({"ph_final": ["7.0", "7.1", "6.9", "6.8-7.2", "7.05"]})

    result = normalize_dataframe_values(dataframe)

    assert pd.isna(result.dataframe["ph_final"].iloc[3])
    assert any("6.8-7.2" in warning for warning in result.warnings)


@pytest.mark.parametrize(
    ("raw_value", "expected_value"),
    [
        ("5 x 10^6", 5e6),
        ("5x10^6 cells/mL", 5e6),
        ("2,5 × 10^5", 2.5e5),
        ("1.2 x 10**-3", 1.2e-3),
        ("5 x 10⁶", 5e6),
        ("3 × 10⁻²", 3e-2),
    ],
)
def test_powers_of_ten_written_out_keep_their_exponent(raw_value, expected_value):
    """5 x 10^6 was read as 5."""
    assert parse_numeric_value(raw_value)[0] == pytest.approx(expected_value)


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
