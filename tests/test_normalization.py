import pandas as pd

from analysis.normalization import normalize_batch_id_series, normalize_dataframe_values


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


def test_batch_id_normalization_matches_common_field_formats():
    values = pd.Series([" B-001 ", "B 001", "b_001", "12.0", None])

    normalized = normalize_batch_id_series(values)

    assert normalized.iloc[0] == "B001"
    assert normalized.iloc[1] == "B001"
    assert normalized.iloc[2] == "B001"
    assert normalized.iloc[3] == "12"
    assert pd.isna(normalized.iloc[4])
