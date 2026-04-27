"""Normalization helpers for messy field data.

The app should accept polite demo files, but real production exports often
contain decimal commas, units in cells, hidden spaces, and inconsistent batch
ID formatting. These helpers make those issues visible and produce a cleaner
canonical table for downstream analysis.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

import numpy as np
import pandas as pd


MISSING_TOKENS = {
    "",
    "-",
    "--",
    "na",
    "n/a",
    "nan",
    "none",
    "null",
    "not available",
    "not tested",
    "not done",
    "nd",
    "n.d.",
    "missing",
}

QUALIFIER_PATTERN = re.compile(r"^(<=|>=|<|>|~|approx\.?|about)\s*", re.IGNORECASE)
NUMBER_PATTERN = re.compile(r"[-+]?\d+(?:[\s.,]\d+)*")
HIDDEN_SPACE_PATTERN = re.compile(r"[\u00a0\u200b\u200c\u200d\ufeff]")
IDENTIFIER_COLUMN_TOKENS = [
    "id",
    "batch",
    "lot",
    "operator",
    "shift",
    "reactor",
    "supplier",
    "site",
    "code",
    "name",
    "campaign",
    "date",
]


@dataclass(frozen=True)
class NumericParseReport:
    """Summary of one text column converted to numeric."""

    column: str
    non_missing_values: int
    parsed_values: int
    parsed_fraction: float
    qualifier_count: int
    examples: str


@dataclass(frozen=True)
class NormalizationResult:
    """A normalized DataFrame plus user-facing conversion notes."""

    dataframe: pd.DataFrame
    numeric_parse_report: pd.DataFrame
    warnings: list[str]
    renamed_columns: dict[str, str]


def normalize_dataframe_values(
    dataframe: pd.DataFrame,
    parse_numeric_like_columns: bool = True,
    exclude_columns: list[str] | None = None,
    min_parse_fraction: float = 0.75,
) -> NormalizationResult:
    """Clean column names and convert numeric-like text columns.

    We convert only when most non-missing values parse as numbers. That keeps
    categorical IDs, lot names, and operator codes from being accidentally
    turned into numbers.
    """
    exclude_columns = exclude_columns or []
    normalized_dataframe = dataframe.copy()
    normalized_dataframe, renamed_columns = normalize_column_names(normalized_dataframe)
    warnings: list[str] = []
    parse_reports: list[NumericParseReport] = []

    if renamed_columns:
        warnings.append(
            "Duplicate or blank column names were normalized: "
            + ", ".join(f"{old} -> {new}" for old, new in list(renamed_columns.items())[:6])
        )

    if parse_numeric_like_columns:
        for column_name in normalized_dataframe.columns:
            if column_name in exclude_columns:
                continue
            series = normalized_dataframe[column_name]
            if not should_attempt_numeric_parse(column_name, series):
                continue

            parsed_values, qualifiers = parse_numeric_series(series)
            non_missing_mask = ~series.map(is_missing_like)
            non_missing_count = int(non_missing_mask.sum())
            parsed_count = int(parsed_values[non_missing_mask].notna().sum())
            if non_missing_count == 0:
                continue

            parsed_fraction = parsed_count / non_missing_count
            qualifier_count = int(qualifiers.notna().sum())
            if parsed_fraction >= min_parse_fraction:
                normalized_dataframe[column_name] = parsed_values
                parse_reports.append(
                    NumericParseReport(
                        column=column_name,
                        non_missing_values=non_missing_count,
                        parsed_values=parsed_count,
                        parsed_fraction=round(parsed_fraction, 3),
                        qualifier_count=qualifier_count,
                        examples=example_values(series),
                    )
                )

    report_dataframe = pd.DataFrame([report.__dict__ for report in parse_reports])
    return NormalizationResult(
        dataframe=normalized_dataframe,
        numeric_parse_report=report_dataframe,
        warnings=warnings,
        renamed_columns=renamed_columns,
    )


def normalize_column_names(dataframe: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    """Strip, deblank, and make duplicate column names unique."""
    output_dataframe = dataframe.copy()
    seen: dict[str, int] = {}
    new_columns: list[str] = []
    renamed_columns: dict[str, str] = {}

    for index, original_column in enumerate(output_dataframe.columns):
        original_name = str(original_column).strip()
        cleaned_name = HIDDEN_SPACE_PATTERN.sub("", original_name).strip()
        if not cleaned_name or cleaned_name.lower().startswith("unnamed:"):
            cleaned_name = f"unnamed_column_{index + 1}"

        base_name = cleaned_name
        seen[base_name] = seen.get(base_name, 0) + 1
        if seen[base_name] > 1:
            cleaned_name = f"{base_name}__{seen[base_name]}"

        if cleaned_name != original_name:
            renamed_columns[original_name or f"column_{index + 1}"] = cleaned_name
        new_columns.append(cleaned_name)

    output_dataframe.columns = new_columns
    return output_dataframe, renamed_columns


def normalize_batch_id_series(
    batch_id_series: pd.Series,
    remove_separators: bool = True,
) -> pd.Series:
    """Normalize batch IDs for matching across messy exports."""
    normalized = batch_id_series.astype("string").map(normalize_batch_id_value)
    if remove_separators:
        normalized = normalized.str.replace(r"[\s\-_]+", "", regex=True)
    return normalized


def normalize_batch_id_value(value: Any) -> str | pd.NA:
    """Normalize one batch ID value while preserving missingness."""
    if pd.isna(value):
        return pd.NA

    text = str(value)
    text = HIDDEN_SPACE_PATTERN.sub("", text)
    text = re.sub(r"\s+", " ", text).strip().upper()
    if text.lower() in MISSING_TOKENS:
        return pd.NA

    if re.fullmatch(r"\d+\.0", text):
        text = text[:-2]

    return text


def should_attempt_numeric_parse(column_name: str, series: pd.Series) -> bool:
    """Return whether a column is a plausible numeric-like text column."""
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False

    normalized_column_name = str(column_name).lower().replace(" ", "_")
    column_tokens = set(re.split(r"[^a-z0-9]+", normalized_column_name))
    if column_tokens.intersection(IDENTIFIER_COLUMN_TOKENS):
        return False

    non_missing_values = series[~series.map(is_missing_like)].astype(str)
    if non_missing_values.empty:
        return False

    sample = non_missing_values.head(80)
    numeric_looking_count = int(sample.map(lambda value: NUMBER_PATTERN.search(value) is not None).sum())
    return numeric_looking_count / len(sample) >= 0.50


def parse_numeric_series(series: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Parse a Series containing numbers, units, and qualifiers."""
    parsed_values = []
    qualifiers = []
    for value in series:
        parsed_value, qualifier = parse_numeric_value(value)
        parsed_values.append(parsed_value)
        qualifiers.append(qualifier)

    return (
        pd.Series(parsed_values, index=series.index, dtype="float64"),
        pd.Series(qualifiers, index=series.index, dtype="object"),
    )


def parse_numeric_value(value: Any) -> tuple[float | None, str | None]:
    """Parse a single messy numeric value."""
    if is_missing_like(value):
        return None, None

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None, None
        return float(value), None

    text = str(value)
    text = HIDDEN_SPACE_PATTERN.sub("", text).strip()
    if not text:
        return None, None

    qualifier = None
    qualifier_match = QUALIFIER_PATTERN.match(text)
    if qualifier_match:
        qualifier = qualifier_match.group(1)
        text = text[qualifier_match.end() :].strip()

    number_match = NUMBER_PATTERN.search(text)
    if not number_match:
        return None, qualifier

    numeric_text = normalize_number_text(number_match.group(0))
    try:
        return float(numeric_text), qualifier
    except ValueError:
        return None, qualifier


def normalize_number_text(number_text: str) -> str:
    """Normalize decimal commas and thousands separators."""
    text = number_text.replace(" ", "")

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        if text.count(",") == 1:
            text = text.replace(",", ".")
        else:
            parts = text.split(",")
            text = "".join(parts[:-1]) + "." + parts[-1]

    return text


def is_missing_like(value: Any) -> bool:
    """Return whether a value should be treated as missing."""
    if pd.isna(value):
        return True
    text = str(value).strip().lower()
    text = HIDDEN_SPACE_PATTERN.sub("", text)
    return text in MISSING_TOKENS


def example_values(series: pd.Series, max_examples: int = 4) -> str:
    """Return compact examples for user-facing parse logs."""
    values = (
        series[~series.map(is_missing_like)]
        .astype(str)
        .drop_duplicates()
        .head(max_examples)
        .tolist()
    )
    return ", ".join(values)
