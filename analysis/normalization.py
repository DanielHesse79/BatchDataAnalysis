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
# Matches plain numbers, grouped/decimal-comma numbers, leading-dot decimals, and
# scientific notation. QC exports (endotoxin, bioburden, titer) routinely use
# E-notation, so the exponent must be part of the match or values are truncated
# to their mantissa.
NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:[\s.,]\d+)*|\.\d+)(?:[eE][-+]?\d+)?")
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
    """Strip, deblank, and make duplicate column names unique.

    The generated suffix must not collide with a name that already exists in the
    file. A header such as ["temp", "temp", "temp__2"] otherwise produces two
    `temp__2` columns, and every later `dataframe[column]` lookup silently
    returns a DataFrame instead of a Series.

    Rename notes are keyed by position, so several columns sharing one original
    name each keep their own entry.
    """
    output_dataframe = dataframe.copy()
    original_names = [str(column_name).strip() for column_name in output_dataframe.columns]
    base_names = [
        clean_column_name(original_name, index)
        for index, original_name in enumerate(original_names)
    ]
    reserved_names = set(base_names)

    used_names: set[str] = set()
    new_columns: list[str] = []
    renamed_columns: dict[str, str] = {}

    for index, (original_name, base_name) in enumerate(zip(original_names, base_names)):
        cleaned_name = base_name
        suffix = 1
        while cleaned_name in used_names or (
            cleaned_name != base_name and cleaned_name in reserved_names
        ):
            suffix += 1
            cleaned_name = f"{base_name}__{suffix}"

        used_names.add(cleaned_name)
        if cleaned_name != original_name:
            renamed_columns[describe_column_position(original_name, index)] = cleaned_name
        new_columns.append(cleaned_name)

    output_dataframe.columns = new_columns
    return output_dataframe, renamed_columns


def clean_column_name(original_name: str, index: int) -> str:
    """Clean one column name without worrying about uniqueness."""
    cleaned_name = HIDDEN_SPACE_PATTERN.sub("", original_name).strip()
    if not cleaned_name or cleaned_name.lower().startswith("unnamed:"):
        return f"unnamed_column_{index + 1}"
    return cleaned_name


def describe_column_position(original_name: str, index: int) -> str:
    """Label a renamed column by position so duplicate names stay distinguishable."""
    label = original_name or "(blank)"
    return f"{label} [column {index + 1}]"


def make_internal_column_name(base_name: str, *taken_columns) -> str:
    """Return a helper column name that no supplied frame already uses.

    Helper columns are added to user data and dropped again afterwards, so a
    real column of the same name would be overwritten and then deleted. Real
    exports do contain names like `__batch_id_key`.
    """
    used_names: set[str] = set()
    for columns in taken_columns:
        used_names.update(str(column_name) for column_name in columns)

    candidate_name = base_name
    suffix = 1
    while candidate_name in used_names:
        suffix += 1
        candidate_name = f"{base_name}_{suffix}"

    return candidate_name


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

    # Excel writes integer-looking IDs as floats with a varying number of zeros
    # ("12.0", "12.00"). Stripping only a single ".0" leaves "12.00" unable to
    # match "12", which shows up as a hard readiness blocker.
    excel_float_match = re.fullmatch(r"(\d+)\.0+", text)
    if excel_float_match:
        text = excel_float_match.group(1)

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

    plain_value = parse_plain_number(text)
    if plain_value is not None:
        return plain_value, qualifier

    number_match = NUMBER_PATTERN.search(text)
    if not number_match:
        return None, qualifier

    numeric_text = normalize_number_text(number_match.group(0))
    try:
        parsed_value = float(numeric_text)
    except ValueError:
        return None, qualifier

    return (parsed_value if math.isfinite(parsed_value) else None), qualifier


def parse_plain_number(text: str) -> float | None:
    """Parse text that is already a valid number, including scientific notation.

    Trying this before the messy-format regex keeps exponents intact; the regex
    path handles decimal commas, thousands separators, and embedded units.
    """
    if not text or "_" in text:
        return None

    try:
        value = float(text)
    except ValueError:
        return None

    return value if math.isfinite(value) else None


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


def is_text_like_series(series: pd.Series) -> bool:
    """Return whether a column can hold missing-like text tokens.

    pandas 3 stores text as a dedicated string dtype, so an object-dtype check
    alone would miss ordinary text columns.
    """
    return not (
        pd.api.types.is_numeric_dtype(series)
        or pd.api.types.is_bool_dtype(series)
        or pd.api.types.is_datetime64_any_dtype(series)
    )


def build_missing_like_mask(series: pd.Series) -> pd.Series:
    """Return a boolean mask of missing values, including text tokens.

    Text columns kept below the numeric parse threshold can carry "n/a", "-",
    or "not tested". Counting only NaN reports those columns as complete.
    """
    if not is_text_like_series(series):
        return series.isna()

    return pd.Series(
        [is_missing_like(value) for value in series],
        index=series.index,
        dtype=bool,
        name=series.name,
    )


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
