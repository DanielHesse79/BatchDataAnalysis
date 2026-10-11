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
# A whole cell that is a calendar date, optionally with a time. Read as a number
# it became its year (2024-01-15 -> 2024) or month (01/15/2024 -> 1), reported
# as a clean parse.
DATE_LIKE_PATTERN = re.compile(
    r"^(?:\d{4}[-/.]\d{1,2}[-/.]\d{1,2}|\d{1,2}[-/.]\d{1,2}[-/.]\d{2,4})"
    r"(?:[ T]\d{1,2}:\d{2}(?::\d{2}(?:\.\d+)?)?)?(?:\s*(?:Z|[+-]\d{2}:?\d{2}))?$"
)
# h:mm or h:mm:ss, as a time of day or a duration. Read as a number it became
# its hour (08:30 -> 8).
CLOCK_PATTERN = re.compile(
    r"^(\d{1,3}):([0-5]\d)(?::([0-5]\d(?:\.\d+)?))?(?:\s*([AaPp])\.?[Mm]\.?)?$"
)
# Two numbers joined by a dash or "to": a range, which has no single value.
# Taking the first number reported 6.8-7.2 as 6.8.
RANGE_PATTERN = re.compile(
    r"^[-+]?\d+(?:[.,]\d+)?\s*(?:-|\u2013|\u2014|to\b|\.\.)\s*[-+]?\d",
    re.IGNORECASE,
)
# 5 x 10^6, 5x10**6, 5 \u00d7 10^6 - the way cell counts and titers are often typed.
POWER_OF_TEN_PATTERN = re.compile(
    r"^([-+]?\d+(?:[.,]\d+)?)\s*[xX\u00d7*\u00b7]\s*10\s*(?:\^|\*\*)\s*([-+]?\d+)"
)
SUPERSCRIPT_EXPONENT_PATTERN = re.compile("10([\u207a\u207b]?[\u2070\u00b9\u00b2\u00b3\u2074-\u2079]+)")
SUPERSCRIPT_DIGITS = str.maketrans(
    "\u2070\u00b9\u00b2\u00b3\u2074\u2075\u2076\u2077\u2078\u2079\u207a\u207b", "0123456789+-"
)
DIGITS_COLON_DIGITS_PATTERN = re.compile(r"\d\s*:\s*\d")
DATE_OR_CLOCK_COLUMN_FRACTION = 0.80
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
    notes: str = ""


@dataclass(frozen=True)
class ColumnNumberFormat:
    """How the numbers in one column are written, decided from the whole column.

    A single value cannot say whether "1,250" is 1.25 or 1250; the rest of the
    column usually can.
    """

    comma_style: str = "decimal"  # "decimal" or "thousands"
    comma_ambiguous: bool = False
    clock_as_hours: bool = False
    range_count: int = 0
    range_example: str = ""


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

            number_format = infer_column_number_format(series)
            parsed_values, qualifiers = parse_numeric_series(series, number_format)
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
                        notes=describe_number_format(number_format),
                    )
                )
                warnings.extend(number_format_warnings(column_name, number_format))

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
    """Return whether a column is a plausible numeric-like text column.

    Dates are never numbers here, whatever the column is called: only names
    containing "date" used to be protected, so `inoculation_timestamp` was
    turned into its year with a parsed fraction of 1.0.
    """
    if pd.api.types.is_numeric_dtype(series) or pd.api.types.is_bool_dtype(series):
        return False
    if pd.api.types.is_datetime64_any_dtype(series) or pd.api.types.is_timedelta64_dtype(series):
        return False

    normalized_column_name = str(column_name).lower().replace(" ", "_")
    column_tokens = set(re.split(r"[^a-z0-9]+", normalized_column_name))
    if column_tokens.intersection(IDENTIFIER_COLUMN_TOKENS):
        return False

    non_missing_values = series[~series.map(is_missing_like)].astype(str)
    if non_missing_values.empty:
        return False

    sample = non_missing_values.head(80)
    if looks_like_dates(sample):
        return False

    numeric_looking_count = int(sample.map(lambda value: NUMBER_PATTERN.search(value) is not None).sum())
    return numeric_looking_count / len(sample) >= 0.50


def looks_like_dates(values: pd.Series) -> bool:
    """Return whether most values are whole calendar dates or timestamps."""
    texts = [clean_cell_text(value) for value in values]
    texts = [text for text in texts if text]
    if not texts:
        return False
    date_count = sum(bool(DATE_LIKE_PATTERN.match(text)) for text in texts)
    return date_count / len(texts) >= DATE_OR_CLOCK_COLUMN_FRACTION


def clean_cell_text(value: Any) -> str:
    """Return a cell as stripped text without hidden spaces."""
    return HIDDEN_SPACE_PATTERN.sub("", str(value)).strip()


def infer_column_number_format(series: pd.Series) -> ColumnNumberFormat:
    """Decide how a column writes its numbers before parsing any one of them.

    The comma is the hard part. "95,5" can only be a decimal comma and
    "1,234.5" only a thousands separator, but "1,250" can be either. When the
    column itself holds no unambiguous example, the values written without a
    comma decide: in ["1,250", "980", "2,100"] they are hundreds to thousands,
    so the commas are thousands separators; in ["1,523", "1.487"] they are near
    1, so the comma is a decimal comma. A column that gives no hint at all is
    read with decimal commas and reported as ambiguous.
    """
    texts = [
        clean_cell_text(value)
        for value in series
        if not is_missing_like(value) and not isinstance(value, (int, float, np.number))
    ]

    clock_count = sum(bool(CLOCK_PATTERN.match(text)) for text in texts)
    clock_as_hours = bool(texts) and clock_count / len(texts) >= DATE_OR_CLOCK_COLUMN_FRACTION

    range_texts = [text for text in texts if RANGE_PATTERN.match(strip_qualifier(text)[0])]

    thousands_evidence = 0
    decimal_evidence = 0
    ambiguous_tokens: list[tuple[str, str]] = []
    reference_magnitudes: list[float] = []
    for value in series:
        if is_missing_like(value):
            continue
        if isinstance(value, (int, float, np.number)) and not isinstance(value, bool):
            if math.isfinite(float(value)) and float(value) != 0:
                reference_magnitudes.append(abs(float(value)))
            continue

        text, _ = strip_qualifier(clean_cell_text(value))
        if RANGE_PATTERN.match(text) or CLOCK_PATTERN.match(text):
            continue
        match = NUMBER_PATTERN.search(text)
        if not match:
            continue
        token = match.group(0).replace(" ", "").lstrip("+-")
        if "," in token and "." in token:
            if token.rfind(",") < token.rfind("."):
                thousands_evidence += 1
            else:
                decimal_evidence += 1
        elif token.count(",") > 1:
            thousands_evidence += 1
        elif token.count(",") == 1:
            whole, fraction = token.split(",")
            if len(fraction) != 3 or whole in ("", "0"):
                decimal_evidence += 1
            else:
                ambiguous_tokens.append((whole, fraction))
        else:
            try:
                magnitude = abs(float(token))
            except ValueError:
                continue
            if magnitude:
                reference_magnitudes.append(magnitude)

    comma_style = "decimal"
    comma_ambiguous = False
    if ambiguous_tokens:
        if thousands_evidence and not decimal_evidence:
            comma_style = "thousands"
        elif decimal_evidence and not thousands_evidence:
            comma_style = "decimal"
        elif reference_magnitudes:
            reference = math.log10(float(np.median(reference_magnitudes)))
            as_decimal = [float(f"{whole}.{fraction}") for whole, fraction in ambiguous_tokens]
            as_thousands = [float(f"{whole}{fraction}") for whole, fraction in ambiguous_tokens]
            decimal_distance = abs(math.log10(float(np.median(as_decimal))) - reference)
            thousands_distance = abs(math.log10(float(np.median(as_thousands))) - reference)
            comma_style = "thousands" if thousands_distance < decimal_distance else "decimal"
            comma_ambiguous = abs(thousands_distance - decimal_distance) < 1.0
        else:
            comma_ambiguous = True

    return ColumnNumberFormat(
        comma_style=comma_style,
        comma_ambiguous=comma_ambiguous,
        clock_as_hours=clock_as_hours,
        range_count=len(range_texts),
        range_example=range_texts[0] if range_texts else "",
    )


def describe_number_format(number_format: ColumnNumberFormat) -> str:
    """Say in the parse log how a column's numbers were read."""
    notes = []
    if number_format.clock_as_hours:
        notes.append("h:mm read as hours")
    if number_format.comma_style == "thousands":
        notes.append("comma read as thousands separator")
    if number_format.comma_ambiguous:
        notes.append(f"comma read as {number_format.comma_style}, but ambiguous")
    if number_format.range_count:
        notes.append(f"{number_format.range_count} range value(s) left blank")
    return "; ".join(notes)


def number_format_warnings(column_name: str, number_format: ColumnNumberFormat) -> list[str]:
    """Warnings for readings the user should confirm against the source file."""
    warnings = []
    if number_format.comma_ambiguous:
        reading = "decimal commas (1,250 -> 1.25)" if number_format.comma_style == "decimal" else (
            "thousands separators (1,250 -> 1250)"
        )
        warnings.append(
            f"{column_name}: values such as 1,250 could be a decimal or a thousands separator, "
            f"and the rest of the column does not settle it. They were read as {reading}; "
            "check the source file."
        )
    if number_format.range_count:
        warnings.append(
            f"{column_name}: {number_format.range_count} value(s) are ranges such as "
            f"'{number_format.range_example}' and have no single value, so they were left blank "
            "rather than reduced to their first number."
        )
    return warnings


def parse_numeric_series(
    series: pd.Series,
    number_format: ColumnNumberFormat | None = None,
) -> tuple[pd.Series, pd.Series]:
    """Parse a Series containing numbers, units, and qualifiers.

    The column's number format is inferred from the whole Series unless given.
    """
    if number_format is None:
        number_format = infer_column_number_format(series)

    parsed_values = []
    qualifiers = []
    for value in series:
        parsed_value, qualifier = parse_numeric_value(value, number_format)
        parsed_values.append(parsed_value)
        qualifiers.append(qualifier)

    return (
        pd.Series(parsed_values, index=series.index, dtype="float64"),
        pd.Series(qualifiers, index=series.index, dtype="object"),
    )


def strip_qualifier(text: str) -> tuple[str, str | None]:
    """Split a leading qualifier such as < or approx. from the rest of a cell."""
    qualifier_match = QUALIFIER_PATTERN.match(text)
    if not qualifier_match:
        return text, None
    return text[qualifier_match.end() :].strip(), qualifier_match.group(1)


def parse_numeric_value(
    value: Any,
    number_format: ColumnNumberFormat | None = None,
) -> tuple[float | None, str | None]:
    """Parse a single messy numeric value.

    Values with no single number - a range, a ratio such as 1:100, a date - are
    returned as None rather than reduced to their first number.
    """
    number_format = number_format or ColumnNumberFormat()
    if is_missing_like(value):
        return None, None

    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool):
        if math.isnan(float(value)):
            return None, None
        return float(value), None

    text = clean_cell_text(value)
    if not text:
        return None, None

    text, qualifier = strip_qualifier(text)

    plain_value = parse_plain_number(text)
    if plain_value is not None:
        return plain_value, qualifier

    if DATE_LIKE_PATTERN.match(text) or RANGE_PATTERN.match(text):
        return None, qualifier

    clock_match = CLOCK_PATTERN.match(text)
    if clock_match:
        if not number_format.clock_as_hours:
            return None, qualifier
        return clock_to_hours(clock_match), qualifier
    if DIGITS_COLON_DIGITS_PATTERN.search(text):
        return None, qualifier

    power_value = parse_power_of_ten(text, number_format)
    if power_value is not None:
        return power_value, qualifier

    number_match = NUMBER_PATTERN.search(text)
    if not number_match:
        return None, qualifier

    numeric_text = normalize_number_text(number_match.group(0), number_format.comma_style)
    try:
        parsed_value = float(numeric_text)
    except ValueError:
        return None, qualifier

    return (parsed_value if math.isfinite(parsed_value) else None), qualifier


def clock_to_hours(clock_match: re.Match) -> float:
    """Convert an h:mm[:ss] [am|pm] match to hours."""
    hours = int(clock_match.group(1))
    minutes = int(clock_match.group(2))
    seconds = float(clock_match.group(3) or 0)
    meridiem = (clock_match.group(4) or "").lower()
    if meridiem == "p" and hours < 12:
        hours += 12
    elif meridiem == "a" and hours == 12:
        hours = 0
    return hours + minutes / 60.0 + seconds / 3600.0


def parse_power_of_ten(text: str, number_format: ColumnNumberFormat) -> float | None:
    """Parse 5 x 10^6 and 5 x 10⁶; the first number alone is a 10^n error."""
    text = SUPERSCRIPT_EXPONENT_PATTERN.sub(
        lambda match: "10^" + match.group(1).translate(SUPERSCRIPT_DIGITS), text
    )
    match = POWER_OF_TEN_PATTERN.match(text)
    if not match:
        return None
    mantissa = float(normalize_number_text(match.group(1), number_format.comma_style))
    value = mantissa * 10.0 ** int(match.group(2))
    return value if math.isfinite(value) else None


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


def normalize_number_text(number_text: str, comma_style: str = "decimal") -> str:
    """Normalize decimal commas and thousands separators.

    ``comma_style`` settles the one case the text cannot: a single comma
    followed by exactly three digits. Everything else is unambiguous.
    """
    text = number_text.replace(" ", "")

    if "," in text and "." in text:
        if text.rfind(",") > text.rfind("."):
            text = text.replace(".", "").replace(",", ".")
        else:
            text = text.replace(",", "")
    elif "," in text:
        parts = text.split(",")
        if len(parts) > 2 and all(len(part) == 3 for part in parts[1:]):
            # 1,234,567: only a thousands separator repeats. This used to keep
            # the last group as decimals and read 1234.567.
            text = "".join(parts)
        elif len(parts) == 2:
            if comma_style == "thousands" and len(parts[1]) == 3:
                text = "".join(parts)
            else:
                text = ".".join(parts)
        else:
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
