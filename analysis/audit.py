"""Pre-flight data audit checks.

These checks are deliberately conservative. They do not block analysis; they
surface ways a historical batch dataset can mislead a user before modeling.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any
import warnings

import numpy as np
import pandas as pd


HIGH_FEATURE_TARGET_CORRELATION_THRESHOLD = 0.95
MULTICOLLINEARITY_CORRELATION_THRESHOLD = 0.85
HIGH_CARDINALITY_RATIO_THRESHOLD = 0.20
HIGH_CARDINALITY_MINIMUM_UNIQUE_VALUES = 20

LEAKAGE_KEYWORDS = [
    "qc",
    "cqa",
    "result",
    "release",
    "pass",
    "fail",
    "spec",
    "reject",
    "complaint",
    "stability",
    "yield",
    "purity",
    "hcp",
    "aggregate",
    "endotoxin",
    "deviation",
    "investigation",
]

DATE_DRIFT_KEYWORDS = [
    "date",
    "time",
    "campaign",
    "sequence",
    "order",
    "run",
    "batch_number",
]

CONFOUNDING_KEYWORDS = [
    "lot",
    "supplier",
    "operator",
    "shift",
    "reactor",
    "bioreactor",
    "equipment",
    "line",
    "site",
    "room",
    "campaign",
]

OUTCOME_TOKEN_STOPWORDS = {
    "percent",
    "ppm",
    "ppb",
    "g",
    "l",
    "ml",
    "h",
    "hr",
    "hrs",
    "hour",
    "hours",
    "c",
    "kg",
    "mg",
    "mean",
    "avg",
    "average",
    "result",
    "results",
    "value",
}


@dataclass(frozen=True)
class AuditResult:
    """Pre-flight audit output used by the app and later report export."""

    summary: dict[str, Any]
    warning_count: int
    sample_size_guidance: str
    leakage_name_warnings: pd.DataFrame
    feature_target_correlations: pd.DataFrame
    date_or_drift_columns: pd.DataFrame
    high_cardinality_categoricals: pd.DataFrame
    confounding_categoricals: pd.DataFrame
    row_missingness: pd.DataFrame
    outlier_flags: pd.DataFrame
    multicollinearity_pairs: pd.DataFrame
    warnings: list[str]


def run_preflight_audit(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_columns: list[str],
    variable_types: pd.DataFrame,
) -> AuditResult:
    """Run all conservative pre-flight checks for a merged dataset."""
    matched_batch_count = len(dataframe)
    sample_size_guidance = get_sample_size_guidance(matched_batch_count)

    leakage_name_warnings = detect_leakage_name_warnings(process_columns, outcome_columns)
    feature_target_correlations = detect_high_feature_target_correlations(
        dataframe,
        process_columns,
        outcome_columns,
    )
    date_or_drift_columns = detect_date_or_drift_columns(dataframe, process_columns)
    high_cardinality_categoricals = detect_high_cardinality_categoricals(
        dataframe,
        variable_types,
    )
    confounding_categoricals = detect_confounding_categoricals(
        dataframe,
        variable_types,
    )
    row_missingness = summarize_row_missingness(dataframe, process_columns, outcome_columns)
    outlier_flags = detect_numeric_outlier_flags(dataframe, process_columns, outcome_columns)
    multicollinearity_pairs = detect_multicollinearity_pairs(dataframe, process_columns)

    warnings = build_audit_warnings(
        matched_batch_count=matched_batch_count,
        leakage_name_warnings=leakage_name_warnings,
        feature_target_correlations=feature_target_correlations,
        date_or_drift_columns=date_or_drift_columns,
        high_cardinality_categoricals=high_cardinality_categoricals,
        confounding_categoricals=confounding_categoricals,
        row_missingness=row_missingness,
        outlier_flags=outlier_flags,
        multicollinearity_pairs=multicollinearity_pairs,
    )

    warning_count = len(warnings)
    summary = {
        "matched_batch_count": matched_batch_count,
        "sample_size_guidance": sample_size_guidance,
        "leakage_name_warning_count": len(leakage_name_warnings),
        "high_feature_target_correlation_count": len(feature_target_correlations),
        "date_or_drift_column_count": len(date_or_drift_columns),
        "high_cardinality_categorical_count": len(high_cardinality_categoricals),
        "confounding_categorical_count": len(confounding_categoricals),
        "rows_with_any_missing_selected_field": get_row_missing_value(
            row_missingness,
            "rows_with_any_missing_selected_field",
        ),
        "columns_with_outliers": int(outlier_flags["column"].nunique())
        if not outlier_flags.empty
        else 0,
        "highly_correlated_numeric_pairs": len(multicollinearity_pairs),
    }

    return AuditResult(
        summary=summary,
        warning_count=warning_count,
        sample_size_guidance=sample_size_guidance,
        leakage_name_warnings=leakage_name_warnings,
        feature_target_correlations=feature_target_correlations,
        date_or_drift_columns=date_or_drift_columns,
        high_cardinality_categoricals=high_cardinality_categoricals,
        confounding_categoricals=confounding_categoricals,
        row_missingness=row_missingness,
        outlier_flags=outlier_flags,
        multicollinearity_pairs=multicollinearity_pairs,
        warnings=warnings,
    )


def get_sample_size_guidance(batch_count: int) -> str:
    """Return practical modeling guidance based on batch count."""
    if batch_count < 10:
        return (
            "Very small dataset: use plots and descriptive checks only. Predictive modeling is likely unstable."
        )
    if batch_count < 30:
        return (
            "Small dataset: PCA exploration and cautious PLS/Random Forest checks are reasonable; avoid advanced ML."
        )
    if batch_count < 80:
        return (
            "Moderate SME dataset: PCA, PLS, and Random Forest are appropriate with conservative validation."
        )
    return (
        "Larger batch dataset: PCA, PLS, Random Forest, and later CatBoost/SHAP-style advanced checks are reasonable."
    )


def detect_leakage_name_warnings(
    process_columns: list[str],
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Flag process columns whose names look like post-hoc QC or release information."""
    outcome_tokens = build_outcome_tokens(outcome_columns)
    rows = []

    for column_name in process_columns:
        normalized_name = normalize_name(column_name)
        matched_keywords = [
            keyword
            for keyword in LEAKAGE_KEYWORDS
            if keyword in normalized_name
        ]
        matched_outcome_tokens = [
            token
            for token in outcome_tokens
            if len(token) >= 4 and token in normalized_name
        ]

        if matched_keywords or matched_outcome_tokens:
            rows.append(
                {
                    "column": column_name,
                    "matched_terms": ", ".join(sorted(set(matched_keywords + matched_outcome_tokens))),
                    "why_it_matters": (
                        "This name may represent QC results, release decisions, deviations, or other post-hoc information."
                    ),
                }
            )

    return dataframe_from_rows(
        rows,
        columns=["column", "matched_terms", "why_it_matters"],
    )


def detect_high_feature_target_correlations(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Flag numeric process columns that are almost identical to an outcome."""
    rows = []

    for process_column in process_columns:
        process_values = pd.to_numeric(dataframe[process_column], errors="coerce")
        if process_values.nunique(dropna=True) < 3:
            continue

        for outcome_column in outcome_columns:
            outcome_values = pd.to_numeric(dataframe[outcome_column], errors="coerce")
            usable_values = pd.DataFrame(
                {"process": process_values, "outcome": outcome_values}
            ).dropna()

            if len(usable_values) < 8 or usable_values["outcome"].nunique() < 3:
                continue

            correlation = usable_values["process"].corr(usable_values["outcome"])
            if pd.notna(correlation) and abs(correlation) >= HIGH_FEATURE_TARGET_CORRELATION_THRESHOLD:
                rows.append(
                    {
                        "process_column": process_column,
                        "outcome": outcome_column,
                        "correlation": round(float(correlation), 4),
                        "absolute_correlation": round(abs(float(correlation)), 4),
                        "why_it_matters": (
                            "A near-perfect feature-target correlation can indicate leakage or duplicate information."
                        ),
                    }
                )

    return dataframe_from_rows(
        rows,
        columns=[
            "process_column",
            "outcome",
            "correlation",
            "absolute_correlation",
            "why_it_matters",
        ],
    ).sort_values("absolute_correlation", ascending=False, ignore_index=True)


def detect_date_or_drift_columns(
    dataframe: pd.DataFrame,
    process_columns: list[str],
) -> pd.DataFrame:
    """Flag columns that may encode calendar time, sequence, campaign, or drift."""
    rows = []

    for column_name in process_columns:
        normalized_name = normalize_name(column_name)
        matched_terms = [term for term in DATE_DRIFT_KEYWORDS if term in normalized_name]
        parseable_date_fraction = estimate_parseable_date_fraction(dataframe[column_name])
        looks_date_like = parseable_date_fraction >= 0.80

        if matched_terms or looks_date_like:
            rows.append(
                {
                    "column": column_name,
                    "matched_terms": ", ".join(matched_terms) if matched_terms else "date-like values",
                    "parseable_date_fraction": round(parseable_date_fraction, 3),
                    "why_it_matters": (
                        "Date, sequence, and campaign columns can absorb process drift or hidden equipment/raw-material changes."
                    ),
                }
            )

    return dataframe_from_rows(
        rows,
        columns=["column", "matched_terms", "parseable_date_fraction", "why_it_matters"],
    )


def detect_high_cardinality_categoricals(
    dataframe: pd.DataFrame,
    variable_types: pd.DataFrame,
) -> pd.DataFrame:
    """Flag categorical variables with many distinct levels."""
    categorical_columns = variable_types[
        variable_types["type"].isin(["categorical", "binary"])
    ]["column"].tolist()
    rows = []
    row_count = max(len(dataframe), 1)

    for column_name in categorical_columns:
        unique_count = int(dataframe[column_name].nunique(dropna=True))
        unique_ratio = unique_count / row_count
        is_high_cardinality = (
            unique_count >= HIGH_CARDINALITY_MINIMUM_UNIQUE_VALUES
            or unique_ratio >= HIGH_CARDINALITY_RATIO_THRESHOLD
        )

        if is_high_cardinality:
            rows.append(
                {
                    "column": column_name,
                    "unique_count": unique_count,
                    "unique_ratio": round(unique_ratio, 3),
                    "why_it_matters": (
                        "High-cardinality categoricals can overfit and may need grouping, CatBoost-style handling, or validation by time/campaign."
                    ),
                }
            )

    return dataframe_from_rows(
        rows,
        columns=["column", "unique_count", "unique_ratio", "why_it_matters"],
    ).sort_values("unique_count", ascending=False, ignore_index=True)


def detect_confounding_categoricals(
    dataframe: pd.DataFrame,
    variable_types: pd.DataFrame,
) -> pd.DataFrame:
    """Flag lot/operator/equipment-style variables that can be confounded with time."""
    categorical_columns = variable_types[
        variable_types["type"].isin(["categorical", "binary"])
    ]["column"].tolist()
    rows = []

    for column_name in categorical_columns:
        normalized_name = normalize_name(column_name)
        matched_terms = [term for term in CONFOUNDING_KEYWORDS if term in normalized_name]
        if not matched_terms:
            continue

        value_counts = dataframe[column_name].value_counts(dropna=True)
        top_level = value_counts.index[0] if not value_counts.empty else ""
        top_level_percent = (
            float(value_counts.iloc[0] / max(value_counts.sum(), 1) * 100.0)
            if not value_counts.empty
            else 0.0
        )
        rows.append(
            {
                "column": column_name,
                "matched_terms": ", ".join(matched_terms),
                "unique_count": int(dataframe[column_name].nunique(dropna=True)),
                "most_common_level": str(top_level),
                "most_common_level_percent": round(top_level_percent, 2),
                "why_it_matters": (
                    "Lot, supplier, operator, shift, room, and equipment effects are often partially confounded with date or campaign."
                ),
            }
        )

    return dataframe_from_rows(
        rows,
        columns=[
            "column",
            "matched_terms",
            "unique_count",
            "most_common_level",
            "most_common_level_percent",
            "why_it_matters",
        ],
    ).sort_values("unique_count", ascending=False, ignore_index=True)


def summarize_row_missingness(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Summarize row-level missingness without dropping rows."""
    selected_columns = process_columns + outcome_columns
    selected_frame = dataframe[selected_columns]

    rows_with_any_missing = int(selected_frame.isna().any(axis=1).sum())
    rows_with_missing_outcome = int(dataframe[outcome_columns].isna().any(axis=1).sum())
    rows_with_missing_process = int(dataframe[process_columns].isna().any(axis=1).sum())
    row_count = max(len(dataframe), 1)

    rows = [
        {
            "metric": "rows_with_any_missing_selected_field",
            "count": rows_with_any_missing,
            "percent": round(rows_with_any_missing / row_count * 100.0, 2),
            "why_it_matters": (
                "Complete-case analysis would lose these rows. This app imputes process variables and only drops missing outcomes for modeling."
            ),
        },
        {
            "metric": "rows_with_missing_process_field",
            "count": rows_with_missing_process,
            "percent": round(rows_with_missing_process / row_count * 100.0, 2),
            "why_it_matters": "Missing process fields are imputed before modeling.",
        },
        {
            "metric": "rows_with_missing_outcome_field",
            "count": rows_with_missing_outcome,
            "percent": round(rows_with_missing_outcome / row_count * 100.0, 2),
            "why_it_matters": "Rows with missing selected outcomes cannot train that outcome model.",
        },
    ]
    return pd.DataFrame(rows)


def detect_numeric_outlier_flags(
    dataframe: pd.DataFrame,
    process_columns: list[str],
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Flag numeric columns with IQR outliers. Outliers are never removed automatically."""
    rows = []

    for column_name in process_columns + outcome_columns:
        numeric_values = pd.to_numeric(dataframe[column_name], errors="coerce").dropna()
        if numeric_values.nunique() < 4:
            continue

        first_quartile = numeric_values.quantile(0.25)
        third_quartile = numeric_values.quantile(0.75)
        interquartile_range = third_quartile - first_quartile
        if interquartile_range <= 0:
            continue

        lower_limit = first_quartile - 1.5 * interquartile_range
        upper_limit = third_quartile + 1.5 * interquartile_range
        outlier_count = int(((numeric_values < lower_limit) | (numeric_values > upper_limit)).sum())

        if outlier_count > 0:
            rows.append(
                {
                    "column": column_name,
                    "outlier_count": outlier_count,
                    "outlier_percent": round(outlier_count / max(len(numeric_values), 1) * 100.0, 2),
                    "lower_flag_limit": round_numeric(lower_limit),
                    "upper_flag_limit": round_numeric(upper_limit),
                    "why_it_matters": (
                        "Outliers may be the most informative batches. Review them; do not drop automatically."
                    ),
                }
            )

    return dataframe_from_rows(
        rows,
        columns=[
            "column",
            "outlier_count",
            "outlier_percent",
            "lower_flag_limit",
            "upper_flag_limit",
            "why_it_matters",
        ],
    ).sort_values("outlier_count", ascending=False, ignore_index=True)


def detect_multicollinearity_pairs(
    dataframe: pd.DataFrame,
    process_columns: list[str],
) -> pd.DataFrame:
    """Flag highly correlated numeric process variable pairs."""
    numeric_process_columns = [
        column_name
        for column_name in process_columns
        if pd.api.types.is_numeric_dtype(dataframe[column_name])
        and dataframe[column_name].nunique(dropna=True) >= 3
    ]
    rows = []

    for first_column, second_column in combinations(numeric_process_columns, 2):
        usable_values = dataframe[[first_column, second_column]].apply(
            pd.to_numeric,
            errors="coerce",
        ).dropna()
        if len(usable_values) < 8:
            continue

        correlation = usable_values[first_column].corr(usable_values[second_column])
        if pd.notna(correlation) and abs(correlation) >= MULTICOLLINEARITY_CORRELATION_THRESHOLD:
            rows.append(
                {
                    "first_column": first_column,
                    "second_column": second_column,
                    "correlation": round(float(correlation), 4),
                    "absolute_correlation": round(abs(float(correlation)), 4),
                    "why_it_matters": (
                        "Highly correlated predictors can split importance across variables; PLS handles this better than ordinary regression."
                    ),
                }
            )

    return dataframe_from_rows(
        rows,
        columns=[
            "first_column",
            "second_column",
            "correlation",
            "absolute_correlation",
            "why_it_matters",
        ],
    ).sort_values("absolute_correlation", ascending=False, ignore_index=True)


def build_audit_warnings(
    matched_batch_count: int,
    leakage_name_warnings: pd.DataFrame,
    feature_target_correlations: pd.DataFrame,
    date_or_drift_columns: pd.DataFrame,
    high_cardinality_categoricals: pd.DataFrame,
    confounding_categoricals: pd.DataFrame,
    row_missingness: pd.DataFrame,
    outlier_flags: pd.DataFrame,
    multicollinearity_pairs: pd.DataFrame,
) -> list[str]:
    """Build concise user-facing warnings."""
    warnings = []

    if matched_batch_count < 30:
        warnings.append("Fewer than 30 matched batches: treat modeling results as exploratory.")
    elif matched_batch_count < 80:
        warnings.append("Fewer than 80 matched batches: advanced ML may overfit; prefer PCA, PLS, and conservative RF.")

    if not leakage_name_warnings.empty:
        warnings.append(
            f"{len(leakage_name_warnings):,} process column(s) have names that may indicate leakage or post-hoc information."
        )

    if not feature_target_correlations.empty:
        warnings.append(
            f"{len(feature_target_correlations):,} feature/outcome pair(s) have near-perfect correlation and may indicate leakage."
        )

    if not date_or_drift_columns.empty:
        warnings.append(
            f"{len(date_or_drift_columns):,} column(s) may encode date, sequence, campaign, or process drift."
        )

    if not high_cardinality_categoricals.empty:
        warnings.append(
            f"{len(high_cardinality_categoricals):,} categorical column(s) have high cardinality and may overfit."
        )

    if not confounding_categoricals.empty:
        warnings.append(
            f"{len(confounding_categoricals):,} categorical column(s) may be confounded with date, campaign, or operating practice."
        )

    row_missing_percent = get_row_missing_value(
        row_missingness,
        "rows_with_any_missing_selected_field",
        value_column="percent",
    )
    if row_missing_percent and row_missing_percent > 20:
        warnings.append(
            f"{row_missing_percent:.1f}% of rows have at least one missing selected field; avoid silent row dropping."
        )

    if not outlier_flags.empty:
        warnings.append(
            f"{int(outlier_flags['column'].nunique()):,} numeric column(s) contain IQR outliers. Review, but do not auto-remove."
        )

    if not multicollinearity_pairs.empty:
        warnings.append(
            f"{len(multicollinearity_pairs):,} numeric process variable pair(s) are highly correlated."
        )

    return warnings


def build_outcome_tokens(outcome_columns: list[str]) -> set[str]:
    """Create simple tokens from outcome column names."""
    tokens: set[str] = set()
    for outcome_column in outcome_columns:
        tokens.update(
            part
            for part in normalize_name(outcome_column).split("_")
            if part and part not in OUTCOME_TOKEN_STOPWORDS
        )
    return tokens


def estimate_parseable_date_fraction(series: pd.Series) -> float:
    """Estimate whether a column contains date-like values."""
    non_missing_values = series.dropna()
    if non_missing_values.empty:
        return 0.0

    if pd.api.types.is_datetime64_any_dtype(non_missing_values):
        return 1.0

    if pd.api.types.is_numeric_dtype(non_missing_values):
        return 0.0

    sample_values = non_missing_values.astype(str).head(100)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        parsed_values = pd.to_datetime(sample_values, errors="coerce")
    return float(parsed_values.notna().mean())


def get_row_missing_value(
    row_missingness: pd.DataFrame,
    metric_name: str,
    value_column: str = "count",
) -> float:
    """Get a row-missingness metric value."""
    matching_rows = row_missingness[row_missingness["metric"] == metric_name]
    if matching_rows.empty:
        return 0.0
    return float(matching_rows.iloc[0][value_column])


def dataframe_from_rows(rows: list[dict[str, Any]], columns: list[str]) -> pd.DataFrame:
    """Create a stable empty-or-filled DataFrame."""
    if not rows:
        return pd.DataFrame(columns=columns)
    return pd.DataFrame(rows, columns=columns)


def normalize_name(column_name: str) -> str:
    """Normalize a column name for keyword checks."""
    return str(column_name).strip().lower().replace(" ", "_").replace("-", "_")


def round_numeric(value: Any, digits: int = 4) -> float | None:
    """Round a numeric value for display."""
    if pd.isna(value):
        return None
    return round(float(value), digits)
