"""Readiness scoring before the app runs statistical analysis."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from analysis.aggregation import count_duplicate_keys
from analysis.normalization import normalize_batch_id_series, parse_numeric_series


@dataclass(frozen=True)
class DataReadinessResult:
    """User-facing assessment of whether a dataset is ready to merge/analyze."""

    score: int
    blockers: list[str]
    warnings: list[str]
    info: list[str]
    details: dict[str, Any]


def calculate_data_readiness(
    process_dataframe: pd.DataFrame,
    qc_dataframe: pd.DataFrame,
    process_batch_id_column: str,
    qc_batch_id_column: str,
    outcome_columns: list[str],
    process_duplicate_strategy: str,
    qc_duplicate_strategy: str,
    process_numeric_parse_report: pd.DataFrame | None = None,
    qc_numeric_parse_report: pd.DataFrame | None = None,
) -> DataReadinessResult:
    """Calculate blockers, warnings, and a simple readiness score."""
    blockers: list[str] = []
    warnings: list[str] = []
    info: list[str] = []

    if process_batch_id_column not in process_dataframe.columns:
        blockers.append("The selected process batch ID column is missing.")
    if qc_batch_id_column not in qc_dataframe.columns:
        blockers.append("The selected QC batch ID column is missing.")
    missing_outcomes = [column for column in outcome_columns if column not in qc_dataframe.columns]
    if missing_outcomes:
        blockers.append("Selected QC outcome columns are missing: " + ", ".join(missing_outcomes))

    details: dict[str, Any] = {
        "process_rows": len(process_dataframe),
        "qc_rows": len(qc_dataframe),
        "selected_outcomes": len(outcome_columns),
    }

    if blockers:
        return build_result(blockers, warnings, info, details)

    process_keys = normalize_batch_id_series(process_dataframe[process_batch_id_column])
    qc_keys = normalize_batch_id_series(qc_dataframe[qc_batch_id_column])
    process_missing = int(process_keys.isna().sum() + (process_keys == "").sum())
    qc_missing = int(qc_keys.isna().sum() + (qc_keys == "").sum())
    if process_missing:
        blockers.append(f"Process batch ID column has {process_missing} missing value(s).")
    if qc_missing:
        blockers.append(f"QC batch ID column has {qc_missing} missing value(s).")

    # The batch ID columns are normalized once and the keys are reused below.
    # Re-normalizing per check ran the same regex work four to six times on
    # every screen refresh.
    process_duplicate_count = count_duplicate_keys(process_keys)
    qc_duplicate_count = count_duplicate_keys(qc_keys)
    details["process_duplicate_batch_ids"] = process_duplicate_count
    details["qc_duplicate_batch_ids"] = qc_duplicate_count

    if process_duplicate_count and process_duplicate_strategy == "error":
        blockers.append(
            f"Process data has {process_duplicate_count} duplicate batch ID(s). Choose a duplicate handling rule."
        )
    elif process_duplicate_count:
        warnings.append(
            f"Process data has {process_duplicate_count} duplicate batch ID(s); they will be resolved during merge."
        )

    if qc_duplicate_count and qc_duplicate_strategy == "error":
        blockers.append(
            f"QC data has {qc_duplicate_count} duplicate batch ID(s). Choose a duplicate handling rule."
        )
    elif qc_duplicate_count:
        warnings.append(
            f"QC data has {qc_duplicate_count} duplicate batch ID(s); they will be resolved during merge."
        )

    process_key_set = set(process_keys.dropna()) - {""}
    qc_key_set = set(qc_keys.dropna()) - {""}
    matched_keys = process_key_set.intersection(qc_key_set)
    details["candidate_process_batches"] = len(process_key_set)
    details["candidate_qc_batches"] = len(qc_key_set)
    details["candidate_matched_batches"] = len(matched_keys)

    if not matched_keys:
        blockers.append(
            "No matching batch IDs were found after normalization. Check batch ID format or selected columns."
        )
    else:
        process_match_fraction = len(matched_keys) / max(len(process_key_set), 1)
        qc_match_fraction = len(matched_keys) / max(len(qc_key_set), 1)
        details["process_match_fraction"] = round(process_match_fraction, 3)
        details["qc_match_fraction"] = round(qc_match_fraction, 3)
        if min(process_match_fraction, qc_match_fraction) < 0.70:
            warnings.append(
                f"Only {len(matched_keys)} batch ID(s) overlap after normalization. Review unmatched IDs before modeling."
            )
        else:
            info.append(f"{len(matched_keys)} candidate batch ID(s) overlap after normalization.")

    for outcome_column in outcome_columns:
        numeric_count = count_parseable_numeric_values(qc_dataframe[outcome_column])
        missing_fraction = 1.0 - (numeric_count / max(len(qc_dataframe), 1))
        if numeric_count == 0:
            blockers.append(f"`{outcome_column}` has no numeric values after parsing.")
        elif missing_fraction > 0.20:
            warnings.append(
                f"`{outcome_column}` has {missing_fraction:.0%} missing or non-numeric values."
            )

    add_parse_report_notes("Process", process_numeric_parse_report, warnings, info)
    add_parse_report_notes("QC", qc_numeric_parse_report, warnings, info)

    unnamed_columns = [
        column
        for column in [*process_dataframe.columns, *qc_dataframe.columns]
        if str(column).lower().startswith("unnamed_column")
    ]
    if unnamed_columns:
        warnings.append(
            "Some blank/unnamed columns were found and renamed. Review whether they contain useful data."
        )

    return build_result(blockers, warnings, info, details)


def count_parseable_numeric_values(series: pd.Series) -> int:
    """Count values the app's own numeric parser can read.

    Plain `pd.to_numeric` rejects "81,2" and "18.5 %", so a decimal-comma QC
    export raised a hard readiness blocker for outcomes the app parses fine.
    """
    if pd.api.types.is_numeric_dtype(series):
        return int(series.notna().sum())

    parsed_values, _ = parse_numeric_series(series)
    return int(parsed_values.notna().sum())


def add_parse_report_notes(
    label: str,
    parse_report: pd.DataFrame | None,
    warnings: list[str],
    info: list[str],
) -> None:
    """Add user-facing notes from numeric parsing."""
    if parse_report is None or parse_report.empty:
        return

    converted_columns = parse_report["column"].tolist()
    info.append(
        f"{label}: converted {len(converted_columns)} numeric-like text column(s), including "
        + ", ".join(converted_columns[:4])
        + "."
    )
    qualifier_rows = parse_report[parse_report["qualifier_count"] > 0]
    if not qualifier_rows.empty:
        warnings.append(
            f"{label}: {int(qualifier_rows['qualifier_count'].sum())} value(s) had qualifiers such as < or >. "
            "Treat exact numeric interpretation cautiously."
        )


def build_result(
    blockers: list[str],
    warnings: list[str],
    info: list[str],
    details: dict[str, Any],
) -> DataReadinessResult:
    """Build a scored readiness result."""
    score = 100
    score -= 35 * len(blockers)
    score -= 8 * len(warnings)
    score -= min(5, len(info))
    score = max(0, min(100, score))
    return DataReadinessResult(
        score=score,
        blockers=blockers,
        warnings=warnings,
        info=info,
        details=details,
    )
