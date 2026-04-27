"""Specification and operating-window assessment helpers.

The functions here compare stated process/QC limits with historical batch data.
They are deliberately conservative: the goal is to challenge whether a spec or
window deserves review, not to prove a validated design space.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


SPEC_COLUMNS = [
    "variable",
    "role",
    "target",
    "lower_limit",
    "upper_limit",
    "unit",
    "criticality",
    "notes",
]
VALID_ROLES = {"process", "qc"}
IMPORTANT_DRIVER_THRESHOLD = 0.55
LOW_DRIVER_THRESHOLD = 0.20
EDGE_FRACTION = 0.10


class SpecError(ValueError):
    """Raised when a spec file cannot be used."""


@dataclass(frozen=True)
class SpecAssessmentResult:
    """Container for spec/window assessment outputs."""

    specs: pd.DataFrame
    matched_specs: pd.DataFrame
    unmatched_specs: pd.DataFrame
    variable_summary: pd.DataFrame
    out_of_spec_batches: pd.DataFrame
    outcome_zone_summary: pd.DataFrame
    warnings: list[str]

    @property
    def has_specs(self) -> bool:
        """Return whether any spec rows were supplied."""
        return not self.specs.empty


def build_default_spec_template() -> pd.DataFrame:
    """Return a blank-ish template with synthetic examples."""
    return pd.DataFrame(
        [
            {
                "variable": "feed_rate_day3_mL_h",
                "role": "process",
                "target": 38,
                "lower_limit": 28,
                "upper_limit": 50,
                "unit": "mL/h",
                "criticality": "CPP",
                "notes": "Example process operating window.",
            },
            {
                "variable": "temperature_C",
                "role": "process",
                "target": 36.9,
                "lower_limit": 36.2,
                "upper_limit": 37.8,
                "unit": "C",
                "criticality": "CPP",
                "notes": "Example process operating window.",
            },
            {
                "variable": "duration_hours",
                "role": "process",
                "target": 166,
                "lower_limit": 150,
                "upper_limit": 180,
                "unit": "h",
                "criticality": "CPP",
                "notes": "Example duration window.",
            },
            {
                "variable": "ph_setpoint",
                "role": "process",
                "target": 7.0,
                "lower_limit": 6.8,
                "upper_limit": 7.4,
                "unit": "pH",
                "criticality": "CPP",
                "notes": "Example pH window.",
            },
            {
                "variable": "purity_percent",
                "role": "qc",
                "target": "",
                "lower_limit": 90,
                "upper_limit": "",
                "unit": "%",
                "criticality": "CQA",
                "notes": "Example lower-only QC spec.",
            },
            {
                "variable": "hcp_ppm",
                "role": "qc",
                "target": "",
                "lower_limit": "",
                "upper_limit": 250,
                "unit": "ppm",
                "criticality": "CQA",
                "notes": "Example upper-only QC spec.",
            },
            {
                "variable": "aggregate_percent",
                "role": "qc",
                "target": "",
                "lower_limit": "",
                "upper_limit": 5,
                "unit": "%",
                "criticality": "CQA",
                "notes": "Example upper-only QC spec.",
            },
        ],
        columns=SPEC_COLUMNS,
    )


def normalize_spec_dataframe(spec_dataframe: pd.DataFrame) -> pd.DataFrame:
    """Validate and normalize an uploaded spec table."""
    if spec_dataframe is None or spec_dataframe.empty:
        return pd.DataFrame(columns=SPEC_COLUMNS)

    normalized = spec_dataframe.copy()
    normalized.columns = [str(column_name).strip() for column_name in normalized.columns]

    missing_required = [
        column_name for column_name in ["variable"] if column_name not in normalized.columns
    ]
    if missing_required:
        missing_text = ", ".join(missing_required)
        raise SpecError(f"Spec file is missing required column(s): {missing_text}.")

    for column_name in SPEC_COLUMNS:
        if column_name not in normalized.columns:
            normalized[column_name] = ""

    normalized = normalized[SPEC_COLUMNS].copy()
    normalized["variable"] = normalized["variable"].astype("string").str.strip()
    normalized = normalized[normalized["variable"].notna() & (normalized["variable"] != "")]

    if normalized.empty:
        return pd.DataFrame(columns=SPEC_COLUMNS)

    normalized["role"] = (
        normalized["role"]
        .astype("string")
        .str.strip()
        .str.lower()
        .replace({"cqa": "qc", "outcome": "qc", "quality": "qc", "cpp": "process"})
    )
    normalized["unit"] = normalized["unit"].fillna("").astype(str).str.strip()
    normalized["criticality"] = normalized["criticality"].fillna("").astype(str).str.strip()
    normalized["notes"] = normalized["notes"].fillna("").astype(str).str.strip()

    for numeric_column in ["target", "lower_limit", "upper_limit"]:
        normalized[numeric_column] = pd.to_numeric(normalized[numeric_column], errors="coerce")

    duplicate_variables = normalized["variable"][
        normalized["variable"].duplicated()
    ].drop_duplicates()
    if not duplicate_variables.empty:
        duplicate_text = ", ".join(duplicate_variables.head(5).tolist())
        raise SpecError(
            "Spec file contains duplicate variable rows. Keep one row per variable. "
            f"Examples: {duplicate_text}."
        )

    return normalized.reset_index(drop=True)


def assess_specs(
    dataframe: pd.DataFrame,
    spec_dataframe: pd.DataFrame | None,
    process_columns: list[str],
    outcome_columns: list[str],
    ranked_drivers: pd.DataFrame | None = None,
    audit_result: Any | None = None,
) -> SpecAssessmentResult:
    """Assess specs/windows against historical data and model driver results."""
    specs = normalize_spec_dataframe(spec_dataframe)
    if specs.empty:
        return empty_spec_assessment()

    specs = infer_missing_roles(specs, process_columns, outcome_columns)
    matched_specs = specs[specs["variable"].isin(dataframe.columns)].copy()
    unmatched_specs = specs[~specs["variable"].isin(dataframe.columns)].copy()

    warnings = []
    if not unmatched_specs.empty:
        warnings.append(
            f"{len(unmatched_specs):,} spec variable(s) were not found in the merged data."
        )

    invalid_roles = matched_specs[
        matched_specs["role"].notna() & ~matched_specs["role"].isin(VALID_ROLES)
    ]
    if not invalid_roles.empty:
        warnings.append(
            "Some matched spec rows have roles other than 'process' or 'qc'. "
            "They will be treated as process windows unless the variable is a selected QC outcome."
        )

    driver_lookup = build_driver_lookup(ranked_drivers)
    confounded_variables = get_confounded_variables(audit_result)

    summary_rows = []
    out_of_spec_rows = []
    zone_rows = []

    for _, spec_row in matched_specs.iterrows():
        variable_name = spec_row["variable"]
        role = normalize_role(spec_row["role"], variable_name, outcome_columns)
        values = pd.to_numeric(dataframe[variable_name], errors="coerce")
        numeric_count = int(values.notna().sum())
        limit_kind = infer_limit_kind(spec_row)
        driver_score = driver_lookup.get(variable_name, np.nan)
        zone_series = classify_spec_zones(values, spec_row)

        summary = build_variable_summary_row(
            spec_row=spec_row,
            role=role,
            values=values,
            limit_kind=limit_kind,
            driver_score=driver_score,
            confounded=variable_name in confounded_variables,
            zone_series=zone_series,
        )
        summary_rows.append(summary)

        out_of_spec_rows.extend(
            build_out_of_spec_rows(
                dataframe=dataframe,
                variable_name=variable_name,
                values=values,
                spec_row=spec_row,
                role=role,
            )
        )

        if role == "process" and numeric_count >= 10:
            zone_rows.extend(
                build_outcome_zone_rows(
                    dataframe=dataframe,
                    variable_name=variable_name,
                    zone_series=zone_series,
                    outcome_columns=outcome_columns,
                )
            )

    variable_summary = pd.DataFrame(summary_rows)
    if not variable_summary.empty:
        variable_summary = variable_summary.sort_values(
            ["classification_priority", "percent_outside", "variable"],
            ascending=[True, False, True],
        ).drop(columns=["classification_priority"])

    out_of_spec_batches = pd.DataFrame(out_of_spec_rows)
    outcome_zone_summary = pd.DataFrame(zone_rows)

    return SpecAssessmentResult(
        specs=specs,
        matched_specs=matched_specs.reset_index(drop=True),
        unmatched_specs=unmatched_specs.reset_index(drop=True),
        variable_summary=variable_summary.reset_index(drop=True),
        out_of_spec_batches=out_of_spec_batches.reset_index(drop=True),
        outcome_zone_summary=outcome_zone_summary.reset_index(drop=True),
        warnings=warnings,
    )


def empty_spec_assessment() -> SpecAssessmentResult:
    """Return an empty assessment object."""
    return SpecAssessmentResult(
        specs=pd.DataFrame(columns=SPEC_COLUMNS),
        matched_specs=pd.DataFrame(columns=SPEC_COLUMNS),
        unmatched_specs=pd.DataFrame(columns=SPEC_COLUMNS),
        variable_summary=pd.DataFrame(),
        out_of_spec_batches=pd.DataFrame(),
        outcome_zone_summary=pd.DataFrame(),
        warnings=[],
    )


def infer_missing_roles(
    specs: pd.DataFrame,
    process_columns: list[str],
    outcome_columns: list[str],
) -> pd.DataFrame:
    """Infer missing or invalid roles from selected process/outcome columns."""
    output = specs.copy()
    process_set = set(process_columns)
    outcome_set = set(outcome_columns)

    def infer_role(row) -> str:
        current_role = str(row.get("role", "")).strip().lower()
        variable_name = row["variable"]
        if current_role in VALID_ROLES:
            return current_role
        if variable_name in outcome_set:
            return "qc"
        if variable_name in process_set:
            return "process"
        return current_role or "process"

    output["role"] = output.apply(infer_role, axis=1)
    return output


def normalize_role(role: Any, variable_name: str, outcome_columns: list[str]) -> str:
    """Return process/qc for downstream calculations."""
    role_text = str(role or "").strip().lower()
    if role_text in VALID_ROLES:
        return role_text
    return "qc" if variable_name in outcome_columns else "process"


def infer_limit_kind(spec_row: pd.Series) -> str:
    """Infer whether a spec has two-sided, one-sided, target-only, or no limits."""
    has_target = pd.notna(spec_row.get("target"))
    has_lower = pd.notna(spec_row.get("lower_limit"))
    has_upper = pd.notna(spec_row.get("upper_limit"))

    if has_lower and has_upper:
        return "two_sided"
    if has_lower:
        return "lower_only"
    if has_upper:
        return "upper_only"
    if has_target:
        return "target_only"
    return "no_limits"


def build_variable_summary_row(
    spec_row: pd.Series,
    role: str,
    values: pd.Series,
    limit_kind: str,
    driver_score: float,
    confounded: bool,
    zone_series: pd.Series,
) -> dict[str, Any]:
    """Build one row for the spec challenge table."""
    variable_name = spec_row["variable"]
    usable_values = values.dropna()
    count = int(usable_values.shape[0])
    missing_count = int(values.isna().sum())

    lower_limit = spec_row.get("lower_limit")
    upper_limit = spec_row.get("upper_limit")
    target = spec_row.get("target")

    below_count = int((usable_values < lower_limit).sum()) if pd.notna(lower_limit) else 0
    above_count = int((usable_values > upper_limit).sum()) if pd.notna(upper_limit) else 0
    outside_count = below_count + above_count
    percent_outside = safe_percent(outside_count, count)
    percent_inside = safe_percent(count - outside_count, count)

    mean_value = round_or_none(usable_values.mean())
    std_value = round_or_none(usable_values.std(ddof=1))
    min_value = round_or_none(usable_values.min())
    max_value = round_or_none(usable_values.max())
    observed_range = (
        float(usable_values.max() - usable_values.min()) if count and usable_values.nunique() > 1 else np.nan
    )
    allowed_range = (
        float(upper_limit - lower_limit)
        if pd.notna(lower_limit) and pd.notna(upper_limit) and upper_limit > lower_limit
        else np.nan
    )
    used_range_ratio = (
        observed_range / allowed_range
        if pd.notna(observed_range) and pd.notna(allowed_range) and allowed_range > 0
        else np.nan
    )

    margins = calculate_limit_margins(usable_values, spec_row)
    median_nearest_margin = round_or_none(margins.median()) if not margins.empty else None
    worst_margin = round_or_none(margins.min()) if not margins.empty else None

    close_to_limit_count = int(
        zone_series.isin(["near_lower_edge", "near_upper_edge"]).sum()
    )
    percent_close_to_limit = safe_percent(close_to_limit_count, count)

    cp, cpk = calculate_capability(usable_values, lower_limit, upper_limit)
    classification, reason, priority = classify_spec_window(
        role=role,
        count=count,
        limit_kind=limit_kind,
        percent_outside=percent_outside,
        percent_close_to_limit=percent_close_to_limit,
        used_range_ratio=used_range_ratio,
        driver_score=driver_score,
        confounded=confounded,
        target=target,
        mean_value=usable_values.mean() if count else np.nan,
        allowed_range=allowed_range,
    )

    return {
        "variable": variable_name,
        "role": role,
        "unit": spec_row.get("unit", ""),
        "criticality": spec_row.get("criticality", ""),
        "target": round_or_none(target),
        "lower_limit": round_or_none(lower_limit),
        "upper_limit": round_or_none(upper_limit),
        "limit_kind": limit_kind,
        "count": count,
        "missing_count": missing_count,
        "mean": mean_value,
        "std": std_value,
        "min": min_value,
        "max": max_value,
        "percent_inside": percent_inside,
        "percent_outside": percent_outside,
        "below_limit_count": below_count,
        "above_limit_count": above_count,
        "percent_close_to_limit": percent_close_to_limit,
        "used_range_ratio": round_or_none(used_range_ratio),
        "median_nearest_limit_margin": median_nearest_margin,
        "worst_limit_margin": worst_margin,
        "cp": round_or_none(cp),
        "cpk": round_or_none(cpk),
        "max_driver_score": round_or_none(driver_score),
        "confounding_flag": bool(confounded),
        "classification": classification,
        "reason": reason,
        "classification_priority": priority,
        "notes": spec_row.get("notes", ""),
    }


def classify_spec_window(
    role: str,
    count: int,
    limit_kind: str,
    percent_outside: float,
    percent_close_to_limit: float,
    used_range_ratio: float,
    driver_score: float,
    confounded: bool,
    target: float,
    mean_value: float,
    allowed_range: float,
) -> tuple[str, str, int]:
    """Return a conservative spec challenge classification."""
    if limit_kind == "no_limits":
        return (
            "Reference only",
            "No target or limits were supplied for this variable.",
            7,
        )

    if count < 20:
        return (
            "Insufficient data",
            "Fewer than 20 usable numeric observations are available.",
            6,
        )

    if role == "qc":
        if percent_outside > 0:
            return (
                "High failure risk",
                f"{percent_outside:.1f}% of historical batches are outside the QC spec.",
                1,
            )
        if percent_close_to_limit >= 20:
            return (
                "High failure risk",
                f"{percent_close_to_limit:.1f}% of batches are close to a QC limit.",
                1,
            )
        return (
            "Looks reasonable",
            "Historical QC values are mostly inside the supplied spec with usable margin.",
            5,
        )

    if confounded:
        return (
            "Confounded - review manually",
            "This variable overlaps with lot, supplier, operator, equipment, room, or campaign-style audit warnings.",
            2,
        )

    if pd.notna(used_range_ratio) and used_range_ratio < 0.35 and percent_outside == 0:
        return (
            "Insufficient data",
            f"Historical batches used only {used_range_ratio:.0%} of the allowed range, so the full window is not tested.",
            6,
        )

    high_driver = pd.notna(driver_score) and driver_score >= IMPORTANT_DRIVER_THRESHOLD
    low_driver = pd.notna(driver_score) and driver_score < LOW_DRIVER_THRESHOLD
    near_or_outside = percent_close_to_limit >= 15 or percent_outside > 0

    if pd.isna(driver_score) and near_or_outside:
        return (
            "Insufficient data",
            "Run driver analysis before judging whether this process window is too narrow or too wide.",
            6,
        )

    if (
        pd.notna(target)
        and pd.notna(mean_value)
        and pd.notna(allowed_range)
        and allowed_range > 0
        and abs(mean_value - target) > 0.20 * allowed_range
    ):
        return (
            "Target may be off-center",
            "The historical mean sits noticeably away from the stated target.",
            4,
        )

    if high_driver and near_or_outside:
        return (
            "Possibly too wide",
            "This variable is a strong model driver and historical batches approach or exceed the supplied window.",
            3,
        )

    if low_driver and (percent_close_to_limit >= 30 or percent_outside >= 5):
        return (
            "Possibly too narrow",
            "This variable has low current model-driver signal, but many batches are close to or outside the supplied window.",
            4,
        )

    return (
        "Looks reasonable",
        "Historical behavior is broadly consistent with the supplied window, based on available data.",
        5,
    )


def calculate_limit_margins(values: pd.Series, spec_row: pd.Series) -> pd.Series:
    """Calculate signed distance to the nearest relevant limit."""
    lower_limit = spec_row.get("lower_limit")
    upper_limit = spec_row.get("upper_limit")

    if values.empty:
        return pd.Series(dtype=float)

    if pd.notna(lower_limit) and pd.notna(upper_limit):
        return pd.concat([values - lower_limit, upper_limit - values], axis=1).min(axis=1)
    if pd.notna(upper_limit):
        return upper_limit - values
    if pd.notna(lower_limit):
        return values - lower_limit
    return pd.Series(dtype=float)


def classify_spec_zones(values: pd.Series, spec_row: pd.Series) -> pd.Series:
    """Classify numeric values into below, inside, near-edge, or above zones."""
    lower_limit = spec_row.get("lower_limit")
    upper_limit = spec_row.get("upper_limit")
    zones = pd.Series("inside", index=values.index, dtype="object")
    zones.loc[values.isna()] = "missing"

    if pd.notna(lower_limit):
        zones.loc[values < lower_limit] = "below_spec"
    if pd.notna(upper_limit):
        zones.loc[values > upper_limit] = "above_spec"

    if pd.notna(lower_limit) and pd.notna(upper_limit) and upper_limit > lower_limit:
        edge_width = EDGE_FRACTION * (upper_limit - lower_limit)
        inside_mask = zones.eq("inside")
        zones.loc[inside_mask & (values <= lower_limit + edge_width)] = "near_lower_edge"
        zones.loc[inside_mask & (values >= upper_limit - edge_width)] = "near_upper_edge"
    elif pd.notna(upper_limit):
        observed_range = values.max(skipna=True) - values.min(skipna=True)
        edge_width = EDGE_FRACTION * observed_range if pd.notna(observed_range) else 0
        inside_mask = zones.eq("inside")
        zones.loc[inside_mask & (values >= upper_limit - edge_width)] = "near_upper_edge"
    elif pd.notna(lower_limit):
        observed_range = values.max(skipna=True) - values.min(skipna=True)
        edge_width = EDGE_FRACTION * observed_range if pd.notna(observed_range) else 0
        inside_mask = zones.eq("inside")
        zones.loc[inside_mask & (values <= lower_limit + edge_width)] = "near_lower_edge"

    return zones


def build_out_of_spec_rows(
    dataframe: pd.DataFrame,
    variable_name: str,
    values: pd.Series,
    spec_row: pd.Series,
    role: str,
) -> list[dict[str, Any]]:
    """Return batch-level rows for values outside supplied limits."""
    lower_limit = spec_row.get("lower_limit")
    upper_limit = spec_row.get("upper_limit")
    outside_mask = pd.Series(False, index=dataframe.index)
    status = pd.Series("", index=dataframe.index, dtype="object")

    if pd.notna(lower_limit):
        below_mask = values < lower_limit
        outside_mask = outside_mask | below_mask
        status.loc[below_mask] = "below_lower_limit"
    if pd.notna(upper_limit):
        above_mask = values > upper_limit
        outside_mask = outside_mask | above_mask
        status.loc[above_mask] = "above_upper_limit"

    output_rows = []
    batch_ids = dataframe["batch_id"] if "batch_id" in dataframe.columns else dataframe.index.astype(str)
    for row_index in dataframe.index[outside_mask.fillna(False)]:
        output_rows.append(
            {
                "batch_id": batch_ids.loc[row_index],
                "row_index": int(row_index) if isinstance(row_index, (int, np.integer)) else str(row_index),
                "variable": variable_name,
                "role": role,
                "value": round_or_none(values.loc[row_index]),
                "status": status.loc[row_index],
                "lower_limit": round_or_none(lower_limit),
                "upper_limit": round_or_none(upper_limit),
                "unit": spec_row.get("unit", ""),
            }
        )

    return output_rows


def build_outcome_zone_rows(
    dataframe: pd.DataFrame,
    variable_name: str,
    zone_series: pd.Series,
    outcome_columns: list[str],
) -> list[dict[str, Any]]:
    """Summarize outcome means by spec zone for one process variable."""
    output_rows = []
    for outcome_column in outcome_columns:
        if outcome_column not in dataframe.columns:
            continue
        outcome_values = pd.to_numeric(dataframe[outcome_column], errors="coerce")
        zone_data = pd.DataFrame(
            {"zone": zone_series, "outcome": outcome_values}
        ).dropna(subset=["outcome"])
        if zone_data.empty:
            continue
        grouped = (
            zone_data.groupby("zone", observed=True)["outcome"]
            .agg(["count", "mean"])
            .reset_index()
        )
        for _, row in grouped.iterrows():
            output_rows.append(
                {
                    "variable": variable_name,
                    "outcome": outcome_column,
                    "zone": row["zone"],
                    "count": int(row["count"]),
                    "mean": round_or_none(row["mean"]),
                }
            )

    return output_rows


def calculate_capability(
    values: pd.Series,
    lower_limit: float,
    upper_limit: float,
) -> tuple[float | None, float | None]:
    """Calculate simple Cp/Cpk when two-sided limits are available."""
    usable_values = values.dropna()
    if (
        usable_values.shape[0] < 10
        or pd.isna(lower_limit)
        or pd.isna(upper_limit)
        or upper_limit <= lower_limit
    ):
        return None, None

    standard_deviation = usable_values.std(ddof=1)
    if pd.isna(standard_deviation) or standard_deviation <= 0:
        return None, None

    mean_value = usable_values.mean()
    cp = (upper_limit - lower_limit) / (6.0 * standard_deviation)
    cpk = min(
        (upper_limit - mean_value) / (3.0 * standard_deviation),
        (mean_value - lower_limit) / (3.0 * standard_deviation),
    )
    return float(cp), float(cpk)


def build_driver_lookup(ranked_drivers: pd.DataFrame | None) -> dict[str, float]:
    """Return max combined score per process variable."""
    if ranked_drivers is None or ranked_drivers.empty:
        return {}
    grouped = ranked_drivers.groupby("process_variable")["combined_score"].max()
    return grouped.to_dict()


def get_confounded_variables(audit_result: Any | None) -> set[str]:
    """Return audit-detected categorical variables that may be confounded."""
    if audit_result is None or getattr(audit_result, "confounding_categoricals", None) is None:
        return set()
    dataframe = audit_result.confounding_categoricals
    if dataframe.empty or "column" not in dataframe.columns:
        return set()
    return set(dataframe["column"].astype(str).tolist())


def safe_percent(numerator: int, denominator: int) -> float:
    """Return a percentage while avoiding division by zero."""
    if denominator <= 0:
        return 0.0
    return 100.0 * numerator / denominator


def round_or_none(value: Any, digits: int = 3) -> float | None:
    """Round numeric values for display/JSON while preserving missing values."""
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
