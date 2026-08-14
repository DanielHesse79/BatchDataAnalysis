"""Deterministic evidence pack for LLM report generation.

The analysis code should be the source of truth. This module converts model,
audit, profiling, and spec outputs into compact verified facts that a language
model can narrate without rediscovering patterns from large raw tables.
"""

from __future__ import annotations

import re
from typing import Any

import pandas as pd


REPORT_PACK_VERSION = "2026-04-26"
TOP_DRIVER_COUNT = 6
TOP_METHOD_COUNT = 5
TOP_CATEGORY_LEVEL_COUNT = 4
TOP_NUMERIC_PATTERN_COUNT = 4
TOP_INTERACTION_COUNT = 4
SWEET_SPOT_TOLERANCE_FRACTION = 0.10
REFINED_MIN_ROWS = 40
REFINED_MIN_BIN_COUNT = 4
REFINED_MAX_BIN_COUNT = 10
REFINED_TARGET_ROWS_PER_BIN = 15
# The winning refined bin is the best of several noisy bin means, so its lift is
# optimistically biased by construction. Evenly filled quantile bins hold at
# least ten rows under the bin-count policy above, so this floor only removes
# bins left thin by pd.qcut edge collapse on duplicated values.
REFINED_MIN_WINNING_BIN_COUNT = 8
VALID_OUTCOME_OBJECTIVES = {"maximize", "minimize", "unknown"}

LOWER_IS_BETTER_OUTCOME_KEYWORDS = [
    "loss",
    "hcp",
    "aggregate",
    "impurity",
    "moisture",
    "residual",
    "endotoxin",
    "bioburden",
    "host_cell",
    "color_index",
    "defect",
    "deviation",
    "ppm",
]
HIGHER_IS_BETTER_OUTCOME_KEYWORDS = [
    "yield",
    "purity",
    "potency",
    "titer",
    "titre",
    "recovery",
    "activity",
    "viability",
]

OPERATING_WINDOW_HINT_COLUMNS = [
    "outcome",
    "process_variable",
    "objective",
    "range_label",
    "range_min",
    "range_max",
    "mean_in_range",
    "mean_outside_range",
    "directional_lift_vs_outside",
    "refined_range_label",
    "refined_range_min",
    "refined_range_max",
    "refined_mean",
    "refined_count",
    "refined_bin_count",
    "refined_directional_lift_vs_other_bins",
    "refined_directional_lift_standard_error",
    "refined_directional_lift_z_score",
    "refined_pattern_type",
    "pattern_type",
    "evidence_note",
    "refined_evidence_note",
]


def build_report_pack(
    profile_result,
    audit_result,
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    outcomes: list[str],
    spec_assessment=None,
    outcome_objectives: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Create a compact, JSON-ready report pack for LLM interpretation.

    ``outcome_objectives`` optionally overrides the keyword-based maximize /
    minimize guess per outcome, for cases the name cannot settle.
    """
    ranked_drivers = analysis_results["ranked_drivers"]

    return {
        "report_pack_version": REPORT_PACK_VERSION,
        "guardrails": [
            "Use only the facts in this report pack.",
            "Findings are associations in historical data, not causal proof.",
            "Quartile thresholds and interaction screens are exploratory, not optimized setpoints.",
            "Specs and process windows are historical assessments, not proven design spaces.",
            "Do not invent typical industry specs, release limits, variables, lots, reactors, or numeric values.",
            "Operational actions should be phrased as investigations or confirmation runs.",
        ],
        "allowed_variables": {
            "process_columns": profile_result.process_columns,
            "outcome_columns": outcomes,
            "categorical_levels": build_allowed_categorical_levels(
                merged_dataframe,
                profile_result,
            ),
        },
        "data_profile": build_data_profile_pack(profile_result),
        "audit": build_audit_pack(audit_result),
        "specs_and_windows": build_specs_pack(spec_assessment),
        "pca": build_pca_pack(analysis_results),
        "outcomes": {
            outcome: build_outcome_dossier(
                outcome=outcome,
                ranked_drivers=ranked_drivers,
                analysis_results=analysis_results,
                merged_dataframe=merged_dataframe,
                profile_result=profile_result,
                outcome_objectives=outcome_objectives,
            )
            for outcome in outcomes
        },
    }


def build_data_profile_pack(profile_result) -> dict[str, Any]:
    """Summarize profiling facts."""
    high_missing = profile_result.missingness[profile_result.missingness["flag"]]
    return {
        "matched_batch_count": int(profile_result.matched_batch_count),
        "process_variable_count": len(profile_result.process_columns),
        "outcome_count": len(profile_result.outcome_columns),
        "variable_type_counts": profile_result.variable_type_counts,
        "warnings": profile_result.warnings,
        "high_missing_columns": dataframe_to_records(high_missing, max_rows=8),
        "near_constant_columns": dataframe_to_records(
            profile_result.near_constant_columns,
            max_rows=8,
        ),
        "outcome_statistics": dataframe_to_records(
            profile_result.outcome_statistics,
            max_rows=20,
        ),
    }


def build_audit_pack(audit_result) -> dict[str, Any]:
    """Summarize audit facts that should affect confidence."""
    if audit_result is None:
        return {"available": False}

    return {
        "available": True,
        "summary": audit_result.summary,
        "sample_size_guidance": audit_result.sample_size_guidance,
        "warnings": audit_result.warnings[:10],
        "leakage_name_warnings": dataframe_to_records(
            audit_result.leakage_name_warnings,
            max_rows=8,
        ),
        "date_or_drift_columns": dataframe_to_records(
            audit_result.date_or_drift_columns,
            max_rows=8,
        ),
        "confounding_categoricals": dataframe_to_records(
            audit_result.confounding_categoricals,
            max_rows=8,
        ),
        "outlier_flags": dataframe_to_records(audit_result.outlier_flags, max_rows=8),
        "high_feature_target_correlations": dataframe_to_records(
            audit_result.feature_target_correlations,
            max_rows=8,
        ),
        "multicollinearity_pairs": dataframe_to_records(
            audit_result.multicollinearity_pairs,
            max_rows=8,
        ),
    }


def build_specs_pack(spec_assessment) -> dict[str, Any]:
    """Summarize specs and process-window evidence."""
    if spec_assessment is None or not getattr(spec_assessment, "has_specs", False):
        return {
            "available": False,
            "instruction": "No spec/window file was supplied. Do not discuss release specs or process-window classifications.",
        }

    variable_summary = spec_assessment.variable_summary
    if variable_summary.empty:
        return {
            "available": True,
            "matched_spec_count": len(spec_assessment.matched_specs),
            "unmatched_spec_count": len(spec_assessment.unmatched_specs),
            "warnings": spec_assessment.warnings,
            "spec_challenge_table": [],
            "out_of_spec_batches": [],
            "outcome_means_by_spec_zone": [],
        }

    challenge_columns = [
        "variable",
        "role",
        "target",
        "lower_limit",
        "upper_limit",
        "unit",
        "criticality",
        "count",
        "mean",
        "std",
        "min",
        "max",
        "percent_inside",
        "percent_outside",
        "below_limit_count",
        "above_limit_count",
        "percent_close_to_limit",
        "used_range_ratio",
        "max_driver_score",
        "confounding_flag",
        "classification",
        "reason",
    ]
    available_columns = [
        column_name for column_name in challenge_columns if column_name in variable_summary.columns
    ]
    sorted_summary = variable_summary.sort_values(
        ["percent_outside", "max_driver_score"],
        ascending=[False, False],
    )

    return {
        "available": True,
        "instruction": (
            "Use these supplied limits only. Do not replace them with assumed industry specs. "
            "Process rows are operating windows; QC rows are QC specs or internal QC limits."
        ),
        "matched_spec_count": len(spec_assessment.matched_specs),
        "unmatched_spec_count": len(spec_assessment.unmatched_specs),
        "warnings": spec_assessment.warnings,
        "classification_counts": variable_summary["classification"].value_counts().to_dict()
        if "classification" in variable_summary.columns
        else {},
        "spec_challenge_table": dataframe_to_records(
            sorted_summary[available_columns],
            max_rows=20,
        ),
        "out_of_spec_batches": dataframe_to_records(
            spec_assessment.out_of_spec_batches,
            max_rows=18,
        ),
        "outcome_means_by_spec_zone": dataframe_to_records(
            spec_assessment.outcome_zone_summary,
            max_rows=30,
        ),
    }


def build_pca_pack(analysis_results: dict[str, Any]) -> dict[str, Any]:
    """Summarize PCA output compactly."""
    pca_result = analysis_results["pca"]
    return {
        "explained_variance": dataframe_to_records(
            pca_result["explained_variance"],
            max_rows=5,
        ),
        "top_process_loadings": dataframe_to_records(
            pca_result["process_loadings"].head(8),
            max_rows=8,
        ),
    }


def build_outcome_dossier(
    outcome: str,
    ranked_drivers: pd.DataFrame,
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    profile_result,
    outcome_objectives: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Build one deterministic dossier per outcome."""
    outcome_drivers = ranked_drivers[ranked_drivers["outcome"] == outcome].head(TOP_DRIVER_COUNT)
    top_driver_names = outcome_drivers["process_variable"].tolist()
    numeric_driver_patterns = build_numeric_driver_patterns(
        merged_dataframe=merged_dataframe,
        profile_result=profile_result,
        outcome=outcome,
        candidate_variables=top_driver_names,
        outcome_objectives=outcome_objectives,
    )

    return {
        "outcome": outcome,
        "model_quality": build_model_quality_for_outcome(analysis_results, outcome),
        "top_drivers": dataframe_to_records(outcome_drivers, max_rows=TOP_DRIVER_COUNT),
        "method_evidence": build_method_evidence(analysis_results, outcome),
        "categorical_level_effects": build_categorical_level_effects(
            merged_dataframe=merged_dataframe,
            profile_result=profile_result,
            outcome=outcome,
            candidate_variables=top_driver_names,
        ),
        "numeric_driver_patterns": numeric_driver_patterns,
        "historical_response_bands": [
            pattern["historical_response_band"]
            for pattern in numeric_driver_patterns
            if pattern.get("historical_response_band")
        ],
        "exploratory_interactions": build_interaction_screen(
            merged_dataframe=merged_dataframe,
            profile_result=profile_result,
            outcome=outcome,
            candidate_variables=top_driver_names,
        ),
    }


def build_model_quality_for_outcome(
    analysis_results: dict[str, Any],
    outcome: str,
) -> dict[str, Any]:
    """Build compact model quality facts for one outcome."""
    pls_result = analysis_results["pls"][outcome]
    random_forest_result = analysis_results["random_forest"][outcome]
    return {
        "pls_cv_q2": round_or_none(pls_result.get("q2")),
        "pls_training_r2": round_or_none(pls_result.get("training_r2")),
        "pls_components": pls_result.get("n_components"),
        "pls_time_ordered_validation": sanitize_validation_summary(
            pls_result.get("time_ordered_validation", {})
        ),
        "rf_oob_r2": round_or_none(random_forest_result.get("oob_r2")),
        "rf_training_r2": round_or_none(random_forest_result.get("training_r2")),
        "rf_time_ordered_validation": sanitize_validation_summary(
            random_forest_result.get("time_ordered_validation", {})
        ),
        "rows_used": random_forest_result.get("n_rows_used"),
    }


def build_method_evidence(
    analysis_results: dict[str, Any],
    outcome: str,
) -> dict[str, Any]:
    """Return method-specific top variables for one outcome."""
    pls_result = analysis_results["pls"][outcome]
    random_forest_result = analysis_results["random_forest"][outcome]
    return {
        "pls_top_variables": dataframe_to_records(
            pls_result["variable_importance"].head(TOP_METHOD_COUNT),
            max_rows=TOP_METHOD_COUNT,
        ),
        "rf_top_variables": dataframe_to_records(
            random_forest_result["variable_importance"].head(TOP_METHOD_COUNT),
            max_rows=TOP_METHOD_COUNT,
        ),
    }


def build_categorical_level_effects(
    merged_dataframe: pd.DataFrame,
    profile_result,
    outcome: str,
    candidate_variables: list[str],
) -> list[dict[str, Any]]:
    """Return best/worst category levels for candidate categorical drivers."""
    categorical_variables = get_variables_by_type(profile_result, {"categorical", "binary"})
    outcome_values = pd.to_numeric(merged_dataframe[outcome], errors="coerce")
    all_rows_outcome_mean = outcome_values.mean()
    results = []

    for variable in candidate_variables:
        if variable not in categorical_variables or variable not in merged_dataframe.columns:
            continue

        # The level means below exclude rows where the variable itself is
        # missing, so the baseline has to exclude them too. Comparing against an
        # all-rows mean makes the deltas fail to decompose whenever a variable
        # has informative missingness.
        comparable_data = merged_dataframe.assign(_outcome=outcome_values).dropna(
            subset=[variable, "_outcome"]
        )
        if comparable_data.empty:
            continue

        comparable_outcome_mean = float(comparable_data["_outcome"].mean())
        grouped = (
            comparable_data.groupby(variable)["_outcome"].agg(["count", "mean"]).reset_index()
        )
        if grouped.empty:
            continue

        grouped["level"] = grouped[variable].astype(str)
        grouped["level_label"] = variable + "=" + grouped["level"]
        grouped["delta_from_overall_mean"] = grouped["mean"] - comparable_outcome_mean
        grouped = grouped.sort_values(
            "delta_from_overall_mean",
            key=lambda values: values.abs(),
            ascending=False,
        )
        grouped["mean"] = grouped["mean"].round(3)
        grouped["delta_from_overall_mean"] = grouped["delta_from_overall_mean"].round(3)
        results.append(
            {
                "variable": variable,
                "overall_outcome_mean": round_or_none(comparable_outcome_mean),
                "overall_outcome_mean_all_rows": round_or_none(all_rows_outcome_mean),
                "rows_used": int(len(comparable_data)),
                "rows_excluded_missing_variable_or_outcome": int(
                    len(merged_dataframe) - len(comparable_data)
                ),
                "baseline_note": (
                    "delta_from_overall_mean is measured against overall_outcome_mean, "
                    "which uses only rows where this variable and the outcome are both present."
                ),
                "largest_level_effects": dataframe_to_records(
                    grouped[
                        ["level_label", "count", "mean", "delta_from_overall_mean"]
                    ],
                    max_rows=TOP_CATEGORY_LEVEL_COUNT,
                ),
            }
        )

    return results


def build_numeric_driver_patterns(
    merged_dataframe: pd.DataFrame,
    profile_result,
    outcome: str,
    candidate_variables: list[str],
    outcome_objectives: dict[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return quartile summaries and correlations for numeric drivers."""
    numeric_variables = get_variables_by_type(profile_result, {"continuous"})
    outcome_values = pd.to_numeric(merged_dataframe[outcome], errors="coerce")
    results = []

    for variable in candidate_variables:
        if len(results) >= TOP_NUMERIC_PATTERN_COUNT:
            break
        if variable not in numeric_variables or variable not in merged_dataframe.columns:
            continue

        variable_values = pd.to_numeric(merged_dataframe[variable], errors="coerce")
        usable_data = pd.DataFrame(
            {"variable_value": variable_values, "outcome_value": outcome_values}
        ).dropna()
        if usable_data["variable_value"].nunique() < 4:
            continue

        usable_data["quartile"] = pd.qcut(
            usable_data["variable_value"],
            q=4,
            duplicates="drop",
        )
        quartile_summary = (
            usable_data.groupby("quartile", observed=True)
            .agg(
                count=("outcome_value", "count"),
                variable_min=("variable_value", "min"),
                variable_max=("variable_value", "max"),
                outcome_mean=("outcome_value", "mean"),
            )
            .reset_index()
        )
        quartile_summary["quartile"] = quartile_summary["quartile"].astype(str)
        for column in ["variable_min", "variable_max", "outcome_mean"]:
            quartile_summary[column] = quartile_summary[column].round(3)

        correlation = usable_data["variable_value"].corr(usable_data["outcome_value"])
        results.append(
            {
                "variable": variable,
                "pearson_correlation": round_or_none(correlation),
                "direction_hint": classify_correlation_direction(correlation),
                "quartile_summary": dataframe_to_records(quartile_summary),
                "historical_response_band": summarize_historical_response_band(
                    variable=variable,
                    outcome=outcome,
                    quartile_summary=quartile_summary,
                    usable_data=usable_data,
                    outcome_objectives=outcome_objectives,
                ),
            }
        )

    return results


def get_numeric_driver_candidates(
    outcome_drivers: pd.DataFrame,
    profile_result,
    merged_dataframe: pd.DataFrame,
    minimum_distinct_values: int = 3,
) -> list[str]:
    """Return ranked numeric drivers that can support a response-shape plot.

    A variable needs enough distinct values for a shape to mean anything; two
    distinct values describe a group difference, not a response curve.
    """
    continuous_columns = set(
        profile_result.variable_types[
            profile_result.variable_types["type"] == "continuous"
        ]["column"].tolist()
    )

    return [
        variable_name
        for variable_name in outcome_drivers["process_variable"].tolist()
        if variable_name in continuous_columns
        and variable_name in merged_dataframe.columns
        and pd.to_numeric(merged_dataframe[variable_name], errors="coerce").nunique()
        >= minimum_distinct_values
    ]


def get_response_band_for_variable(
    operating_window_hints: pd.DataFrame,
    variable_name: str,
) -> dict[str, Any] | None:
    """Return one response-band row as a dict, if one was produced."""
    if operating_window_hints is None or operating_window_hints.empty:
        return None

    matching_rows = operating_window_hints[
        operating_window_hints["process_variable"] == variable_name
    ]
    if matching_rows.empty:
        return None

    return matching_rows.iloc[0].to_dict()


def build_operating_window_hints(
    ranked_drivers: pd.DataFrame,
    merged_dataframe: pd.DataFrame,
    profile_result,
    outcomes: list[str] | None = None,
    max_variables_per_outcome: int = TOP_NUMERIC_PATTERN_COUNT,
    outcome_objectives: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Return suggested historical response bands for top numeric drivers.

    These rows are exploratory. They summarize where historical quartile means
    were best for the inferred outcome objective; they are not optimized
    setpoints or validated design spaces.
    """
    if ranked_drivers is None or ranked_drivers.empty:
        return pd.DataFrame(columns=OPERATING_WINDOW_HINT_COLUMNS)

    selected_outcomes = outcomes or ranked_drivers["outcome"].drop_duplicates().tolist()
    rows: list[dict[str, Any]] = []

    for outcome in selected_outcomes:
        candidate_variables = (
            ranked_drivers[ranked_drivers["outcome"] == outcome]["process_variable"]
            .head(max_variables_per_outcome)
            .tolist()
        )
        numeric_patterns = build_numeric_driver_patterns(
            merged_dataframe=merged_dataframe,
            profile_result=profile_result,
            outcome=outcome,
            candidate_variables=candidate_variables,
            outcome_objectives=outcome_objectives,
        )
        for pattern in numeric_patterns:
            response_band = pattern.get("historical_response_band")
            if not response_band:
                continue
            rows.append(
                {
                    "outcome": outcome,
                    "process_variable": pattern["variable"],
                    **response_band,
                }
            )

    if not rows:
        return pd.DataFrame(columns=OPERATING_WINDOW_HINT_COLUMNS)

    output_dataframe = pd.DataFrame(rows)
    for column_name in OPERATING_WINDOW_HINT_COLUMNS:
        if column_name not in output_dataframe.columns:
            output_dataframe[column_name] = None

    return output_dataframe[OPERATING_WINDOW_HINT_COLUMNS]


def summarize_historical_response_band(
    variable: str,
    outcome: str,
    quartile_summary: pd.DataFrame,
    usable_data: pd.DataFrame,
    outcome_objectives: dict[str, str] | None = None,
) -> dict[str, Any] | None:
    """Identify the best contiguous quartile band for an inferred objective."""
    objective = infer_outcome_objective(outcome, outcome_objectives=outcome_objectives)
    if objective == "unknown" or quartile_summary.empty or len(quartile_summary) < 3:
        return None

    means = pd.to_numeric(quartile_summary["outcome_mean"], errors="coerce")
    counts = pd.to_numeric(quartile_summary["count"], errors="coerce").fillna(0)
    if means.isna().any() or counts.sum() <= 0:
        return None

    response_range = float(means.max() - means.min())
    if response_range <= 0:
        return None

    tolerance = max(response_range * SWEET_SPOT_TOLERANCE_FRACTION, 1e-9)
    mean_values = means.to_numpy(dtype=float)
    if objective == "maximize":
        best_position = int(mean_values.argmax())
        qualifies = means >= float(mean_values[best_position]) - tolerance
    else:
        best_position = int(mean_values.argmin())
        qualifies = means <= float(mean_values[best_position]) + tolerance

    selected_positions = expand_contiguous_positions(
        qualifies=qualifies.tolist(),
        starting_position=best_position,
    )
    if not selected_positions or len(selected_positions) == len(quartile_summary):
        return None

    selected_summary = quartile_summary.iloc[selected_positions]
    outside_summary = quartile_summary.drop(quartile_summary.index[selected_positions])
    if outside_summary.empty:
        return None

    mean_in_range = weighted_mean(
        selected_summary["outcome_mean"],
        selected_summary["count"],
    )
    mean_outside_range = weighted_mean(
        outside_summary["outcome_mean"],
        outside_summary["count"],
    )
    if mean_in_range is None or mean_outside_range is None:
        return None

    if objective == "maximize":
        directional_lift = mean_in_range - mean_outside_range
        objective_label = "maximize outcome"
        evidence_note = (
            "Broad quartile-based historical response band; confirm before treating as a setpoint."
        )
    else:
        directional_lift = mean_outside_range - mean_in_range
        objective_label = "minimize outcome"
        evidence_note = (
            "Broad quartile-based historical response band; confirm before treating as a setpoint."
        )

    if directional_lift <= 0:
        return None

    range_min = float(selected_summary["variable_min"].min())
    range_max = float(selected_summary["variable_max"].max())
    pattern_type = classify_response_band(selected_positions, len(quartile_summary))

    response_band = {
        "variable": variable,
        "objective": objective_label,
        "range_label": f"{format_range_number(range_min)} to {format_range_number(range_max)}",
        "range_min": round_or_none(range_min),
        "range_max": round_or_none(range_max),
        "mean_in_range": round_or_none(mean_in_range),
        "mean_outside_range": round_or_none(mean_outside_range),
        "directional_lift_vs_outside": round_or_none(directional_lift),
        "best_quartile_mean": round_or_none(mean_values[best_position]),
        "selected_quartile_count": len(selected_positions),
        "pattern_type": pattern_type,
        "evidence_note": evidence_note,
    }
    refined_window = summarize_refined_response_window(
        variable=variable,
        objective=objective,
        usable_data=usable_data,
        lower_bound=range_min,
        upper_bound=range_max,
    )
    if refined_window:
        response_band.update(refined_window)

    return response_band


def summarize_refined_response_window(
    variable: str,
    objective: str,
    usable_data: pd.DataFrame,
    lower_bound: float,
    upper_bound: float,
) -> dict[str, Any] | None:
    """Identify the best narrower quantile bin for a numeric driver.

    The refined bin is intentionally separate from the broader quartile band. It
    is more precise, but more sensitive to random noise and uneven historical
    coverage.

    The winning bin is picked by argmax over several noisy bin means, so its
    reported lift is a max-of-noisy-means and is optimistically biased. Two
    guards keep that honest: the winning bin must hold at least
    ``REFINED_MIN_WINNING_BIN_COUNT`` batches, and the lift is reported together
    with a normal-approximation standard error and z score.
    """
    bounded_data = usable_data[
        (usable_data["variable_value"] >= lower_bound)
        & (usable_data["variable_value"] <= upper_bound)
    ].copy()

    if len(bounded_data) < REFINED_MIN_ROWS or bounded_data["variable_value"].nunique() < 8:
        return None

    bin_count = min(
        REFINED_MAX_BIN_COUNT,
        max(REFINED_MIN_BIN_COUNT, int(len(bounded_data) // REFINED_TARGET_ROWS_PER_BIN)),
    )
    refined_data = bounded_data.copy()
    refined_data["refined_bin"] = pd.qcut(
        refined_data["variable_value"],
        q=bin_count,
        duplicates="drop",
    )
    refined_summary = (
        refined_data.groupby("refined_bin", observed=True)
        .agg(
            count=("outcome_value", "count"),
            variable_min=("variable_value", "min"),
            variable_max=("variable_value", "max"),
            outcome_mean=("outcome_value", "mean"),
        )
        .reset_index()
    )
    if len(refined_summary) < REFINED_MIN_BIN_COUNT:
        return None

    mean_values = pd.to_numeric(refined_summary["outcome_mean"], errors="coerce").to_numpy(
        dtype=float
    )
    if objective == "maximize":
        best_position = int(mean_values.argmax())
    else:
        best_position = int(mean_values.argmin())

    best_row = refined_summary.iloc[best_position]
    if int(best_row["count"]) < REFINED_MIN_WINNING_BIN_COUNT:
        return None

    other_summary = refined_summary.drop(refined_summary.index[best_position])
    refined_mean = float(best_row["outcome_mean"])
    other_mean = weighted_mean(other_summary["outcome_mean"], other_summary["count"])
    if other_mean is None:
        return None

    best_bin_mask = refined_data["refined_bin"] == best_row["refined_bin"]
    standard_error = difference_standard_error(
        refined_data.loc[best_bin_mask, "outcome_value"],
        refined_data.loc[~best_bin_mask, "outcome_value"],
    )

    if objective == "maximize":
        directional_lift = refined_mean - other_mean
        evidence_note = (
            "Narrower best observed quantile bin, chosen as the highest of several noisy bin means, "
            "so the lift is optimistically biased. Compare it with its standard error before treating it as real."
        )
    else:
        directional_lift = other_mean - refined_mean
        evidence_note = (
            "Narrower lowest observed quantile bin, chosen as the lowest of several noisy bin means, "
            "so the lift is optimistically biased. Compare it with its standard error before treating it as real."
        )

    if directional_lift <= 0:
        return None

    refined_min = float(best_row["variable_min"])
    refined_max = float(best_row["variable_max"])
    return {
        "refined_range_label": f"{format_range_number(refined_min)} to {format_range_number(refined_max)}",
        "refined_range_min": round_or_none(refined_min),
        "refined_range_max": round_or_none(refined_max),
        "refined_mean": round_or_none(refined_mean),
        "refined_count": int(best_row["count"]),
        "refined_bin_count": int(len(refined_summary)),
        "refined_directional_lift_vs_other_bins": round_or_none(directional_lift),
        "refined_directional_lift_standard_error": round_or_none(standard_error),
        "refined_directional_lift_z_score": round_or_none(
            directional_lift / standard_error
            if standard_error is not None and standard_error > 0
            else None
        ),
        "refined_pattern_type": classify_refined_bin_position(
            best_position=best_position,
            bin_count=len(refined_summary),
        ),
        "refined_evidence_note": evidence_note,
    }


def difference_standard_error(
    first_values: pd.Series,
    second_values: pd.Series,
) -> float | None:
    """Return the normal-approximation standard error of a difference of means."""
    first_numeric = pd.to_numeric(first_values, errors="coerce").dropna()
    second_numeric = pd.to_numeric(second_values, errors="coerce").dropna()
    first_count = int(first_numeric.shape[0])
    second_count = int(second_numeric.shape[0])
    if first_count < 2 or second_count < 2:
        return None

    first_variance = float(first_numeric.var(ddof=1))
    second_variance = float(second_numeric.var(ddof=1))
    if pd.isna(first_variance) or pd.isna(second_variance):
        return None

    return float((first_variance / first_count + second_variance / second_count) ** 0.5)


def classify_refined_bin_position(best_position: int, bin_count: int) -> str:
    """Describe where the best refined bin sits in the observed range."""
    if best_position == 0:
        return "lower_edge_best_bin"
    if best_position == bin_count - 1:
        return "upper_edge_best_bin"
    return "middle_best_bin"


def tokenize_outcome_name(outcome: str) -> list[str]:
    """Split an outcome name into lowercase whole tokens."""
    spaced_name = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", str(outcome))
    return [token for token in re.split(r"[^0-9a-zA-Z]+", spaced_name.lower()) if token]


def keyword_matches_tokens(keyword: str, tokens: list[str]) -> bool:
    """Return whether a keyword appears as a whole run of tokens in a name."""
    keyword_tokens = tokenize_outcome_name(keyword)
    if not keyword_tokens or len(keyword_tokens) > len(tokens):
        return False
    span = len(keyword_tokens)
    return any(
        tokens[start : start + span] == keyword_tokens
        for start in range(len(tokens) - span + 1)
    )


def infer_outcome_objective(
    outcome: str,
    outcome_objectives: dict[str, str] | None = None,
) -> str:
    """Infer whether a QC outcome is usually maximized or minimized.

    Keywords are matched as whole tokens. Substring matching inverted the
    optimization direction for names such as ``residual_activity``, where
    "residual" hit the lower-is-better list even though the quantity should be
    maximized, so the response band, sweet spot, and every directional lift were
    computed for the wrong direction. When keywords from both lists match, the
    objective stays "unknown" and no band is emitted, which is the conservative
    answer. ``outcome_objectives`` lets a caller override the guess per outcome.
    """
    if outcome_objectives:
        override = str(outcome_objectives.get(outcome, "") or "").strip().lower()
        if override in VALID_OUTCOME_OBJECTIVES:
            return override

    tokens = tokenize_outcome_name(outcome)
    lower_is_better = any(
        keyword_matches_tokens(keyword, tokens)
        for keyword in LOWER_IS_BETTER_OUTCOME_KEYWORDS
    )
    higher_is_better = any(
        keyword_matches_tokens(keyword, tokens)
        for keyword in HIGHER_IS_BETTER_OUTCOME_KEYWORDS
    )

    if lower_is_better and higher_is_better:
        return "unknown"
    if lower_is_better:
        return "minimize"
    if higher_is_better:
        return "maximize"
    return "unknown"


def expand_contiguous_positions(
    qualifies: list[bool],
    starting_position: int,
) -> list[int]:
    """Expand left/right from the best quartile while adjacent bins qualify."""
    selected_positions = [starting_position]

    left_position = starting_position - 1
    while left_position >= 0 and qualifies[left_position]:
        selected_positions.insert(0, left_position)
        left_position -= 1

    right_position = starting_position + 1
    while right_position < len(qualifies) and qualifies[right_position]:
        selected_positions.append(right_position)
        right_position += 1

    return selected_positions


def classify_response_band(selected_positions: list[int], bin_count: int) -> str:
    """Describe whether the best response band is in the middle or at an edge."""
    first_position = min(selected_positions)
    last_position = max(selected_positions)
    if first_position > 0 and last_position < bin_count - 1:
        return "middle_sweet_spot"
    if first_position == 0 and last_position < bin_count - 1:
        return "lower_band"
    if first_position > 0 and last_position == bin_count - 1:
        return "upper_band"
    return "broad_or_edge_band"


def weighted_mean(values: pd.Series, weights: pd.Series) -> float | None:
    """Return a weighted mean, or None when no weight is available."""
    numeric_values = pd.to_numeric(values, errors="coerce")
    numeric_weights = pd.to_numeric(weights, errors="coerce").fillna(0)
    usable_mask = numeric_values.notna() & (numeric_weights > 0)
    if not usable_mask.any():
        return None
    return float(
        (numeric_values.loc[usable_mask] * numeric_weights.loc[usable_mask]).sum()
        / numeric_weights.loc[usable_mask].sum()
    )


def format_range_number(value: float) -> str:
    """Format a range endpoint compactly without hiding useful tenths."""
    absolute_value = abs(float(value))
    if absolute_value >= 10:
        formatted_value = f"{float(value):.1f}"
    elif absolute_value >= 1:
        formatted_value = f"{float(value):.3g}"
    else:
        formatted_value = f"{float(value):.3g}"
    return formatted_value.rstrip("0").rstrip(".")


def build_interaction_screen(
    merged_dataframe: pd.DataFrame,
    profile_result,
    outcome: str,
    candidate_variables: list[str],
) -> list[dict[str, Any]]:
    """Return compact exploratory high-high interaction screens."""
    numeric_variables = get_variables_by_type(profile_result, {"continuous"})
    numeric_candidates = [
        variable
        for variable in candidate_variables
        if variable in numeric_variables and variable in merged_dataframe.columns
    ][:6]
    outcome_values = pd.to_numeric(merged_dataframe[outcome], errors="coerce")
    rows = []

    for first_index, first_variable in enumerate(numeric_candidates):
        for second_variable in numeric_candidates[first_index + 1 :]:
            first_values = pd.to_numeric(merged_dataframe[first_variable], errors="coerce")
            second_values = pd.to_numeric(merged_dataframe[second_variable], errors="coerce")
            usable_data = pd.DataFrame(
                {
                    "first": first_values,
                    "second": second_values,
                    "outcome": outcome_values,
                }
            ).dropna()
            if len(usable_data) < 10:
                continue

            first_threshold = usable_data["first"].quantile(0.75)
            second_threshold = usable_data["second"].quantile(0.75)
            high_high_mask = (
                (usable_data["first"] >= first_threshold)
                & (usable_data["second"] >= second_threshold)
            )
            high_high_count = int(high_high_mask.sum())
            if high_high_count < 3:
                continue

            high_high_mean = usable_data.loc[high_high_mask, "outcome"].mean()
            other_mean = usable_data.loc[~high_high_mask, "outcome"].mean()
            rows.append(
                {
                    "first_variable": first_variable,
                    "second_variable": second_variable,
                    "first_high_threshold": round_or_none(first_threshold),
                    "second_high_threshold": round_or_none(second_threshold),
                    "high_high_count": high_high_count,
                    "high_high_outcome_mean": round_or_none(high_high_mean),
                    "other_outcome_mean": round_or_none(other_mean),
                    "delta_high_high_vs_other": round_or_none(high_high_mean - other_mean),
                    "interpretation_status": "exploratory_screen_only",
                }
            )

    rows = sorted(
        rows,
        key=lambda row: abs(row["delta_high_high_vs_other"] or 0.0),
        reverse=True,
    )
    return rows[:TOP_INTERACTION_COUNT]


def build_allowed_categorical_levels(
    merged_dataframe: pd.DataFrame,
    profile_result,
    max_levels_per_variable: int = 30,
) -> dict[str, list[str]]:
    """List allowed categorical levels so validators can catch mixed-up labels."""
    categorical_variables = get_variables_by_type(profile_result, {"categorical", "binary"})
    levels: dict[str, list[str]] = {}
    for variable in categorical_variables:
        if variable not in merged_dataframe.columns:
            continue
        unique_levels = (
            merged_dataframe[variable]
            .dropna()
            .astype(str)
            .sort_values()
            .unique()
            .tolist()
        )
        levels[variable] = unique_levels[:max_levels_per_variable]
    return levels


def get_variables_by_type(profile_result, types: set[str]) -> set[str]:
    """Return process variable names matching profile type labels."""
    return set(
        profile_result.variable_types[
            profile_result.variable_types["type"].isin(types)
        ]["column"].tolist()
    )


def sanitize_validation_summary(validation_result: dict[str, Any]) -> dict[str, Any]:
    """Return JSON-safe validation metadata without internal indices."""
    return {
        key: round_or_none(value) if isinstance(value, float) else value
        for key, value in validation_result.items()
        if key not in {"train_index", "test_index"}
    }


def classify_correlation_direction(correlation: float | None) -> str:
    """Return a plain direction hint for linear correlations."""
    if correlation is None or pd.isna(correlation):
        return "not_available"
    if correlation >= 0.25:
        return "positive_linear_association"
    if correlation <= -0.25:
        return "negative_linear_association"
    return "weak_or_non_linear_signal"


def dataframe_to_records(dataframe: pd.DataFrame, max_rows: int | None = None) -> list[dict[str, Any]]:
    """Convert a DataFrame to JSON-safe records."""
    if dataframe is None or dataframe.empty:
        return []
    # `if max_rows` treats an explicit 0 as "no limit" and returns every row.
    output_dataframe = (
        dataframe.head(max_rows).copy() if max_rows is not None else dataframe.copy()
    )
    return output_dataframe.astype(object).where(pd.notna(output_dataframe), None).to_dict("records")


def round_or_none(value: Any, digits: int = 3) -> float | None:
    """Round values for JSON summaries while preserving missing values."""
    try:
        if pd.isna(value):
            return None
        return round(float(value), digits)
    except (TypeError, ValueError):
        return None
