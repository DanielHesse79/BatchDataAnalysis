"""Driver confidence breakdowns for transparent UI explanations."""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.methods import USABLE_VALIDATION_R2, get_best_validation_r2


MODEL_SCORE_THRESHOLDS = {
    "strong": 0.50,
    "moderate": 0.20,
}


def build_driver_confidence_breakdown(
    ranked_driver_row: pd.Series,
    analysis_results: dict[str, Any],
    profile_result,
    audit_result=None,
) -> pd.DataFrame:
    """Return a readable confidence breakdown for one ranked driver."""
    outcome = ranked_driver_row["outcome"]
    variable = ranked_driver_row["process_variable"]
    rows = [
        score_component(
            "Method agreement",
            summarize_method_agreement(ranked_driver_row),
            "Stronger when PLS and Random Forest both identify the variable.",
        ),
        score_component(
            "PLS signal",
            classify_score(float(ranked_driver_row.get("pls_score", 0.0))),
            "Linear/latent-variable relationship with the selected outcome.",
        ),
        score_component(
            "Random Forest signal",
            classify_score(float(ranked_driver_row.get("rf_score", 0.0))),
            "Non-linear model contribution after permutation/group importance.",
        ),
        score_component(
            "PCA context",
            classify_score(float(ranked_driver_row.get("pca_score", 0.0))),
            "Overall process-variation signal; useful context, not outcome-specific proof.",
        ),
        score_component(
            "Model validation",
            summarize_model_validation(analysis_results, outcome),
            "Validation scores matter more than training fit.",
        ),
        score_component(
            "Sample size",
            summarize_sample_size(profile_result.matched_batch_count),
            "Small datasets make driver rankings less stable.",
        ),
        score_component(
            "Missingness",
            summarize_variable_missingness(profile_result, variable),
            "High missingness can weaken or bias driver evidence.",
        ),
    ]

    if audit_result is not None:
        rows.append(
            score_component(
                "Audit cautions",
                summarize_audit_cautions(audit_result, variable),
                "Leakage, confounding, drift, and outliers reduce confidence.",
            )
        )

    return pd.DataFrame(rows)


def score_component(component: str, assessment: str, why_it_matters: str) -> dict[str, str]:
    """Build one confidence table row."""
    return {
        "component": component,
        "assessment": assessment,
        "why_it_matters": why_it_matters,
    }


def summarize_method_agreement(ranked_driver_row: pd.Series) -> str:
    """Summarize which methods support a driver."""
    evidence_methods = str(ranked_driver_row.get("evidence_methods", "none"))
    confidence = str(ranked_driver_row.get("confidence", "exploratory"))
    if evidence_methods == "none":
        return f"{confidence}: no individual method crossed the strong-signal threshold."
    return f"{confidence}: supported by {evidence_methods}."


def classify_score(score: float) -> str:
    """Classify a normalized method score."""
    if score >= MODEL_SCORE_THRESHOLDS["strong"]:
        return f"strong ({score:.3f})"
    if score >= MODEL_SCORE_THRESHOLDS["moderate"]:
        return f"moderate ({score:.3f})"
    return f"weak/exploratory ({score:.3f})"


def summarize_model_validation(analysis_results: dict[str, Any], outcome: str) -> str:
    """Summarize available validation scores for one outcome."""
    pls_result = analysis_results["pls"].get(outcome, {})
    rf_result = analysis_results["random_forest"].get(outcome, {})
    pls_q2 = format_score(pls_result.get("q2"))
    rf_oob = format_score(rf_result.get("oob_r2"))
    pls_test = format_score(pls_result.get("time_ordered_validation", {}).get("test_r2"))
    rf_test = format_score(rf_result.get("time_ordered_validation", {}).get("test_r2"))

    best_validation_r2 = get_best_validation_r2(pls_result, rf_result)
    gate_note = (
        "No model predicted held-out batches well enough "
        f"(best {format_score(best_validation_r2)} < {USABLE_VALIDATION_R2}), "
        "so every driver for this outcome is capped at exploratory."
        if not (
            best_validation_r2 is not None
            and pd.notna(best_validation_r2)
            and float(best_validation_r2) >= USABLE_VALIDATION_R2
        )
        else f"Best held-out score {format_score(best_validation_r2)} clears the "
        f"{USABLE_VALIDATION_R2} bar needed for a confident label."
    )

    return (
        f"PLS Q2 {pls_q2} (mean across folds); RF OOB R2 {rf_oob}; "
        f"time-ordered test R2 PLS {pls_test}, RF {rf_test}. {gate_note}"
    )


def summarize_sample_size(batch_count: int) -> str:
    """Return practical sample-size confidence."""
    if batch_count < 30:
        return f"low ({batch_count} batches)"
    if batch_count < 80:
        return f"moderate ({batch_count} batches)"
    return f"stronger ({batch_count} batches)"


def summarize_variable_missingness(profile_result, variable: str) -> str:
    """Summarize missingness for one process variable."""
    missingness = profile_result.missingness
    if missingness is None or missingness.empty or variable not in missingness["column"].tolist():
        return "not available"

    row = missingness[missingness["column"] == variable].iloc[0]
    missing_percent = float(row["missing_percent"])
    if missing_percent > 20:
        return f"high missingness ({missing_percent:.1f}%)"
    if missing_percent > 5:
        return f"some missingness ({missing_percent:.1f}%)"
    return f"low missingness ({missing_percent:.1f}%)"


def summarize_audit_cautions(audit_result, variable: str) -> str:
    """Summarize audit cautions touching a variable."""
    caution_labels = []
    for table_name, label in [
        ("leakage_name_warnings", "leakage-name warning"),
        ("date_or_drift_columns", "date/drift warning"),
        ("confounding_categoricals", "possible confounding"),
        ("outlier_flags", "outlier flag"),
    ]:
        table = getattr(audit_result, table_name, pd.DataFrame())
        if not table.empty and "column" in table.columns and variable in table["column"].tolist():
            caution_labels.append(label)

    if not caution_labels:
        return "no direct audit caution for this variable"
    return ", ".join(caution_labels)


def format_score(value: Any) -> str:
    """Format a model score for compact display."""
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"
