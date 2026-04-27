"""Deterministic key findings for the Streamlit UI.

The LLM is useful for narrative, but the app should also show concise facts
computed directly from the analysis results. These findings are deliberately
conservative and repeat the key caveats in machine-checkable form.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from analysis.evidence import build_report_pack


KEY_FINDING_COLUMNS = [
    "area",
    "outcome",
    "finding",
    "confidence",
    "source",
    "caveat",
]


def build_deterministic_key_findings(
    profile_result,
    audit_result,
    analysis_results: dict[str, Any],
    merged_dataframe: pd.DataFrame,
    outcomes: list[str],
    spec_assessment=None,
) -> pd.DataFrame:
    """Return compact, Python-generated findings for display before LLM output."""
    report_pack = build_report_pack(
        profile_result=profile_result,
        audit_result=audit_result,
        analysis_results=analysis_results,
        merged_dataframe=merged_dataframe,
        outcomes=outcomes,
        spec_assessment=spec_assessment,
    )

    rows: list[dict[str, str]] = []
    for outcome in outcomes:
        outcome_pack = report_pack["outcomes"].get(outcome, {})
        rows.extend(build_driver_findings(outcome, outcome_pack))
        rows.extend(build_categorical_findings(outcome, outcome_pack))
        rows.extend(build_response_band_findings(outcome, outcome_pack))
        rows.extend(build_interaction_findings(outcome, outcome_pack))
        rows.extend(build_validation_findings(outcome, outcome_pack))

    rows.extend(build_spec_findings(spec_assessment))
    rows.extend(build_audit_findings(audit_result))

    if not rows:
        return pd.DataFrame(columns=KEY_FINDING_COLUMNS)

    return pd.DataFrame(rows, columns=KEY_FINDING_COLUMNS)


def build_driver_findings(outcome: str, outcome_pack: dict[str, Any]) -> list[dict[str, str]]:
    """Build top-driver facts for one outcome."""
    top_drivers = outcome_pack.get("top_drivers", [])
    if not top_drivers:
        return []

    first_driver = top_drivers[0]
    score = format_number(first_driver.get("combined_score"))
    evidence = first_driver.get("evidence_methods") or "no strong individual method"
    return [
        finding_row(
            area="Driver ranking",
            outcome=outcome,
            finding=(
                f"`{first_driver.get('process_variable')}` is the top ranked associated driver "
                f"(score {score}; evidence: {evidence})."
            ),
            confidence=str(first_driver.get("confidence", "exploratory")),
            source="combined PCA/PLS/RF ranking",
            caveat="Association in historical data, not proof of causation.",
        )
    ]


def build_categorical_findings(
    outcome: str,
    outcome_pack: dict[str, Any],
) -> list[dict[str, str]]:
    """Build categorical level-effect facts."""
    rows = []
    for effect in outcome_pack.get("categorical_level_effects", [])[:2]:
        level_effects = effect.get("largest_level_effects", [])
        if not level_effects:
            continue
        strongest_level = level_effects[0]
        rows.append(
            finding_row(
                area="Categorical pattern",
                outcome=outcome,
                finding=(
                    f"`{effect.get('variable')}` has a notable level pattern: "
                    f"{strongest_level.get('level_label')} had mean {format_number(strongest_level.get('mean'))} "
                    f"(delta {format_number(strongest_level.get('delta_from_overall_mean'))} from overall mean)."
                ),
                confidence="exploratory",
                source="grouped historical means",
                caveat="Levels may be confounded with time, campaign, equipment, or operator patterns.",
            )
        )
    return rows


def build_response_band_findings(
    outcome: str,
    outcome_pack: dict[str, Any],
) -> list[dict[str, str]]:
    """Build historical response-band findings."""
    rows = []
    for response_band in outcome_pack.get("historical_response_bands", [])[:2]:
        refined_label = response_band.get("refined_range_label")
        refined_text = f" Narrower observed bin: {refined_label}." if refined_label else ""
        rows.append(
            finding_row(
                area="Response band",
                outcome=outcome,
                finding=(
                    f"`{response_band.get('variable')}` has a historical {response_band.get('pattern_type')} "
                    f"band at {response_band.get('range_label')} for objective '{response_band.get('objective')}'."
                    f"{refined_text}"
                ),
                confidence="hypothesis",
                source="quartile and refined-bin outcome means",
                caveat="Historical sweet-spot hypothesis, not an optimized setpoint or design space.",
            )
        )
    return rows


def build_interaction_findings(
    outcome: str,
    outcome_pack: dict[str, Any],
) -> list[dict[str, str]]:
    """Build exploratory interaction findings."""
    rows = []
    for interaction in outcome_pack.get("exploratory_interactions", [])[:2]:
        rows.append(
            finding_row(
                area="Interaction screen",
                outcome=outcome,
                finding=(
                    f"High `{interaction.get('first_variable')}` plus high `{interaction.get('second_variable')}` "
                    f"changed the mean outcome by {format_number(interaction.get('delta_high_high_vs_other'))} "
                    f"versus other batches."
                ),
                confidence="hypothesis",
                source="high-high quartile comparison",
                caveat="Simple screen only; confirm with plots, SHAP-style interaction checks, or designed experiments.",
            )
        )
    return rows


def build_validation_findings(
    outcome: str,
    outcome_pack: dict[str, Any],
) -> list[dict[str, str]]:
    """Add validation cautions when model quality is weak or unavailable."""
    model_quality = outcome_pack.get("model_quality", {})
    pls_q2 = model_quality.get("pls_cv_q2")
    rf_oob = model_quality.get("rf_oob_r2")
    if score_is_usable(pls_q2) or score_is_usable(rf_oob):
        return []

    return [
        finding_row(
            area="Validation caution",
            outcome=outcome,
            finding=(
                f"Validation support is weak or unavailable for `{outcome}` "
                f"(PLS Q2 {format_number(pls_q2)}, RF OOB R2 {format_number(rf_oob)})."
            ),
            confidence="caution",
            source="model validation metrics",
            caveat="Treat driver ranking as exploratory until validation improves.",
        )
    ]


def build_spec_findings(spec_assessment) -> list[dict[str, str]]:
    """Build concise spec/window findings."""
    if spec_assessment is None or not getattr(spec_assessment, "has_specs", False):
        return []

    variable_summary = getattr(spec_assessment, "variable_summary", pd.DataFrame())
    if variable_summary.empty:
        return []

    review_rows = variable_summary[
        ~variable_summary["classification"].isin(["Looks reasonable", "Reference only"])
    ].head(5)
    rows = []
    for _, spec_row in review_rows.iterrows():
        rows.append(
            finding_row(
                area="Specs/windows",
                outcome=str(spec_row.get("role", "")),
                finding=(
                    f"`{spec_row.get('variable')}` is classified as {spec_row.get('classification')}: "
                    f"{spec_row.get('reason')}"
                ),
                confidence="heuristic",
                source="supplied spec/window file",
                caveat="Historical classification only; not formal validation or change-control advice.",
            )
        )
    return rows


def build_audit_findings(audit_result) -> list[dict[str, str]]:
    """Build a short audit caution when warnings exist."""
    if audit_result is None or not getattr(audit_result, "warnings", []):
        return []

    return [
        finding_row(
            area="Data audit",
            outcome="all outcomes",
            finding=f"{len(audit_result.warnings)} audit warning(s) may affect confidence.",
            confidence="caution",
            source="pre-flight audit",
            caveat="Review leakage, confounding, drift, outlier, and missingness tables before acting.",
        )
    ]


def finding_row(
    area: str,
    outcome: str,
    finding: str,
    confidence: str,
    source: str,
    caveat: str,
) -> dict[str, str]:
    """Build one display row."""
    return {
        "area": area,
        "outcome": outcome,
        "finding": finding,
        "confidence": confidence,
        "source": source,
        "caveat": caveat,
    }


def score_is_usable(value: Any, threshold: float = 0.15) -> bool:
    """Return whether a validation metric gives at least weak support."""
    try:
        if pd.isna(value):
            return False
        return float(value) >= threshold
    except (TypeError, ValueError):
        return False


def format_number(value: Any) -> str:
    """Format numbers without throwing on missing values."""
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.3f}"
    except (TypeError, ValueError):
        return "n/a"
