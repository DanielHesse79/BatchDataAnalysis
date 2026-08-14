"""Lightweight validation for LLM-generated reports."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any


REQUIRED_HEADINGS = [
    "## Executive Summary",
    "## Top Drivers by Outcome",
    "## Specs and Operating-Window Notes",
    "## Cross-Cutting Process Patterns",
    "## Hypotheses to Investigate Next",
    "## Data Quality and Confidence Notes",
]

CAUSAL_PATTERNS = [
    ("directly causes", r"\bdirectly causes?\b"),
    ("caused by", r"\bcaused by\b"),
    ("causing", r"\bcausing\b"),
    ("will improve", r"\bwill improve\b"),
    ("will reduce", r"\bwill reduce\b"),
    ("will increase", r"\bwill increase\b"),
    ("must reduce", r"\bmust reduce\b"),
    ("must increase", r"\bmust increase\b"),
    ("change the spec", r"\bchange the spec\b"),
    ("adjust the spec", r"\badjust the spec\b"),
    ("widen the spec", r"\bwiden the spec\b"),
    ("narrow the spec", r"\bnarrow the spec\b"),
    ("tighten the spec", r"\btighten the spec\b"),
]

ASSUMED_SPEC_PHRASES = [
    "assumed typical",
    "typical spec",
    "industry spec",
    "release specs (assumed",
    "qc specs (assumed",
]

# Phrases that usually mean a local model leaked its scratchpad. The streaming
# layer used to truncate the report at these, which silently deleted legitimate
# text; they are reported here so the user can judge instead.
SUSPECT_REASONING_PHRASES = [
    "thinking process:",
    "analyze the request:",
    "self-correction",
    "drafting the response:",
    "internal reasoning:",
    "chain of thought:",
]


@dataclass(frozen=True)
class ReportValidationResult:
    """Validation output for a generated report."""

    warnings: list[str]
    blocked: bool = False


def validate_interpretation_text(
    report_text: str,
    report_pack: dict[str, Any],
) -> ReportValidationResult:
    """Check an LLM report for likely hallucination or overclaiming."""
    warnings: list[str] = []
    normalized_text = report_text.lower()

    warnings.extend(check_required_headings(report_text))
    warnings.extend(check_assumed_specs(normalized_text))
    warnings.extend(check_leaked_reasoning(normalized_text))
    warnings.extend(check_causal_or_spec_change_claims(normalized_text))
    warnings.extend(check_unknown_variables(report_text, report_pack))
    warnings.extend(check_mislabeled_categorical_levels(report_text, report_pack))
    warnings.extend(check_spec_limit_mentions(report_text, report_pack))

    return ReportValidationResult(warnings=dedupe_preserve_order(warnings))


def check_required_headings(report_text: str) -> list[str]:
    """Warn when the model skipped expected structure."""
    return [
        f"Missing expected section heading: {heading}"
        for heading in REQUIRED_HEADINGS
        if heading not in report_text
    ]


def check_leaked_reasoning(normalized_text: str) -> list[str]:
    """Warn when the report looks like it contains scratchpad text."""
    return [
        f"The report contains '{phrase}', which often means the model leaked its "
        "reasoning. Check that section before sharing the report."
        for phrase in SUSPECT_REASONING_PHRASES
        if phrase in normalized_text
    ]


def check_assumed_specs(normalized_text: str) -> list[str]:
    """Warn when the model invents assumed spec sections."""
    for phrase in ASSUMED_SPEC_PHRASES:
        if phrase in normalized_text:
            return [
                "The report appears to invent or assume typical specs. Regenerate with a stricter model or review manually."
            ]
    return []


def check_causal_or_spec_change_claims(normalized_text: str) -> list[str]:
    """Warn when output sounds too action-directive for historical data."""
    warnings = []
    for phrase, pattern in CAUSAL_PATTERNS:
        if re.search(pattern, normalized_text):
            warnings.append(
                f"Potential overclaim detected: '{phrase}'. The report should frame this as association/investigation."
            )
    return warnings


def check_unknown_variables(report_text: str, report_pack: dict[str, Any]) -> list[str]:
    """Warn when backtick-style variables are not in supplied columns."""
    allowed_variables = set(report_pack.get("allowed_variables", {}).get("process_columns", []))
    allowed_variables.update(report_pack.get("allowed_variables", {}).get("outcome_columns", []))
    if not allowed_variables:
        return []

    candidate_tokens = set(re.findall(r"`([A-Za-z_][A-Za-z0-9_]*)`", report_text))
    unknown_tokens = sorted(
        token
        for token in candidate_tokens
        if token not in allowed_variables and token not in {"batch_id"}
    )
    if not unknown_tokens:
        return []

    return [
        "The report mentions unknown backtick variables: "
        + ", ".join(unknown_tokens[:8])
        + "."
    ]


def check_mislabeled_categorical_levels(report_text: str, report_pack: dict[str, Any]) -> list[str]:
    """Warn when known categorical levels are attached to the wrong variable name."""
    allowed_levels = report_pack.get("allowed_variables", {}).get("categorical_levels", {})
    if not allowed_levels:
        return []

    level_to_variables: dict[str, set[str]] = {}
    for variable, levels in allowed_levels.items():
        for level in levels:
            level_to_variables.setdefault(str(level), set()).add(variable)

    # Match the levels this pack actually contains rather than a fixed shape such
    # as ABC-123; real level names include Supplier_B, RM_005, and bare numbers.
    # Backticks around either side are optional because models format both ways.
    known_levels = sorted(level_to_variables, key=len, reverse=True)
    if not known_levels:
        return []

    warnings = []
    labeled_mentions = re.findall(
        r"`?([A-Za-z_][A-Za-z0-9_]*)`?\s*=\s*`?("
        + "|".join(re.escape(level) for level in known_levels)
        + r")`?",
        report_text,
    )
    for variable, level in labeled_mentions:
        if variable not in level_to_variables[level]:
            warnings.append(
                f"Possible categorical mix-up: {variable}={level}, but {level} belongs to {', '.join(sorted(level_to_variables[level]))}."
            )

    return warnings


def check_spec_limit_mentions(report_text: str, report_pack: dict[str, Any]) -> list[str]:
    """Warn when specs are discussed although no specs were supplied."""
    specs_pack = report_pack.get("specs_and_windows", {})
    if specs_pack.get("available"):
        return []

    normalized_text = report_text.lower()

    no_spec_statement = (
        re.search(r"\bno\b.{0,80}\bspec", normalized_text)
        or re.search(r"\bspec.{0,80}\bnot (?:provided|supplied)", normalized_text)
        or re.search(r"\bspec.{0,80}\bwere not (?:provided|supplied)", normalized_text)
    )
    if no_spec_statement:
        return []

    risky_spec_terms = [
        "release limit",
        "spec limit",
        "specification limit",
        "upper spec",
        "lower spec",
        "within spec",
        "out of spec",
    ]
    # "oos" needs word boundaries; as a bare substring it matches loose, choose, moose.
    if any(term in normalized_text for term in risky_spec_terms) or re.search(
        r"\boos\b", normalized_text
    ):
        return ["The report discusses specs even though no spec/window file was supplied."]
    return []


def dedupe_preserve_order(values: list[str]) -> list[str]:
    """Remove duplicate warnings while preserving order."""
    seen = set()
    output = []
    for value in values:
        if value not in seen:
            seen.add(value)
            output.append(value)
    return output
