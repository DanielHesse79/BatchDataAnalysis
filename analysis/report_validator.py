"""Lightweight validation for LLM-generated reports."""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any

from analysis.evidence import tokenize_outcome_name


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
    warnings.extend(check_empty_sections(report_text))
    warnings.extend(check_outcome_coverage(report_text, report_pack))
    warnings.extend(check_numbers_exist_in_pack(report_text, report_pack))

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
    """Warn when a known categorical level is attached to the wrong variable.

    Two restrictions keep this from firing on correct reports.

    Only mentions whose left-hand side is itself a known categorical variable
    count. Dummy-coded variables have bare numbers as levels, so without this
    every statistic written as name=number matched one: `Q2=0`, `R2=0` and
    `mean=1.0` were all reported as category mix-ups. A left-hand side that is
    not a categorical variable is check_unknown_variables' business, not this
    check's.

    Names are compared case-insensitively. A model that writes `bioreactor_ID`
    has the variable right, and saying otherwise trains the reader to ignore
    the warnings that matter.
    """
    allowed_levels = report_pack.get("allowed_variables", {}).get("categorical_levels", {})
    if not allowed_levels:
        return []

    canonical_variables: dict[str, str] = {}
    level_owners: dict[str, set[str]] = {}
    for variable, levels in allowed_levels.items():
        canonical_variables[str(variable).lower()] = str(variable)
        for level in levels:
            level_owners.setdefault(str(level), set()).add(str(variable).lower())

    # Match the levels this pack actually contains rather than a fixed shape such
    # as ABC-123; real level names include Supplier_B, RM_005, and bare numbers.
    # Backticks around either side are optional because models format both ways.
    known_levels = sorted(level_owners, key=len, reverse=True)
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
        normalized_variable = variable.lower()
        if normalized_variable not in canonical_variables:
            continue
        if normalized_variable in level_owners[level]:
            continue

        owners = sorted(canonical_variables[owner] for owner in level_owners[level])
        warnings.append(
            f"Possible categorical mix-up: {variable}={level}, but {level} belongs to {', '.join(owners)}."
        )

    return warnings


# --------------------------------------------------------------- numeric grounding
# Python computes the facts and the model narrates them, so every number in the
# narrative should be traceable to the evidence pack. Enforcing that is the
# strictest check available and the one that catches the most dangerous failure:
# a fabricated statistic reads exactly like a real one.
#
# It is deliberately conservative. A check that cries wolf gets ignored, and the
# categorical check already taught that lesson - it fired on `Q2=0` for a year.
# Numbers below this threshold are ordinals, list markers and counts ("the top 3
# drivers"), not statistics.
# Tokens that describe a unit or a measure rather than what was measured. An
# outcome recognised only by these has not really been mentioned.
GENERIC_NAME_TOKENS = {
    "percent", "pct", "ppm", "ppb", "index", "value", "ratio", "per", "mean",
    "avg", "average", "total", "count", "level", "score", "g", "l", "ml", "mg",
    "kg", "h", "hr", "hrs", "c", "k", "min", "sec", "unit", "units",
}

SMALLEST_CHECKED_NUMBER = 10.0
# Digits with optional thousands groups, decimals and exponent. Whether a match
# is a number at all - not part of `day7`, BR-3 or a decimal's tail - is decided
# by the characters around it; see extract_numbers.
NUMBER_TOKEN = re.compile(r"\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\d,])|\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")
ORDINAL_SUFFIX = re.compile(r"(?:st|nd|rd|th)\b", re.IGNORECASE)
# A number the report itself marks as approximate, such as "~32 ppm" for a mean
# of 31.4, is allowed one unit of slack in its last digit.
APPROXIMATE_MARKER = re.compile(
    r"(?:~|≈|\b(?:approx\.?|approximately|about|around|roughly|nearly))\s*-?$",
    re.IGNORECASE,
)
MAX_REPORTED_NUMBERS_LISTED = 6
# A section shorter than this is empty in substance whatever its heading says.
# Kept low on purpose: a terse but real section - "No spec/window file was
# supplied." - is legitimate, and the check exists to catch nothing at all, not
# brevity. It fires on none of the 48 benchmarked reports.
MINIMUM_SECTION_CHARACTERS = 25


def collect_pack_numbers(value: Any, found: set[float] | None = None) -> set[float]:
    """Every number anywhere in the evidence pack, including inside strings."""
    if found is None:
        found = set()

    if isinstance(value, bool):
        return found
    if isinstance(value, (int, float)):
        found.add(float(value))
    elif isinstance(value, str):
        for number in extract_numbers(value):
            found.add(number.value)
    elif isinstance(value, dict):
        for item in value.values():
            collect_pack_numbers(item, found)
    elif isinstance(value, (list, tuple)):
        for item in value:
            collect_pack_numbers(item, found)

    return found


@dataclass(frozen=True)
class NumberInText:
    """One number as written in a report."""

    literal: str
    value: float
    decimals: int
    signed: bool
    approximate: bool = False


def extract_numbers(text: str) -> list[NumberInText]:
    """Every number written in ``text``, read the way a person would read it.

    - ``12-48`` and ``36.5-37.5`` are ranges: both ends are numbers, neither is
      negative. A minus sign counts only after a space, bracket or operator.
    - ``120rpm``, ``37C`` and ``5x`` are numbers with a unit attached;
      ``3rd`` and ``10th`` are ordinals and are skipped.
    - ``1,250`` is one thousand two hundred and fifty, and ``1.2e5`` is 120000.
    - Digits inside a name - ``Q2``, ``day7``, ``x10``, ``BR-3``, ``RM-005``,
      ``titer_day7_g_L`` - are not numbers.
    """
    numbers = []
    for match in NUMBER_TOKEN.finditer(text):
        start, end = match.span()
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""

        if before.isalpha() or before in ("_", "."):
            continue
        if after == "_" or (after == "." and text[end + 1 : end + 2].isdigit()):
            continue

        literal = match.group(0)
        signed = False
        if before == "-":
            ahead_of_sign = text[start - 2] if start > 1 else ""
            if ahead_of_sign.isalpha() or ahead_of_sign == "_":
                continue  # BR-3, RM-005: a level name, not a number
            signed = not (ahead_of_sign.isdigit() or ahead_of_sign in (")", "%"))

        if after.isalpha() and ORDINAL_SUFFIX.match(text, end) and "." not in literal:
            continue

        plain = literal.replace(",", "")
        mantissa = re.split(r"[eE]", plain)[0]
        decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
        if re.search(r"[eE]", plain):
            decimals = max(0, decimals - int(re.split(r"[eE]", plain)[1]))
        value = float(plain)
        numbers.append(
            NumberInText(
                literal=("-" if signed else "") + literal,
                value=-value if signed else value,
                decimals=decimals,
                signed=signed,
                approximate=bool(APPROXIMATE_MARKER.search(text[max(0, start - 16) : start])),
            )
        )
    return numbers


def number_is_grounded(
    reported: float,
    pack_numbers: set[float],
    decimals: int,
    signed: bool = True,
    approximate: bool = False,
) -> bool:
    """Whether a reported number matches something the pack actually contains.

    A pack value counts as a match when rounding it to the precision the report
    used produces the reported number - which is how a person writes 8.15 as
    8.2. Percentages are checked in both directions because a pack may hold a
    fraction where the report writes a percent; the precision moves with the
    decimal point, so 47% has to match a pack value that rounds to 0.47. It
    used to be compared at the report's precision, where 0.47 rounds to 0 and
    matched every small value in the pack: 0 of 500 random integers were
    flagged against a real evidence pack.

    An unsigned number may match a negative pack value, because "a decrease of
    5.6" is how a person writes a delta of -5.6. A number the report writes
    with a minus sign has to match the sign as well.

    A whole number may also sit within half a percent of a pack value, which
    is how 1248 gets written as 1,250. A number with decimals gets no such
    slack: its precision already says how it was rounded, and the 0.5% band let
    a garbled 6.994 pass as the 7.0 next to it.

    Truncating counts as well as rounding, at the pack value's own scale. Local
    models write 277.879 as 277.8 and 87.996 as 87.99: across seven reports from
    four local models, seven of the eight numbers that rounding alone flagged
    were truncations, and flagging them would teach the reader to ignore the
    warning. A percent truncated from a fraction is not accepted - 0.638 does
    not ground "63%" - because that doubles the window for every number.

    A number the report marks as approximate ("~31", "about 31") may be off by
    one unit in its last digit in either direction.
    """
    candidates = (
        (reported, decimals),
        (reported / 100.0, decimals + 2),
        (reported * 100.0, max(decimals - 2, 0)),
    )
    for candidate, precision in candidates:
        same_scale = candidate == reported
        scale = 10.0 ** precision
        for pack_number in pack_numbers:
            compared = pack_number if signed else abs(pack_number)
            if round(compared, precision) == round(candidate, precision):
                return True
            if same_scale:
                truncated = math.trunc(compared * scale + math.copysign(1e-9, compared)) / scale
                if round(truncated, precision) == round(candidate, precision):
                    return True
            if approximate and abs(compared - candidate) < 10.0 ** -precision:
                return True
            if (
                decimals == 0
                and abs(compared) > 1e-9
                and abs(compared - candidate) / abs(compared) < 0.005
            ):
                return True
    return False


def check_numbers_exist_in_pack(report_text: str, report_pack: dict[str, Any]) -> list[str]:
    """Warn when the report states a number the evidence pack does not contain."""
    if not report_pack:
        return []

    pack_numbers = collect_pack_numbers(report_pack)
    if not pack_numbers:
        return []

    # Level names such as BR-3 and RM-005 carry digits that are not statistics;
    # the categorical check owns those.
    level_text = " ".join(
        str(level)
        for levels in report_pack.get("allowed_variables", {}).get("categorical_levels", {}).values()
        for level in levels
    )
    level_numbers = collect_pack_numbers(level_text)

    ungrounded: list[str] = []
    for number in extract_numbers(report_text):
        # Small whole numbers are counts, ordinals and list markers ("the top 3
        # drivers"). Anything with a decimal point is a statistic - R2, Q2,
        # Ppk, a correlation - and those are exactly the figures a reader takes
        # on trust, so they are checked whatever their size.
        if number.decimals == 0 and abs(number.value) < SMALLEST_CHECKED_NUMBER:
            continue
        if number.value in level_numbers or abs(number.value) in level_numbers:
            continue

        if not number_is_grounded(
            number.value, pack_numbers, number.decimals, number.signed, number.approximate
        ):
            if number.literal not in ungrounded:
                ungrounded.append(number.literal)

    if not ungrounded:
        return []

    listed = ", ".join(ungrounded[:MAX_REPORTED_NUMBERS_LISTED])
    suffix = " and others" if len(ungrounded) > MAX_REPORTED_NUMBERS_LISTED else ""
    return [
        f"The report states {len(ungrounded)} number(s) that are not in the evidence "
        f"pack: {listed}{suffix}. Every figure should come from the analysis, not "
        "from the model."
    ]


def check_empty_sections(report_text: str) -> list[str]:
    """Warn when a required heading has nothing of substance under it.

    Without this, the cheapest way to satisfy check_required_headings is to emit
    the headings and write nothing - which matters as soon as a repair loop is
    optimising against these checks.
    """
    warnings = []
    for heading in REQUIRED_HEADINGS:
        position = report_text.find(heading)
        if position < 0:
            continue  # check_required_headings reports it as missing

        body_start = position + len(heading)
        next_positions = [
            report_text.find(other, body_start)
            for other in REQUIRED_HEADINGS
            if report_text.find(other, body_start) > -1
        ]
        body_end = min(next_positions) if next_positions else len(report_text)
        body = report_text[body_start:body_end].strip()

        if len(body) < MINIMUM_SECTION_CHARACTERS:
            warnings.append(
                f"Section '{heading.lstrip('# ').strip()}' has a heading but no "
                "content under it."
            )

    return warnings


def check_outcome_coverage(report_text: str, report_pack: dict[str, Any]) -> list[str]:
    """Warn when an analysed outcome is never discussed.

    Matched on distinctive tokens rather than the exact column name. A readable
    report writes "host cell protein (HCP) ppm", not `hcp_ppm`, and demanding the
    literal name would flag almost every well-written report - measured at 40 of
    48 on the model benchmark before this was loosened.
    """
    outcomes = report_pack.get("allowed_variables", {}).get("outcome_columns", [])
    if not outcomes:
        return []

    lowercase_report = report_text.lower()
    missing = []
    for outcome in outcomes:
        distinctive = [
            token for token in tokenize_outcome_name(outcome)
            if token not in GENERIC_NAME_TOKENS and len(token) > 1
        ]
        if not distinctive:
            continue  # nothing to look for but units
        if not any(token in lowercase_report for token in distinctive):
            missing.append(str(outcome))

    if not missing:
        return []

    return [
        "The report never discusses " + ", ".join(missing[:6])
        + ". Every analysed outcome should appear."
    ]


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
