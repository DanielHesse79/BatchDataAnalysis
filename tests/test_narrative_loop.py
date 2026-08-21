"""Tests for the report repair loop.

The loop is driven with a fake generator rather than Ollama, so the escalation
logic is tested rather than a model's mood.
"""

import pytest

from analysis.narrative_loop import (
    Attempt,
    LoopResult,
    generate_validated_report,
    missing_headings,
    plan_attempt,
)
from analysis.report_validator import REQUIRED_HEADINGS


PACK = {
    "allowed_variables": {
        "process_columns": ["temperature_C"],
        "outcome_columns": ["yield_g_L", "moisture_percent"],
        "categorical_levels": {},
    },
    "specs_and_windows": {"available": False},
    "outcomes": {"yield_g_L": {"mean": 41.53}},
}

BODY = "Feed rate is associated with yield, and moisture stayed inside the window.\n"
CLEAN_REPORT = "".join(f"{heading}\n{BODY}\n" for heading in REQUIRED_HEADINGS)
MISSING_HEADING_REPORT = "".join(
    f"{heading}\n{BODY}\n" for heading in REQUIRED_HEADINGS[:-1]
)
UNGROUNDED_REPORT = CLEAN_REPORT + "\nYield improved by 87.4% after the change.\n"


def scripted_generator(*reports):
    """A stand-in for Ollama that returns each report in turn."""
    remaining = list(reports)
    calls = []

    def generate(**kwargs):
        calls.append(kwargs)
        return [remaining.pop(0) if remaining else reports[-1]]

    generate.calls = calls
    return generate


def test_a_clean_first_attempt_costs_one_call():
    generate = scripted_generator(CLEAN_REPORT)

    result = generate_validated_report(PACK, model="test", generate=generate)

    assert result.accepted
    assert len(generate.calls) == 1
    assert result.strategy == "base"


def test_the_first_attempt_uses_the_unmodified_prompt():
    """The loop must not change what a passing report would have looked like."""
    generate = scripted_generator(CLEAN_REPORT)

    generate_validated_report(PACK, model="test", generate=generate)

    assert generate.calls[0]["messages"] is None
    assert generate.calls[0]["option_overrides"] is None


def test_an_ungrounded_number_triggers_a_repair_attempt():
    generate = scripted_generator(UNGROUNDED_REPORT, CLEAN_REPORT)

    result = generate_validated_report(PACK, model="test", generate=generate)

    assert result.accepted
    assert [attempt.strategy for attempt in result.attempts] == ["base", "repair"]
    # A repair is a correction, not another sample.
    assert generate.calls[1]["option_overrides"]["temperature"] == 0.0
    assert generate.calls[1]["messages"][-1]["role"] == "user"


def test_a_missing_heading_goes_straight_to_the_skeleton():
    """Feeding the text back rarely restores structure; handing it over does."""
    generate = scripted_generator(MISSING_HEADING_REPORT, CLEAN_REPORT)

    result = generate_validated_report(PACK, model="test", generate=generate)

    assert [attempt.strategy for attempt in result.attempts] == ["base", "skeleton"]
    skeleton_prompt = generate.calls[1]["messages"][-1]["content"]
    for heading in REQUIRED_HEADINGS:
        assert heading in skeleton_prompt


def test_a_loop_that_never_succeeds_returns_its_warnings():
    """The failure mode that matters: never present a bad report as clean."""
    generate = scripted_generator(UNGROUNDED_REPORT)

    result = generate_validated_report(PACK, model="test", generate=generate)

    assert not result.accepted
    assert result.warnings
    assert "not a clean one" in result.describe()


def test_the_best_attempt_is_kept_when_a_repair_makes_things_worse():
    worse = "## Executive Summary\nToo short.\n"
    generate = scripted_generator(UNGROUNDED_REPORT, worse, worse)

    result = generate_validated_report(PACK, model="test", generate=generate)

    # The first attempt is kept: one ungrounded number beats a report that has
    # lost five of its six sections.
    assert "87.4" in result.text
    assert "Too short" not in result.text
    assert not result.accepted


def test_the_loop_is_bounded():
    generate = scripted_generator(UNGROUNDED_REPORT)

    generate_validated_report(PACK, model="test", generate=generate, max_attempts=3)

    assert len(generate.calls) == 3


def test_the_skeleton_prompt_names_outcomes_that_were_dropped():
    dropped = "".join(
        f"{heading}\nYield rose steadily over the campaign.\n\n"
        for heading in REQUIRED_HEADINGS
    )
    generate = scripted_generator(dropped, dropped, CLEAN_REPORT)

    generate_validated_report(PACK, model="test", generate=generate)

    skeleton_prompt = generate.calls[-1]["messages"][-1]["content"]
    assert "moisture_percent" in skeleton_prompt


def test_missing_headings_reports_what_is_absent():
    assert missing_headings(CLEAN_REPORT) == []
    assert missing_headings(MISSING_HEADING_REPORT) == [REQUIRED_HEADINGS[-1]]


def test_plan_attempt_never_rewrites_the_first_prompt():
    strategy, messages, overrides = plan_attempt(1, PACK, "", [], ["yield_g_L"])

    assert strategy == "base"
    assert messages is None and overrides is None


def test_the_attempt_log_stays_in_the_order_it_ran():
    """A caption reading "repair then skeleton then base" describes a sequence
    that never happened. The kept attempt is named, not reordered."""
    worse = "## Executive Summary\nToo short.\n"
    generate = scripted_generator(UNGROUNDED_REPORT, worse, worse)

    result = generate_validated_report(PACK, model="test", generate=generate)

    assert [attempt.strategy for attempt in result.attempts][0] == "base"
    assert result.strategy == "base"
    assert "was kept as the best of them" in result.describe()
