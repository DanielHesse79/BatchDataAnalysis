from analysis.report_validator import validate_interpretation_text


# Every section carries a line of text, because a heading with nothing under it
# is now a finding in itself: otherwise the cheapest way to satisfy the heading
# check is to emit the headings and write nothing.
HEADINGS = """
## Executive Summary
Nothing of note was identified for this section of the report.

## Top Drivers by Outcome
Nothing of note was identified for this section of the report.

## Specs and Operating-Window Notes
Nothing of note was identified for this section of the report.

## Cross-Cutting Process Patterns
Nothing of note was identified for this section of the report.

## Hypotheses to Investigate Next
Nothing of note was identified for this section of the report.

## Data Quality and Confidence Notes
Nothing of note was identified for this section of the report.
"""


def test_report_validator_allows_root_cause_hypothesis_wording():
    report_pack = {
        "allowed_variables": {
            "process_columns": ["temperature_C"],
            "outcome_columns": ["yield_g_L"],
            "categorical_levels": {},
        },
        "specs_and_windows": {"available": False},
    }
    text = (
        HEADINGS
        + "\n`temperature_C` is associated with `yield_g_L`.\n"
        + "Root-cause hypotheses should be investigated next.\n"
        + "No spec/window file was supplied."
    )

    result = validate_interpretation_text(text, report_pack)

    assert result.warnings == []


def test_report_validator_flags_unknown_variables_and_action_directives():
    report_pack = {
        "allowed_variables": {
            "process_columns": ["temperature_C"],
            "outcome_columns": ["yield_g_L"],
            "categorical_levels": {},
        },
        "specs_and_windows": {"available": False},
    }
    text = (
        HEADINGS
        + "\n`made_up_variable` directly causes low `yield_g_L`.\n"
        + "You should change the spec and check the upper spec limit."
    )

    result = validate_interpretation_text(text, report_pack)

    assert any("unknown backtick variables" in warning for warning in result.warnings)
    assert any("directly causes" in warning for warning in result.warnings)
    assert any("change the spec" in warning for warning in result.warnings)
    assert any("no spec/window file was supplied" in warning for warning in result.warnings)


# The categorical check used to fire on correct reports. Both shapes below were
# found by benchmarking local models against the real evidence pack: almost every
# generated report collected at least one of them, which made the warning
# meaningless.
CATEGORICAL_PACK = {
    "allowed_variables": {
        "process_columns": ["bioreactor_id", "media_lot", "deviation_reported"],
        "outcome_columns": ["yield_g_L"],
        "categorical_levels": {
            "bioreactor_id": ["BR-1", "BR-3", "BR-5"],
            "media_lot": ["RM-004", "RM-005"],
            # Dummy-coded, so its levels are bare numbers.
            "deviation_reported": ["0", "1.0"],
        },
    },
    "specs_and_windows": {"available": False},
}


def test_statistics_are_not_read_as_categorical_levels():
    """`Q2=0` is a model-quality number, not a claim about deviation_reported."""
    text = (
        HEADINGS
        + "\nPLS cross-validated `Q2=0` and R2=0 for `yield_g_L`.\n"
        + "The mean=1.0 across batches.\n"
        + "No spec/window file was supplied."
    )

    result = validate_interpretation_text(text, CATEGORICAL_PACK)

    assert not [w for w in result.warnings if "categorical mix-up" in w.lower()]


def test_a_differently_cased_variable_name_is_not_a_mix_up():
    """A model that writes bioreactor_ID has the variable right."""
    text = (
        HEADINGS
        + "\n`bioreactor_ID=BR-5` shows lower `yield_g_L`.\n"
        + "Deviation_reported=1.0 batches ran lower.\n"
        + "No spec/window file was supplied."
    )

    result = validate_interpretation_text(text, CATEGORICAL_PACK)

    assert not [w for w in result.warnings if "categorical mix-up" in w.lower()]


def test_a_level_attached_to_the_wrong_variable_is_still_caught():
    """The check has to keep working: BR-3 is a bioreactor, not a media lot."""
    text = (
        HEADINGS
        + "\n`media_lot=BR-3` is associated with lower `yield_g_L`.\n"
        + "No spec/window file was supplied."
    )

    result = validate_interpretation_text(text, CATEGORICAL_PACK)

    mix_ups = [w for w in result.warnings if "categorical mix-up" in w.lower()]
    assert len(mix_ups) == 1
    assert "media_lot=BR-3" in mix_ups[0]
    assert "bioreactor_id" in mix_ups[0]


# Python computes the facts and the model narrates them, so a number in the
# narrative that is not in the evidence pack is a fabrication, however plausible
# it reads. These three checks were added after benchmarking local models: the
# coverage one caught every model except gpt-oss silently dropping an analysed
# outcome from the mock-spec reports.
GROUNDING_PACK = {
    "allowed_variables": {
        "process_columns": ["temperature_C"],
        "outcome_columns": ["yield_g_L", "moisture_percent"],
        "categorical_levels": {"bioreactor_id": ["BR-3"]},
    },
    "specs_and_windows": {"available": False},
    "outcomes": {"yield_g_L": {"mean": 41.53, "r_squared": 0.87, "n": 150}},
}


def numeric_warnings(text):
    from analysis.report_validator import check_numbers_exist_in_pack

    return check_numbers_exist_in_pack(text, GROUNDING_PACK)


def test_a_number_from_the_pack_is_accepted_at_the_precision_a_person_writes():
    """41.53 in the pack, written 41.5 in the report, is the same number."""
    assert numeric_warnings("Mean yield was 41.5 g/L across 150 batches.") == []


def test_an_invented_number_is_flagged():
    assert numeric_warnings("Yield improved by 23.7% after the change.")


def test_counts_and_ordinals_are_not_treated_as_statistics():
    """`the top 3 drivers` is not a claim about the data."""
    assert numeric_warnings("The top 3 drivers are listed below, in 2 groups.") == []


def test_level_names_carrying_digits_are_left_to_the_categorical_check():
    assert numeric_warnings("Batches from BR-3 ran lower.") == []


def test_a_heading_with_nothing_under_it_is_reported():
    from analysis.report_validator import check_empty_sections

    text = "## Executive Summary\n\n## Top Drivers by Outcome\nA real paragraph of findings goes here."

    warnings = check_empty_sections(text)

    assert len(warnings) == 1
    assert "Executive Summary" in warnings[0]


def test_an_outcome_discussed_in_prose_counts_as_covered():
    """A readable report writes "moisture (%)", not `moisture_percent`."""
    from analysis.report_validator import check_outcome_coverage

    text = "Yield rose over the campaign, and moisture stayed within the window."

    assert check_outcome_coverage(text, GROUNDING_PACK) == []


def test_an_outcome_that_is_never_discussed_is_reported():
    from analysis.report_validator import check_outcome_coverage

    warnings = check_outcome_coverage("Yield rose over the campaign.", GROUNDING_PACK)

    assert len(warnings) == 1
    assert "moisture_percent" in warnings[0]
