from analysis.report_validator import validate_interpretation_text


HEADINGS = """
## Executive Summary
## Top Drivers by Outcome
## Specs and Operating-Window Notes
## Cross-Cutting Process Patterns
## Hypotheses to Investigate Next
## Data Quality and Confidence Notes
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
