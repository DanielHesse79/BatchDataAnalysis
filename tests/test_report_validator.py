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
