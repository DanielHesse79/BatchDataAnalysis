"""End-to-end checks that the Streamlit script actually renders.

The unit tests cover `analysis/` and `utils/` but never execute `app.py`, so a
broken import, a missing entry point, or a renamed helper could pass the whole
suite and still leave a blank page.
"""

from pathlib import Path

import pytest

from streamlit.testing.v1 import AppTest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
APP_PATH = PROJECT_ROOT / "app.py"
HOME_PATH = PROJECT_ROOT / "home.py"
STARTUP_TIMEOUT_SECONDS = 60


@pytest.fixture(scope="module")
def started_app() -> AppTest:
    app_test = AppTest.from_file(str(APP_PATH), default_timeout=STARTUP_TIMEOUT_SECONDS)
    app_test.run()
    return app_test


@pytest.fixture(scope="module")
def started_home() -> AppTest:
    app_test = AppTest.from_file(str(HOME_PATH), default_timeout=STARTUP_TIMEOUT_SECONDS)
    app_test.run()
    return app_test


def test_shared_workspace_chooser_runs_without_raising(started_home):
    assert not started_home.exception


def test_shared_workspace_chooser_exposes_both_analysis_paths(started_home):
    headings = [element.value for element in started_home.subheader]
    links = [element.label for element in started_home.get("page_link")]

    assert headings == ["Batch Insight Analyzer", "QC Intelligence Layer"]
    assert links == ["Open batch driver analysis", "Open QC monitoring"]


def test_app_runs_without_raising(started_app):
    """A dropped `if __name__ == "__main__"` guard renders an empty page."""
    assert not started_app.exception


def test_app_renders_its_first_stage(started_app):
    """The upload stage must appear, or main() never ran."""
    headings = [element.value for element in started_app.subheader]

    assert "Upload and prepare source data" in headings


def test_app_waits_for_both_files_before_matching(started_app):
    """Without uploads the app should stop at the upload prompt, not error."""
    info_messages = [element.value for element in started_app.info]

    assert any("Upload both files" in message for message in info_messages)


@pytest.mark.parametrize(
    "state_key",
    [
        "merged_dataframe",
        "merge_result",
        "analysis_results",
        "analysis_input_fingerprint",
        "analysis_run_token",
        "spec_assessment",
        "selected_outcome_columns",
    ],
)
def test_session_state_is_initialized_for_the_pipeline(started_app, state_key):
    """Later stages read these keys directly and would raise on a missing one."""
    assert state_key in started_app.session_state


def test_app_module_stays_a_thin_shell():
    """Analysis logic belongs in analysis/; app.py owns flow and session state."""
    line_count = len(APP_PATH.read_text(encoding="utf-8").splitlines())

    assert line_count < 600, (
        f"app.py has grown to {line_count} lines. Move rendering into ui/ and "
        "analysis into analysis/."
    )
