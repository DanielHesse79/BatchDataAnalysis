"""Tests for the QC Intelligence Layer.

The synthetic dataset carries its planted behaviour in ground_truth.json, so the
detector tests assert recovery of known patterns rather than reproducing
hand-computed numbers.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from qc_intel import db
from qc_intel.alerts import render_alert
from qc_intel.config import ConfigError, load_all_method_configs, load_method_config
from qc_intel.detectors import find_events_near
from qc_intel.ingest.generic_tabular import (
    IngestError,
    import_frame,
    import_table,
    read_tabular,
)
from qc_intel.ingest.interactive import (
    apply_mapping,
    missing_required_columns,
    screen_for_personal_data,
    suggest_column_mapping,
    suggest_target_table,
)
from qc_intel.pipeline import build_from_scratch, control_false_discovery
from qc_intel.stats import (
    align_timestamp_to,
    apply_false_discovery_control,
    between_run_cv,
    build_run_series,
    coefficient_of_variation,
    compute_baseline,
    theil_sen_trend,
    variance_ratio_p_value,
    within_run_cv,
)
from qc_intel.synth.generate import generate, write_source_files


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = PACKAGE_ROOT / "config"


@pytest.fixture(scope="module")
def prototype(tmp_path_factory):
    """Generate, ingest and analyse a full synthetic history once."""
    working_dir = tmp_path_factory.mktemp("qc_intel")
    dataset = generate()
    source_dir = working_dir / "source"
    write_source_files(dataset, source_dir)
    connection, result = build_from_scratch(
        source_dir,
        working_dir / "qc_intel.sqlite",
        load_all_method_configs(CONFIG_DIR),
    )
    return {"connection": connection, "result": result, "truth": dataset.truth, "dir": working_dir}


# --------------------------------------------------------------- statistics

def test_cv_needs_enough_points_and_a_usable_mean():
    assert np.isnan(coefficient_of_variation(pd.Series([1.0, 2.0])))
    assert np.isnan(coefficient_of_variation(pd.Series([0.0, 0.0, 0.0])))
    assert coefficient_of_variation(pd.Series([10.0, 10.0, 10.0, 10.0])) == pytest.approx(0.0)


def test_cv_matches_a_hand_computed_value():
    values = pd.Series([9.0, 10.0, 11.0])  # mean 10, sd 1
    assert coefficient_of_variation(values) == pytest.approx(10.0)


def test_within_and_between_run_cv_measure_different_things():
    """Pooling them hides which of preparation or injection is degrading."""
    observations = pd.DataFrame(
        {
            "run_id": ["R1", "R1", "R2", "R2", "R3", "R3"],
            # Replicates agree tightly inside each run, run means differ a lot.
            "measured_value": [10.0, 10.02, 12.0, 12.02, 14.0, 14.02],
        }
    )

    assert within_run_cv(observations) < 0.5
    assert between_run_cv(observations) > 10.0


def test_theil_sen_recovers_a_known_slope():
    days = np.arange(40)
    frame = pd.DataFrame(
        {
            "acquisition_timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(days, unit="D"),
            "mean_bias_percent": 2.0 + 0.25 * days,
        }
    )

    trend = theil_sen_trend(frame)

    assert trend["slope_per_day"] == pytest.approx(0.25, rel=1e-6)
    assert trend["kendall_p"] < 1e-6


def test_theil_sen_ignores_an_isolated_outlier():
    """QC series contain occasional wild points; the trend should survive them."""
    days = np.arange(30)
    values = 1.0 + 0.1 * days
    values[15] = 95.0

    frame = pd.DataFrame(
        {
            "acquisition_timestamp": pd.Timestamp("2025-01-01") + pd.to_timedelta(days, unit="D"),
            "mean_bias_percent": values,
        }
    )

    assert theil_sen_trend(frame)["slope_per_day"] == pytest.approx(0.1, rel=0.05)


def test_variance_ratio_p_value_reacts_to_sample_size():
    """The same CV ratio is not equally convincing at different run counts."""
    small = variance_ratio_p_value(pd.Series(range(6)), 1.0, 6, recent_cv=6.0, baseline_cv=3.0)
    large = variance_ratio_p_value(pd.Series(range(40)), 1.0, 40, recent_cv=6.0, baseline_cv=3.0)

    assert large < small


def test_false_discovery_control_rejects_a_lone_marginal_result():
    p_values = pd.Series([0.001, 0.002, 0.003, 0.060])

    survives = apply_false_discovery_control(p_values, alpha=0.05)

    assert survives.tolist() == [True, True, True, False]


def test_baseline_uses_only_the_frozen_window():
    """A baseline recomputed over drifting history hides the drift."""
    frame = pd.DataFrame(
        {
            "acquisition_timestamp": pd.date_range("2025-01-01", periods=20, freq="7D"),
            "mean_bias_percent": [0.0] * 10 + [20.0] * 10,
            "mean_measured": [100.0] * 10 + [120.0] * 10,
        }
    )

    baseline = compute_baseline(frame, "2025-01-01", "2025-03-01")

    assert baseline.n_runs < 20
    assert baseline.mean_bias_percent == pytest.approx(0.0)


def test_timestamp_alignment_survives_mixed_awareness():
    """Exports carry timezone-aware times; config files carry plain dates."""
    aware = pd.Series(pd.date_range("2025-01-01", periods=3, tz="UTC"))
    naive = pd.Series(pd.date_range("2025-01-01", periods=3))

    assert align_timestamp_to(aware, "2025-01-02").tzinfo is not None
    assert align_timestamp_to(naive, "2025-01-02").tzinfo is None


# ------------------------------------------------------------- configuration

def test_every_shipped_config_loads():
    configs = load_all_method_configs(CONFIG_DIR)

    assert set(configs) == {"ASSAY_A", "ASSAY_B", "ASSAY_C"}
    for method_config in configs.values():
        assert method_config.qc_levels
        assert method_config.trending.baseline.start


def test_a_config_without_a_frozen_baseline_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        'method_id = "X"\n[acceptance.qc_levels.low]\nnominal = 1.0\n'
        'max_bias_percent = 15.0\n[trending]\nstep_window_runs = 8\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="baseline"):
        load_method_config(path)


def test_a_qc_level_without_acceptance_is_rejected(tmp_path):
    path = tmp_path / "bad.toml"
    path.write_text(
        'method_id = "X"\n[acceptance.qc_levels.low]\nnominal = 1.0\n'
        '[trending.baseline]\nstart = "2025-01-01"\nend = "2025-02-01"\n',
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="max_bias_percent"):
        load_method_config(path)


# ----------------------------------------------------------------- ingestion

def test_importing_the_same_file_twice_is_refused(tmp_path):
    """Re-importing would silently double every statistic built on it."""
    connection = db.reset_database(tmp_path / "test.sqlite")
    source = tmp_path / "instruments.csv"
    pd.DataFrame([{"instrument_id": "LCMS-01", "instrument_model": "Xevo"}]).to_csv(
        source, index=False
    )

    assert import_table(connection, source, "instrument") == 1
    assert import_table(connection, source, "instrument", skip_if_imported=True) == 0

    with pytest.raises(db.DuplicateSourceError):
        import_table(connection, source, "instrument")


def test_a_file_missing_required_columns_is_rejected(tmp_path):
    connection = db.reset_database(tmp_path / "test.sqlite")
    source = tmp_path / "qc_results.csv"
    pd.DataFrame([{"qc_level": "low_qc"}]).to_csv(source, index=False)

    with pytest.raises(IngestError, match="missing required column"):
        import_table(connection, source, "qc_result")


def test_an_empty_file_is_rejected(tmp_path):
    connection = db.reset_database(tmp_path / "test.sqlite")
    source = tmp_path / "instruments.csv"
    pd.DataFrame(columns=["instrument_id"]).to_csv(source, index=False)

    with pytest.raises(IngestError, match="no rows"):
        import_table(connection, source, "instrument")


def test_unreadable_files_raise_a_clear_error(tmp_path):
    """A file whose extension lies about its contents must fail loudly.

    Arbitrary bytes in a .csv are not a good test: pandas will happily read them
    as a single odd column rather than raising, and the required-column check is
    what catches that case.
    """
    source = tmp_path / "not_really.xlsx"
    source.write_text("run_id,qc_level\nR1,low_qc\n", encoding="utf-8")

    with pytest.raises(IngestError, match="Could not read"):
        read_tabular(source)


# ------------------------------------------------------- synthetic recovery

def test_every_run_and_result_survives_ingestion(prototype):
    observations = prototype["result"].observations

    assert observations["run_id"].nunique() == 324
    assert len(observations) == 1944


def test_all_three_instruments_run_all_three_methods(prototype):
    """A generator that pins a method to one instrument would hide the drift."""
    observations = prototype["result"].observations
    coverage = observations.groupby("method_id")["instrument_id"].nunique()

    assert set(coverage) == {3}


def test_the_planted_instrument_drift_is_found(prototype):
    """ASSAY_B on LCMS-02 walks downward from month 4."""
    truth = prototype["truth"]["gradual_drift"]
    findings = prototype["result"].findings_frame
    drift = findings[
        (findings["rule_id"] == "gradual_drift")
        & (findings["method_id"] == truth["method_id"])
        & (findings["instrument_id"] == truth["instrument_id"])
    ]

    assert not drift.empty
    assert (drift["magnitude"] < 0).all(), "planted drift is downward"
    assert (drift["severity"] == "critical").any()


def test_the_drift_is_specific_to_the_affected_instrument(prototype):
    truth = prototype["truth"]["gradual_drift"]
    findings = prototype["result"].findings_frame
    other_instruments = findings[
        (findings["rule_id"] == "gradual_drift")
        & (findings["method_id"] == truth["method_id"])
        & (findings["instrument_id"] != truth["instrument_id"])
    ]

    assert other_instruments.empty


def test_the_planted_step_change_is_found(prototype):
    """ASSAY_C shifts upward at a reference-standard lot change."""
    truth = prototype["truth"]["step_change"]
    findings = prototype["result"].findings_frame
    steps = findings[
        (findings["rule_id"] == "step_change") & (findings["method_id"] == truth["method_id"])
    ]

    assert not steps.empty
    assert (steps["magnitude"] > 0).all(), "planted step is upward"


def test_the_step_change_lines_up_with_the_lot_change_event(prototype):
    """The event overlay is the whole differentiator; it has to line up."""
    result = prototype["result"]
    truth = prototype["truth"]["step_change"]
    step = next(
        finding for finding in result.findings
        if finding.rule_id == "step_change" and finding.method_id == truth["method_id"]
    )

    nearby = find_events_near(
        result.events, step.instrument_id, step.method_id,
        pd.Timestamp(step.window_start), lookback_days=200,
    )

    assert truth["coincides_with"] in set(nearby["event_type"])


def test_the_planted_variance_problem_is_found(prototype):
    truth = prototype["truth"]["variance_increase"]
    findings = prototype["result"].findings_frame
    variance = findings[
        (findings["rule_id"] == "variance_increase")
        & (findings["method_id"] == truth["method_id"])
        & (findings["instrument_id"] == truth["instrument_id"])
    ]

    assert not variance.empty
    assert (variance["magnitude"] > 0).all(), "spread grew"


def test_the_stable_method_produces_no_bias_findings(prototype):
    """ASSAY_A is the negative control for the bias detectors.

    Variance findings are deliberately not asserted absent here. At this run
    cadence a doubling of the sample CV is within sampling noise, so a test
    demanding zero variance false positives would be demanding something the
    statistics cannot deliver - see README, "What the prototype showed".
    """
    truth = prototype["truth"]
    findings = prototype["result"].findings_frame
    bias_findings = findings[
        (findings["method_id"] == truth["stable_method"])
        & (findings["rule_id"].isin(["gradual_drift", "step_change"]))
    ]

    assert bias_findings.empty


def test_findings_are_ranked_by_acceptance_fraction_within_severity(prototype):
    findings = prototype["result"].findings_frame
    for _, rows in findings.groupby("severity", sort=False):
        fractions = rows["acceptance_fraction"].tolist()
        assert fractions == sorted(fractions, reverse=True)


def test_multiplicity_control_passes_through_untested_rules():
    """Step change compares by effect size only and runs no hypothesis test."""
    from qc_intel.detectors import Finding

    untested = Finding(
        rule_id="step_change", method_id="M", method_version="1", instrument_id="I",
        qc_level="low_qc", severity="warning", headline="h", magnitude=1.0,
        magnitude_units="percent_bias", acceptance_fraction=0.5, n_runs=8,
        window_start="2025-01-01", window_end="2025-02-01", evidence={}, p_value=None,
    )

    assert control_false_discovery([untested]) == [untested]


# --------------------------------------------------------------- provenance

def test_every_qc_row_is_traceable_to_its_source(prototype):
    observations = prototype["result"].observations

    assert observations["ingest_id"].notna().all()
    assert observations["source_row"].notna().all()


def test_the_ingest_log_records_a_checksum(prototype):
    log = db.read_sql(prototype["connection"], "SELECT * FROM ingest_log")

    assert not log.empty
    assert log["source_checksum"].str.len().eq(64).all()
    assert log["script_version"].notna().all()


# ------------------------------------------------------------------- alerts

def test_alert_text_names_the_rule_and_carries_the_caveat(prototype):
    findings = prototype["result"].findings
    assert findings, "expected at least one finding to render"

    text = render_alert(findings[0])

    assert findings[0].rule_id in text or "Rule triggered" in text
    assert "Not a run-acceptance decision" in text
    assert "%" in text


def test_alert_text_is_deterministic(prototype):
    finding = prototype["result"].findings[0]

    assert render_alert(finding) == render_alert(finding)


# -------------------------------------------------------------------- charts

def test_every_control_chart_renders_with_its_event_overlay(prototype):
    """The event overlay is the differentiator, so it must survive rendering.

    Plotly's add_vline positions its annotation by averaging the x values, which
    raises on a datetime axis; the chart uses an explicit shape instead. Nothing
    else in the test suite executes the figure code, so this is what stops that
    regressing.
    """
    from qc_intel.app import control_chart

    result = prototype["result"]
    events = result.events
    assert not events.empty, "expected laboratory events to overlay"

    rendered = 0
    for key, series in result.run_series.items():
        _, _, instrument_id, _ = key
        window_events = events[
            (
                ((events["entity_type"] == "instrument") & (events["entity_id"] == instrument_id))
                | (events["entity_type"] == "material")
            )
            & (events["event_timestamp"] >= series["acquisition_timestamp"].min())
            & (events["event_timestamp"] <= series["acquisition_timestamp"].max())
        ]
        figure = control_chart(
            series, result.baselines[key], 15.0, window_events, "chart"
        )
        assert figure.data, "chart produced no traces"
        rendered += 1

    assert rendered == len(result.run_series)


def test_a_chart_draws_a_line_for_each_event(prototype):
    from qc_intel.app import control_chart

    result = prototype["result"]
    key = next(iter(result.run_series))
    series = result.run_series[key]
    events = result.events.head(3)

    figure = control_chart(series, result.baselines[key], 15.0, events, "chart")

    event_annotations = [
        annotation for annotation in figure.layout.annotations
        if annotation.textangle == -90
    ]
    assert len(event_annotations) == len(events)


def test_a_held_connection_breaks_across_threads(prototype):
    """Documents why the dashboard must not cache a connection.

    Streamlit runs each script run on a different script-runner thread, so a
    handle kept between runs raises. This test pins the behaviour that the
    design works around.
    """
    import threading

    connection = db.connect(prototype["dir"] / "qc_intel.sqlite")
    failures: list[Exception] = []

    def query_from_another_thread():
        try:
            db.read_sql(connection, "SELECT 1")
        except Exception as error:  # sqlite3.ProgrammingError, wrapped by pandas
            failures.append(error)

    worker = threading.Thread(target=query_from_another_thread)
    worker.start()
    worker.join()
    connection.close()

    assert failures, "expected SQLite to refuse a cross-thread connection"
    assert "thread" in str(failures[0]).lower()


def test_a_scoped_connection_works_from_any_thread(prototype):
    """The fix: open per unit of work rather than holding the handle."""
    import threading

    database_path = prototype["dir"] / "qc_intel.sqlite"
    row_counts: list[int] = []
    failures: list[Exception] = []

    def query_from_another_thread():
        try:
            with db.connection_scope(database_path) as connection:
                row_counts.append(len(db.read_sql(connection, "SELECT * FROM ingest_log")))
        except Exception as error:
            failures.append(error)

    worker = threading.Thread(target=query_from_another_thread)
    worker.start()
    worker.join()

    assert not failures, f"scoped connection failed: {failures}"
    assert row_counts and row_counts[0] > 0


def test_the_scope_closes_the_connection():
    """A leaked handle per rerun would exhaust file descriptors over a session."""
    import sqlite3

    with db.connection_scope(":memory:") as connection:
        connection.execute("SELECT 1")

    with pytest.raises(sqlite3.ProgrammingError):
        connection.execute("SELECT 1")


def test_a_chart_without_events_still_renders(prototype):
    from qc_intel.app import control_chart

    result = prototype["result"]
    key = next(iter(result.run_series))

    figure = control_chart(
        result.run_series[key], result.baselines[key], 15.0,
        result.events.iloc[0:0], "chart",
    )

    assert figure.data


def test_the_database_stays_in_the_package_when_running_from_source():
    """Build scripts, tests and the docs all expect qc_intel/data/ from source."""
    assert db.resolve_database_path() == db.EXAMPLE_DATABASE_PATH


def test_an_installed_copy_writes_outside_its_own_directory(monkeypatch):
    """A bundle may sit in Program Files or on a read-only share."""
    monkeypatch.setattr(db.sys, "frozen", True, raising=False)

    resolved = db.resolve_database_path()

    assert resolved != db.EXAMPLE_DATABASE_PATH
    assert db.PACKAGE_ROOT not in resolved.parents


def test_the_writable_directory_is_per_user(monkeypatch, tmp_path):
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path))
    monkeypatch.setenv("XDG_DATA_HOME", str(tmp_path))

    directory = db.user_data_directory()

    assert tmp_path in directory.parents
    assert directory.name == "qc_intel"


# --------------------------------------------------------- interactive intake
# Headings taken from the shape of real chromatography exports: a run is a
# "Sample Set", the number is a "Result", and decimals arrive with commas.
QC_RESULT_EXPORT = pd.DataFrame({
    "Sample Set Name": ["RUN-001", "RUN-001"],
    "QC Level": ["LQC", "HQC"],
    "Result": ["1,25", "9,80"],
    "Acq. Date/Time": ["2026-01-04 08:15", "2026-01-04 08:40"],
    "Operator": ["A-17", "A-17"],
})

RUN_EXPORT = pd.DataFrame({
    "Sample Set Name": ["RUN-001"],
    "Method": ["M-CORT-01"],
    "System": ["LC-02"],
    "Acq. Date/Time": ["2026-01-04 08:15"],
})


def test_a_qc_export_is_recognised_from_its_headings():
    assert suggest_target_table(QC_RESULT_EXPORT.columns) == "qc_result"


def test_a_run_export_is_not_mistaken_for_its_results():
    """Both carry a run identifier; only one carries method and instrument."""
    assert suggest_target_table(RUN_EXPORT.columns) == "analytical_run"


def test_vendor_headings_map_to_canonical_columns():
    mapping = suggest_column_mapping(QC_RESULT_EXPORT.columns, "qc_result")

    assert mapping["run_id"] == "Sample Set Name"
    assert mapping["qc_level"] == "QC Level"
    assert mapping["measured_value"] == "Result"
    assert not missing_required_columns(mapping, "qc_result")


def test_decimal_commas_survive_the_mapping():
    mapping = suggest_column_mapping(QC_RESULT_EXPORT.columns, "qc_result")

    canonical, report = apply_mapping(QC_RESULT_EXPORT, mapping, "qc_result")

    assert list(canonical["measured_value"]) == [1.25, 9.80]
    assert not report.unparsed_numbers


def test_columns_nobody_mapped_are_discarded_rather_than_stored():
    """The database should never receive a column no one chose to send."""
    mapping = suggest_column_mapping(QC_RESULT_EXPORT.columns, "qc_result")

    canonical, report = apply_mapping(QC_RESULT_EXPORT, mapping, "qc_result")

    assert "Operator" in report.dropped_columns
    assert not any("perator" in str(column) for column in canonical.columns)


def test_timestamps_are_stored_as_iso_text():
    mapping = suggest_column_mapping(RUN_EXPORT.columns, "analytical_run")

    canonical, _report = apply_mapping(RUN_EXPORT, mapping, "analytical_run")

    assert canonical.loc[0, "acquisition_timestamp"] == "2026-01-04T08:15:00"


def test_a_named_person_is_refused_before_anything_is_written():
    frame = pd.DataFrame({
        "Patient Name": ["Anna Svensson"],
        "QC Level": ["LQC"],
        "Result": [1.2],
    })

    findings = screen_for_personal_data(frame)

    assert [finding.column for finding in findings] == ["Patient Name"]
    assert findings[0].blocking


def test_identity_numbers_are_caught_under_a_neutral_heading():
    """An export can call a column 'Ref 2' and fill it with personal numbers."""
    frame = pd.DataFrame({
        "Ref 2": ["19850101-1234", "19900312-5678", "19771122-9012"],
    })

    findings = screen_for_personal_data(frame)

    assert findings and findings[0].blocking


def test_ordinary_qc_columns_are_not_flagged():
    """A screen that cries wolf gets clicked through."""
    assert screen_for_personal_data(QC_RESULT_EXPORT) == []


def test_an_upload_records_provenance_from_its_bytes(tmp_path):
    """An upload has no path to re-read, so the checksum comes from the file."""
    connection = db.reset_database(tmp_path / "qc.sqlite")
    checksum = db.bytes_checksum(b"instrument export")

    written = import_frame(
        connection,
        pd.DataFrame({"instrument_id": ["LC-01", "LC-02"]}),
        "instrument",
        source_name="instruments.csv",
        source_format="csv",
        checksum=checksum,
    )
    log = db.read_sql(connection, "SELECT * FROM ingest_log")
    connection.close()

    assert written == 2
    assert log.loc[0, "source_checksum"] == checksum


def test_the_same_upload_twice_is_refused(tmp_path):
    connection = db.reset_database(tmp_path / "qc.sqlite")
    arguments = {
        "table": "instrument",
        "source_name": "instruments.csv",
        "source_format": "csv",
        "checksum": db.bytes_checksum(b"same file"),
    }
    frame = pd.DataFrame({"instrument_id": ["LC-01"]})

    import_frame(connection, frame, **arguments)

    with pytest.raises(db.DuplicateSourceError):
        import_frame(connection, frame, **arguments)
    connection.close()


def test_a_rejected_file_leaves_no_provenance_behind(tmp_path):
    """A committed ingest_log row claims the file loaded and blocks the retry."""
    connection = db.reset_database(tmp_path / "qc.sqlite")
    orphan_results = pd.DataFrame({
        "run_id": ["RUN-404"],
        "qc_level": ["LQC"],
        "measured_value": [1.2],
    })
    arguments = {
        "table": "qc_result",
        "source_name": "qc_results.csv",
        "source_format": "csv",
        "checksum": db.bytes_checksum(b"orphan results"),
    }

    with pytest.raises(IngestError, match="runs"):
        import_frame(connection, orphan_results, **arguments)

    assert db.read_sql(connection, "SELECT * FROM ingest_log").empty

    # The same file must still be retryable rather than refused as a duplicate.
    with pytest.raises(IngestError, match="runs"):
        import_frame(connection, orphan_results, **arguments)
    connection.close()


def test_a_missing_parent_is_explained_in_the_analyst_s_words(tmp_path):
    connection = db.reset_database(tmp_path / "qc.sqlite")

    with pytest.raises(IngestError) as failure:
        import_frame(
            connection,
            pd.DataFrame({"run_id": ["R1"], "qc_level": ["LQC"], "measured_value": [1.0]}),
            "qc_result",
            source_name="qc_results.csv",
            source_format="csv",
            checksum=db.bytes_checksum(b"x"),
        )
    connection.close()

    message = str(failure.value)
    assert "runs" in message and "FOREIGN KEY" not in message
    assert "Nothing was written" in message


def test_timestamps_from_different_exports_load_together(tmp_path):
    """A CDS writes an offset, a LIMS export does not. One database holds both."""
    connection = db.reset_database(tmp_path / "qc.sqlite")
    db.insert_frame(connection, "method", pd.DataFrame([{
        "method_id": "M1", "method_name": "Assay", "method_version": "1",
        "analyte": "cortisol",
    }]))
    db.insert_frame(connection, "instrument", pd.DataFrame([{"instrument_id": "LC-1"}]))
    db.insert_frame(connection, "analytical_run", pd.DataFrame([
        {"run_id": "R1", "method_id": "M1", "instrument_id": "LC-1",
         "acquisition_timestamp": "2026-02-02T08:15:00+00:00"},
        {"run_id": "R2", "method_id": "M1", "instrument_id": "LC-1",
         "acquisition_timestamp": "2026-02-03T08:20:00"},
    ]))
    db.insert_frame(connection, "qc_result", pd.DataFrame([
        {"run_id": "R1", "qc_level": "low_qc", "measured_value": 1.0,
         "evaluation_type": "quantitative"},
        {"run_id": "R2", "qc_level": "low_qc", "measured_value": 1.1,
         "evaluation_type": "quantitative"},
    ]))

    observations = db.load_qc_observations(connection)
    connection.close()

    assert len(observations) == 2
    assert observations["acquisition_timestamp"].is_monotonic_increasing
