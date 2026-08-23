"""Orchestration: source files to findings.

    RAW (source files, checksummed)
      -> NORMALIZED (canonical tables, immutable)
      -> DERIVED (descriptives, run series, findings)
      -> DASHBOARD / ALERTS

Derived output is always rebuilt from the normalized tables, never edited in
place, so any number on screen can be regenerated from the source file that
produced it.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sqlite3

import pandas as pd

from qc_intel import db
from qc_intel.config import MethodConfig, load_all_method_configs
from qc_intel.detectors import DETECTORS, Finding, find_events_near
from qc_intel.ingest.generic_tabular import import_table
from qc_intel.stats import (
    apply_false_discovery_control,
    GROUP_KEYS,
    add_rolling_statistics,
    build_run_series,
    compute_baseline,
    describe_all_groups,
)


SOURCE_TABLE_ORDER = [
    ("methods.csv", "method"),
    ("instruments.csv", "instrument"),
    ("materials.csv", "material_lot"),
    ("runs.csv", "analytical_run"),
    ("run_materials.csv", "run_material"),
    ("qc_results.csv", "qc_result"),
    ("calibrations.csv", "calibration"),
    ("calibrator_points.csv", "calibrator_point"),
    ("events.csv", "lab_event"),
]


@dataclass
class AnalysisResult:
    """Everything the dashboard renders."""

    observations: pd.DataFrame
    descriptives: pd.DataFrame
    run_series: dict[tuple, pd.DataFrame]
    baselines: dict[tuple, object]
    findings: list[Finding]
    events: pd.DataFrame
    configs: dict[str, MethodConfig]

    @property
    def findings_frame(self) -> pd.DataFrame:
        if not self.findings:
            return pd.DataFrame(
                columns=[
                    "rule_id", "method_id", "method_version", "instrument_id", "qc_level",
                    "severity", "headline", "magnitude", "acceptance_fraction", "n_runs",
                    "window_start", "window_end",
                ]
            )
        frame = pd.DataFrame([finding.as_dict() for finding in self.findings])
        severity_order = {"critical": 0, "warning": 1, "watch": 2}
        frame["_order"] = frame["severity"].map(severity_order).fillna(9)
        return (
            frame.sort_values(["_order", "acceptance_fraction"], ascending=[True, False])
            .drop(columns="_order")
            .reset_index(drop=True)
        )


def ingest_source_directory(
    connection: sqlite3.Connection,
    source_dir: Path | str,
    skip_if_imported: bool = True,
) -> dict[str, int]:
    """Import every recognised source file, in dependency order."""
    source_dir = Path(source_dir)
    imported: dict[str, int] = {}

    for file_name, table in SOURCE_TABLE_ORDER:
        path = source_dir / file_name
        if not path.exists():
            continue
        imported[table] = import_table(connection, path, table, skip_if_imported=skip_if_imported)

    return imported


def analyze(
    connection: sqlite3.Connection,
    configs: dict[str, MethodConfig] | None = None,
) -> AnalysisResult:
    """Rebuild every derived artefact from the normalized tables."""
    configs = configs if configs is not None else load_all_method_configs()
    observations = db.load_qc_observations(connection)
    events = db.load_events(connection)

    descriptives = describe_all_groups(observations)
    run_series: dict[tuple, pd.DataFrame] = {}
    baselines: dict[tuple, object] = {}
    findings: list[Finding] = []

    if observations.empty:
        return AnalysisResult(observations, descriptives, run_series, baselines, findings, events, configs)

    for group_key, group_rows in observations.groupby(GROUP_KEYS, sort=False):
        method_id = group_key[0]
        qc_level = group_key[3]
        method_config = configs.get(method_id)
        if method_config is None:
            continue

        level_config = method_config.level(qc_level)
        if level_config is None or level_config.evaluation_type != "quantitative":
            continue

        series = add_rolling_statistics(build_run_series(group_rows))
        baseline = compute_baseline(
            series,
            method_config.trending.baseline.start,
            method_config.trending.baseline.end,
        )
        run_series[group_key] = series
        baselines[group_key] = baseline

        for detector in DETECTORS:
            finding = detector(series, group_rows, baseline, level_config, method_config)
            if finding is not None:
                findings.append(finding)

    findings = control_false_discovery(findings)
    return AnalysisResult(observations, descriptives, run_series, baselines, findings, events, configs)


def control_false_discovery(findings: list[Finding], alpha: float = 0.05) -> list[Finding]:
    """Drop findings that do not survive multiplicity control.

    Method x instrument x QC level x detector is a large family of simultaneous
    tests. Without this, roughly one test in twenty fires on a perfectly stable
    method every cycle, and an alert list that is mostly noise gets ignored.

    Rules that compare against the frozen baseline by effect size alone run no
    hypothesis test and are passed through untouched.
    """
    tested = [finding for finding in findings if finding.p_value is not None]
    untested = [finding for finding in findings if finding.p_value is None]
    if not tested:
        return findings

    p_values = pd.Series([finding.p_value for finding in tested])
    survives = apply_false_discovery_control(p_values, alpha=alpha)
    return untested + [finding for finding, keep in zip(tested, survives) if keep]


def events_for_finding(result: AnalysisResult, finding: Finding, lookback_days: int = 30) -> pd.DataFrame:
    """Return laboratory events shortly before a finding's window opened."""
    return find_events_near(
        result.events,
        instrument_id=finding.instrument_id,
        method_id=finding.method_id,
        around=pd.Timestamp(finding.window_start),
        lookback_days=lookback_days,
    )


def build_from_scratch(
    source_dir: Path | str,
    database_path: Path | str = db.DEFAULT_DATABASE_PATH,
    configs: dict[str, MethodConfig] | None = None,
) -> tuple[sqlite3.Connection, AnalysisResult]:
    """Reset the database, import the sources, and analyse. For the prototype."""
    connection = db.reset_database(database_path)
    ingest_source_directory(connection, source_dir, skip_if_imported=False)
    return connection, analyze(connection, configs)
