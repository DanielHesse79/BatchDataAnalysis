"""Deterministic drift detectors.

Three patterns, chosen because they fail for different reasons and call for
different investigations:

  step change      the mean moves at a point in time      -> look for an event
  gradual drift    the mean walks steadily                -> consumables, ageing
  variance change  the mean holds, the spread grows       -> preparation, injection

Every finding is ranked by how much of the acceptance window it consumes, not by
a p-value. A 0.4% shift can be highly significant with enough runs and still be
irrelevant against a +/-15% criterion.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd

from qc_intel.config import MethodConfig, QcLevelConfig
from qc_intel.stats import BaselineStatistics, align_timestamp_to, theil_sen_trend


STEP_CHANGE = "step_change"
GRADUAL_DRIFT = "gradual_drift"
VARIANCE_INCREASE = "variance_increase"


@dataclass(frozen=True)
class Finding:
    """One detected pattern, traceable to the rule that produced it."""

    rule_id: str
    method_id: str
    method_version: str
    instrument_id: str
    qc_level: str
    severity: str
    headline: str
    magnitude: float
    magnitude_units: str
    acceptance_fraction: float
    n_runs: int
    window_start: str
    window_end: str
    evidence: dict[str, Any]
    # None for rules that compare against a frozen baseline by effect size only
    # and therefore run no hypothesis test.
    p_value: float | None = None

    def as_dict(self) -> dict[str, Any]:
        record = asdict(self)
        record["evidence"] = dict(self.evidence)
        return record


def _group_identity(run_series: pd.DataFrame, observations: pd.DataFrame) -> dict[str, str]:
    first = observations.iloc[0]
    return {
        "method_id": str(first["method_id"]),
        "method_version": str(first["method_version"]),
        "instrument_id": str(first["instrument_id"]),
        "qc_level": str(first["qc_level"]),
    }


def _window_bounds(run_series: pd.DataFrame) -> tuple[str, str]:
    timestamps = pd.to_datetime(run_series["acquisition_timestamp"])
    return (
        timestamps.min().date().isoformat(),
        timestamps.max().date().isoformat(),
    )


def _severity_from_fraction(acceptance_fraction: float) -> str:
    """Grade a finding by how much of the acceptance window it uses."""
    if acceptance_fraction >= 0.75:
        return "critical"
    if acceptance_fraction >= 0.50:
        return "warning"
    return "watch"


def detect_step_change(
    run_series: pd.DataFrame,
    observations: pd.DataFrame,
    baseline: BaselineStatistics,
    level_config: QcLevelConfig,
    method_config: MethodConfig,
) -> Finding | None:
    """Compare the most recent runs against the frozen baseline mean."""
    trending = method_config.trending
    recent = run_series.tail(trending.step_window_runs)
    if len(recent) < trending.step_window_runs or not baseline.is_usable:
        return None

    recent_mean = float(pd.to_numeric(recent["mean_bias_percent"], errors="coerce").mean())
    shift = recent_mean - baseline.mean_bias_percent
    acceptance_fraction = abs(shift) / level_config.acceptance_half_width

    if acceptance_fraction < trending.step_min_shift_fraction:
        return None

    identity = _group_identity(run_series, observations)
    window_start, window_end = _window_bounds(recent)
    direction = "increased" if shift > 0 else "decreased"

    return Finding(
        rule_id=STEP_CHANGE,
        severity=_severity_from_fraction(acceptance_fraction),
        headline=(
            f"Mean bias {direction} from {baseline.mean_bias_percent:+.1f}% to "
            f"{recent_mean:+.1f}% over the last {len(recent)} runs"
        ),
        magnitude=float(shift),
        magnitude_units="percent_bias",
        acceptance_fraction=float(acceptance_fraction),
        n_runs=int(len(recent)),
        window_start=window_start,
        window_end=window_end,
        evidence={
            "baseline_mean_bias_percent": round(baseline.mean_bias_percent, 3),
            "baseline_runs": baseline.n_runs,
            "baseline_window": f"{baseline.start} to {baseline.end}",
            "recent_mean_bias_percent": round(recent_mean, 3),
            "acceptance_half_width_percent": level_config.acceptance_half_width,
            "shift_as_fraction_of_acceptance": round(acceptance_fraction, 3),
        },
        **identity,
    )


def detect_gradual_drift(
    run_series: pd.DataFrame,
    observations: pd.DataFrame,
    baseline: BaselineStatistics,
    level_config: QcLevelConfig,
    method_config: MethodConfig,
) -> Finding | None:
    """Flag a sustained monotonic walk in the per-run bias."""
    trending = method_config.trending
    if len(run_series) < trending.trend_min_runs:
        return None

    trend = theil_sen_trend(run_series)
    slope_per_day = trend.get("slope_per_day", float("nan"))
    span_days = trend.get("span_days", float("nan"))
    if not np.isfinite(slope_per_day) or not np.isfinite(span_days):
        return None

    total_shift = slope_per_day * span_days
    acceptance_fraction = abs(total_shift) / level_config.acceptance_half_width

    if abs(total_shift) < trending.trend_min_total_shift_percent:
        return None
    if trend.get("kendall_p", 1.0) > trending.trend_max_kendall_p:
        return None

    identity = _group_identity(run_series, observations)
    window_start, window_end = _window_bounds(run_series)
    direction = "upward" if total_shift > 0 else "downward"

    return Finding(
        rule_id=GRADUAL_DRIFT,
        severity=_severity_from_fraction(acceptance_fraction),
        headline=(
            f"Sustained {direction} drift of {total_shift:+.1f}% bias across "
            f"{len(run_series)} runs ({span_days:.0f} days)"
        ),
        magnitude=float(total_shift),
        magnitude_units="percent_bias",
        acceptance_fraction=float(acceptance_fraction),
        n_runs=int(len(run_series)),
        window_start=window_start,
        window_end=window_end,
        evidence={
            "theil_sen_slope_percent_per_day": round(slope_per_day, 5),
            "span_days": round(span_days, 1),
            "kendall_tau": round(trend.get("kendall_tau", float("nan")), 3),
            "kendall_p": round(trend.get("kendall_p", float("nan")), 5),
            "acceptance_half_width_percent": level_config.acceptance_half_width,
        },
        p_value=trend.get("kendall_p"),
        **identity,
    )


def detect_variance_increase(
    run_series: pd.DataFrame,
    observations: pd.DataFrame,
    baseline: BaselineStatistics,
    level_config: QcLevelConfig,
    method_config: MethodConfig,
) -> Finding | None:
    """Flag growing spread while the mean may be perfectly acceptable."""
    trending = method_config.trending
    recent = run_series.tail(trending.variance_min_runs)
    if len(recent) < trending.variance_min_runs:
        return None
    if not np.isfinite(baseline.cv_percent) or baseline.cv_percent <= 0:
        return None

    from qc_intel.stats import coefficient_of_variation, variance_ratio_p_value

    recent_cv = coefficient_of_variation(recent["mean_measured"])
    if not np.isfinite(recent_cv):
        return None

    cv_ratio = recent_cv / baseline.cv_percent
    if cv_ratio < trending.variance_cv_ratio:
        return None

    # Effect size alone is not enough here. With a handful of runs the sample CV
    # wanders widely, and a ratio threshold on its own turns ordinary sampling
    # noise on a stable method into a steady stream of alerts.
    spread_p_value = variance_ratio_p_value(
        recent_values=recent["mean_measured"],
        baseline_sd=baseline.sd_bias_percent,
        baseline_n=baseline.n_runs,
        recent_cv=recent_cv,
        baseline_cv=baseline.cv_percent,
    )
    if spread_p_value > trending.variance_max_p:
        return None

    identity = _group_identity(run_series, observations)
    window_start, window_end = _window_bounds(recent)
    # Spread is graded against the acceptance window too: a CV that consumes a
    # large share of the allowance matters more than a large relative jump from
    # a very tight baseline.
    acceptance_fraction = recent_cv / level_config.acceptance_half_width

    return Finding(
        rule_id=VARIANCE_INCREASE,
        severity=_severity_from_fraction(acceptance_fraction),
        headline=(
            f"Between-run CV rose from {baseline.cv_percent:.1f}% to {recent_cv:.1f}% "
            f"over the last {len(recent)} runs"
        ),
        magnitude=float(recent_cv - baseline.cv_percent),
        magnitude_units="cv_percent",
        acceptance_fraction=float(acceptance_fraction),
        n_runs=int(len(recent)),
        window_start=window_start,
        window_end=window_end,
        evidence={
            "baseline_cv_percent": round(baseline.cv_percent, 3),
            "recent_cv_percent": round(recent_cv, 3),
            "cv_ratio": round(cv_ratio, 3),
            "variance_ratio_p": round(spread_p_value, 5),
            "baseline_runs": baseline.n_runs,
            "baseline_window": f"{baseline.start} to {baseline.end}",
        },
        p_value=spread_p_value,
        **identity,
    )


DETECTORS = (detect_step_change, detect_gradual_drift, detect_variance_increase)


def find_events_near(
    events: pd.DataFrame,
    instrument_id: str,
    method_id: str,
    around: pd.Timestamp,
    lookback_days: int,
) -> pd.DataFrame:
    """Return laboratory events shortly before a detected change.

    Presented as candidate associations only. Several events usually sit in the
    same window, and this function cannot tell which one mattered.
    """
    if events.empty:
        return events

    around = align_timestamp_to(events["event_timestamp"], around)
    window_start = around - pd.Timedelta(days=lookback_days)
    relevant = events[
        (events["event_timestamp"] >= window_start)
        & (events["event_timestamp"] <= around)
        & (
            ((events["entity_type"] == "instrument") & (events["entity_id"] == instrument_id))
            | ((events["entity_type"] == "method") & (events["entity_id"] == method_id))
            | (events["entity_type"] == "material")
        )
    ]
    return relevant.sort_values("event_timestamp", ascending=False)
