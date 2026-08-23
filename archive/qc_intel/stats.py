"""Descriptive and time-series statistics for QC observations.

Two conventions run through this module.

Run order, not calendar order. Runs cluster during a study and stop between
studies, so anything shaped like a control chart is computed in run sequence.
Calendar time is used only for drift rate, where it is the meaningful axis.

Within-run and between-run precision are kept apart. A pooled CV mixes them,
and they fail for different reasons: within-run points at injection, integration
or autosampler; between-run points at calibration, preparation or standards.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats


GROUP_KEYS = ["method_id", "method_version", "instrument_id", "qc_level"]
MINIMUM_POINTS_FOR_SPREAD = 3


@dataclass(frozen=True)
class GroupDescriptives:
    """Summary of one method x version x instrument x QC level combination."""

    method_id: str
    method_version: str
    instrument_id: str
    qc_level: str
    n_observations: int
    n_runs: int
    mean_bias_percent: float
    median_bias_percent: float
    sd_bias_percent: float
    cv_percent: float
    within_run_cv_percent: float
    between_run_cv_percent: float
    minimum_bias_percent: float
    maximum_bias_percent: float
    mean_bias_ci_low: float
    mean_bias_ci_high: float
    failure_rate_percent: float

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def coefficient_of_variation(
    values: pd.Series,
    minimum_points: int = MINIMUM_POINTS_FOR_SPREAD,
) -> float:
    """Return CV% of measured values. Undefined without a usable mean.

    ``minimum_points`` is loosened to 2 for replicate CV: a bioanalytical run
    normally carries two replicates per QC level, so a floor of three would make
    within-run precision permanently undefined.
    """
    usable = pd.to_numeric(values, errors="coerce").dropna()
    if len(usable) < max(2, minimum_points):
        return float("nan")

    mean_value = usable.mean()
    if not np.isfinite(mean_value) or abs(mean_value) < 1e-12:
        return float("nan")

    return float(usable.std(ddof=1) / abs(mean_value) * 100.0)


def within_run_cv(observations: pd.DataFrame) -> float:
    """Mean CV% across replicates inside a run, pooled over runs.

    Replicates in one run share a preparation and a calibration curve, so their
    spread isolates injection and integration noise.
    """
    per_run = []
    for _, run_rows in observations.groupby("run_id", sort=False):
        if len(run_rows) >= 2:
            run_cv = coefficient_of_variation(run_rows["measured_value"], minimum_points=2)
            if np.isfinite(run_cv):
                per_run.append(run_cv)

    return float(np.mean(per_run)) if per_run else float("nan")


def between_run_cv(observations: pd.DataFrame) -> float:
    """CV% of the per-run mean, which isolates run-to-run behaviour."""
    run_means = observations.groupby("run_id", sort=False)["measured_value"].mean()
    return coefficient_of_variation(run_means)


def mean_confidence_interval(values: pd.Series, confidence: float = 0.95) -> tuple[float, float]:
    """Return a t-based confidence interval for the mean."""
    usable = pd.to_numeric(values, errors="coerce").dropna()
    if len(usable) < MINIMUM_POINTS_FOR_SPREAD:
        return (float("nan"), float("nan"))

    mean_value = float(usable.mean())
    standard_error = float(usable.std(ddof=1) / np.sqrt(len(usable)))
    if not np.isfinite(standard_error) or standard_error == 0.0:
        return (mean_value, mean_value)

    half_width = stats.t.ppf(0.5 + confidence / 2.0, len(usable) - 1) * standard_error
    return (mean_value - half_width, mean_value + half_width)


def describe_group(observations: pd.DataFrame) -> GroupDescriptives:
    """Summarize one group of QC observations."""
    bias = pd.to_numeric(observations["percent_bias"], errors="coerce").dropna()
    confidence_low, confidence_high = mean_confidence_interval(bias)
    failures = observations["pass_fail"].astype(str).str.lower().eq("fail").sum()

    return GroupDescriptives(
        method_id=str(observations["method_id"].iloc[0]),
        method_version=str(observations["method_version"].iloc[0]),
        instrument_id=str(observations["instrument_id"].iloc[0]),
        qc_level=str(observations["qc_level"].iloc[0]),
        n_observations=int(len(observations)),
        n_runs=int(observations["run_id"].nunique()),
        mean_bias_percent=float(bias.mean()) if len(bias) else float("nan"),
        median_bias_percent=float(bias.median()) if len(bias) else float("nan"),
        sd_bias_percent=float(bias.std(ddof=1)) if len(bias) >= 2 else float("nan"),
        cv_percent=coefficient_of_variation(observations["measured_value"]),
        within_run_cv_percent=within_run_cv(observations),
        between_run_cv_percent=between_run_cv(observations),
        minimum_bias_percent=float(bias.min()) if len(bias) else float("nan"),
        maximum_bias_percent=float(bias.max()) if len(bias) else float("nan"),
        mean_bias_ci_low=confidence_low,
        mean_bias_ci_high=confidence_high,
        failure_rate_percent=float(failures) / len(observations) * 100.0 if len(observations) else 0.0,
    )


def describe_all_groups(observations: pd.DataFrame) -> pd.DataFrame:
    """Summarize every method x version x instrument x QC level combination."""
    if observations.empty:
        return pd.DataFrame(columns=[field for field in GroupDescriptives.__annotations__])

    rows = [
        describe_group(group_rows).as_dict()
        for _, group_rows in observations.groupby(GROUP_KEYS, sort=False)
    ]
    return pd.DataFrame(rows)


def build_run_series(observations: pd.DataFrame) -> pd.DataFrame:
    """Collapse observations to one row per run, in acquisition order.

    Control charts are drawn on this: a run is the unit that is accepted or
    rejected, so it is also the unit that should carry a point on the chart.
    """
    if observations.empty:
        return pd.DataFrame(
            columns=[
                "run_id", "acquisition_timestamp", "run_index", "mean_bias_percent",
                "mean_measured", "within_run_cv_percent", "n_replicates",
                "any_failure", "mean_is_area",
            ]
        )

    grouped = observations.groupby("run_id", sort=False)
    series = grouped.agg(
        acquisition_timestamp=("acquisition_timestamp", "min"),
        mean_bias_percent=("percent_bias", "mean"),
        mean_measured=("measured_value", "mean"),
        n_replicates=("qc_result_id", "count"),
        mean_is_area=("is_area", "mean"),
    ).reset_index()

    replicate_cv = grouped.apply(
        lambda rows: coefficient_of_variation(rows["measured_value"], minimum_points=2),
        include_groups=False,
    )
    series["within_run_cv_percent"] = series["run_id"].map(replicate_cv)

    failure_flags = grouped["pass_fail"].apply(
        lambda values: bool(values.astype(str).str.lower().eq("fail").any())
    )
    series["any_failure"] = series["run_id"].map(failure_flags)

    series = series.sort_values("acquisition_timestamp").reset_index(drop=True)
    series["run_index"] = np.arange(1, len(series) + 1)
    return series


def add_rolling_statistics(run_series: pd.DataFrame, window: int = 8) -> pd.DataFrame:
    """Add run-ordered rolling mean, SD and CV of the per-run bias."""
    output = run_series.copy()
    if output.empty:
        for column in ("rolling_mean_bias", "rolling_sd_bias", "rolling_cv_percent"):
            output[column] = pd.Series(dtype=float)
        return output

    bias = output["mean_bias_percent"]
    rolling = bias.rolling(window=window, min_periods=max(3, window // 2))
    output["rolling_mean_bias"] = rolling.mean()
    output["rolling_sd_bias"] = rolling.std(ddof=1)

    measured = output["mean_measured"].rolling(window=window, min_periods=max(3, window // 2))
    with np.errstate(invalid="ignore", divide="ignore"):
        output["rolling_cv_percent"] = (
            measured.std(ddof=1) / measured.mean().abs() * 100.0
        )
    return output


def align_timestamp_to(timestamps: pd.Series, value) -> pd.Timestamp:
    """Match a boundary timestamp to the timezone of a timestamp series.

    Instrument exports carry timezone-aware acquisition times while config files
    hold plain dates, and comparing the two raises rather than coercing.
    """
    boundary = pd.Timestamp(value)
    series_timezone = getattr(pd.to_datetime(timestamps).dtype, "tz", None)

    if series_timezone is None:
        return boundary.tz_localize(None) if boundary.tzinfo is not None else boundary

    if boundary.tzinfo is None:
        return boundary.tz_localize(series_timezone)
    return boundary.tz_convert(series_timezone)


@dataclass(frozen=True)
class BaselineStatistics:
    """Statistics from the frozen baseline window."""

    n_runs: int
    mean_bias_percent: float
    sd_bias_percent: float
    cv_percent: float
    start: str
    end: str

    @property
    def is_usable(self) -> bool:
        return self.n_runs >= MINIMUM_POINTS_FOR_SPREAD and np.isfinite(self.sd_bias_percent)


def compute_baseline(
    run_series: pd.DataFrame,
    baseline_start: str,
    baseline_end: str,
) -> BaselineStatistics:
    """Summarize the frozen baseline window of a run series."""
    if run_series.empty:
        return BaselineStatistics(0, float("nan"), float("nan"), float("nan"), baseline_start, baseline_end)

    timestamps = pd.to_datetime(run_series["acquisition_timestamp"])
    window = run_series[
        (timestamps >= align_timestamp_to(timestamps, baseline_start))
        & (timestamps <= align_timestamp_to(timestamps, baseline_end))
    ]
    if window.empty:
        return BaselineStatistics(0, float("nan"), float("nan"), float("nan"), baseline_start, baseline_end)

    bias = pd.to_numeric(window["mean_bias_percent"], errors="coerce").dropna()
    return BaselineStatistics(
        n_runs=int(len(window)),
        mean_bias_percent=float(bias.mean()) if len(bias) else float("nan"),
        sd_bias_percent=float(bias.std(ddof=1)) if len(bias) >= 2 else float("nan"),
        cv_percent=coefficient_of_variation(window["mean_measured"]),
        start=baseline_start,
        end=baseline_end,
    )


def theil_sen_trend(run_series: pd.DataFrame, value_column: str = "mean_bias_percent") -> dict[str, float]:
    """Estimate a robust drift rate per day, with a rank-based significance test.

    Theil-Sen resists the isolated outliers that are normal in QC data, and
    Kendall's tau tests monotonicity without assuming a linear model.
    """
    usable = run_series.dropna(subset=[value_column])
    if len(usable) < MINIMUM_POINTS_FOR_SPREAD:
        return {"slope_per_day": float("nan"), "kendall_tau": float("nan"), "kendall_p": float("nan")}

    timestamps = pd.to_datetime(usable["acquisition_timestamp"])
    elapsed_days = (timestamps - timestamps.min()).dt.total_seconds().to_numpy() / 86400.0
    values = pd.to_numeric(usable[value_column], errors="coerce").to_numpy(dtype=float)

    if np.ptp(elapsed_days) <= 0:
        return {"slope_per_day": float("nan"), "kendall_tau": float("nan"), "kendall_p": float("nan")}

    slope = float(stats.theilslopes(values, elapsed_days)[0])
    tau_result = stats.kendalltau(elapsed_days, values)
    return {
        "slope_per_day": slope,
        "kendall_tau": float(tau_result.statistic),
        "kendall_p": float(tau_result.pvalue),
        "span_days": float(np.ptp(elapsed_days)),
    }


def variance_ratio_p_value(
    recent_values: pd.Series,
    baseline_sd: float,
    baseline_n: int,
    recent_cv: float,
    baseline_cv: float,
) -> float:
    """One-sided F test that recent spread exceeds baseline spread.

    The CV ratio says how large the change is; this says whether a change that
    size is distinguishable from sampling noise at these run counts. Both are
    required, because either alone produces alerts nobody should act on.
    """
    usable = pd.to_numeric(recent_values, errors="coerce").dropna()
    recent_n = len(usable)

    if recent_n < 2 or baseline_n < 2:
        return 1.0
    if not np.isfinite(recent_cv) or not np.isfinite(baseline_cv) or baseline_cv <= 0:
        return 1.0

    # Compared on the CV scale so the two windows stay comparable even if the
    # QC level's absolute magnitude differs between them.
    variance_ratio = (recent_cv / baseline_cv) ** 2
    if not np.isfinite(variance_ratio) or variance_ratio <= 0:
        return 1.0

    return float(stats.f.sf(variance_ratio, recent_n - 1, baseline_n - 1))


def apply_false_discovery_control(p_values: pd.Series, alpha: float = 0.05) -> pd.Series:
    """Benjamini-Hochberg across the analysis grid.

    Method x instrument x QC level x metric is a large family of simultaneous
    tests; without control roughly one in twenty fires every cycle with nothing
    wrong, and an alerting system that cries wolf gets switched off.
    """
    usable = pd.to_numeric(p_values, errors="coerce")
    ranked = usable.rank(method="first")
    count = usable.notna().sum()
    if count == 0:
        return pd.Series(False, index=p_values.index)

    thresholds = alpha * ranked / count
    below = usable <= thresholds
    if not below.any():
        return pd.Series(False, index=p_values.index)

    largest_passing_rank = ranked[below].max()
    return (ranked <= largest_passing_rank) & usable.notna()
