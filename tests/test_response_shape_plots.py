import numpy as np
import pandas as pd

from utils.plots import build_response_shape_summary, response_shape_plot


def test_response_shape_summary_builds_quantile_bins_for_numeric_driver():
    variable_values = np.linspace(80.0, 120.0, 40)
    outcome_values = 85.0 - 0.05 * (variable_values - 100.0) ** 2
    dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{i:03d}" for i in range(40)],
            "sodium_hydroxide_g": variable_values,
            "yield_percent": outcome_values,
        }
    )

    summary = build_response_shape_summary(
        dataframe=dataframe,
        variable_col="sodium_hydroxide_g",
        outcome_col="yield_percent",
        bin_count=5,
    )

    assert len(summary) == 5
    assert summary["count"].sum() == 40
    assert summary["variable_midpoint"].is_monotonic_increasing
    middle_mean = summary.iloc[2]["outcome_mean"]
    edge_mean = min(summary.iloc[0]["outcome_mean"], summary.iloc[-1]["outcome_mean"])
    assert middle_mean > edge_mean


def test_response_shape_plot_adds_raw_points_binned_line_and_band_overlays():
    dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{i:03d}" for i in range(30)],
            "ph_setpoint": np.linspace(6.6, 7.6, 30),
            "aggregate_percent": np.linspace(3.0, 1.0, 30),
        }
    )
    response_band = {
        "range_min": 6.9,
        "range_max": 7.2,
        "refined_range_min": 6.98,
        "refined_range_max": 7.05,
    }

    figure = response_shape_plot(
        dataframe=dataframe,
        variable_col="ph_setpoint",
        outcome_col="aggregate_percent",
        response_band=response_band,
        bin_count=6,
    )

    assert len(figure.data) == 2
    assert figure.data[0].name == "Batch"
    assert figure.data[1].name == "Binned mean"
    assert len(figure.layout.shapes) == 2
