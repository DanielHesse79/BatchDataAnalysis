import numpy as np
import pandas as pd
import pytest

pytest.importorskip("xgboost")

from analysis.advanced_methods import run_xgboost_shap


def test_real_xgboost_shap_runs_on_small_mixed_dataset():
    row_count = 60
    rng = np.random.default_rng(42)
    feed_rate = np.linspace(30.0, 50.0, row_count)
    supplier = np.where(np.arange(row_count) % 3 == 0, "Supplier_B", "Supplier_A")
    noise = rng.normal(0.0, 1.0, size=row_count)
    yield_g_l = (
        1.5
        + 0.11 * feed_rate
        + np.where(supplier == "Supplier_B", -0.4, 0.0)
        + rng.normal(0.0, 0.02, size=row_count)
    )
    dataframe = pd.DataFrame(
        {
            "feed_rate_day3_mL_h": feed_rate,
            "raw_material_lot_supplier": supplier,
            "noise_signal": noise,
            "yield_g_L": yield_g_l,
        }
    )

    result = run_xgboost_shap(
        dataframe=dataframe,
        process_columns=["feed_rate_day3_mL_h", "raw_material_lot_supplier", "noise_signal"],
        outcome_column="yield_g_L",
        n_estimators=120,
    )

    assert result["available"] is True
    assert result["n_rows_used"] == row_count
    assert result["numeric_feature_count"] == 2  # feed_rate, noise_signal
    assert result["categorical_feature_count"] == 1  # supplier
    assert not result["variable_importance"].empty
    assert not result["shap_values"].empty

    # feed_rate is the planted dominant driver; the random noise column should rank below it.
    ranked = result["variable_importance"].set_index("process_variable")["xgboost_score"]
    assert ranked.idxmax() == "feed_rate_day3_mL_h"
    assert ranked["feed_rate_day3_mL_h"] > ranked["noise_signal"]

    # Encoded SHAP collapses back to the three original process variables.
    assert set(result["shap_values"]["process_variable"].unique()) == {
        "feed_rate_day3_mL_h",
        "raw_material_lot_supplier",
        "noise_signal",
    }
