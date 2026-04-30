import numpy as np
import pandas as pd
import pytest

pytest.importorskip("catboost")

from analysis.advanced_methods import run_catboost_shap


def test_real_catboost_shap_runs_on_small_mixed_dataset():
    row_count = 48
    feed_rate = np.linspace(30.0, 50.0, row_count)
    supplier = np.where(np.arange(row_count) % 3 == 0, "Supplier_B", "Supplier_A")
    yield_g_l = 1.5 + 0.11 * feed_rate + np.where(supplier == "Supplier_B", -0.4, 0.0)
    dataframe = pd.DataFrame(
        {
            "feed_rate_day3_mL_h": feed_rate,
            "raw_material_lot_supplier": supplier,
            "yield_g_L": yield_g_l,
        }
    )

    result = run_catboost_shap(
        dataframe=dataframe,
        process_columns=["feed_rate_day3_mL_h", "raw_material_lot_supplier"],
        outcome_column="yield_g_L",
        iterations=60,
    )

    assert result["available"] is True
    assert result["n_rows_used"] == row_count
    assert result["categorical_feature_count"] == 1
    assert not result["variable_importance"].empty
    assert not result["shap_values"].empty
    assert result["variable_importance"].iloc[0]["process_variable"] == "feed_rate_day3_mL_h"
