import numpy as np
import pandas as pd

from analysis.evidence import build_operating_window_hints
from analysis.profiling import profile_merged_data


def test_operating_window_hints_preserve_middle_sweet_spots():
    sodium_hydroxide = np.linspace(76.0, 124.0, 96)
    yield_percent = 82.0 - 0.035 * (sodium_hydroxide - 101.0) ** 2
    dataframe = pd.DataFrame(
        {
            "batch_id": [f"B{i:03d}" for i in range(96)],
            "sodium_hydroxide_g": sodium_hydroxide,
            "yield_percent": yield_percent,
        }
    )
    profile_result = profile_merged_data(
        dataframe,
        outcome_columns=["yield_percent"],
        batch_id_column="batch_id",
    )
    ranked_drivers = pd.DataFrame(
        [
            {
                "outcome": "yield_percent",
                "rank": 1,
                "process_variable": "sodium_hydroxide_g",
                "combined_score": 0.9,
                "confidence": "high",
                "evidence_methods": "PLS, Random Forest",
            }
        ]
    )

    hints = build_operating_window_hints(
        ranked_drivers=ranked_drivers,
        merged_dataframe=dataframe,
        profile_result=profile_result,
        outcomes=["yield_percent"],
    )

    assert len(hints) == 1
    hint = hints.iloc[0]
    assert hint["pattern_type"] == "middle_sweet_spot"
    assert 88.0 <= hint["range_min"] <= 101.0
    assert 101.0 <= hint["range_max"] <= 113.0
    assert pd.notna(hint["refined_range_label"])
    assert hint["refined_pattern_type"] == "middle_best_bin"
