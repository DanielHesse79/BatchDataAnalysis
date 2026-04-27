"""Generate mock production/QC/spec data for testing Specs & Windows.

This dataset is intentionally different from the main bioprocess synthetic data.
It is designed to test the new spec/window workflow with concrete manufacturing
examples:

Known spec/window stories
-------------------------
1. sodium_hydroxide_g has a stated process window of 85-115 g around a 100 g
   target. Quality is best near 100 g; very low or very high NaOH worsens
   residual_naoh_ppm, purity_percent, and color_index. This should challenge
   whether the process window is too wide.
2. moisture_percent has an upper QC spec of 20%. Moisture increases when
   drying_time_hours is low and ambient_humidity_percent is high. Several
   batches are intentionally near or above the limit.
3. drying_temperature_C has a deliberately tight process window of 54-58 C, but
   the mock quality outcomes are mostly insensitive inside the broader observed
   range. This is meant to look potentially too narrow after driver analysis.
4. pH_after_neutralization has a target of 7.0, but the historical process runs
   closer to 7.18. This should trigger a target-off-center review.
5. reactor_id == "RX-3" has slightly lower yield, creating a categorical
   equipment pattern that should be investigated separately from numeric specs.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 123
NUMBER_OF_BATCHES = 180

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESS_OUTPUT_PATH = DATA_DIRECTORY / "mock_spec_process.csv"
QC_OUTPUT_PATH = DATA_DIRECTORY / "mock_spec_qc.csv"
SPECS_OUTPUT_PATH = DATA_DIRECTORY / "mock_spec_specs.csv"


def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    standard_deviation: float,
    minimum: float,
    maximum: float,
    size: int,
) -> np.ndarray:
    """Return clipped normally distributed values."""
    values = rng.normal(mean, standard_deviation, size)
    return np.clip(values, minimum, maximum)


def build_process_dataframe(rng: np.random.Generator) -> pd.DataFrame:
    """Create mock one-row-per-batch production data."""
    batch_numbers = np.arange(1, NUMBER_OF_BATCHES + 1)
    batch_ids = [f"MS{batch_number:03d}" for batch_number in batch_numbers]

    sodium_hydroxide = clipped_normal(rng, 100.0, 8.5, 76.0, 124.0, NUMBER_OF_BATCHES)
    # Force a few deliberate edge/out-of-window batches.
    low_edge_indices = rng.choice(NUMBER_OF_BATCHES, size=8, replace=False)
    high_edge_indices = rng.choice(
        [index for index in range(NUMBER_OF_BATCHES) if index not in low_edge_indices],
        size=8,
        replace=False,
    )
    sodium_hydroxide[low_edge_indices] = rng.uniform(78.0, 88.0, size=len(low_edge_indices))
    sodium_hydroxide[high_edge_indices] = rng.uniform(112.0, 124.0, size=len(high_edge_indices))

    drying_time = clipped_normal(rng, 8.2, 1.5, 4.5, 13.5, NUMBER_OF_BATCHES)
    short_dry_indices = rng.choice(NUMBER_OF_BATCHES, size=12, replace=False)
    drying_time[short_dry_indices] = rng.uniform(4.5, 6.2, size=len(short_dry_indices))

    dataframe = pd.DataFrame(
        {
            "batch_id": batch_ids,
            "batch_sequence_number": batch_numbers,
            "sodium_hydroxide_g": sodium_hydroxide,
            "drying_temperature_C": clipped_normal(
                rng,
                56.0,
                3.0,
                48.0,
                64.0,
                NUMBER_OF_BATCHES,
            ),
            "drying_time_hours": drying_time,
            "ambient_humidity_percent": clipped_normal(
                rng,
                52.0,
                11.0,
                24.0,
                82.0,
                NUMBER_OF_BATCHES,
            ),
            "pH_after_neutralization": clipped_normal(
                rng,
                7.18,
                0.09,
                6.85,
                7.45,
                NUMBER_OF_BATCHES,
            ),
            "mixing_time_min": clipped_normal(rng, 42.0, 6.0, 25.0, 60.0, NUMBER_OF_BATCHES),
            "water_addition_L": clipped_normal(rng, 250.0, 12.0, 220.0, 285.0, NUMBER_OF_BATCHES),
            "filtration_pressure_bar": clipped_normal(
                rng,
                1.15,
                0.12,
                0.85,
                1.55,
                NUMBER_OF_BATCHES,
            ),
            "hold_time_hours": clipped_normal(rng, 5.0, 1.2, 2.0, 9.0, NUMBER_OF_BATCHES),
            "reactor_id": rng.choice(["RX-1", "RX-2", "RX-3", "RX-4"], size=NUMBER_OF_BATCHES),
            "operator_shift": rng.choice(["Day", "Night"], size=NUMBER_OF_BATCHES, p=[0.6, 0.4]),
            "raw_material_lot": rng.choice(
                [f"RM-{lot_number:03d}" for lot_number in range(1, 10)],
                size=NUMBER_OF_BATCHES,
            ),
            "deviation_reported": rng.binomial(1, 0.07, NUMBER_OF_BATCHES),
        }
    )

    introduce_missing_values(dataframe, rng)
    return dataframe


def build_qc_dataframe(
    process_dataframe: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    """Create mock QC outcomes from planted production relationships."""
    sodium_distance = np.abs(process_dataframe["sodium_hydroxide_g"] - 100.0)
    low_sodium_penalty = np.clip(92.0 - process_dataframe["sodium_hydroxide_g"], 0.0, None)
    high_sodium_penalty = np.clip(process_dataframe["sodium_hydroxide_g"] - 108.0, 0.0, None)
    short_drying_penalty = np.clip(7.0 - process_dataframe["drying_time_hours"], 0.0, None)
    high_humidity_penalty = np.clip(process_dataframe["ambient_humidity_percent"] - 60.0, 0.0, None)
    reactor_penalty = np.where(process_dataframe["reactor_id"] == "RX-3", 3.5, 0.0)

    moisture = (
        12.3
        + 3.2 * short_drying_penalty
        + 0.22 * high_humidity_penalty
        + rng.normal(0.0, 1.0, NUMBER_OF_BATCHES)
    )
    purity = (
        98.2
        - 0.050 * sodium_distance**2
        - 0.25 * process_dataframe["deviation_reported"]
        + rng.normal(0.0, 0.7, NUMBER_OF_BATCHES)
    )
    residual_naoh = (
        120.0
        + 34.0 * high_sodium_penalty
        + 8.0 * sodium_distance
        + rng.normal(0.0, 35.0, NUMBER_OF_BATCHES)
    )
    color_index = (
        1.4
        + 0.11 * low_sodium_penalty
        + 0.04 * sodium_distance
        + rng.normal(0.0, 0.25, NUMBER_OF_BATCHES)
    )
    yield_percent = (
        87.0
        - 0.030 * sodium_distance**2
        - 0.35 * moisture
        - reactor_penalty
        + rng.normal(0.0, 1.4, NUMBER_OF_BATCHES)
    )

    qc_dataframe = pd.DataFrame(
        {
            "batch_id": process_dataframe["batch_id"],
            "moisture_percent": np.round(np.clip(moisture, 6.0, 28.0), 3),
            "purity_percent": np.round(np.clip(purity, 86.0, 100.0), 3),
            "yield_percent": np.round(np.clip(yield_percent, 60.0, 95.0), 3),
            "residual_naoh_ppm": np.round(np.clip(residual_naoh, 20.0, 950.0), 3),
            "color_index": np.round(np.clip(color_index, 0.5, 6.5), 3),
        }
    )
    return qc_dataframe


def build_spec_dataframe() -> pd.DataFrame:
    """Create specs/windows that exercise the spec assessment feature."""
    return pd.DataFrame(
        [
            {
                "variable": "sodium_hydroxide_g",
                "role": "process",
                "target": 100,
                "lower_limit": 85,
                "upper_limit": 115,
                "unit": "g",
                "criticality": "CPP",
                "notes": "Quality worsens near low/high NaOH edges; test for possibly too wide.",
            },
            {
                "variable": "drying_temperature_C",
                "role": "process",
                "target": 56,
                "lower_limit": 54,
                "upper_limit": 58,
                "unit": "C",
                "criticality": "process parameter",
                "notes": "Deliberately tight window with weak planted QC impact.",
            },
            {
                "variable": "drying_time_hours",
                "role": "process",
                "target": 8,
                "lower_limit": 6,
                "upper_limit": 12,
                "unit": "h",
                "criticality": "CPP",
                "notes": "Short drying time increases moisture.",
            },
            {
                "variable": "pH_after_neutralization",
                "role": "process",
                "target": 7.0,
                "lower_limit": 6.8,
                "upper_limit": 7.3,
                "unit": "pH",
                "criticality": "process parameter",
                "notes": "Historical mean is intentionally shifted above target.",
            },
            {
                "variable": "moisture_percent",
                "role": "qc",
                "target": "",
                "lower_limit": "",
                "upper_limit": 20,
                "unit": "%",
                "criticality": "CQA",
                "notes": "Upper release-style QC spec.",
            },
            {
                "variable": "purity_percent",
                "role": "qc",
                "target": "",
                "lower_limit": 95,
                "upper_limit": "",
                "unit": "%",
                "criticality": "CQA",
                "notes": "Lower release-style QC spec.",
            },
            {
                "variable": "yield_percent",
                "role": "qc",
                "target": "",
                "lower_limit": 75,
                "upper_limit": "",
                "unit": "%",
                "criticality": "business KPI",
                "notes": "Internal yield floor, not a release spec.",
            },
            {
                "variable": "residual_naoh_ppm",
                "role": "qc",
                "target": "",
                "lower_limit": "",
                "upper_limit": 500,
                "unit": "ppm",
                "criticality": "CQA",
                "notes": "Upper residual NaOH spec.",
            },
            {
                "variable": "color_index",
                "role": "qc",
                "target": "",
                "lower_limit": "",
                "upper_limit": 3.5,
                "unit": "index",
                "criticality": "quality attribute",
                "notes": "Upper appearance index spec.",
            },
        ]
    )


def introduce_missing_values(
    process_dataframe: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    """Add a small amount of missing non-critical production data."""
    columns = [
        "mixing_time_min",
        "water_addition_L",
        "filtration_pressure_bar",
        "hold_time_hours",
        "operator_shift",
        "raw_material_lot",
    ]
    missing_mask = rng.random((len(process_dataframe), len(columns))) < 0.025
    for column_index, column_name in enumerate(columns):
        process_dataframe.loc[missing_mask[:, column_index], column_name] = np.nan


def validate_outputs(process_dataframe: pd.DataFrame, qc_dataframe: pd.DataFrame) -> None:
    """Fail fast if generated mock files drift unexpectedly."""
    if process_dataframe.shape[0] != NUMBER_OF_BATCHES:
        raise ValueError("Mock process data has the wrong number of batches.")
    if qc_dataframe.shape[0] != NUMBER_OF_BATCHES:
        raise ValueError("Mock QC data has the wrong number of batches.")
    if process_dataframe["batch_id"].duplicated().any() or qc_dataframe["batch_id"].duplicated().any():
        raise ValueError("Mock data should not contain duplicate batch IDs.")
    if not process_dataframe["batch_id"].equals(qc_dataframe["batch_id"]):
        raise ValueError("Mock process and QC batch IDs should match exactly.")


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    DATA_DIRECTORY.mkdir(exist_ok=True)

    process_dataframe = build_process_dataframe(rng)
    qc_dataframe = build_qc_dataframe(process_dataframe, rng)
    spec_dataframe = build_spec_dataframe()

    validate_outputs(process_dataframe, qc_dataframe)

    process_dataframe.to_csv(PROCESS_OUTPUT_PATH, index=False)
    qc_dataframe.to_csv(QC_OUTPUT_PATH, index=False)
    spec_dataframe.to_csv(SPECS_OUTPUT_PATH, index=False)

    print(f"Wrote {PROCESS_OUTPUT_PATH}")
    print(f"Wrote {QC_OUTPUT_PATH}")
    print(f"Wrote {SPECS_OUTPUT_PATH}")
    print(f"Process shape: {process_dataframe.shape}")
    print(f"QC shape: {qc_dataframe.shape}")
    print(f"Moisture failures >20%: {(qc_dataframe['moisture_percent'] > 20).sum()}")
    print(f"Residual NaOH failures >500 ppm: {(qc_dataframe['residual_naoh_ppm'] > 500).sum()}")


if __name__ == "__main__":
    main()
