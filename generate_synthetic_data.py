"""
Generate synthetic test data for Batch Insight Analyzer.

Built-in validation relationships
---------------------------------
This dataset intentionally contains known relationships so the app can be
validated end to end:

1. yield_g_L is strongly and positively correlated with:
   - feed_rate_day3_mL_h
   - temperature_C
2. purity_percent drops by about 8 percentage points when:
   - raw_material_lot_supplier == "Supplier_B"
3. aggregate_percent has a non-linear U-shaped relationship with:
   - ph_setpoint
   The minimum is around pH 7.0, with aggregates rising below about 6.8 and
   above about 7.4.
4. hcp_ppm has an interaction effect:
   - high when temperature_C > 37.2 AND duration_hours > 168
5. bioreactor_id == "BR-3" produces about 10% lower yield independent of
   other parameters.
6. Seven outlier batches are failed runs with extreme process and QC values.
7. About 3% missing data is introduced randomly in non-critical process
   columns only. Batch IDs, QC outcomes, and planted-driver columns are kept
   complete so the intended signals remain easy to validate.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


RANDOM_SEED = 42
NUMBER_OF_BATCHES = 150
NUMBER_OF_OUTLIER_BATCHES = 7

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIRECTORY = PROJECT_ROOT / "data"
PROCESS_OUTPUT_PATH = DATA_DIRECTORY / "synthetic_process.csv"
QC_OUTPUT_PATH = DATA_DIRECTORY / "synthetic_qc.csv"


def clipped_normal(
    rng: np.random.Generator,
    mean: float,
    standard_deviation: float,
    minimum: float,
    maximum: float,
    size: int,
) -> np.ndarray:
    """Return normally distributed values clipped to a realistic range."""
    values = rng.normal(mean, standard_deviation, size)
    return np.clip(values, minimum, maximum)


def build_process_dataframe(rng: np.random.Generator) -> pd.DataFrame:
    """Create one row per batch with realistic single-value process parameters."""
    batch_numbers = np.arange(1, NUMBER_OF_BATCHES + 1)
    batch_ids = [f"B{batch_number:03d}" for batch_number in batch_numbers]

    bioreactor_ids = rng.choice(
        ["BR-1", "BR-2", "BR-3", "BR-4", "BR-5"],
        size=NUMBER_OF_BATCHES,
        p=[0.20, 0.21, 0.20, 0.20, 0.19],
    )
    suppliers = rng.choice(
        ["Supplier_A", "Supplier_B", "Supplier_C"],
        size=NUMBER_OF_BATCHES,
        p=[0.46, 0.28, 0.26],
    )
    operator_shifts = rng.choice(["Day", "Night"], size=NUMBER_OF_BATCHES, p=[0.58, 0.42])
    media_lots = rng.choice(
        [f"Lot-{lot_number:03d}" for lot_number in range(1, 13)],
        size=NUMBER_OF_BATCHES,
    )
    lab_rooms = rng.choice(["Lab-101", "Lab-102", "Lab-201"], size=NUMBER_OF_BATCHES)

    process_dataframe = pd.DataFrame(
        {
            "batch_id": batch_ids,
            "ph_setpoint": clipped_normal(rng, 7.03, 0.18, 6.55, 7.65, NUMBER_OF_BATCHES),
            "temperature_C": clipped_normal(rng, 36.9, 0.35, 35.8, 38.2, NUMBER_OF_BATCHES),
            "rpm": clipped_normal(rng, 122.0, 12.0, 90.0, 160.0, NUMBER_OF_BATCHES),
            "DO_percent": clipped_normal(rng, 42.0, 7.0, 25.0, 65.0, NUMBER_OF_BATCHES),
            "feed_rate_day3_mL_h": clipped_normal(
                rng, 38.0, 5.5, 24.0, 54.0, NUMBER_OF_BATCHES
            ),
            "feed_rate_day5_mL_h": clipped_normal(
                rng, 46.0, 6.5, 30.0, 64.0, NUMBER_OF_BATCHES
            ),
            "inoculation_density_E6_cells_mL": clipped_normal(
                rng, 0.82, 0.12, 0.50, 1.15, NUMBER_OF_BATCHES
            ),
            "glucose_setpoint_g_L": clipped_normal(
                rng, 4.6, 0.55, 3.0, 6.2, NUMBER_OF_BATCHES
            ),
            "harvest_VCD_E6_cells_mL": clipped_normal(
                rng, 14.5, 2.2, 8.0, 21.0, NUMBER_OF_BATCHES
            ),
            "duration_hours": clipped_normal(rng, 166.0, 9.0, 140.0, 190.0, NUMBER_OF_BATCHES),
            "aeration_rate_vvm": clipped_normal(
                rng, 0.38, 0.055, 0.22, 0.55, NUMBER_OF_BATCHES
            ),
            "pressure_bar": clipped_normal(rng, 0.82, 0.08, 0.62, 1.05, NUMBER_OF_BATCHES),
            "agitator_tip_speed_m_s": clipped_normal(
                rng, 1.55, 0.16, 1.10, 2.05, NUMBER_OF_BATCHES
            ),
            "bioreactor_id": bioreactor_ids,
            "raw_material_lot_supplier": suppliers,
            "operator_shift": operator_shifts,
            "media_lot": media_lots,
            "deviation_reported": rng.binomial(1, 0.08, NUMBER_OF_BATCHES),
            "feed_strategy_alternative": rng.binomial(1, 0.18, NUMBER_OF_BATCHES),
            "ambient_humidity_percent": clipped_normal(
                rng, 47.0, 8.0, 25.0, 72.0, NUMBER_OF_BATCHES
            ),
            "batch_sequence_number": batch_numbers,
            "lab_room_id": lab_rooms,
            # Additional neutral process variables to reach 25 process parameters.
            "osmolality_mOsm_kg": clipped_normal(
                rng, 315.0, 12.0, 280.0, 355.0, NUMBER_OF_BATCHES
            ),
            "seed_train_age_hours": clipped_normal(
                rng, 72.0, 6.0, 55.0, 90.0, NUMBER_OF_BATCHES
            ),
            "antifoam_addition_mL": clipped_normal(
                rng, 4.0, 1.4, 0.0, 8.5, NUMBER_OF_BATCHES
            ),
        }
    )

    return process_dataframe


def build_qc_dataframe(
    process_dataframe: pd.DataFrame,
    rng: np.random.Generator,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Create QC outcomes with known planted relationships."""
    centered_feed_rate = process_dataframe["feed_rate_day3_mL_h"] - 38.0
    centered_temperature = process_dataframe["temperature_C"] - 36.9

    base_yield = (
        4.6
        + 0.105 * centered_feed_rate
        + 0.95 * centered_temperature
        + rng.normal(0.0, 0.24, NUMBER_OF_BATCHES)
    )
    br3_yield_penalty = np.where(process_dataframe["bioreactor_id"] == "BR-3", 0.90, 1.0)
    yield_g_l = base_yield * br3_yield_penalty

    purity_percent = (
        96.0
        - np.where(process_dataframe["raw_material_lot_supplier"] == "Supplier_B", 8.0, 0.0)
        + rng.normal(0.0, 1.1, NUMBER_OF_BATCHES)
    )

    ph_distance_from_ideal = process_dataframe["ph_setpoint"] - 7.0
    aggregate_percent = (
        1.0
        + 24.0 * ph_distance_from_ideal**2
        + np.where(process_dataframe["ph_setpoint"] < 6.8, 1.2, 0.0)
        + np.where(process_dataframe["ph_setpoint"] > 7.4, 1.6, 0.0)
        + rng.normal(0.0, 0.22, NUMBER_OF_BATCHES)
    )

    high_temperature_long_duration = (
        (process_dataframe["temperature_C"] > 37.2)
        & (process_dataframe["duration_hours"] > 168.0)
    )
    hcp_ppm = (
        125.0
        + 9.5 * (process_dataframe["duration_hours"] - 166.0)
        + np.where(high_temperature_long_duration, 310.0, 0.0)
        + rng.normal(0.0, 35.0, NUMBER_OF_BATCHES)
    )

    qc_dataframe = pd.DataFrame(
        {
            "batch_id": process_dataframe["batch_id"],
            "yield_g_L": np.clip(yield_g_l, 1.2, None),
            "purity_percent": np.clip(purity_percent, 70.0, 99.9),
            "hcp_ppm": np.clip(hcp_ppm, 20.0, None),
            "aggregate_percent": np.clip(aggregate_percent, 0.2, None),
        }
    )

    outlier_indices = rng.choice(
        process_dataframe.index,
        size=NUMBER_OF_OUTLIER_BATCHES,
        replace=False,
    )

    return qc_dataframe, outlier_indices


def apply_outlier_runs(
    process_dataframe: pd.DataFrame,
    qc_dataframe: pd.DataFrame,
    outlier_indices: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """Mutate selected batches into failed runs with extreme values."""
    for row_index in outlier_indices:
        low_ph_failure = rng.random() < 0.5

        process_dataframe.loc[row_index, "deviation_reported"] = 1
        process_dataframe.loc[row_index, "temperature_C"] = rng.uniform(34.9, 35.7)
        process_dataframe.loc[row_index, "duration_hours"] = rng.uniform(184.0, 205.0)
        process_dataframe.loc[row_index, "DO_percent"] = rng.uniform(12.0, 23.0)
        process_dataframe.loc[row_index, "feed_rate_day3_mL_h"] = rng.uniform(15.0, 23.0)
        process_dataframe.loc[row_index, "rpm"] = rng.uniform(65.0, 88.0)
        process_dataframe.loc[row_index, "ph_setpoint"] = (
            rng.uniform(6.35, 6.65) if low_ph_failure else rng.uniform(7.58, 7.85)
        )

        qc_dataframe.loc[row_index, "yield_g_L"] = rng.uniform(1.4, 2.6)
        qc_dataframe.loc[row_index, "purity_percent"] = rng.uniform(74.0, 84.0)
        qc_dataframe.loc[row_index, "hcp_ppm"] = rng.uniform(620.0, 1050.0)
        qc_dataframe.loc[row_index, "aggregate_percent"] = rng.uniform(7.0, 13.5)


def introduce_missing_process_values(
    process_dataframe: pd.DataFrame,
    rng: np.random.Generator,
) -> None:
    """Introduce about 3% missing values in non-critical process columns."""
    non_critical_columns = [
        "rpm",
        "DO_percent",
        "feed_rate_day5_mL_h",
        "inoculation_density_E6_cells_mL",
        "glucose_setpoint_g_L",
        "harvest_VCD_E6_cells_mL",
        "aeration_rate_vvm",
        "pressure_bar",
        "agitator_tip_speed_m_s",
        "operator_shift",
        "media_lot",
        "deviation_reported",
        "feed_strategy_alternative",
        "ambient_humidity_percent",
        "batch_sequence_number",
        "lab_room_id",
        "osmolality_mOsm_kg",
        "seed_train_age_hours",
        "antifoam_addition_mL",
    ]

    missing_mask = rng.random((len(process_dataframe), len(non_critical_columns))) < 0.03
    for column_index, column_name in enumerate(non_critical_columns):
        process_dataframe.loc[missing_mask[:, column_index], column_name] = np.nan


def validate_outputs(process_dataframe: pd.DataFrame, qc_dataframe: pd.DataFrame) -> None:
    """Fail fast if the generated files drift from the phase 1 contract."""
    expected_process_shape = (NUMBER_OF_BATCHES, 26)
    expected_qc_shape = (NUMBER_OF_BATCHES, 5)

    if process_dataframe.shape != expected_process_shape:
        raise ValueError(
            f"Expected process data shape {expected_process_shape}, "
            f"got {process_dataframe.shape}."
        )

    if qc_dataframe.shape != expected_qc_shape:
        raise ValueError(f"Expected QC data shape {expected_qc_shape}, got {qc_dataframe.shape}.")

    if process_dataframe["batch_id"].duplicated().any() or qc_dataframe["batch_id"].duplicated().any():
        raise ValueError("Synthetic data should not contain duplicate batch IDs.")

    if not process_dataframe["batch_id"].equals(qc_dataframe["batch_id"]):
        raise ValueError("Process and QC batch IDs should match exactly.")


def main() -> None:
    rng = np.random.default_rng(RANDOM_SEED)
    DATA_DIRECTORY.mkdir(exist_ok=True)

    process_dataframe = build_process_dataframe(rng)
    qc_dataframe, outlier_indices = build_qc_dataframe(process_dataframe, rng)

    apply_outlier_runs(process_dataframe, qc_dataframe, outlier_indices, rng)
    introduce_missing_process_values(process_dataframe, rng)
    validate_outputs(process_dataframe, qc_dataframe)

    process_dataframe.to_csv(PROCESS_OUTPUT_PATH, index=False)
    qc_dataframe.to_csv(QC_OUTPUT_PATH, index=False)

    outlier_batch_ids = process_dataframe.loc[outlier_indices, "batch_id"].sort_values().to_list()

    print(f"Wrote {PROCESS_OUTPUT_PATH}")
    print(f"Wrote {QC_OUTPUT_PATH}")
    print(f"Process shape: {process_dataframe.shape}")
    print(f"QC shape: {qc_dataframe.shape}")
    print(f"Outlier batches: {', '.join(outlier_batch_ids)}")


if __name__ == "__main__":
    main()
