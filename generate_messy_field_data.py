"""Generate deliberately messy field-style files for intake testing.

Outputs:
- data/messy_process_export.xlsx
- data/messy_qc_long.csv

The files include common real-world issues: title rows before headers, decimal
commas, units inside cells, inconsistent batch ID separators/case, duplicate
batch rows, and long-format QC results.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from openpyxl import load_workbook


RANDOM_SEED = 42
DATA_DIR = Path("data")


def main() -> None:
    DATA_DIR.mkdir(exist_ok=True)
    process_dataframe, qc_long_dataframe = build_messy_data()
    write_process_excel(process_dataframe, DATA_DIR / "messy_process_export.xlsx")
    qc_long_dataframe.to_csv(DATA_DIR / "messy_qc_long.csv", index=False)
    print("Wrote data/messy_process_export.xlsx")
    print("Wrote data/messy_qc_long.csv")


def build_messy_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build messy process and QC tables with known relationships."""
    rng = np.random.default_rng(RANDOM_SEED)
    batch_count = 72
    batch_numbers = np.arange(1, batch_count + 1)
    canonical_batch_ids = [f"MB{batch_number:03d}" for batch_number in batch_numbers]

    sodium_hydroxide = rng.normal(100, 7, batch_count).clip(78, 123)
    drying_time = rng.normal(8.0, 1.4, batch_count).clip(4.5, 12.5)
    humidity = rng.normal(52, 11, batch_count).clip(25, 85)
    reactor = rng.choice(["RX-1", "RX-2", "RX-3"], size=batch_count, p=[0.40, 0.35, 0.25])

    yield_percent = (
        82.0
        - 0.16 * (sodium_hydroxide - 101.0) ** 2
        - np.where(reactor == "RX-3", 2.2, 0.0)
        + rng.normal(0, 1.2, batch_count)
    )
    moisture_percent = (
        12.0
        + np.maximum(0, 7.0 - drying_time) * 3.0
        + np.maximum(0, humidity - 65.0) * 0.15
        + rng.normal(0, 1.0, batch_count)
    )

    process_rows = []
    for index, batch_id in enumerate(canonical_batch_ids):
        messy_batch_id = messy_process_batch_id(batch_id, index)
        process_rows.append(
            {
                " Batch ID ": messy_batch_id,
                "NaOH Added": format_grams(sodium_hydroxide[index], index),
                "Drying Time": f"{drying_time[index]:.2f} h",
                "Ambient Humidity": f"{humidity[index]:.1f} %",
                "Reactor ID": reactor[index],
                "Operator Shift": rng.choice(["Day", "Night"]),
                "Notes": "manual entry" if index % 17 == 0 else "",
            }
        )

    # Add duplicate rows to exercise the duplicate handling UI.
    process_rows.append({**process_rows[9], "NaOH Added": "101,2 g", "Notes": "duplicate export row"})
    process_rows.append({**process_rows[24], "Drying Time": "7,85 h", "Notes": "duplicate export row"})
    process_dataframe = pd.DataFrame(process_rows)

    qc_rows = []
    for index, batch_id in enumerate(canonical_batch_ids):
        for test_name, value in [
            ("yield_percent", yield_percent[index]),
            ("moisture_percent", moisture_percent[index]),
            ("color_index", max(0.7, 1.5 + 0.02 * abs(sodium_hydroxide[index] - 101.0) + rng.normal(0, 0.12))),
        ]:
            qc_rows.append(
                {
                    "Batch Identifier": messy_qc_batch_id(batch_id, index),
                    "Test Name": test_name,
                    "Result": format_qc_result(test_name, value, index),
                    "Result Date": f"2026-02-{(index % 27) + 1:02d}",
                }
            )

    # A QC replicate result for one batch/test.
    qc_rows.append(
        {
            "Batch Identifier": "MB010",
            "Test Name": "yield_percent",
            "Result": "80,4",
            "Result Date": "2026-03-02",
        }
    )
    qc_long_dataframe = pd.DataFrame(qc_rows)
    return process_dataframe, qc_long_dataframe


def write_process_excel(process_dataframe: pd.DataFrame, output_path: Path) -> None:
    """Write process data with title rows before the real header."""
    with pd.ExcelWriter(output_path, engine="openpyxl") as writer:
        process_dataframe.to_excel(
            writer,
            sheet_name="Production Export",
            index=False,
            startrow=2,
        )

    workbook = load_workbook(output_path)
    worksheet = workbook["Production Export"]
    worksheet["A1"] = "Example MES production export"
    worksheet["A2"] = "Generated for Batch Insight Analyzer intake testing"
    workbook.save(output_path)


def messy_process_batch_id(batch_id: str, index: int) -> str:
    """Return intentionally inconsistent process batch IDs."""
    if index % 5 == 0:
        return batch_id.replace("MB", "mb-")
    if index % 7 == 0:
        return f" {batch_id} "
    return batch_id


def messy_qc_batch_id(batch_id: str, index: int) -> str:
    """Return intentionally inconsistent QC batch IDs."""
    if index % 6 == 0:
        return batch_id.replace("MB", "MB-")
    if index % 11 == 0:
        return f" {batch_id.lower()} "
    return batch_id


def format_grams(value: float, index: int) -> str:
    """Format grams with units and occasional decimal commas."""
    if index % 13 == 0:
        return f"<{value:.1f} g".replace(".", ",")
    if index % 3 == 0:
        return f"{value:.1f} g".replace(".", ",")
    return f"{value:.1f} g"


def format_qc_result(test_name: str, value: float, index: int) -> str:
    """Format QC result with realistic text quirks."""
    if test_name == "moisture_percent" and value < 10 and index % 9 == 0:
        return "<10 %"
    if test_name.endswith("percent"):
        return f"{value:.2f} %".replace(".", ",") if index % 4 == 0 else f"{value:.2f} %"
    return f"{value:.3f}".replace(".", ",") if index % 4 == 0 else f"{value:.3f}"


if __name__ == "__main__":
    main()
