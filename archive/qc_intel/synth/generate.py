"""Synthetic QC history with known, planted behaviour.

Three methods, three LC-MS/MS instruments, three QC levels, twelve months.
The planted patterns are the acceptance test for the detectors, so the ground
truth is written alongside the data as JSON rather than living only in prose:

    ASSAY_A   stable everywhere                         (negative control)
    ASSAY_B   LCMS-02 drifts downward from month 4      (gradual drift)
    ASSAY_C   step change at a reference-standard lot   (step change)
    ASSAY_C   LCMS-03 loses precision from month 6      (variance increase)
    all       occasional isolated QC failures           (random noise)

Synthetic data can only show that the code works. Whether the product is useful
is a question only a real method-year can answer.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
import json

import numpy as np
import pandas as pd


RANDOM_SEED = 20260814
STUDY_START = datetime(2025, 1, 6, 8, 0, tzinfo=timezone.utc)
MONTHS = 12
# Nine runs a month per method, rotated across three instruments, gives about
# 36 runs per method-instrument pair over the year - a realistic cadence for a
# busy specialty method, and enough to populate a four-month baseline.
RUNS_PER_MONTH = 9
REPLICATES_PER_LEVEL = 2

INSTRUMENTS = ["LCMS-01", "LCMS-02", "LCMS-03"]

METHOD_SPECS = {
    "ASSAY_A": {
        "name": "Compound A in human plasma",
        "version": "3.2",
        "analyte": "Compound A",
        "units": "ng/mL",
        "lloq": 1.0,
        "uloq": 100.0,
        "levels": {"low_qc": 3.0, "mid_qc": 30.0, "high_qc": 80.0},
    },
    "ASSAY_B": {
        "name": "Compound B in human plasma",
        "version": "1.4",
        "analyte": "Compound B",
        "units": "ng/mL",
        "lloq": 0.5,
        "uloq": 250.0,
        "levels": {"low_qc": 1.5, "mid_qc": 75.0, "high_qc": 200.0},
    },
    "ASSAY_C": {
        "name": "Compound C in human serum",
        "version": "2.0",
        "analyte": "Compound C",
        "units": "ng/mL",
        "lloq": 2.0,
        "uloq": 500.0,
        "levels": {"low_qc": 6.0, "mid_qc": 150.0, "high_qc": 400.0},
    },
}

# --- planted behaviour -------------------------------------------------------
DRIFT_METHOD, DRIFT_INSTRUMENT = "ASSAY_B", "LCMS-02"
DRIFT_START_MONTH = 4
DRIFT_TOTAL_PERCENT = -11.0

STEP_METHOD = "ASSAY_C"
STEP_MONTH = 7
STEP_PERCENT = 6.5

VARIANCE_METHOD, VARIANCE_INSTRUMENT = "ASSAY_C", "LCMS-03"
VARIANCE_START_MONTH = 6
VARIANCE_CV_START, VARIANCE_CV_END = 3.0, 8.5

BASELINE_CV_PERCENT = 3.0
RANDOM_FAILURE_RATE = 0.012


@dataclass(frozen=True)
class SyntheticDataset:
    """Generated tables plus the ground truth they encode."""

    methods: pd.DataFrame
    instruments: pd.DataFrame
    materials: pd.DataFrame
    runs: pd.DataFrame
    qc_results: pd.DataFrame
    calibrations: pd.DataFrame
    calibrator_points: pd.DataFrame
    events: pd.DataFrame
    run_materials: pd.DataFrame
    truth: dict


def _month_index(timestamp: datetime) -> int:
    return (timestamp.year - STUDY_START.year) * 12 + (timestamp.month - STUDY_START.month)


def _planted_bias(method_id: str, instrument_id: str, timestamp: datetime) -> float:
    """Return the systematic bias in percent for one run."""
    month = _month_index(timestamp)
    bias = 0.0

    if method_id == DRIFT_METHOD and instrument_id == DRIFT_INSTRUMENT and month >= DRIFT_START_MONTH:
        progress = (month - DRIFT_START_MONTH) / max(1, MONTHS - 1 - DRIFT_START_MONTH)
        bias += DRIFT_TOTAL_PERCENT * min(1.0, progress)

    if method_id == STEP_METHOD and month >= STEP_MONTH:
        bias += STEP_PERCENT

    return bias


def _planted_cv(method_id: str, instrument_id: str, timestamp: datetime) -> float:
    """Return the run-level CV in percent."""
    month = _month_index(timestamp)
    if (
        method_id == VARIANCE_METHOD
        and instrument_id == VARIANCE_INSTRUMENT
        and month >= VARIANCE_START_MONTH
    ):
        progress = (month - VARIANCE_START_MONTH) / max(1, MONTHS - 1 - VARIANCE_START_MONTH)
        return VARIANCE_CV_START + (VARIANCE_CV_END - VARIANCE_CV_START) * min(1.0, progress)
    return BASELINE_CV_PERCENT


def generate(seed: int = RANDOM_SEED) -> SyntheticDataset:
    """Generate the full synthetic dataset."""
    generator = np.random.default_rng(seed)

    methods = pd.DataFrame(
        [
            {
                "method_id": method_id,
                "method_name": spec["name"],
                "method_version": spec["version"],
                "analyte": spec["analyte"],
                "matrix": "human plasma" if "plasma" in spec["name"] else "human serum",
                "platform": "LC-MS/MS",
                "units": spec["units"],
                "lloq": spec["lloq"],
                "uloq": spec["uloq"],
                "effective_date": "2024-11-01",
                "retired_date": None,
            }
            for method_id, spec in METHOD_SPECS.items()
        ]
    )

    instruments = pd.DataFrame(
        [
            {
                "instrument_id": instrument_id,
                "instrument_model": "Xevo TQ-S",
                "asset_tag": f"ASSET-{index + 4100}",
                "installation_date": "2022-03-15",
                "software_version": "MassLynx 4.2 SCN1015",
            }
            for index, instrument_id in enumerate(INSTRUMENTS)
        ]
    )

    material_rows, event_rows = [], []
    reference_lots: dict[str, list[tuple[str, datetime]]] = {}

    for method_id in METHOD_SPECS:
        for lot_index in range(2):
            opened = STUDY_START + timedelta(days=lot_index * 30 * STEP_MONTH)
            material_id = f"REF-{method_id}-{lot_index + 1:02d}"
            material_rows.append(
                {
                    "material_id": material_id,
                    "material_type": "reference_standard",
                    "manufacturer": "Certified Standards Ltd",
                    "lot_number": f"RS{2025_000 + lot_index + 1}",
                    "date_opened": opened.date().isoformat(),
                    "expiry_date": (opened + timedelta(days=365)).date().isoformat(),
                    "concentration": 1000.0,
                }
            )
            reference_lots.setdefault(method_id, []).append((material_id, opened))
            if lot_index > 0:
                event_rows.append(
                    {
                        "event_id": f"EV-LOT-{method_id}-{lot_index}",
                        "entity_type": "material",
                        "entity_id": material_id,
                        "event_type": "new_reference_standard_lot",
                        "event_timestamp": opened.isoformat(timespec="seconds"),
                        "description": f"New reference standard lot opened for {method_id}",
                        "ingest_id": None,
                    }
                )

    for instrument_index, instrument_id in enumerate(INSTRUMENTS):
        for month in range(0, MONTHS, 4):
            service_time = STUDY_START + timedelta(days=month * 30 + instrument_index * 3 + 2)
            event_rows.append(
                {
                    "event_id": f"EV-PM-{instrument_id}-{month}",
                    "entity_type": "instrument",
                    "entity_id": instrument_id,
                    "event_type": "preventive_maintenance",
                    "event_timestamp": service_time.isoformat(timespec="seconds"),
                    "description": "Scheduled PM including source clean",
                    "ingest_id": None,
                }
            )

    # A column change close to the variance problem, so the dashboard shows a
    # plausible-but-wrong association next to the real one.
    column_change_time = STUDY_START + timedelta(days=VARIANCE_START_MONTH * 30 + 1)
    material_rows.append(
        {
            "material_id": "COL-0007",
            "material_type": "column",
            "manufacturer": "Waters",
            "lot_number": "ACQ-BEH-C18-0007",
            "date_opened": column_change_time.date().isoformat(),
            "expiry_date": None,
            "concentration": None,
        }
    )
    event_rows.append(
        {
            "event_id": "EV-COL-0007",
            "entity_type": "instrument",
            "entity_id": VARIANCE_INSTRUMENT,
            "event_type": "column_replacement",
            "event_timestamp": column_change_time.isoformat(timespec="seconds"),
            "description": "Analytical column replaced (ACQUITY BEH C18)",
            "ingest_id": None,
        }
    )

    run_rows, qc_rows, calibration_rows, calibrator_rows, run_material_rows = [], [], [], [], []
    run_counter = 0

    for month in range(MONTHS):
        for run_in_month in range(RUNS_PER_MONTH):
            for method_index, (method_id, spec) in enumerate(METHOD_SPECS.items()):
                # Rotate deterministically so every method runs on every
                # instrument. hash() would be randomised per process, and an
                # offset that tracks the method index alone would pin each
                # method to one instrument, making cross-instrument comparison
                # impossible and hiding the planted instrument-specific drift.
                rotation = month * RUNS_PER_MONTH + run_in_month + method_index
                instrument_id = INSTRUMENTS[rotation % len(INSTRUMENTS)]
                timestamp = STUDY_START + timedelta(
                    days=month * 30 + run_in_month * 4,
                    hours=(run_counter % 5) * 2,
                )
                run_counter += 1
                run_id = f"R{run_counter:05d}"

                lots = reference_lots[method_id]
                active_lot = lots[0][0]
                for material_id, opened in lots:
                    if timestamp >= opened:
                        active_lot = material_id

                systematic_bias = _planted_bias(method_id, instrument_id, timestamp)
                run_cv = _planted_cv(method_id, instrument_id, timestamp)
                run_offset = generator.normal(0.0, run_cv * 0.6)

                run_rows.append(
                    {
                        "run_id": run_id,
                        "method_id": method_id,
                        "instrument_id": instrument_id,
                        "study_id": f"STUDY-{2025}-{(month // 3) + 1:02d}",
                        "acquisition_timestamp": timestamp.isoformat(timespec="seconds"),
                        "run_type": "sample_analysis",
                        "acceptance_status": "accepted",
                        "acceptance_reason": None,
                        "column_id": "COL-0007" if instrument_id == VARIANCE_INSTRUMENT else None,
                        "column_injection_count": 40 * (month + 1),
                        "processing_method_version": "1.0",
                        "analyst_ref": f"ANALYST-{(run_counter % 4) + 1:02d}",
                        "ingest_id": None,
                        "source_row": f"synthetic:{run_id}",
                    }
                )
                run_material_rows.append(
                    {"run_id": run_id, "material_id": active_lot, "role": "reference_standard"}
                )

                injection_index = 0
                run_had_failure = False
                for level_name, nominal in spec["levels"].items():
                    for replicate in range(1, REPLICATES_PER_LEVEL + 1):
                        injection_index += 1
                        noise = generator.normal(0.0, run_cv)
                        bias_percent = systematic_bias + run_offset + noise

                        if generator.random() < RANDOM_FAILURE_RATE:
                            bias_percent += generator.choice([-1, 1]) * generator.uniform(16, 24)

                        measured = nominal * (1.0 + bias_percent / 100.0)
                        limit = 20.0 if level_name == "lloq_qc" else 15.0
                        passed = abs(bias_percent) <= limit
                        run_had_failure = run_had_failure or not passed

                        internal_standard_area = generator.normal(150000, 6000)
                        if method_id == DRIFT_METHOD and instrument_id == DRIFT_INSTRUMENT:
                            internal_standard_area *= 1.0 + systematic_bias / 250.0

                        qc_rows.append(
                            {
                                "run_id": run_id,
                                "qc_level": level_name,
                                "evaluation_type": "quantitative",
                                "nominal_value": nominal,
                                "measured_value": round(measured, 4),
                                "units": spec["units"],
                                "percent_bias": round(bias_percent, 4),
                                "replicate_number": replicate,
                                "injection_index": injection_index,
                                "analyte_area": round(internal_standard_area * measured / nominal, 1),
                                "is_area": round(internal_standard_area, 1),
                                "area_ratio": round(measured / nominal, 5),
                                "retention_time": round(generator.normal(4.21, 0.02), 3),
                                "manual_integration": 0,
                                "pass_fail": "pass" if passed else "fail",
                                "acceptance_limit_lower": -limit,
                                "acceptance_limit_upper": limit,
                                "ingest_id": None,
                                "source_row": f"synthetic:{run_id}:{level_name}:{replicate}",
                            }
                        )

                calibration_id = f"CAL-{run_id}"
                slope = 0.0412 * (1.0 + generator.normal(0.0, 0.012) + systematic_bias / 400.0)
                calibration_rows.append(
                    {
                        "calibration_id": calibration_id,
                        "run_id": run_id,
                        "calibration_model": "linear",
                        "weighting": "1/x^2",
                        "slope": round(slope, 6),
                        "intercept": round(generator.normal(0.0006, 0.0002), 6),
                        "r_squared": round(min(0.9999, generator.normal(0.9985, 0.0009)), 5),
                        "calibration_status": "accepted",
                        "number_of_calibrators": 8,
                        "ingest_id": None,
                    }
                )
                for calibrator_index in range(8):
                    nominal_calibrator = spec["lloq"] * (2.2 ** calibrator_index)
                    deviation = generator.normal(systematic_bias, run_cv)
                    calibrator_rows.append(
                        {
                            "calibration_id": calibration_id,
                            "level_name": f"STD{calibrator_index + 1}",
                            "nominal_value": round(nominal_calibrator, 4),
                            "back_calculated": round(
                                nominal_calibrator * (1 + deviation / 100.0), 4
                            ),
                            "percent_deviation": round(deviation, 3),
                            "included": int(abs(deviation) <= 15.0),
                        }
                    )

                if run_had_failure:
                    run_rows[-1]["acceptance_reason"] = "one or more QC replicates outside limits"

    truth = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "seed": seed,
        "stable_method": "ASSAY_A",
        "gradual_drift": {
            "method_id": DRIFT_METHOD,
            "instrument_id": DRIFT_INSTRUMENT,
            "starts_month": DRIFT_START_MONTH,
            "total_percent": DRIFT_TOTAL_PERCENT,
        },
        "step_change": {
            "method_id": STEP_METHOD,
            "at_month": STEP_MONTH,
            "percent": STEP_PERCENT,
            "coincides_with": "new_reference_standard_lot",
        },
        "variance_increase": {
            "method_id": VARIANCE_METHOD,
            "instrument_id": VARIANCE_INSTRUMENT,
            "starts_month": VARIANCE_START_MONTH,
            "cv_from": VARIANCE_CV_START,
            "cv_to": VARIANCE_CV_END,
        },
        "random_failure_rate": RANDOM_FAILURE_RATE,
    }

    return SyntheticDataset(
        methods=methods,
        instruments=instruments,
        materials=pd.DataFrame(material_rows),
        runs=pd.DataFrame(run_rows),
        qc_results=pd.DataFrame(qc_rows),
        calibrations=pd.DataFrame(calibration_rows),
        calibrator_points=pd.DataFrame(calibrator_rows),
        events=pd.DataFrame(event_rows),
        run_materials=pd.DataFrame(run_material_rows),
        truth=truth,
    )


def write_source_files(dataset: SyntheticDataset, output_dir: Path | str) -> dict[str, Path]:
    """Write the dataset as source files, the way a lab export would arrive."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    written: dict[str, Path] = {}
    tables = {
        "methods": dataset.methods,
        "instruments": dataset.instruments,
        "materials": dataset.materials,
        "runs": dataset.runs,
        "qc_results": dataset.qc_results,
        "calibrations": dataset.calibrations,
        "calibrator_points": dataset.calibrator_points,
        "events": dataset.events,
        "run_materials": dataset.run_materials,
    }
    for name, frame in tables.items():
        path = output_dir / f"{name}.csv"
        frame.to_csv(path, index=False)
        written[name] = path

    truth_path = output_dir / "ground_truth.json"
    truth_path.write_text(json.dumps(dataset.truth, indent=2), encoding="utf-8")
    written["ground_truth"] = truth_path
    return written


if __name__ == "__main__":
    dataset = generate()
    paths = write_source_files(dataset, Path(__file__).resolve().parent.parent / "data" / "source")
    print(f"runs={len(dataset.runs):,}  qc_results={len(dataset.qc_results):,}")
    for name, path in paths.items():
        print(f"  {name}: {path.name}")
