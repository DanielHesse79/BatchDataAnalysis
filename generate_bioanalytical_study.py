"""Generate a synthetic bioanalytical study sample analysis dataset.

Run-level bioanalytical data - calibration curves, QC runs, incurred sample
reanalysis, repeats, deviations - is a CRO's working material and is never
published. There is no public dataset to test a report generator against, so
this makes one.

The point is not that the numbers look plausible. The point is the awkward
cases, planted on purpose:

    a rejected run whose samples were re-assayed
    a run with a rejected calibrator that still passes the 75% rule
    repeats with different reasons and different reporting rules
    samples below LLOQ and above ULOQ
    incurred sample reanalysis with a couple of results outside the criterion
    a deviation spanning several runs

Those are what break naive automation. A clean dataset where everything passes
proves nothing, because the report a CRO actually writes is mostly about the
exceptions.

Everything planted is written to ground_truth.json, so a future report engine
can be asserted against known answers rather than hand-computed ones. That is
the same discipline as qc_intel/synth/generate.py.

Deterministic: a fixed seed and a fixed start date, so the dataset is identical
on every machine and a diff means something changed.

    python generate_bioanalytical_study.py
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
import json

import numpy as np
import pandas as pd


OUTPUT_DIR = Path(__file__).resolve().parent / "data" / "bioanalytical_study"
SEED = 20260821

# ------------------------------------------------------------------ the assay
# A fictional small molecule in human plasma by LC-MS/MS. The calibration range
# is deliberately narrower than the observed peak concentrations, so some
# samples land above ULOQ and have to be diluted and repeated.
STUDY_ID = "BE-2026-014"
METHOD_ID = "MTX-207-PL-01"
METHOD_VERSION = "2.0"
ANALYTE = "MTX-207"
MATRIX = "Human plasma (K2EDTA)"
UNITS = "ng/mL"

LLOQ = 1.00
ULOQ = 500.0
CALIBRATOR_LEVELS = {
    "STD-1": 1.00, "STD-2": 2.00, "STD-3": 5.00, "STD-4": 20.0,
    "STD-5": 50.0, "STD-6": 150.0, "STD-7": 350.0, "STD-8": 500.0,
}
QC_LEVELS = {"LLOQ QC": 1.00, "Low QC": 3.00, "Mid QC": 50.0, "High QC": 400.0}

# ICH M10 acceptance criteria. Kept here as data so the generator and any
# future report engine read the same numbers from one place.
CALIBRATOR_TOLERANCE_PERCENT = 15.0
CALIBRATOR_TOLERANCE_AT_LLOQ_PERCENT = 20.0
CALIBRATOR_PASS_FRACTION = 0.75
QC_TOLERANCE_PERCENT = 15.0
QC_TOLERANCE_AT_LLOQ_PERCENT = 20.0
QC_RUN_PASS_FRACTION = 2 / 3
ISR_TOLERANCE_PERCENT = 20.0
ISR_PASS_FRACTION = 2 / 3

# ------------------------------------------------------------------ the study
SUBJECT_COUNT = 30
TIMEPOINTS_HOURS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 24.0, 48.0]
SAMPLES_PER_RUN = 26
FIRST_RUN_DATE = datetime(2026, 3, 2, 8, 15)

INSTRUMENTS = ["LCMS-04", "LCMS-05"]
ANALYSTS = ["A-17", "A-04", "A-23"]

# ------------------------------------------------------------- planted events
# Referenced by run number so the ground truth and the generator cannot drift
# apart. Run numbering is 1-based and matches the run_id suffix.
REJECTED_RUN = 7                 # QC failure; its samples are re-assayed
REJECTED_CALIBRATOR_RUN = 12     # one calibrator dropped, curve still accepted
REJECTED_CALIBRATOR_LEVEL = "STD-6"
FREEZER_EXCURSION_RUNS = (9, 10, 11)

# A slow between-run bias drift across the study. Not a failure - it stays well
# inside +/-15% - but it is the kind of thing a trending report should surface
# and a summary table should not hide.
BIAS_DRIFT_START_PERCENT = -1.0
BIAS_DRIFT_END_PERCENT = 4.0

WITHIN_RUN_CV_PERCENT = 3.5
BETWEEN_RUN_CV_PERCENT = 4.5


@dataclass
class Dataset:
    """Everything the generator produces, plus what it planted."""

    runs: pd.DataFrame
    calibration_curves: pd.DataFrame
    calibration_standards: pd.DataFrame
    qc_results: pd.DataFrame
    study_samples: pd.DataFrame
    repeat_analysis: pd.DataFrame
    incurred_sample_reanalysis: pd.DataFrame
    deviations: pd.DataFrame
    truth: dict = field(default_factory=dict)


def concentration_profile(rng: np.random.Generator) -> dict:
    """Draw one subject's PK parameters.

    A one-compartment oral model is enough. What matters for the report is that
    concentrations span the calibration range and spill over both ends of it.
    """
    return {
        "dose_ng": 35e6,
        "clearance": float(np.exp(rng.normal(np.log(6.0), 0.30))),
        "volume": float(np.exp(rng.normal(np.log(70.0), 0.25))),
        "absorption_rate": float(np.exp(rng.normal(np.log(1.1), 0.35))),
    }


def true_concentration(parameters: dict, hours: float) -> float:
    """Concentration at one timepoint, before any assay error."""
    if hours <= 0:
        return 0.0

    elimination_rate = parameters["clearance"] / parameters["volume"]
    absorption_rate = parameters["absorption_rate"]
    if abs(absorption_rate - elimination_rate) < 1e-6:
        absorption_rate += 1e-3

    scale = (parameters["dose_ng"] * absorption_rate) / (
        parameters["volume"] * 1000.0 * (absorption_rate - elimination_rate)
    )
    return float(
        scale * (np.exp(-elimination_rate * hours) - np.exp(-absorption_rate * hours))
    )


def run_bias_percent(run_number: int, run_count: int) -> float:
    """The planted between-run bias drift, in percent, for one run."""
    if run_count <= 1:
        return BIAS_DRIFT_START_PERCENT
    position = (run_number - 1) / (run_count - 1)
    span = BIAS_DRIFT_END_PERCENT - BIAS_DRIFT_START_PERCENT
    return BIAS_DRIFT_START_PERCENT + span * position


def tolerance_for(level_name: str) -> float:
    """QC and calibrator tolerance widens at the lower limit."""
    return (
        QC_TOLERANCE_AT_LLOQ_PERCENT
        if "LLOQ" in level_name or level_name == "STD-1"
        else QC_TOLERANCE_PERCENT
    )


def build_runs(run_count: int, rng: np.random.Generator) -> pd.DataFrame:
    """The run list, including the rejected run and its reason."""
    rows = []
    for run_number in range(1, run_count + 1):
        run_date = FIRST_RUN_DATE + timedelta(days=int(1.6 * (run_number - 1)))
        rejected = run_number == REJECTED_RUN
        rows.append({
            "run_id": f"R{run_number:03d}",
            "study_id": STUDY_ID,
            "method_id": METHOD_ID,
            "instrument_id": INSTRUMENTS[run_number % len(INSTRUMENTS)],
            "analyst_ref": ANALYSTS[run_number % len(ANALYSTS)],
            "acquisition_timestamp": run_date.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_type": "study sample analysis",
            "acceptance_status": "rejected" if rejected else "accepted",
            "acceptance_reason": (
                "Both Mid QC replicates and one High QC replicate outside "
                "+/-15% of nominal; fewer than 2/3 of QC results acceptable"
                if rejected else ""
            ),
        })
    return pd.DataFrame(rows)


def build_calibration(
    runs: pd.DataFrame, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Calibration curves and their back-calculated standards.

    One run carries a rejected calibrator. Seven of eight remain, which is 87.5%
    and above the 75% rule, so the curve is still accepted - the case a report
    has to show correctly rather than quietly drop.
    """
    curve_rows = []
    standard_rows = []

    for _, run in runs.iterrows():
        run_number = int(run["run_id"][1:])
        accepted_count = 0

        for level_name, nominal in CALIBRATOR_LEVELS.items():
            forced_rejection = (
                run_number == REJECTED_CALIBRATOR_RUN
                and level_name == REJECTED_CALIBRATOR_LEVEL
            )
            if forced_rejection:
                deviation = float(rng.uniform(18.0, 24.0)) * rng.choice([-1.0, 1.0])
            else:
                deviation = float(rng.normal(0.0, 4.2))

            measured = nominal * (1.0 + deviation / 100.0)
            included = abs(deviation) <= tolerance_for(level_name)
            accepted_count += int(included)

            standard_rows.append({
                "run_id": run["run_id"],
                "level_name": level_name,
                "nominal_value": nominal,
                "back_calculated": round(measured, 4 if nominal < 10 else 2),
                "percent_deviation": round(deviation, 2),
                "included": int(included),
                "units": UNITS,
            })

        curve_rows.append({
            "run_id": run["run_id"],
            "calibration_model": "linear",
            "weighting": "1/x^2",
            "slope": round(float(rng.normal(0.00412, 0.00009)), 6),
            "intercept": round(float(rng.normal(0.00021, 0.00006)), 6),
            "r_squared": round(float(rng.uniform(0.9958, 0.9994)), 4),
            "number_of_calibrators": len(CALIBRATOR_LEVELS),
            "calibrators_accepted": accepted_count,
            "calibration_status": (
                "accepted"
                if accepted_count / len(CALIBRATOR_LEVELS) >= CALIBRATOR_PASS_FRACTION
                else "rejected"
            ),
        })

    return pd.DataFrame(curve_rows), pd.DataFrame(standard_rows)


def build_qc_results(runs: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """QC in duplicate at four levels per run, with one run failed on purpose."""
    rows = []
    run_count = len(runs)

    for _, run in runs.iterrows():
        run_number = int(run["run_id"][1:])
        between_run_offset = float(rng.normal(0.0, BETWEEN_RUN_CV_PERCENT))
        drift = run_bias_percent(run_number, run_count)

        for level_name, nominal in QC_LEVELS.items():
            for replicate in (1, 2):
                if run_number == REJECTED_RUN and level_name in {"Mid QC", "High QC"}:
                    # The planted failure: large, one-sided, and confined to
                    # this run - which is what an injection or preparation
                    # problem looks like.
                    bias = float(rng.uniform(16.5, 22.0))
                else:
                    bias = (
                        drift
                        + between_run_offset
                        + float(rng.normal(0.0, WITHIN_RUN_CV_PERCENT))
                    )

                measured = nominal * (1.0 + bias / 100.0)
                rows.append({
                    "run_id": run["run_id"],
                    "qc_level": level_name,
                    "nominal_value": nominal,
                    "measured_value": round(measured, 4 if nominal < 10 else 2),
                    "percent_bias": round(bias, 2),
                    "replicate_number": replicate,
                    "units": UNITS,
                    "pass_fail": "pass" if abs(bias) <= tolerance_for(level_name) else "fail",
                })

    return pd.DataFrame(rows)


def build_study_samples(
    runs: pd.DataFrame, rng: np.random.Generator,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Study samples assigned to runs, plus the repeat analysis table.

    Samples from the rejected run are re-assayed in the next run; samples above
    ULOQ are diluted and repeated; a few are repeated for chromatography. Each
    reason carries a different rule for which value gets reported, which is
    exactly the logic a generated table has to get right.
    """
    accepted_run_ids = runs.loc[runs["acceptance_status"] == "accepted", "run_id"].tolist()
    rejected_run_id = f"R{REJECTED_RUN:03d}"

    sample_rows = []
    repeat_rows = []
    sample_index = 0

    for subject_number in range(1, SUBJECT_COUNT + 1):
        subject_ref = f"S{subject_number:03d}"
        parameters = concentration_profile(rng)

        for hours in TIMEPOINTS_HOURS:
            sample_id = f"{STUDY_ID}-{sample_index + 1:04d}"
            run_position = sample_index // SAMPLES_PER_RUN
            assigned_run = (
                rejected_run_id
                if run_position == REJECTED_RUN - 1
                else accepted_run_ids[min(run_position, len(accepted_run_ids) - 1)]
            )

            true_value = true_concentration(parameters, hours)
            measured = true_value * (1.0 + float(rng.normal(0.0, 0.055)))

            if measured < LLOQ:
                flag, reported = "BLQ", None
            elif measured > ULOQ:
                flag, reported = "ALQ", None
            else:
                flag, reported = "reportable", round(measured, 4 if measured < 10 else 2)

            sample_rows.append({
                "sample_id": sample_id,
                "subject_ref": subject_ref,
                "timepoint_hours": hours,
                "run_id": assigned_run,
                "dilution_factor": 1,
                "measured_value": reported,
                "result_flag": flag,
                "units": UNITS,
            })

            # Above the upper limit: dilute and repeat, report the repeat.
            if flag == "ALQ":
                repeat_run = _next_run(assigned_run, accepted_run_ids)
                diluted = measured * (1.0 + float(rng.normal(0.0, 0.05)))
                repeat_rows.append({
                    "sample_id": sample_id,
                    "original_run_id": assigned_run,
                    "original_value": None,
                    "repeat_run_id": repeat_run,
                    "repeat_value": round(diluted, 2),
                    "dilution_factor": 5,
                    "reason": "Result above ULOQ; reanalysed with 5-fold dilution",
                    "reported_value": round(diluted, 2),
                    "reported_from": "repeat",
                })

            sample_index += 1

    samples = pd.DataFrame(sample_rows)

    # The rejected run: every sample in it is re-assayed in the following run.
    reassigned = samples["run_id"] == rejected_run_id
    replacement_run = _next_run(rejected_run_id, accepted_run_ids)
    for sample_id in samples.loc[reassigned, "sample_id"]:
        repeat_rows.append({
            "sample_id": sample_id,
            "original_run_id": rejected_run_id,
            "original_value": None,
            "repeat_run_id": replacement_run,
            "repeat_value": None,
            "dilution_factor": 1,
            "reason": "Run rejected on QC acceptance criteria; samples reanalysed",
            "reported_value": None,
            "reported_from": "repeat",
        })

    # A few chromatographic repeats, reported from the original by SOP.
    reportable = samples.loc[samples["result_flag"] == "reportable", "sample_id"].tolist()
    for sample_id in rng.choice(reportable, size=3, replace=False):
        original = samples.loc[samples["sample_id"] == sample_id].iloc[0]
        repeat_value = float(original["measured_value"]) * (1.0 + float(rng.normal(0.0, 0.06)))
        repeat_rows.append({
            "sample_id": str(sample_id),
            "original_run_id": original["run_id"],
            "original_value": original["measured_value"],
            "repeat_run_id": _next_run(original["run_id"], accepted_run_ids),
            "repeat_value": round(repeat_value, 2),
            "dilution_factor": 1,
            "reason": "Integration failure; chromatographic interference at the peak",
            "reported_value": original["measured_value"],
            "reported_from": "original",
        })

    return samples, pd.DataFrame(repeat_rows)


def _next_run(run_id: str, accepted_run_ids: list[str]) -> str:
    """The next accepted run after this one, for reanalysis."""
    later = [candidate for candidate in accepted_run_ids if candidate > run_id]
    return later[0] if later else accepted_run_ids[-1]


def build_incurred_sample_reanalysis(
    samples: pd.DataFrame, runs: pd.DataFrame, rng: np.random.Generator,
) -> pd.DataFrame:
    """ISR on 10% of reportable samples, with two results outside the criterion.

    A dataset showing 100% agreement would be the wrong test: the report has to
    state the pass rate against the 2/3 criterion, so some failures must exist.
    """
    reportable = samples.loc[samples["result_flag"] == "reportable"].copy()
    selected_count = max(1, int(round(0.10 * len(reportable))))
    selected = reportable.sample(n=selected_count, random_state=int(rng.integers(1e6)))

    accepted_run_ids = runs.loc[runs["acceptance_status"] == "accepted", "run_id"].tolist()
    outliers = set(selected["sample_id"].tolist()[:2])

    rows = []
    for _, sample in selected.iterrows():
        original = float(sample["measured_value"])
        if sample["sample_id"] in outliers:
            difference = float(rng.uniform(22.0, 28.0)) * rng.choice([-1.0, 1.0])
        else:
            difference = float(rng.normal(0.0, 6.5))

        repeat_value = original * (1.0 + difference / 100.0)
        mean_value = (original + repeat_value) / 2.0
        percent_difference = 100.0 * (repeat_value - original) / mean_value

        rows.append({
            "sample_id": sample["sample_id"],
            "original_run_id": sample["run_id"],
            "original_value": original,
            "isr_run_id": accepted_run_ids[-1],
            "isr_value": round(repeat_value, 4 if repeat_value < 10 else 2),
            "percent_difference": round(percent_difference, 2),
            "within_criterion": int(abs(percent_difference) <= ISR_TOLERANCE_PERCENT),
            "units": UNITS,
        })

    return pd.DataFrame(rows)


def build_deviations() -> pd.DataFrame:
    """Deviations, including one spanning several runs."""
    affected = ", ".join(f"R{number:03d}" for number in FREEZER_EXCURSION_RUNS)
    return pd.DataFrame([
        {
            "deviation_id": "DEV-001",
            "deviation_type": "Storage",
            "description": (
                "Freezer FR-12 recorded -14 C for 6 hours. Study samples for the "
                "affected runs were stored in this unit."
            ),
            "runs_affected": affected,
            "assessed_impact": (
                "No impact. Long-term stability at -20 C covers the excursion "
                "duration and temperature; QC results in the affected runs met "
                "acceptance criteria."
            ),
        },
        {
            "deviation_id": "DEV-002",
            "deviation_type": "Analytical",
            "description": (
                f"Run R{REJECTED_RUN:03d} rejected on QC acceptance criteria. "
                "Samples reanalysed in the following run."
            ),
            "runs_affected": f"R{REJECTED_RUN:03d}",
            "assessed_impact": (
                "No impact on reported results. No sample was reported from the "
                "rejected run."
            ),
        },
    ])


def summarise_qc(qc_results: pd.DataFrame, runs: pd.DataFrame) -> list[dict]:
    """The QC summary a report must reproduce, computed from accepted runs only."""
    accepted = set(runs.loc[runs["acceptance_status"] == "accepted", "run_id"])
    usable = qc_results[qc_results["run_id"].isin(accepted)]

    summary = []
    for level_name, nominal in QC_LEVELS.items():
        level_rows = usable[usable["qc_level"] == level_name]
        measured = level_rows["measured_value"]
        summary.append({
            "qc_level": level_name,
            "nominal_value": nominal,
            "n_runs": int(level_rows["run_id"].nunique()),
            "n_observations": int(len(level_rows)),
            "mean": round(float(measured.mean()), 4),
            "standard_deviation": round(float(measured.std(ddof=1)), 4),
            "cv_percent": round(float(100.0 * measured.std(ddof=1) / measured.mean()), 2),
            "bias_percent": round(float(100.0 * (measured.mean() - nominal) / nominal), 2),
        })
    return summary


def generate() -> Dataset:
    """Build the whole study."""
    rng = np.random.default_rng(SEED)

    total_samples = SUBJECT_COUNT * len(TIMEPOINTS_HOURS)
    run_count = int(np.ceil(total_samples / SAMPLES_PER_RUN)) + 1

    runs = build_runs(run_count, rng)
    curves, standards = build_calibration(runs, rng)
    qc_results = build_qc_results(runs, rng)
    samples, repeats = build_study_samples(runs, rng)
    isr = build_incurred_sample_reanalysis(samples, runs, rng)
    deviations = build_deviations()

    isr_pass_rate = 100.0 * isr["within_criterion"].mean()
    truth = {
        "study_id": STUDY_ID,
        "method_id": METHOD_ID,
        "analyte": ANALYTE,
        "matrix": MATRIX,
        "units": UNITS,
        "calibration_range": [LLOQ, ULOQ],
        "acceptance_criteria": {
            "calibrator_tolerance_percent": CALIBRATOR_TOLERANCE_PERCENT,
            "calibrator_tolerance_at_lloq_percent": CALIBRATOR_TOLERANCE_AT_LLOQ_PERCENT,
            "calibrator_pass_fraction": CALIBRATOR_PASS_FRACTION,
            "qc_tolerance_percent": QC_TOLERANCE_PERCENT,
            "qc_tolerance_at_lloq_percent": QC_TOLERANCE_AT_LLOQ_PERCENT,
            "qc_run_pass_fraction": QC_RUN_PASS_FRACTION,
            "isr_tolerance_percent": ISR_TOLERANCE_PERCENT,
            "isr_pass_fraction": ISR_PASS_FRACTION,
        },
        "planted": {
            "rejected_runs": [f"R{REJECTED_RUN:03d}"],
            "rejected_calibrator": {
                "run_id": f"R{REJECTED_CALIBRATOR_RUN:03d}",
                "level_name": REJECTED_CALIBRATOR_LEVEL,
                "note": "7 of 8 calibrators accepted; curve accepted under the 75% rule",
            },
            "freezer_excursion_runs": [f"R{number:03d}" for number in FREEZER_EXCURSION_RUNS],
            "between_run_bias_drift_percent": [
                BIAS_DRIFT_START_PERCENT, BIAS_DRIFT_END_PERCENT,
            ],
        },
        "expected": {
            "run_count": int(len(runs)),
            "accepted_run_count": int((runs["acceptance_status"] == "accepted").sum()),
            "rejected_run_count": int((runs["acceptance_status"] == "rejected").sum()),
            "sample_count": int(len(samples)),
            "reportable_sample_count": int((samples["result_flag"] == "reportable").sum()),
            "below_lloq_count": int((samples["result_flag"] == "BLQ").sum()),
            "above_uloq_count": int((samples["result_flag"] == "ALQ").sum()),
            "repeat_count": int(len(repeats)),
            "isr_sample_count": int(len(isr)),
            "isr_pass_rate_percent": round(float(isr_pass_rate), 2),
            "isr_meets_criterion": bool(isr_pass_rate >= 100.0 * ISR_PASS_FRACTION),
            "qc_summary_accepted_runs_only": summarise_qc(qc_results, runs),
        },
    }

    return Dataset(
        runs=runs,
        calibration_curves=curves,
        calibration_standards=standards,
        qc_results=qc_results,
        study_samples=samples,
        repeat_analysis=repeats,
        incurred_sample_reanalysis=isr,
        deviations=deviations,
        truth=truth,
    )


def write_dataset(dataset: Dataset, output_dir: Path = OUTPUT_DIR) -> None:
    """Write the CSVs and the ground truth."""
    output_dir.mkdir(parents=True, exist_ok=True)

    tables = {
        "runs.csv": dataset.runs,
        "calibration_curves.csv": dataset.calibration_curves,
        "calibration_standards.csv": dataset.calibration_standards,
        "qc_results.csv": dataset.qc_results,
        "study_samples.csv": dataset.study_samples,
        "repeat_analysis.csv": dataset.repeat_analysis,
        "incurred_sample_reanalysis.csv": dataset.incurred_sample_reanalysis,
        "deviations.csv": dataset.deviations,
    }
    for file_name, frame in tables.items():
        frame.to_csv(output_dir / file_name, index=False)

    (output_dir / "ground_truth.json").write_text(
        json.dumps(dataset.truth, indent=2), encoding="utf-8",
    )


def main() -> None:
    dataset = generate()
    write_dataset(dataset)

    expected = dataset.truth["expected"]
    print(f"Wrote {OUTPUT_DIR}")
    print(f"  study            {STUDY_ID}, {ANALYTE} in {MATRIX}")
    print(f"  runs             {expected['run_count']} "
          f"({expected['rejected_run_count']} rejected)")
    print(f"  samples          {expected['sample_count']} "
          f"({expected['reportable_sample_count']} reportable, "
          f"{expected['below_lloq_count']} BLQ, {expected['above_uloq_count']} ALQ)")
    print(f"  repeats          {expected['repeat_count']}")
    print(f"  ISR              {expected['isr_sample_count']} samples, "
          f"{expected['isr_pass_rate_percent']}% within +/-20% "
          f"(criterion {100 * ISR_PASS_FRACTION:.0f}%)")
    print()
    print("  QC summary, accepted runs only:")
    for row in expected["qc_summary_accepted_runs_only"]:
        print(f"    {row['qc_level']:<9} n={row['n_observations']:<4} "
              f"mean={row['mean']:<9} %CV={row['cv_percent']:<6} "
              f"%bias={row['bias_percent']}")


if __name__ == "__main__":
    main()
