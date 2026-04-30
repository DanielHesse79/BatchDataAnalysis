from pathlib import Path

import pandas as pd
import pytest

from analysis.audit import run_preflight_audit
from analysis.data_prep import merge_process_and_qc_data
from analysis.methods import run_all_analyses
from analysis.profiling import profile_merged_data
from analysis.specs import assess_specs


PROJECT_ROOT = Path(__file__).resolve().parents[1]


SYNTHETIC_OUTCOMES = [
    "yield_g_L",
    "purity_percent",
    "hcp_ppm",
    "aggregate_percent",
]

MOCK_SPEC_OUTCOMES = [
    "yield_percent",
    "moisture_percent",
    "residual_naoh_ppm",
    "color_index",
]


@pytest.fixture(scope="session")
def synthetic_pipeline():
    """Run the main synthetic validation pipeline once per test session."""
    process_dataframe = pd.read_csv(PROJECT_ROOT / "data" / "synthetic_process.csv")
    qc_dataframe = pd.read_csv(PROJECT_ROOT / "data" / "synthetic_qc.csv")
    merge_result = merge_process_and_qc_data(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="batch_id",
        qc_batch_id_column="batch_id",
        outcome_columns=SYNTHETIC_OUTCOMES,
    )
    profile_result = profile_merged_data(
        merge_result.dataframe,
        outcome_columns=SYNTHETIC_OUTCOMES,
        matched_batch_count=merge_result.matched_batch_count,
    )
    audit_result = run_preflight_audit(
        merge_result.dataframe,
        process_columns=profile_result.process_columns,
        outcome_columns=profile_result.outcome_columns,
        variable_types=profile_result.variable_types,
    )
    analysis_results = run_all_analyses(
        merge_result.dataframe,
        process_columns=profile_result.process_columns,
        outcome_columns=profile_result.outcome_columns,
    )
    return {
        "merge": merge_result,
        "dataframe": merge_result.dataframe,
        "profile": profile_result,
        "audit": audit_result,
        "analysis": analysis_results,
        "outcomes": SYNTHETIC_OUTCOMES,
    }


@pytest.fixture(scope="session")
def mock_spec_pipeline():
    """Run the mock specs/windows pipeline once per test session."""
    process_dataframe = pd.read_csv(PROJECT_ROOT / "data" / "mock_spec_process.csv")
    qc_dataframe = pd.read_csv(PROJECT_ROOT / "data" / "mock_spec_qc.csv")
    spec_dataframe = pd.read_csv(PROJECT_ROOT / "data" / "mock_spec_specs.csv")
    merge_result = merge_process_and_qc_data(
        process_dataframe=process_dataframe,
        qc_dataframe=qc_dataframe,
        process_batch_id_column="batch_id",
        qc_batch_id_column="batch_id",
        outcome_columns=MOCK_SPEC_OUTCOMES,
    )
    profile_result = profile_merged_data(
        merge_result.dataframe,
        outcome_columns=MOCK_SPEC_OUTCOMES,
        matched_batch_count=merge_result.matched_batch_count,
    )
    audit_result = run_preflight_audit(
        merge_result.dataframe,
        process_columns=profile_result.process_columns,
        outcome_columns=profile_result.outcome_columns,
        variable_types=profile_result.variable_types,
    )
    analysis_results = run_all_analyses(
        merge_result.dataframe,
        process_columns=profile_result.process_columns,
        outcome_columns=profile_result.outcome_columns,
    )
    spec_assessment = assess_specs(
        dataframe=merge_result.dataframe,
        spec_dataframe=spec_dataframe,
        process_columns=profile_result.process_columns,
        outcome_columns=profile_result.outcome_columns,
        ranked_drivers=analysis_results["ranked_drivers"],
        audit_result=audit_result,
    )
    return {
        "merge": merge_result,
        "dataframe": merge_result.dataframe,
        "profile": profile_result,
        "audit": audit_result,
        "analysis": analysis_results,
        "specs": spec_assessment,
        "outcomes": MOCK_SPEC_OUTCOMES,
    }
