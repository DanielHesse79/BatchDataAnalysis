"""Mapping profile helpers for repeat customer file formats."""

from __future__ import annotations

from datetime import datetime
import json
from typing import Any


def build_mapping_profile(
    process_file_name: str,
    qc_file_name: str,
    process_intake_options: dict[str, Any],
    qc_intake_options: dict[str, Any],
    process_batch_id_column: str,
    qc_batch_id_column: str,
    outcome_columns: list[str],
    process_duplicate_strategy: str,
    qc_duplicate_strategy: str,
) -> dict[str, Any]:
    """Build a portable mapping profile for repeated imports."""
    return {
        "profile_type": "batch_insight_mapping_profile",
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "version": 1,
        "process_file_name": process_file_name,
        "qc_file_name": qc_file_name,
        "process_intake": process_intake_options,
        "qc_intake": qc_intake_options,
        "mapping": {
            "process_batch_id_column": process_batch_id_column,
            "qc_batch_id_column": qc_batch_id_column,
            "outcome_columns": outcome_columns,
        },
        "duplicate_handling": {
            "process": process_duplicate_strategy,
            "qc": qc_duplicate_strategy,
        },
    }


def mapping_profile_to_json(profile: dict[str, Any]) -> str:
    """Serialize a mapping profile for download."""
    return json.dumps(profile, indent=2, ensure_ascii=True)
