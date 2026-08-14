"""Shared display formatting for the Streamlit UI and generated downloads."""

from __future__ import annotations

from datetime import datetime
from typing import Any

import pandas as pd


DISPLAY_DECIMALS = 3


def format_percent(value: Any) -> str:
    """Format a decimal ratio as a percentage."""
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.0%}"
    except (TypeError, ValueError):
        return "n/a"


def format_metric(value: Any) -> str:
    """Format a metric value that may be missing."""
    try:
        if pd.isna(value):
            return "n/a"
        return f"{float(value):.{DISPLAY_DECIMALS}f}"
    except (TypeError, ValueError):
        return "n/a"


def format_numeric_columns(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Round numeric columns for compact display."""
    display_dataframe = dataframe.copy()
    numeric_columns = display_dataframe.select_dtypes(include="number").columns
    display_dataframe[numeric_columns] = display_dataframe[numeric_columns].round(
        DISPLAY_DECIMALS
    )
    return display_dataframe


def datetime_stamp_for_filename() -> str:
    """Return a compact timestamp for generated downloads."""
    return datetime.now().strftime("%Y%m%d_%H%M")
