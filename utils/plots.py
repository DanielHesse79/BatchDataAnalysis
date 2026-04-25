"""Plotly chart helpers for Batch Insight Analyzer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


PLOT_TEMPLATE = "plotly_white"
METHOD_COLORS = {
    "PCA": "#64748b",
    "PLS": "#2563eb",
    "Random Forest": "#16a34a",
}


def pca_scatter(pca_result: dict, outcome_values: pd.Series) -> go.Figure:
    """PC1 vs PC2 scatter plot, colored by an outcome value."""
    transformed_data = pca_result["transformed_data"].copy()

    if "PC2" not in transformed_data.columns:
        transformed_data["PC2"] = 0.0

    aligned_outcome_values = outcome_values.reindex(transformed_data["source_index"])
    transformed_data["outcome_value"] = pd.to_numeric(aligned_outcome_values, errors="coerce").to_numpy()

    figure = px.scatter(
        transformed_data,
        x="PC1",
        y="PC2",
        color="outcome_value",
        hover_data={"source_index": True, "outcome_value": ":.3f"},
        color_continuous_scale="Viridis",
        template=PLOT_TEMPLATE,
        labels={
            "PC1": "PC1",
            "PC2": "PC2",
            "outcome_value": outcome_values.name or "Outcome",
            "source_index": "Row",
        },
    )
    figure.update_layout(height=460, margin=dict(l=20, r=20, t=40, b=20))
    return figure


def loading_plot(pca_loadings: pd.DataFrame, top_n: int = 25) -> go.Figure:
    """Plot signed PCA loadings on PC1 and PC2."""
    if "PC2" not in pca_loadings.columns:
        pca_loadings = pca_loadings.copy()
        pca_loadings["PC2"] = 0.0

    loadings = pca_loadings.copy()
    label_column = "feature" if "feature" in loadings.columns else "process_variable"
    loadings["distance"] = (loadings["PC1"] ** 2 + loadings["PC2"] ** 2) ** 0.5
    loadings = loadings.sort_values("distance", ascending=False).head(top_n)

    figure = px.scatter(
        loadings,
        x="PC1",
        y="PC2",
        text=label_column,
        color="process_variable" if "process_variable" in loadings.columns else None,
        hover_data=[label_column],
        template=PLOT_TEMPLATE,
        labels={"PC1": "PC1 loading", "PC2": "PC2 loading"},
    )
    figure.update_traces(textposition="top center", marker=dict(size=9))
    figure.add_hline(y=0, line_width=1, line_dash="dash", line_color="#94a3b8")
    figure.add_vline(x=0, line_width=1, line_dash="dash", line_color="#94a3b8")
    figure.update_layout(
        height=560,
        margin=dict(l=20, r=20, t=40, b=20),
        showlegend=False,
    )
    return figure


def variable_importance_bar(
    ranked_drivers: pd.DataFrame,
    top_n: int = 15,
    outcome: str | None = None,
) -> go.Figure:
    """Horizontal top-driver chart, colored by analysis method."""
    driver_data = ranked_drivers.copy()
    if outcome is not None:
        driver_data = driver_data[driver_data["outcome"] == outcome]

    driver_data = driver_data.sort_values("combined_score", ascending=False).head(top_n)
    driver_data = driver_data.sort_values("combined_score", ascending=True)

    method_score_data = driver_data.melt(
        id_vars=["process_variable", "combined_score"],
        value_vars=["pca_score", "pls_score", "rf_score"],
        var_name="method",
        value_name="score",
    )
    method_score_data["method"] = method_score_data["method"].map(
        {
            "pca_score": "PCA",
            "pls_score": "PLS",
            "rf_score": "Random Forest",
        }
    )

    figure = px.bar(
        method_score_data,
        x="score",
        y="process_variable",
        color="method",
        orientation="h",
        barmode="stack",
        color_discrete_map=METHOD_COLORS,
        template=PLOT_TEMPLATE,
        labels={
            "score": "Method score",
            "process_variable": "Process variable",
            "method": "Method",
        },
    )
    figure.update_layout(height=max(420, 26 * len(driver_data)), margin=dict(l=20, r=20, t=30, b=20))
    return figure


def outcome_distribution(dataframe: pd.DataFrame, outcome_col: str) -> go.Figure:
    """Histogram with mean and median reference lines."""
    outcome_values = pd.to_numeric(dataframe[outcome_col], errors="coerce").dropna()

    figure = px.histogram(
        x=outcome_values,
        nbins=24,
        template=PLOT_TEMPLATE,
        labels={"x": outcome_col, "y": "Batch count"},
    )

    if not outcome_values.empty:
        mean_value = float(outcome_values.mean())
        median_value = float(outcome_values.median())
        figure.add_vline(
            x=mean_value,
            line_width=2,
            line_dash="solid",
            line_color="#2563eb",
            annotation_text=f"Mean {mean_value:.2f}",
            annotation_position="top right",
        )
        figure.add_vline(
            x=median_value,
            line_width=2,
            line_dash="dash",
            line_color="#16a34a",
            annotation_text=f"Median {median_value:.2f}",
            annotation_position="top left",
        )

    figure.update_layout(height=360, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
    return figure


def qc_trend_plot(
    dataframe: pd.DataFrame,
    outcome_col: str,
    order_column: str | None = None,
    color_column: str | None = None,
    moving_window: int = 7,
) -> go.Figure:
    """Plot a QC outcome over batch order with an optional moving average."""
    plot_data = build_ordered_outcome_frame(dataframe, outcome_col, order_column)
    if color_column and color_column in dataframe.columns:
        plot_data[color_column] = dataframe.loc[plot_data["source_index"], color_column].astype(str).to_numpy()
    else:
        color_column = None

    if plot_data.empty:
        return empty_figure(f"No numeric values available for {outcome_col}.")

    plot_data["moving_average"] = (
        plot_data["outcome_value"].rolling(window=moving_window, min_periods=1).mean()
    )

    if color_column:
        figure = px.scatter(
            plot_data,
            x="order_value",
            y="outcome_value",
            color=color_column,
            template=PLOT_TEMPLATE,
            hover_data=["source_index"],
            labels={
                "order_value": get_order_axis_label(order_column),
                "outcome_value": outcome_col,
            },
        )
    else:
        figure = px.scatter(
            plot_data,
            x="order_value",
            y="outcome_value",
            template=PLOT_TEMPLATE,
            hover_data=["source_index"],
            labels={
                "order_value": get_order_axis_label(order_column),
                "outcome_value": outcome_col,
            },
        )

    figure.update_traces(marker=dict(size=8))
    figure.add_scatter(
        x=plot_data["order_value"],
        y=plot_data["moving_average"],
        mode="lines",
        name=f"{moving_window}-batch moving average",
        line=dict(color="#111827", width=2),
    )
    figure.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title=outcome_col,
        xaxis_title=get_order_axis_label(order_column),
    )
    return figure


def qc_control_chart(
    dataframe: pd.DataFrame,
    outcome_col: str,
    order_column: str | None = None,
) -> go.Figure:
    """Simple Shewhart-style control chart using mean +/- 3 standard deviations."""
    plot_data = build_ordered_outcome_frame(dataframe, outcome_col, order_column)
    if plot_data.empty:
        return empty_figure(f"No numeric values available for {outcome_col}.")

    mean_value = float(plot_data["outcome_value"].mean())
    standard_deviation = float(plot_data["outcome_value"].std(ddof=1))
    upper_control_limit = mean_value + 3.0 * standard_deviation
    lower_control_limit = mean_value - 3.0 * standard_deviation
    plot_data["control_status"] = np.where(
        (plot_data["outcome_value"] > upper_control_limit)
        | (plot_data["outcome_value"] < lower_control_limit),
        "Outside 3 sigma",
        "Inside limits",
    )

    figure = px.scatter(
        plot_data,
        x="order_value",
        y="outcome_value",
        color="control_status",
        color_discrete_map={
            "Inside limits": "#2563eb",
            "Outside 3 sigma": "#dc2626",
        },
        template=PLOT_TEMPLATE,
        hover_data=["source_index"],
        labels={
            "order_value": get_order_axis_label(order_column),
            "outcome_value": outcome_col,
            "control_status": "Status",
        },
    )
    figure.update_traces(marker=dict(size=8))
    figure.add_hline(
        y=mean_value,
        line_width=2,
        line_color="#111827",
        annotation_text=f"Mean {mean_value:.2f}",
        annotation_position="bottom right",
    )
    figure.add_hline(
        y=upper_control_limit,
        line_width=2,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"UCL {upper_control_limit:.2f}",
        annotation_position="top right",
    )
    figure.add_hline(
        y=lower_control_limit,
        line_width=2,
        line_dash="dash",
        line_color="#dc2626",
        annotation_text=f"LCL {lower_control_limit:.2f}",
        annotation_position="bottom right",
    )
    figure.update_layout(
        height=420,
        margin=dict(l=20, r=20, t=30, b=20),
        yaxis_title=outcome_col,
        xaxis_title=get_order_axis_label(order_column),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure


def qc_control_summary(
    dataframe: pd.DataFrame,
    outcome_col: str,
    order_column: str | None = None,
) -> pd.DataFrame:
    """Return rows outside simple mean +/- 3 sigma control limits."""
    plot_data = build_ordered_outcome_frame(dataframe, outcome_col, order_column)
    if plot_data.empty:
        return pd.DataFrame(
            columns=[
                "source_index",
                "order_value",
                "outcome_value",
                "mean",
                "lower_control_limit",
                "upper_control_limit",
                "status",
            ]
        )

    mean_value = float(plot_data["outcome_value"].mean())
    standard_deviation = float(plot_data["outcome_value"].std(ddof=1))
    upper_control_limit = mean_value + 3.0 * standard_deviation
    lower_control_limit = mean_value - 3.0 * standard_deviation
    outside_mask = (
        (plot_data["outcome_value"] > upper_control_limit)
        | (plot_data["outcome_value"] < lower_control_limit)
    )
    flagged_rows = plot_data.loc[outside_mask, ["source_index", "order_value", "outcome_value"]].copy()
    flagged_rows["mean"] = round(mean_value, 4)
    flagged_rows["lower_control_limit"] = round(lower_control_limit, 4)
    flagged_rows["upper_control_limit"] = round(upper_control_limit, 4)
    flagged_rows["status"] = "Outside 3 sigma"
    return flagged_rows.reset_index(drop=True)


def build_ordered_outcome_frame(
    dataframe: pd.DataFrame,
    outcome_col: str,
    order_column: str | None = None,
) -> pd.DataFrame:
    """Build an ordered frame for QC trend and control charts."""
    plot_data = pd.DataFrame(
        {
            "source_index": dataframe.index,
            "outcome_value": pd.to_numeric(dataframe[outcome_col], errors="coerce"),
        }
    )

    if order_column and order_column in dataframe.columns:
        plot_data["order_value"] = dataframe[order_column].to_numpy()
        sort_values = convert_sort_values(dataframe[order_column])
        plot_data["_sort_value"] = sort_values.to_numpy()
    else:
        plot_data["order_value"] = np.arange(1, len(dataframe) + 1)
        plot_data["_sort_value"] = plot_data["order_value"]

    plot_data = plot_data.dropna(subset=["outcome_value", "_sort_value"])
    return plot_data.sort_values("_sort_value").reset_index(drop=True)


def convert_sort_values(series: pd.Series) -> pd.Series:
    """Convert dates or sequence values to sortable numeric values."""
    if pd.api.types.is_datetime64_any_dtype(series):
        return series.astype("int64")

    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().mean() >= 0.80:
        return numeric_values

    parsed_dates = pd.to_datetime(series, errors="coerce")
    sort_values = pd.Series(np.nan, index=series.index, dtype=float)
    sort_values.loc[parsed_dates.notna()] = parsed_dates.loc[parsed_dates.notna()].astype("int64")
    return sort_values


def get_order_axis_label(order_column: str | None) -> str:
    """Return a readable x-axis label."""
    return order_column if order_column else "Batch order"


def empty_figure(message: str) -> go.Figure:
    """Return an empty figure with a centered message."""
    figure = go.Figure()
    figure.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
    )
    figure.update_layout(template=PLOT_TEMPLATE, height=360)
    return figure


def scree_plot(explained_variance: pd.DataFrame) -> go.Figure:
    """Bar and line chart showing variance explained by PCA components."""
    figure = go.Figure()
    figure.add_bar(
        x=explained_variance["component"],
        y=explained_variance["explained_variance_ratio"],
        name="Component variance",
        marker_color="#2563eb",
    )
    figure.add_scatter(
        x=explained_variance["component"],
        y=explained_variance["cumulative_variance_ratio"],
        name="Cumulative variance",
        mode="lines+markers",
        line=dict(color="#16a34a", width=3),
        yaxis="y2",
    )
    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=400,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="PCA component",
        yaxis=dict(title="Explained variance ratio", tickformat=".0%"),
        yaxis2=dict(
            title="Cumulative variance ratio",
            tickformat=".0%",
            overlaying="y",
            side="right",
            range=[0, 1],
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure
