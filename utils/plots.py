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
SPEC_ZONE_COLORS = {
    "below_spec": "#dc2626",
    "near_lower_edge": "#f59e0b",
    "inside": "#0f766e",
    "near_upper_edge": "#f59e0b",
    "above_spec": "#dc2626",
    "missing": "#94a3b8",
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
    driver_data["display_label"] = driver_data["process_variable"].astype(str)
    ordered_labels = driver_data["display_label"].tolist()

    method_score_data = driver_data.melt(
        id_vars=["process_variable", "display_label", "combined_score"],
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
    method_score_data["display_label"] = pd.Categorical(
        method_score_data["display_label"],
        categories=ordered_labels,
        ordered=True,
    )

    figure = px.bar(
        method_score_data,
        x="score",
        y="display_label",
        color="method",
        orientation="h",
        barmode="stack",
        color_discrete_map=METHOD_COLORS,
        category_orders={"display_label": ordered_labels},
        template=PLOT_TEMPLATE,
        labels={
            "score": "Method score",
            "display_label": "Process variable",
            "method": "Method",
        },
        hover_data={
            "process_variable": True,
            "display_label": False,
            "combined_score": ":.3f",
            "score": ":.3f",
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


def spec_distribution_plot(
    dataframe: pd.DataFrame,
    spec_summary_row: pd.Series,
) -> go.Figure:
    """Histogram of a variable with target and spec/window limits overlaid."""
    variable_name = spec_summary_row["variable"]
    if variable_name not in dataframe.columns:
        return empty_figure(f"{variable_name} is not available in the merged data.")

    values = pd.to_numeric(dataframe[variable_name], errors="coerce").dropna()
    if values.empty:
        return empty_figure(f"No numeric values available for {variable_name}.")

    figure = px.histogram(
        x=values,
        nbins=24,
        template=PLOT_TEMPLATE,
        labels={"x": variable_name, "y": "Batch count"},
    )
    figure.update_traces(marker_color="#0f766e", opacity=0.78)

    add_spec_reference_lines(figure, spec_summary_row)
    figure.update_layout(
        height=390,
        margin=dict(l=20, r=20, t=30, b=20),
        showlegend=False,
    )
    return figure


def spec_margin_bar(spec_summary: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Bar chart of variables by percent outside their supplied limits."""
    if spec_summary.empty:
        return empty_figure("No spec summary is available.")

    plot_data = spec_summary.copy()
    plot_data = plot_data.sort_values("percent_outside", ascending=False).head(top_n)
    ordered_variables = plot_data["variable"].astype(str).tolist()
    figure = px.bar(
        plot_data,
        x="percent_outside",
        y="variable",
        color="classification",
        orientation="h",
        template=PLOT_TEMPLATE,
        category_orders={"variable": ordered_variables},
        labels={
            "percent_outside": "Batches outside spec/window (%)",
            "variable": "Variable",
            "classification": "Assessment",
        },
        hover_data=[
            "role",
            "percent_inside",
            "percent_close_to_limit",
            "used_range_ratio",
            "reason",
        ],
    )
    figure.update_layout(
        height=max(380, 28 * len(plot_data)),
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_ticksuffix="%",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure


def outcome_vs_spec_variable_plot(
    dataframe: pd.DataFrame,
    spec_summary_row: pd.Series,
    outcome_col: str,
) -> go.Figure:
    """Scatter plot of an outcome against a spec-controlled process variable."""
    variable_name = spec_summary_row["variable"]
    if variable_name not in dataframe.columns or outcome_col not in dataframe.columns:
        return empty_figure("Selected variable or outcome is not available.")

    plot_data = pd.DataFrame(
        {
            "variable_value": pd.to_numeric(dataframe[variable_name], errors="coerce"),
            "outcome_value": pd.to_numeric(dataframe[outcome_col], errors="coerce"),
            "batch_id": dataframe["batch_id"] if "batch_id" in dataframe.columns else dataframe.index.astype(str),
        }
    ).dropna(subset=["variable_value", "outcome_value"])

    if plot_data.empty:
        return empty_figure(f"No numeric values available for {variable_name} and {outcome_col}.")

    plot_data["spec_zone"] = classify_plot_spec_zones(plot_data["variable_value"], spec_summary_row)

    figure = px.scatter(
        plot_data,
        x="variable_value",
        y="outcome_value",
        color="spec_zone",
        color_discrete_map=SPEC_ZONE_COLORS,
        template=PLOT_TEMPLATE,
        hover_data=["batch_id"],
        labels={
            "variable_value": variable_name,
            "outcome_value": outcome_col,
            "spec_zone": "Spec zone",
        },
    )
    figure.update_traces(marker=dict(size=8))
    add_spec_reference_lines(figure, spec_summary_row)
    figure.update_layout(
        height=440,
        margin=dict(l=20, r=20, t=30, b=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure


def response_shape_plot(
    dataframe: pd.DataFrame,
    variable_col: str,
    outcome_col: str,
    response_band: pd.Series | dict | None = None,
    bin_count: int = 10,
) -> go.Figure:
    """Plot raw outcome response plus binned means for one numeric driver."""
    if variable_col not in dataframe.columns or outcome_col not in dataframe.columns:
        return empty_figure("Selected variable or outcome is not available.")

    plot_data = build_response_shape_frame(dataframe, variable_col, outcome_col)
    if plot_data.empty:
        return empty_figure(f"No numeric values available for {variable_col} and {outcome_col}.")

    binned_summary = build_response_shape_summary(
        dataframe=dataframe,
        variable_col=variable_col,
        outcome_col=outcome_col,
        bin_count=bin_count,
    )

    figure = go.Figure()
    add_response_band_overlays(figure, response_band)
    figure.add_trace(
        go.Scatter(
            x=plot_data["variable_value"],
            y=plot_data["outcome_value"],
            mode="markers",
            name="Batch",
            marker=dict(color="#64748b", size=7, opacity=0.42),
            customdata=np.stack(
                [plot_data["batch_id"], plot_data["source_index"]],
                axis=-1,
            ),
            hovertemplate=(
                "Batch: %{customdata[0]}<br>"
                "Row: %{customdata[1]}<br>"
                f"{variable_col}: %{{x:.3g}}<br>"
                f"{outcome_col}: %{{y:.3g}}<extra></extra>"
            ),
        )
    )

    if not binned_summary.empty:
        figure.add_trace(
            go.Scatter(
                x=binned_summary["variable_midpoint"],
                y=binned_summary["outcome_mean"],
                mode="lines+markers",
                name="Binned mean",
                line=dict(color="#0f766e", width=3),
                marker=dict(color="#0f766e", size=9),
                customdata=np.stack(
                    [
                        binned_summary["count"],
                        binned_summary["variable_min"],
                        binned_summary["variable_max"],
                    ],
                    axis=-1,
                ),
                hovertemplate=(
                    "Batches: %{customdata[0]}<br>"
                    "Range: %{customdata[1]:.3g} to %{customdata[2]:.3g}<br>"
                    "Mean outcome: %{y:.3g}<extra></extra>"
                ),
            )
        )

    figure.update_layout(
        template=PLOT_TEMPLATE,
        height=470,
        margin=dict(l=20, r=20, t=36, b=20),
        xaxis_title=variable_col,
        yaxis_title=outcome_col,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return figure


def build_response_shape_frame(
    dataframe: pd.DataFrame,
    variable_col: str,
    outcome_col: str,
) -> pd.DataFrame:
    """Return numeric x/y values and batch labels for response-shape plots."""
    batch_labels = (
        dataframe["batch_id"].astype(str)
        if "batch_id" in dataframe.columns
        else dataframe.index.astype(str)
    )
    plot_data = pd.DataFrame(
        {
            "source_index": dataframe.index,
            "batch_id": batch_labels,
            "variable_value": pd.to_numeric(dataframe[variable_col], errors="coerce"),
            "outcome_value": pd.to_numeric(dataframe[outcome_col], errors="coerce"),
        }
    ).dropna(subset=["variable_value", "outcome_value"])
    if len(plot_data) < 3 or plot_data["variable_value"].nunique() < 3:
        return pd.DataFrame(
            columns=["source_index", "batch_id", "variable_value", "outcome_value"]
        )
    return plot_data.sort_values("variable_value").reset_index(drop=True)


def build_response_shape_summary(
    dataframe: pd.DataFrame,
    variable_col: str,
    outcome_col: str,
    bin_count: int = 10,
) -> pd.DataFrame:
    """Return quantile-bin response means used by response-shape plots."""
    plot_data = build_response_shape_frame(dataframe, variable_col, outcome_col)
    if plot_data.empty:
        return pd.DataFrame(
            columns=[
                "bin_label",
                "count",
                "variable_min",
                "variable_max",
                "variable_midpoint",
                "outcome_mean",
                "outcome_median",
                "outcome_std",
            ]
        )

    usable_bin_count = min(max(int(bin_count), 3), plot_data["variable_value"].nunique())
    plot_data = plot_data.copy()
    plot_data["response_bin"] = pd.qcut(
        plot_data["variable_value"],
        q=usable_bin_count,
        duplicates="drop",
    )
    summary = (
        plot_data.groupby("response_bin", observed=True)
        .agg(
            count=("outcome_value", "count"),
            variable_min=("variable_value", "min"),
            variable_max=("variable_value", "max"),
            outcome_mean=("outcome_value", "mean"),
            outcome_median=("outcome_value", "median"),
            outcome_std=("outcome_value", "std"),
        )
        .reset_index()
    )
    summary["bin_label"] = summary["response_bin"].astype(str)
    summary["variable_midpoint"] = (summary["variable_min"] + summary["variable_max"]) / 2.0
    return summary[
        [
            "bin_label",
            "count",
            "variable_min",
            "variable_max",
            "variable_midpoint",
            "outcome_mean",
            "outcome_median",
            "outcome_std",
        ]
    ].reset_index(drop=True)


def add_response_band_overlays(
    figure: go.Figure,
    response_band: pd.Series | dict | None,
) -> None:
    """Shade broad and refined historical response bands on a response plot."""
    if response_band is None:
        return

    band_values = response_band.to_dict() if isinstance(response_band, pd.Series) else response_band
    range_min = band_values.get("range_min")
    range_max = band_values.get("range_max")
    if pd.notna(range_min) and pd.notna(range_max):
        figure.add_vrect(
            x0=float(range_min),
            x1=float(range_max),
            fillcolor="#0f766e",
            opacity=0.12,
            line_width=0,
            annotation_text="Broad historical band",
            annotation_position="top left",
        )

    refined_min = band_values.get("refined_range_min")
    refined_max = band_values.get("refined_range_max")
    if pd.notna(refined_min) and pd.notna(refined_max):
        figure.add_vrect(
            x0=float(refined_min),
            x1=float(refined_max),
            fillcolor="#2563eb",
            opacity=0.16,
            line_width=0,
            annotation_text="Best narrow bin",
            annotation_position="top right",
        )


def add_spec_reference_lines(figure: go.Figure, spec_summary_row: pd.Series) -> None:
    """Add target/lower/upper vertical reference lines to a figure."""
    lower_limit = spec_summary_row.get("lower_limit")
    upper_limit = spec_summary_row.get("upper_limit")
    target = spec_summary_row.get("target")

    if pd.notna(lower_limit):
        figure.add_vline(
            x=float(lower_limit),
            line_width=2,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Lower {float(lower_limit):.3g}",
            annotation_position="top left",
        )
    if pd.notna(upper_limit):
        figure.add_vline(
            x=float(upper_limit),
            line_width=2,
            line_dash="dash",
            line_color="#dc2626",
            annotation_text=f"Upper {float(upper_limit):.3g}",
            annotation_position="top right",
        )
    if pd.notna(target):
        figure.add_vline(
            x=float(target),
            line_width=2,
            line_dash="solid",
            line_color="#111827",
            annotation_text=f"Target {float(target):.3g}",
            annotation_position="bottom right",
        )


def classify_plot_spec_zones(values: pd.Series, spec_summary_row: pd.Series) -> pd.Series:
    """Classify plot values by spec zone using summary row limits."""
    lower_limit = spec_summary_row.get("lower_limit")
    upper_limit = spec_summary_row.get("upper_limit")
    zones = pd.Series("inside", index=values.index, dtype="object")
    zones.loc[values.isna()] = "missing"

    if pd.notna(lower_limit):
        zones.loc[values < lower_limit] = "below_spec"
    if pd.notna(upper_limit):
        zones.loc[values > upper_limit] = "above_spec"

    if pd.notna(lower_limit) and pd.notna(upper_limit) and upper_limit > lower_limit:
        edge_width = 0.10 * (upper_limit - lower_limit)
        inside_mask = zones.eq("inside")
        zones.loc[inside_mask & (values <= lower_limit + edge_width)] = "near_lower_edge"
        zones.loc[inside_mask & (values >= upper_limit - edge_width)] = "near_upper_edge"

    return zones


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
        # astype("int64") turns NaT into INT64_MIN, which would plot batches with
        # missing dates first and skew the moving average and control limits.
        return datetime_to_sort_values(series)

    numeric_values = pd.to_numeric(series, errors="coerce")
    if numeric_values.notna().mean() >= 0.80:
        return numeric_values

    return datetime_to_sort_values(pd.to_datetime(series, errors="coerce"))


def datetime_to_sort_values(series: pd.Series) -> pd.Series:
    """Convert a datetime Series to float sort keys, keeping NaT missing."""
    sort_values = pd.Series(np.nan, index=series.index, dtype=float)
    present_mask = series.notna()
    sort_values.loc[present_mask] = series.loc[present_mask].astype("int64")
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
