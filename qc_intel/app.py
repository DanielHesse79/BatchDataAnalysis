"""QC Intelligence Layer dashboard.

Read-only trending over QC metadata already produced by the LIMS and the
chromatography data system. It does not decide run acceptance and is not a
validated system; that statement is repeated in the UI on purpose.

Run with:
    .\\.venv\\Scripts\\python.exe -m streamlit run qc_intel/app.py
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from qc_intel import db
from qc_intel.alerts import render_alert
from qc_intel.config import load_all_method_configs
from qc_intel.detectors import find_events_near
from qc_intel.pipeline import analyze

SEVERITY_COLOURS = {"critical": "#9E3833", "warning": "#8A6414", "watch": "#41627A"}
NOT_VALIDATED = (
    "Exploratory trending only. Not a validated system, not for run acceptance "
    "and not for regulatory reporting. Events shown alongside a change are "
    "coincident in time, not established causes."
)


st.set_page_config(page_title="QC Intelligence Layer", page_icon="QC", layout="wide")


@st.cache_data(show_spinner="Rebuilding derived statistics...")
def run_analysis(database_path: str, config_dir: str, cache_token: float):
    """Rebuild every derived artefact. Keyed so edits to config invalidate it.

    The connection lives and dies inside this call. Caching a SQLite connection
    across Streamlit runs fails, because each run executes on a different
    script-runner thread and SQLite handles are bound to their creating thread.
    """
    configs = load_all_method_configs(config_dir)
    with db.connection_scope(database_path) as connection:
        result = analyze(connection, configs)
        ingest_log = db.read_sql(connection, "SELECT * FROM ingest_log ORDER BY ingest_id")

    return (
        result.observations,
        result.descriptives,
        result.findings_frame,
        result.events,
        {key: frame for key, frame in result.run_series.items()},
        {key: baseline for key, baseline in result.baselines.items()},
        result.findings,
        configs,
        ingest_log,
    )


def severity_badge(severity: str) -> str:
    colour = SEVERITY_COLOURS.get(severity, "#55636D")
    return (
        f"<span style='background:{colour};color:#fff;padding:2px 8px;"
        f"border-radius:3px;font-size:0.72rem;font-weight:700;"
        f"letter-spacing:0.06em'>{severity.upper()}</span>"
    )


def control_chart(
    series: pd.DataFrame,
    baseline,
    acceptance_half_width: float,
    events: pd.DataFrame,
    title: str,
) -> go.Figure:
    """Run-ordered chart: fixed acceptance limits first, baseline as an overlay."""
    figure = go.Figure()

    figure.add_hrect(
        y0=-acceptance_half_width, y1=acceptance_half_width,
        fillcolor="#2C6650", opacity=0.07, line_width=0,
        annotation_text="acceptance window", annotation_position="top left",
    )
    for sign in (1, -1):
        figure.add_hline(
            y=sign * acceptance_half_width, line_dash="solid",
            line_color="#2C6650", line_width=1.5,
        )

    if baseline is not None and baseline.is_usable:
        figure.add_hline(y=baseline.mean_bias_percent, line_dash="dash", line_color="#1F5673")
        for multiple in (2, 3):
            for sign in (1, -1):
                figure.add_hline(
                    y=baseline.mean_bias_percent + sign * multiple * baseline.sd_bias_percent,
                    line_dash="dot", line_color="#8694A0", line_width=1,
                )

    failed = series[series["any_failure"].fillna(False)]
    passed = series[~series["any_failure"].fillna(False)]

    figure.add_trace(go.Scatter(
        x=passed["acquisition_timestamp"], y=passed["mean_bias_percent"],
        mode="lines+markers", name="Run mean bias",
        line=dict(color="#1F5673", width=1.4), marker=dict(size=6),
    ))
    if not failed.empty:
        figure.add_trace(go.Scatter(
            x=failed["acquisition_timestamp"], y=failed["mean_bias_percent"],
            mode="markers", name="Run with a failed QC",
            marker=dict(size=11, color="#9E3833", symbol="x"),
        ))
    if "rolling_mean_bias" in series:
        figure.add_trace(go.Scatter(
            x=series["acquisition_timestamp"], y=series["rolling_mean_bias"],
            mode="lines", name="Rolling mean (8 runs)",
            line=dict(color="#C2703D", width=2, dash="dot"),
        ))

    # add_vline positions its annotation by arithmetic on the x value, which
    # fails on a datetime axis. An explicit shape plus annotation avoids that
    # entirely and gives the event marker its own styling.
    for _, event in events.iterrows():
        event_time = pd.Timestamp(event["event_timestamp"]).to_pydatetime()
        figure.add_shape(
            type="line", x0=event_time, x1=event_time, y0=0, y1=1, yref="paper",
            line=dict(color="#8694A0", width=1, dash="dot"),
        )
        figure.add_annotation(
            x=event_time, y=1.0, yref="paper", yanchor="bottom",
            text=str(event["event_type"]).replace("_", " "),
            showarrow=False, textangle=-90, font=dict(size=9, color="#55636D"),
        )

    figure.update_layout(
        title=title, height=460, hovermode="x unified",
        yaxis_title="Bias vs nominal (%)", xaxis_title="Acquisition date",
        margin=dict(l=10, r=10, t=52, b=10), legend=dict(orientation="h", y=-0.18),
    )
    return figure


def main() -> None:
    st.title("QC Intelligence Layer")
    st.caption(NOT_VALIDATED)

    package_root = Path(__file__).resolve().parent
    database_path = str(package_root / "data" / "qc_intel.sqlite")
    config_dir = str(package_root / "config")

    if not Path(database_path).exists():
        st.error(
            "No analytics database found. Build it first:\n\n"
            "`python -m qc_intel.synth.generate`\n\n"
            "`python -m qc_intel.build_prototype`"
        )
        return

    cache_token = max(path.stat().st_mtime for path in Path(config_dir).glob("*.toml"))
    (
        observations, descriptives, findings_frame, events,
        run_series, baselines, findings, configs, ingest_log,
    ) = run_analysis(database_path, config_dir, cache_token)

    if observations.empty:
        st.warning("The database contains no QC observations.")
        return

    overview, chart, drift, comparison, provenance = st.tabs(
        ["Overview", "Control chart", "Drift ranking", "Instrument comparison", "Provenance"]
    )

    # ------------------------------------------------------------- overview
    with overview:
        columns = st.columns(5)
        columns[0].metric("Methods", observations["method_id"].nunique())
        columns[1].metric("Instruments", observations["instrument_id"].nunique())
        columns[2].metric("Runs", observations["run_id"].nunique())
        columns[3].metric("QC observations", f"{len(observations):,}")
        failure_rate = observations["pass_fail"].str.lower().eq("fail").mean() * 100
        columns[4].metric("QC failure rate", f"{failure_rate:.1f}%")

        st.subheader("Open findings")
        if findings_frame.empty:
            st.success("No findings above the configured thresholds.")
        else:
            counts = findings_frame["severity"].value_counts()
            badge_columns = st.columns(3)
            for index, severity in enumerate(("critical", "warning", "watch")):
                badge_columns[index].metric(severity.title(), int(counts.get(severity, 0)))

            st.dataframe(
                findings_frame[[
                    "severity", "rule_id", "method_id", "instrument_id", "qc_level",
                    "headline", "acceptance_fraction", "n_runs",
                ]].round(3),
                width="stretch", hide_index=True,
            )

        st.subheader("Method x instrument x level summary")
        st.dataframe(descriptives.round(3), width="stretch", hide_index=True)

    # -------------------------------------------------------- control chart
    with chart:
        selector = st.columns(4)
        method_id = selector[0].selectbox("Method", sorted(observations["method_id"].unique()))
        method_rows = observations[observations["method_id"] == method_id]
        instrument_id = selector[1].selectbox(
            "Instrument", sorted(method_rows["instrument_id"].unique())
        )
        qc_level = selector[2].selectbox("QC level", sorted(method_rows["qc_level"].unique()))
        lookback = selector[3].selectbox("Event lookback (days)", [7, 30, 90], index=1)

        method_version = method_rows["method_version"].iloc[0]
        key = (method_id, method_version, instrument_id, qc_level)

        if key not in run_series:
            st.info("No runs for this combination.")
        else:
            series = run_series[key]
            baseline = baselines.get(key)
            level_config = configs[method_id].level(qc_level)
            half_width = level_config.acceptance_half_width if level_config else 15.0

            window_events = events[
                (
                    ((events["entity_type"] == "instrument") & (events["entity_id"] == instrument_id))
                    | (events["entity_type"] == "material")
                )
                & (events["event_timestamp"] >= series["acquisition_timestamp"].min())
                & (events["event_timestamp"] <= series["acquisition_timestamp"].max())
            ]

            st.plotly_chart(
                control_chart(
                    series, baseline, half_width, window_events,
                    f"{method_id} v{method_version} / {instrument_id} / {qc_level}",
                ),
                width="stretch",
            )
            st.caption(
                "Solid green: fixed acceptance limits from validation. Dashed blue and "
                "dotted grey: mean and 2/3 SD from the frozen baseline window, shown for "
                "trending only. They are different concepts and are never combined."
            )

            if baseline is not None and baseline.is_usable:
                baseline_columns = st.columns(4)
                baseline_columns[0].metric("Baseline runs", baseline.n_runs)
                baseline_columns[1].metric("Baseline mean bias", f"{baseline.mean_bias_percent:+.2f}%")
                baseline_columns[2].metric("Baseline SD", f"{baseline.sd_bias_percent:.2f}%")
                baseline_columns[3].metric("Baseline window", f"{baseline.start} to {baseline.end}")
            else:
                st.warning(
                    "No usable frozen baseline for this combination. Exploratory limits "
                    "are not shown; the fixed acceptance window still applies."
                )

    # --------------------------------------------------------- drift ranking
    with drift:
        st.subheader("Ranked by share of the acceptance window consumed")
        st.caption(
            "Ranked by effect size, not by p-value. With this many method x instrument "
            "x level combinations, a p-value ranking would surface whichever group "
            "happens to have the most runs."
        )
        if findings_frame.empty:
            st.success("No findings above the configured thresholds.")
        else:
            for finding in findings:
                nearby = find_events_near(
                    events, finding.instrument_id, finding.method_id,
                    pd.Timestamp(finding.window_start), 30,
                )
                header = (
                    f"{finding.severity.upper()} - {finding.method_id} / "
                    f"{finding.instrument_id} / {finding.qc_level} - {finding.rule_id}"
                )
                with st.expander(header, expanded=finding.severity == "critical"):
                    st.markdown(severity_badge(finding.severity), unsafe_allow_html=True)
                    st.code(render_alert(finding, nearby), language="text")

    # -------------------------------------------------- instrument comparison
    with comparison:
        st.subheader("Instruments running the same method")
        st.warning(
            "Instruments are not randomly assigned to runs; they are entangled with "
            "time, study and column. Read these as descriptions of what happened, not "
            "as an estimate of an instrument effect.",
            icon=":material/warning:",
        )
        compare_method = st.selectbox(
            "Method", sorted(observations["method_id"].unique()), key="compare_method"
        )
        subset = descriptives[descriptives["method_id"] == compare_method]
        st.dataframe(
            subset[[
                "instrument_id", "qc_level", "n_runs", "mean_bias_percent",
                "within_run_cv_percent", "between_run_cv_percent", "failure_rate_percent",
            ]].round(3),
            width="stretch", hide_index=True,
        )

        pivot_figure = go.Figure()
        for instrument_id, rows in subset.groupby("instrument_id"):
            pivot_figure.add_trace(go.Bar(
                x=rows["qc_level"], y=rows["mean_bias_percent"], name=str(instrument_id),
            ))
        pivot_figure.update_layout(
            barmode="group", height=380, yaxis_title="Mean bias (%)",
            title=f"{compare_method}: mean bias by instrument and QC level",
            margin=dict(l=10, r=10, t=52, b=10),
        )
        st.plotly_chart(pivot_figure, width="stretch")

    # ------------------------------------------------------------ provenance
    with provenance:
        st.subheader("Where did this number come from?")
        st.caption(
            "Every QC observation carries the file and record that produced it, so any "
            "point on a chart can be traced back to its source."
        )
        st.dataframe(ingest_log, width="stretch", hide_index=True)

        selected_run = st.selectbox("Inspect a run", sorted(observations["run_id"].unique()))
        run_rows = observations[observations["run_id"] == selected_run]
        st.dataframe(
            run_rows[[
                "qc_level", "replicate_number", "injection_index", "nominal_value",
                "measured_value", "percent_bias", "pass_fail", "is_area",
                "ingest_id", "source_row",
            ]].round(4),
            width="stretch", hide_index=True,
        )


if __name__ == "__main__":
    main()
