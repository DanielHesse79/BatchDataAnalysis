"""Deterministic alert text.

Every alert is rendered from a template and a Finding. No generative model is
involved, so the same inputs always produce the same words, and the rule that
fired is always named in the output.
"""

from __future__ import annotations

import pandas as pd

from qc_intel.detectors import GRADUAL_DRIFT, STEP_CHANGE, VARIANCE_INCREASE, Finding


SEVERITY_LABELS = {
    "critical": "CRITICAL",
    "warning": "WARNING",
    "watch": "WATCH",
}

RULE_LABELS = {
    STEP_CHANGE: "Step change vs frozen baseline",
    GRADUAL_DRIFT: "Sustained monotonic drift (Theil-Sen + Kendall)",
    VARIANCE_INCREASE: "Between-run CV ratio vs frozen baseline",
}

CAVEAT = (
    "Exploratory trending only. Not a run-acceptance decision and not a "
    "validated system output. Events listed are coincident in time, not "
    "established causes."
)


def render_alert(finding: Finding, nearby_events: pd.DataFrame | None = None) -> str:
    """Render one finding as fixed-format alert text."""
    severity = SEVERITY_LABELS.get(finding.severity, finding.severity.upper())
    lines = [
        f"{severity}:",
        f"{finding.method_id} v{finding.method_version} / {finding.instrument_id} / {finding.qc_level}",
        "",
        finding.headline + ".",
        "",
        f"Consumes {finding.acceptance_fraction * 100:.0f}% of the "
        f"{finding.evidence.get('acceptance_half_width_percent', 'configured')}% acceptance window.",
        f"Observed over {finding.n_runs} runs, {finding.window_start} to {finding.window_end}.",
        "",
        f"Rule triggered: {RULE_LABELS.get(finding.rule_id, finding.rule_id)}",
    ]

    for key, value in finding.evidence.items():
        lines.append(f"    {key}: {value}")

    if nearby_events is not None and not nearby_events.empty:
        lines.extend(["", "Laboratory events in the preceding window:"])
        for _, event in nearby_events.iterrows():
            event_date = pd.Timestamp(event["event_timestamp"]).date().isoformat()
            lines.append(f"    {event_date}  {event['event_type']}: {event['description']}")

    lines.extend(["", CAVEAT])
    return "\n".join(lines)


def render_all_alerts(findings: list[Finding]) -> str:
    """Render a full alert digest, most severe first."""
    if not findings:
        return "No findings above the configured thresholds."

    order = {"critical": 0, "warning": 1, "watch": 2}
    ranked = sorted(
        findings,
        key=lambda item: (order.get(item.severity, 9), -item.acceptance_fraction),
    )
    return "\n\n" + ("\n\n" + "-" * 68 + "\n\n").join(render_alert(item) for item in ranked)
