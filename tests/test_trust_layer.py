from analysis.confidence import build_driver_confidence_breakdown
from analysis.evidence import build_operating_window_hints, build_report_pack
from analysis.key_findings import build_deterministic_key_findings


def test_confidence_breakdown_explains_top_synthetic_yield_driver(synthetic_pipeline):
    ranked_drivers = synthetic_pipeline["analysis"]["ranked_drivers"]
    top_yield_driver = ranked_drivers[ranked_drivers["outcome"] == "yield_g_L"].iloc[0]

    breakdown = build_driver_confidence_breakdown(
        ranked_driver_row=top_yield_driver,
        analysis_results=synthetic_pipeline["analysis"],
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
    )

    components = breakdown["component"].tolist()
    assert "Method agreement" in components
    assert "Model validation" in components
    assert "Sample size" in components
    assert "Missingness" in components
    assert "Audit cautions" in components
    assert "PLS" in breakdown.loc[
        breakdown["component"] == "Method agreement",
        "assessment",
    ].iloc[0]


def test_key_findings_include_deterministic_caveats_and_known_patterns(synthetic_pipeline):
    findings = build_deterministic_key_findings(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=synthetic_pipeline["analysis"],
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
    )
    finding_text = "\n".join(findings["finding"].tolist())
    caveat_text = "\n".join(findings["caveat"].tolist())

    assert "feed_rate_day3_mL_h" in finding_text
    assert "Supplier_B" in finding_text
    assert "duration_hours" in finding_text and "temperature_C" in finding_text
    assert "not proof of causation" in caveat_text


def test_report_pack_contains_response_bands_and_guardrails(synthetic_pipeline):
    report_pack = build_report_pack(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=synthetic_pipeline["analysis"],
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
    )

    assert any("not causal proof" in guardrail for guardrail in report_pack["guardrails"])
    aggregate_bands = report_pack["outcomes"]["aggregate_percent"]["historical_response_bands"]
    assert any(band["variable"] == "ph_setpoint" for band in aggregate_bands)


def test_mock_spec_operating_window_hints_find_naoh_middle_band(mock_spec_pipeline):
    hints = build_operating_window_hints(
        ranked_drivers=mock_spec_pipeline["analysis"]["ranked_drivers"],
        merged_dataframe=mock_spec_pipeline["dataframe"],
        profile_result=mock_spec_pipeline["profile"],
        outcomes=["yield_percent"],
    )

    sodium_hydroxide_hint = hints[hints["process_variable"] == "sodium_hydroxide_g"]
    assert not sodium_hydroxide_hint.empty
    row = sodium_hydroxide_hint.iloc[0]
    assert row["pattern_type"] == "middle_sweet_spot"
    assert row["range_min"] < 100 < row["range_max"]
    assert row["refined_range_min"] < 101 < row["refined_range_max"]
