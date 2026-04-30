from analysis.evidence import build_report_pack


def top_driver_names(analysis_results, outcome: str, top_n: int = 5) -> list[str]:
    ranked_drivers = analysis_results["ranked_drivers"]
    return ranked_drivers[ranked_drivers["outcome"] == outcome].head(top_n)[
        "process_variable"
    ].tolist()


def test_full_synthetic_pipeline_matches_all_batches(synthetic_pipeline):
    assert synthetic_pipeline["merge"].matched_batch_count == 150


def test_synthetic_yield_drivers_recover_feed_temperature_and_br3(synthetic_pipeline):
    analysis_results = synthetic_pipeline["analysis"]
    report_pack = build_report_pack(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=analysis_results,
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
    )

    top_yield_drivers = top_driver_names(analysis_results, "yield_g_L", top_n=5)
    assert "feed_rate_day3_mL_h" in top_yield_drivers[:2]
    assert "temperature_C" in top_yield_drivers[:3]

    level_effects = report_pack["outcomes"]["yield_g_L"]["categorical_level_effects"]
    flattened_levels = [
        level["level_label"]
        for effect in level_effects
        for level in effect["largest_level_effects"]
    ]
    assert "bioreactor_id=BR-3" in flattened_levels


def test_synthetic_purity_driver_recovers_supplier_b(synthetic_pipeline):
    analysis_results = synthetic_pipeline["analysis"]
    report_pack = build_report_pack(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=analysis_results,
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
    )

    top_purity_drivers = top_driver_names(analysis_results, "purity_percent", top_n=3)
    assert "raw_material_lot_supplier" in top_purity_drivers

    level_effects = report_pack["outcomes"]["purity_percent"]["categorical_level_effects"]
    supplier_effect = next(
        effect
        for effect in level_effects
        if effect["variable"] == "raw_material_lot_supplier"
    )
    supplier_b = next(
        level
        for level in supplier_effect["largest_level_effects"]
        if level["level_label"] == "raw_material_lot_supplier=Supplier_B"
    )
    assert supplier_b["delta_from_overall_mean"] < -4.0


def test_synthetic_aggregate_recovers_ph_middle_band(synthetic_pipeline):
    report_pack = build_report_pack(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=synthetic_pipeline["analysis"],
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
    )

    top_aggregate_drivers = top_driver_names(
        synthetic_pipeline["analysis"],
        "aggregate_percent",
        top_n=3,
    )
    assert "ph_setpoint" in top_aggregate_drivers

    ph_pattern = next(
        pattern
        for pattern in report_pack["outcomes"]["aggregate_percent"]["numeric_driver_patterns"]
        if pattern["variable"] == "ph_setpoint"
    )
    response_band = ph_pattern["historical_response_band"]
    assert response_band["pattern_type"] == "middle_sweet_spot"
    assert 6.8 <= response_band["range_min"] <= 7.0
    assert 7.0 <= response_band["range_max"] <= 7.4


def test_synthetic_hcp_recovers_temperature_duration_interaction(synthetic_pipeline):
    report_pack = build_report_pack(
        profile_result=synthetic_pipeline["profile"],
        audit_result=synthetic_pipeline["audit"],
        analysis_results=synthetic_pipeline["analysis"],
        merged_dataframe=synthetic_pipeline["dataframe"],
        outcomes=synthetic_pipeline["outcomes"],
    )

    interactions = report_pack["outcomes"]["hcp_ppm"]["exploratory_interactions"]
    top_interaction = interactions[0]
    assert {
        top_interaction["first_variable"],
        top_interaction["second_variable"],
    } == {"duration_hours", "temperature_C"}
    assert top_interaction["delta_high_high_vs_other"] > 250.0
