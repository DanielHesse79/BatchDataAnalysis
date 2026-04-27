# Synthetic Validation Checklist

The synthetic dataset is a test harness for the app. It contains known planted
relationships so the workflow can be checked end to end.

## Files

```text
data/synthetic_process.csv
data/synthetic_qc.csv
```

Regenerate them with:

```powershell
python generate_synthetic_data.py
```

## Dataset Size

Process file:

```text
150 batches x 26 columns
```

QC file:

```text
150 batches x 5 columns
```

Spec file:

```text
data/synthetic_specs.csv
```

The shared key is:

```text
batch_id
```

## Planted Relationships

### Yield

Known relationship:

- `yield_g_L` is strongly positive with `feed_rate_day3_mL_h`.
- `yield_g_L` is also positive with `temperature_C`.
- `bioreactor_id == "BR-3"` has lower yield independent of other variables.

Expected app behavior:

- `feed_rate_day3_mL_h` should rank near the top for yield.
- `temperature_C` should rank near the top for yield.
- `bioreactor_id` should appear as a categorical yield signal.
- The interpretation should specifically mention BR-3 underperformance when
  categorical effects are summarized well.

### Purity

Known relationship:

- `purity_percent` drops by about 8 percentage points when
  `raw_material_lot_supplier == "Supplier_B"`.

Expected app behavior:

- `raw_material_lot_supplier` should be a top purity driver.
- The interpretation should identify Supplier_B as the lower-purity level, not
  just mention the supplier column.

### Aggregate

Known relationship:

- `aggregate_percent` is U-shaped with `ph_setpoint`.
- The minimum is around pH 7.0.
- Aggregates rise below about 6.8 and above about 7.4.

Expected app behavior:

- Random Forest should be more helpful than PLS for this relationship because
  the effect is non-linear.
- The interpretation should describe the U-shape and avoid saying simply
  "higher pH is better" or "lower pH is better."

### HCP

Known relationship:

- `hcp_ppm` spikes when `temperature_C` is high and `duration_hours` is long.

Expected app behavior:

- `duration_hours` and `temperature_C` should be important for HCP.
- The interaction screen should highlight high duration plus high temperature.
- The interpretation should frame this as an interaction risk, not a simple
  one-variable rule.

### Outliers And Missingness

Known behavior:

- Several failed-run outliers are included.
- About 3 percent missing data is added to non-critical process columns.

Expected app behavior:

- Outliers should be flagged but not removed.
- Missingness should be reported.
- The analysis should still run.

## Manual Acceptance Checklist

After uploading the synthetic files and running analysis, check:

```text
[ ] Matched batch count is 150.
[ ] Yield top drivers include feed_rate_day3_mL_h and temperature_C.
[ ] Yield interpretation mentions BR-3 lower yield.
[ ] Purity top drivers include raw_material_lot_supplier.
[ ] Purity interpretation mentions Supplier_B lower purity.
[ ] Aggregate interpretation describes pH as U-shaped near pH 7.0.
[ ] HCP interpretation warns about high temperature plus long duration.
[ ] Specs & margins tab appears when data/synthetic_specs.csv is uploaded.
[ ] QC specs with synthetic failures are flagged as high failure risk.
[ ] Process windows show percent outside and used-range ratios.
[ ] Spec/window interpretation avoids calling the historical window a design
    space.
[ ] Pre-flight audit shows useful cautions without blocking analysis.
[ ] LLM interpretation avoids causal overclaiming.
[ ] LLM interpretation avoids "optimized setpoint" language unless clearly
    marked exploratory.
```

## Known Current Weaknesses

The modeling layer can recover the major synthetic relationships, but the LLM
interpretation may still need stronger templating to reliably mention:

- BR-3 underperformance.
- Supplier_B specifically.
- The pH U-shape for aggregate.
- The difference between exploratory quartile thresholds and true optimized
  process setpoints.

The recommended next improvement is a deterministic findings layer that computes
these categorical and non-linear summaries before the LLM writes the report.

## Mock Spec Dataset

The mock spec dataset is focused on testing the Specs & Margins workflow:

```text
data/mock_spec_process.csv
data/mock_spec_qc.csv
data/mock_spec_specs.csv
```

Regenerate it with:

```powershell
python generate_mock_spec_data.py
```

Planted stories:

```text
[ ] sodium_hydroxide_g has a 100 g target and 85-115 g process window.
[ ] Quality worsens near low/high sodium_hydroxide_g edges.
[ ] The Top Drivers tab shows a suggested historical response band for
    sodium_hydroxide_g near the middle quartiles, not a simple "higher is
    better" statement.
[ ] The same table shows a narrower best observed NaOH bin inside the broader
    middle band, with wording that treats it as exploratory.
[ ] moisture_percent has an upper spec of 20%.
[ ] Several batches are near or above the moisture limit.
[ ] drying_time_hours affects moisture.
[ ] drying_temperature_C has a deliberately tight process window and weak
    planted QC impact.
[ ] pH_after_neutralization runs above its stated target.
[ ] RX-3 has lower yield as an equipment pattern.
```

This dataset is better than the main synthetic bioprocess dataset for testing
whether the app can discuss process windows and QC specs in practical terms.
