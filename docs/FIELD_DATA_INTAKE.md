# Field Data Intake

Real production and QC exports rarely look like the synthetic validation data.
The intake layer is designed to make messy input visible before analysis.

## What The Intake Layer Handles

- CSV, XLSX, and XLS files.
- Excel sheet selection.
- Header rows below title or metadata rows.
- Blank or duplicate column names.
- Numeric text such as `100 g`, `12,5 %`, `<20 ppm`, and `7.5 h`.
- Inconsistent batch ID formatting such as `MB-001`, `mb001`, and ` MB001 `.
- Duplicate batch IDs with user-selected resolution.
- Long-format tables such as `batch_id | test_name | result`.
- A readiness score before merge/analyze.
- Downloadable mapping profile JSON for repeated customer formats.

## Intake Workflow

1. Choose whether process and QC data are in separate files or one workbook.
2. Upload process and QC files, or upload one combined workbook.
3. Download templates if the source team needs a recommended format.
4. Confirm sheet and header row for each table.
5. Review numeric parsing logs.
6. Pivot long-format files if needed.
7. Select batch ID columns and QC outcomes.
8. Choose duplicate handling if duplicates exist.
9. Review the readiness score and issue cards.
10. Download a mapping profile if the format will be reused.
11. Click Analyze only when blockers are resolved.

## Recommended Templates

The app provides four downloads:

- Process CSV template: one row per batch with process variables.
- QC wide CSV template: one row per batch with QC outcomes as columns.
- QC long CSV template: one row per batch/test/result.
- Combined Excel workbook template: separate sheets for process data, QC wide,
  QC long, and specs/windows.

The combined workbook template is useful when a customer wants to send one file
with process data and analytical data on separate sheets.

## Same Workbook, Different Sheets

Many customers will export one Excel workbook with multiple sheets, for example:

```text
process_data
qc_results
specs_windows
```

Use `Process and QC data are in the same Excel workbook`, upload the workbook
once, then choose the process sheet in the process intake panel and the QC sheet
in the QC intake panel.

## Duplicate Handling

The conservative default is to stop when duplicate batch IDs exist.

Available rules:

- Stop and let me fix duplicates.
- Keep first row per batch.
- Keep last row per batch.
- Average numeric replicate rows.
- Median numeric replicate rows.

For QC replicate results, median or mean can be reasonable. For process batch
records, averaging duplicate rows should be used only when the duplicate rows
represent repeated measurements, not conflicting batch records.

## Long Format

Some QC exports look like this:

```text
batch_id | test_name        | result
MB001    | moisture_percent | 12.4 %
MB001    | yield_percent    | 81.2
MB002    | moisture_percent | <20 %
```

The app can pivot this to:

```text
batch_id | moisture_percent | yield_percent
MB001    | 12.4             | 81.2
MB002    | 20.0             | ...
```

Values with `<` or `>` qualifiers are converted to numeric boundary values and
flagged in the readiness warnings. Treat those values cautiously.

## Readiness Score

The readiness score is a practical pre-flight check. It is not a statistical
validation score.

The UI shows readiness as:

- A numeric score.
- Counts of blockers and warnings.
- Visual issue cards for the most important problems.
- A detailed issue table.

Blockers prevent analysis:

- Missing batch ID columns.
- Missing selected QC outcomes.
- Missing batch IDs.
- Duplicate batch IDs with no resolution rule.
- No matching batch IDs after normalization.
- Selected outcome has no numeric values.

Warnings allow analysis but reduce confidence:

- Low batch-ID overlap.
- Duplicate rows resolved by aggregation.
- High missingness in selected outcomes.
- Qualified values such as `<20`.
- Blank or renamed columns.

## Mapping Profiles

The app can download a JSON mapping profile containing:

- File names.
- Sheet selections.
- Header rows.
- Numeric parsing choices.
- Long-format pivot choices.
- Batch ID mappings.
- Selected QC outcomes.
- Duplicate handling rules.

This is meant for repeat customer exports. The current version exports profiles
but does not yet re-import them automatically.

## Test Files

Run:

```powershell
python generate_messy_field_data.py
```

This creates:

```text
data/messy_process_export.xlsx
data/messy_qc_long.csv
```

These files deliberately include messy field-data patterns so the intake layer
can be tested without using customer data.

## Current Limitations

- PDF batch records are not parsed.
- Free-text deviation extraction is not implemented.
- Units are detected only from text patterns; no formal unit conversion exists.
- Mapping profiles are export-only for now.
- Censored values such as `<20` are flagged, but advanced censored-data
  statistics are not implemented.
- Time-series process trajectories are still out of scope for v1.
