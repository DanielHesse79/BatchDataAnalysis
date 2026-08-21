# User Guide

This guide walks through the Batch Insight Analyzer workflow. The suite also
contains the separate QC Intelligence Layer for longitudinal analytical-method
monitoring.

## 1. Start The App

From the project root, open the shared workspace chooser:

```powershell
.\.venv\Scripts\python.exe launcher.py
```

Choose **Batch drivers** for the workflow described below. **QC monitoring**
opens the separate longitudinal dashboard. The top navigation remains available
so you can switch workspace without restarting the suite.

For a direct developer launch of this workspace:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

## 2. Upload Files

Upload two files:

- Batch process data.
- QC results.

Or enable `Process and QC data are in the same Excel workbook` if both tables
are separate sheets in one workbook.

Supported formats:

- CSV.
- XLSX.
- XLS.

Each file should contain a batch ID column. The names do not have to match, but
the values should refer to the same batches.

Use `Download data templates` if you need examples for a production or QC team.
The combined workbook template includes separate sheets for process data, QC
wide format, QC long format, and specs/windows.

## 3. Confirm Intake Settings

For each uploaded file, the app shows an intake panel.

Use it to confirm:

- Excel sheet.
- Header row number.
- Whether numeric-looking text should be parsed.
- Whether a long-format table should be pivoted to wide format.

If one Excel workbook contains both process and analytical data, the process
intake panel and QC intake panel can use different sheets from the same file.

The raw preview shows the file before parsing. The prepared preview shows the
canonical table that will be used for matching and analysis.

Numeric parsing can convert values such as:

```text
100 g
12,5 %
<20 ppm
7.5 h
```

Values with `<` or `>` qualifiers are converted to boundary values and flagged
in readiness warnings.

## 4. Select Batch ID Columns

After both files load, choose:

- Batch ID column in the process data.
- Batch ID column in the QC results.

The app performs an inner join, meaning it keeps only batches that appear in
both files.

The merge summary reports:

- Number of matched batches.
- Number of process IDs.
- Number of QC IDs.
- Examples of unmatched IDs, if any.

## 5. Select Quality Outcomes

Choose one or more QC outcome columns from the QC file.

For version 1, choose continuous numeric outcomes such as:

- Yield.
- Purity.
- HCP.
- Aggregate percent.

Do not choose pass/fail columns yet. Classification outcomes are out of scope
for the current version.

## 6. Resolve Duplicates And Review Readiness

If duplicate batch IDs are detected, choose a duplicate handling rule:

- Stop and let me fix duplicates.
- Keep first row per batch.
- Keep last row per batch.
- Average numeric replicate rows.
- Median numeric replicate rows.

The readiness panel then shows:

- Readiness score.
- Blockers.
- Warnings.
- Candidate matched batch count.

Blockers prevent analysis. Warnings allow analysis, but should be reviewed
before trusting results.

You can also download a mapping profile JSON after choosing mappings and
duplicate rules. This helps reuse the same customer export format later.

## 7. Optionally Upload Specs And Windows

Upload a spec/window file if you want the app to compare historical values with
official process windows or QC specs.

Expected columns:

```text
variable
role
target
lower_limit
upper_limit
unit
criticality
notes
```

You can download a template directly from the app.

Examples:

```text
sodium_hydroxide_g,process,100,85,115,g,CPP
moisture_percent,qc,,,20,%,CQA
```

You can continue without specs. The rest of the analysis still works.

## 8. Review Data Profile

The data profile shows:

- Matched batch count.
- Number of process variables.
- Number of selected outcomes.
- Variable types.
- Missing data.
- Near-constant variables.
- Outcome statistics.
- Outcome histograms.

Use this section to decide whether the dataset is worth modeling.

## 9. Review Pre-Flight Audit

The pre-flight audit highlights risks:

- Leakage suspects.
- Date, sequence, or campaign-like columns.
- Confounding categorical variables.
- High-cardinality categorical variables.
- Row-level missingness.
- Outlier flags.
- Highly correlated numeric variable pairs.

These checks are warnings, not automatic blockers. They help the user interpret
results with the right level of caution.

## 10. Review Specs And Margins

If a spec file was uploaded, the app shows:

- Matched and unmatched spec variables.
- Percent inside and outside the supplied limits.
- Percent close to a limit.
- Used historical range versus allowed range.
- Out-of-spec batch details.
- Conservative challenge labels.

These labels are not automatic recommendations. They are review prompts.

## 11. Review QC Trend Overview

The QC trend overview shows:

- Outcome trend over batch order.
- Simple mean and control-limit style charts.
- Points outside mean plus or minus 3 standard deviations.

If a sequence column is detected, charts are ordered by that column. Otherwise
they use upload row order.

## 12. Run Analysis

Click `Run analysis`.

The app runs:

- PCA.
- PLS regression for each selected outcome.
- Random Forest regression for each selected outcome.
- Combined ranked driver scoring.

If an order column is detected, the app uses it for time-ordered validation and
excludes that column from driver modeling.

## 13. Read Results

The results page has five tabs.

### Overview

Use this tab for the fastest read:

- Python-verified key findings.
- Top drivers for selected outcome.
- Outcome distribution.
- PCA map colored by selected outcome.
- Validation-order note.

### Top Drivers

Use this tab to inspect the ranked driver table and any suggested historical
response bands for numeric drivers.

Columns include:

- Outcome.
- Rank.
- Process variable.
- Combined score.
- Confidence.
- Evidence methods.

The confidence breakdown table explains why a driver received its practical
confidence label. It combines method agreement, validation metrics, sample size,
missingness, and audit cautions. It is not a formal validation or causal proof.

The response-shape plot shows the selected outcome against a top numeric driver.
Grey points are individual batches. The green line shows binned historical means.
When a broad response band or best narrow bin is available, the plot highlights
those regions so sweet spots are visible rather than buried in the table.

If the app sees a middle-band pattern, it shows a suggested historical response
band. For example, if NaOH quartiles show that yield is highest around 95-107 g
but lower below and above that region, the app should describe that as a
sweet-spot hypothesis rather than saying simply "higher NaOH is better".

The table may include both:

- A broad quartile band, which is more stable but can be wide.
- A best narrow bin, which is more exact but more sensitive to random noise.

These bands are exploratory. They are useful targets for investigation or
confirmation runs, not validated setpoints or automatic spec changes.

### Individual Methods

Use this tab when you want to understand which model produced which signal.

PCA shows:

- Scree plot.
- Loading plot.
- Top process loadings.

PLS shows:

- Rows used.
- Component count.
- Cross-validated Q2.
- Time-ordered validation.
- Variable importance.
- Feature-level coefficients.

Random Forest shows:

- Rows used.
- Training R2.
- OOB R2.
- Time-ordered validation.
- Variable importance.
- Feature-level importance.

### Specs & Margins

Use this tab to inspect spec/window behavior after model evidence is available.

It includes:

- Spec challenge overview.
- Full assessment table.
- Variable distribution with target and limit lines.
- Outcome response near process-window limits.
- Outcome means by spec zone.
- Out-of-spec batch table.

### Explanation

Use this tab to generate a plain-language interpretation through Ollama.

Local models keep the interpretation on the machine. Cloud-tagged models may
send the analysis summary to Ollama Cloud, so they are hidden until you tick
`Allow Ollama Cloud models`. A cloud model is never selected for you.

The tab also shows deterministic key findings. If those rows disagree with the
LLM narrative, trust the deterministic table and regenerate or manually review
the interpretation.

### PDF Report

Use `Prepare PDF report` after analysis to create a local PDF. The report
includes:

- Data summary and audit warnings.
- Outcome statistics.
- Current Ollama interpretation, if one has been generated.
- Top-driver charts and tables.
- PCA overview.
- Specs and operating-window assessment, if specs were uploaded.
- Ranked-driver appendix.

If you generate a new interpretation or rerun the analysis, prepare the PDF
again so the report uses the latest results.

## 14. Reading Documentation In The App

The top of the app includes a `Documentation and theory` section. Use it to read
the user guide, theory notes, specs notes, validation notes, development notes,
and README without leaving the Streamlit UI.

## 15. Choosing An Ollama Model

Measured on the two synthetic datasets, three runs each, scored with
`analysis/report_validator.py`. Lower is better; a warning means the report
misrepresents the evidence pack, not that it reads badly.

| Model | Time | Warnings per report | Clean reports |
|---|---|---|---|
| `gemma4:e4b` | 21 s | 0.17 | 5 of 6 |
| `ministral-3:14b` | 35 s | 0.33 | 4 of 6 |
| `gemma4:12b` | 26 s | 0.50 | 4 of 6 |
| `qwen3.5:9b` | 37 s | 2.00 | 0 of 6 |
| `gpt-oss:20b` | 35 s | 2.00 | 0 of 6 |

- `gemma4:e4b`: recommended. Fastest, and an effective-4B runs on a laptop
  without a GPU.
- `gemma4:12b`, `ministral-3:14b`: equivalent alternatives. The accuracy gap
  between the top three is one warning across six runs, which is noise; the
  speed difference is not.
- `qwen3.5:9b`, `gpt-oss:20b`: not recommended. Both write long reports that
  omit the required section headings, which the PDF export depends on.
- `nemotron-3-super:cloud`: cloud model, not local. Requires ticking
  `Allow Ollama Cloud models` first.

**Leave thinking off.** It was measured on and off for every model that supports
it, and it never helped: `gemma4:12b` got slower and slightly worse,
`gpt-oss:20b` was unchanged, and `qwen3.5:9b` returned an empty report in six
runs out of six because the whole generation budget went to hidden reasoning.

If a model repeats itself, leaks thinking text, or invents file paths, regenerate
with a different model. The app includes stop tokens and output cleanup, but
local models can still behave unevenly.

## 16. How The Report Is Checked And Repaired

Every generated narrative is checked by `analysis/report_validator.py` before you
see it. The checks are deterministic Python, not a second model: the same report
always produces the same verdict, and each check is unit-tested.

What is checked:

| Check | Catches |
|---|---|
| Required headings | A section the model skipped |
| Empty sections | A heading with nothing under it |
| Outcome coverage | An analysed outcome the report never discusses |
| Numbers in the pack | A figure that is not in the evidence pack |
| Unknown variables | A column name the analysis never produced |
| Categorical levels | A level attached to the wrong variable |
| Assumed specs | Invented "typical" limits when no spec file was supplied |
| Causal wording | Claims of causation, or instructions to change a setpoint |

**Numbers in the pack** is the strictest of these and the reason for the rest.
Python computes the facts and the model narrates them, so a figure the analysis
never produced is a fabrication however plausible it reads.

### The repair loop

When a check fails and **Repair the report if the checks fail** is ticked, the
app does not simply hand you the warnings. It tells the model what failed and
asks again, escalating only as far as it has to:

| Rung | What changes |
|---|---|
| `base` | The normal prompt. Most reports stop here. |
| `repair` | The failed checks fed back with the previous report |
| `skeleton` | The required headings supplied pre-filled, plus any outcome that was left out |
| `sections` | One section requested at a time, with the headings written by the app |
| `sections` on another model | The same, on a different local model |

A missing heading skips `repair` and goes straight to `skeleton`: feeding the
text back rarely restores structure, while handing the structure over does.
Retries run at temperature 0, so a repair is a correction rather than another
roll of the dice.

The line under the report says what happened, for example:

> Passed every check after base then repair then skeleton (126s).

**A failed report is never presented as clean.** If the ladder is exhausted, the
attempt with the fewest warnings is shown, its warnings are still listed, and
the line says so:

> Tried base then skeleton then sections (168s) and 1 check(s) still fail. The
> report below is the best attempt, not a clean one.

Turn the repair off if you would rather see the first attempt and its warnings
without waiting. A clean first attempt costs nothing either way; only a failed
one triggers further generation.

### What it is worth

Measured on both synthetic datasets, two runs per model, all on the same code:

| Model | Clean without the loop | With it | Rungs needed | Time |
|---|---|---|---|---|
| `gemma4:e4b` | 5 of 6 | **4 of 4** | 1.5 | 35 s |
| `gemma4:12b` | 4 of 6 | **4 of 4** | 1.8 | 46 s |
| `ministral-3:14b` | 4 of 6 | **4 of 4** | 3.5 | 121 s |
| `qwen3.5:9b` | 0 of 6 | **4 of 4** | 3.5 | 138 s |
| `gpt-oss:20b` | 0 of 6 | **4 of 4** | 4.0 | 144 s |

Every model reaches a clean report with the loop. What separates them is how
much help they need, which is why `gemma4:e4b` is the recommendation: not
because the others cannot get there, but because it is usually already right.

**The loop repairs a blind spot rather than removing it.** `gemma4:e4b` fails
outcome coverage on the mock-spec dataset at the first attempt every time - it
never mentions `moisture_percent` - and the `repair` rung fixes it in one extra
call. With the repair turned off, that report would go out silently missing one
of the four outcomes you selected. A missing heading is visible; a missing
outcome is not, which is why the check exists.

## 17. Interpreting Results Safely

Use this checklist:

- Did multiple methods agree?
- Is the finding supported by validation metrics?
- Is the variable confounded with date, lot, operator, or equipment?
- Is the finding linear, non-linear, categorical, or exploratory?
- Does it match process knowledge?
- Would a designed experiment or confirmation run be needed before action?
- If specs were uploaded, does the data actually cover the full allowed range?

## 18. Synthetic Data Walkthrough

Use the included files:

```text
data/synthetic_process.csv
data/synthetic_qc.csv
```

Select:

```text
batch_id
```

Select all QC outcomes:

```text
yield_g_L
purity_percent
hcp_ppm
aggregate_percent
```

Optionally upload:

```text
data/synthetic_specs.csv
```

Expected high-level findings:

- Yield: feed rate and temperature should be important.
- Purity: raw material supplier should be important, especially Supplier_B.
- HCP: high duration plus high temperature should be risky.
- Aggregate: pH should show a U-shaped pattern.
- Bioreactor: BR-3 should show lower yield.

## 19. Mock Spec Data Walkthrough

For testing the Specs & Margins feature more directly, use:

```text
data/mock_spec_process.csv
data/mock_spec_qc.csv
data/mock_spec_specs.csv
```

Select `batch_id` in both process and QC files.

Select all QC outcomes:

```text
moisture_percent
purity_percent
yield_percent
residual_naoh_ppm
color_index
```

Upload `data/mock_spec_specs.csv` as the optional spec/window file.

Expected high-level spec stories:

- `sodium_hydroxide_g` has a 100 g target and 85-115 g window. Quality worsens
  near low/high NaOH edges, so the window should be challenged.
- `moisture_percent` has an upper QC spec of 20%. Several batches are near or
  above this limit.
- `drying_time_hours` affects moisture, especially for short drying.
- `drying_temperature_C` has a deliberately tight process window but weak
  planted QC impact, making it a candidate for "possibly too narrow" after model
  evidence is available.
- `pH_after_neutralization` runs historically above the stated target, so the
  target may deserve review.
- `reactor_id == RX-3` has lower yield and should be interpreted as an equipment
  pattern, not a numeric spec-window issue.

## 20. Messy Field Data Walkthrough

For testing intake robustness, run:

```powershell
python generate_messy_field_data.py
```

Then upload:

```text
data/messy_process_export.xlsx
data/messy_qc_long.csv
```

Expected intake behavior:

- The process Excel file should suggest header row `2` because two title rows
  appear above the real header.
- `NaOH Added` and `Drying Time` should be parsed from text with units.
- `Batch ID` should remain a text ID, not become numeric.
- The QC CSV should be detected as long format.
- Pivot QC using:

```text
Batch Identifier -> batch ID
Test Name        -> variable/test name
Result           -> value
```

- The readiness panel should report 72 candidate matched batches after batch ID
  normalization.
- The process duplicate rows should require a duplicate handling choice before
  analysis.
