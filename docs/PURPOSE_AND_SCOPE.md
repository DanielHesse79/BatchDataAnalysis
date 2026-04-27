# Purpose And Scope

## Purpose

Batch Insight Analyzer helps life-science manufacturing teams turn historical
batch records into process insight. The goal is to identify process parameters,
materials, equipment, or batch conditions that are associated with quality
outcomes such as yield, purity, HCP, or aggregate level.

The app is meant for process development scientists, manufacturing engineers,
MSAT teams, and small-company technical leaders who have batch data but do not
have a full statistics or data science function.

## Product Thesis

Many small and mid-sized life-science manufacturers keep useful batch and QC
data in spreadsheets, LIMS exports, batch records, or equipment reports. The
problem is rarely that the math does not exist. The problem is that the workflow
from "two messy files" to "credible process insight" is too manual, too
expert-driven, or too expensive.

Batch Insight Analyzer is intended to provide a guided, local-first path:

1. Upload process and QC tables.
2. Match by batch ID.
3. Profile and audit the merged dataset.
4. Optionally compare historical behavior with process windows and QC specs.
5. Run complementary models.
6. Review ranked drivers, spec margins, and visualizations.
7. Generate a plain-language interpretation.

## Intended User

The target user is technical, but not necessarily a statistician:

- Process development scientist.
- Manufacturing engineer.
- MSAT engineer.
- QC or QA-adjacent technical lead.
- Small-company founder or consultant working with batch data.

The UI should therefore explain what matters, avoid statistical jargon where
possible, and surface caveats clearly before conclusions become recommendations.

## What The App Does

The current version:

- Accepts process and QC data as CSV, XLSX, or XLS.
- Lets the user choose batch ID columns in both files.
- Lets the user choose continuous QC outcome columns.
- Performs an inner join on batch ID.
- Profiles variable types, missingness, near-constant columns, and outcome
  statistics.
- Runs conservative pre-flight audit checks.
- Accepts optional process window and QC spec files.
- Flags out-of-spec batches and conservative spec/window challenge labels.
- Shows QC trends and simple control charts.
- Runs PCA, PLS regression, and Random Forest regression.
- Combines method evidence into a ranked driver table.
- Generates a local or cloud Ollama interpretation.

## What The App Does Not Do

The current version does not:

- Store user accounts or analysis history.
- Use a database.
- Run in the cloud by default.
- Provide a GMP audit trail.
- Validate models for batch release.
- Analyze time-series trajectories or online sensor curves.
- Fit classification outcomes such as pass/fail.
- Prove causality.
- Automatically optimize process setpoints.

## Decision-Support Posture

The correct posture is:

> "This app helps find hypotheses and process patterns worth investigating."

The incorrect posture is:

> "This app proves what causes quality outcomes and tells us what setpoints to
> run."

The distinction matters. Historical batch data can be confounded by time,
campaigns, operators, raw material lots, equipment changes, or QC method drift.
The app should therefore label findings as associations unless supported by
designed experiments or strong domain evidence.

## Regulatory Framing

The methods used in the app are common in process understanding and multivariate
analysis:

- PCA for exploratory batch structure.
- PLS for supervised relationships between process variables and CQAs.
- Random Forest for non-linear and mixed-type tabular relationships.
- Pre-flight audit checks for leakage, drift, outliers, and confounding.

For regulated environments, this should be framed as process understanding,
investigation support, or continuous improvement. A customer who wants to use
the output for batch release, automated decisions, or validated GMP workflows
would need a much stronger validation, lifecycle, audit trail, and change-control
framework.

## Privacy And Deployment Scope

Version 1 is local-first:

- Streamlit runs on the user's machine.
- Uploaded files are held in the Streamlit session.
- No database is used.
- Local Ollama models keep interpretation on the machine.

Cloud-tagged Ollama models, such as `nemotron-3-super:cloud`, may send the
analysis summary to Ollama Cloud. The UI warns the user when such a model is
selected.

## Success Criteria

Using the included synthetic data, the app should recover these known patterns:

- Feed rate and temperature drive yield.
- Supplier_B lowers purity.
- pH has a U-shaped relationship with aggregate percent.
- High temperature plus long duration raises HCP risk.
- BR-3 underperforms on yield.

These are the first practical checks for whether the app is doing useful work.
