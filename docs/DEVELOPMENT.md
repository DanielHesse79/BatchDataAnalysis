# Development Notes

## Architecture

The app is intentionally small and modular.

```text
app.py
```

Streamlit entry point. Handles UI flow, session state, and result rendering.

```text
analysis/intake.py
```

Inspects uploaded CSV/Excel files before canonical loading. It detects sheets,
suggests header rows, and loads the selected sheet/header combination.

```text
analysis/normalization.py
```

Normalizes field-data values: column names, numeric-looking text, decimal
commas, unit-bearing cells, qualified values, and batch ID formatting.

```text
analysis/aggregation.py
```

Handles duplicate batch IDs and long-format batch/test/value tables. This keeps
replicate and pivot logic out of the statistical methods.

```text
analysis/readiness.py
```

Calculates the pre-merge readiness score, blockers, warnings, and match
statistics.

```text
analysis/mapping.py
```

Builds downloadable JSON mapping profiles for repeated customer export formats.

```text
analysis/data_prep.py
```

Validates batch ID columns, detects default outcome columns, and merges cleaned
canonical process and QC data.

```text
analysis/profiling.py
```

Profiles the merged dataset: variable types, missingness, near-constant
columns, outcome statistics, and user-facing warnings.

```text
analysis/audit.py
```

Runs conservative pre-flight checks for leakage, drift, confounding,
missingness, outliers, and multicollinearity.

```text
analysis/evidence.py
```

Builds a compact deterministic report pack from profiling, audit, specs, PCA,
PLS, and Random Forest outputs. This keeps the LLM focused on verified facts
instead of asking it to rediscover patterns from large tables.

It also computes suggested historical response bands from numeric-driver
quartile means, plus a narrower best observed quantile bin when there is enough
data. These bands help the app preserve middle-band sweet spots, such as a NaOH
target region, instead of flattening them into simple higher/lower relationships.

```text
analysis/key_findings.py
```

Builds Python-generated key findings for the UI. This gives users a concise
source-of-truth table before any LLM narrative is generated.

```text
analysis/confidence.py
```

Explains why a ranked driver received its practical confidence label. It combines
method agreement, validation metrics, sample size, missingness, and audit
cautions into a readable table.

```text
analysis/method_registry.py
```

Contains short explanations of active and planned methods. The UI uses this to
make clear what PCA, PLS, Random Forest, response bands, specs, LLM validation,
CatBoost/SHAP, and OPLS can and cannot prove.

```text
analysis/methods.py
```

Runs PCA, PLS, Random Forest, preprocessing, validation, and combined driver
ranking.

```text
analysis/advanced_methods.py
```

Runs optional advanced models. CatBoost + native SHAP values are available when
the optional `catboost` package is installed. The result is shown as a
comparison/explainability layer and does not currently change the main ranked
driver score.

Install optional dependencies with:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt
```

```text
analysis/specs.py
```

Normalizes spec/window files and calculates historical margins,
out-of-spec batches, spec-zone outcome summaries, Pp/Ppk, and conservative
challenge labels.

```text
analysis/interpreter.py
```

Streams an Ollama interpretation from the deterministic report pack. Also
handles known cloud model shortcuts and output cleanup.

```text
analysis/report_validator.py
```

Runs heuristic checks on generated interpretations. It flags likely assumed
specs, unknown variables, categorical level mix-ups, and overly causal or
action-directive wording.

```text
utils/plots.py
```

Plotly chart helpers for PCA, loadings, importance, outcome distributions,
response-shape plots, QC trends, control charts, and spec/window plots.

```text
utils/report.py
```

ReportLab PDF generation. It builds a local report with data summary, key
findings, top-driver chart images, PCA chart image, specs, and ranked-driver
appendix.

```text
generate_synthetic_data.py
```

Creates deterministic synthetic validation files.

```text
generate_mock_spec_data.py
```

Creates deterministic production/QC/spec files for testing the Specs & Windows
workflow with sodium hydroxide, moisture, residual NaOH, drying, and target-shift
examples.

```text
generate_messy_field_data.py
```

Creates deliberately messy process/QC files for intake testing: title rows,
decimal commas, units in cells, inconsistent batch IDs, duplicate rows, and
long-format QC output.

## Local Environment

Recommended Python version:

```text
Python 3.11+
```

Create and activate a virtual environment:

```powershell
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

Run the app:

```powershell
streamlit run app.py
```

or:

```powershell
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501
```

## Checks

Run compile checks before committing:

```powershell
.\.venv\Scripts\python.exe -m compileall -q app.py generate_synthetic_data.py analysis utils
```

Run the automated tests:

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Current automated coverage includes:

- Data prep: batch ID normalization, duplicate ID rejection, and outcome-name
  collision handling.
- Normalization and aggregation: unit-bearing values, decimal commas, qualified
  values, duplicate batch aggregation, and long-format QC pivoting.
- Profiling and audit: variable typing, missingness, near-constant columns,
  leakage-like names, date/sequence drift, confounding categoricals, outliers,
  and multicollinearity.
- Specs/windows: planted mock-spec classifications, out-of-spec rows,
  unmatched spec rows, and duplicate spec-row rejection.
- Deterministic evidence: response bands, confidence breakdowns, key findings,
  guardrails, and report validation.
- Response-shape plots: quantile-bin summaries, raw points, and response-band
  overlays.
- Synthetic recovery: the planted yield, purity, aggregate, HCP interaction,
  BR-3, and Supplier_B stories.
- Optional CatBoost integration when `catboost` is installed; the test is
  skipped automatically on lean environments.

Manual smoke test:

1. Run `python generate_synthetic_data.py`.
2. Start Streamlit.
3. Upload both synthetic CSVs.
4. Select `batch_id` in both files.
5. Select all four outcomes.
6. Click `Analyze`.
7. Review profiling and audit.
8. Optionally upload `data/synthetic_specs.csv`.
9. Click `Run analysis`.
10. Confirm the synthetic validation checklist.

Spec-focused smoke test:

1. Run `python generate_mock_spec_data.py`.
2. Upload `data/mock_spec_process.csv`.
3. Upload `data/mock_spec_qc.csv`.
4. Upload `data/mock_spec_specs.csv`.
5. Select all QC outcomes.
6. Confirm that the Specs & Margins section shows moisture and residual NaOH
   failures, sodium hydroxide window behavior, and target-shift behavior.

Field-intake smoke test:

1. Run `python generate_messy_field_data.py`.
2. Upload `data/messy_process_export.xlsx`.
3. Confirm sheet `Production Export` and header row `2`.
4. Upload `data/messy_qc_long.csv`.
5. Pivot QC long format using `Batch Identifier`, `Test Name`, and `Result`.
6. Select `Batch ID` and `Batch Identifier` as batch ID columns.
7. Select `yield_percent`, `moisture_percent`, and `color_index`.
8. Choose a duplicate handling rule for process duplicates.
9. Confirm readiness shows 72 candidate matched batches.
10. Click `Analyze`, then `Run analysis`.

## Design Principles

Code style:

- Prefer descriptive variable names.
- Keep modules focused.
- Return dictionaries and DataFrames from analysis functions so Streamlit stays
  simple.
- Add comments where they clarify a non-obvious calculation.
- Avoid clever abstractions until there is repeated complexity.

Product style:

- Functional before fancy.
- Clear warnings before strong claims.
- Dense but readable analysis UI.
- Do not present exploratory thresholds as optimized setpoints.
- Keep the regulatory posture as decision support and process understanding.
- Let Python compute facts; let the LLM narrate those facts.

## Git Hygiene

Ignored by default:

- `.env`
- `.venv/`
- `__pycache__/`
- Python bytecode.
- `.streamlit/secrets.toml`
- `Documents/`
- local Streamlit log files.
- local Ollama restart/debug log files.

Synthetic CSV files are intentionally tracked because they are part of the app's
validation story.

## Current Technical Debt

- Automated tests now cover the core deterministic pipeline and planted
  synthetic stories, but they still do not click through the Streamlit UI or
  validate PDF visual fidelity.
- Interpretation quality still depends on the selected Ollama model.
- Report validation is heuristic and should not be treated as final approval.
- Ranked-driver confidence is practical, not formal statistical validation.
- Interaction screening is simple high-high quartile comparison.
- Suggested response bands and refined bins are historical hypotheses, not
  optimized setpoints.
- Spec/window classifications are conservative heuristics, not formal
  validation or change-control recommendations.
- CatBoost + native SHAP is optional and on-demand; it is not yet folded into
  the main ranked-driver score or PDF export.
- No OPLS implementation yet.

## Suggested Next Engineering Steps

1. Add a stricter structured JSON report mode for models that support reliable JSON.
2. Add deterministic U-shape detector for numeric drivers beyond quartile-band
   hints.
3. Improve interpretation prompt to require synthetic-validation checks when the
   synthetic column names are present.
4. Expand tests to cover PDF export, chart generation, and Streamlit UI behavior.
5. Decide how CatBoost/SHAP should influence ranked-driver confidence, if at all.
6. Add OPLS1 for regulator-familiar root-cause interpretation.
7. Add requirements for reproducible model and report metadata.
8. Add richer PDF report metadata and optional chart selection.
