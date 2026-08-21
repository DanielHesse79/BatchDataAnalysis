# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

Batch Insight Analyzer is a local Streamlit app that links bioprocess/manufacturing batch parameters to QC outcomes. It targets life-science teams without an in-house statistician. The app ingests messy process and QC files, joins them by batch ID, runs PCA/PLS/Random Forest, and uses Ollama to narrate a deterministic evidence pack.

The product is decision support, not a validated GMP release tool. Findings are associative, not causal.

## Common Commands

PowerShell on Windows is the primary shell. Use the bundled venv interpreter explicitly to avoid PATH confusion.

```powershell
# First-time setup
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt

# Test tooling (pytest lives here, not in requirements.txt)
pip install -r requirements-dev.txt

# Generate the synthetic datasets used in tests and smoke tests
python generate_synthetic_data.py
python generate_mock_spec_data.py
python generate_messy_field_data.py

# Not used by either app: a synthetic bioanalytical study for the report plan
python generate_bioanalytical_study.py

# Run the app
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501

# Tests (pytest — see tests/)
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_evidence.py::test_specific_thing -q

# Optional explainability + experiment-design extras (CatBoost, XGBoost, DoE/BO)
.\.venv\Scripts\python.exe -m pip install -r requirements-optional.txt

# Optional, HEAVY: time-series forecasting (torch + Chronos-2). See plan doc.
.\.venv\Scripts\python.exe -m pip install -r requirements-forecast.txt

# Compile check before commits
.\.venv\Scripts\python.exe -m compileall -q app.py generate_synthetic_data.py analysis utils ui
```

Python 3.11+ is expected. Pinned dependencies are in `requirements.txt`.

## Architecture

`app.py` is the Streamlit entry point and is deliberately small (~440 lines). It owns page setup, session state, and the stage-by-stage flow, and nothing else. Rendering lives in `ui/`; all real logic lives in `analysis/` and `utils/`. **Do not put analysis logic in `app.py` or in `ui/`** — add it to the appropriate `analysis/*` module and call it. Analysis functions return dictionaries and DataFrames so the view layer stays a thin shell. `tests/test_app_smoke.py` fails the build if `app.py` grows past 600 lines.

The pipeline is staged: intake → normalization → aggregation → readiness/mapping → data_prep → profiling/audit → methods → evidence → key_findings/confidence → interpreter (LLM) → report_validator → utils/report (PDF). Each stage is a separate module so the LLM step gets a small, verified evidence pack rather than raw tables.

### `analysis/` modules and their responsibilities

- **intake.py** — inspects uploaded CSV/Excel before canonical loading; sheet detection, header-row suggestion.
- **normalization.py** — column-name cleanup, decimal commas, units in cells, qualified values (`<`, `>`), batch ID formatting.
- **aggregation.py** — duplicate batch handling and long-format batch/test/value pivoting. Keeps replicate logic out of stat methods.
- **readiness.py** — pre-merge readiness score, blockers, warnings, match stats.
- **mapping.py** — exports/imports JSON mapping profiles for repeat customer formats.
- **data_prep.py** — validates batch ID columns, picks default outcomes, performs the inner join.
- **profiling.py** — variable types, missingness, near-constant columns, outcome stats.
- **audit.py** — pre-flight checks: leakage, drift, confounding, missingness, outliers, multicollinearity.
- **methods.py** — PCA, PLS, Random Forest, preprocessing, validation, combined ranked driver table. Cross-validation fits preprocessing inside each fold; confidence labels are gated on held-out performance, so a driver cannot be called "high" confidence unless some model actually predicts unseen batches.
- **templates.py** — canonical import templates (process/QC wide/QC long/spec sheets) offered as downloads.
- **advanced_methods.py** — optional CatBoost + native SHAP explanations. This is on-demand and should not make the base app depend on CatBoost.
- **specs.py** — spec/window normalization, historical margins, OOS batches, Pp/Ppk capability (overall sigma, including one-sided PpU/PpL), conservative challenge labels.
- **evidence.py** — builds the deterministic report pack consumed by the LLM (and by key_findings). Also computes historical response bands (quartile means + best-observed bin) so middle-band sweet spots aren't flattened to "higher is better".
- **key_findings.py** — Python-generated source-of-truth findings shown **before** the LLM narrative.
- **confidence.py** — explains the practical confidence label per ranked driver (method agreement, validation, n, missingness, audit cautions).
- **method_registry.py** — short explanations of what each method can/cannot prove. The UI uses this to set expectations.
- **interpreter.py** — streams an Ollama interpretation from the evidence pack. Handles cloud-model shortcuts and output cleanup.
- **report_validator.py** — heuristic checks on LLM output: assumed specs, unknown variables, categorical level mix-ups, overly causal/action-directive wording.

### `ui/` modules (view layer only — no analysis logic)

- **state.py** — Streamlit caching, input fingerprints, and per-run memos. Streamlit reruns the whole script on every interaction, so file parsing, the evidence pack, and the Ollama model list are all cached here rather than recomputed.
- **theme.py** — page styling, header, sidebar, in-app documentation reader.
- **intake_panel.py** — templates, file intake, long-format pivot, readiness, mapping profiles, spec input.
- **diagnostics_panel.py** — merge summary, profile, pre-flight audit, QC trends, spec assessment.
- **results_panel.py** — ranked drivers, per-method views, optional CatBoost/XGBoost tabs, outcome-direction override.
- **explanation_panel.py** — PDF export and the local Ollama narrative.

### `utils/`

- **plots.py** — Plotly helpers (PCA, loadings, importance, QC trends, control charts, spec plots).
- **report.py** — ReportLab PDF export.
- **format.py** — shared display formatting (percentages, metrics, rounding, filename timestamps).

### Key design rules (from docs/DEVELOPMENT.md)

- **Python computes facts; the LLM narrates those facts.** Anything that looks like a number, threshold, or confidence label should be derivable from a deterministic module before the interpreter sees it.
- Conservative claims by default. Never present exploratory quartile thresholds or response bands as optimized setpoints — they are historical hypotheses.
- Spec/window classifications are heuristics, not validation.
- The synthetic dataset has planted relationships (see README "Synthetic Validation Data"). They are the reality check: if changes break recovery of those signals, the workflow is broken.

## Ollama integration

`analysis/interpreter.py` talks to Ollama's chat API. Defaults come from `.env` (see `.env.example`):

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=gemma4:e4b
```

Cloud model shortcuts (e.g. `nemotron-3-super:cloud`) route data through Ollama Cloud — only suitable for non-sensitive data.

## Tests

`tests/` covers data prep, normalization, aggregation, profiling, audit, specs/windows, deterministic evidence, confidence breakdowns, report validation, synthetic-truth recovery, statistical hardening (including a pure-noise negative control), PDF generation, interpreter prompt budgeting, and optional CatBoost/XGBoost integration when installed. `tests/test_app_smoke.py` runs `app.py` headlessly with Streamlit's `AppTest` to catch import errors, a missing entry point, and an over-large `app.py`. It still does not drive file uploads through the UI or verify chart visual fidelity — treat the manual smoke tests in `docs/DEVELOPMENT.md` (synthetic, mock-spec, and messy-field flows) as the remaining UI-level check.

## Synthetic test data

Generated by the `generate_*.py` scripts and committed under `data/`. The synthetic CSVs are tracked intentionally — they are part of the validation story. Don't delete or regenerate them carelessly.

`generate_bioanalytical_study.py` is the exception: it feeds neither app and nothing imports it. It produces `data/bioanalytical_study/`, a synthetic study sample analysis dataset with deliberately awkward cases (a rejected run, a dropped calibrator that still passes the 75% rule, ISR failures, samples above ULOQ) and a `ground_truth.json` of the expected answers. It exists to test the unbuilt report engine in `docs/REPORT_AUTOMATION_PLAN.md`.

## Documentation

In-depth docs live in `docs/` and are also surfaced through an in-UI reader at the top of the main page:

- `PURPOSE_AND_SCOPE.md`, `THEORY.md`, `USER_GUIDE.md`, `FIELD_DATA_INTAKE.md`, `SPECS_AND_WINDOWS.md`, `VALIDATION.md`, `DEVELOPMENT.md`.

Planned work (XGBoost+SHAP, Chronos-2 forecasting, DoE/Bayesian-optimization experiment proposals) and its dependency tiers are specced in `docs/PREDICTIVE_AND_OPTIMIZATION_PLAN.md`. Key rule from that plan: optimization/DoE outputs are **candidate experiments, never setpoints**, and all three additions stay optional (gated like `advanced_methods.py`) so the base app needs no extra packages.

A separate, unbuilt proposal for automating bioanalytical study reports (ICH M10 table generation, per-cell traceability, DOCX output) is sketched in `docs/REPORT_AUTOMATION_PLAN.md`. It would be the first work here that targets regulatory reporting rather than decision support, so read its validation section before starting any of it.

When adding a new analysis method, also add an entry to `analysis/method_registry.py` and a short note in `docs/THEORY.md`.
