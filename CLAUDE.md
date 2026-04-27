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

# Generate the three synthetic datasets used in tests and smoke tests
python generate_synthetic_data.py
python generate_mock_spec_data.py
python generate_messy_field_data.py

# Run the app
.\.venv\Scripts\python.exe -m streamlit run app.py --server.port 8501

# Tests (pytest, smoke-level — see tests/)
.\.venv\Scripts\python.exe -m pytest -q
.\.venv\Scripts\python.exe -m pytest tests/test_evidence.py::test_specific_thing -q

# Compile check before commits
.\.venv\Scripts\python.exe -m compileall -q app.py generate_synthetic_data.py analysis utils
```

Python 3.11+ is expected. Pinned dependencies are in `requirements.txt`.

## Architecture

`app.py` is the Streamlit entry point and is large (~93 KB). It owns UI flow, session state, and rendering, but delegates all real logic to the `analysis/` and `utils/` modules. **Do not put new analysis logic in `app.py`** — add it to the appropriate `analysis/*` module and call it from `app.py`. Analysis functions return dictionaries and DataFrames so Streamlit code stays a thin shell.

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
- **methods.py** — PCA, PLS, Random Forest, preprocessing, validation, combined ranked driver table.
- **specs.py** — spec/window normalization, historical margins, OOS batches, Cp/Cpk, conservative challenge labels.
- **evidence.py** — builds the deterministic report pack consumed by the LLM (and by key_findings). Also computes historical response bands (quartile means + best-observed bin) so middle-band sweet spots aren't flattened to "higher is better".
- **key_findings.py** — Python-generated source-of-truth findings shown **before** the LLM narrative.
- **confidence.py** — explains the practical confidence label per ranked driver (method agreement, validation, n, missingness, audit cautions).
- **method_registry.py** — short explanations of what each method can/cannot prove. The UI uses this to set expectations.
- **interpreter.py** — streams an Ollama interpretation from the evidence pack. Handles cloud-model shortcuts and output cleanup.
- **report_validator.py** — heuristic checks on LLM output: assumed specs, unknown variables, categorical level mix-ups, overly causal/action-directive wording.

### `utils/`

- **plots.py** — Plotly helpers (PCA, loadings, importance, QC trends, control charts, spec plots).
- **report.py** — ReportLab PDF export.

### Key design rules (from docs/DEVELOPMENT.md)

- **Python computes facts; the LLM narrates those facts.** Anything that looks like a number, threshold, or confidence label should be derivable from a deterministic module before the interpreter sees it.
- Conservative claims by default. Never present exploratory quartile thresholds or response bands as optimized setpoints — they are historical hypotheses.
- Spec/window classifications are heuristics, not validation.
- The synthetic dataset has planted relationships (see README "Synthetic Validation Data"). They are the reality check: if changes break recovery of those signals, the workflow is broken.

## Ollama integration

`analysis/interpreter.py` talks to Ollama's chat API. Defaults come from `.env` (see `.env.example`):

```
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=ministral-3:14b
```

Cloud model shortcuts (e.g. `nemotron-3-super:cloud`) route data through Ollama Cloud — only suitable for non-sensitive data.

## Tests

`tests/` currently covers `aggregation`, `evidence`, `normalization`, and `report_validator` — intake/evidence helpers and validation, not full Streamlit runs. Treat the manual smoke tests in `docs/DEVELOPMENT.md` (synthetic, mock-spec, and messy-field flows) as the end-to-end check.

## Synthetic test data

Generated by the three `generate_*.py` scripts and committed under `data/`. The synthetic CSVs are tracked intentionally — they are part of the validation story. Don't delete or regenerate them carelessly.

## Documentation

In-depth docs live in `docs/` and are also surfaced through an in-UI reader at the top of the main page:

- `PURPOSE_AND_SCOPE.md`, `THEORY.md`, `USER_GUIDE.md`, `FIELD_DATA_INTAKE.md`, `SPECS_AND_WINDOWS.md`, `VALIDATION.md`, `DEVELOPMENT.md`.

When adding a new analysis method, also add an entry to `analysis/method_registry.py` and a short note in `docs/THEORY.md`.
