# QC Intelligence Layer

A read-only trending layer over QC metadata that the LIMS and the chromatography
data system already produce. It looks for analytical drift, instrument-specific
effects, lot-related step changes and precision deterioration across runs,
instruments and months — the longitudinal view no single existing system holds,
because that data lives in three places that never talk.

**This is a separate subsystem from Batch Insight Analyzer.** Different customer,
different question. Batch Insight Analyzer asks *which process variables drive a
QC outcome*, cross-sectionally, over a set of batches. This asks *is a stable
analytical method quietly moving*, longitudinally, over a year.

## What it deliberately does not do

- **It does not decide run acceptance.** UNIFI, MassLynx/QuanLynx and the LIMS
  apply the validated criteria (typically 4-6-15 in regulated bioanalysis). The
  verdict is ingested, never recomputed. Recomputing a regulated number and
  disagreeing with the validated system is a compliance problem, not a feature.
- **It does not fit calibration curves.** Slope, intercept, r² and back-calculated
  accuracy come from the CDS under validated processing methods.
- **It is not a validated system**, not a record system, and produces nothing for
  regulatory reporting. That statement is repeated in the UI and in every alert.
- **It uses no machine learning and no generative AI.** Alerts are rendered from
  fixed templates, so identical inputs always produce identical words.

## Privacy by architecture

The layer holds QC, method, instrument, calibration, run, material and event
metadata. No sample-level rows exist in the schema at all, so subject data cannot
leak into it downstream. Incurred Sample Reanalysis is deliberately out of scope
for the same reason.

`analyst_ref` exists as a pseudonymous column but is **not surfaced in the
dashboard**. Employee data is personal data under GDPR, pseudonymisation is not
anonymisation, and linking analyst identity to performance is employee
monitoring — an HR and works-council decision, not a configuration flag.

## Running it

No additional dependencies. Uses `sqlite3` and `tomllib` from the standard
library plus pandas, NumPy, SciPy, Plotly and Streamlit, which the repository
already pins.

```powershell
# Generate synthetic data, ingest, analyse, print an alert digest
.\.venv\Scripts\python.exe -m qc_intel.build_prototype

# Dashboard
.\.venv\Scripts\python.exe -m streamlit run qc_intel/app.py

# Tests
.\.venv\Scripts\python.exe -m pytest qc_intel/tests -q
```

## Layout

```
qc_intel/
    schema.sql          canonical DDL, plain SQL so the Postgres path stays visible
    db.py               connection, checksummed ingest log, queries
    config.py           TOML method configuration, validated on load
    config/*.toml       one file per method
    ingest/
        generic_tabular.py   CSV/TSV/XLSX -> canonical rows, with provenance
    stats.py            descriptives, run-ordered rolling stats, Theil-Sen, BH
    detectors.py        step change, gradual drift, variance increase
    alerts.py           deterministic alert templates
    pipeline.py         RAW -> NORMALIZED -> DERIVED orchestration
    app.py              Streamlit dashboard
    synth/generate.py   synthetic history with planted, machine-readable truth
    tests/              30 tests
```

## Data flow

```
source files (checksummed)
    -> ingest_log + canonical tables      immutable; re-import is refused
    -> descriptives, run series, findings rebuilt from scratch every time
    -> dashboard and alerts
```

Every QC row carries `ingest_id` and `source_row`, so any point on a chart can be
traced to the file and record that produced it. The Provenance tab does exactly
that.

## Statistical choices, and why

**Fixed acceptance limits are primary; empirical limits are a secondary
overlay.** Clinical chemistry derives control limits from historical spread.
Regulated bioanalysis accepts a run against criteria fixed at validation. Showing
±3SD next to ±15% invites someone to read a passing QC as failing, so the two are
kept in separate config blocks and drawn differently on the chart.

**The baseline is frozen and explicit.** A baseline recomputed over history that
already contains the drift inflates the SD until the chart hides the signal it
exists to show. `[trending.baseline]` requires a start, an end and an approver.

**Run order for charts, calendar time for drift rate.** Runs cluster during a
study and stop between studies, so EWMA-style even spacing does not hold. Nothing
mixes the two axes.

**Within-run and between-run precision stay separate.** A pooled CV mixes them,
and they fail differently: within-run points at injection, integration or
autosampler; between-run at calibration, preparation or standards.

**Findings rank by effect size, not p-value.** A 0.4% shift can be highly
significant with enough runs and still be irrelevant against ±15%. Severity is
the share of the acceptance window consumed. Benjamini-Hochberg then controls
false discovery across the grid, because method × instrument × level × detector
is a large family of simultaneous tests.

**Theil-Sen and Kendall rather than least squares.** QC series carry occasional
wild points; a robust slope and a rank-based monotonicity test survive them.

## Synthetic dataset

3 methods × 3 instruments × 3 QC levels × 12 months = 324 runs, 1,944 QC
observations, about 36 runs per method-instrument pair. Ground truth is written
to `data/source/ground_truth.json` so tests assert recovery rather than
hand-computed numbers.

| Planted | Where | Detected |
|---|---|---|
| Stable method (negative control) | ASSAY_A | no bias findings |
| Gradual instrument drift, −11% | ASSAY_B / LCMS-02 from month 4 | critical, and only on LCMS-02 |
| Step change, +6.5% at a lot change | ASSAY_C from month 7 | step change, aligned to the lot event |
| Variance increase, CV 3% → 8.5% | ASSAY_C / LCMS-03 from month 6 | detected, but see below |
| Occasional isolated QC failures | everywhere | marked on the chart, not alerted |

## What the prototype showed

Two results matter more than the code.

**Bias drift and step changes are detected cleanly and specifically.** The
planted instrument drift is flagged critical on LCMS-02 and on no other
instrument, and the step change lines up with the reference-standard lot event.
This is the part that works, and it is the part no existing per-run screen shows.

**Variance deterioration is at the edge of detectability at this cadence.** With
about 36 runs per method-instrument per year, the p-values of the planted
variance problem (0.011, 0.020, 0.042) interleave with those of stable groups
(0.009, 0.018, 0.024). A CV estimated from a dozen runs is simply too noisy to
separate a 2× change from sampling noise. The detector is honest rather than
tuned: it reports what the data can support, and the test suite asserts the
planted problem is found without demanding zero false positives, because at this
N that is not something the statistics can deliver.

The practical consequence: **variance monitoring needs either a longer baseline,
a higher run cadence, or pooling across QC levels** before it earns a place in an
alerting workflow. Bias trending does not — it works now.

## Next, if this passes its gate

The decision gate is not "do the tests pass". It is: run this against **one real
exported method-year** and see whether it surfaces one drift, step or variance
change the lab had not already seen. Synthetic data can only show the code works.

If it does pass, in order: a UNIFI adapter against the real API; ingesting
internal-standard response and injection index, which are the highest-value
signals and are absent from most exports; SQLAlchemy and PostgreSQL if the data
outgrows SQLite; a scheduler. EWMA and CUSUM only after the real cadence is
measured — clinical-chemistry constants assume daily data and will be either
hair-trigger or blind at 36 runs a year.
