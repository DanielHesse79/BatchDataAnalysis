# Theory And Method Notes

This document explains the core statistical ideas behind Batch Insight Analyzer.
It is written for process scientists and engineers rather than statisticians.

## Canonical Data Shape

The app expects one row per batch.

Recommended table shape:

```text
batch_id | process variable 1 | process variable 2 | ... | process variable N
batch_id | quality outcome 1  | quality outcome 2  | ... | quality outcome M
```

The process table and QC table are joined by batch ID. This wide, one-row-per-
batch structure works well for PCA, PLS, Random Forest, and most spreadsheet-
based process investigation workflows.

Time-series data, such as pH curves or temperature trajectories, should be a
separate future workflow. It usually requires alignment, interpolation, or
dynamic time warping before multiway PCA or trajectory modeling.

## Process Variables And Outcomes

The app separates columns into:

- Process variables: candidate drivers or explanatory variables.
- Quality outcomes: selected QC response variables.

Quality outcomes should be continuous for version 1. Examples:

- `yield_g_L`
- `purity_percent`
- `hcp_ppm`
- `aggregate_percent`

Binary pass/fail classification is intentionally out of scope for now.

## Data Profiling

Before modeling, the app checks:

- Number of matched batches.
- Number of process variables.
- Number of selected outcomes.
- Continuous, categorical, and binary process-variable counts.
- Missingness per column.
- Near-constant process variables.
- Basic outcome statistics.

This matters because modeling can look impressive even when the data are too
small, too sparse, or too low-quality to support strong conclusions.

## Pre-Flight Audit

The audit layer is deliberately conservative. It does not block analysis; it
warns the user about common ways historical batch data can mislead.

### Leakage

Data leakage happens when a feature contains information that would not have
been available at the time of prediction or investigation. Examples include:

- QC result fields included as process variables.
- Release, pass/fail, complaint, or investigation fields.
- Post-hoc deviation flags.

Leakage can make a model look accurate for the wrong reason.

### Date And Sequence Drift

Batch sequence, campaign, or date columns can absorb hidden process changes.
They may be useful for trend analysis, but they can also hide the true process
driver.

The app uses detected order columns for time-ordered validation when possible
and excludes the selected order column from driver modeling.

### Confounding

Variables are confounded when they move together. For example:

- Supplier_B may only appear during one campaign.
- BR-3 may be used by one shift more often than others.
- A media lot may coincide with a seasonal process change.

Confounding does not make the result useless, but it means the user should be
careful before treating a driver as causal.

### Outliers

Outliers are flagged, not removed automatically. In batch manufacturing, a
"bad" or unusual batch is often exactly the case the user wants to learn from.

## Specs And Operating Windows

Specs and operating windows add manufacturing context to the analysis.

There are two main types:

- Process windows: intended or allowed ranges for process variables.
- QC specs: quality limits that outcomes must satisfy.

Examples:

```text
sodium_hydroxide_g target 100, lower 85, upper 115
moisture_percent upper 20
```

The app compares each supplied spec/window with historical data:

- How often batches are inside or outside the limits.
- How close batches are to the nearest limit.
- Whether historical production used the full allowed range.
- Whether the variable is also a model-ranked quality driver.
- Whether outcomes look different near the window edges.

### Capability: Pp And Ppk, Not Cp And Cpk

The app reports `pp` and `ppk`, not `cp` and `cpk`. The distinction is the
standard deviation used:

- Cp and Cpk use within-subgroup sigma, estimated from rational subgroups
  collected under short-term, common-cause-only conditions.
- Pp and Ppk use the overall (long-term) sample sigma of all the supplied
  values.

Batch records uploaded here almost never carry rational subgroups, so the only
honest spread estimate is the overall sample standard deviation, and the
matching index names are Pp and Ppk. Reporting those numbers as Cp/Cpk would
overstate what the data supports: overall sigma includes between-batch drift and
special-cause variation, so Pp/Ppk are usually lower than the Cp/Cpk a
subgrouped study would produce.

One-sided specs get an index too, because the app's own default spec template is
mostly one-sided (purity lower-only, HCP and aggregate upper-only):

```text
two-sided:   pp  = (USL - LSL) / (6 * sigma)
             ppk = min((USL - mean), (mean - LSL)) / (3 * sigma)
upper-only:  ppk = (USL - mean) / (3 * sigma)   (PpU; pp is not defined)
lower-only:  ppk = (mean - LSL) / (3 * sigma)   (PpL; pp is not defined)
```

A one-sided spec has no two-sided tolerance width, so `pp` stays empty and only
`ppk` is reported. The `capability_basis` column names which of these three
cases produced the number. Capability is withheld entirely when fewer than ten
usable values are available or when sigma is zero, since neither case supports a
meaningful index.

Like every other number in this section, these indices assume stable, roughly
normal behavior and describe history only. They are not a capability
qualification.

The output should be read as a historical operating-window assessment, not a
validated design space. A spec can be challenged by the data, but changing it
requires domain review and usually confirmatory evidence.

## PCA

Principal Component Analysis, or PCA, is an exploratory method. It compresses a
wide set of process variables into a smaller number of components that explain
the largest sources of variation.

In this app, PCA helps answer:

- Do batches cluster by similar process behavior?
- Are some batches unusual?
- Which variables explain the main structure in the process data?

Outputs:

- Explained variance table.
- Score plot, usually PC1 vs PC2.
- Loading plot.
- Process-variable loading strength.

Important caveat:

PCA is unsupervised. It does not know which quality outcome matters. A variable
can be important in PCA because it varies a lot, not because it affects QC.

## PLS Regression

Partial Least Squares, or PLS, is a supervised method. It looks for directions
in process-variable space that are useful for predicting a quality outcome.

PLS is useful when:

- Process variables are correlated with each other.
- There are many variables relative to the number of batches.
- The user wants a regulator-familiar linear model.

The app reports variable importance using aggregated feature-level PLS signals.
For encoded categorical variables, the app maps one-hot features back to the
original process variable.

Outputs:

- Cross-validated Q2-style score.
- Training R2 for fit diagnostics.
- Time-ordered validation when an order column exists.
- Variable importance table.
- Feature-level coefficients.

Important caveat:

PLS is mainly linear. It can miss U-shaped effects and strong interactions unless
those are explicitly represented.

## Random Forest

Random Forest is a tree-based supervised model. It can capture non-linear
patterns and interactions more naturally than PLS.

It is useful for:

- Mixed numeric and categorical data.
- Non-linear process relationships.
- Interaction screening.
- Robust tabular baseline modeling.

Outputs:

- Training R2.
- Out-of-bag R2.
- Time-ordered validation when possible.
- Impurity-based feature importance.
- Permutation importance.
- Aggregated process-variable importance.

Important caveat:

Tree models can overfit, especially with small data. Training R2 should not be
treated as proof of predictive value. OOB and time-ordered validation are more
important.

## Ranked Drivers

The app combines evidence from PCA, PLS, and Random Forest into a unified ranked
driver table.

The basic idea:

- PCA contributes exploratory structure evidence.
- PLS contributes linear supervised evidence.
- Random Forest contributes non-linear supervised evidence.
- Variables that appear important across multiple methods get higher
  confidence.

Confidence levels are not regulatory validation. They are a practical guide for
which findings deserve attention first.

The UI now includes a confidence breakdown for each top driver. It explains:

- Which methods support the driver.
- Whether PLS and Random Forest validation scores are usable.
- Whether sample size is small, moderate, or stronger.
- Whether missingness or audit warnings reduce confidence.

This breakdown is meant to make the ranking inspectable. It should not be read
as a p-value, causal estimate, or change-control approval.

## Numeric Interaction Screen

The interpretation summary includes a simple high-high interaction screen for
top numeric drivers. It compares the outcome mean when two variables are both in
their upper quartile against the rest of the data.

This is useful for finding patterns such as:

```text
high temperature + long duration -> high HCP
```

Important caveat:

These quartile cutoffs are exploratory. They are not optimized setpoints and
should not be presented as operating recommendations without confirmatory work.

The app also summarizes suggested historical response bands for numeric drivers
when the quartile means show a best observed region. This is meant to catch
patterns such as:

```text
NaOH low          -> lower yield
NaOH near target  -> higher yield
NaOH high         -> lower yield
```

That pattern should be described as a sweet-spot hypothesis or historical
response band, not as a simple "higher is better" trend. It is still
observational evidence and should be confirmed before it becomes a setpoint,
design-space claim, or spec change.

The app reports two levels when enough data is available:

- Broad quartile band: more robust, but sometimes wide.
- Refined narrow bin: more exact, but more sensitive to random noise and uneven
  historical coverage.

The Top Drivers tab also includes a response-shape plot for top numeric drivers.
It shows:

- individual batch points,
- quantile-binned outcome means,
- the broad historical response band, when available,
- the refined best observed bin, when available.

This plot is meant to help a process scientist see whether the pattern looks
linear, flat, U-shaped, edge-sensitive, or noisy. It still describes historical
data only. It does not create an optimized setpoint.

## Categorical Level Effects

For categorical drivers, the app summarizes outcome means by level. This can
surface patterns such as:

- Supplier_B having lower purity.
- BR-3 having lower yield.
- A specific media lot behaving differently.

Important caveat:

Categorical effects are especially vulnerable to confounding. A supplier, lot,
operator, or reactor can be mixed up with date, campaign, scale, or recipe.

## Missing Data Handling

The modeling layer imputes missing data:

- Numeric columns: median imputation.
- Categorical columns: mode imputation.

The app does not silently drop rows for normal modeling. However, rows with a
missing selected outcome cannot be used for that outcome's supervised model.

Missingness is reported because it affects confidence.

## Validation

The app uses several validation signals:

- PLS cross-validation.
- Random Forest out-of-bag score.
- Time-ordered train/test split when an order column is available.

For batch manufacturing, time-ordered validation is often more realistic than
random splits because future batches may differ from earlier batches.

## Interpretation With Ollama

The LLM layer is not the statistical engine. It receives a structured summary of
the analysis and turns it into readable markdown.

The prompt asks the model to:

- Use only supplied results.
- Avoid causal claims.
- Treat audit warnings as important.
- Treat spec/window assessments as historical checks, not design-space proof.
- Label exploratory interaction screens correctly.
- Prefer validation metrics over training scores.

The interpretation should be reviewed by the user. The numerical analysis tables
remain the source of truth.

The app also shows Python-generated key findings before the LLM narrative.
Those rows are deterministic summaries of top drivers, categorical level
effects, response bands, interaction screens, specs, and audit cautions. If the
LLM and the deterministic findings disagree, trust the deterministic tables and
regenerate or manually edit the narrative.

## Optional And Planned Advanced Methods

CatBoost + SHAP is available as an optional, on-demand comparison layer when the
`catboost` Python package is installed. CatBoost handles categorical variables
directly, which is useful for lot, supplier, reactor, operator, and site labels.
The app uses CatBoost's native SHAP values to explain model behavior.

Important caveats:

- CatBoost + SHAP explains the CatBoost model, not the true causal process.
- It does not currently change the main ranked-driver score.
- It should be compared against PLS and Random Forest rather than used alone.
- Time-ordered validation is more important than training fit.

XGBoost + SHAP is a second optional comparison layer, available when the
`xgboost` Python package is installed. It is a gradient-boosted tree model and
acts as a third non-linear vote next to Random Forest and CatBoost. Because
XGBoost has no native categorical handling, the app one-hot encodes categoricals
with the same preprocessing as PLS and Random Forest, then collapses the encoded
SHAP contributions back to the original process variables. SHAP comes from
XGBoost's own TreeSHAP (`pred_contribs`), so no extra `shap` package is needed.
The same caveats as CatBoost apply: it explains model behavior on historical
data, not causation, and it does not currently change the main ranked-driver
score.

OPLS is still planned. It would add a chemometrics root-cause view that separates
outcome-related variation from structured variation unrelated to the outcome.
OPLS should be added with cross-validation and permutation testing before it is
treated as more than exploratory.

## Correlation Is Not Causation

The app can identify associations. It does not prove that changing a variable
will change the outcome.

To support stronger causal claims, a team needs one or more of:

- Designed experiments.
- Domain knowledge and mechanism.
- Replicate confirmation.
- Causal modeling with explicit assumptions.
- Prospective validation on future batches.

## Why These Methods First

PCA, PLS, and Random Forest give a useful first triad:

- PCA: What structure exists in the process data?
- PLS: Which variables linearly predict each outcome?
- Random Forest: Which non-linear or categorical patterns may matter?

Together, agreement across these methods is more useful than any single model
alone. Disagreement is also useful because it may point to non-linearity,
confounding, or data quality issues.
