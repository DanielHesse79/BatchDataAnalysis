# Specs And Operating Windows

Batch Insight Analyzer can optionally compare stated process windows and QC
specs against historical batch behavior.

This feature is meant to help answer:

- Are batches operating close to a process limit?
- Are QC outcomes close to failure?
- Does a process window look potentially too wide or too narrow?
- Has historical production actually explored the approved range?
- Are model-important variables also spec-controlled variables?

## Why This Matters

Manufacturing data is usually not just a set of free-floating variables. Many
columns have intended or approved ranges.

Examples:

```text
sodium_hydroxide_g target 100 g, allowed range 85-115 g
moisture_percent must be <= 20%
temperature_C target 36.9 C, allowed range 36.2-37.8 C
```

The useful question is not only "what drives quality?" It is also:

> Does the current operating window make sense in light of historical quality
> behavior?

## Spec File Format

Upload a CSV, XLSX, or XLS file with these columns:

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

Only `variable` is strictly required, but useful assessments need at least one
of `target`, `lower_limit`, or `upper_limit`.

Example:

```text
variable,role,target,lower_limit,upper_limit,unit,criticality,notes
sodium_hydroxide_g,process,100,85,115,g,CPP,Addition amount in neutralization step
moisture_percent,qc,,,20,%,CQA,Release moisture limit
```

The repository includes a mock example:

```text
data/mock_spec_process.csv
data/mock_spec_qc.csv
data/mock_spec_specs.csv
```

This mock dataset includes a sodium hydroxide process window and a moisture QC
limit so the Specs & Margins workflow can be tested without creating files by
hand.

Roles:

- `process`: an operating window or target for a process variable.
- `qc`: a release/specification limit for a quality outcome.

Limit types are inferred:

- Two-sided: lower and upper limits.
- Lower-only: lower limit only.
- Upper-only: upper limit only.
- Target-only: target but no limits.
- Reference-only: no numeric target or limit.

## What The App Calculates

For each matched spec variable, the app calculates:

- Usable numeric batch count.
- Missing count.
- Mean, standard deviation, minimum, maximum.
- Percent inside the supplied limits.
- Percent outside the supplied limits.
- Percent close to a limit.
- Historical used range divided by allowed range.
- Median margin to nearest limit.
- Worst margin to nearest limit.
- Cp and Cpk for two-sided limits when enough data exists.
- Out-of-spec batch details.
- Outcome means by spec zone for process variables.

Spec zones:

```text
below_spec
near_lower_edge
inside
near_upper_edge
above_spec
missing
```

## Challenge Labels

The app assigns conservative labels.

### Looks reasonable

Historical behavior is broadly consistent with the supplied limit or window.

This does not prove the spec is correct. It only means the current data does not
raise an obvious challenge.

### Possibly too wide

Used when a process variable is an important model driver and historical values
approach or exceed the supplied window.

Interpretation:

> The approved window may allow regions associated with worse quality.

This should trigger investigation, not automatic narrowing.

### Possibly too narrow

Used when a variable has low model-driver signal but many batches are close to
or outside the supplied window.

Interpretation:

> The window may create operational burden without obvious quality benefit in
> the current dataset.

This requires domain review and often prospective confirmation.

### Target may be off-center

Used when the historical mean sits noticeably away from the stated target.

Interpretation:

> The process may naturally run away from the target, or the target may not
> reflect current best operation.

### High failure risk

Used for QC specs when batches are outside or close to the supplied QC limit.

Interpretation:

> The process may have limited margin to a quality requirement.

### Insufficient data

Used when the app cannot make a meaningful challenge.

Common reasons:

- Too few usable numeric values.
- Historical production used only a small part of the allowed range.
- Driver analysis has not been run yet.

### Confounded - review manually

Used when the variable overlaps with lot, supplier, operator, equipment, room,
or campaign-style audit warnings.

Interpretation:

> The apparent spec issue may actually be mixed up with another operational
> factor.

## How This Should Be Interpreted

The correct phrase is:

```text
historical operating-window assessment
```

Do not call this a design space.

A design space requires stronger evidence, usually designed experiments,
mechanistic understanding, or confirmatory runs.

## Example Interpretation

Good:

```text
The sodium hydroxide window is 85-115 g around a 100 g target. Historical
batches used 96-104 g, so the dataset does not test the full allowed range.
Within the observed range, sodium hydroxide was not a strong model driver for
moisture or purity. This means the current data cannot justify widening or
narrowing the range, but it suggests the range may be worth reviewing if it is
operationally burdensome.
```

Too strong:

```text
The sodium hydroxide spec should be widened.
```

Too strong:

```text
The historical data proves the design space is 85-115 g.
```

## Relationship To Modeling

Specs are not a modeling method by themselves. They add context to model
outputs.

Useful combinations:

- A variable is a high-confidence model driver and values near the spec edge
  show worse QC: investigate whether the window is too wide.
- A numeric driver has a middle-band sweet spot inside the current window:
  investigate whether routine operation should target that historical response
  band more tightly, but do not call it a validated setpoint yet.
- A refined narrow bin may suggest where inside the broader band the historical
  response looked best, but it should be treated as less stable than the broad
  band until confirmed.
- A variable is not a model driver and many batches fail a narrow process
  window: investigate whether the window is too tight.
- A QC outcome is frequently close to its limit: prioritize process margin.
- Historical values cover only a small part of the allowed range: do not claim
  the full range is proven acceptable.

## Current Limitations

- The challenge rules are intentionally simple.
- No formal tolerance interval or Bayesian capability model is used.
- Cp/Cpk assumes stable, roughly normal behavior.
- Categorical specs are treated as reference rows for now.
- Operational cost of tight specs is not captured unless the user supplies it
  separately.
- The app does not yet generate formal change-control or validation documents.

## Recommended Next Improvements

- Add user-entered operational pain/cost for narrow process windows.
- Add deterministic "best/worst spec zone" summaries.
- Add tolerance intervals for QC specs.
- Add multivariate spec interactions, such as temperature plus duration.
- Add DoE/RSM support before making true optimization recommendations.
