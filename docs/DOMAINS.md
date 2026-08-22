# Two Domains, And The Words That Collide

This repository serves two different customers asking different questions about
different data. Three words mean different things in each, and the collisions
are already in the code. This document fixes the vocabulary so the confusion
stops spreading.

## The two domains

|  | **Production** | **Clinical / bioanalytical** |
|---|---|---|
| Who | Bioprocess and manufacturing teams | Bioanalytical laboratories and CROs |
| The unit of work | A manufactured batch | An analytical run |
| The question | Which process parameters associate with release outcomes? | Is a stable method drifting, and what does the study report say? |
| The data | Process parameters joined to release assay results | Runs, calibration curves, QC samples, study samples, deviations |
| Statistics | PCA, PLS, Random Forest - cross-sectional over batches | Theil-Sen, step change, variance ratio - longitudinal over runs |
| Governed by | GMP process understanding | ICH M10, FDA BMV, VICH GL49 for veterinary |
| Output | Driver rankings and a narrative | Trending alerts, and a study report for submission |

A customer in one domain has no use for the other. A bioanalytical CRO
manufactures nothing; a fermentation team has no incurred sample reanalysis.

## Collision 1: "QC"

The most damaging of the three, because both domains use the word constantly.

| In the production domain | In the clinical domain |
|---|---|
| The release assay results of a manufactured batch | Samples of known concentration run beside the study samples |
| A property of the **product** | A check on the **method and instrument** |
| The outcome being explained | A measurement whose deviation from nominal is the whole point |
| `purity_percent`, `hcp_ppm`, `yield_g_L` | `Low QC`, `Mid QC`, `High QC` at fixed nominal values |

"QC failed" means *this batch is out of specification* in one domain and *this
analytical run cannot be reported* in the other. Nothing about the two is
interchangeable.

## Collision 2: "validation"

Three meanings, all live in this project:

- **Synthetic-data validation.** Checking that the app recovers known planted
  relationships. This is what `docs/VALIDATION.md` contains, despite its name.
- **Bioanalytical method validation.** Proving an assay is fit for purpose under
  ICH M10 - selectivity, accuracy, precision, stability. A laboratory activity,
  not a software one.
- **Computerised system validation.** Proving software does what it is specified
  to do, under GAMP 5, with a URS, risk assessment, IQ/OQ/PQ and a traceability
  matrix. This is what a QA function means by the word.

A QA reviewer opening `docs/VALIDATION.md` expecting the third will find the
first. Say which one is meant, every time.

## Collision 3: "batch"

A manufacturing lot in the production domain; an analytical run or sample set in
the clinical domain. The clinical code uses `run` throughout for this reason and
should keep doing so.

## Where the current code sits

| Component | Domain |
|---|---|
| `app.py`, `analysis/`, `ui/`, `utils/` | Production |
| `qc_intel/` | Clinical - analytical QC trending |
| `docs/REPORT_AUTOMATION_PLAN.md`, `generate_bioanalytical_study.py` | Clinical - study reporting |
| `home.py`, `launcher.py`, `packaging/` | Shared shell |

## The boundary that is not negotiable

`qc_intel` holds **no sample-level rows**. That is privacy by architecture: data
about a person cannot leak into a schema with nowhere to put it. Its README says
so and the intake screens for personal data before the first insert.

The study report engine requires exactly what that schema excludes - study
samples, subject references, concentrations, incurred sample reanalysis. The two
therefore cannot share a database, and the report engine cannot be built on top
of `qc_intel`.

This is a second, independent reason for the separation argued in
`docs/PRODUCT_SPLIT_PLAN.md`.

## Veterinary work changes the criteria

Veterinary bioanalysis follows **VICH GL49**, not ICH M10, and the acceptance
criteria differ. Anything that hardcodes M10 limits is wrong for a laboratory
that does both.

`generate_bioanalytical_study.py` currently holds them as module constants:

```python
CALIBRATOR_TOLERANCE_PERCENT = 15.0
QC_RUN_PASS_FRACTION = 2 / 3
```

They belong in per-method configuration, the way `qc_intel/config/*.toml`
already handles acceptance limits. Cheap to change now, expensive later.

## Rules for future work

1. **Name the domain in the module docstring.** A reader should not have to
   infer it.
2. **Never let one domain's vocabulary into the other's code.** No `batch_id` in
   clinical modules, no `qc_level` in production modules.
3. **Say which validation you mean.** Prefix it: synthetic-data validation,
   method validation, computerised system validation.
4. **Do not share a database across the boundary.** The privacy argument and the
   statistical argument both forbid it.
5. **Acceptance criteria are configuration, never constants.** Human and
   veterinary work follow different guidelines, and both are in scope.
