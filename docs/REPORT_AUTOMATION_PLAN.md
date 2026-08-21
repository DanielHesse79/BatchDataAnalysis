# Bioanalytical Report Automation — Plan

A sketch, not a commitment. Nothing here is built.

## Where this came from

A conversation with David at Lablytica, a specialty bioanalytical CRO. His
interest is not the analysis — that is finished — but the report that follows
it, which can take **up to 160 hours per study with all the analytical data
already in hand**.

This document works out what could be automated, what the ceiling on the saving
is, and what it would cost to reach it.

## First: the operative guideline is not EudraLex Volume 10

Volume 10 governs *clinical trials* in the EU — application, safety reporting,
IMP quality, GCP inspections. It contains **no requirements for the content or
structure of a study report**. The closest is Annex VII, which touches the
bioanalytical part of bioequivalence trials from an inspection perspective, not
as a report specification.

What actually governs the document:

| Report | Operative guideline |
|---|---|
| Bioanalytical method validation report | **ICH M10**, documentation section |
| Study sample analysis report | **ICH M10**, documentation section |
| EU bioequivalence bioanalysis | ICH M10 plus the EMA BMV guideline |
| Clinical study report | **ICH E3** |
| Non-clinical GLP study report | OECD GLP principles, reporting section |

This is good news. ICH M10 enumerates what the report must contain, with tables.
**A document whose mandatory content is listed in a guideline is exactly what can
be templated.**

## Where the 160 hours actually go

Working hypothesis, **to be confirmed with David before anything is built**:
"writing the report" is perhaps 20% of the time. The rest is:

| Activity | Nature |
|---|---|
| Table production | Export to Excel, reformat into Word by hand |
| Transcribing numbers into narrative | Derive, type, then re-check |
| QC review | A second person checks every number against source |
| Review cycles | Each sponsor comment re-triggers partial re-verification |

If that holds, the bottleneck is **transcription and verification, not
composition**. That distinction decides what gets built. Ask for the 160 hours
broken down by activity; if QC review is 50 of them, traceability matters more
than text generation.

## Core idea: a table engine, not twenty tables

Nearly every table ICH M10 requires has the same shape: *a grouping, a set of
measurements, a fixed set of derived statistics, an acceptance criterion.*
Calibration standards per run, QC per level, ISR pairs — the same skeleton with a
different grouping key.

So not twenty table builders, but:

- **a declarative table specification** — source, grouping, statistics,
  criterion, formatting, M10 reference
- **a computation engine** that executes a specification
- **a renderer** that turns the result into DOCX
- **a catalogue of specifications**, one per required table

Shape of a specification:

```
table: qc_summary_across_runs
  m10_reference: "QC accuracy and precision, study sample analysis"
  source:        qc_results where run.acceptance = accepted
  group_by:      analyte x qc_level
  statistics:    n_runs, n_obs, mean, sd, cv_percent, bias_percent, min, max
  criterion:     abs(bias) <= 15 and cv <= 15   (20 at LLOQ)
  columns:       [level, nominal, n, mean, sd, cv_percent, bias_percent]
```

Adding a table becomes writing a declaration rather than writing code.

**This form is also what makes qualification affordable.** The engine is
validated once; the specifications are reviewed as data. Twenty hand-written
functions would each need validating, and again on every change.

## Table catalogue (study sample analysis)

| Table | Grouping | Core content |
|---|---|---|
| Run list | per run | date, instrument, analyst, status, failure reason |
| Calibration curves | per run | model, weighting, slope, intercept, r squared |
| Back-calculated standards | run x level | measured, %bias, accepted or rejected |
| QC per run | run x level | n, mean, %CV, %bias, pass or fail |
| QC across the study | analyte x level | n_runs, n_obs, mean, SD, %CV, %bias |
| Repeat analysis | per sample | reason, original, repeat, reported value |
| ISR | per pair | original, repeat, %difference, share within criterion |
| In-study stability | per condition | storage duration covered vs actual |
| Deviations | per deviation | type, runs affected, assessed impact |
| Samples outside the curve | per sample | below LLOQ or above ULOQ, action taken |

A method validation report needs a different catalogue — selectivity, matrix
effect, carryover, dilution integrity, the stability matrix, within-run and
between-run accuracy and precision. Same engine, different declarations.

## Traceability per cell

This carries the time saving, more than the tables themselves.

Every row entering the engine already carries a source identifier — the pattern
`qc_intel` uses with `ingest_id` and `source_row`. Aggregation retains the set of
contributing rows, so every **cell** can answer which source records produced it.

The traceability appendix then reads: table, cell, value, contributing rows as a
range, source file, checksum. Not 180 rows enumerated, but
`QC-LOW R01-R40, ingest 12, sha256 d263d5...`.

That is what moves QC review from "check every number" to "check the binding,
then sample". **Without this appendix there is no saving in the review step, only
in the writing step** — and the writing step is the smaller half.

## Three traps that decide whether it holds up

**Rounding.** Reports collect review findings because the text says 8.2% and the
table says 8.15%. Round in one place, once, and have both the table and the
narrative consume the rounded value. Never two rounding paths.

**What "n" means.** Number of runs, number of observations, or number accepted?
Ambiguity here produces findings. Every specification must declare it.

**Rejected runs.** They are excluded from accuracy and precision summaries but
**must** still appear in the run list with their reason. A generator that
silently drops them produces a report that does not reconcile — and the sponsor
finds it, not the CRO.

## Module sketch

```
report_engine/
    model.py          canonical data model: runs, standards, QCs, samples, deviations
    ingest/           adapters: Watson, Analyst/MultiQuant, generic tabular
    specs/
        catalogue.py  the table specifications
        m10.py        M10 references and acceptance criteria
    compute.py        specification -> computed table plus contributing source ids
    rounding.py       one rounding policy, applied once
    render/
        docx.py       computed table -> Word table in the customer's template
        appendix.py   the traceability appendix
    verify.py         self-check: does every number in the text exist in a table?
```

DOCX output should fill **the CRO's own template** with their named styles, not a
format we invent. QA recognises the document and the formatting argument never
happens. `python-docx` does this locally, with no AI and no network.

## Where it should live

This is a **third subsystem**, sibling to `analysis/` and `qc_intel/`. It reuses
the provenance pattern from `qc_intel` and the facts-before-narrative discipline
from `analysis/`.

But the moment one part of the codebase enters GxP qualification, change control
becomes a problem for the other two: every commit then touches a qualified
system. **That argues for a separate repository and a separate product**, with
the shared patterns copied rather than imported. It feels wrong to a developer
and is probably right here.

Decide this before writing the first module, not after.

## What it deliberately does not do

- **It does not recompute regressions or concentrations.** Slope, intercept,
  r squared and back-calculated values come from the validated CDS and are
  tabulated, never recalculated. Same principle as `qc_intel`: never recompute a
  regulated number and risk disagreeing with the validated system.
- It does not decide run acceptance.
- It does not write the discussion or the conclusions.
- It does not produce chromatograms.

## The validation crux

**In a GxP environment the time saving comes from qualifying the tool, not from
automating the typing.**

| | Unqualified tool | Qualified tool |
|---|---|---|
| Generated numbers | 100% human verification still required | Review of inputs and outputs |
| Realistic saving | about 30%, the typing | 160 h to 40 h is credible |

Qualification costs a URS, a functional specification, IQ/OQ/PQ, test scripts, a
traceability matrix and SOPs — GAMP 5 category 5, bespoke software. It is a
one-time cost amortised across every future report, but it must be in the
business case from the start or the promised saving does not materialise.

Note also that both existing applications **declare themselves not validated and
not for regulatory reporting**. This work crosses that line deliberately. It is a
strategic decision, not a detail.

## Why the local-AI constraint helps here

The valuable part — deterministic generation — needs no AI at all. And a
deterministic template can be qualified; a stochastic generator cannot. The
constraint pushes toward the right architecture rather than away from it. That
sponsor data never leaves the building is a selling point, not a handicap.

If a local model is used later it should be confined to connective narrative over
facts already computed, behind the guard `analysis/report_validator.py` already
implements.

## Sequencing

1. Data model plus one adapter against their actual export.
2. Engine plus the five tables covering most of the time: QC across the study,
   calibration, run list, repeat analysis, ISR.
3. The traceability appendix.
4. DOCX into their template.
5. **Measure** against one real anonymised study — hours before and after.

Steps 1 to 3 carry the saving. Step 4 is what makes anyone actually use it.

## Open questions

1. **Which report type?** A method validation report and a study sample analysis
   report look nothing alike. This determines the catalogue.
2. **Where do the hours go?** The 160 broken down by table production, writing,
   QC review and review cycles.
3. **What does the data come out of?** Watson, Analyst, Excel? That determines
   the ingest adapter, and `qc_intel/ingest/` already has the pattern.
