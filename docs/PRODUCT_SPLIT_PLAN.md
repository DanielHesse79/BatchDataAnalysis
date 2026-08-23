# Splitting Into Two Products

Decisions taken: **two repositories**, the clinical product is called
**lab_insight**, and the production app **keeps its LLM narrative for now**.

Nothing is executed yet.

Read `docs/DOMAINS.md` first: it establishes that the two workspaces serve
different customers, different data and different regulations, and that three
words mean different things on either side of the line.

## The decision

Stop shipping one program with two workspaces. Ship two programs.

Four reasons, in order of weight:

1. **Different regulatory posture.** The clinical product is heading toward
   computerised system validation, because its output goes into a submitted
   report. The production product is decision support and should stay out of
   that scope. Once one part of a codebase is qualified, every commit touches a
   qualified system and change control applies to all of it.

2. **Incompatible data models.** `qc_intel` deliberately holds no sample-level
   rows. The report engine requires them. They cannot share a database, and the
   report engine cannot be built on the trending schema.

3. **No customer wants both.** A bioanalytical CRO manufactures nothing. Shipping
   Lablytica a bundle containing a manufacturing workspace is confusing rather
   than generous, and it makes the product harder to explain in the one sentence
   that matters.

4. **Different release cadence.** A qualified product changes slowly and on
   purpose. A decision-support tool should be free to change weekly.

## What the clinical product is, and is not

It is tempting to describe it as an alternative to a LIMS. **It is not, and
positioning it that way loses.**

A LIMS does sample registration, chain of custody, scheduling, instrument
integration, audit trail and electronic signatures. Watson LIMS is used by 18 of
the 20 largest pharmaceutical companies and most large CROs. Competing with that
means matching a decade of qualified functionality and inheriting its entire
validation burden.

What this product does is answer the questions the LIMS does not:

- **Retrospective, longitudinal analysis.** Is this method drifting across runs,
  instruments and months? A LIMS decides run acceptance one run at a time and
  holds no opinion about the year.
- **Turning finished analysis into a report.** The 160 hours David described
  happen *after* the LIMS has done its job.

It sits **beside** the LIMS and reads what the LIMS already produced. That is a
winnable position, a far smaller validation scope, and an easier sale: it does
not ask anyone to replace a qualified system.

## Measured coupling

Direction first:

```
qc_intel -> analysis     2 import statements
analysis -> qc_intel     none
home.py  -> both         the workspace chooser, which the split removes
```

The two statements are:

- `qc_intel/ingest/interactive.py` needs `parse_numeric_series`
- `qc_intel/intake_panel.py` needs `IntakeLoadOptions`, `inspect_tabular_file`,
  `load_intake_dataframe`

Two imports looks trivial, and it is not. Walking the call graph from those four
names gives the real number:

| Source | Definitions |
|---|---|
| `analysis/data_prep.py` | 14 |
| `analysis/intake.py` | 9 |
| `analysis/normalization.py` | 5 |
| **Total** | **28 definitions, about 440 lines** |

`read_csv_flexibly` alone pulls in encoding detection, delimiter sniffing and
ragged-row recovery.

**The important part is what the closure does *not* contain.** No `MergeResult`,
no `merge_process_and_qc_data`, no batch identifiers - none of the production
domain's logic. What both products share is file handling, which lives inside
`analysis/` only because that is where it was written first.

It is not one component, though. Checking who else uses those 28
definitions shows two, tangled together:

| | Used by | Nature |
|---|---|---|
| **File reading** - encoding, delimiter, ragged rows, sheets, header row | `ui/state.py`, the intake panels | genuinely intake |
| **Number parsing** - `parse_numeric_series` and friends | `analysis/aggregation.py`, `analysis/readiness.py` | used all over the production domain |
| `DataPrepError` | five modules | a cross-cutting exception |

A package called `fileintake` holding number parsing would be misnamed. One
package with two modules is the right shape:

```
tabular/
    errors.py     the shared exception
    files.py      reading a messy export: encoding, delimiter, sheets, headers
    numbers.py    numbers as humans write them: decimal commas, units, "<0.5"
```

## Target structure

Two repository roots.

```
Batch process data analysis/      production, this repository
    app.py, analysis/, ui/, utils/, tests/, docs/

lab_insight/                      clinical, new repository
    app.py
    trending/                     from qc_intel
    report_engine/                unbuilt; see REPORT_AUTOMATION_PLAN.md
    tabular/                      vendored from this repository, pinned
    tests/, docs/
```

This repository keeps its current layout and drops the clinical half. No
rearranging on this side.

## The one real decision: how `tabular/` is held

The clinical product needs the file-reading and number-parsing code. With two
repositories the choice is a shared dependency - a package published and
installed by both - or a vendored copy.

| | Shared dependency | Vendored copy |
|---|---|---|
| A fix reaches both | on the next release | when someone copies it |
| Change control once the clinical side is qualified | every fix touches the qualified product | each product moves on its own |
| Risk of silent divergence | none | real |
| Infrastructure needed | a package index, versioning, releases | none |

**Recommendation: vendor it, with the source commit recorded at the top of each
module.**

Sharing is correct engineering right until one side is qualified. After that, a
fix needed only by the production product still triggers impact assessment on
the validated one. Vendoring gives the clinical product a frozen copy it can
validate against, and it needs no packaging infrastructure that does not exist
yet.

The cost is real and should be stated: a bug fixed in one copy is not fixed in
the other. Record the origin commit so the divergence is at least visible.

## Migration, in order

Each step leaves the repository working and the tests passing.

Choosing two repositories changes this, and simplifies it. An earlier draft
began by extracting `tabular/` here so both products could share it. With the
clinical product leaving, **that extraction is no longer needed in this
repository**: the coupling disappears when `qc_intel` goes, and the production
app can keep its code where it is until there is a reason of its own to move it.

The work is therefore building the new repository, not rearranging this one.

1. **Create the `lab_insight` repository** with `trending/` (from `qc_intel`),
   its tests, its config and its synthetic data.
2. **Vendor `tabular/`** into it - files, numbers, errors - copied from
   `analysis/`, with the source commit recorded at the top of each module. The
   clinical product now depends on nothing from here.
3. **Move the report work across**: `docs/REPORT_AUTOMATION_PLAN.md`,
   `generate_bioanalytical_study.py`, `data/bioanalytical_study/`. They were
   always clinical.
4. **Give it its own entry point and packaging.** The clinical bundle needs no
   PCA, PLS or Random Forest, so it should be far smaller than 683 MB.
5. **In this repository:** delete `qc_intel/`, retire `home.py`, drop the
   workspace chooser from `launcher.py` and the spec, and remove the clinical
   sections from the docs. `DOMAINS.md` stays in both, because both need to know
   where the line is.

Only after all five does this repository become single-purpose again. Until
then, keep `qc_intel` here and working - deleting it before the new repository
runs is how work gets lost.

Step 2 is the only one with real judgement in it. The rest is moving files.

## What this costs

- A day of mechanical work, most of it in step 3.
- Two bundles to build and two test runs instead of one.
- The shared launcher and packaging machinery is duplicated, not reused.

## What it buys

- The clinical product can enter validation without freezing the other.
- Each product can be explained in one sentence to one buyer.
- The clinical bundle drops the scientific stack it never uses, which matters
  when the current bundle is 683 MB.
- The privacy boundary becomes structural rather than a convention inside one
  codebase.

## Open questions

1. **Where does the `lab_insight` repository live?** A sibling directory beside
   this one is the obvious answer, but it is outside this project and worth
   saying out loud before anything is created.
2. **Does `lab_insight` start with history or clean?** Copying `qc_intel` loses
   its commit history; `git subtree split` keeps it and costs a little more.
   History is worth having on a product heading for validation, where "when did
   this change and why" is a question someone will ask.
