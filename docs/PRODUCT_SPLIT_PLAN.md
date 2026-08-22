# Splitting Into Two Products

A plan, not a commitment. Nothing here is executed yet.

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
domain's logic. What both products share is exactly one coherent thing: reading
a messy tabular file that a laboratory exported. It lives inside `analysis/`
only because that is where it was written first.

That is a component, not a pile of helpers, and it should be named as one.

## Target structure

```
batch_insight/              production: process parameters -> release outcomes
    app.py
    analysis/
    ui/
    utils/
    tests/
    docs/

lab_insight/                clinical: QC trending and study reporting
    app.py
    trending/               formerly qc_intel
    report_engine/          currently unbuilt, see REPORT_AUTOMATION_PLAN.md
    fileintake/             vendored, pinned; see the shared-component section
    tests/
    docs/

packaging/                  builds either product
```

`lab_insight` is a placeholder name. It needs one that says *retrospective
analysis and reporting for a bioanalytical laboratory* without claiming to be a
LIMS. That is a naming decision, not a technical one.

## The one real decision: how the shared component is held

Extract the 28 definitions into a `fileintake/` package of their own, belonging
to neither product. That much is clear from the measurement: it is one component
with one job.

What is not obvious is how each product should then hold it.

| | Shared dependency | Vendored copy each |
|---|---|---|
| A fix reaches both | immediately | when someone copies it |
| Change control once one side is qualified | every fix touches the qualified product | each product moves on its own |
| Risk of silent divergence | none | real |

**Recommendation: one package, vendored into each product at a pinned version.**

Sharing a live dependency is correct engineering right until one side is
qualified. After that, a fix needed only by the unvalidated product still
triggers impact assessment and regression testing on the validated one. Pinning
gives the clinical product a frozen, documented version it can validate against,
while the production product stays free to move.

Write the version and the reason at the top of each vendored copy, so nobody
helpfully refactors them back together.

## Migration, in order

Each step leaves the repository working and the tests passing.

1. **Extract `fileintake/`.** Move the 28 definitions out of `analysis/` into
   a package of their own, and have the production app import from there. One
   move, no duplication yet, and the production tests prove it still works. This
   is the step that turns an accidental dependency into a named component.
2. **Split the test suites.** `tests/` stays with the production app,
   `qc_intel/tests/` is already separate. Add a runner for each.
3. **Move the production app** into `batch_insight/`. Mechanical: imports,
   `packaging/batch_insight.spec`, `launcher.py`, CI.
4. **Move and rename `qc_intel`** into `lab_insight/trending/`.
5. **Retire `home.py`.** Each product gets its own entry point. The chooser
   exists only because the two shared a shell.
6. **Split packaging.** Two spec files, two bundles. The clinical bundle is much
   smaller: it needs no PCA, PLS or Random Forest.
7. **Split the documentation.** `docs/` divides along the same line; `DOMAINS.md`
   stays at the root as the map.

Steps 1 and 2 are worth doing whether or not the rest happens: they turn an
accidental dependency into a named component and let each product's tests run on
their own.

Step 1 is larger than the two import statements suggest - 440 lines across three
modules - so it is its own piece of work, not a warm-up.

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

1. **What is the clinical product called?** It needs to say retrospective
   analysis and reporting without claiming to replace a LIMS.
2. **One repository with two folders, or two repositories?** Two folders first;
   the split to two repositories becomes obvious once the clinical product has a
   validation package, because that package versions with its code.
3. **Does the production app keep the LLM narrative?** It is the part hardest to
   validate, and only one of the two products is heading that way.
