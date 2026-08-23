# Archive

Nothing in this folder is part of Batch Insight Analyzer. It is the clinical
half of what this repository used to be, kept here rather than deleted while the
product it moved to proves itself.

**The live version is [LabInsight](https://github.com/DanielHesse79/LabInsight).**
It carries this code's history, has its own tests and packaging, and depends on
nothing here. Fix things there, not here.

| Here | There |
|---|---|
| `qc_intel/` | `trending/` |
| `home.py` | not needed; one product, one entry point |
| `generate_bioanalytical_study.py` | same name |
| `bioanalytical_study/` | `data/bioanalytical_study/` |
| `REPORT_AUTOMATION_PLAN.md` | `docs/REPORT_AUTOMATION_PLAN.md` |

Why the two were split, and the vocabulary that made the split necessary, is in
`docs/DOMAINS.md` and `docs/PRODUCT_SPLIT_PLAN.md`.

This folder is excluded from the test run and from the bundle. When the new
product has run against real work for a while, it can go - and even then git
history keeps it recoverable.
