"""Build the whole prototype from scratch: generate, ingest, analyse, report.

    python -m qc_intel.build_prototype

Stands in for the scheduled job a real deployment would run overnight. A real
scheduler would skip already-imported files rather than rebuilding the database.
"""

from __future__ import annotations

from pathlib import Path

from qc_intel import db
from qc_intel.alerts import render_all_alerts
from qc_intel.pipeline import build_from_scratch
from qc_intel.synth.generate import generate, write_source_files


PACKAGE_ROOT = Path(__file__).resolve().parent
SOURCE_DIR = PACKAGE_ROOT / "data" / "source"


def main() -> None:
    print("1. Generating synthetic QC history")
    dataset = generate()
    write_source_files(dataset, SOURCE_DIR)
    print(f"   runs={len(dataset.runs):,}  qc_results={len(dataset.qc_results):,}")

    print("2. Ingesting into the analytics database")
    connection, result = build_from_scratch(SOURCE_DIR, db.DEFAULT_DATABASE_PATH)
    imported = db.read_sql(connection, "SELECT COUNT(*) AS n FROM qc_result")["n"].iloc[0]
    print(f"   qc_result rows in database: {imported:,}")

    print("3. Deriving statistics and evaluating rules")
    print(f"   groups analysed: {len(result.run_series)}")
    print(f"   findings: {len(result.findings)}")

    findings_frame = result.findings_frame
    if not findings_frame.empty:
        counts = findings_frame["severity"].value_counts().to_dict()
        print(f"   by severity: {counts}")

    print("\n4. Alert digest")
    print(render_all_alerts(result.findings[:3]))

    print(f"\nDatabase: {db.DEFAULT_DATABASE_PATH}")
    print("Dashboard: .\\.venv\\Scripts\\python.exe -m streamlit run qc_intel/app.py")


if __name__ == "__main__":
    main()
