"""Download QCEW data for a given year from the BLS CSV API and append to the vintage store.

Usage:
    uv run python scripts/add_qcew_to_vintage_store.py [YEAR]
    uv run python scripts/add_qcew_to_vintage_store.py 2016

The BLS QCEW CSV API has data from 2014 onward. Each row in the panel is
converted to vintage store format (ref_date, employment, vintage_date, etc.)
and appended; existing keys in the store are skipped.
"""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl

# Add src to path when run as script
if __name__ == "__main__":
    src = Path(__file__).resolve().parents[1] / "src"
    sys.path.insert(0, str(src))

from alt_nfp.ingest.qcew import ingest_qcew
from alt_nfp.ingest.vintage_store import VINTAGE_STORE_PATH, append_to_vintage_store


def panel_to_vintage_store_rows(panel: pl.DataFrame) -> pl.DataFrame:
    """Convert a QCEW panel (PANEL_SCHEMA) to vintage store rows."""
    return panel.select(
        pl.col("geographic_type"),
        pl.col("geographic_code"),
        pl.col("industry_level").alias("industry_type"),
        pl.col("industry_code"),
        pl.col("period").alias("ref_date"),
        pl.col("vintage_date"),
        pl.col("revision_number").cast(pl.UInt8).alias("revision"),
        pl.lit(0, pl.UInt8).alias("benchmark_revision"),
        pl.col("employment_level").alias("employment"),
        pl.lit("qcew", pl.Utf8).alias("source"),
        pl.lit(False).alias("seasonally_adjusted"),
    )


def main() -> None:
    year = int(sys.argv[1]) if len(sys.argv) > 1 else 2016
    print(f"Downloading QCEW data for {year}...")
    panel = ingest_qcew(start_year=year, end_year=year, include_states=False)
    if len(panel) == 0:
        print("No data fetched. Exiting.")
        sys.exit(1)
    print(f"  Fetched {len(panel):,} panel rows")
    vintage_rows = panel_to_vintage_store_rows(panel)
    print(f"  Appending to store at {VINTAGE_STORE_PATH}...")
    n = append_to_vintage_store(vintage_rows)
    print(f"  Appended {n:,} new rows to the vintage store.")


if __name__ == "__main__":
    main()
