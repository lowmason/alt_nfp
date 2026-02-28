"""Backfill QCEW data (2003-2015) from BLS annual singlefile CSVs.

Downloads the annual singlefile ZIP for each year, extracts US national
total-nonfarm monthly employment, assigns synthetic vintage dates based
on the QCEW publication schedule (~6 months after quarter end), and
appends to the vintage store.

Usage:
    uv run python scripts/backfill_qcew.py
"""

from __future__ import annotations

import io
import sys
import zipfile
from datetime import date
from pathlib import Path

import httpx
import polars as pl

src = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(src))

from alt_nfp.config import DATA_DIR

STORE_PATH = DATA_DIR / "raw" / "vintages" / "vintage_store"

BLS_URL = "https://data.bls.gov/cew/data/files/{year}/csv/{year}_qtrly_singlefile.zip"

QCEW_PUB_LAG_MONTHS = 6


def _synthetic_vintage_date(ref: date) -> date:
    """Approximate QCEW publication date: ~6 months after quarter end."""
    qtr_end_month = ((ref.month - 1) // 3 + 1) * 3
    pub_month = qtr_end_month + QCEW_PUB_LAG_MONTHS
    pub_year = ref.year + (pub_month - 1) // 12
    pub_month = ((pub_month - 1) % 12) + 1
    return date(pub_year, pub_month, 23)


def download_year(year: int, client: httpx.Client) -> pl.DataFrame:
    """Download and parse one year of QCEW quarterly data."""
    url = BLS_URL.format(year=year)
    print(f"  Downloading {year}...", end=" ", flush=True)
    r = client.get(url)
    r.raise_for_status()
    print(f"({len(r.content) / 1e6:.0f} MB)", end=" ", flush=True)

    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        csv_name = [n for n in zf.namelist() if n.endswith(".csv")][0]
        with zf.open(csv_name) as f:
            # Stream through to find US national rows only
            import csv as csv_mod
            import io as io_mod

            text = io_mod.TextIOWrapper(f, encoding="utf-8")
            reader = csv_mod.DictReader(text)
            rows = []
            for row in reader:
                if row.get("area_fips") == "US000" and row.get("own_code") == "0":
                    rows.append(row)

    if not rows:
        print("no US national rows")
        return pl.DataFrame()

    df = pl.DataFrame(rows)
    # agglvl_code '70' = US total all industries, all ownerships
    df = df.filter(pl.col("agglvl_code") == "70")

    if df.is_empty():
        # Try industry_code '10' as fallback
        df = pl.DataFrame(rows).filter(pl.col("industry_code") == "10")

    if df.is_empty():
        print("no total-nonfarm rows")
        return pl.DataFrame()

    months = []
    for row in df.iter_rows(named=True):
        qtr = int(row["qtr"])
        for m_offset, col in enumerate(["month1_emplvl", "month2_emplvl", "month3_emplvl"]):
            month_num = (qtr - 1) * 3 + m_offset + 1
            ref = date(year, month_num, 12)
            try:
                emp = float(row[col])
            except (ValueError, TypeError):
                continue
            months.append({
                "geographic_type": "national",
                "geographic_code": "00",
                "industry_type": "national",
                "industry_code": "00",
                "ref_date": ref,
                "vintage_date": _synthetic_vintage_date(ref),
                "revision": 0,
                "benchmark_revision": 0,
                "employment": emp / 1000.0,
                "source": "qcew",
                "seasonally_adjusted": False,
            })

    print(f"{len(months)} monthly obs")
    return pl.DataFrame(months)


def main() -> None:
    start_year = 2003
    end_year = 2015

    client = httpx.Client(
        follow_redirects=True,
        timeout=60.0,
        headers={"User-Agent": "alt-nfp-research/1.0"},
    )

    try:
        frames = []
        for year in range(start_year, end_year + 1):
            df = download_year(year, client)
            if not df.is_empty():
                frames.append(df)

        if not frames:
            print("No data downloaded.")
            return

        combined = pl.concat(frames)
        print(f"\nTotal: {len(combined)} rows ({start_year}-{end_year})")

        combined = combined.with_columns(
            pl.col("revision").cast(pl.UInt8),
            pl.col("benchmark_revision").cast(pl.UInt8),
        )

        out_path = STORE_PATH / "source=qcew" / "seasonally_adjusted=false"
        out_path.mkdir(parents=True, exist_ok=True)
        out_file = out_path / f"backfill_{start_year}_{end_year}.parquet"
        combined.drop("source", "seasonally_adjusted").write_parquet(out_file)
        print(f"Wrote {out_file}")

    finally:
        client.close()


if __name__ == "__main__":
    main()
