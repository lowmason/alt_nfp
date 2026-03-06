"""Backfill missing CES intermediate prints from FRED ALFRED.

When BLS has not updated cesvinall.zip after an annual benchmark revision,
the triangular extraction misses intermediate prints for recent months.
This script reconstructs rev 0/1/2 from ALFRED (Archival FRED), which
stores every historical vintage of PAYEMS (SA) and PAYNSA (NSA).

Usage:
    source .env && export FRED_API_KEY
    python scripts/backfill_alfred.py

After running, verify with:
    python scripts/check_ces_coverage.py
"""

from __future__ import annotations

import os
import time
from collections import defaultdict
from datetime import date

import httpx
import polars as pl

from alt_nfp.ingest.vintage_store import (
    VINTAGE_STORE_PATH,
    append_to_vintage_store,
    compact_partition,
)

FRED_API = "https://api.stlouisfed.org/fred/series"

SERIES = {"PAYEMS": True, "PAYNSA": False}

MIN_REF = date(2024, 12, 1)

BENCHMARK_VINTAGE = date(2026, 2, 11)


def _api_key() -> str:
    k = os.environ.get("FRED_API_KEY")
    if not k:
        raise RuntimeError("Set FRED_API_KEY in environment")
    return k


def _fetch_vintage_dates(series_id: str, key: str) -> list[date]:
    r = httpx.get(
        f"{FRED_API}/vintagedates",
        params={
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "realtime_start": "2025-01-01",
            "realtime_end": str(BENCHMARK_VINTAGE),
        },
    )
    r.raise_for_status()
    return [date.fromisoformat(d) for d in r.json()["vintage_dates"]]


def _fetch_observations(
    series_id: str, vd: date, key: str
) -> list[tuple[date, float]]:
    """Return [(fred_ref_date, employment)] sorted by date descending."""
    r = httpx.get(
        f"{FRED_API}/observations",
        params={
            "series_id": series_id,
            "api_key": key,
            "file_type": "json",
            "realtime_start": str(vd),
            "realtime_end": str(vd),
            "observation_start": "2024-10-01",
            "sort_order": "desc",
            "limit": 20,
        },
    )
    r.raise_for_status()
    return [
        (date.fromisoformat(o["date"]), float(o["value"]))
        for o in r.json().get("observations", [])
        if o["value"] != "."
    ]


def backfill() -> None:
    key = _api_key()
    rows: list[dict] = []

    for series_id, is_sa in SERIES.items():
        print(f"\n{'=' * 60}")
        print(f"{series_id} ({'SA' if is_sa else 'NSA'})")
        print(f"{'=' * 60}")

        vdates = _fetch_vintage_dates(series_id, key)
        # Exclude the benchmark vintage itself
        vdates = [v for v in vdates if v < BENCHMARK_VINTAGE]
        print(f"Pre-benchmark ALFRED vintages: {len(vdates)}")
        for v in vdates:
            print(f"  {v}")

        # Track each ref_date's ordered appearances across vintages
        ref_history: dict[date, list[tuple[date, float]]] = defaultdict(list)

        for vd in sorted(vdates):
            obs = _fetch_observations(series_id, vd, key)
            for fred_ref, emp in obs:
                if fred_ref < MIN_REF:
                    continue
                history = ref_history[fred_ref]
                if not history or history[-1][0] != vd:
                    history.append((vd, emp))
            time.sleep(0.3)

        print(f"\nRef-date history (revision assignments):")
        for fred_ref in sorted(ref_history):
            store_ref = fred_ref.replace(day=12)
            history = ref_history[fred_ref]
            for rev, (vd, emp) in enumerate(history[:3]):
                label = f"  {store_ref} rev {rev}: vintage {vd}, emp {emp:>10,.0f}"
                print(label)
                rows.append(
                    {
                        "geographic_type": "national",
                        "geographic_code": "00",
                        "industry_type": "national",
                        "industry_code": "00",
                        "ref_date": store_ref,
                        "vintage_date": vd,
                        "revision": rev,
                        "benchmark_revision": 0,
                        "employment": emp,
                        "source": "ces",
                        "seasonally_adjusted": is_sa,
                    }
                )

    if not rows:
        print("\nNothing to backfill.")
        return

    df = pl.DataFrame(rows).with_columns(
        pl.col("revision").cast(pl.UInt8),
        pl.col("benchmark_revision").cast(pl.UInt8),
    )
    print(f"\n{'=' * 60}")
    print(f"Total backfill rows: {len(df)}")

    n = append_to_vintage_store(df, VINTAGE_STORE_PATH)
    print(f"Appended {n} new rows to vintage store")

    for sa in [True, False]:
        compact_partition(VINTAGE_STORE_PATH, "ces", sa)
    print("Compacted CES partitions")

    print("\nDone. Verify with: python scripts/check_ces_coverage.py")


if __name__ == "__main__":
    backfill()
