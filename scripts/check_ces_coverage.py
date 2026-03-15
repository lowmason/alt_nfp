"""Verify CES vintage availability for each backtest month in 2024-01 to 2026-01.

For each target month, builds a censored panel (as_of_ref=target) and checks
that the CES SA frontier reaches at least target minus one month (i.e. the
first print for the month before the target is available).
"""

from __future__ import annotations

from datetime import date

import polars as pl

from nfp_models.config import STORE_DIR
from nfp_ingest.vintage_store import read_vintage_store, transform_to_panel


def check_ces_coverage() -> None:
    targets = [
        date(y, m, 12)
        for y in (2024, 2025, 2026)
        for m in range(1, 13)
        if date(2024, 1, 12) <= date(y, m, 12) <= date(2026, 1, 12)
    ]

    print(f"{'Target':>12}  {'CES SA latest':>14}  {'CES SA obs':>10}  {'Gap':>4}  Status")
    print("-" * 65)

    failures = []
    for target in targets:
        lf = read_vintage_store(STORE_DIR)
        panel = transform_to_panel(lf, geographic_scope="national", as_of_ref=target)

        ces_sa = panel.filter(pl.col("source") == "ces_sa").sort("period")
        n_obs = len(ces_sa)
        latest = ces_sa["period"].max() if n_obs > 0 else None

        if latest is not None:
            gap = (target.year - latest.year) * 12 + (target.month - latest.month)
        else:
            gap = None

        ok = gap is not None and gap <= 1
        status = "OK" if ok else "MISSING"
        gap_str = str(gap) if gap is not None else "N/A"
        latest_str = str(latest) if latest is not None else "none"

        print(f"  {target}  {latest_str:>14}  {n_obs:>10}  {gap_str:>4}  {status}")
        if not ok:
            failures.append((target, latest, gap))

    print("-" * 65)
    if failures:
        print(f"\n{len(failures)} targets with missing CES coverage:")
        for t, lat, g in failures:
            print(f"  {t}: latest CES SA = {lat}, gap = {g}")
    else:
        print("\nAll targets have CES SA coverage within 1 month. Store looks good.")


if __name__ == "__main__":
    check_ces_coverage()
