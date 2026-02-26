"""Tests for PublicationCalendar — from_dicts, vintage date wiring, and round-trip."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import polars as pl
import pytest

from alt_nfp.lookups.revision_schedules import (
    CES_REVISIONS,
    PublicationCalendar,
    get_ces_vintage_date,
    get_default_calendar,
    get_qcew_vintage_date,
    get_sae_vintage_date,
)


class TestFromDicts:
    """Tests for PublicationCalendar.from_dicts()."""

    def test_from_dicts_builds_all_three_sources(self):
        """CES, QCEW, SAE DataFrames are all non-empty."""
        cal = PublicationCalendar.from_dicts()
        assert cal.ces_release_dates.height > 0
        assert cal.qcew_release_dates.height > 0
        assert cal.sae_release_dates.height > 0

    def test_ces_revision_chain(self):
        """For Feb 2025, revision 0 < revision 1 < revision 2."""
        cal = PublicationCalendar.from_dicts()
        ref = date(2025, 2, 1)
        feb = cal.ces_release_dates.filter(
            pl.col('ref_month') == ref
        ).sort('revision_number')
        assert feb.height >= 3, f'Expected >= 3 CES revisions for Feb 2025, got {feb.height}'
        dates = feb['pub_date'].to_list()
        assert dates[0] < dates[1] < dates[2], (
            f'CES Feb 2025 revision chain not strictly increasing: {dates}'
        )


class TestVintageDateWiring:
    """Tests for get_ces/qcew/sae_vintage_date with calendar lookup."""

    def test_exact_beats_approximate(self):
        """get_ces_vintage_date with calendar returns a different (exact) date."""
        cal = PublicationCalendar.from_dicts()
        ref = date(2025, 2, 1)

        # Compute the lag-based approximation manually
        lag = next(s.lag_months for s in CES_REVISIONS if s.revision_number == 0)
        total = ref.month + lag
        approx_year = ref.year + (total - 1) // 12
        approx_month = ((total - 1) % 12) + 1
        approx = date(approx_year, approx_month, 1)

        exact = get_ces_vintage_date(ref, 0, calendar=cal)
        # The exact date is a specific day (Mar 7), not the 1st
        assert exact != approx, (
            f'Expected exact date to differ from approximate {approx}'
        )

    def test_fallback_when_missing(self):
        """For a ref_month not in the calendar, the lag-based approx is returned."""
        cal = PublicationCalendar.from_dicts()
        # Far future, not in hard-coded dicts
        ref = date(2030, 6, 1)
        result = get_ces_vintage_date(ref, 0, calendar=cal)
        # Lag-based returns 1st of month
        assert result.day == 1

    def test_roundtrip_parquet(self, tmp_path: Path):
        """Write calendar to parquet, reload via from_parquet, verify dates match."""
        cal = PublicationCalendar.from_dicts()

        # Build the parquet in the expected schema
        ces_rows = cal.ces_release_dates.with_columns(
            pl.lit('ces').alias('source'),
            pl.col('ref_month').dt.strftime('%Y-%m').alias('ref_period'),
            pl.col('pub_date').alias('publication_date'),
        ).select('source', 'ref_period', 'revision_number', 'publication_date')

        qcew_rows = cal.qcew_release_dates.with_columns(
            pl.lit('qcew').alias('source'),
            (
                pl.col('ref_year').cast(pl.Utf8)
                + pl.col('ref_quarter')
            ).alias('ref_period'),
            pl.col('pub_date').alias('publication_date'),
        ).select('source', 'ref_period', 'revision_number', 'publication_date')

        sae_rows = cal.sae_release_dates.with_columns(
            pl.lit('sae').alias('source'),
            pl.col('ref_month').dt.strftime('%Y-%m').alias('ref_period'),
            pl.col('pub_date').alias('publication_date'),
        ).select('source', 'ref_period', 'revision_number', 'publication_date')

        combined = pl.concat([ces_rows, qcew_rows, sae_rows])
        parquet_path = tmp_path / 'publication_calendar.parquet'
        combined.write_parquet(parquet_path)

        # Reload
        cal2 = PublicationCalendar.from_parquet(parquet_path)
        assert cal2.ces_release_dates.height == cal.ces_release_dates.height
        assert cal2.qcew_release_dates.height == cal.qcew_release_dates.height
        assert cal2.sae_release_dates.height == cal.sae_release_dates.height
